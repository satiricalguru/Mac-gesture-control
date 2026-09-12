"""Camera, inference, preview, and CLI shell for Gesture Mac."""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import cv2
import mediapipe as mp
import numpy as np

from .calibration import CalibrationEngine, CalibrationOutput, CalibrationStep
from .controller import MacOSController, PreviewController
from .gesture_engine import GestureConfig, GestureEngine, Point

DEFAULT_PROFILE_PATH = Path.home() / ".config" / "gesture-mac" / "profile.json"
FALLBACK_PROFILE_PATH = (
    Path.home()
    / "Library"
    / "Application Support"
    / "MacGestureControl"
    / "profile.json"
)

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_SHA256 = "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1"
MODEL_PATH_ENV = "GESTURE_MAC_MODEL_PATH"
MODEL_DOWNLOAD_TIMEOUT_S = 30.0
MODEL_MAX_BYTES = 32 * 1024 * 1024
CONNECTIONS = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (0, 17),
)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _model_path() -> Path:
    override = os.environ.get(MODEL_PATH_ENV)
    if override:
        return Path(override).expanduser()

    # Reuse the checked-out model during development. Installed wheels do not
    # contain the 7.8 MB asset, so their writable fallback is the user cache.
    source_model = _project_root() / "models" / "hand_landmarker.task"
    if source_model.exists():
        return source_model
    return (
        Path.home()
        / "Library"
        / "Caches"
        / "MacGestureControl"
        / "hand_landmarker-float16-v1.task"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_model(path: Path | None = None) -> Path:
    path = path or _model_path()
    if path.exists() and _sha256(path) == MODEL_SHA256:
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    print("Downloading the pinned MediaPipe Hand Landmarker model (7.8 MB)…")
    temporary: Path | None = None
    try:
        if urlsplit(MODEL_URL).scheme != "https":
            raise RuntimeError("refusing to download the model over a non-HTTPS URL")
        request = urllib.request.Request(
            MODEL_URL,
            headers={"User-Agent": "MacGestureControl/0.1"},
        )
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f"{path.name}.download-",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            downloaded = 0
            # MODEL_URL is HTTPS-only and the payload has a pinned SHA-256.
            with urllib.request.urlopen(  # nosec B310
                request, timeout=MODEL_DOWNLOAD_TIMEOUT_S
            ) as response:
                while chunk := response.read(1024 * 1024):
                    downloaded += len(chunk)
                    if downloaded > MODEL_MAX_BYTES:
                        raise RuntimeError("model download exceeded the safety limit")
                    handle.write(chunk)

        actual_hash = _sha256(temporary)
        if actual_hash != MODEL_SHA256:
            raise RuntimeError(
                f"model checksum mismatch: expected {MODEL_SHA256}, got {actual_hash}"
            )
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path


def _select_mac_camera(requested_index: int | None) -> tuple[int, str]:
    if requested_index is not None:
        return requested_index, f"camera index {requested_index}"

    try:
        import objc
        from Foundation import NSBundle

        NSBundle.bundleWithPath_("/System/Library/Frameworks/AVFoundation.framework")
        objc.loadBundle(
            "AVFoundation",
            globals(),
            bundle_path="/System/Library/Frameworks/AVFoundation.framework",
        )

        av_device_cls = objc.lookUpClass("AVCaptureDevice")
        discovery_cls = objc.lookUpClass("AVCaptureDeviceDiscoverySession")

        discovery = discovery_cls.discoverySessionWithDeviceTypes_mediaType_position_(
            ["AVCaptureDeviceTypeBuiltInWideAngleCamera"], "vide", 0
        )
        builtin_devices = tuple(discovery.devices() if discovery else ())

        devices = av_device_cls.devicesWithMediaType_("vide")
        for idx, dev in enumerate(devices):
            name = str(dev.localizedName())
            dev_type = str(dev.deviceType())
            if "Continuity" in dev_type or "iPhone" in name:
                continue
            if dev in builtin_devices or "BuiltIn" in dev_type or "FaceTime" in name:
                return idx, f"{name} (Mac built-in, device {idx})"
    except (ImportError, AttributeError, OSError):
        pass

    return 0, "default camera (index 0)"


def _open_camera(requested_index: int | None = None) -> cv2.VideoCapture:
    index, desc = _select_mac_camera(requested_index)
    print(f"[Gesture Mac] Using camera: {desc}")
    capture = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
    if not capture.isOpened():
        capture.release()
        capture = cv2.VideoCapture(index)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
    capture.set(cv2.CAP_PROP_FPS, 30)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"could not open {desc}; check Privacy & Security > Camera")
    return capture


def _landmarks(result: Any) -> list[Point] | None:
    if not result.hand_landmarks:
        return None
    hand = result.hand_landmarks[0]
    return [Point(float(p.x), float(p.y), float(p.z)) for p in hand]


def _draw_hand(frame: Any, points: list[Point] | None) -> None:
    if not points:
        return
    height, width = frame.shape[:2]
    pixels = [(int(p.x * width), int(p.y * height)) for p in points]
    for start, end in CONNECTIONS:
        cv2.line(frame, pixels[start], pixels[end], (112, 224, 190), 2, cv2.LINE_AA)
    for index, pixel in enumerate(pixels):
        radius = 6 if index in (4, 8, 12, 16, 20) else 3
        cv2.circle(frame, pixel, radius, (248, 250, 252), -1, cv2.LINE_AA)


def _draw_overlay(
    frame: Any,
    output: Any,
    config: GestureConfig,
    fps: float,
    control: bool,
) -> None:
    height, width = frame.shape[:2]
    x1, y1 = int(config.active_left * width), int(config.active_top * height)
    x2, y2 = int(config.active_right * width), int(config.active_bottom * height)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (112, 224, 190), 2)

    live = output.enabled
    mode = "CONTROL" if control else "PREVIEW"
    state = "ON" if live else "PAUSED"
    color = (88, 220, 120) if live else (90, 170, 255)
    cv2.rectangle(frame, (0, 0), (width, 106), (24, 27, 32), -1)
    cv2.putText(
        frame,
        f"{mode}  |  {state}",
        (22, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        color,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"Gesture: {output.stable_gesture.value}",
        (22, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.66,
        (244, 246, 248),
        2,
        cv2.LINE_AA,
    )
    detail = f"Raw: {output.raw_gesture.value}   FPS: {fps:4.1f}"
    if output.pinch_ratio is not None:
        detail += f"   Pinch: {output.pinch_ratio:.2f}"
    cv2.putText(
        frame,
        detail,
        (22, 94),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (174, 180, 190),
        1,
        cv2.LINE_AA,
    )
    footer = "SPACE pause/resume     Q or ESC quit"
    cv2.rectangle(frame, (0, height - 36), (width, height), (24, 27, 32), -1)
    cv2.putText(
        frame,
        footer,
        (22, height - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (220, 224, 230),
        1,
        cv2.LINE_AA,
    )


def _resolve_profile_path(explicit_path: Path | None = None) -> Path:
    if explicit_path is not None:
        return explicit_path.expanduser().resolve()
    if FALLBACK_PROFILE_PATH.exists() and not DEFAULT_PROFILE_PATH.exists():
        return FALLBACK_PROFILE_PATH.resolve()
    return DEFAULT_PROFILE_PATH.resolve()


def _load_profile_or_default(args: argparse.Namespace) -> GestureConfig:
    if args.no_profile:
        return GestureConfig(invert_scroll=args.invert_scroll)

    path = _resolve_profile_path(args.profile)
    if args.profile is not None:
        config = GestureConfig.load(path)
        print(f"[Gesture Mac] Loaded custom profile: {path}")
    elif path.exists():
        try:
            config = GestureConfig.load(path)
            print(f"[Gesture Mac] Loaded user profile: {path}")
        except (ValueError, OSError) as err:
            print(
                f"[Gesture Mac] Warning: could not load profile {path} ({err}); using defaults",
                file=sys.stderr,
            )
            config = GestureConfig()
    else:
        config = GestureConfig()

    if args.invert_scroll:
        config = GestureConfig.from_dict({**config.to_dict(), "invert_scroll": True})

    return config


def _draw_calibration_overlay(frame: Any, output: CalibrationOutput) -> None:
    height, width = frame.shape[:2]

    # Header banner
    cv2.rectangle(frame, (0, 0), (width, 106), (24, 27, 32), -1)

    title = (
        f"CALIBRATION  |  Step {output.step_index}/{output.total_steps}: "
        f"{output.step.value.replace('_', ' ').title()}"
    )
    cv2.putText(
        frame,
        title,
        (22, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.66,
        (112, 224, 190),
        2,
        cv2.LINE_AA,
    )

    # Instruction
    cv2.putText(
        frame,
        output.instruction,
        (22, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.54,
        (244, 246, 248),
        1,
        cv2.LINE_AA,
    )

    # Progress bar
    bar_width = width - 44
    fill_width = int(bar_width * max(0.0, min(1.0, output.progress)))
    cv2.rectangle(frame, (22, 78), (22 + bar_width, 88), (45, 50, 58), -1)
    if fill_width > 0:
        cv2.rectangle(frame, (22, 78), (22 + fill_width, 88), (112, 224, 190), -1)

    # Metric text
    if output.live_metric is not None and output.metric_label:
        metric_text = f"{output.metric_label}: {output.live_metric:.2f}"
        cv2.putText(
            frame,
            metric_text,
            (width - 240, 98),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 215, 0),
            1,
            cv2.LINE_AA,
        )

    # Footer
    footer = "SPACE toggle invert / advance     ENTER / Q save & finish     ESC cancel"
    cv2.rectangle(frame, (0, height - 36), (width, height), (24, 27, 32), -1)
    cv2.putText(
        frame,
        footer,
        (22, height - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (220, 224, 230),
        1,
        cv2.LINE_AA,
    )


def _run_calibration_loop(
    args: argparse.Namespace,
    model_path: Path,
    engine: CalibrationEngine,
    capture: cv2.VideoCapture,
    profile_path: Path,
) -> int:
    del args
    options = _landmarker_options(model_path)
    window = "Gesture Mac Calibration"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 960, 540)
    start_s = time.monotonic()

    consecutive_drop_count = 0
    with mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:
        while not engine.is_complete and not engine.is_cancelled:
            ok, frame = capture.read()
            if not ok:
                consecutive_drop_count += 1
                if consecutive_drop_count > 45:
                    raise RuntimeError(
                        "camera stopped returning frames after 45 retries"
                    )
                time.sleep(0.01)
                continue
            consecutive_drop_count = 0

            now_s = time.monotonic()
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int((now_s - start_s) * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)
            points = _landmarks(result)
            output = engine.update(points, now_s)

            _draw_hand(frame, points)
            _draw_calibration_overlay(frame, output)
            cv2.imshow(window, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                engine.cancel()
                break
            if key in (13, 10, ord("q")):  # ENTER or Q
                if engine.step in (
                    CalibrationStep.SCROLL_PREFERENCE,
                    CalibrationStep.COMPLETE,
                ):
                    engine.skip_step()
                    break
                engine.skip_step()
            elif key == ord(" "):
                if engine.step == CalibrationStep.SCROLL_PREFERENCE:
                    engine.toggle_invert_scroll()
                else:
                    engine.skip_step()

            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break

    if engine.is_cancelled:
        print("\n[Gesture Mac] Calibration cancelled.")
        return 1

    calibrated_config = engine._compile_config()
    calibrated_config.save(profile_path)
    print(f"\n[Gesture Mac] Calibration complete! Profile saved to: {profile_path}")
    print(
        f"  Active area: x=[{calibrated_config.active_left:.2f}, {calibrated_config.active_right:.2f}], "
        f"y=[{calibrated_config.active_top:.2f}, {calibrated_config.active_bottom:.2f}]"
    )
    print(
        f"  Pinch close / open ratios: {calibrated_config.pinch_close_ratio:.2f} / "
        f"{calibrated_config.pinch_open_ratio:.2f}"
    )
    print(f"  Scroll inverted: {calibrated_config.invert_scroll}")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preview or run camera-driven macOS hand gestures."
    )
    parser.add_argument(
        "--control",
        action="store_true",
        help="post real mouse/scroll events (default is safe preview only)",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=None,
        help="camera index (default: auto-detect Mac built-in camera)",
    )
    parser.add_argument(
        "--invert-scroll", action="store_true", help="reverse both scroll axes"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the model/runtime without opening the camera or controlling input",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="run guided onboarding and hand calibration wizard",
    )
    parser.add_argument(
        "--profile",
        type=Path,
        default=None,
        help=f"path to custom profile JSON (default: {DEFAULT_PROFILE_PATH})",
    )
    parser.add_argument(
        "--no-profile",
        action="store_true",
        help="ignore saved user profile and use defaults",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help=f"use a specific Hand Landmarker model (or set {MODEL_PATH_ENV})",
    )
    return parser


def _landmarker_options(model_path: Path) -> Any:
    return mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.60,
        min_hand_presence_confidence=0.60,
        min_tracking_confidence=0.55,
    )


def _check_runtime(model_path: Path) -> None:
    image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=np.zeros((240, 320, 3), dtype=np.uint8),
    )
    with mp.tasks.vision.HandLandmarker.create_from_options(
        _landmarker_options(model_path)
    ) as landmarker:
        result = landmarker.detect_for_video(image, 0)
    if result.hand_landmarks:
        raise RuntimeError("unexpected hand detection in the blank check frame")
    print(
        f"Runtime check passed (MediaPipe {mp.__version__}, model checksum verified)."
    )


def _run_capture_loop(
    args: argparse.Namespace,
    model_path: Path,
    engine: GestureEngine,
    controller: MacOSController | PreviewController,
    capture: cv2.VideoCapture,
) -> None:
    options = _landmarker_options(model_path)
    window = "Gesture Mac"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 960, 540)
    last_frame_s = time.monotonic()
    smooth_fps = 0.0
    start_s = last_frame_s

    consecutive_drop_count = 0
    with mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = capture.read()
            if not ok:
                consecutive_drop_count += 1
                if consecutive_drop_count > 45:
                    raise RuntimeError(
                        "camera stopped returning frames after 45 retries"
                    )
                time.sleep(0.01)
                continue
            consecutive_drop_count = 0

            now_s = time.monotonic()
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int((now_s - start_s) * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)
            points = _landmarks(result)
            output = engine.update(points, now_s)

            for action in output.actions:
                controller.apply(action)

            dt = max(now_s - last_frame_s, 1e-6)
            current_fps = 1.0 / dt
            smooth_fps = (
                current_fps
                if smooth_fps == 0.0
                else (0.9 * smooth_fps + 0.1 * current_fps)
            )
            last_frame_s = now_s

            _draw_hand(frame, points)
            _draw_overlay(frame, output, engine.config, smooth_fps, args.control)
            cv2.imshow(window, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                for action in engine.toggle_enabled():
                    controller.apply(action)
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break


def run(args: argparse.Namespace) -> int:
    if sys.platform != "darwin":
        raise RuntimeError("Gesture Mac's event backend supports macOS only")

    model_path = ensure_model(args.model)
    if args.check:
        _check_runtime(model_path)
        return 0

    if args.calibrate:
        profile_path = _resolve_profile_path(args.profile)
        cal_engine = CalibrationEngine()
        capture = _open_camera(args.camera)
        try:
            return _run_calibration_loop(
                args, model_path, cal_engine, capture, profile_path
            )
        finally:
            capture.release()
            cv2.destroyAllWindows()

    config = _load_profile_or_default(args)
    engine = GestureEngine(config)
    controller = MacOSController() if args.control else PreviewController()
    capture = _open_camera(args.camera)
    try:
        _run_capture_loop(args, model_path, engine, controller, capture)
    finally:
        controller.release_all()
        capture.release()
        cv2.destroyAllWindows()
    return 0


def main() -> None:
    try:
        raise SystemExit(run(_parser().parse_args()))
    except PermissionError as error:
        print(f"\nPermission needed: {error}", file=sys.stderr)
        raise SystemExit(2) from error
    except (RuntimeError, OSError) as error:
        print(f"\nGesture Mac could not start: {error}", file=sys.stderr)
        raise SystemExit(1) from error
    except KeyboardInterrupt:
        print("\nGesture Mac stopped.")
        raise SystemExit(130) from None


if __name__ == "__main__":
    main()
