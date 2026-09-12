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

from .controller import MacOSController, PreviewController
from .gesture_engine import GestureConfig, GestureEngine, Point

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

    config = GestureConfig(invert_scroll=args.invert_scroll)
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
