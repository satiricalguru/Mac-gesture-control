"""Tests for startup helpers that must work outside a source checkout."""

from __future__ import annotations

import hashlib
import io
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import ClassVar

import numpy as np
import pytest

from gesture_mac_prototype import app
from gesture_mac_prototype.gesture_engine import (
    EngineOutput,
    Gesture,
    GestureConfig,
    GestureEngine,
    Point,
)


def test_model_path_honors_explicit_environment_override(monkeypatch, tmp_path):
    expected = tmp_path / "offline-model.task"
    monkeypatch.setenv(app.MODEL_PATH_ENV, str(expected))

    assert app._model_path() == expected


def test_model_path_uses_existing_source_checkout_model(monkeypatch, tmp_path):
    expected = tmp_path / "models" / "hand_landmarker.task"
    expected.parent.mkdir()
    expected.write_bytes(b"existing")
    monkeypatch.delenv(app.MODEL_PATH_ENV, raising=False)
    monkeypatch.setattr(app, "_project_root", lambda: tmp_path)

    assert app._model_path() == expected


def test_sha256_streams_file_contents(tmp_path):
    target = tmp_path / "payload"
    target.write_bytes(b"abc" * 1_000_000)

    assert app._sha256(target) == hashlib.sha256(target.read_bytes()).hexdigest()


def test_ensure_model_downloads_atomically_and_checks_hash(monkeypatch, tmp_path):
    payload = b"verified hand model"
    expected_hash = hashlib.sha256(payload).hexdigest()
    target = tmp_path / "nested" / "hand_landmarker.task"
    calls = []

    def fake_urlopen(request, *, timeout):
        calls.append((request.full_url, timeout, request.headers))
        return io.BytesIO(payload)

    monkeypatch.setattr(app, "MODEL_SHA256", expected_hash)
    monkeypatch.setattr(app.urllib.request, "urlopen", fake_urlopen)

    assert app.ensure_model(target) == target
    assert target.read_bytes() == payload
    assert calls[0][0] == app.MODEL_URL
    assert calls[0][1] > 0
    assert list(target.parent.glob("*.download-*")) == []


def test_ensure_model_reuses_a_verified_local_copy(monkeypatch, tmp_path):
    payload = b"already verified"
    target = tmp_path / "model.task"
    target.write_bytes(payload)
    monkeypatch.setattr(app, "MODEL_SHA256", hashlib.sha256(payload).hexdigest())
    monkeypatch.setattr(
        app.urllib.request,
        "urlopen",
        lambda *args, **kwargs: pytest.fail("download should not be attempted"),
    )

    assert app.ensure_model(target) == target


def test_failed_download_does_not_replace_existing_model(monkeypatch, tmp_path):
    target = tmp_path / "hand_landmarker.task"
    target.write_bytes(b"old corrupt model")
    monkeypatch.setattr(app, "MODEL_SHA256", hashlib.sha256(b"expected").hexdigest())
    monkeypatch.setattr(
        app.urllib.request,
        "urlopen",
        lambda request, timeout: io.BytesIO(b"unexpected"),
    )

    with pytest.raises(RuntimeError, match="checksum mismatch"):
        app.ensure_model(target)

    assert target.read_bytes() == b"old corrupt model"


def test_oversized_download_is_removed(monkeypatch, tmp_path):
    target = tmp_path / "model.task"
    monkeypatch.setattr(app, "MODEL_MAX_BYTES", 4)
    monkeypatch.setattr(
        app.urllib.request,
        "urlopen",
        lambda request, timeout: io.BytesIO(b"too large"),
    )

    with pytest.raises(RuntimeError, match="safety limit"):
        app.ensure_model(target)

    assert not target.exists()
    assert list(tmp_path.glob("*.download-*")) == []


def test_model_download_refuses_non_https_url(monkeypatch, tmp_path):
    monkeypatch.setattr(app, "MODEL_URL", "http://example.test/model.task")

    with pytest.raises(RuntimeError, match="non-HTTPS"):
        app.ensure_model(tmp_path / "model.task")


def test_parser_accepts_an_offline_model_path(tmp_path):
    model_path = tmp_path / "model.task"

    args = app._parser().parse_args(["--model", str(model_path)])

    assert args.model == model_path


def test_landmark_conversion_handles_empty_and_populated_results():
    assert app._landmarks(SimpleNamespace(hand_landmarks=[])) is None
    source = [SimpleNamespace(x=i, y=i + 1, z=i + 2) for i in range(21)]

    assert app._landmarks(SimpleNamespace(hand_landmarks=[source])) == [
        Point(float(i), float(i + 1), float(i + 2)) for i in range(21)
    ]


def test_overlay_rendering_accepts_hand_and_no_hand_frames():
    frame = np.zeros((540, 960, 3), dtype=np.uint8)
    points = [Point(0.5, 0.5) for _ in range(21)]
    output = EngineOutput(
        raw_gesture=Gesture.MOVE,
        stable_gesture=Gesture.MOVE,
        enabled=True,
        dragging=False,
        pinch_ratio=0.25,
        actions=(),
    )

    app._draw_hand(frame, points)
    app._draw_hand(frame, None)
    app._draw_overlay(frame, output, GestureConfig(), 30.0, True)

    assert np.any(frame)


def test_run_releases_camera_when_preview_setup_fails(monkeypatch, tmp_path):
    class FakeCapture:
        released = False

        def release(self):
            self.released = True

    class FakeController:
        released = False

        def release_all(self):
            self.released = True

    capture = FakeCapture()
    controller = FakeController()
    monkeypatch.setattr(app.sys, "platform", "darwin")
    monkeypatch.setattr(app, "ensure_model", lambda path=None: tmp_path / "model")
    monkeypatch.setattr(app, "MacOSController", lambda: controller)
    monkeypatch.setattr(app, "_open_camera", lambda camera: capture)
    monkeypatch.setattr(app, "_landmarker_options", lambda model: object())
    monkeypatch.setattr(
        app.cv2,
        "namedWindow",
        lambda *args: (_ for _ in ()).throw(RuntimeError("window failed")),
    )

    with pytest.raises(RuntimeError, match="window failed"):
        app.run(app._parser().parse_args(["--control"]))

    assert capture.released
    assert controller.released


def test_run_check_does_not_open_camera(monkeypatch, tmp_path):
    checked = []
    model = tmp_path / "model.task"
    monkeypatch.setattr(app.sys, "platform", "darwin")
    monkeypatch.setattr(app, "ensure_model", lambda path=None: model)
    monkeypatch.setattr(app, "_check_runtime", checked.append)
    monkeypatch.setattr(
        app,
        "_open_camera",
        lambda camera: pytest.fail("camera should remain closed"),
    )

    assert app.run(app._parser().parse_args(["--check"])) == 0
    assert checked == [model]


def test_run_rejects_non_macos_before_model_or_camera_access(monkeypatch):
    monkeypatch.setattr(app.sys, "platform", "linux")
    monkeypatch.setattr(
        app,
        "ensure_model",
        lambda path=None: pytest.fail("model should not be accessed"),
    )

    with pytest.raises(RuntimeError, match="macOS only"):
        app.run(app._parser().parse_args([]))


def test_capture_loop_processes_a_frame_and_quits(monkeypatch, tmp_path):
    class FakeCapture:
        def read(self):
            return True, np.zeros((240, 320, 3), dtype=np.uint8)

    class FakeLandmarker:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            del args

        def detect_for_video(self, image, timestamp_ms):
            assert image is not None
            assert timestamp_ms >= 0
            return SimpleNamespace(hand_landmarks=[])

    landmarker_class = SimpleNamespace(
        create_from_options=lambda options: FakeLandmarker()
    )
    monkeypatch.setattr(app, "_landmarker_options", lambda model: object())
    monkeypatch.setattr(app.mp.tasks.vision, "HandLandmarker", landmarker_class)
    monkeypatch.setattr(app.cv2, "namedWindow", lambda *args: None)
    monkeypatch.setattr(app.cv2, "resizeWindow", lambda *args: None)
    monkeypatch.setattr(app.cv2, "imshow", lambda *args: None)
    monkeypatch.setattr(app.cv2, "waitKey", lambda delay: ord("q"))

    app._run_capture_loop(
        app._parser().parse_args([]),
        tmp_path / "model.task",
        GestureEngine(),
        app.PreviewController(),
        FakeCapture(),
    )


def test_camera_selection_prioritizes_builtin_without_changing_system_preference(
    monkeypatch,
):
    external = SimpleNamespace(
        localizedName=lambda: "Logitech Camera",
        deviceType=lambda: "AVCaptureDeviceTypeExternal",
    )
    builtin = SimpleNamespace(
        localizedName=lambda: "MacBook Pro Camera",
        deviceType=lambda: "AVCaptureDeviceTypeBuiltInWideAngleCamera",
    )

    class FakeDeviceClass:
        preference_changes: ClassVar[list[object]] = []

        @staticmethod
        def devicesWithMediaType_(media_type):
            assert media_type == "vide"
            return [external, builtin]

        @classmethod
        def setUserPreferredCamera_(cls, device):
            cls.preference_changes.append(device)

    class FakeDiscoveryClass:
        @staticmethod
        def discoverySessionWithDeviceTypes_mediaType_position_(*args):
            del args
            return SimpleNamespace(devices=lambda: [builtin])

    objc = ModuleType("objc")
    objc.loadBundle = lambda *args, **kwargs: None
    objc.lookUpClass = lambda name: {
        "AVCaptureDevice": FakeDeviceClass,
        "AVCaptureDeviceDiscoverySession": FakeDiscoveryClass,
    }[name]
    foundation = ModuleType("Foundation")
    foundation.NSBundle = SimpleNamespace(bundleWithPath_=lambda path: object())
    monkeypatch.setitem(sys.modules, "objc", objc)
    monkeypatch.setitem(sys.modules, "Foundation", foundation)

    assert app._select_mac_camera(None) == (
        1,
        "MacBook Pro Camera (Mac built-in, device 1)",
    )
    assert FakeDeviceClass.preference_changes == []


def test_open_camera_releases_every_failed_handle(monkeypatch):
    class FailedCapture:
        def __init__(self):
            self.released = False

        def isOpened(self):
            return False

        def release(self):
            self.released = True

        def set(self, *args):
            del args

    created: list[FailedCapture] = []

    def make_capture(*args):
        del args
        capture = FailedCapture()
        created.append(capture)
        return capture

    monkeypatch.setattr(app, "_list_mac_cameras", lambda: [(0, "FakeCamera", False)])
    monkeypatch.setattr(app, "_select_mac_camera", lambda requested: (7, "test camera"))
    monkeypatch.setattr(app.cv2, "VideoCapture", make_capture)

    with pytest.raises(RuntimeError, match="could not open test camera"):
        app._open_camera()

    assert len(created) == 2
    assert all(capture.released for capture in created)


def test_resolve_profile_path_explicit(tmp_path):
    custom = tmp_path / "custom.json"
    assert app._resolve_profile_path(custom) == custom.resolve()


def test_resolve_profile_path_fallback(monkeypatch, tmp_path):
    default_p = tmp_path / "default.json"
    fallback_p = tmp_path / "fallback.json"
    fallback_p.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(app, "DEFAULT_PROFILE_PATH", default_p)
    monkeypatch.setattr(app, "FALLBACK_PROFILE_PATH", fallback_p)

    assert app._resolve_profile_path(None) == fallback_p.resolve()


def test_load_profile_or_default_no_profile():
    args = app._parser().parse_args(["--no-profile", "--invert-scroll"])
    cfg = app._load_profile_or_default(args)
    assert cfg.invert_scroll is True
    assert cfg == GestureConfig(invert_scroll=True)


def test_load_profile_or_default_loads_existing_profile(tmp_path):
    custom_profile = tmp_path / "my_profile.json"
    saved_cfg = GestureConfig(active_left=0.22, pinch_close_ratio=0.31)
    saved_cfg.save(custom_profile)

    args = app._parser().parse_args(["--profile", str(custom_profile)])
    loaded_cfg = app._load_profile_or_default(args)
    assert loaded_cfg.active_left == 0.22
    assert loaded_cfg.pinch_close_ratio == 0.31


def test_load_profile_or_default_handles_corrupt_profile(monkeypatch, tmp_path):
    profile = tmp_path / "corrupt.json"
    profile.write_text("not json", encoding="utf-8")
    monkeypatch.setattr(app, "DEFAULT_PROFILE_PATH", profile)
    monkeypatch.setattr(app, "FALLBACK_PROFILE_PATH", tmp_path / "none.json")

    args = app._parser().parse_args([])
    cfg = app._load_profile_or_default(args)
    assert cfg == GestureConfig()


def test_parser_accepts_calibrate_flags():
    parser = app._parser()
    args = parser.parse_args(["--calibrate", "--profile", "/tmp/p.json"])
    assert args.calibrate is True
    assert args.profile == Path("/tmp/p.json")
    assert args.no_profile is False


def test_draw_calibration_overlay_executes_without_error():
    frame = np.zeros((540, 960, 3), dtype=np.uint8)
    engine = app.CalibrationEngine()
    out = engine.update([Point(0.5, 0.5)] * 21, 1.0)

    app._draw_calibration_overlay(frame, out)
    assert frame.shape == (540, 960, 3)


def test_run_dispatches_to_calibration_when_flag_is_set(monkeypatch, tmp_path):
    called = []

    monkeypatch.setattr(app.sys, "platform", "darwin")
    monkeypatch.setattr(app, "ensure_model", lambda p: tmp_path / "model.task")
    monkeypatch.setattr(
        app, "_open_camera", lambda cam: SimpleNamespace(release=lambda: None)
    )
    monkeypatch.setattr(app.cv2, "destroyAllWindows", lambda: None)

    def fake_cal_loop(args, model_path, engine, capture, profile_path):
        called.append((model_path, profile_path))
        return 0

    monkeypatch.setattr(app, "_run_calibration_loop", fake_cal_loop)

    target_profile = tmp_path / "profile.json"
    args = app._parser().parse_args(["--calibrate", "--profile", str(target_profile)])
    res = app.run(args)
    assert res == 0
    assert len(called) == 1
    assert called[0][1] == target_profile.resolve()


def test_run_calibration_loop_completion(monkeypatch, tmp_path):
    # Mock landmarker
    class FakeLandmarker:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def detect_for_video(self, img, ts):
            return SimpleNamespace(hand_landmarks=[])

    monkeypatch.setattr(
        app.mp.tasks.vision.HandLandmarker,
        "create_from_options",
        lambda opts: FakeLandmarker(),
    )
    monkeypatch.setattr(app.cv2, "namedWindow", lambda *args: None)
    monkeypatch.setattr(app.cv2, "resizeWindow", lambda *args: None)
    monkeypatch.setattr(app.cv2, "imshow", lambda *args: None)
    monkeypatch.setattr(app.cv2, "getWindowProperty", lambda *args: 1)

    # Return key 13 (ENTER) to skip to completion quickly
    keys = [ord(" "), ord(" "), ord(" "), ord(" "), 13]

    def fake_wait_key(delay):
        return keys.pop(0) if keys else 27

    monkeypatch.setattr(app.cv2, "waitKey", fake_wait_key)

    dummy_frame = np.zeros((540, 960, 3), dtype=np.uint8)
    fake_cap = SimpleNamespace(read=lambda: (True, dummy_frame))

    engine = app.CalibrationEngine()
    profile_out = tmp_path / "calibrated_profile.json"
    args = app._parser().parse_args(["--calibrate"])

    code = app._run_calibration_loop(
        args, tmp_path / "model.task", engine, fake_cap, profile_out
    )
    assert code == 0
    assert profile_out.exists()


def test_run_calibration_loop_cancellation(monkeypatch, tmp_path):
    class FakeLandmarker:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def detect_for_video(self, img, ts):
            return SimpleNamespace(hand_landmarks=[])

    monkeypatch.setattr(
        app.mp.tasks.vision.HandLandmarker,
        "create_from_options",
        lambda opts: FakeLandmarker(),
    )
    monkeypatch.setattr(app.cv2, "namedWindow", lambda *args: None)
    monkeypatch.setattr(app.cv2, "resizeWindow", lambda *args: None)
    monkeypatch.setattr(app.cv2, "imshow", lambda *args: None)
    monkeypatch.setattr(app.cv2, "getWindowProperty", lambda *args: 1)
    monkeypatch.setattr(app.cv2, "waitKey", lambda delay: 27)  # ESC

    dummy_frame = np.zeros((540, 960, 3), dtype=np.uint8)
    fake_cap = SimpleNamespace(read=lambda: (True, dummy_frame))

    engine = app.CalibrationEngine()
    profile_out = tmp_path / "cancelled_profile.json"
    args = app._parser().parse_args(["--calibrate"])

    code = app._run_calibration_loop(
        args, tmp_path / "model.task", engine, fake_cap, profile_out
    )
    assert code == 1
    assert not profile_out.exists()


def test_list_mac_cameras(monkeypatch):
    class FakeDevice:
        def __init__(self, name, dev_type):
            self._name = name
            self._type = dev_type

        def localizedName(self):
            return self._name

        def deviceType(self):
            return self._type

    class FakeDeviceClass:
        @classmethod
        def devicesWithMediaType_(cls, media_type):
            return [
                FakeDevice(
                    "FaceTime HD Camera", "AVCaptureDeviceTypeBuiltInWideAngleCamera"
                ),
                FakeDevice("My iPhone Camera", "AVCaptureDeviceTypeExternal"),
            ]

    objc = ModuleType("objc")
    objc.loadBundle = lambda *args, **kwargs: None
    objc.lookUpClass = lambda name: (
        FakeDeviceClass if name == "AVCaptureDevice" else None
    )

    foundation = ModuleType("Foundation")
    foundation.NSBundle = SimpleNamespace(bundleWithPath_=lambda path: object())
    monkeypatch.setitem(sys.modules, "objc", objc)
    monkeypatch.setitem(sys.modules, "Foundation", foundation)

    cams = app._list_mac_cameras()
    assert len(cams) == 2
    assert cams[0] == (0, "FaceTime HD Camera", False)
    assert cams[1] == (1, "My iPhone Camera", True)


def test_run_list_cameras(monkeypatch, capsys):
    monkeypatch.setattr(app.sys, "platform", "darwin")
    monkeypatch.setattr(
        app,
        "_list_mac_cameras",
        lambda: [(0, "FaceTime HD Camera", False), (1, "iPhone Camera", True)],
    )

    args = app._parser().parse_args(["--list-cameras"])
    code = app.run(args)
    assert code == 0
    captured = capsys.readouterr().out
    assert "FaceTime HD Camera (Built-in / Standard)" in captured
    assert "iPhone Camera (Continuity Camera (iPhone))" in captured


def test_open_camera_prints_continuity_notice(monkeypatch, capsys):
    class DummyCapture:
        def isOpened(self):
            return True

        def set(self, *args):
            del args

    monkeypatch.setattr(
        app,
        "_list_mac_cameras",
        lambda: [(0, "FaceTime HD Camera", False), (1, "Jatin's iPhone", True)],
    )
    monkeypatch.setattr(
        app, "_select_mac_camera", lambda req: (0, "FaceTime HD Camera")
    )
    monkeypatch.setattr(app.cv2, "VideoCapture", lambda *args: DummyCapture())

    app._open_camera()
    captured = capsys.readouterr().out
    assert "Apple Continuity Camera" in captured
    assert "Jatin's iPhone" in captured
    assert "turn OFF 'Continuity Camera'" in captured
