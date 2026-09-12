"""Tests for startup helpers that must work outside a source checkout."""

from __future__ import annotations

import hashlib
import io
import sys
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

    monkeypatch.setattr(app, "_select_mac_camera", lambda requested: (7, "test camera"))
    monkeypatch.setattr(app.cv2, "VideoCapture", make_capture)

    with pytest.raises(RuntimeError, match="could not open test camera"):
        app._open_camera()

    assert len(created) == 2
    assert all(capture.released for capture in created)
