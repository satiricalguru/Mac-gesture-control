"""Unit tests for native_camera module (AVFoundation capture and discovery)."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from gesture_mac_prototype import app, native_camera


def test_setup_ctypes():
    """Verify ctypes definitions and bindings are loaded without error."""
    native_camera._setup_ctypes()
    assert native_camera._LockBase is not None
    assert native_camera._UnlockBase is not None
    assert native_camera._GetBase is not None
    assert native_camera._GetW is not None
    assert native_camera._GetH is not None
    assert native_camera._GetBPR is not None
    assert native_camera._GetImageBuffer is not None


def test_load_avfoundation_success():
    """Test loading AVFoundation classes returns 5 expected elements on macOS."""
    res = native_camera._load_avfoundation()
    assert len(res) == 5
    av_device_cls, session_cls, input_cls, output_cls, px_fmt_key = res
    assert av_device_cls is not None
    assert session_cls is not None
    assert input_cls is not None
    assert output_cls is not None
    assert px_fmt_key is not None


def test_load_avfoundation_without_load_bundle(monkeypatch):
    """Test loading AVFoundation when objc.loadBundle is not callable."""
    import objc

    monkeypatch.setattr(objc, "loadBundle", None, raising=False)
    res = native_camera._load_avfoundation()
    assert len(res) == 5


def test_load_avfoundation_lookup_failure(monkeypatch):
    """Test that _load_avfoundation raises RuntimeError when lookUpClass is missing."""
    import objc

    monkeypatch.setattr(objc, "lookUpClass", None)
    with pytest.raises(RuntimeError, match=r"objc\.lookUpClass is not available"):
        native_camera._load_avfoundation()


def test_make_grabber_class_idempotent():
    """Verify _make_grabber_class returns the cached class on subsequent calls."""
    cls1 = native_camera._make_grabber_class()
    cls2 = native_camera._make_grabber_class()
    assert cls1 is cls2


def test_frame_grabber_init_and_null_buffer():
    """Test _FrameGrabber initializes cleanly and ignores null sample buffers."""
    GrabberCls = native_camera._make_grabber_class()
    grabber = GrabberCls.alloc().init()
    assert grabber._latest is None
    assert grabber._count == 0

    # Calling delegate with null buffer should return early without modifying state
    grabber.captureOutput_didOutputSampleBuffer_fromConnection_(None, None, None)
    assert grabber._latest is None
    assert grabber._count == 0


def test_frame_grabber_successful_frame_decode(monkeypatch):
    """Test decoding a simulated BGRA buffer into a BGR frame."""
    from Foundation import NSObject

    GrabberCls = native_camera._make_grabber_class()
    grabber = GrabberCls.alloc().init()

    sample_obj = NSObject.alloc().init()
    w, h, bpr = 8, 8, 32
    bgra_data = np.zeros((h, bpr), dtype=np.uint8)
    bgra_data[0, 0] = 210  # Blue
    bgra_data[0, 1] = 120  # Green
    bgra_data[0, 2] = 50  # Red
    bgra_data[0, 3] = 255  # Alpha
    base_addr = bgra_data.ctypes.data

    monkeypatch.setattr(native_camera, "_GetImageBuffer", lambda ptr: 9999)
    locked = []
    unlocked = []
    monkeypatch.setattr(
        native_camera, "_LockBase", lambda buf, flag: locked.append((buf, flag))
    )
    monkeypatch.setattr(
        native_camera, "_UnlockBase", lambda buf, flag: unlocked.append((buf, flag))
    )
    monkeypatch.setattr(native_camera, "_GetBase", lambda buf: base_addr)
    monkeypatch.setattr(native_camera, "_GetW", lambda buf: w)
    monkeypatch.setattr(native_camera, "_GetH", lambda buf: h)
    monkeypatch.setattr(native_camera, "_GetBPR", lambda buf: bpr)

    grabber.captureOutput_didOutputSampleBuffer_fromConnection_(None, sample_obj, None)
    assert grabber._count == 1
    assert grabber._latest is not None
    assert grabber._latest.shape == (h, w, 3)
    assert np.array_equal(grabber._latest[0, 0], [210, 120, 50])
    assert len(locked) == 1
    assert len(unlocked) == 1


def test_frame_grabber_null_base_address(monkeypatch):
    """Test delegate when base address or width is 0."""
    from Foundation import NSObject

    GrabberCls = native_camera._make_grabber_class()
    grabber = GrabberCls.alloc().init()
    sample_obj = NSObject.alloc().init()

    monkeypatch.setattr(native_camera, "_GetImageBuffer", lambda ptr: 9999)
    monkeypatch.setattr(native_camera, "_LockBase", lambda buf, flag: None)
    monkeypatch.setattr(native_camera, "_UnlockBase", lambda buf, flag: None)
    monkeypatch.setattr(native_camera, "_GetBase", lambda buf: 0)  # Null base pointer
    monkeypatch.setattr(native_camera, "_GetW", lambda buf: 0)
    monkeypatch.setattr(native_camera, "_GetH", lambda buf: 0)
    monkeypatch.setattr(native_camera, "_GetBPR", lambda buf: 0)

    grabber.captureOutput_didOutputSampleBuffer_fromConnection_(None, sample_obj, None)
    assert grabber._count == 0
    assert grabber._latest is None


def test_frame_grabber_exception_is_suppressed(monkeypatch):
    """Test that exceptions raised inside delegate are swallowed without throwing."""
    from Foundation import NSObject

    GrabberCls = native_camera._make_grabber_class()
    grabber = GrabberCls.alloc().init()
    sample_obj = NSObject.alloc().init()

    def bad_get_buffer(ptr):
        raise ValueError("simulated CoreMedia failure")

    monkeypatch.setattr(native_camera, "_GetImageBuffer", bad_get_buffer)
    grabber.captureOutput_didOutputSampleBuffer_fromConnection_(None, sample_obj, None)
    assert grabber._count == 0


class FakeAVDevice:
    def __init__(self, uid: str = "mock-uid", name: str = "Mock Camera"):
        self._uid = uid
        self._name = name

    def uniqueID(self):
        return self._uid

    def localizedName(self):
        return self._name


class FakeAVSession:
    def __init__(self):
        self.preset = None
        self.inputs = []
        self.outputs = []
        self._running = False
        self._can_add_input = True
        self._can_add_output = True

    def setSessionPreset_(self, preset):
        self.preset = preset

    def canAddInput_(self, dev_input):
        return self._can_add_input

    def addInput_(self, dev_input):
        self.inputs.append(dev_input)

    def canAddOutput_(self, dev_output):
        return self._can_add_output

    def addOutput_(self, dev_output):
        self.outputs.append(dev_output)

    def startRunning(self):
        self._running = True

    def stopRunning(self):
        self._running = False

    def isRunning(self):
        return self._running


class FakeAVInputCls:
    def __init__(self, should_succeed: bool = True):
        self.should_succeed = should_succeed

    def deviceInputWithDevice_error_(self, device, error):
        if not self.should_succeed:
            return None
        return SimpleNamespace(device=device)


class FakeAVOutput:
    def __init__(self):
        self.grabber = None
        self.queue = None

    def setAlwaysDiscardsLateVideoFrames_(self, val):
        pass

    def setVideoSettings_(self, val):
        pass

    def setSampleBufferDelegate_queue_(self, grabber, queue):
        self.grabber = grabber
        self.queue = queue
        # Signal warm-up frame arrived
        with grabber._lock:
            grabber._count = 1


def test_native_capture_device_not_found(monkeypatch):
    """Test NativeCapture raises when the device unique ID is not found."""
    mock_av_device_cls = SimpleNamespace(deviceWithUniqueID_=lambda uid: None)
    monkeypatch.setattr(
        native_camera,
        "_load_avfoundation",
        lambda: (mock_av_device_cls, None, None, None, "PixelFormatType"),
    )

    with pytest.raises(
        RuntimeError, match="AVCaptureDevice with uniqueID 'xyz' not found"
    ):
        native_camera.NativeCapture("xyz")


def test_native_capture_cannot_add_input(monkeypatch):
    """Test NativeCapture raises when capture input cannot be created or added."""
    mock_device = FakeAVDevice("cam-1", "Mac Built-in")
    mock_av_device_cls = SimpleNamespace(deviceWithUniqueID_=lambda uid: mock_device)
    session = FakeAVSession()
    session._can_add_input = False
    mock_session_cls = SimpleNamespace(
        alloc=lambda: SimpleNamespace(init=lambda: session)
    )
    mock_input_cls = FakeAVInputCls(should_succeed=True)

    monkeypatch.setattr(
        native_camera,
        "_load_avfoundation",
        lambda: (
            mock_av_device_cls,
            mock_session_cls,
            mock_input_cls,
            None,
            "PixelFormatType",
        ),
    )

    with pytest.raises(
        RuntimeError, match="cannot create capture input for Mac Built-in"
    ):
        native_camera.NativeCapture("cam-1")


def test_native_capture_cannot_add_output(monkeypatch):
    """Test NativeCapture raises when video output cannot be added to session."""
    mock_device = FakeAVDevice("cam-2", "Mac Built-in")
    mock_av_device_cls = SimpleNamespace(deviceWithUniqueID_=lambda uid: mock_device)
    session = FakeAVSession()
    session._can_add_output = False
    mock_session_cls = SimpleNamespace(
        alloc=lambda: SimpleNamespace(init=lambda: session)
    )
    mock_input_cls = FakeAVInputCls(should_succeed=True)
    mock_output_cls = SimpleNamespace(
        alloc=lambda: SimpleNamespace(init=lambda: FakeAVOutput())
    )

    monkeypatch.setattr(
        native_camera,
        "_load_avfoundation",
        lambda: (
            mock_av_device_cls,
            mock_session_cls,
            mock_input_cls,
            mock_output_cls,
            "PixelFormatType",
        ),
    )

    with pytest.raises(
        RuntimeError, match="cannot add video output to capture session"
    ):
        native_camera.NativeCapture("cam-2")


def test_native_capture_lifecycle_and_read(monkeypatch):
    """Test full successful lifecycle: init, isOpened, read, set, and release."""
    mock_device = FakeAVDevice("cam-3", "FaceTime HD Camera")
    mock_av_device_cls = SimpleNamespace(deviceWithUniqueID_=lambda uid: mock_device)
    session = FakeAVSession()
    mock_session_cls = SimpleNamespace(
        alloc=lambda: SimpleNamespace(init=lambda: session)
    )
    mock_input_cls = FakeAVInputCls(should_succeed=True)
    mock_output_cls = SimpleNamespace(
        alloc=lambda: SimpleNamespace(init=lambda: FakeAVOutput())
    )

    monkeypatch.setattr(
        native_camera,
        "_load_avfoundation",
        lambda: (
            mock_av_device_cls,
            mock_session_cls,
            mock_input_cls,
            mock_output_cls,
            "PixelFormatType",
        ),
    )

    cap = native_camera.NativeCapture("cam-3", preset="AVCaptureSessionPreset640x480")
    assert cap.device_name == "FaceTime HD Camera"
    assert cap.isOpened() is True
    assert cap.set(3, 1280) is False

    # Read when no frame is ready
    ret, frame = cap.read()
    assert ret is False
    assert frame is None

    # Read when a frame is available
    dummy_frame = np.ones((480, 640, 3), dtype=np.uint8)
    with cap._grabber._lock:
        cap._grabber._latest = dummy_frame

    ret, frame = cap.read()
    assert ret is True
    assert frame is not None
    assert np.array_equal(frame, dummy_frame)
    # Ensure buffer was cleared after read
    assert cap._grabber._latest is None

    # Release
    cap.release()
    assert cap.isOpened() is False
    assert session.isRunning() is False

    # Idempotent release
    cap.release()
    assert cap.isOpened() is False


def test_native_capture_warmup_timeout(monkeypatch):
    """Test warm-up loop times out cleanly when no frames are produced."""
    mock_device = FakeAVDevice("cam-timeout", "Silent Camera")
    mock_av_device_cls = SimpleNamespace(deviceWithUniqueID_=lambda uid: mock_device)
    session = FakeAVSession()
    mock_session_cls = SimpleNamespace(
        alloc=lambda: SimpleNamespace(init=lambda: session)
    )
    mock_input_cls = FakeAVInputCls(should_succeed=True)

    class SilentOutput:
        def setAlwaysDiscardsLateVideoFrames_(self, val):
            pass

        def setVideoSettings_(self, val):
            pass

        def setSampleBufferDelegate_queue_(self, grabber, queue):
            # Does not increment count
            pass

    mock_output_cls = SimpleNamespace(
        alloc=lambda: SimpleNamespace(init=lambda: SilentOutput())
    )

    monkeypatch.setattr(
        native_camera,
        "_load_avfoundation",
        lambda: (
            mock_av_device_cls,
            mock_session_cls,
            mock_input_cls,
            mock_output_cls,
            "PixelFormatType",
        ),
    )

    # Fast forward time to avoid waiting 1.5s
    times = [0.0, 0.0, 2.0]
    monkeypatch.setattr(
        native_camera.time,
        "monotonic",
        lambda: times.pop(0) if times else 2.0,
    )
    monkeypatch.setattr(native_camera.time, "sleep", lambda s: None)

    cap = native_camera.NativeCapture("cam-timeout")
    assert cap.isOpened() is True
    cap.release()


def test_find_builtin_camera_uid_non_darwin(monkeypatch):
    """Verify find_builtin_camera_uid returns None on non-macOS platforms."""
    monkeypatch.setattr(sys, "platform", "linux")
    assert native_camera.find_builtin_camera_uid() is None


def test_find_builtin_camera_uid_discovery_session(monkeypatch):
    """Test finding built-in camera via AVCaptureDeviceDiscoverySession."""
    monkeypatch.setattr(sys, "platform", "darwin")

    dev = SimpleNamespace(uniqueID=lambda: "discovered-built-in-uid")
    discovery = SimpleNamespace(devices=lambda: [dev])
    discovery_cls = SimpleNamespace(
        discoverySessionWithDeviceTypes_mediaType_position_=lambda *args: discovery
    )
    av_device_cls = SimpleNamespace(devicesWithMediaType_=lambda *args: [])

    import objc

    def mock_lookup(name):
        if name == "AVCaptureDeviceDiscoverySession":
            return discovery_cls
        if name == "AVCaptureDevice":
            return av_device_cls
        return None

    monkeypatch.setattr(objc, "lookUpClass", mock_lookup)

    assert native_camera.find_builtin_camera_uid() == "discovered-built-in-uid"


def test_find_builtin_camera_uid_fallback_devices(monkeypatch):
    """Test discovery fallback scanning video devices and skipping Continuity cameras."""
    monkeypatch.setattr(sys, "platform", "darwin")

    iphone = SimpleNamespace(
        deviceType=lambda: "AVCaptureDeviceTypeContinuityCamera",
        localizedName=lambda: "Jatin's iPhone",
        uniqueID=lambda: "iphone-uid",
        isContinuityCamera=lambda: True,
    )
    facetime = SimpleNamespace(
        deviceType=lambda: "AVCaptureDeviceTypeBuiltInWideAngleCamera",
        localizedName=lambda: "FaceTime HD Camera",
        uniqueID=lambda: "facetime-uid-999",
        isContinuityCamera=lambda: False,
    )

    discovery = SimpleNamespace(devices=lambda: [])
    discovery_cls = SimpleNamespace(
        discoverySessionWithDeviceTypes_mediaType_position_=lambda *args: discovery
    )
    av_device_cls = SimpleNamespace(
        devicesWithMediaType_=lambda *args: [iphone, facetime]
    )

    import objc

    def mock_lookup(name):
        if name == "AVCaptureDeviceDiscoverySession":
            return discovery_cls
        if name == "AVCaptureDevice":
            return av_device_cls
        return None

    monkeypatch.setattr(objc, "lookUpClass", mock_lookup)

    assert native_camera.find_builtin_camera_uid() == "facetime-uid-999"


def test_find_builtin_camera_uid_none_found(monkeypatch):
    """Test returning None when only Continuity devices are present."""
    monkeypatch.setattr(sys, "platform", "darwin")

    iphone = SimpleNamespace(
        deviceType=lambda: "AVCaptureDeviceTypeContinuityCamera",
        localizedName=lambda: "iPhone",
        uniqueID=lambda: "iphone-uid",
    )
    discovery_cls = SimpleNamespace(
        discoverySessionWithDeviceTypes_mediaType_position_=lambda *args: (
            SimpleNamespace(devices=lambda: [])
        )
    )
    av_device_cls = SimpleNamespace(devicesWithMediaType_=lambda *args: [iphone])

    import objc

    def mock_lookup(name):
        if name == "AVCaptureDeviceDiscoverySession":
            return discovery_cls
        if name == "AVCaptureDevice":
            return av_device_cls
        return None

    monkeypatch.setattr(objc, "lookUpClass", mock_lookup)

    assert native_camera.find_builtin_camera_uid() is None


def test_find_builtin_camera_uid_lookup_not_callable(monkeypatch):
    """Test returning None when lookUpClass is missing or not callable."""
    monkeypatch.setattr(sys, "platform", "darwin")
    import objc

    monkeypatch.setattr(objc, "lookUpClass", None)
    assert native_camera.find_builtin_camera_uid() is None


def test_find_builtin_camera_uid_handles_exception(monkeypatch):
    """Test returning None when an exception occurs during discovery."""
    monkeypatch.setattr(sys, "platform", "darwin")
    import objc

    def bad_lookup(name):
        raise OSError("framework unavailable")

    monkeypatch.setattr(objc, "lookUpClass", bad_lookup)
    assert native_camera.find_builtin_camera_uid() is None


def test_open_native_camera_success(monkeypatch):
    """Test open_native_camera successfully instantiates NativeCapture."""
    monkeypatch.setattr(
        native_camera, "find_builtin_camera_uid", lambda: "mock-builtin-uid"
    )

    class DummyNativeCapture:
        def __init__(self, uid, preset):
            self.uid = uid
            self.preset = preset
            self.device_name = "Mock Device"

        def isOpened(self):
            return True

        def release(self):
            pass

    monkeypatch.setattr(native_camera, "NativeCapture", DummyNativeCapture)

    cap = native_camera.open_native_camera()
    assert isinstance(cap, DummyNativeCapture)
    assert cap.uid == "mock-builtin-uid"


def test_open_native_camera_no_uid(monkeypatch):
    """Test open_native_camera raises when no built-in camera UID is found."""
    monkeypatch.setattr(native_camera, "find_builtin_camera_uid", lambda: None)

    with pytest.raises(
        RuntimeError, match="could not detect the Mac's built-in camera"
    ):
        native_camera.open_native_camera()


def test_open_native_camera_start_failure(monkeypatch):
    """Test open_native_camera raises and releases if session fails to open."""
    released = []

    class FailingNativeCapture:
        def __init__(self, uid, preset):
            self.device_name = "Failing Cam"

        def isOpened(self):
            return False

        def release(self):
            released.append(True)

    monkeypatch.setattr(native_camera, "NativeCapture", FailingNativeCapture)

    with pytest.raises(RuntimeError, match="failed to start"):
        native_camera.open_native_camera("fake-uid")

    assert len(released) == 1


def test_app_open_camera_uses_native_when_continuity_detected(monkeypatch, capsys):
    """Test that app._open_camera invokes open_native_camera when Continuity is detected."""
    monkeypatch.setattr(
        app,
        "_list_mac_cameras",
        lambda: [(0, "FaceTime HD Camera", False), (1, "iPhone 15", True)],
    )

    class FakeNativeCap:
        def __init__(self):
            self.device_name = "FaceTime HD Camera (native)"

        def isOpened(self):
            return True

    monkeypatch.setattr(
        native_camera, "find_builtin_camera_uid", lambda: "builtin-uid-123"
    )
    monkeypatch.setattr(
        native_camera, "open_native_camera", lambda uid: FakeNativeCap()
    )

    cap = app._open_camera(requested_index=None)
    assert isinstance(cap, FakeNativeCap)
    captured = capsys.readouterr().out
    assert "Apple Continuity Camera (iPhone 15) detected" in captured
    assert "Locked to: FaceTime HD Camera (native)" in captured


def test_app_open_camera_falls_back_when_native_fails(monkeypatch, capsys):
    """Test that app._open_camera falls back to OpenCV when native capture throws."""
    monkeypatch.setattr(
        app,
        "_list_mac_cameras",
        lambda: [(0, "FaceTime HD Camera", False), (1, "iPhone 15", True)],
    )
    monkeypatch.setattr(
        app, "_select_mac_camera", lambda req: (0, "FaceTime HD Camera")
    )

    class DummyOpenCVCapture:
        def isOpened(self):
            return True

        def set(self, *args):
            pass

    def failing_open_native(uid):
        raise RuntimeError("AVCaptureSession error")

    monkeypatch.setattr(
        native_camera, "find_builtin_camera_uid", lambda: "builtin-uid-123"
    )
    monkeypatch.setattr(native_camera, "open_native_camera", failing_open_native)
    monkeypatch.setattr(app.cv2, "VideoCapture", lambda *args: DummyOpenCVCapture())

    cap = app._open_camera(requested_index=None)
    assert isinstance(cap, DummyOpenCVCapture)
    captured = capsys.readouterr()
    assert "Native capture failed" in captured.err
    assert "Apple Continuity Camera" in captured.out
