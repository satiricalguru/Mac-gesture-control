"""Native AVFoundation camera capture that bypasses OpenCV's device selection.

macOS Continuity Camera can transparently redirect OpenCV's ``cv2.VideoCapture``
to a nearby iPhone even when the correct device index is specified. This module
creates its own ``AVCaptureSession`` via PyObjC, explicitly binding to the
built-in camera by unique-ID so the system cannot substitute another device.

The public helper ``open_native_camera`` returns an object with the same
``read()`` / ``release()`` / ``isOpened()`` / ``set()`` interface that
``cv2.VideoCapture`` exposes, so callers in *app.py* need no structural change.
"""

from __future__ import annotations

import ctypes
import sys
import threading
import time
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# ctypes wrappers for CoreVideo / CoreMedia C functions
# ---------------------------------------------------------------------------

_cv_lib = ctypes.cdll.LoadLibrary(
    "/System/Library/Frameworks/CoreVideo.framework/CoreVideo"
)
_cm_lib = ctypes.cdll.LoadLibrary(
    "/System/Library/Frameworks/CoreMedia.framework/CoreMedia"
)


def _setup_ctypes() -> None:
    """Register argument / return types once."""
    for fn_name, argtypes, restype in (
        ("CVPixelBufferLockBaseAddress", [ctypes.c_void_p, ctypes.c_uint64], ctypes.c_int32),
        ("CVPixelBufferUnlockBaseAddress", [ctypes.c_void_p, ctypes.c_uint64], ctypes.c_int32),
        ("CVPixelBufferGetBaseAddress", [ctypes.c_void_p], ctypes.c_void_p),
        ("CVPixelBufferGetWidth", [ctypes.c_void_p], ctypes.c_size_t),
        ("CVPixelBufferGetHeight", [ctypes.c_void_p], ctypes.c_size_t),
        ("CVPixelBufferGetBytesPerRow", [ctypes.c_void_p], ctypes.c_size_t),
    ):
        fn = getattr(_cv_lib, fn_name)
        fn.argtypes = argtypes
        fn.restype = restype

    _cm_lib.CMSampleBufferGetImageBuffer.argtypes = [ctypes.c_void_p]
    _cm_lib.CMSampleBufferGetImageBuffer.restype = ctypes.c_void_p


_setup_ctypes()

# Convenience aliases
_LockBase = _cv_lib.CVPixelBufferLockBaseAddress
_UnlockBase = _cv_lib.CVPixelBufferUnlockBaseAddress
_GetBase = _cv_lib.CVPixelBufferGetBaseAddress
_GetW = _cv_lib.CVPixelBufferGetWidth
_GetH = _cv_lib.CVPixelBufferGetHeight
_GetBPR = _cv_lib.CVPixelBufferGetBytesPerRow
_GetImageBuffer = _cm_lib.CMSampleBufferGetImageBuffer

_READ_ONLY_FLAG: int = 1  # kCVPixelBufferLock_ReadOnly
_BGRA: int = 1111970369  # kCVPixelFormatType_32BGRA


# ---------------------------------------------------------------------------
# ObjC bootstrap helpers
# ---------------------------------------------------------------------------


def _load_avfoundation() -> tuple[Any, ...]:
    """Return (AVCaptureDevice, AVCaptureSession, AVCaptureDeviceInput,
    AVCaptureVideoDataOutput, kCVPixelBufferPixelFormatTypeKey)."""
    import objc
    from Foundation import NSBundle

    for fw in (
        "/System/Library/Frameworks/AVFoundation.framework",
        "/System/Library/Frameworks/CoreMedia.framework",
    ):
        NSBundle.bundleWithPath_(fw)
        load = getattr(objc, "loadBundle", None)
        if callable(load):
            name = fw.rsplit("/", 1)[-1].replace(".framework", "")
            load(name, {}, bundle_path=fw)

    from Quartz import kCVPixelBufferPixelFormatTypeKey

    lookup = getattr(objc, "lookUpClass", None)
    if not callable(lookup):
        raise RuntimeError("objc.lookUpClass is not available")
    return (
        lookup("AVCaptureDevice"),
        lookup("AVCaptureSession"),
        lookup("AVCaptureDeviceInput"),
        lookup("AVCaptureVideoDataOutput"),
        kCVPixelBufferPixelFormatTypeKey,
    )


# ---------------------------------------------------------------------------
# Frame-grabbing delegate (runs on a private dispatch queue)
# ---------------------------------------------------------------------------


def _make_grabber_class() -> type:
    """Dynamically define the ObjC delegate class."""
    import objc
    from Foundation import NSObject

    _objc: Any = objc

    class _FrameGrabber(NSObject):  # type: ignore[misc]
        def init(self) -> _FrameGrabber:
            self = _objc.super(_FrameGrabber, self).init()
            if self is None:  # pragma: no cover
                return self
            self._latest: np.ndarray | None = None
            self._count: int = 0
            self._lock = threading.Lock()
            return self

        def captureOutput_didOutputSampleBuffer_fromConnection_(
            self, _output: Any, sample_buffer: Any, _connection: Any
        ) -> None:
            try:
                buf_ptr = _objc.pyobjc_id(sample_buffer)
                img_buf = _GetImageBuffer(buf_ptr)
                if not img_buf:
                    return
                _LockBase(img_buf, _READ_ONLY_FLAG)
                try:
                    base = _GetBase(img_buf)
                    w = _GetW(img_buf)
                    h = _GetH(img_buf)
                    bpr = _GetBPR(img_buf)
                    if base and w > 0 and h > 0:
                        raw = (ctypes.c_uint8 * (bpr * h)).from_address(base)
                        arr = np.frombuffer(raw, dtype=np.uint8).reshape((h, bpr))
                        bgra = arr[:, : w * 4].reshape((h, w, 4))
                        with self._lock:
                            self._latest = bgra[:, :, :3].copy()  # BGR
                            self._count += 1
                finally:
                    _UnlockBase(img_buf, _READ_ONLY_FLAG)
            except Exception:  # delegate must not throw  # nosec B110
                pass

        captureOutput_didOutputSampleBuffer_fromConnection_ = _objc.selector(
            captureOutput_didOutputSampleBuffer_fromConnection_,
            signature=b"v@:@@@",
        )

    return _FrameGrabber


# ---------------------------------------------------------------------------
# Public capture wrapper
# ---------------------------------------------------------------------------


class NativeCapture:
    """``cv2.VideoCapture``-compatible wrapper around a native AVCaptureSession.

    Only the subset of the OpenCV API that *app.py* uses is implemented:
    ``isOpened()``, ``read()``, ``release()``, and ``set()`` (width/height/fps
    are ignored—preset controls resolution).
    """

    def __init__(self, device_uid: str, preset: str = "AVCaptureSessionPreset640x480") -> None:
        import objc
        from Foundation import NSNumber

        _objc: Any = objc

        (
            av_device_cls,
            session_cls,
            input_cls,
            output_cls,
            px_fmt_key,
        ) = _load_avfoundation()

        device = av_device_cls.deviceWithUniqueID_(device_uid)
        if device is None:
            raise RuntimeError(f"AVCaptureDevice with uniqueID '{device_uid}' not found")

        self._device_name: str = str(device.localizedName())
        self._session = session_cls.alloc().init()
        self._session.setSessionPreset_(preset)

        dev_input = input_cls.deviceInputWithDevice_error_(device, None)
        if dev_input is None or not self._session.canAddInput_(dev_input):
            raise RuntimeError(f"cannot create capture input for {self._device_name}")
        self._session.addInput_(dev_input)

        video_output = output_cls.alloc().init()
        video_output.setAlwaysDiscardsLateVideoFrames_(True)
        video_output.setVideoSettings_(
            {px_fmt_key: NSNumber.numberWithInt_(_BGRA)}
        )

        GrabberCls = _make_grabber_class()
        self._grabber = GrabberCls.alloc().init()  # type: ignore[attr-defined]

        # Create a serial dispatch queue via libdispatch
        _libdispatch = ctypes.cdll.LoadLibrary("/usr/lib/system/libdispatch.dylib")
        _dq_create = _libdispatch.dispatch_queue_create
        _dq_create.argtypes = [ctypes.c_char_p, ctypes.c_void_p]
        _dq_create.restype = ctypes.c_void_p
        queue_ptr = _dq_create(b"gesture_mac_camera", None)
        queue = _objc.objc_object(c_void_p=queue_ptr)

        video_output.setSampleBufferDelegate_queue_(self._grabber, queue)

        if not self._session.canAddOutput_(video_output):
            raise RuntimeError("cannot add video output to capture session")
        self._session.addOutput_(video_output)

        self._session.startRunning()
        self._opened = True

        # Wait briefly for the first frame (camera warm-up)
        deadline = time.monotonic() + 1.5
        while time.monotonic() < deadline:
            with self._grabber._lock:
                if self._grabber._count > 0:
                    break
            time.sleep(0.02)

    # -- cv2.VideoCapture-compatible interface --

    def isOpened(self) -> bool:
        return self._opened and self._session.isRunning()

    def read(self) -> tuple[bool, np.ndarray | None]:
        with self._grabber._lock:
            frame = self._grabber._latest
            self._grabber._latest = None
        if frame is None:
            return False, None
        return True, frame

    def release(self) -> None:
        if self._opened:
            self._session.stopRunning()
            self._opened = False

    def set(self, _prop_id: int, _value: float) -> bool:
        """No-op; resolution is controlled by the session preset."""
        return False

    @property
    def device_name(self) -> str:
        return self._device_name


# ---------------------------------------------------------------------------
# Device discovery
# ---------------------------------------------------------------------------


def find_builtin_camera_uid() -> str | None:
    """Return the unique-ID of the Mac's built-in wide-angle camera, or None."""
    if sys.platform != "darwin":
        return None
    try:
        import objc
        from Foundation import NSBundle

        NSBundle.bundleWithPath_("/System/Library/Frameworks/AVFoundation.framework")
        load = getattr(objc, "loadBundle", None)
        if callable(load):
            load(
                "AVFoundation",
                {},
                bundle_path="/System/Library/Frameworks/AVFoundation.framework",
            )
        lookup = getattr(objc, "lookUpClass", None)
        if not callable(lookup):
            return None

        av_device_cls: Any = lookup("AVCaptureDevice")
        discovery_cls: Any = lookup("AVCaptureDeviceDiscoverySession")

        if discovery_cls is not None:
            discovery: Any = discovery_cls.discoverySessionWithDeviceTypes_mediaType_position_(
                ["AVCaptureDeviceTypeBuiltInWideAngleCamera"], "vide", 0
            )
            devices = discovery.devices() if discovery else ()
            for dev in devices:
                return str(dev.uniqueID())

        # Fallback: scan all video devices for a non-Continuity built-in
        for dev in av_device_cls.devicesWithMediaType_("vide"):
            dev_type = str(dev.deviceType())
            name = str(dev.localizedName())
            is_continuity = (
                "Continuity" in dev_type
                or "iPhone" in name
                or bool(getattr(dev, "isContinuityCamera", lambda: False)())
            )
            if not is_continuity and ("BuiltIn" in dev_type or "FaceTime" in name):
                return str(dev.uniqueID())
    except (ImportError, AttributeError, OSError):
        pass
    return None


def open_native_camera(
    requested_uid: str | None = None,
    preset: str = "AVCaptureSessionPreset960x540",
) -> NativeCapture:
    """Open the Mac's built-in camera via a native AVCaptureSession.

    Falls back to *requested_uid* or auto-detection of the built-in camera.
    """
    uid = requested_uid or find_builtin_camera_uid()
    if uid is None:
        raise RuntimeError(
            "could not detect the Mac's built-in camera; "
            "use --camera <index> to fall back to OpenCV capture"
        )
    cap = NativeCapture(uid, preset=preset)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(
            f"native AVCaptureSession for {cap.device_name} failed to start"
        )
    return cap
