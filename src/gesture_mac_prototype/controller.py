"""macOS Quartz adapter for semantic gesture actions."""

from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass
from math import hypot, isfinite
from time import monotonic
from typing import Any

from .gesture_engine import Action, ActionKind


class PreviewController:
    """Safe sink used while the user checks recognition in the camera preview."""

    def apply(self, action: Action) -> None:
        del action

    def release_all(self) -> None:
        pass


@dataclass(frozen=True)
class DisplayBounds:
    x: float
    y: float
    width: float
    height: float

    def __post_init__(self) -> None:
        if not all(
            isfinite(value) for value in (self.x, self.y, self.width, self.height)
        ):
            raise ValueError("display bounds must be finite with positive dimensions")
        if self.width <= 0.0 or self.height <= 0.0:
            raise ValueError("display bounds must be finite with positive dimensions")

    @property
    def right(self) -> float:
        return self.x + self.width

    @property
    def bottom(self) -> float:
        return self.y + self.height

    def contains(self, x: float, y: float) -> bool:
        return self.x <= x < self.right and self.y <= y < self.bottom

    def clamp(self, x: float, y: float) -> tuple[float, float]:
        return (
            min(max(x, self.x), self.right - 1.0),
            min(max(y, self.y), self.bottom - 1.0),
        )


class DisplayLayout:
    """Maps normalized hand coordinates into the complete macOS desktop."""

    def __init__(self, displays: tuple[DisplayBounds, ...]) -> None:
        if not displays:
            raise ValueError("at least one active display is required")
        self.displays = displays
        left = min(display.x for display in displays)
        top = min(display.y for display in displays)
        right = max(display.right for display in displays)
        bottom = max(display.bottom for display in displays)
        self.bounds = DisplayBounds(left, top, right - left, bottom - top)

    def map(self, x: float, y: float) -> tuple[float, float]:
        """Map [0, 1] coordinates and snap virtual-desktop gaps to a display."""
        if not isfinite(x) or not isfinite(y):
            raise ValueError("pointer coordinates must be finite")
        x = min(max(x, 0.0), 1.0)
        y = min(max(y, 0.0), 1.0)
        target = (
            self.bounds.x + x * max(1.0, self.bounds.width - 1.0),
            self.bounds.y + y * max(1.0, self.bounds.height - 1.0),
        )
        if any(display.contains(*target) for display in self.displays):
            return target

        candidates = [display.clamp(*target) for display in self.displays]
        return min(
            candidates,
            key=lambda point: hypot(point[0] - target[0], point[1] - target[1]),
        )


def _active_display_layout(quartz: Any) -> DisplayLayout:
    error, display_ids, count = quartz.CGGetActiveDisplayList(32, None, None)
    if error != quartz.kCGErrorSuccess:
        raise RuntimeError(
            f"could not enumerate active displays (Quartz error {error})"
        )
    display_ids = tuple(display_ids)[:count]
    if not display_ids:
        raise RuntimeError("macOS reported no active displays")

    displays = []
    seen: set[tuple[float, float, float, float]] = set()
    for display_id in display_ids:
        rect = quartz.CGDisplayBounds(display_id)
        values = (
            float(rect.origin.x),
            float(rect.origin.y),
            float(rect.size.width),
            float(rect.size.height),
        )
        # Mirrored displays can report the same global rectangle.
        if values not in seen:
            seen.add(values)
            displays.append(DisplayBounds(*values))
    return DisplayLayout(tuple(displays))


class MacOSController:
    def __init__(self, *, prompt_for_accessibility: bool = True) -> None:
        import ApplicationServices
        import Quartz

        self.q = Quartz
        options = {
            ApplicationServices.kAXTrustedCheckOptionPrompt: prompt_for_accessibility
        }
        if not ApplicationServices.AXIsProcessTrustedWithOptions(options):
            raise PermissionError(
                "Accessibility permission is required. Enable your terminal in "
                "System Settings > Privacy & Security > Accessibility, then restart it."
            )

        self.layout = _active_display_layout(Quartz)
        self._layout_refreshed_s = monotonic()
        self.position = self._cursor_position()
        self.left_is_down = False

    def _refresh_layout_if_needed(self) -> None:
        now_s = monotonic()
        if now_s - self._layout_refreshed_s < 2.0:
            return
        # Display reconfiguration can report a transient empty/error state.
        # Keep the last valid layout and retry on a future pointer frame.
        with suppress(RuntimeError):
            self.layout = _active_display_layout(self.q)
        self._layout_refreshed_s = now_s

    def _cursor_position(self) -> tuple[float, float]:
        event = self.q.CGEventCreate(None)
        if event is None:
            return self.layout.map(0.5, 0.5)
        point = self.q.CGEventGetLocation(event)
        return float(point.x), float(point.y)

    def _post_mouse(self, event_type: int, button: int) -> None:
        event = self.q.CGEventCreateMouseEvent(None, event_type, self.position, button)
        if event is None:
            raise RuntimeError("Quartz could not create a mouse event")
        self.q.CGEventPost(self.q.kCGHIDEventTap, event)

    def _move(self, x: float, y: float) -> None:
        self._refresh_layout_if_needed()
        self.position = self.layout.map(x, y)
        event_type = (
            self.q.kCGEventLeftMouseDragged
            if self.left_is_down
            else self.q.kCGEventMouseMoved
        )
        self._post_mouse(event_type, self.q.kCGMouseButtonLeft)

    def _left_click(self) -> None:
        self._post_mouse(self.q.kCGEventLeftMouseDown, self.q.kCGMouseButtonLeft)
        self._post_mouse(self.q.kCGEventLeftMouseUp, self.q.kCGMouseButtonLeft)

    def _right_click(self) -> None:
        # A right-pinch does not continuously emit MOVE actions. Re-read the real
        # cursor so a physical mouse move or a fresh session cannot click stale
        # coordinates.
        if not self.left_is_down:
            self.position = self._cursor_position()
        self._post_mouse(self.q.kCGEventRightMouseDown, self.q.kCGMouseButtonRight)
        self._post_mouse(self.q.kCGEventRightMouseUp, self.q.kCGMouseButtonRight)

    def _scroll(self, dx: float, dy: float) -> None:
        event = self.q.CGEventCreateScrollWheelEvent(
            None,
            self.q.kCGScrollEventUnitPixel,
            2,
            round(dy),
            round(dx),
        )
        if event is None:
            raise RuntimeError("Quartz could not create a scroll event")
        self.q.CGEventPost(self.q.kCGHIDEventTap, event)

    def apply(self, action: Action) -> None:
        if action.kind == ActionKind.MOVE:
            self._move(action.x, action.y)
        elif action.kind == ActionKind.LEFT_CLICK:
            self._left_click()
        elif action.kind == ActionKind.LEFT_DOWN and not self.left_is_down:
            self._post_mouse(self.q.kCGEventLeftMouseDown, self.q.kCGMouseButtonLeft)
            self.left_is_down = True
        elif action.kind == ActionKind.LEFT_UP and self.left_is_down:
            self._post_mouse(self.q.kCGEventLeftMouseUp, self.q.kCGMouseButtonLeft)
            self.left_is_down = False
        elif action.kind == ActionKind.RIGHT_CLICK:
            self._right_click()
        elif action.kind == ActionKind.SCROLL:
            self._scroll(action.dx, action.dy)

    def release_all(self) -> None:
        if self.left_is_down:
            self._post_mouse(self.q.kCGEventLeftMouseUp, self.q.kCGMouseButtonLeft)
            self.left_is_down = False
