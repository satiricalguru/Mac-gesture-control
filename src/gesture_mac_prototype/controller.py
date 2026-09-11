"""macOS Quartz adapter for semantic gesture actions."""

from __future__ import annotations

from dataclasses import dataclass

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

        display_id = Quartz.CGMainDisplayID()
        rect = Quartz.CGDisplayBounds(display_id)
        self.bounds = DisplayBounds(
            float(rect.origin.x),
            float(rect.origin.y),
            float(rect.size.width),
            float(rect.size.height),
        )
        self.position = (
            self.bounds.x + self.bounds.width / 2.0,
            self.bounds.y + self.bounds.height / 2.0,
        )
        self.left_is_down = False

    def _post_mouse(self, event_type: int, button: int) -> None:
        event = self.q.CGEventCreateMouseEvent(None, event_type, self.position, button)
        self.q.CGEventPost(self.q.kCGHIDEventTap, event)

    def _move(self, x: float, y: float) -> None:
        self.position = (
            self.bounds.x + x * max(1.0, self.bounds.width - 1.0),
            self.bounds.y + y * max(1.0, self.bounds.height - 1.0),
        )
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
