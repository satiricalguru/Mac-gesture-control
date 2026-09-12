"""Tests for virtual-desktop mapping and Quartz event safety."""

from __future__ import annotations

from math import nan
from types import SimpleNamespace

import pytest

from gesture_mac_prototype.controller import (
    DisplayBounds,
    DisplayLayout,
    MacOSController,
    _active_display_layout,
)
from gesture_mac_prototype.gesture_engine import Action, ActionKind


class FakeApplicationServices:
    kAXTrustedCheckOptionPrompt = "prompt"

    def __init__(self, *, trusted: bool = True) -> None:
        self.trusted = trusted
        self.options = None

    def AXIsProcessTrustedWithOptions(self, options):
        self.options = options
        return self.trusted


class FakeQuartz:
    kCGErrorSuccess = 0
    kCGEventLeftMouseDragged = "left_dragged"
    kCGEventMouseMoved = "moved"
    kCGEventLeftMouseDown = "left_down"
    kCGEventLeftMouseUp = "left_up"
    kCGEventRightMouseDown = "right_down"
    kCGEventRightMouseUp = "right_up"
    kCGMouseButtonLeft = "left"
    kCGMouseButtonRight = "right"
    kCGScrollEventUnitPixel = "pixel"
    kCGHIDEventTap = "hid"

    def __init__(self) -> None:
        self.cursor = (37.0, 41.0)
        self.posted: list[dict] = []
        self.fail_mouse_event = False
        self.fail_scroll_event = False
        self.displays = {
            11: (0.0, 0.0, 100.0, 100.0),
            12: (100.0, -50.0, 200.0, 150.0),
        }

    def CGGetActiveDisplayList(self, maximum, displays, count):
        del maximum, displays, count
        return self.kCGErrorSuccess, tuple(self.displays), len(self.displays)

    def CGDisplayBounds(self, display_id):
        x, y, width, height = self.displays[display_id]
        return SimpleNamespace(
            origin=SimpleNamespace(x=x, y=y),
            size=SimpleNamespace(width=width, height=height),
        )

    def CGEventCreate(self, source):
        del source
        return {"kind": "current", "position": self.cursor}

    def CGEventGetLocation(self, event):
        return SimpleNamespace(x=event["position"][0], y=event["position"][1])

    def CGEventCreateMouseEvent(self, source, event_type, position, button):
        del source
        if self.fail_mouse_event:
            return None
        return {
            "kind": event_type,
            "position": tuple(position),
            "button": button,
        }

    def CGEventCreateScrollWheelEvent(self, source, unit, axes, dy, dx):
        del source
        if self.fail_scroll_event:
            return None
        return {"kind": "scroll", "unit": unit, "axes": axes, "dx": dx, "dy": dy}

    def CGEventPost(self, tap, event):
        assert tap == self.kCGHIDEventTap
        self.posted.append(event)


def _controller(monkeypatch, *, trusted: bool = True):
    quartz = FakeQuartz()
    accessibility = FakeApplicationServices(trusted=trusted)
    monkeypatch.setitem(__import__("sys").modules, "Quartz", quartz)
    monkeypatch.setitem(__import__("sys").modules, "ApplicationServices", accessibility)
    return MacOSController(), quartz, accessibility


def test_display_layout_maps_entire_desktop_and_avoids_gaps():
    layout = DisplayLayout(
        (
            DisplayBounds(0.0, 0.0, 100.0, 100.0),
            DisplayBounds(200.0, 100.0, 100.0, 100.0),
        )
    )

    assert layout.map(0.0, 0.0) == (0.0, 0.0)
    assert layout.map(1.0, 1.0) == (299.0, 199.0)
    assert layout.map(0.5, 0.25) == pytest.approx((99.0, 49.75))


def test_display_bounds_reject_invalid_dimensions():
    with pytest.raises(ValueError, match="positive"):
        DisplayBounds(0.0, 0.0, 0.0, 100.0)
    with pytest.raises(ValueError, match="positive"):
        DisplayBounds(0.0, 0.0, nan, 100.0)


def test_display_layout_rejects_non_finite_pointer_coordinates():
    layout = DisplayLayout((DisplayBounds(0.0, 0.0, 100.0, 100.0),))

    with pytest.raises(ValueError, match="finite"):
        layout.map(nan, 0.5)


def test_display_discovery_rejects_quartz_errors_and_empty_results():
    quartz = FakeQuartz()
    quartz.kCGErrorSuccess = 0
    quartz.CGGetActiveDisplayList = lambda *args: (17, (), 0)
    with pytest.raises(RuntimeError, match="Quartz error 17"):
        _active_display_layout(quartz)

    quartz.CGGetActiveDisplayList = lambda *args: (0, (), 0)
    with pytest.raises(RuntimeError, match="no active displays"):
        _active_display_layout(quartz)


def test_controller_uses_current_cursor_for_right_click(monkeypatch):
    controller, quartz, accessibility = _controller(monkeypatch)

    assert accessibility.options == {"prompt": True}
    assert controller.position == quartz.cursor

    quartz.cursor = (83.0, 69.0)
    controller.apply(Action(ActionKind.RIGHT_CLICK))

    assert [event["kind"] for event in quartz.posted] == [
        "right_down",
        "right_up",
    ]
    assert all(event["position"] == quartz.cursor for event in quartz.posted)


def test_controller_maps_pointer_across_all_active_displays(monkeypatch):
    controller, quartz, _ = _controller(monkeypatch)

    controller.apply(Action(ActionKind.MOVE, x=1.0, y=0.0))

    assert quartz.posted[-1]["kind"] == "moved"
    assert quartz.posted[-1]["position"] == (299.0, -50.0)


def test_controller_releases_drag_exactly_once(monkeypatch):
    controller, quartz, _ = _controller(monkeypatch)

    controller.apply(Action(ActionKind.LEFT_DOWN))
    controller.release_all()
    controller.release_all()

    assert [event["kind"] for event in quartz.posted] == ["left_down", "left_up"]


def test_controller_emits_click_drag_and_pixel_scroll_events(monkeypatch):
    controller, quartz, _ = _controller(monkeypatch)

    controller.apply(Action(ActionKind.LEFT_CLICK))
    controller.apply(Action(ActionKind.LEFT_DOWN))
    controller.apply(Action(ActionKind.LEFT_DOWN))
    controller.apply(Action(ActionKind.MOVE, x=0.25, y=0.5))
    controller.apply(Action(ActionKind.LEFT_UP))
    controller.apply(Action(ActionKind.LEFT_UP))
    controller.apply(Action(ActionKind.SCROLL, dx=3.6, dy=-7.4))
    controller.apply(Action(ActionKind.ENABLED_CHANGED, enabled=False))

    assert [event["kind"] for event in quartz.posted] == [
        "left_down",
        "left_up",
        "left_down",
        "left_dragged",
        "left_up",
        "scroll",
    ]
    assert quartz.posted[-1] == {
        "kind": "scroll",
        "unit": "pixel",
        "axes": 2,
        "dx": 4,
        "dy": -7,
    }


def test_controller_surfaces_native_event_creation_failures(monkeypatch):
    controller, quartz, _ = _controller(monkeypatch)
    quartz.fail_mouse_event = True
    with pytest.raises(RuntimeError, match="mouse event"):
        controller.apply(Action(ActionKind.LEFT_CLICK))

    quartz.fail_mouse_event = False
    quartz.fail_scroll_event = True
    with pytest.raises(RuntimeError, match="scroll event"):
        controller.apply(Action(ActionKind.SCROLL, dx=1.0, dy=1.0))


def test_controller_refuses_to_start_without_accessibility(monkeypatch):
    with pytest.raises(PermissionError, match="Accessibility"):
        _controller(monkeypatch, trusted=False)
