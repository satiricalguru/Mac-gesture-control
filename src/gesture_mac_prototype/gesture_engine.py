"""I/O-free gesture classification and transition logic.

This is the portable part of the prototype. It accepts normalized 21-point hand
landmarks and returns semantic actions. It never touches the camera or macOS.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from math import hypot, pi


@dataclass(frozen=True)
class Point:
    x: float
    y: float
    z: float = 0.0


class Gesture(str, Enum):
    NO_HAND = "no hand"
    NEUTRAL = "neutral"
    MOVE = "move"
    PINCH = "pinch / drag"
    RIGHT_PINCH = "right click"
    SCROLL = "scroll"
    FIST = "pause fist"


class ActionKind(str, Enum):
    MOVE = "move"
    LEFT_CLICK = "left_click"
    LEFT_DOWN = "left_down"
    LEFT_UP = "left_up"
    RIGHT_CLICK = "right_click"
    SCROLL = "scroll"
    ENABLED_CHANGED = "enabled_changed"


@dataclass(frozen=True)
class Action:
    kind: ActionKind
    x: float = 0.0
    y: float = 0.0
    dx: float = 0.0
    dy: float = 0.0
    enabled: bool | None = None


@dataclass(frozen=True)
class EngineOutput:
    raw_gesture: Gesture
    stable_gesture: Gesture
    enabled: bool
    dragging: bool
    pinch_ratio: float | None
    actions: tuple[Action, ...]


@dataclass
class GestureConfig:
    active_left: float = 0.14
    active_right: float = 0.86
    active_top: float = 0.12
    active_bottom: float = 0.84
    pinch_close_ratio: float = 0.36
    pinch_open_ratio: float = 0.48
    right_pinch_ratio: float = 0.34
    finger_extension_ratio: float = 1.12
    stable_for_s: float = 0.075
    drag_after_s: float = 0.36
    fist_toggle_after_s: float = 0.80
    fist_toggle_cooldown_s: float = 1.20
    hand_lost_after_s: float = 0.22
    scroll_deadzone: float = 0.003
    scroll_gain: float = 1450.0
    invert_scroll: bool = False
    smoothing_min_cutoff: float = 1.25
    smoothing_beta: float = 0.055


def _distance(a: Point, b: Point) -> float:
    return hypot(a.x - b.x, a.y - b.y)


def _mean(points: Sequence[Point]) -> Point:
    count = len(points)
    return Point(
        sum(p.x for p in points) / count,
        sum(p.y for p in points) / count,
        sum(p.z for p in points) / count,
    )


class _LowPass:
    def __init__(self) -> None:
        self.value: float | None = None

    def apply(self, value: float, alpha: float) -> float:
        if self.value is None:
            self.value = value
        else:
            self.value = alpha * value + (1.0 - alpha) * self.value
        return self.value

    def reset(self) -> None:
        self.value = None


class _OneEuro:
    """Adaptive low-pass filter: steady hands are smooth, fast motion stays responsive."""

    def __init__(self, min_cutoff: float, beta: float, d_cutoff: float = 1.0) -> None:
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.signal = _LowPass()
        self.derivative = _LowPass()
        self.last_raw: float | None = None
        self.last_t: float | None = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def apply(self, value: float, now_s: float) -> float:
        if self.last_t is None or self.last_raw is None:
            self.last_t = now_s
            self.last_raw = value
            return self.signal.apply(value, 1.0)

        dt = max(1.0 / 240.0, min(0.25, now_s - self.last_t))
        raw_derivative = (value - self.last_raw) / dt
        smooth_derivative = self.derivative.apply(
            raw_derivative, self._alpha(self.d_cutoff, dt)
        )
        cutoff = self.min_cutoff + self.beta * abs(smooth_derivative)
        result = self.signal.apply(value, self._alpha(cutoff, dt))
        self.last_t = now_s
        self.last_raw = value
        return result

    def reset(self) -> None:
        self.signal.reset()
        self.derivative.reset()
        self.last_raw = None
        self.last_t = None


class GestureEngine:
    def __init__(self, config: GestureConfig | None = None) -> None:
        self.config = config or GestureConfig()
        self.enabled = True
        self.dragging = False
        self._stable = Gesture.NO_HAND
        self._candidate = Gesture.NO_HAND
        self._candidate_since = 0.0
        self._stable_since = 0.0
        self._last_seen_s: float | None = None
        self._pinch_latched = False
        self._pinch_started_s: float | None = None
        self._fist_fired = False
        self._last_fist_toggle_s = -100.0
        self._scroll_anchor: Point | None = None
        self._filter_x = _OneEuro(
            self.config.smoothing_min_cutoff, self.config.smoothing_beta
        )
        self._filter_y = _OneEuro(
            self.config.smoothing_min_cutoff, self.config.smoothing_beta
        )

    @property
    def stable_gesture(self) -> Gesture:
        return self._stable

    def set_enabled(self, enabled: bool) -> tuple[Action, ...]:
        actions: list[Action] = []
        if self.dragging:
            actions.append(Action(ActionKind.LEFT_UP))
            self.dragging = False
        if self.enabled != enabled:
            self.enabled = enabled
            actions.append(Action(ActionKind.ENABLED_CHANGED, enabled=enabled))
        self._pinch_started_s = None
        self._scroll_anchor = None
        return tuple(actions)

    def toggle_enabled(self) -> tuple[Action, ...]:
        return self.set_enabled(not self.enabled)

    def _finger_extended(self, points: Sequence[Point], tip: int, pip: int) -> bool:
        wrist = points[0]
        return _distance(points[tip], wrist) > (
            _distance(points[pip], wrist) * self.config.finger_extension_ratio
        )

    def _classify(self, points: Sequence[Point]) -> tuple[Gesture, float]:
        palm_size = max(_distance(points[0], points[9]), 0.04)
        index_pinch = _distance(points[4], points[8]) / palm_size
        middle_pinch = _distance(points[4], points[12]) / palm_size

        if self._pinch_latched:
            self._pinch_latched = index_pinch < self.config.pinch_open_ratio
        else:
            self._pinch_latched = index_pinch < self.config.pinch_close_ratio

        if self._pinch_latched:
            return Gesture.PINCH, index_pinch
        if middle_pinch < self.config.right_pinch_ratio:
            return Gesture.RIGHT_PINCH, index_pinch

        index = self._finger_extended(points, 8, 6)
        middle = self._finger_extended(points, 12, 10)
        ring = self._finger_extended(points, 16, 14)
        pinky = self._finger_extended(points, 20, 18)

        if not any((index, middle, ring, pinky)):
            return Gesture.FIST, index_pinch
        if index and middle and not ring and not pinky:
            return Gesture.SCROLL, index_pinch
        if index and not middle and not ring and not pinky:
            return Gesture.MOVE, index_pinch
        return Gesture.NEUTRAL, index_pinch

    def _pointer(self, point: Point, now_s: float) -> Point:
        cfg = self.config
        x = (point.x - cfg.active_left) / (cfg.active_right - cfg.active_left)
        y = (point.y - cfg.active_top) / (cfg.active_bottom - cfg.active_top)
        x = max(0.0, min(1.0, x))
        y = max(0.0, min(1.0, y))
        return Point(self._filter_x.apply(x, now_s), self._filter_y.apply(y, now_s))

    @staticmethod
    def _scroll_point(points: Sequence[Point]) -> Point:
        return _mean([points[i] for i in (0, 5, 9, 13, 17)])

    def _stabilize(self, raw: Gesture, now_s: float) -> tuple[Gesture, Gesture]:
        previous = self._stable
        if raw != self._candidate:
            self._candidate = raw
            self._candidate_since = now_s
        elif (
            raw != self._stable
            and now_s - self._candidate_since >= self.config.stable_for_s
        ):
            self._stable = raw
            self._stable_since = now_s
            self._scroll_anchor = None
            if raw != Gesture.FIST:
                self._fist_fired = False
        return previous, self._stable

    def update(self, landmarks: Sequence[Point] | None, now_s: float) -> EngineOutput:
        actions: list[Action] = []

        if landmarks is None:
            if (
                self._last_seen_s is not None
                and now_s - self._last_seen_s < self.config.hand_lost_after_s
            ):
                return EngineOutput(
                    Gesture.NO_HAND,
                    self._stable,
                    self.enabled,
                    self.dragging,
                    None,
                    (),
                )
            if self.dragging:
                actions.append(Action(ActionKind.LEFT_UP))
                self.dragging = False
            self._stable = Gesture.NO_HAND
            self._candidate = Gesture.NO_HAND
            self._candidate_since = now_s
            self._pinch_started_s = None
            self._scroll_anchor = None
            self._pinch_latched = False
            self._filter_x.reset()
            self._filter_y.reset()
            return EngineOutput(
                Gesture.NO_HAND,
                self._stable,
                self.enabled,
                self.dragging,
                None,
                tuple(actions),
            )

        if len(landmarks) != 21:
            raise ValueError(f"expected 21 hand landmarks, got {len(landmarks)}")

        self._last_seen_s = now_s
        raw, pinch_ratio = self._classify(landmarks)
        previous, stable = self._stabilize(raw, now_s)

        if previous == Gesture.PINCH and stable != Gesture.PINCH:
            if self.dragging:
                actions.append(Action(ActionKind.LEFT_UP))
                self.dragging = False
            elif self.enabled and self._pinch_started_s is not None:
                actions.append(Action(ActionKind.LEFT_CLICK))
            self._pinch_started_s = None

        if stable == Gesture.FIST:
            held_for = now_s - self._stable_since
            cooldown_ok = (
                now_s - self._last_fist_toggle_s >= self.config.fist_toggle_cooldown_s
            )
            if (
                held_for >= self.config.fist_toggle_after_s
                and not self._fist_fired
                and cooldown_ok
            ):
                actions.extend(self.toggle_enabled())
                self._fist_fired = True
                self._last_fist_toggle_s = now_s
        elif stable != Gesture.FIST:
            self._fist_fired = False

        if self.enabled:
            if stable == Gesture.RIGHT_PINCH and previous != Gesture.RIGHT_PINCH:
                actions.append(Action(ActionKind.RIGHT_CLICK))

            if stable == Gesture.PINCH:
                if previous != Gesture.PINCH or self._pinch_started_s is None:
                    self._pinch_started_s = now_s
                pointer = self._pointer(landmarks[8], now_s)
                actions.insert(0, Action(ActionKind.MOVE, x=pointer.x, y=pointer.y))
                if (
                    not self.dragging
                    and now_s - self._pinch_started_s >= self.config.drag_after_s
                ):
                    actions.append(Action(ActionKind.LEFT_DOWN))
                    self.dragging = True
            elif stable == Gesture.MOVE:
                pointer = self._pointer(landmarks[8], now_s)
                actions.append(Action(ActionKind.MOVE, x=pointer.x, y=pointer.y))
            elif stable == Gesture.SCROLL:
                current = self._scroll_point(landmarks)
                if self._scroll_anchor is not None:
                    dx = current.x - self._scroll_anchor.x
                    dy = current.y - self._scroll_anchor.y
                    if abs(dx) < self.config.scroll_deadzone:
                        dx = 0.0
                    if abs(dy) < self.config.scroll_deadzone:
                        dy = 0.0
                    if dx or dy:
                        direction = -1.0 if self.config.invert_scroll else 1.0
                        actions.append(
                            Action(
                                ActionKind.SCROLL,
                                dx=direction * dx * self.config.scroll_gain,
                                dy=direction * dy * self.config.scroll_gain,
                            )
                        )
                self._scroll_anchor = current
            else:
                self._scroll_anchor = None

        return EngineOutput(
            raw,
            stable,
            self.enabled,
            self.dragging,
            pinch_ratio,
            tuple(actions),
        )
