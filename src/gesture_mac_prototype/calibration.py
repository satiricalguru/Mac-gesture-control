"""Deterministic calibration engine for per-user Gesture Mac tuning.

Accepts normalized 21-point hand landmarks and timestamps, guiding the
user through measuring palm scale, pinch hysteresis, and active reach boundaries.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from math import hypot, isfinite
from statistics import median

from .gesture_engine import GestureConfig, Point


class CalibrationStep(StrEnum):
    WAIT_FOR_HAND = "wait_for_hand"
    REST_HAND = "rest_hand"
    PINCH_CLOSED = "pinch_closed"
    PINCH_OPEN = "pinch_open"
    ACTIVE_BOUNDS = "active_bounds"
    SCROLL_PREFERENCE = "scroll_preference"
    COMPLETE = "complete"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class CalibrationOutput:
    step: CalibrationStep
    instruction: str
    progress: float
    step_index: int
    total_steps: int
    hand_detected: bool
    live_metric: float | None = None
    metric_label: str | None = None
    calibrated_config: GestureConfig | None = None


def _distance(a: Point, b: Point) -> float:
    return hypot(a.x - b.x, a.y - b.y)


class CalibrationEngine:
    """State machine guiding a user through calibrating their personal hand profile."""

    TOTAL_STEPS = 5
    SAMPLE_DURATION_S = 2.0

    def __init__(self, base_config: GestureConfig | None = None) -> None:
        self.base_config = base_config or GestureConfig()
        self.step = CalibrationStep.WAIT_FOR_HAND
        self._step_start_s: float | None = None
        self._last_update_s: float | None = None

        # Sample storage
        self._palm_samples: list[float] = []
        self._pinch_closed_samples: list[float] = []
        self._pinch_open_samples: list[float] = []
        self._bounds_x: list[float] = []
        self._bounds_y: list[float] = []
        self._invert_scroll = self.base_config.invert_scroll

        # Calibrated output cache
        self._calibrated_config: GestureConfig | None = None

    @property
    def is_complete(self) -> bool:
        return self.step == CalibrationStep.COMPLETE

    @property
    def is_cancelled(self) -> bool:
        return self.step == CalibrationStep.CANCELLED

    def toggle_invert_scroll(self) -> bool:
        self._invert_scroll = not self._invert_scroll
        if self._calibrated_config is not None:
            self._calibrated_config = self._compile_config()
        return self._invert_scroll

    def cancel(self) -> None:
        self.step = CalibrationStep.CANCELLED

    def skip_step(self) -> None:
        """Manually advance to the next step, using fallback data if uncollected."""
        self._advance_to_next_step()

    def _palm_size(self, points: Sequence[Point]) -> float:
        return max(_distance(points[0], points[9]), 0.04)

    def _pinch_ratio(self, points: Sequence[Point], palm_size: float) -> float:
        return _distance(points[4], points[8]) / palm_size

    def _advance_to_next_step(self) -> None:
        self._step_start_s = None
        if self.step in (CalibrationStep.WAIT_FOR_HAND, CalibrationStep.REST_HAND):
            self.step = CalibrationStep.PINCH_CLOSED
        elif self.step == CalibrationStep.PINCH_CLOSED:
            self.step = CalibrationStep.PINCH_OPEN
        elif self.step == CalibrationStep.PINCH_OPEN:
            self.step = CalibrationStep.ACTIVE_BOUNDS
        elif self.step == CalibrationStep.ACTIVE_BOUNDS:
            self.step = CalibrationStep.SCROLL_PREFERENCE
        elif self.step == CalibrationStep.SCROLL_PREFERENCE:
            self._calibrated_config = self._compile_config()
            self.step = CalibrationStep.COMPLETE

    def _compile_config(self) -> GestureConfig:
        cfg = self.base_config

        # 1. Pinch ratios with guaranteed hysteresis margin
        close_ratio = cfg.pinch_close_ratio
        open_ratio = cfg.pinch_open_ratio
        if self._pinch_closed_samples and self._pinch_open_samples:
            med_closed = median(self._pinch_closed_samples)
            med_open = median(self._pinch_open_samples)
            if med_open > med_closed + 0.05:
                # Place close threshold slightly above closed pinch, open slightly below open pinch
                delta = med_open - med_closed
                close_ratio = round(med_closed + 0.35 * delta, 3)
                open_ratio = round(med_closed + 0.70 * delta, 3)
            elif med_closed < open_ratio:
                close_ratio = round(min(med_closed * 1.25, 0.40), 3)
                open_ratio = round(max(close_ratio + 0.08, cfg.pinch_open_ratio), 3)

        # 2. Active bounds
        active_l = cfg.active_left
        active_r = cfg.active_right
        active_t = cfg.active_top
        active_b = cfg.active_bottom
        if len(self._bounds_x) >= 10 and len(self._bounds_y) >= 10:
            min_x, max_x = min(self._bounds_x), max(self._bounds_x)
            min_y, max_y = min(self._bounds_y), max(self._bounds_y)
            # Ensure sufficient reach range was recorded (at least 0.20 span)
            if (max_x - min_x) >= 0.20 and (max_y - min_y) >= 0.20:
                # Add 5% comfortable margin
                margin_x = 0.05 * (max_x - min_x)
                margin_y = 0.05 * (max_y - min_y)
                active_l = round(max(0.02, min_x - margin_x), 2)
                active_r = round(min(0.98, max_x + margin_x), 2)
                active_t = round(max(0.02, min_y - margin_y), 2)
                active_b = round(min(0.98, max_y + margin_y), 2)

        # Safe sanity check on active area range
        if active_r <= active_l + 0.15:
            active_l, active_r = cfg.active_left, cfg.active_right
        if active_b <= active_t + 0.15:
            active_t, active_b = cfg.active_top, cfg.active_bottom

        # Safe sanity check on pinch thresholds
        if open_ratio <= close_ratio + 0.04:
            close_ratio = cfg.pinch_close_ratio
            open_ratio = cfg.pinch_open_ratio

        return GestureConfig(
            active_left=active_l,
            active_right=active_r,
            active_top=active_t,
            active_bottom=active_b,
            pinch_close_ratio=close_ratio,
            pinch_open_ratio=open_ratio,
            right_pinch_ratio=cfg.right_pinch_ratio,
            finger_extension_ratio=cfg.finger_extension_ratio,
            stable_for_s=cfg.stable_for_s,
            drag_after_s=cfg.drag_after_s,
            fist_toggle_after_s=cfg.fist_toggle_after_s,
            fist_toggle_cooldown_s=cfg.fist_toggle_cooldown_s,
            hand_lost_after_s=cfg.hand_lost_after_s,
            scroll_deadzone=cfg.scroll_deadzone,
            scroll_gain=cfg.scroll_gain,
            invert_scroll=self._invert_scroll,
            smoothing_min_cutoff=cfg.smoothing_min_cutoff,
            smoothing_beta=cfg.smoothing_beta,
        )

    def update(
        self, landmarks: Sequence[Point] | None, now_s: float
    ) -> CalibrationOutput:
        if not isfinite(now_s):
            raise ValueError("now_s must be finite")
        if self._last_update_s is not None and now_s < self._last_update_s:
            raise ValueError("now_s must increase monotonically")
        self._last_update_s = now_s

        if self.step == CalibrationStep.COMPLETE:
            return CalibrationOutput(
                step=CalibrationStep.COMPLETE,
                instruction="Calibration complete! Press Q or ENTER to save and finish.",
                progress=1.0,
                step_index=5,
                total_steps=self.TOTAL_STEPS,
                hand_detected=landmarks is not None,
                calibrated_config=self._calibrated_config,
            )

        if self.step == CalibrationStep.CANCELLED:
            return CalibrationOutput(
                step=CalibrationStep.CANCELLED,
                instruction="Calibration cancelled.",
                progress=0.0,
                step_index=0,
                total_steps=self.TOTAL_STEPS,
                hand_detected=False,
            )

        if landmarks is None:
            # Hand missing: pause progress timer
            self._step_start_s = None
            return CalibrationOutput(
                step=self.step,
                instruction="Place your hand clearly in front of the camera to continue.",
                progress=0.0,
                step_index=self._current_step_index(),
                total_steps=self.TOTAL_STEPS,
                hand_detected=False,
            )

        if len(landmarks) != 21:
            raise ValueError(f"expected 21 hand landmarks, got {len(landmarks)}")
        if any(not isfinite(c) for p in landmarks for c in (p.x, p.y, p.z)):
            raise ValueError("hand landmarks must contain only finite coordinates")

        if self.step == CalibrationStep.WAIT_FOR_HAND:
            self.step = CalibrationStep.REST_HAND

        palm_size = self._palm_size(landmarks)
        pinch_ratio = self._pinch_ratio(landmarks, palm_size)

        if self._step_start_s is None:
            self._step_start_s = now_s

        elapsed = max(0.0, now_s - self._step_start_s)
        progress = min(1.0, elapsed / self.SAMPLE_DURATION_S)

        metric: float | None = None
        metric_lbl: str | None = None

        if self.step == CalibrationStep.REST_HAND:
            self._palm_samples.append(palm_size)
            metric = palm_size
            metric_lbl = "Palm Scale"
            instruction = "Hold your hand open in a relaxed, resting position."
            if progress >= 1.0:
                self._advance_to_next_step()

        elif self.step == CalibrationStep.PINCH_CLOSED:
            self._pinch_closed_samples.append(pinch_ratio)
            metric = pinch_ratio
            metric_lbl = "Pinch Distance"
            instruction = "Pinch thumb and index finger together firmly (click pose)."
            if progress >= 1.0:
                self._advance_to_next_step()

        elif self.step == CalibrationStep.PINCH_OPEN:
            self._pinch_open_samples.append(pinch_ratio)
            metric = pinch_ratio
            metric_lbl = "Pinch Distance"
            instruction = "Release your fingers into a relaxed open pinch."
            if progress >= 1.0:
                self._advance_to_next_step()

        elif self.step == CalibrationStep.ACTIVE_BOUNDS:
            idx_tip = landmarks[8]
            self._bounds_x.append(idx_tip.x)
            self._bounds_y.append(idx_tip.y)
            metric = float(len(self._bounds_x))
            metric_lbl = "Points Sampled"
            instruction = "Move index finger across your comfortable screen reach area."
            # Allow 3.5s for reaching corners
            progress = min(1.0, elapsed / 3.5)
            if progress >= 1.0:
                self._advance_to_next_step()

        elif self.step == CalibrationStep.SCROLL_PREFERENCE:
            direction_str = "INVERTED (Natural)" if self._invert_scroll else "STANDARD"
            instruction = f"Scroll direction: {direction_str}. Press SPACE to toggle, ENTER to finish."
            metric = 1.0 if self._invert_scroll else 0.0
            metric_lbl = "Invert Active"
            progress = 1.0

        return CalibrationOutput(
            step=self.step,
            instruction=instruction,
            progress=progress,
            step_index=self._current_step_index(),
            total_steps=self.TOTAL_STEPS,
            hand_detected=True,
            live_metric=metric,
            metric_label=metric_lbl,
            calibrated_config=self._calibrated_config,
        )

    def _current_step_index(self) -> int:
        mapping = {
            CalibrationStep.WAIT_FOR_HAND: 1,
            CalibrationStep.REST_HAND: 1,
            CalibrationStep.PINCH_CLOSED: 2,
            CalibrationStep.PINCH_OPEN: 3,
            CalibrationStep.ACTIVE_BOUNDS: 4,
            CalibrationStep.SCROLL_PREFERENCE: 5,
            CalibrationStep.COMPLETE: 5,
            CalibrationStep.CANCELLED: 0,
        }
        return mapping.get(self.step, 1)
