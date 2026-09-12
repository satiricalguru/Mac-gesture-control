"""Tests for the per-user hand calibration engine and config persistence."""

from __future__ import annotations

from pathlib import Path

import pytest

from gesture_mac_prototype.calibration import (
    CalibrationEngine,
    CalibrationStep,
)
from gesture_mac_prototype.gesture_engine import GestureConfig, Point


def _create_hand(
    *,
    center_x: float = 0.5,
    center_y: float = 0.5,
    index_pinched: bool = False,
    pinch_offset: float = 0.0,
    palm_scale: float = 0.15,
) -> list[Point]:
    """Synthetic 21-point hand model for deterministic calibration tests."""
    wrist = Point(center_x, center_y + palm_scale)
    mcp_middle = Point(center_x, center_y)

    idx_pip = Point(center_x - 0.05, center_y - 0.1)
    idx_tip = Point(center_x - 0.05, center_y - 0.25)

    if index_pinched:
        thumb_tip = Point(idx_tip.x + pinch_offset, idx_tip.y)
    else:
        thumb_tip = Point(center_x - 0.10, center_y - 0.05)

    mid_pip = Point(center_x, center_y - 0.1)
    mid_tip = Point(center_x, center_y - 0.25)
    rng_pip = Point(center_x + 0.05, center_y - 0.1)
    rng_tip = Point(center_x + 0.05, center_y - 0.25)
    pnk_pip = Point(center_x + 0.1, center_y - 0.05)
    pnk_tip = Point(center_x + 0.1, center_y - 0.2)

    return [
        wrist,
        Point(center_x - 0.04, center_y + 0.1),
        Point(center_x - 0.07, center_y + 0.06),
        Point(center_x - 0.09, center_y + 0.03),
        thumb_tip,
        Point(center_x - 0.05, center_y - 0.01),
        idx_pip,
        Point(center_x - 0.05, (idx_pip.y + idx_tip.y) / 2),
        idx_tip,
        mcp_middle,
        mid_pip,
        Point(center_x, (mid_pip.y + mid_tip.y) / 2),
        mid_tip,
        Point(center_x + 0.05, center_y - 0.01),
        rng_pip,
        Point(center_x + 0.05, (rng_pip.y + rng_tip.y) / 2),
        rng_tip,
        Point(center_x + 0.1, center_y),
        pnk_pip,
        Point(center_x + 0.1, (pnk_pip.y + pnk_tip.y) / 2),
        pnk_tip,
    ]


def test_calibration_full_lifecycle():
    engine = CalibrationEngine()
    assert engine.step == CalibrationStep.WAIT_FOR_HAND
    assert not engine.is_complete
    assert not engine.is_cancelled

    t = 1.0

    # Step 1: REST_HAND
    rest_hand = _create_hand(center_x=0.5, center_y=0.5)
    out = engine.update(rest_hand, t)
    assert out.step == CalibrationStep.REST_HAND
    assert out.hand_detected
    assert out.step_index == 1

    # Feed REST_HAND until duration passes (SAMPLE_DURATION_S = 2.0)
    for _ in range(25):
        t += 0.1
        out = engine.update(rest_hand, t)

    # Step 2: PINCH_CLOSED
    assert out.step == CalibrationStep.PINCH_CLOSED
    assert out.step_index == 2
    closed_hand = _create_hand(index_pinched=True, pinch_offset=0.02)
    for _ in range(25):
        t += 0.1
        out = engine.update(closed_hand, t)

    # Step 3: PINCH_OPEN
    assert out.step == CalibrationStep.PINCH_OPEN
    assert out.step_index == 3
    open_hand = _create_hand(index_pinched=False)
    for _ in range(25):
        t += 0.1
        out = engine.update(open_hand, t)

    # Step 4: ACTIVE_BOUNDS
    assert out.step == CalibrationStep.ACTIVE_BOUNDS
    assert out.step_index == 4

    # Move index tip across corners
    positions = [(0.15, 0.15), (0.85, 0.15), (0.85, 0.85), (0.15, 0.85)]
    for pos_x, pos_y in positions * 10:
        t += 0.1
        reach_hand = _create_hand(center_x=pos_x, center_y=pos_y)
        out = engine.update(reach_hand, t)

    # Step 5: SCROLL_PREFERENCE
    assert out.step == CalibrationStep.SCROLL_PREFERENCE
    assert out.step_index == 5
    assert not engine.base_config.invert_scroll

    # Toggle invert
    new_invert = engine.toggle_invert_scroll()
    assert new_invert is True

    # Finish calibration
    engine.skip_step()
    assert engine.is_complete
    assert engine.step == CalibrationStep.COMPLETE

    complete_out = engine.update(rest_hand, t + 0.1)
    assert complete_out.step == CalibrationStep.COMPLETE
    assert complete_out.progress == 1.0
    assert complete_out.calibrated_config is not None

    cfg = complete_out.calibrated_config
    cfg.validate()
    assert cfg.invert_scroll is True
    assert cfg.pinch_close_ratio < cfg.pinch_open_ratio
    assert cfg.active_left < cfg.active_right
    assert cfg.active_top < cfg.active_bottom


def test_calibration_missing_hand_resets_step_progress():
    engine = CalibrationEngine()
    hand = _create_hand()

    # Detect hand
    out1 = engine.update(hand, 1.0)
    assert out1.hand_detected
    assert out1.step == CalibrationStep.REST_HAND

    # Advance time a bit
    engine.update(hand, 1.8)

    # Hand drops out
    dropout = engine.update(None, 2.0)
    assert not dropout.hand_detected
    assert dropout.progress == 0.0

    # Hand returns: timer should restart from current time
    resumed = engine.update(hand, 3.0)
    assert resumed.hand_detected
    assert resumed.progress == 0.0


def test_calibration_cancel_and_skip():
    engine = CalibrationEngine()
    hand = _create_hand()
    engine.update(hand, 1.0)
    assert engine.step == CalibrationStep.REST_HAND

    # Skip to next step
    engine.skip_step()
    assert engine.step == CalibrationStep.PINCH_CLOSED

    engine.skip_step()
    assert engine.step == CalibrationStep.PINCH_OPEN

    # Cancel
    engine.cancel()
    assert engine.is_cancelled
    out = engine.update(hand, 2.0)
    assert out.step == CalibrationStep.CANCELLED


def test_calibration_rejects_non_monotonic_time():
    engine = CalibrationEngine()
    hand = _create_hand()

    engine.update(hand, 5.0)
    with pytest.raises(ValueError, match="monotonically"):
        engine.update(hand, 4.0)


def test_calibration_rejects_invalid_landmarks():
    engine = CalibrationEngine()

    with pytest.raises(ValueError, match="21 hand landmarks"):
        engine.update([Point(0.0, 0.0)] * 10, 1.0)

    invalid_points = [Point(0.0, 0.0)] * 21
    invalid_points[5] = Point(float("nan"), 0.0)
    with pytest.raises(ValueError, match="finite coordinates"):
        engine.update(invalid_points, 1.0)


def test_gesture_config_serialization_roundtrip(tmp_path: Path):
    cfg = GestureConfig(
        active_left=0.18,
        active_right=0.82,
        active_top=0.15,
        active_bottom=0.80,
        pinch_close_ratio=0.32,
        pinch_open_ratio=0.45,
        invert_scroll=True,
    )

    data = cfg.to_dict()
    assert isinstance(data, dict)
    assert data["active_left"] == 0.18
    assert data["invert_scroll"] is True

    restored = GestureConfig.from_dict(data)
    assert restored == cfg

    # Test file save and load
    file_path = tmp_path / "subdir" / "profile.json"
    cfg.save(file_path)
    assert file_path.exists()

    loaded = GestureConfig.load(file_path)
    assert loaded == cfg


def test_gesture_config_load_error_cases(tmp_path: Path):
    missing = tmp_path / "nonexistent.json"
    with pytest.raises(FileNotFoundError):
        GestureConfig.load(missing)

    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(ValueError, match="malformed profile JSON"):
        GestureConfig.load(corrupt)

    non_dict = tmp_path / "array.json"
    non_dict.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        GestureConfig.load(non_dict)


def test_gesture_config_from_dict_validation():
    with pytest.raises(ValueError, match="dictionary"):
        GestureConfig.from_dict("not a dict")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="unknown configuration parameter"):
        GestureConfig.from_dict({"fake_param": 123})

    with pytest.raises(ValueError, match="invert_scroll must be a boolean"):
        GestureConfig.from_dict({"invert_scroll": "yes"})

    with pytest.raises(ValueError, match="scroll_gain must be a number"):
        GestureConfig.from_dict({"scroll_gain": True})

    with pytest.raises(ValueError, match="pinch thresholds must be positive"):
        GestureConfig.from_dict({"pinch_close_ratio": 0.6, "pinch_open_ratio": 0.4})


def test_calibration_small_pinch_separation_uses_safe_ratio():
    engine = CalibrationEngine()
    engine._pinch_closed_samples = [0.30, 0.31, 0.32]
    engine._pinch_open_samples = [0.33, 0.34, 0.33]  # Very small difference <= 0.05
    cfg = engine._compile_config()
    cfg.validate()
    assert cfg.pinch_close_ratio < cfg.pinch_open_ratio


def test_calibration_insufficient_reach_retains_default_bounds():
    engine = CalibrationEngine()
    # Span is very tiny: only 0.05
    engine._bounds_x = [0.50, 0.51, 0.52] * 5
    engine._bounds_y = [0.50, 0.51, 0.52] * 5
    cfg = engine._compile_config()
    assert cfg.active_left == engine.base_config.active_left
    assert cfg.active_right == engine.base_config.active_right


def test_toggle_invert_scroll_after_completion_updates_cached_config():
    engine = CalibrationEngine()
    engine.step = CalibrationStep.SCROLL_PREFERENCE
    engine.skip_step()
    assert engine.is_complete
    assert engine._calibrated_config is not None
    assert engine._calibrated_config.invert_scroll is False

    engine.toggle_invert_scroll()
    assert engine._calibrated_config.invert_scroll is True
