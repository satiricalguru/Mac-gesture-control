"""Unit tests for the GestureEngine state machine and classifier."""

from __future__ import annotations

from gesture_mac_prototype.gesture_engine import (
    Action,
    ActionKind,
    Gesture,
    GestureConfig,
    GestureEngine,
    Point,
)


def _create_hand(
    *,
    index_extended: bool = False,
    middle_extended: bool = False,
    ring_extended: bool = False,
    pinky_extended: bool = False,
    index_pinched: bool = False,
    middle_pinched: bool = False,
    center_x: float = 0.5,
    center_y: float = 0.5,
) -> list[Point]:
    """Generates synthetic 21-point hand landmarks."""
    # Palm size ~0.3
    wrist = Point(center_x, center_y + 0.15)
    mcp_middle = Point(center_x, center_y - 0.15)

    # Default thumb tip (4) away from index tip unless pinched
    thumb_tip = Point(center_x - 0.18, center_y)

    # Default fingertips folded (close to wrist/palm)
    idx_pip = Point(center_x - 0.05, center_y - 0.05)
    mid_pip = Point(center_x, center_y - 0.05)
    rng_pip = Point(center_x + 0.05, center_y - 0.05)
    pnk_pip = Point(center_x + 0.1, center_y - 0.05)

    # Extended distance > pip distance * 1.12
    # Folded distance < pip distance
    idx_tip = Point(center_x - 0.05, center_y - 0.25) if index_extended else Point(center_x - 0.02, center_y + 0.05)
    mid_tip = Point(center_x, center_y - 0.25) if middle_extended else Point(center_x, center_y + 0.05)
    rng_tip = Point(center_x + 0.05, center_y - 0.25) if ring_extended else Point(center_x + 0.05, center_y + 0.05)
    pnk_tip = Point(center_x + 0.1, center_y - 0.25) if pinky_extended else Point(center_x + 0.1, center_y + 0.05)

    if index_pinched:
        thumb_tip = Point(center_x - 0.05, center_y - 0.10)
        idx_tip = Point(thumb_tip.x + 0.01, thumb_tip.y + 0.01)

    if middle_pinched:
        thumb_tip = Point(center_x, center_y - 0.10)
        mid_tip = Point(thumb_tip.x + 0.01, thumb_tip.y + 0.01)
        idx_tip = Point(center_x - 0.15, center_y - 0.25)  # index held away

    # 21 points:
    # 0: wrist
    # 1,2,3,4: thumb
    # 5,6,7,8: index (6=pip, 8=tip)
    # 9,10,11,12: middle (9=mcp, 10=pip, 12=tip)
    # 13,14,15,16: ring (14=pip, 16=tip)
    # 17,18,19,20: pinky (18=pip, 20=tip)
    points = [
        wrist,                               # 0
        Point(center_x - 0.04, center_y + 0.1), # 1
        Point(center_x - 0.07, center_y + 0.06), # 2
        Point(center_x - 0.09, center_y + 0.03), # 3
        thumb_tip,                           # 4
        Point(center_x - 0.05, center_y - 0.01), # 5
        idx_pip,                             # 6
        Point(center_x - 0.05, (idx_pip.y + idx_tip.y) / 2), # 7
        idx_tip,                             # 8
        mcp_middle,                          # 9
        mid_pip,                             # 10
        Point(center_x, (mid_pip.y + mid_tip.y) / 2), # 11
        mid_tip,                             # 12
        Point(center_x + 0.05, center_y - 0.01), # 13
        rng_pip,                             # 14
        Point(center_x + 0.05, (rng_pip.y + rng_tip.y) / 2), # 15
        rng_tip,                             # 16
        Point(center_x + 0.1, center_y),     # 17
        pnk_pip,                             # 18
        Point(center_x + 0.1, (pnk_pip.y + pnk_tip.y) / 2), # 19
        pnk_tip,                             # 20
    ]
    return points


def test_neutral_pose_does_nothing():
    engine = GestureEngine()
    # All 5 fingers extended -> NEUTRAL
    open_hand = _create_hand(
        index_extended=True,
        middle_extended=True,
        ring_extended=True,
        pinky_extended=True,
    )
    # Feed frames to stabilize
    t = 1.0
    out = None
    for _ in range(5):
        out = engine.update(open_hand, t)
        t += 0.03

    assert out is not None
    assert out.stable_gesture == Gesture.NEUTRAL
    assert len(out.actions) == 0


def test_move_gesture():
    engine = GestureEngine()
    move_hand = _create_hand(index_extended=True)

    t = 1.0
    actions_accumulated: list[Action] = []
    out = None
    for _ in range(5):
        out = engine.update(move_hand, t)
        actions_accumulated.extend(out.actions)
        t += 0.03

    assert out is not None
    assert out.stable_gesture == Gesture.MOVE
    move_actions = [a for a in actions_accumulated if a.kind == ActionKind.MOVE]
    assert len(move_actions) > 0
    assert 0.0 <= move_actions[-1].x <= 1.0
    assert 0.0 <= move_actions[-1].y <= 1.0


def test_pinch_click():
    engine = GestureEngine()
    pinched = _create_hand(index_pinched=True)
    neutral = _create_hand(index_extended=True, middle_extended=True, ring_extended=True, pinky_extended=True)

    t = 1.0
    # Hold pinch for 0.15s (less than drag_after_s = 0.36s)
    for _ in range(5):
        engine.update(pinched, t)
        t += 0.03

    assert engine.stable_gesture == Gesture.PINCH
    assert not engine.dragging

    # Release to neutral
    click_action_found = False
    for _ in range(5):
        out = engine.update(neutral, t)
        for a in out.actions:
            if a.kind == ActionKind.LEFT_CLICK:
                click_action_found = True
        t += 0.03

    assert click_action_found, "Pinch and quick release must trigger LEFT_CLICK"


def test_hold_to_drag_and_release():
    cfg = GestureConfig(drag_after_s=0.20)
    engine = GestureEngine(cfg)
    pinched = _create_hand(index_pinched=True)
    neutral = _create_hand(index_extended=True, middle_extended=True, ring_extended=True, pinky_extended=True)

    t = 1.0
    # First stabilize pinch
    left_down_found = False
    for _ in range(12):
        out = engine.update(pinched, t)
        for a in out.actions:
            if a.kind == ActionKind.LEFT_DOWN:
                left_down_found = True
        t += 0.03

    assert left_down_found, "Holding pinch past drag_after_s must trigger LEFT_DOWN"
    assert engine.dragging

    # Release pinch -> must trigger LEFT_UP
    left_up_found = False
    for _ in range(5):
        out = engine.update(neutral, t)
        for a in out.actions:
            if a.kind == ActionKind.LEFT_UP:
                left_up_found = True
        t += 0.03

    assert left_up_found, "Releasing drag must trigger LEFT_UP"
    assert not engine.dragging


def test_hand_loss_releases_drag():
    cfg = GestureConfig(drag_after_s=0.15, hand_lost_after_s=0.1)
    engine = GestureEngine(cfg)
    pinched = _create_hand(index_pinched=True)

    t = 1.0
    for _ in range(10):
        engine.update(pinched, t)
        t += 0.03
    assert engine.dragging

    # Suddenly hand is lost (None) past hand_lost_after_s
    t += 0.2
    out = engine.update(None, t)
    left_up = [a for a in out.actions if a.kind == ActionKind.LEFT_UP]
    assert len(left_up) == 1, "Losing hand while dragging must release LEFT_UP"
    assert not engine.dragging
    assert out.stable_gesture == Gesture.NO_HAND


def test_right_pinch():
    engine = GestureEngine()
    right_pinched = _create_hand(middle_pinched=True)

    t = 1.0
    right_clicks = []
    for _ in range(5):
        out = engine.update(right_pinched, t)
        for a in out.actions:
            if a.kind == ActionKind.RIGHT_CLICK:
                right_clicks.append(a)
        t += 0.03

    assert engine.stable_gesture == Gesture.RIGHT_PINCH
    assert len(right_clicks) == 1, "Right pinch must trigger exactly one RIGHT_CLICK"


def test_two_finger_scroll():
    engine = GestureEngine()
    # Scroll: index & middle extended, ring & pinky folded
    scroll_1 = _create_hand(index_extended=True, middle_extended=True, center_y=0.5)
    scroll_2 = _create_hand(index_extended=True, middle_extended=True, center_y=0.45) # moved up

    t = 1.0
    # Stabilize
    for _ in range(4):
        engine.update(scroll_1, t)
        t += 0.03

    assert engine.stable_gesture == Gesture.SCROLL

    # Move hand vertically
    scroll_actions = []
    for _ in range(3):
        out = engine.update(scroll_2, t)
        for a in out.actions:
            if a.kind == ActionKind.SCROLL:
                scroll_actions.append(a)
        t += 0.03

    assert len(scroll_actions) > 0, "Hand displacement during SCROLL must generate SCROLL actions"
    assert scroll_actions[0].dy != 0.0


def test_fist_hold_toggles_pause():
    cfg = GestureConfig(fist_toggle_after_s=0.2, fist_toggle_cooldown_s=0.5)
    engine = GestureEngine(cfg)
    fist = _create_hand()  # all fingers folded

    assert engine.enabled is True

    t = 1.0
    # Hold fist for > 0.3s
    toggle_actions = []
    for _ in range(12):
        out = engine.update(fist, t)
        for a in out.actions:
            if a.kind == ActionKind.ENABLED_CHANGED:
                toggle_actions.append(a)
        t += 0.03

    assert engine.enabled is False
    assert len(toggle_actions) == 1
    assert toggle_actions[0].enabled is False


def test_boundary_clamping():
    cfg = GestureConfig(active_left=0.2, active_right=0.8, active_top=0.2, active_bottom=0.8)
    engine = GestureEngine(cfg)

    # Hand far left beyond active_left (x = 0.05)
    far_left_hand = _create_hand(index_extended=True, center_x=0.05, center_y=0.5)
    t = 1.0
    out = None
    for _ in range(5):
        out = engine.update(far_left_hand, t)
        t += 0.03

    assert out is not None
    move_actions = [a for a in out.actions if a.kind == ActionKind.MOVE]
    assert len(move_actions) > 0
    assert move_actions[-1].x == 0.0  # clamped to 0.0

    # Hand far right beyond active_right (x = 0.95)
    engine_right = GestureEngine(cfg)
    far_right_hand = _create_hand(index_extended=True, center_x=0.95, center_y=0.5)
    t = 1.0
    for _ in range(15):
        out = engine_right.update(far_right_hand, t)
        t += 0.03

    move_actions = [a for a in out.actions if a.kind == ActionKind.MOVE]
    assert len(move_actions) > 0
    assert abs(move_actions[-1].x - 1.0) < 0.001  # converges to 1.0


def test_invert_scroll():
    normal_engine = GestureEngine(GestureConfig(invert_scroll=False))
    inverted_engine = GestureEngine(GestureConfig(invert_scroll=True))

    scroll_start = _create_hand(index_extended=True, middle_extended=True, center_y=0.5)
    scroll_down = _create_hand(index_extended=True, middle_extended=True, center_y=0.55)

    t = 1.0
    for _ in range(5):
        normal_engine.update(scroll_start, t)
        inverted_engine.update(scroll_start, t)
        t += 0.03

    normal_out = normal_engine.update(scroll_down, t)
    inverted_out = inverted_engine.update(scroll_down, t)

    normal_scroll = next(a for a in normal_out.actions if a.kind == ActionKind.SCROLL)
    inverted_scroll = next(a for a in inverted_out.actions if a.kind == ActionKind.SCROLL)

    assert normal_scroll.dy == -inverted_scroll.dy

