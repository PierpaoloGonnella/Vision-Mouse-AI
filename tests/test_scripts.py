import pytest

from conftest import SCREEN, make_hand

W, H = SCREEN
FW, FH = 640, 480  # frame width/height used by the fake camera

# Pinch: thumb and index tip at the same point; open: far apart.
PINCH = dict(INDEX_FINGER_TIP=(0.5, 0.5), THUMB_TIP=(0.5, 0.5))
OPEN = dict(INDEX_FINGER_TIP=(0.2, 0.2), THUMB_TIP=(0.8, 0.8))


def kinds(calls):
    return [c[0] for c in calls]


# ---------------------------------------------------------------- common ---

@pytest.mark.parametrize(
    "script", ["click_ai_v1.1.py", "handsMouse_ai.py", "mouse_ai_v1.3.py"]
)
def test_no_hands_no_actions_and_cleanup(harness, script):
    calls = harness.run(script, [[], [], []])
    assert calls == []
    assert harness.released
    assert harness.windows_destroyed


# ----------------------------------------------------------- click_ai_v1.1 ---

def test_click_v11_pinch_clicks_without_moving(harness):
    assert harness.run("click_ai_v1.1.py", [[make_hand(**PINCH)]]) == [("click",)]


def test_click_v11_open_hand_does_nothing(harness):
    assert harness.run("click_ai_v1.1.py", [[make_hand(**OPEN)]]) == []


def test_click_v11_threshold_in_frame_pixels(harness):
    # 49 px apart in a 640 px wide frame -> click; 51 px -> no click
    near = make_hand(INDEX_FINGER_TIP=(0.5, 0.5), THUMB_TIP=(0.5 + 49 / FW, 0.5))
    far = make_hand(INDEX_FINGER_TIP=(0.5, 0.5), THUMB_TIP=(0.5 + 51 / FW, 0.5))
    assert harness.run("click_ai_v1.1.py", [[near], [far]]) == [("click",)]


def test_click_v11_shows_video_and_tracks_single_hand(harness):
    harness.run("click_ai_v1.1.py", [[], []])
    assert harness.imshow_count == 2
    assert harness.hands_kwargs["max_num_hands"] == 1
    assert harness.hands_kwargs["min_detection_confidence"] == 0.5


# ----------------------------------------------------------- handsMouse_ai ---

def test_hands_mouse_right_hand_moves_to_wrist_middle_mcp_midpoint(harness):
    hand = make_hand("Right", WRIST=(0.2, 0.4), MIDDLE_FINGER_MCP=(0.4, 0.6), **OPEN)
    calls = harness.run("handsMouse_ai.py", [[hand]])
    assert calls == [("moveTo", int(0.3 * W), int(0.5 * H))]


def test_hands_mouse_right_pinch_moves_then_clicks(harness):
    calls = harness.run("handsMouse_ai.py", [[make_hand("Right", **PINCH)]])
    assert kinds(calls) == ["moveTo", "click"]


def test_hands_mouse_left_pinch_right_clicks_without_moving(harness):
    calls = harness.run("handsMouse_ai.py", [[make_hand("Left", **PINCH)]])
    assert calls == [("rightClick",)]


def test_hands_mouse_left_open_does_nothing(harness):
    assert harness.run("handsMouse_ai.py", [[make_hand("Left", **OPEN)]]) == []


def test_hands_mouse_two_hands_in_one_frame(harness):
    frame = [make_hand("Right", **OPEN), make_hand("Left", **PINCH)]
    assert kinds(harness.run("handsMouse_ai.py", [frame])) == ["moveTo", "rightClick"]


def test_hands_mouse_threshold_in_screen_pixels(harness):
    near = make_hand("Left", INDEX_FINGER_TIP=(0.5, 0.5), THUMB_TIP=(0.5 + 49 / W, 0.5))
    far = make_hand("Left", INDEX_FINGER_TIP=(0.5, 0.5), THUMB_TIP=(0.5 + 51 / W, 0.5))
    assert harness.run("handsMouse_ai.py", [[near], [far]]) == [("rightClick",)]


# --------------------------------------------------------- mouse_ai_v1.3 ---

def test_mouse_v13_moves_to_average_of_four_landmarks_for_any_hand(harness):
    pts = dict(
        WRIST=(0.1, 0.2),
        MIDDLE_FINGER_MCP=(0.3, 0.4),
        INDEX_FINGER_MCP=(0.5, 0.6),
        RING_FINGER_MCP=(0.7, 0.8),
    )
    calls = harness.run("mouse_ai_v1.3.py", [[make_hand("Left", **pts, **OPEN)]])
    assert calls == [("moveTo", int(0.4 * W), int(0.5 * H))]


def test_mouse_v13_right_pinch_clicks_left_pinch_right_clicks(harness):
    calls = harness.run(
        "mouse_ai_v1.3.py",
        [[make_hand("Right", **PINCH)], [make_hand("Left", **PINCH)]],
    )
    assert kinds(calls) == ["moveTo", "click", "moveTo", "rightClick"]


def test_mouse_v13_tighter_30px_threshold(harness):
    # 40 px would click in the older scripts but not in v1.3
    mid = make_hand("Right", INDEX_FINGER_TIP=(0.5, 0.5), THUMB_TIP=(0.5 + 40 / W, 0.5))
    near = make_hand("Right", INDEX_FINGER_TIP=(0.5, 0.5), THUMB_TIP=(0.5 + 29 / W, 0.5))
    calls = harness.run("mouse_ai_v1.3.py", [[mid], [near]])
    assert kinds(calls) == ["moveTo", "moveTo", "click"]


def test_mouse_v13_headless_by_default(harness):
    harness.run("mouse_ai_v1.3.py", [[make_hand(**OPEN)]])
    assert harness.imshow_count == 0
    assert harness.hands_kwargs["min_detection_confidence"] == 0.8
