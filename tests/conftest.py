"""Test harness: runs the Vision-Mouse-AI scripts against fake cv2/mediapipe/pyautogui.

The original scripts execute their capture loop at import time, so each test
injects stub modules into sys.modules, feeds a scripted sequence of hand
detections, runs the script with runpy and records the mouse actions issued.
No camera, display or real mouse is needed.
"""
import runpy
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCREEN = (1920, 1080)
FRAME_SHAPE = (480, 640, 3)  # height, width, channels

LANDMARK_NAMES = [
    "WRIST",
    "THUMB_TIP",
    "INDEX_FINGER_MCP",
    "INDEX_FINGER_TIP",
    "MIDDLE_FINGER_MCP",
    "RING_FINGER_MCP",
]


def make_hand(label="Right", **points):
    """Build a fake detected hand. Unspecified landmarks default to (0.5, 0.5)."""
    landmark = {name: SimpleNamespace(x=0.5, y=0.5) for name in LANDMARK_NAMES}
    for name, (x, y) in points.items():
        landmark[name] = SimpleNamespace(x=x, y=y)
    hand_landmarks = SimpleNamespace(landmark=landmark)
    handedness = SimpleNamespace(classification=[SimpleNamespace(label=label)])
    return hand_landmarks, handedness


class Harness:
    def __init__(self):
        self.calls = []
        self.frames = []  # one list of hands per frame
        self.hands_kwargs = None
        self.released = False
        self.windows_destroyed = False
        self.imshow_count = 0
        self._current = []

    def _build_modules(self):
        h = self
        frame = SimpleNamespace(shape=FRAME_SHAPE)

        class FakeCapture:
            def __init__(self, index):
                self._frames = iter(h.frames)

            def read(self):
                try:
                    h._current = next(self._frames)
                except StopIteration:
                    return False, None
                return True, frame

            def release(self):
                h.released = True

        def imshow(*args):
            h.imshow_count += 1

        def destroy_all():
            h.windows_destroyed = True

        cv2 = types.ModuleType("cv2")
        cv2.VideoCapture = FakeCapture
        cv2.flip = lambda f, code: f
        cv2.cvtColor = lambda f, code: f
        cv2.COLOR_BGR2RGB = 4
        cv2.imshow = imshow
        cv2.waitKey = lambda delay: -1
        cv2.destroyAllWindows = destroy_all

        class FakeHands:
            def __init__(self, **kwargs):
                h.hands_kwargs = kwargs

            def process(self, image):
                hands = h._current
                if not hands:
                    return SimpleNamespace(multi_hand_landmarks=None, multi_handedness=None)
                return SimpleNamespace(
                    multi_hand_landmarks=[lm for lm, _ in hands],
                    multi_handedness=[hd for _, hd in hands],
                )

        hands_mod = SimpleNamespace(
            Hands=FakeHands,
            HandLandmark=SimpleNamespace(**{n: n for n in LANDMARK_NAMES}),
            HAND_CONNECTIONS=(),
        )
        mp = types.ModuleType("mediapipe")
        mp.solutions = SimpleNamespace(
            hands=hands_mod,
            drawing_utils=SimpleNamespace(draw_landmarks=lambda *a, **k: None),
        )

        pg = types.ModuleType("pyautogui")
        pg.size = lambda: SCREEN
        pg.click = lambda *a, **k: h.calls.append(("click",))
        pg.rightClick = lambda *a, **k: h.calls.append(("rightClick",))
        pg.moveTo = lambda x, y, **k: h.calls.append(("moveTo", x, y))
        pg.sleep = lambda s: None
        return {"cv2": cv2, "mediapipe": mp, "pyautogui": pg}

    def run(self, script, frames):
        self.frames = frames
        modules = self._build_modules()
        saved = {k: sys.modules.get(k) for k in modules}
        sys.modules.update(modules)
        try:
            runpy.run_path(str(REPO_ROOT / script), run_name="__main__")
        finally:
            for k, v in saved.items():
                if v is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = v
        return self.calls


@pytest.fixture
def harness():
    return Harness()
