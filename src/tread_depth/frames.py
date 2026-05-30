from __future__ import annotations

import cv2
import numpy as np

from .config import Settings


class FrameStore:
    def __init__(self, color: dict[int, np.ndarray], gray: dict[int, np.ndarray],
                 width: int, height: int):
        self.color = color
        self.gray = gray
        self.width = width
        self.height = height

    @property
    def frame_ids(self) -> list[int]:
        return sorted(self.color)

    def intrinsics(self) -> np.ndarray:
        """Pinhole guess: focal ~= image width, principal point at the centre.

        We never recover metric structure, so an approximate focal length is
        fine -- it only affects the recovered *rotation*, which is what the
        goniometer needs, and rotation is insensitive to focal error here.
        """
        f = float(self.width)
        return np.array([[f, 0, self.width / 2],
                         [0, f, self.height / 2],
                         [0, 0, 1.0]])


def load_frames(video_path: str, settings: Settings) -> FrameStore:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"could not open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or settings.frame_stop
    stop = min(settings.frame_stop, total - 1)

    color: dict[int, np.ndarray] = {}
    gray: dict[int, np.ndarray] = {}
    for fid in range(settings.frame_start, stop + 1, settings.frame_step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ok, frame = cap.read()
        if not ok:
            continue
        color[fid] = frame
        gray[fid] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cap.release()

    if not color:
        raise RuntimeError("no frames were read in the requested range")

    h, w = next(iter(gray.values())).shape
    return FrameStore(color, gray, w, h)
