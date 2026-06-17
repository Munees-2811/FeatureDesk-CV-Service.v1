from typing import Optional
import numpy as np


class PoseAnalyzer:
    """MediaPipe pose landmark analysis."""

    def __init__(self) -> None:
        self._mp = None

    def _load(self):
        from loaders.mediapipe_loader import MediaPipeLoader
        self._mp = MediaPipeLoader.pose_instance()

    def analyse(self, frame: np.ndarray, roi: tuple) -> Optional[dict]:
        if self._mp is None:
            self._load()
        x1, y1, x2, y2 = roi
        crop = frame[y1:y2, x1:x2]
        result = self._mp.detect(crop)
        if not result.pose_landmarks:
            return None
        return {"landmarks": result.pose_landmarks[0]}
