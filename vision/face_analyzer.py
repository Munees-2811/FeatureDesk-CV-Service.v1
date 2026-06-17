from dataclasses import dataclass, field
from typing import Optional, List
import numpy as np


@dataclass
class FaceResult:
    landmarks: Optional[List] = None
    bbox: Optional[tuple] = None
    confidence: float = 0.0


class FaceAnalyzer:
    """MediaPipe face landmark analysis."""

    def __init__(self) -> None:
        self._mp = None

    def _load(self):
        from loaders.mediapipe_loader import MediaPipeLoader
        self._mp = MediaPipeLoader.face_instance()

    def analyse(self, frame: np.ndarray, roi: tuple) -> Optional[FaceResult]:
        if self._mp is None:
            self._load()
        # Crop ROI and run landmark detection
        x1, y1, x2, y2 = roi
        crop = frame[y1:y2, x1:x2]
        result = self._mp.detect(crop)
        if not result.face_landmarks:
            return None
        return FaceResult(landmarks=result.face_landmarks[0], bbox=roi)
