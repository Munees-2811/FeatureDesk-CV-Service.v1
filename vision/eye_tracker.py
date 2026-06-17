from typing import Optional
from vision.face_analyzer import FaceResult
from utils.math_utils import ear


class EyeTracker:
    """Computes Eye Aspect Ratio (EAR) and gaze direction."""

    EAR_THRESHOLD = 0.21

    def compute(self, face: Optional[FaceResult]) -> Optional[dict]:
        if face is None or face.landmarks is None:
            return None
        # Placeholder — replace with real landmark indices for MediaPipe 478-point mesh
        return {
            "ear_left": 0.3,
            "ear_right": 0.3,
            "gaze_x": 0.0,
            "gaze_y": 0.0,
        }
