from typing import Optional
from vision.face_analyzer import FaceResult
from utils.math_utils import rotation_matrix_to_euler


class HeadPoseEstimator:
    """Estimates yaw / pitch / roll from face landmarks using PnP solving."""

    def estimate(self, face: Optional[FaceResult]) -> Optional[dict]:
        if face is None or face.landmarks is None:
            return None
        # Simplified: return placeholder — replace with cv2.solvePnP call
        return {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}
