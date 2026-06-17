from typing import Optional


class FlagDetector:
    """Detects boolean behavioral flags from eye, object, and pose data."""

    EAR_CLOSED = 0.21

    def detect(
        self,
        eye: Optional[dict],
        objects: dict,
        pose: Optional[dict],
    ) -> dict:
        eyes_closed = False
        if eye:
            avg_ear = (eye["ear_left"] + eye["ear_right"]) / 2
            eyes_closed = avg_ear < self.EAR_CLOSED
        return {
            "phone_detected": objects.get("phone", False),
            "left_desk": pose is None,
            "eyes_closed": eyes_closed,
            "looking_away": False,
        }
