from typing import Optional


class AttentionScorer:
    """Produces an attention score 0–100 from head pose and eye metrics."""

    def score(self, head: Optional[dict], eye: Optional[dict]) -> float:
        if head is None and eye is None:
            return 0.0
        score = 100.0
        if head:
            yaw_penalty = min(abs(head["yaw"]) / 45.0, 1.0) * 30
            pitch_penalty = min(abs(head["pitch"]) / 30.0, 1.0) * 20
            score -= yaw_penalty + pitch_penalty
        if eye:
            avg_ear = (eye["ear_left"] + eye["ear_right"]) / 2
            if avg_ear < 0.21:
                score -= 40
        return max(0.0, min(100.0, round(score, 2)))
