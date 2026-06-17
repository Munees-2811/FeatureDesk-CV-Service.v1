from typing import Optional
from api.v1.schemas.responses import AttentionState

_ABSENT_THRESHOLD = 20
_SLEEPING_THRESHOLD = 35
_DISTRACTED_THRESHOLD = 60


class StateMachine:
    """Per-student state transitions: Focused / Distracted / Sleeping / Absent."""

    def __init__(self) -> None:
        self._states: dict[int, AttentionState] = {}

    def transition(
        self, track_id: int, attention: float, eye: Optional[dict]
    ) -> AttentionState:
        if attention < _ABSENT_THRESHOLD:
            state = AttentionState.ABSENT
        elif eye and (eye["ear_left"] + eye["ear_right"]) / 2 < 0.21:
            state = AttentionState.SLEEPING
        elif attention < _DISTRACTED_THRESHOLD:
            state = AttentionState.DISTRACTED
        else:
            state = AttentionState.FOCUSED
        self._states[track_id] = state
        return state
