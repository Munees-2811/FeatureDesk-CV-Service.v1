import time
import uuid
from typing import Optional
from api.v1.schemas.responses import SessionResponse
from core.exceptions import SessionNotFoundError


class SessionService:
    def __init__(self) -> None:
        self._store: dict[str, dict] = {}

    def create(self, student_id: str, metadata: Optional[dict] = None) -> SessionResponse:
        session_id = str(uuid.uuid4())
        now = int(time.time() * 1000)
        self._store[session_id] = {
            "student_id": student_id,
            "created_at": now,
            "frame_count": 0,
            "metadata": metadata or {},
        }
        return self._to_response(session_id)

    def get(self, session_id: str) -> SessionResponse:
        if session_id not in self._store:
            raise SessionNotFoundError(session_id)
        return self._to_response(session_id)

    def increment_frame(self, session_id: str) -> None:
        if session_id in self._store:
            self._store[session_id]["frame_count"] += 1

    def _to_response(self, session_id: str) -> SessionResponse:
        data = self._store[session_id]
        return SessionResponse(
            session_id=session_id,
            student_id=data["student_id"],
            created_at=data["created_at"],
            frame_count=data["frame_count"],
        )
