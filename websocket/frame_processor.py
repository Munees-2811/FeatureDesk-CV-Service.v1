import base64
from services.analysis_service import AnalysisService
from api.v1.schemas.requests import FrameAnalysisRequest

_svc = AnalysisService()


class FrameProcessor:
    """Receives raw bytes from WebSocket, wraps as FrameAnalysisRequest, runs pipeline."""

    async def process(self, session_id: str, raw_bytes: bytes) -> dict:
        b64 = base64.b64encode(raw_bytes).decode()
        req = FrameAnalysisRequest(session_id=session_id, frame_b64=b64)
        result = await _svc.process_frame(req)
        return result.model_dump()
