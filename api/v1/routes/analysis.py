from fastapi import APIRouter, Depends

from api.v1.schemas.requests import FrameAnalysisRequest
from api.v1.schemas.responses import AnalysisResponse
from core.dependencies import (
    get_analysis_service,
    get_session_service,
    verify_api_key,
)
from services.analysis_service import AnalysisService
from services.session_service import SessionService

router = APIRouter(tags=["analysis"])


@router.post("/analysis/frame", response_model=AnalysisResponse)
async def analyse_frame(
    body: FrameAnalysisRequest,
    _: str = Depends(verify_api_key),
    svc: AnalysisService = Depends(get_analysis_service),
    sessions: SessionService = Depends(get_session_service),
) -> AnalysisResponse:
    result = await svc.process_frame(body)
    sessions.record(result)
    return result
