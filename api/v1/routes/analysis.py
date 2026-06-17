import time
from fastapi import APIRouter, Depends
from api.v1.schemas.requests import FrameAnalysisRequest
from api.v1.schemas.responses import AnalysisResponse
from core.dependencies import verify_api_key, get_analysis_service
from services.analysis_service import AnalysisService

router = APIRouter(tags=["analysis"])


@router.post("/analysis/frame", response_model=AnalysisResponse)
async def analyse_frame(
    body: FrameAnalysisRequest,
    _: str = Depends(verify_api_key),
    svc: AnalysisService = Depends(get_analysis_service),
) -> AnalysisResponse:
    return await svc.process_frame(body)
