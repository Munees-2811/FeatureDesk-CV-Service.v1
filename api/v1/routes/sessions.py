from fastapi import APIRouter, Depends, HTTPException, status
from api.v1.schemas.requests import SessionCreateRequest
from api.v1.schemas.responses import SessionResponse
from core.dependencies import verify_api_key, get_session_service
from core.exceptions import SessionNotFoundError
from services.session_service import SessionService

router = APIRouter(tags=["sessions"])


@router.post("/sessions", response_model=SessionResponse, status_code=201)
async def create_session(
    body: SessionCreateRequest,
    _: str = Depends(verify_api_key),
    svc: SessionService = Depends(get_session_service),
) -> SessionResponse:
    return svc.create(body.student_id, body.metadata)


@router.get("/sessions/{session_id}", response_model=SessionResponse)
async def get_session(
    session_id: str,
    _: str = Depends(verify_api_key),
    svc: SessionService = Depends(get_session_service),
) -> SessionResponse:
    try:
        return svc.get(session_id)
    except SessionNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
