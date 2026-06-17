from functools import lru_cache
from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader

from core.config import settings
from services.analysis_service import AnalysisService
from services.session_service import SessionService

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str = Security(_api_key_header)) -> str:
    if api_key != settings.API_KEY:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")
    return api_key


@lru_cache(maxsize=1)
def get_analysis_service() -> AnalysisService:
    return AnalysisService()


@lru_cache(maxsize=1)
def get_session_service() -> SessionService:
    return SessionService()
