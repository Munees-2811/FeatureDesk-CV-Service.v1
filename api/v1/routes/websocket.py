from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from core.dependencies import get_session_service
from services.session_service import SessionService
from websocket.connection_manager import ConnectionManager
from websocket.frame_processor import FrameProcessor

router = APIRouter(tags=["websocket"])
manager = ConnectionManager()
processor = FrameProcessor()


@router.websocket("/ws/stream/{session_id}")
async def ws_stream(
    websocket: WebSocket,
    session_id: str,
    svc: SessionService = Depends(get_session_service),
):
    await manager.connect(session_id, websocket)
    try:
        while True:
            data = await websocket.receive_bytes()
            result = await processor.process(session_id, data)
            await manager.send(session_id, result.model_dump(mode="json"))
    except WebSocketDisconnect:
        manager.disconnect(session_id)
