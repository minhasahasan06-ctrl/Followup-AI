from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.database import get_db
from app.dependencies import get_current_doctor
from app.models.user import User
from app.services.chatbot_service import ChatbotService
from pydantic import BaseModel

router = APIRouter(prefix="/api/v1/chatbot", tags=["chatbot"])


class ChatRequest(BaseModel):
    session_id: str
    message: str


@router.post("/chat")
async def chat(
    request: ChatRequest,
    current_user: User = Depends(get_current_doctor),
    db: Session = Depends(get_db)
):
    service = ChatbotService(db)
    result = await service.chat(current_user.id, request.session_id, request.message)
    return result


@router.get("/history/{session_id}")
async def get_history(
    session_id: str,
    current_user: User = Depends(get_current_doctor),
    db: Session = Depends(get_db)
):
    service = ChatbotService(db)
    history = service.get_conversation_history(current_user.id, session_id)
    return {"history": history}
