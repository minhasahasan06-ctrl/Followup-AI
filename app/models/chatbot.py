from sqlalchemy import Column, String, DateTime, Text, Integer
from sqlalchemy.sql import func
from app.database import Base


class ChatbotConversation(Base):
    __tablename__ = "chatbot_conversations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    doctor_id = Column(String, nullable=False, index=True)
    session_id = Column(String, nullable=False, index=True)
    
    user_message = Column(Text, nullable=False)
    bot_response = Column(Text, nullable=False)
    
    created_at = Column(DateTime, server_default=func.now())
