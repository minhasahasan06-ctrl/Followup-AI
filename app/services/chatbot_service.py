from typing import Dict, List
from sqlalchemy.orm import Session
from app.models.chatbot import ChatbotConversation
from app.config import settings, check_openai_baa_compliance
import openai


class ChatbotService:
    def __init__(self, db: Session):
        self.db = db
        self.openai_enabled = check_openai_baa_compliance()
        if self.openai_enabled and settings.OPENAI_API_KEY:
            openai.api_key = settings.OPENAI_API_KEY
    
    def get_conversation_history(self, doctor_id: str, session_id: str, limit: int = 10) -> List[Dict]:
        conversations = self.db.query(ChatbotConversation).filter(
            ChatbotConversation.doctor_id == doctor_id,
            ChatbotConversation.session_id == session_id
        ).order_by(ChatbotConversation.created_at.desc()).limit(limit).all()
        
        return [
            {
                "user_message": conv.user_message,
                "bot_response": conv.bot_response,
                "created_at": conv.created_at.isoformat()
            }
            for conv in reversed(conversations)
        ]
    
    async def chat(self, doctor_id: str, session_id: str, message: str) -> Dict:
        if not self.openai_enabled:
            fallback_response = self._get_fallback_response(message)
            conversation = ChatbotConversation(
                doctor_id=doctor_id,
                session_id=session_id,
                user_message=message,
                bot_response=fallback_response
            )
            self.db.add(conversation)
            self.db.commit()
            
            return {
                "response": fallback_response,
                "using_ai": False
            }
        
        history = self.get_conversation_history(doctor_id, session_id)
        
        messages = [
            {"role": "system", "content": "You are a helpful medical clinic assistant. Provide professional, concise responses to help doctors manage their practice."}
        ]
        
        for conv in history[-5:]:
            messages.append({"role": "user", "content": conv["user_message"]})
            messages.append({"role": "assistant", "content": conv["bot_response"]})
        
        messages.append({"role": "user", "content": message})
        
        try:
            response = await openai.ChatCompletion.acreate(
                model="gpt-4o",
                messages=messages,
                max_tokens=500,
                temperature=0.7
            )
            
            bot_response = response.choices[0].message.content
        except Exception as e:
            print(f"OpenAI API error: {e}")
            bot_response = self._get_fallback_response(message)
        
        conversation = ChatbotConversation(
            doctor_id=doctor_id,
            session_id=session_id,
            user_message=message,
            bot_response=bot_response
        )
        self.db.add(conversation)
        self.db.commit()
        
        return {
            "response": bot_response,
            "using_ai": self.openai_enabled
        }
    
    def _get_fallback_response(self, message: str) -> str:
        message_lower = message.lower()
        
        if "appointment" in message_lower or "schedule" in message_lower:
            return "I can help you with appointments. Please check the appointments section to view, create, or modify appointments."
        elif "patient" in message_lower:
            return "For patient information, please use the patient management section of the dashboard."
        elif "reminder" in message_lower:
            return "Appointment reminders are automatically sent 24 hours before scheduled appointments via SMS and email."
        else:
            return "I'm here to help! You can ask me about appointments, patient management, reminders, or general clinic operations."
