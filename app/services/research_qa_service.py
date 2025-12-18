"""
Research Q&A AI Service
Production-grade Q&A service for research data analysis using OpenAI.
Follows HIPAA compliance with de-identified data only.
"""

import logging
import os
from typing import Optional, Dict, Any, List
from datetime import datetime

from sqlalchemy.orm import Session
from openai import OpenAI

from app.models.research_models import (
    ResearchQASession,
    ResearchQAMessage,
    ResearchDataset,
    ResearchStudy,
)
from app.services.access_control import HIPAAAuditLogger

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a medical research AI assistant helping doctors analyze de-identified patient cohort data.

IMPORTANT GUIDELINES:
1. You ONLY work with aggregate, de-identified data. Never reference individual patients.
2. All responses must maintain k-anonymity (minimum 5 patients in any group).
3. Be precise and evidence-based in your analysis.
4. Cite statistical methods when presenting findings.
5. Flag any potential data quality issues or limitations.
6. Suggest follow-up analyses when appropriate.

You have access to research datasets with the following types of information:
- Aggregate demographic distributions (age ranges, gender percentages)
- Condition prevalence statistics
- Medication patterns and adherence rates
- Risk score distributions
- Outcome metrics and trends

When asked about specific patients, remind the user that only aggregate data is available.
Provide insights that help researchers understand patterns and trends in the data."""


class ResearchQAService:
    """
    AI-powered Q&A service for research data analysis.
    Uses OpenAI GPT-4o with research-specific prompting.
    """
    
    def __init__(self, db: Session):
        self.db = db
        self.client = self._get_openai_client()
        self.model = os.getenv("OPENAI_RESEARCH_MODEL", "gpt-4o")
    
    def _get_openai_client(self) -> Optional[OpenAI]:
        """Get OpenAI client, return None if not configured"""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning("OpenAI API key not configured")
            return None
        return OpenAI(api_key=api_key)
    
    def _get_dataset_context(self, dataset_id: Optional[str]) -> str:
        """Get dataset schema context for the AI"""
        if not dataset_id:
            return "No specific dataset selected. Working with general research data."
        
        dataset = self.db.query(ResearchDataset).filter(
            ResearchDataset.id == dataset_id
        ).first()
        
        if not dataset:
            return "Dataset not found."
        
        columns_info = []
        if dataset.columns_json:
            for col in dataset.columns_json:
                col_name = col.get("name", "unknown")
                col_type = col.get("type", "unknown")
                columns_info.append(f"- {col_name} ({col_type})")
        
        return f"""Dataset Context:
Name: {dataset.name}
Description: {dataset.description or 'No description'}
Rows: {dataset.row_count or 'Unknown'}
Columns: {dataset.column_count or 'Unknown'}
Privacy Level: {dataset.pii_classification}

Available Columns:
{chr(10).join(columns_info) if columns_info else 'No column information available'}"""
    
    def _get_study_context(self, study_id: Optional[str]) -> str:
        """Get study context for the AI"""
        if not study_id:
            return ""
        
        study = self.db.query(ResearchStudy).filter(
            ResearchStudy.id == study_id
        ).first()
        
        if not study:
            return ""
        
        return f"""Study Context:
Title: {study.title}
Description: {study.description or 'No description'}
Status: {study.status}
Target Enrollment: {study.target_enrollment}
Current Enrollment: {study.current_enrollment}
Inclusion Criteria: {study.inclusion_criteria or 'Not specified'}
Exclusion Criteria: {study.exclusion_criteria or 'Not specified'}"""
    
    def _build_messages(
        self,
        session: ResearchQASession,
        user_message: str,
    ) -> List[Dict[str, str]]:
        """Build OpenAI messages array with context"""
        context_parts = [SYSTEM_PROMPT]
        
        dataset_context = self._get_dataset_context(str(session.dataset_id) if session.dataset_id else None)
        if dataset_context:
            context_parts.append(dataset_context)
        
        study_context = self._get_study_context(str(session.study_id) if session.study_id else None)
        if study_context:
            context_parts.append(study_context)
        
        if session.context_json:
            context_parts.append(f"Additional Context: {session.context_json}")
        
        messages = [{"role": "system", "content": "\n\n".join(context_parts)}]
        
        history = self.db.query(ResearchQAMessage).filter(
            ResearchQAMessage.session_id == session.id
        ).order_by(ResearchQAMessage.created_at.asc()).limit(20).all()
        
        for msg in history:
            messages.append({
                "role": str(msg.role),
                "content": str(msg.content),
            })
        
        messages.append({"role": "user", "content": user_message})
        
        return messages
    
    async def generate_response(
        self,
        session: ResearchQASession,
        user_message: str,
        user_id: str,
        user_role: str,
    ) -> Dict[str, Any]:
        """
        Generate AI response to user message.
        
        Returns dict with:
        - user_message: saved user message
        - assistant_message: saved AI response
        - token_usage: total tokens used
        """
        user_msg = ResearchQAMessage(
            session_id=str(session.id),
            role="user",
            content=user_message,
        )
        self.db.add(user_msg)
        
        if not self.client:
            ai_response = self._fallback_response(user_message)
            token_usage = 0
            model_name = "fallback"
        else:
            try:
                messages = self._build_messages(session, user_message)
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.3,
                    max_tokens=1500,
                )
                
                ai_response = response.choices[0].message.content or ""
                token_usage = response.usage.total_tokens if response.usage else 0
                model_name = self.model
                
            except Exception as e:
                logger.error(f"OpenAI API error: {e}")
                ai_response = self._fallback_response(user_message)
                token_usage = 0
                model_name = "fallback"
        
        assistant_msg = ResearchQAMessage(
            session_id=str(session.id),
            role="assistant",
            content=ai_response,
            model_name=model_name,
            token_usage=token_usage,
        )
        self.db.add(assistant_msg)
        
        session.total_messages = (session.total_messages or 0) + 2
        session.total_tokens = (session.total_tokens or 0) + token_usage
        session.last_message_at = datetime.utcnow()
        
        self.db.commit()
        
        HIPAAAuditLogger.log_phi_access(
            actor_id=user_id,
            actor_role=user_role,
            patient_id="aggregate",
            action="ai_qa_interaction",
            phi_categories=["de_identified"],
            resource_type="qa_session",
            resource_id=str(session.id),
            access_scope="research",
            access_reason="ai_research_analysis",
            consent_verified=True,
            additional_context={
                "tokens_used": token_usage,
                "model": model_name,
            }
        )
        
        return {
            "user_message": {
                "id": user_msg.id,
                "role": "user",
                "content": user_message,
            },
            "assistant_message": {
                "id": assistant_msg.id,
                "role": "assistant",
                "content": ai_response,
            },
            "token_usage": token_usage,
        }
    
    def _fallback_response(self, user_message: str) -> str:
        """Generate a helpful fallback response when OpenAI is unavailable"""
        return f"""I'm currently operating in fallback mode due to AI service unavailability.

Your query: "{user_message}"

To help you with research data analysis, I can:
1. Provide aggregate statistics from your cohort
2. Show demographic distributions
3. Analyze condition prevalence
4. Report on medication patterns

Please ensure the OpenAI API key is configured for full AI-powered analysis capabilities.

In the meantime, you can:
- Use the Cohort Builder to filter and preview patient groups
- Export datasets for offline analysis
- Review existing study results and artifacts"""
    
    def summarize_session(self, session_id: str) -> str:
        """Generate a summary of a Q&A session"""
        messages = self.db.query(ResearchQAMessage).filter(
            ResearchQAMessage.session_id == session_id
        ).order_by(ResearchQAMessage.created_at.asc()).all()
        
        if not messages:
            return "No messages in this session."
        
        topics = []
        for msg in messages:
            if msg.role == "user":
                topics.append(str(msg.content)[:100])
        
        return f"Session with {len(messages)} messages covering: {', '.join(topics[:5])}"


def get_research_qa_service(db: Session) -> ResearchQAService:
    """Factory function for dependency injection"""
    return ResearchQAService(db)
