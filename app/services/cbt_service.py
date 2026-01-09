"""
CBT (Cognitive Behavioral Therapy) Service
Structured CBT tools with crisis detection and clinician notification.
"""

import os
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
from sqlalchemy.orm import Session
from sqlalchemy import desc

from app.models.lysa_drafts import CBTSession

logger = logging.getLogger(__name__)

CRISIS_KEYWORDS = [
    "suicide", "suicidal", "kill myself", "end my life", "don't want to live",
    "self-harm", "hurt myself", "cutting", "overdose", "end it all",
    "better off dead", "no reason to live", "can't go on", "want to die"
]

CRISIS_RESOURCES = {
    "national_suicide_prevention": "988 (Suicide and Crisis Lifeline)",
    "crisis_text_line": "Text HOME to 741741",
    "international": "findahelpline.com",
    "emergency": "911 for immediate danger"
}

CBT_THOUGHT_RECORD_PROMPTS = [
    {
        "id": "situation",
        "prompt": "Describe the situation that triggered your distress. What happened? When and where?",
        "help_text": "Be specific about the actual event or trigger."
    },
    {
        "id": "automatic_thoughts",
        "prompt": "What thoughts went through your mind? What were you saying to yourself?",
        "help_text": "Try to capture the exact words or images that came to mind."
    },
    {
        "id": "emotions",
        "prompt": "What emotions did you feel? Rate each emotion from 0-100%.",
        "help_text": "Common emotions: anxiety, sadness, anger, guilt, shame, fear, frustration."
    },
    {
        "id": "evidence_for",
        "prompt": "What evidence supports this thought? What facts suggest it might be true?",
        "help_text": "Focus on objective facts, not feelings or assumptions."
    },
    {
        "id": "evidence_against",
        "prompt": "What evidence goes against this thought? What facts suggest it might not be entirely true?",
        "help_text": "Consider alternative explanations or past experiences."
    },
    {
        "id": "balanced_thought",
        "prompt": "Based on the evidence, what's a more balanced or realistic way to think about this?",
        "help_text": "This isn't about being positive - it's about being accurate."
    },
    {
        "id": "action_plan",
        "prompt": "What can you do now? What's one small step you can take?",
        "help_text": "Focus on what's within your control."
    }
]


class CBTService:
    """
    CBT therapy tools service with crisis detection.
    All sessions are stored for clinician review.
    """
    
    def __init__(self, db: Session):
        self.db = db
    
    def check_crisis(self, text: str) -> Dict[str, Any]:
        """
        Check text for crisis indicators.
        
        Returns:
            dict with:
            - is_crisis: bool
            - matched_keywords: list of matched keywords
            - severity: low/medium/high
        """
        if not text:
            return {"is_crisis": False, "matched_keywords": [], "severity": None}
        
        text_lower = text.lower()
        matched = []
        
        for keyword in CRISIS_KEYWORDS:
            if keyword in text_lower:
                matched.append(keyword)
        
        if not matched:
            return {"is_crisis": False, "matched_keywords": [], "severity": None}
        
        high_severity_keywords = ["suicide", "suicidal", "kill myself", "end my life", "want to die"]
        has_high_severity = any(k in matched for k in high_severity_keywords)
        
        return {
            "is_crisis": True,
            "matched_keywords": matched,
            "severity": "high" if has_high_severity else "medium",
            "resources": CRISIS_RESOURCES
        }
    
    async def create_session(
        self,
        patient_id: str,
        session_type: str = "thought_record"
    ) -> Dict[str, Any]:
        """Create a new CBT session."""
        session = CBTSession(
            patient_id=patient_id,
            session_type=session_type,
            prompts_used=CBT_THOUGHT_RECORD_PROMPTS,
            responses={}
        )
        
        self.db.add(session)
        self.db.commit()
        self.db.refresh(session)
        
        return {
            "id": session.id,
            "session_type": session_type,
            "prompts": CBT_THOUGHT_RECORD_PROMPTS,
            "created_at": session.created_at.isoformat() if session.created_at else None
        }
    
    async def update_session(
        self,
        session_id: str,
        patient_id: str,
        updates: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Update a CBT session with new responses.
        Checks for crisis indicators in each response.
        """
        session = self.db.query(CBTSession).filter(
            CBTSession.id == session_id,
            CBTSession.patient_id == patient_id
        ).first()
        
        if not session:
            raise ValueError("Session not found")
        
        if updates.get("situation"):
            session.situation = updates["situation"]
        if updates.get("automatic_thoughts"):
            session.automatic_thoughts = updates["automatic_thoughts"]
        if updates.get("emotions"):
            session.emotions = updates["emotions"]
        if updates.get("evidence_for"):
            session.evidence_for = updates["evidence_for"]
        if updates.get("evidence_against"):
            session.evidence_against = updates["evidence_against"]
        if updates.get("balanced_thought"):
            session.balanced_thought = updates["balanced_thought"]
        if updates.get("action_plan"):
            session.action_plan = updates["action_plan"]
        if updates.get("distress_before") is not None:
            session.distress_before = updates["distress_before"]
        if updates.get("distress_after") is not None:
            session.distress_after = updates["distress_after"]
        
        all_text = " ".join([
            session.situation or "",
            session.automatic_thoughts or "",
            session.balanced_thought or "",
            session.action_plan or ""
        ])
        
        crisis_check = self.check_crisis(all_text)
        
        if crisis_check["is_crisis"]:
            session.crisis_detected = True
            await self._handle_crisis(session, crisis_check)
        
        current_responses = session.responses or {}
        current_responses.update(updates)
        session.responses = current_responses
        
        session.updated_at = datetime.utcnow()
        self.db.commit()
        
        return {
            "id": session.id,
            "updated": True,
            "crisis_detected": session.crisis_detected,
            "crisis_resources": CRISIS_RESOURCES if session.crisis_detected else None
        }
    
    async def complete_session(
        self,
        session_id: str,
        patient_id: str
    ) -> Dict[str, Any]:
        """Mark a session as completed."""
        session = self.db.query(CBTSession).filter(
            CBTSession.id == session_id,
            CBTSession.patient_id == patient_id
        ).first()
        
        if not session:
            raise ValueError("Session not found")
        
        session.completed = True
        session.completed_at = datetime.utcnow()
        self.db.commit()
        
        return {
            "id": session.id,
            "completed": True,
            "completed_at": session.completed_at.isoformat() if session.completed_at else None,
            "distress_reduction": self._calculate_distress_reduction(session)
        }
    
    async def get_sessions(
        self,
        patient_id: str,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Get patient's CBT sessions."""
        sessions = self.db.query(CBTSession).filter(
            CBTSession.patient_id == patient_id
        ).order_by(desc(CBTSession.created_at)).limit(limit).all()
        
        return [
            {
                "id": s.id,
                "session_type": s.session_type,
                "completed": s.completed,
                "crisis_detected": s.crisis_detected,
                "distress_before": s.distress_before,
                "distress_after": s.distress_after,
                "created_at": s.created_at.isoformat() if s.created_at else None,
                "completed_at": s.completed_at.isoformat() if s.completed_at else None
            }
            for s in sessions
        ]
    
    async def get_session_detail(
        self,
        session_id: str,
        patient_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get full session details."""
        session = self.db.query(CBTSession).filter(
            CBTSession.id == session_id,
            CBTSession.patient_id == patient_id
        ).first()
        
        if not session:
            return None
        
        return {
            "id": session.id,
            "session_type": session.session_type,
            "situation": session.situation,
            "automatic_thoughts": session.automatic_thoughts,
            "emotions": session.emotions,
            "evidence_for": session.evidence_for,
            "evidence_against": session.evidence_against,
            "balanced_thought": session.balanced_thought,
            "action_plan": session.action_plan,
            "distress_before": session.distress_before,
            "distress_after": session.distress_after,
            "completed": session.completed,
            "crisis_detected": session.crisis_detected,
            "created_at": session.created_at.isoformat() if session.created_at else None,
            "completed_at": session.completed_at.isoformat() if session.completed_at else None
        }
    
    async def _handle_crisis(
        self,
        session: CBTSession,
        crisis_check: Dict[str, Any]
    ):
        """Handle detected crisis - notify clinician."""
        session.crisis_action_taken = f"Crisis detected: {crisis_check['matched_keywords']}"
        
        try:
            from app.models.patient_doctor_connection import PatientDoctorConnection
            connections = self.db.query(PatientDoctorConnection).filter(
                PatientDoctorConnection.patient_id == session.patient_id,
                PatientDoctorConnection.status == "active"
            ).all()
            
            if connections:
                from app.services.alert_orchestration_engine import AlertOrchestrationEngine
                alert_engine = AlertOrchestrationEngine(self.db)
                
                for conn in connections:
                    await alert_engine.create_crisis_alert(
                        patient_id=session.patient_id,
                        doctor_id=conn.doctor_id,
                        crisis_type="cbt_session_crisis",
                        severity=crisis_check["severity"],
                        details={
                            "session_id": session.id,
                            "matched_keywords": crisis_check["matched_keywords"]
                        }
                    )
                
                session.clinician_notified = True
                session.clinician_notified_at = datetime.utcnow()
                logger.info(f"Crisis alert sent for session {session.id}")
        except Exception as e:
            logger.error(f"Failed to send crisis alert: {e}")
    
    def _calculate_distress_reduction(self, session: CBTSession) -> Optional[int]:
        """Calculate distress reduction percentage."""
        if session.distress_before is None or session.distress_after is None:
            return None
        if session.distress_before == 0:
            return 0
        return int(((session.distress_before - session.distress_after) / session.distress_before) * 100)


def get_cbt_service(db: Session) -> CBTService:
    """Factory function."""
    return CBTService(db)
