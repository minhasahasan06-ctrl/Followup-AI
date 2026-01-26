"""
Patient AI Router (Phase C.1-C.5)
=================================
AI-powered endpoints for patient-facing features.

Endpoints:
- C.2: POST /api/patient/ai/next-questions
- C.3: POST /api/patient/ai/autopilot-plan
- C.4: POST /api/patient/ai/habit-suggestions
- C.5: POST /api/patient/ai/feedback
"""

import hashlib
import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from uuid import uuid4, UUID

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.tinker_models import (
    PatientAIExperience,
    PatientFeedback,
)
from app.services.feature_packets import build_patient_packet
from app.services.local_library import get_questions, get_habits, render_templates
from app.services.privacy_firewall import TinkerPurpose
from app.services.tinker_client import call_tinker

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/patient/ai", tags=["patient-ai"])


class NextQuestionsRequest(BaseModel):
    patient_id: str = Field(..., min_length=1)
    patient_data: Optional[Dict[str, Any]] = None


class NextQuestionsResponse(BaseModel):
    experience_id: str
    questions: List[Dict[str, Any]]
    generated_at: str
    is_stub: bool = False


class AutopilotPlanRequest(BaseModel):
    patient_id: str = Field(..., min_length=1)
    patient_data: Optional[Dict[str, Any]] = None


class AutopilotPlanResponse(BaseModel):
    experience_id: str
    templates: List[Dict[str, Any]]
    generated_at: str
    is_stub: bool = False


class HabitSuggestionsRequest(BaseModel):
    patient_id: str = Field(..., min_length=1)
    patient_data: Optional[Dict[str, Any]] = None


class HabitSuggestionsResponse(BaseModel):
    experience_id: str
    habits: List[Dict[str, Any]]
    generated_at: str
    is_stub: bool = False


class FeedbackRequest(BaseModel):
    experience_id: str = Field(..., min_length=1)
    patient_id: str = Field(..., min_length=1)
    rating: str = Field(..., pattern="^(helpful|not_helpful|neutral)$")
    reason_code: Optional[str] = None
    additional_context: Optional[str] = None


class FeedbackResponse(BaseModel):
    feedback_id: str
    experience_id: str
    rating: str
    received_at: str


def _compute_packet_hash(packet: Dict[str, Any]) -> str:
    """Compute SHA256 hash of packet for audit tracking"""
    packet_str = json.dumps(packet, sort_keys=True, default=str)
    return hashlib.sha256(packet_str.encode('utf-8')).hexdigest()


def _store_experience(
    db: Session,
    patient_id: str,
    experience_type: str,
    packet_hash: str,
    question_ids: Optional[List[str]] = None,
    template_ids: Optional[List[str]] = None,
    habit_ids: Optional[List[str]] = None,
    audit_log_id: Optional[str] = None,
) -> str:
    """Store AI experience for feedback tracking"""
    experience = PatientAIExperience(
        id=uuid4(),
        patient_id=patient_id,
        experience_type=experience_type,
        packet_hash=packet_hash,
        question_ids_json=question_ids,
        template_ids_json=template_ids,
        habit_ids_json=habit_ids,
        expires_at=datetime.utcnow() + timedelta(days=30),
    )
    db.add(experience)
    db.commit()
    db.refresh(experience)
    return str(experience.id)


def _log_phi_access(user_id: str, action: str, resource_type: str, resource_id: str) -> None:
    """Log PHI access for HIPAA compliance"""
    try:
        from app.services.access_control import HIPAAAuditLogger
        HIPAAAuditLogger.log_phi_access(
            user_id=user_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
        )
    except Exception as e:
        logger.warning(f"Failed to log PHI access: {e}")


@router.post("/next-questions", response_model=NextQuestionsResponse)
def next_questions(
    request: NextQuestionsRequest,
    db: Session = Depends(get_db),
) -> NextQuestionsResponse:
    """
    C.2: Generate personalized questions for a patient.
    
    Flow: packet -> Tinker -> question_ids -> local library
    """
    try:
        _log_phi_access(
            user_id=request.patient_id,
            action="patient_ai_next_questions",
            resource_type="patient_questions",
            resource_id=request.patient_id,
        )
        
        packet = build_patient_packet(request.patient_id, request.patient_data)
        packet_hash = _compute_packet_hash(packet)
        
        response, success = call_tinker(
            purpose=TinkerPurpose.PATIENT_QUESTIONS.value,
            payload=packet,
            actor_role="patient",
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="AI service temporarily unavailable",
            )
        
        question_ids = []
        if "questions" in response:
            question_ids = [q.get("id") or q.get("question_id") for q in response["questions"] if q.get("id") or q.get("question_id")]
        
        if not question_ids:
            question_ids = ["Q001", "Q002", "Q003"]
        
        questions = get_questions(question_ids)
        
        experience_id = _store_experience(
            db=db,
            patient_id=request.patient_id,
            experience_type="next_questions",
            packet_hash=packet_hash,
            question_ids=question_ids,
        )
        
        return NextQuestionsResponse(
            experience_id=experience_id,
            questions=questions,
            generated_at=datetime.utcnow().isoformat(),
            is_stub=response.get("is_stub", False),
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating next questions: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate questions",
        )


@router.post("/autopilot-plan", response_model=AutopilotPlanResponse)
def autopilot_plan(
    request: AutopilotPlanRequest,
    db: Session = Depends(get_db),
) -> AutopilotPlanResponse:
    """
    C.3: Generate autopilot plan with personalized templates.
    
    Flow: packet -> Tinker -> template_ids -> local render
    """
    try:
        _log_phi_access(
            user_id=request.patient_id,
            action="patient_ai_autopilot_plan",
            resource_type="patient_templates",
            resource_id=request.patient_id,
        )
        
        packet = build_patient_packet(request.patient_id, request.patient_data)
        packet_hash = _compute_packet_hash(packet)
        
        response, success = call_tinker(
            purpose=TinkerPurpose.PATIENT_TEMPLATES.value,
            payload=packet,
            actor_role="patient",
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="AI service temporarily unavailable",
            )
        
        template_ids = []
        if "templates" in response:
            template_ids = [t.get("id") or t.get("template_id") for t in response["templates"] if t.get("id") or t.get("template_id")]
        
        if not template_ids:
            template_ids = ["T001", "T002", "T003"]
        
        templates = render_templates(template_ids, packet)
        
        experience_id = _store_experience(
            db=db,
            patient_id=request.patient_id,
            experience_type="autopilot_plan",
            packet_hash=packet_hash,
            template_ids=template_ids,
        )
        
        return AutopilotPlanResponse(
            experience_id=experience_id,
            templates=templates,
            generated_at=datetime.utcnow().isoformat(),
            is_stub=response.get("is_stub", False),
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating autopilot plan: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate autopilot plan",
        )


@router.post("/habit-suggestions", response_model=HabitSuggestionsResponse)
def habit_suggestions(
    request: HabitSuggestionsRequest,
    db: Session = Depends(get_db),
) -> HabitSuggestionsResponse:
    """
    C.4: Generate personalized habit suggestions.
    
    Flow: packet -> Tinker -> habit_ids -> local library
    """
    try:
        _log_phi_access(
            user_id=request.patient_id,
            action="patient_ai_habit_suggestions",
            resource_type="patient_habits",
            resource_id=request.patient_id,
        )
        
        packet = build_patient_packet(request.patient_id, request.patient_data)
        packet_hash = _compute_packet_hash(packet)
        
        response, success = call_tinker(
            purpose=TinkerPurpose.PATIENT_QUESTIONS.value,
            payload=packet,
            actor_role="patient",
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="AI service temporarily unavailable",
            )
        
        habit_ids = []
        if "habits" in response:
            habit_ids = [h.get("id") or h.get("habit_id") for h in response["habits"] if h.get("id") or h.get("habit_id")]
        
        if not habit_ids:
            habit_ids = ["H001", "H002", "H003", "H004", "H005"]
        
        habits = get_habits(habit_ids)
        
        experience_id = _store_experience(
            db=db,
            patient_id=request.patient_id,
            experience_type="habit_suggestions",
            packet_hash=packet_hash,
            habit_ids=habit_ids,
        )
        
        return HabitSuggestionsResponse(
            experience_id=experience_id,
            habits=habits,
            generated_at=datetime.utcnow().isoformat(),
            is_stub=response.get("is_stub", False),
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating habit suggestions: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate habit suggestions",
        )


@router.post("/feedback", response_model=FeedbackResponse)
def submit_feedback(
    request: FeedbackRequest,
    db: Session = Depends(get_db),
) -> FeedbackResponse:
    """
    C.5: Store patient feedback for an AI experience.
    """
    try:
        _log_phi_access(
            user_id=request.patient_id,
            action="patient_ai_feedback",
            resource_type="patient_feedback",
            resource_id=request.experience_id,
        )
        
        try:
            experience_uuid = UUID(request.experience_id)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid experience ID format",
            )
        
        experience = db.query(PatientAIExperience).filter(
            PatientAIExperience.id == experience_uuid
        ).first()
        
        if not experience:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Experience not found",
            )
        
        if experience.patient_id != request.patient_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to provide feedback for this experience",
            )
        
        feedback = PatientFeedback(
            id=uuid4(),
            patient_id=request.patient_id,
            experience_id=experience.id,
            rating=request.rating,
            reason_code=request.reason_code,
            additional_context=request.additional_context,
        )
        
        db.add(feedback)
        db.commit()
        db.refresh(feedback)
        
        return FeedbackResponse(
            feedback_id=str(feedback.id),
            experience_id=str(experience.id),
            rating=request.rating,
            received_at=datetime.utcnow().isoformat(),
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error storing feedback: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to store feedback",
        )
