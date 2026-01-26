"""
CBT Router
Endpoints for Cognitive Behavioral Therapy tools.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import logging

from app.database import get_db
from app.services.cbt_service import get_cbt_service, CRISIS_RESOURCES, CBT_THOUGHT_RECORD_PROMPTS

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/cbt", tags=["cbt"])


class CreateSessionRequest(BaseModel):
    """Request to create a new CBT session."""
    session_type: str = Field(default="thought_record")


class UpdateSessionRequest(BaseModel):
    """Request to update CBT session."""
    situation: Optional[str] = None
    automatic_thoughts: Optional[str] = None
    emotions: Optional[Dict[str, int]] = None
    evidence_for: Optional[str] = None
    evidence_against: Optional[str] = None
    balanced_thought: Optional[str] = None
    action_plan: Optional[str] = None
    distress_before: Optional[int] = Field(None, ge=0, le=100)
    distress_after: Optional[int] = Field(None, ge=0, le=100)


@router.get("/prompts")
async def get_prompts():
    """Get CBT thought record prompts."""
    return {
        "prompts": CBT_THOUGHT_RECORD_PROMPTS,
        "crisis_resources": CRISIS_RESOURCES
    }


@router.post("/patient/{patient_id}/sessions")
async def create_session(
    patient_id: str,
    request: CreateSessionRequest,
    db: Session = Depends(get_db)
):
    """Create a new CBT session."""
    cbt_service = get_cbt_service(db)
    
    try:
        result = await cbt_service.create_session(
            patient_id=patient_id,
            session_type=request.session_type
        )
        return result
    except Exception as e:
        logger.error(f"Error creating CBT session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/patient/{patient_id}/sessions")
async def get_sessions(
    patient_id: str,
    limit: int = 20,
    db: Session = Depends(get_db)
):
    """Get patient's CBT sessions."""
    cbt_service = get_cbt_service(db)
    
    try:
        sessions = await cbt_service.get_sessions(patient_id, limit)
        return {"sessions": sessions}
    except Exception as e:
        logger.error(f"Error getting CBT sessions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/patient/{patient_id}/sessions/{session_id}")
async def get_session_detail(
    patient_id: str,
    session_id: str,
    db: Session = Depends(get_db)
):
    """Get full session details."""
    cbt_service = get_cbt_service(db)
    
    try:
        session = await cbt_service.get_session_detail(session_id, patient_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        return session
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting CBT session detail: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/patient/{patient_id}/sessions/{session_id}")
async def update_session(
    patient_id: str,
    session_id: str,
    request: UpdateSessionRequest,
    db: Session = Depends(get_db)
):
    """
    Update a CBT session.
    
    Note: The system automatically checks for crisis indicators.
    If detected, crisis resources are returned and the clinician is notified.
    """
    cbt_service = get_cbt_service(db)
    
    try:
        result = await cbt_service.update_session(
            session_id=session_id,
            patient_id=patient_id,
            updates=request.dict(exclude_none=True)
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error updating CBT session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/patient/{patient_id}/sessions/{session_id}/complete")
async def complete_session(
    patient_id: str,
    session_id: str,
    db: Session = Depends(get_db)
):
    """Mark a session as completed."""
    cbt_service = get_cbt_service(db)
    
    try:
        result = await cbt_service.complete_session(session_id, patient_id)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error completing CBT session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/check-crisis")
async def check_crisis(
    text: str,
    db: Session = Depends(get_db)
):
    """
    Check text for crisis indicators.
    
    This is a helper endpoint for real-time crisis detection
    as users type their responses.
    """
    cbt_service = get_cbt_service(db)
    result = cbt_service.check_crisis(text)
    return result


class AddAsHabitRequest(BaseModel):
    """Request to create a habit from CBT session."""
    habit_name: str = Field(..., description="Name for the habit")
    description: Optional[str] = Field(None, description="Optional description")
    frequency: str = Field(default="daily", description="daily, weekly, etc")
    category: str = Field(default="mental_health")


@router.post("/patient/{patient_id}/sessions/{session_id}/add-as-habit")
async def add_session_as_habit(
    patient_id: str,
    session_id: str,
    request: AddAsHabitRequest,
    db: Session = Depends(get_db)
):
    """
    Create a habit from a CBT session's action plan.
    
    This allows patients to turn their CBT insights into trackable habits.
    """
    from app.models.habit_models import HabitHabit
    
    cbt_service = get_cbt_service(db)
    
    try:
        session = await cbt_service.get_session_detail(session_id, patient_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        description = request.description or session.get("action_plan") or session.get("balanced_thought")
        
        habit = HabitHabit(
            user_id=patient_id,
            name=request.habit_name,
            description=description,
            category=request.category,
            frequency=request.frequency,
            goal_count=1,
            streak_count=0,
            total_completions=0,
            is_active=True
        )
        
        db.add(habit)
        db.commit()
        db.refresh(habit)
        
        logger.info(f"Created habit from CBT session [session_id={session_id}]")
        
        return {
            "success": True,
            "habit_id": habit.id,
            "habit_name": habit.name,
            "source_session_id": session_id
        }
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to create habit from CBT session [session_id={session_id}]: {e}")
        raise HTTPException(
            status_code=500, 
            detail="Failed to create habit from session. Please try again."
        )
