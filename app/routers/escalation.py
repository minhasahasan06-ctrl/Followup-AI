"""
Escalation API Router

REST API endpoints for red flag detection and escalation flow management.
Integrates with Agent Clona for real-time safety monitoring.

HIPAA Compliance:
- All endpoints require authentication
- PHI access is audit logged
- Consent verification for doctor sharing
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime

from app.database import get_db
from app.dependencies import get_current_user
from app.models.user import User
from app.services.red_flag_detection_service import (
    get_red_flag_service,
    RedFlagSeverity,
    EscalationType,
    RedFlagCategory
)
from app.services.escalation_flow_service import (
    get_escalation_service,
    get_doctor_availability_service,
    EscalationState
)

router = APIRouter(prefix="/api/escalation", tags=["escalation"])


class RedFlagCheckRequest(BaseModel):
    """Request to check text for red flags"""
    text: str = Field(..., min_length=1, max_length=10000)
    use_ai_analysis: bool = True
    context: Optional[Dict[str, Any]] = None


class RedFlagCheckResponse(BaseModel):
    """Response from red flag check"""
    detected: bool
    symptoms: List[Dict[str, Any]]
    highest_severity: Optional[str]
    escalation_type: Optional[str]
    categories: List[str]
    confidence_score: float
    recommended_actions: List[str]
    emergency_instructions: Optional[str]
    ai_analysis: Optional[str]


class EscalationInitiateRequest(BaseModel):
    """Request to initiate an escalation"""
    patient_id: str
    conversation_history: List[Dict[str, str]] = []
    conversation_id: Optional[str] = None
    detected_symptoms: Optional[List[Dict[str, Any]]] = None
    manual_trigger: bool = False
    manual_reason: Optional[str] = None


class EscalationStatusResponse(BaseModel):
    """Response with escalation status"""
    escalation_id: str
    state: str
    patient_id: str
    doctor_id: Optional[str]
    doctor_name: Optional[str]
    initiated_at: str
    handoff_established: bool
    channels_used: List[str]


class DoctorAvailabilityResponse(BaseModel):
    """Response with doctor availability"""
    available: bool
    doctor_id: str
    doctor_name: str
    online_status: str
    contact_method: str
    can_receive_calls: bool
    can_receive_chat: bool


@router.post("/check-red-flags", response_model=RedFlagCheckResponse)
async def check_red_flags(
    request: RedFlagCheckRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Check text for medical red flags.
    
    Used by Agent Clona during conversations to detect emergencies.
    Returns detected symptoms, severity, and recommended actions.
    """
    service = get_red_flag_service(db)
    
    detection = service.detect_red_flags(
        text=request.text,
        patient_id=str(current_user.id),
        use_ai_analysis=request.use_ai_analysis,
        context=request.context
    )
    
    return RedFlagCheckResponse(
        detected=detection.detected,
        symptoms=detection.symptoms,
        highest_severity=detection.highest_severity.value if detection.highest_severity else None,
        escalation_type=detection.escalation_type.value if detection.escalation_type else None,
        categories=[c.value for c in detection.categories],
        confidence_score=detection.confidence_score,
        recommended_actions=detection.recommended_actions,
        emergency_instructions=detection.emergency_instructions,
        ai_analysis=detection.ai_analysis
    )


@router.post("/initiate")
async def initiate_escalation(
    request: EscalationInitiateRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Initiate an escalation flow for a patient.
    
    This triggers the full escalation flow:
    1. Clona acknowledges the emergency
    2. Lysa prepares doctor briefing
    3. Doctor is alerted through multiple channels
    4. Direct communication channel is established
    """
    user_role = getattr(current_user, 'role', 'patient')
    
    if user_role == "patient":
        patient_id = str(current_user.id)
    elif user_role == "doctor" or user_role == "admin":
        patient_id = request.patient_id
    else:
        raise HTTPException(status_code=403, detail="Unauthorized to initiate escalation")
    
    service = get_red_flag_service(db)
    
    if request.detected_symptoms:
        from app.services.red_flag_detection_service import RedFlagDetection
        from dataclasses import fields
        
        detection = RedFlagDetection(
            detected=True,
            symptoms=request.detected_symptoms,
            highest_severity=RedFlagSeverity.HIGH,
            escalation_type=EscalationType.IMMEDIATE_DOCTOR,
            categories=[RedFlagCategory.OTHER],
            confidence_score=1.0,
            recommended_actions=["Contact doctor immediately"],
            emergency_instructions=None,
            ai_analysis=request.manual_reason
        )
    else:
        combined_text = " ".join([
            msg.get("content", "") 
            for msg in request.conversation_history 
            if msg.get("role") == "user"
        ])
        
        detection = service.detect_red_flags(
            text=combined_text,
            patient_id=patient_id,
            use_ai_analysis=True
        )
        
        if not detection.detected and not request.manual_trigger:
            return {
                "success": False,
                "message": "No red flags detected in conversation",
                "escalation_id": None
            }
    
    escalation_service = get_escalation_service(db)
    
    result = await escalation_service.initiate_escalation(
        patient_id=patient_id,
        red_flag_detection=detection,
        conversation_history=request.conversation_history,
        conversation_id=request.conversation_id
    )
    
    return {
        "success": result.success,
        "escalation_id": result.escalation_id,
        "final_state": result.final_state.value,
        "doctor_contacted": result.doctor_contacted,
        "handoff_established": result.handoff_established,
        "channels_used": [c.value for c in result.channels_used],
        "patient_guidance": result.patient_guidance,
        "next_steps": result.next_steps,
        "fallback_used": result.fallback_used
    }


@router.get("/doctor-availability/{patient_id}")
async def get_doctor_availability(
    patient_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get availability of all doctors connected to a patient.
    
    Returns online status and best contact method for each doctor.
    """
    user_role = getattr(current_user, 'role', 'patient')
    
    if user_role == "patient" and str(current_user.id) != patient_id:
        raise HTTPException(status_code=403, detail="Can only check your own doctors")
    
    service = get_doctor_availability_service(db)
    doctors = service.get_all_connected_doctors(patient_id)
    
    return {
        "patient_id": patient_id,
        "doctors": doctors,
        "any_available": any(d.get("available") for d in doctors),
        "checked_at": datetime.utcnow().isoformat()
    }


@router.get("/severity-guidance/{severity}")
async def get_severity_guidance(
    severity: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get guidance for a specific severity level.
    
    Returns action required, timeframe, and instructions.
    """
    try:
        severity_enum = RedFlagSeverity(severity)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid severity: {severity}")
    
    service = get_red_flag_service(db)
    guidance = service.get_severity_guidance(severity_enum)
    
    return {
        "severity": severity,
        "guidance": {
            "action_required": guidance["action_required"],
            "timeframe": guidance["timeframe"],
            "instructions": guidance["instructions"],
            "escalation_type": guidance["escalation"].value
        }
    }


@router.get("/taxonomy")
async def get_red_flag_taxonomy(
    current_user: User = Depends(get_current_user)
):
    """
    Get the complete red flag taxonomy for reference.
    
    Returns all monitored symptoms with their categories and severities.
    """
    from app.services.red_flag_detection_service import RedFlagDetectionService
    
    taxonomy = []
    for symptom in RedFlagDetectionService.RED_FLAG_TAXONOMY:
        taxonomy.append({
            "name": symptom.name,
            "category": symptom.category.value,
            "severity": symptom.severity.value,
            "escalation_type": symptom.escalation_type.value,
            "description": symptom.description,
            "keywords": symptom.keywords[:5]
        })
    
    return {
        "taxonomy": taxonomy,
        "categories": [c.value for c in RedFlagCategory],
        "severities": [s.value for s in RedFlagSeverity],
        "escalation_types": [e.value for e in EscalationType]
    }


@router.post("/analyze-conversation")
async def analyze_conversation_history(
    conversation_history: List[Dict[str, str]],
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Analyze a full conversation for red flags.
    
    Used for retrospective analysis or when reviewing past conversations.
    """
    service = get_red_flag_service(db)
    
    detection = service.analyze_conversation_history(
        messages=conversation_history,
        patient_id=str(current_user.id)
    )
    
    return {
        "detected": detection.detected,
        "symptoms": detection.symptoms,
        "highest_severity": detection.highest_severity.value if detection.highest_severity else None,
        "escalation_type": detection.escalation_type.value if detection.escalation_type else None,
        "ai_analysis": detection.ai_analysis,
        "confidence_score": detection.confidence_score
    }
