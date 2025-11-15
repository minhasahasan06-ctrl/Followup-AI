"""
Home Clinical Exam Coach (HCEC) Router
AI-powered system that teaches patients proper self-examination techniques
with real-time coaching and quality feedback
"""

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session
from sqlalchemy import desc, and_
from typing import List, Optional
from datetime import datetime, timedelta
from pydantic import BaseModel
import base64
import json

from app.database import get_db
from app.models.exam_coach import (
    ExamSession, ExamStep, CoachingFeedback, ExamPacket, ExamProtocol,
    ExamType, SessionStatus, QualityLevel, FeedbackType
)
from app.models.user import User
from app.models.patient_doctor_connection import PatientDoctorConnection
from app.dependencies import require_role
from app.services.openai_service import OpenAIService
from app.services.s3_service import upload_file_to_s3, generate_presigned_url


router = APIRouter(prefix="/api/v1/exam-coach", tags=["Exam Coach"])


# Pydantic models
class StartSessionRequest(BaseModel):
    exam_type: str
    is_pre_consultation: bool = False
    consultation_date: Optional[datetime] = None


class AnalyzeFrameRequest(BaseModel):
    session_id: int
    step_id: int
    frame_data: str  # Base64 encoded image
    current_instruction: str


class CompleteStepRequest(BaseModel):
    session_id: int
    step_id: int
    image_file: Optional[str] = None  # Base64 encoded
    video_file: Optional[str] = None  # Base64 encoded


class ExamSessionResponse(BaseModel):
    id: int
    exam_type: str
    status: str
    total_steps: int
    completed_steps: int
    current_step: Optional[dict]
    overall_quality: Optional[str]
    started_at: datetime


# Protocol definitions for each exam type
EXAM_PROTOCOLS = {
    ExamType.SKIN: {
        "name": "Skin Examination",
        "description": "Visual inspection of skin for lesions, rashes, or changes",
        "steps": [
            {
                "step_number": 1,
                "instruction": "Position camera to view the affected area",
                "type": "photo",
                "coaching_hints": [
                    "Ensure good lighting - natural light is best",
                    "Hold camera 6-12 inches from skin",
                    "Keep camera steady"
                ],
                "quality_criteria": {
                    "lighting": "good",
                    "angle": "perpendicular",
                    "distance": "close"
                }
            },
            {
                "step_number": 2,
                "instruction": "Take a close-up photo of the area",
                "type": "photo",
                "coaching_hints": [
                    "Move closer for detail",
                    "Lesion should fill most of the frame",
                    "Keep in focus"
                ]
            },
            {
                "step_number": 3,
                "instruction": "Capture the surrounding area for context",
                "type": "photo",
                "coaching_hints": [
                    "Move back slightly",
                    "Show landmarks (moles, freckles nearby)",
                    "Include size reference if possible"
                ]
            }
        ]
    },
    ExamType.THROAT: {
        "name": "Throat Examination",
        "description": "Visual inspection of throat and tonsils",
        "steps": [
            {
                "step_number": 1,
                "instruction": "Position yourself in front of a mirror with good lighting",
                "type": "photo",
                "coaching_hints": [
                    "Use bright overhead light or flashlight",
                    "Face the light source",
                    "Have phone/camera ready"
                ]
            },
            {
                "step_number": 2,
                "instruction": "Open your mouth wide and say 'Ahh'",
                "type": "photo",
                "coaching_hints": [
                    "Open as wide as comfortable",
                    "Depress tongue with spoon if needed",
                    "Keep throat relaxed"
                ]
            },
            {
                "step_number": 3,
                "instruction": "Capture a photo showing your throat and tonsils",
                "type": "photo",
                "coaching_hints": [
                    "Center the throat in frame",
                    "Ensure tonsils are visible",
                    "Use flash if available"
                ]
            }
        ]
    },
    ExamType.LEGS: {
        "name": "Leg Examination (Edema)",
        "description": "Visual and palpation assessment for swelling",
        "steps": [
            {
                "step_number": 1,
                "instruction": "Position camera to view both legs",
                "type": "photo",
                "coaching_hints": [
                    "Sit with legs extended",
                    "Good lighting on legs",
                    "Camera at leg level"
                ]
            },
            {
                "step_number": 2,
                "instruction": "Take photo of ankles and lower legs",
                "type": "photo",
                "coaching_hints": [
                    "Compare left and right",
                    "Look for swelling or asymmetry",
                    "Capture from front and side"
                ]
            },
            {
                "step_number": 3,
                "instruction": "Press finger on shin for 5 seconds (palpation)",
                "type": "video",
                "coaching_hints": [
                    "Press firmly on bone",
                    "Hold for 5 seconds",
                    "Release and observe"
                ]
            },
            {
                "step_number": 4,
                "instruction": "Capture any indentation left after pressing",
                "type": "photo",
                "coaching_hints": [
                    "Photograph immediately after releasing",
                    "Close-up of the area",
                    "Note if indent remains"
                ]
            }
        ]
    },
    ExamType.ROM: {
        "name": "Range of Motion (Joint)",
        "description": "Assessment of joint flexibility and movement",
        "steps": [
            {
                "step_number": 1,
                "instruction": "Position camera to view the joint",
                "type": "video",
                "coaching_hints": [
                    "Show full joint (e.g., entire arm for elbow)",
                    "Camera at joint level",
                    "Clear view of movement"
                ]
            },
            {
                "step_number": 2,
                "instruction": "Slowly move the joint through full range",
                "type": "video",
                "coaching_hints": [
                    "Move slowly and smoothly",
                    "Go as far as comfortable",
                    "Record for 10-15 seconds"
                ]
            },
            {
                "step_number": 3,
                "instruction": "Capture photo at maximum extension/flexion",
                "type": "photo",
                "coaching_hints": [
                    "Photo at furthest comfortable position",
                    "Show angle clearly",
                    "Compare to other side if possible"
                ]
            }
        ]
    },
    ExamType.RESPIRATORY: {
        "name": "Respiratory Effort Assessment",
        "description": "Visual assessment of breathing pattern",
        "steps": [
            {
                "step_number": 1,
                "instruction": "Position camera to view chest/torso",
                "type": "video",
                "coaching_hints": [
                    "Camera at chest level",
                    "Show full torso",
                    "Sit or stand upright"
                ]
            },
            {
                "step_number": 2,
                "instruction": "Breathe normally for 30 seconds",
                "type": "video",
                "coaching_hints": [
                    "Breathe naturally - don't force it",
                    "Keep camera steady",
                    "Record for full 30 seconds"
                ]
            },
            {
                "step_number": 3,
                "instruction": "Take a deep breath and exhale",
                "type": "video",
                "coaching_hints": [
                    "Deep breath in through nose",
                    "Slow exhale through mouth",
                    "Observe chest movement"
                ]
            }
        ]
    }
}


@router.post("/start-session", dependencies=[Depends(require_role("patient"))])
async def start_exam_session(
    request: StartSessionRequest,
    current_user: User = Depends(require_role("patient")),
    db: Session = Depends(get_db)
):
    """
    Start a new exam coaching session
    Patient access only
    """
    try:
        exam_type = ExamType(request.exam_type)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid exam type: {request.exam_type}")
    
    # Get protocol for this exam type
    protocol = EXAM_PROTOCOLS.get(exam_type)
    if not protocol:
        raise HTTPException(status_code=400, detail=f"No protocol defined for {request.exam_type}")
    
    # Create session
    session = ExamSession(
        patient_id=current_user.id,
        exam_type=exam_type,
        status=SessionStatus.IN_PROGRESS,
        total_steps=len(protocol["steps"]),
        completed_steps=0,
        is_pre_consultation=request.is_pre_consultation,
        consultation_date=request.consultation_date
    )
    
    db.add(session)
    db.commit()
    db.refresh(session)
    
    # Create first step
    first_step_data = protocol["steps"][0]
    first_step = ExamStep(
        session_id=session.id,
        patient_id=current_user.id,
        step_number=first_step_data["step_number"],
        step_instruction=first_step_data["instruction"],
        step_type=first_step_data["type"],
        coaching_feedback=[],
        feedback_count=0
    )
    
    db.add(first_step)
    db.commit()
    db.refresh(first_step)
    
    return {
        "session_id": session.id,
        "exam_type": exam_type.value,
        "protocol": protocol,
        "current_step": {
            "step_id": first_step.id,
            "step_number": first_step.step_number,
            "instruction": first_step.step_instruction,
            "type": first_step.step_type,
            "hints": first_step_data.get("coaching_hints", [])
        }
    }


@router.post("/analyze-frame", dependencies=[Depends(require_role("patient"))])
async def analyze_camera_frame(
    request: AnalyzeFrameRequest,
    current_user: User = Depends(require_role("patient")),
    db: Session = Depends(get_db)
):
    """
    Analyze a camera frame in real-time and provide coaching feedback
    Patient access only - own sessions
    SECURITY: Double verification - session must belong to current user
    """
    # SECURITY: Verify session belongs to authenticated user
    session = db.query(ExamSession).filter(
        and_(
            ExamSession.id == request.session_id,
            ExamSession.patient_id == current_user.id  # CRITICAL: Must match authenticated user
        )
    ).first()
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found or access denied")
    
    # SECURITY: Verify step belongs to user's session
    step = db.query(ExamStep).filter(
        and_(
            ExamStep.id == request.step_id,
            ExamStep.patient_id == current_user.id,  # CRITICAL: Must match authenticated user
            ExamStep.session_id == session.id  # CRITICAL: Must match verified session
        )
    ).first()
    
    if not step:
        raise HTTPException(status_code=404, detail="Step not found or access denied")
    
    # Use OpenAI Vision to analyze frame quality
    openai_service = OpenAIService()
    
    prompt = f"""Analyze this camera frame for a medical self-examination: {request.current_instruction}

Assess the following and provide specific coaching feedback:
1. LIGHTING: Is there adequate lighting? Can details be seen clearly?
2. ANGLE: Is the camera positioned correctly? Is the target area centered?
3. DISTANCE: Is the camera the right distance away?
4. VISIBILITY: Is the target area (throat/skin/joint/etc.) clearly visible?
5. READINESS: Is this frame good enough to capture as the final image?

Respond in JSON format:
{{
    "lighting_quality": "excellent|good|acceptable|poor",
    "lighting_feedback": "<specific guidance like 'Turn on a light' or 'Good lighting'>",
    "angle_quality": "excellent|good|acceptable|poor",
    "angle_feedback": "<specific guidance like 'Tilt camera 20 degrees down' or 'Angle optimal'>",
    "distance_quality": "excellent|good|acceptable|poor",
    "distance_feedback": "<specific guidance like 'Move camera closer' or 'Distance good'>",
    "visibility_quality": "excellent|good|acceptable|poor",
    "visibility_feedback": "<what's visible or not>",
    "is_ready": true|false,
    "readiness_message": "<'Good to capture' or specific issue>",
    "voice_guidance": "<short spoken instruction like 'Tilt camera closer' or 'Perfect, ready to capture'>"
}}"""
    
    try:
        # Call OpenAI Vision API
        response = await openai_service.analyze_image_with_context(
            image_data=request.frame_data,
            context=prompt
        )
        
        # Parse JSON response
        analysis = json.loads(response)
        
        # Create coaching feedback if issues detected
        if not analysis.get("is_ready", False):
            feedback = CoachingFeedback(
                session_id=session.id,
                step_id=step.id,
                patient_id=current_user.id,
                feedback_type=FeedbackType.READINESS,
                feedback_message=analysis.get("readiness_message", "Adjustments needed"),
                voice_guidance=analysis.get("voice_guidance"),
                was_spoken=False,
                issue_resolved=False
            )
            db.add(feedback)
            
            # Increment feedback count
            step.feedback_count += 1
            db.commit()
        
        return {
            "analysis": analysis,
            "coaching_needed": not analysis.get("is_ready", False),
            "step_id": step.id
        }
        
    except Exception as e:
        # Fallback response
        return {
            "analysis": {
                "lighting_quality": "unknown",
                "lighting_feedback": "Unable to analyze",
                "is_ready": False,
                "error": str(e)
            },
            "coaching_needed": True
        }


@router.post("/complete-step", dependencies=[Depends(require_role("patient"))])
async def complete_exam_step(
    session_id: int = Form(...),
    step_id: int = Form(...),
    image_file: Optional[UploadFile] = File(None),
    video_file: Optional[UploadFile] = File(None),
    current_user: User = Depends(require_role("patient")),
    db: Session = Depends(get_db)
):
    """
    Complete a step and move to next step
    Upload captured image or video
    Patient access only - own sessions
    SECURITY: Triple verification - session, step, and patient must all match
    """
    # SECURITY: Verify session belongs to authenticated user
    session = db.query(ExamSession).filter(
        and_(
            ExamSession.id == session_id,
            ExamSession.patient_id == current_user.id  # CRITICAL: Must match authenticated user
        )
    ).first()
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found or access denied")
    
    # SECURITY: Verify step belongs to user and session
    step = db.query(ExamStep).filter(
        and_(
            ExamStep.id == step_id,
            ExamStep.patient_id == current_user.id,  # CRITICAL: Must match authenticated user
            ExamStep.session_id == session.id  # CRITICAL: Must match verified session
        )
    ).first()
    
    if not step:
        raise HTTPException(status_code=404, detail="Step not found or access denied")
    
    # Upload media to S3
    s3_key = None
    if image_file:
        file_data = await image_file.read()
        s3_key = await upload_file_to_s3(
            file_data=file_data,
            filename=f"exam_{session.id}_step_{step.step_number}.jpg",
            content_type="image/jpeg"
        )
    elif video_file:
        file_data = await video_file.read()
        s3_key = await upload_file_to_s3(
            file_data=file_data,
            filename=f"exam_{session.id}_step_{step.step_number}.mp4",
            content_type="video/mp4"
        )
    
    # Update step
    step.s3_key = s3_key
    step.completed_at = datetime.now()
    step.is_ready = True
    
    db.commit()
    
    # Update session progress
    session.completed_steps += 1
    
    # Check if session is complete
    if session.completed_steps >= session.total_steps:
        session.status = SessionStatus.COMPLETED
        session.completed_at = datetime.now()
    
    db.commit()
    db.refresh(session)
    
    # Get next step if not complete
    next_step = None
    if session.status == SessionStatus.IN_PROGRESS:
        protocol = EXAM_PROTOCOLS.get(session.exam_type)
        next_step_data = protocol["steps"][session.completed_steps]
        
        # Create next step
        next_step_obj = ExamStep(
            session_id=session.id,
            patient_id=current_user.id,
            step_number=next_step_data["step_number"],
            step_instruction=next_step_data["instruction"],
            step_type=next_step_data["type"],
            coaching_feedback=[],
            feedback_count=0
        )
        
        db.add(next_step_obj)
        db.commit()
        db.refresh(next_step_obj)
        
        next_step = {
            "step_id": next_step_obj.id,
            "step_number": next_step_obj.step_number,
            "instruction": next_step_obj.step_instruction,
            "type": next_step_obj.step_type,
            "hints": next_step_data.get("coaching_hints", [])
        }
    
    return {
        "step_completed": True,
        "session_status": session.status.value,
        "completed_steps": session.completed_steps,
        "total_steps": session.total_steps,
        "next_step": next_step
    }


@router.get("/session/{session_id}", dependencies=[Depends(require_role("patient"))])
async def get_exam_session(
    session_id: int,
    current_user: User = Depends(require_role("patient")),
    db: Session = Depends(get_db)
):
    """
    Get exam session details
    Patient access only - own sessions
    SECURITY: Verify session and all steps belong to authenticated user
    """
    # SECURITY: Verify session belongs to authenticated user
    session = db.query(ExamSession).filter(
        and_(
            ExamSession.id == session_id,
            ExamSession.patient_id == current_user.id  # CRITICAL: Must match authenticated user
        )
    ).first()
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found or access denied")
    
    # SECURITY: Get steps with patient_id verification
    steps = db.query(ExamStep).filter(
        and_(
            ExamStep.session_id == session_id,
            ExamStep.patient_id == current_user.id  # CRITICAL: Double-check patient owns steps
        )
    ).order_by(ExamStep.step_number).all()
    
    return {
        "id": session.id,
        "exam_type": session.exam_type.value,
        "status": session.status.value,
        "total_steps": session.total_steps,
        "completed_steps": session.completed_steps,
        "started_at": session.started_at,
        "completed_at": session.completed_at,
        "steps": [{
            "id": s.id,
            "step_number": s.step_number,
            "instruction": s.step_instruction,
            "type": s.step_type,
            "completed": s.completed_at is not None,
            "feedback_count": s.feedback_count
        } for s in steps]
    }


@router.get("/packets", dependencies=[Depends(require_role("patient"))])
async def get_exam_packets(
    current_user: User = Depends(require_role("patient")),
    db: Session = Depends(get_db)
):
    """
    Get exam packets for patient
    Shows daily followup and pre-consultation packets
    Patient access only - own packets
    """
    packets = db.query(ExamPacket).filter(
        ExamPacket.patient_id == current_user.id
    ).order_by(desc(ExamPacket.exam_date)).limit(30).all()
    
    return {
        "packets": [{
            "id": p.id,
            "packet_type": p.packet_type,
            "exam_date": p.exam_date,
            "exams_included": p.exams_included,
            "total_images": p.total_images,
            "total_videos": p.total_videos,
            "overall_quality": p.overall_quality.value if p.overall_quality else None,
            "doctor_reviewed": p.doctor_reviewed
        } for p in packets]
    }


@router.get("/protocols", dependencies=[Depends(require_role("patient"))])
async def get_exam_protocols(
    current_user: User = Depends(require_role("patient"))
):
    """
    Get available exam protocols
    Patient access - returns list of available exam types
    """
    return {
        "protocols": [{
            "exam_type": exam_type.value,
            "name": protocol["name"],
            "description": protocol["description"],
            "total_steps": len(protocol["steps"])
        } for exam_type, protocol in EXAM_PROTOCOLS.items()]
    }
