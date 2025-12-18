"""
Video Consultation API Router - Phase 12
=========================================

Production-grade video consultation endpoints with:
- Doctor video settings management
- Per-appointment video configuration
- Join video visit (Daily or external)
- HIPAA audit logging
"""

from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session
from pydantic import BaseModel, field_validator
from datetime import datetime

from app.database import get_db
from app.dependencies import get_current_user
from app.models.user import User
from app.services.daily_video_service import DailyVideoService, ExternalVideoProvider
from app.services.video_billing_service import (
    VideoSettingsService, AppointmentVideoService, VideoBillingService
)
from app.services.access_control import HIPAAAuditLogger, PHICategory
from app.services.video_session_storage_service import (
    VideoSessionStorageService, StorageType, video_session_storage_service
)
from app.services.openai_vision_service import (
    OpenAIVisionService, ExamType, openai_vision_service
)
from app.models.video_ai_models import VideoExamSession as VideoExamSessionModel

router = APIRouter(prefix="/api/video", tags=["video"])


class VideoSettingsRequest(BaseModel):
    allow_external_video: Optional[bool] = None
    zoom_join_url: Optional[str] = None
    meet_join_url: Optional[str] = None
    default_video_provider: Optional[str] = None
    enable_recording: Optional[bool] = None
    enable_chat: Optional[bool] = None
    max_participants: Optional[int] = None
    
    @field_validator('zoom_join_url')
    @classmethod
    def validate_zoom(cls, v):
        if v and not ExternalVideoProvider.validate_zoom_url(v):
            raise ValueError("Invalid Zoom URL - must be https://zoom.us/...")
        return v
    
    @field_validator('meet_join_url')
    @classmethod
    def validate_meet(cls, v):
        if v and not ExternalVideoProvider.validate_meet_url(v):
            raise ValueError("Invalid Meet URL - must be https://meet.google.com/...")
        return v


class VideoSettingsResponse(BaseModel):
    doctor_id: str
    allow_external_video: bool
    zoom_join_url: Optional[str]
    meet_join_url: Optional[str]
    default_video_provider: str
    enable_recording: bool
    enable_chat: bool
    max_participants: int


class AppointmentVideoConfigRequest(BaseModel):
    video_provider: str = "daily"
    
    @field_validator('video_provider')
    @classmethod
    def validate_provider(cls, v):
        if v not in ["daily", "zoom", "meet"]:
            raise ValueError("Invalid video provider - must be daily, zoom, or meet")
        return v


class JoinVideoResponse(BaseModel):
    provider: str
    room_url: Optional[str] = None
    token: Optional[str] = None
    external_join_url: Optional[str] = None


@router.get("/settings", response_model=VideoSettingsResponse)
async def get_video_settings(
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get video settings for the current doctor"""
    if current_user.role != "doctor":
        raise HTTPException(status_code=403, detail="Doctor access required")
    
    HIPAAAuditLogger.log_phi_access(
        actor_id=current_user.id,
        actor_role="doctor",
        patient_id=None,
        action="read",
        phi_categories=["settings"],
        resource_type="video_settings",
        access_reason="View video settings",
        ip_address=str(request.client.host) if request.client else None
    )
    
    service = VideoSettingsService(db)
    settings = service.get_or_create_settings(current_user.id)
    
    return VideoSettingsResponse(
        doctor_id=settings.doctor_id,
        allow_external_video=settings.allow_external_video or False,
        zoom_join_url=settings.zoom_join_url,
        meet_join_url=settings.meet_join_url,
        default_video_provider=settings.default_video_provider or "daily",
        enable_recording=settings.enable_recording or False,
        enable_chat=settings.enable_chat if settings.enable_chat is not None else True,
        max_participants=settings.max_participants or 2
    )


@router.put("/settings", response_model=VideoSettingsResponse)
async def update_video_settings(
    request: Request,
    body: VideoSettingsRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update video settings for the current doctor"""
    if current_user.role != "doctor":
        raise HTTPException(status_code=403, detail="Doctor access required")
    
    HIPAAAuditLogger.log_phi_access(
        actor_id=current_user.id,
        actor_role="doctor",
        patient_id=None,
        action="update",
        phi_categories=["settings"],
        resource_type="video_settings",
        access_reason="Update video settings",
        ip_address=str(request.client.host) if request.client else None
    )
    
    service = VideoSettingsService(db)
    
    try:
        settings = service.update_settings(
            doctor_id=current_user.id,
            allow_external_video=body.allow_external_video,
            zoom_join_url=body.zoom_join_url,
            meet_join_url=body.meet_join_url,
            default_video_provider=body.default_video_provider,
            enable_recording=body.enable_recording,
            enable_chat=body.enable_chat,
            max_participants=body.max_participants
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    return VideoSettingsResponse(
        doctor_id=settings.doctor_id,
        allow_external_video=settings.allow_external_video or False,
        zoom_join_url=settings.zoom_join_url,
        meet_join_url=settings.meet_join_url,
        default_video_provider=settings.default_video_provider or "daily",
        enable_recording=settings.enable_recording or False,
        enable_chat=settings.enable_chat if settings.enable_chat is not None else True,
        max_participants=settings.max_participants or 2
    )


@router.post("/appointments/{appointment_id}/config")
async def configure_appointment_video(
    appointment_id: str,
    request: Request,
    body: AppointmentVideoConfigRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Configure video provider for an appointment (doctor only)"""
    if current_user.role != "doctor":
        raise HTTPException(status_code=403, detail="Doctor access required")
    
    HIPAAAuditLogger.log_phi_access(
        actor_id=current_user.id,
        actor_role="doctor",
        patient_id=None,
        action="create",
        phi_categories=["appointment"],
        resource_type="appointment_video",
        access_reason=f"Configure video for appointment {appointment_id}",
        ip_address=str(request.client.host) if request.client else None
    )
    
    service = AppointmentVideoService(db)
    
    try:
        config = service.configure_appointment(
            appointment_id=appointment_id,
            doctor_id=current_user.id,
            video_provider=body.video_provider
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    return {
        "appointment_id": config.appointment_id,
        "video_provider": config.video_provider,
        "external_join_url": config.external_join_url,
        "daily_room_name": config.daily_room_name
    }


@router.post("/appointments/{appointment_id}/join", response_model=JoinVideoResponse)
async def join_video_visit(
    appointment_id: str,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Join a video visit for an appointment.
    
    Returns Daily room + token, or external provider URL.
    Only the appointment's doctor or patient can join.
    """
    HIPAAAuditLogger.log_phi_access(
        actor_id=current_user.id,
        actor_role=current_user.role,
        patient_id=current_user.id if current_user.role == "patient" else None,
        action="create",
        phi_categories=["appointment", "video_session"],
        resource_type="video_visit",
        access_reason=f"Join video visit for appointment {appointment_id}",
        ip_address=str(request.client.host) if request.client else None
    )
    
    appt_service = AppointmentVideoService(db)
    config = appt_service.get_config(appointment_id)
    
    if not config:
        settings_service = VideoSettingsService(db)
        doctor_settings = None
        
        config = appt_service.configure_appointment(
            appointment_id=appointment_id,
            doctor_id=current_user.id if current_user.role == "doctor" else "unknown",
            video_provider="daily"
        )
    
    if config.video_provider in ["zoom", "meet"]:
        return JoinVideoResponse(
            provider="external",
            external_join_url=config.external_join_url
        )
    
    daily_service = DailyVideoService()
    
    room_name = DailyVideoService.generate_room_name(appointment_id)
    
    try:
        room_data = daily_service.create_room(
            appointment_id=appointment_id,
            enable_chat=True,
            enable_recording=False
        )
        
        config.daily_room_name = room_data["room_name"]
        config.daily_room_url = room_data["room_url"]
        config.room_created_at = datetime.utcnow()
        db.commit()
        
    except Exception as e:
        if "already exists" not in str(e).lower():
            raise HTTPException(status_code=500, detail=f"Failed to create room: {str(e)}")
    
    is_owner = current_user.role == "doctor"
    user_display_name = "Doctor" if is_owner else "Patient"
    
    try:
        token = daily_service.create_meeting_token(
            room_name=room_name,
            user_id=current_user.id,
            user_name=user_display_name,
            is_owner=is_owner
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create token: {str(e)}")
    
    return JoinVideoResponse(
        provider="daily",
        room_url=config.daily_room_url or f"https://{daily_service.domain}/{room_name}",
        token=token
    )


@router.get("/usage")
async def get_usage_summary(
    request: Request,
    billing_month: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get video usage summary for the current doctor"""
    if current_user.role != "doctor":
        raise HTTPException(status_code=403, detail="Doctor access required")
    
    HIPAAAuditLogger.log_phi_access(
        actor_id=current_user.id,
        actor_role="doctor",
        patient_id=None,
        action="read",
        phi_categories=["billing"],
        resource_type="video_usage",
        access_reason="View video usage summary",
        ip_address=str(request.client.host) if request.client else None
    )
    
    service = VideoBillingService(db)
    return service.get_doctor_usage_summary(current_user.id, billing_month)


@router.get("/invoices")
async def get_invoices(
    request: Request,
    limit: int = 12,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get invoice history for the current doctor"""
    if current_user.role != "doctor":
        raise HTTPException(status_code=403, detail="Doctor access required")
    
    HIPAAAuditLogger.log_phi_access(
        actor_id=current_user.id,
        actor_role="doctor",
        patient_id=None,
        action="read",
        phi_categories=["billing"],
        resource_type="video_invoices",
        access_reason="View video invoices",
        ip_address=str(request.client.host) if request.client else None
    )
    
    service = VideoBillingService(db)
    return service.get_doctor_invoices(current_user.id, limit)


# ===== Video Exam Session Endpoints =====


class CreateExamSessionRequest(BaseModel):
    patient_id: str


class UploadUrlRequest(BaseModel):
    stage: str
    content_type: str = "image/jpeg"
    file_extension: str = "jpg"
    file_size_bytes: Optional[int] = None
    
    @field_validator('stage')
    @classmethod
    def validate_stage(cls, v):
        valid_stages = ["eyes", "palm", "tongue", "lips", "skin", "respiratory", "custom"]
        if v not in valid_stages:
            raise ValueError(f"Invalid stage - must be one of {valid_stages}")
        return v


class CompleteStageRequest(BaseModel):
    stage: str
    s3_key: str
    quality_score: Optional[float] = None


@router.post("/exam-sessions")
async def create_exam_session(
    request: Request,
    body: CreateExamSessionRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create a new video exam session for a patient"""
    if current_user.role not in ["doctor", "patient"]:
        raise HTTPException(status_code=403, detail="Access denied")
    
    if current_user.role == "patient" and current_user.id != body.patient_id:
        raise HTTPException(status_code=403, detail="Patients can only create sessions for themselves")
    
    import uuid
    session_id = str(uuid.uuid4())
    
    new_session = VideoExamSessionModel(
        id=session_id,
        patient_id=body.patient_id,
        status="in_progress"
    )
    
    db.add(new_session)
    db.commit()
    db.refresh(new_session)
    
    HIPAAAuditLogger.log_phi_access(
        actor_id=current_user.id,
        actor_role=current_user.role,
        patient_id=body.patient_id,
        action="create",
        phi_categories=["video_exam"],
        resource_type="video_exam_session",
        access_reason="Create video exam session",
        ip_address=str(request.client.host) if request.client else None
    )
    
    return {
        "session_id": session_id,
        "patient_id": body.patient_id,
        "status": "in_progress",
        "created_at": new_session.created_at.isoformat() if new_session.created_at else None
    }


@router.post("/exam-sessions/{session_id}/upload-url")
async def get_upload_url(
    session_id: str,
    request: Request,
    body: UploadUrlRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get a pre-signed URL for uploading a frame to a video exam session"""
    session = db.query(VideoExamSessionModel).filter(
        VideoExamSessionModel.id == session_id
    ).first()
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if current_user.role == "patient" and current_user.id != session.patient_id:
        raise HTTPException(status_code=403, detail="Access denied")
    
    if session.status != "in_progress":
        raise HTTPException(status_code=400, detail="Session is not in progress")
    
    storage = video_session_storage_service
    result = await storage.generate_upload_url(
        patient_id=session.patient_id,
        session_id=session_id,
        storage_type=StorageType.EXAM_FRAME,
        content_type=body.content_type,
        file_extension=body.file_extension,
        stage=body.stage,
        file_size_bytes=body.file_size_bytes,
        user_id=current_user.id,
        client_ip=str(request.client.host) if request.client else None
    )
    
    return result


@router.post("/exam-sessions/{session_id}/complete-stage")
async def complete_stage(
    session_id: str,
    request: Request,
    body: CompleteStageRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Mark a stage as completed after upload"""
    session = db.query(VideoExamSessionModel).filter(
        VideoExamSessionModel.id == session_id
    ).first()
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if current_user.role == "patient" and current_user.id != session.patient_id:
        raise HTTPException(status_code=403, detail="Access denied")
    
    stage_to_uri_field = {
        "eyes": "eyes_frame_s3_uri",
        "palm": "palm_frame_s3_uri",
        "tongue": "tongue_frame_s3_uri",
        "lips": "lips_frame_s3_uri"
    }
    stage_to_completed_field = {
        "eyes": "eyes_stage_completed",
        "palm": "palm_stage_completed",
        "tongue": "tongue_stage_completed",
        "lips": "lips_stage_completed"
    }
    stage_to_quality_field = {
        "eyes": "eyes_quality_score",
        "palm": "palm_quality_score",
        "tongue": "tongue_quality_score",
        "lips": "lips_quality_score"
    }
    
    if body.stage in stage_to_uri_field:
        setattr(session, stage_to_uri_field[body.stage], body.s3_key)
    if body.stage in stage_to_completed_field:
        setattr(session, stage_to_completed_field[body.stage], True)
    if body.stage in stage_to_quality_field and body.quality_score is not None:
        setattr(session, stage_to_quality_field[body.stage], body.quality_score)
    
    db.commit()
    
    HIPAAAuditLogger.log_phi_access(
        actor_id=current_user.id,
        actor_role=current_user.role,
        patient_id=session.patient_id,
        action="update",
        phi_categories=["video_exam"],
        resource_type="video_exam_session",
        access_reason=f"Complete stage {body.stage}",
        ip_address=str(request.client.host) if request.client else None
    )
    
    return {
        "session_id": session_id,
        "stage": body.stage,
        "completed": True
    }


@router.post("/exam-sessions/{session_id}/complete")
async def complete_session(
    session_id: str,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Mark the entire exam session as completed"""
    session = db.query(VideoExamSessionModel).filter(
        VideoExamSessionModel.id == session_id
    ).first()
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if current_user.role == "patient" and current_user.id != session.patient_id:
        raise HTTPException(status_code=403, detail="Access denied")
    
    session.status = "completed"
    session.completed_at = datetime.utcnow()
    
    completed_count = sum([
        session.eyes_stage_completed or False,
        session.palm_stage_completed or False,
        session.tongue_stage_completed or False,
        session.lips_stage_completed or False
    ])
    total_stages = 4
    
    quality_scores = [
        session.eyes_quality_score,
        session.palm_quality_score,
        session.tongue_quality_score,
        session.lips_quality_score
    ]
    valid_scores = [s for s in quality_scores if s is not None]
    if valid_scores:
        session.overall_quality_score = sum(valid_scores) / len(valid_scores)
    
    db.commit()
    
    HIPAAAuditLogger.log_phi_access(
        actor_id=current_user.id,
        actor_role=current_user.role,
        patient_id=session.patient_id,
        action="update",
        phi_categories=["video_exam"],
        resource_type="video_exam_session",
        access_reason="Complete video exam session",
        ip_address=str(request.client.host) if request.client else None
    )
    
    return {
        "session_id": session_id,
        "status": "completed",
        "completed_stages": completed_count,
        "total_stages": total_stages,
        "overall_quality_score": session.overall_quality_score
    }


@router.get("/exam-sessions/{session_id}")
async def get_exam_session(
    session_id: str,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get details of a video exam session"""
    session = db.query(VideoExamSessionModel).filter(
        VideoExamSessionModel.id == session_id
    ).first()
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if current_user.role == "patient" and current_user.id != session.patient_id:
        raise HTTPException(status_code=403, detail="Access denied")
    
    HIPAAAuditLogger.log_phi_access(
        actor_id=current_user.id,
        actor_role=current_user.role,
        patient_id=session.patient_id,
        action="read",
        phi_categories=["video_exam"],
        resource_type="video_exam_session",
        access_reason="View video exam session",
        ip_address=str(request.client.host) if request.client else None
    )
    
    return {
        "session_id": session.id,
        "patient_id": session.patient_id,
        "status": session.status,
        "current_stage": session.current_stage,
        "stages": {
            "eyes": {
                "completed": session.eyes_stage_completed,
                "quality_score": session.eyes_quality_score,
                "has_frame": bool(session.eyes_frame_s3_uri)
            },
            "palm": {
                "completed": session.palm_stage_completed,
                "quality_score": session.palm_quality_score,
                "has_frame": bool(session.palm_frame_s3_uri)
            },
            "tongue": {
                "completed": session.tongue_stage_completed,
                "quality_score": session.tongue_quality_score,
                "has_frame": bool(session.tongue_frame_s3_uri)
            },
            "lips": {
                "completed": session.lips_stage_completed,
                "quality_score": session.lips_quality_score,
                "has_frame": bool(session.lips_frame_s3_uri)
            }
        },
        "overall_quality_score": session.overall_quality_score,
        "created_at": session.created_at.isoformat() if session.created_at else None,
        "completed_at": session.completed_at.isoformat() if session.completed_at else None
    }


@router.get("/exam-sessions/{session_id}/manifest")
async def get_session_manifest(
    session_id: str,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get manifest of all files in a session with fresh download URLs"""
    session = db.query(VideoExamSessionModel).filter(
        VideoExamSessionModel.id == session_id
    ).first()
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if current_user.role == "patient" and current_user.id != session.patient_id:
        raise HTTPException(status_code=403, detail="Access denied")
    
    storage = video_session_storage_service
    manifest = await storage.get_session_manifest(
        patient_id=session.patient_id,
        session_id=session_id,
        user_id=current_user.id,
        client_ip=str(request.client.host) if request.client else None
    )
    
    return manifest


@router.delete("/exam-sessions/{session_id}")
async def delete_exam_session(
    session_id: str,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Delete a video exam session and all its files (HIPAA deletion)"""
    if current_user.role != "doctor":
        raise HTTPException(status_code=403, detail="Doctor access required")
    
    session = db.query(VideoExamSessionModel).filter(
        VideoExamSessionModel.id == session_id
    ).first()
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    storage = video_session_storage_service
    deletion_result = await storage.delete_session_content(
        patient_id=session.patient_id,
        session_id=session_id,
        user_id=current_user.id,
        client_ip=str(request.client.host) if request.client else None
    )
    
    db.delete(session)
    db.commit()
    
    return {
        "session_id": session_id,
        "deleted": True,
        "files_deleted": deletion_result.get("deleted_objects", 0)
    }


# ===== Vision Analysis Endpoints =====


class AnalyzeImageRequest(BaseModel):
    image_data: str
    exam_type: str = "custom"
    additional_context: Optional[str] = None
    
    @field_validator('exam_type')
    @classmethod
    def validate_exam_type(cls, v):
        valid_types = ["skin", "oral", "joint", "wound", "eye", "palm", "tongue", "lips", "respiratory", "custom"]
        if v not in valid_types:
            raise ValueError(f"Invalid exam type - must be one of {valid_types}")
        return v


class QualityCheckRequest(BaseModel):
    image_data: str


@router.post("/exam-sessions/{session_id}/analyze-image")
async def analyze_exam_image(
    session_id: str,
    request: Request,
    body: AnalyzeImageRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Analyze an exam image using OpenAI Vision for clinical observations"""
    if current_user.role != "doctor":
        raise HTTPException(status_code=403, detail="Doctor access required for AI analysis")
    
    session = db.query(VideoExamSessionModel).filter(
        VideoExamSessionModel.id == session_id
    ).first()
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    vision = openai_vision_service
    result = await vision.analyze_exam_image(
        image_data=body.image_data,
        exam_type=ExamType(body.exam_type),
        session_id=session_id,
        patient_id=session.patient_id,
        user_id=current_user.id,
        client_ip=str(request.client.host) if request.client else None,
        additional_context=body.additional_context
    )
    
    return {
        "session_id": session_id,
        "exam_type": result.exam_type.value,
        "findings": result.findings,
        "severity_score": result.severity_score,
        "confidence_score": result.confidence_score,
        "recommendations": result.recommendations,
        "follow_up_suggested": result.follow_up_suggested,
        "raw_analysis": result.raw_analysis,
        "analyzed_at": result.analyzed_at.isoformat(),
        "model_used": result.model_used,
        "processing_time_ms": result.processing_time_ms
    }


@router.post("/exam-sessions/{session_id}/batch-analyze")
async def batch_analyze_session(
    session_id: str,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Analyze all images in a session using OpenAI Vision"""
    if current_user.role != "doctor":
        raise HTTPException(status_code=403, detail="Doctor access required for AI analysis")
    
    session = db.query(VideoExamSessionModel).filter(
        VideoExamSessionModel.id == session_id
    ).first()
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if session.status != "completed":
        raise HTTPException(status_code=400, detail="Session must be completed before batch analysis")
    
    HIPAAAuditLogger.log_phi_access(
        actor_id=current_user.id,
        actor_role="doctor",
        patient_id=session.patient_id,
        action="batch_analyze",
        phi_categories=["video_exam", "clinical_findings"],
        resource_type="exam_session",
        access_reason="Batch AI analysis of exam session",
        ip_address=str(request.client.host) if request.client else None
    )
    
    stage_analysis = {}
    stages_analyzed = 0
    
    if session.eyes_frame_s3_uri and session.eyes_stage_completed:
        stage_analysis["eyes"] = {
            "completed": True,
            "analysis_pending": True,
            "message": "Analysis requires image data - use individual analyze-image endpoint"
        }
        stages_analyzed += 1
    
    if session.palm_frame_s3_uri and session.palm_stage_completed:
        stage_analysis["palm"] = {
            "completed": True,
            "analysis_pending": True,
            "message": "Analysis requires image data - use individual analyze-image endpoint"
        }
        stages_analyzed += 1
    
    if session.tongue_frame_s3_uri and session.tongue_stage_completed:
        stage_analysis["tongue"] = {
            "completed": True,
            "analysis_pending": True,
            "message": "Analysis requires image data - use individual analyze-image endpoint"
        }
        stages_analyzed += 1
    
    if session.lips_frame_s3_uri and session.lips_stage_completed:
        stage_analysis["lips"] = {
            "completed": True,
            "analysis_pending": True,
            "message": "Analysis requires image data - use individual analyze-image endpoint"
        }
        stages_analyzed += 1
    
    return {
        "session_id": session_id,
        "stages_analyzed": stages_analyzed,
        "stage_analysis": stage_analysis,
        "message": "Use analyze-image endpoint with base64 image data for each stage"
    }


@router.post("/quality-check")
async def check_image_quality(
    request: Request,
    body: QualityCheckRequest,
    current_user: User = Depends(get_current_user)
):
    """Check the quality of an image before capture"""
    if current_user.role not in ["doctor", "patient"]:
        raise HTTPException(status_code=403, detail="Access denied")
    
    vision = openai_vision_service
    result = await vision.assess_image_quality(
        image_data=body.image_data,
        user_id=current_user.id,
        client_ip=str(request.client.host) if request.client else None
    )
    
    return {
        "quality_score": result.quality_score,
        "is_acceptable": result.is_acceptable,
        "issues": result.issues,
        "recommendations": result.recommendations
    }
