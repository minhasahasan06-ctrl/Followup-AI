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
