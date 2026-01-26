"""
Video Consultation Router - HIPAA-Compliant Video Sessions.
Integrates Daily.co for live physical examination monitoring.
"""

from typing import Optional
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel

from app.database import get_db
from app.dependencies import get_current_user
from app.models.user import User
from app.services.daily_video_service import DailyVideoService


router = APIRouter(prefix="/api/consultations", tags=["video-consultations"])


class StartVideoRequest(BaseModel):
    doctor_id: str
    duration_minutes: int = 60
    enable_recording: bool = False


class VideoSessionResponse(BaseModel):
    room_name: str
    room_url: str
    access_token: str
    expires_at: str
    config: dict


@router.post("/{consultation_id}/start-video", response_model=VideoSessionResponse)
async def start_video_consultation(
    consultation_id: int,
    request: StartVideoRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Start a HIPAA-compliant video consultation session.
    
    Creates a Daily.co room and returns access tokens for patient and doctor.
    Requires Daily.co BAA to be signed (set DAILY_API_KEY environment variable).
    """
    try:
        service = DailyVideoService()
        
        user_role = str(getattr(current_user, 'role', None) or "patient")
        user_id = str(current_user.id)
        is_doctor = user_role == "doctor"
        
        patient_id = user_id if not is_doctor else str(request.doctor_id)
        doctor_id = str(request.doctor_id) if not is_doctor else user_id
        
        room_data = service.create_consultation_room(
            patient_id=patient_id,
            doctor_id=doctor_id,
            duration_minutes=request.duration_minutes,
            enable_chat=True,
            enable_recording=request.enable_recording
        )
        
        access_token = room_data["doctor_token"] if is_doctor else room_data["patient_token"]
        
        return {
            "room_name": room_data["room_name"],
            "room_url": room_data["room_url"],
            "access_token": access_token,
            "expires_at": room_data["expires_at"],
            "config": room_data["config"]
        }
        
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Configuration error: {str(e)}. Ensure DAILY_API_KEY is set."
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start video consultation: {str(e)}"
        )


@router.delete("/{consultation_id}/end-video")
async def end_video_consultation(
    consultation_id: int,
    room_name: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    End a video consultation and delete the Daily.co room.
    
    HIPAA best practice: Delete rooms immediately after consultation to minimize data retention.
    Only doctors can end video consultations.
    """
    user_role = str(getattr(current_user, 'role', None) or "patient")
    if user_role != "doctor":
        raise HTTPException(
            status_code=403,
            detail="Only doctors can end video consultations"
        )
    
    try:
        service = DailyVideoService()
        
        success = service.delete_room(room_name)
        
        if not success:
            raise HTTPException(
                status_code=404,
                detail="Room not found or already deleted"
            )
        
        return {
            "message": "Video consultation ended successfully",
            "room_name": room_name
        }
        
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Configuration error: {str(e)}. Ensure DAILY_API_KEY is set."
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to end video consultation: {str(e)}"
        )


@router.get("/{consultation_id}/video-status")
async def get_video_status(
    consultation_id: int,
    room_name: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get the current status of a video consultation room.
    Returns room information and active participants.
    """
    try:
        service = DailyVideoService()
        
        room_info = service.get_room_info(room_name)
        participants = service.get_session_participants(room_name)
        
        return {
            "room_info": room_info,
            "participants": participants,
            "consultation_id": consultation_id
        }
        
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Configuration error: {str(e)}. Ensure DAILY_API_KEY is set."
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get video status: {str(e)}"
        )
