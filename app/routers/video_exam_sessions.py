"""
Video Exam Sessions API - Guided live video examination workflow
Handles session lifecycle, segment uploads, and AI analysis coordination
"""

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session
from sqlalchemy import desc
from typing import List, Optional
from datetime import datetime, timedelta
import os
import boto3
from botocore.exceptions import ClientError
import secrets

from app.database import get_db
from app.models import User, VideoExamSession
from app.auth import get_current_user
from app.services.audit_logger import AuditLogger
from pydantic import BaseModel

router = APIRouter(prefix="/api/v1/video-ai/exam-sessions", tags=["video-exam-sessions"])


# Camera Access Audit Endpoint
@router.post("/audit/camera-access")
async def log_camera_access_event(
    status: str = Form(...),  # 'granted' or 'denied'
    exam_type: Optional[str] = Form(None),
    error_message: Optional[str] = Form(None),
    current_user: User = Depends(get_current_user),
):
    """
    HIPAA compliance endpoint: Log camera access attempts
    Frontend must call this when requesting camera access
    """
    try:
        AuditLogger.log_camera_access(
            user_id=current_user.id,
            status=status,
            exam_type=exam_type
        )
        
        return {
            "logged": True,
            "status": status,
            "message": f"Camera access {status} event logged"
        }
    except Exception as e:
        # Don't fail the request if logging fails
        print(f"[AUDIT ERROR] Failed to log camera access: {str(e)}")
        return {
            "logged": False,
            "error": str(e)
        }

# AWS S3 configuration
S3_BUCKET = os.getenv("AWS_S3_BUCKET_NAME", "followup-ai-hipaa-storage")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
KMS_KEY_ID = os.getenv("AWS_KMS_KEY_ID")

# Initialize S3 client
s3_client = boto3.client('s3', region_name=AWS_REGION)


# Request/Response Models
class StartSessionResponse(BaseModel):
    session_id: str
    started_at: str
    status: str
    message: str


class UploadSegmentResponse(BaseModel):
    segment_id: str
    session_id: str
    exam_type: str
    status: str
    message: str


class SessionStatusResponse(BaseModel):
    session_id: str
    status: str
    total_segments: int
    completed_segments: int
    skipped_segments: int
    started_at: str
    completed_at: Optional[str]


class SessionListItem(BaseModel):
    id: str
    started_at: str
    completed_at: Optional[str]
    status: str
    total_segments: int
    completed_segments: int
    skipped_segments: int


# Helper function to parse AWS region
def parse_aws_region(region_str: str) -> str:
    """Extract actual region code from potentially formatted string"""
    if not region_str:
        return 'us-east-1'
    parts = region_str.split()
    for part in parts:
        if '-' in part and len(part) > 5:
            return part
    return region_str.strip()


@router.post("/start", response_model=StartSessionResponse)
async def start_exam_session(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Start a new guided video examination session.
    Creates a session record and initializes workflow.
    """
    try:
        # Create new session
        session = VideoExamSession(
            patient_id=current_user.id,
            status="in_progress",
            total_segments=7,  # 7 exam types
            completed_segments=0,
            skipped_segments=0,
            total_duration_seconds=0
        )
        
        db.add(session)
        db.commit()
        db.refresh(session)
        
        # HIPAA Audit Log
        AuditLogger.log_video_exam_session_started(
            user_id=current_user.id,
            session_id=str(session.id)
        )
        
        return StartSessionResponse(
            session_id=str(session.id),
            started_at=session.started_at.isoformat(),
            status=session.status,
            message="Examination session started successfully"
        )
        
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start examination session: {str(e)}"
        )


@router.post("/upload-segment", response_model=UploadSegmentResponse)
async def upload_exam_segment(
    file: UploadFile = File(...),
    session_id: str = Form(...),
    exam_type: str = Form(...),
    sequence_order: int = Form(...),
    duration_seconds: int = Form(...),
    custom_location: Optional[str] = Form(None),
    custom_description: Optional[str] = Form(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Upload a video segment for a specific examination.
    Encrypts and stores in S3, creates segment record, triggers AI analysis.
    """
    try:
        # Verify session exists and belongs to user
        session = db.query(VideoExamSession).filter(
            VideoExamSession.id == session_id,
            VideoExamSession.patient_id == current_user.id
        ).first()
        
        if not session:
            raise HTTPException(status_code=404, detail="Examination session not found")
        
        if session.status != "in_progress":
            raise HTTPException(status_code=400, detail="Session is not active")
        
        # Read file content
        file_content = await file.read()
        file_size = len(file_content)
        
        # Generate S3 key with timestamp and random suffix
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        random_suffix = secrets.token_hex(8)
        s3_key = f"video-exam-segments/{current_user.id}/{session_id}/{exam_type}_{timestamp}_{random_suffix}.webm"
        
        # Upload to S3 with encryption
        upload_params = {
            "Bucket": S3_BUCKET,
            "Key": s3_key,
            "Body": file_content,
            "ContentType": "video/webm",
            "ServerSideEncryption": "aws:kms" if KMS_KEY_ID else "AES256",
            "Metadata": {
                "patient-id": current_user.id,
                "session-id": session_id,
                "exam-type": exam_type,
                "sequence-order": str(sequence_order),
                "uploaded-at": datetime.utcnow().isoformat()
            }
        }
        
        if KMS_KEY_ID:
            upload_params["SSEKMSKeyId"] = KMS_KEY_ID
        
        s3_client.put_object(**upload_params)
        
        # HIPAA Audit Log - S3 Upload
        AuditLogger.log_s3_operation(
            user_id=current_user.id,
            operation="upload",
            s3_key=s3_key,
            bucket=S3_BUCKET,
            encrypted=True,
            kms_key_id=KMS_KEY_ID,
            status="success"
        )
        
        # Create segment record
        segment = VideoExamSegment(
            session_id=session_id,
            exam_type=exam_type,
            sequence_order=sequence_order,
            skipped=False,
            capture_started_at=datetime.utcnow() - timedelta(seconds=duration_seconds),
            capture_ended_at=datetime.utcnow(),
            duration_seconds=duration_seconds,
            s3_key=s3_key,
            s3_bucket=S3_BUCKET,
            kms_key_id=KMS_KEY_ID,
            file_size_bytes=file_size,
            status="pending",
            custom_location=custom_location,
            custom_description=custom_description,
            uploaded_by=current_user.id
        )
        
        db.add(segment)
        
        # Update session progress
        session.completed_segments += 1
        session.total_duration_seconds += duration_seconds
        session.updated_at = datetime.utcnow()
        
        db.commit()
        db.refresh(segment)
        
        # HIPAA Audit Log - Video Segment Upload
        AuditLogger.log_video_segment_uploaded(
            user_id=current_user.id,
            session_id=session_id,
            segment_id=str(segment.id),
            exam_type=exam_type,
            s3_key=s3_key,
            file_size_bytes=file_size,
            encrypted=True
        )
        
        # TODO: Trigger AI analysis asynchronously
        # This would call the video AI engine to analyze the specific exam type
        
        return UploadSegmentResponse(
            segment_id=str(segment.id),
            session_id=session_id,
            exam_type=exam_type,
            status="uploaded",
            message=f"{exam_type} examination uploaded successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to upload video segment: {str(e)}"
        )


@router.post("/{session_id}/complete")
async def complete_exam_session(
    session_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Complete an examination session.
    Finalizes session, triggers combined analysis if applicable.
    """
    try:
        # Verify session
        session = db.query(VideoExamSession).filter(
            VideoExamSession.id == session_id,
            VideoExamSession.patient_id == current_user.id
        ).first()
        
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Update session status
        session.status = "completed"
        session.completed_at = datetime.utcnow()
        session.updated_at = datetime.utcnow()
        
        db.commit()
        
        # HIPAA Audit Log - Session Completion
        AuditLogger.log_video_exam_session_completed(
            user_id=current_user.id,
            session_id=session_id,
            completed_segments=session.completed_segments,
            skipped_segments=session.skipped_segments,
            total_duration_seconds=session.total_duration_seconds
        )
        
        # TODO: Trigger combined AI analysis across all segments
        
        return {
            "session_id": session_id,
            "status": "completed",
            "message": "Examination session completed successfully",
            "completed_segments": session.completed_segments,
            "total_segments": session.total_segments
        }
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to complete session: {str(e)}"
        )


@router.get("", response_model=List[SessionListItem])
async def get_exam_sessions(
    days: int = 15,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get examination sessions for the current user.
    Default: last 15 days
    """
    try:
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        sessions = db.query(VideoExamSession).filter(
            VideoExamSession.patient_id == current_user.id,
            VideoExamSession.started_at >= cutoff_date
        ).order_by(desc(VideoExamSession.started_at)).all()
        
        return [
            SessionListItem(
                id=str(s.id),
                started_at=s.started_at.isoformat(),
                completed_at=s.completed_at.isoformat() if s.completed_at else None,
                status=s.status,
                total_segments=s.total_segments,
                completed_segments=s.completed_segments,
                skipped_segments=s.skipped_segments
            )
            for s in sessions
        ]
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch sessions: {str(e)}"
        )


@router.get("/{session_id}/status", response_model=SessionStatusResponse)
async def get_session_status(
    session_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get detailed status of an examination session"""
    try:
        session = db.query(VideoExamSession).filter(
            VideoExamSession.id == session_id,
            VideoExamSession.patient_id == current_user.id
        ).first()
        
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return SessionStatusResponse(
            session_id=str(session.id),
            status=session.status,
            total_segments=session.total_segments,
            completed_segments=session.completed_segments,
            skipped_segments=session.skipped_segments,
            started_at=session.started_at.isoformat(),
            completed_at=session.completed_at.isoformat() if session.completed_at else None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get session status: {str(e)}"
        )
