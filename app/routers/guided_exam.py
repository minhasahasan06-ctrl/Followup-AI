"""
Guided Video Examination API Endpoints

Provides staged video examination workflow with AI-powered analysis:
- Session creation and management
- Frame capture per examination stage (eyes, palm, tongue, lips)
- S3 encrypted storage
- VideoAI analysis integration
- HIPAA-compliant audit logging

Stages:
1. Eyes - Scleral jaundice detection
2. Palm - Palmar pallor analysis
3. Tongue - Color and coating analysis
4. Lips - Hydration and cyanosis detection
"""

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime
import os
import base64
import tempfile
import boto3
from botocore.exceptions import ClientError
import secrets
import cv2
import numpy as np

# Import database and models
from app.database import get_db
from app.models.video_ai_models import VideoExamSession, VideoMetrics, MediaSession
from app.models.security_models import AuditLog
from app.models.user import User

# Import dependencies
from app.dependencies import get_current_user

# Import AI engine
from app.services.video_ai_engine import VideoAIEngine

# AWS S3 setup
aws_region = os.getenv("AWS_REGION", "us-east-1")
if " " in aws_region:
    aws_region = aws_region.split()[-1]

s3_client = boto3.client(
    's3',
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=aws_region
)
S3_BUCKET = os.getenv("AWS_S3_BUCKET_NAME", "followupai-media")
KMS_KEY_ID = os.getenv("AWS_KMS_KEY_ID")

# Create router
router = APIRouter(prefix="/api/v1/guided-exam", tags=["Guided Video Exam"])


# ==================== Dependencies ====================

async def audit_log_request(
    request: Request,
    db: Session,
    user: User,
    action_type: str,
    resource_type: str,
    patient_id: Optional[str] = None,
    phi_accessed: bool = False
):
    """
    Create HIPAA audit log for all PHI access
    """
    try:
        audit = AuditLog(
            user_id=str(user.id),
            user_role=str(user.role) if user.role else "unknown",
            action_type=action_type,
            resource_type=resource_type,
            patient_id_accessed=patient_id,
            phi_accessed=phi_accessed,
            ip_address=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent"),
            request_details={
                "method": request.method,
                "path": str(request.url.path),
                "query_params": dict(request.query_params)
            }
        )
        db.add(audit)
        db.commit()
    except Exception as e:
        # Don't fail request if audit logging fails, but log error
        print(f"Audit logging error: {e}")


# ==================== Pydantic Models ====================

class SessionCreateRequest(BaseModel):
    patient_id: str
    device_info: Optional[Dict[str, Any]] = None

class SessionCreateResponse(BaseModel):
    session_id: str
    status: str
    current_stage: Optional[str]
    prep_time_seconds: int

class FrameCaptureRequest(BaseModel):
    stage: str = Field(..., pattern="^(eyes|palm|tongue|lips)$")
    frame_base64: str

class FrameCaptureResponse(BaseModel):
    success: bool
    stage_completed: bool
    next_stage: Optional[str]
    message: str

class SessionCompleteResponse(BaseModel):
    video_metrics_id: Optional[int]
    analysis_complete: bool
    message: str
    error: Optional[str] = None

class VideoMetricsResponse(BaseModel):
    session_id: str
    patient_id: str
    metrics: Dict[str, Any]
    analyzed_at: datetime


# ==================== Helper Functions ====================

def get_next_stage(current_stage: Optional[str]) -> Optional[str]:
    """Get next stage in examination workflow"""
    stages = ["eyes", "palm", "tongue", "lips"]
    
    if current_stage is None:
        return stages[0]
    
    try:
        current_idx = stages.index(current_stage)
        if current_idx < len(stages) - 1:
            return stages[current_idx + 1]
    except ValueError:
        pass
    
    return None


async def upload_frame_to_s3(
    patient_id: str,
    session_id: str,
    stage: str,
    frame_data: bytes
) -> str:
    """
    Upload frame to S3 with encryption
    Returns S3 URI
    """
    try:
        # Generate S3 key
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        s3_key = f"guided-exam/{patient_id}/{session_id}/{stage}/{timestamp}.jpg"
        
        # Upload with SSE-KMS encryption
        upload_params = {
            'Bucket': S3_BUCKET,
            'Key': s3_key,
            'Body': frame_data,
            'ContentType': 'image/jpeg',
            'ServerSideEncryption': 'AES256',
            'Metadata': {
                'patient-id': patient_id,
                'session-id': session_id,
                'exam-stage': stage,
                'upload-timestamp': timestamp
            }
        }
        
        if KMS_KEY_ID:
            upload_params['ServerSideEncryption'] = 'aws:kms'
            upload_params['SSEKMSKeyId'] = KMS_KEY_ID
        
        s3_client.put_object(**upload_params)
        
        s3_uri = f"s3://{S3_BUCKET}/{s3_key}"
        return s3_uri
        
    except ClientError as e:
        raise HTTPException(
            status_code=500,
            detail=f"S3 upload failed: {str(e)}"
        )


async def download_frame_from_s3(s3_uri: str) -> bytes:
    """Download frame from S3"""
    try:
        # Parse S3 URI: s3://bucket/key
        parts = s3_uri.replace("s3://", "").split("/", 1)
        bucket = parts[0]
        key = parts[1]
        
        response = s3_client.get_object(Bucket=bucket, Key=key)
        return response['Body'].read()
        
    except ClientError as e:
        raise HTTPException(
            status_code=500,
            detail=f"S3 download failed: {str(e)}"
        )


def create_video_from_frames(frame_paths: list, output_path: str, fps: int = 1):
    """
    Combine frames into MP4 video using OpenCV
    
    Args:
        frame_paths: List of paths to frame images
        output_path: Path to save video
        fps: Frames per second (default 1 for static frames)
    """
    if not frame_paths:
        raise ValueError("No frames provided")
    
    # Read first frame to get dimensions
    first_frame = cv2.imread(frame_paths[0])
    if first_frame is None:
        raise ValueError(f"Cannot read frame: {frame_paths[0]}")
    
    height, width, _ = first_frame.shape
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Write frames
    for frame_path in frame_paths:
        frame = cv2.imread(frame_path)
        if frame is not None:
            # Resize if needed
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height))
            video_writer.write(frame)
    
    video_writer.release()


# ==================== API Endpoints ====================

@router.post("/sessions", response_model=SessionCreateResponse)
async def create_exam_session(
    request_body: SessionCreateRequest,
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user)
):
    """
    Create new guided video examination session
    
    Workflow:
    1. Create VideoExamSession record
    2. Set initial stage to 'eyes'
    3. Return session ID for frame capture
    """
    try:
        # Audit log
        await audit_log_request(
            request, db, user, "create", "guided_exam_session",
            request_body.patient_id, phi_accessed=True
        )
        
        # Verify patient access
        user_role = str(user.role) if user.role else ""
        user_id = str(user.id)
        if user_role == "patient" and user_id != request_body.patient_id:
            raise HTTPException(
                status_code=403,
                detail="Cannot create exam session for another patient"
            )
        
        # Create session
        session = VideoExamSession(
            patient_id=request_body.patient_id,
            status='in_progress',
            current_stage='eyes',
            device_info=request_body.device_info,
            prep_time_seconds=30
        )
        
        db.add(session)
        db.commit()
        db.refresh(session)
        
        return SessionCreateResponse(
            session_id=str(session.id),
            status=str(session.status),
            current_stage=str(session.current_stage) if session.current_stage else None,
            prep_time_seconds=int(session.prep_time_seconds)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create exam session: {str(e)}"
        )


@router.get("/sessions/{session_id}")
async def get_exam_session(
    session_id: str,
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user)
):
    """
    Get guided exam session details
    
    Returns full VideoExamSession object with all stage completion status
    """
    try:
        # Query session
        session = db.query(VideoExamSession).filter(
            VideoExamSession.id == session_id
        ).first()
        
        if not session:
            raise HTTPException(
                status_code=404,
                detail=f"Session {session_id} not found"
            )
        
        # Verify patient access
        patient_id_str = str(session.patient_id)
        user_role = str(user.role) if user.role else ""
        user_id = str(user.id)
        if user_role == "patient" and user_id != patient_id_str:
            raise HTTPException(
                status_code=403,
                detail="Cannot access another patient's exam session"
            )
        
        # Audit log
        await audit_log_request(
            request, db, user, "read", "guided_exam_session",
            patient_id_str, phi_accessed=True
        )
        
        # Return session data
        return {
            "session_id": session.id,
            "patient_id": session.patient_id,
            "status": session.status,
            "current_stage": session.current_stage,
            "eyes_stage_completed": session.eyes_stage_completed,
            "palm_stage_completed": session.palm_stage_completed,
            "tongue_stage_completed": session.tongue_stage_completed,
            "lips_stage_completed": session.lips_stage_completed,
            "eyes_quality_score": session.eyes_quality_score,
            "palm_quality_score": session.palm_quality_score,
            "tongue_quality_score": session.tongue_quality_score,
            "lips_quality_score": session.lips_quality_score,
            "overall_quality_score": session.overall_quality_score,
            "video_metrics_id": session.video_metrics_id,
            "prep_time_seconds": session.prep_time_seconds,
            "total_duration_seconds": session.total_duration_seconds,
            "device_info": session.device_info,
            "error_message": session.error_message,
            "created_at": session.created_at,
            "completed_at": session.completed_at
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve session: {str(e)}"
        )


@router.post("/sessions/{session_id}/capture", response_model=FrameCaptureResponse)
async def capture_exam_frame(
    session_id: str,
    request_body: FrameCaptureRequest,
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user)
):
    """
    Capture and upload frame for specific examination stage
    
    Workflow:
    1. Decode base64 frame
    2. Upload to S3 with encryption
    3. Update session stage completion
    4. Move to next stage or mark complete
    """
    try:
        # Query session
        session = db.query(VideoExamSession).filter(
            VideoExamSession.id == session_id
        ).first()
        
        if not session:
            raise HTTPException(
                status_code=404,
                detail=f"Session {session_id} not found"
            )
        
        # Verify patient access
        patient_id_str = str(session.patient_id)
        user_role = str(user.role) if user.role else ""
        user_id = str(user.id)
        if user_role == "patient" and user_id != patient_id_str:
            raise HTTPException(
                status_code=403,
                detail="Cannot capture frame for another patient's session"
            )
        
        # Audit log
        await audit_log_request(
            request, db, user, "create", "guided_exam_frame",
            patient_id_str, phi_accessed=True
        )
        
        # Decode base64 frame
        try:
            frame_bytes = base64.b64decode(request_body.frame_base64)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid base64 frame data: {str(e)}"
            )
        
        # Upload frame to S3
        s3_uri = await upload_frame_to_s3(
            str(session.patient_id),
            str(session.id),
            request_body.stage,
            frame_bytes
        )
        
        # Update session with S3 URI and stage completion using update()
        stage = request_body.stage
        update_dict = {}
        if stage == "eyes":
            update_dict = {
                "eyes_frame_s3_uri": s3_uri,
                "eyes_stage_completed": True,
                "eyes_quality_score": 85.0
            }
        elif stage == "palm":
            update_dict = {
                "palm_frame_s3_uri": s3_uri,
                "palm_stage_completed": True,
                "palm_quality_score": 85.0
            }
        elif stage == "tongue":
            update_dict = {
                "tongue_frame_s3_uri": s3_uri,
                "tongue_stage_completed": True,
                "tongue_quality_score": 85.0
            }
        elif stage == "lips":
            update_dict = {
                "lips_frame_s3_uri": s3_uri,
                "lips_stage_completed": True,
                "lips_quality_score": 85.0
            }
        
        # Determine next stage
        next_stage = get_next_stage(stage)
        update_dict["current_stage"] = next_stage
        
        # Apply updates
        db.query(VideoExamSession).filter(
            VideoExamSession.id == session_id
        ).update(update_dict)
        db.commit()
        db.refresh(session)
        
        return FrameCaptureResponse(
            success=True,
            stage_completed=True,
            next_stage=next_stage,
            message=f"Frame captured for {stage} stage"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to capture frame: {str(e)}"
        )


@router.post("/sessions/{session_id}/complete", response_model=SessionCompleteResponse)
async def complete_exam_session(
    session_id: str,
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user)
):
    """
    Complete guided exam and trigger ML analysis
    
    Workflow:
    1. Verify all stages completed
    2. Download frames from S3
    3. Combine into MP4 video
    4. Call VideoAIEngine.analyze_video()
    5. Create VideoMetrics record with guided_exam_session_id and exam_stage
    6. Link metrics to session
    """
    try:
        # Query session
        session = db.query(VideoExamSession).filter(
            VideoExamSession.id == session_id
        ).first()
        
        if not session:
            raise HTTPException(
                status_code=404,
                detail=f"Session {session_id} not found"
            )
        
        # Verify patient access
        patient_id_str = str(session.patient_id)
        user_role = str(user.role) if user.role else ""
        user_id = str(user.id)
        if user_role == "patient" and user_id != patient_id_str:
            raise HTTPException(
                status_code=403,
                detail="Cannot complete another patient's exam session"
            )
        
        # Audit log
        await audit_log_request(
            request, db, user, "update", "guided_exam_session",
            patient_id_str, phi_accessed=True
        )
        
        # Check if already completed
        status_str = str(session.status)
        if status_str == "completed":
            video_metrics_id_val = int(session.video_metrics_id) if session.video_metrics_id is not None else None
            return SessionCompleteResponse(
                video_metrics_id=video_metrics_id_val,
                analysis_complete=True,
                message="Session already completed"
            )
        
        # CRITICAL ISSUE #2 FIX: Validate all required stages completed
        required_stages = ['eyes', 'palm', 'tongue', 'lips']
        missing_stages = []
        
        # Check each stage completion flag
        eyes_completed = bool(session.eyes_stage_completed)
        palm_completed = bool(session.palm_stage_completed)
        tongue_completed = bool(session.tongue_stage_completed)
        lips_completed = bool(session.lips_stage_completed)
        
        if not eyes_completed:
            missing_stages.append('eyes')
        if not palm_completed:
            missing_stages.append('palm')
        if not tongue_completed:
            missing_stages.append('tongue')
        if not lips_completed:
            missing_stages.append('lips')
        
        if missing_stages:
            # Mark session as failed and save error message
            db.query(VideoExamSession).filter(
                VideoExamSession.id == session_id
            ).update({
                "status": "failed",
                "error_message": f"Incomplete exam: missing stages {', '.join(missing_stages)}"
            })
            db.commit()
            
            raise HTTPException(
                status_code=400,
                detail=f"Cannot complete exam. Missing required stages: {', '.join(missing_stages)}"
            )
        
        # Collect S3 URIs
        frame_uris = []
        eyes_uri = str(session.eyes_frame_s3_uri) if session.eyes_frame_s3_uri else None
        palm_uri = str(session.palm_frame_s3_uri) if session.palm_frame_s3_uri else None
        tongue_uri = str(session.tongue_frame_s3_uri) if session.tongue_frame_s3_uri else None
        lips_uri = str(session.lips_frame_s3_uri) if session.lips_frame_s3_uri else None
        
        if eyes_uri:
            frame_uris.append(("eyes", eyes_uri))
        if palm_uri:
            frame_uris.append(("palm", palm_uri))
        if tongue_uri:
            frame_uris.append(("tongue", tongue_uri))
        if lips_uri:
            frame_uris.append(("lips", lips_uri))
        
        if not frame_uris:
            raise HTTPException(
                status_code=400,
                detail="No frames captured - cannot complete exam"
            )
        
        # Download frames from S3 to temp files
        temp_dir = tempfile.mkdtemp()
        temp_frame_paths = []
        video_path = None
        
        try:
            for stage, s3_uri in frame_uris:
                frame_data = await download_frame_from_s3(s3_uri)
                temp_path = os.path.join(temp_dir, f"{stage}.jpg")
                with open(temp_path, 'wb') as f:
                    f.write(frame_data)
                temp_frame_paths.append(temp_path)
            
            # Create video from frames
            video_path = os.path.join(temp_dir, "exam_video.mp4")
            create_video_from_frames(temp_frame_paths, video_path, fps=1)
            
            # Initialize VideoAI engine
            video_ai = VideoAIEngine()
            
            # Analyze video
            analysis_result = await video_ai.analyze_video(
                video_path=video_path,
                patient_baseline=None  # Could fetch baseline if available
            )
            
            # Create VideoMetrics record
            # Note: session_id field expects MediaSession.id, but we use NULL for guided exams
            metrics = VideoMetrics(
                session_id=None,  # No MediaSession for guided exams
                patient_id=session.patient_id,
                guided_exam_session_id=session.id,
                exam_stage="combined",  # All stages combined
                
                # Respiratory metrics
                respiratory_rate_bpm=analysis_result.get('respiratory_rate_bpm'),
                respiratory_rate_confidence=analysis_result.get('respiratory_rate_confidence'),
                breathing_pattern=analysis_result.get('breathing_pattern'),
                chest_movement_amplitude=analysis_result.get('chest_movement_amplitude'),
                
                # Skin pallor
                skin_pallor_score=analysis_result.get('skin_pallor_score'),
                face_brightness_avg=analysis_result.get('face_brightness_avg'),
                face_saturation_avg=analysis_result.get('face_saturation_avg'),
                pallor_confidence=analysis_result.get('pallor_confidence'),
                
                # Sclera analysis (jaundice)
                sclera_yellowness_score=analysis_result.get('sclera_yellowness_score'),
                jaundice_risk_level=analysis_result.get('jaundice_risk_level'),
                scleral_chromaticity_index=analysis_result.get('scleral_chromaticity_index'),
                scleral_skin_delta=analysis_result.get('scleral_skin_delta'),
                scleral_l_lightness=analysis_result.get('scleral_l_lightness'),
                scleral_a_red_green=analysis_result.get('scleral_a_red_green'),
                scleral_b_yellow_blue=analysis_result.get('scleral_b_yellow_blue'),
                scleral_roi_detected=analysis_result.get('scleral_roi_detected'),
                
                # Conjunctival analysis (anemia)
                conjunctival_pallor_index=analysis_result.get('conjunctival_pallor_index'),
                conjunctival_red_saturation=analysis_result.get('conjunctival_red_saturation'),
                conjunctival_l_lightness=analysis_result.get('conjunctival_l_lightness'),
                conjunctival_a_red_green=analysis_result.get('conjunctival_a_red_green'),
                conjunctival_b_yellow_blue=analysis_result.get('conjunctival_b_yellow_blue'),
                conjunctival_roi_detected=analysis_result.get('conjunctival_roi_detected'),
                
                # Palmar analysis
                palmar_pallor_lab_index=analysis_result.get('palmar_pallor_lab_index'),
                palmar_l_lightness=analysis_result.get('palmar_l_lightness'),
                palmar_a_red_green=analysis_result.get('palmar_a_red_green'),
                palmar_b_yellow_blue=analysis_result.get('palmar_b_yellow_blue'),
                palmar_roi_detected=analysis_result.get('palmar_roi_detected'),
                
                # Tongue analysis
                tongue_color_index=analysis_result.get('tongue_color_index'),
                tongue_color_l=analysis_result.get('tongue_color_l'),
                tongue_color_a=analysis_result.get('tongue_color_a'),
                tongue_color_b=analysis_result.get('tongue_color_b'),
                tongue_coating_detected=analysis_result.get('tongue_coating_detected'),
                tongue_coating_color=analysis_result.get('tongue_coating_color'),
                tongue_roi_detected=analysis_result.get('tongue_roi_detected'),
                
                # Lip analysis
                lip_hydration_score=analysis_result.get('lip_hydration_score'),
                lip_color_l=analysis_result.get('lip_color_l'),
                lip_color_a=analysis_result.get('lip_color_a'),
                lip_color_b=analysis_result.get('lip_color_b'),
                lip_dryness_score=analysis_result.get('lip_dryness_score'),
                lip_cyanosis_detected=analysis_result.get('lip_cyanosis_detected'),
                lip_roi_detected=analysis_result.get('lip_roi_detected'),
                
                # Facial analysis
                facial_swelling_score=analysis_result.get('facial_swelling_score'),
                eye_puffiness_left=analysis_result.get('eye_puffiness_left'),
                eye_puffiness_right=analysis_result.get('eye_puffiness_right'),
                
                # Quality metrics
                face_detection_confidence=analysis_result.get('face_detection_confidence'),
                lighting_quality_score=analysis_result.get('lighting_quality_score'),
                frame_quality_avg=analysis_result.get('frame_quality_avg'),
                
                # Processing metadata
                frames_analyzed=analysis_result.get('frames_analyzed'),
                processing_time_seconds=analysis_result.get('processing_time_seconds'),
                model_version=analysis_result.get('model_version', 'v1.0'),
                
                # Raw metrics
                raw_metrics=analysis_result
            )
            
            db.add(metrics)
            db.commit()
            db.refresh(metrics)
            
            # Update session with completion status
            eyes_quality = float(session.eyes_quality_score) if session.eyes_quality_score is not None else 0.0
            palm_quality = float(session.palm_quality_score) if session.palm_quality_score is not None else 0.0
            tongue_quality = float(session.tongue_quality_score) if session.tongue_quality_score is not None else 0.0
            lips_quality = float(session.lips_quality_score) if session.lips_quality_score is not None else 0.0
            
            quality_scores = [eyes_quality, palm_quality, tongue_quality, lips_quality]
            overall_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
            
            # Calculate duration
            created_at = session.created_at
            if created_at is not None:
                duration = (datetime.utcnow() - created_at).total_seconds()
            else:
                duration = 0.0
            
            # Update using query.update() to avoid Column assignment issues
            db.query(VideoExamSession).filter(
                VideoExamSession.id == session_id
            ).update({
                "status": "completed",
                "video_metrics_id": metrics.id,
                "completed_at": datetime.utcnow(),
                "overall_quality_score": overall_quality,
                "total_duration_seconds": duration
            })
            db.commit()
            
            return SessionCompleteResponse(
                video_metrics_id=int(metrics.id),
                analysis_complete=True,
                message="Exam completed and analyzed successfully"
            )
            
        except Exception as e:
            # CRITICAL: Set terminal status even on failure
            db.rollback()
            
            try:
                db.query(VideoExamSession).filter(
                    VideoExamSession.id == session_id
                ).update({
                    "status": "failed",
                    "error_message": f"ML analysis failed: {str(e)}",
                    "completed_at": datetime.utcnow()
                })
                db.commit()
            except Exception as update_error:
                # Log error but don't fail the exception handling
                print(f"Failed to update session status: {update_error}")
            
            # Re-raise as HTTPException
            raise HTTPException(
                status_code=500,
                detail=f"Analysis failed: {str(e)}"
            )
        
        finally:
            # Clean up temporary video file and directory
            import shutil
            if video_path and os.path.exists(video_path):
                try:
                    os.unlink(video_path)
                except Exception as e:
                    print(f"Failed to delete video file: {e}")
            
            if os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                except Exception as e:
                    print(f"Failed to delete temp directory: {e}")
        
    except HTTPException:
        raise
    except Exception as e:
        # Catch any other unexpected errors
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error: {str(e)}"
        )


@router.get("/sessions/{session_id}/results", response_model=VideoMetricsResponse)
async def get_exam_results(
    session_id: str,
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user)
):
    """
    Get analyzed results for completed guided exam
    
    Returns VideoMetrics with all hepatic/anemia color fields
    
    CRITICAL FIX: Query by guided_exam_session_id for session-specific metrics
    This prevents PHI leakage by ensuring we only return metrics for THIS session
    """
    try:
        # Query session
        session = db.query(VideoExamSession).filter(
            VideoExamSession.id == session_id
        ).first()
        
        if not session:
            raise HTTPException(
                status_code=404,
                detail=f"Session {session_id} not found"
            )
        
        # Verify patient access
        patient_id_str = str(session.patient_id)
        user_role = str(user.role) if user.role else ""
        user_id = str(user.id)
        if user_role == "patient" and user_id != patient_id_str:
            raise HTTPException(
                status_code=403,
                detail="Cannot access another patient's exam results"
            )
        
        # CRITICAL ISSUE #1 FIX: Query by guided_exam_session_id to get session-specific metrics
        # This prevents returning metrics from other sessions for the same patient
        metrics = db.query(VideoMetrics).filter(
            VideoMetrics.guided_exam_session_id == str(session_id)
        ).first()
        
        if not metrics:
            raise HTTPException(
                status_code=404,
                detail="Analysis results not found for this session"
            )
        
        # Audit log
        await audit_log_request(
            request, db, user, "read", "video_metrics",
            patient_id_str, phi_accessed=True
        )
        
        # Build comprehensive metrics response
        metrics_dict = {
            # Scleral metrics (jaundice)
            "scleral_chromaticity_index": metrics.scleral_chromaticity_index,
            "scleral_skin_delta": metrics.scleral_skin_delta,
            "scleral_l_lightness": metrics.scleral_l_lightness,
            "scleral_a_red_green": metrics.scleral_a_red_green,
            "scleral_b_yellow_blue": metrics.scleral_b_yellow_blue,
            "scleral_roi_detected": metrics.scleral_roi_detected,
            "sclera_yellowness_score": metrics.sclera_yellowness_score,
            "jaundice_risk_level": metrics.jaundice_risk_level,
            
            # Conjunctival metrics (anemia)
            "conjunctival_pallor_index": metrics.conjunctival_pallor_index,
            "conjunctival_red_saturation": metrics.conjunctival_red_saturation,
            "conjunctival_l_lightness": metrics.conjunctival_l_lightness,
            "conjunctival_a_red_green": metrics.conjunctival_a_red_green,
            "conjunctival_b_yellow_blue": metrics.conjunctival_b_yellow_blue,
            "conjunctival_roi_detected": metrics.conjunctival_roi_detected,
            
            # Palmar metrics
            "palmar_pallor_lab_index": metrics.palmar_pallor_lab_index,
            "palmar_l_lightness": metrics.palmar_l_lightness,
            "palmar_a_red_green": metrics.palmar_a_red_green,
            "palmar_b_yellow_blue": metrics.palmar_b_yellow_blue,
            "palmar_roi_detected": metrics.palmar_roi_detected,
            
            # Tongue metrics
            "tongue_color_index": metrics.tongue_color_index,
            "tongue_color_l": metrics.tongue_color_l,
            "tongue_color_a": metrics.tongue_color_a,
            "tongue_color_b": metrics.tongue_color_b,
            "tongue_coating_detected": metrics.tongue_coating_detected,
            "tongue_coating_color": metrics.tongue_coating_color,
            "tongue_roi_detected": metrics.tongue_roi_detected,
            
            # Lip metrics
            "lip_hydration_score": metrics.lip_hydration_score,
            "lip_color_l": metrics.lip_color_l,
            "lip_color_a": metrics.lip_color_a,
            "lip_color_b": metrics.lip_color_b,
            "lip_dryness_score": metrics.lip_dryness_score,
            "lip_cyanosis_detected": metrics.lip_cyanosis_detected,
            "lip_roi_detected": metrics.lip_roi_detected,
            
            # Skin and facial metrics
            "skin_pallor_score": metrics.skin_pallor_score,
            "face_brightness_avg": metrics.face_brightness_avg,
            "face_saturation_avg": metrics.face_saturation_avg,
            "facial_swelling_score": metrics.facial_swelling_score,
            "eye_puffiness_left": metrics.eye_puffiness_left,
            "eye_puffiness_right": metrics.eye_puffiness_right,
            
            # Respiratory metrics
            "respiratory_rate_bpm": metrics.respiratory_rate_bpm,
            "respiratory_rate_confidence": metrics.respiratory_rate_confidence,
            "breathing_pattern": metrics.breathing_pattern,
            
            # Quality metrics
            "face_detection_confidence": metrics.face_detection_confidence,
            "lighting_quality_score": metrics.lighting_quality_score,
            "frame_quality_avg": metrics.frame_quality_avg,
            
            # Processing metadata
            "frames_analyzed": metrics.frames_analyzed,
            "processing_time_seconds": metrics.processing_time_seconds,
            "model_version": metrics.model_version,
            
            # Session metadata
            "guided_exam_session_id": metrics.guided_exam_session_id,
            "exam_stage": metrics.exam_stage
        }
        
        return VideoMetricsResponse(
            session_id=session_id,
            patient_id=str(metrics.patient_id),
            metrics=metrics_dict,
            analyzed_at=datetime.fromisoformat(str(metrics.analyzed_at)) if metrics.analyzed_at else datetime.utcnow()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve results: {str(e)}"
        )
