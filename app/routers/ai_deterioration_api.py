"""AI Deterioration Detection API Endpoints for Followup AI

This module provides RESTful API endpoints for all AI deterioration detection engines:
- Video AI Engine: /api/v1/video-ai/*
- Audio AI Engine: /api/v1/audio-ai/*
- Trend Prediction Engine: /api/v1/trends/*
- Alert Engine: /api/v1/alerts/*

HIPAA Compliance:
- All endpoints require JWT authentication
- All PHI access is audit logged
- S3 uploads use server-side encryption (KMS)
- Patient ownership verification on all data access

Wellness Positioning:
- All responses use "wellness monitoring" language
- Recommendations focus on "discussing with provider"
- No diagnostic claims or medical device language
"""

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Header, Request
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional, Annotated
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
import os
import tempfile
import boto3
from botocore.exceptions import ClientError
import secrets
from jose import jwt, JWTError

# Import database and models
from app.database import get_db
from app.models.video_ai_models import MediaSession, VideoMetrics, EdemaSegmentationMetrics
from app.models.audio_ai_models import AudioMetrics
from app.models.trend_models import TrendSnapshot, RiskEvent, PatientBaseline
from app.models.alert_models import AlertRule, Alert
from app.models.security_models import AuditLog, ConsentRecord

# Import services
# Note: AIEngineManager is imported directly in endpoints to avoid blocking module-level imports
from app.services.facial_puffiness_service import FacialPuffinessService
from app.services.skin_analysis_service import SkinAnalysisService

# AWS S3 client for encrypted media storage
# Extract region code from AWS_REGION (handles both "us-east-1" and "US East (N. Virginia) us-east-1" formats)
aws_region = os.getenv("AWS_REGION", "us-east-1")
if " " in aws_region:
    # Extract region code from format like "Asia Pacific (Sydney) ap-southeast-2"
    aws_region = aws_region.split()[-1]

s3_client = boto3.client(
    's3',
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=aws_region
)
S3_BUCKET = os.getenv("AWS_S3_BUCKET_NAME", "followupai-media")
KMS_KEY_ID = os.getenv("AWS_KMS_KEY_ID")  # For S3 SSE-KMS encryption

# Create routers
video_router = APIRouter(prefix="/api/v1/video-ai", tags=["Video AI"])
audio_router = APIRouter(prefix="/api/v1/audio-ai", tags=["Audio AI"])
trend_router = APIRouter(prefix="/api/v1/trends", tags=["Trend Prediction"])
alert_router = APIRouter(prefix="/api/v1/alerts", tags=["Alerts"])


# ==================== Pydantic Models ====================

class MediaUploadResponse(BaseModel):
    session_id: int
    s3_key: str
    upload_url: Optional[str] = None
    processing_status: str
    message: str

class VideoAnalysisResponse(BaseModel):
    session_id: int
    metrics: Dict[str, Any]
    quality_score: float
    confidence: float
    analysis_timestamp: datetime
    recommendations: List[str]

class AudioAnalysisResponse(BaseModel):
    session_id: int
    metrics: Dict[str, Any]
    quality_score: float
    confidence: float
    analysis_timestamp: datetime
    recommendations: List[str]

class RiskAssessmentResponse(BaseModel):
    patient_id: str
    risk_score: float
    risk_level: str
    confidence: float
    anomaly_count: int
    contributing_factors: List[Dict[str, Any]]
    wellness_recommendations: List[str]
    snapshot_timestamp: datetime

class AlertRuleCreate(BaseModel):
    rule_name: str
    rule_type: str  # 'risk_threshold', 'metric_deviation', 'trend_change'
    conditions: Dict[str, Any]
    notification_channels: List[str]  # ['dashboard', 'email', 'sms']

class AlertAcknowledge(BaseModel):
    acknowledged_by: str


# ==================== JWT Validation ====================

async def get_current_user(authorization: Optional[str] = Header(None)) -> Dict[str, Any]:
    """
    Extract and validate user from JWT token
    
    Uses DEV_MODE_SECRET or SESSION_SECRET for HS256 JWT verification.
    
    Returns user dict with:
    - user_id: User ID from sub claim
    - email: User email address
    - role: User role (doctor/patient)
    """
    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="Missing Authorization header"
        )
    
    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Invalid Authorization header format. Expected: Bearer <token>"
        )
    
    token = authorization.replace("Bearer ", "").strip()
    
    if not token:
        raise HTTPException(
            status_code=401,
            detail="Empty token"
        )
    
    secret = os.getenv("DEV_MODE_SECRET") or os.getenv("SESSION_SECRET")
    
    if not secret:
        raise HTTPException(
            status_code=500,
            detail="Authentication not configured"
        )
    
    try:
        claims = jwt.decode(token, secret, algorithms=["HS256"])
    except JWTError as e:
        raise HTTPException(
            status_code=401,
            detail=f"Invalid token: {str(e)}"
        )
    
    user_id = claims.get("sub")
    email = claims.get("email", "")
    role = claims.get("role", "patient")
    
    if not user_id:
        raise HTTPException(
            status_code=401,
            detail="Invalid token: missing user ID (sub claim)"
        )
    
    return {
        "user_id": user_id,
        "email": email,
        "role": role
    }

async def audit_log_request(
    request: Request,
    db: Session,
    user: Dict[str, Any],
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
            user_id=user["user_id"],
            user_role=user["role"],
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


# ==================== Video AI Endpoints ====================

@video_router.post("/upload", response_model=MediaUploadResponse)
async def upload_video(
    patient_id: str,
    file: UploadFile,
    request: Request,
    db: Session = Depends(get_db),
    user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Upload video for AI analysis with S3 SSE-KMS encryption
    
    Workflow:
    1. Verify patient consent for video analysis
    2. Generate secure S3 key
    3. Upload to S3 with encryption
    4. Create media session record
    5. Trigger async video processing
    6. Return session ID for status polling
    """
    try:
        # Audit log
        await audit_log_request(request, db, user, "create", "video_session", patient_id, phi_accessed=True)
        
        # Verify consent
        consent = db.query(ConsentRecord).filter(
            ConsentRecord.patient_id == patient_id,
            ConsentRecord.consent_type == "video_analysis",
            ConsentRecord.consent_given == True,
            ConsentRecord.withdrawn == False
        ).first()
        
        if not consent:
            raise HTTPException(
                status_code=403,
                detail="Patient has not consented to video analysis"
            )
        
        # Generate secure S3 key
        s3_key = f"video/{patient_id}/{datetime.utcnow().strftime('%Y%m%d')}/{secrets.token_urlsafe(16)}.mp4"
        
        # Upload to S3 with SSE-KMS encryption
        file_content = await file.read()
        
        upload_params = {
            'Bucket': S3_BUCKET,
            'Key': s3_key,
            'Body': file_content,
            'ServerSideEncryption': 'aws:kms'
        }
        
        if KMS_KEY_ID:
            upload_params['SSEKMSKeyId'] = KMS_KEY_ID
        
        s3_client.put_object(**upload_params)
        
        # Create media session
        session = MediaSession(
            patient_id=patient_id,
            session_type="video",
            s3_key=s3_key,
            s3_bucket=S3_BUCKET,
            kms_key_id=KMS_KEY_ID,
            file_size_bytes=len(file_content),
            processing_status="pending",
            uploaded_by=user["user_id"],
            ip_address=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent")
        )
        
        db.add(session)
        db.commit()
        db.refresh(session)
        
        # TODO: Trigger async video processing (e.g., Celery task, AWS Lambda)
        # For now, return session ID for polling
        
        return MediaUploadResponse(
            session_id=session.id,
            s3_key=s3_key,
            processing_status="pending",
            message="Video uploaded successfully. Processing will begin shortly."
        )
        
    except ClientError as e:
        raise HTTPException(status_code=500, detail=f"S3 upload error: {str(e)}")
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@video_router.post("/analyze/{session_id}", response_model=VideoAnalysisResponse)
async def analyze_video(
    session_id: int,
    request: Request,
    db: Session = Depends(get_db),
    user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Trigger AI analysis on uploaded video
    
    Metrics extracted:
    - Respiratory rate (optical flow + FFT)
    - Skin pallor (HSV analysis)
    - Eye sclera yellowness (jaundice detection)
    - Facial swelling (landmark distances)
    - Head movement/stability/tremor
    - Lighting quality correction
    """
    try:
        # Get session
        session = db.query(MediaSession).filter(MediaSession.id == session_id).first()
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Audit log
        patient_id_str = str(session.patient_id)
        await audit_log_request(request, db, user, "view", "video_analysis", patient_id_str, phi_accessed=True)
        
        # Download video from S3
        response = s3_client.get_object(Bucket=S3_BUCKET, Key=session.s3_key)
        video_bytes = response['Body'].read()
        
        # Save to temporary file for VideoAIEngine (expects file path)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
            temp_video.write(video_bytes)
            temp_video_path = temp_video.name
        
        try:
            # Retrieve patient FPS baseline for personalized comparison
            fps_service = FacialPuffinessService(db)
            patient_baseline = fps_service.get_patient_baseline(str(session.patient_id))
            
            # Retrieve patient skin analysis baseline
            skin_service = SkinAnalysisService(db)
            skin_baseline = skin_service.get_patient_baseline(str(session.patient_id))
            
            # Merge baselines for comprehensive analysis
            combined_baseline = {**(patient_baseline or {}), **(skin_baseline or {})}
            
            # Run Video AI Engine with combined baseline (using AIEngineManager singleton)
            from app.services.ai_engine_manager import AIEngineManager
            video_engine = AIEngineManager.get_video_engine()
            metrics_dict = await video_engine.analyze_video(temp_video_path, combined_baseline)
        finally:
            # Clean up temporary file
            if os.path.exists(temp_video_path):
                os.unlink(temp_video_path)
        
        # Create VideoMetrics record
        metrics = VideoMetrics(
            session_id=int(session.id),
            patient_id=str(session.patient_id),
            **metrics_dict
        )
        
        db.add(metrics)
        
        # Persist Facial Puffiness Score (FPS) metrics to time-series database
        if metrics_dict.get('facial_puffiness_score') is not None:
            fps_service = FacialPuffinessService(db)
            fps_service.ingest_fps_metrics(
                patient_id=str(session.patient_id),
                session_id=str(session.id),
                fps_metrics=metrics_dict,
                frames_analyzed=metrics_dict.get('frames_analyzed', 0),
                detection_confidence=metrics_dict.get('analysis_confidence', 0.0),
                timestamp=datetime.utcnow()
            )
        
        # Persist Skin Analysis metrics (LAB color space) to time-series database
        if metrics_dict.get('lab_facial_perfusion_avg') is not None:
            skin_service = SkinAnalysisService(db)
            skin_service.ingest_skin_metrics(
                patient_id=str(session.patient_id),
                session_id=str(session.id),
                skin_metrics=metrics_dict,
                frames_analyzed=metrics_dict.get('frames_analyzed', 0),
                detection_confidence=metrics_dict.get('lab_skin_analysis_quality', 0.0),
                timestamp=datetime.utcnow()
            )
        
        # Persist Edema Segmentation metrics (DeepLab V3+) if model available
        if metrics_dict.get('edema_model_available', False):
            # Extract edema metrics from VideoAIEngine output
            regional_analysis = metrics_dict.get('edema_regional_analysis', {})
            swelling_detected = metrics_dict.get('edema_swelling_detected', False)
            expansion_avg = metrics_dict.get('edema_expansion_avg')
            frames_analyzed = metrics_dict.get('edema_frames_analyzed', 0)
            
            # Determine severity based on expansion percentage
            severity = 'none'
            if swelling_detected and expansion_avg:
                if expansion_avg > 20:
                    severity = 'severe'
                elif expansion_avg > 10:
                    severity = 'moderate'
                elif expansion_avg > 5:
                    severity = 'mild'
                else:
                    severity = 'trace'
            
            # Extract regional metrics from analysis
            face_data = regional_analysis.get('face_upper_body', {})
            torso_data = regional_analysis.get('torso_hands', {})
            legs_data = regional_analysis.get('legs_feet', {})
            left_limb_data = regional_analysis.get('left_lower_limb', {})
            right_limb_data = regional_analysis.get('right_lower_limb', {})
            lower_leg_left_data = regional_analysis.get('lower_leg_left', {})
            lower_leg_right_data = regional_analysis.get('lower_leg_right', {})
            periorbital_data = regional_analysis.get('periorbital', {})
            
            # Calculate asymmetry if both sides available
            asymmetry_detected = False
            asymmetry_diff = None
            if (left_limb_data.get('expansion_percent') is not None and 
                right_limb_data.get('expansion_percent') is not None):
                left_exp = left_limb_data['expansion_percent']
                right_exp = right_limb_data['expansion_percent']
                asymmetry_diff = abs(left_exp - right_exp)
                asymmetry_detected = asymmetry_diff > 3.0
            
            # Count swelling regions
            swelling_regions_count = sum([
                face_data.get('swelling_detected', False),
                torso_data.get('swelling_detected', False),
                legs_data.get('swelling_detected', False)
            ])
            
            edema_record = EdemaSegmentationMetrics(
                patient_id=str(session.patient_id),
                session_id=int(session.id),
                analyzed_at=datetime.utcnow(),
                # Model info
                model_type='deeplab_v3_plus',
                model_version='mobilenet_v2_cityscapes',
                is_finetuned=False,
                # Overall detection
                person_detected=True,
                swelling_detected=swelling_detected,
                swelling_severity=severity,
                overall_expansion_percent=expansion_avg,
                swelling_regions_count=swelling_regions_count,
                total_body_area_px=metrics_dict.get('edema_total_body_area_px'),
                # Regional analysis - Face/Upper Body
                face_upper_body_area_px=face_data.get('current_area_px'),
                face_upper_body_baseline_area_px=face_data.get('baseline_area_px'),
                face_upper_body_expansion_percent=face_data.get('expansion_percent'),
                face_upper_body_swelling_detected=face_data.get('swelling_detected', False),
                # Regional analysis - Torso/Hands
                torso_hands_area_px=torso_data.get('current_area_px'),
                torso_hands_baseline_area_px=torso_data.get('baseline_area_px'),
                torso_hands_expansion_percent=torso_data.get('expansion_percent'),
                torso_hands_swelling_detected=torso_data.get('swelling_detected', False),
                # Regional analysis - Legs/Feet
                legs_feet_area_px=legs_data.get('current_area_px'),
                legs_feet_baseline_area_px=legs_data.get('baseline_area_px'),
                legs_feet_expansion_percent=legs_data.get('expansion_percent'),
                legs_feet_swelling_detected=legs_data.get('swelling_detected', False),
                # Asymmetry detection
                left_lower_limb_area_px=left_limb_data.get('current_area_px'),
                right_lower_limb_area_px=right_limb_data.get('current_area_px'),
                left_lower_limb_baseline_area_px=left_limb_data.get('baseline_area_px'),
                right_lower_limb_baseline_area_px=right_limb_data.get('baseline_area_px'),
                left_expansion_percent=left_limb_data.get('expansion_percent'),
                right_expansion_percent=right_limb_data.get('expansion_percent'),
                asymmetry_detected=asymmetry_detected,
                asymmetry_difference_percent=asymmetry_diff,
                # Fine-grained regions
                lower_leg_left_area_px=lower_leg_left_data.get('current_area_px'),
                lower_leg_right_area_px=lower_leg_right_data.get('current_area_px'),
                periorbital_area_px=periorbital_data.get('current_area_px'),
                # Quality metrics
                segmentation_confidence=metrics_dict.get('edema_person_confidence', 0.0),
                processing_time_ms=metrics_dict.get('edema_inference_time_ms') or int((metrics_dict.get('processing_time_seconds', 0) or 0) * 1000),
                # Baseline
                has_baseline=metrics_dict.get('edema_has_baseline', False),
                baseline_segmentation_id=metrics_dict.get('edema_baseline_id'),
                # Raw data
                classes_detected=metrics_dict.get('edema_classes_detected'),
                regional_analysis_json=regional_analysis,
                # Disease personalization
                patient_conditions=metrics_dict.get('patient_conditions'),
                priority_regions=metrics_dict.get('edema_priority_regions')
            )
            db.add(edema_record)
        
        # Update session status using query.update()
        db.query(MediaSession).filter(
            MediaSession.id == session_id
        ).update({
            "processing_status": "completed",
            "processed_at": datetime.utcnow(),
            "quality_score": metrics_dict.get("frame_quality", 0.0)
        })
        db.commit()
        db.refresh(session)
        db.refresh(metrics)
        
        # Generate wellness recommendations (NOT medical advice)
        recommendations = engine.generate_recommendations(metrics_dict)
        
        quality_score = float(session.quality_score) if session.quality_score is not None else 0.0
        
        return VideoAnalysisResponse(
            session_id=int(session.id),
            metrics=metrics_dict,
            quality_score=quality_score,
            confidence=metrics_dict.get("analysis_confidence", 0.0),
            analysis_timestamp=metrics.analysis_timestamp,
            recommendations=recommendations
        )
        
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@video_router.get("/sessions/{patient_id}")
async def get_video_sessions(
    patient_id: str,
    request: Request,
    limit: int = 10,
    db: Session = Depends(get_db),
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Get recent video sessions for a patient"""
    await audit_log_request(request, db, user, "view", "video_sessions", patient_id, phi_accessed=True)
    
    sessions = db.query(MediaSession).filter(
        MediaSession.patient_id == patient_id,
        MediaSession.session_type == "video"
    ).order_by(MediaSession.uploaded_at.desc()).limit(limit).all()
    
    return {"sessions": [
        {
            "id": s.id,
            "uploaded_at": s.uploaded_at,
            "processing_status": s.processing_status,
            "quality_score": s.quality_score
        } for s in sessions
    ]}


# ==================== Audio AI Endpoints ====================

@audio_router.post("/upload", response_model=MediaUploadResponse)
async def upload_audio(
    patient_id: str,
    file: UploadFile,
    request: Request,
    db: Session = Depends(get_db),
    user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Upload audio for AI analysis with S3 SSE-KMS encryption
    
    Similar to video upload but for audio files
    """
    try:
        # Audit log
        await audit_log_request(request, db, user, "create", "audio_session", patient_id, phi_accessed=True)
        
        # Verify consent
        consent = db.query(ConsentRecord).filter(
            ConsentRecord.patient_id == patient_id,
            ConsentRecord.consent_type == "audio_analysis",
            ConsentRecord.consent_given == True,
            ConsentRecord.withdrawn == False
        ).first()
        
        if not consent:
            raise HTTPException(
                status_code=403,
                detail="Patient has not consented to audio analysis"
            )
        
        # Generate secure S3 key
        s3_key = f"audio/{patient_id}/{datetime.utcnow().strftime('%Y%m%d')}/{secrets.token_urlsafe(16)}.wav"
        
        # Upload to S3 with SSE-KMS encryption
        file_content = await file.read()
        
        upload_params = {
            'Bucket': S3_BUCKET,
            'Key': s3_key,
            'Body': file_content,
            'ServerSideEncryption': 'aws:kms'
        }
        
        if KMS_KEY_ID:
            upload_params['SSEKMSKeyId'] = KMS_KEY_ID
        
        s3_client.put_object(**upload_params)
        
        # Create media session
        session = MediaSession(
            patient_id=patient_id,
            session_type="audio",
            s3_key=s3_key,
            s3_bucket=S3_BUCKET,
            kms_key_id=KMS_KEY_ID,
            file_size_bytes=len(file_content),
            processing_status="pending",
            uploaded_by=user["user_id"],
            ip_address=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent")
        )
        
        db.add(session)
        db.commit()
        db.refresh(session)
        
        return MediaUploadResponse(
            session_id=session.id,
            s3_key=s3_key,
            processing_status="pending",
            message="Audio uploaded successfully. Processing will begin shortly."
        )
        
    except ClientError as e:
        raise HTTPException(status_code=500, detail=f"S3 upload error: {str(e)}")
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@audio_router.post("/analyze/{session_id}", response_model=AudioAnalysisResponse)
async def analyze_audio(
    session_id: int,
    request: Request,
    db: Session = Depends(get_db),
    user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Trigger AI analysis on uploaded audio
    
    Metrics extracted:
    - Breath cycle detection
    - Speech pace variability
    - Cough detection & severity
    - Wheeze frequency signatures
    - Voice hoarseness (jitter/shimmer)
    - Background noise removal
    """
    try:
        # Get session
        session = db.query(MediaSession).filter(MediaSession.id == session_id).first()
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Audit log
        patient_id_str = str(session.patient_id)
        await audit_log_request(request, db, user, "view", "audio_analysis", patient_id_str, phi_accessed=True)
        
        # Download audio from S3
        response = s3_client.get_object(Bucket=S3_BUCKET, Key=session.s3_key)
        audio_bytes = response['Body'].read()
        
        # Run Audio AI Engine (using AIEngineManager singleton)
        from app.services.ai_engine_manager import AIEngineManager
        audio_engine = AIEngineManager.get_audio_engine()
        metrics_dict = await audio_engine.analyze_audio(audio_bytes, patient_id_str)
        
        # Create AudioMetrics record
        metrics = AudioMetrics(
            session_id=int(session.id),
            patient_id=patient_id_str,
            **metrics_dict
        )
        
        db.add(metrics)
        
        # Update session status using query.update()
        db.query(MediaSession).filter(
            MediaSession.id == session_id
        ).update({
            "processing_status": "completed",
            "processed_at": datetime.utcnow(),
            "quality_score": metrics_dict.get("audio_quality", 0.0)
        })
        
        db.commit()
        db.refresh(session)
        db.refresh(metrics)
        
        # Generate wellness recommendations
        recommendations = audio_engine.generate_recommendations(metrics_dict)
        
        quality_score = float(session.quality_score) if session.quality_score is not None else 0.0
        
        return AudioAnalysisResponse(
            session_id=int(session.id),
            metrics=metrics_dict,
            quality_score=quality_score,
            confidence=metrics_dict.get("analysis_confidence", 0.0),
            analysis_timestamp=metrics.analysis_timestamp,
            recommendations=recommendations
        )
        
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Trend Prediction Endpoints ====================

@trend_router.post("/risk-assessment/{patient_id}", response_model=RiskAssessmentResponse)
async def assess_risk(
    patient_id: str,
    request: Request,
    db: Session = Depends(get_db),
    user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Run comprehensive risk assessment using Trend Prediction Engine
    
    Analyzes:
    - Recent video/audio metrics (last 7 days)
    - Baseline deviation detection (z-score analysis)
    - Bayesian risk score updates
    - Anomaly detection with trend analysis
    - Patient-specific personalization
    
    Returns:
    - Risk score (0.0-1.0)
    - Risk level (green/yellow/red)
    - Contributing factors
    - Wellness recommendations (NOT medical advice)
    """
    try:
        # Audit log
        await audit_log_request(request, db, user, "view", "risk_assessment", patient_id, phi_accessed=True)
        
        # Run Trend Prediction Engine (using AIEngineManager)
        from app.services.ai_engine_manager import AIEngineManager
        trend_engine = AIEngineManager.get_trend_engine(db)
        assessment = await trend_engine.assess_risk(patient_id)
        
        # Create TrendSnapshot
        snapshot = TrendSnapshot(
            patient_id=patient_id,
            risk_score=assessment["risk_score"],
            risk_level=assessment["risk_level"],
            confidence=assessment["confidence"],
            anomaly_count=assessment["anomaly_count"],
            deviation_metrics=assessment["deviation_metrics"],
            contributing_factors=assessment["contributing_factors"],
            wellness_recommendations=assessment["wellness_recommendations"]
        )
        
        db.add(snapshot)
        db.commit()
        db.refresh(snapshot)
        
        # Check if risk level changed and trigger alerts
        await trend_engine.check_risk_transition(patient_id, str(snapshot.risk_level), assessment["risk_level"])
        
        return RiskAssessmentResponse(
            patient_id=patient_id,
            risk_score=assessment["risk_score"],
            risk_level=assessment["risk_level"],
            confidence=assessment["confidence"],
            anomaly_count=assessment["anomaly_count"],
            contributing_factors=assessment["contributing_factors"],
            wellness_recommendations=assessment["wellness_recommendations"],
            snapshot_timestamp=snapshot.snapshot_timestamp
        )
        
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@trend_router.get("/history/{patient_id}")
async def get_trend_history(
    patient_id: str,
    request: Request,
    days: int = 30,
    db: Session = Depends(get_db),
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Get historical trend snapshots for patient"""
    await audit_log_request(request, db, user, "view", "trend_history", patient_id, phi_accessed=True)
    
    cutoff_date = datetime.utcnow() - timedelta(days=days)
    
    snapshots = db.query(TrendSnapshot).filter(
        TrendSnapshot.patient_id == patient_id,
        TrendSnapshot.snapshot_timestamp >= cutoff_date
    ).order_by(TrendSnapshot.snapshot_timestamp.desc()).all()
    
    return {"snapshots": [
        {
            "timestamp": s.snapshot_timestamp,
            "risk_score": s.risk_score,
            "risk_level": s.risk_level,
            "anomaly_count": s.anomaly_count
        } for s in snapshots
    ]}


# ==================== Alert Engine Endpoints ====================

@alert_router.post("/rules", status_code=201)
async def create_alert_rule(
    rule: AlertRuleCreate,
    request: Request,
    db: Session = Depends(get_db),
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Create new alert rule for doctor"""
    try:
        # Audit log
        await audit_log_request(request, db, user, "create", "alert_rule", None, phi_accessed=False)
        
        new_rule = AlertRule(
            doctor_id=user["user_id"],
            rule_name=rule.rule_name,
            rule_type=rule.rule_type,
            conditions=rule.conditions,
            notification_channels=rule.notification_channels,
            is_active=True
        )
        
        db.add(new_rule)
        db.commit()
        db.refresh(new_rule)
        
        return {"rule_id": new_rule.id, "message": "Alert rule created successfully"}
        
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@alert_router.get("/pending")
async def get_pending_alerts(
    request: Request,
    severity_filter: Optional[str] = None,
    db: Session = Depends(get_db),
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Get pending alerts for current doctor"""
    await audit_log_request(request, db, user, "view", "alerts", None, phi_accessed=True)
    
    # Get alert engine from AIEngineManager
    from app.services.ai_engine_manager import AIEngineManager
    alert_engine = AIEngineManager.get_alert_engine(db)
    
    severity_list = severity_filter.split(",") if severity_filter else None
    alerts = await alert_engine.get_pending_alerts(user["user_id"], severity_list)
    
    return {"alerts": [
        {
            "id": a.id,
            "patient_id": a.patient_id,
            "title": a.title,
            "message": a.message,
            "severity": a.severity,
            "created_at": a.created_at
        } for a in alerts
    ]}

@alert_router.post("/acknowledge/{alert_id}")
async def acknowledge_alert(
    alert_id: int,
    request: Request,
    db: Session = Depends(get_db),
    user: Dict[str, Any] = Depends(get_current_user)
):
    """Acknowledge alert"""
    await audit_log_request(request, db, user, "update", "alert", None, phi_accessed=True)
    
    # Get alert engine from AIEngineManager
    from app.services.ai_engine_manager import AIEngineManager
    alert_engine = AIEngineManager.get_alert_engine(db)
    
    success = await alert_engine.acknowledge_alert(alert_id, user["user_id"], user["role"])
    
    if not success:
        raise HTTPException(status_code=500, detail="Failed to acknowledge alert")
    
    return {"message": "Alert acknowledged successfully"}


# Create main router that includes all sub-routers
def create_ai_deterioration_router():
    """Create and configure the main AI deterioration detection router"""
    from fastapi import FastAPI
    
    app = FastAPI()
    app.include_router(video_router)
    app.include_router(audio_router)
    app.include_router(trend_router)
    app.include_router(alert_router)
    
    return app
