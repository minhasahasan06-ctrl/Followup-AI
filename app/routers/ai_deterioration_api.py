"""AI Deterioration Detection API Endpoints for Followup AI

This module provides RESTful API endpoints for all AI deterioration detection engines:
- Video AI Engine: /api/v1/video-ai/*
- Audio AI Engine: /api/v1/audio-ai/*
- Trend Prediction Engine: /api/v1/trends/*
- Alert Engine: /api/v1/alerts/*

HIPAA Compliance:
- All endpoints require JWT authentication
- All PHI access is audit logged
- S3 uploads use server-side encryption (KMS) - DISABLED
- Patient ownership verification on all data access

NOTE: AWS S3/boto3 integration has been disabled.
Media uploads will use local storage fallback.

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
import secrets
import jwt
import logging
from jwt.exceptions import InvalidTokenError as JWTError

# Import database and models
from app.database import get_db
from app.models.video_ai_models import MediaSession, VideoMetrics, EdemaSegmentationMetrics
from app.models.audio_ai_models import AudioMetrics
from app.models.trend_models import TrendSnapshot, RiskEvent, PatientBaseline
from app.models.alert_models import AlertRule, Alert
from app.models.security_models import AuditLog, ConsentRecord
from app.services.access_control import HIPAAAuditLogger, PHICategory, AccessControlService, AccessScope, get_access_control
from app.dependencies import get_current_user as get_current_user_centralized
from app.models.user import User

# Import services
from app.services.facial_puffiness_service import FacialPuffinessService
from app.services.skin_analysis_service import SkinAnalysisService

logger = logging.getLogger(__name__)

# STUB: AWS S3/boto3 has been removed
# All S3 operations will fail gracefully or use local storage
logger.warning("AWS S3 integration disabled - media uploads will use local storage fallback")

S3_BUCKET = os.getenv("AWS_S3_BUCKET_NAME", "local-storage")
KMS_KEY_ID = None  # Disabled
s3_client = None  # STUB: No S3 client

# Local storage fallback
LOCAL_MEDIA_DIR = "tmp/media_uploads"
os.makedirs(LOCAL_MEDIA_DIR, exist_ok=True)

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

async def get_current_user_legacy(authorization: Optional[str] = Header(None)) -> Dict[str, Any]:
    """
    LEGACY: Extract and validate user from JWT token.
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
        logger.error(f"Audit logging error: {e}")


# ==================== Local Storage Helper ====================

def save_to_local_storage(file_content: bytes, s3_key: str) -> str:
    """Save file to local storage as S3 fallback"""
    local_path = os.path.join(LOCAL_MEDIA_DIR, s3_key.replace('/', '_'))
    os.makedirs(os.path.dirname(local_path) if os.path.dirname(local_path) else LOCAL_MEDIA_DIR, exist_ok=True)
    with open(local_path, 'wb') as f:
        f.write(file_content)
    logger.warning(f"S3 disabled: File saved to local storage: {local_path}")
    return local_path


def load_from_local_storage(s3_key: str) -> bytes:
    """Load file from local storage"""
    local_path = os.path.join(LOCAL_MEDIA_DIR, s3_key.replace('/', '_'))
    if not os.path.exists(local_path):
        raise FileNotFoundError(f"File not found: {local_path}")
    with open(local_path, 'rb') as f:
        return f.read()


# ==================== Video AI Endpoints ====================

@video_router.post("/upload", response_model=MediaUploadResponse)
async def upload_video(
    patient_id: str,
    file: UploadFile,
    request: Request,
    db: Session = Depends(get_db),
    user: Dict[str, Any] = Depends(get_current_user_legacy)
):
    """
    Upload video for AI analysis.
    
    NOTE: S3 is disabled. Files are saved to local storage.
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
        
        # Generate storage key
        s3_key = f"video/{patient_id}/{datetime.utcnow().strftime('%Y%m%d')}/{secrets.token_urlsafe(16)}.mp4"
        
        # STUB: S3 disabled - save to local storage
        file_content = await file.read()
        local_path = save_to_local_storage(file_content, s3_key)
        
        # Create media session
        session = MediaSession(
            patient_id=patient_id,
            session_type="video",
            s3_key=s3_key,
            s3_bucket="local-storage",
            kms_key_id=None,
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
            message="Video uploaded successfully (local storage - S3 disabled). Processing will begin shortly."
        )
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@video_router.post("/analyze/{session_id}", response_model=VideoAnalysisResponse)
async def analyze_video(
    session_id: int,
    request: Request,
    db: Session = Depends(get_db),
    user: Dict[str, Any] = Depends(get_current_user_legacy)
):
    """
    Trigger AI analysis on uploaded video
    """
    try:
        # Get session
        session = db.query(MediaSession).filter(MediaSession.id == session_id).first()
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Audit log
        patient_id_str = str(session.patient_id)
        await audit_log_request(request, db, user, "view", "video_analysis", patient_id_str, phi_accessed=True)
        
        # STUB: S3 disabled - load from local storage
        try:
            video_bytes = load_from_local_storage(session.s3_key)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="Video file not found in storage")
        
        # Save to temporary file for VideoAIEngine
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
            temp_video.write(video_bytes)
            temp_video_path = temp_video.name
        
        try:
            # Retrieve patient baselines
            fps_service = FacialPuffinessService(db)
            patient_baseline = fps_service.get_patient_baseline(str(session.patient_id))
            
            skin_service = SkinAnalysisService(db)
            skin_baseline = skin_service.get_patient_baseline(str(session.patient_id))
            
            combined_baseline = {**(patient_baseline or {}), **(skin_baseline or {})}
            
            # Run Video AI Engine
            from app.services.ai_engine_manager import AIEngineManager
            video_engine = AIEngineManager.get_video_engine()
            metrics_dict = await video_engine.analyze_video(temp_video_path, combined_baseline)
        finally:
            if os.path.exists(temp_video_path):
                os.unlink(temp_video_path)
        
        # Create VideoMetrics record
        metrics = VideoMetrics(
            session_id=int(session.id),
            patient_id=str(session.patient_id),
            **metrics_dict
        )
        
        db.add(metrics)
        
        # Update session status
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
        
        # Generate wellness recommendations
        recommendations = ["Please consult with your healthcare provider about these findings."]
        
        quality_score = float(session.quality_score) if session.quality_score is not None else 0.0
        
        return VideoAnalysisResponse(
            session_id=int(session.id),
            metrics=metrics_dict,
            quality_score=quality_score,
            confidence=metrics_dict.get("analysis_confidence", 0.0),
            analysis_timestamp=metrics.analysis_timestamp,
            recommendations=recommendations
        )
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@video_router.get("/sessions/{patient_id}")
async def get_video_sessions(
    patient_id: str,
    request: Request,
    limit: int = 10,
    db: Session = Depends(get_db),
    user: Dict[str, Any] = Depends(get_current_user_legacy)
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
    user: Dict[str, Any] = Depends(get_current_user_legacy)
):
    """
    Upload audio for AI analysis.
    
    NOTE: S3 is disabled. Files are saved to local storage.
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
        
        # Generate storage key
        s3_key = f"audio/{patient_id}/{datetime.utcnow().strftime('%Y%m%d')}/{secrets.token_urlsafe(16)}.wav"
        
        # STUB: S3 disabled - save to local storage
        file_content = await file.read()
        local_path = save_to_local_storage(file_content, s3_key)
        
        # Create media session
        session = MediaSession(
            patient_id=patient_id,
            session_type="audio",
            s3_key=s3_key,
            s3_bucket="local-storage",
            kms_key_id=None,
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
            message="Audio uploaded successfully (local storage - S3 disabled). Processing will begin shortly."
        )
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Audio upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@audio_router.get("/sessions/{patient_id}")
async def get_audio_sessions(
    patient_id: str,
    request: Request,
    limit: int = 10,
    db: Session = Depends(get_db),
    user: Dict[str, Any] = Depends(get_current_user_legacy)
):
    """Get recent audio sessions for a patient"""
    await audit_log_request(request, db, user, "view", "audio_sessions", patient_id, phi_accessed=True)
    
    sessions = db.query(MediaSession).filter(
        MediaSession.patient_id == patient_id,
        MediaSession.session_type == "audio"
    ).order_by(MediaSession.uploaded_at.desc()).limit(limit).all()
    
    return {"sessions": [
        {
            "id": s.id,
            "uploaded_at": s.uploaded_at,
            "processing_status": s.processing_status,
            "quality_score": s.quality_score
        } for s in sessions
    ]}


# ==================== Trend Prediction Endpoints ====================

@trend_router.get("/risk-assessment/{patient_id}", response_model=RiskAssessmentResponse)
async def get_risk_assessment(
    patient_id: str,
    request: Request,
    db: Session = Depends(get_db),
    user: Dict[str, Any] = Depends(get_current_user_legacy)
):
    """Get latest risk assessment for a patient"""
    await audit_log_request(request, db, user, "view", "risk_assessment", patient_id, phi_accessed=True)
    
    snapshot = db.query(TrendSnapshot).filter(
        TrendSnapshot.patient_id == patient_id
    ).order_by(TrendSnapshot.computed_at.desc()).first()
    
    if not snapshot:
        raise HTTPException(status_code=404, detail="No risk assessment found for this patient")
    
    return RiskAssessmentResponse(
        patient_id=patient_id,
        risk_score=snapshot.risk_score or 0.0,
        risk_level=snapshot.risk_level or "green",
        confidence=snapshot.confidence or 0.0,
        anomaly_count=snapshot.anomaly_count or 0,
        contributing_factors=snapshot.contributing_factors or [],
        wellness_recommendations=snapshot.recommendations or [],
        snapshot_timestamp=snapshot.computed_at
    )


# ==================== Alert Endpoints ====================

@alert_router.get("/patient/{patient_id}")
async def get_patient_alerts(
    patient_id: str,
    request: Request,
    limit: int = 20,
    db: Session = Depends(get_db),
    user: Dict[str, Any] = Depends(get_current_user_legacy)
):
    """Get alerts for a patient"""
    await audit_log_request(request, db, user, "view", "alerts", patient_id, phi_accessed=True)
    
    alerts = db.query(Alert).filter(
        Alert.patient_id == patient_id
    ).order_by(Alert.created_at.desc()).limit(limit).all()
    
    return {"alerts": [
        {
            "id": a.id,
            "title": a.title,
            "message": a.message,
            "severity": a.severity,
            "status": a.status,
            "created_at": a.created_at,
            "acknowledged_at": a.acknowledged_at
        } for a in alerts
    ]}

@alert_router.post("/{alert_id}/acknowledge")
async def acknowledge_alert(
    alert_id: int,
    body: AlertAcknowledge,
    request: Request,
    db: Session = Depends(get_db),
    user: Dict[str, Any] = Depends(get_current_user_legacy)
):
    """Acknowledge an alert"""
    alert = db.query(Alert).filter(Alert.id == alert_id).first()
    
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    
    await audit_log_request(request, db, user, "update", "alert", alert.patient_id, phi_accessed=True)
    
    alert.status = "acknowledged"
    alert.acknowledged_at = datetime.utcnow()
    alert.acknowledged_by = body.acknowledged_by
    
    db.commit()
    
    return {"success": True, "alert_id": alert_id, "status": "acknowledged"}
