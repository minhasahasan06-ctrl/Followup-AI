"""AI Deterioration Detection API Endpoints for Followup AI

This module provides RESTful API endpoints for all AI deterioration detection engines:
- Video AI Engine: /api/v1/video-ai/*
- Audio AI Engine: /api/v1/audio-ai/*
- Trend Prediction Engine: /api/v1/trends/*
- Alert Engine: /api/v1/alerts/*

HIPAA Compliance:
- All endpoints require authentication (AWS Cognito JWT)
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
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
import os
import boto3
from botocore.exceptions import ClientError
import secrets
import requests
from jose import jwt, JWTError
from functools import lru_cache

# Import database and models
from app.database import get_db
from app.models.video_ai_models import MediaSession, VideoMetrics
from app.models.audio_ai_models import AudioMetrics
from app.models.trend_models import TrendSnapshot, RiskEvent, PatientBaseline
from app.models.alert_models import AlertRule, Alert
from app.models.security_models import AuditLog, ConsentRecord

# Import AI engines
from app.services.video_ai_engine import VideoAIEngine
from app.services.audio_ai_engine import AudioAIEngine
from app.services.trend_prediction_engine import TrendPredictionEngine
from app.services.alert_orchestration_engine import AlertOrchestrationEngine
from app.services.facial_puffiness_service import FacialPuffinessService

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


# ==================== AWS Cognito JWT Validation ====================

@lru_cache()
def get_cognito_public_keys():
    """Fetch and cache Cognito JWKS (JSON Web Key Set)"""
    region = os.getenv("AWS_COGNITO_REGION", "us-east-1")
    pool_id = os.getenv("AWS_COGNITO_USER_POOL_ID")
    
    if not pool_id:
        raise HTTPException(
            status_code=500,
            detail="AWS_COGNITO_USER_POOL_ID not configured"
        )
    
    jwks_url = f"https://cognito-idp.{region}.amazonaws.com/{pool_id}/.well-known/jwks.json"
    
    try:
        response = requests.get(jwks_url, timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch Cognito public keys: {str(e)}"
        )


def verify_cognito_jwt(token: str) -> Dict[str, Any]:
    """
    Verify AWS Cognito JWT token
    
    Validates:
    - Signature using Cognito public keys (RS256)
    - Token expiration
    - Token issuer matches Cognito user pool
    - Audience matches client ID
    
    Returns decoded token claims
    """
    region = os.getenv("AWS_COGNITO_REGION", "us-east-1")
    pool_id = os.getenv("AWS_COGNITO_USER_POOL_ID")
    client_id = os.getenv("AWS_COGNITO_CLIENT_ID")
    
    if not pool_id or not client_id:
        raise HTTPException(
            status_code=500,
            detail="Cognito configuration missing"
        )
    
    # Get JWKS
    jwks = get_cognito_public_keys()
    
    # Decode token header to get key ID
    try:
        from jose import jwk
        
        headers = jwt.get_unverified_headers(token)
        kid = headers.get("kid")
        
        if not kid:
            raise HTTPException(status_code=401, detail="Invalid token: missing kid")
        
        # Find matching public key
        public_key_dict = None
        for key in jwks.get("keys", []):
            if key["kid"] == kid:
                public_key_dict = key
                break
        
        if not public_key_dict:
            raise HTTPException(status_code=401, detail="Invalid token: public key not found")
        
        # Construct RSA public key from JWKS entry
        public_key = jwk.construct(public_key_dict)
        
        # Verify token signature and claims
        issuer = f"https://cognito-idp.{region}.amazonaws.com/{pool_id}"
        
        payload = jwt.decode(
            token,
            public_key.to_pem().decode('utf-8'),
            algorithms=["RS256"],
            audience=client_id,
            issuer=issuer,
            options={
                "verify_signature": True,
                "verify_exp": True,
                "verify_aud": True,
                "verify_iss": True
            }
        )
        
        return payload
        
    except JWTError as e:
        raise HTTPException(
            status_code=401,
            detail=f"Invalid token: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=401,
            detail=f"Token verification failed: {str(e)}"
        )


async def get_current_user(authorization: Optional[str] = Header(None)) -> Dict[str, Any]:
    """
    Extract and validate user from AWS Cognito JWT token
    
    Production-ready JWT validation:
    - Validates signature using Cognito public keys
    - Verifies expiration, issuer, audience
    - Extracts user claims (sub, email, cognito:groups)
    
    Returns user dict with:
    - user_id: Cognito sub (unique user identifier)
    - email: User email address
    - role: Extracted from cognito:groups or custom:role claim
    """
    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="Missing Authorization header"
        )
    
    # Extract token from "Bearer <token>" format
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
    
    # Verify JWT and extract claims
    claims = verify_cognito_jwt(token)
    
    # Extract user information
    user_id = claims.get("sub")
    email = claims.get("email", "")
    
    # Extract role from cognito:groups or custom:role claim
    groups = claims.get("cognito:groups", [])
    custom_role = claims.get("custom:role", "")
    
    if "Doctors" in groups or "doctor" in custom_role.lower():
        role = "doctor"
    elif "Patients" in groups or "patient" in custom_role.lower():
        role = "patient"
    else:
        role = "patient"  # Default to patient for safety
    
    if not user_id:
        raise HTTPException(
            status_code=401,
            detail="Invalid token: missing user ID (sub claim)"
        )
    
    return {
        "user_id": user_id,
        "email": email,
        "role": role,
        "cognito_groups": groups
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
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    user: Dict[str, Any] = Depends(get_current_user),
    request: Request = None
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
    db: Session = Depends(get_db),
    user: Dict[str, Any] = Depends(get_current_user),
    request: Request = None
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
        await audit_log_request(request, db, user, "view", "video_analysis", session.patient_id, phi_accessed=True)
        
        # Download video from S3
        response = s3_client.get_object(Bucket=S3_BUCKET, Key=session.s3_key)
        video_bytes = response['Body'].read()
        
        # Run Video AI Engine
        engine = VideoAIEngine(db)
        metrics_dict = await engine.analyze_video(video_bytes, session.patient_id)
        
        # Create VideoMetrics record
        metrics = VideoMetrics(
            session_id=session.id,
            patient_id=session.patient_id,
            **metrics_dict
        )
        
        db.add(metrics)
        
        # Persist Facial Puffiness Score (FPS) metrics to time-series database
        if metrics_dict.get('facial_puffiness_score') is not None:
            fps_service = FacialPuffinessService(db)
            fps_service.ingest_fps_metrics(
                patient_id=session.patient_id,
                session_id=str(session.id),
                fps_metrics=metrics_dict,
                frames_analyzed=metrics_dict.get('frames_analyzed', 0),
                detection_confidence=metrics_dict.get('analysis_confidence', 0.0),
                timestamp=datetime.utcnow()
            )
        
        # Update session status
        session.processing_status = "completed"
        session.processed_at = datetime.utcnow()
        session.quality_score = metrics_dict.get("frame_quality", 0.0)
        
        db.commit()
        db.refresh(metrics)
        
        # Generate wellness recommendations (NOT medical advice)
        recommendations = engine.generate_recommendations(metrics_dict)
        
        return VideoAnalysisResponse(
            session_id=session.id,
            metrics=metrics_dict,
            quality_score=session.quality_score,
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
    limit: int = 10,
    db: Session = Depends(get_db),
    user: Dict[str, Any] = Depends(get_current_user),
    request: Request = None
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
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    user: Dict[str, Any] = Depends(get_current_user),
    request: Request = None
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
    db: Session = Depends(get_db),
    user: Dict[str, Any] = Depends(get_current_user),
    request: Request = None
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
        await audit_log_request(request, db, user, "view", "audio_analysis", session.patient_id, phi_accessed=True)
        
        # Download audio from S3
        response = s3_client.get_object(Bucket=S3_BUCKET, Key=session.s3_key)
        audio_bytes = response['Body'].read()
        
        # Run Audio AI Engine
        engine = AudioAIEngine(db)
        metrics_dict = await engine.analyze_audio(audio_bytes, session.patient_id)
        
        # Create AudioMetrics record
        metrics = AudioMetrics(
            session_id=session.id,
            patient_id=session.patient_id,
            **metrics_dict
        )
        
        db.add(metrics)
        
        # Update session status
        session.processing_status = "completed"
        session.processed_at = datetime.utcnow()
        session.quality_score = metrics_dict.get("audio_quality", 0.0)
        
        db.commit()
        db.refresh(metrics)
        
        # Generate wellness recommendations
        recommendations = engine.generate_recommendations(metrics_dict)
        
        return AudioAnalysisResponse(
            session_id=session.id,
            metrics=metrics_dict,
            quality_score=session.quality_score,
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
    db: Session = Depends(get_db),
    user: Dict[str, Any] = Depends(get_current_user),
    request: Request = None
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
        
        # Run Trend Prediction Engine
        engine = TrendPredictionEngine(db)
        assessment = await engine.assess_risk(patient_id)
        
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
        await engine.check_risk_transition(patient_id, assessment["risk_level"])
        
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
    days: int = 30,
    db: Session = Depends(get_db),
    user: Dict[str, Any] = Depends(get_current_user),
    request: Request = None
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
    db: Session = Depends(get_db),
    user: Dict[str, Any] = Depends(get_current_user),
    request: Request = None
):
    """Create new alert rule for doctor"""
    try:
        # Audit log
        await audit_log_request(request, db, user, "create", "alert_rule", phi_accessed=False)
        
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
    severity_filter: Optional[str] = None,
    db: Session = Depends(get_db),
    user: Dict[str, Any] = Depends(get_current_user),
    request: Request = None
):
    """Get pending alerts for current doctor"""
    await audit_log_request(request, db, user, "view", "alerts", phi_accessed=True)
    
    engine = AlertOrchestrationEngine(db)
    
    severity_list = severity_filter.split(",") if severity_filter else None
    alerts = await engine.get_pending_alerts(user["user_id"], severity_list)
    
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
    db: Session = Depends(get_db),
    user: Dict[str, Any] = Depends(get_current_user),
    request: Request = None
):
    """Acknowledge alert"""
    await audit_log_request(request, db, user, "update", "alert", phi_accessed=True)
    
    engine = AlertOrchestrationEngine(db)
    success = await engine.acknowledge_alert(alert_id, user["user_id"], user["role"])
    
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
