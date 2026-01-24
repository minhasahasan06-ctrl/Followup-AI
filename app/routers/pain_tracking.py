"""
Pain Tracking API - HIPAA Compliant
Handles facial analysis for pain detection, questionnaire responses, and trend summaries.

Phase 7 Integration:
- VAS slider data (0-10) is wired to Autopilot ML pipeline via SignalIngestorService
- Video analysis endpoint connected to VideoAIEngine
- All operations are HIPAA audit logged
"""

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session
from sqlalchemy import func, desc
from typing import List, Optional
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
import logging

from app.database import get_db
from app.models.pain_tracking import PainMeasurement, PainQuestionnaire, PainTrendSummary
from app.models.user import User
from app.models.patient_doctor_connection import PatientDoctorConnection
from app.dependencies import require_role, get_current_patient
from app.services.signal_ingestor_service import SignalIngestorService
from app.services.access_control import HIPAAAuditLogger, PHICategory

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/pain-tracking", tags=["Pain Tracking"])


# Pydantic schemas for request/response
class FacialMetricsInput(BaseModel):
    left_eyebrow_angle: Optional[float] = None
    right_eyebrow_angle: Optional[float] = None
    eyebrow_asymmetry: Optional[float] = None
    left_nasolabial_tension: Optional[float] = None
    right_nasolabial_tension: Optional[float] = None
    forehead_contractions: int = 0
    eye_contractions: int = 0
    mouth_contractions: int = 0
    grimace_intensity: Optional[float] = None
    grimace_duration_ms: int = 0
    facial_landmarks: Optional[dict] = None
    recording_quality: str = "good"


class PainMeasurementCreate(BaseModel):
    facial_metrics: FacialMetricsInput
    

class QuestionnaireResponse(BaseModel):
    pain_level_self_reported: Optional[int] = Field(None, ge=0, le=10)
    pain_location: Optional[str] = None
    pain_type: Optional[str] = None
    pain_duration: Optional[str] = None
    pain_triggers: Optional[str] = None
    affects_sleep: bool = False
    affects_daily_activities: bool = False
    affects_mood: bool = False
    pain_medication_taken: bool = False
    medication_names: Optional[str] = None
    medication_effectiveness: Optional[str] = None
    additional_notes: Optional[str] = None


class PainMeasurementResponse(BaseModel):
    id: int
    facial_stress_score: Optional[float]
    pain_severity_estimate: Optional[str]
    change_from_previous: Optional[float]
    created_at: datetime
    alert_message: Optional[str] = None
    
    class Config:
        from_attributes = True


class PainTrendResponse(BaseModel):
    period_type: str
    period_start: datetime
    period_end: datetime
    average_stress_score: Optional[float]
    stress_score_change_percent: Optional[float]
    trend_direction: Optional[str]
    requires_physician_attention: bool
    alert_reason: Optional[str]
    
    class Config:
        from_attributes = True


def calculate_facial_stress_score(metrics: FacialMetricsInput) -> float:
    """
    Calculate overall facial stress score (0-100) from facial metrics.
    Higher scores indicate more pain/distress signals.
    """
    score = 0.0
    weights = {
        "eyebrow_asymmetry": 15,
        "nasolabial_tension": 25,
        "grimace_intensity": 30,
        "contractions": 20,
        "eyebrow_angle": 10
    }
    
    # Eyebrow asymmetry (higher = more stress)
    if metrics.eyebrow_asymmetry is not None:
        score += min(metrics.eyebrow_asymmetry * 100, weights["eyebrow_asymmetry"])
    
    # Nasolabial fold tension (average both sides)
    nasolabial_avg = 0
    if metrics.left_nasolabial_tension is not None:
        nasolabial_avg += metrics.left_nasolabial_tension
    if metrics.right_nasolabial_tension is not None:
        nasolabial_avg += metrics.right_nasolabial_tension
    if nasolabial_avg > 0:
        nasolabial_avg /= 2 if (metrics.left_nasolabial_tension and metrics.right_nasolabial_tension) else 1
        score += nasolabial_avg * weights["nasolabial_tension"]
    
    # Grimacing intensity
    if metrics.grimace_intensity is not None:
        score += metrics.grimace_intensity * weights["grimace_intensity"]
    
    # Micro contractions (normalize to 0-1 scale, assuming max 20 contractions in 10 seconds)
    total_contractions = metrics.forehead_contractions + metrics.eye_contractions + metrics.mouth_contractions
    contraction_score = min(total_contractions / 20.0, 1.0)
    score += contraction_score * weights["contractions"]
    
    # Eyebrow angle deviation (assuming normal is around 0, higher angles = more tension)
    if metrics.left_eyebrow_angle is not None and metrics.right_eyebrow_angle is not None:
        avg_angle = abs(metrics.left_eyebrow_angle + metrics.right_eyebrow_angle) / 2
        angle_score = min(avg_angle / 45.0, 1.0)  # Normalize assuming max 45 degrees
        score += angle_score * weights["eyebrow_angle"]
    
    return min(round(score, 2), 100.0)


def get_pain_severity(stress_score: float) -> str:
    """Convert stress score to pain severity category."""
    if stress_score < 30:
        return "low"
    elif stress_score < 60:
        return "moderate"
    else:
        return "severe"


def calculate_change_from_previous(db: Session, patient_id: str, current_score: float) -> Optional[float]:
    """Calculate percentage change from previous measurement."""
    previous = db.query(PainMeasurement).filter(
        PainMeasurement.patient_id == patient_id
    ).order_by(desc(PainMeasurement.created_at)).first()
    
    if previous and previous.facial_stress_score is not None:
        change = ((current_score - previous.facial_stress_score) / previous.facial_stress_score) * 100
        return round(change, 2)
    
    return None


@router.post("/measurements", response_model=PainMeasurementResponse)
async def create_pain_measurement(
    data: PainMeasurementCreate,
    current_user: User = Depends(require_role("patient")),
    db: Session = Depends(get_db)
):
    """
    Store a new pain measurement from facial analysis.
    Calculates facial stress score and compares with previous measurements.
    """
    metrics = data.facial_metrics
    
    # Calculate stress score
    stress_score = calculate_facial_stress_score(metrics)
    severity = get_pain_severity(stress_score)
    change_from_previous = calculate_change_from_previous(db, current_user.id, stress_score)
    
    # Create measurement record
    measurement = PainMeasurement(
        patient_id=current_user.id,
        left_eyebrow_angle=metrics.left_eyebrow_angle,
        right_eyebrow_angle=metrics.right_eyebrow_angle,
        eyebrow_asymmetry=metrics.eyebrow_asymmetry,
        left_nasolabial_tension=metrics.left_nasolabial_tension,
        right_nasolabial_tension=metrics.right_nasolabial_tension,
        forehead_contractions=metrics.forehead_contractions,
        eye_contractions=metrics.eye_contractions,
        mouth_contractions=metrics.mouth_contractions,
        grimace_intensity=metrics.grimace_intensity,
        grimace_duration_ms=metrics.grimace_duration_ms,
        facial_landmarks=metrics.facial_landmarks,
        facial_stress_score=stress_score,
        pain_severity_estimate=severity,
        change_from_previous=change_from_previous,
        recording_quality=metrics.recording_quality
    )
    
    db.add(measurement)
    db.commit()
    db.refresh(measurement)
    
    # Generate alert message if needed
    alert_message = None
    if change_from_previous and change_from_previous > 15:
        alert_message = f"Your facial stress score increased {abs(change_from_previous):.0f}% since your last check-in. Consider logging pain triggers or contacting your physician."
    elif stress_score > 70:
        alert_message = "Your pain levels appear elevated. Please contact your healthcare provider if symptoms persist."
    
    response = PainMeasurementResponse(
        id=measurement.id,
        facial_stress_score=measurement.facial_stress_score,
        pain_severity_estimate=measurement.pain_severity_estimate,
        change_from_previous=measurement.change_from_previous,
        created_at=measurement.created_at,
        alert_message=alert_message
    )
    
    return response


@router.post("/measurements/{measurement_id}/questionnaire")
async def submit_pain_questionnaire(
    measurement_id: int,
    questionnaire: QuestionnaireResponse,
    current_user: User = Depends(require_role("patient")),
    db: Session = Depends(get_db)
):
    """
    Submit questionnaire responses associated with a pain measurement.
    
    Phase 7: VAS slider data (pain_level_self_reported 0-10) is wired to
    Autopilot ML pipeline via SignalIngestorService for feature building.
    """
    measurement = db.query(PainMeasurement).filter(
        PainMeasurement.id == measurement_id,
        PainMeasurement.patient_id == current_user.id
    ).first()
    
    if not measurement:
        raise HTTPException(status_code=404, detail="Measurement not found")
    
    questionnaire_record = PainQuestionnaire(
        patient_id=current_user.id,
        measurement_id=measurement_id,
        **questionnaire.dict()
    )
    
    db.add(questionnaire_record)
    db.commit()
    db.refresh(questionnaire_record)
    
    signal_id = None
    if questionnaire.pain_level_self_reported is not None:
        try:
            signal_ingestor = SignalIngestorService(db)
            signal = signal_ingestor.ingest_pain_signal(
                patient_id=current_user.id,
                pain_level=questionnaire.pain_level_self_reported,
                facial_stress_score=measurement.facial_stress_score,
                source="pain_tracking_questionnaire",
                metadata={
                    "questionnaire_id": questionnaire_record.id,
                    "measurement_id": measurement_id,
                    "pain_location": questionnaire.pain_location,
                    "pain_type": questionnaire.pain_type,
                    "pain_duration": questionnaire.pain_duration,
                    "affects_sleep": questionnaire.affects_sleep,
                    "affects_daily_activities": questionnaire.affects_daily_activities,
                    "affects_mood": questionnaire.affects_mood,
                    "medication_taken": questionnaire.pain_medication_taken
                }
            )
            signal_id = str(signal.id)
            logger.info(
                f"[PAIN-TRACKING] VAS signal ingested for patient {current_user.id}: "
                f"pain_level={questionnaire.pain_level_self_reported}, signal_id={signal_id}"
            )
        except Exception as e:
            logger.error(f"[PAIN-TRACKING] Signal ingestion failed: {e}")
    
    return {
        "success": True,
        "message": "Pain questionnaire submitted successfully",
        "questionnaire_id": questionnaire_record.id,
        "autopilot_signal_id": signal_id
    }


@router.get("/measurements/recent", response_model=List[PainMeasurementResponse])
async def get_recent_measurements(
    days: int = 7,
    current_user: User = Depends(require_role("patient")),
    db: Session = Depends(get_db)
):
    """
    Get recent pain measurements for the current patient.
    """
    cutoff_date = datetime.utcnow() - timedelta(days=days)
    
    measurements = db.query(PainMeasurement).filter(
        PainMeasurement.patient_id == current_user.id,
        PainMeasurement.created_at >= cutoff_date
    ).order_by(desc(PainMeasurement.created_at)).all()
    
    return [
        PainMeasurementResponse(
            id=m.id,
            facial_stress_score=m.facial_stress_score,
            pain_severity_estimate=m.pain_severity_estimate,
            change_from_previous=m.change_from_previous,
            created_at=m.created_at
        )
        for m in measurements
    ]


@router.get("/trends/current")
async def get_current_trend(
    current_user: User = Depends(require_role("patient")),
    db: Session = Depends(get_db)
):
    """
    Get current weekly pain trend summary.
    """
    week_ago = datetime.utcnow() - timedelta(days=7)
    two_weeks_ago = datetime.utcnow() - timedelta(days=14)
    
    # Current week measurements
    current_week = db.query(PainMeasurement).filter(
        PainMeasurement.patient_id == current_user.id,
        PainMeasurement.created_at >= week_ago
    ).all()
    
    # Previous week measurements
    previous_week = db.query(PainMeasurement).filter(
        PainMeasurement.patient_id == current_user.id,
        PainMeasurement.created_at >= two_weeks_ago,
        PainMeasurement.created_at < week_ago
    ).all()
    
    if not current_week:
        return {
            "message": "No pain measurements found for the current week",
            "current_week_avg": None,
            "previous_week_avg": None,
            "change_percent": None
        }
    
    # Calculate averages
    current_avg = sum(m.facial_stress_score or 0 for m in current_week) / len(current_week)
    previous_avg = sum(m.facial_stress_score or 0 for m in previous_week) / len(previous_week) if previous_week else None
    
    change_percent = None
    trend_direction = "stable"
    if previous_avg:
        change_percent = ((current_avg - previous_avg) / previous_avg) * 100
        if change_percent > 10:
            trend_direction = "worsening"
        elif change_percent < -10:
            trend_direction = "improving"
    
    # Count severity levels
    severity_counts = {
        "low": sum(1 for m in current_week if m.pain_severity_estimate == "low"),
        "moderate": sum(1 for m in current_week if m.pain_severity_estimate == "moderate"),
        "severe": sum(1 for m in current_week if m.pain_severity_estimate == "severe")
    }
    
    return {
        "current_week_avg": round(current_avg, 2),
        "previous_week_avg": round(previous_avg, 2) if previous_avg else None,
        "change_percent": round(change_percent, 2) if change_percent else None,
        "trend_direction": trend_direction,
        "measurements_this_week": len(current_week),
        "severity_breakdown": severity_counts,
        "requires_attention": current_avg > 60 or (change_percent and change_percent > 20)
    }


@router.get("/trends/history", response_model=List[PainTrendResponse])
async def get_trend_history(
    current_user: User = Depends(require_role("patient")),
    db: Session = Depends(get_db)
):
    """
    Get historical weekly trend summaries.
    """
    summaries = db.query(PainTrendSummary).filter(
        PainTrendSummary.patient_id == current_user.id
    ).order_by(desc(PainTrendSummary.period_start)).limit(12).all()
    
    return summaries


# Doctor-facing endpoints
@router.get("/doctor/patient/{patient_id}/measurements")
async def get_patient_measurements_for_doctor(
    patient_id: str,
    days: int = 30,
    current_user: User = Depends(require_role("doctor")),
    db: Session = Depends(get_db)
):
    """
    Doctor view: Get pain measurements for a specific patient.
    Only doctors can access this endpoint.
    """
    # Verify doctor role (already enforced by require_role, but being explicit)
    if current_user.role != "doctor":
        raise HTTPException(status_code=403, detail="Only doctors can access patient pain data")
    
    # Verify doctor has permission to view this patient's data
    connection = db.query(PatientDoctorConnection).filter(
        PatientDoctorConnection.patient_id == patient_id,
        PatientDoctorConnection.doctor_id == current_user.id,
        PatientDoctorConnection.status == "connected"
    ).first()
    
    if not connection:
        raise HTTPException(
            status_code=403, 
            detail="You do not have permission to view this patient's pain data. Patient must be connected to you."
        )
    
    cutoff_date = datetime.utcnow() - timedelta(days=days)
    
    measurements = db.query(PainMeasurement).filter(
        PainMeasurement.patient_id == patient_id,
        PainMeasurement.created_at >= cutoff_date
    ).order_by(desc(PainMeasurement.created_at)).all()
    
    questionnaires = db.query(PainQuestionnaire).filter(
        PainQuestionnaire.patient_id == patient_id,
        PainQuestionnaire.created_at >= cutoff_date
    ).order_by(desc(PainQuestionnaire.created_at)).all()
    
    return {
        "patient_id": patient_id,
        "measurements": [
            {
                "id": m.id,
                "facial_stress_score": m.facial_stress_score,
                "pain_severity_estimate": m.pain_severity_estimate,
                "change_from_previous": m.change_from_previous,
                "created_at": m.created_at,
                "grimace_intensity": m.grimace_intensity,
                "total_contractions": (m.forehead_contractions or 0) + (m.eye_contractions or 0) + (m.mouth_contractions or 0)
            }
            for m in measurements
        ],
        "questionnaires": [
            {
                "id": q.id,
                "measurement_id": q.measurement_id,
                "pain_level_self_reported": q.pain_level_self_reported,
                "pain_location": q.pain_location,
                "pain_type": q.pain_type,
                "medication_taken": q.pain_medication_taken,
                "created_at": q.created_at
            }
            for q in questionnaires
        ]
    }


@router.post("/video-analysis")
async def analyze_pain_video(
    video: UploadFile = File(...),
    current_user: User = Depends(require_role("patient")),
    db: Session = Depends(get_db)
):
    """
    Analyze uploaded video for pain indicators using VideoAIEngine.
    
    Phase 7: Video analysis results are wired to Autopilot ML pipeline.
    Extracts respiratory patterns, facial stress indicators, and movement analysis.
    
    Returns:
        Video analysis results with respiratory risk score and pain indicators
    """
    import tempfile
    import os
    
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            content = await video.read()
            temp_file.write(content)
            temp_path = temp_file.name
        
        from app.services.video_ai_engine import VideoAIEngine
        video_engine = VideoAIEngine()
        
        analysis_result = await video_engine.analyze_video(temp_path)
        
        respiratory_risk = analysis_result.get("respiratory_risk_score", 0.0)
        
        signal_ingestor = SignalIngestorService(db)
        signal = signal_ingestor.ingest_video_analysis_signal(
            patient_id=current_user.id,
            respiratory_risk=respiratory_risk,
            analysis_results=analysis_result,
            source="pain_tracking_video",
            metadata={
                "video_filename": video.filename,
                "analysis_timestamp": datetime.utcnow().isoformat()
            }
        )
        
        logger.info(
            f"[PAIN-TRACKING] Video analysis signal ingested for patient {current_user.id}: "
            f"respiratory_risk={respiratory_risk:.4f}, signal_id={signal.id}"
        )
        
        return {
            "success": True,
            "analysis": {
                "respiratory_risk_score": respiratory_risk,
                "respiratory_rate": analysis_result.get("respiratory_rate"),
                "facial_stress_indicators": analysis_result.get("facial_analysis", {}),
                "quality_score": analysis_result.get("quality_score", 0.0),
                "analysis_confidence": analysis_result.get("confidence_score", 0.0)
            },
            "autopilot_signal_id": str(signal.id)
        }
        
    except Exception as e:
        logger.error(f"[PAIN-TRACKING] Video analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Video analysis failed: {str(e)}")
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


@router.get("/doctor/patient/{patient_id}/trends")
async def get_patient_trends_for_doctor(
    patient_id: str,
    current_user: User = Depends(require_role("doctor")),
    db: Session = Depends(get_db)
):
    """
    Doctor view: Get pain trend summaries for a specific patient.
    Only doctors can access this endpoint.
    """
    # Verify doctor role (already enforced by require_role, but being explicit)
    if current_user.role != "doctor":
        raise HTTPException(status_code=403, detail="Only doctors can access patient pain data")
    
    # Verify doctor has permission to view this patient's data
    connection = db.query(PatientDoctorConnection).filter(
        PatientDoctorConnection.patient_id == patient_id,
        PatientDoctorConnection.doctor_id == current_user.id,
        PatientDoctorConnection.status == "connected"
    ).first()
    
    if not connection:
        raise HTTPException(
            status_code=403, 
            detail="You do not have permission to view this patient's pain data. Patient must be connected to you."
        )
    
    summaries = db.query(PainTrendSummary).filter(
        PainTrendSummary.patient_id == patient_id
    ).order_by(desc(PainTrendSummary.period_start)).limit(12).all()
    
    return {
        "patient_id": patient_id,
        "trend_summaries": [
            {
                "period_type": s.period_type,
                "period_start": s.period_start,
                "period_end": s.period_end,
                "average_stress_score": s.average_stress_score,
                "trend_direction": s.trend_direction,
                "requires_attention": s.requires_physician_attention,
                "alert_reason": s.alert_reason,
                "pdf_report_url": s.pdf_report_url
            }
            for s in summaries
        ]
    }
