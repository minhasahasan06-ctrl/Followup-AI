"""
Deviation Detection API Endpoints

Provides endpoints for health deviation detection and retrieval.
Used for wellness monitoring and change detection (NOT medical diagnosis).
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import Optional, List
from datetime import datetime
from pydantic import BaseModel, Field

from app.database import get_db
from app.config import get_current_user
from app.models.user import User
from app.models.health_baseline import BaselineDeviation
from app.services.deviation_service import DeviationDetectionService


router = APIRouter(prefix="/api/v1/deviation", tags=["Health Deviations"])


# Response Models
class DeviationResponse(BaseModel):
    """Deviation detection response"""
    id: int
    patient_id: str
    baseline_id: int
    metric_name: str
    measurement_value: float
    measurement_date: datetime
    
    z_score: float
    percent_change: float
    baseline_mean: float
    baseline_std: float
    
    trend_3day_slope: Optional[float] = None
    trend_7day_slope: Optional[float] = None
    trend_direction: Optional[str] = None
    
    deviation_type: str
    severity_level: str
    
    alert_triggered: bool
    alert_message: Optional[str] = None
    
    source_measurement_id: Optional[int] = None
    source_table: Optional[str] = None
    created_at: datetime
    
    class Config:
        from_attributes = True


class DeviationSummary(BaseModel):
    """Summary of deviations by metric"""
    metric_name: str
    total_deviations: int
    critical_count: int
    moderate_count: int
    recent_z_score: Optional[float] = None
    trend_direction: Optional[str] = None


# Endpoints

@router.post("/detect/pain_facial/{measurement_id}", response_model=Optional[DeviationResponse])
async def detect_pain_facial_deviation(
    measurement_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Detect deviation for new pain facial measurement.
    
    PATIENT ONLY: Analyzes new measurement against personal baseline.
    Automatically called when new pain measurement is recorded.
    """
    if current_user.role != "patient":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only patients can detect deviations for their own measurements"
        )
    
    deviation = DeviationDetectionService.detect_pain_facial_deviation(
        db=db,
        patient_id=current_user.id,
        measurement_id=measurement_id
    )
    
    return deviation


@router.post("/detect/pain_self_reported/{questionnaire_id}", response_model=Optional[DeviationResponse])
async def detect_pain_self_reported_deviation(
    questionnaire_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Detect deviation for self-reported pain score.
    
    PATIENT ONLY: Analyzes new pain report against personal baseline.
    """
    if current_user.role != "patient":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only patients can detect deviations for their own measurements"
        )
    
    deviation = DeviationDetectionService.detect_pain_self_reported_deviation(
        db=db,
        patient_id=current_user.id,
        questionnaire_id=questionnaire_id
    )
    
    return deviation


@router.post("/detect/respiratory_rate/{measurement_id}", response_model=Optional[DeviationResponse])
async def detect_respiratory_rate_deviation(
    measurement_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Detect deviation for respiratory rate measurement.
    
    PATIENT ONLY: Analyzes new respiratory rate against personal baseline.
    """
    if current_user.role != "patient":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only patients can detect deviations for their own measurements"
        )
    
    deviation = DeviationDetectionService.detect_respiratory_rate_deviation(
        db=db,
        patient_id=current_user.id,
        measurement_id=measurement_id
    )
    
    return deviation


@router.post("/detect/symptom_severity/{symptom_log_id}", response_model=Optional[DeviationResponse])
async def detect_symptom_severity_deviation(
    symptom_log_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Detect deviation for symptom severity.
    
    PATIENT ONLY: Analyzes new symptom against personal baseline.
    """
    if current_user.role != "patient":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only patients can detect deviations for their own measurements"
        )
    
    deviation = DeviationDetectionService.detect_symptom_severity_deviation(
        db=db,
        patient_id=current_user.id,
        symptom_log_id=symptom_log_id
    )
    
    return deviation


@router.get("/me", response_model=List[DeviationResponse])
async def get_my_deviations(
    days: int = 7,
    alert_only: bool = False,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get deviation records for logged-in patient.
    
    PATIENT ONLY: Returns wellness monitoring deviations.
    
    Args:
        days: Number of days to look back (default: 7)
        alert_only: Only return deviations that triggered alerts
    """
    if current_user.role != "patient":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only patients can access their own deviations"
        )
    
    deviations = DeviationDetectionService.get_patient_deviations(
        db=db,
        patient_id=current_user.id,
        days=days,
        alert_only=alert_only
    )
    
    return deviations


@router.get("/patient/{patient_id}", response_model=List[DeviationResponse])
async def get_patient_deviations(
    patient_id: str,
    days: int = 7,
    alert_only: bool = False,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get deviation records for specific patient (doctor only).
    
    DOCTOR ONLY: Requires active patient-doctor connection.
    Used for wellness monitoring and change detection review.
    """
    if current_user.role != "doctor":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only doctors can access patient deviations"
        )
    
    # Verify patient-doctor connection
    from app.models.patient_doctor_connection import PatientDoctorConnection
    connection = db.query(PatientDoctorConnection).filter(
        PatientDoctorConnection.patient_id == patient_id,
        PatientDoctorConnection.doctor_id == current_user.id,
        PatientDoctorConnection.status == "accepted"
    ).first()
    
    if not connection:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="No active connection with this patient"
        )
    
    deviations = DeviationDetectionService.get_patient_deviations(
        db=db,
        patient_id=patient_id,
        days=days,
        alert_only=alert_only
    )
    
    return deviations


@router.get("/summary/me", response_model=List[DeviationSummary])
async def get_my_deviation_summary(
    days: int = 7,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get deviation summary by metric for logged-in patient.
    
    PATIENT ONLY: Provides overview of health pattern changes.
    """
    if current_user.role != "patient":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only patients can access their own deviation summary"
        )
    
    deviations = DeviationDetectionService.get_patient_deviations(
        db=db,
        patient_id=current_user.id,
        days=days,
        alert_only=False
    )
    
    # Group by metric
    summary_dict = {}
    for dev in deviations:
        if dev.metric_name not in summary_dict:
            summary_dict[dev.metric_name] = {
                "metric_name": dev.metric_name,
                "total_deviations": 0,
                "critical_count": 0,
                "moderate_count": 0,
                "recent_z_score": dev.z_score,
                "trend_direction": dev.trend_direction
            }
        
        summary_dict[dev.metric_name]["total_deviations"] += 1
        if dev.severity_level == "critical":
            summary_dict[dev.metric_name]["critical_count"] += 1
        elif dev.severity_level == "moderate":
            summary_dict[dev.metric_name]["moderate_count"] += 1
    
    return list(summary_dict.values())
