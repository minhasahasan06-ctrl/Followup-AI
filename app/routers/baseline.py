"""
Baseline Calculation API Endpoints

Provides endpoints for health baseline calculation and retrieval.
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
from app.models.health_baseline import HealthBaseline
from app.services.baseline_service import BaselineCalculationService


router = APIRouter(prefix="/api/v1/baseline", tags=["Health Baselines"])


# Response Models
class BaselineStats(BaseModel):
    """Statistics for a single metric baseline"""
    mean: Optional[float] = None
    std: Optional[float] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    
    class Config:
        from_attributes = True


class BaselineResponse(BaseModel):
    """Complete baseline response"""
    id: int
    patient_id: str
    baseline_start_date: datetime
    baseline_end_date: datetime
    data_points_count: int
    
    pain_facial: BaselineStats
    pain_self_reported: BaselineStats
    respiratory_rate: BaselineStats
    symptom_severity: BaselineStats
    activity_impact_rate: Optional[float] = None
    
    baseline_quality: Optional[str] = None
    is_current: bool
    created_at: datetime
    
    class Config:
        from_attributes = True


class BatchBaselineResponse(BaseModel):
    """Response for batch baseline calculation"""
    total_patients: int
    baselines_created: int
    baselines_skipped: int
    errors: List[dict]


# Endpoints

@router.post("/calculate/me", response_model=BaselineResponse)
async def calculate_my_baseline(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Calculate 7-day rolling baseline for current patient.
    
    PATIENT ONLY: Calculates baseline from patient's own health data.
    Creates new baseline and marks previous baselines as not current.
    """
    if current_user.role != "patient":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only patients can calculate their own baseline"
        )
    
    baseline = BaselineCalculationService.calculate_patient_baseline(
        db=db,
        patient_id=current_user.id
    )
    
    if not baseline:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Insufficient data to calculate baseline. Need at least 3 health measurements in last 7 days."
        )
    
    return _format_baseline_response(baseline)


@router.get("/current/me", response_model=Optional[BaselineResponse])
async def get_my_current_baseline(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get current baseline for logged-in patient.
    
    PATIENT ONLY: Returns most recent baseline for wellness monitoring.
    """
    if current_user.role != "patient":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only patients can access their own baseline"
        )
    
    baseline = BaselineCalculationService.get_current_baseline(
        db=db,
        patient_id=current_user.id
    )
    
    if not baseline:
        return None
    
    return _format_baseline_response(baseline)


@router.get("/current/patient/{patient_id}", response_model=Optional[BaselineResponse])
async def get_patient_current_baseline(
    patient_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get current baseline for specific patient (doctor only).
    
    DOCTOR ONLY: Requires active patient-doctor connection.
    Used for wellness monitoring and change detection review.
    """
    if current_user.role != "doctor":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only doctors can access patient baselines"
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
    
    baseline = BaselineCalculationService.get_current_baseline(
        db=db,
        patient_id=patient_id
    )
    
    if not baseline:
        return None
    
    return _format_baseline_response(baseline)


@router.post("/calculate/patient/{patient_id}", response_model=BaselineResponse)
async def calculate_patient_baseline(
    patient_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Calculate baseline for specific patient (doctor only).
    
    DOCTOR ONLY: Requires active patient-doctor connection.
    Useful when doctor wants to trigger baseline recalculation.
    """
    if current_user.role != "doctor":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only doctors can calculate patient baselines"
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
    
    baseline = BaselineCalculationService.calculate_patient_baseline(
        db=db,
        patient_id=patient_id
    )
    
    if not baseline:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Insufficient data to calculate baseline for this patient"
        )
    
    return _format_baseline_response(baseline)


@router.get("/history/me", response_model=List[BaselineResponse])
async def get_my_baseline_history(
    limit: int = 10,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get baseline history for logged-in patient.
    
    PATIENT ONLY: Returns historical baselines for trend tracking.
    """
    if current_user.role != "patient":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only patients can access their own baseline history"
        )
    
    baselines = db.query(HealthBaseline).filter(
        HealthBaseline.patient_id == current_user.id
    ).order_by(HealthBaseline.created_at.desc()).limit(limit).all()
    
    return [_format_baseline_response(b) for b in baselines]


# Helper functions

def _format_baseline_response(baseline: HealthBaseline) -> BaselineResponse:
    """Format HealthBaseline model to API response"""
    return BaselineResponse(
        id=baseline.id,
        patient_id=baseline.patient_id,
        baseline_start_date=baseline.baseline_start_date,
        baseline_end_date=baseline.baseline_end_date,
        data_points_count=baseline.data_points_count,
        pain_facial=BaselineStats(
            mean=baseline.pain_facial_mean,
            std=baseline.pain_facial_std,
            min_value=baseline.pain_facial_min,
            max_value=baseline.pain_facial_max
        ),
        pain_self_reported=BaselineStats(
            mean=baseline.pain_self_reported_mean,
            std=baseline.pain_self_reported_std,
            min_value=baseline.pain_self_reported_min,
            max_value=baseline.pain_self_reported_max
        ),
        respiratory_rate=BaselineStats(
            mean=baseline.respiratory_rate_mean,
            std=baseline.respiratory_rate_std,
            min_value=baseline.respiratory_rate_min,
            max_value=baseline.respiratory_rate_max
        ),
        symptom_severity=BaselineStats(
            mean=baseline.symptom_severity_mean,
            std=baseline.symptom_severity_std,
            min_value=baseline.symptom_severity_min,
            max_value=baseline.symptom_severity_max
        ),
        activity_impact_rate=baseline.activity_impact_rate,
        baseline_quality=baseline.baseline_quality,
        is_current=baseline.is_current,
        created_at=baseline.created_at
    )
