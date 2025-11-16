"""
Risk Score API Endpoints

Provides composite risk scoring for health deterioration prediction.
Used for wellness monitoring (NOT medical diagnosis).
"""

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from typing import List, Dict

from app.database import get_db
from app.config import get_current_user
from app.models.user import User
from app.services.risk_scoring_service import RiskScoringService, RiskScore


router = APIRouter(prefix="/api/v1/risk", tags=["Risk Scoring"])


@router.get("/score/me", response_model=RiskScore)
async def get_my_risk_score(
    lookback_hours: int = Query(24, ge=1, le=168, description="Hours to analyze (1-168)"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Calculate current risk score for logged-in patient.
    
    PATIENT ONLY: Analyzes recent health pattern deviations.
    
    Returns composite risk score (0-15) with:
    - Risk level: stable (0-2), monitoring (3-5), urgent (6-15)
    - Risk factor breakdown by metric
    - Wellness recommendations and action items
    
    Note: This is wellness monitoring, NOT medical diagnosis.
    Always discuss significant changes with your healthcare provider.
    
    Args:
        lookback_hours: Hours to analyze (default: 24, max: 168 = 7 days)
    """
    if current_user.role != "patient":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only patients can access their own risk score"
        )
    
    risk_score = RiskScoringService.calculate_patient_risk_score(
        db=db,
        patient_id=current_user.id,
        lookback_hours=lookback_hours
    )
    
    return risk_score


@router.get("/score/patient/{patient_id}", response_model=RiskScore)
async def get_patient_risk_score(
    patient_id: str,
    lookback_hours: int = Query(24, ge=1, le=168, description="Hours to analyze (1-168)"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Calculate risk score for specific patient (doctor only).
    
    DOCTOR ONLY: Requires active patient-doctor connection.
    Used for wellness monitoring and deterioration tracking.
    
    Args:
        patient_id: Patient identifier
        lookback_hours: Hours to analyze (default: 24, max: 168)
    """
    if current_user.role != "doctor":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only doctors can access patient risk scores"
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
    
    risk_score = RiskScoringService.calculate_patient_risk_score(
        db=db,
        patient_id=patient_id,
        lookback_hours=lookback_hours
    )
    
    return risk_score


@router.get("/history/me", response_model=List[Dict])
async def get_my_risk_history(
    days: int = Query(7, ge=1, le=30, description="Days of history (1-30)"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get historical risk scores for logged-in patient.
    
    PATIENT ONLY: Returns daily risk score trend.
    Useful for visualizing wellness pattern changes over time.
    
    Args:
        days: Number of days of history (default: 7, max: 30)
    
    Returns:
        List of {date, score, level, deviation_count} dictionaries
    """
    if current_user.role != "patient":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only patients can access their own risk history"
        )
    
    history = RiskScoringService.get_risk_score_history(
        db=db,
        patient_id=current_user.id,
        days=days
    )
    
    return history


@router.get("/history/patient/{patient_id}", response_model=List[Dict])
async def get_patient_risk_history(
    patient_id: str,
    days: int = Query(7, ge=1, le=30, description="Days of history (1-30)"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get historical risk scores for specific patient (doctor only).
    
    DOCTOR ONLY: Requires active patient-doctor connection.
    Returns daily risk score trend for deterioration tracking.
    
    Args:
        patient_id: Patient identifier
        days: Number of days of history (default: 7, max: 30)
    """
    if current_user.role != "doctor":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only doctors can access patient risk history"
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
    
    history = RiskScoringService.get_risk_score_history(
        db=db,
        patient_id=patient_id,
        days=days
    )
    
    return history
