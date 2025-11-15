"""
Symptom Logging Router
Captures symptoms from multiple sources: manual, daily followup, Agent Clona, pain tracking, exam coach
Provides unified symptom timeline for medication side-effect correlation analysis
HIPAA-compliant with patient ownership and doctor-patient connection validation
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc
from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel, Field

from app.database import get_db
from app.models.medication_side_effects import SymptomLog, SymptomSource
from app.models.user import User
from app.models.patient_doctor_connection import PatientDoctorConnection
from app.dependencies import get_current_user


router = APIRouter(prefix="/api/v1/symptom-logging", tags=["Symptom Logging"])


# Pydantic request/response models
class SymptomLogCreate(BaseModel):
    patient_id: str  # Allow doctors to log for patients
    symptom_name: str = Field(..., min_length=1, max_length=200)
    symptom_description: Optional[str] = None
    severity: Optional[int] = Field(None, ge=1, le=10)
    source: SymptomSource
    source_id: Optional[int] = None  # FK to source table
    reported_at: datetime
    onset_date: Optional[datetime] = None
    duration_hours: Optional[int] = Field(None, ge=0)
    body_area: Optional[str] = None
    triggers: Optional[List[str]] = None
    associated_symptoms: Optional[List[str]] = None
    extracted_by_ai: bool = False


class SymptomLogUpdate(BaseModel):
    symptom_name: Optional[str] = None
    symptom_description: Optional[str] = None
    severity: Optional[int] = Field(None, ge=1, le=10)
    onset_date: Optional[datetime] = None
    duration_hours: Optional[int] = Field(None, ge=0)
    body_area: Optional[str] = None
    triggers: Optional[List[str]] = None
    associated_symptoms: Optional[List[str]] = None


class SymptomLogResponse(BaseModel):
    id: int
    patient_id: str
    symptom_name: str
    symptom_description: Optional[str]
    severity: Optional[int]
    source: SymptomSource
    source_id: Optional[int]
    reported_at: datetime
    onset_date: Optional[datetime]
    duration_hours: Optional[int]
    body_area: Optional[str]
    triggers: Optional[List[str]]
    associated_symptoms: Optional[List[str]]
    extracted_by_ai: bool
    created_at: datetime

    class Config:
        from_attributes = True


@router.post("/", response_model=SymptomLogResponse)
def create_symptom_log(
    symptom: SymptomLogCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Create new symptom log entry
    Patients can only create for themselves
    Doctors can create for their patients (with connection verification)
    Used by: manual logging, daily followup, Agent Clona, pain tracking, exam coach
    """
    # Only patients and doctors can create symptom logs
    if current_user.role not in ["patient", "doctor"]:
        raise HTTPException(status_code=403, detail="Access denied")
    
    # Validate patient_id based on role
    patient_id = symptom.patient_id
    
    # FIX: Verify patient exists first
    patient = db.query(User).filter(and_(User.id == patient_id, User.role == "patient")).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    if current_user.role == "patient":
        # Patients can only create for themselves
        if patient_id != current_user.id:
            raise HTTPException(status_code=403, detail="Patients can only log symptoms for themselves")
    elif current_user.role == "doctor":
        # Doctors must have active connection to patient
        connection = db.query(PatientDoctorConnection).filter(
            and_(
                PatientDoctorConnection.patient_id == patient_id,
                PatientDoctorConnection.doctor_id == current_user.id,
                PatientDoctorConnection.status == "active"
            )
        ).first()
        if not connection:
            raise HTTPException(status_code=403, detail="Not connected to this patient")
    
    # Create symptom log
    new_symptom = SymptomLog(
        patient_id=patient_id,
        symptom_name=symptom.symptom_name,
        symptom_description=symptom.symptom_description,
        severity=symptom.severity,
        source=symptom.source,
        source_id=symptom.source_id,
        reported_at=symptom.reported_at,
        onset_date=symptom.onset_date,
        duration_hours=symptom.duration_hours,
        body_area=symptom.body_area,
        triggers=symptom.triggers,
        associated_symptoms=symptom.associated_symptoms,
        extracted_by_ai=symptom.extracted_by_ai
    )
    
    db.add(new_symptom)
    db.commit()
    db.refresh(new_symptom)
    
    return new_symptom


@router.get("/patient/{patient_id}", response_model=List[SymptomLogResponse])
def get_patient_symptoms(
    patient_id: str,
    current_user: User = Depends(get_current_user),
    source: Optional[SymptomSource] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    symptom_name: Optional[str] = None,
    min_severity: Optional[int] = Query(None, ge=1, le=10),
    limit: int = Query(100, le=500),
    db: Session = Depends(get_db)
):
    """
    Get symptom logs for a patient with optional filters
    Patients can only see their own
    Doctors can see their connected patients
    """
    # Only patients and doctors can view symptoms
    if current_user.role not in ["patient", "doctor"]:
        raise HTTPException(status_code=403, detail="Access denied")
    
    # FIX: Verify patient exists first
    patient = db.query(User).filter(and_(User.id == patient_id, User.role == "patient")).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    # Security: Verify access
    if current_user.role == "patient":
        if patient_id != current_user.id:
            raise HTTPException(status_code=403, detail="Access denied")
    elif current_user.role == "doctor":
        connection = db.query(PatientDoctorConnection).filter(
            and_(
                PatientDoctorConnection.patient_id == patient_id,
                PatientDoctorConnection.doctor_id == current_user.id,
                PatientDoctorConnection.status == "active"
            )
        ).first()
        if not connection:
            raise HTTPException(status_code=403, detail="Not connected to this patient")
    
    # Build query
    query = db.query(SymptomLog).filter(SymptomLog.patient_id == patient_id)
    
    # Apply filters
    if source:
        query = query.filter(SymptomLog.source == source)
    if start_date:
        query = query.filter(SymptomLog.reported_at >= start_date)
    if end_date:
        query = query.filter(SymptomLog.reported_at <= end_date)
    if symptom_name:
        query = query.filter(SymptomLog.symptom_name.ilike(f"%{symptom_name}%"))
    if min_severity:
        query = query.filter(SymptomLog.severity >= min_severity)
    
    # Order by most recent first
    symptoms = query.order_by(desc(SymptomLog.reported_at)).limit(limit).all()
    
    return symptoms


@router.get("/{log_id}", response_model=SymptomLogResponse)
def get_symptom_details(
    log_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get details of a specific symptom log
    """
    # Only patients and doctors can view symptom details
    if current_user.role not in ["patient", "doctor"]:
        raise HTTPException(status_code=403, detail="Access denied")
    
    symptom = db.query(SymptomLog).filter(SymptomLog.id == log_id).first()
    
    if not symptom:
        raise HTTPException(status_code=404, detail="Symptom log not found")
    
    # Security: Verify access
    if current_user.role == "patient":
        if symptom.patient_id != current_user.id:
            raise HTTPException(status_code=403, detail="Access denied")
    elif current_user.role == "doctor":
        connection = db.query(PatientDoctorConnection).filter(
            and_(
                PatientDoctorConnection.patient_id == symptom.patient_id,
                PatientDoctorConnection.doctor_id == current_user.id,
                PatientDoctorConnection.status == "active"
            )
        ).first()
        if not connection:
            raise HTTPException(status_code=403, detail="Not connected to this patient")
    
    return symptom


@router.put("/{log_id}", response_model=SymptomLogResponse)
def update_symptom_log(
    log_id: int,
    updates: SymptomLogUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Update symptom log details
    Patients can update their own symptoms
    Doctors can update symptoms for connected patients
    """
    # Only patients and doctors can update symptoms
    if current_user.role not in ["patient", "doctor"]:
        raise HTTPException(status_code=403, detail="Access denied")
    
    symptom = db.query(SymptomLog).filter(SymptomLog.id == log_id).first()
    
    if not symptom:
        raise HTTPException(status_code=404, detail="Symptom log not found")
    
    # Security: Verify access
    if current_user.role == "patient":
        if symptom.patient_id != current_user.id:
            raise HTTPException(status_code=403, detail="Access denied")
    elif current_user.role == "doctor":
        connection = db.query(PatientDoctorConnection).filter(
            and_(
                PatientDoctorConnection.patient_id == symptom.patient_id,
                PatientDoctorConnection.doctor_id == current_user.id,
                PatientDoctorConnection.status == "active"
            )
        ).first()
        if not connection:
            raise HTTPException(status_code=403, detail="Not connected to this patient")
    
    # Apply updates
    if updates.symptom_name is not None:
        symptom.symptom_name = updates.symptom_name
    if updates.symptom_description is not None:
        symptom.symptom_description = updates.symptom_description
    if updates.severity is not None:
        symptom.severity = updates.severity
    if updates.onset_date is not None:
        symptom.onset_date = updates.onset_date
    if updates.duration_hours is not None:
        symptom.duration_hours = updates.duration_hours
    if updates.body_area is not None:
        symptom.body_area = updates.body_area
    if updates.triggers is not None:
        symptom.triggers = updates.triggers
    if updates.associated_symptoms is not None:
        symptom.associated_symptoms = updates.associated_symptoms
    
    db.commit()
    db.refresh(symptom)
    
    return symptom


@router.delete("/{log_id}")
def delete_symptom_log(
    log_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Delete symptom log (patient only)
    Note: This is a hard delete - consider soft delete in production
    """
    # Only patients can delete their own symptoms
    if current_user.role != "patient":
        raise HTTPException(status_code=403, detail="Only patients can delete symptoms")
    
    symptom = db.query(SymptomLog).filter(SymptomLog.id == log_id).first()
    
    if not symptom:
        raise HTTPException(status_code=404, detail="Symptom log not found")
    
    # Security: Patient can only delete their own
    if symptom.patient_id != current_user.id:
        raise HTTPException(status_code=403, detail="Access denied")
    
    db.delete(symptom)
    db.commit()
    
    return {"message": "Symptom log deleted successfully"}


@router.get("/patient/{patient_id}/timeline", response_model=List[SymptomLogResponse])
def get_symptom_timeline(
    patient_id: str,
    current_user: User = Depends(get_current_user),
    days: int = Query(30, ge=1, le=365),
    db: Session = Depends(get_db)
):
    """
    Get symptom timeline for correlation analysis
    Returns symptoms from the last N days (default 30)
    Ordered chronologically for medication correlation
    """
    # Only patients and doctors can view timeline
    if current_user.role not in ["patient", "doctor"]:
        raise HTTPException(status_code=403, detail="Access denied")
    
    # FIX: Verify patient exists first
    patient = db.query(User).filter(and_(User.id == patient_id, User.role == "patient")).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    # Security: Verify access
    if current_user.role == "patient":
        if patient_id != current_user.id:
            raise HTTPException(status_code=403, detail="Access denied")
    elif current_user.role == "doctor":
        connection = db.query(PatientDoctorConnection).filter(
            and_(
                PatientDoctorConnection.patient_id == patient_id,
                PatientDoctorConnection.doctor_id == current_user.id,
                PatientDoctorConnection.status == "active"
            )
        ).first()
        if not connection:
            raise HTTPException(status_code=403, detail="Not connected to this patient")
    
    # Get symptoms from last N days
    from datetime import timedelta
    cutoff_date = datetime.utcnow() - timedelta(days=days)
    
    symptoms = db.query(SymptomLog).filter(
        and_(
            SymptomLog.patient_id == patient_id,
            SymptomLog.reported_at >= cutoff_date
        )
    ).order_by(SymptomLog.reported_at).all()
    
    return symptoms


@router.get("/patient/{patient_id}/aggregated")
def get_aggregated_symptoms(
    patient_id: str,
    current_user: User = Depends(get_current_user),
    days: int = Query(30, ge=1, le=365),
    db: Session = Depends(get_db)
):
    """
    Get aggregated symptom statistics for a patient
    Returns symptom frequency, severity trends, and common patterns
    Used for dashboard visualization and correlation analysis
    """
    # Only patients and doctors can view aggregated data
    if current_user.role not in ["patient", "doctor"]:
        raise HTTPException(status_code=403, detail="Access denied")
    
    # FIX: Verify patient exists first
    patient = db.query(User).filter(and_(User.id == patient_id, User.role == "patient")).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    # Security: Verify access
    if current_user.role == "patient":
        if patient_id != current_user.id:
            raise HTTPException(status_code=403, detail="Access denied")
    elif current_user.role == "doctor":
        connection = db.query(PatientDoctorConnection).filter(
            and_(
                PatientDoctorConnection.patient_id == patient_id,
                PatientDoctorConnection.doctor_id == current_user.id,
                PatientDoctorConnection.status == "active"
            )
        ).first()
        if not connection:
            raise HTTPException(status_code=403, detail="Not connected to this patient")
    
    # Get symptoms from last N days
    from datetime import timedelta
    cutoff_date = datetime.utcnow() - timedelta(days=days)
    
    symptoms = db.query(SymptomLog).filter(
        and_(
            SymptomLog.patient_id == patient_id,
            SymptomLog.reported_at >= cutoff_date
        )
    ).all()
    
    # Aggregate by symptom name
    symptom_counts = {}
    symptom_avg_severity = {}
    symptom_sources = {}
    
    for symptom in symptoms:
        name = symptom.symptom_name
        if name not in symptom_counts:
            symptom_counts[name] = 0
            symptom_avg_severity[name] = []
            symptom_sources[name] = {}
        
        symptom_counts[name] += 1
        
        if symptom.severity:
            symptom_avg_severity[name].append(symptom.severity)
        
        source = symptom.source.value
        if source not in symptom_sources[name]:
            symptom_sources[name][source] = 0
        symptom_sources[name][source] += 1
    
    # Calculate averages
    for name in symptom_avg_severity:
        if symptom_avg_severity[name]:
            avg = sum(symptom_avg_severity[name]) / len(symptom_avg_severity[name])
            symptom_avg_severity[name] = round(avg, 1)
        else:
            symptom_avg_severity[name] = None
    
    # Build response
    aggregated = []
    for name in symptom_counts:
        aggregated.append({
            "symptom_name": name,
            "count": symptom_counts[name],
            "average_severity": symptom_avg_severity[name],
            "sources": symptom_sources[name]
        })
    
    # Sort by count (most frequent first)
    aggregated.sort(key=lambda x: x["count"], reverse=True)
    
    return {
        "patient_id": patient_id,
        "days": days,
        "total_symptoms": len(symptoms),
        "unique_symptoms": len(symptom_counts),
        "symptoms": aggregated
    }
