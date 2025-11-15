"""
Medication Timeline Router - HIPAA-compliant medication tracking
Tracks medications, dosage changes, and supports side-effect correlation
WITHOUT making medical diagnoses - pattern detection only
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import desc, and_
from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel, Field

from app.database import get_db
from app.models.medication_side_effects import (
    MedicationTimeline, DosageChange, ChangeReason
)
from app.models.user import User
from app.models.patient_doctor_connection import PatientDoctorConnection
from app.dependencies import get_current_user


router = APIRouter(prefix="/api/v1/medication-timeline", tags=["Medication Timeline"])


# Pydantic request/response models
class MedicationTimelineCreate(BaseModel):
    patient_id: str  # FIX: Allow doctors to specify which patient
    medication_name: str = Field(..., min_length=1, max_length=200)
    generic_name: Optional[str] = None
    drug_class: Optional[str] = None
    dosage: str = Field(..., min_length=1, max_length=100)
    frequency: str = Field(..., min_length=1, max_length=100)
    route: Optional[str] = None
    started_at: datetime
    prescribed_by: Optional[str] = None
    prescription_reason: Optional[str] = None
    patient_notes: Optional[str] = None


class MedicationTimelineUpdate(BaseModel):
    dosage: Optional[str] = None
    frequency: Optional[str] = None
    stopped_at: Optional[datetime] = None
    is_active: Optional[bool] = None
    patient_notes: Optional[str] = None


class DosageChangeCreate(BaseModel):
    new_dosage: str = Field(..., min_length=1, max_length=100)
    new_frequency: str = Field(..., min_length=1, max_length=100)
    change_reason: ChangeReason
    change_date: datetime
    change_notes: Optional[str] = None
    changed_by: Optional[str] = None


class StopMedicationRequest(BaseModel):
    stop_date: datetime
    reason: Optional[str] = None


class MedicationTimelineResponse(BaseModel):
    id: int
    patient_id: str
    medication_name: str
    generic_name: Optional[str]
    drug_class: Optional[str]
    dosage: str
    frequency: str
    route: Optional[str]
    started_at: datetime
    stopped_at: Optional[datetime]
    is_active: bool
    prescribed_by: Optional[str]
    prescription_reason: Optional[str]
    patient_notes: Optional[str]
    created_at: datetime
    updated_at: datetime


class DosageChangeResponse(BaseModel):
    id: int
    medication_timeline_id: int
    change_reason: str
    change_date: datetime
    old_dosage: Optional[str]
    new_dosage: str
    old_frequency: Optional[str]
    new_frequency: str
    changed_by: Optional[str]
    change_notes: Optional[str]
    created_at: datetime


@router.post("/", response_model=MedicationTimelineResponse)
def create_medication_timeline(
    medication: MedicationTimelineCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Create new medication timeline entry
    Patients can only create for themselves
    Doctors can create for their patients (with connection verification)
    """
    # Only patients and doctors can create medications
    if current_user.role not in ["patient", "doctor"]:
        raise HTTPException(status_code=403, detail="Access denied")
    
    # FIX: Validate patient_id based on role
    patient_id = medication.patient_id
    
    if current_user.role == "patient":
        # Patients can only create for themselves
        if patient_id != current_user.id:
            raise HTTPException(status_code=403, detail="Patients can only create medications for themselves")
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
    
    # Create medication timeline
    new_medication = MedicationTimeline(
        patient_id=patient_id,
        medication_name=medication.medication_name,
        generic_name=medication.generic_name,
        drug_class=medication.drug_class,
        dosage=medication.dosage,
        frequency=medication.frequency,
        route=medication.route,
        started_at=medication.started_at,
        is_active=True,
        prescribed_by=medication.prescribed_by,
        prescription_reason=medication.prescription_reason,
        patient_notes=medication.patient_notes
    )
    
    db.add(new_medication)
    db.commit()
    db.refresh(new_medication)
    
    # Create initial dosage change record
    initial_change = DosageChange(
        medication_timeline_id=new_medication.id,
        patient_id=patient_id,
        change_reason=ChangeReason.INITIAL_DOSE,
        change_date=medication.started_at,
        new_dosage=medication.dosage,
        new_frequency=medication.frequency,
        changed_by=medication.prescribed_by or current_user.id,
        change_notes=f"Initial prescription: {medication.medication_name}"
    )
    
    db.add(initial_change)
    db.commit()
    
    return new_medication


@router.get("/patient/{patient_id}", response_model=List[MedicationTimelineResponse])
def get_patient_medications(
    patient_id: str,
    current_user: User = Depends(get_current_user),
    active_only: bool = True,
    db: Session = Depends(get_db)
):
    """
    Get all medications for a patient
    Patients can only see their own
    Doctors can see their connected patients
    """
    # Only patients and doctors can view medications
    if current_user.role not in ["patient", "doctor"]:
        raise HTTPException(status_code=403, detail="Access denied")
    
    # Security: Verify access
    if current_user.role == "patient":
        if patient_id != current_user.id:
            raise HTTPException(status_code=403, detail="Access denied")
    elif current_user.role == "doctor":
        # Verify doctor-patient connection
        connection = db.query(PatientDoctorConnection).filter(
            and_(
                PatientDoctorConnection.patient_id == patient_id,
                PatientDoctorConnection.doctor_id == current_user.id,
                PatientDoctorConnection.status == "active"
            )
        ).first()
        if not connection:
            raise HTTPException(status_code=403, detail="Not connected to this patient")
    
    # Query medications
    query = db.query(MedicationTimeline).filter(
        MedicationTimeline.patient_id == patient_id
    )
    
    if active_only:
        query = query.filter(MedicationTimeline.is_active == True)
    
    medications = query.order_by(desc(MedicationTimeline.started_at)).all()
    
    return medications


@router.get("/{medication_id}", response_model=MedicationTimelineResponse)
def get_medication_details(
    medication_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get details of a specific medication
    """
    # Only patients and doctors can view medication details
    if current_user.role not in ["patient", "doctor"]:
        raise HTTPException(status_code=403, detail="Access denied")
    medication = db.query(MedicationTimeline).filter(
        MedicationTimeline.id == medication_id
    ).first()
    
    if not medication:
        raise HTTPException(status_code=404, detail="Medication not found")
    
    # Security: Verify access
    if current_user.role == "patient":
        if medication.patient_id != current_user.id:
            raise HTTPException(status_code=403, detail="Access denied")
    elif current_user.role == "doctor":
        # Verify doctor-patient connection
        connection = db.query(PatientDoctorConnection).filter(
            and_(
                PatientDoctorConnection.patient_id == medication.patient_id,
                PatientDoctorConnection.doctor_id == current_user.id,
                PatientDoctorConnection.status == "active"
            )
        ).first()
        if not connection:
            raise HTTPException(status_code=403, detail="Not connected to this patient")
    
    return medication


@router.put("/{medication_id}", response_model=MedicationTimelineResponse)
def update_medication_timeline(
    medication_id: int,
    updates: MedicationTimelineUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Update medication timeline
    Automatically tracks dosage changes
    """
    # Only patients and doctors can update medications
    if current_user.role not in ["patient", "doctor"]:
        raise HTTPException(status_code=403, detail="Access denied")
    medication = db.query(MedicationTimeline).filter(
        MedicationTimeline.id == medication_id
    ).first()
    
    if not medication:
        raise HTTPException(status_code=404, detail="Medication not found")
    
    # Security: Verify access
    if current_user.role == "patient":
        if medication.patient_id != current_user.id:
            raise HTTPException(status_code=403, detail="Access denied")
    elif current_user.role == "doctor":
        # Verify doctor-patient connection
        connection = db.query(PatientDoctorConnection).filter(
            and_(
                PatientDoctorConnection.patient_id == medication.patient_id,
                PatientDoctorConnection.doctor_id == current_user.id,
                PatientDoctorConnection.status == "active"
            )
        ).first()
        if not connection:
            raise HTTPException(status_code=403, detail="Not connected to this patient")
    
    # Track dosage change if dosage or frequency changed
    dosage_changed = (updates.dosage and updates.dosage != medication.dosage) or \
                     (updates.frequency and updates.frequency != medication.frequency)
    
    if dosage_changed:
        # Determine change reason
        change_reason = ChangeReason.DOSE_INCREASE  # Default
        if updates.dosage:
            # Simple heuristic - could be enhanced with dosage parsing
            try:
                old_val = float(''.join(filter(str.isdigit, medication.dosage)))
                new_val = float(''.join(filter(str.isdigit, updates.dosage)))
                if new_val < old_val:
                    change_reason = ChangeReason.DOSE_DECREASE
            except:
                pass  # Keep default
        
        # Create dosage change record
        dosage_change = DosageChange(
            medication_timeline_id=medication_id,
            patient_id=medication.patient_id,
            change_reason=change_reason,
            change_date=datetime.now(),
            old_dosage=medication.dosage,
            new_dosage=updates.dosage or medication.dosage,
            old_frequency=medication.frequency,
            new_frequency=updates.frequency or medication.frequency,
            changed_by=current_user.id,
            change_notes=updates.patient_notes
        )
        
        db.add(dosage_change)
    
    # Apply updates
    if updates.dosage:
        medication.dosage = updates.dosage
    if updates.frequency:
        medication.frequency = updates.frequency
    if updates.stopped_at is not None:
        medication.stopped_at = updates.stopped_at
    if updates.is_active is not None:
        medication.is_active = updates.is_active
    if updates.patient_notes:
        medication.patient_notes = updates.patient_notes
    
    db.commit()
    db.refresh(medication)
    
    return medication


@router.post("/{medication_id}/stop", response_model=MedicationTimelineResponse)
def stop_medication(
    medication_id: int,
    request: StopMedicationRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Stop/discontinue a medication
    Creates dosage change record for tracking
    """
    # Only patients and doctors can stop medications
    if current_user.role not in ["patient", "doctor"]:
        raise HTTPException(status_code=403, detail="Access denied")
    medication = db.query(MedicationTimeline).filter(
        MedicationTimeline.id == medication_id
    ).first()
    
    if not medication:
        raise HTTPException(status_code=404, detail="Medication not found")
    
    # Security: Verify access
    if current_user.role == "patient":
        if medication.patient_id != current_user.id:
            raise HTTPException(status_code=403, detail="Access denied")
    elif current_user.role == "doctor":
        # Verify doctor-patient connection
        connection = db.query(PatientDoctorConnection).filter(
            and_(
                PatientDoctorConnection.patient_id == medication.patient_id,
                PatientDoctorConnection.doctor_id == current_user.id,
                PatientDoctorConnection.status == "active"
            )
        ).first()
        if not connection:
            raise HTTPException(status_code=403, detail="Not connected to this patient")
    
    # Stop medication
    medication.stopped_at = request.stop_date
    medication.is_active = False
    
    # Create discontinuation record
    discontinuation = DosageChange(
        medication_timeline_id=medication_id,
        patient_id=medication.patient_id,
        change_reason=ChangeReason.DISCONTINUED,
        change_date=request.stop_date,
        old_dosage=medication.dosage,
        new_dosage="0",
        old_frequency=medication.frequency,
        new_frequency="discontinued",
        changed_by=current_user.id,
        change_notes=request.reason
    )
    
    db.add(discontinuation)
    db.commit()
    db.refresh(medication)
    
    return medication


@router.get("/{medication_id}/dosage-history", response_model=List[DosageChangeResponse])
def get_dosage_history(
    medication_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get complete dosage change history for a medication
    """
    # Only patients and doctors can view dosage history
    if current_user.role not in ["patient", "doctor"]:
        raise HTTPException(status_code=403, detail="Access denied")
    medication = db.query(MedicationTimeline).filter(
        MedicationTimeline.id == medication_id
    ).first()
    
    if not medication:
        raise HTTPException(status_code=404, detail="Medication not found")
    
    # Security: Verify access
    if current_user.role == "patient":
        if medication.patient_id != current_user.id:
            raise HTTPException(status_code=403, detail="Access denied")
    elif current_user.role == "doctor":
        # Verify doctor-patient connection
        connection = db.query(PatientDoctorConnection).filter(
            and_(
                PatientDoctorConnection.patient_id == medication.patient_id,
                PatientDoctorConnection.doctor_id == current_user.id,
                PatientDoctorConnection.status == "active"
            )
        ).first()
        if not connection:
            raise HTTPException(status_code=403, detail="Not connected to this patient")
    
    # Get dosage changes
    changes = db.query(DosageChange).filter(
        DosageChange.medication_timeline_id == medication_id
    ).order_by(desc(DosageChange.change_date)).all()
    
    return changes


@router.delete("/{medication_id}")
def delete_medication_timeline(
    medication_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Delete medication timeline (patient only)
    Note: This is a hard delete - consider soft delete in production
    """
    # Only patients can delete their own medications
    if current_user.role != "patient":
        raise HTTPException(status_code=403, detail="Only patients can delete medications")
    medication = db.query(MedicationTimeline).filter(
        MedicationTimeline.id == medication_id
    ).first()
    
    if not medication:
        raise HTTPException(status_code=404, detail="Medication not found")
    
    # Security: Only patient can delete their own medications
    if medication.patient_id != current_user.id:
        raise HTTPException(status_code=403, detail="Access denied")
    
    # Delete associated dosage changes first (cascade)
    db.query(DosageChange).filter(
        DosageChange.medication_timeline_id == medication_id
    ).delete()
    
    # Delete medication
    db.delete(medication)
    db.commit()
    
    return {"status": "deleted", "medication_id": medication_id}
