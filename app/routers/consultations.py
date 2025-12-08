from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database import get_db
from app.dependencies import get_current_doctor, get_current_user
from app.models.user import User
from app.services.doctor_consultation_service import DoctorConsultationService
from pydantic import BaseModel
from typing import Optional
from datetime import datetime

router = APIRouter(prefix="/api/v1/consultations", tags=["consultations"])


class ConsultationRequest(BaseModel):
    consulted_doctor_id: str
    patient_id: str
    reason: str


class ConsultationResponse(BaseModel):
    consultation_id: int
    reason: str | None = None


@router.post("/request")
async def request_consultation(
    request: ConsultationRequest,
    current_user: User = Depends(get_current_doctor),
    db: Session = Depends(get_db)
):
    service = DoctorConsultationService(db)
    result = service.request_consultation(
        requesting_doctor_id=current_user.id,
        consulted_doctor_id=request.consulted_doctor_id,
        patient_id=request.patient_id,
        reason=request.reason
    )
    return result


@router.get("/")
async def get_consultations(
    current_user: User = Depends(get_current_doctor),
    db: Session = Depends(get_db)
):
    service = DoctorConsultationService(db)
    consultations = service.get_consultations(current_user.id)
    return {"consultations": consultations}


@router.post("/{consultation_id}/approve")
async def approve_consultation(
    consultation_id: int,
    current_user: User = Depends(get_current_doctor),
    db: Session = Depends(get_db)
):
    service = DoctorConsultationService(db)
    result = service.approve_consultation(consultation_id, current_user.id)
    return result


@router.post("/{consultation_id}/decline")
async def decline_consultation(
    consultation_id: int,
    response: ConsultationResponse,
    current_user: User = Depends(get_current_doctor),
    db: Session = Depends(get_db)
):
    service = DoctorConsultationService(db)
    result = service.decline_consultation(
        consultation_id,
        current_user.id,
        response.reason or "No reason provided"
    )
    return result


# Patient Consultation Request Endpoints

class PatientConsultationRequest(BaseModel):
    doctor_id: str
    consultation_reason: str
    symptoms: Optional[str] = None
    urgency: Optional[str] = "routine"
    mode: Optional[str] = "video"


@router.post("/patient/request")
async def create_patient_consultation_request(
    request: PatientConsultationRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Patient requests a consultation with a connected doctor.
    """
    if current_user.role != "patient":
        raise HTTPException(status_code=403, detail="Only patients can request consultations")
    
    from app.models.patient_doctor_connection import PatientDoctorConnection
    
    # Verify patient is connected to this doctor
    connection = db.query(PatientDoctorConnection).filter(
        PatientDoctorConnection.patient_id == current_user.id,
        PatientDoctorConnection.doctor_id == request.doctor_id,
        PatientDoctorConnection.status == "connected"
    ).first()
    
    if not connection:
        raise HTTPException(
            status_code=403, 
            detail="You must be connected to this doctor before requesting a consultation"
        )
    
    from app.models.patient_doctor_connection import PatientConsultation
    
    consultation = PatientConsultation(
        patient_id=current_user.id,
        doctor_id=request.doctor_id,
        consultation_reason=request.consultation_reason,
        symptoms=request.symptoms,
        urgency=request.urgency or "routine",
        mode=request.mode or "video",
        status="requested"
    )
    
    db.add(consultation)
    db.commit()
    db.refresh(consultation)
    
    return {
        "success": True,
        "consultation_id": consultation.id,
        "status": "requested",
        "message": "Consultation request submitted successfully"
    }


@router.get("/patient/my-requests")
async def get_my_consultation_requests(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get all consultation requests for the current patient.
    """
    if current_user.role != "patient":
        raise HTTPException(status_code=403, detail="Only patients can view their consultation requests")
    
    from app.models.patient_doctor_connection import PatientConsultation
    
    requests = db.query(PatientConsultation).filter(
        PatientConsultation.patient_id == current_user.id
    ).order_by(PatientConsultation.created_at.desc()).all()
    
    return {
        "requests": [
            {
                "id": req.id,
                "doctor_id": req.doctor_id,
                "consultation_reason": req.consultation_reason,
                "symptoms": req.symptoms,
                "urgency": req.urgency,
                "mode": req.mode,
                "status": req.status,
                "created_at": req.created_at.isoformat() if req.created_at else None,
                "scheduled_for": req.scheduled_for.isoformat() if req.scheduled_for else None
            }
            for req in requests
        ]
    }


@router.get("/{consultation_id}/details")
async def get_consultation_details(
    consultation_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get consultation details for video call setup.
    Only the patient or doctor involved can access.
    """
    from app.models.patient_doctor_connection import PatientConsultation
    
    consultation = db.query(PatientConsultation).filter(
        PatientConsultation.id == consultation_id
    ).first()
    
    if not consultation:
        raise HTTPException(status_code=404, detail="Consultation not found")
    
    if current_user.id != consultation.patient_id and current_user.id != consultation.doctor_id:
        raise HTTPException(status_code=403, detail="You are not authorized to view this consultation")
    
    return {
        "id": consultation.id,
        "patient_id": consultation.patient_id,
        "doctor_id": consultation.doctor_id,
        "status": consultation.status,
        "scheduled_for": consultation.scheduled_for.isoformat() if consultation.scheduled_for else None
    }


@router.get("/doctor/patient-requests")
async def get_patient_consultation_requests(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get all consultation requests from patients for the current doctor.
    """
    if current_user.role != "doctor":
        raise HTTPException(status_code=403, detail="Only doctors can view patient consultation requests")
    
    from app.models.patient_doctor_connection import PatientConsultation
    
    requests = db.query(PatientConsultation).filter(
        PatientConsultation.doctor_id == current_user.id
    ).order_by(PatientConsultation.created_at.desc()).all()
    
    return {
        "requests": [
            {
                "id": req.id,
                "patient_id": req.patient_id,
                "consultation_reason": req.consultation_reason,
                "symptoms": req.symptoms,
                "urgency": req.urgency,
                "mode": req.mode,
                "status": req.status,
                "created_at": req.created_at.isoformat() if req.created_at else None,
                "scheduled_for": req.scheduled_for.isoformat() if req.scheduled_for else None
            }
            for req in requests
        ]
    }


class ConsultationApproval(BaseModel):
    scheduled_date: str


@router.post("/patient-request/{request_id}/approve")
async def approve_patient_consultation(
    request_id: int,
    approval: ConsultationApproval,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Doctor approves a patient consultation request and schedules it.
    """
    if current_user.role != "doctor":
        raise HTTPException(status_code=403, detail="Only doctors can approve consultations")
    
    from app.models.patient_doctor_connection import PatientConsultation
    from datetime import datetime
    
    consultation = db.query(PatientConsultation).filter(
        PatientConsultation.id == request_id,
        PatientConsultation.doctor_id == current_user.id
    ).first()
    
    if not consultation:
        raise HTTPException(status_code=404, detail="Consultation request not found")
    
    consultation.status = "approved"
    consultation.scheduled_for = datetime.fromisoformat(approval.scheduled_date)
    
    db.commit()
    db.refresh(consultation)
    
    return {
        "success": True,
        "message": "Consultation approved and scheduled",
        "scheduled_for": consultation.scheduled_for.isoformat() if consultation.scheduled_for else None
    }


@router.post("/patient-request/{request_id}/decline")
async def decline_patient_consultation(
    request_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Doctor declines a patient consultation request.
    """
    if current_user.role != "doctor":
        raise HTTPException(status_code=403, detail="Only doctors can decline consultations")
    
    from app.models.patient_doctor_connection import PatientConsultation
    from datetime import datetime
    
    consultation = db.query(PatientConsultation).filter(
        PatientConsultation.id == request_id,
        PatientConsultation.doctor_id == current_user.id
    ).first()
    
    if not consultation:
        raise HTTPException(status_code=404, detail="Consultation request not found")
    
    consultation.status = "declined"
    consultation.cancelled_at = datetime.utcnow()
    
    db.commit()
    
    return {
        "success": True,
        "message": "Consultation request declined"
    }
