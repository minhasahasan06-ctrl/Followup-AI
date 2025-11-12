from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.database import get_db
from app.dependencies import get_current_doctor
from app.models.user import User
from app.services.doctor_consultation_service import DoctorConsultationService
from pydantic import BaseModel

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
