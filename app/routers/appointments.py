from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from app.database import get_db
from app.dependencies import get_current_user, get_current_doctor
from app.models.appointment import Appointment
from app.models.user import User
from pydantic import BaseModel
from datetime import datetime

router = APIRouter(prefix="/api/v1/appointments", tags=["appointments"])


class AppointmentCreate(BaseModel):
    patient_id: str
    appointment_date: datetime
    duration_minutes: int = 30
    appointment_type: str
    notes: str | None = None


class AppointmentResponse(BaseModel):
    id: int
    patient_id: str
    doctor_id: str
    appointment_date: datetime
    duration_minutes: int
    status: str
    appointment_type: str
    notes: str | None
    
    class Config:
        from_attributes = True


@router.get("/", response_model=List[AppointmentResponse])
async def get_appointments(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if current_user.role == "doctor":
        appointments = db.query(Appointment).filter(
            Appointment.doctor_id == current_user.id
        ).all()
    else:
        appointments = db.query(Appointment).filter(
            Appointment.patient_id == current_user.id
        ).all()
    
    return appointments


@router.post("/", response_model=AppointmentResponse)
async def create_appointment(
    appointment_data: AppointmentCreate,
    current_user: User = Depends(get_current_doctor),
    db: Session = Depends(get_db)
):
    appointment = Appointment(
        doctor_id=current_user.id,
        **appointment_data.model_dump()
    )
    
    db.add(appointment)
    db.commit()
    db.refresh(appointment)
    
    return appointment
