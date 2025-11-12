from sqlalchemy import Column, String, DateTime, Text, Integer, Boolean
from sqlalchemy.sql import func
from app.database import Base


class Appointment(Base):
    __tablename__ = "appointments"

    id = Column(Integer, primary_key=True, autoincrement=True)
    patient_id = Column(String, nullable=False, index=True)
    doctor_id = Column(String, nullable=False, index=True)
    
    appointment_date = Column(DateTime, nullable=False)
    duration_minutes = Column(Integer, default=30)
    
    status = Column(String, default="scheduled")
    appointment_type = Column(String, nullable=False)
    
    notes = Column(Text, nullable=True)
    reminder_sent = Column(Boolean, default=False)
    
    google_calendar_event_id = Column(String, nullable=True)
    
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
