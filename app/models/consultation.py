from sqlalchemy import Column, String, DateTime, Text, Integer
from sqlalchemy.sql import func
from app.database import Base


class DoctorConsultation(Base):
    __tablename__ = "doctor_consultations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    requesting_doctor_id = Column(String, nullable=False, index=True)
    consulted_doctor_id = Column(String, nullable=False, index=True)
    patient_id = Column(String, nullable=False, index=True)
    
    consultation_reason = Column(Text, nullable=False)
    status = Column(String, default="pending")
    
    response = Column(Text, nullable=True)
    response_at = Column(DateTime, nullable=True)
    
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())


class ConsultationRecordAccess(Base):
    __tablename__ = "consultation_record_access"

    id = Column(Integer, primary_key=True, autoincrement=True)
    consultation_id = Column(Integer, nullable=False, index=True)
    doctor_id = Column(String, nullable=False, index=True)
    patient_id = Column(String, nullable=False, index=True)
    
    access_granted_at = Column(DateTime, server_default=func.now())
    accessed_at = Column(DateTime, nullable=True)
    
    created_at = Column(DateTime, server_default=func.now())
