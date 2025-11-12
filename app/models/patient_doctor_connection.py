from sqlalchemy import Column, String, Integer, Text, DateTime
from sqlalchemy.sql import func
from app.database import Base


class PatientDoctorConnection(Base):
    """Tracks which doctors are connected to which patients"""
    __tablename__ = "patient_doctor_connections"

    id = Column(Integer, primary_key=True, autoincrement=True)
    patient_id = Column(String, nullable=False, index=True)
    doctor_id = Column(String, nullable=False, index=True)
    
    status = Column(String, default="pending", index=True)
    connection_type = Column(String, default="primary_care")
    
    notes = Column(Text, nullable=True)
    
    requested_at = Column(DateTime, server_default=func.now())
    connected_at = Column(DateTime, nullable=True)
    disconnected_at = Column(DateTime, nullable=True)
    
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())


class PatientConsultation(Base):
    """Patient requesting consultation with a doctor"""
    __tablename__ = "patient_consultations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    patient_id = Column(String, nullable=False, index=True)
    doctor_id = Column(String, nullable=False, index=True)
    
    consultation_reason = Column(Text, nullable=False)
    symptoms = Column(Text, nullable=True)
    urgency = Column(String, default="routine")
    mode = Column(String, default="video")
    
    status = Column(String, default="requested", index=True)
    
    scheduled_for = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    cancelled_at = Column(DateTime, nullable=True)
    cancellation_reason = Column(Text, nullable=True)
    
    doctor_notes = Column(Text, nullable=True)
    
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())


class AISymptomSession(Base):
    """Stores AI symptom analysis sessions with Agent Clona"""
    __tablename__ = "ai_symptom_sessions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    patient_id = Column(String, nullable=False, index=True)
    
    initial_symptoms = Column(Text, nullable=False)
    differential_diagnoses = Column(Text, nullable=True)
    recommended_tests = Column(Text, nullable=True)
    physical_exam_findings = Column(Text, nullable=True)
    treatment_suggestions = Column(Text, nullable=True)
    
    suggested_doctor_ids = Column(Text, nullable=True)
    suggested_specialties = Column(Text, nullable=True)
    
    session_transcript = Column(Text, nullable=True)
    
    status = Column(String, default="active")
    
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
