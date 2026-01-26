"""
Terms Acceptance and Audit Log Models
HIPAA-compliant tracking of terms acceptance and key user events
"""

from sqlalchemy import Column, Integer, String, DateTime, Text, Boolean
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func
from app.database import Base


class TermsAcceptance(Base):
    """Records user acceptance of Terms & Conditions with full audit trail"""
    __tablename__ = "terms_acceptances"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, nullable=False, index=True)
    terms_version = Column(String(20), nullable=False)
    accepted_at = Column(DateTime(timezone=True), nullable=False)
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(Text, nullable=True)
    research_consent = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class UserAuditLog(Base):
    """User-facing audit log for key events (terms, consent, profile changes)"""
    __tablename__ = "user_audit_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, nullable=False, index=True)
    event_type = Column(String(50), nullable=False, index=True)
    event_data = Column(JSONB, default={})
    actor_id = Column(String, nullable=True)
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)


class PatientProfileExtended(Base):
    """Extended patient profile with clinical data"""
    __tablename__ = "patient_profiles_extended"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, unique=True, nullable=False, index=True)
    assigned_doctor_id = Column(String, nullable=True, index=True)
    emergency_contacts = Column(JSONB, default=[])
    medications = Column(JSONB, default=[])
    allergies = Column(JSONB, default=[])
    chronic_conditions = Column(JSONB, default=[])
    recent_labs = Column(JSONB, default=[])
    hospitalizations = Column(JSONB, default=[])
    connected_devices = Column(JSONB, default=[])
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class DoctorProfileExtended(Base):
    """Extended doctor profile with professional data"""
    __tablename__ = "doctor_profiles_extended"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, unique=True, nullable=False, index=True)
    npi = Column(String(10), nullable=True)
    hospital_affiliations = Column(JSONB, default=[])
    board_certifications = Column(JSONB, default=[])
    languages = Column(JSONB, default=[])
    accepted_insurances = Column(JSONB, default=[])
    telemedicine_available = Column(Boolean, default=False)
    telemedicine_fee = Column(Integer, nullable=True)
    consultation_fee = Column(Integer, nullable=True)
    publications = Column(JSONB, default=[])
    malpractice_info = Column(JSONB, default={})
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
