"""
SOAP Notes Model
Clinical documentation following Subjective, Objective, Assessment, Plan format
HIPAA-compliant with comprehensive audit logging
"""

from sqlalchemy import Column, String, DateTime, Text, Integer, Boolean, JSON, ForeignKey
from sqlalchemy.sql import func
from app.database import Base


class SOAPNote(Base):
    """
    SOAP Note - Structured clinical documentation
    S: Subjective - Patient's symptoms, complaints, history
    O: Objective - Physical exam findings, vitals, lab results
    A: Assessment - Diagnosis, differential diagnosis, ICD-10 codes
    P: Plan - Treatment plan, medications, follow-up
    """
    __tablename__ = "soap_notes"

    id = Column(String, primary_key=True)
    patient_id = Column(String, nullable=False, index=True)
    doctor_id = Column(String, nullable=False, index=True)
    encounter_date = Column(DateTime, nullable=False, server_default=func.now())
    
    subjective = Column(Text)
    chief_complaint = Column(String(500))
    history_present_illness = Column(Text)
    review_of_systems = Column(JSON)
    
    objective = Column(Text)
    vital_signs = Column(JSON)
    physical_exam = Column(Text)
    lab_results = Column(JSON)
    
    assessment = Column(Text)
    primary_diagnosis = Column(String(500))
    primary_icd10 = Column(String(20))
    secondary_diagnoses = Column(JSON)
    differential_diagnoses = Column(JSON)
    
    plan = Column(Text)
    medications_prescribed = Column(JSON)
    procedures_ordered = Column(JSON)
    referrals = Column(JSON)
    patient_education = Column(Text)
    follow_up_instructions = Column(Text)
    follow_up_date = Column(DateTime)
    
    status = Column(String(50), nullable=False, default="draft")
    signed_at = Column(DateTime)
    signed_by = Column(String)
    
    linked_prescription_ids = Column(JSON)
    linked_appointment_id = Column(String)
    
    ai_suggestions_used = Column(Boolean, default=False)
    ai_icd10_suggestions = Column(JSON)
    
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())


class ICD10Code(Base):
    """
    ICD-10 Code Reference Table
    For caching commonly used codes and AI suggestions
    """
    __tablename__ = "icd10_codes"

    id = Column(Integer, primary_key=True, autoincrement=True)
    code = Column(String(20), nullable=False, unique=True, index=True)
    description = Column(String(500), nullable=False)
    category = Column(String(100))
    chapter = Column(String(200))
    is_billable = Column(Boolean, default=True)
    
    usage_count = Column(Integer, default=0)
    last_used_at = Column(DateTime)
    
    created_at = Column(DateTime, server_default=func.now())


class ClinicalEncounter(Base):
    """
    Clinical Encounter - Links SOAP notes, prescriptions, and appointments
    Provides unified view of patient encounters
    """
    __tablename__ = "clinical_encounters"

    id = Column(String, primary_key=True)
    patient_id = Column(String, nullable=False, index=True)
    doctor_id = Column(String, nullable=False, index=True)
    encounter_type = Column(String(50), nullable=False)
    encounter_date = Column(DateTime, nullable=False, server_default=func.now())
    
    soap_note_id = Column(String)
    appointment_id = Column(String)
    prescription_ids = Column(JSON)
    
    chief_complaint = Column(String(500))
    disposition = Column(String(100))
    
    billing_codes = Column(JSON)
    cpt_codes = Column(JSON)
    
    status = Column(String(50), nullable=False, default="open")
    closed_at = Column(DateTime)
    closed_by = Column(String)
    
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
