"""
Lysa Clinical Documentation Drafts Models
SQLAlchemy models for AI-generated clinical documentation with provenance tracking.
"""

from sqlalchemy import Column, String, Integer, Text, DateTime, Boolean, ForeignKey, Index, Enum as SQLEnum
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func
from app.database import Base
import enum


class DraftStatus(enum.Enum):
    """Status of a Lysa draft document."""
    DRAFT = "draft"
    REVISED = "revised"
    APPROVED = "approved"
    REJECTED = "rejected"
    INSERTED_TO_CHART = "inserted_to_chart"


class DraftType(enum.Enum):
    """Type of clinical document."""
    DIFFERENTIAL = "differential"
    ASSESSMENT_PLAN = "assessment_plan"
    HISTORY_PHYSICAL = "history_physical"
    PROGRESS_NOTE = "progress_note"
    DISCHARGE_SUMMARY = "discharge_summary"
    CHART_SUMMARY = "chart_summary"


class LysaDraft(Base):
    """Stores AI-generated clinical documentation drafts with full provenance."""
    __tablename__ = "lysa_drafts"
    
    id = Column(String, primary_key=True, server_default=func.gen_random_uuid())
    
    patient_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    doctor_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    
    draft_type = Column(String, nullable=False, default="differential")
    status = Column(String, nullable=False, default="draft")
    
    question = Column(Text)
    
    content_json = Column(JSONB, nullable=False)
    
    raw_output = Column(Text)
    
    provenance = Column(JSONB, nullable=False, default=dict)
    
    ehr_sources_used = Column(JSONB)
    
    revision_count = Column(Integer, default=0)
    revision_history = Column(JSONB, default=list)
    
    approved_at = Column(DateTime(timezone=True))
    approved_by = Column(String)
    
    inserted_to_chart_at = Column(DateTime(timezone=True))
    ehr_note_id = Column(String)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    __table_args__ = (
        Index('idx_lysa_draft_patient', 'patient_id'),
        Index('idx_lysa_draft_doctor', 'doctor_id'),
        Index('idx_lysa_draft_status', 'status'),
        Index('idx_lysa_draft_created', 'created_at'),
    )


class LysaDraftAuditLog(Base):
    """Audit log for all Lysa draft operations - HIPAA compliance."""
    __tablename__ = "lysa_draft_audit_logs"
    
    id = Column(String, primary_key=True, server_default=func.gen_random_uuid())
    
    draft_id = Column(String, ForeignKey("lysa_drafts.id"), index=True)
    patient_id = Column(String, nullable=False, index=True)
    doctor_id = Column(String, nullable=False, index=True)
    
    action = Column(String, nullable=False)
    
    ehr_resources_accessed = Column(JSONB)
    
    request_hash = Column(String)
    response_hash = Column(String)
    
    model_used = Column(String)
    
    ip_address = Column(String)
    user_agent = Column(String)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    __table_args__ = (
        Index('idx_lysa_audit_patient', 'patient_id'),
        Index('idx_lysa_audit_doctor', 'doctor_id'),
        Index('idx_lysa_audit_action', 'action'),
        Index('idx_lysa_audit_created', 'created_at'),
    )


class CBTSession(Base):
    """Stores CBT therapy sessions with structured templates."""
    __tablename__ = "cbt_sessions"
    
    id = Column(String, primary_key=True, server_default=func.gen_random_uuid())
    
    patient_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    
    session_type = Column(String, nullable=False, default="thought_record")
    
    situation = Column(Text)
    automatic_thoughts = Column(Text)
    emotions = Column(JSONB)
    evidence_for = Column(Text)
    evidence_against = Column(Text)
    balanced_thought = Column(Text)
    action_plan = Column(Text)
    
    distress_before = Column(Integer)
    distress_after = Column(Integer)
    
    prompts_used = Column(JSONB)
    responses = Column(JSONB)
    
    crisis_detected = Column(Boolean, default=False)
    crisis_action_taken = Column(String)
    clinician_notified = Column(Boolean, default=False)
    clinician_notified_at = Column(DateTime(timezone=True))
    
    completed = Column(Boolean, default=False)
    completed_at = Column(DateTime(timezone=True))
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    __table_args__ = (
        Index('idx_cbt_session_patient', 'patient_id'),
        Index('idx_cbt_session_created', 'created_at'),
        Index('idx_cbt_session_crisis', 'crisis_detected'),
    )
