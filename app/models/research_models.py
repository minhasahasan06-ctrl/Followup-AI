"""
Phase 10: Research Center + Medical NLP Models
Production-grade SQLAlchemy models for cohort building, studies, document analysis.
All models include HIPAA compliance fields and k-anonymity support.
"""

from sqlalchemy import (
    Column, Integer, String, Text, Boolean, Float, DateTime, Date, JSON, 
    ForeignKey, Enum as SQLEnum, Index, CheckConstraint
)
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
import enum
from uuid import uuid4

from app.database import Base


class JobStatus(str, enum.Enum):
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class PHILevel(str, enum.Enum):
    NONE = "none"
    DE_IDENTIFIED = "de_identified"
    LIMITED = "limited"
    FULL = "full"


class VisibilityScope(str, enum.Enum):
    PRIVATE = "private"
    STUDY = "study"
    ORGANIZATION = "organization"
    PUBLIC = "public"


class AnalysisArtifact(Base):
    """
    Analysis artifacts storage for job outputs (plots, models, reports).
    Supports S3 or local storage with signed URL access.
    """
    __tablename__ = "analysis_artifacts"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    job_id = Column(String, ForeignKey("study_jobs.id"), nullable=True, index=True)
    study_id = Column(String, index=True)
    
    artifact_type = Column(String, nullable=False)
    format = Column(String, nullable=False)
    filename = Column(String, nullable=False)
    
    storage_uri = Column(String, nullable=False)
    size_bytes = Column(Integer)
    checksum = Column(String)
    
    phi_level = Column(String, default=PHILevel.DE_IDENTIFIED.value)
    visibility_scope = Column(String, default=VisibilityScope.STUDY.value)
    retention_days = Column(Integer, default=365)
    expires_at = Column(DateTime)
    
    metadata_json = Column(JSON)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    created_by = Column(String, nullable=False)
    deleted_at = Column(DateTime)
    
    __table_args__ = (
        Index('idx_artifact_study', 'study_id'),
        Index('idx_artifact_type', 'artifact_type'),
        Index('idx_artifact_expires', 'expires_at'),
    )


class ResearchDataset(Base):
    """
    Immutable versioned datasets for research studies.
    Tracks lineage and schema changes.
    """
    __tablename__ = "research_datasets"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    study_id = Column(String, index=True)
    cohort_snapshot_id = Column(String, ForeignKey("research_cohort_snapshots.id"), index=True)
    
    name = Column(String, nullable=False)
    description = Column(Text)
    version = Column(Integer, nullable=False, default=1)
    
    storage_uri = Column(String, nullable=False)
    format = Column(String, default="parquet")
    
    row_count = Column(Integer)
    column_count = Column(Integer)
    columns_json = Column(JSON)
    schema_hash = Column(String)
    checksum = Column(String)
    
    pii_classification = Column(String, default=PHILevel.DE_IDENTIFIED.value)
    is_immutable = Column(Boolean, default=True)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    created_by = Column(String, nullable=False)
    
    __table_args__ = (
        Index('idx_dataset_study_version', 'study_id', 'version'),
        Index('idx_dataset_cohort', 'cohort_snapshot_id'),
    )


class DatasetLineage(Base):
    """
    Tracks parent-child relationships between datasets for reproducibility.
    """
    __tablename__ = "dataset_lineage"
    
    id = Column(Integer, primary_key=True)
    parent_dataset_id = Column(String, ForeignKey("research_datasets.id"), nullable=False, index=True)
    child_dataset_id = Column(String, ForeignKey("research_datasets.id"), nullable=False, index=True)
    
    transformation_type = Column(String, nullable=False)
    transformation_params = Column(JSON)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    __table_args__ = (
        Index('idx_lineage_parent_child', 'parent_dataset_id', 'child_dataset_id'),
    )


class StudyJob(Base):
    """
    Background job orchestration for study analyses.
    Supports retries, progress tracking, and audit logging.
    """
    __tablename__ = "study_jobs"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    study_id = Column(String, nullable=False, index=True)
    
    job_type = Column(String, nullable=False)
    status = Column(String, default=JobStatus.PENDING.value, index=True)
    priority = Column(Integer, default=5)
    
    payload_json = Column(JSON)
    result_json = Column(JSON)
    error_log = Column(Text)
    
    progress = Column(Integer, default=0)
    progress_message = Column(String)
    
    retry_count = Column(Integer, default=0)
    max_retries = Column(Integer, default=3)
    
    scheduled_at = Column(DateTime)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    
    created_by = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    events = relationship("StudyJobEvent", back_populates="job", cascade="all, delete-orphan")
    artifacts = relationship("AnalysisArtifact", backref="job", foreign_keys=[AnalysisArtifact.job_id])
    
    __table_args__ = (
        Index('idx_job_study_status', 'study_id', 'status'),
        Index('idx_job_type_status', 'job_type', 'status'),
        Index('idx_job_scheduled', 'scheduled_at'),
        CheckConstraint('progress >= 0 AND progress <= 100', name='check_progress_range'),
    )


class StudyJobEvent(Base):
    """
    Audit trail for job state changes.
    """
    __tablename__ = "study_job_events"
    
    id = Column(Integer, primary_key=True)
    job_id = Column(String, ForeignKey("study_jobs.id"), nullable=False, index=True)
    
    event_type = Column(String, nullable=False)
    old_value = Column(String)
    new_value = Column(String)
    message = Column(Text)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    job = relationship("StudyJob", back_populates="events")


class ResearchCohortSnapshot(Base):
    """
    Point-in-time snapshot of cohort query results.
    Applies k-anonymity suppression for privacy.
    """
    __tablename__ = "research_cohort_snapshots"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    cohort_id = Column(String, index=True)
    
    filter_json = Column(JSON, nullable=False)
    query_sql = Column(Text)
    
    patient_count = Column(Integer)
    patient_ids_hash = Column(String)
    
    metrics_json = Column(JSON)
    
    suppressed = Column(Boolean, default=False)
    suppression_reason = Column(String)
    k_anonymity_applied = Column(Boolean, default=True)
    
    computed_at = Column(DateTime(timezone=True), server_default=func.now())
    expires_at = Column(DateTime)
    created_by = Column(String, nullable=False)
    
    __table_args__ = (
        Index('idx_snapshot_cohort', 'cohort_id'),
        Index('idx_snapshot_computed', 'computed_at'),
    )


class ResearchExport(Base):
    """
    Export job tracking for CSV/JSON dataset exports.
    """
    __tablename__ = "research_exports"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    dataset_id = Column(String, ForeignKey("research_datasets.id"), index=True)
    study_id = Column(String, index=True)
    
    format = Column(String, nullable=False)
    status = Column(String, default=JobStatus.PENDING.value)
    
    include_phi = Column(Boolean, default=False)
    columns_included = Column(JSON)
    filters_applied = Column(JSON)
    
    storage_uri = Column(String)
    file_size_bytes = Column(Integer)
    row_count = Column(Integer)
    
    signed_url = Column(Text)
    signed_url_expires_at = Column(DateTime)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime)
    created_by = Column(String, nullable=False)
    
    __table_args__ = (
        Index('idx_export_dataset', 'dataset_id'),
        Index('idx_export_status', 'status'),
    )


class NLPDocument(Base):
    """
    Clinical documents for NLP processing and PHI redaction.
    """
    __tablename__ = "nlp_documents"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    study_id = Column(String, index=True)
    patient_id = Column(String, index=True)
    
    source_type = Column(String, nullable=False)
    source_uri = Column(String, nullable=False)
    
    status = Column(String, default="pending", index=True)
    
    phi_detected_json = Column(JSON)
    phi_count = Column(Integer, default=0)
    
    redacted_uri = Column(String)
    redaction_method = Column(String)
    
    processed_at = Column(DateTime)
    processing_time_ms = Column(Integer)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    created_by = Column(String)
    
    redaction_runs = relationship("NLPRedactionRun", back_populates="document", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index('idx_nlp_doc_study', 'study_id'),
        Index('idx_nlp_doc_status', 'status'),
        Index('idx_nlp_doc_patient', 'patient_id'),
    )


class NLPRedactionRun(Base):
    """
    Audit trail for PHI redaction operations.
    """
    __tablename__ = "nlp_redaction_runs"
    
    id = Column(Integer, primary_key=True)
    document_id = Column(String, ForeignKey("nlp_documents.id"), nullable=False, index=True)
    
    model_name = Column(String, nullable=False)
    model_version = Column(String)
    
    entities_detected = Column(Integer, default=0)
    entities_redacted = Column(Integer, default=0)
    entity_types_json = Column(JSON)
    
    confidence_threshold = Column(Float, default=0.8)
    
    reviewer_id = Column(String)
    reviewed_at = Column(DateTime)
    review_status = Column(String)
    
    audit_log_id = Column(String)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    document = relationship("NLPDocument", back_populates="redaction_runs")


class ResearchQASession(Base):
    """
    Q&A conversation sessions with AI research assistant.
    """
    __tablename__ = "research_qa_sessions"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    user_id = Column(String, nullable=False, index=True)
    study_id = Column(String, index=True)
    dataset_id = Column(String, ForeignKey("research_datasets.id"), index=True)
    
    title = Column(String)
    status = Column(String, default="active")
    
    total_tokens = Column(Integer, default=0)
    total_messages = Column(Integer, default=0)
    
    context_json = Column(JSON)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    last_message_at = Column(DateTime)
    
    messages = relationship("ResearchQAMessage", back_populates="session", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index('idx_qa_session_user', 'user_id'),
        Index('idx_qa_session_study', 'study_id'),
    )


class ResearchQAMessage(Base):
    """
    Individual messages in Q&A sessions.
    """
    __tablename__ = "research_qa_messages"
    
    id = Column(Integer, primary_key=True)
    session_id = Column(String, ForeignKey("research_qa_sessions.id"), nullable=False, index=True)
    
    role = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    
    references_json = Column(JSON)
    token_usage = Column(Integer, default=0)
    
    model_name = Column(String)
    response_time_ms = Column(Integer)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    session = relationship("ResearchQASession", back_populates="messages")


class ResearchCohort(Base):
    """
    Saved cohort definitions with filter criteria.
    """
    __tablename__ = "research_cohorts"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    name = Column(String, nullable=False)
    description = Column(Text)
    
    filters_json = Column(JSON, nullable=False)
    
    patient_count = Column(Integer)
    last_computed_at = Column(DateTime)
    
    is_active = Column(Boolean, default=True)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    created_by = Column(String, nullable=False)
    
    __table_args__ = (
        Index('idx_cohort_name', 'name'),
        Index('idx_cohort_active', 'is_active'),
    )


class ResearchStudy(Base):
    """
    Research study definitions with lifecycle management.
    """
    __tablename__ = "research_studies"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    title = Column(String, nullable=False)
    description = Column(Text)
    
    status = Column(String, default="planning", index=True)
    
    cohort_id = Column(String, ForeignKey("research_cohorts.id"), index=True)
    
    start_date = Column(Date)
    end_date = Column(Date)
    
    target_enrollment = Column(Integer, default=100)
    current_enrollment = Column(Integer, default=0)
    
    inclusion_criteria = Column(Text)
    exclusion_criteria = Column(Text)
    
    auto_reanalysis = Column(Boolean, default=False)
    reanalysis_frequency = Column(String)
    last_reanalysis_at = Column(DateTime)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    created_by = Column(String, nullable=False)
    
    __table_args__ = (
        Index('idx_study_status', 'status'),
        Index('idx_study_cohort', 'cohort_id'),
    )
