"""
Tinker Thinking Machine Database Models
NON-BAA Mode - Tinker does NOT have a BAA, never store PHI in these tables

Phase A: Platform foundations for Tinker integration including:
- AI audit logs (SHA256 hashes only, no raw payloads)
- Patient AI experiences and feedback
- Research cohorts, studies, protocols, trials
- ML governance: job reports, metrics, thresholds, drift tracking
"""

from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Boolean, 
    JSON, Text, ForeignKey, Index, Enum as SQLEnum
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import UUID
from app.database import Base
import enum
import uuid


class TinkerPurpose(str, enum.Enum):
    """Valid purposes for Tinker API calls - determines allowed payload keys"""
    PATIENT_QUESTIONS = "patient_questions"
    PATIENT_TEMPLATES = "patient_templates"
    COHORT_BUILDER = "cohort_builder"
    COHORT_ANALYSIS = "cohort_analysis"
    STUDY_PROTOCOL = "study_protocol"
    RESEARCH_ANALYSIS = "research_analysis"
    CLINICAL_TRIAL = "clinical_trial"
    JOB_PLANNER = "job_planner"
    MODEL_CARD = "model_card"
    MODEL_EVALUATION = "model_evaluation"
    THRESHOLD_OPTIMIZATION = "threshold_optimization"
    DRIFT_SUMMARY = "drift_summary"
    DRIFT_DETECTION = "drift_detection"


class ActorRole(str, enum.Enum):
    """Actor roles for audit logging"""
    PATIENT = "patient"
    DOCTOR = "doctor"
    ADMIN = "admin"
    SYSTEM = "system"


class FeedbackRating(str, enum.Enum):
    """Patient feedback ratings"""
    HELPFUL = "helpful"
    NOT_HELPFUL = "not_helpful"
    NEUTRAL = "neutral"


class TrialStatus(str, enum.Enum):
    """Trial run statuses"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class DriftSeverity(str, enum.Enum):
    """Drift alert severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AIAuditLog(Base):
    """
    Audit log for all Tinker API calls.
    CRITICAL: Only stores SHA256 hashes of payloads, NEVER raw data.
    """
    __tablename__ = "ai_audit_logs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    purpose = Column(String(50), nullable=False, index=True)
    actor_role = Column(String(20), nullable=False, index=True)
    actor_id = Column(String(255), index=True)
    payload_hash = Column(String(64), nullable=False)
    response_hash = Column(String(64))
    model_used = Column(String(100))
    latency_ms = Column(Float)
    success = Column(Boolean, default=True)
    error_code = Column(String(50))
    k_anon_verified = Column(Boolean, default=True)
    tinker_mode = Column(String(20), default="NON_BAA")
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)

    __table_args__ = (
        Index('idx_ai_audit_purpose_created', 'purpose', 'created_at'),
        Index('idx_ai_audit_actor', 'actor_role', 'actor_id'),
    )


class PatientAIExperience(Base):
    """
    Track AI-generated experiences shown to patients.
    Used for feedback collection and experience improvement.
    """
    __tablename__ = "patient_ai_experiences"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    patient_id = Column(String(255), nullable=False, index=True)
    experience_type = Column(String(50), nullable=False, index=True)
    packet_hash = Column(String(64), nullable=False)
    question_ids_json = Column(JSON)
    template_ids_json = Column(JSON)
    habit_ids_json = Column(JSON)
    action_ids_json = Column(JSON)
    audit_log_id = Column(UUID(as_uuid=True), ForeignKey("ai_audit_logs.id"))
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    expires_at = Column(DateTime(timezone=True))

    __table_args__ = (
        Index('idx_patient_exp_patient_type', 'patient_id', 'experience_type'),
        Index('idx_patient_exp_created', 'created_at'),
    )


class PatientFeedback(Base):
    """
    Patient feedback on AI-generated experiences.
    Used to improve question/template selection.
    """
    __tablename__ = "patient_feedback"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    patient_id = Column(String(255), nullable=False, index=True)
    experience_id = Column(UUID(as_uuid=True), ForeignKey("patient_ai_experiences.id"), nullable=False)
    rating = Column(String(20), nullable=False)
    reason_code = Column(String(50))
    additional_context = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)

    experience = relationship("PatientAIExperience", backref="feedback")

    __table_args__ = (
        Index('idx_patient_feedback_patient', 'patient_id'),
        Index('idx_patient_feedback_rating', 'rating'),
    )


class TinkerCohortDefinition(Base):
    """
    Cohort definitions created via NL Cohort Builder.
    Stores CohortDSL JSON for reproducible cohort queries.
    """
    __tablename__ = "tinker_cohort_definitions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    dsl_json = Column(JSON, nullable=False)
    dsl_hash = Column(String(64), nullable=False)
    nl_query_hash = Column(String(64))
    schema_version = Column(String(20), default="1.0")
    is_validated = Column(Boolean, default=False)
    validation_errors = Column(JSON)
    created_by = Column(String(255), nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    is_archived = Column(Boolean, default=False)

    snapshots = relationship("TinkerCohortSnapshot", back_populates="cohort_definition")
    studies = relationship("TinkerStudy", back_populates="cohort_definition")

    __table_args__ = (
        Index('idx_tinker_cohort_created_by', 'created_by'),
        Index('idx_tinker_cohort_name', 'name'),
    )


class TinkerCohortSnapshot(Base):
    """
    Point-in-time snapshots of cohort membership.
    Only stores k-anonymized aggregates, never individual patient IDs.
    """
    __tablename__ = "tinker_cohort_snapshots"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    cohort_id = Column(UUID(as_uuid=True), ForeignKey("tinker_cohort_definitions.id"), nullable=False)
    snapshot_hash = Column(String(64), nullable=False)
    patient_count = Column(Integer, nullable=False)
    k_anon_passed = Column(Boolean, nullable=False, default=True)
    k_threshold = Column(Integer, default=25)
    aggregate_stats = Column(JSON)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    created_by = Column(String(255))

    cohort_definition = relationship("TinkerCohortDefinition", back_populates="snapshots")

    __table_args__ = (
        Index('idx_cohort_snapshot_cohort', 'cohort_id'),
        Index('idx_cohort_snapshot_created', 'created_at'),
    )


class TinkerStudyProtocol(Base):
    """
    AI-generated study protocols.
    Contains methodology, analysis plan, confounders, limitations.
    """
    __tablename__ = "tinker_study_protocols"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    objective_hash = Column(String(64), nullable=False)
    protocol_json = Column(JSON, nullable=False)
    protocol_version = Column(String(20), default="1.0")
    analysis_types = Column(JSON)
    confounders = Column(JSON)
    sensitivity_analyses = Column(JSON)
    limitations = Column(JSON)
    bias_warnings = Column(JSON)
    audit_log_id = Column(UUID(as_uuid=True), ForeignKey("ai_audit_logs.id"))
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    created_by = Column(String(255))

    __table_args__ = (
        Index('idx_study_protocol_created', 'created_at'),
    )


class TinkerStudy(Base):
    """
    Research studies linking cohorts, protocols, and trials.
    """
    __tablename__ = "tinker_studies"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    cohort_id = Column(UUID(as_uuid=True), ForeignKey("tinker_cohort_definitions.id"))
    protocol_id = Column(UUID(as_uuid=True), ForeignKey("tinker_study_protocols.id"))
    status = Column(String(20), default="draft", index=True)
    reproducibility_hash = Column(String(64))
    preregistration_json = Column(JSON)
    created_by = Column(String(255), nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    is_archived = Column(Boolean, default=False)

    cohort_definition = relationship("TinkerCohortDefinition", back_populates="studies")
    protocol = relationship("TinkerStudyProtocol")
    trial_specs = relationship("TinkerTrialSpec", back_populates="study")

    __table_args__ = (
        Index('idx_study_created_by', 'created_by'),
        Index('idx_study_status', 'status'),
    )


class TinkerTrialSpec(Base):
    """
    Trial emulation specifications generated by Tinker.
    """
    __tablename__ = "tinker_trial_specs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    study_id = Column(UUID(as_uuid=True), ForeignKey("tinker_studies.id"), nullable=False)
    name = Column(String(255))
    spec_json = Column(JSON, nullable=False)
    spec_hash = Column(String(64), nullable=False)
    treatment_definition = Column(JSON)
    outcome_definition = Column(JSON)
    eligibility_criteria = Column(JSON)
    follow_up_period = Column(JSON)
    audit_log_id = Column(UUID(as_uuid=True), ForeignKey("ai_audit_logs.id"))
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    created_by = Column(String(255))

    study = relationship("TinkerStudy", back_populates="trial_specs")
    trial_runs = relationship("TinkerTrialRun", back_populates="trial_spec")

    __table_args__ = (
        Index('idx_trial_spec_study', 'study_id'),
    )


class TinkerTrialRun(Base):
    """
    Execution records for trial emulations.
    Stores results and reproducibility information.
    """
    __tablename__ = "tinker_trial_runs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    trial_spec_id = Column(UUID(as_uuid=True), ForeignKey("tinker_trial_specs.id"), nullable=False)
    status = Column(String(20), default="pending", index=True)
    results_json = Column(JSON)
    results_hash = Column(String(64))
    sample_size = Column(Integer)
    k_anon_passed = Column(Boolean)
    execution_time_seconds = Column(Float)
    error_message = Column(Text)
    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    created_by = Column(String(255))

    trial_spec = relationship("TinkerTrialSpec", back_populates="trial_runs")

    __table_args__ = (
        Index('idx_trial_run_spec', 'trial_spec_id'),
        Index('idx_trial_run_status', 'status'),
    )


class TinkerJobReport(Base):
    """
    Reports generated from ML training jobs.
    Contains validation results, leakage scans, etc.
    """
    __tablename__ = "tinker_job_reports"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    job_id = Column(String(255), nullable=False, index=True)
    report_type = Column(String(50), nullable=False, index=True)
    report_json = Column(JSON, nullable=False)
    report_hash = Column(String(64))
    job_plan_json = Column(JSON)
    audit_log_id = Column(UUID(as_uuid=True), ForeignKey("ai_audit_logs.id"))
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    created_by = Column(String(255))

    __table_args__ = (
        Index('idx_job_report_job', 'job_id'),
        Index('idx_job_report_type', 'report_type'),
    )


class TinkerModelMetrics(Base):
    """
    Model performance metrics for governance and monitoring.
    Stores subgroup metrics, calibration results, etc.
    """
    __tablename__ = "tinker_model_metrics"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_id = Column(String(255), nullable=False, index=True)
    model_version = Column(String(50))
    metrics_json = Column(JSON, nullable=False)
    subgroup_metrics = Column(JSON)
    calibration_metrics = Column(JSON)
    fairness_metrics = Column(JSON)
    feature_importance = Column(JSON)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    created_by = Column(String(255))

    __table_args__ = (
        Index('idx_model_metrics_model', 'model_id'),
    )


class TinkerThresholdProfile(Base):
    """
    Optimized threshold profiles for ML models.
    Balances alert burden vs sensitivity.
    """
    __tablename__ = "tinker_threshold_profiles"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_id = Column(String(255), nullable=False, index=True)
    profile_name = Column(String(100))
    thresholds_json = Column(JSON, nullable=False)
    optimization_target = Column(String(50))
    alert_budget = Column(JSON)
    performance_at_threshold = Column(JSON)
    is_active = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    created_by = Column(String(255))

    __table_args__ = (
        Index('idx_threshold_model', 'model_id'),
        Index('idx_threshold_active', 'model_id', 'is_active'),
    )


class TinkerDriftRun(Base):
    """
    Drift detection runs for deployed models.
    Tracks feature distribution changes over time.
    """
    __tablename__ = "tinker_drift_runs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_id = Column(String(255), nullable=False, index=True)
    reference_period_start = Column(DateTime(timezone=True))
    reference_period_end = Column(DateTime(timezone=True))
    comparison_period_start = Column(DateTime(timezone=True))
    comparison_period_end = Column(DateTime(timezone=True))
    drift_metrics_json = Column(JSON, nullable=False)
    feature_drifts = Column(JSON)
    psi_score = Column(Float)
    kl_divergence = Column(Float)
    overall_drift_detected = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    created_by = Column(String(255))

    alerts = relationship("TinkerDriftAlert", back_populates="drift_run")

    __table_args__ = (
        Index('idx_drift_run_model', 'model_id'),
        Index('idx_drift_run_created', 'created_at'),
    )


class TinkerDriftAlert(Base):
    """
    Alerts generated from drift detection.
    Notifies when model performance may be degrading.
    """
    __tablename__ = "tinker_drift_alerts"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    drift_run_id = Column(UUID(as_uuid=True), ForeignKey("tinker_drift_runs.id"), nullable=False)
    alert_type = Column(String(50), nullable=False)
    severity = Column(String(20), nullable=False, index=True)
    feature_name = Column(String(255))
    drift_score = Column(Float)
    threshold_exceeded = Column(Float)
    recommendation = Column(Text)
    acknowledged = Column(Boolean, default=False)
    acknowledged_by = Column(String(255))
    acknowledged_at = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)

    drift_run = relationship("TinkerDriftRun", back_populates="alerts")

    __table_args__ = (
        Index('idx_drift_alert_run', 'drift_run_id'),
        Index('idx_drift_alert_severity', 'severity'),
        Index('idx_drift_alert_ack', 'acknowledged'),
    )


class TinkerGovernancePack(Base):
    """
    Governance documentation packs for ML models.
    Required before model deployment.
    """
    __tablename__ = "tinker_governance_packs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_id = Column(String(255), nullable=False, index=True)
    model_version = Column(String(50))
    model_card_json = Column(JSON)
    datasheet_json = Column(JSON)
    validation_summary = Column(JSON)
    calibration_summary = Column(JSON)
    drift_config = Column(JSON)
    reproducibility_hash = Column(String(64))
    is_complete = Column(Boolean, default=False)
    is_approved = Column(Boolean, default=False)
    approved_by = Column(String(255))
    approved_at = Column(DateTime(timezone=True))
    audit_log_id = Column(UUID(as_uuid=True), ForeignKey("ai_audit_logs.id"))
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    created_by = Column(String(255))

    __table_args__ = (
        Index('idx_governance_model', 'model_id'),
        Index('idx_governance_approved', 'is_approved'),
    )
