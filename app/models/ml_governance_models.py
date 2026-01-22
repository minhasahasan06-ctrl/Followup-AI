"""
ML Governance Database Models
=============================
HIPAA-compliant model governance for clinical AI/ML:
1. Model metadata with training data hash and version tracking
2. Clinical validation protocols and sign-off requirements
3. Data provenance and lineage tracking
4. Human approval gates for production deployment
5. Research-only flagging for clinical models

All clinical models require explicit validation before production use.
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, JSON, Text, ForeignKey, Index, Enum as SQLEnum
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.database import Base
import enum


class ModelValidationStatus(str, enum.Enum):
    PENDING = "pending"
    IN_REVIEW = "in_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    REVOKED = "revoked"


class ModelDeploymentEnvironment(str, enum.Enum):
    RESEARCH_ONLY = "research_only"
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class ClinicalRiskLevel(str, enum.Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class MLModelGovernance(Base):
    """
    Extended governance metadata for ML models.
    Links to MLModel and adds clinical validation requirements.
    """
    __tablename__ = "ml_model_governance"

    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(String, ForeignKey("ml_models.id"), nullable=False, unique=True, index=True)
    
    training_data_hash = Column(String)
    training_data_version = Column(String)
    training_data_size = Column(Integer)
    training_data_date_range = Column(JSON)
    training_patient_count = Column(Integer)
    training_de_identified = Column(Boolean, default=True)
    
    is_clinical_model = Column(Boolean, default=False, index=True)
    clinical_risk_level = Column(SQLEnum(ClinicalRiskLevel), default=ClinicalRiskLevel.LOW)
    requires_human_review = Column(Boolean, default=False)
    allowed_environment = Column(SQLEnum(ModelDeploymentEnvironment), default=ModelDeploymentEnvironment.DEVELOPMENT)
    
    validation_status = Column(SQLEnum(ModelValidationStatus), default=ModelValidationStatus.PENDING, index=True)
    validation_protocol_id = Column(Integer, ForeignKey("ml_validation_protocols.id"))
    validated_by = Column(String)
    validated_at = Column(DateTime(timezone=True))
    validation_notes = Column(Text)
    
    evaluation_metrics = Column(JSON)
    benchmark_dataset_hash = Column(String)
    benchmark_results = Column(JSON)
    
    sensitivity = Column(Float)
    specificity = Column(Float)
    positive_predictive_value = Column(Float)
    negative_predictive_value = Column(Float)
    auc_roc = Column(Float)
    f1_score = Column(Float)
    
    bias_assessment = Column(JSON)
    fairness_metrics = Column(JSON)
    demographic_performance = Column(JSON)
    
    intended_use = Column(Text)
    contraindications = Column(Text)
    limitations = Column(Text)
    regulatory_notes = Column(Text)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    created_by = Column(String)
    
    validation_protocol = relationship("MLValidationProtocol", back_populates="governed_models")
    approvals = relationship("MLModelApproval", back_populates="governance")
    provenance_records = relationship("MLDataProvenance", back_populates="governance")
    
    __table_args__ = (
        Index('idx_gov_clinical', 'is_clinical_model', 'validation_status'),
        Index('idx_gov_environment', 'allowed_environment'),
    )


class MLValidationProtocol(Base):
    """
    Validation protocols for clinical model approval.
    Defines required tests, metrics thresholds, and sign-off requirements.
    """
    __tablename__ = "ml_validation_protocols"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False, unique=True)
    version = Column(String, nullable=False)
    description = Column(Text)
    
    model_types = Column(JSON)
    clinical_risk_levels = Column(JSON)
    
    required_metrics = Column(JSON)
    metric_thresholds = Column(JSON)
    required_test_datasets = Column(JSON)
    required_bias_tests = Column(JSON)
    
    required_approvers = Column(JSON)
    min_approver_count = Column(Integer, default=1)
    requires_clinical_review = Column(Boolean, default=False)
    requires_regulatory_review = Column(Boolean, default=False)
    
    approval_validity_days = Column(Integer, default=365)
    requires_periodic_revalidation = Column(Boolean, default=True)
    revalidation_interval_days = Column(Integer, default=90)
    
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    created_by = Column(String)
    
    governed_models = relationship("MLModelGovernance", back_populates="validation_protocol")
    
    __table_args__ = (
        Index('idx_protocol_active', 'is_active'),
    )


class MLModelApproval(Base):
    """
    Human approval records for model deployment.
    Tracks who approved what and when.
    """
    __tablename__ = "ml_model_approvals"

    id = Column(Integer, primary_key=True, index=True)
    governance_id = Column(Integer, ForeignKey("ml_model_governance.id"), nullable=False, index=True)
    
    approval_type = Column(String, nullable=False)
    approver_id = Column(String, nullable=False)
    approver_name = Column(String, nullable=False)
    approver_role = Column(String, nullable=False)
    
    decision = Column(String, nullable=False)
    decision_reason = Column(Text)
    conditions = Column(JSON)
    
    valid_from = Column(DateTime(timezone=True))
    valid_until = Column(DateTime(timezone=True))
    revoked_at = Column(DateTime(timezone=True))
    revoked_by = Column(String)
    revoke_reason = Column(Text)
    
    signature_hash = Column(String)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    governance = relationship("MLModelGovernance", back_populates="approvals")
    
    __table_args__ = (
        Index('idx_approval_governance', 'governance_id'),
        Index('idx_approval_approver', 'approver_id'),
        Index('idx_approval_decision', 'decision'),
    )


class MLDataProvenance(Base):
    """
    Data provenance and lineage tracking for model training data.
    Enables tracing what data was used to train models.
    """
    __tablename__ = "ml_data_provenance"

    id = Column(Integer, primary_key=True, index=True)
    governance_id = Column(Integer, ForeignKey("ml_model_governance.id"), nullable=False, index=True)
    
    dataset_id = Column(String, nullable=False, index=True)
    dataset_name = Column(String, nullable=False)
    dataset_version = Column(String, nullable=False)
    dataset_hash = Column(String, nullable=False)
    
    data_source = Column(String)
    source_system = Column(String)
    extraction_date = Column(DateTime(timezone=True))
    
    row_count = Column(Integer)
    column_count = Column(Integer)
    date_range_start = Column(DateTime(timezone=True))
    date_range_end = Column(DateTime(timezone=True))
    patient_count = Column(Integer)
    
    phi_handling = Column(String)
    de_identification_method = Column(String)
    k_anonymity_value = Column(Integer)
    
    consent_types = Column(JSON)
    consent_verification = Column(String)
    
    transformations = Column(JSON)
    preprocessing_script_hash = Column(String)
    feature_engineering_hash = Column(String)
    
    parent_dataset_id = Column(String)
    lineage_graph = Column(JSON)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    created_by = Column(String)
    
    governance = relationship("MLModelGovernance", back_populates="provenance_records")
    
    __table_args__ = (
        Index('idx_provenance_dataset', 'dataset_id'),
        Index('idx_provenance_governance', 'governance_id'),
    )


class MLAuditLog(Base):
    """
    Immutable audit log for all ML operations.
    HIPAA-compliant logging of model usage and access.
    """
    __tablename__ = "ml_audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    
    event_type = Column(String, nullable=False, index=True)
    event_subtype = Column(String)
    
    actor_id = Column(String, nullable=False, index=True)
    actor_type = Column(String, nullable=False)
    actor_role = Column(String)
    
    model_id = Column(String, index=True)
    patient_id = Column(String, index=True)
    
    action = Column(String, nullable=False)
    resource_type = Column(String)
    resource_id = Column(String)
    
    request_id = Column(String, index=True)
    session_id = Column(String)
    ip_address = Column(String)
    user_agent = Column(String)
    
    input_hash = Column(String)
    output_hash = Column(String)
    phi_accessed = Column(Boolean, default=False)
    phi_categories = Column(JSON)
    
    success = Column(Boolean, default=True)
    error_code = Column(String)
    error_message = Column(Text)
    
    duration_ms = Column(Float)
    event_metadata = Column(JSON)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    
    __table_args__ = (
        Index('idx_audit_event', 'event_type', 'created_at'),
        Index('idx_audit_actor', 'actor_id', 'created_at'),
        Index('idx_audit_patient', 'patient_id', 'created_at'),
        Index('idx_audit_model', 'model_id', 'created_at'),
    )


class EmbeddingStandardization(Base):
    """
    Track embedding standardization across the system.
    Ensures all embeddings have proper model attribution.
    """
    __tablename__ = "embedding_standardization"

    id = Column(Integer, primary_key=True, index=True)
    
    embedding_source = Column(String, nullable=False, index=True)
    table_name = Column(String, nullable=False, index=True)
    record_id = Column(String, nullable=False)
    
    embedding_model = Column(String)
    embedding_version = Column(String)
    embedding_dimension = Column(Integer)
    
    has_null_model = Column(Boolean, default=False, index=True)
    needs_re_embedding = Column(Boolean, default=False, index=True)
    
    last_checked_at = Column(DateTime(timezone=True), server_default=func.now())
    re_embedded_at = Column(DateTime(timezone=True))
    
    __table_args__ = (
        Index('idx_embed_std_source', 'embedding_source', 'table_name'),
        Index('idx_embed_std_null', 'has_null_model'),
    )
