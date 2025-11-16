"""
Security & Compliance Database Models
Consent management, audit logging, data retention, HIPAA compliance
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, JSON, Text, ForeignKey, Index
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.database import Base


class ConsentRecord(Base):
    """Track patient consent for data collection and sharing"""
    __tablename__ = "consent_records"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(String, nullable=False, index=True)
    
    # Consent type
    consent_type = Column(String, nullable=False)  # "video_collection", "audio_collection", "data_sharing", "research", "ai_analysis"
    consent_category = Column(String)  # "required", "optional", "research"
    
    # Consent status
    consent_given = Column(Boolean, nullable=False)
    consent_version = Column(String)  # Version of consent form
    
    # Consent details
    consent_text = Column(Text)  # Full consent text shown to patient
    patient_signature = Column(String)  # Digital signature or IP confirmation
    witness_signature = Column(String)  # If applicable
    
    # Granular permissions
    allow_video_recording = Column(Boolean, default=False)
    allow_audio_recording = Column(Boolean, default=False)
    allow_ai_analysis = Column(Boolean, default=False)
    allow_data_sharing_clinician = Column(Boolean, default=True)
    allow_data_sharing_research = Column(Boolean, default=False)
    allow_third_party_analytics = Column(Boolean, default=False)
    
    # Withdrawal
    can_withdraw = Column(Boolean, default=True)
    withdrawn = Column(Boolean, default=False)
    withdrawn_at = Column(DateTime(timezone=True))
    withdrawal_reason = Column(Text)
    
    # Expiration
    expires_at = Column(DateTime(timezone=True))
    requires_renewal = Column(Boolean, default=False)
    renewal_frequency_days = Column(Integer)  # How often to renew consent
    
    # Audit trail
    consent_given_at = Column(DateTime(timezone=True), server_default=func.now())
    ip_address = Column(String)
    user_agent = Column(String)
    geo_location = Column(String)  # Country/state for jurisdiction
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    __table_args__ = (
        Index('idx_consent_patient_type', 'patient_id', 'consent_type'),
        Index('idx_consent_status', 'consent_given'),
        Index('idx_consent_withdrawn', 'withdrawn'),
    )


class DataRetentionPolicy(Base):
    """Define data retention policies for HIPAA compliance"""
    __tablename__ = "data_retention_policies"

    id = Column(Integer, primary_key=True, index=True)
    
    # Policy details
    policy_name = Column(String, nullable=False, unique=True)
    data_type = Column(String, nullable=False)  # "video", "audio", "metrics", "alerts", "logs"
    description = Column(Text)
    
    # Retention period
    retention_days = Column(Integer, nullable=False)  # How long to keep data
    archive_before_delete = Column(Boolean, default=True)  # Archive to cold storage
    archive_location = Column(String)  # S3 Glacier, etc.
    
    # Deletion rules
    auto_delete_enabled = Column(Boolean, default=True)
    deletion_method = Column(String, default="secure_wipe")  # "secure_wipe", "archive", "anonymize"
    requires_approval = Column(Boolean, default=False)  # Require manual approval before deletion
    
    # Exceptions
    legal_hold_override = Column(Boolean, default=True)  # Don't delete if legal hold
    active_treatment_override = Column(Boolean, default=True)  # Don't delete if patient still active
    
    # Compliance
    regulatory_basis = Column(String)  # "HIPAA", "GDPR", "state_law", etc.
    audit_logging_required = Column(Boolean, default=True)
    
    # Active status
    is_active = Column(Boolean, default=True)
    applies_to_all_patients = Column(Boolean, default=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    created_by = Column(String)
    
    __table_args__ = (
        Index('idx_retention_data_type', 'data_type'),
        Index('idx_retention_active', 'is_active'),
    )


class AuditLog(Base):
    """Comprehensive audit logging for HIPAA compliance"""
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    
    # Actor (who performed the action)
    user_id = Column(String, nullable=False, index=True)
    user_type = Column(String, nullable=False)  # "patient", "clinician", "admin", "system"
    user_email = Column(String)
    user_role = Column(String)
    
    # Action details
    action_type = Column(String, nullable=False, index=True)  # "view", "create", "update", "delete", "download", "share"
    action_category = Column(String)  # "phi_access", "configuration", "alert", "user_management"
    resource_type = Column(String, nullable=False)  # "patient_record", "video", "audio", "alert", "user"
    resource_id = Column(String)  # ID of the resource accessed
    
    # PHI access tracking
    phi_accessed = Column(Boolean, default=False, index=True)
    patient_id_accessed = Column(String, index=True)  # Which patient's data was accessed
    data_fields_accessed = Column(JSON)  # Specific fields viewed/modified
    
    # Action context
    action_description = Column(Text)  # Human-readable description
    action_result = Column(String)  # "success", "failure", "partial"
    error_message = Column(Text)  # If action failed
    
    # Request details
    ip_address = Column(String, index=True)
    user_agent = Column(String)
    request_id = Column(String)  # Correlation ID for request tracing
    session_id = Column(String)  # User session ID
    
    # Changes made (for update/delete actions)
    before_value = Column(JSON)  # State before change
    after_value = Column(JSON)  # State after change
    change_reason = Column(Text)  # Why change was made
    
    # Authorization
    authorization_method = Column(String)  # "jwt", "api_key", "oauth"
    permission_level = Column(String)  # Permission used for action
    
    # Geo-location
    geo_location = Column(String)  # Country/state
    timezone = Column(String)
    
    # Timestamps
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), index=True, nullable=False)
    
    __table_args__ = (
        Index('idx_audit_user_action', 'user_id', 'action_type'),
        Index('idx_audit_patient_access', 'patient_id_accessed', 'timestamp'),
        Index('idx_audit_phi_access', 'phi_accessed', 'timestamp'),
        Index('idx_audit_timestamp', 'timestamp'),
    )


class SecurityEvent(Base):
    """Track security-related events and potential threats"""
    __tablename__ = "security_events"

    id = Column(Integer, primary_key=True, index=True)
    
    # Event classification
    event_type = Column(String, nullable=False)  # "login_failure", "suspicious_access", "rate_limit", "unauthorized_attempt"
    severity = Column(String, nullable=False)  # "low", "medium", "high", "critical"
    
    # Event details
    description = Column(Text, nullable=False)
    event_data = Column(JSON)  # Additional event details
    
    # Actor
    user_id = Column(String, index=True)
    ip_address = Column(String, index=True)
    user_agent = Column(String)
    
    # Detection
    detection_method = Column(String)  # "manual", "automated_rule", "ml_model", "rate_limiter"
    confidence_score = Column(Float)  # 0-1 confidence in threat detection
    
    # Response
    action_taken = Column(String)  # "blocked", "logged", "alert_sent", "account_locked"
    response_status = Column(String)  # "pending", "handled", "escalated", "resolved"
    handled_by = Column(String)  # Admin user ID who handled it
    handled_at = Column(DateTime(timezone=True))
    resolution_notes = Column(Text)
    
    # Timestamps
    occurred_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    
    __table_args__ = (
        Index('idx_security_severity', 'severity'),
        Index('idx_security_type', 'event_type'),
        Index('idx_security_status', 'response_status'),
    )


class LegalHold(Base):
    """Track legal holds on patient data (prevent deletion)"""
    __tablename__ = "legal_holds"

    id = Column(Integer, primary_key=True, index=True)
    
    # Scope
    patient_id = Column(String, index=True)  # Specific patient (or null for org-wide)
    hold_scope = Column(String, nullable=False)  # "patient_specific", "organization_wide", "data_type_specific"
    
    # Hold details
    hold_reason = Column(Text, nullable=False)  # "litigation", "investigation", "regulatory_request"
    case_number = Column(String)  # Legal case reference
    requesting_authority = Column(String)  # Who requested the hold
    
    # Data coverage
    data_types_affected = Column(JSON)  # ["video", "audio", "metrics", "all"]
    date_range_start = Column(DateTime(timezone=True))  # Hold data from this date
    date_range_end = Column(DateTime(timezone=True))  # Hold data until this date
    
    # Status
    is_active = Column(Boolean, default=True, index=True)
    lifted_at = Column(DateTime(timezone=True))
    lifted_by = Column(String)
    lift_reason = Column(Text)
    
    # Audit
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    created_by = Column(String, nullable=False)
    approved_by = Column(String)  # Legal/compliance approval
    
    __table_args__ = (
        Index('idx_legal_hold_patient', 'patient_id', 'is_active'),
        Index('idx_legal_hold_active', 'is_active'),
    )
