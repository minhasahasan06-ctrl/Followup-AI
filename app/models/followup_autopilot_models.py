"""
Followup Autopilot Database Models
Patient signals, daily features, autopilot state, tasks, triggers, notifications

HIPAA Compliance:
- All patient data requires consent verification before access
- All operations are audit logged
- Data retention policies apply

Wellness Positioning:
- All features and naming use wellness/monitoring language
- NOT medical diagnosis or treatment
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, JSON, Text, ForeignKey, Index, Date, Enum as SQLEnum
from sqlalchemy.dialects.postgresql import UUID as PGUUID, JSONB
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.database import Base
import uuid
import enum


class SignalCategory(str, enum.Enum):
    DEVICE = "device"
    SYMPTOM = "symptom"
    VIDEO = "video"
    AUDIO = "audio"
    PAIN = "pain"
    MENTAL = "mental"
    ENVIRONMENT = "environment"
    MEDS = "meds"
    EXPOSURE = "exposure"


class RiskState(str, enum.Enum):
    STABLE = "Stable"
    AT_RISK = "AtRisk"
    WORSENING = "Worsening"
    CRITICAL = "Critical"


class TaskPriority(str, enum.Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TaskStatus(str, enum.Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class TriggerSeverity(str, enum.Enum):
    INFO = "info"
    WARNING = "warning"
    ALERT = "alert"


class NotificationChannel(str, enum.Enum):
    IN_APP = "in_app"
    PUSH = "push"
    EMAIL = "email"
    SMS = "sms"


class NotificationStatus(str, enum.Enum):
    PENDING = "pending"
    SENT = "sent"
    FAILED = "failed"


class AutopilotPatientSignal(Base):
    """Raw signals from all modules (device, symptoms, video, audio, pain, mental, meds, environment, exposures)"""
    __tablename__ = "autopilot_patient_signals"

    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    patient_id = Column(String, nullable=False, index=True)
    
    category = Column(String, nullable=False)
    source = Column(String, nullable=False)
    raw_payload = Column(JSONB)
    ml_score = Column(Float, nullable=True)
    signal_time = Column(DateTime(timezone=True), nullable=False)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index('idx_autopilot_signal_patient_time', 'patient_id', 'signal_time'),
        Index('idx_autopilot_signal_category', 'category'),
        Index('idx_autopilot_signal_patient_category', 'patient_id', 'category'),
    )


class AutopilotDailyFeatures(Base):
    """Daily aggregated features (input to ML models)"""
    __tablename__ = "autopilot_daily_features"

    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    patient_id = Column(String, nullable=False, index=True)
    date = Column(Date, nullable=False, index=True)
    
    avg_pain = Column(Float, default=0.0)
    avg_fatigue = Column(Float, default=0.0)
    avg_mood = Column(Float, default=0.0)
    checkins_count = Column(Integer, default=0)
    steps = Column(Integer, default=0)
    resting_hr = Column(Float, default=0.0)
    sleep_hours = Column(Float, default=0.0)
    weight = Column(Float, nullable=True)
    
    env_risk_score = Column(Float, default=0.0)
    pollen_index = Column(Float, default=0.0)
    aqi = Column(Float, default=0.0)
    temp_c = Column(Float, nullable=True)
    
    med_adherence_7d = Column(Float, default=1.0)
    mh_score = Column(Float, default=0.0)
    video_resp_risk = Column(Float, default=0.0)
    audio_emotion_score = Column(Float, default=0.0)
    pain_severity_score = Column(Float, default=0.0)
    engagement_rate_14d = Column(Float, default=0.0)
    
    infectious_exposure_score = Column(Float, default=0.0)
    immunization_status = Column(Float, default=1.0)
    occupational_risk_score = Column(Float, default=0.0)
    genetic_risk_flags = Column(JSONB, default=list)
    
    had_worsening_event_next7d = Column(Boolean, nullable=True)
    had_mh_crisis_next7d = Column(Boolean, nullable=True)
    had_non_adherence_issue_next7d = Column(Boolean, nullable=True)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index('idx_autopilot_features_patient_date', 'patient_id', 'date', unique=True),
    )


class AutopilotPatientState(Base):
    """Patient state (Autopilot brain snapshot)"""
    __tablename__ = "autopilot_patient_states"

    patient_id = Column(String, primary_key=True)
    
    risk_score = Column(Float, default=0.0)
    risk_state = Column(String, default="Stable")
    risk_components = Column(JSONB, default=dict)
    
    last_updated = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    last_checkin_at = Column(DateTime(timezone=True), nullable=True)
    next_followup_at = Column(DateTime(timezone=True), nullable=True)
    preferred_contact_hour = Column(Integer, nullable=True)
    
    notification_preferences = Column(JSONB, default=lambda: {
        "in_app_enabled": True,
        "push_enabled": True,
        "email_enabled": True,
        "quiet_hours_start": 22,
        "quiet_hours_end": 7,
        "urgency_threshold": "medium"
    })
    
    model_version = Column(String, default="1.0.0")
    inference_confidence = Column(Float, default=0.0)

    __table_args__ = (
        Index('idx_autopilot_state_risk', 'risk_state'),
        Index('idx_autopilot_state_next_followup', 'next_followup_at'),
    )


class AutopilotFollowupTask(Base):
    """Follow-up tasks for patients"""
    __tablename__ = "autopilot_followup_tasks"

    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    patient_id = Column(String, nullable=False, index=True)
    
    task_type = Column(String, nullable=False)
    priority = Column(String, default="medium")
    status = Column(String, default="pending")
    
    due_at = Column(DateTime(timezone=True), nullable=False)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    
    created_by = Column(String, default="autopilot")
    task_metadata = Column(JSONB, default=dict)
    trigger_name = Column(String, nullable=True)
    reason = Column(Text, nullable=True)
    
    ui_tab_target = Column(String, nullable=True)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index('idx_autopilot_task_patient_status', 'patient_id', 'status'),
        Index('idx_autopilot_task_due', 'due_at'),
        Index('idx_autopilot_task_patient_due', 'patient_id', 'due_at'),
        Index('idx_autopilot_task_priority', 'priority'),
    )


class AutopilotTriggerEvent(Base):
    """Trigger events log"""
    __tablename__ = "autopilot_trigger_events"

    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    patient_id = Column(String, nullable=False, index=True)
    
    name = Column(String, nullable=False)
    severity = Column(String, default="info")
    context = Column(JSONB, default=dict)
    
    task_ids_created = Column(JSONB, default=list)
    alert_ids_created = Column(JSONB, default=list)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index('idx_autopilot_trigger_patient_name', 'patient_id', 'name'),
        Index('idx_autopilot_trigger_created', 'created_at'),
        Index('idx_autopilot_trigger_severity', 'severity'),
    )


class AutopilotNotification(Base):
    """Notification queue (for push/email/SMS)"""
    __tablename__ = "autopilot_notifications"

    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    patient_id = Column(String, nullable=False, index=True)
    
    channel = Column(String, default="in_app")
    title = Column(String, nullable=False)
    body = Column(Text, nullable=False)
    priority = Column(String, default="medium")
    
    related_task_id = Column(PGUUID(as_uuid=True), ForeignKey("autopilot_followup_tasks.id"), nullable=True)
    
    status = Column(String, default="pending")
    error_message = Column(Text, nullable=True)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    sent_at = Column(DateTime(timezone=True), nullable=True)

    related_task = relationship("AutopilotFollowupTask", foreign_keys=[related_task_id])

    __table_args__ = (
        Index('idx_autopilot_notification_patient_status', 'patient_id', 'status'),
        Index('idx_autopilot_notification_channel', 'channel'),
        Index('idx_autopilot_notification_priority', 'priority'),
    )


class AutopilotAuditLog(Base):
    """Audit log for HIPAA compliance"""
    __tablename__ = "autopilot_audit_logs"

    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    patient_id = Column(String, nullable=True, index=True)
    
    action = Column(String, nullable=False)
    entity_type = Column(String, nullable=False)
    entity_id = Column(String, nullable=True)
    
    user_id = Column(String, nullable=True)
    ip_address = Column(String, nullable=True)
    user_agent = Column(String, nullable=True)
    
    old_values = Column(JSONB, nullable=True)
    new_values = Column(JSONB, nullable=True)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index('idx_autopilot_audit_patient', 'patient_id'),
        Index('idx_autopilot_audit_action', 'action'),
        Index('idx_autopilot_audit_created', 'created_at'),
    )


class ApprovalStatus(str, enum.Enum):
    PENDING = "pending"
    APPROVED = "approved"
    MODIFIED = "modified"
    REJECTED = "rejected"
    EXPIRED = "expired"


class ApprovalActionType(str, enum.Enum):
    SCHEDULE_FOLLOWUP = "schedule_followup"
    SEND_REMINDER = "send_reminder"
    ESCALATE_CARE = "escalate_care"
    ADJUST_MEDICATION = "adjust_medication"
    REQUEST_CHECKIN = "request_checkin"
    SCHEDULE_CONSULTATION = "schedule_consultation"
    NOTIFY_PATIENT = "notify_patient"


class AutopilotPendingApproval(Base):
    """
    Human-in-the-Loop pending approvals for doctor review.
    Created when Autopilot detects high-risk events requiring doctor oversight.
    """
    __tablename__ = "autopilot_pending_approvals"

    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    patient_id = Column(String, nullable=False, index=True)
    doctor_id = Column(String, nullable=False, index=True)
    
    trigger_event_id = Column(PGUUID(as_uuid=True), ForeignKey("autopilot_trigger_events.id"), nullable=True)
    
    action_type = Column(String, nullable=False)
    status = Column(String, default="pending", index=True)
    priority = Column(String, default="medium")
    
    title = Column(String, nullable=False)
    ai_recommendation = Column(Text, nullable=False)
    ai_reasoning = Column(Text, nullable=True)
    confidence_score = Column(Float, default=0.0)
    
    patient_context = Column(JSONB, default=dict)
    risk_score = Column(Float, nullable=True)
    risk_state = Column(String, nullable=True)
    
    doctor_notes = Column(Text, nullable=True)
    modified_action = Column(JSONB, nullable=True)
    rejection_reason = Column(Text, nullable=True)
    
    expires_at = Column(DateTime(timezone=True), nullable=True)
    reviewed_at = Column(DateTime(timezone=True), nullable=True)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    trigger_event = relationship("AutopilotTriggerEvent", foreign_keys=[trigger_event_id])

    __table_args__ = (
        Index('idx_pending_approval_doctor_status', 'doctor_id', 'status'),
        Index('idx_pending_approval_patient', 'patient_id'),
        Index('idx_pending_approval_priority', 'priority'),
        Index('idx_pending_approval_created', 'created_at'),
        Index('idx_pending_approval_expires', 'expires_at'),
    )


class AutopilotApprovalHistory(Base):
    """
    Historical log of all approval decisions for audit and analytics.
    """
    __tablename__ = "autopilot_approval_history"

    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    approval_id = Column(PGUUID(as_uuid=True), ForeignKey("autopilot_pending_approvals.id"), nullable=False)
    doctor_id = Column(String, nullable=False, index=True)
    patient_id = Column(String, nullable=False, index=True)
    
    action_taken = Column(String, nullable=False)
    original_recommendation = Column(Text, nullable=True)
    final_action = Column(JSONB, nullable=True)
    doctor_notes = Column(Text, nullable=True)
    
    time_to_decision_seconds = Column(Integer, nullable=True)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index('idx_approval_history_doctor', 'doctor_id'),
        Index('idx_approval_history_patient', 'patient_id'),
        Index('idx_approval_history_created', 'created_at'),
    )
