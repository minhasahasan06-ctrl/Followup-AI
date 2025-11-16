"""
Alert Engine Database Models
Alert rules, delivery, multi-channel notifications
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, JSON, Text, ForeignKey, Index
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.database import Base


class AlertRule(Base):
    """Define alert triggering rules"""
    __tablename__ = "alert_rules"

    id = Column(Integer, primary_key=True, index=True)
    
    # Rule identification
    rule_name = Column(String, nullable=False)
    rule_type = Column(String, nullable=False)  # "risk_transition", "metric_spike", "baseline_deviation", "duration_threshold"
    description = Column(Text)
    
    # Trigger conditions
    trigger_condition = Column(String, nullable=False)  # "green_to_yellow", "yellow_to_red", "yellow_persists_48h", etc.
    metric_name = Column(String)  # Specific metric to monitor (if applicable)
    threshold_value = Column(Float)  # Threshold for metric-based alerts
    comparison_operator = Column(String)  # ">", "<", ">=", "<=", "==", "!="
    
    # Duration-based triggers
    persistence_duration_hours = Column(Integer)  # How long condition must persist
    
    # Severity mapping
    severity_level = Column(String, nullable=False)  # "low", "medium", "high", "critical"
    urgency = Column(String)  # "routine", "urgent", "emergency"
    
    # Target recipients
    notify_patient = Column(Boolean, default=False)
    notify_clinician = Column(Boolean, default=True)
    notify_admin = Column(Boolean, default=False)
    specific_clinician_ids = Column(JSON)  # List of clinician IDs to notify
    
    # Delivery channels
    send_dashboard = Column(Boolean, default=True)
    send_email = Column(Boolean, default=True)
    send_sms = Column(Boolean, default=False)
    send_push = Column(Boolean, default=True)
    
    # Active status
    is_active = Column(Boolean, default=True)
    is_global = Column(Boolean, default=True)  # Apply to all patients
    patient_specific_ids = Column(JSON)  # If not global, apply to these patients
    
    # Rate limiting
    max_alerts_per_day = Column(Integer, default=10)
    cooldown_minutes = Column(Integer, default=60)  # Minimum time between alerts
    
    # Audit
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    created_by = Column(String)
    
    # Relationships
    alerts = relationship("Alert", back_populates="rule")
    
    __table_args__ = (
        Index('idx_alert_rule_active', 'is_active'),
        Index('idx_alert_rule_type', 'rule_type'),
    )


class Alert(Base):
    """Track individual alert instances and delivery"""
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, index=True)
    
    # Alert details
    alert_rule_id = Column(Integer, ForeignKey("alert_rules.id"), nullable=False, index=True)
    patient_id = Column(String, nullable=False, index=True)
    risk_event_id = Column(Integer, ForeignKey("risk_events.id"))
    
    # Alert content
    alert_title = Column(String, nullable=False)
    alert_message = Column(Text, nullable=False)
    severity = Column(String, nullable=False)  # "low", "medium", "high", "critical"
    urgency = Column(String)  # "routine", "urgent", "emergency"
    
    # Trigger context
    trigger_metric = Column(String)  # What triggered this alert
    trigger_value = Column(Float)
    threshold_crossed = Column(Float)
    risk_level_current = Column(String)  # "green", "yellow", "red"
    risk_level_previous = Column(String)
    
    # Delivery status
    delivery_status = Column(String, default="pending")  # pending, sent, delivered, failed, acknowledged
    
    # Dashboard delivery
    dashboard_sent = Column(Boolean, default=False)
    dashboard_sent_at = Column(DateTime(timezone=True))
    dashboard_viewed = Column(Boolean, default=False)
    dashboard_viewed_at = Column(DateTime(timezone=True))
    
    # Email delivery
    email_sent = Column(Boolean, default=False)
    email_sent_at = Column(DateTime(timezone=True))
    email_delivered = Column(Boolean, default=False)
    email_opened = Column(Boolean, default=False)
    email_message_id = Column(String)  # SES message ID
    email_recipients = Column(JSON)  # List of email addresses
    
    # SMS delivery
    sms_sent = Column(Boolean, default=False)
    sms_sent_at = Column(DateTime(timezone=True))
    sms_delivered = Column(Boolean, default=False)
    sms_message_sid = Column(String)  # Twilio message SID
    sms_recipients = Column(JSON)  # List of phone numbers
    
    # Push notification delivery
    push_sent = Column(Boolean, default=False)
    push_sent_at = Column(DateTime(timezone=True))
    push_delivered = Column(Boolean, default=False)
    push_receipt_ids = Column(JSON)  # Expo push receipt IDs
    push_recipients = Column(JSON)  # List of push tokens
    
    # Clinician response
    acknowledged = Column(Boolean, default=False)
    acknowledged_at = Column(DateTime(timezone=True))
    acknowledged_by = Column(String)  # Clinician user ID
    acknowledgement_notes = Column(Text)
    
    # Actions taken
    action_taken = Column(String)  # "contacted_patient", "scheduled_appointment", "no_action", etc.
    action_taken_at = Column(DateTime(timezone=True))
    action_notes = Column(Text)
    
    # Escalation
    escalated = Column(Boolean, default=False)
    escalated_at = Column(DateTime(timezone=True))
    escalated_to = Column(String)  # User ID of person escalated to
    escalation_reason = Column(Text)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    expires_at = Column(DateTime(timezone=True))  # Auto-dismiss time
    
    # Relationships
    rule = relationship("AlertRule", back_populates="alerts")
    
    __table_args__ = (
        Index('idx_alert_patient_date', 'patient_id', 'created_at'),
        Index('idx_alert_status', 'delivery_status'),
        Index('idx_alert_acknowledged', 'acknowledged'),
        Index('idx_alert_severity', 'severity'),
    )


class NotificationPreference(Base):
    """Store user notification preferences"""
    __tablename__ = "notification_preferences"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, nullable=False, unique=True, index=True)
    user_type = Column(String, nullable=False)  # "patient", "clinician", "admin"
    
    # Channel preferences
    email_enabled = Column(Boolean, default=True)
    email_address = Column(String)
    email_verified = Column(Boolean, default=False)
    
    sms_enabled = Column(Boolean, default=False)
    sms_phone_number = Column(String)
    sms_verified = Column(Boolean, default=False)
    
    push_enabled = Column(Boolean, default=True)
    push_tokens = Column(JSON)  # List of Expo push tokens
    
    # Alert severity filters
    receive_low_severity = Column(Boolean, default=True)
    receive_medium_severity = Column(Boolean, default=True)
    receive_high_severity = Column(Boolean, default=True)
    receive_critical_severity = Column(Boolean, default=True)
    
    # Quiet hours
    quiet_hours_enabled = Column(Boolean, default=False)
    quiet_hours_start = Column(String)  # "22:00"
    quiet_hours_end = Column(String)  # "08:00"
    quiet_hours_timezone = Column(String)  # "America/New_York"
    
    # Digest preferences
    daily_digest_enabled = Column(Boolean, default=False)
    daily_digest_time = Column(String)  # "09:00"
    weekly_digest_enabled = Column(Boolean, default=False)
    weekly_digest_day = Column(String)  # "monday"
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
