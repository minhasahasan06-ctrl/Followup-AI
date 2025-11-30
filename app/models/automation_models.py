"""
Lysa Automation Engine Models

Production-grade automation system for Assistant Lysa.
Supports job scheduling, execution tracking, retry logic, and audit logging.
"""

from sqlalchemy import (
    Column, String, Text, Integer, Boolean, DateTime, Float,
    ForeignKey, Enum, JSON, Index, func
)
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
from app.database import Base
from datetime import datetime
import enum
import uuid


class JobStatus(str, enum.Enum):
    """Automation job statuses"""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


class JobPriority(str, enum.Enum):
    """Job priority levels"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class JobType(str, enum.Enum):
    """Types of automation jobs"""
    EMAIL_SYNC = "email_sync"
    EMAIL_CLASSIFY = "email_classify"
    EMAIL_AUTO_REPLY = "email_auto_reply"
    EMAIL_FORWARD_URGENT = "email_forward_urgent"
    WHATSAPP_SYNC = "whatsapp_sync"
    WHATSAPP_AUTO_REPLY = "whatsapp_auto_reply"
    WHATSAPP_SEND_TEMPLATE = "whatsapp_send_template"
    APPOINTMENT_REQUEST = "appointment_request"
    APPOINTMENT_BOOK = "appointment_book"
    APPOINTMENT_CANCEL = "appointment_cancel"
    APPOINTMENT_RESCHEDULE = "appointment_reschedule"
    REMINDER_MEDICATION = "reminder_medication"
    REMINDER_APPOINTMENT = "reminder_appointment"
    REMINDER_FOLLOWUP = "reminder_followup"
    REMINDER_NOSHOW = "reminder_noshow"
    CALENDAR_SYNC = "calendar_sync"
    PATIENT_LOOKUP = "patient_lookup"
    DIAGNOSIS_SUMMARY = "diagnosis_summary"
    SOAP_NOTE = "soap_note"
    ICD10_SUGGEST = "icd10_suggest"
    DIFFERENTIAL_DIAGNOSIS = "differential_diagnosis"
    PRESCRIPTION_GENERATE = "prescription_generate"
    DAILY_REPORT = "daily_report"
    WEEKLY_DIGEST = "weekly_digest"
    ALERT_PROCESS = "alert_process"
    PATIENT_MONITOR = "patient_monitor"


class ScheduleFrequency(str, enum.Enum):
    """Schedule frequency options"""
    ONCE = "once"
    EVERY_MINUTE = "every_minute"
    EVERY_5_MINUTES = "every_5_minutes"
    EVERY_10_MINUTES = "every_10_minutes"
    EVERY_15_MINUTES = "every_15_minutes"
    EVERY_30_MINUTES = "every_30_minutes"
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    CUSTOM_CRON = "custom_cron"


class AutomationJob(Base):
    """
    Individual automation job instance.
    Tracks execution, results, and retry attempts.
    """
    __tablename__ = "automation_jobs"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    doctor_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    patient_id = Column(String, ForeignKey("users.id"), nullable=True, index=True)
    
    job_type = Column(String(50), nullable=False, index=True)
    priority = Column(String(20), default="normal", index=True)
    status = Column(String(20), default="pending", index=True)
    
    input_data = Column(JSON, nullable=True)
    output_data = Column(JSON, nullable=True)
    error_message = Column(Text, nullable=True)
    error_details = Column(JSON, nullable=True)
    
    attempts = Column(Integer, default=0)
    max_attempts = Column(Integer, default=3)
    
    scheduled_for = Column(DateTime, nullable=True, index=True)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    
    parent_job_id = Column(String, ForeignKey("automation_jobs.id"), nullable=True)
    schedule_id = Column(String, ForeignKey("automation_schedules.id"), nullable=True)
    
    idempotency_key = Column(String(255), nullable=True, unique=True)
    
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    __table_args__ = (
        Index("ix_automation_jobs_doctor_status", "doctor_id", "status"),
        Index("ix_automation_jobs_doctor_type", "doctor_id", "job_type"),
        Index("ix_automation_jobs_scheduled", "scheduled_for", "status"),
        Index("ix_automation_jobs_pending", "status", "priority", "scheduled_for"),
    )


class AutomationSchedule(Base):
    """
    Recurring automation schedule.
    Defines what jobs run automatically and when.
    """
    __tablename__ = "automation_schedules"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    doctor_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    
    job_type = Column(String(50), nullable=False, index=True)
    job_config = Column(JSON, nullable=True)
    
    frequency = Column(String(30), nullable=False)
    cron_expression = Column(String(100), nullable=True)
    timezone = Column(String(50), default="UTC")
    
    is_enabled = Column(Boolean, default=True, index=True)
    priority = Column(String(20), default="normal")
    
    last_run_at = Column(DateTime, nullable=True)
    next_run_at = Column(DateTime, nullable=True, index=True)
    last_run_status = Column(String(20), nullable=True)
    last_run_job_id = Column(String, nullable=True)
    
    run_count = Column(Integer, default=0)
    success_count = Column(Integer, default=0)
    failure_count = Column(Integer, default=0)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    __table_args__ = (
        Index("ix_automation_schedules_doctor_enabled", "doctor_id", "is_enabled"),
        Index("ix_automation_schedules_next_run", "next_run_at", "is_enabled"),
    )


class AutomationLog(Base):
    """
    Detailed execution log for automation jobs.
    HIPAA-compliant audit trail of all automation actions.
    """
    __tablename__ = "automation_logs"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    job_id = Column(String, ForeignKey("automation_jobs.id"), nullable=False, index=True)
    doctor_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    patient_id = Column(String, ForeignKey("users.id"), nullable=True, index=True)
    
    log_level = Column(String(20), default="info")
    message = Column(Text, nullable=False)
    details = Column(JSON, nullable=True)
    
    action_type = Column(String(100), nullable=True)
    resource_type = Column(String(50), nullable=True)
    resource_id = Column(String, nullable=True)
    
    duration_ms = Column(Integer, nullable=True)
    
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(String(500), nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    __table_args__ = (
        Index("ix_automation_logs_job", "job_id"),
        Index("ix_automation_logs_doctor_date", "doctor_id", "created_at"),
        Index("ix_automation_logs_patient_date", "patient_id", "created_at"),
        Index("ix_automation_logs_action", "action_type", "created_at"),
    )


class AutomationMetric(Base):
    """
    Aggregated metrics for automation performance tracking.
    Used for dashboards and monitoring.
    """
    __tablename__ = "automation_metrics"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    doctor_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    
    metric_date = Column(DateTime, nullable=False, index=True)
    metric_hour = Column(Integer, nullable=True)
    
    job_type = Column(String(50), nullable=False, index=True)
    
    jobs_created = Column(Integer, default=0)
    jobs_completed = Column(Integer, default=0)
    jobs_failed = Column(Integer, default=0)
    jobs_retried = Column(Integer, default=0)
    
    total_duration_ms = Column(Integer, default=0)
    avg_duration_ms = Column(Float, nullable=True)
    max_duration_ms = Column(Integer, nullable=True)
    
    emails_processed = Column(Integer, default=0)
    emails_auto_replied = Column(Integer, default=0)
    whatsapp_messages_sent = Column(Integer, default=0)
    appointments_booked = Column(Integer, default=0)
    reminders_sent = Column(Integer, default=0)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    __table_args__ = (
        Index("ix_automation_metrics_doctor_date", "doctor_id", "metric_date"),
        Index("ix_automation_metrics_type_date", "job_type", "metric_date"),
    )


class EmailAutomationConfig(Base):
    """
    Doctor-specific email automation configuration.
    Controls auto-reply templates, classification rules, and forwarding.
    """
    __tablename__ = "email_automation_configs"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    doctor_id = Column(String, ForeignKey("users.id"), nullable=False, unique=True)
    
    is_enabled = Column(Boolean, default=True)
    auto_classify = Column(Boolean, default=True)
    auto_reply_enabled = Column(Boolean, default=False)
    forward_urgent_enabled = Column(Boolean, default=True)
    
    forward_urgent_to = Column(String(255), nullable=True)
    
    auto_reply_template = Column(Text, nullable=True)
    auto_reply_conditions = Column(JSON, nullable=True)
    
    classification_rules = Column(JSON, nullable=True)
    priority_keywords = Column(JSON, nullable=True)
    
    sync_frequency_minutes = Column(Integer, default=5)
    last_sync_at = Column(DateTime, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class WhatsAppAutomationConfig(Base):
    """
    Doctor-specific WhatsApp Business automation configuration.
    Controls message templates, auto-replies, and patient engagement.
    """
    __tablename__ = "whatsapp_automation_configs"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    doctor_id = Column(String, ForeignKey("users.id"), nullable=False, unique=True)
    
    is_enabled = Column(Boolean, default=True)
    auto_reply_enabled = Column(Boolean, default=False)
    business_hours_only = Column(Boolean, default=True)
    
    greeting_template = Column(Text, nullable=True)
    out_of_hours_template = Column(Text, nullable=True)
    appointment_confirmation_template = Column(Text, nullable=True)
    reminder_template = Column(Text, nullable=True)
    
    business_hours_start = Column(String(5), default="09:00")
    business_hours_end = Column(String(5), default="17:00")
    business_days = Column(JSON, default=["monday", "tuesday", "wednesday", "thursday", "friday"])
    
    sync_frequency_minutes = Column(Integer, default=5)
    last_sync_at = Column(DateTime, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class AppointmentAutomationConfig(Base):
    """
    Doctor-specific appointment automation configuration.
    Controls auto-booking rules, availability, and notifications.
    """
    __tablename__ = "appointment_automation_configs"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    doctor_id = Column(String, ForeignKey("users.id"), nullable=False, unique=True)
    
    is_enabled = Column(Boolean, default=True)
    auto_book_enabled = Column(Boolean, default=False)
    auto_confirm_enabled = Column(Boolean, default=True)
    
    default_duration_minutes = Column(Integer, default=30)
    buffer_minutes = Column(Integer, default=15)
    
    available_hours_start = Column(String(5), default="09:00")
    available_hours_end = Column(String(5), default="17:00")
    available_days = Column(JSON, default=["monday", "tuesday", "wednesday", "thursday", "friday"])
    
    appointment_types = Column(JSON, nullable=True)
    
    confirmation_email_enabled = Column(Boolean, default=True)
    confirmation_whatsapp_enabled = Column(Boolean, default=False)
    
    reminder_enabled = Column(Boolean, default=True)
    reminder_hours_before = Column(JSON, default=[24, 2])
    
    calendar_sync_enabled = Column(Boolean, default=True)
    calendar_sync_frequency_minutes = Column(Integer, default=15)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class ReminderAutomationConfig(Base):
    """
    Doctor-specific reminder automation configuration.
    Controls medication, appointment, and follow-up reminders.
    """
    __tablename__ = "reminder_automation_configs"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    doctor_id = Column(String, ForeignKey("users.id"), nullable=False, unique=True)
    
    is_enabled = Column(Boolean, default=True)
    
    medication_reminders_enabled = Column(Boolean, default=True)
    medication_reminder_times = Column(JSON, default=["08:00", "12:00", "20:00"])
    
    appointment_reminders_enabled = Column(Boolean, default=True)
    appointment_reminder_hours = Column(JSON, default=[24, 2])
    
    followup_reminders_enabled = Column(Boolean, default=True)
    followup_reminder_days = Column(JSON, default=[1, 3, 7])
    
    noshow_followup_enabled = Column(Boolean, default=True)
    noshow_followup_hours = Column(Integer, default=4)
    
    email_enabled = Column(Boolean, default=True)
    whatsapp_enabled = Column(Boolean, default=False)
    sms_enabled = Column(Boolean, default=False)
    
    quiet_hours_start = Column(String(5), default="21:00")
    quiet_hours_end = Column(String(5), default="08:00")
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class ClinicalAutomationConfig(Base):
    """
    Doctor-specific clinical automation configuration.
    Controls SOAP notes, diagnosis support, and prescription automation.
    """
    __tablename__ = "clinical_automation_configs"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    doctor_id = Column(String, ForeignKey("users.id"), nullable=False, unique=True)
    
    is_enabled = Column(Boolean, default=True)
    
    auto_soap_notes = Column(Boolean, default=True)
    soap_template_id = Column(String, nullable=True)
    
    auto_icd10_suggest = Column(Boolean, default=True)
    auto_differential_diagnosis = Column(Boolean, default=True)
    
    prescription_assist_enabled = Column(Boolean, default=True)
    require_prescription_approval = Column(Boolean, default=True)
    
    use_patient_history = Column(Boolean, default=True)
    include_vitals_in_notes = Column(Boolean, default=True)
    include_medications_in_notes = Column(Boolean, default=True)
    
    preferred_language = Column(String(10), default="en")
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
