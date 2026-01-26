"""
Video Billing Models - Phase 12
HIPAA-compliant video consultation and billing tracking

Tables:
- DoctorVideoSettings: Per-doctor video configuration
- AppointmentVideo: Per-appointment video setup
- VideoUsageEvent: Raw webhook events from Daily.co
- VideoUsageSession: Paired join/leave intervals
- VideoUsageLedger: Per-appointment billing rollup
- DoctorSubscription: Plan and billing settings
- DoctorMonthlyInvoice: Aggregated monthly totals
"""

from sqlalchemy import (
    Column, String, Integer, Boolean, DateTime, Text, 
    ForeignKey, Index, Numeric, JSON
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from decimal import Decimal

from app.database import Base


class DoctorVideoSettings(Base):
    """Per-doctor video configuration including external provider settings"""
    __tablename__ = "doctor_video_settings"
    
    doctor_id = Column(String, ForeignKey("users.id"), primary_key=True)
    
    allow_external_video = Column(Boolean, default=False)
    zoom_join_url = Column(String, nullable=True)
    meet_join_url = Column(String, nullable=True)
    default_video_provider = Column(String, default="daily")
    
    enable_recording = Column(Boolean, default=False)
    enable_chat = Column(Boolean, default=True)
    max_participants = Column(Integer, default=2)
    
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class AppointmentVideo(Base):
    """Per-appointment video configuration"""
    __tablename__ = "appointment_video"
    
    appointment_id = Column(String, ForeignKey("appointments.id"), primary_key=True)
    
    video_provider = Column(String, nullable=False, default="daily")
    
    daily_room_name = Column(String, unique=True, nullable=True)
    daily_room_url = Column(String, nullable=True)
    
    external_join_url = Column(String, nullable=True)
    
    room_created_at = Column(DateTime(timezone=True), nullable=True)
    room_expires_at = Column(DateTime(timezone=True), nullable=True)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    __table_args__ = (
        Index('appointment_video_provider_idx', 'video_provider'),
        Index('appointment_video_room_name_idx', 'daily_room_name'),
    )


class VideoUsageEvent(Base):
    """Raw webhook events from Daily.co for audit trail"""
    __tablename__ = "video_usage_events"
    
    id = Column(String, primary_key=True, server_default=func.gen_random_uuid())
    
    appointment_id = Column(String, ForeignKey("appointments.id"), nullable=True)
    doctor_id = Column(String, ForeignKey("users.id"), nullable=False)
    
    provider = Column(String, nullable=False, default="daily")
    event_type = Column(String, nullable=False)
    event_id = Column(String, unique=True, nullable=True)
    
    participant_id = Column(String, nullable=False)
    participant_name = Column(String, nullable=True)
    participant_role = Column(String, nullable=True)
    
    event_ts = Column(DateTime(timezone=True), nullable=False)
    received_ts = Column(DateTime(timezone=True), server_default=func.now())
    
    payload = Column(JSON, nullable=True)
    
    __table_args__ = (
        Index('video_usage_events_appt_participant_idx', 'appointment_id', 'participant_id', 'event_ts'),
        Index('video_usage_events_doctor_idx', 'doctor_id', 'event_ts'),
        Index('video_usage_events_event_id_idx', 'event_id'),
    )


class VideoUsageSession(Base):
    """Paired join/leave intervals per participant"""
    __tablename__ = "video_usage_sessions"
    
    id = Column(String, primary_key=True, server_default=func.gen_random_uuid())
    
    appointment_id = Column(String, ForeignKey("appointments.id"), nullable=False)
    doctor_id = Column(String, ForeignKey("users.id"), nullable=False)
    
    participant_id = Column(String, nullable=False)
    participant_role = Column(String, nullable=True)
    
    joined_at = Column(DateTime(timezone=True), nullable=False)
    left_at = Column(DateTime(timezone=True), nullable=True)
    duration_seconds = Column(Integer, nullable=True)
    
    billing_month = Column(String, nullable=True)
    
    __table_args__ = (
        Index('video_usage_sessions_appt_participant_idx', 'appointment_id', 'participant_id', 'joined_at'),
        Index('video_usage_sessions_doctor_idx', 'doctor_id', 'joined_at'),
        Index('video_usage_sessions_billing_month_idx', 'doctor_id', 'billing_month'),
    )


class VideoUsageLedger(Base):
    """Per-appointment billing rollup"""
    __tablename__ = "video_usage_ledger"
    
    appointment_id = Column(String, ForeignKey("appointments.id"), primary_key=True)
    doctor_id = Column(String, ForeignKey("users.id"), nullable=False)
    
    billing_month = Column(String, nullable=False)
    
    participant_minutes = Column(Integer, default=0)
    
    cost_usd = Column(Numeric(12, 4), default=Decimal("0"))
    overage_billable_minutes = Column(Integer, default=0)
    billed_to_doctor_usd = Column(Numeric(12, 4), default=Decimal("0"))
    
    finalized = Column(Boolean, default=False)
    finalized_at = Column(DateTime(timezone=True), nullable=True)
    
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    __table_args__ = (
        Index('video_usage_ledger_doctor_billing_idx', 'doctor_id', 'billing_month'),
        Index('video_usage_ledger_billing_finalized_idx', 'billing_month', 'finalized'),
    )


class DoctorSubscription(Base):
    """Internal plan record for billing"""
    __tablename__ = "doctor_subscription"
    
    doctor_id = Column(String, ForeignKey("users.id"), primary_key=True)
    
    plan = Column(String, nullable=False, default="TRIAL")
    status = Column(String, nullable=False, default="active")
    
    included_participant_minutes = Column(Integer, default=300)
    overage_rate_usd_per_pm = Column(Numeric(12, 4), default=Decimal("0.008"))
    
    period_start = Column(DateTime(timezone=True), server_default=func.now())
    period_end = Column(DateTime(timezone=True), nullable=True)
    
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    __table_args__ = (
        Index('doctor_subscription_status_period_idx', 'status', 'period_end'),
    )


class DoctorMonthlyInvoice(Base):
    """Aggregated monthly billing totals"""
    __tablename__ = "doctor_monthly_invoice"
    
    id = Column(String, primary_key=True, server_default=func.gen_random_uuid())
    doctor_id = Column(String, ForeignKey("users.id"), nullable=False)
    billing_month = Column(String, nullable=False)
    
    total_participant_minutes = Column(Integer, default=0)
    included_participant_minutes = Column(Integer, default=0)
    overage_minutes = Column(Integer, default=0)
    
    amount_due_usd = Column(Numeric(12, 4), default=Decimal("0"))
    
    status = Column(String, default="pending")
    paid_at = Column(DateTime(timezone=True), nullable=True)
    
    generated_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    __table_args__ = (
        Index('doctor_monthly_invoice_doctor_billing_idx', 'doctor_id', 'billing_month'),
        Index('doctor_monthly_invoice_status_idx', 'status'),
        Index('doctor_monthly_invoice_unique_idx', 'doctor_id', 'billing_month', unique=True),
    )
