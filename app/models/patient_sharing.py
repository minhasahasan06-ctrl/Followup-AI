"""
Secure Patient Sharing Link System

HIPAA-compliant secure sharing between doctors and patients with:
- Time-limited secure tokens
- Consent management
- Continuous data sharing
- Audit logging
"""

from sqlalchemy import (
    Column, String, Text, Integer, Boolean, DateTime, Float,
    ForeignKey, Enum, JSON, Index, func
)
from sqlalchemy.orm import relationship
from app.database import Base
from datetime import datetime, timedelta
import enum
import uuid
import secrets


class SharingAccessLevel(str, enum.Enum):
    """Access levels for data sharing"""
    VIEW_ONLY = "view_only"
    FULL_ACCESS = "full_access"
    EMERGENCY_ONLY = "emergency_only"


class SharingStatus(str, enum.Enum):
    """Status of sharing link"""
    PENDING = "pending"
    ACTIVE = "active"
    PAUSED = "paused"
    REVOKED = "revoked"
    EXPIRED = "expired"


class PatientSharingLink(Base):
    """
    Secure sharing link between doctor and patient.
    Enables continuous health data sharing with consent management.
    """
    __tablename__ = "patient_sharing_links"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    doctor_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    patient_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    
    secure_token = Column(String(128), unique=True, nullable=False, index=True)
    
    access_level = Column(String(30), default=SharingAccessLevel.VIEW_ONLY.value)
    status = Column(String(20), default=SharingStatus.PENDING.value, index=True)
    
    continuous_sharing_enabled = Column(Boolean, default=False)
    
    share_vitals = Column(Boolean, default=True)
    share_symptoms = Column(Boolean, default=True)
    share_medications = Column(Boolean, default=True)
    share_activities = Column(Boolean, default=True)
    share_mental_health = Column(Boolean, default=False)
    share_video_exams = Column(Boolean, default=False)
    share_audio_exams = Column(Boolean, default=False)
    
    followup_monitoring_enabled = Column(Boolean, default=False)
    followup_check_frequency_hours = Column(Integer, default=24)
    last_followup_check_at = Column(DateTime, nullable=True)
    
    alert_on_deterioration = Column(Boolean, default=True)
    alert_on_missed_medication = Column(Boolean, default=True)
    alert_on_symptoms = Column(Boolean, default=True)
    
    consent_given_at = Column(DateTime, nullable=True)
    consent_ip_address = Column(String(45), nullable=True)
    consent_user_agent = Column(String(500), nullable=True)
    
    expires_at = Column(DateTime, nullable=True, index=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    revoked_at = Column(DateTime, nullable=True)
    revoked_by = Column(String, nullable=True)
    revoked_reason = Column(Text, nullable=True)

    __table_args__ = (
        Index("ix_sharing_doctor_patient", "doctor_id", "patient_id"),
        Index("ix_sharing_status_active", "status", "continuous_sharing_enabled"),
        Index("ix_sharing_followup", "followup_monitoring_enabled", "last_followup_check_at"),
    )

    @staticmethod
    def generate_secure_token() -> str:
        """Generate a cryptographically secure token"""
        return secrets.token_urlsafe(64)

    @classmethod
    def create_sharing_link(
        cls,
        doctor_id: str,
        patient_id: str,
        access_level: str = SharingAccessLevel.VIEW_ONLY.value,
        expires_in_days: int = 30,
        continuous_sharing: bool = False,
        followup_monitoring: bool = False
    ) -> "PatientSharingLink":
        """Create a new sharing link"""
        return cls(
            id=str(uuid.uuid4()),
            doctor_id=doctor_id,
            patient_id=patient_id,
            secure_token=cls.generate_secure_token(),
            access_level=access_level,
            status=SharingStatus.PENDING.value,
            continuous_sharing_enabled=continuous_sharing,
            followup_monitoring_enabled=followup_monitoring,
            expires_at=datetime.utcnow() + timedelta(days=expires_in_days) if expires_in_days else None
        )

    def activate(self, ip_address: str = None, user_agent: str = None):
        """Activate the sharing link with consent"""
        self.status = SharingStatus.ACTIVE.value
        self.consent_given_at = datetime.utcnow()
        self.consent_ip_address = ip_address
        self.consent_user_agent = user_agent

    def revoke(self, revoked_by: str, reason: str = None):
        """Revoke the sharing link"""
        self.status = SharingStatus.REVOKED.value
        self.revoked_at = datetime.utcnow()
        self.revoked_by = revoked_by
        self.revoked_reason = reason

    def pause(self):
        """Pause the sharing link"""
        self.status = SharingStatus.PAUSED.value

    def resume(self):
        """Resume a paused sharing link"""
        if self.status == SharingStatus.PAUSED.value:
            self.status = SharingStatus.ACTIVE.value

    def is_active(self) -> bool:
        """Check if sharing link is active and not expired"""
        if self.status != SharingStatus.ACTIVE.value:
            return False
        if self.expires_at and datetime.utcnow() > self.expires_at:
            return False
        return True

    def needs_followup_check(self) -> bool:
        """Check if patient needs followup monitoring"""
        if not self.followup_monitoring_enabled or not self.is_active():
            return False
        
        if not self.last_followup_check_at:
            return True
        
        check_interval = timedelta(hours=self.followup_check_frequency_hours or 24)
        return datetime.utcnow() - self.last_followup_check_at >= check_interval

    def mark_followup_checked(self):
        """Mark that followup check was performed"""
        self.last_followup_check_at = datetime.utcnow()


class SharingAccessLog(Base):
    """
    HIPAA-compliant audit log for all data access via sharing links.
    """
    __tablename__ = "sharing_access_logs"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    sharing_link_id = Column(String, ForeignKey("patient_sharing_links.id"), nullable=False, index=True)
    doctor_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    patient_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    
    action = Column(String(50), nullable=False)
    resource_type = Column(String(50), nullable=True)
    resource_id = Column(String, nullable=True)
    
    details = Column(JSON, nullable=True)
    
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(String(500), nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    __table_args__ = (
        Index("ix_access_log_patient_date", "patient_id", "created_at"),
        Index("ix_access_log_doctor_date", "doctor_id", "created_at"),
    )


class PatientFollowupAlert(Base):
    """
    Alerts generated from patient followup monitoring.
    """
    __tablename__ = "patient_followup_alerts"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    sharing_link_id = Column(String, ForeignKey("patient_sharing_links.id"), nullable=False, index=True)
    doctor_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    patient_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    
    alert_type = Column(String(50), nullable=False)
    severity = Column(String(20), default="medium")
    
    title = Column(String(255), nullable=False)
    message = Column(Text, nullable=True)
    
    detected_data = Column(JSON, nullable=True)
    
    is_read = Column(Boolean, default=False)
    is_acknowledged = Column(Boolean, default=False)
    acknowledged_at = Column(DateTime, nullable=True)
    acknowledged_by = Column(String, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    __table_args__ = (
        Index("ix_alert_doctor_unread", "doctor_id", "is_read"),
        Index("ix_alert_patient_date", "patient_id", "created_at"),
    )
