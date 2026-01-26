"""
Voice/Video Consent Service
===========================

HIPAA-compliant consent management for voice and video features
Implements explicit consent collection before any recording
"""

from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import uuid
import logging

from app.services.access_control import HIPAAAuditLogger

logger = logging.getLogger(__name__)


class ConsentType(str, Enum):
    """Types of consent"""
    VOICE_RECORDING = "voice_recording"
    VIDEO_RECORDING = "video_recording"
    VOICE_TRANSCRIPTION = "voice_transcription"
    AI_VOICE_INTERACTION = "ai_voice_interaction"
    DATA_STORAGE = "data_storage"
    DATA_SHARING = "data_sharing"


class ConsentStatus(str, Enum):
    """Consent status"""
    PENDING = "pending"
    GRANTED = "granted"
    DENIED = "denied"
    REVOKED = "revoked"
    EXPIRED = "expired"


@dataclass
class ConsentRecord:
    """Individual consent record"""
    consent_id: str
    patient_id: str
    consent_type: ConsentType
    status: ConsentStatus
    granted_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    revoked_at: Optional[datetime] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    consent_text: str = ""
    version: str = "1.0"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConsentAuditEntry:
    """Audit entry for consent changes"""
    audit_id: str
    consent_id: str
    patient_id: str
    action: str
    old_status: Optional[ConsentStatus]
    new_status: ConsentStatus
    timestamp: datetime
    actor_id: str
    ip_address: Optional[str] = None


class VoiceConsentService:
    """
    Production-grade consent management for voice/video features
    
    Features:
    - Explicit consent collection with version tracking
    - Consent revocation with immediate effect
    - HIPAA audit trail for all consent changes
    - Consent verification before feature access
    """
    
    CONSENT_TEXTS = {
        ConsentType.VOICE_RECORDING: (
            "I consent to having my voice recorded during conversations with "
            "the Followup AI health platform. These recordings may be used to "
            "improve the quality of care and for my personal health records."
        ),
        ConsentType.VIDEO_RECORDING: (
            "I consent to having video consultations recorded. These recordings "
            "will be stored securely and may be reviewed by my healthcare provider."
        ),
        ConsentType.VOICE_TRANSCRIPTION: (
            "I consent to having my voice conversations transcribed to text. "
            "These transcriptions help maintain accurate health records."
        ),
        ConsentType.AI_VOICE_INTERACTION: (
            "I consent to interacting with AI-powered voice agents (Agent Clona) "
            "for health monitoring and support purposes."
        ),
        ConsentType.DATA_STORAGE: (
            "I consent to having my health data stored securely on the "
            "Followup AI platform in accordance with HIPAA regulations."
        ),
        ConsentType.DATA_SHARING: (
            "I consent to having my health data shared with my designated "
            "healthcare providers for treatment purposes."
        ),
    }
    
    def __init__(self):
        self._consents: Dict[str, ConsentRecord] = {}
        self._patient_consents: Dict[str, Dict[ConsentType, str]] = {}
        self._audit_log: List[ConsentAuditEntry] = []
    
    def _log_audit(
        self,
        consent_id: str,
        patient_id: str,
        action: str,
        old_status: Optional[ConsentStatus],
        new_status: ConsentStatus,
        actor_id: str,
        ip_address: Optional[str] = None,
    ) -> None:
        """Log audit entry for consent operation"""
        entry = ConsentAuditEntry(
            audit_id=str(uuid.uuid4()),
            consent_id=consent_id,
            patient_id=patient_id,
            action=action,
            old_status=old_status,
            new_status=new_status,
            timestamp=datetime.utcnow(),
            actor_id=actor_id,
            ip_address=ip_address,
        )
        self._audit_log.append(entry)
        
        HIPAAAuditLogger.log_phi_access(
            actor_id=actor_id,
            actor_role="patient",
            patient_id=patient_id,
            action=f"consent_{action}",
            phi_categories=["consent"],
            resource_type="consent",
            resource_id=consent_id,
            access_reason=f"Consent {action}: {new_status.value}",
        )
    
    def get_consent_text(self, consent_type: ConsentType) -> str:
        """Get the consent text for a specific type"""
        return self.CONSENT_TEXTS.get(consent_type, "")
    
    def request_consent(
        self,
        patient_id: str,
        consent_type: ConsentType,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> ConsentRecord:
        """
        Create a pending consent request
        
        Args:
            patient_id: Patient requesting consent
            consent_type: Type of consent needed
            ip_address: Client IP address
            user_agent: Client user agent
            
        Returns:
            Pending consent record
        """
        consent_id = str(uuid.uuid4())
        
        consent = ConsentRecord(
            consent_id=consent_id,
            patient_id=patient_id,
            consent_type=consent_type,
            status=ConsentStatus.PENDING,
            consent_text=self.get_consent_text(consent_type),
            ip_address=ip_address,
            user_agent=user_agent,
        )
        
        self._consents[consent_id] = consent
        
        self._log_audit(
            consent_id=consent_id,
            patient_id=patient_id,
            action="request",
            old_status=None,
            new_status=ConsentStatus.PENDING,
            actor_id=patient_id,
            ip_address=ip_address,
        )
        
        return consent
    
    def grant_consent(
        self,
        consent_id: str,
        patient_id: str,
        ip_address: Optional[str] = None,
        expires_in_days: Optional[int] = 365,
    ) -> Optional[ConsentRecord]:
        """
        Grant a pending consent
        
        Args:
            consent_id: ID of consent to grant
            patient_id: Patient granting (must match)
            ip_address: Client IP for audit
            expires_in_days: Consent expiration
            
        Returns:
            Updated consent record
        """
        consent = self._consents.get(consent_id)
        if not consent or consent.patient_id != patient_id:
            return None
        
        old_status = consent.status
        now = datetime.utcnow()
        
        consent.status = ConsentStatus.GRANTED
        consent.granted_at = now
        if expires_in_days:
            from datetime import timedelta
            consent.expires_at = now + timedelta(days=expires_in_days)
        
        if patient_id not in self._patient_consents:
            self._patient_consents[patient_id] = {}
        self._patient_consents[patient_id][consent.consent_type] = consent_id
        
        self._log_audit(
            consent_id=consent_id,
            patient_id=patient_id,
            action="grant",
            old_status=old_status,
            new_status=ConsentStatus.GRANTED,
            actor_id=patient_id,
            ip_address=ip_address,
        )
        
        logger.info(f"Consent granted: {consent_id} for {consent.consent_type}")
        return consent
    
    def deny_consent(
        self,
        consent_id: str,
        patient_id: str,
        ip_address: Optional[str] = None,
    ) -> Optional[ConsentRecord]:
        """Deny a pending consent request"""
        consent = self._consents.get(consent_id)
        if not consent or consent.patient_id != patient_id:
            return None
        
        old_status = consent.status
        consent.status = ConsentStatus.DENIED
        
        self._log_audit(
            consent_id=consent_id,
            patient_id=patient_id,
            action="deny",
            old_status=old_status,
            new_status=ConsentStatus.DENIED,
            actor_id=patient_id,
            ip_address=ip_address,
        )
        
        return consent
    
    def revoke_consent(
        self,
        consent_type: ConsentType,
        patient_id: str,
        ip_address: Optional[str] = None,
    ) -> bool:
        """
        Revoke a previously granted consent
        
        Args:
            consent_type: Type of consent to revoke
            patient_id: Patient revoking
            ip_address: Client IP for audit
            
        Returns:
            True if consent was revoked
        """
        patient_consents = self._patient_consents.get(patient_id, {})
        consent_id = patient_consents.get(consent_type)
        
        if not consent_id:
            return False
        
        consent = self._consents.get(consent_id)
        if not consent:
            return False
        
        old_status = consent.status
        consent.status = ConsentStatus.REVOKED
        consent.revoked_at = datetime.utcnow()
        
        del self._patient_consents[patient_id][consent_type]
        
        self._log_audit(
            consent_id=consent_id,
            patient_id=patient_id,
            action="revoke",
            old_status=old_status,
            new_status=ConsentStatus.REVOKED,
            actor_id=patient_id,
            ip_address=ip_address,
        )
        
        logger.info(f"Consent revoked: {consent_type} for patient {patient_id}")
        return True
    
    def has_consent(
        self,
        patient_id: str,
        consent_type: ConsentType,
    ) -> bool:
        """
        Check if patient has active consent for a feature
        
        Args:
            patient_id: Patient to check
            consent_type: Type of consent needed
            
        Returns:
            True if valid consent exists
        """
        patient_consents = self._patient_consents.get(patient_id, {})
        consent_id = patient_consents.get(consent_type)
        
        if not consent_id:
            return False
        
        consent = self._consents.get(consent_id)
        if not consent:
            return False
        
        if consent.status != ConsentStatus.GRANTED:
            return False
        
        if consent.expires_at and consent.expires_at < datetime.utcnow():
            consent.status = ConsentStatus.EXPIRED
            return False
        
        return True
    
    def get_patient_consents(
        self,
        patient_id: str,
    ) -> List[ConsentRecord]:
        """Get all consent records for a patient"""
        return [
            c for c in self._consents.values()
            if c.patient_id == patient_id
        ]
    
    def get_required_consents(
        self,
        patient_id: str,
        feature: str,
    ) -> List[ConsentType]:
        """
        Get list of consents required for a feature
        
        Args:
            patient_id: Patient
            feature: Feature name
            
        Returns:
            List of required but missing consents
        """
        feature_consents = {
            "voice_call": [ConsentType.AI_VOICE_INTERACTION, ConsentType.VOICE_RECORDING],
            "video_call": [ConsentType.VIDEO_RECORDING, ConsentType.DATA_STORAGE],
            "voice_transcription": [ConsentType.VOICE_TRANSCRIPTION],
            "data_sharing": [ConsentType.DATA_SHARING],
        }
        
        required = feature_consents.get(feature, [])
        missing = [ct for ct in required if not self.has_consent(patient_id, ct)]
        
        return missing
    
    def get_consent_audit(
        self,
        patient_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[ConsentAuditEntry]:
        """Get audit log entries for consents"""
        entries = self._audit_log
        if patient_id:
            entries = [e for e in entries if e.patient_id == patient_id]
        return sorted(entries, key=lambda e: e.timestamp, reverse=True)[:limit]


_consent_service: Optional[VoiceConsentService] = None


def get_consent_service() -> VoiceConsentService:
    """Get singleton consent service instance"""
    global _consent_service
    if _consent_service is None:
        _consent_service = VoiceConsentService()
    return _consent_service
