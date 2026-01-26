"""
Immutable Audit Log Service for HIPAA Compliance

Provides append-only audit logging for all PHI accesses with:
- Actor, action, patient, timestamp tracking
- Cryptographic hash chain for tamper detection
- Retention policy management
- Query interface for compliance audits

HIPAA Requirement: 45 CFR 164.312(b) - Audit controls
"""

import hashlib
import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field, asdict
from enum import Enum
import uuid


logger = logging.getLogger(__name__)


class AuditAction(str, Enum):
    """Standard HIPAA audit actions"""
    CREATE = "CREATE"
    READ = "READ"
    UPDATE = "UPDATE"
    DELETE = "DELETE"
    EXPORT = "EXPORT"
    PRINT = "PRINT"
    QUERY = "QUERY"
    LOGIN = "LOGIN"
    LOGOUT = "LOGOUT"
    ACCESS_DENIED = "ACCESS_DENIED"
    PHI_DETECTED = "PHI_DETECTED"
    EMBEDDING_CREATED = "EMBEDDING_CREATED"
    MEMORY_STORED = "MEMORY_STORED"
    MEMORY_RETRIEVED = "MEMORY_RETRIEVED"
    MODEL_INFERENCE = "MODEL_INFERENCE"
    CONSENT_GRANTED = "CONSENT_GRANTED"
    CONSENT_REVOKED = "CONSENT_REVOKED"


class PHICategory(str, Enum):
    """Categories of PHI for audit classification"""
    DEMOGRAPHICS = "DEMOGRAPHICS"
    MEDICAL_RECORD = "MEDICAL_RECORD"
    DIAGNOSIS = "DIAGNOSIS"
    TREATMENT = "TREATMENT"
    MEDICATION = "MEDICATION"
    LAB_RESULTS = "LAB_RESULTS"
    IMAGING = "IMAGING"
    MENTAL_HEALTH = "MENTAL_HEALTH"
    SUBSTANCE_ABUSE = "SUBSTANCE_ABUSE"
    GENETIC = "GENETIC"
    HIV_AIDS = "HIV_AIDS"
    BEHAVIORAL = "BEHAVIORAL"
    VOICE_RECORDING = "VOICE_RECORDING"
    VIDEO_RECORDING = "VIDEO_RECORDING"


@dataclass
class AuditEntry:
    """Immutable audit log entry"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    actor_id: str = ""
    actor_type: str = ""  # user, system, agent, service
    actor_ip: Optional[str] = None
    action: AuditAction = AuditAction.READ
    resource_type: str = ""  # patient, memory, model, etc.
    resource_id: Optional[str] = None
    patient_id: Optional[str] = None
    phi_categories: List[str] = field(default_factory=list)
    access_reason: Optional[str] = None
    success: bool = True
    error_message: Optional[str] = None
    request_id: Optional[str] = None
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    previous_hash: Optional[str] = None
    entry_hash: str = ""
    
    def __post_init__(self):
        if not self.entry_hash:
            self.entry_hash = self._compute_hash()
    
    def _compute_hash(self) -> str:
        """Compute cryptographic hash for tamper detection"""
        hash_content = {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "actor_id": self.actor_id,
            "action": self.action.value if isinstance(self.action, AuditAction) else self.action,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "patient_id": self.patient_id,
            "previous_hash": self.previous_hash
        }
        content_str = json.dumps(hash_content, sort_keys=True)
        return hashlib.sha256(content_str.encode()).hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["timestamp"] = self.timestamp.isoformat()
        result["action"] = self.action.value if isinstance(self.action, AuditAction) else self.action
        return result


class ImmutableAuditLog:
    """
    Append-only audit log with hash chain verification.
    
    Features:
    - Append-only writes (no updates or deletes)
    - Cryptographic hash chain for tamper detection
    - Retention policy enforcement
    - Query interface for compliance audits
    """
    
    RETENTION_DAYS_DEFAULT = 2190  # 6 years per HIPAA
    
    def __init__(
        self,
        storage_backend: Optional[Any] = None,
        retention_days: int = RETENTION_DAYS_DEFAULT
    ):
        self._entries: List[AuditEntry] = []
        self._storage = storage_backend
        self._retention_days = retention_days
        self._last_hash: Optional[str] = None
    
    def append(
        self,
        actor_id: str,
        action: AuditAction,
        resource_type: str,
        resource_id: Optional[str] = None,
        patient_id: Optional[str] = None,
        phi_categories: Optional[List[PHICategory]] = None,
        actor_type: str = "user",
        actor_ip: Optional[str] = None,
        access_reason: Optional[str] = None,
        success: bool = True,
        error_message: Optional[str] = None,
        request_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AuditEntry:
        """
        Append an entry to the immutable audit log.
        
        This is the ONLY write operation - no updates or deletes allowed.
        """
        entry = AuditEntry(
            actor_id=actor_id,
            actor_type=actor_type,
            actor_ip=actor_ip,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            patient_id=patient_id,
            phi_categories=[c.value if isinstance(c, PHICategory) else c for c in (phi_categories or [])],
            access_reason=access_reason,
            success=success,
            error_message=error_message,
            request_id=request_id,
            session_id=session_id,
            metadata=metadata or {},
            previous_hash=self._last_hash
        )
        
        self._entries.append(entry)
        self._last_hash = entry.entry_hash
        
        if self._storage:
            self._persist_entry(entry)
        
        logger.info(
            f"AUDIT: {action.value} on {resource_type}/{resource_id} "
            f"by {actor_type}:{actor_id} patient={patient_id} success={success}"
        )
        
        return entry
    
    def _persist_entry(self, entry: AuditEntry) -> None:
        """Persist entry to storage backend"""
        if hasattr(self._storage, 'append_audit_entry'):
            self._storage.append_audit_entry(entry.to_dict())
    
    def verify_chain_integrity(self) -> Tuple[bool, Optional[str]]:
        """
        Verify the hash chain integrity of the audit log.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not self._entries:
            return True, None
        
        for i, entry in enumerate(self._entries):
            expected_previous = self._entries[i-1].entry_hash if i > 0 else None
            
            if entry.previous_hash != expected_previous:
                return False, f"Chain broken at entry {entry.id}: previous_hash mismatch"
            
            recomputed_hash = entry._compute_hash()
            if entry.entry_hash != recomputed_hash:
                return False, f"Entry {entry.id} has been tampered with"
        
        return True, None
    
    def query(
        self,
        patient_id: Optional[str] = None,
        actor_id: Optional[str] = None,
        action: Optional[AuditAction] = None,
        resource_type: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        success_only: Optional[bool] = None,
        limit: int = 100
    ) -> List[AuditEntry]:
        """Query audit log entries with filters"""
        results = []
        
        for entry in reversed(self._entries):
            if patient_id and entry.patient_id != patient_id:
                continue
            if actor_id and entry.actor_id != actor_id:
                continue
            if action and entry.action != action:
                continue
            if resource_type and entry.resource_type != resource_type:
                continue
            if start_time and entry.timestamp < start_time:
                continue
            if end_time and entry.timestamp > end_time:
                continue
            if success_only is not None and entry.success != success_only:
                continue
            
            results.append(entry)
            
            if len(results) >= limit:
                break
        
        return results
    
    def get_patient_access_report(
        self,
        patient_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Generate HIPAA-required access report for a patient.
        
        Per 45 CFR 164.528, patients have right to accounting of disclosures.
        """
        entries = self.query(
            patient_id=patient_id,
            start_time=start_time,
            end_time=end_time,
            limit=10000
        )
        
        return {
            "patient_id": patient_id,
            "report_generated": datetime.utcnow().isoformat(),
            "period_start": start_time.isoformat() if start_time else None,
            "period_end": end_time.isoformat() if end_time else None,
            "total_accesses": len(entries),
            "accesses_by_action": self._group_by_action(entries),
            "accesses_by_actor": self._group_by_actor(entries),
            "phi_categories_accessed": self._get_phi_categories(entries),
            "entries": [e.to_dict() for e in entries[:100]]
        }
    
    def _group_by_action(self, entries: List[AuditEntry]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for entry in entries:
            action = entry.action.value if isinstance(entry.action, AuditAction) else entry.action
            counts[action] = counts.get(action, 0) + 1
        return counts
    
    def _group_by_actor(self, entries: List[AuditEntry]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for entry in entries:
            counts[entry.actor_id] = counts.get(entry.actor_id, 0) + 1
        return counts
    
    def _get_phi_categories(self, entries: List[AuditEntry]) -> List[str]:
        categories = set()
        for entry in entries:
            categories.update(entry.phi_categories)
        return sorted(list(categories))
    
    def apply_retention_policy(self) -> int:
        """
        Apply retention policy - mark entries beyond retention for archival.
        
        Note: Does NOT delete entries - moves to cold storage for compliance.
        Returns count of entries marked for archival.
        """
        cutoff = datetime.utcnow() - timedelta(days=self._retention_days)
        archived_count = 0
        
        for entry in self._entries:
            if entry.timestamp < cutoff and "archived" not in entry.metadata:
                entry.metadata["archived"] = True
                entry.metadata["archived_at"] = datetime.utcnow().isoformat()
                archived_count += 1
        
        logger.info(f"Retention policy applied: {archived_count} entries marked for archival")
        return archived_count
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get audit log statistics"""
        if not self._entries:
            return {
                "total_entries": 0,
                "chain_valid": True,
                "oldest_entry": None,
                "newest_entry": None
            }
        
        is_valid, error = self.verify_chain_integrity()
        
        return {
            "total_entries": len(self._entries),
            "chain_valid": is_valid,
            "chain_error": error,
            "oldest_entry": self._entries[0].timestamp.isoformat(),
            "newest_entry": self._entries[-1].timestamp.isoformat(),
            "retention_days": self._retention_days,
            "entries_by_action": self._group_by_action(self._entries),
            "unique_patients": len(set(e.patient_id for e in self._entries if e.patient_id)),
            "unique_actors": len(set(e.actor_id for e in self._entries))
        }


from typing import Tuple

_global_audit_log: Optional[ImmutableAuditLog] = None


def get_audit_log() -> ImmutableAuditLog:
    """Get global audit log instance"""
    global _global_audit_log
    if _global_audit_log is None:
        _global_audit_log = ImmutableAuditLog()
    return _global_audit_log


def log_phi_access(
    actor_id: str,
    action: AuditAction,
    resource_type: str,
    resource_id: Optional[str] = None,
    patient_id: Optional[str] = None,
    phi_categories: Optional[List[PHICategory]] = None,
    **kwargs
) -> AuditEntry:
    """Convenience function for logging PHI access"""
    return get_audit_log().append(
        actor_id=actor_id,
        action=action,
        resource_type=resource_type,
        resource_id=resource_id,
        patient_id=patient_id,
        phi_categories=phi_categories,
        **kwargs
    )
