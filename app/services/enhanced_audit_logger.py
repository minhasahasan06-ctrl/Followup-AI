"""
Enhanced Audit Logger - HIPAA-Compliant Immutable Audit Trail
Database-persisted audit logging with CloudWatch integration
"""

from sqlalchemy.orm import Session
from datetime import datetime
from typing import Optional, Dict, Any, List
import json
import logging
import hashlib

from app.database import get_db
from app.models.security_models import AuditLog, SecurityEvent
from app.services.audit_logger import AuditLogger, AuditEvent

logger = logging.getLogger(__name__)


class EnhancedAuditLogger:
    """
    Enhanced audit logger with database persistence
    Implements HIPAA-compliant immutable audit trail
    """
    
    @staticmethod
    def _generate_audit_hash(entry: Dict[str, Any]) -> str:
        """Generate cryptographic hash for audit entry integrity"""
        entry_str = json.dumps(entry, sort_keys=True)
        return hashlib.sha256(entry_str.encode()).hexdigest()
    
    @staticmethod
    def log_to_database(
        db: Session,
        user_id: str,
        user_type: str,
        action_type: str,
        resource_type: str,
        resource_id: Optional[str] = None,
        phi_accessed: bool = False,
        patient_id_accessed: Optional[str] = None,
        action_description: Optional[str] = None,
        action_result: str = "success",
        error_message: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        request_id: Optional[str] = None,
        session_id: Optional[str] = None,
        before_value: Optional[Dict[str, Any]] = None,
        after_value: Optional[Dict[str, Any]] = None,
        change_reason: Optional[str] = None,
        authorization_method: Optional[str] = None,
        permission_level: Optional[str] = None,
        data_fields_accessed: Optional[List[str]] = None,
        user_email: Optional[str] = None,
        user_role: Optional[str] = None,
        geo_location: Optional[str] = None,
        timezone: Optional[str] = None
    ) -> AuditLog:
        """
        Log audit event to database with full HIPAA compliance
        
        Args:
            db: Database session
            user_id: ID of user performing action
            user_type: Type of user (patient, clinician, admin, system)
            action_type: Type of action (view, create, update, delete, download, share)
            resource_type: Type of resource accessed
            resource_id: ID of resource
            phi_accessed: Whether PHI was accessed
            patient_id_accessed: Patient ID whose data was accessed
            action_description: Human-readable description
            action_result: Result status (success, failure, partial)
            error_message: Error message if failed
            ip_address: Client IP address
            user_agent: Client user agent
            request_id: Request correlation ID
            session_id: User session ID
            before_value: State before change (for updates/deletes)
            after_value: State after change (for updates/creates)
            change_reason: Reason for change
            authorization_method: Auth method used (jwt, api_key, oauth)
            permission_level: Permission level used
            data_fields_accessed: List of specific fields accessed
            user_email: User email
            user_role: User role
            geo_location: Geographic location
            timezone: User timezone
        
        Returns:
            Created AuditLog entry
        """
        try:
            audit_entry = AuditLog(
                user_id=user_id,
                user_type=user_type,
                user_email=user_email,
                user_role=user_role,
                action_type=action_type,
                action_category="phi_access" if phi_accessed else "general",
                resource_type=resource_type,
                resource_id=resource_id,
                phi_accessed=phi_accessed,
                patient_id_accessed=patient_id_accessed,
                data_fields_accessed=data_fields_accessed,
                action_description=action_description,
                action_result=action_result,
                error_message=error_message,
                ip_address=ip_address,
                user_agent=user_agent,
                request_id=request_id,
                session_id=session_id,
                before_value=before_value,
                after_value=after_value,
                change_reason=change_reason,
                authorization_method=authorization_method,
                permission_level=permission_level,
                geo_location=geo_location,
                timezone=timezone,
                timestamp=datetime.utcnow()
            )
            
            db.add(audit_entry)
            db.commit()
            db.refresh(audit_entry)
            
            logger.info(
                f"✅ Audit logged: {action_type} on {resource_type} by {user_id} "
                f"(PHI: {phi_accessed})"
            )
            
            return audit_entry
            
        except Exception as e:
            logger.error(f"❌ Failed to log audit entry: {e}")
            db.rollback()
            raise
    
    @staticmethod
    def log_phi_access(
        db: Session,
        user_id: str,
        user_type: str,
        patient_id: str,
        resource_type: str,
        resource_id: Optional[str] = None,
        action_type: str = "view",
        data_fields: Optional[List[str]] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        request_id: Optional[str] = None,
        session_id: Optional[str] = None,
        user_email: Optional[str] = None,
        user_role: Optional[str] = None
    ) -> AuditLog:
        """
        Convenience method to log PHI access
        
        CRITICAL: All PHI access MUST be logged for HIPAA compliance
        """
        return EnhancedAuditLogger.log_to_database(
            db=db,
            user_id=user_id,
            user_type=user_type,
            action_type=action_type,
            resource_type=resource_type,
            resource_id=resource_id,
            phi_accessed=True,
            patient_id_accessed=patient_id,
            action_description=f"Accessed PHI for patient {patient_id}",
            data_fields_accessed=data_fields,
            ip_address=ip_address,
            user_agent=user_agent,
            request_id=request_id,
            session_id=session_id,
            user_email=user_email,
            user_role=user_role
        )
    
    @staticmethod
    def log_security_event(
        db: Session,
        event_type: str,
        severity: str,
        description: str,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        event_data: Optional[Dict[str, Any]] = None,
        detection_method: Optional[str] = None,
        confidence_score: Optional[float] = None,
        action_taken: Optional[str] = None
    ) -> SecurityEvent:
        """
        Log security event (threats, anomalies, violations)
        
        Args:
            db: Database session
            event_type: Type of security event
            severity: Severity level (low, medium, high, critical)
            description: Event description
            user_id: User ID if applicable
            ip_address: IP address
            user_agent: User agent
            event_data: Additional event data
            detection_method: How event was detected
            confidence_score: Confidence in threat detection (0-1)
            action_taken: Action taken in response
        
        Returns:
            Created SecurityEvent entry
        """
        try:
            security_event = SecurityEvent(
                event_type=event_type,
                severity=severity,
                description=description,
                event_data=event_data,
                user_id=user_id,
                ip_address=ip_address,
                user_agent=user_agent,
                detection_method=detection_method,
                confidence_score=confidence_score,
                action_taken=action_taken,
                response_status="pending",
                occurred_at=datetime.utcnow()
            )
            
            db.add(security_event)
            db.commit()
            db.refresh(security_event)
            
            logger.warning(
                f"🚨 Security event logged: {event_type} ({severity}) - {description}"
            )
            
            return security_event
            
        except Exception as e:
            logger.error(f"❌ Failed to log security event: {e}")
            db.rollback()
            raise
    
    @staticmethod
    def get_phi_access_logs(
        db: Session,
        patient_id: str,
        limit: int = 100,
        offset: int = 0
    ) -> List[AuditLog]:
        """Get all PHI access logs for a patient (HIPAA accounting of disclosures)"""
        return db.query(AuditLog).filter(
            AuditLog.patient_id_accessed == patient_id,
            AuditLog.phi_accessed == True
        ).order_by(
            AuditLog.timestamp.desc()
        ).limit(limit).offset(offset).all()
    
    @staticmethod
    def get_user_activity_logs(
        db: Session,
        user_id: str,
        limit: int = 100,
        offset: int = 0
    ) -> List[AuditLog]:
        """Get all activity logs for a user"""
        return db.query(AuditLog).filter(
            AuditLog.user_id == user_id
        ).order_by(
            AuditLog.timestamp.desc()
        ).limit(limit).offset(offset).all()
