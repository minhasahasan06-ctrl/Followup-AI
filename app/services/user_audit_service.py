"""
User Audit Service
Centralized logging for key user events (terms, consent, profile changes, doctor assignment)
Separate from HIPAA PHI audit logging - this is user-facing event history
"""

from typing import Optional, Dict, Any, List
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import desc
import logging

from app.database import SessionLocal
from app.models.terms_audit import UserAuditLog

logger = logging.getLogger(__name__)


class AuditEventType:
    """User-facing audit event types"""
    TERMS_ACCEPTED = "terms_accepted"
    RESEARCH_CONSENT_CHANGED = "research_consent_changed"
    DOCTOR_ASSIGNED = "doctor_assigned"
    DOCTOR_UNASSIGNED = "doctor_unassigned"
    MEDICATIONS_UPDATED = "medications_updated"
    ALLERGIES_UPDATED = "allergies_updated"
    EMERGENCY_CONTACTS_UPDATED = "emergency_contacts_updated"
    CHRONIC_CONDITIONS_UPDATED = "chronic_conditions_updated"
    PROFILE_UPDATED = "profile_updated"
    DEVICE_CONNECTED = "device_connected"
    DEVICE_DISCONNECTED = "device_disconnected"


def log_user_audit(
    user_id: str,
    event_type: str,
    event_data: Optional[Dict[str, Any]] = None,
    actor_id: Optional[str] = None,
    ip_address: Optional[str] = None,
    user_agent: Optional[str] = None,
    db: Optional[Session] = None
) -> bool:
    """
    Log a user-facing audit event
    
    Args:
        user_id: ID of the user this event relates to
        event_type: Type of event (use AuditEventType constants)
        event_data: Additional event details
        actor_id: ID of user who performed the action (if different from user_id)
        ip_address: Client IP address
        user_agent: Client user agent
        db: Optional database session (will create one if not provided)
    
    Returns:
        True if logged successfully, False otherwise
    """
    should_close_db = db is None
    if db is None:
        db = SessionLocal()
    
    try:
        log_entry = UserAuditLog(
            user_id=user_id,
            event_type=event_type,
            event_data=event_data or {},
            actor_id=actor_id,
            ip_address=ip_address,
            user_agent=user_agent
        )
        db.add(log_entry)
        db.commit()
        
        logger.info(f"[USER_AUDIT] {event_type} for user {user_id}")
        return True
        
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to log user audit event: {e}")
        return False
        
    finally:
        if should_close_db:
            db.close()


def get_user_audit_logs(
    user_id: str,
    page: int = 1,
    page_size: int = 20,
    event_types: Optional[List[str]] = None,
    db: Optional[Session] = None
) -> Dict[str, Any]:
    """
    Get audit logs for a specific user
    
    Args:
        user_id: ID of the user
        page: Page number (1-indexed)
        page_size: Number of entries per page
        event_types: Optional filter for specific event types
        db: Optional database session
    
    Returns:
        Dict with logs, total count, and pagination info
    """
    should_close_db = db is None
    if db is None:
        db = SessionLocal()
    
    try:
        query = db.query(UserAuditLog).filter(UserAuditLog.user_id == user_id)
        
        if event_types:
            query = query.filter(UserAuditLog.event_type.in_(event_types))
        
        total = query.count()
        
        logs = (
            query
            .order_by(desc(UserAuditLog.created_at))
            .offset((page - 1) * page_size)
            .limit(page_size)
            .all()
        )
        
        return {
            "logs": [
                {
                    "id": log.id,
                    "event_type": log.event_type,
                    "event_data": log.event_data,
                    "actor_id": log.actor_id,
                    "created_at": log.created_at.isoformat() if log.created_at else None
                }
                for log in logs
            ],
            "total": total,
            "page": page,
            "page_size": page_size
        }
        
    except Exception as e:
        logger.error(f"Failed to get user audit logs: {e}")
        return {"logs": [], "total": 0, "page": page, "page_size": page_size}
        
    finally:
        if should_close_db:
            db.close()
