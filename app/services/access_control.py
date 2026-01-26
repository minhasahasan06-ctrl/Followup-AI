"""
HIPAA-Compliant Access Control Service
Unified access control for doctor-patient relationships with comprehensive audit logging.
All patient data access must go through this service.
"""

import logging
import json
import uuid
from typing import Optional, Dict, Any, Tuple, List, Callable
from datetime import datetime
from functools import wraps
from enum import Enum

from fastapi import Depends, HTTPException, Request, status
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.database import get_db, SessionLocal
from app.dependencies import get_current_user
from app.models.user import User

logger = logging.getLogger(__name__)


class AccessScope(str, Enum):
    """Access scope levels for doctor-patient relationships"""
    FULL = "full"
    LIMITED = "limited"
    SUMMARY_ONLY = "summary_only"
    EMERGENCY = "emergency"


class PHICategory(str, Enum):
    """Categories of Protected Health Information"""
    VITALS = "vitals"
    SYMPTOMS = "symptoms"
    MEDICATIONS = "medications"
    MENTAL_HEALTH = "mental_health"
    LAB_RESULTS = "lab_results"
    IMAGING = "imaging"
    APPOINTMENTS = "appointments"
    PRESCRIPTIONS = "prescriptions"
    HABITS = "habits"
    DEVICE_DATA = "device_data"
    VIDEO_EXAMS = "video_exams"
    AUDIO_EXAMS = "audio_exams"
    MESSAGES = "messages"
    CLINICAL_NOTES = "clinical_notes"


class AccessDecision:
    """Result of an access control decision"""
    def __init__(
        self,
        allowed: bool,
        actor_id: str,
        actor_role: str,
        patient_id: Optional[str] = None,
        access_scope: AccessScope = AccessScope.FULL,
        assignment_id: Optional[str] = None,
        reason: Optional[str] = None,
        is_emergency: bool = False,
        connection_info: Optional[Dict[str, Any]] = None
    ):
        self.allowed = allowed
        self.actor_id = actor_id
        self.actor_role = actor_role
        self.patient_id = patient_id
        self.access_scope = access_scope
        self.assignment_id = assignment_id
        self.reason = reason
        self.is_emergency = is_emergency
        self.connection_info = connection_info or {}


class HIPAAAuditLogger:
    """
    HIPAA-compliant audit logging for all PHI access.
    Logs to hipaa_audit_logs table with required fields.
    """
    
    @staticmethod
    def log_phi_access(
        actor_id: str,
        actor_role: str,
        patient_id: str,
        action: str,
        phi_categories: List[str],
        resource_type: str,
        resource_id: Optional[str] = None,
        access_scope: str = "full",
        access_reason: str = "clinical_care",
        consent_verified: bool = True,
        assignment_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        request_path: Optional[str] = None,
        success: bool = True,
        error_message: Optional[str] = None,
        additional_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Log PHI access with full HIPAA audit trail.
        Returns audit_id for tracking.
        """
        audit_id = str(uuid.uuid4())
        
        audit_entry = {
            "audit_id": audit_id,
            "timestamp": datetime.utcnow().isoformat(),
            "actor_id": actor_id,
            "actor_role": actor_role,
            "patient_id": patient_id,
            "action": action,
            "phi_categories": phi_categories,
            "resource_type": resource_type,
            "resource_id": resource_id,
            "access_scope": access_scope,
            "access_reason": access_reason,
            "consent_verified": consent_verified,
            "assignment_id": assignment_id,
            "ip_address": ip_address,
            "user_agent": user_agent,
            "request_path": request_path,
            "success": success,
            "error_message": error_message
        }
        
        logger.warning(f"[HIPAA_AUDIT] {json.dumps(audit_entry)}")
        
        db = None
        try:
            db = SessionLocal()
            
            query = text("""
                INSERT INTO hipaa_audit_logs (
                    id, actor_id, actor_role, patient_id, action,
                    phi_categories, resource_type, resource_id,
                    access_scope, access_reason, consent_verified,
                    assignment_id, ip_address, user_agent, request_path,
                    success, error_message, additional_context, created_at
                ) VALUES (
                    :id, :actor_id, :actor_role, :patient_id, :action,
                    :phi_categories, :resource_type, :resource_id,
                    :access_scope, :access_reason, :consent_verified,
                    :assignment_id, :ip_address, :user_agent, :request_path,
                    :success, :error_message, :additional_context, NOW()
                )
            """)
            
            db.execute(query, {
                "id": audit_id,
                "actor_id": actor_id,
                "actor_role": actor_role,
                "patient_id": patient_id,
                "action": action,
                "phi_categories": json.dumps(phi_categories),
                "resource_type": resource_type,
                "resource_id": resource_id,
                "access_scope": access_scope,
                "access_reason": access_reason,
                "consent_verified": consent_verified,
                "assignment_id": assignment_id,
                "ip_address": ip_address,
                "user_agent": user_agent,
                "request_path": request_path,
                "success": success,
                "error_message": error_message,
                "additional_context": json.dumps(additional_context) if additional_context else None
            })
            db.commit()
            
        except Exception as e:
            logger.error(f"Failed to persist HIPAA audit log: {e}")
            if db:
                db.rollback()
        finally:
            if db:
                db.close()
        
        return audit_id
    
    @staticmethod
    def log_consent_event(
        actor_id: str,
        actor_role: str,
        patient_id: str,
        doctor_id: str,
        event_type: str,
        consent_scope: Optional[str] = None,
        ip_address: Optional[str] = None
    ) -> str:
        """Log consent grant/revoke/modify events"""
        return HIPAAAuditLogger.log_phi_access(
            actor_id=actor_id,
            actor_role=actor_role,
            patient_id=patient_id,
            action=f"consent_{event_type}",
            phi_categories=["consent_records"],
            resource_type="consent",
            access_reason="consent_management",
            ip_address=ip_address,
            additional_context={
                "doctor_id": doctor_id,
                "consent_scope": consent_scope
            }
        )
    
    @staticmethod
    def log_emergency_access(
        actor_id: str,
        actor_role: str,
        patient_id: str,
        emergency_reason: str,
        phi_categories: List[str],
        resource_type: str,
        ip_address: Optional[str] = None
    ) -> str:
        """Log break-the-glass emergency access"""
        return HIPAAAuditLogger.log_phi_access(
            actor_id=actor_id,
            actor_role=actor_role,
            patient_id=patient_id,
            action="emergency_access",
            phi_categories=phi_categories,
            resource_type=resource_type,
            access_scope="emergency",
            access_reason=emergency_reason,
            consent_verified=False,
            additional_context={
                "break_the_glass": True,
                "emergency_reason": emergency_reason
            }
        )


class AccessControlService:
    """
    Unified access control for doctor-patient relationships.
    Provides consistent authorization checks across all routes.
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def verify_doctor_patient_access(
        self,
        db: Session,
        doctor_id: str,
        patient_id: str,
        required_scope: AccessScope = AccessScope.LIMITED,
        phi_categories: Optional[List[str]] = None
    ) -> AccessDecision:
        """
        Verify doctor has appropriate access to patient data.
        
        Args:
            db: Database session
            doctor_id: ID of the doctor requesting access
            patient_id: ID of the patient whose data is being accessed
            required_scope: Minimum required access scope
            phi_categories: Categories of PHI being accessed
        
        Returns:
            AccessDecision with authorization result
        """
        try:
            result = db.execute(
                text("""
                    SELECT id, status, patient_consented, access_scope,
                           is_primary_care_provider, assignment_source
                    FROM doctor_patient_assignments
                    WHERE doctor_id = :doctor_id
                    AND patient_id = :patient_id
                    AND status = 'active'
                    LIMIT 1
                """),
                {"doctor_id": doctor_id, "patient_id": patient_id}
            )
            row = result.fetchone()
            
            if row:
                assignment_id = str(row[0])
                patient_consented = row[2]
                access_scope_str = row[3] or "full"
                is_primary = row[4]
                
                if not patient_consented:
                    return AccessDecision(
                        allowed=False,
                        actor_id=doctor_id,
                        actor_role="doctor",
                        patient_id=patient_id,
                        reason="patient_consent_not_given",
                        assignment_id=assignment_id
                    )
                
                try:
                    access_scope = AccessScope(access_scope_str)
                except ValueError:
                    access_scope = AccessScope.FULL
                
                scope_hierarchy = {
                    AccessScope.SUMMARY_ONLY: 1,
                    AccessScope.LIMITED: 2,
                    AccessScope.FULL: 3,
                    AccessScope.EMERGENCY: 4
                }
                
                if scope_hierarchy.get(access_scope, 0) < scope_hierarchy.get(required_scope, 0):
                    return AccessDecision(
                        allowed=False,
                        actor_id=doctor_id,
                        actor_role="doctor",
                        patient_id=patient_id,
                        access_scope=access_scope,
                        reason="insufficient_access_scope",
                        assignment_id=assignment_id
                    )
                
                return AccessDecision(
                    allowed=True,
                    actor_id=doctor_id,
                    actor_role="doctor",
                    patient_id=patient_id,
                    access_scope=access_scope,
                    assignment_id=assignment_id,
                    connection_info={
                        "is_primary_care_provider": is_primary,
                        "assignment_source": row[5]
                    }
                )
            
            sharing_result = db.execute(
                text("""
                    SELECT id, status, access_level, expires_at
                    FROM patient_sharing_links
                    WHERE doctor_id = :doctor_id
                    AND patient_id = :patient_id
                    AND status = 'active'
                    AND (expires_at IS NULL OR expires_at > NOW())
                    LIMIT 1
                """),
                {"doctor_id": doctor_id, "patient_id": patient_id}
            )
            sharing_row = sharing_result.fetchone()
            
            if sharing_row:
                access_level = sharing_row[2] or "limited"
                try:
                    access_scope = AccessScope(access_level)
                except ValueError:
                    access_scope = AccessScope.LIMITED
                
                return AccessDecision(
                    allowed=True,
                    actor_id=doctor_id,
                    actor_role="doctor",
                    patient_id=patient_id,
                    access_scope=access_scope,
                    assignment_id=str(sharing_row[0]),
                    connection_info={"source": "sharing_link"}
                )
            
            return AccessDecision(
                allowed=False,
                actor_id=doctor_id,
                actor_role="doctor",
                patient_id=patient_id,
                reason="no_active_connection"
            )
            
        except Exception as e:
            logger.error(f"Error verifying doctor-patient access: {e}")
            return AccessDecision(
                allowed=False,
                actor_id=doctor_id,
                actor_role="doctor",
                patient_id=patient_id,
                reason=f"verification_error: {str(e)}"
            )
    
    def verify_patient_self_access(
        self,
        patient_id: str,
        requesting_user_id: str
    ) -> AccessDecision:
        """Verify patient is accessing their own data"""
        if patient_id == requesting_user_id:
            return AccessDecision(
                allowed=True,
                actor_id=requesting_user_id,
                actor_role="patient",
                patient_id=patient_id,
                access_scope=AccessScope.FULL,
                reason="self_access"
            )
        return AccessDecision(
            allowed=False,
            actor_id=requesting_user_id,
            actor_role="patient",
            patient_id=patient_id,
            reason="cannot_access_other_patient_data"
        )
    
    def check_access(
        self,
        db: Session,
        current_user: User,
        patient_id: str,
        required_scope: AccessScope = AccessScope.LIMITED,
        phi_categories: Optional[List[str]] = None,
        request: Optional[Request] = None,
        resource_type: str = "patient_data",
        resource_id: Optional[str] = None,
        access_reason: str = "clinical_care"
    ) -> AccessDecision:
        """
        Universal access check for any user type.
        Automatically logs PHI access to audit trail.
        
        Args:
            db: Database session
            current_user: The authenticated user
            patient_id: Patient whose data is being accessed
            required_scope: Minimum access scope required
            phi_categories: PHI categories being accessed
            request: FastAPI request for IP/user-agent logging
            resource_type: Type of resource being accessed
            resource_id: Specific resource ID
            access_reason: Reason for accessing data
        
        Returns:
            AccessDecision with authorization result
        """
        user_role = str(current_user.role) if current_user.role else "unknown"
        user_id = str(current_user.id)
        
        ip_address = None
        user_agent = None
        request_path = None
        
        if request:
            ip_address = request.client.host if request.client else None
            user_agent = request.headers.get("user-agent")
            request_path = str(request.url.path)
        
        if user_role == "patient":
            decision = self.verify_patient_self_access(patient_id, user_id)
        elif user_role == "doctor":
            decision = self.verify_doctor_patient_access(
                db=db,
                doctor_id=user_id,
                patient_id=patient_id,
                required_scope=required_scope,
                phi_categories=phi_categories
            )
        elif user_role == "admin":
            decision = AccessDecision(
                allowed=True,
                actor_id=user_id,
                actor_role="admin",
                patient_id=patient_id,
                access_scope=AccessScope.FULL,
                reason="admin_access"
            )
        else:
            decision = AccessDecision(
                allowed=False,
                actor_id=user_id,
                actor_role=user_role,
                patient_id=patient_id,
                reason="unknown_role"
            )
        
        HIPAAAuditLogger.log_phi_access(
            actor_id=user_id,
            actor_role=user_role,
            patient_id=patient_id,
            action="access_check",
            phi_categories=phi_categories or [],
            resource_type=resource_type,
            resource_id=resource_id,
            access_scope=decision.access_scope.value if decision.access_scope else "unknown",
            access_reason=access_reason,
            consent_verified=decision.allowed,
            assignment_id=decision.assignment_id,
            ip_address=ip_address,
            user_agent=user_agent,
            request_path=request_path,
            success=decision.allowed,
            error_message=decision.reason if not decision.allowed else None
        )
        
        return decision


_access_control_service = AccessControlService()


def get_access_control() -> AccessControlService:
    """Get the singleton access control service"""
    return _access_control_service


class RequirePatientAccess:
    """
    FastAPI dependency for requiring patient data access.
    
    Usage:
        @router.get("/patient/{patient_id}/data")
        async def get_patient_data(
            patient_id: str,
            access: AccessDecision = Depends(RequirePatientAccess(
                phi_categories=[PHICategory.VITALS, PHICategory.SYMPTOMS],
                required_scope=AccessScope.LIMITED
            ))
        ):
            # access.allowed is guaranteed True at this point
            # access.access_scope, access.assignment_id etc. available
    """
    
    def __init__(
        self,
        phi_categories: Optional[List[PHICategory]] = None,
        required_scope: AccessScope = AccessScope.LIMITED,
        resource_type: str = "patient_data",
        access_reason: str = "clinical_care",
        allow_emergency: bool = False
    ):
        self.phi_categories = [c.value for c in (phi_categories or [])]
        self.required_scope = required_scope
        self.resource_type = resource_type
        self.access_reason = access_reason
        self.allow_emergency = allow_emergency
    
    async def __call__(
        self,
        patient_id: str,
        request: Request,
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_db)
    ) -> AccessDecision:
        """Execute access control check"""
        
        access_service = get_access_control()
        
        decision = access_service.check_access(
            db=db,
            current_user=current_user,
            patient_id=patient_id,
            required_scope=self.required_scope,
            phi_categories=self.phi_categories,
            request=request,
            resource_type=self.resource_type,
            access_reason=self.access_reason
        )
        
        if not decision.allowed:
            if self.allow_emergency:
                emergency_reason = request.headers.get("X-Emergency-Reason")
                if emergency_reason:
                    HIPAAAuditLogger.log_emergency_access(
                        actor_id=decision.actor_id,
                        actor_role=decision.actor_role,
                        patient_id=patient_id,
                        emergency_reason=emergency_reason,
                        phi_categories=self.phi_categories,
                        resource_type=self.resource_type,
                        ip_address=request.client.host if request.client else None
                    )
                    return AccessDecision(
                        allowed=True,
                        actor_id=decision.actor_id,
                        actor_role=decision.actor_role,
                        patient_id=patient_id,
                        access_scope=AccessScope.EMERGENCY,
                        is_emergency=True,
                        reason="emergency_override"
                    )
            
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={
                    "error": "access_denied",
                    "reason": decision.reason,
                    "patient_id": patient_id
                }
            )
        
        return decision


def require_patient_access(
    phi_categories: Optional[List[PHICategory]] = None,
    required_scope: AccessScope = AccessScope.LIMITED,
    resource_type: str = "patient_data",
    access_reason: str = "clinical_care",
    allow_emergency: bool = False
) -> RequirePatientAccess:
    """
    Factory function for creating patient access dependencies.
    
    Usage:
        access_dep = require_patient_access(
            phi_categories=[PHICategory.SYMPTOMS],
            required_scope=AccessScope.LIMITED
        )
        
        @router.get("/patient/{patient_id}/symptoms")
        async def get_symptoms(
            patient_id: str,
            access: AccessDecision = Depends(access_dep)
        ):
            ...
    """
    return RequirePatientAccess(
        phi_categories=phi_categories,
        required_scope=required_scope,
        resource_type=resource_type,
        access_reason=access_reason,
        allow_emergency=allow_emergency
    )


async def verify_and_log_access(
    db: Session,
    current_user: User,
    patient_id: str,
    action: str,
    phi_categories: List[str],
    resource_type: str,
    resource_id: Optional[str] = None,
    required_scope: AccessScope = AccessScope.LIMITED,
    request: Optional[Request] = None,
    access_reason: str = "clinical_care"
) -> AccessDecision:
    """
    Verify access and log to audit trail in one call.
    
    For use in routes that need more control than the dependency.
    Raises HTTPException if access denied.
    """
    access_service = get_access_control()
    
    decision = access_service.check_access(
        db=db,
        current_user=current_user,
        patient_id=patient_id,
        required_scope=required_scope,
        phi_categories=phi_categories,
        request=request,
        resource_type=resource_type,
        resource_id=resource_id,
        access_reason=access_reason
    )
    
    if not decision.allowed:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "error": "access_denied",
                "reason": decision.reason,
                "patient_id": patient_id
            }
        )
    
    HIPAAAuditLogger.log_phi_access(
        actor_id=decision.actor_id,
        actor_role=decision.actor_role,
        patient_id=patient_id,
        action=action,
        phi_categories=phi_categories,
        resource_type=resource_type,
        resource_id=resource_id,
        access_scope=decision.access_scope.value,
        access_reason=access_reason,
        consent_verified=True,
        assignment_id=decision.assignment_id,
        ip_address=request.client.host if request and request.client else None,
        user_agent=request.headers.get("user-agent") if request else None,
        request_path=str(request.url.path) if request else None
    )
    
    return decision


def log_data_modification(
    actor_id: str,
    actor_role: str,
    patient_id: str,
    action: str,
    phi_categories: List[str],
    resource_type: str,
    resource_id: str,
    old_value: Optional[Dict] = None,
    new_value: Optional[Dict] = None,
    request: Optional[Request] = None
) -> str:
    """
    Log data modifications (create/update/delete) to audit trail.
    """
    return HIPAAAuditLogger.log_phi_access(
        actor_id=actor_id,
        actor_role=actor_role,
        patient_id=patient_id,
        action=action,
        phi_categories=phi_categories,
        resource_type=resource_type,
        resource_id=resource_id,
        access_reason="data_modification",
        ip_address=request.client.host if request and request.client else None,
        user_agent=request.headers.get("user-agent") if request else None,
        request_path=str(request.url.path) if request else None,
        additional_context={
            "modification_type": action,
            "old_value_summary": _summarize_for_audit(old_value) if old_value else None,
            "new_value_summary": _summarize_for_audit(new_value) if new_value else None
        }
    )


def _summarize_for_audit(data: Dict) -> Dict:
    """Create a summary of data for audit logging (avoid storing full PHI in logs)"""
    if not data:
        return {}
    
    summary = {}
    for key in data:
        value = data[key]
        if isinstance(value, str) and len(value) > 50:
            summary[key] = f"[string:{len(value)} chars]"
        elif isinstance(value, (list, dict)):
            summary[key] = f"[{type(value).__name__}:{len(value)} items]"
        else:
            summary[key] = "[redacted]" if key in ("ssn", "password", "token") else str(value)[:20]
    
    return summary
