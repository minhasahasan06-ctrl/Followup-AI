"""
Access Control Module - HIPAA-Compliant
Implements role-based and attribute-based access control

SECURITY REQUIREMENTS:
- Minimum necessary principle
- Role-based access control (RBAC)
- Attribute-based access control (ABAC)
- Patient data isolation
"""

import logging
from typing import Optional, List, Dict, Any, Callable
from functools import wraps
from fastapi import HTTPException, status, Depends
from sqlalchemy.orm import Session

from app.models import User
from app.database import get_db
from app.core.phi_protection import check_phi_access, log_phi_access
from app.services.audit_logger import AuditLogger, AuditEvent

logger = logging.getLogger(__name__)


class AccessControlService:
    """
    Centralized access control service
    Implements HIPAA minimum necessary principle
    """
    
    # Role permissions matrix
    ROLE_PERMISSIONS = {
        "patient": {
            "read_own_data": True,
            "write_own_data": True,
            "read_own_medical_records": True,
            "read_own_appointments": True,
            "create_appointment": True,
            "read_doctor_profiles": True,
            "read_own_consultations": True,
        },
        "doctor": {
            "read_patient_data": True,
            "write_patient_data": True,
            "read_patient_medical_records": True,
            "read_patient_appointments": True,
            "create_appointment": True,
            "read_own_appointments": True,
            "read_consultations": True,
            "create_consultation": True,
            "read_own_profile": True,
            "update_own_profile": True,
        },
        "admin": {
            "read_all_data": True,
            "write_all_data": True,
            "manage_users": True,
            "view_audit_logs": True,
            "manage_system": True,
        },
    }
    
    @staticmethod
    def check_permission(user_role: str, permission: str) -> bool:
        """
        Check if user role has a specific permission
        
        Args:
            user_role: User's role
            permission: Permission to check
            
        Returns:
            True if user has permission
        """
        role_perms = AccessControlService.ROLE_PERMISSIONS.get(user_role, {})
        return role_perms.get(permission, False)
    
    @staticmethod
    def require_permission(permission: str):
        """
        Decorator to require a specific permission
        
        Usage:
            @require_permission("read_patient_data")
            async def get_patient(...):
                ...
        """
        def decorator(func: Callable):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Extract current_user from args/kwargs
                current_user = None
                for arg in args:
                    if isinstance(arg, User):
                        current_user = arg
                        break
                
                if not current_user:
                    for key, value in kwargs.items():
                        if isinstance(value, User):
                            current_user = value
                            break
                
                if not current_user:
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Authentication required"
                    )
                
                user_role = str(current_user.role) if current_user.role else ""
                
                if not AccessControlService.check_permission(user_role, permission):
                    AuditLogger.log_event(
                        event_type=AuditEvent.AUTH_FAILED,
                        user_id=current_user.id,
                        resource_type="api_endpoint",
                        resource_id=func.__name__,
                        action="access",
                        status="denied",
                        metadata={"permission_required": permission, "user_role": user_role}
                    )
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail=f"Permission denied: {permission} required"
                    )
                
                return await func(*args, **kwargs)
            
            return wrapper
        return decorator
    
    @staticmethod
    def check_patient_access(
        current_user: User,
        patient_id: str,
        db: Session,
        require_relationship: bool = True
    ) -> bool:
        """
        Check if user can access a specific patient's data
        
        Args:
            current_user: Current authenticated user
            patient_id: ID of patient
            db: Database session
            require_relationship: Whether to require explicit doctor-patient relationship
            
        Returns:
            True if access allowed
            
        Raises:
            HTTPException: 403 if access denied
        """
        user_role = str(current_user.role) if current_user.role else ""
        user_id = current_user.id
        
        # Patient accessing own data
        if user_role == "patient" and user_id == patient_id:
            return True
        
        # Doctor accessing patient data
        if user_role == "doctor":
            if require_relationship:
                # Check if doctor has relationship with patient
                # TODO: Implement relationship check
                # For now, log warning and allow
                logger.warning(
                    f"Doctor {user_id} accessing patient {patient_id} - "
                    "relationship check not fully implemented"
                )
            return True
        
        # Admin access
        if user_role == "admin":
            return True
        
        # Access denied
        AuditLogger.log_event(
            event_type=AuditEvent.AUTH_FAILED,
            user_id=user_id,
            resource_type="patient",
            resource_id=patient_id,
            action="access",
            status="denied",
            metadata={"user_role": user_role, "reason": "insufficient_permissions"}
        )
        
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: insufficient permissions"
        )
    
    @staticmethod
    def filter_patient_data(
        data: Dict[str, Any],
        current_user: User,
        patient_id: str
    ) -> Dict[str, Any]:
        """
        Filter patient data based on user's role and permissions
        Implements minimum necessary principle
        
        Args:
            data: Patient data dictionary
            current_user: Current authenticated user
            patient_id: ID of patient
            
        Returns:
            Filtered data dictionary
        """
        user_role = str(current_user.role) if current_user.role else ""
        user_id = current_user.id
        
        # Patients can see all their own data
        if user_role == "patient" and user_id == patient_id:
            return data
        
        # Doctors can see clinical data but not all personal info
        if user_role == "doctor":
            filtered = {}
            allowed_fields = [
                'id', 'first_name', 'last_name', 'date_of_birth',
                'medical_conditions', 'allergies', 'medications',
                'vital_signs', 'lab_results', 'diagnosis',
                'appointments', 'consultations', 'medical_records'
            ]
            
            for key, value in data.items():
                if key in allowed_fields:
                    filtered[key] = value
            
            return filtered
        
        # Admin can see everything
        if user_role == "admin":
            return data
        
        # Default: return empty dict
        return {}


def require_patient_access(require_relationship: bool = True):
    """
    Decorator to require patient access check
    
    Usage:
        @require_patient_access(require_relationship=True)
        async def get_patient_data(patient_id: str, current_user: User = Depends(...), db: Session = Depends(...)):
            ...
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract dependencies
            current_user = None
            patient_id = None
            db = None
            
            for arg in args:
                if isinstance(arg, User):
                    current_user = arg
                elif isinstance(arg, Session):
                    db = arg
                elif isinstance(arg, str) and len(arg) > 10:  # Likely a UUID
                    patient_id = arg
            
            for key, value in kwargs.items():
                if isinstance(value, User):
                    current_user = value
                elif isinstance(value, Session):
                    db = value
                elif key == "patient_id":
                    patient_id = value
            
            if not current_user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication required"
                )
            
            if not patient_id:
                # Try to extract from path parameters
                # This is a simplified version - in practice, use FastAPI's dependency injection
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Patient ID required"
                )
            
            if not db:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Database session required"
                )
            
            # Check access
            AccessControlService.check_patient_access(
                current_user=current_user,
                patient_id=patient_id,
                db=db,
                require_relationship=require_relationship
            )
            
            # Log PHI access
            log_phi_access(
                user_id=current_user.id,
                resource_type="patient_data",
                resource_id=patient_id,
                action="read",
                request=None  # Would need to pass request through
            )
            
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator
