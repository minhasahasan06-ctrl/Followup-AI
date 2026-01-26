"""
PHI Protection Module - HIPAA-Compliant
Provides decorators and utilities for protecting Protected Health Information

SECURITY REQUIREMENTS:
- All PHI access must be logged
- PHI must be encrypted at rest
- Access control checks before PHI access
- Audit trail for all PHI operations
"""

import logging
from typing import Callable, Optional, Dict, Any, List
from functools import wraps
from fastapi import Request, HTTPException, status

from app.services.audit_logger import AuditLogger, AuditEvent
from app.services.gcp_kms_service import get_kms_service
from app.models import User

logger = logging.getLogger(__name__)


# PHI field identifiers
PHI_FIELDS = [
    'name', 'first_name', 'last_name', 'full_name',
    'date_of_birth', 'dob', 'birthdate',
    'ssn', 'social_security_number',
    'address', 'street_address', 'city', 'state', 'zip_code', 'postal_code',
    'phone_number', 'phone', 'mobile', 'telephone',
    'email', 'email_address',
    'medical_record_number', 'mrn',
    'diagnosis', 'condition', 'disease',
    'medication', 'prescription',
    'lab_result', 'test_result',
    'vital_signs', 'blood_pressure', 'heart_rate', 'temperature',
    'allergy', 'allergies',
    'procedure', 'surgery',
    'note', 'clinical_note', 'doctor_note',
]


def is_phi_field(field_name: str) -> bool:
    """
    Check if a field name indicates PHI
    
    Args:
        field_name: Name of the field
        
    Returns:
        True if field is likely PHI
    """
    field_lower = field_name.lower()
    return any(phi_field in field_lower for phi_field in PHI_FIELDS)


def log_phi_access(
    user_id: str,
    resource_type: str,
    resource_id: Optional[str],
    action: str,
    fields_accessed: Optional[List[str]] = None,
    request: Optional[Request] = None
):
    """
    Log PHI access for audit trail
    
    Args:
        user_id: ID of user accessing PHI
        resource_type: Type of resource (e.g., 'patient_profile', 'medical_record')
        resource_id: ID of resource
        action: Action performed ('read', 'write', 'delete', 'export')
        fields_accessed: List of PHI fields accessed
        request: FastAPI request object for IP/UA
    """
    ip_address = request.client.host if request and request.client else None
    user_agent = request.headers.get("user-agent") if request else None
    
    AuditLogger.log_event(
        event_type=AuditEvent.PHI_ACCESSED,
        user_id=user_id,
        resource_type=resource_type,
        resource_id=resource_id,
        action=action,
        status="success",
        metadata={
            "fields_accessed": fields_accessed or [],
            "phi_fields_count": len([f for f in (fields_accessed or []) if is_phi_field(f)])
        },
        ip_address=ip_address,
        user_agent=user_agent
    )


def require_phi_access(
    resource_type: str,
    get_resource_id: Optional[Callable] = None,
    get_patient_id: Optional[Callable] = None
):
    """
    Decorator to require PHI access logging and access control
    
    Usage:
        @require_phi_access("patient_profile", get_resource_id=lambda r: r.path_params['patient_id'])
        async def get_patient_profile(...):
            ...
    
    Args:
        resource_type: Type of resource being accessed
        get_resource_id: Function to extract resource ID from request
        get_patient_id: Function to extract patient ID from request
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract request and current_user from kwargs
            request = None
            current_user = None
            
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                elif isinstance(arg, User):
                    current_user = arg
            
            for key, value in kwargs.items():
                if isinstance(value, Request):
                    request = value
                elif isinstance(value, User):
                    current_user = value
            
            if not current_user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication required"
                )
            
            # Extract resource/patient ID
            resource_id = None
            patient_id = None
            
            if get_resource_id and request:
                try:
                    resource_id = get_resource_id(request)
                except Exception:
                    pass
            
            if get_patient_id and request:
                try:
                    patient_id = get_patient_id(request)
                except Exception:
                    pass
            
            # Log PHI access
            log_phi_access(
                user_id=current_user.id,
                resource_type=resource_type,
                resource_id=resource_id or patient_id,
                action="read",
                request=request
            )
            
            # Call original function
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator


class PHIEncryptionWrapper:
    """
    Wrapper for encrypting/decrypting PHI fields in database operations
    """
    
    def __init__(self):
        self.kms_service = get_kms_service()
    
    def encrypt_phi_value(
        self,
        value: Any,
        patient_id: str,
        field_name: str,
        user_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Encrypt a PHI value before storage
        
        Args:
            value: Value to encrypt
            patient_id: Patient ID for encryption context
            field_name: Name of field
            user_id: User performing encryption
            
        Returns:
            Encrypted value dictionary or None if value is None/empty
        """
        if value is None or (isinstance(value, str) and not value.strip()):
            return None
        
        if not is_phi_field(field_name):
            # Not PHI, return as-is
            return value
        
        try:
            encrypted = self.kms_service.encrypt_phi_field(
                value=str(value),
                patient_id=patient_id,
                field_name=field_name,
                user_id=user_id
            )
            return encrypted
        except Exception as e:
            logger.error(f"Failed to encrypt PHI field {field_name}: {e}", exc_info=True)
            raise
    
    def decrypt_phi_value(
        self,
        encrypted_value: Any,
        patient_id: str,
        field_name: str,
        user_id: str
    ) -> Optional[str]:
        """
        Decrypt a PHI value after retrieval
        
        Args:
            encrypted_value: Encrypted value dictionary
            patient_id: Patient ID for decryption context
            field_name: Name of field
            user_id: User performing decryption
            
        Returns:
            Decrypted value or None
        """
        if encrypted_value is None:
            return None
        
        if not isinstance(encrypted_value, dict):
            # Not encrypted, return as-is
            return encrypted_value
        
        if not is_phi_field(field_name):
            # Not PHI, return as-is
            return encrypted_value
        
        try:
            decrypted = self.kms_service.decrypt_phi_field(
                encrypted_value=encrypted_value,
                patient_id=patient_id,
                field_name=field_name,
                user_id=user_id
            )
            return decrypted
        except Exception as e:
            logger.error(f"Failed to decrypt PHI field {field_name}: {e}", exc_info=True)
            raise


# Global PHI encryption wrapper instance
_phi_wrapper: Optional[PHIEncryptionWrapper] = None


def get_phi_wrapper() -> PHIEncryptionWrapper:
    """Get or create PHI encryption wrapper instance"""
    global _phi_wrapper
    if _phi_wrapper is None:
        _phi_wrapper = PHIEncryptionWrapper()
    return _phi_wrapper


def check_phi_access(
    current_user: User,
    patient_id: str,
    resource_type: str = "patient_data"
) -> bool:
    """
    Check if user has access to patient's PHI
    
    HIPAA Rules:
    - Patients can access their own PHI
    - Doctors can access PHI of their patients
    - System admins can access all PHI
    
    Args:
        current_user: Current authenticated user
        patient_id: ID of patient whose PHI is being accessed
        resource_type: Type of resource
        
    Returns:
        True if access is allowed
        
    Raises:
        HTTPException: 403 if access denied
    """
    user_role = str(current_user.role) if current_user.role else ""
    user_id = current_user.id
    
    # Patient accessing their own data
    if user_role == "patient" and user_id == patient_id:
        return True
    
    # Doctor accessing patient data
    if user_role == "doctor":
        # TODO: Check if doctor has relationship with patient
        # For now, allow all doctors (should be restricted in production)
        logger.warning(f"Doctor {user_id} accessing patient {patient_id} PHI - relationship check not implemented")
        return True
    
    # Admin access
    if user_role == "admin":
        return True
    
    # Access denied
    logger.warning(f"Access denied: user {user_id} (role: {user_role}) attempted to access patient {patient_id} PHI")
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Access denied: insufficient permissions to access this patient's data"
    )
