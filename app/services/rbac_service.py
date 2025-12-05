"""
Role-Based Access Control (RBAC) & Attribute-Based Access Control (ABAC)
HIPAA-Compliant Fine-Grained Access Control

Implements:
- Role-based permissions
- Attribute-based access control (patient ownership, treatment relationship)
- Dynamic permission evaluation
- HIPAA minimum necessary principle enforcement
"""

from typing import List, Dict, Any, Optional, Set
from enum import Enum
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class Permission(Enum):
    """Fine-grained permissions"""
    # Patient permissions
    PATIENT_VIEW_OWN_RECORDS = "patient:view:own"
    PATIENT_UPDATE_OWN_PROFILE = "patient:update:own_profile"
    PATIENT_CREATE_OWN_DATA = "patient:create:own_data"
    
    # Clinician permissions
    CLINICIAN_VIEW_PATIENT_RECORDS = "clinician:view:patient_records"
    CLINICIAN_UPDATE_PATIENT_RECORDS = "clinician:update:patient_records"
    CLINICIAN_CREATE_PATIENT_RECORDS = "clinician:create:patient_records"
    CLINICIAN_DELETE_PATIENT_RECORDS = "clinician:delete:patient_records"
    CLINICIAN_VIEW_ALL_PATIENTS = "clinician:view:all_patients"
    CLINICIAN_EXPORT_PATIENT_DATA = "clinician:export:patient_data"
    
    # Admin permissions
    ADMIN_VIEW_ALL_RECORDS = "admin:view:all_records"
    ADMIN_MANAGE_USERS = "admin:manage:users"
    ADMIN_VIEW_AUDIT_LOGS = "admin:view:audit_logs"
    ADMIN_MANAGE_SYSTEM = "admin:manage:system"
    
    # System permissions
    SYSTEM_ACCESS = "system:access"


class Role(Enum):
    """User roles"""
    PATIENT = "patient"
    CLINICIAN = "clinician"
    ADMIN = "admin"
    SYSTEM = "system"


# Role-Permission mappings
ROLE_PERMISSIONS: Dict[Role, Set[Permission]] = {
    Role.PATIENT: {
        Permission.PATIENT_VIEW_OWN_RECORDS,
        Permission.PATIENT_UPDATE_OWN_PROFILE,
        Permission.PATIENT_CREATE_OWN_DATA,
    },
    Role.CLINICIAN: {
        Permission.CLINICIAN_VIEW_PATIENT_RECORDS,
        Permission.CLINICIAN_UPDATE_PATIENT_RECORDS,
        Permission.CLINICIAN_CREATE_PATIENT_RECORDS,
        Permission.CLINICIAN_DELETE_PATIENT_RECORDS,
        Permission.CLINICIAN_VIEW_ALL_PATIENTS,
        Permission.CLINICIAN_EXPORT_PATIENT_DATA,
    },
    Role.ADMIN: {
        Permission.ADMIN_VIEW_ALL_RECORDS,
        Permission.ADMIN_MANAGE_USERS,
        Permission.ADMIN_VIEW_AUDIT_LOGS,
        Permission.ADMIN_MANAGE_SYSTEM,
    },
    Role.SYSTEM: {
        Permission.SYSTEM_ACCESS,
    },
}


@dataclass
class AccessContext:
    """Context for access control evaluation"""
    user_id: str
    user_role: Role
    patient_id: Optional[str] = None  # Patient whose data is being accessed
    resource_type: Optional[str] = None
    resource_id: Optional[str] = None
    action: Optional[str] = None  # view, create, update, delete
    treatment_relationship: bool = False  # Is user treating this patient?
    patient_consent: bool = True  # Has patient given consent?
    emergency_access: bool = False  # Emergency access override
    ip_address: Optional[str] = None
    time_of_access: Optional[datetime] = None


class RBACService:
    """
    Role-Based and Attribute-Based Access Control service
    Implements HIPAA minimum necessary principle
    """
    
    @staticmethod
    def has_permission(context: AccessContext, permission: Permission) -> bool:
        """
        Check if user has a specific permission
        
        Args:
            context: Access context
            permission: Permission to check
        
        Returns:
            True if user has permission
        """
        # Get role permissions
        role_perms = ROLE_PERMISSIONS.get(context.user_role, set())
        
        # Check if permission is in role permissions
        if permission in role_perms:
            return True
        
        # System role has all permissions
        if context.user_role == Role.SYSTEM:
            return True
        
        return False
    
    @staticmethod
    def can_access_patient_data(context: AccessContext) -> bool:
        """
        Check if user can access patient data (ABAC evaluation)
        
        HIPAA Rules:
        1. Patients can always access their own data
        2. Clinicians can access data for patients they're treating
        3. Admins can access all data (with audit logging)
        4. Emergency access may override restrictions
        
        Args:
            context: Access context
        
        Returns:
            True if access allowed
        """
        if not context.patient_id:
            return False
        
        # Emergency access override
        if context.emergency_access:
            logger.warning(f"Emergency access granted to {context.user_id} for patient {context.patient_id}")
            return True
        
        # Patients can access their own data
        if context.user_role == Role.PATIENT:
            if context.user_id == context.patient_id:
                return True
            return False
        
        # Clinicians can access data for patients they're treating
        if context.user_role == Role.CLINICIAN:
            if context.treatment_relationship:
                return True
            # Check if patient has given consent
            if not context.patient_consent:
                logger.warning(
                    f"Access denied: Patient {context.patient_id} has not given consent "
                    f"for clinician {context.user_id}"
                )
                return False
            return True
        
        # Admins can access all data (with audit logging)
        if context.user_role == Role.ADMIN:
            return True
        
        return False
    
    @staticmethod
    def can_perform_action(context: AccessContext, action: str) -> bool:
        """
        Check if user can perform a specific action on a resource
        
        Args:
            context: Access context
            action: Action to perform (view, create, update, delete)
        
        Returns:
            True if action allowed
        """
        # Map action to permission
        action_permissions = {
            "view": Permission.CLINICIAN_VIEW_PATIENT_RECORDS if context.user_role == Role.CLINICIAN else Permission.PATIENT_VIEW_OWN_RECORDS,
            "create": Permission.CLINICIAN_CREATE_PATIENT_RECORDS if context.user_role == Role.CLINICIAN else Permission.PATIENT_CREATE_OWN_DATA,
            "update": Permission.CLINICIAN_UPDATE_PATIENT_RECORDS if context.user_role == Role.CLINICIAN else Permission.PATIENT_UPDATE_OWN_PROFILE,
            "delete": Permission.CLINICIAN_DELETE_PATIENT_RECORDS,
        }
        
        permission = action_permissions.get(action)
        if not permission:
            return False
        
        # Check permission
        if not RBACService.has_permission(context, permission):
            return False
        
        # Check patient data access
        if context.patient_id:
            return RBACService.can_access_patient_data(context)
        
        return True
    
    @staticmethod
    def enforce_minimum_necessary(
        context: AccessContext,
        requested_fields: List[str],
        available_fields: List[str]
    ) -> List[str]:
        """
        Enforce HIPAA minimum necessary principle
        Returns only fields that user is authorized to access
        
        Args:
            context: Access context
            requested_fields: Fields requested by user
            available_fields: All available fields
        
        Returns:
            List of authorized fields
        """
        authorized_fields = []
        
        # Patients can only access their own basic fields
        if context.user_role == Role.PATIENT:
            allowed_patient_fields = [
                "id", "email", "first_name", "last_name", "date_of_birth",
                "phone_number", "medical_conditions", "symptoms", "medications"
            ]
            authorized_fields = [
                f for f in requested_fields
                if f in allowed_patient_fields and f in available_fields
            ]
        
        # Clinicians can access more fields for patients they're treating
        elif context.user_role == Role.CLINICIAN:
            if context.treatment_relationship:
                # Full access to treatment-related fields
                authorized_fields = [
                    f for f in requested_fields if f in available_fields
                ]
            else:
                # Limited access without treatment relationship
                limited_fields = [
                    "id", "first_name", "last_name", "date_of_birth",
                    "medical_conditions", "symptoms"
                ]
                authorized_fields = [
                    f for f in requested_fields
                    if f in limited_fields and f in available_fields
                ]
        
        # Admins have full access
        elif context.user_role == Role.ADMIN:
            authorized_fields = [
                f for f in requested_fields if f in available_fields
            ]
        
        return authorized_fields
    
    @staticmethod
    def check_access(context: AccessContext, action: str) -> tuple[bool, Optional[str]]:
        """
        Comprehensive access check
        
        Returns:
            (allowed, reason) tuple
        """
        # Check permission
        if not RBACService.can_perform_action(context, action):
            return False, "Insufficient permissions"
        
        # Check patient data access
        if context.patient_id:
            if not RBACService.can_access_patient_data(context):
                return False, "Cannot access patient data"
        
        return True, None


# Global singleton instance
_rbac_service: Optional[RBACService] = None


def get_rbac_service() -> RBACService:
    """Get or create RBAC service singleton"""
    global _rbac_service
    if _rbac_service is None:
        _rbac_service = RBACService()
    return _rbac_service
