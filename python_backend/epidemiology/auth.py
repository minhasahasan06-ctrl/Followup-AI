"""
Epidemiology Authentication & Authorization Module
===================================================
Provides role-based access control for epidemiology endpoints.

PRODUCTION NOTES:
- This module should integrate with the main application's auth service
- JWT verification should be replaced with actual token validation
- Role permissions should be loaded from database/config
"""

import os
from typing import Optional, Dict, Any, List
from fastapi import HTTPException, Header
from enum import Enum

class Role(str, Enum):
    ADMIN = "admin"
    RESEARCHER = "researcher"
    DOCTOR = "doctor"
    PATIENT = "patient"
    VIEWER = "viewer"

ROLE_PERMISSIONS = {
    Role.ADMIN: {
        "can_view_all_signals": True,
        "can_view_all_locations": True,
        "can_run_scans": True,
        "can_view_ml_features": True,
        "can_modify_data": True,
        "min_cell_size": 1,
    },
    Role.RESEARCHER: {
        "can_view_all_signals": True,
        "can_view_all_locations": True,
        "can_run_scans": True,
        "can_view_ml_features": True,
        "can_modify_data": False,
        "min_cell_size": 10,
    },
    Role.DOCTOR: {
        "can_view_all_signals": False,
        "can_view_all_locations": False,
        "can_run_scans": False,
        "can_view_ml_features": False,
        "can_modify_data": False,
        "min_cell_size": 10,
    },
    Role.VIEWER: {
        "can_view_all_signals": False,
        "can_view_all_locations": False,
        "can_run_scans": False,
        "can_view_ml_features": False,
        "can_modify_data": False,
        "min_cell_size": 50,
    },
}


class AuthenticatedUser:
    """Authenticated user context for epidemiology endpoints."""
    
    def __init__(
        self,
        user_id: str,
        role: Role,
        location_ids: Optional[List[str]] = None,
        patient_ids: Optional[List[str]] = None
    ):
        self.user_id = user_id
        self.role = role
        self.location_ids = location_ids or []
        self.patient_ids = patient_ids or []
        self.permissions = ROLE_PERMISSIONS.get(role, ROLE_PERMISSIONS[Role.VIEWER])
    
    def can_access_location(self, location_id: Optional[str]) -> bool:
        """Check if user can access data for a specific location."""
        if self.permissions.get("can_view_all_locations"):
            return True
        if not location_id:
            return True
        return location_id in self.location_ids
    
    def can_view_signal(self, n_patients: int) -> bool:
        """Check if user can view a signal based on cell size."""
        min_size = self.permissions.get("min_cell_size", 10)
        return n_patients >= min_size
    
    def get_scope_filter(self, scope: str) -> Optional[List[str]]:
        """Get patient ID filter based on scope."""
        if scope == "my_patients":
            return self.patient_ids
        if scope == "all" and self.permissions.get("can_view_all_signals"):
            return None
        return self.patient_ids if self.patient_ids else []
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "role": self.role.value,
            "location_ids": self.location_ids,
            "permissions": self.permissions
        }


async def verify_epidemiology_auth(
    authorization: Optional[str] = Header(None)
) -> AuthenticatedUser:
    """
    Verify authentication and return user context.
    
    PRODUCTION TODO: 
    - Replace with actual JWT verification
    - Load user roles from database
    - Load user's location/patient access from database
    """
    if not authorization:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header")
    
    return AuthenticatedUser(
        user_id="authenticated_user",
        role=Role.RESEARCHER,
        location_ids=None,
        patient_ids=None
    )


def require_permission(permission: str):
    """Decorator factory for permission-based access control."""
    async def check_permission(user: AuthenticatedUser) -> AuthenticatedUser:
        if not user.permissions.get(permission, False):
            raise HTTPException(
                status_code=403,
                detail=f"Permission denied: {permission} required"
            )
        return user
    return check_permission
