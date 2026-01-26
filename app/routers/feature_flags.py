"""
Feature Flags API Router

REST API for feature flag management.
Supports viewing and toggling feature flags at runtime.

Access Control:
- Read: All authenticated users
- Write: Admin users only
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List

from app.dependencies import get_current_user
from app.models.user import User
from app.services.feature_flag_service import (
    get_feature_flag_service,
    is_feature_enabled
)

router = APIRouter(prefix="/api/feature-flags", tags=["feature-flags"])


class SetFlagRequest(BaseModel):
    """Request to set a feature flag"""
    enabled: bool
    persist: bool = Field(default=False, description="Persist to JSON config")


class UserOverrideRequest(BaseModel):
    """Request to set user-specific override"""
    user_id: str
    enabled: bool


class RoleOverrideRequest(BaseModel):
    """Request to set role-based override"""
    role: str
    enabled: bool


class FlagCheckRequest(BaseModel):
    """Request to check flag with context"""
    flag_name: str
    context: Optional[Dict[str, Any]] = None


@router.get("")
async def get_all_flags(
    current_user: User = Depends(get_current_user)
):
    """
    Get all feature flags with current status.
    
    Returns flag status considering user-specific and role-based overrides.
    """
    service = get_feature_flag_service()
    user_role = getattr(current_user, 'role', 'patient')
    
    flags = service.get_all_flags(
        user_id=str(current_user.id),
        user_role=user_role
    )
    
    return {
        "flags": flags,
        "user_id": str(current_user.id),
        "user_role": user_role
    }


@router.get("/{flag_name}")
async def get_flag(
    flag_name: str,
    current_user: User = Depends(get_current_user)
):
    """Get details for a specific feature flag"""
    service = get_feature_flag_service()
    
    details = service.get_flag_details(flag_name)
    if not details:
        raise HTTPException(status_code=404, detail=f"Flag not found: {flag_name}")
    
    user_role = getattr(current_user, 'role', 'patient')
    effective_value = service.is_enabled(
        flag_name,
        user_id=str(current_user.id),
        user_role=user_role
    )
    
    return {
        "flag": details,
        "effective_value": effective_value
    }


@router.post("/{flag_name}/check")
async def check_flag_with_context(
    flag_name: str,
    request: FlagCheckRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Check if a flag is enabled with optional context.
    
    Context can include triggers like {"escalation_active": true}
    """
    service = get_feature_flag_service()
    user_role = getattr(current_user, 'role', 'patient')
    
    enabled = service.is_enabled(
        flag_name,
        user_id=str(current_user.id),
        user_role=user_role,
        context=request.context
    )
    
    return {
        "flag_name": flag_name,
        "enabled": enabled,
        "context_applied": request.context is not None
    }


@router.put("/{flag_name}")
async def set_flag(
    flag_name: str,
    request: SetFlagRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Set a feature flag value.
    
    Admin only. Optionally persists to JSON config.
    """
    user_role = getattr(current_user, 'role', 'patient')
    if user_role not in ['admin', 'doctor']:
        raise HTTPException(status_code=403, detail="Admin or doctor access required")
    
    service = get_feature_flag_service()
    
    success = service.set_flag(
        flag_name,
        enabled=request.enabled,
        persist=request.persist
    )
    
    if not success:
        raise HTTPException(status_code=400, detail="Failed to set flag")
    
    return {
        "success": True,
        "flag_name": flag_name,
        "enabled": request.enabled,
        "persisted": request.persist
    }


@router.post("/{flag_name}/user-override")
async def set_user_override(
    flag_name: str,
    request: UserOverrideRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Set a user-specific flag override.
    
    Admin only.
    """
    user_role = getattr(current_user, 'role', 'patient')
    if user_role != 'admin':
        raise HTTPException(status_code=403, detail="Admin access required")
    
    service = get_feature_flag_service()
    
    success = service.set_user_override(
        flag_name,
        user_id=request.user_id,
        enabled=request.enabled
    )
    
    if not success:
        raise HTTPException(status_code=400, detail="Failed to set user override")
    
    return {
        "success": True,
        "flag_name": flag_name,
        "user_id": request.user_id,
        "enabled": request.enabled
    }


@router.post("/{flag_name}/role-override")
async def set_role_override(
    flag_name: str,
    request: RoleOverrideRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Set a role-based flag override.
    
    Admin only.
    """
    user_role = getattr(current_user, 'role', 'patient')
    if user_role != 'admin':
        raise HTTPException(status_code=403, detail="Admin access required")
    
    service = get_feature_flag_service()
    
    success = service.set_role_override(
        flag_name,
        role=request.role,
        enabled=request.enabled
    )
    
    if not success:
        raise HTTPException(status_code=400, detail="Failed to set role override")
    
    return {
        "success": True,
        "flag_name": flag_name,
        "role": request.role,
        "enabled": request.enabled
    }


@router.delete("/{flag_name}/runtime-override")
async def clear_runtime_override(
    flag_name: str,
    current_user: User = Depends(get_current_user)
):
    """
    Clear a runtime override, reverting to config value.
    
    Admin only.
    """
    user_role = getattr(current_user, 'role', 'patient')
    if user_role != 'admin':
        raise HTTPException(status_code=403, detail="Admin access required")
    
    service = get_feature_flag_service()
    service.clear_runtime_override(flag_name)
    
    return {
        "success": True,
        "flag_name": flag_name,
        "message": "Runtime override cleared"
    }


@router.post("/reload")
async def reload_flags(
    current_user: User = Depends(get_current_user)
):
    """
    Reload all flags from config sources.
    
    Admin only.
    """
    user_role = getattr(current_user, 'role', 'patient')
    if user_role != 'admin':
        raise HTTPException(status_code=403, detail="Admin access required")
    
    service = get_feature_flag_service()
    service.reload()
    
    return {
        "success": True,
        "message": "Feature flags reloaded"
    }
