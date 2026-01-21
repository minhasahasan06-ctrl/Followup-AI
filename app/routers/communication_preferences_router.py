"""
Communication Preferences API Router
=====================================

API endpoints for managing communication preferences
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from datetime import time

from app.services.communication_preferences_service import (
    get_preferences_service,
    CommunicationMethod,
    PreferenceLevel,
    TimeSlot,
)

router = APIRouter(prefix="/api/communication-preferences", tags=["Communication Preferences"])


class MethodPreferenceRequest(BaseModel):
    method: str
    level: str
    is_default: bool = False
    require_scheduling: bool = False
    max_duration_minutes: Optional[int] = None
    notes: str = ""


class AvailabilityRequest(BaseModel):
    day: str
    slots: List[str]
    is_available: bool = True
    start_time: Optional[str] = None
    end_time: Optional[str] = None


class DNDRequest(BaseModel):
    enabled: bool
    start: Optional[str] = None
    end: Optional[str] = None
    emergency_override: bool = True


@router.get("/{user_id}")
async def get_preferences(user_id: str, user_role: str = "patient"):
    """Get communication preferences for a user"""
    service = get_preferences_service()
    prefs = service.get_or_create_preferences(user_id, user_role)
    
    return {
        "user_id": prefs.user_id,
        "user_role": prefs.user_role,
        "timezone": prefs.timezone,
        "do_not_disturb": prefs.do_not_disturb,
        "emergency_override": prefs.emergency_override,
        "preferences": {
            method.value: {
                "level": pref.level.value,
                "is_default": pref.is_default,
                "require_scheduling": pref.require_scheduling,
                "max_duration_minutes": pref.max_duration_minutes,
            }
            for method, pref in prefs.preferences.items()
        },
        "weekly_availability": {
            day: {
                "slots": [s.value for s in avail.slots],
                "is_available": avail.is_available,
            }
            for day, avail in prefs.weekly_availability.items()
        },
    }


@router.put("/{user_id}/method")
async def update_method_preference(user_id: str, request: MethodPreferenceRequest):
    """Update preference for a communication method"""
    service = get_preferences_service()
    
    try:
        method = CommunicationMethod(request.method)
        level = PreferenceLevel(request.level)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    prefs = service.update_method_preference(
        user_id=user_id,
        method=method,
        level=level,
        is_default=request.is_default,
        require_scheduling=request.require_scheduling,
        max_duration_minutes=request.max_duration_minutes,
        notes=request.notes,
    )
    
    if not prefs:
        raise HTTPException(status_code=404, detail="Preferences not found")
    
    return {"success": True, "message": f"Updated {request.method} preference"}


@router.put("/{user_id}/availability")
async def update_availability(user_id: str, request: AvailabilityRequest):
    """Update availability for a specific day"""
    service = get_preferences_service()
    
    slots = [TimeSlot(s) for s in request.slots if s in [ts.value for ts in TimeSlot]]
    
    prefs = service.update_availability(
        user_id=user_id,
        day=request.day,
        slots=slots,
        is_available=request.is_available,
    )
    
    if not prefs:
        raise HTTPException(status_code=404, detail="Preferences not found")
    
    return {"success": True, "message": f"Updated {request.day} availability"}


@router.put("/{user_id}/dnd")
async def set_do_not_disturb(user_id: str, request: DNDRequest):
    """Set Do Not Disturb mode"""
    service = get_preferences_service()
    
    prefs = service.set_do_not_disturb(
        user_id=user_id,
        enabled=request.enabled,
        emergency_override=request.emergency_override,
    )
    
    if not prefs:
        raise HTTPException(status_code=404, detail="Preferences not found")
    
    return {"success": True, "dnd_enabled": request.enabled}


@router.get("/{user_id}/can-contact")
async def check_can_contact(
    user_id: str,
    method: str = "chat",
    is_emergency: bool = False,
):
    """Check if user can be contacted"""
    service = get_preferences_service()
    
    try:
        comm_method = CommunicationMethod(method)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid method: {method}")
    
    can_contact = service.can_contact(user_id, comm_method, is_emergency)
    
    return {
        "can_contact": can_contact,
        "method": method,
        "is_emergency": is_emergency,
    }


@router.get("/{user_id}/preferred-method")
async def get_preferred_method(user_id: str):
    """Get user's preferred communication method"""
    service = get_preferences_service()
    method = service.get_preferred_method(user_id)
    
    return {
        "user_id": user_id,
        "preferred_method": method.value if method else "chat",
    }


@router.get("/{user_id}/availability")
async def check_availability(user_id: str):
    """Check if user is currently available"""
    service = get_preferences_service()
    
    is_available = service.is_available_now(user_id)
    next_available = service.get_next_available_time(user_id)
    
    return {
        "user_id": user_id,
        "is_available_now": is_available,
        "next_available": next_available,
    }
