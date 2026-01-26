"""
Communication Preferences Service
==================================

Manages patient and doctor communication method preferences
Supports voice, video, and chat preferences with scheduling
"""

from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime, time
from enum import Enum
import uuid
import logging

from app.services.access_control import HIPAAAuditLogger

logger = logging.getLogger(__name__)


class CommunicationMethod(str, Enum):
    """Available communication methods"""
    CHAT = "chat"
    VOICE = "voice"
    VIDEO = "video"
    SMS = "sms"
    EMAIL = "email"


class PreferenceLevel(str, Enum):
    """Preference level for a method"""
    PREFERRED = "preferred"
    ACCEPTABLE = "acceptable"
    EMERGENCY_ONLY = "emergency_only"
    DISABLED = "disabled"


class TimeSlot(str, Enum):
    """Time slots for availability"""
    MORNING = "morning"
    AFTERNOON = "afternoon"
    EVENING = "evening"
    NIGHT = "night"
    ANYTIME = "anytime"


@dataclass
class DailyAvailability:
    """Availability for a specific day"""
    day: str
    slots: List[TimeSlot] = field(default_factory=list)
    start_time: Optional[time] = None
    end_time: Optional[time] = None
    is_available: bool = True


@dataclass
class MethodPreference:
    """Preference for a specific communication method"""
    method: CommunicationMethod
    level: PreferenceLevel
    is_default: bool = False
    require_scheduling: bool = False
    max_duration_minutes: Optional[int] = None
    notes: str = ""


@dataclass
class CommunicationPreferences:
    """Complete communication preferences for a user"""
    user_id: str
    user_role: str
    preferences: Dict[CommunicationMethod, MethodPreference] = field(default_factory=dict)
    weekly_availability: Dict[str, DailyAvailability] = field(default_factory=dict)
    timezone: str = "UTC"
    do_not_disturb: bool = False
    dnd_start: Optional[time] = None
    dnd_end: Optional[time] = None
    emergency_override: bool = True
    updated_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


class CommunicationPreferencesService:
    """
    Production-grade communication preferences service
    
    Features:
    - Method-specific preferences (voice, video, chat)
    - Weekly availability scheduling
    - Do Not Disturb settings
    - Emergency override configuration
    """
    
    def __init__(self):
        self._preferences: Dict[str, CommunicationPreferences] = {}
    
    def get_default_preferences(
        self,
        user_id: str,
        user_role: str,
    ) -> CommunicationPreferences:
        """Create default preferences for a user"""
        is_doctor = user_role == "doctor"
        
        preferences = {
            CommunicationMethod.CHAT: MethodPreference(
                method=CommunicationMethod.CHAT,
                level=PreferenceLevel.PREFERRED,
                is_default=True,
            ),
            CommunicationMethod.VOICE: MethodPreference(
                method=CommunicationMethod.VOICE,
                level=PreferenceLevel.ACCEPTABLE,
                require_scheduling=is_doctor,
                max_duration_minutes=30 if is_doctor else None,
            ),
            CommunicationMethod.VIDEO: MethodPreference(
                method=CommunicationMethod.VIDEO,
                level=PreferenceLevel.ACCEPTABLE,
                require_scheduling=True,
                max_duration_minutes=60,
            ),
            CommunicationMethod.SMS: MethodPreference(
                method=CommunicationMethod.SMS,
                level=PreferenceLevel.ACCEPTABLE,
            ),
            CommunicationMethod.EMAIL: MethodPreference(
                method=CommunicationMethod.EMAIL,
                level=PreferenceLevel.ACCEPTABLE,
            ),
        }
        
        days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
        availability = {}
        for day in days:
            is_weekend = day in ["saturday", "sunday"]
            availability[day] = DailyAvailability(
                day=day,
                slots=[TimeSlot.MORNING, TimeSlot.AFTERNOON, TimeSlot.EVENING] if not is_weekend else [TimeSlot.AFTERNOON],
                is_available=not is_weekend if is_doctor else True,
            )
        
        return CommunicationPreferences(
            user_id=user_id,
            user_role=user_role,
            preferences=preferences,
            weekly_availability=availability,
        )
    
    def get_preferences(self, user_id: str) -> Optional[CommunicationPreferences]:
        """Get user preferences"""
        return self._preferences.get(user_id)
    
    def get_or_create_preferences(
        self,
        user_id: str,
        user_role: str,
    ) -> CommunicationPreferences:
        """Get existing preferences or create defaults"""
        if user_id not in self._preferences:
            self._preferences[user_id] = self.get_default_preferences(user_id, user_role)
        return self._preferences[user_id]
    
    def update_method_preference(
        self,
        user_id: str,
        method: CommunicationMethod,
        level: PreferenceLevel,
        is_default: bool = False,
        require_scheduling: bool = False,
        max_duration_minutes: Optional[int] = None,
        notes: str = "",
    ) -> Optional[CommunicationPreferences]:
        """
        Update preference for a specific method
        
        Args:
            user_id: User ID
            method: Communication method
            level: Preference level
            is_default: Set as default method
            require_scheduling: Require appointments
            max_duration_minutes: Max session duration
            notes: Additional notes
            
        Returns:
            Updated preferences
        """
        prefs = self._preferences.get(user_id)
        if not prefs:
            return None
        
        if is_default:
            for m in prefs.preferences.values():
                m.is_default = False
        
        prefs.preferences[method] = MethodPreference(
            method=method,
            level=level,
            is_default=is_default,
            require_scheduling=require_scheduling,
            max_duration_minutes=max_duration_minutes,
            notes=notes,
        )
        prefs.updated_at = datetime.utcnow()
        
        HIPAAAuditLogger.log_phi_access(
            actor_id=user_id,
            actor_role=prefs.user_role,
            patient_id=user_id,
            action="update_communication_preference",
            phi_categories=["preferences"],
            resource_type="preferences",
            resource_id=user_id,
            access_reason=f"Update {method.value} preference to {level.value}",
        )
        
        return prefs
    
    def update_availability(
        self,
        user_id: str,
        day: str,
        slots: List[TimeSlot],
        is_available: bool = True,
        start_time: Optional[time] = None,
        end_time: Optional[time] = None,
    ) -> Optional[CommunicationPreferences]:
        """Update availability for a specific day"""
        prefs = self._preferences.get(user_id)
        if not prefs:
            return None
        
        prefs.weekly_availability[day.lower()] = DailyAvailability(
            day=day.lower(),
            slots=slots,
            start_time=start_time,
            end_time=end_time,
            is_available=is_available,
        )
        prefs.updated_at = datetime.utcnow()
        
        return prefs
    
    def set_do_not_disturb(
        self,
        user_id: str,
        enabled: bool,
        start: Optional[time] = None,
        end: Optional[time] = None,
        emergency_override: bool = True,
    ) -> Optional[CommunicationPreferences]:
        """Set Do Not Disturb mode"""
        prefs = self._preferences.get(user_id)
        if not prefs:
            return None
        
        prefs.do_not_disturb = enabled
        prefs.dnd_start = start
        prefs.dnd_end = end
        prefs.emergency_override = emergency_override
        prefs.updated_at = datetime.utcnow()
        
        logger.info(f"DND {'enabled' if enabled else 'disabled'} for user {user_id}")
        return prefs
    
    def can_contact(
        self,
        user_id: str,
        method: CommunicationMethod,
        is_emergency: bool = False,
    ) -> bool:
        """
        Check if a user can be contacted via a method
        
        Args:
            user_id: User to contact
            method: Method to use
            is_emergency: Is this an emergency
            
        Returns:
            True if contact is allowed
        """
        prefs = self._preferences.get(user_id)
        if not prefs:
            return True
        
        if prefs.do_not_disturb:
            if is_emergency and prefs.emergency_override:
                return True
            return False
        
        method_pref = prefs.preferences.get(method)
        if not method_pref:
            return True
        
        if method_pref.level == PreferenceLevel.DISABLED:
            return is_emergency and prefs.emergency_override
        
        if method_pref.level == PreferenceLevel.EMERGENCY_ONLY:
            return is_emergency
        
        return True
    
    def get_preferred_method(
        self,
        user_id: str,
        available_methods: Optional[List[CommunicationMethod]] = None,
    ) -> Optional[CommunicationMethod]:
        """Get user's preferred available method"""
        prefs = self._preferences.get(user_id)
        if not prefs:
            return CommunicationMethod.CHAT
        
        for method_pref in prefs.preferences.values():
            if method_pref.is_default:
                if available_methods is None or method_pref.method in available_methods:
                    return method_pref.method
        
        for method_pref in sorted(
            prefs.preferences.values(),
            key=lambda p: 0 if p.level == PreferenceLevel.PREFERRED else 1,
        ):
            if method_pref.level in [PreferenceLevel.PREFERRED, PreferenceLevel.ACCEPTABLE]:
                if available_methods is None or method_pref.method in available_methods:
                    return method_pref.method
        
        return CommunicationMethod.CHAT
    
    def is_available_now(
        self,
        user_id: str,
    ) -> bool:
        """Check if user is currently available"""
        prefs = self._preferences.get(user_id)
        if not prefs:
            return True
        
        if prefs.do_not_disturb:
            return False
        
        now = datetime.utcnow()
        day_name = now.strftime("%A").lower()
        
        day_avail = prefs.weekly_availability.get(day_name)
        if not day_avail:
            return True
        
        return day_avail.is_available
    
    def get_next_available_time(
        self,
        user_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Get next available time slot"""
        prefs = self._preferences.get(user_id)
        if not prefs:
            return None
        
        days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
        now = datetime.utcnow()
        current_day_idx = now.weekday()
        
        for i in range(7):
            day_idx = (current_day_idx + i) % 7
            day_name = days[day_idx]
            day_avail = prefs.weekly_availability.get(day_name)
            
            if day_avail and day_avail.is_available and day_avail.slots:
                return {
                    "day": day_name,
                    "slots": [s.value for s in day_avail.slots],
                    "days_from_now": i,
                }
        
        return None


_prefs_service: Optional[CommunicationPreferencesService] = None


def get_preferences_service() -> CommunicationPreferencesService:
    """Get singleton preferences service instance"""
    global _prefs_service
    if _prefs_service is None:
        _prefs_service = CommunicationPreferencesService()
    return _prefs_service
