"""
Feature Flag Service

Production-grade feature flag management for Followup AI.
Supports runtime toggles, user-specific overrides, and JSON/ENV configuration.

Features:
- JSON/ENV-driven configuration
- Runtime toggle support
- User and role-based overrides
- Escalation-context-aware flags
- Admin API for flag management
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


class FeatureFlagType(str, Enum):
    """Types of feature flags"""
    BOOLEAN = "boolean"
    STRING = "string"
    NUMBER = "number"
    JSON = "json"


@dataclass
class FeatureFlag:
    """Individual feature flag definition"""
    name: str
    enabled: bool
    flag_type: FeatureFlagType = FeatureFlagType.BOOLEAN
    value: Any = None
    description: str = ""
    default_enabled: bool = False
    user_overrides: Dict[str, bool] = field(default_factory=dict)
    role_overrides: Dict[str, bool] = field(default_factory=dict)
    context_triggers: Dict[str, bool] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class FeatureFlagService:
    """
    Centralized feature flag management.
    
    Configuration sources (in order of priority):
    1. Runtime overrides (in-memory)
    2. User-specific overrides
    3. Role-based overrides
    4. Context triggers (e.g., escalation active)
    5. Environment variables
    6. JSON config file
    7. Default values
    """
    
    DEFAULT_FLAGS: Dict[str, Dict[str, Any]] = {
        "showClonaCallIcons": {
            "enabled": False,
            "description": "Show phone/video call icons in Clona chat interface",
            "context_triggers": {
                "escalation_active": True
            }
        },
        "enableVoice": {
            "enabled": True,
            "description": "Enable voice chat features for Clona and Lysa"
        },
        "enableMemory": {
            "enabled": True,
            "description": "Enable long-term memory persistence for agents"
        },
        "enablePersonaMarketplace": {
            "enabled": False,
            "description": "Enable persona marketplace for custom agent personalities"
        },
        "enableRealTimeEscalation": {
            "enabled": True,
            "description": "Enable real-time escalation flow with red flag detection"
        },
        "enableVoiceRecording": {
            "enabled": True,
            "description": "Allow voice conversation recording (requires consent)"
        },
        "enableAIAnalysis": {
            "enabled": True,
            "description": "Enable AI-powered symptom analysis during red flag detection"
        },
        "enableActionCards": {
            "enabled": True,
            "description": "Enable action cards for voice conversation actions"
        },
        "showEmergencyBanner": {
            "enabled": True,
            "description": "Show emergency banner when red flags detected"
        },
        "enableDoctorVideoCall": {
            "enabled": True,
            "description": "Enable video call functionality with connected doctors"
        },
        "enablePushToTalk": {
            "enabled": True,
            "description": "Enable push-to-talk mode for voice conversations"
        },
        "enableAlwaysListening": {
            "enabled": True,
            "description": "Enable always-listening mode for voice conversations"
        },
        "showMemoryEditor": {
            "enabled": True,
            "description": "Show memory editor UI for managing stored facts"
        },
        "enableConsentFlow": {
            "enabled": True,
            "description": "Require consent before recording or memory persistence"
        },
        "showPersonaSelector": {
            "enabled": True,
            "description": "Show persona selection in agent interface"
        },
        "enableStreamingTTS": {
            "enabled": True,
            "description": "Enable streaming text-to-speech for lower latency"
        },
        "enableInterruptionHandling": {
            "enabled": True,
            "description": "Allow user to interrupt agent during speech"
        }
    }
    
    _instance: Optional['FeatureFlagService'] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self.logger = logging.getLogger(__name__)
        self._flags: Dict[str, FeatureFlag] = {}
        self._runtime_overrides: Dict[str, bool] = {}
        self._config_file_path = Path("config/feature_flags.json")
        self._load_flags()
        self._initialized = True
    
    def _load_flags(self) -> None:
        """Load flags from all configuration sources"""
        for name, config in self.DEFAULT_FLAGS.items():
            self._flags[name] = FeatureFlag(
                name=name,
                enabled=config.get("enabled", False),
                description=config.get("description", ""),
                default_enabled=config.get("enabled", False),
                context_triggers=config.get("context_triggers", {})
            )
        
        self._load_from_json()
        
        self._load_from_env()
    
    def _load_from_json(self) -> None:
        """Load flags from JSON config file"""
        try:
            if self._config_file_path.exists():
                with open(self._config_file_path, 'r') as f:
                    config = json.load(f)
                    
                for name, value in config.get("flags", {}).items():
                    if name in self._flags:
                        if isinstance(value, bool):
                            self._flags[name].enabled = value
                        elif isinstance(value, dict):
                            self._flags[name].enabled = value.get("enabled", self._flags[name].enabled)
                            self._flags[name].user_overrides = value.get("user_overrides", {})
                            self._flags[name].role_overrides = value.get("role_overrides", {})
        except Exception as e:
            self.logger.warning(f"Failed to load feature flags from JSON: {e}")
    
    def _load_from_env(self) -> None:
        """Load flags from environment variables"""
        for name in self._flags:
            env_key = f"FEATURE_FLAG_{name.upper()}"
            env_value = os.getenv(env_key)
            
            if env_value is not None:
                self._flags[name].enabled = env_value.lower() in ('true', '1', 'yes', 'on')
    
    def is_enabled(
        self,
        flag_name: str,
        user_id: Optional[str] = None,
        user_role: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Check if a feature flag is enabled.
        
        Args:
            flag_name: Name of the feature flag
            user_id: Optional user ID for user-specific overrides
            user_role: Optional user role for role-based overrides
            context: Optional context for context-triggered flags
            
        Returns:
            True if flag is enabled for the given context
        """
        if flag_name in self._runtime_overrides:
            return self._runtime_overrides[flag_name]
        
        flag = self._flags.get(flag_name)
        if not flag:
            self.logger.warning(f"Unknown feature flag: {flag_name}")
            return False
        
        if user_id and user_id in flag.user_overrides:
            return flag.user_overrides[user_id]
        
        if user_role and user_role in flag.role_overrides:
            return flag.role_overrides[user_role]
        
        if context and flag.context_triggers:
            for trigger, value in flag.context_triggers.items():
                if context.get(trigger):
                    return value
        
        return flag.enabled
    
    def set_flag(
        self,
        flag_name: str,
        enabled: bool,
        persist: bool = False
    ) -> bool:
        """
        Set a feature flag value.
        
        Args:
            flag_name: Name of the feature flag
            enabled: New enabled state
            persist: Whether to persist to JSON config
            
        Returns:
            True if successful
        """
        self._runtime_overrides[flag_name] = enabled
        
        if flag_name in self._flags:
            self._flags[flag_name].enabled = enabled
            self._flags[flag_name].updated_at = datetime.now(timezone.utc)
        
        if persist:
            self._save_to_json()
        
        self.logger.info(f"Feature flag '{flag_name}' set to {enabled}")
        return True
    
    def set_user_override(
        self,
        flag_name: str,
        user_id: str,
        enabled: bool
    ) -> bool:
        """Set a user-specific override for a flag"""
        if flag_name not in self._flags:
            return False
            
        self._flags[flag_name].user_overrides[user_id] = enabled
        self._flags[flag_name].updated_at = datetime.now(timezone.utc)
        return True
    
    def set_role_override(
        self,
        flag_name: str,
        role: str,
        enabled: bool
    ) -> bool:
        """Set a role-based override for a flag"""
        if flag_name not in self._flags:
            return False
            
        self._flags[flag_name].role_overrides[role] = enabled
        self._flags[flag_name].updated_at = datetime.now(timezone.utc)
        return True
    
    def clear_runtime_override(self, flag_name: str) -> None:
        """Clear a runtime override"""
        if flag_name in self._runtime_overrides:
            del self._runtime_overrides[flag_name]
    
    def get_all_flags(
        self,
        user_id: Optional[str] = None,
        user_role: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """Get all flags with their current status"""
        result = {}
        
        for name, flag in self._flags.items():
            result[name] = {
                "enabled": self.is_enabled(name, user_id, user_role, context),
                "description": flag.description,
                "default_enabled": flag.default_enabled,
                "has_override": name in self._runtime_overrides,
                "updated_at": flag.updated_at.isoformat()
            }
        
        return result
    
    def get_flag_details(self, flag_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific flag"""
        flag = self._flags.get(flag_name)
        if not flag:
            return None
            
        return {
            "name": flag.name,
            "enabled": flag.enabled,
            "description": flag.description,
            "default_enabled": flag.default_enabled,
            "user_overrides": flag.user_overrides,
            "role_overrides": flag.role_overrides,
            "context_triggers": flag.context_triggers,
            "has_runtime_override": flag_name in self._runtime_overrides,
            "runtime_value": self._runtime_overrides.get(flag_name),
            "created_at": flag.created_at.isoformat(),
            "updated_at": flag.updated_at.isoformat()
        }
    
    def _save_to_json(self) -> None:
        """Save current flag state to JSON config"""
        try:
            self._config_file_path.parent.mkdir(parents=True, exist_ok=True)
            
            config = {
                "flags": {},
                "updated_at": datetime.now(timezone.utc).isoformat()
            }
            
            for name, flag in self._flags.items():
                config["flags"][name] = {
                    "enabled": flag.enabled,
                    "user_overrides": flag.user_overrides,
                    "role_overrides": flag.role_overrides
                }
            
            with open(self._config_file_path, 'w') as f:
                json.dump(config, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to save feature flags to JSON: {e}")
    
    def reload(self) -> None:
        """Reload flags from all sources"""
        self._runtime_overrides.clear()
        self._load_flags()
        self.logger.info("Feature flags reloaded")


_feature_flag_service: Optional[FeatureFlagService] = None


def get_feature_flag_service() -> FeatureFlagService:
    """Get the singleton feature flag service instance"""
    global _feature_flag_service
    if _feature_flag_service is None:
        _feature_flag_service = FeatureFlagService()
    return _feature_flag_service


def is_feature_enabled(
    flag_name: str,
    user_id: Optional[str] = None,
    user_role: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None
) -> bool:
    """Convenience function to check if a feature is enabled"""
    return get_feature_flag_service().is_enabled(flag_name, user_id, user_role, context)
