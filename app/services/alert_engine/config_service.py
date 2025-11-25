"""
Alert Engine Configuration Service - Admin-configurable thresholds and policies.

Provides centralized configuration for:
- Baseline window lengths
- Z-score thresholds for Yellow/Red
- DPI bucket thresholds
- Alert suppression windows
- Rate limits per patient
- Escalation timing
- Organ group weights
"""

import os
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import timedelta
import json

logger = logging.getLogger(__name__)


@dataclass
class OrganGroupConfig:
    """Configuration for organ-level scoring"""
    name: str
    metrics: list
    weight: float = 1.0
    yellow_threshold: float = 50.0
    red_threshold: float = 75.0


@dataclass
class AlertEngineConfig:
    """Complete Alert Engine configuration"""
    
    # Baseline computation settings
    baseline_window_days: int = 14
    min_baseline_days: int = 7
    ewma_alpha: float = 0.3
    epsilon: float = 1e-6
    
    # Z-score thresholds
    z_yellow_threshold: float = 2.0
    z_red_threshold: float = 3.0
    
    # DPI bucket thresholds (0-100 scale)
    dpi_green_max: float = 25.0
    dpi_yellow_max: float = 50.0
    dpi_orange_max: float = 75.0
    
    # Alert suppression settings
    suppression_window_hours: int = 6
    max_alerts_per_patient_per_day: int = 4
    ack_snooze_minutes: int = 120
    
    # Escalation settings
    escalation_timeout_critical_hours: float = 2.0
    escalation_timeout_high_hours: float = 4.0
    escalation_timeout_moderate_hours: float = 8.0
    
    # Notification settings
    sms_enabled: bool = True
    email_enabled: bool = True
    push_enabled: bool = True
    dashboard_enabled: bool = True
    
    # ML ranking settings
    ml_ranking_enabled: bool = True
    ml_model_version: str = "v1"
    
    # Quality gating
    min_confidence_threshold: float = 0.4
    max_capture_age_hours: int = 48
    corroboration_required_for_high_severity: bool = True
    min_corroborating_signals: int = 2
    
    # Respiratory spike absolute rules
    respiratory_rate_absolute_max: int = 30
    respiratory_confidence_min: float = 0.8
    
    # Jump detection
    dpi_jump_threshold_24h: float = 8.0
    
    # Organ group configurations
    organ_groups: Dict[str, OrganGroupConfig] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize default organ groups"""
        if not self.organ_groups:
            self.organ_groups = {
                "respiratory": OrganGroupConfig(
                    name="Respiratory",
                    metrics=["respiratory_rate", "thoracoabdominal_async", "cough_freq", 
                             "wheeze_prob", "audio_rr", "breathlessness_level", "breath_rate"],
                    weight=1.2,
                    yellow_threshold=50.0,
                    red_threshold=75.0
                ),
                "cardio_fluid": OrganGroupConfig(
                    name="Cardio/Fluid",
                    metrics=["edema_pct", "weight_delta", "perfusion_index", 
                             "facial_puffiness", "swelling_score"],
                    weight=1.1,
                    yellow_threshold=50.0,
                    red_threshold=75.0
                ),
                "hepatic_hematologic": OrganGroupConfig(
                    name="Hepatic/Hematologic",
                    metrics=["sclera_jaundice_index", "palmar_pallor_index", 
                             "conjunctiva_pallor", "sclera_yellowness", "lip_pallor"],
                    weight=1.0,
                    yellow_threshold=50.0,
                    red_threshold=75.0
                ),
                "mobility": OrganGroupConfig(
                    name="Mobility",
                    metrics=["gait_speed", "step_count_trend", "gait_variability",
                             "mobility_score", "tremor_amplitude", "movement_pattern"],
                    weight=0.9,
                    yellow_threshold=50.0,
                    red_threshold=75.0
                ),
                "cognitive_behavioral": OrganGroupConfig(
                    name="Cognitive/Behavioral",
                    metrics=["mood_sentiment", "reaction_time", "checkin_compliance",
                             "mood", "sleep_quality", "cognitive_score", "anxiety_score"],
                    weight=0.8,
                    yellow_threshold=50.0,
                    red_threshold=75.0
                )
            }
    
    def get_organ_weight(self, organ_name: str) -> float:
        """Get weight for specific organ group"""
        if organ_name in self.organ_groups:
            return self.organ_groups[organ_name].weight
        return 1.0
    
    def get_metric_organ_group(self, metric_name: str) -> Optional[str]:
        """Find which organ group a metric belongs to"""
        for group_name, config in self.organ_groups.items():
            if metric_name in config.metrics:
                return group_name
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for storage/API"""
        return {
            "baseline_window_days": self.baseline_window_days,
            "min_baseline_days": self.min_baseline_days,
            "z_yellow_threshold": self.z_yellow_threshold,
            "z_red_threshold": self.z_red_threshold,
            "dpi_green_max": self.dpi_green_max,
            "dpi_yellow_max": self.dpi_yellow_max,
            "dpi_orange_max": self.dpi_orange_max,
            "suppression_window_hours": self.suppression_window_hours,
            "max_alerts_per_patient_per_day": self.max_alerts_per_patient_per_day,
            "ack_snooze_minutes": self.ack_snooze_minutes,
            "escalation_timeout_critical_hours": self.escalation_timeout_critical_hours,
            "escalation_timeout_high_hours": self.escalation_timeout_high_hours,
            "escalation_timeout_moderate_hours": self.escalation_timeout_moderate_hours,
            "sms_enabled": self.sms_enabled,
            "email_enabled": self.email_enabled,
            "push_enabled": self.push_enabled,
            "ml_ranking_enabled": self.ml_ranking_enabled,
            "min_confidence_threshold": self.min_confidence_threshold,
            "corroboration_required_for_high_severity": self.corroboration_required_for_high_severity,
            "respiratory_rate_absolute_max": self.respiratory_rate_absolute_max,
            "dpi_jump_threshold_24h": self.dpi_jump_threshold_24h,
            "organ_groups": {
                name: {
                    "name": cfg.name,
                    "metrics": cfg.metrics,
                    "weight": cfg.weight,
                    "yellow_threshold": cfg.yellow_threshold,
                    "red_threshold": cfg.red_threshold
                }
                for name, cfg in self.organ_groups.items()
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AlertEngineConfig':
        """Create config from dictionary"""
        organ_groups = {}
        if "organ_groups" in data:
            for name, cfg_data in data["organ_groups"].items():
                organ_groups[name] = OrganGroupConfig(**cfg_data)
            del data["organ_groups"]
        
        config = cls(**data)
        if organ_groups:
            config.organ_groups = organ_groups
        return config


class AlertConfigService:
    """Service for managing Alert Engine configuration"""
    
    _instance: Optional['AlertConfigService'] = None
    _config: AlertEngineConfig = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._config = AlertEngineConfig()
        return cls._instance
    
    @property
    def config(self) -> AlertEngineConfig:
        """Get current configuration"""
        return self._config
    
    def update_config(self, updates: Dict[str, Any]) -> AlertEngineConfig:
        """Update configuration with new values"""
        try:
            current_dict = self._config.to_dict()
            current_dict.update(updates)
            self._config = AlertEngineConfig.from_dict(current_dict)
            logger.info(f"Alert Engine config updated: {list(updates.keys())}")
            return self._config
        except Exception as e:
            logger.error(f"Error updating config: {e}")
            raise
    
    def reset_to_defaults(self) -> AlertEngineConfig:
        """Reset to default configuration"""
        self._config = AlertEngineConfig()
        logger.info("Alert Engine config reset to defaults")
        return self._config
    
    def get_escalation_timeout(self, severity: str) -> timedelta:
        """Get escalation timeout based on severity"""
        if severity == "critical":
            return timedelta(hours=self._config.escalation_timeout_critical_hours)
        elif severity == "high":
            return timedelta(hours=self._config.escalation_timeout_high_hours)
        else:
            return timedelta(hours=self._config.escalation_timeout_moderate_hours)
    
    def get_dpi_bucket(self, dpi_score: float) -> str:
        """Get DPI color bucket from score"""
        if dpi_score < self._config.dpi_green_max:
            return "green"
        elif dpi_score < self._config.dpi_yellow_max:
            return "yellow"
        elif dpi_score < self._config.dpi_orange_max:
            return "orange"
        else:
            return "red"
    
    def get_z_severity(self, z_score: float) -> str:
        """Get severity level from z-score"""
        abs_z = abs(z_score)
        if abs_z >= self._config.z_red_threshold:
            return "critical"
        elif abs_z >= self._config.z_yellow_threshold:
            return "elevated"
        else:
            return "normal"
