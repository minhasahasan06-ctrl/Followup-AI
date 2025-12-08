"""
Followup Autopilot ML Engine

This module provides the ML-powered autopilot system for patient follow-ups:
- Risk prediction models (LSTM/Transformer)
- Adherence forecasting (XGBoost)
- Anomaly detection (IsolationForest)
- Engagement optimization (XGBoost)

HIPAA Compliance:
- All patient data is encrypted at rest and in transit
- Access requires valid consent verification
- All operations are audit logged

Wellness Positioning:
- Outputs are for wellness monitoring only
- NOT medical diagnosis or treatment recommendations
"""

from .signal_ingestor import SignalIngestor
from .feature_builder import FeatureBuilder
from .autopilot_core import AutopilotCore
from .trigger_engine import TriggerEngine
from .task_engine import TaskEngine
from .notification_engine import NotificationEngine

__all__ = [
    "SignalIngestor",
    "FeatureBuilder", 
    "AutopilotCore",
    "TriggerEngine",
    "TaskEngine",
    "NotificationEngine",
]
