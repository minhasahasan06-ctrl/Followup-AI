"""
Alert Engine Service Package - Comprehensive health deterioration detection system.

Components:
1. MetricsIngestService - Real-time metric ingestion with Redis streaming
2. OrganScoringService - 5-organ system scoring (Respiratory, Cardio, Hepatic, Mobility, Cognitive)
3. DPIComputationService - Composite Deterioration Index with color buckets
4. RuleBasedAlertEngine - 7 clinical alert rules with deduplication
5. NotificationService - Multi-channel delivery (Dashboard, SMS, Email)
6. EscalationService - Unacknowledged alert escalation
7. MLRankingService - XGBoost-based alert prioritization
8. AlertConfigService - Admin-configurable thresholds and policies
9. AlertEngineWorker - Background worker that orchestrates the complete pipeline
10. AlertEngineCronJob - Cron-style periodic pipeline execution
"""

from .metrics_ingest import MetricsIngestService
from .organ_scoring import OrganScoringService
from .dpi_computation import DPIComputationService
from .rule_engine import RuleBasedAlertEngine
from .notification_service import NotificationService
from .escalation_service import EscalationService
from .ml_ranking import MLRankingService
from .config_service import AlertConfigService
from .background_worker import AlertEngineWorker, AlertEngineCronJob, start_worker_in_thread

__all__ = [
    'MetricsIngestService',
    'OrganScoringService',
    'DPIComputationService',
    'RuleBasedAlertEngine',
    'NotificationService',
    'EscalationService',
    'MLRankingService',
    'AlertConfigService',
    'AlertEngineWorker',
    'AlertEngineCronJob',
    'start_worker_in_thread'
]
