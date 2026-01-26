"""
ML/AI Alerting Service
=======================
Threshold-based alerting for:
1. Retrieval failure rate exceeds threshold
2. Embedding model null checks
3. Latency SLA violations
4. Error rate spikes

Alerts are logged and can be routed to external systems.
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from threading import Lock, Thread
import time

from app.services.ml_observability import get_observability_service

logger = logging.getLogger(__name__)


class AlertSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertStatus(str, Enum):
    FIRING = "firing"
    RESOLVED = "resolved"


@dataclass
class Alert:
    """Alert definition"""
    id: str
    name: str
    severity: AlertSeverity
    status: AlertStatus
    message: str
    details: Dict[str, Any]
    triggered_at: datetime
    resolved_at: Optional[datetime] = None
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class AlertRule:
    """Rule for triggering alerts"""
    id: str
    name: str
    description: str
    severity: AlertSeverity
    check_fn: Callable[[], Optional[str]]
    cooldown_seconds: int = 300
    last_triggered: Optional[datetime] = None
    last_resolved: Optional[datetime] = None
    is_firing: bool = False


class MLAlertingService:
    """
    Alerting service for ML/AI operations.
    
    Monitors metrics and triggers alerts when thresholds are exceeded.
    """
    
    def __init__(self):
        self._observability = get_observability_service()
        self._rules: Dict[str, AlertRule] = {}
        self._alerts: List[Alert] = []
        self._max_alerts = 1000
        self._lock = Lock()
        self._running = False
        self._check_thread: Optional[Thread] = None
        
        self._register_default_rules()
        logger.info("ML Alerting Service initialized")
    
    def _register_default_rules(self):
        """Register default alert rules"""
        
        self.register_rule(AlertRule(
            id="retrieval_failure_rate",
            name="High Retrieval Failure Rate",
            description="Retrieval operations are failing at a high rate",
            severity=AlertSeverity.ERROR,
            check_fn=self._check_retrieval_failure_rate,
            cooldown_seconds=300,
        ))
        
        self.register_rule(AlertRule(
            id="embedding_null_model",
            name="Embedding Model Null",
            description="Memories found with null embedding_model field",
            severity=AlertSeverity.WARNING,
            check_fn=self._check_embedding_model_null,
            cooldown_seconds=600,
        ))
        
        self.register_rule(AlertRule(
            id="retrieval_latency_sla",
            name="Retrieval Latency SLA Violation",
            description="Retrieval p99 latency exceeds 1000ms threshold",
            severity=AlertSeverity.WARNING,
            check_fn=self._check_retrieval_latency_sla,
            cooldown_seconds=300,
        ))
        
        self.register_rule(AlertRule(
            id="api_error_rate",
            name="High API Error Rate",
            description="OpenAI API error rate exceeds 5%",
            severity=AlertSeverity.ERROR,
            check_fn=self._check_api_error_rate,
            cooldown_seconds=300,
        ))
        
        self.register_rule(AlertRule(
            id="embedding_latency_sla",
            name="Embedding Latency SLA Violation",
            description="Embedding generation p99 latency exceeds 2000ms",
            severity=AlertSeverity.WARNING,
            check_fn=self._check_embedding_latency_sla,
            cooldown_seconds=300,
        ))
        
        self.register_rule(AlertRule(
            id="phi_detection_spike",
            name="PHI Detection Spike",
            description="Unusual number of PHI detections",
            severity=AlertSeverity.WARNING,
            check_fn=self._check_phi_detection_spike,
            cooldown_seconds=600,
        ))
    
    def _check_retrieval_failure_rate(self) -> Optional[str]:
        """Check if retrieval failure rate is too high"""
        ops = self._observability.memory_operations.by_label()
        
        search_success = ops.get('{"operation": "search", "success": "True"}', 0)
        search_fail = ops.get('{"operation": "search", "success": "False"}', 0)
        
        total = search_success + search_fail
        if total < 10:
            return None
        
        failure_rate = search_fail / total
        if failure_rate > 0.1:
            return f"Retrieval failure rate is {failure_rate:.1%} (threshold: 10%)"
        
        return None
    
    def _check_embedding_model_null(self) -> Optional[str]:
        """Check for null embedding models - placeholder for DB query"""
        return None
    
    def _check_retrieval_latency_sla(self) -> Optional[str]:
        """Check retrieval latency p99"""
        p99 = self._observability.retrieval_latency.percentile(0.99)
        if p99 > 1000 and self._observability.retrieval_latency.count() > 10:
            return f"Retrieval p99 latency is {p99:.0f}ms (threshold: 1000ms)"
        return None
    
    def _check_api_error_rate(self) -> Optional[str]:
        """Check API error rate"""
        error_metrics = self._observability.get_error_metrics()
        if error_metrics["total_api_calls"] < 10:
            return None
        
        if error_metrics["error_rate"] > 0.05:
            return f"API error rate is {error_metrics['error_rate']:.1%} (threshold: 5%)"
        return None
    
    def _check_embedding_latency_sla(self) -> Optional[str]:
        """Check embedding latency p99"""
        p99 = self._observability.embedding_latency.percentile(0.99)
        if p99 > 2000 and self._observability.embedding_latency.count() > 10:
            return f"Embedding p99 latency is {p99:.0f}ms (threshold: 2000ms)"
        return None
    
    def _check_phi_detection_spike(self) -> Optional[str]:
        """Check for unusual PHI detection activity"""
        total = self._observability.phi_detections.total()
        if total > 100:
            return f"High number of PHI detections: {total}"
        return None
    
    def register_rule(self, rule: AlertRule):
        """Register an alert rule"""
        with self._lock:
            self._rules[rule.id] = rule
    
    def check_rules(self):
        """Check all rules and trigger/resolve alerts"""
        for rule_id, rule in self._rules.items():
            try:
                self._check_rule(rule)
            except Exception as e:
                logger.error(f"Error checking rule {rule_id}: {e}")
    
    def _check_rule(self, rule: AlertRule):
        """Check a single rule"""
        now = datetime.utcnow()
        
        if rule.last_triggered:
            cooldown_end = rule.last_triggered + timedelta(seconds=rule.cooldown_seconds)
            if rule.is_firing and now < cooldown_end:
                return
        
        message = rule.check_fn()
        
        if message:
            if not rule.is_firing:
                alert = Alert(
                    id=f"{rule.id}_{now.timestamp()}",
                    name=rule.name,
                    severity=rule.severity,
                    status=AlertStatus.FIRING,
                    message=message,
                    details={"rule_id": rule.id, "description": rule.description},
                    triggered_at=now,
                    labels={"rule": rule.id},
                )
                
                self._fire_alert(alert)
                rule.is_firing = True
                rule.last_triggered = now
        else:
            if rule.is_firing:
                self._resolve_rule(rule)
                rule.is_firing = False
                rule.last_resolved = now
    
    def _fire_alert(self, alert: Alert):
        """Fire an alert"""
        with self._lock:
            self._alerts.append(alert)
            
            if len(self._alerts) > self._max_alerts:
                self._alerts = self._alerts[-self._max_alerts:]
        
        log_fn = logger.warning if alert.severity in [AlertSeverity.INFO, AlertSeverity.WARNING] else logger.error
        log_fn(f"ALERT [{alert.severity.value.upper()}] {alert.name}: {alert.message}")
    
    def _resolve_rule(self, rule: AlertRule):
        """Resolve alerts for a rule"""
        now = datetime.utcnow()
        
        with self._lock:
            for alert in self._alerts:
                if alert.labels.get("rule") == rule.id and alert.status == AlertStatus.FIRING:
                    alert.status = AlertStatus.RESOLVED
                    alert.resolved_at = now
        
        logger.info(f"ALERT RESOLVED: {rule.name}")
    
    def get_firing_alerts(self) -> List[Alert]:
        """Get all currently firing alerts"""
        with self._lock:
            return [a for a in self._alerts if a.status == AlertStatus.FIRING]
    
    def get_recent_alerts(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent alerts"""
        with self._lock:
            alerts = self._alerts[-limit:]
            return [
                {
                    "id": a.id,
                    "name": a.name,
                    "severity": a.severity.value,
                    "status": a.status.value,
                    "message": a.message,
                    "triggered_at": a.triggered_at.isoformat(),
                    "resolved_at": a.resolved_at.isoformat() if a.resolved_at else None,
                }
                for a in reversed(alerts)
            ]
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get alert summary"""
        firing = self.get_firing_alerts()
        
        by_severity = {}
        for alert in firing:
            sev = alert.severity.value
            by_severity[sev] = by_severity.get(sev, 0) + 1
        
        return {
            "total_firing": len(firing),
            "by_severity": by_severity,
            "alerts": [
                {"name": a.name, "severity": a.severity.value, "message": a.message}
                for a in firing
            ]
        }
    
    def start_background_check(self, interval_seconds: int = 60):
        """Start background alert checking"""
        if self._running:
            return
        
        self._running = True
        
        def check_loop():
            while self._running:
                try:
                    self.check_rules()
                except Exception as e:
                    logger.error(f"Error in alert check loop: {e}")
                time.sleep(interval_seconds)
        
        self._check_thread = Thread(target=check_loop, daemon=True)
        self._check_thread.start()
        logger.info(f"Started background alert checking (interval: {interval_seconds}s)")
    
    def stop_background_check(self):
        """Stop background alert checking"""
        self._running = False
        if self._check_thread:
            self._check_thread.join(timeout=5)


_alerting_service: Optional[MLAlertingService] = None


def get_alerting_service() -> MLAlertingService:
    """Get singleton alerting service"""
    global _alerting_service
    if _alerting_service is None:
        _alerting_service = MLAlertingService()
    return _alerting_service
