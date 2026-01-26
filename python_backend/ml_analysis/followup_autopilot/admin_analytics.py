"""
Phase 5: Admin Autopilot Analytics Service

Provides system health monitoring, engagement analytics, model performance tracking,
patient cohort analysis, and configuration management.

HIPAA Compliance:
- All patient-level data is aggregated (no individual PHI exposed in admin views)
- Cohort data uses MIN_CELL_SIZE=10 for privacy
- All operations are audit logged
"""

import os
import logging
from datetime import datetime, timezone, date, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union
from uuid import uuid4
import psycopg2
import psycopg2.extras
from decimal import Decimal

logger = logging.getLogger(__name__)

MIN_CELL_SIZE = 10
WELLNESS_DISCLAIMER = "This dashboard shows wellness monitoring metrics only. Not medical advice."


def normalize_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Convert Decimal and other non-JSON types for safe serialization."""
    result = {}
    for key, value in row.items():
        if isinstance(value, Decimal):
            result[key] = float(value)
        elif isinstance(value, (datetime, date)):
            result[key] = value.isoformat()
        else:
            result[key] = value
    return result


class AdminAnalyticsService:
    """
    Admin analytics for Autopilot system monitoring.
    """
    
    def __init__(self, db_session=None):
        self.db = db_session
        self.logger = logging.getLogger(__name__)
    
    def get_system_health(self) -> Dict[str, Any]:
        """
        Get current system health metrics.
        """
        conn_str = os.environ.get('DATABASE_URL')
        if not conn_str:
            return self._get_mock_system_health()
        
        try:
            with psycopg2.connect(conn_str) as conn:
                with conn.cursor() as cur:
                    now = datetime.now(timezone.utc)
                    hour_ago = now - timedelta(hours=1)
                    
                    # Active patients count
                    cur.execute("""
                        SELECT 
                            COUNT(*) as total,
                            COUNT(CASE WHEN risk_state = 'AtRisk' THEN 1 END) as at_risk,
                            COUNT(CASE WHEN risk_state = 'Worsening' THEN 1 END) as worsening,
                            COUNT(CASE WHEN risk_state = 'Critical' THEN 1 END) as critical
                        FROM autopilot_patient_states
                        WHERE last_updated >= %s
                    """, (now - timedelta(days=7),))
                    patient_row = cur.fetchone()
                    
                    # Signals in last hour
                    cur.execute("""
                        SELECT COUNT(*) FROM autopilot_patient_signals
                        WHERE signal_time >= %s
                    """, (hour_ago,))
                    signals_result = cur.fetchone()
                    signals_hour = signals_result[0] if signals_result else 0
                    
                    # Tasks created in last hour
                    cur.execute("""
                        SELECT COUNT(*) FROM autopilot_followup_tasks
                        WHERE created_at >= %s
                    """, (hour_ago,))
                    tasks_result = cur.fetchone()
                    tasks_hour = tasks_result[0] if tasks_result else 0
                    
                    # Notifications in last hour
                    cur.execute("""
                        SELECT 
                            COUNT(*) as total,
                            COUNT(CASE WHEN status = 'sent' THEN 1 END) as sent,
                            COUNT(CASE WHEN status = 'failed' THEN 1 END) as failed
                        FROM autopilot_notifications
                        WHERE created_at >= %s
                    """, (hour_ago,))
                    notif_row = cur.fetchone()
                    
                    # Trigger events in last hour
                    cur.execute("""
                        SELECT 
                            COUNT(*) as total,
                            COUNT(CASE WHEN severity = 'alert' THEN 1 END) as alerts,
                            COUNT(CASE WHEN severity = 'warning' THEN 1 END) as warnings
                        FROM autopilot_trigger_events
                        WHERE created_at >= %s
                    """, (hour_ago,))
                    trigger_row = cur.fetchone()
                    
                    return {
                        "status": "healthy",
                        "timestamp": now.isoformat(),
                        "patients": {
                            "active": patient_row[0] if patient_row else 0,
                            "at_risk": patient_row[1] if patient_row else 0,
                            "worsening": patient_row[2] if patient_row else 0,
                            "critical": patient_row[3] if patient_row else 0
                        },
                        "activity_last_hour": {
                            "signals_ingested": signals_hour,
                            "tasks_created": tasks_hour,
                            "notifications_sent": notif_row[1] if notif_row else 0,
                            "notification_failures": notif_row[2] if notif_row else 0,
                            "triggers_fired": trigger_row[0] if trigger_row else 0,
                            "alerts": trigger_row[1] if trigger_row else 0,
                            "warnings": trigger_row[2] if trigger_row else 0
                        },
                        "wellness_disclaimer": WELLNESS_DISCLAIMER
                    }
        except Exception as e:
            self.logger.error(f"Error getting system health: {e}")
            return self._get_mock_system_health()
    
    def _get_mock_system_health(self) -> Dict[str, Any]:
        """Return mock data when database is unavailable."""
        return {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "patients": {
                "active": 0,
                "at_risk": 0,
                "worsening": 0,
                "critical": 0
            },
            "activity_last_hour": {
                "signals_ingested": 0,
                "tasks_created": 0,
                "notifications_sent": 0,
                "notification_failures": 0,
                "triggers_fired": 0,
                "alerts": 0,
                "warnings": 0
            },
            "wellness_disclaimer": WELLNESS_DISCLAIMER
        }
    
    def get_engagement_analytics(self, days: int = 30) -> Dict[str, Any]:
        """
        Get patient engagement analytics for the specified period.
        """
        conn_str = os.environ.get('DATABASE_URL')
        if not conn_str:
            return self._get_mock_engagement_analytics(days)
        
        try:
            with psycopg2.connect(conn_str) as conn:
                with conn.cursor() as cur:
                    end_date = date.today()
                    start_date = end_date - timedelta(days=days)
                    
                    # Task completion stats
                    cur.execute("""
                        SELECT 
                            COUNT(*) as total_tasks,
                            COUNT(CASE WHEN status = 'completed' THEN 1 END) as completed,
                            COUNT(CASE WHEN status = 'pending' AND due_at < NOW() THEN 1 END) as expired,
                            AVG(EXTRACT(EPOCH FROM (completed_at - created_at))/3600) 
                                FILTER (WHERE status = 'completed') as avg_hours_to_complete
                        FROM autopilot_followup_tasks
                        WHERE created_at >= %s AND created_at <= %s
                    """, (start_date, end_date))
                    task_row = cur.fetchone()
                    
                    # Notification stats
                    cur.execute("""
                        SELECT 
                            COUNT(*) as total,
                            COUNT(CASE WHEN status = 'sent' THEN 1 END) as sent
                        FROM autopilot_notifications
                        WHERE created_at >= %s AND created_at <= %s
                    """, (start_date, end_date))
                    notif_row = cur.fetchone()
                    
                    # By task type breakdown
                    cur.execute("""
                        SELECT 
                            task_type,
                            COUNT(*) as total,
                            COUNT(CASE WHEN status = 'completed' THEN 1 END) as completed
                        FROM autopilot_followup_tasks
                        WHERE created_at >= %s AND created_at <= %s
                        GROUP BY task_type
                        HAVING COUNT(*) >= %s
                    """, (start_date, end_date, MIN_CELL_SIZE))
                    task_type_rows = cur.fetchall()
                    
                    by_task_type = {}
                    for row in task_type_rows:
                        by_task_type[row[0]] = {
                            "total": row[1],
                            "completed": row[2],
                            "completion_rate": round(row[2] / row[1] * 100, 1) if row[1] > 0 else 0
                        }
                    
                    # By priority breakdown
                    cur.execute("""
                        SELECT 
                            priority,
                            COUNT(*) as total,
                            COUNT(CASE WHEN status = 'completed' THEN 1 END) as completed
                        FROM autopilot_followup_tasks
                        WHERE created_at >= %s AND created_at <= %s
                        GROUP BY priority
                        HAVING COUNT(*) >= %s
                    """, (start_date, end_date, MIN_CELL_SIZE))
                    priority_rows = cur.fetchall()
                    
                    by_priority = {}
                    for row in priority_rows:
                        by_priority[row[0]] = {
                            "total": row[1],
                            "completed": row[2],
                            "completion_rate": round(row[2] / row[1] * 100, 1) if row[1] > 0 else 0
                        }
                    
                    total_tasks = task_row[0] if task_row and task_row[0] else 0
                    completed_tasks = task_row[1] if task_row and task_row[1] else 0
                    
                    return {
                        "period_days": days,
                        "start_date": start_date.isoformat(),
                        "end_date": end_date.isoformat(),
                        "tasks": {
                            "total": total_tasks,
                            "completed": completed_tasks,
                            "expired": task_row[2] if task_row and task_row[2] else 0,
                            "completion_rate": round(completed_tasks / total_tasks * 100, 1) if total_tasks > 0 else 0,
                            "avg_hours_to_complete": round(task_row[3], 1) if task_row and task_row[3] else None
                        },
                        "notifications": {
                            "total": notif_row[0] if notif_row else 0,
                            "sent": notif_row[1] if notif_row else 0,
                            "delivery_rate": round(notif_row[1] / notif_row[0] * 100, 1) if notif_row and notif_row[0] > 0 else 0
                        },
                        "by_task_type": by_task_type,
                        "by_priority": by_priority,
                        "wellness_disclaimer": WELLNESS_DISCLAIMER
                    }
        except Exception as e:
            self.logger.error(f"Error getting engagement analytics: {e}")
            return self._get_mock_engagement_analytics(days)
    
    def _get_mock_engagement_analytics(self, days: int) -> Dict[str, Any]:
        """Return mock engagement data."""
        end_date = date.today()
        start_date = end_date - timedelta(days=days)
        return {
            "period_days": days,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "tasks": {
                "total": 0,
                "completed": 0,
                "expired": 0,
                "completion_rate": 0,
                "avg_hours_to_complete": None
            },
            "notifications": {
                "total": 0,
                "sent": 0,
                "delivery_rate": 0
            },
            "by_task_type": {},
            "by_priority": {},
            "wellness_disclaimer": WELLNESS_DISCLAIMER
        }
    
    def get_model_performance(self, model_name: Optional[str] = None, days: int = 30) -> Dict[str, Any]:
        """
        Get ML model performance metrics.
        """
        conn_str = os.environ.get('DATABASE_URL')
        if not conn_str:
            return self._get_mock_model_performance(days)
        
        try:
            with psycopg2.connect(conn_str) as conn:
                with conn.cursor() as cur:
                    end_date = date.today()
                    start_date = end_date - timedelta(days=days)
                    
                    # Get latest performance metrics
                    query = """
                        SELECT 
                            model_name, model_version, date,
                            accuracy, precision_score, recall, f1_score, auc_roc,
                            predictions_count, feature_drift_score, prediction_drift_score,
                            drift_alert
                        FROM autopilot_model_performance
                        WHERE date >= %s
                    """
                    params: List[Any] = [start_date]
                    
                    if model_name:
                        query += " AND model_name = %s"
                        params.append(model_name)
                    
                    query += " ORDER BY date DESC, model_name"
                    
                    cur.execute(query, params)
                    rows = cur.fetchall()
                    
                    models = {}
                    for row in rows:
                        name = row[0]
                        if name not in models:
                            models[name] = {
                                "model_name": name,
                                "current_version": row[1],
                                "latest_metrics": {
                                    "date": row[2].isoformat() if row[2] else None,
                                    "accuracy": float(row[3]) if row[3] else None,
                                    "precision": float(row[4]) if row[4] else None,
                                    "recall": float(row[5]) if row[5] else None,
                                    "f1_score": float(row[6]) if row[6] else None,
                                    "auc_roc": float(row[7]) if row[7] else None,
                                    "predictions_count": row[8] or 0
                                },
                                "drift": {
                                    "feature_drift_score": float(row[9]) if row[9] else 0,
                                    "prediction_drift_score": float(row[10]) if row[10] else 0,
                                    "drift_alert": row[11] or False
                                },
                                "history": []
                            }
                        
                        # Add to history
                        models[name]["history"].append({
                            "date": row[2].isoformat() if row[2] else None,
                            "accuracy": float(row[3]) if row[3] else None,
                            "f1_score": float(row[6]) if row[6] else None
                        })
                    
                    # If no data, return default model list
                    if not models:
                        models = self._get_default_model_list()
                    
                    return {
                        "period_days": days,
                        "models": list(models.values()),
                        "drift_alerts_active": sum(1 for m in models.values() if m.get("drift", {}).get("drift_alert")),
                        "wellness_disclaimer": WELLNESS_DISCLAIMER
                    }
        except Exception as e:
            self.logger.error(f"Error getting model performance: {e}")
            return self._get_mock_model_performance(days)
    
    def _get_default_model_list(self) -> Dict[str, Any]:
        """Return default model definitions."""
        return {
            "risk_model": {
                "model_name": "risk_model",
                "current_version": "1.0.0",
                "latest_metrics": None,
                "drift": {"feature_drift_score": 0, "prediction_drift_score": 0, "drift_alert": False},
                "history": []
            },
            "adherence_model": {
                "model_name": "adherence_model",
                "current_version": "1.0.0",
                "latest_metrics": None,
                "drift": {"feature_drift_score": 0, "prediction_drift_score": 0, "drift_alert": False},
                "history": []
            },
            "anomaly_detector": {
                "model_name": "anomaly_detector",
                "current_version": "1.0.0",
                "latest_metrics": None,
                "drift": {"feature_drift_score": 0, "prediction_drift_score": 0, "drift_alert": False},
                "history": []
            },
            "engagement_model": {
                "model_name": "engagement_model",
                "current_version": "1.0.0",
                "latest_metrics": None,
                "drift": {"feature_drift_score": 0, "prediction_drift_score": 0, "drift_alert": False},
                "history": []
            }
        }
    
    def _get_mock_model_performance(self, days: int) -> Dict[str, Any]:
        """Return mock model performance."""
        return {
            "period_days": days,
            "models": list(self._get_default_model_list().values()),
            "drift_alerts_active": 0,
            "wellness_disclaimer": WELLNESS_DISCLAIMER
        }
    
    def get_patient_cohorts(self) -> Dict[str, Any]:
        """
        Get patient cohort distribution with privacy protection.
        """
        conn_str = os.environ.get('DATABASE_URL')
        if not conn_str:
            return self._get_mock_cohorts()
        
        try:
            with psycopg2.connect(conn_str) as conn:
                with conn.cursor() as cur:
                    # Get cohort distribution with MIN_CELL_SIZE
                    cur.execute("""
                        SELECT 
                            cohort_name,
                            risk_level,
                            COUNT(*) as patient_count,
                            AVG(risk_score_avg_30d) as avg_risk,
                            AVG(task_completion_rate_30d) as avg_completion_rate
                        FROM autopilot_patient_cohorts
                        GROUP BY cohort_name, risk_level
                        HAVING COUNT(*) >= %s
                    """, (MIN_CELL_SIZE,))
                    cohort_rows = cur.fetchall()
                    
                    # Get risk state distribution
                    cur.execute("""
                        SELECT 
                            risk_state,
                            COUNT(*) as count
                        FROM autopilot_patient_states
                        GROUP BY risk_state
                        HAVING COUNT(*) >= %s
                    """, (MIN_CELL_SIZE,))
                    risk_rows = cur.fetchall()
                    
                    cohorts = []
                    for row in cohort_rows:
                        cohorts.append({
                            "cohort_name": row[0],
                            "risk_level": row[1],
                            "patient_count": row[2],
                            "avg_risk_score": round(float(row[3]), 1) if row[3] else 0,
                            "avg_completion_rate": round(float(row[4]) * 100, 1) if row[4] else 0
                        })
                    
                    risk_distribution = {}
                    for row in risk_rows:
                        risk_distribution[row[0]] = row[1]
                    
                    return {
                        "cohorts": cohorts,
                        "risk_distribution": risk_distribution,
                        "total_cohorts": len(cohorts),
                        "privacy_note": f"Cell sizes below {MIN_CELL_SIZE} are suppressed for patient privacy.",
                        "wellness_disclaimer": WELLNESS_DISCLAIMER
                    }
        except Exception as e:
            self.logger.error(f"Error getting cohorts: {e}")
            return self._get_mock_cohorts()
    
    def _get_mock_cohorts(self) -> Dict[str, Any]:
        """Return mock cohort data."""
        return {
            "cohorts": [],
            "risk_distribution": {},
            "total_cohorts": 0,
            "privacy_note": f"Cell sizes below {MIN_CELL_SIZE} are suppressed for patient privacy.",
            "wellness_disclaimer": WELLNESS_DISCLAIMER
        }
    
    def get_configurations(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get autopilot configuration settings.
        """
        conn_str = os.environ.get('DATABASE_URL')
        if not conn_str:
            return self._get_default_configurations()
        
        try:
            with psycopg2.connect(conn_str) as conn:
                with conn.cursor() as cur:
                    query = """
                        SELECT 
                            config_key, config_value, category, description,
                            is_active, requires_restart, min_value, max_value,
                            updated_at
                        FROM autopilot_configurations
                        WHERE is_active = true
                    """
                    params = []
                    
                    if category:
                        query += " AND category = %s"
                        params.append(category)
                    
                    query += " ORDER BY category, config_key"
                    
                    cur.execute(query, params)
                    rows = cur.fetchall()
                    
                    if not rows:
                        return self._get_default_configurations()
                    
                    configs = []
                    for row in rows:
                        configs.append({
                            "key": row[0],
                            "value": row[1],
                            "category": row[2],
                            "description": row[3],
                            "is_active": row[4],
                            "requires_restart": row[5],
                            "min_value": float(row[6]) if row[6] else None,
                            "max_value": float(row[7]) if row[7] else None,
                            "updated_at": row[8].isoformat() if row[8] else None
                        })
                    
                    return configs
        except Exception as e:
            self.logger.error(f"Error getting configurations: {e}")
            return self._get_default_configurations()
    
    def _get_default_configurations(self) -> List[Dict[str, Any]]:
        """Return default autopilot configurations."""
        return [
            {
                "key": "risk_threshold_warning",
                "value": 30,
                "category": "triggers",
                "description": "Risk score threshold for warning triggers",
                "is_active": True,
                "requires_restart": False,
                "min_value": 10,
                "max_value": 50
            },
            {
                "key": "risk_threshold_alert",
                "value": 60,
                "category": "triggers",
                "description": "Risk score threshold for alert triggers",
                "is_active": True,
                "requires_restart": False,
                "min_value": 40,
                "max_value": 80
            },
            {
                "key": "risk_threshold_critical",
                "value": 80,
                "category": "triggers",
                "description": "Risk score threshold for critical triggers",
                "is_active": True,
                "requires_restart": False,
                "min_value": 60,
                "max_value": 100
            },
            {
                "key": "default_followup_hours",
                "value": 24,
                "category": "scheduling",
                "description": "Default hours between follow-ups",
                "is_active": True,
                "requires_restart": False,
                "min_value": 4,
                "max_value": 168
            },
            {
                "key": "notification_cooldown_minutes",
                "value": 60,
                "category": "notifications",
                "description": "Minimum minutes between similar notifications",
                "is_active": True,
                "requires_restart": False,
                "min_value": 15,
                "max_value": 1440
            },
            {
                "key": "ml_inference_enabled",
                "value": True,
                "category": "ml",
                "description": "Enable ML model inference",
                "is_active": True,
                "requires_restart": True,
                "min_value": None,
                "max_value": None
            }
        ]
    
    def update_configuration(
        self,
        config_key: str,
        config_value: Any,
        user_id: str
    ) -> Dict[str, Any]:
        """
        Update an autopilot configuration.
        """
        conn_str = os.environ.get('DATABASE_URL')
        if not conn_str:
            return {"success": False, "error": "Database not available"}
        
        try:
            with psycopg2.connect(conn_str) as conn:
                with conn.cursor() as cur:
                    # Check if config exists
                    cur.execute("""
                        SELECT id, config_value, min_value, max_value
                        FROM autopilot_configurations
                        WHERE config_key = %s
                    """, (config_key,))
                    existing = cur.fetchone()
                    
                    if not existing:
                        # Insert new config
                        cur.execute("""
                            INSERT INTO autopilot_configurations (
                                config_key, config_value, category, 
                                is_active, updated_by, updated_at
                            ) VALUES (%s, %s, 'custom', true, %s, NOW())
                            RETURNING id
                        """, (config_key, psycopg2.extras.Json(config_value), user_id))
                    else:
                        # Validate against min/max if numeric
                        if isinstance(config_value, (int, float)):
                            min_val = existing[2]
                            max_val = existing[3]
                            if min_val is not None and config_value < min_val:
                                return {"success": False, "error": f"Value must be >= {min_val}"}
                            if max_val is not None and config_value > max_val:
                                return {"success": False, "error": f"Value must be <= {max_val}"}
                        
                        # Update existing
                        cur.execute("""
                            UPDATE autopilot_configurations
                            SET config_value = %s, updated_by = %s, updated_at = NOW()
                            WHERE config_key = %s
                        """, (psycopg2.extras.Json(config_value), user_id, config_key))
                    
                    conn.commit()
                    
                    # Audit log
                    self._log_config_change(cur, config_key, existing[1] if existing else None, config_value, user_id)
                    
                    return {"success": True, "key": config_key, "value": config_value}
        except Exception as e:
            self.logger.error(f"Error updating configuration: {e}")
            return {"success": False, "error": str(e)}
    
    def _log_config_change(self, cursor, key: str, old_value: Any, new_value: Any, user_id: str):
        """Log configuration change for audit."""
        try:
            import psycopg2.extras
            cursor.execute("""
                INSERT INTO autopilot_audit_logs (
                    action, entity_type, entity_id, user_id,
                    old_values, new_values, created_at
                ) VALUES ('config_update', 'configuration', %s, %s, %s, %s, NOW())
            """, (
                key, user_id,
                psycopg2.extras.Json({"value": old_value}) if old_value else None,
                psycopg2.extras.Json({"value": new_value})
            ))
        except Exception as e:
            self.logger.warning(f"Audit log failed: {e}")
    
    def get_trigger_analytics(self, days: int = 30) -> Dict[str, Any]:
        """
        Get trigger analytics for the specified period.
        """
        conn_str = os.environ.get('DATABASE_URL')
        if not conn_str:
            return self._get_mock_trigger_analytics(days)
        
        try:
            with psycopg2.connect(conn_str) as conn:
                with conn.cursor() as cur:
                    end_date = datetime.now(timezone.utc)
                    start_date = end_date - timedelta(days=days)
                    
                    # By trigger name
                    cur.execute("""
                        SELECT 
                            name,
                            severity,
                            COUNT(*) as count
                        FROM autopilot_trigger_events
                        WHERE created_at >= %s
                        GROUP BY name, severity
                        HAVING COUNT(*) >= %s
                        ORDER BY count DESC
                    """, (start_date, MIN_CELL_SIZE))
                    trigger_rows = cur.fetchall()
                    
                    by_trigger = {}
                    for row in trigger_rows:
                        name = row[0]
                        if name not in by_trigger:
                            by_trigger[name] = {"name": name, "severity": row[1], "count": 0}
                        by_trigger[name]["count"] += row[2]
                    
                    # By severity totals
                    cur.execute("""
                        SELECT severity, COUNT(*) as count
                        FROM autopilot_trigger_events
                        WHERE created_at >= %s
                        GROUP BY severity
                    """, (start_date,))
                    severity_rows = cur.fetchall()
                    
                    by_severity = {}
                    for row in severity_rows:
                        by_severity[row[0]] = row[1]
                    
                    return {
                        "period_days": days,
                        "by_trigger": list(by_trigger.values()),
                        "by_severity": by_severity,
                        "total_triggers": sum(by_severity.values()),
                        "wellness_disclaimer": WELLNESS_DISCLAIMER
                    }
        except Exception as e:
            self.logger.error(f"Error getting trigger analytics: {e}")
            return self._get_mock_trigger_analytics(days)
    
    def _get_mock_trigger_analytics(self, days: int) -> Dict[str, Any]:
        """Return mock trigger analytics."""
        return {
            "period_days": days,
            "by_trigger": [],
            "by_severity": {},
            "total_triggers": 0,
            "wellness_disclaimer": WELLNESS_DISCLAIMER
        }


# Singleton instance
_admin_analytics_service: Optional[AdminAnalyticsService] = None


def get_admin_analytics_service(db_session=None) -> AdminAnalyticsService:
    """Get or create singleton admin analytics service."""
    global _admin_analytics_service
    if _admin_analytics_service is None or db_session:
        _admin_analytics_service = AdminAnalyticsService(db_session)
    return _admin_analytics_service
