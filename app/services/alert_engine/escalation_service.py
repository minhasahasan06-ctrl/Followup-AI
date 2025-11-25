"""
Escalation Service - Manages alert escalation for unacknowledged alerts.

Features:
- Configurable escalation timeouts by severity
- Supervisor/backup clinician notification
- Escalation chain management
- Escalation audit logging
"""

import os
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from sqlalchemy.orm import Session
from sqlalchemy import text

logger = logging.getLogger(__name__)

from .config_service import AlertConfigService
from .notification_service import NotificationService, NotificationRequest, NotificationChannel


@dataclass
class EscalationPolicy:
    """Escalation policy configuration"""
    severity: str
    timeout_hours: float
    escalate_to_role: str  # supervisor, backup, on_call
    notification_channels: List[NotificationChannel]
    max_escalation_level: int = 3


@dataclass
class EscalationRecord:
    """Record of an escalation action"""
    id: str
    alert_id: str
    patient_id: str
    escalation_level: int
    escalated_to: str
    escalated_from: str
    reason: str
    escalated_at: datetime


class EscalationService:
    """Service for managing alert escalations"""
    
    def __init__(self, db: Session, notification_service: NotificationService = None):
        self.db = db
        self.config_service = AlertConfigService()
        self.notification_service = notification_service or NotificationService(db)
        
        # Default escalation policies
        self.policies = {
            "critical": EscalationPolicy(
                severity="critical",
                timeout_hours=2.0,
                escalate_to_role="supervisor",
                notification_channels=[NotificationChannel.SMS, NotificationChannel.EMAIL, NotificationChannel.DASHBOARD],
                max_escalation_level=3
            ),
            "high": EscalationPolicy(
                severity="high",
                timeout_hours=4.0,
                escalate_to_role="supervisor",
                notification_channels=[NotificationChannel.EMAIL, NotificationChannel.DASHBOARD],
                max_escalation_level=2
            ),
            "moderate": EscalationPolicy(
                severity="moderate",
                timeout_hours=8.0,
                escalate_to_role="backup",
                notification_channels=[NotificationChannel.EMAIL, NotificationChannel.DASHBOARD],
                max_escalation_level=2
            ),
            "low": EscalationPolicy(
                severity="low",
                timeout_hours=24.0,
                escalate_to_role="backup",
                notification_channels=[NotificationChannel.DASHBOARD],
                max_escalation_level=1
            )
        }
    
    async def check_and_escalate_alerts(self) -> List[Dict[str, Any]]:
        """
        Check all unacknowledged alerts and escalate those past timeout.
        This should be called periodically by a background worker.
        
        Returns list of escalated alerts.
        """
        escalated = []
        
        for severity, policy in self.policies.items():
            alerts = await self._get_alerts_for_escalation(severity, policy.timeout_hours)
            
            for alert in alerts:
                result = await self._escalate_alert(alert, policy)
                if result:
                    escalated.append(result)
        
        return escalated
    
    async def _get_alerts_for_escalation(
        self,
        severity: str,
        timeout_hours: float
    ) -> List[Dict[str, Any]]:
        """Get alerts that need escalation"""
        timeout_cutoff = datetime.utcnow() - timedelta(hours=timeout_hours)
        
        query = text("""
            SELECT a.id, a.patient_id, a.alert_type, a.title, a.message,
                   a.severity, a.priority, a.created_at, a.status,
                   COALESCE(e.escalation_level, 0) as current_escalation_level,
                   a.assigned_clinician_id
            FROM ai_health_alerts a
            LEFT JOIN (
                SELECT alert_id, MAX(escalation_level) as escalation_level
                FROM alert_escalations
                GROUP BY alert_id
            ) e ON a.id = e.alert_id
            WHERE a.severity = :severity
            AND a.status NOT IN ('acknowledged', 'dismissed', 'closed')
            AND a.created_at <= :timeout_cutoff
            ORDER BY a.priority DESC, a.created_at ASC
        """)
        
        try:
            results = self.db.execute(query, {
                "severity": severity,
                "timeout_cutoff": timeout_cutoff
            }).fetchall()
            
            return [
                {
                    "id": str(row[0]),
                    "patient_id": row[1],
                    "alert_type": row[2],
                    "title": row[3],
                    "message": row[4],
                    "severity": row[5],
                    "priority": row[6],
                    "created_at": row[7],
                    "status": row[8],
                    "escalation_level": row[9] or 0,
                    "assigned_clinician_id": row[10]
                }
                for row in results
            ]
        except Exception as e:
            logger.error(f"Error getting alerts for escalation: {e}")
            return []
    
    async def _escalate_alert(
        self,
        alert: Dict[str, Any],
        policy: EscalationPolicy
    ) -> Optional[Dict[str, Any]]:
        """Escalate a single alert"""
        current_level = alert.get("escalation_level", 0)
        new_level = current_level + 1
        
        # Check if max escalation reached
        if new_level > policy.max_escalation_level:
            logger.info(f"Alert {alert['id']} at max escalation level")
            return None
        
        # Find escalation target
        escalation_target = await self._find_escalation_target(
            alert["patient_id"],
            alert.get("assigned_clinician_id"),
            policy.escalate_to_role,
            new_level
        )
        
        if not escalation_target:
            logger.warning(f"No escalation target found for alert {alert['id']}")
            return None
        
        try:
            # Update alert status
            update_query = text("""
                UPDATE ai_health_alerts
                SET status = 'escalated',
                    escalation_level = :level,
                    escalated_at = NOW(),
                    escalated_to = :escalated_to
                WHERE id = :alert_id
            """)
            
            self.db.execute(update_query, {
                "level": new_level,
                "escalated_to": escalation_target["id"],
                "alert_id": alert["id"]
            })
            
            # Record escalation
            escalation_id = await self._record_escalation(
                alert_id=alert["id"],
                patient_id=alert["patient_id"],
                level=new_level,
                escalated_to=escalation_target["id"],
                escalated_from=alert.get("assigned_clinician_id"),
                reason=f"Unacknowledged for {policy.timeout_hours} hours"
            )
            
            self.db.commit()
            
            # Send escalation notification
            await self._send_escalation_notification(
                alert=alert,
                target=escalation_target,
                level=new_level,
                channels=policy.notification_channels
            )
            
            logger.info(f"Alert {alert['id']} escalated to level {new_level}")
            
            return {
                "alert_id": alert["id"],
                "patient_id": alert["patient_id"],
                "new_level": new_level,
                "escalated_to": escalation_target["id"],
                "escalation_id": escalation_id
            }
            
        except Exception as e:
            logger.error(f"Error escalating alert {alert['id']}: {e}")
            self.db.rollback()
            return None
    
    async def _find_escalation_target(
        self,
        patient_id: str,
        current_assignee: Optional[str],
        role: str,
        level: int
    ) -> Optional[Dict[str, Any]]:
        """Find the appropriate escalation target"""
        # Try to find supervisor or backup based on role
        if role == "supervisor":
            # Find supervisor of current assignee
            query = text("""
                SELECT u.id, u.email, u.first_name, u.last_name, u.phone_number
                FROM users u
                WHERE u.role = 'doctor'
                AND u.id != :current_assignee
                ORDER BY u.created_at ASC
                LIMIT 1
            """)
        else:
            # Find backup clinician
            query = text("""
                SELECT u.id, u.email, u.first_name, u.last_name, u.phone_number
                FROM users u
                WHERE u.role = 'doctor'
                AND u.id != :current_assignee
                ORDER BY RANDOM()
                LIMIT 1
            """)
        
        try:
            result = self.db.execute(query, {
                "current_assignee": current_assignee or ""
            }).fetchone()
            
            if result:
                return {
                    "id": result[0],
                    "email": result[1],
                    "name": f"{result[2]} {result[3]}",
                    "phone": result[4]
                }
            return None
            
        except Exception as e:
            logger.error(f"Error finding escalation target: {e}")
            return None
    
    async def _record_escalation(
        self,
        alert_id: str,
        patient_id: str,
        level: int,
        escalated_to: str,
        escalated_from: Optional[str],
        reason: str
    ) -> str:
        """Record escalation in database"""
        import uuid
        escalation_id = str(uuid.uuid4())
        
        insert_query = text("""
            INSERT INTO alert_escalations (
                id, alert_id, patient_id, escalation_level,
                escalated_to, escalated_from, reason, escalated_at
            ) VALUES (
                :id, :alert_id, :patient_id, :level,
                :escalated_to, :escalated_from, :reason, NOW()
            )
        """)
        
        self.db.execute(insert_query, {
            "id": escalation_id,
            "alert_id": alert_id,
            "patient_id": patient_id,
            "level": level,
            "escalated_to": escalated_to,
            "escalated_from": escalated_from,
            "reason": reason
        })
        
        return escalation_id
    
    async def _send_escalation_notification(
        self,
        alert: Dict[str, Any],
        target: Dict[str, Any],
        level: int,
        channels: List[NotificationChannel]
    ):
        """Send notification about escalation"""
        # Create a mock alert record for notification
        from .rule_engine import AlertRecord
        
        alert_record = AlertRecord(
            id=alert["id"],
            patient_id=alert["patient_id"],
            alert_type=alert["alert_type"],
            alert_category="escalation",
            severity=alert["severity"],
            priority=alert["priority"],
            title=f"[ESCALATED L{level}] {alert['title']}",
            message=f"This alert has been escalated to you. {alert['message']}",
            disclaimer="This is an observational pattern alert requiring review.",
            trigger_rule=alert["alert_type"],
            trigger_metrics=[],
            dpi_at_trigger=None,
            organ_scores=None,
            suppression_key="",
            corroborated=False,
            status="escalated"
        )
        
        request = NotificationRequest(
            alert_id=alert["id"],
            patient_id=alert["patient_id"],
            recipient_id=target["id"],
            recipient_email=target.get("email"),
            recipient_phone=target.get("phone"),
            channels=channels,
            priority=alert["priority"],
            is_escalation=True
        )
        
        await self.notification_service.send_alert_notification(alert_record, request)
    
    async def get_escalation_history(
        self,
        alert_id: str
    ) -> List[Dict[str, Any]]:
        """Get escalation history for an alert"""
        query = text("""
            SELECT e.id, e.escalation_level, e.escalated_to, e.escalated_from,
                   e.reason, e.escalated_at,
                   u1.first_name || ' ' || u1.last_name as escalated_to_name,
                   u2.first_name || ' ' || u2.last_name as escalated_from_name
            FROM alert_escalations e
            LEFT JOIN users u1 ON e.escalated_to = u1.id
            LEFT JOIN users u2 ON e.escalated_from = u2.id
            WHERE e.alert_id = :alert_id
            ORDER BY e.escalated_at ASC
        """)
        
        try:
            results = self.db.execute(query, {"alert_id": alert_id}).fetchall()
            
            return [
                {
                    "id": str(row[0]),
                    "level": row[1],
                    "escalated_to": row[2],
                    "escalated_from": row[3],
                    "reason": row[4],
                    "escalated_at": row[5].isoformat() if row[5] else None,
                    "escalated_to_name": row[6],
                    "escalated_from_name": row[7]
                }
                for row in results
            ]
        except Exception as e:
            logger.error(f"Error getting escalation history: {e}")
            return []
    
    async def manual_escalate(
        self,
        alert_id: str,
        escalated_by: str,
        escalate_to: str,
        reason: str
    ) -> bool:
        """Manually escalate an alert"""
        try:
            # Get current alert info
            alert_query = text("""
                SELECT patient_id, severity, priority, title, message, alert_type,
                       COALESCE(escalation_level, 0) as current_level
                FROM ai_health_alerts
                WHERE id = :alert_id
            """)
            
            alert = self.db.execute(alert_query, {"alert_id": alert_id}).fetchone()
            if not alert:
                return False
            
            new_level = (alert[6] or 0) + 1
            
            # Update alert
            update_query = text("""
                UPDATE ai_health_alerts
                SET status = 'escalated',
                    escalation_level = :level,
                    escalated_at = NOW(),
                    escalated_to = :escalated_to
                WHERE id = :alert_id
            """)
            
            self.db.execute(update_query, {
                "level": new_level,
                "escalated_to": escalate_to,
                "alert_id": alert_id
            })
            
            # Record escalation
            await self._record_escalation(
                alert_id=alert_id,
                patient_id=alert[0],
                level=new_level,
                escalated_to=escalate_to,
                escalated_from=escalated_by,
                reason=reason
            )
            
            self.db.commit()
            
            # Get target info and send notification
            target_query = text("""
                SELECT id, email, first_name, last_name, phone_number
                FROM users WHERE id = :user_id
            """)
            target = self.db.execute(target_query, {"user_id": escalate_to}).fetchone()
            
            if target:
                await self._send_escalation_notification(
                    alert={
                        "id": alert_id,
                        "patient_id": alert[0],
                        "severity": alert[1],
                        "priority": alert[2],
                        "title": alert[3],
                        "message": alert[4],
                        "alert_type": alert[5]
                    },
                    target={
                        "id": target[0],
                        "email": target[1],
                        "name": f"{target[2]} {target[3]}",
                        "phone": target[4]
                    },
                    level=new_level,
                    channels=[NotificationChannel.EMAIL, NotificationChannel.DASHBOARD]
                )
            
            return True
            
        except Exception as e:
            logger.error(f"Error manually escalating alert: {e}")
            self.db.rollback()
            return False
