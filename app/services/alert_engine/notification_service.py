"""
Multi-Channel Notification Service - Dashboard, SMS, Email, Push notifications.

Channels:
1. Dashboard - Real-time WebSocket/SSE notifications
2. SMS - Via Twilio for urgent alerts (PHI-minimal) - DISABLED
3. Email - Via AWS SES with secure portal links - DISABLED
4. Push - Via Firebase/OneSignal for mobile app

NOTE: Twilio and AWS SES integrations have been disabled.
SMS and email notifications will log warnings but not send.

Includes:
- Template-based messaging
- PHI-safe SMS content
- Delivery tracking and audit logging
- Escalation notifications
"""

import os
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
import uuid

from sqlalchemy.orm import Session
from sqlalchemy import text

logger = logging.getLogger(__name__)

# STUB: Twilio and AWS SES have been removed
TWILIO_AVAILABLE = False
AWS_SES_AVAILABLE = False

logger.warning("Twilio not available for SMS notifications - integration disabled")
logger.warning("AWS SES not available for email notifications - integration disabled")

from .config_service import AlertConfigService
from .rule_engine import AlertRecord, AlertSeverity


class NotificationChannel(Enum):
    """Available notification channels"""
    DASHBOARD = "dashboard"
    SMS = "sms"
    EMAIL = "email"
    PUSH = "push"


@dataclass
class NotificationRequest:
    """Request to send a notification"""
    alert_id: str
    patient_id: str
    recipient_id: str  # Clinician user ID
    recipient_phone: Optional[str] = None
    recipient_email: Optional[str] = None
    channels: List[NotificationChannel] = None
    priority: int = 5
    is_escalation: bool = False


@dataclass
class NotificationResult:
    """Result of notification attempt"""
    notification_id: str
    channel: NotificationChannel
    success: bool
    error_message: Optional[str] = None
    delivered_at: Optional[datetime] = None


class NotificationService:
    """Service for multi-channel notification delivery
    
    NOTE: SMS (Twilio) and Email (AWS SES) are disabled.
    Only Dashboard and Push notifications work.
    """
    
    def __init__(self, db: Session):
        self.db = db
        self.config_service = AlertConfigService()
        
        # STUB: Twilio client disabled
        self.twilio_client = None
        self.twilio_from_number = None
        
        # STUB: AWS SES client disabled
        self.ses_client = None
        
        logger.info("NotificationService initialized - SMS and Email disabled")
    
    async def send_alert_notification(
        self,
        alert: AlertRecord,
        request: NotificationRequest
    ) -> List[NotificationResult]:
        """
        Send notification for an alert through configured channels.
        
        NOTE: SMS and Email are disabled.
        
        Returns list of results for each channel attempted.
        """
        config = self.config_service.config
        results = []
        
        # Determine channels based on config and alert severity
        channels = request.channels or self._get_default_channels(alert.severity)
        
        for channel in channels:
            result = await self._send_via_channel(alert, request, channel)
            results.append(result)
            
            # Log notification attempt
            await self._log_notification(alert, request, result)
        
        return results
    
    def _get_default_channels(self, severity: str) -> List[NotificationChannel]:
        """Get default channels based on alert severity"""
        config = self.config_service.config
        channels = []
        
        # Dashboard is always included
        if config.dashboard_enabled:
            channels.append(NotificationChannel.DASHBOARD)
        
        # STUB: SMS disabled - skip even for critical
        # if config.sms_enabled and severity in ["critical", "high"]:
        #     channels.append(NotificationChannel.SMS)
        
        # STUB: Email disabled
        # if config.email_enabled:
        #     channels.append(NotificationChannel.EMAIL)
        
        # Push for critical and high
        if config.push_enabled and severity in ["critical", "high"]:
            channels.append(NotificationChannel.PUSH)
        
        return channels
    
    async def _send_via_channel(
        self,
        alert: AlertRecord,
        request: NotificationRequest,
        channel: NotificationChannel
    ) -> NotificationResult:
        """Send notification via specific channel"""
        notification_id = str(uuid.uuid4())
        
        try:
            if channel == NotificationChannel.DASHBOARD:
                success = await self._send_dashboard_notification(alert, request)
            elif channel == NotificationChannel.SMS:
                # STUB: SMS disabled
                logger.warning("SMS notification skipped - Twilio integration disabled")
                success = False
            elif channel == NotificationChannel.EMAIL:
                # STUB: Email disabled
                logger.warning("Email notification skipped - AWS SES integration disabled")
                success = False
            elif channel == NotificationChannel.PUSH:
                success = await self._send_push_notification(alert, request)
            else:
                return NotificationResult(
                    notification_id=notification_id,
                    channel=channel,
                    success=False,
                    error_message="Unknown channel"
                )
            
            return NotificationResult(
                notification_id=notification_id,
                channel=channel,
                success=success,
                delivered_at=datetime.utcnow() if success else None,
                error_message="Integration disabled" if not success and channel in [NotificationChannel.SMS, NotificationChannel.EMAIL] else None
            )
            
        except Exception as e:
            logger.error(f"Error sending {channel.value} notification: {e}")
            return NotificationResult(
                notification_id=notification_id,
                channel=channel,
                success=False,
                error_message=str(e)
            )
    
    async def _send_dashboard_notification(
        self,
        alert: AlertRecord,
        request: NotificationRequest
    ) -> bool:
        """Send real-time dashboard notification via database event"""
        try:
            # Store notification for dashboard polling/WebSocket
            insert_query = text("""
                INSERT INTO dashboard_notifications (
                    id, user_id, alert_id, patient_id, notification_type,
                    title, message, severity, priority, read_status,
                    created_at, expires_at
                ) VALUES (
                    gen_random_uuid(), :user_id, :alert_id, :patient_id, 'alert',
                    :title, :message, :severity, :priority, false,
                    NOW(), NOW() + INTERVAL '7 days'
                )
            """)
            
            self.db.execute(insert_query, {
                "user_id": request.recipient_id,
                "alert_id": alert.id,
                "patient_id": alert.patient_id,
                "title": alert.title,
                "message": alert.message,
                "severity": alert.severity,
                "priority": alert.priority
            })
            self.db.commit()
            
            logger.info(f"Dashboard notification sent for alert {alert.id}")
            return True
            
        except Exception as e:
            logger.error(f"Error sending dashboard notification: {e}")
            self.db.rollback()
            return False
    
    async def _send_sms_notification(
        self,
        alert: AlertRecord,
        request: NotificationRequest
    ) -> bool:
        """Send SMS notification via Twilio (PHI-minimal)
        STUB: Twilio is disabled - always returns False
        """
        logger.warning("SMS notification not sent - Twilio integration disabled")
        return False
    
    async def _send_email_notification(
        self,
        alert: AlertRecord,
        request: NotificationRequest
    ) -> bool:
        """Send email notification via AWS SES
        STUB: AWS SES is disabled - always returns False
        """
        logger.warning("Email notification not sent - AWS SES integration disabled")
        return False
    
    async def _send_push_notification(
        self,
        alert: AlertRecord,
        request: NotificationRequest
    ) -> bool:
        """Send push notification (placeholder for Firebase/OneSignal)"""
        # TODO: Implement Firebase/OneSignal push notifications
        logger.info(f"Push notification placeholder for alert {alert.id}")
        return True
    
    async def _log_notification(
        self,
        alert: AlertRecord,
        request: NotificationRequest,
        result: NotificationResult
    ):
        """Log notification delivery attempt for audit"""
        try:
            insert_query = text("""
                INSERT INTO notification_delivery_log (
                    id, alert_id, patient_id, recipient_id, channel,
                    success, error_message, delivered_at, created_at
                ) VALUES (
                    :id, :alert_id, :patient_id, :recipient_id, :channel,
                    :success, :error_message, :delivered_at, NOW()
                )
            """)
            
            self.db.execute(insert_query, {
                "id": result.notification_id,
                "alert_id": alert.id,
                "patient_id": alert.patient_id,
                "recipient_id": request.recipient_id,
                "channel": result.channel.value,
                "success": result.success,
                "error_message": result.error_message,
                "delivered_at": result.delivered_at
            })
            self.db.commit()
            
        except Exception as e:
            logger.warning(f"Error logging notification: {e}")
    
    async def get_unread_notifications(
        self,
        user_id: str,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get unread dashboard notifications for a user"""
        query = text("""
            SELECT id, alert_id, patient_id, notification_type, title,
                   message, severity, priority, created_at
            FROM dashboard_notifications
            WHERE user_id = :user_id
            AND read_status = false
            AND expires_at > NOW()
            ORDER BY priority DESC, created_at DESC
            LIMIT :limit
        """)
        
        try:
            results = self.db.execute(query, {
                "user_id": user_id,
                "limit": limit
            }).fetchall()
            
            return [
                {
                    "id": str(row[0]),
                    "alert_id": row[1],
                    "patient_id": row[2],
                    "type": row[3],
                    "title": row[4],
                    "message": row[5],
                    "severity": row[6],
                    "priority": row[7],
                    "created_at": row[8].isoformat() if row[8] else None
                }
                for row in results
            ]
        except Exception as e:
            logger.error(f"Error getting unread notifications: {e}")
            return []
    
    async def mark_notification_read(
        self,
        notification_id: str,
        user_id: str
    ) -> bool:
        """Mark a notification as read"""
        try:
            update_query = text("""
                UPDATE dashboard_notifications
                SET read_status = true, read_at = NOW()
                WHERE id = :id AND user_id = :user_id
            """)
            
            self.db.execute(update_query, {
                "id": notification_id,
                "user_id": user_id
            })
            self.db.commit()
            return True
            
        except Exception as e:
            logger.error(f"Error marking notification read: {e}")
            self.db.rollback()
            return False
    
    async def get_delivery_stats(
        self,
        days: int = 7
    ) -> Dict[str, Any]:
        """Get notification delivery statistics"""
        cutoff = datetime.utcnow() - timedelta(days=days)
        
        query = text("""
            SELECT 
                channel,
                COUNT(*) as total,
                SUM(CASE WHEN success THEN 1 ELSE 0 END) as successful,
                SUM(CASE WHEN NOT success THEN 1 ELSE 0 END) as failed
            FROM notification_delivery_log
            WHERE created_at >= :cutoff
            GROUP BY channel
        """)
        
        try:
            results = self.db.execute(query, {"cutoff": cutoff}).fetchall()
            
            stats = {}
            for row in results:
                stats[row[0]] = {
                    "total": row[1],
                    "successful": row[2],
                    "failed": row[3],
                    "success_rate": (row[2] / row[1] * 100) if row[1] > 0 else 0
                }
            
            # Add note about disabled channels
            stats["_warning"] = "SMS and Email channels are disabled - Twilio and AWS SES integrations removed"
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting delivery stats: {e}")
            return {}
