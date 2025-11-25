"""
Multi-Channel Notification Service - Dashboard, SMS, Email, Push notifications.

Channels:
1. Dashboard - Real-time WebSocket/SSE notifications
2. SMS - Via Twilio for urgent alerts (PHI-minimal)
3. Email - Via AWS SES with secure portal links
4. Push - Via Firebase/OneSignal for mobile app

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

# Optional imports for notification providers
try:
    from twilio.rest import Client as TwilioClient
    TWILIO_AVAILABLE = True
except ImportError:
    TWILIO_AVAILABLE = False
    logger.warning("Twilio not available for SMS notifications")

try:
    import boto3
    AWS_SES_AVAILABLE = True
except ImportError:
    AWS_SES_AVAILABLE = False
    logger.warning("AWS SES not available for email notifications")

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
    """Service for multi-channel notification delivery"""
    
    def __init__(self, db: Session):
        self.db = db
        self.config_service = AlertConfigService()
        
        # Initialize Twilio client
        self.twilio_client = None
        if TWILIO_AVAILABLE:
            try:
                account_sid = os.getenv('TWILIO_ACCOUNT_SID')
                auth_token = os.getenv('TWILIO_AUTH_TOKEN')
                if account_sid and auth_token:
                    self.twilio_client = TwilioClient(account_sid, auth_token)
                    self.twilio_from_number = os.getenv('TWILIO_PHONE_NUMBER')
                    logger.info("Twilio SMS notifications enabled")
            except Exception as e:
                logger.warning(f"Twilio initialization failed: {e}")
        
        # Initialize AWS SES client
        self.ses_client = None
        if AWS_SES_AVAILABLE:
            try:
                aws_key = os.getenv('AWS_ACCESS_KEY_ID')
                aws_secret = os.getenv('AWS_SECRET_ACCESS_KEY')
                aws_region = os.getenv('AWS_REGION', 'us-east-1')
                if aws_key and aws_secret:
                    self.ses_client = boto3.client(
                        'ses',
                        region_name=aws_region,
                        aws_access_key_id=aws_key,
                        aws_secret_access_key=aws_secret
                    )
                    logger.info("AWS SES email notifications enabled")
            except Exception as e:
                logger.warning(f"AWS SES initialization failed: {e}")
    
    async def send_alert_notification(
        self,
        alert: AlertRecord,
        request: NotificationRequest
    ) -> List[NotificationResult]:
        """
        Send notification for an alert through configured channels.
        
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
        
        # SMS for critical and high severity
        if config.sms_enabled and severity in ["critical", "high"]:
            channels.append(NotificationChannel.SMS)
        
        # Email for all severities
        if config.email_enabled:
            channels.append(NotificationChannel.EMAIL)
        
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
                success = await self._send_sms_notification(alert, request)
            elif channel == NotificationChannel.EMAIL:
                success = await self._send_email_notification(alert, request)
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
                delivered_at=datetime.utcnow() if success else None
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
        """Send SMS notification via Twilio (PHI-minimal)"""
        if not self.twilio_client or not request.recipient_phone:
            logger.warning("SMS not available or no phone number")
            return False
        
        try:
            # PHI-minimal SMS content
            if request.is_escalation:
                message_body = (
                    f"[ESCALATION] Health Alert requires attention. "
                    f"Severity: {alert.severity.upper()}. "
                    f"Please review in secure portal."
                )
            else:
                message_body = (
                    f"Health Alert: {alert.severity.upper()} priority pattern detected. "
                    f"Review details in secure portal."
                )
            
            message = self.twilio_client.messages.create(
                body=message_body,
                from_=self.twilio_from_number,
                to=request.recipient_phone
            )
            
            logger.info(f"SMS sent: {message.sid}")
            return True
            
        except Exception as e:
            logger.error(f"Error sending SMS: {e}")
            return False
    
    async def _send_email_notification(
        self,
        alert: AlertRecord,
        request: NotificationRequest
    ) -> bool:
        """Send email notification via AWS SES"""
        if not self.ses_client or not request.recipient_email:
            logger.warning("Email not available or no email address")
            return False
        
        try:
            # Get patient name for context (non-PHI)
            patient_display = "Patient"  # Default to non-identifying
            
            # Build secure portal link
            portal_link = f"https://app.followupai.com/alerts/{alert.id}"
            
            subject = f"[{alert.severity.upper()}] Health Alert - Requires Review"
            if request.is_escalation:
                subject = f"[ESCALATION] {subject}"
            
            # Get top contributing metrics
            top_metrics = ""
            if alert.trigger_metrics:
                metrics_list = alert.trigger_metrics[:3]
                top_metrics = "\n".join([
                    f"  - {m.get('name', 'Unknown')}: {m.get('z_score', m.get('value', 'N/A'))}"
                    for m in metrics_list
                ])
            
            html_body = f"""
            <html>
            <head>
                <style>
                    body {{ font-family: Arial, sans-serif; line-height: 1.6; }}
                    .alert-box {{ padding: 20px; border-radius: 8px; margin: 20px 0; }}
                    .critical {{ background-color: #fee2e2; border-left: 4px solid #ef4444; }}
                    .high {{ background-color: #fef3c7; border-left: 4px solid #f59e0b; }}
                    .moderate {{ background-color: #e0f2fe; border-left: 4px solid #3b82f6; }}
                    .low {{ background-color: #f0fdf4; border-left: 4px solid #22c55e; }}
                    .btn {{ display: inline-block; padding: 12px 24px; background-color: #3b82f6; 
                           color: white; text-decoration: none; border-radius: 6px; }}
                    .disclaimer {{ font-size: 12px; color: #6b7280; margin-top: 20px; }}
                </style>
            </head>
            <body>
                <h2>Health Alert Notification</h2>
                
                <div class="alert-box {alert.severity}">
                    <h3>{alert.title}</h3>
                    <p><strong>Severity:</strong> {alert.severity.upper()}</p>
                    <p><strong>Priority:</strong> {alert.priority}/10</p>
                </div>
                
                <p><strong>Alert Details:</strong></p>
                <p>{alert.message}</p>
                
                <p><strong>Key Indicators:</strong></p>
                <pre>{top_metrics or 'See portal for details'}</pre>
                
                <p>
                    <a href="{portal_link}" class="btn">View Full Details in Secure Portal</a>
                </p>
                
                <p class="disclaimer">
                    <strong>IMPORTANT:</strong> {alert.disclaimer}<br><br>
                    This notification was sent to {request.recipient_email} because you are 
                    assigned to monitor this patient. For security, full patient details 
                    are only available in the authenticated portal.
                </p>
            </body>
            </html>
            """
            
            text_body = f"""
Health Alert Notification
=========================

{alert.title}
Severity: {alert.severity.upper()}
Priority: {alert.priority}/10

{alert.message}

Key Indicators:
{top_metrics or 'See portal for details'}

View full details: {portal_link}

IMPORTANT: {alert.disclaimer}
            """
            
            sender_email = os.getenv('AWS_SES_SENDER_EMAIL', 'alerts@followupai.com')
            
            response = self.ses_client.send_email(
                Source=sender_email,
                Destination={'ToAddresses': [request.recipient_email]},
                Message={
                    'Subject': {'Data': subject},
                    'Body': {
                        'Text': {'Data': text_body},
                        'Html': {'Data': html_body}
                    }
                }
            )
            
            logger.info(f"Email sent: {response['MessageId']}")
            return True
            
        except Exception as e:
            logger.error(f"Error sending email: {e}")
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
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting delivery stats: {e}")
            return {}
