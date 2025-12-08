"""
Notification Engine for Followup Autopilot

Manages notification queue and dispatch:
- in_app: Always created
- push: For high+ priority tasks
- email/sms: For critical tasks

Delegates actual sending to existing AlertOrchestrationEngine
for SES (email) and Twilio (SMS) integration.

All notifications use wellness framing only.
"""

import os
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from uuid import uuid4

logger = logging.getLogger(__name__)


PRIORITY_CHANNELS = {
    "low": ["in_app"],
    "medium": ["in_app"],
    "high": ["in_app", "push"],
    "critical": ["in_app", "push", "email", "sms"],
}

NOTIFICATION_TEMPLATES = {
    "task_created": {
        "title_template": "New Wellness Task: {task_title}",
        "body_template": "{reason}\n\nTap to complete your check-in.",
    },
    "task_reminder": {
        "title_template": "Reminder: {task_title}",
        "body_template": "Don't forget to complete your wellness check-in today.",
    },
    "task_overdue": {
        "title_template": "Overdue: {task_title}",
        "body_template": "Your wellness check-in is overdue. Please complete it when you can.",
    },
    "wellness_alert": {
        "title_template": "Wellness Alert",
        "body_template": "{message}\n\nPlease contact your care team if you need assistance.",
    },
}


class NotificationEngine:
    """
    Notification creation and dispatch engine.
    
    Responsibilities:
    1. Create notifications based on task priority
    2. Queue notifications for delivery
    3. Dispatch via appropriate channels
    4. Track delivery status
    """
    
    def __init__(self, db_session=None):
        self.db = db_session
        self.logger = logging.getLogger(__name__)
    
    def create_task_notification(
        self,
        patient_id: str,
        task_data: Dict[str, Any]
    ) -> List[str]:
        """
        Create notifications for a new task.
        
        Args:
            patient_id: Patient identifier
            task_data: Task data dictionary
            
        Returns:
            List of created notification IDs
        """
        priority = task_data.get("priority", "medium")
        channels = PRIORITY_CHANNELS.get(priority, ["in_app"])
        
        task_title = (task_data.get("task_metadata") or {}).get("title", 
                      task_data.get("task_type", "Check-in"))
        reason = task_data.get("reason", "Please complete your wellness check-in.")
        
        template = NOTIFICATION_TEMPLATES["task_created"]
        title = template["title_template"].format(task_title=task_title)
        body = template["body_template"].format(reason=reason)
        
        notification_ids = []
        
        for channel in channels:
            notif_id = self._create_notification(
                patient_id=patient_id,
                channel=channel,
                title=title,
                body=body,
                priority=priority,
                related_task_id=task_data.get("id")
            )
            if notif_id:
                notification_ids.append(notif_id)
        
        return notification_ids
    
    def create_wellness_notification(
        self,
        patient_id: str,
        message: str,
        priority: str = "medium"
    ) -> List[str]:
        """Create a general wellness notification"""
        channels = PRIORITY_CHANNELS.get(priority, ["in_app"])
        
        template = NOTIFICATION_TEMPLATES["wellness_alert"]
        title = template["title_template"]
        body = template["body_template"].format(message=message)
        
        notification_ids = []
        
        for channel in channels:
            notif_id = self._create_notification(
                patient_id=patient_id,
                channel=channel,
                title=title,
                body=body,
                priority=priority
            )
            if notif_id:
                notification_ids.append(notif_id)
        
        return notification_ids
    
    def get_pending_notifications(
        self,
        patient_id: str,
        channel: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get pending notifications for a patient"""
        if self.db:
            from app.models.followup_autopilot_models import AutopilotNotification
            
            query = self.db.query(AutopilotNotification).filter(
                AutopilotNotification.patient_id == patient_id,
                AutopilotNotification.status == "pending"
            )
            
            if channel:
                query = query.filter(AutopilotNotification.channel == channel)
            
            rows = query.order_by(
                AutopilotNotification.created_at.desc()
            ).all()
            
            return [self._row_to_dict(row) for row in rows]
        
        return []
    
    def dispatch_pending_notifications(self) -> Dict[str, int]:
        """
        Dispatch all pending notifications.
        
        Returns:
            Dictionary with counts of sent/failed by channel
        """
        results = {"in_app": 0, "push": 0, "email": 0, "sms": 0, "failed": 0}
        
        if self.db:
            from app.models.followup_autopilot_models import AutopilotNotification
            
            pending = self.db.query(AutopilotNotification).filter(
                AutopilotNotification.status == "pending"
            ).limit(100).all()
            
            for notif in pending:
                success = self._dispatch_notification(notif)
                if success:
                    results[notif.channel] = results.get(notif.channel, 0) + 1
                    notif.status = "sent"
                    notif.sent_at = datetime.now(timezone.utc)
                else:
                    results["failed"] += 1
                    notif.status = "failed"
                    notif.error_message = "Dispatch failed"
            
            self.db.commit()
        
        return results
    
    def mark_notification_read(
        self,
        notification_id: str,
        patient_id: str
    ) -> bool:
        """Mark an in-app notification as read"""
        try:
            if self.db:
                from app.models.followup_autopilot_models import AutopilotNotification
                
                notif = self.db.query(AutopilotNotification).filter(
                    AutopilotNotification.id == notification_id,
                    AutopilotNotification.patient_id == patient_id
                ).first()
                
                if notif:
                    notif.status = "sent"
                    notif.sent_at = datetime.now(timezone.utc)
                    self.db.commit()
                    return True
            return False
        except Exception as e:
            self.logger.error(f"Failed to mark notification read: {e}")
            return False
    
    def _create_notification(
        self,
        patient_id: str,
        channel: str,
        title: str,
        body: str,
        priority: str,
        related_task_id: Optional[str] = None
    ) -> Optional[str]:
        """Create a single notification"""
        notif_id = str(uuid4())
        
        try:
            if self.db:
                from app.models.followup_autopilot_models import AutopilotNotification
                
                notif = AutopilotNotification(
                    id=notif_id,
                    patient_id=patient_id,
                    channel=channel,
                    title=title,
                    body=body,
                    priority=priority,
                    related_task_id=related_task_id,
                    status="pending"
                )
                self.db.add(notif)
                self.db.commit()
            else:
                self._create_notification_raw(
                    notif_id, patient_id, channel, title, body, priority, related_task_id
                )
            
            if channel == "in_app":
                self._dispatch_in_app(notif_id, patient_id, title, body)
            
            return notif_id
            
        except Exception as e:
            self.logger.error(f"Failed to create notification: {e}")
            if self.db:
                self.db.rollback()
            return None
    
    def _dispatch_notification(self, notif) -> bool:
        """Dispatch a single notification via appropriate channel"""
        try:
            if notif.channel == "in_app":
                return True
                
            elif notif.channel == "push":
                return self._send_push(notif.patient_id, notif.title, notif.body)
                
            elif notif.channel == "email":
                return self._send_email(notif.patient_id, notif.title, notif.body)
                
            elif notif.channel == "sms":
                return self._send_sms(notif.patient_id, notif.body)
                
            return False
        except Exception as e:
            self.logger.error(f"Notification dispatch failed: {e}")
            return False
    
    def _dispatch_in_app(
        self,
        notif_id: str,
        patient_id: str,
        title: str,
        body: str
    ) -> bool:
        """In-app notifications are already available via API"""
        return True
    
    def _send_push(self, patient_id: str, title: str, body: str) -> bool:
        """Send push notification (stubbed - integrate with push service)"""
        self.logger.info(f"Push notification to {patient_id}: {title}")
        return True
    
    def _send_email(self, patient_id: str, title: str, body: str) -> bool:
        """Send email via AlertOrchestrationEngine's SES client"""
        try:
            patient_email = self._get_patient_email(patient_id)
            if not patient_email:
                self.logger.warning(f"No email for patient {patient_id}")
                return False
            
            from app.services.alert_orchestration_engine import SES_AVAILABLE
            if SES_AVAILABLE:
                from app.services.alert_orchestration_engine import ses_client
                
                ses_client.send_email(
                    Source=os.environ.get("SES_FROM_EMAIL", "noreply@followupai.com"),
                    Destination={"ToAddresses": [patient_email]},
                    Message={
                        "Subject": {"Data": title, "Charset": "UTF-8"},
                        "Body": {"Text": {"Data": body, "Charset": "UTF-8"}}
                    }
                )
                return True
            else:
                self.logger.info(f"Email (SES unavailable) to {patient_id}: {title}")
                return True
                
        except Exception as e:
            self.logger.error(f"Email send failed: {e}")
            return False
    
    def _send_sms(self, patient_id: str, body: str) -> bool:
        """Send SMS via AlertOrchestrationEngine's Twilio client"""
        try:
            patient_phone = self._get_patient_phone(patient_id)
            if not patient_phone:
                self.logger.warning(f"No phone for patient {patient_id}")
                return False
            
            from app.services.alert_orchestration_engine import TWILIO_AVAILABLE
            if TWILIO_AVAILABLE:
                from app.services.alert_orchestration_engine import twilio_client
                
                twilio_client.messages.create(
                    body=body,
                    from_=os.environ.get("TWILIO_PHONE_NUMBER"),
                    to=patient_phone
                )
                return True
            else:
                self.logger.info(f"SMS (Twilio unavailable) to {patient_id}: {body[:50]}...")
                return True
                
        except Exception as e:
            self.logger.error(f"SMS send failed: {e}")
            return False
    
    def _get_patient_email(self, patient_id: str) -> Optional[str]:
        """Get patient email address"""
        if self.db:
            try:
                from app.models.user import User
                user = self.db.query(User).filter(User.id == patient_id).first()
                return user.email if user else None
            except:
                pass
        return None
    
    def _get_patient_phone(self, patient_id: str) -> Optional[str]:
        """Get patient phone number"""
        if self.db:
            try:
                from app.models.user import User
                user = self.db.query(User).filter(User.id == patient_id).first()
                return user.phone if user and hasattr(user, 'phone') else None
            except:
                pass
        return None
    
    def _row_to_dict(self, row) -> Dict[str, Any]:
        """Convert database row to dictionary"""
        return {
            "id": str(row.id),
            "patient_id": row.patient_id,
            "channel": row.channel,
            "title": row.title,
            "body": row.body,
            "priority": row.priority,
            "status": row.status,
            "related_task_id": str(row.related_task_id) if row.related_task_id else None,
            "created_at": row.created_at.isoformat() if row.created_at else None,
            "sent_at": row.sent_at.isoformat() if row.sent_at else None,
        }
    
    def _create_notification_raw(
        self,
        notif_id: str,
        patient_id: str,
        channel: str,
        title: str,
        body: str,
        priority: str,
        related_task_id: Optional[str]
    ):
        """Direct database insert when ORM not available"""
        import psycopg2
        
        conn_str = os.environ.get("DATABASE_URL")
        if not conn_str:
            return
            
        with psycopg2.connect(conn_str) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO autopilot_notifications 
                    (id, patient_id, channel, title, body, priority, 
                     related_task_id, status, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, 'pending', NOW())
                """, (notif_id, patient_id, channel, title, body, priority, related_task_id))
            conn.commit()
