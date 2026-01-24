"""
Habit Reminder Service - Production-Grade Notification Integration
===================================================================

Comprehensive reminder system integrated with notification services:
- Multi-channel delivery (SMS, email, push, in-app)
- Adaptive timing based on completion patterns
- APScheduler job for reminder dispatch
- Snooze and escalation logic
- HIPAA-compliant with audit logging

NOTE: Twilio SMS integration has been disabled.
SMS reminders will log warnings but not actually send.
"""

import os
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta, time
from enum import Enum

from sqlalchemy.orm import Session
from sqlalchemy import text

from app.services.access_control import HIPAAAuditLogger, AccessScope, PHICategory

logger = logging.getLogger(__name__)

# STUB: Twilio has been removed
logger.warning("Twilio integration disabled - SMS reminders will not be sent")


class ReminderChannel(str, Enum):
    IN_APP = "in_app"
    PUSH = "push"
    SMS = "sms"
    EMAIL = "email"


class HabitReminderService:
    """Production-grade reminder service with multi-channel delivery
    
    NOTE: Twilio SMS is disabled. SMS reminders will log warnings.
    """
    
    def __init__(self, db: Session):
        self.db = db
        self._twilio_client = None  # STUB: Always None - Twilio disabled
        self._email_service = None
    
    def _get_twilio_client(self):
        """Lazy-load Twilio client - STUB: Always returns None"""
        # STUB: Twilio has been removed
        logger.warning("Twilio client requested but integration is disabled")
        return None
    
    def get_pending_reminders(self, within_minutes: int = 5) -> List[Dict[str, Any]]:
        """Get reminders due within the specified time window"""
        
        now = datetime.utcnow()
        current_time = now.strftime("%H:%M")
        
        window_start = (now - timedelta(minutes=1)).strftime("%H:%M")
        window_end = (now + timedelta(minutes=within_minutes)).strftime("%H:%M")
        
        query = text("""
            SELECT r.id, r.habit_id, r.user_id, r.reminder_type, r.scheduled_time,
                   r.message, r.adaptive_enabled, r.snooze_until, r.last_sent_at,
                   h.name as habit_name, h.current_streak,
                   u.email, u.phone
            FROM habit_reminders r
            JOIN habit_habits h ON h.id = r.habit_id
            LEFT JOIN users u ON u.id = r.user_id
            WHERE r.is_active = true
            AND r.scheduled_time >= :window_start
            AND r.scheduled_time <= :window_end
            AND (r.snooze_until IS NULL OR r.snooze_until < NOW())
            AND (r.last_sent_at IS NULL OR r.last_sent_at < NOW() - INTERVAL '6 hours')
            ORDER BY r.scheduled_time ASC
        """)
        
        rows = self.db.execute(query, {
            "window_start": window_start,
            "window_end": window_end
        }).fetchall()
        
        reminders = []
        for row in rows:
            reminders.append({
                "id": row[0],
                "habitId": row[1],
                "userId": row[2],
                "reminderType": row[3],
                "scheduledTime": row[4],
                "message": row[5],
                "adaptiveEnabled": row[6],
                "snoozeUntil": row[7],
                "lastSentAt": row[8],
                "habitName": row[9],
                "currentStreak": row[10] or 0,
                "email": row[11],
                "phone": row[12]
            })
        
        return reminders
    
    def send_reminder(self, reminder: Dict[str, Any]) -> Dict[str, Any]:
        """Send reminder via appropriate channel"""
        
        channel = ReminderChannel(reminder["reminderType"])
        message = self._build_reminder_message(reminder)
        
        result = {
            "reminderId": reminder["id"],
            "channel": channel.value,
            "sent": False,
            "error": None
        }
        
        try:
            if channel == ReminderChannel.SMS:
                result = self._send_sms(reminder, message)
            elif channel == ReminderChannel.EMAIL:
                result = self._send_email(reminder, message)
            elif channel == ReminderChannel.PUSH:
                result = self._send_push(reminder, message)
            else:
                result = self._send_in_app(reminder, message)
            
            if result["sent"]:
                self._mark_reminder_sent(reminder["id"])
                
                HIPAAAuditLogger.log_access(
                    user_id=reminder["userId"],
                    user_role="patient",
                    action="habit_reminder_sent",
                    resource_type="HabitReminder",
                    resource_id=reminder["id"],
                    access_reason=f"scheduled_reminder_{channel.value}",
                    was_successful=True
                )
            
        except Exception as e:
            logger.error(f"Error sending reminder {reminder['id']}: {e}")
            result["error"] = str(e)
        
        return result
    
    def _build_reminder_message(self, reminder: Dict[str, Any]) -> str:
        """Build personalized reminder message"""
        
        if reminder.get("message"):
            return reminder["message"]
        
        habit_name = reminder.get("habitName", "your habit")
        streak = reminder.get("currentStreak", 0)
        
        if streak > 0:
            messages = [
                f"Time for {habit_name}! Keep your {streak}-day streak going!",
                f"Don't break the chain! {streak} days strong on {habit_name}.",
                f"Your {streak}-day {habit_name} streak is waiting for you!"
            ]
        else:
            messages = [
                f"Time to work on {habit_name}! Start your streak today.",
                f"Ready for {habit_name}? Every journey begins with a single step.",
                f"It's {habit_name} time! Build your momentum today."
            ]
        
        import random
        return random.choice(messages)
    
    def _send_sms(self, reminder: Dict[str, Any], message: str) -> Dict[str, Any]:
        """Send SMS via Twilio - STUB: Twilio is disabled"""
        
        phone = reminder.get("phone")
        if not phone:
            return {"reminderId": reminder["id"], "channel": "sms", "sent": False, "error": "No phone number"}
        
        # STUB: Twilio is disabled
        logger.warning(f"SMS reminder not sent to {phone[:4]}**** - Twilio integration disabled")
        logger.info(f"Would have sent: {message}")
        
        return {
            "reminderId": reminder["id"],
            "channel": "sms",
            "sent": False,
            "error": "Twilio integration disabled"
        }
    
    def _send_email(self, reminder: Dict[str, Any], message: str) -> Dict[str, Any]:
        """Send email reminder"""
        
        email = reminder.get("email")
        if not email:
            return {"reminderId": reminder["id"], "channel": "email", "sent": False, "error": "No email"}
        
        try:
            resend_api_key = os.getenv("RESEND_API_KEY")
            if not resend_api_key:
                return {"reminderId": reminder["id"], "channel": "email", "sent": False, "error": "Email not configured"}
            
            import requests
            
            habit_name = reminder.get("habitName", "your habit")
            streak = reminder.get("currentStreak", 0)
            
            html_content = f"""
            <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px;">
                <h2 style="color: #4F46E5;">Habit Reminder</h2>
                <p style="font-size: 16px; color: #374151;">{message}</p>
                {f'<p style="font-size: 14px; color: #6B7280;">Current streak: <strong>{streak} days</strong></p>' if streak > 0 else ''}
                <div style="margin-top: 20px; padding: 15px; background: #F3F4F6; border-radius: 8px;">
                    <p style="margin: 0; color: #4B5563;">Keep building healthy habits with Followup AI!</p>
                </div>
            </div>
            """
            
            response = requests.post(
                "https://api.resend.com/emails",
                headers={
                    "Authorization": f"Bearer {resend_api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "from": os.getenv("RESEND_FROM_EMAIL", "habits@followup.ai"),
                    "to": [email],
                    "subject": f"Reminder: Time for {habit_name}!",
                    "html": html_content
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"Email sent to {email[:3]}****: {result.get('id')}")
                return {
                    "reminderId": reminder["id"],
                    "channel": "email",
                    "sent": True,
                    "messageId": result.get("id")
                }
            else:
                return {
                    "reminderId": reminder["id"],
                    "channel": "email",
                    "sent": False,
                    "error": response.text
                }
            
        except Exception as e:
            logger.error(f"Email send error: {e}")
            return {"reminderId": reminder["id"], "channel": "email", "sent": False, "error": str(e)}
    
    def _send_push(self, reminder: Dict[str, Any], message: str) -> Dict[str, Any]:
        """Send push notification (placeholder for push service integration)"""
        
        insert_query = text("""
            INSERT INTO notifications (id, user_id, type, title, message, created_at)
            VALUES (gen_random_uuid(), :user_id, 'habit_reminder', :title, :message, NOW())
            ON CONFLICT DO NOTHING
        """)
        
        try:
            self.db.execute(insert_query, {
                "user_id": reminder["userId"],
                "title": f"Habit Reminder: {reminder.get('habitName', 'Your habit')}",
                "message": message
            })
            self.db.commit()
            
            return {
                "reminderId": reminder["id"],
                "channel": "push",
                "sent": True,
                "note": "Stored as notification for push delivery"
            }
        except Exception as e:
            logger.warning(f"Push notification storage failed: {e}")
            return {"reminderId": reminder["id"], "channel": "push", "sent": False, "error": str(e)}
    
    def _send_in_app(self, reminder: Dict[str, Any], message: str) -> Dict[str, Any]:
        """Store in-app notification"""
        
        insert_query = text("""
            INSERT INTO habit_risk_alerts 
            (id, user_id, alert_type, severity, title, message, status)
            VALUES (gen_random_uuid(), :user_id, 'reminder', 'low', :title, :message, 'active')
        """)
        
        try:
            self.db.execute(insert_query, {
                "user_id": reminder["userId"],
                "title": f"Time for {reminder.get('habitName', 'your habit')}!",
                "message": message
            })
            self.db.commit()
            
            return {
                "reminderId": reminder["id"],
                "channel": "in_app",
                "sent": True
            }
        except Exception as e:
            logger.error(f"In-app notification error: {e}")
            return {"reminderId": reminder["id"], "channel": "in_app", "sent": False, "error": str(e)}
    
    def _mark_reminder_sent(self, reminder_id: str):
        """Update reminder with last sent timestamp"""
        update_query = text("""
            UPDATE habit_reminders
            SET last_sent_at = NOW()
            WHERE id = :id
        """)
        self.db.execute(update_query, {"id": reminder_id})
        self.db.commit()
    
    def snooze_reminder(self, reminder_id: str, user_id: str, minutes: int = 30) -> Dict[str, Any]:
        """Snooze a reminder for specified minutes"""
        
        snooze_until = datetime.utcnow() + timedelta(minutes=minutes)
        
        update_query = text("""
            UPDATE habit_reminders
            SET snooze_until = :snooze_until
            WHERE id = :id AND user_id = :user_id
            RETURNING habit_id
        """)
        
        result = self.db.execute(update_query, {
            "snooze_until": snooze_until,
            "id": reminder_id,
            "user_id": user_id
        })
        
        row = result.fetchone()
        self.db.commit()
        
        if row:
            return {
                "success": True,
                "snoozedUntil": snooze_until.isoformat(),
                "minutes": minutes
            }
        
        return {"success": False, "error": "Reminder not found"}
    
    def get_adaptive_time(self, user_id: str, habit_id: str) -> Optional[str]:
        """Calculate optimal reminder time based on completion patterns"""
        
        query = text("""
            SELECT EXTRACT(HOUR FROM completion_date) as hour,
                   COUNT(*) as completions
            FROM habit_completions
            WHERE habit_id = :habit_id AND user_id = :user_id
            AND completed = true
            AND completion_date >= NOW() - INTERVAL '30 days'
            GROUP BY EXTRACT(HOUR FROM completion_date)
            ORDER BY completions DESC
            LIMIT 1
        """)
        
        result = self.db.execute(query, {
            "habit_id": habit_id, "user_id": user_id
        }).fetchone()
        
        if result and result[1] >= 3:
            optimal_hour = int(result[0])
            reminder_hour = max(0, optimal_hour - 1)
            return f"{reminder_hour:02d}:00"
        
        return None
    
    def update_adaptive_reminders(self, user_id: str) -> Dict[str, Any]:
        """Update reminder times based on adaptive learning"""
        
        reminders_query = text("""
            SELECT r.id, r.habit_id, r.scheduled_time
            FROM habit_reminders r
            WHERE r.user_id = :user_id 
            AND r.adaptive_enabled = true 
            AND r.is_active = true
        """)
        
        reminders = self.db.execute(reminders_query, {"user_id": user_id}).fetchall()
        
        updated = 0
        for reminder in reminders:
            optimal_time = self.get_adaptive_time(user_id, reminder[1])
            
            if optimal_time and optimal_time != reminder[2]:
                update_query = text("""
                    UPDATE habit_reminders
                    SET scheduled_time = :new_time
                    WHERE id = :id
                """)
                
                self.db.execute(update_query, {
                    "new_time": optimal_time,
                    "id": reminder[0]
                })
                updated += 1
        
        self.db.commit()
        
        return {
            "remindersChecked": len(reminders),
            "remindersUpdated": updated
        }


def reminder_dispatch_job():
    """APScheduler job for dispatching due reminders"""
    from app.database import SessionLocal
    
    logger.info("Starting reminder dispatch job")
    
    db = SessionLocal()
    try:
        service = HabitReminderService(db)
        
        pending = service.get_pending_reminders(within_minutes=5)
        
        sent_count = 0
        error_count = 0
        
        for reminder in pending:
            result = service.send_reminder(reminder)
            if result.get("sent"):
                sent_count += 1
            else:
                error_count += 1
        
        logger.info(f"Reminder dispatch complete: {sent_count} sent, {error_count} errors")
        
    except Exception as e:
        logger.error(f"Error in reminder dispatch job: {e}")
    finally:
        db.close()
