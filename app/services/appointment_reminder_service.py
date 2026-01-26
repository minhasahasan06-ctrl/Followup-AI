"""
Appointment Reminder Service

NOTE: AWS SES and Twilio integrations have been disabled.
Email and SMS reminders will log warnings but not actually send.
"""

from datetime import datetime, timedelta
from typing import List, Dict
from sqlalchemy.orm import Session
from app.models.appointment import Appointment
from app.models.user import User
from app.config import settings
import logging

logger = logging.getLogger(__name__)

# STUB: boto3 and twilio have been removed
logger.warning("AWS SES integration disabled - email reminders will not be sent")
logger.warning("Twilio integration disabled - SMS reminders will not be sent")


class AppointmentReminderService:
    """
    Appointment reminder service.
    
    NOTE: Email (AWS SES) and SMS (Twilio) are disabled.
    All reminder operations will log warnings but not actually send.
    """
    
    def __init__(self, db: Session):
        self.db = db
        
        # STUB: Twilio client disabled
        self.twilio_client = None
        
        # STUB: SES client disabled
        self.ses_client = None
    
    def get_upcoming_appointments(self) -> List[Appointment]:
        tomorrow = datetime.utcnow() + timedelta(hours=24)
        window_start = tomorrow - timedelta(minutes=30)
        window_end = tomorrow + timedelta(minutes=30)
        
        return self.db.query(Appointment).filter(
            Appointment.appointment_date >= window_start,
            Appointment.appointment_date <= window_end,
            Appointment.status == "scheduled",
            Appointment.reminder_sent == False
        ).all()
    
    def send_sms_reminder(self, phone_number: str, appointment: Appointment) -> bool:
        """
        Send SMS reminder via Twilio.
        STUB: Twilio is disabled - logs warning and returns False.
        """
        logger.warning(f"SMS reminder not sent to {phone_number[:4]}**** - Twilio integration disabled")
        logger.info(f"Would have sent: Reminder for appointment at {appointment.appointment_date.strftime('%I:%M %p')}")
        return False
    
    def send_email_reminder(self, email: str, appointment: Appointment) -> bool:
        """
        Send email reminder via AWS SES.
        STUB: AWS SES is disabled - logs warning and returns False.
        """
        logger.warning(f"Email reminder not sent to {email} - AWS SES integration disabled")
        logger.info(f"Would have sent reminder for appointment on {appointment.appointment_date.strftime('%B %d, %Y')}")
        return False
    
    def process_reminders(self) -> Dict:
        """
        Process all pending reminders.
        
        NOTE: Actual sending is disabled, but appointments will still be marked as reminded
        to prevent repeated processing.
        """
        appointments = self.get_upcoming_appointments()
        
        sms_attempted = 0
        email_attempted = 0
        
        for appointment in appointments:
            patient = self.db.query(User).filter(User.id == appointment.patient_id).first()
            
            if not patient:
                continue
            
            if patient.phone_number:
                self.send_sms_reminder(patient.phone_number, appointment)
                sms_attempted += 1
            
            if patient.email:
                self.send_email_reminder(patient.email, appointment)
                email_attempted += 1
            
            # Mark as reminded even though sending is disabled
            # to prevent repeated processing
            appointment.reminder_sent = True
        
        if appointments:
            self.db.commit()
        
        logger.warning(f"Processed {len(appointments)} appointments - actual sending disabled")
        
        return {
            "total_appointments": len(appointments),
            "sms_attempted": sms_attempted,
            "email_attempted": email_attempted,
            "sms_sent": 0,  # Always 0 since disabled
            "email_sent": 0,  # Always 0 since disabled
            "warning": "AWS SES and Twilio integrations are disabled - no actual reminders sent"
        }
