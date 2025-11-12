from datetime import datetime, timedelta
from typing import List, Dict
from sqlalchemy.orm import Session
from app.models.appointment import Appointment
from app.models.user import User
from app.config import settings
import boto3
from twilio.rest import Client as TwilioClient


class AppointmentReminderService:
    def __init__(self, db: Session):
        self.db = db
        
        if settings.TWILIO_ACCOUNT_SID and settings.TWILIO_AUTH_TOKEN:
            self.twilio_client = TwilioClient(
                settings.TWILIO_ACCOUNT_SID,
                settings.TWILIO_AUTH_TOKEN
            )
        else:
            self.twilio_client = None
        
        if settings.AWS_ACCESS_KEY_ID and settings.AWS_SECRET_ACCESS_KEY:
            self.ses_client = boto3.client(
                'ses',
                region_name=settings.AWS_REGION,
                aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY
            )
        else:
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
        if not self.twilio_client or not settings.TWILIO_PHONE_NUMBER:
            print("Twilio not configured")
            return False
        
        try:
            message_body = f"Reminder: You have an appointment tomorrow at {appointment.appointment_date.strftime('%I:%M %p')}. Please arrive 10 minutes early."
            
            self.twilio_client.messages.create(
                body=message_body,
                from_=settings.TWILIO_PHONE_NUMBER,
                to=phone_number
            )
            return True
        except Exception as e:
            print(f"Error sending SMS: {e}")
            return False
    
    def send_email_reminder(self, email: str, appointment: Appointment) -> bool:
        if not self.ses_client:
            print("AWS SES not configured")
            return False
        
        try:
            subject = "Appointment Reminder - Tomorrow"
            body = f"""
            <html>
            <body>
                <h2>Appointment Reminder</h2>
                <p>This is a friendly reminder about your upcoming appointment:</p>
                <ul>
                    <li><strong>Date:</strong> {appointment.appointment_date.strftime('%B %d, %Y')}</li>
                    <li><strong>Time:</strong> {appointment.appointment_date.strftime('%I:%M %p')}</li>
                    <li><strong>Duration:</strong> {appointment.duration_minutes} minutes</li>
                </ul>
                <p>Please arrive 10 minutes early.</p>
            </body>
            </html>
            """
            
            self.ses_client.send_email(
                Source="noreply@followupai.com",
                Destination={'ToAddresses': [email]},
                Message={
                    'Subject': {'Data': subject},
                    'Body': {'Html': {'Data': body}}
                }
            )
            return True
        except Exception as e:
            print(f"Error sending email: {e}")
            return False
    
    def process_reminders(self) -> Dict:
        appointments = self.get_upcoming_appointments()
        
        sms_sent = 0
        email_sent = 0
        
        for appointment in appointments:
            patient = self.db.query(User).filter(User.id == appointment.patient_id).first()
            
            if not patient:
                continue
            
            if patient.phone_number:
                if self.send_sms_reminder(patient.phone_number, appointment):
                    sms_sent += 1
            
            if patient.email:
                if self.send_email_reminder(patient.email, appointment):
                    email_sent += 1
            
            appointment.reminder_sent = True
        
        if appointments:
            self.db.commit()
        
        return {
            "total_appointments": len(appointments),
            "sms_sent": sms_sent,
            "email_sent": email_sent
        }
