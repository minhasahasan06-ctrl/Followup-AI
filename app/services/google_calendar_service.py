from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from datetime import datetime, timedelta
from typing import Optional, List, Dict
from sqlalchemy.orm import Session
from app.models.calendar_sync import GoogleCalendarSync
from app.models.appointment import Appointment


class GoogleCalendarService:
    def __init__(self, db: Session):
        self.db = db
    
    def get_sync_config(self, doctor_id: str) -> Optional[GoogleCalendarSync]:
        return self.db.query(GoogleCalendarSync).filter(
            GoogleCalendarSync.doctor_id == doctor_id
        ).first()
    
    def create_credentials(self, sync_config: GoogleCalendarSync) -> Credentials:
        return Credentials(
            token=sync_config.access_token,
            refresh_token=sync_config.refresh_token,
            token_uri="https://oauth2.googleapis.com/token",
            client_id=None,
            client_secret=None
        )
    
    def sync_appointments_to_google(self, doctor_id: str) -> Dict:
        sync_config = self.get_sync_config(doctor_id)
        if not sync_config or not sync_config.sync_enabled:
            return {"error": "Calendar sync not enabled"}
        
        creds = self.create_credentials(sync_config)
        service = build('calendar', 'v3', credentials=creds)
        
        appointments = self.db.query(Appointment).filter(
            Appointment.doctor_id == doctor_id,
            Appointment.appointment_date >= datetime.utcnow()
        ).all()
        
        synced_count = 0
        for appointment in appointments:
            if not appointment.google_calendar_event_id:
                event = {
                    'summary': f'Appointment - Patient {appointment.patient_id}',
                    'description': appointment.notes or '',
                    'start': {
                        'dateTime': appointment.appointment_date.isoformat(),
                        'timeZone': 'UTC',
                    },
                    'end': {
                        'dateTime': (appointment.appointment_date + timedelta(minutes=appointment.duration_minutes)).isoformat(),
                        'timeZone': 'UTC',
                    },
                }
                
                created_event = service.events().insert(
                    calendarId=sync_config.calendar_id,
                    body=event
                ).execute()
                
                appointment.google_calendar_event_id = created_event['id']
                synced_count += 1
        
        if synced_count > 0:
            self.db.commit()
            sync_config.last_sync_at = datetime.utcnow()
            self.db.commit()
        
        return {
            "success": True,
            "synced_count": synced_count,
            "total_appointments": len(appointments)
        }
    
    def sync_from_google_calendar(self, doctor_id: str) -> Dict:
        sync_config = self.get_sync_config(doctor_id)
        if not sync_config or not sync_config.sync_enabled:
            return {"error": "Calendar sync not enabled"}
        
        creds = self.create_credentials(sync_config)
        service = build('calendar', 'v3', credentials=creds)
        
        now = datetime.utcnow()
        time_min = now.isoformat() + 'Z'
        
        events_result = service.events().list(
            calendarId=sync_config.calendar_id,
            timeMin=time_min,
            maxResults=100,
            singleEvents=True,
            orderBy='startTime'
        ).execute()
        
        events = events_result.get('items', [])
        
        return {
            "success": True,
            "events_count": len(events),
            "message": "Calendar sync from Google completed"
        }
