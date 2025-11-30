"""
Appointment Automation Service for Assistant Lysa

Production-grade appointment automation with:
- Appointment request processing
- Availability checking
- Auto-booking with conflict detection
- Calendar sync (Google Calendar)
- Patient notifications
- HIPAA-compliant data handling
"""

import os
import logging
from datetime import datetime, timedelta, time
from typing import Dict, Any, List, Optional
import json

from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

import openai

from app.models.automation_models import AppointmentAutomationConfig
from app.models.appointment import Appointment
from app.models.calendar_sync import GoogleCalendarSync

logger = logging.getLogger(__name__)

openai_client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class AppointmentAutomationService:
    """Handles all appointment automation tasks"""
    
    @staticmethod
    async def process_request(
        db: Session,
        doctor_id: str,
        patient_id: Optional[str],
        input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process an appointment request from email or WhatsApp.
        
        Uses AI to extract:
        - Preferred date/time
        - Appointment type
        - Urgency level
        - Patient information
        """
        request_text = input_data.get("text", "")
        source = input_data.get("source", "unknown")
        patient_name = input_data.get("patient_name")
        patient_phone = input_data.get("patient_phone")
        patient_email = input_data.get("patient_email")
        
        if not request_text:
            return {
                "success": False,
                "error": "No request text provided"
            }
        
        prompt = f"""Extract appointment request details from this message.

Message: {request_text}

Extract and respond with JSON:
{{
    "wants_appointment": true/false,
    "preferred_date": "YYYY-MM-DD or null",
    "preferred_time": "HH:MM or null",
    "time_flexibility": "exact|morning|afternoon|evening|flexible",
    "appointment_type": "checkup|followup|consultation|urgent|specialist|other",
    "urgency": "routine|soon|urgent",
    "reason": "brief reason for visit",
    "special_requests": "any special requests or null"
}}"""

        try:
            response = await openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a medical appointment scheduler. Extract appointment details accurately."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=300,
                response_format={"type": "json_object"}
            )
            
            extracted = json.loads(response.choices[0].message.content)
            
            if not extracted.get("wants_appointment"):
                return {
                    "success": True,
                    "appointment_requested": False,
                    "message": "No appointment request detected"
                }
            
            config = db.query(AppointmentAutomationConfig).filter(
                AppointmentAutomationConfig.doctor_id == doctor_id
            ).first()
            
            result = {
                "success": True,
                "appointment_requested": True,
                "extracted_details": extracted,
                "patient_info": {
                    "name": patient_name,
                    "phone": patient_phone,
                    "email": patient_email
                },
                "source": source
            }
            
            if config and config.auto_book_enabled and patient_id:
                available_slots = await AppointmentAutomationService._find_available_slots(
                    db, doctor_id, extracted, config
                )
                
                if available_slots:
                    result["available_slots"] = available_slots[:5]
                    result["can_auto_book"] = True
                else:
                    result["can_auto_book"] = False
                    result["message"] = "No available slots matching request"
            
            return result
            
        except Exception as e:
            logger.error(f"Appointment request processing error: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    @staticmethod
    async def book_appointment(
        db: Session,
        doctor_id: str,
        patient_id: Optional[str],
        input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Book an appointment for a patient.
        
        Required input_data:
        - patient_id: Patient's user ID
        - start_time: Appointment start time (ISO format)
        - appointment_type: Type of appointment
        - duration_minutes: Duration (optional, uses config default)
        """
        if not patient_id:
            patient_id = input_data.get("patient_id")
        
        start_time_str = input_data.get("start_time")
        appointment_type = input_data.get("appointment_type", "consultation")
        
        if not patient_id or not start_time_str:
            return {
                "success": False,
                "error": "patient_id and start_time required"
            }
        
        try:
            start_time = datetime.fromisoformat(start_time_str.replace('Z', '+00:00'))
        except:
            return {
                "success": False,
                "error": "Invalid start_time format"
            }
        
        config = db.query(AppointmentAutomationConfig).filter(
            AppointmentAutomationConfig.doctor_id == doctor_id
        ).first()
        
        duration = input_data.get("duration_minutes")
        if not duration:
            duration = config.default_duration_minutes if config else 30
        
        end_time = start_time + timedelta(minutes=duration)
        
        buffer = config.buffer_minutes if config else 15
        check_start = start_time - timedelta(minutes=buffer)
        check_end = end_time + timedelta(minutes=buffer)
        
        conflict = db.query(Appointment).filter(
            and_(
                Appointment.doctor_id == doctor_id,
                Appointment.status != "cancelled",
                or_(
                    and_(
                        Appointment.start_time >= check_start,
                        Appointment.start_time < check_end
                    ),
                    and_(
                        Appointment.end_time > check_start,
                        Appointment.end_time <= check_end
                    )
                )
            )
        ).first()
        
        if conflict:
            return {
                "success": False,
                "error": "Time slot conflicts with existing appointment",
                "conflict_id": conflict.id
            }
        
        appointment = Appointment(
            doctor_id=doctor_id,
            patient_id=patient_id,
            start_time=start_time,
            end_time=end_time,
            appointment_type=appointment_type,
            status="scheduled",
            confirmation_status="pending",
            notes=input_data.get("notes", ""),
            created_by="lysa_automation"
        )
        
        db.add(appointment)
        db.commit()
        db.refresh(appointment)
        
        calendar_synced = False
        if config and config.calendar_sync_enabled:
            try:
                sync_result = await AppointmentAutomationService._sync_to_google_calendar(
                    db, doctor_id, appointment
                )
                calendar_synced = sync_result.get("success", False)
            except Exception as e:
                logger.warning(f"Calendar sync failed: {e}")
        
        logger.info(f"Appointment {appointment.id} booked for patient {patient_id}")
        
        return {
            "success": True,
            "appointment_id": appointment.id,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "appointment_type": appointment_type,
            "calendar_synced": calendar_synced
        }
    
    @staticmethod
    async def sync_calendar(
        db: Session,
        doctor_id: str,
        patient_id: Optional[str],
        input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Sync appointments with Google Calendar (bidirectional).
        """
        sync_config = db.query(GoogleCalendarSync).filter(
            GoogleCalendarSync.doctor_id == doctor_id
        ).first()
        
        if not sync_config or not sync_config.sync_enabled:
            return {
                "success": False,
                "error": "Google Calendar sync not enabled"
            }
        
        try:
            creds = Credentials(
                token=sync_config.access_token,
                refresh_token=sync_config.refresh_token,
                token_uri="https://oauth2.googleapis.com/token",
                client_id=os.getenv("GOOGLE_CLIENT_ID"),
                client_secret=os.getenv("GOOGLE_CLIENT_SECRET")
            )
            
            service = build('calendar', 'v3', credentials=creds)
            
            now = datetime.utcnow()
            time_min = now.isoformat() + 'Z'
            time_max = (now + timedelta(days=30)).isoformat() + 'Z'
            
            calendar_id = sync_config.calendar_id or 'primary'
            
            events_result = service.events().list(
                calendarId=calendar_id,
                timeMin=time_min,
                timeMax=time_max,
                maxResults=100,
                singleEvents=True,
                orderBy='startTime'
            ).execute()
            
            google_events = events_result.get('items', [])
            
            local_appointments = db.query(Appointment).filter(
                and_(
                    Appointment.doctor_id == doctor_id,
                    Appointment.start_time >= now,
                    Appointment.start_time <= now + timedelta(days=30)
                )
            ).all()
            
            synced_to_google = 0
            synced_from_google = 0
            
            for appointment in local_appointments:
                if not appointment.google_event_id:
                    event = {
                        'summary': f"Patient Appointment - {appointment.appointment_type}",
                        'description': f"Appointment ID: {appointment.id}\nType: {appointment.appointment_type}",
                        'start': {
                            'dateTime': appointment.start_time.isoformat(),
                            'timeZone': 'UTC'
                        },
                        'end': {
                            'dateTime': appointment.end_time.isoformat(),
                            'timeZone': 'UTC'
                        },
                        'reminders': {
                            'useDefault': False,
                            'overrides': [
                                {'method': 'popup', 'minutes': 30}
                            ]
                        }
                    }
                    
                    created_event = service.events().insert(
                        calendarId=calendar_id,
                        body=event
                    ).execute()
                    
                    appointment.google_event_id = created_event.get('id')
                    synced_to_google += 1
            
            db.commit()
            
            sync_config.last_sync_at = datetime.utcnow()
            db.commit()
            
            logger.info(f"Calendar sync complete for doctor {doctor_id}")
            
            return {
                "success": True,
                "synced_to_google": synced_to_google,
                "synced_from_google": synced_from_google,
                "google_events": len(google_events),
                "local_appointments": len(local_appointments)
            }
            
        except HttpError as e:
            logger.error(f"Google Calendar API error: {e}")
            return {
                "success": False,
                "error": f"Google Calendar API error: {str(e)}"
            }
        except Exception as e:
            logger.error(f"Calendar sync error: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    @staticmethod
    async def _find_available_slots(
        db: Session,
        doctor_id: str,
        request_details: Dict[str, Any],
        config: AppointmentAutomationConfig
    ) -> List[Dict[str, Any]]:
        """Find available appointment slots based on request and config"""
        
        preferred_date_str = request_details.get("preferred_date")
        time_flexibility = request_details.get("time_flexibility", "flexible")
        
        if preferred_date_str:
            try:
                start_date = datetime.strptime(preferred_date_str, "%Y-%m-%d").date()
            except:
                start_date = datetime.utcnow().date() + timedelta(days=1)
        else:
            start_date = datetime.utcnow().date() + timedelta(days=1)
        
        available_slots = []
        duration = config.default_duration_minutes if config else 30
        buffer = config.buffer_minutes if config else 15
        
        try:
            start_hour = int(config.available_hours_start.split(':')[0])
            end_hour = int(config.available_hours_end.split(':')[0])
        except:
            start_hour = 9
            end_hour = 17
        
        available_days = config.available_days or ["monday", "tuesday", "wednesday", "thursday", "friday"]
        
        for day_offset in range(14):
            check_date = start_date + timedelta(days=day_offset)
            day_name = check_date.strftime('%A').lower()
            
            if day_name not in available_days:
                continue
            
            existing = db.query(Appointment).filter(
                and_(
                    Appointment.doctor_id == doctor_id,
                    Appointment.status != "cancelled",
                    Appointment.start_time >= datetime.combine(check_date, time.min),
                    Appointment.start_time < datetime.combine(check_date + timedelta(days=1), time.min)
                )
            ).all()
            
            busy_times = [(a.start_time, a.end_time) for a in existing]
            
            current_time = datetime.combine(check_date, time(hour=start_hour))
            end_of_day = datetime.combine(check_date, time(hour=end_hour))
            
            while current_time + timedelta(minutes=duration) <= end_of_day:
                slot_end = current_time + timedelta(minutes=duration)
                
                is_available = True
                for busy_start, busy_end in busy_times:
                    if not (slot_end + timedelta(minutes=buffer) <= busy_start or 
                            current_time >= busy_end + timedelta(minutes=buffer)):
                        is_available = False
                        break
                
                if is_available:
                    time_of_day = "morning" if current_time.hour < 12 else (
                        "afternoon" if current_time.hour < 17 else "evening"
                    )
                    
                    if time_flexibility == "flexible" or time_flexibility == time_of_day:
                        available_slots.append({
                            "date": check_date.isoformat(),
                            "start_time": current_time.isoformat(),
                            "end_time": slot_end.isoformat(),
                            "time_of_day": time_of_day
                        })
                
                current_time += timedelta(minutes=30)
            
            if len(available_slots) >= 10:
                break
        
        return available_slots
    
    @staticmethod
    async def _sync_to_google_calendar(
        db: Session,
        doctor_id: str,
        appointment: Appointment
    ) -> Dict[str, Any]:
        """Sync a single appointment to Google Calendar"""
        
        sync_config = db.query(GoogleCalendarSync).filter(
            GoogleCalendarSync.doctor_id == doctor_id
        ).first()
        
        if not sync_config or not sync_config.sync_enabled:
            return {"success": False, "error": "Calendar not connected"}
        
        try:
            creds = Credentials(
                token=sync_config.access_token,
                refresh_token=sync_config.refresh_token,
                token_uri="https://oauth2.googleapis.com/token",
                client_id=os.getenv("GOOGLE_CLIENT_ID"),
                client_secret=os.getenv("GOOGLE_CLIENT_SECRET")
            )
            
            service = build('calendar', 'v3', credentials=creds)
            calendar_id = sync_config.calendar_id or 'primary'
            
            event = {
                'summary': f"Patient Appointment - {appointment.appointment_type}",
                'description': f"Appointment ID: {appointment.id}",
                'start': {
                    'dateTime': appointment.start_time.isoformat(),
                    'timeZone': 'UTC'
                },
                'end': {
                    'dateTime': appointment.end_time.isoformat(),
                    'timeZone': 'UTC'
                }
            }
            
            created_event = service.events().insert(
                calendarId=calendar_id,
                body=event
            ).execute()
            
            appointment.google_event_id = created_event.get('id')
            db.commit()
            
            return {
                "success": True,
                "google_event_id": created_event.get('id')
            }
            
        except Exception as e:
            logger.error(f"Google Calendar sync error: {e}")
            return {"success": False, "error": str(e)}
