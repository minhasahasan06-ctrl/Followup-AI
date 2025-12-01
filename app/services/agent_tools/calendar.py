"""
Calendar Tool Microservice
Appointment scheduling and calendar management for agents
"""

import logging
import json
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import uuid

from app.services.agent_tools.base import BaseTool, ToolExecutionContext
from app.models.agent_models import ToolCallResult, ToolStatus

logger = logging.getLogger(__name__)


class CalendarTool(BaseTool):
    """
    Calendar management tool for Assistant Lysa.
    Handles appointment scheduling, rescheduling, and calendar queries.
    """
    
    def __init__(self):
        super().__init__()
        self.name = "calendar"
        self.display_name = "Calendar Management"
        self.description = """
Manage appointments and calendar for healthcare providers.
Actions: schedule_appointment, reschedule_appointment, cancel_appointment, 
get_availability, get_appointments, get_patient_appointments
"""
        self.tool_type = "calendar"
        self.requires_approval = False
        self.allowed_roles = ["doctor"]
        self.required_permissions = ["calendar:read", "calendar:write"]
        self.version = 1
    
    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "schedule_appointment",
                        "reschedule_appointment", 
                        "cancel_appointment",
                        "get_availability",
                        "get_appointments",
                        "get_patient_appointments"
                    ],
                    "description": "The calendar action to perform"
                },
                "patient_id": {
                    "type": "string",
                    "description": "Patient ID for appointment-related actions"
                },
                "appointment_id": {
                    "type": "string",
                    "description": "Existing appointment ID for reschedule/cancel"
                },
                "start_datetime": {
                    "type": "string",
                    "description": "Appointment start time in ISO format"
                },
                "end_datetime": {
                    "type": "string",
                    "description": "Appointment end time in ISO format"
                },
                "duration_minutes": {
                    "type": "integer",
                    "description": "Duration in minutes (default: 30)"
                },
                "appointment_type": {
                    "type": "string",
                    "enum": ["followup", "consultation", "checkup", "urgent", "telehealth"],
                    "description": "Type of appointment"
                },
                "notes": {
                    "type": "string",
                    "description": "Additional notes for the appointment"
                },
                "date_from": {
                    "type": "string",
                    "description": "Start date for availability/appointments query (ISO format)"
                },
                "date_to": {
                    "type": "string",
                    "description": "End date for availability/appointments query (ISO format)"
                }
            },
            "required": ["action"]
        }
    
    async def execute(
        self,
        parameters: Dict[str, Any],
        context: ToolExecutionContext
    ) -> ToolCallResult:
        """Execute calendar action"""
        action = parameters.get("action")
        
        try:
            if action == "schedule_appointment":
                return await self._schedule_appointment(parameters, context)
            elif action == "reschedule_appointment":
                return await self._reschedule_appointment(parameters, context)
            elif action == "cancel_appointment":
                return await self._cancel_appointment(parameters, context)
            elif action == "get_availability":
                return await self._get_availability(parameters, context)
            elif action == "get_appointments":
                return await self._get_appointments(parameters, context)
            elif action == "get_patient_appointments":
                return await self._get_patient_appointments(parameters, context)
            else:
                return ToolCallResult(
                    tool_call_id=context.message_id,
                    tool_name=self.name,
                    status=ToolStatus.FAILED,
                    error=f"Unknown action: {action}"
                )
        except Exception as e:
            logger.error(f"Calendar tool error: {e}")
            return ToolCallResult(
                tool_call_id=context.message_id,
                tool_name=self.name,
                status=ToolStatus.FAILED,
                error=str(e)
            )
    
    async def _schedule_appointment(
        self,
        parameters: Dict[str, Any],
        context: ToolExecutionContext
    ) -> ToolCallResult:
        """Schedule a new appointment"""
        from app.database import SessionLocal
        from sqlalchemy import text
        
        patient_id = parameters.get("patient_id") or context.patient_id
        if not patient_id:
            return ToolCallResult(
                tool_call_id=context.message_id,
                tool_name=self.name,
                status=ToolStatus.FAILED,
                error="Patient ID required for scheduling"
            )
        
        doctor_id = context.doctor_id or context.user_id
        start_datetime = parameters.get("start_datetime")
        duration_minutes = parameters.get("duration_minutes", 30)
        appointment_type = parameters.get("appointment_type", "followup")
        notes = parameters.get("notes", "")
        
        if not start_datetime:
            return ToolCallResult(
                tool_call_id=context.message_id,
                tool_name=self.name,
                status=ToolStatus.FAILED,
                error="Start datetime required for scheduling"
            )
        
        try:
            start_dt = datetime.fromisoformat(start_datetime.replace('Z', '+00:00'))
        except ValueError:
            return ToolCallResult(
                tool_call_id=context.message_id,
                tool_name=self.name,
                status=ToolStatus.FAILED,
                error="Invalid datetime format. Use ISO format (YYYY-MM-DDTHH:MM:SS)"
            )
        
        end_dt = start_dt + timedelta(minutes=duration_minutes)
        appointment_id = str(uuid.uuid4())
        
        db = SessionLocal()
        try:
            conflict_result = db.execute(
                text("""
                    SELECT id FROM appointments
                    WHERE doctor_id = :doctor_id
                    AND status = 'scheduled'
                    AND (
                        (scheduled_at <= :start_time AND scheduled_at + (duration_minutes * interval '1 minute') > :start_time)
                        OR (scheduled_at < :end_time AND scheduled_at >= :start_time)
                    )
                    LIMIT 1
                """),
                {
                    "doctor_id": doctor_id,
                    "start_time": start_dt,
                    "end_time": end_dt
                }
            )
            
            if conflict_result.fetchone():
                return ToolCallResult(
                    tool_call_id=context.message_id,
                    tool_name=self.name,
                    status=ToolStatus.FAILED,
                    error="Time slot has a conflict with existing appointment"
                )
            
            db.execute(
                text("""
                    INSERT INTO appointments (
                        id, patient_id, doctor_id, scheduled_at, duration_minutes,
                        appointment_type, status, notes, created_at, updated_at
                    ) VALUES (
                        :id, :patient_id, :doctor_id, :scheduled_at, :duration_minutes,
                        :appointment_type, 'scheduled', :notes, NOW(), NOW()
                    )
                """),
                {
                    "id": appointment_id,
                    "patient_id": patient_id,
                    "doctor_id": doctor_id,
                    "scheduled_at": start_dt,
                    "duration_minutes": duration_minutes,
                    "appointment_type": appointment_type,
                    "notes": notes
                }
            )
            db.commit()
            
            return ToolCallResult(
                tool_call_id=context.message_id,
                tool_name=self.name,
                status=ToolStatus.COMPLETED,
                result={
                    "appointment_id": appointment_id,
                    "patient_id": patient_id,
                    "doctor_id": doctor_id,
                    "scheduled_at": start_dt.isoformat(),
                    "end_at": end_dt.isoformat(),
                    "duration_minutes": duration_minutes,
                    "appointment_type": appointment_type,
                    "status": "scheduled",
                    "message": f"Appointment scheduled for {start_dt.strftime('%B %d, %Y at %I:%M %p')}"
                }
            )
        finally:
            db.close()
    
    async def _reschedule_appointment(
        self,
        parameters: Dict[str, Any],
        context: ToolExecutionContext
    ) -> ToolCallResult:
        """Reschedule an existing appointment"""
        from app.database import SessionLocal
        from sqlalchemy import text
        
        appointment_id = parameters.get("appointment_id")
        new_start_datetime = parameters.get("start_datetime")
        
        if not appointment_id:
            return ToolCallResult(
                tool_call_id=context.message_id,
                tool_name=self.name,
                status=ToolStatus.FAILED,
                error="Appointment ID required for rescheduling"
            )
        
        if not new_start_datetime:
            return ToolCallResult(
                tool_call_id=context.message_id,
                tool_name=self.name,
                status=ToolStatus.FAILED,
                error="New start datetime required for rescheduling"
            )
        
        try:
            new_start_dt = datetime.fromisoformat(new_start_datetime.replace('Z', '+00:00'))
        except ValueError:
            return ToolCallResult(
                tool_call_id=context.message_id,
                tool_name=self.name,
                status=ToolStatus.FAILED,
                error="Invalid datetime format"
            )
        
        db = SessionLocal()
        try:
            result = db.execute(
                text("""
                    UPDATE appointments
                    SET scheduled_at = :new_time, updated_at = NOW()
                    WHERE id = :appointment_id
                    AND (doctor_id = :doctor_id OR :is_admin)
                    AND status = 'scheduled'
                    RETURNING id, patient_id, doctor_id, scheduled_at, duration_minutes
                """),
                {
                    "new_time": new_start_dt,
                    "appointment_id": appointment_id,
                    "doctor_id": context.user_id,
                    "is_admin": context.user_role == "admin"
                }
            )
            
            row = result.fetchone()
            if not row:
                return ToolCallResult(
                    tool_call_id=context.message_id,
                    tool_name=self.name,
                    status=ToolStatus.FAILED,
                    error="Appointment not found or not authorized to modify"
                )
            
            db.commit()
            
            return ToolCallResult(
                tool_call_id=context.message_id,
                tool_name=self.name,
                status=ToolStatus.COMPLETED,
                result={
                    "appointment_id": row[0],
                    "patient_id": row[1],
                    "doctor_id": row[2],
                    "new_scheduled_at": new_start_dt.isoformat(),
                    "message": f"Appointment rescheduled to {new_start_dt.strftime('%B %d, %Y at %I:%M %p')}"
                }
            )
        finally:
            db.close()
    
    async def _cancel_appointment(
        self,
        parameters: Dict[str, Any],
        context: ToolExecutionContext
    ) -> ToolCallResult:
        """Cancel an existing appointment"""
        from app.database import SessionLocal
        from sqlalchemy import text
        
        appointment_id = parameters.get("appointment_id")
        
        if not appointment_id:
            return ToolCallResult(
                tool_call_id=context.message_id,
                tool_name=self.name,
                status=ToolStatus.FAILED,
                error="Appointment ID required for cancellation"
            )
        
        db = SessionLocal()
        try:
            result = db.execute(
                text("""
                    UPDATE appointments
                    SET status = 'cancelled', updated_at = NOW()
                    WHERE id = :appointment_id
                    AND (doctor_id = :doctor_id OR :is_admin)
                    AND status = 'scheduled'
                    RETURNING id, patient_id, scheduled_at
                """),
                {
                    "appointment_id": appointment_id,
                    "doctor_id": context.user_id,
                    "is_admin": context.user_role == "admin"
                }
            )
            
            row = result.fetchone()
            if not row:
                return ToolCallResult(
                    tool_call_id=context.message_id,
                    tool_name=self.name,
                    status=ToolStatus.FAILED,
                    error="Appointment not found or not authorized to cancel"
                )
            
            db.commit()
            
            return ToolCallResult(
                tool_call_id=context.message_id,
                tool_name=self.name,
                status=ToolStatus.COMPLETED,
                result={
                    "appointment_id": row[0],
                    "patient_id": row[1],
                    "original_scheduled_at": row[2].isoformat() if row[2] else None,
                    "status": "cancelled",
                    "message": "Appointment has been cancelled"
                }
            )
        finally:
            db.close()
    
    async def _get_availability(
        self,
        parameters: Dict[str, Any],
        context: ToolExecutionContext
    ) -> ToolCallResult:
        """Get doctor's available time slots"""
        from app.database import SessionLocal
        from sqlalchemy import text
        
        doctor_id = context.doctor_id or context.user_id
        date_from = parameters.get("date_from")
        date_to = parameters.get("date_to")
        
        if not date_from:
            date_from = datetime.utcnow().date().isoformat()
        if not date_to:
            date_to_dt = datetime.fromisoformat(date_from) + timedelta(days=7)
            date_to = date_to_dt.date().isoformat()
        
        db = SessionLocal()
        try:
            result = db.execute(
                text("""
                    SELECT scheduled_at, duration_minutes
                    FROM appointments
                    WHERE doctor_id = :doctor_id
                    AND status = 'scheduled'
                    AND scheduled_at >= :date_from
                    AND scheduled_at < :date_to
                    ORDER BY scheduled_at
                """),
                {
                    "doctor_id": doctor_id,
                    "date_from": date_from,
                    "date_to": date_to
                }
            )
            
            busy_slots = []
            for row in result.fetchall():
                start = row[0]
                end = start + timedelta(minutes=row[1])
                busy_slots.append({
                    "start": start.isoformat(),
                    "end": end.isoformat()
                })
            
            return ToolCallResult(
                tool_call_id=context.message_id,
                tool_name=self.name,
                status=ToolStatus.COMPLETED,
                result={
                    "doctor_id": doctor_id,
                    "date_range": {"from": date_from, "to": date_to},
                    "busy_slots": busy_slots,
                    "message": f"Found {len(busy_slots)} scheduled appointments in the given range"
                }
            )
        finally:
            db.close()
    
    async def _get_appointments(
        self,
        parameters: Dict[str, Any],
        context: ToolExecutionContext
    ) -> ToolCallResult:
        """Get doctor's upcoming appointments"""
        from app.database import SessionLocal
        from sqlalchemy import text
        
        doctor_id = context.doctor_id or context.user_id
        date_from = parameters.get("date_from", datetime.utcnow().isoformat())
        
        db = SessionLocal()
        try:
            result = db.execute(
                text("""
                    SELECT a.id, a.patient_id, a.scheduled_at, a.duration_minutes,
                           a.appointment_type, a.status, a.notes,
                           u.first_name, u.last_name
                    FROM appointments a
                    LEFT JOIN users u ON a.patient_id = u.id
                    WHERE a.doctor_id = :doctor_id
                    AND a.scheduled_at >= :date_from
                    AND a.status = 'scheduled'
                    ORDER BY a.scheduled_at
                    LIMIT 20
                """),
                {
                    "doctor_id": doctor_id,
                    "date_from": date_from
                }
            )
            
            appointments = []
            for row in result.fetchall():
                patient_name = f"{row[7] or ''} {row[8] or ''}".strip()
                appointments.append({
                    "appointment_id": row[0],
                    "patient_id": row[1],
                    "patient_name": patient_name or "Unknown Patient",
                    "scheduled_at": row[2].isoformat() if row[2] else None,
                    "duration_minutes": row[3],
                    "appointment_type": row[4],
                    "status": row[5],
                    "notes": row[6]
                })
            
            return ToolCallResult(
                tool_call_id=context.message_id,
                tool_name=self.name,
                status=ToolStatus.COMPLETED,
                result={
                    "doctor_id": doctor_id,
                    "appointments": appointments,
                    "count": len(appointments),
                    "message": f"Found {len(appointments)} upcoming appointments"
                }
            )
        finally:
            db.close()
    
    async def _get_patient_appointments(
        self,
        parameters: Dict[str, Any],
        context: ToolExecutionContext
    ) -> ToolCallResult:
        """Get appointments for a specific patient"""
        from app.database import SessionLocal
        from sqlalchemy import text
        
        patient_id = parameters.get("patient_id") or context.patient_id
        if not patient_id:
            return ToolCallResult(
                tool_call_id=context.message_id,
                tool_name=self.name,
                status=ToolStatus.FAILED,
                error="Patient ID required"
            )
        
        db = SessionLocal()
        try:
            result = db.execute(
                text("""
                    SELECT a.id, a.doctor_id, a.scheduled_at, a.duration_minutes,
                           a.appointment_type, a.status, a.notes,
                           u.first_name, u.last_name
                    FROM appointments a
                    LEFT JOIN users u ON a.doctor_id = u.id
                    WHERE a.patient_id = :patient_id
                    AND a.scheduled_at >= :now
                    ORDER BY a.scheduled_at
                    LIMIT 20
                """),
                {
                    "patient_id": patient_id,
                    "now": datetime.utcnow()
                }
            )
            
            appointments = []
            for row in result.fetchall():
                doctor_name = f"Dr. {row[8] or row[7] or 'Unknown'}"
                appointments.append({
                    "appointment_id": row[0],
                    "doctor_id": row[1],
                    "doctor_name": doctor_name,
                    "scheduled_at": row[2].isoformat() if row[2] else None,
                    "duration_minutes": row[3],
                    "appointment_type": row[4],
                    "status": row[5],
                    "notes": row[6]
                })
            
            return ToolCallResult(
                tool_call_id=context.message_id,
                tool_name=self.name,
                status=ToolStatus.COMPLETED,
                result={
                    "patient_id": patient_id,
                    "appointments": appointments,
                    "count": len(appointments),
                    "message": f"Found {len(appointments)} upcoming appointments for patient"
                }
            )
        finally:
            db.close()
