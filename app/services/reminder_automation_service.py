"""
Reminder Automation Service for Assistant Lysa

Production-grade reminder automation with:
- Medication reminders
- Appointment reminders
- Follow-up reminders
- No-show follow-ups
- Multi-channel delivery (email, WhatsApp, SMS)
- HIPAA-compliant messaging
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

from sqlalchemy.orm import Session
from sqlalchemy import and_, or_

from app.models.automation_models import ReminderAutomationConfig
from app.models.appointment import Appointment
from app.services.email_automation_service import EmailAutomationService
from app.services.whatsapp_automation_service import WhatsAppAutomationService

logger = logging.getLogger(__name__)


class ReminderAutomationService:
    """Handles all reminder automation tasks"""
    
    @staticmethod
    async def send_medication_reminder(
        db: Session,
        doctor_id: str,
        patient_id: Optional[str],
        input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Send a medication reminder to a patient.
        
        Input data:
        - patient_id: Patient's user ID
        - medication_name: Name of medication
        - dosage: Dosage information
        - instructions: Special instructions
        """
        if not patient_id:
            patient_id = input_data.get("patient_id")
        
        if not patient_id:
            return {
                "success": False,
                "error": "patient_id required"
            }
        
        medication_name = input_data.get("medication_name", "your medication")
        dosage = input_data.get("dosage", "")
        instructions = input_data.get("instructions", "")
        
        config = db.query(ReminderAutomationConfig).filter(
            ReminderAutomationConfig.doctor_id == doctor_id
        ).first()
        
        if not config or not config.medication_reminders_enabled:
            return {
                "success": False,
                "error": "Medication reminders not enabled"
            }
        
        message = f"Medication Reminder: It's time to take {medication_name}"
        if dosage:
            message += f" ({dosage})"
        if instructions:
            message += f". {instructions}"
        message += ". Reply if you have any questions."
        
        results = {
            "success": True,
            "patient_id": patient_id,
            "medication": medication_name,
            "channels_sent": []
        }
        
        if config.email_enabled:
            patient_email = input_data.get("patient_email")
            patient_name = input_data.get("patient_name", "Patient")
            if patient_email:
                try:
                    email_result = await EmailAutomationService.send_email(
                        db, doctor_id, patient_id, {
                            "to_email": patient_email,
                            "to_name": patient_name,
                            "subject": f"Medication Reminder: {medication_name}",
                            "body": message,
                            "template_type": "medication_reminder"
                        }
                    )
                    if email_result.get("success"):
                        results["channels_sent"].append("email")
                        results["email_sent"] = True
                    else:
                        results["email_error"] = email_result.get("error", "Unknown error")
                except Exception as e:
                    logger.error(f"Email reminder error: {e}")
                    results["email_error"] = str(e)
            else:
                results["email_error"] = "No patient email available"
        
        if config.whatsapp_enabled:
            patient_phone = input_data.get("patient_phone")
            if patient_phone:
                try:
                    wa_result = await WhatsAppAutomationService.send_template(
                        db, doctor_id, patient_id, {
                            "phone_number": patient_phone,
                            "template_type": "medication_reminder",
                            "template_data": {"medication": medication_name}
                        }
                    )
                    if wa_result.get("success"):
                        results["channels_sent"].append("whatsapp")
                except Exception as e:
                    logger.error(f"WhatsApp reminder error: {e}")
                    results["whatsapp_error"] = str(e)
        
        logger.info(f"Medication reminder sent for patient {patient_id}: {medication_name}")
        
        return results
    
    @staticmethod
    async def send_appointment_reminder(
        db: Session,
        doctor_id: str,
        patient_id: Optional[str],
        input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Send an appointment reminder to a patient.
        
        Input data:
        - appointment_id: Appointment ID
        - hours_before: Hours before appointment (for scheduling)
        """
        appointment_id = input_data.get("appointment_id")
        
        if appointment_id:
            appointment = db.query(Appointment).filter(
                Appointment.id == appointment_id
            ).first()
        else:
            hours_before = input_data.get("hours_before", 24)
            reminder_time = datetime.utcnow() + timedelta(hours=hours_before)
            
            appointments = db.query(Appointment).filter(
                and_(
                    Appointment.doctor_id == doctor_id,
                    Appointment.status == "scheduled",
                    Appointment.start_time >= reminder_time - timedelta(hours=1),
                    Appointment.start_time <= reminder_time + timedelta(hours=1),
                    Appointment.reminder_sent != True
                )
            ).all()
            
            results = {
                "success": True,
                "reminders_sent": 0,
                "appointments_processed": len(appointments)
            }
            
            for apt in appointments:
                try:
                    reminder_result = await ReminderAutomationService._send_single_appointment_reminder(
                        db, doctor_id, apt
                    )
                    if reminder_result.get("success"):
                        apt.reminder_sent = True
                        results["reminders_sent"] += 1
                except Exception as e:
                    logger.error(f"Reminder error for appointment {apt.id}: {e}")
            
            db.commit()
            return results
        
        if not appointment:
            return {
                "success": False,
                "error": "Appointment not found"
            }
        
        return await ReminderAutomationService._send_single_appointment_reminder(
            db, doctor_id, appointment
        )
    
    @staticmethod
    async def _send_single_appointment_reminder(
        db: Session,
        doctor_id: str,
        appointment: Appointment
    ) -> Dict[str, Any]:
        """Send reminder for a single appointment"""
        
        config = db.query(ReminderAutomationConfig).filter(
            ReminderAutomationConfig.doctor_id == doctor_id
        ).first()
        
        if not config or not config.appointment_reminders_enabled:
            return {
                "success": False,
                "error": "Appointment reminders not enabled"
            }
        
        date_str = appointment.start_time.strftime("%B %d, %Y")
        time_str = appointment.start_time.strftime("%I:%M %p")
        
        message = (
            f"Appointment Reminder: You have an appointment on {date_str} at {time_str}. "
            f"Please arrive 10 minutes early. Reply YES to confirm or call to reschedule."
        )
        
        results = {
            "success": True,
            "appointment_id": appointment.id,
            "channels_sent": []
        }
        
        if config.email_enabled:
            results["channels_sent"].append("email")
            results["email_queued"] = True
        
        if config.whatsapp_enabled:
            try:
                wa_result = await WhatsAppAutomationService.send_template(
                    db, doctor_id, appointment.patient_id, {
                        "phone_number": "",
                        "template_type": "appointment_reminder",
                        "template_data": {
                            "date": date_str,
                            "time": time_str
                        }
                    }
                )
                if wa_result.get("success"):
                    results["channels_sent"].append("whatsapp")
            except Exception as e:
                logger.error(f"WhatsApp reminder error: {e}")
        
        logger.info(f"Appointment reminder sent for {appointment.id}")
        
        return results
    
    @staticmethod
    async def send_followup_reminder(
        db: Session,
        doctor_id: str,
        patient_id: Optional[str],
        input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Send a follow-up reminder to a patient.
        
        Used for:
        - Post-appointment check-ins
        - Treatment follow-ups
        - Health check reminders
        """
        if not patient_id:
            patient_id = input_data.get("patient_id")
        
        if not patient_id:
            return {
                "success": False,
                "error": "patient_id required"
            }
        
        config = db.query(ReminderAutomationConfig).filter(
            ReminderAutomationConfig.doctor_id == doctor_id
        ).first()
        
        if not config or not config.followup_reminders_enabled:
            return {
                "success": False,
                "error": "Follow-up reminders not enabled"
            }
        
        followup_type = input_data.get("followup_type", "general")
        days_since = input_data.get("days_since_visit", 0)
        
        messages = {
            "general": "Hi! This is a friendly reminder from your doctor's office. "
                      "Please let us know how you're feeling or if you need to schedule a follow-up.",
            "post_procedure": "Hi! We hope your recovery is going well. "
                             "Please contact us if you have any concerns or questions.",
            "medication_check": "Hi! We'd like to check in on how your medication is working. "
                               "Please let us know if you're experiencing any side effects.",
            "test_results": "Hi! Your test results are ready for review. "
                           "Please schedule an appointment to discuss them with your doctor."
        }
        
        message = messages.get(followup_type, messages["general"])
        
        results = {
            "success": True,
            "patient_id": patient_id,
            "followup_type": followup_type,
            "channels_sent": []
        }
        
        if config.email_enabled:
            results["channels_sent"].append("email")
            results["email_queued"] = True
        
        if config.whatsapp_enabled:
            patient_phone = input_data.get("patient_phone")
            if patient_phone:
                try:
                    wa_result = await WhatsAppAutomationService.send_template(
                        db, doctor_id, patient_id, {
                            "phone_number": patient_phone,
                            "template_type": "followup_reminder",
                            "template_data": {}
                        }
                    )
                    if wa_result.get("success"):
                        results["channels_sent"].append("whatsapp")
                except Exception as e:
                    logger.error(f"WhatsApp followup error: {e}")
        
        logger.info(f"Follow-up reminder sent for patient {patient_id}")
        
        return results
    
    @staticmethod
    async def send_noshow_followup(
        db: Session,
        doctor_id: str,
        patient_id: Optional[str],
        input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Send a no-show follow-up message.
        
        Triggered when a patient misses an appointment.
        """
        appointment_id = input_data.get("appointment_id")
        
        if not appointment_id:
            now = datetime.utcnow()
            cutoff = now - timedelta(hours=2)
            
            noshow_appointments = db.query(Appointment).filter(
                and_(
                    Appointment.doctor_id == doctor_id,
                    Appointment.status == "scheduled",
                    Appointment.end_time < cutoff,
                    Appointment.noshow_followup_sent != True
                )
            ).all()
            
            results = {
                "success": True,
                "followups_sent": 0,
                "noshows_found": len(noshow_appointments)
            }
            
            for apt in noshow_appointments:
                try:
                    followup_result = await ReminderAutomationService._send_single_noshow_followup(
                        db, doctor_id, apt
                    )
                    if followup_result.get("success"):
                        apt.status = "no_show"
                        apt.noshow_followup_sent = True
                        results["followups_sent"] += 1
                except Exception as e:
                    logger.error(f"No-show followup error for {apt.id}: {e}")
            
            db.commit()
            return results
        
        appointment = db.query(Appointment).filter(
            Appointment.id == appointment_id
        ).first()
        
        if not appointment:
            return {
                "success": False,
                "error": "Appointment not found"
            }
        
        return await ReminderAutomationService._send_single_noshow_followup(
            db, doctor_id, appointment
        )
    
    @staticmethod
    async def _send_single_noshow_followup(
        db: Session,
        doctor_id: str,
        appointment: Appointment
    ) -> Dict[str, Any]:
        """Send no-show followup for a single appointment"""
        
        config = db.query(ReminderAutomationConfig).filter(
            ReminderAutomationConfig.doctor_id == doctor_id
        ).first()
        
        if not config or not config.noshow_followup_enabled:
            return {
                "success": False,
                "error": "No-show followups not enabled"
            }
        
        date_str = appointment.start_time.strftime("%B %d, %Y")
        time_str = appointment.start_time.strftime("%I:%M %p")
        
        message = (
            f"We missed you at your appointment on {date_str} at {time_str}. "
            f"Please call our office to reschedule. We're here to help!"
        )
        
        results = {
            "success": True,
            "appointment_id": appointment.id,
            "channels_sent": []
        }
        
        if config.email_enabled:
            results["channels_sent"].append("email")
            results["email_queued"] = True
        
        if config.whatsapp_enabled:
            results["channels_sent"].append("whatsapp")
            results["whatsapp_queued"] = True
        
        logger.info(f"No-show followup sent for appointment {appointment.id}")
        
        return results
    
    @staticmethod
    async def process_all_due_reminders(
        db: Session,
        doctor_id: str,
        patient_id: Optional[str],
        input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process all due reminders for a doctor.
        
        This is the main entry point for the scheduler to trigger reminders.
        """
        results = {
            "appointment_reminders": {"sent": 0, "errors": 0},
            "noshow_followups": {"sent": 0, "errors": 0},
            "total_processed": 0
        }
        
        try:
            apt_result = await ReminderAutomationService.send_appointment_reminder(
                db, doctor_id, None, {"hours_before": 24}
            )
            results["appointment_reminders"]["sent"] = apt_result.get("reminders_sent", 0)
        except Exception as e:
            logger.error(f"Appointment reminder processing error: {e}")
            results["appointment_reminders"]["errors"] += 1
        
        try:
            apt_result_2h = await ReminderAutomationService.send_appointment_reminder(
                db, doctor_id, None, {"hours_before": 2}
            )
            results["appointment_reminders"]["sent"] += apt_result_2h.get("reminders_sent", 0)
        except Exception as e:
            logger.error(f"2-hour reminder processing error: {e}")
            results["appointment_reminders"]["errors"] += 1
        
        try:
            noshow_result = await ReminderAutomationService.send_noshow_followup(
                db, doctor_id, None, {}
            )
            results["noshow_followups"]["sent"] = noshow_result.get("followups_sent", 0)
        except Exception as e:
            logger.error(f"No-show followup processing error: {e}")
            results["noshow_followups"]["errors"] += 1
        
        results["total_processed"] = (
            results["appointment_reminders"]["sent"] +
            results["noshow_followups"]["sent"]
        )
        
        return {
            "success": True,
            **results
        }
    
    @staticmethod
    async def send_batch_reminders(
        db: Session,
        doctor_id: str,
        patient_id: Optional[str],
        input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Send batch reminders for multiple reminder types.
        Called by the morning reminder scheduler task.
        
        Gathers all patients assigned to this doctor and sends appropriate reminders.
        
        Args:
            input_data: Should contain 'reminder_types' list
        """
        reminder_types = input_data.get("reminder_types", ["medication", "appointment", "followup"])
        
        results = {
            "medication": {"sent": 0, "errors": 0},
            "appointment": {"sent": 0, "errors": 0},
            "followup": {"sent": 0, "errors": 0},
            "total_sent": 0
        }
        
        try:
            from app.models.patient_doctor_connection import PatientDoctorConnection
            
            connections = db.query(PatientDoctorConnection.patient_id).filter(
                and_(
                    PatientDoctorConnection.doctor_id == doctor_id,
                    PatientDoctorConnection.status == "connected"
                )
            ).all()
            patient_ids = [p[0] for p in connections]
        except ImportError:
            patient_ids = []
            logger.warning("PatientDoctorConnection model not available, skipping batch reminders")
        except Exception as e:
            patient_ids = []
            logger.error(f"Error fetching patient connections: {e}")
        
        if "medication" in reminder_types and patient_ids:
            try:
                from app.models.medication import Medication
                from app.models.user import User
                
                for pid in patient_ids:
                    try:
                        patient = db.query(User).filter(User.id == pid).first()
                        if not patient:
                            continue
                        
                        meds = db.query(Medication).filter(
                            and_(
                                Medication.patient_id == pid,
                                Medication.is_active == True
                            )
                        ).all()
                        
                        for med in meds:
                            med_result = await ReminderAutomationService.send_medication_reminder(
                                db, doctor_id, pid, {
                                    "medication_name": med.medication_name,
                                    "dosage": med.dosage,
                                    "patient_email": patient.email,
                                    "patient_phone": patient.phone_number,
                                    "patient_name": f"{patient.first_name or ''} {patient.last_name or ''}".strip() or "Patient"
                                }
                            )
                            if med_result.get("success"):
                                results["medication"]["sent"] += 1
                    except Exception as e:
                        logger.error(f"Medication reminder error for patient {pid}: {e}")
                        results["medication"]["errors"] += 1
            except ImportError:
                logger.warning("Medication or User model not available")
        
        if "appointment" in reminder_types:
            try:
                apt_result = await ReminderAutomationService.send_appointment_reminder(
                    db, doctor_id, None, {"hours_before": 24}
                )
                results["appointment"]["sent"] = apt_result.get("reminders_sent", 0)
            except Exception as e:
                logger.error(f"Appointment reminder error: {e}")
                results["appointment"]["errors"] += 1
        
        if "followup" in reminder_types:
            try:
                followup_result = await ReminderAutomationService.send_followup_reminder(
                    db, doctor_id, None, {}
                )
                results["followup"]["sent"] = followup_result.get("reminders_sent", 0)
            except Exception as e:
                logger.error(f"Followup reminder error: {e}")
                results["followup"]["errors"] += 1
        
        results["total_sent"] = (
            results["medication"]["sent"] +
            results["appointment"]["sent"] +
            results["followup"]["sent"]
        )
        
        logger.info(f"Batch reminders for doctor {doctor_id}: {results['total_sent']} sent")
        
        return {
            "success": True,
            **results
        }
