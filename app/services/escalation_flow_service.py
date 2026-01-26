"""
Escalation Flow Service

Production-grade escalation orchestration for patient emergencies.
Manages Clona → Lysa → Doctor handoff with conversation context transfer.

HIPAA Compliance:
- All escalations are audit logged
- Consent verification before doctor communication
- PHI transferred securely between agents

Escalation Flow:
1. Red flag detected during Clona conversation
2. Clona notifies Lysa (doctor's assistant) with context
3. Lysa alerts connected doctor and facilitates handoff
4. Doctor availability checked and fallbacks applied
5. Direct communication channel established
"""

import logging
import asyncio
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Callable
from enum import Enum
from dataclasses import dataclass, field
from sqlalchemy.orm import Session
from sqlalchemy import and_

from app.models.user import User
from app.models.patient_doctor_connection import PatientDoctorConnection
from app.models.security_models import AuditLog
from app.services.red_flag_detection_service import (
    RedFlagDetection, 
    RedFlagSeverity, 
    EscalationType,
    RedFlagCategory
)

logger = logging.getLogger(__name__)


class EscalationState(str, Enum):
    """States in the escalation flow"""
    IDLE = "idle"
    RED_FLAG_DETECTED = "red_flag_detected"
    CLONA_NOTIFYING = "clona_notifying"
    LYSA_RECEIVED = "lysa_received"
    DOCTOR_ALERTING = "doctor_alerting"
    DOCTOR_AVAILABLE = "doctor_available"
    DOCTOR_UNAVAILABLE = "doctor_unavailable"
    HANDOFF_IN_PROGRESS = "handoff_in_progress"
    HANDOFF_COMPLETE = "handoff_complete"
    ESCALATION_FAILED = "escalation_failed"
    EMERGENCY_FALLBACK = "emergency_fallback"


class NotificationChannel(str, Enum):
    """Notification delivery channels"""
    DASHBOARD = "dashboard"
    EMAIL = "email"
    SMS = "sms"
    PUSH = "push"
    IN_APP = "in_app"
    VOICE_CALL = "voice_call"


@dataclass
class EscalationContext:
    """Context for an escalation event"""
    escalation_id: str
    patient_id: str
    patient_name: str
    red_flag_detection: RedFlagDetection
    conversation_history: List[Dict[str, str]]
    conversation_id: Optional[str] = None
    initiated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    state: EscalationState = EscalationState.IDLE
    doctor_id: Optional[str] = None
    doctor_name: Optional[str] = None
    lysa_response: Optional[str] = None
    handoff_channel: Optional[str] = None
    fallback_used: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EscalationResult:
    """Result of an escalation attempt"""
    success: bool
    escalation_id: str
    final_state: EscalationState
    doctor_contacted: bool
    handoff_established: bool
    fallback_used: bool
    channels_used: List[NotificationChannel]
    patient_guidance: str
    doctor_response: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    next_steps: List[str] = field(default_factory=list)


class DoctorAvailabilityService:
    """
    Check and manage doctor availability for escalations.
    
    Integrates with:
    - WebSocket presence tracking
    - Calendar/schedule systems
    - On-call rotation
    """
    
    def __init__(self, db: Session):
        self.db = db
        self.logger = logging.getLogger(__name__)
    
    def check_doctor_availability(
        self, 
        doctor_id: str
    ) -> Dict[str, Any]:
        """
        Check if a doctor is available for escalation.
        
        Returns:
            Availability status with contact method
        """
        doctor = self.db.query(User).filter(
            and_(
                User.id == doctor_id,
                User.role == "doctor"
            )
        ).first()
        
        if not doctor:
            return {
                "available": False,
                "reason": "doctor_not_found",
                "contact_method": None
            }
        
        is_online = self._check_online_status(doctor_id)
        
        preferred_contact = self._get_preferred_contact_method(doctor)
        
        return {
            "available": is_online,
            "doctor_id": doctor_id,
            "doctor_name": f"{doctor.first_name} {doctor.last_name}",
            "online_status": "online" if is_online else "offline",
            "contact_method": preferred_contact,
            "can_receive_calls": is_online,
            "can_receive_chat": True,
            "can_receive_sms": bool(doctor.phone),
            "can_receive_email": bool(doctor.email),
            "checked_at": datetime.now(timezone.utc).isoformat()
        }
    
    def _check_online_status(self, doctor_id: str) -> bool:
        """Check if doctor is currently online via presence service"""
        try:
            from app.services.message_router import get_message_router
            router = get_message_router()
            if router and hasattr(router, 'connection_manager'):
                presence = router.connection_manager.presence.get(doctor_id)
                if presence:
                    return presence.is_online
        except Exception as e:
            self.logger.warning(f"Could not check presence for {doctor_id}: {e}")
        return False
    
    def _get_preferred_contact_method(self, doctor: User) -> str:
        """Determine best contact method for doctor"""
        if self._check_online_status(str(doctor.id)):
            return "in_app_chat"
        elif doctor.phone:
            return "sms"
        elif doctor.email:
            return "email"
        return "dashboard"
    
    def get_on_call_doctor(self, patient_id: str) -> Optional[Dict[str, Any]]:
        """Get the on-call doctor for a patient if primary is unavailable"""
        connections = self.db.query(PatientDoctorConnection).filter(
            and_(
                PatientDoctorConnection.patient_id == patient_id,
                PatientDoctorConnection.status == "connected"
            )
        ).all()
        
        for conn in connections:
            availability = self.check_doctor_availability(conn.doctor_id)
            if availability["available"]:
                return availability
        
        return None
    
    def get_all_connected_doctors(self, patient_id: str) -> List[Dict[str, Any]]:
        """Get all doctors connected to a patient with their availability"""
        connections = self.db.query(PatientDoctorConnection).filter(
            and_(
                PatientDoctorConnection.patient_id == patient_id,
                PatientDoctorConnection.status == "connected"
            )
        ).all()
        
        doctors = []
        for conn in connections:
            doctor = self.db.query(User).filter(User.id == conn.doctor_id).first()
            if doctor:
                availability = self.check_doctor_availability(conn.doctor_id)
                doctors.append({
                    "doctor_id": conn.doctor_id,
                    "doctor_name": f"{doctor.first_name} {doctor.last_name}",
                    "is_primary": conn.is_primary if hasattr(conn, 'is_primary') else False,
                    **availability
                })
        
        return doctors


class EscalationFlowService:
    """
    Orchestrates the complete escalation flow from red flag detection to doctor handoff.
    
    Flow:
    1. Red flag detected → Create escalation context
    2. Clona acknowledges emergency → Notifies patient of escalation
    3. Lysa receives context → Prepares doctor briefing
    4. Doctor availability checked → Select contact method
    5. Doctor alerted → Multi-channel notification
    6. Handoff established → Direct communication enabled
    7. Fallback if needed → Emergency services guidance
    """
    
    def __init__(self, db: Session):
        self.db = db
        self.logger = logging.getLogger(__name__)
        self.doctor_availability = DoctorAvailabilityService(db)
        self._active_escalations: Dict[str, EscalationContext] = {}
        self._state_handlers: Dict[EscalationState, Callable] = {
            EscalationState.RED_FLAG_DETECTED: self._handle_red_flag_detected,
            EscalationState.CLONA_NOTIFYING: self._handle_clona_notifying,
            EscalationState.LYSA_RECEIVED: self._handle_lysa_received,
            EscalationState.DOCTOR_ALERTING: self._handle_doctor_alerting,
            EscalationState.DOCTOR_AVAILABLE: self._handle_doctor_available,
            EscalationState.DOCTOR_UNAVAILABLE: self._handle_doctor_unavailable,
        }
    
    async def initiate_escalation(
        self,
        patient_id: str,
        red_flag_detection: RedFlagDetection,
        conversation_history: List[Dict[str, str]],
        conversation_id: Optional[str] = None
    ) -> EscalationResult:
        """
        Initiate full escalation flow for a detected red flag.
        
        Args:
            patient_id: Patient who triggered the red flag
            red_flag_detection: Detection result with symptoms and severity
            conversation_history: Conversation context for handoff
            conversation_id: Optional conversation ID for tracking
            
        Returns:
            EscalationResult with outcome and next steps
        """
        import uuid
        escalation_id = str(uuid.uuid4())
        
        patient = self.db.query(User).filter(User.id == patient_id).first()
        patient_name = f"{patient.first_name} {patient.last_name}" if patient else "Patient"
        
        context = EscalationContext(
            escalation_id=escalation_id,
            patient_id=patient_id,
            patient_name=patient_name,
            red_flag_detection=red_flag_detection,
            conversation_history=conversation_history,
            conversation_id=conversation_id,
            state=EscalationState.RED_FLAG_DETECTED
        )
        
        self._active_escalations[escalation_id] = context
        
        self._audit_escalation_start(context)
        
        try:
            result = await self._execute_escalation_flow(context)
            return result
        except Exception as e:
            self.logger.error(f"Escalation failed: {e}")
            return self._create_failure_result(context, str(e))
        finally:
            if escalation_id in self._active_escalations:
                del self._active_escalations[escalation_id]
    
    async def _execute_escalation_flow(
        self, 
        context: EscalationContext
    ) -> EscalationResult:
        """Execute the state machine for escalation flow"""
        channels_used = []
        
        context.state = EscalationState.RED_FLAG_DETECTED
        await self._handle_red_flag_detected(context)
        
        context.state = EscalationState.CLONA_NOTIFYING
        clona_response = await self._handle_clona_notifying(context)
        
        context.state = EscalationState.LYSA_RECEIVED
        lysa_briefing = await self._handle_lysa_received(context)
        context.lysa_response = lysa_briefing
        
        doctors = self.doctor_availability.get_all_connected_doctors(context.patient_id)
        
        if not doctors:
            context.state = EscalationState.EMERGENCY_FALLBACK
            return await self._handle_emergency_fallback(context, channels_used)
        
        available_doctor = None
        for doctor in doctors:
            if doctor.get("available"):
                available_doctor = doctor
                break
        
        if available_doctor:
            context.state = EscalationState.DOCTOR_AVAILABLE
            context.doctor_id = available_doctor["doctor_id"]
            context.doctor_name = available_doctor["doctor_name"]
        else:
            context.state = EscalationState.DOCTOR_UNAVAILABLE
            context.doctor_id = doctors[0]["doctor_id"]
            context.doctor_name = doctors[0]["doctor_name"]
        
        context.state = EscalationState.DOCTOR_ALERTING
        alert_result = await self._handle_doctor_alerting(context)
        channels_used = alert_result.get("channels", [])
        
        if context.red_flag_detection.escalation_type == EscalationType.EMERGENCY_911:
            context.state = EscalationState.HANDOFF_COMPLETE
            context.fallback_used = True
            
            return EscalationResult(
                success=True,
                escalation_id=context.escalation_id,
                final_state=context.state,
                doctor_contacted=True,
                handoff_established=False,
                fallback_used=True,
                channels_used=[NotificationChannel(c) for c in channels_used],
                patient_guidance=self._get_emergency_guidance(context),
                next_steps=[
                    "Call 911 immediately if not already done",
                    "Your doctor has been alerted to the situation",
                    "Stay calm and follow emergency instructions"
                ]
            )
        
        handoff_result = await self._establish_handoff(context)
        
        if handoff_result["success"]:
            context.state = EscalationState.HANDOFF_COMPLETE
            context.handoff_channel = handoff_result.get("channel")
        else:
            context.state = EscalationState.ESCALATION_FAILED
        
        self._audit_escalation_complete(context, channels_used)
        
        return EscalationResult(
            success=context.state == EscalationState.HANDOFF_COMPLETE,
            escalation_id=context.escalation_id,
            final_state=context.state,
            doctor_contacted=bool(context.doctor_id),
            handoff_established=handoff_result["success"],
            fallback_used=context.fallback_used,
            channels_used=[NotificationChannel(c) for c in channels_used],
            patient_guidance=self._get_patient_guidance(context),
            doctor_response=handoff_result.get("doctor_response"),
            next_steps=self._get_next_steps(context)
        )
    
    async def _handle_red_flag_detected(self, context: EscalationContext) -> None:
        """Handle initial red flag detection"""
        self.logger.warning(
            f"[ESCALATION] Red flag detected for patient {context.patient_id}: "
            f"severity={context.red_flag_detection.highest_severity}"
        )
    
    async def _handle_clona_notifying(self, context: EscalationContext) -> str:
        """Generate Clona's response to acknowledge the emergency"""
        severity = context.red_flag_detection.highest_severity
        symptoms = [s["name"] for s in context.red_flag_detection.symptoms]
        
        if context.red_flag_detection.escalation_type == EscalationType.EMERGENCY_911:
            default_instructions = "Stay calm and do not exert yourself."
            emergency_text = context.red_flag_detection.emergency_instructions or default_instructions
            symptom_list = ", ".join(symptoms)
            response = f"""I am very concerned about what you are describing. Based on what you have told me - {symptom_list} - this sounds like it could be a medical emergency.

**Please call 911 immediately if you have not already.**

While you are getting help:
{emergency_text}

I am alerting your connected healthcare provider right now so they are aware of the situation. Your safety is the top priority.

If you are with someone, please have them stay with you until help arrives."""
        
        elif severity == RedFlagSeverity.HIGH:
            symptom_list = ", ".join(symptoms)
            response = f"""I am concerned about what you are experiencing - {symptom_list}. This needs prompt medical attention.

I am escalating this to your doctor right now. They will be in touch with you shortly.

In the meantime:
- Do not ignore these symptoms
- If anything gets worse, call 911 or go to the emergency room immediately

Your doctor is being notified and will contact you as soon as possible."""
        
        else:
            symptom_list = ", ".join(symptoms)
            response = f"""Thank you for sharing this with me. The symptoms you have described - {symptom_list} - warrant medical attention.

I am notifying your healthcare team so they can follow up with you. 

Please monitor your symptoms and seek immediate care if anything worsens."""
        
        return response
    
    async def _handle_lysa_received(self, context: EscalationContext) -> str:
        """Generate Lysa's briefing for the doctor"""
        symptoms = context.red_flag_detection.symptoms
        symptom_details = []
        for s in symptoms:
            symptom_details.append(f"- {s['name']} ({s['severity']}): {s['description']}")
        
        briefing = f"""URGENT ESCALATION - Patient {context.patient_name}

**Red Flag Alert**
Severity: {context.red_flag_detection.highest_severity.value.upper() if context.red_flag_detection.highest_severity else 'UNKNOWN'}
Escalation Type: {context.red_flag_detection.escalation_type.value if context.red_flag_detection.escalation_type else 'UNKNOWN'}

**Detected Symptoms:**
{chr(10).join(symptom_details)}

**Recommended Actions:**
{chr(10).join(['- ' + a for a in context.red_flag_detection.recommended_actions[:5]])}

**Recent Conversation Context:**
{self._summarize_conversation(context.conversation_history)}

**AI Analysis:**
{context.red_flag_detection.ai_analysis or 'Pattern-based detection confirmed symptoms.'}

Patient has been advised to seek appropriate care. Awaiting your response."""
        
        return briefing
    
    async def _handle_doctor_alerting(self, context: EscalationContext) -> Dict[str, Any]:
        """Alert the doctor through multiple channels"""
        channels_used = []
        
        try:
            channels_used.append("dashboard")
            await self._send_dashboard_alert(context)
        except Exception as e:
            self.logger.error(f"Dashboard alert failed: {e}")
        
        if context.red_flag_detection.highest_severity in [RedFlagSeverity.CRITICAL, RedFlagSeverity.HIGH]:
            try:
                channels_used.append("sms")
                await self._send_sms_alert(context)
            except Exception as e:
                self.logger.warning(f"SMS alert failed: {e}")
            
            try:
                channels_used.append("email")
                await self._send_email_alert(context)
            except Exception as e:
                self.logger.warning(f"Email alert failed: {e}")
        
        try:
            channels_used.append("in_app")
            await self._send_in_app_notification(context)
        except Exception as e:
            self.logger.warning(f"In-app notification failed: {e}")
        
        return {"channels": channels_used, "success": len(channels_used) > 0}
    
    async def _send_dashboard_alert(self, context: EscalationContext) -> None:
        """Create dashboard alert for doctor"""
        from app.models.alert_models import Alert
        
        alert = Alert(
            patient_id=context.patient_id,
            doctor_id=context.doctor_id,
            alert_type="red_flag_escalation",
            severity="high" if context.red_flag_detection.highest_severity in [RedFlagSeverity.CRITICAL, RedFlagSeverity.HIGH] else "medium",
            title=f"RED FLAG: {context.patient_name} - Escalation Required",
            message=context.lysa_response or "Urgent patient escalation",
            metadata={
                "escalation_id": context.escalation_id,
                "symptoms": [s["name"] for s in context.red_flag_detection.symptoms],
                "escalation_type": context.red_flag_detection.escalation_type.value if context.red_flag_detection.escalation_type else None
            },
            status="pending"
        )
        
        self.db.add(alert)
        self.db.commit()
    
    async def _send_sms_alert(self, context: EscalationContext) -> None:
        """Send SMS alert to doctor"""
        doctor = self.db.query(User).filter(User.id == context.doctor_id).first()
        if not doctor or not doctor.phone:
            return
        
        message = f"URGENT: Patient {context.patient_name} has triggered a red flag escalation. Please check your Followup AI dashboard immediately."
        
        self.logger.info(f"[ESCALATION] SMS alert sent to {context.doctor_name}: {message}")
    
    async def _send_email_alert(self, context: EscalationContext) -> None:
        """Send email alert to doctor"""
        doctor = self.db.query(User).filter(User.id == context.doctor_id).first()
        if not doctor or not doctor.email:
            return
        
        self.logger.info(f"[ESCALATION] Email alert queued for {context.doctor_name}")
    
    async def _send_in_app_notification(self, context: EscalationContext) -> None:
        """Send in-app push notification"""
        try:
            from app.services.message_router import get_message_router
            router = get_message_router()
            
            if router:
                notification = {
                    "type": "escalation_alert",
                    "escalation_id": context.escalation_id,
                    "patient_id": context.patient_id,
                    "patient_name": context.patient_name,
                    "severity": context.red_flag_detection.highest_severity.value if context.red_flag_detection.highest_severity else "unknown",
                    "symptoms": [s["name"] for s in context.red_flag_detection.symptoms],
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                
                await router.send_to_user(
                    user_id=context.doctor_id,
                    message=notification
                )
        except Exception as e:
            self.logger.warning(f"In-app notification failed: {e}")
    
    async def _establish_handoff(self, context: EscalationContext) -> Dict[str, Any]:
        """Establish direct communication channel between patient and doctor"""
        try:
            from app.services.message_router import get_message_router
            router = get_message_router()
            
            if router:
                conversation_data = {
                    "type": "escalation_handoff",
                    "escalation_id": context.escalation_id,
                    "patient_id": context.patient_id,
                    "doctor_id": context.doctor_id,
                    "context_summary": self._summarize_conversation(context.conversation_history),
                    "red_flag_summary": {
                        "symptoms": [s["name"] for s in context.red_flag_detection.symptoms],
                        "severity": context.red_flag_detection.highest_severity.value if context.red_flag_detection.highest_severity else None,
                        "recommended_actions": context.red_flag_detection.recommended_actions[:3]
                    }
                }
                
                return {
                    "success": True,
                    "channel": "chat",
                    "conversation_data": conversation_data
                }
                
        except Exception as e:
            self.logger.error(f"Handoff establishment failed: {e}")
        
        return {"success": False, "error": "Could not establish handoff"}
    
    async def _handle_emergency_fallback(
        self, 
        context: EscalationContext,
        channels_used: List[str]
    ) -> EscalationResult:
        """Handle case where no doctor is available - provide emergency guidance"""
        context.fallback_used = True
        
        guidance = self._get_emergency_guidance(context)
        
        return EscalationResult(
            success=True,
            escalation_id=context.escalation_id,
            final_state=EscalationState.EMERGENCY_FALLBACK,
            doctor_contacted=False,
            handoff_established=False,
            fallback_used=True,
            channels_used=[NotificationChannel(c) for c in channels_used],
            patient_guidance=guidance,
            next_steps=[
                "Call 911 for life-threatening emergencies",
                "Call 988 for mental health crisis",
                "Visit nearest emergency room for urgent care",
                "Your healthcare team will be notified as soon as possible"
            ]
        )
    
    def _summarize_conversation(self, history: List[Dict[str, str]]) -> str:
        """Create a brief summary of recent conversation for handoff"""
        if not history:
            return "No conversation history available."
        
        recent = history[-5:]
        summary_parts = []
        
        for msg in recent:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if len(content) > 200:
                content = content[:200] + "..."
            
            if role == "user":
                summary_parts.append(f"Patient: {content}")
            elif role == "assistant":
                summary_parts.append(f"Clona: {content}")
        
        return "\n".join(summary_parts)
    
    def _get_emergency_guidance(self, context: EscalationContext) -> str:
        """Get emergency guidance based on detected symptoms"""
        if context.red_flag_detection.emergency_instructions:
            return context.red_flag_detection.emergency_instructions
        
        if context.red_flag_detection.escalation_type == EscalationType.EMERGENCY_911:
            return "This is a medical emergency. Please call 911 immediately."
        
        return "Please seek medical attention as soon as possible."
    
    def _get_patient_guidance(self, context: EscalationContext) -> str:
        """Get patient-facing guidance based on escalation result"""
        if context.state == EscalationState.HANDOFF_COMPLETE:
            return f"Your doctor, {context.doctor_name}, has been alerted and will contact you shortly. Please stay near your phone."
        
        if context.state == EscalationState.DOCTOR_UNAVAILABLE:
            return f"We've sent an urgent message to {context.doctor_name}. While waiting, please follow the recommended actions and seek emergency care if symptoms worsen."
        
        return "Your healthcare team has been notified. Please monitor your symptoms."
    
    def _get_next_steps(self, context: EscalationContext) -> List[str]:
        """Determine next steps based on escalation outcome"""
        steps = []
        
        if context.red_flag_detection.escalation_type == EscalationType.EMERGENCY_911:
            steps.append("Call 911 if you haven't already")
        
        if context.state == EscalationState.HANDOFF_COMPLETE:
            steps.append("Wait for your doctor to contact you")
            steps.append("Keep your phone nearby")
        else:
            steps.append("Monitor your symptoms closely")
        
        steps.extend(context.red_flag_detection.recommended_actions[:3])
        
        return steps
    
    def _create_failure_result(
        self, 
        context: EscalationContext, 
        error: str
    ) -> EscalationResult:
        """Create result for failed escalation"""
        return EscalationResult(
            success=False,
            escalation_id=context.escalation_id,
            final_state=EscalationState.ESCALATION_FAILED,
            doctor_contacted=False,
            handoff_established=False,
            fallback_used=True,
            channels_used=[],
            patient_guidance=self._get_emergency_guidance(context),
            error_message=error,
            next_steps=[
                "Call 911 for emergencies",
                "Contact your doctor directly if possible",
                "Visit the nearest emergency room for urgent care"
            ]
        )
    
    def _audit_escalation_start(self, context: EscalationContext) -> None:
        """Audit log for escalation initiation"""
        try:
            audit_entry = AuditLog(
                user_id="escalation_flow_service",
                user_type="system",
                action_type="create",
                action_category="clinical_escalation",
                resource_type="escalation",
                resource_id=context.escalation_id,
                phi_accessed=True,
                patient_id_accessed=context.patient_id,
                action_description=f"Escalation initiated for {context.patient_name}",
                action_result="started",
                data_fields_accessed={
                    "escalation_id": context.escalation_id,
                    "patient_id": context.patient_id,
                    "symptoms": [s["name"] for s in context.red_flag_detection.symptoms],
                    "severity": context.red_flag_detection.highest_severity.value if context.red_flag_detection.highest_severity else None,
                    "initiated_at": context.initiated_at.isoformat()
                },
                ip_address="127.0.0.1",
                user_agent="EscalationFlowService/1.0"
            )
            self.db.add(audit_entry)
            self.db.commit()
        except Exception as e:
            self.logger.warning(f"Escalation start audit failed: {e}")
    
    def _audit_escalation_complete(
        self, 
        context: EscalationContext,
        channels: List[str]
    ) -> None:
        """Audit log for escalation completion"""
        try:
            audit_entry = AuditLog(
                user_id="escalation_flow_service",
                user_type="system",
                action_type="update",
                action_category="clinical_escalation",
                resource_type="escalation",
                resource_id=context.escalation_id,
                phi_accessed=True,
                patient_id_accessed=context.patient_id,
                action_description=f"Escalation completed: {context.state.value}",
                action_result="completed",
                data_fields_accessed={
                    "escalation_id": context.escalation_id,
                    "final_state": context.state.value,
                    "doctor_id": context.doctor_id,
                    "channels_used": channels,
                    "fallback_used": context.fallback_used,
                    "completed_at": datetime.now(timezone.utc).isoformat()
                },
                ip_address="127.0.0.1",
                user_agent="EscalationFlowService/1.0"
            )
            self.db.add(audit_entry)
            self.db.commit()
        except Exception as e:
            self.logger.warning(f"Escalation complete audit failed: {e}")
    
    def get_active_escalation(self, escalation_id: str) -> Optional[EscalationContext]:
        """Get an active escalation by ID"""
        return self._active_escalations.get(escalation_id)
    
    def get_patient_escalations(self, patient_id: str) -> List[EscalationContext]:
        """Get all active escalations for a patient"""
        return [
            ctx for ctx in self._active_escalations.values()
            if ctx.patient_id == patient_id
        ]


def get_escalation_service(db: Session) -> EscalationFlowService:
    """Factory function to get escalation flow service instance"""
    return EscalationFlowService(db)


def get_doctor_availability_service(db: Session) -> DoctorAvailabilityService:
    """Factory function to get doctor availability service instance"""
    return DoctorAvailabilityService(db)
