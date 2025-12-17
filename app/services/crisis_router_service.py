"""
Crisis Router Service for Mental Health Escalation

Production-grade crisis detection and routing service for mental health emergencies.
Handles automatic escalation to connected doctors when crisis indicators are detected.

HIPAA Compliance:
- All operations are audit logged
- PHI is handled securely with encryption
- Consent verification before sharing with connected doctors

Safety Focus:
- Immediate intervention messaging for patients
- Automatic escalation to connected healthcare providers
- Multi-channel notification delivery (dashboard, email, SMS)
"""

import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from sqlalchemy.orm import Session
from sqlalchemy import and_

from app.models.patient_doctor_connection import PatientDoctorConnection
from app.models.user import User
from app.models.mental_health_models import MentalHealthResponse
from app.models.security_models import AuditLog
from app.services.alert_orchestration_engine import AlertOrchestrationEngine
from app.models.alert_models import AlertRule, Alert
from app.models.trend_models import RiskEvent

logger = logging.getLogger(__name__)


class CrisisRouterService:
    """
    Crisis routing service for mental health emergencies.
    
    Responsibilities:
    1. Detect PHQ-9 question 9 (suicidal ideation) responses
    2. Evaluate crisis severity levels
    3. Find and notify connected doctors
    4. Generate crisis alerts for immediate attention
    5. Audit log all crisis events for HIPAA compliance
    """
    
    CRISIS_SEVERITY_THRESHOLDS = {
        "severe": 3,      # Nearly every day
        "high": 2,        # More than half the days  
        "moderate": 1,    # Several days
    }
    
    def __init__(self, db: Session):
        self.db = db
        self.logger = logging.getLogger(__name__)
        self.alert_engine = AlertOrchestrationEngine(db)
    
    async def evaluate_crisis(
        self,
        patient_id: str,
        questionnaire_type: str,
        responses: List[Dict[str, Any]],
        total_score: int,
        max_score: int,
        crisis_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluate crisis indicators and route to connected doctors if needed.
        
        Args:
            patient_id: Patient identifier
            questionnaire_type: Type of questionnaire (PHQ9, GAD7, PSS10)
            responses: List of question responses
            total_score: Total questionnaire score
            max_score: Maximum possible score
            crisis_result: Crisis detection result from MentalHealthService
            
        Returns:
            Crisis routing result with escalation status
        """
        routing_result = {
            "crisis_detected": crisis_result.get("crisis_detected", False),
            "severity": crisis_result.get("severity", "none"),
            "escalated": False,
            "doctors_notified": [],
            "alert_ids": [],
            "next_steps": crisis_result.get("next_steps", [])
        }
        
        if not crisis_result.get("crisis_detected", False):
            return routing_result
        
        self.logger.warning(
            f"[CRISIS-ROUTER] Crisis detected for patient {patient_id}: "
            f"severity={crisis_result['severity']}"
        )
        
        connected_doctors = self._get_connected_doctors(patient_id)
        
        self._audit_crisis_detection(
            patient_id=patient_id,
            questionnaire_type=questionnaire_type,
            severity=crisis_result["severity"],
            crisis_responses=crisis_result.get("crisis_responses", []),
            connected_doctor_ids=[d.id for d in connected_doctors] if connected_doctors else []
        )
        
        if not connected_doctors:
            self.logger.warning(
                f"[CRISIS-ROUTER] No connected doctors for patient {patient_id}"
            )
            routing_result["escalation_note"] = "No connected healthcare providers found"
            return routing_result
        
        alert_ids = await self._escalate_to_doctors(
            patient_id=patient_id,
            doctors=connected_doctors,
            questionnaire_type=questionnaire_type,
            severity=crisis_result["severity"],
            crisis_responses=crisis_result.get("crisis_responses", []),
            total_score=total_score,
            max_score=max_score
        )
        
        routing_result["escalated"] = True
        routing_result["doctors_notified"] = [d.id for d in connected_doctors]
        routing_result["alert_ids"] = alert_ids
        
        return routing_result
    
    def _get_connected_doctors(self, patient_id: str) -> List[User]:
        """Get all doctors connected to this patient with active connections"""
        connections = self.db.query(PatientDoctorConnection).filter(
            and_(
                PatientDoctorConnection.patient_id == patient_id,
                PatientDoctorConnection.status == "connected"
            )
        ).all()
        
        if not connections:
            return []
        
        doctor_ids = [c.doctor_id for c in connections]
        
        doctors = self.db.query(User).filter(
            and_(
                User.id.in_(doctor_ids),
                User.role == "doctor"
            )
        ).all()
        
        return doctors
    
    async def _escalate_to_doctors(
        self,
        patient_id: str,
        doctors: List[User],
        questionnaire_type: str,
        severity: str,
        crisis_responses: List[Dict[str, Any]],
        total_score: int,
        max_score: int
    ) -> List[int]:
        """
        Escalate crisis to connected doctors via alerts.
        
        Returns:
            List of generated alert IDs
        """
        alert_ids = []
        
        patient = self.db.query(User).filter(User.id == patient_id).first()
        patient_name = f"{patient.first_name} {patient.last_name}" if patient else "Patient"
        
        for doctor in doctors:
            try:
                alert_id = await self._create_crisis_alert(
                    patient_id=patient_id,
                    patient_name=patient_name,
                    doctor_id=doctor.id,
                    questionnaire_type=questionnaire_type,
                    severity=severity,
                    crisis_responses=crisis_responses,
                    total_score=total_score,
                    max_score=max_score
                )
                
                if alert_id:
                    alert_ids.append(alert_id)
                    
                    self._audit_escalation(
                        patient_id=patient_id,
                        doctor_id=doctor.id,
                        alert_id=alert_id,
                        severity=severity,
                        delivery_channels=["dashboard", "email", "sms"]
                    )
                    
            except Exception as e:
                self.logger.error(
                    f"[CRISIS-ROUTER] Failed to escalate to doctor {doctor.id}: {e}"
                )
        
        return alert_ids
    
    async def _create_crisis_alert(
        self,
        patient_id: str,
        patient_name: str,
        doctor_id: str,
        questionnaire_type: str,
        severity: str,
        crisis_responses: List[Dict[str, Any]],
        total_score: int,
        max_score: int
    ) -> Optional[int]:
        """Create a crisis alert for a specific doctor"""
        
        severity_mapping = {
            "severe": "high",
            "high": "high", 
            "moderate": "medium"
        }
        alert_severity = severity_mapping.get(severity, "medium")
        
        title = f"CRISIS ALERT: Mental Health Crisis Indicator - {patient_name}"
        
        message_parts = [
            f"Patient {patient_name} has reported crisis indicators on their {questionnaire_type} questionnaire.",
            f"",
            f"Severity: {severity.upper()}",
            f"Overall Score: {total_score}/{max_score}",
            f"",
            "Crisis Responses Detected:"
        ]
        
        for cr in crisis_responses:
            response_label = self._get_response_label(cr.get("response", 0))
            message_parts.append(f"• {cr.get('questionText', 'Question')}: {response_label}")
        
        message_parts.extend([
            "",
            "RECOMMENDED ACTIONS:",
            "1. Review patient's mental health history immediately",
            "2. Contact patient within 24 hours for wellness check",
            "3. Consider referral to mental health specialist if appropriate",
            "",
            "Note: This is a wellness monitoring alert. The patient has been provided with crisis resources including 988 Suicide & Crisis Lifeline."
        ])
        
        message = "\n".join(message_parts)
        
        try:
            alert = Alert(
                patient_id=patient_id,
                doctor_id=doctor_id,
                alert_type="mental_health_crisis",
                severity=alert_severity,
                title=title,
                message=message,
                metadata={
                    "questionnaire_type": questionnaire_type,
                    "crisis_severity": severity,
                    "total_score": total_score,
                    "max_score": max_score,
                    "crisis_responses": crisis_responses
                },
                status="pending"
            )
            
            self.db.add(alert)
            self.db.commit()
            self.db.refresh(alert)
            
            await self.alert_engine._deliver_alert_to_doctor(
                alert=alert,
                doctor_id=doctor_id
            )
            
            self.logger.info(
                f"[CRISIS-ROUTER] Created crisis alert {alert.id} for doctor {doctor_id}"
            )
            
            return alert.id
            
        except Exception as e:
            self.logger.error(f"[CRISIS-ROUTER] Failed to create alert: {e}")
            self.db.rollback()
            return None
    
    def _get_response_label(self, response_value: int) -> str:
        """Convert numeric response to human-readable label"""
        labels = {
            0: "Not at all",
            1: "Several days",
            2: "More than half the days",
            3: "Nearly every day"
        }
        return labels.get(response_value, f"Response value: {response_value}")
    
    def _audit_crisis_detection(
        self,
        patient_id: str,
        questionnaire_type: str,
        severity: str,
        crisis_responses: List[Dict[str, Any]],
        connected_doctor_ids: Optional[List[str]] = None
    ) -> None:
        """Audit log for crisis detection - HIPAA compliant (system-initiated)"""
        try:
            audit_entry = AuditLog(
                user_id="crisis_router_service",
                user_type="system",
                action_type="create",
                action_category="clinical_alert",
                resource_type="mental_health_crisis",
                resource_id="",
                phi_accessed=True,
                patient_id_accessed=patient_id,
                action_description=f"Mental health crisis detected via {questionnaire_type} - severity: {severity}",
                action_result="success",
                data_fields_accessed={
                    "patient_id": patient_id,
                    "questionnaire_type": questionnaire_type,
                    "severity": severity,
                    "crisis_question_count": len(crisis_responses),
                    "detected_at": datetime.now(timezone.utc).isoformat(),
                    "doctors_to_notify": connected_doctor_ids or [],
                    "notification_channels": ["dashboard", "email", "sms"],
                    "consent_basis": "patient_doctor_connection_active"
                },
                ip_address="127.0.0.1",
                user_agent="CrisisRouterService/1.0"
            )
            self.db.add(audit_entry)
            self.db.commit()
        except Exception as e:
            self.logger.warning(f"Crisis detection audit log failed (non-blocking): {e}")
    
    def _audit_escalation(
        self,
        patient_id: str,
        doctor_id: str,
        alert_id: int,
        severity: str,
        delivery_channels: Optional[List[str]] = None
    ) -> None:
        """Audit log for crisis escalation - HIPAA compliant (system-initiated)"""
        try:
            connection = self.db.query(PatientDoctorConnection).filter(
                and_(
                    PatientDoctorConnection.patient_id == patient_id,
                    PatientDoctorConnection.doctor_id == doctor_id,
                    PatientDoctorConnection.status == "connected"
                )
            ).first()
            
            consent_verified = connection is not None
            
            audit_entry = AuditLog(
                user_id="crisis_router_service",
                user_type="system",
                action_type="share",
                action_category="clinical_alert",
                resource_type="crisis_escalation",
                resource_id=str(alert_id),
                phi_accessed=True,
                patient_id_accessed=patient_id,
                action_description=f"Crisis alert {alert_id} escalated to connected doctor for patient {patient_id}",
                action_result="success",
                data_fields_accessed={
                    "patient_id": patient_id,
                    "doctor_id": doctor_id,
                    "severity": severity,
                    "alert_id": alert_id,
                    "escalated_at": datetime.now(timezone.utc).isoformat(),
                    "escalation_reason": "mental_health_crisis_indicator",
                    "consent_verified": consent_verified,
                    "connection_status": connection.status if connection else "not_found",
                    "delivery_channels": delivery_channels or ["dashboard"]
                },
                ip_address="127.0.0.1",
                user_agent="CrisisRouterService/1.0"
            )
            self.db.add(audit_entry)
            self.db.commit()
        except Exception as e:
            self.logger.warning(f"Crisis escalation audit log failed (non-blocking): {e}")
