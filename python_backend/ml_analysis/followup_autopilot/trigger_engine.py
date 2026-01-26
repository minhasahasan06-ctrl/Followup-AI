"""
Trigger Engine for Followup Autopilot

Implements trigger playbooks:
1. missed_meds_pattern - Medication adherence issues
2. env_risk_high_with_symptoms - Environmental + symptom combination
3. mh_score_spike - Mental health score elevation
4. anomaly_day - Unusual day detected
5. exposure_risk_elevated - Risk & Exposures concerns

Each trigger has severity, cooldown, and generates patient tasks.
High/critical triggers also create alerts via AlertOrchestrationEngine.

All outputs are for wellness monitoring only, NOT medical diagnosis.
"""

import os
import logging
from datetime import datetime, date, timedelta, timezone
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
from uuid import uuid4

logger = logging.getLogger(__name__)


@dataclass
class TriggerDefinition:
    """Definition of a trigger playbook"""
    name: str
    severity: str
    cooldown_hours: int
    task_type: str
    task_priority: str
    ui_tab_target: str
    reason_template: str


TRIGGERS = {
    "missed_meds_pattern": TriggerDefinition(
        name="missed_meds_pattern",
        severity="warning",
        cooldown_hours=24,
        task_type="med_adherence_check",
        task_priority="medium",
        ui_tab_target="medications",
        reason_template="Medication adherence has been below optimal levels. Please review your medication schedule."
    ),
    "env_risk_high_with_symptoms": TriggerDefinition(
        name="env_risk_high_with_symptoms",
        severity="alert",
        cooldown_hours=12,
        task_type="resp_symptom_check",
        task_priority="high",
        ui_tab_target="symptoms",
        reason_template="Environmental conditions combined with symptoms suggest a wellness check. Consider staying indoors if possible."
    ),
    "mh_score_spike": TriggerDefinition(
        name="mh_score_spike",
        severity="alert",
        cooldown_hours=24,
        task_type="mh_check",
        task_priority="high",
        ui_tab_target="mental_health",
        reason_template="Your wellness scores indicate increased stress. A mental health check-in may be helpful."
    ),
    "anomaly_day": TriggerDefinition(
        name="anomaly_day",
        severity="warning",
        cooldown_hours=24,
        task_type="symptom_check",
        task_priority="medium",
        ui_tab_target="symptoms",
        reason_template="Today's patterns are different from your usual baseline. A quick check-in would help us understand how you're doing."
    ),
    "exposure_risk_elevated": TriggerDefinition(
        name="exposure_risk_elevated",
        severity="alert",
        cooldown_hours=48,
        task_type="exposure_check",
        task_priority="high",
        ui_tab_target="risk_exposures",
        reason_template="Potential exposure risk detected. Please review your risk status and take appropriate precautions."
    ),
    "pain_spike": TriggerDefinition(
        name="pain_spike",
        severity="warning",
        cooldown_hours=12,
        task_type="pain_check",
        task_priority="medium",
        ui_tab_target="paintrack",
        reason_template="Pain levels have increased. Please log your current pain status."
    ),
    "critical_wellness": TriggerDefinition(
        name="critical_wellness",
        severity="alert",
        cooldown_hours=6,
        task_type="urgent_check",
        task_priority="critical",
        ui_tab_target="symptoms",
        reason_template="IMPORTANT: Your wellness indicators require attention. Please contact your care team if you feel unwell."
    ),
}


class TriggerEngine:
    """
    Trigger evaluation and task generation engine.
    
    Responsibilities:
    1. Evaluate all triggers against patient data
    2. Respect cooldown windows
    3. Generate follow-up tasks for patients
    4. Create alerts for high/critical triggers
    5. Log trigger events for audit
    """
    
    def __init__(self, db_session=None):
        self.db = db_session
        self.logger = logging.getLogger(__name__)
    
    def run_triggers(
        self,
        patient_id: str,
        features_today: Dict[str, Any],
        patient_state: Dict[str, Any],
        risk_probs: Dict[str, float],
        anomaly_score: float
    ) -> List[Dict[str, Any]]:
        """
        Evaluate all triggers and create tasks/events.
        
        Args:
            patient_id: Patient identifier
            features_today: Today's aggregated features
            patient_state: Current patient state
            risk_probs: Risk model predictions
            anomaly_score: Anomaly detector score
            
        Returns:
            List of triggered events with task IDs
        """
        triggered_events = []
        
        trigger_checks = [
            ("missed_meds_pattern", self._check_missed_meds),
            ("env_risk_high_with_symptoms", self._check_env_risk),
            ("mh_score_spike", self._check_mh_spike),
            ("anomaly_day", self._check_anomaly),
            ("exposure_risk_elevated", self._check_exposure_risk),
            ("pain_spike", self._check_pain_spike),
            ("critical_wellness", self._check_critical_wellness),
        ]
        
        for trigger_name, check_fn in trigger_checks:
            if check_fn(features_today, patient_state, risk_probs, anomaly_score):
                if not self._is_in_cooldown(patient_id, trigger_name):
                    event = self._fire_trigger(
                        patient_id, trigger_name,
                        features_today, patient_state
                    )
                    if event:
                        triggered_events.append(event)
        
        return triggered_events
    
    def _check_missed_meds(
        self,
        features: Dict,
        state: Dict,
        risk_probs: Dict,
        anomaly_score: float
    ) -> bool:
        """Check for medication adherence issues"""
        adherence = features.get("med_adherence_7d", 1.0)
        p_non_adherence = risk_probs.get("p_non_adherence", 0)
        
        return adherence < 0.7 or p_non_adherence > 0.6
    
    def _check_env_risk(
        self,
        features: Dict,
        state: Dict,
        risk_probs: Dict,
        anomaly_score: float
    ) -> bool:
        """Check for high environmental risk with symptoms"""
        env_risk = features.get("env_risk_score", 0)
        avg_pain = features.get("avg_pain", 0)
        avg_fatigue = features.get("avg_fatigue", 0)
        risk_state = state.get("risk_state", "Stable")
        
        high_env = env_risk > 70
        has_symptoms = avg_pain > 5 or avg_fatigue > 6
        elevated_risk = risk_state in ("AtRisk", "Worsening", "Critical")
        
        return high_env and has_symptoms and elevated_risk
    
    def _check_mh_spike(
        self,
        features: Dict,
        state: Dict,
        risk_probs: Dict,
        anomaly_score: float
    ) -> bool:
        """Check for mental health score spike"""
        mh_score = features.get("mh_score", 0)
        p_mh_crisis = risk_probs.get("p_mental_health_crisis", 0)
        
        return mh_score > 0.6 or p_mh_crisis > 0.5
    
    def _check_anomaly(
        self,
        features: Dict,
        state: Dict,
        risk_probs: Dict,
        anomaly_score: float
    ) -> bool:
        """Check for anomalous day"""
        return anomaly_score > 0.6
    
    def _check_exposure_risk(
        self,
        features: Dict,
        state: Dict,
        risk_probs: Dict,
        anomaly_score: float
    ) -> bool:
        """Check for elevated exposure risk"""
        infectious = features.get("infectious_exposure_score", 0)
        immunization = features.get("immunization_status", 1.0)
        occupational = features.get("occupational_risk_score", 0)
        
        return infectious > 0.5 or immunization < 0.5 or occupational > 0.6
    
    def _check_pain_spike(
        self,
        features: Dict,
        state: Dict,
        risk_probs: Dict,
        anomaly_score: float
    ) -> bool:
        """Check for pain level spike"""
        avg_pain = features.get("avg_pain", 0)
        pain_severity = features.get("pain_severity_score", 0)
        
        return avg_pain > 7 or pain_severity > 0.7
    
    def _check_critical_wellness(
        self,
        features: Dict,
        state: Dict,
        risk_probs: Dict,
        anomaly_score: float
    ) -> bool:
        """Check for critical wellness state"""
        risk_state = state.get("risk_state", "Stable")
        risk_score = state.get("risk_score", 0)
        p_clinical = risk_probs.get("p_clinical_deterioration", 0)
        
        return risk_state == "Critical" or risk_score > 80 or p_clinical > 0.7
    
    def _is_in_cooldown(self, patient_id: str, trigger_name: str) -> bool:
        """Check if trigger is in cooldown period"""
        trigger_def = TRIGGERS.get(trigger_name)
        if not trigger_def:
            return False
            
        cooldown_hours = trigger_def.cooldown_hours
        cooldown_start = datetime.now(timezone.utc) - timedelta(hours=cooldown_hours)
        
        if self.db:
            from app.models.followup_autopilot_models import AutopilotTriggerEvent
            
            recent = self.db.query(AutopilotTriggerEvent).filter(
                AutopilotTriggerEvent.patient_id == patient_id,
                AutopilotTriggerEvent.name == trigger_name,
                AutopilotTriggerEvent.created_at >= cooldown_start
            ).first()
            
            return recent is not None
        
        return False
    
    def _fire_trigger(
        self,
        patient_id: str,
        trigger_name: str,
        features: Dict[str, Any],
        state: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Fire a trigger: create event, task, and possibly alert"""
        trigger_def = TRIGGERS.get(trigger_name)
        if not trigger_def:
            return None
        
        from .task_engine import TaskEngine
        task_engine = TaskEngine(self.db)
        
        task_id = task_engine.create_followup_task(
            patient_id=patient_id,
            task_type=trigger_def.task_type,
            priority=trigger_def.task_priority,
            due_at=self._compute_due_at(trigger_def.task_priority),
            trigger_name=trigger_name,
            reason=trigger_def.reason_template,
            ui_tab_target=trigger_def.ui_tab_target,
            metadata={"features_snapshot": {k: v for k, v in features.items() 
                                            if isinstance(v, (int, float, str, bool))}}
        )
        
        event_id = self._log_trigger_event(
            patient_id, trigger_name, trigger_def.severity,
            {"task_id": str(task_id)} if task_id else {}
        )
        
        alert_id = None
        if trigger_def.severity == "alert" and trigger_def.task_priority in ("high", "critical"):
            alert_id = self._create_alert(
                patient_id, trigger_name, trigger_def, features, state
            )
        
        approval_id = None
        if trigger_def.task_priority in ("high", "critical"):
            approval_id = self._create_hitl_approval(
                patient_id, trigger_name, trigger_def, features, state, event_id
            )
        
        return {
            "trigger_name": trigger_name,
            "severity": trigger_def.severity,
            "event_id": event_id,
            "task_id": task_id,
            "alert_id": alert_id,
            "approval_id": approval_id,
            "reason": trigger_def.reason_template,
        }
    
    def _compute_due_at(self, priority: str) -> datetime:
        """Compute task due time based on priority"""
        now = datetime.now(timezone.utc)
        
        if priority == "critical":
            return now + timedelta(hours=2)
        elif priority == "high":
            return now + timedelta(hours=6)
        elif priority == "medium":
            return now + timedelta(hours=24)
        else:
            return now + timedelta(hours=48)
    
    def _log_trigger_event(
        self,
        patient_id: str,
        trigger_name: str,
        severity: str,
        context: Dict[str, Any]
    ) -> Optional[str]:
        """Log trigger event for audit"""
        event_id = str(uuid4())
        
        try:
            if self.db:
                from app.models.followup_autopilot_models import AutopilotTriggerEvent
                
                event = AutopilotTriggerEvent(
                    id=event_id,
                    patient_id=patient_id,
                    name=trigger_name,
                    severity=severity,
                    context=context,
                    task_ids_created=[context.get("task_id")] if context.get("task_id") else [],
                )
                self.db.add(event)
                self.db.commit()
            else:
                self._log_event_raw(event_id, patient_id, trigger_name, severity, context)
                
            return event_id
        except Exception as e:
            self.logger.error(f"Failed to log trigger event: {e}")
            if self.db:
                self.db.rollback()
            return None
    
    def _create_alert(
        self,
        patient_id: str,
        trigger_name: str,
        trigger_def: TriggerDefinition,
        features: Dict[str, Any],
        state: Dict[str, Any]
    ) -> Optional[int]:
        """Create alert via AlertOrchestrationEngine"""
        try:
            from app.models.alert_models import Alert
            
            if self.db:
                alert = Alert(
                    alert_rule_id=1,
                    patient_id=patient_id,
                    alert_title=f"Wellness Alert: {trigger_name.replace('_', ' ').title()}",
                    alert_message=trigger_def.reason_template,
                    severity=trigger_def.task_priority,
                    urgency="urgent" if trigger_def.task_priority in ("high", "critical") else "routine",
                    trigger_metric=trigger_name,
                    risk_level_current=state.get("risk_state", "unknown"),
                    delivery_status="pending",
                    dashboard_sent=True,
                    dashboard_sent_at=datetime.now(timezone.utc),
                )
                self.db.add(alert)
                self.db.commit()
                return alert.id
        except Exception as e:
            self.logger.warning(f"Alert creation failed (non-critical): {e}")
            if self.db:
                self.db.rollback()
        return None
    
    def _create_hitl_approval(
        self,
        patient_id: str,
        trigger_name: str,
        trigger_def: TriggerDefinition,
        features: Dict[str, Any],
        state: Dict[str, Any],
        event_id: Optional[str] = None
    ) -> Optional[str]:
        """
        Create Human-in-the-Loop pending approval for doctor review.
        High/critical triggers require doctor oversight before actions are executed.
        """
        try:
            doctor_id = self._get_assigned_doctor(patient_id)
            if not doctor_id:
                self.logger.warning(f"No doctor assigned to patient {patient_id}, skipping HITL approval")
                return None
            
            action_type = self._map_trigger_to_action_type(trigger_name)
            
            ai_recommendation = self._generate_ai_recommendation(trigger_name, trigger_def, features, state)
            ai_reasoning = self._generate_ai_reasoning(trigger_name, features, state)
            
            confidence = 0.7 if trigger_def.task_priority == "critical" else 0.6
            
            patient_context = {
                "risk_score": state.get("risk_score", 0),
                "risk_state": state.get("risk_state", "Stable"),
                "med_adherence_7d": features.get("med_adherence_7d"),
                "avg_pain": features.get("avg_pain"),
                "mh_score": features.get("mh_score"),
                "env_risk_score": features.get("env_risk_score"),
            }
            
            if self.db:
                from app.models.followup_autopilot_models import AutopilotPendingApproval
                
                expires_hours = 24 if trigger_def.task_priority == "critical" else 48
                
                approval = AutopilotPendingApproval(
                    patient_id=patient_id,
                    doctor_id=doctor_id,
                    trigger_event_id=event_id,
                    action_type=action_type,
                    status="pending",
                    priority=trigger_def.task_priority,
                    title=f"Review: {trigger_name.replace('_', ' ').title()}",
                    ai_recommendation=ai_recommendation,
                    ai_reasoning=ai_reasoning,
                    confidence_score=confidence,
                    patient_context=patient_context,
                    risk_score=state.get("risk_score"),
                    risk_state=state.get("risk_state"),
                    expires_at=datetime.now(timezone.utc) + timedelta(hours=expires_hours)
                )
                self.db.add(approval)
                self.db.commit()
                self.logger.info(f"Created HITL approval {approval.id} for patient {patient_id}")
                return str(approval.id)
                
        except Exception as e:
            self.logger.error(f"Failed to create HITL approval: {e}")
            if self.db:
                self.db.rollback()
        return None
    
    def _get_assigned_doctor(self, patient_id: str) -> Optional[str]:
        """Get the assigned doctor for a patient"""
        if not self.db:
            return None
        
        try:
            from sqlalchemy import text
            
            result = self.db.execute(
                text("""
                    SELECT doctor_id FROM doctor_patient_assignments
                    WHERE patient_id = :patient_id AND status = 'active'
                    ORDER BY created_at DESC
                    LIMIT 1
                """),
                {"patient_id": patient_id}
            )
            row = result.fetchone()
            if row:
                return row[0]
            
            result = self.db.execute(
                text("""
                    SELECT assigned_doctor_id FROM users
                    WHERE id = :patient_id AND assigned_doctor_id IS NOT NULL
                """),
                {"patient_id": patient_id}
            )
            row = result.fetchone()
            if row:
                return row[0]
                
        except Exception as e:
            self.logger.warning(f"Failed to get assigned doctor: {e}")
        
        return None
    
    def _map_trigger_to_action_type(self, trigger_name: str) -> str:
        """Map trigger name to approval action type"""
        mapping = {
            "missed_meds_pattern": "send_reminder",
            "env_risk_high_with_symptoms": "request_checkin",
            "mh_score_spike": "schedule_followup",
            "anomaly_day": "request_checkin",
            "exposure_risk_elevated": "notify_patient",
            "pain_spike": "schedule_followup",
            "critical_wellness": "schedule_consultation",
        }
        return mapping.get(trigger_name, "notify_patient")
    
    def _generate_ai_recommendation(
        self,
        trigger_name: str,
        trigger_def: TriggerDefinition,
        features: Dict[str, Any],
        state: Dict[str, Any]
    ) -> str:
        """Generate AI recommendation text for doctor review"""
        recommendations = {
            "missed_meds_pattern": f"Patient has medication adherence of {features.get('med_adherence_7d', 0)*100:.0f}% over the past 7 days. Recommend sending a gentle reminder about medication importance and scheduling a brief check-in to understand barriers.",
            "env_risk_high_with_symptoms": f"Environmental risk score is {features.get('env_risk_score', 0):.0f} with reported pain level {features.get('avg_pain', 0):.1f}/10. Recommend requesting an immediate symptom check-in and advising the patient to limit outdoor exposure.",
            "mh_score_spike": f"Mental health indicators show elevated scores (MH score: {features.get('mh_score', 0):.2f}). Recommend scheduling a mental health follow-up and providing supportive resources.",
            "anomaly_day": "Patient's daily patterns are significantly different from their baseline. Recommend a wellness check-in to understand any changes in their routine or health status.",
            "exposure_risk_elevated": f"Elevated exposure risk detected (infectious: {features.get('infectious_exposure_score', 0):.2f}). Recommend notifying the patient about precautionary measures and monitoring symptoms.",
            "pain_spike": f"Pain levels have increased to {features.get('avg_pain', 0):.1f}/10. Recommend scheduling a follow-up to assess pain management and adjust treatment if needed.",
            "critical_wellness": f"Critical wellness state detected (Risk Score: {state.get('risk_score', 0):.0f}). Recommend urgent consultation to assess the patient's current condition and care needs.",
        }
        return recommendations.get(trigger_name, trigger_def.reason_template)
    
    def _generate_ai_reasoning(
        self,
        trigger_name: str,
        features: Dict[str, Any],
        state: Dict[str, Any]
    ) -> str:
        """Generate AI reasoning for the recommendation"""
        risk_state = state.get("risk_state", "Stable")
        risk_score = state.get("risk_score", 0)
        
        base_reasoning = f"Based on analysis of patient data, the current risk state is '{risk_state}' with a composite score of {risk_score:.0f}/100. "
        
        factors = []
        if features.get("med_adherence_7d", 1) < 0.8:
            factors.append(f"medication adherence at {features.get('med_adherence_7d', 0)*100:.0f}%")
        if features.get("avg_pain", 0) > 5:
            factors.append(f"elevated pain levels ({features.get('avg_pain', 0):.1f}/10)")
        if features.get("mh_score", 0) > 0.5:
            factors.append("concerning mental health indicators")
        if features.get("env_risk_score", 0) > 50:
            factors.append("high environmental risk exposure")
        
        if factors:
            base_reasoning += f"Key contributing factors: {', '.join(factors)}. "
        
        base_reasoning += "This recommendation is generated by the Autopilot wellness monitoring system and requires doctor approval before any action is taken."
        
        return base_reasoning

    def _log_event_raw(
        self,
        event_id: str,
        patient_id: str,
        trigger_name: str,
        severity: str,
        context: Dict[str, Any]
    ):
        """Direct database logging when ORM not available"""
        import psycopg2
        from psycopg2.extras import Json
        
        conn_str = os.environ.get("DATABASE_URL")
        if not conn_str:
            return
            
        with psycopg2.connect(conn_str) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO autopilot_trigger_events 
                    (id, patient_id, name, severity, context, created_at)
                    VALUES (%s, %s, %s, %s, %s, NOW())
                """, (event_id, patient_id, trigger_name, severity, Json(context)))
            conn.commit()
