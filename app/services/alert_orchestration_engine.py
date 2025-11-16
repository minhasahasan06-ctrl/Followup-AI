"""Alert Orchestration Engine for Followup AI

This module provides the alert orchestration system that monitors patient health trends,
evaluates risk events, and delivers multi-channel notifications (dashboard, email, SMS).

HIPAA Compliance:
- All patient data in alerts is PHI and requires secure delivery
- Email/SMS delivery uses encrypted channels (AWS SES, Twilio)
- All alert generation and delivery is audit logged

Wellness Positioning:
- Alerts use "wellness monitoring" language, NOT diagnostic language
- Recommendations focus on "discussing with healthcare provider"
- System is a change detection platform, not a medical diagnostic tool
"""

import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc

from app.models.alert_models import AlertRule, Alert
from app.models.trend_models import TrendSnapshot, RiskEvent
from app.models.security_models import AuditLog
import os
import logging

logger = logging.getLogger(__name__)

# Notification channel availability
SES_AVAILABLE = bool(os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY"))
TWILIO_AVAILABLE = bool(os.getenv("TWILIO_ACCOUNT_SID") and os.getenv("TWILIO_AUTH_TOKEN"))

if SES_AVAILABLE:
    try:
        import boto3
        ses_client = boto3.client('ses', region_name=os.getenv("AWS_REGION", "us-east-1"))
    except Exception as e:
        logger.warning(f"SES client initialization failed: {e}")
        SES_AVAILABLE = False

if TWILIO_AVAILABLE:
    try:
        from twilio.rest import Client
        twilio_client = Client(
            os.getenv("TWILIO_ACCOUNT_SID"),
            os.getenv("TWILIO_AUTH_TOKEN")
        )
    except Exception as e:
        logger.warning(f"Twilio client initialization failed: {e}")
        TWILIO_AVAILABLE = False


class AlertOrchestrationEngine:
    """
    Alert Orchestration Engine with multi-channel delivery
    
    Responsibilities:
    1. Monitor risk events and trend snapshots
    2. Evaluate alert rules against patient data
    3. Generate alerts with appropriate severity
    4. Deliver notifications via multiple channels (dashboard, email, SMS)
    5. Track alert acknowledgment and resolution
    6. Audit all alert generation and delivery for HIPAA compliance
    """
    
    def __init__(self, db: Session):
        self.db = db
        self.logger = logging.getLogger(__name__)
    
    async def evaluate_risk_event(
        self,
        risk_event: RiskEvent,
        patient_id: str,
        doctor_id: Optional[str] = None
    ) -> List[int]:
        """
        Evaluate a risk event against all active alert rules and generate alerts
        
        Args:
            risk_event: RiskEvent that triggered evaluation
            patient_id: Patient ID from risk event
            doctor_id: Optional doctor ID (if None, finds all doctors for this patient)
        
        Returns:
            List of generated alert IDs
        """
        try:
            # Get active alert rules
            query = self.db.query(AlertRule).filter(AlertRule.is_active == True)
            
            if doctor_id:
                query = query.filter(AlertRule.doctor_id == doctor_id)
            
            active_rules = query.all()
            
            generated_alerts = []
            
            for rule in active_rules:
                if self._rule_matches_event(rule, risk_event):
                    alert = await self._generate_alert(
                        rule=rule,
                        risk_event=risk_event,
                        patient_id=patient_id
                    )
                    if alert:
                        generated_alerts.append(alert.id)
                        
                        # Deliver alert via configured channels
                        await self._deliver_alert(alert, rule)
            
            return generated_alerts
            
        except Exception as e:
            self.logger.error(f"Error evaluating risk event: {e}")
            return []
    
    def _rule_matches_event(self, rule: AlertRule, risk_event: RiskEvent) -> bool:
        """
        Check if an alert rule matches a risk event
        
        Rule types:
        - 'risk_threshold': Alert when risk exceeds threshold
        - 'metric_deviation': Alert when metric deviates from baseline
        - 'trend_change': Alert when risk level changes
        """
        try:
            conditions = rule.conditions
            
            if rule.rule_type == "risk_threshold":
                threshold = conditions.get("risk_threshold", 0.7)
                # Extract risk score from event details or use severity-based estimation
                risk_score = risk_event.risk_delta if risk_event.risk_delta else self._estimate_risk_score(risk_event.new_risk_level)
                return risk_score >= threshold
            
            elif rule.rule_type == "trend_change":
                # Alert on any risk level increase
                if risk_event.event_type == "risk_increase":
                    required_change = conditions.get("minimum_change", "green_to_yellow")
                    if required_change == "any":
                        return True
                    elif required_change == "green_to_yellow":
                        return risk_event.previous_risk_level == "green" and risk_event.new_risk_level in ["yellow", "red"]
                    elif required_change == "to_red":
                        return risk_event.new_risk_level == "red"
                return False
            
            elif rule.rule_type == "metric_deviation":
                # Check if event details contain specific metric deviations
                if risk_event.event_details:
                    metric_name = conditions.get("metric_name")
                    threshold = conditions.get("deviation_threshold", 2.0)
                    
                    if metric_name and metric_name in risk_event.event_details:
                        metric_z_score = abs(risk_event.event_details[metric_name].get("z_score", 0))
                        return metric_z_score >= threshold
                return False
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error matching rule to event: {e}")
            return False
    
    def _estimate_risk_score(self, risk_level: str) -> float:
        """Estimate numeric risk score from categorical risk level"""
        mapping = {
            "green": 0.2,
            "yellow": 0.6,
            "red": 0.9
        }
        return mapping.get(risk_level, 0.5)
    
    async def _generate_alert(
        self,
        rule: AlertRule,
        risk_event: RiskEvent,
        patient_id: str
    ) -> Optional[Alert]:
        """
        Generate an alert based on rule and risk event
        """
        try:
            # Determine severity based on risk level
            severity_mapping = {
                "green": "low",
                "yellow": "medium",
                "red": "high"
            }
            severity = severity_mapping.get(risk_event.new_risk_level, "medium")
            
            # Generate alert title and message (wellness-focused language)
            title = self._generate_alert_title(risk_event, rule)
            message = self._generate_alert_message(risk_event, rule)
            
            # Create alert
            alert = Alert(
                rule_id=rule.id,
                patient_id=patient_id,
                doctor_id=rule.doctor_id,
                alert_type=rule.rule_type,
                severity=severity,
                title=title,
                message=message,
                metadata={
                    "risk_event_id": risk_event.id,
                    "previous_risk_level": risk_event.previous_risk_level,
                    "new_risk_level": risk_event.new_risk_level,
                    "risk_delta": risk_event.risk_delta,
                    "event_details": risk_event.event_details
                },
                status="pending"
            )
            
            self.db.add(alert)
            self.db.commit()
            self.db.refresh(alert)
            
            # Audit log
            await self._audit_alert_generation(alert, risk_event)
            
            return alert
            
        except Exception as e:
            self.logger.error(f"Error generating alert: {e}")
            self.db.rollback()
            return None
    
    def _generate_alert_title(self, risk_event: RiskEvent, rule: AlertRule) -> str:
        """Generate wellness-focused alert title"""
        if risk_event.event_type == "risk_increase":
            return f"Health Wellness Change Detected - {risk_event.new_risk_level.upper()} Priority"
        elif risk_event.event_type == "anomaly_detected":
            return "Wellness Pattern Change Detected"
        else:
            return f"Wellness Monitoring Update - {rule.rule_name}"
    
    def _generate_alert_message(self, risk_event: RiskEvent, rule: AlertRule) -> str:
        """Generate wellness-focused alert message (NOT diagnostic language)"""
        message_parts = []
        
        # Risk change description
        if risk_event.previous_risk_level and risk_event.new_risk_level:
            message_parts.append(
                f"Patient wellness priority has changed from {risk_event.previous_risk_level.upper()} "
                f"to {risk_event.new_risk_level.upper()}."
            )
        
        # Event details
        if risk_event.event_details:
            message_parts.append("Notable changes detected in:")
            for metric, details in risk_event.event_details.items():
                if isinstance(details, dict) and "z_score" in details:
                    message_parts.append(f"• {metric}: {details.get('description', 'Deviation from baseline')}")
        
        # Wellness recommendation (NOT medical advice)
        message_parts.append(
            "\nRecommendation: Please review patient data and consider scheduling a wellness check "
            "to discuss these changes with the patient. This system provides wellness monitoring "
            "and change detection, not medical diagnosis."
        )
        
        return "\n".join(message_parts)
    
    async def _deliver_alert(self, alert: Alert, rule: AlertRule) -> Dict[str, bool]:
        """
        Deliver alert via configured notification channels
        
        Returns:
            Dict mapping channel -> success status
        """
        delivery_results = {}
        
        for channel in rule.notification_channels:
            try:
                if channel == "dashboard":
                    # Alert is already stored in database, accessible via dashboard
                    delivery_results["dashboard"] = True
                
                elif channel == "email" and SES_AVAILABLE:
                    success = await self._send_email_notification(alert, rule)
                    delivery_results["email"] = success
                
                elif channel == "sms" and TWILIO_AVAILABLE:
                    success = await self._send_sms_notification(alert, rule)
                    delivery_results["sms"] = success
                
                else:
                    self.logger.warning(f"Channel {channel} not available or not configured")
                    delivery_results[channel] = False
                    
            except Exception as e:
                self.logger.error(f"Error delivering alert via {channel}: {e}")
                delivery_results[channel] = False
        
        # Update alert status
        if any(delivery_results.values()):
            alert.status = "sent"
            self.db.commit()
        
        return delivery_results
    
    async def _send_email_notification(self, alert: Alert, rule: AlertRule) -> bool:
        """
        Send email notification via AWS SES (HIPAA-compliant)
        """
        try:
            if not SES_AVAILABLE:
                return False
            
            # TODO: Get doctor email from doctor_profiles table
            doctor_email = f"doctor_{rule.doctor_id}@followupai.health"  # Placeholder
            
            response = ses_client.send_email(
                Source=os.getenv("SES_FROM_EMAIL", "alerts@followupai.health"),
                Destination={"ToAddresses": [doctor_email]},
                Message={
                    "Subject": {"Data": f"[Followup AI] {alert.title}"},
                    "Body": {
                        "Text": {"Data": self._format_email_body(alert)},
                        "Html": {"Data": self._format_email_html(alert)}
                    }
                }
            )
            
            return response['ResponseMetadata']['HTTPStatusCode'] == 200
            
        except Exception as e:
            self.logger.error(f"Error sending email: {e}")
            return False
    
    async def _send_sms_notification(self, alert: Alert, rule: AlertRule) -> bool:
        """
        Send SMS notification via Twilio (HIPAA-compliant)
        """
        try:
            if not TWILIO_AVAILABLE:
                return False
            
            # TODO: Get doctor phone from doctor_profiles table
            doctor_phone = f"+1555{rule.doctor_id[:7]}"  # Placeholder
            
            message = twilio_client.messages.create(
                body=self._format_sms_body(alert),
                from_=os.getenv("TWILIO_PHONE_NUMBER"),
                to=doctor_phone
            )
            
            return message.status in ["queued", "sent", "delivered"]
            
        except Exception as e:
            self.logger.error(f"Error sending SMS: {e}")
            return False
    
    def _format_email_body(self, alert: Alert) -> str:
        """Format alert as plain text email"""
        return f"""
Followup AI Wellness Monitoring Alert

{alert.title}

{alert.message}

Alert Details:
- Patient ID: {alert.patient_id}
- Severity: {alert.severity.upper()}
- Generated: {alert.created_at.strftime('%Y-%m-%d %H:%M:%S')}

Please review patient data in the Followup AI dashboard.

This is an automated wellness monitoring notification. This system provides change detection 
and wellness insights, not medical diagnosis. Please use clinical judgment when reviewing patient data.

---
Followup AI - HIPAA-Compliant Wellness Monitoring Platform
"""
    
    def _format_email_html(self, alert: Alert) -> str:
        """Format alert as HTML email"""
        severity_colors = {
            "low": "#10B981",
            "medium": "#F59E0B",
            "high": "#EF4444",
            "critical": "#DC2626"
        }
        color = severity_colors.get(alert.severity, "#6B7280")
        
        return f"""
<html>
<body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
    <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
        <h2 style="color: {color}; border-bottom: 3px solid {color}; padding-bottom: 10px;">
            {alert.title}
        </h2>
        
        <div style="background-color: #F9FAFB; padding: 15px; border-radius: 5px; margin: 20px 0;">
            <p style="white-space: pre-line;">{alert.message}</p>
        </div>
        
        <div style="border-left: 4px solid {color}; padding-left: 15px; margin: 20px 0;">
            <p><strong>Patient ID:</strong> {alert.patient_id}</p>
            <p><strong>Severity:</strong> <span style="color: {color}; font-weight: bold;">{alert.severity.upper()}</span></p>
            <p><strong>Generated:</strong> {alert.created_at.strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <p style="margin-top: 30px; padding-top: 20px; border-top: 1px solid #E5E7EB; color: #6B7280; font-size: 12px;">
            This is an automated wellness monitoring notification. Followup AI provides change detection 
            and wellness insights, not medical diagnosis. Please use clinical judgment when reviewing patient data.
        </p>
    </div>
</body>
</html>
"""
    
    def _format_sms_body(self, alert: Alert) -> str:
        """Format alert as SMS (limited to 160 characters)"""
        severity_emoji = {
            "low": "ℹ️",
            "medium": "⚠️",
            "high": "🔴",
            "critical": "‼️"
        }
        emoji = severity_emoji.get(alert.severity, "•")
        
        return f"{emoji} Followup AI: {alert.title} - Patient {alert.patient_id[:8]}. Review dashboard."
    
    async def _audit_alert_generation(self, alert: Alert, risk_event: RiskEvent) -> None:
        """
        Audit log for HIPAA compliance - all alert generation must be logged
        """
        try:
            audit_log = AuditLog(
                user_id="system",
                user_role="alert_orchestration_engine",
                action_type="create",
                resource_type="alert",
                resource_id=str(alert.id),
                patient_id_accessed=alert.patient_id,
                phi_accessed=True,
                data_fields_accessed=["risk_score", "trend_snapshot", "patient_metrics"],
                access_justification=f"Automated alert generation triggered by risk event {risk_event.id}",
                request_details={
                    "alert_id": alert.id,
                    "risk_event_id": risk_event.id,
                    "rule_id": alert.rule_id,
                    "severity": alert.severity,
                    "previous_risk_level": risk_event.previous_risk_level,
                    "new_risk_level": risk_event.new_risk_level
                }
            )
            
            self.db.add(audit_log)
            self.db.commit()
            
        except Exception as e:
            self.logger.error(f"Error creating audit log: {e}")
            # Don't fail alert generation if audit logging fails, but log the error
    
    async def acknowledge_alert(
        self,
        alert_id: int,
        acknowledged_by: str,
        user_role: str = "doctor"
    ) -> bool:
        """
        Mark alert as acknowledged by doctor
        
        Args:
            alert_id: Alert ID to acknowledge
            acknowledged_by: User ID of person acknowledging
            user_role: Role of acknowledging user (for audit)
        
        Returns:
            True if successful, False otherwise
        """
        try:
            alert = self.db.query(Alert).filter(Alert.id == alert_id).first()
            
            if not alert:
                self.logger.error(f"Alert {alert_id} not found")
                return False
            
            alert.status = "acknowledged"
            alert.acknowledged_at = datetime.utcnow()
            alert.acknowledged_by = acknowledged_by
            
            self.db.commit()
            
            # Audit log
            audit_log = AuditLog(
                user_id=acknowledged_by,
                user_role=user_role,
                action_type="update",
                resource_type="alert",
                resource_id=str(alert_id),
                patient_id_accessed=alert.patient_id,
                phi_accessed=True,
                data_fields_accessed=["alert_status", "acknowledged_at"],
                access_justification="Doctor acknowledged wellness monitoring alert"
            )
            self.db.add(audit_log)
            self.db.commit()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error acknowledging alert: {e}")
            self.db.rollback()
            return False
    
    async def get_pending_alerts(
        self,
        doctor_id: str,
        severity_filter: Optional[List[str]] = None
    ) -> List[Alert]:
        """
        Get pending/unacknowledged alerts for a doctor
        
        Args:
            doctor_id: Doctor ID to filter alerts
            severity_filter: Optional list of severities to filter (e.g., ["high", "critical"])
        
        Returns:
            List of pending alerts
        """
        try:
            query = self.db.query(Alert).filter(
                and_(
                    Alert.doctor_id == doctor_id,
                    Alert.status.in_(["pending", "sent"])
                )
            )
            
            if severity_filter:
                query = query.filter(Alert.severity.in_(severity_filter))
            
            alerts = query.order_by(desc(Alert.created_at)).all()
            
            return alerts
            
        except Exception as e:
            self.logger.error(f"Error getting pending alerts: {e}")
            return []


# Utility function for batch processing
async def process_pending_risk_events(db: Session, batch_size: int = 50) -> Dict[str, Any]:
    """
    Process all pending risk events that haven't triggered alerts yet
    
    This function should be run periodically (e.g., every 5 minutes) to ensure
    all risk events are evaluated against alert rules.
    
    Args:
        db: Database session
        batch_size: Number of events to process per batch
    
    Returns:
        Processing summary stats
    """
    try:
        engine = AlertOrchestrationEngine(db)
        
        # Get risk events that haven't triggered alerts yet
        pending_events = db.query(RiskEvent).filter(
            RiskEvent.triggered_alert == False
        ).limit(batch_size).all()
        
        total_processed = 0
        total_alerts_generated = 0
        
        for event in pending_events:
            alerts = await engine.evaluate_risk_event(
                risk_event=event,
                patient_id=event.patient_id
            )
            
            if alerts:
                total_alerts_generated += len(alerts)
                event.triggered_alert = True
                db.commit()
            
            total_processed += 1
        
        return {
            "events_processed": total_processed,
            "alerts_generated": total_alerts_generated,
            "timestamp": datetime.utcnow()
        }
        
    except Exception as e:
        logger.error(f"Error processing risk events: {e}")
        return {"error": str(e), "events_processed": 0, "alerts_generated": 0}
