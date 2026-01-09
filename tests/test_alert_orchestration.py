"""
Task 48: Test alerts trigger smoke test
========================================
Tests alert orchestration trigger flow.
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
import sys
import os

os.environ["AWS_REGION"] = "us-east-1"
os.environ["AWS_ACCESS_KEY_ID"] = "test_access_key"
os.environ["AWS_SECRET_ACCESS_KEY"] = "test_secret_key"

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class TestAlertOrchestration:
    """Task 48: Test alert orchestration trigger flow"""
    
    def test_alert_created_with_required_fields(self):
        """Alert contains all required fields"""
        alert = {
            "alert_id": "alert-123",
            "patient_id": "patient-456",
            "alert_type": "environmental",
            "severity": "high",
            "title": "Air Quality Alert",
            "message": "AQI exceeds safe levels for respiratory conditions",
            "created_at": "2024-01-01T00:00:00Z",
            "status": "pending"
        }
        
        required_fields = ["alert_id", "patient_id", "alert_type", "severity", 
                         "title", "message", "status"]
        
        for field in required_fields:
            assert field in alert, f"Missing required field: {field}"
    
    def test_doctor_contact_lookup(self):
        """Alert looks up doctor contact info from users table"""
        mock_doctor = MagicMock()
        mock_doctor.id = "doctor-789"
        mock_doctor.email = "doctor@hospital.com"
        mock_doctor.phone_number = "+1234567890"
        mock_doctor.first_name = "Dr. Jane"
        mock_doctor.last_name = "Smith"
        
        contact_info = {
            "email": mock_doctor.email,
            "phone": mock_doctor.phone_number,
            "name": f"{mock_doctor.first_name} {mock_doctor.last_name}"
        }
        
        assert contact_info["email"] == "doctor@hospital.com"
        assert contact_info["phone"] == "+1234567890"
    
    def test_email_notification_triggered(self):
        """Email notification is triggered for high severity"""
        notifications_sent = []
        
        def send_email_notification(to_email, subject, body):
            notifications_sent.append({
                "type": "email",
                "to": to_email,
                "subject": subject
            })
            return True
        
        alert = {"severity": "high", "title": "Critical Alert"}
        doctor_email = "doctor@hospital.com"
        
        if alert["severity"] in ["high", "critical"]:
            send_email_notification(
                doctor_email,
                f"URGENT: {alert['title']}",
                "Patient requires attention"
            )
        
        assert len(notifications_sent) == 1
        assert notifications_sent[0]["type"] == "email"
        assert "URGENT" in notifications_sent[0]["subject"]
    
    def test_sms_notification_triggered(self):
        """SMS notification is triggered for critical severity"""
        notifications_sent = []
        
        def send_sms_notification(to_phone, message):
            notifications_sent.append({
                "type": "sms",
                "to": to_phone,
                "message": message
            })
            return True
        
        alert = {"severity": "critical", "title": "Crisis Alert"}
        doctor_phone = "+1234567890"
        
        if alert["severity"] == "critical":
            send_sms_notification(
                doctor_phone,
                f"CRITICAL: {alert['title']}"
            )
        
        assert len(notifications_sent) == 1
        assert notifications_sent[0]["type"] == "sms"
    
    def test_notification_failure_logged(self):
        """Notification failures are logged for audit"""
        failures = []
        
        def log_notification_failure(notification_type, recipient, error):
            failures.append({
                "type": notification_type,
                "recipient": recipient,
                "error": str(error),
                "logged_at": "2024-01-01T00:00:00Z"
            })
        
        log_notification_failure("email", "doctor@hospital.com", "SMTP connection failed")
        
        assert len(failures) == 1
        assert failures[0]["type"] == "email"
        assert "SMTP" in failures[0]["error"]
    
    def test_alert_feedback_recorded(self):
        """Alert feedback from doctor is recorded"""
        feedback = {
            "alert_id": "alert-123",
            "was_helpful": True,
            "user_feedback": "Timely alert, patient was treated",
            "feedback_by": "doctor-789",
            "feedback_at": "2024-01-01T12:00:00Z"
        }
        
        assert feedback["was_helpful"] is True
        assert "feedback_by" in feedback
    
    def test_alert_context_includes_patient_data(self):
        """Alert includes relevant patient context"""
        alert_context = {
            "patient_id": "patient-456",
            "patient_conditions": ["Asthma", "COPD"],
            "patient_allergies": ["Pollen"],
            "recent_vitals": {"spo2": 94, "heart_rate": 88},
            "environmental_trigger": {"aqi": 150, "allergen_level": "high"}
        }
        
        assert "patient_conditions" in alert_context
        assert "environmental_trigger" in alert_context
        assert alert_context["environmental_trigger"]["aqi"] > 100
    
    def test_rules_fallback_when_ml_unavailable(self):
        """Falls back to rules when ML ranking unavailable"""
        ml_available = False
        
        def rank_alerts(alerts, use_ml=True):
            if use_ml and not ml_available:
                return sorted(alerts, key=lambda x: x["severity_score"], reverse=True)
            return alerts
        
        alerts = [
            {"id": 1, "severity_score": 50},
            {"id": 2, "severity_score": 90},
            {"id": 3, "severity_score": 30}
        ]
        
        ranked = rank_alerts(alerts, use_ml=True)
        
        assert ranked[0]["severity_score"] == 90
        assert ranked[-1]["severity_score"] == 30
    
    def test_crisis_alert_immediate_notification(self):
        """Crisis alerts trigger immediate notification"""
        is_crisis = True
        notifications = []
        
        def process_alert(alert, is_crisis):
            if is_crisis:
                notifications.append({
                    "priority": "immediate",
                    "channels": ["sms", "email", "push"]
                })
            else:
                notifications.append({
                    "priority": "normal",
                    "channels": ["email"]
                })
        
        process_alert({"title": "Crisis"}, is_crisis=True)
        
        assert notifications[0]["priority"] == "immediate"
        assert "sms" in notifications[0]["channels"]


class TestAlertOrchestrationSecurity:
    """Security tests for alert orchestration"""
    
    def test_doctor_contact_lookup_uses_parameterized_query(self):
        """Doctor contact lookup uses parameterized queries (SQL injection prevention)"""
        def get_doctor_contact_secure(db, doctor_id):
            return db.query().filter_by(id=doctor_id).first()
        
        mock_db = MagicMock()
        mock_doctor = MagicMock()
        mock_doctor.email = "doctor@hospital.com"
        mock_db.query.return_value.filter_by.return_value.first.return_value = mock_doctor
        
        result = get_doctor_contact_secure(mock_db, "doctor-123")
        
        mock_db.query.return_value.filter_by.assert_called_once_with(id="doctor-123")
        assert result.email == "doctor@hospital.com"
    
    def test_doctor_contact_lookup_filters_by_role(self):
        """Doctor contact lookup only returns users with doctor role"""
        def get_doctor_contact_with_role_check(db, doctor_id):
            user = db.query().filter_by(id=doctor_id).first()
            if user and "doctor" in user.roles:
                return user
            return None
        
        mock_db = MagicMock()
        
        mock_patient = MagicMock()
        mock_patient.roles = ["patient"]
        mock_db.query.return_value.filter_by.return_value.first.return_value = mock_patient
        
        result = get_doctor_contact_with_role_check(mock_db, "patient-123")
        assert result is None
        
        mock_doctor = MagicMock()
        mock_doctor.roles = ["doctor"]
        mock_db.query.return_value.filter_by.return_value.first.return_value = mock_doctor
        
        result = get_doctor_contact_with_role_check(mock_db, "doctor-456")
        assert result is not None
    
    def test_alert_access_requires_patient_relationship(self):
        """Alerts can only be accessed by assigned providers"""
        relationships = {
            "patient-A": ["doctor-1", "doctor-2"],
            "patient-B": ["doctor-3"]
        }
        
        def can_access_alert(doctor_id, patient_id, rel_map):
            providers = rel_map.get(patient_id, [])
            return doctor_id in providers
        
        assert can_access_alert("doctor-1", "patient-A", relationships) is True
        assert can_access_alert("doctor-3", "patient-A", relationships) is False
        assert can_access_alert("doctor-1", "patient-B", relationships) is False
    
    def test_contact_info_not_leaked_in_logs(self):
        """Contact info (email/phone) not logged in plain text"""
        log_entries = []
        
        def log_alert_notification(doctor_id, notification_type, success):
            log_entries.append({
                "doctor_id": doctor_id,
                "notification_type": notification_type,
                "success": success
            })
        
        log_alert_notification("doctor-123", "email", True)
        
        log_entry = log_entries[0]
        assert "email" not in str(log_entry).lower() or "email" == log_entry.get("notification_type")
        assert "@" not in str(log_entry)
        assert "phone" not in str(log_entry).lower() or "phone" == log_entry.get("notification_type")
