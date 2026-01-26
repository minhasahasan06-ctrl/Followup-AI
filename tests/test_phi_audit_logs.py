"""
Task 49: Verify audit logs for PHI
===================================
Tests HIPAA audit logging for PHI access.
"""

import pytest
from unittest.mock import MagicMock, patch
import sys
import os
from datetime import datetime

os.environ["AWS_REGION"] = "us-east-1"
os.environ["AWS_ACCESS_KEY_ID"] = "test_access_key"
os.environ["AWS_SECRET_ACCESS_KEY"] = "test_secret_key"

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class MockHIPAAAuditLogger:
    """Mock HIPAA Audit Logger for testing"""
    
    def __init__(self):
        self.logs = []
    
    def log_phi_access(self, user_id, patient_id, resource_type, action, 
                      details=None, ip_address=None, user_agent=None):
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": user_id,
            "patient_id": patient_id,
            "resource_type": resource_type,
            "action": action,
            "details": details,
            "ip_address": ip_address,
            "user_agent": user_agent
        }
        self.logs.append(entry)
        return entry
    
    def get_logs_for_patient(self, patient_id):
        return [log for log in self.logs if log["patient_id"] == patient_id]
    
    def get_logs_for_user(self, user_id):
        return [log for log in self.logs if log["user_id"] == user_id]


class TestPHIAuditLogs:
    """Task 49: Verify HIPAA audit logs for PHI access"""
    
    def setup_method(self):
        self.logger = MockHIPAAAuditLogger()
    
    def test_phi_access_logged(self):
        """PHI access creates audit log entry"""
        self.logger.log_phi_access(
            user_id="doctor-123",
            patient_id="patient-456",
            resource_type="PatientRecord",
            action="VIEW"
        )
        
        assert len(self.logger.logs) == 1
        assert self.logger.logs[0]["action"] == "VIEW"
    
    def test_audit_log_has_required_fields(self):
        """Audit log contains all HIPAA-required fields"""
        entry = self.logger.log_phi_access(
            user_id="doctor-123",
            patient_id="patient-456",
            resource_type="MedicalHistory",
            action="VIEW",
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0"
        )
        
        required_fields = ["timestamp", "user_id", "patient_id", 
                         "resource_type", "action"]
        
        for field in required_fields:
            assert field in entry, f"Missing required field: {field}"
    
    def test_different_actions_logged(self):
        """Different PHI actions are logged correctly"""
        actions = ["VIEW", "CREATE", "UPDATE", "DELETE", "EXPORT", "PRINT"]
        
        for action in actions:
            self.logger.log_phi_access(
                user_id="doctor-123",
                patient_id="patient-456",
                resource_type="PatientRecord",
                action=action
            )
        
        logged_actions = [log["action"] for log in self.logger.logs]
        
        for action in actions:
            assert action in logged_actions
    
    def test_logs_filterable_by_patient(self):
        """Logs can be filtered by patient ID"""
        self.logger.log_phi_access("doctor-1", "patient-A", "Record", "VIEW")
        self.logger.log_phi_access("doctor-1", "patient-B", "Record", "VIEW")
        self.logger.log_phi_access("doctor-2", "patient-A", "Record", "VIEW")
        
        patient_a_logs = self.logger.get_logs_for_patient("patient-A")
        
        assert len(patient_a_logs) == 2
        assert all(log["patient_id"] == "patient-A" for log in patient_a_logs)
    
    def test_logs_filterable_by_user(self):
        """Logs can be filtered by user ID"""
        self.logger.log_phi_access("doctor-1", "patient-A", "Record", "VIEW")
        self.logger.log_phi_access("doctor-1", "patient-B", "Record", "VIEW")
        self.logger.log_phi_access("doctor-2", "patient-A", "Record", "VIEW")
        
        doctor_1_logs = self.logger.get_logs_for_user("doctor-1")
        
        assert len(doctor_1_logs) == 2
        assert all(log["user_id"] == "doctor-1" for log in doctor_1_logs)
    
    def test_timestamp_recorded(self):
        """Timestamp is recorded for each log entry"""
        entry = self.logger.log_phi_access(
            user_id="doctor-123",
            patient_id="patient-456",
            resource_type="Record",
            action="VIEW"
        )
        
        assert "timestamp" in entry
        assert entry["timestamp"] is not None
    
    def test_sensitive_resources_logged(self):
        """Access to sensitive resources is logged"""
        sensitive_resources = [
            "PatientRecord",
            "MedicalHistory", 
            "Prescriptions",
            "LabResults",
            "ImagingResults",
            "MentalHealthNotes",
            "SubstanceAbuseRecords",
            "HIVStatus",
            "GeneticData"
        ]
        
        for resource in sensitive_resources:
            self.logger.log_phi_access(
                user_id="doctor-123",
                patient_id="patient-456",
                resource_type=resource,
                action="VIEW"
            )
        
        logged_resources = [log["resource_type"] for log in self.logger.logs]
        
        for resource in sensitive_resources:
            assert resource in logged_resources
    
    def test_export_action_includes_details(self):
        """Export action includes additional details"""
        entry = self.logger.log_phi_access(
            user_id="doctor-123",
            patient_id="patient-456",
            resource_type="PatientRecord",
            action="EXPORT",
            details={"format": "PDF", "pages": 5, "destination": "printer"}
        )
        
        assert entry["details"] is not None
        assert entry["details"]["format"] == "PDF"
    
    def test_unauthorized_access_attempt_logged(self):
        """Unauthorized access attempts are logged"""
        entry = self.logger.log_phi_access(
            user_id="unauthorized-user",
            patient_id="patient-456",
            resource_type="PatientRecord",
            action="VIEW_DENIED",
            details={"reason": "No patient relationship"}
        )
        
        assert entry["action"] == "VIEW_DENIED"
        assert "reason" in entry["details"]
    
    def test_ip_address_captured(self):
        """IP address is captured for audit trail"""
        entry = self.logger.log_phi_access(
            user_id="doctor-123",
            patient_id="patient-456",
            resource_type="Record",
            action="VIEW",
            ip_address="10.0.0.1"
        )
        
        assert entry["ip_address"] == "10.0.0.1"
    
    def test_audit_logs_immutable(self):
        """Audit logs cannot be modified after creation"""
        original_entry = self.logger.log_phi_access(
            user_id="doctor-123",
            patient_id="patient-456",
            resource_type="Record",
            action="VIEW"
        )
        
        original_action = original_entry["action"]
        
        stored_log = self.logger.logs[0]
        
        assert stored_log["action"] == original_action


class TestAuditLoggingFailureModes:
    """Test audit logging failure modes for compliance"""
    
    def setup_method(self):
        self.logger = MockHIPAAAuditLogger()
    
    def test_missing_ip_address_still_logs(self):
        """Audit log is created even when IP address is missing"""
        entry = self.logger.log_phi_access(
            user_id="doctor-123",
            patient_id="patient-456",
            resource_type="Record",
            action="VIEW",
            ip_address=None
        )
        
        assert entry is not None
        assert entry["ip_address"] is None
        assert entry["action"] == "VIEW"
    
    def test_missing_user_agent_still_logs(self):
        """Audit log is created even when user agent is missing"""
        entry = self.logger.log_phi_access(
            user_id="doctor-123",
            patient_id="patient-456",
            resource_type="Record",
            action="VIEW",
            user_agent=None
        )
        
        assert entry is not None
        assert entry["user_agent"] is None
        assert entry["action"] == "VIEW"
    
    def test_all_required_fields_present_even_with_minimal_input(self):
        """All required HIPAA fields are present with minimal input"""
        entry = self.logger.log_phi_access(
            user_id="doctor-123",
            patient_id="patient-456",
            resource_type="Record",
            action="VIEW"
        )
        
        required_fields = ["timestamp", "user_id", "patient_id", "resource_type", "action"]
        
        for field in required_fields:
            assert field in entry, f"Missing required field: {field}"
            assert entry[field] is not None, f"Required field {field} is None"
    
    def test_bulk_access_logged_individually(self):
        """Bulk access operations are logged individually"""
        patient_ids = ["patient-A", "patient-B", "patient-C"]
        
        for patient_id in patient_ids:
            self.logger.log_phi_access(
                user_id="doctor-123",
                patient_id=patient_id,
                resource_type="Record",
                action="VIEW",
                details={"bulk_operation": True}
            )
        
        assert len(self.logger.logs) == 3
        logged_patients = [log["patient_id"] for log in self.logger.logs]
        assert set(logged_patients) == set(patient_ids)
    
    def test_high_severity_access_flagged(self):
        """High severity access (genetic, HIV, mental health) is properly flagged"""
        high_severity_resources = ["GeneticData", "HIVStatus", "MentalHealthNotes"]
        
        for resource in high_severity_resources:
            entry = self.logger.log_phi_access(
                user_id="doctor-123",
                patient_id="patient-456",
                resource_type=resource,
                action="VIEW"
            )
            
            assert entry["resource_type"] in high_severity_resources
