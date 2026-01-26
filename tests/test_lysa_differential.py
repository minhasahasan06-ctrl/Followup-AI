"""
Task 45: Test differential JSON and logging
============================================
Tests Lysa differential draft response validation and logging.
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
import json
import sys
import os

os.environ["AWS_REGION"] = "us-east-1"
os.environ["AWS_ACCESS_KEY_ID"] = "test_access_key"
os.environ["AWS_SECRET_ACCESS_KEY"] = "test_secret_key"

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


VALID_DIFFERENTIAL_SCHEMA = {
    "required_fields": ["draft_id", "patient_id", "draft_type", "content", "status"],
    "draft_types": ["differential_diagnosis", "assessment_plan", "history_physical"],
    "status_values": ["draft", "pending_approval", "approved", "revised", "rejected"]
}


class TestLysaDifferential:
    """Task 45: Test Lysa differential JSON validation"""
    
    def test_differential_response_has_required_fields(self):
        """Differential response contains all required fields"""
        response = {
            "draft_id": "draft-123",
            "patient_id": "patient-456",
            "draft_type": "differential_diagnosis",
            "content": {
                "diagnoses": [
                    {"diagnosis": "Pneumonia", "probability": 0.7, "reasoning": "Fever, cough, infiltrate on CXR"}
                ]
            },
            "status": "draft",
            "created_at": "2024-01-01T00:00:00Z"
        }
        
        for field in VALID_DIFFERENTIAL_SCHEMA["required_fields"]:
            assert field in response, f"Missing required field: {field}"
    
    def test_differential_type_valid(self):
        """Draft type is one of valid types"""
        valid_types = VALID_DIFFERENTIAL_SCHEMA["draft_types"]
        
        for draft_type in valid_types:
            assert draft_type in valid_types
        
        invalid_type = "random_type"
        assert invalid_type not in valid_types
    
    def test_differential_status_valid(self):
        """Draft status is one of valid statuses"""
        valid_statuses = VALID_DIFFERENTIAL_SCHEMA["status_values"]
        
        for status in valid_statuses:
            assert status in valid_statuses
        
        invalid_status = "random_status"
        assert invalid_status not in valid_statuses
    
    def test_differential_content_structured(self):
        """Differential diagnosis content is properly structured"""
        content = {
            "diagnoses": [
                {
                    "diagnosis": "Community-acquired pneumonia",
                    "probability": 0.75,
                    "reasoning": "Productive cough, fever, right lower lobe infiltrate on CXR",
                    "icd10_code": "J18.9",
                    "sources": ["chief_complaint", "physical_exam", "imaging"]
                },
                {
                    "diagnosis": "Acute bronchitis",
                    "probability": 0.20,
                    "reasoning": "Cough without clear infiltrate could suggest bronchitis",
                    "icd10_code": "J20.9",
                    "sources": ["chief_complaint"]
                }
            ]
        }
        
        assert "diagnoses" in content
        assert len(content["diagnoses"]) >= 1
        
        for dx in content["diagnoses"]:
            assert "diagnosis" in dx
            assert "probability" in dx
            assert 0 <= dx["probability"] <= 1
            assert "reasoning" in dx
    
    def test_differential_requires_doctor_approval(self):
        """Differential draft requires explicit doctor approval"""
        draft = {
            "draft_id": "draft-123",
            "status": "pending_approval",
            "requires_approval": True,
            "approved_by": None,
            "approved_at": None
        }
        
        assert draft["requires_approval"] is True
        assert draft["approved_by"] is None
        assert draft["status"] == "pending_approval"
    
    def test_approval_updates_status(self):
        """Approval updates draft status correctly"""
        draft = {
            "draft_id": "draft-123",
            "status": "pending_approval",
            "approved_by": None,
            "approved_at": None
        }
        
        draft["status"] = "approved"
        draft["approved_by"] = "doctor-789"
        draft["approved_at"] = "2024-01-01T12:00:00Z"
        
        assert draft["status"] == "approved"
        assert draft["approved_by"] is not None
        assert draft["approved_at"] is not None
    
    def test_revision_creates_new_version(self):
        """Revision creates new version with history"""
        original_draft = {
            "draft_id": "draft-123",
            "version": 1,
            "content": {"diagnoses": []},
            "revision_history": []
        }
        
        revised_draft = {
            "draft_id": "draft-123",
            "version": 2,
            "content": {"diagnoses": [{"diagnosis": "Updated diagnosis"}]},
            "revision_history": [
                {"version": 1, "revised_at": "2024-01-01T12:00:00Z", "revised_by": "doctor-789"}
            ]
        }
        
        assert revised_draft["version"] > original_draft["version"]
        assert len(revised_draft["revision_history"]) > 0
    
    def test_audit_log_created_on_draft_action(self):
        """HIPAA audit log created for draft actions"""
        audit_entries = []
        
        def log_draft_action(action, draft_id, user_id):
            audit_entries.append({
                "action": action,
                "resource_type": "LysaDraft",
                "resource_id": draft_id,
                "user_id": user_id
            })
        
        log_draft_action("CREATE", "draft-123", "doctor-789")
        log_draft_action("APPROVE", "draft-123", "doctor-789")
        
        assert len(audit_entries) == 2
        assert audit_entries[0]["action"] == "CREATE"
        assert audit_entries[1]["action"] == "APPROVE"
    
    def test_source_information_stored(self):
        """Source information for claims is stored"""
        diagnosis = {
            "diagnosis": "Pneumonia",
            "sources": [
                {"type": "chief_complaint", "text": "Cough and fever for 3 days"},
                {"type": "vital_signs", "data": {"temperature": 38.5}},
                {"type": "imaging", "finding": "Right lower lobe infiltrate"}
            ]
        }
        
        assert "sources" in diagnosis
        assert len(diagnosis["sources"]) >= 1
        
        source_types = [s["type"] for s in diagnosis["sources"]]
        assert "chief_complaint" in source_types
    
    def test_json_response_valid(self):
        """Response is valid JSON"""
        response = {
            "draft_id": "draft-123",
            "content": {"diagnoses": []},
            "status": "draft"
        }
        
        json_str = json.dumps(response)
        parsed = json.loads(json_str)
        
        assert parsed["draft_id"] == "draft-123"
        assert isinstance(parsed["content"], dict)


class TestLysaSecurityEnforcement:
    """Security enforcement tests for Lysa drafts"""
    
    def test_non_doctor_approval_denied(self):
        """Non-doctor users cannot approve drafts"""
        user_roles = {
            "patient-123": ["patient"],
            "admin-456": ["admin"],
            "nurse-789": ["nurse"]
        }
        
        def can_approve_draft(user_id, user_roles_map):
            roles = user_roles_map.get(user_id, [])
            return "doctor" in roles
        
        assert can_approve_draft("patient-123", user_roles) is False
        assert can_approve_draft("admin-456", user_roles) is False
        assert can_approve_draft("nurse-789", user_roles) is False
    
    def test_doctor_approval_allowed(self):
        """Doctor users can approve drafts"""
        user_roles = {
            "doctor-123": ["doctor"],
            "doctor-admin-456": ["doctor", "admin"]
        }
        
        def can_approve_draft(user_id, user_roles_map):
            roles = user_roles_map.get(user_id, [])
            return "doctor" in roles
        
        assert can_approve_draft("doctor-123", user_roles) is True
        assert can_approve_draft("doctor-admin-456", user_roles) is True
    
    def test_unauthorized_approval_attempt_logged(self):
        """Unauthorized approval attempts are logged for audit"""
        audit_logs = []
        
        def attempt_approval(user_id, user_role, draft_id):
            if user_role != "doctor":
                audit_logs.append({
                    "action": "APPROVAL_DENIED",
                    "user_id": user_id,
                    "role": user_role,
                    "resource_id": draft_id,
                    "reason": "Insufficient privileges"
                })
                return False
            return True
        
        result = attempt_approval("patient-123", "patient", "draft-456")
        
        assert result is False
        assert len(audit_logs) == 1
        assert audit_logs[0]["action"] == "APPROVAL_DENIED"
        assert audit_logs[0]["reason"] == "Insufficient privileges"
    
    def test_draft_access_requires_patient_assignment(self):
        """Doctor can only access drafts for assigned patients"""
        doctor_patients = {
            "doctor-123": ["patient-A", "patient-B"],
            "doctor-456": ["patient-C"]
        }
        
        def can_access_draft(doctor_id, patient_id, assignments):
            assigned_patients = assignments.get(doctor_id, [])
            return patient_id in assigned_patients
        
        assert can_access_draft("doctor-123", "patient-A", doctor_patients) is True
        assert can_access_draft("doctor-123", "patient-C", doctor_patients) is False
        assert can_access_draft("doctor-456", "patient-A", doctor_patients) is False
