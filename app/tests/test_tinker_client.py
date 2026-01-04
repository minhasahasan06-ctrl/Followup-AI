"""
E.3: Unit test - tinker_client returns safe defaults on failure
===============================================================
Tests that tinker_client returns deterministic stub responses when:
- TINKER_ENABLED=false
- API call fails
"""

import pytest
from unittest.mock import patch, MagicMock
from app.services.tinker_client import (
    call_tinker,
    STUB_RESPONSES,
    _get_stub_response,
)
from app.services.privacy_firewall import TinkerPurpose


class TestStubResponses:
    """E.3: Test stub responses are returned when Tinker disabled"""
    
    def test_stub_responses_defined_for_all_purposes(self):
        """All TinkerPurpose values have stub responses defined"""
        for purpose in TinkerPurpose:
            assert purpose.value in STUB_RESPONSES, f"Missing stub for {purpose.value}"
    
    def test_stub_response_structure_patient_questions(self):
        """Patient questions stub has expected structure"""
        stub = STUB_RESPONSES[TinkerPurpose.PATIENT_QUESTIONS.value]
        assert "questions" in stub
        assert isinstance(stub["questions"], list)
        assert len(stub["questions"]) > 0
    
    def test_stub_response_structure_patient_templates(self):
        """Patient templates stub has expected structure"""
        stub = STUB_RESPONSES[TinkerPurpose.PATIENT_TEMPLATES.value]
        assert "templates" in stub
        assert isinstance(stub["templates"], list)
    
    def test_stub_response_structure_cohort_builder(self):
        """Cohort builder stub has expected structure"""
        stub = STUB_RESPONSES[TinkerPurpose.COHORT_BUILDER.value]
        assert "cohort_definition" in stub
        assert "sql_preview" in stub
    
    def test_stub_response_structure_job_planner(self):
        """Job planner stub has expected structure"""
        stub = STUB_RESPONSES[TinkerPurpose.JOB_PLANNER.value]
        assert "jobs" in stub
        assert isinstance(stub["jobs"], list)
    
    def test_stub_response_structure_model_card(self):
        """Model card stub has expected structure"""
        stub = STUB_RESPONSES[TinkerPurpose.MODEL_CARD.value]
        assert "model_card" in stub
        assert "name" in stub["model_card"]
        assert "metrics" in stub["model_card"]
    
    def test_stub_response_structure_drift_summary(self):
        """Drift summary stub has expected structure"""
        stub = STUB_RESPONSES[TinkerPurpose.DRIFT_SUMMARY.value]
        assert "drift_detected" in stub
        assert "drift_score" in stub
        assert "recommendations" in stub
    
    def test_get_stub_response_adds_timestamp(self):
        """_get_stub_response adds generated_at timestamp"""
        stub = _get_stub_response(TinkerPurpose.PATIENT_QUESTIONS.value)
        assert "generated_at" in stub
        assert stub.get("is_stub") is True
    
    def test_get_stub_response_unknown_purpose(self):
        """_get_stub_response returns basic stub for unknown purpose"""
        stub = _get_stub_response("unknown_purpose")
        assert "status" in stub
        assert stub["status"] == "stub"
        assert stub.get("is_stub") is True


class TestCallTinkerDisabled:
    """E.3: Test call_tinker returns stubs when TINKER_ENABLED=false"""
    
    @patch("app.services.tinker_client.settings")
    @patch("app.services.tinker_client.audit_log_sync")
    def test_returns_stub_when_disabled(self, mock_audit, mock_settings):
        """call_tinker returns stub response when TINKER_ENABLED=false"""
        mock_settings.TINKER_ENABLED = False
        mock_settings.TINKER_API_KEY = None
        mock_audit.return_value = "audit-123"
        
        payload = {"age_bucket": "30-34", "risk_bucket": "low"}
        response, success = call_tinker(
            TinkerPurpose.PATIENT_QUESTIONS.value,
            payload,
            actor_role="patient"
        )
        
        assert success is True
        assert response.get("is_stub") is True
        assert "questions" in response
    
    @patch("app.services.tinker_client.settings")
    @patch("app.services.tinker_client.audit_log_sync")
    def test_returns_stub_when_no_api_key(self, mock_audit, mock_settings):
        """call_tinker returns stub response when no API key"""
        mock_settings.TINKER_ENABLED = True
        mock_settings.TINKER_API_KEY = None
        mock_audit.return_value = "audit-123"
        
        payload = {"age_bucket": "30-34"}
        response, success = call_tinker(
            TinkerPurpose.PATIENT_QUESTIONS.value,
            payload,
            actor_role="patient"
        )
        
        assert success is True
        assert response.get("is_stub") is True
    
    @patch("app.services.tinker_client.settings")
    @patch("app.services.tinker_client.audit_log_sync")
    def test_stub_audited(self, mock_audit, mock_settings):
        """Stub responses are still audited"""
        mock_settings.TINKER_ENABLED = False
        mock_settings.TINKER_API_KEY = None
        mock_audit.return_value = "audit-123"
        
        payload = {"age_bucket": "30-34"}
        call_tinker(TinkerPurpose.PATIENT_QUESTIONS.value, payload, actor_role="patient")
        
        mock_audit.assert_called_once()
        call_args = mock_audit.call_args
        assert call_args.kwargs.get("model_used") == "stub"
        assert call_args.kwargs.get("success") is True


class TestCallTinkerFailure:
    """E.3: Test call_tinker handles API failures gracefully"""
    
    @patch("app.services.tinker_client.settings")
    @patch("app.services.tinker_client._call_tinker_live")
    @patch("app.services.tinker_client.audit_log_sync")
    def test_returns_error_on_api_failure(self, mock_audit, mock_live, mock_settings):
        """call_tinker returns error dict on API failure"""
        mock_settings.TINKER_ENABLED = True
        mock_settings.TINKER_API_KEY = "test-key"
        mock_live.side_effect = Exception("Connection failed")
        mock_audit.return_value = "audit-123"
        
        payload = {"age_bucket": "30-34"}
        response, success = call_tinker(
            TinkerPurpose.PATIENT_QUESTIONS.value,
            payload,
            actor_role="patient"
        )
        
        assert success is False
        assert "error" in response
        assert "Connection failed" in response["error"]
    
    @patch("app.services.tinker_client.settings")
    @patch("app.services.tinker_client.audit_log_sync")
    def test_sanitization_failure_returns_error(self, mock_audit, mock_settings):
        """call_tinker returns error when sanitization fails"""
        mock_settings.TINKER_ENABLED = True
        mock_settings.TINKER_API_KEY = "test-key"
        mock_audit.return_value = "audit-123"
        
        payload = {"patient_id": "12345"}
        response, success = call_tinker(
            TinkerPurpose.PATIENT_QUESTIONS.value,
            payload,
            actor_role="patient"
        )
        
        assert success is False
        assert "error" in response
    
    @patch("app.services.tinker_client.settings")
    @patch("app.services.tinker_client.audit_log_sync")
    def test_k_anon_failure_returns_error(self, mock_audit, mock_settings):
        """call_tinker returns error when k-anonymity check fails"""
        mock_settings.TINKER_ENABLED = True
        mock_settings.TINKER_API_KEY = "test-key"
        mock_audit.return_value = "audit-123"
        
        payload = {"age_bucket": "30-34"}
        response, success = call_tinker(
            TinkerPurpose.PATIENT_QUESTIONS.value,
            payload,
            actor_role="patient",
            cohort_count=5
        )
        
        assert success is False
        assert "error" in response
        assert "K-ANONYMITY" in response["error"]
    
    def test_invalid_purpose_returns_error(self):
        """call_tinker returns error for invalid purpose"""
        payload = {"age_bucket": "30-34"}
        response, success = call_tinker(
            "invalid_purpose",
            payload,
            actor_role="patient"
        )
        
        assert success is False
        assert "error" in response
        assert "Invalid purpose" in response["error"]


class TestCallTinkerDeterministic:
    """E.3: Test stub responses are deterministic"""
    
    @patch("app.services.tinker_client.settings")
    @patch("app.services.tinker_client.audit_log_sync")
    def test_stub_structure_is_deterministic(self, mock_audit, mock_settings):
        """Same purpose returns same stub structure"""
        mock_settings.TINKER_ENABLED = False
        mock_settings.TINKER_API_KEY = None
        mock_audit.return_value = "audit-123"
        
        payload = {"age_bucket": "30-34"}
        
        response1, _ = call_tinker(TinkerPurpose.PATIENT_QUESTIONS.value, payload, "patient")
        response2, _ = call_tinker(TinkerPurpose.PATIENT_QUESTIONS.value, payload, "patient")
        
        response1.pop("generated_at", None)
        response2.pop("generated_at", None)
        
        assert response1["questions"] == response2["questions"]
