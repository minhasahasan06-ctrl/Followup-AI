"""
E.1: Unit test - privacy_firewall blocks PHI keys
=================================================
Tests that the privacy firewall correctly blocks all PHI patterns.
"""

import pytest
from app.services.privacy_firewall import (
    FORBIDDEN_KEYS,
    phi_scan,
    sanitize,
    k_anon_check,
    TinkerPurpose,
)


class TestForbiddenKeys:
    """E.1: Test that privacy_firewall blocks all PHI keys"""
    
    def test_forbidden_keys_exist(self):
        """All 12 forbidden keys are defined"""
        expected_keys = {
            "patient_id", "mrn", "name", "email", "phone", "address",
            "notes", "symptoms_text", "clinician_note", "timestamp", "date", "dob"
        }
        assert FORBIDDEN_KEYS == expected_keys
        assert len(FORBIDDEN_KEYS) == 12
    
    def test_sanitize_blocks_patient_id(self):
        """Sanitize rejects patient_id at any level"""
        with pytest.raises(ValueError, match="FORBIDDEN_KEYS"):
            sanitize(TinkerPurpose.PATIENT_QUESTIONS.value, {"patient_id": "12345"})
    
    def test_sanitize_blocks_mrn(self):
        """Sanitize rejects MRN"""
        with pytest.raises(ValueError, match="FORBIDDEN_KEYS"):
            sanitize(TinkerPurpose.PATIENT_QUESTIONS.value, {"mrn": "MRN123456"})
    
    def test_sanitize_blocks_name(self):
        """Sanitize rejects name field"""
        with pytest.raises(ValueError, match="FORBIDDEN_KEYS"):
            sanitize(TinkerPurpose.PATIENT_QUESTIONS.value, {"name": "John Doe"})
    
    def test_sanitize_blocks_email(self):
        """Sanitize rejects email field"""
        with pytest.raises(ValueError, match="FORBIDDEN_KEYS"):
            sanitize(TinkerPurpose.PATIENT_QUESTIONS.value, {"email": "test@example.com"})
    
    def test_sanitize_blocks_phone(self):
        """Sanitize rejects phone field"""
        with pytest.raises(ValueError, match="FORBIDDEN_KEYS"):
            sanitize(TinkerPurpose.PATIENT_QUESTIONS.value, {"phone": "555-1234"})
    
    def test_sanitize_blocks_address(self):
        """Sanitize rejects address field"""
        with pytest.raises(ValueError, match="FORBIDDEN_KEYS"):
            sanitize(TinkerPurpose.PATIENT_QUESTIONS.value, {"address": "123 Main St"})
    
    def test_sanitize_blocks_notes(self):
        """Sanitize rejects notes field"""
        with pytest.raises(ValueError, match="FORBIDDEN_KEYS"):
            sanitize(TinkerPurpose.PATIENT_QUESTIONS.value, {"notes": "Patient notes"})
    
    def test_sanitize_blocks_symptoms_text(self):
        """Sanitize rejects symptoms_text field"""
        with pytest.raises(ValueError, match="FORBIDDEN_KEYS"):
            sanitize(TinkerPurpose.PATIENT_QUESTIONS.value, {"symptoms_text": "Headache"})
    
    def test_sanitize_blocks_clinician_note(self):
        """Sanitize rejects clinician_note field"""
        with pytest.raises(ValueError, match="FORBIDDEN_KEYS"):
            sanitize(TinkerPurpose.PATIENT_QUESTIONS.value, {"clinician_note": "Follow up"})
    
    def test_sanitize_blocks_timestamp(self):
        """Sanitize rejects timestamp field"""
        with pytest.raises(ValueError, match="FORBIDDEN_KEYS"):
            sanitize(TinkerPurpose.PATIENT_QUESTIONS.value, {"timestamp": "2024-01-01T00:00:00"})
    
    def test_sanitize_blocks_date(self):
        """Sanitize rejects date field"""
        with pytest.raises(ValueError, match="FORBIDDEN_KEYS"):
            sanitize(TinkerPurpose.PATIENT_QUESTIONS.value, {"date": "2024-01-01"})
    
    def test_sanitize_blocks_dob(self):
        """Sanitize rejects dob field"""
        with pytest.raises(ValueError, match="FORBIDDEN_KEYS"):
            sanitize(TinkerPurpose.PATIENT_QUESTIONS.value, {"dob": "1990-01-01"})


class TestPHIScan:
    """E.1: Test PHI pattern detection"""
    
    def test_detects_email_pattern(self):
        """phi_scan detects email addresses"""
        assert phi_scan("contact me at john@example.com") is True
        assert phi_scan("email: test.user@hospital.org") is True
    
    def test_detects_phone_pattern(self):
        """phi_scan detects phone numbers"""
        assert phi_scan("call me at 555-123-4567") is True
        assert phi_scan("phone: (555) 123-4567") is True
        assert phi_scan("1-800-555-1234") is True
    
    def test_detects_ssn_pattern(self):
        """phi_scan detects SSN patterns"""
        assert phi_scan("SSN: 123-45-6789") is True
        assert phi_scan("social: 123 45 6789") is True
    
    def test_detects_date_patterns(self):
        """phi_scan detects date patterns"""
        assert phi_scan("born on 01/15/1990") is True
        assert phi_scan("date: 2024-03-15") is True
    
    def test_detects_address_pattern(self):
        """phi_scan detects address patterns"""
        assert phi_scan("lives at 123 Main Street") is True
        assert phi_scan("456 Oak Avenue") is True
    
    def test_detects_mrn_pattern(self):
        """phi_scan detects MRN patterns"""
        assert phi_scan("MRN: 123456789") is True
        assert phi_scan("MRN#12345678") is True
    
    def test_safe_text_passes(self):
        """phi_scan allows safe text"""
        assert phi_scan("Patient is stable") is False
        assert phi_scan("Low risk bucket") is False
        assert phi_scan("condition_codes: A00, B01") is False
    
    def test_empty_string_passes(self):
        """phi_scan handles empty strings"""
        assert phi_scan("") is False
        assert phi_scan(None) is False


class TestSanitizeAllowedKeys:
    """E.1: Test that only allowed keys pass through"""
    
    def test_patient_questions_allowed_keys(self):
        """Only allowed keys for patient_questions pass through"""
        valid_payload = {
            "age_bucket": "30-34",
            "condition_codes": ["A00", "B01"],
            "risk_bucket": "medium",
            "trend_flags": ["improving"],
            "engagement_bucket": "high",
            "missingness_bucket": "low",
        }
        result = sanitize(TinkerPurpose.PATIENT_QUESTIONS.value, valid_payload)
        assert set(result.keys()) == set(valid_payload.keys())
    
    def test_unknown_keys_dropped(self):
        """Unknown keys are silently dropped"""
        payload = {
            "age_bucket": "30-34",
            "unknown_key": "should be dropped",
            "another_unknown": 123,
        }
        result = sanitize(TinkerPurpose.PATIENT_QUESTIONS.value, payload)
        assert "unknown_key" not in result
        assert "another_unknown" not in result
        assert "age_bucket" in result
    
    def test_empty_payload_returns_empty(self):
        """Empty payload returns empty dict"""
        result = sanitize(TinkerPurpose.PATIENT_QUESTIONS.value, {})
        assert result == {}
    
    def test_unknown_purpose_returns_empty(self):
        """Unknown purpose returns empty dict"""
        result = sanitize("unknown_purpose", {"age_bucket": "30-34"})
        assert result == {}


class TestPHIInValues:
    """E.1: Test PHI detection in field values"""
    
    def test_phi_in_allowed_string_blocked(self):
        """PHI patterns in allowed string fields are blocked"""
        payload = {
            "age_bucket": "contact john@example.com for more"
        }
        with pytest.raises(ValueError, match="PHI detected"):
            sanitize(TinkerPurpose.PATIENT_QUESTIONS.value, payload)
    
    def test_phi_in_string_list_blocked(self):
        """PHI patterns in string list items are blocked"""
        payload = {
            "condition_codes": ["A00", "call 555-123-4567"]
        }
        with pytest.raises(ValueError, match="PHI detected"):
            sanitize(TinkerPurpose.PATIENT_QUESTIONS.value, payload)
