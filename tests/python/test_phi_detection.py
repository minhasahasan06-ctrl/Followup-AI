"""
PHI Detection Service Tests
Tests for PHIDetectionService using actual production APIs.
"""

import pytest
import os
from unittest.mock import MagicMock, patch, AsyncMock
from dataclasses import dataclass

from app.services.phi_detection_service import (
    PHIDetectionService,
    PHICategory,
    MedicalEntityCategory,
    PHIEntity,
    MedicalEntity,
    PHIDetectionResult,
    MedicalNLPResult
)


class TestPHICategory:
    """Test PHICategory enum values."""

    def test_phi_category_values(self):
        """Verify all PHI categories are defined."""
        required_categories = [
            "NAME", "DATE", "PHONE", "EMAIL", "ADDRESS",
            "SSN", "MRN", "AGE", "ID", "URL", "IP_ADDRESS",
            "DEVICE_ID", "BIOMETRIC", "PHOTO", "OTHER"
        ]
        for cat in required_categories:
            assert hasattr(PHICategory, cat), f"PHICategory.{cat} not found"

    def test_ssn_category(self):
        """SSN is a critical PHI category."""
        assert PHICategory.SSN.value == "SSN"

    def test_mrn_category(self):
        """MRN (Medical Record Number) is a critical PHI category."""
        assert PHICategory.MRN.value == "MRN"


class TestMedicalEntityCategory:
    """Test MedicalEntityCategory enum values."""

    def test_medical_entity_categories(self):
        """Verify medical entity categories are defined."""
        required = ["MEDICATION", "MEDICAL_CONDITION", "PROCEDURE", "ANATOMY"]
        for cat in required:
            assert hasattr(MedicalEntityCategory, cat)

    def test_medication_category(self):
        """MEDICATION category for drug detection."""
        assert MedicalEntityCategory.MEDICATION.value == "MEDICATION"


class TestPHIEntity:
    """Test PHIEntity data class."""

    def test_phi_entity_creation(self):
        """Test creating a PHI entity."""
        entity = PHIEntity(
            text="John Smith",
            category=PHICategory.NAME,
            start_offset=0,
            end_offset=10,
            confidence=0.95,
            placeholder="[PATIENT_NAME]"
        )
        
        assert entity.text == "John Smith"
        assert entity.category == PHICategory.NAME
        assert entity.confidence == 0.95
        assert entity.placeholder == "[PATIENT_NAME]"

    def test_ssn_entity(self):
        """Test creating an SSN PHI entity."""
        entity = PHIEntity(
            text="123-45-6789",
            category=PHICategory.SSN,
            start_offset=0,
            end_offset=11,
            confidence=0.99,
            placeholder="[SSN_REDACTED]"
        )
        
        assert entity.category == PHICategory.SSN
        assert "[SSN_REDACTED]" in entity.placeholder


class TestPHIDetectionResult:
    """Test PHIDetectionResult data class."""

    def test_detection_result_with_phi(self):
        """Test detection result when PHI is found."""
        entity = PHIEntity(
            text="john@example.com",
            category=PHICategory.EMAIL,
            start_offset=0,
            end_offset=16,
            confidence=0.98,
            placeholder="[EMAIL_REDACTED]"
        )
        
        result = PHIDetectionResult(
            original_text="Contact john@example.com for info",
            redacted_text="Contact [EMAIL_REDACTED] for info",
            phi_detected=True,
            phi_entities=[entity],
            redaction_count=1,
            processing_time_ms=50.0
        )
        
        assert result.phi_detected is True
        assert result.redaction_count == 1
        assert len(result.phi_entities) == 1

    def test_detection_result_no_phi(self):
        """Test detection result when no PHI is found."""
        result = PHIDetectionResult(
            original_text="The patient has a fever.",
            redacted_text="The patient has a fever.",
            phi_detected=False,
            phi_entities=[],
            redaction_count=0,
            processing_time_ms=30.0
        )
        
        assert result.phi_detected is False
        assert result.redaction_count == 0


class TestPHIDetectionServiceInit:
    """Test PHIDetectionService initialization."""

    @patch.dict(os.environ, {
        "OPENAI_API_KEY": "test-key",
        "OPENAI_BAA_SIGNED": "true",
        "OPENAI_ZDR_ENABLED": "true"
    })
    def test_service_init_with_compliance(self):
        """Test service initialization with HIPAA compliance settings."""
        with patch('app.services.phi_detection_service.OpenAI'), \
             patch('app.services.phi_detection_service.AsyncOpenAI'):
            service = PHIDetectionService()
            assert service is not None

    @patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=True)
    def test_service_init_no_api_key_raises(self):
        """Test service initialization fails without API key."""
        with pytest.raises(ValueError) as exc_info:
            PHIDetectionService()
        
        assert "OPENAI_API_KEY" in str(exc_info.value)


class TestPHIDetectionPatterns:
    """Test PHI detection pattern matching."""

    @patch.dict(os.environ, {
        "OPENAI_API_KEY": "test-key",
        "OPENAI_BAA_SIGNED": "true",
        "OPENAI_ZDR_ENABLED": "true"
    })
    def test_email_pattern_detection(self):
        """Test email pattern detection via regex fallback."""
        with patch('app.services.phi_detection_service.OpenAI'), \
             patch('app.services.phi_detection_service.AsyncOpenAI'):
            service = PHIDetectionService()
            
            entities = service._regex_fallback_detection(
                "Contact john.doe@hospital.com for appointments"
            )
            
            email_entities = [e for e in entities if e.category == PHICategory.EMAIL]
            assert len(email_entities) >= 1
            assert "john.doe@hospital.com" in email_entities[0].text

    @patch.dict(os.environ, {
        "OPENAI_API_KEY": "test-key",
        "OPENAI_BAA_SIGNED": "true",
        "OPENAI_ZDR_ENABLED": "true"
    })
    def test_ssn_pattern_detection(self):
        """Test SSN pattern detection via regex fallback."""
        with patch('app.services.phi_detection_service.OpenAI'), \
             patch('app.services.phi_detection_service.AsyncOpenAI'):
            service = PHIDetectionService()
            
            entities = service._regex_fallback_detection(
                "Patient SSN: 123-45-6789"
            )
            
            ssn_entities = [e for e in entities if e.category == PHICategory.SSN]
            assert len(ssn_entities) >= 1
            assert "123-45-6789" in ssn_entities[0].text

    @patch.dict(os.environ, {
        "OPENAI_API_KEY": "test-key",
        "OPENAI_BAA_SIGNED": "true",
        "OPENAI_ZDR_ENABLED": "true"
    })
    def test_phone_pattern_detection(self):
        """Test phone number pattern detection via regex fallback."""
        with patch('app.services.phi_detection_service.OpenAI'), \
             patch('app.services.phi_detection_service.AsyncOpenAI'):
            service = PHIDetectionService()
            
            entities = service._regex_fallback_detection(
                "Call us at 555-123-4567"
            )
            
            phone_entities = [e for e in entities if e.category == PHICategory.PHONE]
            assert len(phone_entities) >= 1

    @patch.dict(os.environ, {
        "OPENAI_API_KEY": "test-key",
        "OPENAI_BAA_SIGNED": "true",
        "OPENAI_ZDR_ENABLED": "true"
    })
    def test_mrn_pattern_detection(self):
        """Test MRN pattern detection via regex fallback."""
        with patch('app.services.phi_detection_service.OpenAI'), \
             patch('app.services.phi_detection_service.AsyncOpenAI'):
            service = PHIDetectionService()
            
            entities = service._regex_fallback_detection(
                "MRN: ABC123456789"
            )
            
            mrn_entities = [e for e in entities if e.category == PHICategory.MRN]
            assert len(mrn_entities) >= 1


class TestPHIDetectionServiceDetectPHI:
    """Test detect_phi method."""

    @patch.dict(os.environ, {
        "OPENAI_API_KEY": "test-key",
        "OPENAI_BAA_SIGNED": "true",
        "OPENAI_ZDR_ENABLED": "true"
    })
    def test_detect_phi_empty_text(self):
        """Test detecting PHI in empty text returns empty result."""
        with patch('app.services.phi_detection_service.OpenAI'), \
             patch('app.services.phi_detection_service.AsyncOpenAI'):
            service = PHIDetectionService()
            
            result = service.detect_phi("")
            
            assert result.phi_detected is False
            assert result.redaction_count == 0

    @patch.dict(os.environ, {
        "OPENAI_API_KEY": "test-key",
        "OPENAI_BAA_SIGNED": "true",
        "OPENAI_ZDR_ENABLED": "true"
    })
    def test_detect_phi_returns_result(self):
        """Test detect_phi returns PHIDetectionResult."""
        with patch('app.services.phi_detection_service.OpenAI') as mock_openai, \
             patch('app.services.phi_detection_service.AsyncOpenAI'):
            
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = '{"phi_entities": [], "phi_detected": false}'
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.return_value = mock_client
            
            service = PHIDetectionService()
            result = service.detect_phi("Patient has a headache")
            
            assert isinstance(result, PHIDetectionResult)


class TestBAAEnforcement:
    """Test BAA (Business Associate Agreement) enforcement."""

    @patch.dict(os.environ, {
        "OPENAI_API_KEY": "test-key",
        "OPENAI_BAA_SIGNED": "false",
        "OPENAI_ZDR_ENABLED": "true"
    })
    def test_baa_warning_when_not_signed(self):
        """Test that BAA warning is logged when not signed."""
        with patch('app.services.phi_detection_service.OpenAI'), \
             patch('app.services.phi_detection_service.AsyncOpenAI'), \
             patch('app.services.phi_detection_service.logger') as mock_logger:
            
            service = PHIDetectionService()
            
            mock_logger.warning.assert_called()

    @patch.dict(os.environ, {
        "OPENAI_API_KEY": "test-key",
        "OPENAI_BAA_SIGNED": "true",
        "OPENAI_ZDR_ENABLED": "true",
        "OPENAI_ENTERPRISE": "true"
    })
    def test_baa_no_warning_when_compliant(self):
        """Test no critical warnings when fully compliant."""
        with patch('app.services.phi_detection_service.OpenAI'), \
             patch('app.services.phi_detection_service.AsyncOpenAI'), \
             patch('app.services.phi_detection_service.logger') as mock_logger:
            
            service = PHIDetectionService()
            
            for call in mock_logger.warning.call_args_list:
                assert "CRITICAL" not in str(call)


class TestZDREnforcement:
    """Test ZDR (Zero Data Retention) enforcement."""

    @patch.dict(os.environ, {
        "OPENAI_API_KEY": "test-key",
        "OPENAI_BAA_SIGNED": "true",
        "OPENAI_ZDR_ENABLED": "false"
    })
    def test_zdr_warning_when_disabled(self):
        """Test that ZDR warning is logged when disabled."""
        with patch('app.services.phi_detection_service.OpenAI'), \
             patch('app.services.phi_detection_service.AsyncOpenAI'), \
             patch('app.services.phi_detection_service.logger') as mock_logger:
            
            service = PHIDetectionService()
            
            mock_logger.warning.assert_called()


class TestLLMSafeClient:
    """Test LLM Safe Client wrapper for PHI protection."""

    def test_openai_client_exists(self):
        """Test that openai_client module exists."""
        from app.services import openai_client
        assert openai_client is not None

    def test_openai_client_class_exists(self):
        """Test OpenAIClient class is available."""
        from app.services.openai_client import OpenAIClient
        assert OpenAIClient is not None


class TestMedicalNLPResult:
    """Test MedicalNLPResult data class."""

    def test_medical_nlp_result_creation(self):
        """Test creating a medical NLP result."""
        medication = MedicalEntity(
            text="Lisinopril 10mg",
            category=MedicalEntityCategory.MEDICATION,
            confidence=0.95
        )
        
        result = MedicalNLPResult(
            text="Patient takes Lisinopril 10mg daily",
            entities=[medication],
            icd10_codes=[],
            rxnorm_concepts=[{"code": "314076", "description": "Lisinopril 10 MG"}],
            snomed_concepts=[],
            phi_detected=False,
            phi_entities=[]
        )
        
        assert len(result.entities) == 1
        assert result.entities[0].category == MedicalEntityCategory.MEDICATION
        assert len(result.rxnorm_concepts) == 1


class TestPHIPlaceholders:
    """Test PHI placeholder generation."""

    @patch.dict(os.environ, {
        "OPENAI_API_KEY": "test-key",
        "OPENAI_BAA_SIGNED": "true",
        "OPENAI_ZDR_ENABLED": "true"
    })
    def test_get_placeholder_for_categories(self):
        """Test placeholder generation for different PHI categories."""
        with patch('app.services.phi_detection_service.OpenAI'), \
             patch('app.services.phi_detection_service.AsyncOpenAI'):
            
            service = PHIDetectionService()
            
            assert service._get_placeholder(PHICategory.NAME) == "[PATIENT_NAME]"
            assert service._get_placeholder(PHICategory.SSN) == "[SSN_REDACTED]"
            assert service._get_placeholder(PHICategory.EMAIL) == "[EMAIL_REDACTED]"
            assert service._get_placeholder(PHICategory.PHONE) == "[PHONE_REDACTED]"
            assert service._get_placeholder(PHICategory.MRN) == "[MRN_REDACTED]"
