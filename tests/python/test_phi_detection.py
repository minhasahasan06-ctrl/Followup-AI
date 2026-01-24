"""
Tests for PHI detection and protection services.
Validates PHI pattern detection, BAA/ZDR enforcement, and safe LLM client usage.
"""
import pytest
from unittest.mock import MagicMock, patch
import os


class TestPHIPatternDetection:
    """Test PHI pattern detection in text."""

    def test_detects_ssn_pattern(self):
        """Should detect SSN patterns."""
        from app.services.openai_client import detect_phi_patterns
        
        text_with_ssn = "Patient SSN is 123-45-6789"
        result = detect_phi_patterns(text_with_ssn)
        
        assert result['has_phi'] is True
        assert 'ssn' in result['detected_types']

    def test_detects_mrn_pattern(self):
        """Should detect MRN patterns."""
        from app.services.openai_client import detect_phi_patterns
        
        text_with_mrn = "MRN: 12345678"
        result = detect_phi_patterns(text_with_mrn)
        
        assert result['has_phi'] is True
        assert 'mrn' in result['detected_types']

    def test_detects_email_pattern(self):
        """Should detect email addresses."""
        from app.services.openai_client import detect_phi_patterns
        
        text_with_email = "Contact: patient@example.com"
        result = detect_phi_patterns(text_with_email)
        
        assert result['has_phi'] is True
        assert 'email' in result['detected_types']

    def test_detects_phone_pattern(self):
        """Should detect phone numbers."""
        from app.services.openai_client import detect_phi_patterns
        
        text_with_phone = "Call me at (555) 123-4567"
        result = detect_phi_patterns(text_with_phone)
        
        assert result['has_phi'] is True
        assert 'phone' in result['detected_types']

    def test_detects_credit_card_pattern(self):
        """Should detect credit card numbers."""
        from app.services.openai_client import detect_phi_patterns
        
        text_with_cc = "Card: 4111-1111-1111-1111"
        result = detect_phi_patterns(text_with_cc)
        
        assert result['has_phi'] is True
        assert 'credit_card' in result['detected_types']

    def test_clean_text_passes(self):
        """Text without PHI should pass."""
        from app.services.openai_client import detect_phi_patterns
        
        clean_text = "Patient reports mild headache for 3 days."
        result = detect_phi_patterns(clean_text)
        
        assert result['has_phi'] is False
        assert len(result['detected_types']) == 0


class TestBAAZDREnforcement:
    """Test BAA/ZDR runtime enforcement."""

    def test_production_requires_baa(self):
        """Production should require BAA flag."""
        from app.services.openai_client import check_baa_compliance
        
        with patch.dict(os.environ, {'ENVIRONMENT': 'production', 'OPENAI_BAA': ''}):
            result = check_baa_compliance()
            assert result['compliant'] is False
            assert 'BAA' in result['missing']

    def test_production_requires_zdr(self):
        """Production should require ZDR flag."""
        from app.services.openai_client import check_zdr_compliance
        
        with patch.dict(os.environ, {'ENVIRONMENT': 'production', 'OPENAI_ZDR': ''}):
            result = check_zdr_compliance()
            assert result['compliant'] is False
            assert 'ZDR' in result['missing']

    def test_development_allows_without_flags(self):
        """Development should allow without BAA/ZDR."""
        from app.services.openai_client import check_baa_compliance
        
        with patch.dict(os.environ, {'ENVIRONMENT': 'development', 'OPENAI_BAA': '', 'OPENAI_ZDR': ''}):
            result = check_baa_compliance()
            assert result['compliant'] is True


class TestLLMSafeClient:
    """Test LLM Safe Client for HIPAA compliance."""

    def test_client_blocks_phi_in_prompts(self):
        """Safe client should block PHI in prompts."""
        from app.services.llm_safe_client import LLMSafeClient
        
        client = LLMSafeClient()
        
        with pytest.raises(ValueError) as exc_info:
            client.safe_completion(
                prompt="Patient SSN 123-45-6789 has headache"
            )
        
        assert "PHI detected" in str(exc_info.value)

    def test_client_logs_all_calls(self):
        """Safe client should log all API calls for audit."""
        from app.services.llm_safe_client import LLMSafeClient
        
        with patch('app.services.llm_safe_client.audit_log') as mock_log:
            client = LLMSafeClient()
            client.safe_completion(prompt="What is hypertension?")
            
            mock_log.assert_called()

    def test_client_sanitizes_responses(self):
        """Safe client should sanitize responses containing PHI."""
        from app.services.llm_safe_client import LLMSafeClient
        
        client = LLMSafeClient()
        
        with patch.object(client, '_call_openai') as mock_call:
            mock_call.return_value = "Call patient at 555-123-4567"
            
            response = client.safe_completion(
                prompt="Get patient callback info",
                sanitize_response=True
            )
            
            assert "555-123-4567" not in response


class TestPHIRedaction:
    """Test PHI redaction for research/logging."""

    def test_redacts_all_phi_types(self):
        """Should redact all PHI types from text."""
        from app.services.openai_client import redact_phi
        
        text = """
        Patient John Smith (SSN: 123-45-6789)
        Email: john@example.com
        Phone: (555) 123-4567
        MRN: 12345678
        """
        
        redacted = redact_phi(text)
        
        assert "123-45-6789" not in redacted
        assert "john@example.com" not in redacted
        assert "(555) 123-4567" not in redacted
        assert "[REDACTED" in redacted

    def test_preserves_medical_content(self):
        """Should preserve medical content while redacting PHI."""
        from app.services.openai_client import redact_phi
        
        text = "Patient john@test.com reports BP 120/80, HR 72, temp 98.6F"
        redacted = redact_phi(text)
        
        assert "john@test.com" not in redacted
        assert "120/80" in redacted
        assert "HR 72" in redacted
        assert "98.6F" in redacted


class TestEmbeddingStandardization:
    """Test embedding model standardization."""

    def test_uses_standard_model(self):
        """Should use standardized embedding model."""
        from app.services.openai_client import get_embeddings
        
        with patch('openai.OpenAI') as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client
            mock_client.embeddings.create.return_value = MagicMock(
                data=[MagicMock(embedding=[0.1] * 1536)]
            )
            
            get_embeddings("test text")
            
            call_args = mock_client.embeddings.create.call_args
            assert call_args.kwargs['model'] == 'text-embedding-3-small'

    def test_returns_correct_dimensions(self):
        """Should return 1536-dimensional vectors."""
        from app.services.openai_client import get_embeddings
        
        with patch('openai.OpenAI') as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client
            mock_client.embeddings.create.return_value = MagicMock(
                data=[MagicMock(embedding=[0.1] * 1536)]
            )
            
            result = get_embeddings("test text")
            
            assert len(result) == 1536
