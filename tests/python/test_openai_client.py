"""
Unit tests for OpenAI client wrapper with PHI/BAA/ZDR enforcement.
"""

import os
import pytest
from unittest.mock import patch, MagicMock, AsyncMock

os.environ["ENV"] = "dev"
os.environ["OPENAI_API_KEY"] = "test-key"
os.environ["PHI_DETECTION_ENABLED"] = "true"
os.environ["PHI_BLOCK_ON_DETECT"] = "true"

from app.services.openai_client import (
    OpenAIClientWrapper,
    PHIDetectionError,
    OpenAIConfigError,
    DirectIdentifierPatterns,
    EMBEDDING_MODEL,
    EMBEDDING_VERSION,
)


class TestPHIDetection:
    """Tests for PHI detection patterns"""
    
    def test_ssn_detection(self):
        """Test SSN pattern detection"""
        text = "My SSN is 123-45-6789"
        matches = list(DirectIdentifierPatterns.SSN.finditer(text))
        assert len(matches) == 1
        assert matches[0].group() == "123-45-6789"
    
    def test_email_detection(self):
        """Test email pattern detection"""
        text = "Contact me at john.doe@example.com"
        matches = list(DirectIdentifierPatterns.EMAIL.finditer(text))
        assert len(matches) == 1
        assert matches[0].group() == "john.doe@example.com"
    
    def test_phone_detection(self):
        """Test phone number detection"""
        test_cases = [
            ("Call me at 555-123-4567", "555-123-4567"),
            ("Phone: (555) 123-4567", "(555) 123-4567"),
            ("Cell: +1 555-123-4567", "+1 555-123-4567"),
        ]
        for text, expected in test_cases:
            matches = list(DirectIdentifierPatterns.PHONE.finditer(text))
            assert len(matches) >= 1, f"Failed for: {text}"
    
    def test_mrn_detection(self):
        """Test MRN pattern detection"""
        test_cases = [
            "MRN: 123456",
            "MRN#123456789",
            "Medical Record Number: 987654",
        ]
        for text in test_cases:
            matches = list(DirectIdentifierPatterns.MRN.finditer(text))
            assert len(matches) >= 1, f"Failed for: {text}"
    
    def test_credit_card_detection(self):
        """Test credit card pattern detection"""
        text = "Card number: 4111-1111-1111-1111"
        matches = list(DirectIdentifierPatterns.CREDIT_CARD.finditer(text))
        assert len(matches) == 1
    
    def test_no_phi_in_clean_text(self):
        """Test that clean text doesn't trigger PHI detection"""
        clean_text = "The patient reports mild headache symptoms for the past 3 days."
        
        patterns = [
            DirectIdentifierPatterns.SSN,
            DirectIdentifierPatterns.MRN,
            DirectIdentifierPatterns.EMAIL,
            DirectIdentifierPatterns.CREDIT_CARD,
        ]
        
        for pattern in patterns:
            matches = list(pattern.finditer(clean_text))
            assert len(matches) == 0, f"False positive for pattern"


class TestOpenAIClientConfiguration:
    """Tests for OpenAI client configuration"""
    
    @patch.dict(os.environ, {"ENV": "dev", "OPENAI_API_KEY": "test-key"})
    def test_dev_environment_allows_no_baa_zdr(self):
        """Test that dev environment doesn't require BAA/ZDR"""
        with patch.dict(os.environ, {"OPENAI_BAA": "false", "OPENAI_ZDR": "false"}):
            client = OpenAIClientWrapper()
            assert client is not None
    
    def test_prod_environment_requires_baa_zdr(self):
        """Test that prod environment blocks without BAA/ZDR"""
        import importlib
        import app.services.openai_client as client_module
        
        with patch.dict(os.environ, {
            "ENV": "prod",
            "OPENAI_API_KEY": "test-key",
            "OPENAI_BAA": "false",
            "OPENAI_ZDR": "false"
        }):
            importlib.reload(client_module)
            
            with pytest.raises(client_module.OpenAIConfigError) as exc_info:
                client_module.OpenAIClientWrapper()
            
            assert "BAA" in str(exc_info.value)
            assert "ZDR" in str(exc_info.value)
        
        with patch.dict(os.environ, {"ENV": "dev", "OPENAI_API_KEY": "test-key"}):
            importlib.reload(client_module)
    
    @patch.dict(os.environ, {
        "ENV": "prod",
        "OPENAI_API_KEY": "test-key",
        "OPENAI_BAA": "true",
        "OPENAI_ZDR": "true"
    })
    def test_prod_environment_allows_with_baa_zdr(self):
        """Test that prod environment works with BAA/ZDR enabled"""
        client = OpenAIClientWrapper()
        assert client is not None
    
    def test_missing_api_key_raises_error(self):
        """Test that missing API key raises error"""
        import importlib
        import app.services.openai_client as client_module
        
        with patch.dict(os.environ, {"OPENAI_API_KEY": "", "ENV": "dev"}):
            importlib.reload(client_module)
            
            with pytest.raises(client_module.OpenAIConfigError):
                client_module.OpenAIClientWrapper()
        
        with patch.dict(os.environ, {"ENV": "dev", "OPENAI_API_KEY": "test-key"}):
            importlib.reload(client_module)


class TestPHIBlocking:
    """Tests for PHI blocking behavior"""
    
    def test_phi_detected_raises_error(self):
        """Test that PHI triggers blocking when enabled"""
        import importlib
        import app.services.openai_client as client_module
        
        with patch.dict(os.environ, {
            "ENV": "dev",
            "OPENAI_API_KEY": "test-key",
            "PHI_DETECTION_ENABLED": "true",
            "PHI_BLOCK_ON_DETECT": "true"
        }):
            importlib.reload(client_module)
            
            client = client_module.OpenAIClientWrapper()
            text_with_phi = "Patient SSN is 123-45-6789"
            
            with pytest.raises(client_module.PHIDetectionError) as exc_info:
                client._check_and_handle_phi(text_with_phi, "test")
            
            assert "SSN" in str(exc_info.value)
        
        with patch.dict(os.environ, {"ENV": "dev", "OPENAI_API_KEY": "test-key"}):
            importlib.reload(client_module)
    
    @patch.dict(os.environ, {
        "ENV": "dev",
        "OPENAI_API_KEY": "test-key",
        "PHI_DETECTION_ENABLED": "true",
        "PHI_BLOCK_ON_DETECT": "false"
    })
    def test_phi_redacted_when_not_blocking(self):
        """Test that PHI is redacted when blocking is disabled"""
        client = OpenAIClientWrapper()
        
        text_with_phi = "Patient SSN is 123-45-6789"
        result = client._check_and_handle_phi(text_with_phi, "test")
        
        assert "123-45-6789" not in result
        assert "[SSN_REDACTED]" in result
    
    @patch.dict(os.environ, {
        "ENV": "dev",
        "OPENAI_API_KEY": "test-key",
        "PHI_DETECTION_ENABLED": "false",
        "PHI_BLOCK_ON_DETECT": "true"
    })
    def test_phi_detection_disabled(self):
        """Test that disabled PHI detection doesn't block"""
        client = OpenAIClientWrapper()
        
        text_with_phi = "Patient SSN is 123-45-6789"
        result = client._check_and_handle_phi(text_with_phi, "test")
        
        assert result == text_with_phi


class TestEmbeddingConfiguration:
    """Tests for embedding configuration"""
    
    def test_embedding_model_configured(self):
        """Test that embedding model is properly configured"""
        assert EMBEDDING_MODEL == "text-embedding-3-small"
        assert EMBEDDING_VERSION == "v1.0.0"
    
    @patch.dict(os.environ, {"ENV": "dev", "OPENAI_API_KEY": "test-key"})
    def test_embedding_metadata(self):
        """Test embedding metadata function"""
        from app.services.openai_client import get_embedding_metadata
        
        metadata = get_embedding_metadata()
        
        assert "embedding_model" in metadata
        assert "embedding_version" in metadata
        assert "embedding_dimension" in metadata
        assert metadata["embedding_dimension"] == "1536"


class TestAuditLogging:
    """Tests for audit logging"""
    
    @patch.dict(os.environ, {"ENV": "dev", "OPENAI_API_KEY": "test-key"})
    def test_input_hashing(self):
        """Test that inputs are hashed for audit logs"""
        client = OpenAIClientWrapper()
        
        hash1 = client._hash_input("test input")
        hash2 = client._hash_input("test input")
        hash3 = client._hash_input("different input")
        
        assert hash1 == hash2
        assert hash1 != hash3
        assert len(hash1) == 16


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
