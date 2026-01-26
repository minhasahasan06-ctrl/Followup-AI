"""
Unit tests for ML Governance Service.
"""

import os
import pytest
from unittest.mock import patch, MagicMock

os.environ["ENV"] = "dev"
os.environ["OPENAI_API_KEY"] = "test-key"

from app.services.ml_governance_service import (
    MLGovernanceService,
    GovernanceError,
    ClinicalValidationRequired,
    ResearchOnlyModel,
    HumanApprovalRequired,
    EmbeddingStandardizationError,
    get_governance_service,
)
from app.models.ml_governance_models import (
    ModelValidationStatus,
    ModelDeploymentEnvironment,
    ClinicalRiskLevel,
)


class TestModelDeploymentCheck:
    """Tests for model deployment governance checks"""
    
    def test_non_clinical_model_can_deploy_to_development(self):
        """Test that non-clinical models can deploy to development"""
        service = MLGovernanceService()
        
        result = service.check_model_deployment(
            model_id="test-model",
            target_environment="development",
            is_clinical=False,
            validation_status=None,
            allowed_environment="development",
        )
        
        assert result.can_deploy is True
        assert len(result.issues) == 0
    
    def test_clinical_model_requires_validation(self):
        """Test that clinical models require validation for production"""
        service = MLGovernanceService()
        
        result = service.check_model_deployment(
            model_id="clinical-model",
            target_environment="production",
            is_clinical=True,
            validation_status="pending",
            allowed_environment="production",
        )
        
        assert result.can_deploy is False
        assert "clinical_validation" in result.required_approvals
    
    def test_clinical_model_with_approval_can_deploy(self):
        """Test that approved clinical models can deploy"""
        service = MLGovernanceService()
        
        result = service.check_model_deployment(
            model_id="clinical-model",
            target_environment="production",
            is_clinical=True,
            validation_status="approved",
            allowed_environment="production",
            approvals=[{"type": "clinical_review", "approver": "dr_smith"}],
        )
        
        assert result.can_deploy is True
        assert len(result.issues) == 0
    
    def test_research_only_blocks_production(self):
        """Test that research-only models cannot go to production"""
        service = MLGovernanceService()
        
        result = service.check_model_deployment(
            model_id="research-model",
            target_environment="production",
            is_clinical=False,
            validation_status=None,
            allowed_environment="research_only",
        )
        
        assert result.can_deploy is False
        assert any("RESEARCH ONLY" in issue for issue in result.issues)
    
    def test_environment_level_comparison(self):
        """Test environment level ordering"""
        service = MLGovernanceService()
        
        assert service._environment_level("research_only") == 0
        assert service._environment_level("development") == 1
        assert service._environment_level("staging") == 2
        assert service._environment_level("production") == 3
    
    def test_cannot_deploy_above_allowed_environment(self):
        """Test that models cannot deploy above allowed environment"""
        service = MLGovernanceService()
        
        result = service.check_model_deployment(
            model_id="dev-model",
            target_environment="production",
            is_clinical=False,
            allowed_environment="staging",
        )
        
        assert result.can_deploy is False
        assert any("only approved for" in issue for issue in result.issues)


class TestClinicalValidation:
    """Tests for clinical model validation"""
    
    def test_validation_passes_with_good_metrics(self):
        """Test validation passes when metrics meet thresholds"""
        service = MLGovernanceService()
        
        metrics = {
            "auc_roc": 0.92,
            "sensitivity": 0.90,
            "specificity": 0.88,
            "f1_score": 0.85,
        }
        
        result = service.validate_clinical_model("test-model", metrics)
        
        assert result.is_valid is True
        assert result.status == ModelValidationStatus.APPROVED
        assert len(result.issues) == 0
    
    def test_validation_fails_with_low_metrics(self):
        """Test validation fails when metrics below thresholds"""
        service = MLGovernanceService()
        
        metrics = {
            "auc_roc": 0.65,
            "sensitivity": 0.70,
            "specificity": 0.60,
            "f1_score": 0.50,
        }
        
        result = service.validate_clinical_model("test-model", metrics)
        
        assert result.is_valid is False
        assert result.status == ModelValidationStatus.PENDING
        assert len(result.issues) > 0
    
    def test_validation_requires_all_metrics(self):
        """Test validation fails when required metrics missing"""
        service = MLGovernanceService()
        
        metrics = {
            "auc_roc": 0.92,
        }
        
        result = service.validate_clinical_model("test-model", metrics)
        
        assert result.is_valid is False
        assert any("missing" in issue.lower() for issue in result.issues)
    
    def test_custom_thresholds(self):
        """Test validation with custom thresholds"""
        service = MLGovernanceService()
        
        metrics = {"custom_metric": 0.75}
        thresholds = {"custom_metric": 0.70}
        
        result = service.validate_clinical_model("test-model", metrics, thresholds)
        
        assert result.is_valid is True


class TestEmbeddingStandardization:
    """Tests for embedding standardization checks"""
    
    def test_compliant_embeddings_pass(self):
        """Test that compliant embeddings pass standardization"""
        service = MLGovernanceService()
        
        embeddings = [
            {"id": "1", "embedding_model": "text-embedding-3-small", "embedding_version": "v1.0.0"},
            {"id": "2", "embedding_model": "text-embedding-3-small", "embedding_version": "v1.0.0"},
        ]
        
        is_valid, non_compliant = service.check_embedding_standardization(embeddings)
        
        assert is_valid is True
        assert len(non_compliant) == 0
    
    def test_null_model_fails(self):
        """Test that null embedding_model fails standardization"""
        service = MLGovernanceService()
        
        embeddings = [
            {"id": "1", "embedding_model": None, "embedding_version": "v1.0.0"},
            {"id": "2", "embedding_model": "text-embedding-3-small", "embedding_version": "v1.0.0"},
        ]
        
        is_valid, non_compliant = service.check_embedding_standardization(embeddings)
        
        assert is_valid is False
        assert len(non_compliant) == 1
        assert non_compliant[0]["id"] == "1"
    
    def test_null_version_fails(self):
        """Test that null embedding_version fails standardization"""
        service = MLGovernanceService()
        
        embeddings = [
            {"id": "1", "embedding_model": "text-embedding-3-small", "embedding_version": None},
        ]
        
        is_valid, non_compliant = service.check_embedding_standardization(embeddings)
        
        assert is_valid is False
        assert len(non_compliant) == 1


class TestResearchOnlyEnforcement:
    """Tests for research-only model enforcement"""
    
    def test_research_only_allows_development(self):
        """Test that research-only models work in development"""
        service = MLGovernanceService()
        
        result = service.enforce_research_only(
            model_id="research-model",
            allowed_environment="research_only",
            current_environment="dev",
        )
        
        assert result is True
    
    def test_research_only_blocks_production(self):
        """Test that research-only models are blocked in production"""
        service = MLGovernanceService()
        
        with pytest.raises(ResearchOnlyModel) as exc_info:
            service.enforce_research_only(
                model_id="research-model",
                allowed_environment="research_only",
                current_environment="production",
            )
        
        assert "RESEARCH ONLY" in str(exc_info.value)
        assert "research-model" in str(exc_info.value)
    
    def test_production_model_allows_production(self):
        """Test that production-approved models work in production"""
        service = MLGovernanceService()
        
        result = service.enforce_research_only(
            model_id="prod-model",
            allowed_environment="production",
            current_environment="production",
        )
        
        assert result is True


class TestRequiredApprovals:
    """Tests for required approval determination"""
    
    def test_production_requires_technical_and_qa(self):
        """Test that production requires technical and QA review"""
        service = MLGovernanceService()
        
        approvals = service.get_required_approvals(
            model_id="test-model",
            target_environment="production",
            clinical_risk_level="low",
        )
        
        approval_types = [a["type"] for a in approvals]
        assert "technical_review" in approval_types
        assert "qa_validation" in approval_types
    
    def test_high_risk_requires_clinical_review(self):
        """Test that high-risk models require clinical review"""
        service = MLGovernanceService()
        
        approvals = service.get_required_approvals(
            model_id="test-model",
            target_environment="production",
            clinical_risk_level="high",
        )
        
        approval_types = [a["type"] for a in approvals]
        assert "clinical_review" in approval_types
        assert "regulatory_review" in approval_types
    
    def test_critical_requires_executive_approval(self):
        """Test that critical models require executive approval"""
        service = MLGovernanceService()
        
        approvals = service.get_required_approvals(
            model_id="test-model",
            target_environment="production",
            clinical_risk_level="critical",
        )
        
        approval_types = [a["type"] for a in approvals]
        assert "executive_approval" in approval_types


class TestAuditEntry:
    """Tests for audit entry creation"""
    
    def test_audit_entry_format(self):
        """Test that audit entries have correct format"""
        service = MLGovernanceService()
        
        entry = service.create_audit_entry(
            event_type="model_deployment",
            actor_id="user-123",
            model_id="model-456",
            action="deploy_to_production",
            details={"environment": "production"},
            patient_id=None,
            success=True,
        )
        
        assert entry["event_type"] == "model_deployment"
        assert entry["actor_id"] == "user-123"
        assert entry["model_id"] == "model-456"
        assert entry["action"] == "deploy_to_production"
        assert entry["success"] is True
        assert "created_at" in entry


class TestSingletonService:
    """Tests for singleton service pattern"""
    
    def test_get_governance_service_returns_same_instance(self):
        """Test that singleton returns same instance"""
        service1 = get_governance_service()
        service2 = get_governance_service()
        
        assert service1 is service2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
