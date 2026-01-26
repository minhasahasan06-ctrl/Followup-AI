"""
ML Governance Service
=====================
Production-grade governance enforcement for clinical AI/ML:
1. Clinical model validation requirements
2. Research-only flag enforcement
3. Human approval gates for production
4. Embedding standardization checks
5. Audit logging for all governance actions

This service MUST be used before deploying any clinical model to production.
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum

from app.models.ml_governance_models import (
    ModelValidationStatus,
    ModelDeploymentEnvironment,
    ClinicalRiskLevel,
)
from app.services.ml_observability import get_observability_service

logger = logging.getLogger(__name__)

ENV = os.environ.get("ENV", "dev")


class GovernanceError(Exception):
    """Base exception for governance violations"""
    pass


class ClinicalValidationRequired(GovernanceError):
    """Raised when clinical validation is required but not completed"""
    pass


class ResearchOnlyModel(GovernanceError):
    """Raised when attempting to use research-only model in production"""
    pass


class HumanApprovalRequired(GovernanceError):
    """Raised when human approval is required but not obtained"""
    pass


class EmbeddingStandardizationError(GovernanceError):
    """Raised when embeddings don't meet standardization requirements"""
    pass


@dataclass
class ValidationResult:
    """Result of model validation check"""
    is_valid: bool
    status: ModelValidationStatus
    environment_allowed: ModelDeploymentEnvironment
    issues: List[str]
    warnings: List[str]
    requires_action: List[str]


@dataclass
class GovernanceCheck:
    """Result of governance check for model deployment"""
    can_deploy: bool
    model_id: str
    target_environment: str
    validation_status: str
    approval_status: str
    issues: List[str]
    required_approvals: List[str]


class MLGovernanceService:
    """
    Central governance service for ML model lifecycle management.
    
    Enforces:
    - Clinical models must be validated before production
    - Research-only models cannot be deployed to production
    - Human approval required for high-risk models
    - All embeddings must have model attribution
    """
    
    def __init__(self):
        self._observability = get_observability_service()
        self._current_env = ENV
        logger.info(f"ML Governance Service initialized (env: {self._current_env})")
    
    def check_model_deployment(
        self,
        model_id: str,
        target_environment: str,
        is_clinical: bool = False,
        validation_status: Optional[str] = None,
        allowed_environment: Optional[str] = None,
        approvals: Optional[List[Dict]] = None,
    ) -> GovernanceCheck:
        """
        Check if a model can be deployed to target environment.
        
        Args:
            model_id: The model identifier
            target_environment: Target deployment environment
            is_clinical: Whether this is a clinical model
            validation_status: Current validation status
            allowed_environment: Maximum allowed environment
            approvals: List of approval records
        
        Returns:
            GovernanceCheck with deployment decision and issues
        """
        issues = []
        required_approvals = []
        can_deploy = True
        
        target_level = self._environment_level(target_environment)
        allowed_level = self._environment_level(allowed_environment or "development")
        
        if target_level > allowed_level:
            can_deploy = False
            issues.append(
                f"Model is only approved for '{allowed_environment}' but deployment "
                f"requested for '{target_environment}'"
            )
        
        if is_clinical:
            if validation_status != ModelValidationStatus.APPROVED.value:
                can_deploy = False
                issues.append(
                    f"Clinical model requires APPROVED validation status. "
                    f"Current status: {validation_status}"
                )
                required_approvals.append("clinical_validation")
            
            if target_environment == "production":
                if not approvals or len(approvals) == 0:
                    can_deploy = False
                    issues.append("Clinical model requires human approval for production")
                    required_approvals.append("human_approval")
        
        if allowed_environment == ModelDeploymentEnvironment.RESEARCH_ONLY.value:
            if target_environment in ["staging", "production"]:
                can_deploy = False
                issues.append(
                    "This model is flagged as RESEARCH ONLY and cannot be "
                    "deployed to staging or production environments"
                )
        
        if can_deploy:
            logger.info(f"Governance check PASSED for model {model_id} -> {target_environment}")
        else:
            logger.warning(f"Governance check FAILED for model {model_id}: {issues}")
        
        return GovernanceCheck(
            can_deploy=can_deploy,
            model_id=model_id,
            target_environment=target_environment,
            validation_status=validation_status or "unknown",
            approval_status="approved" if approvals else "none",
            issues=issues,
            required_approvals=required_approvals,
        )
    
    def _environment_level(self, env: str) -> int:
        """Get numeric level for environment comparison"""
        levels = {
            "research_only": 0,
            "development": 1,
            "staging": 2,
            "production": 3,
        }
        return levels.get(env.lower(), 0)
    
    def validate_clinical_model(
        self,
        model_id: str,
        metrics: Dict[str, float],
        thresholds: Optional[Dict[str, float]] = None,
    ) -> ValidationResult:
        """
        Validate a clinical model against required metrics.
        
        Args:
            model_id: The model identifier
            metrics: Model performance metrics
            thresholds: Required metric thresholds (uses defaults if not provided)
        
        Returns:
            ValidationResult with validation decision and issues
        """
        if thresholds is None:
            thresholds = self._get_default_clinical_thresholds()
        
        issues = []
        warnings = []
        requires_action = []
        
        for metric_name, threshold in thresholds.items():
            actual = metrics.get(metric_name)
            if actual is None:
                issues.append(f"Required metric '{metric_name}' is missing")
                requires_action.append(f"Calculate and provide {metric_name}")
            elif actual < threshold:
                issues.append(
                    f"Metric '{metric_name}' ({actual:.3f}) below threshold ({threshold:.3f})"
                )
        
        if "sensitivity" in metrics and metrics["sensitivity"] < 0.90:
            warnings.append("Sensitivity below 90% - may miss positive cases")
        
        if "specificity" in metrics and metrics["specificity"] < 0.85:
            warnings.append("Specificity below 85% - may have high false positive rate")
        
        is_valid = len(issues) == 0
        status = ModelValidationStatus.APPROVED if is_valid else ModelValidationStatus.PENDING
        
        if not is_valid:
            requires_action.append("Address all validation issues")
            requires_action.append("Re-run validation after improvements")
        
        return ValidationResult(
            is_valid=is_valid,
            status=status,
            environment_allowed=(
                ModelDeploymentEnvironment.PRODUCTION if is_valid 
                else ModelDeploymentEnvironment.DEVELOPMENT
            ),
            issues=issues,
            warnings=warnings,
            requires_action=requires_action,
        )
    
    def _get_default_clinical_thresholds(self) -> Dict[str, float]:
        """Get default thresholds for clinical model validation"""
        return {
            "auc_roc": 0.80,
            "sensitivity": 0.85,
            "specificity": 0.80,
            "f1_score": 0.75,
        }
    
    def check_embedding_standardization(
        self,
        embeddings: List[Dict],
        require_model: bool = True,
        require_version: bool = True,
    ) -> Tuple[bool, List[Dict]]:
        """
        Check if embeddings meet standardization requirements.
        
        Args:
            embeddings: List of embedding records with metadata
            require_model: Whether embedding_model is required
            require_version: Whether embedding_version is required
        
        Returns:
            Tuple of (all_valid, list of non-compliant records)
        """
        non_compliant = []
        
        for emb in embeddings:
            issues = []
            
            if require_model and not emb.get("embedding_model"):
                issues.append("embedding_model is NULL")
            
            if require_version and not emb.get("embedding_version"):
                issues.append("embedding_version is NULL")
            
            if issues:
                non_compliant.append({
                    "id": emb.get("id"),
                    "issues": issues,
                })
        
        all_valid = len(non_compliant) == 0
        
        if not all_valid:
            self._observability.record_phi_detection(
                category="embedding_standardization",
                action="check_failed",
                operation="standardization_check",
            )
            logger.warning(f"Embedding standardization check failed: {len(non_compliant)} non-compliant records")
        
        return all_valid, non_compliant
    
    def enforce_research_only(
        self,
        model_id: str,
        allowed_environment: str,
        current_environment: Optional[str] = None,
    ) -> bool:
        """
        Enforce research-only restrictions on model usage.
        
        Args:
            model_id: The model identifier
            allowed_environment: The model's allowed environment
            current_environment: Current runtime environment
        
        Returns:
            True if usage is allowed, raises ResearchOnlyModel otherwise
        
        Raises:
            ResearchOnlyModel: If model is research-only and current env is production
        """
        env = current_environment or self._current_env
        
        if allowed_environment == ModelDeploymentEnvironment.RESEARCH_ONLY.value:
            if env in ["prod", "production"]:
                raise ResearchOnlyModel(
                    f"Model {model_id} is flagged as RESEARCH ONLY and cannot be "
                    "used in production. This model requires clinical validation "
                    "and human approval before production deployment."
                )
        
        return True
    
    def get_required_approvals(
        self,
        model_id: str,
        target_environment: str,
        clinical_risk_level: str,
    ) -> List[Dict[str, Any]]:
        """
        Get list of required approvals for model deployment.
        
        Args:
            model_id: The model identifier
            target_environment: Target deployment environment
            clinical_risk_level: The model's clinical risk level
        
        Returns:
            List of required approval specifications
        """
        required = []
        
        if target_environment in ["staging", "production"]:
            required.append({
                "type": "technical_review",
                "role": "ml_engineer",
                "required": True,
                "description": "Technical review of model architecture and implementation",
            })
        
        if target_environment == "production":
            required.append({
                "type": "qa_validation",
                "role": "qa_engineer",
                "required": True,
                "description": "Quality assurance testing and validation",
            })
        
        if clinical_risk_level in [ClinicalRiskLevel.HIGH.value, ClinicalRiskLevel.CRITICAL.value]:
            required.append({
                "type": "clinical_review",
                "role": "clinical_officer",
                "required": True,
                "description": "Clinical review by qualified medical officer",
            })
            
            required.append({
                "type": "regulatory_review",
                "role": "compliance_officer",
                "required": True,
                "description": "Regulatory compliance review for clinical use",
            })
        
        if clinical_risk_level == ClinicalRiskLevel.CRITICAL.value:
            required.append({
                "type": "executive_approval",
                "role": "executive",
                "required": True,
                "description": "Executive sign-off for critical clinical model",
            })
        
        return required
    
    def create_audit_entry(
        self,
        event_type: str,
        actor_id: str,
        model_id: str,
        action: str,
        details: Optional[Dict] = None,
        patient_id: Optional[str] = None,
        success: bool = True,
    ) -> Dict[str, Any]:
        """
        Create an audit log entry for governance actions.
        
        This is a helper method - actual persistence should be done
        by the calling code.
        
        Args:
            event_type: Type of governance event
            actor_id: ID of the actor performing the action
            model_id: The model being acted upon
            action: The action being performed
            details: Additional event details
            patient_id: Patient ID if applicable
            success: Whether the action succeeded
        
        Returns:
            Audit entry dict ready for persistence
        """
        return {
            "event_type": event_type,
            "actor_id": actor_id,
            "actor_type": "user",
            "model_id": model_id,
            "patient_id": patient_id,
            "action": action,
            "success": success,
            "event_metadata": details or {},
            "created_at": datetime.utcnow().isoformat(),
        }


_governance_service: Optional[MLGovernanceService] = None


def get_governance_service() -> MLGovernanceService:
    """Get singleton governance service"""
    global _governance_service
    if _governance_service is None:
        _governance_service = MLGovernanceService()
    return _governance_service
