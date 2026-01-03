"""
ML AI Router (Phase C.14-C.17)
==============================
AI-powered endpoints for ML governance and job planning.

Endpoints:
- C.15: POST /api/ml/ai/compose-job
- C.16: POST /api/ml/models/{id}/generate-governance
- C.17: Deploy gate implementation
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4, UUID

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.tinker_models import (
    TinkerJobReport,
    TinkerModelMetrics,
    TinkerGovernancePack,
    TinkerThresholdProfile,
    TinkerDriftRun,
)
from app.services.privacy_firewall import TinkerPurpose
from app.services.tinker_client import call_tinker

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/ml", tags=["ml-ai"])


def _compute_hash(data: Any) -> str:
    """Compute SHA256 hash of data"""
    data_str = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(data_str.encode('utf-8')).hexdigest()


class ComposeJobRequest(BaseModel):
    task_type: str = Field(..., pattern="^(training|validation|inference|drift_check)$")
    model_id: Optional[str] = None
    dataset_summary: str = Field(..., min_length=10, max_length=2000)
    constraints: Optional[str] = None
    actor_id: str = Field(..., min_length=1)


class ComposeJobResponse(BaseModel):
    job_id: str
    job_plan: Dict[str, Any]
    estimated_duration: str
    created_at: str


class GenerateGovernanceRequest(BaseModel):
    metrics_summary: str = Field(..., min_length=10)
    feature_names: List[str] = Field(default_factory=list)
    training_config: Optional[str] = None
    subgroup_metrics: Optional[str] = None
    calibration: Optional[str] = None
    actor_id: str = Field(..., min_length=1)


class GenerateGovernanceResponse(BaseModel):
    governance_id: str
    model_card: Dict[str, Any]
    datasheet: Dict[str, Any]
    is_complete: bool
    created_at: str


class DeployCheckRequest(BaseModel):
    model_id: str = Field(..., min_length=1)
    actor_id: str = Field(..., min_length=1)


class DeployCheckResponse(BaseModel):
    model_id: str
    can_deploy: bool
    checks: Dict[str, bool]
    missing_requirements: List[str]


def _log_phi_access(user_id: str, action: str, resource_type: str, resource_id: str) -> None:
    """Log PHI access for HIPAA compliance"""
    try:
        from app.services.access_control import HIPAAAuditLogger
        HIPAAAuditLogger.log_phi_access(
            user_id=user_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
        )
    except Exception as e:
        logger.warning(f"Failed to log PHI access: {e}")


@router.post("/ai/compose-job", response_model=ComposeJobResponse)
def compose_job(
    request: ComposeJobRequest,
    db: Session = Depends(get_db),
) -> ComposeJobResponse:
    """
    C.15: Compose ML job plan using AI.
    
    Flow: build dataset_summary -> Tinker -> JobPlan JSON
    """
    try:
        _log_phi_access(
            user_id=request.actor_id,
            action="ml_compose_job",
            resource_type="job_plan",
            resource_id="new",
        )
        
        payload = {
            "task_type": request.task_type,
            "dataset_summary": request.dataset_summary,
            "constraints": request.constraints or "standard",
        }
        
        response, success = call_tinker(
            purpose=TinkerPurpose.JOB_PLANNER.value,
            payload=payload,
            actor_role="admin",
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="AI service temporarily unavailable",
            )
        
        job_plan = {
            "task_type": request.task_type,
            "steps": response.get("jobs", [
                {"type": "data_validation", "priority": "high"},
                {"type": "feature_extraction", "priority": "medium"},
                {"type": request.task_type, "priority": "high"},
            ]),
            "resources": {
                "cpu": "4 cores",
                "memory": "16GB",
                "estimated_time": "30 minutes",
            },
            "constraints": request.constraints,
        }
        
        job_id = str(uuid4())
        
        job_report = TinkerJobReport(
            id=uuid4(),
            job_id=job_id,
            report_type="job_plan",
            report_json={"plan": job_plan, "status": "created"},
            report_hash=_compute_hash(job_plan),
            job_plan_json=job_plan,
            created_by=request.actor_id,
        )
        
        db.add(job_report)
        db.commit()
        
        return ComposeJobResponse(
            job_id=job_id,
            job_plan=job_plan,
            estimated_duration=job_plan["resources"]["estimated_time"],
            created_at=datetime.utcnow().isoformat(),
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error composing job: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to compose job plan",
        )


@router.post("/models/{model_id}/generate-governance", response_model=GenerateGovernanceResponse)
def generate_governance(
    model_id: str,
    request: GenerateGovernanceRequest,
    db: Session = Depends(get_db),
) -> GenerateGovernanceResponse:
    """
    C.16: Generate governance pack for ML model.
    
    Flow: metrics -> Tinker -> governance pack
    """
    try:
        _log_phi_access(
            user_id=request.actor_id,
            action="ml_generate_governance",
            resource_type="governance_pack",
            resource_id=model_id,
        )
        
        payload = {
            "metrics_summary": request.metrics_summary,
            "feature_names": request.feature_names,
            "training_config": request.training_config or "default",
            "subgroup_metrics": request.subgroup_metrics or "not_available",
            "calibration": request.calibration or "not_calibrated",
        }
        
        response, success = call_tinker(
            purpose=TinkerPurpose.MODEL_CARD.value,
            payload=payload,
            actor_role="admin",
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="AI service temporarily unavailable",
            )
        
        model_card = response.get("model_card", {
            "name": f"Model {model_id}",
            "version": "1.0.0",
            "description": "AI-generated model card",
            "intended_use": "Clinical decision support",
            "limitations": ["Retrospective data only", "Requires validation"],
            "ethical_considerations": ["Fairness across demographics", "Transparency"],
            "metrics": {},
            "feature_importance": {},
        })
        
        datasheet = {
            "dataset_description": "Patient health records",
            "collection_process": "EHR extraction",
            "preprocessing": "Standard clinical data pipeline",
            "features": request.feature_names,
            "label_definition": "Clinical outcomes",
            "known_biases": ["Selection bias", "Missing data patterns"],
        }
        
        is_complete = (
            request.subgroup_metrics is not None and
            request.calibration is not None and
            len(request.feature_names) > 0
        )
        
        governance = TinkerGovernancePack(
            id=uuid4(),
            model_id=model_id,
            model_version="1.0.0",
            model_card_json=model_card,
            datasheet_json=datasheet,
            validation_summary={"status": "pending"},
            calibration_summary={"status": "pending" if not request.calibration else "complete"},
            drift_config={"enabled": True, "frequency": "daily"},
            reproducibility_hash=_compute_hash({"model_id": model_id, "config": payload}),
            is_complete=is_complete,
            created_by=request.actor_id,
        )
        
        db.add(governance)
        db.commit()
        db.refresh(governance)
        
        return GenerateGovernanceResponse(
            governance_id=str(governance.id),
            model_card=model_card,
            datasheet=datasheet,
            is_complete=is_complete,
            created_at=datetime.utcnow().isoformat(),
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating governance: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate governance pack",
        )


@router.post("/models/{model_id}/deploy-check", response_model=DeployCheckResponse)
def check_deploy_gate(
    model_id: str,
    request: DeployCheckRequest,
    db: Session = Depends(get_db),
) -> DeployCheckResponse:
    """
    C.17: Deploy gate check - blocks deploy unless all requirements met.
    
    Requirements:
    - governance pack exists and is complete
    - validation results exist
    - calibration completed
    - drift config exists
    """
    try:
        _log_phi_access(
            user_id=request.actor_id,
            action="ml_deploy_check",
            resource_type="deploy_gate",
            resource_id=model_id,
        )
        
        checks = {
            "governance_exists": False,
            "governance_complete": False,
            "validation_complete": False,
            "calibration_complete": False,
            "drift_config_exists": False,
        }
        
        missing = []
        
        governance = db.query(TinkerGovernancePack).filter(
            TinkerGovernancePack.model_id == model_id
        ).order_by(TinkerGovernancePack.created_at.desc()).first()
        
        if governance:
            checks["governance_exists"] = True
            checks["governance_complete"] = governance.is_complete
            
            if governance.validation_summary and governance.validation_summary.get("status") == "complete":
                checks["validation_complete"] = True
            else:
                missing.append("Validation results not complete")
            
            if governance.calibration_summary and governance.calibration_summary.get("status") == "complete":
                checks["calibration_complete"] = True
            else:
                missing.append("Calibration not complete")
            
            if governance.drift_config and governance.drift_config.get("enabled"):
                checks["drift_config_exists"] = True
            else:
                missing.append("Drift monitoring not configured")
            
            if not governance.is_complete:
                missing.append("Governance pack incomplete")
        else:
            missing.append("Governance pack not found")
        
        can_deploy = all(checks.values())
        
        return DeployCheckResponse(
            model_id=model_id,
            can_deploy=can_deploy,
            checks=checks,
            missing_requirements=missing,
        )
        
    except Exception as e:
        logger.error(f"Error checking deploy gate: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to check deploy gate",
        )
