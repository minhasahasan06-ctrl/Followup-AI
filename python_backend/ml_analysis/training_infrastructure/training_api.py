"""
Training Infrastructure API
============================
FastAPI router for ML training infrastructure with:
- Job creation and management
- Status monitoring
- Model registry access
- Artifact management

All endpoints are HIPAA-compliant with audit logging.
"""

import logging
from datetime import datetime
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/ml-training", tags=["ML Training"])


# Pydantic models for request/response
class CreateJobRequest(BaseModel):
    job_type: str = Field(..., description="Type of training job")
    model_name: str = Field(..., description="Name for the model")
    config: Dict[str, Any] = Field(default_factory=dict, description="Training configuration")
    priority: int = Field(default=5, ge=1, le=20, description="Job priority (1-20)")


class JobResponse(BaseModel):
    job_id: str
    job_type: str
    model_name: str
    status: str
    priority: int
    progress_percent: int
    current_step: str
    created_at: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error_message: Optional[str] = None
    metrics: Dict[str, Any] = Field(default_factory=dict)
    artifact_path: Optional[str] = None
    consent_verified: bool = False
    governance_approved: bool = False


class QueueStatsResponse(BaseModel):
    queue_size: int
    status_counts: Dict[str, int]
    total_jobs: int
    avg_duration_seconds: float
    completed_count: int
    failed_count: int
    success_rate: float
    redis_connected: bool


class ModelVersionResponse(BaseModel):
    model_id: str
    model_name: str
    version: str
    deployment_status: str
    is_active: bool
    metrics: Dict[str, Any]
    created_at: Optional[str] = None
    deployed_at: Optional[str] = None


class ArtifactResponse(BaseModel):
    artifact_id: str
    model_name: str
    model_type: str
    version: str
    file_path: str
    file_size_bytes: int
    metrics: Dict[str, Any]
    created_at: Optional[str] = None


# Lazy initialization of services
_queue = None
_worker = None
_version_manager = None
_artifact_storage = None


def get_queue():
    """Get or create training job queue"""
    global _queue
    if _queue is None:
        from .training_job_queue import TrainingJobQueue
        _queue = TrainingJobQueue()
    return _queue


def get_worker():
    """Get or create training job worker"""
    global _worker
    if _worker is None:
        from .training_job_worker import TrainingJobWorker
        _worker = TrainingJobWorker(queue=get_queue())
    return _worker


def get_version_manager():
    """Get or create model version manager"""
    global _version_manager
    if _version_manager is None:
        from .model_versioning import ModelVersionManager
        _version_manager = ModelVersionManager()
    return _version_manager


def get_artifact_storage():
    """Get or create artifact storage"""
    global _artifact_storage
    if _artifact_storage is None:
        from .artifact_storage import ArtifactStorage
        _artifact_storage = ArtifactStorage()
    return _artifact_storage


def job_to_response(job) -> JobResponse:
    """Convert TrainingJob to response model"""
    return JobResponse(
        job_id=job.job_id,
        job_type=job.job_type,
        model_name=job.model_name,
        status=job.status.value if hasattr(job.status, 'value') else str(job.status),
        priority=job.priority,
        progress_percent=job.progress_percent,
        current_step=job.current_step,
        created_at=job.created_at.isoformat() if job.created_at else None,
        started_at=job.started_at.isoformat() if job.started_at else None,
        completed_at=job.completed_at.isoformat() if job.completed_at else None,
        error_message=job.error_message,
        metrics=job.metrics,
        artifact_path=job.artifact_path,
        consent_verified=job.consent_verified,
        governance_approved=job.governance_approved
    )


# ========================
# Job Management Endpoints
# ========================

@router.post("/jobs", response_model=JobResponse)
async def create_training_job(request: CreateJobRequest):
    """Create a new training job"""
    try:
        queue = get_queue()
        job = queue.create_job(
            job_type=request.job_type,
            model_name=request.model_name,
            config=request.config,
            priority=request.priority,
            created_by="api"
        )
        
        return job_to_response(job)
        
    except Exception as e:
        logger.error(f"Error creating job: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/jobs", response_model=List[JobResponse])
async def list_jobs(
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(50, ge=1, le=100)
):
    """List training jobs"""
    try:
        queue = get_queue()
        jobs = queue.get_recent_jobs(limit=limit, status=status)
        return [job_to_response(job) for job in jobs]
        
    except Exception as e:
        logger.error(f"Error listing jobs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job(job_id: str):
    """Get a specific job by ID"""
    try:
        queue = get_queue()
        job = queue.get_job(job_id)
        
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        return job_to_response(job)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting job: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/jobs/{job_id}/history")
async def get_job_history(job_id: str):
    """Get audit history for a job"""
    try:
        queue = get_queue()
        history = queue.get_job_history(job_id)
        
        # Convert datetime objects
        for event in history:
            if event.get('created_at'):
                event['created_at'] = event['created_at'].isoformat()
        
        return {"job_id": job_id, "history": history}
        
    except Exception as e:
        logger.error(f"Error getting job history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/jobs/{job_id}/cancel")
async def cancel_job(job_id: str):
    """Cancel a pending or running job"""
    try:
        queue = get_queue()
        success = queue.cancel_job(job_id, cancelled_by="api")
        
        if not success:
            raise HTTPException(status_code=400, detail="Could not cancel job")
        
        return {"success": True, "job_id": job_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelling job: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/jobs/{job_id}/retry")
async def retry_job(job_id: str):
    """Retry a failed job"""
    try:
        queue = get_queue()
        success = queue.retry_job(job_id, retried_by="api")
        
        if not success:
            raise HTTPException(status_code=400, detail="Could not retry job")
        
        return {"success": True, "job_id": job_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrying job: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ========================
# Queue Stats & Worker
# ========================

@router.get("/queue/stats", response_model=QueueStatsResponse)
async def get_queue_stats():
    """Get queue statistics"""
    try:
        queue = get_queue()
        stats = queue.get_queue_stats()
        return QueueStatsResponse(**stats)
        
    except Exception as e:
        logger.error(f"Error getting queue stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/worker/status")
async def get_worker_status():
    """Get worker status"""
    try:
        worker = get_worker()
        current_job = worker.get_current_job()
        
        return {
            "worker_id": worker.worker_id,
            "is_running": worker.is_running(),
            "current_job": job_to_response(current_job).dict() if current_job else None
        }
        
    except Exception as e:
        logger.error(f"Error getting worker status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/worker/start")
async def start_worker():
    """Start the training worker"""
    try:
        worker = get_worker()
        
        if worker.is_running():
            return {"success": True, "message": "Worker already running"}
        
        worker.start()
        return {"success": True, "message": "Worker started"}
        
    except Exception as e:
        logger.error(f"Error starting worker: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/worker/stop")
async def stop_worker():
    """Stop the training worker"""
    try:
        worker = get_worker()
        worker.stop()
        return {"success": True, "message": "Worker stopped"}
        
    except Exception as e:
        logger.error(f"Error stopping worker: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ========================
# Model Versions
# ========================

@router.get("/models/{model_name}/versions", response_model=List[ModelVersionResponse])
async def list_model_versions(model_name: str, limit: int = Query(20, ge=1, le=100)):
    """Get version history for a model"""
    try:
        vm = get_version_manager()
        versions = vm.get_version_history(model_name, limit=limit)
        
        return [
            ModelVersionResponse(
                model_id=v.model_id,
                model_name=v.model_name,
                version=v.version,
                deployment_status=v.deployment_status.value if hasattr(v.deployment_status, 'value') else str(v.deployment_status),
                is_active=v.is_active,
                metrics=v.metrics,
                created_at=v.created_at.isoformat() if v.created_at else None,
                deployed_at=v.deployed_at.isoformat() if v.deployed_at else None
            )
            for v in versions
        ]
        
    except Exception as e:
        logger.error(f"Error listing model versions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/{model_name}/active", response_model=Optional[ModelVersionResponse])
async def get_active_version(model_name: str):
    """Get the currently active version for a model"""
    try:
        vm = get_version_manager()
        version = vm.get_active_version(model_name)
        
        if not version:
            return None
        
        return ModelVersionResponse(
            model_id=version.model_id,
            model_name=version.model_name,
            version=version.version,
            deployment_status=version.deployment_status.value if hasattr(version.deployment_status, 'value') else str(version.deployment_status),
            is_active=version.is_active,
            metrics=version.metrics,
            created_at=version.created_at.isoformat() if version.created_at else None,
            deployed_at=version.deployed_at.isoformat() if version.deployed_at else None
        )
        
    except Exception as e:
        logger.error(f"Error getting active version: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/{model_name}/promote/{version}")
async def promote_version(model_name: str, version: str):
    """Promote a version to production"""
    try:
        vm = get_version_manager()
        success = vm.promote_to_production(model_name, version, promoted_by="api")
        
        if not success:
            raise HTTPException(status_code=400, detail="Could not promote version")
        
        return {"success": True, "model_name": model_name, "version": version}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error promoting version: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/{model_name}/rollback/{version}")
async def rollback_version(model_name: str, version: str):
    """Rollback to a previous version"""
    try:
        vm = get_version_manager()
        success = vm.rollback_to_version(model_name, version, rolled_back_by="api")
        
        if not success:
            raise HTTPException(status_code=400, detail="Could not rollback version")
        
        return {"success": True, "model_name": model_name, "version": version}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error rolling back version: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/{model_name}/compare")
async def compare_versions(
    model_name: str,
    version1: str = Query(...),
    version2: str = Query(...)
):
    """Compare metrics between two versions"""
    try:
        vm = get_version_manager()
        comparison = vm.compare_versions_metrics(model_name, version1, version2)
        return comparison
        
    except Exception as e:
        logger.error(f"Error comparing versions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ========================
# Artifacts
# ========================

@router.get("/artifacts", response_model=List[ArtifactResponse])
async def list_artifacts(
    model_name: Optional[str] = Query(None),
    model_type: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=100)
):
    """List model artifacts"""
    try:
        storage = get_artifact_storage()
        artifacts = storage.list_artifacts(
            model_name=model_name,
            model_type=model_type,
            limit=limit
        )
        
        return [
            ArtifactResponse(
                artifact_id=a.artifact_id,
                model_name=a.model_name,
                model_type=a.model_type,
                version=a.version,
                file_path=a.file_path,
                file_size_bytes=a.file_size_bytes,
                metrics=a.metrics,
                created_at=a.created_at.isoformat() if a.created_at else None
            )
            for a in artifacts
        ]
        
    except Exception as e:
        logger.error(f"Error listing artifacts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/artifacts/{artifact_id}")
async def get_artifact(artifact_id: str):
    """Get artifact details"""
    try:
        storage = get_artifact_storage()
        artifact = storage.load_model_artifact(artifact_id, verify_checksum=False)
        
        if not artifact:
            raise HTTPException(status_code=404, detail="Artifact not found")
        
        # Return metadata only, not the full model data
        return {
            "artifact_id": artifact_id,
            "loaded": True,
            "keys": list(artifact.keys()) if isinstance(artifact, dict) else ["data"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting artifact: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ========================
# Consent & Governance
# ========================

@router.post("/consent/check")
async def check_consent(
    patient_ids: List[str],
    data_categories: List[str],
    purpose: str = "ml_training"
):
    """Check patient consent for data access"""
    try:
        from .consent_enforcer import ConsentEnforcer, DataCategory
        
        enforcer = ConsentEnforcer()
        
        # Convert string categories to enum
        categories = []
        for cat in data_categories:
            try:
                categories.append(DataCategory(cat))
            except ValueError:
                pass
        
        result = enforcer.check_patient_consent(
            patient_ids=patient_ids,
            data_categories=categories,
            purpose=purpose,
            requester_id="api"
        )
        
        return {
            "allowed": result.allowed,
            "consented_count": len(result.patient_ids_with_consent),
            "denied_count": len(result.patient_ids_denied),
            "categories_allowed": result.categories_allowed,
            "categories_denied": result.categories_denied,
            "k_anonymity_met": result.k_anonymity_met,
            "message": result.message
        }
        
    except Exception as e:
        logger.error(f"Error checking consent: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/governance/pre-build-check")
async def governance_pre_build_check(
    cohort_spec: Dict[str, Any],
    analysis_spec: Dict[str, Any],
    purpose: str = "ml_training"
):
    """Run governance pre-build checks"""
    try:
        from .governance_hooks import GovernanceHooks
        
        hooks = GovernanceHooks()
        result = hooks.run_pre_build_checks(
            cohort_spec=cohort_spec,
            analysis_spec=analysis_spec,
            requester_id="api",
            purpose=purpose
        )
        
        return result.to_dict()
        
    except Exception as e:
        logger.error(f"Error in governance check: {e}")
        raise HTTPException(status_code=500, detail=str(e))
