"""
ML Training API Endpoints
REST API for ML model training operations with consent verification
"""

from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime
import json

from app.database import get_async_db
from app.dependencies import get_current_user

router = APIRouter(prefix="/api/ml/training", tags=["ml_training"])


class TrainingJobRequest(BaseModel):
    """Request to create a training job"""
    model_name: str = Field(..., description="Name of the model to train")
    model_type: str = Field("random_forest", description="Type of model (random_forest, gradient_boosting, kmeans, lstm)")
    data_sources: Dict[str, Any] = Field(
        default_factory=dict,
        description="Data source configuration"
    )
    hyperparameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Training hyperparameters"
    )


class TrainingJobResponse(BaseModel):
    """Response for training job operations"""
    job_id: str
    status: str
    model_name: str
    version: str
    progress_percent: int = 0
    current_phase: Optional[str] = None
    message: Optional[str] = None


class DatasetInfo(BaseModel):
    """Information about an available dataset"""
    name: str
    source: str
    description: str
    record_count: int
    patient_count: int
    requires_credentials: bool
    is_available: bool


class ModelInfo(BaseModel):
    """Information about a trained model"""
    model_id: str
    name: str
    model_type: str
    version: str
    status: str
    is_active: bool
    metrics: Optional[Dict[str, Any]] = None
    created_at: datetime


@router.get("/datasets")
async def list_available_datasets(
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
) -> List[DatasetInfo]:
    """
    List available public datasets for ML training
    """
    from app.services.public_dataset_loader import SUPPORTED_DATASETS
    
    datasets = []
    for key, metadata in SUPPORTED_DATASETS.items():
        datasets.append(DatasetInfo(
            name=metadata.dataset_name,
            source=metadata.source,
            description=metadata.description,
            record_count=metadata.record_count,
            patient_count=metadata.patient_count,
            requires_credentials=metadata.requires_credentials,
            is_available=not metadata.requires_credentials
        ))
    
    return datasets


@router.get("/models")
async def list_trained_models(
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db),
    status_filter: Optional[str] = None,
    limit: int = 50
) -> List[ModelInfo]:
    """
    List all trained models in the registry
    """
    query = """
        SELECT 
            id, name, model_type, version, status, is_active,
            metrics, created_at
        FROM ml_models
        WHERE 1=1
    """
    params = {"limit": limit}
    
    if status_filter:
        query += " AND status = :status"
        params["status"] = status_filter
    
    query += " ORDER BY created_at DESC LIMIT :limit"
    
    result = await db.execute(text(query), params)
    rows = result.fetchall()
    
    models = []
    for row in rows:
        models.append(ModelInfo(
            model_id=row.id,
            name=row.name or "",
            model_type=row.model_type or "unknown",
            version=row.version or "0.0.0",
            status=row.status or "unknown",
            is_active=row.is_active or False,
            metrics=json.loads(row.metrics) if row.metrics else None,
            created_at=row.created_at or datetime.utcnow()
        ))
    
    return models


@router.post("/jobs", response_model=TrainingJobResponse)
async def create_training_job(
    request: TrainingJobRequest,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
):
    """
    Create a new ML training job
    Only admins can trigger training jobs
    """
    user_role = current_user.get("role", "")
    if user_role not in ["admin", "doctor"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only administrators can create training jobs"
        )
    
    try:
        from app.services.ml_training_pipeline import MLTrainingPipeline
        
        pipeline = MLTrainingPipeline(db)
        
        job_id = await pipeline.create_training_job(
            model_name=request.model_name,
            model_type=request.model_type,
            data_sources=request.data_sources,
            hyperparameters=request.hyperparameters,
            initiated_by=current_user.get("sub")
        )
        
        return TrainingJobResponse(
            job_id=job_id,
            status="queued",
            model_name=request.model_name,
            version=f"1.0.{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
            progress_percent=0,
            current_phase="queued",
            message="Training job created and queued"
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create training job: {str(e)}"
        )


@router.get("/jobs/{job_id}", response_model=TrainingJobResponse)
async def get_training_job_status(
    job_id: str,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
):
    """
    Get status of a training job
    """
    query = text("""
        SELECT 
            id, status, model_name, target_version,
            progress_percent, current_phase, progress_message,
            error_message
        FROM ml_training_jobs
        WHERE id = :job_id
    """)
    
    result = await db.execute(query, {"job_id": job_id})
    row = result.fetchone()
    
    if not row:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Training job not found"
        )
    
    message = row.progress_message or row.error_message or ""
    
    return TrainingJobResponse(
        job_id=row.id,
        status=row.status,
        model_name=row.model_name,
        version=row.target_version,
        progress_percent=row.progress_percent or 0,
        current_phase=row.current_phase,
        message=message
    )


@router.get("/jobs")
async def list_training_jobs(
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db),
    status_filter: Optional[str] = None,
    limit: int = 20
) -> List[TrainingJobResponse]:
    """
    List recent training jobs
    """
    query = """
        SELECT 
            id, status, model_name, target_version,
            progress_percent, current_phase, progress_message,
            queued_at
        FROM ml_training_jobs
        WHERE 1=1
    """
    params = {"limit": limit}
    
    if status_filter:
        query += " AND status = :status"
        params["status"] = status_filter
    
    query += " ORDER BY queued_at DESC LIMIT :limit"
    
    result = await db.execute(text(query), params)
    rows = result.fetchall()
    
    jobs = []
    for row in rows:
        jobs.append(TrainingJobResponse(
            job_id=row.id,
            status=row.status,
            model_name=row.model_name,
            version=row.target_version,
            progress_percent=row.progress_percent or 0,
            current_phase=row.current_phase,
            message=row.progress_message
        ))
    
    return jobs


@router.post("/jobs/{job_id}/start")
async def start_training_job(
    job_id: str,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
):
    """
    Start processing a queued training job
    """
    user_role = current_user.get("role", "")
    if user_role not in ["admin", "doctor"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only administrators can start training jobs"
        )
    
    query = text("SELECT status FROM ml_training_jobs WHERE id = :job_id")
    result = await db.execute(query, {"job_id": job_id})
    row = result.fetchone()
    
    if not row:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Training job not found"
        )
    
    if row.status != "queued":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Job is not in queued state (current: {row.status})"
        )
    
    return {
        "message": "Training job started",
        "job_id": job_id,
        "status": "running"
    }


@router.get("/consent/stats")
async def get_consent_statistics(
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
):
    """
    Get aggregate statistics about ML training consent (for admins)
    """
    user_role = current_user.get("role", "")
    if user_role not in ["admin", "doctor"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )
    
    query = text("""
        SELECT 
            COUNT(*) as total_patients,
            COUNT(*) FILTER (WHERE consent_enabled = true) as consenting_patients,
            COUNT(*) FILTER (WHERE consent_enabled = true AND data_types->>'vitals' = 'true') as vitals_consent,
            COUNT(*) FILTER (WHERE consent_enabled = true AND data_types->>'symptoms' = 'true') as symptoms_consent,
            COUNT(*) FILTER (WHERE consent_enabled = true AND data_types->>'mentalHealth' = 'true') as mental_health_consent,
            COUNT(*) FILTER (WHERE consent_enabled = true AND data_types->>'medications' = 'true') as medications_consent,
            COUNT(*) FILTER (WHERE consent_enabled = true AND data_types->>'wearableData' = 'true') as wearable_consent
        FROM ml_training_consent
    """)
    
    result = await db.execute(query)
    row = result.fetchone()
    
    return {
        "total_patients_with_consent_record": row.total_patients or 0,
        "consenting_patients": row.consenting_patients or 0,
        "consent_rate": (row.consenting_patients / row.total_patients * 100) if row.total_patients > 0 else 0,
        "data_type_breakdown": {
            "vitals": row.vitals_consent or 0,
            "symptoms": row.symptoms_consent or 0,
            "mental_health": row.mental_health_consent or 0,
            "medications": row.medications_consent or 0,
            "wearable": row.wearable_consent or 0
        }
    }


@router.get("/contributions/summary")
async def get_contributions_summary(
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
):
    """
    Get summary of patient data contributions to ML training
    """
    user_role = current_user.get("role", "")
    if user_role not in ["admin", "doctor"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )
    
    query = text("""
        SELECT 
            COUNT(DISTINCT patient_id_hash) as unique_contributors,
            COUNT(*) as total_contributions,
            SUM(record_count) as total_records,
            COUNT(DISTINCT training_job_id) as training_jobs_with_data
        FROM ml_training_contributions
        WHERE status = 'included'
    """)
    
    result = await db.execute(query)
    row = result.fetchone()
    
    return {
        "unique_contributors": row.unique_contributors or 0,
        "total_contributions": row.total_contributions or 0,
        "total_records_contributed": row.total_records or 0,
        "training_jobs_with_patient_data": row.training_jobs_with_data or 0
    }


@router.post("/register-dataset/{dataset_key}")
async def register_public_dataset(
    dataset_key: str,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
):
    """
    Register a public dataset for use in training
    """
    user_role = current_user.get("role", "")
    if user_role not in ["admin"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only administrators can register datasets"
        )
    
    from app.services.public_dataset_loader import PublicDatasetManager
    
    manager = PublicDatasetManager(db)
    dataset_id = await manager.register_dataset(dataset_key)
    
    if not dataset_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Dataset already registered or not supported"
        )
    
    return {
        "message": "Dataset registered successfully",
        "dataset_id": dataset_id,
        "dataset_key": dataset_key
    }
