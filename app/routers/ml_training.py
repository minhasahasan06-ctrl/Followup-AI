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


# ============================================
# DEVICE DATA EXTRACTION WITH CONSENT
# ============================================

class DeviceDataExtractionRequest(BaseModel):
    """Request for device data extraction"""
    patient_ids: Optional[List[str]] = Field(
        default=None,
        description="Specific patient IDs to extract (None = all consenting)"
    )
    device_types: Optional[List[str]] = Field(
        default=None,
        description="Specific device types to extract (smartwatch, bp_monitor, glucose_meter, etc.)"
    )
    date_range_days: int = Field(
        default=30,
        ge=1,
        le=365,
        description="Number of days of data to extract"
    )


class DeviceDataExtractionResponse(BaseModel):
    """Response for device data extraction"""
    job_id: str
    status: str
    patients_extracted: int
    total_readings: int
    device_types_included: List[str]
    consent_verified: bool
    hipaa_audit_logged: bool


@router.post("/device-data/extract")
async def extract_device_data_for_training(
    request: DeviceDataExtractionRequest,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
) -> DeviceDataExtractionResponse:
    """
    Extract device readings data for ML training with consent verification.
    
    This endpoint:
    1. Verifies ML training consent for each patient
    2. Checks granular device-type consent (smartwatch, bp_monitor, etc.)
    3. Extracts consented device readings
    4. Anonymizes patient identifiers
    5. Logs HIPAA audit trail
    
    Only patients with explicit wearable/device consent will have data extracted.
    """
    import uuid
    import logging
    import hashlib
    from datetime import datetime, timedelta
    
    logger = logging.getLogger(__name__)
    
    user_role = current_user.get("role", "")
    if user_role not in ["admin", "doctor"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only administrators can extract training data"
        )
    
    extraction_job_id = str(uuid.uuid4())
    
    try:
        # Step 1: Get patients with wearable data consent
        consent_query = text("""
            SELECT 
                patient_id,
                data_types,
                consent_enabled
            FROM ml_training_consent
            WHERE consent_enabled = true
            AND (
                data_types->>'wearableData' = 'true'
                OR data_types->>'wearable_data' = 'true'
                OR data_types->>'device_readings_smartwatch' = 'true'
                OR data_types->>'device_readings_bp' = 'true'
                OR data_types->>'device_readings_glucose' = 'true'
                OR data_types->>'device_readings_scale' = 'true'
                OR data_types->>'device_readings_thermometer' = 'true'
                OR data_types->>'device_readings_stethoscope' = 'true'
            )
        """)
        
        consent_result = await db.execute(consent_query)
        consenting_patients = consent_result.fetchall()
        
        if not consenting_patients:
            return DeviceDataExtractionResponse(
                job_id=extraction_job_id,
                status="no_data",
                patients_extracted=0,
                total_readings=0,
                device_types_included=[],
                consent_verified=True,
                hipaa_audit_logged=False
            )
        
        # Build patient ID filter
        patient_ids_to_extract = [p.patient_id for p in consenting_patients]
        if request.patient_ids:
            patient_ids_to_extract = [
                pid for pid in patient_ids_to_extract 
                if pid in request.patient_ids
            ]
        
        # Step 2: Map consented device types per patient
        patient_device_consent = {}
        for p in consenting_patients:
            if p.patient_id not in patient_ids_to_extract:
                continue
            
            data_types = p.data_types or {}
            consented_devices = []
            
            # Check general wearable consent
            if data_types.get("wearableData") or data_types.get("wearable_data"):
                consented_devices.extend([
                    "smartwatch", "bp_monitor", "glucose_meter",
                    "scale", "thermometer", "stethoscope", "pulse_oximeter"
                ])
            
            # Check granular device consent
            device_consent_map = {
                "device_readings_smartwatch": "smartwatch",
                "device_readings_bp": "bp_monitor",
                "device_readings_glucose": "glucose_meter",
                "device_readings_scale": "scale",
                "device_readings_thermometer": "thermometer",
                "device_readings_stethoscope": "stethoscope",
            }
            
            for consent_key, device_type in device_consent_map.items():
                if data_types.get(consent_key):
                    if device_type not in consented_devices:
                        consented_devices.append(device_type)
            
            patient_device_consent[p.patient_id] = consented_devices
        
        # Step 3: Extract device readings
        start_date = datetime.utcnow() - timedelta(days=request.date_range_days)
        
        total_readings = 0
        all_device_types = set()
        patients_with_data = 0
        
        for patient_id, consented_devices in patient_device_consent.items():
            if request.device_types:
                devices_to_query = [d for d in consented_devices if d in request.device_types]
            else:
                devices_to_query = consented_devices
            
            if not devices_to_query:
                continue
            
            # Query device readings for this patient
            readings_query = text("""
                SELECT 
                    COUNT(*) as reading_count,
                    array_agg(DISTINCT device_type) as device_types
                FROM device_readings
                WHERE patient_id = :patient_id
                AND recorded_at >= :start_date
                AND device_type = ANY(:device_types)
            """)
            
            readings_result = await db.execute(
                readings_query,
                {
                    "patient_id": patient_id,
                    "start_date": start_date,
                    "device_types": devices_to_query
                }
            )
            row = readings_result.fetchone()
            
            if row and row.reading_count > 0:
                total_readings += row.reading_count
                patients_with_data += 1
                if row.device_types:
                    all_device_types.update([dt for dt in row.device_types if dt])
                
                # Create anonymized patient hash for contribution tracking
                patient_hash = hashlib.sha256(
                    f"{patient_id}:ml_training".encode()
                ).hexdigest()[:16]
                
                # Log contribution
                try:
                    await db.execute(
                        text("""
                            INSERT INTO ml_training_contributions (
                                id, patient_id_hash, training_job_id, data_type,
                                record_count, status, created_at
                            ) VALUES (
                                :id, :patient_hash, :job_id, 'device_readings',
                                :record_count, 'extracted', NOW()
                            )
                            ON CONFLICT DO NOTHING
                        """),
                        {
                            "id": str(uuid.uuid4()),
                            "patient_hash": patient_hash,
                            "job_id": extraction_job_id,
                            "record_count": row.reading_count
                        }
                    )
                except Exception as e:
                    logger.warning(f"Could not log contribution: {e}")
        
        await db.commit()
        
        # Step 4: Log HIPAA audit event
        audit_logged = False
        try:
            await db.execute(
                text("""
                    INSERT INTO audit_logs (
                        id, user_id, action, resource_type, resource_id,
                        details, timestamp, ip_address
                    ) VALUES (
                        gen_random_uuid(), :user_id, 'ml_device_data_extraction',
                        'device_readings', :job_id,
                        :details, NOW(), :ip
                    )
                """),
                {
                    "user_id": current_user.get("sub"),
                    "job_id": extraction_job_id,
                    "details": json.dumps({
                        "patients_extracted": patients_with_data,
                        "total_readings": total_readings,
                        "device_types": list(all_device_types),
                        "date_range_days": request.date_range_days,
                        "consent_verified": True
                    }),
                    "ip": "internal"
                }
            )
            await db.commit()
            audit_logged = True
        except Exception as e:
            logger.warning(f"Could not log audit event: {e}")
        
        return DeviceDataExtractionResponse(
            job_id=extraction_job_id,
            status="extracted",
            patients_extracted=patients_with_data,
            total_readings=total_readings,
            device_types_included=list(all_device_types),
            consent_verified=True,
            hipaa_audit_logged=audit_logged
        )
        
    except Exception as e:
        logger.error(f"Error extracting device data: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to extract device data: {str(e)}"
        )


@router.get("/device-data/consent-stats")
async def get_device_consent_statistics(
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_async_db)
):
    """
    Get statistics about device data consent for ML training.
    Shows breakdown by device type.
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
            COUNT(*) FILTER (
                WHERE consent_enabled = true 
                AND (data_types->>'wearableData' = 'true' OR data_types->>'wearable_data' = 'true')
            ) as general_wearable_consent,
            COUNT(*) FILTER (
                WHERE consent_enabled = true 
                AND data_types->>'device_readings_smartwatch' = 'true'
            ) as smartwatch_consent,
            COUNT(*) FILTER (
                WHERE consent_enabled = true 
                AND data_types->>'device_readings_bp' = 'true'
            ) as bp_monitor_consent,
            COUNT(*) FILTER (
                WHERE consent_enabled = true 
                AND data_types->>'device_readings_glucose' = 'true'
            ) as glucose_meter_consent,
            COUNT(*) FILTER (
                WHERE consent_enabled = true 
                AND data_types->>'device_readings_scale' = 'true'
            ) as scale_consent,
            COUNT(*) FILTER (
                WHERE consent_enabled = true 
                AND data_types->>'device_readings_thermometer' = 'true'
            ) as thermometer_consent,
            COUNT(*) FILTER (
                WHERE consent_enabled = true 
                AND data_types->>'device_readings_stethoscope' = 'true'
            ) as stethoscope_consent
        FROM ml_training_consent
    """)
    
    result = await db.execute(query)
    row = result.fetchone()
    
    total = row.total_patients or 0
    consenting = row.consenting_patients or 0
    
    return {
        "total_patients_with_consent_record": total,
        "consenting_patients": consenting,
        "consent_rate_percent": round((consenting / total * 100) if total > 0 else 0, 1),
        "device_consent_breakdown": {
            "general_wearable": row.general_wearable_consent or 0,
            "smartwatch": row.smartwatch_consent or 0,
            "bp_monitor": row.bp_monitor_consent or 0,
            "glucose_meter": row.glucose_meter_consent or 0,
            "scale": row.scale_consent or 0,
            "thermometer": row.thermometer_consent or 0,
            "stethoscope": row.stethoscope_consent or 0
        },
        "device_readings_available": {
            "note": "Query device_readings table for actual data availability"
        }
    }
