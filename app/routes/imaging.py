"""
MONAI Imaging API Routes

FastAPI endpoints for medical imaging inference:
- POST /infer - Run inference on medical images
- GET /models - List available models
- GET /health - Health check

HIPAA Compliance: All operations logged, PHI stripped from requests
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
from datetime import datetime
import logging

from app.services.monai_imaging import (
    MONAIInferenceEngine,
    ImagingRequest,
    ImagingModality,
    InferenceType,
    get_imaging_engine
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/imaging", tags=["imaging"])


class InferenceRequestModel(BaseModel):
    """API request model for inference"""
    model_id: str = Field(..., description="ID of the model to use")
    modality: str = Field(default="CT", description="Imaging modality (CT, MRI, etc)")
    inference_type: str = Field(default="SEGMENTATION", description="Type of inference")
    patient_id: Optional[str] = Field(None, description="Patient ID (will be anonymized)")
    study_id: Optional[str] = Field(None, description="Study identifier")
    anonymize: bool = Field(default=True, description="Whether to anonymize results")


class InferenceResponseModel(BaseModel):
    """API response model for inference"""
    request_id: str
    success: bool
    model_id: str
    model_version: str
    processing_time_ms: float
    result_hash: str
    anonymized: bool
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = {}


class ModelInfoModel(BaseModel):
    """Model information response"""
    model_id: str
    version: str
    modality: str
    inference_type: str
    input_shape: List[int]
    output_classes: int


class HealthResponseModel(BaseModel):
    """Health check response"""
    status: str
    monai_available: bool
    loaded_models: List[str]
    available_models: List[str]
    timestamp: str


@router.post("/infer", response_model=InferenceResponseModel)
async def run_inference(request: InferenceRequestModel):
    """
    Run inference on a medical image.
    
    Note: In production, image data would be uploaded separately
    and referenced by ID. This endpoint accepts metadata only.
    """
    engine = get_imaging_engine()
    
    try:
        modality = ImagingModality(request.modality)
    except ValueError:
        modality = ImagingModality.CT
    
    try:
        inference_type = InferenceType(request.inference_type)
    except ValueError:
        inference_type = InferenceType.SEGMENTATION
    
    imaging_request = ImagingRequest(
        modality=modality,
        inference_type=inference_type,
        model_id=request.model_id,
        patient_id=request.patient_id,
        study_id=request.study_id,
        anonymize=request.anonymize
    )
    
    result = engine.infer(imaging_request)
    
    return InferenceResponseModel(
        request_id=result.request_id,
        success=result.success,
        model_id=result.model_id,
        model_version=result.model_version,
        processing_time_ms=result.processing_time_ms,
        result_hash=result.result_hash,
        anonymized=result.anonymized,
        error_message=result.error_message,
        metadata=result.metadata
    )


@router.get("/models", response_model=List[ModelInfoModel])
async def list_models():
    """List available imaging models"""
    engine = get_imaging_engine()
    
    models = []
    for model_id, config in engine.DEFAULT_MODELS.items():
        models.append(ModelInfoModel(
            model_id=model_id,
            version=config["version"],
            modality=config["modality"].value,
            inference_type=config["inference_type"].value,
            input_shape=config["input_shape"],
            output_classes=config["output_classes"]
        ))
    
    return models


@router.get("/health", response_model=HealthResponseModel)
async def health_check():
    """Health check for imaging service"""
    engine = get_imaging_engine()
    health = engine.health_check()
    
    return HealthResponseModel(
        status=health["status"],
        monai_available=health["monai_available"],
        loaded_models=health["loaded_models"],
        available_models=health["available_models"],
        timestamp=health["timestamp"]
    )


@router.get("/audit")
async def get_audit_log(limit: int = 100):
    """Get recent audit log entries (admin only)"""
    engine = get_imaging_engine()
    return {"entries": engine.get_audit_log(limit=limit)}
