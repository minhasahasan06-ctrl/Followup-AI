"""
ML Inference API Endpoints
Provides REST API for ML predictions with caching and monitoring
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime, timedelta

from app.database import get_db
from app.dependencies import get_current_user
from app.services.ml_inference import ml_registry
from app.models.ml_models import MLModel, MLPrediction, MLPerformanceLog, MLBatchJob

router = APIRouter(prefix="/api/v1/ml", tags=["ml_inference"])


# ==================== Request/Response Models ====================

class PredictionRequest(BaseModel):
    """Generic prediction request"""
    model_name: str = Field(..., description="Name of the model to use")
    input_data: Dict[str, Any] = Field(..., description="Input features as dictionary")
    use_cache: bool = Field(True, description="Whether to use Redis cache")


class TextAnalysisRequest(BaseModel):
    """Request for NLP text analysis"""
    text: str = Field(..., description="Text to analyze")
    analysis_type: str = Field("symptom_ner", description="Type of analysis (symptom_ner, sentiment, etc.)")


class DeteriorationPredictionRequest(BaseModel):
    """Request for deterioration prediction"""
    patient_id: str
    baseline_data: Dict[str, Any]
    recent_measurements: List[Dict[str, Any]]


class PredictionResponse(BaseModel):
    """Standard prediction response"""
    prediction: Any
    confidence: Optional[float] = None
    metadata: Dict[str, Any]


# ==================== Endpoints ====================

@router.post("/predict", response_model=PredictionResponse)
async def generic_prediction(
    request: PredictionRequest,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Generic ML prediction endpoint
    Supports any loaded model with caching and audit logging
    """
    try:
        result = await ml_registry.predict(
            model_name=request.model_name,
            input_data=request.input_data,
            use_cache=request.use_cache,
            db=db,
            patient_id=current_user.get("sub")
        )
        
        return {
            "prediction": result.get("prediction"),
            "confidence": result.get("confidence"),
            "metadata": result.get("_metadata", {})
        }
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.post("/predict/symptom-analysis")
async def analyze_symptom_text(
    request: TextAnalysisRequest,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Analyze symptom text using Clinical-BERT NER
    Extracts medical entities (symptoms, medications, conditions)
    """
    try:
        result = await ml_registry.predict(
            model_name="clinical_ner",
            input_data={"text": request.text},
            use_cache=True,
            db=db,
            patient_id=current_user.get("sub")
        )
        
        # Process NER results
        entities = result.get("prediction", [])
        
        # Group entities by type
        grouped_entities = {}
        for entity in entities:
            entity_type = entity.get("entity_group", "UNKNOWN")
            if entity_type not in grouped_entities:
                grouped_entities[entity_type] = []
            grouped_entities[entity_type].append({
                "text": entity.get("word"),
                "score": entity.get("score"),
                "start": entity.get("start"),
                "end": entity.get("end")
            })
        
        return {
            "entities": grouped_entities,
            "raw_entities": entities,
            "metadata": result.get("_metadata", {})
        }
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.post("/predict/deterioration")
async def predict_deterioration(
    request: DeteriorationPredictionRequest,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Predict patient deterioration risk using LSTM model
    TODO: Implement once LSTM model is trained
    """
    # For now, return placeholder
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Deterioration prediction model not yet trained. Use rule-based risk scoring for now."
    )


@router.get("/models")
async def list_models(
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    List all available ML models
    """
    models = db.query(MLModel).filter(MLModel.is_active == True).all()
    
    return {
        "models": [
            {
                "id": model.id,
                "name": model.name,
                "version": model.version,
                "type": model.model_type,
                "task": model.task_type,
                "is_deployed": model.is_deployed,
                "metrics": model.metrics
            }
            for model in models
        ]
    }


@router.get("/models/{model_name}/performance")
async def get_model_performance(
    model_name: str,
    hours: int = 24,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get model performance metrics over time
    """
    # Get model
    model = db.query(MLModel).filter(
        MLModel.name == model_name,
        MLModel.is_active == True
    ).first()
    
    if not model:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model not found")
    
    # Get performance logs from last N hours
    since = datetime.utcnow() - timedelta(hours=hours)
    logs = db.query(MLPerformanceLog).filter(
        MLPerformanceLog.model_id == model.id,
        MLPerformanceLog.measured_at >= since
    ).all()
    
    # Group by metric name
    metrics_by_name = {}
    for log in logs:
        if log.metric_name not in metrics_by_name:
            metrics_by_name[log.metric_name] = []
        metrics_by_name[log.metric_name].append({
            "value": log.metric_value,
            "unit": log.metric_unit,
            "measured_at": log.measured_at.isoformat(),
            "aggregation": log.aggregation_type
        })
    
    return {
        "model_name": model_name,
        "model_version": model.version,
        "time_window_hours": hours,
        "metrics": metrics_by_name
    }


@router.get("/predictions/history")
async def get_prediction_history(
    limit: int = 50,
    prediction_type: Optional[str] = None,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get prediction history for current patient
    """
    patient_id = current_user.get("sub")
    
    query = db.query(MLPrediction).filter(
        MLPrediction.patient_id == patient_id
    )
    
    if prediction_type:
        query = query.filter(MLPrediction.prediction_type == prediction_type)
    
    predictions = query.order_by(MLPrediction.predicted_at.desc()).limit(limit).all()
    
    return {
        "predictions": [
            {
                "id": pred.id,
                "type": pred.prediction_type,
                "result": pred.prediction_result,
                "confidence": pred.confidence_score,
                "predicted_at": pred.predicted_at.isoformat(),
                "inference_time_ms": pred.inference_time_ms,
                "cache_hit": pred.cache_hit
            }
            for pred in predictions
        ],
        "total": len(predictions)
    }


@router.get("/stats")
async def get_ml_stats(
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get ML system statistics
    """
    # Total predictions
    total_predictions = db.query(MLPrediction).count()
    
    # Predictions today
    today = datetime.utcnow().date()
    predictions_today = db.query(MLPrediction).filter(
        MLPrediction.predicted_at >= today
    ).count()
    
    # Cache hit rate
    cache_hits = db.query(MLPrediction).filter(
        MLPrediction.cache_hit == True
    ).count()
    cache_hit_rate = (cache_hits / total_predictions * 100) if total_predictions > 0 else 0
    
    # Active models
    active_models = db.query(MLModel).filter(MLModel.is_active == True).count()
    
    # Average inference time
    from sqlalchemy import func
    avg_inference_time = db.query(func.avg(MLPrediction.inference_time_ms)).scalar() or 0
    
    return {
        "total_predictions": total_predictions,
        "predictions_today": predictions_today,
        "cache_hit_rate_percent": round(cache_hit_rate, 2),
        "active_models": active_models,
        "avg_inference_time_ms": round(avg_inference_time, 2),
        "redis_enabled": ml_registry._redis_client is not None
    }
