"""
ML Prediction API Endpoints
===========================

Production-grade FastAPI endpoints for ML predictions with HIPAA compliance.

Endpoints:
- POST /api/ml/predict/disease-risk - Disease risk predictions (stroke, sepsis, diabetes)
- POST /api/ml/predict/deterioration - Clinical deterioration and readmission risk
- POST /api/ml/predict/time-series - Vital trend forecasting
- POST /api/ml/predict/segment - Patient segmentation
- POST /api/ml/predict/comprehensive - All predictions combined
"""

from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy.orm import Session
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime
import logging

from app.database import get_db
from app.dependencies import get_current_user
from app.services.ml_prediction_service import MLPredictionService
from app.services.audit_logger import AuditLogger

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/ml/predict", tags=["ml_predictions"])


# ==================== Request/Response Models ====================

class DiseaseRiskRequest(BaseModel):
    """Request for disease risk prediction."""
    patient_id: str = Field(..., description="Patient identifier")
    diseases: Optional[List[str]] = Field(
        None,
        description="List of diseases to predict (stroke, sepsis, diabetes). Default: all"
    )


class DeteriorationRequest(BaseModel):
    """Request for deterioration prediction."""
    patient_id: str = Field(..., description="Patient identifier")


class TimeSeriesRequest(BaseModel):
    """Request for time-series prediction."""
    patient_id: str = Field(..., description="Patient identifier")
    sequence_length: int = Field(14, description="Number of days to analyze", ge=3, le=30)


class SegmentationRequest(BaseModel):
    """Request for patient segmentation."""
    patient_id: str = Field(..., description="Patient identifier")


class ComprehensiveRequest(BaseModel):
    """Request for comprehensive ML assessment."""
    patient_id: str = Field(..., description="Patient identifier")


class PredictionResponse(BaseModel):
    """Standard prediction response."""
    patient_id: str
    predictions: Dict[str, Any]
    predicted_at: str
    model_version: str


# ==================== Helper Functions ====================

def verify_doctor_patient_access(
    db: Session,
    doctor_id: str,
    patient_id: str
) -> bool:
    """Verify doctor has access to patient data."""
    from app.models.patient_doctor_connection import DoctorPatientAssignment
    
    assignment = db.query(DoctorPatientAssignment).filter(
        DoctorPatientAssignment.doctor_id == doctor_id,
        DoctorPatientAssignment.patient_id == patient_id,
        DoctorPatientAssignment.status == "active"
    ).first()
    
    return assignment is not None


# ==================== Endpoints ====================

@router.post("/disease-risk")
async def predict_disease_risk(
    request: DiseaseRiskRequest,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Predict disease risks using Logistic Regression models.
    
    Diseases available:
    - stroke: Cardiovascular stroke risk
    - sepsis: Infection/sepsis risk  
    - diabetes: Type 2 diabetes risk
    
    Returns probability, risk level, confidence, and contributing factors.
    """
    doctor_id = current_user.get("sub")
    patient_id = request.patient_id
    
    # Verify access for doctor role
    if current_user.get("role") == "doctor":
        if not verify_doctor_patient_access(db, doctor_id, patient_id):
            AuditLogger.log_phi_access(
                db=db,
                user_id=doctor_id,
                patient_id=patient_id,
                action="ml_prediction_denied",
                resource_type="disease_risk",
                resource_id="unauthorized",
                phi_categories=["health_metrics"],
                success=False,
                details={"reason": "No active doctor-patient assignment"}
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="No authorized access to this patient"
            )
    
    try:
        service = MLPredictionService(db)
        result = await service.predict_disease_risks(
            patient_id=patient_id,
            diseases=request.diseases,
            doctor_id=doctor_id
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Disease risk prediction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@router.post("/deterioration")
async def predict_deterioration(
    request: DeteriorationRequest,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Predict clinical deterioration and readmission risk.
    
    Uses XGBoost/Random Forest ensemble approach to predict:
    - Deterioration risk score (0-10)
    - 30-day readmission probability
    
    Returns severity level, time to action, and contributing factors.
    """
    doctor_id = current_user.get("sub")
    patient_id = request.patient_id
    
    # Verify access for doctor role
    if current_user.get("role") == "doctor":
        if not verify_doctor_patient_access(db, doctor_id, patient_id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="No authorized access to this patient"
            )
    
    try:
        service = MLPredictionService(db)
        result = await service.predict_deterioration(
            patient_id=patient_id,
            doctor_id=doctor_id
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Deterioration prediction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@router.post("/time-series")
async def predict_vital_trends(
    request: TimeSeriesRequest,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Predict vital sign trends using LSTM-style time-series analysis.
    
    Analyzes historical vital data to:
    - Forecast next 24-hour values
    - Detect anomalies in vital patterns
    - Identify concerning trends
    
    Returns trend predictions with confidence intervals.
    """
    doctor_id = current_user.get("sub")
    patient_id = request.patient_id
    
    # Verify access for doctor role
    if current_user.get("role") == "doctor":
        if not verify_doctor_patient_access(db, doctor_id, patient_id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="No authorized access to this patient"
            )
    
    try:
        service = MLPredictionService(db)
        result = await service.predict_vital_trends(
            patient_id=patient_id,
            sequence_length=request.sequence_length,
            doctor_id=doctor_id
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Time-series prediction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@router.post("/segment")
async def segment_patient(
    request: SegmentationRequest,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Segment patient into health phenotype using K-Means clustering.
    
    Assigns patient to one of four segments:
    - wellness_engaged: Active, good outcomes
    - moderate_risk: Some challenges, moderate engagement
    - high_complexity: Multiple challenges, needs support
    - critical_needs: Significant concerns, urgent attention
    
    Returns segment assignment with confidence and recommendations.
    """
    doctor_id = current_user.get("sub")
    patient_id = request.patient_id
    
    # Verify access for doctor role
    if current_user.get("role") == "doctor":
        if not verify_doctor_patient_access(db, doctor_id, patient_id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="No authorized access to this patient"
            )
    
    try:
        service = MLPredictionService(db)
        result = await service.segment_patient(
            patient_id=patient_id,
            doctor_id=doctor_id
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Patient segmentation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Segmentation failed: {str(e)}"
        )


@router.post("/comprehensive")
async def get_comprehensive_assessment(
    request: ComprehensiveRequest,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get comprehensive ML assessment combining all prediction models.
    
    Returns unified view of:
    - Composite risk score
    - Disease risk predictions
    - Deterioration and readmission risk
    - Vital trend analysis
    - Patient segment
    
    Ideal for patient overview dashboards.
    """
    doctor_id = current_user.get("sub")
    patient_id = request.patient_id
    
    # Verify access for doctor role
    if current_user.get("role") == "doctor":
        if not verify_doctor_patient_access(db, doctor_id, patient_id):
            AuditLogger.log_phi_access(
                db=db,
                user_id=doctor_id,
                patient_id=patient_id,
                action="comprehensive_ml_denied",
                resource_type="comprehensive_assessment",
                resource_id="unauthorized",
                phi_categories=["health_metrics", "predictions"],
                success=False,
                details={"reason": "No active doctor-patient assignment"}
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="No authorized access to this patient"
            )
    
    try:
        service = MLPredictionService(db)
        result = await service.get_comprehensive_ml_assessment(
            patient_id=patient_id,
            doctor_id=doctor_id
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Comprehensive assessment failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Assessment failed: {str(e)}"
        )


@router.get("/models/info")
async def get_model_info(
    current_user: dict = Depends(get_current_user)
):
    """
    Get information about available ML models.
    """
    return {
        "models": [
            {
                "id": "disease_risk_logistic",
                "name": "Disease Risk Predictor",
                "type": "Logistic Regression",
                "version": "1.0.0",
                "diseases": ["stroke", "sepsis", "diabetes"],
                "description": "Predicts disease risk probabilities based on patient health data"
            },
            {
                "id": "deterioration_xgboost",
                "name": "Clinical Deterioration Predictor",
                "type": "XGBoost/Random Forest",
                "version": "1.0.0",
                "predictions": ["deterioration_risk", "readmission_risk"],
                "description": "Predicts clinical deterioration and 30-day readmission risk"
            },
            {
                "id": "vital_trends_lstm",
                "name": "Vital Trends Analyzer",
                "type": "LSTM/Time-Series",
                "version": "1.0.0",
                "predictions": ["vital_forecasts", "anomaly_detection"],
                "description": "Analyzes vital sign trends and predicts future values"
            },
            {
                "id": "patient_segmentation_kmeans",
                "name": "Patient Segmentation Model",
                "type": "K-Means Clustering",
                "version": "1.0.0",
                "segments": ["wellness_engaged", "moderate_risk", "high_complexity", "critical_needs"],
                "description": "Segments patients into health phenotypes for personalized care"
            }
        ],
        "total_models": 4,
        "last_updated": "2024-12-02"
    }


@router.get("/health")
async def ml_prediction_health():
    """Health check for ML prediction service."""
    return {
        "status": "healthy",
        "service": "ml_prediction",
        "models_loaded": True,
        "timestamp": datetime.utcnow().isoformat()
    }


# ==================== GET Endpoints for Frontend ====================

@router.get("/disease-risk/{patient_id}")
async def get_disease_risk(
    patient_id: str,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    GET endpoint for disease risk predictions (frontend-compatible).
    Returns disease risk predictions for the specified patient.
    """
    doctor_id = current_user.get("sub")
    user_role = current_user.get("role", "patient")
    
    # Access control: patients can view own data, doctors need assignment
    if user_role == "doctor":
        if not verify_doctor_patient_access(db, doctor_id, patient_id):
            AuditLogger.log_phi_access(
                db=db,
                user_id=doctor_id,
                patient_id=patient_id,
                action="ml_prediction_denied",
                resource_type="disease_risk",
                resource_id="unauthorized",
                phi_categories=["health_metrics", "ml_predictions", "risk_scores"],
                success=False,
                details={"reason": "No active doctor-patient assignment"}
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="No authorized access to this patient"
            )
    elif user_role == "patient" and doctor_id != patient_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cannot access other patient's data"
        )
    
    try:
        service = MLPredictionService(db)
        result = await service.predict_disease_risks(
            patient_id=patient_id,
            diseases=None,  # Get all diseases
            doctor_id=doctor_id
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Disease risk prediction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@router.get("/deterioration/{patient_id}")
async def get_deterioration(
    patient_id: str,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    GET endpoint for deterioration/readmission risk (frontend-compatible).
    Returns clinical deterioration and readmission predictions.
    """
    doctor_id = current_user.get("sub")
    user_role = current_user.get("role", "patient")
    
    # Access control
    if user_role == "doctor":
        if not verify_doctor_patient_access(db, doctor_id, patient_id):
            AuditLogger.log_phi_access(
                db=db,
                user_id=doctor_id,
                patient_id=patient_id,
                action="ml_prediction_denied",
                resource_type="deterioration",
                resource_id="unauthorized",
                phi_categories=["health_metrics", "ml_predictions", "risk_scores"],
                success=False,
                details={"reason": "No active doctor-patient assignment"}
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="No authorized access to this patient"
            )
    elif user_role == "patient" and doctor_id != patient_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cannot access other patient's data"
        )
    
    try:
        service = MLPredictionService(db)
        result = await service.predict_deterioration(
            patient_id=patient_id,
            doctor_id=doctor_id
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Deterioration prediction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@router.get("/time-series/{patient_id}")
async def get_vital_trends(
    patient_id: str,
    sequence_length: int = 14,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    GET endpoint for vital trend predictions (frontend-compatible).
    Returns LSTM-style vital sign forecasts.
    """
    doctor_id = current_user.get("sub")
    user_role = current_user.get("role", "patient")
    
    # Access control
    if user_role == "doctor":
        if not verify_doctor_patient_access(db, doctor_id, patient_id):
            AuditLogger.log_phi_access(
                db=db,
                user_id=doctor_id,
                patient_id=patient_id,
                action="ml_prediction_denied",
                resource_type="time_series",
                resource_id="unauthorized",
                phi_categories=["health_metrics", "ml_predictions", "vital_signs"],
                success=False,
                details={"reason": "No active doctor-patient assignment"}
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="No authorized access to this patient"
            )
    elif user_role == "patient" and doctor_id != patient_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cannot access other patient's data"
        )
    
    try:
        service = MLPredictionService(db)
        result = await service.predict_vital_trends(
            patient_id=patient_id,
            sequence_length=min(max(sequence_length, 3), 30),
            doctor_id=doctor_id
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Time-series prediction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@router.get("/patient-segments/{patient_id}")
async def get_patient_segment(
    patient_id: str,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    GET endpoint for patient segmentation (frontend-compatible).
    Returns K-Means cluster assignment and phenotype profile.
    """
    doctor_id = current_user.get("sub")
    user_role = current_user.get("role", "patient")
    
    # Access control
    if user_role == "doctor":
        if not verify_doctor_patient_access(db, doctor_id, patient_id):
            AuditLogger.log_phi_access(
                db=db,
                user_id=doctor_id,
                patient_id=patient_id,
                action="ml_prediction_denied",
                resource_type="segmentation",
                resource_id="unauthorized",
                phi_categories=["health_metrics", "ml_predictions", "demographic_info"],
                success=False,
                details={"reason": "No active doctor-patient assignment"}
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="No authorized access to this patient"
            )
    elif user_role == "patient" and doctor_id != patient_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cannot access other patient's data"
        )
    
    try:
        service = MLPredictionService(db)
        result = await service.segment_patient(
            patient_id=patient_id,
            doctor_id=doctor_id
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Patient segmentation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Segmentation failed: {str(e)}"
        )


@router.get("/comprehensive/{patient_id}")
async def get_comprehensive_ml_assessment(
    patient_id: str,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    GET endpoint for comprehensive ML assessment (frontend-compatible).
    Returns all prediction types in a unified response.
    """
    doctor_id = current_user.get("sub")
    user_role = current_user.get("role", "patient")
    
    # Access control
    if user_role == "doctor":
        if not verify_doctor_patient_access(db, doctor_id, patient_id):
            AuditLogger.log_phi_access(
                db=db,
                user_id=doctor_id,
                patient_id=patient_id,
                action="ml_prediction_denied",
                resource_type="comprehensive_assessment",
                resource_id="unauthorized",
                phi_categories=["health_metrics", "ml_predictions", "risk_scores", "vital_signs"],
                success=False,
                details={"reason": "No active doctor-patient assignment"}
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="No authorized access to this patient"
            )
    elif user_role == "patient" and doctor_id != patient_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cannot access other patient's data"
        )
    
    try:
        service = MLPredictionService(db)
        result = await service.get_comprehensive_ml_assessment(
            patient_id=patient_id,
            doctor_id=doctor_id
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Comprehensive assessment failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Assessment failed: {str(e)}"
        )
