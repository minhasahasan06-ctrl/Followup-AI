"""
Tremor Analysis API Endpoints
Upload accelerometer data, analyze tremor, retrieve metrics
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from datetime import datetime
import logging

from app.database import get_db
from app.models.video_ai_models import AccelerometerTremorData
from app.services.tremor_analysis_service import TremorAnalysisService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/tremor", tags=["Tremor Analysis"])


class AccelerometerDataUpload(BaseModel):
    """Accelerometer data from phone"""
    patient_id: str = Field(..., description="Patient ID")
    timestamps: List[float] = Field(..., description="Timestamps in milliseconds")
    accel_x: List[float] = Field(..., description="X-axis acceleration (m/s²)")
    accel_y: List[float] = Field(..., description="Y-axis acceleration (m/s²)")
    accel_z: List[float] = Field(..., description="Z-axis acceleration (m/s²)")
    device_type: str = Field(default="phone", description="Device type")
    device_model: str = Field(default="unknown", description="Device model")
    browser_info: str = Field(default="unknown", description="Browser info")


@router.post("/upload")
def upload_accelerometer_data(
    data: AccelerometerDataUpload,
    db: Session = Depends(get_db)
):
    """
    Upload accelerometer data for tremor analysis
    
    **Patient uploads phone accelerometer readings (5-10 seconds recommended)**
    """
    logger.info(f"Uploading accelerometer data for patient {data.patient_id}, {len(data.timestamps)} samples")
    
    # Validate data
    if len(data.timestamps) < 100:
        raise HTTPException(status_code=400, detail="Insufficient data. Record for at least 5 seconds.")
    
    if not (len(data.timestamps) == len(data.accel_x) == len(data.accel_y) == len(data.accel_z)):
        raise HTTPException(status_code=400, detail="Timestamps and acceleration arrays must have equal length")
    
    try:
        # Analyze tremor
        tremor_service = TremorAnalysisService(db)
        result = tremor_service.analyze_tremor(
            patient_id=data.patient_id,
            timestamps=data.timestamps,
            accel_x=data.accel_x,
            accel_y=data.accel_y,
            accel_z=data.accel_z,
            device_info={
                'type': data.device_type,
                'model': data.device_model,
                'browser': data.browser_info,
            }
        )
        
        return {
            "tremor_data_id": result['tremor_data_id'],
            "status": "completed",
            "tremor_detected": result['tremor_detected'],
            "tremor_index": result['tremor_index'],
            "dominant_frequency_hz": result['dominant_frequency_hz'],
            "analysis": {
                "parkinsonian_likelihood": result['parkinsonian_likelihood'],
                "essential_tremor_likelihood": result['essential_tremor_likelihood'],
                "physiological_tremor": result['physiological_tremor'],
            }
        }
    
    except Exception as e:
        logger.error(f"Error analyzing tremor data: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error analyzing tremor: {str(e)}")


@router.get("/latest/{patient_id}")
def get_latest_tremor(
    patient_id: str,
    db: Session = Depends(get_db)
):
    """Get patient's most recent tremor analysis"""
    tremor_service = TremorAnalysisService(db)
    result = tremor_service.get_latest_tremor(patient_id)
    
    if not result:
        return {
            "patient_id": patient_id,
            "has_data": False,
            "message": "No tremor data recorded yet"
        }
    
    return {
        "patient_id": patient_id,
        "has_data": True,
        **result
    }


@router.get("/history/{patient_id}")
def get_tremor_history(
    patient_id: str,
    limit: int = 10,
    db: Session = Depends(get_db)
):
    """Get patient's tremor history"""
    tremor_records = db.query(AccelerometerTremorData).filter(
        AccelerometerTremorData.patient_id == patient_id
    ).order_by(AccelerometerTremorData.created_at.desc()).limit(limit).all()
    
    return {
        "patient_id": patient_id,
        "total_records": len(tremor_records),
        "records": [
            {
                "id": r.id,
                "tremor_index": r.tremor_index,
                "tremor_detected": r.tremor_detected,
                "dominant_frequency_hz": r.dominant_frequency_hz,
                "tremor_amplitude_mg": r.tremor_amplitude_mg,
                "parkinsonian_likelihood": r.parkinsonian_tremor_likelihood,
                "essential_tremor_likelihood": r.essential_tremor_likelihood,
                "physiological_tremor": r.physiological_tremor,
                "duration_seconds": r.duration_seconds,
                "sampling_rate_hz": r.sampling_rate_hz,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in tremor_records
        ]
    }


@router.get("/dashboard/{patient_id}")
def get_tremor_dashboard(
    patient_id: str,
    db: Session = Depends(get_db)
):
    """Get comprehensive tremor dashboard data"""
    # Get latest tremor
    tremor_service = TremorAnalysisService(db)
    latest = tremor_service.get_latest_tremor(patient_id)
    
    # Get recent history (last 7 days)
    from datetime import timedelta
    seven_days_ago = datetime.utcnow() - timedelta(days=7)
    
    recent_records = db.query(AccelerometerTremorData).filter(
        AccelerometerTremorData.patient_id == patient_id,
        AccelerometerTremorData.created_at >= seven_days_ago
    ).order_by(AccelerometerTremorData.created_at.asc()).all()
    
    # Calculate trends
    if len(recent_records) >= 2:
        tremor_indices = [r.tremor_index for r in recent_records if r.tremor_index is not None]
        avg_tremor_index = sum(tremor_indices) / len(tremor_indices) if tremor_indices else 0
        trend = "stable"
        
        if len(tremor_indices) >= 3:
            recent_avg = sum(tremor_indices[-3:]) / 3
            older_avg = sum(tremor_indices[:-3]) / len(tremor_indices[:-3]) if len(tremor_indices) > 3 else recent_avg
            
            if recent_avg > older_avg * 1.2:
                trend = "increasing"
            elif recent_avg < older_avg * 0.8:
                trend = "decreasing"
    else:
        avg_tremor_index = 0
        trend = "insufficient_data"
    
    return {
        "patient_id": patient_id,
        "latest_tremor": latest,
        "trend": {
            "status": trend,
            "avg_tremor_index_7days": avg_tremor_index,
            "recordings_count_7days": len(recent_records),
        },
        "history_7days": [
            {
                "tremor_index": r.tremor_index,
                "tremor_detected": r.tremor_detected,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in recent_records
        ]
    }
