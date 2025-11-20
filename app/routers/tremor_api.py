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
from app.dependencies import get_current_user, get_current_patient
from app.models import User

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
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_patient)
):
    """
    Upload accelerometer data for tremor analysis
    
    **Patient uploads phone accelerometer readings (5-10 seconds recommended)**
    
    **HIPAA-compliant PHI endpoint with validation and audit logging**
    """
    import numpy as np
    
    # HIPAA audit logging
    logger.info(f"[AUDIT] Tremor data upload initiated - Patient: {data.patient_id}, Samples: {len(data.timestamps)}")
    
    # Validation: Sample count bounds (100 min, 5000 max to prevent DoS)
    if len(data.timestamps) < 100:
        logger.warning(f"[VALIDATION] Patient {data.patient_id} - Insufficient samples: {len(data.timestamps)}")
        raise HTTPException(status_code=400, detail="Insufficient data. Record for at least 5 seconds (100 samples minimum).")
    
    if len(data.timestamps) > 5000:
        logger.warning(f"[VALIDATION] Patient {data.patient_id} - Excessive samples: {len(data.timestamps)}")
        raise HTTPException(status_code=400, detail="Data too large. Maximum 5000 samples (~50 seconds at 100Hz).")
    
    # Validation: Array length consistency
    if not (len(data.timestamps) == len(data.accel_x) == len(data.accel_y) == len(data.accel_z)):
        logger.warning(f"[VALIDATION] Patient {data.patient_id} - Array length mismatch")
        raise HTTPException(status_code=400, detail="Timestamps and acceleration arrays must have equal length")
    
    # Validation: Check for NaN values
    try:
        timestamps_arr = np.array(data.timestamps)
        x_arr = np.array(data.accel_x)
        y_arr = np.array(data.accel_y)
        z_arr = np.array(data.accel_z)
        
        if np.any(np.isnan(timestamps_arr)) or np.any(np.isnan(x_arr)) or np.any(np.isnan(y_arr)) or np.any(np.isnan(z_arr)):
            logger.warning(f"[VALIDATION] Patient {data.patient_id} - NaN values detected")
            raise HTTPException(status_code=400, detail="Data contains invalid values (NaN). Please re-record.")
        
        # Validation: Check for infinite values
        if np.any(np.isinf(timestamps_arr)) or np.any(np.isinf(x_arr)) or np.any(np.isinf(y_arr)) or np.any(np.isinf(z_arr)):
            logger.warning(f"[VALIDATION] Patient {data.patient_id} - Infinite values detected")
            raise HTTPException(status_code=400, detail="Data contains invalid values (infinity). Please re-record.")
        
        # Validation: Monotonic timestamps (strictly increasing)
        if not np.all(np.diff(timestamps_arr) > 0):
            logger.warning(f"[VALIDATION] Patient {data.patient_id} - Non-monotonic timestamps")
            raise HTTPException(status_code=400, detail="Timestamps must be strictly increasing. Please check device time sync.")
        
        # Validation: Reasonable acceleration bounds (-50 to +50 m/s² = ~5g)
        for axis, arr in [("x", x_arr), ("y", y_arr), ("z", z_arr)]:
            if np.any(np.abs(arr) > 50):
                logger.warning(f"[VALIDATION] Patient {data.patient_id} - Outlier in axis {axis}: {np.max(np.abs(arr))}")
                raise HTTPException(status_code=400, detail=f"Acceleration values out of range. Please keep phone steady during recording.")
        
        # Validation: Reasonable time intervals (1-500 Hz sampling rate)
        time_diffs = np.diff(timestamps_arr)
        avg_interval_ms = np.mean(time_diffs)
        if avg_interval_ms < 2 or avg_interval_ms > 1000:  # 2ms = 500Hz, 1000ms = 1Hz
            logger.warning(f"[VALIDATION] Patient {data.patient_id} - Invalid sampling rate: {1000/avg_interval_ms:.1f} Hz")
            raise HTTPException(status_code=400, detail=f"Sampling rate out of range ({1000/avg_interval_ms:.1f} Hz). Expected 1-500 Hz.")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[VALIDATION] Patient {data.patient_id} - Validation failed: {str(e)}")
        raise HTTPException(status_code=400, detail="Data validation failed. Please re-record.")
    
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
        
        # HIPAA audit log - success
        logger.info(f"[AUDIT] Tremor analysis completed - Patient: {data.patient_id}, ID: {result['tremor_data_id']}, Index: {result['tremor_index']:.1f}")
        
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
    
    except ValueError as e:
        # Clinical validation errors (patient-actionable)
        error_msg = str(e)
        logger.warning(f"[VALIDATION] Patient {data.patient_id} - {error_msg}")
        
        # Map to patient-friendly messages
        if "Invalid timestamps" in error_msg:
            raise HTTPException(status_code=400, detail="Recording timing issue. Please try again with steady recording.")
        elif "too short" in error_msg:
            raise HTTPException(status_code=400, detail="Recording too short. Please record for at least 5 seconds.")
        else:
            raise HTTPException(status_code=400, detail="Unable to analyze recording. Please ensure steady hand position and try again.")
    
    except Exception as e:
        # System errors (sanitized)
        logger.error(f"[ERROR] Tremor analysis system error for patient {data.patient_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="System error during analysis. Our team has been notified. Please try again later.")


@router.get("/latest/{patient_id}")
def get_latest_tremor(
    patient_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get patient's most recent tremor analysis
    **HIPAA-compliant endpoint with authentication and audit logging**
    """
    # HIPAA audit log
    logger.info(f"[AUDIT] Tremor latest retrieval - User: {current_user.id}, Patient: {patient_id}")
    
    # Authorization: patients can only access their own data, doctors can access any
    user_role = str(current_user.role) if current_user.role else ""
    if user_role == "patient" and current_user.id != patient_id:
        logger.warning(f"[AUTH] Patient {current_user.id} attempted to access tremor data for {patient_id}")
        raise HTTPException(status_code=403, detail="Access denied. You can only view your own tremor data.")
    
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
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get patient's tremor history
    **HIPAA-compliant endpoint with authentication and audit logging**
    """
    # HIPAA audit log
    logger.info(f"[AUDIT] Tremor history retrieval - User: {current_user.id}, Patient: {patient_id}, Limit: {limit}")
    
    # Authorization: patients can only access their own data, doctors can access any
    user_role = str(current_user.role) if current_user.role else ""
    if user_role == "patient" and current_user.id != patient_id:
        logger.warning(f"[AUTH] Patient {current_user.id} attempted to access tremor history for {patient_id}")
        raise HTTPException(status_code=403, detail="Access denied. You can only view your own tremor history.")
    
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
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get comprehensive tremor dashboard data
    **HIPAA-compliant endpoint with authentication and audit logging**
    """
    # HIPAA audit log
    logger.info(f"[AUDIT] Tremor dashboard retrieval - User: {current_user.id}, Patient: {patient_id}")
    
    # Authorization: patients can only access their own data, doctors can access any
    user_role = str(current_user.role) if current_user.role else ""
    if user_role == "patient" and current_user.id != patient_id:
        logger.warning(f"[AUTH] Patient {current_user.id} attempted to access tremor dashboard for {patient_id}")
        raise HTTPException(status_code=403, detail="Access denied. You can only view your own tremor dashboard.")
    
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
