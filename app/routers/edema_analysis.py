"""
Edema Analysis API Endpoints
DeepLab V3+ semantic segmentation for swelling/edema monitoring
"""

from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime

from app.database import get_db
from app.models.video_ai_models import MediaSession, EdemaSegmentationMetrics
from app.services.edema_segmentation_service import EdemaSegmentationService
from app.services.ai_engine_manager import get_ai_engine_manager

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/edema", tags=["Edema Analysis"])


@router.post("/analyze-session/{session_id}")
async def analyze_session_for_edema(
    session_id: int,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Analyze existing video session for edema/swelling
    
    Uses DeepLab V3+ to segment body regions and detect swelling.
    Compares to patient baseline if available.
    
    Args:
        session_id: MediaSession ID to analyze
    
    Returns:
        Edema analysis results with regional breakdown
    """
    try:
        # Get session
        session = db.query(MediaSession).filter(MediaSession.id == session_id).first()
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        if not session.video_s3_uri:
            raise HTTPException(status_code=400, detail="No video file in session")
        
        # Initialize edema service
        edema_service = EdemaSegmentationService()
        
        if not edema_service.model:
            raise HTTPException(
                status_code=503,
                detail="DeepLab model unavailable - check TensorFlow/TF-Hub installation"
            )
        
        # TODO: Download video from S3 and run segmentation
        # For now, return placeholder response
        
        return {
            "session_id": session_id,
            "patient_id": session.patient_id,
            "status": "pending_implementation",
            "message": "Edema segmentation service initialized. Full S3 video processing to be implemented.",
            "model_info": {
                "model_type": "deeplab_v3_plus",
                "model_available": edema_service.model is not None,
                "uses_finetuned": edema_service.use_finetuned
            }
        }
        
    except Exception as e:
        logger.error(f"Edema analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics/{patient_id}")
async def get_patient_edema_metrics(
    patient_id: str,
    limit: int = 10,
    db: Session = Depends(get_db)
) -> List[Dict[str, Any]]:
    """
    Get patient's edema segmentation history
    
    Args:
        patient_id: Patient identifier
        limit: Number of recent results to return
    
    Returns:
        List of edema analysis results
    """
    try:
        metrics = db.query(EdemaSegmentationMetrics)\
            .filter(EdemaSegmentationMetrics.patient_id == patient_id)\
            .order_by(EdemaSegmentationMetrics.analyzed_at.desc())\
            .limit(limit)\
            .all()
        
        results = []
        for metric in metrics:
            results.append({
                "id": metric.id,
                "session_id": metric.session_id,
                "analyzed_at": metric.analyzed_at.isoformat() if metric.analyzed_at else None,
                "swelling_detected": metric.swelling_detected,
                "swelling_severity": metric.swelling_severity,
                "overall_expansion_percent": metric.overall_expansion_percent,
                "regions_affected": metric.swelling_regions_count,
                "regional_analysis": {
                    "face_upper_body": {
                        "swelling_detected": metric.face_upper_body_swelling_detected,
                        "expansion_percent": metric.face_upper_body_expansion_percent
                    },
                    "legs_feet": {
                        "swelling_detected": metric.legs_feet_swelling_detected,
                        "expansion_percent": metric.legs_feet_expansion_percent
                    },
                    "asymmetry_detected": metric.asymmetry_detected,
                    "asymmetry_difference": metric.asymmetry_difference_percent
                },
                "model_version": metric.model_version,
                "processing_time_ms": metric.processing_time_ms
            })
        
        return results
        
    except Exception as e:
        logger.error(f"Failed to get edema metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/baseline/{patient_id}")
async def create_edema_baseline(
    patient_id: str,
    session_id: int,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Create patient's edema baseline from a session
    
    This baseline will be used for future comparisons to detect swelling.
    
    Args:
        patient_id: Patient identifier
        session_id: Session to use as baseline
    
    Returns:
        Baseline creation status
    """
    try:
        # Get edema metrics from session
        metric = db.query(EdemaSegmentationMetrics)\
            .filter(
                EdemaSegmentationMetrics.session_id == session_id,
                EdemaSegmentationMetrics.patient_id == patient_id
            )\
            .first()
        
        if not metric:
            raise HTTPException(
                status_code=404,
                detail="No edema metrics found for this session"
            )
        
        # TODO: Store baseline in patient baseline table
        # For now, return success
        
        return {
            "patient_id": patient_id,
            "baseline_session_id": session_id,
            "baseline_created_at": datetime.utcnow().isoformat(),
            "total_body_area_px": metric.total_body_area_px,
            "regions_captured": {
                "face_upper_body": metric.face_upper_body_area_px is not None,
                "torso_hands": metric.torso_hands_area_px is not None,
                "legs_feet": metric.legs_feet_area_px is not None
            },
            "status": "baseline_created"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create baseline: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/trends/{patient_id}")
async def get_edema_trends(
    patient_id: str,
    days: int = 30,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Get patient's edema trends over time
    
    Shows progression/regression of swelling in different body regions.
    
    Args:
        patient_id: Patient identifier
        days: Number of days to analyze
    
    Returns:
        Trend analysis with time-series data
    """
    try:
        # Get metrics from last N days
        from datetime import timedelta
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        metrics = db.query(EdemaSegmentationMetrics)\
            .filter(
                EdemaSegmentationMetrics.patient_id == patient_id,
                EdemaSegmentationMetrics.analyzed_at >= cutoff_date
            )\
            .order_by(EdemaSegmentationMetrics.analyzed_at.asc())\
            .all()
        
        if not metrics:
            return {
                "patient_id": patient_id,
                "days_analyzed": days,
                "data_points": 0,
                "message": "No edema data in selected time range"
            }
        
        # Build time-series for each region
        timestamps = [m.analyzed_at.isoformat() for m in metrics]
        
        trends = {
            "patient_id": patient_id,
            "days_analyzed": days,
            "data_points": len(metrics),
            "timestamps": timestamps,
            "overall_swelling": [m.swelling_detected for m in metrics],
            "overall_expansion": [m.overall_expansion_percent for m in metrics if m.overall_expansion_percent is not None],
            "regional_trends": {
                "face_upper_body": [m.face_upper_body_expansion_percent for m in metrics if m.face_upper_body_expansion_percent is not None],
                "legs_feet": [m.legs_feet_expansion_percent for m in metrics if m.legs_feet_expansion_percent is not None],
                "torso_hands": [m.torso_hands_expansion_percent for m in metrics if m.torso_hands_expansion_percent is not None]
            },
            "asymmetry_timeline": [m.asymmetry_detected for m in metrics],
            "trend_summary": {
                "worsening": len([m for m in metrics if m.swelling_detected]),
                "stable": len([m for m in metrics if not m.swelling_detected]),
                "total_sessions": len(metrics)
            }
        }
        
        return trends
        
    except Exception as e:
        logger.error(f"Failed to get edema trends: {e}")
        raise HTTPException(status_code=500, detail=str(e))
