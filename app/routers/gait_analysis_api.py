"""
Gait Analysis API Endpoints
Upload video, analyze gait patterns, retrieve metrics
"""

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import logging
import os
import tempfile

from app.database import get_db
from app.models.gait_analysis_models import (
    GaitSession, GaitMetrics, GaitPattern, GaitBaseline
)
from app.services.gait_analysis_service import GaitAnalysisService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/gait-analysis", tags=["Gait Analysis"])


@router.post("/upload")
async def upload_gait_video(
    patient_id: str,
    video: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    db: Session = Depends(get_db)
):
    """
    Upload gait analysis video and start processing
    
    **Patient uploads walking video (10-30 seconds recommended)**
    """
    logger.info(f"Uploading gait video for patient {patient_id}")
    
    # Validate file type
    if not video.content_type or not video.content_type.startswith('video/'):
        raise HTTPException(status_code=400, detail="File must be a video")
    
    # Create GaitSession record
    gait_session = GaitSession(
        patient_id=patient_id,
        processing_status="pending",
        uploaded_by=patient_id,
    )
    db.add(gait_session)
    db.commit()
    db.refresh(gait_session)
    
    # Save video temporarily
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    try:
        content = await video.read()
        temp_file.write(content)
        temp_file.close()
        
        # Get video metadata
        import cv2
        cap = cv2.VideoCapture(temp_file.name)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0
        cap.release()
        
        # Update session with metadata
        gait_session.duration_seconds = duration
        gait_session.fps = fps
        gait_session.total_frames = total_frames
        gait_session.resolution = f"{width}x{height}"
        db.commit()
        
        # Process video in background
        background_tasks.add_task(
            process_gait_video_background,
            session_id=gait_session.id,
            patient_id=patient_id,
            video_path=temp_file.name
        )
        
        return {
            "session_id": gait_session.id,
            "status": "processing",
            "message": "Video uploaded successfully. Analysis in progress.",
            "estimated_time_seconds": int(duration * 2),  # ~2x video length
        }
    
    except Exception as e:
        # HIPAA-compliant error handling: internal details in logs, sanitized message in DB and HTTP
        logger.error(f"[ERROR] Error uploading gait video for patient {patient_id}: {str(e)}", exc_info=True)
        gait_session.processing_status = "failed"
        gait_session.error_message = "Upload failed. Please try again or contact support."  # Sanitized
        db.commit()
        raise HTTPException(status_code=500, detail="Error processing video upload. Please try again later.")


def process_gait_video_background(session_id: int, patient_id: str, video_path: str):
    """Background task to analyze gait video"""
    from app.database import SessionLocal
    db = SessionLocal()
    
    try:
        logger.info(f"Starting gait analysis for session {session_id}")
        
        # Update status
        session = db.query(GaitSession).filter(GaitSession.id == session_id).first()
        if not session:
            raise ValueError(f"GaitSession {session_id} not found")
        
        session.processing_status = "processing"
        session.processing_started_at = datetime.utcnow()
        db.commit()
        
        # Analyze video
        try:
            gait_service = GaitAnalysisService(db)
        except ImportError as e:
            logger.error(f"MediaPipe not available: {str(e)}")
            session.processing_status = "failed"
            session.error_message = "MediaPipe library not installed. Install with: pip install mediapipe"
            session.processing_completed_at = datetime.utcnow()
            db.commit()
            return
        
        try:
            result = gait_service.analyze_video(
                video_path=video_path,
                patient_id=patient_id,
                session_id=session_id
            )
            
            # Update session with results
            session.processing_status = "completed"
            session.processing_completed_at = datetime.utcnow()
            session.total_strides_detected = result['total_strides']
            session.walking_detected = result['walking_detected']
            session.gait_abnormality_detected = result['gait_abnormality_detected']
            session.gait_abnormality_score = result['summary'].get('fall_risk', 0)
            session.overall_quality_score = result.get('overall_quality_score', 0.0)
            db.commit()
            
            logger.info(f"Gait analysis completed for session {session_id}")
            
        except ValueError as e:
            # Clinical validation errors (expected edge cases)
            error_msg = str(e).lower()  # Normalize for matching
            
            # Map to patient-friendly messages (NO internal details leaked)
            if "low pose detection" in error_msg or "detection rate" in error_msg:
                patient_message = "Video quality issue: Unable to detect your full body clearly. Please ensure you're fully visible in the frame and try again."
                error_code = "LOW_DETECTION"
            elif "no walking" in error_msg or "walking not detected" in error_msg:
                patient_message = "No walking motion detected. Please record yourself walking naturally for 10-30 seconds and try again."
                error_code = "NO_WALKING"
            elif "failed to open" in error_msg or "corrupt" in error_msg or "invalid video" in error_msg:
                patient_message = "Video file error. Please try recording again."
                error_code = "VIDEO_CORRUPT"
            elif "too short" in error_msg or "insufficient" in error_msg:
                patient_message = "Video too short. Please record at least 10 seconds of walking."
                error_code = "VIDEO_TOO_SHORT"
            elif "lighting" in error_msg or "too dark" in error_msg:
                patient_message = "Video lighting issue. Please record in better lighting conditions."
                error_code = "POOR_LIGHTING"
            else:
                # Generic fallback - NO raw exception text
                patient_message = "Unable to analyze video. Please ensure good lighting, full body visible, and steady camera position."
                error_code = "VALIDATION_ERROR"
            
            # HIPAA-compliant logging: internal details in logs, sanitized message in DB
            logger.warning(f"[VALIDATION] Session {session_id} - Code: {error_code}, Internal: {str(e)}")
            
            session.processing_status = "failed"
            session.error_message = patient_message  # Sanitized message only
            session.processing_completed_at = datetime.utcnow()
            db.commit()
        
        except Exception as e:
            # System errors (unexpected failures)
            # HIPAA-compliant: Internal details in logs only, sanitized message in DB
            logger.error(f"[ERROR] System error in gait analysis for session {session_id}: {str(e)}", exc_info=True)
            session.processing_status = "failed"
            session.error_message = "System error during analysis. Our team has been notified. Please try again later."
            session.processing_completed_at = datetime.utcnow()
            db.commit()
        
    except Exception as e:
        # Outer exception handler (should rarely hit this)
        # HIPAA-compliant: Internal stack trace in logs, sanitized message in DB
        logger.error(f"[FATAL] Fatal error in background gait analysis for session {session_id}: {str(e)}", exc_info=True)
        try:
            session = db.query(GaitSession).filter(GaitSession.id == session_id).first()
            if session:
                session.processing_status = "failed"
                session.error_message = "Unexpected error. Please contact support."  # Sanitized
                session.processing_completed_at = datetime.utcnow()
                db.commit()
        except Exception as db_error:
            logger.error(f"[FATAL] Failed to update session status: {str(db_error)}")
            pass
    
    finally:
        # Cleanup temp file
        try:
            os.unlink(video_path)
        except:
            pass
        db.close()


@router.get("/sessions/{patient_id}")
def get_gait_sessions(
    patient_id: str,
    limit: int = 10,
    db: Session = Depends(get_db)
):
    """Get gait analysis sessions for a patient"""
    sessions = db.query(GaitSession).filter(
        GaitSession.patient_id == patient_id
    ).order_by(GaitSession.created_at.desc()).limit(limit).all()
    
    return {
        "patient_id": patient_id,
        "total_sessions": len(sessions),
        "sessions": [
            {
                "session_id": s.id,
                "created_at": s.created_at.isoformat() if s.created_at else None,
                "status": s.processing_status,
                "duration_seconds": s.duration_seconds,
                "total_strides": s.total_strides_detected,
                "walking_detected": s.walking_detected,
                "abnormality_detected": s.gait_abnormality_detected,
                "abnormality_score": s.gait_abnormality_score,
                "quality_score": s.overall_quality_score,
            }
            for s in sessions
        ]
    }


@router.get("/metrics/{session_id}")
def get_gait_metrics(
    session_id: int,
    db: Session = Depends(get_db)
):
    """Get detailed gait metrics for a session"""
    metrics = db.query(GaitMetrics).filter(
        GaitMetrics.session_id == session_id
    ).first()
    
    if not metrics:
        raise HTTPException(status_code=404, detail="Gait metrics not found")
    
    return {
        "session_id": session_id,
        "patient_id": metrics.patient_id,
        "created_at": metrics.created_at.isoformat() if metrics.created_at else None,
        
        # Temporal parameters
        "temporal": {
            "stride_time_avg_sec": metrics.stride_time_avg_sec,
            "stride_time_left_sec": metrics.stride_time_left_sec,
            "stride_time_right_sec": metrics.stride_time_right_sec,
            "cadence_steps_per_min": metrics.cadence_steps_per_min,
            "walking_speed_m_per_sec": metrics.walking_speed_m_per_sec,
            "double_support_time_sec": metrics.double_support_time_sec,
        },
        
        # Spatial parameters
        "spatial": {
            "stride_length_avg_cm": metrics.stride_length_avg_cm,
            "stride_length_left_cm": metrics.stride_length_left_cm,
            "stride_length_right_cm": metrics.stride_length_right_cm,
            "step_width_cm": metrics.step_width_cm,
        },
        
        # Joint angles
        "joint_angles": {
            "hip": {
                "left_flexion_deg": metrics.hip_flexion_angle_left_deg,
                "right_flexion_deg": metrics.hip_flexion_angle_right_deg,
                "left_rom_deg": metrics.hip_range_of_motion_left_deg,
                "right_rom_deg": metrics.hip_range_of_motion_right_deg,
            },
            "knee": {
                "left_flexion_deg": metrics.knee_flexion_angle_left_deg,
                "right_flexion_deg": metrics.knee_flexion_angle_right_deg,
                "left_rom_deg": metrics.knee_range_of_motion_left_deg,
                "right_rom_deg": metrics.knee_range_of_motion_right_deg,
            },
            "ankle": {
                "left_dorsiflexion_deg": metrics.ankle_dorsiflexion_angle_left_deg,
                "right_dorsiflexion_deg": metrics.ankle_dorsiflexion_angle_right_deg,
                "left_rom_deg": metrics.ankle_range_of_motion_left_deg,
                "right_rom_deg": metrics.ankle_range_of_motion_right_deg,
            },
        },
        
        # Symmetry & stability
        "symmetry_stability": {
            "temporal_symmetry": metrics.temporal_symmetry_index,
            "spatial_symmetry": metrics.spatial_symmetry_index,
            "overall_symmetry": metrics.overall_gait_symmetry_index,
            "trunk_sway_lateral_cm": metrics.trunk_sway_lateral_cm,
            "head_stability_score": metrics.head_stability_score,
            "balance_confidence_score": metrics.balance_confidence_score,
            "stride_time_variability_percent": metrics.stride_time_variability_percent,
        },
        
        # Activity classification
        "activity": {
            "primary_activity": metrics.primary_activity_detected,
            "walking_confidence": metrics.walking_confidence,
            "shuffling_confidence": metrics.shuffling_confidence,
            "limping_confidence": metrics.limping_confidence,
            "unsteady_confidence": metrics.unsteady_confidence,
        },
        
        # Clinical risks
        "clinical_risks": {
            "fall_risk_score": metrics.fall_risk_score,
            "parkinson_indicators": metrics.parkinson_gait_indicators,
            "neuropathy_indicators": metrics.neuropathy_indicators,
            "pain_indicators": metrics.pain_gait_indicators,
        },
        
        # Baseline comparison
        "baseline": {
            "has_baseline": metrics.has_baseline,
            "deviation_percent": metrics.deviation_from_baseline_percent,
            "deterioration_detected": metrics.significant_deterioration_detected,
        },
        
        # Metadata
        "metadata": {
            "analysis_method": metrics.analysis_method,
            "model_version": metrics.model_version,
            "frames_analyzed": metrics.frames_analyzed,
            "landmarks_detected_percent": metrics.landmarks_detected_percent,
        }
    }


@router.get("/patterns/{session_id}")
def get_gait_patterns(
    session_id: int,
    db: Session = Depends(get_db)
):
    """Get stride-by-stride gait patterns"""
    patterns = db.query(GaitPattern).filter(
        GaitPattern.session_id == session_id
    ).order_by(GaitPattern.stride_number).all()
    
    return {
        "session_id": session_id,
        "total_strides": len(patterns),
        "strides": [
            {
                "stride_number": p.stride_number,
                "side": p.side,
                "duration_sec": p.stride_duration_sec,
                "length_cm": p.stride_length_cm,
                "width_cm": p.step_width_cm,
                "hip_angle_heel_strike": p.hip_angle_at_heel_strike_deg,
                "knee_angle_heel_strike": p.knee_angle_at_heel_strike_deg,
                "ankle_angle_heel_strike": p.ankle_angle_at_heel_strike_deg,
                "confidence": p.detection_confidence,
            }
            for p in patterns
        ]
    }


@router.get("/baseline/{patient_id}")
def get_patient_baseline(
    patient_id: str,
    db: Session = Depends(get_db)
):
    """Get patient's gait baseline"""
    baseline = db.query(GaitBaseline).filter(
        GaitBaseline.patient_id == patient_id
    ).first()
    
    if not baseline:
        return {
            "patient_id": patient_id,
            "has_baseline": False,
            "message": "No baseline established yet. Complete 3+ gait analyses to establish baseline."
        }
    
    return {
        "patient_id": patient_id,
        "has_baseline": True,
        "baseline": {
            "stride_time_sec": baseline.baseline_stride_time_sec,
            "cadence_steps_per_min": baseline.baseline_cadence_steps_per_min,
            "walking_speed_m_per_sec": baseline.baseline_walking_speed_m_per_sec,
            "stride_length_cm": baseline.baseline_stride_length_cm,
            "step_width_cm": baseline.baseline_step_width_cm,
            "symmetry_index": baseline.baseline_symmetry_index,
            "fall_risk_score": baseline.baseline_fall_risk_score,
        },
        "metadata": {
            "established_date": baseline.baseline_established_date.isoformat() if baseline.baseline_established_date else None,
            "last_updated": baseline.last_updated.isoformat() if baseline.last_updated else None,
            "sessions_used": baseline.sessions_used_for_baseline,
            "quality_score": baseline.baseline_quality_score,
        }
    }


@router.get("/dashboard/{patient_id}")
def get_gait_dashboard(
    patient_id: str,
    days: int = 30,
    db: Session = Depends(get_db)
):
    """
    Get comprehensive gait analysis dashboard
    Shows trends, recent metrics, alerts
    """
    cutoff_date = datetime.utcnow() - timedelta(days=days)
    
    # Get recent sessions
    sessions = db.query(GaitSession).filter(
        GaitSession.patient_id == patient_id,
        GaitSession.created_at >= cutoff_date,
        GaitSession.processing_status == "completed"
    ).order_by(GaitSession.created_at.desc()).all()
    
    # Get latest metrics
    latest_session = sessions[0] if sessions else None
    latest_metrics = None
    if latest_session:
        latest_metrics = db.query(GaitMetrics).filter(
            GaitMetrics.session_id == latest_session.id
        ).first()
    
    # Get baseline
    baseline = db.query(GaitBaseline).filter(
        GaitBaseline.patient_id == patient_id
    ).first()
    
    # Calculate trends
    trends = {}
    if len(sessions) >= 2:
        all_metrics = db.query(GaitMetrics).filter(
            GaitMetrics.patient_id == patient_id,
            GaitMetrics.created_at >= cutoff_date
        ).order_by(GaitMetrics.created_at).all()
        
        if len(all_metrics) >= 2:
            # Trend analysis (first vs last)
            first = all_metrics[0]
            last = all_metrics[-1]
            
            if first.cadence_steps_per_min and last.cadence_steps_per_min:
                trends['cadence_change_percent'] = ((last.cadence_steps_per_min - first.cadence_steps_per_min) / first.cadence_steps_per_min) * 100
            
            if first.stride_length_avg_cm and last.stride_length_avg_cm:
                trends['stride_length_change_percent'] = ((last.stride_length_avg_cm - first.stride_length_avg_cm) / first.stride_length_avg_cm) * 100
            
            if first.overall_gait_symmetry_index and last.overall_gait_symmetry_index:
                trends['symmetry_change_percent'] = ((last.overall_gait_symmetry_index - first.overall_gait_symmetry_index) / first.overall_gait_symmetry_index) * 100
    
    # Alerts
    alerts = []
    if latest_metrics:
        if latest_metrics.fall_risk_score and latest_metrics.fall_risk_score > 50:
            alerts.append({
                "type": "fall_risk",
                "severity": "high" if latest_metrics.fall_risk_score > 70 else "medium",
                "message": f"Elevated fall risk detected ({latest_metrics.fall_risk_score:.0f}/100)",
            })
        
        if latest_metrics.significant_deterioration_detected:
            alerts.append({
                "type": "deterioration",
                "severity": "high",
                "message": f"Significant gait deterioration from baseline ({latest_metrics.deviation_from_baseline_percent:.1f}% deviation)",
            })
        
        if latest_metrics.shuffling_confidence and latest_metrics.shuffling_confidence > 0.5:
            alerts.append({
                "type": "shuffling_gait",
                "severity": "medium",
                "message": "Shuffling gait pattern detected (potential Parkinson's indicator)",
            })
    
    return {
        "patient_id": patient_id,
        "period_days": days,
        "total_sessions": len(sessions),
        "has_baseline": baseline is not None,
        
        "latest_metrics": {
            "session_id": latest_session.id if latest_session else None,
            "date": latest_session.created_at.isoformat() if latest_session and latest_session.created_at else None,
            "cadence": latest_metrics.cadence_steps_per_min if latest_metrics else None,
            "walking_speed": latest_metrics.walking_speed_m_per_sec if latest_metrics else None,
            "stride_length": latest_metrics.stride_length_avg_cm if latest_metrics else None,
            "symmetry": latest_metrics.overall_gait_symmetry_index if latest_metrics else None,
            "fall_risk": latest_metrics.fall_risk_score if latest_metrics else None,
            "primary_activity": latest_metrics.primary_activity_detected if latest_metrics else None,
        } if latest_metrics else None,
        
        "trends": trends,
        "alerts": alerts,
        
        "recent_sessions": [
            {
                "session_id": s.id,
                "date": s.created_at.isoformat() if s.created_at else None,
                "strides": s.total_strides_detected,
                "abnormality_score": s.gait_abnormality_score,
            }
            for s in sessions[:5]  # Last 5 sessions
        ]
    }
