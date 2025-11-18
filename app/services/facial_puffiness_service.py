"""
Facial Puffiness Service - Persist and manage FPS metrics
Handles baseline calculation, temporal tracking, and personalized thresholds
"""

import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import and_, desc
import logging

from app.models import FacialPuffinessBaseline, FacialPuffinessMetric

logger = logging.getLogger(__name__)


class FacialPuffinessService:
    """Facial Puffiness Score persistence and temporal analytics"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def ingest_fps_metrics(
        self,
        patient_id: str,
        session_id: str,
        fps_metrics: Dict[str, Any],
        frames_analyzed: int,
        detection_confidence: float,
        timestamp: Optional[datetime] = None
    ) -> FacialPuffinessMetric:
        """
        Ingest FPS metrics from video analysis session
        
        Args:
            patient_id: Patient ID
            session_id: Video examination session ID
            fps_metrics: Dict containing all FPS scores and raw measurements
            frames_analyzed: Number of frames with face detected
            detection_confidence: 0-1 confidence score
            timestamp: Optional timestamp (defaults to now)
        
        Returns:
            Persisted FacialPuffinessMetric record
        """
        if timestamp is None:
            timestamp = datetime.utcnow()
        
        # Extract metrics from VideoAIEngine output
        composite_fps = fps_metrics.get('facial_puffiness_score', 0.0)
        fps_periorbital = fps_metrics.get('fps_periorbital', 0.0)
        fps_cheek = fps_metrics.get('fps_cheek', 0.0)
        fps_jawline = fps_metrics.get('fps_jawline', 0.0)
        fps_forehead = fps_metrics.get('fps_forehead', 0.0)
        fps_overall_contour = fps_metrics.get('fps_overall_contour', 0.0)
        facial_asymmetry_score = fps_metrics.get('facial_asymmetry_score', 0.0)
        fps_risk_level = fps_metrics.get('facial_puffiness_risk', 'unknown')
        
        # Raw measurements
        raw_eye_area = fps_metrics.get('raw_eye_area', 0.0)
        raw_cheek_width = fps_metrics.get('raw_cheek_width', 0.0)
        raw_cheek_projection = fps_metrics.get('raw_cheek_projection', 0.0)
        raw_jawline_width = fps_metrics.get('raw_jawline_width', 0.0)
        raw_forehead_width = fps_metrics.get('raw_forehead_width', 0.0)
        raw_face_perimeter = fps_metrics.get('raw_face_perimeter', 0.0)
        
        # Update baseline if detection confidence is high and FPS is low (healthy state)
        if detection_confidence > 0.7 and composite_fps < 10.0:
            self._update_baseline(
                patient_id=patient_id,
                raw_eye_area=raw_eye_area,
                raw_cheek_width=raw_cheek_width,
                raw_cheek_projection=raw_cheek_projection,
                raw_jawline_width=raw_jawline_width,
                raw_forehead_width=raw_forehead_width,
                raw_face_perimeter=raw_face_perimeter,
                confidence=detection_confidence
            )
        
        # Determine if asymmetry is significant (>20% difference)
        asymmetry_detected = facial_asymmetry_score > 20.0
        
        # Create FPS metric record
        metric = FacialPuffinessMetric(
            patient_id=patient_id,
            session_id=session_id,
            recorded_at=timestamp,
            facial_puffiness_score=composite_fps,
            fps_risk_level=fps_risk_level,
            fps_periorbital=fps_periorbital,
            fps_cheek=fps_cheek,
            fps_jawline=fps_jawline,
            fps_forehead=fps_forehead,
            fps_overall_contour=fps_overall_contour,
            facial_asymmetry_score=facial_asymmetry_score,
            asymmetry_detected=asymmetry_detected,
            raw_eye_area=raw_eye_area,
            raw_cheek_width=raw_cheek_width,
            raw_cheek_projection=raw_cheek_projection,
            raw_jawline_width=raw_jawline_width,
            raw_forehead_width=raw_forehead_width,
            raw_face_perimeter=raw_face_perimeter,
            detection_confidence=detection_confidence,
            frames_analyzed=frames_analyzed,
            metrics_metadata={
                'full_fps_metrics': fps_metrics,
                'timestamp_iso': timestamp.isoformat()
            }
        )
        
        self.db.add(metric)
        self.db.commit()
        self.db.refresh(metric)
        
        logger.info(f"Facial Puffiness metrics ingested for {patient_id}: FPS={composite_fps:.1f}, Risk={fps_risk_level}")
        
        return metric
    
    def _get_or_create_baseline(self, patient_id: str) -> Optional[FacialPuffinessBaseline]:
        """Get existing baseline or return None"""
        return self.db.query(FacialPuffinessBaseline).filter(
            FacialPuffinessBaseline.patient_id == patient_id
        ).first()
    
    def _update_baseline(
        self,
        patient_id: str,
        raw_eye_area: float,
        raw_cheek_width: float,
        raw_cheek_projection: float,
        raw_jawline_width: float,
        raw_forehead_width: float,
        raw_face_perimeter: float,
        confidence: float
    ):
        """
        Update patient baseline using exponential moving average
        Only updates when in healthy state (low FPS, high confidence)
        """
        baseline = self.db.query(FacialPuffinessBaseline).filter(
            FacialPuffinessBaseline.patient_id == patient_id
        ).first()
        
        if not baseline:
            # Initialize baseline with current measurements
            baseline = FacialPuffinessBaseline(
                patient_id=patient_id,
                baseline_eye_area=raw_eye_area,
                baseline_cheek_width=raw_cheek_width,
                baseline_cheek_projection=raw_cheek_projection,
                baseline_jawline_width=raw_jawline_width,
                baseline_forehead_width=raw_forehead_width,
                baseline_face_perimeter=raw_face_perimeter,
                sample_size=1,
                confidence=confidence,
                source="auto",
                last_calibration_at=datetime.utcnow()
            )
            self.db.add(baseline)
            logger.info(f"Initialized facial puffiness baseline for {patient_id}")
        else:
            # Exponential moving average update (α = 0.1 for slower adaptation)
            alpha = 0.1 if baseline.sample_size >= 5 else 1.0 / (baseline.sample_size + 1)
            
            baseline.baseline_eye_area = baseline.baseline_eye_area * (1 - alpha) + raw_eye_area * alpha
            baseline.baseline_cheek_width = baseline.baseline_cheek_width * (1 - alpha) + raw_cheek_width * alpha
            baseline.baseline_cheek_projection = baseline.baseline_cheek_projection * (1 - alpha) + raw_cheek_projection * alpha
            baseline.baseline_jawline_width = baseline.baseline_jawline_width * (1 - alpha) + raw_jawline_width * alpha
            baseline.baseline_forehead_width = baseline.baseline_forehead_width * (1 - alpha) + raw_forehead_width * alpha
            baseline.baseline_face_perimeter = baseline.baseline_face_perimeter * (1 - alpha) + raw_face_perimeter * alpha
            
            baseline.sample_size += 1
            baseline.confidence = min(0.95, baseline.confidence + 0.05)
            baseline.last_calibration_at = datetime.utcnow()
            
            logger.info(f"Updated facial puffiness baseline for {patient_id} (sample_size={baseline.sample_size})")
        
        self.db.commit()
    
    def get_patient_baseline(self, patient_id: str) -> Optional[Dict[str, float]]:
        """
        Get patient baseline as dict for VideoAIEngine
        Returns format expected by _compute_aggregate_metrics
        """
        baseline = self._get_or_create_baseline(patient_id)
        
        if not baseline:
            return None
        
        return {
            'baseline_eye_area': baseline.baseline_eye_area,
            'baseline_cheek_width': baseline.baseline_cheek_width,
            'baseline_cheek_projection': baseline.baseline_cheek_projection,
            'baseline_jawline_width': baseline.baseline_jawline_width,
            'baseline_forehead_width': baseline.baseline_forehead_width,
            'baseline_face_perimeter': baseline.baseline_face_perimeter
        }
    
    def get_recent_metrics(
        self,
        patient_id: str,
        days: int = 7,
        limit: Optional[int] = None
    ) -> List[FacialPuffinessMetric]:
        """Get recent FPS metrics for patient"""
        cutoff = datetime.utcnow() - timedelta(days=days)
        
        query = self.db.query(FacialPuffinessMetric).filter(
            and_(
                FacialPuffinessMetric.patient_id == patient_id,
                FacialPuffinessMetric.recorded_at >= cutoff
            )
        ).order_by(desc(FacialPuffinessMetric.recorded_at))
        
        if limit:
            query = query.limit(limit)
        
        return query.all()
    
    def get_patient_summary(self, patient_id: str) -> Dict[str, Any]:
        """Get comprehensive FPS summary for patient"""
        baseline = self._get_or_create_baseline(patient_id)
        
        latest = self.db.query(FacialPuffinessMetric).filter(
            FacialPuffinessMetric.patient_id == patient_id
        ).order_by(desc(FacialPuffinessMetric.recorded_at)).first()
        
        cutoff = datetime.utcnow() - timedelta(days=7)
        recent_metrics = self.db.query(FacialPuffinessMetric).filter(
            and_(
                FacialPuffinessMetric.patient_id == patient_id,
                FacialPuffinessMetric.recorded_at >= cutoff
            )
        ).order_by(FacialPuffinessMetric.recorded_at).all()
        
        # Compute 7-day trend
        trend = 'stable'
        if len(recent_metrics) >= 3:
            fps_values = [m.facial_puffiness_score for m in recent_metrics if m.facial_puffiness_score]
            if len(fps_values) >= 3:
                # Simple linear trend
                x = np.arange(len(fps_values))
                slope = np.polyfit(x, fps_values, 1)[0] if len(fps_values) > 1 else 0
                
                if slope > 2.0:  # Increasing >2% per day
                    trend = 'increasing'
                elif slope < -2.0:  # Decreasing >2% per day
                    trend = 'decreasing'
        
        return {
            'baseline': {
                'eye_area': baseline.baseline_eye_area if baseline else None,
                'cheek_width': baseline.baseline_cheek_width if baseline else None,
                'sample_size': baseline.sample_size if baseline else 0,
                'confidence': baseline.confidence if baseline else 0.0,
                'last_calibration': baseline.last_calibration_at.isoformat() if baseline and baseline.last_calibration_at else None
            },
            'latest': {
                'facial_puffiness_score': latest.facial_puffiness_score if latest else None,
                'recorded_at': latest.recorded_at.isoformat() if latest else None,
                'fps_risk_level': latest.fps_risk_level if latest else None,
                'fps_periorbital': latest.fps_periorbital if latest else None,
                'fps_cheek': latest.fps_cheek if latest else None,
                'facial_asymmetry_score': latest.facial_asymmetry_score if latest else None,
                'asymmetry_detected': latest.asymmetry_detected if latest else False
            },
            'recent_count': len(recent_metrics),
            'trend': trend,
            'avg_fps_7day': float(np.mean([m.facial_puffiness_score for m in recent_metrics if m.facial_puffiness_score])) if recent_metrics else None
        }
    
    def detect_temporal_patterns(self, patient_id: str) -> Dict[str, Any]:
        """
        Detect temporal patterns (morning vs evening puffiness)
        Useful for kidney disease patients
        """
        cutoff = datetime.utcnow() - timedelta(days=7)
        metrics = self.db.query(FacialPuffinessMetric).filter(
            and_(
                FacialPuffinessMetric.patient_id == patient_id,
                FacialPuffinessMetric.recorded_at >= cutoff
            )
        ).order_by(FacialPuffinessMetric.recorded_at).all()
        
        if len(metrics) < 3:
            return {
                'pattern_detected': False,
                'pattern_type': 'insufficient_data'
            }
        
        # Classify by time of day
        morning_fps = []  # 4am - 11am
        afternoon_fps = []  # 11am - 5pm
        evening_fps = []  # 5pm - 10pm
        
        for m in metrics:
            hour = m.recorded_at.hour
            if 4 <= hour < 11:
                morning_fps.append(m.facial_puffiness_score)
            elif 11 <= hour < 17:
                afternoon_fps.append(m.facial_puffiness_score)
            elif 17 <= hour < 22:
                evening_fps.append(m.facial_puffiness_score)
        
        # Detect morning predominance (kidney disease pattern)
        morning_avg = np.mean(morning_fps) if morning_fps else 0
        afternoon_avg = np.mean(afternoon_fps) if afternoon_fps else 0
        evening_avg = np.mean(evening_fps) if evening_fps else 0
        
        pattern_detected = False
        pattern_type = 'none'
        
        if morning_avg > 0 and afternoon_avg > 0:
            # Morning puffiness pattern (resolves during day)
            if morning_avg > afternoon_avg * 1.5:
                pattern_detected = True
                pattern_type = 'morning_predominant'
        
        if evening_avg > 0 and morning_avg > 0:
            # Evening puffiness pattern (accumulates during day)
            if evening_avg > morning_avg * 1.5:
                pattern_detected = True
                pattern_type = 'evening_predominant'
        
        return {
            'pattern_detected': pattern_detected,
            'pattern_type': pattern_type,
            'morning_avg_fps': float(morning_avg) if morning_avg > 0 else None,
            'afternoon_avg_fps': float(afternoon_avg) if afternoon_avg > 0 else None,
            'evening_avg_fps': float(evening_avg) if evening_avg > 0 else None,
            'sample_counts': {
                'morning': len(morning_fps),
                'afternoon': len(afternoon_fps),
                'evening': len(evening_fps)
            }
        }
