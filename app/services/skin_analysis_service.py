"""
Skin Analysis Service - Manages persistence and temporal analytics for skin metrics
Tracks LAB color space perfusion, capillary refill, nailbed health, and clinical color changes
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import func, desc
import statistics

from app.models.video_ai_models import SkinAnalysisMetric, SkinBaseline

logger = logging.getLogger(__name__)


class SkinAnalysisService:
    """
    Service for persisting and analyzing skin metrics with disease-specific personalization
    Tracks temporal trends, baseline deviations, and risk scoring
    """
    
    def __init__(self, db: Session):
        self.db = db
    
    def ingest_skin_metrics(
        self,
        patient_id: str,
        session_id: str,
        skin_metrics: Dict,
        timestamp: Optional[datetime] = None
    ) -> SkinAnalysisMetric:
        """
        Persist comprehensive skin analysis metrics to database
        
        Args:
            patient_id: Patient identifier
            session_id: Video examination session ID
            skin_metrics: Dict containing all LAB color metrics, capillary refill, nailbed analysis
            timestamp: Recording timestamp (defaults to now)
            
        Returns:
            Persisted SkinAnalysisMetric instance
        """
        if timestamp is None:
            timestamp = datetime.utcnow()
        
        # Extract LAB color metrics (facial, palmar, nailbed)
        facial_l = skin_metrics.get('facial_l_lightness', 0.0)
        facial_a = skin_metrics.get('facial_a_red_green', 0.0)
        facial_b = skin_metrics.get('facial_b_yellow_blue', 0.0)
        facial_perfusion = skin_metrics.get('facial_perfusion_index', 0.0)
        
        palmar_l = skin_metrics.get('palmar_l_lightness', 0.0)
        palmar_a = skin_metrics.get('palmar_a_red_green', 0.0)
        palmar_b = skin_metrics.get('palmar_b_yellow_blue', 0.0)
        palmar_perfusion = skin_metrics.get('palmar_perfusion_index', 0.0)
        
        nailbed_l = skin_metrics.get('nailbed_l_lightness', 0.0)
        nailbed_a = skin_metrics.get('nailbed_a_red_green', 0.0)
        nailbed_b = skin_metrics.get('nailbed_b_yellow_blue', 0.0)
        nailbed_color_index = skin_metrics.get('nailbed_color_index', 0.0)
        
        # Clinical color changes
        pallor_detected = skin_metrics.get('pallor_detected', False)
        pallor_severity = skin_metrics.get('pallor_severity', 0.0)
        pallor_region = skin_metrics.get('pallor_region', 'none')
        
        cyanosis_detected = skin_metrics.get('cyanosis_detected', False)
        cyanosis_severity = skin_metrics.get('cyanosis_severity', 0.0)
        cyanosis_region = skin_metrics.get('cyanosis_region', 'none')
        
        jaundice_detected = skin_metrics.get('jaundice_detected', False)
        jaundice_severity = skin_metrics.get('jaundice_severity', 0.0)
        jaundice_region = skin_metrics.get('jaundice_region', 'none')
        
        # Capillary refill
        capillary_refill_time = skin_metrics.get('capillary_refill_time_sec', None)
        capillary_refill_method = skin_metrics.get('capillary_refill_method', 'not_measured')
        capillary_refill_quality = skin_metrics.get('capillary_refill_quality', 0.0)
        capillary_refill_abnormal = skin_metrics.get('capillary_refill_abnormal', False)
        
        # Nailbed analysis
        nail_clubbing_detected = skin_metrics.get('nail_clubbing_detected', False)
        nail_clubbing_severity = skin_metrics.get('nail_clubbing_severity', 0.0)
        nail_pitting_detected = skin_metrics.get('nail_pitting_detected', False)
        nail_pitting_count = skin_metrics.get('nail_pitting_count', 0)
        nail_abnormalities = skin_metrics.get('nail_abnormalities', [])
        
        # Texture & temperature
        skin_texture_score = skin_metrics.get('skin_texture_score', 0.0)
        hydration_status = skin_metrics.get('hydration_status', 'normal')
        temperature_proxy = skin_metrics.get('temperature_proxy', 'normal')
        
        # Rash/lesion detection
        rash_detected = skin_metrics.get('rash_detected', False)
        rash_distribution = skin_metrics.get('rash_distribution', 'none')
        lesions_bruises_detected = skin_metrics.get('lesions_bruises_detected', False)
        lesion_details = skin_metrics.get('lesion_details', [])
        
        # Detection quality
        detection_confidence = skin_metrics.get('detection_confidence', 0.0)
        frames_analyzed = skin_metrics.get('frames_analyzed', 0)
        facial_roi_detected = skin_metrics.get('facial_roi_detected', False)
        palmar_roi_detected = skin_metrics.get('palmar_roi_detected', False)
        nailbed_roi_detected = skin_metrics.get('nailbed_roi_detected', False)
        
        # Compute Z-scores vs baseline
        baseline = self._get_or_create_baseline(patient_id)
        z_score_perfusion = self._compute_z_score(
            facial_perfusion,
            baseline.baseline_facial_perfusion if baseline else 70.0,
            10.0  # Default std
        )
        z_score_capillary = self._compute_z_score(
            capillary_refill_time if capillary_refill_time else 1.5,
            baseline.baseline_capillary_refill_sec if baseline else 1.5,
            baseline.baseline_capillary_refill_std if baseline else 0.3
        )
        z_score_nailbed = self._compute_z_score(
            nailbed_color_index,
            baseline.baseline_nailbed_color_index if baseline else 70.0,
            10.0  # Default std
        )
        
        # Create metric instance
        metric = SkinAnalysisMetric(
            patient_id=patient_id,
            session_id=session_id,
            recorded_at=timestamp,
            # LAB color metrics
            facial_l_lightness=facial_l,
            facial_a_red_green=facial_a,
            facial_b_yellow_blue=facial_b,
            facial_perfusion_index=facial_perfusion,
            palmar_l_lightness=palmar_l,
            palmar_a_red_green=palmar_a,
            palmar_b_yellow_blue=palmar_b,
            palmar_perfusion_index=palmar_perfusion,
            nailbed_l_lightness=nailbed_l,
            nailbed_a_red_green=nailbed_a,
            nailbed_b_yellow_blue=nailbed_b,
            nailbed_color_index=nailbed_color_index,
            # Clinical color changes
            pallor_detected=pallor_detected,
            pallor_severity=pallor_severity,
            pallor_region=pallor_region,
            cyanosis_detected=cyanosis_detected,
            cyanosis_severity=cyanosis_severity,
            cyanosis_region=cyanosis_region,
            jaundice_detected=jaundice_detected,
            jaundice_severity=jaundice_severity,
            jaundice_region=jaundice_region,
            # Capillary refill
            capillary_refill_time_sec=capillary_refill_time,
            capillary_refill_method=capillary_refill_method,
            capillary_refill_quality=capillary_refill_quality,
            capillary_refill_abnormal=capillary_refill_abnormal,
            # Nailbed analysis
            nail_clubbing_detected=nail_clubbing_detected,
            nail_clubbing_severity=nail_clubbing_severity,
            nail_pitting_detected=nail_pitting_detected,
            nail_pitting_count=nail_pitting_count,
            nail_abnormalities=nail_abnormalities,
            # Texture & temperature
            skin_texture_score=skin_texture_score,
            hydration_status=hydration_status,
            temperature_proxy=temperature_proxy,
            # Rash/lesion detection
            rash_detected=rash_detected,
            rash_distribution=rash_distribution,
            lesions_bruises_detected=lesions_bruises_detected,
            lesion_details=lesion_details,
            # Baseline comparison
            z_score_perfusion_vs_baseline=z_score_perfusion,
            z_score_capillary_vs_baseline=z_score_capillary,
            z_score_nailbed_vs_baseline=z_score_nailbed,
            # Detection quality
            detection_confidence=detection_confidence,
            frames_analyzed=frames_analyzed,
            facial_roi_detected=facial_roi_detected,
            palmar_roi_detected=palmar_roi_detected,
            nailbed_roi_detected=nailbed_roi_detected,
            # Metadata
            metrics_metadata={
                'full_skin_metrics': skin_metrics,
                'timestamp_iso': timestamp.isoformat()
            }
        )
        
        self.db.add(metric)
        self.db.commit()
        self.db.refresh(metric)
        
        # Compute and update temporal analytics
        self._compute_temporal_metrics(patient_id, metric.id)
        
        # Update baseline if conditions are favorable (healthy state, high confidence)
        if detection_confidence > 0.7 and not any([pallor_detected, cyanosis_detected, jaundice_detected]):
            self._update_baseline(
                patient_id,
                facial_l, facial_a, facial_b, facial_perfusion,
                palmar_l, palmar_a, palmar_b, palmar_perfusion,
                nailbed_l, nailbed_a, nailbed_b, nailbed_color_index,
                capillary_refill_time if capillary_refill_time else 1.5,
                skin_texture_score,
                hydration_status,
                detection_confidence
            )
        
        logger.info(f"Skin analysis metrics ingested for {patient_id}: Perfusion={facial_perfusion:.1f}, "
                   f"Pallor={pallor_detected}, Cyanosis={cyanosis_detected}, Jaundice={jaundice_detected}")
        
        return metric
    
    def _get_or_create_baseline(self, patient_id: str) -> Optional[SkinBaseline]:
        """Get existing baseline or return None"""
        return self.db.query(SkinBaseline).filter(
            SkinBaseline.patient_id == patient_id
        ).first()
    
    def _update_baseline(
        self,
        patient_id: str,
        facial_l: float, facial_a: float, facial_b: float, facial_perfusion: float,
        palmar_l: float, palmar_a: float, palmar_b: float, palmar_perfusion: float,
        nailbed_l: float, nailbed_a: float, nailbed_b: float, nailbed_color_index: float,
        capillary_refill_sec: float,
        texture_score: float,
        hydration_status: str,
        confidence: float
    ):
        """
        Update patient baseline using exponential moving average
        Only updates when in healthy state (high confidence, no abnormalities)
        """
        baseline = self.db.query(SkinBaseline).filter(
            SkinBaseline.patient_id == patient_id
        ).first()
        
        if not baseline:
            # Initialize baseline with current measurements
            baseline = SkinBaseline(
                patient_id=patient_id,
                baseline_facial_l=facial_l,
                baseline_facial_a=facial_a,
                baseline_facial_b=facial_b,
                baseline_facial_perfusion=facial_perfusion,
                baseline_palmar_l=palmar_l,
                baseline_palmar_a=palmar_a,
                baseline_palmar_b=palmar_b,
                baseline_palmar_perfusion=palmar_perfusion,
                baseline_nailbed_l=nailbed_l,
                baseline_nailbed_a=nailbed_a,
                baseline_nailbed_b=nailbed_b,
                baseline_nailbed_color_index=nailbed_color_index,
                baseline_capillary_refill_sec=capillary_refill_sec,
                baseline_capillary_refill_std=0.3,
                baseline_texture_score=texture_score,
                baseline_hydration_status=hydration_status,
                sample_size=1,
                confidence=confidence,
                source="auto",
                last_calibration_at=datetime.utcnow()
            )
            self.db.add(baseline)
            logger.info(f"Initialized skin analysis baseline for {patient_id}")
        else:
            # Exponential moving average update (α = 0.1 for slower adaptation)
            alpha = 0.1 if baseline.sample_size >= 5 else 1.0 / (baseline.sample_size + 1)
            
            # Update facial baseline
            baseline.baseline_facial_l = baseline.baseline_facial_l * (1 - alpha) + facial_l * alpha
            baseline.baseline_facial_a = baseline.baseline_facial_a * (1 - alpha) + facial_a * alpha
            baseline.baseline_facial_b = baseline.baseline_facial_b * (1 - alpha) + facial_b * alpha
            baseline.baseline_facial_perfusion = baseline.baseline_facial_perfusion * (1 - alpha) + facial_perfusion * alpha
            
            # Update palmar baseline
            baseline.baseline_palmar_l = baseline.baseline_palmar_l * (1 - alpha) + palmar_l * alpha
            baseline.baseline_palmar_a = baseline.baseline_palmar_a * (1 - alpha) + palmar_a * alpha
            baseline.baseline_palmar_b = baseline.baseline_palmar_b * (1 - alpha) + palmar_b * alpha
            baseline.baseline_palmar_perfusion = baseline.baseline_palmar_perfusion * (1 - alpha) + palmar_perfusion * alpha
            
            # Update nailbed baseline
            baseline.baseline_nailbed_l = baseline.baseline_nailbed_l * (1 - alpha) + nailbed_l * alpha
            baseline.baseline_nailbed_a = baseline.baseline_nailbed_a * (1 - alpha) + nailbed_a * alpha
            baseline.baseline_nailbed_b = baseline.baseline_nailbed_b * (1 - alpha) + nailbed_b * alpha
            baseline.baseline_nailbed_color_index = baseline.baseline_nailbed_color_index * (1 - alpha) + nailbed_color_index * alpha
            
            # Update capillary refill baseline
            baseline.baseline_capillary_refill_sec = baseline.baseline_capillary_refill_sec * (1 - alpha) + capillary_refill_sec * alpha
            
            # Update texture baseline
            baseline.baseline_texture_score = baseline.baseline_texture_score * (1 - alpha) + texture_score * alpha if baseline.baseline_texture_score else texture_score
            baseline.baseline_hydration_status = hydration_status
            
            baseline.sample_size += 1
            baseline.confidence = min(0.95, baseline.confidence + 0.05)
            baseline.last_calibration_at = datetime.utcnow()
            
            logger.info(f"Updated skin analysis baseline for {patient_id} (sample_size={baseline.sample_size})")
        
        self.db.commit()
    
    def _compute_temporal_metrics(self, patient_id: str, current_metric_id: int):
        """
        Compute rolling 24hr average perfusion, 3-day trend slope, capillary refill trends
        Updates the current metric with computed statistics
        """
        # Get last 24 hours of data
        cutoff_24hr = datetime.utcnow() - timedelta(hours=24)
        recent_metrics = self.db.query(SkinAnalysisMetric).filter(
            SkinAnalysisMetric.patient_id == patient_id,
            SkinAnalysisMetric.recorded_at >= cutoff_24hr
        ).order_by(SkinAnalysisMetric.recorded_at).all()
        
        if len(recent_metrics) < 2:
            return  # Need at least 2 data points
        
        # Compute 24-hour rolling averages
        perfusion_values = [m.facial_perfusion_index for m in recent_metrics if m.facial_perfusion_index]
        capillary_values = [m.capillary_refill_time_sec for m in recent_metrics if m.capillary_refill_time_sec]
        
        rolling_24hr_perfusion = statistics.mean(perfusion_values) if perfusion_values else None
        rolling_24hr_capillary = statistics.mean(capillary_values) if capillary_values else None
        
        # Compute 3-day trend slope (linear regression)
        cutoff_3day = datetime.utcnow() - timedelta(days=3)
        three_day_metrics = self.db.query(SkinAnalysisMetric).filter(
            SkinAnalysisMetric.patient_id == patient_id,
            SkinAnalysisMetric.recorded_at >= cutoff_3day
        ).order_by(SkinAnalysisMetric.recorded_at).all()
        
        slope_3day = None
        if len(three_day_metrics) >= 3:
            # Simple linear regression: slope = Cov(X,Y) / Var(X)
            timestamps = [(m.recorded_at - three_day_metrics[0].recorded_at).total_seconds() / 3600 for m in three_day_metrics]
            perfusion_vals = [m.facial_perfusion_index for m in three_day_metrics if m.facial_perfusion_index]
            
            if len(perfusion_vals) >= 3:
                mean_time = statistics.mean(timestamps[:len(perfusion_vals)])
                mean_perfusion = statistics.mean(perfusion_vals)
                
                covariance = sum((timestamps[i] - mean_time) * (perfusion_vals[i] - mean_perfusion) 
                               for i in range(len(perfusion_vals))) / len(perfusion_vals)
                variance = sum((t - mean_time) ** 2 for t in timestamps[:len(perfusion_vals)]) / len(perfusion_vals)
                
                if variance > 0:
                    slope_3day = covariance / variance
        
        # Update current metric
        current_metric = self.db.query(SkinAnalysisMetric).filter(
            SkinAnalysisMetric.id == current_metric_id
        ).first()
        
        if current_metric:
            current_metric.rolling_24hr_avg_perfusion = rolling_24hr_perfusion
            current_metric.rolling_3day_perfusion_slope = slope_3day
            current_metric.rolling_24hr_avg_capillary_refill = rolling_24hr_capillary
            self.db.commit()
    
    def _compute_z_score(self, value: float, baseline_mean: float, baseline_std: float) -> float:
        """Compute Z-score for anomaly detection"""
        if baseline_std == 0:
            return 0.0
        return (value - baseline_mean) / baseline_std
    
    def get_patient_baseline(self, patient_id: str) -> Optional[Dict[str, float]]:
        """
        Get patient baseline as dict for VideoAIEngine
        Returns format expected by skin analysis pipeline
        """
        baseline = self._get_or_create_baseline(patient_id)
        
        if not baseline:
            return None
        
        return {
            'baseline_facial_l': baseline.baseline_facial_l,
            'baseline_facial_a': baseline.baseline_facial_a,
            'baseline_facial_b': baseline.baseline_facial_b,
            'baseline_facial_perfusion': baseline.baseline_facial_perfusion,
            'baseline_palmar_l': baseline.baseline_palmar_l,
            'baseline_palmar_a': baseline.baseline_palmar_a,
            'baseline_palmar_b': baseline.baseline_palmar_b,
            'baseline_palmar_perfusion': baseline.baseline_palmar_perfusion,
            'baseline_nailbed_l': baseline.baseline_nailbed_l,
            'baseline_nailbed_a': baseline.baseline_nailbed_a,
            'baseline_nailbed_b': baseline.baseline_nailbed_b,
            'baseline_nailbed_color_index': baseline.baseline_nailbed_color_index,
            'baseline_capillary_refill_sec': baseline.baseline_capillary_refill_sec,
        }
    
    def get_patient_summary(
        self,
        patient_id: str,
        days: int = 7
    ) -> Dict:
        """
        Get comprehensive skin analysis summary for a patient
        
        Args:
            patient_id: Patient identifier
            days: Number of days to look back
            
        Returns:
            Dict with baseline, recent metrics, trends, and alerts
        """
        cutoff = datetime.utcnow() - timedelta(days=days)
        
        # Get baseline
        baseline = self._get_or_create_baseline(patient_id)
        
        # Get recent metrics
        recent_metrics = self.db.query(SkinAnalysisMetric).filter(
            SkinAnalysisMetric.patient_id == patient_id,
            SkinAnalysisMetric.recorded_at >= cutoff
        ).order_by(desc(SkinAnalysisMetric.recorded_at)).all()
        
        # Compute summary statistics
        perfusion_values = [m.facial_perfusion_index for m in recent_metrics if m.facial_perfusion_index]
        capillary_values = [m.capillary_refill_time_sec for m in recent_metrics if m.capillary_refill_time_sec]
        
        pallor_count = sum(1 for m in recent_metrics if m.pallor_detected)
        cyanosis_count = sum(1 for m in recent_metrics if m.cyanosis_detected)
        jaundice_count = sum(1 for m in recent_metrics if m.jaundice_detected)
        
        return {
            'patient_id': patient_id,
            'baseline': {
                'facial_perfusion': baseline.baseline_facial_perfusion if baseline else None,
                'capillary_refill_sec': baseline.baseline_capillary_refill_sec if baseline else None,
                'sample_size': baseline.sample_size if baseline else 0,
                'confidence': baseline.confidence if baseline else 0.0,
            },
            'recent_metrics_count': len(recent_metrics),
            'avg_perfusion': statistics.mean(perfusion_values) if perfusion_values else None,
            'avg_capillary_refill': statistics.mean(capillary_values) if capillary_values else None,
            'pallor_detections': pallor_count,
            'cyanosis_detections': cyanosis_count,
            'jaundice_detections': jaundice_count,
            'abnormal_capillary_refill_count': sum(1 for m in recent_metrics if m.capillary_refill_abnormal),
            'nail_clubbing_detections': sum(1 for m in recent_metrics if m.nail_clubbing_detected),
            'latest_metric': {
                'recorded_at': recent_metrics[0].recorded_at.isoformat() if recent_metrics else None,
                'facial_perfusion': recent_metrics[0].facial_perfusion_index if recent_metrics else None,
                'z_score_perfusion': recent_metrics[0].z_score_perfusion_vs_baseline if recent_metrics else None,
                '3day_trend_slope': recent_metrics[0].rolling_3day_perfusion_slope if recent_metrics else None,
            }
        }
