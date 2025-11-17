"""
Respiratory Metrics Service - Advanced temporal analytics
Computes baseline RR, rolling averages, trends, variability index, and anomaly scores
"""

import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import and_, desc
import logging

from app.models import RespiratoryBaseline, RespiratoryMetric

# Scipy optional - fallback to simple calculations if not available
try:
    from scipy import stats as scipy_stats
    from scipy.signal import find_peaks
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logging.warning("scipy not available - using simplified respiratory calculations")

logger = logging.getLogger(__name__)


class RespiratoryMetricsService:
    """Advanced respiratory metrics computation with temporal analytics"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def ingest_session(
        self,
        patient_id: str,
        session_id: str,
        rr_bpm: float,
        rr_confidence: float,
        chest_movements: List[float],
        accessory_muscle_scores: Optional[List[float]] = None,
        chest_widths: Optional[List[float]] = None,
        fps: float = 30.0,
        timestamp: Optional[datetime] = None
    ) -> RespiratoryMetric:
        """
        Ingest respiratory session and compute all advanced metrics
        """
        if timestamp is None:
            timestamp = datetime.utcnow()
        
        # Compute variability metrics
        variability_metrics = self._compute_variability_index(chest_movements, fps)
        
        # Detect abnormal patterns
        gasping_detected = self._detect_gasping(chest_movements)
        
        # Compute shape metrics
        chest_shape_asymmetry = self._compute_chest_asymmetry(chest_widths or [])
        
        # Accessory muscle scoring
        accessory_muscle_score = float(np.mean(accessory_muscle_scores)) if accessory_muscle_scores else 0.0
        
        # Chest expansion amplitude
        chest_expansion_amplitude = float(np.std(chest_movements)) if chest_movements else 0.0
        
        # Thoracoabdominal synchrony
        thoracoabdominal_synchrony = self._estimate_synchrony(chest_movements)
        
        # Update baseline FIRST if high confidence
        if rr_confidence > 0.7:
            self._update_baseline(patient_id, rr_bpm, rr_confidence)
        
        # Refresh baseline after update
        baseline = self._get_or_create_baseline(patient_id)
        
        # Compute Z-score with updated baseline (guards against zero std)
        z_score = self._compute_z_score(rr_bpm, baseline) if baseline else None
        
        # Create metric record (rolling stats will be computed AFTER insert)
        metric = RespiratoryMetric(
            patient_id=patient_id,
            session_id=session_id,
            recorded_at=timestamp,
            rr_bpm=rr_bpm,
            rr_confidence=rr_confidence,
            breath_interval_std=variability_metrics['breath_interval_std'],
            variability_index=variability_metrics['variability_index'],
            accessory_muscle_score=accessory_muscle_score,
            chest_expansion_amplitude=chest_expansion_amplitude,
            gasping_detected=gasping_detected,
            chest_shape_asymmetry=chest_shape_asymmetry,
            thoracoabdominal_synchrony=thoracoabdominal_synchrony,
            z_score_vs_baseline=z_score,
            rolling_daily_avg=None,  # Will compute after insert
            rolling_three_day_slope=None,  # Will compute after insert
            metadata={
                'chest_movements_sample': chest_movements[:100] if len(chest_movements) > 100 else chest_movements,
                'fps': fps,
                'session_duration': len(chest_movements) / fps if fps > 0 else 0
            }
        )
        
        self.db.add(metric)
        self.db.commit()
        self.db.refresh(metric)
        
        # NOW compute rolling statistics (includes current session)
        metric.rolling_daily_avg = self._compute_rolling_daily_avg(patient_id, timestamp)
        metric.rolling_three_day_slope = self._compute_rolling_three_day_slope(patient_id, timestamp)
        self.db.commit()
        self.db.refresh(metric)
        
        logger.info(f"Respiratory metrics ingested for {patient_id}: RR={rr_bpm:.1f}, Z={z_score:.2f if z_score else 0:.2f}")
        
        return metric
    
    def _compute_variability_index(
        self,
        chest_movements: List[float],
        fps: float
    ) -> Dict[str, float]:
        """
        Compute Respiratory Variability Index (RVI)
        RVI = (std_dev / mean) * 100 of breath intervals
        """
        if len(chest_movements) < 10:
            return {'breath_interval_std': 0.0, 'variability_index': 0.0}
        
        chest_array = np.array(chest_movements)
        detrended = chest_array - np.mean(chest_array)
        
        # Find peaks (inhalation points)
        if SCIPY_AVAILABLE:
            peaks, _ = find_peaks(detrended, distance=int(fps * 2))
        else:
            # Simple peak detection fallback
            peaks = []
            for i in range(1, len(detrended) - 1):
                if detrended[i] > detrended[i-1] and detrended[i] > detrended[i+1]:
                    if not peaks or i - peaks[-1] > fps * 2:
                        peaks.append(i)
            peaks = np.array(peaks)
        
        if len(peaks) < 3:
            return {'breath_interval_std': 0.0, 'variability_index': 0.0}
        
        # Compute inter-breath intervals (seconds)
        breath_intervals = np.diff(peaks) / fps
        breath_interval_mean = np.mean(breath_intervals)
        breath_interval_std = np.std(breath_intervals)
        
        # Variability Index
        variability_index = (breath_interval_std / breath_interval_mean * 100) if breath_interval_mean > 0 else 0.0
        
        return {
            'breath_interval_std': float(breath_interval_std),
            'variability_index': float(variability_index)
        }
    
    def _detect_gasping(self, chest_movements: List[float]) -> bool:
        """Detect gasping - irregular breathing with sudden deep breaths"""
        if len(chest_movements) < 20:
            return False
        
        chest_array = np.array(chest_movements)
        window_size = 10
        amplitudes = []
        
        for i in range(0, len(chest_array) - window_size, window_size):
            window = chest_array[i:i+window_size]
            amplitude = np.max(window) - np.min(window)
            amplitudes.append(amplitude)
        
        if len(amplitudes) < 3:
            return False
        
        mean_amp = np.mean(amplitudes)
        std_amp = np.std(amplitudes)
        
        if std_amp == 0:
            return False
        
        outliers = [a for a in amplitudes if abs(a - mean_amp) > 2 * std_amp]
        return len(outliers) / len(amplitudes) > 0.2
    
    def _compute_chest_asymmetry(self, chest_widths: List[float]) -> float:
        """Compute chest shape asymmetry score"""
        if len(chest_widths) < 5:
            return 0.0
        
        std = np.std(chest_widths)
        mean = np.mean(chest_widths)
        asymmetry = (std / mean * 100) if mean > 0 else 0.0
        return float(asymmetry)
    
    def _estimate_synchrony(self, chest_movements: List[float]) -> float:
        """Estimate thoracoabdominal synchrony (0=asynchronous, 1=synchronous)"""
        if len(chest_movements) < 10:
            return 0.5
        
        # Regularity proxy: inverse of variability
        std = np.std(chest_movements)
        mean = np.mean(chest_movements)
        cv = (std / mean) if mean > 0 else 1.0
        
        # Convert to 0-1 scale (lower CV = higher synchrony)
        synchrony = max(0.0, min(1.0, 1.0 - cv))
        return float(synchrony)
    
    def _get_or_create_baseline(self, patient_id: str) -> Optional[RespiratoryBaseline]:
        """Get existing baseline or return None"""
        return self.db.query(RespiratoryBaseline).filter(
            RespiratoryBaseline.patient_id == patient_id
        ).first()
    
    def _update_baseline(self, patient_id: str, new_rr: float, confidence: float):
        """Update patient baseline using exponential moving average"""
        baseline = self.db.query(RespiratoryBaseline).filter(
            RespiratoryBaseline.patient_id == patient_id
        ).first()
        
        if not baseline:
            # Initialize with sensible defaults for Z-score computation
            baseline = RespiratoryBaseline(
                patient_id=patient_id,
                baseline_rr_bpm=new_rr,
                baseline_rr_std=2.0,  # Reasonable default std (guards against divide-by-zero)
                sample_size=1,
                confidence=confidence,
                source="auto"
            )
            self.db.add(baseline)
        else:
            alpha = 0.2 if baseline.sample_size >= 5 else 1.0 / (baseline.sample_size + 1)
            old_mean = baseline.baseline_rr_bpm
            new_mean = old_mean * (1 - alpha) + new_rr * alpha
            deviation = abs(new_rr - old_mean)
            new_std = baseline.baseline_rr_std * (1 - alpha) + deviation * alpha
            
            baseline.baseline_rr_bpm = new_mean
            baseline.baseline_rr_std = max(new_std, 1.0)
            baseline.sample_size += 1
            baseline.confidence = min(0.95, baseline.confidence + 0.05)
        
        self.db.commit()
    
    def _compute_z_score(self, rr_bpm: float, baseline: RespiratoryBaseline) -> float:
        """Compute Z-score anomaly score"""
        if baseline.baseline_rr_std == 0:
            return 0.0
        
        z_score = (rr_bpm - baseline.baseline_rr_bpm) / baseline.baseline_rr_std
        return float(z_score)
    
    def _compute_rolling_daily_avg(self, patient_id: str, timestamp: datetime) -> Optional[float]:
        """Compute rolling 24-hour average RR"""
        cutoff = timestamp - timedelta(hours=24)
        
        metrics = self.db.query(RespiratoryMetric).filter(
            and_(
                RespiratoryMetric.patient_id == patient_id,
                RespiratoryMetric.recorded_at >= cutoff,
                RespiratoryMetric.rr_bpm.isnot(None)
            )
        ).all()
        
        if not metrics:
            return None
        
        rr_values = [m.rr_bpm for m in metrics]
        return float(np.mean(rr_values))
    
    def _compute_rolling_three_day_slope(self, patient_id: str, timestamp: datetime) -> Optional[float]:
        """Compute 3-day trend slope"""
        cutoff = timestamp - timedelta(days=3)
        
        metrics = self.db.query(RespiratoryMetric).filter(
            and_(
                RespiratoryMetric.patient_id == patient_id,
                RespiratoryMetric.recorded_at >= cutoff,
                RespiratoryMetric.rr_bpm.isnot(None)
            )
        ).order_by(RespiratoryMetric.recorded_at).all()
        
        if len(metrics) < 3:
            return None
        
        # Group by day
        daily_avgs = {}
        for m in metrics:
            day_key = m.recorded_at.date()
            if day_key not in daily_avgs:
                daily_avgs[day_key] = []
            daily_avgs[day_key].append(m.rr_bpm)
        
        days = sorted(daily_avgs.keys())
        daily_means = [np.mean(daily_avgs[day]) for day in days]
        
        if len(daily_means) < 2:
            return None
        
        # Linear regression
        if SCIPY_AVAILABLE:
            x = np.arange(len(daily_means))
            slope, _, _, _, _ = scipy_stats.linregress(x, daily_means)
        else:
            # Simple slope calculation
            x = np.arange(len(daily_means))
            slope = np.polyfit(x, daily_means, 1)[0]
        
        return float(slope)
    
    def get_patient_summary(self, patient_id: str) -> Dict[str, Any]:
        """Get comprehensive respiratory summary"""
        baseline = self._get_or_create_baseline(patient_id)
        
        latest = self.db.query(RespiratoryMetric).filter(
            RespiratoryMetric.patient_id == patient_id
        ).order_by(desc(RespiratoryMetric.recorded_at)).first()
        
        cutoff = datetime.utcnow() - timedelta(days=7)
        recent_metrics = self.db.query(RespiratoryMetric).filter(
            and_(
                RespiratoryMetric.patient_id == patient_id,
                RespiratoryMetric.recorded_at >= cutoff
            )
        ).order_by(RespiratoryMetric.recorded_at).all()
        
        return {
            'baseline': {
                'rr_bpm': baseline.baseline_rr_bpm if baseline else None,
                'rr_std': baseline.baseline_rr_std if baseline else None,
                'sample_size': baseline.sample_size if baseline else 0,
                'confidence': baseline.confidence if baseline else 0.0
            },
            'latest': {
                'rr_bpm': latest.rr_bpm if latest else None,
                'recorded_at': latest.recorded_at.isoformat() if latest else None,
                'z_score': latest.z_score_vs_baseline if latest else None,
                'rolling_daily_avg': latest.rolling_daily_avg if latest else None,
                'rolling_three_day_slope': latest.rolling_three_day_slope if latest else None,
                'variability_index': latest.variability_index if latest else None
            },
            'recent_count': len(recent_metrics),
            'trend': 'increasing' if latest and latest.rolling_three_day_slope and latest.rolling_three_day_slope > 0.5 else 
                     'decreasing' if latest and latest.rolling_three_day_slope and latest.rolling_three_day_slope < -0.5 else 
                     'stable'
        }
