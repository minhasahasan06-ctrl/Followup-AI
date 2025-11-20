"""
Digital Biomarkers Service
==========================

Analyzes phone/wearable data for activity patterns, circadian rhythm,
and mobility changes.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import and_
import numpy as np

logger = logging.getLogger(__name__)


class DigitalBiomarkersService:
    """
    Service for analyzing digital biomarkers from phone/wearable data
    
    Tracks:
    - Daily step counts and activity patterns
    - Circadian rhythm stability
    - Phone usage patterns
    - Mobility changes and drops
    """
    
    def __init__(self, db: Session):
        self.db = db
    
    def calculate_daily_biomarkers(
        self,
        patient_id: str,
        date: datetime,
        raw_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate digital biomarkers for a specific day
        
        Args:
            patient_id: Patient identifier
            date: Date to calculate biomarkers for
            raw_data: Raw sensor data from phone/wearable
        
        Returns:
            Dictionary of digital biomarkers
        """
        
        # Extract activity metrics
        daily_steps = raw_data.get('step_count', 0)
        step_trend = self._calculate_step_trend(patient_id, days=7)
        
        # Activity bursts (periods of high activity)
        activity_burst_count = self._detect_activity_bursts(raw_data.get('hourly_steps', []))
        
        # Sedentary time
        sedentary_minutes = raw_data.get('sedentary_duration_minutes', 0)
        
        # Movement variability
        accelerometer_data = raw_data.get('accelerometer', [])
        movement_variability = self._calculate_movement_variability(accelerometer_data)
        
        # Circadian rhythm
        circadian_stability = self._calculate_circadian_rhythm(
            patient_id,
            raw_data.get('hourly_activity', [])
        )
        
        sleep_wake_irregularity = raw_data.get('sleep_wake_irregularity_minutes', 0)
        
        # Peak activity time
        hourly_steps = raw_data.get('hourly_steps', [])
        peak_hour = np.argmax(hourly_steps) if hourly_steps else 12
        peak_time = f"{peak_hour:02d}:00"
        
        # Phone usage patterns
        phone_usage_irregularity = self._calculate_phone_usage_irregularity(
            raw_data.get('screen_on_events', [])
        )
        
        night_interactions = sum(
            1 for event in raw_data.get('screen_on_events', [])
            if 22 <= event.get('hour', 12) or event.get('hour', 12) < 6
        )
        
        screen_on_minutes = raw_data.get('screen_on_duration_minutes', 0)
        
        # Mobility analysis
        baseline_steps = self._get_baseline_steps(patient_id)
        mobility_change = ((daily_steps - baseline_steps) / max(baseline_steps, 1)) * 100 if baseline_steps else 0.0
        mobility_drop = mobility_change < -30.0  # 30% drop threshold
        
        # Accelerometer statistics
        accel_std = np.std(accelerometer_data) if accelerometer_data else 0.0
        accel_mean = np.mean(np.abs(accelerometer_data)) if accelerometer_data else 0.0
        
        return {
            'daily_step_count': daily_steps,
            'step_trend_7day': step_trend,
            'activity_burst_count': activity_burst_count,
            'sedentary_duration_minutes': sedentary_minutes,
            'movement_variability_score': movement_variability,
            'circadian_rhythm_stability': circadian_stability,
            'sleep_wake_irregularity_minutes': sleep_wake_irregularity,
            'daily_peak_activity_time': peak_time,
            'phone_usage_irregularity': phone_usage_irregularity,
            'night_phone_interaction_count': night_interactions,
            'screen_on_duration_minutes': screen_on_minutes,
            'mobility_drop_detected': mobility_drop,
            'mobility_change_percent': float(mobility_change),
            'accelerometer_std_dev': float(accel_std),
            'accelerometer_mean_magnitude': float(accel_mean)
        }
    
    def _calculate_step_trend(self, patient_id: str, days: int = 7) -> float:
        """Calculate 7-day trend slope of step counts"""
        from app.models.behavior_models import DigitalBiomarker
        
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        biomarkers = self.db.query(DigitalBiomarker).filter(
            and_(
                DigitalBiomarker.patient_id == patient_id,
                DigitalBiomarker.date >= cutoff_date
            )
        ).order_by(DigitalBiomarker.date).all()
        
        if len(biomarkers) < 2:
            return 0.0
        
        steps = [b.daily_step_count for b in biomarkers if b.daily_step_count]
        if len(steps) < 2:
            return 0.0
        
        x = np.arange(len(steps))
        slope, _ = np.polyfit(x, steps, 1)
        
        return float(slope)
    
    def _detect_activity_bursts(self, hourly_steps: List[int]) -> int:
        """Detect sudden bursts of activity"""
        if not hourly_steps:
            return 0
        
        # Burst = hour with >2x median steps
        median_steps = np.median(hourly_steps)
        burst_threshold = median_steps * 2
        
        bursts = sum(1 for steps in hourly_steps if steps > burst_threshold)
        
        return bursts
    
    def _calculate_movement_variability(self, accelerometer_data: List[float]) -> float:
        """Calculate movement variability score (0-1)"""
        if not accelerometer_data or len(accelerometer_data) < 10:
            return 0.0
        
        # Coefficient of variation
        std = np.std(accelerometer_data)
        mean = np.mean(np.abs(accelerometer_data))
        
        if mean == 0:
            return 0.0
        
        cv = std / mean
        
        # Normalize to 0-1 (higher = more variable)
        variability = min(cv / 2.0, 1.0)
        
        return float(variability)
    
    def _calculate_circadian_rhythm(
        self,
        patient_id: str,
        hourly_activity: List[float]
    ) -> float:
        """
        Calculate circadian rhythm stability (0-1)
        
        Higher score = more consistent daily rhythm
        """
        from app.models.behavior_models import DigitalBiomarker
        
        # Get historical hourly patterns
        cutoff_date = datetime.utcnow() - timedelta(days=7)
        
        past_biomarkers = self.db.query(DigitalBiomarker).filter(
            and_(
                DigitalBiomarker.patient_id == patient_id,
                DigitalBiomarker.date >= cutoff_date
            )
        ).all()
        
        if not past_biomarkers or not hourly_activity:
            return 0.5  # Neutral baseline
        
        # Simple heuristic: check if peak activity time is consistent
        # In production, would use autocorrelation or FFT
        peak_hours = []
        for bio in past_biomarkers:
            if bio.daily_peak_activity_time:
                hour = int(bio.daily_peak_activity_time.split(':')[0])
                peak_hours.append(hour)
        
        if len(peak_hours) < 3:
            return 0.5
        
        # Stability = 1 - (std of peak hours / 12)
        peak_std = np.std(peak_hours)
        stability = 1.0 - min(peak_std / 12.0, 1.0)
        
        return float(stability)
    
    def _calculate_phone_usage_irregularity(self, screen_events: List[Dict]) -> float:
        """Calculate phone usage irregularity (0-1)"""
        if not screen_events:
            return 0.0
        
        # Extract event times (hour of day)
        event_hours = [event.get('hour', 12) for event in screen_events]
        
        # Irregularity = coefficient of variation of event times
        if len(event_hours) < 2:
            return 0.0
        
        std = np.std(event_hours)
        mean = np.mean(event_hours)
        
        if mean == 0:
            return 1.0
        
        irregularity = min((std / mean), 1.0)
        
        return float(irregularity)
    
    def _get_baseline_steps(self, patient_id: str, days: int = 30) -> float:
        """Get patient's baseline step count (30-day average)"""
        from app.models.behavior_models import DigitalBiomarker
        
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        biomarkers = self.db.query(DigitalBiomarker).filter(
            and_(
                DigitalBiomarker.patient_id == patient_id,
                DigitalBiomarker.date >= cutoff_date
            )
        ).all()
        
        if not biomarkers:
            return 5000.0  # Default baseline
        
        steps = [b.daily_step_count for b in biomarkers if b.daily_step_count]
        
        return float(np.mean(steps)) if steps else 5000.0
    
    def detect_mobility_drop(
        self,
        patient_id: str,
        current_steps: int
    ) -> Dict[str, Any]:
        """
        Detect sudden drop in mobility
        
        Args:
            patient_id: Patient identifier
            current_steps: Today's step count
        
        Returns:
            {
                'drop_detected': bool,
                'change_percent': float,
                'baseline_steps': float
            }
        """
        baseline = self._get_baseline_steps(patient_id)
        change_percent = ((current_steps - baseline) / max(baseline, 1)) * 100
        
        drop_detected = change_percent < -30.0
        
        return {
            'drop_detected': drop_detected,
            'change_percent': float(change_percent),
            'baseline_steps': baseline
        }
