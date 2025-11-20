"""
Behavioral Metrics Service
==========================

Tracks and analyzes user engagement patterns, medication adherence,
avoidance behaviors, and routine deviations.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import and_, func
import numpy as np

logger = logging.getLogger(__name__)


class BehavioralMetricsService:
    """
    Service for analyzing behavioral patterns and engagement metrics
    
    Tracks:
    - Check-in time consistency
    - Medication adherence
    - App engagement duration
    - Avoidance language patterns
    - Routine deviations
    """
    
    def __init__(self, db: Session):
        self.db = db
    
    def calculate_daily_metrics(
        self,
        patient_id: str,
        date: datetime
    ) -> Dict[str, Any]:
        """
        Calculate behavioral metrics for a specific day
        
        Args:
            patient_id: Patient identifier
            date: Date to calculate metrics for
        
        Returns:
            Dictionary of behavioral metrics
        """
        from app.models.behavior_models import BehaviorCheckin
        
        # Get all check-ins for the day
        start_date = date.replace(hour=0, minute=0, second=0, microsecond=0)
        end_date = start_date + timedelta(days=1)
        
        checkins = self.db.query(BehaviorCheckin).filter(
            and_(
                BehaviorCheckin.patient_id == patient_id,
                BehaviorCheckin.scheduled_time >= start_date,
                BehaviorCheckin.scheduled_time < end_date
            )
        ).all()
        
        if not checkins:
            logger.warning(f"No check-ins found for patient {patient_id} on {date.date()}")
            return self._empty_metrics()
        
        # Calculate check-in consistency
        completed_checkins = [c for c in checkins if not c.skipped]
        total_checkins = len(checkins)
        completion_rate = len(completed_checkins) / max(total_checkins, 1)
        
        # Calculate time consistency (std dev of check-in times)
        if completed_checkins:
            checkin_times = [
                (c.completed_at.hour * 60 + c.completed_at.minute)
                for c in completed_checkins
                if c.completed_at
            ]
            time_consistency = 1.0 - min(np.std(checkin_times) / 60.0, 1.0) if checkin_times else 0.0
        else:
            time_consistency = 0.0
        
        # Average response latency
        latencies = [c.response_latency_minutes for c in completed_checkins if c.response_latency_minutes is not None]
        avg_latency = np.mean(latencies) if latencies else 0.0
        
        # Skipped check-ins
        skipped_count = sum(1 for c in checkins if c.skipped)
        
        # Medication adherence
        medication_taken_count = sum(1 for c in completed_checkins if c.medication_taken)
        medication_adherence = medication_taken_count / max(len(completed_checkins), 1)
        medication_skips = len(completed_checkins) - medication_taken_count
        
        # App engagement
        session_durations = [c.session_duration_seconds for c in completed_checkins if c.session_duration_seconds]
        total_engagement_minutes = sum(session_durations) / 60.0 if session_durations else 0.0
        
        # Avoidance pattern detection
        avoidance_phrases_all = []
        for c in checkins:
            if c.avoidance_language_detected and c.avoidance_phrases:
                avoidance_phrases_all.extend(c.avoidance_phrases)
        
        avoidance_detected = len(avoidance_phrases_all) > 0
        avoidance_count = len(avoidance_phrases_all)
        
        # Sentiment analysis
        sentiment_polarities = [c.sentiment_polarity for c in completed_checkins if c.sentiment_polarity is not None]
        avg_sentiment = float(np.mean(sentiment_polarities)) if sentiment_polarities else 0.0
        
        # Routine deviation (based on time consistency and completion rate)
        routine_deviation = 1.0 - ((time_consistency + completion_rate) / 2.0)
        
        return {
            'checkin_time_consistency_score': time_consistency,
            'checkin_completion_rate': completion_rate,
            'avg_response_latency_minutes': float(avg_latency),
            'skipped_checkins_count': skipped_count,
            'routine_deviation_score': routine_deviation,
            'medication_adherence_rate': medication_adherence,
            'medication_skips_count': medication_skips,
            'app_engagement_duration_minutes': total_engagement_minutes,
            'app_open_count': len(completed_checkins),
            'avoidance_patterns_detected': avoidance_detected,
            'avoidance_count': avoidance_count,
            'avoidance_phrases': list(set(avoidance_phrases_all)),
            'avg_sentiment_polarity': avg_sentiment
        }
    
    def calculate_sentiment_trend(
        self,
        patient_id: str,
        days: int = 7
    ) -> float:
        """
        Calculate trend slope of sentiment over time
        
        Args:
            patient_id: Patient identifier
            days: Number of days to analyze
        
        Returns:
            Trend slope (negative = declining sentiment)
        """
        from app.models.behavior_models import BehaviorMetric
        
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        metrics = self.db.query(BehaviorMetric).filter(
            and_(
                BehaviorMetric.patient_id == patient_id,
                BehaviorMetric.date >= cutoff_date
            )
        ).order_by(BehaviorMetric.date).all()
        
        if len(metrics) < 2:
            return 0.0
        
        # Linear regression on sentiment
        sentiments = [float(m.avg_sentiment_polarity) for m in metrics if m.avg_sentiment_polarity is not None]
        if len(sentiments) < 2:
            return 0.0
        
        x = np.arange(len(sentiments))
        slope, _ = np.polyfit(x, sentiments, 1)
        
        return float(slope)
    
    def detect_avoidance_patterns(
        self,
        text: str
    ) -> Dict[str, Any]:
        """
        Detect avoidance language in user input
        
        Args:
            text: User input text
        
        Returns:
            {
                'detected': bool,
                'phrases': list of detected phrases
            }
        """
        avoidance_patterns = [
            'too tired',
            'not today',
            'maybe later',
            'can\'t right now',
            'don\'t feel like',
            'not in the mood',
            'next time',
            'skip',
            'pass',
            'forget it'
        ]
        
        text_lower = text.lower()
        found_phrases = [pattern for pattern in avoidance_patterns if pattern in text_lower]
        
        return {
            'detected': len(found_phrases) > 0,
            'phrases': found_phrases
        }
    
    def get_engagement_trend(
        self,
        patient_id: str,
        days: int = 14
    ) -> Dict[str, Any]:
        """
        Get trend of app engagement over time
        
        Args:
            patient_id: Patient identifier
            days: Number of days to analyze
        
        Returns:
            Engagement trend data
        """
        from app.models.behavior_models import BehaviorMetric
        
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        metrics = self.db.query(BehaviorMetric).filter(
            and_(
                BehaviorMetric.patient_id == patient_id,
                BehaviorMetric.date >= cutoff_date
            )
        ).order_by(BehaviorMetric.date).all()
        
        if not metrics:
            return {
                'trend_direction': 'insufficient_data',
                'average_duration_minutes': 0.0,
                'trend_slope': 0.0
            }
        
        durations = [float(m.app_engagement_duration_minutes) for m in metrics if m.app_engagement_duration_minutes]
        
        if len(durations) < 2:
            return {
                'trend_direction': 'insufficient_data',
                'average_duration_minutes': float(np.mean(durations)) if durations else 0.0,
                'trend_slope': 0.0
            }
        
        # Calculate trend
        x = np.arange(len(durations))
        slope, intercept = np.polyfit(x, durations, 1)
        
        if slope < -0.5:
            trend_direction = 'declining'
        elif slope > 0.5:
            trend_direction = 'increasing'
        else:
            trend_direction = 'stable'
        
        return {
            'trend_direction': trend_direction,
            'average_duration_minutes': float(np.mean(durations)),
            'trend_slope': float(slope),
            'recent_duration_minutes': durations[-1] if durations else 0.0
        }
    
    def _empty_metrics(self) -> Dict[str, Any]:
        """Return empty metrics structure"""
        return {
            'checkin_time_consistency_score': 0.0,
            'checkin_completion_rate': 0.0,
            'avg_response_latency_minutes': 0.0,
            'skipped_checkins_count': 0,
            'routine_deviation_score': 0.0,
            'medication_adherence_rate': 0.0,
            'medication_skips_count': 0,
            'app_engagement_duration_minutes': 0.0,
            'app_open_count': 0,
            'avoidance_patterns_detected': False,
            'avoidance_count': 0,
            'avoidance_phrases': [],
            'avg_sentiment_polarity': 0.0
        }
