"""
Deterioration Trend Detection Engine
====================================

Detects temporal patterns of health deterioration across all biomarker streams.
Uses statistical analysis and time-series modeling.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import and_
import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


class DeteriorationTrendEngine:
    """
    Detects gradual and sudden deterioration trends
    
    Trend Types:
    - Declining engagement (behavioral)
    - Mobility drop (digital biomarker)
    - Cognitive decline (test performance)
    - Sentiment deterioration (language)
    """
    
    def __init__(self, db: Session):
        self.db = db
    
    def detect_all_trends(
        self,
        patient_id: str,
        lookback_days: int = 14
    ) -> List[Dict[str, Any]]:
        """
        Detect all deterioration trends for patient
        
        Args:
            patient_id: Patient identifier
            lookback_days: Days of historical data to analyze
        
        Returns:
            List of detected trends
        """
        
        logger.info(f"Detecting deterioration trends for patient {patient_id}")
        
        trends = []
        
        # Check each modality
        behavioral_trend = self._detect_behavioral_trend(patient_id, lookback_days)
        if behavioral_trend:
            trends.append(behavioral_trend)
        
        mobility_trend = self._detect_mobility_trend(patient_id, lookback_days)
        if mobility_trend:
            trends.append(mobility_trend)
        
        cognitive_trend = self._detect_cognitive_decline(patient_id, lookback_days)
        if cognitive_trend:
            trends.append(cognitive_trend)
        
        sentiment_trend = self._detect_sentiment_deterioration(patient_id, lookback_days)
        if sentiment_trend:
            trends.append(sentiment_trend)
        
        logger.info(f"Detected {len(trends)} deterioration trends for patient {patient_id}")
        
        return trends
    
    def _detect_behavioral_trend(
        self,
        patient_id: str,
        days: int
    ) -> Optional[Dict[str, Any]]:
        """Detect declining engagement trend"""
        from app.models.behavior_models import BehaviorMetric
        
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        metrics = self.db.query(BehaviorMetric).filter(
            and_(
                BehaviorMetric.patient_id == patient_id,
                BehaviorMetric.date >= cutoff_date
            )
        ).order_by(BehaviorMetric.date).all()
        
        if len(metrics) < 3:
            return None
        
        # Check check-in completion rate trend
        completion_rates = [float(m.checkin_completion_rate) for m in metrics if m.checkin_completion_rate]
        
        if len(completion_rates) < 3:
            return None
        
        # Linear regression
        x = np.arange(len(completion_rates))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, completion_rates)
        
        # Significant decline: slope < -0.05 and p < 0.05
        if slope < -0.05 and p_value < 0.05:
            
            # Z-score for severity
            z_score = abs(slope) / max(std_err, 0.01)
            
            if z_score > 3:
                severity = 'severe'
            elif z_score > 2:
                severity = 'moderate'
            else:
                severity = 'mild'
            
            return {
                'trend_type': 'declining_engagement',
                'severity': severity,
                'trend_start_date': metrics[0].date,
                'trend_duration_days': days,
                'trend_slope': float(slope),
                'z_score': float(z_score),
                'p_value': float(p_value),
                'confidence_level': 1.0 - p_value,
                'affected_metrics': ['checkin_completion_rate'],
                'metric_values': {'checkin_completion_rate': completion_rates},
                'clinical_significance': f"Check-in completion declining at {abs(slope)*100:.1f}% per day",
                'recommended_actions': [
                    'Increase check-in reminders',
                    'Schedule follow-up call',
                    'Review medication adherence'
                ]
            }
        
        return None
    
    def _detect_mobility_trend(
        self,
        patient_id: str,
        days: int
    ) -> Optional[Dict[str, Any]]:
        """Detect mobility drop trend"""
        from app.models.behavior_models import DigitalBiomarker
        
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        biomarkers = self.db.query(DigitalBiomarker).filter(
            and_(
                DigitalBiomarker.patient_id == patient_id,
                DigitalBiomarker.date >= cutoff_date
            )
        ).order_by(DigitalBiomarker.date).all()
        
        if len(biomarkers) < 3:
            return None
        
        # Check step count trend
        step_counts = [b.daily_step_count for b in biomarkers if b.daily_step_count]
        
        if len(step_counts) < 3:
            return None
        
        # Linear regression
        x = np.arange(len(step_counts))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, step_counts)
        
        # Significant decline: slope < -100 steps/day and p < 0.05
        if slope < -100 and p_value < 0.05:
            
            z_score = abs(slope) / max(std_err, 1.0)
            
            if z_score > 3:
                severity = 'severe'
            elif z_score > 2:
                severity = 'moderate'
            else:
                severity = 'mild'
            
            return {
                'trend_type': 'mobility_drop',
                'severity': severity,
                'trend_start_date': biomarkers[0].date,
                'trend_duration_days': days,
                'trend_slope': float(slope),
                'z_score': float(z_score),
                'p_value': float(p_value),
                'confidence_level': 1.0 - p_value,
                'affected_metrics': ['daily_step_count'],
                'metric_values': {'daily_step_count': step_counts},
                'clinical_significance': f"Daily steps declining by {abs(slope):.0f} steps/day",
                'recommended_actions': [
                    'Assess mobility limitations',
                    'Check for pain/discomfort',
                    'Consider physical therapy referral'
                ]
            }
        
        return None
    
    def _detect_cognitive_decline(
        self,
        patient_id: str,
        days: int
    ) -> Optional[Dict[str, Any]]:
        """Detect cognitive performance decline"""
        from app.models.behavior_models import CognitiveTest
        
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        tests = self.db.query(CognitiveTest).filter(
            and_(
                CognitiveTest.patient_id == patient_id,
                CognitiveTest.started_at >= cutoff_date
            )
        ).order_by(CognitiveTest.started_at).all()
        
        if len(tests) < 3:
            return None
        
        # Check reaction time trend (higher = worse)
        reaction_times = [t.reaction_time_ms for t in tests if t.reaction_time_ms]
        
        if len(reaction_times) < 3:
            return None
        
        # Linear regression
        x = np.arange(len(reaction_times))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, reaction_times)
        
        # Significant increase (worsening): slope > 50ms/test and p < 0.05
        if slope > 50 and p_value < 0.05:
            
            z_score = abs(slope) / max(std_err, 1.0)
            
            if z_score > 3:
                severity = 'severe'
            elif z_score > 2:
                severity = 'moderate'
            else:
                severity = 'mild'
            
            return {
                'trend_type': 'cognitive_decline',
                'severity': severity,
                'trend_start_date': tests[0].started_at,
                'trend_duration_days': days,
                'trend_slope': float(slope),
                'z_score': float(z_score),
                'p_value': float(p_value),
                'confidence_level': 1.0 - p_value,
                'affected_metrics': ['reaction_time_ms'],
                'metric_values': {'reaction_time_ms': reaction_times},
                'clinical_significance': f"Reaction time increasing by {slope:.0f}ms per test",
                'recommended_actions': [
                    'Cognitive assessment',
                    'Check medication side effects',
                    'Review sleep quality'
                ]
            }
        
        return None
    
    def _detect_sentiment_deterioration(
        self,
        patient_id: str,
        days: int
    ) -> Optional[Dict[str, Any]]:
        """Detect sentiment decline trend"""
        from app.models.behavior_models import SentimentAnalysis
        
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        analyses = self.db.query(SentimentAnalysis).filter(
            and_(
                SentimentAnalysis.patient_id == patient_id,
                SentimentAnalysis.analyzed_at >= cutoff_date
            )
        ).order_by(SentimentAnalysis.analyzed_at).all()
        
        if len(analyses) < 5:  # Need more data points for sentiment
            return None
        
        # Check sentiment polarity trend
        polarities = [float(a.sentiment_polarity) for a in analyses if a.sentiment_polarity]
        
        if len(polarities) < 5:
            return None
        
        # Linear regression
        x = np.arange(len(polarities))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, polarities)
        
        # Significant decline: slope < -0.05 and p < 0.05
        if slope < -0.05 and p_value < 0.05:
            
            z_score = abs(slope) / max(std_err, 0.01)
            
            if z_score > 3:
                severity = 'severe'
            elif z_score > 2:
                severity = 'moderate'
            else:
                severity = 'mild'
            
            return {
                'trend_type': 'sentiment_deterioration',
                'severity': severity,
                'trend_start_date': analyses[0].analyzed_at,
                'trend_duration_days': days,
                'trend_slope': float(slope),
                'z_score': float(z_score),
                'p_value': float(p_value),
                'confidence_level': 1.0 - p_value,
                'affected_metrics': ['sentiment_polarity'],
                'metric_values': {'sentiment_polarity': polarities},
                'clinical_significance': f"Sentiment declining at {abs(slope):.3f} points per message",
                'recommended_actions': [
                    'Mental health check-in',
                    'Review stressors',
                    'Consider counseling referral'
                ]
            }
        
        return None
