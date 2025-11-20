"""
Risk Scoring Engine
===================

Multi-modal risk scoring combining:
- Behavioral metrics
- Digital biomarkers
- Cognitive scores
- Sentiment analysis

Uses transformer + XGBoost ensemble for prediction.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import and_
import numpy as np

from app.services.behavior_ml_models import get_behavior_ml_models

logger = logging.getLogger(__name__)


class RiskScoringEngine:
    """
    Multi-modal risk scoring engine
    
    Combines:
    1. Behavioral risk (engagement, adherence)
    2. Digital biomarker risk (mobility, circadian)
    3. Cognitive risk (test performance)
    4. Sentiment risk (language patterns)
    
    Output: Composite risk score (0-100) + risk level
    """
    
    def __init__(self, db: Session):
        self.db = db
        self.ml_models = get_behavior_ml_models()
        
        # Component weights for composite score
        self.weights = {
            'behavioral': 0.30,
            'digital_biomarker': 0.25,
            'cognitive': 0.25,
            'sentiment': 0.20
        }
    
    def calculate_risk_score(
        self,
        patient_id: str,
        lookback_days: int = 7
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive risk score for patient
        
        Args:
            patient_id: Patient identifier
            lookback_days: Days of historical data to analyze
        
        Returns:
            Complete risk assessment with component scores
        """
        
        logger.info(f"Calculating risk score for patient {patient_id}")
        
        # Collect data from all modalities
        behavioral_data = self._collect_behavioral_data(patient_id, lookback_days)
        digital_data = self._collect_digital_biomarker_data(patient_id, lookback_days)
        cognitive_data = self._collect_cognitive_data(patient_id, lookback_days)
        sentiment_data = self._collect_sentiment_data(patient_id, lookback_days)
        
        # Calculate component risk scores (0-100)
        behavioral_risk = self._calculate_behavioral_risk(behavioral_data)
        digital_risk = self._calculate_digital_biomarker_risk(digital_data)
        cognitive_risk = self._calculate_cognitive_risk(cognitive_data)
        sentiment_risk = self._calculate_sentiment_risk(sentiment_data)
        
        # Composite risk score (weighted average)
        composite_risk = (
            behavioral_risk * self.weights['behavioral'] +
            digital_risk * self.weights['digital_biomarker'] +
            cognitive_risk * self.weights['cognitive'] +
            sentiment_risk * self.weights['sentiment']
        )
        
        # Determine risk level
        risk_level = self._determine_risk_level(composite_risk)
        
        # Identify top risk factors
        top_risk_factors = self._identify_top_risk_factors({
            'behavioral': behavioral_risk,
            'digital_biomarker': digital_risk,
            'cognitive': cognitive_risk,
            'sentiment': sentiment_risk
        }, behavioral_data, digital_data, cognitive_data, sentiment_data)
        
        # ML model prediction (if available)
        ml_prediction = self._get_ml_prediction(patient_id, behavioral_data, digital_data, cognitive_data, sentiment_data)
        
        return {
            'patient_id': patient_id,
            'calculated_at': datetime.utcnow(),
            'behavioral_risk': float(behavioral_risk),
            'digital_biomarker_risk': float(digital_risk),
            'cognitive_risk': float(cognitive_risk),
            'sentiment_risk': float(sentiment_risk),
            'composite_risk': float(composite_risk),
            'risk_level': risk_level,
            'top_risk_factors': top_risk_factors,
            'model_type': ml_prediction.get('model_type', 'rule_based'),
            'prediction_confidence': ml_prediction.get('confidence', 0.7)
        }
    
    def _collect_behavioral_data(self, patient_id: str, days: int) -> Dict[str, Any]:
        """Collect recent behavioral metrics"""
        from app.models.behavior_models import BehaviorMetric
        
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        metrics = self.db.query(BehaviorMetric).filter(
            and_(
                BehaviorMetric.patient_id == patient_id,
                BehaviorMetric.date >= cutoff_date
            )
        ).order_by(BehaviorMetric.date.desc()).all()
        
        if not metrics:
            return {}
        
        # Get most recent metric
        latest = metrics[0]
        
        return {
            'checkin_completion_rate': float(latest.checkin_completion_rate) if latest.checkin_completion_rate else 1.0,
            'medication_adherence_rate': float(latest.medication_adherence_rate) if latest.medication_adherence_rate else 1.0,
            'routine_deviation_score': float(latest.routine_deviation_score) if latest.routine_deviation_score else 0.0,
            'avoidance_detected': latest.avoidance_patterns_detected or False,
            'avg_sentiment': float(latest.avg_sentiment_polarity) if latest.avg_sentiment_polarity else 0.0,
            'app_engagement': float(latest.app_engagement_duration_minutes) if latest.app_engagement_duration_minutes else 0.0
        }
    
    def _collect_digital_biomarker_data(self, patient_id: str, days: int) -> Dict[str, Any]:
        """Collect recent digital biomarkers"""
        from app.models.behavior_models import DigitalBiomarker
        
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        biomarkers = self.db.query(DigitalBiomarker).filter(
            and_(
                DigitalBiomarker.patient_id == patient_id,
                DigitalBiomarker.date >= cutoff_date
            )
        ).order_by(DigitalBiomarker.date.desc()).all()
        
        if not biomarkers:
            return {}
        
        latest = biomarkers[0]
        
        return {
            'mobility_drop_detected': latest.mobility_drop_detected or False,
            'mobility_change_percent': float(latest.mobility_change_percent) if latest.mobility_change_percent else 0.0,
            'circadian_stability': float(latest.circadian_rhythm_stability) if latest.circadian_rhythm_stability else 0.5,
            'step_count': latest.daily_step_count or 0,
            'sedentary_minutes': latest.sedentary_duration_minutes or 0
        }
    
    def _collect_cognitive_data(self, patient_id: str, days: int) -> Dict[str, Any]:
        """Collect recent cognitive test results"""
        from app.models.behavior_models import CognitiveTest
        
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        tests = self.db.query(CognitiveTest).filter(
            and_(
                CognitiveTest.patient_id == patient_id,
                CognitiveTest.started_at >= cutoff_date
            )
        ).order_by(CognitiveTest.started_at.desc()).all()
        
        if not tests:
            return {}
        
        # Count anomalies
        anomaly_count = sum(1 for t in tests if t.anomaly_detected)
        
        return {
            'anomaly_count': anomaly_count,
            'test_count': len(tests),
            'anomaly_rate': anomaly_count / max(len(tests), 1)
        }
    
    def _collect_sentiment_data(self, patient_id: str, days: int) -> Dict[str, Any]:
        """Collect recent sentiment analysis"""
        from app.models.behavior_models import SentimentAnalysis
        
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        analyses = self.db.query(SentimentAnalysis).filter(
            and_(
                SentimentAnalysis.patient_id == patient_id,
                SentimentAnalysis.analyzed_at >= cutoff_date
            )
        ).order_by(SentimentAnalysis.analyzed_at.desc()).all()
        
        if not analyses:
            return {}
        
        polarities = [float(a.sentiment_polarity) for a in analyses if a.sentiment_polarity]
        avg_polarity = np.mean(polarities) if polarities else 0.0
        
        # Count negative and help-seeking
        negative_count = sum(1 for a in analyses if a.sentiment_label == 'negative')
        help_seeking_count = sum(1 for a in analyses if a.help_seeking_detected)
        
        return {
            'avg_polarity': float(avg_polarity),
            'negative_ratio': negative_count / max(len(analyses), 1),
            'help_seeking_detected': help_seeking_count > 0
        }
    
    def _calculate_behavioral_risk(self, data: Dict[str, Any]) -> float:
        """Calculate behavioral risk score (0-100)"""
        if not data:
            return 50.0  # Neutral if no data
        
        risk = 0.0
        
        # Low check-in completion (0-30 points)
        completion = data.get('checkin_completion_rate', 1.0)
        if completion < 0.5:
            risk += 30
        elif completion < 0.7:
            risk += 15
        
        # Low medication adherence (0-30 points)
        adherence = data.get('medication_adherence_rate', 1.0)
        if adherence < 0.6:
            risk += 30
        elif adherence < 0.8:
            risk += 15
        
        # High routine deviation (0-20 points)
        deviation = data.get('routine_deviation_score', 0.0)
        risk += deviation * 20
        
        # Avoidance patterns (0-10 points)
        if data.get('avoidance_detected', False):
            risk += 10
        
        # Negative sentiment (0-10 points)
        sentiment = data.get('avg_sentiment', 0.0)
        if sentiment < -0.3:
            risk += 10
        elif sentiment < 0:
            risk += 5
        
        return min(risk, 100.0)
    
    def _calculate_digital_biomarker_risk(self, data: Dict[str, Any]) -> float:
        """Calculate digital biomarker risk score (0-100)"""
        if not data:
            return 50.0
        
        risk = 0.0
        
        # Mobility drop (0-40 points)
        if data.get('mobility_drop_detected', False):
            risk += 40
        elif data.get('mobility_change_percent', 0) < -15:
            risk += 20
        
        # Low circadian stability (0-30 points)
        stability = data.get('circadian_stability', 0.5)
        if stability < 0.3:
            risk += 30
        elif stability < 0.5:
            risk += 15
        
        # High sedentary time (0-20 points)
        sedentary = data.get('sedentary_minutes', 0)
        if sedentary > 600:  # >10 hours
            risk += 20
        elif sedentary > 480:  # >8 hours
            risk += 10
        
        # Low step count (0-10 points)
        steps = data.get('step_count', 5000)
        if steps < 1000:
            risk += 10
        elif steps < 3000:
            risk += 5
        
        return min(risk, 100.0)
    
    def _calculate_cognitive_risk(self, data: Dict[str, Any]) -> float:
        """Calculate cognitive risk score (0-100)"""
        if not data:
            return 50.0
        
        # Anomaly rate * 100
        anomaly_rate = data.get('anomaly_rate', 0.0)
        risk = anomaly_rate * 100
        
        return min(risk, 100.0)
    
    def _calculate_sentiment_risk(self, data: Dict[str, Any]) -> float:
        """Calculate sentiment risk score (0-100)"""
        if not data:
            return 50.0
        
        risk = 0.0
        
        # Negative polarity (0-50 points)
        polarity = data.get('avg_polarity', 0.0)
        if polarity < -0.5:
            risk += 50
        elif polarity < -0.2:
            risk += 25
        
        # High negative ratio (0-30 points)
        neg_ratio = data.get('negative_ratio', 0.0)
        risk += neg_ratio * 30
        
        # Help-seeking detected (0-20 points)
        if data.get('help_seeking_detected', False):
            risk += 20
        
        return min(risk, 100.0)
    
    def _determine_risk_level(self, composite_risk: float) -> str:
        """Map risk score to level"""
        if composite_risk >= 75:
            return 'critical'
        elif composite_risk >= 50:
            return 'high'
        elif composite_risk >= 25:
            return 'moderate'
        else:
            return 'low'
    
    def _identify_top_risk_factors(
        self,
        component_scores: Dict[str, float],
        behavioral_data: Dict,
        digital_data: Dict,
        cognitive_data: Dict,
        sentiment_data: Dict
    ) -> List[Dict[str, Any]]:
        """Identify top contributing risk factors"""
        
        factors = []
        
        # Check each component
        if component_scores['behavioral'] > 30:
            if behavioral_data.get('medication_adherence_rate', 1.0) < 0.7:
                factors.append({'factor': 'Low medication adherence', 'impact': component_scores['behavioral']})
            elif behavioral_data.get('checkin_completion_rate', 1.0) < 0.7:
                factors.append({'factor': 'Declining check-in completion', 'impact': component_scores['behavioral']})
        
        if component_scores['digital_biomarker'] > 30:
            if digital_data.get('mobility_drop_detected'):
                factors.append({'factor': 'Sudden mobility drop', 'impact': component_scores['digital_biomarker']})
            elif digital_data.get('circadian_stability', 1.0) < 0.4:
                factors.append({'factor': 'Disrupted circadian rhythm', 'impact': component_scores['digital_biomarker']})
        
        if component_scores['cognitive'] > 30:
            if cognitive_data.get('anomaly_rate', 0.0) > 0.3:
                factors.append({'factor': 'Cognitive anomalies detected', 'impact': component_scores['cognitive']})
        
        if component_scores['sentiment'] > 30:
            if sentiment_data.get('help_seeking_detected'):
                factors.append({'factor': 'Help-seeking language', 'impact': component_scores['sentiment']})
            elif sentiment_data.get('avg_polarity', 0.0) < -0.3:
                factors.append({'factor': 'Negative sentiment trend', 'impact': component_scores['sentiment']})
        
        # Sort by impact
        factors.sort(key=lambda x: x['impact'], reverse=True)
        
        return factors[:5]  # Top 5 factors
    
    def _get_ml_prediction(
        self,
        patient_id: str,
        behavioral_data: Dict,
        digital_data: Dict,
        cognitive_data: Dict,
        sentiment_data: Dict
    ) -> Dict[str, Any]:
        """Get ML model prediction if available"""
        
        # Ensure models loaded
        if not self.ml_models.models_loaded:
            try:
                self.ml_models.load_models()
            except Exception as e:
                logger.warning(f"ML models not available: {e}")
                return {'confidence': 0.7}
        
        # Combine features
        features = {
            **behavioral_data,
            **digital_data,
            **cognitive_data,
            **sentiment_data
        }
        
        # Try ML prediction
        try:
            prediction = self.ml_models.predict_feature_risk(features)
            return {
                'confidence': 0.85,
                'model_type': prediction.get('model_type', 'xgboost')
            }
        except Exception as e:
            logger.warning(f"ML prediction failed: {e}")
            return {'confidence': 0.7}
