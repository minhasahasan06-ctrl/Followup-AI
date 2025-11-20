"""
Sentiment Analysis Service
==========================

Analyzes text inputs for sentiment and language biomarkers using DistilBERT.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from sqlalchemy.orm import Session

from app.services.behavior_ml_models import get_behavior_ml_models

logger = logging.getLogger(__name__)


class SentimentAnalysisService:
    """
    Service for sentiment analysis and language biomarker extraction
    
    Uses:
    - DistilBERT for sentiment classification
    - Rule-based extraction for linguistic features
    """
    
    def __init__(self, db: Session):
        self.db = db
        self.ml_models = get_behavior_ml_models()
    
    def analyze_text(
        self,
        patient_id: str,
        text: str,
        source_type: str,
        source_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze text for sentiment and language biomarkers
        
        Args:
            patient_id: Patient identifier
            text: Input text to analyze
            source_type: Type of source ('checkin', 'symptom_journal', 'chat', 'audio_transcript')
            source_id: ID of source record
        
        Returns:
            Complete sentiment analysis results
        """
        
        # Ensure ML models are loaded
        if not self.ml_models.models_loaded:
            self.ml_models.load_models()
        
        # Run sentiment analysis
        analysis_results = self.ml_models.analyze_sentiment(text)
        
        # Add metadata
        analysis_results['patient_id'] = patient_id
        analysis_results['source_type'] = source_type
        analysis_results['source_id'] = source_id
        analysis_results['text_content'] = text
        analysis_results['analyzed_at'] = datetime.utcnow()
        
        logger.info(
            f"Sentiment analysis for patient {patient_id}: "
            f"polarity={analysis_results['polarity']:.3f}, "
            f"label={analysis_results['label']}"
        )
        
        return analysis_results
    
    def get_patient_sentiment_trend(
        self,
        patient_id: str,
        days: int = 7
    ) -> Dict[str, Any]:
        """
        Get sentiment trend for patient over time
        
        Args:
            patient_id: Patient identifier
            days: Number of days to analyze
        
        Returns:
            Sentiment trend data
        """
        from app.models.behavior_models import SentimentAnalysis
        from datetime import timedelta
        from sqlalchemy import and_
        import numpy as np
        
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        analyses = self.db.query(SentimentAnalysis).filter(
            and_(
                SentimentAnalysis.patient_id == patient_id,
                SentimentAnalysis.analyzed_at >= cutoff_date
            )
        ).order_by(SentimentAnalysis.analyzed_at).all()
        
        if not analyses:
            return {
                'trend_direction': 'insufficient_data',
                'average_polarity': 0.0,
                'trend_slope': 0.0,
                'negative_ratio': 0.0
            }
        
        # Extract polarities
        polarities = [float(a.sentiment_polarity) for a in analyses if a.sentiment_polarity is not None]
        
        if len(polarities) < 2:
            return {
                'trend_direction': 'insufficient_data',
                'average_polarity': polarities[0] if polarities else 0.0,
                'trend_slope': 0.0,
                'negative_ratio': 1.0 if polarities and polarities[0] < 0 else 0.0
            }
        
        # Calculate trend
        x = np.arange(len(polarities))
        slope, _ = np.polyfit(x, polarities, 1)
        
        if slope < -0.05:
            trend_direction = 'declining'
        elif slope > 0.05:
            trend_direction = 'improving'
        else:
            trend_direction = 'stable'
        
        # Negative ratio
        negative_count = sum(1 for p in polarities if p < 0)
        negative_ratio = negative_count / len(polarities)
        
        return {
            'trend_direction': trend_direction,
            'average_polarity': float(np.mean(polarities)),
            'trend_slope': float(slope),
            'negative_ratio': float(negative_ratio),
            'recent_polarity': polarities[-1]
        }
    
    def detect_help_seeking(
        self,
        patient_id: str,
        days: int = 3
    ) -> Dict[str, Any]:
        """
        Detect recent help-seeking language
        
        Args:
            patient_id: Patient identifier
            days: Number of days to check
        
        Returns:
            Help-seeking detection results
        """
        from app.models.behavior_models import SentimentAnalysis
        from datetime import timedelta
        from sqlalchemy import and_
        
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        analyses = self.db.query(SentimentAnalysis).filter(
            and_(
                SentimentAnalysis.patient_id == patient_id,
                SentimentAnalysis.analyzed_at >= cutoff_date,
                SentimentAnalysis.help_seeking_detected == True
            )
        ).all()
        
        if not analyses:
            return {
                'help_seeking_detected': False,
                'occurrences': 0,
                'phrases': []
            }
        
        # Collect all help-seeking phrases
        all_phrases = []
        for analysis in analyses:
            if analysis.help_seeking_phrases:
                all_phrases.extend(analysis.help_seeking_phrases)
        
        return {
            'help_seeking_detected': True,
            'occurrences': len(analyses),
            'phrases': list(set(all_phrases)),
            'most_recent': analyses[-1].analyzed_at.isoformat() if analyses else None
        }
