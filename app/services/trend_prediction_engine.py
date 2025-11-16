"""
Trend Prediction Engine - "The Brain of the System"
Time-series risk modeling, Bayesian updates, anomaly detection, patient-specific personalization
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
import logging

try:
    from scipy import stats
    from scipy.signal import find_peaks
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import IsolationForest
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from app.models.trend_models import TrendSnapshot, RiskEvent, PatientBaseline
from app.models.video_ai_models import VideoMetrics
from app.models.audio_ai_models import AudioMetrics

logger = logging.getLogger(__name__)


class TrendPredictionEngine:
    """
    Production-grade trend prediction engine
    
    Output:
    - Risk level (Green/Yellow/Red)
    - Confidence score
    - What changed
    - Why risk increased
    """
    
    def __init__(self):
        # Risk thresholds
        self.RISK_THRESHOLDS = {
            'green': (0, 30),      # 0-30: Low risk
            'yellow': (30, 70),    # 30-70: Medium risk
            'red': (70, 100)       # 70-100: High risk
        }
        
        # Bayesian prior parameters
        self.PRIOR_ALPHA = 2.0  # Success count
        self.PRIOR_BETA = 8.0   # Failure count (assume low risk initially)
    
    async def compute_risk_assessment(
        self,
        db: Session,
        patient_id: str,
        current_measurements: Dict[str, Any],
        video_metrics_id: Optional[int] = None,
        audio_metrics_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Compute comprehensive risk assessment from all data sources
        
        Args:
            db: Database session
            patient_id: Patient identifier
            current_measurements: Current health measurements (symptoms, vitals, etc.)
            video_metrics_id: Optional video analysis ID
            audio_metrics_id: Optional audio analysis ID
        
        Returns:
            Risk assessment with level, score, confidence, explanations
        """
        logger.info(f"Computing risk assessment for patient {patient_id}")
        
        # 1. Get or create patient baseline
        baseline = self._get_patient_baseline(db, patient_id)
        
        # 2. Get historical trends
        historical_snapshots = self._get_historical_trends(db, patient_id, days=30)
        
        # 3. Compute rolling averages
        rolling_averages = self._compute_rolling_averages(
            historical_snapshots, windows=[7, 14, 30]
        )
        
        # 4. Detect anomalies
        anomaly_results = self._detect_anomalies(
            current_measurements, baseline, historical_snapshots
        )
        
        # 5. Calculate baseline deviations
        deviations = self._calculate_baseline_deviations(
            current_measurements, baseline
        )
        
        # 6. Compute time-series risk score
        time_series_risk = self._compute_time_series_risk(
            current_measurements, historical_snapshots, deviations
        )
        
        # 7. Apply Bayesian update
        bayesian_risk = self._apply_bayesian_update(
            time_series_risk, historical_snapshots
        )
        
        # 8. Patient-specific personalization
        personalized_risk = self._apply_personalization(
            bayesian_risk, patient_id, baseline, current_measurements
        )
        
        # 9. Determine risk level
        risk_level = self._classify_risk_level(personalized_risk)
        
        # 10. Compute confidence score
        confidence_score = self._compute_confidence(
            current_measurements, baseline, len(historical_snapshots)
        )
        
        # 11. Generate explanations
        explanation = self._generate_explanation(
            risk_level, deviations, anomaly_results, historical_snapshots
        )
        
        # 12. Identify changed metrics
        changed_metrics = self._identify_changed_metrics(deviations, threshold=2.0)
        
        # 13. Determine trend direction
        trend_direction, trend_velocity = self._compute_trend_direction(
            historical_snapshots
        )
        
        # Assemble final result
        result = {
            'risk_level': risk_level,
            'risk_score': float(personalized_risk),
            'confidence_score': float(confidence_score),
            
            # Explainability
            'changed_metrics': changed_metrics,
            'primary_concern': explanation['primary_concern'],
            'secondary_concerns': explanation['secondary_concerns'],
            'risk_explanation': explanation['detailed_explanation'],
            
            # Technical details
            'deviation_from_baseline': deviations,
            'anomaly_flags': anomaly_results,
            'rolling_average_7day': rolling_averages.get('7day', {}),
            'rolling_average_14day': rolling_averages.get('14day', {}),
            'rolling_average_30day': rolling_averages.get('30day', {}),
            
            # Bayesian update
            'prior_risk': float(time_series_risk),
            'posterior_risk': float(bayesian_risk),
            'likelihood_ratio': float(bayesian_risk / (time_series_risk + 1e-10)),
            
            # Trend analysis
            'trend_direction': trend_direction,
            'trend_velocity': float(trend_velocity),
            
            # Metadata
            'baseline_metrics': self._serialize_baseline(baseline) if baseline else {},
            'current_metrics': current_measurements,
            'video_metrics_id': video_metrics_id,
            'audio_metrics_id': audio_metrics_id,
            'model_version': '1.0.0',
            'algorithm_used': 'bayesian_time_series'
        }
        
        return result
    
    def _get_patient_baseline(
        self,
        db: Session,
        patient_id: str
    ) -> Optional[PatientBaseline]:
        """Get patient baseline or return None if not established"""
        return db.query(PatientBaseline).filter(
            PatientBaseline.patient_id == patient_id
        ).first()
    
    def _get_historical_trends(
        self,
        db: Session,
        patient_id: str,
        days: int = 30
    ) -> List[TrendSnapshot]:
        """Get historical trend snapshots"""
        since = datetime.utcnow() - timedelta(days=days)
        
        return db.query(TrendSnapshot).filter(
            TrendSnapshot.patient_id == patient_id,
            TrendSnapshot.snapshot_date >= since
        ).order_by(TrendSnapshot.snapshot_date.desc()).all()
    
    def _compute_rolling_averages(
        self,
        snapshots: List[TrendSnapshot],
        windows: List[int] = [7, 14, 30]
    ) -> Dict[str, Dict[str, float]]:
        """Compute rolling averages for different time windows"""
        result = {}
        
        for window in windows:
            recent_snapshots = snapshots[:window] if len(snapshots) >= window else snapshots
            
            if len(recent_snapshots) > 0:
                # Average risk scores
                avg_risk = np.mean([s.risk_score for s in recent_snapshots])
                avg_confidence = np.mean([s.confidence_score for s in recent_snapshots])
                
                result[f'{window}day'] = {
                    'average_risk_score': float(avg_risk),
                    'average_confidence': float(avg_confidence),
                    'sample_size': len(recent_snapshots)
                }
        
        return result
    
    def _detect_anomalies(
        self,
        current_measurements: Dict[str, Any],
        baseline: Optional[PatientBaseline],
        historical_snapshots: List[TrendSnapshot]
    ) -> Dict[str, Any]:
        """
        Detect anomalies using multiple methods
        - Statistical outliers (z-score)
        - Isolation Forest (if sklearn available)
        - Sudden spikes
        """
        anomalies = {
            'anomaly_detected': False,
            'anomaly_type': None,
            'anomaly_severity': 0.0,
            'anomalous_metrics': []
        }
        
        if not baseline:
            return anomalies
        
        # Z-score anomaly detection
        z_scores = {}
        anomalous_metrics = []
        
        for metric, value in current_measurements.items():
            if isinstance(value, (int, float)):
                baseline_mean = getattr(baseline, f'{metric}_baseline', None)
                baseline_std = getattr(baseline, f'{metric}_std', None)
                
                if baseline_mean is not None and baseline_std is not None and baseline_std > 0:
                    z_score = (value - baseline_mean) / baseline_std
                    z_scores[metric] = float(z_score)
                    
                    # Flag as anomalous if |z-score| > 3
                    if abs(z_score) > 3:
                        anomalous_metrics.append({
                            'metric': metric,
                            'z_score': float(z_score),
                            'current_value': float(value),
                            'baseline_mean': float(baseline_mean)
                        })
        
        # Check for sudden spikes (compared to recent trend)
        if len(historical_snapshots) >= 3:
            recent_risks = [s.risk_score for s in historical_snapshots[:3]]
            avg_recent_risk = np.mean(recent_risks)
            
            current_risk_estimate = self._estimate_current_risk(current_measurements)
            
            if current_risk_estimate > avg_recent_risk + 20:  # 20-point spike
                anomalies['anomaly_type'] = 'spike'
                anomalies['anomaly_severity'] = min(100, current_risk_estimate - avg_recent_risk)
        
        # Set anomaly status
        if len(anomalous_metrics) > 0:
            anomalies['anomaly_detected'] = True
            if not anomalies['anomaly_type']:
                anomalies['anomaly_type'] = 'outlier'
            anomalies['anomalous_metrics'] = anomalous_metrics
            anomalies['anomaly_severity'] = np.mean([abs(m['z_score']) for m in anomalous_metrics]) * 10
        
        return anomalies
    
    def _calculate_baseline_deviations(
        self,
        current_measurements: Dict[str, Any],
        baseline: Optional[PatientBaseline]
    ) -> Dict[str, float]:
        """Calculate z-scores for all metrics vs baseline"""
        deviations = {}
        
        if not baseline:
            return deviations
        
        for metric, value in current_measurements.items():
            if isinstance(value, (int, float)):
                baseline_mean = getattr(baseline, f'{metric}_baseline', None)
                baseline_std = getattr(baseline, f'{metric}_std', None)
                
                if baseline_mean is not None and baseline_std is not None and baseline_std > 0:
                    z_score = (value - baseline_mean) / baseline_std
                    deviations[metric] = float(z_score)
        
        return deviations
    
    def _compute_time_series_risk(
        self,
        current_measurements: Dict[str, Any],
        historical_snapshots: List[TrendSnapshot],
        deviations: Dict[str, float]
    ) -> float:
        """
        Compute risk score from time-series analysis
        Considers:
        - Absolute metric values
        - Deviations from baseline
        - Rate of change
        - Historical pattern
        """
        risk_score = 0.0
        
        # 1. Baseline deviation component (40% weight)
        if len(deviations) > 0:
            z_scores = list(deviations.values())
            avg_z_score = np.mean([abs(z) for z in z_scores])
            deviation_risk = min(40, avg_z_score * 10)
            risk_score += deviation_risk
        
        # 2. Rate of change component (30% weight)
        if len(historical_snapshots) >= 2:
            recent_risks = [s.risk_score for s in historical_snapshots[:7]]
            if len(recent_risks) > 1:
                # Linear regression to find trend
                x = np.arange(len(recent_risks))
                slope, _ = np.polyfit(x, recent_risks, 1)
                
                # Positive slope = worsening
                rate_risk = min(30, max(0, slope * 5))
                risk_score += rate_risk
        
        # 3. Absolute values component (30% weight)
        # Check for critical values
        critical_metrics_risk = self._assess_critical_metrics(current_measurements)
        risk_score += critical_metrics_risk
        
        return min(100, risk_score)
    
    def _apply_bayesian_update(
        self,
        prior_risk: float,
        historical_snapshots: List[TrendSnapshot]
    ) -> float:
        """
        Apply Bayesian update to risk estimate
        Uses Beta distribution for risk probability
        """
        if not SCIPY_AVAILABLE:
            return prior_risk
        
        # Prior: Beta distribution
        alpha = self.PRIOR_ALPHA
        beta = self.PRIOR_BETA
        
        # Update with historical evidence
        if len(historical_snapshots) > 0:
            # Count deterioration events (risk > 50)
            deterioration_events = sum(1 for s in historical_snapshots if s.risk_score > 50)
            stable_events = len(historical_snapshots) - deterioration_events
            
            # Update parameters
            alpha += deterioration_events
            beta += stable_events
        
        # Posterior mean
        posterior_mean = alpha / (alpha + beta)
        
        # Combine with current observation
        likelihood_weight = 0.7  # Weight current observation higher
        posterior_risk = likelihood_weight * prior_risk + (1 - likelihood_weight) * posterior_mean * 100
        
        return min(100, posterior_risk)
    
    def _apply_personalization(
        self,
        risk_score: float,
        patient_id: str,
        baseline: Optional[PatientBaseline],
        current_measurements: Dict[str, Any]
    ) -> float:
        """
        Apply patient-specific adjustments
        Account for individual variability, age, condition severity
        """
        adjusted_risk = risk_score
        
        # Placeholder for personalization factors
        # In production, would consider:
        # - Patient age
        # - Comorbidities
        # - Medication regimen
        # - Historical stability
        # - Immune status
        
        return adjusted_risk
    
    def _classify_risk_level(self, risk_score: float) -> str:
        """Classify numerical risk score into Green/Yellow/Red"""
        for level, (low, high) in self.RISK_THRESHOLDS.items():
            if low <= risk_score < high:
                return level
        return 'red'  # Default to highest risk
    
    def _compute_confidence(
        self,
        current_measurements: Dict[str, Any],
        baseline: Optional[PatientBaseline],
        historical_count: int
    ) -> float:
        """
        Compute confidence in risk assessment
        Based on data quality and quantity
        """
        confidence = 1.0
        
        # Reduce confidence if no baseline
        if not baseline:
            confidence *= 0.5
        
        # Reduce confidence if limited history
        if historical_count < 7:
            confidence *= (historical_count / 7)
        
        # Reduce confidence if missing key measurements
        required_metrics = ['respiratory_rate', 'pain_score', 'symptom_severity']
        missing_count = sum(1 for m in required_metrics if m not in current_measurements)
        if missing_count > 0:
            confidence *= (1 - missing_count / len(required_metrics) * 0.3)
        
        return max(0.1, min(1.0, confidence))
    
    def _generate_explanation(
        self,
        risk_level: str,
        deviations: Dict[str, float],
        anomaly_results: Dict[str, Any],
        historical_snapshots: List[TrendSnapshot]
    ) -> Dict[str, Any]:
        """
        Generate human-readable risk explanation
        Identifies primary and secondary concerns
        """
        # Find top deviating metrics
        sorted_deviations = sorted(
            deviations.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        # Primary concern (largest deviation)
        if len(sorted_deviations) > 0:
            primary_metric, primary_z = sorted_deviations[0]
            primary_concern = self._format_concern(primary_metric, primary_z)
        else:
            primary_concern = "No significant changes detected"
        
        # Secondary concerns
        secondary_concerns = [
            self._format_concern(metric, z_score)
            for metric, z_score in sorted_deviations[1:4]
            if abs(z_score) > 1.5
        ]
        
        # Detailed explanation
        if risk_level == 'red':
            explanation = f"⚠️ HIGH RISK: {primary_concern}. "
        elif risk_level == 'yellow':
            explanation = f"⚡ MEDIUM RISK: {primary_concern}. "
        else:
            explanation = f"✅ LOW RISK: Patient metrics are within normal range. "
        
        if anomaly_results['anomaly_detected']:
            explanation += f"Anomaly detected: {anomaly_results['anomaly_type']}. "
        
        if len(historical_snapshots) >= 3:
            recent_trend = np.mean([s.risk_score for s in historical_snapshots[:3]])
            if recent_trend > 50:
                explanation += "Risk has been elevated over the past few days. "
        
        return {
            'primary_concern': primary_concern,
            'secondary_concerns': secondary_concerns,
            'detailed_explanation': explanation
        }
    
    def _format_concern(self, metric: str, z_score: float) -> str:
        """Format a metric deviation as human-readable concern"""
        direction = "increased" if z_score > 0 else "decreased"
        magnitude = "significantly" if abs(z_score) > 2 else "moderately"
        
        # Clean metric name
        metric_name = metric.replace('_', ' ').title()
        
        return f"{metric_name} has {magnitude} {direction} (z={z_score:.1f})"
    
    def _identify_changed_metrics(
        self,
        deviations: Dict[str, float],
        threshold: float = 2.0
    ) -> List[str]:
        """Identify metrics that have changed significantly"""
        return [
            metric for metric, z_score in deviations.items()
            if abs(z_score) > threshold
        ]
    
    def _compute_trend_direction(
        self,
        snapshots: List[TrendSnapshot]
    ) -> Tuple[str, float]:
        """
        Compute trend direction and velocity
        Returns: (direction, velocity_per_day)
        """
        if len(snapshots) < 3:
            return "insufficient_data", 0.0
        
        # Get recent risk scores
        recent_risks = [s.risk_score for s in snapshots[:14]]  # Last 2 weeks
        
        # Linear regression
        x = np.arange(len(recent_risks))
        slope, intercept = np.polyfit(x, recent_risks, 1)
        
        # Classify direction
        if slope > 2:
            direction = "worsening"
        elif slope < -2:
            direction = "improving"
        elif np.std(recent_risks) > 10:
            direction = "fluctuating"
        else:
            direction = "stable"
        
        return direction, float(slope)
    
    def _estimate_current_risk(self, measurements: Dict[str, Any]) -> float:
        """Quick estimate of current risk for spike detection"""
        # Simplified risk estimation
        risk = 0.0
        
        if measurements.get('respiratory_rate', 0) > 24:
            risk += 20
        if measurements.get('pain_score', 0) > 7:
            risk += 20
        if measurements.get('symptom_severity', 0) > 8:
            risk += 20
        
        return risk
    
    def _assess_critical_metrics(self, measurements: Dict[str, Any]) -> float:
        """Assess if any metrics are in critical range"""
        critical_risk = 0.0
        
        # Respiratory rate
        rr = measurements.get('respiratory_rate', 0)
        if rr > 30 or rr < 8:
            critical_risk += 15
        elif rr > 24 or rr < 12:
            critical_risk += 10
        
        # Pain score
        pain = measurements.get('pain_score', 0)
        if pain >= 8:
            critical_risk += 10
        elif pain >= 6:
            critical_risk += 5
        
        # Symptom severity
        severity = measurements.get('symptom_severity', 0)
        if severity >= 9:
            critical_risk += 5
        
        return min(30, critical_risk)
    
    def _serialize_baseline(self, baseline: PatientBaseline) -> Dict[str, Any]:
        """Serialize baseline to dictionary"""
        return {
            'respiratory_rate': baseline.respiratory_rate_baseline,
            'heart_rate': baseline.heart_rate_baseline,
            'pain_score': baseline.pain_score_baseline,
            'skin_pallor': baseline.skin_pallor_baseline,
            'voice_pitch': baseline.voice_pitch_baseline
        }
    
    async def assess_risk(self, patient_id: str) -> Dict[str, Any]:
        """
        Comprehensive risk assessment for a patient
        
        This is the main API method called by endpoints
        Analyzes recent video/audio metrics, calculates deviations,
        and returns risk score with wellness recommendations
        
        Args:
            patient_id: Patient ID to assess
        
        Returns:
            Dict containing:
            - risk_score: float (0.0-1.0)
            - risk_level: str ('green', 'yellow', 'red')
            - confidence: float (0.0-1.0)
            - anomaly_count: int
            - deviation_metrics: Dict[str, Any]
            - contributing_factors: List[Dict[str, Any]]
            - wellness_recommendations: List[str]
        """
        from app.models.video_ai_models import VideoMetrics
        from app.models.audio_ai_models import AudioMetrics
        from datetime import timedelta
        
        # Get recent metrics (last 7 days)
        cutoff_date = datetime.utcnow() - timedelta(days=7)
        
        video_metrics = self.db.query(VideoMetrics).filter(
            VideoMetrics.patient_id == patient_id,
            VideoMetrics.analysis_timestamp >= cutoff_date
        ).all()
        
        audio_metrics = self.db.query(AudioMetrics).filter(
            AudioMetrics.patient_id == patient_id,
            AudioMetrics.analysis_timestamp >= cutoff_date
        ).all()
        
        # Get or create patient baseline
        baseline_data = await self._get_patient_baseline(patient_id)
        
        # Calculate deviations from baseline
        deviations = []
        anomaly_count = 0
        
        # Video metric deviations
        for metric in video_metrics[-5:]:  # Last 5 recordings
            if metric.respiratory_rate_bpm:
                baseline_rr = baseline_data.get('respiratory_rate', 16)
                z_score = abs((metric.respiratory_rate_bpm - baseline_rr) / max(baseline_rr * 0.15, 2))
                if z_score > 2.0:
                    anomaly_count += 1
                    deviations.append({
                        'metric': 'respiratory_rate',
                        'current': metric.respiratory_rate_bpm,
                        'baseline': baseline_rr,
                        'z_score': z_score,
                        'description': 'Respiratory rate deviation from baseline'
                    })
            
            if metric.skin_pallor_score and metric.skin_pallor_score > 0.6:
                anomaly_count += 1
                deviations.append({
                    'metric': 'skin_pallor',
                    'current': metric.skin_pallor_score,
                    'description': 'Elevated skin pallor score'
                })
        
        # Audio metric deviations
        for metric in audio_metrics[-5:]:
            if metric.cough_count and metric.cough_count > 5:
                anomaly_count += 1
                deviations.append({
                    'metric': 'cough_count',
                    'current': metric.cough_count,
                    'description': 'Frequent coughing detected'
                })
        
        # Calculate composite risk score (0.0-1.0)
        risk_score = min(1.0, anomaly_count * 0.15 + len(deviations) * 0.1)
        
        # Classify risk level
        if risk_score < 0.3:
            risk_level = "green"
        elif risk_score < 0.7:
            risk_level = "yellow"
        else:
            risk_level = "red"
        
        # Calculate confidence based on data availability
        total_metrics = len(video_metrics) + len(audio_metrics)
        confidence = min(1.0, total_metrics / 10)  # High confidence with 10+ metrics
        
        # Generate contributing factors
        contributing_factors = []
        for dev in deviations[:5]:  # Top 5 deviations
            contributing_factors.append({
                'factor': dev['metric'],
                'severity': 'high' if dev.get('z_score', 0) > 3 else 'medium',
                'description': dev['description']
            })
        
        # Generate wellness recommendations
        wellness_recommendations = self._generate_wellness_recommendations(
            risk_level, anomaly_count, deviations
        )
        
        return {
            'risk_score': float(risk_score),
            'risk_level': risk_level,
            'confidence': float(confidence),
            'anomaly_count': anomaly_count,
            'deviation_metrics': {dev['metric']: dev for dev in deviations},
            'contributing_factors': contributing_factors,
            'wellness_recommendations': wellness_recommendations
        }
    
    async def check_risk_transition(self, patient_id: str, new_risk_level: str) -> None:
        """
        Check if patient's risk level has changed and trigger alerts if needed
        
        Args:
            patient_id: Patient ID
            new_risk_level: New risk level ('green', 'yellow', 'red')
        """
        from app.models.trend_models import TrendSnapshot, RiskEvent
        
        # Get previous risk level
        previous_snapshot = self.db.query(TrendSnapshot).filter(
            TrendSnapshot.patient_id == patient_id
        ).order_by(TrendSnapshot.snapshot_timestamp.desc()).first()
        
        previous_risk_level = previous_snapshot.risk_level if previous_snapshot else "green"
        
        # Check if risk level changed
        if previous_risk_level != new_risk_level:
            # Determine event type
            risk_levels = {"green": 0, "yellow": 1, "red": 2}
            if risk_levels.get(new_risk_level, 0) > risk_levels.get(previous_risk_level, 0):
                event_type = "risk_increase"
            else:
                event_type = "risk_decrease"
            
            # Calculate risk delta
            risk_delta = risk_levels.get(new_risk_level, 0) - risk_levels.get(previous_risk_level, 0)
            
            # Create risk event
            risk_event = RiskEvent(
                patient_id=patient_id,
                event_type=event_type,
                previous_risk_level=previous_risk_level,
                new_risk_level=new_risk_level,
                risk_delta=float(risk_delta) * 0.5,  # Scale to 0-1
                triggered_alert=False,  # Alert engine will update this
                event_details={
                    'transition': f"{previous_risk_level} -> {new_risk_level}",
                    'timestamp': datetime.utcnow().isoformat()
                }
            )
            
            self.db.add(risk_event)
            self.db.commit()
            self.db.refresh(risk_event)
            
            # Trigger alert evaluation (imported to avoid circular dependency)
            try:
                from app.services.alert_orchestration_engine import AlertOrchestrationEngine
                
                alert_engine = AlertOrchestrationEngine(self.db)
                await alert_engine.evaluate_risk_event(risk_event, patient_id)
            except Exception as e:
                logger.error(f"Error triggering alerts for risk transition: {e}")
    
    def _generate_wellness_recommendations(
        self,
        risk_level: str,
        anomaly_count: int,
        deviations: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Generate wellness recommendations based on risk assessment
        
        IMPORTANT: Uses wellness language, NOT diagnostic language
        """
        recommendations = []
        
        if risk_level == "red":
            recommendations.append(
                "🔴 Wellness monitoring indicates notable changes in your health patterns. "
                "We strongly recommend contacting your healthcare provider soon to discuss these observations."
            )
        elif risk_level == "yellow":
            recommendations.append(
                "🟡 Wellness monitoring detected some changes in your patterns. "
                "Consider scheduling a wellness check with your healthcare provider to discuss these observations."
            )
        else:
            recommendations.append(
                "🟢 Wellness patterns appear stable. Continue regular monitoring to track your wellness journey."
            )
        
        # Specific recommendations based on deviations
        metrics_mentioned = set()
        for dev in deviations[:3]:  # Top 3 deviations
            metric = dev['metric']
            if metric not in metrics_mentioned:
                metrics_mentioned.add(metric)
                
                if 'respiratory' in metric:
                    recommendations.append(
                        "🫁 Breathing pattern changes detected. Discuss respiratory wellness with your provider."
                    )
                elif 'skin' in metric or 'pallor' in metric:
                    recommendations.append(
                        "🔍 Skin tone analysis detected changes. This may relate to circulation, hydration, or other wellness factors."
                    )
                elif 'cough' in metric:
                    recommendations.append(
                        "🤧 Cough frequency increased. If persistent, discuss with your healthcare team."
                    )
        
        # General wellness recommendations
        recommendations.append(
            "💡 Continue regular wellness monitoring to help you and your healthcare team track patterns over time."
        )
        
        recommendations.append(
            "ℹ️ This system provides wellness monitoring and change detection, not medical diagnosis. "
            "Always consult your healthcare provider for medical advice."
        )
        
        return recommendations


# Global instance
trend_engine = TrendPredictionEngine()
