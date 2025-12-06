"""
ML-Assisted Alert Ranking Service - XGBoost-based alert prioritization.

Purpose: Reduce false positives and prioritize alerts for clinician attention.
NOT used for diagnosis or prediction - only for ordering alerts.

Features:
- XGBoost ranking model trained on historical clinician actions
- SHAP explanations for transparency
- Priority score (0-1) for alert ordering
- Feature extraction from metrics, DPI, adherence, volatility
"""

import os
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import pickle
import numpy as np

from sqlalchemy.orm import Session
from sqlalchemy import text

logger = logging.getLogger(__name__)

# Optional ML imports
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("XGBoost not available for ML ranking")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP not available for explanations")

from .config_service import AlertConfigService


@dataclass
class AlertFeatures:
    """Features extracted for ML ranking"""
    alert_id: str
    
    # DPI and organ scores
    dpi_score: float
    dpi_bucket_numeric: int  # green=0, yellow=1, orange=2, red=3
    respiratory_score: float
    cardio_score: float
    hepatic_score: float
    mobility_score: float
    cognitive_score: float
    
    # Z-scores
    max_z_score: float
    avg_z_score: float
    num_elevated_metrics: int
    
    # Patient context
    baseline_volatility: float
    adherence_score: float
    recent_alert_count: int
    days_since_last_alert: float
    
    # Temporal
    hour_of_day: int
    day_of_week: int
    
    # Alert properties
    severity_numeric: int  # low=0, moderate=1, high=2, critical=3
    is_corroborated: bool
    trigger_rule_numeric: int


@dataclass
class RankingResult:
    """Result of ML ranking"""
    alert_id: str
    priority_score: float  # 0-1, higher = more urgent
    confidence: float
    feature_importances: Dict[str, float]
    top_features: List[Tuple[str, float]]
    explanation: str


class MLRankingService:
    """Service for ML-assisted alert ranking"""
    
    def __init__(self, db: Session):
        self.db = db
        self.config_service = AlertConfigService()
        self.model = None
        self.explainer = None
        self._model_loaded = False
        
        # Feature names for model
        self.feature_names = [
            "dpi_score", "dpi_bucket_numeric", "respiratory_score", "cardio_score",
            "hepatic_score", "mobility_score", "cognitive_score", "max_z_score",
            "avg_z_score", "num_elevated_metrics", "baseline_volatility",
            "adherence_score", "recent_alert_count", "days_since_last_alert",
            "hour_of_day", "day_of_week", "severity_numeric", "is_corroborated",
            "trigger_rule_numeric"
        ]
        
        # Rule type to numeric mapping
        self.rule_mapping = {
            "risk_jump": 0,
            "persistent_yellow": 1,
            "any_red": 2,
            "respiratory_spike": 3,
            "checkin_deviation": 4,
            "composite_jump": 5,
            "multi_signal_corroboration": 6
        }
        
        # Severity to numeric mapping
        self.severity_mapping = {
            "low": 0,
            "moderate": 1,
            "high": 2,
            "critical": 3
        }
    
    async def initialize(self):
        """Initialize or load the ranking model"""
        if not XGBOOST_AVAILABLE:
            logger.warning("XGBoost not available - using rule-based fallback")
            return
        
        config = self.config_service.config
        if not config.ml_ranking_enabled:
            logger.info("ML ranking disabled in config")
            return
        
        try:
            # Try to load pre-trained model
            model_path = os.path.join(os.path.dirname(__file__), "models", "alert_ranker.pkl")
            
            if os.path.exists(model_path):
                with open(model_path, "rb") as f:
                    self.model = pickle.load(f)
                logger.info("Loaded pre-trained alert ranking model")
            else:
                # Create a default model with reasonable parameters
                self.model = xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.1,
                    objective='binary:logistic',
                    random_state=42
                )
                logger.info("Created default XGBoost model (untrained)")
            
            # Initialize SHAP explainer if available
            if SHAP_AVAILABLE and self.model is not None:
                try:
                    self.explainer = shap.TreeExplainer(self.model)
                except Exception as e:
                    logger.warning(f"SHAP explainer initialization failed: {e}")
            
            self._model_loaded = True
            
        except Exception as e:
            logger.error(f"Error initializing ML ranking model: {e}")
            self._model_loaded = False
    
    async def rank_alerts(
        self,
        alerts: List[Dict[str, Any]],
        patient_context: Dict[str, Any] = None
    ) -> List[RankingResult]:
        """
        Rank a list of alerts by priority.
        Returns alerts sorted by priority_score descending.
        """
        if not self._model_loaded or not XGBOOST_AVAILABLE:
            # Fallback to rule-based ranking
            return await self._rule_based_ranking(alerts)
        
        results = []
        
        for alert in alerts:
            features = await self._extract_features(alert, patient_context)
            result = await self._predict_priority(features)
            results.append(result)
        
        # Sort by priority score descending
        results.sort(key=lambda x: x.priority_score, reverse=True)
        
        return results
    
    async def _extract_features(
        self,
        alert: Dict[str, Any],
        patient_context: Dict[str, Any] = None
    ) -> AlertFeatures:
        """Extract features from an alert for ML prediction"""
        patient_context = patient_context or {}
        
        # Get organ scores from alert or context
        organ_scores = alert.get("organ_scores", {})
        
        # Get trigger metrics for z-scores
        trigger_metrics = alert.get("trigger_metrics", [])
        z_scores = [m.get("z_score", 0) for m in trigger_metrics if "z_score" in m]
        
        # Calculate recent alert count
        recent_count = patient_context.get("recent_alert_count", 0)
        days_since_last = patient_context.get("days_since_last_alert", 30)
        
        now = datetime.utcnow()
        
        return AlertFeatures(
            alert_id=alert.get("id", ""),
            
            # DPI and organ scores
            dpi_score=alert.get("dpi_at_trigger", 50),
            dpi_bucket_numeric=self._bucket_to_numeric(alert.get("dpi_bucket", "green")),
            respiratory_score=organ_scores.get("respiratory", organ_scores.get("Respiratory", 50)),
            cardio_score=organ_scores.get("cardio_fluid", organ_scores.get("Cardio/Fluid", 50)),
            hepatic_score=organ_scores.get("hepatic_hematologic", organ_scores.get("Hepatic/Hematologic", 50)),
            mobility_score=organ_scores.get("mobility", organ_scores.get("Mobility", 50)),
            cognitive_score=organ_scores.get("cognitive_behavioral", organ_scores.get("Cognitive/Behavioral", 50)),
            
            # Z-scores
            max_z_score=max(z_scores) if z_scores else 0,
            avg_z_score=np.mean(z_scores) if z_scores else 0,
            num_elevated_metrics=len([z for z in z_scores if abs(z) >= 2.0]),
            
            # Patient context
            baseline_volatility=patient_context.get("baseline_volatility", 1.0),
            adherence_score=patient_context.get("adherence_score", 70),
            recent_alert_count=recent_count,
            days_since_last_alert=days_since_last,
            
            # Temporal
            hour_of_day=now.hour,
            day_of_week=now.weekday(),
            
            # Alert properties
            severity_numeric=self.severity_mapping.get(alert.get("severity", "moderate"), 1),
            is_corroborated=alert.get("corroborated", False),
            trigger_rule_numeric=self.rule_mapping.get(alert.get("trigger_rule", ""), 0)
        )
    
    def _bucket_to_numeric(self, bucket: str) -> int:
        """Convert DPI bucket to numeric"""
        mapping = {"green": 0, "yellow": 1, "orange": 2, "red": 3}
        return mapping.get(bucket, 0)
    
    async def _predict_priority(self, features: AlertFeatures) -> RankingResult:
        """Predict priority score using ML model"""
        # Convert features to array
        feature_array = np.array([[
            features.dpi_score,
            features.dpi_bucket_numeric,
            features.respiratory_score,
            features.cardio_score,
            features.hepatic_score,
            features.mobility_score,
            features.cognitive_score,
            features.max_z_score,
            features.avg_z_score,
            features.num_elevated_metrics,
            features.baseline_volatility,
            features.adherence_score,
            features.recent_alert_count,
            features.days_since_last_alert,
            features.hour_of_day,
            features.day_of_week,
            features.severity_numeric,
            1 if features.is_corroborated else 0,
            features.trigger_rule_numeric
        ]])
        
        try:
            # Get probability prediction
            if hasattr(self.model, 'predict_proba'):
                proba = self.model.predict_proba(feature_array)
                priority_score = float(proba[0][1]) if len(proba[0]) > 1 else float(proba[0][0])
            else:
                # Fallback to prediction
                pred = self.model.predict(feature_array)
                priority_score = float(pred[0])
            
            # Get feature importances
            feature_importances = {}
            top_features = []
            
            if self.explainer is not None and SHAP_AVAILABLE:
                try:
                    shap_values = self.explainer.shap_values(feature_array)
                    if isinstance(shap_values, list):
                        shap_values = shap_values[1]  # For binary classification
                    
                    for i, name in enumerate(self.feature_names):
                        feature_importances[name] = float(shap_values[0][i])
                    
                    # Get top features
                    sorted_features = sorted(
                        feature_importances.items(),
                        key=lambda x: abs(x[1]),
                        reverse=True
                    )
                    top_features = sorted_features[:5]
                except Exception as e:
                    logger.warning(f"SHAP explanation failed: {e}")
            
            # Generate explanation
            explanation = self._generate_explanation(features, priority_score, top_features)
            
            return RankingResult(
                alert_id=features.alert_id,
                priority_score=round(priority_score, 4),
                confidence=0.85,  # Placeholder confidence
                feature_importances=feature_importances,
                top_features=top_features,
                explanation=explanation
            )
            
        except Exception as e:
            logger.error(f"Error in ML prediction: {e}")
            # Fallback to rule-based score
            return await self._fallback_priority(features)
    
    async def _fallback_priority(self, features: AlertFeatures) -> RankingResult:
        """Calculate priority using rule-based fallback"""
        # Simple weighted score
        score = 0.0
        
        # Severity weight (0-0.4)
        score += features.severity_numeric * 0.1 + 0.1
        
        # DPI bucket weight (0-0.3)
        score += features.dpi_bucket_numeric * 0.075 + 0.075
        
        # Z-score weight (0-0.2)
        score += min(features.max_z_score / 5.0, 0.2)
        
        # Corroboration bonus
        if features.is_corroborated:
            score += 0.1
        
        return RankingResult(
            alert_id=features.alert_id,
            priority_score=min(round(score, 4), 1.0),
            confidence=0.6,  # Lower confidence for rule-based
            feature_importances={},
            top_features=[],
            explanation="Priority calculated using rule-based scoring (ML model not available)"
        )
    
    async def _rule_based_ranking(
        self,
        alerts: List[Dict[str, Any]]
    ) -> List[RankingResult]:
        """Fallback rule-based ranking when ML is not available"""
        results = []
        
        for alert in alerts:
            features = await self._extract_features(alert, {})
            result = await self._fallback_priority(features)
            results.append(result)
        
        results.sort(key=lambda x: x.priority_score, reverse=True)
        return results
    
    def _generate_explanation(
        self,
        features: AlertFeatures,
        priority_score: float,
        top_features: List[Tuple[str, float]]
    ) -> str:
        """Generate human-readable explanation for priority"""
        if not top_features:
            return f"Priority score: {priority_score:.2f} (ML explanation not available)"
        
        explanations = []
        
        for name, importance in top_features[:3]:
            if importance > 0:
                direction = "increases"
            else:
                direction = "decreases"
            
            readable_name = name.replace("_", " ").title()
            explanations.append(f"{readable_name} {direction} priority")
        
        return f"Priority score: {priority_score:.2f}. Key factors: {'; '.join(explanations)}."
    
    async def train_model(
        self,
        training_data: List[Dict[str, Any]],
        labels: List[int]
    ) -> bool:
        """
        Train the ranking model on historical data.
        
        training_data: List of alert features
        labels: 1 if clinician acknowledged quickly, 0 if dismissed/ignored
        """
        if not XGBOOST_AVAILABLE:
            logger.error("XGBoost not available for training")
            return False
        
        try:
            # Convert training data to feature arrays
            X = []
            for data in training_data:
                features = await self._extract_features(data, {})
                X.append([
                    features.dpi_score, features.dpi_bucket_numeric,
                    features.respiratory_score, features.cardio_score,
                    features.hepatic_score, features.mobility_score,
                    features.cognitive_score, features.max_z_score,
                    features.avg_z_score, features.num_elevated_metrics,
                    features.baseline_volatility, features.adherence_score,
                    features.recent_alert_count, features.days_since_last_alert,
                    features.hour_of_day, features.day_of_week,
                    features.severity_numeric, 1 if features.is_corroborated else 0,
                    features.trigger_rule_numeric
                ])
            
            X = np.array(X)
            y = np.array(labels)
            
            # Train model
            self.model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                objective='binary:logistic',
                random_state=42
            )
            self.model.fit(X, y)
            
            # Update SHAP explainer
            if SHAP_AVAILABLE:
                self.explainer = shap.TreeExplainer(self.model)
            
            # Save model
            model_dir = os.path.join(os.path.dirname(__file__), "models")
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, "alert_ranker.pkl")
            
            with open(model_path, "wb") as f:
                pickle.dump(self.model, f)
            
            logger.info("ML ranking model trained and saved")
            return True
            
        except Exception as e:
            logger.error(f"Error training ML model: {e}")
            return False
    
    async def get_model_metrics(self) -> Dict[str, Any]:
        """Get model performance metrics"""
        if not self._model_loaded or self.model is None:
            return {"status": "not_loaded", "metrics": None}
        
        return {
            "status": "loaded",
            "model_type": "XGBoost",
            "feature_count": len(self.feature_names),
            "shap_available": SHAP_AVAILABLE and self.explainer is not None
        }
