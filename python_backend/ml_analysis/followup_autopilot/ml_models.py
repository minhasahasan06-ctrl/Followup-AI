"""
ML Models for Followup Autopilot

1. Risk Model (PyTorch LSTM) - Predict 7-day deterioration risks
2. Adherence Model (XGBoost) - Predict medication non-adherence
3. Anomaly Detector (IsolationForest) - Catch unusual days
4. Engagement Model (XGBoost) - Optimize notification timing

All models are for wellness monitoring only, NOT medical diagnosis.
"""

import os
import logging
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)


class RiskModel:
    """
    Sequence Risk Model (PyTorch LSTM/Transformer)
    
    Predicts 3 risk probabilities for next 7 days:
    - p_clinical_deterioration
    - p_mental_health_crisis  
    - p_non_adherence
    """
    
    RISK_WEIGHTS = {
        "clinical": 0.5,
        "mental_health": 0.3,
        "non_adherence": 0.2
    }
    
    def __init__(self):
        self.model = None
        self.device = "cpu"
        self.input_dim = 21
        self.hidden_dim = 64
        self.num_layers = 2
        self.output_dim = 3
        self.sequence_length = 30
        
    def load(self, model_path: Optional[str] = None) -> bool:
        """Load trained model from disk"""
        path = model_path or str(MODELS_DIR / "risk_model.pt")
        
        try:
            import torch
            import torch.nn as nn
            
            class LSTMRiskModel(nn.Module):
                def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
                    super().__init__()
                    self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, 
                                       batch_first=True, dropout=0.2)
                    self.fc = nn.Sequential(
                        nn.Linear(hidden_dim, 32),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(32, output_dim),
                        nn.Sigmoid()
                    )
                    
                def forward(self, x):
                    lstm_out, _ = self.lstm(x)
                    last_hidden = lstm_out[:, -1, :]
                    return self.fc(last_hidden)
            
            self.model = LSTMRiskModel(
                self.input_dim, self.hidden_dim, 
                self.num_layers, self.output_dim
            )
            
            if os.path.exists(path):
                state_dict = torch.load(path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                logger.info(f"Loaded risk model from {path}")
            else:
                logger.info("No trained model found, using untrained model")
                
            self.model.eval()
            return True
            
        except Exception as e:
            logger.error(f"Failed to load risk model: {e}")
            self.model = None
            return False
    
    def predict(self, feature_sequence: List[List[float]]) -> Dict[str, Any]:
        """
        Predict risk probabilities from 30-day feature sequence.
        
        Args:
            feature_sequence: List of 30 daily feature vectors
            
        Returns:
            Dictionary with risk predictions and components
        """
        if self.model is None:
            return self._fallback_prediction(feature_sequence)
            
        try:
            import torch
            
            seq = np.array(feature_sequence, dtype=np.float32)
            if len(seq) < self.sequence_length:
                padding = np.zeros((self.sequence_length - len(seq), self.input_dim))
                seq = np.vstack([padding, seq])
            seq = seq[-self.sequence_length:]
            
            x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)
            
            with torch.no_grad():
                probs = self.model(x).squeeze().numpy()
            
            p_clinical = float(probs[0])
            p_mh = float(probs[1])
            p_adherence = float(probs[2])
            
            risk_score = 100 * (
                self.RISK_WEIGHTS["clinical"] * p_clinical +
                self.RISK_WEIGHTS["mental_health"] * p_mh +
                self.RISK_WEIGHTS["non_adherence"] * p_adherence
            )
            
            return {
                "risk_score": round(risk_score, 2),
                "p_clinical_deterioration": round(p_clinical, 4),
                "p_mental_health_crisis": round(p_mh, 4),
                "p_non_adherence": round(p_adherence, 4),
                "risk_components": {
                    "clinical": round(p_clinical * 100, 1),
                    "mental_health": round(p_mh * 100, 1),
                    "adherence": round(p_adherence * 100, 1)
                },
                "confidence": 0.85,
                "model_version": "1.0.0"
            }
            
        except Exception as e:
            logger.error(f"Risk prediction failed: {e}")
            return self._fallback_prediction(feature_sequence)
    
    def _fallback_prediction(self, feature_sequence: List[List[float]]) -> Dict[str, Any]:
        """Rule-based fallback when model unavailable"""
        if not feature_sequence:
            return self._default_low_risk()
            
        recent = feature_sequence[-7:] if len(feature_sequence) >= 7 else feature_sequence
        
        avg_pain = np.mean([day[0] if len(day) > 0 else 0 for day in recent])
        avg_mh = np.mean([day[13] if len(day) > 13 else 0 for day in recent])
        avg_adherence = np.mean([day[12] if len(day) > 12 else 1.0 for day in recent])
        
        p_clinical = min(1.0, avg_pain / 10.0 * 0.5 + 0.1)
        p_mh = min(1.0, avg_mh * 0.7 + 0.1)
        p_adherence = min(1.0, (1 - avg_adherence) * 0.8 + 0.1)
        
        risk_score = 100 * (
            self.RISK_WEIGHTS["clinical"] * p_clinical +
            self.RISK_WEIGHTS["mental_health"] * p_mh +
            self.RISK_WEIGHTS["non_adherence"] * p_adherence
        )
        
        return {
            "risk_score": round(risk_score, 2),
            "p_clinical_deterioration": round(p_clinical, 4),
            "p_mental_health_crisis": round(p_mh, 4),
            "p_non_adherence": round(p_adherence, 4),
            "risk_components": {
                "clinical": round(p_clinical * 100, 1),
                "mental_health": round(p_mh * 100, 1),
                "adherence": round(p_adherence * 100, 1)
            },
            "confidence": 0.5,
            "model_version": "fallback"
        }
    
    def _default_low_risk(self) -> Dict[str, Any]:
        """Default low risk when no data available"""
        return {
            "risk_score": 15.0,
            "p_clinical_deterioration": 0.1,
            "p_mental_health_crisis": 0.1,
            "p_non_adherence": 0.1,
            "risk_components": {
                "clinical": 10.0,
                "mental_health": 10.0,
                "adherence": 10.0
            },
            "confidence": 0.3,
            "model_version": "default"
        }


class AdherenceModel:
    """
    Adherence Forecast Model (XGBoost)
    
    Predicts probability of medication non-adherence in next 7 days.
    """
    
    def __init__(self):
        self.model = None
        self.feature_names = [
            "adherence_mean_30d", "adherence_min_30d", "adherence_max_30d",
            "adherence_trend", "med_changes", "engagement_rate",
            "past_adherence_triggers", "avg_pain", "mh_score"
        ]
        
    def load(self, model_path: Optional[str] = None) -> bool:
        """Load trained model from disk"""
        path = model_path or str(MODELS_DIR / "adherence_xgb.pkl")
        
        try:
            if os.path.exists(path):
                import joblib
                self.model = joblib.load(path)
                logger.info(f"Loaded adherence model from {path}")
            else:
                logger.info("No trained adherence model found")
            return True
        except Exception as e:
            logger.error(f"Failed to load adherence model: {e}")
            return False
    
    def predict(self, features: Dict[str, float]) -> float:
        """
        Predict non-adherence probability.
        
        Args:
            features: Dictionary of feature values
            
        Returns:
            Probability of non-adherence (0-1)
        """
        if self.model is None:
            return self._fallback_prediction(features)
            
        try:
            feature_vector = [features.get(name, 0) for name in self.feature_names]
            prob = self.model.predict_proba([feature_vector])[0][1]
            return float(prob)
        except Exception as e:
            logger.error(f"Adherence prediction failed: {e}")
            return self._fallback_prediction(features)
    
    def _fallback_prediction(self, features: Dict[str, float]) -> float:
        """Rule-based fallback"""
        adherence_mean = features.get("adherence_mean_30d", 1.0)
        adherence_trend = features.get("adherence_trend", 0)
        
        base_prob = max(0, 1 - adherence_mean)
        trend_adjustment = -adherence_trend * 0.2
        
        return min(1.0, max(0.0, base_prob + trend_adjustment))


class AnomalyDetector:
    """
    Anomaly Detector (IsolationForest)
    
    Detects unusual days based on vitals/symptoms vector.
    """
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = [
            "avg_pain", "avg_fatigue", "avg_mood", "steps", "resting_hr",
            "sleep_hours", "env_risk_score", "mh_score", "video_resp_risk",
            "audio_emotion_score", "pain_severity_score"
        ]
        self.anomaly_threshold = 0.6
        
    def load(self, model_path: Optional[str] = None) -> bool:
        """Load trained model from disk"""
        path = model_path or str(MODELS_DIR / "anomaly_iforest.pkl")
        
        try:
            if os.path.exists(path):
                import joblib
                data = joblib.load(path)
                self.model = data.get("model")
                self.scaler = data.get("scaler")
                logger.info(f"Loaded anomaly model from {path}")
            else:
                logger.info("No trained anomaly model found")
            return True
        except Exception as e:
            logger.error(f"Failed to load anomaly model: {e}")
            return False
    
    def score(self, features: Dict[str, float]) -> float:
        """
        Compute anomaly score for a day's features.
        
        Args:
            features: Dictionary of feature values
            
        Returns:
            Anomaly score (0 = normal, 1 = highly anomalous)
        """
        if self.model is None:
            return self._fallback_score(features)
            
        try:
            feature_vector = np.array([[features.get(name, 0) for name in self.feature_names]])
            
            if self.scaler:
                feature_vector = self.scaler.transform(feature_vector)
            
            raw_score = self.model.decision_function(feature_vector)[0]
            anomaly_score = 1 / (1 + np.exp(raw_score))
            
            return float(anomaly_score)
        except Exception as e:
            logger.error(f"Anomaly scoring failed: {e}")
            return self._fallback_score(features)
    
    def _fallback_score(self, features: Dict[str, float]) -> float:
        """Rule-based fallback using z-score-like detection"""
        typical_ranges = {
            "avg_pain": (0, 5),
            "avg_fatigue": (2, 6),
            "resting_hr": (60, 80),
            "sleep_hours": (6, 9),
            "steps": (3000, 10000),
        }
        
        deviations = 0
        checks = 0
        
        for feature, (low, high) in typical_ranges.items():
            value = features.get(feature, (low + high) / 2)
            if value < low or value > high:
                deviation = abs(value - (low + high) / 2) / max(1, (high - low))
                deviations += min(1, deviation)
            checks += 1
            
        return min(1.0, deviations / max(1, checks))
    
    def is_anomaly(self, features: Dict[str, float]) -> bool:
        """Check if today is anomalous"""
        return self.score(features) > self.anomaly_threshold


class EngagementModel:
    """
    Engagement Time Model (XGBoost)
    
    Predicts optimal notification hour for task completion.
    """
    
    CANDIDATE_HOURS = [8, 9, 10, 12, 14, 16, 18, 20]
    
    def __init__(self):
        self.model = None
        self.feature_names = [
            "hour", "day_of_week", "risk_state_numeric",
            "engagement_rate_14d", "avg_pain", "mh_score"
        ]
        
    def load(self, model_path: Optional[str] = None) -> bool:
        """Load trained model from disk"""
        path = model_path or str(MODELS_DIR / "engagement_xgb.pkl")
        
        try:
            if os.path.exists(path):
                import joblib
                self.model = joblib.load(path)
                logger.info(f"Loaded engagement model from {path}")
            else:
                logger.info("No trained engagement model found")
            return True
        except Exception as e:
            logger.error(f"Failed to load engagement model: {e}")
            return False
    
    def predict_best_hour(
        self,
        patient_features: Dict[str, float],
        preferred_hour: Optional[int] = None
    ) -> int:
        """
        Predict optimal notification hour.
        
        Args:
            patient_features: Current patient features
            preferred_hour: Patient's preferred contact hour (0-23)
            
        Returns:
            Optimal hour (0-23)
        """
        if preferred_hour is not None:
            return preferred_hour
            
        if self.model is None:
            return self._fallback_best_hour(patient_features)
            
        try:
            best_hour = 9
            best_prob = 0
            
            day_of_week = datetime.now().weekday()
            
            for hour in self.CANDIDATE_HOURS:
                features = [
                    hour,
                    day_of_week,
                    patient_features.get("risk_state_numeric", 1),
                    patient_features.get("engagement_rate_14d", 0.5),
                    patient_features.get("avg_pain", 3),
                    patient_features.get("mh_score", 0.3)
                ]
                
                prob = self.model.predict_proba([features])[0][1]
                
                if prob > best_prob:
                    best_prob = prob
                    best_hour = hour
                    
            return best_hour
            
        except Exception as e:
            logger.error(f"Engagement prediction failed: {e}")
            return self._fallback_best_hour(patient_features)
    
    def _fallback_best_hour(self, patient_features: Dict[str, float]) -> int:
        """Rule-based fallback based on risk state"""
        risk_numeric = patient_features.get("risk_state_numeric", 1)
        
        if risk_numeric >= 3:
            return 9
        elif risk_numeric >= 2:
            return 10
        else:
            return 18


class ModelManager:
    """
    Singleton manager for all ML models.
    
    Handles lazy loading and provides unified access.
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self.risk_model = RiskModel()
        self.adherence_model = AdherenceModel()
        self.anomaly_detector = AnomalyDetector()
        self.engagement_model = EngagementModel()
        self._loaded = False
        self._initialized = True
        
    def load_all(self) -> bool:
        """Load all models"""
        if self._loaded:
            return True
            
        success = True
        success &= self.risk_model.load()
        success &= self.adherence_model.load()
        success &= self.anomaly_detector.load()
        success &= self.engagement_model.load()
        
        self._loaded = True
        return success
    
    def get_risk_model(self) -> RiskModel:
        return self.risk_model
    
    def get_adherence_model(self) -> AdherenceModel:
        return self.adherence_model
    
    def get_anomaly_detector(self) -> AnomalyDetector:
        return self.anomaly_detector
    
    def get_engagement_model(self) -> EngagementModel:
        return self.engagement_model


def get_model_manager() -> ModelManager:
    """Get singleton model manager instance"""
    manager = ModelManager()
    manager.load_all()
    return manager
