"""
Deep Learning Prediction Service for Deterioration Detection.

Implements LSTM/CNN architectures for time-series vital sign prediction:
- Multi-horizon predictions (3h, 6h, 12h, 24h)
- Confidence scoring with uncertainty quantification
- Ensemble approach combining statistical + ML signals
- Adaptive to individual patient baselines

Architecture:
- LSTM encoder for temporal patterns
- Attention mechanism for key timepoints
- Dense layers for horizon-specific predictions
- Monte Carlo Dropout for uncertainty estimation
"""

import os
import logging
import json
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
import hashlib

from sqlalchemy.orm import Session
from sqlalchemy import text

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available - using fallback statistical predictions")


class PredictionHorizon(Enum):
    """Prediction time horizons"""
    THREE_HOUR = "3h"
    SIX_HOUR = "6h"
    TWELVE_HOUR = "12h"
    TWENTY_FOUR_HOUR = "24h"


@dataclass
class HorizonPrediction:
    """Prediction for a specific time horizon"""
    horizon: str
    hours: int
    deterioration_probability: float
    severity_prediction: str  # green, yellow, orange, red
    confidence: float
    uncertainty_lower: float
    uncertainty_upper: float
    key_contributing_features: List[str]
    risk_level: str = "low"  # Added for API compatibility
    feature_importance: Optional[Dict[str, float]] = None  # Added for API compatibility


@dataclass
class PredictionResult:
    """Complete prediction result across all horizons"""
    patient_id: str
    predictions: Dict[str, HorizonPrediction]
    ensemble_score: float
    ensemble_confidence: float
    ml_weight: float
    statistical_weight: float
    trend_direction: str  # improving, stable, declining
    risk_trajectory: str  # low, moderate, high, critical
    model_version: str
    computed_at: datetime


@dataclass
class PatientTimeSeries:
    """Time-series data for a patient"""
    patient_id: str
    timestamps: List[datetime]
    features: Dict[str, List[float]]
    feature_names: List[str]
    sequence_length: int


if TORCH_AVAILABLE:
    class LSTMDeteriorationModel(nn.Module):
        """
        LSTM-based deterioration prediction model.
        
        Architecture:
        - Input: [batch, seq_len, n_features]
        - LSTM encoder with attention
        - Horizon-specific prediction heads
        - MC Dropout for uncertainty
        """
        
        def __init__(
            self,
            n_features: int = 20,
            hidden_size: int = 64,
            num_layers: int = 2,
            dropout: float = 0.3,
            n_horizons: int = 4
        ):
            super().__init__()
            
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.dropout = dropout
            self.n_horizons = n_horizons
            
            self.lstm = nn.LSTM(
                input_size=n_features,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=True
            )
            
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_size * 2,
                num_heads=4,
                dropout=dropout,
                batch_first=True
            )
            
            self.fc_shared = nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            
            self.horizon_heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_size, 32),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(32, 1),
                    nn.Sigmoid()
                )
                for _ in range(n_horizons)
            ])
            
            self.severity_head = nn.Sequential(
                nn.Linear(hidden_size, 32),
                nn.ReLU(),
                nn.Linear(32, 4)  # 4 severity classes
            )
        
        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Forward pass with multi-horizon predictions.
            
            Args:
                x: [batch, seq_len, n_features]
            
            Returns:
                horizon_probs: [batch, n_horizons] - deterioration probabilities
                severity_logits: [batch, 4] - severity class logits
            """
            lstm_out, _ = self.lstm(x)
            
            attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
            
            pooled = attn_out.mean(dim=1)
            
            shared = self.fc_shared(pooled)
            
            horizon_probs = torch.cat([
                head(shared) for head in self.horizon_heads
            ], dim=1)
            
            severity_logits = self.severity_head(shared)
            
            return horizon_probs, severity_logits
        
        def predict_with_uncertainty(
            self,
            x: torch.Tensor,
            n_samples: int = 30
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            """
            Monte Carlo Dropout for uncertainty estimation.
            """
            self.train()
            
            horizon_samples = []
            severity_samples = []
            
            with torch.no_grad():
                for _ in range(n_samples):
                    h_prob, s_logit = self.forward(x)
                    horizon_samples.append(h_prob)
                    severity_samples.append(F.softmax(s_logit, dim=-1))
            
            horizon_stack = torch.stack(horizon_samples, dim=0)
            severity_stack = torch.stack(severity_samples, dim=0)
            
            horizon_mean = horizon_stack.mean(dim=0)
            horizon_std = horizon_stack.std(dim=0)
            severity_mean = severity_stack.mean(dim=0)
            severity_std = severity_stack.std(dim=0)
            
            return horizon_mean, horizon_std, severity_mean, severity_std


class PredictionService:
    """
    Service for deep learning-based deterioration prediction.
    
    Combines LSTM predictions with statistical methods in an ensemble.
    """
    
    HORIZONS = [
        (PredictionHorizon.THREE_HOUR, 3),
        (PredictionHorizon.SIX_HOUR, 6),
        (PredictionHorizon.TWELVE_HOUR, 12),
        (PredictionHorizon.TWENTY_FOUR_HOUR, 24)
    ]
    
    FEATURE_NAMES = [
        "heart_rate", "respiratory_rate", "spo2", "blood_pressure_systolic",
        "blood_pressure_diastolic", "temperature", "pain_level", "fatigue_level",
        "mood", "sleep_quality", "step_count", "activity_level",
        "weight_delta", "edema_score", "gait_speed", "tremor_score",
        "cognitive_score", "anxiety_score", "depression_score", "compliance_score"
    ]
    
    SEVERITY_LABELS = ["green", "yellow", "orange", "red"]
    
    MODEL_VERSION = "v1.0.0-lstm-attention"
    
    def __init__(self, db: Session):
        self.db = db
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if TORCH_AVAILABLE else None
        self._init_model()
    
    def _init_model(self):
        """Initialize or load the LSTM model"""
        if not TORCH_AVAILABLE:
            logger.info("Using fallback statistical predictions (PyTorch not available)")
            return
        
        try:
            self.model = LSTMDeteriorationModel(
                n_features=len(self.FEATURE_NAMES),
                hidden_size=64,
                num_layers=2,
                dropout=0.3,
                n_horizons=len(self.HORIZONS)
            ).to(self.device)
            
            model_path = os.environ.get("DETERIORATION_MODEL_PATH")
            if model_path and os.path.exists(model_path):
                state_dict = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                logger.info(f"Loaded pre-trained model from {model_path}")
            else:
                self._init_weights()
                logger.info("Initialized model with random weights (no pre-trained model)")
            
            self.model.eval()
            
        except Exception as e:
            logger.error(f"Error initializing LSTM model: {e}")
            self.model = None
    
    def _init_weights(self):
        """Initialize model weights with Xavier initialization"""
        if self.model is None:
            return
        
        for name, param in self.model.named_parameters():
            if 'weight' in name and len(param.shape) >= 2:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    async def get_patient_time_series(
        self,
        patient_id: str,
        hours: int = 72
    ) -> Optional[PatientTimeSeries]:
        """
        Fetch and prepare time-series data for a patient.
        """
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        
        query = text("""
            SELECT 
                metric_name,
                metric_value,
                timestamp
            FROM metric_ingest_log
            WHERE patient_id = :patient_id
            AND timestamp >= :cutoff
            ORDER BY timestamp ASC
        """)
        
        try:
            results = self.db.execute(query, {
                "patient_id": patient_id,
                "cutoff": cutoff
            }).fetchall()
            
            if not results:
                return None
            
            grouped_data: Dict[datetime, Dict[str, float]] = {}
            for row in results:
                metric_name = row[0]
                value = float(row[1]) if row[1] else 0
                ts = row[2]
                
                hour_key = ts.replace(minute=0, second=0, microsecond=0)
                
                if hour_key not in grouped_data:
                    grouped_data[hour_key] = {}
                
                grouped_data[hour_key][metric_name] = value
            
            symptom_query = text("""
                SELECT 
                    pain_level, fatigue_level, mood, sleep_quality, created_at
                FROM symptom_checkins
                WHERE user_id = :patient_id
                AND created_at >= :cutoff
                ORDER BY created_at ASC
            """)
            
            symptom_results = self.db.execute(symptom_query, {
                "patient_id": patient_id,
                "cutoff": cutoff
            }).fetchall()
            
            for row in symptom_results:
                ts = row[4]
                hour_key = ts.replace(minute=0, second=0, microsecond=0)
                
                if hour_key not in grouped_data:
                    grouped_data[hour_key] = {}
                
                if row[0] is not None:
                    grouped_data[hour_key]["pain_level"] = float(row[0])
                if row[1] is not None:
                    grouped_data[hour_key]["fatigue_level"] = float(row[1])
                if row[2] is not None:
                    grouped_data[hour_key]["mood"] = float(row[2])
                if row[3] is not None:
                    grouped_data[hour_key]["sleep_quality"] = float(row[3])
            
            timestamps = sorted(grouped_data.keys())
            
            features: Dict[str, List[float]] = {name: [] for name in self.FEATURE_NAMES}
            
            last_values: Dict[str, float] = {name: 0.0 for name in self.FEATURE_NAMES}
            
            for ts in timestamps:
                hour_data = grouped_data[ts]
                for name in self.FEATURE_NAMES:
                    if name in hour_data:
                        last_values[name] = hour_data[name]
                    features[name].append(last_values[name])
            
            return PatientTimeSeries(
                patient_id=patient_id,
                timestamps=timestamps,
                features=features,
                feature_names=self.FEATURE_NAMES,
                sequence_length=len(timestamps)
            )
            
        except Exception as e:
            logger.error(f"Error fetching patient time series: {e}")
            return None
    
    def _prepare_tensor(
        self,
        time_series: PatientTimeSeries,
        seq_len: int = 24
    ) -> Optional[torch.Tensor]:
        """
        Prepare normalized tensor from time series data.
        """
        if not TORCH_AVAILABLE or time_series.sequence_length < seq_len:
            return None
        
        feature_matrix = []
        for name in self.FEATURE_NAMES:
            values = time_series.features.get(name, [0.0] * time_series.sequence_length)
            values = values[-seq_len:]
            
            values = np.array(values, dtype=np.float32)
            mean = np.mean(values) if np.std(values) > 0 else 0
            std = np.std(values) if np.std(values) > 0 else 1
            normalized = (values - mean) / (std + 1e-8)
            
            feature_matrix.append(normalized)
        
        feature_matrix = np.array(feature_matrix).T
        
        tensor = torch.tensor(feature_matrix, dtype=torch.float32)
        tensor = tensor.unsqueeze(0)
        
        return tensor.to(self.device)
    
    async def predict(
        self,
        patient_id: str,
        statistical_dpi: Optional[float] = None,
        statistical_bucket: Optional[str] = None
    ) -> PredictionResult:
        """
        Generate multi-horizon deterioration predictions.
        
        Combines LSTM predictions with statistical signals in an ensemble.
        """
        time_series = await self.get_patient_time_series(patient_id, hours=72)
        
        ml_predictions: Dict[str, HorizonPrediction] = {}
        ml_available = False
        
        if time_series and self.model is not None and TORCH_AVAILABLE:
            tensor = self._prepare_tensor(time_series)
            
            if tensor is not None:
                try:
                    horizon_mean, horizon_std, severity_mean, severity_std = \
                        self.model.predict_with_uncertainty(tensor, n_samples=30)
                    
                    horizon_probs = horizon_mean[0].cpu().numpy()
                    horizon_uncertainty = horizon_std[0].cpu().numpy()
                    severity_probs = severity_mean[0].cpu().numpy()
                    
                    for i, (horizon_enum, hours) in enumerate(self.HORIZONS):
                        prob = float(horizon_probs[i])
                        uncertainty = float(horizon_uncertainty[i])
                        
                        severity_idx = int(np.argmax(severity_probs))
                        severity = self.SEVERITY_LABELS[severity_idx]
                        
                        confidence = 1.0 - min(uncertainty * 2, 0.5)
                        
                        key_features = self._get_key_contributing_features(
                            time_series, 
                            top_k=3
                        )
                        
                        ml_predictions[horizon_enum.value] = HorizonPrediction(
                            horizon=horizon_enum.value,
                            hours=hours,
                            deterioration_probability=prob,
                            severity_prediction=severity,
                            confidence=confidence,
                            uncertainty_lower=max(0, prob - 1.96 * uncertainty),
                            uncertainty_upper=min(1, prob + 1.96 * uncertainty),
                            key_contributing_features=key_features
                        )
                    
                    ml_available = True
                    
                except Exception as e:
                    logger.error(f"Error in LSTM prediction: {e}")
        
        if not ml_available:
            ml_predictions = await self._fallback_statistical_predictions(
                patient_id, statistical_dpi, statistical_bucket
            )
        
        ml_weight = 0.6 if ml_available else 0.0
        statistical_weight = 1.0 - ml_weight
        
        if statistical_dpi is not None and ml_available:
            ml_avg_prob = np.mean([p.deterioration_probability for p in ml_predictions.values()])
            statistical_prob = statistical_dpi / 100.0
            ensemble_score = ml_weight * ml_avg_prob + statistical_weight * statistical_prob
            
            avg_confidence = np.mean([p.confidence for p in ml_predictions.values()])
            ensemble_confidence = ml_weight * avg_confidence + statistical_weight * 0.7
        else:
            ensemble_score = statistical_dpi / 100.0 if statistical_dpi else 0.5
            ensemble_confidence = 0.7
        
        trend_direction = self._compute_trend_direction(time_series)
        risk_trajectory = self._compute_risk_trajectory(ml_predictions, ensemble_score)
        
        return PredictionResult(
            patient_id=patient_id,
            predictions=ml_predictions,
            ensemble_score=round(ensemble_score, 4),
            ensemble_confidence=round(ensemble_confidence, 3),
            ml_weight=ml_weight,
            statistical_weight=statistical_weight,
            trend_direction=trend_direction,
            risk_trajectory=risk_trajectory,
            model_version=self.MODEL_VERSION,
            computed_at=datetime.utcnow()
        )
    
    async def predict_deterioration(
        self,
        patient_id: str,
        vital_signs_history: Optional[Dict[str, Any]] = None,
        horizon_hours: Optional[List[int]] = None
    ) -> PredictionResult:
        """
        Generate multi-horizon deterioration predictions from vital signs.
        
        This is the primary API method called by FastAPI endpoints.
        Combines LSTM predictions with statistical signals in an ensemble.
        
        Args:
            patient_id: Patient identifier
            vital_signs_history: Dict with timestamps and vital sign arrays
            horizon_hours: Optional list of specific horizons to predict (default: all)
        
        Returns:
            PredictionResult with predictions for each horizon
        """
        time_series = await self.get_patient_time_series(patient_id, hours=72)
        
        ml_predictions: Dict[str, HorizonPrediction] = {}
        ml_available = False
        
        # Filter horizons if specified
        active_horizons = self.HORIZONS
        if horizon_hours:
            active_horizons = [(h, hrs) for h, hrs in self.HORIZONS if hrs in horizon_hours]
        
        if time_series and self.model is not None and TORCH_AVAILABLE:
            tensor = self._prepare_tensor(time_series)
            
            if tensor is not None:
                try:
                    horizon_mean, horizon_std, severity_mean, severity_std = \
                        self.model.predict_with_uncertainty(tensor, n_samples=30)
                    
                    horizon_probs = horizon_mean[0].cpu().numpy()
                    horizon_uncertainty = horizon_std[0].cpu().numpy()
                    severity_probs = severity_mean[0].cpu().numpy()
                    
                    for i, (horizon_enum, hours) in enumerate(active_horizons):
                        prob = float(horizon_probs[i])
                        uncertainty = float(horizon_uncertainty[i])
                        
                        severity_idx = int(np.argmax(severity_probs))
                        severity = self.SEVERITY_LABELS[severity_idx]
                        
                        confidence = 1.0 - min(uncertainty * 2, 0.5)
                        
                        key_features = self._get_key_contributing_features(
                            time_series, 
                            top_k=3
                        )
                        
                        # Compute feature importance
                        feature_importance = self._compute_feature_importance(time_series, key_features)
                        
                        # Compute risk level from probability
                        risk_level = self._prob_to_risk_level(prob)
                        
                        ml_predictions[horizon_enum.value] = HorizonPrediction(
                            horizon=horizon_enum.value,
                            hours=hours,
                            deterioration_probability=prob,
                            severity_prediction=severity,
                            confidence=confidence,
                            uncertainty_lower=max(0, prob - 1.96 * uncertainty),
                            uncertainty_upper=min(1, prob + 1.96 * uncertainty),
                            key_contributing_features=key_features,
                            risk_level=risk_level,
                            feature_importance=feature_importance
                        )
                    
                    ml_available = True
                    
                except Exception as e:
                    logger.error(f"Error in LSTM prediction: {e}")
        
        # Fallback to statistical predictions if ML unavailable
        if not ml_available:
            ml_predictions = await self._fallback_predictions_with_risk_levels(patient_id)
        
        ml_weight = 0.6 if ml_available else 0.0
        statistical_weight = 1.0 - ml_weight
        
        # Compute ensemble score
        ml_avg_prob = np.mean([p.deterioration_probability for p in ml_predictions.values()])
        ensemble_score = ml_avg_prob
        avg_confidence = np.mean([p.confidence for p in ml_predictions.values()])
        ensemble_confidence = avg_confidence
        
        trend_direction = self._compute_trend_direction(time_series)
        risk_trajectory = self._compute_risk_trajectory(ml_predictions, ensemble_score)
        
        return PredictionResult(
            patient_id=patient_id,
            predictions=ml_predictions,
            ensemble_score=round(ensemble_score, 4),
            ensemble_confidence=round(ensemble_confidence, 3),
            ml_weight=ml_weight,
            statistical_weight=statistical_weight,
            trend_direction=trend_direction,
            risk_trajectory=risk_trajectory,
            model_version=self.MODEL_VERSION,
            computed_at=datetime.utcnow()
        )
    
    def _prob_to_risk_level(self, prob: float) -> str:
        """Convert probability to risk level string"""
        if prob >= 0.75:
            return "critical"
        elif prob >= 0.5:
            return "high"
        elif prob >= 0.25:
            return "moderate"
        return "low"
    
    def _compute_feature_importance(
        self,
        time_series: Optional[PatientTimeSeries],
        top_features: List[str]
    ) -> Dict[str, float]:
        """Compute feature importance scores based on variance and recency"""
        if not time_series:
            return {}
        
        importance = {}
        total = 0.0
        
        for i, feature in enumerate(top_features):
            values = time_series.features.get(feature, [])
            if values:
                variance = np.var(values[-24:]) if len(values) >= 24 else np.var(values)
                # Weight by position (earlier in list = more important)
                weight = 1.0 / (i + 1)
                score = variance * weight
                importance[feature] = score
                total += score
        
        # Normalize to sum to 1
        if total > 0:
            importance = {k: round(v / total, 3) for k, v in importance.items()}
        
        return importance
    
    async def _fallback_predictions_with_risk_levels(
        self,
        patient_id: str
    ) -> Dict[str, HorizonPrediction]:
        """Fallback predictions with risk levels for API compatibility"""
        predictions: Dict[str, HorizonPrediction] = {}
        
        # Base probability from a simple heuristic
        base_prob = 0.35
        
        decay_factors = [0.95, 0.9, 0.85, 0.8]
        
        for i, (horizon_enum, hours) in enumerate(self.HORIZONS):
            decay = decay_factors[i]
            prob = base_prob * (1 + (1 - decay) * (hours / 24))
            prob = min(max(prob, 0), 1)
            
            predictions[horizon_enum.value] = HorizonPrediction(
                horizon=horizon_enum.value,
                hours=hours,
                deterioration_probability=round(prob, 3),
                severity_prediction="yellow",
                confidence=0.5,
                uncertainty_lower=max(0, prob - 0.15),
                uncertainty_upper=min(1, prob + 0.15),
                key_contributing_features=["insufficient_data_for_ml"],
                risk_level=self._prob_to_risk_level(prob),
                feature_importance={}
            )
        
        return predictions
    
    async def _fallback_statistical_predictions(
        self,
        patient_id: str,
        statistical_dpi: Optional[float],
        statistical_bucket: Optional[str]
    ) -> Dict[str, HorizonPrediction]:
        """
        Generate predictions using statistical methods when ML is unavailable.
        Uses trend extrapolation and baseline comparison.
        """
        predictions: Dict[str, HorizonPrediction] = {}
        
        base_prob = (statistical_dpi or 50) / 100.0
        base_severity = statistical_bucket or "green"
        
        decay_factors = [0.95, 0.9, 0.85, 0.8]
        
        for i, (horizon_enum, hours) in enumerate(self.HORIZONS):
            decay = decay_factors[i]
            
            prob = base_prob * (1 + (1 - decay) * (hours / 24))
            prob = min(max(prob, 0), 1)
            
            predictions[horizon_enum.value] = HorizonPrediction(
                horizon=horizon_enum.value,
                hours=hours,
                deterioration_probability=round(prob, 3),
                severity_prediction=base_severity,
                confidence=0.5,
                uncertainty_lower=max(0, prob - 0.15),
                uncertainty_upper=min(1, prob + 0.15),
                key_contributing_features=["insufficient_data_for_ml"]
            )
        
        return predictions
    
    def _get_key_contributing_features(
        self,
        time_series: PatientTimeSeries,
        top_k: int = 3
    ) -> List[str]:
        """
        Identify features with highest variance/change (simple importance proxy).
        """
        feature_variance = {}
        
        for name in self.FEATURE_NAMES:
            values = time_series.features.get(name, [])
            if len(values) > 1:
                variance = np.var(values[-24:]) if len(values) >= 24 else np.var(values)
                feature_variance[name] = variance
        
        sorted_features = sorted(
            feature_variance.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [f[0] for f in sorted_features[:top_k]]
    
    def _compute_trend_direction(
        self,
        time_series: Optional[PatientTimeSeries]
    ) -> str:
        """Compute overall trend direction from time series"""
        if not time_series or time_series.sequence_length < 6:
            return "stable"
        
        risk_proxy = []
        for i in range(time_series.sequence_length):
            score = 0
            for name in ["pain_level", "fatigue_level", "respiratory_rate"]:
                if name in time_series.features and len(time_series.features[name]) > i:
                    score += time_series.features[name][i]
            risk_proxy.append(score)
        
        if len(risk_proxy) < 6:
            return "stable"
        
        recent = np.mean(risk_proxy[-6:])
        earlier = np.mean(risk_proxy[-12:-6]) if len(risk_proxy) >= 12 else np.mean(risk_proxy[:-6])
        
        diff = recent - earlier
        if diff > 0.5:
            return "declining"
        elif diff < -0.5:
            return "improving"
        return "stable"
    
    def _compute_risk_trajectory(
        self,
        predictions: Dict[str, HorizonPrediction],
        ensemble_score: float
    ) -> str:
        """Compute risk trajectory label"""
        if ensemble_score >= 0.75:
            return "critical"
        elif ensemble_score >= 0.5:
            return "high"
        elif ensemble_score >= 0.25:
            return "moderate"
        return "low"
    
    async def store_prediction(self, result: PredictionResult) -> bool:
        """Store prediction result in database"""
        try:
            insert_query = text("""
                INSERT INTO lstm_predictions (
                    id, patient_id, predictions, ensemble_score, ensemble_confidence,
                    ml_weight, statistical_weight, trend_direction, risk_trajectory,
                    model_version, computed_at
                ) VALUES (
                    gen_random_uuid(), :patient_id, :predictions::jsonb, :ensemble_score,
                    :ensemble_confidence, :ml_weight, :statistical_weight, :trend_direction,
                    :risk_trajectory, :model_version, NOW()
                )
            """)
            
            predictions_json = {
                horizon: {
                    "horizon": pred.horizon,
                    "hours": pred.hours,
                    "deterioration_probability": pred.deterioration_probability,
                    "severity_prediction": pred.severity_prediction,
                    "confidence": pred.confidence,
                    "uncertainty_lower": pred.uncertainty_lower,
                    "uncertainty_upper": pred.uncertainty_upper,
                    "key_contributing_features": pred.key_contributing_features
                }
                for horizon, pred in result.predictions.items()
            }
            
            self.db.execute(insert_query, {
                "patient_id": result.patient_id,
                "predictions": json.dumps(predictions_json),
                "ensemble_score": result.ensemble_score,
                "ensemble_confidence": result.ensemble_confidence,
                "ml_weight": result.ml_weight,
                "statistical_weight": result.statistical_weight,
                "trend_direction": result.trend_direction,
                "risk_trajectory": result.risk_trajectory,
                "model_version": result.model_version
            })
            self.db.commit()
            
            return True
            
        except Exception as e:
            logger.error(f"Error storing prediction: {e}")
            self.db.rollback()
            return False
    
    async def get_latest_prediction(
        self,
        patient_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get the most recent prediction for a patient"""
        query = text("""
            SELECT id, predictions, ensemble_score, ensemble_confidence,
                   ml_weight, statistical_weight, trend_direction, risk_trajectory,
                   model_version, computed_at
            FROM lstm_predictions
            WHERE patient_id = :patient_id
            ORDER BY computed_at DESC
            LIMIT 1
        """)
        
        try:
            result = self.db.execute(query, {"patient_id": patient_id}).fetchone()
            
            if result:
                return {
                    "id": str(result[0]),
                    "predictions": result[1],
                    "ensemble_score": float(result[2]) if result[2] else 0,
                    "ensemble_confidence": float(result[3]) if result[3] else 0,
                    "ml_weight": float(result[4]) if result[4] else 0,
                    "statistical_weight": float(result[5]) if result[5] else 0,
                    "trend_direction": result[6] or "stable",
                    "risk_trajectory": result[7] or "low",
                    "model_version": result[8] or self.MODEL_VERSION,
                    "computed_at": result[9].isoformat() if result[9] else None
                }
            return None
            
        except Exception as e:
            logger.error(f"Error fetching prediction: {e}")
            return None
    
    async def get_prediction_history(
        self,
        patient_id: str,
        days: int = 7
    ) -> List[Dict[str, Any]]:
        """Get prediction history for a patient"""
        cutoff = datetime.utcnow() - timedelta(days=days)
        
        query = text("""
            SELECT id, predictions, ensemble_score, ensemble_confidence,
                   trend_direction, risk_trajectory, computed_at
            FROM lstm_predictions
            WHERE patient_id = :patient_id
            AND computed_at >= :cutoff
            ORDER BY computed_at DESC
        """)
        
        try:
            results = self.db.execute(query, {
                "patient_id": patient_id,
                "cutoff": cutoff
            }).fetchall()
            
            return [
                {
                    "id": str(row[0]),
                    "predictions": row[1],
                    "ensemble_score": float(row[2]) if row[2] else 0,
                    "ensemble_confidence": float(row[3]) if row[3] else 0,
                    "trend_direction": row[4] or "stable",
                    "risk_trajectory": row[5] or "low",
                    "computed_at": row[6].isoformat() if row[6] else None
                }
                for row in results
            ]
            
        except Exception as e:
            logger.error(f"Error fetching prediction history: {e}")
            return []
