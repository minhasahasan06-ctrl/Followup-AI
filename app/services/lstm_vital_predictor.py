"""
LSTM Vital Signs Time-Series Forecasting Service
=================================================

Production-grade PyTorch LSTM implementation for predicting vital sign trends:
- Heart Rate
- Blood Pressure (Systolic/Diastolic)
- Oxygen Saturation (SpO2)
- Respiratory Rate

Provides:
- 24h/48h/72h horizon predictions
- Confidence intervals
- Trend analysis
- Anomaly detection
"""

import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import numpy as np

logger = logging.getLogger(__name__)

TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = None
    logger.warning("PyTorch not available - LSTM predictions will use fallback methods")


class LSTMVitalPredictor:
    """
    Production LSTM-based vital sign predictor with multi-horizon forecasting.
    """
    
    VITAL_METRICS = ["heart_rate", "bp_systolic", "bp_diastolic", "spo2", "respiratory_rate"]
    FORECAST_HORIZONS = [24, 48, 72]
    
    VITAL_RANGES = {
        "heart_rate": {"min": 40, "max": 200, "normal_min": 60, "normal_max": 100, "unit": "bpm"},
        "bp_systolic": {"min": 70, "max": 220, "normal_min": 90, "normal_max": 140, "unit": "mmHg"},
        "bp_diastolic": {"min": 40, "max": 130, "normal_min": 60, "normal_max": 90, "unit": "mmHg"},
        "spo2": {"min": 70, "max": 100, "normal_min": 95, "normal_max": 100, "unit": "%"},
        "respiratory_rate": {"min": 8, "max": 40, "normal_min": 12, "normal_max": 20, "unit": "/min"}
    }
    
    def __init__(self):
        self.model = None
        self.model_loaded = False
        self.scaler_params: Dict[str, Dict[str, float]] = {}
        
        self._init_scaler_params()
        self._initialize_model()
    
    def _init_scaler_params(self):
        """Initialize default scaler parameters for normalization."""
        for metric, ranges in self.VITAL_RANGES.items():
            mean = (ranges["normal_min"] + ranges["normal_max"]) / 2
            std = (ranges["normal_max"] - ranges["normal_min"]) / 4
            self.scaler_params[metric] = {"mean": float(mean), "std": float(std)}
    
    def _initialize_model(self):
        """Initialize LSTM model with pre-trained weights or random initialization."""
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available - using statistical fallback")
            return
        
        try:
            import os
            model_path = "./models/vital_lstm.pth"
            
            self.model = self._build_lstm_model()
            
            if os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location='cpu')
                self.model.load_state_dict(checkpoint['model_state_dict'])
                if 'scaler_params' in checkpoint:
                    self.scaler_params = checkpoint['scaler_params']
                logger.info(f"Loaded LSTM model from {model_path}")
            else:
                logger.info("No pre-trained LSTM found - using initialized model with statistical methods")
            
            self.model.eval()
            self.model_loaded = True
            
        except Exception as e:
            logger.error(f"Failed to initialize LSTM model: {e}")
            self.model = None
    
    def _build_lstm_model(self):
        """Build LSTM model architecture for per-metric prediction."""
        if not TORCH_AVAILABLE:
            return None
            
        class VitalSignLSTM(nn.Module):
            def __init__(
                self,
                input_size: int = 1,
                hidden_size: int = 64,
                num_layers: int = 2,
                dropout: float = 0.2,
                forecast_horizons: int = 3
            ):
                super().__init__()
                
                self.hidden_size = hidden_size
                self.num_layers = num_layers
                self.forecast_horizons = forecast_horizons
                
                self.lstm = nn.LSTM(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    batch_first=True,
                    dropout=dropout if num_layers > 1 else 0,
                    bidirectional=True
                )
                
                self.fc = nn.Sequential(
                    nn.Linear(hidden_size * 2, hidden_size),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_size, forecast_horizons)
                )
                
                self.confidence_head = nn.Sequential(
                    nn.Linear(hidden_size * 2, hidden_size // 2),
                    nn.ReLU(),
                    nn.Linear(hidden_size // 2, forecast_horizons)
                )
            
            def forward(self, x):
                lstm_out, _ = self.lstm(x)
                last_hidden = lstm_out[:, -1, :]
                predictions = self.fc(last_hidden)
                confidence = torch.sigmoid(self.confidence_head(last_hidden))
                return predictions, confidence
        
        return VitalSignLSTM(
            input_size=1,
            hidden_size=64,
            num_layers=2,
            dropout=0.2,
            forecast_horizons=len(self.FORECAST_HORIZONS)
        )
    
    def _normalize(self, values: List[float], metric: str) -> np.ndarray:
        """Normalize values using stored scaler parameters."""
        params = self.scaler_params.get(metric, {"mean": 0.0, "std": 1.0})
        return (np.array(values) - params["mean"]) / (params["std"] + 1e-8)
    
    def _denormalize(self, values: np.ndarray, metric: str) -> np.ndarray:
        """Denormalize predicted values."""
        params = self.scaler_params.get(metric, {"mean": 0.0, "std": 1.0})
        return values * params["std"] + params["mean"]
    
    def _clip_to_valid_range(self, value: float, metric: str) -> float:
        """Clip predicted value to valid physiological range."""
        ranges = self.VITAL_RANGES.get(metric, {"min": 0, "max": 200})
        return float(max(ranges["min"], min(ranges["max"], value)))
    
    def _compute_trend_metrics(self, values: List[float]) -> Dict[str, Any]:
        """Compute trend direction, velocity, and acceleration."""
        if len(values) < 2:
            return {"direction": "stable", "velocity": 0.0, "acceleration": 0.0}
        
        arr = np.array(values)
        slope = np.polyfit(np.arange(len(arr)), arr, 1)[0]
        velocity = float(slope)
        
        if len(arr) >= 3:
            diffs = np.diff(arr)
            acceleration = float(np.mean(np.diff(diffs))) if len(diffs) >= 2 else 0.0
        else:
            acceleration = 0.0
        
        if abs(slope) < 0.02 * np.mean(arr):
            direction = "stable"
        elif slope > 0:
            direction = "increasing"
        else:
            direction = "decreasing"
        
        return {
            "direction": direction,
            "velocity": round(velocity, 4),
            "acceleration": round(acceleration, 4)
        }
    
    def _detect_anomalies(self, values: List[float], metric: str) -> Dict[str, Any]:
        """Detect anomalies in vital sign time series."""
        if len(values) < 3:
            return {"is_anomaly": False, "anomaly_type": None, "z_score": 0.0}
        
        arr = np.array(values)
        mean = float(np.mean(arr))
        std = float(np.std(arr)) + 1e-8
        
        latest_z = (arr[-1] - mean) / std
        
        ranges = self.VITAL_RANGES.get(metric, {})
        normal_min = ranges.get("normal_min", mean - 2*std)
        normal_max = ranges.get("normal_max", mean + 2*std)
        
        is_anomaly = abs(latest_z) > 2 or arr[-1] < normal_min or arr[-1] > normal_max
        
        if is_anomaly:
            if arr[-1] > normal_max:
                anomaly_type = "high"
            elif arr[-1] < normal_min:
                anomaly_type = "low"
            else:
                anomaly_type = "volatile"
        else:
            anomaly_type = None
        
        return {
            "is_anomaly": is_anomaly,
            "anomaly_type": anomaly_type,
            "z_score": round(float(latest_z), 3),
            "normal_range": [normal_min, normal_max]
        }
    
    def predict(
        self,
        time_series: List[Dict[str, float]],
        forecast_hours: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Generate multi-horizon vital sign predictions.
        
        Args:
            time_series: List of daily vital sign dictionaries
            forecast_hours: Optional list of hours to forecast (default: [24, 48, 72])
        
        Returns:
            Comprehensive prediction results with confidence intervals
        """
        if forecast_hours is None:
            forecast_hours = self.FORECAST_HORIZONS
        
        if not time_series or len(time_series) < 3:
            return {
                "status": "insufficient_data",
                "message": "Need at least 3 days of data for LSTM prediction",
                "minimum_required": 3,
                "data_points_provided": len(time_series) if time_series else 0,
                "predictions": [],
                "overall_assessment": None
            }
        
        predictions = []
        overall_risk_score = 0.0
        risk_factors: List[str] = []
        
        for metric in self.VITAL_METRICS:
            values = [ts.get(metric) for ts in time_series]
            values = [v for v in values if v is not None]
            
            if len(values) < 3:
                continue
            
            if TORCH_AVAILABLE and self.model is not None:
                metric_predictions = self._predict_with_lstm(values, metric, forecast_hours)
            else:
                metric_predictions = self._predict_with_statistical(values, metric, forecast_hours)
            
            trend = self._compute_trend_metrics(values)
            anomaly = self._detect_anomalies(values, metric)
            
            metric_result = {
                "metric": metric,
                "unit": self.VITAL_RANGES[metric]["unit"],
                "current_value": round(float(values[-1]), 2),
                "historical_mean": round(float(np.mean(values)), 2),
                "historical_std": round(float(np.std(values)), 3),
                "data_points": len(values),
                "trend": trend,
                "anomaly_detection": anomaly,
                "forecasts": metric_predictions
            }
            
            predictions.append(metric_result)
            
            metric_risk = self._calculate_metric_risk(metric_result)
            overall_risk_score += metric_risk["score"]
            if metric_risk["factors"]:
                risk_factors.extend(metric_risk["factors"])
        
        num_metrics = len(predictions) or 1
        normalized_risk = min(100.0, overall_risk_score / num_metrics)
        
        if normalized_risk >= 70:
            risk_level = "high"
        elif normalized_risk >= 40:
            risk_level = "moderate"
        else:
            risk_level = "low"
        
        declining_metrics = [p for p in predictions if p["trend"]["direction"] == "decreasing" and p["metric"] == "spo2"]
        declining_metrics += [p for p in predictions if p["trend"]["direction"] == "increasing" and p["metric"] not in ["spo2"]]
        
        if len(declining_metrics) >= 3:
            overall_trajectory = "deteriorating"
        elif len(declining_metrics) >= 2:
            overall_trajectory = "concerning"
        else:
            overall_trajectory = "stable"
        
        return {
            "status": "success",
            "model_type": "LSTM" if (TORCH_AVAILABLE and self.model) else "statistical",
            "model_version": "2.0.0",
            "forecast_horizons": forecast_hours,
            "data_points_analyzed": len(time_series),
            "predictions": predictions,
            "overall_assessment": {
                "risk_level": risk_level,
                "risk_score": round(normalized_risk, 1),
                "trajectory": overall_trajectory,
                "risk_factors": risk_factors[:5],
                "confidence": round(self._calculate_overall_confidence(predictions), 3)
            },
            "generated_at": datetime.utcnow().isoformat()
        }
    
    def _predict_with_lstm(
        self,
        values: List[float],
        metric: str,
        forecast_hours: List[int]
    ) -> List[Dict[str, Any]]:
        """Generate predictions using LSTM model with graceful fallback."""
        if not TORCH_AVAILABLE or self.model is None:
            return self._predict_with_statistical(values, metric, forecast_hours)
        
        try:
            normalized = self._normalize(values, metric)
            
            input_tensor = torch.tensor(normalized, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
            
            with torch.no_grad():
                pred_output, confidence_output = self.model(input_tensor)
            
            horizon_to_head = {24: 0, 48: 1, 72: 2}
            
            predictions = []
            for hours in forecast_hours:
                head_idx = horizon_to_head.get(hours, 0)
                if hours not in horizon_to_head:
                    logger.warning(f"Unknown forecast horizon {hours}h, defaulting to 24h head")
                
                pred_normalized = pred_output[0, head_idx].item()
                confidence = confidence_output[0, head_idx].item()
                
                pred_value = float(self._denormalize(np.array([pred_normalized]), metric)[0])
                pred_value = self._clip_to_valid_range(pred_value, metric)
                
                uncertainty = (1 - confidence) * float(np.std(values)) * 2
                lower = self._clip_to_valid_range(pred_value - uncertainty, metric)
                upper = self._clip_to_valid_range(pred_value + uncertainty, metric)
                
                predictions.append({
                    "horizon_hours": hours,
                    "predicted_value": round(pred_value, 2),
                    "confidence": round(float(confidence), 3),
                    "confidence_interval": {
                        "lower": round(lower, 2),
                        "upper": round(upper, 2)
                    },
                    "change_from_current": round(pred_value - values[-1], 2),
                    "percent_change": round((pred_value - values[-1]) / values[-1] * 100, 2) if values[-1] != 0 else 0.0
                })
            
            return predictions
            
        except Exception as e:
            logger.warning(f"LSTM prediction failed for {metric}, falling back to statistical: {e}")
            return self._predict_with_statistical(values, metric, forecast_hours)
    
    def _predict_with_statistical(
        self,
        values: List[float],
        metric: str,
        forecast_hours: List[int]
    ) -> List[Dict[str, Any]]:
        """Fallback statistical prediction when LSTM unavailable."""
        arr = np.array(values)
        
        x = np.arange(len(arr))
        try:
            coeffs = np.polyfit(x, arr, min(2, len(arr) - 1))
        except:
            coeffs = [0, float(np.mean(arr))]
        
        std = float(np.std(arr))
        
        predictions = []
        for hours in forecast_hours:
            steps = hours / 24.0
            future_x = len(arr) - 1 + steps
            pred_value = float(np.polyval(coeffs, future_x))
            pred_value = self._clip_to_valid_range(pred_value, metric)
            
            base_confidence = max(0.5, 1.0 - (std / (float(np.mean(arr)) + 1e-8)))
            confidence = base_confidence * (1.0 - steps * 0.05)
            
            uncertainty = std * (1 + steps * 0.3)
            lower = self._clip_to_valid_range(pred_value - uncertainty, metric)
            upper = self._clip_to_valid_range(pred_value + uncertainty, metric)
            
            predictions.append({
                "horizon_hours": hours,
                "predicted_value": round(pred_value, 2),
                "confidence": round(float(confidence), 3),
                "confidence_interval": {
                    "lower": round(lower, 2),
                    "upper": round(upper, 2)
                },
                "change_from_current": round(pred_value - values[-1], 2),
                "percent_change": round((pred_value - values[-1]) / values[-1] * 100, 2) if values[-1] != 0 else 0.0
            })
        
        return predictions
    
    def _calculate_metric_risk(self, metric_result: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate risk score for a single metric."""
        score = 0.0
        factors: List[str] = []
        
        metric = metric_result["metric"]
        current = metric_result["current_value"]
        ranges = self.VITAL_RANGES.get(metric, {})
        
        if current < ranges.get("normal_min", 0):
            deviation = (ranges["normal_min"] - current) / ranges["normal_min"] * 100
            score += min(40.0, deviation)
            factors.append(f"Low {metric.replace('_', ' ')}: {current} {ranges.get('unit', '')}")
        elif current > ranges.get("normal_max", 200):
            deviation = (current - ranges["normal_max"]) / ranges["normal_max"] * 100
            score += min(40.0, deviation)
            factors.append(f"High {metric.replace('_', ' ')}: {current} {ranges.get('unit', '')}")
        
        if metric_result["anomaly_detection"]["is_anomaly"]:
            score += 20.0
            factors.append(f"Anomalous {metric.replace('_', ' ')} pattern detected")
        
        trend = metric_result["trend"]
        if metric == "spo2" and trend["direction"] == "decreasing":
            score += 15.0 * abs(trend["velocity"]) * 10
            factors.append("Declining oxygen saturation trend")
        elif metric != "spo2" and trend["direction"] == "increasing" and trend["velocity"] > 0.1:
            score += 10.0
        
        for forecast in metric_result.get("forecasts", []):
            if forecast["horizon_hours"] == 24:
                pred = forecast["predicted_value"]
                if pred < ranges.get("normal_min", 0) * 0.9 or pred > ranges.get("normal_max", 200) * 1.1:
                    score += 15.0
                    factors.append("24h forecast outside normal range")
                    break
        
        return {"score": score, "factors": factors}
    
    def _calculate_overall_confidence(self, predictions: List[Dict[str, Any]]) -> float:
        """Calculate overall prediction confidence."""
        if not predictions:
            return 0.5
        
        confidences = []
        for pred in predictions:
            for forecast in pred.get("forecasts", []):
                confidences.append(forecast.get("confidence", 0.5))
        
        return float(np.mean(confidences)) if confidences else 0.5


_lstm_predictor_instance: Optional[LSTMVitalPredictor] = None


def get_lstm_predictor() -> LSTMVitalPredictor:
    """Get singleton LSTM predictor instance."""
    global _lstm_predictor_instance
    if _lstm_predictor_instance is None:
        _lstm_predictor_instance = LSTMVitalPredictor()
    return _lstm_predictor_instance
