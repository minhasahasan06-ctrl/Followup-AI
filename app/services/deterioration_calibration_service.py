"""
Deterioration Calibration Service
==================================

Phase 13: Production-grade probability calibration for ML models.

Implements:
- Platt scaling (sigmoid calibration)
- Temperature scaling for neural networks
- Isotonic regression for non-parametric calibration
- Expected Calibration Error (ECE) and Brier score metrics

All calibration parameters are stored in PostgreSQL for reproducibility.
"""

import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from sqlalchemy.orm import Session

try:
    from scipy.optimize import minimize
    from scipy.special import expit
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from sklearn.isotonic import IsotonicRegression
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from app.models.ml_models import MLCalibrationParams
from app.services.audit_logger import HIPAAAuditLogger

logger = logging.getLogger(__name__)


class CalibrationService:
    """
    Service for calibrating ML model probabilities.
    
    Provides multiple calibration methods to convert raw model outputs
    to well-calibrated probability estimates.
    """
    
    def __init__(self, db: Session):
        self.db = db
        self.logger = logging.getLogger(__name__)
    
    def compute_ece(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        n_bins: int = 10
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Compute Expected Calibration Error (ECE).
        
        ECE measures the difference between predicted probabilities and
        actual outcomes, weighted by the number of samples in each bin.
        
        Args:
            y_true: True binary labels (0 or 1)
            y_prob: Predicted probabilities
            n_bins: Number of probability bins
            
        Returns:
            Tuple of (ECE value, reliability diagram data)
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        reliability_data = {
            "bin_midpoints": [],
            "bin_accuracies": [],
            "bin_confidences": [],
            "bin_counts": []
        }
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            prop_in_bin = np.mean(in_bin)
            
            if prop_in_bin > 0:
                accuracy_in_bin = np.mean(y_true[in_bin])
                avg_confidence_in_bin = np.mean(y_prob[in_bin])
                ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin
                
                reliability_data["bin_midpoints"].append((bin_lower + bin_upper) / 2)
                reliability_data["bin_accuracies"].append(float(accuracy_in_bin))
                reliability_data["bin_confidences"].append(float(avg_confidence_in_bin))
                reliability_data["bin_counts"].append(int(np.sum(in_bin)))
        
        return float(ece), reliability_data
    
    def compute_brier_score(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray
    ) -> float:
        """
        Compute Brier score (mean squared error of probabilities).
        
        Lower is better. Perfect calibration = 0.
        """
        return float(np.mean((y_prob - y_true) ** 2))
    
    def fit_platt_scaling(
        self,
        y_true: np.ndarray,
        logits: np.ndarray
    ) -> Tuple[float, float]:
        """
        Fit Platt scaling parameters (A, B) for sigmoid calibration.
        
        Calibrated probability = 1 / (1 + exp(A * logit + B))
        
        Args:
            y_true: True binary labels
            logits: Raw model outputs (before sigmoid)
            
        Returns:
            Tuple of (A, B) parameters
        """
        if not SCIPY_AVAILABLE:
            self.logger.warning("scipy not available, using default Platt parameters")
            return -1.0, 0.0
        
        prior1 = np.sum(y_true)
        prior0 = len(y_true) - prior1
        
        hi_target = (prior1 + 1) / (prior1 + 2)
        lo_target = 1 / (prior0 + 2)
        targets = np.where(y_true == 1, hi_target, lo_target)
        
        def objective(params):
            A, B = params
            p = expit(A * logits + B)
            p = np.clip(p, 1e-10, 1 - 1e-10)
            loss = -np.sum(targets * np.log(p) + (1 - targets) * np.log(1 - p))
            return loss
        
        result = minimize(
            objective,
            x0=[-1.0, 0.0],
            method='L-BFGS-B',
            bounds=[(-10, 10), (-10, 10)]
        )
        
        return float(result.x[0]), float(result.x[1])
    
    def apply_platt_scaling(
        self,
        logits: np.ndarray,
        A: float,
        B: float
    ) -> np.ndarray:
        """Apply Platt scaling to convert logits to calibrated probabilities."""
        if SCIPY_AVAILABLE:
            return expit(A * logits + B)
        return 1 / (1 + np.exp(-(A * logits + B)))
    
    def fit_temperature_scaling(
        self,
        y_true: np.ndarray,
        logits: np.ndarray
    ) -> float:
        """
        Fit temperature scaling parameter for neural network calibration.
        
        Temperature scaling divides logits by T before softmax/sigmoid.
        T > 1 softens predictions, T < 1 sharpens them.
        
        Returns:
            Optimal temperature value
        """
        if not SCIPY_AVAILABLE:
            return 1.0
        
        def nll_loss(T):
            T = max(T[0], 0.01)
            scaled_logits = logits / T
            probs = expit(scaled_logits)
            probs = np.clip(probs, 1e-10, 1 - 1e-10)
            nll = -np.mean(y_true * np.log(probs) + (1 - y_true) * np.log(1 - probs))
            return nll
        
        result = minimize(
            nll_loss,
            x0=[1.0],
            method='L-BFGS-B',
            bounds=[(0.01, 10.0)]
        )
        
        return float(result.x[0])
    
    def apply_temperature_scaling(
        self,
        logits: np.ndarray,
        temperature: float
    ) -> np.ndarray:
        """Apply temperature scaling to logits."""
        scaled_logits = logits / max(temperature, 0.01)
        if SCIPY_AVAILABLE:
            return expit(scaled_logits)
        return 1 / (1 + np.exp(-scaled_logits))
    
    def fit_isotonic_regression(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray
    ) -> Tuple[List[float], List[float]]:
        """
        Fit isotonic regression for non-parametric calibration.
        
        Returns:
            Tuple of (x_values, y_values) for the isotonic mapping
        """
        if not SKLEARN_AVAILABLE:
            return [0.0, 1.0], [0.0, 1.0]
        
        ir = IsotonicRegression(out_of_bounds='clip')
        ir.fit(y_prob, y_true)
        
        x_unique = np.unique(y_prob)
        y_calibrated = ir.predict(x_unique)
        
        return x_unique.tolist(), y_calibrated.tolist()
    
    def apply_isotonic_regression(
        self,
        y_prob: np.ndarray,
        x_values: List[float],
        y_values: List[float]
    ) -> np.ndarray:
        """Apply isotonic regression mapping to probabilities."""
        return np.interp(y_prob, x_values, y_values)
    
    def calibrate_and_save(
        self,
        model_id: str,
        y_true: np.ndarray,
        logits_or_probs: np.ndarray,
        method: str = "platt",
        is_logits: bool = True,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Calibrate a model and save parameters to PostgreSQL.
        
        Args:
            model_id: ID of the model to calibrate
            y_true: True labels
            logits_or_probs: Raw logits or probabilities
            method: Calibration method (platt, temperature, isotonic)
            is_logits: Whether input is logits (True) or probabilities (False)
            user_id: User performing calibration
            
        Returns:
            Dict with calibration results and metrics
        """
        y_true = np.array(y_true)
        logits_or_probs = np.array(logits_or_probs)
        
        if is_logits:
            uncalibrated_probs = 1 / (1 + np.exp(-logits_or_probs))
        else:
            uncalibrated_probs = logits_or_probs
        
        ece_before, _ = self.compute_ece(y_true, uncalibrated_probs)
        brier_before = self.compute_brier_score(y_true, uncalibrated_probs)
        
        calibration_record = MLCalibrationParams(
            model_id=model_id,
            calibration_method=method,
            ece_before=ece_before,
            brier_before=brier_before,
            validation_samples=len(y_true),
            created_by=user_id
        )
        
        if method == "platt":
            if is_logits:
                A, B = self.fit_platt_scaling(y_true, logits_or_probs)
                calibrated_probs = self.apply_platt_scaling(logits_or_probs, A, B)
            else:
                logits_approx = np.log(np.clip(logits_or_probs, 1e-10, 1-1e-10) / 
                                       np.clip(1-logits_or_probs, 1e-10, 1-1e-10))
                A, B = self.fit_platt_scaling(y_true, logits_approx)
                calibrated_probs = self.apply_platt_scaling(logits_approx, A, B)
            
            calibration_record.platt_a = A
            calibration_record.platt_b = B
            
        elif method == "temperature":
            if is_logits:
                T = self.fit_temperature_scaling(y_true, logits_or_probs)
                calibrated_probs = self.apply_temperature_scaling(logits_or_probs, T)
            else:
                logits_approx = np.log(np.clip(logits_or_probs, 1e-10, 1-1e-10) / 
                                       np.clip(1-logits_or_probs, 1e-10, 1-1e-10))
                T = self.fit_temperature_scaling(y_true, logits_approx)
                calibrated_probs = self.apply_temperature_scaling(logits_approx, T)
            
            calibration_record.temperature = T
            
        elif method == "isotonic":
            x_vals, y_vals = self.fit_isotonic_regression(y_true, uncalibrated_probs)
            calibrated_probs = self.apply_isotonic_regression(uncalibrated_probs, x_vals, y_vals)
            
            calibration_record.isotonic_x = x_vals
            calibration_record.isotonic_y = y_vals
        else:
            raise ValueError(f"Unknown calibration method: {method}")
        
        ece_after, reliability = self.compute_ece(y_true, calibrated_probs)
        brier_after = self.compute_brier_score(y_true, calibrated_probs)
        
        calibration_record.ece_after = ece_after
        calibration_record.brier_after = brier_after
        calibration_record.reliability_diagram = reliability
        calibration_record.is_active = True
        
        self.db.query(MLCalibrationParams).filter(
            MLCalibrationParams.model_id == model_id,
            MLCalibrationParams.is_active == True
        ).update({"is_active": False})
        
        self.db.add(calibration_record)
        self.db.commit()
        self.db.refresh(calibration_record)
        
        HIPAAAuditLogger.log_phi_access(
            actor_id=user_id or "system",
            actor_role="system",
            patient_id="N/A",
            resource_type="ml_calibration",
            action="create",
            access_reason=f"Calibrate model using {method}",
            additional_context={
                "model_id": model_id,
                "method": method,
                "ece_before": ece_before,
                "ece_after": ece_after,
                "improvement": (ece_before - ece_after) / max(ece_before, 1e-10) * 100
            }
        )
        
        self.logger.info(
            f"Calibrated model {model_id} using {method}: "
            f"ECE {ece_before:.4f} -> {ece_after:.4f}"
        )
        
        return {
            "model_id": model_id,
            "method": method,
            "calibration_id": calibration_record.id,
            "metrics": {
                "ece_before": ece_before,
                "ece_after": ece_after,
                "brier_before": brier_before,
                "brier_after": brier_after,
                "improvement_percent": (ece_before - ece_after) / max(ece_before, 1e-10) * 100
            },
            "reliability_diagram": reliability
        }
    
    def get_active_calibration(self, model_id: str) -> Optional[MLCalibrationParams]:
        """Get the active calibration parameters for a model."""
        return self.db.query(MLCalibrationParams).filter(
            MLCalibrationParams.model_id == model_id,
            MLCalibrationParams.is_active == True
        ).first()
    
    def apply_calibration(
        self,
        model_id: str,
        raw_output: np.ndarray,
        is_logits: bool = True
    ) -> np.ndarray:
        """
        Apply stored calibration to model outputs.
        
        Args:
            model_id: Model ID to get calibration for
            raw_output: Raw model outputs
            is_logits: Whether inputs are logits or probabilities
            
        Returns:
            Calibrated probabilities
        """
        calibration = self.get_active_calibration(model_id)
        
        if not calibration:
            if is_logits:
                return 1 / (1 + np.exp(-np.array(raw_output)))
            return np.array(raw_output)
        
        raw_output = np.array(raw_output)
        
        if calibration.calibration_method == "platt":
            if not is_logits:
                logits = np.log(np.clip(raw_output, 1e-10, 1-1e-10) / 
                               np.clip(1-raw_output, 1e-10, 1-1e-10))
            else:
                logits = raw_output
            return self.apply_platt_scaling(logits, calibration.platt_a, calibration.platt_b)
            
        elif calibration.calibration_method == "temperature":
            if not is_logits:
                logits = np.log(np.clip(raw_output, 1e-10, 1-1e-10) / 
                               np.clip(1-raw_output, 1e-10, 1-1e-10))
            else:
                logits = raw_output
            return self.apply_temperature_scaling(logits, calibration.temperature)
            
        elif calibration.calibration_method == "isotonic":
            if is_logits:
                probs = 1 / (1 + np.exp(-raw_output))
            else:
                probs = raw_output
            return self.apply_isotonic_regression(
                probs, 
                calibration.isotonic_x, 
                calibration.isotonic_y
            )
        
        if is_logits:
            return 1 / (1 + np.exp(-raw_output))
        return raw_output


def get_calibration_service(db: Session) -> CalibrationService:
    """Factory function to get CalibrationService instance."""
    return CalibrationService(db)
