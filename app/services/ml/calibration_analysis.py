"""
Calibration Analysis Service (Phase C.20)
=========================================
Reliability curve, ECE, and Brier score calculation for model calibration.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import math

logger = logging.getLogger(__name__)


@dataclass
class CalibrationBin:
    """Single bin in calibration analysis"""
    bin_id: int
    lower_bound: float
    upper_bound: float
    mean_predicted: float
    mean_observed: float
    count: int
    calibration_error: float


@dataclass
class CalibrationReport:
    """Complete calibration analysis report"""
    report_id: str
    timestamp: str
    bins: List[CalibrationBin]
    ece: float
    mce: float
    brier_score: float
    reliability_diagram: Dict[str, List[float]]
    is_well_calibrated: bool
    recommendations: List[str]


class CalibrationAnalysisService:
    """
    C.20: Calibration analysis with reliability curves, ECE, and Brier score.
    
    Evaluates how well predicted probabilities match observed frequencies.
    """
    
    def __init__(
        self,
        n_bins: int = 10,
        ece_threshold: float = 0.1,
    ):
        self.n_bins = n_bins
        self.ece_threshold = ece_threshold
    
    def analyze_calibration(
        self,
        predicted_probs: List[float],
        actual_outcomes: List[int],
    ) -> CalibrationReport:
        """
        Analyze calibration of predicted probabilities.
        
        Args:
            predicted_probs: List of predicted probabilities (0-1)
            actual_outcomes: List of actual binary outcomes (0 or 1)
            
        Returns:
            CalibrationReport with all metrics and visualization data
        """
        from uuid import uuid4
        
        if len(predicted_probs) != len(actual_outcomes):
            raise ValueError("predicted_probs and actual_outcomes must have same length")
        
        if not predicted_probs:
            return self._empty_report()
        
        bins = self._compute_bins(predicted_probs, actual_outcomes)
        
        ece = self._compute_ece(bins, len(predicted_probs))
        
        mce = self._compute_mce(bins)
        
        brier = self._compute_brier_score(predicted_probs, actual_outcomes)
        
        reliability_diagram = {
            "predicted": [b.mean_predicted for b in bins if b.count > 0],
            "observed": [b.mean_observed for b in bins if b.count > 0],
            "counts": [b.count for b in bins if b.count > 0],
            "perfect_calibration": list(self._linspace(0, 1, 11)),
        }
        
        is_well_calibrated = ece <= self.ece_threshold
        
        recommendations = self._generate_recommendations(
            ece, mce, brier, bins, is_well_calibrated
        )
        
        return CalibrationReport(
            report_id=str(uuid4()),
            timestamp=datetime.utcnow().isoformat(),
            bins=bins,
            ece=ece,
            mce=mce,
            brier_score=brier,
            reliability_diagram=reliability_diagram,
            is_well_calibrated=is_well_calibrated,
            recommendations=recommendations,
        )
    
    def _linspace(self, start: float, stop: float, num: int) -> List[float]:
        """Generate linearly spaced values"""
        if num < 2:
            return [start]
        step = (stop - start) / (num - 1)
        return [start + step * i for i in range(num)]
    
    def _compute_bins(
        self,
        predicted_probs: List[float],
        actual_outcomes: List[int],
    ) -> List[CalibrationBin]:
        """Compute calibration bins"""
        bin_edges = self._linspace(0, 1, self.n_bins + 1)
        bins = []
        
        for i in range(self.n_bins):
            lower = bin_edges[i]
            upper = bin_edges[i + 1]
            
            indices = [
                j for j, p in enumerate(predicted_probs)
                if lower <= p < upper or (i == self.n_bins - 1 and p == upper)
            ]
            
            if not indices:
                bins.append(CalibrationBin(
                    bin_id=i,
                    lower_bound=lower,
                    upper_bound=upper,
                    mean_predicted=0.0,
                    mean_observed=0.0,
                    count=0,
                    calibration_error=0.0,
                ))
                continue
            
            bin_probs = [predicted_probs[j] for j in indices]
            bin_outcomes = [actual_outcomes[j] for j in indices]
            
            mean_pred = sum(bin_probs) / len(bin_probs)
            mean_obs = sum(bin_outcomes) / len(bin_outcomes)
            cal_error = abs(mean_pred - mean_obs)
            
            bins.append(CalibrationBin(
                bin_id=i,
                lower_bound=lower,
                upper_bound=upper,
                mean_predicted=mean_pred,
                mean_observed=mean_obs,
                count=len(indices),
                calibration_error=cal_error,
            ))
        
        return bins
    
    def _compute_ece(self, bins: List[CalibrationBin], n_samples: int) -> float:
        """Compute Expected Calibration Error"""
        if n_samples == 0:
            return 0.0
        
        ece = sum(
            (bin.count / n_samples) * bin.calibration_error
            for bin in bins
            if bin.count > 0
        )
        
        return ece
    
    def _compute_mce(self, bins: List[CalibrationBin]) -> float:
        """Compute Maximum Calibration Error"""
        errors = [bin.calibration_error for bin in bins if bin.count > 0]
        return max(errors) if errors else 0.0
    
    def _compute_brier_score(
        self,
        predicted_probs: List[float],
        actual_outcomes: List[int],
    ) -> float:
        """Compute Brier Score (lower is better)"""
        if not predicted_probs:
            return 0.0
        
        squared_errors = [
            (p - y) ** 2
            for p, y in zip(predicted_probs, actual_outcomes)
        ]
        
        return sum(squared_errors) / len(squared_errors)
    
    def _generate_recommendations(
        self,
        ece: float,
        mce: float,
        brier: float,
        bins: List[CalibrationBin],
        is_well_calibrated: bool,
    ) -> List[str]:
        """Generate calibration improvement recommendations"""
        recommendations = []
        
        if not is_well_calibrated:
            recommendations.append(
                f"Model is not well-calibrated (ECE={ece:.3f} > {self.ece_threshold}). "
                "Consider applying calibration methods like Platt scaling or isotonic regression."
            )
        
        if mce > 0.2:
            worst_bin = max(bins, key=lambda b: b.calibration_error if b.count > 0 else 0)
            recommendations.append(
                f"High maximum calibration error ({mce:.3f}) in probability range "
                f"[{worst_bin.lower_bound:.1f}, {worst_bin.upper_bound:.1f}]. "
                "Focus calibration efforts on this region."
            )
        
        if brier > 0.25:
            recommendations.append(
                f"Brier score ({brier:.3f}) indicates poor probability estimates. "
                "Consider retraining with better features or more data."
            )
        
        low_count_bins = [b for b in bins if 0 < b.count < 10]
        if low_count_bins:
            recommendations.append(
                f"{len(low_count_bins)} bins have fewer than 10 samples. "
                "Consider using fewer bins or gathering more data for reliable calibration."
            )
        
        if is_well_calibrated and brier < 0.15:
            recommendations.append(
                "Model is well-calibrated with good probability estimates. "
                "Probabilities can be used directly for clinical decision-making."
            )
        
        return recommendations
    
    def _empty_report(self) -> CalibrationReport:
        """Return empty report for edge cases"""
        from uuid import uuid4
        return CalibrationReport(
            report_id=str(uuid4()),
            timestamp=datetime.utcnow().isoformat(),
            bins=[],
            ece=0.0,
            mce=0.0,
            brier_score=0.0,
            reliability_diagram={"predicted": [], "observed": [], "counts": [], "perfect_calibration": []},
            is_well_calibrated=True,
            recommendations=["No data to analyze"],
        )


def analyze_calibration(
    predicted_probs: List[float],
    actual_outcomes: List[int],
    n_bins: int = 10,
) -> CalibrationReport:
    """Convenience function to analyze calibration"""
    service = CalibrationAnalysisService(n_bins=n_bins)
    return service.analyze_calibration(predicted_probs, actual_outcomes)


__all__ = [
    "CalibrationAnalysisService",
    "CalibrationBin",
    "CalibrationReport",
    "analyze_calibration",
]
