"""
Threshold Optimizer Service (Phase C.21)
========================================
Decision threshold optimization balancing alert burden vs sensitivity.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import math

logger = logging.getLogger(__name__)


@dataclass
class ThresholdCandidate:
    """Single threshold candidate with metrics"""
    threshold: float
    sensitivity: float
    specificity: float
    ppv: float
    npv: float
    f1_score: float
    alert_rate: float
    cost: float


@dataclass
class ThresholdOptimizationResult:
    """Complete threshold optimization result"""
    result_id: str
    timestamp: str
    optimal_threshold: float
    candidates: List[ThresholdCandidate]
    optimization_target: str
    constraints_met: bool
    performance_summary: Dict[str, float]
    alert_burden_analysis: Dict[str, Any]


class ThresholdOptimizerService:
    """
    C.21: Threshold optimization service.
    
    Optimizes decision thresholds balancing:
    - Clinical sensitivity (don't miss true positives)
    - Alert burden (minimize false positives)
    - Resource constraints (alert budget)
    """
    
    OPTIMIZATION_TARGETS = [
        "f1_score",
        "sensitivity",
        "specificity",
        "youden_j",
        "alert_budget",
    ]
    
    def __init__(
        self,
        n_thresholds: int = 100,
        min_sensitivity: float = 0.8,
        max_alert_rate: Optional[float] = None,
    ):
        self.n_thresholds = n_thresholds
        self.min_sensitivity = min_sensitivity
        self.max_alert_rate = max_alert_rate
    
    def optimize_threshold(
        self,
        predicted_probs: List[float],
        actual_outcomes: List[int],
        optimization_target: str = "f1_score",
        alert_cost: float = 1.0,
        miss_cost: float = 10.0,
    ) -> ThresholdOptimizationResult:
        """
        Find optimal decision threshold.
        
        Args:
            predicted_probs: List of predicted probabilities
            actual_outcomes: List of actual binary outcomes
            optimization_target: Metric to optimize
            alert_cost: Cost per false positive alert
            miss_cost: Cost per missed true positive
            
        Returns:
            ThresholdOptimizationResult with optimal threshold and analysis
        """
        from uuid import uuid4
        
        if len(predicted_probs) != len(actual_outcomes):
            raise ValueError("predicted_probs and actual_outcomes must have same length")
        
        if not predicted_probs:
            return self._empty_result()
        
        thresholds = self._generate_thresholds(predicted_probs)
        
        candidates = []
        for thresh in thresholds:
            metrics = self._compute_metrics(
                predicted_probs, actual_outcomes, thresh, alert_cost, miss_cost
            )
            candidates.append(metrics)
        
        valid_candidates = self._filter_by_constraints(candidates)
        
        optimal = self._select_optimal(
            valid_candidates or candidates,
            optimization_target
        )
        
        alert_burden = self._analyze_alert_burden(
            candidates, optimal.threshold, len(actual_outcomes)
        )
        
        return ThresholdOptimizationResult(
            result_id=str(uuid4()),
            timestamp=datetime.utcnow().isoformat(),
            optimal_threshold=optimal.threshold,
            candidates=candidates,
            optimization_target=optimization_target,
            constraints_met=len(valid_candidates) > 0,
            performance_summary={
                "sensitivity": optimal.sensitivity,
                "specificity": optimal.specificity,
                "ppv": optimal.ppv,
                "npv": optimal.npv,
                "f1_score": optimal.f1_score,
                "alert_rate": optimal.alert_rate,
            },
            alert_burden_analysis=alert_burden,
        )
    
    def _generate_thresholds(self, predicted_probs: List[float]) -> List[float]:
        """Generate candidate thresholds"""
        min_prob = min(predicted_probs)
        max_prob = max(predicted_probs)
        
        step = (max_prob - min_prob) / (self.n_thresholds - 1)
        thresholds = [min_prob + i * step for i in range(self.n_thresholds)]
        
        thresholds = sorted(set(thresholds) | {0.5})
        
        return thresholds
    
    def _compute_metrics(
        self,
        predicted_probs: List[float],
        actual_outcomes: List[int],
        threshold: float,
        alert_cost: float,
        miss_cost: float,
    ) -> ThresholdCandidate:
        """Compute metrics for a single threshold"""
        predictions = [1 if p >= threshold else 0 for p in predicted_probs]
        
        tp = sum(1 for p, a in zip(predictions, actual_outcomes) if p == 1 and a == 1)
        fp = sum(1 for p, a in zip(predictions, actual_outcomes) if p == 1 and a == 0)
        tn = sum(1 for p, a in zip(predictions, actual_outcomes) if p == 0 and a == 0)
        fn = sum(1 for p, a in zip(predictions, actual_outcomes) if p == 0 and a == 1)
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        
        precision = ppv
        recall = sensitivity
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        alert_rate = (tp + fp) / len(predictions) if predictions else 0.0
        
        cost = (fp * alert_cost) + (fn * miss_cost)
        
        return ThresholdCandidate(
            threshold=threshold,
            sensitivity=sensitivity,
            specificity=specificity,
            ppv=ppv,
            npv=npv,
            f1_score=f1,
            alert_rate=alert_rate,
            cost=cost,
        )
    
    def _filter_by_constraints(
        self,
        candidates: List[ThresholdCandidate],
    ) -> List[ThresholdCandidate]:
        """Filter candidates by constraints"""
        valid = []
        
        for c in candidates:
            if c.sensitivity < self.min_sensitivity:
                continue
            
            if self.max_alert_rate is not None and c.alert_rate > self.max_alert_rate:
                continue
            
            valid.append(c)
        
        return valid
    
    def _select_optimal(
        self,
        candidates: List[ThresholdCandidate],
        target: str,
    ) -> ThresholdCandidate:
        """Select optimal threshold based on target metric"""
        if not candidates:
            return ThresholdCandidate(
                threshold=0.5,
                sensitivity=0.0,
                specificity=0.0,
                ppv=0.0,
                npv=0.0,
                f1_score=0.0,
                alert_rate=0.0,
                cost=0.0,
            )
        
        if target == "f1_score":
            return max(candidates, key=lambda c: c.f1_score)
        elif target == "sensitivity":
            viable = [c for c in candidates if c.specificity >= 0.5]
            if viable:
                return max(viable, key=lambda c: c.sensitivity)
            return max(candidates, key=lambda c: c.sensitivity)
        elif target == "specificity":
            viable = [c for c in candidates if c.sensitivity >= 0.5]
            if viable:
                return max(viable, key=lambda c: c.specificity)
            return max(candidates, key=lambda c: c.specificity)
        elif target == "youden_j":
            return max(candidates, key=lambda c: c.sensitivity + c.specificity - 1)
        elif target == "alert_budget":
            return min(candidates, key=lambda c: c.cost)
        else:
            return max(candidates, key=lambda c: c.f1_score)
    
    def _analyze_alert_burden(
        self,
        candidates: List[ThresholdCandidate],
        optimal_threshold: float,
        n_samples: int,
    ) -> Dict[str, Any]:
        """Analyze alert burden at different thresholds"""
        current = next(
            (c for c in candidates if abs(c.threshold - optimal_threshold) < 0.01),
            candidates[0] if candidates else None
        )
        
        conservative = next(
            (c for c in candidates if abs(c.threshold - 0.7) < 0.05),
            None
        )
        
        aggressive = next(
            (c for c in candidates if abs(c.threshold - 0.3) < 0.05),
            None
        )
        
        return {
            "current_threshold": {
                "threshold": optimal_threshold,
                "daily_alerts_per_1000": current.alert_rate * 1000 if current else 0,
                "sensitivity": current.sensitivity if current else 0,
            },
            "conservative_alternative": {
                "threshold": 0.7,
                "daily_alerts_per_1000": conservative.alert_rate * 1000 if conservative else 0,
                "sensitivity": conservative.sensitivity if conservative else 0,
            } if conservative else None,
            "aggressive_alternative": {
                "threshold": 0.3,
                "daily_alerts_per_1000": aggressive.alert_rate * 1000 if aggressive else 0,
                "sensitivity": aggressive.sensitivity if aggressive else 0,
            } if aggressive else None,
            "recommendation": self._generate_recommendation(current, n_samples),
        }
    
    def _generate_recommendation(
        self,
        current: Optional[ThresholdCandidate],
        n_samples: int,
    ) -> str:
        """Generate threshold recommendation"""
        if current is None:
            return "Insufficient data for threshold optimization"
        
        if current.sensitivity >= 0.9 and current.alert_rate <= 0.1:
            return "Excellent balance: High sensitivity with manageable alert burden"
        elif current.sensitivity >= 0.8 and current.alert_rate <= 0.2:
            return "Good balance: Adequate sensitivity with acceptable alert burden"
        elif current.sensitivity >= 0.9 and current.alert_rate > 0.3:
            return "High sensitivity but significant alert burden. Consider raising threshold if resources constrained"
        elif current.sensitivity < 0.7:
            return "Low sensitivity warning: Consider lowering threshold to catch more true positives"
        else:
            return "Moderate performance. Review clinical context to adjust threshold appropriately"
    
    def _empty_result(self) -> ThresholdOptimizationResult:
        """Return empty result for edge cases"""
        from uuid import uuid4
        return ThresholdOptimizationResult(
            result_id=str(uuid4()),
            timestamp=datetime.utcnow().isoformat(),
            optimal_threshold=0.5,
            candidates=[],
            optimization_target="f1_score",
            constraints_met=False,
            performance_summary={},
            alert_burden_analysis={},
        )


def optimize_threshold(
    predicted_probs: List[float],
    actual_outcomes: List[int],
    optimization_target: str = "f1_score",
) -> ThresholdOptimizationResult:
    """Convenience function to optimize threshold"""
    service = ThresholdOptimizerService()
    return service.optimize_threshold(
        predicted_probs, actual_outcomes, optimization_target
    )


__all__ = [
    "ThresholdOptimizerService",
    "ThresholdCandidate",
    "ThresholdOptimizationResult",
    "optimize_threshold",
]
