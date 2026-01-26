"""
Drift Detection Service (Phase C.22)
====================================
Feature distribution drift detection using PSI and KL divergence.
"""

import logging
import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter

logger = logging.getLogger(__name__)


@dataclass
class FeatureDrift:
    """Drift metrics for a single feature"""
    feature_name: str
    psi: float
    kl_divergence: float
    drift_detected: bool
    severity: str
    reference_distribution: Dict[str, float]
    comparison_distribution: Dict[str, float]


@dataclass
class DriftReport:
    """Complete drift detection report"""
    report_id: str
    timestamp: str
    features_analyzed: int
    features_drifted: int
    overall_drift_detected: bool
    overall_psi: float
    feature_drifts: List[FeatureDrift]
    recommendations: List[str]


class DriftDetectionService:
    """
    C.22: Drift detection service using PSI and KL divergence.
    
    Monitors feature distribution changes between reference and comparison periods.
    """
    
    PSI_THRESHOLDS = {
        "low": 0.1,
        "medium": 0.2,
        "high": 0.5,
    }
    
    def __init__(
        self,
        n_bins: int = 10,
        psi_threshold: float = 0.2,
        epsilon: float = 1e-10,
    ):
        self.n_bins = n_bins
        self.psi_threshold = psi_threshold
        self.epsilon = epsilon
    
    def detect_drift(
        self,
        reference_data: List[Dict[str, Any]],
        comparison_data: List[Dict[str, Any]],
        feature_columns: Optional[List[str]] = None,
    ) -> DriftReport:
        """
        Detect drift between reference and comparison datasets.
        
        Args:
            reference_data: Baseline/training data
            comparison_data: New/production data to compare
            feature_columns: Columns to analyze (if None, infer from data)
            
        Returns:
            DriftReport with all feature drifts and recommendations
        """
        from uuid import uuid4
        
        if not reference_data or not comparison_data:
            return self._empty_report()
        
        columns = feature_columns or list(reference_data[0].keys())
        
        feature_drifts = []
        total_psi = 0.0
        
        for col in columns:
            ref_values = [row.get(col) for row in reference_data if row.get(col) is not None]
            comp_values = [row.get(col) for row in comparison_data if row.get(col) is not None]
            
            if not ref_values or not comp_values:
                continue
            
            if all(isinstance(v, (int, float)) for v in ref_values[:10]):
                drift = self._analyze_numeric_drift(col, ref_values, comp_values)
            else:
                drift = self._analyze_categorical_drift(col, ref_values, comp_values)
            
            feature_drifts.append(drift)
            total_psi += drift.psi
        
        overall_psi = total_psi / len(feature_drifts) if feature_drifts else 0.0
        features_drifted = sum(1 for d in feature_drifts if d.drift_detected)
        overall_drift = overall_psi >= self.psi_threshold or features_drifted > len(columns) * 0.3
        
        recommendations = self._generate_recommendations(
            feature_drifts, overall_psi, overall_drift
        )
        
        return DriftReport(
            report_id=str(uuid4()),
            timestamp=datetime.utcnow().isoformat(),
            features_analyzed=len(feature_drifts),
            features_drifted=features_drifted,
            overall_drift_detected=overall_drift,
            overall_psi=overall_psi,
            feature_drifts=feature_drifts,
            recommendations=recommendations,
        )
    
    def _analyze_numeric_drift(
        self,
        feature_name: str,
        ref_values: List[Any],
        comp_values: List[Any],
    ) -> FeatureDrift:
        """Analyze drift for numeric features"""
        ref_floats = [float(v) for v in ref_values]
        comp_floats = [float(v) for v in comp_values]
        
        all_values = ref_floats + comp_floats
        min_val = min(all_values)
        max_val = max(all_values)
        
        if min_val == max_val:
            return FeatureDrift(
                feature_name=feature_name,
                psi=0.0,
                kl_divergence=0.0,
                drift_detected=False,
                severity="none",
                reference_distribution={},
                comparison_distribution={},
            )
        
        bin_edges = self._compute_bin_edges(min_val, max_val, self.n_bins)
        
        ref_hist = self._compute_histogram(ref_floats, bin_edges)
        comp_hist = self._compute_histogram(comp_floats, bin_edges)
        
        psi = self._compute_psi(ref_hist, comp_hist)
        kl_div = self._compute_kl_divergence(ref_hist, comp_hist)
        
        severity = self._classify_severity(psi)
        drift_detected = psi >= self.psi_threshold
        
        return FeatureDrift(
            feature_name=feature_name,
            psi=psi,
            kl_divergence=kl_div,
            drift_detected=drift_detected,
            severity=severity,
            reference_distribution=ref_hist,
            comparison_distribution=comp_hist,
        )
    
    def _analyze_categorical_drift(
        self,
        feature_name: str,
        ref_values: List[Any],
        comp_values: List[Any],
    ) -> FeatureDrift:
        """Analyze drift for categorical features"""
        ref_strs = [str(v) for v in ref_values]
        comp_strs = [str(v) for v in comp_values]
        
        all_categories = set(ref_strs) | set(comp_strs)
        
        ref_counts = Counter(ref_strs)
        comp_counts = Counter(comp_strs)
        
        ref_total = len(ref_strs)
        comp_total = len(comp_strs)
        
        ref_dist = {cat: ref_counts.get(cat, 0) / ref_total for cat in all_categories}
        comp_dist = {cat: comp_counts.get(cat, 0) / comp_total for cat in all_categories}
        
        psi = sum(
            (comp_dist[cat] - ref_dist[cat]) * 
            math.log((comp_dist[cat] + self.epsilon) / (ref_dist[cat] + self.epsilon))
            for cat in all_categories
        )
        
        kl_div = sum(
            comp_dist[cat] * math.log((comp_dist[cat] + self.epsilon) / (ref_dist[cat] + self.epsilon))
            for cat in all_categories
            if comp_dist[cat] > 0
        )
        
        severity = self._classify_severity(psi)
        drift_detected = psi >= self.psi_threshold
        
        return FeatureDrift(
            feature_name=feature_name,
            psi=abs(psi),
            kl_divergence=abs(kl_div),
            drift_detected=drift_detected,
            severity=severity,
            reference_distribution=ref_dist,
            comparison_distribution=comp_dist,
        )
    
    def _compute_bin_edges(
        self,
        min_val: float,
        max_val: float,
        n_bins: int,
    ) -> List[float]:
        """Compute bin edges for histogram"""
        step = (max_val - min_val) / n_bins
        return [min_val + i * step for i in range(n_bins + 1)]
    
    def _compute_histogram(
        self,
        values: List[float],
        bin_edges: List[float],
    ) -> Dict[str, float]:
        """Compute normalized histogram"""
        n_bins = len(bin_edges) - 1
        counts = [0] * n_bins
        
        for v in values:
            for i in range(n_bins):
                if bin_edges[i] <= v < bin_edges[i + 1] or (i == n_bins - 1 and v == bin_edges[-1]):
                    counts[i] += 1
                    break
        
        total = len(values)
        hist = {
            f"bin_{i}": count / total if total > 0 else 0
            for i, count in enumerate(counts)
        }
        
        return hist
    
    def _compute_psi(
        self,
        ref_hist: Dict[str, float],
        comp_hist: Dict[str, float],
    ) -> float:
        """Compute Population Stability Index"""
        psi = 0.0
        
        for key in ref_hist:
            ref_p = ref_hist.get(key, 0) + self.epsilon
            comp_p = comp_hist.get(key, 0) + self.epsilon
            
            psi += (comp_p - ref_p) * math.log(comp_p / ref_p)
        
        return abs(psi)
    
    def _compute_kl_divergence(
        self,
        ref_hist: Dict[str, float],
        comp_hist: Dict[str, float],
    ) -> float:
        """Compute KL Divergence"""
        kl = 0.0
        
        for key in comp_hist:
            ref_p = ref_hist.get(key, 0) + self.epsilon
            comp_p = comp_hist.get(key, 0) + self.epsilon
            
            if comp_p > self.epsilon:
                kl += comp_p * math.log(comp_p / ref_p)
        
        return abs(kl)
    
    def _classify_severity(self, psi: float) -> str:
        """Classify drift severity based on PSI"""
        if psi < self.PSI_THRESHOLDS["low"]:
            return "none"
        elif psi < self.PSI_THRESHOLDS["medium"]:
            return "low"
        elif psi < self.PSI_THRESHOLDS["high"]:
            return "medium"
        else:
            return "high"
    
    def _generate_recommendations(
        self,
        feature_drifts: List[FeatureDrift],
        overall_psi: float,
        overall_drift: bool,
    ) -> List[str]:
        """Generate drift mitigation recommendations"""
        recommendations = []
        
        if not overall_drift and overall_psi < 0.1:
            recommendations.append(
                "No significant drift detected. Model performance should remain stable."
            )
            return recommendations
        
        high_drift = [d for d in feature_drifts if d.severity == "high"]
        medium_drift = [d for d in feature_drifts if d.severity == "medium"]
        
        if high_drift:
            feature_names = ", ".join(d.feature_name for d in high_drift[:3])
            recommendations.append(
                f"High drift detected in features: {feature_names}. "
                "Consider retraining the model with recent data."
            )
        
        if medium_drift:
            recommendations.append(
                f"{len(medium_drift)} features show moderate drift. "
                "Monitor model performance closely for degradation."
            )
        
        if overall_psi > 0.3:
            recommendations.append(
                "Substantial overall distribution shift detected. "
                "Recommend immediate model validation and potential retraining."
            )
        
        if len(high_drift) + len(medium_drift) > len(feature_drifts) * 0.5:
            recommendations.append(
                "Majority of features show drift. "
                "Consider if data collection or preprocessing has changed."
            )
        
        return recommendations
    
    def _empty_report(self) -> DriftReport:
        """Return empty report for edge cases"""
        from uuid import uuid4
        return DriftReport(
            report_id=str(uuid4()),
            timestamp=datetime.utcnow().isoformat(),
            features_analyzed=0,
            features_drifted=0,
            overall_drift_detected=False,
            overall_psi=0.0,
            feature_drifts=[],
            recommendations=["No data to analyze"],
        )


def detect_drift(
    reference_data: List[Dict[str, Any]],
    comparison_data: List[Dict[str, Any]],
    feature_columns: Optional[List[str]] = None,
) -> DriftReport:
    """Convenience function to detect drift"""
    service = DriftDetectionService()
    return service.detect_drift(reference_data, comparison_data, feature_columns)


__all__ = [
    "DriftDetectionService",
    "FeatureDrift",
    "DriftReport",
    "detect_drift",
]
