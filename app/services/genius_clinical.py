"""
Clinical Operations Genius Features (E.9-E.11)
==============================================
Advanced clinical operations capabilities for alert management.

E.9: Alert budget tuning per clinic
E.10: Alert burden fairness checker (subgroup alert distribution)
E.11: Dynamic threshold by clinic workload
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import statistics

logger = logging.getLogger(__name__)


class WorkloadLevel(str, Enum):
    """Clinic workload levels"""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AlertBudget:
    """Alert budget configuration for a clinic"""
    clinic_id: str
    daily_alert_limit: int
    current_daily_count: int
    weekly_alert_limit: int
    current_weekly_count: int
    priority_distribution: Dict[str, float]
    auto_escalation_threshold: float
    created_at: str
    updated_at: str


@dataclass
class SubgroupAlertDistribution:
    """Alert distribution across a patient subgroup"""
    subgroup_name: str
    subgroup_size: int
    alert_count: int
    alert_rate: float
    expected_rate: float
    deviation: float
    is_over_represented: bool
    is_under_represented: bool


@dataclass
class FairnessReport:
    """Alert burden fairness analysis report"""
    clinic_id: str
    analysis_period_days: int
    total_alerts: int
    subgroup_distributions: List[SubgroupAlertDistribution]
    fairness_score: float
    warnings: List[str]
    recommendations: List[str]
    generated_at: str


@dataclass
class DynamicThreshold:
    """Dynamically calculated alert threshold"""
    clinic_id: str
    base_threshold: float
    adjusted_threshold: float
    workload_level: WorkloadLevel
    adjustment_factor: float
    reason: str
    valid_until: str


class GeniusClinicalService:
    """
    E.9-E.11: Clinical Operations Genius Features
    
    Provides intelligent alert management and workload balancing.
    """
    
    DEFAULT_DAILY_LIMIT = 50
    DEFAULT_WEEKLY_LIMIT = 200
    FAIRNESS_DEVIATION_THRESHOLD = 0.3
    
    def __init__(self):
        self._alert_budgets: Dict[str, AlertBudget] = {}
        self._workload_cache: Dict[str, Tuple[WorkloadLevel, datetime]] = {}
        logger.info("GeniusClinicalService initialized")
    
    def configure_alert_budget(
        self,
        clinic_id: str,
        daily_limit: Optional[int] = None,
        weekly_limit: Optional[int] = None,
        priority_distribution: Optional[Dict[str, float]] = None
    ) -> AlertBudget:
        """
        E.9: Alert budget tuning per clinic.
        
        Configure alert budgets to prevent alert fatigue while ensuring
        critical alerts are not suppressed.
        
        Args:
            clinic_id: Unique clinic identifier
            daily_limit: Maximum daily alerts (default: 50)
            weekly_limit: Maximum weekly alerts (default: 200)
            priority_distribution: Distribution across priority levels
            
        Returns:
            Configured AlertBudget
        """
        now = datetime.utcnow().isoformat()
        
        default_priority_dist = {
            "critical": 0.15,
            "high": 0.25,
            "medium": 0.35,
            "low": 0.25
        }
        
        budget = AlertBudget(
            clinic_id=clinic_id,
            daily_alert_limit=daily_limit or self.DEFAULT_DAILY_LIMIT,
            current_daily_count=0,
            weekly_alert_limit=weekly_limit or self.DEFAULT_WEEKLY_LIMIT,
            current_weekly_count=0,
            priority_distribution=priority_distribution or default_priority_dist,
            auto_escalation_threshold=0.8,
            created_at=now,
            updated_at=now
        )
        
        self._alert_budgets[clinic_id] = budget
        logger.info(f"Configured alert budget for clinic {clinic_id}: daily={budget.daily_alert_limit}, weekly={budget.weekly_alert_limit}")
        
        return budget
    
    def get_alert_budget(self, clinic_id: str) -> Optional[AlertBudget]:
        """Get current alert budget for a clinic"""
        return self._alert_budgets.get(clinic_id)
    
    def update_alert_count(
        self,
        clinic_id: str,
        alert_count: int,
        period: str = "daily"
    ) -> AlertBudget:
        """Update alert count for a clinic"""
        budget = self._alert_budgets.get(clinic_id)
        if not budget:
            budget = self.configure_alert_budget(clinic_id)
        
        if period == "daily":
            budget.current_daily_count = alert_count
        elif period == "weekly":
            budget.current_weekly_count = alert_count
        
        budget.updated_at = datetime.utcnow().isoformat()
        return budget
    
    def check_budget_available(
        self,
        clinic_id: str,
        priority: str = "medium"
    ) -> Tuple[bool, str]:
        """
        Check if alert budget is available for a new alert.
        
        Returns:
            Tuple of (is_available, reason)
        """
        budget = self._alert_budgets.get(clinic_id)
        if not budget:
            return True, "No budget configured - allowing alert"
        
        if budget.current_daily_count >= budget.daily_alert_limit:
            if priority == "critical":
                return True, "Critical alert allowed despite budget exhaustion"
            return False, f"Daily alert budget exhausted ({budget.current_daily_count}/{budget.daily_alert_limit})"
        
        if budget.current_weekly_count >= budget.weekly_alert_limit:
            if priority in ["critical", "high"]:
                return True, "High-priority alert allowed despite weekly budget exhaustion"
            return False, f"Weekly alert budget exhausted ({budget.current_weekly_count}/{budget.weekly_alert_limit})"
        
        utilization = budget.current_daily_count / budget.daily_alert_limit
        priority_cap = budget.priority_distribution.get(priority, 0.25)
        
        if priority == "low" and utilization > 0.7:
            return False, "Low-priority alert suppressed due to high budget utilization"
        
        return True, "Alert within budget"
    
    def analyze_alert_fairness(
        self,
        clinic_id: str,
        alerts_by_subgroup: Dict[str, List[Dict[str, Any]]],
        subgroup_sizes: Dict[str, int],
        analysis_period_days: int = 30
    ) -> FairnessReport:
        """
        E.10: Alert burden fairness checker.
        
        Analyzes if certain patient subgroups receive disproportionately
        more or fewer alerts compared to their population size.
        
        Args:
            clinic_id: Clinic identifier
            alerts_by_subgroup: Dict mapping subgroup name to list of alerts
            subgroup_sizes: Dict mapping subgroup name to population size
            analysis_period_days: Period of analysis in days
            
        Returns:
            FairnessReport with distribution analysis and recommendations
        """
        distributions = []
        warnings = []
        recommendations = []
        
        total_patients = sum(subgroup_sizes.values())
        total_alerts = sum(len(alerts) for alerts in alerts_by_subgroup.values())
        
        if total_patients == 0 or total_alerts == 0:
            return FairnessReport(
                clinic_id=clinic_id,
                analysis_period_days=analysis_period_days,
                total_alerts=0,
                subgroup_distributions=[],
                fairness_score=1.0,
                warnings=["Insufficient data for fairness analysis"],
                recommendations=[],
                generated_at=datetime.utcnow().isoformat()
            )
        
        overall_alert_rate = total_alerts / total_patients
        deviations = []
        
        for subgroup_name, subgroup_size in subgroup_sizes.items():
            if subgroup_size == 0:
                continue
            
            alert_count = len(alerts_by_subgroup.get(subgroup_name, []))
            alert_rate = alert_count / subgroup_size
            expected_rate = overall_alert_rate
            
            deviation = (alert_rate - expected_rate) / expected_rate if expected_rate > 0 else 0
            deviations.append(abs(deviation))
            
            is_over = deviation > self.FAIRNESS_DEVIATION_THRESHOLD
            is_under = deviation < -self.FAIRNESS_DEVIATION_THRESHOLD
            
            dist = SubgroupAlertDistribution(
                subgroup_name=subgroup_name,
                subgroup_size=subgroup_size,
                alert_count=alert_count,
                alert_rate=round(alert_rate, 4),
                expected_rate=round(expected_rate, 4),
                deviation=round(deviation, 4),
                is_over_represented=is_over,
                is_under_represented=is_under
            )
            distributions.append(dist)
            
            if is_over:
                warnings.append(
                    f"Subgroup '{subgroup_name}' receives {abs(deviation)*100:.1f}% more alerts than expected"
                )
                recommendations.append(
                    f"Review alert thresholds for '{subgroup_name}' - may need calibration"
                )
            
            if is_under:
                warnings.append(
                    f"Subgroup '{subgroup_name}' receives {abs(deviation)*100:.1f}% fewer alerts than expected"
                )
                recommendations.append(
                    f"Review alert sensitivity for '{subgroup_name}' - may be missing deterioration signals"
                )
        
        if deviations:
            avg_deviation = statistics.mean(deviations)
            fairness_score = max(0, 1 - avg_deviation)
        else:
            fairness_score = 1.0
        
        report = FairnessReport(
            clinic_id=clinic_id,
            analysis_period_days=analysis_period_days,
            total_alerts=total_alerts,
            subgroup_distributions=distributions,
            fairness_score=round(fairness_score, 4),
            warnings=warnings,
            recommendations=recommendations,
            generated_at=datetime.utcnow().isoformat()
        )
        
        logger.info(f"Fairness analysis for clinic {clinic_id}: score={fairness_score:.2f}, warnings={len(warnings)}")
        return report
    
    def calculate_dynamic_threshold(
        self,
        clinic_id: str,
        base_threshold: float,
        current_workload: Dict[str, Any]
    ) -> DynamicThreshold:
        """
        E.11: Dynamic threshold by clinic workload.
        
        Adjusts alert thresholds based on current clinic workload to prevent
        alert overload during busy periods while maintaining sensitivity
        during quieter times.
        
        Args:
            clinic_id: Clinic identifier
            base_threshold: Base alert threshold (0-1)
            current_workload: Dict with workload indicators
            
        Returns:
            DynamicThreshold with adjusted values
        """
        active_patients = current_workload.get("active_patients", 0)
        pending_alerts = current_workload.get("pending_alerts", 0)
        staff_available = current_workload.get("staff_available", 1)
        avg_response_time_mins = current_workload.get("avg_response_time_mins", 15)
        
        if staff_available == 0:
            staff_available = 1
        
        patients_per_staff = active_patients / staff_available
        alerts_per_staff = pending_alerts / staff_available
        
        if patients_per_staff > 50 or alerts_per_staff > 10 or avg_response_time_mins > 60:
            workload_level = WorkloadLevel.CRITICAL
            adjustment_factor = 1.3
            reason = "Critical workload - raising thresholds to reduce alert volume"
        elif patients_per_staff > 30 or alerts_per_staff > 5 or avg_response_time_mins > 30:
            workload_level = WorkloadLevel.HIGH
            adjustment_factor = 1.15
            reason = "High workload - slightly raising thresholds"
        elif patients_per_staff > 15 or alerts_per_staff > 2:
            workload_level = WorkloadLevel.MODERATE
            adjustment_factor = 1.0
            reason = "Moderate workload - using base thresholds"
        else:
            workload_level = WorkloadLevel.LOW
            adjustment_factor = 0.9
            reason = "Low workload - lowering thresholds for higher sensitivity"
        
        adjusted_threshold = min(0.95, base_threshold * adjustment_factor)
        
        valid_until = (datetime.utcnow() + timedelta(hours=1)).isoformat()
        
        threshold = DynamicThreshold(
            clinic_id=clinic_id,
            base_threshold=base_threshold,
            adjusted_threshold=round(adjusted_threshold, 4),
            workload_level=workload_level,
            adjustment_factor=adjustment_factor,
            reason=reason,
            valid_until=valid_until
        )
        
        self._workload_cache[clinic_id] = (workload_level, datetime.utcnow())
        
        logger.info(f"Dynamic threshold for clinic {clinic_id}: {base_threshold} -> {adjusted_threshold} ({workload_level.value})")
        return threshold
    
    def get_workload_level(self, clinic_id: str) -> Optional[WorkloadLevel]:
        """Get cached workload level for a clinic"""
        cached = self._workload_cache.get(clinic_id)
        if cached:
            level, timestamp = cached
            if datetime.utcnow() - timestamp < timedelta(hours=1):
                return level
        return None


_genius_clinical_service: Optional[GeniusClinicalService] = None


def get_genius_clinical_service() -> GeniusClinicalService:
    """Get or create singleton GeniusClinicalService"""
    global _genius_clinical_service
    if _genius_clinical_service is None:
        _genius_clinical_service = GeniusClinicalService()
    return _genius_clinical_service
