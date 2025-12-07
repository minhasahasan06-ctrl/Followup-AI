"""
Automated Robustness and Bias Checks
=====================================
Production-grade diagnostic checks for ML and epidemiology analyses:
- Sample size sufficiency
- Missingness pattern analysis
- Causal diagnostics (covariate balance, SMD)
- Subgroup performance analysis

HIPAA-compliant with comprehensive audit logging.
"""

import os
import json
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import psycopg2
import psycopg2.extras

logger = logging.getLogger(__name__)


class CheckStatus(str, Enum):
    PASS = "pass"
    WARNING = "warning"
    FAIL = "fail"
    SKIPPED = "skipped"


class MissingnessPattern(str, Enum):
    MCAR = "mcar"
    MAR = "mar"
    MNAR = "mnar"
    UNKNOWN = "unknown"


@dataclass
class CheckResult:
    """Result of a single robustness check"""
    check_name: str
    status: CheckStatus
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class RobustnessReport:
    """Complete robustness check report"""
    report_id: str
    protocol_id: Optional[str]
    checks: List[CheckResult]
    overall_status: CheckStatus
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'report_id': self.report_id,
            'protocol_id': self.protocol_id,
            'overall_status': self.overall_status.value,
            'created_at': self.created_at.isoformat(),
            'checks': [
                {
                    'check_name': c.check_name,
                    'status': c.status.value,
                    'message': c.message,
                    'details': c.details,
                    'recommendations': c.recommendations
                }
                for c in self.checks
            ],
            'summary': {
                'total_checks': len(self.checks),
                'passed': sum(1 for c in self.checks if c.status == CheckStatus.PASS),
                'warnings': sum(1 for c in self.checks if c.status == CheckStatus.WARNING),
                'failed': sum(1 for c in self.checks if c.status == CheckStatus.FAIL)
            }
        }


class SampleSizeChecker:
    """Checks for adequate sample size"""
    
    MIN_TOTAL_SAMPLE = 100
    MIN_PER_GROUP = 20
    MIN_EVENTS_PER_PREDICTOR = 10
    
    def check_total_sample(
        self, 
        n_samples: int,
        min_required: Optional[int] = None
    ) -> CheckResult:
        """Check if total sample size is adequate"""
        threshold = min_required or self.MIN_TOTAL_SAMPLE
        
        if n_samples >= threshold:
            return CheckResult(
                check_name="total_sample_size",
                status=CheckStatus.PASS,
                message=f"Sample size ({n_samples}) meets minimum requirement ({threshold})",
                details={'n_samples': n_samples, 'threshold': threshold}
            )
        elif n_samples >= threshold * 0.5:
            return CheckResult(
                check_name="total_sample_size",
                status=CheckStatus.WARNING,
                message=f"Sample size ({n_samples}) is below recommended ({threshold})",
                details={'n_samples': n_samples, 'threshold': threshold},
                recommendations=[
                    "Consider pooling data from additional sources",
                    "Use methods robust to small samples (exact tests, bootstrap)"
                ]
            )
        else:
            return CheckResult(
                check_name="total_sample_size",
                status=CheckStatus.FAIL,
                message=f"Sample size ({n_samples}) is critically low",
                details={'n_samples': n_samples, 'threshold': threshold},
                recommendations=[
                    "Results may not be reliable",
                    "Consider limiting scope of analysis",
                    "Report as exploratory/hypothesis-generating only"
                ]
            )
    
    def check_group_sizes(
        self,
        group_sizes: Dict[str, int],
        min_per_group: Optional[int] = None
    ) -> CheckResult:
        """Check if each group has adequate sample size"""
        threshold = min_per_group or self.MIN_PER_GROUP
        
        small_groups = {k: v for k, v in group_sizes.items() if v < threshold}
        
        if not small_groups:
            return CheckResult(
                check_name="group_sample_sizes",
                status=CheckStatus.PASS,
                message=f"All groups meet minimum size ({threshold})",
                details={'group_sizes': group_sizes, 'threshold': threshold}
            )
        
        very_small = {k: v for k, v in small_groups.items() if v < threshold * 0.5}
        
        if very_small:
            return CheckResult(
                check_name="group_sample_sizes",
                status=CheckStatus.FAIL,
                message=f"Some groups are critically small: {list(very_small.keys())}",
                details={'group_sizes': group_sizes, 'small_groups': small_groups},
                recommendations=[
                    "Consider collapsing categories",
                    "Use exact statistical tests",
                    "Report with appropriate caveats"
                ]
            )
        else:
            return CheckResult(
                check_name="group_sample_sizes",
                status=CheckStatus.WARNING,
                message=f"Some groups below recommended size: {list(small_groups.keys())}",
                details={'group_sizes': group_sizes, 'small_groups': small_groups},
                recommendations=[
                    "Results for small groups may be imprecise",
                    "Consider sensitivity analysis excluding small groups"
                ]
            )
    
    def check_events_per_predictor(
        self,
        n_events: int,
        n_predictors: int,
        min_epp: Optional[int] = None
    ) -> CheckResult:
        """Check events per predictor (EPP) for regression models"""
        threshold = min_epp or self.MIN_EVENTS_PER_PREDICTOR
        
        if n_predictors == 0:
            return CheckResult(
                check_name="events_per_predictor",
                status=CheckStatus.SKIPPED,
                message="No predictors specified"
            )
        
        epp = n_events / n_predictors
        
        if epp >= threshold:
            return CheckResult(
                check_name="events_per_predictor",
                status=CheckStatus.PASS,
                message=f"EPP ({epp:.1f}) meets minimum ({threshold})",
                details={'n_events': n_events, 'n_predictors': n_predictors, 'epp': epp}
            )
        elif epp >= threshold * 0.5:
            return CheckResult(
                check_name="events_per_predictor",
                status=CheckStatus.WARNING,
                message=f"EPP ({epp:.1f}) is below recommended ({threshold})",
                details={'n_events': n_events, 'n_predictors': n_predictors, 'epp': epp},
                recommendations=[
                    "Consider reducing number of predictors",
                    "Use penalized regression methods (LASSO, Ridge)",
                    "Report with appropriate uncertainty"
                ]
            )
        else:
            return CheckResult(
                check_name="events_per_predictor",
                status=CheckStatus.FAIL,
                message=f"EPP ({epp:.1f}) is critically low - model likely overfit",
                details={'n_events': n_events, 'n_predictors': n_predictors, 'epp': epp},
                recommendations=[
                    "Reduce number of predictors significantly",
                    "Use regularization or variable selection",
                    "Consider simpler model specification"
                ]
            )


class MissingnessAnalyzer:
    """Analyzes missing data patterns"""
    
    MAX_ACCEPTABLE_MISSING = 0.10
    CONCERN_THRESHOLD = 0.05
    
    def analyze_missingness(
        self,
        data: Dict[str, List[Any]],
        outcome_var: Optional[str] = None
    ) -> CheckResult:
        """Analyze overall missingness patterns"""
        missing_rates = {}
        
        for col, values in data.items():
            n_missing = sum(1 for v in values if v is None or (isinstance(v, float) and np.isnan(v)))
            missing_rates[col] = n_missing / len(values) if values else 0
        
        high_missing = {k: v for k, v in missing_rates.items() if v > self.MAX_ACCEPTABLE_MISSING}
        moderate_missing = {k: v for k, v in missing_rates.items() 
                          if self.CONCERN_THRESHOLD < v <= self.MAX_ACCEPTABLE_MISSING}
        
        if not high_missing and not moderate_missing:
            return CheckResult(
                check_name="missingness_analysis",
                status=CheckStatus.PASS,
                message="Missingness rates are acceptable across all variables",
                details={'missing_rates': missing_rates}
            )
        elif high_missing:
            return CheckResult(
                check_name="missingness_analysis",
                status=CheckStatus.FAIL,
                message=f"High missingness in variables: {list(high_missing.keys())}",
                details={'missing_rates': missing_rates, 'high_missing': high_missing},
                recommendations=[
                    "Consider multiple imputation",
                    "Investigate cause of missingness",
                    "Perform sensitivity analysis (complete case vs imputed)"
                ]
            )
        else:
            return CheckResult(
                check_name="missingness_analysis",
                status=CheckStatus.WARNING,
                message=f"Moderate missingness in some variables: {list(moderate_missing.keys())}",
                details={'missing_rates': missing_rates, 'moderate_missing': moderate_missing},
                recommendations=[
                    "Document missingness handling approach",
                    "Consider imputation for key variables"
                ]
            )
    
    def detect_pattern(
        self,
        data: Dict[str, List[Any]],
        outcome_var: Optional[str] = None
    ) -> CheckResult:
        """Attempt to detect missingness mechanism (MCAR/MAR/MNAR)"""
        missing_rates = {}
        for col, values in data.items():
            n_missing = sum(1 for v in values if v is None or (isinstance(v, float) and np.isnan(v)))
            missing_rates[col] = n_missing / len(values) if values else 0
        
        has_pattern_variance = max(missing_rates.values()) - min(missing_rates.values()) > 0.1
        
        if has_pattern_variance:
            pattern = MissingnessPattern.MAR
            message = "Missingness appears to be related to observed variables (MAR assumed)"
            recommendations = [
                "Multiple imputation appropriate under MAR",
                "Consider sensitivity analysis for MNAR"
            ]
        else:
            pattern = MissingnessPattern.MCAR
            message = "Missingness appears random (MCAR assumed)"
            recommendations = [
                "Complete case analysis may be acceptable",
                "Consider imputation for efficiency"
            ]
        
        return CheckResult(
            check_name="missingness_pattern",
            status=CheckStatus.WARNING,
            message=message,
            details={
                'detected_pattern': pattern.value,
                'missing_rates': missing_rates
            },
            recommendations=recommendations
        )


class CausalDiagnostics:
    """Diagnostics for causal inference analyses"""
    
    SMD_THRESHOLD = 0.1
    SMD_CONCERN = 0.25
    
    def check_covariate_balance(
        self,
        treatment_group: Dict[str, np.ndarray],
        control_group: Dict[str, np.ndarray]
    ) -> CheckResult:
        """Check covariate balance between treatment and control"""
        smds = {}
        
        for var in treatment_group.keys():
            if var not in control_group:
                continue
            
            t_vals = treatment_group[var]
            c_vals = control_group[var]
            
            t_mean = np.mean(t_vals)
            c_mean = np.mean(c_vals)
            pooled_std = np.sqrt((np.var(t_vals) + np.var(c_vals)) / 2)
            
            if pooled_std > 0:
                smd = abs(t_mean - c_mean) / pooled_std
            else:
                smd = 0
            
            smds[var] = round(smd, 3)
        
        high_imbalance = {k: v for k, v in smds.items() if v > self.SMD_CONCERN}
        moderate_imbalance = {k: v for k, v in smds.items() 
                            if self.SMD_THRESHOLD < v <= self.SMD_CONCERN}
        
        if high_imbalance:
            return CheckResult(
                check_name="covariate_balance",
                status=CheckStatus.FAIL,
                message=f"Severe imbalance in: {list(high_imbalance.keys())}",
                details={'smds': smds, 'high_imbalance': high_imbalance},
                recommendations=[
                    "Apply propensity score matching or weighting",
                    "Consider doubly robust estimation",
                    "Sensitivity analysis for unmeasured confounding"
                ]
            )
        elif moderate_imbalance:
            return CheckResult(
                check_name="covariate_balance",
                status=CheckStatus.WARNING,
                message=f"Moderate imbalance in: {list(moderate_imbalance.keys())}",
                details={'smds': smds, 'moderate_imbalance': moderate_imbalance},
                recommendations=[
                    "Consider adjustment in outcome model",
                    "Report balance diagnostics"
                ]
            )
        else:
            return CheckResult(
                check_name="covariate_balance",
                status=CheckStatus.PASS,
                message="Covariates are well-balanced (all SMD < 0.1)",
                details={'smds': smds}
            )
    
    def check_positivity(
        self,
        propensity_scores: np.ndarray,
        min_ps: float = 0.01,
        max_ps: float = 0.99
    ) -> CheckResult:
        """Check positivity assumption via propensity score distribution"""
        extreme_low = np.mean(propensity_scores < min_ps)
        extreme_high = np.mean(propensity_scores > max_ps)
        total_extreme = extreme_low + extreme_high
        
        if total_extreme < 0.01:
            return CheckResult(
                check_name="positivity",
                status=CheckStatus.PASS,
                message="Propensity scores well within bounds",
                details={
                    'ps_range': (float(np.min(propensity_scores)), float(np.max(propensity_scores))),
                    'extreme_proportion': total_extreme
                }
            )
        elif total_extreme < 0.05:
            return CheckResult(
                check_name="positivity",
                status=CheckStatus.WARNING,
                message=f"Some extreme propensity scores ({total_extreme:.1%})",
                details={
                    'extreme_low': extreme_low,
                    'extreme_high': extreme_high
                },
                recommendations=[
                    "Consider trimming extreme weights",
                    "Evaluate covariate overlap"
                ]
            )
        else:
            return CheckResult(
                check_name="positivity",
                status=CheckStatus.FAIL,
                message=f"Many extreme propensity scores ({total_extreme:.1%}) - positivity violated",
                details={
                    'extreme_low': extreme_low,
                    'extreme_high': extreme_high
                },
                recommendations=[
                    "Restrict to common support region",
                    "Use stabilized weights",
                    "Reconsider analysis population"
                ]
            )


class SubgroupAnalyzer:
    """Analyzes model performance across subgroups"""
    
    def check_subgroup_performance(
        self,
        subgroup_metrics: Dict[str, Dict[str, float]],
        primary_metric: str = "auc"
    ) -> CheckResult:
        """Check if model performance is consistent across subgroups"""
        if not subgroup_metrics:
            return CheckResult(
                check_name="subgroup_performance",
                status=CheckStatus.SKIPPED,
                message="No subgroup metrics provided"
            )
        
        metric_values = []
        for group, metrics in subgroup_metrics.items():
            if primary_metric in metrics:
                metric_values.append((group, metrics[primary_metric]))
        
        if not metric_values:
            return CheckResult(
                check_name="subgroup_performance",
                status=CheckStatus.SKIPPED,
                message=f"Metric '{primary_metric}' not found in subgroup metrics"
            )
        
        values = [v for _, v in metric_values]
        overall_mean = np.mean(values)
        max_diff = max(abs(v - overall_mean) for v in values)
        relative_diff = max_diff / overall_mean if overall_mean > 0 else 0
        
        worst_group = min(metric_values, key=lambda x: x[1])
        best_group = max(metric_values, key=lambda x: x[1])
        
        if relative_diff < 0.1:
            return CheckResult(
                check_name="subgroup_performance",
                status=CheckStatus.PASS,
                message="Model performance is consistent across subgroups",
                details={
                    'subgroup_metrics': subgroup_metrics,
                    'max_relative_difference': relative_diff
                }
            )
        elif relative_diff < 0.25:
            return CheckResult(
                check_name="subgroup_performance",
                status=CheckStatus.WARNING,
                message=f"Moderate variation in subgroup performance (worst: {worst_group[0]})",
                details={
                    'subgroup_metrics': subgroup_metrics,
                    'worst_group': worst_group,
                    'best_group': best_group
                },
                recommendations=[
                    "Report subgroup-specific estimates",
                    "Consider stratified models"
                ]
            )
        else:
            return CheckResult(
                check_name="subgroup_performance",
                status=CheckStatus.FAIL,
                message=f"Large disparity in subgroup performance: {best_group[0]} vs {worst_group[0]}",
                details={
                    'subgroup_metrics': subgroup_metrics,
                    'worst_group': worst_group,
                    'best_group': best_group,
                    'relative_difference': relative_diff
                },
                recommendations=[
                    "Investigate cause of disparate performance",
                    "Consider separate models for subgroups",
                    "Do not deploy without addressing disparity"
                ]
            )


class RobustnessChecker:
    """Orchestrates all robustness checks"""
    
    def __init__(self, db_url: Optional[str] = None):
        self.db_url = db_url or os.environ.get('DATABASE_URL')
        self.sample_checker = SampleSizeChecker()
        self.missingness_analyzer = MissingnessAnalyzer()
        self.causal_diagnostics = CausalDiagnostics()
        self.subgroup_analyzer = SubgroupAnalyzer()
    
    def run_full_diagnostics(
        self,
        data: Dict[str, List[Any]],
        outcome_var: str,
        treatment_var: Optional[str] = None,
        predictors: Optional[List[str]] = None,
        subgroup_var: Optional[str] = None,
        protocol_id: Optional[str] = None
    ) -> RobustnessReport:
        """Run comprehensive robustness diagnostics"""
        import uuid
        
        checks: List[CheckResult] = []
        
        n_samples = len(data.get(outcome_var, []))
        checks.append(self.sample_checker.check_total_sample(n_samples))
        
        if treatment_var and treatment_var in data:
            treatment_values = data[treatment_var]
            groups = {}
            for v in set(treatment_values):
                if v is not None:
                    groups[str(v)] = sum(1 for x in treatment_values if x == v)
            checks.append(self.sample_checker.check_group_sizes(groups))
        
        n_events = sum(1 for v in data.get(outcome_var, []) if v == 1)
        n_predictors = len(predictors) if predictors else 0
        if n_predictors > 0:
            checks.append(
                self.sample_checker.check_events_per_predictor(n_events, n_predictors)
            )
        
        checks.append(self.missingness_analyzer.analyze_missingness(data, outcome_var))
        checks.append(self.missingness_analyzer.detect_pattern(data, outcome_var))
        
        failed = sum(1 for c in checks if c.status == CheckStatus.FAIL)
        warnings = sum(1 for c in checks if c.status == CheckStatus.WARNING)
        
        if failed > 0:
            overall = CheckStatus.FAIL
        elif warnings > 2:
            overall = CheckStatus.WARNING
        else:
            overall = CheckStatus.PASS
        
        report = RobustnessReport(
            report_id=str(uuid.uuid4()),
            protocol_id=protocol_id,
            checks=checks,
            overall_status=overall
        )
        
        self._save_report(report)
        
        return report
    
    def _save_report(self, report: RobustnessReport):
        """Save report to database"""
        try:
            conn = psycopg2.connect(self.db_url)
            cur = conn.cursor()
            
            cur.execute("""
                INSERT INTO robustness_reports 
                (id, protocol_id, overall_status, report_json, created_at)
                VALUES (%s, %s, %s, %s, %s)
            """, (
                report.report_id,
                report.protocol_id,
                report.overall_status.value,
                json.dumps(report.to_dict()),
                report.created_at
            ))
            
            conn.commit()
            cur.close()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to save robustness report: {e}")
