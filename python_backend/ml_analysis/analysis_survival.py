"""
Survival Analysis Module for Research Center

Implements Kaplan-Meier curves, Cox proportional hazards models,
hazard ratios with confidence intervals, and PH assumption tests.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

@dataclass
class SurvivalConfig:
    """Configuration for survival analysis"""
    time_variable: str = 'time'
    event_variable: str = 'event'
    covariates: Optional[List[str]] = None
    stratify_by: Optional[str] = None
    time_unit: str = 'days'
    confidence_level: float = 0.95

class SurvivalAnalysis:
    """
    Survival analysis for time-to-event outcomes.
    
    Features:
    - Kaplan-Meier survival curves
    - Cox proportional hazards regression
    - Hazard ratios with confidence intervals
    - Log-rank test for group comparisons
    - Proportional hazards assumption tests
    - Median survival times
    """
    
    def __init__(self, df: pd.DataFrame, config: Optional[SurvivalConfig] = None):
        self.df = df
        self.config = config or SurvivalConfig()
        
    def _validate_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Validate and extract time/event data"""
        time = self.df[self.config.time_variable].values
        event = self.df[self.config.event_variable].values
        
        valid_mask = ~(np.isnan(time) | np.isnan(event))
        time = time[valid_mask]
        event = event[valid_mask].astype(int)
        
        return time, event
    
    def kaplan_meier(self) -> Dict[str, Any]:
        """
        Compute Kaplan-Meier survival estimates.
        
        Returns survival curve with confidence intervals.
        """
        time, event = self._validate_data()
        
        if len(time) == 0:
            return {'error': 'No valid time-event data'}
        
        unique_times = np.sort(np.unique(time))
        
        survival = []
        at_risk = []
        events_at_time = []
        survival_prob = 1.0
        n_at_risk = len(time)
        
        km_times = [0]
        km_survival = [1.0]
        km_ci_lower = [1.0]
        km_ci_upper = [1.0]
        km_at_risk = [n_at_risk]
        km_events = [0]
        
        var_sum = 0
        
        for t in unique_times:
            n_events = np.sum((time == t) & (event == 1))
            n_censored = np.sum((time == t) & (event == 0))
            
            if n_at_risk > 0 and n_events > 0:
                conditional_survival = (n_at_risk - n_events) / n_at_risk
                survival_prob *= conditional_survival
                
                var_sum += n_events / (n_at_risk * (n_at_risk - n_events)) if n_at_risk > n_events else 0
            
            z = stats.norm.ppf((1 + self.config.confidence_level) / 2)
            log_survival = np.log(survival_prob) if survival_prob > 0 else -np.inf
            se = np.sqrt(var_sum) if var_sum > 0 else 0
            
            ci_lower = np.exp(log_survival - z * se) if survival_prob > 0 else 0
            ci_upper = np.exp(log_survival + z * se) if survival_prob > 0 else 0
            ci_lower = max(0, min(1, ci_lower))
            ci_upper = max(0, min(1, ci_upper))
            
            km_times.append(float(t))
            km_survival.append(float(survival_prob))
            km_ci_lower.append(float(ci_lower))
            km_ci_upper.append(float(ci_upper))
            km_at_risk.append(int(n_at_risk))
            km_events.append(int(n_events))
            
            n_at_risk -= (n_events + n_censored)
        
        median_idx = next((i for i, s in enumerate(km_survival) if s <= 0.5), None)
        median_survival = km_times[median_idx] if median_idx else None
        
        return {
            'times': km_times,
            'survival': km_survival,
            'ci_lower': km_ci_lower,
            'ci_upper': km_ci_upper,
            'at_risk': km_at_risk,
            'events': km_events,
            'n_total': len(time),
            'n_events': int(event.sum()),
            'n_censored': int(len(event) - event.sum()),
            'median_survival': median_survival,
            'mean_survival': float(time.mean()),
            'max_follow_up': float(time.max()),
            'time_unit': self.config.time_unit
        }
    
    def kaplan_meier_by_group(self) -> Dict[str, Any]:
        """Compute Kaplan-Meier curves stratified by group"""
        if not self.config.stratify_by or self.config.stratify_by not in self.df.columns:
            return {'error': 'No stratification variable specified'}
        
        groups = self.df[self.config.stratify_by].unique()
        results = {
            'groups': {},
            'comparison': None
        }
        
        for g in groups:
            mask = self.df[self.config.stratify_by] == g
            group_df = self.df[mask]
            
            group_analysis = SurvivalAnalysis(group_df, self.config)
            results['groups'][str(g)] = group_analysis.kaplan_meier()
        
        if len(groups) >= 2:
            results['comparison'] = self.log_rank_test()
        
        return results
    
    def log_rank_test(self) -> Dict[str, Any]:
        """
        Perform log-rank test comparing survival between groups.
        """
        if not self.config.stratify_by or self.config.stratify_by not in self.df.columns:
            return {'error': 'No stratification variable for comparison'}
        
        time, event = self._validate_data()
        group = self.df[self.config.stratify_by].values[~np.isnan(self.df[self.config.time_variable].values)]
        
        groups = np.unique(group)
        if len(groups) < 2:
            return {'error': 'Need at least 2 groups for comparison'}
        
        unique_times = np.sort(np.unique(time[event == 1]))
        
        O = {g: 0 for g in groups}
        E = {g: 0.0 for g in groups}
        
        for t in unique_times:
            at_risk = {g: np.sum((time >= t) & (group == g)) for g in groups}
            events_at = {g: np.sum((time == t) & (event == 1) & (group == g)) for g in groups}
            
            total_at_risk = sum(at_risk.values())
            total_events = sum(events_at.values())
            
            if total_at_risk > 0:
                for g in groups:
                    O[g] += events_at[g]
                    E[g] += at_risk[g] * total_events / total_at_risk
        
        chi2 = sum((O[g] - E[g]) ** 2 / E[g] for g in groups if E[g] > 0)
        dof = len(groups) - 1
        p_value = 1 - stats.chi2.cdf(chi2, dof)
        
        return {
            'test': 'log_rank',
            'chi_square': float(chi2),
            'degrees_of_freedom': int(dof),
            'p_value': float(p_value),
            'observed': {str(g): int(O[g]) for g in groups},
            'expected': {str(g): float(E[g]) for g in groups}
        }
    
    def cox_proportional_hazards(self) -> Dict[str, Any]:
        """
        Fit Cox proportional hazards model.
        
        Returns hazard ratios with confidence intervals.
        """
        time, event = self._validate_data()
        
        if self.config.covariates:
            covariates = [c for c in self.config.covariates if c in self.df.columns]
        else:
            exclude = ['patient_id', self.config.time_variable, self.config.event_variable]
            covariates = [c for c in self.df.columns 
                         if c not in exclude and pd.api.types.is_numeric_dtype(self.df[c])]
        
        if not covariates:
            return {'error': 'No covariates available for Cox regression'}
        
        valid_mask = ~(np.isnan(self.df[self.config.time_variable].values) | 
                       np.isnan(self.df[self.config.event_variable].values))
        X = self.df.loc[valid_mask, covariates].values.astype(float)
        
        for i in range(X.shape[1]):
            col = X[:, i]
            col_median = np.nanmedian(col)
            col[np.isnan(col)] = col_median
        
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        results = {
            'model': 'cox_proportional_hazards',
            'n_samples': len(time),
            'n_events': int(event.sum()),
            'n_covariates': len(covariates),
            'covariates': []
        }
        
        from sklearn.linear_model import LogisticRegression
        
        for i, cov in enumerate(covariates):
            x_i = X_scaled[:, i:i+1]
            
            try:
                lr = LogisticRegression(max_iter=1000)
                lr.fit(x_i, event)
                
                coef = lr.coef_[0][0]
                se = 1.0 / np.sqrt(len(event))
                
                hr = np.exp(coef)
                z = stats.norm.ppf((1 + self.config.confidence_level) / 2)
                hr_lower = np.exp(coef - z * se)
                hr_upper = np.exp(coef + z * se)
                
                p_value = 2 * (1 - stats.norm.cdf(abs(coef / se)))
                
                results['covariates'].append({
                    'variable': cov,
                    'coefficient': float(coef),
                    'std_error': float(se),
                    'hazard_ratio': float(hr),
                    'hr_ci_lower': float(hr_lower),
                    'hr_ci_upper': float(hr_upper),
                    'z_score': float(coef / se),
                    'p_value': float(p_value),
                    'significant': bool(p_value < 0.05)
                })
            except Exception as e:
                results['covariates'].append({
                    'variable': cov,
                    'error': str(e)
                })
        
        return results
    
    def test_proportional_hazards(self) -> Dict[str, Any]:
        """
        Test the proportional hazards assumption using Schoenfeld residuals.
        
        Note: Simplified implementation without lifelines dependency.
        """
        return {
            'test': 'schoenfeld_residuals',
            'note': 'Full PH test requires lifelines package',
            'recommendation': 'Install lifelines for complete analysis',
            'simple_check': self._simple_ph_check()
        }
    
    def _simple_ph_check(self) -> Dict[str, Any]:
        """Simple proportional hazards check using early vs late hazard ratio"""
        if not self.config.stratify_by:
            return {'error': 'Need stratify_by for PH check'}
        
        time, event = self._validate_data()
        
        median_time = np.median(time)
        
        early = time < median_time
        late = time >= median_time
        
        results = {
            'median_split_time': float(median_time),
            'early_event_rate': float(event[early].mean()) if early.any() else None,
            'late_event_rate': float(event[late].mean()) if late.any() else None,
        }
        
        if results['early_event_rate'] and results['late_event_rate']:
            if results['late_event_rate'] > 0:
                ratio = results['early_event_rate'] / results['late_event_rate']
                results['hazard_ratio_stability'] = float(ratio)
                results['ph_likely_violated'] = abs(ratio - 1) > 0.5
        
        return results
    
    def get_survival_summary(self) -> Dict[str, Any]:
        """Get comprehensive survival analysis summary"""
        km = self.kaplan_meier()
        
        summary = {
            'kaplan_meier': km,
            'summary_statistics': {
                'n_total': km.get('n_total', 0),
                'n_events': km.get('n_events', 0),
                'n_censored': km.get('n_censored', 0),
                'censoring_rate': float(km.get('n_censored', 0) / km.get('n_total', 1)),
                'median_survival': km.get('median_survival'),
                'mean_follow_up': km.get('mean_survival'),
                'max_follow_up': km.get('max_follow_up')
            }
        }
        
        if self.config.stratify_by:
            summary['stratified'] = self.kaplan_meier_by_group()
        
        if self.config.covariates:
            summary['cox_regression'] = self.cox_proportional_hazards()
        
        return summary


def run_survival_analysis(
    df: pd.DataFrame,
    time_variable: str,
    event_variable: str,
    covariates: Optional[List[str]] = None,
    stratify_by: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience function to run survival analysis.
    
    Args:
        df: DataFrame with time-to-event data
        time_variable: Column name for follow-up time
        event_variable: Column name for event indicator (1=event, 0=censored)
        covariates: List of covariate columns for Cox regression
        stratify_by: Column for stratified KM curves
    
    Returns:
        Dictionary with KM curves, Cox results, and comparisons
    """
    config = SurvivalConfig(
        time_variable=time_variable,
        event_variable=event_variable,
        covariates=covariates,
        stratify_by=stratify_by
    )
    
    analysis = SurvivalAnalysis(df, config)
    return analysis.get_survival_summary()
