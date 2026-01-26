"""
Causal Analysis Module for Research Center

Implements propensity score estimation, IPTW/matching,
ATE estimates with confidence intervals, and covariate balance diagnostics.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings('ignore')

@dataclass
class CausalConfig:
    """Configuration for causal analysis"""
    treatment_variable: str = 'treatment'
    outcome_variable: str = 'outcome'
    covariates: Optional[List[str]] = None
    method: str = 'iptw'  # 'iptw', 'matching', 'stratification'
    matching_ratio: int = 1
    caliper: Optional[float] = None
    trim_propensity: Tuple[float, float] = (0.01, 0.99)
    bootstrap_n: int = 1000
    confidence_level: float = 0.95

class CausalAnalysis:
    """
    Causal inference analysis for treatment effect estimation.
    
    Features:
    - Propensity score estimation
    - Inverse probability of treatment weighting (IPTW)
    - Nearest neighbor matching
    - Propensity score stratification
    - Average treatment effect (ATE) estimation
    - Covariate balance diagnostics
    - Sensitivity analysis
    """
    
    def __init__(self, df: pd.DataFrame, config: Optional[CausalConfig] = None):
        self.df = df.copy()
        self.config = config or CausalConfig()
        self.propensity_scores = None
        self.weights = None
        self.matched_data = None
        
    def _prepare_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """Prepare treatment, outcome, and covariates"""
        if self.config.covariates:
            covariates = [c for c in self.config.covariates if c in self.df.columns]
        else:
            exclude = ['patient_id', self.config.treatment_variable, self.config.outcome_variable]
            covariates = [c for c in self.df.columns 
                         if c not in exclude and pd.api.types.is_numeric_dtype(self.df[c])]
        
        treatment = self.df[self.config.treatment_variable].values.astype(int)
        outcome = self.df[self.config.outcome_variable].values.astype(float)
        
        X = self.df[covariates].values.astype(float)
        
        imputer = SimpleImputer(strategy='median')
        X = imputer.fit_transform(X)
        
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        return treatment, outcome, X, covariates
    
    def estimate_propensity_scores(self) -> Dict[str, Any]:
        """
        Estimate propensity scores using logistic regression.
        
        Returns propensity score distribution and model diagnostics.
        """
        treatment, outcome, X, covariates = self._prepare_data()
        
        valid_mask = ~(np.isnan(treatment) | np.isnan(outcome))
        treatment = treatment[valid_mask]
        outcome = outcome[valid_mask]
        X = X[valid_mask]
        
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X, treatment)
        
        ps = model.predict_proba(X)[:, 1]
        self.propensity_scores = ps
        
        ps = np.clip(ps, self.config.trim_propensity[0], self.config.trim_propensity[1])
        
        return {
            'n_treated': int(treatment.sum()),
            'n_control': int(len(treatment) - treatment.sum()),
            'propensity_treated': {
                'mean': float(ps[treatment == 1].mean()),
                'std': float(ps[treatment == 1].std()),
                'median': float(np.median(ps[treatment == 1])),
                'min': float(ps[treatment == 1].min()),
                'max': float(ps[treatment == 1].max())
            },
            'propensity_control': {
                'mean': float(ps[treatment == 0].mean()),
                'std': float(ps[treatment == 0].std()),
                'median': float(np.median(ps[treatment == 0])),
                'min': float(ps[treatment == 0].min()),
                'max': float(ps[treatment == 0].max())
            },
            'overlap_region': {
                'min': float(max(ps[treatment == 0].min(), ps[treatment == 1].min())),
                'max': float(min(ps[treatment == 0].max(), ps[treatment == 1].max()))
            },
            'histogram_treated': self._histogram(ps[treatment == 1]),
            'histogram_control': self._histogram(ps[treatment == 0])
        }
    
    def _histogram(self, values: np.ndarray, n_bins: int = 20) -> Dict[str, List]:
        """Compute histogram"""
        hist, bin_edges = np.histogram(values, bins=n_bins, range=(0, 1))
        return {
            'counts': hist.tolist(),
            'bin_edges': bin_edges.tolist()
        }
    
    def compute_iptw_weights(self) -> Dict[str, Any]:
        """
        Compute inverse probability of treatment weights.
        """
        treatment, outcome, X, covariates = self._prepare_data()
        
        if self.propensity_scores is None:
            self.estimate_propensity_scores()
        
        ps = np.clip(self.propensity_scores, self.config.trim_propensity[0], 
                     self.config.trim_propensity[1])
        
        ate_weights = treatment / ps + (1 - treatment) / (1 - ps)
        
        att_weights = np.where(treatment == 1, 1.0, ps / (1 - ps))
        
        self.weights = ate_weights
        
        return {
            'ate_weights': {
                'mean': float(ate_weights.mean()),
                'std': float(ate_weights.std()),
                'min': float(ate_weights.min()),
                'max': float(ate_weights.max()),
                'effective_sample_size': float(ate_weights.sum() ** 2 / (ate_weights ** 2).sum())
            },
            'att_weights': {
                'mean': float(att_weights.mean()),
                'std': float(att_weights.std()),
                'min': float(att_weights.min()),
                'max': float(att_weights.max())
            }
        }
    
    def propensity_matching(self) -> Dict[str, Any]:
        """
        Perform propensity score matching.
        """
        treatment, outcome, X, covariates = self._prepare_data()
        
        if self.propensity_scores is None:
            self.estimate_propensity_scores()
        
        ps = self.propensity_scores
        
        treated_idx = np.where(treatment == 1)[0]
        control_idx = np.where(treatment == 0)[0]
        
        if len(control_idx) == 0 or len(treated_idx) == 0:
            return {'error': 'Need both treated and control units'}
        
        nn = NearestNeighbors(n_neighbors=self.config.matching_ratio)
        nn.fit(ps[control_idx].reshape(-1, 1))
        
        matched_pairs = []
        used_controls = set()
        
        for t_idx in treated_idx:
            t_ps = ps[t_idx].reshape(1, -1)
            distances, indices = nn.kneighbors(t_ps)
            
            for i, c_local_idx in enumerate(indices[0]):
                c_idx = control_idx[c_local_idx]
                
                if c_idx in used_controls:
                    continue
                
                if self.config.caliper:
                    if distances[0][i] > self.config.caliper:
                        continue
                
                matched_pairs.append({
                    'treated_idx': int(t_idx),
                    'control_idx': int(c_idx),
                    'ps_treated': float(ps[t_idx]),
                    'ps_control': float(ps[c_idx]),
                    'ps_difference': float(abs(ps[t_idx] - ps[c_idx]))
                })
                used_controls.add(c_idx)
                break
        
        matched_treated = [p['treated_idx'] for p in matched_pairs]
        matched_control = [p['control_idx'] for p in matched_pairs]
        all_matched = matched_treated + matched_control
        
        self.matched_data = self.df.iloc[all_matched].copy()
        
        return {
            'n_matched_pairs': len(matched_pairs),
            'n_unmatched_treated': len(treated_idx) - len(matched_pairs),
            'n_unmatched_control': len(control_idx) - len(used_controls),
            'mean_ps_difference': float(np.mean([p['ps_difference'] for p in matched_pairs])) if matched_pairs else None,
            'caliper_used': self.config.caliper,
            'matching_ratio': self.config.matching_ratio
        }
    
    def estimate_ate(self) -> Dict[str, Any]:
        """
        Estimate Average Treatment Effect using configured method.
        """
        treatment, outcome, X, covariates = self._prepare_data()
        
        valid_mask = ~(np.isnan(treatment) | np.isnan(outcome))
        treatment = treatment[valid_mask]
        outcome = outcome[valid_mask]
        
        if self.config.method == 'iptw':
            return self._ate_iptw(treatment, outcome)
        elif self.config.method == 'matching':
            return self._ate_matching()
        elif self.config.method == 'stratification':
            return self._ate_stratification(treatment, outcome)
        else:
            return {'error': f'Unknown method: {self.config.method}'}
    
    def _ate_iptw(self, treatment: np.ndarray, outcome: np.ndarray) -> Dict[str, Any]:
        """Estimate ATE using IPTW"""
        if self.weights is None:
            self.compute_iptw_weights()
        
        weights = self.weights
        
        weights = weights[~(np.isnan(treatment) | np.isnan(outcome))]
        
        weighted_outcome_treated = np.sum(weights * treatment * outcome) / np.sum(weights * treatment)
        weighted_outcome_control = np.sum(weights * (1 - treatment) * outcome) / np.sum(weights * (1 - treatment))
        
        ate = weighted_outcome_treated - weighted_outcome_control
        
        bootstrap_ates = []
        n = len(outcome)
        
        for _ in range(self.config.bootstrap_n):
            idx = np.random.choice(n, n, replace=True)
            t_b = treatment[idx]
            o_b = outcome[idx]
            w_b = weights[idx]
            
            if t_b.sum() > 0 and (1 - t_b).sum() > 0:
                wo_t = np.sum(w_b * t_b * o_b) / np.sum(w_b * t_b)
                wo_c = np.sum(w_b * (1 - t_b) * o_b) / np.sum(w_b * (1 - t_b))
                bootstrap_ates.append(wo_t - wo_c)
        
        bootstrap_ates = np.array(bootstrap_ates)
        se = bootstrap_ates.std()
        
        alpha = 1 - self.config.confidence_level
        ci_lower = np.percentile(bootstrap_ates, alpha / 2 * 100)
        ci_upper = np.percentile(bootstrap_ates, (1 - alpha / 2) * 100)
        
        p_value = 2 * min(
            (bootstrap_ates < 0).mean(),
            (bootstrap_ates > 0).mean()
        )
        
        return {
            'method': 'iptw',
            'ate': float(ate),
            'se': float(se),
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper),
            'p_value': float(p_value),
            'significant': bool(p_value < 0.05),
            'mean_outcome_treated': float(weighted_outcome_treated),
            'mean_outcome_control': float(weighted_outcome_control),
            'n_bootstrap': len(bootstrap_ates)
        }
    
    def _ate_matching(self) -> Dict[str, Any]:
        """Estimate ATE using matched sample"""
        if self.matched_data is None:
            self.propensity_matching()
        
        if self.matched_data is None or len(self.matched_data) == 0:
            return {'error': 'No matched pairs available'}
        
        matched = self.matched_data
        treatment = matched[self.config.treatment_variable].values
        outcome = matched[self.config.outcome_variable].values
        
        mean_treated = outcome[treatment == 1].mean()
        mean_control = outcome[treatment == 0].mean()
        ate = mean_treated - mean_control
        
        n_t = (treatment == 1).sum()
        n_c = (treatment == 0).sum()
        var_t = outcome[treatment == 1].var()
        var_c = outcome[treatment == 0].var()
        
        se = np.sqrt(var_t / n_t + var_c / n_c) if n_t > 0 and n_c > 0 else np.nan
        
        z = stats.norm.ppf((1 + self.config.confidence_level) / 2)
        ci_lower = ate - z * se
        ci_upper = ate + z * se
        
        t_stat = ate / se if se > 0 else np.nan
        p_value = 2 * (1 - stats.norm.cdf(abs(t_stat))) if not np.isnan(t_stat) else np.nan
        
        return {
            'method': 'matching',
            'ate': float(ate),
            'se': float(se) if not np.isnan(se) else None,
            'ci_lower': float(ci_lower) if not np.isnan(ci_lower) else None,
            'ci_upper': float(ci_upper) if not np.isnan(ci_upper) else None,
            'p_value': float(p_value) if not np.isnan(p_value) else None,
            'significant': bool(p_value < 0.05) if not np.isnan(p_value) else None,
            'mean_outcome_treated': float(mean_treated),
            'mean_outcome_control': float(mean_control),
            'n_matched_treated': int(n_t),
            'n_matched_control': int(n_c)
        }
    
    def _ate_stratification(self, treatment: np.ndarray, outcome: np.ndarray) -> Dict[str, Any]:
        """Estimate ATE using propensity score stratification"""
        if self.propensity_scores is None:
            self.estimate_propensity_scores()
        
        ps = self.propensity_scores
        n_strata = 5
        strata = pd.qcut(ps, n_strata, labels=False, duplicates='drop')
        
        stratum_ates = []
        stratum_weights = []
        
        for s in range(n_strata):
            mask = strata == s
            t_s = treatment[mask]
            o_s = outcome[mask]
            
            if t_s.sum() > 0 and (len(t_s) - t_s.sum()) > 0:
                ate_s = o_s[t_s == 1].mean() - o_s[t_s == 0].mean()
                stratum_ates.append(ate_s)
                stratum_weights.append(mask.sum())
        
        weights = np.array(stratum_weights) / sum(stratum_weights)
        ate = sum(a * w for a, w in zip(stratum_ates, weights))
        
        return {
            'method': 'stratification',
            'ate': float(ate),
            'n_strata': n_strata,
            'stratum_ates': [float(a) for a in stratum_ates],
            'stratum_weights': [float(w) for w in weights]
        }
    
    def covariate_balance(self) -> Dict[str, Any]:
        """
        Assess covariate balance before and after adjustment.
        """
        treatment, outcome, X, covariates = self._prepare_data()
        
        balance = {
            'covariates': [],
            'overall_smd_before': 0.0,
            'overall_smd_after': 0.0
        }
        
        for i, cov in enumerate(covariates):
            x = X[:, i]
            
            mean_t = x[treatment == 1].mean()
            mean_c = x[treatment == 0].mean()
            std_pooled = np.sqrt((x[treatment == 1].var() + x[treatment == 0].var()) / 2)
            
            smd_before = (mean_t - mean_c) / std_pooled if std_pooled > 0 else 0
            
            if self.weights is not None:
                weights = self.weights
                wmean_t = np.average(x[treatment == 1], weights=weights[treatment == 1])
                wmean_c = np.average(x[treatment == 0], weights=weights[treatment == 0])
                smd_after = (wmean_t - wmean_c) / std_pooled if std_pooled > 0 else 0
            else:
                smd_after = smd_before
            
            balance['covariates'].append({
                'variable': cov,
                'mean_treated': float(mean_t),
                'mean_control': float(mean_c),
                'smd_before': float(smd_before),
                'smd_after': float(smd_after),
                'balanced': abs(smd_after) < 0.1
            })
        
        balance['overall_smd_before'] = float(np.mean([abs(c['smd_before']) for c in balance['covariates']]))
        balance['overall_smd_after'] = float(np.mean([abs(c['smd_after']) for c in balance['covariates']]))
        
        return balance
    
    def run_analysis(self) -> Dict[str, Any]:
        """Run complete causal analysis"""
        results = {
            'propensity_scores': self.estimate_propensity_scores()
        }
        
        if self.config.method == 'iptw':
            results['weights'] = self.compute_iptw_weights()
        elif self.config.method == 'matching':
            results['matching'] = self.propensity_matching()
        
        results['ate'] = self.estimate_ate()
        results['covariate_balance'] = self.covariate_balance()
        
        return results


def run_causal_analysis(
    df: pd.DataFrame,
    treatment_variable: str,
    outcome_variable: str,
    covariates: Optional[List[str]] = None,
    method: str = 'iptw'
) -> Dict[str, Any]:
    """
    Convenience function to run causal analysis.
    
    Args:
        df: DataFrame with treatment, outcome, and covariates
        treatment_variable: Column name for treatment indicator (0/1)
        outcome_variable: Column name for outcome
        covariates: List of covariate columns for propensity model
        method: 'iptw', 'matching', or 'stratification'
    
    Returns:
        Dictionary with propensity scores, ATE, and balance diagnostics
    """
    config = CausalConfig(
        treatment_variable=treatment_variable,
        outcome_variable=outcome_variable,
        covariates=covariates,
        method=method
    )
    
    analysis = CausalAnalysis(df, config)
    return analysis.run_analysis()
