"""
Descriptive Analysis Module for Research Center

Generates baseline characteristics tables with means, SDs, medians, 
proportions, and comparative statistics between groups.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from scipy import stats

@dataclass
class DescriptiveConfig:
    """Configuration for descriptive analysis"""
    stratify_by: Optional[str] = None
    continuous_vars: Optional[List[str]] = None
    categorical_vars: Optional[List[str]] = None
    compare_groups: bool = True
    include_confidence_intervals: bool = True
    confidence_level: float = 0.95

class DescriptiveAnalysis:
    """
    Generates comprehensive descriptive statistics for research cohorts.
    
    Features:
    - Baseline characteristics tables (Table 1)
    - Continuous variable summaries (mean, SD, median, IQR)
    - Categorical variable frequencies and proportions
    - Group comparisons with statistical tests
    - Missing data reporting
    """
    
    def __init__(self, df: pd.DataFrame, config: Optional[DescriptiveConfig] = None):
        self.df = df
        self.config = config or DescriptiveConfig()
        self._identify_variable_types()
    
    def _identify_variable_types(self):
        """Automatically identify continuous vs categorical variables"""
        if self.config.continuous_vars is None:
            self.continuous_vars = []
            for col in self.df.columns:
                if col in ['patient_id', self.config.stratify_by]:
                    continue
                if pd.api.types.is_numeric_dtype(self.df[col]):
                    nunique = self.df[col].nunique()
                    if nunique > 10:
                        self.continuous_vars.append(col)
        else:
            self.continuous_vars = self.config.continuous_vars
            
        if self.config.categorical_vars is None:
            self.categorical_vars = []
            for col in self.df.columns:
                if col in ['patient_id', self.config.stratify_by]:
                    continue
                if not pd.api.types.is_numeric_dtype(self.df[col]):
                    self.categorical_vars.append(col)
                elif pd.api.types.is_numeric_dtype(self.df[col]):
                    nunique = self.df[col].nunique()
                    if nunique <= 10 and col not in self.continuous_vars:
                        self.categorical_vars.append(col)
        else:
            self.categorical_vars = self.config.categorical_vars
    
    def generate_table1(self) -> Dict[str, Any]:
        """
        Generate Table 1 (Baseline Characteristics Table)
        
        Returns comprehensive summary with stratification if configured.
        """
        results = {
            'title': 'Baseline Characteristics',
            'n_total': len(self.df),
            'variables': [],
            'stratified': self.config.stratify_by is not None
        }
        
        if self.config.stratify_by and self.config.stratify_by in self.df.columns:
            groups = self.df[self.config.stratify_by].unique()
            results['groups'] = {
                str(g): int((self.df[self.config.stratify_by] == g).sum()) 
                for g in groups
            }
        else:
            groups = None
        
        for var in self.continuous_vars:
            var_result = self._analyze_continuous(var, groups)
            results['variables'].append(var_result)
        
        for var in self.categorical_vars:
            var_result = self._analyze_categorical(var, groups)
            results['variables'].append(var_result)
        
        return results
    
    def _analyze_continuous(
        self, 
        variable: str, 
        groups: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Analyze a continuous variable"""
        result = {
            'variable': variable,
            'type': 'continuous',
            'n': int(self.df[variable].notna().sum()),
            'missing': int(self.df[variable].isna().sum()),
            'missing_pct': float(self.df[variable].isna().mean() * 100)
        }
        
        data = self.df[variable].dropna()
        
        if len(data) > 0:
            result['overall'] = {
                'mean': float(data.mean()),
                'std': float(data.std()),
                'median': float(data.median()),
                'q1': float(data.quantile(0.25)),
                'q3': float(data.quantile(0.75)),
                'min': float(data.min()),
                'max': float(data.max())
            }
            
            if self.config.include_confidence_intervals:
                ci = stats.t.interval(
                    self.config.confidence_level,
                    len(data) - 1,
                    loc=data.mean(),
                    scale=stats.sem(data)
                )
                result['overall']['ci_lower'] = float(ci[0])
                result['overall']['ci_upper'] = float(ci[1])
        
        if groups is not None and self.config.stratify_by:
            result['by_group'] = {}
            group_data = []
            
            for g in groups:
                mask = self.df[self.config.stratify_by] == g
                gdata = self.df.loc[mask, variable].dropna()
                group_data.append(gdata)
                
                if len(gdata) > 0:
                    result['by_group'][str(g)] = {
                        'n': int(len(gdata)),
                        'mean': float(gdata.mean()),
                        'std': float(gdata.std()),
                        'median': float(gdata.median())
                    }
            
            if self.config.compare_groups and len(groups) == 2:
                if len(group_data[0]) > 0 and len(group_data[1]) > 0:
                    t_stat, t_pval = stats.ttest_ind(group_data[0], group_data[1])
                    u_stat, u_pval = stats.mannwhitneyu(
                        group_data[0], group_data[1], alternative='two-sided'
                    )
                    result['comparison'] = {
                        't_test': {'statistic': float(t_stat), 'p_value': float(t_pval)},
                        'mann_whitney': {'statistic': float(u_stat), 'p_value': float(u_pval)}
                    }
            elif self.config.compare_groups and len(groups) > 2:
                valid_groups = [g for g in group_data if len(g) > 0]
                if len(valid_groups) >= 2:
                    f_stat, f_pval = stats.f_oneway(*valid_groups)
                    h_stat, h_pval = stats.kruskal(*valid_groups)
                    result['comparison'] = {
                        'anova': {'statistic': float(f_stat), 'p_value': float(f_pval)},
                        'kruskal_wallis': {'statistic': float(h_stat), 'p_value': float(h_pval)}
                    }
        
        return result
    
    def _analyze_categorical(
        self, 
        variable: str, 
        groups: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Analyze a categorical variable"""
        result = {
            'variable': variable,
            'type': 'categorical',
            'n': int(self.df[variable].notna().sum()),
            'missing': int(self.df[variable].isna().sum()),
            'missing_pct': float(self.df[variable].isna().mean() * 100)
        }
        
        data = self.df[variable].dropna()
        value_counts = data.value_counts()
        
        result['overall'] = {
            'categories': {}
        }
        
        for cat, count in value_counts.items():
            result['overall']['categories'][str(cat)] = {
                'n': int(count),
                'pct': float(count / len(data) * 100)
            }
        
        if groups is not None and self.config.stratify_by:
            result['by_group'] = {}
            contingency_data = []
            
            for g in groups:
                mask = self.df[self.config.stratify_by] == g
                gdata = self.df.loc[mask, variable].dropna()
                gvalue_counts = gdata.value_counts()
                
                result['by_group'][str(g)] = {
                    'categories': {}
                }
                
                for cat in value_counts.index:
                    count = int(gvalue_counts.get(cat, 0))
                    result['by_group'][str(g)]['categories'][str(cat)] = {
                        'n': count,
                        'pct': float(count / len(gdata) * 100) if len(gdata) > 0 else 0
                    }
                
                contingency_data.append([gvalue_counts.get(cat, 0) for cat in value_counts.index])
            
            if self.config.compare_groups:
                try:
                    contingency = np.array(contingency_data)
                    chi2, chi2_pval, dof, expected = stats.chi2_contingency(contingency)
                    result['comparison'] = {
                        'chi_square': {
                            'statistic': float(chi2),
                            'p_value': float(chi2_pval),
                            'dof': int(dof)
                        }
                    }
                    
                    if len(groups) == 2 and len(value_counts) == 2:
                        try:
                            odds_ratio_res = stats.fisher_exact(contingency)
                            result['comparison']['fisher_exact'] = {
                                'odds_ratio': float(odds_ratio_res[0]),
                                'p_value': float(odds_ratio_res[1])
                            }
                        except:
                            pass
                except:
                    pass
        
        return result
    
    def get_missing_data_report(self) -> Dict[str, Any]:
        """Generate missing data report"""
        report = {
            'total_records': len(self.df),
            'complete_cases': int(self.df.dropna().shape[0]),
            'complete_case_pct': float(self.df.dropna().shape[0] / len(self.df) * 100),
            'variables': []
        }
        
        for col in self.df.columns:
            if col == 'patient_id':
                continue
            
            missing = self.df[col].isna().sum()
            report['variables'].append({
                'variable': col,
                'missing_n': int(missing),
                'missing_pct': float(missing / len(self.df) * 100),
                'available_n': int(len(self.df) - missing)
            })
        
        report['variables'].sort(key=lambda x: x['missing_pct'], reverse=True)
        
        return report
    
    def get_correlation_matrix(
        self, 
        variables: Optional[List[str]] = None,
        method: str = 'pearson'
    ) -> Dict[str, Any]:
        """Compute correlation matrix for continuous variables"""
        if variables is None:
            variables = self.continuous_vars
        
        valid_vars = [v for v in variables if v in self.df.columns]
        
        if len(valid_vars) < 2:
            return {'error': 'Need at least 2 variables for correlation'}
        
        corr_df = self.df[valid_vars].corr(method=method)
        
        return {
            'method': method,
            'variables': valid_vars,
            'matrix': corr_df.to_dict(),
            'n_pairs': len(valid_vars)
        }
    
    def format_table1_text(self, table1: Dict[str, Any]) -> str:
        """Format Table 1 as text for reports"""
        lines = []
        lines.append(f"Table 1: {table1['title']}")
        lines.append(f"Total N = {table1['n_total']}")
        lines.append("")
        
        if table1.get('groups'):
            group_headers = " | ".join([f"{k}: n={v}" for k, v in table1['groups'].items()])
            lines.append(f"Groups: {group_headers}")
            lines.append("")
        
        lines.append("-" * 80)
        
        for var in table1['variables']:
            if var['type'] == 'continuous':
                overall = var.get('overall', {})
                mean_sd = f"{overall.get('mean', 'NA'):.2f} ({overall.get('std', 'NA'):.2f})" if overall else "NA"
                median_iqr = f"{overall.get('median', 'NA'):.2f} [{overall.get('q1', 'NA'):.2f}-{overall.get('q3', 'NA'):.2f}]" if overall else "NA"
                lines.append(f"{var['variable']}: Mean (SD) = {mean_sd}, Median [IQR] = {median_iqr}")
            else:
                lines.append(f"{var['variable']}:")
                for cat, info in var.get('overall', {}).get('categories', {}).items():
                    lines.append(f"  {cat}: {info['n']} ({info['pct']:.1f}%)")
        
        return "\n".join(lines)


def run_descriptive_analysis(
    df: pd.DataFrame,
    stratify_by: Optional[str] = None,
    continuous_vars: Optional[List[str]] = None,
    categorical_vars: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Convenience function to run descriptive analysis.
    
    Returns Table 1 and missing data report.
    """
    config = DescriptiveConfig(
        stratify_by=stratify_by,
        continuous_vars=continuous_vars,
        categorical_vars=categorical_vars
    )
    
    analysis = DescriptiveAnalysis(df, config)
    
    return {
        'table1': analysis.generate_table1(),
        'missing_data': analysis.get_missing_data_report(),
        'correlation': analysis.get_correlation_matrix()
    }
