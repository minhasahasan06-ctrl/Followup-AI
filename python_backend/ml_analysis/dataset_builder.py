"""
Dataset Builder for Research Center ML Analysis Engine

Transforms cohort definitions and analysis specifications into 
pandas DataFrames, joining clinical, followup, immune, and environmental data.
"""

import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass
import psycopg2
from psycopg2.extras import RealDictCursor

@dataclass
class CohortSpec:
    """Specification for cohort selection"""
    cohort_id: Optional[str] = None
    patient_ids: Optional[List[str]] = None
    age_min: Optional[int] = None
    age_max: Optional[int] = None
    sex: Optional[str] = None
    conditions: Optional[List[str]] = None
    exclude_conditions: Optional[List[str]] = None
    min_followup_days: Optional[int] = None
    enrollment_date_start: Optional[str] = None
    enrollment_date_end: Optional[str] = None
    risk_score_min: Optional[float] = None
    risk_score_max: Optional[float] = None

@dataclass
class AnalysisSpec:
    """Specification for analysis variables"""
    outcome_variable: str
    outcome_type: str  # 'binary', 'continuous', 'time_to_event'
    exposure_variable: Optional[str] = None
    covariates: Optional[List[str]] = None
    time_variable: Optional[str] = None
    event_variable: Optional[str] = None
    stratify_by: Optional[List[str]] = None
    follow_up_window_days: Optional[int] = None

class DatasetBuilder:
    """
    Builds analysis-ready datasets from cohort specifications.
    
    Joins data from:
    - Patient demographics (patient_profiles)
    - Daily followups (daily_followups)
    - Immune markers (immune_markers)
    - Environmental exposures (environmental_exposures)
    - Health alerts (interaction_alerts)
    - Medications (patient_medications)
    - Conditions (conditions)
    - Deterioration indices
    """
    
    def __init__(self, connection_string: Optional[str] = None):
        self.connection_string = connection_string or os.environ.get('DATABASE_URL')
        
    def get_connection(self):
        """Create database connection"""
        return psycopg2.connect(self.connection_string)
    
    def build_cohort_dataset(
        self,
        cohort_spec: CohortSpec,
        analysis_spec: AnalysisSpec,
        include_data_types: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Build a complete dataset for analysis from cohort and analysis specs.
        
        Args:
            cohort_spec: Cohort selection criteria
            analysis_spec: Variables to include
            include_data_types: Data sources to join ('demographics', 'followups', 
                              'immune', 'environmental', 'medications', 'conditions')
        
        Returns:
            pd.DataFrame with analysis-ready data
        """
        if include_data_types is None:
            include_data_types = ['demographics', 'followups', 'immune', 
                                  'environmental', 'medications', 'conditions']
        
        patient_ids = self._get_cohort_patients(cohort_spec)
        
        if not patient_ids:
            return pd.DataFrame()
        
        df = pd.DataFrame({'patient_id': patient_ids})
        
        if 'demographics' in include_data_types:
            df = self._join_demographics(df, patient_ids)
            
        if 'followups' in include_data_types:
            df = self._join_followups(df, patient_ids, analysis_spec)
            
        if 'immune' in include_data_types:
            df = self._join_immune_markers(df, patient_ids)
            
        if 'environmental' in include_data_types:
            df = self._join_environmental(df, patient_ids)
            
        if 'medications' in include_data_types:
            df = self._join_medications(df, patient_ids)
            
        if 'conditions' in include_data_types:
            df = self._join_conditions(df, patient_ids)
        
        if analysis_spec.outcome_variable:
            df = self._compute_outcome(df, analysis_spec)
        
        return df
    
    def _get_cohort_patients(self, spec: CohortSpec) -> List[str]:
        """Get patient IDs matching cohort criteria"""
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                if spec.cohort_id:
                    cur.execute(
                        "SELECT patient_ids FROM cohorts WHERE id = %s",
                        (spec.cohort_id,)
                    )
                    row = cur.fetchone()
                    if row and row['patient_ids']:
                        return row['patient_ids']
                    return []
                
                if spec.patient_ids:
                    return spec.patient_ids
                
                query = """
                    SELECT DISTINCT pp.user_id as patient_id
                    FROM patient_profiles pp
                    WHERE 1=1
                """
                params = []
                
                if spec.age_min:
                    query += " AND EXTRACT(YEAR FROM AGE(pp.date_of_birth)) >= %s"
                    params.append(spec.age_min)
                    
                if spec.age_max:
                    query += " AND EXTRACT(YEAR FROM AGE(pp.date_of_birth)) <= %s"
                    params.append(spec.age_max)
                    
                if spec.sex:
                    query += " AND pp.sex = %s"
                    params.append(spec.sex)
                
                if spec.conditions:
                    query += """
                        AND EXISTS (
                            SELECT 1 FROM conditions c
                            WHERE c.patient_id = pp.user_id
                            AND c.name = ANY(%s)
                        )
                    """
                    params.append(spec.conditions)
                
                if spec.exclude_conditions:
                    query += """
                        AND NOT EXISTS (
                            SELECT 1 FROM conditions c
                            WHERE c.patient_id = pp.user_id
                            AND c.name = ANY(%s)
                        )
                    """
                    params.append(spec.exclude_conditions)
                
                cur.execute(query, params)
                return [row['patient_id'] for row in cur.fetchall()]
    
    def _join_demographics(self, df: pd.DataFrame, patient_ids: List[str]) -> pd.DataFrame:
        """Join patient demographics data"""
        with self.get_connection() as conn:
            query = """
                SELECT 
                    user_id as patient_id,
                    date_of_birth,
                    EXTRACT(YEAR FROM AGE(date_of_birth)) as age,
                    sex,
                    blood_type,
                    height_cm,
                    weight_kg
                FROM patient_profiles
                WHERE user_id = ANY(%s)
            """
            demo_df = pd.read_sql(query, conn, params=(patient_ids,))
            
        if not demo_df.empty:
            demo_df['bmi'] = demo_df['weight_kg'] / ((demo_df['height_cm'] / 100) ** 2)
            
        return df.merge(demo_df, on='patient_id', how='left')
    
    def _join_followups(
        self, 
        df: pd.DataFrame, 
        patient_ids: List[str],
        analysis_spec: AnalysisSpec
    ) -> pd.DataFrame:
        """Join and aggregate daily followup data"""
        with self.get_connection() as conn:
            query = """
                SELECT 
                    patient_id,
                    date,
                    energy_level,
                    pain_level,
                    sleep_quality,
                    mood_score,
                    symptoms,
                    notes
                FROM daily_followups
                WHERE patient_id = ANY(%s)
                ORDER BY patient_id, date
            """
            followups_df = pd.read_sql(query, conn, params=(patient_ids,))
        
        if followups_df.empty:
            return df
        
        agg_funcs = {
            'energy_level': ['mean', 'std', 'min', 'max'],
            'pain_level': ['mean', 'std', 'min', 'max'],
            'sleep_quality': ['mean', 'std'],
            'mood_score': ['mean', 'std'],
        }
        
        agg_df = followups_df.groupby('patient_id').agg(agg_funcs)
        agg_df.columns = ['_'.join(col).strip() for col in agg_df.columns.values]
        agg_df = agg_df.reset_index()
        
        count_df = followups_df.groupby('patient_id').agg(
            followup_count=('date', 'count'),
            followup_days=('date', lambda x: (x.max() - x.min()).days if len(x) > 1 else 0),
            first_followup=('date', 'min'),
            last_followup=('date', 'max')
        ).reset_index()
        
        followups_agg = agg_df.merge(count_df, on='patient_id', how='left')
        
        return df.merge(followups_agg, on='patient_id', how='left')
    
    def _join_immune_markers(self, df: pd.DataFrame, patient_ids: List[str]) -> pd.DataFrame:
        """Join immune marker data with pivoting for each marker type"""
        with self.get_connection() as conn:
            query = """
                SELECT 
                    patient_id,
                    marker_type,
                    value,
                    unit,
                    reference_range_low,
                    reference_range_high,
                    measured_at
                FROM immune_markers
                WHERE patient_id = ANY(%s)
                ORDER BY patient_id, measured_at DESC
            """
            immune_df = pd.read_sql(query, conn, params=(patient_ids,))
        
        if immune_df.empty:
            return df
        
        latest = immune_df.groupby(['patient_id', 'marker_type']).first().reset_index()
        
        pivot_df = latest.pivot(
            index='patient_id',
            columns='marker_type',
            values='value'
        ).reset_index()
        
        pivot_df.columns = [f'immune_{col}' if col != 'patient_id' else col 
                           for col in pivot_df.columns]
        
        return df.merge(pivot_df, on='patient_id', how='left')
    
    def _join_environmental(self, df: pd.DataFrame, patient_ids: List[str]) -> pd.DataFrame:
        """Join environmental exposure data"""
        with self.get_connection() as conn:
            query = """
                SELECT 
                    patient_id,
                    exposure_type,
                    value,
                    unit,
                    measured_at
                FROM environmental_exposures
                WHERE patient_id = ANY(%s)
                AND measured_at >= NOW() - INTERVAL '30 days'
            """
            env_df = pd.read_sql(query, conn, params=(patient_ids,))
        
        if env_df.empty:
            return df
        
        agg_df = env_df.groupby(['patient_id', 'exposure_type']).agg({
            'value': ['mean', 'max']
        }).reset_index()
        agg_df.columns = ['patient_id', 'exposure_type', 'value_mean', 'value_max']
        
        pivot_df = agg_df.pivot(
            index='patient_id',
            columns='exposure_type',
            values=['value_mean', 'value_max']
        )
        pivot_df.columns = [f'env_{col[1]}_{col[0]}' for col in pivot_df.columns]
        pivot_df = pivot_df.reset_index()
        
        return df.merge(pivot_df, on='patient_id', how='left')
    
    def _join_medications(self, df: pd.DataFrame, patient_ids: List[str]) -> pd.DataFrame:
        """Join medication data with counts by category"""
        with self.get_connection() as conn:
            query = """
                SELECT 
                    patient_id,
                    medication_name,
                    status,
                    start_date,
                    end_date
                FROM patient_medications
                WHERE patient_id = ANY(%s)
            """
            meds_df = pd.read_sql(query, conn, params=(patient_ids,))
        
        if meds_df.empty:
            return df
        
        agg_df = meds_df.groupby('patient_id').agg(
            medication_count=('medication_name', 'count'),
            active_medication_count=('status', lambda x: (x == 'active').sum())
        ).reset_index()
        
        return df.merge(agg_df, on='patient_id', how='left')
    
    def _join_conditions(self, df: pd.DataFrame, patient_ids: List[str]) -> pd.DataFrame:
        """Join conditions with flags for common conditions"""
        with self.get_connection() as conn:
            query = """
                SELECT 
                    patient_id,
                    name,
                    status,
                    diagnosed_at
                FROM conditions
                WHERE patient_id = ANY(%s)
            """
            cond_df = pd.read_sql(query, conn, params=(patient_ids,))
        
        if cond_df.empty:
            return df
        
        agg_df = cond_df.groupby('patient_id').agg(
            condition_count=('name', 'count'),
            active_condition_count=('status', lambda x: (x == 'active').sum())
        ).reset_index()
        
        common_conditions = ['diabetes', 'hypertension', 'asthma', 'copd', 
                           'heart_failure', 'kidney_disease', 'liver_disease']
        for cond in common_conditions:
            cond_df[f'has_{cond}'] = cond_df['name'].str.lower().str.contains(cond).astype(int)
        
        cond_flags = cond_df.groupby('patient_id')[
            [f'has_{c}' for c in common_conditions]
        ].max().reset_index()
        
        df = df.merge(agg_df, on='patient_id', how='left')
        df = df.merge(cond_flags, on='patient_id', how='left')
        
        return df
    
    def _compute_outcome(self, df: pd.DataFrame, spec: AnalysisSpec) -> pd.DataFrame:
        """Compute outcome variable based on analysis specification"""
        
        if spec.outcome_type == 'binary':
            if spec.outcome_variable == 'deterioration':
                df['outcome'] = (df.get('pain_level_max', 0) > 7).astype(int)
            elif spec.outcome_variable == 'hospitalization':
                df['outcome'] = 0
            elif spec.outcome_variable == 'high_risk':
                df['outcome'] = (df.get('pain_level_mean', 0) > 5).astype(int)
            else:
                df['outcome'] = df.get(spec.outcome_variable, 0)
                
        elif spec.outcome_type == 'continuous':
            df['outcome'] = df.get(spec.outcome_variable, np.nan)
            
        elif spec.outcome_type == 'time_to_event':
            if spec.time_variable and spec.event_variable:
                df['time'] = df.get(spec.time_variable, np.nan)
                df['event'] = df.get(spec.event_variable, 0)
        
        return df
    
    def get_variable_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get summary statistics for all variables in dataset"""
        summary = {
            'n_patients': len(df),
            'n_variables': len(df.columns),
            'variables': {}
        }
        
        for col in df.columns:
            if col == 'patient_id':
                continue
                
            col_summary = {
                'dtype': str(df[col].dtype),
                'missing_count': int(df[col].isna().sum()),
                'missing_pct': float(df[col].isna().mean() * 100)
            }
            
            if pd.api.types.is_numeric_dtype(df[col]):
                col_summary.update({
                    'mean': float(df[col].mean()) if not df[col].isna().all() else None,
                    'std': float(df[col].std()) if not df[col].isna().all() else None,
                    'min': float(df[col].min()) if not df[col].isna().all() else None,
                    'max': float(df[col].max()) if not df[col].isna().all() else None,
                    'median': float(df[col].median()) if not df[col].isna().all() else None
                })
            else:
                value_counts = df[col].value_counts().head(10).to_dict()
                col_summary['value_counts'] = {str(k): int(v) for k, v in value_counts.items()}
            
            summary['variables'][col] = col_summary
        
        return summary
    
    def export_dataset(
        self, 
        df: pd.DataFrame, 
        output_path: str,
        format: str = 'csv',
        include_summary: bool = True
    ) -> Dict[str, str]:
        """Export dataset to file with optional summary"""
        paths = {}
        
        if format == 'csv':
            df.to_csv(output_path, index=False)
            paths['data'] = output_path
        elif format == 'parquet':
            df.to_parquet(output_path, index=False)
            paths['data'] = output_path
        elif format == 'json':
            df.to_json(output_path, orient='records', date_format='iso')
            paths['data'] = output_path
        
        if include_summary:
            summary = self.get_variable_summary(df)
            summary_path = output_path.rsplit('.', 1)[0] + '_summary.json'
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            paths['summary'] = summary_path
        
        return paths


def build_analysis_dataset(
    cohort_spec_dict: Dict,
    analysis_spec_dict: Dict,
    include_data_types: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, Dict]:
    """
    Convenience function to build dataset from dictionary specs.
    
    Returns tuple of (dataframe, summary)
    """
    builder = DatasetBuilder()
    
    cohort_spec = CohortSpec(**cohort_spec_dict)
    analysis_spec = AnalysisSpec(**analysis_spec_dict)
    
    df = builder.build_cohort_dataset(cohort_spec, analysis_spec, include_data_types)
    summary = builder.get_variable_summary(df)
    
    return df, summary
