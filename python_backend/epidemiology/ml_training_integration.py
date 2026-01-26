"""
ML Training Integration for Epidemiology
==========================================
Connects epidemiology data pipelines to ML Training Hub.
Creates training datasets from aggregated signals for:
- Drug safety prediction
- Outbreak prediction
- Vaccine effectiveness modeling
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
import psycopg2
import psycopg2.extras
import uuid

from .privacy import PrivacyGuard, PrivacyConfig, MIN_CELL_SIZE
from .audit import EpidemiologyAuditLogger, AuditAction

logger = logging.getLogger(__name__)


@dataclass
class MLDatasetConfig:
    """Configuration for ML dataset creation"""
    min_sample_size: int = MIN_CELL_SIZE
    include_location_features: bool = True
    include_temporal_features: bool = True
    lookback_days: int = 365
    consent_required: bool = True


class EpidemiologyMLPipeline:
    """
    Creates ML training datasets from epidemiology signals.
    
    All datasets are aggregated and privacy-protected.
    No individual patient data is exposed.
    """
    
    def __init__(self, db_url: Optional[str] = None, config: Optional[MLDatasetConfig] = None):
        self.db_url = db_url or os.environ.get('DATABASE_URL')
        self.config = config or MLDatasetConfig()
        self.privacy_guard = PrivacyGuard(PrivacyConfig(min_cell_size=self.config.min_sample_size))
        self.audit_logger = EpidemiologyAuditLogger(self.db_url)
    
    def create_drug_safety_dataset(
        self,
        name: str,
        description: str,
        drug_codes: Optional[List[str]] = None,
        outcome_codes: Optional[List[str]] = None,
        location_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Create ML training dataset for drug safety prediction.
        
        Features: drug exposure patterns, demographic aggregates, environmental factors
        Target: adverse event signals (flagged/not flagged)
        """
        logger.info(f"Creating drug safety dataset: {name}")
        
        try:
            conn = psycopg2.connect(self.db_url)
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            query = """
                SELECT 
                    s.drug_code,
                    s.drug_name,
                    s.outcome_code,
                    s.outcome_name,
                    s.patient_location_id,
                    l.city,
                    l.state,
                    s.estimate as odds_ratio,
                    s.ci_lower,
                    s.ci_upper,
                    s.p_value,
                    s.signal_strength,
                    s.n_patients,
                    s.n_events,
                    s.flagged,
                    COALESCE(ee.avg_aqi, 0) as avg_environmental_aqi,
                    COALESCE(mf.chronic_rate, 0) as chronic_prescribing_rate
                FROM drug_outcome_signals s
                LEFT JOIN locations l ON s.patient_location_id = l.id
                LEFT JOIN (
                    SELECT location_id, AVG(air_quality_index) as avg_aqi
                    FROM environmental_exposures
                    GROUP BY location_id
                ) ee ON s.patient_location_id = ee.location_id
                LEFT JOIN ml_drug_features mf ON s.drug_code = mf.drug_code 
                    AND s.patient_location_id = mf.location_id
                WHERE s.n_patients >= %s
            """
            params: List[Any] = [self.config.min_sample_size]
            
            if drug_codes:
                query += " AND s.drug_code = ANY(%s)"
                params.append(drug_codes)
            
            if outcome_codes:
                query += " AND s.outcome_code = ANY(%s)"
                params.append(outcome_codes)
            
            if location_ids:
                query += " AND s.patient_location_id = ANY(%s)"
                params.append(location_ids)
            
            cur.execute(query, params)
            rows = [dict(row) for row in cur.fetchall()]
            
            dataset_id = str(uuid.uuid4())
            
            features = []
            for row in rows:
                feature_row = {
                    'drug_code': row['drug_code'],
                    'outcome_code': row['outcome_code'],
                    'location_id': row['patient_location_id'],
                    'odds_ratio': float(row['odds_ratio']) if row['odds_ratio'] else 1.0,
                    'signal_strength': float(row['signal_strength']) if row['signal_strength'] else 0,
                    'n_patients': row['n_patients'],
                    'n_events': row['n_events'],
                    'avg_environmental_aqi': float(row['avg_environmental_aqi']),
                    'chronic_prescribing_rate': float(row['chronic_prescribing_rate']),
                    'target': 1 if row['flagged'] else 0
                }
                features.append(feature_row)
            
            cur.execute("""
                INSERT INTO ml_training_datasets (
                    id, name, description, data_type, 
                    record_count, feature_count, 
                    config, status, created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW())
            """, (
                dataset_id,
                name,
                description,
                'drug_safety',
                len(features),
                len(features[0]) if features else 0,
                json.dumps({
                    'drug_codes': drug_codes,
                    'outcome_codes': outcome_codes,
                    'location_ids': location_ids,
                    'min_sample_size': self.config.min_sample_size
                }),
                'ready'
            ))
            
            conn.commit()
            conn.close()
            
            self.audit_logger.log(
                action=AuditAction.ACCESS_ML_FEATURES,
                user_id='system',
                resource_type='ml_dataset',
                resource_id=dataset_id,
                details={
                    'dataset_type': 'drug_safety',
                    'record_count': len(features),
                    'filters': {
                        'drug_codes': drug_codes,
                        'outcome_codes': outcome_codes
                    }
                }
            )
            
            return {
                'dataset_id': dataset_id,
                'name': name,
                'record_count': len(features),
                'feature_columns': list(features[0].keys()) if features else [],
                'status': 'ready'
            }
            
        except Exception as e:
            logger.error(f"Failed to create drug safety dataset: {e}")
            raise
    
    def create_outbreak_prediction_dataset(
        self,
        name: str,
        description: str,
        pathogen_codes: Optional[List[str]] = None,
        location_ids: Optional[List[str]] = None,
        lookback_days: int = 365
    ) -> Dict[str, Any]:
        """
        Create ML training dataset for outbreak prediction.
        
        Features: historical case counts, R values, environmental factors
        Target: outbreak occurrence in next 7/14/30 days
        """
        logger.info(f"Creating outbreak prediction dataset: {name}")
        
        try:
            conn = psycopg2.connect(self.db_url)
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            cutoff_date = (datetime.utcnow() - timedelta(days=lookback_days)).date()
            
            query = """
                SELECT 
                    mof.pathogen_code,
                    mof.location_id,
                    l.city,
                    l.state,
                    mof.feature_date,
                    mof.case_count,
                    mof.death_count,
                    mof.hospitalization_rate,
                    mof.r_estimate,
                    mof.doubling_time_days,
                    COALESCE(ee.avg_aqi, 0) as avg_aqi,
                    CASE WHEN o.id IS NOT NULL THEN 1 ELSE 0 END as outbreak_occurred
                FROM ml_outbreak_features mof
                LEFT JOIN locations l ON mof.location_id = l.id
                LEFT JOIN (
                    SELECT location_id, AVG(air_quality_index) as avg_aqi
                    FROM environmental_exposures
                    GROUP BY location_id
                ) ee ON mof.location_id = ee.location_id
                LEFT JOIN outbreaks o ON mof.pathogen_code = o.pathogen_code 
                    AND mof.location_id = o.location_id
                    AND o.start_date BETWEEN mof.feature_date AND mof.feature_date + INTERVAL '30 days'
                WHERE mof.feature_date >= %s
            """
            params: List[Any] = [cutoff_date]
            
            if pathogen_codes:
                query += " AND mof.pathogen_code = ANY(%s)"
                params.append(pathogen_codes)
            
            if location_ids:
                query += " AND mof.location_id = ANY(%s)"
                params.append(location_ids)
            
            cur.execute(query, params)
            rows = [dict(row) for row in cur.fetchall()]
            
            dataset_id = str(uuid.uuid4())
            
            features = []
            for row in rows:
                feature_row = {
                    'pathogen_code': row['pathogen_code'],
                    'location_id': row['location_id'],
                    'date': row['feature_date'].isoformat() if row['feature_date'] else None,
                    'case_count': row['case_count'] or 0,
                    'death_count': row['death_count'] or 0,
                    'hospitalization_rate': float(row['hospitalization_rate']) if row['hospitalization_rate'] else 0,
                    'r_estimate': float(row['r_estimate']) if row['r_estimate'] else 1.0,
                    'doubling_time_days': float(row['doubling_time_days']) if row['doubling_time_days'] else 0,
                    'avg_aqi': float(row['avg_aqi']),
                    'target': row['outbreak_occurred']
                }
                features.append(feature_row)
            
            cur.execute("""
                INSERT INTO ml_training_datasets (
                    id, name, description, data_type, 
                    record_count, feature_count, 
                    config, status, created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW())
            """, (
                dataset_id,
                name,
                description,
                'outbreak_prediction',
                len(features),
                len(features[0]) if features else 0,
                json.dumps({
                    'pathogen_codes': pathogen_codes,
                    'location_ids': location_ids,
                    'lookback_days': lookback_days
                }),
                'ready'
            ))
            
            conn.commit()
            conn.close()
            
            return {
                'dataset_id': dataset_id,
                'name': name,
                'record_count': len(features),
                'feature_columns': list(features[0].keys()) if features else [],
                'status': 'ready'
            }
            
        except Exception as e:
            logger.error(f"Failed to create outbreak dataset: {e}")
            raise
    
    def create_vaccine_effectiveness_dataset(
        self,
        name: str,
        description: str,
        vaccine_codes: Optional[List[str]] = None,
        outcome_codes: Optional[List[str]] = None,
        location_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Create ML training dataset for vaccine effectiveness modeling.
        
        Features: coverage rates, demographic aggregates, pathogen characteristics
        Target: effectiveness estimate
        """
        logger.info(f"Creating vaccine effectiveness dataset: {name}")
        
        try:
            conn = psycopg2.connect(self.db_url)
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            query = """
                SELECT 
                    mvf.vaccine_code,
                    mvf.location_id,
                    l.city,
                    l.state,
                    mvf.feature_date,
                    mvf.coverage_rate,
                    mvf.doses_administered,
                    mvf.adverse_event_rate,
                    mvf.effectiveness_estimate
                FROM ml_vaccine_features mvf
                LEFT JOIN locations l ON mvf.location_id = l.id
                WHERE mvf.doses_administered >= %s
            """
            params: List[Any] = [self.config.min_sample_size]
            
            if vaccine_codes:
                query += " AND mvf.vaccine_code = ANY(%s)"
                params.append(vaccine_codes)
            
            if location_ids:
                query += " AND mvf.location_id = ANY(%s)"
                params.append(location_ids)
            
            cur.execute(query, params)
            rows = [dict(row) for row in cur.fetchall()]
            
            dataset_id = str(uuid.uuid4())
            
            features = []
            for row in rows:
                feature_row = {
                    'vaccine_code': row['vaccine_code'],
                    'location_id': row['location_id'],
                    'date': row['feature_date'].isoformat() if row['feature_date'] else None,
                    'coverage_rate': float(row['coverage_rate']) if row['coverage_rate'] else 0,
                    'doses_administered': row['doses_administered'] or 0,
                    'adverse_event_rate': float(row['adverse_event_rate']) if row['adverse_event_rate'] else 0,
                    'target': float(row['effectiveness_estimate']) if row['effectiveness_estimate'] else 0
                }
                features.append(feature_row)
            
            cur.execute("""
                INSERT INTO ml_training_datasets (
                    id, name, description, data_type, 
                    record_count, feature_count, 
                    config, status, created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW())
            """, (
                dataset_id,
                name,
                description,
                'vaccine_effectiveness',
                len(features),
                len(features[0]) if features else 0,
                json.dumps({
                    'vaccine_codes': vaccine_codes,
                    'outcome_codes': outcome_codes,
                    'location_ids': location_ids
                }),
                'ready'
            ))
            
            conn.commit()
            conn.close()
            
            return {
                'dataset_id': dataset_id,
                'name': name,
                'record_count': len(features),
                'feature_columns': list(features[0].keys()) if features else [],
                'status': 'ready'
            }
            
        except Exception as e:
            logger.error(f"Failed to create vaccine effectiveness dataset: {e}")
            raise
    
    def materialize_ml_features(self) -> Dict[str, int]:
        """
        Run feature materialization for all ML pipelines.
        
        Updates ml_drug_features, ml_outbreak_features, ml_vaccine_features tables.
        """
        logger.info("Materializing ML features")
        
        try:
            conn = psycopg2.connect(self.db_url)
            cur = conn.cursor()
            
            today = datetime.utcnow().date()
            
            cur.execute("""
                INSERT INTO ml_drug_features (
                    drug_code, location_id, feature_date,
                    n_patients, n_prescriptions, avg_dose_mg,
                    chronic_rate, adverse_event_rate, feature_vector
                )
                SELECT 
                    de.drug_code,
                    pl.location_id,
                    %s as feature_date,
                    COUNT(DISTINCT de.patient_id) as n_patients,
                    COUNT(*) as n_prescriptions,
                    AVG(de.daily_dose_mg) as avg_dose_mg,
                    AVG(CASE WHEN dp.is_chronic THEN 1.0 ELSE 0.0 END) as chronic_rate,
                    COALESCE(
                        (SELECT COUNT(*) FROM adverse_events ae 
                         WHERE ae.related_drug_code = de.drug_code)::float / 
                        NULLIF(COUNT(*), 0), 0
                    ) as adverse_event_rate,
                    NULL as feature_vector
                FROM drug_exposures de
                JOIN patient_locations pl ON de.patient_id = pl.patient_id
                LEFT JOIN drug_prescriptions dp ON de.drug_code = dp.drug_code
                GROUP BY de.drug_code, pl.location_id
                HAVING COUNT(DISTINCT de.patient_id) >= %s
                ON CONFLICT (drug_code, location_id, feature_date)
                DO UPDATE SET
                    n_patients = EXCLUDED.n_patients,
                    n_prescriptions = EXCLUDED.n_prescriptions,
                    avg_dose_mg = EXCLUDED.avg_dose_mg,
                    chronic_rate = EXCLUDED.chronic_rate,
                    adverse_event_rate = EXCLUDED.adverse_event_rate
            """, (today, self.config.min_sample_size))
            drug_count = cur.rowcount
            
            cur.execute("""
                INSERT INTO ml_outbreak_features (
                    pathogen_code, location_id, feature_date,
                    case_count, death_count, hospitalization_rate,
                    r_estimate, doubling_time_days, feature_vector
                )
                SELECT 
                    ie.pathogen_code,
                    ie.location_id,
                    %s as feature_date,
                    COUNT(*) as case_count,
                    SUM(CASE WHEN ie.outcome = 'death' THEN 1 ELSE 0 END) as death_count,
                    AVG(CASE WHEN ie.hospitalized THEN 1.0 ELSE 0.0 END) as hospitalization_rate,
                    (SELECT r_value FROM reproduction_numbers rn 
                     WHERE rn.pathogen_code = ie.pathogen_code 
                     AND rn.location_id = ie.location_id
                     ORDER BY calculation_date DESC LIMIT 1) as r_estimate,
                    NULL as doubling_time_days,
                    NULL as feature_vector
                FROM infectious_events ie
                WHERE ie.onset_date >= %s
                GROUP BY ie.pathogen_code, ie.location_id
                HAVING COUNT(*) >= %s
                ON CONFLICT (pathogen_code, location_id, feature_date)
                DO UPDATE SET
                    case_count = EXCLUDED.case_count,
                    death_count = EXCLUDED.death_count,
                    hospitalization_rate = EXCLUDED.hospitalization_rate,
                    r_estimate = EXCLUDED.r_estimate
            """, (today, today - timedelta(days=30), self.config.min_sample_size))
            outbreak_count = cur.rowcount
            
            cur.execute("""
                INSERT INTO ml_vaccine_features (
                    vaccine_code, location_id, feature_date,
                    coverage_rate, doses_administered, adverse_event_rate,
                    effectiveness_estimate, feature_vector
                )
                SELECT 
                    i.vaccine_code,
                    pl.location_id,
                    %s as feature_date,
                    COUNT(DISTINCT i.patient_id)::float / 
                        NULLIF((SELECT COUNT(DISTINCT patient_id) FROM patient_locations WHERE location_id = pl.location_id), 0) as coverage_rate,
                    COUNT(*) as doses_administered,
                    COALESCE(
                        (SELECT COUNT(*) FROM vaccine_adverse_events vae 
                         WHERE vae.vaccine_code = i.vaccine_code)::float / 
                        NULLIF(COUNT(*), 0), 0
                    ) as adverse_event_rate,
                    NULL as effectiveness_estimate,
                    NULL as feature_vector
                FROM epi_immunizations i
                JOIN patient_locations pl ON i.patient_id = pl.patient_id
                GROUP BY i.vaccine_code, pl.location_id
                HAVING COUNT(*) >= %s
                ON CONFLICT (vaccine_code, location_id, feature_date)
                DO UPDATE SET
                    coverage_rate = EXCLUDED.coverage_rate,
                    doses_administered = EXCLUDED.doses_administered,
                    adverse_event_rate = EXCLUDED.adverse_event_rate
            """, (today, self.config.min_sample_size))
            vaccine_count = cur.rowcount
            
            conn.commit()
            conn.close()
            
            logger.info(f"Materialized features: drug={drug_count}, outbreak={outbreak_count}, vaccine={vaccine_count}")
            
            return {
                'drug_features': drug_count,
                'outbreak_features': outbreak_count,
                'vaccine_features': vaccine_count
            }
            
        except Exception as e:
            logger.error(f"Failed to materialize features: {e}")
            raise


def run_ml_feature_materialization():
    """Entry point for scheduled ML feature materialization"""
    logger.info("Running scheduled ML feature materialization")
    
    db_url = os.environ.get('DATABASE_URL')
    if not db_url:
        logger.error("DATABASE_URL not set")
        return
    
    pipeline = EpidemiologyMLPipeline(db_url)
    result = pipeline.materialize_ml_features()
    
    return result
