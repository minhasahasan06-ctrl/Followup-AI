"""
Drug Safety Signal Scanner
===========================
Background job that continuously scans for drug-outcome associations
across patient locations, computing aggregated signals and summaries.
"""

import os
import logging
import json
import math
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
import psycopg2
import psycopg2.extras
import numpy as np
from scipy import stats

from .privacy import PrivacyGuard, PrivacyConfig, MIN_CELL_SIZE
from .audit import EpidemiologyAuditLogger

logger = logging.getLogger(__name__)


@dataclass
class ScanConfig:
    """Configuration for drug safety scanning"""
    min_sample_size: int = MIN_CELL_SIZE
    signal_threshold_pvalue: float = 0.05
    signal_threshold_effect: float = 1.5
    run_causal_analysis: bool = True
    multiple_testing_correction: str = "bonferroni"
    max_drugs_per_run: int = 100
    max_outcomes_per_run: int = 50


class DrugSafetyScanner:
    """
    Scans for drug-outcome safety signals across patient locations.
    
    For each (drug, outcome, location) combination:
    1. Builds exposed vs unexposed cohorts
    2. Computes descriptive summaries
    3. Runs statistical models (logistic, survival)
    4. Generates aggregated signals and alerts
    """
    
    def __init__(self, db_url: Optional[str] = None, config: Optional[ScanConfig] = None):
        self.db_url = db_url or os.environ.get('DATABASE_URL')
        self.config = config or ScanConfig()
        self.privacy_guard = PrivacyGuard(PrivacyConfig(min_cell_size=self.config.min_sample_size))
        self.audit_logger = EpidemiologyAuditLogger(self.db_url)
    
    def run_scan(
        self,
        drug_codes: Optional[List[str]] = None,
        outcome_codes: Optional[List[str]] = None,
        location_ids: Optional[List[str]] = None,
        scope: str = "all"
    ) -> Dict[str, Any]:
        """
        Run the drug safety scan.
        
        Args:
            drug_codes: Specific drugs to scan (None = all)
            outcome_codes: Specific outcomes to scan (None = all)
            location_ids: Specific locations to scan (None = all)
            scope: Analysis scope
            
        Returns:
            Scan results summary
        """
        start_time = datetime.utcnow()
        logger.info(f"Starting drug safety scan at {start_time}")
        
        try:
            conn = psycopg2.connect(self.db_url)
            
            drugs = self._get_drugs(conn, drug_codes)
            outcomes = self._get_outcomes(conn, outcome_codes)
            locations = self._get_locations(conn, location_ids)
            
            signals_generated = 0
            summaries_generated = 0
            alerts_created = 0
            
            for drug in drugs[:self.config.max_drugs_per_run]:
                for outcome in outcomes[:self.config.max_outcomes_per_run]:
                    for location in locations:
                        try:
                            result = self._analyze_drug_outcome_location(
                                conn, drug, outcome, location, scope
                            )
                            
                            if result:
                                if result.get('signal_saved'):
                                    signals_generated += 1
                                if result.get('summary_saved'):
                                    summaries_generated += 1
                                if result.get('alert_created'):
                                    alerts_created += 1
                                    
                        except Exception as e:
                            logger.error(f"Error analyzing {drug['code']}/{outcome['code']}/{location['id']}: {e}")
            
            for drug in drugs[:self.config.max_drugs_per_run]:
                for outcome in outcomes[:self.config.max_outcomes_per_run]:
                    try:
                        result = self._analyze_drug_outcome_location(
                            conn, drug, outcome, None, scope
                        )
                        if result:
                            if result.get('signal_saved'):
                                signals_generated += 1
                            if result.get('summary_saved'):
                                summaries_generated += 1
                    except Exception as e:
                        logger.error(f"Error analyzing {drug['code']}/{outcome['code']}/global: {e}")
            
            conn.close()
            
            end_time = datetime.utcnow()
            duration = (end_time - start_time).total_seconds()
            
            result = {
                'status': 'completed',
                'started_at': start_time.isoformat(),
                'completed_at': end_time.isoformat(),
                'duration_seconds': duration,
                'drugs_scanned': len(drugs),
                'outcomes_scanned': len(outcomes),
                'locations_scanned': len(locations),
                'signals_generated': signals_generated,
                'summaries_generated': summaries_generated,
                'alerts_created': alerts_created
            }
            
            logger.info(f"Drug safety scan completed: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Drug safety scan failed: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'started_at': start_time.isoformat()
            }
    
    def _get_drugs(self, conn, drug_codes: Optional[List[str]] = None) -> List[Dict]:
        """Get list of drugs to analyze"""
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            if drug_codes:
                cur.execute("""
                    SELECT DISTINCT drug_code as code, drug_name as name
                    FROM drug_exposures
                    WHERE drug_code = ANY(%s)
                """, (drug_codes,))
            else:
                cur.execute("""
                    SELECT DISTINCT drug_code as code, 
                           COALESCE(MAX(drug_name), drug_code) as name
                    FROM drug_exposures de
                    LEFT JOIN drug_prescriptions dp ON de.drug_code = dp.drug_code
                    GROUP BY de.drug_code
                    ORDER BY COUNT(*) DESC
                    LIMIT %s
                """, (self.config.max_drugs_per_run,))
            
            return [dict(row) for row in cur.fetchall()]
    
    def _get_outcomes(self, conn, outcome_codes: Optional[List[str]] = None) -> List[Dict]:
        """Get list of outcomes to analyze"""
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            if outcome_codes:
                cur.execute("""
                    SELECT DISTINCT event_code as code, event_name as name
                    FROM adverse_events
                    WHERE event_code = ANY(%s)
                """, (outcome_codes,))
            else:
                cur.execute("""
                    SELECT DISTINCT event_code as code, event_name as name
                    FROM adverse_events
                    ORDER BY COUNT(*) OVER (PARTITION BY event_code) DESC
                    LIMIT %s
                """, (self.config.max_outcomes_per_run,))
            
            return [dict(row) for row in cur.fetchall()]
    
    def _get_locations(self, conn, location_ids: Optional[List[str]] = None) -> List[Dict]:
        """Get list of locations to analyze"""
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            if location_ids:
                cur.execute("""
                    SELECT id, name FROM locations WHERE id = ANY(%s)
                """, (location_ids,))
            else:
                cur.execute("""
                    SELECT DISTINCT l.id, l.name
                    FROM locations l
                    JOIN patient_locations pl ON pl.location_id = l.id
                    LIMIT 50
                """)
            
            return [dict(row) for row in cur.fetchall()]
    
    def _analyze_drug_outcome_location(
        self,
        conn,
        drug: Dict,
        outcome: Dict,
        location: Optional[Dict],
        scope: str
    ) -> Optional[Dict]:
        """Analyze a specific drug-outcome-location combination"""
        
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            location_filter = ""
            params = [drug['code'], outcome['code']]
            
            if location:
                location_filter = "AND pl.location_id = %s"
                params.append(location['id'])
            
            cur.execute(f"""
                WITH exposed_patients AS (
                    SELECT DISTINCT de.patient_id
                    FROM drug_exposures de
                    JOIN patient_locations pl ON de.patient_id = pl.patient_id
                    WHERE de.drug_code = %s
                    {location_filter}
                ),
                outcome_patients AS (
                    SELECT DISTINCT ae.patient_id
                    FROM adverse_events ae
                    WHERE ae.event_code = %s
                ),
                all_patients AS (
                    SELECT DISTINCT pl.patient_id
                    FROM patient_locations pl
                    {'WHERE pl.location_id = %s' if location else ''}
                )
                SELECT
                    (SELECT COUNT(*) FROM all_patients) as n_total,
                    (SELECT COUNT(*) FROM exposed_patients) as n_exposed,
                    (SELECT COUNT(*) FROM outcome_patients op 
                     WHERE op.patient_id IN (SELECT patient_id FROM all_patients)) as n_events,
                    (SELECT COUNT(*) FROM exposed_patients ep 
                     WHERE ep.patient_id IN (SELECT patient_id FROM outcome_patients)) as n_exposed_events
            """, params if not location else params + [location['id']])
            
            counts = cur.fetchone()
            
            if not counts:
                return None
            
            n_total = counts['n_total'] or 0
            n_exposed = counts['n_exposed'] or 0
            n_events = counts['n_events'] or 0
            n_exposed_events = counts['n_exposed_events'] or 0
            n_unexposed = n_total - n_exposed
            n_unexposed_events = n_events - n_exposed_events
            
            if n_total < self.config.min_sample_size:
                return None
            
            incidence_exposed = n_exposed_events / n_exposed if n_exposed > 0 else 0
            incidence_unexposed = n_unexposed_events / n_unexposed if n_unexposed > 0 else 0
            
            summary_saved = self._save_summary(
                conn, drug, outcome, location, scope,
                n_total, n_events, n_exposed, n_exposed_events,
                n_unexposed, n_unexposed_events,
                incidence_exposed, incidence_unexposed
            )
            
            signal_saved = False
            alert_created = False
            
            if n_exposed >= self.config.min_sample_size and n_unexposed >= self.config.min_sample_size:
                estimate, ci_lower, ci_upper, p_value = self._compute_odds_ratio(
                    n_exposed_events, n_exposed - n_exposed_events,
                    n_unexposed_events, n_unexposed - n_unexposed_events
                )
                
                signal_strength = self._compute_signal_strength(estimate, p_value, n_events)
                
                flagged = (
                    p_value < self.config.signal_threshold_pvalue and
                    estimate > self.config.signal_threshold_effect
                )
                
                signal_saved = self._save_signal(
                    conn, drug, outcome, location, scope,
                    'logistic', 'OR', estimate, ci_lower, ci_upper, p_value,
                    signal_strength, n_total, n_events, flagged
                )
                
                if flagged:
                    alert_created = self._create_alert(
                        conn, drug, outcome, location,
                        estimate, ci_lower, ci_upper, p_value, n_total, n_events
                    )
            
            return {
                'signal_saved': signal_saved,
                'summary_saved': summary_saved,
                'alert_created': alert_created
            }
    
    def _compute_odds_ratio(
        self,
        a: int,
        b: int,
        c: int,
        d: int
    ) -> Tuple[float, float, float, float]:
        """Compute odds ratio with confidence interval"""
        a, b, c, d = max(a, 0.5), max(b, 0.5), max(c, 0.5), max(d, 0.5)
        
        odds_ratio = (a * d) / (b * c)
        
        log_or = math.log(odds_ratio)
        se_log_or = math.sqrt(1/a + 1/b + 1/c + 1/d)
        
        z = 1.96
        ci_lower = math.exp(log_or - z * se_log_or)
        ci_upper = math.exp(log_or + z * se_log_or)
        
        z_score = log_or / se_log_or
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        
        return round(odds_ratio, 4), round(ci_lower, 4), round(ci_upper, 4), round(p_value, 6)
    
    def _compute_signal_strength(
        self,
        estimate: float,
        p_value: float,
        n_events: int
    ) -> float:
        """Compute signal strength score (0-100)"""
        effect_component = min(estimate / 5.0, 1.0) * 40
        
        pvalue_component = (1 - p_value) * 40
        
        sample_component = min(n_events / 100, 1.0) * 20
        
        return round(effect_component + pvalue_component + sample_component, 2)
    
    def _save_summary(
        self,
        conn,
        drug: Dict,
        outcome: Dict,
        location: Optional[Dict],
        scope: str,
        n_patients: int,
        n_events: int,
        n_exposed: int,
        n_exposed_events: int,
        n_unexposed: int,
        n_unexposed_events: int,
        incidence_exposed: float,
        incidence_unexposed: float
    ) -> bool:
        """Save drug outcome summary to database"""
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO drug_outcome_summaries (
                        drug_code, drug_name, outcome_code, outcome_name,
                        patient_location_id, analysis_scope,
                        n_patients, n_events, n_exposed, n_exposed_events,
                        n_unexposed, n_unexposed_events,
                        incidence_exposed, incidence_unexposed, updated_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
                    ON CONFLICT (drug_code, outcome_code, patient_location_id, analysis_scope)
                    DO UPDATE SET
                        drug_name = EXCLUDED.drug_name,
                        outcome_name = EXCLUDED.outcome_name,
                        n_patients = EXCLUDED.n_patients,
                        n_events = EXCLUDED.n_events,
                        n_exposed = EXCLUDED.n_exposed,
                        n_exposed_events = EXCLUDED.n_exposed_events,
                        n_unexposed = EXCLUDED.n_unexposed,
                        n_unexposed_events = EXCLUDED.n_unexposed_events,
                        incidence_exposed = EXCLUDED.incidence_exposed,
                        incidence_unexposed = EXCLUDED.incidence_unexposed,
                        updated_at = NOW()
                """, (
                    drug['code'], drug['name'], outcome['code'], outcome['name'],
                    location['id'] if location else None, scope,
                    n_patients, n_events, n_exposed, n_exposed_events,
                    n_unexposed, n_unexposed_events,
                    round(incidence_exposed, 6), round(incidence_unexposed, 6)
                ))
            conn.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to save summary: {e}")
            conn.rollback()
            return False
    
    def _save_signal(
        self,
        conn,
        drug: Dict,
        outcome: Dict,
        location: Optional[Dict],
        scope: str,
        model_type: str,
        effect_measure: str,
        estimate: float,
        ci_lower: float,
        ci_upper: float,
        p_value: float,
        signal_strength: float,
        n_patients: int,
        n_events: int,
        flagged: bool
    ) -> bool:
        """Save drug outcome signal to database"""
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO drug_outcome_signals (
                        drug_code, drug_name, outcome_code, outcome_name,
                        patient_location_id, analysis_scope, model_type, effect_measure,
                        estimate, ci_lower, ci_upper, p_value, signal_strength,
                        n_patients, n_events, flagged, updated_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
                    ON CONFLICT (drug_code, outcome_code, patient_location_id, analysis_scope, model_type)
                    DO UPDATE SET
                        drug_name = EXCLUDED.drug_name,
                        outcome_name = EXCLUDED.outcome_name,
                        estimate = EXCLUDED.estimate,
                        ci_lower = EXCLUDED.ci_lower,
                        ci_upper = EXCLUDED.ci_upper,
                        p_value = EXCLUDED.p_value,
                        signal_strength = EXCLUDED.signal_strength,
                        n_patients = EXCLUDED.n_patients,
                        n_events = EXCLUDED.n_events,
                        flagged = EXCLUDED.flagged,
                        updated_at = NOW()
                """, (
                    drug['code'], drug['name'], outcome['code'], outcome['name'],
                    location['id'] if location else None, scope, model_type, effect_measure,
                    estimate, ci_lower, ci_upper, p_value, signal_strength,
                    n_patients, n_events, flagged
                ))
            conn.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to save signal: {e}")
            conn.rollback()
            return False
    
    def _create_alert(
        self,
        conn,
        drug: Dict,
        outcome: Dict,
        location: Optional[Dict],
        estimate: float,
        ci_lower: float,
        ci_upper: float,
        p_value: float,
        n_patients: int,
        n_events: int
    ) -> bool:
        """Create alert for flagged signal"""
        try:
            import uuid
            
            severity = 'high' if estimate > 3.0 else 'medium' if estimate > 2.0 else 'low'
            
            details = {
                'type': 'drug_safety_signal',
                'drug_code': drug['code'],
                'drug_name': drug['name'],
                'outcome_code': outcome['code'],
                'outcome_name': outcome['name'],
                'location_id': location['id'] if location else None,
                'location_name': location['name'] if location else 'All locations',
                'odds_ratio': estimate,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'p_value': p_value,
                'n_patients': n_patients,
                'n_events': n_events,
                'recommendations': [
                    'Review prescribing patterns for this drug',
                    'Consider pharmacovigilance monitoring',
                    'Evaluate patient subgroups at higher risk'
                ]
            }
            
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO research_alerts (
                        id, alert_type, severity, risk_score,
                        details, status, created_at
                    ) VALUES (%s, %s, %s, %s, %s, 'active', NOW())
                """, (
                    str(uuid.uuid4()),
                    'drug_safety_signal',
                    severity,
                    min(estimate * 10, 100),
                    json.dumps(details)
                ))
            conn.commit()
            
            logger.info(f"Created drug safety alert: {drug['name']} -> {outcome['name']} (OR={estimate})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create alert: {e}")
            conn.rollback()
            return False


def run_scheduled_drug_scan():
    """Entry point for scheduled drug scanning job"""
    logger.info("Running scheduled drug safety scan")
    
    db_url = os.environ.get('DATABASE_URL')
    if not db_url:
        logger.error("DATABASE_URL not set")
        return
    
    scanner = DrugSafetyScanner(db_url)
    result = scanner.run_scan()
    
    audit_logger = EpidemiologyAuditLogger(db_url)
    audit_logger.log_background_scan(
        scan_type='scheduled',
        drugs_scanned=result.get('drugs_scanned', 0),
        outcomes_scanned=result.get('outcomes_scanned', 0),
        signals_generated=result.get('signals_generated', 0),
        alerts_created=result.get('alerts_created', 0),
        duration_seconds=result.get('duration_seconds', 0)
    )
    
    return result
