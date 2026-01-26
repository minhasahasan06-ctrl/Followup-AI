"""
Risk & Exposures ETL Jobs

Background jobs for automatically deriving risk exposure data:
- Infectious events from conditions + visits
- Immunizations from vaccine records and prescriptions
- Occupational exposures from job titles
- Genetic risk flags from lab results
"""

import json
import os
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from pathlib import Path

import psycopg2
from psycopg2.extras import RealDictCursor

logger = logging.getLogger(__name__)

RESOURCES_DIR = Path(__file__).parent / "resources"


def load_mapping(filename: str) -> Dict[str, Any]:
    """Load a mapping file from resources."""
    filepath = RESOURCES_DIR / filename
    if not filepath.exists():
        logger.warning(f"Mapping file not found: {filepath}")
        return {}
    with open(filepath, "r") as f:
        return json.load(f)


def get_db_connection():
    """Get database connection using DATABASE_URL."""
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise ValueError("DATABASE_URL environment variable not set")
    return psycopg2.connect(database_url)


class InfectiousEventsETL:
    """ETL job for deriving infectious events from conditions and visits."""
    
    def __init__(self):
        self.infection_map = load_mapping("infection_code_map.json")
        self.codes = self.infection_map.get("codes", {})
        self.severity_modifiers = self.infection_map.get("severity_modifiers", {})
    
    def get_severity(self, base_severity: str, hospitalization: bool, icu: bool, ventilator: bool) -> str:
        """Determine final severity based on hospitalization status."""
        severity = base_severity
        
        if ventilator and "ventilator" in self.severity_modifiers:
            modifiers = self.severity_modifiers["ventilator"]
            severity = modifiers.get(severity, "critical")
        elif icu and "icu_admission" in self.severity_modifiers:
            modifiers = self.severity_modifiers["icu_admission"]
            severity = modifiers.get(severity, "severe")
        elif hospitalization and "hospitalization" in self.severity_modifiers:
            modifiers = self.severity_modifiers["hospitalization"]
            severity = modifiers.get(severity, severity)
        
        return severity
    
    def run(self) -> Dict[str, int]:
        """Run the ETL job to update infectious_events table."""
        stats = {"processed": 0, "created": 0, "updated": 0, "skipped": 0}
        
        try:
            conn = get_db_connection()
            cur = conn.cursor(cursor_factory=RealDictCursor)
            
            cur.execute("""
                SELECT c.id, c.patient_id, c.condition_code, c.condition_name,
                       c.onset_date, c.resolution_date, c.severity as condition_severity,
                       v.id as visit_id, v.is_hospitalization, v.icu_admission, 
                       v.ventilator_required, v.location_id
                FROM patient_conditions c
                LEFT JOIN patient_visits v ON v.patient_id = c.patient_id
                    AND v.primary_diagnosis_code = c.condition_code
                    AND v.admission_date BETWEEN c.onset_date - INTERVAL '7 days' 
                        AND COALESCE(c.resolution_date, NOW()) + INTERVAL '7 days'
                WHERE c.condition_category = 'infectious'
                   OR c.condition_code = ANY(%s)
            """, [list(self.codes.keys())])
            
            conditions = cur.fetchall()
            
            for condition in conditions:
                stats["processed"] += 1
                code = condition["condition_code"]
                
                if code not in self.codes:
                    stats["skipped"] += 1
                    continue
                
                mapping = self.codes[code]
                
                hospitalization = bool(condition.get("is_hospitalization"))
                icu = bool(condition.get("icu_admission"))
                ventilator = bool(condition.get("ventilator_required"))
                
                base_severity = mapping.get("default_severity", "moderate")
                final_severity = self.get_severity(base_severity, hospitalization, icu, ventilator)
                
                cur.execute("""
                    SELECT id FROM infectious_events
                    WHERE patient_id = %s 
                      AND related_condition_id = %s
                """, [condition["patient_id"], condition["id"]])
                
                existing = cur.fetchone()
                
                duration_days = None
                if condition.get("onset_date") and condition.get("resolution_date"):
                    delta = condition["resolution_date"] - condition["onset_date"]
                    duration_days = delta.days
                
                if existing:
                    cur.execute("""
                        UPDATE infectious_events SET
                            infection_type = %s,
                            pathogen = %s,
                            pathogen_category = %s,
                            severity = %s,
                            onset_date = %s,
                            resolution_date = %s,
                            duration_days = %s,
                            hospitalization = %s,
                            icu_admission = %s,
                            ventilator_required = %s,
                            related_visit_id = %s,
                            location_id = %s,
                            last_etl_processed_at = NOW(),
                            updated_at = NOW()
                        WHERE id = %s AND manual_override = FALSE
                    """, [
                        mapping.get("infection_type"),
                        mapping.get("pathogen"),
                        mapping.get("pathogen_category"),
                        final_severity,
                        condition.get("onset_date"),
                        condition.get("resolution_date"),
                        duration_days,
                        hospitalization,
                        icu,
                        ventilator,
                        condition.get("visit_id"),
                        condition.get("location_id"),
                        existing["id"]
                    ])
                    stats["updated"] += 1
                else:
                    cur.execute("""
                        INSERT INTO infectious_events (
                            patient_id, infection_type, pathogen, pathogen_category,
                            severity, onset_date, resolution_date, duration_days,
                            hospitalization, icu_admission, ventilator_required,
                            related_condition_id, related_condition_code, related_visit_id,
                            location_id, auto_generated, last_etl_processed_at
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, TRUE, NOW())
                    """, [
                        condition["patient_id"],
                        mapping.get("infection_type"),
                        mapping.get("pathogen"),
                        mapping.get("pathogen_category"),
                        final_severity,
                        condition.get("onset_date"),
                        condition.get("resolution_date"),
                        duration_days,
                        hospitalization,
                        icu,
                        ventilator,
                        condition["id"],
                        code,
                        condition.get("visit_id"),
                        condition.get("location_id")
                    ])
                    stats["created"] += 1
            
            conn.commit()
            cur.close()
            conn.close()
            
            logger.info(f"[RISK_ETL] Infectious events ETL completed: {stats}")
            
        except Exception as e:
            logger.error(f"[RISK_ETL] Infectious events ETL failed: {str(e)}")
            raise
        
        return stats


class ImmunizationsETL:
    """ETL job for normalizing vaccine records from prescriptions."""
    
    def __init__(self):
        self.vaccine_map = load_mapping("vaccine_code_map.json")
        self.cvx_codes = self.vaccine_map.get("cvx_codes", {})
        self.atc_codes = self.vaccine_map.get("atc_codes", {})
    
    def run(self) -> Dict[str, int]:
        """Run the ETL job to normalize immunization records."""
        stats = {"processed": 0, "created": 0, "updated": 0, "skipped": 0}
        
        try:
            conn = get_db_connection()
            cur = conn.cursor(cursor_factory=RealDictCursor)
            
            cur.execute("""
                SELECT p.id, p.user_id as patient_id, p.drug_code, p.drug_name,
                       p.start_date, p.prescribing_doctor_id
                FROM prescriptions p
                WHERE p.drug_code = ANY(%s)
                   OR p.drug_code = ANY(%s)
                   OR LOWER(p.drug_name) LIKE '%%vaccine%%'
                   OR LOWER(p.drug_name) LIKE '%%immunization%%'
            """, [list(self.cvx_codes.keys()), list(self.atc_codes.keys())])
            
            prescriptions = cur.fetchall()
            
            for rx in prescriptions:
                stats["processed"] += 1
                drug_code = rx.get("drug_code", "")
                
                mapping = None
                if drug_code in self.cvx_codes:
                    mapping = self.cvx_codes[drug_code]
                elif drug_code in self.atc_codes:
                    mapping = self.atc_codes[drug_code]
                
                if not mapping:
                    stats["skipped"] += 1
                    continue
                
                cur.execute("""
                    SELECT id FROM patient_immunizations
                    WHERE patient_id = %s 
                      AND vaccine_code = %s
                      AND DATE(administration_date) = DATE(%s)
                """, [rx["patient_id"], drug_code, rx.get("start_date")])
                
                existing = cur.fetchone()
                
                if existing:
                    stats["skipped"] += 1
                    continue
                
                cur.execute("""
                    INSERT INTO patient_immunizations (
                        patient_id, vaccine_code, vaccine_name, vaccine_manufacturer,
                        series_name, administration_date, source_type, source_record_id,
                        auto_generated
                    ) VALUES (%s, %s, %s, %s, %s, %s, 'prescription', %s, TRUE)
                """, [
                    rx["patient_id"],
                    drug_code,
                    mapping.get("vaccine_name"),
                    mapping.get("manufacturer"),
                    mapping.get("series_name"),
                    rx.get("start_date"),
                    rx["id"]
                ])
                stats["created"] += 1
            
            conn.commit()
            cur.close()
            conn.close()
            
            logger.info(f"[RISK_ETL] Immunizations ETL completed: {stats}")
            
        except Exception as e:
            logger.error(f"[RISK_ETL] Immunizations ETL failed: {str(e)}")
            raise
        
        return stats


class OccupationalExposuresETL:
    """ETL job for inferring occupational exposures from job titles."""
    
    def __init__(self):
        self.occupation_map = load_mapping("occupation_exposure_map.json")
        self.industries = self.occupation_map.get("industries", {})
        self.job_titles = self.occupation_map.get("job_titles", {})
    
    def normalize_job_title(self, title: str) -> str:
        """Normalize job title for matching."""
        return title.lower().strip()
    
    def find_matching_title(self, title: str) -> Optional[Dict[str, Any]]:
        """Find a matching job title in the map."""
        normalized = self.normalize_job_title(title)
        
        if normalized in self.job_titles:
            return self.job_titles[normalized]
        
        for key, value in self.job_titles.items():
            if key in normalized or normalized in key:
                return value
        
        return None
    
    def run(self) -> Dict[str, int]:
        """Run the ETL job to infer occupational exposures."""
        stats = {"processed": 0, "created": 0, "updated": 0, "skipped": 0}
        
        try:
            conn = get_db_connection()
            cur = conn.cursor(cursor_factory=RealDictCursor)
            
            cur.execute("""
                SELECT o.id, o.patient_id, o.job_title, o.industry, 
                       o.physical_demand_level, o.auto_enriched
                FROM patient_occupations o
                WHERE o.auto_enriched = FALSE OR o.physical_demand_level IS NULL
            """)
            
            occupations = cur.fetchall()
            
            for occupation in occupations:
                stats["processed"] += 1
                
                job_match = self.find_matching_title(occupation["job_title"])
                industry = occupation.get("industry")
                
                if job_match:
                    industry = job_match.get("industry", industry)
                    physical_demand = job_match.get("physical_demand_level")
                    exposures = job_match.get("overrides", [])
                elif industry and industry.lower() in self.industries:
                    industry_data = self.industries[industry.lower()]
                    physical_demand = industry_data.get("physical_demand_level")
                    exposures = industry_data.get("default_exposures", [])
                else:
                    stats["skipped"] += 1
                    continue
                
                if industry and industry.lower() in self.industries and not job_match:
                    industry_data = self.industries[industry.lower()]
                    exposures = industry_data.get("default_exposures", [])
                elif job_match and not exposures:
                    if industry and industry.lower() in self.industries:
                        industry_data = self.industries[industry.lower()]
                        exposures = industry_data.get("default_exposures", [])
                
                cur.execute("""
                    UPDATE patient_occupations SET
                        industry = COALESCE(industry, %s),
                        physical_demand_level = COALESCE(physical_demand_level, %s),
                        auto_enriched = TRUE,
                        enriched_at = NOW(),
                        updated_at = NOW()
                    WHERE id = %s
                """, [industry, physical_demand, occupation["id"]])
                stats["updated"] += 1
                
                for exposure in exposures:
                    cur.execute("""
                        SELECT id FROM occupational_exposures
                        WHERE occupation_id = %s AND exposure_type = %s
                    """, [occupation["id"], exposure.get("type")])
                    
                    if not cur.fetchone():
                        cur.execute("""
                            INSERT INTO occupational_exposures (
                                occupation_id, patient_id, exposure_type, 
                                exposure_agent, exposure_level, auto_generated
                            ) VALUES (%s, %s, %s, %s, %s, TRUE)
                        """, [
                            occupation["id"],
                            occupation["patient_id"],
                            exposure.get("type"),
                            exposure.get("agent"),
                            exposure.get("level", "medium")
                        ])
                        stats["created"] += 1
            
            conn.commit()
            cur.close()
            conn.close()
            
            logger.info(f"[RISK_ETL] Occupational exposures ETL completed: {stats}")
            
        except Exception as e:
            logger.error(f"[RISK_ETL] Occupational exposures ETL failed: {str(e)}")
            raise
        
        return stats


class GeneticRiskFlagsETL:
    """ETL job for deriving genetic risk flags from lab/genetic results."""
    
    def __init__(self):
        self.genetic_map = load_mapping("genetic_marker_map.json")
        self.markers = self.genetic_map.get("genetic_markers", {})
        self.lab_markers = self.genetic_map.get("lab_markers", {})
    
    def run(self) -> Dict[str, int]:
        """Run the ETL job to derive genetic risk flags."""
        stats = {"processed": 0, "created": 0, "updated": 0, "skipped": 0}
        
        try:
            conn = get_db_connection()
            cur = conn.cursor(cursor_factory=RealDictCursor)
            
            cur.execute("""
                SELECT gv.id, gv.user_id as patient_id, gv.gene, gv.variant,
                       gv.genotype, gv.phenotype, gv.clinical_significance,
                       gv.testing_provider, gv.test_date
                FROM genetic_variants gv
                WHERE gv.clinical_significance IN ('pathogenic', 'likely_pathogenic')
            """)
            
            variants = cur.fetchall()
            
            for variant in variants:
                stats["processed"] += 1
                
                marker_key = f"{variant['gene']}{variant.get('variant', '')}"
                marker_key_simple = variant['gene']
                
                mapping = None
                for key in [marker_key, marker_key_simple]:
                    if key in self.markers:
                        mapping = self.markers[key]
                        break
                    if key in self.lab_markers:
                        mapped_key = self.lab_markers[key].get("maps_to")
                        if mapped_key in self.markers:
                            mapping = self.markers[mapped_key]
                            break
                
                if not mapping:
                    stats["skipped"] += 1
                    continue
                
                cur.execute("""
                    SELECT id FROM genetic_risk_flags
                    WHERE patient_id = %s AND flag_name = %s
                """, [variant["patient_id"], mapping.get("flag_name")])
                
                existing = cur.fetchone()
                
                if existing:
                    cur.execute("""
                        UPDATE genetic_risk_flags SET
                            value = 'present',
                            risk_level = %s,
                            clinical_implications = %s,
                            affected_medications = %s,
                            affected_conditions = %s,
                            source_record_id = %s,
                            testing_provider = %s,
                            recorded_date = %s,
                            updated_at = NOW()
                        WHERE id = %s AND manual_override = FALSE
                    """, [
                        mapping.get("risk_level_if_present"),
                        mapping.get("clinical_implications"),
                        json.dumps(mapping.get("affected_medications", [])),
                        json.dumps(mapping.get("affected_conditions", [])),
                        variant["id"],
                        variant.get("testing_provider"),
                        variant.get("test_date"),
                        existing["id"]
                    ])
                    stats["updated"] += 1
                else:
                    cur.execute("""
                        INSERT INTO genetic_risk_flags (
                            patient_id, flag_name, flag_type, value, risk_level,
                            clinical_implications, affected_medications, affected_conditions,
                            source, source_record_id, testing_provider, recorded_date,
                            auto_generated
                        ) VALUES (%s, %s, %s, 'present', %s, %s, %s, %s, 'genetic_panel', %s, %s, %s, TRUE)
                    """, [
                        variant["patient_id"],
                        mapping.get("flag_name"),
                        mapping.get("flag_type"),
                        mapping.get("risk_level_if_present"),
                        mapping.get("clinical_implications"),
                        json.dumps(mapping.get("affected_medications", [])),
                        json.dumps(mapping.get("affected_conditions", [])),
                        variant["id"],
                        variant.get("testing_provider"),
                        variant.get("test_date")
                    ])
                    stats["created"] += 1
            
            conn.commit()
            cur.close()
            conn.close()
            
            logger.info(f"[RISK_ETL] Genetic risk flags ETL completed: {stats}")
            
        except Exception as e:
            logger.error(f"[RISK_ETL] Genetic risk flags ETL failed: {str(e)}")
            raise
        
        return stats


def log_etl_job(job_type: str, status: str, stats: Dict[str, int], 
                error_message: Optional[str] = None, execution_time_ms: Optional[int] = None):
    """Log ETL job execution to database."""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        cur.execute("""
            INSERT INTO risk_exposures_etl_jobs (
                job_type, status, records_processed, records_created,
                records_updated, records_skipped, execution_time_ms,
                error_message, started_at, completed_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW())
        """, [
            job_type,
            status,
            stats.get("processed", 0),
            stats.get("created", 0),
            stats.get("updated", 0),
            stats.get("skipped", 0),
            execution_time_ms,
            error_message
        ])
        
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        logger.error(f"[RISK_ETL] Failed to log ETL job: {str(e)}")


def run_all_etl_jobs():
    """Run all Risk & Exposures ETL jobs."""
    jobs = [
        ("infectious_events", InfectiousEventsETL),
        ("immunizations", ImmunizationsETL),
        ("occupational_exposures", OccupationalExposuresETL),
        ("genetic_flags", GeneticRiskFlagsETL),
    ]
    
    results = {}
    
    for job_type, job_class in jobs:
        start_time = datetime.now()
        try:
            job = job_class()
            stats = job.run()
            execution_time = int((datetime.now() - start_time).total_seconds() * 1000)
            log_etl_job(job_type, "completed", stats, execution_time_ms=execution_time)
            results[job_type] = {"status": "completed", "stats": stats}
        except Exception as e:
            execution_time = int((datetime.now() - start_time).total_seconds() * 1000)
            log_etl_job(job_type, "failed", {}, error_message=str(e), execution_time_ms=execution_time)
            results[job_type] = {"status": "failed", "error": str(e)}
            logger.error(f"[RISK_ETL] Job {job_type} failed: {str(e)}")
    
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    results = run_all_etl_jobs()
    print(json.dumps(results, indent=2))
