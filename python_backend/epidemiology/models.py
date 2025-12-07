"""
Epidemiology Database Models
============================
Defines data classes and SQL schema for epidemiology research tables.
Uses raw SQL with psycopg2 following existing codebase patterns.
"""

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional, List, Dict, Any
from enum import Enum
import json


class AnalysisScope(str, Enum):
    ALL = "all"
    MY_PATIENTS = "my_patients"
    RESEARCH_COHORT = "research_cohort"


class EffectMeasure(str, Enum):
    OR = "OR"
    HR = "HR"
    RR = "RR"
    ATE = "ATE"


class ModelType(str, Enum):
    LOGISTIC = "logistic"
    COX = "cox"
    IPTW = "iptw"
    MATCHING = "matching"
    DEEP_SURVIVAL = "deep_survival"


@dataclass
class DrugPrescription:
    """Patient-level drug prescription record (internal only)"""
    id: str
    patient_id: str
    prescriber_user_id: Optional[int]
    drug_code: str
    drug_name: str
    indication: Optional[str]
    dose: Optional[str]
    frequency: Optional[str]
    route: Optional[str]
    start_date: Optional[date]
    end_date: Optional[date]
    is_chronic: bool = False
    location_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class DrugExposure:
    """Processed drug exposure for analysis (internal only)"""
    id: str
    patient_id: str
    drug_code: str
    start_date: Optional[date]
    end_date: Optional[date]
    exposure_type: str  # 'current', 'recent', 'cumulative'
    daily_dose_mg: Optional[float] = None
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class AdverseEvent:
    """Adverse event record linked to drug (internal only)"""
    id: str
    patient_id: str
    related_drug_code: Optional[str]
    event_code: str
    event_name: str
    onset_date: Optional[date]
    seriousness: str  # 'non-serious', 'serious'
    outcome: str  # 'recovered', 'ongoing', 'death'
    study_id: Optional[int] = None
    location_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class DrugOutcomeSummary:
    """Aggregated summary for UI/LLM (anonymous, safe to expose)"""
    id: int
    drug_code: str
    drug_name: str
    outcome_code: str
    outcome_name: str
    patient_location_id: Optional[str]
    analysis_scope: str
    n_patients: int
    n_events: int
    n_exposed: int
    n_exposed_events: int
    n_unexposed: int
    n_unexposed_events: int
    incidence_exposed: Optional[float]
    incidence_unexposed: Optional[float]
    updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class DrugOutcomeSignal:
    """ML/Causal analysis output (anonymous, safe to expose)"""
    id: int
    drug_code: str
    drug_name: str
    outcome_code: str
    outcome_name: str
    patient_location_id: Optional[str]
    analysis_scope: str
    model_type: str
    effect_measure: str
    estimate: float
    ci_lower: float
    ci_upper: float
    p_value: float
    signal_strength: float
    n_patients: int
    n_events: int
    multiple_testing_adjusted: bool = False
    flagged: bool = False
    details_json: Optional[Dict[str, Any]] = None
    updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class InfectiousEvent:
    """Infectious disease event record"""
    id: str
    patient_id: str
    pathogen_code: str
    pathogen_name: str
    infection_type: str  # 'confirmed', 'probable', 'suspected'
    onset_date: Optional[date]
    resolution_date: Optional[date]
    hospitalized: bool = False
    icu_admission: bool = False
    outcome: str = "recovered"  # 'recovered', 'ongoing', 'death'
    location_id: Optional[str] = None
    outbreak_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ContactTrace:
    """Contact tracing record for infectious disease"""
    id: str
    source_patient_id: str
    contact_patient_id: str
    contact_date: Optional[date]
    contact_type: str  # 'household', 'workplace', 'community', 'healthcare'
    exposure_duration_minutes: Optional[int]
    distance_category: str  # 'close', 'proximate', 'distant'
    resulted_in_infection: bool = False
    outbreak_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class Outbreak:
    """Outbreak definition and tracking"""
    id: str
    name: str
    pathogen_code: str
    pathogen_name: str
    start_date: date
    end_date: Optional[date]
    location_id: Optional[str]
    status: str = "active"  # 'active', 'contained', 'ended'
    total_cases: int = 0
    total_deaths: int = 0
    notes: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ReproductionNumber:
    """R0/Rt calculation results"""
    id: str
    outbreak_id: Optional[str]
    pathogen_code: str
    location_id: Optional[str]
    calculation_date: date
    r_value: float
    r_lower: float
    r_upper: float
    generation_interval_days: float
    method: str  # 'exponential_growth', 'likelihood', 'bayesian'
    n_cases_used: int
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class SerologyResult:
    """Serology/seroprevalence test results"""
    id: str
    patient_id: str
    pathogen_code: str
    test_date: date
    test_type: str  # 'IgG', 'IgM', 'IgA', 'neutralizing'
    result: str  # 'positive', 'negative', 'indeterminate'
    titer_value: Optional[float] = None
    location_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class Immunization:
    """Vaccination record"""
    id: str
    patient_id: str
    vaccine_code: str
    vaccine_name: str
    dose_number: int
    administration_date: date
    lot_number: Optional[str] = None
    site: Optional[str] = None
    route: Optional[str] = None
    administering_provider_id: Optional[int] = None
    location_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class VaccineAdverseEvent:
    """Vaccine adverse event report (VAERS-style)"""
    id: str
    patient_id: str
    immunization_id: str
    vaccine_code: str
    event_code: str
    event_name: str
    onset_date: Optional[date]
    onset_interval_days: Optional[int]
    seriousness: str  # 'non-serious', 'serious', 'life-threatening'
    outcome: str  # 'recovered', 'ongoing', 'death', 'unknown'
    hospitalized: bool = False
    description: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class BiobankSample:
    """Biobank sample for genetic/molecular studies"""
    id: str
    patient_id: str
    sample_type: str  # 'blood', 'saliva', 'tissue', 'plasma'
    collection_date: date
    storage_location: Optional[str] = None
    quality_score: Optional[float] = None
    consent_for_research: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class GeneticVariant:
    """Genetic variant for pharmacogenomics/disease risk"""
    id: str
    patient_id: str
    gene: str
    variant_id: str  # rsID or HGVS notation
    genotype: str
    consequence: Optional[str] = None  # 'missense', 'synonymous', etc.
    clinical_significance: Optional[str] = None
    drug_interactions: Optional[List[str]] = None
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class GWASResult:
    """GWAS analysis results (aggregated)"""
    id: int
    study_id: Optional[int]
    trait: str
    variant_id: str
    gene: Optional[str]
    chromosome: str
    position: int
    effect_allele: str
    other_allele: str
    effect_size: float
    standard_error: float
    p_value: float
    n_samples: int
    population: str
    created_at: datetime = field(default_factory=datetime.utcnow)


SQL_SCHEMA = """
-- Pharmaco-epidemiology Tables
CREATE TABLE IF NOT EXISTS drug_prescriptions (
    id VARCHAR(36) PRIMARY KEY,
    patient_id VARCHAR(36) REFERENCES users(id),
    prescriber_user_id INTEGER REFERENCES users(id),
    drug_code VARCHAR(100) NOT NULL,
    drug_name TEXT NOT NULL,
    indication TEXT,
    dose VARCHAR(100),
    frequency VARCHAR(100),
    route VARCHAR(50),
    start_date DATE,
    end_date DATE,
    is_chronic BOOLEAN DEFAULT FALSE,
    location_id VARCHAR(36) REFERENCES locations(id),
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS drug_exposures (
    id VARCHAR(36) PRIMARY KEY,
    patient_id VARCHAR(36) REFERENCES users(id),
    drug_code VARCHAR(100) NOT NULL,
    start_date DATE,
    end_date DATE,
    exposure_type VARCHAR(50) NOT NULL,
    daily_dose_mg NUMERIC,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS adverse_events (
    id VARCHAR(36) PRIMARY KEY,
    patient_id VARCHAR(36) REFERENCES users(id),
    related_drug_code VARCHAR(100),
    event_code VARCHAR(100) NOT NULL,
    event_name TEXT NOT NULL,
    onset_date DATE,
    seriousness VARCHAR(50) NOT NULL,
    outcome VARCHAR(50) NOT NULL,
    study_id INTEGER REFERENCES research_studies(id),
    location_id VARCHAR(36) REFERENCES locations(id),
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS drug_outcome_summaries (
    id SERIAL PRIMARY KEY,
    drug_code VARCHAR(100) NOT NULL,
    drug_name TEXT NOT NULL,
    outcome_code VARCHAR(100) NOT NULL,
    outcome_name TEXT NOT NULL,
    patient_location_id VARCHAR(36) REFERENCES locations(id),
    analysis_scope VARCHAR(20) NOT NULL,
    n_patients INTEGER NOT NULL,
    n_events INTEGER NOT NULL,
    n_exposed INTEGER NOT NULL,
    n_exposed_events INTEGER NOT NULL,
    n_unexposed INTEGER NOT NULL,
    n_unexposed_events INTEGER NOT NULL,
    incidence_exposed NUMERIC,
    incidence_unexposed NUMERIC,
    updated_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(drug_code, outcome_code, patient_location_id, analysis_scope)
);

CREATE TABLE IF NOT EXISTS drug_outcome_signals (
    id SERIAL PRIMARY KEY,
    drug_code VARCHAR(100) NOT NULL,
    drug_name TEXT NOT NULL,
    outcome_code VARCHAR(100) NOT NULL,
    outcome_name TEXT NOT NULL,
    patient_location_id VARCHAR(36) REFERENCES locations(id),
    analysis_scope VARCHAR(20) NOT NULL,
    model_type VARCHAR(50) NOT NULL,
    effect_measure VARCHAR(20) NOT NULL,
    estimate NUMERIC NOT NULL,
    ci_lower NUMERIC NOT NULL,
    ci_upper NUMERIC NOT NULL,
    p_value NUMERIC NOT NULL,
    signal_strength NUMERIC NOT NULL,
    n_patients INTEGER NOT NULL,
    n_events INTEGER NOT NULL,
    multiple_testing_adjusted BOOLEAN DEFAULT FALSE,
    flagged BOOLEAN DEFAULT FALSE,
    details_json JSONB,
    updated_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(drug_code, outcome_code, patient_location_id, analysis_scope, model_type)
);

-- Infectious Disease Tables
CREATE TABLE IF NOT EXISTS infectious_events (
    id VARCHAR(36) PRIMARY KEY,
    patient_id VARCHAR(36) REFERENCES users(id),
    pathogen_code VARCHAR(100) NOT NULL,
    pathogen_name TEXT NOT NULL,
    infection_type VARCHAR(50) NOT NULL,
    onset_date DATE,
    resolution_date DATE,
    hospitalized BOOLEAN DEFAULT FALSE,
    icu_admission BOOLEAN DEFAULT FALSE,
    outcome VARCHAR(50) DEFAULT 'recovered',
    location_id VARCHAR(36) REFERENCES locations(id),
    outbreak_id VARCHAR(36),
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS outbreaks (
    id VARCHAR(36) PRIMARY KEY,
    name TEXT NOT NULL,
    pathogen_code VARCHAR(100) NOT NULL,
    pathogen_name TEXT NOT NULL,
    start_date DATE NOT NULL,
    end_date DATE,
    location_id VARCHAR(36) REFERENCES locations(id),
    status VARCHAR(50) DEFAULT 'active',
    total_cases INTEGER DEFAULT 0,
    total_deaths INTEGER DEFAULT 0,
    notes TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS contact_traces (
    id VARCHAR(36) PRIMARY KEY,
    source_patient_id VARCHAR(36) REFERENCES users(id),
    contact_patient_id VARCHAR(36) REFERENCES users(id),
    contact_date DATE,
    contact_type VARCHAR(50) NOT NULL,
    exposure_duration_minutes INTEGER,
    distance_category VARCHAR(50) NOT NULL,
    resulted_in_infection BOOLEAN DEFAULT FALSE,
    outbreak_id VARCHAR(36) REFERENCES outbreaks(id),
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS reproduction_numbers (
    id VARCHAR(36) PRIMARY KEY,
    outbreak_id VARCHAR(36) REFERENCES outbreaks(id),
    pathogen_code VARCHAR(100) NOT NULL,
    location_id VARCHAR(36) REFERENCES locations(id),
    calculation_date DATE NOT NULL,
    r_value NUMERIC NOT NULL,
    r_lower NUMERIC NOT NULL,
    r_upper NUMERIC NOT NULL,
    generation_interval_days NUMERIC NOT NULL,
    method VARCHAR(50) NOT NULL,
    n_cases_used INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS serology_results (
    id VARCHAR(36) PRIMARY KEY,
    patient_id VARCHAR(36) REFERENCES users(id),
    pathogen_code VARCHAR(100) NOT NULL,
    test_date DATE NOT NULL,
    test_type VARCHAR(50) NOT NULL,
    result VARCHAR(50) NOT NULL,
    titer_value NUMERIC,
    location_id VARCHAR(36) REFERENCES locations(id),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Vaccine Epidemiology Tables
CREATE TABLE IF NOT EXISTS epi_immunizations (
    id VARCHAR(36) PRIMARY KEY,
    patient_id VARCHAR(36) REFERENCES users(id),
    vaccine_code VARCHAR(100) NOT NULL,
    vaccine_name TEXT NOT NULL,
    dose_number INTEGER NOT NULL,
    administration_date DATE NOT NULL,
    lot_number VARCHAR(100),
    site VARCHAR(50),
    route VARCHAR(50),
    administering_provider_id INTEGER REFERENCES users(id),
    location_id VARCHAR(36) REFERENCES locations(id),
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS vaccine_adverse_events (
    id VARCHAR(36) PRIMARY KEY,
    patient_id VARCHAR(36) REFERENCES users(id),
    immunization_id VARCHAR(36) REFERENCES epi_immunizations(id),
    vaccine_code VARCHAR(100) NOT NULL,
    event_code VARCHAR(100) NOT NULL,
    event_name TEXT NOT NULL,
    onset_date DATE,
    onset_interval_days INTEGER,
    seriousness VARCHAR(50) NOT NULL,
    outcome VARCHAR(50) NOT NULL,
    hospitalized BOOLEAN DEFAULT FALSE,
    description TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Genetic/Molecular Epidemiology Tables
CREATE TABLE IF NOT EXISTS biobank_samples (
    id VARCHAR(36) PRIMARY KEY,
    patient_id VARCHAR(36) REFERENCES users(id),
    sample_type VARCHAR(50) NOT NULL,
    collection_date DATE NOT NULL,
    storage_location VARCHAR(100),
    quality_score NUMERIC,
    consent_for_research BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS genetic_variants (
    id VARCHAR(36) PRIMARY KEY,
    patient_id VARCHAR(36) REFERENCES users(id),
    gene VARCHAR(50) NOT NULL,
    variant_id VARCHAR(100) NOT NULL,
    genotype VARCHAR(20) NOT NULL,
    consequence VARCHAR(50),
    clinical_significance VARCHAR(50),
    drug_interactions JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS gwas_results (
    id SERIAL PRIMARY KEY,
    study_id INTEGER REFERENCES research_studies(id),
    trait VARCHAR(200) NOT NULL,
    variant_id VARCHAR(100) NOT NULL,
    gene VARCHAR(50),
    chromosome VARCHAR(10) NOT NULL,
    position BIGINT NOT NULL,
    effect_allele VARCHAR(10) NOT NULL,
    other_allele VARCHAR(10) NOT NULL,
    effect_size NUMERIC NOT NULL,
    standard_error NUMERIC NOT NULL,
    p_value NUMERIC NOT NULL,
    n_samples INTEGER NOT NULL,
    population VARCHAR(50) NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

-- ML Training Feature Store Tables
CREATE TABLE IF NOT EXISTS ml_drug_features (
    id SERIAL PRIMARY KEY,
    drug_code VARCHAR(100) NOT NULL,
    location_id VARCHAR(36),
    feature_date DATE NOT NULL,
    n_patients INTEGER,
    n_prescriptions INTEGER,
    avg_dose_mg NUMERIC,
    chronic_rate NUMERIC,
    adverse_event_rate NUMERIC,
    feature_vector JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(drug_code, location_id, feature_date)
);

CREATE TABLE IF NOT EXISTS ml_outbreak_features (
    id SERIAL PRIMARY KEY,
    pathogen_code VARCHAR(100) NOT NULL,
    location_id VARCHAR(36),
    feature_date DATE NOT NULL,
    case_count INTEGER,
    death_count INTEGER,
    hospitalization_rate NUMERIC,
    r_estimate NUMERIC,
    doubling_time_days NUMERIC,
    feature_vector JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(pathogen_code, location_id, feature_date)
);

CREATE TABLE IF NOT EXISTS ml_vaccine_features (
    id SERIAL PRIMARY KEY,
    vaccine_code VARCHAR(100) NOT NULL,
    location_id VARCHAR(36),
    feature_date DATE NOT NULL,
    coverage_rate NUMERIC,
    doses_administered INTEGER,
    adverse_event_rate NUMERIC,
    effectiveness_estimate NUMERIC,
    feature_vector JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(vaccine_code, location_id, feature_date)
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_drug_exposures_patient ON drug_exposures(patient_id);
CREATE INDEX IF NOT EXISTS idx_drug_exposures_drug ON drug_exposures(drug_code);
CREATE INDEX IF NOT EXISTS idx_adverse_events_patient ON adverse_events(patient_id);
CREATE INDEX IF NOT EXISTS idx_adverse_events_drug ON adverse_events(related_drug_code);
CREATE INDEX IF NOT EXISTS idx_drug_signals_drug ON drug_outcome_signals(drug_code);
CREATE INDEX IF NOT EXISTS idx_drug_signals_location ON drug_outcome_signals(patient_location_id);
CREATE INDEX IF NOT EXISTS idx_infectious_events_patient ON infectious_events(patient_id);
CREATE INDEX IF NOT EXISTS idx_infectious_events_pathogen ON infectious_events(pathogen_code);
CREATE INDEX IF NOT EXISTS idx_infectious_events_outbreak ON infectious_events(outbreak_id);
CREATE INDEX IF NOT EXISTS idx_contact_traces_source ON contact_traces(source_patient_id);
CREATE INDEX IF NOT EXISTS idx_reproduction_numbers_outbreak ON reproduction_numbers(outbreak_id);
CREATE INDEX IF NOT EXISTS idx_immunizations_patient ON epi_immunizations(patient_id);
CREATE INDEX IF NOT EXISTS idx_immunizations_vaccine ON epi_immunizations(vaccine_code);
"""


def initialize_epidemiology_schema(conn):
    """Initialize all epidemiology tables in database"""
    with conn.cursor() as cur:
        cur.execute(SQL_SCHEMA)
    conn.commit()
