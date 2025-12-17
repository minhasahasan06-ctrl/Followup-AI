"""
Epidemiology Data Warehouse Models
Production-grade SQLAlchemy models for surveillance, occupational, and genetic epidemiology.
Includes fact tables for data warehouse aggregation and HIPAA-compliant audit support.
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, JSON, Text, Date, Index, ForeignKey
from sqlalchemy.sql import func
from app.database import Base


class EpidemiologyCase(Base):
    """Individual disease/condition case for surveillance tracking"""
    __tablename__ = "epidemiology_cases"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(String, nullable=False, index=True)
    
    condition_code = Column(String, nullable=False, index=True)
    condition_name = Column(String, nullable=False)
    condition_category = Column(String)
    
    onset_date = Column(Date, nullable=False, index=True)
    diagnosis_date = Column(Date)
    resolution_date = Column(Date)
    
    severity = Column(String)
    outcome = Column(String)
    hospitalized = Column(Boolean, default=False)
    icu_admission = Column(Boolean, default=False)
    
    location_id = Column(String, index=True)
    location_name = Column(String)
    geo_region = Column(String, index=True)
    
    age_group = Column(String, index=True)
    sex = Column(String)
    
    lab_confirmed = Column(Boolean, default=False)
    specimen_type = Column(String)
    
    exposure_source = Column(String)
    transmission_mode = Column(String)
    
    extra_metadata = Column(JSON)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    __table_args__ = (
        Index('idx_epi_case_condition_date', 'condition_code', 'onset_date'),
        Index('idx_epi_case_location_date', 'location_id', 'onset_date'),
        Index('idx_epi_case_region_condition', 'geo_region', 'condition_code'),
    )


class InfectiousEvent(Base):
    """Infectious disease surveillance event"""
    __tablename__ = "infectious_events"

    id = Column(Integer, primary_key=True, index=True)
    
    pathogen_code = Column(String, nullable=False, index=True)
    pathogen_name = Column(String, nullable=False)
    pathogen_type = Column(String)
    
    case_id = Column(Integer, ForeignKey("epidemiology_cases.id"), index=True)
    patient_id = Column(String, nullable=False, index=True)
    
    event_date = Column(Date, nullable=False, index=True)
    event_type = Column(String, nullable=False)
    
    r_value = Column(Float)
    generation_time = Column(Float)
    serial_interval = Column(Float)
    
    contact_count = Column(Integer)
    secondary_cases = Column(Integer)
    
    variant_code = Column(String, index=True)
    variant_name = Column(String)
    
    vaccination_status = Column(String)
    days_since_vaccination = Column(Integer)
    
    location_id = Column(String, index=True)
    cluster_id = Column(String, index=True)
    
    extra_metadata = Column(JSON)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index('idx_infectious_pathogen_date', 'pathogen_code', 'event_date'),
        Index('idx_infectious_variant_date', 'variant_code', 'event_date'),
    )


class OccupationalIncident(Base):
    """Occupational health incident/exposure tracking"""
    __tablename__ = "occupational_incidents"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(String, nullable=False, index=True)
    
    industry_code = Column(String, nullable=False, index=True)
    industry_name = Column(String, nullable=False)
    occupation_code = Column(String, index=True)
    occupation_name = Column(String)
    
    hazard_code = Column(String, nullable=False, index=True)
    hazard_name = Column(String, nullable=False)
    hazard_category = Column(String)
    
    exposure_type = Column(String)
    exposure_duration_hours = Column(Float)
    exposure_frequency = Column(String)
    cumulative_exposure_years = Column(Float)
    
    incident_date = Column(Date, nullable=False, index=True)
    reported_date = Column(Date)
    
    outcome_code = Column(String, nullable=False, index=True)
    outcome_name = Column(String)
    outcome_severity = Column(String)
    
    lost_work_days = Column(Integer)
    permanent_disability = Column(Boolean, default=False)
    
    employer_id = Column(String, index=True)
    worksite_location = Column(String)
    
    ppe_used = Column(Boolean)
    ppe_type = Column(JSON)
    
    extra_metadata = Column(JSON)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    __table_args__ = (
        Index('idx_occ_industry_hazard', 'industry_code', 'hazard_code'),
        Index('idx_occ_hazard_outcome', 'hazard_code', 'outcome_code'),
        Index('idx_occ_incident_date', 'incident_date'),
    )


class GenomicMarkerStat(Base):
    """Genetic/molecular epidemiology marker statistics"""
    __tablename__ = "genomic_marker_stats"

    id = Column(Integer, primary_key=True, index=True)
    
    rsid = Column(String, nullable=False, index=True)
    gene_symbol = Column(String, nullable=False, index=True)
    gene_name = Column(String)
    chromosome = Column(String)
    position = Column(Integer)
    
    outcome_code = Column(String, nullable=False, index=True)
    outcome_name = Column(String, nullable=False)
    
    allele_frequency = Column(Float)
    effect_allele = Column(String)
    reference_allele = Column(String)
    
    odds_ratio = Column(Float)
    beta = Column(Float)
    se = Column(Float)
    ci_lower = Column(Float)
    ci_upper = Column(Float)
    p_value = Column(Float)
    
    n_carriers = Column(Integer)
    n_non_carriers = Column(Integer)
    n_total = Column(Integer)
    
    study_ancestry = Column(String)
    
    signal_strength = Column(Float)
    flagged = Column(Boolean, default=False)
    
    calculation_date = Column(Date, nullable=False)
    
    extra_metadata = Column(JSON)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index('idx_genomic_rsid_outcome', 'rsid', 'outcome_code'),
        Index('idx_genomic_gene_outcome', 'gene_symbol', 'outcome_code'),
        Index('idx_genomic_pvalue', 'p_value'),
    )


class PharmacogenomicInteraction(Base):
    """Drug-gene interaction for pharmacogenomics"""
    __tablename__ = "pharmacogenomic_interactions"

    id = Column(Integer, primary_key=True, index=True)
    
    rsid = Column(String, nullable=False, index=True)
    gene_symbol = Column(String, nullable=False, index=True)
    gene_name = Column(String)
    
    drug_code = Column(String, nullable=False, index=True)
    drug_name = Column(String, nullable=False)
    drug_class = Column(String)
    
    interaction_type = Column(String, nullable=False)
    phenotype = Column(String, nullable=False)
    
    recommendation = Column(Text)
    dosing_guidance = Column(Text)
    
    evidence_level = Column(String)
    clinical_impact = Column(String)
    
    n_patients = Column(Integer)
    effect_size = Column(Float)
    
    source = Column(String)
    pmid = Column(String)
    
    extra_metadata = Column(JSON)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    __table_args__ = (
        Index('idx_pgx_gene_drug', 'gene_symbol', 'drug_code'),
        Index('idx_pgx_rsid_drug', 'rsid', 'drug_code'),
    )


class DrugSafetySignal(Base):
    """Drug safety signal detection results"""
    __tablename__ = "drug_safety_signals"

    id = Column(Integer, primary_key=True, index=True)
    
    drug_code = Column(String, nullable=False, index=True)
    drug_name = Column(String, nullable=False)
    drug_class = Column(String)
    
    outcome_code = Column(String, nullable=False, index=True)
    outcome_name = Column(String, nullable=False)
    outcome_category = Column(String)
    
    location_id = Column(String, index=True)
    
    estimate = Column(Float, nullable=False)
    ci_lower = Column(Float)
    ci_upper = Column(Float)
    p_value = Column(Float)
    
    signal_strength = Column(Float)
    
    n_patients = Column(Integer)
    n_events = Column(Integer)
    n_controls = Column(Integer)
    
    flagged = Column(Boolean, default=False, index=True)
    suppressed = Column(Boolean, default=False)
    suppression_reason = Column(String)
    
    detection_method = Column(String)
    calculation_date = Column(Date, nullable=False, index=True)
    
    extra_metadata = Column(JSON)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index('idx_drug_signal_drug_outcome', 'drug_code', 'outcome_code'),
        Index('idx_drug_signal_flagged', 'flagged', 'calculation_date'),
    )


class VaccineCoverageRecord(Base):
    """Vaccine coverage tracking"""
    __tablename__ = "vaccine_coverage_records"

    id = Column(Integer, primary_key=True, index=True)
    
    vaccine_code = Column(String, nullable=False, index=True)
    vaccine_name = Column(String, nullable=False)
    vaccine_type = Column(String)
    
    location_id = Column(String, index=True)
    location_name = Column(String)
    
    report_date = Column(Date, nullable=False, index=True)
    
    dose_number = Column(Integer)
    
    coverage_rate = Column(Float)
    vaccinated_count = Column(Integer)
    eligible_population = Column(Integer)
    
    age_group = Column(String, index=True)
    
    extra_metadata = Column(JSON)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index('idx_vaccine_coverage_code_date', 'vaccine_code', 'report_date'),
        Index('idx_vaccine_coverage_location', 'location_id', 'report_date'),
    )


class VaccineEffectivenessRecord(Base):
    """Vaccine effectiveness study results"""
    __tablename__ = "vaccine_effectiveness_records"

    id = Column(Integer, primary_key=True, index=True)
    
    vaccine_code = Column(String, nullable=False, index=True)
    vaccine_name = Column(String, nullable=False)
    
    outcome_code = Column(String, nullable=False, index=True)
    outcome_name = Column(String, nullable=False)
    
    location_id = Column(String, index=True)
    
    report_date = Column(Date, nullable=False, index=True)
    
    effectiveness = Column(Float, nullable=False)
    ci_lower = Column(Float)
    ci_upper = Column(Float)
    
    n_vaccinated = Column(Integer)
    n_unvaccinated = Column(Integer)
    n_vaccinated_cases = Column(Integer)
    n_unvaccinated_cases = Column(Integer)
    
    dose_number = Column(Integer)
    days_since_vaccination = Column(String)
    variant_code = Column(String)
    
    study_design = Column(String)
    
    extra_metadata = Column(JSON)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index('idx_ve_vaccine_outcome', 'vaccine_code', 'outcome_code'),
        Index('idx_ve_report_date', 'report_date'),
    )


class DailySurveillanceAggregate(Base):
    """Daily aggregated surveillance data for warehouse queries"""
    __tablename__ = "daily_surveillance_aggregates"

    id = Column(Integer, primary_key=True, index=True)
    
    report_date = Column(Date, nullable=False, index=True)
    
    condition_code = Column(String, nullable=False, index=True)
    condition_name = Column(String, nullable=False)
    
    location_id = Column(String, index=True)
    geo_region = Column(String, index=True)
    
    case_count = Column(Integer, default=0)
    death_count = Column(Integer, default=0)
    hospitalization_count = Column(Integer, default=0)
    icu_count = Column(Integer, default=0)
    
    cumulative_cases = Column(Integer, default=0)
    cumulative_deaths = Column(Integer, default=0)
    
    r_value = Column(Float)
    r_lower = Column(Float)
    r_upper = Column(Float)
    
    by_age_group = Column(JSON)
    by_sex = Column(JSON)
    
    suppressed = Column(Boolean, default=False)
    suppression_reason = Column(String)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    __table_args__ = (
        Index('idx_daily_surv_date_condition', 'report_date', 'condition_code'),
        Index('idx_daily_surv_location', 'location_id', 'report_date'),
        Index('idx_daily_surv_region', 'geo_region', 'report_date'),
    )


class WeeklyIncidenceSummary(Base):
    """Weekly incidence rate summaries for trend analysis"""
    __tablename__ = "weekly_incidence_summaries"

    id = Column(Integer, primary_key=True, index=True)
    
    week_start = Column(Date, nullable=False, index=True)
    week_end = Column(Date, nullable=False)
    epi_week = Column(Integer)
    epi_year = Column(Integer)
    
    condition_code = Column(String, nullable=False, index=True)
    condition_name = Column(String, nullable=False)
    
    location_id = Column(String, index=True)
    geo_region = Column(String, index=True)
    
    case_count = Column(Integer, default=0)
    death_count = Column(Integer, default=0)
    
    incidence_rate = Column(Float)
    incidence_rate_per_100k = Column(Float)
    
    population_denominator = Column(Integer)
    
    percent_change_prev_week = Column(Float)
    
    trend_direction = Column(String)
    
    age_specific_rates = Column(JSON)
    
    suppressed = Column(Boolean, default=False)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index('idx_weekly_inc_week_condition', 'week_start', 'condition_code'),
        Index('idx_weekly_inc_epi_week', 'epi_year', 'epi_week', 'condition_code'),
    )


class OccupationalCohort(Base):
    """Occupational cohort for longitudinal exposure studies"""
    __tablename__ = "occupational_cohorts"

    id = Column(Integer, primary_key=True, index=True)
    
    cohort_name = Column(String, nullable=False)
    cohort_description = Column(Text)
    
    industry_code = Column(String, nullable=False, index=True)
    industry_name = Column(String, nullable=False)
    
    hazard_code = Column(String, index=True)
    hazard_name = Column(String)
    
    n_workers = Column(Integer)
    n_active = Column(Integer)
    
    enrollment_start = Column(Date)
    enrollment_end = Column(Date)
    
    mean_exposure_years = Column(Float)
    median_exposure_years = Column(Float)
    
    follow_up_person_years = Column(Float)
    
    outcome_summary = Column(JSON)
    
    extra_metadata = Column(JSON)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())



class SurveillanceLocation(Base):
    """Location reference for surveillance data"""
    __tablename__ = "surveillance_locations"

    id = Column(String, primary_key=True, index=True)
    
    name = Column(String, nullable=False)
    location_type = Column(String)
    
    city = Column(String)
    state = Column(String, index=True)
    country = Column(String, index=True)
    postal_code = Column(String)
    
    latitude = Column(Float)
    longitude = Column(Float)
    
    parent_location_id = Column(String, ForeignKey("surveillance_locations.id"))
    
    population = Column(Integer)
    
    extra_metadata = Column(JSON)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    __table_args__ = (
        Index('idx_surv_location_state', 'state'),
        Index('idx_surv_location_type', 'location_type'),
    )
