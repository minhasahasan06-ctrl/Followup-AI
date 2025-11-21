"""
Medication Side-Effect Predictor Models - HIPAA-compliant medication tracking
Correlates symptom onset with medication changes using AI analysis
WITHOUT making medical diagnoses - focuses on pattern detection
"""

from sqlalchemy import Column, String, Integer, Float, DateTime, Text, Boolean, JSON, ForeignKey, Enum as SQLEnum
from sqlalchemy.sql import func
from app.database import Base
import enum


class SymptomSource(str, enum.Enum):
    """Where the symptom data came from"""
    MANUAL = "manual"  # Patient manually logged
    DAILY_FOLLOWUP = "daily_followup"  # From Daily Followup Camera
    AGENT_CLONA = "agent_clona"  # Extracted from chat
    PAIN_TRACKING = "pain_tracking"  # From pain detection camera
    EXAM_COACH = "exam_coach"  # From home clinical exam


class CorrelationStrength(str, enum.Enum):
    """AI-assessed correlation strength (NOT diagnosis)"""
    UNLIKELY = "unlikely"  # < 30% correlation
    POSSIBLE = "possible"  # 30-60% correlation
    LIKELY = "likely"  # 60-85% correlation
    STRONG = "strong"  # > 85% correlation


class ChangeReason(str, enum.Enum):
    """Why medication dosage changed"""
    INITIAL_DOSE = "initial_dose"
    DOSE_INCREASE = "dose_increase"
    DOSE_DECREASE = "dose_decrease"
    SWITCHED_MEDICATION = "switched_medication"
    DISCONTINUED = "discontinued"
    RESUMED = "resumed"


class MedicationTimeline(Base):
    """
    Extended medication tracking with dosage history
    Links to existing Medication model data
    """
    __tablename__ = "medication_timeline"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(String, nullable=False, index=True)
    medication_id = Column(String, ForeignKey("medications.id"))
    
    # Medication details (denormalized for historical accuracy)
    medication_name = Column(String, nullable=False)
    generic_name = Column(String)  # Generic drug name
    drug_class = Column(String)  # e.g., "NSAID", "ACE Inhibitor"
    
    # Current dosage
    dosage = Column(String, nullable=False)  # e.g., "10mg"
    frequency = Column(String, nullable=False)  # e.g., "twice daily"
    route = Column(String)  # e.g., "oral", "topical"
    
    # Timeline
    started_at = Column(DateTime(timezone=True), nullable=False)
    stopped_at = Column(DateTime(timezone=True))
    is_active = Column(Boolean, default=True, index=True)
    
    # Prescriber info
    prescribed_by = Column(String)  # Doctor name or ID
    prescription_reason = Column(Text)  # Why prescribed
    
    # Patient-reported
    patient_notes = Column(Text)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class DosageChange(Base):
    """
    Tracks every dosage change for pattern analysis
    Critical for correlating symptom onset with medication changes
    """
    __tablename__ = "dosage_changes"

    id = Column(Integer, primary_key=True, index=True)
    medication_timeline_id = Column(Integer, ForeignKey("medication_timeline.id"), nullable=False)
    patient_id = Column(String, nullable=False, index=True)
    
    # Change details
    change_reason = Column(SQLEnum(ChangeReason), nullable=False)
    change_date = Column(DateTime(timezone=True), nullable=False, index=True)
    
    # Dosage details
    old_dosage = Column(String)
    new_dosage = Column(String, nullable=False)
    old_frequency = Column(String)
    new_frequency = Column(String, nullable=False)
    
    # Context
    changed_by = Column(String)  # Doctor ID or name
    change_notes = Column(Text)
    patient_reported = Column(Boolean, default=False)  # If patient initiated
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class SymptomLog(Base):
    """
    Unified symptom logging from all sources
    Aggregates symptoms for correlation analysis
    """
    __tablename__ = "symptom_logs"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(String, nullable=False, index=True)
    
    # Symptom details
    symptom_name = Column(String, nullable=False, index=True)  # Normalized name
    symptom_description = Column(Text)  # Raw patient description
    severity = Column(Integer)  # 1-10 scale (if reported)
    
    # Source tracking
    source = Column(SQLEnum(SymptomSource), nullable=False, index=True)
    source_id = Column(Integer)  # ID from source table (measurement_id, message_id, etc.)
    
    # Timeline
    reported_at = Column(DateTime(timezone=True), nullable=False, index=True)
    onset_date = Column(DateTime(timezone=True))  # When symptom first started
    duration_hours = Column(Integer)  # How long symptom lasted
    
    # Context
    body_area = Column(String)  # Where symptom occurred
    triggers = Column(JSON)  # ["exercise", "eating", "stress"]
    associated_symptoms = Column(JSON)  # Other symptoms at same time
    
    # AI extraction metadata (if from chat)
    extracted_by_ai = Column(Boolean, default=False)
    extraction_confidence = Column(Float)  # 0-1 confidence
    original_text = Column(Text)  # Raw text from chat
    
    # Status
    resolved = Column(Boolean, default=False)
    resolved_at = Column(DateTime(timezone=True))
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class SideEffectCorrelation(Base):
    """
    AI-generated correlations between medications and symptoms
    IMPORTANT: These are patterns, NOT medical diagnoses
    """
    __tablename__ = "side_effect_correlations"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(String, nullable=False, index=True)
    
    # What's being correlated
    medication_timeline_id = Column(Integer, ForeignKey("medication_timeline.id"), nullable=False)
    symptom_log_id = Column(Integer, ForeignKey("symptom_logs.id"), nullable=False)
    dosage_change_id = Column(Integer, ForeignKey("dosage_changes.id"))  # Optional: specific to dosage change
    
    # Medication details (denormalized)
    medication_name = Column(String, nullable=False)
    dosage_at_onset = Column(String)
    
    # Symptom details (denormalized)
    symptom_name = Column(String, nullable=False)
    
    # Timing analysis
    time_to_onset_hours = Column(Integer)  # Hours between med change and symptom
    symptom_onset_date = Column(DateTime(timezone=True))
    medication_change_date = Column(DateTime(timezone=True))
    
    # AI correlation analysis
    correlation_strength = Column(SQLEnum(CorrelationStrength), nullable=False)
    confidence_score = Column(Float, nullable=False)  # 0-1 AI confidence
    
    # Analysis details
    supporting_evidence = Column(JSON)  # List of evidence points
    temporal_pattern = Column(Text)  # Time-to-onset curve description
    dose_response = Column(Text)  # If higher dose = stronger symptom
    rechallenge_data = Column(Text)  # If symptom recurred after re-exposure
    
    # AI reasoning (from OpenAI)
    ai_reasoning = Column(Text)  # Why this correlation was made
    known_side_effect = Column(Boolean)  # If this is a known side effect
    similar_cases = Column(JSON)  # Similar patterns in medical literature
    
    # Clinical relevance
    patient_impact = Column(String)  # "mild", "moderate", "severe"
    action_recommended = Column(Text)  # "Monitor closely", "Consider alternative", etc.
    
    # Review status
    reviewed_by_doctor = Column(Boolean, default=False)
    doctor_notes = Column(Text)
    reviewed_at = Column(DateTime(timezone=True))
    
    # Analysis metadata
    analysis_date = Column(DateTime(timezone=True), nullable=False, index=True)
    model_version = Column(String)  # OpenAI model used
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class MedicationEffectsSummary(Base):
    """
    Patient-level summary of medication side effects
    Used for doctor consultations and patient dashboard
    """
    __tablename__ = "medication_effects_summaries"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(String, nullable=False, index=True)
    
    # Time period
    summary_date = Column(DateTime(timezone=True), nullable=False)
    period_start = Column(DateTime(timezone=True), nullable=False)
    period_end = Column(DateTime(timezone=True), nullable=False)
    
    # Counts
    active_medications = Column(Integer, default=0)
    total_correlations = Column(Integer, default=0)
    strong_correlations = Column(Integer, default=0)
    
    # Top correlations
    medications_with_effects = Column(JSON)  # [{"med": "X", "symptoms": ["Y", "Z"]}]
    symptoms_by_medication = Column(JSON)  # Grouped view
    
    # Overall trends
    new_symptoms_count = Column(Integer, default=0)
    resolved_symptoms_count = Column(Integer, default=0)
    
    # AI-generated insights
    key_findings = Column(JSON)  # Top 3-5 findings
    clinician_recommendations = Column(Text)  # For doctor review
    patient_friendly_summary = Column(Text)  # Non-technical summary
    
    # Risk flags
    high_impact_correlations = Column(JSON)  # Correlations needing attention
    medication_interactions_possible = Column(Boolean, default=False)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
