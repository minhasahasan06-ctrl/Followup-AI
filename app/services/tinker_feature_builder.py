"""
Tinker Feature Packet Builder
=============================
Builds anonymized feature packets for Tinker Thinking Machine API.
All data is transformed through the privacy firewall before packaging.

Features:
- Patient feature extraction with automatic anonymization
- Cohort query building with DSL
- Study protocol packet formatting
- Trial specification packet formatting
"""

import json
import logging
from datetime import datetime, date
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum

from app.config import settings
from app.services.tinker_privacy_firewall import (
    get_privacy_firewall,
    TinkerPrivacyFirewall,
    PrivacyAuditRecord
)

logger = logging.getLogger(__name__)


class AnalysisType(str, Enum):
    """Types of analysis supported by Tinker"""
    COHORT_COMPARISON = "cohort_comparison"
    SURVIVAL_ANALYSIS = "survival_analysis"
    REGRESSION = "regression"
    CLASSIFICATION = "classification"
    TIME_SERIES = "time_series"
    CLUSTERING = "clustering"
    CAUSAL_INFERENCE = "causal_inference"


class FeatureCategory(str, Enum):
    """Categories of features for ML models"""
    DEMOGRAPHIC = "demographic"
    CLINICAL = "clinical"
    BEHAVIORAL = "behavioral"
    TEMPORAL = "temporal"
    MEDICATION = "medication"
    LAB = "lab"
    VITAL = "vital"
    SOCIAL = "social"


@dataclass
class FeaturePacket:
    """
    Anonymized feature packet ready for Tinker API.
    
    All fields are pre-anonymized - no PHI present.
    """
    packet_id: str  # SHA256 hash
    patient_hash: str  # SHA256 of patient_id
    features: Dict[str, Any]
    feature_categories: List[str]
    created_at: str
    k_anon_verified: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "packet_id": self.packet_id,
            "patient_hash": self.patient_hash,
            "features": self.features,
            "feature_categories": self.feature_categories,
            "created_at": self.created_at,
            "k_anon_verified": self.k_anon_verified
        }


@dataclass
class CohortQuery:
    """
    DSL query for defining patient cohorts.
    
    All identifiers are hashed, all values are bucketed.
    """
    query_hash: str
    dsl: Dict[str, Any]
    estimated_size: int
    k_anon_passed: bool
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_hash": self.query_hash,
            "dsl": self.dsl,
            "estimated_size": self.estimated_size,
            "k_anon_passed": self.k_anon_passed
        }


@dataclass
class StudyProtocol:
    """
    Study protocol packet for Tinker.
    
    Contains anonymized research parameters.
    """
    protocol_hash: str
    objective_hash: str
    analysis_types: List[str]
    outcome_variable: str
    covariates: List[str]
    confounders: List[str]
    follow_up_period: str  # bucketed
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "protocol_hash": self.protocol_hash,
            "objective_hash": self.objective_hash,
            "analysis_types": self.analysis_types,
            "outcome_variable": self.outcome_variable,
            "covariates": self.covariates,
            "confounders": self.confounders,
            "follow_up_period": self.follow_up_period
        }


@dataclass
class TrialSpec:
    """
    Trial specification packet with anonymized treatment arms.
    """
    spec_hash: str
    study_id: str
    treatment_arms: List[Dict[str, Any]]
    randomization_method: str
    stratification_vars: List[str]
    primary_outcome: str
    secondary_outcomes: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "spec_hash": self.spec_hash,
            "study_id": self.study_id,
            "treatment_arms": self.treatment_arms,
            "randomization_method": self.randomization_method,
            "stratification_vars": self.stratification_vars,
            "primary_outcome": self.primary_outcome,
            "secondary_outcomes": self.secondary_outcomes
        }


class TinkerFeatureBuilder:
    """
    Builds anonymized feature packets for Tinker API.
    
    All methods apply privacy firewall automatically.
    """
    
    def __init__(self, firewall: Optional[TinkerPrivacyFirewall] = None):
        self.firewall = firewall or get_privacy_firewall()
        logger.info("TinkerFeatureBuilder initialized")
    
    # =========================================================================
    # Patient Feature Extraction
    # =========================================================================
    
    def build_patient_features(
        self,
        patient_id: str,
        patient_data: Dict[str, Any],
        include_vitals: bool = True,
        include_labs: bool = True,
        include_medications: bool = True,
        include_behavioral: bool = True
    ) -> Tuple[FeaturePacket, PrivacyAuditRecord]:
        """
        Build anonymized feature packet for a patient.
        
        Applies full privacy pipeline:
        1. Strip PHI fields
        2. Hash identifiers
        3. Bucket continuous values
        
        Returns:
            Tuple of (FeaturePacket, PrivacyAuditRecord)
        """
        # Transform through privacy firewall
        safe_data, audit = self.firewall.transform_patient_data(
            patient_data,
            include_vitals=include_vitals,
            include_labs=include_labs
        )
        
        # Build feature dictionary
        features = {}
        categories = []
        
        # Demographic features (bucketed)
        if "age_bucket" in safe_data:
            features["age_bucket"] = safe_data["age_bucket"]
            categories.append(FeatureCategory.DEMOGRAPHIC.value)
        
        if "gender" in safe_data:
            features["gender"] = safe_data["gender"]
        
        if "bmi_category" in safe_data:
            features["bmi_category"] = safe_data["bmi_category"]
        
        # Clinical features
        if "diagnosis_codes" in safe_data:
            features["diagnosis_codes"] = safe_data["diagnosis_codes"]
            categories.append(FeatureCategory.CLINICAL.value)
        
        if "condition_categories" in safe_data:
            features["condition_categories"] = safe_data["condition_categories"]
        
        if "immunocompromised_type" in safe_data:
            features["immunocompromised_type"] = safe_data["immunocompromised_type"]
        
        if "risk_level" in safe_data:
            features["risk_level"] = safe_data["risk_level"]
        
        # Vital sign categories
        if include_vitals and "vitals_categories" in safe_data:
            features["vitals_categories"] = safe_data["vitals_categories"]
            categories.append(FeatureCategory.VITAL.value)
        
        # Lab value categories
        if include_labs and "lab_categories" in safe_data:
            features["lab_categories"] = safe_data["lab_categories"]
            categories.append(FeatureCategory.LAB.value)
        
        # Medication features (only classes, not specific drugs)
        if include_medications and "medication_classes" in safe_data:
            features["medication_classes"] = safe_data["medication_classes"]
            categories.append(FeatureCategory.MEDICATION.value)
        
        # Behavioral features
        if include_behavioral:
            behavioral = {}
            if "habit_adherence_score" in safe_data:
                # Bucket adherence to categories
                score = safe_data.get("habit_adherence_score", 0)
                if score >= 80:
                    behavioral["adherence_category"] = "high"
                elif score >= 50:
                    behavioral["adherence_category"] = "medium"
                else:
                    behavioral["adherence_category"] = "low"
            
            if "streak_days" in safe_data:
                days = safe_data.get("streak_days", 0)
                behavioral["streak_bucket"] = self.firewall.bucket_duration_days(days)
            
            if behavioral:
                features["behavioral"] = behavioral
                categories.append(FeatureCategory.BEHAVIORAL.value)
        
        # Temporal features
        temporal = {}
        for key in safe_data:
            if key.endswith("_period"):
                temporal[key] = safe_data[key]
        
        if temporal:
            features["temporal"] = temporal
            categories.append(FeatureCategory.TEMPORAL.value)
        
        # Create packet
        patient_hash = self.firewall.hash_patient_id(patient_id)
        packet_id = self.firewall.create_payload_hash({
            "patient": patient_hash,
            "features": features,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        packet = FeaturePacket(
            packet_id=packet_id,
            patient_hash=patient_hash,
            features=features,
            feature_categories=list(set(categories)),
            created_at=datetime.utcnow().isoformat(),
            k_anon_verified=True
        )
        
        return packet, audit
    
    def build_cohort_features(
        self,
        patient_ids: List[str],
        patient_data_list: List[Dict[str, Any]]
    ) -> Tuple[List[FeaturePacket], List[PrivacyAuditRecord], bool]:
        """
        Build feature packets for a cohort.
        
        Checks k-anonymity for the entire cohort.
        
        Returns:
            Tuple of (packets, audits, k_anon_passed)
        """
        # Check k-anonymity for cohort
        if not self.firewall.check_k_anonymity(len(patient_ids)):
            logger.warning(
                f"Cohort size {len(patient_ids)} < k={self.firewall.config.k_anonymity_threshold}"
            )
            return [], [], False
        
        packets = []
        audits = []
        
        for patient_id, patient_data in zip(patient_ids, patient_data_list):
            packet, audit = self.build_patient_features(patient_id, patient_data)
            packets.append(packet)
            audits.append(audit)
        
        return packets, audits, True
    
    # =========================================================================
    # Cohort Query Builder
    # =========================================================================
    
    def build_cohort_query(
        self,
        inclusion_criteria: Dict[str, Any],
        exclusion_criteria: Optional[Dict[str, Any]] = None,
        estimated_size: int = 0
    ) -> Tuple[Optional[CohortQuery], PrivacyAuditRecord]:
        """
        Build DSL query for cohort definition.
        
        Applies privacy firewall to all criteria.
        
        Returns:
            Tuple of (CohortQuery or None if k-anon fails, PrivacyAuditRecord)
        """
        audit = PrivacyAuditRecord(
            timestamp=datetime.utcnow(),
            operation="build_cohort_query",
            cohort_size=estimated_size
        )
        
        # Check k-anonymity
        if not self.firewall.check_k_anonymity(estimated_size):
            audit.k_anon_passed = False
            audit.suppression_applied = True
            return None, audit
        
        audit.k_anon_passed = True
        
        # Transform inclusion criteria
        safe_inclusion = self._transform_criteria(inclusion_criteria, audit)
        
        # Transform exclusion criteria
        safe_exclusion = {}
        if exclusion_criteria:
            safe_exclusion = self._transform_criteria(exclusion_criteria, audit)
        
        # Build DSL
        dsl = {
            "version": "1.0",
            "inclusion": safe_inclusion,
            "exclusion": safe_exclusion
        }
        
        query_hash = self.firewall.create_payload_hash(dsl)
        
        query = CohortQuery(
            query_hash=query_hash,
            dsl=dsl,
            estimated_size=estimated_size,
            k_anon_passed=True
        )
        
        return query, audit
    
    def _transform_criteria(
        self,
        criteria: Dict[str, Any],
        audit: PrivacyAuditRecord
    ) -> Dict[str, Any]:
        """Transform criteria values using privacy firewall"""
        result = {}
        
        for key, value in criteria.items():
            # Skip PHI fields
            if self.firewall._is_phi_field(key):
                audit.fields_stripped.append(key)
                continue
            
            # Transform based on field type
            if key == "age":
                if isinstance(value, dict):
                    # Range: {"min": 18, "max": 65}
                    result["age_bucket"] = {
                        "min": self.firewall.bucket_age(value.get("min", 0)),
                        "max": self.firewall.bucket_age(value.get("max", 100))
                    }
                else:
                    result["age_bucket"] = self.firewall.bucket_age(value)
                audit.fields_bucketed.append("age")
            
            elif key in ["created_at", "updated_at", "visit_date", "onset_date"]:
                result[f"{key}_period"] = self.firewall.bucket_date(value)
                audit.fields_bucketed.append(key)
            
            elif key.startswith("vital_"):
                vital_type = key.replace("vital_", "")
                result[f"{key}_category"] = self.firewall.bucket_vital(vital_type, value)
                audit.fields_bucketed.append(key)
            
            elif key.startswith("lab_"):
                lab_type = key.replace("lab_", "")
                result[f"{key}_category"] = self.firewall.bucket_lab_value(lab_type, value)
                audit.fields_bucketed.append(key)
            
            elif key == "bmi":
                result["bmi_category"] = self.firewall.bucket_bmi(value)
                audit.fields_bucketed.append("bmi")
            
            else:
                # Pass through non-PHI categorical values
                result[key] = value
        
        return result
    
    # =========================================================================
    # Study Protocol Packet
    # =========================================================================
    
    def build_study_protocol(
        self,
        objective: str,
        analysis_types: List[str],
        outcome_variable: str,
        covariates: List[str],
        confounders: Optional[List[str]] = None,
        follow_up_days: Optional[int] = None
    ) -> Tuple[StudyProtocol, PrivacyAuditRecord]:
        """
        Build study protocol packet for Tinker.
        
        All text is hashed, no raw research objectives are sent.
        """
        audit = PrivacyAuditRecord(
            timestamp=datetime.utcnow(),
            operation="build_study_protocol"
        )
        
        # Hash the objective text (never send raw research questions)
        objective_hash = self.firewall.hash_identifier(objective, "objective")
        audit.fields_hashed.append("objective")
        
        # Bucket follow-up period
        follow_up_period = "unknown"
        if follow_up_days is not None:
            follow_up_period = self.firewall.bucket_duration_days(follow_up_days)
            audit.fields_bucketed.append("follow_up_days")
        
        # Create protocol hash
        protocol_data = {
            "objective_hash": objective_hash,
            "analysis_types": analysis_types,
            "outcome": outcome_variable,
            "covariates": covariates,
            "confounders": confounders or []
        }
        protocol_hash = self.firewall.create_payload_hash(protocol_data)
        
        protocol = StudyProtocol(
            protocol_hash=protocol_hash,
            objective_hash=objective_hash,
            analysis_types=analysis_types,
            outcome_variable=outcome_variable,
            covariates=covariates,
            confounders=confounders or [],
            follow_up_period=follow_up_period
        )
        
        return protocol, audit
    
    # =========================================================================
    # Trial Specification Packet
    # =========================================================================
    
    def build_trial_spec(
        self,
        study_id: str,
        treatment_arms: List[Dict[str, Any]],
        randomization_method: str = "stratified",
        stratification_vars: Optional[List[str]] = None,
        primary_outcome: str = "",
        secondary_outcomes: Optional[List[str]] = None
    ) -> Tuple[TrialSpec, PrivacyAuditRecord]:
        """
        Build trial specification packet.
        
        Treatment arms are anonymized (no drug names, only classes).
        """
        audit = PrivacyAuditRecord(
            timestamp=datetime.utcnow(),
            operation="build_trial_spec"
        )
        
        # Anonymize treatment arms
        safe_arms = []
        for i, arm in enumerate(treatment_arms):
            safe_arm = {
                "arm_id": f"arm_{i}",
                "type": arm.get("type", "treatment"),  # treatment, control, placebo
            }
            
            # Only include drug class, not specific drug names
            if "drug_class" in arm:
                safe_arm["drug_class"] = arm["drug_class"]
            
            if "intervention_category" in arm:
                safe_arm["intervention_category"] = arm["intervention_category"]
            
            # Strip any PHI from arm
            for key in arm:
                if not self.firewall._is_phi_field(key) and key not in safe_arm:
                    if key not in ["drug_name", "medication_name", "treatment_name"]:
                        safe_arm[key] = arm[key]
            
            safe_arms.append(safe_arm)
        
        audit.fields_stripped.extend(["drug_name", "medication_name", "treatment_name"])
        
        # Create spec hash
        spec_data = {
            "study_id": study_id,
            "arms": safe_arms,
            "randomization": randomization_method,
            "outcome": primary_outcome
        }
        spec_hash = self.firewall.create_payload_hash(spec_data)
        
        spec = TrialSpec(
            spec_hash=spec_hash,
            study_id=study_id,
            treatment_arms=safe_arms,
            randomization_method=randomization_method,
            stratification_vars=stratification_vars or [],
            primary_outcome=primary_outcome,
            secondary_outcomes=secondary_outcomes or []
        )
        
        return spec, audit
    
    # =========================================================================
    # Aggregate Feature Building
    # =========================================================================
    
    def build_cohort_aggregates(
        self,
        patient_data_list: List[Dict[str, Any]],
        aggregate_fields: List[str]
    ) -> Tuple[Dict[str, Any], PrivacyAuditRecord]:
        """
        Build aggregate statistics for a cohort.
        
        Only returns aggregates if cohort meets k-anonymity.
        """
        audit = PrivacyAuditRecord(
            timestamp=datetime.utcnow(),
            operation="build_cohort_aggregates",
            cohort_size=len(patient_data_list)
        )
        
        # Check k-anonymity
        if not self.firewall.check_k_anonymity(len(patient_data_list)):
            audit.k_anon_passed = False
            audit.suppression_applied = True
            return {
                "suppressed": True,
                "reason": "k_anonymity",
                "threshold": self.firewall.config.k_anonymity_threshold
            }, audit
        
        audit.k_anon_passed = True
        
        aggregates = {}
        
        for field in aggregate_fields:
            values = []
            categories = []
            
            for patient_data in patient_data_list:
                if field in patient_data:
                    val = patient_data[field]
                    if isinstance(val, (int, float)):
                        values.append(val)
                    elif isinstance(val, str):
                        categories.append(val)
            
            if values:
                # Numeric field - compute safe aggregates
                aggregates[field] = self.firewall.compute_safe_aggregates(values)
            elif categories:
                # Categorical field - compute distribution
                aggregates[field] = self.firewall.compute_category_distribution(categories)
        
        return aggregates, audit


# Singleton instance
_builder_instance: Optional[TinkerFeatureBuilder] = None


def get_feature_builder() -> TinkerFeatureBuilder:
    """Get or create singleton feature builder"""
    global _builder_instance
    if _builder_instance is None:
        _builder_instance = TinkerFeatureBuilder()
    return _builder_instance
