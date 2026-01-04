"""
Research Center Genius Features (E.5-E.8)
=========================================
Advanced research capabilities for the Research Center.

E.5: Auto preregistration - Tinker drafts prereg outline from protocol
E.6: Bias checklist - confounding, selection bias, immortal time bias warnings
E.7: Sensitivity suite - negative controls, placebo outcome suggestions
E.8: Exportable study bundle - ZIP with CohortDSL, Protocol, Trial spec, metrics, hashes
"""

import hashlib
import json
import logging
import io
import zipfile
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum

from app.services.tinker_client import call_tinker_async
from app.services.privacy_firewall import TinkerPurpose

logger = logging.getLogger(__name__)


class BiasType(str, Enum):
    """Types of bias that can affect research studies"""
    CONFOUNDING = "confounding"
    SELECTION = "selection"
    IMMORTAL_TIME = "immortal_time"
    INFORMATION = "information"
    ATTRITION = "attrition"
    MEASUREMENT = "measurement"
    PUBLICATION = "publication"


class SensitivityCheckType(str, Enum):
    """Types of sensitivity analyses"""
    NEGATIVE_CONTROL_EXPOSURE = "negative_control_exposure"
    NEGATIVE_CONTROL_OUTCOME = "negative_control_outcome"
    PLACEBO_OUTCOME = "placebo_outcome"
    DOSE_RESPONSE = "dose_response"
    SUBGROUP = "subgroup"
    MISSING_DATA = "missing_data"


@dataclass
class BiasWarning:
    """A bias warning with severity and mitigation suggestions"""
    bias_type: BiasType
    severity: str
    description: str
    affected_variables: List[str]
    mitigation_strategies: List[str]


@dataclass
class SensitivityCheck:
    """A sensitivity analysis recommendation"""
    check_type: SensitivityCheckType
    name: str
    description: str
    suggested_variables: List[str]
    rationale: str


@dataclass
class PreregistrationOutline:
    """Auto-generated preregistration outline"""
    title: str
    hypotheses: List[str]
    study_design: str
    sample_size_justification: str
    primary_outcomes: List[str]
    secondary_outcomes: List[str]
    analysis_plan: str
    exclusion_criteria: List[str]
    timeline: str
    generated_at: str
    protocol_hash: str


@dataclass
class StudyBundle:
    """Exportable study bundle with all artifacts"""
    study_id: str
    study_name: str
    cohort_dsl: Dict[str, Any]
    protocol: Dict[str, Any]
    trial_spec: Optional[Dict[str, Any]]
    metrics: Dict[str, Any]
    reproducibility_hashes: Dict[str, str]
    created_at: str


class GeniusResearchService:
    """
    E.5-E.8: Research Center Genius Features
    
    Provides advanced research capabilities while maintaining HIPAA compliance.
    All outputs are safe for Tinker API (no PHI).
    """
    
    def __init__(self):
        logger.info("GeniusResearchService initialized")
    
    def _compute_hash(self, data: Any) -> str:
        """Compute SHA256 hash of data for reproducibility"""
        if isinstance(data, dict):
            data_str = json.dumps(data, sort_keys=True, default=str)
        else:
            data_str = str(data)
        return hashlib.sha256(data_str.encode('utf-8')).hexdigest()
    
    async def generate_preregistration(
        self,
        protocol: Dict[str, Any],
        actor_role: str = "doctor"
    ) -> Tuple[PreregistrationOutline, bool]:
        """
        E.5: Auto preregistration - Generate preregistration outline from protocol.
        
        Uses Tinker to draft a structured preregistration based on the study protocol.
        
        Args:
            protocol: Study protocol dictionary
            actor_role: Role of the requesting user
            
        Returns:
            Tuple of (PreregistrationOutline, success)
        """
        protocol_hash = self._compute_hash(protocol)
        
        payload = {
            "objective": protocol.get("objective", ""),
            "schema_summary": json.dumps(protocol.get("schema_summary", {})),
            "cohort_size_range": protocol.get("cohort_size_range", "50-200"),
            "analysis_types_available": protocol.get("analysis_types", []),
        }
        
        response, success = await call_tinker_async(
            purpose=TinkerPurpose.STUDY_PROTOCOL.value,
            payload=payload,
            actor_role=actor_role
        )
        
        if not success:
            return self._get_default_preregistration(protocol_hash), False
        
        prereg = PreregistrationOutline(
            title=response.get("title", f"Study: {protocol.get('objective', 'Untitled')[:50]}"),
            hypotheses=response.get("hypotheses", [
                "Primary hypothesis to be specified based on protocol objectives"
            ]),
            study_design=response.get("study_design", "Retrospective cohort study"),
            sample_size_justification=response.get(
                "sample_size_justification",
                "Sample size determined by available data meeting inclusion criteria"
            ),
            primary_outcomes=response.get("primary_outcomes", ["Primary outcome TBD"]),
            secondary_outcomes=response.get("secondary_outcomes", ["Secondary outcomes TBD"]),
            analysis_plan=response.get(
                "analysis_plan",
                "Statistical analysis plan to include descriptive statistics and regression models"
            ),
            exclusion_criteria=response.get("exclusion_criteria", ["Exclusion criteria TBD"]),
            timeline=response.get("timeline", "Study timeline TBD"),
            generated_at=datetime.utcnow().isoformat(),
            protocol_hash=protocol_hash
        )
        
        logger.info(f"Generated preregistration outline: {prereg.title}")
        return prereg, True
    
    def _get_default_preregistration(self, protocol_hash: str) -> PreregistrationOutline:
        """Return default preregistration when Tinker unavailable"""
        return PreregistrationOutline(
            title="Study Preregistration (Template)",
            hypotheses=["Primary hypothesis to be defined"],
            study_design="Retrospective cohort study",
            sample_size_justification="Sample size determined by available data",
            primary_outcomes=["Primary outcome measure"],
            secondary_outcomes=["Secondary outcome measures"],
            analysis_plan="Statistical analysis to include appropriate methods",
            exclusion_criteria=["Standard exclusion criteria"],
            timeline="Timeline to be determined",
            generated_at=datetime.utcnow().isoformat(),
            protocol_hash=protocol_hash
        )
    
    def generate_bias_checklist(
        self,
        protocol: Dict[str, Any],
        cohort_definition: Dict[str, Any]
    ) -> List[BiasWarning]:
        """
        E.6: Bias checklist - Identify potential biases in study design.
        
        Analyzes the protocol and cohort definition to identify:
        - Confounding bias
        - Selection bias
        - Immortal time bias
        - Other methodological concerns
        
        Args:
            protocol: Study protocol dictionary
            cohort_definition: Cohort definition filters
            
        Returns:
            List of BiasWarning objects
        """
        warnings = []
        filters = cohort_definition.get("filters", [])
        objective = protocol.get("objective", "")
        
        warnings.append(BiasWarning(
            bias_type=BiasType.CONFOUNDING,
            severity="medium",
            description="Unmeasured confounders may affect treatment-outcome relationships",
            affected_variables=["age_bucket", "condition_codes", "risk_bucket"],
            mitigation_strategies=[
                "Use propensity score matching/weighting",
                "Perform sensitivity analysis for unmeasured confounding",
                "Include additional covariates if available",
                "Consider instrumental variable analysis"
            ]
        ))
        
        time_filters = [f for f in filters if "period" in f.get("field", "") or "date" in f.get("field", "")]
        if time_filters:
            warnings.append(BiasWarning(
                bias_type=BiasType.SELECTION,
                severity="high" if len(time_filters) > 1 else "medium",
                description="Time-based cohort selection may introduce selection bias",
                affected_variables=[f.get("field", "") for f in time_filters],
                mitigation_strategies=[
                    "Ensure consistent time window for all cohort members",
                    "Consider time-varying exposure analysis",
                    "Validate cohort entry criteria",
                    "Perform sensitivity analysis with different time windows"
                ]
            ))
        
        if any("treatment" in objective.lower() or "medication" in objective.lower() 
               for _ in [1]):
            warnings.append(BiasWarning(
                bias_type=BiasType.IMMORTAL_TIME,
                severity="high",
                description="Time between cohort entry and treatment initiation may create immortal time bias",
                affected_variables=["treatment_start_period", "enrollment_period"],
                mitigation_strategies=[
                    "Use time-to-event analysis with proper time origin",
                    "Align index date with treatment initiation",
                    "Consider landmark analysis",
                    "Use time-varying treatment indicator"
                ]
            ))
        
        if any("outcome" in f.get("field", "").lower() for f in filters):
            warnings.append(BiasWarning(
                bias_type=BiasType.INFORMATION,
                severity="medium",
                description="Outcome ascertainment may vary across exposure groups",
                affected_variables=["outcome_category"],
                mitigation_strategies=[
                    "Validate outcome definitions consistently",
                    "Use objective outcome measures when possible",
                    "Perform sensitivity analysis with different outcome definitions"
                ]
            ))
        
        warnings.append(BiasWarning(
            bias_type=BiasType.ATTRITION,
            severity="low",
            description="Loss to follow-up may differ between groups",
            affected_variables=["follow_up_period", "engagement_bucket"],
            mitigation_strategies=[
                "Document and compare attrition rates",
                "Use intention-to-treat analysis",
                "Perform multiple imputation for missing outcomes",
                "Conduct sensitivity analysis for missing data"
            ]
        ))
        
        logger.info(f"Generated {len(warnings)} bias warnings for study")
        return warnings
    
    def generate_sensitivity_suite(
        self,
        protocol: Dict[str, Any],
        cohort_definition: Dict[str, Any]
    ) -> List[SensitivityCheck]:
        """
        E.7: Sensitivity suite - Generate sensitivity analysis recommendations.
        
        Suggests negative controls, placebo outcomes, and other sensitivity checks.
        
        Args:
            protocol: Study protocol dictionary
            cohort_definition: Cohort definition filters
            
        Returns:
            List of SensitivityCheck recommendations
        """
        checks = []
        
        checks.append(SensitivityCheck(
            check_type=SensitivityCheckType.NEGATIVE_CONTROL_EXPOSURE,
            name="Negative Control Exposure",
            description="Test an exposure that should NOT affect the outcome",
            suggested_variables=["unrelated_medication_category", "unrelated_procedure_category"],
            rationale="If negative control shows effect, suggests residual confounding"
        ))
        
        checks.append(SensitivityCheck(
            check_type=SensitivityCheckType.NEGATIVE_CONTROL_OUTCOME,
            name="Negative Control Outcome",
            description="Test an outcome that should NOT be affected by the exposure",
            suggested_variables=["unrelated_diagnosis_category", "administrative_outcome"],
            rationale="If exposure affects negative control outcome, suggests bias or confounding"
        ))
        
        checks.append(SensitivityCheck(
            check_type=SensitivityCheckType.PLACEBO_OUTCOME,
            name="Placebo Outcome Analysis",
            description="Analyze outcomes in a pre-exposure period",
            suggested_variables=["pre_exposure_outcome_rate"],
            rationale="Pre-exposure outcomes should be similar between groups if properly matched"
        ))
        
        checks.append(SensitivityCheck(
            check_type=SensitivityCheckType.DOSE_RESPONSE,
            name="Dose-Response Analysis",
            description="Evaluate if effect varies with exposure intensity",
            suggested_variables=["dose_category", "duration_bucket", "frequency_bucket"],
            rationale="True causal effects often show dose-response relationship"
        ))
        
        age_filters = [f for f in cohort_definition.get("filters", []) 
                      if "age" in f.get("field", "").lower() or "risk" in f.get("field", "").lower()]
        if age_filters:
            checks.append(SensitivityCheck(
                check_type=SensitivityCheckType.SUBGROUP,
                name="Subgroup Analysis",
                description="Evaluate effect heterogeneity across key subgroups",
                suggested_variables=["age_bucket", "risk_bucket", "condition_codes"],
                rationale="Effect may differ across patient subgroups"
            ))
        
        checks.append(SensitivityCheck(
            check_type=SensitivityCheckType.MISSING_DATA,
            name="Missing Data Sensitivity",
            description="Evaluate impact of different missing data assumptions",
            suggested_variables=["missingness_bucket"],
            rationale="Results should be robust to reasonable missing data assumptions"
        ))
        
        logger.info(f"Generated {len(checks)} sensitivity analysis recommendations")
        return checks
    
    def create_study_bundle(
        self,
        study_id: str,
        study_name: str,
        cohort_dsl: Dict[str, Any],
        protocol: Dict[str, Any],
        trial_spec: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, Any]] = None
    ) -> Tuple[bytes, StudyBundle]:
        """
        E.8: Exportable study bundle - Create ZIP with all study artifacts.
        
        Creates a reproducible research package containing:
        - CohortDSL JSON
        - Protocol JSON
        - Trial spec JSON (if applicable)
        - Metrics summary
        - Reproducibility hashes
        
        Args:
            study_id: Unique study identifier
            study_name: Human-readable study name
            cohort_dsl: Cohort definition DSL
            protocol: Study protocol
            trial_spec: Optional trial specification
            metrics: Optional study metrics
            
        Returns:
            Tuple of (ZIP file bytes, StudyBundle metadata)
        """
        reproducibility_hashes = {
            "cohort_dsl": self._compute_hash(cohort_dsl),
            "protocol": self._compute_hash(protocol),
            "bundle_created": datetime.utcnow().isoformat()
        }
        
        if trial_spec:
            reproducibility_hashes["trial_spec"] = self._compute_hash(trial_spec)
        if metrics:
            reproducibility_hashes["metrics"] = self._compute_hash(metrics)
        
        bundle = StudyBundle(
            study_id=study_id,
            study_name=study_name,
            cohort_dsl=cohort_dsl,
            protocol=protocol,
            trial_spec=trial_spec,
            metrics=metrics or {},
            reproducibility_hashes=reproducibility_hashes,
            created_at=datetime.utcnow().isoformat()
        )
        
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(
                "cohort_definition.json",
                json.dumps(cohort_dsl, indent=2, default=str)
            )
            
            zf.writestr(
                "protocol.json",
                json.dumps(protocol, indent=2, default=str)
            )
            
            if trial_spec:
                zf.writestr(
                    "trial_spec.json",
                    json.dumps(trial_spec, indent=2, default=str)
                )
            
            if metrics:
                zf.writestr(
                    "metrics.json",
                    json.dumps(metrics, indent=2, default=str)
                )
            
            zf.writestr(
                "reproducibility_hashes.json",
                json.dumps(reproducibility_hashes, indent=2)
            )
            
            manifest = {
                "study_id": study_id,
                "study_name": study_name,
                "created_at": bundle.created_at,
                "files": [
                    "cohort_definition.json",
                    "protocol.json",
                    "trial_spec.json" if trial_spec else None,
                    "metrics.json" if metrics else None,
                    "reproducibility_hashes.json",
                    "manifest.json"
                ],
                "version": "1.0.0",
                "format": "followup-ai-study-bundle"
            }
            manifest["files"] = [f for f in manifest["files"] if f]
            
            zf.writestr(
                "manifest.json",
                json.dumps(manifest, indent=2)
            )
        
        zip_bytes = zip_buffer.getvalue()
        
        logger.info(f"Created study bundle: {study_name} ({len(zip_bytes)} bytes)")
        return zip_bytes, bundle


_genius_research_service: Optional[GeniusResearchService] = None


def get_genius_research_service() -> GeniusResearchService:
    """Get or create singleton GeniusResearchService"""
    global _genius_research_service
    if _genius_research_service is None:
        _genius_research_service = GeniusResearchService()
    return _genius_research_service
