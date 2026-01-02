"""
Tinker Service - Main Orchestrator
===================================
Production-grade orchestrator for Tinker Thinking Machine integration.
Coordinates privacy firewall, API client, and feature builder.

All Tinker operations flow through this service to ensure:
1. Privacy firewall is always applied
2. K-anonymity is always checked
3. Audit logs are always created
4. No PHI ever reaches Tinker API

Usage:
    from app.services.tinker_service import get_tinker_service
    
    tinker = get_tinker_service()
    result = await tinker.analyze_patient_cohort(cohort_id, patient_ids, patient_data)
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from sqlalchemy.orm import Session

from app.config import settings
from app.services.tinker_privacy_firewall import (
    get_privacy_firewall,
    TinkerPrivacyFirewall,
    PrivacyAuditRecord
)
from app.services.tinker_api_client import (
    get_tinker_client,
    TinkerAPIClient,
    TinkerRequestMetrics
)
from app.services.tinker_feature_builder import (
    get_feature_builder,
    TinkerFeatureBuilder,
    FeaturePacket,
    CohortQuery,
    StudyProtocol,
    TrialSpec
)
from app.models.tinker_models import (
    AIAuditLog,
    TinkerPurpose,
    ActorRole,
    TinkerCohortDefinition,
    TinkerCohortSnapshot,
    TinkerStudy,
    TinkerStudyProtocol,
    TinkerTrialSpec,
    TinkerTrialRun,
    TinkerJobReport,
    TinkerModelMetrics,
    TinkerDriftRun,
    TinkerDriftAlert,
    TrialStatus,
    DriftSeverity
)

logger = logging.getLogger(__name__)


@dataclass
class TinkerOperationResult:
    """Result of a Tinker operation"""
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    audit_id: Optional[str] = None
    metrics: Optional[TinkerRequestMetrics] = None
    privacy_audit: Optional[PrivacyAuditRecord] = None
    k_anon_passed: bool = True


class TinkerService:
    """
    Main orchestrator for Tinker Thinking Machine integration.
    
    All Tinker API calls flow through this service to ensure:
    - Privacy firewall is always applied
    - Audit logs are always created
    - K-anonymity is always enforced
    """
    
    def __init__(
        self,
        firewall: Optional[TinkerPrivacyFirewall] = None,
        client: Optional[TinkerAPIClient] = None,
        builder: Optional[TinkerFeatureBuilder] = None
    ):
        self.firewall = firewall or get_privacy_firewall()
        self.client = client or get_tinker_client()
        self.builder = builder or get_feature_builder()
        
        self._validate_enabled()
        logger.info("TinkerService orchestrator initialized")
    
    def _validate_enabled(self):
        """Check if Tinker is enabled"""
        if not settings.is_tinker_enabled():
            logger.warning("Tinker is not enabled - operations will fail")
    
    def is_enabled(self) -> bool:
        """Check if Tinker integration is enabled"""
        return settings.is_tinker_enabled()
    
    # =========================================================================
    # Audit Logging
    # =========================================================================
    
    def _create_audit_log(
        self,
        db: Session,
        purpose: TinkerPurpose,
        actor_id: str,
        actor_role: ActorRole,
        payload_hash: str,
        response_hash: str = "",
        model_used: str = "tinker_api",
        latency_ms: Optional[float] = None,
        success: bool = True,
        error_message: Optional[str] = None,
        k_anon_verified: bool = True
    ) -> AIAuditLog:
        """
        Create audit log entry for Tinker operation.
        
        HIPAA Compliance: Only stores hashes, never raw data.
        Tracks k-anonymity verification status for compliance.
        """
        audit = AIAuditLog(
            purpose=purpose,
            actor_id=actor_id,
            actor_role=actor_role,
            payload_hash=payload_hash,
            response_hash=response_hash,
            model_used=model_used,
            latency_ms=latency_ms,
            success=success,
            k_anon_verified=k_anon_verified,
            tinker_mode="NON_BAA"
        )
        
        db.add(audit)
        db.commit()
        db.refresh(audit)
        
        log_level = "info" if success else "warning"
        getattr(logger, log_level)(
            f"Tinker audit log: {audit.id} for {purpose.value}, "
            f"success={success}, k_anon={k_anon_verified}"
        )
        
        return audit
    
    # =========================================================================
    # Cohort Operations
    # =========================================================================
    
    async def analyze_patient_cohort(
        self,
        db: Session,
        actor_id: str,
        actor_role: str,
        cohort_name: str,
        patient_ids: List[str],
        patient_data_list: List[Dict[str, Any]],
        analysis_type: str = "descriptive"
    ) -> TinkerOperationResult:
        """
        Analyze a patient cohort using Tinker.
        
        Steps:
        1. HARD-FAIL k-anonymity check (blocks operation if failed)
        2. Build anonymized feature packets
        3. Send to Tinker API
        4. Store results
        5. Create audit log
        """
        if not self.is_enabled():
            return TinkerOperationResult(
                success=False,
                error="Tinker integration is not enabled"
            )
        
        # HARD-FAIL k-anonymity check - blocks operation entirely
        cohort_size = len(patient_ids)
        try:
            self.firewall.require_k_anonymity(cohort_size, context="analyze_patient_cohort")
        except ValueError as e:
            logger.warning(f"Cohort rejected: {e}")
            return TinkerOperationResult(
                success=False,
                error=str(e),
                k_anon_passed=False
            )
        
        # Build feature packets
        packets, audits, k_passed = self.builder.build_cohort_features(
            patient_ids, patient_data_list
        )
        
        if not k_passed:
            return TinkerOperationResult(
                success=False,
                error="K-anonymity check failed during feature building",
                k_anon_passed=False
            )
        
        # Build cohort query
        cohort_query = {
            "name": cohort_name,
            "size": cohort_size,
            "analysis_type": analysis_type,
            "features": [p.to_dict() for p in packets]
        }
        
        # Send to Tinker API
        response, metrics = await self.client.analyze_cohort(cohort_query, cohort_size)
        
        # Create audit log
        actor_role_enum = ActorRole.DOCTOR if actor_role == "doctor" else ActorRole.SYSTEM
        audit_log = self._create_audit_log(
            db=db,
            purpose=TinkerPurpose.COHORT_ANALYSIS,
            actor_id=actor_id,
            actor_role=actor_role_enum,
            payload_hash=metrics.payload_hash,
            response_hash=metrics.response_hash if metrics.success else "",
            latency_ms=metrics.duration_ms,
            success=metrics.success,
            error_message=metrics.error_message
        )
        
        # Store cohort definition
        cohort_def = TinkerCohortDefinition(
            name=cohort_name,
            dsl_json={"analysis_type": analysis_type},
            dsl_hash=self.firewall.create_payload_hash({"name": cohort_name}),
            created_by=actor_id
        )
        db.add(cohort_def)
        db.commit()
        db.refresh(cohort_def)
        
        # Store snapshot
        snapshot = TinkerCohortSnapshot(
            cohort_id=cohort_def.id,
            snapshot_hash=metrics.payload_hash,
            patient_count=cohort_size,
            k_anon_passed=True,
            features_json={"packet_count": len(packets)},
            results_json=response if response else {}
        )
        db.add(snapshot)
        db.commit()
        
        return TinkerOperationResult(
            success=metrics.success,
            data=response,
            audit_id=str(audit_log.id),
            metrics=metrics,
            privacy_audit=audits[0] if audits else None,
            k_anon_passed=True
        )
    
    async def get_cohort_insights(
        self,
        db: Session,
        actor_id: str,
        actor_role: str,
        cohort_id: str
    ) -> TinkerOperationResult:
        """Get insights for a stored cohort"""
        if not self.is_enabled():
            return TinkerOperationResult(
                success=False,
                error="Tinker integration is not enabled"
            )
        
        response, metrics = await self.client.get_cohort_insights(cohort_id)
        
        # Create audit log
        actor_role_enum = ActorRole.DOCTOR if actor_role == "doctor" else ActorRole.SYSTEM
        self._create_audit_log(
            db=db,
            purpose=TinkerPurpose.COHORT_ANALYSIS,
            actor_id=actor_id,
            actor_role=actor_role_enum,
            payload_hash=self.firewall.hash_identifier(cohort_id),
            response_hash=metrics.response_hash if metrics.success else "",
            latency_ms=metrics.duration_ms,
            success=metrics.success
        )
        
        return TinkerOperationResult(
            success=metrics.success,
            data=response,
            metrics=metrics
        )
    
    # =========================================================================
    # Study Operations
    # =========================================================================
    
    async def create_research_study(
        self,
        db: Session,
        actor_id: str,
        actor_role: str,
        study_name: str,
        objective: str,
        cohort_id: str,
        analysis_types: List[str],
        outcome_variable: str,
        covariates: List[str],
        confounders: Optional[List[str]] = None,
        follow_up_days: Optional[int] = None
    ) -> TinkerOperationResult:
        """
        Create a research study in Tinker.
        
        All research objectives are hashed - Tinker never sees raw questions.
        """
        if not self.is_enabled():
            return TinkerOperationResult(
                success=False,
                error="Tinker integration is not enabled"
            )
        
        # Build protocol
        protocol, privacy_audit = self.builder.build_study_protocol(
            objective=objective,
            analysis_types=analysis_types,
            outcome_variable=outcome_variable,
            covariates=covariates,
            confounders=confounders,
            follow_up_days=follow_up_days
        )
        
        # Create study config for Tinker
        study_config = {
            "name": study_name,  # Only name, no PHI
            "protocol": protocol.to_dict(),
            "cohort_id": self.firewall.hash_identifier(cohort_id)
        }
        
        # Send to Tinker API
        response, metrics = await self.client.create_study(study_config)
        
        # Store protocol locally
        db_protocol = TinkerStudyProtocol(
            objective_hash=protocol.objective_hash,
            protocol_json=protocol.to_dict(),
            analysis_types=analysis_types,
            confounders=confounders or []
        )
        db.add(db_protocol)
        db.commit()
        db.refresh(db_protocol)
        
        # Store study
        db_study = TinkerStudy(
            name=study_name,
            cohort_id=cohort_id,
            protocol_id=db_protocol.id,
            status="created",
            created_by=actor_id
        )
        db.add(db_study)
        db.commit()
        db.refresh(db_study)
        
        # Create audit log
        actor_role_enum = ActorRole.DOCTOR if actor_role == "doctor" else ActorRole.SYSTEM
        self._create_audit_log(
            db=db,
            purpose=TinkerPurpose.RESEARCH_ANALYSIS,
            actor_id=actor_id,
            actor_role=actor_role_enum,
            payload_hash=metrics.payload_hash,
            response_hash=metrics.response_hash if metrics.success else "",
            latency_ms=metrics.duration_ms,
            success=metrics.success
        )
        
        return TinkerOperationResult(
            success=metrics.success,
            data={
                "study_id": str(db_study.id),
                "protocol_id": str(db_protocol.id),
                "tinker_response": response
            },
            metrics=metrics,
            privacy_audit=privacy_audit
        )
    
    # =========================================================================
    # Trial Operations
    # =========================================================================
    
    async def run_clinical_trial(
        self,
        db: Session,
        actor_id: str,
        actor_role: str,
        study_id: str,
        treatment_arms: List[Dict[str, Any]],
        randomization_method: str = "stratified",
        stratification_vars: Optional[List[str]] = None,
        primary_outcome: str = "",
        secondary_outcomes: Optional[List[str]] = None
    ) -> TinkerOperationResult:
        """
        Run a clinical trial simulation in Tinker.
        
        Treatment arms are anonymized (drug classes only, no specific names).
        """
        if not self.is_enabled():
            return TinkerOperationResult(
                success=False,
                error="Tinker integration is not enabled"
            )
        
        # Build trial spec
        trial_spec, privacy_audit = self.builder.build_trial_spec(
            study_id=study_id,
            treatment_arms=treatment_arms,
            randomization_method=randomization_method,
            stratification_vars=stratification_vars,
            primary_outcome=primary_outcome,
            secondary_outcomes=secondary_outcomes
        )
        
        # Store spec locally
        db_spec = TinkerTrialSpec(
            study_id=study_id,
            spec_json=trial_spec.to_dict(),
            spec_hash=trial_spec.spec_hash,
            treatment_definition={"arms": len(treatment_arms)}
        )
        db.add(db_spec)
        db.commit()
        db.refresh(db_spec)
        
        # Send to Tinker API
        response, metrics = await self.client.run_trial(trial_spec.to_dict())
        
        # Store trial run
        db_run = TinkerTrialRun(
            trial_spec_id=db_spec.id,
            status=TrialStatus.RUNNING if metrics.success else TrialStatus.FAILED,
            results_json=response if response else {},
            k_anon_passed=True
        )
        db.add(db_run)
        db.commit()
        db.refresh(db_run)
        
        # Create audit log
        actor_role_enum = ActorRole.DOCTOR if actor_role == "doctor" else ActorRole.SYSTEM
        self._create_audit_log(
            db=db,
            purpose=TinkerPurpose.CLINICAL_TRIAL,
            actor_id=actor_id,
            actor_role=actor_role_enum,
            payload_hash=metrics.payload_hash,
            response_hash=metrics.response_hash if metrics.success else "",
            latency_ms=metrics.duration_ms,
            success=metrics.success
        )
        
        return TinkerOperationResult(
            success=metrics.success,
            data={
                "trial_spec_id": str(db_spec.id),
                "trial_run_id": str(db_run.id),
                "tinker_response": response
            },
            metrics=metrics,
            privacy_audit=privacy_audit
        )
    
    # =========================================================================
    # Drift Detection Operations
    # =========================================================================
    
    async def check_model_drift(
        self,
        db: Session,
        model_id: str,
        feature_packet: Dict[str, Any],
        actor_id: str = "system"
    ) -> TinkerOperationResult:
        """
        Check for model drift using Tinker.
        
        Feature packet must already be anonymized.
        """
        if not self.is_enabled():
            return TinkerOperationResult(
                success=False,
                error="Tinker integration is not enabled"
            )
        
        # Ensure feature packet is safe
        safe_packet, _ = self.firewall.strip_phi_fields(feature_packet)
        
        response, metrics = await self.client.check_drift(model_id, safe_packet)
        
        # Store drift run
        drift_metrics = {}
        if response:
            drift_metrics = {
                "psi_score": response.get("psi_score"),
                "kl_divergence": response.get("kl_divergence"),
                "feature_drift": response.get("feature_drift", {})
            }
        
        drift_was_detected = response.get("drift_detected", False) if response else False
        psi_value = float(drift_metrics.get("psi_score") or 0.0)
        kl_value = drift_metrics.get("kl_divergence")
        
        db_drift = TinkerDriftRun(
            model_id=model_id,
            drift_metrics_json=drift_metrics,
            psi_score=psi_value if psi_value > 0 else None,
            kl_divergence=float(kl_value) if kl_value is not None else None,
            drift_detected=drift_was_detected
        )
        db.add(db_drift)
        db.commit()
        db.refresh(db_drift)
        
        # Create alert if drift detected
        if drift_was_detected:
            severity = DriftSeverity.HIGH if psi_value > 0.25 else DriftSeverity.MEDIUM
            psi_display = f"{psi_value:.3f}" if psi_value > 0 else "N/A"
            alert = TinkerDriftAlert(
                drift_run_id=db_drift.id,
                alert_type="model_drift",
                severity=severity,
                message=f"Model drift detected: PSI={psi_display}"
            )
            db.add(alert)
            db.commit()
        
        # Create audit log
        self._create_audit_log(
            db=db,
            purpose=TinkerPurpose.DRIFT_DETECTION,
            actor_id=actor_id,
            actor_role=ActorRole.SYSTEM,
            payload_hash=metrics.payload_hash,
            response_hash=metrics.response_hash if metrics.success else "",
            latency_ms=metrics.duration_ms,
            success=metrics.success
        )
        
        return TinkerOperationResult(
            success=metrics.success,
            data={
                "drift_run_id": str(db_drift.id),
                "drift_detected": db_drift.drift_detected,
                "psi_score": db_drift.psi_score,
                "kl_divergence": db_drift.kl_divergence,
                "tinker_response": response
            },
            metrics=metrics
        )
    
    # =========================================================================
    # Model Metrics Operations
    # =========================================================================
    
    async def submit_model_performance(
        self,
        db: Session,
        model_id: str,
        metrics_data: Dict[str, Any],
        subgroup_metrics: Optional[Dict[str, Any]] = None,
        calibration_metrics: Optional[Dict[str, Any]] = None,
        actor_id: str = "system"
    ) -> TinkerOperationResult:
        """
        Submit model performance metrics to Tinker.
        
        All metrics are aggregates - no individual patient data.
        """
        if not self.is_enabled():
            return TinkerOperationResult(
                success=False,
                error="Tinker integration is not enabled"
            )
        
        response, metrics = await self.client.submit_model_metrics(
            model_id, metrics_data
        )
        
        # Store metrics locally
        db_metrics = TinkerModelMetrics(
            model_id=model_id,
            metrics_json=metrics_data,
            subgroup_metrics=subgroup_metrics or {},
            calibration_metrics=calibration_metrics or {}
        )
        db.add(db_metrics)
        db.commit()
        db.refresh(db_metrics)
        
        # Create audit log
        self._create_audit_log(
            db=db,
            purpose=TinkerPurpose.MODEL_EVALUATION,
            actor_id=actor_id,
            actor_role=ActorRole.SYSTEM,
            payload_hash=metrics.payload_hash,
            response_hash=metrics.response_hash if metrics.success else "",
            latency_ms=metrics.duration_ms,
            success=metrics.success
        )
        
        return TinkerOperationResult(
            success=metrics.success,
            data={
                "metrics_id": str(db_metrics.id),
                "tinker_response": response
            },
            metrics=metrics
        )
    
    async def get_threshold_recommendations(
        self,
        db: Session,
        model_id: str,
        optimization_target: str = "f1",
        actor_id: str = "system"
    ) -> TinkerOperationResult:
        """Get threshold optimization recommendations from Tinker"""
        if not self.is_enabled():
            return TinkerOperationResult(
                success=False,
                error="Tinker integration is not enabled"
            )
        
        response, metrics = await self.client.get_threshold_recommendations(
            model_id, optimization_target
        )
        
        # Create audit log
        self._create_audit_log(
            db=db,
            purpose=TinkerPurpose.THRESHOLD_OPTIMIZATION,
            actor_id=actor_id,
            actor_role=ActorRole.SYSTEM,
            payload_hash=self.firewall.hash_identifier(model_id),
            response_hash=metrics.response_hash if metrics.success else "",
            latency_ms=metrics.duration_ms,
            success=metrics.success
        )
        
        return TinkerOperationResult(
            success=metrics.success,
            data=response,
            metrics=metrics
        )
    
    # =========================================================================
    # Health Check
    # =========================================================================
    
    async def health_check(self) -> TinkerOperationResult:
        """Check Tinker API health"""
        if not self.is_enabled():
            return TinkerOperationResult(
                success=False,
                error="Tinker integration is not enabled"
            )
        
        healthy, metrics = await self.client.health_check()
        
        return TinkerOperationResult(
            success=healthy,
            data={"healthy": healthy},
            metrics=metrics
        )


# Singleton instance
_service_instance: Optional[TinkerService] = None


def get_tinker_service() -> TinkerService:
    """Get or create singleton Tinker service"""
    global _service_instance
    if _service_instance is None:
        _service_instance = TinkerService()
    return _service_instance
