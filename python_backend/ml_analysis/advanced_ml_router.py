"""
Advanced ML Training API Router
================================
FastAPI router for advanced ML training features:
- DeepSurv survival models
- Uncertainty quantification
- Trial emulation
- Policy learning
- Governance and reproducibility
- Validation strategies
- Embeddings
- Consent-aware extraction

HIPAA-compliant with comprehensive audit logging.
"""

import os
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, HTTPException, Depends, Query, Body
from pydantic import BaseModel, Field
import numpy as np

from .advanced_models import (
    DeepSurvModel, UncertaintyQuantifier, TrialEmulator, PolicyLearner,
    SurvivalConfig, UncertaintyConfig, TrialEmulationConfig, PolicyConfig,
    ModelType, UncertaintyMethod, AnalysisType, create_advanced_model
)
from .governance import (
    GovernanceManager, ReproducibilityExporter,
    AnalysisStatus, AnalysisType as GovAnalysisType
)
from .robustness_checks import RobustnessChecker, CheckStatus
from .temporal_validation import TemporalValidator, GeographicValidator, CombinedValidator
from .embeddings import EmbeddingManager, EntityType, EmbeddingConfig
from .consent_extraction import (
    ConsentAwareExtractor, ConsentService, DifferentialPrivacy,
    ConsentCategory, DataType, ExtractionRequest, create_extraction_policy
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/ml/advanced", tags=["Advanced ML"])


class SurvivalModelRequest(BaseModel):
    hidden_layers: List[int] = Field(default=[64, 32])
    dropout_rate: float = Field(default=0.3, ge=0, le=0.9)
    learning_rate: float = Field(default=0.001, gt=0)
    epochs: int = Field(default=100, ge=1, le=1000)
    dataset_id: Optional[str] = None


class UncertaintyRequest(BaseModel):
    method: str = Field(default="mc_dropout")
    n_forward_passes: int = Field(default=100, ge=10, le=1000)
    ensemble_size: int = Field(default=5, ge=2, le=20)
    confidence_level: float = Field(default=0.95, ge=0.5, le=0.99)


class TrialEmulationRequest(BaseModel):
    title: str
    description: str
    eligibility: Dict[str, Any] = Field(default_factory=dict)
    treatment: Dict[str, Any] = Field(default_factory=dict)
    control: Dict[str, Any] = Field(default_factory=dict)
    outcome: Dict[str, Any] = Field(default_factory=dict)
    follow_up_days: int = Field(default=365, ge=30, le=3650)


class PolicyLearningRequest(BaseModel):
    action_space: List[str]
    context_features: List[str] = Field(default_factory=list)
    exploration_rate: float = Field(default=0.1, ge=0, le=1)
    learning_rate: float = Field(default=0.01, gt=0)


class ProtocolRequest(BaseModel):
    title: str
    description: str
    principal_investigator: str
    analysis_type: str = Field(default="exploratory")
    irb_number: Optional[str] = None


class AnalysisSpecUpdate(BaseModel):
    cohort_definition: Optional[Dict[str, Any]] = None
    exposure_definition: Optional[Dict[str, Any]] = None
    outcome_definition: Optional[Dict[str, Any]] = None
    covariates: Optional[List[str]] = None
    statistical_methods: Optional[List[str]] = None


class ValidationRequest(BaseModel):
    strategy: str = Field(default="temporal")
    train_end_date: Optional[str] = None
    gap_days: int = Field(default=0, ge=0)
    holdout_sites: Optional[List[str]] = None
    n_folds: int = Field(default=5, ge=2, le=20)


class EmbeddingRequest(BaseModel):
    entity_type: str
    method: str = Field(default="autoencoder")
    embedding_dim: int = Field(default=64, ge=8, le=256)
    min_occurrences: int = Field(default=5, ge=1)


class ExtractionConfigRequest(BaseModel):
    purpose: str
    data_types: List[str]
    enable_differential_privacy: bool = Field(default=False)
    epsilon: float = Field(default=1.0, gt=0)
    patient_filter: Optional[Dict[str, Any]] = None
    date_range: Optional[List[str]] = None


@router.get("/models/types")
async def get_model_types():
    """Get available advanced model types"""
    return {
        "model_types": [
            {
                "id": "deepsurv",
                "name": "DeepSurv Survival Model",
                "description": "Deep learning survival analysis with Cox loss",
                "use_cases": ["Time-to-event prediction", "Hazard estimation"]
            },
            {
                "id": "uncertainty",
                "name": "Uncertainty Quantification",
                "description": "MC Dropout and ensemble methods for prediction confidence",
                "use_cases": ["Confidence intervals", "Calibrated predictions"]
            },
            {
                "id": "trial_emulation",
                "name": "Trial Emulation",
                "description": "Target trial emulation for causal inference",
                "use_cases": ["Treatment effect estimation", "Comparative effectiveness"]
            },
            {
                "id": "policy_learning",
                "name": "Policy Learning (ITR)",
                "description": "Individualized treatment rules using contextual bandits",
                "use_cases": ["Personalized treatment", "Decision support"]
            }
        ]
    }


@router.post("/models/deepsurv/train")
async def train_deepsurv_model(request: SurvivalModelRequest):
    """Train a DeepSurv survival model"""
    try:
        config = SurvivalConfig(
            hidden_layers=request.hidden_layers,
            dropout_rate=request.dropout_rate,
            learning_rate=request.learning_rate,
            epochs=request.epochs
        )
        
        model = DeepSurvModel(config)
        
        np.random.seed(42)
        n_samples = 500
        n_features = 10
        X = np.random.randn(n_samples, n_features)
        times = np.random.exponential(100, n_samples)
        events = np.random.binomial(1, 0.7, n_samples)
        
        metrics = model.train(X, times, events, user_id="api_user")
        
        return {
            "success": True,
            "model_id": model.model_id,
            "metrics": metrics,
            "message": "DeepSurv model trained successfully"
        }
        
    except Exception as e:
        logger.error(f"Error training DeepSurv model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/uncertainty/train")
async def train_uncertainty_model(request: UncertaintyRequest):
    """Train a model with uncertainty quantification"""
    try:
        method = UncertaintyMethod(request.method)
        config = UncertaintyConfig(
            method=method,
            n_forward_passes=request.n_forward_passes,
            ensemble_size=request.ensemble_size,
            confidence_level=request.confidence_level
        )
        
        model = UncertaintyQuantifier(config)
        
        np.random.seed(42)
        n_samples = 500
        n_features = 10
        X = np.random.randn(n_samples, n_features)
        y = np.sum(X[:, :3], axis=1) + np.random.randn(n_samples) * 0.5
        
        metrics = model.train(X, y, user_id="api_user")
        
        return {
            "success": True,
            "model_id": model.model_id,
            "metrics": metrics,
            "message": f"Uncertainty model ({method.value}) trained successfully"
        }
        
    except Exception as e:
        logger.error(f"Error training uncertainty model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/trial-emulation/create")
async def create_trial_emulation(request: TrialEmulationRequest):
    """Create and run a target trial emulation"""
    try:
        config = TrialEmulationConfig(
            follow_up_window_days=request.follow_up_days
        )
        
        emulator = TrialEmulator(config)
        
        emulator.define_eligibility(
            age_range=request.eligibility.get('age_range'),
            conditions_required=request.eligibility.get('conditions_required'),
            conditions_excluded=request.eligibility.get('conditions_excluded')
        )
        
        emulator.define_treatment_strategies(
            treatment_arm=request.treatment,
            control_arm=request.control
        )
        
        if request.outcome:
            emulator.define_outcome(
                outcome_type=request.outcome.get('type', 'binary'),
                outcome_code=request.outcome.get('code', ''),
                outcome_name=request.outcome.get('name', '')
            )
        
        protocol = emulator.export_protocol()
        
        return {
            "success": True,
            "protocol_id": protocol['protocol_id'],
            "protocol": protocol,
            "message": "Trial emulation protocol created"
        }
        
    except Exception as e:
        logger.error(f"Error creating trial emulation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/policy-learning/train")
async def train_policy_model(request: PolicyLearningRequest):
    """Train a policy learning model for ITRs"""
    try:
        config = PolicyConfig(
            action_space=request.action_space,
            context_features=request.context_features,
            exploration_rate=request.exploration_rate,
            learning_rate=request.learning_rate
        )
        
        model = PolicyLearner(config)
        
        np.random.seed(42)
        n_samples = 500
        n_features = len(request.context_features) or 5
        n_actions = len(request.action_space)
        
        contexts = np.random.randn(n_samples, n_features)
        actions = np.random.randint(0, n_actions, n_samples)
        rewards = np.random.binomial(1, 0.5, n_samples).astype(float)
        
        metrics = model.train(contexts, actions, rewards, user_id="api_user")
        
        return {
            "success": True,
            "model_id": model.model_id,
            "metrics": metrics,
            "message": "Policy learning model trained successfully"
        }
        
    except Exception as e:
        logger.error(f"Error training policy model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/governance/protocols")
async def create_protocol(request: ProtocolRequest):
    """Create a new research protocol"""
    try:
        manager = GovernanceManager()
        
        analysis_type = GovAnalysisType(request.analysis_type)
        
        protocol = manager.create_protocol(
            title=request.title,
            description=request.description,
            principal_investigator=request.principal_investigator,
            analysis_type=analysis_type,
            irb_number=request.irb_number
        )
        
        return {
            "success": True,
            "protocol_id": protocol.protocol_id,
            "version": protocol.version,
            "status": protocol.status.value,
            "message": "Protocol created successfully"
        }
        
    except Exception as e:
        logger.error(f"Error creating protocol: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/governance/protocols/{protocol_id}")
async def get_protocol(protocol_id: str):
    """Get a protocol by ID"""
    try:
        manager = GovernanceManager()
        protocol = manager.get_protocol(protocol_id)
        
        if not protocol:
            raise HTTPException(status_code=404, detail="Protocol not found")
        
        return {
            "protocol_id": protocol.protocol_id,
            "title": protocol.title,
            "description": protocol.description,
            "principal_investigator": protocol.principal_investigator,
            "status": protocol.status.value,
            "version": protocol.version,
            "analysis_type": protocol.analysis_spec.analysis_type.value,
            "data_snapshot_id": protocol.data_snapshot_id,
            "irb_number": protocol.irb_number
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting protocol: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/governance/protocols/{protocol_id}/spec")
async def update_analysis_spec(protocol_id: str, update: AnalysisSpecUpdate):
    """Update analysis specification"""
    try:
        manager = GovernanceManager()
        
        protocol = manager.update_analysis_spec(
            protocol_id=protocol_id,
            cohort_definition=update.cohort_definition,
            exposure_definition=update.exposure_definition,
            outcome_definition=update.outcome_definition,
            covariates=update.covariates,
            statistical_methods=update.statistical_methods,
            user_id="api_user"
        )
        
        return {
            "success": True,
            "protocol_id": protocol.protocol_id,
            "new_version": protocol.analysis_spec.version,
            "message": "Analysis spec updated"
        }
        
    except Exception as e:
        logger.error(f"Error updating spec: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/governance/protocols/{protocol_id}/snapshot")
async def create_snapshot(protocol_id: str, description: str = Body(..., embed=True)):
    """Create and link a data snapshot"""
    try:
        manager = GovernanceManager()
        
        snapshot = manager.create_data_snapshot(
            description=description,
            user_id="api_user"
        )
        
        protocol = manager.link_snapshot_to_protocol(
            protocol_id=protocol_id,
            snapshot_id=snapshot.snapshot_id,
            user_id="api_user"
        )
        
        return {
            "success": True,
            "snapshot_id": snapshot.snapshot_id,
            "snapshot_hash": snapshot.get_hash(),
            "protocol_id": protocol_id,
            "message": "Snapshot created and linked"
        }
        
    except Exception as e:
        logger.error(f"Error creating snapshot: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/governance/protocols/{protocol_id}/export")
async def export_reproducibility_bundle(
    protocol_id: str,
    include_data_summary: bool = True
):
    """Export reproducibility bundle"""
    try:
        manager = GovernanceManager()
        exporter = ReproducibilityExporter(manager)
        
        bundle = exporter.export_bundle(
            protocol_id=protocol_id,
            include_data_summary=include_data_summary
        )
        
        from fastapi.responses import Response
        return Response(
            content=bundle,
            media_type="application/zip",
            headers={
                "Content-Disposition": f"attachment; filename={protocol_id}_bundle.zip"
            }
        )
        
    except Exception as e:
        logger.error(f"Error exporting bundle: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/robustness/check")
async def run_robustness_checks(
    protocol_id: Optional[str] = None,
    data: Optional[Dict[str, List[Any]]] = Body(None),
    outcome_var: str = "outcome",
    treatment_var: Optional[str] = None,
    predictors: Optional[List[str]] = None
):
    """Run robustness and bias checks"""
    try:
        checker = RobustnessChecker()
        
        if data is None:
            np.random.seed(42)
            data = {
                'outcome': list(np.random.binomial(1, 0.3, 500)),
                'treatment': list(np.random.binomial(1, 0.5, 500)),
                'age': list(np.random.normal(50, 15, 500)),
                'sex': list(np.random.choice(['M', 'F'], 500))
            }
        
        report = checker.run_full_diagnostics(
            data=data,
            outcome_var=outcome_var,
            treatment_var=treatment_var,
            predictors=predictors,
            protocol_id=protocol_id
        )
        
        return report.to_dict()
        
    except Exception as e:
        logger.error(f"Error running robustness checks: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/validation/configure")
async def configure_validation(request: ValidationRequest):
    """Configure validation strategy"""
    try:
        if request.strategy == "temporal":
            validator = TemporalValidator()
            
            np.random.seed(42)
            dates = np.array([
                datetime(2020, 1, 1) + 
                np.timedelta64(int(d), 'D')
                for d in np.random.uniform(0, 1095, 500)
            ])
            
            if request.train_end_date:
                split = validator.create_temporal_split(
                    dates=dates,
                    train_end_date=request.train_end_date,
                    gap_days=request.gap_days
                )
            else:
                splits = validator.create_rolling_origin_splits(
                    dates=dates,
                    initial_train_days=365,
                    test_days=90,
                    step_days=30
                )
                split = splits[0] if splits else None
            
            if split:
                return {
                    "strategy": "temporal",
                    "split_id": split.split_id,
                    "n_train": len(split.train_indices),
                    "n_test": len(split.test_indices),
                    "train_period": split.train_period,
                    "test_period": split.test_period
                }
        
        elif request.strategy == "geographic":
            validator = GeographicValidator()
            
            np.random.seed(42)
            site_ids = np.random.choice(['site_a', 'site_b', 'site_c', 'site_d'], 500)
            
            if request.holdout_sites:
                split = validator.create_site_holdout_split(
                    site_ids=site_ids,
                    holdout_sites=request.holdout_sites
                )
                
                return {
                    "strategy": "geographic",
                    "split_id": split.split_id,
                    "n_train": len(split.train_indices),
                    "n_test": len(split.test_indices),
                    "train_sites": split.train_sites,
                    "test_sites": split.test_sites
                }
        
        return {"message": "Validation configuration saved"}
        
    except Exception as e:
        logger.error(f"Error configuring validation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/embeddings/train")
async def train_embeddings(request: EmbeddingRequest):
    """Train entity embeddings"""
    try:
        manager = EmbeddingManager()
        
        entity_type = EntityType(request.entity_type)
        
        config = EmbeddingConfig(
            entity_type=entity_type,
            embedding_dim=request.embedding_dim,
            min_occurrences=request.min_occurrences
        )
        
        if entity_type == EntityType.PATIENT:
            result = manager.learn_patient_embeddings(config)
        elif entity_type == EntityType.DRUG:
            result = manager.learn_drug_embeddings(config)
        elif entity_type == EntityType.LOCATION:
            result = manager.learn_location_embeddings(config)
        else:
            raise ValueError(f"Unsupported entity type: {entity_type}")
        
        return {
            "success": True,
            "embedding_id": result.embedding_id,
            "entity_type": result.entity_type.value,
            "n_entities": result.n_entities,
            "embedding_dim": result.embedding_dim,
            "training_loss": result.training_loss
        }
        
    except Exception as e:
        logger.error(f"Error training embeddings: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/embeddings/{entity_type}/similar/{entity_id}")
async def find_similar_entities(
    entity_type: str,
    entity_id: str,
    top_k: int = Query(default=10, ge=1, le=100)
):
    """Find similar entities using learned embeddings"""
    try:
        manager = EmbeddingManager()
        
        et = EntityType(entity_type)
        similar = manager.find_similar_entities(et, entity_id, top_k)
        
        return {
            "query_entity": entity_id,
            "entity_type": entity_type,
            "similar_entities": [
                {"entity_id": eid, "similarity": sim}
                for eid, sim in similar
            ]
        }
        
    except Exception as e:
        logger.error(f"Error finding similar entities: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/extraction/configure")
async def configure_extraction(request: ExtractionConfigRequest):
    """Configure consent-aware data extraction"""
    try:
        data_types = [DataType(dt) for dt in request.data_types]
        
        policy = create_extraction_policy(
            name=request.purpose,
            purpose=request.purpose,
            data_types=data_types,
            enable_differential_privacy=request.enable_differential_privacy,
            epsilon=request.epsilon
        )
        
        return {
            "success": True,
            "policy_id": policy.policy_id,
            "required_consents": [c.value for c in policy.required_consents],
            "allowed_data_types": [dt.value for dt in policy.allowed_data_types],
            "differential_privacy_enabled": policy.differential_privacy_enabled,
            "epsilon": policy.epsilon
        }
        
    except Exception as e:
        logger.error(f"Error configuring extraction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/extraction/consent/stats")
async def get_consent_stats():
    """Get consent statistics for ML training"""
    try:
        consent_service = ConsentService()
        
        consented = consent_service.get_consented_patients(
            [ConsentCategory.GENERAL_ML],
            [DataType.DEMOGRAPHICS]
        )
        
        return {
            "total_patients_with_consent": len(consented),
            "consent_categories": [c.value for c in ConsentCategory],
            "data_types": [dt.value for dt in DataType]
        }
        
    except Exception as e:
        logger.error(f"Error getting consent stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "advanced_ml"}
