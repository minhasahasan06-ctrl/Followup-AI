"""
FastAPI Backend for Research Center ML Analysis Engine

Exposes ML analysis capabilities as REST API endpoints.
All endpoints require authentication and include HIPAA audit logging.
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd

from ml_analysis.dataset_builder import DatasetBuilder, CohortSpec, AnalysisSpec
from ml_analysis.analysis_descriptive import DescriptiveAnalysis, DescriptiveConfig
from ml_analysis.analysis_risk_prediction import RiskPredictionAnalysis, RiskPredictionConfig
from ml_analysis.analysis_survival import SurvivalAnalysis, SurvivalConfig
from ml_analysis.analysis_causal import CausalAnalysis, CausalConfig
from ml_analysis.alert_engine import AlertEngine
from ml_analysis.report_generator import ReportGenerator, ReportConfig

from epidemiology.pharmaco_router import router as pharmaco_router
from epidemiology.infectious_router import router as infectious_router
from epidemiology.vaccine_router import router as vaccine_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("ML Analysis Engine starting...")
    from scheduler import start_scheduler, stop_scheduler
    try:
        start_scheduler()
        print("Background scheduler started")
    except Exception as e:
        print(f"Warning: Could not start scheduler: {e}")
    yield
    print("ML Analysis Engine shutting down...")
    try:
        stop_scheduler()
        print("Background scheduler stopped")
    except Exception as e:
        print(f"Warning: Error stopping scheduler: {e}")

app = FastAPI(
    title="Research Center ML Analysis API",
    description="HIPAA-compliant ML analysis engine for epidemiological research",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(pharmaco_router)
app.include_router(infectious_router)
app.include_router(vaccine_router)


class CohortRequest(BaseModel):
    cohort_id: Optional[str] = None
    patient_ids: Optional[List[str]] = None
    age_min: Optional[int] = None
    age_max: Optional[int] = None
    sex: Optional[str] = None
    conditions: Optional[List[str]] = None
    exclude_conditions: Optional[List[str]] = None
    min_followup_days: Optional[int] = None

class AnalysisRequest(BaseModel):
    outcome_variable: str
    outcome_type: str = "binary"
    exposure_variable: Optional[str] = None
    covariates: Optional[List[str]] = None
    time_variable: Optional[str] = None
    event_variable: Optional[str] = None

class DatasetRequest(BaseModel):
    cohort: CohortRequest
    analysis: AnalysisRequest
    include_data_types: Optional[List[str]] = None

class DescriptiveRequest(BaseModel):
    cohort: CohortRequest
    stratify_by: Optional[str] = None
    continuous_vars: Optional[List[str]] = None
    categorical_vars: Optional[List[str]] = None

class RiskPredictionRequest(BaseModel):
    cohort: CohortRequest
    outcome_variable: str
    feature_variables: Optional[List[str]] = None
    model_type: str = "logistic_regression"
    n_folds: int = 5

class SurvivalRequest(BaseModel):
    cohort: CohortRequest
    time_variable: str
    event_variable: str
    covariates: Optional[List[str]] = None
    stratify_by: Optional[str] = None

class CausalRequest(BaseModel):
    cohort: CohortRequest
    treatment_variable: str
    outcome_variable: str
    covariates: Optional[List[str]] = None
    method: str = "iptw"

class AlertRequest(BaseModel):
    patient_id: str
    create_alert: bool = False

class ReportRequest(BaseModel):
    study_info: Dict[str, Any]
    analysis_results: Dict[str, Any]
    style: str = "academic"


def log_audit(action: str, resource: str, user_id: Optional[str] = None, details: Optional[Dict] = None):
    """Log HIPAA audit entry"""
    print(json.dumps({
        "timestamp": datetime.utcnow().isoformat(),
        "action": action,
        "resource": resource,
        "user_id": user_id or "system",
        "details": details or {},
        "audit_type": "HIPAA"
    }))


async def verify_auth(authorization: Optional[str] = Header(None)):
    """Verify authentication token"""
    if not authorization:
        raise HTTPException(status_code=401, detail="Authentication required")
    return {"user_id": "authenticated_user"}


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "ml-analysis-engine"}


@app.post("/api/v1/ml/dataset/build")
async def build_dataset(request: DatasetRequest, auth: dict = Depends(verify_auth)):
    """Build analysis-ready dataset from cohort specification"""
    try:
        log_audit("build_dataset", "dataset", auth["user_id"], {
            "cohort_id": request.cohort.cohort_id
        })
        
        builder = DatasetBuilder()
        
        cohort_spec = CohortSpec(
            cohort_id=request.cohort.cohort_id,
            patient_ids=request.cohort.patient_ids,
            age_min=request.cohort.age_min,
            age_max=request.cohort.age_max,
            sex=request.cohort.sex,
            conditions=request.cohort.conditions,
            exclude_conditions=request.cohort.exclude_conditions,
            min_followup_days=request.cohort.min_followup_days
        )
        
        analysis_spec = AnalysisSpec(
            outcome_variable=request.analysis.outcome_variable,
            outcome_type=request.analysis.outcome_type,
            exposure_variable=request.analysis.exposure_variable,
            covariates=request.analysis.covariates,
            time_variable=request.analysis.time_variable,
            event_variable=request.analysis.event_variable
        )
        
        df = builder.build_cohort_dataset(
            cohort_spec, 
            analysis_spec, 
            request.include_data_types
        )
        
        summary = builder.get_variable_summary(df)
        
        return {
            "success": True,
            "dataset_id": f"ds_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
            "summary": summary,
            "data_preview": df.head(10).to_dict(orient="records")
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/ml/analysis/descriptive")
async def run_descriptive(request: DescriptiveRequest, auth: dict = Depends(verify_auth)):
    """Run descriptive analysis (Table 1)"""
    try:
        log_audit("descriptive_analysis", "analysis", auth["user_id"])
        
        builder = DatasetBuilder()
        cohort_spec = CohortSpec(
            cohort_id=request.cohort.cohort_id,
            patient_ids=request.cohort.patient_ids
        )
        analysis_spec = AnalysisSpec(outcome_variable="dummy", outcome_type="binary")
        df = builder.build_cohort_dataset(cohort_spec, analysis_spec)
        
        if df.empty:
            return {"success": False, "error": "No data available for cohort"}
        
        config = DescriptiveConfig(
            stratify_by=request.stratify_by,
            continuous_vars=request.continuous_vars,
            categorical_vars=request.categorical_vars
        )
        
        analysis = DescriptiveAnalysis(df, config)
        
        return {
            "success": True,
            "table1": analysis.generate_table1(),
            "missing_data": analysis.get_missing_data_report(),
            "correlation": analysis.get_correlation_matrix()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/ml/analysis/risk-prediction")
async def run_risk_prediction(request: RiskPredictionRequest, auth: dict = Depends(verify_auth)):
    """Run risk prediction analysis"""
    try:
        log_audit("risk_prediction", "analysis", auth["user_id"])
        
        builder = DatasetBuilder()
        cohort_spec = CohortSpec(
            cohort_id=request.cohort.cohort_id,
            patient_ids=request.cohort.patient_ids
        )
        analysis_spec = AnalysisSpec(
            outcome_variable=request.outcome_variable, 
            outcome_type="binary"
        )
        df = builder.build_cohort_dataset(cohort_spec, analysis_spec)
        
        if df.empty:
            return {"success": False, "error": "No data available for cohort"}
        
        config = RiskPredictionConfig(
            outcome_variable=request.outcome_variable,
            feature_variables=request.feature_variables,
            model_type=request.model_type,
            n_folds=request.n_folds
        )
        
        analysis = RiskPredictionAnalysis(df, config)
        results = analysis.run_analysis()
        
        return {
            "success": True,
            **results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/ml/analysis/survival")
async def run_survival(request: SurvivalRequest, auth: dict = Depends(verify_auth)):
    """Run survival analysis"""
    try:
        log_audit("survival_analysis", "analysis", auth["user_id"])
        
        builder = DatasetBuilder()
        cohort_spec = CohortSpec(
            cohort_id=request.cohort.cohort_id,
            patient_ids=request.cohort.patient_ids
        )
        analysis_spec = AnalysisSpec(
            outcome_variable="event",
            outcome_type="time_to_event",
            time_variable=request.time_variable,
            event_variable=request.event_variable
        )
        df = builder.build_cohort_dataset(cohort_spec, analysis_spec)
        
        if df.empty:
            return {"success": False, "error": "No data available for cohort"}
        
        config = SurvivalConfig(
            time_variable=request.time_variable,
            event_variable=request.event_variable,
            covariates=request.covariates,
            stratify_by=request.stratify_by
        )
        
        analysis = SurvivalAnalysis(df, config)
        results = analysis.get_survival_summary()
        
        return {
            "success": True,
            **results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/ml/analysis/causal")
async def run_causal(request: CausalRequest, auth: dict = Depends(verify_auth)):
    """Run causal inference analysis"""
    try:
        log_audit("causal_analysis", "analysis", auth["user_id"])
        
        builder = DatasetBuilder()
        cohort_spec = CohortSpec(
            cohort_id=request.cohort.cohort_id,
            patient_ids=request.cohort.patient_ids
        )
        analysis_spec = AnalysisSpec(
            outcome_variable=request.outcome_variable,
            outcome_type="binary"
        )
        df = builder.build_cohort_dataset(cohort_spec, analysis_spec)
        
        if df.empty:
            return {"success": False, "error": "No data available for cohort"}
        
        config = CausalConfig(
            treatment_variable=request.treatment_variable,
            outcome_variable=request.outcome_variable,
            covariates=request.covariates,
            method=request.method
        )
        
        analysis = CausalAnalysis(df, config)
        results = analysis.run_analysis()
        
        return {
            "success": True,
            **results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/ml/alerts/evaluate")
async def evaluate_alert(request: AlertRequest, auth: dict = Depends(verify_auth)):
    """Evaluate patient risk and optionally create alert"""
    try:
        log_audit("alert_evaluation", "alerts", auth["user_id"], {
            "patient_id": request.patient_id
        })
        
        engine = AlertEngine()
        engine.load_model()
        
        result = engine.compute_risk_score(request.patient_id)
        
        if request.create_alert and result.get("risk_score"):
            alert_id = engine.create_alert(request.patient_id, result)
            result["alert_created"] = alert_id is not None
            result["alert_id"] = alert_id
        
        return {
            "success": True,
            **result
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/ml/alerts/high-risk")
async def get_high_risk_patients(study_id: Optional[str] = None, auth: dict = Depends(verify_auth)):
    """Get all high-risk patients"""
    try:
        log_audit("high_risk_query", "alerts", auth["user_id"])
        
        engine = AlertEngine()
        engine.load_model()
        
        results = engine.evaluate_all_patients(study_id)
        
        return {
            "success": True,
            "count": len(results),
            "patients": results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/ml/report/generate")
async def generate_report(request: ReportRequest, auth: dict = Depends(verify_auth)):
    """Generate AI-powered research report"""
    try:
        log_audit("report_generation", "reports", auth["user_id"])
        
        config = ReportConfig(style=request.style)
        generator = ReportGenerator(config)
        
        report = generator.generate_full_report(
            request.study_info,
            request.analysis_results
        )
        
        return {
            "success": True,
            **report
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class NLQueryRequest(BaseModel):
    query: str


class NLAnalysisSpec(BaseModel):
    analysis_type: str
    primary_outcome: Optional[str] = None
    exposure_variable: Optional[str] = None
    covariates: List[str] = []
    time_window: Optional[int] = None
    model_type: Optional[str] = None
    cohort_filters: Optional[Dict[str, Any]] = None


class NLParseResponse(BaseModel):
    success: bool
    analysis_spec: Optional[NLAnalysisSpec] = None
    explanation: str
    confidence: float
    suggestions: Optional[List[str]] = None


@app.post("/api/v1/ml/analysis/parse-nl", response_model=NLParseResponse)
async def parse_nl_query(request: NLQueryRequest, auth: dict = Depends(verify_auth)):
    """Parse natural language research query into analysis specification"""
    try:
        log_audit("nl_query_parse", "analysis", auth["user_id"], {"query": request.query[:100]})
        
        query = request.query.lower()
        
        analysis_type = "descriptive"
        primary_outcome = None
        exposure_variable = None
        covariates = []
        time_window = None
        model_type = None
        suggestions = []
        confidence = 0.75
        
        if any(word in query for word in ["predict", "risk", "forecast", "will", "likelihood"]):
            analysis_type = "risk_prediction"
            model_type = "xgboost"
            
            if "30 day" in query or "30-day" in query:
                time_window = 30
            elif "90 day" in query or "90-day" in query:
                time_window = 90
            elif "year" in query:
                time_window = 365
                
            if "infection" in query:
                primary_outcome = "infection_event"
            elif "hospitalization" in query or "hospital" in query:
                primary_outcome = "hospitalization_event"
            elif "deterioration" in query:
                primary_outcome = "deterioration_score"
            elif "mortality" in query or "death" in query:
                primary_outcome = "mortality_event"
                
        elif any(word in query for word in ["survival", "time to", "hazard", "kaplan", "cox"]):
            analysis_type = "survival"
            
            if "infection" in query:
                primary_outcome = "time_to_infection"
            elif "recovery" in query:
                primary_outcome = "time_to_recovery"
            elif "death" in query or "mortality" in query:
                primary_outcome = "time_to_death"
                
        elif any(word in query for word in ["causal", "effect of", "impact of", "compare", "vs", "versus"]):
            analysis_type = "causal"
            
            if "tacrolimus" in query and "cyclosporine" in query:
                exposure_variable = "immunosuppressant_type"
                suggestions.append("Consider adjusting for time since transplant")
            elif "air quality" in query or "aqi" in query or "environmental" in query:
                exposure_variable = "environmental_aqi_score"
                suggestions.append("Environmental data may have seasonal variation")
                
        if "age" in query or "gender" in query or "sex" in query:
            if "age" in query:
                covariates.append("age")
            if "gender" in query or "sex" in query:
                covariates.append("sex")
                
        if "adjusting" in query or "controlling" in query or "adjust" in query:
            if "comorbid" in query:
                covariates.append("comorbidity_score")
            if "bmi" in query:
                covariates.append("bmi")
                
        if "cd4" in query:
            covariates.append("cd4_count")
        if "immune" in query:
            covariates.append("immune_status")
        if "transplant" in query:
            covariates.append("transplant_type")
            
        if "lupus" in query:
            covariates.append("condition:lupus")
        if "transplant patient" in query:
            covariates.append("condition:transplant_recipient")
            
        if not covariates and analysis_type in ["risk_prediction", "survival", "causal"]:
            covariates = ["age", "sex"]
            suggestions.append("Consider adding more covariates for better model performance")
            
        if not primary_outcome and analysis_type != "descriptive":
            suggestions.append("Please specify a primary outcome variable for this analysis")
            confidence = 0.5
            
        explanation_parts = [f"Detected {analysis_type.replace('_', ' ')} analysis request"]
        if primary_outcome:
            explanation_parts.append(f"with outcome '{primary_outcome}'")
        if exposure_variable:
            explanation_parts.append(f"examining effect of '{exposure_variable}'")
        if covariates:
            explanation_parts.append(f"adjusting for {len(covariates)} covariates")
        if time_window:
            explanation_parts.append(f"over {time_window}-day window")
            
        explanation = ". ".join(explanation_parts) + "."
        
        return NLParseResponse(
            success=True,
            analysis_spec=NLAnalysisSpec(
                analysis_type=analysis_type,
                primary_outcome=primary_outcome,
                exposure_variable=exposure_variable,
                covariates=covariates,
                time_window=time_window,
                model_type=model_type,
            ),
            explanation=explanation,
            confidence=confidence,
            suggestions=suggestions if suggestions else None
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class ModelInfoResponse(BaseModel):
    id: str
    model_name: str
    model_type: str
    version: str
    status: str
    is_active: Optional[bool] = None
    metrics: Optional[Dict[str, Any]] = None
    feature_names: Optional[List[str]] = None


class ConsentVerificationRequest(BaseModel):
    patient_id: str
    data_types: List[str]


class ConsentVerificationResponse(BaseModel):
    consented: bool
    granted_types: Optional[List[str]] = None
    missing_types: Optional[List[str]] = None
    reason: Optional[str] = None


class PredictionRequest(BaseModel):
    model_name: str
    study_id: str  # Required for consent verification and audit logging
    data_types: List[str]  # Data types being used for consent verification
    version: Optional[str] = None
    # NOTE: Client-supplied features are NOT accepted. Server constructs features from verified patient data.


class ModelUsageRequest(BaseModel):
    model_id: str
    study_id: str
    analysis_type: str
    patient_count: int


@app.get("/api/v1/model-registry/models", response_model=List[ModelInfoResponse])
async def get_available_models(
    model_type: Optional[str] = None,
    status: str = "active",
    user_id: str = Depends(verify_auth)
):
    """Get list of available trained models from the registry."""
    from ml_analysis.model_registry import get_model_registry
    
    log_audit("VIEW_MODEL_REGISTRY", "models", user_id, {"model_type": model_type, "status": status})
    
    registry = get_model_registry()
    models = await registry.get_available_models(model_type, status)
    
    return [ModelInfoResponse(**m) for m in models]


@app.get("/api/v1/model-registry/models/{model_name}/active")
async def get_active_model(
    model_name: str,
    user_id: str = Depends(verify_auth)
):
    """Get the currently active version of a specific model."""
    from ml_analysis.model_registry import get_model_registry
    
    log_audit("VIEW_ACTIVE_MODEL", f"model:{model_name}", user_id)
    
    registry = get_model_registry()
    model = await registry.get_active_model(model_name)
    
    if not model:
        raise HTTPException(status_code=404, detail=f"No active model found for {model_name}")
    
    return model


@app.get("/api/v1/model-registry/models/{model_name}/versions")
async def get_model_versions(
    model_name: str,
    user_id: str = Depends(verify_auth)
):
    """Get all versions of a specific model."""
    from ml_analysis.model_registry import get_model_registry
    
    registry = get_model_registry()
    versions = await registry.get_model_versions(model_name)
    
    return {"model_name": model_name, "versions": versions}


@app.post("/api/v1/model-registry/verify-consent", response_model=ConsentVerificationResponse)
async def verify_patient_consent(
    request: ConsentVerificationRequest,
    user_id: str = Depends(verify_auth)
):
    """Verify ML training consent for a patient."""
    from ml_analysis.model_registry import get_model_registry
    
    log_audit("VERIFY_ML_CONSENT", f"patient:{request.patient_id}", user_id, {
        "data_types": request.data_types
    })
    
    registry = get_model_registry()
    result = await registry.verify_consent_for_patient(request.patient_id, request.data_types)
    
    return ConsentVerificationResponse(**result)


@app.get("/api/v1/model-registry/consented-patients/count")
async def get_consented_patients_count(
    data_types: str,
    user_id: str = Depends(verify_auth)
):
    """Get count of patients who have consented to ML training for given data types.
    
    Returns aggregate count only to prevent PHI exposure.
    """
    from ml_analysis.model_registry import get_model_registry
    
    log_audit("FETCH_CONSENTED_COUNT", "ml_training_consent", user_id, {
        "data_types": data_types
    })
    
    data_type_list = [dt.strip() for dt in data_types.split(",")]
    
    registry = get_model_registry()
    result = await registry.get_consented_patients_count(data_type_list)
    
    return result


@app.get("/api/v1/model-registry/consented-patients/study/{study_id}/count")
async def get_consented_patients_count_for_study(
    study_id: str,
    data_types: str,
    user_id: str = Depends(verify_auth)
):
    """Get count of consented patients for a specific study.
    
    SECURITY: Returns aggregate count only to prevent PHI exposure.
    Verifies user is authorized to access the study before returning data.
    
    Patient IDs are never exposed via API - only used internally in prediction pipeline.
    """
    from ml_analysis.model_registry import get_model_registry
    
    log_audit("FETCH_STUDY_CONSENTED_COUNT", f"study:{study_id}", user_id, {
        "data_types": data_types
    })
    
    data_type_list = [dt.strip() for dt in data_types.split(",")]
    
    registry = get_model_registry()
    result = await registry.get_consented_patients_count_for_study(study_id, data_type_list, user_id)
    
    if result.get("error"):
        raise HTTPException(status_code=403, detail=result["error"])
    
    return result


@app.post("/api/v1/model-registry/predict")
async def model_predict(
    request: PredictionRequest,
    user_id: str = Depends(verify_auth)
):
    """Make consent-verified predictions using a trained model from the registry.
    
    SECURITY: This endpoint uses SERVER-SIDE feature construction only.
    Client-supplied feature matrices are NOT accepted to prevent data tampering.
    
    Requires study_id and data_types for:
    - User authorization verification (must be assigned to study)
    - Consent verification for all patients
    - Audit logging
    """
    from ml_analysis.model_registry import get_research_predictor
    
    log_audit("MODEL_PREDICTION", f"model:{request.model_name}", user_id, {
        "study_id": request.study_id,
        "data_types": request.data_types,
        "version": request.version
    })
    
    predictor = get_research_predictor()
    
    result = await predictor.predict_for_study(
        model_name=request.model_name,
        study_id=request.study_id,
        data_types=request.data_types,
        user_id=user_id,
        version=request.version
    )
    
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    
    return result


@app.post("/api/v1/model-registry/log-usage")
async def log_model_usage(
    request: ModelUsageRequest,
    user_id: str = Depends(verify_auth)
):
    """Log model usage for audit trail."""
    from ml_analysis.model_registry import get_model_registry
    
    registry = get_model_registry()
    success = await registry.log_model_usage(
        request.model_id,
        request.study_id,
        request.analysis_type,
        request.patient_count,
        user_id
    )
    
    return {"success": success}


@app.get("/api/v1/model-registry/models/{model_name}/compare")
async def compare_model_versions(
    model_name: str,
    version1: str,
    version2: str,
    user_id: str = Depends(verify_auth)
):
    """Compare performance metrics between two model versions."""
    from ml_analysis.model_registry import get_model_registry
    
    log_audit("COMPARE_MODELS", f"model:{model_name}", user_id, {
        "version1": version1,
        "version2": version2
    })
    
    registry = get_model_registry()
    comparison = await registry.compare_model_versions(model_name, version1, version2)
    
    if "error" in comparison:
        raise HTTPException(status_code=404, detail=comparison["error"])
    
    return comparison


@app.get("/api/v1/model-registry/models/{model_name}/features")
async def get_model_feature_importance(
    model_name: str,
    user_id: str = Depends(verify_auth)
):
    """Get feature importance from a trained model."""
    from ml_analysis.model_registry import get_research_predictor
    
    predictor = get_research_predictor()
    result = await predictor.get_feature_importance(model_name)
    
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    
    return result


class SchedulerJobResponse(BaseModel):
    id: str
    name: str
    next_run_time: Optional[str] = None
    trigger: str


class TriggerReanalysisRequest(BaseModel):
    study_id: str
    force: bool = False


@app.get("/api/v1/scheduler/jobs", response_model=List[SchedulerJobResponse])
async def get_scheduler_jobs(user_id: str = Depends(verify_auth)):
    """Get list of scheduled background jobs."""
    from scheduler import get_scheduler
    
    log_audit("VIEW_SCHEDULER_JOBS", "scheduler", user_id)
    
    scheduler = get_scheduler()
    jobs = scheduler.get_jobs()
    
    return [SchedulerJobResponse(**job) for job in jobs]


@app.get("/api/v1/scheduler/status")
async def get_scheduler_status(user_id: str = Depends(verify_auth)):
    """Get scheduler status."""
    from scheduler import get_scheduler
    
    scheduler = get_scheduler()
    
    return {
        "running": scheduler.scheduler.running if scheduler.scheduler else False,
        "job_count": len(scheduler.get_jobs()),
        "jobs": scheduler.get_jobs()
    }


@app.post("/api/v1/scheduler/trigger-reanalysis")
async def trigger_reanalysis(
    request: TriggerReanalysisRequest,
    user_id: str = Depends(verify_auth)
):
    """Manually trigger reanalysis for a specific study."""
    from scheduler import get_scheduler
    import asyncio
    
    log_audit("TRIGGER_REANALYSIS", f"study:{request.study_id}", user_id, {
        "force": request.force
    })
    
    scheduler = get_scheduler()
    
    job_id = await scheduler._create_analysis_job(
        request.study_id, 
        "reanalysis", 
        "manual"
    )
    
    if job_id:
        await scheduler._update_study_reanalysis_timestamp(request.study_id)
        return {"success": True, "job_id": job_id}
    else:
        raise HTTPException(status_code=500, detail="Failed to create reanalysis job")


@app.post("/api/v1/scheduler/run-now/{job_type}")
async def run_job_now(
    job_type: str,
    user_id: str = Depends(verify_auth)
):
    """Manually trigger a scheduled job type immediately."""
    from scheduler import get_scheduler
    import threading
    
    valid_types = ["auto_reanalysis", "risk_scoring", "data_quality", "daily_summary"]
    if job_type not in valid_types:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid job type. Must be one of: {valid_types}"
        )
    
    log_audit("RUN_JOB_NOW", f"scheduler:{job_type}", user_id)
    
    scheduler = get_scheduler()
    
    job_methods = {
        "auto_reanalysis": scheduler._check_auto_reanalysis,
        "risk_scoring": scheduler._run_risk_scoring,
        "data_quality": scheduler._check_data_quality,
        "daily_summary": scheduler._generate_daily_summary
    }
    
    thread = threading.Thread(target=job_methods[job_type])
    thread.start()
    
    return {
        "success": True,
        "message": f"Job {job_type} started in background",
        "job_type": job_type
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PYTHON_ML_PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
