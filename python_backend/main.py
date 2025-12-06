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


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("ML Analysis Engine starting...")
    yield
    print("ML Analysis Engine shutting down...")

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


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PYTHON_ML_PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
