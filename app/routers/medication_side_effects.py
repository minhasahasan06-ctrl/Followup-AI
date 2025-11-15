"""
Medication Side-Effect Analysis API endpoints
Provides AI-powered correlation analysis between medications and symptoms
HIPAA-compliant with defense-in-depth security
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime, timedelta

from app.database import get_db
from app.dependencies import get_current_user
from app.models.user import User
from app.models.medication_side_effects import (
    SideEffectCorrelation,
    MedicationEffectsSummary,
    CorrelationStrength,
    SymptomLog
)
from app.models.patient_doctor_connection import PatientDoctorConnection
from app.services.medication_correlation import MedicationCorrelationEngine
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/v1/medication-side-effects", tags=["Medication Side Effects"])


# ============================================================================
# Request/Response Models
# ============================================================================

class AnalyzeCorrelationsRequest(BaseModel):
    """Request to analyze medication-symptom correlations"""
    days_back: int = Field(default=90, ge=7, le=365, description="Analysis window in days (7-365)")
    min_confidence: float = Field(default=0.4, ge=0.0, le=1.0, description="Minimum confidence score (0-1)")


class CorrelationResponse(BaseModel):
    """Individual correlation details"""
    id: int
    medication_name: str
    symptom_name: str
    correlation_strength: str
    confidence_score: float
    time_to_onset_hours: int
    symptom_onset_date: datetime
    medication_change_date: datetime
    temporal_pattern: Optional[str]
    ai_reasoning: Optional[str]
    patient_impact: str
    action_recommended: str
    analysis_date: datetime
    
    class Config:
        from_attributes = True


class MedicationCorrelationGroup(BaseModel):
    """Correlations grouped by medication"""
    medication_name: str
    dosage: str
    total_correlations: int
    strong_correlations: int
    correlations: List[CorrelationResponse]


class EffectsSummaryResponse(BaseModel):
    """Summary of medication effects analysis"""
    patient_id: str
    analysis_period_days: int
    total_correlations_found: int
    strong_correlations_count: int
    medications_analyzed: int
    medication_groups: List[MedicationCorrelationGroup]
    summary_generated_at: datetime
    recommendations: str


# ============================================================================
# Helper Functions (Defense-in-Depth Security)
# ============================================================================

def verify_patient_access(
    patient_id: str,
    current_user: User,
    db: Session
) -> None:
    """
    Defense-in-depth: Verify user has access to patient data
    Prevents unauthorized access
    """
    # Verify user has permission to access
    if current_user.role == "patient":
        # Patients can only access their own data
        if current_user.id != patient_id:
            raise HTTPException(status_code=403, detail="Patients can only access their own medication analysis")
    elif current_user.role == "doctor":
        # Doctors need active connection
        connection = db.query(PatientDoctorConnection).filter(
            PatientDoctorConnection.patient_id == patient_id,
            PatientDoctorConnection.doctor_id == current_user.id,
            PatientDoctorConnection.status == "active"
        ).first()
        if not connection:
            raise HTTPException(status_code=403, detail="No active connection with this patient")
    else:
        raise HTTPException(status_code=403, detail="Invalid role for medication analysis access")


# ============================================================================
# API Endpoints
# ============================================================================

@router.post("/analyze/{patient_id}")
async def analyze_patient_correlations(
    patient_id: str,
    request: AnalyzeCorrelationsRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Trigger AI-powered correlation analysis for patient's medications and symptoms
    
    Security: Patients (self-access only) + Doctors (require active connection)
    Analysis: Uses OpenAI GPT-4o for temporal pattern detection
    """
    # Defense-in-depth security verification
    verify_patient_access(patient_id, current_user, db)
    
    # Initialize correlation engine
    engine = MedicationCorrelationEngine(db)
    
    try:
        # Run correlation analysis
        correlations = engine.analyze_patient_correlations(
            patient_id=patient_id,
            days_back=request.days_back,
            min_correlation_score=request.min_confidence
        )
        
        # Convert to response format
        correlation_responses = [
            CorrelationResponse.model_validate(corr)
            for corr in correlations
        ]
        
        return {
            "patient_id": patient_id,
            "analysis_period_days": request.days_back,
            "min_confidence_threshold": request.min_confidence,
            "total_correlations_found": len(correlations),
            "strong_correlations": sum(1 for c in correlations if c.correlation_strength == CorrelationStrength.STRONG),
            "correlations": correlation_responses,
            "analyzed_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Correlation analysis failed: {str(e)}")


@router.get("/correlations/{patient_id}")
async def get_patient_correlations(
    patient_id: str,
    days_back: int = Query(default=90, ge=7, le=365),
    min_strength: Optional[str] = Query(default=None, description="Filter by strength: STRONG, LIKELY, POSSIBLE, UNLIKELY"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> List[CorrelationResponse]:
    """
    Get existing correlation records for a patient
    
    Security: Patients (self-access only) + Doctors (require active connection)
    Filtering: Optional by time window and correlation strength
    """
    # Defense-in-depth security verification
    verify_patient_access(patient_id, current_user, db)
    
    # Build query
    cutoff_date = datetime.utcnow() - timedelta(days=days_back)
    query = db.query(SideEffectCorrelation).filter(
        SideEffectCorrelation.patient_id == patient_id,
        SideEffectCorrelation.analysis_date >= cutoff_date
    )
    
    # Optional strength filter
    if min_strength:
        try:
            strength_enum = CorrelationStrength[min_strength.upper()]
            query = query.filter(SideEffectCorrelation.correlation_strength == strength_enum)
        except KeyError:
            raise HTTPException(status_code=400, detail=f"Invalid strength filter: {min_strength}")
    
    correlations = query.order_by(SideEffectCorrelation.analysis_date.desc()).all()
    
    return [CorrelationResponse.model_validate(corr) for corr in correlations]


@router.get("/summary/{patient_id}")
async def get_effects_summary(
    patient_id: str,
    days_back: int = Query(default=90, ge=7, le=365),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> EffectsSummaryResponse:
    """
    Get comprehensive medication effects summary with grouped correlations
    
    Security: Patients (self-access only) + Doctors (require active connection)
    Output: Correlations grouped by medication with analysis metadata
    """
    # Defense-in-depth security verification
    verify_patient_access(patient_id, current_user, db)
    
    # Initialize correlation engine
    engine = MedicationCorrelationEngine(db)
    
    try:
        # Generate comprehensive summary
        summary = engine.generate_patient_summary(patient_id, days_back)
        
        # Parse medication groups from summary_data
        medication_groups = []
        for med_id, med_data in summary.summary_data.get("medication_summaries", {}).items():
            # Get correlations for this medication
            med_correlations = db.query(SideEffectCorrelation).filter(
                SideEffectCorrelation.patient_id == patient_id,
                SideEffectCorrelation.medication_timeline_id == int(med_id),
                SideEffectCorrelation.analysis_date >= datetime.utcnow() - timedelta(days=days_back)
            ).all()
            
            # Count strong correlations
            strong_count = sum(1 for c in med_correlations if c.correlation_strength == CorrelationStrength.STRONG)
            
            medication_groups.append(
                MedicationCorrelationGroup(
                    medication_name=med_data["medication_name"],
                    dosage=med_data["dosage"],
                    total_correlations=len(med_correlations),
                    strong_correlations=strong_count,
                    correlations=[CorrelationResponse.model_validate(c) for c in med_correlations]
                )
            )
        
        return EffectsSummaryResponse(
            patient_id=summary.patient_id,
            analysis_period_days=summary.analysis_period_days,
            total_correlations_found=summary.total_correlations_found,
            strong_correlations_count=summary.strong_correlations_count,
            medications_analyzed=summary.summary_data.get("medications_analyzed", 0),
            medication_groups=medication_groups,
            summary_generated_at=summary.created_at,
            recommendations=summary.recommendations
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Summary generation failed: {str(e)}")


@router.get("/correlation/{correlation_id}")
async def get_correlation_details(
    correlation_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> CorrelationResponse:
    """
    Get detailed information about a specific correlation
    
    Security: Patients (own correlations only) + Doctors (require active connection)
    Details: Full correlation record with AI reasoning and evidence
    """
    # Get correlation
    correlation = db.query(SideEffectCorrelation).filter(
        SideEffectCorrelation.id == correlation_id
    ).first()
    
    if not correlation:
        raise HTTPException(status_code=404, detail="Correlation not found")
    
    # Defense-in-depth security verification
    verify_patient_access(correlation.patient_id, current_user, db)
    
    return CorrelationResponse.model_validate(correlation)
