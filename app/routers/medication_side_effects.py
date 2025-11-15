"""
Medication Side-Effect Analysis API endpoints
Provides AI-powered correlation analysis between medications and symptoms
HIPAA-compliant with defense-in-depth security
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
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


# ============================================================================
# Doctor Consultation Report Endpoints
# ============================================================================

@router.get("/doctor/patient/{patient_id}/consultation-report")
async def get_doctor_consultation_report(
    patient_id: str,
    days_back: int = Query(default=90, ge=7, le=365),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Doctor-facing comprehensive medication effects report for patient consultation
    
    Security: Doctors only - requires active patient-doctor connection
    Returns: Complete medication timeline + symptoms + correlations + recommendations
    """
    # Verify doctor role
    if current_user.role != "doctor":
        raise HTTPException(status_code=403, detail="Only doctors can access consultation reports")
    
    # Verify active doctor-patient connection
    connection = db.query(PatientDoctorConnection).filter(
        PatientDoctorConnection.patient_id == patient_id,
        PatientDoctorConnection.doctor_id == current_user.id,
        PatientDoctorConnection.status == "connected"
    ).first()
    
    if not connection:
        raise HTTPException(
            status_code=403,
            detail="You do not have permission to view this patient's data. Patient must be connected to you."
        )
    
    cutoff_date = datetime.utcnow() - timedelta(days=days_back)
    
    # Fetch patient's active medications
    from app.models.medication_side_effects import MedicationTimeline
    
    active_medications = db.query(MedicationTimeline).filter(
        MedicationTimeline.patient_id == patient_id,
        MedicationTimeline.is_active == True
    ).all()
    
    all_medications = db.query(MedicationTimeline).filter(
        MedicationTimeline.patient_id == patient_id,
        MedicationTimeline.started_at >= cutoff_date
    ).all()
    
    # Fetch recent symptoms (all sources)
    recent_symptoms = db.query(SymptomLog).filter(
        SymptomLog.patient_id == patient_id,
        SymptomLog.reported_at >= cutoff_date
    ).order_by(SymptomLog.reported_at.desc()).all()
    
    # Fetch correlations
    correlations = db.query(SideEffectCorrelation).filter(
        SideEffectCorrelation.patient_id == patient_id,
        SideEffectCorrelation.analysis_date >= cutoff_date
    ).all()
    
    # Group correlations by strength
    strong_correlations = [c for c in correlations if c.correlation_strength == CorrelationStrength.STRONG]
    likely_correlations = [c for c in correlations if c.correlation_strength == CorrelationStrength.LIKELY]
    possible_correlations = [c for c in correlations if c.correlation_strength == CorrelationStrength.POSSIBLE]
    
    # Get most recent analysis summary
    from app.models.medication_side_effects import MedicationEffectsSummary
    
    summary = db.query(MedicationEffectsSummary).filter(
        MedicationEffectsSummary.patient_id == patient_id
    ).order_by(MedicationEffectsSummary.created_at.desc()).first()
    
    # Build medication details with correlations
    medication_details = []
    for med in active_medications:
        med_correlations = [c for c in correlations if c.medication_timeline_id == med.id]
        
        medication_details.append({
            "medication_name": med.medication_name,
            "generic_name": med.generic_name,
            "drug_class": med.drug_class,
            "dosage": med.dosage,
            "frequency": med.frequency,
            "route": med.route,
            "started_at": med.started_at.isoformat() if med.started_at else None,
            "prescribed_by": med.prescribed_by,
            "prescription_reason": med.prescription_reason,
            "correlation_count": len(med_correlations),
            "strong_correlation_count": sum(1 for c in med_correlations if c.correlation_strength == CorrelationStrength.STRONG),
            "correlations": [
                {
                    "symptom_name": c.symptom_name,
                    "correlation_strength": c.correlation_strength.value,
                    "confidence_score": c.confidence_score,
                    "time_to_onset_hours": c.time_to_onset_hours,
                    "patient_impact": c.patient_impact,
                    "action_recommended": c.action_recommended,
                    "temporal_pattern": c.temporal_pattern
                }
                for c in sorted(med_correlations, key=lambda x: x.confidence_score, reverse=True)[:5]
            ]
        })
    
    # Symptom summary by source
    from app.models.medication_side_effects import SymptomSource
    symptom_sources = {}
    for symptom in recent_symptoms:
        source_name = symptom.source.value if symptom.source else "unknown"
        if source_name not in symptom_sources:
            symptom_sources[source_name] = []
        symptom_sources[source_name].append({
            "symptom_name": symptom.symptom_name,
            "severity": symptom.severity,
            "reported_at": symptom.reported_at.isoformat() if symptom.reported_at else None,
            "description": symptom.symptom_description
        })
    
    # Clinical recommendations (from AI analysis)
    clinical_recommendations = []
    if summary and summary.recommendations:
        clinical_recommendations = summary.recommendations
    else:
        # Generate basic recommendations from strong correlations
        for corr in strong_correlations[:5]:  # Top 5 strong correlations
            clinical_recommendations.append({
                "priority": "HIGH",
                "medication": corr.medication_name,
                "symptom": corr.symptom_name,
                "action": corr.action_recommended,
                "reasoning": corr.ai_reasoning
            })
    
    return {
        "patient_id": patient_id,
        "report_generated_at": datetime.utcnow().isoformat(),
        "analysis_period_days": days_back,
        "doctor_id": current_user.id,
        "doctor_name": f"{current_user.first_name} {current_user.last_name}" if hasattr(current_user, 'first_name') else "Doctor",
        
        # Medication Summary
        "active_medications_count": len(active_medications),
        "all_medications_in_period": len(all_medications),
        "medications": medication_details,
        
        # Symptom Summary
        "total_symptoms_reported": len(recent_symptoms),
        "symptoms_by_source": symptom_sources,
        "unique_symptoms": len(set(s.symptom_name for s in recent_symptoms)),
        
        # Correlation Summary
        "total_correlations": len(correlations),
        "strong_correlations_count": len(strong_correlations),
        "likely_correlations_count": len(likely_correlations),
        "possible_correlations_count": len(possible_correlations),
        "critical_findings": [
            {
                "id": c.id,
                "medication_name": c.medication_name,
                "symptom_name": c.symptom_name,
                "confidence_score": c.confidence_score,
                "patient_impact": c.patient_impact,
                "action_recommended": c.action_recommended,
                "temporal_pattern": c.temporal_pattern,
                "ai_reasoning": c.ai_reasoning
            }
            for c in strong_correlations[:10]  # Top 10 critical findings
        ],
        
        # Clinical Recommendations
        "clinical_recommendations": clinical_recommendations,
        
        # Analysis metadata
        "last_analysis_date": summary.created_at.isoformat() if summary else None,
        "analysis_status": "current" if summary and (datetime.utcnow() - summary.created_at).days < 7 else "needs_update"
    }
