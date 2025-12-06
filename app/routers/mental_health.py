"""
Mental Health Questionnaire API Endpoints

Standardized mental health screening questionnaires with AI-powered analysis:
- PHQ-9 (Patient Health Questionnaire-9) for depression screening
- GAD-7 (Generalized Anxiety Disorder-7) for anxiety screening
- PSS-10 (Perceived Stress Scale-10) for stress assessment

Features:
- Validated questionnaire templates (public domain instruments)
- Automated scoring with severity classification
- Crisis detection with immediate intervention messaging
- LLM-based pattern recognition and symptom clustering
- Non-diagnostic neutral summaries
- Temporal trend analysis
- Export/print functionality
- HIPAA-compliant privacy controls

CRITICAL: No diagnostic language. All summaries are neutral and non-clinical.
"""

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session
from sqlalchemy import desc, and_
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import logging
import json

from app.database import get_db
from app.models.mental_health_models import MentalHealthResponse, MentalHealthPatternAnalysis
from app.models.user import User
from app.dependencies import get_current_user
from app.services.mental_health_service import MentalHealthService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/mental-health", tags=["Mental Health Questionnaires"])

# ==================== Pydantic Models ====================

class QuestionnaireQuestionResponse(BaseModel):
    question_id: str
    question_text: str
    response: int | str
    response_text: Optional[str] = None

class SubmitQuestionnaireRequest(BaseModel):
    questionnaire_type: str = Field(..., pattern="^(PHQ9|GAD7|PSS10)$")
    responses: List[QuestionnaireQuestionResponse]
    duration_seconds: Optional[int] = None
    allow_storage: bool = True
    allow_clinical_sharing: bool = False

class QuestionnaireScoreResponse(BaseModel):
    total_score: int
    max_score: int
    severity_level: str
    severity_description: str
    cluster_scores: Dict[str, Any]
    neutral_summary: str
    key_observations: List[str]

class CrisisInterventionResponse(BaseModel):
    crisis_detected: bool
    crisis_severity: str  # 'moderate', 'high', 'severe'
    intervention_message: str
    crisis_hotlines: List[Dict[str, str]]
    next_steps: List[str]

class QuestionnaireSubmissionResponse(BaseModel):
    response_id: str
    questionnaire_type: str
    score: QuestionnaireScoreResponse
    crisis_intervention: Optional[CrisisInterventionResponse] = None
    analysis_id: Optional[str] = None

class QuestionnaireHistoryItem(BaseModel):
    response_id: str
    questionnaire_type: str
    completed_at: datetime
    total_score: int
    max_score: int
    severity_level: str
    crisis_detected: bool

class PatternAnalysisResponse(BaseModel):
    analysis_id: str
    patterns: List[Dict[str, Any]]
    symptom_clusters: Dict[str, Any]
    temporal_trends: List[Dict[str, Any]]
    neutral_summary: str
    key_observations: List[str]
    suggested_actions: List[Dict[str, str]]

class ExportSummaryResponse(BaseModel):
    patient_name: str
    assessment_date: datetime
    questionnaire_type: str
    questionnaire_full_name: str
    score_summary: QuestionnaireScoreResponse
    pattern_analysis: Optional[PatternAnalysisResponse]
    disclaimers: List[str]
    export_timestamp: datetime

# ==================== Crisis Resources ====================

CRISIS_RESOURCES = {
    "hotlines": [
        {
            "name": "988 Suicide & Crisis Lifeline",
            "phone": "988",
            "description": "24/7 free and confidential support",
            "website": "https://988lifeline.org"
        },
        {
            "name": "Crisis Text Line",
            "sms": "Text HOME to 741741",
            "description": "24/7 crisis support via text",
            "website": "https://www.crisistextline.org"
        },
        {
            "name": "SAMHSA National Helpline",
            "phone": "1-800-662-4357",
            "description": "Treatment referral and information service",
            "website": "https://www.samhsa.gov/find-help/national-helpline"
        }
    ],
    "emergency": {
        "phone": "911",
        "description": "For immediate medical emergencies"
    }
}

# ==================== Questionnaire Templates ====================

@router.get("/questionnaires")
async def get_available_questionnaires() -> Dict[str, Any]:
    """
    Retrieve all available standardized questionnaire templates.
    Returns questionnaire metadata, questions, and scoring information.
    
    NOTE: Public endpoint - questionnaire templates are public domain instruments.
    """
    logger.info("[MH-API] Fetching questionnaire templates (public endpoint)")
    
    service = MentalHealthService()
    questionnaires = service.get_all_questionnaire_templates()
    
    return {
        "questionnaires": questionnaires,
        "total_count": len(questionnaires),
        "disclaimer": "These are screening tools only and do not provide medical diagnosis. Always consult with a licensed healthcare provider."
    }

@router.get("/questionnaires/{questionnaire_type}")
async def get_questionnaire_template(
    questionnaire_type: str
) -> Dict[str, Any]:
    """
    Get a specific questionnaire template with all questions and instructions.
    
    NOTE: Public endpoint - questionnaire templates are public domain instruments.
    """
    if questionnaire_type not in ["PHQ9", "GAD7", "PSS10"]:
        raise HTTPException(status_code=400, detail="Invalid questionnaire type")
    
    logger.info(f"[MH-API] Fetching {questionnaire_type} template (public endpoint)")
    
    service = MentalHealthService()
    template = service.get_questionnaire_template(questionnaire_type)
    
    if not template:
        raise HTTPException(status_code=404, detail="Questionnaire template not found")
    
    return template

# ==================== Questionnaire Submission ====================

@router.post("/submit", response_model=QuestionnaireSubmissionResponse)
async def submit_questionnaire(
    request: Request,
    data: SubmitQuestionnaireRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> QuestionnaireSubmissionResponse:
    """
    Submit a completed mental health questionnaire for scoring and analysis.
    
    Features:
    - Automated scoring with validated algorithms
    - Severity classification (minimal, mild, moderate, moderately severe, severe)
    - Crisis detection with immediate intervention
    - Symptom cluster analysis
    - Non-diagnostic neutral summaries
    - Optional LLM-powered pattern recognition
    """
    patient_id = current_user.id
    logger.info(f"[MH-API] [AUDIT] Patient {patient_id} submitting {data.questionnaire_type}")
    
    try:
        service = MentalHealthService(db)
        
        # Score the questionnaire
        score_result = service.score_questionnaire(
            questionnaire_type=data.questionnaire_type,
            responses=data.responses
        )
        
        # Detect crisis indicators
        crisis_result = service.detect_crisis(
            questionnaire_type=data.questionnaire_type,
            responses=data.responses,
            total_score=score_result['total_score']
        )
        
        # Store response in database (if allowed)
        response_record = None
        if data.allow_storage:
            response_record = service.save_questionnaire_response(
                patient_id=patient_id,
                questionnaire_type=data.questionnaire_type,
                responses=[r.model_dump() for r in data.responses],
                score_result=score_result,
                crisis_result=crisis_result,
                duration_seconds=data.duration_seconds,
                allow_clinical_sharing=data.allow_clinical_sharing
            )
            
            logger.info(f"[MH-API] [AUDIT] Response saved: {response_record.id}")
        
        # Generate LLM-based pattern analysis
        analysis_id = None
        if response_record:
            try:
                analysis = await service.generate_pattern_analysis(
                    patient_id=patient_id,
                    response_id=response_record.id,
                    questionnaire_type=data.questionnaire_type,
                    responses=data.responses,
                    score_result=score_result
                )
                analysis_id = analysis.id if analysis else None
            except Exception as e:
                logger.error(f"[MH-API] Pattern analysis failed: {str(e)}")
                # Continue even if analysis fails
        
        # Prepare crisis intervention if needed
        crisis_intervention = None
        if crisis_result['crisis_detected']:
            crisis_intervention = CrisisInterventionResponse(
                crisis_detected=True,
                crisis_severity=crisis_result['severity'],
                intervention_message=crisis_result['message'],
                crisis_hotlines=CRISIS_RESOURCES['hotlines'],
                next_steps=crisis_result['next_steps']
            )
        
        # Build response
        return QuestionnaireSubmissionResponse(
            response_id=response_record.id if response_record else "not_stored",
            questionnaire_type=data.questionnaire_type,
            score=QuestionnaireScoreResponse(
                total_score=score_result['total_score'],
                max_score=score_result['max_score'],
                severity_level=score_result['severity_level'],
                severity_description=score_result['severity_description'],
                cluster_scores=score_result['cluster_scores'],
                neutral_summary=score_result['neutral_summary'],
                key_observations=score_result['key_observations']
            ),
            crisis_intervention=crisis_intervention,
            analysis_id=analysis_id
        )
        
    except ValueError as e:
        logger.error(f"[MH-API] Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"[MH-API] Submission error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to process questionnaire")

# ==================== History and Trends ====================

@router.get("/history")
async def get_questionnaire_history(
    questionnaire_type: Optional[str] = None,
    limit: int = 20,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Retrieve patient's questionnaire response history.
    Supports filtering by questionnaire type and temporal analysis.
    """
    patient_id = current_user.id
    logger.info(f"[MH-API] Patient {patient_id} requesting history (type={questionnaire_type})")
    
    try:
        query = db.query(MentalHealthResponse).filter(
            MentalHealthResponse.patient_id == patient_id
        )
        
        if questionnaire_type:
            query = query.filter(MentalHealthResponse.questionnaire_type == questionnaire_type)
        
        responses = query.order_by(desc(MentalHealthResponse.completed_at)).limit(limit).all()
        
        history_items = [
            QuestionnaireHistoryItem(
                response_id=r.id,
                questionnaire_type=r.questionnaire_type,
                completed_at=r.completed_at,
                total_score=r.total_score,
                max_score=r.max_score,
                severity_level=r.severity_level,
                crisis_detected=r.crisis_detected
            )
            for r in responses
        ]
        
        # Calculate trends if multiple responses exist
        trends = None
        if len(responses) >= 2:
            service = MentalHealthService(db)
            trends = service.calculate_temporal_trends(responses)
        
        return {
            "history": history_items,
            "total_count": len(history_items),
            "trends": trends
        }
        
    except Exception as e:
        logger.error(f"[MH-API] History retrieval error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve history")

@router.get("/response/{response_id}")
async def get_response_details(
    response_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get detailed information about a specific questionnaire response.
    """
    patient_id = current_user.id
    
    response = db.query(MentalHealthResponse).filter(
        and_(
            MentalHealthResponse.id == response_id,
            MentalHealthResponse.patient_id == patient_id
        )
    ).first()
    
    if not response:
        raise HTTPException(status_code=404, detail="Response not found")
    
    # Get associated pattern analysis if exists
    analysis = db.query(MentalHealthPatternAnalysis).filter(
        MentalHealthPatternAnalysis.response_id == response_id
    ).first()
    
    return {
        "response": {
            "id": response.id,
            "questionnaire_type": response.questionnaire_type,
            "completed_at": response.completed_at,
            "total_score": response.total_score,
            "max_score": response.max_score,
            "severity_level": response.severity_level,
            "cluster_scores": response.cluster_scores,
            "responses": response.responses,
            "crisis_detected": response.crisis_detected
        },
        "analysis": {
            "id": analysis.id,
            "patterns": analysis.patterns,
            "symptom_clusters": analysis.symptom_clusters,
            "temporal_trends": analysis.temporal_trends,
            "neutral_summary": analysis.neutral_summary,
            "key_observations": analysis.key_observations,
            "suggested_actions": analysis.suggested_actions
        } if analysis else None
    }

# ==================== Pattern Analysis ====================

@router.get("/analysis/{response_id}")
async def get_pattern_analysis(
    response_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> PatternAnalysisResponse:
    """
    Retrieve LLM-based pattern analysis for a questionnaire response.
    """
    patient_id = current_user.id
    
    # Verify ownership
    response = db.query(MentalHealthResponse).filter(
        and_(
            MentalHealthResponse.id == response_id,
            MentalHealthResponse.patient_id == patient_id
        )
    ).first()
    
    if not response:
        raise HTTPException(status_code=404, detail="Response not found")
    
    # Get analysis
    analysis = db.query(MentalHealthPatternAnalysis).filter(
        MentalHealthPatternAnalysis.response_id == response_id
    ).first()
    
    if not analysis:
        raise HTTPException(status_code=404, detail="Pattern analysis not found")
    
    return PatternAnalysisResponse(
        analysis_id=analysis.id,
        patterns=analysis.patterns or [],
        symptom_clusters=analysis.symptom_clusters or {},
        temporal_trends=analysis.temporal_trends or [],
        neutral_summary=analysis.neutral_summary or "",
        key_observations=analysis.key_observations or [],
        suggested_actions=analysis.suggested_actions or []
    )

# ==================== Export Functionality ====================

@router.get("/export/{response_id}")
async def export_response_summary(
    response_id: str,
    format: str = "json",
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Export questionnaire response summary for clinical sharing.
    Formats: json, pdf (future), print-friendly html
    """
    patient_id = current_user.id
    
    # Get response
    response = db.query(MentalHealthResponse).filter(
        and_(
            MentalHealthResponse.id == response_id,
            MentalHealthResponse.patient_id == patient_id
        )
    ).first()
    
    if not response:
        raise HTTPException(status_code=404, detail="Response not found")
    
    if not response.allow_clinical_sharing:
        raise HTTPException(status_code=403, detail="Clinical sharing not permitted for this response")
    
    # Get analysis
    analysis = db.query(MentalHealthPatternAnalysis).filter(
        MentalHealthPatternAnalysis.response_id == response_id
    ).first()
    
    service = MentalHealthService(db)
    questionnaire_names = {
        "PHQ9": "Patient Health Questionnaire-9 (PHQ-9)",
        "GAD7": "Generalized Anxiety Disorder-7 (GAD-7)",
        "PSS10": "Perceived Stress Scale-10 (PSS-10)"
    }
    
    export_data = ExportSummaryResponse(
        patient_name=f"{current_user.first_name} {current_user.last_name}",
        assessment_date=response.completed_at,
        questionnaire_type=response.questionnaire_type,
        questionnaire_full_name=questionnaire_names.get(response.questionnaire_type, response.questionnaire_type),
        score_summary=QuestionnaireScoreResponse(
            total_score=response.total_score,
            max_score=response.max_score,
            severity_level=response.severity_level,
            severity_description=service.get_severity_description(response.questionnaire_type, response.severity_level),
            cluster_scores=response.cluster_scores or {},
            neutral_summary="",
            key_observations=[]
        ),
        pattern_analysis=PatternAnalysisResponse(
            analysis_id=analysis.id,
            patterns=analysis.patterns or [],
            symptom_clusters=analysis.symptom_clusters or {},
            temporal_trends=analysis.temporal_trends or [],
            neutral_summary=analysis.neutral_summary or "",
            key_observations=analysis.key_observations or [],
            suggested_actions=analysis.suggested_actions or []
        ) if analysis else None,
        disclaimers=[
            "This is a self-reported screening tool and does not constitute a medical diagnosis.",
            "These results should be reviewed by a qualified healthcare professional.",
            "This information is for clinical consultation purposes only.",
            "Not to be used as the sole basis for treatment decisions."
        ],
        export_timestamp=datetime.utcnow()
    )
    
    return {
        "export": export_data.model_dump(),
        "format": format
    }

# ==================== Crisis Resources ====================

@router.get("/crisis-resources")
async def get_crisis_resources() -> Dict[str, Any]:
    """
    Retrieve crisis intervention resources and hotlines.
    Available to all users without authentication.
    """
    return {
        "crisis_resources": CRISIS_RESOURCES,
        "disclaimer": "If you are in immediate danger, please call 911 or go to your nearest emergency room."
    }
