"""
Agent Clona - Enhanced AI Chatbot Router.
Provides symptom analysis, differential diagnosis, and doctor suggestions.
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from pydantic import BaseModel

from app.database import get_db, SessionLocal
from app.dependencies import get_current_user, require_role
from app.models.user import User
from app.services.agent_clona_service import AgentClonaService
from app.services.symptom_extraction_service import SymptomExtractionService


router = APIRouter(prefix="/api/agent-clona", tags=["agent-clona"])


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    conversation_history: List[Message] = []


class SymptomAnalysisResponse(BaseModel):
    ai_response: str
    structured_recommendations: Optional[dict] = None
    suggested_doctors: List[dict] = []
    timestamp: str


class DiagnosticQuestionsRequest(BaseModel):
    symptom_category: str
    initial_symptoms: List[str]


class LabTestRecommendationRequest(BaseModel):
    symptoms: List[str]
    patient_history: Optional[dict] = None


class TreatmentSuggestionRequest(BaseModel):
    diagnosis_considerations: List[str]
    patient_medications: Optional[List[str]] = None


def extract_symptoms_background(
    patient_id: str,
    patient_message: str,
    ai_response: str
):
    """
    Background task to extract and save symptoms from chat conversation.
    
    FIX: Creates fresh DB session to avoid using closed request-scoped session.
    """
    # Create fresh database session for background task
    db = SessionLocal()
    try:
        SymptomExtractionService.extract_and_save_symptoms(
            db=db,
            patient_id=patient_id,
            patient_message=patient_message,
            ai_response=ai_response
        )
    except Exception as e:
        print(f"Background symptom extraction error: {e}")
    finally:
        db.close()


@router.post("/chat", response_model=SymptomAnalysisResponse)
async def chat_with_clona(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(require_role("patient")),
    db: Session = Depends(get_db)
):
    """
    Enhanced chat with Agent Clona for symptom analysis.
    Provides differential diagnosis, lab test recommendations, and doctor suggestions.
    
    AUTOMATIC SYMPTOM EXTRACTION:
    - Analyzes patient messages for symptom mentions
    - Automatically creates SymptomLog entries in background
    - Integrates with Medication Side-Effect Predictor
    """
    try:
        conversation = [
            {"role": msg.role, "content": msg.content}
            for msg in request.conversation_history
        ]
        
        analysis = AgentClonaService.analyze_symptoms(
            db=db,
            patient=current_user,
            symptom_description=request.message,
            conversation_history=conversation
        )
        
        # Background task: Extract and save symptoms from this conversation
        # FIX: No DB session passed - background task creates its own
        background_tasks.add_task(
            extract_symptoms_background,
            patient_id=current_user.id,
            patient_message=request.message,
            ai_response=analysis["ai_response"]
        )
        
        return analysis
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error analyzing symptoms: {str(e)}"
        )


@router.post("/diagnostic-questions")
async def get_diagnostic_questions(
    request: DiagnosticQuestionsRequest,
    current_user: User = Depends(require_role("patient"))
):
    """
    Generate follow-up questions for differential diagnosis.
    Helps narrow down possible conditions based on symptom category.
    """
    try:
        questions = AgentClonaService.generate_differential_diagnosis_questions(
            symptom_category=request.symptom_category,
            initial_symptoms=request.initial_symptoms
        )
        
        return {"questions": questions}
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating questions: {str(e)}"
        )


@router.post("/recommend-tests")
async def recommend_lab_tests(
    request: LabTestRecommendationRequest,
    current_user: User = Depends(require_role("patient"))
):
    """
    Recommend lab tests and physical examinations based on symptoms.
    Provides reasoning and priority levels for each recommendation.
    """
    try:
        recommendations = AgentClonaService.recommend_lab_tests(
            symptoms=request.symptoms,
            patient_history=request.patient_history
        )
        
        return recommendations
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error recommending tests: {str(e)}"
        )


@router.post("/treatment-suggestions")
async def suggest_treatment(
    request: TreatmentSuggestionRequest,
    current_user: User = Depends(require_role("patient"))
):
    """
    Suggest treatment approaches based on possible conditions.
    Includes medication options, lifestyle modifications, and monitoring recommendations.
    """
    try:
        suggestions = AgentClonaService.suggest_treatment_approach(
            diagnosis_considerations=request.diagnosis_considerations,
            patient_medications=request.patient_medications
        )
        
        return suggestions
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating treatment suggestions: {str(e)}"
        )


@router.get("/recommendations")
async def get_habit_recommendations(
    patientId: str = "me",
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get AI-powered personalized habit recommendations from Agent Clona.
    
    Uses the patient's health profile, conditions, and medications to generate
    evidence-based habit suggestions categorized by health domain.
    
    Returns:
        recommendations: List of habit suggestions by category
        generated_at: Timestamp of generation
        source: "agent_clona"
    """
    from datetime import datetime
    from app.services.personalized_recommendations_service import get_personalized_recommendations_service
    
    try:
        # Determine patient ID
        pid = current_user.id if patientId == "me" else patientId
        
        # If accessing another patient's data, verify authorization
        if patientId != "me" and str(pid) != str(current_user.id):
            if current_user.role != "doctor":
                raise HTTPException(
                    status_code=403,
                    detail="Not authorized to access this patient's recommendations"
                )
        
        service = get_personalized_recommendations_service(db)
        raw_recommendations = await service.get_recommendations(
            patient_id=str(pid),
            accessor_id=str(current_user.id),
            max_recommendations=15
        )
        
        # Transform to Agent Clona format with categories
        category_map = {}
        for rec in raw_recommendations:
            category = rec.get("category", "wellness")
            if category not in category_map:
                category_map[category] = {
                    "category": category,
                    "condition_context": rec.get("reason", "").split(" for ")[-1] if " for " in rec.get("reason", "") else None,
                    "habits": []
                }
            
            category_map[category]["habits"].append({
                "name": rec.get("name", ""),
                "description": rec.get("description", ""),
                "frequency": rec.get("frequency", "daily"),
                "priority": "high" if "important" in rec.get("reason", "").lower() else "medium",
                "evidence_based": True,
                "condition_link": rec.get("safety_notes")
            })
        
        return {
            "recommendations": list(category_map.values()),
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "source": "agent_clona"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating recommendations: {str(e)}"
        )
