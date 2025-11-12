"""
Agent Clona - Enhanced AI Chatbot Router.
Provides symptom analysis, differential diagnosis, and doctor suggestions.
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel

from app.database import get_db
from app.dependencies import get_current_user, require_role
from app.models.user import User
from app.services.agent_clona_service import AgentClonaService


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


@router.post("/chat", response_model=SymptomAnalysisResponse)
async def chat_with_clona(
    request: ChatRequest,
    current_user: User = Depends(require_role("patient")),
    db: Session = Depends(get_db)
):
    """
    Enhanced chat with Agent Clona for symptom analysis.
    Provides differential diagnosis, lab test recommendations, and doctor suggestions.
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
