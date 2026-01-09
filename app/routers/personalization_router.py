"""
Personalization Router
Endpoints for EHR-driven personalized recommendations and autopilot suggestions.
"""

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Optional
import logging

from app.database import get_db
from app.services.personalized_recommendations_service import get_personalized_recommendations_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/personalization/patient", tags=["personalization"])


class RecommendationResponse(BaseModel):
    """A single habit recommendation."""
    name: str
    description: str
    category: str
    frequency: str
    goalCount: int
    reason: str
    safety_notes: Optional[str] = None


class RecommendationsResponse(BaseModel):
    """Response containing list of recommendations."""
    recommendations: List[RecommendationResponse]
    personalized: bool = True
    conditions_count: int = 0


class AutopilotSuggestion(BaseModel):
    """A single autopilot suggestion."""
    id: str
    question: str
    type: str
    reason: str
    severity: str


class AutopilotResponse(BaseModel):
    """Response containing autopilot suggestions."""
    items: List[AutopilotSuggestion]
    metadata: dict


@router.get("/{patient_id}/recommendations", response_model=RecommendationsResponse)
async def get_recommendations(
    patient_id: str,
    max_recommendations: int = 10,
    db: Session = Depends(get_db)
):
    """
    Get personalized habit recommendations based on patient's EHR data.
    
    Returns recommendations based on:
    - Problem list (diagnoses/conditions)
    - Recent complaints
    - Current medications
    
    Each recommendation includes:
    - name, description, category, frequency, goalCount
    - reason: Why this is recommended based on their specific conditions
    - safety_notes: When to contact provider
    """
    service = get_personalized_recommendations_service(db)
    
    try:
        recommendations = await service.get_recommendations(
            patient_id=patient_id,
            accessor_id=patient_id,
            max_recommendations=max_recommendations
        )
        
        return RecommendationsResponse(
            recommendations=recommendations,
            personalized=True,
            conditions_count=len(recommendations)
        )
    except Exception as e:
        logger.error(f"Error getting recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{patient_id}/autopilot/suggestions", response_model=AutopilotResponse)
async def get_autopilot_suggestions(
    patient_id: str,
    db: Session = Depends(get_db)
):
    """
    Get personalized daily followup suggestions based on patient's EHR data.
    
    Returns questions/checks personalized to:
    - Patient's conditions (respiratory, cardiac, mental health, etc.)
    - Current medications
    - Recent complaints
    
    This endpoint always returns suggestions (never "Autopilot Unavailable").
    Uses EHR-based rules as the default, with ML enhancement when available.
    """
    service = get_personalized_recommendations_service(db)
    
    try:
        result = await service.get_autopilot_suggestions(
            patient_id=patient_id,
            accessor_id=patient_id
        )
        
        return AutopilotResponse(
            items=result["items"],
            metadata=result["metadata"]
        )
    except Exception as e:
        logger.error(f"Error getting autopilot suggestions: {e}")
        raise HTTPException(status_code=500, detail=str(e))
