"""
Action Cards API Router
=======================

API endpoints for voice-triggered action cards
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any, List

from app.services.action_cards_service import (
    get_action_cards_service,
    ActionType,
    ActionPriority,
)

router = APIRouter(prefix="/api/action-cards", tags=["Action Cards"])


class CreateCardRequest(BaseModel):
    patient_id: str
    action_type: str
    title: str
    description: str
    voice_prompt: str
    priority: str = "normal"
    options: Optional[List[Dict[str, Any]]] = None
    metadata: Optional[Dict[str, Any]] = None
    expires_in_minutes: Optional[int] = 30


class MedicationReminderRequest(BaseModel):
    patient_id: str
    medication_name: str
    dosage: str
    time: str


class SymptomCheckRequest(BaseModel):
    patient_id: str
    symptom: str
    severity_prompt: Optional[str] = None


class QuickConfirmRequest(BaseModel):
    patient_id: str
    question: str
    context: Optional[Dict[str, Any]] = None


class CompleteCardRequest(BaseModel):
    selected_option: str


class VoiceMatchRequest(BaseModel):
    voice_text: str


@router.get("/{patient_id}/pending")
async def get_pending_cards(patient_id: str, limit: int = 10):
    """Get pending action cards for a patient"""
    service = get_action_cards_service()
    cards = service.get_pending_cards(patient_id, limit)
    
    return {
        "patient_id": patient_id,
        "cards": [
            {
                "card_id": card.card_id,
                "action_type": card.action_type.value,
                "title": card.title,
                "description": card.description,
                "voice_prompt": card.voice_prompt,
                "priority": card.priority.value,
                "options": [
                    {
                        "option_id": opt.option_id,
                        "label": opt.label,
                        "value": opt.value,
                        "voice_trigger": opt.voice_trigger,
                        "is_primary": opt.is_primary,
                    }
                    for opt in card.options
                ],
                "created_at": card.created_at.isoformat(),
                "expires_at": card.expires_at.isoformat() if card.expires_at else None,
            }
            for card in cards
        ],
        "count": len(cards),
    }


@router.get("/{card_id}")
async def get_card(card_id: str):
    """Get a specific action card"""
    service = get_action_cards_service()
    card = service.get_card(card_id)
    
    if not card:
        raise HTTPException(status_code=404, detail="Card not found")
    
    return {
        "card_id": card.card_id,
        "patient_id": card.patient_id,
        "action_type": card.action_type.value,
        "title": card.title,
        "description": card.description,
        "voice_prompt": card.voice_prompt,
        "priority": card.priority.value,
        "status": card.status.value,
        "options": [
            {
                "option_id": opt.option_id,
                "label": opt.label,
                "value": opt.value,
                "voice_trigger": opt.voice_trigger,
            }
            for opt in card.options
        ],
        "selected_option": card.selected_option,
        "created_at": card.created_at.isoformat(),
        "completed_at": card.completed_at.isoformat() if card.completed_at else None,
    }


@router.post("/medication-reminder")
async def create_medication_reminder(request: MedicationReminderRequest):
    """Create a medication reminder action card"""
    service = get_action_cards_service()
    
    card = service.create_medication_reminder(
        patient_id=request.patient_id,
        medication_name=request.medication_name,
        dosage=request.dosage,
        time=request.time,
    )
    
    return {
        "card_id": card.card_id,
        "voice_prompt": card.voice_prompt,
        "created": True,
    }


@router.post("/symptom-check")
async def create_symptom_check(request: SymptomCheckRequest):
    """Create a symptom severity check card"""
    service = get_action_cards_service()
    
    card = service.create_symptom_check(
        patient_id=request.patient_id,
        symptom=request.symptom,
        severity_prompt=request.severity_prompt or "How severe is it on a scale of 1 to 10?",
    )
    
    return {
        "card_id": card.card_id,
        "voice_prompt": card.voice_prompt,
        "created": True,
    }


@router.post("/quick-confirm")
async def create_quick_confirm(request: QuickConfirmRequest):
    """Create a yes/no confirmation card"""
    service = get_action_cards_service()
    
    card = service.create_quick_confirm(
        patient_id=request.patient_id,
        question=request.question,
        context=request.context,
    )
    
    return {
        "card_id": card.card_id,
        "voice_prompt": card.voice_prompt,
        "created": True,
    }


@router.post("/emergency")
async def create_emergency_card(patient_id: str, emergency_type: str):
    """Create an emergency action card"""
    service = get_action_cards_service()
    
    card = service.create_emergency_card(
        patient_id=patient_id,
        emergency_type=emergency_type,
    )
    
    return {
        "card_id": card.card_id,
        "voice_prompt": card.voice_prompt,
        "priority": "urgent",
        "created": True,
    }


@router.post("/{card_id}/display")
async def mark_displayed(card_id: str):
    """Mark a card as displayed"""
    service = get_action_cards_service()
    
    if not service.mark_displayed(card_id):
        raise HTTPException(status_code=404, detail="Card not found")
    
    return {"success": True, "card_id": card_id}


@router.post("/{card_id}/complete")
async def complete_card(card_id: str, request: CompleteCardRequest):
    """Complete an action card with selected option"""
    service = get_action_cards_service()
    
    card = service.complete_card(card_id, request.selected_option)
    
    if not card:
        raise HTTPException(status_code=404, detail="Card not found")
    
    return {
        "success": True,
        "card_id": card_id,
        "selected_option": request.selected_option,
        "action_type": card.action_type.value,
    }


@router.post("/{card_id}/dismiss")
async def dismiss_card(card_id: str):
    """Dismiss an action card"""
    service = get_action_cards_service()
    
    if not service.dismiss_card(card_id):
        raise HTTPException(status_code=404, detail="Card not found")
    
    return {"success": True, "card_id": card_id}


@router.post("/{card_id}/voice-match")
async def match_voice_response(card_id: str, request: VoiceMatchRequest):
    """Match voice input to card options"""
    service = get_action_cards_service()
    
    option = service.match_voice_response(card_id, request.voice_text)
    
    if not option:
        return {
            "matched": False,
            "voice_text": request.voice_text,
            "suggestion": "Please try again or select an option manually",
        }
    
    return {
        "matched": True,
        "option_id": option.option_id,
        "label": option.label,
        "value": option.value,
    }


@router.post("/cleanup")
async def cleanup_expired():
    """Clean up expired cards"""
    service = get_action_cards_service()
    count = service.cleanup_expired()
    
    return {"expired_count": count}
