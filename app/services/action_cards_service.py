"""
Action Cards Service
====================

Voice-triggered task cards for quick patient actions
Enables hands-free health management through AI voice interactions
"""

from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import uuid
import logging

logger = logging.getLogger(__name__)


class ActionType(str, Enum):
    """Types of action cards"""
    MEDICATION_REMINDER = "medication_reminder"
    SYMPTOM_LOG = "symptom_log"
    VITAL_CHECK = "vital_check"
    APPOINTMENT_SCHEDULE = "appointment_schedule"
    EMERGENCY_CONTACT = "emergency_contact"
    WELLNESS_TIP = "wellness_tip"
    FOLLOW_UP = "follow_up"
    QUICK_RESPONSE = "quick_response"
    CONFIRM_ACTION = "confirm_action"


class ActionPriority(str, Enum):
    """Priority levels for actions"""
    URGENT = "urgent"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"


class ActionStatus(str, Enum):
    """Status of an action card"""
    PENDING = "pending"
    DISPLAYED = "displayed"
    COMPLETED = "completed"
    DISMISSED = "dismissed"
    EXPIRED = "expired"


@dataclass
class ActionOption:
    """Single option in an action card"""
    option_id: str
    label: str
    value: str
    voice_trigger: str
    icon: Optional[str] = None
    is_primary: bool = False


@dataclass
class ActionCard:
    """Voice-triggered action card"""
    card_id: str
    patient_id: str
    action_type: ActionType
    title: str
    description: str
    priority: ActionPriority = ActionPriority.NORMAL
    status: ActionStatus = ActionStatus.PENDING
    options: List[ActionOption] = field(default_factory=list)
    voice_prompt: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    displayed_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    selected_option: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)


class ActionCardsService:
    """
    Production-grade action cards service for voice interactions
    
    Features:
    - Voice-triggered action creation
    - Multiple response options with voice triggers
    - Priority-based card ordering
    - Automatic expiration
    """
    
    def __init__(self):
        self._cards: Dict[str, ActionCard] = {}
        self._patient_cards: Dict[str, List[str]] = {}
    
    def create_medication_reminder(
        self,
        patient_id: str,
        medication_name: str,
        dosage: str,
        time: str,
    ) -> ActionCard:
        """Create a medication reminder action card"""
        return self.create_card(
            patient_id=patient_id,
            action_type=ActionType.MEDICATION_REMINDER,
            title=f"Time for {medication_name}",
            description=f"Take {dosage} of {medication_name}",
            voice_prompt=f"It's time for your {medication_name}. Did you take {dosage}?",
            priority=ActionPriority.HIGH,
            options=[
                ActionOption(
                    option_id="taken",
                    label="I took it",
                    value="taken",
                    voice_trigger="yes I took it",
                    is_primary=True,
                ),
                ActionOption(
                    option_id="remind_later",
                    label="Remind me later",
                    value="remind_later",
                    voice_trigger="remind me later",
                ),
                ActionOption(
                    option_id="skip",
                    label="Skip this dose",
                    value="skip",
                    voice_trigger="skip this dose",
                ),
            ],
            metadata={"medication": medication_name, "dosage": dosage, "scheduled_time": time},
        )
    
    def create_symptom_check(
        self,
        patient_id: str,
        symptom: str,
        severity_prompt: str = "How severe is it on a scale of 1 to 10?",
    ) -> ActionCard:
        """Create a symptom severity check card"""
        options = [
            ActionOption(
                option_id=f"severity_{i}",
                label=str(i),
                value=str(i),
                voice_trigger=str(i),
            )
            for i in range(1, 11)
        ]
        
        return self.create_card(
            patient_id=patient_id,
            action_type=ActionType.SYMPTOM_LOG,
            title=f"Rate your {symptom}",
            description=severity_prompt,
            voice_prompt=f"I noticed you mentioned {symptom}. {severity_prompt}",
            priority=ActionPriority.NORMAL,
            options=options,
            metadata={"symptom": symptom},
        )
    
    def create_quick_confirm(
        self,
        patient_id: str,
        question: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> ActionCard:
        """Create a yes/no confirmation card"""
        return self.create_card(
            patient_id=patient_id,
            action_type=ActionType.CONFIRM_ACTION,
            title="Confirmation",
            description=question,
            voice_prompt=question,
            priority=ActionPriority.NORMAL,
            options=[
                ActionOption(
                    option_id="yes",
                    label="Yes",
                    value="yes",
                    voice_trigger="yes",
                    is_primary=True,
                ),
                ActionOption(
                    option_id="no",
                    label="No",
                    value="no",
                    voice_trigger="no",
                ),
            ],
            context=context or {},
        )
    
    def create_emergency_card(
        self,
        patient_id: str,
        emergency_type: str,
    ) -> ActionCard:
        """Create emergency action card"""
        return self.create_card(
            patient_id=patient_id,
            action_type=ActionType.EMERGENCY_CONTACT,
            title="Emergency Detected",
            description=f"We detected a potential {emergency_type}. Do you need immediate help?",
            voice_prompt=f"I'm concerned about what you shared. Do you need immediate medical help?",
            priority=ActionPriority.URGENT,
            options=[
                ActionOption(
                    option_id="call_911",
                    label="Call 911",
                    value="call_911",
                    voice_trigger="call 911",
                    is_primary=True,
                ),
                ActionOption(
                    option_id="call_doctor",
                    label="Contact my doctor",
                    value="call_doctor",
                    voice_trigger="call my doctor",
                ),
                ActionOption(
                    option_id="im_okay",
                    label="I'm okay",
                    value="im_okay",
                    voice_trigger="I'm okay",
                ),
            ],
            metadata={"emergency_type": emergency_type},
        )
    
    def create_card(
        self,
        patient_id: str,
        action_type: ActionType,
        title: str,
        description: str,
        voice_prompt: str,
        priority: ActionPriority = ActionPriority.NORMAL,
        options: Optional[List[ActionOption]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
        expires_in_minutes: Optional[int] = 30,
    ) -> ActionCard:
        """
        Create a new action card
        
        Args:
            patient_id: Patient the card is for
            action_type: Type of action
            title: Card title
            description: Card description
            voice_prompt: What the AI says to present this card
            priority: Priority level
            options: Response options
            metadata: Additional metadata
            context: Conversation context
            expires_in_minutes: Expiration time
            
        Returns:
            Created action card
        """
        card_id = str(uuid.uuid4())
        now = datetime.utcnow()
        
        expires_at = None
        if expires_in_minutes:
            from datetime import timedelta
            expires_at = now + timedelta(minutes=expires_in_minutes)
        
        card = ActionCard(
            card_id=card_id,
            patient_id=patient_id,
            action_type=action_type,
            title=title,
            description=description,
            voice_prompt=voice_prompt,
            priority=priority,
            options=options or [],
            metadata=metadata or {},
            context=context or {},
            created_at=now,
            expires_at=expires_at,
        )
        
        self._cards[card_id] = card
        
        if patient_id not in self._patient_cards:
            self._patient_cards[patient_id] = []
        self._patient_cards[patient_id].append(card_id)
        
        logger.info(f"Created action card {card_id} type {action_type} for patient {patient_id}")
        return card
    
    def get_card(self, card_id: str) -> Optional[ActionCard]:
        """Get an action card by ID"""
        return self._cards.get(card_id)
    
    def get_pending_cards(
        self,
        patient_id: str,
        limit: int = 10,
    ) -> List[ActionCard]:
        """
        Get pending action cards for a patient
        
        Args:
            patient_id: Patient ID
            limit: Maximum cards to return
            
        Returns:
            List of pending cards, sorted by priority
        """
        card_ids = self._patient_cards.get(patient_id, [])
        now = datetime.utcnow()
        
        pending = []
        for card_id in card_ids:
            card = self._cards.get(card_id)
            if not card:
                continue
            if card.status != ActionStatus.PENDING:
                continue
            if card.expires_at and card.expires_at < now:
                card.status = ActionStatus.EXPIRED
                continue
            pending.append(card)
        
        priority_order = {
            ActionPriority.URGENT: 0,
            ActionPriority.HIGH: 1,
            ActionPriority.NORMAL: 2,
            ActionPriority.LOW: 3,
        }
        pending.sort(key=lambda c: priority_order[c.priority])
        
        return pending[:limit]
    
    def mark_displayed(self, card_id: str) -> bool:
        """Mark a card as displayed to the user"""
        card = self._cards.get(card_id)
        if not card:
            return False
        
        card.status = ActionStatus.DISPLAYED
        card.displayed_at = datetime.utcnow()
        return True
    
    def complete_card(
        self,
        card_id: str,
        selected_option: str,
    ) -> Optional[ActionCard]:
        """
        Complete an action card with selected response
        
        Args:
            card_id: Card to complete
            selected_option: ID of selected option
            
        Returns:
            Updated card
        """
        card = self._cards.get(card_id)
        if not card:
            return None
        
        card.status = ActionStatus.COMPLETED
        card.completed_at = datetime.utcnow()
        card.selected_option = selected_option
        
        logger.info(f"Completed action card {card_id} with option {selected_option}")
        return card
    
    def dismiss_card(self, card_id: str) -> bool:
        """Dismiss an action card"""
        card = self._cards.get(card_id)
        if not card:
            return False
        
        card.status = ActionStatus.DISMISSED
        return True
    
    def match_voice_response(
        self,
        card_id: str,
        voice_text: str,
    ) -> Optional[ActionOption]:
        """
        Match voice input to card options
        
        Args:
            card_id: Card to match against
            voice_text: Transcribed voice input
            
        Returns:
            Matched option or None
        """
        card = self._cards.get(card_id)
        if not card:
            return None
        
        voice_lower = voice_text.lower().strip()
        
        for option in card.options:
            if option.voice_trigger.lower() in voice_lower:
                return option
            if voice_lower in option.voice_trigger.lower():
                return option
            if option.value.lower() == voice_lower:
                return option
        
        return None
    
    def cleanup_expired(self) -> int:
        """Clean up expired cards"""
        now = datetime.utcnow()
        expired_count = 0
        
        for card in self._cards.values():
            if card.status == ActionStatus.PENDING and card.expires_at:
                if card.expires_at < now:
                    card.status = ActionStatus.EXPIRED
                    expired_count += 1
        
        return expired_count


_action_cards_service: Optional[ActionCardsService] = None


def get_action_cards_service() -> ActionCardsService:
    """Get singleton action cards service instance"""
    global _action_cards_service
    if _action_cards_service is None:
        _action_cards_service = ActionCardsService()
    return _action_cards_service
