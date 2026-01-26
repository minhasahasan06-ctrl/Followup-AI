"""
Local Library Service (Phase B.19-B.21)
=======================================
Provides access to local question bank, habit library, and template rendering.

Tasks Covered:
- B.19: get_questions(question_ids)
- B.20: get_habits(habit_ids)
- B.21: render_templates(template_ids, packet) - NO PHI output
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from app.services.privacy_firewall import FORBIDDEN_KEYS, phi_scan

logger = logging.getLogger(__name__)

# Path to static JSON files
STATIC_DIR = Path(__file__).parent.parent / "static"


# =============================================================================
# Data loading utilities
# =============================================================================
_question_bank: Optional[Dict[str, Any]] = None
_habit_library: Optional[Dict[str, Any]] = None
_template_library: Optional[Dict[str, Any]] = None


def _load_json_file(filename: str) -> Dict[str, Any]:
    """Load JSON file from static directory"""
    filepath = STATIC_DIR / filename
    if not filepath.exists():
        logger.warning(f"Static file not found: {filepath}")
        return {}
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON file {filename}: {e}")
        return {}


def _get_question_bank() -> Dict[str, Any]:
    """Lazy load question bank"""
    global _question_bank
    if _question_bank is None:
        _question_bank = _load_json_file("question_bank.json")
        logger.info(f"Loaded question bank with {len(_question_bank)} questions")
    return _question_bank


def _get_habit_library() -> Dict[str, Any]:
    """Lazy load habit library"""
    global _habit_library
    if _habit_library is None:
        _habit_library = _load_json_file("habit_library.json")
        logger.info(f"Loaded habit library with {len(_habit_library)} habits")
    return _habit_library


def _get_template_library() -> Dict[str, Any]:
    """Lazy load template library"""
    global _template_library
    if _template_library is None:
        _template_library = _load_json_file("patient_templates.json")
        logger.info(f"Loaded template library with {len(_template_library)} templates")
    return _template_library


# =============================================================================
# B.19: get_questions(question_ids)
# =============================================================================
def get_questions(question_ids: List[str]) -> List[Dict[str, Any]]:
    """
    B.19: Retrieve questions by their IDs from the question bank.
    
    Args:
        question_ids: List of question IDs (e.g., ["Q001", "Q002"])
        
    Returns:
        List of question dictionaries with id, type, label, options, required
    """
    question_bank = _get_question_bank()
    result = []
    
    for q_id in question_ids:
        if q_id in question_bank:
            question = question_bank[q_id].copy()
            question["id"] = q_id
            result.append(question)
        else:
            logger.warning(f"Question not found: {q_id}")
    
    logger.debug(f"Retrieved {len(result)} of {len(question_ids)} requested questions")
    return result


def get_all_question_ids() -> List[str]:
    """Get all available question IDs"""
    return list(_get_question_bank().keys())


def get_questions_by_type(question_type: str) -> List[Dict[str, Any]]:
    """Get all questions of a specific type"""
    question_bank = _get_question_bank()
    result = []
    
    for q_id, question in question_bank.items():
        if question.get("type") == question_type:
            q = question.copy()
            q["id"] = q_id
            result.append(q)
    
    return result


# =============================================================================
# B.20: get_habits(habit_ids)
# =============================================================================
def get_habits(habit_ids: List[str]) -> List[Dict[str, Any]]:
    """
    B.20: Retrieve habits by their IDs from the habit library.
    
    Args:
        habit_ids: List of habit IDs (e.g., ["H001", "H002"])
        
    Returns:
        List of habit dictionaries with id, name, description, category, frequency_templates
    """
    habit_library = _get_habit_library()
    result = []
    
    for h_id in habit_ids:
        if h_id in habit_library:
            habit = habit_library[h_id].copy()
            habit["id"] = h_id
            result.append(habit)
        else:
            logger.warning(f"Habit not found: {h_id}")
    
    logger.debug(f"Retrieved {len(result)} of {len(habit_ids)} requested habits")
    return result


def get_all_habit_ids() -> List[str]:
    """Get all available habit IDs"""
    return list(_get_habit_library().keys())


def get_habits_by_category(category: str) -> List[Dict[str, Any]]:
    """Get all habits of a specific category"""
    habit_library = _get_habit_library()
    result = []
    
    for h_id, habit in habit_library.items():
        if habit.get("category") == category:
            h = habit.copy()
            h["id"] = h_id
            result.append(h)
    
    return result


# =============================================================================
# B.21: render_templates(template_ids, packet) - NO PHI output
# =============================================================================
# Safe placeholder keys that can be used in templates (NO PHI)
SAFE_PLACEHOLDER_KEYS: Set[str] = {
    "medication_time",
    "appointment_date",
    "appointment_time",
    "symptom_type",
    "goal_name",
    "completed_count",
    "total_count",
    "habit_name",
    "streak_days",
    "vital_type",
    "message_content",
    "medication_name",
    "tip_content",
    "activity_type",
    # Bucketed/anonymized values are safe
    "age_bucket",
    "risk_bucket",
    "adherence_bucket",
    "engagement_bucket",
}


def _sanitize_placeholder_value(key: str, value: Any) -> str:
    """
    Sanitize placeholder value to ensure NO PHI in output.
    
    Blocks:
    - Forbidden keys
    - Values that contain PHI patterns
    - Direct patient identifiers
    """
    # Reject forbidden keys
    if key.lower() in {k.lower() for k in FORBIDDEN_KEYS}:
        logger.warning(f"Blocked forbidden placeholder key: {key}")
        return "[REDACTED]"
    
    # Convert to string
    str_value = str(value) if value is not None else ""
    
    # PHI scan the value
    if phi_scan(str_value):
        logger.warning(f"PHI detected in placeholder value for key: {key}")
        return "[REDACTED]"
    
    return str_value


def render_templates(
    template_ids: List[str],
    packet: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    B.21: Render templates with packet data. NO PHI in output.
    
    This function:
    1. Loads templates by IDs
    2. Substitutes placeholders from packet
    3. Ensures NO PHI in rendered output via sanitization
    
    Args:
        template_ids: List of template IDs (e.g., ["T001", "T002"])
        packet: Dictionary with placeholder values (must be safe, no PHI)
        
    Returns:
        List of rendered template dictionaries with id, name, rendered_message, category
    """
    template_library = _get_template_library()
    result = []
    
    # Pre-sanitize all packet values
    safe_packet = {}
    for key, value in packet.items():
        safe_packet[key] = _sanitize_placeholder_value(key, value)
    
    for t_id in template_ids:
        if t_id not in template_library:
            logger.warning(f"Template not found: {t_id}")
            continue
        
        template = template_library[t_id]
        message = template.get("message", "")
        
        # Substitute placeholders using safe packet values
        for placeholder in template.get("placeholders", []):
            placeholder_pattern = "{" + placeholder + "}"
            if placeholder in safe_packet:
                message = message.replace(placeholder_pattern, safe_packet[placeholder])
            else:
                # Leave placeholder as-is if no value provided
                pass
        
        # Final PHI check on rendered message
        if phi_scan(message):
            logger.error(f"PHI detected in rendered template {t_id}, blocking output")
            message = "[Content blocked due to privacy concerns]"
        
        result.append({
            "id": t_id,
            "name": template.get("name", ""),
            "rendered_message": message,
            "category": template.get("category", ""),
        })
    
    logger.debug(f"Rendered {len(result)} of {len(template_ids)} requested templates")
    return result


def get_all_template_ids() -> List[str]:
    """Get all available template IDs"""
    return list(_get_template_library().keys())


def get_templates_by_category(category: str) -> List[Dict[str, Any]]:
    """Get all templates of a specific category"""
    template_library = _get_template_library()
    result = []
    
    for t_id, template in template_library.items():
        if template.get("category") == category:
            t = template.copy()
            t["id"] = t_id
            result.append(t)
    
    return result


# =============================================================================
# Reload functions for testing/updates
# =============================================================================
def reload_all():
    """Reload all static data (useful for testing or hot updates)"""
    global _question_bank, _habit_library, _template_library
    _question_bank = None
    _habit_library = None
    _template_library = None
    logger.info("Cleared all cached library data")


__all__ = [
    # B.19
    "get_questions",
    "get_all_question_ids",
    "get_questions_by_type",
    # B.20
    "get_habits",
    "get_all_habit_ids",
    "get_habits_by_category",
    # B.21
    "render_templates",
    "get_all_template_ids",
    "get_templates_by_category",
    "SAFE_PLACEHOLDER_KEYS",
    # Utilities
    "reload_all",
]
