"""
Personalized Recommendations Service
EHR-driven habit and activity recommendations based on patient conditions.
Maps conditions to evidence-based habits with reasons.
"""

import logging
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from datetime import datetime

from app.services.ehr_service import EHRService, get_ehr_service

logger = logging.getLogger(__name__)


CONDITION_HABIT_MAPPINGS: Dict[str, List[Dict[str, Any]]] = {
    "respiratory": [
        {
            "name": "Breathing Exercises",
            "description": "Practice diaphragmatic breathing for 5-10 minutes",
            "category": "respiratory",
            "frequency": "daily",
            "goalCount": 2,
            "reason": "Helps strengthen respiratory muscles and improve lung function",
            "safety_notes": "Stop if you feel dizzy. Contact provider if breathing worsens."
        },
        {
            "name": "Check Air Quality",
            "description": "Review local AQI before outdoor activities",
            "category": "environmental",
            "frequency": "daily",
            "goalCount": 1,
            "reason": "Avoid triggers that can worsen respiratory symptoms",
            "safety_notes": "Stay indoors when AQI exceeds 100."
        },
        {
            "name": "Inhaler Adherence Check",
            "description": "Take prescribed inhaler as directed",
            "category": "medication",
            "frequency": "daily",
            "goalCount": 2,
            "reason": "Consistent medication use prevents flare-ups",
            "safety_notes": "Contact provider if using rescue inhaler more than 2x/week."
        }
    ],
    "cardiac": [
        {
            "name": "Daily Weight Check",
            "description": "Weigh yourself each morning before eating",
            "category": "monitoring",
            "frequency": "daily",
            "goalCount": 1,
            "reason": "Sudden weight gain can indicate fluid retention",
            "safety_notes": "Contact provider if weight increases >3 lbs in a day or >5 lbs in a week."
        },
        {
            "name": "Blood Pressure Log",
            "description": "Measure and record blood pressure",
            "category": "monitoring",
            "frequency": "daily",
            "goalCount": 2,
            "reason": "Track cardiovascular health trends",
            "safety_notes": "Contact provider if BP consistently above 180/120 or below 90/60."
        },
        {
            "name": "Low-Sodium Meal",
            "description": "Track sodium intake and aim for <2000mg daily",
            "category": "nutrition",
            "frequency": "daily",
            "goalCount": 3,
            "reason": "Reduces fluid retention and blood pressure",
            "safety_notes": "Read food labels carefully."
        },
        {
            "name": "Gentle Walking",
            "description": "Take a 10-20 minute walk at comfortable pace",
            "category": "exercise",
            "frequency": "daily",
            "goalCount": 1,
            "reason": "Light activity supports heart health",
            "safety_notes": "Stop if you experience chest pain, severe shortness of breath, or dizziness."
        }
    ],
    "mental_health": [
        {
            "name": "CBT Thought Record",
            "description": "Complete a structured thought record when feeling distressed",
            "category": "mental_health",
            "frequency": "daily",
            "goalCount": 1,
            "reason": "Helps identify and reframe negative thought patterns",
            "safety_notes": "If having thoughts of self-harm, contact crisis line or provider immediately."
        },
        {
            "name": "Mindful Breathing",
            "description": "Practice 5 minutes of mindful breathing",
            "category": "mindfulness",
            "frequency": "daily",
            "goalCount": 2,
            "reason": "Reduces anxiety and stress response",
            "safety_notes": "Find a quiet, comfortable space."
        },
        {
            "name": "Daily Walk Outdoors",
            "description": "Take a 15-30 minute walk outside",
            "category": "exercise",
            "frequency": "daily",
            "goalCount": 1,
            "reason": "Exercise and sunlight improve mood",
            "safety_notes": "Walk with someone if feeling unsafe."
        },
        {
            "name": "Gratitude Journal",
            "description": "Write 3 things you're grateful for",
            "category": "journaling",
            "frequency": "daily",
            "goalCount": 1,
            "reason": "Shifts focus to positive aspects of life",
            "safety_notes": None
        },
        {
            "name": "Sleep Schedule",
            "description": "Go to bed and wake up at consistent times",
            "category": "sleep",
            "frequency": "daily",
            "goalCount": 1,
            "reason": "Regular sleep improves mood and mental clarity",
            "safety_notes": "Avoid screens 1 hour before bed."
        }
    ],
    "pain": [
        {
            "name": "Pain Diary",
            "description": "Log pain levels, triggers, and what helps",
            "category": "tracking",
            "frequency": "daily",
            "goalCount": 2,
            "reason": "Identify patterns and effective treatments",
            "safety_notes": "Share with provider at next visit."
        },
        {
            "name": "Gentle Stretching",
            "description": "5-10 minutes of gentle stretches",
            "category": "exercise",
            "frequency": "daily",
            "goalCount": 1,
            "reason": "Maintains flexibility and reduces stiffness",
            "safety_notes": "Never stretch to the point of pain."
        },
        {
            "name": "Hydration Check",
            "description": "Drink 8 glasses of water",
            "category": "hydration",
            "frequency": "daily",
            "goalCount": 8,
            "reason": "Dehydration can worsen muscle pain and headaches",
            "safety_notes": None
        },
        {
            "name": "Relaxation Practice",
            "description": "Practice progressive muscle relaxation",
            "category": "relaxation",
            "frequency": "daily",
            "goalCount": 1,
            "reason": "Reduces muscle tension and pain perception",
            "safety_notes": None
        }
    ],
    "metabolic": [
        {
            "name": "Blood Sugar Check",
            "description": "Monitor blood glucose as prescribed",
            "category": "monitoring",
            "frequency": "daily",
            "goalCount": 2,
            "reason": "Track metabolic control",
            "safety_notes": "Contact provider if readings consistently outside target range."
        },
        {
            "name": "Balanced Meals",
            "description": "Eat regular, balanced meals with protein and fiber",
            "category": "nutrition",
            "frequency": "daily",
            "goalCount": 3,
            "reason": "Stabilizes blood sugar levels",
            "safety_notes": "Avoid skipping meals."
        },
        {
            "name": "Post-Meal Walk",
            "description": "Take a 10-15 minute walk after meals",
            "category": "exercise",
            "frequency": "daily",
            "goalCount": 2,
            "reason": "Helps regulate blood sugar after eating",
            "safety_notes": None
        }
    ],
    "immune": [
        {
            "name": "Symptom Check",
            "description": "Log any new symptoms or changes",
            "category": "monitoring",
            "frequency": "daily",
            "goalCount": 1,
            "reason": "Early detection of flares or infections",
            "safety_notes": "Contact provider immediately for fever, unusual fatigue, or new symptoms."
        },
        {
            "name": "Medication Adherence",
            "description": "Take immunosuppressants as prescribed",
            "category": "medication",
            "frequency": "daily",
            "goalCount": 1,
            "reason": "Consistent medication prevents flares",
            "safety_notes": "Never skip doses without provider guidance."
        },
        {
            "name": "Rest Period",
            "description": "Take a 20-30 minute rest break",
            "category": "rest",
            "frequency": "daily",
            "goalCount": 1,
            "reason": "Fatigue management is essential",
            "safety_notes": "Listen to your body's signals."
        },
        {
            "name": "Hand Hygiene",
            "description": "Wash hands frequently, especially before meals",
            "category": "hygiene",
            "frequency": "daily",
            "goalCount": 5,
            "reason": "Reduces infection risk",
            "safety_notes": None
        }
    ],
    "gastrointestinal": [
        {
            "name": "Food Diary",
            "description": "Log meals and any digestive symptoms",
            "category": "tracking",
            "frequency": "daily",
            "goalCount": 3,
            "reason": "Identify food triggers",
            "safety_notes": "Share patterns with provider."
        },
        {
            "name": "Mindful Eating",
            "description": "Eat slowly and chew thoroughly",
            "category": "nutrition",
            "frequency": "daily",
            "goalCount": 3,
            "reason": "Improves digestion and reduces symptoms",
            "safety_notes": None
        },
        {
            "name": "Stress Management",
            "description": "Practice relaxation techniques",
            "category": "stress",
            "frequency": "daily",
            "goalCount": 1,
            "reason": "Stress often triggers GI symptoms",
            "safety_notes": None
        }
    ],
    "neurological": [
        {
            "name": "Medication Timing",
            "description": "Take neurological medications at exact times",
            "category": "medication",
            "frequency": "daily",
            "goalCount": 2,
            "reason": "Consistent timing maintains therapeutic levels",
            "safety_notes": "Never adjust doses without provider guidance."
        },
        {
            "name": "Sleep Hygiene",
            "description": "Maintain regular sleep schedule of 7-9 hours",
            "category": "sleep",
            "frequency": "daily",
            "goalCount": 1,
            "reason": "Sleep is crucial for neurological health",
            "safety_notes": None
        },
        {
            "name": "Symptom Log",
            "description": "Record any neurological symptoms or changes",
            "category": "tracking",
            "frequency": "daily",
            "goalCount": 1,
            "reason": "Track patterns and treatment effectiveness",
            "safety_notes": "Report new or worsening symptoms immediately."
        }
    ],
    "general": [
        {
            "name": "Daily Hydration",
            "description": "Drink 8 glasses of water",
            "category": "hydration",
            "frequency": "daily",
            "goalCount": 8,
            "reason": "Supports overall health",
            "safety_notes": None
        },
        {
            "name": "Movement Break",
            "description": "Take short movement breaks every hour",
            "category": "exercise",
            "frequency": "daily",
            "goalCount": 4,
            "reason": "Reduces sedentary behavior",
            "safety_notes": None
        },
        {
            "name": "Medication Check",
            "description": "Take all prescribed medications",
            "category": "medication",
            "frequency": "daily",
            "goalCount": 1,
            "reason": "Medication adherence is crucial",
            "safety_notes": None
        }
    ]
}


class PersonalizedRecommendationsService:
    """
    Generates personalized habit recommendations based on patient EHR data.
    """
    
    def __init__(self, db: Session):
        self.db = db
        self.ehr_service = get_ehr_service(db)
    
    async def get_recommendations(
        self,
        patient_id: str,
        accessor_id: str,
        max_recommendations: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Generate personalized recommendations based on patient's conditions.
        
        Returns list of habit recommendations with:
        - name, description, category, frequency, goalCount
        - reason: Why this is recommended based on their conditions
        - safety_notes: When to contact provider
        """
        problems = await self.ehr_service.get_problem_list(patient_id, accessor_id)
        complaints = await self.ehr_service.get_recent_complaints(patient_id, accessor_id, days=90)
        medications = await self.ehr_service.get_medications(patient_id, accessor_id)
        
        categories = set()
        for problem in problems:
            category = problem.get("category", "general")
            categories.add(category)
        
        for complaint in complaints:
            category = complaint.get("category", "general")
            categories.add(category)
        
        if not categories:
            categories.add("general")
        
        recommendations = []
        seen_names = set()
        
        for category in categories:
            category_habits = CONDITION_HABIT_MAPPINGS.get(category, [])
            for habit in category_habits:
                if habit["name"] not in seen_names:
                    seen_names.add(habit["name"])
                    
                    enhanced_habit = habit.copy()
                    
                    relevant_conditions = [
                        p["name"] for p in problems 
                        if p.get("category") == category
                    ]
                    if relevant_conditions:
                        condition_names = ", ".join(relevant_conditions[:2])
                        enhanced_habit["reason"] = f"Based on {condition_names}: {habit['reason']}"
                    
                    recommendations.append(enhanced_habit)
        
        recommendations.sort(key=lambda x: (
            1 if x.get("category") in ["medication", "monitoring"] else 2,
            x.get("name", "")
        ))
        
        return recommendations[:max_recommendations]
    
    async def get_autopilot_suggestions(
        self,
        patient_id: str,
        accessor_id: str
    ) -> Dict[str, Any]:
        """
        Generate personalized daily followup suggestions based on EHR.
        
        Returns:
        - items: List of suggested questions/checks
        - metadata: Source and generation info
        """
        problems = await self.ehr_service.get_problem_list(patient_id, accessor_id)
        complaints = await self.ehr_service.get_recent_complaints(patient_id, accessor_id, days=30)
        medications = await self.ehr_service.get_medications(patient_id, accessor_id)
        
        suggestions = []
        
        categories = set()
        for problem in problems:
            categories.add(problem.get("category", "general"))
        
        base_questions = [
            {
                "id": "general_wellness",
                "question": "How are you feeling overall today?",
                "type": "scale",
                "reason": "Daily wellness check",
                "severity": "low"
            },
            {
                "id": "energy_level",
                "question": "How is your energy level today?",
                "type": "scale",
                "reason": "Track fatigue patterns",
                "severity": "low"
            }
        ]
        suggestions.extend(base_questions)
        
        if "respiratory" in categories:
            suggestions.extend([
                {
                    "id": "breathing_difficulty",
                    "question": "Have you experienced any breathing difficulties today?",
                    "type": "yesno",
                    "reason": "Based on respiratory condition",
                    "severity": "medium"
                },
                {
                    "id": "rescue_inhaler",
                    "question": "How many times did you use your rescue inhaler?",
                    "type": "metric",
                    "reason": "Monitor asthma control",
                    "severity": "medium"
                }
            ])
        
        if "cardiac" in categories:
            suggestions.extend([
                {
                    "id": "weight_today",
                    "question": "What is your weight today (lbs)?",
                    "type": "metric",
                    "reason": "Based on cardiac condition - monitor fluid retention",
                    "severity": "high"
                },
                {
                    "id": "swelling",
                    "question": "Have you noticed any swelling in your legs or ankles?",
                    "type": "yesno",
                    "reason": "Based on cardiac condition",
                    "severity": "high"
                }
            ])
        
        if "mental_health" in categories:
            suggestions.extend([
                {
                    "id": "mood_today",
                    "question": "How would you rate your mood today?",
                    "type": "scale",
                    "reason": "Based on mental health history",
                    "severity": "medium"
                },
                {
                    "id": "sleep_quality",
                    "question": "How well did you sleep last night?",
                    "type": "scale",
                    "reason": "Sleep affects mental health",
                    "severity": "low"
                }
            ])
        
        if "pain" in categories:
            suggestions.extend([
                {
                    "id": "pain_level",
                    "question": "What is your pain level today (0-10)?",
                    "type": "scale",
                    "reason": "Based on pain history",
                    "severity": "medium"
                }
            ])
        
        if "metabolic" in categories:
            suggestions.extend([
                {
                    "id": "blood_sugar",
                    "question": "What was your fasting blood sugar this morning?",
                    "type": "metric",
                    "reason": "Based on metabolic condition",
                    "severity": "medium"
                }
            ])
        
        if "immune" in categories:
            suggestions.extend([
                {
                    "id": "fever_check",
                    "question": "Have you had any fever or chills?",
                    "type": "yesno",
                    "reason": "Based on immunocompromised status - infection monitoring",
                    "severity": "high"
                },
                {
                    "id": "new_symptoms",
                    "question": "Have you noticed any new or unusual symptoms?",
                    "type": "yesno",
                    "reason": "Early detection is critical for immunocompromised patients",
                    "severity": "high"
                }
            ])
        
        if medications:
            suggestions.append({
                "id": "medication_adherence",
                "question": "Did you take all your medications as prescribed today?",
                "type": "yesno",
                "reason": f"You have {len(medications)} active medication(s)",
                "severity": "medium"
            })
        
        return {
            "items": suggestions,
            "metadata": {
                "source": "ehr_personalized",
                "generated_at": datetime.utcnow().isoformat(),
                "conditions_count": len(problems),
                "categories": list(categories)
            }
        }


def get_personalized_recommendations_service(db: Session) -> PersonalizedRecommendationsService:
    """Factory function."""
    return PersonalizedRecommendationsService(db)
