"""
Enhanced Agent Clona Service for Symptom Analysis.
Provides differential diagnosis, doctor suggestions, lab test recommendations.
Integrated with Red Flag Detection for real-time emergency escalation.
"""

from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
import json
import logging
from datetime import datetime

from app.config import settings, get_openai_client, check_openai_baa_compliance
from app.models.user import User
from app.services.doctor_search_service import DoctorSearchService

logger = logging.getLogger(__name__)


class AgentClonaService:
    """Enhanced AI service for patient symptom analysis and doctor recommendations"""
    
    SYSTEM_PROMPT = """You are Agent Clona, Your Health Companion - a wellness monitoring AI designed to support immunocompromised patients through change detection and health pattern tracking.

CRITICAL REGULATORY DISCLAIMER:
You are NOT a medical device. You do NOT provide medical diagnosis, treatment, or clinical decisions. 
Your role is STRICTLY limited to:
1. Wellness monitoring and change detection
2. Health pattern tracking for personal awareness
3. Informational support to discuss with healthcare providers

You must ALWAYS include this disclaimer in your responses:
"⚠️ Important: I'm your wellness companion, not a medical professional. This information is for personal tracking and discussion with your healthcare provider. All medical decisions remain your responsibility."

Your wellness support role includes:
1. Listen empathetically to patient health changes and patterns
2. Ask clarifying questions to understand symptom patterns better
3. Identify health pattern changes that may warrant healthcare provider discussion
4. Suggest discussing specific tests or examinations with their doctor
5. Recommend consulting appropriate medical specialists for evaluation
6. Provide wellness insights and lifestyle considerations

STRICT REGULATORY GUIDELINES:
- ALWAYS be empathetic and supportive
- Ask specific follow-up questions about patterns (onset, duration, progression, triggers)
- Consider immune system status when providing wellness insights
- NEVER diagnose medical conditions - you detect changes and patterns only
- NEVER prescribe treatments - you suggest wellness approaches to discuss with their doctor
- ALWAYS emphasize that all insights must be discussed with their healthcare provider
- Frame recommendations as "information to share with your doctor" NOT medical advice
- When suggesting specialists, say "you may want to discuss consulting [specialty] with your doctor"

For wellness pattern analysis, structure your responses like this:
1. Wellness companion disclaimer (see above)
2. Empathetic acknowledgment of their concern
3. Clarifying questions about the pattern or change (if needed)
4. Health pattern observations (NOT diagnosis - describe changes detected)
5. Suggested topics to discuss with their healthcare provider (tests, exams, specialist consultation)
6. Wellness considerations and lifestyle factors
7. Urgency guidance (discuss at next visit, schedule appointment soon, seek prompt evaluation, seek immediate care)
"""
    
    @staticmethod
    def analyze_symptoms(
        db: Session,
        patient: User,
        symptom_description: str,
        conversation_history: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """
        Analyze patient symptoms and provide recommendations.
        Integrates with Red Flag Detection for real-time emergency escalation.
        
        HIPAA COMPLIANCE: Only works if OpenAI BAA is signed.
        PHI (Patient Health Information) is transmitted to OpenAI.
        
        Args:
            db: Database session
            patient: Patient user object
            symptom_description: Current symptom description from patient
            conversation_history: Previous messages in the conversation
            
        Returns:
            Dictionary with analysis, recommendations, suggested doctors, and red flag info
        """
        check_openai_baa_compliance()
        client = get_openai_client()
        
        red_flag_result = None
        escalation_triggered = False
        
        try:
            from app.services.red_flag_detection_service import get_red_flag_detection_service
            red_flag_service = get_red_flag_detection_service()
            
            full_conversation_text = symptom_description
            for msg in conversation_history:
                if msg.get("role") == "user":
                    full_conversation_text = f"{msg.get('content', '')} {full_conversation_text}"
            
            red_flag_result = red_flag_service.detect_red_flags(
                text=full_conversation_text,
                patient_id=str(patient.id),
                use_ai_analysis=True,
                context={"source": "clona_chat", "message": symptom_description}
            )
            
        except Exception as e:
            logger.warning(f"[Clona] Red flag detection failed: {e}")
            red_flag_result = None
        
        messages = [{"role": "system", "content": AgentClonaService.SYSTEM_PROMPT}]
        
        messages.extend(conversation_history)
        
        messages.append({
            "role": "user",
            "content": f"""Patient Context:
- Location: {patient.location_city}, {patient.location_state}
- Immunocompromised status: Yes

Current Health Pattern/Question:
{symptom_description}

Please provide wellness companion response including:
1. Wellness companion disclaimer (required in every response)
2. Empathetic acknowledgment
3. Follow-up questions about the pattern (if needed)
4. Health pattern observations (change detection - NOT diagnosis)
5. Topics to discuss with their healthcare provider (tests, exams, specialist consideration)
6. Wellness and lifestyle considerations
7. Urgency guidance for healthcare provider consultation

REMEMBER: Frame as "wellness insights to discuss with your doctor" NOT medical advice or diagnosis.

If you suggest medical specialty consultation, structure your response to include a JSON block with this format:
{{
  "suggested_specialties": ["Cardiology", "Internal Medicine"],
  "urgency": "routine|soon|urgent|emergency",
  "lab_tests": ["Complete Blood Count", "Chest X-ray"],
  "physical_exams": ["Blood pressure measurement", "Heart auscultation"]
}}
"""
        })
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.7,
            max_tokens=1000
        )
        
        ai_response = response.choices[0].message.content
        
        suggested_doctors = []
        structured_data = AgentClonaService._extract_structured_data(ai_response)
        
        if structured_data and "suggested_specialties" in structured_data:
            for specialty in structured_data["suggested_specialties"]:
                doctors = DoctorSearchService.suggest_doctors_by_specialty(
                    db=db,
                    patient_location_city=patient.location_city,
                    patient_location_state=patient.location_state,
                    specialty=specialty,
                    limit=3
                )
                suggested_doctors.extend(doctors)
        
        red_flag_info = None
        if red_flag_result and hasattr(red_flag_result, 'detected') and red_flag_result.detected:
            escalation_triggered = True
            categories_list = [cat.value if hasattr(cat, 'value') else str(cat) for cat in red_flag_result.categories]
            red_flag_info = {
                "detected": True,
                "severity": red_flag_result.highest_severity.value if red_flag_result.highest_severity else None,
                "categories": categories_list,
                "symptoms": red_flag_result.symptoms,
                "escalation_type": red_flag_result.escalation_type.value if red_flag_result.escalation_type else None,
                "emergency_instructions": red_flag_result.emergency_instructions,
                "confidence": red_flag_result.confidence_score
            }
            logger.warning(
                f"[Clona] RED FLAG DETECTED for patient {patient.id}: "
                f"severity={red_flag_result.highest_severity}, categories={categories_list}"
            )
        
        return {
            "ai_response": ai_response,
            "structured_recommendations": structured_data,
            "suggested_doctors": suggested_doctors,
            "timestamp": datetime.utcnow().isoformat(),
            "red_flag_detection": red_flag_info,
            "escalation_triggered": escalation_triggered
        }
    
    @staticmethod
    def _extract_structured_data(ai_response: str) -> Optional[Dict[str, Any]]:
        """
        Extract structured JSON data from AI response.
        Returns None if no valid JSON found.
        """
        try:
            start_idx = ai_response.find("{")
            end_idx = ai_response.rfind("}") + 1
            
            if start_idx != -1 and end_idx > start_idx:
                json_str = ai_response[start_idx:end_idx]
                return json.loads(json_str)
        except (json.JSONDecodeError, ValueError):
            pass
        
        return None
    
    @staticmethod
    def generate_differential_diagnosis_questions(
        symptom_category: str,
        initial_symptoms: List[str]
    ) -> List[str]:
        """
        Generate targeted questions for differential diagnosis.
        
        HIPAA COMPLIANCE: Requires OpenAI BAA.
        
        Args:
            symptom_category: Main category (e.g., "respiratory", "cardiac", "digestive")
            initial_symptoms: List of symptoms patient mentioned
            
        Returns:
            List of follow-up questions
        """
        check_openai_baa_compliance()
        client = get_openai_client()
        
        prompt = f"""Given a patient with immunocompromised status presenting with {symptom_category} symptoms:
Initial symptoms: {', '.join(initial_symptoms)}

Generate 5-7 specific follow-up questions to help narrow differential diagnosis. Questions should ask about:
- Onset and duration
- Severity and progression
- Associated symptoms
- Triggers or relieving factors
- Impact on daily activities
- Previous occurrences

Format as a JSON array of strings."""
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=500
        )
        
        try:
            questions = json.loads(response.choices[0].message.content)
            if isinstance(questions, list):
                return questions
        except json.JSONDecodeError:
            pass
        
        return [
            "When did these symptoms first begin?",
            "How severe are your symptoms on a scale of 1-10?",
            "Have you noticed any patterns or triggers?",
            "Are there any other symptoms you're experiencing?",
            "How are these symptoms affecting your daily life?"
        ]
    
    @staticmethod
    def recommend_lab_tests(
        symptoms: List[str],
        patient_history: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Recommend appropriate lab tests based on symptoms.
        
        HIPAA COMPLIANCE: Requires OpenAI BAA. PHI transmitted.
        
        Args:
            symptoms: List of patient symptoms
            patient_history: Optional medical history
            
        Returns:
            Dictionary with recommended tests and reasoning
        """
        check_openai_baa_compliance()
        client = get_openai_client()
        
        history_context = ""
        if patient_history:
            history_context = f"\nPatient History: {json.dumps(patient_history)}"
        
        prompt = f"""For an immunocompromised patient with health pattern changes: {', '.join(symptoms)}{history_context}

REGULATORY CONTEXT: You are a wellness monitoring tool, NOT a medical device. Provide informational suggestions for the patient to DISCUSS with their healthcare provider.

Suggest topics for healthcare provider discussion regarding tests and examinations. For each suggestion, explain:
1. What the test/exam typically evaluates
2. Why it may be relevant to discuss for these pattern changes
3. Suggested discussion priority (routine checkup, schedule appointment, prompt consultation)

Frame all suggestions as "you may want to ask your doctor about [test]" NOT medical recommendations.

Format as JSON:
{{
  "lab_tests": [
    {{"name": "Complete Blood Count", "reason": "...", "priority": "routine|important|urgent"}}
  ],
  "physical_exams": [
    {{"name": "Blood Pressure", "reason": "...", "priority": "routine|important|urgent"}}
  ],
  "imaging": [
    {{"name": "Chest X-ray", "reason": "...", "priority": "routine|important|urgent"}}
  ]
}}
"""
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=800
        )
        
        try:
            recommendations = json.loads(response.choices[0].message.content)
            return recommendations
        except json.JSONDecodeError:
            return {
                "lab_tests": [],
                "physical_exams": [],
                "imaging": [],
                "error": "Unable to parse recommendations"
            }
    
    @staticmethod
    def suggest_treatment_approach(
        diagnosis_considerations: List[str],
        patient_medications: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Suggest treatment approaches based on possible conditions.
        
        HIPAA COMPLIANCE: Requires OpenAI BAA. PHI transmitted.
        
        Args:
            diagnosis_considerations: Possible conditions being considered
            patient_medications: Current medications (to check interactions)
            
        Returns:
            Treatment suggestions with medication, lifestyle, and monitoring recommendations
        """
        check_openai_baa_compliance()
        client = get_openai_client()
        
        meds_context = ""
        if patient_medications:
            meds_context = f"\nCurrent Medications: {', '.join(patient_medications)}"
        
        prompt = f"""For an immunocompromised patient with observed health pattern changes: {', '.join(diagnosis_considerations)}{meds_context}

CRITICAL REGULATORY CONTEXT: You are a wellness monitoring tool, NOT a medical device. You do NOT diagnose or prescribe.

Provide wellness insights and topics to DISCUSS with their healthcare provider, INCLUDING:
1. Wellness approaches to ask their doctor about (mention potential medication options as discussion topics only - NOT prescriptions)
2. Lifestyle modifications for general wellness
3. Self-monitoring suggestions for pattern tracking
4. Situations that warrant prompt healthcare provider consultation

STRICT REQUIREMENT: Frame ALL suggestions as "topics to discuss with your healthcare provider" or "questions to ask your doctor" - NOT medical advice or treatment recommendations.

Format as JSON:
{{
  "medication_discussion_topics": [
    {{"topic": "Ask your doctor about...", "purpose": "...", "considerations": "..."}}
  ],
  "lifestyle_wellness_suggestions": ["...", "..."],
  "self_monitoring_guidance": ["...", "..."],
  "when_to_consult_provider": ["Schedule appointment if...", "Seek prompt evaluation if...", "Seek immediate care if..."]
}}
"""
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6,
            max_tokens=1000
        )
        
        try:
            suggestions = json.loads(response.choices[0].message.content)
            return suggestions
        except json.JSONDecodeError:
            return {
                "medication_suggestions": [],
                "lifestyle_modifications": [],
                "monitoring": [],
                "red_flags": [],
                "error": "Unable to parse treatment suggestions"
            }
