"""
Enhanced Agent Clona Service for Symptom Analysis.
Provides differential diagnosis, doctor suggestions, lab test recommendations.
"""

from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
import json
from datetime import datetime

from app.config import settings, get_openai_client
from app.models.user import User
from app.services.doctor_search_service import DoctorSearchService


class AgentClonaService:
    """Enhanced AI service for patient symptom analysis and doctor recommendations"""
    
    SYSTEM_PROMPT = """You are Agent Clona, a compassionate AI health assistant for immunocompromised patients. 
Your role is to:
1. Listen empathetically to patient symptoms
2. Ask clarifying questions for differential diagnosis
3. Recommend appropriate lab tests and physical examinations
4. Suggest treatment approaches (medication, lifestyle changes)
5. Recommend specialists when needed

IMPORTANT GUIDELINES:
- Always be empathetic and supportive
- Ask specific follow-up questions about symptoms (onset, duration, severity, triggers)
- Consider immune system status when making recommendations
- NEVER diagnose definitively - always suggest consulting with healthcare providers
- Provide clear, actionable recommendations
- When suggesting specialists, explain WHY that specialty is appropriate

For symptom analysis, structure your responses like this:
1. Empathetic acknowledgment
2. Clarifying questions (if needed)
3. Possible conditions to consider (differential diagnosis)
4. Recommended lab tests/physical exams
5. Suggested specialists (if applicable)
6. General treatment recommendations
7. Urgency level (routine, soon, urgent, emergency)
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
        
        Args:
            db: Database session
            patient: Patient user object
            symptom_description: Current symptom description from patient
            conversation_history: Previous messages in the conversation
            
        Returns:
            Dictionary with analysis, recommendations, and suggested doctors
        """
        client = get_openai_client()
        
        messages = [{"role": "system", "content": AgentClonaService.SYSTEM_PROMPT}]
        
        messages.extend(conversation_history)
        
        messages.append({
            "role": "user",
            "content": f"""Patient Context:
- Location: {patient.location_city}, {patient.location_state}
- Immunocompromised status: Yes

Current Symptoms/Question:
{symptom_description}

Please provide:
1. Empathetic response
2. Follow-up questions (if needed)
3. Possible conditions (differential diagnosis)
4. Recommended lab tests or physical exams
5. Suggested medical specialties to consult
6. Treatment recommendations
7. Urgency level

If you suggest a medical specialty, structure your response to include a JSON block with this format:
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
        
        return {
            "ai_response": ai_response,
            "structured_recommendations": structured_data,
            "suggested_doctors": suggested_doctors,
            "timestamp": datetime.utcnow().isoformat()
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
        
        Args:
            symptom_category: Main category (e.g., "respiratory", "cardiac", "digestive")
            initial_symptoms: List of symptoms patient mentioned
            
        Returns:
            List of follow-up questions
        """
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
        
        Args:
            symptoms: List of patient symptoms
            patient_history: Optional medical history
            
        Returns:
            Dictionary with recommended tests and reasoning
        """
        client = get_openai_client()
        
        history_context = ""
        if patient_history:
            history_context = f"\nPatient History: {json.dumps(patient_history)}"
        
        prompt = f"""For an immunocompromised patient with symptoms: {', '.join(symptoms)}{history_context}

Recommend appropriate lab tests and physical examinations. For each recommendation, explain:
1. What the test checks for
2. Why it's relevant to these symptoms
3. Priority level (routine, important, urgent)

Format as JSON:
{{
  "lab_tests": [
    {{"name": "Complete Blood Count", "reason": "...", "priority": "important"}}
  ],
  "physical_exams": [
    {{"name": "Blood Pressure", "reason": "...", "priority": "routine"}}
  ],
  "imaging": [
    {{"name": "Chest X-ray", "reason": "...", "priority": "important"}}
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
        
        Args:
            diagnosis_considerations: Possible conditions being considered
            patient_medications: Current medications (to check interactions)
            
        Returns:
            Treatment suggestions with medication, lifestyle, and monitoring recommendations
        """
        client = get_openai_client()
        
        meds_context = ""
        if patient_medications:
            meds_context = f"\nCurrent Medications: {', '.join(patient_medications)}"
        
        prompt = f"""For an immunocompromised patient with possible conditions: {', '.join(diagnosis_considerations)}{meds_context}

Provide treatment approach suggestions INCLUDING:
1. Medication options (with drug interaction warnings if applicable)
2. Lifestyle modifications
3. Monitoring recommendations
4. When to seek immediate care

IMPORTANT: These are suggestions to discuss with their doctor, NOT prescriptions.

Format as JSON:
{{
  "medication_suggestions": [
    {{"name": "...", "purpose": "...", "considerations": "..."}}
  ],
  "lifestyle_modifications": ["...", "..."],
  "monitoring": ["...", "..."],
  "red_flags": ["Seek immediate care if..."]
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
