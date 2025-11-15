"""
Symptom Extraction Service for Agent Clona Integration.
Automatically detects and logs symptoms mentioned in chat conversations.
"""

from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from datetime import datetime
import json

from app.config import get_openai_client, check_openai_baa_compliance
from app.models.medication_side_effects import SymptomLog, SymptomSource


class SymptomExtractionService:
    """Service for extracting symptoms from Agent Clona conversations"""
    
    EXTRACTION_PROMPT = """You are a medical symptom extraction assistant. Analyze the patient's message and extract any symptoms they mention.

For each symptom, identify:
1. symptom_type: The specific symptom (e.g., "headache", "nausea", "fever", "fatigue")
2. severity: mild, moderate, or severe
3. description: Brief description including any details (onset, duration, characteristics)

Return ONLY a JSON array of symptoms. If no symptoms are mentioned, return an empty array.

Example input: "I've been having terrible headaches for 3 days and feeling really nauseous"
Example output:
[
  {
    "symptom_type": "headache",
    "severity": "severe",
    "description": "Terrible headaches lasting 3 days"
  },
  {
    "symptom_type": "nausea",
    "severity": "moderate",
    "description": "Feeling really nauseous"
  }
]

Patient message: {message}
"""
    
    @staticmethod
    def extract_symptoms_from_message(
        patient_message: str,
        ai_response: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """
        Extract symptoms mentioned in a patient's message using OpenAI.
        
        HIPAA COMPLIANCE: Requires OpenAI BAA.
        
        Args:
            patient_message: The message from the patient
            ai_response: Optional AI response (for additional context)
            
        Returns:
            List of extracted symptoms with type, severity, and description
        """
        try:
            check_openai_baa_compliance()
            client = get_openai_client()
            
            # Create extraction prompt
            prompt = SymptomExtractionService.EXTRACTION_PROMPT.format(
                message=patient_message
            )
            
            # Call OpenAI for symptom extraction
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a medical symptom extraction assistant. Return only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,  # Low temperature for consistent extraction
                max_tokens=500
            )
            
            content = response.choices[0].message.content
            if content is None:
                return []
            content = content.strip()
            
            # Parse JSON response
            # Remove markdown code blocks if present
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            
            symptoms = json.loads(content)
            
            # Validate structure
            if isinstance(symptoms, list):
                validated = []
                for symptom in symptoms:
                    if isinstance(symptom, dict) and "symptom_type" in symptom:
                        validated.append({
                            "symptom_type": symptom.get("symptom_type", "unknown"),
                            "severity": symptom.get("severity", "moderate"),
                            "description": symptom.get("description", "")
                        })
                return validated
            
            return []
            
        except Exception as e:
            print(f"Error extracting symptoms: {e}")
            return []
    
    @staticmethod
    def _severity_to_int(severity_str: str) -> int:
        """Convert text severity to 1-10 integer scale"""
        severity_map = {
            "mild": 3,
            "moderate": 5,
            "severe": 8,
            "low": 3,
            "medium": 5,
            "high": 8
        }
        return severity_map.get(severity_str.lower(), 5)  # Default to moderate
    
    @staticmethod
    def save_extracted_symptoms(
        db: Session,
        patient_id: str,
        symptoms: List[Dict[str, str]]
    ) -> List[SymptomLog]:
        """
        Save extracted symptoms to the database as SymptomLog entries.
        
        FIX: Uses correct schema fields (symptom_name, symptom_description, integer severity)
        
        Args:
            db: Database session
            patient_id: ID of the patient
            symptoms: List of extracted symptoms with symptom_type, severity, description
            
        Returns:
            List of created SymptomLog objects
        """
        created_logs = []
        
        for symptom in symptoms:
            try:
                # FIX: Map extracted fields to database schema
                symptom_log = SymptomLog(
                    patient_id=patient_id,
                    symptom_name=symptom["symptom_type"],  # symptom_type → symptom_name
                    symptom_description=symptom.get("description", ""),  # description → symptom_description
                    severity=SymptomExtractionService._severity_to_int(symptom["severity"]),  # Convert to integer
                    source=SymptomSource.AGENT_CLONA,  # Use enum, not string
                    reported_at=datetime.utcnow()  # FIX: reported_at, not logged_at
                )
                
                db.add(symptom_log)
                created_logs.append(symptom_log)
                
            except Exception as e:
                print(f"Error saving symptom log: {e}")
                continue
        
        if created_logs:
            try:
                db.commit()
            except Exception as e:
                print(f"Error committing symptom logs: {e}")
                db.rollback()
                return []
        
        return created_logs
    
    @staticmethod
    def extract_and_save_symptoms(
        db: Session,
        patient_id: str,
        patient_message: str,
        ai_response: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Complete workflow: Extract symptoms from message and save to database.
        
        Args:
            db: Database session
            patient_id: ID of the patient
            patient_message: Message from patient
            ai_response: Optional AI response for context
            
        Returns:
            Dictionary with extraction results and saved logs
        """
        # Extract symptoms
        symptoms = SymptomExtractionService.extract_symptoms_from_message(
            patient_message=patient_message,
            ai_response=ai_response
        )
        
        # Save to database
        saved_logs = []
        if symptoms:
            saved_logs = SymptomExtractionService.save_extracted_symptoms(
                db=db,
                patient_id=patient_id,
                symptoms=symptoms
            )
        
        return {
            "symptoms_extracted": len(symptoms),
            "symptoms_saved": len(saved_logs),
            "symptoms": symptoms,
            "saved_logs": [
                {
                    "id": log.id,
                    "symptom_name": log.symptom_name,
                    "severity": log.severity,
                    "reported_at": log.reported_at.isoformat()  # FIX: reported_at, not logged_at
                }
                for log in saved_logs
            ]
        }
