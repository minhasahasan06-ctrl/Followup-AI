"""
Universal Red Flag Detection Service

Production-grade medical emergency detection service for real-time patient safety.
Detects critical symptoms during conversations and triggers appropriate escalation.

HIPAA Compliance:
- All detections are audit logged
- PHI handled securely with encryption
- Consent verification before sharing with connected doctors

Safety Focus:
- Real-time symptom analysis during Clona conversations
- Comprehensive medical emergency taxonomy
- Severity-based escalation routing
- Multi-channel notification delivery
"""

import logging
import re
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Tuple
from enum import Enum
from dataclasses import dataclass, field
from sqlalchemy.orm import Session

from app.config import settings, get_openai_client, check_openai_baa_compliance
from app.models.security_models import AuditLog

logger = logging.getLogger(__name__)


class RedFlagCategory(str, Enum):
    """Categories of medical red flags"""
    CARDIOVASCULAR = "cardiovascular"
    RESPIRATORY = "respiratory"
    NEUROLOGICAL = "neurological"
    ALLERGIC = "allergic"
    HEMORRHAGIC = "hemorrhagic"
    INFECTIOUS = "infectious"
    METABOLIC = "metabolic"
    TRAUMA = "trauma"
    MENTAL_HEALTH = "mental_health"
    GASTROINTESTINAL = "gastrointestinal"
    RENAL = "renal"
    OTHER = "other"


class RedFlagSeverity(str, Enum):
    """Severity levels for red flags"""
    CRITICAL = "critical"      # Immediate emergency (call 911)
    HIGH = "high"              # Urgent - contact doctor immediately
    MODERATE = "moderate"      # Soon - schedule urgent appointment
    LOW = "low"                # Monitor - discuss at next visit


class EscalationType(str, Enum):
    """Types of escalation actions"""
    EMERGENCY_911 = "emergency_911"
    IMMEDIATE_DOCTOR = "immediate_doctor"
    URGENT_APPOINTMENT = "urgent_appointment"
    ROUTINE_FOLLOWUP = "routine_followup"
    MONITOR_AND_LOG = "monitor_and_log"


@dataclass
class RedFlagSymptom:
    """Individual red flag symptom definition"""
    name: str
    category: RedFlagCategory
    severity: RedFlagSeverity
    keywords: List[str]
    patterns: List[str]
    escalation_type: EscalationType
    description: str
    recommended_actions: List[str]
    emergency_instructions: Optional[str] = None


@dataclass
class RedFlagDetection:
    """Result of a red flag detection"""
    detected: bool
    symptoms: List[Dict[str, Any]] = field(default_factory=list)
    highest_severity: Optional[RedFlagSeverity] = None
    escalation_type: Optional[EscalationType] = None
    categories: List[RedFlagCategory] = field(default_factory=list)
    confidence_score: float = 0.0
    recommended_actions: List[str] = field(default_factory=list)
    emergency_instructions: Optional[str] = None
    ai_analysis: Optional[str] = None
    detected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class RedFlagDetectionService:
    """
    Universal red flag detection service for medical emergencies.
    
    Capabilities:
    1. Pattern-based detection for common emergency symptoms
    2. AI-powered analysis for complex symptom combinations
    3. Severity scoring and escalation routing
    4. Real-time detection during patient conversations
    5. HIPAA-compliant audit logging
    """
    
    RED_FLAG_TAXONOMY: List[RedFlagSymptom] = [
        RedFlagSymptom(
            name="Chest Pain",
            category=RedFlagCategory.CARDIOVASCULAR,
            severity=RedFlagSeverity.CRITICAL,
            keywords=["chest pain", "chest tightness", "chest pressure", "chest discomfort",
                     "pain in chest", "heart pain", "squeezing chest", "crushing chest"],
            patterns=[
                r"\b(chest|heart)\s*(pain|ache|hurt|tight|pressure|squeezing|crushing|discomfort)\b",
                r"\b(pain|ache|pressure|tightness)\s*(in|on|around)\s*(my\s*)?(chest|heart)\b",
                r"\b(feel|feeling|felt)\s*(pain|pressure|tightness)\s*(in|on)\s*(my\s*)?chest\b"
            ],
            escalation_type=EscalationType.EMERGENCY_911,
            description="Chest pain may indicate heart attack or other cardiac emergency",
            recommended_actions=[
                "Call 911 immediately if pain is severe or accompanied by other symptoms",
                "Stop all physical activity and rest",
                "Chew aspirin if not allergic and available",
                "Do not drive yourself to the hospital"
            ],
            emergency_instructions="If experiencing severe chest pain, call 911 immediately. Do not wait."
        ),
        RedFlagSymptom(
            name="Difficulty Breathing",
            category=RedFlagCategory.RESPIRATORY,
            severity=RedFlagSeverity.CRITICAL,
            keywords=["can't breathe", "difficulty breathing", "hard to breathe", "shortness of breath",
                     "struggling to breathe", "gasping", "choking", "suffocating", "breathless",
                     "trouble breathing", "breathing problem", "labored breathing"],
            patterns=[
                r"\b(can't|cannot|can\s*not|unable\s*to|hard\s*to|difficult\s*to|struggling\s*to|trouble)\s*(breathe|breathing)\b",
                r"\b(short(ness)?\s*of\s*breath|breathless|gasping|choking|suffocating)\b",
                r"\b(breathing)\s*(is\s*)?(hard|difficult|labored|rapid|shallow)\b",
                r"\b(feel|feeling)\s*(like\s*)?(I\s*)?(can't|cannot)\s*(get\s*)?(enough\s*)?(air|breath)\b"
            ],
            escalation_type=EscalationType.EMERGENCY_911,
            description="Severe breathing difficulty may indicate respiratory failure or cardiac emergency",
            recommended_actions=[
                "Call 911 if breathing difficulty is severe or worsening",
                "Sit upright to ease breathing",
                "Use rescue inhaler if prescribed and available",
                "Stay calm and take slow, deep breaths if possible"
            ],
            emergency_instructions="Severe breathing difficulty is a medical emergency. Call 911 immediately."
        ),
        RedFlagSymptom(
            name="Stroke Signs",
            category=RedFlagCategory.NEUROLOGICAL,
            severity=RedFlagSeverity.CRITICAL,
            keywords=["face drooping", "arm weakness", "speech difficulty", "sudden numbness",
                     "sudden confusion", "sudden vision problems", "sudden severe headache",
                     "slurred speech", "can't speak", "one side weak", "facial droop"],
            patterns=[
                r"\b(face|facial)\s*(drooping|droop|numb|weak)\b",
                r"\b(arm|leg|one\s*side)\s*(weak|numb|can't\s*move|cannot\s*move)\b",
                r"\b(slurred|slurring|trouble|difficulty)\s*(speech|speaking|talking)\b",
                r"\b(sudden)\s*(numbness|confusion|headache|vision|dizziness)\b",
                r"\b(can't|cannot)\s*(speak|talk|move|see)\s*(clearly|properly|well)?\b"
            ],
            escalation_type=EscalationType.EMERGENCY_911,
            description="Stroke symptoms require immediate emergency care - time is critical",
            recommended_actions=[
                "Call 911 immediately - every minute counts",
                "Note the time when symptoms first appeared",
                "Do not give food or water",
                "Keep the person still and calm"
            ],
            emergency_instructions="STROKE IS A MEDICAL EMERGENCY. Call 911 immediately. Note: FAST - Face drooping, Arm weakness, Speech difficulty, Time to call 911."
        ),
        RedFlagSymptom(
            name="Severe Allergic Reaction",
            category=RedFlagCategory.ALLERGIC,
            severity=RedFlagSeverity.CRITICAL,
            keywords=["anaphylaxis", "throat swelling", "tongue swelling", "lips swelling",
                     "can't swallow", "hives spreading", "severe allergic", "allergic reaction",
                     "face swelling", "throat closing"],
            patterns=[
                r"\b(throat|tongue|lips|face)\s*(swelling|swollen|closing|tight)\b",
                r"\b(can't|cannot|hard\s*to|difficulty)\s*(swallow|breathe)\s*(after|following)?\s*(eating|food|medication|sting)?\b",
                r"\b(anaphylax|anaphylactic|severe\s*allergic|allergic\s*reaction)\b",
                r"\b(hives|rash)\s*(spreading|all\s*over|everywhere)\b"
            ],
            escalation_type=EscalationType.EMERGENCY_911,
            description="Anaphylaxis is a life-threatening allergic reaction requiring immediate treatment",
            recommended_actions=[
                "Call 911 immediately",
                "Use epinephrine auto-injector (EpiPen) if available and prescribed",
                "Lay person flat with legs elevated unless breathing is difficult",
                "Be prepared to give CPR if needed"
            ],
            emergency_instructions="Anaphylaxis can be fatal within minutes. Call 911 and use EpiPen immediately if available."
        ),
        RedFlagSymptom(
            name="Uncontrolled Bleeding",
            category=RedFlagCategory.HEMORRHAGIC,
            severity=RedFlagSeverity.CRITICAL,
            keywords=["won't stop bleeding", "heavy bleeding", "bleeding profusely", "blood everywhere",
                     "can't stop the blood", "spurting blood", "arterial bleeding", "hemorrhage",
                     "coughing blood", "vomiting blood", "blood in stool"],
            patterns=[
                r"\b(bleeding|blood)\s*(won't|will\s*not|can't|cannot)\s*stop\b",
                r"\b(heavy|profuse|severe|uncontrolled|spurting|arterial)\s*(bleeding|hemorrhage)\b",
                r"\b(coughing|vomiting|throwing)\s*(up\s*)?(blood)\b",
                r"\b(blood)\s*(in|on)\s*(stool|urine|vomit)\b",
                r"\b(bleeding)\s*(from)\s*(head|neck|chest|abdomen)\b"
            ],
            escalation_type=EscalationType.EMERGENCY_911,
            description="Severe or uncontrolled bleeding requires immediate emergency care",
            recommended_actions=[
                "Call 911 immediately for severe bleeding",
                "Apply firm, direct pressure to wound with clean cloth",
                "Keep the injured area elevated if possible",
                "Do not remove embedded objects"
            ],
            emergency_instructions="Apply pressure to the wound and call 911 immediately. Do not remove any embedded objects."
        ),
        RedFlagSymptom(
            name="High Fever with Confusion",
            category=RedFlagCategory.INFECTIOUS,
            severity=RedFlagSeverity.HIGH,
            keywords=["high fever", "confusion", "fever and confused", "disoriented", "altered mental status",
                     "fever with stiff neck", "fever with rash", "meningitis", "sepsis"],
            patterns=[
                r"\b(fever|temperature)\s*(with|and)\s*(confusion|disoriented|altered|stiff\s*neck|rash)\b",
                r"\b(very\s*high|extremely\s*high|dangerous)\s*(fever|temperature)\b",
                r"\b(fever)\s*(over|above)\s*(103|104|105|39|40)\s*(degrees|f|c)?\b",
                r"\b(confused|disoriented|not\s*making\s*sense)\s*(with|and)\s*(fever|temperature)\b"
            ],
            escalation_type=EscalationType.IMMEDIATE_DOCTOR,
            description="High fever with altered mental status may indicate serious infection like meningitis or sepsis",
            recommended_actions=[
                "Seek immediate medical attention",
                "Monitor temperature closely",
                "Keep hydrated",
                "Watch for worsening symptoms"
            ],
            emergency_instructions="High fever with confusion or stiff neck may indicate meningitis. Seek emergency care immediately."
        ),
        RedFlagSymptom(
            name="Severe Headache",
            category=RedFlagCategory.NEUROLOGICAL,
            severity=RedFlagSeverity.HIGH,
            keywords=["worst headache", "thunderclap headache", "sudden severe headache", "worst headache of my life",
                     "explosive headache", "headache with stiff neck", "headache with fever"],
            patterns=[
                r"\b(worst|severe|sudden|explosive|thunderclap)\s*(headache)\s*(of\s*my\s*life|ever|I've\s*ever\s*had)?\b",
                r"\b(headache)\s*(came\s*on|started)\s*(suddenly|out\s*of\s*nowhere|instantly)\b",
                r"\b(headache)\s*(with|and)\s*(stiff\s*neck|fever|vision\s*changes|confusion)\b"
            ],
            escalation_type=EscalationType.IMMEDIATE_DOCTOR,
            description="Sudden severe headache may indicate brain aneurysm or hemorrhage",
            recommended_actions=[
                "Seek immediate medical evaluation",
                "Note exact time of onset",
                "Do not take blood thinners",
                "Rest in a dark, quiet room"
            ],
            emergency_instructions="Sudden, severe 'worst headache of your life' may indicate a brain bleed. Seek emergency care immediately."
        ),
        RedFlagSymptom(
            name="Diabetic Emergency",
            category=RedFlagCategory.METABOLIC,
            severity=RedFlagSeverity.HIGH,
            keywords=["blood sugar very low", "blood sugar very high", "hypoglycemic", "hyperglycemic",
                     "diabetic ketoacidosis", "DKA", "fruity breath", "extreme thirst"],
            patterns=[
                r"\b(blood\s*sugar|glucose)\s*(very\s*)?(low|high|dropping|spiking)\b",
                r"\b(hypoglycemi|hyperglycemi|diabetic\s*ketoacidosis|DKA)\b",
                r"\b(shaking|trembling|sweating|confused)\s*(and\s*)?(diabetic|diabetes)\b",
                r"\b(fruity\s*breath|extreme\s*thirst|frequent\s*urination)\s*(diabetic|diabetes)?\b"
            ],
            escalation_type=EscalationType.IMMEDIATE_DOCTOR,
            description="Diabetic emergencies can be life-threatening if not treated promptly",
            recommended_actions=[
                "Check blood sugar immediately if possible",
                "For low blood sugar: consume fast-acting sugar",
                "For high blood sugar: seek medical care",
                "Call 911 if person is unconscious or unresponsive"
            ],
            emergency_instructions="Diabetic emergencies can cause unconsciousness. If blood sugar is dangerously low or high, seek immediate medical care."
        ),
        RedFlagSymptom(
            name="Suicidal Ideation",
            category=RedFlagCategory.MENTAL_HEALTH,
            severity=RedFlagSeverity.CRITICAL,
            keywords=["want to die", "kill myself", "end my life", "suicidal", "self-harm",
                     "hurt myself", "better off dead", "no reason to live", "suicide"],
            patterns=[
                r"\b(want|wanting|going)\s*(to)?\s*(die|kill\s*myself|end\s*(my\s*)?life|hurt\s*myself)\b",
                r"\b(suicid|self.?harm|self.?injury)\b",
                r"\b(better\s*off\s*dead|no\s*reason\s*to\s*live|wish\s*I\s*was\s*dead)\b",
                r"\b(plan|planning|thinking)\s*(to|about)\s*(kill|hurt|harm)\s*(myself|me)\b"
            ],
            escalation_type=EscalationType.EMERGENCY_911,
            description="Suicidal thoughts require immediate crisis intervention",
            recommended_actions=[
                "You are not alone - help is available",
                "Call 988 (Suicide & Crisis Lifeline) immediately",
                "Do not leave the person alone if at immediate risk",
                "Remove access to means of self-harm if possible"
            ],
            emergency_instructions="If you or someone is in immediate danger, call 911 or 988 (Suicide & Crisis Lifeline) now."
        ),
        RedFlagSymptom(
            name="Severe Abdominal Pain",
            category=RedFlagCategory.GASTROINTESTINAL,
            severity=RedFlagSeverity.HIGH,
            keywords=["severe stomach pain", "severe abdominal pain", "abdomen rigid", "belly hard",
                     "appendicitis", "can't stand up straight", "worst stomach pain"],
            patterns=[
                r"\b(severe|intense|unbearable|worst)\s*(stomach|abdominal|belly|abdomen)\s*(pain|ache)\b",
                r"\b(stomach|abdomen|belly)\s*(is\s*)?(rigid|hard|distended|bloated)\b",
                r"\b(can't|cannot)\s*(stand\s*up\s*straight|move|walk)\s*(stomach|abdominal|pain)?\b",
                r"\b(appendicitis|ruptured|perforate)\b"
            ],
            escalation_type=EscalationType.IMMEDIATE_DOCTOR,
            description="Severe abdominal pain may indicate appendicitis, bowel obstruction, or other surgical emergency",
            recommended_actions=[
                "Do not eat or drink anything",
                "Seek immediate medical evaluation",
                "Note when pain started and any associated symptoms",
                "Watch for fever, vomiting, or worsening pain"
            ],
            emergency_instructions="Severe abdominal pain, especially with fever or rigid abdomen, may require emergency surgery."
        ),
        RedFlagSymptom(
            name="Severe Dehydration",
            category=RedFlagCategory.METABOLIC,
            severity=RedFlagSeverity.MODERATE,
            keywords=["very dehydrated", "no urine", "dark urine", "dry mouth", "dizzy and thirsty",
                     "sunken eyes", "can't keep fluids down"],
            patterns=[
                r"\b(severe|extreme|very)\s*(dehydrat)\b",
                r"\b(haven't|have\s*not|no)\s*(urinated|peed)\s*(in|for)\s*(\d+)?\s*(hours|day)\b",
                r"\b(urine)\s*(is\s*)?(very\s*)?(dark|brown|orange)\b",
                r"\b(can't|cannot)\s*(keep)\s*(fluids|water|anything)\s*(down)\b"
            ],
            escalation_type=EscalationType.URGENT_APPOINTMENT,
            description="Severe dehydration can lead to organ failure and requires medical treatment",
            recommended_actions=[
                "Seek medical attention",
                "Try small sips of water or electrolyte solution",
                "Avoid caffeinated and alcoholic beverages",
                "Rest in a cool environment"
            ],
            emergency_instructions="Severe dehydration, especially with confusion or inability to keep fluids down, requires medical evaluation."
        ),
        RedFlagSymptom(
            name="Loss of Consciousness",
            category=RedFlagCategory.NEUROLOGICAL,
            severity=RedFlagSeverity.CRITICAL,
            keywords=["passed out", "fainted", "unconscious", "blacked out", "lost consciousness",
                     "collapsed", "unresponsive"],
            patterns=[
                r"\b(passed\s*out|fainted|unconscious|blacked\s*out|collapsed|unresponsive)\b",
                r"\b(lost)\s*(consciousness|my\s*vision)\b",
                r"\b(woke\s*up)\s*(on\s*the\s*floor|didn't\s*know|confused)\b"
            ],
            escalation_type=EscalationType.IMMEDIATE_DOCTOR,
            description="Loss of consciousness may indicate cardiac, neurological, or metabolic emergency",
            recommended_actions=[
                "If currently unconscious, call 911",
                "Check for breathing and pulse",
                "Place in recovery position if breathing",
                "Do not give food or water until fully alert"
            ],
            emergency_instructions="If someone is unconscious and you don't know why, call 911 immediately."
        ),
        RedFlagSymptom(
            name="Seizure",
            category=RedFlagCategory.NEUROLOGICAL,
            severity=RedFlagSeverity.HIGH,
            keywords=["seizure", "convulsion", "fit", "shaking uncontrollably", "epileptic"],
            patterns=[
                r"\b(seizure|convulsion|epileptic\s*fit)\b",
                r"\b(shaking|jerking|trembling)\s*(uncontrollably|whole\s*body)\b",
                r"\b(first\s*time)\s*(seizure|convulsion)\b"
            ],
            escalation_type=EscalationType.IMMEDIATE_DOCTOR,
            description="New onset seizures or prolonged seizures require immediate evaluation",
            recommended_actions=[
                "Call 911 if seizure lasts more than 5 minutes",
                "Clear area of dangerous objects",
                "Do not restrain the person or put anything in mouth",
                "Turn on side after shaking stops"
            ],
            emergency_instructions="Call 911 if this is a first seizure, seizure lasts over 5 minutes, or person doesn't regain consciousness."
        ),
    ]
    
    SEVERITY_PRIORITY = {
        RedFlagSeverity.CRITICAL: 4,
        RedFlagSeverity.HIGH: 3,
        RedFlagSeverity.MODERATE: 2,
        RedFlagSeverity.LOW: 1
    }
    
    def __init__(self, db: Optional[Session] = None):
        self.db = db
        self.logger = logging.getLogger(__name__)
        self._compiled_patterns = self._compile_patterns()
    
    def _compile_patterns(self) -> Dict[str, List[re.Pattern]]:
        """Pre-compile regex patterns for efficient matching"""
        compiled = {}
        for symptom in self.RED_FLAG_TAXONOMY:
            compiled[symptom.name] = [
                re.compile(pattern, re.IGNORECASE) 
                for pattern in symptom.patterns
            ]
        return compiled
    
    def detect_red_flags(
        self,
        text: str,
        patient_id: Optional[str] = None,
        use_ai_analysis: bool = True,
        context: Optional[Dict[str, Any]] = None
    ) -> RedFlagDetection:
        """
        Detect red flags in patient text.
        
        Args:
            text: Patient message or symptom description
            patient_id: Optional patient ID for audit logging
            use_ai_analysis: Whether to use AI for enhanced detection
            context: Optional context (conversation history, patient conditions)
            
        Returns:
            RedFlagDetection with detected symptoms and escalation guidance
        """
        if not text or not text.strip():
            return RedFlagDetection(detected=False)
        
        detected_symptoms = []
        categories_found = set()
        max_severity = None
        max_severity_priority = 0
        
        for symptom in self.RED_FLAG_TAXONOMY:
            patterns = self._compiled_patterns.get(symptom.name, [])
            
            matched = False
            for pattern in patterns:
                if pattern.search(text):
                    matched = True
                    break
            
            if not matched:
                text_lower = text.lower()
                for keyword in symptom.keywords:
                    if keyword.lower() in text_lower:
                        matched = True
                        break
            
            if matched:
                detected_symptoms.append({
                    "name": symptom.name,
                    "category": symptom.category.value,
                    "severity": symptom.severity.value,
                    "escalation_type": symptom.escalation_type.value,
                    "description": symptom.description,
                    "recommended_actions": symptom.recommended_actions,
                    "emergency_instructions": symptom.emergency_instructions
                })
                categories_found.add(symptom.category)
                
                priority = self.SEVERITY_PRIORITY.get(symptom.severity, 0)
                if priority > max_severity_priority:
                    max_severity_priority = priority
                    max_severity = symptom.severity
        
        if not detected_symptoms:
            return RedFlagDetection(detected=False)
        
        escalation_type = self._determine_escalation_type(detected_symptoms)
        
        all_actions = []
        emergency_instruction = None
        for symptom in detected_symptoms:
            all_actions.extend(symptom["recommended_actions"])
            if symptom["emergency_instructions"]:
                emergency_instruction = symptom["emergency_instructions"]
        
        unique_actions = list(dict.fromkeys(all_actions))
        
        ai_analysis = None
        confidence_score = 0.85
        
        if use_ai_analysis and len(detected_symptoms) >= 1:
            try:
                ai_result = self._perform_ai_analysis(text, detected_symptoms, context)
                ai_analysis = ai_result.get("analysis")
                confidence_score = ai_result.get("confidence", 0.85)
            except Exception as e:
                self.logger.warning(f"AI analysis failed (continuing without): {e}")
        
        detection = RedFlagDetection(
            detected=True,
            symptoms=detected_symptoms,
            highest_severity=max_severity,
            escalation_type=escalation_type,
            categories=list(categories_found),
            confidence_score=confidence_score,
            recommended_actions=unique_actions[:10],
            emergency_instructions=emergency_instruction,
            ai_analysis=ai_analysis
        )
        
        if patient_id and self.db:
            self._audit_detection(patient_id, detection)
        
        return detection
    
    def _determine_escalation_type(
        self, 
        symptoms: List[Dict[str, Any]]
    ) -> EscalationType:
        """Determine highest priority escalation type from detected symptoms"""
        escalation_priority = {
            EscalationType.EMERGENCY_911.value: 5,
            EscalationType.IMMEDIATE_DOCTOR.value: 4,
            EscalationType.URGENT_APPOINTMENT.value: 3,
            EscalationType.ROUTINE_FOLLOWUP.value: 2,
            EscalationType.MONITOR_AND_LOG.value: 1
        }
        
        max_priority = 0
        max_escalation = EscalationType.MONITOR_AND_LOG
        
        for symptom in symptoms:
            escalation = symptom.get("escalation_type", EscalationType.MONITOR_AND_LOG.value)
            priority = escalation_priority.get(escalation, 0)
            if priority > max_priority:
                max_priority = priority
                max_escalation = EscalationType(escalation)
        
        return max_escalation
    
    def _perform_ai_analysis(
        self,
        text: str,
        detected_symptoms: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Use AI for enhanced symptom analysis and confirmation"""
        try:
            check_openai_baa_compliance()
            client = get_openai_client()
            
            symptom_names = [s["name"] for s in detected_symptoms]
            
            system_prompt = """You are a clinical decision support system analyzing patient symptoms for potential medical emergencies.

Your role is to:
1. Confirm or refine the detected symptoms
2. Identify any additional concerning patterns
3. Assess overall risk level
4. Provide brief clinical reasoning

IMPORTANT: This is for wellness monitoring and decision support only. You do not diagnose or treat - you identify patterns that warrant medical attention.

Respond in JSON format:
{
    "confirmed_symptoms": ["list of confirmed concerning symptoms"],
    "additional_concerns": ["any other patterns noted"],
    "risk_assessment": "critical|high|moderate|low",
    "confidence": 0.0 to 1.0,
    "clinical_reasoning": "brief explanation",
    "recommendation": "seek_emergency_care|contact_doctor_now|schedule_urgent_appointment|monitor_and_followup"
}"""
            
            user_prompt = f"""Patient statement: {text}

Pattern-detected symptoms: {', '.join(symptom_names)}

Additional context: {context if context else 'None provided'}

Analyze the clinical significance and confirm severity assessment."""
            
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=500,
                response_format={"type": "json_object"}
            )
            
            import json
            result = json.loads(response.choices[0].message.content)
            
            return {
                "analysis": result.get("clinical_reasoning", ""),
                "confidence": result.get("confidence", 0.85),
                "ai_recommendation": result.get("recommendation", ""),
                "confirmed_symptoms": result.get("confirmed_symptoms", []),
                "additional_concerns": result.get("additional_concerns", [])
            }
            
        except Exception as e:
            self.logger.error(f"AI analysis error: {e}")
            return {"analysis": None, "confidence": 0.85}
    
    def _audit_detection(
        self,
        patient_id: str,
        detection: RedFlagDetection
    ) -> None:
        """HIPAA-compliant audit logging for red flag detection"""
        if not self.db:
            return
            
        try:
            symptom_names = [s["name"] for s in detection.symptoms]
            
            audit_entry = AuditLog(
                user_id="red_flag_detection_service",
                user_type="system",
                action_type="analyze",
                action_category="clinical_safety",
                resource_type="red_flag_detection",
                resource_id="",
                phi_accessed=True,
                patient_id_accessed=patient_id,
                action_description=f"Red flag detection: {', '.join(symptom_names)}",
                action_result="detected" if detection.detected else "clear",
                data_fields_accessed={
                    "patient_id": patient_id,
                    "symptoms_detected": symptom_names,
                    "highest_severity": detection.highest_severity.value if detection.highest_severity else None,
                    "escalation_type": detection.escalation_type.value if detection.escalation_type else None,
                    "categories": [c.value for c in detection.categories],
                    "confidence_score": detection.confidence_score,
                    "detected_at": detection.detected_at.isoformat()
                },
                ip_address="127.0.0.1",
                user_agent="RedFlagDetectionService/1.0"
            )
            self.db.add(audit_entry)
            self.db.commit()
        except Exception as e:
            self.logger.warning(f"Audit logging failed (non-blocking): {e}")
    
    def get_severity_guidance(self, severity: RedFlagSeverity) -> Dict[str, Any]:
        """Get guidance for a specific severity level"""
        guidance = {
            RedFlagSeverity.CRITICAL: {
                "action_required": "IMMEDIATE EMERGENCY ACTION",
                "timeframe": "Within minutes",
                "instructions": [
                    "Call 911 immediately if not already done",
                    "Do not wait - this is a medical emergency",
                    "Stay on the line with emergency services"
                ],
                "escalation": EscalationType.EMERGENCY_911
            },
            RedFlagSeverity.HIGH: {
                "action_required": "URGENT MEDICAL ATTENTION",
                "timeframe": "Within 1-2 hours",
                "instructions": [
                    "Contact your doctor immediately",
                    "Go to urgent care or emergency room if doctor unavailable",
                    "Do not drive yourself if symptoms affect consciousness or vision"
                ],
                "escalation": EscalationType.IMMEDIATE_DOCTOR
            },
            RedFlagSeverity.MODERATE: {
                "action_required": "PROMPT MEDICAL EVALUATION",
                "timeframe": "Within 24 hours",
                "instructions": [
                    "Schedule an urgent appointment with your doctor",
                    "Monitor symptoms closely for worsening",
                    "Seek emergency care if symptoms suddenly worsen"
                ],
                "escalation": EscalationType.URGENT_APPOINTMENT
            },
            RedFlagSeverity.LOW: {
                "action_required": "MONITOR AND FOLLOW UP",
                "timeframe": "At next scheduled visit",
                "instructions": [
                    "Discuss these symptoms at your next doctor visit",
                    "Keep a symptom diary",
                    "Seek earlier care if symptoms worsen or persist"
                ],
                "escalation": EscalationType.ROUTINE_FOLLOWUP
            }
        }
        return guidance.get(severity, guidance[RedFlagSeverity.LOW])
    
    def analyze_conversation_history(
        self,
        messages: List[Dict[str, str]],
        patient_id: Optional[str] = None
    ) -> RedFlagDetection:
        """Analyze a full conversation history for red flags"""
        patient_messages = [
            msg.get("content", "") 
            for msg in messages 
            if msg.get("role") == "user"
        ]
        
        combined_text = " ".join(patient_messages)
        
        return self.detect_red_flags(
            text=combined_text,
            patient_id=patient_id,
            use_ai_analysis=True,
            context={"message_count": len(messages)}
        )


def get_red_flag_service(db: Optional[Session] = None) -> RedFlagDetectionService:
    """Factory function to get red flag detection service instance"""
    return RedFlagDetectionService(db=db)
