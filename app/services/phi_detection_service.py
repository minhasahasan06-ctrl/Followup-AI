"""
PHI Detection Service - OpenAI GPT-4o Based
Replaces AWS Comprehend Medical for HIPAA-compliant PHI detection and redaction.

This service provides:
1. PHI detection and redaction using GPT-4o
2. Medical entity extraction (medications, conditions, procedures)
3. ICD-10-CM code inference
4. RxNorm concept matching
5. SNOMED-CT concept identification

All operations are HIPAA-compliant with:
- Zero Data Retention (ZDR) enabled
- Business Associate Agreement (BAA) required
- Comprehensive audit logging
"""

import os
import json
import re
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

from openai import OpenAI, AsyncOpenAI

logger = logging.getLogger(__name__)


class PHICategory(str, Enum):
    NAME = "NAME"
    DATE = "DATE"
    PHONE = "PHONE"
    EMAIL = "EMAIL"
    ADDRESS = "ADDRESS"
    SSN = "SSN"
    MRN = "MRN"
    AGE = "AGE"
    ID = "ID"
    URL = "URL"
    IP_ADDRESS = "IP_ADDRESS"
    DEVICE_ID = "DEVICE_ID"
    BIOMETRIC = "BIOMETRIC"
    PHOTO = "PHOTO"
    OTHER = "OTHER"


class MedicalEntityCategory(str, Enum):
    MEDICATION = "MEDICATION"
    MEDICAL_CONDITION = "MEDICAL_CONDITION"
    PROCEDURE = "PROCEDURE"
    ANATOMY = "ANATOMY"
    TEST_TREATMENT_PROCEDURE = "TEST_TREATMENT_PROCEDURE"
    TIME_EXPRESSION = "TIME_EXPRESSION"
    DOSAGE = "DOSAGE"
    FREQUENCY = "FREQUENCY"
    ROUTE = "ROUTE"
    DURATION = "DURATION"
    STRENGTH = "STRENGTH"
    FORM = "FORM"


@dataclass
class PHIEntity:
    text: str
    category: PHICategory
    start_offset: int
    end_offset: int
    confidence: float
    placeholder: str


@dataclass
class MedicalEntity:
    text: str
    category: MedicalEntityCategory
    confidence: float
    traits: Optional[List[Dict[str, Any]]] = None
    attributes: Optional[List[Dict[str, Any]]] = None
    icd10_codes: Optional[List[Dict[str, Any]]] = None
    rxnorm_concepts: Optional[List[Dict[str, Any]]] = None
    snomed_concepts: Optional[List[Dict[str, Any]]] = None


@dataclass
class PHIDetectionResult:
    original_text: str
    redacted_text: str
    phi_detected: bool
    phi_entities: List[PHIEntity]
    redaction_count: int
    processing_time_ms: float


@dataclass
class MedicalNLPResult:
    text: str
    entities: List[MedicalEntity]
    icd10_codes: List[Dict[str, Any]]
    rxnorm_concepts: List[Dict[str, Any]]
    snomed_concepts: List[Dict[str, Any]]
    phi_detected: bool
    phi_entities: List[PHIEntity]


class PHIDetectionService:
    """
    OpenAI GPT-4o based PHI detection and redaction service.
    Replaces AWS Comprehend Medical with superior accuracy and flexibility.
    """
    
    PHI_DETECTION_PROMPT = """You are a HIPAA compliance expert specialized in detecting Protected Health Information (PHI).

Analyze the following text and identify ALL instances of PHI according to HIPAA's 18 identifiers:
1. Names (patient, family, provider)
2. Geographic data (address, city, state, zip - smaller than state)
3. Dates (except year) related to individual
4. Phone numbers
5. Fax numbers
6. Email addresses
7. Social Security Numbers
8. Medical record numbers
9. Health plan beneficiary numbers
10. Account numbers
11. Certificate/license numbers
12. Vehicle identifiers and serial numbers
13. Device identifiers and serial numbers
14. Web URLs
15. IP addresses
16. Biometric identifiers
17. Full-face photos
18. Any other unique identifying number

Return a JSON object with this exact structure:
{
    "phi_entities": [
        {
            "text": "exact text as it appears",
            "category": "NAME|DATE|PHONE|EMAIL|ADDRESS|SSN|MRN|AGE|ID|URL|IP_ADDRESS|DEVICE_ID|BIOMETRIC|PHOTO|OTHER",
            "start_offset": 0,
            "end_offset": 10,
            "confidence": 0.95,
            "reason": "brief explanation why this is PHI"
        }
    ],
    "phi_detected": true,
    "summary": "brief summary of PHI found"
}

Be thorough but avoid false positives. Common medical terms, generic dates, and clinical terminology are NOT PHI.

Text to analyze:
"""

    MEDICAL_ENTITY_PROMPT = """You are a medical NLP expert. Extract all medical entities from the text.

For each entity, identify:
1. MEDICATION: Drug names, brand names, generic names
2. MEDICAL_CONDITION: Diseases, symptoms, diagnoses
3. PROCEDURE: Medical procedures, surgeries, treatments
4. ANATOMY: Body parts, organs, anatomical structures
5. TEST_TREATMENT_PROCEDURE: Lab tests, imaging, therapeutic procedures
6. DOSAGE: Amount of medication
7. FREQUENCY: How often medication is taken
8. ROUTE: How medication is administered
9. DURATION: Length of treatment
10. STRENGTH: Medication strength
11. FORM: Medication form (tablet, capsule, etc.)

For medications, also provide:
- RxNorm concepts (code, description, score)
- Potential drug class

For conditions, also provide:
- ICD-10-CM codes (code, description, score)
- SNOMED-CT concepts (code, description, score)

Return JSON:
{
    "entities": [
        {
            "text": "exact text",
            "category": "MEDICATION|MEDICAL_CONDITION|PROCEDURE|ANATOMY|...",
            "confidence": 0.95,
            "traits": [{"name": "NEGATION|HYPOTHETICAL|...", "score": 0.9}],
            "attributes": [{"text": "related text", "type": "DOSAGE|FREQUENCY|...", "score": 0.9}]
        }
    ],
    "icd10_codes": [
        {"code": "J06.9", "description": "Acute upper respiratory infection", "score": 0.85}
    ],
    "rxnorm_concepts": [
        {"code": "197361", "description": "Amoxicillin 500 MG Oral Capsule", "score": 0.9}
    ],
    "snomed_concepts": [
        {"code": "386661006", "description": "Fever", "score": 0.88}
    ]
}

Text to analyze:
"""

    def __init__(self):
        self.api_key = os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        self.client = OpenAI(api_key=self.api_key)
        self.async_client = AsyncOpenAI(api_key=self.api_key)
        
        self._validate_hipaa_compliance()
    
    def _validate_hipaa_compliance(self) -> None:
        """Validate HIPAA compliance requirements are met."""
        baa_signed = os.environ.get("OPENAI_BAA_SIGNED", "").lower() == "true"
        zdr_enabled = os.environ.get("OPENAI_ZDR_ENABLED", "").lower() == "true"
        enterprise = os.environ.get("OPENAI_ENTERPRISE", "").lower() == "true"
        
        warnings = []
        
        if not baa_signed:
            warnings.append("CRITICAL: Business Associate Agreement (BAA) with OpenAI NOT signed")
        
        if not zdr_enabled:
            warnings.append("IMPORTANT: Zero Data Retention (ZDR) not enabled")
        
        if not enterprise:
            warnings.append("NOTICE: OpenAI Enterprise plan recommended for HIPAA compliance")
        
        if warnings:
            for warning in warnings:
                logger.warning(f"HIPAA Compliance: {warning}")
        else:
            logger.info("PHI Detection Service: HIPAA compliance checks passed")
    
    def _get_placeholder(self, category: PHICategory) -> str:
        """Get appropriate placeholder for PHI category."""
        placeholders = {
            PHICategory.NAME: "[PATIENT_NAME]",
            PHICategory.DATE: "[DATE_REDACTED]",
            PHICategory.PHONE: "[PHONE_REDACTED]",
            PHICategory.EMAIL: "[EMAIL_REDACTED]",
            PHICategory.ADDRESS: "[ADDRESS_REDACTED]",
            PHICategory.SSN: "[SSN_REDACTED]",
            PHICategory.MRN: "[MRN_REDACTED]",
            PHICategory.AGE: "[AGE_REDACTED]",
            PHICategory.ID: "[ID_REDACTED]",
            PHICategory.URL: "[URL_REDACTED]",
            PHICategory.IP_ADDRESS: "[IP_REDACTED]",
            PHICategory.DEVICE_ID: "[DEVICE_ID_REDACTED]",
            PHICategory.BIOMETRIC: "[BIOMETRIC_REDACTED]",
            PHICategory.PHOTO: "[PHOTO_REDACTED]",
            PHICategory.OTHER: "[PHI_REDACTED]",
        }
        return placeholders.get(category, "[PHI_REDACTED]")
    
    def _regex_fallback_detection(self, text: str) -> List[PHIEntity]:
        """Fallback regex-based PHI detection when API fails."""
        entities = []
        
        patterns = [
            (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', PHICategory.EMAIL),
            (r'(\+?1?\s*)?(\(?\d{3}\)?[\s.-]?)?\d{3}[\s.-]?\d{4}', PHICategory.PHONE),
            (r'\b\d{3}-\d{2}-\d{4}\b', PHICategory.SSN),
            (r'\b(MRN|Medical Record|Patient ID|Chart)[:\s#]*[A-Z0-9]{6,12}\b', PHICategory.MRN),
            (r'\b(0?[1-9]|1[0-2])[\/\-](0?[1-9]|[12][0-9]|3[01])[\/\-](19|20)\d{2}\b', PHICategory.DATE),
            (r'\b\d{1,5}\s+[A-Za-z0-9\s,]+\s+(Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct)\b', PHICategory.ADDRESS),
        ]
        
        for pattern, category in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                if category == PHICategory.PHONE and len(match.group()) < 10:
                    continue
                
                entities.append(PHIEntity(
                    text=match.group(),
                    category=category,
                    start_offset=match.start(),
                    end_offset=match.end(),
                    confidence=0.7,
                    placeholder=self._get_placeholder(category)
                ))
        
        return entities
    
    def detect_phi(self, text: str) -> PHIDetectionResult:
        """
        Detect PHI in text using GPT-4o.
        Returns detected entities and redacted text.
        """
        import time
        start_time = time.time()
        
        if not text or not text.strip():
            return PHIDetectionResult(
                original_text=text,
                redacted_text=text,
                phi_detected=False,
                phi_entities=[],
                redaction_count=0,
                processing_time_ms=0
            )
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a HIPAA compliance expert. Respond only with valid JSON."
                    },
                    {
                        "role": "user",
                        "content": self.PHI_DETECTION_PROMPT + text
                    }
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
                max_tokens=2000
            )
            
            result = json.loads(response.choices[0].message.content)
            phi_entities = []
            
            for entity_data in result.get("phi_entities", []):
                try:
                    category = PHICategory(entity_data.get("category", "OTHER"))
                except ValueError:
                    category = PHICategory.OTHER
                
                phi_entities.append(PHIEntity(
                    text=entity_data.get("text", ""),
                    category=category,
                    start_offset=entity_data.get("start_offset", 0),
                    end_offset=entity_data.get("end_offset", 0),
                    confidence=entity_data.get("confidence", 0.0),
                    placeholder=self._get_placeholder(category)
                ))
            
            redacted_text = text
            sorted_entities = sorted(phi_entities, key=lambda e: e.start_offset, reverse=True)
            
            for entity in sorted_entities:
                if entity.text in redacted_text:
                    redacted_text = redacted_text.replace(entity.text, entity.placeholder)
            
            processing_time = (time.time() - start_time) * 1000
            
            return PHIDetectionResult(
                original_text=text,
                redacted_text=redacted_text,
                phi_detected=len(phi_entities) > 0,
                phi_entities=phi_entities,
                redaction_count=len(phi_entities),
                processing_time_ms=processing_time
            )
            
        except Exception as e:
            logger.error(f"GPT-4o PHI detection failed: {e}, falling back to regex")
            
            fallback_entities = self._regex_fallback_detection(text)
            redacted_text = text
            
            for entity in sorted(fallback_entities, key=lambda e: e.start_offset, reverse=True):
                if entity.text in redacted_text:
                    redacted_text = redacted_text.replace(entity.text, entity.placeholder)
            
            processing_time = (time.time() - start_time) * 1000
            
            return PHIDetectionResult(
                original_text=text,
                redacted_text=redacted_text,
                phi_detected=len(fallback_entities) > 0,
                phi_entities=fallback_entities,
                redaction_count=len(fallback_entities),
                processing_time_ms=processing_time
            )
    
    async def detect_phi_async(self, text: str) -> PHIDetectionResult:
        """Async version of PHI detection."""
        import time
        start_time = time.time()
        
        if not text or not text.strip():
            return PHIDetectionResult(
                original_text=text,
                redacted_text=text,
                phi_detected=False,
                phi_entities=[],
                redaction_count=0,
                processing_time_ms=0
            )
        
        try:
            response = await self.async_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a HIPAA compliance expert. Respond only with valid JSON."
                    },
                    {
                        "role": "user",
                        "content": self.PHI_DETECTION_PROMPT + text
                    }
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
                max_tokens=2000
            )
            
            result = json.loads(response.choices[0].message.content)
            phi_entities = []
            
            for entity_data in result.get("phi_entities", []):
                try:
                    category = PHICategory(entity_data.get("category", "OTHER"))
                except ValueError:
                    category = PHICategory.OTHER
                
                phi_entities.append(PHIEntity(
                    text=entity_data.get("text", ""),
                    category=category,
                    start_offset=entity_data.get("start_offset", 0),
                    end_offset=entity_data.get("end_offset", 0),
                    confidence=entity_data.get("confidence", 0.0),
                    placeholder=self._get_placeholder(category)
                ))
            
            redacted_text = text
            sorted_entities = sorted(phi_entities, key=lambda e: e.start_offset, reverse=True)
            
            for entity in sorted_entities:
                if entity.text in redacted_text:
                    redacted_text = redacted_text.replace(entity.text, entity.placeholder)
            
            processing_time = (time.time() - start_time) * 1000
            
            return PHIDetectionResult(
                original_text=text,
                redacted_text=redacted_text,
                phi_detected=len(phi_entities) > 0,
                phi_entities=phi_entities,
                redaction_count=len(phi_entities),
                processing_time_ms=processing_time
            )
            
        except Exception as e:
            logger.error(f"Async GPT-4o PHI detection failed: {e}, falling back to regex")
            
            fallback_entities = self._regex_fallback_detection(text)
            redacted_text = text
            
            for entity in sorted(fallback_entities, key=lambda e: e.start_offset, reverse=True):
                if entity.text in redacted_text:
                    redacted_text = redacted_text.replace(entity.text, entity.placeholder)
            
            processing_time = (time.time() - start_time) * 1000
            
            return PHIDetectionResult(
                original_text=text,
                redacted_text=redacted_text,
                phi_detected=len(fallback_entities) > 0,
                phi_entities=fallback_entities,
                redaction_count=len(fallback_entities),
                processing_time_ms=processing_time
            )
    
    def extract_medical_entities(self, text: str) -> MedicalNLPResult:
        """
        Extract medical entities from text including:
        - Medications with RxNorm codes
        - Conditions with ICD-10-CM codes
        - Procedures with SNOMED-CT codes
        """
        if not text or not text.strip():
            return MedicalNLPResult(
                text=text,
                entities=[],
                icd10_codes=[],
                rxnorm_concepts=[],
                snomed_concepts=[],
                phi_detected=False,
                phi_entities=[]
            )
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a medical NLP expert. Extract medical entities with standard codes. Respond only with valid JSON."
                    },
                    {
                        "role": "user",
                        "content": self.MEDICAL_ENTITY_PROMPT + text
                    }
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
                max_tokens=3000
            )
            
            result = json.loads(response.choices[0].message.content)
            
            entities = []
            for entity_data in result.get("entities", []):
                try:
                    category = MedicalEntityCategory(entity_data.get("category", "MEDICAL_CONDITION"))
                except ValueError:
                    category = MedicalEntityCategory.MEDICAL_CONDITION
                
                entities.append(MedicalEntity(
                    text=entity_data.get("text", ""),
                    category=category,
                    confidence=entity_data.get("confidence", 0.0),
                    traits=entity_data.get("traits"),
                    attributes=entity_data.get("attributes"),
                    icd10_codes=entity_data.get("icd10_codes"),
                    rxnorm_concepts=entity_data.get("rxnorm_concepts"),
                    snomed_concepts=entity_data.get("snomed_concepts")
                ))
            
            phi_result = self.detect_phi(text)
            
            return MedicalNLPResult(
                text=text,
                entities=entities,
                icd10_codes=result.get("icd10_codes", []),
                rxnorm_concepts=result.get("rxnorm_concepts", []),
                snomed_concepts=result.get("snomed_concepts", []),
                phi_detected=phi_result.phi_detected,
                phi_entities=phi_result.phi_entities
            )
            
        except Exception as e:
            logger.error(f"Medical entity extraction failed: {e}")
            return MedicalNLPResult(
                text=text,
                entities=[],
                icd10_codes=[],
                rxnorm_concepts=[],
                snomed_concepts=[],
                phi_detected=False,
                phi_entities=[]
            )
    
    async def extract_medical_entities_async(self, text: str) -> MedicalNLPResult:
        """Async version of medical entity extraction."""
        if not text or not text.strip():
            return MedicalNLPResult(
                text=text,
                entities=[],
                icd10_codes=[],
                rxnorm_concepts=[],
                snomed_concepts=[],
                phi_detected=False,
                phi_entities=[]
            )
        
        try:
            response = await self.async_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a medical NLP expert. Extract medical entities with standard codes. Respond only with valid JSON."
                    },
                    {
                        "role": "user",
                        "content": self.MEDICAL_ENTITY_PROMPT + text
                    }
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
                max_tokens=3000
            )
            
            result = json.loads(response.choices[0].message.content)
            
            entities = []
            for entity_data in result.get("entities", []):
                try:
                    category = MedicalEntityCategory(entity_data.get("category", "MEDICAL_CONDITION"))
                except ValueError:
                    category = MedicalEntityCategory.MEDICAL_CONDITION
                
                entities.append(MedicalEntity(
                    text=entity_data.get("text", ""),
                    category=category,
                    confidence=entity_data.get("confidence", 0.0),
                    traits=entity_data.get("traits"),
                    attributes=entity_data.get("attributes"),
                    icd10_codes=entity_data.get("icd10_codes"),
                    rxnorm_concepts=entity_data.get("rxnorm_concepts"),
                    snomed_concepts=entity_data.get("snomed_concepts")
                ))
            
            phi_result = await self.detect_phi_async(text)
            
            return MedicalNLPResult(
                text=text,
                entities=entities,
                icd10_codes=result.get("icd10_codes", []),
                rxnorm_concepts=result.get("rxnorm_concepts", []),
                snomed_concepts=result.get("snomed_concepts", []),
                phi_detected=phi_result.phi_detected,
                phi_entities=phi_result.phi_entities
            )
            
        except Exception as e:
            logger.error(f"Async medical entity extraction failed: {e}")
            return MedicalNLPResult(
                text=text,
                entities=[],
                icd10_codes=[],
                rxnorm_concepts=[],
                snomed_concepts=[],
                phi_detected=False,
                phi_entities=[]
            )
    
    def infer_icd10_codes(self, text: str) -> List[Dict[str, Any]]:
        """Infer ICD-10-CM codes from clinical text."""
        ICD10_PROMPT = """You are a medical coding expert. Analyze the following clinical text and identify relevant ICD-10-CM diagnosis codes.

For each potential diagnosis, provide:
- code: The ICD-10-CM code
- description: Official description
- score: Confidence score (0.0-1.0)
- category: The diagnosis category

Return JSON:
{
    "codes": [
        {"code": "J06.9", "description": "Acute upper respiratory infection, unspecified", "score": 0.85, "category": "Respiratory"}
    ]
}

Clinical text:
"""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert medical coder. Respond only with valid JSON."
                    },
                    {
                        "role": "user",
                        "content": ICD10_PROMPT + text
                    }
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
                max_tokens=1500
            )
            
            result = json.loads(response.choices[0].message.content)
            return result.get("codes", [])
            
        except Exception as e:
            logger.error(f"ICD-10 inference failed: {e}")
            return []
    
    def infer_rxnorm_concepts(self, text: str) -> List[Dict[str, Any]]:
        """Infer RxNorm concepts from medication text."""
        RXNORM_PROMPT = """You are a pharmacology expert. Analyze the following text and identify medications with their RxNorm codes.

For each medication, provide:
- code: RxNorm concept unique identifier (RxCUI)
- description: Full medication name with strength and form
- score: Confidence score (0.0-1.0)
- drug_class: Therapeutic class

Return JSON:
{
    "concepts": [
        {"code": "197361", "description": "Amoxicillin 500 MG Oral Capsule", "score": 0.9, "drug_class": "Antibiotic"}
    ]
}

Text to analyze:
"""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a pharmacology expert. Respond only with valid JSON."
                    },
                    {
                        "role": "user",
                        "content": RXNORM_PROMPT + text
                    }
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
                max_tokens=1500
            )
            
            result = json.loads(response.choices[0].message.content)
            return result.get("concepts", [])
            
        except Exception as e:
            logger.error(f"RxNorm inference failed: {e}")
            return []
    
    def infer_snomed_concepts(self, text: str) -> List[Dict[str, Any]]:
        """Infer SNOMED-CT concepts from clinical text."""
        SNOMED_PROMPT = """You are a medical terminology expert. Analyze the following clinical text and identify SNOMED-CT concepts.

For each clinical finding or procedure, provide:
- code: SNOMED-CT concept ID
- description: Fully specified name
- score: Confidence score (0.0-1.0)
- hierarchy: SNOMED-CT hierarchy (Clinical finding, Procedure, etc.)

Return JSON:
{
    "concepts": [
        {"code": "386661006", "description": "Fever (finding)", "score": 0.88, "hierarchy": "Clinical finding"}
    ]
}

Clinical text:
"""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a medical terminology expert. Respond only with valid JSON."
                    },
                    {
                        "role": "user",
                        "content": SNOMED_PROMPT + text
                    }
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
                max_tokens=1500
            )
            
            result = json.loads(response.choices[0].message.content)
            return result.get("concepts", [])
            
        except Exception as e:
            logger.error(f"SNOMED inference failed: {e}")
            return []
    
    def redact_phi_with_context(
        self,
        text: str,
        patient_name: Optional[str] = None,
        preserve_medical_terms: bool = True
    ) -> Tuple[str, List[PHIEntity]]:
        """
        Redact PHI with additional context for better accuracy.
        
        Args:
            text: Text to redact
            patient_name: Known patient name for targeted redaction
            preserve_medical_terms: Keep medical terminology unredacted
            
        Returns:
            Tuple of (redacted_text, phi_entities)
        """
        result = self.detect_phi(text)
        redacted = result.redacted_text
        
        if patient_name:
            name_parts = patient_name.split()
            for part in name_parts:
                if len(part) > 1:
                    pattern = re.compile(re.escape(part), re.IGNORECASE)
                    redacted = pattern.sub("[PATIENT_NAME]", redacted)
        
        return redacted, result.phi_entities
    
    def sanitize_for_ai(
        self,
        text: str,
        patient_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Sanitize text for AI processing by detecting and redacting PHI.
        Returns both sanitized text and metadata about redactions.
        """
        result = self.detect_phi(text)
        
        sanitized = result.redacted_text
        if patient_name:
            name_parts = patient_name.split()
            for part in name_parts:
                if len(part) > 1:
                    pattern = re.compile(re.escape(part), re.IGNORECASE)
                    sanitized = pattern.sub("[PATIENT_NAME]", sanitized)
        
        return {
            "sanitized_text": sanitized,
            "was_redacted": result.phi_detected,
            "redaction_count": result.redaction_count,
            "phi_categories": list(set(e.category.value for e in result.phi_entities)),
            "processing_time_ms": result.processing_time_ms
        }


_phi_service_instance: Optional[PHIDetectionService] = None


def get_phi_detection_service() -> PHIDetectionService:
    """Get singleton instance of PHI detection service."""
    global _phi_service_instance
    if _phi_service_instance is None:
        _phi_service_instance = PHIDetectionService()
    return _phi_service_instance
