"""
Healthcare NLP Fallback Service
===============================

Production-grade fallback service that:
1. First attempts GCP Healthcare Natural Language API
2. Falls back to OpenAI GPT-4o if GCP fails or is not configured
3. Returns data in GCP Healthcare API-compatible JSON format

HIPAA Compliance:
- All API calls use HIPAA-compliant providers (GCP Healthcare API, OpenAI BAA)
- Clinical formatting maintained in prompts
- PHI handled securely with audit logging
"""

import os
import json
import logging
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

from openai import AsyncOpenAI

from app.config.gcp_constants import GCP_CONFIG, is_healthcare_configured

logger = logging.getLogger(__name__)


@dataclass
class MedicalEntity:
    """Represents an extracted medical entity."""
    text: str
    type: str
    category: str
    score: float
    offset: Optional[int] = None
    linkedEntities: Optional[List[Dict]] = None


@dataclass
class MedicalInsights:
    """Complete medical insights extraction result."""
    entities: List[Dict]
    phiDetected: bool
    phiEntities: List[Dict]
    icdCodes: List[Dict]
    rxNormConcepts: List[Dict]
    snomedConcepts: List[Dict]
    source: str  # "gcp" or "openai_fallback"
    processingTimeMs: int


OPENAI_MEDICAL_ENTITY_PROMPT = """You are a HIPAA-compliant clinical NLP system. Extract medical entities from the following clinical text.

IMPORTANT CLINICAL FORMATTING REQUIREMENTS:
- Use standardized medical terminology
- Maintain clinical precision in entity classification
- Follow ICD-10, RxNorm, and SNOMED CT coding conventions where applicable
- Do NOT include any commentary or explanations outside the JSON

Extract the following entity types:
1. MEDICATIONS: Drug names, dosages, frequencies, routes of administration
2. DIAGNOSES/CONDITIONS: Medical conditions, symptoms, diseases (map to ICD-10 when possible)
3. PROCEDURES: Medical procedures, surgeries, tests, examinations
4. ANATOMY: Body parts, organs, anatomical locations
5. PHI (Protected Health Information): Names, dates, addresses, phone numbers, emails, SSN, MRN, account numbers

For each entity provide:
- text: The exact text span from the input
- type: One of [MEDICATION, PROBLEM, PROCEDURE, ANATOMY, DATE, AGE, NAME, ADDRESS, PHONE, EMAIL, ID, LOCATION]
- category: Mapped category (MEDICATION, MEDICAL_CONDITION, PROCEDURE, ANATOMY, DATE_TIME, PROTECTED_HEALTH_INFORMATION)
- score: Confidence score 0.0-1.0
- linkedEntities: Array of related codes if identifiable:
  - For medications: RxNorm codes
  - For conditions: ICD-10 codes  
  - For procedures/conditions: SNOMED CT codes

Return ONLY valid JSON in this exact format:
{
    "entities": [
        {
            "text": "extracted text",
            "type": "MEDICATION|PROBLEM|PROCEDURE|ANATOMY|DATE|AGE|NAME|ADDRESS|PHONE|EMAIL|ID|LOCATION",
            "category": "MEDICATION|MEDICAL_CONDITION|PROCEDURE|ANATOMY|DATE_TIME|PROTECTED_HEALTH_INFORMATION|OTHER",
            "score": 0.95,
            "linkedEntities": [
                {"entityId": "code", "vocabulary": "RXNORM|ICD10|SNOMEDCT_US", "preferredTerm": "term"}
            ]
        }
    ]
}

Clinical Text to Analyze:
"""


PHI_ENTITY_TYPES = [
    "DATE", "AGE", "LOCATION", "ID", "CONTACT", "NAME",
    "ADDRESS", "PHONE", "EMAIL", "URL", "SSN", "MRN", "ACCOUNT_NUMBER"
]


class HealthcareNLPFallbackService:
    """
    Healthcare NLP service with intelligent fallback.
    
    Priority:
    1. GCP Healthcare Natural Language API (if configured)
    2. OpenAI GPT-4o fallback (HIPAA-compliant via BAA)
    """
    
    def __init__(self):
        self.gcp_configured = is_healthcare_configured()
        self._gcp_token = None
        self._gcp_token_expiry = 0
        self._openai_client: Optional[AsyncOpenAI] = None
        
        if self.gcp_configured:
            logger.info("[Healthcare NLP Fallback] GCP Healthcare API configured as primary")
        else:
            logger.info("[Healthcare NLP Fallback] Using OpenAI GPT-4o as primary (GCP not configured)")
        
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("AI_INTEGRATIONS_OPENAI_API_KEY")
        if api_key:
            self._openai_client = AsyncOpenAI(api_key=api_key)
            logger.info("[Healthcare NLP Fallback] OpenAI fallback initialized")
        else:
            logger.warning("[Healthcare NLP Fallback] OpenAI API key not found - fallback unavailable")
    
    async def _get_gcp_access_token(self) -> str:
        """Get OAuth 2.0 access token for GCP Healthcare API."""
        import time
        from google.auth.transport.requests import Request
        from google.auth import default as google_auth_default
        
        if self._gcp_token and time.time() < self._gcp_token_expiry:
            return self._gcp_token
        
        def _refresh_token():
            try:
                credentials, project = google_auth_default(
                    scopes=["https://www.googleapis.com/auth/cloud-healthcare"]
                )
                if hasattr(credentials, 'refresh'):
                    credentials.refresh(Request())
                token = getattr(credentials, 'token', None)
                expiry = getattr(credentials, 'expiry', None)
                expiry_ts = expiry.timestamp() if expiry else time.time() + 3600
                return token, expiry_ts
            except Exception as e:
                logger.error(f"[Healthcare NLP Fallback] GCP credentials error: {e}")
                raise
        
        self._gcp_token, self._gcp_token_expiry = await asyncio.to_thread(_refresh_token)
        return self._gcp_token
    
    async def _call_gcp_healthcare_nlp(self, text: str) -> Dict:
        """Call the GCP Healthcare Natural Language API."""
        import httpx
        
        token = await self._get_gcp_access_token()
        url = (
            f"https://healthcare.googleapis.com/v1/projects/{GCP_CONFIG.PROJECT_ID}/"
            f"locations/{GCP_CONFIG.HEALTHCARE.LOCATION}/services/nlp:analyzeEntities"
        )
        
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        
        body = {
            "nlpService": f"projects/{GCP_CONFIG.PROJECT_ID}/locations/{GCP_CONFIG.HEALTHCARE.LOCATION}/services/nlp",
            "documentContent": text,
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(url, headers=headers, json=body, timeout=30.0)
            
            if response.status_code != 200:
                raise Exception(f"GCP Healthcare NLP API error: {response.status_code} - {response.text}")
            
            return response.json()
    
    def _parse_gcp_response(self, response: Dict) -> Dict:
        """Parse GCP Healthcare NLP API response into standardized format."""
        entities = []
        phi_entities = []
        icd_codes = []
        rxnorm_concepts = []
        snomed_concepts = []
        
        entity_mentions = response.get("entityMentions", [])
        
        for mention in entity_mentions:
            entity = {
                "text": mention.get("text", {}).get("content", ""),
                "type": mention.get("type", "UNKNOWN"),
                "category": self._categorize_entity(mention.get("type", "")),
                "score": mention.get("confidence", 0),
                "offset": mention.get("text", {}).get("beginOffset"),
            }
            
            linked_entities = []
            for linked in mention.get("linkedEntities", []):
                linked_entities.append({
                    "entityId": linked.get("entityId", ""),
                    "vocabulary": linked.get("vocabulary", ""),
                    "preferredTerm": linked.get("preferredTerm"),
                })
                
                entity_id = linked.get("entityId", "")
                vocab = linked.get("vocabulary", "")
                
                if entity_id.startswith("ICD"):
                    icd_codes.append({
                        "code": entity_id,
                        "description": linked.get("preferredTerm", ""),
                        "score": mention.get("confidence", 0),
                    })
                elif vocab == "RXNORM":
                    rxnorm_concepts.append({
                        "code": entity_id,
                        "description": linked.get("preferredTerm", ""),
                        "score": mention.get("confidence", 0),
                    })
                elif vocab == "SNOMEDCT_US":
                    snomed_concepts.append({
                        "code": entity_id,
                        "description": linked.get("preferredTerm", ""),
                        "score": mention.get("confidence", 0),
                    })
            
            entity["linkedEntities"] = linked_entities
            entities.append(entity)
            
            if mention.get("type") in PHI_ENTITY_TYPES:
                phi_entities.append(entity)
        
        return {
            "entities": entities,
            "phiDetected": len(phi_entities) > 0,
            "phiEntities": phi_entities,
            "icdCodes": icd_codes,
            "rxNormConcepts": rxnorm_concepts,
            "snomedConcepts": snomed_concepts,
        }
    
    async def _call_openai_fallback(self, text: str) -> Dict:
        """Use OpenAI GPT-4o as fallback for medical entity extraction."""
        if not self._openai_client:
            raise Exception("OpenAI client not initialized")
        
        try:
            response = await self._openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a HIPAA-compliant clinical NLP system that extracts medical entities from clinical text. Always respond with valid JSON only."
                    },
                    {
                        "role": "user",
                        "content": OPENAI_MEDICAL_ENTITY_PROMPT + text
                    }
                ],
                response_format={"type": "json_object"},
                max_tokens=4000,
                temperature=0.1,
            )
            
            result = response.choices[0].message.content
            return json.loads(result)
            
        except Exception as e:
            logger.error(f"[Healthcare NLP Fallback] OpenAI call failed: {e}")
            raise
    
    def _parse_openai_response(self, response: Dict) -> Dict:
        """Parse OpenAI response into GCP-compatible format."""
        entities = response.get("entities", [])
        phi_entities = []
        icd_codes = []
        rxnorm_concepts = []
        snomed_concepts = []
        
        for entity in entities:
            entity_type = entity.get("type", "")
            
            if entity_type in PHI_ENTITY_TYPES:
                phi_entities.append(entity)
            
            for linked in entity.get("linkedEntities", []):
                vocab = linked.get("vocabulary", "")
                entity_id = linked.get("entityId", "")
                
                if vocab == "ICD10" or entity_id.startswith("ICD"):
                    icd_codes.append({
                        "code": entity_id,
                        "description": linked.get("preferredTerm", ""),
                        "score": entity.get("score", 0),
                    })
                elif vocab == "RXNORM":
                    rxnorm_concepts.append({
                        "code": entity_id,
                        "description": linked.get("preferredTerm", ""),
                        "score": entity.get("score", 0),
                    })
                elif vocab == "SNOMEDCT_US":
                    snomed_concepts.append({
                        "code": entity_id,
                        "description": linked.get("preferredTerm", ""),
                        "score": entity.get("score", 0),
                    })
        
        return {
            "entities": entities,
            "phiDetected": len(phi_entities) > 0,
            "phiEntities": phi_entities,
            "icdCodes": icd_codes,
            "rxNormConcepts": rxnorm_concepts,
            "snomedConcepts": snomed_concepts,
        }
    
    def _categorize_entity(self, entity_type: str) -> str:
        """Map entity type to category."""
        categories = {
            "PROBLEM": "MEDICAL_CONDITION",
            "MEDICATION": "MEDICATION",
            "PROCEDURE": "PROCEDURE",
            "ANATOMY": "ANATOMY",
            "DATE": "DATE_TIME",
            "AGE": "PROTECTED_HEALTH_INFORMATION",
            "NAME": "PROTECTED_HEALTH_INFORMATION",
            "ADDRESS": "PROTECTED_HEALTH_INFORMATION",
            "PHONE": "PROTECTED_HEALTH_INFORMATION",
            "EMAIL": "PROTECTED_HEALTH_INFORMATION",
            "ID": "PROTECTED_HEALTH_INFORMATION",
            "LOCATION": "PROTECTED_HEALTH_INFORMATION",
        }
        return categories.get(entity_type, "OTHER")
    
    def _audit_log(self, action: str, source: str, success: bool, metadata: Optional[Dict] = None):
        """Log HIPAA audit entry."""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "action": action,
            "source": source,
            "success": success,
            "metadata": metadata or {},
        }
        logger.info(f"[HIPAA Audit] {log_entry}")
    
    async def extract_medical_entities(self, text: str) -> MedicalInsights:
        """
        Extract medical entities from clinical text.
        
        Uses GCP Healthcare NLP API if configured, falls back to OpenAI GPT-4o.
        
        Returns:
            MedicalInsights with entities, PHI detection, and medical codes.
        """
        import time
        start_time = time.time()
        source = "unknown"
        
        self._audit_log("MEDICAL_ENTITY_EXTRACTION", "healthcare-nlp-fallback", True, {"textLength": len(text)})
        
        try:
            if self.gcp_configured:
                try:
                    gcp_response = await self._call_gcp_healthcare_nlp(text)
                    result = self._parse_gcp_response(gcp_response)
                    source = "gcp"
                    logger.info("[Healthcare NLP Fallback] Successfully used GCP Healthcare API")
                except Exception as gcp_error:
                    logger.warning(f"[Healthcare NLP Fallback] GCP failed, falling back to OpenAI: {gcp_error}")
                    openai_response = await self._call_openai_fallback(text)
                    result = self._parse_openai_response(openai_response)
                    source = "openai_fallback"
            else:
                openai_response = await self._call_openai_fallback(text)
                result = self._parse_openai_response(openai_response)
                source = "openai_fallback"
            
            processing_time_ms = int((time.time() - start_time) * 1000)
            
            return MedicalInsights(
                entities=result["entities"],
                phiDetected=result["phiDetected"],
                phiEntities=result["phiEntities"],
                icdCodes=result["icdCodes"],
                rxNormConcepts=result["rxNormConcepts"],
                snomedConcepts=result["snomedConcepts"],
                source=source,
                processingTimeMs=processing_time_ms,
            )
            
        except Exception as e:
            logger.error(f"[Healthcare NLP Fallback] All extraction methods failed: {e}")
            processing_time_ms = int((time.time() - start_time) * 1000)
            return MedicalInsights(
                entities=[],
                phiDetected=False,
                phiEntities=[],
                icdCodes=[],
                rxNormConcepts=[],
                snomedConcepts=[],
                source="error",
                processingTimeMs=processing_time_ms,
            )
    
    async def detect_phi(self, text: str) -> Dict:
        """Detect PHI in text."""
        insights = await self.extract_medical_entities(text)
        return {
            "phiDetected": insights.phiDetected,
            "phiEntities": insights.phiEntities,
            "source": insights.source,
        }
    
    async def infer_icd10_codes(self, text: str) -> List[Dict]:
        """Infer ICD-10 codes from clinical text."""
        insights = await self.extract_medical_entities(text)
        return insights.icdCodes
    
    async def infer_rxnorm_codes(self, text: str) -> List[Dict]:
        """Infer RxNorm medication codes."""
        insights = await self.extract_medical_entities(text)
        return insights.rxNormConcepts
    
    async def infer_snomed_codes(self, text: str) -> List[Dict]:
        """Infer SNOMED CT codes."""
        insights = await self.extract_medical_entities(text)
        return insights.snomedConcepts
    
    def get_service_status(self) -> Dict:
        """Get current service configuration status."""
        return {
            "gcpConfigured": self.gcp_configured,
            "openaiAvailable": self._openai_client is not None,
            "primarySource": "gcp" if self.gcp_configured else "openai",
            "fallbackAvailable": self._openai_client is not None,
        }


healthcare_nlp_fallback_service = HealthcareNLPFallbackService()
