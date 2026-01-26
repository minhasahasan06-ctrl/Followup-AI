"""
Google Cloud Healthcare NLP and FHIR Service (Python)

Replaces AWS Comprehend Medical and HealthLake:
- Medical entity extraction
- PHI detection
- ICD-10, RxNorm, SNOMED CT coding
- FHIR R4 resource management
"""

import os
import logging
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any

from google.oauth2 import service_account
from google.auth import default as google_auth_default

from app.config.gcp_constants import GCP_CONFIG, HIPAA_AUDIT_ACTIONS, is_healthcare_configured

logger = logging.getLogger(__name__)


PHI_ENTITY_TYPES = [
    "DATE", "AGE", "LOCATION", "ID", "CONTACT", "NAME",
    "ADDRESS", "PHONE", "EMAIL", "URL", "SSN", "MRN", "ACCOUNT_NUMBER"
]


class GCPHealthcareNLPService:
    """
    Google Cloud Healthcare NLP API service.
    Replaces AWS Comprehend Medical.
    """
    
    def __init__(self):
        self.is_configured = is_healthcare_configured()
        self._credentials = None
        self._token = None
        self._token_expiry = 0
        
        if self.is_configured:
            logger.info("[Healthcare NLP] Service initialized")
        else:
            logger.warning("[Healthcare NLP] Healthcare API not configured, using fallback")
    
    async def _get_access_token(self) -> str:
        """Get OAuth 2.0 access token."""
        import time
        from google.auth.transport.requests import Request
        
        if self._token and time.time() < self._token_expiry:
            return self._token
        
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
                logger.error(f"[Healthcare NLP] Failed to get credentials: {e}")
                raise
        
        self._token, self._token_expiry = await asyncio.to_thread(_refresh_token)
        return self._token
    
    async def _call_healthcare_nlp(self, text: str) -> Dict:
        """Call the Healthcare NLP API."""
        import httpx
        
        token = await self._get_access_token()
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
                logger.error(f"[Healthcare NLP] API error: {response.status_code} - {response.text}")
                raise Exception(f"Healthcare NLP API error: {response.status_code}")
            
            return response.json()
    
    async def extract_medical_entities(self, text: str) -> Dict:
        """
        Extract medical entities from clinical text.
        
        Returns:
            Dict with entities, phiDetected, phiEntities, icdCodes, rxNormConcepts, snomedConcepts
        """
        self._audit_log("PHI_DETECTION", "healthcare-nlp", True, {"textLength": len(text)})
        
        if not self.is_configured:
            return self._empty_insights()
        
        try:
            result = await self._call_healthcare_nlp(text)
            return self._parse_nlp_response(result)
        except Exception as e:
            logger.error(f"[Healthcare NLP] Entity extraction failed: {e}")
            return self._empty_insights()
    
    def _parse_nlp_response(self, response: Dict) -> Dict:
        """Parse the Healthcare NLP API response."""
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
        }
        return categories.get(entity_type, "OTHER")
    
    def _empty_insights(self) -> Dict:
        """Return empty insights structure."""
        return {
            "entities": [],
            "phiDetected": False,
            "phiEntities": [],
            "icdCodes": [],
            "rxNormConcepts": [],
            "snomedConcepts": [],
        }
    
    def _audit_log(self, action: str, resource_type: str, success: bool, metadata: Optional[Dict] = None):
        """Log HIPAA audit entry."""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "action": action,
            "resource_type": resource_type,
            "success": success,
            "metadata": metadata or {},
        }
        logger.info(f"[HIPAA Audit] {log_entry}")
    
    async def detect_phi(self, text: str) -> Dict:
        """Detect PHI in text."""
        insights = await self.extract_medical_entities(text)
        return {
            "phiDetected": insights["phiDetected"],
            "phiEntities": insights["phiEntities"],
        }
    
    async def infer_icd10_codes(self, text: str) -> List[Dict]:
        """Infer ICD-10 codes from clinical text."""
        insights = await self.extract_medical_entities(text)
        return insights["icdCodes"]
    
    async def infer_rxnorm_codes(self, text: str) -> List[Dict]:
        """Infer RxNorm medication codes."""
        insights = await self.extract_medical_entities(text)
        return insights["rxNormConcepts"]
    
    async def infer_snomed_codes(self, text: str) -> List[Dict]:
        """Infer SNOMED CT codes."""
        insights = await self.extract_medical_entities(text)
        return insights["snomedConcepts"]


class GCPFHIRService:
    """
    Google Cloud Healthcare API FHIR Service.
    Replaces AWS HealthLake.
    """
    
    def __init__(self):
        self.is_configured = is_healthcare_configured()
        self._credentials = None
        self._token = None
        self._token_expiry = 0
        
        dataset_path = (
            f"projects/{GCP_CONFIG.PROJECT_ID}/locations/{GCP_CONFIG.HEALTHCARE.LOCATION}/"
            f"datasets/{GCP_CONFIG.HEALTHCARE.DATASET_ID}"
        )
        self.base_url = f"https://healthcare.googleapis.com/v1/{dataset_path}"
        
        if self.is_configured:
            logger.info("[FHIR Service] Initialized")
        else:
            logger.warning("[FHIR Service] Healthcare API not configured")
    
    async def _get_access_token(self) -> str:
        """Get OAuth 2.0 access token."""
        import time
        from google.auth.transport.requests import Request
        
        if self._token and time.time() < self._token_expiry:
            return self._token
        
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
                logger.error(f"[FHIR] Failed to get credentials: {e}")
                raise
        
        self._token, self._token_expiry = await asyncio.to_thread(_refresh_token)
        return self._token
    
    def _get_fhir_store_url(self) -> str:
        """Get the FHIR store URL."""
        return f"{self.base_url}/fhirStores/{GCP_CONFIG.HEALTHCARE.FHIR_STORE_ID}/fhir"
    
    def _audit_log(self, action: str, resource_type: str, resource_id: Optional[str] = None, metadata: Optional[Dict] = None):
        """Log HIPAA audit entry."""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "action": action,
            "resource_type": resource_type,
            "resource_id": resource_id,
            "metadata": metadata or {},
        }
        logger.info(f"[HIPAA Audit] {log_entry}")
    
    async def create_resource(self, resource: Dict) -> Dict:
        """Create a FHIR resource."""
        import httpx
        
        if not self.is_configured:
            raise RuntimeError("FHIR service not configured")
        
        resource_type = str(resource.get("resourceType", "Unknown"))
        self._audit_log("FHIR_WRITE", resource_type, metadata={"action": "create"})
        
        token = await self._get_access_token()
        url = f"{self._get_fhir_store_url()}/{resource_type}"
        
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/fhir+json",
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(url, headers=headers, json=resource, timeout=30.0)
            
            if response.status_code not in (200, 201):
                raise Exception(f"FHIR create error: {response.status_code}")
            
            return response.json()
    
    async def read_resource(self, resource_type: str, resource_id: str) -> Optional[Dict]:
        """Read a FHIR resource by ID."""
        import httpx
        
        if not self.is_configured:
            raise RuntimeError("FHIR service not configured")
        
        self._audit_log("FHIR_READ", resource_type, resource_id)
        
        token = await self._get_access_token()
        url = f"{self._get_fhir_store_url()}/{resource_type}/{resource_id}"
        
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/fhir+json",
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=headers, timeout=30.0)
            
            if response.status_code == 404:
                return None
            
            if response.status_code != 200:
                raise Exception(f"FHIR read error: {response.status_code}")
            
            return response.json()
    
    async def search_resources(self, resource_type: str, params: Optional[Dict[str, str]] = None) -> Dict:
        """Search FHIR resources."""
        import httpx
        
        if not self.is_configured:
            return {"resourceType": "Bundle", "type": "searchset", "total": 0, "entry": []}
        
        self._audit_log("FHIR_READ", resource_type, metadata={"searchParams": list((params or {}).keys())})
        
        token = await self._get_access_token()
        url = f"{self._get_fhir_store_url()}/{resource_type}"
        
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/fhir+json",
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=headers, params=params, timeout=30.0)
            
            if response.status_code != 200:
                return {"resourceType": "Bundle", "type": "searchset", "total": 0, "entry": []}
            
            return response.json()
    
    async def update_resource(self, resource: Dict) -> Dict:
        """Update a FHIR resource."""
        import httpx
        
        if not self.is_configured or not resource.get("id"):
            raise RuntimeError("FHIR service not configured or resource ID missing")
        
        resource_type = str(resource.get("resourceType", "Unknown"))
        resource_id = str(resource.get("id", ""))
        
        self._audit_log("FHIR_WRITE", resource_type, resource_id, {"action": "update"})
        
        token = await self._get_access_token()
        url = f"{self._get_fhir_store_url()}/{resource_type}/{resource_id}"
        
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/fhir+json",
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.put(url, headers=headers, json=resource, timeout=30.0)
            
            if response.status_code != 200:
                raise Exception(f"FHIR update error: {response.status_code}")
            
            return response.json()
    
    async def delete_resource(self, resource_type: str, resource_id: str) -> bool:
        """Delete a FHIR resource."""
        import httpx
        
        if not self.is_configured:
            raise RuntimeError("FHIR service not configured")
        
        self._audit_log("FHIR_WRITE", resource_type, resource_id, {"action": "delete"})
        
        token = await self._get_access_token()
        url = f"{self._get_fhir_store_url()}/{resource_type}/{resource_id}"
        
        headers = {
            "Authorization": f"Bearer {token}",
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.delete(url, headers=headers, timeout=30.0)
            return response.status_code in (200, 204)


healthcare_nlp_service = GCPHealthcareNLPService()
fhir_service = GCPFHIRService()
