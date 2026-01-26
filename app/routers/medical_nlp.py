"""
Medical NLP Router - OpenAI GPT-4o Based
Provides REST API endpoints for PHI detection and medical entity extraction.

Replaces AWS Comprehend Medical with OpenAI GPT-4o for:
- PHI detection and redaction
- Medical entity extraction
- ICD-10-CM code inference
- RxNorm concept matching
- SNOMED-CT concept identification
"""

import logging
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.database import get_db
from app.services.phi_detection_service import (
    get_phi_detection_service,
    PHIDetectionResult,
    MedicalNLPResult,
    PHICategory,
    MedicalEntityCategory
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/medical-nlp", tags=["Medical NLP"])


class PHIDetectionRequest(BaseModel):
    text: str = Field(..., description="Text to analyze for PHI")
    patient_name: Optional[str] = Field(None, description="Known patient name for targeted redaction")
    preserve_email_domains: Optional[bool] = Field(False, description="Keep email domain visible")


class PHIEntityResponse(BaseModel):
    text: str
    category: str
    start_offset: int
    end_offset: int
    confidence: float
    placeholder: str


class PHIDetectionResponse(BaseModel):
    original_text: str
    redacted_text: str
    phi_detected: bool
    phi_entities: List[PHIEntityResponse]
    redaction_count: int
    processing_time_ms: float


class MedicalEntityResponse(BaseModel):
    text: str
    category: str
    confidence: float
    traits: Optional[List[Dict[str, Any]]] = None
    attributes: Optional[List[Dict[str, Any]]] = None
    icd10_codes: Optional[List[Dict[str, Any]]] = None
    rxnorm_concepts: Optional[List[Dict[str, Any]]] = None
    snomed_concepts: Optional[List[Dict[str, Any]]] = None


class MedicalEntityExtractionRequest(BaseModel):
    text: str = Field(..., description="Clinical text to analyze")
    include_phi_check: Optional[bool] = Field(True, description="Also check for PHI")


class MedicalEntityExtractionResponse(BaseModel):
    text: str
    entities: List[MedicalEntityResponse]
    icd10_codes: List[Dict[str, Any]]
    rxnorm_concepts: List[Dict[str, Any]]
    snomed_concepts: List[Dict[str, Any]]
    phi_detected: bool
    phi_entities: List[PHIEntityResponse]


class ICD10InferenceRequest(BaseModel):
    text: str = Field(..., description="Clinical text for ICD-10 code inference")


class ICD10InferenceResponse(BaseModel):
    codes: List[Dict[str, Any]]


class RxNormInferenceRequest(BaseModel):
    text: str = Field(..., description="Medication text for RxNorm concept matching")


class RxNormInferenceResponse(BaseModel):
    concepts: List[Dict[str, Any]]


class SNOMEDInferenceRequest(BaseModel):
    text: str = Field(..., description="Clinical text for SNOMED-CT concept matching")


class SNOMEDInferenceResponse(BaseModel):
    concepts: List[Dict[str, Any]]


class SanitizeForAIRequest(BaseModel):
    text: str = Field(..., description="Text to sanitize for AI processing")
    patient_name: Optional[str] = Field(None, description="Known patient name")


class SanitizeForAIResponse(BaseModel):
    sanitized_text: str
    was_redacted: bool
    redaction_count: int
    phi_categories: List[str]
    processing_time_ms: float


class EmailSanitizeRequest(BaseModel):
    subject: str = Field(..., description="Email subject")
    content: str = Field(..., description="Email content")
    patient_name: Optional[str] = Field(None, description="Known patient name")
    sender_name: Optional[str] = Field(None, description="Sender name to redact")


class EmailSanitizeResponse(BaseModel):
    subject: str
    content: str
    was_redacted: bool
    redaction_count: int


class HIPAAComplianceStatus(BaseModel):
    is_configured: bool
    can_use_ai: bool
    warnings: List[str]
    baa_signed: bool
    zdr_enabled: bool
    enterprise: bool


def _map_phi_category_to_legacy(category: str) -> str:
    """Map PHI category to legacy type expected by TypeScript consumers."""
    category_map = {
        "NAME": "name",
        "DATE": "date",
        "PHONE": "phone",
        "EMAIL": "email",
        "ADDRESS": "address",
        "SSN": "ssn",
        "MRN": "mrn",
        "AGE": "other",
        "ID": "mrn",
        "URL": "other",
        "IP_ADDRESS": "other",
        "DEVICE_ID": "other",
        "BIOMETRIC": "other",
        "PHOTO": "other",
        "OTHER": "other"
    }
    return category_map.get(category, "other")


@router.post("/detect-phi", response_model=PHIDetectionResponse)
async def detect_phi(
    request: PHIDetectionRequest,
    db: Session = Depends(get_db)
) -> PHIDetectionResponse:
    """
    Detect and redact Protected Health Information (PHI) from text.
    
    Uses GPT-4o for comprehensive PHI detection following HIPAA's 18 identifiers.
    Falls back to regex-based detection if API fails.
    """
    try:
        service = get_phi_detection_service()
        
        if request.patient_name:
            redacted_text, phi_entities = service.redact_phi_with_context(
                request.text,
                patient_name=request.patient_name
            )
            base_result = service.detect_phi(request.text)
            
            return PHIDetectionResponse(
                original_text=base_result.original_text,
                redacted_text=redacted_text,
                phi_detected=base_result.phi_detected,
                phi_entities=[
                    PHIEntityResponse(
                        text=e.text,
                        category=_map_phi_category_to_legacy(e.category.value if hasattr(e.category, 'value') else str(e.category)),
                        start_offset=e.start_offset,
                        end_offset=e.end_offset,
                        confidence=e.confidence,
                        placeholder=e.placeholder
                    )
                    for e in base_result.phi_entities
                ],
                redaction_count=base_result.redaction_count,
                processing_time_ms=base_result.processing_time_ms
            )
        else:
            result = await service.detect_phi_async(request.text)
        
            return PHIDetectionResponse(
                original_text=result.original_text,
                redacted_text=result.redacted_text,
                phi_detected=result.phi_detected,
                phi_entities=[
                    PHIEntityResponse(
                        text=e.text,
                        category=_map_phi_category_to_legacy(e.category.value if hasattr(e.category, 'value') else str(e.category)),
                        start_offset=e.start_offset,
                        end_offset=e.end_offset,
                        confidence=e.confidence,
                        placeholder=e.placeholder
                    )
                    for e in result.phi_entities
                ],
                redaction_count=result.redaction_count,
                processing_time_ms=result.processing_time_ms
            )
        
    except Exception as e:
        logger.error(f"PHI detection failed: {e}")
        raise HTTPException(status_code=500, detail=f"PHI detection failed: {str(e)}")


@router.post("/extract-entities", response_model=MedicalEntityExtractionResponse)
async def extract_medical_entities(
    request: MedicalEntityExtractionRequest,
    db: Session = Depends(get_db)
) -> MedicalEntityExtractionResponse:
    """
    Extract medical entities from clinical text.
    
    Identifies medications, conditions, procedures, anatomy, and more.
    Also provides ICD-10, RxNorm, and SNOMED-CT codes.
    """
    try:
        service = get_phi_detection_service()
        result = await service.extract_medical_entities_async(request.text)
        
        return MedicalEntityExtractionResponse(
            text=result.text,
            entities=[
                MedicalEntityResponse(
                    text=e.text,
                    category=e.category.value,
                    confidence=e.confidence,
                    traits=e.traits,
                    attributes=e.attributes,
                    icd10_codes=e.icd10_codes,
                    rxnorm_concepts=e.rxnorm_concepts,
                    snomed_concepts=e.snomed_concepts
                )
                for e in result.entities
            ],
            icd10_codes=result.icd10_codes,
            rxnorm_concepts=result.rxnorm_concepts,
            snomed_concepts=result.snomed_concepts,
            phi_detected=result.phi_detected,
            phi_entities=[
                PHIEntityResponse(
                    text=e.text,
                    category=e.category.value,
                    start_offset=e.start_offset,
                    end_offset=e.end_offset,
                    confidence=e.confidence,
                    placeholder=e.placeholder
                )
                for e in result.phi_entities
            ]
        )
        
    except Exception as e:
        logger.error(f"Medical entity extraction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Entity extraction failed: {str(e)}")


@router.post("/infer-icd10", response_model=ICD10InferenceResponse)
async def infer_icd10_codes(
    request: ICD10InferenceRequest,
    db: Session = Depends(get_db)
) -> ICD10InferenceResponse:
    """
    Infer ICD-10-CM diagnosis codes from clinical text.
    
    Useful for automated medical coding and diagnosis documentation.
    """
    try:
        service = get_phi_detection_service()
        codes = service.infer_icd10_codes(request.text)
        
        return ICD10InferenceResponse(codes=codes)
        
    except Exception as e:
        logger.error(f"ICD-10 inference failed: {e}")
        raise HTTPException(status_code=500, detail=f"ICD-10 inference failed: {str(e)}")


@router.post("/infer-rxnorm", response_model=RxNormInferenceResponse)
async def infer_rxnorm_concepts(
    request: RxNormInferenceRequest,
    db: Session = Depends(get_db)
) -> RxNormInferenceResponse:
    """
    Infer RxNorm concepts from medication text.
    
    Identifies medications and provides standardized RxNorm codes.
    """
    try:
        service = get_phi_detection_service()
        concepts = service.infer_rxnorm_concepts(request.text)
        
        return RxNormInferenceResponse(concepts=concepts)
        
    except Exception as e:
        logger.error(f"RxNorm inference failed: {e}")
        raise HTTPException(status_code=500, detail=f"RxNorm inference failed: {str(e)}")


@router.post("/infer-snomed", response_model=SNOMEDInferenceResponse)
async def infer_snomed_concepts(
    request: SNOMEDInferenceRequest,
    db: Session = Depends(get_db)
) -> SNOMEDInferenceResponse:
    """
    Infer SNOMED-CT concepts from clinical text.
    
    Provides standardized clinical terminology codes.
    """
    try:
        service = get_phi_detection_service()
        concepts = service.infer_snomed_concepts(request.text)
        
        return SNOMEDInferenceResponse(concepts=concepts)
        
    except Exception as e:
        logger.error(f"SNOMED inference failed: {e}")
        raise HTTPException(status_code=500, detail=f"SNOMED inference failed: {str(e)}")


@router.post("/sanitize-for-ai", response_model=SanitizeForAIResponse)
async def sanitize_for_ai(
    request: SanitizeForAIRequest,
    db: Session = Depends(get_db)
) -> SanitizeForAIResponse:
    """
    Sanitize text for AI processing by detecting and redacting PHI.
    
    Use this before sending patient data to any AI service.
    """
    try:
        service = get_phi_detection_service()
        result = service.sanitize_for_ai(request.text, patient_name=request.patient_name)
        
        return SanitizeForAIResponse(
            sanitized_text=result["sanitized_text"],
            was_redacted=result["was_redacted"],
            redaction_count=result["redaction_count"],
            phi_categories=result["phi_categories"],
            processing_time_ms=result["processing_time_ms"]
        )
        
    except Exception as e:
        logger.error(f"Sanitization failed: {e}")
        raise HTTPException(status_code=500, detail=f"Sanitization failed: {str(e)}")


@router.post("/sanitize-email", response_model=EmailSanitizeResponse)
async def sanitize_email(
    request: EmailSanitizeRequest,
    db: Session = Depends(get_db)
) -> EmailSanitizeResponse:
    """
    Sanitize email content for AI processing.
    
    Redacts PHI from both subject and content.
    """
    try:
        service = get_phi_detection_service()
        
        subject_result = await service.detect_phi_async(request.subject)
        content_result = await service.detect_phi_async(request.content)
        
        sanitized_subject = subject_result.redacted_text
        sanitized_content = content_result.redacted_text
        
        if request.patient_name:
            name_parts = request.patient_name.split()
            for part in name_parts:
                if len(part) > 1:
                    import re
                    pattern = re.compile(re.escape(part), re.IGNORECASE)
                    sanitized_subject = pattern.sub("[PATIENT_NAME]", sanitized_subject)
                    sanitized_content = pattern.sub("[PATIENT_NAME]", sanitized_content)
        
        if request.sender_name:
            name_parts = request.sender_name.split()
            for part in name_parts:
                if len(part) > 1:
                    import re
                    pattern = re.compile(re.escape(part), re.IGNORECASE)
                    sanitized_subject = pattern.sub("[SENDER_NAME]", sanitized_subject)
                    sanitized_content = pattern.sub("[SENDER_NAME]", sanitized_content)
        
        total_redactions = subject_result.redaction_count + content_result.redaction_count
        
        return EmailSanitizeResponse(
            subject=sanitized_subject,
            content=sanitized_content,
            was_redacted=total_redactions > 0,
            redaction_count=total_redactions
        )
        
    except Exception as e:
        logger.error(f"Email sanitization failed: {e}")
        raise HTTPException(status_code=500, detail=f"Email sanitization failed: {str(e)}")


@router.get("/hipaa-status", response_model=HIPAAComplianceStatus)
async def get_hipaa_status() -> HIPAAComplianceStatus:
    """
    Check HIPAA compliance status for AI services.
    
    Validates that required configurations are in place.
    """
    import os
    
    baa_signed = os.environ.get("OPENAI_BAA_SIGNED", "").lower() == "true"
    zdr_enabled = os.environ.get("OPENAI_ZDR_ENABLED", "").lower() == "true"
    enterprise = os.environ.get("OPENAI_ENTERPRISE", "").lower() == "true"
    has_api_key = bool(os.environ.get("OPENAI_API_KEY"))
    
    warnings = []
    can_use_ai = True
    
    if not has_api_key:
        warnings.append("OpenAI API key not configured")
        can_use_ai = False
    
    if not baa_signed:
        warnings.append("CRITICAL: Business Associate Agreement (BAA) with OpenAI NOT signed")
        can_use_ai = False
    
    if not zdr_enabled:
        warnings.append("IMPORTANT: Zero Data Retention (ZDR) not enabled")
    
    if not enterprise:
        warnings.append("NOTICE: OpenAI Enterprise plan recommended")
    
    return HIPAAComplianceStatus(
        is_configured=len(warnings) == 0,
        can_use_ai=can_use_ai,
        warnings=warnings,
        baa_signed=baa_signed,
        zdr_enabled=zdr_enabled,
        enterprise=enterprise
    )


@router.get("/health")
async def health_check():
    """Health check endpoint for Medical NLP service."""
    try:
        service = get_phi_detection_service()
        return {
            "status": "healthy",
            "service": "Medical NLP (OpenAI GPT-4o)",
            "features": [
                "PHI Detection",
                "Medical Entity Extraction",
                "ICD-10-CM Inference",
                "RxNorm Concept Matching",
                "SNOMED-CT Identification"
            ]
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }
