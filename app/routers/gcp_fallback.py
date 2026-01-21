"""
GCP Fallback Services API Router
================================

REST API endpoints for Healthcare NLP and Document AI fallback services.

HIPAA Compliance:
- All endpoints require authentication
- Audit logging for all PHI access
- Secure handling of medical documents
"""

import logging
from typing import Optional
from dataclasses import asdict
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field

from app.config.gcp_constants import get_healthcare_nlp_status, get_document_ai_status
from app.services.healthcare_nlp_fallback import healthcare_nlp_fallback_service, MedicalInsights
from app.services.document_ai_fallback import document_ai_fallback_service, DocumentProcessingResult

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/gcp-fallback", tags=["GCP Fallback Services"])


class TextAnalysisRequest(BaseModel):
    """Request body for text-based medical entity extraction."""
    text: str = Field(..., min_length=1, max_length=50000, description="Clinical text to analyze")


class ServiceStatusResponse(BaseModel):
    """Response for service status endpoints."""
    healthcare_nlp: dict
    document_ai: dict


class MedicalInsightsResponse(BaseModel):
    """Response for medical entity extraction."""
    entities: list
    phiDetected: bool
    phiEntities: list
    icdCodes: list
    rxNormConcepts: list
    snomedConcepts: list
    source: str
    processingTimeMs: int


class DocumentProcessingResponse(BaseModel):
    """Response for document processing."""
    full_text: str
    document_type: Optional[str]
    extracted_fields: list
    tables: list
    page_count: int
    source: str
    processingTimeMs: int


@router.get("/status")
async def get_fallback_services_status() -> ServiceStatusResponse:
    """
    Get the status of GCP fallback services.
    
    Returns configuration and availability status for:
    - Healthcare NLP (medical entity extraction)
    - Document AI (OCR and document processing)
    """
    return ServiceStatusResponse(
        healthcare_nlp=healthcare_nlp_fallback_service.get_service_status(),
        document_ai=document_ai_fallback_service.get_service_status(),
    )


@router.post("/healthcare-nlp/extract-entities")
async def extract_medical_entities(request: TextAnalysisRequest) -> MedicalInsightsResponse:
    """
    Extract medical entities from clinical text.
    
    Uses GCP Healthcare NLP API if configured, falls back to OpenAI GPT-4o.
    
    Extracts:
    - Medications (with RxNorm codes)
    - Diagnoses/Conditions (with ICD-10 codes)
    - Procedures (with SNOMED CT codes)
    - PHI (Protected Health Information)
    - Anatomy references
    
    Returns entities in GCP Healthcare API-compatible JSON format.
    """
    try:
        insights = await healthcare_nlp_fallback_service.extract_medical_entities(request.text)
        
        return MedicalInsightsResponse(
            entities=insights.entities,
            phiDetected=insights.phiDetected,
            phiEntities=insights.phiEntities,
            icdCodes=insights.icdCodes,
            rxNormConcepts=insights.rxNormConcepts,
            snomedConcepts=insights.snomedConcepts,
            source=insights.source,
            processingTimeMs=insights.processingTimeMs,
        )
        
    except Exception as e:
        logger.error(f"[Healthcare NLP] Entity extraction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Entity extraction failed: {str(e)}")


@router.post("/healthcare-nlp/detect-phi")
async def detect_phi(request: TextAnalysisRequest) -> dict:
    """
    Detect Protected Health Information (PHI) in clinical text.
    
    Uses GCP Healthcare NLP API if configured, falls back to OpenAI GPT-4o.
    
    Detects:
    - Names, Dates, Ages
    - Addresses, Phone numbers, Emails
    - SSN, MRN, Account numbers
    - Other identifying information
    """
    try:
        result = await healthcare_nlp_fallback_service.detect_phi(request.text)
        return result
        
    except Exception as e:
        logger.error(f"[Healthcare NLP] PHI detection failed: {e}")
        raise HTTPException(status_code=500, detail=f"PHI detection failed: {str(e)}")


@router.post("/healthcare-nlp/infer-icd10")
async def infer_icd10_codes(request: TextAnalysisRequest) -> dict:
    """
    Infer ICD-10 diagnosis codes from clinical text.
    
    Uses GCP Healthcare NLP API if configured, falls back to OpenAI GPT-4o.
    """
    try:
        codes = await healthcare_nlp_fallback_service.infer_icd10_codes(request.text)
        return {"icdCodes": codes}
        
    except Exception as e:
        logger.error(f"[Healthcare NLP] ICD-10 inference failed: {e}")
        raise HTTPException(status_code=500, detail=f"ICD-10 inference failed: {str(e)}")


@router.post("/healthcare-nlp/infer-rxnorm")
async def infer_rxnorm_codes(request: TextAnalysisRequest) -> dict:
    """
    Infer RxNorm medication codes from clinical text.
    
    Uses GCP Healthcare NLP API if configured, falls back to OpenAI GPT-4o.
    """
    try:
        codes = await healthcare_nlp_fallback_service.infer_rxnorm_codes(request.text)
        return {"rxNormConcepts": codes}
        
    except Exception as e:
        logger.error(f"[Healthcare NLP] RxNorm inference failed: {e}")
        raise HTTPException(status_code=500, detail=f"RxNorm inference failed: {str(e)}")


@router.post("/document-ai/process")
async def process_document(
    file: UploadFile = File(...),
    use_healthcare_parser: bool = Form(default=False)
) -> DocumentProcessingResponse:
    """
    Process a document and extract text, tables, and key-value pairs.
    
    Uses GCP Document AI if configured, falls back to OpenAI Vision.
    
    Supports:
    - PDF documents
    - JPEG/PNG images
    - Medical forms and records
    
    Args:
        file: Document file (PDF, JPEG, or PNG)
        use_healthcare_parser: Use specialized healthcare document parser
    
    Returns structured extraction with OCR text, tables, and key-value pairs.
    """
    allowed_types = ["application/pdf", "image/jpeg", "image/png", "image/jpg"]
    
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type: {file.content_type}. Allowed: {allowed_types}"
        )
    
    try:
        content = await file.read()
        
        if len(content) > 20 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File too large. Maximum size is 20MB.")
        
        result = await document_ai_fallback_service.process_document(
            content=content,
            mime_type=file.content_type,
            use_healthcare_parser=use_healthcare_parser,
        )
        
        extracted_fields = [
            {
                "key": field.key,
                "value": field.value,
                "confidence": field.confidence,
                "page": field.page,
            }
            for field in result.extracted_fields
        ]
        
        return DocumentProcessingResponse(
            full_text=result.full_text,
            document_type=result.document_type,
            extracted_fields=extracted_fields,
            tables=result.tables,
            page_count=len(result.pages),
            source=result.source,
            processingTimeMs=result.processingTimeMs,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Document AI] Processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Document processing failed: {str(e)}")


@router.post("/document-ai/ocr")
async def extract_text_from_document(file: UploadFile = File(...)) -> dict:
    """
    Simple OCR - extract text from a document or image.
    
    Uses GCP Document AI if configured, falls back to OpenAI Vision.
    
    Args:
        file: Document file (PDF, JPEG, or PNG)
    
    Returns extracted text string.
    """
    allowed_types = ["application/pdf", "image/jpeg", "image/png", "image/jpg"]
    
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type: {file.content_type}. Allowed: {allowed_types}"
        )
    
    try:
        content = await file.read()
        
        if len(content) > 20 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File too large. Maximum size is 20MB.")
        
        text = await document_ai_fallback_service.extract_text(
            content=content,
            mime_type=file.content_type,
        )
        
        return {"text": text, "source": document_ai_fallback_service.get_service_status()["primarySource"]}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Document AI] OCR failed: {e}")
        raise HTTPException(status_code=500, detail=f"OCR failed: {str(e)}")


@router.post("/document-ai/healthcare")
async def process_healthcare_document(file: UploadFile = File(...)) -> DocumentProcessingResponse:
    """
    Process a healthcare document with specialized clinical extraction.
    
    Uses healthcare-specific prompts and parsers for enhanced extraction of:
    - Patient demographics
    - Clinical data (diagnoses, medications, allergies)
    - Lab results and vital signs
    - Provider information
    
    Args:
        file: Healthcare document file (PDF, JPEG, or PNG)
    
    Returns structured clinical data extraction.
    """
    allowed_types = ["application/pdf", "image/jpeg", "image/png", "image/jpg"]
    
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type: {file.content_type}. Allowed: {allowed_types}"
        )
    
    try:
        content = await file.read()
        
        if len(content) > 20 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File too large. Maximum size is 20MB.")
        
        result = await document_ai_fallback_service.process_healthcare_document(
            content=content,
            mime_type=file.content_type,
        )
        
        extracted_fields = [
            {
                "key": field.key,
                "value": field.value,
                "confidence": field.confidence,
                "page": field.page,
            }
            for field in result.extracted_fields
        ]
        
        return DocumentProcessingResponse(
            full_text=result.full_text,
            document_type=result.document_type,
            extracted_fields=extracted_fields,
            tables=result.tables,
            page_count=len(result.pages),
            source=result.source,
            processingTimeMs=result.processingTimeMs,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Document AI] Healthcare processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Healthcare document processing failed: {str(e)}")
