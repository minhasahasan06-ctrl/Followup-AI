"""
Document AI Fallback Service
=============================

Production-grade fallback service that:
1. First attempts GCP Document AI for OCR and document processing
2. Falls back to OpenAI Vision (GPT-4o) if Document AI is not configured
3. Supports medical document processing, OCR, and key-value extraction

HIPAA Compliance:
- All API calls use HIPAA-compliant providers (GCP Document AI, OpenAI BAA)
- Clinical formatting maintained in prompts
- PHI handled securely with audit logging
"""

import os
import json
import base64
import logging
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass

from openai import AsyncOpenAI

from app.config.gcp_constants import GCP_CONFIG, is_document_ai_configured

logger = logging.getLogger(__name__)


@dataclass
class ExtractedField:
    """Represents an extracted key-value field from a document."""
    key: str
    value: str
    confidence: float
    page: int
    bounding_box: Optional[Dict] = None


@dataclass
class DocumentPage:
    """Represents a processed document page."""
    page_number: int
    text: str
    tables: List[Dict]
    key_value_pairs: List[ExtractedField]


@dataclass
class DocumentProcessingResult:
    """Complete document processing result."""
    full_text: str
    pages: List[DocumentPage]
    extracted_fields: List[ExtractedField]
    tables: List[Dict]
    document_type: Optional[str]
    source: str  # "gcp" or "openai_fallback"
    processingTimeMs: int


OPENAI_DOCUMENT_OCR_PROMPT = """You are a HIPAA-compliant medical document processing system. Analyze this medical document image and extract all text and structured data.

CRITICAL CLINICAL FORMATTING REQUIREMENTS:
- Preserve exact formatting of medical values (dosages, dates, measurements)
- Maintain clinical precision in extracted data
- Handle PHI (Protected Health Information) with care
- Do NOT interpret or modify any medical values

Perform the following extractions:

1. FULL TEXT OCR: Extract all readable text from the document, preserving layout as much as possible

2. KEY-VALUE PAIRS: Identify and extract structured data fields commonly found in medical documents:
   - Patient demographics (name, DOB, MRN, address, phone)
   - Provider information (doctor name, NPI, facility)
   - Clinical data (diagnoses, medications, allergies, vitals)
   - Insurance information (if present)
   - Dates (admission, discharge, service dates)
   - Lab values and test results

3. TABLES: If tables are present, extract them with headers and values

4. DOCUMENT TYPE: Identify the type of medical document (e.g., "Lab Report", "Prescription", "Discharge Summary", "Insurance Card", "Referral", "Progress Note")

Return ONLY valid JSON in this exact format:
{
    "full_text": "complete OCR text with preserved formatting",
    "document_type": "identified document type",
    "key_value_pairs": [
        {
            "key": "field name",
            "value": "extracted value",
            "confidence": 0.95,
            "page": 1
        }
    ],
    "tables": [
        {
            "title": "table title if any",
            "headers": ["column1", "column2"],
            "rows": [["value1", "value2"]]
        }
    ]
}
"""

OPENAI_HEALTHCARE_DOCUMENT_PROMPT = """You are a HIPAA-compliant healthcare document parser specialized in clinical documents. Analyze this medical document and extract structured clinical data.

CLINICAL FORMATTING REQUIREMENTS:
- Use standardized medical terminology
- Preserve exact medication dosages and frequencies
- Maintain clinical precision in all extracted values
- Follow FHIR-compatible data structures where applicable

Extract the following structured data:

1. PATIENT INFORMATION:
   - Full name, DOB, gender, MRN
   - Contact information, emergency contacts
   - Insurance details

2. CLINICAL DATA:
   - Chief complaint/reason for visit
   - History of present illness
   - Diagnoses (with ICD-10 codes if visible)
   - Medications (name, dose, frequency, route)
   - Allergies (substance, reaction type)
   - Vital signs
   - Lab results (test name, value, units, reference range)
   - Procedures performed

3. PROVIDER INFORMATION:
   - Attending physician, specialists
   - Facility name, department
   - Date of service

Return ONLY valid JSON in this exact format:
{
    "full_text": "complete document text",
    "document_type": "clinical document type",
    "patient": {
        "name": "",
        "dob": "",
        "mrn": "",
        "gender": ""
    },
    "clinical_data": {
        "chief_complaint": "",
        "diagnoses": [{"code": "", "description": ""}],
        "medications": [{"name": "", "dose": "", "frequency": "", "route": ""}],
        "allergies": [{"substance": "", "reaction": ""}],
        "vitals": {},
        "lab_results": [{"test": "", "value": "", "units": "", "reference_range": ""}]
    },
    "provider": {
        "name": "",
        "facility": "",
        "date_of_service": ""
    },
    "key_value_pairs": [
        {"key": "", "value": "", "confidence": 0.95, "page": 1}
    ]
}
"""


class DocumentAIFallbackService:
    """
    Document AI service with intelligent fallback.
    
    Priority:
    1. GCP Document AI (if configured)
    2. OpenAI Vision GPT-4o fallback (HIPAA-compliant via BAA)
    """
    
    def __init__(self):
        self.gcp_configured = is_document_ai_configured()
        self._gcp_token = None
        self._gcp_token_expiry = 0
        self._openai_client: Optional[AsyncOpenAI] = None
        
        if self.gcp_configured:
            logger.info("[Document AI Fallback] GCP Document AI configured as primary")
        else:
            logger.info("[Document AI Fallback] Using OpenAI Vision as primary (GCP not configured)")
        
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("AI_INTEGRATIONS_OPENAI_API_KEY")
        if api_key:
            self._openai_client = AsyncOpenAI(api_key=api_key)
            logger.info("[Document AI Fallback] OpenAI Vision fallback initialized")
        else:
            logger.warning("[Document AI Fallback] OpenAI API key not found - fallback unavailable")
    
    async def _get_gcp_access_token(self) -> str:
        """Get OAuth 2.0 access token for GCP Document AI."""
        import time
        from google.auth.transport.requests import Request
        from google.auth import default as google_auth_default
        
        if self._gcp_token and time.time() < self._gcp_token_expiry:
            return self._gcp_token
        
        def _refresh_token():
            try:
                credentials, project = google_auth_default(
                    scopes=["https://www.googleapis.com/auth/cloud-platform"]
                )
                if hasattr(credentials, 'refresh'):
                    credentials.refresh(Request())
                token = getattr(credentials, 'token', None)
                expiry = getattr(credentials, 'expiry', None)
                expiry_ts = expiry.timestamp() if expiry else time.time() + 3600
                return token, expiry_ts
            except Exception as e:
                logger.error(f"[Document AI Fallback] GCP credentials error: {e}")
                raise
        
        self._gcp_token, self._gcp_token_expiry = await asyncio.to_thread(_refresh_token)
        return self._gcp_token
    
    async def _call_gcp_document_ai(
        self, 
        content: bytes, 
        mime_type: str = "application/pdf",
        use_healthcare_parser: bool = False
    ) -> Dict:
        """Call GCP Document AI API."""
        import httpx
        
        token = await self._get_gcp_access_token()
        
        processor_id = (
            GCP_CONFIG.DOCUMENT_AI.HEALTHCARE_PROCESSOR_ID 
            if use_healthcare_parser and GCP_CONFIG.DOCUMENT_AI.HEALTHCARE_PROCESSOR_ID
            else GCP_CONFIG.DOCUMENT_AI.PROCESSOR_ID
        )
        
        url = (
            f"https://{GCP_CONFIG.DOCUMENT_AI.LOCATION}-documentai.googleapis.com/v1/"
            f"projects/{GCP_CONFIG.PROJECT_ID}/locations/{GCP_CONFIG.DOCUMENT_AI.LOCATION}/"
            f"processors/{processor_id}:process"
        )
        
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        
        encoded_content = base64.b64encode(content).decode('utf-8')
        
        body = {
            "rawDocument": {
                "content": encoded_content,
                "mimeType": mime_type,
            }
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(url, headers=headers, json=body, timeout=60.0)
            
            if response.status_code != 200:
                raise Exception(f"GCP Document AI error: {response.status_code} - {response.text}")
            
            return response.json()
    
    def _parse_gcp_document_response(self, response: Dict) -> Dict:
        """Parse GCP Document AI response into standardized format."""
        document = response.get("document", {})
        
        full_text = document.get("text", "")
        pages = []
        extracted_fields = []
        tables = []
        
        for page_data in document.get("pages", []):
            page_num = page_data.get("pageNumber", 1)
            page_text = ""
            page_kvs = []
            page_tables = []
            
            for block in page_data.get("blocks", []):
                layout = block.get("layout", {})
                text_anchor = layout.get("textAnchor", {})
                for segment in text_anchor.get("textSegments", []):
                    start = int(segment.get("startIndex", 0))
                    end = int(segment.get("endIndex", 0))
                    page_text += full_text[start:end]
            
            for field in page_data.get("formFields", []):
                field_name = self._extract_text_from_layout(field.get("fieldName", {}), full_text)
                field_value = self._extract_text_from_layout(field.get("fieldValue", {}), full_text)
                confidence = field.get("fieldValue", {}).get("confidence", 0)
                
                kv = ExtractedField(
                    key=field_name.strip(),
                    value=field_value.strip(),
                    confidence=confidence,
                    page=page_num,
                )
                page_kvs.append(kv)
                extracted_fields.append(kv)
            
            for table in page_data.get("tables", []):
                table_data = self._parse_table(table, full_text)
                page_tables.append(table_data)
                tables.append(table_data)
            
            pages.append(DocumentPage(
                page_number=page_num,
                text=page_text,
                tables=page_tables,
                key_value_pairs=page_kvs,
            ))
        
        entities = document.get("entities", [])
        document_type = None
        for entity in entities:
            if entity.get("type") == "document_type":
                document_type = entity.get("mentionText", "")
                break
        
        return {
            "full_text": full_text,
            "pages": pages,
            "extracted_fields": extracted_fields,
            "tables": tables,
            "document_type": document_type,
        }
    
    def _extract_text_from_layout(self, layout: Dict, full_text: str) -> str:
        """Extract text from a layout element."""
        text_anchor = layout.get("layout", {}).get("textAnchor", layout.get("textAnchor", {}))
        segments = text_anchor.get("textSegments", [])
        result = ""
        for segment in segments:
            start = int(segment.get("startIndex", 0))
            end = int(segment.get("endIndex", 0))
            result += full_text[start:end]
        return result
    
    def _parse_table(self, table: Dict, full_text: str) -> Dict:
        """Parse a table from Document AI response."""
        headers = []
        rows = []
        
        header_rows = table.get("headerRows", [])
        for header_row in header_rows:
            header_cells = []
            for cell in header_row.get("cells", []):
                cell_text = self._extract_text_from_layout(cell, full_text)
                header_cells.append(cell_text.strip())
            if header_cells:
                headers = header_cells
        
        body_rows = table.get("bodyRows", [])
        for body_row in body_rows:
            row_cells = []
            for cell in body_row.get("cells", []):
                cell_text = self._extract_text_from_layout(cell, full_text)
                row_cells.append(cell_text.strip())
            if row_cells:
                rows.append(row_cells)
        
        return {
            "title": "",
            "headers": headers,
            "rows": rows,
        }
    
    async def _call_openai_vision_fallback(
        self, 
        image_data: Union[bytes, str], 
        use_healthcare_prompt: bool = False
    ) -> Dict:
        """Use OpenAI Vision GPT-4o for document OCR and extraction."""
        if not self._openai_client:
            raise Exception("OpenAI client not initialized")
        
        if isinstance(image_data, bytes):
            base64_image = base64.b64encode(image_data).decode('utf-8')
        else:
            base64_image = image_data
        
        prompt = OPENAI_HEALTHCARE_DOCUMENT_PROMPT if use_healthcare_prompt else OPENAI_DOCUMENT_OCR_PROMPT
        
        try:
            response = await self._openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a HIPAA-compliant medical document processor. Extract text and structured data with clinical precision. Always respond with valid JSON only."
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                response_format={"type": "json_object"},
                max_tokens=4000,
                temperature=0.1,
            )
            
            result = response.choices[0].message.content
            return json.loads(result)
            
        except Exception as e:
            logger.error(f"[Document AI Fallback] OpenAI Vision call failed: {e}")
            raise
    
    def _parse_openai_response(self, response: Dict) -> Dict:
        """Parse OpenAI Vision response into standardized format."""
        full_text = response.get("full_text", "")
        document_type = response.get("document_type")
        
        extracted_fields = []
        for kv in response.get("key_value_pairs", []):
            extracted_fields.append(ExtractedField(
                key=kv.get("key", ""),
                value=kv.get("value", ""),
                confidence=kv.get("confidence", 0.9),
                page=kv.get("page", 1),
            ))
        
        tables = response.get("tables", [])
        
        pages = [DocumentPage(
            page_number=1,
            text=full_text,
            tables=tables,
            key_value_pairs=extracted_fields,
        )]
        
        return {
            "full_text": full_text,
            "pages": pages,
            "extracted_fields": extracted_fields,
            "tables": tables,
            "document_type": document_type,
        }
    
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
    
    async def process_document(
        self, 
        content: bytes, 
        mime_type: str = "application/pdf",
        use_healthcare_parser: bool = False
    ) -> DocumentProcessingResult:
        """
        Process a document and extract text, tables, and key-value pairs.
        
        Uses GCP Document AI if configured, falls back to OpenAI Vision.
        
        Args:
            content: Document content as bytes
            mime_type: MIME type of the document (application/pdf, image/jpeg, image/png)
            use_healthcare_parser: Use specialized healthcare document parser
        
        Returns:
            DocumentProcessingResult with full text, extracted fields, and tables.
        """
        import time
        start_time = time.time()
        source = "unknown"
        
        self._audit_log(
            "DOCUMENT_PROCESSING", 
            "document-ai-fallback", 
            True, 
            {"mimeType": mime_type, "contentSize": len(content)}
        )
        
        try:
            if self.gcp_configured:
                try:
                    gcp_response = await self._call_gcp_document_ai(content, mime_type, use_healthcare_parser)
                    result = self._parse_gcp_document_response(gcp_response)
                    source = "gcp"
                    logger.info("[Document AI Fallback] Successfully used GCP Document AI")
                except Exception as gcp_error:
                    logger.warning(f"[Document AI Fallback] GCP failed, falling back to OpenAI: {gcp_error}")
                    openai_response = await self._call_openai_vision_fallback(content, use_healthcare_parser)
                    result = self._parse_openai_response(openai_response)
                    source = "openai_fallback"
            else:
                openai_response = await self._call_openai_vision_fallback(content, use_healthcare_parser)
                result = self._parse_openai_response(openai_response)
                source = "openai_fallback"
            
            processing_time_ms = int((time.time() - start_time) * 1000)
            
            return DocumentProcessingResult(
                full_text=result["full_text"],
                pages=result["pages"],
                extracted_fields=result["extracted_fields"],
                tables=result["tables"],
                document_type=result.get("document_type"),
                source=source,
                processingTimeMs=processing_time_ms,
            )
            
        except Exception as e:
            logger.error(f"[Document AI Fallback] All processing methods failed: {e}")
            processing_time_ms = int((time.time() - start_time) * 1000)
            return DocumentProcessingResult(
                full_text="",
                pages=[],
                extracted_fields=[],
                tables=[],
                document_type=None,
                source="error",
                processingTimeMs=processing_time_ms,
            )
    
    async def extract_text(self, content: bytes, mime_type: str = "image/jpeg") -> str:
        """
        Simple OCR - extract just the text from a document/image.
        
        Args:
            content: Image/document content as bytes
            mime_type: MIME type
        
        Returns:
            Extracted text string
        """
        result = await self.process_document(content, mime_type, use_healthcare_parser=False)
        return result.full_text
    
    async def extract_key_values(
        self, 
        content: bytes, 
        mime_type: str = "image/jpeg"
    ) -> List[ExtractedField]:
        """
        Extract key-value pairs from a document/image.
        
        Args:
            content: Image/document content as bytes
            mime_type: MIME type
        
        Returns:
            List of ExtractedField objects
        """
        result = await self.process_document(content, mime_type, use_healthcare_parser=False)
        return result.extracted_fields
    
    async def process_healthcare_document(
        self, 
        content: bytes, 
        mime_type: str = "application/pdf"
    ) -> DocumentProcessingResult:
        """
        Process a healthcare document with specialized clinical extraction.
        
        Uses healthcare-specific prompts and parsers.
        
        Args:
            content: Document content as bytes
            mime_type: MIME type
        
        Returns:
            DocumentProcessingResult with clinical data extraction
        """
        return await self.process_document(content, mime_type, use_healthcare_parser=True)
    
    def get_service_status(self) -> Dict:
        """Get current service configuration status."""
        return {
            "gcpConfigured": self.gcp_configured,
            "openaiAvailable": self._openai_client is not None,
            "primarySource": "gcp" if self.gcp_configured else "openai",
            "fallbackAvailable": self._openai_client is not None,
            "processorId": GCP_CONFIG.DOCUMENT_AI.PROCESSOR_ID if self.gcp_configured else None,
            "healthcareProcessorId": GCP_CONFIG.DOCUMENT_AI.HEALTHCARE_PROCESSOR_ID if self.gcp_configured else None,
        }


document_ai_fallback_service = DocumentAIFallbackService()
