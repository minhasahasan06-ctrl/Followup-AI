"""
Lysa Clinical Documentation Service
Generates structured clinical documents: differential diagnosis, A&P, H&P, progress notes.
All outputs are drafts for clinician review - NOT autonomous medical advice.
"""

import os
import json
import hashlib
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
from sqlalchemy.orm import Session

from openai import AsyncOpenAI
from app.services.ehr_service import get_ehr_service
from app.models.lysa_drafts import LysaDraft, LysaDraftAuditLog

logger = logging.getLogger(__name__)


DIFFERENTIAL_SCHEMA = {
    "case_discussion": {
        "key_positives": [],
        "key_negatives": [],
        "contextual_factors": []
    },
    "diagnostic_next_steps": {
        "labs": [],
        "imaging": [],
        "monitoring": []
    },
    "differential_diagnosis": {
        "most_likely": [],
        "expanded_differential": [],
        "cant_miss": []
    },
    "assessment_and_plan": {
        "clinical_impression": "",
        "problems": []
    },
    "history_and_physical": {
        "chief_complaint": "",
        "history_of_present_illness": "",
        "past_medical_history": [],
        "past_surgical_history": [],
        "medications": [],
        "allergies": [],
        "family_history": "",
        "social_history": "",
        "review_of_systems": {},
        "vital_signs": {},
        "physical_examination": {},
        "laboratory_data": [],
        "imaging": [],
        "chronic_problems": []
    },
    "references": []
}


LYSA_SYSTEM_PROMPT = """You are Assistant Lysa, an AI clinical documentation assistant. 
You help clinicians by generating DRAFT clinical documentation for their review.

CRITICAL RULES:
1. You generate DRAFTS only - clinicians must review and approve before use
2. Never provide autonomous medical advice
3. Include evidence-based references for recommendations
4. Use clinical terminology appropriate for medical documentation
5. Flag uncertainty clearly
6. Include provenance for each claim (which data source supported it)

OUTPUT FORMAT:
You must respond with ONLY valid JSON matching the specified schema.
Do not include markdown, explanations, or any text outside the JSON.
"""


def hash_content(content: str) -> str:
    """Generate SHA256 hash of content."""
    return hashlib.sha256(content.encode()).hexdigest()


class LysaDocumentationService:
    """
    Service for generating clinical documentation drafts.
    All outputs require clinician review before chart insertion.
    """
    
    def __init__(self, db: Session):
        self.db = db
        self.ehr_service = get_ehr_service(db)
        api_key = os.getenv("OPENAI_API_KEY")
        self.openai_client = AsyncOpenAI(api_key=api_key) if api_key else None
    
    async def generate_differential(
        self,
        patient_id: str,
        doctor_id: str,
        question: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a structured differential diagnosis draft.
        
        Args:
            patient_id: Patient ID
            doctor_id: Doctor requesting the draft
            question: Optional clinical question/context
            
        Returns:
            Created LysaDraft record with structured content
        """
        ehr_summary = await self.ehr_service.get_full_ehr_summary(patient_id, doctor_id)
        
        prompt = self._build_differential_prompt(ehr_summary, question)
        
        request_hash = hash_content(prompt)
        
        try:
            response = await self._call_openai(prompt)
            response_hash = hash_content(response)
            
            content_json = self._parse_and_validate_json(response)
            
        except Exception as e:
            logger.error(f"Error generating differential: {e}")
            content_json = self._generate_fallback_content(ehr_summary)
            response = json.dumps(content_json)
            response_hash = hash_content(response)
        
        draft = LysaDraft(
            patient_id=patient_id,
            doctor_id=doctor_id,
            draft_type="differential",
            status="draft",
            question=question,
            content_json=content_json,
            raw_output=response,
            provenance=ehr_summary.get("provenance", []),
            ehr_sources_used=self._extract_source_ids(ehr_summary)
        )
        
        self.db.add(draft)
        self.db.commit()
        self.db.refresh(draft)
        
        audit_log = LysaDraftAuditLog(
            draft_id=draft.id,
            patient_id=patient_id,
            doctor_id=doctor_id,
            action="generate_differential",
            ehr_resources_accessed=ehr_summary.get("provenance", []),
            request_hash=request_hash,
            response_hash=response_hash,
            model_used="gpt-4o",
            ip_address=ip_address,
            user_agent=user_agent
        )
        self.db.add(audit_log)
        self.db.commit()
        
        return {
            "id": draft.id,
            "patient_id": patient_id,
            "doctor_id": doctor_id,
            "draft_type": "differential",
            "status": draft.status,
            "content": content_json,
            "provenance": draft.provenance,
            "created_at": draft.created_at.isoformat() if draft.created_at else None
        }
    
    async def get_drafts(
        self,
        patient_id: str,
        doctor_id: str,
        draft_type: Optional[str] = None,
        status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get all drafts for a patient."""
        query = self.db.query(LysaDraft).filter(
            LysaDraft.patient_id == patient_id,
            LysaDraft.doctor_id == doctor_id
        )
        
        if draft_type:
            query = query.filter(LysaDraft.draft_type == draft_type)
        if status:
            query = query.filter(LysaDraft.status == status)
        
        drafts = query.order_by(LysaDraft.created_at.desc()).all()
        
        return [
            {
                "id": d.id,
                "draft_type": d.draft_type,
                "status": d.status,
                "question": d.question,
                "content": d.content_json,
                "provenance": d.provenance,
                "revision_count": d.revision_count,
                "created_at": d.created_at.isoformat() if d.created_at else None,
                "updated_at": d.updated_at.isoformat() if d.updated_at else None
            }
            for d in drafts
        ]
    
    async def revise_draft(
        self,
        draft_id: str,
        doctor_id: str,
        instruction: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Revise a draft based on clinician instruction.
        """
        draft = self.db.query(LysaDraft).filter(
            LysaDraft.id == draft_id,
            LysaDraft.doctor_id == doctor_id
        ).first()
        
        if not draft:
            raise ValueError("Draft not found or access denied")
        
        if draft.status in ["approved", "inserted_to_chart"]:
            raise ValueError("Cannot revise approved or inserted drafts")
        
        revision_history = draft.revision_history or []
        revision_history.append({
            "revision": draft.revision_count,
            "previous_content": draft.content_json,
            "instruction": instruction,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        prompt = self._build_revision_prompt(draft.content_json, instruction)
        request_hash = hash_content(prompt)
        
        try:
            response = await self._call_openai(prompt)
            response_hash = hash_content(response)
            content_json = self._parse_and_validate_json(response)
        except Exception as e:
            logger.error(f"Error revising draft: {e}")
            raise ValueError(f"Failed to revise draft: {e}")
        
        draft.content_json = content_json
        draft.raw_output = response
        draft.revision_count += 1
        draft.revision_history = revision_history
        draft.status = "revised"
        draft.updated_at = datetime.utcnow()
        
        audit_log = LysaDraftAuditLog(
            draft_id=draft.id,
            patient_id=draft.patient_id,
            doctor_id=doctor_id,
            action="revise_draft",
            request_hash=request_hash,
            response_hash=response_hash,
            model_used="gpt-4o",
            ip_address=ip_address,
            user_agent=user_agent
        )
        self.db.add(audit_log)
        self.db.commit()
        
        return {
            "id": draft.id,
            "status": draft.status,
            "content": content_json,
            "revision_count": draft.revision_count,
            "updated_at": draft.updated_at.isoformat() if draft.updated_at else None
        }
    
    async def approve_draft(
        self,
        draft_id: str,
        doctor_id: str,
        insert_to_chart: bool = False,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Approve a draft. Optionally insert into chart (EHR write).
        
        IMPORTANT: This requires explicit clinician confirmation.
        """
        draft = self.db.query(LysaDraft).filter(
            LysaDraft.id == draft_id,
            LysaDraft.doctor_id == doctor_id
        ).first()
        
        if not draft:
            raise ValueError("Draft not found or access denied")
        
        if draft.status == "inserted_to_chart":
            raise ValueError("Draft already inserted to chart")
        
        draft.status = "approved"
        draft.approved_at = datetime.utcnow()
        draft.approved_by = doctor_id
        
        ehr_note_id = None
        if insert_to_chart:
            ehr_note_id = await self._insert_to_chart(draft)
            draft.status = "inserted_to_chart"
            draft.inserted_to_chart_at = datetime.utcnow()
            draft.ehr_note_id = ehr_note_id
        
        audit_log = LysaDraftAuditLog(
            draft_id=draft.id,
            patient_id=draft.patient_id,
            doctor_id=doctor_id,
            action="approve_draft" if not insert_to_chart else "approve_and_insert",
            ip_address=ip_address,
            user_agent=user_agent
        )
        self.db.add(audit_log)
        self.db.commit()
        
        return {
            "id": draft.id,
            "status": draft.status,
            "approved_at": draft.approved_at.isoformat() if draft.approved_at else None,
            "ehr_note_id": ehr_note_id
        }
    
    def _build_differential_prompt(
        self,
        ehr_summary: Dict[str, Any],
        question: Optional[str] = None
    ) -> str:
        """Build the prompt for differential generation."""
        
        prompt = f"""Generate a comprehensive clinical documentation draft based on the following patient data.

PATIENT EHR SUMMARY:
{json.dumps(ehr_summary, indent=2, default=str)}

{"CLINICAL QUESTION: " + question if question else ""}

Generate a complete differential diagnosis document with:
1. Case Discussion (key positives, negatives, contextual factors)
2. Diagnostic Next Steps (labs, imaging, monitoring recommendations)
3. Differential Diagnosis:
   - Most Likely: Top 3-5 diagnoses with supporting and opposing evidence
   - Expanded Differential: 3-5 alternative diagnoses
   - Can't Miss: Critical diagnoses that must be excluded
4. Assessment & Plan:
   - Clinical impression
   - Problem-oriented plan with Dx (diagnostics) and Tx (treatments)
5. History & Physical Note (structured)
6. References (numbered citations for recommendations)

RESPOND WITH ONLY VALID JSON matching this schema:
{json.dumps(DIFFERENTIAL_SCHEMA, indent=2)}

Each diagnosis in the differential should include:
- "diagnosis": name
- "supporting_evidence": list of supporting findings
- "opposing_evidence": list of findings that argue against
- "probability": "high", "moderate", or "low"
- "source_refs": list of reference numbers

Each problem in assessment_and_plan.problems should include:
- "problem": name
- "rationale": clinical reasoning
- "diagnostics": recommended tests
- "treatment": recommended treatment
- "follow_up": follow-up actions
- "citations": reference numbers

IMPORTANT: Output ONLY the JSON object, no markdown or additional text."""
        
        return prompt
    
    def _build_revision_prompt(
        self,
        current_content: Dict[str, Any],
        instruction: str
    ) -> str:
        """Build prompt for draft revision."""
        return f"""Revise the following clinical documentation draft based on the clinician's instruction.

CURRENT DRAFT:
{json.dumps(current_content, indent=2, default=str)}

REVISION INSTRUCTION:
{instruction}

Apply the requested changes while maintaining the overall structure.
RESPOND WITH ONLY the updated JSON object, no explanations."""
    
    async def _call_openai(self, prompt: str) -> str:
        """Call OpenAI API for generation."""
        if not self.openai_client:
            raise ValueError("OpenAI client not configured")
        
        response = self.openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": LYSA_SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=4000,
            response_format={"type": "json_object"}
        )
        
        return response.choices[0].message.content
    
    def _parse_and_validate_json(self, response: str) -> Dict[str, Any]:
        """Parse and validate JSON response."""
        try:
            content = json.loads(response)
            return content
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON response: {e}")
            raise ValueError(f"Invalid JSON response from AI: {e}")
    
    def _generate_fallback_content(self, ehr_summary: Dict[str, Any]) -> Dict[str, Any]:
        """Generate fallback content when AI fails."""
        return {
            "case_discussion": {
                "key_positives": [p.get("name", "Unknown") for p in ehr_summary.get("problems", [])[:5]],
                "key_negatives": [],
                "contextual_factors": ["AI generation failed - manual review required"]
            },
            "diagnostic_next_steps": {
                "labs": ["Complete metabolic panel", "CBC with differential"],
                "imaging": [],
                "monitoring": ["Vital signs monitoring"]
            },
            "differential_diagnosis": {
                "most_likely": [],
                "expanded_differential": [],
                "cant_miss": []
            },
            "assessment_and_plan": {
                "clinical_impression": "Requires clinician review - AI generation incomplete",
                "problems": []
            },
            "history_and_physical": {
                "chief_complaint": "See patient record",
                "medications": [m.get("name", "Unknown") for m in ehr_summary.get("medications", [])],
                "allergies": [a.get("allergen", "Unknown") for a in ehr_summary.get("allergies", [])]
            },
            "references": [],
            "_fallback": True,
            "_reason": "AI generation failed, manual completion required"
        }
    
    def _extract_source_ids(self, ehr_summary: Dict[str, Any]) -> Dict[str, List[str]]:
        """Extract source IDs for provenance tracking."""
        return {
            "problems_count": len(ehr_summary.get("problems", [])),
            "complaints_count": len(ehr_summary.get("complaints", [])),
            "medications_count": len(ehr_summary.get("medications", [])),
            "allergies_count": len(ehr_summary.get("allergies", [])),
            "generated_at": ehr_summary.get("generated_at")
        }
    
    async def _insert_to_chart(self, draft: LysaDraft) -> str:
        """
        Insert approved draft to EHR chart.
        Returns the created note ID.
        
        NOTE: This is a placeholder - actual EHR integration would go here.
        """
        note_id = f"NOTE_{draft.id}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        logger.info(f"Inserting draft {draft.id} to chart as note {note_id}")
        return note_id


def get_lysa_documentation_service(db: Session) -> LysaDocumentationService:
    """Factory function."""
    return LysaDocumentationService(db)
