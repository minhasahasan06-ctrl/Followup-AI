"""
Prescription Draft Tool Microservice
AI-assisted prescription drafting for Assistant Lysa (REQUIRES HUMAN APPROVAL)
"""

import logging
import json
from typing import Dict, Any, Optional, List
from datetime import datetime
import uuid

from app.services.agent_tools.base import BaseTool, ToolExecutionContext
from app.models.agent_models import ToolCallResult, ToolStatus

logger = logging.getLogger(__name__)


class PrescriptionDraftTool(BaseTool):
    """
    Prescription drafting tool for Assistant Lysa.
    CRITICAL: All prescriptions REQUIRE doctor approval before being finalized.
    Supports drug-drug interaction checking and auto-dosage recommendations.
    """
    
    def __init__(self):
        super().__init__()
        self.name = "prescription_draft"
        self.display_name = "Prescription Draft"
        self.description = """
Create prescription drafts for doctor review and approval.
Actions: create_draft, check_interactions, suggest_dosage, get_patient_medications,
get_draft, update_draft, submit_for_approval
IMPORTANT: All prescriptions require doctor approval before finalization.
"""
        self.tool_type = "prescription_draft"
        self.requires_approval = True
        self.approval_role = "doctor"
        self.allowed_roles = ["doctor"]
        self.required_permissions = ["prescription:draft", "prescription:view"]
        self.version = 1
    
    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "create_draft",
                        "check_interactions",
                        "suggest_dosage",
                        "get_patient_medications",
                        "get_draft",
                        "update_draft",
                        "submit_for_approval"
                    ],
                    "description": "The prescription action to perform"
                },
                "patient_id": {
                    "type": "string",
                    "description": "Patient ID for the prescription"
                },
                "draft_id": {
                    "type": "string",
                    "description": "Prescription draft ID for updates"
                },
                "medication_name": {
                    "type": "string",
                    "description": "Name of the medication"
                },
                "rxnorm_cui": {
                    "type": "string",
                    "description": "RxNorm Concept Unique Identifier"
                },
                "dosage": {
                    "type": "string",
                    "description": "Dosage amount and unit (e.g., '500mg')"
                },
                "frequency": {
                    "type": "string",
                    "description": "How often to take (e.g., 'twice daily')"
                },
                "duration_days": {
                    "type": "integer",
                    "description": "Duration of prescription in days"
                },
                "refills": {
                    "type": "integer",
                    "description": "Number of refills allowed"
                },
                "instructions": {
                    "type": "string",
                    "description": "Special instructions for the patient"
                },
                "indication": {
                    "type": "string",
                    "description": "Medical indication for the prescription"
                },
                "notes": {
                    "type": "string",
                    "description": "Additional notes for the pharmacist"
                },
                "medications_to_check": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of medication names to check for interactions"
                }
            },
            "required": ["action"]
        }
    
    async def execute(
        self,
        parameters: Dict[str, Any],
        context: ToolExecutionContext
    ) -> ToolCallResult:
        """Execute prescription action"""
        action = parameters.get("action")
        
        try:
            if action == "create_draft":
                return await self._create_draft(parameters, context)
            elif action == "check_interactions":
                return await self._check_interactions(parameters, context)
            elif action == "suggest_dosage":
                return await self._suggest_dosage(parameters, context)
            elif action == "get_patient_medications":
                return await self._get_patient_medications(parameters, context)
            elif action == "get_draft":
                return await self._get_draft(parameters, context)
            elif action == "update_draft":
                return await self._update_draft(parameters, context)
            elif action == "submit_for_approval":
                return await self._submit_for_approval(parameters, context)
            else:
                return ToolCallResult(
                    tool_call_id=context.message_id,
                    tool_name=self.name,
                    status=ToolStatus.FAILED,
                    error=f"Unknown action: {action}"
                )
        except Exception as e:
            logger.error(f"Prescription tool error: {e}")
            return ToolCallResult(
                tool_call_id=context.message_id,
                tool_name=self.name,
                status=ToolStatus.FAILED,
                error=str(e)
            )
    
    async def _create_draft(
        self,
        parameters: Dict[str, Any],
        context: ToolExecutionContext
    ) -> ToolCallResult:
        """Create a new prescription draft"""
        from app.database import SessionLocal
        from sqlalchemy import text
        
        patient_id = parameters.get("patient_id") or context.patient_id
        if not patient_id:
            return ToolCallResult(
                tool_call_id=context.message_id,
                tool_name=self.name,
                status=ToolStatus.FAILED,
                error="Patient ID required for prescription"
            )
        
        medication_name = parameters.get("medication_name")
        if not medication_name:
            return ToolCallResult(
                tool_call_id=context.message_id,
                tool_name=self.name,
                status=ToolStatus.FAILED,
                error="Medication name required"
            )
        
        draft_id = str(uuid.uuid4())
        doctor_id = context.doctor_id or context.user_id
        
        prescription_data = {
            "medication_name": medication_name,
            "rxnorm_cui": parameters.get("rxnorm_cui"),
            "dosage": parameters.get("dosage"),
            "frequency": parameters.get("frequency"),
            "duration_days": parameters.get("duration_days"),
            "refills": parameters.get("refills", 0),
            "instructions": parameters.get("instructions"),
            "indication": parameters.get("indication"),
            "notes": parameters.get("notes")
        }
        
        db = SessionLocal()
        try:
            db.execute(
                text("""
                    INSERT INTO prescription_drafts (
                        id, patient_id, doctor_id, medication_name,
                        dosage, frequency, duration_days, refills,
                        instructions, indication, notes, rxnorm_cui,
                        status, created_by_agent, created_at
                    ) VALUES (
                        :id, :patient_id, :doctor_id, :medication_name,
                        :dosage, :frequency, :duration_days, :refills,
                        :instructions, :indication, :notes, :rxnorm_cui,
                        'draft', :agent_id, NOW()
                    )
                """),
                {
                    "id": draft_id,
                    "patient_id": patient_id,
                    "doctor_id": doctor_id,
                    "medication_name": medication_name,
                    "dosage": parameters.get("dosage"),
                    "frequency": parameters.get("frequency"),
                    "duration_days": parameters.get("duration_days"),
                    "refills": parameters.get("refills", 0),
                    "instructions": parameters.get("instructions"),
                    "indication": parameters.get("indication"),
                    "notes": parameters.get("notes"),
                    "rxnorm_cui": parameters.get("rxnorm_cui"),
                    "agent_id": context.agent_id
                }
            )
            db.commit()
            
            return ToolCallResult(
                tool_call_id=context.message_id,
                tool_name=self.name,
                status=ToolStatus.PENDING_APPROVAL,
                result={
                    "draft_id": draft_id,
                    "patient_id": patient_id,
                    "doctor_id": doctor_id,
                    "prescription": prescription_data,
                    "status": "pending_approval",
                    "requires_approval": True,
                    "approval_required_from": "doctor",
                    "message": "Prescription draft created. Awaiting doctor approval before finalization."
                }
            )
        finally:
            db.close()
    
    async def _check_interactions(
        self,
        parameters: Dict[str, Any],
        context: ToolExecutionContext
    ) -> ToolCallResult:
        """Check for drug-drug interactions"""
        from app.database import SessionLocal
        from sqlalchemy import text
        
        patient_id = parameters.get("patient_id") or context.patient_id
        medications_to_check = parameters.get("medications_to_check", [])
        new_medication = parameters.get("medication_name")
        
        if new_medication:
            medications_to_check.append(new_medication)
        
        if not medications_to_check:
            return ToolCallResult(
                tool_call_id=context.message_id,
                tool_name=self.name,
                status=ToolStatus.FAILED,
                error="At least one medication required for interaction check"
            )
        
        db = SessionLocal()
        try:
            current_medications = []
            if patient_id:
                result = db.execute(
                    text("""
                        SELECT medication_name, rxnorm_cui
                        FROM patient_medications
                        WHERE patient_id = :patient_id
                        AND status = 'active'
                    """),
                    {"patient_id": patient_id}
                )
                for row in result.fetchall():
                    current_medications.append({
                        "name": row[0],
                        "rxnorm_cui": row[1]
                    })
            
            all_meds = [m.get("name") or m for m in current_medications] + medications_to_check
            
            interactions_found = []
            warnings = []
            
            common_interactions = {
                ("warfarin", "aspirin"): {
                    "severity": "high",
                    "description": "Increased risk of bleeding"
                },
                ("metformin", "alcohol"): {
                    "severity": "moderate", 
                    "description": "Risk of lactic acidosis"
                },
                ("lisinopril", "potassium"): {
                    "severity": "moderate",
                    "description": "Risk of hyperkalemia"
                },
                ("simvastatin", "grapefruit"): {
                    "severity": "moderate",
                    "description": "Increased drug levels and side effects"
                }
            }
            
            all_meds_lower = [m.lower() if isinstance(m, str) else m for m in all_meds]
            for (drug1, drug2), info in common_interactions.items():
                if drug1 in all_meds_lower and drug2 in all_meds_lower:
                    interactions_found.append({
                        "drugs": [drug1, drug2],
                        "severity": info["severity"],
                        "description": info["description"]
                    })
            
            return ToolCallResult(
                tool_call_id=context.message_id,
                tool_name=self.name,
                status=ToolStatus.COMPLETED,
                result={
                    "medications_checked": medications_to_check,
                    "current_medications": [m.get("name") if isinstance(m, dict) else m for m in current_medications],
                    "interactions": interactions_found,
                    "warnings": warnings,
                    "has_severe_interactions": any(i["severity"] == "high" for i in interactions_found),
                    "message": f"Found {len(interactions_found)} potential interactions" if interactions_found else "No known interactions found"
                }
            )
        finally:
            db.close()
    
    async def _suggest_dosage(
        self,
        parameters: Dict[str, Any],
        context: ToolExecutionContext
    ) -> ToolCallResult:
        """Suggest dosage based on patient parameters"""
        patient_id = parameters.get("patient_id") or context.patient_id
        medication_name = parameters.get("medication_name")
        indication = parameters.get("indication")
        
        if not medication_name:
            return ToolCallResult(
                tool_call_id=context.message_id,
                tool_name=self.name,
                status=ToolStatus.FAILED,
                error="Medication name required for dosage suggestion"
            )
        
        common_dosages = {
            "amoxicillin": {
                "adult": {"dosage": "500mg", "frequency": "three times daily", "duration_days": 7},
                "pediatric": {"dosage": "25mg/kg", "frequency": "twice daily", "duration_days": 7}
            },
            "metformin": {
                "adult": {"dosage": "500mg", "frequency": "twice daily", "duration_days": 30},
            },
            "lisinopril": {
                "adult": {"dosage": "10mg", "frequency": "once daily", "duration_days": 30}
            },
            "omeprazole": {
                "adult": {"dosage": "20mg", "frequency": "once daily", "duration_days": 14}
            }
        }
        
        med_lower = medication_name.lower()
        suggestion = None
        for drug, dosages in common_dosages.items():
            if drug in med_lower:
                suggestion = dosages.get("adult", {})
                break
        
        if not suggestion:
            suggestion = {
                "dosage": "Consult reference",
                "frequency": "As directed",
                "duration_days": None,
                "note": "Standard dosage not available. Please consult drug reference."
            }
        
        return ToolCallResult(
            tool_call_id=context.message_id,
            tool_name=self.name,
            status=ToolStatus.COMPLETED,
            result={
                "medication_name": medication_name,
                "indication": indication,
                "suggested_dosage": suggestion,
                "disclaimer": "This is a suggestion only. Doctor must verify appropriate dosage for patient.",
                "requires_verification": True
            }
        )
    
    async def _get_patient_medications(
        self,
        parameters: Dict[str, Any],
        context: ToolExecutionContext
    ) -> ToolCallResult:
        """Get patient's current medications"""
        from app.database import SessionLocal
        from sqlalchemy import text
        
        patient_id = parameters.get("patient_id") or context.patient_id
        if not patient_id:
            return ToolCallResult(
                tool_call_id=context.message_id,
                tool_name=self.name,
                status=ToolStatus.FAILED,
                error="Patient ID required"
            )
        
        db = SessionLocal()
        try:
            result = db.execute(
                text("""
                    SELECT id, medication_name, dosage, frequency, 
                           start_date, end_date, status, prescribing_doctor_id,
                           rxnorm_cui, instructions
                    FROM patient_medications
                    WHERE patient_id = :patient_id
                    ORDER BY status, start_date DESC
                """),
                {"patient_id": patient_id}
            )
            
            medications = []
            for row in result.fetchall():
                medications.append({
                    "id": row[0],
                    "medication_name": row[1],
                    "dosage": row[2],
                    "frequency": row[3],
                    "start_date": row[4].isoformat() if row[4] else None,
                    "end_date": row[5].isoformat() if row[5] else None,
                    "status": row[6],
                    "prescribing_doctor_id": row[7],
                    "rxnorm_cui": row[8],
                    "instructions": row[9]
                })
            
            active_count = len([m for m in medications if m["status"] == "active"])
            
            return ToolCallResult(
                tool_call_id=context.message_id,
                tool_name=self.name,
                status=ToolStatus.COMPLETED,
                result={
                    "patient_id": patient_id,
                    "medications": medications,
                    "total_count": len(medications),
                    "active_count": active_count,
                    "message": f"Patient has {active_count} active medications"
                }
            )
        finally:
            db.close()
    
    async def _get_draft(
        self,
        parameters: Dict[str, Any],
        context: ToolExecutionContext
    ) -> ToolCallResult:
        """Get a prescription draft by ID"""
        from app.database import SessionLocal
        from sqlalchemy import text
        
        draft_id = parameters.get("draft_id")
        if not draft_id:
            return ToolCallResult(
                tool_call_id=context.message_id,
                tool_name=self.name,
                status=ToolStatus.FAILED,
                error="Draft ID required"
            )
        
        db = SessionLocal()
        try:
            result = db.execute(
                text("""
                    SELECT id, patient_id, doctor_id, medication_name,
                           dosage, frequency, duration_days, refills,
                           instructions, indication, notes, status,
                           created_at, approved_at, approved_by
                    FROM prescription_drafts
                    WHERE id = :draft_id
                """),
                {"draft_id": draft_id}
            )
            
            row = result.fetchone()
            if not row:
                return ToolCallResult(
                    tool_call_id=context.message_id,
                    tool_name=self.name,
                    status=ToolStatus.FAILED,
                    error="Draft not found"
                )
            
            return ToolCallResult(
                tool_call_id=context.message_id,
                tool_name=self.name,
                status=ToolStatus.COMPLETED,
                result={
                    "draft_id": row[0],
                    "patient_id": row[1],
                    "doctor_id": row[2],
                    "medication_name": row[3],
                    "dosage": row[4],
                    "frequency": row[5],
                    "duration_days": row[6],
                    "refills": row[7],
                    "instructions": row[8],
                    "indication": row[9],
                    "notes": row[10],
                    "status": row[11],
                    "created_at": row[12].isoformat() if row[12] else None,
                    "approved_at": row[13].isoformat() if row[13] else None,
                    "approved_by": row[14]
                }
            )
        finally:
            db.close()
    
    async def _update_draft(
        self,
        parameters: Dict[str, Any],
        context: ToolExecutionContext
    ) -> ToolCallResult:
        """Update an existing prescription draft"""
        from app.database import SessionLocal
        from sqlalchemy import text
        
        draft_id = parameters.get("draft_id")
        if not draft_id:
            return ToolCallResult(
                tool_call_id=context.message_id,
                tool_name=self.name,
                status=ToolStatus.FAILED,
                error="Draft ID required"
            )
        
        update_fields = []
        update_values = {"draft_id": draft_id}
        
        for field in ["medication_name", "dosage", "frequency", "duration_days", 
                      "refills", "instructions", "indication", "notes"]:
            if field in parameters and parameters[field] is not None:
                update_fields.append(f"{field} = :{field}")
                update_values[field] = parameters[field]
        
        if not update_fields:
            return ToolCallResult(
                tool_call_id=context.message_id,
                tool_name=self.name,
                status=ToolStatus.FAILED,
                error="No fields to update"
            )
        
        db = SessionLocal()
        try:
            update_sql = f"""
                UPDATE prescription_drafts
                SET {', '.join(update_fields)}, updated_at = NOW()
                WHERE id = :draft_id AND status = 'draft'
                RETURNING id
            """
            
            result = db.execute(text(update_sql), update_values)
            row = result.fetchone()
            
            if not row:
                return ToolCallResult(
                    tool_call_id=context.message_id,
                    tool_name=self.name,
                    status=ToolStatus.FAILED,
                    error="Draft not found or already approved"
                )
            
            db.commit()
            
            return ToolCallResult(
                tool_call_id=context.message_id,
                tool_name=self.name,
                status=ToolStatus.COMPLETED,
                result={
                    "draft_id": draft_id,
                    "updated_fields": list(update_values.keys()),
                    "message": "Draft updated successfully"
                }
            )
        finally:
            db.close()
    
    async def _submit_for_approval(
        self,
        parameters: Dict[str, Any],
        context: ToolExecutionContext
    ) -> ToolCallResult:
        """Submit a draft for doctor approval"""
        from app.database import SessionLocal
        from sqlalchemy import text
        
        draft_id = parameters.get("draft_id")
        if not draft_id:
            return ToolCallResult(
                tool_call_id=context.message_id,
                tool_name=self.name,
                status=ToolStatus.FAILED,
                error="Draft ID required"
            )
        
        db = SessionLocal()
        try:
            result = db.execute(
                text("""
                    UPDATE prescription_drafts
                    SET status = 'pending_approval', updated_at = NOW()
                    WHERE id = :draft_id AND status = 'draft'
                    RETURNING id, patient_id, doctor_id, medication_name
                """),
                {"draft_id": draft_id}
            )
            
            row = result.fetchone()
            if not row:
                return ToolCallResult(
                    tool_call_id=context.message_id,
                    tool_name=self.name,
                    status=ToolStatus.FAILED,
                    error="Draft not found or already submitted"
                )
            
            db.commit()
            
            return ToolCallResult(
                tool_call_id=context.message_id,
                tool_name=self.name,
                status=ToolStatus.PENDING_APPROVAL,
                result={
                    "draft_id": row[0],
                    "patient_id": row[1],
                    "doctor_id": row[2],
                    "medication_name": row[3],
                    "status": "pending_approval",
                    "requires_approval": True,
                    "message": "Prescription draft submitted for doctor approval"
                }
            )
        finally:
            db.close()
