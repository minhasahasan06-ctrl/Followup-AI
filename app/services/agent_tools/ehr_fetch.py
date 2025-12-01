"""
EHR Fetch Tool Microservice
Electronic Health Records retrieval for agents
"""

import logging
import json
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import uuid

from app.services.agent_tools.base import BaseTool, ToolExecutionContext
from app.models.agent_models import ToolCallResult, ToolStatus

logger = logging.getLogger(__name__)


class EHRFetchTool(BaseTool):
    """
    EHR (Electronic Health Records) fetch tool for Assistant Lysa.
    Retrieves patient health records, medical history, and clinical data.
    All access is logged for HIPAA compliance.
    """
    
    def __init__(self):
        super().__init__()
        self.name = "ehr_fetch"
        self.display_name = "EHR Records Fetch"
        self.description = """
Retrieve electronic health records and patient clinical data.
Actions: get_patient_summary, get_medical_history, get_vitals, get_diagnoses,
get_allergies, get_immunizations, get_procedures, get_care_plan
All access is logged for HIPAA audit compliance.
"""
        self.tool_type = "ehr_fetch"
        self.requires_approval = False
        self.allowed_roles = ["doctor"]
        self.required_permissions = ["ehr:read", "patient:view"]
        self.version = 1
    
    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "get_patient_summary",
                        "get_medical_history",
                        "get_vitals",
                        "get_diagnoses",
                        "get_allergies",
                        "get_immunizations",
                        "get_procedures",
                        "get_care_plan"
                    ],
                    "description": "The EHR action to perform"
                },
                "patient_id": {
                    "type": "string",
                    "description": "Patient ID to fetch records for"
                },
                "date_from": {
                    "type": "string",
                    "description": "Start date for records (ISO format)"
                },
                "date_to": {
                    "type": "string",
                    "description": "End date for records (ISO format)"
                },
                "include_inactive": {
                    "type": "boolean",
                    "description": "Include inactive/resolved conditions"
                },
                "category": {
                    "type": "string",
                    "description": "Filter by category (varies by action)"
                }
            },
            "required": ["action"]
        }
    
    async def execute(
        self,
        parameters: Dict[str, Any],
        context: ToolExecutionContext
    ) -> ToolCallResult:
        """Execute EHR fetch action"""
        action = parameters.get("action")
        
        await self._log_phi_access(action, parameters, context)
        
        try:
            if action == "get_patient_summary":
                return await self._get_patient_summary(parameters, context)
            elif action == "get_medical_history":
                return await self._get_medical_history(parameters, context)
            elif action == "get_vitals":
                return await self._get_vitals(parameters, context)
            elif action == "get_diagnoses":
                return await self._get_diagnoses(parameters, context)
            elif action == "get_allergies":
                return await self._get_allergies(parameters, context)
            elif action == "get_immunizations":
                return await self._get_immunizations(parameters, context)
            elif action == "get_procedures":
                return await self._get_procedures(parameters, context)
            elif action == "get_care_plan":
                return await self._get_care_plan(parameters, context)
            else:
                return ToolCallResult(
                    tool_call_id=context.message_id,
                    tool_name=self.name,
                    status=ToolStatus.FAILED,
                    error=f"Unknown action: {action}"
                )
        except Exception as e:
            logger.error(f"EHR fetch error: {e}")
            return ToolCallResult(
                tool_call_id=context.message_id,
                tool_name=self.name,
                status=ToolStatus.FAILED,
                error=str(e)
            )
    
    async def _log_phi_access(
        self,
        action: str,
        parameters: Dict[str, Any],
        context: ToolExecutionContext
    ):
        """Log PHI access for HIPAA compliance using AuditLogger"""
        from app.services.audit_logger import AuditLogger, AuditEvent
        
        patient_id = parameters.get("patient_id") or context.patient_id
        
        AuditLogger.log_event(
            event_type=AuditEvent.PHI_ACCESSED,
            user_id=context.user_id,
            resource_type="ehr",
            resource_id=patient_id or "unknown",
            action=f"ehr_fetch:{action}",
            status="success",
            metadata={
                "action": action,
                "patient_id": patient_id,
                "agent_id": context.agent_id,
                "user_role": context.user_role,
                "conversation_id": context.conversation_id,
                "phi_accessed": True,
                "phi_categories": ["health_records", "medical_history"]
            }
        )
    
    async def _get_patient_summary(
        self,
        parameters: Dict[str, Any],
        context: ToolExecutionContext
    ) -> ToolCallResult:
        """Get comprehensive patient summary"""
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
            user_result = db.execute(
                text("""
                    SELECT first_name, last_name, email, date_of_birth, 
                           gender, blood_type, created_at
                    FROM users WHERE id = :patient_id
                """),
                {"patient_id": patient_id}
            )
            user_row = user_result.fetchone()
            
            if not user_row:
                return ToolCallResult(
                    tool_call_id=context.message_id,
                    tool_name=self.name,
                    status=ToolStatus.FAILED,
                    error="Patient not found"
                )
            
            meds_result = db.execute(
                text("""
                    SELECT COUNT(*) FROM patient_medications
                    WHERE patient_id = :patient_id AND status = 'active'
                """),
                {"patient_id": patient_id}
            )
            active_meds = meds_result.scalar() or 0
            
            alerts_result = db.execute(
                text("""
                    SELECT COUNT(*) FROM health_alerts
                    WHERE patient_id = :patient_id AND status = 'active'
                """),
                {"patient_id": patient_id}
            )
            active_alerts = alerts_result.scalar() or 0
            
            vitals_result = db.execute(
                text("""
                    SELECT metric_type, value, recorded_at
                    FROM daily_metrics
                    WHERE patient_id = :patient_id
                    ORDER BY recorded_at DESC
                    LIMIT 10
                """),
                {"patient_id": patient_id}
            )
            recent_vitals = [
                {"type": r[0], "value": r[1], "recorded_at": r[2].isoformat() if r[2] else None}
                for r in vitals_result.fetchall()
            ]
            
            age = None
            if user_row[3]:
                age = (datetime.now().date() - user_row[3]).days // 365
            
            return ToolCallResult(
                tool_call_id=context.message_id,
                tool_name=self.name,
                status=ToolStatus.COMPLETED,
                result={
                    "patient_id": patient_id,
                    "demographics": {
                        "name": f"{user_row[0] or ''} {user_row[1] or ''}".strip(),
                        "age": age,
                        "gender": user_row[4],
                        "blood_type": user_row[5],
                        "patient_since": user_row[6].isoformat() if user_row[6] else None
                    },
                    "summary": {
                        "active_medications": active_meds,
                        "active_alerts": active_alerts,
                        "recent_vitals": recent_vitals
                    },
                    "message": "Patient summary retrieved successfully"
                }
            )
        finally:
            db.close()
    
    async def _get_medical_history(
        self,
        parameters: Dict[str, Any],
        context: ToolExecutionContext
    ) -> ToolCallResult:
        """Get patient's medical history"""
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
        
        include_inactive = parameters.get("include_inactive", False)
        
        db = SessionLocal()
        try:
            conditions_result = db.execute(
                text("""
                    SELECT condition_name, icd10_code, onset_date, 
                           status, severity, notes
                    FROM patient_conditions
                    WHERE patient_id = :patient_id
                    ORDER BY onset_date DESC
                """),
                {"patient_id": patient_id}
            )
            
            conditions = []
            for row in conditions_result.fetchall():
                if not include_inactive and row[3] != "active":
                    continue
                conditions.append({
                    "condition": row[0],
                    "icd10_code": row[1],
                    "onset_date": row[2].isoformat() if row[2] else None,
                    "status": row[3],
                    "severity": row[4],
                    "notes": row[5]
                })
            
            encounters_result = db.execute(
                text("""
                    SELECT encounter_type, encounter_date, provider_id,
                           chief_complaint, diagnosis, notes
                    FROM patient_encounters
                    WHERE patient_id = :patient_id
                    ORDER BY encounter_date DESC
                    LIMIT 20
                """),
                {"patient_id": patient_id}
            )
            
            encounters = []
            for row in encounters_result.fetchall():
                encounters.append({
                    "type": row[0],
                    "date": row[1].isoformat() if row[1] else None,
                    "provider_id": row[2],
                    "chief_complaint": row[3],
                    "diagnosis": row[4],
                    "notes": row[5]
                })
            
            return ToolCallResult(
                tool_call_id=context.message_id,
                tool_name=self.name,
                status=ToolStatus.COMPLETED,
                result={
                    "patient_id": patient_id,
                    "conditions": conditions,
                    "encounters": encounters,
                    "condition_count": len(conditions),
                    "encounter_count": len(encounters),
                    "message": f"Retrieved {len(conditions)} conditions and {len(encounters)} encounters"
                }
            )
        finally:
            db.close()
    
    async def _get_vitals(
        self,
        parameters: Dict[str, Any],
        context: ToolExecutionContext
    ) -> ToolCallResult:
        """Get patient's vital signs"""
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
        
        date_from = parameters.get("date_from")
        date_to = parameters.get("date_to")
        
        if not date_from:
            date_from = (datetime.utcnow() - timedelta(days=30)).isoformat()
        if not date_to:
            date_to = datetime.utcnow().isoformat()
        
        db = SessionLocal()
        try:
            result = db.execute(
                text("""
                    SELECT metric_type, value, unit, recorded_at, source
                    FROM daily_metrics
                    WHERE patient_id = :patient_id
                    AND recorded_at >= :date_from
                    AND recorded_at <= :date_to
                    ORDER BY recorded_at DESC
                """),
                {
                    "patient_id": patient_id,
                    "date_from": date_from,
                    "date_to": date_to
                }
            )
            
            vitals = []
            vital_types = {}
            
            for row in result.fetchall():
                vital = {
                    "type": row[0],
                    "value": row[1],
                    "unit": row[2],
                    "recorded_at": row[3].isoformat() if row[3] else None,
                    "source": row[4]
                }
                vitals.append(vital)
                
                if row[0] not in vital_types:
                    vital_types[row[0]] = []
                vital_types[row[0]].append(float(row[1]) if row[1] else 0)
            
            averages = {}
            for vtype, values in vital_types.items():
                if values:
                    averages[vtype] = {
                        "avg": round(sum(values) / len(values), 2),
                        "min": round(min(values), 2),
                        "max": round(max(values), 2),
                        "count": len(values)
                    }
            
            return ToolCallResult(
                tool_call_id=context.message_id,
                tool_name=self.name,
                status=ToolStatus.COMPLETED,
                result={
                    "patient_id": patient_id,
                    "date_range": {"from": date_from, "to": date_to},
                    "vitals": vitals[:50],
                    "statistics": averages,
                    "total_readings": len(vitals),
                    "message": f"Retrieved {len(vitals)} vital readings"
                }
            )
        finally:
            db.close()
    
    async def _get_diagnoses(
        self,
        parameters: Dict[str, Any],
        context: ToolExecutionContext
    ) -> ToolCallResult:
        """Get patient's diagnoses"""
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
        
        include_inactive = parameters.get("include_inactive", False)
        
        db = SessionLocal()
        try:
            status_filter = "" if include_inactive else "AND status = 'active'"
            result = db.execute(
                text(f"""
                    SELECT condition_name, icd10_code, onset_date,
                           status, severity, diagnosed_by, notes
                    FROM patient_conditions
                    WHERE patient_id = :patient_id
                    {status_filter}
                    ORDER BY onset_date DESC
                """),
                {"patient_id": patient_id}
            )
            
            diagnoses = []
            for row in result.fetchall():
                diagnoses.append({
                    "diagnosis": row[0],
                    "icd10_code": row[1],
                    "onset_date": row[2].isoformat() if row[2] else None,
                    "status": row[3],
                    "severity": row[4],
                    "diagnosed_by": row[5],
                    "notes": row[6]
                })
            
            return ToolCallResult(
                tool_call_id=context.message_id,
                tool_name=self.name,
                status=ToolStatus.COMPLETED,
                result={
                    "patient_id": patient_id,
                    "diagnoses": diagnoses,
                    "active_count": len([d for d in diagnoses if d["status"] == "active"]),
                    "total_count": len(diagnoses),
                    "message": f"Retrieved {len(diagnoses)} diagnoses"
                }
            )
        finally:
            db.close()
    
    async def _get_allergies(
        self,
        parameters: Dict[str, Any],
        context: ToolExecutionContext
    ) -> ToolCallResult:
        """Get patient's allergies"""
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
                    SELECT allergen, allergy_type, severity, reaction,
                           onset_date, status, notes
                    FROM patient_allergies
                    WHERE patient_id = :patient_id
                    ORDER BY severity DESC, allergen
                """),
                {"patient_id": patient_id}
            )
            
            allergies = []
            for row in result.fetchall():
                allergies.append({
                    "allergen": row[0],
                    "type": row[1],
                    "severity": row[2],
                    "reaction": row[3],
                    "onset_date": row[4].isoformat() if row[4] else None,
                    "status": row[5],
                    "notes": row[6]
                })
            
            severe_count = len([a for a in allergies if a["severity"] in ["severe", "life-threatening"]])
            
            return ToolCallResult(
                tool_call_id=context.message_id,
                tool_name=self.name,
                status=ToolStatus.COMPLETED,
                result={
                    "patient_id": patient_id,
                    "allergies": allergies,
                    "total_count": len(allergies),
                    "severe_count": severe_count,
                    "has_severe_allergies": severe_count > 0,
                    "message": f"Patient has {len(allergies)} documented allergies"
                }
            )
        finally:
            db.close()
    
    async def _get_immunizations(
        self,
        parameters: Dict[str, Any],
        context: ToolExecutionContext
    ) -> ToolCallResult:
        """Get patient's immunization records"""
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
                    SELECT vaccine_name, cvx_code, administered_date,
                           lot_number, site, route, administered_by
                    FROM patient_immunizations
                    WHERE patient_id = :patient_id
                    ORDER BY administered_date DESC
                """),
                {"patient_id": patient_id}
            )
            
            immunizations = []
            for row in result.fetchall():
                immunizations.append({
                    "vaccine": row[0],
                    "cvx_code": row[1],
                    "administered_date": row[2].isoformat() if row[2] else None,
                    "lot_number": row[3],
                    "site": row[4],
                    "route": row[5],
                    "administered_by": row[6]
                })
            
            return ToolCallResult(
                tool_call_id=context.message_id,
                tool_name=self.name,
                status=ToolStatus.COMPLETED,
                result={
                    "patient_id": patient_id,
                    "immunizations": immunizations,
                    "total_count": len(immunizations),
                    "message": f"Retrieved {len(immunizations)} immunization records"
                }
            )
        finally:
            db.close()
    
    async def _get_procedures(
        self,
        parameters: Dict[str, Any],
        context: ToolExecutionContext
    ) -> ToolCallResult:
        """Get patient's procedure history"""
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
                    SELECT procedure_name, cpt_code, procedure_date,
                           performed_by, facility, outcome, notes
                    FROM patient_procedures
                    WHERE patient_id = :patient_id
                    ORDER BY procedure_date DESC
                """),
                {"patient_id": patient_id}
            )
            
            procedures = []
            for row in result.fetchall():
                procedures.append({
                    "procedure": row[0],
                    "cpt_code": row[1],
                    "date": row[2].isoformat() if row[2] else None,
                    "performed_by": row[3],
                    "facility": row[4],
                    "outcome": row[5],
                    "notes": row[6]
                })
            
            return ToolCallResult(
                tool_call_id=context.message_id,
                tool_name=self.name,
                status=ToolStatus.COMPLETED,
                result={
                    "patient_id": patient_id,
                    "procedures": procedures,
                    "total_count": len(procedures),
                    "message": f"Retrieved {len(procedures)} procedure records"
                }
            )
        finally:
            db.close()
    
    async def _get_care_plan(
        self,
        parameters: Dict[str, Any],
        context: ToolExecutionContext
    ) -> ToolCallResult:
        """Get patient's active care plan"""
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
                    SELECT id, title, description, start_date, end_date,
                           status, goals, interventions, created_by
                    FROM patient_care_plans
                    WHERE patient_id = :patient_id
                    AND status = 'active'
                    ORDER BY start_date DESC
                """),
                {"patient_id": patient_id}
            )
            
            care_plans = []
            for row in result.fetchall():
                care_plans.append({
                    "id": row[0],
                    "title": row[1],
                    "description": row[2],
                    "start_date": row[3].isoformat() if row[3] else None,
                    "end_date": row[4].isoformat() if row[4] else None,
                    "status": row[5],
                    "goals": row[6],
                    "interventions": row[7],
                    "created_by": row[8]
                })
            
            return ToolCallResult(
                tool_call_id=context.message_id,
                tool_name=self.name,
                status=ToolStatus.COMPLETED,
                result={
                    "patient_id": patient_id,
                    "care_plans": care_plans,
                    "active_count": len(care_plans),
                    "message": f"Retrieved {len(care_plans)} active care plans"
                }
            )
        finally:
            db.close()
