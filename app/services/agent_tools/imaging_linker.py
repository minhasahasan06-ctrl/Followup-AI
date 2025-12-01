"""
Imaging Linker Tool Microservice
Medical imaging and radiology integration for agents
"""

import logging
import json
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import uuid

from app.services.agent_tools.base import BaseTool, ToolExecutionContext
from app.models.agent_models import ToolCallResult, ToolStatus

logger = logging.getLogger(__name__)


class ImagingLinkerTool(BaseTool):
    """
    Medical imaging integration tool for Assistant Lysa.
    Links and retrieves radiology studies, PACS integration, and imaging reports.
    All access is logged for HIPAA compliance.
    """
    
    def __init__(self):
        super().__init__()
        self.name = "imaging_linker"
        self.display_name = "Medical Imaging Linker"
        self.description = """
Link and retrieve medical imaging studies and radiology reports.
Actions: get_imaging_studies, get_study_details, get_radiology_report,
search_studies, get_comparison_studies, link_external_study
All access is logged for HIPAA audit compliance.
"""
        self.tool_type = "imaging_linker"
        self.requires_approval = False
        self.allowed_roles = ["doctor"]
        self.required_permissions = ["imaging:read", "patient:view"]
        self.version = 1
    
    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "get_imaging_studies",
                        "get_study_details",
                        "get_radiology_report",
                        "search_studies",
                        "get_comparison_studies",
                        "link_external_study"
                    ],
                    "description": "The imaging action to perform"
                },
                "patient_id": {
                    "type": "string",
                    "description": "Patient ID for imaging records"
                },
                "study_id": {
                    "type": "string",
                    "description": "Specific imaging study ID"
                },
                "modality": {
                    "type": "string",
                    "enum": ["CT", "MRI", "XR", "US", "NM", "PET", "MG", "CR", "DX"],
                    "description": "Imaging modality type"
                },
                "body_part": {
                    "type": "string",
                    "description": "Body part examined (e.g., 'chest', 'abdomen')"
                },
                "date_from": {
                    "type": "string",
                    "description": "Start date for study search (ISO format)"
                },
                "date_to": {
                    "type": "string",
                    "description": "End date for study search (ISO format)"
                },
                "accession_number": {
                    "type": "string",
                    "description": "Accession number for external study linking"
                },
                "dicom_uid": {
                    "type": "string",
                    "description": "DICOM Study Instance UID"
                }
            },
            "required": ["action"]
        }
    
    async def execute(
        self,
        parameters: Dict[str, Any],
        context: ToolExecutionContext
    ) -> ToolCallResult:
        """Execute imaging action"""
        action = parameters.get("action")
        
        await self._log_phi_access(action, parameters, context)
        
        try:
            if action == "get_imaging_studies":
                return await self._get_imaging_studies(parameters, context)
            elif action == "get_study_details":
                return await self._get_study_details(parameters, context)
            elif action == "get_radiology_report":
                return await self._get_radiology_report(parameters, context)
            elif action == "search_studies":
                return await self._search_studies(parameters, context)
            elif action == "get_comparison_studies":
                return await self._get_comparison_studies(parameters, context)
            elif action == "link_external_study":
                return await self._link_external_study(parameters, context)
            else:
                return ToolCallResult(
                    tool_call_id=context.message_id,
                    tool_name=self.name,
                    status=ToolStatus.FAILED,
                    error=f"Unknown action: {action}"
                )
        except Exception as e:
            logger.error(f"Imaging tool error: {e}")
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
        study_id = parameters.get("study_id")
        
        AuditLogger.log_event(
            event_type=AuditEvent.PHI_ACCESSED,
            user_id=context.user_id,
            resource_type="imaging",
            resource_id=study_id or patient_id or "unknown",
            action=f"imaging_linker:{action}",
            status="success",
            metadata={
                "action": action,
                "patient_id": patient_id,
                "study_id": study_id,
                "agent_id": context.agent_id,
                "user_role": context.user_role,
                "conversation_id": context.conversation_id,
                "phi_accessed": True,
                "phi_categories": ["medical_imaging", "radiology_reports"]
            }
        )
    
    async def _get_imaging_studies(
        self,
        parameters: Dict[str, Any],
        context: ToolExecutionContext
    ) -> ToolCallResult:
        """Get patient's imaging studies"""
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
        
        modality = parameters.get("modality")
        date_from = parameters.get("date_from")
        
        if not date_from:
            date_from = (datetime.utcnow() - timedelta(days=365)).isoformat()
        
        db = SessionLocal()
        try:
            modality_filter = "AND modality = :modality" if modality else ""
            
            result = db.execute(
                text(f"""
                    SELECT id, accession_number, study_date, modality,
                           body_part, study_description, referring_physician,
                           report_status, number_of_images, facility
                    FROM imaging_studies
                    WHERE patient_id = :patient_id
                    AND study_date >= :date_from
                    {modality_filter}
                    ORDER BY study_date DESC
                """),
                {
                    "patient_id": patient_id,
                    "date_from": date_from,
                    "modality": modality
                }
            )
            
            studies = []
            for row in result.fetchall():
                studies.append({
                    "study_id": row[0],
                    "accession_number": row[1],
                    "study_date": row[2].isoformat() if row[2] else None,
                    "modality": row[3],
                    "body_part": row[4],
                    "description": row[5],
                    "referring_physician": row[6],
                    "report_status": row[7],
                    "image_count": row[8],
                    "facility": row[9]
                })
            
            pending_reports = len([s for s in studies if s["report_status"] in ["pending", "preliminary"]])
            
            return ToolCallResult(
                tool_call_id=context.message_id,
                tool_name=self.name,
                status=ToolStatus.COMPLETED,
                result={
                    "patient_id": patient_id,
                    "modality_filter": modality,
                    "studies": studies,
                    "total_count": len(studies),
                    "pending_reports": pending_reports,
                    "message": f"Found {len(studies)} imaging studies"
                }
            )
        finally:
            db.close()
    
    async def _get_study_details(
        self,
        parameters: Dict[str, Any],
        context: ToolExecutionContext
    ) -> ToolCallResult:
        """Get detailed information about a specific study"""
        from app.database import SessionLocal
        from sqlalchemy import text
        
        study_id = parameters.get("study_id")
        if not study_id:
            return ToolCallResult(
                tool_call_id=context.message_id,
                tool_name=self.name,
                status=ToolStatus.FAILED,
                error="Study ID required"
            )
        
        db = SessionLocal()
        try:
            result = db.execute(
                text("""
                    SELECT s.id, s.patient_id, s.accession_number, s.study_date,
                           s.modality, s.body_part, s.study_description,
                           s.referring_physician, s.performing_physician,
                           s.report_status, s.number_of_images, s.number_of_series,
                           s.dicom_study_uid, s.facility, s.indication,
                           s.contrast_used, s.radiation_dose,
                           u.first_name, u.last_name
                    FROM imaging_studies s
                    LEFT JOIN users u ON s.patient_id = u.id
                    WHERE s.id = :study_id
                """),
                {"study_id": study_id}
            )
            
            row = result.fetchone()
            if not row:
                return ToolCallResult(
                    tool_call_id=context.message_id,
                    tool_name=self.name,
                    status=ToolStatus.FAILED,
                    error="Study not found"
                )
            
            patient_name = f"{row[17] or ''} {row[18] or ''}".strip()
            
            series_result = db.execute(
                text("""
                    SELECT series_number, series_description, modality,
                           number_of_images, body_part
                    FROM imaging_series
                    WHERE study_id = :study_id
                    ORDER BY series_number
                """),
                {"study_id": study_id}
            )
            
            series = []
            for sr in series_result.fetchall():
                series.append({
                    "series_number": sr[0],
                    "description": sr[1],
                    "modality": sr[2],
                    "image_count": sr[3],
                    "body_part": sr[4]
                })
            
            return ToolCallResult(
                tool_call_id=context.message_id,
                tool_name=self.name,
                status=ToolStatus.COMPLETED,
                result={
                    "study": {
                        "study_id": row[0],
                        "patient_id": row[1],
                        "patient_name": patient_name,
                        "accession_number": row[2],
                        "study_date": row[3].isoformat() if row[3] else None,
                        "modality": row[4],
                        "body_part": row[5],
                        "description": row[6],
                        "referring_physician": row[7],
                        "performing_physician": row[8],
                        "report_status": row[9],
                        "image_count": row[10],
                        "series_count": row[11],
                        "dicom_uid": row[12],
                        "facility": row[13],
                        "indication": row[14],
                        "contrast_used": row[15],
                        "radiation_dose": row[16]
                    },
                    "series": series,
                    "message": "Study details retrieved successfully"
                }
            )
        finally:
            db.close()
    
    async def _get_radiology_report(
        self,
        parameters: Dict[str, Any],
        context: ToolExecutionContext
    ) -> ToolCallResult:
        """Get radiology report for a study"""
        from app.database import SessionLocal
        from sqlalchemy import text
        
        study_id = parameters.get("study_id")
        if not study_id:
            return ToolCallResult(
                tool_call_id=context.message_id,
                tool_name=self.name,
                status=ToolStatus.FAILED,
                error="Study ID required"
            )
        
        db = SessionLocal()
        try:
            result = db.execute(
                text("""
                    SELECT id, study_id, report_status, findings, impression,
                           comparison_studies, technique, clinical_history,
                           radiologist_id, signed_at, created_at,
                           addendum, addendum_at
                    FROM radiology_reports
                    WHERE study_id = :study_id
                    ORDER BY created_at DESC
                    LIMIT 1
                """),
                {"study_id": study_id}
            )
            
            row = result.fetchone()
            if not row:
                return ToolCallResult(
                    tool_call_id=context.message_id,
                    tool_name=self.name,
                    status=ToolStatus.FAILED,
                    error="Report not found for this study"
                )
            
            radiologist_name = None
            if row[8]:
                rad_result = db.execute(
                    text("SELECT first_name, last_name FROM users WHERE id = :id"),
                    {"id": row[8]}
                )
                rad_row = rad_result.fetchone()
                if rad_row:
                    radiologist_name = f"Dr. {rad_row[1] or rad_row[0]}"
            
            return ToolCallResult(
                tool_call_id=context.message_id,
                tool_name=self.name,
                status=ToolStatus.COMPLETED,
                result={
                    "report": {
                        "report_id": row[0],
                        "study_id": row[1],
                        "status": row[2],
                        "findings": row[3],
                        "impression": row[4],
                        "comparison_studies": row[5],
                        "technique": row[6],
                        "clinical_history": row[7],
                        "radiologist": radiologist_name,
                        "signed_at": row[9].isoformat() if row[9] else None,
                        "created_at": row[10].isoformat() if row[10] else None,
                        "addendum": row[11],
                        "addendum_at": row[12].isoformat() if row[12] else None
                    },
                    "has_addendum": row[11] is not None,
                    "is_final": row[2] == "final",
                    "message": "Radiology report retrieved"
                }
            )
        finally:
            db.close()
    
    async def _search_studies(
        self,
        parameters: Dict[str, Any],
        context: ToolExecutionContext
    ) -> ToolCallResult:
        """Search imaging studies with filters"""
        from app.database import SessionLocal
        from sqlalchemy import text
        
        patient_id = parameters.get("patient_id") or context.patient_id
        modality = parameters.get("modality")
        body_part = parameters.get("body_part")
        date_from = parameters.get("date_from")
        date_to = parameters.get("date_to")
        
        if not patient_id:
            return ToolCallResult(
                tool_call_id=context.message_id,
                tool_name=self.name,
                status=ToolStatus.FAILED,
                error="Patient ID required"
            )
        
        filters = ["patient_id = :patient_id"]
        filter_params = {"patient_id": patient_id}
        
        if modality:
            filters.append("modality = :modality")
            filter_params["modality"] = modality
        
        if body_part:
            filters.append("body_part ILIKE :body_part")
            filter_params["body_part"] = f"%{body_part}%"
        
        if date_from:
            filters.append("study_date >= :date_from")
            filter_params["date_from"] = date_from
        
        if date_to:
            filters.append("study_date <= :date_to")
            filter_params["date_to"] = date_to
        
        db = SessionLocal()
        try:
            result = db.execute(
                text(f"""
                    SELECT id, accession_number, study_date, modality,
                           body_part, study_description, report_status
                    FROM imaging_studies
                    WHERE {' AND '.join(filters)}
                    ORDER BY study_date DESC
                    LIMIT 50
                """),
                filter_params
            )
            
            studies = []
            for row in result.fetchall():
                studies.append({
                    "study_id": row[0],
                    "accession_number": row[1],
                    "study_date": row[2].isoformat() if row[2] else None,
                    "modality": row[3],
                    "body_part": row[4],
                    "description": row[5],
                    "report_status": row[6]
                })
            
            return ToolCallResult(
                tool_call_id=context.message_id,
                tool_name=self.name,
                status=ToolStatus.COMPLETED,
                result={
                    "patient_id": patient_id,
                    "filters_applied": {
                        "modality": modality,
                        "body_part": body_part,
                        "date_range": {"from": date_from, "to": date_to}
                    },
                    "studies": studies,
                    "total_found": len(studies),
                    "message": f"Found {len(studies)} studies matching criteria"
                }
            )
        finally:
            db.close()
    
    async def _get_comparison_studies(
        self,
        parameters: Dict[str, Any],
        context: ToolExecutionContext
    ) -> ToolCallResult:
        """Get comparison studies for a given study"""
        from app.database import SessionLocal
        from sqlalchemy import text
        
        study_id = parameters.get("study_id")
        if not study_id:
            return ToolCallResult(
                tool_call_id=context.message_id,
                tool_name=self.name,
                status=ToolStatus.FAILED,
                error="Study ID required"
            )
        
        db = SessionLocal()
        try:
            current = db.execute(
                text("""
                    SELECT patient_id, modality, body_part, study_date
                    FROM imaging_studies
                    WHERE id = :study_id
                """),
                {"study_id": study_id}
            )
            
            current_row = current.fetchone()
            if not current_row:
                return ToolCallResult(
                    tool_call_id=context.message_id,
                    tool_name=self.name,
                    status=ToolStatus.FAILED,
                    error="Study not found"
                )
            
            patient_id, modality, body_part, study_date = current_row
            
            result = db.execute(
                text("""
                    SELECT id, accession_number, study_date, modality,
                           body_part, study_description, report_status
                    FROM imaging_studies
                    WHERE patient_id = :patient_id
                    AND modality = :modality
                    AND body_part = :body_part
                    AND id != :study_id
                    AND study_date < :study_date
                    ORDER BY study_date DESC
                    LIMIT 10
                """),
                {
                    "patient_id": patient_id,
                    "modality": modality,
                    "body_part": body_part,
                    "study_id": study_id,
                    "study_date": study_date
                }
            )
            
            comparison_studies = []
            for row in result.fetchall():
                days_prior = (study_date - row[2]).days if study_date and row[2] else None
                comparison_studies.append({
                    "study_id": row[0],
                    "accession_number": row[1],
                    "study_date": row[2].isoformat() if row[2] else None,
                    "modality": row[3],
                    "body_part": row[4],
                    "description": row[5],
                    "report_status": row[6],
                    "days_prior": days_prior
                })
            
            return ToolCallResult(
                tool_call_id=context.message_id,
                tool_name=self.name,
                status=ToolStatus.COMPLETED,
                result={
                    "current_study_id": study_id,
                    "current_study_date": study_date.isoformat() if study_date else None,
                    "modality": modality,
                    "body_part": body_part,
                    "comparison_studies": comparison_studies,
                    "total_comparisons": len(comparison_studies),
                    "message": f"Found {len(comparison_studies)} prior studies for comparison"
                }
            )
        finally:
            db.close()
    
    async def _link_external_study(
        self,
        parameters: Dict[str, Any],
        context: ToolExecutionContext
    ) -> ToolCallResult:
        """Link an external imaging study to patient record"""
        from app.database import SessionLocal
        from sqlalchemy import text
        
        patient_id = parameters.get("patient_id") or context.patient_id
        accession_number = parameters.get("accession_number")
        dicom_uid = parameters.get("dicom_uid")
        
        if not patient_id:
            return ToolCallResult(
                tool_call_id=context.message_id,
                tool_name=self.name,
                status=ToolStatus.FAILED,
                error="Patient ID required"
            )
        
        if not accession_number and not dicom_uid:
            return ToolCallResult(
                tool_call_id=context.message_id,
                tool_name=self.name,
                status=ToolStatus.FAILED,
                error="Accession number or DICOM UID required"
            )
        
        db = SessionLocal()
        try:
            if accession_number:
                check = db.execute(
                    text("""
                        SELECT id, patient_id FROM imaging_studies
                        WHERE accession_number = :accession_number
                    """),
                    {"accession_number": accession_number}
                )
            else:
                check = db.execute(
                    text("""
                        SELECT id, patient_id FROM imaging_studies
                        WHERE dicom_study_uid = :dicom_uid
                    """),
                    {"dicom_uid": dicom_uid}
                )
            
            existing = check.fetchone()
            
            if existing:
                if existing[1] == patient_id:
                    return ToolCallResult(
                        tool_call_id=context.message_id,
                        tool_name=self.name,
                        status=ToolStatus.COMPLETED,
                        result={
                            "study_id": existing[0],
                            "already_linked": True,
                            "message": "Study is already linked to this patient"
                        }
                    )
                else:
                    return ToolCallResult(
                        tool_call_id=context.message_id,
                        tool_name=self.name,
                        status=ToolStatus.FAILED,
                        error="Study is linked to a different patient"
                    )
            
            link_id = str(uuid.uuid4())
            db.execute(
                text("""
                    INSERT INTO external_study_links (
                        id, patient_id, accession_number, dicom_uid,
                        link_status, linked_by, linked_at
                    ) VALUES (
                        :id, :patient_id, :accession_number, :dicom_uid,
                        'pending', :linked_by, NOW()
                    )
                """),
                {
                    "id": link_id,
                    "patient_id": patient_id,
                    "accession_number": accession_number,
                    "dicom_uid": dicom_uid,
                    "linked_by": context.user_id
                }
            )
            db.commit()
            
            return ToolCallResult(
                tool_call_id=context.message_id,
                tool_name=self.name,
                status=ToolStatus.COMPLETED,
                result={
                    "link_id": link_id,
                    "patient_id": patient_id,
                    "accession_number": accession_number,
                    "dicom_uid": dicom_uid,
                    "status": "pending",
                    "message": "External study link request created. Study will be imported when available."
                }
            )
        finally:
            db.close()
