"""
Lab Fetch Tool Microservice
Laboratory results retrieval for agents
"""

import logging
import json
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import uuid

from app.services.agent_tools.base import BaseTool, ToolExecutionContext
from app.models.agent_models import ToolCallResult, ToolStatus

logger = logging.getLogger(__name__)


class LabFetchTool(BaseTool):
    """
    Laboratory results fetch tool for Assistant Lysa.
    Retrieves lab results, trends, and abnormal values.
    All access is logged for HIPAA compliance.
    """
    
    def __init__(self):
        super().__init__()
        self.name = "lab_fetch"
        self.display_name = "Laboratory Results Fetch"
        self.description = """
Retrieve laboratory test results and trends.
Actions: get_recent_labs, get_lab_history, get_abnormal_labs, get_lab_trends,
get_pending_orders, get_lab_panel
All access is logged for HIPAA audit compliance.
"""
        self.tool_type = "lab_fetch"
        self.requires_approval = False
        self.allowed_roles = ["doctor"]
        self.required_permissions = ["labs:read", "patient:view"]
        self.version = 1
    
    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "get_recent_labs",
                        "get_lab_history",
                        "get_abnormal_labs",
                        "get_lab_trends",
                        "get_pending_orders",
                        "get_lab_panel"
                    ],
                    "description": "The lab action to perform"
                },
                "patient_id": {
                    "type": "string",
                    "description": "Patient ID to fetch labs for"
                },
                "test_name": {
                    "type": "string",
                    "description": "Specific lab test name (e.g., 'CBC', 'BMP')"
                },
                "loinc_code": {
                    "type": "string",
                    "description": "LOINC code for the test"
                },
                "date_from": {
                    "type": "string",
                    "description": "Start date for lab history (ISO format)"
                },
                "date_to": {
                    "type": "string",
                    "description": "End date for lab history (ISO format)"
                },
                "panel_type": {
                    "type": "string",
                    "enum": ["cbc", "bmp", "cmp", "lipid", "thyroid", "liver", "kidney", "coagulation"],
                    "description": "Type of lab panel to retrieve"
                }
            },
            "required": ["action"]
        }
    
    async def execute(
        self,
        parameters: Dict[str, Any],
        context: ToolExecutionContext
    ) -> ToolCallResult:
        """Execute lab fetch action"""
        action = parameters.get("action")
        
        await self._log_phi_access(action, parameters, context)
        
        try:
            if action == "get_recent_labs":
                return await self._get_recent_labs(parameters, context)
            elif action == "get_lab_history":
                return await self._get_lab_history(parameters, context)
            elif action == "get_abnormal_labs":
                return await self._get_abnormal_labs(parameters, context)
            elif action == "get_lab_trends":
                return await self._get_lab_trends(parameters, context)
            elif action == "get_pending_orders":
                return await self._get_pending_orders(parameters, context)
            elif action == "get_lab_panel":
                return await self._get_lab_panel(parameters, context)
            else:
                return ToolCallResult(
                    tool_call_id=context.message_id,
                    tool_name=self.name,
                    status=ToolStatus.FAILED,
                    error=f"Unknown action: {action}"
                )
        except Exception as e:
            logger.error(f"Lab fetch error: {e}")
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
            resource_type="lab_results",
            resource_id=patient_id or "unknown",
            action=f"lab_fetch:{action}",
            status="success",
            metadata={
                "action": action,
                "patient_id": patient_id,
                "agent_id": context.agent_id,
                "user_role": context.user_role,
                "conversation_id": context.conversation_id,
                "phi_accessed": True,
                "phi_categories": ["laboratory_results"]
            }
        )
    
    async def _get_recent_labs(
        self,
        parameters: Dict[str, Any],
        context: ToolExecutionContext
    ) -> ToolCallResult:
        """Get patient's recent lab results"""
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
                    SELECT test_name, loinc_code, result_value, unit,
                           reference_range_low, reference_range_high,
                           is_abnormal, collected_at, resulted_at
                    FROM lab_results
                    WHERE patient_id = :patient_id
                    AND collected_at >= :date_from
                    ORDER BY collected_at DESC
                    LIMIT 50
                """),
                {
                    "patient_id": patient_id,
                    "date_from": datetime.utcnow() - timedelta(days=30)
                }
            )
            
            labs = []
            abnormal_count = 0
            
            for row in result.fetchall():
                is_abnormal = row[6]
                if is_abnormal:
                    abnormal_count += 1
                
                labs.append({
                    "test_name": row[0],
                    "loinc_code": row[1],
                    "value": row[2],
                    "unit": row[3],
                    "reference_range": f"{row[4]}-{row[5]}" if row[4] and row[5] else None,
                    "is_abnormal": is_abnormal,
                    "collected_at": row[7].isoformat() if row[7] else None,
                    "resulted_at": row[8].isoformat() if row[8] else None
                })
            
            return ToolCallResult(
                tool_call_id=context.message_id,
                tool_name=self.name,
                status=ToolStatus.COMPLETED,
                result={
                    "patient_id": patient_id,
                    "labs": labs,
                    "total_count": len(labs),
                    "abnormal_count": abnormal_count,
                    "has_abnormal_results": abnormal_count > 0,
                    "message": f"Retrieved {len(labs)} recent lab results ({abnormal_count} abnormal)"
                }
            )
        finally:
            db.close()
    
    async def _get_lab_history(
        self,
        parameters: Dict[str, Any],
        context: ToolExecutionContext
    ) -> ToolCallResult:
        """Get lab history for a specific test"""
        from app.database import SessionLocal
        from sqlalchemy import text
        
        patient_id = parameters.get("patient_id") or context.patient_id
        test_name = parameters.get("test_name")
        loinc_code = parameters.get("loinc_code")
        
        if not patient_id:
            return ToolCallResult(
                tool_call_id=context.message_id,
                tool_name=self.name,
                status=ToolStatus.FAILED,
                error="Patient ID required"
            )
        
        if not test_name and not loinc_code:
            return ToolCallResult(
                tool_call_id=context.message_id,
                tool_name=self.name,
                status=ToolStatus.FAILED,
                error="Test name or LOINC code required"
            )
        
        date_from = parameters.get("date_from")
        date_to = parameters.get("date_to")
        
        if not date_from:
            date_from = (datetime.utcnow() - timedelta(days=365)).isoformat()
        if not date_to:
            date_to = datetime.utcnow().isoformat()
        
        db = SessionLocal()
        try:
            filter_clause = "test_name ILIKE :test_name" if test_name else "loinc_code = :loinc_code"
            filter_value = {"test_name": f"%{test_name}%"} if test_name else {"loinc_code": loinc_code}
            
            result = db.execute(
                text(f"""
                    SELECT test_name, loinc_code, result_value, unit,
                           reference_range_low, reference_range_high,
                           is_abnormal, collected_at, ordering_provider
                    FROM lab_results
                    WHERE patient_id = :patient_id
                    AND {filter_clause}
                    AND collected_at >= :date_from
                    AND collected_at <= :date_to
                    ORDER BY collected_at DESC
                """),
                {
                    "patient_id": patient_id,
                    "date_from": date_from,
                    "date_to": date_to,
                    **filter_value
                }
            )
            
            history = []
            values = []
            
            for row in result.fetchall():
                history.append({
                    "test_name": row[0],
                    "loinc_code": row[1],
                    "value": row[2],
                    "unit": row[3],
                    "reference_range": f"{row[4]}-{row[5]}" if row[4] and row[5] else None,
                    "is_abnormal": row[6],
                    "collected_at": row[7].isoformat() if row[7] else None,
                    "ordering_provider": row[8]
                })
                if row[2]:
                    try:
                        values.append(float(row[2]))
                    except ValueError:
                        pass
            
            stats = {}
            if values:
                stats = {
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
                    "test_filter": test_name or loinc_code,
                    "date_range": {"from": date_from, "to": date_to},
                    "history": history,
                    "statistics": stats,
                    "total_results": len(history),
                    "message": f"Retrieved {len(history)} historical results"
                }
            )
        finally:
            db.close()
    
    async def _get_abnormal_labs(
        self,
        parameters: Dict[str, Any],
        context: ToolExecutionContext
    ) -> ToolCallResult:
        """Get abnormal lab results"""
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
        if not date_from:
            date_from = (datetime.utcnow() - timedelta(days=90)).isoformat()
        
        db = SessionLocal()
        try:
            result = db.execute(
                text("""
                    SELECT test_name, loinc_code, result_value, unit,
                           reference_range_low, reference_range_high,
                           abnormal_flag, collected_at
                    FROM lab_results
                    WHERE patient_id = :patient_id
                    AND is_abnormal = true
                    AND collected_at >= :date_from
                    ORDER BY collected_at DESC
                """),
                {
                    "patient_id": patient_id,
                    "date_from": date_from
                }
            )
            
            abnormal_labs = []
            critical_count = 0
            
            for row in result.fetchall():
                flag = row[6] or "abnormal"
                if flag.lower() in ["critical", "panic"]:
                    critical_count += 1
                
                abnormal_labs.append({
                    "test_name": row[0],
                    "loinc_code": row[1],
                    "value": row[2],
                    "unit": row[3],
                    "reference_range": f"{row[4]}-{row[5]}" if row[4] and row[5] else None,
                    "abnormal_flag": flag,
                    "collected_at": row[7].isoformat() if row[7] else None
                })
            
            return ToolCallResult(
                tool_call_id=context.message_id,
                tool_name=self.name,
                status=ToolStatus.COMPLETED,
                result={
                    "patient_id": patient_id,
                    "abnormal_labs": abnormal_labs,
                    "total_abnormal": len(abnormal_labs),
                    "critical_count": critical_count,
                    "has_critical_values": critical_count > 0,
                    "message": f"Found {len(abnormal_labs)} abnormal results ({critical_count} critical)"
                }
            )
        finally:
            db.close()
    
    async def _get_lab_trends(
        self,
        parameters: Dict[str, Any],
        context: ToolExecutionContext
    ) -> ToolCallResult:
        """Get trend analysis for specific lab tests"""
        from app.database import SessionLocal
        from sqlalchemy import text
        
        patient_id = parameters.get("patient_id") or context.patient_id
        test_name = parameters.get("test_name")
        
        if not patient_id:
            return ToolCallResult(
                tool_call_id=context.message_id,
                tool_name=self.name,
                status=ToolStatus.FAILED,
                error="Patient ID required"
            )
        
        if not test_name:
            return ToolCallResult(
                tool_call_id=context.message_id,
                tool_name=self.name,
                status=ToolStatus.FAILED,
                error="Test name required for trend analysis"
            )
        
        db = SessionLocal()
        try:
            result = db.execute(
                text("""
                    SELECT result_value, unit, collected_at,
                           reference_range_low, reference_range_high
                    FROM lab_results
                    WHERE patient_id = :patient_id
                    AND test_name ILIKE :test_name
                    ORDER BY collected_at ASC
                """),
                {
                    "patient_id": patient_id,
                    "test_name": f"%{test_name}%"
                }
            )
            
            data_points = []
            values = []
            
            for row in result.fetchall():
                try:
                    value = float(row[0]) if row[0] else None
                    if value is not None:
                        values.append(value)
                        data_points.append({
                            "value": value,
                            "unit": row[1],
                            "date": row[2].isoformat() if row[2] else None,
                            "ref_low": float(row[3]) if row[3] else None,
                            "ref_high": float(row[4]) if row[4] else None
                        })
                except ValueError:
                    continue
            
            trend = "stable"
            if len(values) >= 3:
                recent = values[-3:]
                if all(recent[i] < recent[i+1] for i in range(len(recent)-1)):
                    trend = "increasing"
                elif all(recent[i] > recent[i+1] for i in range(len(recent)-1)):
                    trend = "decreasing"
            
            stats = {}
            if values:
                stats = {
                    "current": values[-1] if values else None,
                    "avg": round(sum(values) / len(values), 2),
                    "min": round(min(values), 2),
                    "max": round(max(values), 2),
                    "std_dev": round((sum((v - sum(values)/len(values))**2 for v in values) / len(values))**0.5, 2) if len(values) > 1 else 0
                }
            
            return ToolCallResult(
                tool_call_id=context.message_id,
                tool_name=self.name,
                status=ToolStatus.COMPLETED,
                result={
                    "patient_id": patient_id,
                    "test_name": test_name,
                    "data_points": data_points,
                    "trend": trend,
                    "statistics": stats,
                    "total_readings": len(data_points),
                    "message": f"Trend is {trend} based on {len(data_points)} readings"
                }
            )
        finally:
            db.close()
    
    async def _get_pending_orders(
        self,
        parameters: Dict[str, Any],
        context: ToolExecutionContext
    ) -> ToolCallResult:
        """Get pending lab orders"""
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
                    SELECT id, test_name, loinc_code, ordered_at,
                           ordering_provider_id, priority, status, notes
                    FROM lab_orders
                    WHERE patient_id = :patient_id
                    AND status IN ('pending', 'ordered', 'in_progress')
                    ORDER BY priority DESC, ordered_at ASC
                """),
                {"patient_id": patient_id}
            )
            
            pending_orders = []
            for row in result.fetchall():
                pending_orders.append({
                    "order_id": row[0],
                    "test_name": row[1],
                    "loinc_code": row[2],
                    "ordered_at": row[3].isoformat() if row[3] else None,
                    "ordering_provider_id": row[4],
                    "priority": row[5],
                    "status": row[6],
                    "notes": row[7]
                })
            
            urgent_count = len([o for o in pending_orders if o["priority"] in ["urgent", "stat"]])
            
            return ToolCallResult(
                tool_call_id=context.message_id,
                tool_name=self.name,
                status=ToolStatus.COMPLETED,
                result={
                    "patient_id": patient_id,
                    "pending_orders": pending_orders,
                    "total_pending": len(pending_orders),
                    "urgent_count": urgent_count,
                    "message": f"{len(pending_orders)} pending lab orders ({urgent_count} urgent)"
                }
            )
        finally:
            db.close()
    
    async def _get_lab_panel(
        self,
        parameters: Dict[str, Any],
        context: ToolExecutionContext
    ) -> ToolCallResult:
        """Get a specific lab panel's results"""
        from app.database import SessionLocal
        from sqlalchemy import text
        
        patient_id = parameters.get("patient_id") or context.patient_id
        panel_type = parameters.get("panel_type")
        
        if not patient_id:
            return ToolCallResult(
                tool_call_id=context.message_id,
                tool_name=self.name,
                status=ToolStatus.FAILED,
                error="Patient ID required"
            )
        
        if not panel_type:
            return ToolCallResult(
                tool_call_id=context.message_id,
                tool_name=self.name,
                status=ToolStatus.FAILED,
                error="Panel type required"
            )
        
        panel_tests = {
            "cbc": ["WBC", "RBC", "Hemoglobin", "Hematocrit", "Platelets", "MCV", "MCH", "MCHC"],
            "bmp": ["Glucose", "BUN", "Creatinine", "Sodium", "Potassium", "Chloride", "CO2"],
            "cmp": ["Glucose", "BUN", "Creatinine", "Sodium", "Potassium", "Chloride", "CO2", 
                   "Calcium", "Total Protein", "Albumin", "Bilirubin", "AST", "ALT", "ALP"],
            "lipid": ["Total Cholesterol", "LDL", "HDL", "Triglycerides", "VLDL"],
            "thyroid": ["TSH", "T3", "T4", "Free T4"],
            "liver": ["AST", "ALT", "ALP", "Total Bilirubin", "Direct Bilirubin", "Albumin", "Total Protein"],
            "kidney": ["BUN", "Creatinine", "eGFR", "BUN/Creatinine Ratio"],
            "coagulation": ["PT", "PTT", "INR", "Fibrinogen"]
        }
        
        tests = panel_tests.get(panel_type, [])
        if not tests:
            return ToolCallResult(
                tool_call_id=context.message_id,
                tool_name=self.name,
                status=ToolStatus.FAILED,
                error=f"Unknown panel type: {panel_type}"
            )
        
        db = SessionLocal()
        try:
            placeholders = ", ".join([f":test_{i}" for i in range(len(tests))])
            test_params = {f"test_{i}": f"%{test}%" for i, test in enumerate(tests)}
            
            test_conditions = " OR ".join([f"test_name ILIKE :test_{i}" for i in range(len(tests))])
            
            result = db.execute(
                text(f"""
                    SELECT DISTINCT ON (test_name) test_name, result_value, unit,
                           reference_range_low, reference_range_high,
                           is_abnormal, collected_at
                    FROM lab_results
                    WHERE patient_id = :patient_id
                    AND ({test_conditions})
                    ORDER BY test_name, collected_at DESC
                """),
                {"patient_id": patient_id, **test_params}
            )
            
            panel_results = []
            abnormal_count = 0
            
            for row in result.fetchall():
                if row[5]:
                    abnormal_count += 1
                
                panel_results.append({
                    "test_name": row[0],
                    "value": row[1],
                    "unit": row[2],
                    "reference_range": f"{row[3]}-{row[4]}" if row[3] and row[4] else None,
                    "is_abnormal": row[5],
                    "collected_at": row[6].isoformat() if row[6] else None
                })
            
            return ToolCallResult(
                tool_call_id=context.message_id,
                tool_name=self.name,
                status=ToolStatus.COMPLETED,
                result={
                    "patient_id": patient_id,
                    "panel_type": panel_type,
                    "expected_tests": tests,
                    "results": panel_results,
                    "tests_found": len(panel_results),
                    "tests_missing": len(tests) - len(panel_results),
                    "abnormal_count": abnormal_count,
                    "message": f"Retrieved {len(panel_results)} of {len(tests)} panel tests"
                }
            )
        finally:
            db.close()
