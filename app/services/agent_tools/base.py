"""
Base Tool Class and Tool Registry
Abstract base for all agent tool microservices
"""

import os
import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Type
from datetime import datetime
from pydantic import BaseModel

from app.models.agent_models import (
    ToolCallRequest, ToolCallResult, ToolStatus,
    AgentToolDefinition, ToolConfig
)
from app.services.audit_logger import AuditLogger, AuditEvent
from app.services.consent_service import ConsentService, get_consent_service

logger = logging.getLogger(__name__)


class ToolExecutionContext(BaseModel):
    """Context passed to tool execution"""
    user_id: str
    user_role: str
    patient_id: Optional[str] = None
    doctor_id: Optional[str] = None
    conversation_id: str
    message_id: str
    agent_id: str
    request_id: Optional[str] = None


class BaseTool(ABC):
    """
    Abstract base class for all agent tool microservices.
    Each tool must implement execute() and define its schema.
    """
    
    def __init__(self):
        self.name: str = ""
        self.display_name: str = ""
        self.description: str = ""
        self.tool_type: str = ""
        self.requires_approval: bool = False
        self.approval_role: Optional[str] = None
        self.allowed_roles: List[str] = []
        self.required_permissions: List[str] = []
        self.version: int = 1
        self.is_enabled: bool = True
    
    @abstractmethod
    def get_parameters_schema(self) -> Dict[str, Any]:
        """Return JSON schema for tool parameters"""
        pass
    
    @abstractmethod
    async def execute(
        self,
        parameters: Dict[str, Any],
        context: ToolExecutionContext
    ) -> ToolCallResult:
        """Execute the tool with given parameters"""
        pass
    
    def to_definition(self) -> AgentToolDefinition:
        """Convert to AgentToolDefinition for OpenAI function calling"""
        return AgentToolDefinition(
            id=f"tool_{self.name}",
            name=self.name,
            display_name=self.display_name,
            description=self.description,
            tool_type=self.tool_type,
            config=ToolConfig(
                parameters_schema=self.get_parameters_schema(),
                timeout=30
            ),
            parameters_schema=self.get_parameters_schema(),
            required_permissions=self.required_permissions,
            allowed_roles=self.allowed_roles,
            requires_approval=self.requires_approval,
            approval_role=self.approval_role,
            is_enabled=self.is_enabled,
            version=self.version
        )
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """Validate parameters against schema. Returns (is_valid, error_message)"""
        schema = self.get_parameters_schema()
        required = schema.get("required", [])
        properties = schema.get("properties", {})
        
        for field in required:
            if field not in parameters:
                return False, f"Missing required parameter: {field}"
        
        for key, value in parameters.items():
            if key not in properties:
                continue
            
            prop_schema = properties[key]
            expected_type = prop_schema.get("type")
            
            if expected_type == "string" and not isinstance(value, str):
                return False, f"Parameter {key} must be a string"
            elif expected_type == "integer" and not isinstance(value, int):
                return False, f"Parameter {key} must be an integer"
            elif expected_type == "number" and not isinstance(value, (int, float)):
                return False, f"Parameter {key} must be a number"
            elif expected_type == "boolean" and not isinstance(value, bool):
                return False, f"Parameter {key} must be a boolean"
            elif expected_type == "array" and not isinstance(value, list):
                return False, f"Parameter {key} must be an array"
            elif expected_type == "object" and not isinstance(value, dict):
                return False, f"Parameter {key} must be an object"
        
        return True, None
    
    def check_permissions(self, context: ToolExecutionContext) -> tuple[bool, Optional[str]]:
        """Check if user has permission to execute this tool"""
        if not self.allowed_roles:
            return True, None
        
        if context.user_role not in self.allowed_roles:
            return False, f"User role '{context.user_role}' not allowed to use this tool"
        
        return True, None
    
    def verify_consent(self, context: ToolExecutionContext) -> tuple[bool, Optional[str]]:
        """
        Verify doctor-patient consent relationship for tool execution.
        Required for tools that access patient data.
        """
        if not context.patient_id:
            return True, None
        
        if not context.doctor_id and context.user_role != "doctor":
            return True, None
        
        doctor_id = context.doctor_id or context.user_id
        
        try:
            consent_service = get_consent_service()
            is_valid, reason, _ = consent_service.verify_connection(
                doctor_id=doctor_id,
                patient_id=context.patient_id
            )
            
            if not is_valid:
                return False, f"No active consent relationship: {reason}"
            
            return True, None
        except Exception as e:
            logger.error(f"Consent verification failed: {e}")
            return True, None
    
    async def log_execution(
        self,
        context: ToolExecutionContext,
        result: ToolCallResult,
        parameters: Dict[str, Any]
    ):
        """Log tool execution for HIPAA audit trail using AuditLogger"""
        try:
            status = "success" if result.status == ToolStatus.COMPLETED else "failure"
            
            AuditLogger.log_event(
                event_type=AuditEvent.PHI_ACCESSED if self._accesses_phi() else "tool_execution",
                user_id=context.user_id,
                resource_type="tool",
                resource_id=context.message_id,
                action=f"execute:{self.name}",
                status=status,
                metadata={
                    "tool_name": self.name,
                    "agent_id": context.agent_id,
                    "user_role": context.user_role,
                    "patient_id": context.patient_id,
                    "conversation_id": context.conversation_id,
                    "parameters": self._sanitize_params_for_log(parameters),
                    "result_status": result.status.value,
                    "execution_time_ms": result.execution_time_ms,
                    "error": result.error,
                    "requires_approval": self.requires_approval,
                    "phi_accessed": self._accesses_phi()
                }
            )
            
            from app.database import SessionLocal
            from sqlalchemy import text
            
            db = SessionLocal()
            try:
                db.execute(
                    text("""
                        INSERT INTO agent_audit_logs (
                            actor_type, actor_id, action_type, resource_type, resource_id,
                            details, success, created_at
                        ) VALUES (
                            :actor_type, :actor_id, :action_type, :resource_type, :resource_id,
                            :details::jsonb, :success, NOW()
                        )
                    """),
                    {
                        "actor_type": "agent",
                        "actor_id": context.agent_id,
                        "action_type": f"tool_execution:{self.name}",
                        "resource_type": "tool",
                        "resource_id": context.message_id,
                        "details": json.dumps({
                            "tool_name": self.name,
                            "user_id": context.user_id,
                            "user_role": context.user_role,
                            "patient_id": context.patient_id,
                            "conversation_id": context.conversation_id,
                            "parameters": self._sanitize_params_for_log(parameters),
                            "result_status": result.status.value,
                            "execution_time_ms": result.execution_time_ms,
                            "error": result.error,
                            "phi_accessed": self._accesses_phi()
                        }),
                        "success": result.status == ToolStatus.COMPLETED
                    }
                )
                db.commit()
            finally:
                db.close()
        except Exception as e:
            logger.error(f"Failed to log tool execution: {e}")
    
    def _accesses_phi(self) -> bool:
        """Check if this tool accesses PHI data"""
        phi_tools = ["ehr_fetch", "lab_fetch", "imaging_linker", "prescription_draft"]
        return self.name in phi_tools
    
    def _sanitize_params_for_log(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Remove or mask sensitive data from parameters before logging"""
        sensitive_keys = ["password", "secret", "token", "key", "ssn", "dob", "mrn"]
        sanitized = {}
        
        for key, value in parameters.items():
            key_lower = key.lower()
            if any(s in key_lower for s in sensitive_keys):
                sanitized[key] = "[REDACTED]"
            elif isinstance(value, str) and len(value) > 500:
                sanitized[key] = value[:100] + "...[TRUNCATED]"
            else:
                sanitized[key] = value
        
        return sanitized


class ToolRegistry:
    """
    Central registry for all agent tools.
    Manages tool registration, lookup, and execution.
    """
    
    _instance = None
    _tools: Dict[str, BaseTool] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._tools = {}
        return cls._instance
    
    @classmethod
    def register(cls, tool: BaseTool):
        """Register a tool in the registry"""
        instance = cls()
        instance._tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name} ({tool.display_name})")
    
    @classmethod
    def get(cls, tool_name: str) -> Optional[BaseTool]:
        """Get a tool by name"""
        instance = cls()
        return instance._tools.get(tool_name)
    
    @classmethod
    def get_all(cls) -> List[BaseTool]:
        """Get all registered tools"""
        instance = cls()
        return list(instance._tools.values())
    
    @classmethod
    def get_for_agent(cls, agent_id: str) -> List[BaseTool]:
        """Get tools available for a specific agent"""
        instance = cls()
        
        if agent_id == "clona":
            allowed_types = ["messaging", "symptom_log", "medication_reminder"]
        elif agent_id == "lysa":
            allowed_types = ["calendar", "messaging", "prescription_draft", 
                            "ehr_fetch", "lab_fetch", "imaging_linker"]
        else:
            allowed_types = []
        
        return [t for t in instance._tools.values() 
                if t.tool_type in allowed_types and t.is_enabled]
    
    @classmethod
    def get_for_role(cls, user_role: str) -> List[BaseTool]:
        """Get tools available for a specific user role"""
        instance = cls()
        
        return [t for t in instance._tools.values() 
                if not t.allowed_roles or user_role in t.allowed_roles]
    
    @classmethod
    def get_definitions(cls, agent_id: Optional[str] = None) -> List[AgentToolDefinition]:
        """Get tool definitions for OpenAI function calling"""
        instance = cls()
        
        if agent_id:
            tools = cls.get_for_agent(agent_id)
        else:
            tools = instance.get_all()
        
        return [t.to_definition() for t in tools if t.is_enabled]
    
    @classmethod
    async def execute(
        cls,
        tool_name: str,
        parameters: Dict[str, Any],
        context: ToolExecutionContext
    ) -> ToolCallResult:
        """Execute a tool by name"""
        import time
        start_time = time.time()
        
        tool = cls.get(tool_name)
        if not tool:
            return ToolCallResult(
                tool_call_id=context.message_id,
                tool_name=tool_name,
                status=ToolStatus.FAILED,
                error=f"Tool not found: {tool_name}"
            )
        
        if not tool.is_enabled:
            return ToolCallResult(
                tool_call_id=context.message_id,
                tool_name=tool_name,
                status=ToolStatus.FAILED,
                error=f"Tool is disabled: {tool_name}"
            )
        
        perm_ok, perm_error = tool.check_permissions(context)
        if not perm_ok:
            return ToolCallResult(
                tool_call_id=context.message_id,
                tool_name=tool_name,
                status=ToolStatus.FAILED,
                error=perm_error
            )
        
        consent_ok, consent_error = tool.verify_consent(context)
        if not consent_ok:
            AuditLogger.log_event(
                event_type="consent_verification_failed",
                user_id=context.user_id,
                resource_type="tool",
                resource_id=tool_name,
                action="execute",
                status="denied",
                metadata={
                    "tool_name": tool_name,
                    "patient_id": context.patient_id,
                    "doctor_id": context.doctor_id,
                    "reason": consent_error
                }
            )
            return ToolCallResult(
                tool_call_id=context.message_id,
                tool_name=tool_name,
                status=ToolStatus.FAILED,
                error=f"Consent verification failed: {consent_error}"
            )
        
        valid, validation_error = tool.validate_parameters(parameters)
        if not valid:
            return ToolCallResult(
                tool_call_id=context.message_id,
                tool_name=tool_name,
                status=ToolStatus.FAILED,
                error=validation_error
            )
        
        if tool.requires_approval:
            return ToolCallResult(
                tool_call_id=context.message_id,
                tool_name=tool_name,
                status=ToolStatus.PENDING_APPROVAL,
                result={"parameters": parameters, "requires_approval_from": tool.approval_role}
            )
        
        try:
            result = await tool.execute(parameters, context)
            execution_time_ms = int((time.time() - start_time) * 1000)
            result.execution_time_ms = execution_time_ms
            
            await tool.log_execution(context, result, parameters)
            
            return result
            
        except Exception as e:
            logger.error(f"Tool execution failed: {tool_name} - {e}")
            execution_time_ms = int((time.time() - start_time) * 1000)
            
            result = ToolCallResult(
                tool_call_id=context.message_id,
                tool_name=tool_name,
                status=ToolStatus.FAILED,
                error=str(e),
                execution_time_ms=execution_time_ms
            )
            
            await tool.log_execution(context, result, parameters)
            
            return result


def initialize_tools():
    """Initialize and register all tools"""
    from app.services.agent_tools.calendar import CalendarTool
    from app.services.agent_tools.messaging import MessagingTool
    from app.services.agent_tools.prescription_draft import PrescriptionDraftTool
    from app.services.agent_tools.ehr_fetch import EHRFetchTool
    from app.services.agent_tools.lab_fetch import LabFetchTool
    from app.services.agent_tools.imaging_linker import ImagingLinkerTool
    
    ToolRegistry.register(CalendarTool())
    ToolRegistry.register(MessagingTool())
    ToolRegistry.register(PrescriptionDraftTool())
    ToolRegistry.register(EHRFetchTool())
    ToolRegistry.register(LabFetchTool())
    ToolRegistry.register(ImagingLinkerTool())
    
    logger.info(f"Initialized {len(ToolRegistry.get_all())} agent tools")
