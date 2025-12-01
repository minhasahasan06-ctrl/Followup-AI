"""
Agent Engine Service
Manages AI agent decision loop, tool execution, and memory management
"""

import os
import json
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import asyncio
from openai import AsyncOpenAI

from app.models.agent_models import (
    AgentConfig, AgentPersona, AgentMemoryPolicy,
    MessageEnvelope, MessageType, ActorType,
    AgentDecisionContext, AgentDecisionResult,
    ToolCallRequest, ToolCallResult, ToolStatus,
    MemoryCreate, MemoryType, ApprovalRequest
)
from app.services.agent_tools.base import (
    ToolRegistry, ToolExecutionContext, initialize_tools
)
from app.services.audit_logger import AuditLogger, AuditEvent

logger = logging.getLogger(__name__)


# Agent Configurations
AGENT_CLONA_CONFIG = AgentConfig(
    id="clona",
    name="Agent Clona",
    description="Personal AI health companion for immunocompromised patients",
    agent_type="companion",
    target_role="patient",
    persona=AgentPersona(
        system_prompt="""You are Agent Clona, a compassionate and supportive AI health companion for immunocompromised patients.

Your primary responsibilities:
1. Conduct daily health check-ins with empathy and care
2. Help patients log symptoms accurately using proper medical terminology
3. Track medication adherence and provide gentle reminders
4. Monitor wellness indicators and identify concerning patterns
5. Provide emotional support while maintaining appropriate boundaries
6. Escalate urgent health concerns to medical professionals

Communication style:
- Warm, supportive, and non-judgmental
- Use simple language but be medically accurate
- Acknowledge patient concerns and validate emotions
- Never provide medical diagnoses - always recommend consulting healthcare providers
- Be proactive about health deterioration indicators

HIPAA Compliance:
- Never share patient information with unauthorized parties
- Always treat all health information as confidential
- Log all PHI access for audit purposes""",
        personality="Compassionate, supportive, patient, thorough",
        tone="Warm and caring, yet professional",
        specializations=[
            "Daily health monitoring",
            "Symptom tracking and journaling",
            "Medication reminders",
            "Emotional wellness support",
            "Deterioration indicator detection"
        ],
        constraints=[
            "Never provide medical diagnoses",
            "Always recommend professional consultation for medical concerns",
            "Escalate urgent symptoms immediately",
            "Maintain HIPAA compliance at all times",
            "No access to prescription or treatment modification"
        ]
    ),
    memory_policy=AgentMemoryPolicy(
        short_term_ttl_hours=2,
        long_term_enabled=True,
        vectorization_enabled=True,
        max_memory_per_patient=1000,
        summarization_threshold=10
    ),
    openai_model="gpt-4o"
)

AGENT_LYSA_CONFIG = AgentConfig(
    id="lysa",
    name="Assistant Lysa",
    description="AI-powered clinical assistant for healthcare providers",
    agent_type="assistant",
    target_role="doctor",
    persona=AgentPersona(
        system_prompt="""You are Assistant Lysa, an AI-powered clinical assistant for healthcare providers managing immunocompromised patients.

Your primary responsibilities:
1. Provide patient summaries and health status overviews
2. Triage and prioritize patient communications and alerts
3. Assist with appointment scheduling and calendar management
4. Categorize and summarize emails and messages
5. Draft clinical notes and documentation
6. Flag potential drug interactions and contraindications
7. Support clinical decision-making with evidence-based insights

Communication style:
- Professional, concise, and clinically accurate
- Use proper medical terminology
- Present information in prioritized, actionable formats
- Highlight urgent items and potential risks
- Be thorough but time-efficient

Clinical Support:
- Cross-reference patient data with current evidence
- Identify patterns across patient populations
- Support (never replace) clinical judgment
- Flag items requiring physician review

HIPAA Compliance:
- Strict access control to patient records
- Full audit logging of all PHI access
- Human-in-the-loop for all care-affecting decisions
- Never share patient data outside authorized channels""",
        personality="Professional, efficient, thorough, supportive",
        tone="Clinical and precise, yet approachable",
        specializations=[
            "Patient health summaries",
            "Alert triage and prioritization",
            "Schedule management",
            "Email and communication categorization",
            "Clinical documentation support",
            "Drug interaction checking",
            "Evidence-based clinical insights"
        ],
        constraints=[
            "Human approval required for prescriptions",
            "Human approval required for treatment changes",
            "Must flag all potential drug interactions",
            "Cannot modify patient records directly",
            "All clinical suggestions require physician review"
        ]
    ),
    memory_policy=AgentMemoryPolicy(
        short_term_ttl_hours=4,
        long_term_enabled=True,
        vectorization_enabled=True,
        max_memory_per_patient=2000,
        summarization_threshold=15
    ),
    openai_model="gpt-4o"
)


class AgentEngine:
    """Core agent decision engine using OpenAI"""

    def __init__(self):
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.agents = {
            "clona": AGENT_CLONA_CONFIG,
            "lysa": AGENT_LYSA_CONFIG
        }
        self._initialized = False

    async def initialize(self):
        """Initialize the agent engine"""
        if self._initialized:
            return

        logger.info("Initializing Agent Engine...")
        
        # Initialize and register all tools
        try:
            initialize_tools()
            registered_tools = ToolRegistry.get_all()
            logger.info(f"Registered {len(registered_tools)} agent tools")
        except Exception as e:
            logger.warning(f"Failed to initialize tools: {e}")
        
        # Verify OpenAI API access
        try:
            await self.client.models.list()
            logger.info("OpenAI API connection verified")
        except Exception as e:
            logger.error(f"Failed to connect to OpenAI API: {e}")
            raise

        self._initialized = True
        logger.info("Agent Engine initialized successfully")
    
    async def execute_tool_calls(
        self,
        tool_calls: List[ToolCallRequest],
        agent_id: str,
        user_id: str,
        user_role: str,
        patient_id: Optional[str] = None,
        doctor_id: Optional[str] = None,
        conversation_id: Optional[str] = None
    ) -> List[ToolCallResult]:
        """
        Execute tool calls using the ToolRegistry.
        Handles consent verification and HIPAA audit logging.
        
        Returns list of ToolCallResult for each tool call.
        """
        results = []
        
        for tool_call in tool_calls:
            # Skip tools requiring approval - they need human confirmation first
            if tool_call.requires_approval:
                results.append(ToolCallResult(
                    tool_call_id=tool_call.message_id or "",
                    tool_name=tool_call.tool_name,
                    status=ToolStatus.PENDING_APPROVAL,
                    result={
                        "message": "This action requires doctor approval before execution.",
                        "parameters": tool_call.parameters
                    }
                ))
                continue
            
            # Build execution context
            context = ToolExecutionContext(
                user_id=user_id,
                user_role=user_role,
                agent_id=agent_id,
                conversation_id=conversation_id or tool_call.conversation_id,
                message_id=tool_call.message_id or "",
                patient_id=patient_id,
                doctor_id=doctor_id
            )
            
            try:
                # Execute tool via registry
                result = await ToolRegistry.execute(
                    tool_name=tool_call.tool_name,
                    parameters=tool_call.parameters,
                    context=context
                )
                results.append(result)
                
                # Log successful execution
                AuditLogger.log_event(
                    event_type="tool_execution",
                    user_id=user_id,
                    resource_type="tool",
                    resource_id=tool_call.tool_name,
                    action="execute",
                    status="success" if result.status == ToolStatus.COMPLETED else "failed",
                    metadata={
                        "agent_id": agent_id,
                        "tool_name": tool_call.tool_name,
                        "patient_id": patient_id,
                        "execution_time_ms": result.execution_time_ms
                    }
                )
                
            except Exception as e:
                logger.error(f"Tool execution failed: {tool_call.tool_name} - {e}")
                results.append(ToolCallResult(
                    tool_call_id=tool_call.message_id or "",
                    tool_name=tool_call.tool_name,
                    status=ToolStatus.FAILED,
                    error=str(e)
                ))
        
        return results
    
    async def execute_approved_tool(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        approval_id: str,
        approver_id: str,
        user_id: str,
        user_role: str,
        patient_id: Optional[str] = None,
        doctor_id: Optional[str] = None,
        conversation_id: Optional[str] = None
    ) -> ToolCallResult:
        """
        Execute a tool that has been approved by a human (doctor).
        Used for prescription drafts and other sensitive operations.
        """
        context = ToolExecutionContext(
            user_id=user_id,
            user_role=user_role,
            agent_id="lysa",  # Approved tools are typically Lysa's
            conversation_id=conversation_id or "",
            message_id=approval_id,
            patient_id=patient_id,
            doctor_id=doctor_id or approver_id
        )
        
        # Log approval action
        AuditLogger.log_event(
            event_type="tool_approval",
            user_id=approver_id,
            resource_type="tool",
            resource_id=tool_name,
            action="approve",
            status="success",
            metadata={
                "approval_id": approval_id,
                "tool_name": tool_name,
                "patient_id": patient_id,
                "parameters": parameters
            }
        )
        
        # Get tool and execute without approval check
        tool = ToolRegistry.get(tool_name)
        if not tool:
            return ToolCallResult(
                tool_call_id=approval_id,
                tool_name=tool_name,
                status=ToolStatus.FAILED,
                error=f"Tool not found: {tool_name}"
            )
        
        try:
            result = await tool.execute(parameters, context)
            await tool.log_execution(context, result, parameters)
            return result
        except Exception as e:
            logger.error(f"Approved tool execution failed: {tool_name} - {e}")
            return ToolCallResult(
                tool_call_id=approval_id,
                tool_name=tool_name,
                status=ToolStatus.FAILED,
                error=str(e)
            )

    def get_agent(self, agent_id: str) -> Optional[AgentConfig]:
        """Get agent configuration by ID"""
        return self.agents.get(agent_id)

    def get_agent_for_user(self, user_role: str) -> AgentConfig:
        """Get appropriate agent for user role"""
        if user_role == "doctor":
            return self.agents["lysa"]
        return self.agents["clona"]
    
    def get_available_tools(self, user_role: str) -> List[Any]:
        """Get available tools for a user role from the registry"""
        return ToolRegistry.get_for_role(user_role)

    async def process_message(
        self,
        agent_id: str,
        context: AgentDecisionContext
    ) -> AgentDecisionResult:
        """
        Main agent decision loop - processes incoming message and decides response
        """
        agent = self.get_agent(agent_id)
        if not agent:
            raise ValueError(f"Unknown agent: {agent_id}")

        # Build conversation history
        messages = self._build_conversation_messages(agent, context)

        # Build tools list for function calling
        tools = self._build_tools_list(context.available_tools)

        try:
            # Call OpenAI for decision
            create_kwargs = {
                "model": agent.openai_model,
                "messages": messages,  # type: ignore
                "temperature": 0.7,
                "max_tokens": 2000
            }
            if tools:
                create_kwargs["tools"] = tools  # type: ignore
                create_kwargs["tool_choice"] = "auto"
            
            response = await self.client.chat.completions.create(**create_kwargs)  # type: ignore

            # Process response
            return await self._process_response(agent, context, response)

        except Exception as e:
            logger.error(f"Error in agent decision loop: {e}")
            return AgentDecisionResult(
                response_message=f"I apologize, but I encountered an error processing your request. Please try again or contact support if the issue persists.",
                tool_calls=[],
                memory_updates=[],
                requires_human_confirmation=False
            )

    def _build_conversation_messages(
        self,
        agent: AgentConfig,
        context: AgentDecisionContext
    ) -> List[Dict[str, Any]]:
        """Build OpenAI messages array from context"""
        messages = [
            {"role": "system", "content": agent.persona.system_prompt}
        ]

        # Add patient context if available
        if context.patient_context:
            patient_summary = self._format_patient_context(context.patient_context)
            messages.append({
                "role": "system",
                "content": f"Current patient context:\n{patient_summary}"
            })

        # Add relevant memory
        if context.short_term_memory:
            memory_summary = self._format_memory(context.short_term_memory)
            messages.append({
                "role": "system",
                "content": f"Recent conversation context:\n{memory_summary}"
            })

        if context.long_term_memory:
            long_term_summary = self._format_memory(context.long_term_memory)
            messages.append({
                "role": "system",
                "content": f"Relevant patient history:\n{long_term_summary}"
            })

        # Add the current message
        msg = context.message
        if msg.payload and msg.payload.get("content"):
            messages.append({
                "role": "user",
                "content": msg.payload["content"]
            })

        return messages

    def _build_tools_list(
        self,
        available_tools: List[Any]
    ) -> List[Dict[str, Any]]:
        """Build OpenAI tools specification from available tools"""
        if not available_tools:
            return []

        tools = []
        for tool in available_tools:
            tools.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": tool.parameters_schema or {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            })
        return tools

    async def _process_response(
        self,
        agent: AgentConfig,
        context: AgentDecisionContext,
        response: Any
    ) -> AgentDecisionResult:
        """Process OpenAI response and build decision result"""
        choice = response.choices[0]
        message = choice.message

        tool_calls = []
        requires_human_confirmation = False
        confirmation_details = None

        # Check for tool calls
        if message.tool_calls:
            for tc in message.tool_calls:
                tool_name = tc.function.name
                tool_input = json.loads(tc.function.arguments)

                # Find tool definition
                tool_def = next(
                    (t for t in context.available_tools if t.name == tool_name),
                    None
                )

                requires_approval = tool_def.requires_approval if tool_def else False

                tool_call = ToolCallRequest(
                    tool_name=tool_name,
                    parameters=tool_input,
                    conversation_id=context.conversation_id,
                    message_id=context.message.msg_id,
                    requires_approval=requires_approval
                )
                tool_calls.append(tool_call)

                # Check if human approval needed
                if requires_approval:
                    requires_human_confirmation = True
                    confirmation_details = ApprovalRequest(
                        message_id=context.message.msg_id,
                        tool_name=tool_name,
                        tool_input=tool_input,
                        requested_by=agent.id,
                        patient_id=context.patient_context.get("patient_id") if context.patient_context else None,
                        reason=f"Tool {tool_name} requires human approval before execution",
                        urgency="normal"
                    )

        # Extract response message
        response_message = message.content

        # Create memory update for this interaction
        memory_updates = []
        if context.message.payload and context.message.payload.get("content"):
            memory_updates.append(MemoryCreate(
                agent_id=agent.id,
                patient_id=context.patient_context.get("patient_id") if context.patient_context else None,
                user_id=context.message.from_.id if hasattr(context.message, 'from_') else None,
                conversation_id=context.conversation_id,
                memory_type=MemoryType.SHORT_TERM,
                storage_type="redis",
                content=f"User: {context.message.payload['content']}\nAgent: {response_message or '[Tool calls pending]'}",
                importance=0.5,
                expires_at=datetime.utcnow() + timedelta(hours=agent.memory_policy.short_term_ttl_hours),
                metadata={
                    "message_id": context.message.msg_id,
                    "has_tool_calls": len(tool_calls) > 0
                }
            ))

        return AgentDecisionResult(
            response_message=response_message,
            tool_calls=tool_calls,
            memory_updates=memory_updates,
            requires_human_confirmation=requires_human_confirmation,
            confirmation_details=confirmation_details
        )

    def _format_patient_context(self, context: Dict[str, Any]) -> str:
        """Format patient context for system prompt"""
        parts = []
        if context.get("name"):
            parts.append(f"Patient: {context['name']}")
        if context.get("conditions"):
            parts.append(f"Conditions: {', '.join(context['conditions'])}")
        if context.get("medications"):
            parts.append(f"Current medications: {', '.join(context['medications'])}")
        if context.get("recent_symptoms"):
            parts.append(f"Recent symptoms: {', '.join(context['recent_symptoms'])}")
        if context.get("risk_score"):
            parts.append(f"Current risk score: {context['risk_score']}/15")
        return "\n".join(parts)

    def _format_memory(self, memories: List[Dict[str, Any]]) -> str:
        """Format memory entries for context"""
        if not memories:
            return "No relevant context available."
        
        parts = []
        for mem in memories[-5:]:  # Last 5 entries
            content = mem.get("content", "")
            if len(content) > 500:
                content = content[:500] + "..."
            parts.append(content)
        return "\n---\n".join(parts)


# Singleton instance
agent_engine = AgentEngine()


async def get_agent_engine() -> AgentEngine:
    """Get initialized agent engine instance"""
    if not agent_engine._initialized:
        await agent_engine.initialize()
    return agent_engine
