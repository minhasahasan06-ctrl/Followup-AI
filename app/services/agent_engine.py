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
from app.services.memory_service import MemoryService, get_memory_service

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
        self.memory_service: Optional[MemoryService] = None
        self._initialized = False
        self._memory_initialized = False

    def set_memory_service(self, memory_service: Optional[MemoryService]):
        """
        Set the memory service instance.
        Called by FastAPI startup or dependency injection to avoid
        awaiting dependencies during module initialization.
        """
        # Runtime validation to prevent coroutine assignment
        import asyncio
        if memory_service is not None:
            if asyncio.iscoroutine(memory_service) or asyncio.iscoroutinefunction(memory_service):
                logger.error("FATAL: Attempted to set memory_service with coroutine - memory persistence will fail")
                self.memory_service = None
                self._memory_initialized = False
                return
            if not hasattr(memory_service, 'get_short_term_memories'):
                logger.error("FATAL: Invalid memory_service instance - missing required methods")
                self.memory_service = None
                self._memory_initialized = False
                return
        
        self.memory_service = memory_service
        if memory_service:
            self._memory_initialized = True
            logger.info("Memory service connected to Agent Engine via setter")

    async def ensure_memory_service(self) -> bool:
        """
        Ensure memory service is available. Called internally before any memory operation.
        Returns True if memory service is available, False otherwise.
        """
        if self.memory_service and self._memory_initialized:
            return True
        
        # Try to get memory service if not already set
        try:
            mem_service = await get_memory_service()
            self.set_memory_service(mem_service)
            return self.memory_service is not None
        except Exception as e:
            logger.warning(f"Could not initialize memory service: {e}")
            return False

    async def initialize(self):
        """Initialize the agent engine (tools and OpenAI verification only)"""
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

    async def enrich_context_with_memory(
        self,
        agent_id: str,
        user_id: str,
        conversation_id: str,
        context: AgentDecisionContext,
        query: Optional[str] = None
    ) -> AgentDecisionContext:
        """
        Enrich decision context with short-term and long-term memories.
        Called before processing a message to provide historical context.
        Returns a new context instance to respect Pydantic immutability patterns.
        """
        # Ensure memory service is available
        has_memory = await self.ensure_memory_service()
        if not has_memory:
            logger.debug("Memory service not available, skipping context enrichment")
            return context
        
        try:
            # Fetch short-term memories (recent conversation context)
            short_term = await self.memory_service.get_short_term_memories(
                agent_id=agent_id,
                user_id=user_id,
                conversation_id=conversation_id,
                limit=5
            )
            
            # Fetch long-term memories using semantic search if we have a query
            long_term = []
            if query:
                long_term = await self.memory_service.search_long_term(
                    agent_id=agent_id,
                    patient_id=user_id,
                    query=query,
                    limit=3
                )
            
            # Return a new context with updated memories (Pydantic-safe)
            enriched_context = context.model_copy(update={
                "short_term_memory": short_term,
                "long_term_memory": long_term
            })
            
            logger.debug(f"Enriched context with {len(short_term)} short-term and {len(long_term)} long-term memories")
            return enriched_context
            
        except Exception as e:
            logger.error(f"Failed to enrich context with memory: {e}")
            return context

    async def store_interaction_memory(
        self,
        agent_id: str,
        user_id: str,
        conversation_id: str,
        user_message: str,
        agent_response: str,
        extracted_symptoms: Optional[List[str]] = None,
        health_indicators: Optional[Dict[str, Any]] = None
    ):
        """
        Store interaction in short-term memory and optionally long-term memory.
        Called after processing a message to persist the interaction.
        """
        # Ensure memory service is available
        has_memory = await self.ensure_memory_service()
        if not has_memory:
            logger.warning(f"Memory service not available, conversation {conversation_id} not persisted")
            return
        
        try:
            # Store in short-term memory (recent context)
            await self.memory_service.store_short_term(
                agent_id=agent_id,
                user_id=user_id,
                conversation_id=conversation_id,
                content=f"User: {user_message}\nAgent: {agent_response[:500]}",
                ttl_hours=2,
                metadata={
                    "has_symptoms": bool(extracted_symptoms),
                    "symptom_count": len(extracted_symptoms) if extracted_symptoms else 0
                }
            )
            
            # Store in long-term memory if health-relevant
            agent = self.get_agent(agent_id)
            if agent and agent.memory_policy.long_term_enabled:
                # Determine if this interaction should be persisted long-term
                should_persist = (
                    extracted_symptoms or 
                    health_indicators or
                    any(keyword in user_message.lower() for keyword in [
                        "symptom", "pain", "medication", "doctor", "feeling",
                        "fever", "tired", "sick", "medicine", "prescription"
                    ])
                )
                
                if should_persist:
                    # Extract only serializable primitive fields from health_indicators
                    safe_health_data = None
                    if health_indicators:
                        safe_health_data = {
                            "patient_id": health_indicators.get("patient_id"),
                            "risk_score": health_indicators.get("risk_score"),
                            "conditions": health_indicators.get("conditions", []),
                            "medications": health_indicators.get("medications", []),
                            "recent_symptoms": health_indicators.get("recent_symptoms", [])
                        }
                        # Filter out None values
                        safe_health_data = {k: v for k, v in safe_health_data.items() if v is not None}
                    
                    await self.memory_service.store_long_term(
                        agent_id=agent_id,
                        patient_id=user_id,
                        content=f"Conversation on {datetime.utcnow().strftime('%Y-%m-%d')}:\nPatient: {user_message}\nAgent Clona: {agent_response}",
                        memory_type="episodic",
                        source_type="conversation",
                        source_id=conversation_id,
                        importance=0.7 if extracted_symptoms else 0.4,
                        metadata={
                            "extracted_symptoms": extracted_symptoms,
                            "health_context": safe_health_data,
                            "conversation_id": conversation_id
                        },
                        auto_summarize=len(user_message + agent_response) > 1000
                    )
                    logger.debug(f"Stored health-relevant interaction in long-term memory")
            
            logger.debug(f"Stored interaction memory for conversation {conversation_id}")
            
        except Exception as e:
            logger.error(f"Failed to store interaction memory: {e}")

    async def process_message(
        self,
        agent_id: str,
        context: AgentDecisionContext,
        user_id: Optional[str] = None
    ) -> AgentDecisionResult:
        """
        Main agent decision loop - processes incoming message and decides response.
        Now includes memory enrichment for full context awareness.
        """
        agent = self.get_agent(agent_id)
        if not agent:
            raise ValueError(f"Unknown agent: {agent_id}")

        # Extract user_id from context if not provided
        if user_id is None and context.message and context.message.sender:
            user_id = context.message.sender.id
        
        # Extract message content for memory search
        message_content = None
        if context.message and context.message.payload:
            message_content = context.message.payload.get("content")
        
        # Enrich context with memory if we have user_id
        if user_id and self.memory_service:
            context = await self.enrich_context_with_memory(
                agent_id=agent_id,
                user_id=user_id,
                conversation_id=context.conversation_id,
                context=context,
                query=message_content
            )

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
            result = await self._process_response(agent, context, response)
            
            # Store interaction in memory if we have user_id
            if user_id and self.memory_service and result.response_message:
                await self.store_interaction_memory(
                    agent_id=agent_id,
                    user_id=user_id,
                    conversation_id=context.conversation_id,
                    user_message=message_content or "",
                    agent_response=result.response_message,
                    health_indicators=context.patient_context
                )
            
            return result

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
            # Get user_id from sender (the 'sender' attribute aliases 'from' in JSON)
            sender_id = context.message.sender.id if hasattr(context.message, 'sender') and context.message.sender else None
            memory_updates.append(MemoryCreate(
                agent_id=agent.id,
                patient_id=context.patient_context.get("patient_id") if context.patient_context else None,
                user_id=sender_id,
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
    
    async def process_message_streaming(
        self,
        agent_id: str,
        context: AgentDecisionContext,
        on_chunk: Any = None,
        user_id: Optional[str] = None,
        skip_memory: bool = False
    ):
        """
        Stream agent response in real-time for better UX.
        Yields response chunks as they're generated using async iteration.
        Memory enrichment is applied unless skip_memory=True (when caller already enriched).
        """
        agent = self.get_agent(agent_id)
        if not agent:
            raise ValueError(f"Unknown agent: {agent_id}")
        
        # Extract user_id from context if not provided
        if user_id is None and context.message and context.message.sender:
            user_id = context.message.sender.id
        
        # Enrich context with memory if not already done and we have user_id
        if not skip_memory and user_id and self.memory_service:
            # Only enrich if context doesn't already have memories
            if not context.short_term_memory and not context.long_term_memory:
                message_content = None
                if context.message and context.message.payload:
                    message_content = context.message.payload.get("content")
                context = await self.enrich_context_with_memory(
                    agent_id=agent_id,
                    user_id=user_id,
                    conversation_id=context.conversation_id,
                    context=context,
                    query=message_content
                )
        
        messages = self._build_conversation_messages(agent, context)
        tools = self._build_tools_list(context.available_tools)
        
        full_response = ""
        tool_calls_data: List[Dict[str, Any]] = []
        
        try:
            create_kwargs: Dict[str, Any] = {
                "model": agent.openai_model,
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 2000,
                "stream": True
            }
            if tools:
                create_kwargs["tools"] = tools
                create_kwargs["tool_choice"] = "auto"
            
            # Create the stream - OpenAI returns an async iterator directly
            stream = await self.client.chat.completions.create(**create_kwargs)  # type: ignore
            
            # Iterate over streaming chunks
            async for chunk in stream:
                if not chunk.choices:
                    continue
                    
                delta = chunk.choices[0].delta
                
                # Handle content streaming
                if delta and delta.content:
                    full_response += delta.content
                    chunk_data = {"type": "content", "content": delta.content}
                    if on_chunk:
                        try:
                            await on_chunk(chunk_data)
                        except Exception:
                            pass
                    yield chunk_data
                
                # Handle tool call streaming
                if delta and delta.tool_calls:
                    for tc in delta.tool_calls:
                        if tc.index is not None:
                            while len(tool_calls_data) <= tc.index:
                                tool_calls_data.append({"name": "", "arguments": "", "id": None})
                            
                            if tc.function:
                                if tc.function.name:
                                    tool_calls_data[tc.index]["name"] = tc.function.name
                                if tc.id:
                                    tool_calls_data[tc.index]["id"] = tc.id
                                if tc.function.arguments:
                                    tool_calls_data[tc.index]["arguments"] += tc.function.arguments
            
            # Process any tool calls after stream completes
            tool_call_requests = []
            requires_approval = False
            
            for tc_data in tool_calls_data:
                if tc_data.get("name"):
                    try:
                        tool_input = json.loads(tc_data["arguments"]) if tc_data["arguments"] else {}
                    except json.JSONDecodeError:
                        tool_input = {}
                    
                    tool_def = next(
                        (t for t in context.available_tools if t.name == tc_data["name"]),
                        None
                    )
                    
                    tool_call = ToolCallRequest(
                        tool_name=tc_data["name"],
                        parameters=tool_input,
                        conversation_id=context.conversation_id,
                        message_id=context.message.msg_id,
                        requires_approval=tool_def.requires_approval if tool_def else False
                    )
                    tool_call_requests.append(tool_call)
                    
                    tc_requires_approval = tool_def.requires_approval if tool_def else False
                    if tc_requires_approval:
                        requires_approval = True
                    
                    # Emit tool call event
                    yield {
                        "type": "tool_call",
                        "tool_name": tc_data["name"],
                        "parameters": tool_input,
                        "requires_approval": tc_requires_approval,
                        "tool_id": tc_data.get("id")
                    }
            
            # Emit final complete event
            yield {
                "type": "complete",
                "response": full_response,
                "tool_calls": [
                    {
                        "tool_name": tc.tool_name,
                        "parameters": tc.parameters,
                        "requires_approval": tc.requires_approval
                    } for tc in tool_call_requests
                ],
                "requires_approval": requires_approval
            }
            
        except Exception as e:
            logger.error(f"Error in streaming agent response: {e}", exc_info=True)
            yield {
                "type": "error",
                "error": str(e)
            }
    
    async def extract_symptoms_from_message(
        self,
        message: str,
        patient_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Extract symptoms and health indicators from patient message using GPT-4o.
        Used by Agent Clona for health monitoring.
        """
        extraction_prompt = """Analyze the following patient message and extract health-related information.

Return a JSON object with:
{
  "symptoms": [
    {
      "name": "symptom name",
      "body_location": "affected area if mentioned",
      "severity": "mild/moderate/severe or 1-10 if mentioned",
      "duration": "how long if mentioned",
      "frequency": "how often if mentioned"
    }
  ],
  "vital_signs": {
    "temperature": null,
    "heart_rate": null,
    "blood_pressure": null,
    "oxygen_saturation": null
  },
  "mood_indicators": ["list of emotional states mentioned"],
  "medication_mentions": ["medications mentioned"],
  "concerning_patterns": ["any red flags or concerning patterns"],
  "urgency_level": "routine/elevated/urgent/emergency",
  "follow_up_questions": ["suggested follow-up questions to ask"]
}

Only include fields that are actually present or can be inferred from the message.
"""
        
        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": extraction_prompt},
                    {"role": "user", "content": f"Patient message: {message}"}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content or "{}")
            
            # Log symptom extraction for audit
            AuditLogger.log_event(
                event_type=AuditEvent.PHI_ACCESSED,
                user_id="system",
                resource_type="symptom_extraction",
                resource_id=str(datetime.utcnow().timestamp()),
                action="extract",
                status="success",
                metadata={
                    "symptoms_found": len(result.get("symptoms", [])),
                    "urgency_level": result.get("urgency_level", "routine")
                },
                patient_id=patient_context.get("patient_id") if patient_context else None,
                phi_accessed=True,
                phi_categories=["symptoms", "health_data"]
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Symptom extraction failed: {e}")
            return {"symptoms": [], "urgency_level": "unknown", "error": str(e)}
    
    async def get_health_context_summary(
        self,
        patient_id: str
    ) -> Dict[str, Any]:
        """
        Get comprehensive health context for a patient to inform Clona's responses.
        Aggregates data from multiple sources.
        """
        try:
            from app.database import SessionLocal
            from sqlalchemy import text
            
            db = SessionLocal()
            try:
                # Get recent symptoms
                symptoms_result = db.execute(text("""
                    SELECT symptom_type, severity, recorded_at
                    FROM symptom_entries
                    WHERE patient_id = :patient_id
                    AND recorded_at > NOW() - INTERVAL '7 days'
                    ORDER BY recorded_at DESC
                    LIMIT 10
                """), {"patient_id": patient_id})
                recent_symptoms = [
                    {"type": row[0], "severity": row[1], "date": row[2].isoformat() if row[2] else None}
                    for row in symptoms_result.fetchall()
                ]
                
                # Get current medications
                meds_result = db.execute(text("""
                    SELECT medication_name, dosage, frequency
                    FROM patient_medications
                    WHERE patient_id = :patient_id
                    AND status = 'active'
                """), {"patient_id": patient_id})
                medications = [
                    {"name": row[0], "dosage": row[1], "frequency": row[2]}
                    for row in meds_result.fetchall()
                ]
                
                # Get risk score
                risk_result = db.execute(text("""
                    SELECT composite_score, respiratory_score, pain_score
                    FROM risk_scores
                    WHERE patient_id = :patient_id
                    ORDER BY calculated_at DESC
                    LIMIT 1
                """), {"patient_id": patient_id})
                risk_row = risk_result.fetchone()
                risk_score = {
                    "composite": risk_row[0] if risk_row else 0,
                    "respiratory": risk_row[1] if risk_row else 0,
                    "pain": risk_row[2] if risk_row else 0
                } if risk_row else None
                
                # Get recent health alerts
                alerts_result = db.execute(text("""
                    SELECT alert_type, severity, message, created_at
                    FROM health_alerts
                    WHERE patient_id = :patient_id
                    AND created_at > NOW() - INTERVAL '24 hours'
                    AND status = 'active'
                    ORDER BY severity DESC, created_at DESC
                    LIMIT 5
                """), {"patient_id": patient_id})
                alerts = [
                    {"type": row[0], "severity": row[1], "message": row[2]}
                    for row in alerts_result.fetchall()
                ]
                
                return {
                    "recent_symptoms": recent_symptoms,
                    "medications": medications,
                    "risk_score": risk_score,
                    "active_alerts": alerts,
                    "has_concerning_patterns": len(alerts) > 0 or (risk_score and risk_score.get("composite", 0) > 8)
                }
                
            finally:
                db.close()
                
        except Exception as e:
            logger.error(f"Failed to get health context: {e}")
            return {"error": str(e)}


    async def assess_clinical_urgency(
        self,
        patient_id: str,
        current_message: str,
        extracted_symptoms: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Assess clinical urgency combining extracted symptoms with historical data.
        Used to determine if immediate escalation is needed.
        
        Returns urgency assessment with recommended actions.
        """
        try:
            AuditLogger.log_event(
                event_type=AuditEvent.PHI_ACCESSED,
                user_id=user_id or "agent_clona",
                resource_type="clinical_urgency_assessment",
                resource_id=patient_id,
                action="assess",
                status="initiated",
                patient_id=patient_id,
                phi_accessed=True,
                phi_categories=["symptoms", "health_alerts", "risk_scores"]
            )
            
            health_context = await self.get_health_context_summary(patient_id)
            
            if not extracted_symptoms:
                extracted_symptoms = await self.extract_symptoms_from_message(
                    current_message,
                    {"patient_id": patient_id}
                )
            
            urgent_symptoms = [
                s for s in extracted_symptoms.get("symptoms", [])
                if s.get("severity") in ["severe", "9", "10"] or 
                   s.get("name", "").lower() in [
                       "chest pain", "difficulty breathing", "confusion",
                       "severe bleeding", "fainting", "seizure"
                   ]
            ]
            
            concerning_patterns = extracted_symptoms.get("concerning_patterns", [])
            base_urgency = extracted_symptoms.get("urgency_level", "routine")
            risk_score = health_context.get("risk_score", {})
            active_alerts = health_context.get("active_alerts", [])
            
            urgency_score = 0
            reasons = []
            
            if base_urgency == "emergency":
                urgency_score += 10
                reasons.append("Emergency-level symptoms detected")
            elif base_urgency == "urgent":
                urgency_score += 7
                reasons.append("Urgent symptoms requiring prompt attention")
            elif base_urgency == "elevated":
                urgency_score += 4
                reasons.append("Elevated concern level")
            
            if urgent_symptoms:
                urgency_score += len(urgent_symptoms) * 3
                reasons.append(f"{len(urgent_symptoms)} severe/critical symptoms")
            
            if concerning_patterns:
                urgency_score += len(concerning_patterns) * 2
                reasons.append(f"{len(concerning_patterns)} concerning patterns")
            
            if risk_score and risk_score.get("composite", 0) > 10:
                urgency_score += 3
                reasons.append("High baseline risk score")
            
            high_severity_alerts = [a for a in active_alerts if a.get("severity") in ["high", "critical"]]
            if high_severity_alerts:
                urgency_score += len(high_severity_alerts) * 2
                reasons.append(f"{len(high_severity_alerts)} high-severity active alerts")
            
            if urgency_score >= 10:
                urgency_level = "critical"
                action = "immediate_escalation"
            elif urgency_score >= 7:
                urgency_level = "high"
                action = "urgent_notification"
            elif urgency_score >= 4:
                urgency_level = "moderate"
                action = "schedule_followup"
            else:
                urgency_level = "low"
                action = "continue_monitoring"
            
            return {
                "urgency_level": urgency_level,
                "urgency_score": min(urgency_score, 15),
                "recommended_action": action,
                "reasons": reasons,
                "urgent_symptoms": urgent_symptoms,
                "should_escalate": urgency_score >= 7,
                "escalation_priority": "immediate" if urgency_score >= 10 else "soon" if urgency_score >= 7 else "routine"
            }
            
        except Exception as e:
            logger.error(f"Clinical urgency assessment failed: {e}")
            return {
                "urgency_level": "unknown",
                "urgency_score": 0,
                "recommended_action": "manual_review",
                "error": str(e)
            }
    
    async def get_doctor_patient_context(
        self,
        doctor_id: str,
        patient_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get context for Assistant Lysa about a doctor's patients.
        Includes assigned patients, pending alerts, and consent status.
        HIPAA-compliant with audit logging.
        """
        try:
            from app.database import SessionLocal
            from sqlalchemy import text
            
            AuditLogger.log_event(
                event_type=AuditEvent.PHI_ACCESSED,
                user_id=doctor_id,
                resource_type="doctor_patient_context",
                resource_id=patient_id or "all_patients",
                action="query",
                status="initiated",
                patient_id=patient_id,
                phi_accessed=patient_id is not None,
                phi_categories=["assignments", "alerts", "consent"]
            )
            
            db = SessionLocal()
            try:
                if patient_id:
                    assignment_result = db.execute(text("""
                        SELECT 
                            dpa.status,
                            dpa.access_level,
                            dpa.consent_given,
                            dpc.share_symptoms,
                            dpc.share_vitals,
                            dpc.share_medications,
                            dpc.allow_ai_analysis
                        FROM doctor_patient_assignments dpa
                        LEFT JOIN doctor_patient_consent dpc 
                            ON dpa.doctor_id = dpc.doctor_id 
                            AND dpa.patient_id = dpc.patient_id
                        WHERE dpa.doctor_id = :doctor_id
                        AND dpa.patient_id = :patient_id
                        AND dpa.status = 'active'
                    """), {"doctor_id": doctor_id, "patient_id": patient_id})
                    row = assignment_result.fetchone()
                    
                    if not row:
                        return {
                            "has_access": False,
                            "error": "No active assignment for this patient"
                        }
                    
                    permissions = {
                        "share_symptoms": row[3] if row[3] is not None else False,
                        "share_vitals": row[4] if row[4] is not None else False,
                        "share_medications": row[5] if row[5] is not None else False,
                        "allow_ai_analysis": row[6] if row[6] is not None else False
                    }
                    
                    alerts_result = db.execute(text("""
                        SELECT COUNT(*), 
                               SUM(CASE WHEN severity IN ('high', 'critical') THEN 1 ELSE 0 END)
                        FROM health_alerts
                        WHERE patient_id = :patient_id
                        AND status = 'active'
                    """), {"patient_id": patient_id})
                    alert_row = alerts_result.fetchone()
                    
                    return {
                        "has_access": True,
                        "access_level": row[1],
                        "consent_given": row[2],
                        "permissions": permissions,
                        "pending_alerts": alert_row[0] if alert_row else 0,
                        "high_priority_alerts": alert_row[1] if alert_row else 0
                    }
                
                else:
                    patients_result = db.execute(text("""
                        SELECT 
                            dpa.patient_id,
                            u.name as patient_name,
                            dpa.access_level,
                            COUNT(DISTINCT ha.id) as alert_count,
                            MAX(CASE WHEN ha.severity IN ('high', 'critical') THEN 1 ELSE 0 END) as has_critical
                        FROM doctor_patient_assignments dpa
                        JOIN users u ON u.id = dpa.patient_id
                        LEFT JOIN health_alerts ha ON ha.patient_id = dpa.patient_id AND ha.status = 'active'
                        WHERE dpa.doctor_id = :doctor_id
                        AND dpa.status = 'active'
                        GROUP BY dpa.patient_id, u.name, dpa.access_level
                        ORDER BY has_critical DESC, alert_count DESC
                    """), {"doctor_id": doctor_id})
                    
                    patients = [
                        {
                            "patient_id": row[0],
                            "patient_name": row[1],
                            "access_level": row[2],
                            "alert_count": row[3],
                            "has_critical_alerts": bool(row[4])
                        }
                        for row in patients_result.fetchall()
                    ]
                    
                    return {
                        "total_patients": len(patients),
                        "patients_with_alerts": sum(1 for p in patients if p["alert_count"] > 0),
                        "patients_with_critical": sum(1 for p in patients if p["has_critical_alerts"]),
                        "patients": patients[:20]
                    }
                    
            finally:
                db.close()
                
        except Exception as e:
            logger.error(f"Failed to get doctor-patient context: {e}")
            return {"error": str(e)}


# Singleton instance
agent_engine = AgentEngine()


async def get_agent_engine() -> AgentEngine:
    """
    Get initialized agent engine instance.
    Ensures memory service is injected if available.
    """
    if not agent_engine._initialized:
        await agent_engine.initialize()
    
    # Ensure memory service is always injected when available
    # Uses _memory_initialized flag to avoid repeated injection attempts
    if not agent_engine._memory_initialized:
        try:
            memory_service = await get_memory_service()
            agent_engine.set_memory_service(memory_service)
            logger.info("Memory service auto-injected via get_agent_engine()")
        except Exception as e:
            logger.warning(f"Could not inject memory service in get_agent_engine: {e}")
    
    return agent_engine
