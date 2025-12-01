"""
Multi-Agent Communication System Models
Agent Clona (Patient) and Assistant Lysa (Doctor) communication models
"""

from typing import Optional, List, Dict, Any, Literal
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum
import uuid


class ActorType(str, Enum):
    AGENT = "agent"
    USER = "user"
    SYSTEM = "system"


class MessageType(str, Enum):
    CHAT = "chat"
    COMMAND = "command"
    EVENT = "event"
    TOOL_CALL = "tool_call"
    ACK = "ack"


class ToolStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PENDING_APPROVAL = "pending_approval"


class ApprovalStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class TaskStatus(str, Enum):
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class MemoryType(str, Enum):
    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"


# Message Envelope Protocol
class MessageParticipant(BaseModel):
    type: ActorType
    id: str


class MessageEnvelope(BaseModel):
    """Standard message envelope for all agent communications"""
    msg_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    sender: MessageParticipant = Field(alias="from")  # Use 'sender' internally, 'from' for JSON
    to: List[MessageParticipant]
    type: MessageType
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    payload: Optional[Dict[str, Any]] = None

    class Config:
        populate_by_name = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

    @classmethod
    def create(
        cls,
        sender: MessageParticipant,
        to: List[MessageParticipant],
        msg_type: MessageType,
        payload: Optional[Dict[str, Any]] = None,
        msg_id: Optional[str] = None
    ) -> "MessageEnvelope":
        """Factory method for creating message envelopes"""
        return cls(
            msg_id=msg_id or str(uuid.uuid4()),
            sender=sender,
            to=to,
            type=msg_type,
            timestamp=datetime.utcnow(),
            payload=payload
        )


# Agent Models
class AgentPersona(BaseModel):
    system_prompt: str
    personality: str
    tone: str
    specializations: List[str] = []
    constraints: List[str] = []


class AgentMemoryPolicy(BaseModel):
    short_term_ttl_hours: int = 2
    long_term_enabled: bool = True
    vectorization_enabled: bool = True
    max_memory_per_patient: int = 1000
    summarization_threshold: int = 10


class AgentConfig(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    agent_type: str  # 'assistant' (Lysa), 'companion' (Clona)
    target_role: str  # 'doctor', 'patient', 'all'
    persona: AgentPersona
    memory_policy: AgentMemoryPolicy
    openai_assistant_id: Optional[str] = None
    openai_model: str = "gpt-4o"
    is_active: bool = True
    version: int = 1


# Tool Models
class ToolConfig(BaseModel):
    endpoint: Optional[str] = None
    method: Optional[str] = None
    parameters_schema: Optional[Dict[str, Any]] = None
    response_schema: Optional[Dict[str, Any]] = None
    timeout: int = 30


class AgentToolDefinition(BaseModel):
    id: str
    name: str
    display_name: str
    description: Optional[str] = None
    tool_type: str  # 'calendar', 'messaging', 'ehr_fetch', 'prescription_draft', 'lab_fetch', 'imaging_linker'
    config: ToolConfig
    parameters_schema: Optional[Dict[str, Any]] = None
    required_permissions: List[str] = []
    allowed_roles: List[str] = []
    requires_approval: bool = False
    approval_role: Optional[str] = None
    is_enabled: bool = True
    version: int = 1


# Conversation Models
class ConversationParticipant(BaseModel):
    type: str  # 'user', 'agent'
    id: str
    name: Optional[str] = None


class ConversationCreate(BaseModel):
    participant1_type: str
    participant1_id: str
    participant2_type: str
    participant2_id: str
    patient_id: Optional[str] = None
    doctor_id: Optional[str] = None
    title: Optional[str] = None


class ConversationResponse(BaseModel):
    id: str
    participant1_type: str
    participant1_id: str
    participant2_type: str
    participant2_id: str
    patient_id: Optional[str] = None
    doctor_id: Optional[str] = None
    title: Optional[str] = None
    status: str
    message_count: int
    unread_count_1: int
    unread_count_2: int
    last_message_at: Optional[datetime] = None
    last_message_preview: Optional[str] = None
    openai_thread_id: Optional[str] = None
    created_at: datetime
    updated_at: datetime


# Message Models
class MessageCreate(BaseModel):
    conversation_id: str
    msg_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    from_type: ActorType
    from_id: str
    to_json: List[Dict[str, str]]
    message_type: MessageType
    content: Optional[str] = None
    payload_json: Optional[Dict[str, Any]] = None
    tool_call_id: Optional[str] = None
    tool_name: Optional[str] = None
    tool_input: Optional[Dict[str, Any]] = None
    requires_approval: bool = False


class MessageResponse(BaseModel):
    id: str
    msg_id: str
    conversation_id: str
    from_type: str
    from_id: str
    to_json: List[Dict[str, str]]
    message_type: str
    content: Optional[str] = None
    payload_json: Optional[Dict[str, Any]] = None
    tool_call_id: Optional[str] = None
    tool_name: Optional[str] = None
    tool_input: Optional[Dict[str, Any]] = None
    tool_output: Optional[Dict[str, Any]] = None
    tool_status: Optional[str] = None
    requires_approval: bool
    approval_status: Optional[str] = None
    approved_by: Optional[str] = None
    approved_at: Optional[datetime] = None
    delivered: bool
    delivered_at: Optional[datetime] = None
    read_at: Optional[datetime] = None
    contains_phi: bool
    created_at: datetime


# Task Models
class TaskCreate(BaseModel):
    agent_id: str
    user_id: Optional[str] = None
    conversation_id: Optional[str] = None
    message_id: Optional[str] = None
    task_type: str  # 'tool_execution', 'scheduled_checkin', 'medication_reminder', 'workflow', 'inference'
    task_name: Optional[str] = None
    scheduled_at: Optional[datetime] = None
    recurring_pattern: Optional[str] = None  # Cron expression
    timezone: str = "UTC"
    priority: int = 5
    input_payload: Optional[Dict[str, Any]] = None
    max_attempts: int = 3


class TaskResponse(BaseModel):
    id: str
    agent_id: str
    user_id: Optional[str] = None
    conversation_id: Optional[str] = None
    message_id: Optional[str] = None
    task_type: str
    task_name: Optional[str] = None
    scheduled_at: Optional[datetime] = None
    recurring_pattern: Optional[str] = None
    timezone: str
    status: str
    priority: int
    input_payload: Optional[Dict[str, Any]] = None
    output_result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    attempts: int
    max_attempts: int
    worker_id: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime


# Memory Models
class MemoryCreate(BaseModel):
    agent_id: str
    patient_id: Optional[str] = None
    user_id: Optional[str] = None
    conversation_id: Optional[str] = None
    memory_type: MemoryType
    storage_type: str  # 'redis', 'postgres', 'vector'
    content: str
    summary: Optional[str] = None
    embedding: Optional[List[float]] = None
    embedding_model: Optional[str] = None
    source_type: Optional[str] = None
    source_id: Optional[str] = None
    importance: float = 0.5
    expires_at: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None


class MemoryResponse(BaseModel):
    id: str
    agent_id: str
    patient_id: Optional[str] = None
    user_id: Optional[str] = None
    conversation_id: Optional[str] = None
    memory_type: str
    storage_type: str
    content: str
    summary: Optional[str] = None
    source_type: Optional[str] = None
    source_id: Optional[str] = None
    importance: float
    access_count: int
    last_accessed_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None
    created_at: datetime


# Audit Log Models
class AuditLogCreate(BaseModel):
    actor_type: str  # 'user', 'agent', 'system', 'worker'
    actor_id: str
    actor_role: Optional[str] = None
    action: str
    action_category: str  # 'communication', 'data_access', 'clinical_action', 'system'
    object_type: str
    object_id: str
    patient_id: Optional[str] = None
    conversation_id: Optional[str] = None
    message_id: Optional[str] = None
    phi_accessed: bool = False
    phi_categories: Optional[List[str]] = None
    access_reason: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    session_id: Optional[str] = None
    success: bool = True
    error_code: Optional[str] = None
    error_message: Optional[str] = None


# Presence Models
class PresenceUpdate(BaseModel):
    user_id: str
    is_online: bool
    current_activity: Optional[str] = None
    current_conversation_id: Optional[str] = None
    device_info: Optional[Dict[str, Any]] = None


class PresenceStatus(BaseModel):
    user_id: str
    is_online: bool
    last_seen_at: Optional[datetime] = None
    active_connections: int
    current_activity: Optional[str] = None
    current_conversation_id: Optional[str] = None


# WebSocket Models
class WebSocketMessage(BaseModel):
    type: str  # 'auth', 'message', 'typing', 'presence', 'ack', 'error'
    payload: Dict[str, Any]


class WebSocketAuthPayload(BaseModel):
    token: str
    user_id: str
    role: str


class TypingIndicator(BaseModel):
    conversation_id: str
    user_id: str
    is_typing: bool


# Tool Call Models
class ToolCallRequest(BaseModel):
    tool_name: str
    parameters: Dict[str, Any]
    conversation_id: str
    message_id: str
    requires_approval: bool = False


class ToolCallResult(BaseModel):
    tool_call_id: str
    tool_name: str
    status: ToolStatus
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    execution_time_ms: Optional[int] = None


# Human-in-the-Loop Approval Models
class ApprovalRequest(BaseModel):
    message_id: str
    tool_name: str
    tool_input: Dict[str, Any]
    requested_by: str  # agent_id
    patient_id: Optional[str] = None
    reason: str
    urgency: str = "normal"  # 'low', 'normal', 'high', 'urgent'


class ApprovalDecision(BaseModel):
    message_id: str
    approved: bool
    approved_by: str
    notes: Optional[str] = None


# Agent Decision Loop Models
class AgentDecisionContext(BaseModel):
    conversation_id: str
    message: MessageEnvelope
    short_term_memory: List[Dict[str, Any]] = []
    long_term_memory: List[Dict[str, Any]] = []
    patient_context: Optional[Dict[str, Any]] = None
    available_tools: List[AgentToolDefinition] = []


class AgentDecisionResult(BaseModel):
    response_message: Optional[str] = None
    tool_calls: List[ToolCallRequest] = []
    memory_updates: List[MemoryCreate] = []
    requires_human_confirmation: bool = False
    confirmation_details: Optional[ApprovalRequest] = None
