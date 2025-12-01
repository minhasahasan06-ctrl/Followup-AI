"""
SQLAlchemy models for approval queue and tool executions
HIPAA-compliant models for human-in-the-loop approval workflows

IMPORTANT: These models must match the Drizzle schema in shared/schema.ts exactly.
Any changes to the Drizzle schema should be reflected here.
"""

from sqlalchemy import Column, String, DateTime, Text, Boolean, Integer, JSON
from sqlalchemy.sql import func
from app.database import Base


class ApprovalQueue(Base):
    """
    Approval queue for tools requiring human-in-the-loop approval
    Matches: shared/schema.ts -> approvalQueue
    """
    __tablename__ = "approval_queue"
    
    id = Column(String, primary_key=True)
    
    # Request details
    request_type = Column(String, nullable=False, index=True)
    requester_id = Column(String, nullable=False, index=True)
    requester_type = Column(String, nullable=False)  # 'agent', 'user', 'system'
    
    # Approval target
    approver_id = Column(String, nullable=True, index=True)  # Specific approver if assigned
    approver_role = Column(String, nullable=False, default="doctor")  # Role that can approve
    
    # Context
    patient_id = Column(String, nullable=True, index=True)
    conversation_id = Column(String, nullable=True, index=True)
    message_id = Column(String, nullable=True)
    tool_execution_id = Column(String, nullable=True, index=True)
    
    # Request payload
    tool_name = Column(String, nullable=True)
    request_payload = Column(JSON, nullable=False)
    request_summary = Column(Text, nullable=True)  # Human-readable summary
    
    # Risk assessment
    urgency = Column(String, nullable=False, default="normal")  # 'low', 'normal', 'high', 'urgent'
    risk_level = Column(String, nullable=True)  # 'low', 'medium', 'high'
    risk_factors = Column(JSON, nullable=True)  # List of risk factor strings
    
    # Status tracking
    status = Column(String, nullable=False, default="pending", index=True)  # 'pending', 'approved', 'rejected', 'expired', 'cancelled'
    
    # Decision
    decision = Column(String, nullable=True)  # 'approved', 'rejected', 'modified'
    decision_by = Column(String, nullable=True)
    decision_at = Column(DateTime, nullable=True)
    decision_notes = Column(Text, nullable=True)
    modified_payload = Column(JSON, nullable=True)  # If modified before approval
    
    # Timeout
    expires_at = Column(DateTime, nullable=True)
    reminder_sent_at = Column(DateTime, nullable=True)
    escalated_at = Column(DateTime, nullable=True)
    escalated_to = Column(String, nullable=True)
    
    # Execution result
    execution_result = Column(JSON, nullable=True)
    executed_at = Column(DateTime, nullable=True)
    
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())


class ToolExecution(Base):
    """
    Record of tool executions with status tracking
    Matches: shared/schema.ts -> toolExecutions
    """
    __tablename__ = "tool_executions"
    
    id = Column(String, primary_key=True)
    
    # Execution context
    agent_id = Column(String, nullable=False, index=True)
    user_id = Column(String, nullable=False, index=True)
    conversation_id = Column(String, nullable=True, index=True)
    message_id = Column(String, nullable=True)
    
    # Tool info
    tool_name = Column(String, nullable=False, index=True)
    tool_version = Column(Integer, nullable=True, default=1)
    
    # Input/Output
    input_parameters = Column(JSON, nullable=False)
    output_result = Column(JSON, nullable=True)
    
    # Status tracking
    status = Column(String, nullable=False, default="pending", index=True)  # 'pending', 'running', 'completed', 'failed', 'pending_approval', 'approved', 'rejected'
    error_message = Column(Text, nullable=True)
    error_code = Column(String, nullable=True)
    
    # Performance metrics
    execution_time_ms = Column(Integer, nullable=True)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    
    # Patient context (for consent verification)
    patient_id = Column(String, nullable=True, index=True)
    doctor_id = Column(String, nullable=True, index=True)
    
    # PHI tracking
    phi_accessed = Column(Boolean, default=False)
    phi_categories = Column(JSON, nullable=True)
    
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
