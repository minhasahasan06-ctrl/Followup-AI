"""
Agent REST API Router
Handles CRUD operations for agent conversations, messages, and tasks
"""

import os
import json
import logging
from typing import Optional, List, AsyncGenerator
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.models.agent_models import (
    ConversationCreate, ConversationResponse,
    MessageCreate, MessageResponse,
    TaskCreate, TaskResponse,
    PresenceStatus, ApprovalDecision,
    MessageEnvelope, MessageParticipant, MessageType, ActorType,
    AgentDecisionContext
)
from app.services.agent_engine import get_agent_engine, AgentEngine
from app.services.message_router import get_message_router, MessageRouter
from app.services.memory_service import get_memory_service, MemoryService
from app.services.delivery_service import get_delivery_service, DeliveryService
from app.services.audit_logger import AuditLogger, AuditEvent

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/agent", tags=["Agent API"])


# Request/Response models
class SendMessageRequest(BaseModel):
    msgId: str
    conversationId: Optional[str] = None
    content: str
    messageType: str = "chat"
    toAgentId: Optional[str] = None


class ConversationListResponse(BaseModel):
    conversations: List[dict]
    total: int


class MessageListResponse(BaseModel):
    messages: List[dict]
    total: int


# Dependency for getting current user
async def get_current_user(request: Request) -> dict:
    """Get current user from request (session or token)"""
    # Check session
    if hasattr(request.state, "user"):
        return request.state.user

    # Check Authorization header
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header[7:]
        # In production, verify token
        # For dev mode, extract user info
        if os.getenv("DEV_MODE_SECRET"):
            return {"id": "dev-user", "role": "patient"}

    # Check X-User-ID header (dev mode)
    if os.getenv("DEV_MODE_SECRET"):
        user_id = request.headers.get("X-User-ID")
        user_role = request.headers.get("X-User-Role", "patient")
        if user_id:
            return {"id": user_id, "role": user_role}

    raise HTTPException(status_code=401, detail="Not authenticated")


# ==================== CONVERSATIONS ====================

@router.get("/conversations")
async def list_conversations(
    request: Request,
    status: Optional[str] = Query(None),
    limit: int = Query(20, le=100),
    offset: int = Query(0),
    user: dict = Depends(get_current_user)
):
    """List conversations for the current user"""
    user_id = user["id"]
    user_role = user["role"]
    agent_id = "lysa" if user_role == "doctor" else "clona"

    # In production, fetch from database
    # For now, return mock data structure
    conversations = [
        {
            "id": f"conv-{agent_id}-{user_id}",
            "participantType": "agent",
            "participantId": agent_id,
            "participantName": "Assistant Lysa" if agent_id == "lysa" else "Agent Clona",
            "lastMessage": None,
            "lastMessageAt": None,
            "unreadCount": 0,
            "isOnline": True
        }
    ]

    return {
        "conversations": conversations,
        "total": len(conversations)
    }


@router.post("/conversations")
async def create_conversation(
    request: Request,
    body: ConversationCreate,
    user: dict = Depends(get_current_user)
):
    """Create a new conversation"""
    user_id = user["id"]
    user_role = user["role"]

    # In production, create in database
    conversation_id = f"conv-{datetime.utcnow().timestamp()}"

    return {
        "id": conversation_id,
        "participant1Type": body.participant1_type,
        "participant1Id": body.participant1_id,
        "participant2Type": body.participant2_type,
        "participant2Id": body.participant2_id,
        "status": "active",
        "createdAt": datetime.utcnow().isoformat()
    }


@router.get("/conversations/{conversation_id}")
async def get_conversation(
    conversation_id: str,
    request: Request,
    user: dict = Depends(get_current_user)
):
    """Get conversation details"""
    # In production, fetch from database and verify access
    return {
        "id": conversation_id,
        "status": "active",
        "messageCount": 0,
        "unreadCount": 0
    }


# ==================== MESSAGES ====================

@router.get("/messages")
async def list_messages(
    request: Request,
    conversation_id: Optional[str] = Query(None),
    limit: int = Query(50, le=100),
    offset: int = Query(0),
    before: Optional[str] = Query(None),
    user: dict = Depends(get_current_user)
):
    """List messages for a conversation"""
    user_id = user["id"]

    # In production, fetch from database
    messages = []

    return {
        "messages": messages,
        "total": len(messages)
    }


@router.post("/messages")
async def send_message(
    request: Request,
    body: SendMessageRequest,
    user: dict = Depends(get_current_user),
    agent_engine: AgentEngine = Depends(get_agent_engine),
    message_router: MessageRouter = Depends(get_message_router),
    memory_service: MemoryService = Depends(get_memory_service)
):
    """Send a message to an agent"""
    user_id = user["id"]
    user_role = user["role"]
    agent_id = body.toAgentId or ("lysa" if user_role == "doctor" else "clona")

    try:
        # Store message
        message_id = body.msgId
        content = body.content
        conversation_id = body.conversationId or f"conv-{agent_id}-{user_id}"

        # In production, save to database and process through agent engine
        # For now, return acknowledgment

        return {
            "id": message_id,
            "msgId": message_id,
            "conversationId": conversation_id,
            "fromType": "user",
            "fromId": user_id,
            "content": content,
            "delivered": True,
            "createdAt": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Failed to send message: {e}")
        raise HTTPException(status_code=500, detail="Failed to send message")


@router.get("/messages/{message_id}")
async def get_message(
    message_id: str,
    request: Request,
    user: dict = Depends(get_current_user)
):
    """Get a specific message"""
    # In production, fetch from database
    return {"id": message_id, "content": "", "status": "not_found"}


@router.patch("/messages/{message_id}/read")
async def mark_message_read(
    message_id: str,
    request: Request,
    user: dict = Depends(get_current_user)
):
    """Mark a message as read"""
    return {
        "id": message_id,
        "readAt": datetime.utcnow().isoformat()
    }


# ==================== STREAMING CHAT ====================

class StreamingChatRequest(BaseModel):
    """Request model for streaming chat"""
    message: str
    conversationId: Optional[str] = None
    includeHealthContext: bool = True


@router.post("/chat/stream")
async def stream_chat(
    request: Request,
    body: StreamingChatRequest,
    user: dict = Depends(get_current_user),
    agent_engine: AgentEngine = Depends(get_agent_engine),
    memory_service: MemoryService = Depends(get_memory_service)
):
    """
    Stream real-time ChatGPT responses from Agent Clona or Assistant Lysa.
    Uses Server-Sent Events (SSE) for streaming with memory integration.
    """
    user_id = user["id"]
    user_role = user["role"]
    agent_id = "lysa" if user_role == "doctor" else "clona"
    conversation_id = body.conversationId or f"conv-{agent_id}-{user_id}"
    
    # Inject memory service into agent engine to ensure it's available
    agent_engine.set_memory_service(memory_service)
    
    async def generate_stream() -> AsyncGenerator[str, None]:
        full_response = ""
        extracted_symptoms = None
        
        try:
            # Build patient context if health context requested
            patient_context = None
            if body.includeHealthContext and user_role == "patient":
                patient_context = await agent_engine.get_health_context_summary(user_id)
                patient_context["patient_id"] = user_id
            
            # Get tools available for user role
            available_tools = agent_engine.get_available_tools(user_role)
            
            # Create message envelope using factory method
            message = MessageEnvelope.create(
                sender=MessageParticipant(type=ActorType.USER, id=user_id),
                to=[MessageParticipant(type=ActorType.AGENT, id=agent_id)],
                msg_type=MessageType.CHAT,
                payload={"content": body.message}
            )
            
            # Build initial context
            context = AgentDecisionContext(
                message=message,
                conversation_id=conversation_id,
                patient_context=patient_context,
                short_term_memory=[],
                long_term_memory=[],
                available_tools=available_tools
            )
            
            # Enrich context with short-term and long-term memories
            context = await agent_engine.enrich_context_with_memory(
                agent_id=agent_id,
                user_id=user_id,
                conversation_id=conversation_id,
                context=context,
                query=body.message  # Use message for semantic search
            )
            
            # Log the interaction start
            AuditLogger.log_event(
                event_type=AuditEvent.PHI_ACCESSED,
                user_id=user_id,
                resource_type="agent_chat",
                resource_id=conversation_id,
                action="stream_start",
                status="success",
                metadata={
                    "agent_id": agent_id, 
                    "has_health_context": bool(patient_context),
                    "short_term_memories": len(context.short_term_memory) if context.short_term_memory else 0,
                    "long_term_memories": len(context.long_term_memory) if context.long_term_memory else 0
                },
                patient_id=user_id if user_role == "patient" else None
            )
            
            # Stream the response and collect full text (skip_memory=True since we already enriched)
            async for chunk in agent_engine.process_message_streaming(
                agent_id, context, skip_memory=True, user_id=user_id
            ):
                # Collect content chunks for memory storage
                if chunk.get("type") == "content" and chunk.get("content"):
                    full_response += chunk["content"]
                yield f"data: {json.dumps(chunk)}\n\n"
            
            # If streaming to a patient with Agent Clona, extract symptoms
            if agent_id == "clona" and body.message:
                symptoms = await agent_engine.extract_symptoms_from_message(
                    body.message,
                    patient_context
                )
                extracted_symptoms = symptoms.get("symptoms", [])
                if symptoms.get("symptoms") or symptoms.get("urgency_level") not in ["routine", "unknown"]:
                    yield f"data: {json.dumps({'type': 'symptoms_extracted', 'data': symptoms})}\n\n"
            
            # Store interaction in both short-term and long-term memory
            await agent_engine.store_interaction_memory(
                agent_id=agent_id,
                user_id=user_id,
                conversation_id=conversation_id,
                user_message=body.message,
                agent_response=full_response,
                extracted_symptoms=extracted_symptoms,
                health_indicators=patient_context
            )
            
            yield f"data: {json.dumps({'type': 'stream_end'})}\n\n"
            
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@router.post("/chat/extract-symptoms")
async def extract_symptoms(
    request: Request,
    body: SendMessageRequest,
    user: dict = Depends(get_current_user),
    agent_engine: AgentEngine = Depends(get_agent_engine)
):
    """
    Extract symptoms and health indicators from a patient message.
    Used for AI-powered symptom tracking.
    """
    user_id = user["id"]
    user_role = user["role"]
    
    if user_role != "patient":
        raise HTTPException(status_code=403, detail="Symptom extraction is for patients only")
    
    symptoms = await agent_engine.extract_symptoms_from_message(
        body.content,
        {"patient_id": user_id}
    )
    
    return {
        "extracted": symptoms,
        "messageId": body.msgId,
        "processedAt": datetime.utcnow().isoformat()
    }


@router.get("/chat/health-context")
async def get_health_context(
    request: Request,
    user: dict = Depends(get_current_user),
    agent_engine: AgentEngine = Depends(get_agent_engine)
):
    """
    Get comprehensive health context for a patient.
    Used to inform Clona's responses with current health status.
    """
    user_id = user["id"]
    user_role = user["role"]
    
    if user_role != "patient":
        raise HTTPException(status_code=403, detail="Health context is for patients only")
    
    context = await agent_engine.get_health_context_summary(user_id)
    
    return {
        "context": context,
        "patientId": user_id,
        "fetchedAt": datetime.utcnow().isoformat()
    }


# ==================== TASKS ====================

class CreateTaskRequest(BaseModel):
    """Request to create a background task"""
    taskType: str
    toolName: Optional[str] = None
    parameters: Optional[dict] = None
    priority: str = "normal"
    scheduledAt: Optional[str] = None
    patientId: Optional[str] = None


@router.get("/tasks")
async def list_tasks(
    request: Request,
    status: Optional[str] = Query(None),
    task_type: Optional[str] = Query(None),
    limit: int = Query(20, le=100),
    offset: int = Query(0),
    user: dict = Depends(get_current_user)
):
    """List agent tasks for the current user"""
    from app.services.agent_tools.task_worker import get_task_worker, TaskStatus
    
    user_id = user["id"]
    
    try:
        worker = await get_task_worker()
        
        # Filter tasks by user
        tasks = []
        for task_id, task in worker.task_status.items():
            if task.user_id == user_id or task.patient_id == user_id:
                if status is None or task.status.value == status:
                    if task_type is None or task.task_type == task_type:
                        tasks.append(task.to_dict())
        
        # Apply pagination
        total = len(tasks)
        tasks = tasks[offset:offset + limit]
        
        return {
            "tasks": tasks,
            "total": total,
            "hasMore": offset + limit < total
        }
    except Exception as e:
        logger.error(f"Error listing tasks: {e}")
        return {"tasks": [], "total": 0, "hasMore": False}


@router.post("/tasks")
async def create_task(
    request: Request,
    body: CreateTaskRequest,
    user: dict = Depends(get_current_user)
):
    """Create a new background task"""
    from app.services.agent_tools.task_worker import get_task_worker, AgentTask, TaskPriority
    import uuid
    
    user_id = user["id"]
    user_role = user["role"]
    
    try:
        worker = await get_task_worker()
        
        # Map priority string to enum
        priority_map = {
            "low": TaskPriority.LOW,
            "normal": TaskPriority.NORMAL,
            "high": TaskPriority.HIGH,
            "critical": TaskPriority.CRITICAL
        }
        priority = priority_map.get(body.priority, TaskPriority.NORMAL)
        
        # Parse scheduled time if provided
        scheduled_at = None
        if body.scheduledAt:
            scheduled_at = datetime.fromisoformat(body.scheduledAt.replace("Z", "+00:00"))
        
        # Create task
        task = AgentTask(
            task_id=str(uuid.uuid4()),
            task_type=body.taskType,
            tool_name=body.toolName,
            parameters=body.parameters or {},
            user_id=user_id,
            patient_id=body.patientId or (user_id if user_role == "patient" else None),
            agent_id="clona" if user_role == "patient" else "lysa",
            priority=priority,
            scheduled_at=scheduled_at
        )
        
        # Enqueue the task
        task_id = await worker.enqueue_task(task)
        
        AuditLogger.log_event(
            event_type=AuditEvent.PHI_ACCESSED,
            user_id=user_id,
            resource_type="agent_task",
            resource_id=task_id,
            action="create",
            status="success",
            metadata={"task_type": body.taskType, "tool_name": body.toolName}
        )
        
        return {
            "taskId": task_id,
            "status": task.status.value,
            "createdAt": task.created_at.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error creating task: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tasks/{task_id}")
async def get_task(
    task_id: str,
    request: Request,
    user: dict = Depends(get_current_user)
):
    """Get task details and current status"""
    from app.services.agent_tools.task_worker import get_task_worker
    
    user_id = user["id"]
    
    try:
        worker = await get_task_worker()
        task = await worker.get_task_status(task_id)
        
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        
        # Verify access
        if task.user_id != user_id and task.patient_id != user_id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        return task.to_dict()
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting task: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/tasks/{task_id}")
async def cancel_task(
    task_id: str,
    request: Request,
    user: dict = Depends(get_current_user)
):
    """Cancel a pending or queued task"""
    from app.services.agent_tools.task_worker import get_task_worker
    
    user_id = user["id"]
    
    try:
        worker = await get_task_worker()
        task = await worker.get_task_status(task_id)
        
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        
        # Verify access
        if task.user_id != user_id and task.patient_id != user_id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        cancelled = await worker.cancel_task(task_id)
        
        if cancelled:
            AuditLogger.log_event(
                event_type=AuditEvent.PHI_ACCESSED,
                user_id=user_id,
                resource_type="agent_task",
                resource_id=task_id,
                action="cancel",
                status="success"
            )
            return {"taskId": task_id, "status": "cancelled"}
        else:
            raise HTTPException(status_code=400, detail="Task cannot be cancelled")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelling task: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== DELIVERY RECEIPTS ====================

class DeliveryReceiptRequest(BaseModel):
    messageIds: List[str]
    conversationId: Optional[str] = None


class ReadReceiptRequest(BaseModel):
    messageIds: List[str]
    conversationId: Optional[str] = None


def get_delivery_service_dep() -> DeliveryService:
    """Dependency injection for delivery service"""
    return get_delivery_service()


@router.post("/messages/delivered")
async def mark_messages_delivered(
    request: Request,
    body: DeliveryReceiptRequest,
    user: dict = Depends(get_current_user),
    delivery_service: DeliveryService = Depends(get_delivery_service_dep)
):
    """
    Mark messages as delivered to the user.
    
    Updates database, publishes to Redis stream, and notifies senders via WebSocket.
    """
    user_id = user["id"]
    
    result = await delivery_service.mark_messages_delivered(
        message_ids=body.messageIds,
        recipient_id=user_id,
        conversation_id=body.conversationId
    )
    
    if not result["success"]:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to mark messages as delivered: {result['errors']}"
        )
    
    return {
        "success": result["success"],
        "messageIds": result["message_ids"],
        "deliveredAt": result["delivered_at"],
        "deliveredCount": result["delivered_count"],
        "userId": user_id
    }


@router.post("/messages/read")
async def mark_messages_read(
    request: Request,
    body: ReadReceiptRequest,
    user: dict = Depends(get_current_user),
    delivery_service: DeliveryService = Depends(get_delivery_service_dep)
):
    """
    Mark messages as read by the user.
    
    Updates database, publishes to Redis stream, and notifies senders via WebSocket.
    """
    user_id = user["id"]
    
    result = await delivery_service.mark_messages_read(
        message_ids=body.messageIds,
        reader_id=user_id,
        conversation_id=body.conversationId
    )
    
    if not result["success"]:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to mark messages as read: {result['errors']}"
        )
    
    return {
        "success": result["success"],
        "messageIds": result["message_ids"],
        "readAt": result["read_at"],
        "readCount": result["read_count"],
        "userId": user_id
    }


@router.get("/messages/{message_id}/status")
async def get_message_status(
    message_id: str,
    request: Request,
    user: dict = Depends(get_current_user),
    delivery_service: DeliveryService = Depends(get_delivery_service_dep)
):
    """Get delivery status for a specific message"""
    status = await delivery_service.get_message_delivery_status(message_id)
    
    return {
        "messageId": status["message_id"],
        "sent": status["sent"],
        "sentAt": status["sent_at"].isoformat() if status["sent_at"] else None,
        "delivered": status["delivered"],
        "deliveredAt": status["delivered_at"].isoformat() if status["delivered_at"] else None,
        "read": status["read"],
        "readAt": status["read_at"].isoformat() if status["read_at"] else None
    }


@router.get("/messages/undelivered")
async def get_undelivered_messages(
    request: Request,
    conversation_id: Optional[str] = Query(None),
    user: dict = Depends(get_current_user),
    delivery_service: DeliveryService = Depends(get_delivery_service_dep)
):
    """Get all undelivered messages for the current user"""
    user_id = user["id"]
    
    messages = await delivery_service.get_undelivered_messages(
        recipient_id=user_id,
        conversation_id=conversation_id
    )
    
    return {
        "messages": messages,
        "total": len(messages),
        "userId": user_id
    }


# ==================== APPROVALS ====================

class ApprovalQueueItem(BaseModel):
    id: str
    requestType: str
    toolName: Optional[str]
    patientId: Optional[str]
    patientName: Optional[str]
    requestSummary: Optional[str]
    urgency: str
    riskLevel: Optional[str]
    status: str
    createdAt: str
    expiresAt: Optional[str]
    requestPayload: dict


class ApprovalDecisionRequest(BaseModel):
    decision: str  # 'approved', 'rejected', 'modified'
    notes: Optional[str] = None
    modifiedPayload: Optional[dict] = None


@router.get("/approvals/pending")
async def list_pending_approvals(
    request: Request,
    urgency: Optional[str] = Query(None),
    limit: int = Query(50, le=100),
    offset: int = Query(0),
    user: dict = Depends(get_current_user)
):
    """List pending approval requests for the current user (doctors only)"""
    if user["role"] != "doctor":
        raise HTTPException(status_code=403, detail="Only doctors can view approvals")
    
    user_id = user["id"]
    ip_address = request.client.host if request.client else None
    
    try:
        from app.database import SessionLocal
        from app.services.approval_repository import ApprovalRepository
        
        db = SessionLocal()
        try:
            repo = ApprovalRepository(db)
            result = repo.list_pending_approvals(
                approver_id=user_id,
                urgency=urgency,
                limit=limit,
                offset=offset,
                ip_address=ip_address
            )
            return result
            
        finally:
            db.close()
            
    except Exception as e:
        logger.error(f"Failed to list pending approvals: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch approvals")


@router.get("/approvals/{approval_id}")
async def get_approval_details(
    approval_id: str,
    request: Request,
    user: dict = Depends(get_current_user)
):
    """Get detailed information about an approval request"""
    if user["role"] != "doctor":
        raise HTTPException(status_code=403, detail="Only doctors can view approvals")
    
    user_id = user["id"]
    ip_address = request.client.host if request.client else None
    
    try:
        from app.database import SessionLocal
        from app.services.approval_repository import ApprovalRepository
        
        db = SessionLocal()
        try:
            repo = ApprovalRepository(db)
            result = repo.get_approval_by_id(
                approval_id=approval_id,
                approver_id=user_id,
                ip_address=ip_address
            )
            
            if not result:
                raise HTTPException(status_code=404, detail="Approval not found")
            
            return result
            
        finally:
            db.close()
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get approval details: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch approval")


@router.post("/approvals/{approval_id}/decide")
async def submit_approval_decision(
    approval_id: str,
    body: ApprovalDecisionRequest,
    request: Request,
    user: dict = Depends(get_current_user),
    agent_engine: AgentEngine = Depends(get_agent_engine)
):
    """Submit approval decision for a tool call"""
    if user["role"] != "doctor":
        raise HTTPException(status_code=403, detail="Only doctors can approve")
    
    user_id = user["id"]
    ip_address = request.client.host if request.client else None
    
    if body.decision not in ["approved", "rejected", "modified"]:
        raise HTTPException(status_code=400, detail="Invalid decision")
    
    try:
        from app.database import SessionLocal
        from app.services.approval_repository import ApprovalRepository
        from sqlalchemy import text
        import json
        
        db = SessionLocal()
        try:
            repo = ApprovalRepository(db)
            
            # Process decision using repository (prevents SQL injection)
            result = repo.process_decision(
                approval_id=approval_id,
                approver_id=user_id,
                decision=body.decision,
                notes=body.notes,
                modified_payload=body.modifiedPayload,
                ip_address=ip_address
            )
            
            if not result:
                raise HTTPException(status_code=404, detail="Approval not found or already processed")
            
            execution_result = None
            
            # If approved, execute the tool
            if body.decision in ["approved", "modified"]:
                payload = body.modifiedPayload if body.decision == "modified" else result.get("requestPayload", {})
                if payload is None:
                    payload = {}
                
                tool_name = result.get("toolName")
                if tool_name:
                    try:
                        execution_result = await agent_engine.execute_approved_tool(
                            tool_name=tool_name,
                            parameters=payload,
                            approval_id=approval_id,
                            approver_id=user_id,
                            user_id=result.get("requesterId", user_id),
                            user_role="doctor",
                            patient_id=result.get("patientId"),
                            doctor_id=user_id,
                            conversation_id=result.get("conversationId")
                        )
                        
                        # Update with execution result
                        if execution_result and hasattr(execution_result, 'result') and execution_result.result:
                            result_json = json.dumps(execution_result.result)
                            db.execute(text("""
                                UPDATE approval_queue SET
                                    execution_result = :result,
                                    executed_at = NOW()
                                WHERE id = :approval_id
                            """), {
                                "approval_id": approval_id,
                                "result": result_json
                            })
                            db.commit()
                        
                    except Exception as e:
                        logger.error(f"Tool execution after approval failed: {e}")
                        execution_result = None
            
            return {
                "id": approval_id,
                "decision": body.decision,
                "decidedBy": user_id,
                "decidedAt": datetime.utcnow().isoformat(),
                "notes": body.notes,
                "executionResult": execution_result.result if execution_result and hasattr(execution_result, 'result') else None,
                "success": True
            }
            
        finally:
            db.close()
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to submit approval decision: {e}")
        raise HTTPException(status_code=500, detail="Failed to process approval")


@router.post("/approvals")
async def create_approval_request(
    request: Request,
    body: dict,
    user: dict = Depends(get_current_user)
):
    """Create a new approval request (typically called by agents)"""
    ip_address = request.client.host if request.client else None
    
    try:
        from app.database import SessionLocal
        from app.services.approval_repository import ApprovalRepository
        
        db = SessionLocal()
        try:
            repo = ApprovalRepository(db)
            
            result = repo.create_approval(
                request_type=body.get("requestType", "tool_approval"),
                requester_id=body.get("requesterId", user["id"]),
                requester_type=body.get("requesterType", "agent"),
                request_payload=body.get("requestPayload", {}),
                approver_id=body.get("approverId"),
                approver_role=body.get("approverRole", "doctor"),
                patient_id=body.get("patientId"),
                conversation_id=body.get("conversationId"),
                message_id=body.get("messageId"),
                tool_execution_id=body.get("toolExecutionId"),
                tool_name=body.get("toolName"),
                request_summary=body.get("requestSummary"),
                urgency=body.get("urgency", "normal"),
                risk_level=body.get("riskLevel"),
                risk_factors=body.get("riskFactors"),
                expires_hours=24,
                ip_address=ip_address
            )
            
            return {
                "id": result["id"],
                "status": result["status"],
                "expiresAt": result["expiresAt"],
                "createdAt": result["createdAt"]
            }
            
        finally:
            db.close()
            
    except Exception as e:
        logger.error(f"Failed to create approval request: {e}")
        raise HTTPException(status_code=500, detail="Failed to create approval")


# ==================== PRESENCE ====================

@router.get("/presence/{user_id}")
async def get_user_presence(
    user_id: str,
    request: Request,
    user: dict = Depends(get_current_user),
    message_router: MessageRouter = Depends(get_message_router)
):
    """Get presence status for a user or agent"""
    # Check if it's an agent
    if user_id in ["clona", "lysa"]:
        return {
            "userId": user_id,
            "isOnline": True,
            "type": "agent"
        }

    # Get user presence
    presence = message_router.connection_manager.get_presence(user_id)
    if presence:
        return {
            "userId": user_id,
            "isOnline": presence.is_online,
            "lastSeenAt": presence.last_seen_at.isoformat() if presence.last_seen_at else None,
            "type": "user"
        }

    return {
        "userId": user_id,
        "isOnline": False,
        "type": "user"
    }


# ==================== AGENTS ====================

@router.get("/agents")
async def list_agents(
    request: Request,
    user: dict = Depends(get_current_user),
    agent_engine: AgentEngine = Depends(get_agent_engine)
):
    """List available agents"""
    user_role = user["role"]

    # Return agent appropriate for user role
    if user_role == "doctor":
        agent = agent_engine.get_agent("lysa")
        agents = [{
            "id": "lysa",
            "name": agent.name if agent else "Assistant Lysa",
            "description": agent.description if agent else "",
            "type": "assistant",
            "isOnline": True
        }]
    else:
        agent = agent_engine.get_agent("clona")
        agents = [{
            "id": "clona",
            "name": agent.name if agent else "Agent Clona",
            "description": agent.description if agent else "",
            "type": "companion",
            "isOnline": True
        }]

    return {"agents": agents}


@router.get("/agents/{agent_id}")
async def get_agent(
    agent_id: str,
    request: Request,
    user: dict = Depends(get_current_user),
    agent_engine: AgentEngine = Depends(get_agent_engine)
):
    """Get agent details"""
    agent = agent_engine.get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    return {
        "id": agent.id,
        "name": agent.name,
        "description": agent.description,
        "type": agent.agent_type,
        "targetRole": agent.target_role,
        "isOnline": True
    }


# ==================== PRESENCE ====================

class PresenceUpdateRequest(BaseModel):
    isOnline: bool = True
    activity: Optional[str] = None
    conversationId: Optional[str] = None


class HeartbeatRequest(BaseModel):
    conversationId: Optional[str] = None


@router.post("/presence")
async def update_presence(
    request: Request,
    presence_data: PresenceUpdateRequest,
    user: dict = Depends(get_current_user),
    message_router: MessageRouter = Depends(get_message_router)
):
    """Update user presence status"""
    user_id = user["id"]
    
    await message_router.update_presence(
        user_id=user_id,
        is_online=presence_data.isOnline,
        metadata={
            "activity": presence_data.activity,
            "conversation_id": presence_data.conversationId
        }
    )
    
    return {
        "success": True,
        "userId": user_id,
        "isOnline": presence_data.isOnline
    }


@router.post("/presence/heartbeat")
async def send_heartbeat(
    request: Request,
    heartbeat_data: HeartbeatRequest,
    user: dict = Depends(get_current_user),
    message_router: MessageRouter = Depends(get_message_router)
):
    """Send presence heartbeat to keep session alive"""
    user_id = user["id"]
    
    await message_router.heartbeat(user_id)
    
    return {
        "success": True,
        "userId": user_id,
        "timestamp": datetime.utcnow().isoformat()
    }


@router.get("/presence/online")
async def get_online_users(
    request: Request,
    user: dict = Depends(get_current_user),
    message_router: MessageRouter = Depends(get_message_router)
):
    """Get list of currently online users"""
    user_role = user["role"]
    
    # Only doctors can see list of online patients
    if user_role != "doctor":
        raise HTTPException(
            status_code=403,
            detail="Only doctors can view online users list"
        )
    
    online_users = await message_router.get_online_users()
    
    return {
        "users": online_users,
        "count": len(online_users)
    }


@router.get("/presence/user/{target_user_id}")
async def get_specific_user_presence(
    target_user_id: str,
    request: Request,
    user: dict = Depends(get_current_user),
    message_router: MessageRouter = Depends(get_message_router)
):
    """Get presence status for a specific user with detailed info"""
    # Get from connection manager first
    presence = message_router.connection_manager.get_presence(target_user_id)
    
    if presence:
        return {
            "userId": target_user_id,
            "isOnline": presence.is_online,
            "lastSeenAt": presence.last_seen_at.isoformat() if presence.last_seen_at else None,
            "activeConnections": presence.active_connections,
            "currentActivity": presence.current_activity
        }
    
    # Try Redis if available
    if message_router._redis_stream:
        try:
            redis_presence = await message_router._redis_stream.get_presence(target_user_id)
            if redis_presence:
                return {
                    "userId": target_user_id,
                    "isOnline": redis_presence.get("is_online", False),
                    "lastSeenAt": redis_presence.get("last_seen"),
                    "metadata": redis_presence.get("metadata", {})
                }
        except Exception as e:
            logger.error(f"Failed to get Redis presence: {e}")
    
    return {
        "userId": target_user_id,
        "isOnline": False,
        "lastSeenAt": None
    }


# ==================== PATIENT OVERVIEW FOR DOCTORS ====================

class PatientSummary(BaseModel):
    """Summary of a patient for doctor's overview panel"""
    id: str
    name: str
    email: Optional[str] = None
    assignedAt: Optional[str] = None
    accessLevel: Optional[str] = None
    riskScore: Optional[float] = None
    lastInteraction: Optional[str] = None
    activeAlerts: int = 0
    medicationCount: int = 0
    pendingFollowups: int = 0
    isOnline: bool = False


class PatientListResponse(BaseModel):
    """Response for patient list query"""
    patients: List[PatientSummary]
    total: int


@router.get("/patients")
async def get_assigned_patients(
    request: Request,
    status: Optional[str] = Query("active"),
    limit: int = Query(50, le=100),
    offset: int = Query(0),
    user: dict = Depends(get_current_user),
    message_router: MessageRouter = Depends(get_message_router)
):
    """
    Get list of patients assigned to the current doctor.
    Requires doctor role - enforces HIPAA access control.
    """
    if user["role"] != "doctor":
        raise HTTPException(
            status_code=403,
            detail="Only doctors can access patient lists"
        )
    
    doctor_id = user["id"]
    
    try:
        from sqlalchemy import text
        from app.database import get_db
        
        db = next(get_db())
        try:
            # Fetch assigned patients with summary data
            result = db.execute(text("""
                WITH patient_stats AS (
                    SELECT 
                        u.id as patient_id,
                        u.name as patient_name,
                        u.email as patient_email,
                        dpa.created_at as assigned_at,
                        dpa.access_level,
                        dpa.status as assignment_status,
                        COALESCE(
                            (SELECT composite_score 
                             FROM risk_scores rs 
                             WHERE rs.patient_id = u.id 
                             ORDER BY rs.calculated_at DESC LIMIT 1), 0
                        ) as risk_score,
                        COALESCE(
                            (SELECT COUNT(*) 
                             FROM health_alerts ha 
                             WHERE ha.patient_id = u.id 
                             AND ha.status = 'active' 
                             AND ha.created_at > NOW() - INTERVAL '24 hours'), 0
                        ) as active_alerts,
                        COALESCE(
                            (SELECT COUNT(*) 
                             FROM patient_medications pm 
                             WHERE pm.patient_id = u.id 
                             AND pm.status = 'active'), 0
                        ) as medication_count,
                        COALESCE(
                            (SELECT MAX(created_at) 
                             FROM agent_messages am 
                             WHERE (am.from_id = u.id OR am.to_id = u.id)
                             AND am.conversation_id IN (
                                 SELECT id FROM agent_conversations 
                                 WHERE doctor_id = :doctor_id 
                                 AND patient_id = u.id
                             )), NULL
                        ) as last_interaction
                    FROM users u
                    INNER JOIN doctor_patient_assignments dpa ON dpa.patient_id = u.id
                    WHERE dpa.doctor_id = :doctor_id
                    AND dpa.status = :status
                )
                SELECT * FROM patient_stats
                ORDER BY risk_score DESC, active_alerts DESC
                LIMIT :limit OFFSET :offset
            """), {
                "doctor_id": doctor_id,
                "status": status,
                "limit": limit,
                "offset": offset
            })
            
            patients = []
            for row in result.fetchall():
                patient_id = row[0]
                # Check if patient is online
                is_online = False
                presence = message_router.connection_manager.get_presence(patient_id)
                if presence:
                    is_online = presence.is_online
                
                patients.append(PatientSummary(
                    id=patient_id,
                    name=row[1] or "Unknown",
                    email=row[2],
                    assignedAt=row[3].isoformat() if row[3] else None,
                    accessLevel=row[4],
                    riskScore=float(row[6]) if row[6] else None,
                    activeAlerts=int(row[7]) if row[7] else 0,
                    medicationCount=int(row[8]) if row[8] else 0,
                    lastInteraction=row[9].isoformat() if row[9] else None,
                    isOnline=is_online
                ))
            
            # Get total count
            count_result = db.execute(text("""
                SELECT COUNT(*) FROM doctor_patient_assignments
                WHERE doctor_id = :doctor_id AND status = :status
            """), {"doctor_id": doctor_id, "status": status})
            total = count_result.scalar() or 0
            
            logger.info(f"Doctor {doctor_id} fetched {len(patients)} patients")
            
            return PatientListResponse(patients=patients, total=total)
            
        finally:
            db.close()
            
    except Exception as e:
        logger.error(f"Failed to fetch assigned patients: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch patients: {str(e)}"
        )


@router.get("/patients/{patient_id}/overview")
async def get_patient_overview(
    patient_id: str,
    request: Request,
    user: dict = Depends(get_current_user),
    agent_engine: AgentEngine = Depends(get_agent_engine),
    memory_service: MemoryService = Depends(get_memory_service)
):
    """
    Get comprehensive overview of a specific patient.
    Includes health data, medications, alerts, recent symptoms, and Lysa conversation history.
    Requires verified doctor-patient assignment with appropriate access level.
    """
    if user["role"] != "doctor":
        raise HTTPException(
            status_code=403,
            detail="Only doctors can access patient overviews"
        )
    
    doctor_id = user["id"]
    
    try:
        from sqlalchemy import text
        from app.database import get_db
        
        db = next(get_db())
        try:
            # Verify doctor-patient assignment exists with sufficient access
            assignment = db.execute(text("""
                SELECT id, access_level, status, consent_type, expires_at
                FROM doctor_patient_assignments
                WHERE doctor_id = :doctor_id 
                AND patient_id = :patient_id 
                AND status = 'active'
            """), {"doctor_id": doctor_id, "patient_id": patient_id}).fetchone()
            
            if not assignment:
                # Log unauthorized access attempt
                AuditLogger.log_event(
                    event_type="patient_access_denied",
                    user_id=doctor_id,
                    resource_type="patient",
                    resource_id=patient_id,
                    action="unauthorized_patient_access_attempt",
                    status="denied",
                    phi_accessed=False,
                    metadata={"reason": "No active assignment"}
                )
                raise HTTPException(
                    status_code=403,
                    detail="You do not have authorization to view this patient's records"
                )
            
            access_level = assignment[1]
            
            # Log successful access for HIPAA compliance
            AuditLogger.log_phi_access(
                user_id=doctor_id,
                patient_id=patient_id,
                resource_type="patient_overview",
                resource_id=patient_id,
                action="view",
                status="success",
                phi_categories=["demographics", "health_records", "medications"],
                access_reason=f"Doctor viewing assigned patient overview (access_level: {access_level})"
            )
            
            # Fetch patient basic info
            patient_info = db.execute(text("""
                SELECT id, name, email, date_of_birth, phone_number, created_at
                FROM users
                WHERE id = :patient_id
            """), {"patient_id": patient_id}).fetchone()
            
            if not patient_info:
                raise HTTPException(status_code=404, detail="Patient not found")
            
            # Fetch health context using agent engine
            health_context = await agent_engine.get_health_context_summary(patient_id)
            
            # Fetch recent daily followups
            followups = db.execute(text("""
                SELECT date, overall_status, energy_level, pain_level, 
                       symptoms_noted, notes, completed_at
                FROM daily_followups
                WHERE patient_id = :patient_id
                ORDER BY date DESC
                LIMIT 7
            """), {"patient_id": patient_id}).fetchall()
            
            daily_followups = [
                {
                    "date": row[0].isoformat() if row[0] else None,
                    "overallStatus": row[1],
                    "energyLevel": row[2],
                    "painLevel": row[3],
                    "symptomsNoted": row[4],
                    "notes": row[5],
                    "completedAt": row[6].isoformat() if row[6] else None
                }
                for row in followups
            ]
            
            # Fetch Lysa conversation history with this patient
            conversations = db.execute(text("""
                SELECT ac.id, ac.title, ac.created_at, ac.updated_at,
                       (SELECT COUNT(*) FROM agent_messages am WHERE am.conversation_id = ac.id) as message_count,
                       (SELECT content FROM agent_messages am 
                        WHERE am.conversation_id = ac.id 
                        ORDER BY am.created_at DESC LIMIT 1) as last_message
                FROM agent_conversations ac
                WHERE ac.doctor_id = :doctor_id
                AND ac.patient_id = :patient_id
                AND ac.conversation_type IN ('doctor_lysa', 'doctor_clona', 'patient_doctor')
                ORDER BY ac.updated_at DESC
                LIMIT 10
            """), {"doctor_id": doctor_id, "patient_id": patient_id}).fetchall()
            
            conversation_history = [
                {
                    "id": row[0],
                    "title": row[1] or "Untitled Conversation",
                    "createdAt": row[2].isoformat() if row[2] else None,
                    "updatedAt": row[3].isoformat() if row[3] else None,
                    "messageCount": row[4] or 0,
                    "lastMessage": row[5][:100] + "..." if row[5] and len(row[5]) > 100 else row[5]
                }
                for row in conversations
            ]
            
            # Fetch long-term memories from memory service if available
            long_term_insights = []
            if memory_service:
                try:
                    memories = await memory_service.search_long_term(
                        agent_id="lysa",
                        patient_id=patient_id,
                        query="health concerns symptoms medications",
                        limit=5
                    )
                    for m in memories:
                        # Handle both dict and object access patterns
                        if isinstance(m, dict):
                            content = m.get("content", "")
                            memory_type = m.get("memory_type", "unknown")
                            created_at = m.get("created_at")
                            importance = m.get("importance", 0.5)
                        else:
                            content = getattr(m, "content", "")
                            memory_type = getattr(m, "memory_type", "unknown")
                            created_at = getattr(m, "created_at", None)
                            importance = getattr(m, "importance", 0.5)
                        
                        long_term_insights.append({
                            "content": content[:200] + "..." if len(content) > 200 else content,
                            "type": memory_type,
                            "createdAt": created_at.isoformat() if created_at and hasattr(created_at, 'isoformat') else None,
                            "importance": importance
                        })
                except Exception as e:
                    logger.warning(f"Could not fetch long-term memories: {e}")
            
            return {
                "patient": {
                    "id": patient_info[0],
                    "name": patient_info[1],
                    "email": patient_info[2],
                    "dateOfBirth": patient_info[3].isoformat() if patient_info[3] else None,
                    "phone": patient_info[4],
                    "memberSince": patient_info[5].isoformat() if patient_info[5] else None
                },
                "assignment": {
                    "accessLevel": access_level,
                    "consentType": assignment[3],
                    "expiresAt": assignment[4].isoformat() if assignment[4] else None
                },
                "healthContext": health_context,
                "dailyFollowups": daily_followups,
                "conversationHistory": conversation_history,
                "longTermInsights": long_term_insights
            }
            
        finally:
            db.close()
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to fetch patient overview: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch patient overview: {str(e)}"
        )


@router.get("/patients/{patient_id}/conversations")
async def get_patient_conversations(
    patient_id: str,
    request: Request,
    limit: int = Query(20, le=100),
    offset: int = Query(0),
    user: dict = Depends(get_current_user)
):
    """
    Get Lysa conversation history with a specific patient.
    Returns all conversations where the doctor discussed or reviewed this patient.
    """
    if user["role"] != "doctor":
        raise HTTPException(
            status_code=403,
            detail="Only doctors can access conversation history"
        )
    
    doctor_id = user["id"]
    
    try:
        from sqlalchemy import text
        from app.database import get_db
        
        db = next(get_db())
        try:
            # Verify assignment
            has_access = db.execute(text("""
                SELECT 1 FROM doctor_patient_assignments
                WHERE doctor_id = :doctor_id 
                AND patient_id = :patient_id 
                AND status = 'active'
            """), {"doctor_id": doctor_id, "patient_id": patient_id}).fetchone()
            
            if not has_access:
                raise HTTPException(
                    status_code=403,
                    detail="No authorization to view conversations for this patient"
                )
            
            # Fetch conversations
            result = db.execute(text("""
                SELECT 
                    ac.id,
                    ac.title,
                    ac.conversation_type,
                    ac.created_at,
                    ac.updated_at,
                    ac.status,
                    (SELECT COUNT(*) FROM agent_messages am WHERE am.conversation_id = ac.id) as message_count
                FROM agent_conversations ac
                WHERE ac.doctor_id = :doctor_id
                AND ac.patient_id = :patient_id
                ORDER BY ac.updated_at DESC
                LIMIT :limit OFFSET :offset
            """), {
                "doctor_id": doctor_id,
                "patient_id": patient_id,
                "limit": limit,
                "offset": offset
            })
            
            conversations = [
                {
                    "id": row[0],
                    "title": row[1] or "Untitled",
                    "conversationType": row[2],
                    "createdAt": row[3].isoformat() if row[3] else None,
                    "updatedAt": row[4].isoformat() if row[4] else None,
                    "status": row[5],
                    "messageCount": row[6] or 0
                }
                for row in result.fetchall()
            ]
            
            # Get total count
            count = db.execute(text("""
                SELECT COUNT(*) FROM agent_conversations
                WHERE doctor_id = :doctor_id AND patient_id = :patient_id
            """), {"doctor_id": doctor_id, "patient_id": patient_id}).scalar() or 0
            
            return {
                "conversations": conversations,
                "total": count
            }
            
        finally:
            db.close()
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to fetch patient conversations: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch conversations: {str(e)}"
        )


@router.get("/conversations/{conversation_id}/messages")
async def get_conversation_messages(
    conversation_id: str,
    request: Request,
    limit: int = Query(50, le=200),
    offset: int = Query(0),
    user: dict = Depends(get_current_user)
):
    """
    Get messages for a specific conversation with sender role info.
    Returns messages with clear AI vs Human identification.
    """
    if user["role"] != "doctor":
        raise HTTPException(
            status_code=403,
            detail="Only doctors can access conversation messages"
        )
    
    doctor_id = user["id"]
    
    try:
        from sqlalchemy import text
        from app.database import get_db
        from app.services.audit_logger import AuditLogger
        
        db = next(get_db())
        try:
            # Verify conversation belongs to this doctor
            conv_info = db.execute(text("""
                SELECT ac.id, ac.doctor_id, ac.patient_id, ac.title, ac.conversation_type
                FROM agent_conversations ac
                WHERE ac.id = :conversation_id
            """), {"conversation_id": conversation_id}).fetchone()
            
            if not conv_info:
                raise HTTPException(status_code=404, detail="Conversation not found")
            
            if conv_info[1] != doctor_id:
                raise HTTPException(
                    status_code=403,
                    detail="Not authorized to view this conversation"
                )
            
            patient_id = conv_info[2]
            
            # Log PHI access
            AuditLogger.log_phi_access(
                user_id=doctor_id,
                patient_id=patient_id,
                resource_type="conversation",
                resource_id=conversation_id,
                action="VIEW_CONVERSATION_MESSAGES",
                phi_categories=["conversation_history", "health_information"],
                access_reason="Doctor viewing patient conversation history"
            )
            
            # Fetch messages with sender info
            messages = db.execute(text("""
                SELECT 
                    am.id,
                    am.msg_id,
                    am.from_type,
                    am.from_id,
                    am.sender_role,
                    am.sender_name,
                    am.sender_avatar,
                    am.message_type,
                    am.content,
                    am.tool_name,
                    am.tool_status,
                    am.requires_approval,
                    am.approval_status,
                    am.contains_phi,
                    am.created_at
                FROM agent_messages am
                WHERE am.conversation_id = :conversation_id
                ORDER BY am.created_at ASC
                LIMIT :limit OFFSET :offset
            """), {
                "conversation_id": conversation_id,
                "limit": limit,
                "offset": offset
            }).fetchall()
            
            message_list = []
            for row in messages:
                sender_role = row[4] or "unknown"
                
                # Determine if sender is AI or Human
                is_ai = sender_role in ("clona", "lysa", "system")
                
                # Get display name based on role
                if sender_role == "clona":
                    display_name = "Agent Clona"
                    display_subtitle = "Patient AI Assistant"
                elif sender_role == "lysa":
                    display_name = "Assistant Lysa"
                    display_subtitle = "Doctor AI Assistant"
                elif sender_role == "doctor":
                    display_name = row[5] or "Doctor"
                    display_subtitle = "Healthcare Provider"
                elif sender_role == "patient":
                    display_name = row[5] or "Patient"
                    display_subtitle = "Patient"
                elif sender_role == "system":
                    display_name = "System"
                    display_subtitle = "Automated Message"
                else:
                    display_name = row[5] or "Unknown"
                    display_subtitle = "Unknown Sender"
                
                message_list.append({
                    "id": row[0],
                    "msgId": row[1],
                    "fromType": row[2],
                    "fromId": row[3],
                    "senderRole": sender_role,
                    "senderName": display_name,
                    "senderSubtitle": display_subtitle,
                    "senderAvatar": row[6],
                    "isAI": is_ai,
                    "isHuman": not is_ai,
                    "messageType": row[7],
                    "content": row[8],
                    "toolName": row[9],
                    "toolStatus": row[10],
                    "requiresApproval": row[11],
                    "approvalStatus": row[12],
                    "containsPhi": row[13],
                    "createdAt": row[14].isoformat() if row[14] else None
                })
            
            # Get total count
            total = db.execute(text("""
                SELECT COUNT(*) FROM agent_messages
                WHERE conversation_id = :conversation_id
            """), {"conversation_id": conversation_id}).scalar() or 0
            
            return {
                "conversationId": conversation_id,
                "title": conv_info[3] or "Untitled Conversation",
                "conversationType": conv_info[4],
                "patientId": patient_id,
                "messages": message_list,
                "total": total
            }
            
        finally:
            db.close()
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to fetch conversation messages: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch messages: {str(e)}"
        )


# ==================== HEALTH ====================

@router.get("/health")
async def agent_health():
    """Health check for agent services"""
    return {
        "status": "healthy",
        "services": {
            "agentEngine": "ok",
            "messageRouter": "ok",
            "memoryService": "ok"
        },
        "agents": ["clona", "lysa"]
    }
