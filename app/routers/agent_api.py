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
    Uses Server-Sent Events (SSE) for streaming.
    """
    user_id = user["id"]
    user_role = user["role"]
    agent_id = "lysa" if user_role == "doctor" else "clona"
    conversation_id = body.conversationId or f"conv-{agent_id}-{user_id}"
    
    async def generate_stream() -> AsyncGenerator[str, None]:
        try:
            # Build patient context if health context requested
            patient_context = None
            if body.includeHealthContext and user_role == "patient":
                patient_context = await agent_engine.get_health_context_summary(user_id)
                patient_context["patient_id"] = user_id
            
            # Get short-term memory for context
            short_term_memory = await memory_service.get_short_term_memories(
                agent_id=agent_id,
                user_id=user_id,
                conversation_id=conversation_id,
                limit=5
            )
            
            # Get tools available for user role
            available_tools = agent_engine.get_available_tools(user_role)
            
            # Create message envelope using factory method
            message = MessageEnvelope.create(
                sender=MessageParticipant(type=ActorType.USER, id=user_id),
                to=[MessageParticipant(type=ActorType.AGENT, id=agent_id)],
                msg_type=MessageType.CHAT,
                payload={"content": body.message}
            )
            
            # Build context
            context = AgentDecisionContext(
                message=message,
                conversation_id=conversation_id,
                patient_context=patient_context,
                short_term_memory=short_term_memory,
                available_tools=available_tools
            )
            
            # Log the interaction start
            AuditLogger.log_event(
                event_type=AuditEvent.PHI_ACCESSED,
                user_id=user_id,
                resource_type="agent_chat",
                resource_id=conversation_id,
                action="stream_start",
                status="success",
                metadata={"agent_id": agent_id, "has_health_context": bool(patient_context)},
                patient_id=user_id if user_role == "patient" else None
            )
            
            # Stream the response
            async for chunk in agent_engine.process_message_streaming(agent_id, context):
                yield f"data: {json.dumps(chunk)}\n\n"
            
            # If streaming to a patient with Agent Clona, extract symptoms
            if agent_id == "clona" and body.message:
                symptoms = await agent_engine.extract_symptoms_from_message(
                    body.message,
                    patient_context
                )
                if symptoms.get("symptoms") or symptoms.get("urgency_level") not in ["routine", "unknown"]:
                    yield f"data: {json.dumps({'type': 'symptoms_extracted', 'data': symptoms})}\n\n"
            
            # Store interaction in memory
            await memory_service.store_short_term(
                agent_id=agent_id,
                user_id=user_id,
                conversation_id=conversation_id,
                content=f"User: {body.message}",
                ttl_hours=2,
                metadata={"has_symptoms": bool(patient_context)}
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

@router.get("/tasks")
async def list_tasks(
    request: Request,
    status: Optional[str] = Query(None),
    task_type: Optional[str] = Query(None),
    limit: int = Query(20, le=100),
    offset: int = Query(0),
    user: dict = Depends(get_current_user)
):
    """List agent tasks"""
    user_id = user["id"]

    # In production, fetch from database
    tasks = []

    return {
        "tasks": tasks,
        "total": len(tasks)
    }


@router.get("/tasks/{task_id}")
async def get_task(
    task_id: str,
    request: Request,
    user: dict = Depends(get_current_user)
):
    """Get task details"""
    return {"id": task_id, "status": "not_found"}


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
