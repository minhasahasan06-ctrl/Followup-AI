"""
Agent REST API Router
Handles CRUD operations for agent conversations, messages, and tasks
"""

import os
import logging
from typing import Optional, List
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel

from app.models.agent_models import (
    ConversationCreate, ConversationResponse,
    MessageCreate, MessageResponse,
    TaskCreate, TaskResponse,
    PresenceStatus, ApprovalDecision
)
from app.services.agent_engine import get_agent_engine, AgentEngine
from app.services.message_router import get_message_router, MessageRouter
from app.services.memory_service import get_memory_service, MemoryService

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


# ==================== APPROVALS ====================

@router.get("/approvals/pending")
async def list_pending_approvals(
    request: Request,
    user: dict = Depends(get_current_user)
):
    """List pending approval requests for the current user (doctors only)"""
    if user["role"] != "doctor":
        raise HTTPException(status_code=403, detail="Only doctors can view approvals")

    # In production, fetch from database
    approvals = []

    return {
        "approvals": approvals,
        "total": len(approvals)
    }


@router.post("/approvals/{message_id}")
async def submit_approval_decision(
    message_id: str,
    body: ApprovalDecision,
    request: Request,
    user: dict = Depends(get_current_user)
):
    """Submit approval decision for a tool call"""
    if user["role"] != "doctor":
        raise HTTPException(status_code=403, detail="Only doctors can approve")

    # In production, update database and execute tool if approved
    return {
        "messageId": message_id,
        "approved": body.approved,
        "approvedBy": user["id"],
        "approvedAt": datetime.utcnow().isoformat()
    }


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
