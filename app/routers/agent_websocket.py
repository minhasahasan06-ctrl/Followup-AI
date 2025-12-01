"""
Agent WebSocket Router
Handles real-time communication between users and AI agents
"""

import os
import json
import logging
from typing import Optional
from datetime import datetime
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, HTTPException, Query
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
import uuid

from app.models.agent_models import (
    MessageEnvelope, MessageType, ActorType,
    MessageParticipant, WebSocketMessage,
    AgentDecisionContext
)
from app.services.message_router import get_message_router, MessageRouter
from app.services.agent_engine import get_agent_engine, AgentEngine
from app.services.memory_service import get_memory_service, MemoryService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ws", tags=["Agent WebSocket"])

security = HTTPBearer(auto_error=False)


async def verify_token(token: str) -> Optional[dict]:
    """Verify JWT token and return user info"""
    try:
        # In production, verify against Auth0/Cognito
        # For now, decode without verification in dev mode
        if os.getenv("DEV_MODE_SECRET"):
            # Development mode - accept any token
            try:
                payload = jwt.decode(token, options={"verify_signature": False})
                return {
                    "id": payload.get("sub") or payload.get("id"),
                    "role": payload.get("role", "patient")
                }
            except:
                # Allow simple tokens in dev mode
                return {"id": token, "role": "patient"}
        
        # Production mode - verify signature
        secret = os.getenv("SESSION_SECRET", "")
        payload = jwt.decode(token, secret, algorithms=["HS256"])
        return {
            "id": payload.get("sub") or payload.get("id"),
            "role": payload.get("role", "patient")
        }
    except Exception as e:
        logger.error(f"Token verification failed: {e}")
        return None


@router.websocket("/agent")
async def agent_websocket(
    websocket: WebSocket,
    token: Optional[str] = Query(None)
):
    """
    WebSocket endpoint for agent communication
    
    Protocol:
    1. Client sends auth message with token
    2. Server validates and registers connection
    3. Bidirectional message exchange using MessageEnvelope protocol
    """
    user_info = None
    message_router = await get_message_router()
    agent_engine = await get_agent_engine()
    memory_service = await get_memory_service()

    try:
        # Initial connection (not yet authenticated)
        await websocket.accept()
        logger.info("WebSocket connection accepted, awaiting authentication")

        # Wait for authentication message
        auth_timeout = 30  # seconds
        try:
            auth_data = await websocket.receive_json()
            
            if auth_data.get("type") != "auth":
                await websocket.send_json({
                    "type": "error",
                    "payload": {"message": "First message must be authentication"}
                })
                await websocket.close()
                return

            # Extract and verify token
            auth_token = auth_data.get("payload", {}).get("token") or token
            user_id = auth_data.get("payload", {}).get("userId")
            user_role = auth_data.get("payload", {}).get("role", "patient")

            # Verify token or use dev mode bypass
            if auth_token:
                user_info = await verify_token(auth_token)
            elif os.getenv("DEV_MODE_SECRET") and user_id:
                # Dev mode bypass
                user_info = {"id": user_id, "role": user_role}

            if not user_info:
                await websocket.send_json({
                    "type": "error",
                    "payload": {"message": "Authentication failed"}
                })
                await websocket.close()
                return

            # Register connection
            # Note: We're passing websocket directly since we already accepted it
            message_router.connection_manager.active_connections[user_info["id"]].add(websocket)
            message_router.connection_manager.connection_users[websocket] = user_info["id"]

            logger.info(f"User {user_info['id']} authenticated via WebSocket")

            # Send authentication success
            await websocket.send_json({
                "type": "auth_success",
                "payload": {
                    "userId": user_info["id"],
                    "role": user_info["role"],
                    "agentId": "lysa" if user_info["role"] == "doctor" else "clona"
                }
            })

            # Send agent presence
            await websocket.send_json({
                "type": "presence",
                "payload": {
                    "isOnline": True,
                    "agentId": "lysa" if user_info["role"] == "doctor" else "clona"
                }
            })

            # Deliver any pending messages
            await message_router.deliver_pending_messages(user_info["id"])

        except Exception as e:
            logger.error(f"Authentication error: {e}")
            await websocket.close()
            return

        # Main message loop
        while True:
            try:
                data = await websocket.receive_json()
                await handle_websocket_message(
                    websocket=websocket,
                    data=data,
                    user_info=user_info,
                    message_router=message_router,
                    agent_engine=agent_engine,
                    memory_service=memory_service
                )
            except WebSocketDisconnect:
                logger.info(f"User {user_info['id']} disconnected")
                break
            except json.JSONDecodeError:
                await websocket.send_json({
                    "type": "error",
                    "payload": {"message": "Invalid JSON"}
                })
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                await websocket.send_json({
                    "type": "error",
                    "payload": {"message": "Error processing message"}
                })

    except WebSocketDisconnect:
        logger.info("Client disconnected before authentication")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        # Clean up connection
        if user_info:
            await message_router.connection_manager.disconnect(websocket)


async def handle_websocket_message(
    websocket: WebSocket,
    data: dict,
    user_info: dict,
    message_router: MessageRouter,
    agent_engine: AgentEngine,
    memory_service: MemoryService
):
    """Handle incoming WebSocket messages"""
    msg_type = data.get("type")
    payload = data.get("payload", {})

    if msg_type == "message":
        # User sending a chat message to agent
        await handle_chat_message(
            websocket=websocket,
            payload=payload,
            user_info=user_info,
            message_router=message_router,
            agent_engine=agent_engine,
            memory_service=memory_service
        )

    elif msg_type == "typing":
        # User typing indicator
        conversation_id = payload.get("conversationId")
        is_typing = payload.get("isTyping", False)
        message_router.connection_manager.set_typing(
            user_info["id"],
            conversation_id,
            is_typing
        )

    elif msg_type == "read":
        # Mark messages as read
        message_ids = payload.get("messageIds", [])
        # In production, update read status in database
        await websocket.send_json({
            "type": "ack",
            "payload": {
                "action": "read",
                "messageIds": message_ids
            }
        })

    elif msg_type == "approval":
        # Human-in-the-loop approval decision
        await handle_approval_decision(
            websocket=websocket,
            payload=payload,
            user_info=user_info,
            message_router=message_router
        )

    else:
        await websocket.send_json({
            "type": "error",
            "payload": {"message": f"Unknown message type: {msg_type}"}
        })


async def handle_chat_message(
    websocket: WebSocket,
    payload: dict,
    user_info: dict,
    message_router: MessageRouter,
    agent_engine: AgentEngine,
    memory_service: MemoryService
):
    """Process a chat message from user to agent"""
    content = payload.get("content", "")
    conversation_id = payload.get("conversationId")
    msg_id = payload.get("msgId") or str(uuid.uuid4())

    if not content:
        await websocket.send_json({
            "type": "error",
            "payload": {"message": "Message content is required"}
        })
        return

    # Determine target agent based on user role
    agent_id = "lysa" if user_info["role"] == "doctor" else "clona"

    # Create message envelope
    envelope = MessageEnvelope.create(
        sender=MessageParticipant(type=ActorType.USER, id=user_info["id"]),
        to=[MessageParticipant(type=ActorType.AGENT, id=agent_id)],
        msg_type=MessageType.CHAT,
        payload={"content": content, "conversationId": conversation_id},
        msg_id=msg_id
    )

    # Send typing indicator
    await message_router.send_typing_indicator(
        agent_id=agent_id,
        user_id=user_info["id"],
        conversation_id=conversation_id or "default",
        is_typing=True
    )

    try:
        # Get memory context
        short_term = await memory_service.get_short_term_memories(
            agent_id=agent_id,
            user_id=user_info["id"],
            conversation_id=conversation_id,
            limit=5
        )

        # Build decision context
        context = AgentDecisionContext(
            conversation_id=conversation_id or "default",
            message=envelope,
            short_term_memory=short_term,
            long_term_memory=[],
            patient_context={"patient_id": user_info["id"]} if user_info["role"] == "patient" else None,
            available_tools=[]  # Tools would be loaded here
        )

        # Process through agent engine
        result = await agent_engine.process_message(agent_id, context)

        # Store interaction in memory
        if result.memory_updates:
            for mem in result.memory_updates:
                await memory_service.store_short_term(
                    agent_id=agent_id,
                    user_id=user_info["id"],
                    conversation_id=conversation_id or "default",
                    content=mem.content,
                    ttl_hours=2,
                    metadata=mem.metadata
                )

        # Send response to user
        if result.response_message:
            response_envelope = MessageEnvelope.create(
                sender=MessageParticipant(type=ActorType.AGENT, id=agent_id),
                to=[MessageParticipant(type=ActorType.USER, id=user_info["id"])],
                msg_type=MessageType.CHAT,
                payload={
                    "content": result.response_message,
                    "conversationId": conversation_id,
                    "inReplyTo": msg_id
                }
            )

            await websocket.send_json({
                "type": "message",
                "payload": {
                    "msgId": response_envelope.msg_id,
                    "from": {"type": "agent", "id": agent_id},
                    "content": result.response_message,
                    "conversationId": conversation_id,
                    "timestamp": response_envelope.timestamp.isoformat()
                }
            })

        # Handle tool calls if any
        if result.tool_calls:
            for tool_call in result.tool_calls:
                await websocket.send_json({
                    "type": "tool_call",
                    "payload": {
                        "toolName": tool_call.tool_name,
                        "status": "pending" if tool_call.requires_approval else "running",
                        "requiresApproval": tool_call.requires_approval
                    }
                })

        # Handle human approval if needed
        if result.requires_human_confirmation and result.confirmation_details:
            await websocket.send_json({
                "type": "approval_required",
                "payload": {
                    "messageId": result.confirmation_details.message_id,
                    "toolName": result.confirmation_details.tool_name,
                    "reason": result.confirmation_details.reason
                }
            })

    except Exception as e:
        logger.error(f"Error processing chat message: {e}")
        await websocket.send_json({
            "type": "error",
            "payload": {"message": "Failed to process message"}
        })

    finally:
        # Stop typing indicator
        await message_router.send_typing_indicator(
            agent_id=agent_id,
            user_id=user_info["id"],
            conversation_id=conversation_id or "default",
            is_typing=False
        )


async def handle_approval_decision(
    websocket: WebSocket,
    payload: dict,
    user_info: dict,
    message_router: MessageRouter
):
    """Handle human-in-the-loop approval decision"""
    message_id = payload.get("messageId")
    approved = payload.get("approved", False)
    notes = payload.get("notes")

    if not message_id:
        await websocket.send_json({
            "type": "error",
            "payload": {"message": "Message ID is required"}
        })
        return

    # In production, this would update the database and trigger tool execution
    logger.info(f"Approval decision: {message_id} = {approved} by {user_info['id']}")

    await websocket.send_json({
        "type": "approval_result",
        "payload": {
            "messageId": message_id,
            "approved": approved,
            "approvedBy": user_info["id"]
        }
    })
