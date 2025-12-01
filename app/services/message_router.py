"""
Central Message Router Service
Routes messages between agents, users, and tools
"""

import os
import json
import logging
from typing import Optional, Dict, Any, List, Set
from datetime import datetime
import asyncio
from collections import defaultdict
import uuid

from app.models.agent_models import (
    MessageEnvelope, MessageType, ActorType,
    MessageParticipant, PresenceStatus, TypingIndicator,
    WebSocketMessage
)

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages WebSocket connections for real-time messaging"""

    def __init__(self):
        # user_id -> set of websocket connections
        self.active_connections: Dict[str, Set[Any]] = defaultdict(set)
        # websocket -> user_id
        self.connection_users: Dict[Any, str] = {}
        # user_id -> presence status
        self.presence: Dict[str, PresenceStatus] = {}
        # user_id -> current activity
        self.typing_status: Dict[str, Dict[str, bool]] = defaultdict(dict)

    async def connect(self, websocket: Any, user_id: str, accept: bool = True):
        """Register a new WebSocket connection
        
        Args:
            websocket: The WebSocket connection
            user_id: The user ID to associate with this connection
            accept: Whether to accept the websocket (set to False if already accepted)
        """
        if accept:
            await websocket.accept()
        self.active_connections[user_id].add(websocket)
        self.connection_users[websocket] = user_id
        
        # Update presence
        self.presence[user_id] = PresenceStatus(
            user_id=user_id,
            is_online=True,
            last_seen_at=datetime.utcnow(),
            active_connections=len(self.active_connections[user_id]),
            current_activity=None,
            current_conversation_id=None
        )
        
        logger.info(f"User {user_id} connected. Total connections: {len(self.active_connections[user_id])}")
    
    def register_connection(self, websocket: Any, user_id: str):
        """Register an already-accepted WebSocket connection (synchronous)
        
        Use this when the websocket has already been accepted elsewhere.
        """
        self.active_connections[user_id].add(websocket)
        self.connection_users[websocket] = user_id
        
        # Update presence
        self.presence[user_id] = PresenceStatus(
            user_id=user_id,
            is_online=True,
            last_seen_at=datetime.utcnow(),
            active_connections=len(self.active_connections[user_id]),
            current_activity=None,
            current_conversation_id=None
        )
        
        logger.info(f"User {user_id} registered. Total connections: {len(self.active_connections[user_id])}")

    async def disconnect(self, websocket: Any):
        """Remove a WebSocket connection"""
        user_id = self.connection_users.get(websocket)
        if user_id:
            self.active_connections[user_id].discard(websocket)
            del self.connection_users[websocket]
            
            # Update presence
            if len(self.active_connections[user_id]) == 0:
                self.presence[user_id] = PresenceStatus(
                    user_id=user_id,
                    is_online=False,
                    last_seen_at=datetime.utcnow(),
                    active_connections=0,
                    current_activity=None,
                    current_conversation_id=None
                )
                # Clean up empty set
                del self.active_connections[user_id]
            else:
                self.presence[user_id].active_connections = len(self.active_connections[user_id])
            
            logger.info(f"User {user_id} disconnected. Remaining connections: {len(self.active_connections.get(user_id, set()))}")

    async def send_to_user(self, user_id: str, message: Dict[str, Any]):
        """Send message to all connections of a specific user"""
        connections = self.active_connections.get(user_id, set())
        if not connections:
            logger.debug(f"No active connections for user {user_id}")
            return False

        message_text = json.dumps(message)
        disconnected = set()
        
        for websocket in connections:
            try:
                await websocket.send_text(message_text)
            except Exception as e:
                logger.error(f"Error sending to user {user_id}: {e}")
                disconnected.add(websocket)

        # Clean up failed connections
        for ws in disconnected:
            await self.disconnect(ws)

        return len(connections) - len(disconnected) > 0

    async def broadcast(self, message: Dict[str, Any], exclude_user: Optional[str] = None):
        """Broadcast message to all connected users"""
        message_text = json.dumps(message)
        
        for user_id, connections in self.active_connections.items():
            if user_id == exclude_user:
                continue
            
            disconnected = set()
            for websocket in connections:
                try:
                    await websocket.send_text(message_text)
                except Exception:
                    disconnected.add(websocket)
            
            for ws in disconnected:
                await self.disconnect(ws)

    def get_presence(self, user_id: str) -> Optional[PresenceStatus]:
        """Get presence status for a user"""
        return self.presence.get(user_id)

    def is_online(self, user_id: str) -> bool:
        """Check if user is online"""
        presence = self.presence.get(user_id)
        return presence.is_online if presence else False

    def set_typing(self, user_id: str, conversation_id: str, is_typing: bool):
        """Set typing status for a user in a conversation"""
        self.typing_status[user_id][conversation_id] = is_typing

    def get_typing(self, conversation_id: str) -> List[str]:
        """Get list of users typing in a conversation"""
        typing_users = []
        for user_id, conversations in self.typing_status.items():
            if conversations.get(conversation_id, False):
                typing_users.append(user_id)
        return typing_users


class MessageRouter:
    """
    Central message router for multi-agent communication
    Handles routing between:
    - User <-> Agent Clona (patients)
    - User <-> Assistant Lysa (doctors)
    - Agent Clona <-> Assistant Lysa (inter-agent)
    - Agents <-> Tools
    """

    def __init__(self):
        self.connection_manager = ConnectionManager()
        self._message_handlers: Dict[MessageType, List[Any]] = defaultdict(list)
        self._pending_deliveries: Dict[str, MessageEnvelope] = {}
        self._initialized = False

    async def initialize(self):
        """Initialize the message router"""
        if self._initialized:
            return

        logger.info("Initializing Message Router...")
        self._initialized = True
        logger.info("Message Router initialized successfully")

    def register_handler(self, message_type: MessageType, handler: Any):
        """Register a handler for a specific message type"""
        self._message_handlers[message_type].append(handler)

    async def route_message(
        self,
        envelope: MessageEnvelope,
        sender_websocket: Any = None
    ) -> Dict[str, Any]:
        """
        Route a message to its recipients
        Returns delivery status
        """
        results = {
            "msg_id": envelope.msg_id,
            "delivered_to": [],
            "pending": [],
            "failed": []
        }

        for recipient in envelope.to:
            try:
                if recipient.type == ActorType.AGENT:
                    # Route to agent
                    success = await self._route_to_agent(envelope, recipient.id)
                elif recipient.type == ActorType.USER:
                    # Route to user via WebSocket
                    success = await self._route_to_user(envelope, recipient.id)
                else:
                    success = False

                if success:
                    results["delivered_to"].append({
                        "type": recipient.type,
                        "id": recipient.id
                    })
                else:
                    results["pending"].append({
                        "type": recipient.type,
                        "id": recipient.id
                    })
                    # Store for later delivery
                    self._pending_deliveries[f"{recipient.type}:{recipient.id}:{envelope.msg_id}"] = envelope

            except Exception as e:
                logger.error(f"Failed to route message to {recipient}: {e}")
                results["failed"].append({
                    "type": recipient.type,
                    "id": recipient.id,
                    "error": str(e)
                })

        # Send delivery acknowledgment to sender
        if sender_websocket:
            ack_message = {
                "type": "ack",
                "payload": {
                    "msg_id": envelope.msg_id,
                    "status": "routed",
                    "results": results
                }
            }
            try:
                await sender_websocket.send_json(ack_message)
            except Exception as e:
                logger.error(f"Failed to send ack: {e}")

        return results

    async def _route_to_agent(self, envelope: MessageEnvelope, agent_id: str) -> bool:
        """Route message to an AI agent for processing"""
        # Call registered handlers for the message type
        handlers = self._message_handlers.get(envelope.type, [])
        
        for handler in handlers:
            try:
                await handler(envelope, agent_id)
            except Exception as e:
                logger.error(f"Handler error for agent {agent_id}: {e}")

        # Agent messages are always "delivered" (processed asynchronously)
        return True

    async def _route_to_user(self, envelope: MessageEnvelope, user_id: str) -> bool:
        """Route message to a user via WebSocket"""
        message = {
            "type": "message",
            "payload": {
                "msg_id": envelope.msg_id,
                "from": {
                    "type": envelope.sender.type.value if hasattr(envelope.sender.type, 'value') else envelope.sender.type,
                    "id": envelope.sender.id
                },
                "type": envelope.type.value if hasattr(envelope.type, 'value') else envelope.type,
                "timestamp": envelope.timestamp.isoformat() if hasattr(envelope.timestamp, 'isoformat') else envelope.timestamp,
                "payload": envelope.payload
            }
        }
        
        return await self.connection_manager.send_to_user(user_id, message)

    async def send_agent_response(
        self,
        agent_id: str,
        user_id: str,
        content: str,
        conversation_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Send an agent response to a user"""
        envelope = MessageEnvelope.create(
            sender=MessageParticipant(type=ActorType.AGENT, id=agent_id),
            to=[MessageParticipant(type=ActorType.USER, id=user_id)],
            msg_type=MessageType.CHAT,
            payload={
                "content": content,
                "conversation_id": conversation_id,
                **(metadata or {})
            }
        )

        await self.route_message(envelope)

    async def send_typing_indicator(
        self,
        agent_id: str,
        user_id: str,
        conversation_id: str,
        is_typing: bool
    ):
        """Send typing indicator to user"""
        message = {
            "type": "typing",
            "payload": {
                "agent_id": agent_id,
                "conversation_id": conversation_id,
                "is_typing": is_typing
            }
        }
        await self.connection_manager.send_to_user(user_id, message)

    async def send_presence_update(
        self,
        agent_id: str,
        user_id: str,
        is_online: bool
    ):
        """Send agent presence update to user"""
        message = {
            "type": "presence",
            "payload": {
                "agent_id": agent_id,
                "is_online": is_online,
                "last_seen": datetime.utcnow().isoformat()
            }
        }
        await self.connection_manager.send_to_user(user_id, message)

    async def send_tool_call_update(
        self,
        user_id: str,
        conversation_id: str,
        tool_call_id: str,
        status: str,
        result: Optional[Dict[str, Any]] = None
    ):
        """Send tool call status update to user"""
        message = {
            "type": "tool_update",
            "payload": {
                "conversation_id": conversation_id,
                "tool_call_id": tool_call_id,
                "status": status,
                "result": result
            }
        }
        await self.connection_manager.send_to_user(user_id, message)

    async def send_approval_request(
        self,
        doctor_id: str,
        patient_id: str,
        tool_name: str,
        tool_input: Dict[str, Any],
        message_id: str,
        reason: str
    ):
        """Send approval request to doctor"""
        message = {
            "type": "approval_request",
            "payload": {
                "message_id": message_id,
                "patient_id": patient_id,
                "tool_name": tool_name,
                "tool_input": tool_input,
                "reason": reason,
                "timestamp": datetime.utcnow().isoformat()
            }
        }
        await self.connection_manager.send_to_user(doctor_id, message)

    async def deliver_pending_messages(self, user_id: str):
        """Deliver any pending messages when user connects"""
        pending_keys = [
            k for k in self._pending_deliveries.keys()
            if k.startswith(f"user:{user_id}:")
        ]

        for key in pending_keys:
            envelope = self._pending_deliveries.pop(key)
            await self._route_to_user(envelope, user_id)
            logger.info(f"Delivered pending message {envelope.msg_id} to user {user_id}")


# Singleton instance
message_router = MessageRouter()


async def get_message_router() -> MessageRouter:
    """Get initialized message router instance"""
    if not message_router._initialized:
        await message_router.initialize()
    return message_router
