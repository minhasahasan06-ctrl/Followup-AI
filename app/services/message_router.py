"""
Central Message Router Service
Routes messages between agents, users, and tools
Integrates with Redis streams for reliable message delivery and persistence
Enforces consent-based routing for all doctor-patient communications
"""

import os
import json
import logging
from typing import Optional, Dict, Any, List, Set, Tuple
from datetime import datetime
import asyncio
from collections import defaultdict
import uuid

from app.models.agent_models import (
    MessageEnvelope, MessageType, ActorType,
    MessageParticipant, PresenceStatus, TypingIndicator,
    WebSocketMessage
)
from app.services.redis_stream_service import get_redis_stream_service, RedisStreamService
from app.services.consent_service import get_consent_service, ConsentService

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
    
    Integrates with Redis streams for:
    - Reliable message delivery
    - Message persistence
    - Presence management
    - Tool call queuing
    """

    def __init__(self):
        self.connection_manager = ConnectionManager()
        self._message_handlers: Dict[MessageType, List[Any]] = defaultdict(list)
        self._pending_deliveries: Dict[str, MessageEnvelope] = {}
        self._redis_stream: Optional[RedisStreamService] = None
        self._initialized = False

    async def initialize(self):
        """Initialize the message router and Redis stream service"""
        if self._initialized:
            return

        logger.info("Initializing Message Router...")
        
        # Initialize Redis stream service for reliable messaging
        try:
            self._redis_stream = await get_redis_stream_service()
            logger.info("Redis stream service connected to message router")
            
            # Sync presence from Redis (for multi-process recovery)
            await self._sync_presence_from_redis()
        except Exception as e:
            logger.warning(f"Redis stream service not available: {e}")
            self._redis_stream = None
        
        self._initialized = True
        logger.info("Message Router initialized successfully")
    
    async def _sync_presence_from_redis(self):
        """Sync presence data from Redis on startup (for multi-process consistency)"""
        if not self._redis_stream:
            return
        
        try:
            online_users = await self._redis_stream.get_online_users()
            for user_id in online_users:
                presence_data = await self._redis_stream.get_presence(user_id)
                if presence_data and presence_data.get("is_online"):
                    self.connection_manager.presence[user_id] = PresenceStatus(
                        user_id=user_id,
                        is_online=True,
                        last_seen_at=datetime.fromisoformat(presence_data.get("last_seen", datetime.utcnow().isoformat())),
                        active_connections=0,
                        current_activity=presence_data.get("activity"),
                        current_conversation_id=presence_data.get("conversation_id")
                    )
            logger.info(f"Synced {len(online_users)} online users from Redis")
        except Exception as e:
            logger.warning(f"Failed to sync presence from Redis: {e}")
    
    def get_redis_stream(self):
        """Get the Redis stream service (for dependency injection)"""
        return self._redis_stream
    
    def get_connection_manager(self):
        """Get the connection manager (for dependency injection)"""
        return self.connection_manager

    def register_handler(self, message_type: MessageType, handler: Any):
        """Register a handler for a specific message type"""
        self._message_handlers[message_type].append(handler)

    async def route_message(
        self,
        envelope: MessageEnvelope,
        sender_websocket: Any = None,
        persist: bool = True
    ) -> Dict[str, Any]:
        """
        Route a message to its recipients
        Returns delivery status
        
        Args:
            envelope: The message envelope to route
            sender_websocket: Optional websocket to send ack to
            persist: Whether to persist to Redis stream (default True)
        """
        results = {
            "msg_id": envelope.msg_id,
            "delivered_to": [],
            "pending": [],
            "failed": []
        }
        
        # Persist message to Redis stream for reliability (all recipients)
        if persist:
            payload = envelope.payload or {}
            if self._redis_stream:
                try:
                    # Persist to Redis stream for each recipient
                    for recipient in envelope.to:
                        await self._redis_stream.add_message(
                            msg_id=envelope.msg_id,
                            sender_type=envelope.sender.type.value if hasattr(envelope.sender.type, 'value') else str(envelope.sender.type),
                            sender_id=envelope.sender.id,
                            recipient_type=recipient.type.value if hasattr(recipient.type, 'value') else str(recipient.type),
                            recipient_id=recipient.id,
                            message_type=envelope.type.value if hasattr(envelope.type, 'value') else str(envelope.type),
                            payload=payload,
                            conversation_id=payload.get("conversation_id")
                        )
                except Exception as e:
                    logger.error(f"Failed to persist message to Redis stream: {e}")
                    # Fall through to database fallback
                    await self._persist_to_database(envelope, payload)
            else:
                # No Redis available, persist directly to database
                await self._persist_to_database(envelope, payload)

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

    async def _persist_to_database(
        self,
        envelope: MessageEnvelope,
        payload: Dict[str, Any]
    ):
        """
        Persist message directly to database as fallback when Redis is unavailable.
        This ensures message durability even without Redis.
        """
        try:
            from app.database import SessionLocal
            from sqlalchemy import text
            import json
            
            db = SessionLocal()
            try:
                # Serialize recipients for to_json column
                to_json = json.dumps([
                    {"type": r.type.value if hasattr(r.type, 'value') else str(r.type), "id": r.id}
                    for r in envelope.to
                ])
                
                # Insert message into agent_messages table
                db.execute(
                    text("""
                        INSERT INTO agent_messages (
                            msg_id, conversation_id, from_type, from_id,
                            to_json, message_type, content, payload_json,
                            delivered, created_at
                        ) VALUES (
                            :msg_id, :conversation_id, :from_type, :from_id,
                            :to_json::jsonb, :message_type, :content, :payload_json::jsonb,
                            false, NOW()
                        )
                        ON CONFLICT (msg_id) DO NOTHING
                    """),
                    {
                        "msg_id": envelope.msg_id,
                        "conversation_id": payload.get("conversation_id", ""),
                        "from_type": envelope.sender.type.value if hasattr(envelope.sender.type, 'value') else str(envelope.sender.type),
                        "from_id": envelope.sender.id,
                        "to_json": to_json,
                        "message_type": envelope.type.value if hasattr(envelope.type, 'value') else str(envelope.type),
                        "content": payload.get("content", ""),
                        "payload_json": json.dumps(payload)
                    }
                )
                db.commit()
                logger.debug(f"Message {envelope.msg_id} persisted to database")
            finally:
                db.close()
        except Exception as e:
            logger.error(f"Failed to persist message to database: {e}")

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

    async def queue_tool_call(
        self,
        tool_call_id: str,
        agent_id: str,
        tool_name: str,
        parameters: Dict[str, Any],
        requires_approval: bool = False,
        user_id: Optional[str] = None,
        priority: str = "normal"
    ) -> bool:
        """
        Queue a tool call for async processing via Redis streams
        
        Args:
            tool_call_id: Unique ID for the tool call
            agent_id: The agent requesting the tool call
            tool_name: Name of the tool to execute
            parameters: Tool parameters
            requires_approval: Whether human approval is needed
            user_id: User associated with this tool call
            priority: Priority level (low, normal, high)
        """
        if self._redis_stream:
            try:
                entry_id = await self._redis_stream.add_tool_call(
                    tool_call_id=tool_call_id,
                    agent_id=agent_id,
                    tool_name=tool_name,
                    parameters=parameters,
                    requires_approval=requires_approval,
                    user_id=user_id,
                    priority=priority
                )
                logger.info(f"Queued tool call {tool_call_id} for {tool_name}")
                return entry_id is not None
            except Exception as e:
                logger.error(f"Failed to queue tool call: {e}")
                return False
        else:
            logger.warning("Redis stream not available, tool call not queued")
            return False

    async def update_presence(
        self,
        user_id: str,
        is_online: bool,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Update user presence in Redis and broadcast to connected clients
        """
        # Update local connection manager
        self.connection_manager.presence[user_id] = PresenceStatus(
            user_id=user_id,
            is_online=is_online,
            last_seen_at=datetime.utcnow(),
            active_connections=len(self.connection_manager.active_connections.get(user_id, set())),
            current_activity=metadata.get("activity") if metadata else None,
            current_conversation_id=metadata.get("conversation_id") if metadata else None
        )
        
        # Update Redis presence
        if self._redis_stream:
            try:
                await self._redis_stream.set_presence(user_id, is_online, metadata)
            except Exception as e:
                logger.error(f"Failed to update Redis presence: {e}")

    async def heartbeat(self, user_id: str):
        """
        Update presence heartbeat for a user
        """
        if self._redis_stream:
            try:
                await self._redis_stream.heartbeat(user_id)
            except Exception as e:
                logger.debug(f"Heartbeat update failed: {e}")

    async def get_online_users(self) -> List[str]:
        """
        Get list of currently online users
        """
        if self._redis_stream:
            try:
                return await self._redis_stream.get_online_users()
            except Exception as e:
                logger.error(f"Failed to get online users: {e}")
        
        # Fall back to local connection manager
        return [uid for uid, presence in self.connection_manager.presence.items() if presence.is_online]

    async def publish_event(
        self,
        event_type: str,
        actor_type: str,
        actor_id: str,
        data: Dict[str, Any],
        target_users: Optional[List[str]] = None
    ):
        """
        Publish a system event to Redis stream
        """
        if self._redis_stream:
            try:
                await self._redis_stream.add_event(
                    event_type=event_type,
                    actor_type=actor_type,
                    actor_id=actor_id,
                    data=data,
                    target_users=target_users
                )
            except Exception as e:
                logger.error(f"Failed to publish event: {e}")


# Singleton instance
message_router = MessageRouter()


async def get_message_router() -> MessageRouter:
    """Get initialized message router instance"""
    if not message_router._initialized:
        await message_router.initialize()
    return message_router
