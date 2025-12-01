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
        persist: bool = True,
        conversation_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Route a message to its recipients with consent verification.
        Returns delivery status.
        
        CRITICAL: All cross-party communication requires verified doctor-patient consent.
        Patient↔Doctor messages must share the SAME conversation thread.
        
        Args:
            envelope: The message envelope to route
            sender_websocket: Optional websocket to send ack to
            persist: Whether to persist to Redis stream (default True)
            conversation_context: Optional context with patient_id, doctor_id, conversation_type
        """
        results = {
            "msg_id": envelope.msg_id,
            "delivered_to": [],
            "pending": [],
            "failed": [],
            "blocked": []
        }
        
        # Get consent service
        consent_service = get_consent_service()
        
        # Build context from envelope payload if not provided
        payload = envelope.payload or {}
        if not conversation_context:
            conversation_context = {
                "patient_id": payload.get("patient_id"),
                "doctor_id": payload.get("doctor_id"),
                "conversation_id": payload.get("conversation_id"),
                "conversation_type": payload.get("conversation_type")
            }
        
        # Resolve sender role and IDs for consent verification
        sender_type = envelope.sender.type.value if hasattr(envelope.sender.type, 'value') else str(envelope.sender.type)
        sender_id = envelope.sender.id
        resolved_context = await self._resolve_conversation_context(
            sender_type=sender_type,
            sender_id=sender_id,
            recipients=envelope.to,
            context=conversation_context
        )
        
        # Update conversation_context with resolved IDs
        conversation_context.update(resolved_context)
        
        # For patient↔doctor direct messaging, ensure shared conversation thread
        if self._is_patient_doctor_direct_message(sender_type, sender_id, envelope.to, conversation_context):
            patient_id = conversation_context.get("patient_id")
            doctor_id = conversation_context.get("doctor_id")
            assignment_id = conversation_context.get("assignment_id")
            
            if patient_id and doctor_id:
                shared_conv_id = await self.find_or_create_patient_doctor_conversation(
                    patient_id=patient_id,
                    doctor_id=doctor_id,
                    assignment_id=assignment_id
                )
                if shared_conv_id:
                    conversation_context["conversation_id"] = shared_conv_id
                    payload["conversation_id"] = shared_conv_id
                    envelope.payload = payload
        
        # Verify consent for EACH recipient
        verified_recipients = []
        for recipient in envelope.to:
            # Get sender and recipient types
            from_type = envelope.sender.type.value if hasattr(envelope.sender.type, 'value') else str(envelope.sender.type)
            to_type = recipient.type.value if hasattr(recipient.type, 'value') else str(recipient.type)
            
            # Check if communication is allowed
            allowed, reason, connection_info = consent_service.can_communicate(
                from_type=from_type,
                from_id=envelope.sender.id,
                to_type=to_type,
                to_id=recipient.id,
                conversation_context=conversation_context
            )
            
            if not allowed:
                logger.warning(
                    f"Message blocked: {envelope.sender.id} -> {recipient.id}, reason: {reason}"
                )
                results["blocked"].append({
                    "type": to_type,
                    "id": recipient.id,
                    "reason": reason
                })
                
                # Audit log for blocked communication attempt (HIPAA compliance)
                await self._log_blocked_communication(
                    envelope=envelope,
                    recipient=recipient,
                    reason=reason,
                    context=conversation_context
                )
                continue
            
            # Add verified recipient
            verified_recipients.append({
                "recipient": recipient,
                "connection_info": connection_info
            })
        
        # If no verified recipients, return early
        if not verified_recipients:
            if sender_websocket:
                error_message = {
                    "type": "error",
                    "payload": {
                        "msg_id": envelope.msg_id,
                        "error": "message_blocked",
                        "blocked": results["blocked"]
                    }
                }
                try:
                    await sender_websocket.send_json(error_message)
                except Exception as e:
                    logger.error(f"Failed to send block notification: {e}")
            return results
        
        # Persist message to Redis stream for reliability (verified recipients only)
        if persist:
            payload = envelope.payload or {}
            if self._redis_stream:
                try:
                    # Persist to Redis stream for each verified recipient
                    for verified in verified_recipients:
                        recipient = verified["recipient"]
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
                    await self._persist_to_database(envelope, payload, conversation_context)
            else:
                # No Redis available, persist directly to database
                await self._persist_to_database(envelope, payload, conversation_context)

        for verified in verified_recipients:
            recipient = verified["recipient"]
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
        payload: Dict[str, Any],
        conversation_context: Optional[Dict[str, Any]] = None
    ):
        """
        Persist message directly to database as fallback when Redis is unavailable.
        This ensures message durability even without Redis.
        Includes sender_role for proper Patient↔Doctor message tagging.
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
                
                # Determine sender_role based on sender type and conversation context
                sender_role = self._determine_sender_role(
                    sender_type=envelope.sender.type.value if hasattr(envelope.sender.type, 'value') else str(envelope.sender.type),
                    sender_id=envelope.sender.id,
                    context=conversation_context
                )
                
                # Get sender display info
                sender_name, sender_avatar = self._get_sender_display_info(envelope.sender.id, sender_role)
                
                # Insert message into agent_messages table with sender_role
                db.execute(
                    text("""
                        INSERT INTO agent_messages (
                            msg_id, conversation_id, from_type, from_id,
                            sender_role, sender_name, sender_avatar,
                            to_json, message_type, content, payload_json,
                            delivered, created_at
                        ) VALUES (
                            :msg_id, :conversation_id, :from_type, :from_id,
                            :sender_role, :sender_name, :sender_avatar,
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
                        "sender_role": sender_role,
                        "sender_name": sender_name,
                        "sender_avatar": sender_avatar,
                        "to_json": to_json,
                        "message_type": envelope.type.value if hasattr(envelope.type, 'value') else str(envelope.type),
                        "content": payload.get("content", ""),
                        "payload_json": json.dumps(payload)
                    }
                )
                db.commit()
                logger.debug(f"Message {envelope.msg_id} persisted to database with sender_role={sender_role}")
            finally:
                db.close()
        except Exception as e:
            logger.error(f"Failed to persist message to database: {e}")
    
    def _determine_sender_role(
        self,
        sender_type: str,
        sender_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Determine the sender's role for message tagging.
        Returns: 'patient', 'doctor', 'clona', 'lysa', or 'system'
        """
        if sender_type == "agent":
            if sender_id == "clona":
                return "clona"
            elif sender_id == "lysa":
                return "lysa"
            return "system"
        elif sender_type == "system":
            return "system"
        elif sender_type == "user":
            # Look up user role from database
            try:
                from app.database import SessionLocal
                from sqlalchemy import text
                
                db = SessionLocal()
                try:
                    result = db.execute(
                        text("SELECT role FROM users WHERE id = :user_id"),
                        {"user_id": sender_id}
                    )
                    row = result.fetchone()
                    if row and row[0] == "doctor":
                        return "doctor"
                    return "patient"
                finally:
                    db.close()
            except Exception as e:
                logger.warning(f"Failed to lookup user role: {e}")
                return "patient"
        return "patient"
    
    def _get_sender_display_info(
        self,
        sender_id: str,
        sender_role: str
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Get sender display name and avatar for UI.
        Returns: (sender_name, sender_avatar)
        """
        if sender_role in ["clona", "lysa"]:
            # Agent display names
            if sender_role == "clona":
                return "Agent Clona", None
            return "Assistant Lysa", None
        elif sender_role == "system":
            return "System", None
        
        # User display name
        try:
            from app.database import SessionLocal
            from sqlalchemy import text
            
            db = SessionLocal()
            try:
                result = db.execute(
                    text("""
                        SELECT first_name, last_name, profile_image_url, role
                        FROM users WHERE id = :user_id
                    """),
                    {"user_id": sender_id}
                )
                row = result.fetchone()
                if row:
                    first_name = row[0] or ""
                    last_name = row[1] or ""
                    avatar = row[2]
                    role = row[3]
                    
                    if role == "doctor":
                        name = f"Dr. {last_name}" if last_name else f"Dr. {first_name}"
                    else:
                        name = f"{first_name} {last_name}".strip()
                    
                    return name or None, avatar
                return None, None
            finally:
                db.close()
        except Exception as e:
            logger.warning(f"Failed to get sender display info: {e}")
            return None, None
    
    async def _resolve_conversation_context(
        self,
        sender_type: str,
        sender_id: str,
        recipients: List[Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Resolve patient_id and doctor_id from sender/recipient types.
        Essential for consent verification in all communication patterns.
        
        Returns updated context with resolved patient_id, doctor_id, and assignment_id.
        """
        result = {
            "patient_id": context.get("patient_id"),
            "doctor_id": context.get("doctor_id"),
            "assignment_id": context.get("assignment_id")
        }
        
        try:
            from app.database import SessionLocal
            from sqlalchemy import text
            
            db = SessionLocal()
            try:
                # Resolve sender's role if it's a user
                if sender_type == "user" and sender_id:
                    user_result = db.execute(
                        text("SELECT role FROM users WHERE id = :user_id"),
                        {"user_id": sender_id}
                    )
                    row = user_result.fetchone()
                    if row:
                        role = row[0]
                        if role == "doctor":
                            result["doctor_id"] = sender_id
                        elif role == "patient":
                            result["patient_id"] = sender_id
                
                # Resolve recipient roles
                for recipient in recipients:
                    rec_type = recipient.type.value if hasattr(recipient.type, 'value') else str(recipient.type)
                    rec_id = recipient.id
                    
                    if rec_type == "user" and rec_id:
                        user_result = db.execute(
                            text("SELECT role FROM users WHERE id = :user_id"),
                            {"user_id": rec_id}
                        )
                        row = user_result.fetchone()
                        if row:
                            role = row[0]
                            if role == "doctor":
                                result["doctor_id"] = rec_id
                            elif role == "patient":
                                result["patient_id"] = rec_id
                    
                    # For agents, try to resolve from conversation context
                    elif rec_type == "agent":
                        # If recipient is Lysa, the sender might be a patient
                        # If recipient is Clona, the sender might be a doctor
                        pass  # Already resolved from sender
                
                # If we have both patient and doctor, look up assignment
                if result.get("patient_id") and result.get("doctor_id") and not result.get("assignment_id"):
                    assign_result = db.execute(
                        text("""
                            SELECT id FROM doctor_patient_assignments
                            WHERE doctor_id = :doctor_id
                            AND patient_id = :patient_id
                            AND status = 'active'
                            ORDER BY assigned_at DESC
                            LIMIT 1
                        """),
                        {"doctor_id": result["doctor_id"], "patient_id": result["patient_id"]}
                    )
                    assign_row = assign_result.fetchone()
                    if assign_row:
                        result["assignment_id"] = assign_row[0]
            finally:
                db.close()
        except Exception as e:
            logger.warning(f"Failed to resolve conversation context: {e}")
        
        return result
    
    def _is_patient_doctor_direct_message(
        self,
        sender_type: str,
        sender_id: str,
        recipients: List[Any],
        context: Dict[str, Any]
    ) -> bool:
        """
        Determine if this is a direct Patient↔Doctor message.
        These must share the same conversation thread.
        """
        # Must have both patient and doctor IDs
        if not context.get("patient_id") or not context.get("doctor_id"):
            return False
        
        # Check if sender is a user (patient or doctor)
        if sender_type != "user":
            return False
        
        # Check if any recipient is a user
        for recipient in recipients:
            rec_type = recipient.type.value if hasattr(recipient.type, 'value') else str(recipient.type)
            if rec_type == "user":
                return True
        
        return False
    
    async def _log_blocked_communication(
        self,
        envelope: MessageEnvelope,
        recipient: Any,
        reason: str,
        context: Optional[Dict[str, Any]] = None
    ):
        """
        Log blocked communication attempt for HIPAA audit trail.
        This is critical for compliance - all blocked attempts must be logged.
        """
        try:
            from app.database import SessionLocal
            from sqlalchemy import text
            import json
            
            db = SessionLocal()
            try:
                db.execute(
                    text("""
                        INSERT INTO agent_audit_logs (
                            actor_type, actor_id, action_type, resource_type, resource_id,
                            details, success, created_at
                        ) VALUES (
                            :actor_type, :actor_id, :action_type, :resource_type, :resource_id,
                            :details::jsonb, :success, NOW()
                        )
                    """),
                    {
                        "actor_type": envelope.sender.type.value if hasattr(envelope.sender.type, 'value') else str(envelope.sender.type),
                        "actor_id": envelope.sender.id,
                        "action_type": "message_blocked",
                        "resource_type": "communication",
                        "resource_id": envelope.msg_id,
                        "details": json.dumps({
                            "recipient_type": recipient.type.value if hasattr(recipient.type, 'value') else str(recipient.type),
                            "recipient_id": recipient.id,
                            "reason": reason,
                            "conversation_context": context,
                            "timestamp": envelope.timestamp.isoformat() if hasattr(envelope.timestamp, 'isoformat') else str(envelope.timestamp)
                        }),
                        "success": False
                    }
                )
                db.commit()
                logger.info(f"Logged blocked communication: {envelope.sender.id} -> {recipient.id}")
            finally:
                db.close()
        except Exception as e:
            logger.error(f"Failed to log blocked communication: {e}")
    
    async def find_or_create_patient_doctor_conversation(
        self,
        patient_id: str,
        doctor_id: str,
        assignment_id: Optional[str] = None
    ) -> Optional[str]:
        """
        Find or create a SHARED conversation thread for Patient↔Doctor direct messaging.
        CRITICAL: All messages between this patient and doctor MUST use the same thread.
        
        Args:
            patient_id: The patient user ID
            doctor_id: The doctor user ID
            assignment_id: Optional doctor_patient_assignment ID for reference
            
        Returns:
            conversation_id of the shared thread
        """
        try:
            from app.database import SessionLocal
            from sqlalchemy import text
            import uuid
            
            db = SessionLocal()
            try:
                # First, try to find existing patient_doctor conversation
                result = db.execute(
                    text("""
                        SELECT id FROM agent_conversations
                        WHERE conversation_type = 'patient_doctor'
                        AND patient_id = :patient_id
                        AND doctor_id = :doctor_id
                        AND status = 'active'
                        LIMIT 1
                    """),
                    {"patient_id": patient_id, "doctor_id": doctor_id}
                )
                row = result.fetchone()
                
                if row:
                    logger.debug(f"Found existing patient_doctor conversation: {row[0]}")
                    return row[0]
                
                # Create new patient_doctor conversation thread
                conversation_id = str(uuid.uuid4())
                
                # Get patient and doctor names for title
                names_result = db.execute(
                    text("""
                        SELECT u.id, u.first_name, u.last_name, u.role
                        FROM users u WHERE u.id IN (:patient_id, :doctor_id)
                    """),
                    {"patient_id": patient_id, "doctor_id": doctor_id}
                )
                names = {row[0]: row for row in names_result.fetchall()}
                
                patient_name = ""
                doctor_name = ""
                if patient_id in names:
                    patient_name = f"{names[patient_id][1]} {names[patient_id][2]}".strip()
                if doctor_id in names:
                    doctor_name = f"Dr. {names[doctor_id][2]}" if names[doctor_id][2] else f"Dr. {names[doctor_id][1]}"
                
                title = f"Chat: {patient_name} & {doctor_name}"
                
                # Insert new conversation
                db.execute(
                    text("""
                        INSERT INTO agent_conversations (
                            id, conversation_type, 
                            participant1_type, participant1_id,
                            participant2_type, participant2_id,
                            patient_id, doctor_id, assignment_id,
                            title, status, message_count,
                            unread_counts, created_at, updated_at
                        ) VALUES (
                            :id, 'patient_doctor',
                            'user', :patient_id,
                            'user', :doctor_id,
                            :patient_id, :doctor_id, :assignment_id,
                            :title, 'active', 0,
                            :unread_counts::jsonb, NOW(), NOW()
                        )
                    """),
                    {
                        "id": conversation_id,
                        "patient_id": patient_id,
                        "doctor_id": doctor_id,
                        "assignment_id": assignment_id,
                        "title": title,
                        "unread_counts": json.dumps({patient_id: 0, doctor_id: 0})
                    }
                )
                db.commit()
                
                logger.info(f"Created patient_doctor conversation: {conversation_id} for {patient_id} <-> {doctor_id}")
                return conversation_id
                
            finally:
                db.close()
        except Exception as e:
            logger.error(f"Failed to find/create patient_doctor conversation: {e}")
            return None
    
    async def get_conversation_context(
        self,
        conversation_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get context for a conversation including patient_id, doctor_id, and type.
        Used for consent verification.
        """
        try:
            from app.database import SessionLocal
            from sqlalchemy import text
            
            db = SessionLocal()
            try:
                result = db.execute(
                    text("""
                        SELECT conversation_type, patient_id, doctor_id, assignment_id,
                               participant1_type, participant1_id,
                               participant2_type, participant2_id
                        FROM agent_conversations
                        WHERE id = :conversation_id
                    """),
                    {"conversation_id": conversation_id}
                )
                row = result.fetchone()
                
                if row:
                    return {
                        "conversation_id": conversation_id,
                        "conversation_type": row[0],
                        "patient_id": row[1],
                        "doctor_id": row[2],
                        "assignment_id": row[3],
                        "participant1_type": row[4],
                        "participant1_id": row[5],
                        "participant2_type": row[6],
                        "participant2_id": row[7]
                    }
                return None
            finally:
                db.close()
        except Exception as e:
            logger.error(f"Failed to get conversation context: {e}")
            return None

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
