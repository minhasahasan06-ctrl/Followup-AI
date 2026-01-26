"""
Delivery Receipt Service
Handles message delivery tracking, read receipts, and sender notifications
with full database persistence and HIPAA-compliant audit logging.
"""

import os
import logging
import asyncio
import json
from typing import Optional, Dict, Any, List
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

from sqlalchemy import text
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

_executor = ThreadPoolExecutor(max_workers=10)


def _get_db_session():
    """Get a fresh database session using the database module"""
    try:
        from app.database import SessionLocal
        return SessionLocal()
    except Exception as e:
        logger.error(f"Failed to create database session: {e}")
        return None


class DeliveryService:
    """
    Handles message delivery tracking with database persistence.
    
    Features:
    - Database persistence for all delivery/read status updates
    - Optional Redis stream integration for event publishing
    - Real-time WebSocket notifications to message senders
    - HIPAA-compliant audit logging
    - Resilient fallback: DB always persisted, Redis/WebSocket best-effort
    """
    
    def __init__(self):
        self._redis_stream = None
        self._connection_manager = None
        self._initialized = False
        self._dependencies_ready = False
    
    def set_dependencies(
        self,
        redis_stream = None,
        connection_manager = None
    ):
        """
        Set service dependencies.
        
        Args:
            redis_stream: Optional Redis stream service for event publishing
            connection_manager: Optional WebSocket connection manager for notifications
        """
        self._redis_stream = redis_stream
        self._connection_manager = connection_manager
        
        # Track if we have working dependencies
        has_redis = redis_stream is not None
        has_ws = connection_manager is not None
        
        if has_redis or has_ws:
            self._dependencies_ready = True
            logger.info(f"DeliveryService dependencies set (Redis: {has_redis}, WebSocket: {has_ws})")
        else:
            logger.warning("DeliveryService initialized without Redis or WebSocket - notifications disabled")
        
        self._initialized = True
    
    def is_ready(self) -> bool:
        """Check if service is initialized and has working dependencies"""
        return self._initialized
    
    def has_notification_support(self) -> bool:
        """Check if real-time notifications are available"""
        return self._connection_manager is not None
    
    async def _run_in_executor(self, func, *args, **kwargs):
        """Run a sync function in thread pool executor"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            _executor,
            lambda: func(*args, **kwargs)
        )
    
    async def mark_messages_delivered(
        self,
        message_ids: List[str],
        recipient_id: str,
        conversation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Mark messages as delivered to a recipient.
        
        Always persists to database. Redis and WebSocket notifications are best-effort.
        """
        delivered_at = datetime.utcnow()
        results = {
            "success": True,
            "delivered_count": 0,
            "delivered_at": delivered_at.isoformat(),
            "message_ids": message_ids,
            "errors": [],
            "notifications_sent": False
        }
        
        if not message_ids:
            results["success"] = False
            results["errors"].append("No message IDs provided")
            return results
        
        # Step 1: Always persist to database (critical path)
        try:
            results["delivered_count"] = await self._run_in_executor(
                self._update_delivered_in_db,
                message_ids,
                delivered_at
            )
        except Exception as e:
            logger.error(f"Failed to persist delivery status to database: {e}")
            results["success"] = False
            results["errors"].append(f"Database update failed: {str(e)}")
            return results
        
        # Step 2: Always create audit log (HIPAA requirement)
        try:
            await self._run_in_executor(
                self._create_audit_log,
                "message_delivered",
                recipient_id,
                message_ids,
                conversation_id,
                delivered_at
            )
        except Exception as e:
            logger.error(f"Failed to create delivery audit log: {e}")
            results["errors"].append(f"Audit log failed: {str(e)}")
        
        # Step 3: Persist delivery event to database for reliable processing
        try:
            await self._run_in_executor(
                self._persist_delivery_event,
                "delivery_receipt",
                message_ids,
                recipient_id,
                conversation_id,
                delivered_at
            )
        except Exception as e:
            logger.error(f"Failed to persist delivery event: {e}")
        
        # Step 4: Best-effort Redis event publish
        if self._redis_stream:
            try:
                await self._publish_delivery_event(
                    event_type="delivery_receipt",
                    message_ids=message_ids,
                    recipient_id=recipient_id,
                    conversation_id=conversation_id,
                    timestamp=delivered_at
                )
            except Exception as e:
                logger.warning(f"Redis delivery event publish failed (non-critical): {e}")
        
        # Step 5: Best-effort WebSocket notifications
        if self._connection_manager:
            try:
                await self._notify_senders_of_delivery(
                    message_ids=message_ids,
                    recipient_id=recipient_id,
                    delivered_at=delivered_at
                )
                results["notifications_sent"] = True
            except Exception as e:
                logger.warning(f"WebSocket delivery notification failed (non-critical): {e}")
        
        logger.info(
            f"Marked {results['delivered_count']} messages as delivered "
            f"to recipient {recipient_id} (notifications: {results['notifications_sent']})"
        )
        
        return results
    
    async def mark_messages_read(
        self,
        message_ids: List[str],
        reader_id: str,
        conversation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Mark messages as read by a user.
        
        Always persists to database. Redis and WebSocket notifications are best-effort.
        """
        read_at = datetime.utcnow()
        results = {
            "success": True,
            "read_count": 0,
            "read_at": read_at.isoformat(),
            "message_ids": message_ids,
            "errors": [],
            "notifications_sent": False
        }
        
        if not message_ids:
            results["success"] = False
            results["errors"].append("No message IDs provided")
            return results
        
        # Step 1: Always persist to database (critical path)
        try:
            results["read_count"] = await self._run_in_executor(
                self._update_read_in_db,
                message_ids,
                read_at
            )
        except Exception as e:
            logger.error(f"Failed to persist read status to database: {e}")
            results["success"] = False
            results["errors"].append(f"Database update failed: {str(e)}")
            return results
        
        # Step 2: Always create audit log (HIPAA requirement)
        try:
            await self._run_in_executor(
                self._create_audit_log,
                "message_read",
                reader_id,
                message_ids,
                conversation_id,
                read_at
            )
        except Exception as e:
            logger.error(f"Failed to create read audit log: {e}")
            results["errors"].append(f"Audit log failed: {str(e)}")
        
        # Step 3: Persist read event to database for reliable processing
        try:
            await self._run_in_executor(
                self._persist_delivery_event,
                "read_receipt",
                message_ids,
                reader_id,
                conversation_id,
                read_at
            )
        except Exception as e:
            logger.error(f"Failed to persist read event: {e}")
        
        # Step 4: Best-effort Redis event publish
        if self._redis_stream:
            try:
                await self._publish_delivery_event(
                    event_type="read_receipt",
                    message_ids=message_ids,
                    recipient_id=reader_id,
                    conversation_id=conversation_id,
                    timestamp=read_at
                )
            except Exception as e:
                logger.warning(f"Redis read event publish failed (non-critical): {e}")
        
        # Step 5: Best-effort WebSocket notifications
        if self._connection_manager:
            try:
                await self._notify_senders_of_read(
                    message_ids=message_ids,
                    reader_id=reader_id,
                    read_at=read_at
                )
                results["notifications_sent"] = True
            except Exception as e:
                logger.warning(f"WebSocket read notification failed (non-critical): {e}")
        
        logger.info(
            f"Marked {results['read_count']} messages as read "
            f"by reader {reader_id} (notifications: {results['notifications_sent']})"
        )
        
        return results
    
    async def get_message_delivery_status(
        self,
        message_id: str
    ) -> Dict[str, Any]:
        """Get the delivery status of a specific message."""
        status = {
            "message_id": message_id,
            "sent": True,
            "sent_at": None,
            "delivered": False,
            "delivered_at": None,
            "read": False,
            "read_at": None
        }
        
        try:
            message = await self._run_in_executor(
                self._get_message_from_db,
                message_id
            )
            if message:
                status["sent_at"] = message.get("created_at")
                status["delivered"] = message.get("delivered", False)
                status["delivered_at"] = message.get("delivered_at")
                status["read"] = message.get("read_at") is not None
                status["read_at"] = message.get("read_at")
        except Exception as e:
            logger.error(f"Error getting message delivery status: {e}")
        
        return status
    
    async def get_undelivered_messages(
        self,
        recipient_id: str,
        conversation_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get all undelivered messages for a recipient."""
        try:
            return await self._run_in_executor(
                self._get_undelivered_from_db,
                recipient_id,
                conversation_id
            )
        except Exception as e:
            logger.error(f"Error getting undelivered messages: {e}")
            return []
    
    async def get_pending_notification_events(
        self,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get pending delivery/read events that need notification processing."""
        try:
            return await self._run_in_executor(
                self._get_pending_events_from_db,
                limit
            )
        except Exception as e:
            logger.error(f"Error getting pending notification events: {e}")
            return []
    
    def _update_delivered_in_db(
        self,
        message_ids: List[str],
        delivered_at: datetime
    ) -> int:
        """Update delivered status in database (sync) - proper session handling"""
        db = _get_db_session()
        if not db:
            raise RuntimeError("No database session available for delivery update")
        
        try:
            result = db.execute(
                text("""
                    UPDATE agent_messages 
                    SET delivered = true, delivered_at = :delivered_at
                    WHERE msg_id = ANY(:message_ids)
                    AND (delivered = false OR delivered IS NULL)
                """),
                {
                    "delivered_at": delivered_at,
                    "message_ids": message_ids
                }
            )
            db.commit()
            return getattr(result, 'rowcount', len(message_ids))
        except Exception as e:
            db.rollback()
            raise e
        finally:
            db.close()
    
    def _update_read_in_db(
        self,
        message_ids: List[str],
        read_at: datetime
    ) -> int:
        """Update read status in database (sync) - proper session handling"""
        db = _get_db_session()
        if not db:
            raise RuntimeError("No database session available for read update")
        
        try:
            result = db.execute(
                text("""
                    UPDATE agent_messages 
                    SET read_at = :read_at
                    WHERE msg_id = ANY(:message_ids)
                    AND read_at IS NULL
                """),
                {
                    "read_at": read_at,
                    "message_ids": message_ids
                }
            )
            db.commit()
            return getattr(result, 'rowcount', len(message_ids))
        except Exception as e:
            db.rollback()
            raise e
        finally:
            db.close()
    
    def _get_message_from_db(self, message_id: str) -> Optional[Dict[str, Any]]:
        """Get a message by ID from database (sync) - proper session handling"""
        db = _get_db_session()
        if not db:
            return None
        
        try:
            result = db.execute(
                text("""
                    SELECT msg_id, from_type, from_id, delivered, 
                           delivered_at, read_at, created_at
                    FROM agent_messages 
                    WHERE msg_id = :message_id
                """),
                {"message_id": message_id}
            )
            row = result.fetchone()
            if row:
                return dict(row._mapping)
            return None
        except Exception as e:
            logger.error(f"Database query for message failed: {e}")
            return None
        finally:
            db.close()
    
    def _get_undelivered_from_db(
        self,
        recipient_id: str,
        conversation_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get undelivered messages from database (sync) - proper session handling"""
        db = _get_db_session()
        if not db:
            return []
        
        try:
            recipient_json = json.dumps([{"type": "user", "id": recipient_id}])
            
            if conversation_id:
                result = db.execute(
                    text("""
                        SELECT msg_id, from_type, from_id, content, created_at
                        FROM agent_messages 
                        WHERE (delivered = false OR delivered IS NULL)
                        AND to_json @> :recipient_json::jsonb
                        AND conversation_id = :conversation_id
                        ORDER BY created_at ASC
                    """),
                    {
                        "recipient_json": recipient_json,
                        "conversation_id": conversation_id
                    }
                )
            else:
                result = db.execute(
                    text("""
                        SELECT msg_id, from_type, from_id, content, created_at
                        FROM agent_messages 
                        WHERE (delivered = false OR delivered IS NULL)
                        AND to_json @> :recipient_json::jsonb
                        ORDER BY created_at ASC
                    """),
                    {"recipient_json": recipient_json}
                )
            
            rows = result.fetchall()
            return [dict(row._mapping) for row in rows]
        except Exception as e:
            logger.error(f"Database query for undelivered messages failed: {e}")
            return []
        finally:
            db.close()
    
    def _get_message_senders(
        self,
        message_ids: List[str]
    ) -> Dict[str, str]:
        """Get sender IDs for a list of message IDs (sync) - proper session handling"""
        db = _get_db_session()
        if not db:
            return {}
        
        try:
            result = db.execute(
                text("""
                    SELECT msg_id, from_id
                    FROM agent_messages 
                    WHERE msg_id = ANY(:message_ids)
                """),
                {"message_ids": message_ids}
            )
            rows = result.fetchall()
            return {row.msg_id: row.from_id for row in rows}
        except Exception as e:
            logger.error(f"Failed to get message senders: {e}")
            return {}
        finally:
            db.close()
    
    def _persist_delivery_event(
        self,
        event_type: str,
        message_ids: List[str],
        actor_id: str,
        conversation_id: Optional[str],
        timestamp: datetime
    ):
        """Persist delivery event to database for reliable background processing"""
        db = _get_db_session()
        if not db:
            return
        
        try:
            event_data = json.dumps({
                "event_type": event_type,
                "message_ids": message_ids,
                "actor_id": actor_id,
                "conversation_id": conversation_id,
                "timestamp": timestamp.isoformat()
            })
            
            db.execute(
                text("""
                    INSERT INTO agent_tasks (
                        agent_id, task_type, task_name, status,
                        input_payload, created_at
                    ) VALUES (
                        'system', 'notification_event', :event_type, 'pending',
                        :event_data::jsonb, NOW()
                    )
                """),
                {
                    "event_type": event_type,
                    "event_data": event_data
                }
            )
            db.commit()
        except Exception as e:
            logger.error(f"Failed to persist delivery event: {e}")
            db.rollback()
        finally:
            db.close()
    
    def _get_pending_events_from_db(self, limit: int) -> List[Dict[str, Any]]:
        """Get pending notification events for background processing"""
        db = _get_db_session()
        if not db:
            return []
        
        try:
            result = db.execute(
                text("""
                    SELECT id, task_type, task_name, input_payload, created_at
                    FROM agent_tasks
                    WHERE task_type = 'notification_event'
                    AND status = 'pending'
                    ORDER BY created_at ASC
                    LIMIT :limit
                """),
                {"limit": limit}
            )
            rows = result.fetchall()
            return [dict(row._mapping) for row in rows]
        except Exception as e:
            logger.error(f"Failed to get pending events: {e}")
            return []
        finally:
            db.close()
    
    def _create_audit_log(
        self,
        action: str,
        actor_id: str,
        message_ids: List[str],
        conversation_id: Optional[str],
        timestamp: datetime
    ):
        """Create HIPAA-compliant audit log entry (sync) - proper session handling"""
        db = _get_db_session()
        if not db:
            logger.warning("No database session for audit log - HIPAA compliance issue")
            return
        
        try:
            details_dict = {
                "message_count": len(message_ids),
                "message_ids_sample": message_ids[:5] if message_ids else []
            }
            details_json = json.dumps(details_dict)
            object_id = ",".join(message_ids[:10]) if message_ids else ""
            
            db.execute(
                text("""
                    INSERT INTO agent_audit_logs (
                        actor_type, actor_id, action, object_type,
                        object_id, conversation_id, details, timestamp
                    ) VALUES (
                        :actor_type, :actor_id, :action, :object_type,
                        :object_id, :conversation_id, :details::jsonb, :timestamp
                    )
                """),
                {
                    "actor_type": "user",
                    "actor_id": actor_id,
                    "action": action,
                    "object_type": "message",
                    "object_id": object_id,
                    "conversation_id": conversation_id,
                    "details": details_json,
                    "timestamp": timestamp
                }
            )
            db.commit()
        except Exception as e:
            logger.error(f"Failed to create audit log for {action}: {e}")
            db.rollback()
        finally:
            db.close()
    
    async def _publish_delivery_event(
        self,
        event_type: str,
        message_ids: List[str],
        recipient_id: str,
        conversation_id: Optional[str],
        timestamp: datetime
    ):
        """Publish delivery event to Redis stream (best-effort)"""
        if not self._redis_stream:
            return
        
        await self._redis_stream.add_event(
            event_type=event_type,
            actor_type="user",
            actor_id=recipient_id,
            data={
                "message_ids": message_ids,
                "conversation_id": conversation_id,
                "timestamp": timestamp.isoformat()
            }
        )
        logger.debug(f"Published {event_type} event for {len(message_ids)} messages")
    
    async def _notify_senders_of_delivery(
        self,
        message_ids: List[str],
        recipient_id: str,
        delivered_at: datetime
    ):
        """Notify original senders that their messages were delivered (best-effort)"""
        if not self._connection_manager:
            return
        
        sender_messages = await self._run_in_executor(
            self._get_message_senders,
            message_ids
        )
        
        if not sender_messages:
            return
        
        messages_by_sender: Dict[str, List[str]] = {}
        for msg_id, sender_id in sender_messages.items():
            if sender_id not in messages_by_sender:
                messages_by_sender[sender_id] = []
            messages_by_sender[sender_id].append(msg_id)
        
        for sender_id, msg_ids in messages_by_sender.items():
            notification = {
                "type": "delivery_receipt",
                "payload": {
                    "messageIds": msg_ids,
                    "recipientId": recipient_id,
                    "deliveredAt": delivered_at.isoformat(),
                    "status": "delivered"
                }
            }
            await self._connection_manager.send_to_user(sender_id, notification)
            logger.debug(f"Sent delivery notification to sender {sender_id}")
    
    async def _notify_senders_of_read(
        self,
        message_ids: List[str],
        reader_id: str,
        read_at: datetime
    ):
        """Notify original senders that their messages were read (best-effort)"""
        if not self._connection_manager:
            return
        
        sender_messages = await self._run_in_executor(
            self._get_message_senders,
            message_ids
        )
        
        if not sender_messages:
            return
        
        messages_by_sender: Dict[str, List[str]] = {}
        for msg_id, sender_id in sender_messages.items():
            if sender_id not in messages_by_sender:
                messages_by_sender[sender_id] = []
            messages_by_sender[sender_id].append(msg_id)
        
        for sender_id, msg_ids in messages_by_sender.items():
            notification = {
                "type": "read_receipt",
                "payload": {
                    "messageIds": msg_ids,
                    "readerId": reader_id,
                    "readAt": read_at.isoformat(),
                    "status": "read"
                }
            }
            await self._connection_manager.send_to_user(sender_id, notification)
            logger.debug(f"Sent read receipt notification to sender {sender_id}")


_delivery_service: Optional[DeliveryService] = None


def get_delivery_service() -> DeliveryService:
    """Get or create the delivery service singleton"""
    global _delivery_service
    if _delivery_service is None:
        _delivery_service = DeliveryService()
    return _delivery_service


async def init_delivery_service(
    redis_stream = None,
    connection_manager = None
):
    """
    Initialize the delivery service with dependencies.
    
    The service will work even without Redis or WebSocket - it uses
    database persistence as the primary store and Redis/WebSocket
    for real-time notifications (best-effort).
    """
    service = get_delivery_service()
    service.set_dependencies(
        redis_stream=redis_stream,
        connection_manager=connection_manager
    )
    
    deps = []
    if redis_stream:
        deps.append("Redis")
    if connection_manager:
        deps.append("WebSocket")
    
    if deps:
        logger.info(f"Delivery service initialized with: {', '.join(deps)}")
    else:
        logger.info("Delivery service initialized (database-only mode)")
    
    return service
