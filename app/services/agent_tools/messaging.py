"""
Messaging Tool Microservice
Message sending and notification management for agents
"""

import logging
import json
from typing import Dict, Any, Optional, List
from datetime import datetime
import uuid

from app.services.agent_tools.base import BaseTool, ToolExecutionContext
from app.models.agent_models import ToolCallResult, ToolStatus

logger = logging.getLogger(__name__)


class MessagingTool(BaseTool):
    """
    Messaging tool for both Agent Clona and Assistant Lysa.
    Handles sending messages, notifications, and reminders.
    """
    
    def __init__(self):
        super().__init__()
        self.name = "messaging"
        self.display_name = "Messaging & Notifications"
        self.description = """
Send messages, notifications, and reminders to patients or healthcare providers.
Actions: send_message, send_notification, send_reminder, send_email, 
get_message_history, mark_messages_read
"""
        self.tool_type = "messaging"
        self.requires_approval = False
        self.allowed_roles = ["doctor", "patient"]
        self.required_permissions = ["messaging:send", "messaging:read"]
        self.version = 1
    
    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "send_message",
                        "send_notification",
                        "send_reminder",
                        "send_email",
                        "get_message_history",
                        "mark_messages_read"
                    ],
                    "description": "The messaging action to perform"
                },
                "recipient_id": {
                    "type": "string",
                    "description": "User ID of the message recipient"
                },
                "recipient_type": {
                    "type": "string",
                    "enum": ["patient", "doctor", "agent"],
                    "description": "Type of the recipient"
                },
                "subject": {
                    "type": "string",
                    "description": "Subject line for email or notification title"
                },
                "content": {
                    "type": "string",
                    "description": "Message content"
                },
                "priority": {
                    "type": "string",
                    "enum": ["low", "normal", "high", "urgent"],
                    "description": "Message priority level"
                },
                "reminder_time": {
                    "type": "string",
                    "description": "When to send the reminder (ISO datetime)"
                },
                "notification_type": {
                    "type": "string",
                    "enum": ["info", "warning", "alert", "success"],
                    "description": "Type of notification"
                },
                "conversation_id": {
                    "type": "string",
                    "description": "Conversation ID for message history"
                },
                "message_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Message IDs to mark as read"
                }
            },
            "required": ["action"]
        }
    
    async def execute(
        self,
        parameters: Dict[str, Any],
        context: ToolExecutionContext
    ) -> ToolCallResult:
        """Execute messaging action"""
        action = parameters.get("action")
        
        try:
            if action == "send_message":
                return await self._send_message(parameters, context)
            elif action == "send_notification":
                return await self._send_notification(parameters, context)
            elif action == "send_reminder":
                return await self._send_reminder(parameters, context)
            elif action == "send_email":
                return await self._send_email(parameters, context)
            elif action == "get_message_history":
                return await self._get_message_history(parameters, context)
            elif action == "mark_messages_read":
                return await self._mark_messages_read(parameters, context)
            else:
                return ToolCallResult(
                    tool_call_id=context.message_id,
                    tool_name=self.name,
                    status=ToolStatus.FAILED,
                    error=f"Unknown action: {action}"
                )
        except Exception as e:
            logger.error(f"Messaging tool error: {e}")
            return ToolCallResult(
                tool_call_id=context.message_id,
                tool_name=self.name,
                status=ToolStatus.FAILED,
                error=str(e)
            )
    
    async def _send_message(
        self,
        parameters: Dict[str, Any],
        context: ToolExecutionContext
    ) -> ToolCallResult:
        """Send a message to a user"""
        from app.database import SessionLocal
        from sqlalchemy import text
        
        recipient_id = parameters.get("recipient_id")
        content = parameters.get("content")
        
        if not recipient_id:
            return ToolCallResult(
                tool_call_id=context.message_id,
                tool_name=self.name,
                status=ToolStatus.FAILED,
                error="Recipient ID required"
            )
        
        if not content:
            return ToolCallResult(
                tool_call_id=context.message_id,
                tool_name=self.name,
                status=ToolStatus.FAILED,
                error="Message content required"
            )
        
        message_id = str(uuid.uuid4())
        priority = parameters.get("priority", "normal")
        
        db = SessionLocal()
        try:
            db.execute(
                text("""
                    INSERT INTO agent_messages (
                        id, msg_id, conversation_id, from_type, from_id,
                        to_json, message_type, content, priority,
                        delivered, contains_phi, created_at
                    ) VALUES (
                        :id, :msg_id, :conversation_id, :from_type, :from_id,
                        :to_json::jsonb, :message_type, :content, :priority,
                        false, false, NOW()
                    )
                """),
                {
                    "id": message_id,
                    "msg_id": message_id,
                    "conversation_id": context.conversation_id,
                    "from_type": "agent",
                    "from_id": context.agent_id,
                    "to_json": json.dumps([{"type": "user", "id": recipient_id}]),
                    "message_type": "chat",
                    "content": content,
                    "priority": priority
                }
            )
            db.commit()
            
            return ToolCallResult(
                tool_call_id=context.message_id,
                tool_name=self.name,
                status=ToolStatus.COMPLETED,
                result={
                    "message_id": message_id,
                    "recipient_id": recipient_id,
                    "content_preview": content[:100] + "..." if len(content) > 100 else content,
                    "priority": priority,
                    "status": "sent",
                    "message": "Message sent successfully"
                }
            )
        finally:
            db.close()
    
    async def _send_notification(
        self,
        parameters: Dict[str, Any],
        context: ToolExecutionContext
    ) -> ToolCallResult:
        """Send a notification to a user"""
        from app.database import SessionLocal
        from sqlalchemy import text
        
        recipient_id = parameters.get("recipient_id")
        subject = parameters.get("subject", "Notification")
        content = parameters.get("content")
        notification_type = parameters.get("notification_type", "info")
        
        if not recipient_id:
            return ToolCallResult(
                tool_call_id=context.message_id,
                tool_name=self.name,
                status=ToolStatus.FAILED,
                error="Recipient ID required"
            )
        
        if not content:
            return ToolCallResult(
                tool_call_id=context.message_id,
                tool_name=self.name,
                status=ToolStatus.FAILED,
                error="Notification content required"
            )
        
        notification_id = str(uuid.uuid4())
        
        db = SessionLocal()
        try:
            db.execute(
                text("""
                    INSERT INTO notifications (
                        id, user_id, title, message, type,
                        read, created_at
                    ) VALUES (
                        :id, :user_id, :title, :message, :type,
                        false, NOW()
                    )
                """),
                {
                    "id": notification_id,
                    "user_id": recipient_id,
                    "title": subject,
                    "message": content,
                    "type": notification_type
                }
            )
            db.commit()
            
            return ToolCallResult(
                tool_call_id=context.message_id,
                tool_name=self.name,
                status=ToolStatus.COMPLETED,
                result={
                    "notification_id": notification_id,
                    "recipient_id": recipient_id,
                    "title": subject,
                    "type": notification_type,
                    "message": "Notification sent successfully"
                }
            )
        finally:
            db.close()
    
    async def _send_reminder(
        self,
        parameters: Dict[str, Any],
        context: ToolExecutionContext
    ) -> ToolCallResult:
        """Schedule a reminder for a user"""
        from app.database import SessionLocal
        from sqlalchemy import text
        
        recipient_id = parameters.get("recipient_id")
        content = parameters.get("content")
        reminder_time = parameters.get("reminder_time")
        
        if not recipient_id:
            return ToolCallResult(
                tool_call_id=context.message_id,
                tool_name=self.name,
                status=ToolStatus.FAILED,
                error="Recipient ID required"
            )
        
        if not content:
            return ToolCallResult(
                tool_call_id=context.message_id,
                tool_name=self.name,
                status=ToolStatus.FAILED,
                error="Reminder content required"
            )
        
        if not reminder_time:
            return ToolCallResult(
                tool_call_id=context.message_id,
                tool_name=self.name,
                status=ToolStatus.FAILED,
                error="Reminder time required"
            )
        
        try:
            reminder_dt = datetime.fromisoformat(reminder_time.replace('Z', '+00:00'))
        except ValueError:
            return ToolCallResult(
                tool_call_id=context.message_id,
                tool_name=self.name,
                status=ToolStatus.FAILED,
                error="Invalid datetime format for reminder_time"
            )
        
        task_id = str(uuid.uuid4())
        
        db = SessionLocal()
        try:
            db.execute(
                text("""
                    INSERT INTO agent_tasks (
                        id, agent_id, user_id, task_type, task_name,
                        scheduled_at, status, priority, input_payload, created_at
                    ) VALUES (
                        :id, :agent_id, :user_id, :task_type, :task_name,
                        :scheduled_at, 'pending', 5, :input_payload::jsonb, NOW()
                    )
                """),
                {
                    "id": task_id,
                    "agent_id": context.agent_id,
                    "user_id": recipient_id,
                    "task_type": "scheduled_reminder",
                    "task_name": f"Reminder: {content[:50]}",
                    "scheduled_at": reminder_dt,
                    "input_payload": json.dumps({
                        "recipient_id": recipient_id,
                        "content": content,
                        "created_by": context.user_id
                    })
                }
            )
            db.commit()
            
            return ToolCallResult(
                tool_call_id=context.message_id,
                tool_name=self.name,
                status=ToolStatus.COMPLETED,
                result={
                    "task_id": task_id,
                    "recipient_id": recipient_id,
                    "scheduled_for": reminder_dt.isoformat(),
                    "content_preview": content[:100] + "..." if len(content) > 100 else content,
                    "message": f"Reminder scheduled for {reminder_dt.strftime('%B %d, %Y at %I:%M %p')}"
                }
            )
        finally:
            db.close()
    
    async def _send_email(
        self,
        parameters: Dict[str, Any],
        context: ToolExecutionContext
    ) -> ToolCallResult:
        """Send an email to a user"""
        from app.database import SessionLocal
        from sqlalchemy import text
        
        recipient_id = parameters.get("recipient_id")
        subject = parameters.get("subject")
        content = parameters.get("content")
        
        if not recipient_id:
            return ToolCallResult(
                tool_call_id=context.message_id,
                tool_name=self.name,
                status=ToolStatus.FAILED,
                error="Recipient ID required"
            )
        
        if not subject or not content:
            return ToolCallResult(
                tool_call_id=context.message_id,
                tool_name=self.name,
                status=ToolStatus.FAILED,
                error="Email subject and content required"
            )
        
        db = SessionLocal()
        try:
            result = db.execute(
                text("SELECT email FROM users WHERE id = :user_id"),
                {"user_id": recipient_id}
            )
            row = result.fetchone()
            if not row or not row[0]:
                return ToolCallResult(
                    tool_call_id=context.message_id,
                    tool_name=self.name,
                    status=ToolStatus.FAILED,
                    error="Recipient email not found"
                )
            
            recipient_email = row[0]
            
            task_id = str(uuid.uuid4())
            db.execute(
                text("""
                    INSERT INTO agent_tasks (
                        id, agent_id, user_id, task_type, task_name,
                        status, priority, input_payload, created_at
                    ) VALUES (
                        :id, :agent_id, :user_id, :task_type, :task_name,
                        'pending', 3, :input_payload::jsonb, NOW()
                    )
                """),
                {
                    "id": task_id,
                    "agent_id": context.agent_id,
                    "user_id": recipient_id,
                    "task_type": "send_email",
                    "task_name": f"Email: {subject[:50]}",
                    "input_payload": json.dumps({
                        "recipient_email": recipient_email,
                        "recipient_id": recipient_id,
                        "subject": subject,
                        "content": content,
                        "created_by": context.user_id
                    })
                }
            )
            db.commit()
            
            return ToolCallResult(
                tool_call_id=context.message_id,
                tool_name=self.name,
                status=ToolStatus.COMPLETED,
                result={
                    "task_id": task_id,
                    "recipient_id": recipient_id,
                    "recipient_email": recipient_email[:3] + "***" + recipient_email[-10:],
                    "subject": subject,
                    "status": "queued",
                    "message": "Email queued for delivery"
                }
            )
        finally:
            db.close()
    
    async def _get_message_history(
        self,
        parameters: Dict[str, Any],
        context: ToolExecutionContext
    ) -> ToolCallResult:
        """Get message history for a conversation"""
        from app.database import SessionLocal
        from sqlalchemy import text
        
        conversation_id = parameters.get("conversation_id") or context.conversation_id
        
        if not conversation_id:
            return ToolCallResult(
                tool_call_id=context.message_id,
                tool_name=self.name,
                status=ToolStatus.FAILED,
                error="Conversation ID required"
            )
        
        db = SessionLocal()
        try:
            result = db.execute(
                text("""
                    SELECT id, from_type, from_id, content, created_at, delivered, read_at
                    FROM agent_messages
                    WHERE conversation_id = :conversation_id
                    ORDER BY created_at DESC
                    LIMIT 50
                """),
                {"conversation_id": conversation_id}
            )
            
            messages = []
            for row in result.fetchall():
                messages.append({
                    "message_id": row[0],
                    "from_type": row[1],
                    "from_id": row[2],
                    "content_preview": (row[3][:100] + "...") if row[3] and len(row[3]) > 100 else row[3],
                    "created_at": row[4].isoformat() if row[4] else None,
                    "delivered": row[5],
                    "read": row[6] is not None
                })
            
            return ToolCallResult(
                tool_call_id=context.message_id,
                tool_name=self.name,
                status=ToolStatus.COMPLETED,
                result={
                    "conversation_id": conversation_id,
                    "messages": messages,
                    "count": len(messages),
                    "message": f"Retrieved {len(messages)} messages from conversation"
                }
            )
        finally:
            db.close()
    
    async def _mark_messages_read(
        self,
        parameters: Dict[str, Any],
        context: ToolExecutionContext
    ) -> ToolCallResult:
        """Mark messages as read"""
        from app.database import SessionLocal
        from sqlalchemy import text
        
        message_ids = parameters.get("message_ids", [])
        
        if not message_ids:
            return ToolCallResult(
                tool_call_id=context.message_id,
                tool_name=self.name,
                status=ToolStatus.FAILED,
                error="Message IDs required"
            )
        
        db = SessionLocal()
        try:
            result = db.execute(
                text("""
                    UPDATE agent_messages
                    SET read_at = NOW()
                    WHERE id = ANY(:message_ids)
                    AND read_at IS NULL
                    RETURNING id
                """),
                {"message_ids": message_ids}
            )
            
            updated_ids = [row[0] for row in result.fetchall()]
            db.commit()
            
            return ToolCallResult(
                tool_call_id=context.message_id,
                tool_name=self.name,
                status=ToolStatus.COMPLETED,
                result={
                    "marked_read": updated_ids,
                    "count": len(updated_ids),
                    "message": f"Marked {len(updated_ids)} messages as read"
                }
            )
        finally:
            db.close()
