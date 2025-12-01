"""
Redis Stream Service for Agent Communication
Provides pub/sub messaging, stream processing, and real-time event distribution
"""

import os
import json
import logging
import asyncio
from typing import Optional, List, Dict, Any, Callable, Awaitable
from datetime import datetime
import uuid

try:
    import redis.asyncio as aioredis
    from redis.exceptions import ResponseError as RedisResponseError
    REDIS_AVAILABLE = True
except ImportError:
    aioredis = None
    RedisResponseError = Exception
    REDIS_AVAILABLE = False

logger = logging.getLogger(__name__)


class RedisStreamService:
    """
    Redis Stream-based messaging service for multi-agent communication
    
    Features:
    - Message streams for agent-to-agent, user-to-agent communication
    - Pub/Sub for real-time presence and events
    - Consumer groups for reliable message processing
    - Message acknowledgment and retry handling
    """
    
    # Stream names
    STREAM_MESSAGES = "agent:messages"
    STREAM_EVENTS = "agent:events"
    STREAM_TOOL_CALLS = "agent:tool_calls"
    STREAM_TOOL_RESULTS = "agent:tool_results"
    
    # Pub/Sub channels
    CHANNEL_PRESENCE = "agent:presence"
    CHANNEL_NOTIFICATIONS = "agent:notifications"
    
    # Consumer groups
    GROUP_AGENT_WORKERS = "agent_workers"
    GROUP_TOOL_WORKERS = "tool_workers"
    GROUP_NOTIFICATION_WORKERS = "notification_workers"
    
    def __init__(self):
        self._redis_client = None  # aioredis.Redis when connected
        self._pubsub = None  # aioredis.client.PubSub when subscribed
        self._initialized = False
        self._subscriptions: Dict[str, List[Callable]] = {}
        self._consumer_name = f"consumer_{uuid.uuid4().hex[:8]}"
        self._fallback_queues: Dict[str, List[Dict]] = {}
        
    async def initialize(self) -> bool:
        """Initialize Redis connection and stream infrastructure"""
        if self._initialized:
            return self._redis_client is not None
            
        redis_url = os.getenv("REDIS_URL")
        
        if redis_url and REDIS_AVAILABLE and aioredis:
            try:
                self._redis_client = aioredis.from_url(
                    redis_url,
                    decode_responses=True,
                    socket_timeout=5.0,
                    socket_connect_timeout=5.0
                )
                
                ping_result = await self._redis_client.ping()
                if ping_result:
                    logger.info("Redis Stream Service connected")
                    
                    await self._setup_streams()
                    await self._setup_consumer_groups()
                    
                    self._pubsub = self._redis_client.pubsub()
                    self._initialized = True
                    return True
                    
            except Exception as e:
                logger.warning(f"Redis not available, using fallback mode: {e}")
                self._redis_client = None
        else:
            logger.info("Redis not configured, using in-memory fallback")
        
        self._initialized = True
        return False
    
    async def _setup_streams(self):
        """Create streams if they don't exist"""
        streams = [
            self.STREAM_MESSAGES,
            self.STREAM_EVENTS,
            self.STREAM_TOOL_CALLS,
            self.STREAM_TOOL_RESULTS
        ]
        
        for stream in streams:
            try:
                await self._redis_client.xinfo_stream(stream)
            except RedisResponseError:
                await self._redis_client.xadd(stream, {"init": "true"}, id="0-1")
                logger.info(f"Created stream: {stream}")
    
    async def _setup_consumer_groups(self):
        """Set up consumer groups for reliable processing"""
        groups = [
            (self.STREAM_MESSAGES, self.GROUP_AGENT_WORKERS),
            (self.STREAM_TOOL_CALLS, self.GROUP_TOOL_WORKERS),
            (self.STREAM_EVENTS, self.GROUP_NOTIFICATION_WORKERS)
        ]
        
        for stream, group in groups:
            try:
                await self._redis_client.xgroup_create(
                    stream, group, id="0", mkstream=True
                )
                logger.info(f"Created consumer group: {group} on {stream}")
            except RedisResponseError as e:
                if "BUSYGROUP" not in str(e):
                    logger.warning(f"Failed to create consumer group {group}: {e}")

    async def close(self):
        """Clean up connections"""
        if self._pubsub:
            await self._pubsub.close()
        if self._redis_client:
            await self._redis_client.close()
        self._initialized = False
        logger.info("Redis Stream Service closed")

    async def add_message(
        self,
        msg_id: str,
        sender_type: str,
        sender_id: str,
        recipient_type: str,
        recipient_id: str,
        message_type: str,
        payload: Dict[str, Any],
        conversation_id: Optional[str] = None,
        max_len: int = 10000
    ) -> Optional[str]:
        """
        Add a message to the message stream
        
        Returns: stream entry ID or None if failed
        """
        message_data = {
            "msg_id": msg_id,
            "sender_type": sender_type,
            "sender_id": sender_id,
            "recipient_type": recipient_type,
            "recipient_id": recipient_id,
            "message_type": message_type,
            "payload": json.dumps(payload),
            "conversation_id": conversation_id or "",
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if self._redis_client:
            try:
                entry_id = await self._redis_client.xadd(
                    self.STREAM_MESSAGES,
                    message_data,
                    maxlen=max_len
                )
                logger.debug(f"Added message {msg_id} to stream, entry_id: {entry_id}")
                return entry_id
            except Exception as e:
                logger.error(f"Failed to add message to stream: {e}")
                return None
        else:
            if self.STREAM_MESSAGES not in self._fallback_queues:
                self._fallback_queues[self.STREAM_MESSAGES] = []
            self._fallback_queues[self.STREAM_MESSAGES].append(message_data)
            return f"fallback_{msg_id}"

    async def add_tool_call(
        self,
        tool_call_id: str,
        agent_id: str,
        tool_name: str,
        parameters: Dict[str, Any],
        requires_approval: bool = False,
        user_id: Optional[str] = None,
        priority: str = "normal"
    ) -> Optional[str]:
        """
        Add a tool call to the tool calls stream for worker processing
        """
        tool_data = {
            "tool_call_id": tool_call_id,
            "agent_id": agent_id,
            "tool_name": tool_name,
            "parameters": json.dumps(parameters),
            "requires_approval": str(requires_approval),
            "user_id": user_id or "",
            "priority": priority,
            "status": "pending",
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if self._redis_client:
            try:
                entry_id = await self._redis_client.xadd(
                    self.STREAM_TOOL_CALLS,
                    tool_data,
                    maxlen=5000
                )
                logger.debug(f"Added tool call {tool_call_id} to stream")
                return entry_id
            except Exception as e:
                logger.error(f"Failed to add tool call to stream: {e}")
                return None
        else:
            if self.STREAM_TOOL_CALLS not in self._fallback_queues:
                self._fallback_queues[self.STREAM_TOOL_CALLS] = []
            self._fallback_queues[self.STREAM_TOOL_CALLS].append(tool_data)
            return f"fallback_{tool_call_id}"

    async def add_tool_result(
        self,
        tool_call_id: str,
        result: Dict[str, Any],
        status: str = "completed",
        error: Optional[str] = None
    ) -> Optional[str]:
        """
        Add a tool execution result
        """
        result_data = {
            "tool_call_id": tool_call_id,
            "result": json.dumps(result),
            "status": status,
            "error": error or "",
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if self._redis_client:
            try:
                entry_id = await self._redis_client.xadd(
                    self.STREAM_TOOL_RESULTS,
                    result_data,
                    maxlen=5000
                )
                return entry_id
            except Exception as e:
                logger.error(f"Failed to add tool result to stream: {e}")
                return None
        else:
            if self.STREAM_TOOL_RESULTS not in self._fallback_queues:
                self._fallback_queues[self.STREAM_TOOL_RESULTS] = []
            self._fallback_queues[self.STREAM_TOOL_RESULTS].append(result_data)
            return f"fallback_{tool_call_id}"

    async def add_event(
        self,
        event_type: str,
        actor_type: str,
        actor_id: str,
        data: Dict[str, Any],
        target_users: Optional[List[str]] = None
    ) -> Optional[str]:
        """
        Add a system event (presence, status changes, notifications)
        """
        event_data = {
            "event_type": event_type,
            "actor_type": actor_type,
            "actor_id": actor_id,
            "data": json.dumps(data),
            "target_users": json.dumps(target_users or []),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if self._redis_client:
            try:
                entry_id = await self._redis_client.xadd(
                    self.STREAM_EVENTS,
                    event_data,
                    maxlen=5000
                )
                
                if event_type in ["presence_online", "presence_offline"]:
                    await self._redis_client.publish(
                        self.CHANNEL_PRESENCE,
                        json.dumps(event_data)
                    )
                    
                return entry_id
            except Exception as e:
                logger.error(f"Failed to add event to stream: {e}")
                return None
        else:
            if self.STREAM_EVENTS not in self._fallback_queues:
                self._fallback_queues[self.STREAM_EVENTS] = []
            self._fallback_queues[self.STREAM_EVENTS].append(event_data)
            return f"fallback_event_{datetime.utcnow().timestamp()}"

    async def read_messages(
        self,
        last_id: str = "0",
        count: int = 100,
        block_ms: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Read messages from the stream (for simple reading, not consumer group)
        """
        if not self._redis_client:
            return self._fallback_queues.get(self.STREAM_MESSAGES, [])[-count:]
            
        try:
            entries = await self._redis_client.xread(
                {self.STREAM_MESSAGES: last_id},
                count=count,
                block=block_ms
            )
            
            messages = []
            if entries:
                for stream_name, stream_entries in entries:
                    for entry_id, data in stream_entries:
                        data["entry_id"] = entry_id
                        if "payload" in data:
                            data["payload"] = json.loads(data["payload"])
                        messages.append(data)
            
            return messages
        except Exception as e:
            logger.error(f"Failed to read messages: {e}")
            return []

    async def consume_messages(
        self,
        group: str,
        count: int = 10,
        block_ms: int = 5000
    ) -> List[Dict[str, Any]]:
        """
        Consume messages from a consumer group (reliable processing)
        """
        if not self._redis_client:
            queue = self._fallback_queues.get(self.STREAM_MESSAGES, [])
            return queue[:count] if queue else []
            
        try:
            entries = await self._redis_client.xreadgroup(
                group,
                self._consumer_name,
                {self.STREAM_MESSAGES: ">"},
                count=count,
                block=block_ms
            )
            
            messages = []
            if entries:
                for stream_name, stream_entries in entries:
                    for entry_id, data in stream_entries:
                        data["entry_id"] = entry_id
                        if "payload" in data:
                            data["payload"] = json.loads(data["payload"])
                        messages.append(data)
            
            return messages
        except Exception as e:
            logger.error(f"Failed to consume messages: {e}")
            return []

    async def consume_tool_calls(
        self,
        count: int = 10,
        block_ms: int = 5000
    ) -> List[Dict[str, Any]]:
        """
        Consume tool calls from the tool worker queue
        """
        if not self._redis_client:
            queue = self._fallback_queues.get(self.STREAM_TOOL_CALLS, [])
            return queue[:count] if queue else []
            
        try:
            entries = await self._redis_client.xreadgroup(
                self.GROUP_TOOL_WORKERS,
                self._consumer_name,
                {self.STREAM_TOOL_CALLS: ">"},
                count=count,
                block=block_ms
            )
            
            tool_calls = []
            if entries:
                for stream_name, stream_entries in entries:
                    for entry_id, data in stream_entries:
                        data["entry_id"] = entry_id
                        if "parameters" in data:
                            data["parameters"] = json.loads(data["parameters"])
                        data["requires_approval"] = data.get("requires_approval") == "True"
                        tool_calls.append(data)
            
            return tool_calls
        except Exception as e:
            logger.error(f"Failed to consume tool calls: {e}")
            return []

    async def ack_message(self, stream: str, group: str, entry_id: str) -> bool:
        """
        Acknowledge a message has been processed
        """
        if not self._redis_client:
            return True
            
        try:
            result = await self._redis_client.xack(stream, group, entry_id)
            return result > 0
        except Exception as e:
            logger.error(f"Failed to ack message: {e}")
            return False

    async def subscribe(
        self,
        channel: str,
        handler: Callable[[str, Dict], Awaitable[None]]
    ):
        """
        Subscribe to a pub/sub channel
        """
        if channel not in self._subscriptions:
            self._subscriptions[channel] = []
        self._subscriptions[channel].append(handler)
        
        if self._pubsub:
            await self._pubsub.subscribe(channel)
            logger.info(f"Subscribed to channel: {channel}")

    async def publish(self, channel: str, data: Dict[str, Any]) -> int:
        """
        Publish a message to a channel
        """
        if self._redis_client:
            try:
                return await self._redis_client.publish(channel, json.dumps(data))
            except Exception as e:
                logger.error(f"Failed to publish to {channel}: {e}")
                return 0
        else:
            if channel in self._subscriptions:
                for handler in self._subscriptions[channel]:
                    asyncio.create_task(handler(channel, data))
            return len(self._subscriptions.get(channel, []))

    async def set_presence(
        self,
        user_id: str,
        is_online: bool,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Update user presence status
        """
        presence_data = {
            "user_id": user_id,
            "is_online": is_online,
            "last_seen": datetime.utcnow().isoformat(),
            "metadata": metadata or {}
        }
        
        if self._redis_client:
            try:
                presence_key = f"presence:{user_id}"
                
                if is_online:
                    await self._redis_client.hset(presence_key, mapping={
                        "is_online": "true",
                        "last_seen": presence_data["last_seen"],
                        "metadata": json.dumps(metadata or {})
                    })
                    await self._redis_client.expire(presence_key, 300)
                else:
                    await self._redis_client.hset(presence_key, mapping={
                        "is_online": "false",
                        "last_seen": presence_data["last_seen"]
                    })
                
                await self.publish(self.CHANNEL_PRESENCE, {
                    "event": "presence_online" if is_online else "presence_offline",
                    "user_id": user_id,
                    "timestamp": presence_data["last_seen"]
                })
                
            except Exception as e:
                logger.error(f"Failed to set presence: {e}")

    async def get_presence(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Get user presence status
        """
        if not self._redis_client:
            return None
            
        try:
            presence_key = f"presence:{user_id}"
            data = await self._redis_client.hgetall(presence_key)
            
            if data:
                return {
                    "user_id": user_id,
                    "is_online": data.get("is_online") == "true",
                    "last_seen": data.get("last_seen"),
                    "metadata": json.loads(data.get("metadata", "{}"))
                }
            return None
        except Exception as e:
            logger.error(f"Failed to get presence: {e}")
            return None

    async def get_online_users(self) -> List[str]:
        """
        Get list of currently online users
        """
        if not self._redis_client:
            return []
            
        try:
            online = []
            async for key in self._redis_client.scan_iter(match="presence:*"):
                data = await self._redis_client.hgetall(key)
                if data.get("is_online") == "true":
                    user_id = key.replace("presence:", "")
                    online.append(user_id)
            return online
        except Exception as e:
            logger.error(f"Failed to get online users: {e}")
            return []

    async def heartbeat(self, user_id: str):
        """
        Update presence heartbeat (extends TTL)
        """
        if self._redis_client:
            try:
                presence_key = f"presence:{user_id}"
                await self._redis_client.expire(presence_key, 300)
                await self._redis_client.hset(
                    presence_key,
                    "last_seen",
                    datetime.utcnow().isoformat()
                )
            except Exception as e:
                logger.error(f"Failed to update heartbeat: {e}")

    async def get_pending_messages(
        self,
        stream: str,
        group: str,
        count: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get pending (unacknowledged) messages in a consumer group
        """
        if not self._redis_client:
            return []
            
        try:
            pending = await self._redis_client.xpending_range(
                stream,
                group,
                min="-",
                max="+",
                count=count
            )
            
            return [
                {
                    "entry_id": p["message_id"],
                    "consumer": p["consumer"],
                    "time_since_delivered": p["time_since_delivered"],
                    "times_delivered": p["times_delivered"]
                }
                for p in pending
            ]
        except Exception as e:
            logger.error(f"Failed to get pending messages: {e}")
            return []

    async def claim_stale_messages(
        self,
        stream: str,
        group: str,
        min_idle_time_ms: int = 60000,
        count: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Claim messages that have been pending too long (for retry)
        """
        if not self._redis_client:
            return []
            
        try:
            result = await self._redis_client.xautoclaim(
                stream,
                group,
                self._consumer_name,
                min_idle_time=min_idle_time_ms,
                count=count
            )
            
            claimed = []
            if result and len(result) > 1:
                for entry_id, data in result[1]:
                    if data:
                        data["entry_id"] = entry_id
                        claimed.append(data)
            
            return claimed
        except Exception as e:
            logger.error(f"Failed to claim stale messages: {e}")
            return []


_redis_stream_service: Optional[RedisStreamService] = None


async def get_redis_stream_service() -> RedisStreamService:
    """Get or create the Redis stream service singleton"""
    global _redis_stream_service
    
    if _redis_stream_service is None:
        _redis_stream_service = RedisStreamService()
        await _redis_stream_service.initialize()
    
    return _redis_stream_service


async def shutdown_redis_stream_service():
    """Shutdown the Redis stream service"""
    global _redis_stream_service
    
    if _redis_stream_service:
        await _redis_stream_service.close()
        _redis_stream_service = None
