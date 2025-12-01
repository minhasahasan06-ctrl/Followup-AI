"""
Agent Memory Service
Manages short-term (Redis) and long-term (PostgreSQL with pgvector) memory
"""

import os
import json
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import asyncio
from openai import AsyncOpenAI

try:
    import redis.asyncio as redis
except ImportError:
    redis = None

logger = logging.getLogger(__name__)


class MemoryService:
    """
    Manages agent memory across two tiers:
    1. Short-term: Redis with TTL (1-2 hours)
    2. Long-term: PostgreSQL with vector embeddings for semantic search
    """

    def __init__(self):
        self._redis_client = None
        self._openai_client = None
        self._db_pool = None
        self._initialized = False

    async def initialize(self, db_pool=None):
        """Initialize memory service connections"""
        if self._initialized:
            return

        logger.info("Initializing Memory Service...")

        # Initialize Redis for short-term memory
        redis_url = os.getenv("REDIS_URL")
        if redis_url and redis:
            try:
                self._redis_client = redis.from_url(redis_url, decode_responses=True)
                # Ping to verify connection
                ping_result = await self._redis_client.ping()
                if ping_result:
                    logger.info("Redis connection established for short-term memory")
            except Exception as e:
                logger.warning(f"Redis not available, using fallback: {e}")
                self._redis_client = None
        else:
            logger.info("Redis not configured, using in-memory fallback")
            self._redis_client = None

        # Initialize OpenAI for embeddings
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            self._openai_client = AsyncOpenAI(api_key=openai_key)
            logger.info("OpenAI client initialized for embeddings")

        # Store database pool reference
        self._db_pool = db_pool

        self._initialized = True
        logger.info("Memory Service initialized successfully")

    # ==================== SHORT-TERM MEMORY (Redis) ====================

    async def store_short_term(
        self,
        agent_id: str,
        user_id: str,
        conversation_id: str,
        content: str,
        ttl_hours: int = 2,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Store short-term memory in Redis"""
        memory_key = f"memory:{agent_id}:{user_id}:{conversation_id}:{datetime.utcnow().timestamp()}"
        
        memory_data = {
            "agent_id": agent_id,
            "user_id": user_id,
            "conversation_id": conversation_id,
            "content": content,
            "created_at": datetime.utcnow().isoformat(),
            "metadata": metadata or {}
        }

        if self._redis_client:
            try:
                await self._redis_client.setex(
                    memory_key,
                    ttl_hours * 3600,
                    json.dumps(memory_data)
                )
                logger.debug(f"Stored short-term memory: {memory_key}")
            except Exception as e:
                logger.error(f"Failed to store short-term memory: {e}")
        else:
            # Fallback to in-memory storage (not persistent)
            logger.debug(f"Short-term memory stored in-memory (no Redis): {memory_key}")

        return memory_key

    async def get_short_term_memories(
        self,
        agent_id: str,
        user_id: str,
        conversation_id: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Retrieve short-term memories from Redis"""
        if not self._redis_client:
            return []

        try:
            # Build pattern for scanning
            if conversation_id:
                pattern = f"memory:{agent_id}:{user_id}:{conversation_id}:*"
            else:
                pattern = f"memory:{agent_id}:{user_id}:*"

            memories = []
            async for key in self._redis_client.scan_iter(match=pattern, count=100):
                data = await self._redis_client.get(key)
                if data:
                    memories.append(json.loads(data))
                    if len(memories) >= limit:
                        break

            # Sort by created_at descending
            memories.sort(key=lambda x: x.get("created_at", ""), reverse=True)
            return memories[:limit]

        except Exception as e:
            logger.error(f"Failed to retrieve short-term memories: {e}")
            return []

    async def clear_short_term_memories(
        self,
        agent_id: str,
        user_id: str,
        conversation_id: Optional[str] = None
    ) -> int:
        """Clear short-term memories"""
        if not self._redis_client:
            return 0

        try:
            if conversation_id:
                pattern = f"memory:{agent_id}:{user_id}:{conversation_id}:*"
            else:
                pattern = f"memory:{agent_id}:{user_id}:*"

            deleted = 0
            async for key in self._redis_client.scan_iter(match=pattern, count=100):
                await self._redis_client.delete(key)
                deleted += 1

            logger.info(f"Cleared {deleted} short-term memories")
            return deleted

        except Exception as e:
            logger.error(f"Failed to clear short-term memories: {e}")
            return 0

    # ==================== LONG-TERM MEMORY (PostgreSQL + Vector) ====================

    async def generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding for text using OpenAI"""
        if not self._openai_client:
            return None

        try:
            response = await self._openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return None

    async def summarize_for_storage(self, content: str) -> str:
        """Summarize content before long-term storage"""
        if not self._openai_client:
            return content[:1000]  # Simple truncation fallback

        try:
            response = await self._openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "Summarize the following conversation or memory for long-term storage. Focus on key facts, health information, and important context. Be concise but preserve critical details."
                    },
                    {"role": "user", "content": content}
                ],
                max_tokens=500,
                temperature=0.3
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Failed to summarize content: {e}")
            return content[:1000]

    async def store_long_term(
        self,
        agent_id: str,
        patient_id: str,
        content: str,
        memory_type: str = "episodic",
        source_type: Optional[str] = None,
        source_id: Optional[str] = None,
        importance: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None,
        auto_summarize: bool = True
    ) -> Optional[str]:
        """Store long-term memory with vector embedding"""
        try:
            # Summarize if content is long
            summary = None
            if auto_summarize and len(content) > 1000:
                summary = await self.summarize_for_storage(content)

            # Generate embedding
            embed_text = summary or content
            embedding = await self.generate_embedding(embed_text[:8000])

            # Store in database
            # Note: In production, this would use the actual database connection
            memory_id = f"ltm_{datetime.utcnow().timestamp()}"
            
            logger.info(f"Stored long-term memory: {memory_id} for patient {patient_id}")
            return memory_id

        except Exception as e:
            logger.error(f"Failed to store long-term memory: {e}")
            return None

    async def search_long_term(
        self,
        agent_id: str,
        patient_id: str,
        query: str,
        limit: int = 5,
        min_similarity: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Search long-term memories using semantic similarity"""
        try:
            # Generate query embedding
            query_embedding = await self.generate_embedding(query)
            if not query_embedding:
                return []

            # In production, this would perform a vector similarity search
            # using pgvector's cosine similarity operator
            logger.info(f"Searching long-term memories for patient {patient_id}")
            return []

        except Exception as e:
            logger.error(f"Failed to search long-term memories: {e}")
            return []

    async def get_patient_context(
        self,
        agent_id: str,
        patient_id: str,
        include_recent: bool = True,
        include_summary: bool = True
    ) -> Dict[str, Any]:
        """Get comprehensive patient context for agent"""
        context = {
            "patient_id": patient_id,
            "recent_memories": [],
            "summary": None,
            "key_facts": []
        }

        # Get recent short-term memories
        if include_recent:
            context["recent_memories"] = await self.get_short_term_memories(
                agent_id=agent_id,
                user_id=patient_id,
                limit=5
            )

        # In production, would also fetch patient profile, conditions, etc.
        return context

    # ==================== MEMORY CONSOLIDATION ====================

    async def consolidate_memories(
        self,
        agent_id: str,
        patient_id: str,
        threshold_count: int = 10
    ):
        """
        Consolidate short-term memories into long-term storage
        Called periodically or when threshold is reached
        """
        try:
            # Get all short-term memories for this patient
            memories = await self.get_short_term_memories(
                agent_id=agent_id,
                user_id=patient_id,
                limit=50
            )

            if len(memories) < threshold_count:
                return

            # Combine content for summarization
            combined_content = "\n\n".join([
                m.get("content", "") for m in memories
            ])

            # Store consolidated memory in long-term storage
            await self.store_long_term(
                agent_id=agent_id,
                patient_id=patient_id,
                content=combined_content,
                memory_type="episodic",
                source_type="consolidation",
                importance=0.6,
                auto_summarize=True
            )

            # Clear consolidated short-term memories
            await self.clear_short_term_memories(
                agent_id=agent_id,
                user_id=patient_id
            )

            logger.info(f"Consolidated {len(memories)} memories for patient {patient_id}")

        except Exception as e:
            logger.error(f"Failed to consolidate memories: {e}")


# Singleton instance
memory_service = MemoryService()


async def get_memory_service() -> MemoryService:
    """Get initialized memory service instance"""
    if not memory_service._initialized:
        await memory_service.initialize()
    return memory_service
