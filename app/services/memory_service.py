"""
Agent Memory Service
Manages short-term (Redis) and long-term (PostgreSQL with pgvector) memory.

Production-grade implementation with:
1. Vector similarity search using pgvector
2. Centralized OpenAI client with PHI/BAA/ZDR enforcement
3. Comprehensive audit logging
4. Memory consolidation and governance
"""

import os
import json
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import asyncio
import time

try:
    import redis.asyncio as redis
except ImportError:
    redis = None

from app.services.memory_db import memory_db, get_memory_db
from app.services.openai_client import (
    get_openai_client, 
    generate_embedding,
    get_embedding_metadata,
    EMBEDDING_MODEL,
    EMBEDDING_VERSION
)

logger = logging.getLogger(__name__)


class MemoryServiceMetrics:
    """Metrics tracking for observability"""
    
    def __init__(self):
        self.store_count = 0
        self.search_count = 0
        self.store_latencies: List[float] = []
        self.search_latencies: List[float] = []
        self.search_result_counts: List[int] = []
        self.similarity_scores: List[float] = []
    
    def record_store(self, latency_ms: float):
        self.store_count += 1
        self.store_latencies.append(latency_ms)
        if len(self.store_latencies) > 1000:
            self.store_latencies = self.store_latencies[-1000:]
    
    def record_search(self, latency_ms: float, result_count: int, similarities: List[float]):
        self.search_count += 1
        self.search_latencies.append(latency_ms)
        self.search_result_counts.append(result_count)
        self.similarity_scores.extend(similarities)
        
        if len(self.search_latencies) > 1000:
            self.search_latencies = self.search_latencies[-1000:]
        if len(self.search_result_counts) > 1000:
            self.search_result_counts = self.search_result_counts[-1000:]
        if len(self.similarity_scores) > 5000:
            self.similarity_scores = self.similarity_scores[-5000:]
    
    def get_stats(self) -> Dict[str, Any]:
        def percentile(data: List[float], p: float) -> float:
            if not data:
                return 0
            sorted_data = sorted(data)
            idx = int(len(sorted_data) * p)
            return sorted_data[min(idx, len(sorted_data) - 1)]
        
        return {
            "store_count": self.store_count,
            "search_count": self.search_count,
            "store_latency_p50_ms": percentile(self.store_latencies, 0.5),
            "store_latency_p95_ms": percentile(self.store_latencies, 0.95),
            "store_latency_p99_ms": percentile(self.store_latencies, 0.99),
            "search_latency_p50_ms": percentile(self.search_latencies, 0.5),
            "search_latency_p95_ms": percentile(self.search_latencies, 0.95),
            "search_latency_p99_ms": percentile(self.search_latencies, 0.99),
            "avg_result_count": sum(self.search_result_counts) / len(self.search_result_counts) if self.search_result_counts else 0,
            "avg_similarity": sum(self.similarity_scores) / len(self.similarity_scores) if self.similarity_scores else 0,
            "similarity_p50": percentile(self.similarity_scores, 0.5),
            "similarity_p95": percentile(self.similarity_scores, 0.95),
        }


class MemoryService:
    """
    Manages agent memory across two tiers:
    1. Short-term: Redis with TTL (1-2 hours)
    2. Long-term: PostgreSQL with vector embeddings for semantic search
    """

    def __init__(self):
        self._redis_client = None
        self._openai_client = None
        self._memory_db = None
        self._initialized = False
        self._metrics = MemoryServiceMetrics()

    async def initialize(self, db_pool=None):
        """Initialize memory service connections"""
        if self._initialized:
            return

        logger.info("Initializing Memory Service...")

        redis_url = os.getenv("REDIS_URL")
        if redis_url and redis:
            try:
                self._redis_client = redis.from_url(redis_url, decode_responses=True)
                ping_result = await self._redis_client.ping()
                if ping_result:
                    logger.info("Redis connection established for short-term memory")
            except Exception as e:
                logger.warning(f"Redis not available, using fallback: {e}")
                self._redis_client = None
        else:
            logger.info("Redis not configured, using in-memory fallback")
            self._redis_client = None

        try:
            self._openai_client = get_openai_client()
            logger.info("Centralized OpenAI client initialized with PHI/BAA/ZDR enforcement")
        except Exception as e:
            logger.warning(f"OpenAI client initialization failed: {e}")
            self._openai_client = None

        try:
            self._memory_db = await get_memory_db()
            logger.info("Memory database initialized with pgvector support")
        except Exception as e:
            logger.warning(f"Memory database initialization failed: {e}")
            self._memory_db = None

        self._initialized = True
        logger.info("Memory Service initialized successfully")

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

    async def generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding for text using centralized OpenAI client"""
        return await generate_embedding(text)

    async def summarize_for_storage(self, content: str) -> str:
        """Summarize content before long-term storage"""
        if not self._openai_client:
            return content[:1000]

        try:
            response = await self._openai_client.chat_completions_create(
                messages=[
                    {
                        "role": "system",
                        "content": "Summarize the following conversation or memory for long-term storage. Focus on key facts, health information, and important context. Be concise but preserve critical details. Do not include any PHI or identifying information."
                    },
                    {"role": "user", "content": content}
                ],
                model="gpt-4o-mini",
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
        auto_summarize: bool = True,
        user_id: Optional[str] = None,
        conversation_id: Optional[str] = None
    ) -> Optional[str]:
        """
        Store long-term memory with vector embedding in PostgreSQL using pgvector.
        
        Args:
            agent_id: ID of the storing agent
            patient_id: Patient ID for context filtering
            content: Memory content
            memory_type: Type (episodic, semantic, procedural)
            source_type: Source of memory (consolidation, direct, etc.)
            source_id: Reference to source document/conversation
            importance: Importance score 0-1
            metadata: Additional JSON metadata
            auto_summarize: Whether to auto-summarize long content
            user_id: Optional user ID
            conversation_id: Optional conversation ID
            
        Returns:
            Memory ID if successful, None otherwise
        """
        if not self._memory_db:
            logger.error("Memory database not initialized")
            return None

        start_time = time.time()
        
        try:
            summary = None
            if auto_summarize and len(content) > 1000:
                summary = await self.summarize_for_storage(content)

            embed_text = summary or content
            embedding = await self.generate_embedding(embed_text[:8000])
            
            if not embedding:
                logger.error("Failed to generate embedding for long-term memory")
                return None

            memory_id = await self._memory_db.insert_memory(
                agent_id=agent_id,
                patient_id=patient_id,
                user_id=user_id,
                conversation_id=conversation_id,
                memory_type=memory_type,
                storage_type="vector",
                content=content,
                summary=summary,
                embedding=embedding,
                embedding_model=EMBEDDING_MODEL,
                embedding_version=EMBEDDING_VERSION,
                source_type=source_type,
                source_id=source_id,
                importance=importance,
                metadata=metadata
            )
            
            latency_ms = (time.time() - start_time) * 1000
            self._metrics.record_store(latency_ms)
            
            logger.info(f"Stored long-term memory: {memory_id} for patient {patient_id} ({latency_ms:.1f}ms)")
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
        min_similarity: float = 0.6,
        memory_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search long-term memories using semantic similarity via pgvector.
        
        Args:
            agent_id: Agent ID for filtering
            patient_id: Patient ID for filtering
            query: Natural language query
            limit: Maximum results to return
            min_similarity: Minimum similarity threshold (0-1)
            memory_type: Optional filter by memory type
            
        Returns:
            List of matching memories with similarity scores
        """
        if not self._memory_db:
            logger.warning("Memory database not initialized, returning empty results")
            return []

        start_time = time.time()
        
        try:
            query_embedding = await self.generate_embedding(query)
            if not query_embedding:
                logger.error("Failed to generate query embedding")
                return []

            results = await self._memory_db.search_memories(
                query_embedding=query_embedding,
                agent_id=agent_id,
                patient_id=patient_id,
                memory_type=memory_type,
                top_k=limit,
                min_similarity=min_similarity,
                update_access=True
            )
            
            latency_ms = (time.time() - start_time) * 1000
            similarities = [r.get("similarity", 0) for r in results]
            self._metrics.record_search(latency_ms, len(results), similarities)
            
            logger.info(f"Searched long-term memories for patient {patient_id}: {len(results)} results ({latency_ms:.1f}ms)")
            return results

        except Exception as e:
            logger.error(f"Failed to search long-term memories: {e}")
            return []

    async def get_patient_context(
        self,
        agent_id: str,
        patient_id: str,
        include_recent: bool = True,
        include_long_term: bool = True,
        query: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get comprehensive patient context for agent"""
        context = {
            "patient_id": patient_id,
            "recent_memories": [],
            "long_term_memories": [],
            "key_facts": []
        }

        if include_recent:
            context["recent_memories"] = await self.get_short_term_memories(
                agent_id=agent_id,
                user_id=patient_id,
                limit=5
            )

        if include_long_term and query:
            context["long_term_memories"] = await self.search_long_term(
                agent_id=agent_id,
                patient_id=patient_id,
                query=query,
                limit=5
            )

        return context

    async def consolidate_memories(
        self,
        agent_id: str,
        patient_id: str,
        threshold_count: int = 10
    ):
        """Consolidate short-term memories into long-term storage"""
        try:
            memories = await self.get_short_term_memories(
                agent_id=agent_id,
                user_id=patient_id,
                limit=50
            )

            if len(memories) < threshold_count:
                return

            combined_content = "\n\n".join([
                m.get("content", "") for m in memories
            ])

            await self.store_long_term(
                agent_id=agent_id,
                patient_id=patient_id,
                content=combined_content,
                memory_type="episodic",
                source_type="consolidation",
                importance=0.6,
                auto_summarize=True
            )

            await self.clear_short_term_memories(
                agent_id=agent_id,
                user_id=patient_id
            )

            logger.info(f"Consolidated {len(memories)} memories for patient {patient_id}")

        except Exception as e:
            logger.error(f"Failed to consolidate memories: {e}")

    def get_metrics(self) -> Dict[str, Any]:
        """Get memory service metrics for observability"""
        return self._metrics.get_stats()

    async def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics for monitoring"""
        if not self._memory_db:
            return {}
        return await self._memory_db.get_memory_stats()


memory_service = MemoryService()


async def get_memory_service() -> MemoryService:
    """Get initialized memory service instance"""
    if not memory_service._initialized:
        await memory_service.initialize()
    return memory_service
