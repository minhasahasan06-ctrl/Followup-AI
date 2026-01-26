"""
Memory Database Module
Provides low-level database operations for agent memory with pgvector support.

This module handles:
1. Insert/update operations for agent memories with vector embeddings
2. Similarity search using pgvector cosine distance
3. Access tracking and metrics updates
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid

import asyncpg

logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL")


class MemoryDB:
    """
    Database operations for agent memory with pgvector support.
    Uses asyncpg for async Postgres operations.
    """
    
    def __init__(self):
        self._pool: Optional[asyncpg.Pool] = None
    
    async def initialize(self):
        """Initialize database connection pool"""
        if self._pool is not None:
            return
        
        try:
            self._pool = await asyncpg.create_pool(
                DATABASE_URL,
                min_size=2,
                max_size=10,
                command_timeout=60
            )
            logger.info("Memory database pool initialized")
        except Exception as e:
            logger.error(f"Failed to initialize memory database pool: {e}")
            raise
    
    async def close(self):
        """Close database connection pool"""
        if self._pool:
            await self._pool.close()
            self._pool = None
    
    async def insert_memory(
        self,
        agent_id: str,
        patient_id: Optional[str],
        user_id: Optional[str],
        conversation_id: Optional[str],
        memory_type: str,
        storage_type: str,
        content: str,
        summary: Optional[str],
        embedding: List[float],
        embedding_model: str,
        embedding_version: str,
        source_type: Optional[str] = None,
        source_id: Optional[str] = None,
        importance: float = 0.5,
        expires_at: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Insert a new memory with vector embedding.
        
        Args:
            agent_id: ID of the agent storing the memory
            patient_id: Optional patient ID for patient-specific memories
            user_id: Optional user ID
            conversation_id: Optional conversation ID
            memory_type: Type of memory (episodic, semantic, etc.)
            storage_type: Storage type (vector, redis, postgres)
            content: Memory content text
            summary: Optional summary of content
            embedding: Vector embedding as list of floats
            embedding_model: Model used to generate embedding
            embedding_version: Version of embedding model
            source_type: Type of source (consolidation, direct, etc.)
            source_id: ID of source document/conversation
            importance: Importance score 0-1
            expires_at: Optional expiration timestamp
            metadata: Optional JSON metadata
            
        Returns:
            ID of inserted memory
        """
        await self.initialize()
        
        memory_id = str(uuid.uuid4())
        embedding_str = f"[{','.join(str(x) for x in embedding)}]"
        
        query = """
            INSERT INTO agent_memory (
                id, agent_id, patient_id, user_id, conversation_id,
                memory_type, storage_type, content, summary,
                embedding, embedding_model, embedding_version,
                source_type, source_id, importance, expires_at,
                metadata, created_at, updated_at
            ) VALUES (
                $1, $2, $3, $4, $5,
                $6, $7, $8, $9,
                $10::vector, $11, $12,
                $13, $14, $15, $16,
                $17, NOW(), NOW()
            )
            RETURNING id
        """
        
        try:
            async with self._pool.acquire() as conn:
                result = await conn.fetchval(
                    query,
                    memory_id, agent_id, patient_id, user_id, conversation_id,
                    memory_type, storage_type, content, summary,
                    embedding_str, embedding_model, embedding_version,
                    source_type, source_id, importance, expires_at,
                    json.dumps(metadata) if metadata else None
                )
                logger.debug(f"Inserted memory: {result}")
                return result
        except Exception as e:
            logger.error(f"Failed to insert memory: {e}")
            raise
    
    async def search_memories(
        self,
        query_embedding: List[float],
        agent_id: Optional[str] = None,
        patient_id: Optional[str] = None,
        memory_type: Optional[str] = None,
        top_k: int = 5,
        min_similarity: float = 0.6,
        update_access: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Search memories using vector similarity.
        
        Args:
            query_embedding: Query vector for similarity search
            agent_id: Optional filter by agent
            patient_id: Optional filter by patient
            memory_type: Optional filter by memory type
            top_k: Number of results to return
            min_similarity: Minimum similarity threshold (0-1)
            update_access: Whether to update access_count and last_accessed_at
            
        Returns:
            List of matching memories with similarity scores
        """
        await self.initialize()
        
        embedding_str = f"[{','.join(str(x) for x in query_embedding)}]"
        
        where_clauses = ["(1 - (embedding <=> $1::vector)) >= $2"]
        params = [embedding_str, min_similarity]
        param_idx = 3
        
        if agent_id:
            where_clauses.append(f"agent_id = ${param_idx}")
            params.append(agent_id)
            param_idx += 1
        
        if patient_id:
            where_clauses.append(f"patient_id = ${param_idx}")
            params.append(patient_id)
            param_idx += 1
        
        if memory_type:
            where_clauses.append(f"memory_type = ${param_idx}")
            params.append(memory_type)
            param_idx += 1
        
        where_clauses.append("(expires_at IS NULL OR expires_at > NOW())")
        
        where_clause = " AND ".join(where_clauses)
        
        query = f"""
            SELECT 
                id, agent_id, patient_id, user_id, conversation_id,
                memory_type, storage_type, content, summary,
                embedding_model, embedding_version,
                source_type, source_id, importance,
                access_count, last_accessed_at, expires_at,
                metadata, created_at, updated_at,
                (1 - (embedding <=> $1::vector)) AS similarity
            FROM agent_memory
            WHERE {where_clause}
            ORDER BY embedding <=> $1::vector
            LIMIT ${param_idx}
        """
        params.append(top_k)
        
        try:
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(query, *params)
                
                results = []
                memory_ids = []
                
                for row in rows:
                    memory_ids.append(row['id'])
                    results.append({
                        "id": row["id"],
                        "agent_id": row["agent_id"],
                        "patient_id": row["patient_id"],
                        "user_id": row["user_id"],
                        "conversation_id": row["conversation_id"],
                        "memory_type": row["memory_type"],
                        "storage_type": row["storage_type"],
                        "content": row["content"],
                        "summary": row["summary"],
                        "embedding_model": row["embedding_model"],
                        "embedding_version": row["embedding_version"],
                        "source_type": row["source_type"],
                        "source_id": row["source_id"],
                        "importance": float(row["importance"]) if row["importance"] else 0.5,
                        "access_count": row["access_count"] or 0,
                        "last_accessed_at": row["last_accessed_at"],
                        "expires_at": row["expires_at"],
                        "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
                        "created_at": row["created_at"],
                        "updated_at": row["updated_at"],
                        "similarity": float(row["similarity"]),
                    })
                
                if update_access and memory_ids:
                    await self._update_access_metrics(conn, memory_ids)
                
                return results
                
        except Exception as e:
            logger.error(f"Failed to search memories: {e}")
            raise
    
    async def _update_access_metrics(self, conn, memory_ids: List[str]):
        """Update access_count and last_accessed_at for retrieved memories"""
        try:
            await conn.execute("""
                UPDATE agent_memory
                SET access_count = COALESCE(access_count, 0) + 1,
                    last_accessed_at = NOW()
                WHERE id = ANY($1)
            """, memory_ids)
        except Exception as e:
            logger.warning(f"Failed to update access metrics: {e}")
    
    async def get_memory_by_id(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Get a single memory by ID"""
        await self.initialize()
        
        query = """
            SELECT 
                id, agent_id, patient_id, user_id, conversation_id,
                memory_type, storage_type, content, summary,
                embedding_model, embedding_version,
                source_type, source_id, importance,
                access_count, last_accessed_at, expires_at,
                metadata, created_at, updated_at
            FROM agent_memory
            WHERE id = $1
        """
        
        try:
            async with self._pool.acquire() as conn:
                row = await conn.fetchrow(query, memory_id)
                
                if not row:
                    return None
                
                return {
                    "id": row["id"],
                    "agent_id": row["agent_id"],
                    "patient_id": row["patient_id"],
                    "user_id": row["user_id"],
                    "conversation_id": row["conversation_id"],
                    "memory_type": row["memory_type"],
                    "storage_type": row["storage_type"],
                    "content": row["content"],
                    "summary": row["summary"],
                    "embedding_model": row["embedding_model"],
                    "embedding_version": row["embedding_version"],
                    "source_type": row["source_type"],
                    "source_id": row["source_id"],
                    "importance": float(row["importance"]) if row["importance"] else 0.5,
                    "access_count": row["access_count"] or 0,
                    "last_accessed_at": row["last_accessed_at"],
                    "expires_at": row["expires_at"],
                    "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                }
        except Exception as e:
            logger.error(f"Failed to get memory: {e}")
            raise
    
    async def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory by ID"""
        await self.initialize()
        
        try:
            async with self._pool.acquire() as conn:
                result = await conn.execute(
                    "DELETE FROM agent_memory WHERE id = $1",
                    memory_id
                )
                return "DELETE 1" in result
        except Exception as e:
            logger.error(f"Failed to delete memory: {e}")
            raise
    
    async def get_memory_stats(
        self,
        agent_id: Optional[str] = None,
        patient_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get memory statistics for observability"""
        await self.initialize()
        
        where_clauses = []
        params = []
        param_idx = 1
        
        if agent_id:
            where_clauses.append(f"agent_id = ${param_idx}")
            params.append(agent_id)
            param_idx += 1
        
        if patient_id:
            where_clauses.append(f"patient_id = ${param_idx}")
            params.append(patient_id)
            param_idx += 1
        
        where_clause = " AND ".join(where_clauses) if where_clauses else "1=1"
        
        query = f"""
            SELECT 
                COUNT(*) as total_memories,
                COUNT(CASE WHEN embedding IS NOT NULL THEN 1 END) as memories_with_embedding,
                COUNT(CASE WHEN embedding_model IS NULL THEN 1 END) as memories_without_model,
                AVG(importance) as avg_importance,
                AVG(access_count) as avg_access_count,
                COUNT(DISTINCT agent_id) as unique_agents,
                COUNT(DISTINCT patient_id) as unique_patients
            FROM agent_memory
            WHERE {where_clause}
        """
        
        try:
            async with self._pool.acquire() as conn:
                row = await conn.fetchrow(query, *params)
                
                return {
                    "total_memories": row["total_memories"],
                    "memories_with_embedding": row["memories_with_embedding"],
                    "memories_without_model": row["memories_without_model"],
                    "avg_importance": float(row["avg_importance"]) if row["avg_importance"] else 0,
                    "avg_access_count": float(row["avg_access_count"]) if row["avg_access_count"] else 0,
                    "unique_agents": row["unique_agents"],
                    "unique_patients": row["unique_patients"],
                }
        except Exception as e:
            logger.error(f"Failed to get memory stats: {e}")
            raise


memory_db = MemoryDB()


async def get_memory_db() -> MemoryDB:
    """Get initialized memory database instance"""
    await memory_db.initialize()
    return memory_db
