"""
LlamaIndex Memory Service Wrapper
Provides LlamaIndex-compatible interface to the MemoryService using PostgreSQL + pgvector.

This module bridges our production MemoryService with LlamaIndex for:
1. VectorStore-based retrieval patterns
2. Index-based document storage
3. Semantic search with metadata filtering
"""

import os
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime

from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.schema import TextNode, NodeWithScore, QueryBundle
from llama_index.core.vector_stores.types import (
    VectorStore,
    VectorStoreQuery,
    VectorStoreQueryResult,
    MetadataFilters,
    FilterCondition,
    MetadataFilter,
    FilterOperator,
)
from llama_index.embeddings.openai import OpenAIEmbedding

from app.services.memory_db import get_memory_db
from app.services.openai_client import (
    get_openai_client,
    generate_embedding,
    EMBEDDING_MODEL,
    EMBEDDING_VERSION,
    EMBEDDING_DIMENSION,
)

logger = logging.getLogger(__name__)


class AgentMemoryVectorStore(VectorStore):
    """
    LlamaIndex VectorStore adapter for the agent_memory table.
    
    Wraps our existing pgvector-backed memory storage with LlamaIndex's
    VectorStore interface for seamless integration with retrieval pipelines.
    """
    
    stores_text: bool = True
    flat_metadata: bool = True
    
    def __init__(
        self,
        agent_id: str,
        patient_id: Optional[str] = None,
        memory_type: Optional[str] = None,
    ):
        """
        Initialize the vector store adapter.
        
        Args:
            agent_id: Agent ID for filtering memories
            patient_id: Optional patient ID for filtering
            memory_type: Optional memory type filter
        """
        self._agent_id = agent_id
        self._patient_id = patient_id
        self._memory_type = memory_type
        self._memory_db = None
        self._initialized = False
    
    async def _ensure_initialized(self):
        """Ensure memory database is initialized"""
        if not self._initialized:
            self._memory_db = await get_memory_db()
            self._initialized = True
    
    @property
    def client(self) -> Any:
        """Return the underlying memory database client"""
        return self._memory_db
    
    def add(self, nodes: List[TextNode], **kwargs) -> List[str]:
        """
        Synchronous add - calls async version.
        Use async_add for production code.
        """
        import asyncio
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import nest_asyncio
            nest_asyncio.apply()
        return asyncio.run(self.async_add(nodes, **kwargs))
    
    async def async_add(
        self,
        nodes: List[TextNode],
        **kwargs
    ) -> List[str]:
        """
        Add nodes to the vector store.
        
        Args:
            nodes: List of TextNode objects to add
            **kwargs: Additional arguments (user_id, conversation_id, etc.)
            
        Returns:
            List of node IDs that were added
        """
        await self._ensure_initialized()
        
        user_id = kwargs.get("user_id")
        conversation_id = kwargs.get("conversation_id")
        source_type = kwargs.get("source_type", "llama_index")
        source_id = kwargs.get("source_id")
        importance = kwargs.get("importance", 0.5)
        
        added_ids = []
        
        for node in nodes:
            try:
                embedding = node.embedding
                if not embedding:
                    embedding_result = await generate_embedding(node.text)
                    if not embedding_result:
                        logger.error(f"Failed to generate embedding for node {node.node_id}")
                        continue
                    embedding = embedding_result
                
                memory_id = await self._memory_db.insert_memory(
                    agent_id=self._agent_id,
                    patient_id=self._patient_id,
                    user_id=user_id,
                    conversation_id=conversation_id,
                    memory_type=self._memory_type or "semantic",
                    storage_type="llama_index",
                    content=node.text,
                    summary=node.metadata.get("summary"),
                    embedding=embedding,
                    embedding_model=EMBEDDING_MODEL,
                    embedding_version=EMBEDDING_VERSION,
                    source_type=source_type,
                    source_id=source_id or node.node_id,
                    importance=importance,
                    metadata={
                        **node.metadata,
                        "llama_node_id": node.node_id,
                    }
                )
                
                added_ids.append(memory_id)
                logger.debug(f"Added node {node.node_id} as memory {memory_id}")
                
            except Exception as e:
                logger.error(f"Failed to add node {node.node_id}: {e}")
        
        return added_ids
    
    def delete(self, ref_doc_id: str, **kwargs) -> None:
        """
        Synchronous delete - calls async version.
        """
        import asyncio
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import nest_asyncio
            nest_asyncio.apply()
        asyncio.run(self.async_delete(ref_doc_id, **kwargs))
    
    async def async_delete(self, ref_doc_id: str, **kwargs) -> None:
        """Delete a node by reference document ID"""
        await self._ensure_initialized()
        
        try:
            await self._memory_db.delete_memory(ref_doc_id)
            logger.debug(f"Deleted memory {ref_doc_id}")
        except Exception as e:
            logger.error(f"Failed to delete memory {ref_doc_id}: {e}")
    
    def query(self, query: VectorStoreQuery, **kwargs) -> VectorStoreQueryResult:
        """
        Synchronous query - calls async version.
        """
        import asyncio
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import nest_asyncio
            nest_asyncio.apply()
        return asyncio.run(self.async_query(query, **kwargs))
    
    async def async_query(
        self,
        query: VectorStoreQuery,
        **kwargs
    ) -> VectorStoreQueryResult:
        """
        Query the vector store for similar nodes.
        
        Args:
            query: VectorStoreQuery with embedding and filters
            **kwargs: Additional arguments
            
        Returns:
            VectorStoreQueryResult with matching nodes
        """
        await self._ensure_initialized()
        
        if query.query_embedding is None:
            return VectorStoreQueryResult(nodes=[], similarities=[], ids=[])
        
        memory_type_filter = self._memory_type
        if query.filters:
            for filter_item in query.filters.filters:
                if filter_item.key == "memory_type":
                    memory_type_filter = filter_item.value
        
        min_similarity = kwargs.get("min_similarity", 0.6)
        
        try:
            results = await self._memory_db.search_memories(
                query_embedding=list(query.query_embedding),
                agent_id=self._agent_id,
                patient_id=self._patient_id,
                memory_type=memory_type_filter,
                top_k=query.similarity_top_k or 5,
                min_similarity=min_similarity,
                update_access=True,
            )
            
            nodes = []
            similarities = []
            ids = []
            
            for result in results:
                node = TextNode(
                    text=result["content"],
                    id_=result["id"],
                    metadata={
                        "agent_id": result["agent_id"],
                        "patient_id": result["patient_id"],
                        "memory_type": result["memory_type"],
                        "importance": result["importance"],
                        "access_count": result["access_count"],
                        "created_at": result["created_at"].isoformat() if result["created_at"] else None,
                        "summary": result.get("summary"),
                        **(result.get("metadata") or {}),
                    }
                )
                
                nodes.append(node)
                similarities.append(result["similarity"])
                ids.append(result["id"])
            
            return VectorStoreQueryResult(
                nodes=nodes,
                similarities=similarities,
                ids=ids,
            )
            
        except Exception as e:
            logger.error(f"Vector store query failed: {e}")
            return VectorStoreQueryResult(nodes=[], similarities=[], ids=[])


class LlamaMemoryService:
    """
    LlamaIndex-enhanced Memory Service.
    
    Provides LlamaIndex integration on top of the existing MemoryService,
    enabling retrieval patterns like:
    - VectorStoreIndex for semantic search
    - Query engines for conversational retrieval
    - Node-based document processing
    """
    
    def __init__(self):
        self._embed_model = None
        self._initialized = False
    
    async def initialize(self):
        """Initialize the LlamaIndex service"""
        if self._initialized:
            return
        
        try:
            self._embed_model = OpenAIEmbedding(
                model_name=EMBEDDING_MODEL,
                api_key=os.getenv("OPENAI_API_KEY"),
                embed_batch_size=100,
            )
            
            logger.info("LlamaIndex Memory Service initialized")
            self._initialized = True
            
        except Exception as e:
            logger.error(f"Failed to initialize LlamaIndex service: {e}")
            raise
    
    def get_vector_store(
        self,
        agent_id: str,
        patient_id: Optional[str] = None,
        memory_type: Optional[str] = None,
    ) -> AgentMemoryVectorStore:
        """
        Get a VectorStore instance for a specific agent/patient context.
        
        Args:
            agent_id: Agent ID for the store
            patient_id: Optional patient ID filter
            memory_type: Optional memory type filter
            
        Returns:
            AgentMemoryVectorStore instance
        """
        return AgentMemoryVectorStore(
            agent_id=agent_id,
            patient_id=patient_id,
            memory_type=memory_type,
        )
    
    async def create_index(
        self,
        agent_id: str,
        patient_id: Optional[str] = None,
        memory_type: Optional[str] = None,
    ) -> VectorStoreIndex:
        """
        Create a LlamaIndex VectorStoreIndex for retrieval.
        
        Args:
            agent_id: Agent ID
            patient_id: Optional patient ID
            memory_type: Optional memory type filter
            
        Returns:
            VectorStoreIndex connected to agent_memory
        """
        await self.initialize()
        
        vector_store = self.get_vector_store(
            agent_id=agent_id,
            patient_id=patient_id,
            memory_type=memory_type,
        )
        
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store,
        )
        
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            embed_model=self._embed_model,
            storage_context=storage_context,
        )
        
        return index
    
    async def index_documents(
        self,
        agent_id: str,
        documents: List[str],
        patient_id: Optional[str] = None,
        memory_type: str = "semantic",
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[str]:
        """
        Index documents into the memory store.
        
        Args:
            agent_id: Agent ID
            documents: List of document texts
            patient_id: Optional patient ID
            memory_type: Memory type for documents
            metadata: Optional metadata for all documents
            **kwargs: Additional arguments for storage
            
        Returns:
            List of stored memory IDs
        """
        await self.initialize()
        
        vector_store = self.get_vector_store(
            agent_id=agent_id,
            patient_id=patient_id,
            memory_type=memory_type,
        )
        
        nodes = []
        for i, doc in enumerate(documents):
            node = TextNode(
                text=doc,
                metadata={
                    **(metadata or {}),
                    "doc_index": i,
                }
            )
            nodes.append(node)
        
        return await vector_store.async_add(nodes, **kwargs)
    
    async def retrieve(
        self,
        agent_id: str,
        query: str,
        patient_id: Optional[str] = None,
        memory_type: Optional[str] = None,
        top_k: int = 5,
        min_similarity: float = 0.6,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant memories using semantic search.
        
        Args:
            agent_id: Agent ID
            query: Search query
            patient_id: Optional patient filter
            memory_type: Optional memory type filter
            top_k: Number of results
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of relevant memories with scores
        """
        await self.initialize()
        
        query_embedding = await generate_embedding(query)
        if not query_embedding:
            logger.error("Failed to generate query embedding")
            return []
        
        vector_store = self.get_vector_store(
            agent_id=agent_id,
            patient_id=patient_id,
            memory_type=memory_type,
        )
        
        vstore_query = VectorStoreQuery(
            query_embedding=query_embedding,
            similarity_top_k=top_k,
        )
        
        result = await vector_store.async_query(
            vstore_query,
            min_similarity=min_similarity,
        )
        
        memories = []
        for i, node in enumerate(result.nodes):
            memories.append({
                "id": result.ids[i] if result.ids else node.id_,
                "content": node.text,
                "similarity": result.similarities[i] if result.similarities else 0,
                "metadata": node.metadata,
            })
        
        return memories


llama_memory_service = LlamaMemoryService()


async def get_llama_memory_service() -> LlamaMemoryService:
    """Get initialized LlamaIndex memory service"""
    await llama_memory_service.initialize()
    return llama_memory_service
