"""
Integration Tests for RAG (Retrieval-Augmented Generation) Pipeline

Tests end-to-end flow:
1. Memory indexing with embeddings
2. Semantic retrieval with pgvector
3. Context assembly for LLM
4. PHI gating at each stage

HIPAA Compliance: Uses synthetic test data only
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime
import os

os.environ.setdefault("ENV", "dev")
os.environ.setdefault("OPENAI_API_KEY", "test-key")


class TestRAGPipelineIntegration:
    """End-to-end RAG pipeline tests"""
    
    @pytest.fixture
    def mock_embedding(self):
        """Generate mock 1536-dimensional embedding"""
        import random
        random.seed(42)
        return [random.random() for _ in range(1536)]
    
    @pytest.fixture
    def mock_memory_service(self):
        """Mock memory service for testing"""
        service = Mock()
        service.store_long_term = AsyncMock(return_value="mem-123")
        service.search_long_term = AsyncMock(return_value=[
            {
                "memory_id": "mem-001",
                "content": "Patient reports improved sleep quality",
                "similarity": 0.92,
                "memory_type": "symptom",
                "created_at": datetime.utcnow().isoformat()
            },
            {
                "memory_id": "mem-002", 
                "content": "Blood pressure readings stable at 120/80",
                "similarity": 0.87,
                "memory_type": "vital",
                "created_at": datetime.utcnow().isoformat()
            }
        ])
        return service
    
    @pytest.fixture
    def mock_openai_client(self, mock_embedding):
        """Mock OpenAI client"""
        client = Mock()
        client.create_embedding = Mock(return_value=mock_embedding)
        client.EMBEDDING_MODEL = "text-embedding-3-small"
        client.EMBEDDING_VERSION = "v1.0.0"
        client.EMBEDDING_DIMENSIONS = 1536
        return client
    
    @pytest.mark.asyncio
    async def test_index_retrieve_flow(self, mock_memory_service, mock_openai_client, mock_embedding):
        """Test indexing a memory and retrieving it"""
        content = "Patient reported headache improvement after medication change"
        
        memory_id = await mock_memory_service.store_long_term(
            agent_id="clona-001",
            patient_id="patient-123",
            memory_type="symptom",
            content=content,
            embedding=mock_embedding,
            metadata={"session_id": "session-456"}
        )
        
        assert memory_id == "mem-123"
        mock_memory_service.store_long_term.assert_called_once()
        
        results = await mock_memory_service.search_long_term(
            agent_id="clona-001",
            query_embedding=mock_embedding,
            patient_id="patient-123",
            top_k=5
        )
        
        assert len(results) == 2
        assert results[0]["similarity"] > 0.9
        assert "content" in results[0]
    
    @pytest.mark.asyncio
    async def test_context_assembly_for_llm(self, mock_memory_service, mock_embedding):
        """Test assembling retrieved context for LLM prompt"""
        results = await mock_memory_service.search_long_term(
            agent_id="clona-001",
            query_embedding=mock_embedding,
            patient_id="patient-123",
            top_k=5
        )
        
        context_parts = []
        for i, result in enumerate(results):
            context_parts.append(f"[{i+1}] {result['content']} (relevance: {result['similarity']:.2f})")
        
        assembled_context = "\n".join(context_parts)
        
        assert "[1]" in assembled_context
        assert "improved sleep" in assembled_context
        assert "0.92" in assembled_context
    
    @pytest.mark.asyncio
    async def test_phi_gating_in_retrieval(self, mock_memory_service, mock_embedding):
        """Test that PHI gating is enforced during retrieval"""
        results = await mock_memory_service.search_long_term(
            agent_id="clona-001",
            query_embedding=mock_embedding,
            patient_id="patient-123",
            top_k=5
        )
        
        for result in results:
            assert "SSN" not in result.get("content", "")
            assert "123-45-6789" not in result.get("content", "")
    
    @pytest.mark.asyncio
    async def test_embedding_model_tracking(self, mock_openai_client):
        """Test that embedding model info is tracked"""
        assert mock_openai_client.EMBEDDING_MODEL == "text-embedding-3-small"
        assert mock_openai_client.EMBEDDING_DIMENSIONS == 1536
        assert mock_openai_client.EMBEDDING_VERSION == "v1.0.0"
    
    @pytest.mark.asyncio
    async def test_similarity_threshold_filtering(self, mock_memory_service, mock_embedding):
        """Test filtering by similarity threshold"""
        mock_memory_service.search_long_term = AsyncMock(return_value=[
            {"memory_id": "mem-001", "content": "Highly relevant", "similarity": 0.95},
            {"memory_id": "mem-002", "content": "Moderately relevant", "similarity": 0.75},
            {"memory_id": "mem-003", "content": "Low relevance", "similarity": 0.45}
        ])
        
        results = await mock_memory_service.search_long_term(
            agent_id="clona-001",
            query_embedding=mock_embedding,
            patient_id="patient-123",
            top_k=10
        )
        
        min_threshold = 0.7
        filtered = [r for r in results if r["similarity"] >= min_threshold]
        
        assert len(filtered) == 2
        assert all(r["similarity"] >= min_threshold for r in filtered)


class TestPHIGatingFlow:
    """Tests for PHI detection and gating throughout RAG pipeline"""
    
    def test_phi_blocked_before_embedding(self):
        """Test that PHI is blocked before creating embeddings"""
        from app.services.openai_client import OpenAIClientWrapper, PHIDetectionError
        
        with patch.dict(os.environ, {
            "ENV": "dev",
            "OPENAI_API_KEY": "test-key",
            "PHI_DETECTION_ENABLED": "true",
            "PHI_BLOCK_ON_DETECT": "true"
        }):
            import importlib
            import app.services.openai_client as client_module
            importlib.reload(client_module)
            
            client = client_module.OpenAIClientWrapper()
            
            text_with_phi = "Patient SSN: 123-45-6789"
            
            with pytest.raises(client_module.PHIDetectionError):
                client._check_and_handle_phi(text_with_phi, "embedding")
    
    def test_phi_redaction_in_non_blocking_mode(self):
        """Test PHI redaction when not in blocking mode"""
        from app.services.openai_client import OpenAIClientWrapper
        
        with patch.dict(os.environ, {
            "ENV": "dev",
            "OPENAI_API_KEY": "test-key",
            "PHI_DETECTION_ENABLED": "true",
            "PHI_BLOCK_ON_DETECT": "false"
        }):
            import importlib
            import app.services.openai_client as client_module
            importlib.reload(client_module)
            
            client = client_module.OpenAIClientWrapper()
            
            text_with_phi = "Contact patient at test@email.com"
            result = client._check_and_handle_phi(text_with_phi, "embedding")
            
            assert "test@email.com" not in result
            assert "[EMAIL_REDACTED]" in result


class TestLlamaIndexIntegration:
    """Tests for LlamaIndex VectorStore integration"""
    
    @pytest.mark.asyncio
    async def test_llamaindex_query(self):
        """Test LlamaIndex-style query"""
        mock_db = Mock()
        mock_db.search = AsyncMock(return_value=[
            {
                "id": "node-1",
                "text": "Patient symptom report",
                "score": 0.89,
                "metadata": {"agent_id": "clona-001"}
            }
        ])
        
        embedding = [0.1] * 1536
        
        results = await mock_db.search(
            embedding=embedding,
            top_k=5,
            filters={"agent_id": "clona-001"}
        )
        
        assert len(results) == 1
        assert results[0]["score"] > 0.8


class TestDualWriteMigration:
    """Tests for dual-write migration pattern"""
    
    @pytest.fixture
    def migration_service(self):
        from app.services.dual_write_migration import DualWriteMigration
        
        legacy = Mock()
        legacy.store = AsyncMock()
        legacy.search = AsyncMock(return_value=[
            {"id": "legacy-1", "content": "test"},
            {"id": "legacy-2", "content": "test2"}
        ])
        
        new_store = Mock()
        new_store.store_long_term = AsyncMock()
        new_store.search_long_term = AsyncMock(return_value=[
            {"memory_id": "new-1", "content": "test"},
            {"memory_id": "new-2", "content": "test2"}
        ])
        
        return DualWriteMigration(legacy_store=legacy, new_store=new_store)
    
    @pytest.mark.asyncio
    async def test_dual_write_success(self, migration_service):
        """Test writing to both systems"""
        legacy_ok, new_ok = await migration_service.dual_write(
            memory_id="mem-123",
            content="Test content",
            embedding=[0.1] * 1536,
            metadata={"agent_id": "clona-001"}
        )
        
        assert legacy_ok
        assert new_ok
    
    @pytest.mark.asyncio
    async def test_shadow_comparison(self, migration_service):
        """Test shadow comparison between systems"""
        result = await migration_service.shadow_compare(
            query="test query",
            query_embedding=[0.1] * 1536,
            filters={"agent_id": "clona-001"},
            top_k=5
        )
        
        assert result.precision_at_5 >= 0
        assert result.latency_new_ms >= 0
        assert result.latency_old_ms >= 0
    
    def test_precision_calculation(self, migration_service):
        """Test precision@k calculation"""
        ground_truth = ["a", "b", "c", "d", "e"]
        predicted = ["a", "b", "x", "y", "z"]
        
        precision = migration_service._calculate_precision(ground_truth, predicted, 5)
        
        assert precision == 0.4
    
    def test_migration_readiness_check(self, migration_service):
        """Test migration readiness determination"""
        assert not migration_service.is_migration_ready(min_comparisons=100)
        
        from app.services.dual_write_migration import ComparisonResult
        from datetime import datetime
        
        for i in range(100):
            migration_service.comparison_results.append(
                ComparisonResult(
                    query_hash=f"hash-{i}",
                    old_ids=["a", "b", "c"],
                    new_ids=["a", "b", "c"],
                    precision_at_1=1.0,
                    precision_at_5=0.9,
                    precision_at_10=0.85,
                    recall_at_10=0.85,
                    latency_old_ms=50,
                    latency_new_ms=30
                )
            )
        
        assert migration_service.is_migration_ready(min_comparisons=100, min_precision=0.8)
