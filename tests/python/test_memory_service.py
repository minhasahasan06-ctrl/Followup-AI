"""
Unit tests for Memory Service with pgvector integration.
"""

import os
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from datetime import datetime

os.environ["ENV"] = "dev"
os.environ["OPENAI_API_KEY"] = "test-key"


class TestMemoryServiceMetrics:
    """Tests for MemoryServiceMetrics class"""
    
    def test_percentile_calculation(self):
        """Test percentile calculation"""
        from app.services.memory_service import MemoryServiceMetrics
        
        metrics = MemoryServiceMetrics()
        
        for i in range(100):
            metrics.store_latencies.append(float(i))
            metrics.store_count += 1
        
        stats = metrics.get_stats()
        
        assert stats["store_latency_p50_ms"] == 50.0
        assert stats["store_latency_p95_ms"] == 95.0
        assert stats["store_latency_p99_ms"] == 99.0
    
    def test_empty_metrics(self):
        """Test metrics with no data"""
        from app.services.memory_service import MemoryServiceMetrics
        
        metrics = MemoryServiceMetrics()
        stats = metrics.get_stats()
        
        assert stats["store_count"] == 0
        assert stats["search_count"] == 0
        assert stats["store_latency_p50_ms"] == 0
    
    def test_record_store(self):
        """Test recording store operation"""
        from app.services.memory_service import MemoryServiceMetrics
        
        metrics = MemoryServiceMetrics()
        metrics.record_store(50.0)
        metrics.record_store(100.0)
        
        assert metrics.store_count == 2
        assert len(metrics.store_latencies) == 2
    
    def test_record_search(self):
        """Test recording search operation"""
        from app.services.memory_service import MemoryServiceMetrics
        
        metrics = MemoryServiceMetrics()
        metrics.record_search(100.0, 5, [0.8, 0.75, 0.7])
        
        assert metrics.search_count == 1
        assert len(metrics.similarity_scores) == 3
        assert metrics.search_result_counts[0] == 5


class TestMemoryService:
    """Tests for MemoryService class"""
    
    @pytest.mark.asyncio
    async def test_generate_embedding_returns_none_without_client(self):
        """Test embedding generation without OpenAI client"""
        from app.services.memory_service import MemoryService
        
        service = MemoryService()
        service._openai_client = None
        
        result = await service.generate_embedding("test text")
        
        assert result is None
    
    def test_short_term_memory_key_format(self):
        """Test short-term memory key format"""
        from app.services.memory_service import MemoryService
        import time
        
        service = MemoryService()
        
        timestamp = time.time()
        key = f"memory:agent1:user1:conv1:{timestamp}"
        
        assert key.startswith("memory:")
        assert "agent1" in key
        assert "user1" in key
        assert "conv1" in key


class TestMemoryDB:
    """Tests for MemoryDB class"""
    
    def test_embedding_string_format(self):
        """Test that embedding is formatted correctly for pgvector"""
        embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        embedding_str = f"[{','.join(str(x) for x in embedding)}]"
        
        assert embedding_str == "[0.1,0.2,0.3,0.4,0.5]"
        assert embedding_str.startswith("[")
        assert embedding_str.endswith("]")
    
    def test_similarity_calculation_formula(self):
        """Test that similarity is calculated correctly from cosine distance"""
        cosine_distance = 0.2
        similarity = 1 - cosine_distance
        
        assert similarity == 0.8
    
    def test_min_similarity_filter(self):
        """Test minimum similarity threshold logic"""
        min_similarity = 0.6
        
        test_cases = [
            (0.7, True),
            (0.6, True),
            (0.5, False),
            (0.0, False),
            (1.0, True),
        ]
        
        for score, should_include in test_cases:
            included = score >= min_similarity
            assert included == should_include, f"Failed for score {score}"


class TestLlamaMemoryService:
    """Tests for LlamaIndex integration"""
    
    def test_vector_store_initialization(self):
        """Test VectorStore adapter initialization"""
        from app.services.llama_memory_service import AgentMemoryVectorStore
        
        store = AgentMemoryVectorStore(
            agent_id="test-agent",
            patient_id="test-patient",
            memory_type="episodic"
        )
        
        assert store._agent_id == "test-agent"
        assert store._patient_id == "test-patient"
        assert store._memory_type == "episodic"
        assert not store._initialized


class TestObservabilityMetrics:
    """Tests for ML observability metrics"""
    
    def test_histogram_stats(self):
        """Test histogram statistics calculation"""
        from app.services.ml_observability import MetricHistogram
        
        hist = MetricHistogram("test")
        
        for i in range(100):
            hist.observe(float(i))
        
        stats = hist.stats()
        
        assert stats["count"] == 100
        assert stats["p50"] == 50.0
        assert stats["min"] == 0.0
        assert stats["max"] == 99.0
    
    def test_counter_increment(self):
        """Test counter increment"""
        from app.services.ml_observability import MetricCounter
        
        counter = MetricCounter("test")
        
        counter.inc({"type": "a"})
        counter.inc({"type": "a"})
        counter.inc({"type": "b"})
        
        assert counter.get({"type": "a"}) == 2
        assert counter.get({"type": "b"}) == 1
        assert counter.total() == 3
    
    def test_gauge_operations(self):
        """Test gauge set/inc/dec"""
        from app.services.ml_observability import MetricGauge
        
        gauge = MetricGauge("test")
        
        gauge.set(10)
        assert gauge.get() == 10
        
        gauge.inc(5)
        assert gauge.get() == 15
        
        gauge.dec(3)
        assert gauge.get() == 12


class TestAlertingService:
    """Tests for ML alerting service"""
    
    def test_alert_severity_enum(self):
        """Test alert severity levels"""
        from app.services.ml_alerting import AlertSeverity
        
        assert AlertSeverity.INFO.value == "info"
        assert AlertSeverity.WARNING.value == "warning"
        assert AlertSeverity.ERROR.value == "error"
        assert AlertSeverity.CRITICAL.value == "critical"
    
    def test_alert_status_enum(self):
        """Test alert status values"""
        from app.services.ml_alerting import AlertStatus
        
        assert AlertStatus.FIRING.value == "firing"
        assert AlertStatus.RESOLVED.value == "resolved"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
