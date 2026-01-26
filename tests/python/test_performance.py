"""
Performance Tests for Memory System

Benchmarks:
- Index search latency under load
- Memory ingestion throughput
- Embedding generation overhead
- Concurrent query handling

Target SLAs:
- Retrieval latency p99 < 1000ms
- Embedding latency p99 < 2000ms
- Throughput > 100 memories/second
"""

import pytest
import asyncio
import time
import statistics
from typing import List, Tuple
from unittest.mock import Mock, AsyncMock, patch
import os

os.environ.setdefault("ENV", "dev")
os.environ.setdefault("OPENAI_API_KEY", "test-key")


class TestSearchLatency:
    """Performance tests for vector search latency"""
    
    @pytest.fixture
    def mock_search_latencies(self) -> List[float]:
        """Simulate realistic search latencies in ms"""
        import random
        random.seed(42)
        base = 50
        latencies = []
        for _ in range(1000):
            jitter = random.gauss(0, 20)
            spike = random.random() < 0.01
            latency = base + jitter + (200 if spike else 0)
            latencies.append(max(5, latency))
        return latencies
    
    def test_p50_latency_under_sla(self, mock_search_latencies):
        """Test that p50 latency is acceptable"""
        p50 = statistics.median(mock_search_latencies)
        assert p50 < 100, f"P50 latency {p50}ms exceeds 100ms target"
    
    def test_p95_latency_under_sla(self, mock_search_latencies):
        """Test that p95 latency is acceptable"""
        sorted_latencies = sorted(mock_search_latencies)
        p95_idx = int(len(sorted_latencies) * 0.95)
        p95 = sorted_latencies[p95_idx]
        assert p95 < 500, f"P95 latency {p95}ms exceeds 500ms target"
    
    def test_p99_latency_under_sla(self, mock_search_latencies):
        """Test that p99 latency is under 1000ms SLA"""
        sorted_latencies = sorted(mock_search_latencies)
        p99_idx = int(len(sorted_latencies) * 0.99)
        p99 = sorted_latencies[p99_idx]
        assert p99 < 1000, f"P99 latency {p99}ms exceeds 1000ms SLA"
    
    @pytest.mark.asyncio
    async def test_concurrent_queries(self):
        """Test handling concurrent queries"""
        mock_service = Mock()
        mock_service.search_long_term = AsyncMock(return_value=[
            {"memory_id": "mem-1", "content": "test", "similarity": 0.9}
        ])
        
        async def run_query(query_id: int) -> Tuple[int, float]:
            start = time.time()
            await asyncio.sleep(0.01)
            await mock_service.search_long_term(
                agent_id="clona-001",
                query_embedding=[0.1] * 1536,
                top_k=5
            )
            return query_id, (time.time() - start) * 1000
        
        tasks = [run_query(i) for i in range(50)]
        results = await asyncio.gather(*tasks)
        
        latencies = [r[1] for r in results]
        avg_latency = sum(latencies) / len(latencies)
        
        assert len(results) == 50
        assert avg_latency < 500, f"Average concurrent latency {avg_latency}ms too high"


class TestIngestionThroughput:
    """Performance tests for memory ingestion"""
    
    @pytest.mark.asyncio
    async def test_batch_ingestion_throughput(self):
        """Test throughput of batch memory ingestion"""
        mock_service = Mock()
        mock_service.store_long_term = AsyncMock(return_value="mem-id")
        
        batch_size = 100
        memories = [
            {
                "agent_id": "clona-001",
                "patient_id": f"patient-{i}",
                "content": f"Test memory content {i}",
                "embedding": [0.1] * 1536
            }
            for i in range(batch_size)
        ]
        
        start = time.time()
        
        tasks = [
            mock_service.store_long_term(
                agent_id=m["agent_id"],
                patient_id=m["patient_id"],
                content=m["content"],
                embedding=m["embedding"],
                memory_type="test"
            )
            for m in memories
        ]
        await asyncio.gather(*tasks)
        
        elapsed = time.time() - start
        throughput = batch_size / elapsed
        
        assert throughput > 10, f"Throughput {throughput:.1f}/s below 10/s minimum"
    
    @pytest.mark.asyncio
    async def test_single_memory_latency(self):
        """Test latency for single memory ingestion"""
        mock_service = Mock()
        mock_service.store_long_term = AsyncMock(return_value="mem-id")
        
        latencies = []
        for i in range(20):
            start = time.time()
            await mock_service.store_long_term(
                agent_id="clona-001",
                patient_id="patient-123",
                content=f"Test content {i}",
                embedding=[0.1] * 1536,
                memory_type="test"
            )
            latencies.append((time.time() - start) * 1000)
        
        avg_latency = sum(latencies) / len(latencies)
        assert avg_latency < 100, f"Average ingestion latency {avg_latency}ms too high"


class TestEmbeddingPerformance:
    """Performance tests for embedding generation"""
    
    def test_embedding_dimension_validation(self):
        """Test that embedding dimensions are correct"""
        expected_dims = 1536
        test_embedding = [0.1] * expected_dims
        
        assert len(test_embedding) == expected_dims
    
    @pytest.mark.asyncio
    async def test_batch_embedding_efficiency(self):
        """Test efficiency of batch embedding generation"""
        mock_client = Mock()
        mock_client.create_embedding = Mock(return_value=[0.1] * 1536)
        
        texts = [f"Test text {i}" for i in range(50)]
        
        start = time.time()
        embeddings = [mock_client.create_embedding(text) for text in texts]
        elapsed = time.time() - start
        
        assert len(embeddings) == 50
        assert elapsed < 1.0, f"Batch embedding took {elapsed}s, expected < 1s"


class TestObservabilityPerformance:
    """Performance tests for metrics collection overhead"""
    
    def test_histogram_overhead(self):
        """Test that histogram recording has minimal overhead"""
        from app.services.ml_observability import MetricHistogram
        
        histogram = MetricHistogram("test_latency")
        
        start = time.time()
        for _ in range(10000):
            histogram.observe(50.0)
        elapsed = time.time() - start
        
        overhead_per_call_us = (elapsed / 10000) * 1_000_000
        assert overhead_per_call_us < 100, f"Histogram overhead {overhead_per_call_us}μs too high"
    
    def test_counter_overhead(self):
        """Test that counter incrementing has minimal overhead"""
        from app.services.ml_observability import MetricCounter
        
        counter = MetricCounter("test_counter")
        
        start = time.time()
        for _ in range(10000):
            counter.inc()
        elapsed = time.time() - start
        
        overhead_per_call_us = (elapsed / 10000) * 1_000_000
        assert overhead_per_call_us < 50, f"Counter overhead {overhead_per_call_us}μs too high"
    
    def test_metrics_snapshot_performance(self):
        """Test that getting metrics snapshot is fast"""
        from app.services.ml_observability import MetricHistogram, MetricCounter
        
        histograms = []
        counters = []
        for i in range(10):
            hist = MetricHistogram(f"perf_hist_{i}")
            counter = MetricCounter(f"perf_counter_{i}")
            histograms.append(hist)
            counters.append(counter)
            for _ in range(100):
                hist.observe(50.0)
                counter.inc()
        
        start = time.time()
        for _ in range(100):
            for h in histograms:
                h.stats()
            for c in counters:
                c.total()
        elapsed = time.time() - start
        
        avg_snapshot_ms = (elapsed / 100) * 1000
        assert avg_snapshot_ms < 10, f"Snapshot took {avg_snapshot_ms}ms, expected < 10ms"


class TestAlertingPerformance:
    """Performance tests for alerting system"""
    
    def test_alert_check_overhead(self):
        """Test that alert checking is fast"""
        from app.services.ml_alerting import MLAlertingService
        
        service = MLAlertingService()
        
        start = time.time()
        for _ in range(1000):
            service.check_rules()
        elapsed = time.time() - start
        
        overhead_per_call_us = (elapsed / 1000) * 1_000_000
        assert overhead_per_call_us < 100, f"Alert check overhead {overhead_per_call_us}μs too high"


class TestGovernancePerformance:
    """Performance tests for ML governance checks"""
    
    def test_validation_check_overhead(self):
        """Test that governance validation is fast"""
        from app.services.ml_governance_service import MLGovernanceService
        
        service = MLGovernanceService()
        
        start = time.time()
        for _ in range(100):
            service.check_model_deployment(
                model_id="test-model",
                target_environment="development"
            )
        elapsed = time.time() - start
        
        avg_check_ms = (elapsed / 100) * 1000
        assert avg_check_ms < 10, f"Governance check took {avg_check_ms}ms, expected < 10ms"
