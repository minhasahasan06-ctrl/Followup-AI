"""
Dual-Write Migration Service for Memory System

Implements shadow-run pattern for migrating from legacy to new vector storage:
- Writes to both old and new systems simultaneously
- Compares retrieval results for validation
- Collects precision@k metrics for migration confidence

HIPAA Compliance: All operations logged, no PHI in comparison metrics
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import hashlib

from app.services.ml_observability import (
    MetricHistogram,
    MetricCounter,
    MetricGauge
)

logger = logging.getLogger(__name__)


@dataclass
class ComparisonResult:
    """Result of comparing old vs new retrieval systems"""
    query_hash: str
    old_ids: List[str]
    new_ids: List[str]
    precision_at_1: float
    precision_at_5: float
    precision_at_10: float
    recall_at_10: float
    latency_old_ms: float
    latency_new_ms: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_hash": self.query_hash,
            "old_count": len(self.old_ids),
            "new_count": len(self.new_ids),
            "precision_at_1": self.precision_at_1,
            "precision_at_5": self.precision_at_5,
            "precision_at_10": self.precision_at_10,
            "recall_at_10": self.recall_at_10,
            "latency_old_ms": self.latency_old_ms,
            "latency_new_ms": self.latency_new_ms,
            "timestamp": self.timestamp.isoformat()
        }


class DualWriteMigration:
    """
    Manages dual-write migration between legacy and new memory systems.
    
    Strategy:
    1. Write Phase: All writes go to both systems
    2. Shadow Read Phase: Reads from new system, compares with old
    3. Validation Phase: Collect precision@k metrics
    4. Cutover Phase: Switch primary to new system
    """
    
    def __init__(
        self,
        legacy_store: Optional[Any] = None,
        new_store: Optional[Any] = None,
        shadow_mode: bool = True
    ):
        self.legacy_store = legacy_store
        self.new_store = new_store
        self.shadow_mode = shadow_mode
        self.comparison_results: List[ComparisonResult] = []
        
        self._write_success_counter = MetricCounter("dual_write_success")
        self._write_failure_counter = MetricCounter("dual_write_failure")
        self._comparison_counter = MetricCounter("dual_write_comparisons")
        self._precision_gauge = MetricGauge("dual_write_precision_at_5")
        self._latency_histogram = MetricHistogram("dual_write_latency_ms")
    
    async def dual_write(
        self,
        memory_id: str,
        content: str,
        embedding: List[float],
        metadata: Dict[str, Any]
    ) -> Tuple[bool, bool]:
        """
        Write to both legacy and new systems.
        
        Returns:
            Tuple of (legacy_success, new_success)
        """
        legacy_success = False
        new_success = False
        
        if self.legacy_store:
            try:
                await self._write_legacy(memory_id, content, embedding, metadata)
                legacy_success = True
            except Exception as e:
                logger.error(f"Legacy write failed for {memory_id}: {e}")
                self._write_failure_counter.inc(labels={"system": "legacy"})
        
        if self.new_store:
            try:
                await self._write_new(memory_id, content, embedding, metadata)
                new_success = True
            except Exception as e:
                logger.error(f"New system write failed for {memory_id}: {e}")
                self._write_failure_counter.inc(labels={"system": "new"})
        
        if legacy_success and new_success:
            self._write_success_counter.inc()
        
        return legacy_success, new_success
    
    async def _write_legacy(
        self,
        memory_id: str,
        content: str,
        embedding: List[float],
        metadata: Dict[str, Any]
    ) -> None:
        """Write to legacy system (placeholder for actual implementation)"""
        if hasattr(self.legacy_store, 'store'):
            await self.legacy_store.store(memory_id, content, embedding, metadata)
    
    async def _write_new(
        self,
        memory_id: str,
        content: str,
        embedding: List[float],
        metadata: Dict[str, Any]
    ) -> None:
        """Write to new pgvector system (placeholder for actual implementation)"""
        if hasattr(self.new_store, 'store_long_term'):
            await self.new_store.store_long_term(
                agent_id=metadata.get('agent_id', 'default'),
                patient_id=metadata.get('patient_id'),
                memory_type=metadata.get('memory_type', 'general'),
                content=content,
                embedding=embedding,
                metadata=metadata
            )
    
    async def shadow_compare(
        self,
        query: str,
        query_embedding: List[float],
        filters: Dict[str, Any],
        top_k: int = 10
    ) -> ComparisonResult:
        """
        Execute query on both systems and compare results.
        
        This runs in shadow mode - new system results are compared
        but not returned to caller. Legacy results are authoritative.
        """
        import time
        
        query_hash = hashlib.sha256(query.encode()).hexdigest()[:16]
        
        old_start = time.time()
        old_results = await self._query_legacy(query_embedding, filters, top_k)
        old_latency = (time.time() - old_start) * 1000
        
        new_start = time.time()
        new_results = await self._query_new(query_embedding, filters, top_k)
        new_latency = (time.time() - new_start) * 1000
        
        old_ids = [r.get('id', r.get('memory_id', '')) for r in old_results]
        new_ids = [r.get('id', r.get('memory_id', '')) for r in new_results]
        
        precision_1 = self._calculate_precision(old_ids, new_ids, 1)
        precision_5 = self._calculate_precision(old_ids, new_ids, 5)
        precision_10 = self._calculate_precision(old_ids, new_ids, 10)
        recall_10 = self._calculate_recall(old_ids, new_ids, 10)
        
        result = ComparisonResult(
            query_hash=query_hash,
            old_ids=old_ids,
            new_ids=new_ids,
            precision_at_1=precision_1,
            precision_at_5=precision_5,
            precision_at_10=precision_10,
            recall_at_10=recall_10,
            latency_old_ms=old_latency,
            latency_new_ms=new_latency
        )
        
        self.comparison_results.append(result)
        self._comparison_counter.inc()
        self._precision_gauge.set(precision_5)
        self._latency_histogram.observe(new_latency)
        self._latency_histogram.observe(old_latency)
        
        logger.info(
            f"Shadow comparison: P@5={precision_5:.2f}, "
            f"old_latency={old_latency:.1f}ms, new_latency={new_latency:.1f}ms"
        )
        
        return result
    
    async def _query_legacy(
        self,
        embedding: List[float],
        filters: Dict[str, Any],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Query legacy system"""
        if self.legacy_store and hasattr(self.legacy_store, 'search'):
            return await self.legacy_store.search(embedding, filters, top_k)
        return []
    
    async def _query_new(
        self,
        embedding: List[float],
        filters: Dict[str, Any],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Query new pgvector system"""
        if self.new_store and hasattr(self.new_store, 'search_long_term'):
            results = await self.new_store.search_long_term(
                agent_id=filters.get('agent_id', 'default'),
                query_embedding=embedding,
                patient_id=filters.get('patient_id'),
                memory_type=filters.get('memory_type'),
                top_k=top_k
            )
            return results
        return []
    
    def _calculate_precision(
        self,
        ground_truth: List[str],
        predicted: List[str],
        k: int
    ) -> float:
        """Calculate precision@k - how many of top-k predictions are in ground truth"""
        if not predicted or not ground_truth:
            return 0.0
        
        predicted_k = set(predicted[:k])
        ground_truth_set = set(ground_truth)
        
        if not predicted_k:
            return 0.0
        
        overlap = len(predicted_k & ground_truth_set)
        return overlap / len(predicted_k)
    
    def _calculate_recall(
        self,
        ground_truth: List[str],
        predicted: List[str],
        k: int
    ) -> float:
        """Calculate recall@k - what fraction of ground truth is in top-k predictions"""
        if not ground_truth:
            return 0.0
        
        predicted_k = set(predicted[:k])
        ground_truth_set = set(ground_truth[:k])
        
        if not ground_truth_set:
            return 0.0
        
        overlap = len(predicted_k & ground_truth_set)
        return overlap / len(ground_truth_set)
    
    def get_migration_metrics(self) -> Dict[str, Any]:
        """Get aggregated migration metrics"""
        if not self.comparison_results:
            return {
                "total_comparisons": 0,
                "avg_precision_at_5": 0.0,
                "migration_ready": False
            }
        
        precisions = [r.precision_at_5 for r in self.comparison_results]
        avg_precision = sum(precisions) / len(precisions)
        
        latencies_new = [r.latency_new_ms for r in self.comparison_results]
        latencies_old = [r.latency_old_ms for r in self.comparison_results]
        
        return {
            "total_comparisons": len(self.comparison_results),
            "avg_precision_at_1": sum(r.precision_at_1 for r in self.comparison_results) / len(self.comparison_results),
            "avg_precision_at_5": avg_precision,
            "avg_precision_at_10": sum(r.precision_at_10 for r in self.comparison_results) / len(self.comparison_results),
            "avg_recall_at_10": sum(r.recall_at_10 for r in self.comparison_results) / len(self.comparison_results),
            "avg_latency_new_ms": sum(latencies_new) / len(latencies_new),
            "avg_latency_old_ms": sum(latencies_old) / len(latencies_old),
            "latency_improvement_pct": (
                (sum(latencies_old) - sum(latencies_new)) / sum(latencies_old) * 100
                if sum(latencies_old) > 0 else 0
            ),
            "migration_ready": avg_precision >= 0.8,
            "recommendation": "PROCEED" if avg_precision >= 0.8 else "CONTINUE_SHADOW"
        }
    
    def is_migration_ready(self, min_comparisons: int = 100, min_precision: float = 0.8) -> bool:
        """Check if migration is ready based on collected metrics"""
        if len(self.comparison_results) < min_comparisons:
            return False
        
        metrics = self.get_migration_metrics()
        return metrics["avg_precision_at_5"] >= min_precision
