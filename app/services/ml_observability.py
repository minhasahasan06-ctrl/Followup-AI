"""
ML/AI Observability Service
============================
Comprehensive metrics collection and monitoring for:
1. Retrieval latency (p50, p95, p99)
2. Similarity score distribution
3. Vector index performance
4. LLM token usage tracking
5. API error rates
6. Embedding generation metrics

This module provides production-grade observability for all AI/ML operations.
"""

import os
import time
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from collections import defaultdict
from threading import Lock
import json

logger = logging.getLogger(__name__)


@dataclass
class MetricPoint:
    """Single metric measurement"""
    value: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    labels: Dict[str, str] = field(default_factory=dict)


class MetricHistogram:
    """Histogram for latency and distribution metrics"""
    
    def __init__(self, name: str, max_samples: int = 10000):
        self.name = name
        self.max_samples = max_samples
        self._values: List[float] = []
        self._lock = Lock()
        self._total_count = 0
        self._sum = 0.0
    
    def observe(self, value: float):
        """Record a value"""
        with self._lock:
            self._values.append(value)
            self._total_count += 1
            self._sum += value
            
            if len(self._values) > self.max_samples:
                self._values = self._values[-self.max_samples:]
    
    def percentile(self, p: float) -> float:
        """Get percentile value"""
        with self._lock:
            if not self._values:
                return 0.0
            sorted_vals = sorted(self._values)
            idx = int(len(sorted_vals) * p)
            return sorted_vals[min(idx, len(sorted_vals) - 1)]
    
    def mean(self) -> float:
        """Get mean value"""
        with self._lock:
            return self._sum / self._total_count if self._total_count > 0 else 0.0
    
    def count(self) -> int:
        """Get total count"""
        return self._total_count
    
    def stats(self) -> Dict[str, float]:
        """Get all statistics"""
        return {
            "count": self._total_count,
            "sum": self._sum,
            "mean": self.mean(),
            "p50": self.percentile(0.5),
            "p90": self.percentile(0.9),
            "p95": self.percentile(0.95),
            "p99": self.percentile(0.99),
            "min": min(self._values) if self._values else 0,
            "max": max(self._values) if self._values else 0,
        }


class MetricCounter:
    """Counter for events and error tracking"""
    
    def __init__(self, name: str):
        self.name = name
        self._counts: Dict[str, int] = defaultdict(int)
        self._lock = Lock()
    
    def inc(self, labels: Optional[Dict[str, str]] = None, value: int = 1):
        """Increment counter"""
        key = json.dumps(labels or {}, sort_keys=True)
        with self._lock:
            self._counts[key] += value
    
    def get(self, labels: Optional[Dict[str, str]] = None) -> int:
        """Get counter value for labels"""
        key = json.dumps(labels or {}, sort_keys=True)
        return self._counts.get(key, 0)
    
    def total(self) -> int:
        """Get total count across all labels"""
        return sum(self._counts.values())
    
    def by_label(self) -> Dict[str, int]:
        """Get counts grouped by labels"""
        return dict(self._counts)


class MetricGauge:
    """Gauge for current values"""
    
    def __init__(self, name: str):
        self.name = name
        self._value = 0.0
        self._lock = Lock()
        self._last_updated: Optional[datetime] = None
    
    def set(self, value: float):
        """Set gauge value"""
        with self._lock:
            self._value = value
            self._last_updated = datetime.utcnow()
    
    def inc(self, value: float = 1):
        """Increment gauge"""
        with self._lock:
            self._value += value
            self._last_updated = datetime.utcnow()
    
    def dec(self, value: float = 1):
        """Decrement gauge"""
        with self._lock:
            self._value -= value
            self._last_updated = datetime.utcnow()
    
    def get(self) -> float:
        """Get current value"""
        return self._value


class MLObservabilityService:
    """
    Central observability service for ML/AI operations.
    
    Collects and exposes metrics for:
    - Vector search performance
    - Embedding generation
    - LLM API calls
    - Error rates and failures
    """
    
    def __init__(self):
        self.retrieval_latency = MetricHistogram("memory_retrieval_latency_ms")
        self.embedding_latency = MetricHistogram("embedding_generation_latency_ms")
        self.llm_latency = MetricHistogram("llm_completion_latency_ms")
        
        self.similarity_scores = MetricHistogram("retrieval_similarity_score")
        self.result_counts = MetricHistogram("retrieval_result_count")
        
        self.api_calls = MetricCounter("openai_api_calls")
        self.api_errors = MetricCounter("openai_api_errors")
        self.phi_detections = MetricCounter("phi_detections")
        self.memory_operations = MetricCounter("memory_operations")
        
        self.prompt_tokens = MetricCounter("llm_prompt_tokens")
        self.completion_tokens = MetricCounter("llm_completion_tokens")
        self.embedding_tokens = MetricCounter("embedding_tokens")
        
        self.active_memory_count = MetricGauge("active_memory_count")
        self.index_size_bytes = MetricGauge("vector_index_size_bytes")
        self.embedding_queue_size = MetricGauge("embedding_queue_size")
        
        self._error_log: List[Dict[str, Any]] = []
        self._max_error_log = 1000
        self._lock = Lock()
        
        logger.info("ML Observability Service initialized")
    
    def record_retrieval(
        self,
        latency_ms: float,
        result_count: int,
        similarity_scores: List[float],
        agent_id: Optional[str] = None,
        patient_id: Optional[str] = None,
        success: bool = True
    ):
        """Record a memory retrieval operation"""
        self.retrieval_latency.observe(latency_ms)
        self.result_counts.observe(result_count)
        
        for score in similarity_scores:
            self.similarity_scores.observe(score)
        
        labels = {"agent_id": agent_id or "unknown"}
        self.memory_operations.inc({"operation": "search", "success": str(success)})
        
        if not success:
            self._log_error("retrieval_failure", {
                "latency_ms": latency_ms,
                "agent_id": agent_id,
                "patient_id": patient_id,
            })
    
    def record_embedding_generation(
        self,
        latency_ms: float,
        token_count: int,
        model: str,
        success: bool = True
    ):
        """Record embedding generation"""
        self.embedding_latency.observe(latency_ms)
        self.embedding_tokens.inc({"model": model}, token_count)
        self.api_calls.inc({"operation": "embedding", "model": model, "success": str(success)})
        
        if not success:
            self.api_errors.inc({"operation": "embedding", "model": model})
    
    def record_llm_completion(
        self,
        latency_ms: float,
        prompt_tokens: int,
        completion_tokens: int,
        model: str,
        success: bool = True
    ):
        """Record LLM completion"""
        self.llm_latency.observe(latency_ms)
        self.prompt_tokens.inc({"model": model}, prompt_tokens)
        self.completion_tokens.inc({"model": model}, completion_tokens)
        self.api_calls.inc({"operation": "completion", "model": model, "success": str(success)})
        
        if not success:
            self.api_errors.inc({"operation": "completion", "model": model})
    
    def record_phi_detection(
        self,
        category: str,
        action: str,
        operation: str
    ):
        """Record PHI detection event"""
        self.phi_detections.inc({
            "category": category,
            "action": action,
            "operation": operation,
        })
    
    def record_memory_store(
        self,
        latency_ms: float,
        agent_id: Optional[str] = None,
        success: bool = True
    ):
        """Record memory store operation"""
        self.memory_operations.inc({"operation": "store", "success": str(success)})
        
        if success:
            self.active_memory_count.inc()
        else:
            self._log_error("store_failure", {
                "latency_ms": latency_ms,
                "agent_id": agent_id,
            })
    
    def update_index_size(self, size_bytes: int):
        """Update vector index size gauge"""
        self.index_size_bytes.set(size_bytes)
    
    def _log_error(self, error_type: str, details: Dict[str, Any]):
        """Log error for debugging"""
        with self._lock:
            self._error_log.append({
                "timestamp": datetime.utcnow().isoformat(),
                "type": error_type,
                **details
            })
            
            if len(self._error_log) > self._max_error_log:
                self._error_log = self._error_log[-self._max_error_log:]
    
    def get_retrieval_metrics(self) -> Dict[str, Any]:
        """Get retrieval performance metrics"""
        return {
            "latency": self.retrieval_latency.stats(),
            "similarity_distribution": self.similarity_scores.stats(),
            "result_counts": self.result_counts.stats(),
        }
    
    def get_embedding_metrics(self) -> Dict[str, Any]:
        """Get embedding generation metrics"""
        return {
            "latency": self.embedding_latency.stats(),
            "token_usage": self.embedding_tokens.by_label(),
        }
    
    def get_llm_metrics(self) -> Dict[str, Any]:
        """Get LLM usage metrics"""
        return {
            "latency": self.llm_latency.stats(),
            "prompt_tokens": self.prompt_tokens.by_label(),
            "completion_tokens": self.completion_tokens.by_label(),
        }
    
    def get_error_metrics(self) -> Dict[str, Any]:
        """Get error rate metrics"""
        total_calls = self.api_calls.total()
        total_errors = self.api_errors.total()
        
        return {
            "total_api_calls": total_calls,
            "total_api_errors": total_errors,
            "error_rate": total_errors / total_calls if total_calls > 0 else 0,
            "errors_by_operation": self.api_errors.by_label(),
            "recent_errors": self._error_log[-10:],
        }
    
    def get_phi_metrics(self) -> Dict[str, Any]:
        """Get PHI detection metrics"""
        return {
            "total_detections": self.phi_detections.total(),
            "by_category": self.phi_detections.by_label(),
        }
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get system-level metrics"""
        return {
            "active_memory_count": self.active_memory_count.get(),
            "index_size_bytes": self.index_size_bytes.get(),
            "embedding_queue_size": self.embedding_queue_size.get(),
        }
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all observability metrics"""
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "retrieval": self.get_retrieval_metrics(),
            "embedding": self.get_embedding_metrics(),
            "llm": self.get_llm_metrics(),
            "errors": self.get_error_metrics(),
            "phi": self.get_phi_metrics(),
            "system": self.get_system_metrics(),
            "memory_operations": self.memory_operations.by_label(),
        }
    
    def check_health(self) -> Dict[str, Any]:
        """Health check with SLA thresholds"""
        retrieval_p99 = self.retrieval_latency.percentile(0.99)
        embedding_p99 = self.embedding_latency.percentile(0.99)
        error_rate = self.get_error_metrics()["error_rate"]
        
        issues = []
        
        if retrieval_p99 > 1000:
            issues.append(f"Retrieval p99 latency too high: {retrieval_p99:.0f}ms (threshold: 1000ms)")
        
        if embedding_p99 > 2000:
            issues.append(f"Embedding p99 latency too high: {embedding_p99:.0f}ms (threshold: 2000ms)")
        
        if error_rate > 0.05:
            issues.append(f"API error rate too high: {error_rate:.2%} (threshold: 5%)")
        
        return {
            "healthy": len(issues) == 0,
            "issues": issues,
            "metrics": {
                "retrieval_p99_ms": retrieval_p99,
                "embedding_p99_ms": embedding_p99,
                "error_rate": error_rate,
            }
        }


_observability_service: Optional[MLObservabilityService] = None


def get_observability_service() -> MLObservabilityService:
    """Get singleton observability service"""
    global _observability_service
    if _observability_service is None:
        _observability_service = MLObservabilityService()
    return _observability_service
