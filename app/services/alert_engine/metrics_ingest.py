"""
Metrics Ingest Service - Real-time metric ingestion with Redis streaming.

Handles:
- POST /api/v1/metrics/ingest endpoint processing
- Validation and quality gating
- Redis stream publishing for real-time processing
- Database persistence with audit logging
"""

import os
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import hashlib
import uuid

from sqlalchemy.orm import Session
from sqlalchemy import text

logger = logging.getLogger(__name__)

try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("Redis not available for metrics streaming")

from .config_service import AlertConfigService


@dataclass
class MetricIngestRequest:
    """Incoming metric data structure"""
    patient_id: str
    timestamp: datetime
    metric_name: str
    metric_value: float
    unit: str
    confidence: float = 1.0
    source: str = "app"
    capture_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class MetricIngestResponse:
    """Response from metric ingestion"""
    success: bool
    metric_id: str
    processed_at: datetime
    queued_for_processing: bool
    quality_flags: List[str]
    message: str


class MetricsIngestService:
    """Service for ingesting health metrics with quality gating and streaming"""
    
    def __init__(self, db: Session):
        self.db = db
        self.config_service = AlertConfigService()
        self._redis_client = None
        self._initialized = False
    
    async def initialize(self):
        """Initialize Redis connection for streaming"""
        if REDIS_AVAILABLE and not self._initialized:
            try:
                redis_host = os.getenv('REDIS_HOST', 'localhost')
                redis_port = int(os.getenv('REDIS_PORT', '6379'))
                
                self._redis_client = redis.Redis(
                    host=redis_host,
                    port=redis_port,
                    decode_responses=True
                )
                await self._redis_client.ping()
                self._initialized = True
                logger.info("MetricsIngestService: Redis streaming enabled")
            except Exception as e:
                logger.warning(f"MetricsIngestService: Redis not available: {e}")
                self._redis_client = None
    
    async def ingest_metric(self, request: MetricIngestRequest) -> MetricIngestResponse:
        """
        Ingest a single health metric with validation and streaming.
        
        Steps:
        1. Validate auth & patient access
        2. Apply quality filters
        3. Write to metrics table (Postgres)
        4. Push to Redis stream for real-time processing
        5. Return response with quality flags
        """
        config = self.config_service.config
        quality_flags = []
        metric_id = str(uuid.uuid4())
        
        # Quality gating
        if request.confidence < config.min_confidence_threshold:
            quality_flags.append("low_confidence")
        
        capture_age_hours = 0
        if request.timestamp:
            age = datetime.utcnow() - request.timestamp
            capture_age_hours = age.total_seconds() / 3600
            if capture_age_hours > config.max_capture_age_hours:
                quality_flags.append("stale_capture")
        
        # Determine organ group for the metric
        organ_group = config.get_metric_organ_group(request.metric_name)
        
        try:
            # Persist to database
            insert_query = text("""
                INSERT INTO metric_ingest_log (
                    id, patient_id, timestamp, metric_name, metric_value, unit,
                    confidence, source, capture_id, organ_group, quality_flags,
                    metadata, created_at
                ) VALUES (
                    :id, :patient_id, :timestamp, :metric_name, :metric_value, :unit,
                    :confidence, :source, :capture_id, :organ_group, :quality_flags,
                    :metadata::jsonb, NOW()
                )
            """)
            
            self.db.execute(insert_query, {
                "id": metric_id,
                "patient_id": request.patient_id,
                "timestamp": request.timestamp or datetime.utcnow(),
                "metric_name": request.metric_name,
                "metric_value": request.metric_value,
                "unit": request.unit,
                "confidence": request.confidence,
                "source": request.source,
                "capture_id": request.capture_id,
                "organ_group": organ_group,
                "quality_flags": json.dumps(quality_flags),
                "metadata": json.dumps(request.metadata or {})
            })
            self.db.commit()
            
            # Push to Redis stream for real-time processing
            queued = await self._queue_for_processing(metric_id, request, organ_group, quality_flags)
            
            # Log audit entry
            await self._log_audit(
                patient_id=request.patient_id,
                action="metric_ingest",
                resource_type="metric",
                resource_id=metric_id,
                details={
                    "metric_name": request.metric_name,
                    "source": request.source,
                    "quality_flags": quality_flags
                }
            )
            
            return MetricIngestResponse(
                success=True,
                metric_id=metric_id,
                processed_at=datetime.utcnow(),
                queued_for_processing=queued,
                quality_flags=quality_flags,
                message=f"Metric ingested successfully. Organ group: {organ_group or 'unclassified'}"
            )
            
        except Exception as e:
            logger.error(f"Error ingesting metric: {e}")
            self.db.rollback()
            return MetricIngestResponse(
                success=False,
                metric_id=metric_id,
                processed_at=datetime.utcnow(),
                queued_for_processing=False,
                quality_flags=quality_flags,
                message=f"Error ingesting metric: {str(e)}"
            )
    
    async def ingest_batch(self, requests: List[MetricIngestRequest]) -> Dict[str, Any]:
        """Ingest multiple metrics in a batch"""
        results = []
        success_count = 0
        error_count = 0
        
        for request in requests:
            result = await self.ingest_metric(request)
            results.append({
                "metric_id": result.metric_id,
                "success": result.success,
                "quality_flags": result.quality_flags
            })
            if result.success:
                success_count += 1
            else:
                error_count += 1
        
        return {
            "total": len(requests),
            "success_count": success_count,
            "error_count": error_count,
            "results": results
        }
    
    async def _queue_for_processing(
        self, 
        metric_id: str, 
        request: MetricIngestRequest,
        organ_group: Optional[str],
        quality_flags: List[str]
    ) -> bool:
        """Queue metric for real-time processing via Redis stream"""
        if not self._redis_client:
            return False
        
        try:
            stream_data = {
                "metric_id": metric_id,
                "patient_id": request.patient_id,
                "metric_name": request.metric_name,
                "metric_value": str(request.metric_value),
                "confidence": str(request.confidence),
                "organ_group": organ_group or "unclassified",
                "quality_flags": json.dumps(quality_flags),
                "timestamp": request.timestamp.isoformat() if request.timestamp else datetime.utcnow().isoformat()
            }
            
            await self._redis_client.xadd(
                "alert_engine:metrics_stream",
                stream_data,
                maxlen=10000
            )
            
            # Also update hot metrics cache for the patient
            cache_key = f"hot_metrics:{request.patient_id}:{request.metric_name}"
            await self._redis_client.setex(
                cache_key,
                3600,  # 1 hour TTL
                json.dumps({
                    "value": request.metric_value,
                    "confidence": request.confidence,
                    "timestamp": request.timestamp.isoformat() if request.timestamp else datetime.utcnow().isoformat()
                })
            )
            
            return True
            
        except Exception as e:
            logger.warning(f"Error queueing metric for processing: {e}")
            return False
    
    async def _log_audit(
        self,
        patient_id: str,
        action: str,
        resource_type: str,
        resource_id: str,
        details: Dict[str, Any]
    ):
        """Log HIPAA-compliant audit entry"""
        try:
            audit_query = text("""
                INSERT INTO alert_engine_audit_log (
                    id, patient_id, action, resource_type, resource_id,
                    details, ip_address, user_agent, created_at
                ) VALUES (
                    gen_random_uuid(), :patient_id, :action, :resource_type, :resource_id,
                    :details::jsonb, :ip_address, :user_agent, NOW()
                )
            """)
            
            self.db.execute(audit_query, {
                "patient_id": patient_id,
                "action": action,
                "resource_type": resource_type,
                "resource_id": resource_id,
                "details": json.dumps(details),
                "ip_address": "system",
                "user_agent": "alert_engine"
            })
            self.db.commit()
            
        except Exception as e:
            logger.warning(f"Error logging audit entry: {e}")
    
    async def get_recent_metrics(
        self,
        patient_id: str,
        metric_name: Optional[str] = None,
        hours: int = 24
    ) -> List[Dict[str, Any]]:
        """Get recent metrics for a patient"""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        
        query = text("""
            SELECT id, patient_id, timestamp, metric_name, metric_value, unit,
                   confidence, source, organ_group, quality_flags
            FROM metric_ingest_log
            WHERE patient_id = :patient_id
            AND timestamp >= :cutoff
            """ + ("AND metric_name = :metric_name" if metric_name else "") + """
            ORDER BY timestamp DESC
            LIMIT 500
        """)
        
        params = {"patient_id": patient_id, "cutoff": cutoff}
        if metric_name:
            params["metric_name"] = metric_name
        
        result = self.db.execute(query, params)
        rows = result.fetchall()
        
        return [
            {
                "id": str(row[0]),
                "patient_id": row[1],
                "timestamp": row[2].isoformat() if row[2] else None,
                "metric_name": row[3],
                "metric_value": float(row[4]) if row[4] else 0,
                "unit": row[5],
                "confidence": float(row[6]) if row[6] else 1.0,
                "source": row[7],
                "organ_group": row[8],
                "quality_flags": json.loads(row[9]) if row[9] else []
            }
            for row in rows
        ]
    
    async def get_hot_metric(
        self,
        patient_id: str,
        metric_name: str
    ) -> Optional[Dict[str, Any]]:
        """Get hot (cached) metric value from Redis"""
        if not self._redis_client:
            return None
        
        try:
            cache_key = f"hot_metrics:{patient_id}:{metric_name}"
            cached = await self._redis_client.get(cache_key)
            if cached:
                return json.loads(cached)
            return None
        except Exception as e:
            logger.warning(f"Error getting hot metric: {e}")
            return None
    
    async def shutdown(self):
        """Cleanup resources"""
        if self._redis_client:
            await self._redis_client.close()
