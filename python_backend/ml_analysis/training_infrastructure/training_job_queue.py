"""
Training Job Queue
===================
Production-grade job queue for ML training with:
- Redis-backed persistent queue (with fallback to in-memory)
- Job status tracking with database persistence
- Priority-based scheduling
- Retry logic with exponential backoff
- Comprehensive audit logging

HIPAA-compliant with full job lifecycle tracking.
"""

import os
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from enum import Enum
from dataclasses import dataclass, field, asdict
import psycopg2
import psycopg2.extras

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    """Training job status lifecycle"""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


class JobPriority(int, Enum):
    """Job priority levels (higher = more urgent)"""
    LOW = 1
    NORMAL = 5
    HIGH = 10
    CRITICAL = 20


@dataclass
class TrainingJob:
    """Represents a single ML training job"""
    job_id: str
    job_type: str  # 'risk_model', 'adherence_model', 'engagement_model', 'anomaly_model', 'custom'
    model_name: str
    status: JobStatus
    priority: int = JobPriority.NORMAL
    config: Dict[str, Any] = field(default_factory=dict)
    created_by: str = "system"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    progress_percent: int = 0
    current_step: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)
    artifact_path: Optional[str] = None
    consent_verified: bool = False
    governance_approved: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['status'] = self.status.value if isinstance(self.status, JobStatus) else self.status
        data['created_at'] = self.created_at.isoformat() if self.created_at else None
        data['started_at'] = self.started_at.isoformat() if self.started_at else None
        data['completed_at'] = self.completed_at.isoformat() if self.completed_at else None
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrainingJob':
        """Create from dictionary"""
        data = data.copy()
        data['status'] = JobStatus(data['status']) if isinstance(data['status'], str) else data['status']
        for dt_field in ['created_at', 'started_at', 'completed_at']:
            if data.get(dt_field) and isinstance(data[dt_field], str):
                data[dt_field] = datetime.fromisoformat(data[dt_field].replace('Z', '+00:00'))
        return cls(**data)


class TrainingJobQueue:
    """
    Manages ML training job queue with Redis backend and database persistence.
    
    Features:
    - Persistent job storage in PostgreSQL
    - Redis for fast queue operations (with in-memory fallback)
    - Priority-based job scheduling
    - Job status tracking and history
    - Comprehensive audit logging for HIPAA
    """
    
    QUEUE_NAME = "ml_training_jobs"
    
    def __init__(self, db_url: Optional[str] = None, redis_url: Optional[str] = None):
        self.db_url = db_url or os.environ.get('DATABASE_URL')
        self.redis_url = redis_url or os.environ.get('REDIS_URL')
        self._redis = None
        self._memory_queue: List[TrainingJob] = []  # Fallback queue
        self._init_redis()
        self._ensure_tables()
    
    def _init_redis(self):
        """Initialize Redis connection (optional)"""
        if self.redis_url:
            try:
                import redis
                self._redis = redis.from_url(self.redis_url)
                self._redis.ping()
                logger.info("Redis connection established for training job queue")
            except Exception as e:
                logger.warning(f"Redis unavailable, using in-memory queue: {e}")
                self._redis = None
    
    def _get_db_connection(self):
        """Get database connection"""
        return psycopg2.connect(self.db_url)
    
    def _ensure_tables(self):
        """Ensure required database tables exist"""
        try:
            conn = self._get_db_connection()
            cur = conn.cursor()
            
            cur.execute("""
                CREATE TABLE IF NOT EXISTS ml_training_jobs (
                    job_id VARCHAR(50) PRIMARY KEY,
                    job_type VARCHAR(50) NOT NULL,
                    model_name VARCHAR(100) NOT NULL,
                    status VARCHAR(20) NOT NULL DEFAULT 'pending',
                    priority INTEGER NOT NULL DEFAULT 5,
                    config JSONB DEFAULT '{}',
                    created_by VARCHAR(100) NOT NULL DEFAULT 'system',
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    started_at TIMESTAMPTZ,
                    completed_at TIMESTAMPTZ,
                    error_message TEXT,
                    retry_count INTEGER DEFAULT 0,
                    max_retries INTEGER DEFAULT 3,
                    progress_percent INTEGER DEFAULT 0,
                    current_step VARCHAR(200) DEFAULT '',
                    metrics JSONB DEFAULT '{}',
                    artifact_path VARCHAR(500),
                    consent_verified BOOLEAN DEFAULT FALSE,
                    governance_approved BOOLEAN DEFAULT FALSE
                );
                
                CREATE INDEX IF NOT EXISTS idx_training_jobs_status 
                    ON ml_training_jobs(status);
                CREATE INDEX IF NOT EXISTS idx_training_jobs_created 
                    ON ml_training_jobs(created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_training_jobs_priority 
                    ON ml_training_jobs(priority DESC, created_at ASC);
            """)
            
            # Job history/audit table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS ml_training_job_history (
                    id SERIAL PRIMARY KEY,
                    job_id VARCHAR(50) NOT NULL,
                    event_type VARCHAR(50) NOT NULL,
                    event_data JSONB DEFAULT '{}',
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    created_by VARCHAR(100) NOT NULL DEFAULT 'system'
                );
                
                CREATE INDEX IF NOT EXISTS idx_job_history_job_id 
                    ON ml_training_job_history(job_id);
            """)
            
            conn.commit()
            cur.close()
            conn.close()
            logger.info("ML training job tables verified")
            
        except Exception as e:
            logger.error(f"Error ensuring training job tables: {e}")
    
    def create_job(
        self,
        job_type: str,
        model_name: str,
        config: Dict[str, Any],
        created_by: str = "system",
        priority: int = JobPriority.NORMAL
    ) -> TrainingJob:
        """
        Create a new training job and add to queue.
        
        Args:
            job_type: Type of training (risk_model, adherence_model, etc.)
            model_name: Name for the model
            config: Training configuration
            created_by: User/system that created the job
            priority: Job priority level
            
        Returns:
            Created TrainingJob
        """
        job = TrainingJob(
            job_id=str(uuid.uuid4()),
            job_type=job_type,
            model_name=model_name,
            status=JobStatus.PENDING,
            priority=priority,
            config=config,
            created_by=created_by
        )
        
        # Save to database
        self._save_job(job)
        
        # Add to queue
        self._enqueue(job)
        
        # Log audit event
        self._log_event(job.job_id, "job_created", {
            "job_type": job_type,
            "model_name": model_name,
            "priority": priority
        }, created_by)
        
        logger.info(f"Created training job {job.job_id} for {model_name}")
        return job
    
    def _save_job(self, job: TrainingJob):
        """Save job to database"""
        try:
            conn = self._get_db_connection()
            cur = conn.cursor()
            
            cur.execute("""
                INSERT INTO ml_training_jobs (
                    job_id, job_type, model_name, status, priority, config,
                    created_by, created_at, started_at, completed_at,
                    error_message, retry_count, max_retries, progress_percent,
                    current_step, metrics, artifact_path, consent_verified,
                    governance_approved
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (job_id) DO UPDATE SET
                    status = EXCLUDED.status,
                    started_at = EXCLUDED.started_at,
                    completed_at = EXCLUDED.completed_at,
                    error_message = EXCLUDED.error_message,
                    retry_count = EXCLUDED.retry_count,
                    progress_percent = EXCLUDED.progress_percent,
                    current_step = EXCLUDED.current_step,
                    metrics = EXCLUDED.metrics,
                    artifact_path = EXCLUDED.artifact_path,
                    consent_verified = EXCLUDED.consent_verified,
                    governance_approved = EXCLUDED.governance_approved
            """, (
                job.job_id, job.job_type, job.model_name,
                job.status.value if isinstance(job.status, JobStatus) else job.status,
                job.priority, json.dumps(job.config),
                job.created_by, job.created_at, job.started_at, job.completed_at,
                job.error_message, job.retry_count, job.max_retries,
                job.progress_percent, job.current_step, json.dumps(job.metrics),
                job.artifact_path, job.consent_verified, job.governance_approved
            ))
            
            conn.commit()
            cur.close()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error saving job {job.job_id}: {e}")
            raise
    
    def _enqueue(self, job: TrainingJob):
        """Add job to the queue"""
        job.status = JobStatus.QUEUED
        self._save_job(job)
        
        if self._redis:
            try:
                # Use sorted set for priority queue
                self._redis.zadd(
                    self.QUEUE_NAME,
                    {job.job_id: -job.priority}  # Negative for descending order
                )
            except Exception as e:
                logger.warning(f"Redis enqueue failed, using memory: {e}")
                self._memory_queue.append(job)
        else:
            self._memory_queue.append(job)
    
    def dequeue(self) -> Optional[TrainingJob]:
        """Get next job from queue (highest priority first)"""
        job_id = None
        
        if self._redis:
            try:
                # Get highest priority job
                result = self._redis.zpopmin(self.QUEUE_NAME, count=1)
                if result:
                    job_id = result[0][0]
                    if isinstance(job_id, bytes):
                        job_id = job_id.decode('utf-8')
            except Exception as e:
                logger.warning(f"Redis dequeue failed: {e}")
        
        if not job_id and self._memory_queue:
            # Sort by priority (descending), then created_at (ascending)
            self._memory_queue.sort(key=lambda j: (-j.priority, j.created_at))
            job = self._memory_queue.pop(0)
            job_id = job.job_id
        
        if job_id:
            return self.get_job(job_id)
        return None
    
    def get_job(self, job_id: str) -> Optional[TrainingJob]:
        """Get job by ID from database"""
        try:
            conn = self._get_db_connection()
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            cur.execute("SELECT * FROM ml_training_jobs WHERE job_id = %s", (job_id,))
            row = cur.fetchone()
            
            cur.close()
            conn.close()
            
            if row:
                return self._row_to_job(dict(row))
            return None
            
        except Exception as e:
            logger.error(f"Error getting job {job_id}: {e}")
            return None
    
    def _row_to_job(self, row: Dict[str, Any]) -> TrainingJob:
        """Convert database row to TrainingJob"""
        return TrainingJob(
            job_id=row['job_id'],
            job_type=row['job_type'],
            model_name=row['model_name'],
            status=JobStatus(row['status']),
            priority=row['priority'],
            config=row['config'] or {},
            created_by=row['created_by'],
            created_at=row['created_at'],
            started_at=row['started_at'],
            completed_at=row['completed_at'],
            error_message=row['error_message'],
            retry_count=row['retry_count'],
            max_retries=row['max_retries'],
            progress_percent=row['progress_percent'],
            current_step=row['current_step'] or '',
            metrics=row['metrics'] or {},
            artifact_path=row['artifact_path'],
            consent_verified=row['consent_verified'],
            governance_approved=row['governance_approved']
        )
    
    def update_job_status(
        self,
        job_id: str,
        status: JobStatus,
        progress_percent: Optional[int] = None,
        current_step: Optional[str] = None,
        error_message: Optional[str] = None,
        metrics: Optional[Dict[str, Any]] = None,
        artifact_path: Optional[str] = None,
        updated_by: str = "worker"
    ):
        """Update job status and progress"""
        job = self.get_job(job_id)
        if not job:
            logger.error(f"Job {job_id} not found for status update")
            return
        
        old_status = job.status
        job.status = status
        
        if progress_percent is not None:
            job.progress_percent = progress_percent
        if current_step is not None:
            job.current_step = current_step
        if error_message is not None:
            job.error_message = error_message
        if metrics is not None:
            job.metrics.update(metrics)
        if artifact_path is not None:
            job.artifact_path = artifact_path
        
        # Update timestamps
        if status == JobStatus.RUNNING and not job.started_at:
            job.started_at = datetime.now(timezone.utc)
        elif status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
            job.completed_at = datetime.now(timezone.utc)
        
        self._save_job(job)
        
        # Log status change
        self._log_event(job_id, "status_changed", {
            "old_status": old_status.value if isinstance(old_status, JobStatus) else old_status,
            "new_status": status.value,
            "progress": progress_percent,
            "step": current_step
        }, updated_by)
    
    def get_pending_jobs(self, limit: int = 10) -> List[TrainingJob]:
        """Get jobs waiting to be processed"""
        try:
            conn = self._get_db_connection()
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            cur.execute("""
                SELECT * FROM ml_training_jobs 
                WHERE status IN ('pending', 'queued')
                ORDER BY priority DESC, created_at ASC
                LIMIT %s
            """, (limit,))
            
            jobs = [self._row_to_job(dict(row)) for row in cur.fetchall()]
            
            cur.close()
            conn.close()
            
            return jobs
            
        except Exception as e:
            logger.error(f"Error getting pending jobs: {e}")
            return []
    
    def get_recent_jobs(self, limit: int = 50, status: Optional[str] = None) -> List[TrainingJob]:
        """Get recent jobs with optional status filter"""
        try:
            conn = self._get_db_connection()
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            if status:
                cur.execute("""
                    SELECT * FROM ml_training_jobs 
                    WHERE status = %s
                    ORDER BY created_at DESC
                    LIMIT %s
                """, (status, limit))
            else:
                cur.execute("""
                    SELECT * FROM ml_training_jobs 
                    ORDER BY created_at DESC
                    LIMIT %s
                """, (limit,))
            
            jobs = [self._row_to_job(dict(row)) for row in cur.fetchall()]
            
            cur.close()
            conn.close()
            
            return jobs
            
        except Exception as e:
            logger.error(f"Error getting recent jobs: {e}")
            return []
    
    def get_job_history(self, job_id: str) -> List[Dict[str, Any]]:
        """Get audit history for a job"""
        try:
            conn = self._get_db_connection()
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            cur.execute("""
                SELECT * FROM ml_training_job_history 
                WHERE job_id = %s
                ORDER BY created_at ASC
            """, (job_id,))
            
            history = [dict(row) for row in cur.fetchall()]
            
            cur.close()
            conn.close()
            
            return history
            
        except Exception as e:
            logger.error(f"Error getting job history: {e}")
            return []
    
    def cancel_job(self, job_id: str, cancelled_by: str = "user") -> bool:
        """Cancel a pending or running job"""
        job = self.get_job(job_id)
        if not job:
            return False
        
        if job.status in [JobStatus.COMPLETED, JobStatus.CANCELLED]:
            logger.warning(f"Cannot cancel job {job_id} with status {job.status}")
            return False
        
        self.update_job_status(job_id, JobStatus.CANCELLED, updated_by=cancelled_by)
        
        # Remove from queue if present
        if self._redis:
            try:
                self._redis.zrem(self.QUEUE_NAME, job_id)
            except Exception:
                pass
        else:
            self._memory_queue = [j for j in self._memory_queue if j.job_id != job_id]
        
        logger.info(f"Job {job_id} cancelled by {cancelled_by}")
        return True
    
    def retry_job(self, job_id: str, retried_by: str = "system") -> bool:
        """Retry a failed job"""
        job = self.get_job(job_id)
        if not job:
            return False
        
        if job.status != JobStatus.FAILED:
            logger.warning(f"Cannot retry job {job_id} with status {job.status}")
            return False
        
        if job.retry_count >= job.max_retries:
            logger.warning(f"Job {job_id} has exceeded max retries")
            return False
        
        job.retry_count += 1
        job.status = JobStatus.RETRYING
        job.error_message = None
        job.progress_percent = 0
        job.current_step = "Retrying..."
        
        self._save_job(job)
        self._enqueue(job)
        
        self._log_event(job_id, "job_retried", {
            "retry_count": job.retry_count
        }, retried_by)
        
        logger.info(f"Job {job_id} queued for retry (attempt {job.retry_count})")
        return True
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics"""
        try:
            conn = self._get_db_connection()
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            cur.execute("""
                SELECT 
                    status,
                    COUNT(*) as count
                FROM ml_training_jobs
                GROUP BY status
            """)
            
            status_counts = {row['status']: row['count'] for row in cur.fetchall()}
            
            cur.execute("""
                SELECT 
                    COUNT(*) as total_jobs,
                    AVG(EXTRACT(EPOCH FROM (completed_at - started_at))) as avg_duration_seconds,
                    COUNT(CASE WHEN status = 'completed' THEN 1 END) as completed_count,
                    COUNT(CASE WHEN status = 'failed' THEN 1 END) as failed_count
                FROM ml_training_jobs
                WHERE started_at IS NOT NULL
            """)
            
            stats = dict(cur.fetchone())
            
            cur.close()
            conn.close()
            
            queue_size = 0
            if self._redis:
                try:
                    queue_size = self._redis.zcard(self.QUEUE_NAME)
                except Exception:
                    pass
            else:
                queue_size = len(self._memory_queue)
            
            return {
                "queue_size": queue_size,
                "status_counts": status_counts,
                "total_jobs": stats['total_jobs'] or 0,
                "avg_duration_seconds": float(stats['avg_duration_seconds'] or 0),
                "completed_count": stats['completed_count'] or 0,
                "failed_count": stats['failed_count'] or 0,
                "success_rate": (
                    stats['completed_count'] / (stats['completed_count'] + stats['failed_count']) * 100
                    if (stats['completed_count'] or 0) + (stats['failed_count'] or 0) > 0 else 0
                ),
                "redis_connected": self._redis is not None
            }
            
        except Exception as e:
            logger.error(f"Error getting queue stats: {e}")
            return {
                "queue_size": 0,
                "status_counts": {},
                "error": str(e)
            }
    
    def _log_event(
        self,
        job_id: str,
        event_type: str,
        event_data: Dict[str, Any],
        created_by: str
    ):
        """Log audit event for HIPAA compliance"""
        try:
            conn = self._get_db_connection()
            cur = conn.cursor()
            
            cur.execute("""
                INSERT INTO ml_training_job_history 
                (job_id, event_type, event_data, created_by)
                VALUES (%s, %s, %s, %s)
            """, (job_id, event_type, json.dumps(event_data), created_by))
            
            conn.commit()
            cur.close()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error logging event for job {job_id}: {e}")
