"""
Device Sync Worker - Production-Grade Background Sync

Handles:
- Scheduled device data synchronization
- Background sync jobs
- Rate limiting and retry logic
- Data quality validation
- Health section routing
- ML training data pipeline

This worker runs as a background task to keep device data up to date.
"""

import os
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from enum import Enum

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

from app.services.vendor_adapters import (
    get_vendor_adapter,
    TokenInfo,
    SyncResult,
    NormalizedReading,
    HealthSection as VendorHealthSection,
)
from app.services.health_section_analytics import (
    HealthSectionAnalyticsEngine,
    HealthSection,
)

logger = logging.getLogger(__name__)


class SyncJobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class SyncJob:
    id: str
    user_id: str
    device_id: str
    vendor_id: str
    status: SyncJobStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    records_fetched: int = 0
    records_processed: int = 0
    error_message: Optional[str] = None
    retry_count: int = 0


class DeviceSyncWorker:
    """
    Background worker for device data synchronization.
    
    Features:
    - Concurrent sync support
    - Rate limiting per vendor
    - Automatic retry with backoff
    - Health section data routing
    - ML training data extraction
    """
    
    MAX_CONCURRENT_SYNCS = 5
    MAX_RETRIES = 3
    RETRY_BACKOFF_SECONDS = [60, 300, 900]  # 1min, 5min, 15min
    
    # Vendor-specific rate limits (requests per minute)
    VENDOR_RATE_LIMITS = {
        "fitbit": 150,
        "withings": 120,
        "oura": 100,
        "google_fit": 500,
        "ihealth": 60,
        "garmin": 100,
        "whoop": 100,
        "dexcom": 50,
    }
    
    def __init__(self, database_url: str):
        self.database_url = database_url
        self._running = False
        self._sync_semaphore = asyncio.Semaphore(self.MAX_CONCURRENT_SYNCS)
        self._vendor_last_request: Dict[str, datetime] = {}
        self._engine = None
        self._session_factory = None
    
    async def initialize(self):
        """Initialize database connection"""
        # Convert PostgreSQL URL to asyncpg-compatible format
        # asyncpg uses 'ssl' instead of 'sslmode'
        db_url = self.database_url.replace("postgresql://", "postgresql+asyncpg://")
        # Remove sslmode from query params as asyncpg handles SSL differently
        if "?sslmode=" in db_url:
            db_url = db_url.split("?sslmode=")[0]
        elif "&sslmode=" in db_url:
            parts = db_url.split("&sslmode=")
            db_url = parts[0] + ("&" + parts[1].split("&", 1)[1] if "&" in parts[1] else "")
        
        self._engine = create_async_engine(
            db_url,
            pool_size=10,
            max_overflow=20,
            connect_args={"ssl": "require"},  # Use asyncpg's ssl parameter
        )
        self._session_factory = async_sessionmaker(
            self._engine, class_=AsyncSession, expire_on_commit=False
        )
    
    async def shutdown(self):
        """Cleanup resources"""
        self._running = False
        if self._engine:
            await self._engine.dispose()
    
    async def get_session(self) -> AsyncSession:
        """Get database session"""
        if self._session_factory is None:
            raise RuntimeError("Database not initialized. Call initialize() first.")
        return self._session_factory()
    
    async def start(self):
        """Start the sync worker as a background task (non-blocking)"""
        await self.initialize()
        self._running = True
        logger.info("Device sync worker started")
        
        # Start the worker loop as a background task so it doesn't block server startup
        asyncio.create_task(self._run_worker_loop())
    
    async def _run_worker_loop(self):
        """Background worker loop - runs independently"""
        while self._running:
            try:
                await self._process_pending_syncs()
                await asyncio.sleep(30)  # Check for new jobs every 30 seconds
            except Exception as e:
                logger.error(f"Sync worker error: {e}")
                await asyncio.sleep(60)
    
    async def _process_pending_syncs(self):
        """Process all pending sync jobs"""
        async with await self.get_session() as db:
            # Get pending sync jobs
            result = await db.execute(
                text("""
                    SELECT 
                        id, user_id, device_connection_id, vendor_account_id, status,
                        created_at, attempts
                    FROM device_sync_jobs
                    WHERE status = 'pending'
                    AND (attempts = 0 OR next_retry_at <= NOW())
                    ORDER BY created_at ASC
                    LIMIT :limit
                """),
                {"limit": self.MAX_CONCURRENT_SYNCS * 2}
            )
            
            jobs = result.fetchall()
            
            if not jobs:
                return
            
            # Process jobs concurrently
            tasks = []
            for job in jobs:
                task = asyncio.create_task(
                    self._process_sync_job(SyncJob(
                        id=job.id,
                        user_id=job.user_id,
                        device_id=job.device_connection_id,
                        vendor_id=job.vendor_account_id,
                        status=SyncJobStatus(job.status),
                        created_at=job.created_at,
                        retry_count=job.attempts,
                    ))
                )
                tasks.append(task)
            
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _process_sync_job(self, job: SyncJob):
        """Process a single sync job"""
        async with self._sync_semaphore:
            async with await self.get_session() as db:
                try:
                    # Mark as running
                    await db.execute(
                        text("""
                            UPDATE device_sync_jobs
                            SET status = 'running', started_at = NOW()
                            WHERE id = :job_id
                        """),
                        {"job_id": job.id}
                    )
                    await db.commit()
                    
                    # Apply rate limiting
                    await self._apply_rate_limit(job.vendor_id)
                    
                    # Get device tokens
                    tokens = await self._get_device_tokens(db, job.device_id)
                    if not tokens:
                        raise Exception("No valid tokens found for device")
                    
                    # Create vendor adapter
                    adapter = get_vendor_adapter(
                        job.vendor_id,
                        tokens,
                        job.user_id,
                    )
                    
                    if not adapter:
                        raise Exception(f"No adapter for vendor: {job.vendor_id}")
                    
                    # Sync data
                    result = await adapter.sync_data(
                        start_date=datetime.utcnow() - timedelta(days=1),
                        end_date=datetime.utcnow(),
                    )
                    
                    if not result.success:
                        raise Exception(result.error_message or "Sync failed")
                    
                    # Store readings
                    readings_stored = await self._store_readings(
                        db, job, result.normalized_data or []
                    )
                    
                    # Route to health sections
                    await self._route_to_health_sections(
                        db, job.user_id, result.normalized_data or []
                    )
                    
                    # Update device last sync
                    await db.execute(
                        text("""
                            UPDATE device_connections
                            SET last_sync_at = NOW(), sync_status = 'success'
                            WHERE id = :device_id
                        """),
                        {"device_id": job.device_id}
                    )
                    
                    # Mark job as completed
                    await db.execute(
                        text("""
                            UPDATE device_sync_jobs
                            SET 
                                status = 'completed',
                                completed_at = NOW(),
                                records_fetched = :fetched,
                                records_processed = :processed
                            WHERE id = :job_id
                        """),
                        {
                            "job_id": job.id,
                            "fetched": result.records_fetched,
                            "processed": readings_stored,
                        }
                    )
                    await db.commit()
                    
                    logger.info(
                        f"Sync job {job.id} completed: {readings_stored} readings stored"
                    )
                    
                    # Trigger ML data extraction if consented
                    await self._trigger_ml_extraction(db, job.user_id)
                    
                except Exception as e:
                    logger.error(f"Sync job {job.id} failed: {e}")
                    
                    # Handle retry
                    if job.retry_count < self.MAX_RETRIES:
                        backoff = self.RETRY_BACKOFF_SECONDS[job.retry_count]
                        await db.execute(
                            text("""
                                UPDATE device_sync_jobs
                                SET 
                                    status = 'pending',
                                    retry_count = retry_count + 1,
                                    next_retry_at = NOW() + INTERVAL ':backoff seconds',
                                    error_message = :error
                                WHERE id = :job_id
                            """),
                            {
                                "job_id": job.id,
                                "backoff": backoff,
                                "error": str(e),
                            }
                        )
                    else:
                        await db.execute(
                            text("""
                                UPDATE device_sync_jobs
                                SET 
                                    status = 'failed',
                                    completed_at = NOW(),
                                    error_message = :error
                                WHERE id = :job_id
                            """),
                            {"job_id": job.id, "error": str(e)}
                        )
                        
                        # Update device sync status
                        await db.execute(
                            text("""
                                UPDATE device_connections
                                SET sync_status = 'error'
                                WHERE id = :device_id
                            """),
                            {"device_id": job.device_id}
                        )
                    
                    await db.commit()
    
    async def _apply_rate_limit(self, vendor_id: str):
        """Apply vendor-specific rate limiting"""
        rate_limit = self.VENDOR_RATE_LIMITS.get(vendor_id, 60)
        min_interval = 60.0 / rate_limit
        
        last_request = self._vendor_last_request.get(vendor_id)
        if last_request:
            elapsed = (datetime.utcnow() - last_request).total_seconds()
            if elapsed < min_interval:
                await asyncio.sleep(min_interval - elapsed)
        
        self._vendor_last_request[vendor_id] = datetime.utcnow()
    
    async def _get_device_tokens(
        self, db: AsyncSession, device_id: str
    ) -> Optional[TokenInfo]:
        """Get OAuth tokens for device"""
        result = await db.execute(
            text("""
                SELECT 
                    access_token, refresh_token, token_expires_at,
                    token_type, oauth_scope
                FROM device_connections
                WHERE id = :device_id
            """),
            {"device_id": device_id}
        )
        
        row = result.fetchone()
        if not row or not row.access_token:
            return None
        
        return TokenInfo(
            access_token=row.access_token,
            refresh_token=row.refresh_token,
            expires_at=row.token_expires_at,
            token_type=row.token_type or "Bearer",
            scope=row.oauth_scope,
        )
    
    async def _store_readings(
        self,
        db: AsyncSession,
        job: SyncJob,
        readings: List[Dict[str, Any]],
    ) -> int:
        """Store normalized readings in database"""
        stored_count = 0
        
        for reading in readings:
            try:
                # Validate reading
                if not self._validate_reading(reading):
                    continue
                
                # Insert reading
                await db.execute(
                    text("""
                        INSERT INTO device_readings (
                            user_id, device_id, reading_type, data_type,
                            value, unit, timestamp, source, quality_score, metadata
                        ) VALUES (
                            :user_id, :device_id, :reading_type, :data_type,
                            :value, :unit, :timestamp, :source, :quality, :metadata
                        )
                        ON CONFLICT (user_id, device_id, data_type, timestamp)
                        DO UPDATE SET
                            value = EXCLUDED.value,
                            quality_score = EXCLUDED.quality_score
                    """),
                    {
                        "user_id": job.user_id,
                        "device_id": job.device_id,
                        "reading_type": reading.get("data_type", "unknown"),
                        "data_type": reading.get("data_type"),
                        "value": str(reading.get("value")),
                        "unit": reading.get("unit"),
                        "timestamp": reading.get("timestamp"),
                        "source": reading.get("source", job.vendor_id),
                        "quality": self._calculate_quality_score(reading),
                        "metadata": str(reading.get("metadata", {})),
                    }
                )
                stored_count += 1
                
            except Exception as e:
                logger.warning(f"Failed to store reading: {e}")
        
        return stored_count
    
    def _validate_reading(self, reading: Dict[str, Any]) -> bool:
        """Validate reading data quality"""
        required_fields = ["timestamp", "data_type", "value"]
        
        for field in required_fields:
            if field not in reading or reading[field] is None:
                return False
        
        return True
    
    def _calculate_quality_score(self, reading: Dict[str, Any]) -> float:
        """Calculate data quality score (0-100)"""
        score = 100.0
        
        # Penalize missing metadata
        if not reading.get("metadata"):
            score -= 10
        
        # Penalize missing unit
        if not reading.get("unit"):
            score -= 5
        
        # Penalize old timestamps
        timestamp = reading.get("timestamp")
        if timestamp:
            try:
                dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                age_hours = (datetime.utcnow() - dt.replace(tzinfo=None)).total_seconds() / 3600
                if age_hours > 24:
                    score -= min(20, age_hours / 24 * 5)
            except:
                pass
        
        return max(0, min(100, score))
    
    async def _route_to_health_sections(
        self,
        db: AsyncSession,
        user_id: str,
        readings: List[Dict[str, Any]],
    ):
        """Route readings to appropriate health sections for alerts"""
        # Group readings by health section
        from app.services.vendor_adapters import DATA_TYPE_TO_SECTIONS
        
        section_data: Dict[VendorHealthSection, List[Dict]] = {}
        
        for reading in readings:
            data_type = reading.get("data_type")
            if data_type and data_type in DATA_TYPE_TO_SECTIONS:
                sections = DATA_TYPE_TO_SECTIONS[data_type]
                for section in sections:
                    if section not in section_data:
                        section_data[section] = []
                    section_data[section].append(reading)
        
        # Trigger health section analysis for sections with new data
        for section, data in section_data.items():
            if data:
                try:
                    # Queue health section analysis
                    await db.execute(
                        text("""
                            INSERT INTO health_section_analysis_queue (
                                user_id, section, data_count, queued_at
                            ) VALUES (
                                :user_id, :section, :count, NOW()
                            )
                            ON CONFLICT (user_id, section)
                            DO UPDATE SET
                                data_count = health_section_analysis_queue.data_count + EXCLUDED.data_count,
                                queued_at = NOW()
                        """),
                        {
                            "user_id": user_id,
                            "section": section.value,
                            "count": len(data),
                        }
                    )
                except Exception as e:
                    logger.warning(f"Failed to queue health analysis: {e}")
    
    async def _trigger_ml_extraction(self, db: AsyncSession, user_id: str):
        """Trigger ML training data extraction if user consented"""
        try:
            # Check ML consent
            result = await db.execute(
                text("""
                    SELECT consent_granted
                    FROM ml_training_consent
                    WHERE patient_id = :user_id
                    AND consent_type = 'device_data'
                    AND consent_granted = true
                """),
                {"user_id": user_id}
            )
            
            if result.fetchone():
                # Queue ML extraction job
                await db.execute(
                    text("""
                        INSERT INTO ml_data_extraction_queue (
                            user_id, data_source, queued_at
                        ) VALUES (
                            :user_id, 'device_sync', NOW()
                        )
                        ON CONFLICT (user_id, data_source)
                        DO UPDATE SET queued_at = NOW()
                    """),
                    {"user_id": user_id}
                )
                
        except Exception as e:
            logger.warning(f"Failed to trigger ML extraction: {e}")


# Scheduled sync job creator
async def schedule_device_syncs(db: AsyncSession):
    """
    Create sync jobs for all connected devices that need syncing.
    
    Devices are synced based on their sync_frequency setting:
    - hourly: Every hour
    - daily: Every 24 hours
    - realtime: Every 5 minutes (for CGM devices etc.)
    """
    try:
        # Find devices that need syncing
        result = await db.execute(
            text("""
                SELECT 
                    id, user_id, vendor_id, sync_frequency,
                    last_sync_at
                FROM device_connections
                WHERE connection_status = 'connected'
                AND (
                    (sync_frequency = 'realtime' AND (last_sync_at IS NULL OR last_sync_at < NOW() - INTERVAL '5 minutes'))
                    OR (sync_frequency = 'hourly' AND (last_sync_at IS NULL OR last_sync_at < NOW() - INTERVAL '1 hour'))
                    OR (sync_frequency = 'daily' AND (last_sync_at IS NULL OR last_sync_at < NOW() - INTERVAL '24 hours'))
                    OR (last_sync_at IS NULL OR last_sync_at < NOW() - INTERVAL '6 hours')
                )
                AND NOT EXISTS (
                    SELECT 1 FROM device_sync_jobs
                    WHERE device_connection_id = device_connections.id
                    AND status IN ('pending', 'running')
                )
            """)
        )
        
        devices = result.fetchall()
        
        for device in devices:
            await db.execute(
                text("""
                    INSERT INTO device_sync_jobs (
                        id, user_id, device_connection_id, vendor_account_id, status, created_at
                    ) VALUES (
                        gen_random_uuid()::varchar, :user_id, :device_id, :vendor_id, 'pending', NOW()
                    )
                """),
                {
                    "user_id": device.user_id,
                    "device_id": device.id,
                    "vendor_id": device.vendor_id,
                }
            )
        
        await db.commit()
        
        if devices:
            logger.info(f"Scheduled {len(devices)} device sync jobs")
            
    except Exception as e:
        logger.error(f"Failed to schedule device syncs: {e}")


# Global worker instance (singleton)
_sync_worker: Optional[DeviceSyncWorker] = None


def get_sync_worker() -> Optional[DeviceSyncWorker]:
    """Get the global sync worker instance"""
    return _sync_worker


async def start_sync_worker():
    """Start the device sync worker"""
    global _sync_worker
    
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        logger.error("DATABASE_URL not set, sync worker not starting")
        return
    
    _sync_worker = DeviceSyncWorker(database_url)
    await _sync_worker.start()
    logger.info("✅ Device Sync Worker started")


async def stop_sync_worker():
    """Stop the device sync worker"""
    global _sync_worker
    
    if _sync_worker:
        await _sync_worker.shutdown()
        _sync_worker = None
        logger.info("✅ Device Sync Worker stopped")
