"""
Alert Engine Background Worker - Orchestrates the complete alert pipeline.

This worker:
1. Consumes metrics from Redis stream (alert_engine:metrics_ingest)
2. Aggregates metrics by patient
3. Runs Organ Scoring → DPI Computation → Rule Engine → Notifications
4. Handles escalation checks periodically

Can be run as:
- FastAPI background task
- Standalone worker process
- Scheduled cron job
"""

import os
import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)

# Optional Redis import
try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    try:
        import redis
        REDIS_AVAILABLE = True
    except ImportError:
        REDIS_AVAILABLE = False
        logger.warning("Redis not available for background worker")


class AlertEngineWorker:
    """Background worker that orchestrates the Alert Engine pipeline"""
    
    def __init__(self, db_session_factory):
        """
        Initialize the worker.
        
        Args:
            db_session_factory: Callable that returns a database session
        """
        self.db_session_factory = db_session_factory
        self.redis_client = None
        self.stream_name = "alert_engine:metrics_ingest"
        self.consumer_group = "alert_engine_workers"
        self.consumer_name = f"worker_{os.getpid()}"
        self.running = False
        self.metrics_buffer: Dict[str, List[Dict]] = defaultdict(list)
        self.buffer_flush_interval = 30  # seconds
        self.escalation_check_interval = 300  # 5 minutes
        self.last_escalation_check = datetime.utcnow()
    
    async def initialize(self):
        """Initialize Redis connection and consumer group"""
        if not REDIS_AVAILABLE:
            logger.warning("Redis not available - worker will use direct processing mode")
            return
        
        try:
            redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
            self.redis_client = redis.from_url(redis_url, decode_responses=True)
            
            # Create consumer group if it doesn't exist
            try:
                await self.redis_client.xgroup_create(
                    self.stream_name,
                    self.consumer_group,
                    id='0',
                    mkstream=True
                )
                logger.info(f"Created consumer group: {self.consumer_group}")
            except redis.ResponseError as e:
                if "BUSYGROUP" not in str(e):
                    raise
                logger.info(f"Consumer group {self.consumer_group} already exists")
            
            logger.info("Alert Engine Worker initialized with Redis")
            
        except Exception as e:
            logger.warning(f"Redis initialization failed: {e} - using direct processing mode")
            self.redis_client = None
    
    async def start(self):
        """Start the background worker"""
        self.running = True
        logger.info("Alert Engine Worker starting...")
        
        # Run tasks concurrently
        await asyncio.gather(
            self._stream_consumer_loop(),
            self._buffer_flush_loop(),
            self._escalation_check_loop()
        )
    
    async def stop(self):
        """Stop the background worker gracefully"""
        self.running = False
        logger.info("Alert Engine Worker stopping...")
        
        # Flush any remaining buffered metrics
        await self._flush_all_buffers()
        
        if self.redis_client:
            await self.redis_client.close()
    
    async def _stream_consumer_loop(self):
        """Main loop that consumes metrics from Redis stream"""
        if not self.redis_client:
            logger.info("No Redis - skipping stream consumer loop")
            return
        
        while self.running:
            try:
                # Read from stream with blocking
                messages = await self.redis_client.xreadgroup(
                    groupname=self.consumer_group,
                    consumername=self.consumer_name,
                    streams={self.stream_name: '>'},
                    count=100,
                    block=1000  # 1 second block
                )
                
                if messages:
                    for stream_name, stream_messages in messages:
                        for msg_id, msg_data in stream_messages:
                            await self._process_stream_message(msg_id, msg_data)
                            
                            # Acknowledge message
                            await self.redis_client.xack(
                                self.stream_name,
                                self.consumer_group,
                                msg_id
                            )
                
            except Exception as e:
                logger.error(f"Error in stream consumer loop: {e}")
                await asyncio.sleep(5)  # Back off on error
    
    async def _process_stream_message(self, msg_id: str, msg_data: Dict):
        """Process a single message from the stream"""
        try:
            patient_id = msg_data.get('patient_id')
            metric_name = msg_data.get('metric_name')
            metric_value = float(msg_data.get('metric_value', 0))
            timestamp = msg_data.get('timestamp')
            
            if not patient_id:
                logger.warning(f"Message {msg_id} missing patient_id")
                return
            
            # Add to patient's buffer
            self.metrics_buffer[patient_id].append({
                'metric_name': metric_name,
                'metric_value': metric_value,
                'timestamp': timestamp,
                'msg_id': msg_id
            })
            
            logger.debug(f"Buffered metric for patient {patient_id}: {metric_name}")
            
        except Exception as e:
            logger.error(f"Error processing message {msg_id}: {e}")
    
    async def _buffer_flush_loop(self):
        """Periodically flush metric buffers and run pipeline"""
        while self.running:
            await asyncio.sleep(self.buffer_flush_interval)
            await self._flush_all_buffers()
    
    async def _flush_all_buffers(self):
        """Flush all patient buffers and run alert pipeline"""
        if not self.metrics_buffer:
            return
        
        # Copy and clear buffers
        buffers_to_process = dict(self.metrics_buffer)
        self.metrics_buffer = defaultdict(list)
        
        # Process each patient's metrics
        for patient_id, metrics in buffers_to_process.items():
            if metrics:
                await self._run_alert_pipeline(patient_id, metrics)
    
    async def _run_alert_pipeline(
        self,
        patient_id: str,
        metrics: List[Dict]
    ):
        """
        Run the complete alert pipeline for a patient.
        
        Pipeline:
        1. Compute organ scores from recent metrics
        2. Compute DPI from organ scores
        3. Evaluate alert rules
        4. Create alerts and send notifications
        """
        db = self.db_session_factory()
        
        try:
            logger.info(f"Running alert pipeline for patient {patient_id} with {len(metrics)} metrics")
            
            # Import services
            from .organ_scoring import OrganScoringService
            from .dpi_computation import DPIComputationService
            from .rule_engine import RuleBasedAlertEngine
            from .notification_service import NotificationService, NotificationRequest, NotificationChannel
            from .ml_ranking import MLRankingService
            
            # Step 1: Compute organ scores
            organ_service = OrganScoringService(db)
            organ_result = await organ_service.compute_from_recent_data(patient_id)
            await organ_service.store_organ_scores(organ_result)
            
            logger.debug(f"Patient {patient_id} organ scores computed")
            
            # Step 2: Compute DPI
            dpi_service = DPIComputationService(db)
            dpi_result = dpi_service.compute_dpi(patient_id, organ_result)
            await dpi_service.store_dpi(dpi_result)
            
            logger.debug(f"Patient {patient_id} DPI: {dpi_result.dpi_normalized} ({dpi_result.dpi_bucket})")
            
            # Step 3: Get metric z-scores for rule evaluation
            metric_z_scores = {}
            for group_name, organ_score in organ_result.organ_scores.items():
                for metric in organ_score.contributing_metrics:
                    metric_z_scores[metric.metric_name] = metric.z_score
            
            # Step 4: Evaluate alert rules
            rule_engine = RuleBasedAlertEngine(db)
            triggered_alerts = await rule_engine.evaluate_all_rules(
                patient_id=patient_id,
                dpi_result=dpi_result,
                organ_result=organ_result,
                metric_z_scores=metric_z_scores
            )
            
            logger.info(f"Patient {patient_id}: {len(triggered_alerts)} alerts triggered")
            
            # Step 5: Create alert records and send notifications
            notification_service = NotificationService(db)
            
            for trigger in triggered_alerts:
                # Create alert record
                alert_record = await rule_engine.create_alert_record(patient_id, trigger)
                
                # Find assigned clinician
                clinician_id = await self._get_assigned_clinician(db, patient_id)
                
                if clinician_id:
                    clinician_info = await self._get_clinician_info(db, clinician_id)
                    
                    # Create notification request
                    request = NotificationRequest(
                        alert_id=alert_record.id,
                        patient_id=patient_id,
                        recipient_id=clinician_id,
                        recipient_email=clinician_info.get('email'),
                        recipient_phone=clinician_info.get('phone'),
                        priority=alert_record.priority,
                        is_escalation=False
                    )
                    
                    # Send notifications
                    results = await notification_service.send_alert_notification(
                        alert_record, request
                    )
                    
                    for result in results:
                        if result.success:
                            logger.info(f"Notification sent via {result.channel.value}")
                        else:
                            logger.warning(f"Notification failed via {result.channel.value}: {result.error_message}")
            
            # Step 6: Apply ML ranking to existing alerts (for dashboard ordering)
            ml_service = MLRankingService(db)
            await ml_service.initialize()
            
            # This will update priority scores in the database for dashboard display
            await self._update_alert_ml_scores(db, patient_id, ml_service)
            
            db.commit()
            logger.info(f"Alert pipeline completed for patient {patient_id}")
            
        except Exception as e:
            logger.error(f"Error in alert pipeline for patient {patient_id}: {e}")
            db.rollback()
        finally:
            db.close()
    
    async def _escalation_check_loop(self):
        """Periodically check for alerts that need escalation"""
        while self.running:
            await asyncio.sleep(self.escalation_check_interval)
            
            if datetime.utcnow() - self.last_escalation_check >= timedelta(seconds=self.escalation_check_interval):
                await self._run_escalation_check()
                self.last_escalation_check = datetime.utcnow()
    
    async def _run_escalation_check(self):
        """Check and escalate overdue alerts"""
        db = self.db_session_factory()
        
        try:
            from .escalation_service import EscalationService
            
            escalation_service = EscalationService(db)
            escalated = await escalation_service.check_and_escalate_alerts()
            
            if escalated:
                logger.info(f"Escalated {len(escalated)} alerts")
            
            db.commit()
            
        except Exception as e:
            logger.error(f"Error in escalation check: {e}")
            db.rollback()
        finally:
            db.close()
    
    async def _get_assigned_clinician(self, db, patient_id: str) -> Optional[str]:
        """Get the assigned clinician for a patient"""
        from sqlalchemy import text
        
        try:
            query = text("""
                SELECT assigned_doctor_id FROM users
                WHERE id = :patient_id
            """)
            result = db.execute(query, {"patient_id": patient_id}).fetchone()
            return result[0] if result else None
        except Exception:
            return None
    
    async def _get_clinician_info(self, db, clinician_id: str) -> Dict[str, Any]:
        """Get clinician contact information"""
        from sqlalchemy import text
        
        try:
            query = text("""
                SELECT email, phone_number, first_name, last_name
                FROM users WHERE id = :clinician_id
            """)
            result = db.execute(query, {"clinician_id": clinician_id}).fetchone()
            
            if result:
                return {
                    "email": result[0],
                    "phone": result[1],
                    "name": f"{result[2]} {result[3]}"
                }
        except Exception:
            pass
        
        return {}
    
    async def _update_alert_ml_scores(self, db, patient_id: str, ml_service):
        """Update ML priority scores for a patient's alerts"""
        from sqlalchemy import text
        
        try:
            # Get active alerts
            query = text("""
                SELECT id, alert_type, severity, priority, trigger_rule,
                       dpi_at_trigger, organ_scores, corroborated, contributing_metrics
                FROM ai_health_alerts
                WHERE patient_id = :patient_id
                AND status NOT IN ('dismissed', 'closed')
            """)
            
            results = db.execute(query, {"patient_id": patient_id}).fetchall()
            
            if not results:
                return
            
            alerts = [
                {
                    "id": str(row[0]),
                    "patient_id": patient_id,
                    "alert_type": row[1],
                    "severity": row[2],
                    "priority": row[3],
                    "trigger_rule": row[4],
                    "dpi_at_trigger": float(row[5]) if row[5] else None,
                    "organ_scores": row[6],
                    "corroborated": row[7] or False,
                    "trigger_metrics": row[8] or []
                }
                for row in results
            ]
            
            # Get ML rankings
            ranked = await ml_service.rank_alerts(alerts)
            
            # Update scores in database
            for result in ranked:
                update_query = text("""
                    UPDATE ai_health_alerts
                    SET ml_priority_score = :score
                    WHERE id = :alert_id
                """)
                db.execute(update_query, {
                    "score": result.priority_score,
                    "alert_id": result.alert_id
                })
                
        except Exception as e:
            logger.warning(f"Error updating ML scores: {e}")
    
    async def process_metric_directly(
        self,
        patient_id: str,
        metric_name: str,
        metric_value: float,
        timestamp: Optional[datetime] = None
    ):
        """
        Process a metric directly without Redis.
        Used when Redis is unavailable or for immediate processing.
        """
        self.metrics_buffer[patient_id].append({
            'metric_name': metric_name,
            'metric_value': metric_value,
            'timestamp': timestamp or datetime.utcnow().isoformat()
        })
        
        # If buffer is getting large, flush immediately
        if len(self.metrics_buffer[patient_id]) >= 10:
            await self._run_alert_pipeline(
                patient_id,
                self.metrics_buffer.pop(patient_id, [])
            )


class AlertEngineCronJob:
    """
    Cron-style job that runs the alert pipeline for all patients periodically.
    Alternative to stream-based processing.
    """
    
    def __init__(self, db_session_factory, interval_minutes: int = 15):
        self.db_session_factory = db_session_factory
        self.interval_minutes = interval_minutes
        self.running = False
    
    async def start(self):
        """Start the cron job"""
        self.running = True
        logger.info(f"Alert Engine Cron starting (interval: {self.interval_minutes} min)")
        
        while self.running:
            try:
                await self._run_for_all_patients()
            except Exception as e:
                logger.error(f"Error in cron job: {e}")
            
            await asyncio.sleep(self.interval_minutes * 60)
    
    async def stop(self):
        """Stop the cron job"""
        self.running = False
    
    async def _run_for_all_patients(self):
        """Run alert pipeline for all active patients"""
        db = self.db_session_factory()
        
        try:
            from sqlalchemy import text
            
            # Get patients with recent activity
            query = text("""
                SELECT DISTINCT patient_id FROM metric_ingest_log
                WHERE created_at >= NOW() - INTERVAL '1 hour'
                UNION
                SELECT DISTINCT user_id FROM symptom_checkins
                WHERE created_at >= NOW() - INTERVAL '1 hour'
            """)
            
            results = db.execute(query).fetchall()
            patient_ids = [row[0] for row in results if row[0]]
            
            logger.info(f"Running alert pipeline for {len(patient_ids)} patients")
            
            worker = AlertEngineWorker(self.db_session_factory)
            
            for patient_id in patient_ids:
                try:
                    await worker._run_alert_pipeline(patient_id, [])
                except Exception as e:
                    logger.error(f"Error processing patient {patient_id}: {e}")
            
        finally:
            db.close()


def start_worker_in_thread(db_session_factory):
    """
    Start the Alert Engine Worker in a background thread.
    Useful for integration with FastAPI startup events.
    """
    import threading
    
    def run_worker():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        worker = AlertEngineWorker(db_session_factory)
        loop.run_until_complete(worker.initialize())
        loop.run_until_complete(worker.start())
    
    thread = threading.Thread(target=run_worker, daemon=True)
    thread.start()
    logger.info("Alert Engine Worker thread started")
    return thread
