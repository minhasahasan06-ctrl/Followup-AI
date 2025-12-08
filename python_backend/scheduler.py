"""
Background Scheduler for Continuous Research
============================================
Implements APScheduler-based background jobs for:
- Auto-reanalysis of studies with changed data
- Periodic risk scoring and alert generation
- Data quality monitoring
- Comparative analysis result tracking
- Risk & Exposures ETL (infectious events, immunizations, occupational exposures, genetic flags)
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
import asyncio
import json

from risk_exposures_etl import (
    InfectiousEventsETL,
    ImmunizationsETL, 
    OccupationalExposuresETL,
    GeneticRiskFlagsETL
)
from epidemiology.drug_scanner import run_scheduled_drug_scan
from epidemiology.ml_training_integration import run_ml_feature_materialization

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ResearchScheduler:
    """Manages background jobs for continuous research automation."""
    
    def __init__(self, db_url: Optional[str] = None):
        self.scheduler = BackgroundScheduler()
        self.db_url = db_url or os.environ.get("DATABASE_URL")
        self._loop = None
        
    def start(self):
        """Start the background scheduler."""
        if not self.scheduler.running:
            self.scheduler.add_job(
                self._check_auto_reanalysis,
                IntervalTrigger(hours=1),
                id='check_auto_reanalysis',
                replace_existing=True,
                name='Check Auto-Reanalysis'
            )
            
            self.scheduler.add_job(
                self._run_risk_scoring,
                IntervalTrigger(hours=6),
                id='run_risk_scoring',
                replace_existing=True,
                name='Run Risk Scoring'
            )
            
            self.scheduler.add_job(
                self._check_data_quality,
                CronTrigger(hour=3, minute=0),
                id='check_data_quality',
                replace_existing=True,
                name='Check Data Quality'
            )
            
            self.scheduler.add_job(
                self._generate_daily_summary,
                CronTrigger(hour=7, minute=0),
                id='generate_daily_summary',
                replace_existing=True,
                name='Generate Daily Summary'
            )
            
            self.scheduler.add_job(
                self._run_risk_exposures_etl,
                IntervalTrigger(minutes=30),
                id='risk_exposures_etl',
                replace_existing=True,
                name='Risk & Exposures ETL'
            )
            
            self.scheduler.add_job(
                self._run_drug_safety_scan,
                IntervalTrigger(hours=4),
                id='drug_safety_scan',
                replace_existing=True,
                name='Drug Safety Signal Scan'
            )
            
            self.scheduler.add_job(
                self._run_ml_feature_materialization,
                CronTrigger(hour=2, minute=0),
                id='ml_feature_materialization',
                replace_existing=True,
                name='ML Feature Materialization'
            )
            
            self.scheduler.add_job(
                self._run_autopilot_daily_aggregation,
                CronTrigger(hour=3, minute=0),
                id='autopilot_daily_aggregation',
                replace_existing=True,
                name='Autopilot Daily Aggregation'
            )
            
            self.scheduler.add_job(
                self._run_autopilot_inference_sweep,
                IntervalTrigger(hours=1),
                id='autopilot_inference_sweep',
                replace_existing=True,
                name='Autopilot Inference Sweep'
            )
            
            self.scheduler.add_job(
                self._run_autopilot_notification_dispatch,
                IntervalTrigger(minutes=15),
                id='autopilot_notification_dispatch',
                replace_existing=True,
                name='Autopilot Notification Dispatch'
            )
            
            self.scheduler.start()
            logger.info("Research scheduler started with background jobs (including Epidemiology pipelines and Followup Autopilot)")
    
    def stop(self):
        """Stop the background scheduler."""
        if self.scheduler.running:
            self.scheduler.shutdown()
            logger.info("Research scheduler stopped")
    
    def get_jobs(self) -> List[Dict]:
        """Get list of scheduled jobs."""
        jobs = []
        for job in self.scheduler.get_jobs():
            jobs.append({
                "id": job.id,
                "name": job.name,
                "next_run_time": job.next_run_time.isoformat() if job.next_run_time else None,
                "trigger": str(job.trigger)
            })
        return jobs
    
    async def _get_studies_for_reanalysis(self) -> List[Dict]:
        """Get studies that need auto-reanalysis."""
        try:
            import psycopg2
            import psycopg2.extras
            
            conn = psycopg2.connect(self.db_url)
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            now = datetime.utcnow()
            cur.execute("""
                SELECT id, title, reanalysis_frequency, last_reanalysis_at, owner_user_id
                FROM research_studies
                WHERE auto_reanalysis = true
                AND status IN ('active', 'follow_up', 'analysis')
            """)
            
            studies = []
            for row in cur.fetchall():
                study = dict(row)
                last = study.get('last_reanalysis_at')
                freq = study.get('reanalysis_frequency', 'weekly')
                
                needs_reanalysis = False
                if last is None:
                    needs_reanalysis = True
                else:
                    if freq == 'daily':
                        needs_reanalysis = (now - last) > timedelta(days=1)
                    elif freq == 'weekly':
                        needs_reanalysis = (now - last) > timedelta(days=7)
                    elif freq == 'monthly':
                        needs_reanalysis = (now - last) > timedelta(days=30)
                
                if needs_reanalysis:
                    studies.append(study)
            
            cur.close()
            conn.close()
            return studies
            
        except Exception as e:
            logger.error(f"Error getting studies for reanalysis: {e}")
            return []
    
    async def _check_data_changes(self, study_id: str, last_analysis_at: datetime) -> Dict:
        """Check if study data has changed since last analysis."""
        try:
            import psycopg2
            import psycopg2.extras
            
            conn = psycopg2.connect(self.db_url)
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            cur.execute("""
                SELECT 
                    COUNT(*) as new_enrollments,
                    (SELECT COUNT(*) FROM daily_followup_responses dfr
                     JOIN study_enrollments se ON dfr.patient_id = se.patient_id
                     WHERE se.study_id = %s AND dfr.submitted_at > %s) as new_followups,
                    (SELECT COUNT(*) FROM health_alerts ha
                     JOIN study_enrollments se ON ha.patient_id = se.patient_id
                     WHERE se.study_id = %s AND ha.created_at > %s) as new_alerts
                FROM study_enrollments
                WHERE study_id = %s AND enrolled_at > %s
            """, (study_id, last_analysis_at, study_id, last_analysis_at, study_id, last_analysis_at))
            
            result = cur.fetchone()
            cur.close()
            conn.close()
            
            changes = dict(result) if result else {"new_enrollments": 0, "new_followups": 0, "new_alerts": 0}
            changes["has_changes"] = any(v > 0 for v in changes.values() if isinstance(v, int))
            return changes
            
        except Exception as e:
            logger.error(f"Error checking data changes: {e}")
            return {"has_changes": False, "error": str(e)}
    
    async def _create_analysis_job(self, study_id: str, job_type: str, trigger_source: str = "auto_reanalysis") -> Optional[str]:
        """Create a new analysis job in the database."""
        try:
            import psycopg2
            import psycopg2.extras
            import uuid
            
            conn = psycopg2.connect(self.db_url)
            cur = conn.cursor()
            
            job_id = str(uuid.uuid4())
            cur.execute("""
                INSERT INTO research_analysis_jobs 
                (id, study_id, job_type, status, trigger_source, created_at)
                VALUES (%s, %s, %s, 'pending', %s, %s)
                RETURNING id
            """, (job_id, study_id, job_type, trigger_source, datetime.utcnow()))
            
            conn.commit()
            cur.close()
            conn.close()
            
            logger.info(f"Created analysis job {job_id} for study {study_id}")
            return job_id
            
        except Exception as e:
            logger.error(f"Error creating analysis job: {e}")
            return None
    
    async def _update_study_reanalysis_timestamp(self, study_id: str):
        """Update the last_reanalysis_at timestamp for a study."""
        try:
            import psycopg2
            
            conn = psycopg2.connect(self.db_url)
            cur = conn.cursor()
            
            cur.execute("""
                UPDATE research_studies 
                SET last_reanalysis_at = %s
                WHERE id = %s
            """, (datetime.utcnow(), study_id))
            
            conn.commit()
            cur.close()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error updating study reanalysis timestamp: {e}")
    
    async def _compare_analysis_results(self, study_id: str, new_results: Dict, previous_results: Dict) -> Dict:
        """Compare new analysis results with previous results to detect significant changes."""
        changes = {
            "significant": False,
            "details": [],
            "risk_change": None,
            "enrollment_change": None
        }
        
        if not previous_results:
            return changes
        
        prev_risk = previous_results.get("avg_risk_score", 0)
        new_risk = new_results.get("avg_risk_score", 0)
        if prev_risk > 0:
            risk_change = (new_risk - prev_risk) / prev_risk * 100
            changes["risk_change"] = risk_change
            if abs(risk_change) > 20:
                changes["significant"] = True
                changes["details"].append(f"Average risk score changed by {risk_change:.1f}%")
        
        prev_n = previous_results.get("n_subjects", 0)
        new_n = new_results.get("n_subjects", 0)
        if prev_n > 0:
            enrollment_change = (new_n - prev_n) / prev_n * 100
            changes["enrollment_change"] = enrollment_change
            if enrollment_change > 10:
                changes["details"].append(f"Enrollment increased by {enrollment_change:.1f}%")
        
        prev_auc = previous_results.get("model_auroc", 0)
        new_auc = new_results.get("model_auroc", 0)
        if prev_auc > 0 and abs(new_auc - prev_auc) > 0.05:
            changes["significant"] = True
            changes["details"].append(f"Model AUROC changed from {prev_auc:.3f} to {new_auc:.3f}")
        
        return changes
    
    async def _create_research_alert(self, study_id: str, alert_type: str, message: str, severity: str = "medium", details: Optional[Dict] = None):
        """Create a research alert in the database."""
        try:
            import psycopg2
            import uuid
            
            conn = psycopg2.connect(self.db_url)
            cur = conn.cursor()
            
            alert_id = str(uuid.uuid4())
            cur.execute("""
                INSERT INTO research_alerts 
                (id, study_id, alert_type, message, severity, status, details_json, created_at)
                VALUES (%s, %s, %s, %s, %s, 'active', %s, %s)
            """, (alert_id, study_id, alert_type, message, severity, 
                  json.dumps(details or {}), datetime.utcnow()))
            
            conn.commit()
            cur.close()
            conn.close()
            
            logger.info(f"Created research alert {alert_id} for study {study_id}: {message}")
            
        except Exception as e:
            logger.error(f"Error creating research alert: {e}")
    
    def _check_auto_reanalysis(self):
        """Check and trigger auto-reanalysis for eligible studies."""
        logger.info("Running auto-reanalysis check...")
        
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            studies = loop.run_until_complete(self._get_studies_for_reanalysis())
            logger.info(f"Found {len(studies)} studies eligible for reanalysis")
            
            for study in studies:
                study_id = study['id']
                last_analysis = study.get('last_reanalysis_at')
                
                if last_analysis:
                    changes = loop.run_until_complete(
                        self._check_data_changes(study_id, last_analysis)
                    )
                    
                    if changes.get("has_changes"):
                        logger.info(f"Study {study_id} has data changes, triggering reanalysis")
                        loop.run_until_complete(
                            self._create_analysis_job(study_id, "reanalysis", "auto_reanalysis")
                        )
                    else:
                        logger.info(f"Study {study_id} has no data changes, skipping reanalysis")
                else:
                    logger.info(f"Study {study_id} has never been analyzed, triggering initial analysis")
                    loop.run_until_complete(
                        self._create_analysis_job(study_id, "descriptive", "auto_reanalysis")
                    )
                
                loop.run_until_complete(
                    self._update_study_reanalysis_timestamp(study_id)
                )
            
            loop.close()
            
        except Exception as e:
            logger.error(f"Error in auto-reanalysis check: {e}")
    
    def _run_risk_scoring(self):
        """Run periodic risk scoring for all active study participants."""
        logger.info("Running periodic risk scoring...")
        
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            import psycopg2
            import psycopg2.extras
            
            conn = psycopg2.connect(self.db_url)
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            cur.execute("""
                SELECT DISTINCT s.id as study_id, s.title
                FROM research_studies s
                JOIN study_enrollments se ON s.id = se.study_id
                WHERE s.status IN ('active', 'follow_up')
                AND se.status = 'enrolled'
            """)
            
            studies = cur.fetchall()
            cur.close()
            conn.close()
            
            for study in studies:
                loop.run_until_complete(
                    self._create_analysis_job(study['id'], "alert_check", "scheduled")
                )
            
            logger.info(f"Queued risk scoring for {len(studies)} active studies")
            loop.close()
            
        except Exception as e:
            logger.error(f"Error in periodic risk scoring: {e}")
    
    def _check_data_quality(self):
        """Check data quality metrics for all active studies."""
        logger.info("Running data quality check...")
        
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            import psycopg2
            import psycopg2.extras
            
            conn = psycopg2.connect(self.db_url)
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            cur.execute("""
                SELECT s.id, s.title,
                    (SELECT COUNT(*) FROM study_enrollments WHERE study_id = s.id) as enrollment_count,
                    (SELECT COUNT(*) FROM study_enrollments WHERE study_id = s.id AND status = 'withdrawn') as withdrawn_count
                FROM research_studies s
                WHERE s.status IN ('enrolling', 'active', 'follow_up')
            """)
            
            studies = cur.fetchall()
            
            for study in studies:
                if study['enrollment_count'] > 0:
                    withdrawal_rate = study['withdrawn_count'] / study['enrollment_count'] * 100
                    if withdrawal_rate > 20:
                        loop.run_until_complete(
                            self._create_research_alert(
                                study['id'],
                                "data_quality",
                                f"High withdrawal rate ({withdrawal_rate:.1f}%) detected",
                                "high",
                                {"withdrawal_rate": withdrawal_rate}
                            )
                        )
            
            cur.close()
            conn.close()
            loop.close()
            
            logger.info("Data quality check completed")
            
        except Exception as e:
            logger.error(f"Error in data quality check: {e}")
    
    def _run_risk_exposures_etl(self):
        """Run Risk & Exposures ETL jobs to populate derived risk data."""
        logger.info("Running Risk & Exposures ETL jobs...")
        
        try:
            infectious_etl = InfectiousEventsETL()
            infectious_stats = infectious_etl.run()
            logger.info(f"Infectious events ETL: {infectious_stats}")
        except Exception as e:
            logger.error(f"Error in infectious events ETL: {e}")
        
        try:
            immunizations_etl = ImmunizationsETL()
            immunizations_stats = immunizations_etl.run()
            logger.info(f"Immunizations ETL: {immunizations_stats}")
        except Exception as e:
            logger.error(f"Error in immunizations ETL: {e}")
        
        try:
            occupational_etl = OccupationalExposuresETL()
            occupational_stats = occupational_etl.run()
            logger.info(f"Occupational exposures ETL: {occupational_stats}")
        except Exception as e:
            logger.error(f"Error in occupational exposures ETL: {e}")
        
        try:
            genetic_etl = GeneticRiskFlagsETL()
            genetic_stats = genetic_etl.run()
            logger.info(f"Genetic risk flags ETL: {genetic_stats}")
        except Exception as e:
            logger.error(f"Error in genetic risk flags ETL: {e}")
        
        logger.info("Risk & Exposures ETL jobs completed")

    def _run_drug_safety_scan(self):
        """Run drug safety signal scanning job."""
        logger.info("Running drug safety signal scan...")
        
        try:
            result = run_scheduled_drug_scan()
            logger.info(f"Drug safety scan completed: {result}")
        except Exception as e:
            logger.error(f"Error in drug safety scan: {e}")
    
    def _run_ml_feature_materialization(self):
        """Run ML feature materialization job."""
        logger.info("Running ML feature materialization...")
        
        try:
            result = run_ml_feature_materialization()
            logger.info(f"ML feature materialization completed: {result}")
        except Exception as e:
            logger.error(f"Error in ML feature materialization: {e}")

    def _generate_daily_summary(self):
        """Generate daily summary of research activities."""
        logger.info("Generating daily research summary...")
        
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            import psycopg2
            import psycopg2.extras
            
            conn = psycopg2.connect(self.db_url)
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            yesterday = datetime.utcnow() - timedelta(days=1)
            
            cur.execute("""
                SELECT 
                    (SELECT COUNT(*) FROM study_enrollments WHERE enrolled_at > %s) as new_enrollments,
                    (SELECT COUNT(*) FROM research_analysis_jobs WHERE created_at > %s) as analyses_run,
                    (SELECT COUNT(*) FROM research_alerts WHERE created_at > %s) as new_alerts,
                    (SELECT COUNT(*) FROM daily_followup_responses WHERE submitted_at > %s) as followup_responses
            """, (yesterday, yesterday, yesterday, yesterday))
            
            summary = cur.fetchone()
            
            logger.info(f"Daily summary: {dict(summary) if summary else 'No data'}")
            
            cur.close()
            conn.close()
            loop.close()
            
        except Exception as e:
            logger.error(f"Error generating daily summary: {e}")


    def _run_autopilot_daily_aggregation(self):
        """Run daily feature aggregation for all patients with signals."""
        logger.info("Running Autopilot daily aggregation...")
        
        conn = None
        cur = None
        try:
            import psycopg2
            import psycopg2.extras
            from datetime import date
            
            conn = psycopg2.connect(self.db_url)
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            yesterday = date.today() - timedelta(days=1)
            
            cur.execute("""
                SELECT DISTINCT patient_id 
                FROM autopilot_patient_signals 
                WHERE DATE(signal_time) = %s
            """, (yesterday,))
            
            patients = [row['patient_id'] for row in cur.fetchall()]
            
            logger.info(f"Aggregating features for {len(patients)} patients")
            
            for patient_id in patients:
                try:
                    from ml_analysis.followup_autopilot.feature_builder import FeatureBuilder
                    builder = FeatureBuilder()
                    features = builder.build_daily_features(patient_id, yesterday)
                    builder.save_daily_features(patient_id, yesterday, features)
                except Exception as e:
                    logger.error(f"Error aggregating features for {patient_id}: {e}")
            
            logger.info(f"Autopilot daily aggregation completed for {len(patients)} patients")
            
        except Exception as e:
            logger.error(f"Error in Autopilot daily aggregation: {e}")
        finally:
            if cur:
                cur.close()
            if conn:
                conn.close()
    
    def _run_autopilot_inference_sweep(self):
        """Run inference sweep for patients due for follow-up."""
        logger.info("Running Autopilot inference sweep...")
        
        conn = None
        cur = None
        try:
            import psycopg2
            import psycopg2.extras
            
            conn = psycopg2.connect(self.db_url)
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            now = datetime.utcnow()
            
            cur.execute("""
                SELECT patient_id 
                FROM autopilot_patient_states 
                WHERE next_followup_at <= %s OR next_followup_at IS NULL
                LIMIT 100
            """, (now,))
            
            patients = [row['patient_id'] for row in cur.fetchall()]
            
            logger.info(f"Running inference for {len(patients)} patients due for follow-up")
            
            for patient_id in patients:
                try:
                    from ml_analysis.followup_autopilot.autopilot_core import AutopilotCore
                    from ml_analysis.followup_autopilot.trigger_engine import TriggerEngine
                    from ml_analysis.followup_autopilot.feature_builder import FeatureBuilder
                    from datetime import date
                    
                    autopilot = AutopilotCore()
                    state = autopilot.update_patient_state(patient_id)
                    
                    builder = FeatureBuilder()
                    today = date.today()
                    features_today = builder.build_daily_features(patient_id, today)
                    
                    trigger_engine = TriggerEngine()
                    trigger_engine.run_triggers(
                        patient_id=patient_id,
                        features_today=features_today,
                        patient_state=state,
                        risk_probs={
                            "p_clinical_deterioration": state.get("risk_components", {}).get("clinical", 0) / 100,
                            "p_mental_health_crisis": state.get("risk_components", {}).get("mental_health", 0) / 100,
                            "p_non_adherence": state.get("risk_components", {}).get("adherence", 0) / 100,
                        },
                        anomaly_score=state.get("anomaly_score", 0)
                    )
                except Exception as e:
                    logger.error(f"Error in inference for {patient_id}: {e}")
            
            logger.info(f"Autopilot inference sweep completed for {len(patients)} patients")
            
        except Exception as e:
            logger.error(f"Error in Autopilot inference sweep: {e}")
        finally:
            if cur:
                cur.close()
            if conn:
                conn.close()
    
    def _run_autopilot_notification_dispatch(self):
        """Dispatch pending autopilot notifications."""
        logger.info("Running Autopilot notification dispatch...")
        
        try:
            from ml_analysis.followup_autopilot.notification_engine import NotificationEngine
            
            notifier = NotificationEngine()
            results = notifier.dispatch_pending_notifications()
            
            total_sent = sum(v for k, v in results.items() if k != 'failed')
            logger.info(f"Autopilot notification dispatch: {total_sent} sent, {results.get('failed', 0)} failed")
            
        except Exception as e:
            logger.error(f"Error in Autopilot notification dispatch: {e}")


_scheduler_instance: Optional[ResearchScheduler] = None


def get_scheduler() -> ResearchScheduler:
    """Get or create the global scheduler instance."""
    global _scheduler_instance
    if _scheduler_instance is None:
        _scheduler_instance = ResearchScheduler()
    return _scheduler_instance


def start_scheduler():
    """Start the global scheduler."""
    scheduler = get_scheduler()
    scheduler.start()


def stop_scheduler():
    """Stop the global scheduler."""
    global _scheduler_instance
    if _scheduler_instance:
        _scheduler_instance.stop()
        _scheduler_instance = None
