"""
Batch Processing Service for ML Predictions
Handles bulk prediction jobs with progress tracking
"""

import asyncio
from typing import List, Dict, Any
from datetime import datetime
from sqlalchemy.orm import Session
import logging

from app.models.ml_models import MLBatchJob, MLModel
from app.services.ml_inference import ml_registry

logger = logging.getLogger(__name__)


class BatchProcessor:
    """Process ML predictions in batches"""
    
    def __init__(self, db: Session):
        self.db = db
        self.running_jobs: Dict[int, asyncio.Task] = {}
    
    async def create_batch_job(
        self,
        model_name: str,
        job_name: str,
        job_type: str,
        items: List[Dict[str, Any]],
        created_by: str
    ) -> MLBatchJob:
        """
        Create and start a new batch prediction job
        
        Args:
            model_name: Name of ML model to use
            job_name: Human-readable job name
            job_type: Type of job (e.g., "bulk_prediction", "model_evaluation")
            items: List of items to process
            created_by: User ID who created the job
        
        Returns:
            Created batch job record
        """
        # Get model
        model = self.db.query(MLModel).filter(
            MLModel.name == model_name,
            MLModel.is_active == True
        ).first()
        
        if not model:
            raise ValueError(f"Model {model_name} not found")
        
        # Create job record
        job = MLBatchJob(
            model_id=model.id,
            job_name=job_name,
            job_type=job_type,
            status="pending",
            total_items=len(items),
            processed_items=0,
            failed_items=0,
            created_by=created_by
        )
        
        self.db.add(job)
        self.db.commit()
        self.db.refresh(job)
        
        # Start processing in background
        task = asyncio.create_task(
            self._process_batch(job.id, model_name, items)
        )
        self.running_jobs[job.id] = task
        
        logger.info(f"Created batch job {job.id}: {job_name}")
        return job
    
    async def _process_batch(
        self,
        job_id: int,
        model_name: str,
        items: List[Dict[str, Any]]
    ):
        """
        Process batch job in background
        Updates job status and progress as it runs
        """
        job = self.db.query(MLBatchJob).filter(MLBatchJob.id == job_id).first()
        
        try:
            # Update status to running
            job.status = "running"
            job.started_at = datetime.utcnow()
            self.db.commit()
            
            results = []
            processed = 0
            failed = 0
            
            # Process items in batches of 10
            batch_size = 10
            for i in range(0, len(items), batch_size):
                batch = items[i:i + batch_size]
                
                # Process batch concurrently
                batch_tasks = [
                    ml_registry.predict(
                        model_name=model_name,
                        input_data=item,
                        use_cache=True,
                        db=self.db,
                        patient_id=item.get("patient_id")
                    )
                    for item in batch
                ]
                
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                
                # Count successes and failures
                for result in batch_results:
                    if isinstance(result, Exception):
                        failed += 1
                    else:
                        processed += 1
                        results.append(result)
                
                # Update progress
                job.processed_items = processed
                job.failed_items = failed
                self.db.commit()
                
                logger.info(f"Batch job {job_id}: {processed}/{len(items)} processed")
            
            # Calculate summary statistics
            summary = {
                "total_processed": processed,
                "total_failed": failed,
                "success_rate": (processed / len(items) * 100) if len(items) > 0 else 0,
                "predictions": results[:100]  # Store first 100 results
            }
            
            # Update job as completed
            job.status = "completed"
            job.completed_at = datetime.utcnow()
            job.results_summary = summary
            self.db.commit()
            
            logger.info(f"✅ Batch job {job_id} completed: {processed} processed, {failed} failed")
        
        except Exception as e:
            logger.error(f"❌ Batch job {job_id} failed: {e}")
            job.status = "failed"
            job.error_log = str(e)
            job.completed_at = datetime.utcnow()
            self.db.commit()
        
        finally:
            # Remove from running jobs
            if job_id in self.running_jobs:
                del self.running_jobs[job_id]
    
    def get_job_status(self, job_id: int) -> Dict[str, Any]:
        """Get current status of a batch job"""
        job = self.db.query(MLBatchJob).filter(MLBatchJob.id == job_id).first()
        
        if not job:
            raise ValueError(f"Job {job_id} not found")
        
        progress_percent = 0
        if job.total_items and job.total_items > 0:
            progress_percent = (job.processed_items / job.total_items) * 100
        
        return {
            "job_id": job.id,
            "job_name": job.job_name,
            "status": job.status,
            "progress_percent": round(progress_percent, 2),
            "total_items": job.total_items,
            "processed_items": job.processed_items,
            "failed_items": job.failed_items,
            "started_at": job.started_at.isoformat() if job.started_at else None,
            "completed_at": job.completed_at.isoformat() if job.completed_at else None,
            "results_summary": job.results_summary,
            "error_log": job.error_log
        }
    
    async def cancel_job(self, job_id: int):
        """Cancel a running batch job"""
        if job_id in self.running_jobs:
            task = self.running_jobs[job_id]
            task.cancel()
            
            # Update status
            job = self.db.query(MLBatchJob).filter(MLBatchJob.id == job_id).first()
            if job:
                job.status = "cancelled"
                job.completed_at = datetime.utcnow()
                self.db.commit()
            
            logger.info(f"Cancelled batch job {job_id}")


# Global batch processor instance
_batch_processor: BatchProcessor = None


def get_batch_processor(db: Session) -> BatchProcessor:
    """Get or create batch processor instance"""
    global _batch_processor
    if _batch_processor is None:
        _batch_processor = BatchProcessor(db)
    return _batch_processor
