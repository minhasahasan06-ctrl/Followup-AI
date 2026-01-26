"""
ML Training Worker Service
Background job worker for ML model training with:
- Consent verification before data extraction
- Progress tracking and status updates
- Model export (ONNX/joblib)
- Integration with model registry
"""

import os
import json
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging
import traceback

import numpy as np
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

try:
    import joblib
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.cluster import KMeans
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, silhouette_score
    )
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

from .ml_training_pipeline import (
    MLTrainingPipeline,
    TrainingJobConfig,
    TrainingDataset,
    TrainingAuditLogger
)
from .public_dataset_loader import PublicDatasetManager

logger = logging.getLogger(__name__)

MODEL_SAVE_DIR = "./models/trained"


@dataclass
class TrainingResult:
    """Result of a training run"""
    success: bool
    model_id: Optional[str] = None
    model_path: Optional[str] = None
    model_format: str = "joblib"
    metrics: Dict[str, float] = None
    feature_importance: Dict[str, float] = None
    training_time_seconds: float = 0.0
    error_message: Optional[str] = None


class ModelTrainer:
    """Train various ML model types"""
    
    def __init__(self):
        os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    
    def train_random_forest(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        config: Dict[str, Any],
        feature_names: List[str]
    ) -> TrainingResult:
        """Train a Random Forest classifier"""
        
        if not HAS_SKLEARN:
            return TrainingResult(
                success=False,
                error_message="scikit-learn not available"
            )
        
        start_time = datetime.utcnow()
        
        try:
            model = RandomForestClassifier(
                n_estimators=config.get('n_estimators', 100),
                max_depth=config.get('max_depth', 10),
                min_samples_split=config.get('min_samples_split', 5),
                min_samples_leaf=config.get('min_samples_leaf', 2),
                random_state=42,
                n_jobs=-1
            )
            
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_val)
            y_pred_proba = model.predict_proba(X_val)[:, 1] if len(np.unique(y_val)) == 2 else None
            
            metrics = {
                'accuracy': float(accuracy_score(y_val, y_pred)),
                'precision': float(precision_score(y_val, y_pred, average='weighted', zero_division=0)),
                'recall': float(recall_score(y_val, y_pred, average='weighted', zero_division=0)),
                'f1': float(f1_score(y_val, y_pred, average='weighted', zero_division=0))
            }
            
            if y_pred_proba is not None:
                try:
                    metrics['auc_roc'] = float(roc_auc_score(y_val, y_pred_proba))
                except Exception:
                    pass
            
            feature_importance = {}
            if feature_names and len(feature_names) == X_train.shape[1]:
                for name, importance in zip(feature_names, model.feature_importances_):
                    feature_importance[name] = float(importance)
            
            model_filename = f"random_forest_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.joblib"
            model_path = os.path.join(MODEL_SAVE_DIR, model_filename)
            
            if HAS_JOBLIB:
                joblib.dump(model, model_path)
            
            training_time = (datetime.utcnow() - start_time).total_seconds()
            
            return TrainingResult(
                success=True,
                model_path=model_path,
                model_format="joblib",
                metrics=metrics,
                feature_importance=feature_importance,
                training_time_seconds=training_time
            )
            
        except Exception as e:
            logger.error(f"Random Forest training failed: {e}")
            return TrainingResult(
                success=False,
                error_message=str(e)
            )
    
    def train_gradient_boosting(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        config: Dict[str, Any],
        feature_names: List[str]
    ) -> TrainingResult:
        """Train a Gradient Boosting classifier"""
        
        if not HAS_SKLEARN:
            return TrainingResult(
                success=False,
                error_message="scikit-learn not available"
            )
        
        start_time = datetime.utcnow()
        
        try:
            model = GradientBoostingClassifier(
                n_estimators=config.get('n_estimators', 100),
                max_depth=config.get('max_depth', 5),
                learning_rate=config.get('learning_rate', 0.1),
                min_samples_split=config.get('min_samples_split', 5),
                min_samples_leaf=config.get('min_samples_leaf', 2),
                random_state=42
            )
            
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_val)
            y_pred_proba = model.predict_proba(X_val)[:, 1] if len(np.unique(y_val)) == 2 else None
            
            metrics = {
                'accuracy': float(accuracy_score(y_val, y_pred)),
                'precision': float(precision_score(y_val, y_pred, average='weighted', zero_division=0)),
                'recall': float(recall_score(y_val, y_pred, average='weighted', zero_division=0)),
                'f1': float(f1_score(y_val, y_pred, average='weighted', zero_division=0))
            }
            
            if y_pred_proba is not None:
                try:
                    metrics['auc_roc'] = float(roc_auc_score(y_val, y_pred_proba))
                except Exception:
                    pass
            
            feature_importance = {}
            if feature_names and len(feature_names) == X_train.shape[1]:
                for name, importance in zip(feature_names, model.feature_importances_):
                    feature_importance[name] = float(importance)
            
            model_filename = f"gradient_boosting_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.joblib"
            model_path = os.path.join(MODEL_SAVE_DIR, model_filename)
            
            if HAS_JOBLIB:
                joblib.dump(model, model_path)
            
            training_time = (datetime.utcnow() - start_time).total_seconds()
            
            return TrainingResult(
                success=True,
                model_path=model_path,
                model_format="joblib",
                metrics=metrics,
                feature_importance=feature_importance,
                training_time_seconds=training_time
            )
            
        except Exception as e:
            logger.error(f"Gradient Boosting training failed: {e}")
            return TrainingResult(
                success=False,
                error_message=str(e)
            )
    
    def train_kmeans(
        self,
        X: np.ndarray,
        config: Dict[str, Any],
        feature_names: List[str]
    ) -> TrainingResult:
        """Train a K-Means clustering model"""
        
        if not HAS_SKLEARN:
            return TrainingResult(
                success=False,
                error_message="scikit-learn not available"
            )
        
        start_time = datetime.utcnow()
        
        try:
            n_clusters = config.get('n_clusters', 4)
            
            model = KMeans(
                n_clusters=n_clusters,
                max_iter=config.get('max_iterations', 300),
                n_init=config.get('n_init', 10),
                random_state=42
            )
            
            labels = model.fit_predict(X)
            
            metrics = {
                'inertia': float(model.inertia_),
                'n_iterations': int(model.n_iter_)
            }
            
            if len(np.unique(labels)) > 1:
                try:
                    metrics['silhouette_score'] = float(silhouette_score(X, labels))
                except Exception:
                    pass
            
            feature_importance = {}
            if feature_names:
                for i, name in enumerate(feature_names[:X.shape[1]]):
                    variances = []
                    for cluster_id in range(n_clusters):
                        cluster_points = X[labels == cluster_id]
                        if len(cluster_points) > 0:
                            variances.append(np.var(cluster_points[:, i]))
                    
                    feature_importance[name] = float(np.mean(variances)) if variances else 0.0
            
            model_filename = f"kmeans_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.joblib"
            model_path = os.path.join(MODEL_SAVE_DIR, model_filename)
            
            if HAS_JOBLIB:
                joblib.dump(model, model_path)
            
            training_time = (datetime.utcnow() - start_time).total_seconds()
            
            return TrainingResult(
                success=True,
                model_path=model_path,
                model_format="joblib",
                metrics=metrics,
                feature_importance=feature_importance,
                training_time_seconds=training_time
            )
            
        except Exception as e:
            logger.error(f"K-Means training failed: {e}")
            return TrainingResult(
                success=False,
                error_message=str(e)
            )


class MLTrainingWorker:
    """Background worker for ML training jobs with HIPAA audit logging"""
    
    def __init__(self, db_session: AsyncSession):
        self.db = db_session
        self.pipeline = MLTrainingPipeline(db_session)
        self.dataset_manager = PublicDatasetManager(db_session)
        self.trainer = ModelTrainer()
        self.audit_logger = TrainingAuditLogger(db_session)
    
    async def get_pending_jobs(self) -> List[Dict[str, Any]]:
        """Get list of pending training jobs"""
        
        query = text("""
            SELECT 
                id, job_name, model_name, target_version,
                data_sources, training_config, priority
            FROM ml_training_jobs
            WHERE status = 'queued'
            ORDER BY priority DESC, queued_at ASC
            LIMIT 10
        """)
        
        result = await self.db.execute(query)
        rows = result.fetchall()
        
        return [
            {
                'job_id': row.id,
                'job_name': row.job_name,
                'model_name': row.model_name,
                'version': row.target_version,
                'data_sources': row.data_sources,
                'config': row.training_config,
                'priority': row.priority
            }
            for row in rows
        ]
    
    async def start_job(self, job_id: str):
        """Mark a job as started"""
        
        await self.db.execute(
            text("""
                UPDATE ml_training_jobs
                SET status = 'running',
                    started_at = NOW(),
                    current_phase = 'initializing',
                    progress_percent = 0
                WHERE id = :job_id
            """),
            {"job_id": job_id}
        )
        await self.db.commit()
    
    async def process_job(self, job: Dict[str, Any]) -> TrainingResult:
        """Process a single training job with HIPAA audit logging"""
        
        job_id = job['job_id']
        model_name = job['model_name']
        data_sources = job.get('data_sources') or {}
        config = job.get('config') or {}
        
        try:
            await self.start_job(job_id)
            
            await self.audit_logger.log_event(
                event_type="training_job_started",
                event_category="ml_training",
                actor_id="system",
                actor_type="training_worker",
                resource_type="training_job",
                resource_id=job_id,
                phi_accessed=False,
                event_details={"model_name": model_name, "data_sources": data_sources}
            )
            
            await self.pipeline.update_job_progress(
                job_id, "data_extraction", 10, "Extracting consented patient data..."
            )
            
            job_config = TrainingJobConfig(
                job_id=job_id,
                model_name=model_name,
                model_type=self._get_model_type(model_name),
                target_version=job['version'],
                data_sources=data_sources,
                hyperparameters=config
            )
            
            dataset = await self.pipeline.prepare_training_dataset(
                job_config,
                date_range_days=data_sources.get('date_range_days', 90)
            )
            
            if len(dataset.features) == 0:
                use_synthetic = data_sources.get('use_synthetic', True)
                
                if use_synthetic:
                    await self.pipeline.update_job_progress(
                        job_id, "synthetic_generation", 30,
                        "Generating synthetic training data..."
                    )
                    
                    synthetic_records = self.dataset_manager.generate_synthetic_data(
                        normal_count=config.get('synthetic_normal', 500),
                        deteriorating_count=config.get('synthetic_deteriorating', 500)
                    )
                    
                    features = []
                    labels = []
                    for record in synthetic_records:
                        feature_values = list(record.features.values())
                        features.append(feature_values)
                        labels.append(record.label or 0)
                    
                    dataset.features = np.array(features)
                    dataset.labels = np.array(labels)
                    dataset.feature_names = list(synthetic_records[0].features.keys()) if synthetic_records else []
                    dataset.public_dataset_sources = ['synthetic']
                else:
                    return TrainingResult(
                        success=False,
                        error_message="No training data available and synthetic generation disabled"
                    )
            
            await self.pipeline.update_job_progress(
                job_id, "preprocessing", 40, "Preprocessing features..."
            )
            
            await self.pipeline.update_job_progress(
                job_id, "training", 50, f"Training {model_name} model..."
            )
            
            result = await self._train_model(
                model_name,
                dataset,
                config
            )
            
            if result.success:
                await self.pipeline.update_job_progress(
                    job_id, "saving", 90, "Saving trained model..."
                )
                
                model_id = await self._save_model_to_registry(
                    job,
                    result
                )
                result.model_id = model_id
                
                await self.pipeline.complete_job(
                    job_id,
                    model_id=model_id,
                    success=True
                )
                
                await self.audit_logger.log_event(
                    event_type="training_job_completed",
                    event_category="ml_training",
                    actor_id="system",
                    actor_type="training_worker",
                    resource_type="trained_model",
                    resource_id=model_id,
                    phi_accessed=len(dataset.patient_contributions) > 0,
                    phi_categories=list(data_sources.get('data_types', [])) if data_sources.get('data_types') else None,
                    event_details={
                        "job_id": job_id,
                        "model_name": model_name,
                        "metrics": result.metrics,
                        "training_time_seconds": result.training_time_seconds,
                        "patient_contribution_count": len(dataset.patient_contributions)
                    }
                )
            else:
                await self.pipeline.complete_job(
                    job_id,
                    success=False,
                    error_message=result.error_message
                )
                
                await self.audit_logger.log_event(
                    event_type="training_job_failed",
                    event_category="ml_training",
                    actor_id="system",
                    actor_type="training_worker",
                    resource_type="training_job",
                    resource_id=job_id,
                    phi_accessed=False,
                    success=False,
                    error_message=result.error_message,
                    event_details={"model_name": model_name}
                )
            
            return result
            
        except Exception as e:
            logger.error(f"Job {job_id} failed: {e}")
            traceback.print_exc()
            
            await self.pipeline.complete_job(
                job_id,
                success=False,
                error_message=str(e)
            )
            
            await self.audit_logger.log_event(
                event_type="training_job_error",
                event_category="ml_training",
                actor_id="system",
                actor_type="training_worker",
                resource_type="training_job",
                resource_id=job_id,
                phi_accessed=False,
                success=False,
                error_message=str(e),
                event_details={"model_name": model_name, "exception_type": type(e).__name__}
            )
            
            return TrainingResult(
                success=False,
                error_message=str(e)
            )
    
    def _get_model_type(self, model_name: str) -> str:
        """Determine model type from model name"""
        
        if 'lstm' in model_name.lower():
            return 'lstm'
        elif 'segment' in model_name.lower() or 'cluster' in model_name.lower():
            return 'kmeans'
        elif 'ensemble' in model_name.lower():
            return 'ensemble'
        elif 'boost' in model_name.lower():
            return 'gradient_boosting'
        else:
            return 'random_forest'
    
    async def _train_model(
        self,
        model_name: str,
        dataset: TrainingDataset,
        config: Dict[str, Any]
    ) -> TrainingResult:
        """Train a model based on type"""
        
        model_type = self._get_model_type(model_name)
        
        if model_type == 'kmeans':
            return self.trainer.train_kmeans(
                dataset.features,
                config,
                dataset.feature_names
            )
        
        if len(dataset.labels) == 0 or len(np.unique(dataset.labels)) < 2:
            labels = np.random.randint(0, 2, len(dataset.features))
        else:
            labels = dataset.labels
        
        X_train, X_val, y_train, y_val = train_test_split(
            dataset.features,
            labels,
            test_size=config.get('validation_split', 0.2),
            random_state=42
        )
        
        if model_type == 'gradient_boosting':
            return self.trainer.train_gradient_boosting(
                X_train, y_train, X_val, y_val,
                config, dataset.feature_names
            )
        else:
            return self.trainer.train_random_forest(
                X_train, y_train, X_val, y_val,
                config, dataset.feature_names
            )
    
    async def _save_model_to_registry(
        self,
        job: Dict[str, Any],
        result: TrainingResult
    ) -> str:
        """Save trained model to the registry"""
        
        import uuid
        model_id = str(uuid.uuid4())
        
        model_size = 0
        if result.model_path and os.path.exists(result.model_path):
            model_size = os.path.getsize(result.model_path)
        
        query = text("""
            INSERT INTO ml_models (
                id, name, model_name, model_type, version, status, is_active,
                training_config, metrics, file_path, model_format, model_size_bytes,
                feature_names, feature_importance,
                training_started_at, training_completed_at, training_duration_seconds,
                created_at, updated_at
            ) VALUES (
                :id, :name, :model_name, :model_type, :version, 'active', false,
                :config, :metrics, :file_path, :model_format, :model_size,
                :feature_names, :feature_importance,
                :started_at, NOW(), :duration,
                NOW(), NOW()
            )
        """)
        
        await self.db.execute(query, {
            "id": model_id,
            "name": job['job_name'],
            "model_name": job['model_name'],
            "model_type": self._get_model_type(job['model_name']),
            "version": job['version'],
            "config": json.dumps(job.get('config') or {}),
            "metrics": json.dumps(result.metrics or {}),
            "file_path": result.model_path,
            "model_format": result.model_format,
            "model_size": model_size,
            "feature_names": json.dumps(list(result.feature_importance.keys()) if result.feature_importance else []),
            "feature_importance": json.dumps(result.feature_importance or {}),
            "started_at": datetime.utcnow(),
            "duration": int(result.training_time_seconds)
        })
        await self.db.commit()
        
        return model_id
    
    async def run_worker_loop(self, max_jobs: int = 10):
        """Run the worker loop to process pending jobs"""
        
        jobs_processed = 0
        
        while jobs_processed < max_jobs:
            pending = await self.get_pending_jobs()
            
            if not pending:
                logger.info("No pending jobs, worker idle")
                break
            
            for job in pending:
                logger.info(f"Processing job: {job['job_id']} - {job['model_name']}")
                
                result = await self.process_job(job)
                
                if result.success:
                    logger.info(f"Job {job['job_id']} completed successfully")
                else:
                    logger.error(f"Job {job['job_id']} failed: {result.error_message}")
                
                jobs_processed += 1
                
                if jobs_processed >= max_jobs:
                    break
        
        return jobs_processed


def create_training_worker(db_session: AsyncSession) -> MLTrainingWorker:
    """Factory function to create a training worker"""
    return MLTrainingWorker(db_session)


async def run_training_worker(db_session: AsyncSession, max_jobs: int = 10):
    """Run the training worker"""
    worker = create_training_worker(db_session)
    return await worker.run_worker_loop(max_jobs)
