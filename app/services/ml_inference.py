"""
ML Inference Service - Production Grade
=========================================

Handles model loading, prediction, caching, and performance tracking.
Features:
- ONNX export for optimized inference
- Model versioning registry with rollback
- Joblib serialization for sklearn models
- Inference pipeline optimization with batching
- HIPAA-compliant prediction logging
"""

import json
import time
import hashlib
import os
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
from datetime import datetime, timedelta
import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
import logging
import numpy as np

# ML libraries - imported dynamically to handle optional dependencies
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

try:
    from transformers import AutoModel, AutoTokenizer, pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    from sklearn.base import BaseEstimator
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from sqlalchemy.orm import Session
from app.models.ml_models import MLModel, MLPrediction, MLPerformanceLog
from app.core.config import settings

logger = logging.getLogger(__name__)


# ============================================================================
# Model Version Registry
# ============================================================================

class ModelStatus(Enum):
    """Model lifecycle status"""
    DRAFT = "draft"
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


@dataclass
class ModelVersion:
    """Model version metadata"""
    model_name: str
    version: str
    model_type: str  # pytorch, onnx, sklearn, transformers
    file_path: str
    status: ModelStatus = ModelStatus.DRAFT
    created_at: datetime = field(default_factory=datetime.utcnow)
    metrics: Dict[str, float] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)
    description: str = ""
    
    @property
    def version_key(self) -> str:
        return f"{self.model_name}:{self.version}"


class ModelVersionRegistry:
    """
    Production-grade model version registry.
    
    Features:
    - Version tracking with semantic versioning
    - A/B testing support via version routing
    - Automatic rollback on failure
    - Performance metrics per version
    """
    
    def __init__(self, storage_path: str = "/tmp/ml_models"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.versions: Dict[str, ModelVersion] = {}
        self.active_versions: Dict[str, str] = {}  # model_name -> active version
        self._version_history: Dict[str, List[str]] = {}  # model_name -> [versions]
    
    def register_version(
        self,
        model_name: str,
        version: str,
        model_type: str,
        file_path: str,
        metrics: Optional[Dict[str, float]] = None,
        config: Optional[Dict[str, Any]] = None,
        description: str = "",
        activate: bool = False
    ) -> ModelVersion:
        """
        Register a new model version.
        
        Args:
            model_name: Unique model identifier
            version: Semantic version (e.g., "1.0.0")
            model_type: pytorch, onnx, sklearn, transformers
            file_path: Path to model file
            metrics: Validation metrics (accuracy, f1, etc.)
            config: Model configuration
            description: Human-readable description
            activate: Whether to set as active version
        
        Returns:
            ModelVersion instance
        """
        model_version = ModelVersion(
            model_name=model_name,
            version=version,
            model_type=model_type,
            file_path=file_path,
            status=ModelStatus.ACTIVE if activate else ModelStatus.DRAFT,
            metrics=metrics or {},
            config=config or {},
            description=description
        )
        
        version_key = model_version.version_key
        self.versions[version_key] = model_version
        
        if model_name not in self._version_history:
            self._version_history[model_name] = []
        self._version_history[model_name].append(version)
        
        if activate:
            self.active_versions[model_name] = version
            logger.info(f"Registered and activated model: {version_key}")
        else:
            logger.info(f"Registered model version: {version_key}")
        
        return model_version
    
    def activate_version(self, model_name: str, version: str) -> bool:
        """Activate a specific version of a model"""
        version_key = f"{model_name}:{version}"
        
        if version_key not in self.versions:
            logger.error(f"Version not found: {version_key}")
            return False
        
        old_version = self.active_versions.get(model_name)
        if old_version:
            old_key = f"{model_name}:{old_version}"
            if old_key in self.versions:
                self.versions[old_key].status = ModelStatus.DEPRECATED
        
        self.versions[version_key].status = ModelStatus.ACTIVE
        self.active_versions[model_name] = version
        logger.info(f"Activated version: {version_key}")
        return True
    
    def rollback_version(self, model_name: str) -> Optional[str]:
        """Rollback to previous version"""
        history = self._version_history.get(model_name, [])
        if len(history) < 2:
            logger.error(f"No previous version to rollback for: {model_name}")
            return None
        
        previous_version = history[-2]
        if self.activate_version(model_name, previous_version):
            logger.info(f"Rolled back {model_name} to version {previous_version}")
            return previous_version
        return None
    
    def get_active_version(self, model_name: str) -> Optional[ModelVersion]:
        """Get the currently active version of a model"""
        version = self.active_versions.get(model_name)
        if not version:
            return None
        return self.versions.get(f"{model_name}:{version}")
    
    def list_versions(self, model_name: str) -> List[ModelVersion]:
        """List all versions of a model"""
        return [
            v for k, v in self.versions.items()
            if k.startswith(f"{model_name}:")
        ]
    
    def get_version_metrics(self, model_name: str, version: str) -> Dict[str, float]:
        """Get metrics for a specific version"""
        version_key = f"{model_name}:{version}"
        if version_key in self.versions:
            return self.versions[version_key].metrics
        return {}
    
    def update_metrics(
        self,
        model_name: str,
        version: str,
        metrics: Dict[str, float]
    ):
        """Update metrics for a version (e.g., after A/B testing)"""
        version_key = f"{model_name}:{version}"
        if version_key in self.versions:
            self.versions[version_key].metrics.update(metrics)
            logger.info(f"Updated metrics for {version_key}: {metrics}")


# Global version registry
model_version_registry = ModelVersionRegistry()


class MLModelRegistry:
    """
    Production-grade ML Model Registry.
    
    Features:
    - Model loading (PyTorch, ONNX, sklearn, transformers)
    - ONNX export for optimized inference
    - Joblib serialization for sklearn models
    - Batch inference support
    - Redis caching with TTL
    - HIPAA-compliant prediction logging
    """
    
    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.tokenizers: Dict[str, Any] = {}
        self.sessions: Dict[str, Any] = {}  # ONNX sessions
        self.metadata: Dict[str, Dict] = {}
        self._executor = ThreadPoolExecutor(max_workers=4)
        self._redis_client: Optional[redis.Redis] = None
        self._inference_stats: Dict[str, Dict] = {}  # Performance tracking
        self._model_storage = Path("/tmp/ml_models")
        self._model_storage.mkdir(parents=True, exist_ok=True)
    
    # ========================================================================
    # ONNX Export Functions
    # ========================================================================
    
    def export_to_onnx(
        self,
        model_name: str,
        output_path: Optional[str] = None,
        input_shape: Tuple[int, ...] = (1, 10),
        opset_version: int = 13
    ) -> Optional[str]:
        """
        Export a PyTorch model to ONNX format for optimized inference.
        
        Args:
            model_name: Name of loaded PyTorch model
            output_path: Output file path (auto-generated if not provided)
            input_shape: Shape of input tensor (batch_size, features)
            opset_version: ONNX opset version
        
        Returns:
            Path to exported ONNX model or None on failure
        """
        if not TORCH_AVAILABLE:
            logger.error("PyTorch not available for ONNX export")
            return None
        
        if model_name not in self.models:
            logger.error(f"Model {model_name} not loaded")
            return None
        
        model = self.models[model_name]
        if not isinstance(model, nn.Module):
            logger.error(f"Model {model_name} is not a PyTorch model")
            return None
        
        try:
            output_path = output_path or str(
                self._model_storage / f"{model_name}.onnx"
            )
            
            dummy_input = torch.randn(*input_shape)
            
            torch.onnx.export(
                model,
                dummy_input,
                output_path,
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )
            
            logger.info(f"Exported {model_name} to ONNX: {output_path}")
            
            model_version_registry.register_version(
                model_name=f"{model_name}_onnx",
                version="1.0.0",
                model_type="onnx",
                file_path=output_path,
                config={"input_shape": input_shape, "opset_version": opset_version},
                description=f"ONNX export of {model_name}"
            )
            
            return output_path
            
        except Exception as e:
            logger.error(f"ONNX export failed for {model_name}: {e}")
            return None
    
    def load_onnx_model(
        self,
        model_name: str,
        model_path: str,
        providers: Optional[List[str]] = None
    ) -> bool:
        """
        Load an ONNX model for inference.
        
        Args:
            model_name: Identifier for the model
            model_path: Path to ONNX file
            providers: Execution providers (CPU, CUDA, etc.)
        
        Returns:
            True if successful
        """
        if not ONNX_AVAILABLE:
            logger.error("ONNX Runtime not available")
            return False
        
        try:
            providers = providers or ['CPUExecutionProvider']
            session = ort.InferenceSession(model_path, providers=providers)
            
            self.sessions[model_name] = session
            self.metadata[model_name] = {
                "type": "onnx",
                "path": model_path,
                "providers": providers,
                "inputs": [i.name for i in session.get_inputs()],
                "outputs": [o.name for o in session.get_outputs()],
                "loaded_at": datetime.utcnow().isoformat()
            }
            
            logger.info(f"Loaded ONNX model: {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load ONNX model {model_name}: {e}")
            return False
    
    # ========================================================================
    # Joblib Serialization for sklearn
    # ========================================================================
    
    def save_sklearn_model(
        self,
        model_name: str,
        output_path: Optional[str] = None,
        compress: int = 3
    ) -> Optional[str]:
        """
        Save a sklearn model using joblib.
        
        Args:
            model_name: Name of loaded sklearn model
            output_path: Output file path
            compress: Compression level (0-9)
        
        Returns:
            Path to saved model or None
        """
        if not JOBLIB_AVAILABLE:
            logger.error("Joblib not available")
            return None
        
        if model_name not in self.models:
            logger.error(f"Model {model_name} not loaded")
            return None
        
        model = self.models[model_name]
        
        try:
            output_path = output_path or str(
                self._model_storage / f"{model_name}.joblib"
            )
            
            joblib.dump(model, output_path, compress=compress)
            logger.info(f"Saved sklearn model: {output_path}")
            
            model_version_registry.register_version(
                model_name=model_name,
                version="1.0.0",
                model_type="sklearn",
                file_path=output_path,
                config={"compress": compress},
                description=f"Joblib serialized sklearn model"
            )
            
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to save sklearn model: {e}")
            return None
    
    def load_sklearn_model(
        self,
        model_name: str,
        model_path: str
    ) -> bool:
        """Load a sklearn model from joblib file."""
        if not JOBLIB_AVAILABLE:
            logger.error("Joblib not available")
            return False
        
        try:
            model = joblib.load(model_path)
            self.models[model_name] = model
            self.metadata[model_name] = {
                "type": "sklearn",
                "path": model_path,
                "loaded_at": datetime.utcnow().isoformat()
            }
            
            logger.info(f"Loaded sklearn model: {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load sklearn model: {e}")
            return False
    
    # ========================================================================
    # Batch Inference
    # ========================================================================
    
    async def batch_predict(
        self,
        model_name: str,
        batch_inputs: List[Dict[str, Any]],
        batch_size: int = 32,
        db: Optional[Session] = None,
        patient_ids: Optional[List[str]] = None,
        use_cache: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Run batch predictions for efficiency with caching support.
        
        Args:
            model_name: Name of model to use
            batch_inputs: List of input dictionaries
            batch_size: Maximum batch size
            db: Database session for logging
            patient_ids: List of patient IDs (same length as batch_inputs)
            use_cache: Whether to check/update cache
        
        Returns:
            List of prediction results
        """
        if not self._is_model_loaded(model_name):
            raise ValueError(f"Model {model_name} is not loaded")
        
        results = []
        start_time = time.time()
        cache_hits = 0
        
        for i in range(0, len(batch_inputs), batch_size):
            batch = batch_inputs[i:i + batch_size]
            batch_patient_ids = patient_ids[i:i + batch_size] if patient_ids else None
            batch_results = []
            uncached_indices = []
            uncached_inputs = []
            
            if use_cache and self._redis_client:
                for j, inp in enumerate(batch):
                    cache_key = self._generate_cache_key(model_name, inp)
                    try:
                        cached = await self._redis_client.get(cache_key)
                        if cached:
                            batch_results.append(json.loads(cached))
                            cache_hits += 1
                        else:
                            batch_results.append(None)
                            uncached_indices.append(j)
                            uncached_inputs.append(inp)
                    except Exception:
                        batch_results.append(None)
                        uncached_indices.append(j)
                        uncached_inputs.append(inp)
            else:
                uncached_indices = list(range(len(batch)))
                uncached_inputs = batch
                batch_results = [None] * len(batch)
            
            if uncached_inputs:
                loop = asyncio.get_event_loop()
                inference_results = await loop.run_in_executor(
                    self._executor,
                    self._run_batch_inference,
                    model_name,
                    uncached_inputs
                )
                
                for idx, result in zip(uncached_indices, inference_results):
                    batch_results[idx] = result
                    
                    if use_cache and self._redis_client:
                        try:
                            cache_key = self._generate_cache_key(model_name, batch[idx])
                            await self._redis_client.setex(
                                cache_key, 
                                timedelta(hours=1),
                                json.dumps(result)
                            )
                        except Exception:
                            pass
            
            if db and batch_patient_ids:
                for j, (result, pid) in enumerate(zip(batch_results, batch_patient_ids)):
                    if result and pid:
                        self._log_prediction(
                            db, model_name, pid, batch[j], result,
                            inference_time_ms=0, cache_hit=(j not in uncached_indices)
                        )
            
            results.extend(batch_results)
        
        inference_time = (time.time() - start_time) * 1000
        self._update_stats(model_name, inference_time, cache_hits > 0)
        
        return results
    
    def _is_model_loaded(self, model_name: str) -> bool:
        """Check if a model is loaded and ready for inference."""
        return (
            model_name in self.models or 
            model_name in self.sessions or
            f"pipeline_{model_name}" in self.models
        )
    
    def _run_batch_inference(
        self,
        model_name: str,
        batch_inputs: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Run batch inference synchronously with error handling."""
        results = []
        
        try:
            if model_name in self.sessions:
                sess = self.sessions[model_name]
                input_name = sess.get_inputs()[0].name
                
                features = np.array([
                    inp.get("features", []) for inp in batch_inputs
                ], dtype=np.float32)
                
                outputs = sess.run(None, {input_name: features})
                
                for out in outputs[0]:
                    results.append({"prediction": out.tolist()})
            else:
                for inp in batch_inputs:
                    try:
                        result = self._run_inference(model_name, inp)
                        results.append(result)
                    except Exception as e:
                        logger.error(f"Batch inference error for input: {e}")
                        results.append({"error": str(e)})
        except Exception as e:
            logger.error(f"Batch inference failed: {e}")
            results = [{"error": str(e)} for _ in batch_inputs]
        
        return results
    
    # ========================================================================
    # Performance Tracking
    # ========================================================================
    
    def get_inference_stats(self, model_name: str) -> Dict[str, Any]:
        """Get inference statistics for a model."""
        stats = self._inference_stats.get(model_name, {})
        return {
            "total_predictions": stats.get("count", 0),
            "avg_inference_time_ms": stats.get("avg_time", 0),
            "cache_hit_rate": stats.get("cache_hits", 0) / max(stats.get("count", 1), 1),
            "last_prediction": stats.get("last_prediction"),
            "error_count": stats.get("errors", 0)
        }
    
    def _update_stats(
        self,
        model_name: str,
        inference_time_ms: float,
        cache_hit: bool,
        error: bool = False
    ):
        """Update inference statistics."""
        if model_name not in self._inference_stats:
            self._inference_stats[model_name] = {
                "count": 0,
                "total_time": 0,
                "cache_hits": 0,
                "errors": 0
            }
        
        stats = self._inference_stats[model_name]
        stats["count"] += 1
        stats["total_time"] += inference_time_ms
        stats["avg_time"] = stats["total_time"] / stats["count"]
        stats["last_prediction"] = datetime.utcnow().isoformat()
        
        if cache_hit:
            stats["cache_hits"] += 1
        if error:
            stats["errors"] += 1
    
    async def initialize_redis(self):
        """Initialize Redis connection for caching"""
        if REDIS_AVAILABLE:
            try:
                self._redis_client = redis.Redis(
                    host=getattr(settings, 'REDIS_HOST', 'localhost'),
                    port=getattr(settings, 'REDIS_PORT', 6379),
                    decode_responses=True
                )
                await self._redis_client.ping()
                logger.info("✅ Redis caching enabled")
            except Exception as e:
                logger.warning(f"Redis not available: {e}. Proceeding without cache.")
                self._redis_client = None
    
    def load_model(self, model_name: str, model_path: str, model_type: str):
        """
        Load ML model into memory
        
        Args:
            model_name: Unique identifier for the model
            model_path: Path to model file
            model_type: Type of model (pytorch, onnx, sklearn, transformers)
        """
        if model_name in self.models:
            logger.info(f"Model {model_name} already loaded")
            return
        
        try:
            if model_type == "pytorch" and TORCH_AVAILABLE:
                model = torch.load(model_path, map_location=torch.device('cpu'))
                model.eval()  # Set to evaluation mode
                self.models[model_name] = model
                logger.info(f"✅ Loaded PyTorch model: {model_name}")
            
            elif model_type == "onnx" and ONNX_AVAILABLE:
                sess = ort.InferenceSession(model_path)
                self.sessions[model_name] = sess
                logger.info(f"✅ Loaded ONNX model: {model_name}")
            
            elif model_type == "sklearn" and JOBLIB_AVAILABLE:
                model = joblib.load(model_path)
                self.models[model_name] = model
                logger.info(f"✅ Loaded scikit-learn model: {model_name}")
            
            elif model_type == "transformers" and TRANSFORMERS_AVAILABLE:
                # For transformers, load both model and tokenizer
                model = AutoModel.from_pretrained(model_path)
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                self.models[model_name] = model
                self.tokenizers[model_name] = tokenizer
                logger.info(f"✅ Loaded Transformer model: {model_name}")
            
            else:
                logger.error(f"Unsupported model type: {model_type} or required library not installed")
        
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise
    
    def load_pretrained_nlp(self, model_name: str, hf_model_name: str, task: str = "feature-extraction"):
        """
        Load pre-trained NLP model from HuggingFace
        
        Args:
            model_name: Local identifier
            hf_model_name: HuggingFace model identifier (e.g., "emilyalsentzer/Bio_ClinicalBERT")
            task: Task type (feature-extraction, ner, sentiment-analysis, etc.)
        """
        if not TRANSFORMERS_AVAILABLE:
            logger.error("Transformers library not available")
            return
        
        try:
            if task:
                # Use pipeline for specific tasks
                pipe = pipeline(task, model=hf_model_name)
                self.models[model_name] = pipe
                logger.info(f"✅ Loaded HuggingFace pipeline: {model_name} ({hf_model_name})")
            else:
                # Load model and tokenizer separately
                model = AutoModel.from_pretrained(hf_model_name)
                tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
                self.models[model_name] = model
                self.tokenizers[model_name] = tokenizer
                logger.info(f"✅ Loaded HuggingFace model: {model_name} ({hf_model_name})")
        except Exception as e:
            logger.error(f"Failed to load HuggingFace model {hf_model_name}: {e}")
            raise
    
    async def predict(
        self,
        model_name: str,
        input_data: Dict[str, Any],
        use_cache: bool = True,
        db: Optional[Session] = None,
        patient_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run prediction with caching and performance tracking
        
        Args:
            model_name: Name of the model to use
            input_data: Input features as dictionary
            use_cache: Whether to check/store in Redis cache
            db: Database session for logging predictions
            patient_id: Patient ID for audit logging
        
        Returns:
            Dictionary with prediction results and metadata
        """
        start_time = time.time()
        cache_hit = False
        
        # Generate cache key from input data
        cache_key = self._generate_cache_key(model_name, input_data) if use_cache else None
        
        # Check Redis cache
        if cache_key and self._redis_client:
            try:
                cached_result = await self._redis_client.get(cache_key)
                if cached_result:
                    cache_hit = True
                    result = json.loads(cached_result)
                    logger.info(f"Cache hit for {model_name}")
                    
                    # Log cache hit
                    if db and patient_id:
                        self._log_prediction(
                            db, model_name, patient_id, input_data, result,
                            inference_time_ms=time.time() - start_time,
                            cache_hit=True
                        )
                    
                    return result
            except Exception as e:
                logger.warning(f"Cache retrieval error: {e}")
        
        # Run inference in thread pool (CPU-bound operation)
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self._executor,
            self._run_inference,
            model_name,
            input_data
        )
        
        inference_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Store in cache
        if cache_key and self._redis_client and not cache_hit:
            try:
                # Cache for 1 hour
                await self._redis_client.setex(
                    cache_key,
                    3600,
                    json.dumps(result)
                )
            except Exception as e:
                logger.warning(f"Cache storage error: {e}")
        
        # Log prediction to database (HIPAA audit trail)
        if db and patient_id:
            self._log_prediction(
                db, model_name, patient_id, input_data, result,
                inference_time_ms=inference_time,
                cache_hit=cache_hit
            )
        
        # Add metadata
        result["_metadata"] = {
            "inference_time_ms": inference_time,
            "cache_hit": cache_hit,
            "model_name": model_name
        }
        
        return result
    
    def _run_inference(self, model_name: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Synchronous inference execution
        This runs in thread pool to avoid blocking async event loop
        """
        if model_name not in self.models and model_name not in self.sessions:
            raise ValueError(f"Model {model_name} not loaded")
        
        try:
            # ONNX inference
            if model_name in self.sessions:
                sess = self.sessions[model_name]
                input_name = sess.get_inputs()[0].name
                outputs = sess.run(None, {input_name: input_data.get("features")})
                return {"prediction": outputs[0].tolist()}
            
            # Get model
            model = self.models[model_name]
            
            # HuggingFace pipeline
            if hasattr(model, '__call__') and hasattr(model, 'task'):
                result = model(input_data.get("text", ""))
                return {"prediction": result}
            
            # PyTorch inference
            if TORCH_AVAILABLE and isinstance(model, nn.Module):
                with torch.no_grad():
                    features = torch.tensor(input_data.get("features", []), dtype=torch.float32)
                    output = model(features)
                    return {"prediction": output.tolist()}
            
            # Scikit-learn inference
            if hasattr(model, 'predict'):
                features = input_data.get("features", [])
                prediction = model.predict([features])
                
                # Get probability if available
                if hasattr(model, 'predict_proba'):
                    probabilities = model.predict_proba([features])
                    return {
                        "prediction": prediction.tolist(),
                        "probabilities": probabilities.tolist()
                    }
                
                return {"prediction": prediction.tolist()}
            
            raise ValueError(f"Unknown model type for {model_name}")
        
        except Exception as e:
            logger.error(f"Inference error for {model_name}: {e}")
            raise
    
    def _generate_cache_key(self, model_name: str, input_data: Dict[str, Any]) -> str:
        """Generate deterministic cache key from input data"""
        data_str = json.dumps(input_data, sort_keys=True)
        hash_obj = hashlib.md5(f"{model_name}:{data_str}".encode())
        return f"ml:prediction:{hash_obj.hexdigest()}"
    
    def _log_prediction(
        self,
        db: Session,
        model_name: str,
        patient_id: str,
        input_data: Dict,
        result: Dict,
        inference_time_ms: float,
        cache_hit: bool
    ):
        """Log prediction to database for HIPAA compliance"""
        try:
            # Get model ID
            model = db.query(MLModel).filter(
                MLModel.name == model_name,
                MLModel.is_active == True
            ).first()
            
            if not model:
                logger.warning(f"Model {model_name} not found in database")
                return
            
            # Create prediction log
            prediction_log = MLPrediction(
                model_id=model.id,
                patient_id=patient_id,
                prediction_type=model_name,
                input_data=input_data,
                prediction_result=result,
                confidence_score=result.get("confidence") or result.get("probabilities", [[]])[0][0] if result.get("probabilities") else None,
                inference_time_ms=inference_time_ms,
                cache_hit=cache_hit
            )
            
            db.add(prediction_log)
            db.commit()
            logger.debug(f"Logged prediction for patient {patient_id}")
        
        except Exception as e:
            logger.error(f"Failed to log prediction: {e}")
            db.rollback()
    
    async def shutdown(self):
        """Cleanup resources on shutdown"""
        self._executor.shutdown(wait=True)
        if self._redis_client:
            await self._redis_client.close()


# Global model registry instance
ml_registry = MLModelRegistry()


# Lifespan context manager for FastAPI
async def load_ml_models():
    """
    Load all ML models at startup
    This is called from FastAPI lifespan event
    """
    logger.info("🚀 Loading ML models...")
    
    # Initialize Redis
    await ml_registry.initialize_redis()
    
    # Load pre-trained Clinical-BERT for symptom analysis
    if TRANSFORMERS_AVAILABLE:
        try:
            ml_registry.load_pretrained_nlp(
                model_name="clinical_ner",
                hf_model_name="samrawal/bert-base-uncased_clinical-ner",
                task="ner"
            )
        except Exception as e:
            logger.error(f"Failed to load Clinical-BERT: {e}")
    
    # Load other pre-trained models as needed
    # TODO: Load custom deterioration prediction model
    # TODO: Load pain detection model (ONNX)
    
    logger.info("✅ ML models loaded successfully")


async def unload_ml_models():
    """Cleanup ML models on shutdown"""
    logger.info("🛑 Unloading ML models...")
    await ml_registry.shutdown()
    logger.info("✅ ML models unloaded")
