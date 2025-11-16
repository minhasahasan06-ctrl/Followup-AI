"""
ML Inference Service
Handles model loading, prediction, caching, and performance tracking
"""

import json
import time
import hashlib
from typing import Dict, Any, Optional, List
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging

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

from sqlalchemy.orm import Session
from app.models.ml_models import MLModel, MLPrediction, MLPerformanceLog
from app.core.config import settings

logger = logging.getLogger(__name__)


class MLModelRegistry:
    """
    Global registry for loaded ML models
    Models are loaded once at startup and reused across requests
    """
    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.tokenizers: Dict[str, Any] = {}
        self.sessions: Dict[str, Any] = {}  # ONNX sessions
        self.metadata: Dict[str, Dict] = {}
        self._executor = ThreadPoolExecutor(max_workers=4)  # For CPU-bound inference
        self._redis_client: Optional[redis.Redis] = None
    
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
