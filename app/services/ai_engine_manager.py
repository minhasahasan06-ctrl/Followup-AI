"""
AI Engine Manager - Singleton Pattern with Async Initialization

This module provides lazy-loaded, async-initialized AI engines for FastAPI.
Prevents blocking uvicorn startup by deferring heavy library initialization
to FastAPI lifespan startup events.

Usage in FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await AIEngineManager.initialize_all()
        yield
        await AIEngineManager.cleanup_all()
"""

import asyncio
import logging
from typing import Optional, TYPE_CHECKING, Any
from sqlalchemy.orm import Session

# Use TYPE_CHECKING to avoid circular imports while providing type hints
if TYPE_CHECKING:
    from app.services.video_ai_engine import VideoAIEngine
    from app.services.audio_ai_engine import AudioAIEngine
    from app.services.trend_prediction_engine import TrendPredictionEngine
    from app.services.alert_orchestration_engine import AlertOrchestrationEngine

logger = logging.getLogger(__name__)


class AIEngineManager:
    """
    Singleton manager for AI engines with async initialization.
    Prevents blocking uvicorn startup by deferring heavy library loads.
    """
    
    _video_engine: Optional[Any] = None
    _audio_engine: Optional[Any] = None
    _trend_engine: Optional[Any] = None
    _alert_engine: Optional[Any] = None
    _initialized: bool = False
    _lock = asyncio.Lock()
    
    @classmethod
    async def initialize_all(cls) -> None:
        """
        Initialize all AI engines asynchronously.
        Called during FastAPI lifespan startup event.
        """
        async with cls._lock:
            if cls._initialized:
                logger.info("AI engines already initialized")
                return
            
            logger.info("Starting AI engine initialization...")
            
            try:
                # Initialize engines in executor to prevent blocking event loop
                loop = asyncio.get_event_loop()
                
                # Video AI Engine (MediaPipe, TensorFlow)
                # Import happens here (not at module level) to prevent blocking on module import
                logger.info("Initializing Video AI Engine...")
                def _create_video_engine():
                    from app.services.video_ai_engine import VideoAIEngine
                    return VideoAIEngine()
                
                cls._video_engine = await loop.run_in_executor(None, _create_video_engine)
                logger.info("✅ Video AI Engine initialized")
                
                # Audio AI Engine (librosa, scipy)
                # Import happens here (not at module level) to prevent blocking on module import
                logger.info("Initializing Audio AI Engine...")
                def _create_audio_engine():
                    from app.services.audio_ai_engine import AudioAIEngine
                    return AudioAIEngine()
                
                cls._audio_engine = await loop.run_in_executor(None, _create_audio_engine)
                logger.info("✅ Audio AI Engine initialized")
                
                # Trend Prediction Engine (lightweight)
                logger.info("Initializing Trend Prediction Engine...")
                from app.services.trend_prediction_engine import TrendPredictionEngine
                # Note: TrendPredictionEngine needs DB session - will be created per-request
                logger.info("✅ Trend Prediction Engine ready")
                
                # Alert Orchestration Engine (lightweight)
                logger.info("Initializing Alert Orchestration Engine...")
                from app.services.alert_orchestration_engine import AlertOrchestrationEngine
                # Note: AlertOrchestrationEngine needs DB session - will be created per-request
                logger.info("✅ Alert Orchestration Engine ready")
                
                cls._initialized = True
                logger.info("🚀 All AI engines initialized successfully!")
                
            except Exception as e:
                logger.error(f"❌ CRITICAL: AI engine initialization failed!")
                logger.error(f"   Error: {str(e)}")
                logger.error(f"   Video AI engine: {'✅ Loaded' if cls._video_engine else '❌ Failed'}")
                logger.error(f"   Audio AI engine: {'✅ Loaded' if cls._audio_engine else '❌ Failed'}")
                logger.error(f"   All deterioration detection endpoints will be unavailable!")
                cls._initialized = False
                # Re-raise to surface error in FastAPI lifespan event
                raise
    
    @classmethod
    async def cleanup_all(cls) -> None:
        """
        Cleanup all AI engines.
        Called during FastAPI lifespan shutdown event.
        """
        async with cls._lock:
            logger.info("Shutting down AI engines...")
            
            if cls._video_engine and hasattr(cls._video_engine, 'executor'):
                cls._video_engine.executor.shutdown(wait=True)
            
            if cls._audio_engine and hasattr(cls._audio_engine, 'executor'):
                cls._audio_engine.executor.shutdown(wait=True)
            
            cls._video_engine = None
            cls._audio_engine = None
            cls._trend_engine = None
            cls._alert_engine = None
            cls._initialized = False
            
            logger.info("✅ AI engines shut down successfully")
    
    @classmethod
    async def get_video_engine(cls):
        """Get Video AI Engine (lazy-loaded singleton)"""
        if cls._video_engine is None:
            async with cls._lock:
                # Double-check after acquiring lock
                if cls._video_engine is None:
                    logger.info("🔄 Lazy-loading Video AI Engine (first use)...")
                    loop = asyncio.get_event_loop()
                    def _create_video_engine():
                        from app.services.video_ai_engine import VideoAIEngine
                        return VideoAIEngine()
                    cls._video_engine = await loop.run_in_executor(None, _create_video_engine)
                    logger.info("✅ Video AI Engine lazy-loaded successfully")
        return cls._video_engine
    
    @classmethod
    async def get_audio_engine(cls):
        """Get Audio AI Engine (lazy-loaded singleton)"""
        if cls._audio_engine is None:
            async with cls._lock:
                # Double-check after acquiring lock
                if cls._audio_engine is None:
                    logger.info("🔄 Lazy-loading Audio AI Engine (first use)...")
                    loop = asyncio.get_event_loop()
                    def _create_audio_engine():
                        from app.services.audio_ai_engine import AudioAIEngine
                        return AudioAIEngine()
                    cls._audio_engine = await loop.run_in_executor(None, _create_audio_engine)
                    logger.info("✅ Audio AI Engine lazy-loaded successfully")
        return cls._audio_engine
    
    @classmethod
    def get_trend_engine(cls, db: Session) -> "TrendPredictionEngine":
        """Get Trend Prediction Engine (per-request instance)"""
        from app.services.trend_prediction_engine import TrendPredictionEngine
        return TrendPredictionEngine(db)
    
    @classmethod
    def get_alert_engine(cls, db: Session) -> "AlertOrchestrationEngine":
        """Get Alert Orchestration Engine (per-request instance)"""
        from app.services.alert_orchestration_engine import AlertOrchestrationEngine
        return AlertOrchestrationEngine(db)
    
    @classmethod
    def is_initialized(cls) -> bool:
        """Check if engines are initialized"""
        return cls._initialized


# FastAPI dependency functions
# Note: No return type annotations to avoid FastAPI validation issues
# The actual types are checked at runtime, not during route registration
def get_video_ai_engine():
    """FastAPI dependency for Video AI Engine"""
    return AIEngineManager.get_video_engine()


def get_audio_ai_engine():
    """FastAPI dependency for Audio AI Engine"""
    return AIEngineManager.get_audio_engine()


def get_trend_prediction_engine(db: Session):
    """FastAPI dependency for Trend Prediction Engine"""
    return AIEngineManager.get_trend_engine(db)


def get_alert_orchestration_engine(db: Session):
    """FastAPI dependency for Alert Orchestration Engine"""
    return AIEngineManager.get_alert_engine(db)
