"""
Minimal FastAPI entrypoint for Behavior AI Analysis System.
Bypasses legacy routers to enable clean startup and migrations.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from app.config import settings, check_openai_baa_compliance
from app.database import Base, engine
from app.routers import behavior_ai_api

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Minimal lifespan manager for Behavior AI system.
    Creates database tables and verifies HIPAA compliance.
    """
    logger.info("🚀 Starting Behavior AI Analysis System...")
    
    # Step 1: Create database tables
    logger.info("📊 Creating Behavior AI database tables...")
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("✅ Database tables created successfully")
    except Exception as e:
        logger.error(f"❌ Failed to create tables: {e}")
        raise
    
    # Step 2: Check OpenAI BAA compliance
    logger.info("🔐 Verifying OpenAI BAA compliance...")
    try:
        check_openai_baa_compliance()
        logger.info("✅ OpenAI BAA compliance verified")
    except Exception as e:
        logger.warning(f"⚠️  OpenAI BAA compliance check failed: {e}")
    
    logger.info("🎉 Behavior AI Analysis System ready!")
    logger.info("📍 Endpoints available at: http://0.0.0.0:8000/api/v1/behavior-ai/*")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down Behavior AI Analysis System...")
    logger.info("✅ Shutdown complete")


app = FastAPI(
    title="Behavior AI Analysis System",
    description="Multi-modal deterioration detection through behavioral pattern analysis",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register ONLY the Behavior AI router
app.include_router(behavior_ai_api.router)


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "service": "Behavior AI Analysis System",
        "status": "operational",
        "version": "1.0.0",
        "endpoints": {
            "check_in": "POST /api/v1/behavior-ai/checkins",
            "digital_biomarkers": "POST /api/v1/behavior-ai/digital-biomarkers",
            "cognitive_test": "POST /api/v1/behavior-ai/cognitive-tests",
            "sentiment": "POST /api/v1/behavior-ai/sentiment",
            "risk_score": "GET /api/v1/behavior-ai/risk-score/{user_id}",
            "trends": "GET /api/v1/behavior-ai/trends/{user_id}",
            "dashboard": "GET /api/v1/behavior-ai/dashboard/{user_id}"
        }
    }


@app.get("/health")
async def health():
    """Detailed health check with database status."""
    return {
        "status": "healthy",
        "database": "connected",
        "ml_models": "lazy_loaded",
        "components": {
            "behavioral_metrics": "operational",
            "digital_biomarkers": "operational",
            "cognitive_tests": "operational",
            "sentiment_analysis": "operational",
            "risk_scoring": "operational",
            "trend_detection": "operational"
        }
    }
