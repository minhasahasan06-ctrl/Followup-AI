"""
Cloud Run entrypoint for Followup AI Backend
Optimized for containerized deployment with essential routers
"""
import os
import sys

# Suppress warnings for cleaner logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import warnings
warnings.filterwarnings('ignore')

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s:%(message)s'
)
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('absl').setLevel(logging.ERROR)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

# Check if DATABASE_URL is configured
DATABASE_CONFIGURED = bool(os.getenv("DATABASE_URL"))
routers_to_load = []

if DATABASE_CONFIGURED:
    logger.info("📊 DATABASE_URL configured - loading full routers")
    
    # Import configuration and database
    try:
        from app.config import settings
        from app.database import Base, engine
        logger.info("✅ Config and database modules loaded")
    except Exception as e:
        logger.error(f"❌ Failed to load config/database: {e}")
        DATABASE_CONFIGURED = False

    if DATABASE_CONFIGURED:
        # Core routers - each with try/except for graceful degradation
        router_imports = [
            ('appointments', 'app.routers.appointments'),
            ('calendar', 'app.routers.calendar'),
            ('chatbot', 'app.routers.chatbot'),
            ('consultations', 'app.routers.consultations'),
            ('research', 'app.routers.research'),
            ('voice', 'app.routers.voice'),
            ('doctors', 'app.routers.doctors'),
            ('agent_clona', 'app.routers.agent_clona'),
            ('video_consultation', 'app.routers.video_consultation'),
            ('pain_tracking', 'app.routers.pain_tracking'),
            ('symptom_journal', 'app.routers.symptom_journal'),
            ('medication_timeline', 'app.routers.medication_timeline'),
            ('symptom_logging', 'app.routers.symptom_logging'),
            ('symptom_checkin_api', 'app.routers.symptom_checkin_api'),
            ('medication_side_effects', 'app.routers.medication_side_effects'),
            ('baseline', 'app.routers.baseline'),
            ('deviation', 'app.routers.deviation'),
            ('risk_score', 'app.routers.risk_score'),
            ('video_exam_sessions', 'app.routers.video_exam_sessions'),
            ('behavior_ai_api', 'app.routers.behavior_ai_api'),
            ('mental_health', 'app.routers.mental_health'),
            ('drug_normalization_api', 'app.routers.drug_normalization_api'),
            ('ai_health_alerts', 'app.routers.ai_health_alerts'),
            ('habits', 'app.routers.habits'),
            ('automation', 'app.routers.automation'),
            ('webhooks', 'app.routers.webhooks'),
            ('clinical_assessment', 'app.routers.clinical_assessment'),
            ('medical_nlp', 'app.routers.medical_nlp'),
            ('agent_api', 'app.routers.agent_api'),
            ('auth_api', 'app.routers.auth_api'),
            ('ml_prediction_api', 'app.routers.ml_prediction_api'),
            ('environmental_risk_api', 'app.routers.environmental_risk_api'),
            ('followup_autopilot', 'app.routers.followup_autopilot'),
        ]
        
        for name, module_path in router_imports:
            try:
                import importlib
                module = importlib.import_module(module_path)
                routers_to_load.append((name, module.router))
                logger.info(f"✅ Loaded {name}")
            except Exception as e:
                logger.warning(f"❌ Could not import {name}: {e}")
        
        # Optional complex routers
        try:
            from app.routers import ai_deterioration_api
            routers_to_load.append(('ai_deterioration_video', ai_deterioration_api.video_router))
            routers_to_load.append(('ai_deterioration_audio', ai_deterioration_api.audio_router))
            routers_to_load.append(('ai_deterioration_trend', ai_deterioration_api.trend_router))
            routers_to_load.append(('ai_deterioration_alert', ai_deterioration_api.alert_router))
            logger.info("✅ Loaded ai_deterioration_api")
        except Exception as e:
            logger.warning(f"❌ Could not import ai_deterioration_api: {e}")
        
        optional_routers = [
            ('guided_exam', 'app.routers.guided_exam'),
            ('guided_audio_exam', 'app.routers.guided_audio_exam'),
            ('edema_analysis', 'app.routers.edema_analysis'),
            ('device_connect', 'app.routers.device_connect'),
            ('rx_builder', 'app.routers.rx_builder'),
            ('medication_adherence', 'app.routers.medication_adherence'),
            ('autopilot_admin', 'app.routers.autopilot_admin'),
        ]
        
        for name, module_path in optional_routers:
            try:
                import importlib
                module = importlib.import_module(module_path)
                routers_to_load.append((name, module.router))
                logger.info(f"✅ Loaded {name}")
            except Exception as e:
                logger.warning(f"❌ Could not import {name}: {e}")
else:
    logger.warning("⚠️ DATABASE_URL not configured - running in minimal mode")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Cloud Run lifespan - lightweight initialization"""
    logger.info("🚀 Starting Followup AI Backend (Cloud Run)...")
    
    if DATABASE_CONFIGURED:
        # Create database tables
        logger.info("📊 Creating database tables...")
        try:
            Base.metadata.create_all(bind=engine)
            logger.info("✅ Database tables ready")
        except Exception as e:
            logger.error(f"❌ Database initialization failed: {e}")
    
    logger.info(f"✅ Loaded {len(routers_to_load)} routers")
    logger.info("🎉 Followup AI Backend (Cloud Run) ready!")
    
    yield
    
    logger.info("🛑 Shutting down Followup AI Backend (Cloud Run)...")


app = FastAPI(
    title="Followup AI - HIPAA-Compliant Health Platform",
    description="AI-powered health platform for immunocompromised patients (Cloud Run)",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register all loaded routers
for router_name, router in routers_to_load:
    try:
        app.include_router(router)
    except Exception as e:
        logger.warning(f"❌ Failed to register {router_name}: {e}")


@app.get("/")
async def root():
    return {
        "message": "Followup AI API (Cloud Run)",
        "version": "1.0.0",
        "status": "operational",
        "database_configured": DATABASE_CONFIGURED,
        "routers_loaded": len(routers_to_load)
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "followupai-backend",
        "environment": "cloud-run",
        "database_configured": DATABASE_CONFIGURED
    }


@app.get("/api/health")
async def api_health():
    return {
        "status": "healthy",
        "service": "followupai-backend",
        "routers": len(routers_to_load),
        "database_configured": DATABASE_CONFIGURED
    }


logger.info("🚀 Cloud Run app module loaded successfully")
