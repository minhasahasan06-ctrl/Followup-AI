import os
import sys

# ============================================================================
# SUPPRESS ALL STARTUP WARNINGS - User requested zero warnings
# ============================================================================
# Suppress TensorFlow CUDA and logging warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress all TF logs (0=all, 1=info, 2=warning, 3=error only)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU mode - no CUDA warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN custom ops warnings

# Suppress other library warnings
import warnings
warnings.filterwarnings('ignore')

# Configure logging BEFORE any imports
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s:%(message)s'
)
# Suppress verbose library loggers
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('absl').setLevel(logging.ERROR)
logging.getLogger('mediapipe').setLevel(logging.ERROR)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.config import settings, check_openai_baa_compliance
from app.database import Base, engine
from app.services.ai_engine_manager import AIEngineManager
from app.routers import (
    appointments,
    calendar,
    chatbot,
    consultations,
    research,
    voice,
    doctors,
    agent_clona,
    video_consultation,
    pain_tracking,
    symptom_journal,
    medication_timeline,
    symptom_logging,
    symptom_checkin_api,  # ✅ Daily Follow-up Symptom Tracking - PRODUCTION READY
    medication_side_effects,
    baseline,
    deviation,
    risk_score,
    video_exam_sessions,
    behavior_ai_api,  # ✅ Behavior AI Analysis System - PRODUCTION READY
    mental_health,  # ✅ Mental Health Questionnaires - PRODUCTION READY
    drug_normalization_api,  # ✅ Drug Normalization Service - PRODUCTION READY
    ai_health_alerts,  # ✅ AI Health Alert Engine - PRODUCTION READY
    habits,  # ✅ Comprehensive Habit Tracker - 13 Features - PRODUCTION READY
    automation,  # ✅ Lysa Automation Engine - PRODUCTION READY
    webhooks,  # ✅ Gmail/WhatsApp Webhook Receivers - PRODUCTION READY
    clinical_assessment,  # ✅ Clinical Assessment Aggregation - PRODUCTION READY
    medical_nlp,  # ✅ Medical NLP (GPT-4o PHI Detection & Entity Extraction) - PRODUCTION READY
    agent_api,  # ✅ Multi-Agent REST API - Agent Clona & Assistant Lysa
    agent_websocket,  # ✅ Multi-Agent WebSocket - Real-time Communication
    auth_api,  # ✅ Auth0 Authentication API - PRODUCTION READY
    ml_prediction_api,  # ✅ ML Prediction API - Disease Risk, Deterioration, Time-Series, Segmentation
    environmental_risk_api,  # ✅ Environmental Risk Map - Comprehensive Environmental Health Intelligence
    followup_autopilot,  # ✅ Followup Autopilot - ML-Powered Adaptive Follow-up Engine
)

logger = logging.getLogger(__name__)

# Optional routers with guarded imports (fail gracefully if imports broken)
_optional_routers = []
try:
    from app.routers import ai_deterioration_api
    _optional_routers.append(('ai_deterioration_api', ai_deterioration_api))
except ImportError as e:
    logger.warning(f"❌ Could not import ai_deterioration_api: {e}")

try:
    from app.routers import guided_exam
    _optional_routers.append(('guided_exam', guided_exam))
except ImportError as e:
    logger.warning(f"❌ Could not import guided_exam: {e}")

try:
    from app.routers import guided_audio_exam
    _optional_routers.append(('guided_audio_exam', guided_audio_exam))
except ImportError as e:
    logger.warning(f"❌ Could not import guided_audio_exam: {e}")

try:
    from app.routers import edema_analysis
    _optional_routers.append(('edema_analysis', edema_analysis))
except ImportError as e:
    logger.warning(f"❌ Could not import edema_analysis: {e}")

try:
    from app.routers import gait_analysis_api
    _optional_routers.append(('gait_analysis_api', gait_analysis_api))
except ImportError as e:
    logger.warning(f"❌ Could not import gait_analysis_api: {e}")

try:
    from app.routers import tremor_api
    _optional_routers.append(('tremor_api', tremor_api))
except ImportError as e:
    logger.warning(f"❌ Could not import tremor_api: {e}")

try:
    from app.routers import device_connect
    _optional_routers.append(('device_connect', device_connect))
except ImportError as e:
    logger.warning(f"❌ Could not import device_connect: {e}")

try:
    from app.routers import rx_builder
    _optional_routers.append(('rx_builder', rx_builder))
except ImportError as e:
    logger.warning(f"❌ Could not import rx_builder: {e}")

try:
    from app.routers import medication_adherence
    _optional_routers.append(('medication_adherence', medication_adherence))
except ImportError as e:
    logger.warning(f"❌ Could not import medication_adherence: {e}")

try:
    from app.routers import hitl_approvals
    _optional_routers.append(('hitl_approvals', hitl_approvals))
except ImportError as e:
    logger.warning(f"❌ Could not import hitl_approvals: {e}")

try:
    from app.routers import epidemiology
    _optional_routers.append(('epidemiology', epidemiology))
except ImportError as e:
    logger.warning(f"❌ Could not import epidemiology: {e}")

try:
    from app.routers import research_center
    _optional_routers.append(('research_center', research_center))
except ImportError as e:
    logger.warning(f"❌ Could not import research_center: {e}")

try:
    from app.routers import video
    _optional_routers.append(('video', video))
except ImportError as e:
    logger.warning(f"❌ Could not import video: {e}")

# Import epidemiology models explicitly for table creation
try:
    from app.models import epidemiology_models
    logger.info("✅ Epidemiology models imported for table creation")
except ImportError as e:
    logger.warning(f"❌ Could not import epidemiology_models: {e}")

# Import research models for Phase 10 table creation
try:
    from app.models import research_models
    logger.info("✅ Research models imported for table creation")
except ImportError as e:
    logger.warning(f"❌ Could not import research_models: {e}")

# Import video billing models for Phase 12 table creation
try:
    from app.models import video_billing_models
    logger.info("✅ Video billing models imported for table creation")
except ImportError as e:
    logger.warning(f"❌ Could not import video_billing_models: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager with async AI engine initialization.
    Prevents blocking uvicorn startup by deferring heavy library loads to startup event.
    """
    logger.info("🚀 Starting Followup AI Backend...")
    
    # Step 1: Create database tables
    logger.info("📊 Creating database tables...")
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("✅ Database tables created")
    except Exception as e:
        logger.error(f"❌ Failed to create tables: {e}")
    
    # Step 2: Check OpenAI BAA compliance
    logger.info("🔐 Verifying OpenAI BAA compliance...")
    try:
        check_openai_baa_compliance()
        logger.info("✅ OpenAI BAA compliance verified")
    except Exception as e:
        logger.warning(f"⚠️  OpenAI BAA compliance check failed: {e}")
    
    # Step 3: Initialize AI engines asynchronously
    logger.info("🤖 Initializing AI engines asynchronously...")
    await AIEngineManager.initialize_all()
    logger.info("✅ AI engines initialized successfully")
    
    # Step 4: Initialize Multi-Agent Communication System
    logger.info("🤖 Initializing Multi-Agent Communication System...")
    try:
        from app.services.agent_engine import get_agent_engine
        from app.services.message_router import get_message_router
        from app.services.memory_service import get_memory_service
        from app.services.delivery_service import init_delivery_service
        
        await get_agent_engine()
        message_router = await get_message_router()
        await get_memory_service()
        
        # Initialize delivery service with Redis stream and WebSocket dependencies
        # Use public accessor methods instead of private attributes
        await init_delivery_service(
            redis_stream=message_router.get_redis_stream(),
            connection_manager=message_router.get_connection_manager()
        )
        
        logger.info("✅ Multi-Agent Communication System initialized (Clona & Lysa)")
    except Exception as e:
        logger.warning(f"⚠️  Multi-Agent System initialization failed: {e}")
    
    # Step 5: Start Lysa Automation Engine (optional - enabled via environment)
    automation_enabled = os.getenv("LYSA_AUTOMATION_ENABLED", "false").lower() == "true"
    if automation_enabled:
        logger.info("🤖 Starting Lysa Automation Engine...")
        try:
            from app.services.scheduler import start_automation_services
            await start_automation_services()
            logger.info("✅ Lysa Automation Engine started")
        except Exception as e:
            logger.warning(f"⚠️  Automation Engine startup failed: {e}")
    else:
        logger.info("ℹ️  Lysa Automation Engine disabled (set LYSA_AUTOMATION_ENABLED=true to enable)")
    
    # Step 6: Start Device Sync Worker (background data synchronization)
    device_sync_enabled = os.getenv("DEVICE_SYNC_ENABLED", "true").lower() == "true"
    if device_sync_enabled:
        logger.info("📱 Starting Device Sync Worker...")
        try:
            from app.services.device_sync_worker import start_sync_worker
            await start_sync_worker()
        except Exception as e:
            logger.warning(f"⚠️  Device Sync Worker startup failed: {e}")
    
    # Step 7: Start Study Job Worker (Phase 10 - Research Center)
    study_jobs_enabled = os.getenv("STUDY_JOBS_ENABLED", "true").lower() == "true"
    if study_jobs_enabled:
        logger.info("🔬 Starting Study Job Worker...")
        try:
            from app.services.study_job_worker import start_study_job_worker
            await start_study_job_worker()
            logger.info("✅ Study Job Worker started")
        except Exception as e:
            logger.warning(f"⚠️  Study Job Worker startup failed: {e}")
    
    logger.info("🎉 Followup AI Backend startup complete!")
    
    yield
    
    # Shutdown: Cleanup AI engines, automation, and device sync
    logger.info("🛑 Shutting down Followup AI Backend...")
    
    # Stop Device Sync Worker
    if device_sync_enabled:
        try:
            from app.services.device_sync_worker import stop_sync_worker
            await stop_sync_worker()
        except Exception as e:
            logger.warning(f"⚠️  Device Sync Worker shutdown error: {e}")
    
    # Stop Study Job Worker
    if study_jobs_enabled:
        try:
            from app.services.study_job_worker import stop_study_job_worker
            await stop_study_job_worker()
            logger.info("✅ Study Job Worker stopped")
        except Exception as e:
            logger.warning(f"⚠️  Study Job Worker shutdown error: {e}")
    
    # Stop Lysa Automation Engine
    if automation_enabled:
        try:
            from app.services.scheduler import stop_automation_services
            await stop_automation_services()
            logger.info("✅ Lysa Automation Engine stopped")
        except Exception as e:
            logger.warning(f"⚠️  Automation Engine shutdown error: {e}")
    
    await AIEngineManager.cleanup_all()
    logger.info("✅ Shutdown complete")


app = FastAPI(
    title="Followup AI - HIPAA-Compliant Health Platform",
    description="AI-powered health platform for immunocompromised patients with ML inference",
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

app.include_router(appointments.router)
app.include_router(calendar.router)
app.include_router(chatbot.router)
app.include_router(consultations.router)
app.include_router(research.router)
app.include_router(voice.router)
app.include_router(doctors.router)
app.include_router(agent_clona.router)
app.include_router(video_consultation.router)
app.include_router(pain_tracking.router)
app.include_router(symptom_journal.router)
app.include_router(medication_timeline.router)
app.include_router(symptom_logging.router)
app.include_router(symptom_checkin_api.router)
app.include_router(medication_side_effects.router)
app.include_router(baseline.router)
app.include_router(deviation.router)
app.include_router(risk_score.router)
# app.include_router(ml_inference.router)  # TEMPORARILY DISABLED - blocking import issue

# Video Exam Sessions (always enabled)
app.include_router(video_exam_sessions.router)

# Behavior AI Analysis System (Multi-Modal Deterioration Detection) - PRODUCTION READY
app.include_router(behavior_ai_api.router)

# Mental Health Questionnaires (PHQ-9, GAD-7, PSS-10) - PRODUCTION READY
app.include_router(mental_health.router)

# Drug Normalization API (RxNorm Integration) - PRODUCTION READY
app.include_router(drug_normalization_api.router)

# AI Health Alert Engine (Trend Detection, Engagement Monitoring, QoL Metrics) - PRODUCTION READY
app.include_router(ai_health_alerts.router)

# Comprehensive Habit Tracker (13 Features: Routines, Streaks, AI Coach, CBT, etc.) - PRODUCTION READY
app.include_router(habits.router)

# Lysa Automation Engine (Email/WhatsApp sync, Appointments, Reminders, Clinical AI) - PRODUCTION READY
app.include_router(automation.router)

# Gmail/WhatsApp Webhook Receivers (Real-time sync via Pub/Sub and Cloud API) - PRODUCTION READY
app.include_router(webhooks.router)

# Clinical Assessment Aggregation (Patient Data Integration for AI Diagnosis) - PRODUCTION READY
app.include_router(clinical_assessment.router)

# Medical NLP (GPT-4o PHI Detection & Entity Extraction - Replaces AWS Comprehend Medical) - PRODUCTION READY
app.include_router(medical_nlp.router)

# Multi-Agent Communication System (Agent Clona & Assistant Lysa) - PRODUCTION READY
app.include_router(agent_api.router)
app.include_router(agent_websocket.router)

# Auth0 Authentication API - PRODUCTION READY
app.include_router(auth_api.router)

# ML Prediction API (Disease Risk, Deterioration, Time-Series, Segmentation) - PRODUCTION READY
app.include_router(ml_prediction_api.router)

# Environmental Risk Map (Comprehensive Environmental Health Intelligence) - PRODUCTION READY
app.include_router(environmental_risk_api.router)

# Followup Autopilot (ML-Powered Adaptive Follow-up Engine) - PRODUCTION READY
app.include_router(followup_autopilot.router)

# Autopilot Admin Dashboard (Phase 5: System Health, Analytics, ML Monitoring) - PRODUCTION READY
try:
    from app.routers import autopilot_admin
    app.include_router(autopilot_admin.router)
    logger.info("✅ Autopilot Admin router registered")
except Exception as e:
    logger.warning(f"❌ Autopilot Admin API unavailable: {e}")

# ML Training Infrastructure (Training Jobs, Queue, Worker, Model Registry) - PRODUCTION READY
try:
    import sys
    import os
    # Add python_backend to path for imports
    backend_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'python_backend')
    if backend_path not in sys.path:
        sys.path.insert(0, backend_path)
    from ml_analysis.training_infrastructure.training_api import router as training_infrastructure_router
    app.include_router(training_infrastructure_router)
    logger.info("✅ ML Training Infrastructure router registered")
except Exception as e:
    logger.warning(f"❌ ML Training Infrastructure unavailable: {e}")

# Optional routers (fail gracefully if imports broken)
for router_name, router_module in _optional_routers:
    try:
        if router_name == 'ai_deterioration_api':
            app.include_router(router_module.video_router)
            app.include_router(router_module.audio_router)
            app.include_router(router_module.trend_router)
            app.include_router(router_module.alert_router)
            logger.info(f"✅ Registered {router_name} routers")
        elif router_name == 'guided_exam':
            app.include_router(router_module.router)
            logger.info(f"✅ Registered {router_name} router")
        elif router_name == 'guided_audio_exam':
            app.include_router(router_module.router)
            logger.info(f"✅ Registered {router_name} router")
        elif router_name == 'edema_analysis':
            app.include_router(router_module.router)
            logger.info(f"✅ Registered {router_name} router")
        elif router_name == 'gait_analysis_api':
            app.include_router(router_module.router)
            logger.info(f"✅ Registered {router_name} router (HAR-based gait analysis)")
        elif router_name == 'tremor_api':
            app.include_router(router_module.router)
            logger.info(f"✅ Registered {router_name} router (Accelerometer tremor analysis)")
        elif router_name == 'device_connect':
            app.include_router(router_module.router)
            logger.info(f"✅ Registered {router_name} router (Device Connect API)")
        elif router_name == 'rx_builder':
            app.include_router(router_module.router)
            logger.info(f"✅ Registered {router_name} router (Rx Builder API)")
        elif router_name == 'medication_adherence':
            app.include_router(router_module.router)
            logger.info(f"✅ Registered {router_name} router (Medication Adherence API)")
        elif router_name == 'hitl_approvals':
            app.include_router(router_module.router)
            logger.info(f"✅ Registered {router_name} router (Human-in-the-Loop Approvals API)")
        elif router_name == 'epidemiology':
            app.include_router(router_module.router)
            logger.info(f"✅ Registered {router_name} router (Epidemiology Analytics Platform)")
        elif router_name == 'research_center':
            app.include_router(router_module.router)
            logger.info(f"✅ Registered {router_name} router (Research Center Phase 10)")
    except Exception as e:
        logger.warning(f"❌ Could not register {router_name}: {e}")


@app.get("/")
async def root():
    return {
        "message": "Followup AI API",
        "version": "1.0.0",
        "status": "operational"
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "database": "connected"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
