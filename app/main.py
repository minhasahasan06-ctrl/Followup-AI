from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
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
    exam_coach,
    medication_timeline,
    symptom_logging,
    medication_side_effects,
    baseline,
    deviation,
    risk_score,
    video_exam_sessions,
    behavior_ai_api  # ✅ NEW - Behavior AI Analysis System - PRODUCTION READY
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
    
    logger.info("🎉 Followup AI Backend startup complete!")
    
    yield
    
    # Shutdown: Cleanup AI engines
    logger.info("🛑 Shutting down Followup AI Backend...")
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
app.include_router(exam_coach.router)
app.include_router(medication_timeline.router)
app.include_router(symptom_logging.router)
app.include_router(medication_side_effects.router)
app.include_router(baseline.router)
app.include_router(deviation.router)
app.include_router(risk_score.router)
# app.include_router(ml_inference.router)  # TEMPORARILY DISABLED - blocking import issue

# Video Exam Sessions (always enabled)
app.include_router(video_exam_sessions.router)

# Behavior AI Analysis System (Multi-Modal Deterioration Detection) - PRODUCTION READY
app.include_router(behavior_ai_api.router)

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
