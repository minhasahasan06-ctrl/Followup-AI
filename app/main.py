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
    # ml_inference,  # TEMPORARILY DISABLED - blocking import issue
    ai_deterioration_api,  # ✅ RE-ENABLED - Fixed dependency injection
    video_exam_sessions,
    guided_exam,  # ✅ RE-ENABLED - Now uses AIEngineManager
    guided_audio_exam,  # ✅ NEW - Guided audio examination with YAMNet ML
    edema_analysis,  # ✅ NEW - DeepLab V3+ edema segmentation
    behavior_ai_api  # ✅ NEW - Behavior AI Analysis System
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager with async AI engine initialization.
    Prevents blocking uvicorn startup by deferring heavy library loads to startup event.
    """
    logger.info("🚀 Starting Followup AI Backend...")
    
    # Step 1: Create database tables
    logger.info("📊 Creating database tables...")
    Base.metadata.create_all(bind=engine)
    logger.info("✅ Database tables created")
    
    # Step 2: Check OpenAI BAA compliance
    logger.info("🔐 Verifying OpenAI BAA compliance...")
    check_openai_baa_compliance()
    logger.info("✅ OpenAI BAA compliance verified")
    
    # Step 3: Initialize AI engines asynchronously (prevents uvicorn hang)
    logger.info("🤖 Initializing AI engines asynchronously...")
    await AIEngineManager.initialize_all()
    logger.info("✅ AI engines initialized")
    
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

# AI Deterioration Detection System - 52 production endpoints
# ✅ RE-ENABLED November 19, 2025 - Fixed dependency injection using AIEngineManager pattern
app.include_router(ai_deterioration_api.video_router)
app.include_router(ai_deterioration_api.audio_router)
app.include_router(ai_deterioration_api.trend_router)
app.include_router(ai_deterioration_api.alert_router)

# Guided Video Examination System
app.include_router(video_exam_sessions.router)
app.include_router(guided_exam.router)  # ✅ RE-ENABLED - Now uses AIEngineManager

# Guided Audio Examination System
app.include_router(guided_audio_exam.router)  # ✅ NEW - With YAMNet ML, neurological metrics

# Edema/Swelling Analysis System (DeepLab V3+)
app.include_router(edema_analysis.router)  # ✅ NEW - Semantic segmentation for edema detection

# Behavior AI Analysis System (Multi-Modal Deterioration Detection)
app.include_router(behavior_ai_api.router)  # ✅ NEW - Behavioral, digital, cognitive, sentiment analysis


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
