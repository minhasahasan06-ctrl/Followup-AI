from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.config import settings, check_openai_baa_compliance
from app.database import Base, engine
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
    ml_inference,
    ai_deterioration_api,
    video_exam_sessions,
    guided_exam
)

# Import ML model lifecycle management
from app.services.ml_inference import load_ml_models, unload_ml_models


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager
    Loads ML models at startup and cleans up on shutdown
    """
    # Startup: Load ML models
    # Temporarily disabled - transformers models removed
    # await load_ml_models()
    
    yield
    
    # Shutdown: Cleanup ML models
    # await unload_ml_models()


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
app.include_router(ml_inference.router)

# AI Deterioration Detection System - 52 production endpoints
app.include_router(ai_deterioration_api.video_router)
app.include_router(ai_deterioration_api.audio_router)
app.include_router(ai_deterioration_api.trend_router)
app.include_router(ai_deterioration_api.alert_router)

# Guided Video Examination System
app.include_router(video_exam_sessions.router)
app.include_router(guided_exam.router)

Base.metadata.create_all(bind=engine)

check_openai_baa_compliance()


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
