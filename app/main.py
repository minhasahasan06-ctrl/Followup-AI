from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
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
    symptom_journal
)

app = FastAPI(
    title="Followup AI - HIPAA-Compliant Health Platform",
    description="AI-powered health platform for immunocompromised patients",
    version="1.0.0"
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
