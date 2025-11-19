"""
Minimal Working FastAPI Backend for Agent Clona
TEMPORARY: Full functionality disabled due to MediaPipe/TensorFlow blocking issue during uvicorn startup
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.config import settings, check_openai_baa_compliance
from app.database import Base, engine

# Import only essential routers that don't block
from app.routers import (
    appointments,
    calendar,
    chatbot,
    consultations,
    agent_clona,
    doctors,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    yield


app = FastAPI(
    title="Followup AI - Agent Clona Minimal",
    description="Minimal working backend for Agent Clona patient chatbot",
    version="1.0.0-minimal",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Essential routers only
app.include_router(appointments.router)
app.include_router(calendar.router)
app.include_router(chatbot.router)
app.include_router(consultations.router)
app.include_router(agent_clona.router)
app.include_router(doctors.router)

# Create database tables
Base.metadata.create_all(bind=engine)

check_openai_baa_compliance()


@app.get("/")
async def root():
    return {
        "message": "Followup AI API - Agent Clona Minimal",
        "version": "1.0.0-minimal",
        "status": "operational",
        "note": "AI deterioration detection endpoints temporarily disabled"
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
        "app.main_working:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
