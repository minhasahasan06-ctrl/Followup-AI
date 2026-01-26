#!/usr/bin/env python3
"""
MONAI Service Entrypoint

Starts the FastAPI server for medical imaging inference.
"""

import os
import logging
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Followup AI - MONAI Imaging Service",
    description="HIPAA-compliant medical imaging inference API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {
        "service": "MONAI Imaging Service",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "service": "monai-imaging"
    }


try:
    import sys
    sys.path.insert(0, "/app")
    from routes.imaging import router as imaging_router
    app.include_router(imaging_router)
    logger.info("Imaging routes loaded successfully")
except ImportError as e:
    logger.warning(f"Could not load imaging routes: {e}")


if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8001"))
    workers = int(os.getenv("WORKERS", "2"))
    
    logger.info(f"Starting MONAI service on {host}:{port}")
    
    uvicorn.run(
        "entrypoint:app",
        host=host,
        port=port,
        workers=workers,
        log_level="info"
    )
