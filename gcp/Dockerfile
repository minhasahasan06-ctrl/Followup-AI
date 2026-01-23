# Followup AI Backend - Production Dockerfile for GCP Cloud Run
# 
# This Dockerfile creates a production-ready image with:
# - Python 3.12 with CUDA 12.1 support for GPU inference
# - MONAI, PyTorch, and medical imaging dependencies
# - LangGraph with PostgreSQL persistence
# - Security hardening (non-root user, minimal attack surface)
#
# Build: docker build -t gcr.io/PROJECT_ID/followup-backend:latest -f gcp/Dockerfile .
# Run (GPU): docker run --gpus all -p 8000:8000 gcr.io/PROJECT_ID/followup-backend:latest

# ============================================================================
# Stage 1: Base image with CUDA and system dependencies
# ============================================================================
FROM nvidia/cuda:12.1-runtime-ubuntu22.04 AS base

LABEL maintainer="Followup AI <dev@followup.ai>"
LABEL description="HIPAA-Compliant Health Monitoring Backend"
LABEL version="2.0.0"

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# System dependencies for medical imaging and ML
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    python3-pip \
    curl \
    wget \
    git \
    # Medical imaging dependencies
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    # Build dependencies
    build-essential \
    pkg-config \
    # Health check
    procps \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN useradd -m -u 1000 -s /bin/bash appuser && \
    mkdir -p /app /models /data /tmp/checkpoints && \
    chown -R appuser:appuser /app /models /data /tmp/checkpoints

WORKDIR /app

# ============================================================================
# Stage 2: Python dependencies
# ============================================================================
FROM base AS dependencies

# Copy requirements first for layer caching
COPY gcp/requirements-gcp.txt /tmp/requirements.txt

# Install Python dependencies with pip optimizations
RUN python3.12 -m pip install --no-cache-dir --upgrade pip setuptools wheel && \
    python3.12 -m pip install --no-cache-dir -r /tmp/requirements.txt

# ============================================================================
# Stage 3: Production image
# ============================================================================
FROM dependencies AS production

# Copy application code
COPY --chown=appuser:appuser app/ /app/app/
COPY --chown=appuser:appuser shared/ /app/shared/

# Create necessary directories
RUN mkdir -p /app/logs && chown -R appuser:appuser /app/logs

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Environment variables (secrets injected at runtime via Secret Manager)
ENV HOST=0.0.0.0
ENV PORT=8000
ENV WORKERS=4
ENV LOG_LEVEL=INFO
ENV MODEL_REGISTRY_PATH=/models

# Start FastAPI with uvicorn
CMD ["python3.12", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
