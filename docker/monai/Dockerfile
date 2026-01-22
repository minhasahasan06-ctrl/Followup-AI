# MONAI Medical Imaging Microservice Dockerfile
#
# Production-grade container for medical imaging inference
# with GPU support and HIPAA compliance features
#
# Build:
#   docker build -t followup-monai:latest -f docker/monai/Dockerfile .
#
# Run (CPU):
#   docker run -p 8001:8001 followup-monai:latest
#
# Run (GPU):
#   docker run --gpus all -p 8001:8001 followup-monai:latest

FROM nvidia/cuda:12.1-runtime-ubuntu22.04 AS base

LABEL maintainer="Followup AI <dev@followup.ai>"
LABEL description="MONAI Medical Imaging Inference Service"
LABEL version="1.0.0"
LABEL hipaa.compliant="true"

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

RUN useradd -m -u 1000 monai && \
    mkdir -p /app /models /data && \
    chown -R monai:monai /app /models /data

WORKDIR /app

FROM base AS dependencies

COPY requirements-monai.txt .

RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r requirements-monai.txt

FROM dependencies AS production

COPY --chown=monai:monai app/services/monai_imaging.py /app/services/
COPY --chown=monai:monai app/routes/imaging.py /app/routes/
COPY --chown=monai:monai docker/monai/entrypoint.py /app/

USER monai

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8001/api/imaging/health || exit 1

ENV MONAI_DATA_DIRECTORY=/data
ENV MODEL_REGISTRY_PATH=/models
ENV LOG_LEVEL=INFO
ENV WORKERS=2
ENV HOST=0.0.0.0
ENV PORT=8001

CMD ["python3", "/app/entrypoint.py"]
