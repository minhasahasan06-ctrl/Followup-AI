# Followup AI Backend - Cloud Run Deployment Guide

This guide explains how to deploy the Python FastAPI backend to Google Cloud Run while keeping the frontend on Replit.

## Prerequisites

1. **Google Cloud CLI**: Install and configure `gcloud`
   ```bash
   # Install: https://cloud.google.com/sdk/docs/install
   gcloud auth login
   gcloud config set project followupai-medlm-prod
   ```

2. **Docker**: Required for local testing (optional)

3. **Enable Cloud Run API**:
   ```bash
   gcloud services enable run.googleapis.com
   gcloud services enable cloudbuild.googleapis.com
   gcloud services enable artifactregistry.googleapis.com
   ```

## Deployment Steps

### 1. Build and Push Docker Image

From the project root directory:

```bash
# Build the image
gcloud builds submit --config=cloud-run/cloudbuild.yaml

# Or manually:
docker build -f cloud-run/Dockerfile -t gcr.io/followupai-medlm-prod/followupai-backend .
docker push gcr.io/followupai-medlm-prod/followupai-backend
```

### 2. Deploy to Cloud Run

```bash
gcloud run deploy followupai-backend \
  --image gcr.io/followupai-medlm-prod/followupai-backend \
  --platform managed \
  --region us-central1 \
  --memory 4Gi \
  --cpu 2 \
  --timeout 300 \
  --concurrency 80 \
  --min-instances 0 \
  --max-instances 10 \
  --allow-unauthenticated
```

### 3. Set Environment Variables

Set the required environment variables in Cloud Run console or via CLI:

```bash
gcloud run services update followupai-backend \
  --region us-central1 \
  --set-env-vars "DATABASE_URL=postgresql://...,DEV_MODE_SECRET=...,SESSION_SECRET=..."
```

See `cloud-run/.env.template` for the full list of required variables.

**Critical variables:**
- `DATABASE_URL` - Neon PostgreSQL connection string
- `DEV_MODE_SECRET` - Must match Replit's DEV_MODE_SECRET for JWT auth
- `SESSION_SECRET` - Must match Replit's SESSION_SECRET
- `ALLOWED_ORIGINS` - Comma-separated list of allowed CORS origins (include your Replit URL)
- `GCS_SERVICE_ACCOUNT_KEY` - JSON service account key for GCS access
- `OPENAI_API_KEY` - For AI features

### 4. Get the Cloud Run URL

```bash
gcloud run services describe followupai-backend --region us-central1 --format="value(status.url)"
```

This will output something like: `https://followupai-backend-xxxx-uc.a.run.app`

### 5. Configure Replit Frontend

Set the following environment variable in Replit:

```
PYTHON_BACKEND_URL=https://followupai-backend-xxxx-uc.a.run.app
```

The Express server will detect this and proxy Python API calls to Cloud Run instead of spawning a local Python process.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Replit                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                Express Server (Port 5000)                │    │
│  │  - Frontend React app                                    │    │
│  │  - Express API routes (/api/*)                          │    │
│  │  - Proxies /api/python/* to Cloud Run                   │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼ (HTTPS)
┌─────────────────────────────────────────────────────────────────┐
│                      Google Cloud Run                            │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │            Python FastAPI Backend (Port 8080)            │    │
│  │  - AI/ML endpoints                                       │    │
│  │  - Health analytics                                      │    │
│  │  - Device data processing                                │    │
│  │  - ML model inference                                    │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

## Monitoring

### View Logs

```bash
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=followupai-backend" --limit 50
```

### Health Check

The backend exposes `/health` endpoint:

```bash
curl https://followupai-backend-xxxx-uc.a.run.app/health
```

## Troubleshooting

### Cold Start Issues
- Increase `--min-instances` to 1 for faster response times
- Use Cloud Run CPU always allocated option

### Memory Issues
- Increase memory: `--memory 8Gi`
- Check for memory leaks in ML models

### CORS Errors
- Ensure `ALLOWED_ORIGINS` includes your Replit URL
- Check the FastAPI CORS middleware configuration

### Authentication Issues
- Ensure `DEV_MODE_SECRET` matches between Replit and Cloud Run
- Verify JWT tokens are being passed correctly
