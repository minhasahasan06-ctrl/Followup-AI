# Followup AI - GCP Cloud Run Deployment Guide

## Overview

This guide covers deploying the Followup AI FastAPI backend to Google Cloud Run with:
- **Scale-to-zero**: Zero cost when idle (min-instances: 0)
- **GCP Secret Manager**: Secure credential management for Stytch, Neon DB, OpenAI
- **PostgresSaver**: LangGraph state persistence across container scaling
- **Stytch Authentication**: Magic links and reCAPTCHA support
- **HIPAA Compliance**: BAA-signed OpenAI, audit logging, PHI protection

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Replit Frontend                              │
│                    (React/TypeScript/Vite)                          │
│                                                                      │
│   VITE_API_URL ──────────────────────────────────────────────────┐  │
└─────────────────────────────────────────────────────────────────────┘
                                                                       │
                                                                       ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     GCP Cloud Run                                    │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │              FastAPI Backend (Python 3.12)                   │   │
│  │                                                              │   │
│  │  • HIPAA-compliant API                                       │   │
│  │  • Stytch Authentication + reCAPTCHA                         │   │
│  │  • LangGraph with PostgresSaver                             │   │
│  │  • MONAI Medical Imaging (CPU)                               │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  Scale: 0-10 instances | CPU Boost | Gen2 Execution                 │
└─────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     GCP Secret Manager                               │
│                                                                      │
│  • DATABASE_URL (Neon PostgreSQL)                                   │
│  • STYTCH_PROJECT_ID, STYTCH_SECRET, STYTCH_PUBLIC_TOKEN           │
│  • OPENAI_API_KEY                                                   │
│  • SESSION_SECRET, DEV_MODE_SECRET                                  │
│  • STRIPE_API_KEY, STRIPE_WEBHOOK_SECRET                           │
└─────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     Neon PostgreSQL                                  │
│                                                                      │
│  • Application data                                                  │
│  • LangGraph state (PostgresSaver) - persists across scaling       │
│  • Session storage                                                   │
└─────────────────────────────────────────────────────────────────────┘
```

## Prerequisites

1. **GCP Project** with billing enabled
2. **Neon PostgreSQL** database
3. **Stytch Account** with project configured
4. **OpenAI API Key** (with BAA signed for HIPAA compliance)

## Quick Deploy

```bash
# From project root
gcloud builds submit --config cloud-run/cloudbuild.yaml .
```

## Step-by-Step Deployment

### Step 1: Set Up GCP

```bash
# Set your project
export PROJECT_ID=your-project-id
gcloud config set project $PROJECT_ID

# Enable required APIs
gcloud services enable \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  artifactregistry.googleapis.com \
  secretmanager.googleapis.com

# Create Artifact Registry repository
gcloud artifacts repositories create followupai-backend \
  --repository-format=docker \
  --location=us-central1 \
  --description="Followup AI Backend Docker images"
```

### Step 2: Create Secrets in GCP Secret Manager

```bash
# Database
echo -n "postgresql://user:pass@host:5432/db?sslmode=require" | \
  gcloud secrets create DATABASE_URL --data-file=-

# Stytch Authentication
echo -n "project-live-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx" | \
  gcloud secrets create STYTCH_PROJECT_ID --data-file=-

echo -n "secret-live-xxxxxx..." | \
  gcloud secrets create STYTCH_SECRET --data-file=-

echo -n "public-token-live-xxxxxx..." | \
  gcloud secrets create STYTCH_PUBLIC_TOKEN --data-file=-

# OpenAI
echo -n "sk-your-openai-api-key" | \
  gcloud secrets create OPENAI_API_KEY --data-file=-

# Session secrets (generate random)
openssl rand -hex 32 | gcloud secrets create SESSION_SECRET --data-file=-
openssl rand -hex 32 | gcloud secrets create DEV_MODE_SECRET --data-file=-

# Stripe (optional)
echo -n "sk_live_your-stripe-key" | \
  gcloud secrets create STRIPE_API_KEY --data-file=-
echo -n "whsec_your-webhook-secret" | \
  gcloud secrets create STRIPE_WEBHOOK_SECRET --data-file=-
```

### Step 3: Grant Cloud Run Access to Secrets

```bash
export SA="$PROJECT_ID-compute@developer.gserviceaccount.com"

for SECRET in DATABASE_URL STYTCH_PROJECT_ID STYTCH_SECRET STYTCH_PUBLIC_TOKEN \
              OPENAI_API_KEY SESSION_SECRET DEV_MODE_SECRET STRIPE_API_KEY \
              STRIPE_WEBHOOK_SECRET; do
  gcloud secrets add-iam-policy-binding $SECRET \
    --member="serviceAccount:$SA" \
    --role="roles/secretmanager.secretAccessor" 2>/dev/null || true
done
```

### Step 4: Deploy

```bash
gcloud builds submit --config cloud-run/cloudbuild.yaml .
```

### Step 5: Get Service URL

```bash
gcloud run services describe followupai-backend \
  --region us-central1 \
  --format 'value(status.url)'
```

### Step 6: Configure Stytch Dashboard

1. **Redirect URLs** (Dashboard > Configuration > Redirect URLs):
   - `https://your-cloud-run-url.run.app/api/auth/callback`
   - `https://your-replit-app.replit.app/api/auth/callback`

2. **Authorized Domains** (Dashboard > Configuration > Authorized domains):
   - Your Cloud Run domain (without `https://`)
   - Your Replit domain

3. **reCAPTCHA** (Dashboard > Configuration > reCAPTCHA):
   - Enable for Magic Links
   - Enable for SMS OTP (if using)

### Step 7: Configure Replit Frontend

In Replit Secrets, add:
```
VITE_API_URL=https://your-cloud-run-url.run.app
VITE_STYTCH_PUBLIC_TOKEN=public-token-live-xxxxxx...
```

### Step 8: Update CORS

```bash
gcloud run services update followupai-backend \
  --region us-central1 \
  --update-env-vars "CORS_ALLOWED_ORIGINS=https://your-replit.replit.app,https://followup.ai"
```

## Scale-to-Zero Configuration

| Setting | Value | Purpose |
|---------|-------|---------|
| min-instances | 0 | Zero cost when idle |
| max-instances | 10 | Scale up under load |
| cpu-boost | enabled | Faster cold starts |
| execution-environment | gen2 | Better performance |
| concurrency | 80 | Requests per container |
| timeout | 300s | 5 min for AI operations |
| memory | 2Gi | For ML models |
| cpu | 2 | For inference |

### Cold Start Optimization

The deployment includes:
- **CPU Boost**: Extra CPU during startup
- **Gen2 Environment**: Faster startup times
- **Lazy ML Loading**: Heavy models load on-demand

### Keeping Warm (Optional)

If cold starts impact UX:
```bash
gcloud run services update followupai-backend \
  --region us-central1 \
  --min-instances 1
```
Cost: ~$20-50/month for one warm instance.

## LangGraph State Persistence

LangGraph uses `PostgresSaver` to persist conversation state in Neon PostgreSQL:
- State survives container restarts
- State survives scale-to-zero events
- Conversations resume across sessions
- No data loss when containers scale down

Connection is automatic from `DATABASE_URL`.

## Monitoring

### View Logs
```bash
gcloud logging read "resource.type=cloud_run_revision \
  AND resource.labels.service_name=followupai-backend" --limit 50
```

### Health Check
```bash
curl https://your-cloud-run-url.run.app/health
```

## Troubleshooting

### CORS Errors
```bash
gcloud run services describe followupai-backend \
  --region us-central1 \
  --format 'value(spec.template.spec.containers[0].env)' | grep CORS
```

### Magic Links Not Working
1. Check Stytch Dashboard redirect URLs include your Cloud Run domain
2. Verify secrets are accessible: `gcloud secrets versions access latest --secret=STYTCH_SECRET`
3. Check domain is in Stytch Authorized Domains

### Container Won't Start
```bash
gcloud run services logs read followupai-backend --region us-central1 --limit 100
```

### Database Connection Errors
```bash
psql "$(gcloud secrets versions access latest --secret=DATABASE_URL)"
```

## GPU Migration Path

Current: CPU-only PyTorch. To enable GPU:

1. Update `Dockerfile.production` base image:
   ```dockerfile
   FROM nvidia/cuda:12.1-runtime-ubuntu22.04
   ```

2. Add GPU to Cloud Run:
   ```bash
   gcloud run services update followupai-backend \
     --region us-central1 \
     --gpu 1 --gpu-type nvidia-l4
   ```

3. Update requirements for CUDA PyTorch

## Security Checklist

- [ ] All secrets in GCP Secret Manager (not env vars)
- [ ] CORS restricted to specific domains
- [ ] HTTPS only (Cloud Run enforces)
- [ ] Stytch reCAPTCHA enabled
- [ ] OpenAI BAA signed (`OPENAI_BAA_SIGNED=true`)
- [ ] Non-root user in Docker container
- [ ] Health check endpoint verified
- [ ] HIPAA audit logging enabled

## Files Reference

| File | Purpose |
|------|---------|
| `cloud-run/Dockerfile.production` | Production Docker image |
| `cloud-run/requirements-production.txt` | Python dependencies |
| `cloud-run/cloudbuild.yaml` | Cloud Build configuration |
| `cloud-run/.env.template` | Environment variables template |
| `app/services/gcp_secrets.py` | Secret Manager integration |
