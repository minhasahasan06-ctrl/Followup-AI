# Followup AI - GCP Cloud Run Deployment

Production-grade deployment of the Followup AI unified backend (Express + FastAPI) to Google Cloud Run.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Google Cloud Run                          │
│  ┌─────────────────────────────────────────────────────────┐│
│  │              Unified Container (CUDA-ready)              ││
│  │  ┌──────────────────┐    ┌────────────────────────────┐ ││
│  │  │   Express.js     │    │      FastAPI (Python)      │ ││
│  │  │   (Port 5000)    │───▶│       (Port 8000)          │ ││
│  │  │   - Auth         │    │   - AI/ML Services         │ ││
│  │  │   - API Gateway  │    │   - MONAI Imaging          │ ││
│  │  │   - Chat         │    │   - LangGraph              │ ││
│  │  └──────────────────┘    └────────────────────────────┘ ││
│  │                    Supervisord                           ││
│  └─────────────────────────────────────────────────────────┘│
│                           ▲                                  │
│           External traffic (Port 5000)                       │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
              ┌─────────────────────────────┐
              │      Replit Frontend        │
              │    (React/TypeScript)       │
              │   VITE_API_URL → Cloud Run  │
              └─────────────────────────────┘
```

## Cost Comparison

| Mode | Min Instances | Idle Cost | Active Cost | Best For |
|------|---------------|-----------|-------------|----------|
| CPU-only (default) | 0 | **$0/month** | ~$0.00002400/vCPU-sec | Pre-revenue, testing |
| GPU (NVIDIA L4) | 1 | ~$200/month | ~$0.001234/GPU-sec | Production with paying customers |

## Quick Start

### Prerequisites

1. [Google Cloud SDK](https://cloud.google.com/sdk/docs/install) installed and authenticated
2. Docker installed locally
3. GCP project with billing enabled

### 1. Set Up GCP Secrets

```bash
export PROJECT_ID="your-gcp-project-id"

# Create required secrets (replace with actual values)
echo -n "postgresql://user:pass@host/db" | gcloud secrets create DATABASE_URL --data-file=- --project=$PROJECT_ID
echo -n "project-live-xxx" | gcloud secrets create STYTCH_PROJECT_ID --data-file=- --project=$PROJECT_ID  
echo -n "secret-live-xxx" | gcloud secrets create STYTCH_SECRET --data-file=- --project=$PROJECT_ID
echo -n "sk-xxx" | gcloud secrets create OPENAI_API_KEY --data-file=- --project=$PROJECT_ID
echo -n "true" | gcloud secrets create OPENAI_BAA --data-file=- --project=$PROJECT_ID
echo -n "true" | gcloud secrets create OPENAI_ZDR --data-file=- --project=$PROJECT_ID
```

### 2. Deploy (CPU-Only, Scale-to-Zero)

```bash
# Default: CPU-only, scale-to-zero ($0 when idle)
./gcp/deploy.sh your-project-id us-central1
```

### 3. Configure Frontend

After deployment, add to Replit secrets:
```
VITE_API_URL=https://followup-backend-xxxxx-uc.a.run.app
```

Restart the Replit application.

## Deployment Options

### CPU-Only (Default) - Recommended for Pre-Revenue

```bash
./gcp/deploy.sh PROJECT_ID us-central1
```

- Scales to zero when idle (**$0 cost**)
- Cold starts: ~5-10 seconds for first request
- Suitable for: Development, testing, early customers

### GPU Enabled - For Production Workloads

```bash
./gcp/deploy.sh PROJECT_ID us-central1 --gpu
```

- NVIDIA L4 GPU attached
- Min 1 instance (no scale-to-zero)
- Fast MONAI medical imaging inference
- Suitable for: Paying customers, production

### Using Cloud Build (CI/CD)

```bash
# CPU-only
gcloud builds submit --config=gcp/cloudbuild.yaml .

# With GPU
gcloud builds submit --config=gcp/cloudbuild.yaml --substitutions=_ENABLE_GPU=true .
```

## File Structure

```
gcp/
├── Dockerfile.unified    # Combined Express + FastAPI with CUDA
├── supervisord.conf      # Process manager for both services
├── requirements-gcp.txt  # Python dependencies (ML/AI)
├── cloudbuild.yaml       # Cloud Build pipeline
├── cloudrun-service.yaml # Knative service definition
├── deploy.sh            # One-command deployment script
└── README.md            # This file
```

## Environment Variables

| Variable | Description | Set By |
|----------|-------------|--------|
| `DATABASE_URL` | Neon PostgreSQL connection | Secret Manager |
| `STYTCH_PROJECT_ID` | Stytch auth project | Secret Manager |
| `STYTCH_SECRET` | Stytch auth secret | Secret Manager |
| `OPENAI_API_KEY` | OpenAI API key | Secret Manager |
| `OPENAI_BAA` | BAA compliance flag | Secret Manager |
| `OPENAI_ZDR` | ZDR compliance flag | Secret Manager |
| `CORS_ALLOWED_ORIGINS` | Allowed frontend URLs | Deploy script |
| `WORKERS` | Uvicorn worker count | Deploy script (2 CPU, 4 GPU) |

## Upgrading from CPU to GPU

When you have paying customers and need faster inference:

```bash
# Upgrade to GPU (same image, just different resource allocation)
./gcp/deploy.sh PROJECT_ID us-central1 --gpu
```

No rebuild required - the container image includes CUDA support.

## Monitoring

### View Logs

```bash
gcloud run services logs read followup-backend --region=us-central1 --limit=100
```

### Check Service Status

```bash
gcloud run services describe followup-backend --region=us-central1
```

### View Metrics

Visit: https://console.cloud.google.com/run/detail/us-central1/followup-backend/metrics

## Troubleshooting

### Cold Start Timeout

If first requests fail after idle period:
- Increase `startupProbe.failureThreshold` in cloudrun-service.yaml
- Consider min-instances=1 if cold starts are unacceptable

### Memory Issues

For large medical images:
```bash
# Increase memory (requires proportional CPU)
gcloud run services update followup-backend --memory=8Gi --cpu=4 --region=us-central1
```

### CORS Errors

Update CORS origins:
```bash
gcloud run services update followup-backend \
  --set-env-vars="CORS_ALLOWED_ORIGINS=https://your-domain.com,https://another.com" \
  --region=us-central1
```

## Security

- All secrets managed via GCP Secret Manager
- Non-root container user
- HTTPS enforced by Cloud Run
- CORS restricted to configured origins
- HIPAA-compliant audit logging enabled
- Service account with minimal permissions

## Rolling Back

```bash
# List revisions
gcloud run revisions list --service=followup-backend --region=us-central1

# Route traffic to previous revision
gcloud run services update-traffic followup-backend \
  --to-revisions=followup-backend-00001=100 \
  --region=us-central1
```

## Deleting the Deployment

```bash
# Delete service (stops billing)
gcloud run services delete followup-backend --region=us-central1

# Optionally delete repository
gcloud artifacts repositories delete followup-ai --location=us-central1
```
