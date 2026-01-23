# Followup AI - GCP Backend Deployment

This guide covers deploying the Followup AI Python backend to Google Cloud Platform (GCP) Cloud Run with GPU support.

## Architecture

```
┌─────────────────────┐     ┌─────────────────────────────────┐
│  Replit Frontend    │────▶│  GCP Cloud Run (GPU)            │
│  (React/TypeScript) │     │  - FastAPI Backend              │
│                     │     │  - MONAI Medical Imaging        │
└─────────────────────┘     │  - LangGraph Orchestration      │
                            │  - PostgreSQL (Neon)            │
                            └─────────────────────────────────┘
```

## Prerequisites

1. **GCP Account** with billing enabled
2. **gcloud CLI** installed and authenticated
3. **Docker** installed locally (for building images)
4. **Neon PostgreSQL** database with connection string

## Quick Start

### 1. Set up GCP Project

```bash
# Set your project ID
export PROJECT_ID=your-project-id
gcloud config set project $PROJECT_ID

# Enable required APIs
gcloud services enable \
  cloudbuild.googleapis.com \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  secretmanager.googleapis.com
```

### 2. Create Secrets in Secret Manager

```bash
# Database URL (Neon PostgreSQL)
echo -n "postgresql://user:password@host/db" | \
  gcloud secrets create DATABASE_URL --data-file=-

# Stytch Authentication
echo -n "your-stytch-project-id" | \
  gcloud secrets create STYTCH_PROJECT_ID --data-file=-
echo -n "your-stytch-secret" | \
  gcloud secrets create STYTCH_SECRET --data-file=-

# OpenAI with BAA/ZDR for HIPAA compliance
echo -n "your-openai-api-key" | \
  gcloud secrets create OPENAI_API_KEY --data-file=-
echo -n "true" | \
  gcloud secrets create OPENAI_BAA --data-file=-
echo -n "true" | \
  gcloud secrets create OPENAI_ZDR --data-file=-
```

### 3. Deploy

```bash
# Run the deployment script
./gcp/deploy.sh $PROJECT_ID us-central1
```

Or use Cloud Build:

```bash
gcloud builds submit --config=gcp/cloudbuild.yaml .
```

### 4. Configure Frontend

After deployment, update your Replit frontend:

1. Go to Secrets/Environment Variables in Replit
2. Add: `VITE_API_URL=https://your-cloud-run-url.run.app`
3. Restart the frontend

## Files

| File | Description |
|------|-------------|
| `Dockerfile` | Production Docker image with CUDA/MONAI |
| `requirements-gcp.txt` | Python dependencies for GCP |
| `cloudbuild.yaml` | Cloud Build pipeline configuration |
| `cloudrun-service.yaml` | Cloud Run service definition (Knative) |
| `deploy.sh` | Automated deployment script |

## GPU Configuration

The deployment is configured for NVIDIA L4 GPU:

- **GPU Type**: nvidia-l4
- **GPU Count**: 1
- **Memory**: 8Gi
- **CPU**: 4

Modify in `cloudbuild.yaml` or `cloudrun-service.yaml` to change GPU type:
- `nvidia-l4` - Good balance of cost/performance
- `nvidia-t4` - Budget option
- `nvidia-a100-40gb` - High performance

## Environment Variables

### Required Secrets (via Secret Manager)

| Variable | Description |
|----------|-------------|
| `DATABASE_URL` | Neon PostgreSQL connection string |
| `STYTCH_PROJECT_ID` | Stytch project ID for auth |
| `STYTCH_SECRET` | Stytch secret key |
| `OPENAI_API_KEY` | OpenAI API key |
| `OPENAI_BAA` | OpenAI BAA signed flag (true/false) |
| `OPENAI_ZDR` | OpenAI ZDR enabled flag (true/false) |

### Environment Variables (set in deployment)

| Variable | Default | Description |
|----------|---------|-------------|
| `ENV` | production | Environment mode |
| `LOG_LEVEL` | INFO | Logging level |
| `WORKERS` | 4 | Uvicorn worker count |
| `CORS_ALLOWED_ORIGINS` | - | Comma-separated allowed origins |

## CORS Configuration

Set `CORS_ALLOWED_ORIGINS` to allow your Replit frontend:

```bash
CORS_ALLOWED_ORIGINS=https://your-app.replit.app,https://followup.ai
```

## Monitoring

View logs:
```bash
gcloud run services logs read followup-backend --region=us-central1
```

View metrics:
```bash
gcloud run services describe followup-backend \
  --region=us-central1 \
  --format='yaml(status)'
```

## Scaling

Adjust scaling in `cloudbuild.yaml`:

```yaml
substitutions:
  _MIN_INSTANCES: '1'    # Minimum instances (0 = scale to zero)
  _MAX_INSTANCES: '10'   # Maximum instances
  _CONCURRENCY: '80'     # Requests per instance
```

## Troubleshooting

### GPU Not Available

Ensure your project has GPU quota in the region:
```bash
gcloud compute regions describe us-central1 \
  --format="value(quotas[name='NVIDIA_L4_GPUS'])"
```

### Container Fails to Start

Check startup logs:
```bash
gcloud run services logs read followup-backend \
  --region=us-central1 \
  --limit=50
```

### Secret Access Denied

Grant the service account access:
```bash
gcloud secrets add-iam-policy-binding DATABASE_URL \
  --member="serviceAccount:followup-backend@$PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"
```

## Cost Estimation

With 1 GPU instance running 24/7:
- Cloud Run GPU: ~$500-800/month
- Artifact Registry: ~$5/month
- Secret Manager: ~$1/month

Consider setting `_MIN_INSTANCES: '0'` for development to reduce costs.
