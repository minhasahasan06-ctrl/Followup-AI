# Followup AI - Autoscale Deployment Guide

## Overview
This guide explains how to deploy the Followup AI platform with autoscale deployment to get adequate resources for the Python FastAPI backend and all ML models.

## Architecture
- **Express Backend** (Node.js) - Port 5000: Agent Clona, appointments, pain tracking, symptom journal
- **FastAPI Backend** (Python) - Port 8000: AI deterioration detection, Behavior AI, ML inference, video/audio analysis

Both backends run in parallel and share the PostgreSQL database.

## Deployment Configuration

### 1. Build Command
```bash
npm run build
```
This compiles the Node.js/TypeScript backend to `dist/` folder.

### 2. Run Command
```bash
bash start_deployment.sh
```
This script (`start_deployment.sh`) starts both backends in parallel:
- Python FastAPI on port 8000 (background)
- Node.js Express on port 5000 (foreground)

### 3. Recommended Machine Configuration
For the Python ML models (Transformer, XGBoost, DistilBERT, DeepLab V3+, MediaPipe):
- **Minimum**: 2 vCPU, 4 GiB RAM
- **Recommended**: 2 vCPU, 8 GiB RAM
- **For High Traffic**: 4 vCPU, 16 GiB RAM

### 4. Environment Variables (Deployment)
All secrets are already configured in Replit. The deployment will automatically have access to:
- `AWS_*` - AWS credentials for S3, SES, Cognito
- `DATABASE_URL` - PostgreSQL connection
- `OPENAI_API_KEY` - OpenAI API
- `SESSION_SECRET` - Express sessions
- `TWILIO_*` - SMS notifications

### 5. Port Configuration
- External Port 80 → Internal Port 5000 (Express frontend/API)
- External Port 8080 → Internal Port 8000 (Python AI API)

## Setup Steps

### Through Replit UI:
1. Click "Publish" button in header
2. Select "Autoscale" deployment type
3. Configure:
   - **Deployment target**: Autoscale
   - **Machine power**: 2 vCPU, 8 GiB RAM (or higher)
   - **Max machines**: 3 (adjust based on traffic)
   - **Build command**: `npm run build`
   - **Run command**: `bash start_deployment.sh`
4. Click "Publish"

### Verify Deployment:
1. **Express Backend Health**: `https://your-app.replit.app/health`
2. **Python Backend Health**: `https://your-app.replit.app:8080/health`
3. **Behavior AI Endpoints**: `https://your-app.replit.app:8080/api/v1/behavior-ai/dashboard/{user_id}`
4. **API Documentation**: `https://your-app.replit.app:8080/docs`

## Database Migration
The Python backend automatically creates all database tables on startup via:
```python
Base.metadata.create_all(bind=engine)
```

The 9 Behavior AI tables will be created:
- behavioral_checkins
- behavioral_metrics
- digital_biomarkers
- cognitive_tests
- sentiment_analysis
- risk_scores
- deterioration_trends
- behavior_alerts
- behavioral_insights

## Monitoring
- **Application logs**: Available in Replit deployment dashboard
- **Database status**: Check via `/health` endpoints
- **ML model status**: Check via `/docs` - models lazy-load on first use

## Cost Estimation
**Autoscale Deployment** (Pay per use):
- Compute Units: $3.20 per million units
- Requests: $1.20 per million requests
- Base fee: $1/month

**Example**: 2 vCPU, 8 GiB RAM instance running 24/7:
- CPU: 2 * 86400 seconds/day * 30 days * 18 CU/second = ~93M CU = $298/month
- RAM: 8 * 86400 * 30 * 2 CU/second = ~41M CU = $131/month
- **Total**: ~$429/month + requests

For cost optimization, consider Reserved VM ($80/month for 2 vCPU / 8GB RAM fixed cost).

## Troubleshooting

### Python Backend Not Starting
- Check logs for import errors
- Verify ML dependencies installed (torch, transformers, tensorflow-hub)
- Increase machine RAM if OOM errors

### Database Connection Issues
- Verify DATABASE_URL environment variable
- Check PostgreSQL connection limits

### ML Models Not Loading
- Check TensorFlow/PyTorch installation
- Verify sufficient RAM (models need ~2GB each)
- Models lazy-load on first use - check `/docs` endpoint

## Support
For deployment issues specific to Replit platform, contact Replit support.
For application issues, check application logs in deployment dashboard.
