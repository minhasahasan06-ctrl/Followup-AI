# Express Backend - Cloud Run Deployment Guide

## Overview
This guide explains how to deploy the Express backend to Google Cloud Run. The Express backend handles Stytch authentication (magic links, SMS OTP) for the Vercel-hosted frontend.

## Architecture
```
Frontend (Vercel) → Express Backend (Cloud Run) → Python Backend (Cloud Run)
    ↓                        ↓
followupai.io        handles Stytch auth
```

## Prerequisites
1. Google Cloud account with project `followupai-medlm-prod`
2. Artifact Registry repository: `followupai-express`
3. Secrets in Secret Manager: DATABASE_URL, STYTCH_PROJECT_ID, STYTCH_SECRET, SESSION_SECRET
4. GitHub repository must be PUBLIC: https://github.com/minhasahasan06-ctrl/Followup-AI

## Deployment via Cloud Shell

### Step 1: Open Cloud Shell
1. Go to https://console.cloud.google.com
2. Click the terminal icon (>_) in the top right corner

### Step 2: Authenticate
```bash
gcloud auth login
```
Follow the prompts to authenticate with your Google account.

### Step 3: Set Project
```bash
gcloud config set project followupai-medlm-prod
```

### Step 4: Clone and Deploy
```bash
cd ~
rm -rf ~/Followup-AI
git clone https://github.com/minhasahasan06-ctrl/Followup-AI.git
cd Followup-AI
gcloud builds submit --config=cloud-run-express/cloudbuild.yaml
```

### Step 5: Get Service URL
After deployment succeeds, note the Cloud Run service URL from the output:
```
Service URL: https://followupai-express-xxxxx-uc.a.run.app
```

## Post-Deployment Configuration

### 1. Update Vercel Environment Variable
1. Go to Vercel Dashboard → Your Project → Settings → Environment Variables
2. Set `VITE_EXPRESS_BACKEND_URL` to the Cloud Run service URL
3. Redeploy the Vercel frontend

### 2. Configure Stytch Redirect URLs
In Stytch Dashboard → Redirect URLs, ensure these are added:
- `https://followupai.io/auth/magic-link/callback`
- `https://followupai.io/auth/sms/verify`

### 3. Verify Deployment
Test the health endpoint:
```bash
curl https://YOUR-CLOUD-RUN-URL/health
```

Expected response:
```json
{
  "status": "healthy",
  "service": "followupai-express",
  "stytch": "configured",
  "timestamp": "2026-02-03T..."
}
```

## Troubleshooting

### Container Fails to Start
Check Cloud Run logs:
1. Go to Cloud Run → followupai-express → Logs
2. Look for startup errors

Common issues:
- Missing secrets: Verify all secrets exist in Secret Manager
- Port issues: Ensure Dockerfile exposes port 8080

### CORS Errors in Browser
1. Verify CORS_ORIGINS in Cloud Run environment includes your frontend domain
2. Check browser console for specific error details

### Magic Link Not Sending
1. Verify Stytch redirect URLs are configured correctly
2. Check STYTCH_PROJECT_ID and STYTCH_SECRET are correct in Secret Manager
3. Ensure APP_URL environment variable matches your frontend domain

### Authentication Not Working
1. Check cookies are being set with correct SameSite/Secure attributes
2. Verify CORS credentials are enabled
3. Check SESSION_SECRET is configured

## Files Reference
- `cloud-run-express/Dockerfile` - Container build configuration
- `cloud-run-express/cloudbuild.yaml` - Cloud Build and deploy script
- `server/index-cloudrun.ts` - Express entry point (minimal dependencies)
- `server/stytch/authRoutesCloudRun.ts` - Stytch auth routes
- `server/stytch/authMiddlewareCloudRun.ts` - Auth middleware
- `server/stytch/authStorage.ts` - Minimal user database operations

## Environment Variables (Configured in cloudbuild.yaml)
- `NODE_ENV=production`
- `CORS_ORIGINS=https://followupai.io`
- `APP_URL=https://followupai.io`
- `DATABASE_URL` (from Secret Manager)
- `STYTCH_PROJECT_ID` (from Secret Manager)
- `STYTCH_SECRET` (from Secret Manager)
- `SESSION_SECRET` (from Secret Manager)
