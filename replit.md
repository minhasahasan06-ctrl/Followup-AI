# Followup AI - HIPAA-Compliant Health Platform

## Overview
Followup AI is a HIPAA-compliant health monitoring platform designed for chronic care patients. It provides personalized health tracking, medication management, and wellness activities. The platform integrates AI agents (Agent Clona for patient support and Assistant Lysa for doctor assistance) to offer a comprehensive wellness monitoring and change detection system, aiming to deliver fully functional, production-ready applications that transform healthcare for individuals managing ongoing health conditions.

## User Preferences
- **Preferred communication style**: Simple, everyday language
- **Backend Language**: Python only - ALL backend code must be written in Python (FastAPI)
- **Frontend**: React/TypeScript (standard web practice)
- **Development Philosophy**: Building FULL FUNCTIONAL applications, NOT MVPs. All features must be completely functional and production-ready before delivery.
- **Code Quality**: Avoid duplicate code - use proper code reuse, modular design, and shared components/utilities.

## System Architecture

### Frontend
The frontend is built with React, TypeScript, Vite, Wouter for routing, TanStack Query for data fetching, and Tailwind CSS for styling. It uses Radix UI and shadcn/ui for a clinical aesthetic, supporting role-based routing and context-based theming. It routes Python AI endpoints to a FastAPI backend (port 8000) and other endpoints to an Express server (port 5000).

### Backend
The backend comprises a Node.js Express server (Port 5000) for chat, appointments, calendar, consultations, pain tracking, symptom journaling, and risk scoring. A Python FastAPI server (Port 8000) handles AI deterioration detection, guided examinations, mental health questionnaires, database interactions, authentication, and an async AI engine with a singleton manager for ML models.

### Core Features & Technical Implementations
- **AI Integration**: Leverages OpenAI API for PHI detection, symptom extraction, and clinical reasoning.
- **Data & Memory Services**: Includes a centralized OpenAI client with PHI detection and audit logging, a pgvector-based semantic memory service with cosine similarity search, and LlamaIndex integration for RAG.
- **ML Operations**: Features ML Observability for tracking latency and events, ML Alerting for threshold-based notifications, and ML Governance for clinical model validation and data provenance.
- **HIPAA Compliance**: Implements an immutable audit log with cryptographic hash chains, a unified access control service, and route-level access control.
- **Medical Imaging**: Utilizes MONAI for medical imaging inference, including DICOM/NIfTI PHI stripping and UNet segmentation.
- **Agent Orchestration**: Pilots LangGraph for graph-based agent orchestration for patient support.
- **Deterioration Prediction**: Statistical and AI models for health change detection and risk scoring.
- **Medication Management**: A unified medication system with AI assistance and conflict detection.
- **Multi-Agent Communication**: Facilitates communication between AI agents, users, and providers via a Message Router.
- **ML Training Infrastructure**: Production-grade system with patient consent controls and a data extraction pipeline.
- **Background Scheduler**: APScheduler-based jobs for tasks like risk scoring and ETL.
- **Analytics & Research**: Epidemiology Analytics Platform and a Research Center with medical NLP capabilities.
- **Authentication**: Stytch-based passwordless authentication with the following flows:
  - **Magic Links (Email)**: User enters email → Backend sends magic link → User clicks link → `/auth/magic-link/callback` exchanges token → Session created
  - **SMS OTP (Phone)**: User enters phone → Backend sends code → User enters code on `/auth/sms/verify` → Session created
  - **M2M Client Credentials**: Service-to-service authentication for internal APIs
  - **Key Files**: `client/src/pages/Login.tsx`, `client/src/pages/MagicLinkCallback.tsx`, `client/src/pages/SmsOtpVerify.tsx`, `client/src/lib/api.ts`, `server/stytch/authMiddleware.ts`
- **Conversational AI**: Production-grade voice conversation pipeline with real-time ASR/TTS, and a medical emergency detection system.
- **Telemedicine**: Daily.co integration for video consultations and a communication preferences system.

### Development Environment Guardrails
The development environment includes HIPAA compliance guardrails such as a Config Guard for production identifier checks, a Safe Logger for PHI redaction, and CI/CD security for secret scanning and dependency auditing.

### Cloud Run Deployment
The FastAPI backend supports deployment to GCP Cloud Run with features like scale-to-zero, cold start optimization, Secret Manager integration, and LangGraph persistence for HIPAA-compliant production environments.

**Cloud Run Proxy Architecture**:
- **Service URL**: https://followupai-backend-ujttoo34lq-uc.a.run.app
- **Authentication Flow**: Frontend → Express Proxy (/api/cloud/*) → Cloud Run
  - Express validates Stytch session tokens
  - Express adds X-Proxy-Auth header with shared secret
  - Express forwards X-User-* headers with authenticated user info
  - Cloud Run receives Google ID token for IAM authentication
- **Key Files**:
  - `server/cloudRunProxy.ts` - Express proxy router
  - `server/cloudRunAuth.ts` - Google ID token generation
  - `app/dependencies.py` - Python proxy authentication (get_proxy_user)
- **Environment Variables**:
  - `PROXY_AUTH_SECRET` - Shared secret between Express and Python (required, no default)
  - `VITE_USE_CLOUD_RUN_PROXY=true` - Enables frontend routing through proxy
  - `GOOGLE_CLOUD_RUN_URL` - Cloud Run service URL
- **Security**: No auto-user-creation, fail-safe when misconfigured, existing users only

### Vercel Deployment
The frontend is deployed to Vercel as a static SPA with Edge Middleware for password protection.

**HTTP Basic Auth (Password Protection)**:
- Implemented via Vercel Edge Middleware (`middleware.ts` at repo root)
- Triggers browser login popup when credentials are required
- Protects all HTML routes while excluding static assets and API calls
- **Environment Variables** (set in Vercel Project Settings):
  - `BASIC_AUTH_USER` - Username for site access
  - `BASIC_AUTH_PASSWORD` - Password for site access
  - Leave both empty to disable password protection
- **Excluded Paths**: `/api/*`, `/assets/*`, static files (js/css/map), images, fonts, favicon, robots.txt, sitemap.xml, manifest
- **Security Features**: Constant-time string comparison, Base64 decoding with error handling

**Frontend Environment Variables**:
- `VITE_EXPRESS_BACKEND_URL` - Express backend URL for API calls (required for Vercel deployment)
- `VITE_PYTHON_BACKEND_URL` - Python FastAPI backend URL
- `VITE_USE_CLOUD_RUN_PROXY=false` - Direct backend calls (no Express proxy)

**Cross-Domain Authentication**:
- When `CORS_ORIGINS` is configured on Express backend, cookies use `SameSite=None; Secure` for cross-domain support
- Frontend uses `credentials: 'include'` for all auth API calls via `client/src/lib/api.ts`
- The `getExpressApiUrl()` utility handles URL construction for both same-origin and cross-origin requests
- Stytch callback URLs must be configured in Stytch dashboard to match your deployment domain

### Deployment Checklist

**Vercel Environment Variables** (set in Vercel Project Settings → Environment Variables):
| Variable | Required | Description |
|----------|----------|-------------|
| `VITE_EXPRESS_BACKEND_URL` | Yes | Express backend URL (e.g., `https://your-express-backend.com`) |
| `BASIC_AUTH_USER` | No | Username for HTTP Basic Auth protection |
| `BASIC_AUTH_PASSWORD` | No | Password for HTTP Basic Auth protection |

**Stytch Dashboard Configuration**:
1. Go to Stytch Dashboard → Redirect URLs
2. Add your production frontend URLs:
   - Magic Link Callback: `https://your-domain.com/auth/magic-link/callback`
   - SMS OTP Verify: `https://your-domain.com/auth/sms/verify`

**Express Backend Environment Variables**:
| Variable | Required | Description |
|----------|----------|-------------|
| `CORS_ORIGINS` | Yes | Comma-separated list of allowed origins (e.g., `https://followupai.io`) |
| `STYTCH_PROJECT_ID` | Yes | Stytch project ID |
| `STYTCH_SECRET` | Yes | Stytch secret key |

**Post-Deployment Verification**:
1. Test HTTP Basic Auth by accessing the site (should prompt for password if configured)
2. Test patient signup with magic link (no password fields)
3. Test doctor signup with KYC file upload
4. Verify magic link callback redirects correctly
5. Test SMS OTP flow if configured

## External Dependencies

- **Authentication**: Stytch (Magic Links, SMS OTP, M2M)
- **Database**: Neon serverless PostgreSQL
- **AI Services**: OpenAI API, TensorFlow.js, PyTorch, HuggingFace Transformers, ONNX Runtime
- **Caching**: Redis
- **Video Conferencing**: Daily.co
- **Cloud Services**: Google Cloud Platform (GCS, Document AI, Healthcare NLP API, Cloud KMS, Healthcare API FHIR)