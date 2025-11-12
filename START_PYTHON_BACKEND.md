# How to Run the Python Backend

## Quick Start

Since the Replit workflow is configured to run the JavaScript backend (`npm run dev`), you have two options to use the Python backend:

### Option 1: Manual Start (Recommended for Development)

1. **Stop the current workflow** (click Stop in the Replit UI)
2. **Run the Python backend**:
   ```bash
   python3 start_python_server.py
   ```

This will start the FastAPI backend on port 5000.

### Option 2: Use the Shell

Open the Shell tab in Replit and run:
```bash
./run_python_backend.sh
```

## What's Running

- **Python Backend**: FastAPI with SQLAlchemy, running on `http://0.0.0.0:5000`
- **Frontend**: React app (served by FastAPI in production, Vite in development)

## Development Mode Configuration

✅ **DEV_MODE_SECRET** is configured - this allows authentication in development mode without AWS Cognito.

## Endpoints

Once running, you can access:
- API Documentation: `http://localhost:5000/docs` (Swagger UI)
- Health Check: `http://localhost:5000/health`
- Root: `http://localhost:5000/`

## Features

All 8 services are available:
1. **Appointments** - `/api/appointments/*`
2. **Google Calendar Sync** - `/api/calendar/*`
3. **AI Chatbot** - `/api/chatbot/*`
4. **Doctor Consultations** - `/api/consultations/*`
5. **Research Service** - `/api/research/*`
6. **Voice Interface** - `/api/voice/*`
7. **Gmail Integration** - `/api/gmail/*`
8. **Twilio Voice** - `/api/twilio/*`

## Security Features

✅ **Implemented:**
- AWS Cognito JWT verification with JWKS
- Audience (aud) claim validation
- JWKS cache with 1-hour TTL and auto-refresh
- Dev-mode requires explicit DEV_MODE_SECRET (min 32 chars)
- Fails closed in production without Cognito config
- Role-based access control (patient/doctor)
- HIPAA compliance checks (BAA, PHI handling, audit logging)

## Production Deployment

Before deploying to production:
1. Configure AWS Cognito environment variables
2. Remove or secure DEV_MODE_SECRET
3. Set `ENVIRONMENT=production`
4. Sign OpenAI BAA and set `OPENAI_BAA_SIGNED=true`
