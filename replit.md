# Followup AI - HIPAA-Compliant Health Platform

## Overview
Followup AI is a comprehensive HIPAA-compliant health monitoring platform for immunocompromised patients. It integrates two AI agents: **Agent Clona** for patient support and **Assistant Lysa** for doctor assistance. The platform offers daily follow-ups, health tracking, medication management, wellness activities, and research capabilities, aiming to provide personalized health monitoring for patients and AI-powered insights for doctors.

## User Preferences
Preferred communication style: Simple, everyday language.

## Recent Changes

### November 12, 2025 - Complete Python Backend Migration
**Complete conversion of entire backend from JavaScript/TypeScript to Python FastAPI for cleaner architecture.**

**What Was Built:**
1. **Complete Python FastAPI Backend Structure** - Clean app/ directory with proper organization (models/, services/, routers/, utils/)
2. **All 8 Services Converted to Python:**
   - GoogleCalendarService - OAuth 2.0, bidirectional sync, conflict detection
   - GmailService - Email threading, PHI categorization
   - TwilioVoiceService - IVR system, voicemail transcription
   - AppointmentReminderService - SMS/email automation
   - ChatbotService - OpenAI GPT-4o integration
   - DoctorConsultationService - Secure record sharing
   - ResearchService - Population health analytics
   - VoiceInterfaceService - Whisper STT/TTS
3. **SQLAlchemy ORM** - All database models created for PostgreSQL
4. **AWS Cognito Authentication** - JWT verification with role-based access control
5. **API Routers** - Complete FastAPI endpoints with Pydantic validation
6. **HIPAA Compliance** - BAA checks, PHI handling, audit logging

**How to Run Python Backend:**

Since the Replit workflow runs JavaScript by default, manually start Python:

1. **Stop the current workflow** (click Stop button in Replit)
2. **Run Python backend**:
```bash
python3 start_python_server.py
```

Or use uvicorn directly:
```bash
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 5000 --reload
```

The Python backend will start on port 5000 with:
- API Docs: http://localhost:5000/docs
- Health Check: http://localhost:5000/health

**Security Status:**
✅ **ALL CRITICAL SECURITY ISSUES FIXED:**
1. ✅ **Cognito Audience Validation** - Validates `aud`/`client_id` claims when AWS_COGNITO_CLIENT_ID is set
2. ✅ **JWKS Cache Refresh** - 1-hour TTL with automatic kid rotation handling and stale cache fallback
3. ✅ **Dev-Mode Security** - Requires explicit DEV_MODE_SECRET (min 32 chars), fails closed in production
4. ✅ **DATABASE_URL Flexibility** - Optional with validate_database_url() method for flexible testing

**Current Status:**
- ✅ Python backend structurally complete and functional
- ✅ All services converted with proper authentication
- ✅ HIPAA compliance features (BAA checks, audit logging)
- ⚠️ Security hardening needed before production (documented above)
- 🔧 JavaScript backend still exists and functional (in server/ directory)

**Dual Backend Setup:**
- **JavaScript Backend** (server/) - Original implementation, currently running
- **Python Backend** (app/) - New implementation, ready to replace JavaScript
- **Frontend** (client/) - React/TypeScript, framework-agnostic (works with either backend)

### November 11, 2025 - Complete Assistant Lysa Receptionist Backend (Tasks 13-26)
Implemented comprehensive HIPAA-compliant receptionist backend services with full authentication and security:

**Core Receptionist Services (Tasks 13-16, 20):**
1. **Google Calendar Integration** - Bidirectional sync, OAuth 2.0, conflict detection, webhook handling
2. **Gmail Integration** - OAuth 2.0, email threading, PHI redaction flags, smart categorization
3. **Twilio Voice AI** - Complete IVR system, appointment scheduling, voicemail transcription, call routing
4. **Automated Reminders** - SMS (Twilio) and Email (AWS SES) 24h before appointments
5. **AI Chatbot** - GPT-4o powered clinic chatbot with conversation history, fallback responses

**Advanced Features (Tasks 21-25):**
6. **Doctor Consultations** - Secure patient record sharing system with consent management and audit logging
7. **Research Service** - Population health metrics, epidemiological data aggregation, FHIR integration framework (AWS HealthLake ready)
8. **Voice Interface** - OpenAI Whisper STT, TTS synthesis, voice-based health followups with AI analysis

**Security & Compliance:**
- All endpoints require authentication (isAuthenticated middleware)
- Role-based access control (doctor-only for sensitive operations)
- OpenAI client instantiation gated behind BAA verification
- Comprehensive audit logging for patient record access
- PHI handling compliance in email and voice services

**API Endpoints Added:**
- Calendar: sync, webhooks, disconnect (8 endpoints)
- Gmail: connect, emails, threads, disconnect (6 endpoints)
- Voice AI: IVR webhooks, voicemail transcription (4 endpoints)
- Chatbot: chat, history, feedback (3 endpoints)
- Consultations: request, approve/decline, access records (5 endpoints)
- Research: FHIR queries, epidemiology, population health, reports (4 endpoints)
- Voice: transcribe, speech synthesis, followups (3 endpoints)

**New Storage Methods:** `getUserByPhoneNumber()`, `getAllDoctors()`, `getPatientById()`, consultation CRUD methods  
**New Services:** `googleCalendarSyncService`, `gmailService`, `twilioVoiceService`, `appointmentReminderService`, `chatbotService`, `doctorConsultationService`, `researchService`, `voiceInterfaceService`  
**New Tables:** `google_calendar_sync`, `gmail_sync`, `email_threads`, `email_messages`, `doctor_consultations`, `consultation_record_access`

**Production Notes:**
- Gmail requires Google Workspace BAA for full PHI compliance
- AWS HealthLake integration requires datastore configuration (framework ready)
- All AI features require OpenAI Enterprise BAA (properly gated)
- Voice transcription uses fs.createReadStream for Node.js compatibility

## System Architecture

### Frontend
The frontend uses React with TypeScript, Vite, Wouter, TanStack Query, and Tailwind CSS. The UI/UX blends a clinical, trustworthy aesthetic with calming wellness elements, utilizing Radix UI and shadcn/ui. It supports role-based routing and context-based theming.

### Backend
**Two backend implementations are available:**

1. **Python Backend (NEW - app/)** - FastAPI with SQLAlchemy ORM, complete and ready for deployment after security hardening
2. **JavaScript Backend (CURRENT - server/)** - Express.js with Drizzle ORM, currently running

**Python Backend Features:**
- FastAPI framework with automatic OpenAPI documentation
- SQLAlchemy ORM for PostgreSQL database
- AWS Cognito JWT authentication with role-based access control
- Pydantic models for request/response validation
- All 8 services implemented: Calendar, Gmail, Voice AI, Reminders, Chatbot, Consultations, Research, Voice Interface

**JavaScript Backend Features:**
- Express.js with TypeScript
- Drizzle ORM with PostgreSQL
- AWS Cognito authentication
- All legacy services and endpoints

Authentication and authorization are handled via AWS Cognito User Pools with JWTs and role-based access control (patient/doctor), including medical license verification for doctors and optional TOTP-based 2FA. Data persistence uses PostgreSQL database (SQLAlchemy in Python, Drizzle in JavaScript).

**Core Features:**
-   **AI Integration:** Leverages OpenAI API (gpt-4o) for Agent Clona (patient support) and Assistant Lysa (doctor assistance), including sentiment analysis, medical entity extraction, and AI-generated session summaries.
-   **Personalization & Recommendation System:** A rule-based intelligent recommendation engine with RAG-enhanced AI agent personalization. It learns user preferences, provides tailored suggestions for patients and clinical insights for doctors, tracks habits, and monitors doctor wellness.
-   **Advanced Drug Interaction Detection:** An AI-powered system using GNN and NLP (via OpenAI) to detect drug-drug and drug-gene interactions with high accuracy, including automatic medication enrichment, batched processing, and severity classification.
-   **Real-Time Immune Function Monitoring:** An AI-powered digital biomarker tracking system that integrates wearable data (Fitbit, Apple Health, etc.) to create "immune digital twins," predicting immune function and infection risk.
-   **Environmental Pathogen Risk Mapping:** Provides location-based real-time environmental health monitoring, including air quality, outbreak tracking (CDC integration), wastewater surveillance (Biobot Analytics), pollen, and UV index, offering AI-generated safety recommendations.
-   **Voice-Based Daily Followups:** Utilizes OpenAI Whisper for transcription and GPT-4 for analysis of 1-minute voice recordings to extract health data and provide empathetic AI responses.
-   **Assistant Lysa Receptionist Features:** An AI-powered receptionist for doctors, managing appointments (CRUD, conflict detection, authorization), doctor availability, email threads (categorization, priority), call logs (tracking, transcription), and automated appointment reminders.
-   **Health Insight Consent:** Allows patients granular control over data sharing permissions with third-party health apps.
-   **EHR & Wearable Integration:** FHIR-based integration with major EHR systems (Epic, Cerner) and popular wearable devices (Amazfit, Garmin).

## External Dependencies

-   **Authentication:** Replit Auth, AWS Cognito.
-   **Database:** Neon serverless PostgreSQL.
-   **AI Services:** OpenAI API (GPT models).
-   **Communication:** Twilio API (SMS, voice), AWS SES (transactional emails).
-   **Cloud Services:** AWS S3 (storage), AWS Textract (OCR), AWS Comprehend Medical (medical entity extraction), AWS HealthLake (FHIR data store), AWS HealthImaging (medical imaging), AWS HealthOmics (genomic data analysis).
-   **Data Integration APIs:** PubMed E-utilities, PhysioNet WFDB, Kaggle API, WHO Global Health Observatory API, OpenWeatherMap API, Biobot Analytics.