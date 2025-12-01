# Followup AI - HIPAA-Compliant Health Platform

## Overview
Followup AI is a HIPAA-compliant health monitoring platform for immunocompromised patients, offering personalized health tracking, medication management, and wellness activities. It utilizes AI agents (Agent Clona for patient support and Assistant Lysa for doctor assistance) to provide insights, streamline healthcare operations, and act as a comprehensive wellness monitoring and change detection system. The platform aims to enhance patient care through advanced AI and robust health data management.

## User Preferences
- **Preferred communication style**: Simple, everyday language
- **Backend Language**: Python only - ALL backend code must be written in Python (FastAPI)
- **Frontend**: React/TypeScript (standard web practice)
- **Development Philosophy**: Building FULL FUNCTIONAL applications, NOT MVPs. All features must be completely functional and production-ready before delivery.

## System Architecture

### Frontend
The frontend is built with React, TypeScript, Vite, Wouter for routing, TanStack Query for data fetching, and Tailwind CSS for styling. It uses Radix UI and shadcn/ui for a clinical aesthetic, supporting role-based routing and context-based theming. It routes Python AI endpoints to the FastAPI backend on port 8000 and other endpoints to an Express server on port 5000.

### Backend
The backend consists of two services:
- **Node.js Express Backend (Port 5000)**: Manages the Agent Clona chatbot, appointments, calendar, consultations, pain tracking, symptom journal, voice analysis, baseline calculation, deviation detection, and risk scoring.
- **Python FastAPI Backend (Port 8000)**: Handles all AI deterioration detection endpoints, guided video/audio examinations, mental health questionnaires, database interactions, and core authentication. It uses an async AI engine initialization with an `AIEngineManager` singleton and loads heavy ML models (TensorFlow, MediaPipe, YAMNet) at startup.

### Core Features & Technical Implementations
- **AI Integration:** Leverages OpenAI API with a tiered model strategy:
  - **GPT-4o:** PHI detection, symptom extraction, entity categorization, SOAP notes, email categorization
  - **o1 (Advanced Clinical Reasoning):** Differential diagnosis, complex drug interactions, clinical decision support (requires BAA/Enterprise)
  - All AI operations have graceful fallback (o1 → GPT-4o) and comprehensive audit logging
- **PHI Detection Service (Python):** OpenAI GPT-4o-based HIPAA-compliant PHI detection replacing AWS Comprehend Medical:
  - Comprehensive detection of HIPAA's 18 PHI identifiers
  - Medical entity extraction with ICD-10-CM, RxNorm, SNOMED-CT inference
  - Regex-based fallback when API unavailable
  - Legacy category mapping for TypeScript compatibility
- **Real-Time Monitoring:** AI-powered digital biomarker tracking from wearable data.
- **Voice-Based Followups:** Uses OpenAI Whisper for transcription and GPT-4 for analysis.
- **Assistant Lysa:** AI-powered appointment management, email categorization, call log transcription, and automated reminders.
- **Secure Data Sharing:** Patient record sharing with consent management and audit logging.
- **EHR & Wearable Integration:** FHIR-based integration with EHRs and wearable devices.
- **Video Consultations:** HIPAA-compliant video conferencing.
- **Home Clinical Exam Coach (HCEC):** AI-powered guided self-examination using OpenAI Vision, extracting metrics like respiratory rate, skin pallor, and nail bed analysis.
- **Deterioration Prediction System:** Comprehensive health change detection with baseline calculation, Z-score, anomaly detection, Bayesian risk modeling, and time-series trend analysis for a composite risk score.
- **ML Inference Infrastructure:** Self-hosted system with model registry, Redis caching, async inference, HIPAA-compliant audit logging, and ONNX optimization.
- **Guided Video Examination:** 4-stage workflow (Eyes, Palm, Tongue, Lips) with clinical-grade LAB color analysis, disease-specific personalization, and S3 encrypted storage. Includes Facial Puffiness Score (FPS) system with MediaPipe Face Mesh and DeepLab V3+ Edema Segmentation.
- **Guided Audio Examination:** 4-stage workflow (Breathing, Coughing, Speaking, Reading) with YAMNet ML classification, neurological metrics, and S3 encrypted storage.
- **Trend Prediction Engine:** Calculates baselines, performs Z-score analysis, anomaly detection, Bayesian risk modeling, and time-series trend analysis to generate a composite risk score.
- **Alert Orchestration Engine:** Multi-channel delivery (dashboard, email, SMS) with rule-based systems.
- **Behavior AI Analysis System:** Multi-modal deterioration detection via behavioral patterns, digital biomarkers, cognitive testing, and sentiment analysis using ensemble ML models for risk scoring. Includes Gait Analysis (MediaPipe Pose) and Accelerometer Tremor Analysis.
- **Risk Scoring Dashboard:** Composite risk score (0-15 scale) with weighted factors (respiratory, pain, symptoms) and 7-day history.
- **Baseline Calculation UI:** 7-day rolling window statistics with quality badges, recalculate functionality, and history visualization.
- **Google Calendar Sync:** Bidirectional appointment sync for doctors with OAuth, conflict resolution, and HIPAA-compliant PHI handling.
- **Drug-Drug Interaction Detection:** Medication adherence system with RxNorm integration and real-time interaction detection.
- **Automatic Drug Normalization:** On-demand medication normalization against RxNorm API, creating standardized drug records.
- **Clinical Automation (Rx Builder):** AI-assisted prescription system with:
  - AI SOAP notes generation and ICD-10 code suggestions
  - AI differential diagnosis based on patient history
  - Drug-drug interaction checks and contraindication alerts
  - Auto-dosage recommendations based on patient parameters
  - Chronic refill automation with adherence threshold (0-100%, default 80%) and days-before-expiry triggers (1-90 days, default 7)
  - Prescription templates with doctor-specific customization
  - Schema-level validation with safe defaults for chronic refill settings
- **Clinical Assessment (Diagnosis Helper):** AI-assisted clinical assessment system with:
  - Dual authorization flow (doctor_patient_assignments primary, PatientSharingLink fallback)
  - Patient data aggregation (medical files, health alerts, ML predictions, medications, daily follow-ups)
  - Access scope enforcement (full/limited/emergency_only) restricting sensitive data appropriately
  - AI-powered differential diagnosis with GPT-4o based on comprehensive patient data
  - HIPAA-compliant audit logging supporting both assignment and sharing link contexts
  - Consent validation with proper access level permissions for each data category
- **PainTrack Platform:** Chronic pain tracking system with dual-camera video capture, VAS pain slider, and medication tracking.
- **Mental Health AI Dashboard:** Integrated questionnaires (PHQ-9, GAD-7, PSS-10) with AI-powered GPT-4o analysis, crisis detection, and scoring.
- **Agent Clona Symptom Extraction:** AI-powered symptom extraction from patient conversations (GPT-4o) identifying symptoms, body locations, intensity, and temporal information.
- **AI-Powered Habit Tracker (13 Features):** Comprehensive habit management including creation, daily routines, streaks, smart reminders, AI coaching, trigger detection, addiction-mode quit plans, mood tracking, dynamic AI recommendations, social accountability, guided CBT sessions, gamification, and smart journals with AI insights.
- **Daily Follow-up Dashboard Pattern:** Enforces a 24-hour gating for data display across tabs (Device Data, Symptoms, Video AI, Audio AI, PainTrack, Mental Health), prompting completion if no data for today, while allowing additional entries.
- **Doctor-Patient Assignment System:** Explicit doctor-patient relationships with authorization via `doctor_patient_assignments` table, auto-assignment, consent tracking, access levels, revocation, and HIPAA audit logging. All patient data endpoints verify active assignment.
- **Per-Doctor Personal Integrations:** Each doctor can connect their own Gmail (OAuth), WhatsApp Business API, and Twilio VoIP phone accounts for automated communication sync and logging. Features JWT-signed OAuth state tokens for CSRF protection and AES-256-GCM encrypted credential storage for HIPAA compliance. Legacy plaintext tokens are automatically re-encrypted on next update.

### Multi-Agent Communication System
The platform features a sophisticated multi-agent system enabling real-time communication between AI agents, users, and healthcare providers:

**Architecture:**
- **Agent Clona**: Patient-facing AI health companion for daily check-ins, symptom tracking, medication reminders, and wellness support
- **Assistant Lysa**: Doctor-facing AI assistant for patient management, scheduling, clinical summaries, and health alert triage
- **Message Router**: Central routing service handling user↔agent, agent↔agent, and agent↔tool communications
- **Memory Service**: Dual-layer memory (Redis short-term + PostgreSQL pgvector long-term) with TTL-based expiration

**Message Protocol (MessageEnvelope):**
```python
{
  msg_id: str,           # Unique message identifier
  sender: {type, id},    # ActorType (agent/user/system) + ID
  to: [{type, id}],      # Recipients list
  type: MessageType,     # chat/command/event/tool_call/ack
  timestamp: datetime,
  payload: dict          # Message-specific content
}
```

**Agent Hub UI Features:**
- Unified conversation interface for both patients and doctors
- Real-time WebSocket connection with presence indicators
- Delivery checkmarks (sent/delivered/read)
- Tool call status indicators (pending/running/completed/failed)
- Human-in-the-loop approval dialogs for sensitive operations
- HIPAA-compliant audit logging of all interactions

**Database Schema (agent tables):**
- `agents`: Agent configurations (clona, lysa) with personas and tool registries
- `agent_messages`: Message storage with delivery tracking
- `agent_conversations`: Conversation management with participant tracking
- `agent_tasks`: Background task queue for scheduled operations
- `agent_tools`: Tool registry with permissions and versioning
- `agent_memory`: Long-term memory with vector embeddings (pgvector)
- `agent_audit_logs`: HIPAA-compliant comprehensive audit trail

**Files:**
- Backend: `app/models/agent_models.py`, `app/services/agent_engine.py`, `app/services/message_router.py`, `app/services/memory_service.py`
- Routes: `app/routers/agent_api.py`, `app/routers/agent_websocket.py`
- Frontend: `client/src/pages/AgentHub.tsx`
- Schema: `shared/schema.ts` (agent tables)

### Security and Compliance
The platform is HIPAA-compliant, utilizing AWS Cognito for authentication, BAA verification for integrations, comprehensive audit logging, end-to-end encryption for video, strict PHI handling, and explicit doctor-patient assignment authorization. Positioned as a General Wellness Product.

## External Dependencies

- **Authentication:** AWS Cognito
- **Database:** Neon serverless PostgreSQL
- **AI Services:** OpenAI API (GPT models), TensorFlow.js, PyTorch, HuggingFace Transformers, ONNX Runtime
- **Caching:** Redis
- **Communication:** Twilio API, AWS SES
- **Video Conferencing:** Daily.co
- **Cloud Services:** AWS S3, AWS Textract, AWS Comprehend Medical, AWS HealthLake, AWS HealthImaging, AWS HealthOmics
- **Data Integration APIs:** PubMed E-utilities, PhysioNet WFDB, Kaggle API, WHO Global Health Observatory API, OpenWeatherMap API, Biobot Analytics