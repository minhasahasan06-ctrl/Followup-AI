# Followup AI - HIPAA-Compliant Health Platform

## Overview
Followup AI is a HIPAA-compliant health monitoring platform for immunocompromised patients. It offers personalized health tracking, medication management, and wellness activities, leveraging AI agents (Agent Clona for patient support and Assistant Lysa for doctor assistance) to provide insights and streamline healthcare operations. The platform aims to enhance patient care through advanced AI and comprehensive health data management, positioning itself as a wellness monitoring and change detection platform to avoid medical device classifications.

## User Preferences
- **Preferred communication style**: Simple, everyday language
- **Backend Language**: Python only - ALL backend code must be written in Python (FastAPI)
- **Frontend**: React/TypeScript (standard web practice)

## System Architecture

### Frontend
The frontend is built with React, TypeScript, Vite, Wouter for routing, TanStack Query for data fetching, and Tailwind CSS for styling. It uses Radix UI and shadcn/ui for a clinical aesthetic, supporting role-based routing and context-based theming.

**AI Dashboard Pages** (routes at `/ai-video`, `/ai-audio`, `/ai-alerts`):
- **AIVideoDashboard**: Upload videos for analysis, view respiratory rate/skin changes/facial metrics, session history
- **AIAudioDashboard**: Upload audio for analysis, view breath cycles/speech pace/cough detection, session history
- **AIAlertsDashboard**: Manage health change alerts, configure notification rules, review alert history

**Backend Routing**: The queryClient is configured to automatically route Python AI endpoints (`/api/v1/video-ai/*`, `/api/v1/audio-ai/*`, `/api/v1/trends/*`, `/api/v1/alerts/*`) to the FastAPI backend on port 8000, while all other endpoints go to the Express server on port 5000.

### Backend
The primary backend is a Python FastAPI application (port 8000), utilizing SQLAlchemy ORM for PostgreSQL, AWS Cognito for authentication (JWTs with role-based access control), and Pydantic for validation. A Node.js Express server (port 5000) handles legacy endpoints and serves the frontend.

**Python Backend (Port 8000)**: 52 AI endpoints across 4 engines (Video AI, Audio AI, Trend Prediction, Alert Orchestration) + Guided Video Examination System. 

**IMPORTANT - Starting Both Servers:**
The platform requires BOTH servers running simultaneously:
1. Node.js Express (port 5000) - Auto-starts with default workflow
2. Python FastAPI (port 8000) - Must be started manually

**Option A - Use the provided startup script:**
```bash
bash start-all-services.sh
```

**Option B - Manual start in separate terminals:**
Terminal 1: `npm run dev` (Node.js)
Terminal 2: `python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload` (Python)

**Option C - Single command (background mode):**
```bash
npm run dev & python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload & wait
```

The Python backend is required for the Guided Video Examination feature at `/ai-video`

**Comprehensive API Documentation**: See `AI_API_DOCUMENTATION.md` for complete endpoint reference with curl examples, authentication, request/response formats, and testing workflows.

### Core Features & Technical Implementations
-   **AI Integration:** Leverages OpenAI API (GPT-4o) for symptom analysis, wellness suggestions, doctor assistance, sentiment analysis, and medical entity extraction.
-   **Personalization & Recommendation System:** A rule-based intelligent recommendation engine, enhanced with RAG, for tailored suggestions.
-   **Advanced Drug Interaction Detection:** AI-powered system using GNN and NLP.
-   **Real-Time Immune Function Monitoring:** AI-powered digital biomarker tracking from wearable data.
-   **Environmental Pathogen Risk Mapping:** Location-based real-time environmental health monitoring.
-   **Voice-Based Daily Followups:** Uses OpenAI Whisper for transcription and GPT-4 for analysis.
-   **Assistant Lysa Receptionist Features:** AI-powered appointment management, email categorization, call log transcription, and automated reminders with Google Calendar and Gmail integration.
-   **Secure Patient Record Sharing:** System with consent management and audit logging.
-   **EHR & Wearable Integration:** FHIR-based integration with major EHR systems and popular wearable devices.
-   **Video Consultations:** HIPAA-compliant video conferencing via Daily.co.
-   **Pain Detection Camera System:** AI-powered facial analysis for tracking discomfort progression.
-   **Home Clinical Exam Coach (HCEC):** AI-powered guided self-examination using OpenAI Vision.
-   **Symptom Journal Enhancements:** Includes AI-powered respiratory rate analysis and weekly PDF reports.
-   **Medication Side-Effect Predictor:** AI-powered system correlating patient symptoms with medication timelines.
-   **Deterioration Prediction System:** Comprehensive health change detection with baseline calculation, deviation detection (z-score analysis), and risk scoring (0-15 scale).
-   **ML Inference Infrastructure:** Self-hosted machine learning inference system with a model registry, Redis caching, async thread pool inference, HIPAA-compliant audit logging, batch processing, and ONNX optimization. Includes pre-trained Clinical-BERT and custom LSTM models.
-   **AI Deterioration Detection System (Production-Ready):** A full-stack SaaS platform featuring:
    -   **Video AI Engine:** Extracts metrics like respiratory rate, skin pallor, sclera yellowness, facial swelling, and head tremor.
    -   **Audio AI Engine:** Extracts metrics like breath cycles, speech pace, cough detection, wheeze detection, and voice quality.
    -   **Trend Prediction Engine:** Performs baseline calculation, Z-score analysis, anomaly detection, Bayesian risk modeling, and time-series trend analysis to generate a composite risk score.
    -   **Alert Orchestration Engine:** Provides multi-channel delivery (dashboard, email, SMS) with rule-based systems and HIPAA compliance.

### Security and Compliance
The platform is HIPAA-compliant, featuring AWS Cognito for authentication, BAA verification for all integrations, comprehensive audit logging, end-to-end encryption for video consultations, and strict PHI handling. It is strategically positioned as a General Wellness Product to avoid medical device classification.

## External Dependencies

-   **Authentication:** AWS Cognito.
-   **Database:** Neon serverless PostgreSQL.
-   **AI Services:** OpenAI API (GPT models).
-   **Machine Learning:** TensorFlow.js, PyTorch, HuggingFace Transformers, ONNX Runtime, Redis.
-   **Communication:** Twilio API, AWS SES.
-   **Video Conferencing:** Daily.co.
-   **Cloud Services:** AWS S3, AWS Textract, AWS Comprehend Medical, AWS HealthLake, AWS HealthImaging, AWS HealthOmics.
-   **Data Integration APIs:** PubMed E-utilities, PhysioNet WFDB, Kaggle API, WHO Global Health Observatory API, OpenWeatherMap API, Biobot Analytics.