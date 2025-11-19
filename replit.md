# Followup AI - HIPAA-Compliant Health Platform

## Overview
Followup AI is a HIPAA-compliant health monitoring platform for immunocompromised patients. It offers personalized health tracking, medication management, and wellness activities, leveraging AI agents (Agent Clona for patient support and Assistant Lysa for doctor assistance) to provide insights and streamline healthcare operations. The platform aims to enhance patient care through advanced AI and comprehensive health data management, positioning itself as a wellness monitoring and change detection platform to avoid medical device classifications.

## User Preferences
- **Preferred communication style**: Simple, everyday language
- **Backend Language**: Python only - ALL backend code must be written in Python (FastAPI)
- **Frontend**: React/TypeScript (standard web practice)
- **Development Philosophy**: Building FULL FUNCTIONAL applications, NOT MVPs. All features must be completely functional and production-ready before delivery.

## System Architecture

### Frontend
The frontend is built with React, TypeScript, Vite, Wouter for routing, TanStack Query for data fetching, and Tailwind CSS for styling. It uses Radix UI and shadcn/ui for a clinical aesthetic, supporting role-based routing and context-based theming.

**AI Dashboard Pages** (routes at `/ai-video`, `/ai-audio`, `/ai-alerts`):
- **AIVideoDashboard**: Upload videos for analysis, view respiratory rate/skin changes/facial metrics, session history ⚠️ TEMPORARILY DISABLED
- **AIAudioDashboard**: Upload audio for analysis, view breath cycles/speech pace/cough detection, session history ⚠️ TEMPORARILY DISABLED
- **AIAlertsDashboard**: Manage health change alerts, configure notification rules, review alert history ⚠️ TEMPORARILY DISABLED

**Backend Routing**: The queryClient is configured to automatically route Python AI endpoints (`/api/v1/video-ai/*`, `/api/v1/audio-ai/*`, `/api/v1/trends/*`, `/api/v1/alerts/*`) to the FastAPI backend on port 8000, while all other endpoints go to the Express server on port 5000.

### Backend

**Current Status (November 19, 2025 - FULLY OPERATIONAL):**
- ✅ **Node.js Express Backend (Port 5000)**: FULLY OPERATIONAL with 105 routes
  - Agent Clona chatbot (GPT-4o powered patient support)
  - Appointments, Calendar, Consultations
  - Pain tracking, Symptom journal, Voice analysis
  - Baseline calculation, Deviation detection, Risk scoring
  
- ✅ **Python FastAPI Backend (Port 8000)**: FULLY OPERATIONAL with 120 routes
  - ✅ All 52 AI deterioration detection endpoints ENABLED
  - ✅ Guided video examination (5 endpoints) ENABLED
  - ✅ Database tables, authentication, core features work
  - ✅ Zero blocking imports - app starts successfully
  - ✅ Comprehensive smoke test validates router registration

**Python Backend Architecture (Production-Ready November 2025):**
- Implemented async AI engine initialization with FastAPI lifespan events
- Created `AIEngineManager` singleton pattern with thread pool executor
- AI engines (Video, Audio, Trend, Alert) initialize asynchronously during startup
- Manual dependency resolution - endpoints directly call `AIEngineManager.get_*_engine()`
- Defensive runtime checks prevent crashes when engines unavailable
- Enhanced error logging with granular status per engine
- **Resolution:** Removed FastAPI `Depends()` pattern, using direct AIEngineManager calls

**Starting the Backend Servers:**
```bash
# Option 1: Node.js only (Agent Clona chatbot - WORKS)
npm run dev

# Option 2: Both servers (Full AI deterioration detection - WORKS)
bash start-all-services.sh

# Option 3: Manual debugging
Terminal 1: npm run dev
Terminal 2: python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

**Performance Notes:**
- First Python startup: 30-60 seconds (downloads ML models: MediaPipe, TensorFlow)
- Subsequent startups: 5-10 seconds (models cached locally)

**Comprehensive Documentation:**
- `AI_API_DOCUMENTATION.md` - Complete endpoint reference (52 AI endpoints)
- `AGENT_CLONA_DOCUMENTATION.md` - Working chatbot feature documentation
- `PYTHON_BACKEND_DEBUG_REPORT.md` - Complete debugging report with final resolution
- `test_startup_smoke.py` - Comprehensive smoke test (120 routes validated)

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
-   **Guided Video Examination System (Production-Ready):** A HIPAA-compliant 4-stage self-examination workflow with clinical-grade hepatic and anemia color analysis:
    -   **4-Stage Workflow**: Eyes (sclera jaundice detection), Palm (conjunctival pallor for anemia), Tongue (coating/color), Lips (cyanosis/hydration)
    -   **30-Second Prep Screens**: Detailed instructions with countdown timers for each examination stage
    -   **Clinical-Grade LAB Color Analysis**: 31 new database fields capturing hepatic and anemia color metrics using perceptually-uniform LAB color space
    -   **Disease-Specific Personalization**: Extended ConditionPersonalizationService with 3 new methods:
        - `get_guided_exam_config()`: Returns comprehensive examination configuration based on patient conditions
        - `get_hepatic_monitoring_config()`: Personalized jaundice thresholds for liver disease patients (critical priority with sensitive b* channel thresholds: mild 25.0, moderate 35.0, severe 45.0)
        - `get_anemia_monitoring_config()`: Personalized pallor thresholds for anemia patients (critical priority with palmar perfusion thresholds: mild 40.0, moderate 30.0, severe 20.0)
    -   **5 RESTful API Endpoints**: POST /sessions (create), POST /capture (frame upload), POST /complete (trigger ML analysis), GET /sessions/{id} (details), GET /results (31 clinical metrics)
    -   **S3 Encrypted Storage**: All captured frames stored with server-side AES-256 encryption
    -   **React Frontend**: Complete UI at `/guided-exam` route with camera capture, stage progression, countdown timers, and results display
    -   **Database Models**: VideoExamSession (stage tracking, quality scores, S3 URIs) + 31 new VideoMetrics fields for hepatic/anemia analysis
    -   **Documentation**: GUIDED_VIDEO_EXAMINATION_DOCUMENTATION.md
-   **AI Deterioration Detection System (Production-Ready):** A full-stack SaaS platform featuring:
    -   **Video AI Engine:** Extracts metrics like respiratory rate, skin pallor, sclera yellowness, facial swelling, head tremor, nail bed analysis (anaemia, nicotine stains, burns, abnormalities).
    -   **Facial Puffiness Score (FPS) System:** Comprehensive facial contour tracking using MediaPipe Face Mesh 468 landmarks:
        - **5 Regional Scores**: Periorbital (30% weight - critical for thyroid/kidney), Cheeks (30%), Jawline (20%), Forehead (10%), Overall Contour (10%)
        - **Baseline Comparison**: % expansion from patient's personalized baseline
        - **Composite FPS**: Weighted average (0-100+ scale) indicating overall facial puffiness
        - **Asymmetry Detection**: Tracks left/right differences for lymphedema/allergic reactions
        - **Disease-Specific Integration**: Personalized thresholds for thyroid disorder (periorbital focus), kidney disease (facial + peripheral), heart failure (generalized), allergic reactions (rapid swelling alerts)
        - **Temporal Pattern Detection**: Morning vs evening puffiness patterns, acute vs chronic changes
        - **Documentation**: FACIAL_PUFFINESS_SCORE_DOCUMENTATION.md
    -   **Comprehensive Respiratory Metrics System:** 
        - **Respiratory Variability Index (RVI)**: Measures breathing stability (coefficient of variation over 1-5 min)
        - **Baseline Tracking**: Auto-calculated patient baseline RR with exponential moving average updates
        - **Temporal Analytics**: Rolling 24-hour average, 3-day trend slope (linear regression)
        - **Anomaly Detection**: Z-score computation vs patient baseline (|Z| > 2 = significant anomaly)
        - **Advanced Pattern Detection**: Accessory muscle scoring, gasping detection, chest shape asymmetry, thoracoabdominal synchrony
        - **Database Models**: RespiratoryBaseline (patient baselines) and RespiratoryMetric (time-series metrics)
        - **Service Architecture**: respiratory_metrics_service.py handles all temporal analytics and database persistence
    -   **Disease-Specific Personalization System:** Multi-domain condition profiles covering 12 chronic conditions with integrated respiratory + edema monitoring emphasis:
        - **Respiratory Conditions (8)**: asthma, COPD, heart failure, pulmonary embolism, pneumonia, pulmonary TB, bronchiectasis, allergic reactions
        - **Edema-Focused Conditions (4)**: kidney disease, liver disease, thyroid disorders, lymphedema
        - **Edema Emphasis Profiles**: Each condition specifies priority (critical/high/medium/low), expected pattern (bilateral/unilateral/facial), focus locations, personalized PEI thresholds, pitting grade watchpoints
        - **Critical Prioritization**: Heart failure → bilateral leg edema monitoring; allergic reactions → urgent facial swelling alerts; lymphedema → asymmetry detection (>20% threshold)
        - **Multi-Condition Merging**: Intelligent profile combination for patients with multiple conditions (takes highest priority, most sensitive thresholds, combined focus locations)
        - **Service Methods**: get_edema_config(), get_edema_examination_focus() for personalized examination workflows
        - **Documentation**: DISEASE_SPECIFIC_EDEMA_MONITORING.md
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