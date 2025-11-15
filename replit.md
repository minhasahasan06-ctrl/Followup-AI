# Followup AI - HIPAA-Compliant Health Platform

## Overview
Followup AI is a HIPAA-compliant health monitoring platform designed for immunocompromised patients, offering personalized health tracking, medication management, and wellness activities. It features two AI agents: Agent Clona for patient support and Assistant Lysa for doctor assistance, providing AI-powered insights and streamlining healthcare operations. The platform aims to enhance patient care through advanced AI integration and comprehensive health data management.

## User Preferences
- **Preferred communication style**: Simple, everyday language
- **Backend Language**: Python only - ALL backend code must be written in Python (FastAPI)
- **Frontend**: React/TypeScript (standard web practice)

## System Architecture

### Frontend
The frontend is built with React, TypeScript, Vite, Wouter for routing, TanStack Query for data fetching, and Tailwind CSS for styling. It leverages Radix UI and shadcn/ui for a clinical yet calming aesthetic, supporting role-based routing and context-based theming.

### Backend
The backend has two implementations: a new Python FastAPI implementation and a legacy JavaScript Express.js implementation. The Python FastAPI backend is the preferred and actively developed version, using SQLAlchemy ORM for PostgreSQL, AWS Cognito for authentication and authorization (JWTs with role-based access control), and Pydantic for validation.

**Core Features & Technical Implementations:**

-   **AI Integration:** Utilizes OpenAI API (GPT-4o) for Agent Clona (symptom analysis, differential diagnosis, treatment suggestions) and Assistant Lysa (doctor assistance, sentiment analysis, medical entity extraction, session summaries).
-   **Personalization & Recommendation System:** A rule-based intelligent recommendation engine, enhanced with RAG, for tailored patient suggestions and clinical insights for doctors.
-   **Advanced Drug Interaction Detection:** AI-powered system using GNN and NLP (via OpenAI) to detect drug-drug and drug-gene interactions, including medication enrichment and severity classification.
-   **Real-Time Immune Function Monitoring:** AI-powered digital biomarker tracking integrating wearable data (Fitbit, Apple Health) to create "immune digital twins" and predict infection risk.
-   **Environmental Pathogen Risk Mapping:** Provides location-based real-time environmental health monitoring (air quality, outbreak tracking, wastewater surveillance, pollen, UV index) with AI-generated safety recommendations.
-   **Voice-Based Daily Followups:** Uses OpenAI Whisper for transcription and GPT-4 for analysis of voice recordings to extract health data.
-   **Assistant Lysa Receptionist Features:** AI-powered receptionist managing appointments (CRUD, conflict detection), doctor availability, email threads (categorization, PHI redaction), call logs (transcription), and automated reminders. Includes Google Calendar and Gmail integration.
-   **Secure Patient Record Sharing:** System for sharing patient records with consent management and audit logging.
-   **Health Insight Consent:** Granular control for patients over data sharing with third-party apps.
-   **EHR & Wearable Integration:** FHIR-based integration with major EHR systems (Epic, Cerner) and popular wearable devices (Amazfit, Garmin).
-   **Video Consultations:** HIPAA-compliant video conferencing via Daily.co for secure patient-doctor interactions.

### Security and Compliance
The platform is designed to be HIPAA-compliant, featuring:
- AWS Cognito for robust authentication and role-based access control.
- BAA verification checks for all OpenAI and Daily.co integrations.
- Comprehensive audit logging for PHI access.
- End-to-end encryption for video consultations.
- PHI handling compliance in all services.

## Recent Changes

### Pain Detection Camera System (November 2025)
Complete AI-powered facial analysis system for tracking pain progression:
- **Database**: 3 new tables (PainMeasurement, PainQuestionnaire, PainTrendSummary)
- **Backend**: 9 Python FastAPI endpoints with HIPAA-compliant security
- **Frontend**: TensorFlow.js MediaPipe FaceMesh for real-time facial landmark detection
- **Features**: 10-second daily recordings, pain scoring algorithm, comprehensive questionnaire, trend analysis
- **Security**: Full patient data ownership verification, doctor-patient connection validation
- **Testing Note**: End-to-end testing has known limitation with TensorFlow.js in headless browsers (WebGL requirement). Feature works correctly in real browsers.

### Home Clinical Exam Coach (HCEC) System (November 2025)
**Complete End-to-End Implementation**
AI-powered guided self-examination system teaching patients standardized exam techniques before doctor consultations:

**Backend (Python FastAPI - app/routers/exam_coach.py):**
- 7 REST API endpoints with HIPAA-compliant security:
  - POST /start-session: Initialize exam session with protocol selection
  - POST /analyze-frame: Real-time AI frame analysis (OpenAI Vision API)
  - POST /complete-step: Upload captured images/videos with quality scores
  - GET /session/{id}: Retrieve session details with all steps
  - GET /packets: List patient's exam packets
  - GET /protocols: Get available exam protocols
- **Database Models** (app/models/exam_coach.py):
  - ExamSession: Track coaching sessions
  - ExamStep: Individual examination steps
  - CoachingFeedback: AI-generated real-time feedback
  - ExamPacket: Compiled exam data for doctor review
  - ExamProtocol: Standardized examination templates
- **Built-in Protocols**: Skin inspection, throat examination, leg edema, range of motion, respiratory effort
- **AI Services**: OpenAIService class for real-time coaching feedback (lighting, angle, distance, visibility)
- **Security**: Patient-only access with ownership verification on ALL database queries

**Frontend (client/src/pages/ExamCoach.tsx):**
- Complete 3-tab coaching interface:
  - Tab 1: Select Exam - Choose from available protocols
  - Tab 2: Coaching - Real-time camera with AI overlay feedback
  - Tab 3: Complete - Review and finalize exam packet
- Real-time AI analysis every 3 seconds with visual coaching overlays
- Voice guidance using Web Speech API for hands-free operation
- Step-by-step wizard with progress tracking
- HIPAA-compliant camera cleanup on component unmount
- Complete data-testid coverage for testing
- Integrated into App.tsx routing (/exam-coach) and sidebar navigation

**Security Review Status**: ✅ PASSED - All HIPAA violations resolved, defense-in-depth patient ownership verification implemented

### Symptom Journal Enhancements (November 2025)
**Backend Services Added (Python FastAPI):**

1. **Respiratory Rate Analysis** (app/services/respiratory_analysis.py):
   - AI-powered chest movement analysis using OpenAI Vision
   - Breaths per minute calculation from video clips
   - POST /api/v1/symptom-journal/analyze-respiratory endpoint

2. **Comparison View** (app/routers/symptom_journal.py):
   - GET /api/v1/symptom-journal/compare endpoint
   - Side-by-side measurement comparison with change percentages
   - Defense-in-depth security: patient ownership verification on measurements AND images
   - Color change tracking, brightness analysis, area change calculations

3. **Weekly PDF Reports** (app/services/pdf_service.py):
   - POST /api/v1/symptom-journal/generate-weekly-pdf endpoint
   - Structured timeline with images, measurements, and trends
   - Charts for symptom progression over time
   - Formatted for doctor review

**Security Review Status**: ✅ PASSED - All endpoints follow HIPAA compliance patterns

### Medication Side-Effect Predictor (November 2025)
**Complete Full-Stack Implementation**
AI-powered system correlating patient symptoms with medication timelines to detect potential side effects:

**Backend (Python FastAPI):**
- **Database Models** (app/models/medication_side_effects.py):
  - MedicationTimeline: Track medications (name, dosage, start/stop dates, prescriber)
  - DosageChange: Record dosage modifications with timestamps
  - SymptomLog: Multi-source symptom capture (manual, daily followup, Agent Clona chat)
  - SideEffectCorrelation: AI-analyzed medication-symptom patterns
  - CorrelationAnalysis: Batch analysis sessions with metadata
- **Medication Timeline API** (app/routers/medication_timeline.py):
  - 6 REST endpoints: CRUD operations, dosage tracking, active medication queries
  - Role-based access: Patients (self-only), Doctors (with active patient connection)
- **Symptom Logging API** (app/routers/symptom_logging.py):
  - 6 REST endpoints: Create, query, update, delete symptoms
  - Multi-source tracking: manual entries, voice followups, Agent Clona extractions
  - Defense-in-depth security: patient existence check → ownership/connection validation
- **AI Correlation Engine** (app/services/medication_correlation.py):
  - OpenAI GPT-4o analysis of medication-symptom temporal patterns
  - Time-to-onset calculation, likelihood scoring (STRONG/LIKELY/POSSIBLE/UNLIKELY)
  - Patient impact classification (severe/moderate/mild)
  - AI-generated recommendations and reasoning
- **Side-Effect Analysis API** (app/routers/medication_side_effects.py):
  - POST /analyze/me: Trigger new AI-powered correlation analysis
  - GET /correlations/me: Retrieve all correlations for patient
  - GET /summary/me: Get aggregated effects summary by medication
  - GET /correlation/{id}: Detailed view of specific correlation

**Frontend (client/src/pages/MedicationEffects.tsx):**
- **Three-Tab Interface:**
  - Overview: Medication cards with correlation counts and previews
  - By Medication: Split view (medication list + detailed correlations)
  - Recommendations: AI-generated action items with structured cards
- **Configuration Dialog:**
  - Analysis period slider (7-365 days)
  - Minimum confidence threshold slider (0-100%)
  - "Apply & Analyze" triggers custom analysis
- **Correlation Details:**
  - Confidence score percentage with color-coded badges
  - Time to onset (hours after medication)
  - Patient impact classification (severe/moderate/mild)
  - Temporal pattern descriptions
  - AI reasoning callouts with Sparkles icon
  - Recommended actions
- **Summary Statistics:**
  - Total medications analyzed
  - Total correlations found
  - Strong correlations count (urgent attention)
  - Analysis period (days)
- **Design Compliance:**
  - Medical Teal primary color (180 45% 45%)
  - Status colors: Rose (critical), Coral (warning), Teal (success)
  - Proper spacing, typography hierarchy, loading states
  - Accessible with data-testid coverage
- **TanStack Query Integration:**
  - Parameterized query keys for flexible caching
  - Cache invalidation with exact: false for fresh data
  - Mutation feedback with toast notifications

**Security Review Status**: ✅ PASSED - Defense-in-depth authorization, patient enumeration prevention, proper role-based access control

**Regulatory Compliance**: Uses "change detection/monitoring" language (NOT diagnosis) for regulatory compliance

**Architect Review Status**: ✅ PRODUCTION-READY - All critical issues resolved (tab navigation, configuration UI, structured recommendations, cache invalidation)

### Running the Python Backend
The new features (HCEC, Symptom Journal enhancements) require the Python FastAPI backend on port 8000:

```bash
# From workspace root:
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

**Note**: Both backends must run simultaneously:
- JavaScript/Express on port 5000 (legacy - current workflow)
- Python/FastAPI on port 8000 (new features - manual start required)

## External Dependencies

-   **Authentication:** AWS Cognito.
-   **Database:** Neon serverless PostgreSQL.
-   **AI Services:** OpenAI API (GPT models).
-   **Machine Learning:** TensorFlow.js (@tensorflow-models/face-landmarks-detection, @mediapipe/face_mesh) for facial analysis.
-   **Communication:** Twilio API (SMS, voice), AWS SES (transactional emails).
-   **Video Conferencing:** Daily.co.
-   **Cloud Services:** AWS S3, AWS Textract, AWS Comprehend Medical, AWS HealthLake, AWS HealthImaging, AWS HealthOmics.
-   **Data Integration APIs:** PubMed E-utilities, PhysioNet WFDB, Kaggle API, WHO Global Health Observatory API, OpenWeatherMap API, Biobot Analytics.