# Followup AI - HIPAA-Compliant Health Platform

## Overview
Followup AI is a HIPAA-compliant health monitoring platform for chronic care patients, offering personalized health tracking, medication management, and wellness activities. It leverages AI agents (Agent Clona for patient support, Assistant Lysa for doctor assistance) to enhance patient care through advanced AI and robust health data management, aiming to be a comprehensive wellness monitoring and change detection system. The platform's vision is to transform healthcare for individuals managing ongoing health conditions by delivering fully functional, production-ready applications.

## User Preferences
- **Preferred communication style**: Simple, everyday language
- **Backend Language**: Python only - ALL backend code must be written in Python (FastAPI)
- **Frontend**: React/TypeScript (standard web practice)
- **Development Philosophy**: Building FULL FUNCTIONAL applications, NOT MVPs. All features must be completely functional and production-ready before delivery.
- **Code Quality**: Avoid duplicate code - use proper code reuse, modular design, and shared components/utilities.

## System Architecture

### Frontend
The frontend uses React, TypeScript, Vite, Wouter for routing, TanStack Query for data fetching, and Tailwind CSS for styling. It incorporates Radix UI and shadcn/ui for a clinical aesthetic, supporting role-based routing and context-based theming. It routes Python AI endpoints to the FastAPI backend (port 8000) and other endpoints to an Express server (port 5000).

### Backend
The backend consists of two main components:
- **Node.js Express server (Port 5000)**: Handles chatbot, appointments, calendar, consultations, pain tracking, symptom journaling, and risk scoring.
- **Python FastAPI server (Port 8000)**: Manages AI deterioration detection, guided examinations, mental health questionnaires, database interactions, authentication, and an async AI engine with a singleton manager for ML models. In development, the Express server automatically spawns and monitors the FastAPI process.

### Core Features & Technical Implementations
- **AI Integration**: Utilizes OpenAI API (GPT-4o, o1) for PHI detection, symptom extraction, and clinical reasoning, with graceful fallbacks.
- **PHI Detection Service**: GPT-4o-based HIPAA-compliant PHI detection and medical entity extraction.
- **Real-Time Monitoring**: AI-powered digital biomarker tracking from wearable data.
- **Device Connect API (Python FastAPI)**: Manages 8 device types and 13 vendor integrations with OAuth, BLE, QR code pairing, and HIPAA audit logging.
- **Deterioration Prediction System**: Uses statistical and AI models for comprehensive health change detection and risk scoring.
- **ML Inference Infrastructure**: Self-hosted, HIPAA-compliant system with model registry, Redis caching, and ONNX optimization.
- **Alert Orchestration Engine**: Multi-channel (dashboard, email, SMS) rule-based alert delivery.
- **Behavior AI Analysis System**: Multi-modal deterioration detection using ensemble ML models for behavioral patterns, digital biomarkers, cognitive testing, and sentiment analysis.
- **Mental Health AI Dashboard**: Integrates questionnaires (PHQ-9, GAD-7, PSS-10) with AI analysis, including crisis routing for severe PHQ-9 Q9 responses.
- **Followup Autopilot Engine**: An ML-powered adaptive follow-up system aggregating patient signals from 8 modules to predict deterioration, optimize follow-up timing, and generate adaptive tasks. It includes a Signal Ingestor, Feature Builder, ML Models (PyTorch LSTM, XGBoost, IsolationForest), Trigger Engine, Task Engine, and Notification Engine, all with privacy safeguards.
- **Unified Medication System**: Production-grade medication management with AI assistance and conflict detection, supporting doctor-only prescription authoring and role-based routing.
- **Multi-Agent Communication System**: Facilitates communication between AI agents, users, and providers via a Message Router, dual-layer Memory Service, and a `MessageEnvelope` protocol. The Agent Hub UI offers a unified conversation interface with WebSockets, tool calls, and human-in-the-loop approvals.
- **ML Training Infrastructure**: Production-grade system with comprehensive patient consent controls across 22+ granular data types, including a patient-facing Consent UI, Data Extraction Pipeline, and full HIPAA compliance.
- **Background Scheduler**: APScheduler-based jobs on the Python FastAPI backend for tasks like risk scoring, data quality checks, ETL processes, ML inference sweeps, and nightly data warehouse aggregation.
- **Epidemiology Analytics Platform (Phase 9)**: Comprehensive data warehouse and surveillance analytics with:
  - SQLAlchemy models for epidemiology cases, infectious events, occupational incidents, genetic markers, drug safety signals, and vaccine coverage
  - Data warehouse fact tables (DailySurveillanceAggregate, WeeklyIncidenceSummary, OccupationalCohort) for pre-aggregated analytics
  - EpidemiologyService with k-anonymity privacy protection (threshold=5) to suppress low-count records
  - Nightly ETL warehouse aggregation jobs (daily surveillance, weekly incidence, R-value calculation, cohort updates)
  - Frontend EpidemiologyTab with drilldowns for drug safety, infectious disease, vaccine analytics, occupational health, and genetic epidemiology
  - All research data access logged via HIPAAAuditLogger for HIPAA compliance
- **Research Center + Medical NLP (Phase 10)**: Production-grade research data platform with:
  - SQLAlchemy models: AnalysisArtifact, ResearchDataset, DatasetLineage, StudyJob, StudyJobEvent, ResearchCohortSnapshot, ResearchExport, NLPDocument, NLPRedactionRun, ResearchQASession, ResearchQAMessage, ResearchCohort, ResearchStudy
  - ResearchStorageService: Artifact and dataset storage with S3/local support, checksums, signed URLs, retention policies, and HIPAA audit logging
  - PHIRedactionService: OpenAI GPT-4o-based PHI detection with regex pre-detection, 18 PHI entity types, configurable confidence thresholds, and audit trails
  - ResearchQAService: AI-powered Q&A for research data analysis using OpenAI with research-specific prompting and de-identified data constraints
  - Comprehensive REST API (`/api/v1/research-center/`): Cohorts, studies, jobs, artifacts, datasets, exports, NLP documents, and Q&A sessions
  - k-anonymity protection (threshold=5) enforced at export creation time via ResearchExportService
  - PHI export security: Fresh short-lived URLs (15-minute TTL) generated per request, never stored URLs exposed
  - HIPAA audit logging with full request context (client IP, user agent, URL expiry) for all download attempts
  - Doctor-only access with ownership verification on all export downloads
  - Frontend Research Center with 4 tabs: Exports (wizard + status tracking), Datasets (version browser + lineage), NLP Redaction (document upload + PHI review), Research Q&A (AI chat + sessions)
- **Habit Tracker + Behavior AI (Phase 11)**: Production-grade habit tracking system with:
  - GamificationService: XP/points system, 22 badge definitions, 10-level progression, plant growth visualization
  - HabitCoachingService: OpenAI GPT-4o AI coaching with 5 personality types, session memory, proactive interventions
  - StreakCalculationService: Streak management with freeze tokens, milestones (3-365 days), grace period logic
  - HabitReminderService: Multi-channel delivery (SMS, email, push, in-app) with adaptive timing
  - HabitBehaviorIngestorService: Feeds habit data into Behavior AI for cross-signal correlation
  - BehaviorPredictionService: ML-based risk predictions from habit patterns
  - APScheduler jobs: Nightly streak validation (1 AM), reminder dispatch (every 5 min), adaptive timing (3 AM)
  - Frontend HabitGamificationPanel: Badge gallery, level progress, streak visualization
  - Database: habit_streak_freezes table with token management
  - Express proxy routes for all /api/habits/* endpoints to FastAPI
  - HIPAA audit logging on all endpoints with HIPAAAuditLogger.log_phi_access()
- **Tinker Thinking Machine Integration (Phase 12)**: HIPAA-compliant external AI analysis platform operating in NON-BAA mode with:
  - Privacy Firewall: Multi-layer PHI protection with 18+ regex patterns, SHA256 salted hashing, data bucketing (age, vitals)
  - K-Anonymity Enforcement: Hard-fail pattern requiring k≥25 cohort size for all operations
  - API Client: Circuit breaker pattern, rate limiting, automatic retry with exponential backoff
  - Feature Builder: Aggregates patient data into privacy-safe feature vectors for AI analysis
  - Tinker Service Orchestrator: Coordinates cohort analysis, drift detection, study/trial management
  - Database: 15 SQLAlchemy models including TinkerCohort, TinkerStudy, TinkerTrial, DriftAlert, AIAuditLog
  - REST API (`/api/v1/tinker/`): Health, cohort analysis, drift monitoring, study/trial management, audit logs, privacy stats
  - Frontend Dashboard: Overview, cohort analysis UI, drift monitoring charts, privacy statistics (doctor/admin only), audit logs
  - Security: Never sends PHI to external API - only hashed identifiers, bucketed values, k-anonymized aggregates
  - Configuration: TINKER_ENABLED, TINKER_API_KEY env vars required to activate
- **Phase E: Tinker QA & Genius Features**: Comprehensive testing and advanced AI features:
  - **Unit Tests (E.1-E.4)**: 80 pytest tests validating privacy_firewall PHI blocking, k-anonymity enforcement, tinker_client safe defaults, and CohortDSL validation
  - **Research Genius (E.5-E.8)**: GeniusResearchService with auto preregistration from protocol, bias checklist (confounding/selection/immortal-time bias warnings), sensitivity suite (negative controls, placebo outcomes), exportable study bundle (ZIP with CohortDSL, Protocol, metrics, reproducibility hashes)
  - **Clinical Operations Genius (E.9-E.11)**: GeniusClinicalService with alert budget tuning per clinic, alert burden fairness checker (subgroup distribution analysis), dynamic thresholds by clinic workload
  - **Patient Genius (E.12-E.14)**: GeniusPatientService with effort-aware daily check-ins (2-8 questions based on stability), just-in-time micro-habits (template picks by engagement bucket), safe trend explanations (templated language, no raw numbers)
- **HIPAA Compliance & Access Control**: Features a Unified Access Control Service (`AccessControlService`), `HIPAAAuditLogger`, `AccessScope` and `PHICategory` enums, and `RequirePatientAccess` FastAPI dependency for robust, route-level access control and audit logging.
- **Authentication Flow**: Auth0 for frontend, session-based for Express, JWT (DEV_MODE_SECRET) for Express-to-Python, and internal JWT verification in Python. Role-based routing ensures appropriate access for Admin, Doctor, and Patient users.

### Security and Compliance
The platform is HIPAA-compliant, using Auth0 for frontend authentication, session-based for Express routes, and JWT for Express-to-Python communication. It includes BAA verification, comprehensive audit logging, end-to-end encryption, strict PHI handling, explicit doctor-patient assignment authorization, and Google Authenticator TOTP for Admin ML Training Hub.

## External Dependencies

- **Authentication**: Auth0
- **Database**: Neon serverless PostgreSQL
- **AI Services**: OpenAI API, TensorFlow.js, PyTorch, HuggingFace Transformers, ONNX Runtime
- **Caching**: Redis
- **Communication**: Twilio API, AWS SES
- **Video Conferencing**: Daily.co
- **Cloud Services**: AWS S3, AWS Textract