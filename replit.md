# Followup AI - HIPAA-Compliant Health Platform

## Overview
Followup AI is a HIPAA-compliant health monitoring platform designed for chronic care patients. It provides personalized health tracking, medication management, and wellness activities. The platform utilizes AI agents (Agent Clona for patient support and Assistant Lysa for doctor assistance) to enhance patient care through advanced AI and robust health data management. Its primary goal is to be a comprehensive wellness monitoring and change detection system, aiming to deliver fully functional, production-ready applications to transform healthcare for individuals managing ongoing health conditions.

## User Preferences
- **Preferred communication style**: Simple, everyday language
- **Backend Language**: Python only - ALL backend code must be written in Python (FastAPI)
- **Frontend**: React/TypeScript (standard web practice)
- **Development Philosophy**: Building FULL FUNCTIONAL applications, NOT MVPs. All features must be completely functional and production-ready before delivery.
- **Code Quality**: Avoid duplicate code - use proper code reuse, modular design, and shared components/utilities.

## System Architecture

### Frontend
The frontend is built with React, TypeScript, Vite, Wouter for routing, TanStack Query for data fetching, and Tailwind CSS for styling. It uses Radix UI and shadcn/ui for a clinical aesthetic, supporting role-based routing and context-based theming. It routes Python AI endpoints to the FastAPI backend (port 8000) and other endpoints to an Express server (port 5000).

### Backend
The backend comprises two main components:
- **Node.js Express server (Port 5000)**: Handles chatbot, appointments, calendar, consultations, pain tracking, symptom journaling, and risk scoring.
- **Python FastAPI server (Port 8000)**: Manages AI deterioration detection, guided examinations, mental health questionnaires, database interactions, authentication, and an async AI engine with a singleton manager for ML models.

### Core Features & Technical Implementations
- **AI Integration**: Leverages OpenAI API (GPT-4o, o1) for PHI detection, symptom extraction, and clinical reasoning.
- **PHI Detection Service**: GPT-4o-based HIPAA-compliant PHI detection and medical entity extraction.
- **Real-Time Monitoring**: AI-powered digital biomarker tracking from wearable data.
- **Device Connect API**: Manages 8 device types and 13 vendor integrations with OAuth, BLE, QR code pairing, and HIPAA audit logging.
- **Deterioration Prediction System**: Uses statistical and AI models for health change detection and risk scoring.
- **ML Inference Infrastructure**: Self-hosted, HIPAA-compliant system with model registry, Redis caching, and ONNX optimization.
- **Alert Orchestration Engine**: Multi-channel (dashboard, email, SMS) rule-based alert delivery.
- **Behavior AI Analysis System**: Multi-modal deterioration detection using ensemble ML models for behavioral patterns, digital biomarkers, cognitive testing, and sentiment analysis.
- **Mental Health AI Dashboard**: Integrates questionnaires (PHQ-9, GAD-7, PSS-10) with AI analysis and crisis routing.
- **Followup Autopilot Engine**: An ML-powered adaptive follow-up system that aggregates patient signals to predict deterioration, optimize follow-up timing, and generate adaptive tasks, incorporating privacy safeguards.
- **Unified Medication System**: Production-grade medication management with AI assistance and conflict detection, supporting doctor-only prescription authoring and role-based routing.
- **Multi-Agent Communication System**: Facilitates communication between AI agents, users, and providers via a Message Router, Memory Service, and `MessageEnvelope` protocol. The Agent Hub UI provides a unified conversation interface.
- **ML Training Infrastructure**: Production-grade system with comprehensive patient consent controls across 22+ granular data types, including a patient-facing Consent UI, Data Extraction Pipeline, and full HIPAA compliance.
- **Background Scheduler**: APScheduler-based jobs on the Python FastAPI backend for tasks like risk scoring, data quality checks, ETL processes, and ML inference sweeps.
- **Epidemiology Analytics Platform**: Comprehensive data warehouse and surveillance analytics with SQLAlchemy models for various health metrics, nightly ETL, and k-anonymity privacy protection.
- **Research Center + Medical NLP**: Production-grade research data platform with SQLAlchemy models, artifact/dataset storage (S3/local), PHI redaction via GPT-4o, AI-powered Q&A, k-anonymity protection, and robust security for PHI export.
- **Habit Tracker + Behavior AI**: Production-grade habit tracking system with gamification, AI coaching, streak management, multi-channel reminders, and integration with Behavior AI for risk prediction. It also includes Agent Clona for personalized habit recommendations.
- **Tinker Thinking Machine Integration**: HIPAA-compliant external AI analysis platform operating in NON-BAA mode with a Privacy Firewall, K-Anonymity Enforcement, API Client with circuit breaker patterns, Feature Builder, and Tinker Service Orchestrator for cohort analysis and drift detection.
- **Genius Features**: Includes Research Genius for study design and analysis, Clinical Operations Genius for alert management, and Patient Genius for personalized check-ins and micro-habits.
- **Extended Profile System**: Production-grade medical information management for patients (emergency contacts, medications, allergies, conditions) and doctors (NPI, affiliations, certifications, telemedicine details).
- **HIPAA Compliance & Access Control**: Features a Unified Access Control Service, `HIPAAAuditLogger`, and route-level access control with `AccessScope` and `PHICategory` enums.
- **Authentication Flow**: Stytch-based authentication with Magic Links and SMS OTP for passwordless login, session-based cookies for Express, and M2M Client Credentials for Express-to-FastAPI communication, ensuring role-based access control.

### Security and Compliance
The platform is HIPAA-compliant, utilizing Stytch for passwordless authentication, HttpOnly session cookies, and M2M tokens for inter-service communication. It includes comprehensive audit logging, end-to-end encryption, strict PHI handling, explicit doctor-patient assignment authorization, and Google Authenticator TOTP for Admin ML Training Hub.

## External Dependencies

- **Authentication**: Stytch (Magic Links, SMS OTP, M2M)
- **Database**: Neon serverless PostgreSQL
- **AI Services**: OpenAI API, TensorFlow.js, PyTorch, HuggingFace Transformers, ONNX Runtime
- **Caching**: Redis
- **Communication**: Stytch (SMS, Email) - replaces Twilio/SES
- **Video Conferencing**: Daily.co
- **Cloud Services**: Google Cloud Platform (GCS, Document AI, Healthcare NLP API)

## Google Cloud Platform Architecture

### GCP Services Integration
The platform has migrated from AWS to Google Cloud Platform for HIPAA-compliant cloud services:

- **Google Cloud Storage (GCS)**: Replaces AWS S3 for all file storage (audio, images, research exports, medical documents) with signed URLs and local filesystem fallback for development
- **Document AI Healthcare Parser**: Replaces AWS Textract for OCR processing of medical documents with specialized healthcare document parsing
- **Healthcare NLP API**: Replaces AWS Comprehend Medical for medical entity extraction (ICD-10, RxNorm, SNOMED CT codes, PHI detection)
- **Cloud KMS**: Customer-managed encryption keys for PHI data at rest
- **Healthcare API (FHIR)**: Clinical data store for structured health records

### GCP Configuration Files
**TypeScript (Express backend)**:
- `server/config/gcpConstants.ts` - Centralized GCP project IDs, regions, bucket names, KMS keys
- `server/gcpConfig.ts` - Google Cloud client initialization (Storage, KMS, Healthcare API, Document AI)
- `server/services/gcpStorageService.ts` - GCS with signed URLs, KMS encryption, Zod validation
- `server/services/gcpHealthcareNLP.ts` - Healthcare NLP API for medical entity extraction
- `server/services/gcpDocumentAI.ts` - Document AI for OCR and healthcare document parsing
- `server/services/gcpFhirService.ts` - Healthcare API FHIR store operations
- `server/services/gcpKmsService.ts` - Cloud KMS encryption/decryption

**Python (FastAPI backend)**:
- `app/gcp_config.py` - Python Google Cloud client initialization with local fallback
- `app/services/gcs_service.py` - Async GCS storage service with local filesystem fallback
- `app/services/gcp_healthcare_service.py` - Healthcare NLP and FHIR services

### Environment Variables for GCP
- `GOOGLE_APPLICATION_CREDENTIALS` - Path to GCP service account JSON (optional, falls back to local storage)
- `GCP_PROJECT_ID` - Google Cloud project ID
- `GCS_BUCKET_NAME` - Cloud Storage bucket for PHI data
- `GCP_LOCATION` - Region for Healthcare API services (default: us-central1)
- `GCP_KMS_KEY_RING` - KMS key ring name
- `GCP_KMS_KEY_NAME` - KMS key name for encryption

### GCP Fallback Services with OpenAI
The platform includes intelligent fallback services that gracefully degrade to OpenAI GPT-4o when GCP services are unavailable:

**Healthcare NLP Fallback Service** (`app/services/healthcare_nlp_fallback.py`):
- Extracts medical entities (medications, diagnoses, procedures) with clinical precision
- Returns GCP Healthcare API-compatible JSON format
- Provides ICD-10, RxNorm, and SNOMED CT code extraction
- PHI detection and entity categorization
- API: `/api/gcp-fallback/healthcare-nlp/*`

**Document AI Fallback Service** (`app/services/document_ai_fallback.py`):
- OCR text extraction from medical documents
- Key-value extraction for structured data
- Healthcare document parsing
- API: `/api/gcp-fallback/document-ai/*`

**Configuration** (`app/config/gcp_constants.py`):
- `is_openai_fallback_available()` - Check OpenAI availability
- `get_healthcare_nlp_status()` - Healthcare NLP service status
- `get_document_ai_status()` - Document AI service status

**Fallback Priority**:
1. Attempt GCP API if credentials configured
2. Fall back to OpenAI GPT-4o (BAA-protected) if GCP unavailable
3. All prompts maintain HIPAA-compliant clinical formatting

### Development Mode
When GCP credentials are not configured, the system automatically falls back to local filesystem storage in `./local_storage/` for development purposes. All GCS operations gracefully degrade without errors. The Healthcare NLP and Document AI services use OpenAI GPT-4o as their primary source when GCP is not configured.