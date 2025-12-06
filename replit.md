# Followup AI - HIPAA-Compliant Health Platform

## Overview
Followup AI is a HIPAA-compliant health monitoring platform designed for immunocompromised patients. It offers personalized health tracking, medication management, and wellness activities, leveraging AI agents (Agent Clona for patient support and Assistant Lysa for doctor assistance). The platform aims to enhance patient care through advanced AI and robust health data management, acting as a comprehensive wellness monitoring and change detection system, with the ultimate goal of transforming healthcare for immunocompromised individuals.

## User Preferences
- **Preferred communication style**: Simple, everyday language
- **Backend Language**: Python only - ALL backend code must be written in Python (FastAPI)
- **Frontend**: React/TypeScript (standard web practice)
- **Development Philosophy**: Building FULL FUNCTIONAL applications, NOT MVPs. All features must be completely functional and production-ready before delivery.
- **Code Quality**: Avoid duplicate code - use proper code reuse, modular design, and shared components/utilities.

## System Architecture

### Frontend
The frontend is built with React, TypeScript, Vite, Wouter for routing, TanStack Query for data fetching, and Tailwind CSS for styling. It uses Radix UI and shadcn/ui for a clinical aesthetic, supporting role-based routing and context-based theming. It routes Python AI endpoints to the FastAPI backend on port 8000 and other endpoints to an Express server on port 5000.

### Backend
The backend comprises a Node.js Express server (Port 5000) for chatbot, appointments, calendar, consultations, pain tracking, symptom journaling, and risk scoring. A Python FastAPI server (Port 8000) handles AI deterioration detection, guided examinations, mental health questionnaires, database interactions, and core authentication, utilizing an async AI engine with a singleton manager for ML model loading.

### Core Features & Technical Implementations
- **AI Integration**: Utilizes OpenAI API (GPT-4o for PHI detection, symptom extraction; o1 for advanced clinical reasoning) with graceful fallbacks.
- **PHI Detection Service**: GPT-4o-based HIPAA-compliant PHI detection and medical entity extraction.
- **Real-Time Monitoring**: AI-powered digital biomarker tracking from wearable data.
- **Voice-Based Followups**: OpenAI Whisper for transcription and GPT-4 for analysis.
- **Assistant Lysa**: AI for appointment management, email categorization, and reminders.
- **Secure Data Sharing**: Consent-managed and audit-logged patient record sharing.
- **EHR & Wearable Integration**: FHIR-based integration with production-grade Device Connect API.
- **Device Connect API (Python FastAPI)**: Manages 8 device types and 13 vendor integrations with OAuth, BLE, and QR code pairing, robust authentication, HIPAA audit logging, and health analytics.
- **Device Data Pipeline**: Connects device data to health alerts and ML training, including `HealthSectionAnalyticsEngine` and an anonymized data extraction process for ML.
- **Video Consultations**: HIPAA-compliant video conferencing.
- **Home Clinical Exam Coach (HCEC)**: AI-guided self-examinations using OpenAI Vision.
- **Deterioration Prediction System**: Comprehensive health change detection via statistical and AI models for a composite risk score.
- **ML Inference Infrastructure**: Self-hosted, HIPAA-compliant system with model registry, Redis caching, and ONNX optimization.
- **Alert Orchestration Engine**: Multi-channel (dashboard, email, SMS) rule-based alert delivery.
- **Behavior AI Analysis System**: Multi-modal deterioration detection using behavioral patterns, digital biomarkers, cognitive testing, and sentiment analysis via ensemble ML models.
- **Risk Scoring Dashboard**: Composite risk score (0-15) with weighted factors and 7-day history.
- **Baseline Calculation UI**: 7-day rolling window statistics with recalculation and history visualization.
- **Google Calendar Sync**: Bidirectional doctor appointment sync.
- **Drug Interaction & Normalization**: Medication adherence with RxNorm integration.
- **Clinical Automation (Rx Builder)**: AI-assisted prescription system with SOAP notes, ICD-10 suggestions, and drug interaction checks.
- **Clinical Assessment (Diagnosis Helper)**: AI-assisted assessment with dual authorization and AI-powered differential diagnosis.
- **PainTrack Platform**: Chronic pain tracking with video, VAS slider, and medication tracking.
- **Mental Health AI Dashboard**: Integrated questionnaires (PHQ-9, GAD-7, PSS-10) with AI analysis.
- **Agent Clona Symptom Extraction**: AI-powered symptom extraction from patient conversations.
- **AI-Powered Habit Tracker**: Comprehensive habit management with AI coaching and gamification.
- **Daily Follow-up Dashboard Pattern**: Enforces 24-hour gating for data display.
- **Doctor-Patient Assignment System**: Explicit relationships with consent, access levels, and HIPAA audit logging.
- **Per-Doctor Personal Integrations**: OAuth-based integrations for Gmail, WhatsApp Business API, and Twilio VoIP.
- **Admin ML Training Hub**: Dashboard for managing ML training datasets, jobs, models, device data extraction, and consent.
- **Medical NLP Dashboard**: AI-powered document analysis with entity recognition, PHI redaction, and Q&A.
- **Enhanced Research Center**: Epidemiology research hub for cohort visualization, study management, and AI-powered report generation.
- **Unified Medication System**: Production-grade medication management with patient records, active medication dashboard, and doctor-only prescription authoring with AI assistance and conflict detection, supporting role-based routing.

### Multi-Agent Communication System
A multi-agent system facilitates communication between AI agents, users, and providers using a Message Router, dual-layer Memory Service (Redis, PostgreSQL pgvector), and a `MessageEnvelope` protocol. The Agent Hub UI offers a unified conversation interface with WebSockets, tool calls, human-in-the-loop approvals, and streaming responses, backed by a Consent-Verified Approval System and dedicated database schema. Tool Microservices extend AI agent capabilities with secure database operations.

### ML Training Infrastructure
A production-grade ML model training system with comprehensive patient consent controls across 22+ granular data type categories, including detailed medical device readings. It features a patient-facing Consent UI, a Data Extraction Pipeline, Feature Engineering, and full HIPAA compliance with audit logging, k-anonymity, and differential privacy. The pipeline handles consent-filtered patient data extraction, anonymization, and preprocessing, integrating with public datasets and synthetic data generation. A Training Job Worker processes background jobs, and a FastAPI Training API manages datasets, models, and consent statistics.

### Security and Compliance
The platform is HIPAA-compliant, utilizing AWS Cognito for authentication, BAA verification, comprehensive audit logging, end-to-end encryption, strict PHI handling, and explicit doctor-patient assignment authorization. The Admin ML Training Hub is protected with Google Authenticator TOTP for enhanced security, including lockout protection and session-based verification.

## External Dependencies

- **Authentication**: AWS Cognito
- **Database**: Neon serverless PostgreSQL
- **AI Services**: OpenAI API, TensorFlow.js, PyTorch, HuggingFace Transformers, ONNX Runtime
- **Caching**: Redis
- **Communication**: Twilio API, AWS SES
- **Video Conferencing**: Daily.co
- **Cloud Services**: AWS S3, AWS Textract