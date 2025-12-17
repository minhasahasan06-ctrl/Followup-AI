# Followup AI - HIPAA-Compliant Health Platform

## Overview
Followup AI is a HIPAA-compliant health monitoring platform for immunocompromised patients, offering personalized health tracking, medication management, and wellness activities. It leverages AI agents (Agent Clona for patient support, Assistant Lysa for doctor assistance) to enhance patient care through advanced AI and robust health data management, aiming to be a comprehensive wellness monitoring and change detection system. The platform's vision is to transform healthcare for immunocompromised individuals by delivering fully functional, production-ready applications.

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
- **Background Scheduler**: APScheduler-based jobs on the Python FastAPI backend for tasks like risk scoring, data quality checks, ETL processes, and ML inference sweeps.
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