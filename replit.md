# Followup AI - HIPAA-Compliant Health Platform

## Overview
Followup AI is a HIPAA-compliant health monitoring platform designed for immunocompromised patients, offering personalized health tracking, medication management, and wellness activities. It leverages AI agents (Agent Clona for patient support and Assistant Lysa for doctor assistance) to provide insights, streamline healthcare operations, and act as a comprehensive wellness monitoring and change detection system. The platform aims to enhance patient care through advanced AI and robust health data management, with a business vision to transform healthcare for immunocompromised individuals.

## User Preferences
- **Preferred communication style**: Simple, everyday language
- **Backend Language**: Python only - ALL backend code must be written in Python (FastAPI)
- **Frontend**: React/TypeScript (standard web practice)
- **Development Philosophy**: Building FULL FUNCTIONAL applications, NOT MVPs. All features must be completely functional and production-ready before delivery.

## System Architecture

### Frontend
The frontend is built with React, TypeScript, Vite, Wouter for routing, TanStack Query for data fetching, and Tailwind CSS for styling. It uses Radix UI and shadcn/ui for a clinical aesthetic, supporting role-based routing and context-based theming. It routes Python AI endpoints to the FastAPI backend on port 8000 and other endpoints to an Express server on port 5000.

### Backend
The backend comprises two services: a Node.js Express server (Port 5000) for chatbot, appointments, calendar, consultations, pain tracking, symptom journaling, and risk scoring, and a Python FastAPI server (Port 8000) for AI deterioration detection, guided examinations, mental health questionnaires, database interactions, and core authentication. The FastAPI backend uses an async AI engine with a singleton manager to load heavy ML models at startup.

### Core Features & Technical Implementations
- **AI Integration:** Utilizes OpenAI API (GPT-4o for PHI detection, symptom extraction; o1 for advanced clinical reasoning) with graceful fallbacks and audit logging.
- **PHI Detection Service (Python):** GPT-4o-based HIPAA-compliant PHI detection and medical entity extraction with regex fallbacks.
- **Real-Time Monitoring:** AI-powered digital biomarker tracking from wearable data.
- **Voice-Based Followups:** OpenAI Whisper for transcription and GPT-4 for analysis.
- **Assistant Lysa:** AI-powered appointment management, email categorization, and reminders.
- **Secure Data Sharing:** Consent-managed, audit-logged patient record sharing.
- **EHR & Wearable Integration:** FHIR-based integration.
- **Video Consultations:** HIPAA-compliant video conferencing.
- **Home Clinical Exam Coach (HCEC):** AI-guided self-examinations using OpenAI Vision for metrics like respiratory rate and skin pallor.
- **Deterioration Prediction System:** Comprehensive health change detection via baseline calculation, Z-score, anomaly detection, Bayesian risk modeling, and time-series analysis for a composite risk score.
- **ML Inference Infrastructure:** Self-hosted, HIPAA-compliant system with model registry, Redis caching, and ONNX optimization.
- **Guided Video Examination:** 4-stage workflow (Eyes, Palm, Tongue, Lips) with LAB color analysis and S3 encrypted storage, including Facial Puffiness Score (FPS).
- **Guided Audio Examination:** 4-stage workflow (Breathing, Coughing, Speaking, Reading) with YAMNet ML classification and S3 encrypted storage.
- **Alert Orchestration Engine:** Multi-channel (dashboard, email, SMS) rule-based alert delivery.
- **Behavior AI Analysis System:** Multi-modal deterioration detection using behavioral patterns, digital biomarkers, cognitive testing, and sentiment analysis via ensemble ML models. Includes Gait Analysis and Accelerometer Tremor Analysis.
- **Risk Scoring Dashboard:** Composite risk score (0-15) with weighted factors and 7-day history.
- **Baseline Calculation UI:** 7-day rolling window statistics with recalculation and history visualization.
- **Google Calendar Sync:** Bidirectional doctor appointment sync with OAuth and PHI handling.
- **Drug-Drug Interaction Detection:** Medication adherence with RxNorm integration.
- **Automatic Drug Normalization:** On-demand RxNorm API integration for standardized drug records.
- **Clinical Automation (Rx Builder):** AI-assisted prescription system with SOAP notes, ICD-10 suggestions, differential diagnosis, drug interaction checks, auto-dosage, and chronic refill automation.
- **Clinical Assessment (Diagnosis Helper):** AI-assisted assessment with dual authorization, patient data aggregation, access scope enforcement, and AI-powered differential diagnosis.
- **PainTrack Platform:** Chronic pain tracking with dual-camera video, VAS slider, and medication tracking.
- **Mental Health AI Dashboard:** Integrated questionnaires (PHQ-9, GAD-7, PSS-10) with AI analysis and crisis detection.
- **Agent Clona Symptom Extraction:** AI-powered symptom extraction from patient conversations.
- **AI-Powered Habit Tracker:** Comprehensive habit management including routines, reminders, AI coaching, trigger detection, and gamification.
- **Daily Follow-up Dashboard Pattern:** Enforces 24-hour gating for data display across various health tabs.
- **Doctor-Patient Assignment System:** Explicit doctor-patient relationships with consent, access levels, and HIPAA audit logging.
- **Per-Doctor Personal Integrations:** OAuth-based integrations for Gmail, WhatsApp Business API, and Twilio VoIP with encrypted credential storage.

### Multi-Agent Communication System
A multi-agent system facilitates real-time communication between AI agents, users, and healthcare providers.
- **Architecture:** Agent Clona (patient-facing), Assistant Lysa (doctor-facing), Message Router, and a dual-layer Memory Service (Redis for short-term, PostgreSQL pgvector for long-term).
- **Message Protocol:** Standardized `MessageEnvelope` for all communications.
- **Agent Hub UI:** Unified conversation interface with WebSockets, presence indicators, tool call status, human-in-the-loop approvals, and streaming ChatGPT responses with real-time display and symptom extraction.
- **Consent-Verified Approval System:** ORM-based repository with consent verification and HIPAA audit logging for authorization failures.
- **Database Schema:** Dedicated tables for agents, messages, conversations, tasks, tools, memory (pgvector), and audit logs.
- **Tool Microservices:** Extend AI agent capabilities with real database operations, permission checking, consent verification, parameter validation, and dual audit logging (e.g., CalendarTool, MessagingTool, PrescriptionDraftTool, EHRFetchTool, LabFetchTool, ImagingLinkerTool).
- **Memory Persistence:** Centralized memory injection via `ensure_memory_service()` in AgentEngine with idempotent `_memory_initialized` flag guaranteeing consistent memory availability across REST, WebSocket, and worker pathways.
- **Doctor Patient Overview Panel (My Patients):** HIPAA-compliant patient overview for doctors with:
  - `/api/agent/patients` - List assigned patients with risk scores, alert counts, medication counts
  - `/api/agent/patients/{id}/overview` - Comprehensive patient profile (daily followups, health alerts, medications, conditions)
  - `/api/agent/patients/{id}/conversations` - Lysa conversation history with message previews
  - Role-based authorization with active doctor-patient assignment verification
  - Full HIPAA audit logging for PHI access (successful and denied attempts)

### Security and Compliance
HIPAA-compliant, utilizing AWS Cognito for authentication, BAA verification, comprehensive audit logging, end-to-end encryption, strict PHI handling, and explicit doctor-patient assignment authorization.

## External Dependencies

- **Authentication:** AWS Cognito
- **Database:** Neon serverless PostgreSQL
- **AI Services:** OpenAI API, TensorFlow.js, PyTorch, HuggingFace Transformers, ONNX Runtime
- **Caching:** Redis
- **Communication:** Twilio API, AWS SES
- **Video Conferencing:** Daily.co
- **Cloud Services:** AWS S3, AWS Textract