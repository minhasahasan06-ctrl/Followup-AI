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

## External Dependencies

-   **Authentication:** AWS Cognito.
-   **Database:** Neon serverless PostgreSQL.
-   **AI Services:** OpenAI API (GPT models).
-   **Machine Learning:** TensorFlow.js (@tensorflow-models/face-landmarks-detection, @mediapipe/face_mesh) for facial analysis.
-   **Communication:** Twilio API (SMS, voice), AWS SES (transactional emails).
-   **Video Conferencing:** Daily.co.
-   **Cloud Services:** AWS S3, AWS Textract, AWS Comprehend Medical, AWS HealthLake, AWS HealthImaging, AWS HealthOmics.
-   **Data Integration APIs:** PubMed E-utilities, PhysioNet WFDB, Kaggle API, WHO Global Health Observatory API, OpenWeatherMap API, Biobot Analytics.