# Followup AI - HIPAA-Compliant Health Platform

## Overview
Followup AI is a HIPAA-compliant health monitoring platform for immunocompromised patients, offering personalized health tracking, medication management, and wellness activities. It integrates two AI agents, Agent Clona for patient support and Assistant Lysa for doctor assistance, to provide AI-powered insights and streamline healthcare operations. The platform aims to enhance patient care through advanced AI and comprehensive health data management, positioning itself as a wellness monitoring and change detection platform rather than a medical device to avoid FDA/CE approval requirements.

## User Preferences
- **Preferred communication style**: Simple, everyday language
- **Backend Language**: Python only - ALL backend code must be written in Python (FastAPI)
- **Frontend**: React/TypeScript (standard web practice)

## System Architecture

### Frontend
The frontend uses React, TypeScript, Vite, Wouter for routing, TanStack Query for data fetching, and Tailwind CSS for styling. It incorporates Radix UI and shadcn/ui for a clinical aesthetic, supporting role-based routing and context-based theming.

### Backend
The preferred backend is a Python FastAPI implementation, utilizing SQLAlchemy ORM for PostgreSQL, AWS Cognito for authentication and authorization (JWTs with role-based access control), and Pydantic for validation. A legacy JavaScript Express.js backend also exists but is being phased out.

**Core Features & Technical Implementations:**
-   **AI Integration:** Leverages OpenAI API (GPT-4o) for Agent Clona (symptom analysis, wellness suggestions) and Assistant Lysa (doctor assistance, sentiment analysis, medical entity extraction, session summaries).
-   **Personalization & Recommendation System:** A rule-based intelligent recommendation engine, enhanced with RAG, for tailored patient suggestions and clinical insights for doctors.
-   **Advanced Drug Interaction Detection:** AI-powered system using GNN and NLP to detect drug-drug and drug-gene interactions.
-   **Real-Time Immune Function Monitoring:** AI-powered digital biomarker tracking from wearable data to create "immune digital twins."
-   **Environmental Pathogen Risk Mapping:** Provides location-based real-time environmental health monitoring with AI-generated safety recommendations.
-   **Voice-Based Daily Followups:** Uses OpenAI Whisper for transcription and GPT-4 for analysis of voice recordings.
-   **Assistant Lysa Receptionist Features:** AI-powered appointment management, doctor availability, email thread categorization, call log transcription, and automated reminders with Google Calendar and Gmail integration.
-   **Secure Patient Record Sharing:** System with consent management and audit logging.
-   **Health Insight Consent:** Granular control for patients over data sharing.
-   **EHR & Wearable Integration:** FHIR-based integration with major EHR systems (Epic, Cerner) and popular wearable devices.
-   **Video Consultations:** HIPAA-compliant video conferencing via Daily.co.
-   **Pain Detection Camera System:** AI-powered facial analysis for tracking discomfort progression, including daily recordings, pain scoring, and trend analysis.
-   **Home Clinical Exam Coach (HCEC):** AI-powered guided self-examination system using OpenAI Vision for real-time feedback on standardized exam techniques.
-   **Symptom Journal Enhancements:** Includes AI-powered respiratory rate analysis, comparison views for measurements, and weekly PDF reports for doctors.
-   **Medication Side-Effect Predictor:** AI-powered system correlating patient symptoms with medication timelines to detect potential side effects.

### Security and Compliance
The platform is HIPAA-compliant, featuring AWS Cognito for authentication and role-based access control, BAA verification for all integrations, comprehensive audit logging for PHI access, end-to-end encryption for video consultations, and strict PHI handling compliance across all services. The platform is strategically positioned as a General Wellness Product, avoiding medical device classification by prohibiting diagnostic or treatment claims and emphasizing wellness monitoring and discussion with healthcare providers.

## External Dependencies

-   **Authentication:** AWS Cognito.
-   **Database:** Neon serverless PostgreSQL.
-   **AI Services:** OpenAI API (GPT models).
-   **Machine Learning:** TensorFlow.js for facial analysis.
-   **Communication:** Twilio API (SMS, voice), AWS SES (transactional emails).
-   **Video Conferencing:** Daily.co.
-   **Cloud Services:** AWS S3, AWS Textract, AWS Comprehend Medical, AWS HealthLake, AWS HealthImaging, AWS HealthOmics.
-   **Data Integration APIs:** PubMed E-utilities, PhysioNet WFDB, Kaggle API, WHO Global Health Observatory API, OpenWeatherMap API, Biobot Analytics.