# Followup AI - HIPAA-Compliant Health Platform

## Overview
Followup AI is a HIPAA-compliant health monitoring platform for immunocompromised patients, offering personalized health tracking, medication management, and wellness activities. It utilizes AI agents (Agent Clona for patient support and Assistant Lysa for doctor assistance) to enhance patient care through advanced AI and comprehensive health data management. The platform aims to provide insights and streamline healthcare operations, positioning itself as a wellness monitoring and change detection system.

## Recent Changes

- **Symptom Journal Dashboard Integration (Latest)**: Moved symptom journal from sidebar to daily followup dashboard as 4th tab (positioned between Device Data and Video AI). Tab displays recent measurements with AI observations, color change tracking, respiratory rate for chest exams, and active alerts banner. Compact view shows latest 2 measurements with link to full symptom journal page. Removed /symptom-journal from sidebar navigation while keeping route active for detailed tracking functionality.
- **Complete Audio AI Workflow Inline Integration**: Fully integrated 4-stage guided audio examination workflow directly into Dashboard.tsx daily followup section. Complete implementation includes session management (create/upload/complete), MediaRecorder API integration with prep countdown (30s breathing/speaking, 15s coughing, 40s reading), real-time recording with pulsing mic UI, progress tracking across all 4 stages, ML analysis results display with YAMNet classification, and comprehensive error handling. Removed standalone /ai-audio and /guided-audio-exam routes and pages. All audio AI functionality now accessed inline within daily dashboard tabs - no navigation required. Full proxy routing to Python backend (/api/v1/guided-audio-exam/*).
- **HIPAA Security Framework Deployed**: All tremor and gait analysis endpoints (10 total) now implement complete HIPAA security with authentication, patient ownership verification, comprehensive audit logging with [AUDIT] and [AUTH] tags, data integrity checks, and sanitized error handling preventing PHI exposure.
- **Daily Follow-up Dashboard Enhancements**: Dashboard now displays latest Video AI metrics (respiratory rate, skin pallor, jaundice risk, facial swelling, tremor detection) from today's guided video examination instead of placeholder data. Dashboard streamlined to show only most critical health data from 4 tabs: Device Data, Symptom Journal, Video AI Analysis, and Audio AI Analysis.

## User Preferences
- **Preferred communication style**: Simple, everyday language
- **Backend Language**: Python only - ALL backend code must be written in Python (FastAPI)
- **Frontend**: React/TypeScript (standard web practice)
- **Development Philosophy**: Building FULL FUNCTIONAL applications, NOT MVPs. All features must be completely functional and production-ready before delivery.

## System Architecture

### Frontend
The frontend is built with React, TypeScript, Vite, Wouter for routing, TanStack Query for data fetching, and Tailwind CSS for styling. It uses Radix UI and shadcn/ui for a clinical aesthetic, supporting role-based routing and context-based theming. It routes Python AI endpoints to the FastAPI backend on port 8000 and other endpoints to an Express server on port 5000.

### Backend
The backend consists of two services:
- **Node.js Express Backend (Port 5000)**: Manages the Agent Clona chatbot (GPT-4o powered), appointments, calendar, consultations, pain tracking, symptom journal, voice analysis, baseline calculation, deviation detection, and risk scoring.
- **Python FastAPI Backend (Port 8000)**: Handles all AI deterioration detection endpoints, guided video and audio examinations, database interactions, and core authentication. It uses an async AI engine initialization with an `AIEngineManager` singleton.

### Core Features & Technical Implementations
- **AI Integration:** Leverages OpenAI API (GPT-4o) for various AI functionalities, including symptom analysis, wellness suggestions, doctor assistance, sentiment analysis, and medical entity extraction.
- **Personalization & Recommendation System:** A rule-based recommendation engine enhanced with Retrieval-Augmented Generation (RAG).
- **Real-Time Immune Function Monitoring:** AI-powered digital biomarker tracking from wearable data.
- **Voice-Based Daily Followups:** Uses OpenAI Whisper for transcription and GPT-4 for analysis.
- **Assistant Lysa Receptionist Features:** AI-powered appointment management, email categorization, call log transcription, and automated reminders with Google Calendar and Gmail integration.
- **Secure Patient Record Sharing:** System with consent management and audit logging.
- **EHR & Wearable Integration:** FHIR-based integration with major EHR systems and popular wearable devices.
- **Video Consultations:** HIPAA-compliant video conferencing.
- **Home Clinical Exam Coach (HCEC):** AI-powered guided self-examination using OpenAI Vision.
- **Deterioration Prediction System:** Comprehensive health change detection with baseline calculation, Z-score analysis, anomaly detection, Bayesian risk modeling, and time-series trend analysis to generate a composite risk score.
- **ML Inference Infrastructure:** A self-hosted system with a model registry, Redis caching, async inference, HIPAA-compliant audit logging, batch processing, and ONNX optimization, including pre-trained Clinical-BERT and custom LSTM models.
- **Guided Video Examination System:** A HIPAA-compliant 4-stage self-examination workflow (Eyes, Palm, Tongue, Lips) with clinical-grade LAB color analysis for hepatic and anemia detection, disease-specific personalization, and S3 encrypted storage. The Video AI Engine extracts metrics like respiratory rate, skin pallor, sclera yellowness, facial swelling, head tremor, and nail bed analysis.
- **Facial Puffiness Score (FPS) System:** Comprehensive facial contour tracking using MediaPipe Face Mesh, providing regional scores, baseline comparison, composite FPS, and asymmetry detection.
- **Comprehensive Respiratory Metrics System:** Measures Respiratory Variability Index (RVI), tracks patient baselines, performs temporal analytics, anomaly detection, and advanced pattern detection.
- **DeepLab V3+ Edema Segmentation System:** Medical-grade semantic segmentation for swelling/edema detection using DeepLab V3+ MobileNetV2, featuring 8-Region Anatomical Segmentation, Confidence Scoring, and Advanced Regional Analysis.
- **Guided Audio Examination System:** A HIPAA-compliant 4-stage audio recording workflow (Breathing, Coughing, Speaking, Reading) with YAMNet ML classification, neurological metrics, disease-specific personalization, and S3 encrypted storage. The Audio AI Engine extracts metrics like breath cycles, speech pace, cough detection, wheeze detection, and voice quality.
- **Trend Prediction Engine:** Performs baseline calculation, Z-score analysis, anomaly detection, Bayesian risk modeling, and time-series trend analysis to generate a composite risk score.
- **Alert Orchestration Engine:** Provides multi-channel delivery (dashboard, email, SMS) with rule-based systems and HIPAA compliance.
- **Behavior AI Analysis System:** Comprehensive multi-modal deterioration detection through behavioral pattern analysis, digital biomarkers, cognitive testing, and sentiment analysis. It utilizes an ensemble of ML models (Transformer Encoder, XGBoost, DistilBERT) and various services for behavioral metrics, digital biomarkers, cognitive tests, and sentiment analysis, culminating in a risk scoring engine and deterioration trend engine.
- **Gait Analysis System (HAR-based):** Open-source gait analysis using MediaPipe Pose and HAR datasets, extracting over 40 gait parameters including temporal, spatial, and joint angle metrics, symmetry indices, stability scores, and clinical risk flags.
- **Accelerometer Tremor Analysis System:** Tremor detection from phone accelerometer data using FFT-based signal processing, extracting tremor frequency, amplitude, frequency band power, and clinical classification for Parkinsonian, Essential, and Physiological tremors.

### Security and Compliance
The platform is HIPAA-compliant, featuring AWS Cognito for authentication, BAA verification for all integrations, comprehensive audit logging, end-to-end encryption for video consultations, and strict PHI handling. It is positioned as a General Wellness Product.

## External Dependencies

- **Authentication:** AWS Cognito.
- **Database:** Neon serverless PostgreSQL.
- **AI Services:** OpenAI API (GPT models), TensorFlow.js, PyTorch, HuggingFace Transformers, ONNX Runtime.
- **Caching:** Redis.
- **Communication:** Twilio API, AWS SES.
- **Video Conferencing:** Daily.co.
- **Cloud Services:** AWS S3, AWS Textract, AWS Comprehend Medical, AWS HealthLake, AWS HealthImaging, AWS HealthOmics.
- **Data Integration APIs:** PubMed E-utilities, PhysioNet WFDB, Kaggle API, WHO Global Health Observatory API, OpenWeatherMap API, Biobot Analytics.