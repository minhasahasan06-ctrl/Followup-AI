# Followup AI - HIPAA-Compliant Health Platform

## Overview
Followup AI is a HIPAA-compliant health monitoring platform for immunocompromised patients, offering personalized health tracking, medication management, and wellness activities. It leverages AI agents (Agent Clona for patient support and Assistant Lysa for doctor assistance) to provide insights and streamline healthcare operations. The platform aims to enhance patient care through advanced AI and comprehensive health data management, positioning itself as a wellness monitoring and change detection platform to avoid medical device classifications.

## User Preferences
- **Preferred communication style**: Simple, everyday language
- **Backend Language**: Python only - ALL backend code must be written in Python (FastAPI)
- **Frontend**: React/TypeScript (standard web practice)
- **Development Philosophy**: Building FULL FUNCTIONAL applications, NOT MVPs. All features must be completely functional and production-ready before delivery.

## System Architecture

### Frontend
The frontend is built with React, TypeScript, Vite, Wouter for routing, TanStack Query for data fetching, and Tailwind CSS for styling. It uses Radix UI and shadcn/ui for a clinical aesthetic, supporting role-based routing and context-based theming. AI Dashboard pages for video, audio, and alerts are temporarily disabled. The frontend routes Python AI endpoints (`/api/v1/video-ai/*`, `/api/v1/audio-ai/*`, `/api/v1/trends/*`, `/api/v1/alerts/*`) to the FastAPI backend on port 8000, and other endpoints to the Express server on port 5000.

### Backend
The backend consists of two fully operational services:
- **Node.js Express Backend (Port 5000)**: Handles Agent Clona chatbot (GPT-4o powered), appointments, calendar, consultations, pain tracking, symptom journal, voice analysis, baseline calculation, deviation detection, and risk scoring.
- **Python FastAPI Backend (Port 8000)**: Manages all AI deterioration detection endpoints, guided video and audio examinations, database interactions, and core authentication. It uses an async AI engine initialization with an `AIEngineManager` singleton and direct engine calls for dependency resolution.

### Core Features & Technical Implementations
- **AI Integration:** Utilizes OpenAI API (GPT-4o) for symptom analysis, wellness suggestions, doctor assistance, sentiment analysis, and medical entity extraction.
- **Personalization & Recommendation System:** Rule-based intelligent recommendation engine enhanced with RAG.
- **Advanced Drug Interaction Detection:** AI-powered system using GNN and NLP.
- **Real-Time Immune Function Monitoring:** AI-powered digital biomarker tracking from wearable data.
- **Environmental Pathogen Risk Mapping:** Location-based real-time environmental health monitoring.
- **Voice-Based Daily Followups:** Uses OpenAI Whisper for transcription and GPT-4 for analysis.
- **Assistant Lysa Receptionist Features:** AI-powered appointment management, email categorization, call log transcription, and automated reminders with Google Calendar and Gmail integration.
- **Secure Patient Record Sharing:** System with consent management and audit logging.
- **EHR & Wearable Integration:** FHIR-based integration with major EHR systems and popular wearable devices.
- **Video Consultations:** HIPAA-compliant video conferencing via Daily.co.
- **Pain Detection Camera System:** AI-powered facial analysis for tracking discomfort progression.
- **Home Clinical Exam Coach (HCEC):** AI-powered guided self-examination using OpenAI Vision.
- **Symptom Journal Enhancements:** Includes AI-powered respiratory rate analysis and weekly PDF reports.
- **Medication Side-Effect Predictor:** AI-powered system correlating patient symptoms with medication timelines.
- **Deterioration Prediction System:** Comprehensive health change detection with baseline calculation, Z-score analysis, anomaly detection, Bayesian risk modeling, and time-series trend analysis to generate a composite risk score.
- **ML Inference Infrastructure:** Self-hosted system with model registry, Redis caching, async thread pool inference, HIPAA-compliant audit logging, batch processing, and ONNX optimization. Includes pre-trained Clinical-BERT and custom LSTM models.
- **Guided Video Examination System:** A HIPAA-compliant 4-stage self-examination workflow (Eyes, Palm, Tongue, Lips) with 30-second prep screens, clinical-grade LAB color analysis for hepatic and anemia detection, disease-specific personalization, 5 RESTful API endpoints, and S3 encrypted storage for captured frames.
- **Guided Audio Examination System:** A HIPAA-compliant 4-stage audio recording workflow (Breathing, Coughing, Speaking, Reading) with 30-second prep screens, YAMNet ML classification for audio events, neurological metrics (speech fluency, voice weakness), disease-specific personalization, 5 RESTful API endpoints, and S3 encrypted storage for audio recordings.
- **AI Deterioration Detection System:** Full-stack SaaS platform with:
    - **Video AI Engine:** Extracts metrics like respiratory rate, skin pallor, sclera yellowness, facial swelling, head tremor, nail bed analysis.
    - **Facial Puffiness Score (FPS) System:** Comprehensive facial contour tracking using MediaPipe Face Mesh, providing 5 regional scores, baseline comparison, composite FPS, and asymmetry detection, integrated with disease-specific personalization.
    - **Comprehensive Respiratory Metrics System:** Measures Respiratory Variability Index (RVI), tracks patient baselines, performs temporal analytics, anomaly detection (Z-score), and advanced pattern detection.
    - **Disease-Specific Personalization System:** Multi-domain condition profiles for 12 chronic conditions (respiratory and edema-focused) with prioritized monitoring, expected patterns, and personalized thresholds.
    - **DeepLab V3+ Edema Segmentation System:** Medical-grade semantic segmentation for swelling/edema detection using DeepLab V3+ MobileNetV2 (Cityscapes pre-trained). Features lazy-loading initialization, frame-by-frame segmentation, regional body part analysis (face/upper, torso/hands, legs/feet), baseline comparison, swelling detection with severity grading, and integration with disease-specific personalization (heart failure, kidney disease, liver disease). Metrics automatically persisted to EdemaSegmentationMetrics database table during video analysis.
    - **Audio AI Engine:** Extracts metrics like breath cycles, speech pace, cough detection, wheeze detection, and voice quality.
    - **Trend Prediction Engine:** Performs baseline calculation, Z-score analysis, anomaly detection, Bayesian risk modeling, and time-series trend analysis to generate a composite risk score.
    - **Alert Orchestration Engine:** Provides multi-channel delivery (dashboard, email, SMS) with rule-based systems and HIPAA compliance.

### Security and Compliance
The platform is HIPAA-compliant, featuring AWS Cognito for authentication, BAA verification for all integrations, comprehensive audit logging, end-to-end encryption for video consultations, and strict PHI handling. It is strategically positioned as a General Wellness Product.

## External Dependencies

- **Authentication:** AWS Cognito.
- **Database:** Neon serverless PostgreSQL.
- **AI Services:** OpenAI API (GPT models), TensorFlow.js, PyTorch, HuggingFace Transformers, ONNX Runtime.
- **Caching:** Redis.
- **Communication:** Twilio API, AWS SES.
- **Video Conferencing:** Daily.co.
- **Cloud Services:** AWS S3, AWS Textract, AWS Comprehend Medical, AWS HealthLake, AWS HealthImaging, AWS HealthOmics.
- **Data Integration APIs:** PubMed E-utilities, PhysioNet WFDB, Kaggle API, WHO Global Health Observatory API, OpenWeatherMap API, Biobot Analytics.