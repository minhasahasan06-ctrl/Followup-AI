# Followup AI - HIPAA-Compliant Health Platform

## Overview
Followup AI is a HIPAA-compliant health monitoring platform for immunocompromised patients. It provides personalized health tracking, medication management, and wellness activities using AI agents (Agent Clona for patient support and Assistant Lysa for doctor assistance). The platform aims to enhance patient care through advanced AI and comprehensive health data management, positioning itself as a wellness monitoring and change detection system to avoid medical device classifications, while offering insights and streamlining healthcare operations.

## User Preferences
- **Preferred communication style**: Simple, everyday language
- **Backend Language**: Python only - ALL backend code must be written in Python (FastAPI)
- **Frontend**: React/TypeScript (standard web practice)
- **Development Philosophy**: Building FULL FUNCTIONAL applications, NOT MVPs. All features must be completely functional and production-ready before delivery.

## System Architecture

### Frontend
The frontend is built with React, TypeScript, Vite, Wouter for routing, TanStack Query for data fetching, and Tailwind CSS for styling. It utilizes Radix UI and shadcn/ui for a clinical aesthetic, supporting role-based routing and context-based theming. It routes Python AI endpoints (`/api/v1/video-ai/*`, `/api/v1/audio-ai/*`, `/api/v1/trends/*`, `/api/v1/alerts/*`) to the FastAPI backend on port 8000, and other endpoints to the Express server on port 5000.

### Backend
The backend comprises two services:
- **Node.js Express Backend (Port 5000)**: Handles Agent Clona chatbot (GPT-4o powered), appointments, calendar, consultations, pain tracking, symptom journal, voice analysis, baseline calculation, deviation detection, and risk scoring.
- **Python FastAPI Backend (Port 8000)**: Manages all AI deterioration detection endpoints, guided video and audio examinations, database interactions, and core authentication. It uses an async AI engine initialization with an `AIEngineManager` singleton.

### Core Features & Technical Implementations
- **AI Integration:** Leverages OpenAI API (GPT-4o) for symptom analysis, wellness suggestions, doctor assistance, sentiment analysis, and medical entity extraction.
- **Personalization & Recommendation System:** Rule-based recommendation engine enhanced with RAG.
- **Real-Time Immune Function Monitoring:** AI-powered digital biomarker tracking from wearable data.
- **Voice-Based Daily Followups:** Uses OpenAI Whisper for transcription and GPT-4 for analysis.
- **Assistant Lysa Receptionist Features:** AI-powered appointment management, email categorization, call log transcription, and automated reminders with Google Calendar and Gmail integration.
- **Secure Patient Record Sharing:** System with consent management and audit logging.
- **EHR & Wearable Integration:** FHIR-based integration with major EHR systems and popular wearable devices.
- **Video Consultations:** HIPAA-compliant video conferencing via Daily.co.
- **Home Clinical Exam Coach (HCEC):** AI-powered guided self-examination using OpenAI Vision.
- **Deterioration Prediction System:** Comprehensive health change detection with baseline calculation, Z-score analysis, anomaly detection, Bayesian risk modeling, and time-series trend analysis to generate a composite risk score.
- **ML Inference Infrastructure:** Self-hosted system with model registry, Redis caching, async inference, HIPAA-compliant audit logging, batch processing, and ONNX optimization. Includes pre-trained Clinical-BERT and custom LSTM models.
- **Guided Video Examination System:** A HIPAA-compliant 4-stage self-examination workflow (Eyes, Palm, Tongue, Lips) with clinical-grade LAB color analysis for hepatic and anemia detection, disease-specific personalization, and S3 encrypted storage.
- **Guided Audio Examination System:** A HIPAA-compliant 4-stage audio recording workflow (Breathing, Coughing, Speaking, Reading) with YAMNet ML classification, neurological metrics, disease-specific personalization, and S3 encrypted storage.
- **AI Deterioration Detection System:**
    - **Video AI Engine:** Extracts metrics like respiratory rate, skin pallor, sclera yellowness, facial swelling, head tremor, nail bed analysis.
    - **Facial Puffiness Score (FPS) System:** Comprehensive facial contour tracking using MediaPipe Face Mesh, providing regional scores, baseline comparison, composite FPS, and asymmetry detection.
    - **Comprehensive Respiratory Metrics System:** Measures Respiratory Variability Index (RVI), tracks patient baselines, performs temporal analytics, anomaly detection, and advanced pattern detection.
    - **Disease-Specific Personalization System:** Multi-domain condition profiles for 12 chronic conditions with prioritized monitoring, expected patterns, and personalized thresholds.
    - **DeepLab V3+ Edema Segmentation System:** Medical-grade semantic segmentation for swelling/edema detection using DeepLab V3+ MobileNetV2. Features 8-Region Anatomical Segmentation, Confidence Scoring, S3 Encrypted Storage, Baseline Management System, Advanced Regional Analysis (expansion percentages, asymmetry), Fine-Tuning Pipeline, Database Persistence, and Disease-Specific Personalization.
    - **Audio AI Engine:** Extracts metrics like breath cycles, speech pace, cough detection, wheeze detection, and voice quality.
    - **Trend Prediction Engine:** Performs baseline calculation, Z-score analysis, anomaly detection, Bayesian risk modeling, and time-series trend analysis to generate a composite risk score.
    - **Alert Orchestration Engine:** Provides multi-channel delivery (dashboard, email, SMS) with rule-based systems and HIPAA compliance.
- **Behavior AI Analysis System:** Comprehensive multi-modal deterioration detection through behavioral pattern analysis, digital biomarkers, cognitive testing, and sentiment analysis.
    - **Database Architecture:** 9-table PostgreSQL schema tracking behavioral check-ins, aggregated metrics, digital biomarkers, cognitive test results, sentiment analysis, multi-modal risk scores, deterioration trends, and alerts.
    - **ML Models Infrastructure (Python):** Three-model ensemble system including a Transformer Encoder (PyTorch) for temporal sequences, XGBoost for feature-based prediction, and DistilBERT for sentiment analysis.
    - **Behavioral Metrics Service:** Tracks 9 key behavioral patterns (e.g., check-in consistency, medication adherence, app engagement, avoidance patterns, sentiment trends).
    - **Digital Biomarkers Service:** Analyzes phone/wearable data for 10 digital biomarkers (e.g., step count, sedentary duration, circadian rhythm stability, sleep-wake irregularity).
    - **Cognitive Test Service:** Administers and scores 5 weekly micro-tests (e.g., reaction time, memory recall, pattern recognition).
    - **Sentiment Analysis Service:** DistilBERT-powered analysis of text inputs extracting 11 language biomarkers (e.g., polarity score, stress keywords, help-seeking detection).
    - **Risk Scoring Engine:** Multi-modal fusion engine combining 4 component risk scores (behavioral, digital, cognitive, sentiment) with configurable weights to calculate a composite risk score and identify contributing factors.
    - **Deterioration Trend Engine:** Statistical temporal pattern detection using scipy linear regression to detect trend types (e.g., declining engagement, mobility drop, cognitive decline, sentiment deterioration).
    - **FastAPI Endpoints:** 7 production RESTful endpoints for check-ins, digital biomarkers, cognitive tests, sentiment analysis, risk scores, trends, and dashboard overview.
    - **Data Sources Integration:** Utilizes app interaction data, phone/wearable sensor data, cognitive test results, and text/audio inputs.
    - **Production Features:** Includes comprehensive database persistence, lazy model loading, rule-based fallbacks, error handling, HIPAA-compliant storage, background task processing, and multi-channel alert delivery readiness.

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