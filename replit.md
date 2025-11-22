# Followup AI - HIPAA-Compliant Health Platform

## Overview
Followup AI is a HIPAA-compliant health monitoring platform designed for immunocompromised patients. It provides personalized health tracking, medication management, and wellness activities, leveraging AI agents (Agent Clona for patient support and Assistant Lysa for doctor assistance) to enhance patient care. The platform focuses on advanced AI and comprehensive health data management to offer insights and streamline healthcare operations, aiming to be a wellness monitoring and change detection system.

## User Preferences
- **Preferred communication style**: Simple, everyday language
- **Backend Language**: Python only - ALL backend code must be written in Python (FastAPI)
- **Frontend**: React/TypeScript (standard web practice)
- **Development Philosophy**: Building FULL FUNCTIONAL applications, NOT MVPs. All features must be completely functional and production-ready before delivery.

## System Architecture

### Frontend
The frontend is built with React, TypeScript, Vite, Wouter for routing, TanStack Query for data fetching, and Tailwind CSS for styling. It utilizes Radix UI and shadcn/ui for a clinical aesthetic, supporting role-based routing and context-based theming. It routes Python AI endpoints to the FastAPI backend on port 8000 and other endpoints to an Express server on port 5000.

### Backend
The backend comprises two services:
- **Node.js Express Backend (Port 5000)**: Manages the Agent Clona chatbot (GPT-4o powered), appointments, calendar, consultations, pain tracking, symptom journal, voice analysis, baseline calculation, deviation detection, and risk scoring.
- **Python FastAPI Backend (Port 8000)**: Handles all AI deterioration detection endpoints, guided video and audio examinations, mental health questionnaires, database interactions, and core authentication. It uses an async AI engine initialization with an `AIEngineManager` singleton. The Python backend loads heavy ML models at startup (TensorFlow, MediaPipe, YAMNet), which can take 30-60 seconds.

### Core Features & Technical Implementations
- **AI Integration:** Leverages OpenAI API (GPT-4o) for symptom analysis, wellness suggestions, doctor assistance, sentiment analysis, and medical entity extraction.
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
- **Guided Video Examination System:** A HIPAA-compliant 4-stage self-examination workflow (Eyes, Palm, Tongue, Lips) with clinical-grade LAB color analysis, disease-specific personalization, and S3 encrypted storage. Extracts metrics like respiratory rate, skin pallor, sclera yellowness, facial swelling, head tremor, and nail bed analysis.
- **Facial Puffiness Score (FPS) System:** Comprehensive facial contour tracking using MediaPipe Face Mesh, providing regional scores, baseline comparison, composite FPS, and asymmetry detection.
- **Comprehensive Respiratory Metrics System:** Measures Respiratory Variability Index (RVI), tracks patient baselines, performs temporal analytics, anomaly detection, and advanced pattern detection.
- **DeepLab V3+ Edema Segmentation System:** Medical-grade semantic segmentation for swelling/edema detection using DeepLab V3+ MobileNetV2, featuring 8-Region Anatomical Segmentation, Confidence Scoring, and Advanced Regional Analysis.
- **Guided Audio Examination System:** A HIPAA-compliant 4-stage audio recording workflow (Breathing, Coughing, Speaking, Reading) with YAMNet ML classification, neurological metrics, disease-specific personalization, and S3 encrypted storage. Extracts metrics like breath cycles, speech pace, cough detection, wheeze detection, and voice quality.
- **Trend Prediction Engine:** Performs baseline calculation, Z-score analysis, anomaly detection, Bayesian risk modeling, and time-series trend analysis to generate a composite risk score.
- **Alert Orchestration Engine:** Provides multi-channel delivery (dashboard, email, SMS) with rule-based systems and HIPAA compliance.
- **Behavior AI Analysis System:** Comprehensive multi-modal deterioration detection through behavioral pattern analysis, digital biomarkers, cognitive testing, and sentiment analysis. Utilizes an ensemble of ML models (Transformer Encoder, XGBoost, DistilBERT) and various services for behavioral metrics, digital biomarkers, cognitive tests, and sentiment analysis, culminating in a risk scoring engine and deterioration trend engine.
- **Gait Analysis System (HAR-based):** Open-source gait analysis using MediaPipe Pose and HAR datasets, extracting over 40 gait parameters.
- **Accelerometer Tremor Analysis System:** Tremor detection from phone accelerometer data using FFT-based signal processing.
- **Drug-Drug Interaction Detection System:** Medication adherence system with RxNorm-standardized drug library and clinical interaction detection, including RxNorm API integration and real-time interaction detection.
- **Automatic Drug Normalization System:** On-demand medication normalization against RxNorm API. When patients enter medications, the system automatically normalizes names against the RxNorm database, creates standardized drug records, and links medications via drug_id foreign keys. Features non-blocking normalization (5s timeout), HIPAA-compliant audit logging, confidence scoring, and graceful degradation when Python service unavailable. Eliminates need for pre-seeded drug database.
- **PainTrack Platform:** Chronic pain tracking system with dual-camera video capture, self-reported VAS pain slider, medication tracking, and patient notes.
- **Mental Health AI Dashboard Integration:** Integrated questionnaire system (PHQ-9, GAD-7, PSS-10) with AI-powered GPT-4o analysis, crisis detection, severity scoring, and cluster analysis.

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