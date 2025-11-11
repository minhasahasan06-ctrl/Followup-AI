# Followup AI - HIPAA-Compliant Health Platform

## Overview
Followup AI is a comprehensive HIPAA-compliant health monitoring platform for immunocompromised patients. It integrates two AI agents: **Agent Clona** for patient support and **Assistant Lysa** for doctor assistance. The platform offers daily follow-ups, health tracking, medication management, wellness activities, and research capabilities, aiming to provide personalized health monitoring for patients and AI-powered insights for doctors.

## User Preferences
Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend
The frontend uses React with TypeScript, Vite, Wouter, TanStack Query, and Tailwind CSS. The UI/UX blends a clinical, trustworthy aesthetic with calming wellness elements, utilizing Radix UI and shadcn/ui. It supports role-based routing and context-based theming.

### Backend
The backend is built with Express.js and Node.js in TypeScript, featuring a RESTful API. Authentication and authorization are handled via AWS Cognito User Pools with JWTs and role-based access control (patient/doctor), including medical license verification for doctors and optional TOTP-based 2FA. Data persistence uses Drizzle ORM with a PostgreSQL database.

**Core Features:**
-   **AI Integration:** Leverages OpenAI API (gpt-4o) for Agent Clona (patient support) and Assistant Lysa (doctor assistance), including sentiment analysis, medical entity extraction, and AI-generated session summaries.
-   **Personalization & Recommendation System:** A rule-based intelligent recommendation engine with RAG-enhanced AI agent personalization. It learns user preferences, provides tailored suggestions for patients and clinical insights for doctors, tracks habits, and monitors doctor wellness.
-   **Advanced Drug Interaction Detection:** An AI-powered system using GNN and NLP (via OpenAI) to detect drug-drug and drug-gene interactions with high accuracy, including automatic medication enrichment, batched processing, and severity classification.
-   **Real-Time Immune Function Monitoring:** An AI-powered digital biomarker tracking system that integrates wearable data (Fitbit, Apple Health, etc.) to create "immune digital twins," predicting immune function and infection risk.
-   **Environmental Pathogen Risk Mapping:** Provides location-based real-time environmental health monitoring, including air quality, outbreak tracking (CDC integration), wastewater surveillance (Biobot Analytics), pollen, and UV index, offering AI-generated safety recommendations.
-   **Voice-Based Daily Followups:** Utilizes OpenAI Whisper for transcription and GPT-4 for analysis of 1-minute voice recordings to extract health data and provide empathetic AI responses.
-   **Assistant Lysa Receptionist Features:** An AI-powered receptionist for doctors, managing appointments (CRUD, conflict detection, authorization), doctor availability, email threads (categorization, priority), call logs (tracking, transcription), and automated appointment reminders.
-   **Health Insight Consent:** Allows patients granular control over data sharing permissions with third-party health apps.
-   **EHR & Wearable Integration:** FHIR-based integration with major EHR systems (Epic, Cerner) and popular wearable devices (Amazfit, Garmin).

## External Dependencies

-   **Authentication:** Replit Auth, AWS Cognito.
-   **Database:** Neon serverless PostgreSQL.
-   **AI Services:** OpenAI API (GPT models).
-   **Communication:** Twilio API (SMS, voice), AWS SES (transactional emails).
-   **Cloud Services:** AWS S3 (storage), AWS Textract (OCR), AWS Comprehend Medical (medical entity extraction), AWS HealthLake (FHIR data store), AWS HealthImaging (medical imaging), AWS HealthOmics (genomic data analysis).
-   **Data Integration APIs:** PubMed E-utilities, PhysioNet WFDB, Kaggle API, WHO Global Health Observatory API, OpenWeatherMap API, Biobot Analytics.