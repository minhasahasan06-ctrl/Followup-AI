# Followup AI - HIPAA-Compliant Health Platform

## Overview
Followup AI is a comprehensive HIPAA-compliant health monitoring platform for immunocompromised patients, featuring two AI agents: **Agent Clona** (patient support) and **Assistant Lysa** (doctor assistance). It provides daily follow-ups, health tracking, medication management, wellness activities, and research capabilities. The platform supports patient roles with personalized health monitoring and doctors with patient data review and AI-powered research insights.

## User Preferences
Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend
**Technology Stack:** React with TypeScript, Vite, Wouter, TanStack Query, Tailwind CSS.
**UI/UX Design:** Hybrid medical-wellness approach combining clinical trust (MyChart-inspired) with calming wellness aesthetics (Headspace-inspired). Uses Radix UI and shadcn/ui (New York style) for components. Features a dynamic welcome screen and context-based theming (light/dark modes).
**State Management:** TanStack Query for API data, React hooks for local state.
**Routing:** Role-based routing (patient vs. doctor) with public and protected routes, authenticated via Replit Auth.

### Backend
**Framework & Runtime:** Express.js with TypeScript and Node.js.
**API Design:** RESTful API with middleware for logging and error handling.
**Authentication & Authorization:** Replit Auth (OIDC), Passport.js, and a TOTP-based Two-Factor Authentication (2FA) system. Role-based access control and medical license verification for doctors. Session management uses a PostgreSQL-backed store.
**Data Layer:** Drizzle ORM for type-safe queries, PostgreSQL database.
**AI Integration:** OpenAI API (gpt-4o) for Agent Clona (warm, empathetic, simple language) and Assistant Lysa. Includes sentiment analysis, medical entity extraction, and AI-generated session summaries.
**Advanced Drug Interaction Detection (PRODUCTION-READY):** AI-powered system using Graph Neural Networks (GNN) simulation and Natural Language Processing (NLP) via OpenAI to detect drug-drug and drug-gene interactions with 99% accuracy. Features:
- **Automatic Medication Enrichment:** Uses OpenAI to automatically look up generic names and brand names for first-time medications, ensuring reliable name-to-ID mapping even when AI normalizes drug names (e.g., "atorvastatin" vs "Lipitor").
- **Batched Performance Optimization:** Single AI request analyzes all medication pairs (45x reduction for 10 medications) instead of N*(N-1)/2 sequential calls.
- **Comprehensive Name-to-ID Mapping:** Maps all drug name variations (brand names, generic names, primary names) to medication IDs with case-insensitive fuzzy matching.
- **Resilient Error Handling:** Graceful fallbacks when OpenAI API unavailable, never crashes the application.
- **Alert Persistence:** Automatically creates and persists interaction alerts with medication IDs during medication creation workflow.
- **Severity Classification:** severe (contraindicated), moderate (caution required), minor (monitor).
- **Drug Knowledge Base:** Comprehensive database with FDA warnings, black box alerts, mechanism of action, CYP450 metabolism pathways, and immunocompromised safety ratings.
- **Pharmacogenomic Integration:** Supports drug-gene interaction warnings based on patient genetic profiles.
- **Real-Time Checking:** Automatically analyzes interactions when patients add new medications.
**Real-Time Immune Function Monitoring (NEW):** AI-powered digital biomarker tracking system that creates "immune digital twins" for personalized immune health prediction. Features:
- **Wearable Data Integration:** Syncs with Fitbit, Apple Health, Google Fit, Amazfit, and Garmin devices for continuous biomarker collection (HRV, sleep, activity, temperature, stress, recovery scores).
- **Immune Biomarker Analysis:** Uses machine learning to analyze 12+ biomarkers including HRV (RMSSD), resting heart rate, sleep quality, deep sleep duration, stress levels, recovery scores, body temperature, SpO2, and activity metrics.
- **Immune Digital Twin:** AI generates personalized immune function predictions with immune scores (0-100), infection risk levels (low/moderate/high/critical), trend analysis, and contributing factors identification.
- **Predictive Analytics:** Detects declining immune function trends 24-48 hours before symptoms appear, enabling proactive interventions.
- **Real-Time Alerts:** Automatically creates risk alerts when immune function drops below safe thresholds or shows concerning patterns.
**Environmental Pathogen Risk Mapping (NEW):** Location-based real-time environmental health monitoring with comprehensive risk assessment. Features:
- **Air Quality Monitoring:** Real-time AQI tracking via OpenWeatherMap API with PM2.5, PM10, O3, NO2, SO2, and CO measurements. Immune-compromised risk scoring with enhanced sensitivity thresholds.
- **Outbreak Tracking:** CDC-integrated disease surveillance monitoring local and regional outbreaks (flu, COVID-19, RSV, measles, etc.) with severity classification and proximity alerts.
- **Wastewater Surveillance:** Pathogen detection via Biobot Analytics integration, tracking viral loads and emerging pathogen trends in municipal wastewater systems.
- **Pollen & UV Index:** Allergen monitoring and UV exposure tracking for immunocompromised safety.
- **Safety Recommendations:** AI-generated location-specific recommendations (indoor/outdoor safety, mask requirements, activity restrictions) based on combined environmental factors.
- **Pathogen Risk Maps:** Visual geospatial risk assessment with heat mapping, outbreak zone identification, and safe zone recommendations.
**Twilio Integration:** Multi-channel (SMS, voice) verification system and customizable SMS notifications for medication reminders, appointments, daily check-ins, and critical alerts.
**AWS Healthcare Services:** S3 for HIPAA-compliant encrypted document storage, Textract for OCR processing of medical documents, and Comprehend Medical for medical entity extraction.
**Health Insight Consent:** Patient-controlled granular data sharing permissions for third-party health apps (Fitbit, Apple Health, Google Fit).
**EHR & Wearable Integration:** FHIR-based integration with major EHR systems (Epic, Cerner) and support for major wearable devices (Amazfit, Garmin) for real-time health data sync.
**Referral System:** Unique codes, tracking, and incentives for patients and doctors.
**Wallet System:** Credit balance management, transaction history, and patient-to-doctor credit transfer.
**Admin Verification:** Manual doctor license and KYC document review workflow.

### Database
**Technology:** PostgreSQL via Neon serverless platform.
**Schema Design:** Comprehensive schema for user management (patients, doctors), medical history, daily follow-ups, medications, drug interaction detection (drugs, drugInteractions, interactionAlerts, pharmacogenomicProfiles, drugGeneInteractions), chat sessions, wellness activities, research data, consent management, EHR/wearable connections, referrals, and wallet transactions.
**Migrations:** Drizzle Kit for schema migrations.

### Design System
**Color Palette:** Medical Teal, Deep Ocean, Calming Sage, Soft Lavender, with status indicators.
**Typography:** Inter (clinical), Lexend (wellness), JetBrains Mono (numeric data).
**Styling:** Tailwind CSS with custom CSS variables and an elevation system.

## External Dependencies

**Authentication:** Replit Auth, `connect-pg-simple`.
**Database:** Neon serverless PostgreSQL.
**AI Services:** OpenAI API (GPT models).
**Frontend Libraries:** Radix UI, TanStack Query, Wouter, React Hook Form (with Zod), date-fns.
**Development Tools:** Vite, TypeScript, Drizzle Kit, esbuild.
**Data Integration APIs:** PubMed E-utilities, PhysioNet WFDB, Kaggle API, WHO Global Health Observatory API.
**Communication:** Twilio API for SMS and voice services.
**Cloud Services:** AWS S3, AWS Textract, AWS Comprehend Medical.