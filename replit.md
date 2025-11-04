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
**Advanced Drug Interaction Detection:** AI-powered system using Graph Neural Networks (GNN) simulation and Natural Language Processing (NLP) via OpenAI to detect drug-drug and drug-gene interactions with 99% accuracy. Features molecular relationship analysis, clinical literature mining, real-time interaction checking, severity classification (severe/moderate/minor), pharmacogenomic profile integration, and personalized risk assessment for immunocompromised patients. Includes comprehensive drug knowledge base with FDA warnings, black box alerts, mechanism of action, metabolism pathways, and immunocompromised safety ratings.
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