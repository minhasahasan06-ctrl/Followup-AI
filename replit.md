# Followup AI - HIPAA-Compliant Health Platform

## Overview
Followup AI is a HIPAA-compliant health monitoring platform for chronic care patients. It offers personalized health tracking, medication management, and wellness activities, leveraging AI agents (Agent Clona for patient support and Assistant Lysa for doctor assistance). The platform aims to provide a comprehensive wellness monitoring and change detection system, delivering fully functional, production-ready applications to transform healthcare for individuals managing ongoing health conditions.

## User Preferences
- **Preferred communication style**: Simple, everyday language
- **Backend Language**: Python only - ALL backend code must be written in Python (FastAPI)
- **Frontend**: React/TypeScript (standard web practice)
- **Development Philosophy**: Building FULL FUNCTIONAL applications, NOT MVPs. All features must be completely functional and production-ready before delivery.
- **Code Quality**: Avoid duplicate code - use proper code reuse, modular design, and shared components/utilities.

## System Architecture

### Frontend
The frontend uses React, TypeScript, Vite, Wouter for routing, TanStack Query for data fetching, and Tailwind CSS for styling. It integrates Radix UI and shadcn/ui for a clinical aesthetic, supporting role-based routing and context-based theming. It routes Python AI endpoints to a FastAPI backend (port 8000) and other endpoints to an Express server (port 5000).

### Backend
The backend consists of a Node.js Express server (Port 5000) handling chat, appointments, calendar, consultations, pain tracking, symptom journaling, and risk scoring. A Python FastAPI server (Port 8000) manages AI deterioration detection, guided examinations, mental health questionnaires, database interactions, authentication, and an async AI engine with a singleton manager for ML models.

### Core Features & Technical Implementations
- **AI Integration**: Utilizes OpenAI API for PHI detection, symptom extraction, and clinical reasoning.
- **Centralized OpenAI Client** (`app/services/openai_client.py`): Production-grade OpenAI wrapper with PHI detection (SSN, MRN, email, phone, credit card patterns), BAA/ZDR runtime enforcement that blocks production without proper flags, audit logging for all API calls, and embedding standardization (text-embedding-3-small, 1536D, v1.0.0).
- **Long-Term Memory Service** (`app/services/memory_service.py`, `app/services/memory_db.py`): pgvector-based semantic memory with cosine similarity search, embedding model/version tracking, access_count/last_accessed_at metrics, and async PostgreSQL operations.
- **LlamaIndex Integration** (`app/services/llama_memory_service.py`): PostgresVectorStore adapter wrapping MemoryService for retrieval-augmented generation, agent/patient/memory_type filtering.
- **ML Observability** (`app/services/ml_observability.py`): MetricHistogram for latency p50/p95/p99, MetricCounter for events, MetricGauge for current values, comprehensive tracking of retrieval, embedding, and LLM operations.
- **ML Alerting** (`app/services/ml_alerting.py`): Threshold-based alerts for retrieval failure rate (>10%), latency SLA violations (retrieval >1000ms, embedding >2000ms), API error rate (>5%), with background monitoring.
- **ML Governance** (`app/models/ml_governance_models.py`, `app/services/ml_governance_service.py`): Clinical model validation requirements, research-only flagging, human approval gates (technical/QA/clinical/regulatory/executive levels), data provenance tracking with lineage graphs, embedding standardization enforcement.
- **Dual-Write Migration** (`app/services/dual_write_migration.py`): Shadow-run pattern for migrating from legacy to new vector storage with precision@k metrics collection, latency comparison, and migration readiness checks.
- **Immutable Audit Log** (`app/services/immutable_audit_log.py`): HIPAA-compliant append-only audit logging with cryptographic hash chain for tamper detection, retention policy (6 years), and patient access reports per 45 CFR 164.528.
- **MONAI Medical Imaging** (`app/services/monai_imaging.py`, `app/routes/imaging.py`): Medical imaging inference service with DICOM/NIfTI PHI stripping, UNet segmentation, GPU containerization via Docker (`docker/monai/`), and comprehensive audit logging.
- **LangGraph Orchestration** (`app/services/langgraph_orchestration.py`): Graph-based agent orchestration pilot for Clona patient support with state machine pattern, conditional routing, and migration assessment tools.
- **PHI Detection Service**: GPT-4o-based HIPAA-compliant PHI detection and medical entity extraction.
- **Deterioration Prediction System**: Statistical and AI models for health change detection and risk scoring.
- **ML Inference Infrastructure**: Self-hosted, HIPAA-compliant system with model registry, Redis caching, and ONNX optimization.
- **Alert Orchestration Engine**: Multi-channel (dashboard, email, SMS) rule-based alert delivery.
- **Behavior AI Analysis System**: Multi-modal deterioration detection using ensemble ML models for behavioral patterns, digital biomarkers, cognitive testing, and sentiment analysis.
- **Followup Autopilot Engine**: ML-powered adaptive follow-up system optimizing timing and generating adaptive tasks.
- **Unified Medication System**: Production-grade medication management with AI assistance and conflict detection, supporting doctor-only prescription authorization.
- **Multi-Agent Communication System**: Facilitates communication between AI agents, users, and providers via a Message Router, Memory Service, and `MessageEnvelope` protocol, with an Agent Hub UI.
- **ML Training Infrastructure**: Production-grade system with patient consent controls for granular data types, including a patient-facing Consent UI and Data Extraction Pipeline.
- **Background Scheduler**: APScheduler-based jobs on FastAPI for tasks like risk scoring, data quality, ETL, and ML inference.
- **Epidemiology Analytics Platform**: Data warehouse and surveillance analytics with SQLAlchemy models, nightly ETL, and k-anonymity privacy protection.
- **Research Center + Medical NLP**: Production-grade research data platform with SQLAlchemy models, artifact/dataset storage, PHI redaction via GPT-4o, AI-powered Q&A, and k-anonymity protection.
- **Habit Tracker + Behavior AI**: Production-grade habit tracking with gamification, AI coaching, and integration with Behavior AI for risk prediction.
- **Tinker Thinking Machine Integration**: HIPAA-compliant external AI analysis platform with a Privacy Firewall, K-Anonymity Enforcement, and Service Orchestrator for cohort analysis and drift detection.
- **Genius Features**: Includes Research Genius for study design, Clinical Operations Genius for alert management, and Patient Genius for personalized check-ins.
- **Extended Profile System**: Production-grade medical information management for patients and doctors.
- **HIPAA Compliance & Access Control**: Unified Access Control Service, `HIPAAAuditLogger`, and route-level access control with `AccessScope` and `PHICategory` enums.
- **Authentication Flow**: Stytch-based passwordless authentication with Magic Links and SMS OTP, session-based cookies for Express, and M2M Client Credentials for inter-service communication.
- **Conversational AI Voice System**: Production-grade voice conversation pipeline with real-time ASR (OpenAI Whisper) and TTS, session management, and intelligent interruption handling.
- **Medical Emergency Detection System**: Real-time universal red flag detection service integrated with chat, using pattern-based, keyword, and AI-powered analysis, with an automated escalation flow to doctors (13 RedFlagCategories including TOXICOLOGICAL).
- **Feature Flag System**: Production-grade feature flag management for controlled rollouts and A/B testing, configurable via JSON and environment variables, with user and role-based overrides.
- **Voice Session Orchestration**: WebSocket-based voice streaming at 16kHz sample rate with MediaStream API, useVoiceSession hook for frontend integration, and VoiceCallControls UI component.
- **Daily.co Video Console**: Telemedicine video consultations with useDailyCall hook and VideoConsole component (mute, camera toggle, screen share, quality indicators, duration tracking).
- **Communication Preferences System**: User-configurable communication method preferences (voice, video, chat) with availability scheduling, Do Not Disturb mode, and emergency override.
- **Action Cards System**: Voice-triggered task cards with medication reminders, symptom checks, quick confirmations, and emergency escalation cards with voice response matching.
- **Voice Consent Service**: HIPAA-compliant consent management with version tracking, expiration, revocation, and audit logging for voice/video recording.

### Testing & CI Infrastructure
- **Unit Tests** (`tests/python/`): 77 tests covering PHI detection, BAA/ZDR enforcement, memory operations, governance, observability, and alerting.
- **Integration Tests** (`tests/python/test_integration_rag.py`): End-to-end RAG pipeline tests, PHI gating flows, dual-write migration validation.
- **Performance Tests** (`tests/python/test_performance.py`): Latency SLA validation (p50/p95/p99), ingestion throughput, observability overhead.
- **CI Pipeline** (`.github/workflows/ml-tests.yml`): GitHub Actions for unit/integration/performance tests, embedding standardization enforcement, PHI detection validation, governance checks.
- **Re-embedding Script** (`scripts/reembed_memories.py`): Batch migration script with checkpointing, dry-run mode, and OpenAI API integration.
- **Embedding Standardization Check** (`scripts/check_embedding_standardization.py`): CI enforcement for embedding model/version compliance.

### Security and Compliance
The platform ensures HIPAA compliance through Stytch for passwordless authentication, HttpOnly session cookies, M2M tokens, comprehensive audit logging, end-to-end encryption, strict PHI handling, explicit doctor-patient assignment authorization, and Google Authenticator TOTP for Admin ML Training Hub.

### Development Environment Guardrails
The development environment includes HIPAA compliance guardrails to prevent PHI exposure:
- **Config Guard** (`server/config_guard.ts`): Fail-closed startup checks that block production identifiers (GCP projects, Neon hosts, Auth0 domains, live Stripe keys). Application exits immediately with generic error if production patterns detected.
- **Safe Logger** (`server/safe_logger.ts`): PHI-redacting logger with 15+ pattern types (emails, phones, SSNs, MRNs, JWT tokens, API keys). All sensitive data replaced with `[REDACTED-*]` markers.
- **Production Identifiers** (`prod_identifiers.json`): Configurable blocklist of production identifiers. PLACEHOLDER values require review by authorized engineer.
- **CI/CD Security** (`.github/workflows/ci_checks.yml`): GitHub Actions with gitleaks secret scanning, production identifier detection, and dependency auditing.
- **Development Seeding** (`server/dev_seed.ts`): Synthetic data seeder with config guard integration for safe development.
- **Key Environment Variables**: `REPLIT_DEV_ONLY=true` (required for dev mode), `NODE_ENV=development` (enforced).
- **Documentation**: `DEV_README.md` (no-PHI rules), `AUDIT_SUMMARY.md` (compliance controls).

## External Dependencies

- **Authentication**: Stytch (Magic Links, SMS OTP, M2M)
- **Database**: Neon serverless PostgreSQL
- **AI Services**: OpenAI API, TensorFlow.js, PyTorch, HuggingFace Transformers, ONNX Runtime
- **Caching**: Redis
- **Communication**: Stytch (SMS, Email)
- **Video Conferencing**: Daily.co
- **Cloud Services**: Google Cloud Platform (GCS, Document AI, Healthcare NLP API, Cloud KMS, Healthcare API FHIR)