# Followup AI - HIPAA-Compliant Health Platform

## Overview

Followup AI is a comprehensive health monitoring platform designed specifically for immunocompromised patients. The application features two AI agents: **Agent Clona** (patient support AI) and **Assistant Lysa** (doctor assistance AI). The platform provides daily follow-ups, health tracking, medication management, wellness activities, and research capabilities while maintaining HIPAA compliance.

The system supports two user roles: patients who receive personalized health monitoring and wellness support, and doctors who can review patient data and generate AI-powered research insights.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture

**Technology Stack:**
- React with TypeScript for type-safe component development
- Vite as the build tool and development server
- Wouter for lightweight client-side routing
- TanStack Query (React Query) for server state management and caching
- Tailwind CSS for utility-first styling with custom design system

**UI Component System:**
- Radix UI primitives for accessible, unstyled components
- shadcn/ui component library (New York style) for pre-built components
- Custom theme system supporting light/dark modes
- Design follows hybrid medical-wellness approach combining clinical trust (MyChart-inspired) with calming wellness aesthetics (Headspace-inspired)

**Component Architecture:**
- Atomic design pattern with reusable UI components in `/client/src/components/ui`
- Feature-specific components for health metrics, medications, chat, wellness activities
- Dynamic Welcome Screen component for patient dashboard:
  - Time-based personalized greetings (morning, afternoon, evening, night)
  - 10 rotating inspirational quotes for motivation
  - 8 calming nature images from Unsplash for visual serenity
  - Health insights customized to patient data
  - Music notes for ambient suggestions
  - Content varies on each visit for continued engagement
  - Smooth animations using framer-motion
- Context-based theming with `ThemeProvider`
- Sidebar navigation with role-based menu items

**State Management:**
- TanStack Query for all API data fetching and caching
- Local component state with React hooks
- Custom hooks for authentication (`useAuth`) and mobile detection
- Query invalidation patterns for optimistic UI updates

**Routing Strategy:**
- Public routes (unauthenticated access):
  - `/` - Patient-focused landing page with app download options
  - `/doctor-portal` - Doctor-focused landing page with professional features
- Authentication flow: Landing/Doctor Portal → Sign Up/Login → Replit Auth → Role Selection → Dashboard
- Role-based routing: separate router components for patients vs. doctors
- Patient routes: Dashboard (with Dynamic Welcome), Chat, Previous Sessions (Medical History), Wellness, Counseling, App Connections (Consent Management), EHR Integrations, Wearable Devices, Referrals, Wallet, Files, Two-Factor Auth, Profile
- Doctor routes: Doctor Dashboard, Patient Review (with Chat Sessions tab), Research Center, Chat, Counseling, Referrals, Wallet, Admin Verification (admin only), Two-Factor Auth, Profile
- Protected routes requiring authentication via Replit Auth

**Landing Page Features:**
- Separate "Patient Sign Up" and "Patient Login" buttons
- iOS and Android app download section
- Link to Doctor Portal for healthcare providers
- Feature showcase for both patient and doctor experiences

**Doctor Portal Features:**
- Dedicated sign-up and login flows for doctors
- Medical license verification information
- Professional features showcase (Assistant Lysa, Research Center, etc.)
- Link back to patient landing page

### Backend Architecture

**Framework & Runtime:**
- Express.js server with TypeScript
- Node.js runtime with ES modules
- Development mode uses tsx for TypeScript execution
- Production build uses esbuild for server bundling

**API Design:**
- RESTful API endpoints under `/api` prefix
- Route organization in `server/routes.ts`
- Middleware for request logging and error handling
- Session-based authentication with HIPAA-compliant security

**Authentication & Authorization:**
- Replit Auth with OpenID Connect (OIDC) integration
- Passport.js strategy for authentication flow
- Two-Factor Authentication (2FA) system:
  - TOTP-based authentication using speakeasy library
  - QR code generation for easy mobile app setup (Google Authenticator, Authy, etc.)
  - 10 backup codes for account recovery
  - Enable/disable capability for both patients and doctors
  - Verification flow integrated into settings page
- Role-based access control (patient vs. doctor)
- Session management with PostgreSQL-backed session store
- Medical license verification for doctor accounts

**Data Layer:**
- Drizzle ORM for type-safe database queries
- Schema-first design with `shared/schema.ts`
- Separate storage abstraction layer (`server/storage.ts`) for data operations
- Transaction support for complex operations

**AI Integration:**
- OpenAI API (gpt-4o model) for chat functionality with Agent Clona and Assistant Lysa
- Agent Clona personality optimized for elderly immunocompromised patients:
  - Warm, friendly, and conversational tone (like a caring friend)
  - Always greets users warmly and asks how they're feeling
  - Empathetic and encouraging - celebrates small health wins
  - Uses simple, everyday language instead of medical jargon
  - Patient and understanding for elderly users
  - Engaging in genuine dialogue, not just Q&A
  - Ends conversations with warm wishes and gentle reminders
- Sentiment analysis using `sentiment` library for emotional context
- Context-aware responses based on user role (patient vs. doctor)
- Medical entity extraction from chat messages (symptoms, medications)
- AI-generated session summaries for medical history tracking
- Automatic health insights generation from conversation patterns

**Public Medical Data Integration:**
- PubMed E-utilities API integration for accessing research articles and medical literature
- PhysioNet dataset access for physiological signal databases (ECG, clinical notes, etc.)
- Kaggle API integration for public medical datasets with authentication
- WHO Global Health Observatory (GHO) API for accessing 2000+ health indicators including mortality rates, infectious diseases, environmental health, and global health statistics
- Training dataset management system for storing and organizing medical research data
- Doctor-exclusive feature for AI model training and research

**Health Insight Consent Management:**
- Patient-controlled data sharing permissions for third-party health apps
- Granular control over data types shared (heart rate, blood pressure, steps, medications, etc.)
- Configurable sharing frequency (real-time, daily, weekly)
- Consent tracking with creation, revocation, and sync status
- Support for popular health apps (Fitbit, Apple Health, Google Fit, MyFitnessPal, etc.)
- HIPAA-compliant audit trail for all consent changes

**EHR Integration System:**
- FHIR-based integration with major EHR systems (Epic, Oracle Cerner, Athena Health, eClinicalWorks, Allscripts, AdvancedMD, Meditech, NextGen, DrChrono)
- OAuth 2.0 authentication flow for secure EHR connections
- Patient-controlled EHR data access and sync management
- Connection status tracking with last sync timestamps
- Disconnect/revoke capability for data privacy

**Wearable Device Integration:**
- Support for major wearable brands (Amazfit, Garmin, Whoop, Samsung, Eko)
- Real-time health data sync for heart rate, blood pressure, steps, sleep, SpO2, ECG
- OAuth-based device authentication
- Connection management with device status tracking
- Automatic data refresh from connected devices

**Referral System:**
- Unique referral code generation for both patients and doctors
- Referral tracking with signup status monitoring
- 1-month free trial incentive for both referrer and referee
- Shareable referral links with copy-to-clipboard functionality
- Referral history and analytics

**Wallet & Credit System:**
- Credit balance management for patients and doctors
- Transaction history with detailed tracking
- Patient-to-doctor credit transfer for consultations (20 credits per session)
- Doctor credit withdrawal via Stripe integration
- Real-time credit updates with optimistic UI

**Admin Verification System:**
- Manual doctor license verification workflow
- Country-specific professional registry validation
- KYC document review interface
- Approve/reject capability with admin notes
- Email notifications for verification status changes
- Admin-only access control

### Database Architecture

**Database Technology:**
- PostgreSQL via Neon serverless platform
- WebSocket connections for serverless environment
- Connection pooling with `@neondatabase/serverless`

**Schema Design:**
- User management with role-based access (patients, doctors)
- Patient profiles with medical history and wearable data
- Doctor profiles with license verification
- Daily follow-ups for visual health assessments
- Medication tracking with AI suggestions
- Chat sessions with comprehensive medical history tracking:
  - Session metadata (start/end time, message count, agent type)
  - Symptoms discussed (array of extracted medical entities)
  - Recommendations provided (AI-generated guidance)
  - Health insights (structured medical data)
  - AI-generated summaries for each completed session
  - Doctor notes capability for collaborative care
- Dynamic task management
- Chat message storage for AI conversations
- Auto-journaling for behavioral insights
- Wellness activities (meditation, exercise)
- Research consent and AI-generated research reports
- Educational progress tracking
- Psychological counseling sessions (for both patients and doctors)
- Training datasets from public sources (PubMed, PhysioNet, Kaggle, WHO)
- Health insight consents for third-party app data sharing permissions
- EHR connections with OAuth credentials and sync status
- Wearable device integrations with connection metadata
- Referrals with unique codes and tracking statistics
- Wallet transactions and credit balances for users
- Admin verification records for doctor license approval

**Data Relationships:**
- One-to-one: Users to PatientProfile/DoctorProfile
- One-to-many: Users to DailyFollowups, Medications, ChatMessages
- Doctor-to-patient assignment for care coordination
- Session storage for authentication state

**Migration Strategy:**
- Drizzle Kit for schema migrations
- Schema versioning in `migrations/` directory
- Push-based deployment with `db:push` script

### Design System

**Color Palette:**
- Primary: Medical Teal (HSL 180 45% 45%) for clinical actions and trust
- Secondary: Deep Ocean (HSL 200 30% 25%) for headers and authority
- Wellness: Calming Sage (HSL 155 30% 60%) for Agent Clona and meditation
- Accent: Soft Lavender (HSL 250 25% 85%) for Assistant Lysa
- Status indicators: Success Teal, Warning Coral, Critical Rose
- Dark mode with adjusted contrast ratios

**Typography:**
- Primary: Inter for clinical data and dashboard
- Wellness: Lexend for meditation and calming sections
- Monospace: JetBrains Mono for lab results and numeric data
- Loaded via Google Fonts CDN

**Component Styling:**
- Utility-first with Tailwind CSS
- Custom CSS variables for theme tokens
- Elevation system with hover and active states
- Consistent border radius and spacing scales

## External Dependencies

### Third-Party Services

**Authentication:**
- Replit Auth (OpenID Connect provider) for user authentication
- Session management via `connect-pg-simple` with PostgreSQL backend

**Database:**
- Neon serverless PostgreSQL for HIPAA-compliant data storage
- WebSocket connections for serverless architecture

**AI Services:**
- OpenAI API for conversational AI (Agent Clona and Assistant Lysa)
- GPT models for medical conversation and research analysis

**Frontend Libraries:**
- Radix UI primitives for accessible components
- TanStack Query for server state management
- Wouter for routing
- React Hook Form with Zod for form validation
- date-fns for date manipulation

**Development Tools:**
- Vite for build tooling and HMR
- TypeScript for type safety
- Drizzle Kit for database migrations
- esbuild for production builds

**Data Integration APIs:**
- PubMed E-utilities (no auth required) for medical literature access
- PhysioNet WFDB library for physiological datasets
- Kaggle API with credential-based authentication for public medical datasets
- WHO Global Health Observatory API (no auth required) for global health statistics and indicators

**Security & Compliance:**
- HIPAA-compliant session storage
- Encrypted database connections
- Secure cookie handling with httpOnly and secure flags
- Environment-based configuration for secrets