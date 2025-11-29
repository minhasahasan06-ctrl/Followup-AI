# Shared schema & infra audit (Batch 4)

## Observed features
- Drizzle schema defines extensive healthcare domain tables for users, profiles, daily/voice followups, chat sessions/messages, and medication management with Zod insert schemas for validation on inserts.【F:shared/schema.ts†L16-L234】【F:shared/schema.ts†L236-L320】
- Drizzle config enforces presence of `DATABASE_URL` and maps migrations to the shared schema path for Postgres targets.【F:drizzle.config.ts†L1-L14】
- Alembic environment imports all SQLAlchemy models and wires offline/online migration flows with type/default change detection enabled.【F:alembic/env.py†L1-L94】
- `create_tables.py` provides an ad-hoc table creation script for newly added doctor/consultation models, listing expected tables and emitting success/failure logs.【F:create_tables.py†L1-L37】

## Gaps and risks
- Schema stores large volumes of PHI/PII (e.g., comorbidities, medications, AI-derived symptoms/mood) without encryption/retention flags and allows many nullable medical fields, risking partial/ambiguous records and privacy exposure.【F:shared/schema.ts†L95-L174】【F:shared/schema.ts†L185-L234】
- Foreign keys lack on-delete/on-update semantics and indexing, so orphaned rows and slow joins are likely across patient/doctor/chat/medication relations.【F:shared/schema.ts†L92-L312】
- Heavy JSONB usage for medical insights, metrics, and recommendations lacks schema constraints or versioning, increasing risk of malformed payloads and downstream parser errors.【F:shared/schema.ts†L105-L220】【F:shared/schema.ts†L247-L320】
- Infrastructure scripts mix migration strategies: Alembic expects SQLAlchemy models while Drizzle targets `shared/schema.ts`, and the standalone `create_tables.py` bypasses migrations entirely, risking drift between engines and environments.【F:drizzle.config.ts†L1-L14】【F:alembic/env.py†L12-L94】【F:create_tables.py†L1-L37】

## Recommendations
- Classify PHI/PII columns and apply column-level encryption or KMS-backed secrets for sensitive fields; add retention/archival policies and required flags for consent and completeness on medical data.【F:shared/schema.ts†L95-L234】
- Add `onDelete`/`onUpdate` actions and supporting indexes to all foreign keys; enforce uniqueness where appropriate (e.g., patient/session relationships) to prevent orphans and improve performance.【F:shared/schema.ts†L92-L320】
- Replace free-form JSONB with typed child tables or validated JSON schemas plus versioning; add constraints/defaults to guarantee required metrics and guardrails for AI outputs before persistence.【F:shared/schema.ts†L105-L320】
- Standardize on a single migration path: retire ad-hoc table creators, align Drizzle/Alembic to one source of truth, and gate database creation behind explicit environment/approval checks to avoid accidental prod writes.【F:drizzle.config.ts†L1-L14】【F:alembic/env.py†L1-L94】【F:create_tables.py†L1-L37】
