# Server audit (Batch 1)

## Observed features
- Express app initializes JSON/urlencoded parsing, seeds demo data on startup, registers routes, logs API responses, and conditionally serves Vite or static assets depending on environment.【F:server/index.ts†L6-L81】
- Session middleware uses express-session with Postgres-backed store when `DATABASE_URL` exists; otherwise falls back to in-memory store and logs session configuration. It also exposes role-based guards (`isAuthenticated`, `isDoctor`, `isPatient`) backed by storage lookups.【F:server/auth.ts†L19-L124】
- Custom rate limiter provides simple per-user/IP in-memory throttling plus AI- and batch-specific presets and exposes stats helper for monitoring.【F:server/rateLimiting.ts†L3-L125】
- Routes module wires many AI and data-integration helpers (OpenAI, AWS Textract/Comprehend, email AI, personalization, habit schemas, chat receptionist functions), including a mental-health indicator flow that calls GPT-4o and writes results to `mentalHealthRedFlags`.【F:server/routes.ts†L1-L188】

## Gaps and risks
- API logger captures full JSON responses (including PHI) and emits them to logs without redaction or size limits, creating PII/PHI exposure risk; truncation is length-only, not field-based.【F:server/index.ts†L12-L36】
- Seed routine runs automatically on every start and its failure is swallowed, risking unwanted mutations in non-dev environments and hiding critical errors.【F:server/index.ts†L42-L50】
- Error handler rethrows after responding, which can crash the process under load and leak stack traces upstream instead of standardized error responses.【F:server/index.ts†L53-L59】
- Session configuration defaults to a hardcoded secret and lax cookie settings when `NODE_ENV` is not production; also logs session store details to stdout, which can reveal operational info.【F:server/auth.ts†L39-L58】
- Role guards rely on `req.user` set by session but don’t protect against tampering of session payload or missing rotation/idle timeouts; missing brute-force protections and MFA enforcement.【F:server/auth.ts†L63-L124】
- Rate limiting is in-memory and resets on restart, unsuitable for multi-instance deployments and susceptible to bypass via spoofed IPs or shared reverse proxies; no per-route tuning or global circuit breakers.【F:server/rateLimiting.ts†L14-L99】
- Mental health indicator flow runs automatically on chat text with minimal validation, silent catch-and-log failures, and no consent/PHI handling; inserts unvalidated JSON and confidence defaults into DB, raising safety/governance concerns.【F:server/routes.ts†L72-L188】
- Routes file bundles many unrelated responsibilities (AI calls, storage, AWS, auth) without consistent schema validation of request payloads or centralized error handling, increasing surface for input-related bugs.【F:server/routes.ts†L1-L188】

## Recommendations
- Replace response-body logging with structured, redacted logs (e.g., omit sensitive fields, cap size) and prefer request IDs plus audit trails stored in a secure sink rather than stdout.【F:server/index.ts†L12-L36】
- Gate seeding to explicit opt-in (env flag) and treat failures as fatal in non-dev environments with clear telemetry.【F:server/index.ts†L42-L50】
- Convert error handler to avoid throwing after response; add centralized problem-detail responses, stack hiding in production, and correlation IDs.【F:server/index.ts†L53-L59】
- Require strong `SESSION_SECRET`, secure/sameSite=strict cookies, rotation/idle timeouts, and avoid logging session internals; add 2FA/MFA flows and brute-force protection at login if present.【F:server/auth.ts†L39-L124】
- Move rate limiting to a shared store (Redis) with per-route quotas, IP/session keys, and global circuit breakers; emit structured metrics for monitoring.【F:server/rateLimiting.ts†L14-L99】
- For mental-health analysis, require explicit consent flags, enforce schema validation on chat payloads, add PHI redaction, retries with alerts instead of silent failures, and restrict model outputs before DB writes.【F:server/routes.ts†L72-L188】
- Modularize routes into feature routers with Zod validation per endpoint, uniform auth/role middleware, and scoped dependency injection to reduce blast radius.【F:server/routes.ts†L1-L188】
