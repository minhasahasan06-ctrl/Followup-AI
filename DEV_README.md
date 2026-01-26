# Development Environment - HIPAA Compliance Guide

> **CRITICAL: This Replit workspace is for DEVELOPMENT ONLY**  
> Never store, process, or access Protected Health Information (PHI) or production systems from this environment.

## Quick Start

1. Clone this repository
2. Run `npm install`
3. Set up development-only environment variables (see below)
4. Run `npm run dev`

The application will automatically run environment checks before starting.

---

## Absolute Rules (Non-Negotiable)

1. **No PHI** - Never create, store, process, transmit, log, or display Protected Health Information
2. **No Production Secrets** - Never store production API keys, private keys, service account credentials
3. **No Real Patient Data** - Never paste, use, or import real patient data, production database dumps, or production logs
4. **Assume Everything is Logged** - Treat all files, console output, env vars, and run history as persistent and reviewable

---

## Environment Variables

### Allowed in Replit (Development Only)

| Variable | Description | Example |
|----------|-------------|---------|
| `DATABASE_URL` | Development Neon DB connection | `postgres://...dev.neon.tech/...` |
| `AUTH0_DOMAIN` | Development Auth0 tenant | `myapp-dev.auth0.com` |
| `AUTH0_CLIENT_ID` | Development client ID | `abc123...` |
| `GCP_PROJECT_ID` | Development GCP project | `myapp-dev-123` |
| `STRIPE_SECRET_KEY` | Test mode key only | `sk_test_...` |
| `OPENAI_API_KEY` | Development API key | `sk-...` |
| `REPLIT_DEV_ONLY` | Safety flag | `true` |

### NEVER Store in Replit

- Production database URLs
- Production Auth0 credentials
- Production GCP service accounts
- Live Stripe keys (`sk_live_*`, `pk_live_*`)
- Any key/credential with "prod" or "production" in the path

---

## Config Guard

The application includes automatic environment checks that run at startup:

```typescript
// server/config_guard.ts runs these checks:
✓ GCP_PROJECT_ID is not a production project
✓ DATABASE_URL does not contain production Neon hosts
✓ AUTH0_DOMAIN is not production Auth0
✓ API_BASE_URL does not point to production
✓ No env vars with PROD_ prefix have values
✓ No secret patterns detected in env values
```

If any check fails, the application will **exit immediately** with a generic error message that does not reveal which value triggered the block.

### Testing the Config Guard

```bash
# This should fail and exit:
GCP_PROJECT_ID=followupai-prod npm run dev

# This should succeed:
GCP_PROJECT_ID=followupai-dev npm run dev
```

---

## Safe Logger

All application logging uses `safe_logger.ts` which automatically redacts:

- Email addresses
- Phone numbers
- Social Security Numbers
- Medical Record Numbers
- Credit card numbers
- JWT tokens
- API keys
- Date of birth patterns
- Full names with titles

### Example

```typescript
import { safeLogger } from './safe_logger';

// Input: "User john.doe@email.com called 555-123-4567"
// Output: "User [EMAIL_REDACTED] called [PHONE_REDACTED]"
safeLogger.info("User john.doe@email.com called 555-123-4567");
```

---

## Synthetic Data Seeding

For development testing, use only synthetic data:

```bash
# Seed with default 5 patients, 2 doctors
npm run seed:dev

# Seed with custom counts
npm run seed:dev -- --patients 10 --doctors 3
```

**Never import production database dumps or real patient data.**

---

## CI/CD Security Checks

The repository includes automated security scanning:

1. **Gitleaks** - Scans for secrets in git history
2. **Production Identifier Check** - Blocks commits with prod URLs
3. **PHI Pattern Warning** - Alerts on potential PHI patterns
4. **Dependency Audit** - Checks for vulnerable packages

### Pre-commit Hook Installation

```bash
# Install the pre-commit hook
cp scripts/pre-commit-hook.sh .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit
```

---

## Network Hardening

### Limitation

Replit does not support outbound network allowlisting. This means we cannot block connections to production hosts at the network level.

### Mitigations

1. **Fail-closed config guard** - Blocks startup if production identifiers detected
2. **Safe logger** - Redacts PHI in all logs
3. **Pre-commit hooks** - Prevents committing production identifiers
4. **CI checks** - Scans for secrets on every push

### Recommendation

For accessing production systems, use a dedicated jump-hosted environment with:
- VPC isolation
- Network allowlisting
- BAA-covered infrastructure
- Audit logging

---

## File Structure

```
├── prod_identifiers.json     # Production identifiers to block (no secrets)
├── server/
│   ├── config_guard.ts       # Fail-closed environment checks
│   ├── safe_logger.ts        # PHI-redacting logger
│   └── dev_seed.ts           # Synthetic data seeder
├── scripts/
│   └── pre-commit-hook.sh    # Local secret scanning
├── .github/workflows/
│   └── ci_checks.yml         # CI security scanning
├── DEV_README.md             # This file
└── AUDIT_SUMMARY.md          # Guardrail audit report
```

---

## What To Do If...

### You accidentally pasted PHI

1. **Stop immediately**
2. Delete the content from all files
3. Clear terminal history
4. Clear browser console
5. Report to security team

### You need to access production

1. **Do not attempt from Replit**
2. Use the designated production access environment
3. Follow production access procedures
4. Ensure BAA coverage

### The config guard blocks your startup

1. Check which environment variables are set
2. Ensure you're using development credentials only
3. Remove any `PROD_` prefixed variables
4. Verify database URL points to dev environment

### You find a false positive in secret scanning

1. Add the pattern to `.gitleaksignore` (if truly safe)
2. Document why it's a false positive
3. Have security team review

---

## Contact

For HIPAA compliance questions or security concerns, contact:
- Security Team: [security@yourcompany.com]
- Compliance Officer: [compliance@yourcompany.com]

---

**Remember: When in doubt, don't store it in Replit.**
