# Deployment Guide

## Environment Variables

All secrets and configuration should be set via environment variables. See `.env.example` for a complete list.

### Required Environment Variables

- `DATABASE_URL` - PostgreSQL connection string
- `OPENAI_API_KEY` - OpenAI API key for AI features
- `AWS_ACCESS_KEY_ID` - AWS access key for S3 and Cognito
- `AWS_SECRET_ACCESS_KEY` - AWS secret key
- `AWS_S3_BUCKET_NAME` - S3 bucket for file storage
- `AWS_COGNITO_USER_POOL_ID` - Cognito user pool ID
- `AWS_COGNITO_CLIENT_ID` - Cognito client ID
- `AWS_COGNITO_CLIENT_SECRET` - Cognito client secret

### Optional Environment Variables

- `TWILIO_ACCOUNT_SID` - For SMS/voice features
- `TWILIO_AUTH_TOKEN` - Twilio auth token
- `TWILIO_PHONE_NUMBER` - Twilio phone number
- `DEV_MODE_SECRET` - Development mode secret (min 32 chars)
- `SESSION_SECRET` - Session encryption secret

## GitHub Actions

The project includes two GitHub Actions workflows:

1. **Secret Scan** (`.github/workflows/secret-scan.yml`) - Scans for hardcoded secrets using Gitleaks
2. **Unit Tests** (`.github/workflows/unit-tests.yml`) - Runs pytest tests with PostgreSQL service

## Vercel Deployment

1. Install Vercel CLI: `npm i -g vercel`
2. Set environment variables in Vercel dashboard
3. Deploy: `vercel --prod`

The `vercel.json` file configures the Python runtime and routes.

## Local Development

1. Copy `.env.example` to `.env`
2. Fill in your environment variables
3. Install dependencies: `pip install -r requirements.txt`
4. Run migrations: `alembic upgrade head`
5. Start server: `uvicorn app.main:app --reload`

## Testing

Run tests with:
```bash
pytest tests/ -v
```

Tests use a test database configured in `tests/conftest.py`.
