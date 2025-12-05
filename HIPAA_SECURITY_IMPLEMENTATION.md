# HIPAA-Compliant Security Implementation Guide

## Overview

This document describes the comprehensive HIPAA-level security enhancements implemented in the Followup AI codebase. All security measures follow HIPAA Security Rule requirements and industry best practices.

## Security Architecture

### Defense-in-Depth Strategy

The security implementation uses multiple layers of protection:

1. **Network Security**: TLS 1.3 encryption, security headers, CORS policies
2. **Application Security**: Input validation, output encoding, secure coding practices
3. **Authentication & Authorization**: Multi-factor authentication, RBAC/ABAC
4. **Data Protection**: Encryption at rest (AWS KMS), encryption in transit (TLS)
5. **Audit & Monitoring**: Comprehensive logging, anomaly detection, compliance monitoring
6. **Access Controls**: Role-based access, attribute-based access, minimum necessary principle

## Implemented Security Components

### 1. AWS KMS Encryption Service (`app/services/kms_encryption_service.py`)

**Purpose**: Field-level encryption of PHI using AWS KMS envelope encryption pattern.

**Features**:
- Envelope encryption (data key encrypted by KMS master key)
- Field-level encryption for sensitive PHI fields
- Encryption context for key derivation (patient_id, field_name)
- Automatic audit logging of encryption/decryption operations

**Usage**:
```python
from app.services.kms_encryption_service import get_kms_service

kms = get_kms_service()

# Encrypt PHI field
encrypted = kms.encrypt_phi_field(
    value="Patient Name",
    patient_id="patient-123",
    field_name="name",
    user_id="user-456"
)

# Decrypt PHI field
decrypted = kms.decrypt_phi_field(
    encrypted_value=encrypted,
    patient_id="patient-123",
    field_name="name",
    user_id="user-456"
)
```

**Configuration**:
- Set `AWS_KMS_KEY_ID` environment variable
- Configure AWS credentials (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`)

### 2. Enhanced Audit Logging (`app/services/enhanced_audit_logger.py`)

**Purpose**: Immutable audit trail for all PHI access and system events.

**Features**:
- Database-persisted audit logs
- Cryptographic integrity hashing
- Comprehensive metadata capture
- HIPAA accounting of disclosures

**Usage**:
```python
from app.services.enhanced_audit_logger import EnhancedAuditLogger

# Log PHI access
EnhancedAuditLogger.log_phi_access(
    db=db,
    user_id="user-123",
    user_type="clinician",
    patient_id="patient-456",
    resource_type="medical_record",
    resource_id="record-789",
    action_type="view",
    data_fields=["name", "diagnosis"],
    ip_address="192.168.1.1"
)

# Log security event
EnhancedAuditLogger.log_security_event(
    db=db,
    event_type="brute_force_attempt",
    severity="high",
    description="Excessive failed login attempts",
    user_id="user-123",
    ip_address="192.168.1.1"
)
```

### 3. Rate Limiting (`app/middleware/rate_limiter.py`)

**Purpose**: DDoS protection and brute force prevention.

**Features**:
- IP-based and user-based rate limiting
- Configurable limits per endpoint
- Automatic IP blocking for violations
- Redis-backed distributed rate limiting (production)

**Configuration**:
- Edit `RATE_LIMIT_CONFIGS` in `rate_limiter.py` to customize limits
- For production, configure Redis for distributed rate limiting

### 4. Security Headers (`app/middleware/security_headers.py`)

**Purpose**: HTTP security headers for defense-in-depth.

**Headers Implemented**:
- `X-Frame-Options: DENY` - Prevent clickjacking
- `X-Content-Type-Options: nosniff` - Prevent MIME sniffing
- `X-XSS-Protection: 1; mode=block` - XSS protection
- `Content-Security-Policy` - XSS and injection prevention
- `Strict-Transport-Security` - Force HTTPS
- `Referrer-Policy` - Control referrer information
- `Permissions-Policy` - Restrict browser features

### 5. Input Validation (`app/middleware/input_validation.py`)

**Purpose**: Prevent injection attacks (SQL, XSS, command injection).

**Features**:
- Pattern-based threat detection
- Automatic sanitization
- Request body validation
- Query parameter validation

**Threats Detected**:
- SQL injection patterns
- XSS (Cross-Site Scripting)
- Command injection
- Path traversal attempts

### 6. Secrets Management (`app/services/secrets_manager.py`)

**Purpose**: Centralized secrets management using AWS Secrets Manager.

**Features**:
- Secure secret retrieval
- Secret caching with TTL
- Support for JSON and plain text secrets
- Fallback to environment variables (development)

**Usage**:
```python
from app.services.secrets_manager import get_secrets_service

secrets = get_secrets_service()

# Get database credentials
db_creds = secrets.get_database_credentials()

# Get OpenAI API key
openai_key = secrets.get_openai_api_key()
```

**Configuration**:
- Set secret names in environment variables:
  - `DB_SECRET_NAME`
  - `OPENAI_SECRET_NAME`
  - `COGNITO_SECRET_NAME`
  - `TWILIO_SECRET_NAME`

### 7. RBAC/ABAC Service (`app/services/rbac_service.py`)

**Purpose**: Fine-grained access control with HIPAA minimum necessary principle.

**Features**:
- Role-based permissions (Patient, Clinician, Admin, System)
- Attribute-based access control (patient ownership, treatment relationship)
- Minimum necessary principle enforcement
- Dynamic permission evaluation

**Usage**:
```python
from app.services.rbac_service import get_rbac_service, AccessContext, Role

rbac = get_rbac_service()

context = AccessContext(
    user_id="user-123",
    user_role=Role.CLINICIAN,
    patient_id="patient-456",
    treatment_relationship=True,
    action="view"
)

allowed, reason = rbac.check_access(context, "view")
if not allowed:
    raise HTTPException(403, detail=reason)
```

### 8. Security Monitoring (`app/services/security_monitoring.py`)

**Purpose**: Anomaly detection and threat monitoring.

**Detected Anomalies**:
- Brute force login attempts
- Excessive PHI access
- Unusual access patterns (outside business hours)
- Multiple patient access in short time
- Geographic anomalies (rapid location changes)

**Usage**:
```python
from app.services.security_monitoring import get_security_monitoring

monitoring = get_security_monitoring()

# Monitor user activity
monitoring.monitor_user_activity(
    db=db,
    user_id="user-123",
    action_type="view",
    resource_type="medical_record",
    phi_accessed=True,
    patient_id="patient-456",
    ip_address="192.168.1.1"
)

# Get security dashboard data
dashboard = monitoring.get_security_dashboard_data(db, days=7)
```

### 9. Compliance Monitoring (`app/services/compliance_monitoring.py`)

**Purpose**: Automated HIPAA compliance checks and reporting.

**Checks**:
- Audit logging compliance
- Access control compliance
- Consent management compliance
- Data retention policy compliance

**Usage**:
```python
from app.services.compliance_monitoring import get_compliance_monitoring

compliance = get_compliance_monitoring()

# Generate compliance report
report = compliance.generate_compliance_report(db, days=30)
print(f"Compliance Score: {report['overall_compliance_score']}")
print(f"Recommendations: {report['recommendations']}")
```

### 10. Enhanced Database Security (`app/database_enhanced.py`)

**Purpose**: Secure database connections with encryption.

**Features**:
- SSL/TLS encryption for connections
- Connection pooling with security
- Connection timeout and retry logic
- Session-level security settings

**Configuration**:
- Database URL must include SSL parameters
- Set `sslmode=require` in connection string

### 11. API Security (`app/middleware/api_security.py`)

**Purpose**: API key authentication and request signing.

**Features**:
- API key validation
- Request signature verification
- Replay attack prevention (nonce + timestamp)
- API usage tracking

## Environment Variables

### Required for Production

```bash
# AWS Configuration
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
AWS_KMS_KEY_ID=arn:aws:kms:region:account:key/key-id

# Database
DATABASE_URL=postgresql://user:pass@host:5432/db?sslmode=require

# Secrets Manager (optional, falls back to env vars)
DB_SECRET_NAME=followup-ai/database-credentials
OPENAI_SECRET_NAME=followup-ai/openai-api-key
COGNITO_SECRET_NAME=followup-ai/cognito-credentials

# Security
SESSION_SECRET=your-secure-random-secret-min-32-chars
DEV_MODE_SECRET=  # Remove in production

# HIPAA Compliance
OPENAI_BAA_SIGNED=true
OPENAI_ZDR_ENABLED=true
OPENAI_ENTERPRISE=true
```

## Security Best Practices

### 1. Encryption

- **At Rest**: All PHI fields encrypted using AWS KMS
- **In Transit**: TLS 1.3 for all connections
- **Database**: SSL/TLS required for database connections

### 2. Access Control

- **Authentication**: AWS Cognito JWT tokens
- **Authorization**: RBAC/ABAC with minimum necessary principle
- **MFA**: Enable two-factor authentication for all users

### 3. Audit Logging

- **All PHI Access**: Logged with user, patient, timestamp, IP
- **Security Events**: Logged with severity and action taken
- **Immutable Logs**: Cryptographic hashing for integrity

### 4. Monitoring

- **Anomaly Detection**: Automated threat detection
- **Compliance Monitoring**: Regular compliance checks
- **Security Dashboard**: Real-time security metrics

### 5. Data Protection

- **PHI Redaction**: Before sending to external services
- **Data Minimization**: Only collect necessary data
- **Data Retention**: Automated retention policies

## Compliance Checklist

- [x] Encryption at rest (AWS KMS)
- [x] Encryption in transit (TLS 1.3)
- [x] Access controls (RBAC/ABAC)
- [x] Audit logging (comprehensive)
- [x] Input validation (injection prevention)
- [x] Rate limiting (DDoS protection)
- [x] Security headers (defense-in-depth)
- [x] Secrets management (AWS Secrets Manager)
- [x] Anomaly detection (threat monitoring)
- [x] Compliance monitoring (automated checks)

## Security Incident Response

1. **Detection**: Security monitoring service detects anomalies
2. **Logging**: Security event logged to database
3. **Alert**: Admin notified (implement notification service)
4. **Response**: Automatic actions (IP blocking, account locking)
5. **Investigation**: Review audit logs and security events
6. **Remediation**: Fix vulnerabilities and update security policies

## Testing Security

### Test Rate Limiting
```bash
# Should be rate limited after 5 requests
for i in {1..10}; do curl http://localhost:8000/api/auth/login; done
```

### Test Input Validation
```bash
# Should be rejected
curl -X POST http://localhost:8000/api/patients \
  -H "Content-Type: application/json" \
  -d '{"name": "<script>alert(1)</script>"}'
```

### Test Security Headers
```bash
curl -I http://localhost:8000/api/health
# Check for security headers in response
```

## Maintenance

### Regular Tasks

1. **Weekly**: Review security events and anomalies
2. **Monthly**: Generate compliance reports
3. **Quarterly**: Security audit and penetration testing
4. **Annually**: HIPAA compliance review

### Monitoring

- Monitor security dashboard daily
- Review failed authentication attempts
- Check for unusual access patterns
- Verify audit logs are complete

## Support

For security issues or questions:
1. Review this documentation
2. Check security service logs
3. Review compliance reports
4. Contact security team

## References

- [HIPAA Security Rule](https://www.hhs.gov/hipaa/for-professionals/security/index.html)
- [AWS KMS Best Practices](https://docs.aws.amazon.com/kms/latest/developerguide/best-practices.html)
- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [NIST Cybersecurity Framework](https://www.nist.gov/cyberframework)
