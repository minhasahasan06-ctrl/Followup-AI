# Security Quick Reference Guide

## Quick Start

### 1. Enable Security Middleware

Security middleware is automatically enabled in `app/main.py`. No additional configuration needed.

### 2. Encrypt PHI Fields

```python
from app.services.kms_encryption_service import get_kms_service

kms = get_kms_service()

# Encrypt before storing
encrypted_data = kms.encrypt_phi_field(
    value=patient_name,
    patient_id=patient_id,
    field_name="name",
    user_id=current_user.id
)

# Decrypt when reading
decrypted_name = kms.decrypt_phi_field(
    encrypted_value=encrypted_data,
    patient_id=patient_id,
    field_name="name",
    user_id=current_user.id
)
```

### 3. Log PHI Access

```python
from app.services.enhanced_audit_logger import EnhancedAuditLogger

# Log every PHI access
EnhancedAuditLogger.log_phi_access(
    db=db,
    user_id=current_user.id,
    user_type="clinician",
    patient_id=patient_id,
    resource_type="medical_record",
    resource_id=record_id,
    action_type="view",
    data_fields=["name", "diagnosis"],
    ip_address=request.client.host
)
```

### 4. Check Access Permissions

```python
from app.services.rbac_service import get_rbac_service, AccessContext, Role

rbac = get_rbac_service()

context = AccessContext(
    user_id=current_user.id,
    user_role=Role.CLINICIAN,
    patient_id=patient_id,
    treatment_relationship=True,
    action="view"
)

allowed, reason = rbac.check_access(context, "view")
if not allowed:
    raise HTTPException(403, detail=reason)
```

### 5. Monitor Security

```python
from app.services.security_monitoring import get_security_monitoring

monitoring = get_security_monitoring()

# Monitor user activity (call after each action)
monitoring.monitor_user_activity(
    db=db,
    user_id=current_user.id,
    action_type="view",
    resource_type="medical_record",
    phi_accessed=True,
    patient_id=patient_id,
    ip_address=request.client.host
)
```

## Common Patterns

### Pattern 1: Protected Endpoint with PHI Access

```python
from fastapi import Depends, HTTPException
from app.auth import get_current_user
from app.database import get_db
from app.services.rbac_service import get_rbac_service, AccessContext, Role
from app.services.enhanced_audit_logger import EnhancedAuditLogger
from app.services.security_monitoring import get_security_monitoring

@router.get("/patients/{patient_id}/records")
async def get_patient_records(
    patient_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # Check access
    rbac = get_rbac_service()
    context = AccessContext(
        user_id=current_user.id,
        user_role=Role(current_user.role),
        patient_id=patient_id,
        treatment_relationship=check_treatment_relationship(db, current_user.id, patient_id),
        action="view"
    )
    
    allowed, reason = rbac.check_access(context, "view")
    if not allowed:
        raise HTTPException(403, detail=reason)
    
    # Get records
    records = get_records(db, patient_id)
    
    # Log PHI access
    EnhancedAuditLogger.log_phi_access(
        db=db,
        user_id=current_user.id,
        user_type=current_user.role,
        patient_id=patient_id,
        resource_type="medical_record",
        action_type="view",
        ip_address=request.client.host
    )
    
    # Monitor security
    monitoring = get_security_monitoring()
    monitoring.monitor_user_activity(
        db=db,
        user_id=current_user.id,
        action_type="view",
        resource_type="medical_record",
        phi_accessed=True,
        patient_id=patient_id,
        ip_address=request.client.host
    )
    
    return records
```

### Pattern 2: Storing Encrypted PHI

```python
from app.services.kms_encryption_service import get_kms_service

@router.post("/patients/{patient_id}/records")
async def create_record(
    patient_id: str,
    record_data: RecordCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # Encrypt sensitive fields
    kms = get_kms_service()
    
    encrypted_name = kms.encrypt_phi_field(
        value=record_data.patient_name,
        patient_id=patient_id,
        field_name="name",
        user_id=current_user.id
    )
    
    encrypted_diagnosis = kms.encrypt_phi_field(
        value=record_data.diagnosis,
        patient_id=patient_id,
        field_name="diagnosis",
        user_id=current_user.id
    )
    
    # Store encrypted data
    record = Record(
        patient_id=patient_id,
        name_encrypted=encrypted_name,
        diagnosis_encrypted=encrypted_diagnosis,
        created_by=current_user.id
    )
    
    db.add(record)
    db.commit()
    
    # Log creation
    EnhancedAuditLogger.log_phi_access(
        db=db,
        user_id=current_user.id,
        user_type=current_user.role,
        patient_id=patient_id,
        resource_type="medical_record",
        resource_id=record.id,
        action_type="create",
        ip_address=request.client.host
    )
    
    return record
```

### Pattern 3: Reading Encrypted PHI

```python
@router.get("/patients/{patient_id}/records/{record_id}")
async def get_record(
    patient_id: str,
    record_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # Check access (same as Pattern 1)
    # ... access check code ...
    
    # Get record
    record = db.query(Record).filter(Record.id == record_id).first()
    
    # Decrypt sensitive fields
    kms = get_kms_service()
    
    decrypted_name = kms.decrypt_phi_field(
        encrypted_value=record.name_encrypted,
        patient_id=patient_id,
        field_name="name",
        user_id=current_user.id
    )
    
    decrypted_diagnosis = kms.decrypt_phi_field(
        encrypted_value=record.diagnosis_encrypted,
        patient_id=patient_id,
        field_name="diagnosis",
        user_id=current_user.id
    )
    
    # Log access
    EnhancedAuditLogger.log_phi_access(
        db=db,
        user_id=current_user.id,
        user_type=current_user.role,
        patient_id=patient_id,
        resource_type="medical_record",
        resource_id=record_id,
        action_type="view",
        data_fields=["name", "diagnosis"],
        ip_address=request.client.host
    )
    
    return {
        "id": record.id,
        "name": decrypted_name,
        "diagnosis": decrypted_diagnosis,
        "created_at": record.created_at
    }
```

## Security Checklist for New Endpoints

- [ ] Authentication required (`Depends(get_current_user)`)
- [ ] Access control checked (`rbac.check_access()`)
- [ ] PHI access logged (`EnhancedAuditLogger.log_phi_access()`)
- [ ] Security monitoring (`monitoring.monitor_user_activity()`)
- [ ] Sensitive data encrypted (`kms.encrypt_phi_field()`)
- [ ] Input validation (automatic via middleware)
- [ ] Rate limiting (automatic via middleware)
- [ ] Error handling (don't expose sensitive info)

## Environment Variables Checklist

```bash
# Required
AWS_KMS_KEY_ID=arn:aws:kms:...
DATABASE_URL=postgresql://...?sslmode=require
SESSION_SECRET=min-32-chars-random-secret

# Recommended
DB_SECRET_NAME=followup-ai/database-credentials
OPENAI_SECRET_NAME=followup-ai/openai-api-key

# HIPAA Compliance
OPENAI_BAA_SIGNED=true
OPENAI_ZDR_ENABLED=true
OPENAI_ENTERPRISE=true
```

## Testing Security

```bash
# Test rate limiting
for i in {1..10}; do curl http://localhost:8000/api/auth/login; done

# Test input validation
curl -X POST http://localhost:8000/api/patients \
  -d '{"name": "<script>alert(1)</script>"}'

# Test security headers
curl -I http://localhost:8000/api/health
```

## Troubleshooting

### KMS Encryption Not Working
- Check AWS credentials are configured
- Verify `AWS_KMS_KEY_ID` is set
- Check IAM permissions for KMS

### Audit Logging Failing
- Verify database connection
- Check `audit_logs` table exists
- Review database logs

### Rate Limiting Too Strict
- Adjust limits in `app/middleware/rate_limiter.py`
- Check `RATE_LIMIT_CONFIGS`

### Access Denied Errors
- Verify user role in database
- Check treatment relationship exists
- Review RBAC permissions

## Support

See `HIPAA_SECURITY_IMPLEMENTATION.md` for detailed documentation.
