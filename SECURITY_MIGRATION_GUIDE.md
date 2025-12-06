# Security Migration Guide - HIPAA Compliance

This guide documents the security improvements made to the codebase for HIPAA compliance.

## Overview

The codebase has been secured and refactored to meet HIPAA compliance requirements:

1. ✅ Consolidated duplicate authentication code
2. ✅ Implemented secure logging (replacing print statements)
3. ✅ Added comprehensive error sanitization
4. ✅ Fixed insecure session secrets
5. ✅ Added PHI access audit logging
6. ✅ Implemented access control checks
7. ✅ Added encryption wrappers for PHI

## Key Changes

### 1. Unified Authentication Module

**New Location:** `app/core/authentication.py`

All authentication logic has been consolidated into a single secure module:
- JWT verification using AWS Cognito JWKS
- Comprehensive audit logging
- Secure error handling
- No information leakage

**Migration:**
```python
# OLD (deprecated)
from app.auth import get_current_user
from app.dependencies import get_current_user
from app.utils.security import verify_token

# NEW (use this)
from app.core.authentication import (
    get_current_user,
    get_current_doctor,
    get_current_patient,
    require_role
)
```

### 2. Secure Logging

**New Location:** `app/core/logging.py`

All `print()` statements should be replaced with secure logging:

```python
# OLD (insecure)
print(f"User {user_id} accessed data")
print(f"Error: {error}")

# NEW (secure)
from app.core.logging import log_info, log_error, log_warning

log_info(f"User accessed data", logger_name="my_module")
log_error(f"Error occurred", logger_name="my_module", exc_info=True)
```

**Benefits:**
- Automatic sanitization of sensitive data
- Structured logging for audit trails
- Appropriate log levels
- No sensitive information leakage

### 3. Error Sanitization

**New Location:** `app/core/error_handling.py`

All errors are now sanitized to prevent information leakage:

```python
# OLD (leaks information)
raise HTTPException(status_code=500, detail=str(error))

# NEW (sanitized)
from app.core.error_handling import create_error_response
return create_error_response(error, status_code=500)
```

### 4. PHI Protection

**New Location:** `app/core/phi_protection.py`

PHI access is now logged and encrypted:

```python
from app.core.phi_protection import (
    require_phi_access,
    get_phi_wrapper,
    check_phi_access
)

# Log PHI access
@require_phi_access("patient_profile")
async def get_patient_profile(...):
    ...

# Encrypt/decrypt PHI
phi_wrapper = get_phi_wrapper()
encrypted = phi_wrapper.encrypt_phi_value(value, patient_id, "name", user_id)
decrypted = phi_wrapper.decrypt_phi_value(encrypted, patient_id, "name", user_id)
```

### 5. Access Control

**New Location:** `app/core/access_control.py`

Comprehensive access control checks:

```python
from app.core.access_control import (
    AccessControlService,
    require_patient_access
)

# Check permissions
if not AccessControlService.check_permission(user_role, "read_patient_data"):
    raise HTTPException(status_code=403, detail="Permission denied")

# Require patient access
@require_patient_access(require_relationship=True)
async def get_patient_data(...):
    ...
```

## Migration Checklist

### High Priority (Security Critical)

- [ ] Replace all `print()` statements with secure logging
- [ ] Update all authentication imports to use `app.core.authentication`
- [ ] Add PHI access logging to all endpoints that access patient data
- [ ] Add access control checks to all patient data endpoints
- [ ] Replace error messages with sanitized versions

### Medium Priority (Compliance)

- [ ] Encrypt all PHI fields in database operations
- [ ] Add audit logging to all PHI modifications
- [ ] Implement relationship checks for doctor-patient access
- [ ] Add rate limiting to sensitive endpoints

### Low Priority (Code Quality)

- [ ] Refactor duplicate code patterns
- [ ] Add comprehensive unit tests for security modules
- [ ] Update API documentation with security requirements

## Print Statement Replacement

There are 641 print statements that need to be replaced. Use this pattern:

```python
# Pattern 1: Info logging
print(f"[INFO] Message") → log_info("Message", logger_name="module_name")

# Pattern 2: Warning logging
print(f"[WARNING] Message") → log_warning("Message", logger_name="module_name")

# Pattern 3: Error logging
print(f"[ERROR] {error}") → log_error("Error message", logger_name="module_name", exc_info=True)

# Pattern 4: Debug logging
print(f"[DEBUG] {data}") → log_debug("Debug message", logger_name="module_name")
```

## Security Best Practices

1. **Never log sensitive data:**
   - Passwords, tokens, API keys
   - Full PHI (use patient ID instead)
   - Database connection strings

2. **Always sanitize errors:**
   - Use `create_error_response()` for all errors
   - Never expose stack traces to users
   - Log detailed errors securely

3. **Always log PHI access:**
   - Use `require_phi_access` decorator
   - Log all read/write/delete operations
   - Include user ID, resource ID, and action

4. **Always check permissions:**
   - Use `AccessControlService.check_permission()`
   - Verify patient access before returning data
   - Implement minimum necessary principle

## Testing Security Changes

After migration, verify:

1. ✅ Authentication works with new module
2. ✅ No sensitive data in logs
3. ✅ Errors are sanitized
4. ✅ PHI access is logged
5. ✅ Access control works correctly
6. ✅ Encryption/decryption works for PHI

## Rollback Plan

If issues occur:

1. Old authentication modules are still available (marked deprecated)
2. Can revert to old logging by using print() temporarily
3. Security improvements are additive, not breaking

## Support

For questions or issues:
- Check `app/core/` modules for examples
- Review `HIPAA_SECURITY_IMPLEMENTATION.md`
- Consult security team for HIPAA-specific questions
