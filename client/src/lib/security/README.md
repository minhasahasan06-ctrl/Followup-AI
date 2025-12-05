# Security Module Documentation

## Overview

This security module provides comprehensive HIPAA-compliant security features for API requests in the Followup-AI application. It protects against common security vulnerabilities and ensures compliance with healthcare data protection regulations.

## Security Features

### 1. SSRF Protection (`urlValidator.ts`)

**Purpose**: Prevents Server-Side Request Forgery (SSRF) attacks that could expose internal network resources or PHI data.

**Features**:
- URL validation and sanitization
- Private IP address blocking
- Protocol whitelisting (HTTP/HTTPS only)
- Domain whitelisting
- Path traversal prevention
- Query parameter sanitization

**Usage**:
```typescript
import { validateAndSanitizeUrl, getPythonBackendUrl } from './security';

const validation = validateAndSanitizeUrl(userInput);
if (!validation.isValid) {
  throw new Error(validation.error);
}
const safeUrl = validation.sanitizedUrl;
```

### 2. Rate Limiting (`rateLimiter.ts`)

**Purpose**: Prevents abuse, DoS attacks, and ensures fair resource usage.

**Features**:
- Per-origin rate limiting
- Request deduplication
- Configurable rate limits per endpoint type
- Automatic cleanup of expired entries

**Usage**:
```typescript
import { checkRateLimit, DEFAULT_RATE_LIMITS } from './security';

const result = checkRateLimit(url, 'POST', DEFAULT_RATE_LIMITS.strict);
if (!result.allowed) {
  throw new Error(`Rate limit exceeded. Retry after ${result.retryAfter} seconds`);
}
```

### 3. CSRF Protection (`csrfProtection.ts`)

**Purpose**: Prevents Cross-Site Request Forgery attacks.

**Features**:
- Automatic CSRF token generation
- Token validation on state-changing requests
- Session-based token storage

**Usage**:
```typescript
import { addCSRFHeader, requiresCSRF } from './security';

if (requiresCSRF('POST')) {
  headers = addCSRFHeader(headers);
}
```

### 4. Retry Logic with Circuit Breaker (`retryLogic.ts`)

**Purpose**: Ensures reliable data transmission and prevents cascading failures.

**Features**:
- Exponential backoff retry
- Circuit breaker pattern
- Configurable retry strategies
- Jitter to prevent thundering herd

**Usage**:
```typescript
import { retryWithBackoff } from './security';

const result = await retryWithBackoff(
  () => fetch(url),
  { maxRetries: 3, baseDelayMs: 1000 }
);
```

### 5. Audit Logging (`auditLogger.ts`)

**Purpose**: Maintains comprehensive audit trails for HIPAA compliance.

**Features**:
- Automatic logging of all API requests/responses
- PHI access tracking
- Security violation logging
- Remote logging support

**Usage**:
```typescript
import { audit } from './security';

audit.apiRequest(url, 'POST', userId);
audit.phiAccess('patient', patientId, userId);
audit.securityViolation('SSRF_ATTEMPT', { url }, userId);
```

### 6. Error Sanitization (`errorSanitizer.ts`)

**Purpose**: Prevents information leakage through error messages.

**Features**:
- Removes sensitive information from errors
- User-friendly error messages
- Preserves error codes for debugging
- Prevents PHI exposure

**Usage**:
```typescript
import { sanitizeError } from './security';

try {
  // API call
} catch (error) {
  const sanitized = sanitizeError(error);
  showUserMessage(sanitized.userMessage);
}
```

### 7. Secure API Client (`secureApiClient.ts`)

**Purpose**: Unified secure API client that integrates all security features.

**Features**:
- All security features integrated
- Request timeout handling
- Content size validation
- Automatic authentication
- PHI access tracking

**Usage**:
```typescript
import { secureApiRequest, secureGet, securePost } from './security';

// GET request
const response = await secureGet('/api/patients');

// POST request with PHI tracking
const result = await securePost(
  '/api/patients/123/records',
  data,
  {
    phiModification: true,
    resourceType: 'patient_record',
    resourceId: '123'
  }
);
```

## Integration with React Query

The security module is integrated with React Query through `queryClient.ts`:

```typescript
import { queryClient } from './lib/queryClient';

// All queries automatically use secure API client
const { data } = useQuery({
  queryKey: ['/api/patients'],
  // Security features applied automatically
});
```

## Migration Guide

### From Direct Fetch Calls

**Before**:
```typescript
const response = await fetch('/api/patients');
const data = await response.json();
```

**After**:
```typescript
import { secureGet } from './lib/security';

const response = await secureGet('/api/patients');
const data = response.data;
```

### From Axios

**Before**:
```typescript
import api from './lib/api';

const response = await api.get('/api/patients');
```

**After**:
```typescript
import { secureGet } from './lib/security';

const response = await secureGet('/api/patients');
// Or continue using api (now uses secure client internally)
```

## Configuration

### Environment Variables

- `VITE_PYTHON_BACKEND_URL`: Python backend URL (default: http://localhost:8000)
- `VITE_ALLOWED_BACKEND_DOMAINS`: Comma-separated list of allowed domains
- `VITE_APP_VERSION`: Application version for security headers

### Rate Limit Configuration

Default rate limits can be customized:

```typescript
import { DEFAULT_RATE_LIMITS } from './security';

const customLimits = {
  ...DEFAULT_RATE_LIMITS,
  strict: {
    maxRequests: 10,
    windowMs: 60000,
  },
};
```

## HIPAA Compliance

This security module addresses the following HIPAA Security Rule requirements:

1. **Access Control** (§164.312(a)(1)): Authentication and authorization
2. **Audit Controls** (§164.312(b)): Comprehensive audit logging
3. **Integrity** (§164.312(c)(1)): CSRF protection, request validation
4. **Transmission Security** (§164.312(e)(1)): Secure communication, error handling
5. **Workforce Security** (§164.308(a)(3)): Rate limiting, abuse prevention

## Best Practices

1. **Always use secure API client** for PHI-related requests
2. **Enable PHI tracking** for audit compliance:
   ```typescript
   secureGet('/api/patients/123', {
     phiAccess: true,
     resourceType: 'patient',
     resourceId: '123'
   });
   ```
3. **Handle errors gracefully** using sanitized error messages
4. **Monitor audit logs** for security violations
5. **Keep security modules updated** with latest patches

## Troubleshooting

### Rate Limit Errors

If you encounter rate limit errors:
- Check if you're making too many requests
- Consider implementing request batching
- Contact admin to adjust rate limits if needed

### CSRF Errors

If CSRF validation fails:
- Ensure cookies are enabled
- Check that session is active
- Verify CSRF token is being sent in headers

### SSRF Errors

If URL validation fails:
- Check that URL is properly formatted
- Ensure domain is in whitelist (if configured)
- Verify URL doesn't contain private IP addresses

## Support

For security concerns or questions, contact the security team.

