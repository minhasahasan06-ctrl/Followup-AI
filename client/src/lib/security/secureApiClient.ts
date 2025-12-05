/**
 * Secure API Client
 * 
 * HIPAA-Compliant API client with comprehensive security features:
 * - SSRF Protection
 * - Rate Limiting
 * - CSRF Protection
 * - Request Deduplication
 * - Retry Logic with Circuit Breaker
 * - Audit Logging
 * - Error Sanitization
 * - Request Timeout
 * - Content Validation
 */

import { validateAndSanitizeUrl, getPythonBackendUrl, shouldRouteToPythonBackend } from './urlValidator';
import { checkRateLimit, isDuplicateRequest, trackPendingRequest, DEFAULT_RATE_LIMITS, getRateLimitKey } from './rateLimiter';
import { addCSRFHeader, requiresCSRF } from './csrfProtection';
import { retryWithBackoff, getCircuitBreakerKey } from './retryLogic';
import { audit, AuditEventType } from './auditLogger';
import { sanitizeError, createSafeErrorLog } from './errorSanitizer';

export interface SecureApiRequestOptions extends RequestInit {
  json?: unknown;
  timeout?: number;
  skipRateLimit?: boolean;
  skipDeduplication?: boolean;
  skipCSRF?: boolean;
  skipRetry?: boolean;
  rateLimitConfig?: typeof DEFAULT_RATE_LIMITS.default;
  userId?: string;
  phiAccess?: boolean;
  phiModification?: boolean;
  resourceType?: string;
  resourceId?: string;
}

export interface SecureApiResponse<T = any> extends Response {
  data?: T;
}

const DEFAULT_TIMEOUT = 30000; // 30 seconds
const MAX_BODY_SIZE = 10 * 1024 * 1024; // 10MB

/**
 * Gets authentication token from storage
 */
function getAuthToken(): string | null {
  if (typeof window === 'undefined') {
    return null;
  }

  try {
    const authTokens = localStorage.getItem('authTokens');
    if (authTokens) {
      const parsed = JSON.parse(authTokens);
      return parsed.accessToken || null;
    }
  } catch (error) {
    // Silently fail - token might not be available
  }

  return null;
}

/**
 * Gets user ID from storage
 */
export function getUserId(): string | undefined {
  if (typeof window === 'undefined') {
    return undefined;
  }

  try {
    const authUser = localStorage.getItem('authUser');
    if (authUser) {
      const parsed = JSON.parse(authUser);
      return parsed.id;
    }
  } catch (error) {
    // Silently fail
  }

  return undefined;
}

/**
 * Throws an Error instance with sanitized error information attached
 * This ensures errors can be properly caught and re-sanitized if needed
 */
function throwSanitizedError(error: unknown, context?: string): never {
  const sanitized = sanitizeError(error, context);
  const errorInstance = new Error(sanitized.userMessage) as any;
  errorInstance.statusCode = sanitized.statusCode;
  errorInstance.code = sanitized.code;
  errorInstance.userMessage = sanitized.userMessage;
  errorInstance.message = sanitized.message;
  if (error && typeof error === 'object' && 'response' in error) {
    errorInstance.response = (error as any).response;
  }
  throw errorInstance;
}

/**
 * Validates response content size
 */
async function validateResponseSize(response: Response): Promise<void> {
  const contentLength = response.headers.get('content-length');
  if (contentLength) {
    const size = parseInt(contentLength, 10);
    if (size > MAX_BODY_SIZE) {
      throw new Error(`Response too large: ${size} bytes (max: ${MAX_BODY_SIZE})`);
    }
  }
}

/**
 * Creates AbortController with timeout
 */
function createTimeoutController(timeoutMs: number): {
  controller: AbortController;
  clearTimeout: () => void;
} {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => {
    controller.abort();
  }, timeoutMs);
  
  return {
    controller,
    clearTimeout: () => clearTimeout(timeoutId),
  };
}

/**
 * Secure API request function
 */
export async function secureApiRequest<T = any>(
  url: string,
  options: SecureApiRequestOptions = {}
): Promise<SecureApiResponse<T>> {
  const {
    json,
    timeout = DEFAULT_TIMEOUT,
    skipRateLimit = false,
    skipDeduplication = false,
    skipCSRF = false,
    skipRetry = false,
    rateLimitConfig,
    userId: providedUserId,
    phiAccess = false,
    phiModification = false,
    resourceType,
    resourceId,
    ...fetchOptions
  } = options;

  const method = (fetchOptions.method || 'GET').toUpperCase();
  const userId = providedUserId || getUserId();
  const authToken = getAuthToken();

  // Step 1: URL Validation and SSRF Protection
  let finalUrl: string;
  try {
    if (shouldRouteToPythonBackend(url)) {
      finalUrl = getPythonBackendUrl(url);
    } else {
      const validation = validateAndSanitizeUrl(url);
      if (!validation.isValid) {
        audit.securityViolation('SSRF_ATTEMPT', { url, error: validation.error }, userId);
        throw new Error(`Invalid URL: ${validation.error}`);
      }
      finalUrl = validation.sanitizedUrl;
    }
  } catch (error) {
    audit.securityViolation('SSRF_ATTEMPT', { url, error: String(error) }, userId);
    throwSanitizedError(error, 'URL validation');
  }

  // Step 2: Rate Limiting
  if (!skipRateLimit) {
    const rateLimitResult = checkRateLimit(
      finalUrl,
      method,
      rateLimitConfig || DEFAULT_RATE_LIMITS.default
    );

    if (!rateLimitResult.allowed) {
      audit.log({
        eventType: AuditEventType.RATE_LIMIT_EXCEEDED,
        url: finalUrl,
        method,
        userId,
        metadata: { remaining: rateLimitResult.remaining, resetTime: rateLimitResult.resetTime },
      });

      const error = new Error('Rate limit exceeded') as any;
      error.statusCode = 429;
      error.retryAfter = rateLimitResult.retryAfter;
      throwSanitizedError(error);
    }
  }

  // Step 3: Request Deduplication
  const requestKey = `${method}:${finalUrl}:${json ? JSON.stringify(json) : ''}`;
  if (!skipDeduplication && isDuplicateRequest(finalUrl, method, json ? JSON.stringify(json) : undefined)) {
    // Return existing pending request if available
    const existingPending = (window as any).__pendingApiRequests?.get(requestKey);
    if (existingPending) {
      return existingPending;
    }
  }

  // Step 4: Prepare headers
  const headers = new Headers(fetchOptions.headers);

  // Add authentication
  if (authToken) {
    headers.set('Authorization', `Bearer ${authToken}`);
  }

  // Add CSRF protection for state-changing requests
  if (!skipCSRF && requiresCSRF(method)) {
    const csrfHeaders = addCSRFHeader(Object.fromEntries(headers.entries()));
    Object.entries(csrfHeaders).forEach(([key, value]) => {
      headers.set(key, String(value));
    });
  }

  // Add security headers
  headers.set('X-Requested-With', 'XMLHttpRequest');
  headers.set('X-Client-Version', import.meta.env.VITE_APP_VERSION || '1.0.0');

  // Step 5: Prepare body
  let body: BodyInit | undefined = fetchOptions.body;
  if (json !== undefined) {
    headers.set('Content-Type', 'application/json');
    body = JSON.stringify(json);
  }

  // Validate body size
  if (body && typeof body === 'string' && body.length > MAX_BODY_SIZE) {
    throw new Error(`Request body too large: ${body.length} bytes (max: ${MAX_BODY_SIZE})`);
  }

  // Step 6: Create timeout controller
  const timeoutControllerWrapper = createTimeoutController(timeout);
  const timeoutController = timeoutControllerWrapper.controller;
  const abortController = new AbortController();

  // Combine abort signals (fallback for browsers that don't support AbortSignal.any)
  let combinedSignal: AbortSignal;
  if (typeof AbortSignal !== 'undefined' && 'any' in AbortSignal) {
    // Use AbortSignal.any if available
    const signals = [timeoutController.signal, abortController.signal];
    if (fetchOptions.signal) {
      signals.push(fetchOptions.signal);
    }
    combinedSignal = AbortSignal.any(signals) as AbortSignal;
  } else {
    // Fallback: use abortController and wire up other signals
    combinedSignal = abortController.signal;
    
    // Wire timeout signal to abortController
    timeoutController.signal.addEventListener('abort', () => {
      abortController.abort();
    });
    
    // Wire user signal to abortController
    const userSignal = fetchOptions.signal;
    if (userSignal) {
      userSignal.addEventListener('abort', () => {
        abortController.abort();
      });
    }
  }

  // Step 7: Audit logging - request
  audit.apiRequest(finalUrl, method, userId);

  if (phiAccess) {
    audit.phiAccess(resourceType || 'unknown', resourceId || 'unknown', userId || 'unknown');
  }

  if (phiModification) {
    audit.phiModification(
      resourceType || 'unknown',
      resourceId || 'unknown',
      userId || 'unknown',
      method
    );
  }

  // Step 8: Make request with retry logic
  const makeRequest = async (): Promise<Response> => {
    try {
      const response = await fetch(finalUrl, {
        ...fetchOptions,
        method,
        headers,
        body,
        credentials: 'include',
        signal: combinedSignal,
      });

      // Validate response size
      await validateResponseSize(response);

      // Validate CSRF token in response
      if (!skipCSRF && requiresCSRF(method)) {
        const csrfToken = response.headers.get('X-CSRF-Token');
        if (csrfToken && csrfToken !== sessionStorage.getItem('csrf-token')) {
          audit.securityViolation('CSRF_VIOLATION', { url: finalUrl, method }, userId);
          throw new Error('CSRF token validation failed');
        }
      }

      return response;
    } catch (error) {
      if (error instanceof Error && error.name === 'AbortError') {
        throw new Error('Request timeout');
      }
      throw error;
    }
  };

  const requestPromise = skipRetry
    ? makeRequest()
    : retryWithBackoff(
        makeRequest,
        {
          maxRetries: 3,
          baseDelayMs: 1000,
          maxDelayMs: 10000,
        },
        getCircuitBreakerKey(finalUrl)
      );

  // Track pending request for deduplication
  if (!skipDeduplication) {
    if (!(window as any).__pendingApiRequests) {
      (window as any).__pendingApiRequests = new Map();
    }
    (window as any).__pendingApiRequests.set(requestKey, requestPromise);
    requestPromise.finally(() => {
      (window as any).__pendingApiRequests?.delete(requestKey);
    });
  }

  try {
    const response = await requestPromise;

    // Clear timeout since request completed successfully
    timeoutControllerWrapper.clearTimeout();

    // Step 9: Audit logging - response
    audit.apiResponse(finalUrl, method, response.status, userId);

    // Step 10: Handle errors
    if (!response.ok) {
      let errorText: string;
      try {
        errorText = await response.text();
      } catch {
        errorText = response.statusText;
      }

      const error = new Error(`${response.status}: ${errorText}`) as any;
      error.statusCode = response.status;
      error.response = response;

      audit.apiError(finalUrl, method, errorText, response.status, userId);

      // Handle authentication errors
      if (response.status === 401) {
        // Trigger logout
        const event = new CustomEvent('auth:logout');
        window.dispatchEvent(event);
        localStorage.removeItem('authTokens');
        localStorage.removeItem('authUser');
        window.location.href = '/login';
      }

      throwSanitizedError(error);
    }

    // Step 11: Clone response BEFORE reading body (bodies can only be read once)
    const clonedResponse = response.clone();

    // Step 12: Parse response
    const contentType = response.headers.get('content-type') || '';
    let data: T | undefined;

    if (contentType.includes('application/json')) {
      try {
        data = await response.json();
      } catch (error) {
        throwSanitizedError(new Error('Invalid JSON response'), 'Response parsing');
      }
    } else if (contentType.startsWith('text/')) {
      data = (await response.text()) as any;
    } else {
      // For other types, return response as-is
      data = undefined;
    }

    const secureResponse: SecureApiResponse<T> = Object.assign(clonedResponse, {
      data,
    });

    return secureResponse;
  } catch (error) {
    // Clear timeout on error as well
    timeoutControllerWrapper.clearTimeout();
    audit.apiError(finalUrl, method, String(error), undefined, userId);
    throwSanitizedError(error, `API request to ${finalUrl}`);
  }
}

/**
 * Convenience method for GET requests
 */
export async function secureGet<T = any>(
  url: string,
  options?: Omit<SecureApiRequestOptions, 'method' | 'json'>
): Promise<SecureApiResponse<T>> {
  return secureApiRequest<T>(url, { ...options, method: 'GET' });
}

/**
 * Convenience method for POST requests
 */
export async function securePost<T = any>(
  url: string,
  data?: unknown,
  options?: Omit<SecureApiRequestOptions, 'method' | 'json'>
): Promise<SecureApiResponse<T>> {
  return secureApiRequest<T>(url, { ...options, method: 'POST', json: data });
}

/**
 * Convenience method for PUT requests
 */
export async function securePut<T = any>(
  url: string,
  data?: unknown,
  options?: Omit<SecureApiRequestOptions, 'method' | 'json'>
): Promise<SecureApiResponse<T>> {
  return secureApiRequest<T>(url, { ...options, method: 'PUT', json: data });
}

/**
 * Convenience method for PATCH requests
 */
export async function securePatch<T = any>(
  url: string,
  data?: unknown,
  options?: Omit<SecureApiRequestOptions, 'method' | 'json'>
): Promise<SecureApiResponse<T>> {
  return secureApiRequest<T>(url, { ...options, method: 'PATCH', json: data });
}

/**
 * Convenience method for DELETE requests
 */
export async function secureDelete<T = any>(
  url: string,
  options?: Omit<SecureApiRequestOptions, 'method' | 'json'>
): Promise<SecureApiResponse<T>> {
  return secureApiRequest<T>(url, { ...options, method: 'DELETE' });
}

