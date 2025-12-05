/**
 * CSRF Protection
 * 
 * HIPAA Security: Prevents Cross-Site Request Forgery attacks
 * that could lead to unauthorized PHI access or modification.
 */

const CSRF_TOKEN_KEY = 'csrf-token';
const CSRF_HEADER_NAME = 'X-CSRF-Token';

/**
 * Generates a CSRF token
 */
export function generateCSRFToken(): string {
  const array = new Uint8Array(32);
  crypto.getRandomValues(array);
  return Array.from(array, byte => byte.toString(16).padStart(2, '0')).join('');
}

/**
 * Gets or creates CSRF token
 */
export function getCSRFToken(): string {
  if (typeof window === 'undefined') {
    return '';
  }

  let token = sessionStorage.getItem(CSRF_TOKEN_KEY);
  if (!token) {
    token = generateCSRFToken();
    sessionStorage.setItem(CSRF_TOKEN_KEY, token);
  }
  return token;
}

/**
 * Validates CSRF token from response header
 */
export function validateCSRFToken(response: Response): boolean {
  const responseToken = response.headers.get(CSRF_HEADER_NAME);
  if (!responseToken) {
    // Some endpoints may not require CSRF (GET requests, public endpoints)
    return true;
  }

  const storedToken = getCSRFToken();
  return responseToken === storedToken;
}

/**
 * Adds CSRF token to request headers
 */
export function addCSRFHeader(headers: HeadersInit): HeadersInit {
  const token = getCSRFToken();
  if (!token) {
    return headers;
  }

  const headersObj = headers instanceof Headers
    ? Object.fromEntries(headers.entries())
    : Array.isArray(headers)
    ? Object.fromEntries(headers)
    : headers;

  return {
    ...headersObj,
    [CSRF_HEADER_NAME]: token,
  };
}

/**
 * Checks if a request method requires CSRF protection
 */
export function requiresCSRF(method: string): boolean {
  const protectedMethods = ['POST', 'PUT', 'PATCH', 'DELETE'];
  return protectedMethods.includes(method.toUpperCase());
}

