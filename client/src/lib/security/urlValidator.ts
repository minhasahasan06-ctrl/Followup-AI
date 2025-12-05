/**
 * URL Validation and SSRF Protection
 * 
 * HIPAA Security: Prevents Server-Side Request Forgery (SSRF) attacks
 * that could expose internal network resources or PHI data.
 */

export interface UrlValidationResult {
  isValid: boolean;
  sanitizedUrl: string;
  error?: string;
}

// Allowed backend domains (whitelist approach)
const ALLOWED_BACKEND_DOMAINS = new Set<string>([
  'localhost',
  '127.0.0.1',
  // Add production domains here
  ...(import.meta.env.VITE_ALLOWED_BACKEND_DOMAINS?.split(',') || []),
]);

// Blocked protocols and schemes
const BLOCKED_PROTOCOLS = new Set(['file:', 'ftp:', 'javascript:', 'data:', 'vbscript:']);

// Private IP ranges (RFC 1918, RFC 4193, RFC 5735)
const PRIVATE_IP_PATTERNS = [
  /^10\./,
  /^172\.(1[6-9]|2[0-9]|3[01])\./,
  /^192\.168\./,
  /^127\./,
  /^169\.254\./,
  /^::1$/,
  /^fc00:/,
  /^fe80:/,
];

/**
 * Validates and sanitizes URLs to prevent SSRF attacks
 */
export function validateAndSanitizeUrl(url: string, baseUrl?: string): UrlValidationResult {
  try {
    // Step 1: Basic input validation
    if (!url || typeof url !== 'string') {
      return {
        isValid: false,
        sanitizedUrl: '',
        error: 'Invalid URL: must be a non-empty string',
      };
    }

    // Step 2: Check for blocked protocols
    const lowerUrl = url.toLowerCase().trim();
    for (const protocol of BLOCKED_PROTOCOLS) {
      if (lowerUrl.startsWith(protocol)) {
        return {
          isValid: false,
          sanitizedUrl: '',
          error: `Blocked protocol: ${protocol}`,
        };
      }
    }

    // Step 3: Parse URL
    let parsedUrl: URL;
    try {
      // If it's a relative URL, resolve against baseUrl or current origin
      if (url.startsWith('/')) {
        const base = baseUrl || window.location.origin;
        parsedUrl = new URL(url, base);
      } else if (url.startsWith('http://') || url.startsWith('https://')) {
        parsedUrl = new URL(url);
      } else {
        // Relative path without leading slash
        const base = baseUrl || window.location.origin;
        parsedUrl = new URL('/' + url, base);
      }
    } catch (e) {
      return {
        isValid: false,
        sanitizedUrl: '',
        error: `Invalid URL format: ${e instanceof Error ? e.message : 'Unknown error'}`,
      };
    }

    // Step 4: Validate protocol (only http/https allowed)
    if (parsedUrl.protocol !== 'http:' && parsedUrl.protocol !== 'https:') {
      return {
        isValid: false,
        sanitizedUrl: '',
        error: `Only HTTP/HTTPS protocols allowed, got: ${parsedUrl.protocol}`,
      };
    }

    // Step 5: Check for private/internal IP addresses
    const hostname = parsedUrl.hostname;
    if (isPrivateIP(hostname)) {
      return {
        isValid: false,
        sanitizedUrl: '',
        error: 'Private/internal IP addresses are not allowed',
      };
    }

    // Step 6: Validate domain whitelist (if configured)
    if (ALLOWED_BACKEND_DOMAINS.size > 0) {
      const domain = extractDomain(hostname);
      if (!ALLOWED_BACKEND_DOMAINS.has(domain) && !ALLOWED_BACKEND_DOMAINS.has(hostname)) {
        // Allow if it's the current origin (same-origin requests)
        if (parsedUrl.origin !== window.location.origin) {
          return {
            isValid: false,
            sanitizedUrl: '',
            error: `Domain not in whitelist: ${domain}`,
          };
        }
      }
    }

    // Step 7: Sanitize path to prevent path traversal
    const sanitizedPath = sanitizePath(parsedUrl.pathname);
    parsedUrl.pathname = sanitizedPath;

    // Step 8: Remove dangerous query parameters
    const sanitizedSearch = sanitizeQueryParams(parsedUrl.searchParams);
    parsedUrl.search = sanitizedSearch.toString();

    return {
      isValid: true,
      sanitizedUrl: parsedUrl.toString(),
    };
  } catch (error) {
    return {
      isValid: false,
      sanitizedUrl: '',
      error: `URL validation error: ${error instanceof Error ? error.message : 'Unknown error'}`,
    };
  }
}

/**
 * Checks if a hostname resolves to a private IP address
 */
function isPrivateIP(hostname: string): boolean {
  // Check if it's an IP address
  const ipv4Pattern = /^(\d{1,3}\.){3}\d{1,3}$/;
  const ipv6Pattern = /^([0-9a-fA-F]{0,4}:){2,7}[0-9a-fA-F]{0,4}$/;

  if (ipv4Pattern.test(hostname)) {
    return PRIVATE_IP_PATTERNS.some(pattern => pattern.test(hostname));
  }

  if (ipv6Pattern.test(hostname)) {
    return PRIVATE_IP_PATTERNS.some(pattern => pattern.test(hostname));
  }

  // For domain names, we can't check IP without DNS lookup
  // But we can check for localhost variants
  const localhostVariants = ['localhost', 'local', '0.0.0.0', '127.0.0.1'];
  return localhostVariants.includes(hostname.toLowerCase());
}

/**
 * Extracts domain from hostname (removes subdomains)
 */
function extractDomain(hostname: string): string {
  const parts = hostname.split('.');
  if (parts.length >= 2) {
    return parts.slice(-2).join('.');
  }
  return hostname;
}

/**
 * Sanitizes URL path to prevent path traversal attacks
 */
function sanitizePath(path: string): string {
  // Remove path traversal sequences
  let sanitized = path
    .replace(/\.\./g, '') // Remove ..
    .replace(/\/\/+/g, '/') // Remove double slashes
    .replace(/\/\./g, '/') // Remove /. sequences
    .replace(/\.\//g, '/'); // Remove ./ sequences

  // Ensure it starts with /
  if (!sanitized.startsWith('/')) {
    sanitized = '/' + sanitized;
  }

  // Remove trailing slashes (except root)
  if (sanitized.length > 1 && sanitized.endsWith('/')) {
    sanitized = sanitized.slice(0, -1);
  }

  return sanitized;
}

/**
 * Sanitizes query parameters to remove potentially dangerous ones
 */
function sanitizeQueryParams(params: URLSearchParams): URLSearchParams {
  const sanitized = new URLSearchParams();
  const dangerousParams = ['__proto__', 'constructor', 'prototype'];

  for (const [key, value] of params.entries()) {
    // Skip dangerous parameter names
    if (dangerousParams.some(dangerous => key.toLowerCase().includes(dangerous))) {
      continue;
    }

    // Validate parameter value length (prevent DoS)
    if (value.length > 10000) {
      continue;
    }

    sanitized.append(key, value);
  }

  return sanitized;
}

/**
 * Determines if a URL should be routed to Python backend
 */
export function shouldRouteToPythonBackend(url: string): boolean {
  const pythonBackendRoutes = [
    '/api/v1/video-ai',
    '/api/v1/audio-ai',
    '/api/v1/trends',
    '/api/v1/alerts',
    '/api/v1/guided-audio-exam',
    '/api/v1/guided-exam',
    '/api/v1/gait-analysis',
    '/api/v1/tremor',
    '/api/v1/mental-health',
    '/api/v1/pain-tracking',
    '/api/v1/symptom-journal',
  ];

  const normalizedUrl = url.startsWith('/') ? url : `/${url}`;
  return pythonBackendRoutes.some(route => normalizedUrl.startsWith(route));
}

/**
 * Gets the Python backend URL with validation
 */
export function getPythonBackendUrl(url: string): string {
  const PYTHON_BACKEND_URL = import.meta.env.VITE_PYTHON_BACKEND_URL || 'http://localhost:8000';
  
  // Validate the base URL
  const baseValidation = validateAndSanitizeUrl(PYTHON_BACKEND_URL);
  if (!baseValidation.isValid) {
    throw new Error(`Invalid Python backend URL configuration: ${baseValidation.error}`);
  }

  // If URL is already absolute, validate and return
  if (url.startsWith('http://') || url.startsWith('https://')) {
    const validation = validateAndSanitizeUrl(url);
    if (!validation.isValid) {
      throw new Error(`Invalid absolute URL: ${validation.error}`);
    }
    return validation.sanitizedUrl;
  }

  // Normalize relative URL
  const normalizedUrl = url.startsWith('/') ? url : `/${url}`;
  
  // Validate the full URL
  const fullUrl = `${baseValidation.sanitizedUrl}${normalizedUrl}`;
  const validation = validateAndSanitizeUrl(fullUrl, baseValidation.sanitizedUrl);
  
  if (!validation.isValid) {
    throw new Error(`Invalid Python backend URL: ${validation.error}`);
  }

  return validation.sanitizedUrl;
}

