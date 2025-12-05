/**
 * Error Sanitization
 * 
 * HIPAA Security: Prevents information leakage through error messages
 * that could expose PHI, system internals, or security vulnerabilities.
 */

export interface SanitizedError {
  message: string;
  code: string;
  statusCode?: number;
  userMessage: string;
  originalError?: Error;
}

/**
 * Sanitizes error messages to prevent information leakage
 */
export function sanitizeError(error: unknown, context?: string): SanitizedError {
  // If error is already sanitized, return it as-is (avoid double sanitization)
  if (error && typeof error === 'object' && 'code' in error && 'userMessage' in error && 'message' in error) {
    const alreadySanitized = error as SanitizedError;
    // Preserve context if provided
    if (context && !alreadySanitized.message.includes(context)) {
      return {
        ...alreadySanitized,
        message: `${alreadySanitized.message} (${context})`,
      };
    }
    return alreadySanitized;
  }

  // Default sanitized error
  const sanitized: SanitizedError = {
    message: 'An error occurred',
    code: 'UNKNOWN_ERROR',
    userMessage: 'An unexpected error occurred. Please try again later.',
  };

  if (error instanceof Response) {
    // HTTP Response error
    const status = error.status;
    sanitized.statusCode = status;
    sanitized.code = `HTTP_${status}`;

    // Generic user-friendly messages based on status code
    switch (status) {
      case 400:
        sanitized.userMessage = 'Invalid request. Please check your input and try again.';
        break;
      case 401:
        sanitized.userMessage = 'Your session has expired. Please log in again.';
        sanitized.code = 'UNAUTHORIZED';
        break;
      case 403:
        sanitized.userMessage = 'You do not have permission to perform this action.';
        sanitized.code = 'FORBIDDEN';
        break;
      case 404:
        sanitized.userMessage = 'The requested resource was not found.';
        sanitized.code = 'NOT_FOUND';
        break;
      case 408:
        sanitized.userMessage = 'The request timed out. Please try again.';
        sanitized.code = 'TIMEOUT';
        break;
      case 429:
        sanitized.userMessage = 'Too many requests. Please wait a moment and try again.';
        sanitized.code = 'RATE_LIMITED';
        break;
      case 500:
        sanitized.userMessage = 'A server error occurred. Please try again later.';
        sanitized.code = 'SERVER_ERROR';
        break;
      case 502:
      case 503:
      case 504:
        sanitized.userMessage = 'The service is temporarily unavailable. Please try again later.';
        sanitized.code = 'SERVICE_UNAVAILABLE';
        break;
      default:
        sanitized.userMessage = 'An error occurred. Please try again.';
    }

    // Log original error for debugging (server-side only)
    if (import.meta.env.DEV) {
      sanitized.originalError = new Error(`HTTP ${status}: ${error.statusText}`);
    }
  } else if (error instanceof Error) {
    // JavaScript Error
    const errorMessage = error.message.toLowerCase();

    // Check for sensitive information patterns
    const sensitivePatterns = [
      /password/i,
      /token/i,
      /secret/i,
      /key/i,
      /credential/i,
      /auth/i,
      /jwt/i,
      /session/i,
      /cookie/i,
      /sql/i,
      /database/i,
      /connection/i,
      /stack trace/i,
      /file path/i,
      /internal/i,
    ];

    // If error contains sensitive info, use generic message
    if (sensitivePatterns.some(pattern => pattern.test(errorMessage))) {
      sanitized.userMessage = 'An authentication error occurred. Please log in again.';
      sanitized.code = 'AUTH_ERROR';
    } else {
      // Use a sanitized version of the error message
      sanitized.message = sanitizeErrorMessage(error.message);
      sanitized.code = error.name || 'ERROR';
      sanitized.userMessage = getUserFriendlyMessage(error);
    }

    // In development, include original error
    if (import.meta.env.DEV) {
      sanitized.originalError = error;
    }
  } else if (typeof error === 'string') {
    // String error
    sanitized.message = sanitizeErrorMessage(error);
    sanitized.userMessage = getUserFriendlyMessage(new Error(error));
  }

  return sanitized;
}

/**
 * Sanitizes error message to remove sensitive information
 */
function sanitizeErrorMessage(message: string): string {
  // Remove file paths
  let sanitized = message.replace(/\/[^\s]+/g, '[path]');

  // Remove IP addresses
  sanitized = sanitized.replace(/\d+\.\d+\.\d+\.\d+/g, '[ip]');

  // Remove email addresses
  sanitized = sanitized.replace(/[^\s]+@[^\s]+/g, '[email]');

  // Remove tokens (long alphanumeric strings)
  sanitized = sanitized.replace(/[a-zA-Z0-9]{32,}/g, '[token]');

  // Remove stack traces
  sanitized = sanitized.split('\n')[0];

  return sanitized.trim();
}

/**
 * Gets user-friendly error message
 */
function getUserFriendlyMessage(error: Error): string {
  const message = error.message.toLowerCase();

  if (message.includes('network') || message.includes('fetch')) {
    return 'Network error. Please check your internet connection and try again.';
  }

  if (message.includes('timeout')) {
    return 'The request timed out. Please try again.';
  }

  if (message.includes('abort')) {
    return 'The request was cancelled.';
  }

  if (message.includes('cors')) {
    return 'A connection error occurred. Please try again.';
  }

  if (message.includes('json')) {
    return 'Invalid response format. Please try again.';
  }

  return 'An error occurred. Please try again.';
}

/**
 * Creates a safe error object for logging (without sensitive data)
 */
export function createSafeErrorLog(error: unknown, context?: string): Record<string, any> {
  const sanitized = sanitizeError(error, context);
  return {
    code: sanitized.code,
    statusCode: sanitized.statusCode,
    message: sanitized.message,
    context,
    timestamp: new Date().toISOString(),
  };
}

