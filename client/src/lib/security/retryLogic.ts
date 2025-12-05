/**
 * Retry Logic with Exponential Backoff and Circuit Breaker
 * 
 * HIPAA Security: Ensures reliable data transmission and prevents
 * cascading failures while maintaining audit trails.
 */

export interface RetryConfig {
  maxRetries: number;
  baseDelayMs: number;
  maxDelayMs: number;
  retryableStatuses: number[];
  retryableErrors: string[];
  enableCircuitBreaker: boolean;
}

export interface CircuitBreakerState {
  state: 'closed' | 'open' | 'half-open';
  failureCount: number;
  lastFailureTime: number;
  successCount: number;
}

const DEFAULT_RETRY_CONFIG: RetryConfig = {
  maxRetries: 3,
  baseDelayMs: 1000,
  maxDelayMs: 30000,
  retryableStatuses: [408, 429, 500, 502, 503, 504],
  retryableErrors: ['NetworkError', 'TimeoutError', 'AbortError'],
  enableCircuitBreaker: true,
};

class CircuitBreaker {
  private breakers: Map<string, CircuitBreakerState> = new Map();
  private readonly failureThreshold = 5;
  private readonly resetTimeout = 60000; // 1 minute
  private readonly halfOpenMaxAttempts = 3;

  /**
   * Checks if circuit breaker allows the request
   */
  canProceed(key: string): boolean {
    const breaker = this.breakers.get(key) || {
      state: 'closed' as const,
      failureCount: 0,
      lastFailureTime: 0,
      successCount: 0,
    };

    const now = Date.now();

    // Check if we should transition from open to half-open
    if (breaker.state === 'open') {
      if (now - breaker.lastFailureTime > this.resetTimeout) {
        breaker.state = 'half-open';
        breaker.successCount = 0;
      } else {
        return false;
      }
    }

    this.breakers.set(key, breaker);
    return true;
  }

  /**
   * Records a success
   */
  recordSuccess(key: string): void {
    const breaker = this.breakers.get(key) || {
      state: 'closed' as const,
      failureCount: 0,
      lastFailureTime: 0,
      successCount: 0,
    };

    if (breaker.state === 'half-open') {
      breaker.successCount++;
      if (breaker.successCount >= this.halfOpenMaxAttempts) {
        breaker.state = 'closed';
        breaker.failureCount = 0;
        breaker.successCount = 0;
      }
    } else {
      breaker.failureCount = 0;
    }

    this.breakers.set(key, breaker);
  }

  /**
   * Records a failure
   */
  recordFailure(key: string): void {
    const breaker = this.breakers.get(key) || {
      state: 'closed' as const,
      failureCount: 0,
      lastFailureTime: 0,
      successCount: 0,
    };

    breaker.failureCount++;
    breaker.lastFailureTime = Date.now();

    if (breaker.failureCount >= this.failureThreshold) {
      breaker.state = 'open';
    }

    this.breakers.set(key, breaker);
  }

  /**
   * Gets circuit breaker state
   */
  getState(key: string): CircuitBreakerState {
    return (
      this.breakers.get(key) || {
        state: 'closed',
        failureCount: 0,
        lastFailureTime: 0,
        successCount: 0,
      }
    );
  }
}

const globalCircuitBreaker = new CircuitBreaker();

/**
 * Calculates exponential backoff delay
 */
function calculateBackoffDelay(
  attempt: number,
  baseDelayMs: number,
  maxDelayMs: number
): number {
  const delay = Math.min(baseDelayMs * Math.pow(2, attempt), maxDelayMs);
  // Add jitter to prevent thundering herd
  const jitter = Math.random() * 0.3 * delay;
  return Math.floor(delay + jitter);
}

/**
 * Checks if an error is retryable
 */
function isRetryableError(
  error: Error | Response,
  config: RetryConfig
): boolean {
  if (error instanceof Response) {
    return config.retryableStatuses.includes(error.status);
  }

  if (error instanceof Error) {
    return config.retryableErrors.some(
      retryableError => error.name === retryableError || error.message.includes(retryableError)
    );
  }

  return false;
}

/**
 * Retries a function with exponential backoff
 */
export async function retryWithBackoff<T>(
  fn: () => Promise<T>,
  config: Partial<RetryConfig> = {},
  circuitBreakerKey?: string
): Promise<T> {
  const finalConfig = { ...DEFAULT_RETRY_CONFIG, ...config };
  let lastError: Error | Response | null = null;

  // Check circuit breaker if enabled
  if (finalConfig.enableCircuitBreaker && circuitBreakerKey) {
    if (!globalCircuitBreaker.canProceed(circuitBreakerKey)) {
      const state = globalCircuitBreaker.getState(circuitBreakerKey);
      throw new Error(
        `Circuit breaker is open. Too many failures. Retry after ${Math.ceil(
          (state.lastFailureTime + 60000 - Date.now()) / 1000
        )} seconds`
      );
    }
  }

  for (let attempt = 0; attempt <= finalConfig.maxRetries; attempt++) {
    try {
      const result = await fn();

      // Record success in circuit breaker
      if (finalConfig.enableCircuitBreaker && circuitBreakerKey) {
        globalCircuitBreaker.recordSuccess(circuitBreakerKey);
      }

      return result;
    } catch (error) {
      lastError = error as Error | Response;

      // Don't retry if error is not retryable
      if (!isRetryableError(lastError, finalConfig)) {
        if (finalConfig.enableCircuitBreaker && circuitBreakerKey) {
          globalCircuitBreaker.recordFailure(circuitBreakerKey);
        }
        throw lastError;
      }

      // Don't retry on last attempt
      if (attempt === finalConfig.maxRetries) {
        if (finalConfig.enableCircuitBreaker && circuitBreakerKey) {
          globalCircuitBreaker.recordFailure(circuitBreakerKey);
        }
        throw lastError;
      }

      // Calculate delay and wait
      const delay = calculateBackoffDelay(
        attempt,
        finalConfig.baseDelayMs,
        finalConfig.maxDelayMs
      );
      await new Promise(resolve => setTimeout(resolve, delay));
    }
  }

  // This should never be reached, but TypeScript needs it
  throw lastError || new Error('Retry failed');
}

/**
 * Gets circuit breaker key for a URL
 */
export function getCircuitBreakerKey(url: string): string {
  try {
    const urlObj = new URL(url, window.location.origin);
    return urlObj.origin;
  } catch {
    return 'global';
  }
}

