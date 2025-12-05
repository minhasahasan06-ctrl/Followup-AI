/**
 * Rate Limiting and Request Deduplication
 * 
 * HIPAA Security: Prevents abuse, DoS attacks, and ensures
 * fair resource usage while maintaining audit trails.
 */

export interface RateLimitConfig {
  maxRequests: number;
  windowMs: number;
  keyGenerator?: (url: string, method: string) => string;
}

export interface RateLimitResult {
  allowed: boolean;
  remaining: number;
  resetTime: number;
  retryAfter?: number;
}

interface RequestEntry {
  count: number;
  resetTime: number;
  requests: Map<string, number>; // URL -> timestamp for deduplication
}

class RateLimiter {
  private requests: Map<string, RequestEntry> = new Map();
  private deduplicationWindow: number = 1000; // 1 second deduplication window
  private pendingRequests: Map<string, Promise<any>> = new Map();

  /**
   * Checks if a request should be rate limited
   */
  checkRateLimit(
    key: string,
    config: RateLimitConfig
  ): RateLimitResult {
    const now = Date.now();
    let entry = this.requests.get(key);

    // Clean up expired entries
    if (entry && entry.resetTime < now) {
      this.requests.delete(key);
      entry = undefined;
    }

    // Create new entry if needed
    if (!entry) {
      entry = {
        count: 0,
        resetTime: now + config.windowMs,
        requests: new Map(),
      };
      this.requests.set(key, entry);
    }

    // Check rate limit
    if (entry.count >= config.maxRequests) {
      const retryAfter = Math.ceil((entry.resetTime - now) / 1000);
      return {
        allowed: false,
        remaining: 0,
        resetTime: entry.resetTime,
        retryAfter,
      };
    }

    // Increment count
    entry.count++;

    return {
      allowed: true,
      remaining: config.maxRequests - entry.count,
      resetTime: entry.resetTime,
    };
  }

  /**
   * Checks if a request is a duplicate (within deduplication window)
   */
  isDuplicate(url: string, method: string, body?: string): boolean {
    const requestKey = this.getRequestKey(url, method, body);
    const now = Date.now();
    const entry = this.requests.get('dedupe');

    if (!entry) {
      return false;
    }

    const lastRequestTime = entry.requests.get(requestKey);
    if (lastRequestTime && now - lastRequestTime < this.deduplicationWindow) {
      return true;
    }

    // Update timestamp
    entry.requests.set(requestKey, now);

    // Clean up old entries (older than deduplication window)
    for (const [key, timestamp] of entry.requests.entries()) {
      if (now - timestamp > this.deduplicationWindow) {
        entry.requests.delete(key);
      }
    }

    return false;
  }

  /**
   * Tracks a pending request for deduplication
   */
  trackPendingRequest<T>(key: string, promise: Promise<T>): Promise<T> {
    // If there's already a pending request with the same key, return it
    const existing = this.pendingRequests.get(key);
    if (existing) {
      return existing as Promise<T>;
    }

    // Track the new request
    this.pendingRequests.set(key, promise);

    // Clean up when request completes
    promise
      .finally(() => {
        this.pendingRequests.delete(key);
      })
      .catch(() => {
        // Ignore errors, cleanup already handled
      });

    return promise;
  }

  /**
   * Generates a unique key for request deduplication
   */
  private getRequestKey(url: string, method: string, body?: string): string {
    // Create a hash-like key from URL, method, and body
    const bodyHash = body ? this.simpleHash(body) : '';
    return `${method}:${url}:${bodyHash}`;
  }

  /**
   * Simple hash function for request body
   */
  private simpleHash(str: string): string {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      const char = str.charCodeAt(i);
      hash = (hash << 5) - hash + char;
      hash = hash & hash; // Convert to 32-bit integer
    }
    return hash.toString(36);
  }

  /**
   * Cleans up old entries (call periodically)
   */
  cleanup(): void {
    const now = Date.now();
    for (const [key, entry] of this.requests.entries()) {
      if (entry.resetTime < now) {
        this.requests.delete(key);
      }
    }
  }
}

// Global rate limiter instance
const globalRateLimiter = new RateLimiter();

// Cleanup every 5 minutes
if (typeof window !== 'undefined') {
  setInterval(() => {
    globalRateLimiter.cleanup();
  }, 5 * 60 * 1000);
}

/**
 * Default rate limit configurations
 */
export const DEFAULT_RATE_LIMITS: Record<string, RateLimitConfig> = {
  default: {
    maxRequests: 100,
    windowMs: 60 * 1000, // 1 minute
  },
  strict: {
    maxRequests: 20,
    windowMs: 60 * 1000, // 1 minute
  },
  upload: {
    maxRequests: 10,
    windowMs: 60 * 1000, // 1 minute
  },
  auth: {
    maxRequests: 5,
    windowMs: 60 * 1000, // 1 minute
  },
};

/**
 * Gets rate limit key for a request
 */
export function getRateLimitKey(url: string, method: string): string {
  // Use origin + method as key
  try {
    const urlObj = new URL(url, window.location.origin);
    return `${urlObj.origin}:${method}`;
  } catch {
    return `global:${method}`;
  }
}

/**
 * Checks rate limit for a request
 */
export function checkRateLimit(
  url: string,
  method: string,
  config: RateLimitConfig = DEFAULT_RATE_LIMITS.default
): RateLimitResult {
  const key = getRateLimitKey(url, method);
  return globalRateLimiter.checkRateLimit(key, config);
}

/**
 * Checks if request is duplicate
 */
export function isDuplicateRequest(
  url: string,
  method: string,
  body?: string
): boolean {
  return globalRateLimiter.isDuplicate(url, method, body);
}

/**
 * Tracks pending request for deduplication
 */
export function trackPendingRequest<T>(
  key: string,
  promise: Promise<T>
): Promise<T> {
  return globalRateLimiter.trackPendingRequest(key, promise);
}

