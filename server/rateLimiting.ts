import type { Request, Response, NextFunction } from 'express';

interface RateLimitConfig {
  windowMs: number;  // Time window in milliseconds
  maxRequests: number;  // Max requests per window
  keyGenerator?: (req: any) => string;  // Custom key generator
}

interface RateLimitRecord {
  count: number;
  resetTime: number;
}

// In-memory storage for rate limit tracking
const rateLimitStore = new Map<string, RateLimitRecord>();

// Cleanup old entries every 5 minutes
setInterval(() => {
  const now = Date.now();
  const keysToDelete: string[] = [];
  
  rateLimitStore.forEach((record, key) => {
    if (record.resetTime < now) {
      keysToDelete.push(key);
    }
  });
  
  keysToDelete.forEach(key => rateLimitStore.delete(key));
}, 5 * 60 * 1000);

/**
 * Rate limiting middleware for API endpoints
 * Prevents abuse and manages API costs (especially for OpenAI)
 */
export function rateLimit(config: RateLimitConfig) {
  const {
    windowMs,
    maxRequests,
    keyGenerator = (req: any) => {
      // Default: Use userId from session
      return req.session?.userId || req.ip || 'anonymous';
    }
  } = config;

  return (req: Request, res: Response, next: NextFunction) => {
    const key = keyGenerator(req as any);
    const now = Date.now();
    
    let record = rateLimitStore.get(key);
    
    if (!record || record.resetTime < now) {
      // Create new record or reset expired one
      record = {
        count: 0,
        resetTime: now + windowMs
      };
      rateLimitStore.set(key, record);
    }
    
    record.count++;
    
    // Set rate limit headers
    res.setHeader('X-RateLimit-Limit', maxRequests);
    res.setHeader('X-RateLimit-Remaining', Math.max(0, maxRequests - record.count));
    res.setHeader('X-RateLimit-Reset', new Date(record.resetTime).toISOString());
    
    if (record.count > maxRequests) {
      const retryAfter = Math.ceil((record.resetTime - now) / 1000);
      res.setHeader('Retry-After', retryAfter);
      
      return res.status(429).json({
        message: 'Too many requests. Please try again later.',
        retryAfter: retryAfter,
        limit: maxRequests,
        windowMs: windowMs
      });
    }
    
    next();
  };
}

/**
 * AI-specific rate limiter with higher costs
 * More restrictive to prevent OpenAI quota exhaustion
 */
export const aiRateLimit = rateLimit({
  windowMs: 60 * 1000, // 1 minute
  maxRequests: 10, // 10 AI requests per minute per user
});

/**
 * Batch operations rate limiter
 * Even more restrictive for batch processing
 */
export const batchRateLimit = rateLimit({
  windowMs: 60 * 1000, // 1 minute
  maxRequests: 3, // 3 batch requests per minute per user
});

/**
 * Get current rate limit stats for a user
 */
export function getRateLimitStats(userId: string): {
  count: number;
  limit: number;
  remaining: number;
  resetTime: Date;
} | null {
  const record = rateLimitStore.get(userId);
  
  if (!record) {
    return null;
  }
  
  // Use default AI rate limit config for stats
  const maxRequests = 10;
  
  return {
    count: record.count,
    limit: maxRequests,
    remaining: Math.max(0, maxRequests - record.count),
    resetTime: new Date(record.resetTime)
  };
}
