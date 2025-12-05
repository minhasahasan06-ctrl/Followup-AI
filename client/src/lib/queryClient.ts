/**
 * Secure Query Client for React Query
 * 
 * HIPAA-Compliant query client with comprehensive security features.
 * This module provides a secure wrapper around React Query that integrates
 * all security measures including SSRF protection, rate limiting, CSRF
 * protection, audit logging, and error sanitization.
 */

import { QueryClient, QueryFunction } from "@tanstack/react-query";
import {
  secureApiRequest,
  secureGet,
  type SecureApiRequestOptions,
} from "./security/secureApiClient";
import { sanitizeError } from "./security/errorSanitizer";

/**
 * Legacy apiRequest function for backward compatibility
 * @deprecated Use secureApiRequest from './security/secureApiClient' instead
 */
export async function apiRequest(
  url: string,
  options?: RequestInit & { json?: unknown }
): Promise<Response> {
  const { json, ...fetchOptions } = options || {};
  
  const secureOptions: SecureApiRequestOptions = {
    ...fetchOptions,
    json,
  };

  const response = await secureApiRequest(url, secureOptions);
  return response;
}

/**
 * Query function factory with security features
 */
type UnauthorizedBehavior = "returnNull" | "throw";

export const getQueryFn: <T>(options: {
  on401: UnauthorizedBehavior;
  phiAccess?: boolean;
  resourceType?: string;
  resourceId?: string;
}) => QueryFunction<T> =
  ({ on401: unauthorizedBehavior, phiAccess = false, resourceType, resourceId }) =>
  async ({ queryKey }) => {
    // Require first queryKey entry to be a string URL path
    if (queryKey.length === 0 || typeof queryKey[0] !== 'string') {
      const error = new Error('Query key must start with a string URL path');
      throw sanitizeError(error);
    }
    
    // Build URL from query key
    // Support both simple string keys and array keys with optional trailing params
    let url: string;
    if (queryKey.length === 1) {
      url = queryKey[0] as string;
    } else {
      // Join only the string elements, ignore objects/arrays (they're for cache invalidation)
      const stringParts = queryKey.filter(k => typeof k === 'string') as string[];
      url = stringParts.join("/");
    }

    try {
      // Use secure API client
      const response = await secureGet<any>(url, {
        phiAccess,
        resourceType,
        resourceId,
        skipRetry: false, // Allow retries for queries
      });

      // Return parsed data
      return response.data;
    } catch (error) {
      // Handle 401 errors (secureGet throws for 401, so we catch it here)
      if (error && typeof error === 'object' && 'statusCode' in error) {
        const statusCode = (error as any).statusCode;
        if (statusCode === 401 && unauthorizedBehavior === "returnNull") {
          return null;
        }
      }

      // Re-throw sanitized error
      throw sanitizeError(error, `Query function for ${url}`);
    }
  };

/**
 * Secure Query Client with HIPAA-compliant defaults
 */
export const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      queryFn: getQueryFn({ on401: "throw" }),
      refetchInterval: false,
      refetchOnWindowFocus: false,
      staleTime: Infinity,
      retry: (failureCount, error: any) => {
        // Don't retry on 4xx errors (client errors)
        if (error?.statusCode && error.statusCode >= 400 && error.statusCode < 500) {
          return false;
        }
        // Retry up to 2 times for network/server errors
        return failureCount < 2;
      },
      retryDelay: (attemptIndex) => Math.min(1000 * 2 ** attemptIndex, 30000),
    },
    mutations: {
      retry: (failureCount, error: any) => {
        // Don't retry mutations on client errors
        if (error?.statusCode && error.statusCode >= 400 && error.statusCode < 500) {
          return false;
        }
        // Retry once for network/server errors
        return failureCount < 1;
      },
      retryDelay: (attemptIndex) => Math.min(1000 * 2 ** attemptIndex, 10000),
    },
  },
});

// Export secure API functions for direct use
export { secureApiRequest, secureGet, securePost, securePut, securePatch, secureDelete } from "./security/secureApiClient";
export * from "./security";
