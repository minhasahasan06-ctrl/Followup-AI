import { QueryClient, QueryFunction } from "@tanstack/react-query";

// GCP Backend URL - Primary API endpoint for production
// In production: Set VITE_API_URL to your GCP Cloud Run URL
// In development: Falls back to local Python backend
const GCP_API_URL = import.meta.env.VITE_API_URL || "";

// Python backend URL (FastAPI on port 8000) - Used for development
const PYTHON_BACKEND_URL = import.meta.env.VITE_PYTHON_BACKEND_URL || "http://localhost:8000";

// Determine which backend to use
const getBackendUrl = (): string => {
  // If GCP API URL is set, use it for all API calls
  if (GCP_API_URL) {
    return GCP_API_URL;
  }
  // Fall back to Python backend URL for local development
  return PYTHON_BACKEND_URL;
};

// Helper to determine if URL should go to Python backend
function getPythonBackendUrl(url: string): string {
  // If it's already an absolute URL (starts with http:// or https://), return unchanged
  if (url.startsWith("http://") || url.startsWith("https://")) {
    return url;
  }

  // Normalize URL (ensure leading slash for relative paths)
  const normalizedUrl = url.startsWith("/") ? url : `/${url}`;
  
  // If GCP API URL is configured, route all API calls to GCP
  if (GCP_API_URL && normalizedUrl.startsWith("/api/")) {
    return `${GCP_API_URL}${normalizedUrl}`;
  }

  // Routes that should go to Python FastAPI backend
  // NOTE: mental-health routes go through Express proxy for authentication/CORS handling
  if (
    normalizedUrl.startsWith("/api/v1/video-ai") ||
    normalizedUrl.startsWith("/api/v1/audio-ai") ||
    normalizedUrl.startsWith("/api/v1/trends") ||
    normalizedUrl.startsWith("/api/v1/alerts") ||
    normalizedUrl.startsWith("/api/v1/guided-audio-exam") ||
    normalizedUrl.startsWith("/api/v1/guided-exam") ||
    normalizedUrl.startsWith("/api/v1/gait-analysis") ||
    normalizedUrl.startsWith("/api/v1/tremor") ||
    normalizedUrl.startsWith("/api/v1/automation") ||
    normalizedUrl.startsWith("/api/v1/webhooks") ||
    normalizedUrl.startsWith("/api/v1/devices") ||
    normalizedUrl.startsWith("/api/v1/health-analytics") ||
    normalizedUrl.startsWith("/api/v1/epidemiology") ||
    normalizedUrl.startsWith("/api/v1/occupational") ||
    normalizedUrl.startsWith("/api/v1/genetic") ||
    normalizedUrl.startsWith("/api/v1/pharmaco") ||
    normalizedUrl.startsWith("/api/v1/infectious") ||
    normalizedUrl.startsWith("/api/v1/vaccine")
  ) {
    return `${PYTHON_BACKEND_URL}${normalizedUrl}`;
  }
  
  // Handle /api/py/ prefix - strip 'py' and route to Python backend
  if (normalizedUrl.startsWith("/api/py/")) {
    const strippedUrl = normalizedUrl.replace("/api/py/", "/api/");
    return `${PYTHON_BACKEND_URL}${strippedUrl}`;
  }

  return normalizedUrl;
}

async function throwIfResNotOk(res: Response) {
  if (!res.ok) {
    const text = (await res.text()) || res.statusText;
    throw new Error(`${res.status}: ${text}`);
  }
}

export async function apiRequest(
  url: string,
  options?: RequestInit & { json?: unknown }
): Promise<Response> {
  const { json, ...fetchOptions } = options || {};
  
  // Route to Python backend if needed
  const finalUrl = getPythonBackendUrl(url);
  
  // Prepare headers and body
  const headers = { ...((fetchOptions.headers as Record<string, string>) || {}) };
  let body = fetchOptions.body;
  
  // If json payload is provided, stringify and set Content-Type
  if (json !== undefined) {
    headers["Content-Type"] = "application/json";
    body = JSON.stringify(json);
  }
  // If body is already FormData, don't set Content-Type (browser sets it with boundary)
  // For other body types, let them pass through as-is
  
  const res = await fetch(finalUrl, {
    ...fetchOptions,
    headers,
    body,
    credentials: "include", // Always include credentials for authentication
  });

  await throwIfResNotOk(res);
  return res;
}

type UnauthorizedBehavior = "returnNull" | "throw";

// Check if URL requires Python backend authentication
function requiresPythonAuth(url: string): boolean {
  const normalizedUrl = url.startsWith("/") ? url : `/${url}`;
  return (
    normalizedUrl.startsWith("/api/v1/epidemiology") ||
    normalizedUrl.startsWith("/api/v1/occupational") ||
    normalizedUrl.startsWith("/api/v1/genetic") ||
    normalizedUrl.startsWith("/api/v1/pharmaco") ||
    normalizedUrl.startsWith("/api/v1/infectious") ||
    normalizedUrl.startsWith("/api/v1/vaccine")
  );
}

export const getQueryFn: <T>(options: {
  on401: UnauthorizedBehavior;
}) => QueryFunction<T> =
  ({ on401: unauthorizedBehavior }) =>
  async ({ queryKey }) => {
    // Require first queryKey entry to be a string URL path
    if (queryKey.length === 0 || typeof queryKey[0] !== 'string') {
      throw new Error('Query key must start with a string URL path');
    }
    
    // Build URL from query key
    // Support both simple string keys and array keys with optional trailing params
    let url: string;
    let queryParams: Record<string, string | boolean> = {};
    
    if (queryKey.length === 1) {
      url = queryKey[0] as string;
    } else {
      // First element is URL, second element may be query params object
      url = queryKey[0] as string;
      if (queryKey.length > 1 && typeof queryKey[1] === 'object' && queryKey[1] !== null) {
        queryParams = queryKey[1] as Record<string, string | boolean>;
      }
    }
    
    // Build query string from params object
    const params = new URLSearchParams();
    for (const [key, value] of Object.entries(queryParams)) {
      if (value !== undefined && value !== null && value !== '') {
        params.append(key, String(value));
      }
    }
    const queryString = params.toString();
    const urlWithParams = queryString ? `${url}?${queryString}` : url;
    
    const finalUrl = getPythonBackendUrl(urlWithParams);
    
    // Add Authorization header for Python backend epidemiology endpoints
    const headers: Record<string, string> = {};
    if (requiresPythonAuth(url)) {
      // SECURITY: Token must come from environment variable in production
      // Development mode uses VITE_EPIDEMIOLOGY_AUTH_TOKEN env var
      // Production deployments MUST set this environment variable
      const authToken = import.meta.env.VITE_EPIDEMIOLOGY_AUTH_TOKEN;
      if (!authToken && import.meta.env.PROD) {
        console.error('SECURITY: VITE_EPIDEMIOLOGY_AUTH_TOKEN not set in production');
        throw new Error('Authentication configuration missing');
      }
      headers['Authorization'] = `Bearer ${authToken || 'dev-token'}`;
    }
    
    const res = await fetch(finalUrl, {
      credentials: "include",
      headers,
    });

    if (unauthorizedBehavior === "returnNull" && res.status === 401) {
      return null;
    }

    await throwIfResNotOk(res);
    return await res.json();
  };

export const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      queryFn: getQueryFn({ on401: "throw" }),
      refetchInterval: false,
      refetchOnWindowFocus: false,
      staleTime: Infinity,
      retry: false,
    },
    mutations: {
      retry: false,
    },
  },
});
