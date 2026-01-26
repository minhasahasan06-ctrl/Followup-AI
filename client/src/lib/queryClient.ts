import { QueryClient, QueryFunction } from "@tanstack/react-query";

// Cloud Run Proxy Mode - When enabled, routes Python backend calls through Express proxy
// The Express proxy handles: Stytch auth verification -> Google ID token generation -> Cloud Run forwarding
const USE_CLOUD_RUN_PROXY = import.meta.env.VITE_USE_CLOUD_RUN_PROXY === "true";

// Python backend URL (FastAPI on port 8000) - Used for direct development
const PYTHON_BACKEND_URL = import.meta.env.VITE_PYTHON_BACKEND_URL || "http://localhost:8000";

// Routes that should go to Python FastAPI backend (directly or via Cloud Run proxy)
const PYTHON_BACKEND_ROUTES = [
  "/api/v1/video-ai",
  "/api/v1/audio-ai",
  "/api/v1/trends",
  "/api/v1/alerts",
  "/api/v1/guided-audio-exam",
  "/api/v1/guided-exam",
  "/api/v1/gait-analysis",
  "/api/v1/tremor",
  "/api/v1/automation",
  "/api/v1/webhooks",
  "/api/v1/devices",
  "/api/v1/health-analytics",
  "/api/v1/epidemiology",
  "/api/v1/occupational",
  "/api/v1/genetic",
  "/api/v1/pharmaco",
  "/api/v1/infectious",
  "/api/v1/vaccine",
];

function isPythonBackendRoute(url: string): boolean {
  return PYTHON_BACKEND_ROUTES.some((route) => url.startsWith(route));
}

// Helper to determine the final URL for a request
function getApiUrl(url: string): string {
  if (url.startsWith("http://") || url.startsWith("https://")) {
    return url;
  }

  const normalizedUrl = url.startsWith("/") ? url : `/${url}`;
  
  // Handle /api/py/ prefix - strip 'py' for Python backend routing
  if (normalizedUrl.startsWith("/api/py/")) {
    const strippedUrl = normalizedUrl.replace("/api/py/", "/api/");
    if (USE_CLOUD_RUN_PROXY) {
      return `/api/cloud${strippedUrl.replace("/api", "")}`;
    }
    return `${PYTHON_BACKEND_URL}${strippedUrl}`;
  }

  // Route Python backend calls through Cloud Run proxy when enabled
  if (isPythonBackendRoute(normalizedUrl)) {
    if (USE_CLOUD_RUN_PROXY) {
      return `/api/cloud${normalizedUrl.replace("/api", "")}`;
    }
    return `${PYTHON_BACKEND_URL}${normalizedUrl}`;
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
  
  // Route to appropriate backend (Express, Python, or Cloud Run proxy)
  const finalUrl = getApiUrl(url);
  
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
    
    const finalUrl = getApiUrl(urlWithParams);
    
    const res = await fetch(finalUrl, {
      credentials: "include",
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
