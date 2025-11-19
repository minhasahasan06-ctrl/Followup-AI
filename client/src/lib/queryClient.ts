import { QueryClient, QueryFunction } from "@tanstack/react-query";

// Python backend URL (FastAPI on port 8000)
const PYTHON_BACKEND_URL = import.meta.env.VITE_PYTHON_BACKEND_URL || "http://localhost:8000";

// Helper to determine if URL should go to Python backend
function getPythonBackendUrl(url: string): string {
  // If it's already an absolute URL (starts with http:// or https://), return unchanged
  if (url.startsWith('http://') || url.startsWith('https://')) {
    return url;
  }
  
  // Normalize URL (ensure leading slash for relative paths)
  const normalizedUrl = url.startsWith('/') ? url : `/${url}`;
  
  // Routes that should go to Python FastAPI backend
  if (normalizedUrl.startsWith("/api/v1/video-ai") || 
      normalizedUrl.startsWith("/api/v1/audio-ai") || 
      normalizedUrl.startsWith("/api/v1/trends") || 
      normalizedUrl.startsWith("/api/v1/alerts") ||
      normalizedUrl.startsWith("/api/v1/guided-audio-exam") ||
      normalizedUrl.startsWith("/api/v1/guided-exam")) {
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
    if (queryKey.length === 1) {
      url = queryKey[0] as string;
    } else {
      // Join only the string elements, ignore objects/arrays (they're for cache invalidation)
      const stringParts = queryKey.filter(k => typeof k === 'string') as string[];
      url = stringParts.join("/");
    }
    
    const finalUrl = getPythonBackendUrl(url);
    
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
