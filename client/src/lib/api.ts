import axios from 'axios';

/**
 * Unified API client with production-ready routing
 * 
 * Handles:
 * - Same-origin requests in development (relative URLs)
 * - Cross-origin requests to Express backend in production (VITE_EXPRESS_BACKEND_URL)
 * - Consistent credential handling for session cookies
 */

// Express backend URL - For Vercel/external deployments where frontend is separate from backend
// In development (same-origin), leave empty to use relative URLs
const EXPRESS_BACKEND_URL = import.meta.env.VITE_EXPRESS_BACKEND_URL || '';

// Production safety check: warn if backend URL is not configured for external deployments
if (typeof window !== 'undefined' && import.meta.env.PROD) {
  const isExternalDeployment = !window.location.hostname.includes('localhost') && 
                                !window.location.hostname.includes('127.0.0.1') &&
                                !window.location.hostname.includes('.replit');
  
  if (isExternalDeployment && !EXPRESS_BACKEND_URL) {
    console.warn(
      '[API] VITE_EXPRESS_BACKEND_URL not set for production deployment. ' +
      'API calls will use relative URLs and may fail. ' +
      'Set VITE_EXPRESS_BACKEND_URL in your environment variables.'
    );
  }
}

/**
 * Get the full API URL for a given path
 * Shared utility for consistent URL construction across the app
 */
export function getExpressApiUrl(path: string): string {
  // If already a full URL, return as-is
  if (path.startsWith('http://') || path.startsWith('https://')) {
    return path;
  }
  
  // Ensure path starts with /api
  const normalizedPath = path.startsWith('/') ? path : `/${path}`;
  const apiPath = normalizedPath.startsWith('/api') ? normalizedPath : `/api${normalizedPath}`;
  
  // Use absolute URL if EXPRESS_BACKEND_URL is configured (production)
  // Otherwise use relative URL (development/same-origin)
  if (EXPRESS_BACKEND_URL) {
    return `${EXPRESS_BACKEND_URL}${apiPath}`;
  }
  
  return apiPath;
}

// Create axios instance with proper configuration
const api = axios.create({
  // Don't set baseURL - we'll handle URL construction in the interceptor
  headers: {
    'Content-Type': 'application/json',
  },
  withCredentials: true, // Always include credentials for session cookies
});

// Request interceptor to handle URL routing and auth tokens
api.interceptors.request.use(
  (config) => {
    // Construct full URL with proper backend routing
    if (config.url && !config.url.startsWith('http')) {
      config.url = getExpressApiUrl(config.url);
    }
    
    // Add JWT token if stored (for backward compatibility with token-based auth)
    const authTokens = localStorage.getItem('authTokens');
    if (authTokens) {
      try {
        const { accessToken } = JSON.parse(authTokens);
        if (accessToken) {
          config.headers.Authorization = `Bearer ${accessToken}`;
        }
      } catch {
        // Invalid JSON in localStorage, ignore
      }
    }
    
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor to handle auth errors
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      // Trigger logout event to clear AuthContext state
      const event = new CustomEvent('auth:logout');
      window.dispatchEvent(event);
      
      // Clear auth storage
      localStorage.removeItem('authTokens');
      localStorage.removeItem('authUser');
      
      // Don't redirect on auth routes to avoid loops
      const currentPath = window.location.pathname;
      const authRoutes = ['/login', '/signup', '/verify', '/auth', '/forgot-password', '/reset-password'];
      const isAuthRoute = authRoutes.some(route => currentPath.startsWith(route));
      
      if (!isAuthRoute && currentPath !== '/') {
        window.location.href = '/login';
      }
    }
    return Promise.reject(error);
  }
);

export default api;
