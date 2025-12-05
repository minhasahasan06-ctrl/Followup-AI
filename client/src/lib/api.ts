/**
 * Secure Axios-compatible API Client
 * 
 * This module provides an axios-like interface that uses the secure API client
 * under the hood. It maintains backward compatibility with existing axios code
 * while adding comprehensive security features.
 * 
 * @deprecated Consider migrating to secureApiRequest from './security/secureApiClient'
 * for better security and performance.
 */

import { secureApiRequest, secureGet, securePost, securePut, securePatch, secureDelete } from './security/secureApiClient';
import { sanitizeError } from './security/errorSanitizer';

/**
 * Axios-compatible request config
 */
interface AxiosRequestConfig {
  url?: string;
  method?: string;
  baseURL?: string;
  headers?: Record<string, string>;
  data?: any;
  params?: Record<string, any>;
  timeout?: number;
  withCredentials?: boolean;
}

/**
 * Axios-compatible response
 */
interface AxiosResponse<T = any> {
  data: T;
  status: number;
  statusText: string;
  headers: Record<string, string>;
  config: AxiosRequestConfig;
}

/**
 * Axios-compatible error
 */
class AxiosError extends Error {
  response?: AxiosResponse;
  config?: AxiosRequestConfig;
  status?: number;

  constructor(message: string, config?: AxiosRequestConfig, response?: AxiosResponse) {
    super(message);
    this.name = 'AxiosError';
    this.config = config;
    this.response = response;
    this.status = response?.status;
  }
}

/**
 * Converts axios config to secure API request options
 */
function convertAxiosConfig(config: AxiosRequestConfig): {
  url: string;
  options: Parameters<typeof secureApiRequest>[1];
} {
  const { url = '', method = 'GET', baseURL = '', headers = {}, data, params, timeout, withCredentials } = config;

  // Build full URL
  let fullUrl = url;
  if (baseURL && !url.startsWith('http')) {
    fullUrl = `${baseURL.replace(/\/$/, '')}/${url.replace(/^\//, '')}`;
  }

  // Add query parameters
  if (params) {
    const searchParams = new URLSearchParams();
    Object.entries(params).forEach(([key, value]) => {
      if (value !== undefined && value !== null) {
        searchParams.append(key, String(value));
      }
    });
    const queryString = searchParams.toString();
    if (queryString) {
      fullUrl += (fullUrl.includes('?') ? '&' : '?') + queryString;
    }
  }

  const options: Parameters<typeof secureApiRequest>[1] = {
    method: method.toUpperCase() as any,
    headers,
    timeout,
    credentials: withCredentials ? 'include' : 'same-origin',
  };

  if (data !== undefined) {
    options.json = data;
  }

  return { url: fullUrl, options };
}

/**
 * Axios-compatible request function
 */
async function axiosRequest<T = any>(config: AxiosRequestConfig): Promise<AxiosResponse<T>> {
  // Apply request interceptors (support both sync and async)
  let processedConfig = config;
  const requestInterceptor = (api as any).requestInterceptor;
  if (requestInterceptor) {
    const interceptedConfig = requestInterceptor(processedConfig);
    // Handle promise return value
    if (interceptedConfig instanceof Promise) {
      processedConfig = await interceptedConfig;
    } else {
      processedConfig = interceptedConfig || processedConfig;
    }
  }

  const { url: finalUrl, options } = convertAxiosConfig(processedConfig);

  try {
    const response = await secureApiRequest<T>(finalUrl, options);

    // Build axios-compatible response
    const axiosResponse: AxiosResponse<T> = {
      data: response.data as T,
      status: response.status,
      statusText: response.statusText,
      headers: Object.fromEntries(response.headers.entries()),
      config: processedConfig,
    };

    // Apply response interceptors (support both sync and async)
    const responseInterceptor = (api as any).responseInterceptor;
    if (responseInterceptor) {
      const interceptedResponse = responseInterceptor(axiosResponse);
      // Handle promise return value
      if (interceptedResponse instanceof Promise) {
        return await interceptedResponse;
      }
      return interceptedResponse || axiosResponse;
    }

    return axiosResponse;
  } catch (error) {
    // Apply error interceptors
    const errorInterceptor = (api as any).errorInterceptor;
    if (errorInterceptor) {
      try {
        return await errorInterceptor(error);
      } catch (interceptorError) {
        // If interceptor throws, use original error
        error = interceptorError;
      }
    }

    const sanitized = sanitizeError(error);
    const axiosError = new AxiosError(
      sanitized.userMessage,
      processedConfig,
      error && typeof error === 'object' && 'statusCode' in error
        ? {
            data: (error as any).data,
            status: (error as any).statusCode,
            statusText: (error as any).message,
            headers: {},
            config: processedConfig,
          }
        : undefined
    );
    throw axiosError;
  }
}

/**
 * Axios-compatible API instance
 */
const api = {
  request: axiosRequest,
  get: <T = any>(url: string, config?: AxiosRequestConfig) =>
    axiosRequest<T>({ ...(config || {}), url, method: 'GET' }),
  post: <T = any>(url: string, data?: any, config?: AxiosRequestConfig) =>
    axiosRequest<T>({ ...(config || {}), url, method: 'POST', data }),
  put: <T = any>(url: string, data?: any, config?: AxiosRequestConfig) =>
    axiosRequest<T>({ ...(config || {}), url, method: 'PUT', data }),
  patch: <T = any>(url: string, data?: any, config?: AxiosRequestConfig) =>
    axiosRequest<T>({ ...(config || {}), url, method: 'PATCH', data }),
  delete: <T = any>(url: string, config?: AxiosRequestConfig) =>
    axiosRequest<T>({ ...(config || {}), url, method: 'DELETE' }),
  create: (defaultConfig?: AxiosRequestConfig) => {
    // Create a new instance with default config
    return {
      ...api,
      defaults: defaultConfig || {},
      request: (config?: AxiosRequestConfig) =>
        axiosRequest({ ...(defaultConfig || {}), ...(config || {}) }),
    };
  },
  interceptors: {
    request: {
      use: (onFulfilled?: (config: AxiosRequestConfig) => AxiosRequestConfig) => {
        // Store interceptor for later use
        if (onFulfilled) {
          (api as any).requestInterceptor = onFulfilled;
        }
      },
    },
    response: {
      use: (
        onFulfilled?: (response: AxiosResponse) => AxiosResponse,
        onRejected?: (error: any) => any
      ) => {
        // Store interceptors for later use
        if (onFulfilled) {
          (api as any).responseInterceptor = onFulfilled;
        }
        if (onRejected) {
          (api as any).errorInterceptor = onRejected;
        }
      },
    },
  },
  defaults: {
    baseURL: '/api',
    headers: {
      'Content-Type': 'application/json',
    },
  },
} as any;

// Add request interceptor to include JWT token (backward compatibility)
api.interceptors.request.use((config: AxiosRequestConfig) => {
  if (typeof window !== 'undefined') {
    const authTokens = localStorage.getItem('authTokens');
    if (authTokens) {
      try {
        const { accessToken } = JSON.parse(authTokens);
        if (accessToken) {
          config.headers = config.headers || {};
          config.headers.Authorization = `Bearer ${accessToken}`;
        }
      } catch (error) {
        // Silently fail
      }
    }
  }
  return config;
});

// Add response interceptor to handle auth errors (backward compatibility)
api.interceptors.response.use(
  (response: AxiosResponse) => response,
  (error: any) => {
    if (error?.response?.status === 401 || error?.status === 401) {
      // Trigger logout event
      if (typeof window !== 'undefined') {
        const event = new CustomEvent('auth:logout');
        window.dispatchEvent(event);
        localStorage.removeItem('authTokens');
        localStorage.removeItem('authUser');
        window.location.href = '/login';
      }
    }
    return Promise.reject(error);
  }
);

export default api;
