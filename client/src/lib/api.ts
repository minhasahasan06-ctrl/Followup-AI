import axios from 'axios';

// Create axios instance with default config
const api = axios.create({
  baseURL: '/api',
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add request interceptor to include JWT token
api.interceptors.request.use(
  (config) => {
    const authTokens = localStorage.getItem('authTokens');
    if (authTokens) {
      const { accessToken } = JSON.parse(authTokens);
      if (accessToken) {
        config.headers.Authorization = `Bearer ${accessToken}`;
      }
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Add response interceptor to handle auth errors
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
      
      // Redirect to login
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

export default api;
