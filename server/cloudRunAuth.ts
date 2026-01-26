/**
 * Cloud Run Authentication Helper
 * Generates ID tokens for authenticated Cloud Run requests using service account
 */
import { GoogleAuth } from 'google-auth-library';

let auth: GoogleAuth | null = null;
let cachedToken: { token: string; expiry: number } | null = null;

const PYTHON_BACKEND_URL = process.env.PYTHON_BACKEND_URL || 'http://localhost:8000';
const isCloudRun = PYTHON_BACKEND_URL.includes('.run.app');

function getGoogleAuth(): GoogleAuth {
  if (auth) return auth;
  
  const serviceAccountKey = process.env.GCS_SERVICE_ACCOUNT_KEY;
  if (!serviceAccountKey) {
    throw new Error('GCS_SERVICE_ACCOUNT_KEY not configured for Cloud Run authentication');
  }
  
  try {
    const credentials = JSON.parse(serviceAccountKey);
    auth = new GoogleAuth({
      credentials,
      scopes: ['https://www.googleapis.com/auth/cloud-platform'],
    });
    return auth;
  } catch (error) {
    throw new Error(`Failed to parse GCS_SERVICE_ACCOUNT_KEY: ${error}`);
  }
}

/**
 * Get an ID token for authenticating with Cloud Run (with caching)
 */
async function getCloudRunIdToken(): Promise<string> {
  // Return cached token if still valid (with 5 min buffer)
  if (cachedToken && cachedToken.expiry > Date.now() + 300000) {
    return cachedToken.token;
  }
  
  const googleAuth = getGoogleAuth();
  const client = await googleAuth.getIdTokenClient(PYTHON_BACKEND_URL);
  const headers = await client.getRequestHeaders();
  const authHeader = headers['Authorization'];
  if (!authHeader) {
    throw new Error('Failed to get Authorization header from ID token client');
  }
  
  const token = authHeader.replace('Bearer ', '');
  // Cache for 55 minutes (tokens typically valid for 1 hour)
  cachedToken = { token, expiry: Date.now() + 3300000 };
  return token;
}

/**
 * Get the Python backend URL
 */
export function getPythonBackendUrl(): string {
  return PYTHON_BACKEND_URL;
}

/**
 * Make an authenticated fetch request to Python backend
 * Automatically handles Cloud Run auth when needed
 */
export async function fetchFromCloudRun(
  url: string,
  options: RequestInit = {}
): Promise<Response> {
  // If local development, just use regular fetch
  if (!isCloudRun) {
    return fetch(url, options);
  }
  
  // For Cloud Run, add authentication header
  try {
    const token = await getCloudRunIdToken();
    const headers = new Headers(options.headers);
    headers.set('Authorization', `Bearer ${token}`);
    
    return fetch(url, {
      ...options,
      headers,
    });
  } catch (error) {
    console.error('[CloudRunAuth] Failed to get ID token:', error);
    throw error;
  }
}
