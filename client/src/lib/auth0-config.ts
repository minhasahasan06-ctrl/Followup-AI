/**
 * Auth0 Configuration for Followup AI
 * HIPAA-compliant authentication setup
 */

export const auth0Config = {
  domain: import.meta.env.VITE_AUTH0_DOMAIN || '',
  clientId: import.meta.env.VITE_AUTH0_CLIENT_ID || '',
  audience: import.meta.env.VITE_AUTH0_API_AUDIENCE || '',
  redirectUri: typeof window !== 'undefined' ? window.location.origin : '',
  
  // HIPAA-compliant settings
  cacheLocation: 'memory' as const, // More secure than localStorage for PHI
  useRefreshTokens: true,
  
  // Session settings
  authorizationParams: {
    audience: import.meta.env.VITE_AUTH0_API_AUDIENCE || '',
    scope: 'openid profile email offline_access',
  },
};

/**
 * Check if Auth0 is properly configured
 */
export function isAuth0Configured(): boolean {
  return !!(auth0Config.domain && auth0Config.clientId);
}

/**
 * Get Auth0 callback URL for different environments
 */
export function getAuth0CallbackUrl(): string {
  if (typeof window === 'undefined') return '';
  return `${window.location.origin}/callback`;
}
