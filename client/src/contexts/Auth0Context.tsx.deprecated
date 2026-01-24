/**
 * Auth0 Authentication Context for Followup AI
 * Provides HIPAA-compliant authentication wrapper with Auth0
 */

import { createContext, useContext, useEffect, useState, useCallback, ReactNode } from 'react';
import { Auth0Provider, useAuth0, Auth0ContextInterface, User as Auth0User } from '@auth0/auth0-react';
import { auth0Config, isAuth0Configured } from '@/lib/auth0-config';
import type { User } from '@shared/schema';

interface AuthState {
  user: User | null;
  auth0User: Auth0User | undefined;
  isAuthenticated: boolean;
  isLoading: boolean;
  accessToken: string | null;
  error: Error | null;
}

interface Auth0WrapperContextType extends AuthState {
  login: () => Promise<void>;
  logout: () => void;
  getAccessToken: () => Promise<string | null>;
  refreshUser: () => Promise<void>;
}

const Auth0WrapperContext = createContext<Auth0WrapperContextType | undefined>(undefined);

/**
 * Inner Auth0 wrapper that has access to useAuth0 hook
 */
function Auth0InnerProvider({ children }: { children: ReactNode }) {
  const {
    isAuthenticated,
    isLoading: auth0Loading,
    user: auth0User,
    loginWithRedirect,
    logout: auth0Logout,
    getAccessTokenSilently,
    error: auth0Error,
  } = useAuth0();

  const [state, setState] = useState<AuthState>({
    user: null,
    auth0User: undefined,
    isAuthenticated: false,
    isLoading: true,
    accessToken: null,
    error: null,
  });

  // Sync user from backend after Auth0 authentication
  const syncUserWithBackend = useCallback(async (token: string): Promise<User | null> => {
    try {
      const response = await fetch('/api/auth/me', {
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
      });

      if (response.ok) {
        const userData = await response.json();
        return userData;
      } else if (response.status === 404) {
        // User doesn't exist in our system yet, create them
        const createResponse = await fetch('/api/auth/register', {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            email: auth0User?.email,
            name: auth0User?.name,
            picture: auth0User?.picture,
          }),
        });

        if (createResponse.ok) {
          return await createResponse.json();
        }
      }
      return null;
    } catch (error) {
      console.error('[Auth0] Failed to sync user with backend:', error);
      return null;
    }
  }, [auth0User]);

  // Get access token
  const getAccessToken = useCallback(async (): Promise<string | null> => {
    if (!isAuthenticated) return null;
    
    try {
      const token = await getAccessTokenSilently({
        authorizationParams: {
          audience: auth0Config.audience,
        },
      });
      return token;
    } catch (error) {
      console.error('[Auth0] Failed to get access token:', error);
      return null;
    }
  }, [isAuthenticated, getAccessTokenSilently]);

  // Login handler
  const login = useCallback(async () => {
    try {
      await loginWithRedirect({
        authorizationParams: {
          redirect_uri: window.location.origin,
          audience: auth0Config.audience,
        },
      });
    } catch (error) {
      console.error('[Auth0] Login failed:', error);
      setState(prev => ({ ...prev, error: error as Error }));
    }
  }, [loginWithRedirect]);

  // Logout handler
  const logout = useCallback(() => {
    // Clear local state first (memory-only, HIPAA compliant)
    setState({
      user: null,
      auth0User: undefined,
      isAuthenticated: false,
      isLoading: false,
      accessToken: null,
      error: null,
    });

    // Auth0 logout (invalidates refresh tokens on Auth0 side)
    auth0Logout({
      logoutParams: {
        returnTo: window.location.origin,
      },
    });
  }, [auth0Logout]);

  // Refresh user data
  const refreshUser = useCallback(async () => {
    if (!isAuthenticated) return;

    try {
      const token = await getAccessToken();
      if (token) {
        const user = await syncUserWithBackend(token);
        setState(prev => ({
          ...prev,
          user,
          accessToken: token,
        }));
      }
    } catch (error) {
      console.error('[Auth0] Failed to refresh user:', error);
    }
  }, [isAuthenticated, getAccessToken, syncUserWithBackend]);

  // Initialize auth state when Auth0 loads
  // HIPAA SECURITY: Tokens are stored ONLY in memory state, never in localStorage
  useEffect(() => {
    const initializeAuth = async () => {
      if (auth0Loading) return;

      if (isAuthenticated && auth0User) {
        try {
          const token = await getAccessToken();
          if (token) {
            const user = await syncUserWithBackend(token);
            setState({
              user,
              auth0User,
              isAuthenticated: true,
              isLoading: false,
              accessToken: token,
              error: null,
            });
            // HIPAA SECURITY: DO NOT store tokens or user data in localStorage
            // All auth state is kept in memory only. Use refresh tokens for persistence.
          }
        } catch (error) {
          console.error('[Auth0] Initialization error:', error);
          setState(prev => ({
            ...prev,
            isLoading: false,
            error: error as Error,
          }));
        }
      } else {
        setState({
          user: null,
          auth0User: undefined,
          isAuthenticated: false,
          isLoading: false,
          accessToken: null,
          error: auth0Error || null,
        });
      }
    };

    initializeAuth();
  }, [auth0Loading, isAuthenticated, auth0User, auth0Error, getAccessToken, syncUserWithBackend]);

  const contextValue: Auth0WrapperContextType = {
    ...state,
    login,
    logout,
    getAccessToken,
    refreshUser,
  };

  return (
    <Auth0WrapperContext.Provider value={contextValue}>
      {children}
    </Auth0WrapperContext.Provider>
  );
}

/**
 * Main Auth0 Provider component
 */
export function Auth0ProviderWithConfig({ children }: { children: ReactNode }) {
  // Check if Auth0 is configured
  if (!isAuth0Configured()) {
    console.warn('[Auth0] Not configured. Using fallback authentication.');
    return <FallbackAuthProvider>{children}</FallbackAuthProvider>;
  }

  return (
    <Auth0Provider
      domain={auth0Config.domain}
      clientId={auth0Config.clientId}
      authorizationParams={{
        redirect_uri: window.location.origin,
        audience: auth0Config.audience,
        scope: 'openid profile email offline_access',
      }}
      cacheLocation="memory"
      useRefreshTokens={true}
    >
      <Auth0InnerProvider>{children}</Auth0InnerProvider>
    </Auth0Provider>
  );
}

/**
 * Fallback provider for when Auth0 is not configured (dev mode)
 * HIPAA SECURITY: Uses session storage instead of localStorage for dev mode
 * Session storage is cleared when browser tab closes (more secure)
 */
function FallbackAuthProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<AuthState>({
    user: null,
    auth0User: undefined,
    isAuthenticated: false,
    isLoading: true,
    accessToken: null,
    error: null,
  });

  useEffect(() => {
    // Dev mode uses sessionStorage (cleared on tab close) instead of localStorage
    // This provides better security for development testing
    const storedUser = sessionStorage.getItem('devAuthUser');
    const storedToken = sessionStorage.getItem('devAuthToken');

    if (storedUser && storedToken) {
      try {
        const user = JSON.parse(storedUser);
        setState({
          user,
          auth0User: undefined,
          isAuthenticated: true,
          isLoading: false,
          accessToken: storedToken,
          error: null,
        });
      } catch {
        setState(prev => ({ ...prev, isLoading: false }));
      }
    } else {
      setState(prev => ({ ...prev, isLoading: false }));
    }
  }, []);

  const contextValue: Auth0WrapperContextType = {
    ...state,
    login: async () => {
      // In dev mode, redirect to login page
      window.location.href = '/login';
    },
    logout: () => {
      // Clear session storage (dev mode)
      sessionStorage.removeItem('devAuthUser');
      sessionStorage.removeItem('devAuthToken');
      setState({
        user: null,
        auth0User: undefined,
        isAuthenticated: false,
        isLoading: false,
        accessToken: null,
        error: null,
      });
      window.location.href = '/';
    },
    getAccessToken: async () => state.accessToken,
    refreshUser: async () => {},
  };

  return (
    <Auth0WrapperContext.Provider value={contextValue}>
      {children}
    </Auth0WrapperContext.Provider>
  );
}

/**
 * Hook to use Auth0 authentication
 */
export function useAuth0Wrapper(): Auth0WrapperContextType {
  const context = useContext(Auth0WrapperContext);
  if (context === undefined) {
    throw new Error('useAuth0Wrapper must be used within an Auth0ProviderWithConfig');
  }
  return context;
}

/**
 * Hook for protected routes - returns auth state and redirect function
 */
export function useRequireAuth() {
  const auth = useAuth0Wrapper();
  
  useEffect(() => {
    if (!auth.isLoading && !auth.isAuthenticated) {
      // Redirect to login if not authenticated
      auth.login();
    }
  }, [auth.isLoading, auth.isAuthenticated, auth.login]);

  return auth;
}
