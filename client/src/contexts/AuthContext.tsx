import { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import type { User } from '@shared/schema';

interface AuthTokens {
  idToken: string;
  accessToken: string;
  refreshToken: string;
}

interface AuthContextType {
  user: User | null;
  tokens: AuthTokens | null;
  isLoading: boolean;
  setTokens: (tokens: AuthTokens | null) => void;
  logout: () => void;
  refreshSession: () => Promise<User | null>;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [tokens, setTokensState] = useState<AuthTokens | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  // Validate session with server - this is the ONLY way to set user/role
  const validateSession = async (): Promise<User | null> => {
    try {
      const response = await fetch('/api/auth/user', {
        credentials: 'include',
      });
      
      if (response.ok) {
        const serverUser = await response.json();
        // User and role can ONLY come from server - never from client
        setUser(serverUser);
        return serverUser;
      } else {
        // Server session invalid - clear everything
        setUser(null);
        setTokensState(null);
        localStorage.removeItem('authTokens');
        return null;
      }
    } catch (error) {
      console.error('Session validation error:', error);
      // On error, clear auth state for safety
      setUser(null);
      setTokensState(null);
      localStorage.removeItem('authTokens');
      return null;
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    // Always validate with server on mount - user/role ONLY comes from server
    validateSession();

    // Listen for logout events from api interceptor
    const handleLogout = () => {
      setTokensState(null);
      setUser(null);
      localStorage.removeItem('authTokens');
    };

    window.addEventListener('auth:logout', handleLogout);
    return () => window.removeEventListener('auth:logout', handleLogout);
  }, []);

  // Token setter for flows that use JWT tokens (e.g., Cognito auth)
  // NOTE: This only sets tokens, NOT user. User must come from refreshSession()
  const setTokens = (newTokens: AuthTokens | null) => {
    setTokensState(newTokens);
    if (newTokens) {
      localStorage.setItem('authTokens', JSON.stringify(newTokens));
    } else {
      localStorage.removeItem('authTokens');
    }
  };

  const logout = () => {
    setTokensState(null);
    setUser(null);
    localStorage.removeItem('authTokens');
    // Also call server logout to invalidate session
    fetch('/api/auth/logout', { method: 'POST', credentials: 'include' }).catch(() => {});
  };

  const refreshSession = async (): Promise<User | null> => {
    return await validateSession();
  };

  return (
    <AuthContext.Provider value={{ user, tokens, isLoading, setTokens, logout, refreshSession }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}
