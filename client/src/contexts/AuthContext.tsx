import { createContext, useContext, useState, useEffect, ReactNode, useCallback } from 'react';
import type { User } from '@shared/schema';
import { getExpressApiUrl } from '@/lib/api';

interface AuthTokens {
  idToken: string;
  accessToken: string;
  refreshToken: string;
}

interface AuthContextType {
  user: User | null;
  tokens: AuthTokens | null;
  isLoading: boolean;
  isAuthenticated: boolean;
  setTokens: (tokens: AuthTokens | null) => void;
  logout: () => Promise<void>;
  refreshSession: () => Promise<User | null>;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [tokens, setTokensState] = useState<AuthTokens | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  const validateSession = useCallback(async (): Promise<User | null> => {
    try {
      const response = await fetch(getExpressApiUrl('/api/auth/session/me'), {
        credentials: 'include',
      });
      
      if (response.ok) {
        const data = await response.json();
        const serverUser = data.user;
        setUser(serverUser);
        return serverUser;
      } else {
        setUser(null);
        setTokensState(null);
        localStorage.removeItem('authTokens');
        return null;
      }
    } catch (error) {
      console.error('Session validation error:', error);
      setUser(null);
      setTokensState(null);
      localStorage.removeItem('authTokens');
      return null;
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    validateSession();

    const handleLogout = () => {
      setUser(null);
      setTokensState(null);
      localStorage.removeItem('authTokens');
    };

    window.addEventListener('auth:logout', handleLogout);
    return () => window.removeEventListener('auth:logout', handleLogout);
  }, [validateSession]);

  // Token setter for backward compatibility with token-based auth flows
  const setTokens = useCallback((newTokens: AuthTokens | null) => {
    setTokensState(newTokens);
    if (newTokens) {
      localStorage.setItem('authTokens', JSON.stringify(newTokens));
    } else {
      localStorage.removeItem('authTokens');
    }
  }, []);

  const logout = useCallback(async () => {
    try {
      await fetch(getExpressApiUrl('/api/auth/logout'), { 
        method: 'POST', 
        credentials: 'include' 
      });
    } catch {
      // Ignore logout errors
    }
    setUser(null);
    setTokensState(null);
    localStorage.removeItem('authTokens');
  }, []);

  const refreshSession = useCallback(async (): Promise<User | null> => {
    return await validateSession();
  }, [validateSession]);

  return (
    <AuthContext.Provider value={{ 
      user, 
      tokens,
      isLoading, 
      isAuthenticated: !!user,
      setTokens,
      logout, 
      refreshSession 
    }}>
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
