import { useEffect, useState } from "react";
import { useLocation, useSearch } from "wouter";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { CheckCircle, XCircle, Loader2, Stethoscope } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import { useAuth } from "@/contexts/AuthContext";
import api from "@/lib/api";
import { Link } from "wouter";

type CallbackState = 'loading' | 'success' | 'error';

export default function MagicLinkCallback() {
  const [, setLocation] = useLocation();
  const search = useSearch();
  const { toast } = useToast();
  const { refreshSession } = useAuth();
  const [state, setState] = useState<CallbackState>('loading');
  const [errorMessage, setErrorMessage] = useState('');
  const [userRole, setUserRole] = useState<string>('patient');

  useEffect(() => {
    const authenticateMagicLink = async () => {
      // Extract token from URL query params
      const params = new URLSearchParams(search);
      const token = params.get('token') || params.get('stytch_token_type') ? params.get('token') : null;
      
      // Stytch might use different param names
      const stytchToken = params.get('token') || 
                          window.location.hash.replace('#', '').split('&')
                            .find(p => p.startsWith('token='))?.split('=')[1];
      
      const finalToken = token || stytchToken;
      
      if (!finalToken) {
        setState('error');
        setErrorMessage('No authentication token found in the URL. Please request a new magic link.');
        return;
      }

      try {
        const response = await api.post("/auth/magic-link/authenticate", {
          token: finalToken,
        });

        if (response.data.success) {
          const role = response.data.user?.role || 'patient';
          setUserRole(role);
          setState('success');
          
          // Refresh session to update auth context
          await refreshSession();
          
          toast({
            title: "Welcome!",
            description: "You've been successfully authenticated.",
          });
          
          // Redirect based on role after a brief delay
          setTimeout(() => {
            if (role === 'doctor') {
              setLocation('/dashboard');
            } else if (role === 'admin') {
              setLocation('/ml-training');
            } else {
              setLocation('/dashboard');
            }
          }, 2000);
        } else {
          throw new Error('Authentication failed');
        }
      } catch (error: any) {
        console.error('Magic link authentication error:', error);
        setState('error');
        setErrorMessage(
          error.response?.data?.error || 
          error.response?.data?.message || 
          'Invalid or expired magic link. Please request a new one.'
        );
      }
    };

    authenticateMagicLink();
  }, [search, refreshSession, setLocation, toast]);

  return (
    <div className="min-h-screen bg-gradient-to-b from-background to-muted flex items-center justify-center p-6">
      <Card className="w-full max-w-md">
        <CardHeader className="text-center">
          <div className="flex justify-center mb-4">
            {state === 'loading' && (
              <div className="flex h-16 w-16 items-center justify-center rounded-full bg-primary/10 text-primary">
                <Loader2 className="h-8 w-8 animate-spin" />
              </div>
            )}
            {state === 'success' && (
              <div className="flex h-16 w-16 items-center justify-center rounded-full bg-green-100 text-green-600 dark:bg-green-900/30 dark:text-green-400">
                <CheckCircle className="h-8 w-8" />
              </div>
            )}
            {state === 'error' && (
              <div className="flex h-16 w-16 items-center justify-center rounded-full bg-destructive/10 text-destructive">
                <XCircle className="h-8 w-8" />
              </div>
            )}
          </div>
          
          {state === 'loading' && (
            <>
              <CardTitle className="text-2xl">Authenticating...</CardTitle>
              <CardDescription>
                Please wait while we verify your magic link
              </CardDescription>
            </>
          )}
          
          {state === 'success' && (
            <>
              <CardTitle className="text-2xl">Welcome to Followup AI!</CardTitle>
              <CardDescription>
                Your account has been verified successfully
              </CardDescription>
            </>
          )}
          
          {state === 'error' && (
            <>
              <CardTitle className="text-2xl">Authentication Failed</CardTitle>
              <CardDescription className="text-destructive">
                {errorMessage}
              </CardDescription>
            </>
          )}
        </CardHeader>
        
        <CardContent className="space-y-4">
          {state === 'success' && (
            <div className="bg-muted/50 rounded-lg p-4 space-y-2">
              <div className="flex items-start gap-2">
                <CheckCircle className="h-5 w-5 text-green-500 mt-0.5 shrink-0" />
                <p className="text-sm">Your session has been created</p>
              </div>
              <div className="flex items-start gap-2">
                <Stethoscope className="h-5 w-5 text-primary mt-0.5 shrink-0" />
                <p className="text-sm">
                  Redirecting you to your {userRole} dashboard...
                </p>
              </div>
            </div>
          )}
          
          {state === 'error' && (
            <div className="space-y-4">
              <div className="bg-destructive/10 rounded-lg p-4">
                <p className="text-sm">
                  Magic links expire after 30 minutes for security. If your link has expired, you can request a new one.
                </p>
              </div>
              
              <div className="flex flex-col gap-2">
                <Button
                  onClick={() => setLocation('/login')}
                  data-testid="button-back-to-login"
                >
                  Request New Magic Link
                </Button>
                <Button
                  variant="outline"
                  onClick={() => setLocation('/')}
                  data-testid="button-back-home"
                >
                  Back to Home
                </Button>
              </div>
            </div>
          )}
          
          {state === 'loading' && (
            <p className="text-center text-sm text-muted-foreground">
              This should only take a moment...
            </p>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
