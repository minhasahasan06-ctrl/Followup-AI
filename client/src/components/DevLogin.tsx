import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { UserCircle, Stethoscope, Shield, Loader2 } from "lucide-react";
import { apiRequest } from "@/lib/queryClient";
import { useToast } from "@/hooks/use-toast";

export function DevLogin() {
  const [isLoggingIn, setIsLoggingIn] = useState(false);
  const { toast } = useToast();

  const loginAs = async (role: 'patient' | 'doctor' | 'admin') => {
    setIsLoggingIn(true);
    try {
      const endpoints = {
        patient: '/api/dev/login-as-patient',
        doctor: '/api/dev/login-as-doctor',
        admin: '/api/dev/login-as-admin'
      };
      
      // Call dev login endpoint - this sets the session cookie on the server
      await apiRequest(endpoints[role], { method: 'POST' });
      
      // Force full page reload to pick up new session
      window.location.href = '/dashboard';
    } catch (error: any) {
      console.error(`Error logging in as ${role}:`, error);
      toast({
        title: "Login Failed",
        description: error.message || `Failed to login as ${role}`,
        variant: "destructive",
      });
    } finally {
      setIsLoggingIn(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-background via-muted/20 to-background p-4">
      <Card className="w-full max-w-md">
        <CardHeader className="text-center">
          <CardTitle className="text-2xl">Followup AI</CardTitle>
          <CardDescription>Development Mode - Quick Login</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-3">
            <Button
              onClick={() => loginAs('patient')}
              disabled={isLoggingIn}
              className="w-full gap-2"
              size="lg"
              data-testid="button-login-patient"
            >
              {isLoggingIn ? (
                <Loader2 className="h-5 w-5 animate-spin" />
              ) : (
                <UserCircle className="h-5 w-5" />
              )}
              Login as Patient
            </Button>
            
            <Button
              onClick={() => loginAs('doctor')}
              disabled={isLoggingIn}
              variant="outline"
              className="w-full gap-2"
              size="lg"
              data-testid="button-login-doctor"
            >
              {isLoggingIn ? (
                <Loader2 className="h-5 w-5 animate-spin" />
              ) : (
                <Stethoscope className="h-5 w-5" />
              )}
              Login as Doctor
            </Button>
            
            <Button
              onClick={() => loginAs('admin')}
              disabled={isLoggingIn}
              variant="outline"
              className="w-full gap-2"
              size="lg"
              data-testid="button-login-admin"
            >
              {isLoggingIn ? (
                <Loader2 className="h-5 w-5 animate-spin" />
              ) : (
                <Shield className="h-5 w-5" />
              )}
              Login as Admin
            </Button>
          </div>
          
          <div className="text-center text-xs text-muted-foreground">
            <p>These test accounts are pre-configured for development.</p>
            <p className="mt-1">This screen only appears in development mode.</p>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
