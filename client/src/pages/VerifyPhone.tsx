import { useState, useEffect } from "react";
import { useLocation } from "wouter";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Smartphone, ArrowRight } from "lucide-react";
import { Link } from "wouter";

/**
 * Legacy VerifyPhone page - redirects to new Stytch SMS OTP flow
 * 
 * With Stytch authentication, phone verification is handled through
 * the SMS OTP flow. This page now serves as a redirect for any old
 * bookmarks or links.
 */
export default function VerifyPhone() {
  const [, setLocation] = useLocation();

  useEffect(() => {
    // Check if we came from the SMS OTP flow
    const storedPhone = sessionStorage.getItem('smsVerificationPhone');
    if (storedPhone) {
      // Redirect to the new SMS verification page
      setLocation("/auth/sms/verify");
      return;
    }
    
    // Auto-redirect to login after a brief delay
    const timer = setTimeout(() => {
      setLocation("/login");
    }, 5000);
    
    return () => clearTimeout(timer);
  }, [setLocation]);

  return (
    <div className="min-h-screen bg-gradient-to-b from-background to-muted flex items-center justify-center p-6">
      <Card className="w-full max-w-md">
        <CardHeader>
          <div className="flex items-center gap-3 mb-2">
            <div className="flex h-12 w-12 items-center justify-center rounded-md bg-primary text-primary-foreground">
              <Smartphone className="h-7 w-7" />
            </div>
            <div>
              <CardTitle className="text-2xl">Phone Verification</CardTitle>
              <CardDescription>Our verification process has been updated</CardDescription>
            </div>
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="bg-muted/50 rounded-lg p-4">
            <p className="text-sm">
              We now use <strong>SMS verification codes</strong> for secure authentication. 
              When you log in with your phone number, we'll send a 6-digit code that 
              verifies your identity and signs you in.
            </p>
          </div>
          
          <div className="space-y-2">
            <Button
              className="w-full"
              onClick={() => setLocation("/login")}
              data-testid="button-go-to-login"
            >
              Go to Login
              <ArrowRight className="ml-2 h-4 w-4" />
            </Button>
            
            <p className="text-center text-sm text-muted-foreground">
              Don't have an account?{" "}
              <Link href="/signup/patient" className="text-primary hover:underline font-medium">
                Sign up
              </Link>
            </p>
          </div>
          
          <p className="text-center text-xs text-muted-foreground">
            Redirecting to login in 5 seconds...
          </p>
        </CardContent>
      </Card>
    </div>
  );
}
