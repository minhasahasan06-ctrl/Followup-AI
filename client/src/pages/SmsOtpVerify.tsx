import { useState, useEffect } from "react";
import { useLocation } from "wouter";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Form, FormControl, FormField, FormItem, FormLabel, FormMessage, FormDescription } from "@/components/ui/form";
import { Input } from "@/components/ui/input";
import { CheckCircle, Smartphone, Loader2 } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import { useAuth } from "@/contexts/AuthContext";
import api from "@/lib/api";
import { Link } from "wouter";

const verifySchema = z.object({
  code: z.string().length(6, "Verification code must be exactly 6 digits"),
});

type VerifyFormData = z.infer<typeof verifySchema>;

export default function SmsOtpVerify() {
  const [, setLocation] = useLocation();
  const { toast } = useToast();
  const { refreshSession } = useAuth();
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isComplete, setIsComplete] = useState(false);
  const [phoneNumber, setPhoneNumber] = useState<string>('');
  const [userRole, setUserRole] = useState<string>('patient');

  const form = useForm<VerifyFormData>({
    resolver: zodResolver(verifySchema),
    defaultValues: {
      code: "",
    },
  });

  useEffect(() => {
    // Get phone number from session storage (set by Login page)
    const storedPhone = sessionStorage.getItem('smsVerificationPhone');
    const storedRole = sessionStorage.getItem('smsVerificationRole');
    
    if (!storedPhone) {
      toast({
        title: "Phone number required",
        description: "Please start the login process again",
        variant: "destructive",
      });
      setLocation("/login");
      return;
    }
    
    setPhoneNumber(storedPhone);
    if (storedRole) {
      setUserRole(storedRole);
    }
  }, [setLocation, toast]);

  const onSubmit = async (data: VerifyFormData) => {
    if (!phoneNumber) return;

    setIsSubmitting(true);
    try {
      const response = await api.post("/auth/sms-otp/authenticate", {
        phone_number: phoneNumber,
        code: data.code,
      });

      if (response.data.success) {
        setIsComplete(true);
        const role = response.data.user?.role || userRole;
        
        // Refresh session to update auth context
        await refreshSession();

        toast({
          title: "Welcome!",
          description: "Phone verified successfully!",
        });

        // Clear stored data
        sessionStorage.removeItem('smsVerificationPhone');
        sessionStorage.removeItem('smsVerificationRole');

        // Redirect based on role
        setTimeout(() => {
          if (role === 'doctor') {
            setLocation('/dashboard');
          } else if (role === 'admin') {
            setLocation('/ml-training');
          } else {
            setLocation('/dashboard');
          }
        }, 1500);
      }
    } catch (error: any) {
      toast({
        title: "Verification failed",
        description: error.response?.data?.error || "Invalid verification code",
        variant: "destructive",
      });
    } finally {
      setIsSubmitting(false);
    }
  };

  const resendCode = async () => {
    if (!phoneNumber) return;
    
    setIsSubmitting(true);
    try {
      await api.post("/auth/sms-otp/send", {
        phone_number: phoneNumber,
      });
      
      toast({
        title: "Code resent!",
        description: "Check your phone for a new verification code.",
      });
    } catch (error: any) {
      toast({
        title: "Failed to resend",
        description: error.response?.data?.error || "Please try again later",
        variant: "destructive",
      });
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-background to-muted flex items-center justify-center p-6">
      <Card className="w-full max-w-md">
        <CardHeader>
          <div className="flex items-center gap-3 mb-2">
            <div className="flex h-12 w-12 items-center justify-center rounded-md bg-primary text-primary-foreground">
              <Smartphone className="h-7 w-7" />
            </div>
            <div>
              <CardTitle className="text-2xl">Verify Your Phone</CardTitle>
              <CardDescription>
                Enter the 6-digit code sent to {phoneNumber ? `****${phoneNumber.slice(-4)}` : 'your phone'}
              </CardDescription>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          {isComplete ? (
            <div className="flex flex-col items-center gap-4 py-6">
              <CheckCircle className="h-16 w-16 text-green-500" />
              <p className="text-center font-semibold">Verification Complete!</p>
              <p className="text-center text-muted-foreground text-sm">
                Redirecting to your dashboard...
              </p>
            </div>
          ) : (
            <Form {...form}>
              <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-4">
                <FormField
                  control={form.control}
                  name="code"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>SMS Verification Code</FormLabel>
                      <FormControl>
                        <Input 
                          placeholder="123456" 
                          maxLength={6}
                          {...field} 
                          data-testid="input-sms-code"
                          className="text-center text-2xl tracking-widest"
                        />
                      </FormControl>
                      <FormDescription>
                        Check your phone for a text message with a 6-digit code
                      </FormDescription>
                      <FormMessage />
                    </FormItem>
                  )}
                />

                <Button
                  type="submit"
                  className="w-full"
                  disabled={isSubmitting}
                  data-testid="button-verify-sms"
                >
                  {isSubmitting ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      Verifying...
                    </>
                  ) : (
                    "Verify Code"
                  )}
                </Button>

                <div className="text-center space-y-2">
                  <p className="text-sm text-muted-foreground">
                    Didn't receive a code?
                  </p>
                  <Button
                    type="button"
                    variant="ghost"
                    onClick={resendCode}
                    disabled={isSubmitting}
                    data-testid="button-resend-sms"
                  >
                    Resend Code
                  </Button>
                </div>
                
                <p className="text-center text-sm text-muted-foreground pt-2">
                  <Link href="/login" className="text-primary hover:underline">
                    Back to Login
                  </Link>
                </p>
              </form>
            </Form>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
