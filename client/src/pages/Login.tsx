import { useState } from "react";
import { useLocation } from "wouter";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Form, FormControl, FormField, FormItem, FormLabel, FormMessage, FormDescription } from "@/components/ui/form";
import { Input } from "@/components/ui/input";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Stethoscope, Mail, Smartphone, Loader2, Zap, CheckCircle } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import { Link } from "wouter";
import { useAuth } from "@/contexts/AuthContext";
import api from "@/lib/api";
import { Separator } from "@/components/ui/separator";

const emailSchema = z.object({
  email: z.string().email("Invalid email address"),
});

const phoneSchema = z.object({
  phone: z.string().regex(/^\+[1-9]\d{1,14}$/, "Use international format (e.g., +1234567890)"),
});

type EmailFormData = z.infer<typeof emailSchema>;
type PhoneFormData = z.infer<typeof phoneSchema>;

type LoginState = 'form' | 'magic-link-sent' | 'sms-sent';

export default function Login() {
  const [, setLocation] = useLocation();
  const { toast } = useToast();
  const { refreshSession } = useAuth();
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [loginState, setLoginState] = useState<LoginState>('form');
  const [submittedEmail, setSubmittedEmail] = useState('');
  const [submittedPhone, setSubmittedPhone] = useState('');

  const emailForm = useForm<EmailFormData>({
    resolver: zodResolver(emailSchema),
    defaultValues: {
      email: "",
    },
  });

  const phoneForm = useForm<PhoneFormData>({
    resolver: zodResolver(phoneSchema),
    defaultValues: {
      phone: "",
    },
  });

  const onEmailSubmit = async (data: EmailFormData) => {
    setIsSubmitting(true);
    try {
      await api.post("/auth/magic-link/send", {
        email: data.email,
      });

      setSubmittedEmail(data.email);
      setLoginState('magic-link-sent');
      
      toast({
        title: "Magic link sent!",
        description: "Check your email to log in.",
      });
    } catch (error: any) {
      toast({
        title: "Failed to send magic link",
        description: error.response?.data?.error || "Please try again",
        variant: "destructive",
      });
    } finally {
      setIsSubmitting(false);
    }
  };

  const onPhoneSubmit = async (data: PhoneFormData) => {
    setIsSubmitting(true);
    try {
      await api.post("/auth/sms-otp/send", {
        phone_number: data.phone,
      });

      // Store phone for verification page
      sessionStorage.setItem('smsVerificationPhone', data.phone);
      
      setSubmittedPhone(data.phone);
      setLoginState('sms-sent');
      
      toast({
        title: "Code sent!",
        description: "Check your phone for the verification code.",
      });

      // Redirect to SMS verification page
      setTimeout(() => {
        setLocation('/auth/sms/verify');
      }, 1500);
    } catch (error: any) {
      toast({
        title: "Failed to send code",
        description: error.response?.data?.error || "Please try again",
        variant: "destructive",
      });
    } finally {
      setIsSubmitting(false);
    }
  };

  const resendMagicLink = async () => {
    if (!submittedEmail) return;
    
    setIsSubmitting(true);
    try {
      await api.post("/auth/magic-link/send", {
        email: submittedEmail,
      });
      
      toast({
        title: "Magic link resent!",
        description: "Check your email for the new link.",
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

  const devLoginAsPatient = async () => {
    try {
      await api.post("/dev/login-as-patient");
      await refreshSession();
      
      toast({
        title: "Dev Login",
        description: "Logged in as test patient",
      });
      
      setLocation("/");
    } catch (error: any) {
      console.error("Dev login error:", error);
      toast({
        title: "Dev Login Failed",
        description: error.response?.data?.message || error.message || "Unable to login as test patient",
        variant: "destructive",
      });
    }
  };

  const devLoginAsDoctor = async () => {
    try {
      await api.post("/dev/login-as-doctor");
      await refreshSession();
      
      toast({
        title: "Dev Login",
        description: "Logged in as test doctor",
      });
      
      setLocation("/");
    } catch (error: any) {
      console.error("Dev login error:", error);
      toast({
        title: "Dev Login Failed",
        description: error.response?.data?.message || error.message || "Unable to login as test doctor",
        variant: "destructive",
      });
    }
  };

  // Magic link sent state
  if (loginState === 'magic-link-sent') {
    return (
      <div className="min-h-screen bg-gradient-to-b from-background to-muted flex items-center justify-center p-6">
        <Card className="w-full max-w-md">
          <CardHeader className="text-center">
            <div className="flex justify-center mb-4">
              <div className="flex h-16 w-16 items-center justify-center rounded-full bg-primary/10 text-primary">
                <Mail className="h-8 w-8" />
              </div>
            </div>
            <CardTitle className="text-2xl">Check Your Email</CardTitle>
            <CardDescription>
              We've sent a magic link to <strong>{submittedEmail}</strong>
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="bg-muted/50 rounded-lg p-4 space-y-2">
              <div className="flex items-start gap-2">
                <CheckCircle className="h-5 w-5 text-green-500 mt-0.5 shrink-0" />
                <p className="text-sm">Click the link in your email to log in</p>
              </div>
              <div className="flex items-start gap-2">
                <CheckCircle className="h-5 w-5 text-green-500 mt-0.5 shrink-0" />
                <p className="text-sm">The link expires in 30 minutes</p>
              </div>
            </div>
            
            <div className="text-center space-y-3">
              <p className="text-sm text-muted-foreground">
                Didn't receive the email?
              </p>
              <Button
                variant="outline"
                onClick={resendMagicLink}
                disabled={isSubmitting}
                data-testid="button-resend-magic-link"
              >
                {isSubmitting ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Sending...
                  </>
                ) : (
                  "Resend Magic Link"
                )}
              </Button>
            </div>
            
            <Button
              variant="ghost"
              className="w-full"
              onClick={() => setLoginState('form')}
              data-testid="button-try-different-method"
            >
              Try a different method
            </Button>
          </CardContent>
        </Card>
      </div>
    );
  }

  // SMS sent state
  if (loginState === 'sms-sent') {
    return (
      <div className="min-h-screen bg-gradient-to-b from-background to-muted flex items-center justify-center p-6">
        <Card className="w-full max-w-md">
          <CardHeader className="text-center">
            <div className="flex justify-center mb-4">
              <div className="flex h-16 w-16 items-center justify-center rounded-full bg-primary/10 text-primary">
                <Smartphone className="h-8 w-8" />
              </div>
            </div>
            <CardTitle className="text-2xl">Check Your Phone</CardTitle>
            <CardDescription>
              We've sent a code to <strong>****{submittedPhone.slice(-4)}</strong>
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex items-center justify-center gap-2">
              <Loader2 className="h-5 w-5 animate-spin text-primary" />
              <p className="text-sm text-muted-foreground">Redirecting to verification...</p>
            </div>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-b from-background to-muted flex items-center justify-center p-6">
      <Card className="w-full max-w-md">
        <CardHeader>
          <div className="flex items-center gap-3 mb-2">
            <Link href="/" className="flex items-center gap-3 group" data-testid="link-logo-home">
              <div className="flex h-12 w-12 items-center justify-center rounded-md bg-primary text-primary-foreground hover-elevate">
                <Stethoscope className="h-7 w-7" />
              </div>
              <div>
                <CardTitle className="text-2xl group-hover:text-primary transition-colors">Welcome Back</CardTitle>
                <CardDescription>Log in to your Followup AI account</CardDescription>
              </div>
            </Link>
          </div>
        </CardHeader>
        <CardContent>
          <Tabs defaultValue="email" className="w-full">
            <TabsList className="grid w-full grid-cols-2">
              <TabsTrigger value="email" data-testid="tab-email">
                <Mail className="h-4 w-4 mr-2" />
                Email
              </TabsTrigger>
              <TabsTrigger value="phone" data-testid="tab-phone">
                <Smartphone className="h-4 w-4 mr-2" />
                Phone
              </TabsTrigger>
            </TabsList>
            
            <TabsContent value="email" className="space-y-4 mt-4">
              <Form {...emailForm}>
                <form onSubmit={emailForm.handleSubmit(onEmailSubmit)} className="space-y-4">
                  <FormField
                    control={emailForm.control}
                    name="email"
                    render={({ field }) => (
                      <FormItem>
                        <FormLabel>Email</FormLabel>
                        <FormControl>
                          <Input 
                            type="email" 
                            placeholder="your@email.com" 
                            {...field} 
                            data-testid="input-email" 
                          />
                        </FormControl>
                        <FormDescription>
                          We'll send you a magic link to log in - no password needed!
                        </FormDescription>
                        <FormMessage />
                      </FormItem>
                    )}
                  />

                  <Button
                    type="submit"
                    className="w-full"
                    disabled={isSubmitting}
                    data-testid="button-send-magic-link"
                  >
                    {isSubmitting ? (
                      <>
                        <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                        Sending...
                      </>
                    ) : (
                      <>
                        <Mail className="mr-2 h-4 w-4" />
                        Send Magic Link
                      </>
                    )}
                  </Button>
                </form>
              </Form>
            </TabsContent>
            
            <TabsContent value="phone" className="space-y-4 mt-4">
              <Form {...phoneForm}>
                <form onSubmit={phoneForm.handleSubmit(onPhoneSubmit)} className="space-y-4">
                  <FormField
                    control={phoneForm.control}
                    name="phone"
                    render={({ field }) => (
                      <FormItem>
                        <FormLabel>Phone Number</FormLabel>
                        <FormControl>
                          <Input 
                            type="tel" 
                            placeholder="+12025551234" 
                            {...field} 
                            data-testid="input-phone" 
                          />
                        </FormControl>
                        <FormDescription>
                          Include country code. We'll send you a verification code via SMS.
                        </FormDescription>
                        <FormMessage />
                      </FormItem>
                    )}
                  />

                  <Button
                    type="submit"
                    className="w-full"
                    disabled={isSubmitting}
                    data-testid="button-send-sms"
                  >
                    {isSubmitting ? (
                      <>
                        <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                        Sending...
                      </>
                    ) : (
                      <>
                        <Smartphone className="mr-2 h-4 w-4" />
                        Send Verification Code
                      </>
                    )}
                  </Button>
                </form>
              </Form>
            </TabsContent>
          </Tabs>

          <div className="mt-6 text-center space-y-2">
            <p className="text-sm text-muted-foreground">
              Don't have an account?
            </p>
            <div className="flex gap-2">
              <Button
                type="button"
                variant="outline"
                className="flex-1"
                onClick={() => setLocation("/signup/patient")}
                data-testid="button-patient-signup"
              >
                Sign up as Patient
              </Button>
              <Button
                type="button"
                variant="outline"
                className="flex-1"
                onClick={() => setLocation("/signup/doctor")}
                data-testid="button-doctor-signup"
              >
                Sign up as Doctor
              </Button>
            </div>
          </div>

          {/* Dev-only quick login buttons - always shown, backend protects endpoints */}
          <Separator className="my-4" />
          <div className="space-y-2">
            <p className="text-xs text-muted-foreground text-center flex items-center justify-center gap-1">
              <Zap className="w-3 h-3" />
              Quick Login (Development)
            </p>
            <div className="grid grid-cols-2 gap-2">
              <Button
                type="button"
                variant="outline"
                size="sm"
                onClick={devLoginAsPatient}
                data-testid="button-dev-login-patient"
              >
                Test Patient
              </Button>
              <Button
                type="button"
                variant="outline"
                size="sm"
                onClick={devLoginAsDoctor}
                data-testid="button-dev-login-doctor"
              >
                Test Doctor
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
