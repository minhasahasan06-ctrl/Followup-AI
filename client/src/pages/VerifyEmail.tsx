import { useState } from "react";
import { useLocation } from "wouter";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Form, FormControl, FormField, FormItem, FormLabel, FormMessage } from "@/components/ui/form";
import { Input } from "@/components/ui/input";
import { CheckCircle, Mail } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import api from "@/lib/api";

const verifySchema = z.object({
  email: z.string().email("Invalid email address"),
  code: z.string().min(6, "Verification code must be at least 6 characters"),
});

type VerifyFormData = z.infer<typeof verifySchema>;

export default function VerifyEmail() {
  const [, setLocation] = useLocation();
  const { toast } = useToast();
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [phoneNumber, setPhoneNumber] = useState<string | null>(null);

  const form = useForm<VerifyFormData>({
    resolver: zodResolver(verifySchema),
    defaultValues: {
      email: "",
      code: "",
    },
  });

  const onSubmit = async (data: VerifyFormData) => {
    setIsSubmitting(true);
    try {
      const response = await api.post("/auth/verify-email", data);

      if (response.data.requiresPhoneVerification) {
        // Step 1 complete, now move to Step 2 (phone verification)
        toast({
          title: "Email verified!",
          description: response.data.message || "Now verify your phone number with the SMS code.",
        });
        
        setPhoneNumber(response.data.phoneNumber);
        
        // Store email for phone verification page
        sessionStorage.setItem("verificationEmail", data.email);
        
        // Redirect to phone verification
        setTimeout(() => setLocation("/verify-phone"), 1500);
      } else {
        // Both steps complete (shouldn't happen from email verify, but handle gracefully)
        toast({
          title: "Success!",
          description: response.data.message || "Your account is verified!",
        });
        
        setTimeout(() => setLocation("/login"), 2000);
      }
    } catch (error: any) {
      toast({
        title: "Verification failed",
        description: error.response?.data?.message || "Invalid verification code",
        variant: "destructive",
      });
    } finally {
      setIsSubmitting(false);
    }
  };

  const resendCode = async () => {
    const email = form.getValues("email");
    if (!email) {
      toast({
        title: "Email required",
        description: "Please enter your email address first",
        variant: "destructive",
      });
      return;
    }

    try {
      await api.post("/auth/resend-code", { email });
      toast({
        title: "Code resent",
        description: "Check your email for a new verification code",
      });
    } catch (error: any) {
      toast({
        title: "Failed to resend code",
        description: error.response?.data?.message || "An error occurred",
        variant: "destructive",
      });
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-background to-muted flex items-center justify-center p-6">
      <Card className="w-full max-w-md">
        <CardHeader>
          <div className="flex items-center gap-3 mb-2">
            <div className="flex h-12 w-12 items-center justify-center rounded-md bg-primary text-primary-foreground">
              <Mail className="h-7 w-7" />
            </div>
            <div>
              <CardTitle className="text-2xl">Verify Your Email</CardTitle>
              <CardDescription>Enter the verification code sent to your email</CardDescription>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          {phoneNumber ? (
            <div className="flex flex-col items-center gap-4 py-6">
              <CheckCircle className="h-16 w-16 text-green-500" />
              <p className="text-center text-muted-foreground">
                Email verified! Redirecting to phone verification...
              </p>
            </div>
          ) : (
            <Form {...form}>
              <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-4">
                <FormField
                  control={form.control}
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
                      <FormMessage />
                    </FormItem>
                  )}
                />

                <FormField
                  control={form.control}
                  name="code"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Verification Code</FormLabel>
                      <FormControl>
                        <Input 
                          placeholder="Enter 6-digit code" 
                          {...field} 
                          data-testid="input-code" 
                        />
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  )}
                />

                <Button
                  type="submit"
                  className="w-full"
                  disabled={isSubmitting}
                  data-testid="button-verify"
                >
                  {isSubmitting ? "Verifying..." : "Verify Email"}
                </Button>

                <Button
                  type="button"
                  variant="ghost"
                  className="w-full"
                  onClick={resendCode}
                  data-testid="button-resend"
                >
                  Resend Code
                </Button>
              </form>
            </Form>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
