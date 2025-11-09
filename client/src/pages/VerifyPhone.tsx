import { useEffect, useState } from "react";
import { useLocation } from "wouter";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Form, FormControl, FormField, FormItem, FormLabel, FormMessage, FormDescription } from "@/components/ui/form";
import { Input } from "@/components/ui/input";
import { CheckCircle, Smartphone } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import api from "@/lib/api";

const verifySchema = z.object({
  code: z.string().length(6, "Verification code must be exactly 6 digits"),
});

type VerifyFormData = z.infer<typeof verifySchema>;

export default function VerifyPhone() {
  const [, setLocation] = useLocation();
  const { toast } = useToast();
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isComplete, setIsComplete] = useState(false);
  const [email, setEmail] = useState<string | null>(null);
  const [requiresAdminApproval, setRequiresAdminApproval] = useState(false);

  const form = useForm<VerifyFormData>({
    resolver: zodResolver(verifySchema),
    defaultValues: {
      code: "",
    },
  });

  useEffect(() => {
    // Get email from session storage (set by VerifyEmail page)
    const storedEmail = sessionStorage.getItem("verificationEmail");
    if (!storedEmail) {
      toast({
        title: "Email required",
        description: "Please verify your email first",
        variant: "destructive",
      });
      setLocation("/verify-email");
      return;
    }
    setEmail(storedEmail);
  }, [setLocation, toast]);

  const onSubmit = async (data: VerifyFormData) => {
    if (!email) return;

    setIsSubmitting(true);
    try {
      const response = await api.post("/auth/verify-phone", {
        email,
        code: data.code,
      });

      setIsComplete(true);
      setRequiresAdminApproval(response.data.requiresAdminApproval || false);

      toast({
        title: "Success!",
        description: response.data.message || "Phone verified successfully!",
      });

      // Clear stored email
      sessionStorage.removeItem("verificationEmail");

      // Redirect to login after delay
      setTimeout(() => setLocation("/login"), 3000);
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
              <CardDescription>Enter the 6-digit code sent via SMS</CardDescription>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          {isComplete ? (
            <div className="flex flex-col items-center gap-4 py-6">
              <CheckCircle className="h-16 w-16 text-green-500" />
              <p className="text-center font-semibold">Verification Complete!</p>
              {requiresAdminApproval ? (
                <p className="text-center text-muted-foreground text-sm">
                  Your doctor application is under review. You'll receive an email when your account is activated.
                </p>
              ) : (
                <p className="text-center text-muted-foreground text-sm">
                  Redirecting to login...
                </p>
              )}
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
                  data-testid="button-verify-phone"
                >
                  {isSubmitting ? "Verifying..." : "Verify Phone Number"}
                </Button>

                <p className="text-center text-sm text-muted-foreground">
                  Didn't receive a code? Check your spam folder or contact support.
                </p>
              </form>
            </Form>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
