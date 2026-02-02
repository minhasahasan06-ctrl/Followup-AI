import { useState } from "react";
import { useLocation } from "wouter";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Form, FormControl, FormField, FormItem, FormLabel, FormMessage, FormDescription } from "@/components/ui/form";
import { Input } from "@/components/ui/input";
import { Checkbox } from "@/components/ui/checkbox";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Heart, Mail, CheckCircle } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import { Link } from "wouter";
import api from "@/lib/api";

const signupSchema = z.object({
  email: z.string().email("Invalid email address"),
  firstName: z.string().min(1, "First name is required"),
  lastName: z.string().min(1, "Last name is required"),
  phoneNumber: z.string().regex(/^\+?[1-9]\d{1,14}$/, "Invalid phone number (use international format, e.g., +1234567890)"),
  ehrImportMethod: z.enum(["manual", "hospital", "platform"], {
    required_error: "Please select how you want to import your medical history",
  }),
  ehrPlatform: z.string().optional(),
  termsAccepted: z.boolean().refine(val => val === true, "You must accept the terms and conditions"),
  researchConsent: z.boolean().optional().default(false),
});

type SignupFormData = z.infer<typeof signupSchema>;

type SignupStep = 'form' | 'magic-link-sent';

export default function PatientSignup() {
  const [, setLocation] = useLocation();
  const { toast } = useToast();
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [step, setStep] = useState<SignupStep>('form');
  const [submittedEmail, setSubmittedEmail] = useState('');

  const form = useForm<SignupFormData>({
    resolver: zodResolver(signupSchema),
    defaultValues: {
      email: "",
      firstName: "",
      lastName: "",
      phoneNumber: "",
      ehrImportMethod: undefined,
      ehrPlatform: "",
      termsAccepted: false,
      researchConsent: false,
    },
  });

  const ehrImportMethod = form.watch("ehrImportMethod");

  const onSubmit = async (data: SignupFormData) => {
    setIsSubmitting(true);
    try {
      // Step 1: Store profile data in backend
      await api.post("/auth/signup/patient", {
        email: data.email,
        firstName: data.firstName,
        lastName: data.lastName,
        phoneNumber: data.phoneNumber,
        ehrImportMethod: data.ehrImportMethod,
        ehrPlatform: data.ehrPlatform,
        researchConsent: data.researchConsent,
      });

      // Step 2: Send magic link for authentication
      await api.post("/auth/magic-link/send", {
        email: data.email,
        role: "patient",
      });

      setSubmittedEmail(data.email);
      setStep('magic-link-sent');
      
      toast({
        title: "Check your email!",
        description: "We've sent you a magic link to complete your signup.",
      });
    } catch (error: any) {
      toast({
        title: "Signup failed",
        description: error.response?.data?.error || error.response?.data?.message || "An error occurred during signup",
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
        role: "patient",
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

  if (step === 'magic-link-sent') {
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
                <p className="text-sm">Click the link in your email to complete signup</p>
              </div>
              <div className="flex items-start gap-2">
                <CheckCircle className="h-5 w-5 text-green-500 mt-0.5 shrink-0" />
                <p className="text-sm">The link expires in 30 minutes</p>
              </div>
              <div className="flex items-start gap-2">
                <CheckCircle className="h-5 w-5 text-green-500 mt-0.5 shrink-0" />
                <p className="text-sm">Your 7-day free trial starts when you sign in</p>
              </div>
            </div>
            
            <div className="text-center space-y-3">
              <p className="text-sm text-muted-foreground">
                Didn't receive the email? Check your spam folder or
              </p>
              <Button
                variant="outline"
                onClick={resendMagicLink}
                disabled={isSubmitting}
                data-testid="button-resend-magic-link"
              >
                {isSubmitting ? "Sending..." : "Resend Magic Link"}
              </Button>
            </div>
            
            <p className="text-center text-sm text-muted-foreground pt-4">
              Already have an account?{" "}
              <Link href="/login" className="text-primary hover:underline font-medium">
                Log in
              </Link>
            </p>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-b from-background to-muted flex items-center justify-center p-6">
      <Card className="w-full max-w-2xl">
        <CardHeader>
          <div className="flex items-center gap-3 mb-2">
            <div className="flex h-12 w-12 items-center justify-center rounded-md bg-primary text-primary-foreground">
              <Heart className="h-7 w-7" />
            </div>
            <div>
              <CardTitle className="text-2xl">Patient Sign Up</CardTitle>
              <CardDescription>Create your Followup AI patient account - Start your 7-day free trial!</CardDescription>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <Form {...form}>
            <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <FormField
                  control={form.control}
                  name="firstName"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>First Name</FormLabel>
                      <FormControl>
                        <Input placeholder="Jane" {...field} data-testid="input-first-name" />
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  )}
                />
                <FormField
                  control={form.control}
                  name="lastName"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Last Name</FormLabel>
                      <FormControl>
                        <Input placeholder="Smith" {...field} data-testid="input-last-name" />
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  )}
                />
              </div>

              <FormField
                control={form.control}
                name="email"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Email</FormLabel>
                    <FormControl>
                      <Input type="email" placeholder="patient@example.com" {...field} data-testid="input-email" />
                    </FormControl>
                    <FormDescription>
                      We'll send a magic link to this email to complete signup - no password needed!
                    </FormDescription>
                    <FormMessage />
                  </FormItem>
                )}
              />

              <FormField
                control={form.control}
                name="phoneNumber"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Phone Number</FormLabel>
                    <FormControl>
                      <Input type="tel" placeholder="+12025551234" {...field} data-testid="input-phone-number" />
                    </FormControl>
                    <FormDescription>
                      Include country code (e.g., +1 for US). Used for SMS notifications and 2FA.
                    </FormDescription>
                    <FormMessage />
                  </FormItem>
                )}
              />

              <FormField
                control={form.control}
                name="ehrImportMethod"
                render={({ field }) => (
                  <FormItem className="space-y-3">
                    <FormLabel>How would you like to import your medical history?</FormLabel>
                    <FormControl>
                      <RadioGroup
                        onValueChange={field.onChange}
                        value={field.value}
                        className="flex flex-col space-y-1"
                      >
                        <FormItem className="flex items-center space-x-3 space-y-0">
                          <FormControl>
                            <RadioGroupItem value="manual" data-testid="radio-manual" />
                          </FormControl>
                          <FormLabel className="font-normal">
                            Upload manually - I'll upload my medical records myself
                          </FormLabel>
                        </FormItem>
                        <FormItem className="flex items-center space-x-3 space-y-0">
                          <FormControl>
                            <RadioGroupItem value="hospital" data-testid="radio-hospital" />
                          </FormControl>
                          <FormLabel className="font-normal">
                            Request from hospital - I want to request records from my healthcare provider
                          </FormLabel>
                        </FormItem>
                        <FormItem className="flex items-center space-x-3 space-y-0">
                          <FormControl>
                            <RadioGroupItem value="platform" data-testid="radio-platform" />
                          </FormControl>
                          <FormLabel className="font-normal">
                            Connect EHR platform - I use an electronic health record system
                          </FormLabel>
                        </FormItem>
                      </RadioGroup>
                    </FormControl>
                    <FormMessage />
                  </FormItem>
                )}
              />

              {ehrImportMethod === "platform" && (
                <FormField
                  control={form.control}
                  name="ehrPlatform"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Select Your EHR Platform</FormLabel>
                      <Select onValueChange={field.onChange} value={field.value}>
                        <FormControl>
                          <SelectTrigger data-testid="select-ehr-platform">
                            <SelectValue placeholder="Choose your platform" />
                          </SelectTrigger>
                        </FormControl>
                        <SelectContent>
                          <SelectItem value="Epic MyChart">Epic MyChart</SelectItem>
                          <SelectItem value="Cerner Health">Cerner Health</SelectItem>
                          <SelectItem value="Apple Health">Apple Health</SelectItem>
                          <SelectItem value="Google Health">Google Health</SelectItem>
                          <SelectItem value="Allscripts FollowMyHealth">Allscripts FollowMyHealth</SelectItem>
                          <SelectItem value="athenahealth">athenahealth</SelectItem>
                          <SelectItem value="Other">Other</SelectItem>
                        </SelectContent>
                      </Select>
                      <FormDescription>
                        We'll securely connect to your EHR platform using industry-standard FHIR protocols
                      </FormDescription>
                      <FormMessage />
                    </FormItem>
                  )}
                />
              )}

              <FormField
                control={form.control}
                name="termsAccepted"
                render={({ field }) => (
                  <FormItem className="flex flex-row items-start space-x-3 space-y-0 p-4 border rounded-md">
                    <FormControl>
                      <Checkbox
                        checked={field.value}
                        onCheckedChange={field.onChange}
                        data-testid="checkbox-terms"
                      />
                    </FormControl>
                    <div className="space-y-1 leading-none">
                      <FormLabel className="text-sm font-normal">
                        I agree to the{" "}
                        <Link href="/terms" className="text-primary hover:underline font-medium">
                          Terms and Conditions
                        </Link>{" "}
                        and{" "}
                        <Link href="/privacy" className="text-primary hover:underline font-medium">
                          Privacy Policy
                        </Link>
                      </FormLabel>
                      <p className="text-xs text-muted-foreground">
                        By signing up, you get a 7-day free trial with 20 consultation credits
                      </p>
                      <FormMessage />
                    </div>
                  </FormItem>
                )}
              />

              <FormField
                control={form.control}
                name="researchConsent"
                render={({ field }) => (
                  <FormItem className="flex flex-row items-start space-x-3 space-y-0 p-4 border rounded-md bg-muted/30">
                    <FormControl>
                      <Checkbox
                        checked={field.value}
                        onCheckedChange={field.onChange}
                        data-testid="checkbox-research-consent"
                      />
                    </FormControl>
                    <div className="space-y-1 leading-none">
                      <FormLabel className="text-sm font-normal">
                        I consent to sharing my de-identified data for medical research (optional)
                      </FormLabel>
                      <p className="text-xs text-muted-foreground">
                        Help advance chronic care research. Your data will be de-identified and protected with privacy safeguards including k-anonymity (k≥25). You can change this anytime in Settings.
                      </p>
                    </div>
                  </FormItem>
                )}
              />

              <Button
                type="submit"
                className="w-full"
                disabled={isSubmitting}
                data-testid="button-signup"
              >
                {isSubmitting ? "Creating account..." : "Start Free Trial"}
              </Button>

              <p className="text-center text-sm text-muted-foreground">
                Already have an account?{" "}
                <Link href="/login" className="text-primary hover:underline font-medium">
                  Log in
                </Link>
              </p>
            </form>
          </Form>
        </CardContent>
      </Card>
    </div>
  );
}
