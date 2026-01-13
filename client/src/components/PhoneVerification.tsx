import { useState, useEffect } from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Form, FormControl, FormDescription, FormField, FormItem, FormLabel, FormMessage } from "@/components/ui/form";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { Switch } from "@/components/ui/switch";
import { Phone, CheckCircle, AlertCircle, Send, PhoneCall } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import { useQuery, useMutation } from "@tanstack/react-query";
import { queryClient, apiRequest } from "@/lib/queryClient";

const phoneSchema = z.object({
  phoneNumber: z.string().regex(/^\+?[1-9]\d{1,14}$/, "Invalid phone number format. Use international format (e.g., +15551234567)"),
});

const verificationSchema = z.object({
  code: z.string().length(6, "Verification code must be 6 digits"),
});

const smsPreferencesSchema = z.object({
  smsNotificationsEnabled: z.boolean(),
  smsMedicationReminders: z.boolean(),
  smsAppointmentReminders: z.boolean(),
  smsDailyFollowups: z.boolean(),
  smsHealthAlerts: z.boolean(),
});

type PhoneFormData = z.infer<typeof phoneSchema>;
type VerificationFormData = z.infer<typeof verificationSchema>;
type SmsPreferencesData = z.infer<typeof smsPreferencesSchema>;

export default function PhoneVerification() {
  const { toast } = useToast();
  const [verificationSent, setVerificationSent] = useState(false);
  const [channel, setChannel] = useState<'sms' | 'call'>('sms');

  const { data: user, isSuccess } = useQuery<any>({
    queryKey: ["/api/auth/user"],
  });

  const phoneForm = useForm<PhoneFormData>({
    resolver: zodResolver(phoneSchema),
    defaultValues: {
      phoneNumber: user?.phoneNumber || "",
    },
  });

  const verificationForm = useForm<VerificationFormData>({
    resolver: zodResolver(verificationSchema),
    defaultValues: {
      code: "",
    },
  });

  const preferencesForm = useForm<SmsPreferencesData>({
    resolver: zodResolver(smsPreferencesSchema),
    defaultValues: {
      smsNotificationsEnabled: user?.smsNotificationsEnabled ?? true,
      smsMedicationReminders: user?.smsMedicationReminders ?? true,
      smsAppointmentReminders: user?.smsAppointmentReminders ?? true,
      smsDailyFollowups: user?.smsDailyFollowups ?? true,
      smsHealthAlerts: user?.smsHealthAlerts ?? true,
    },
  });

  useEffect(() => {
    if (isSuccess && user) {
      phoneForm.reset({
        phoneNumber: user.phoneNumber || "",
      });
      preferencesForm.reset({
        smsNotificationsEnabled: user.smsNotificationsEnabled ?? true,
        smsMedicationReminders: user.smsMedicationReminders ?? true,
        smsAppointmentReminders: user.smsAppointmentReminders ?? true,
        smsDailyFollowups: user.smsDailyFollowups ?? true,
        smsHealthAlerts: user.smsHealthAlerts ?? true,
      });
    }
  }, [isSuccess, user, phoneForm, preferencesForm]);

  const sendVerificationMutation = useMutation({
    mutationFn: async (data: { phoneNumber: string; channel: 'sms' | 'call' }) => {
      const res = await apiRequest('/api/auth/send-phone-verification', { method: 'POST', json: data });
      return await res.json();
    },
    onSuccess: () => {
      setVerificationSent(true);
      toast({
        title: "Verification code sent",
        description: `A ${channel === 'sms' ? '6-digit code' : 'verification call'} has been sent to your phone.`,
      });
    },
    onError: (error: any) => {
      toast({
        title: "Failed to send verification code",
        description: error.message || "Please try again",
        variant: "destructive",
      });
    },
  });

  const verifyPhoneMutation = useMutation({
    mutationFn: async (data: VerificationFormData) => {
      const res = await apiRequest('/api/auth/verify-phone', { method: 'POST', json: data });
      return await res.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/auth/user"] });
      toast({
        title: "Phone verified",
        description: "Your phone number has been successfully verified!",
      });
      setVerificationSent(false);
      verificationForm.reset();
    },
    onError: (error: any) => {
      toast({
        title: "Verification failed",
        description: error.message || "Invalid code. Please try again.",
        variant: "destructive",
      });
    },
  });

  const updatePreferencesMutation = useMutation({
    mutationFn: async (data: SmsPreferencesData) => {
      const res = await apiRequest('/api/auth/sms-preferences', { method: 'POST', json: data });
      return await res.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/auth/user"] });
      toast({
        title: "Preferences updated",
        description: "Your SMS notification preferences have been saved.",
      });
    },
    onError: (error: any) => {
      toast({
        title: "Update failed",
        description: error.message || "Please try again",
        variant: "destructive",
      });
    },
  });

  const onSendVerification = (data: PhoneFormData) => {
    sendVerificationMutation.mutate({ phoneNumber: data.phoneNumber, channel });
  };

  const onVerifyPhone = (data: VerificationFormData) => {
    verifyPhoneMutation.mutate(data);
  };

  const onUpdatePreferences = (data: SmsPreferencesData) => {
    updatePreferencesMutation.mutate(data);
  };

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <div className="flex items-center gap-2">
            <Phone className="h-5 w-5 text-primary" />
            <CardTitle>Phone Verification</CardTitle>
          </div>
          <CardDescription>
            Verify your phone number to receive SMS notifications and alerts
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {user?.phoneVerified ? (
            <div className="flex items-center gap-2 p-4 bg-green-50 dark:bg-green-950 rounded-md border border-green-200 dark:border-green-800">
              <CheckCircle className="h-5 w-5 text-green-600 dark:text-green-400" />
              <div>
                <p className="font-medium text-green-900 dark:text-green-100">Phone Verified</p>
                <p className="text-sm text-green-700 dark:text-green-300">{user?.phoneNumber}</p>
              </div>
            </div>
          ) : (
            <>
              <Form {...phoneForm}>
                <form onSubmit={phoneForm.handleSubmit(onSendVerification)} className="space-y-4">
                  <FormField
                    control={phoneForm.control}
                    name="phoneNumber"
                    render={({ field }) => (
                      <FormItem>
                        <FormLabel>Phone Number</FormLabel>
                        <FormControl>
                          <Input
                            placeholder="+15551234567"
                            {...field}
                            disabled={verificationSent}
                            data-testid="input-phone-number"
                          />
                        </FormControl>
                        <FormDescription>
                          Enter your phone number in international format (e.g., +1 for US)
                        </FormDescription>
                        <FormMessage />
                      </FormItem>
                    )}
                  />

                  {!verificationSent && (
                    <div className="flex gap-2">
                      <Button
                        type="submit"
                        disabled={sendVerificationMutation.isPending}
                        onClick={() => setChannel('sms')}
                        data-testid="button-send-sms"
                      >
                        <Send className="mr-2 h-4 w-4" />
                        {sendVerificationMutation.isPending && channel === 'sms' ? "Sending..." : "Send SMS Code"}
                      </Button>
                      <Button
                        type="button"
                        variant="outline"
                        disabled={sendVerificationMutation.isPending}
                        onClick={() => {
                          setChannel('call');
                          phoneForm.handleSubmit(onSendVerification)();
                        }}
                        data-testid="button-send-call"
                      >
                        <PhoneCall className="mr-2 h-4 w-4" />
                        {sendVerificationMutation.isPending && channel === 'call' ? "Calling..." : "Call Me"}
                      </Button>
                    </div>
                  )}
                </form>
              </Form>

              {verificationSent && (
                <>
                  <Separator />
                  <Form {...verificationForm}>
                    <form onSubmit={verificationForm.handleSubmit(onVerifyPhone)} className="space-y-4">
                      <FormField
                        control={verificationForm.control}
                        name="code"
                        render={({ field }) => (
                          <FormItem>
                            <FormLabel>Verification Code</FormLabel>
                            <FormControl>
                              <Input
                                placeholder="123456"
                                {...field}
                                maxLength={6}
                                data-testid="input-verification-code"
                              />
                            </FormControl>
                            <FormDescription>
                              Enter the 6-digit code sent to your phone
                            </FormDescription>
                            <FormMessage />
                          </FormItem>
                        )}
                      />

                      <div className="flex gap-2">
                        <Button
                          type="submit"
                          disabled={verifyPhoneMutation.isPending}
                          data-testid="button-verify-phone"
                        >
                          {verifyPhoneMutation.isPending ? "Verifying..." : "Verify Phone"}
                        </Button>
                        <Button
                          type="button"
                          variant="outline"
                          onClick={() => {
                            setVerificationSent(false);
                            verificationForm.reset();
                          }}
                          data-testid="button-cancel-verification"
                        >
                          Cancel
                        </Button>
                      </div>
                    </form>
                  </Form>
                </>
              )}
            </>
          )}
        </CardContent>
      </Card>

      {user?.phoneVerified && (
        <Card>
          <CardHeader>
            <CardTitle>SMS Notification Preferences</CardTitle>
            <CardDescription>
              Choose which SMS notifications you want to receive
            </CardDescription>
          </CardHeader>
          <CardContent>
            <Form {...preferencesForm}>
              <form onSubmit={preferencesForm.handleSubmit(onUpdatePreferences)} className="space-y-4">
                <FormField
                  control={preferencesForm.control}
                  name="smsNotificationsEnabled"
                  render={({ field }) => (
                    <FormItem className="flex items-center justify-between rounded-lg border p-4">
                      <div className="space-y-0.5">
                        <FormLabel className="text-base">Enable SMS Notifications</FormLabel>
                        <FormDescription>
                          Receive all SMS notifications
                        </FormDescription>
                      </div>
                      <FormControl>
                        <Switch
                          checked={field.value}
                          onCheckedChange={field.onChange}
                          data-testid="switch-sms-enabled"
                        />
                      </FormControl>
                    </FormItem>
                  )}
                />

                <FormField
                  control={preferencesForm.control}
                  name="smsMedicationReminders"
                  render={({ field }) => (
                    <FormItem className="flex items-center justify-between rounded-lg border p-4">
                      <div className="space-y-0.5">
                        <FormLabel>Medication Reminders</FormLabel>
                        <FormDescription>
                          Get reminded when it's time to take your medication
                        </FormDescription>
                      </div>
                      <FormControl>
                        <Switch
                          checked={field.value}
                          onCheckedChange={field.onChange}
                          disabled={!preferencesForm.watch('smsNotificationsEnabled')}
                          data-testid="switch-medication-reminders"
                        />
                      </FormControl>
                    </FormItem>
                  )}
                />

                <FormField
                  control={preferencesForm.control}
                  name="smsAppointmentReminders"
                  render={({ field }) => (
                    <FormItem className="flex items-center justify-between rounded-lg border p-4">
                      <div className="space-y-0.5">
                        <FormLabel>Appointment Reminders</FormLabel>
                        <FormDescription>
                          Receive reminders for upcoming appointments
                        </FormDescription>
                      </div>
                      <FormControl>
                        <Switch
                          checked={field.value}
                          onCheckedChange={field.onChange}
                          disabled={!preferencesForm.watch('smsNotificationsEnabled')}
                          data-testid="switch-appointment-reminders"
                        />
                      </FormControl>
                    </FormItem>
                  )}
                />

                <FormField
                  control={preferencesForm.control}
                  name="smsDailyFollowups"
                  render={({ field }) => (
                    <FormItem className="flex items-center justify-between rounded-lg border p-4">
                      <div className="space-y-0.5">
                        <FormLabel>Daily Health Check-ins</FormLabel>
                        <FormDescription>
                          Daily reminders from Agent Clona to check in on your health
                        </FormDescription>
                      </div>
                      <FormControl>
                        <Switch
                          checked={field.value}
                          onCheckedChange={field.onChange}
                          disabled={!preferencesForm.watch('smsNotificationsEnabled')}
                          data-testid="switch-daily-followups"
                        />
                      </FormControl>
                    </FormItem>
                  )}
                />

                <FormField
                  control={preferencesForm.control}
                  name="smsHealthAlerts"
                  render={({ field }) => (
                    <FormItem className="flex items-center justify-between rounded-lg border p-4">
                      <div className="space-y-0.5">
                        <FormLabel>Critical Health Alerts</FormLabel>
                        <FormDescription>
                          Important notifications about your health status
                        </FormDescription>
                      </div>
                      <FormControl>
                        <Switch
                          checked={field.value}
                          onCheckedChange={field.onChange}
                          disabled={!preferencesForm.watch('smsNotificationsEnabled')}
                          data-testid="switch-health-alerts"
                        />
                      </FormControl>
                    </FormItem>
                  )}
                />

                <Button
                  type="submit"
                  disabled={updatePreferencesMutation.isPending}
                  data-testid="button-save-preferences"
                >
                  {updatePreferencesMutation.isPending ? "Saving..." : "Save Preferences"}
                </Button>
              </form>
            </Form>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
