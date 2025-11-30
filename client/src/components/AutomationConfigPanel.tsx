import { useState, useEffect } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Switch } from "@/components/ui/switch";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Separator } from "@/components/ui/separator";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogFooter } from "@/components/ui/dialog";
import { 
  Mail, MessageSquare, Calendar, Clock, Stethoscope, 
  ChevronDown, ChevronRight, Save, RefreshCw, 
  Loader2, Check, AlertCircle, Play, Pause, Settings,
  Zap, Bell, FileText, Pill, Activity, Shield,
  HelpCircle, History, RotateCcw, Brain
} from "lucide-react";
import { SiGoogle, SiWhatsapp } from "react-icons/si";
import { apiRequest, queryClient } from "@/lib/queryClient";
import { useToast } from "@/hooks/use-toast";
import { format, parseISO } from "date-fns";

interface EmailConfig {
  id: string;
  doctor_id: string;
  is_enabled: boolean;
  auto_classify: boolean;
  auto_reply_enabled: boolean;
  forward_urgent_enabled: boolean;
  forward_urgent_to: string | null;
  auto_reply_template: string | null;
  sync_frequency_minutes: number;
  last_sync_at: string | null;
}

interface WhatsAppConfig {
  id: string;
  doctor_id: string;
  is_enabled: boolean;
  auto_reply_enabled: boolean;
  greeting_template: string | null;
  out_of_hours_template: string | null;
  business_hours_start: string;
  business_hours_end: string;
  business_hours_only: boolean;
  sync_frequency_minutes: number;
  last_sync_at: string | null;
}

interface AppointmentConfig {
  id: string;
  doctor_id: string;
  is_enabled: boolean;
  auto_book_enabled: boolean;
  auto_confirm_enabled: boolean;
  default_duration_minutes: number;
  buffer_minutes: number;
  available_hours_start: string;
  available_hours_end: string;
  reminder_enabled: boolean;
  reminder_hours_before: number[];
  calendar_sync_enabled: boolean;
  calendar_sync_frequency_minutes: number;
}

interface ReminderConfig {
  id: string;
  doctor_id: string;
  is_enabled: boolean;
  medication_reminders_enabled: boolean;
  appointment_reminders_enabled: boolean;
  followup_reminders_enabled: boolean;
  noshow_followup_enabled: boolean;
  email_enabled: boolean;
  whatsapp_enabled: boolean;
  sms_enabled: boolean;
  quiet_hours_start: string;
  quiet_hours_end: string;
}

interface ClinicalConfig {
  id: string;
  doctor_id: string;
  is_enabled: boolean;
  auto_soap_notes: boolean;
  auto_icd10_suggest: boolean;
  auto_differential_diagnosis: boolean;
  prescription_assist_enabled: boolean;
  require_prescription_approval: boolean;
  use_patient_history: boolean;
  drug_interaction_check: boolean;
  contraindication_alerts: boolean;
  auto_dosage_recommendations: boolean;
  chronic_refill_enabled: boolean;
  chronic_refill_adherence_threshold: number;
  chronic_refill_days_before_expiry: number;
  chronic_refill_require_approval: boolean;
}

interface RxTemplate {
  id: string;
  doctor_id: string;
  name: string;
  condition: string;
  medication_name: string;
  dosage: string;
  frequency: string;
  duration: string;
  route: string;
  instructions: string;
  is_active: boolean;
}

interface ScheduledJob {
  id: string;
  name: string;
  job_type: string;
  frequency: string;
  is_enabled: boolean;
  last_run_at: string | null;
  next_run_at: string | null;
  last_run_status: string | null;
  run_count: number;
  success_count: number;
  failure_count: number;
}

interface AutomationStatus {
  is_running: boolean;
  jobs_in_queue: number;
  jobs_today: number;
  jobs_completed_today: number;
  jobs_failed_today: number;
  email_sync_status: string;
  email_last_sync: string | null;
  whatsapp_sync_status: string;
  whatsapp_last_sync: string | null;
  calendar_sync_status: string;
  calendar_last_sync: string | null;
}

const DAYS_OF_WEEK = [
  { value: "monday", label: "Mon" },
  { value: "tuesday", label: "Tue" },
  { value: "wednesday", label: "Wed" },
  { value: "thursday", label: "Thu" },
  { value: "friday", label: "Fri" },
  { value: "saturday", label: "Sat" },
  { value: "sunday", label: "Sun" },
];

export function AutomationConfigPanel() {
  const { toast } = useToast();
  const [activeSection, setActiveSection] = useState("email");
  const [hasUnsavedChanges, setHasUnsavedChanges] = useState(false);
  
  const [emailConfig, setEmailConfig] = useState<Partial<EmailConfig>>({});
  const [whatsappConfig, setWhatsappConfig] = useState<Partial<WhatsAppConfig>>({});
  const [appointmentConfig, setAppointmentConfig] = useState<Partial<AppointmentConfig>>({});
  const [reminderConfig, setReminderConfig] = useState<Partial<ReminderConfig>>({});
  const [clinicalConfig, setClinicalConfig] = useState<Partial<ClinicalConfig>>({});

  const { data: emailData, isLoading: emailLoading, refetch: refetchEmail } = useQuery<EmailConfig>({
    queryKey: ["/api/v1/automation/config/email"],
  });

  const { data: whatsappData, isLoading: whatsappLoading, refetch: refetchWhatsapp } = useQuery<WhatsAppConfig>({
    queryKey: ["/api/v1/automation/config/whatsapp"],
  });

  const { data: appointmentData, isLoading: appointmentLoading, refetch: refetchAppointment } = useQuery<AppointmentConfig>({
    queryKey: ["/api/v1/automation/config/appointment"],
  });

  const { data: reminderData, isLoading: reminderLoading, refetch: refetchReminder } = useQuery<ReminderConfig>({
    queryKey: ["/api/v1/automation/config/reminder"],
  });

  const { data: clinicalData, isLoading: clinicalLoading, refetch: refetchClinical } = useQuery<ClinicalConfig>({
    queryKey: ["/api/v1/automation/config/clinical"],
  });

  const { data: automationStatus, refetch: refetchStatus } = useQuery<AutomationStatus>({
    queryKey: ["/api/v1/automation/status"],
    refetchInterval: 30000,
  });

  const { data: scheduledJobs = [], refetch: refetchJobs } = useQuery<ScheduledJob[]>({
    queryKey: ["/api/v1/automation/schedules"],
  });

  const { data: rxTemplates = [], refetch: refetchTemplates } = useQuery<RxTemplate[]>({
    queryKey: ["/api/v1/automation/rx-templates"],
  });

  const [showTemplateDialog, setShowTemplateDialog] = useState(false);
  const [editingTemplate, setEditingTemplate] = useState<RxTemplate | null>(null);
  const [newTemplate, setNewTemplate] = useState<Partial<RxTemplate>>({
    name: "",
    condition: "",
    medication_name: "",
    dosage: "",
    frequency: "once_daily",
    duration: "30 days",
    route: "oral",
    instructions: "",
    is_active: true
  });

  useEffect(() => {
    if (emailData) setEmailConfig(emailData);
  }, [emailData]);

  useEffect(() => {
    if (whatsappData) setWhatsappConfig(whatsappData);
  }, [whatsappData]);

  useEffect(() => {
    if (appointmentData) setAppointmentConfig(appointmentData);
  }, [appointmentData]);

  useEffect(() => {
    if (reminderData) setReminderConfig(reminderData);
  }, [reminderData]);

  useEffect(() => {
    if (clinicalData) setClinicalConfig(clinicalData);
  }, [clinicalData]);

  const saveEmailConfig = useMutation({
    mutationFn: async (config: Partial<EmailConfig>) => {
      const response = await apiRequest("/api/v1/automation/config/email", {
        method: "PUT",
        body: JSON.stringify(config),
      });
      return response.json();
    },
    onSuccess: () => {
      refetchEmail();
      setHasUnsavedChanges(false);
      toast({ title: "Email automation settings saved" });
    },
    onError: () => {
      toast({ title: "Failed to save email settings", variant: "destructive" });
    },
  });

  const saveWhatsappConfig = useMutation({
    mutationFn: async (config: Partial<WhatsAppConfig>) => {
      const response = await apiRequest("/api/v1/automation/config/whatsapp", {
        method: "PUT",
        body: JSON.stringify(config),
      });
      return response.json();
    },
    onSuccess: () => {
      refetchWhatsapp();
      setHasUnsavedChanges(false);
      toast({ title: "WhatsApp automation settings saved" });
    },
    onError: () => {
      toast({ title: "Failed to save WhatsApp settings", variant: "destructive" });
    },
  });

  const saveAppointmentConfig = useMutation({
    mutationFn: async (config: Partial<AppointmentConfig>) => {
      const response = await apiRequest("/api/v1/automation/config/appointment", {
        method: "PUT",
        body: JSON.stringify(config),
      });
      return response.json();
    },
    onSuccess: () => {
      refetchAppointment();
      setHasUnsavedChanges(false);
      toast({ title: "Appointment automation settings saved" });
    },
    onError: () => {
      toast({ title: "Failed to save appointment settings", variant: "destructive" });
    },
  });

  const saveReminderConfig = useMutation({
    mutationFn: async (config: Partial<ReminderConfig>) => {
      const response = await apiRequest("/api/v1/automation/config/reminder", {
        method: "PUT",
        body: JSON.stringify(config),
      });
      return response.json();
    },
    onSuccess: () => {
      refetchReminder();
      setHasUnsavedChanges(false);
      toast({ title: "Reminder automation settings saved" });
    },
    onError: () => {
      toast({ title: "Failed to save reminder settings", variant: "destructive" });
    },
  });

  const saveClinicalConfig = useMutation({
    mutationFn: async (config: Partial<ClinicalConfig>) => {
      const response = await apiRequest("/api/v1/automation/config/clinical", {
        method: "PUT",
        body: JSON.stringify(config),
      });
      return response.json();
    },
    onSuccess: () => {
      refetchClinical();
      setHasUnsavedChanges(false);
      toast({ title: "Clinical automation settings saved" });
    },
    onError: () => {
      toast({ title: "Failed to save clinical settings", variant: "destructive" });
    },
  });

  const saveRxTemplate = useMutation({
    mutationFn: async (template: Partial<RxTemplate>) => {
      const method = template.id ? "PUT" : "POST";
      const url = template.id 
        ? `/api/v1/automation/rx-templates/${template.id}`
        : "/api/v1/automation/rx-templates";
      const response = await apiRequest(url, {
        method,
        body: JSON.stringify(template),
      });
      return response.json();
    },
    onSuccess: () => {
      refetchTemplates();
      setShowTemplateDialog(false);
      setEditingTemplate(null);
      setNewTemplate({
        name: "",
        condition: "",
        medication_name: "",
        dosage: "",
        frequency: "once_daily",
        duration: "30 days",
        route: "oral",
        instructions: "",
        is_active: true
      });
      toast({ title: "Rx template saved successfully" });
    },
    onError: () => {
      toast({ title: "Failed to save Rx template", variant: "destructive" });
    },
  });

  const deleteRxTemplate = useMutation({
    mutationFn: async (templateId: string) => {
      const response = await apiRequest(`/api/v1/automation/rx-templates/${templateId}`, {
        method: "DELETE",
      });
      return response.json();
    },
    onSuccess: () => {
      refetchTemplates();
      toast({ title: "Rx template deleted" });
    },
    onError: () => {
      toast({ title: "Failed to delete template", variant: "destructive" });
    },
  });

  const startEngine = useMutation({
    mutationFn: async () => {
      const response = await apiRequest("/api/v1/automation/engine/start", { method: "POST" });
      return response.json();
    },
    onSuccess: () => {
      refetchStatus();
      toast({ title: "Automation engine started" });
    },
    onError: () => {
      toast({ title: "Failed to start engine", variant: "destructive" });
    },
  });

  const pauseEngine = useMutation({
    mutationFn: async () => {
      const response = await apiRequest("/api/v1/automation/engine/pause", { method: "POST" });
      return response.json();
    },
    onSuccess: () => {
      refetchStatus();
      toast({ title: "Automation engine paused" });
    },
    onError: () => {
      toast({ title: "Failed to pause engine", variant: "destructive" });
    },
  });

  const triggerSync = useMutation({
    mutationFn: async (type: string) => {
      const response = await apiRequest(`/api/v1/automation/trigger/${type}`, { method: "POST" });
      return response.json();
    },
    onSuccess: (_, type) => {
      refetchStatus();
      toast({ title: `${type} sync triggered` });
    },
    onError: (_, type) => {
      toast({ title: `Failed to trigger ${type} sync`, variant: "destructive" });
    },
  });

  const updateEmailField = (field: keyof EmailConfig, value: any) => {
    setEmailConfig((prev) => ({ ...prev, [field]: value }));
    setHasUnsavedChanges(true);
  };

  const updateWhatsappField = (field: keyof WhatsAppConfig, value: any) => {
    setWhatsappConfig((prev) => ({ ...prev, [field]: value }));
    setHasUnsavedChanges(true);
  };

  const updateAppointmentField = (field: keyof AppointmentConfig, value: any) => {
    setAppointmentConfig((prev) => ({ ...prev, [field]: value }));
    setHasUnsavedChanges(true);
  };

  const updateReminderField = (field: keyof ReminderConfig, value: any) => {
    setReminderConfig((prev) => ({ ...prev, [field]: value }));
    setHasUnsavedChanges(true);
  };

  const updateClinicalField = (field: keyof ClinicalConfig, value: any) => {
    setClinicalConfig((prev) => ({ ...prev, [field]: value }));
    setHasUnsavedChanges(true);
  };

  const handleSave = () => {
    switch (activeSection) {
      case "email":
        saveEmailConfig.mutate(emailConfig);
        break;
      case "whatsapp":
        saveWhatsappConfig.mutate(whatsappConfig);
        break;
      case "appointment":
        saveAppointmentConfig.mutate(appointmentConfig);
        break;
      case "reminder":
        saveReminderConfig.mutate(reminderConfig);
        break;
      case "clinical":
        saveClinicalConfig.mutate(clinicalConfig);
        break;
    }
  };

  const isLoading = emailLoading || whatsappLoading || appointmentLoading || reminderLoading || clinicalLoading;
  const isSaving = saveEmailConfig.isPending || saveWhatsappConfig.isPending || 
                   saveAppointmentConfig.isPending || saveReminderConfig.isPending || saveClinicalConfig.isPending;

  return (
    <div className="space-y-6">
      {/* Engine Status Header */}
      <Card data-testid="card-automation-engine-status">
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className={`h-10 w-10 rounded-full flex items-center justify-center ${
                automationStatus?.is_running ? "bg-green-100 dark:bg-green-900/30" : "bg-muted"
              }`}>
                <Activity className={`h-5 w-5 ${
                  automationStatus?.is_running ? "text-green-600" : "text-muted-foreground"
                }`} />
              </div>
              <div>
                <CardTitle className="flex items-center gap-2">
                  Lysa Automation Engine
                  <Badge variant={automationStatus?.is_running ? "default" : "secondary"}>
                    {automationStatus?.is_running ? "Running" : "Paused"}
                  </Badge>
                </CardTitle>
                <CardDescription>
                  {automationStatus?.jobs_in_queue || 0} jobs in queue · {automationStatus?.jobs_completed_today || 0} completed today
                </CardDescription>
              </div>
            </div>
            <div className="flex items-center gap-2">
              {automationStatus?.is_running ? (
                <Button
                  variant="outline"
                  onClick={() => pauseEngine.mutate()}
                  disabled={pauseEngine.isPending}
                  data-testid="button-pause-engine"
                >
                  {pauseEngine.isPending ? (
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  ) : (
                    <Pause className="h-4 w-4 mr-2" />
                  )}
                  Pause
                </Button>
              ) : (
                <Button
                  onClick={() => startEngine.mutate()}
                  disabled={startEngine.isPending}
                  data-testid="button-start-engine"
                >
                  {startEngine.isPending ? (
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  ) : (
                    <Play className="h-4 w-4 mr-2" />
                  )}
                  Start Engine
                </Button>
              )}
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="p-3 rounded-lg border">
              <div className="flex items-center gap-2 mb-1">
                <Mail className="h-4 w-4 text-blue-600" />
                <span className="text-sm font-medium">Email Sync</span>
              </div>
              <p className="text-xs text-muted-foreground">
                {automationStatus?.email_last_sync 
                  ? `Last: ${format(parseISO(automationStatus.email_last_sync), "HH:mm")}`
                  : "Not synced yet"}
              </p>
              <Button 
                variant="ghost" 
                size="sm" 
                className="mt-1 h-7 text-xs"
                onClick={() => triggerSync.mutate("email")}
                disabled={triggerSync.isPending}
                data-testid="button-sync-email"
              >
                <RefreshCw className={`h-3 w-3 mr-1 ${triggerSync.isPending ? 'animate-spin' : ''}`} />
                Sync Now
              </Button>
            </div>
            <div className="p-3 rounded-lg border">
              <div className="flex items-center gap-2 mb-1">
                <SiWhatsapp className="h-4 w-4 text-green-600" />
                <span className="text-sm font-medium">WhatsApp</span>
              </div>
              <p className="text-xs text-muted-foreground">
                {automationStatus?.whatsapp_last_sync 
                  ? `Last: ${format(parseISO(automationStatus.whatsapp_last_sync), "HH:mm")}`
                  : "Not synced yet"}
              </p>
              <Button 
                variant="ghost" 
                size="sm" 
                className="mt-1 h-7 text-xs"
                onClick={() => triggerSync.mutate("whatsapp")}
                disabled={triggerSync.isPending}
                data-testid="button-sync-whatsapp"
              >
                <RefreshCw className={`h-3 w-3 mr-1 ${triggerSync.isPending ? 'animate-spin' : ''}`} />
                Sync Now
              </Button>
            </div>
            <div className="p-3 rounded-lg border">
              <div className="flex items-center gap-2 mb-1">
                <Calendar className="h-4 w-4 text-purple-600" />
                <span className="text-sm font-medium">Calendar</span>
              </div>
              <p className="text-xs text-muted-foreground">
                {automationStatus?.calendar_last_sync 
                  ? `Last: ${format(parseISO(automationStatus.calendar_last_sync), "HH:mm")}`
                  : "Not synced yet"}
              </p>
              <Button 
                variant="ghost" 
                size="sm" 
                className="mt-1 h-7 text-xs"
                onClick={() => triggerSync.mutate("calendar")}
                disabled={triggerSync.isPending}
                data-testid="button-sync-calendar"
              >
                <RefreshCw className={`h-3 w-3 mr-1 ${triggerSync.isPending ? 'animate-spin' : ''}`} />
                Sync Now
              </Button>
            </div>
            <div className="p-3 rounded-lg border">
              <div className="flex items-center gap-2 mb-1">
                <Zap className="h-4 w-4 text-amber-600" />
                <span className="text-sm font-medium">Jobs Today</span>
              </div>
              <div className="flex items-center gap-2 text-sm">
                <span className="text-green-600">{automationStatus?.jobs_completed_today || 0} ✓</span>
                <span className="text-red-600">{automationStatus?.jobs_failed_today || 0} ✗</span>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Configuration Tabs */}
      <Card data-testid="card-automation-config">
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <div>
              <CardTitle>Automation Configuration</CardTitle>
              <CardDescription>Configure how Lysa automates your workflow</CardDescription>
            </div>
            {hasUnsavedChanges && (
              <Button onClick={handleSave} disabled={isSaving} data-testid="button-save-config">
                {isSaving ? (
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                ) : (
                  <Save className="h-4 w-4 mr-2" />
                )}
                Save Changes
              </Button>
            )}
          </div>
        </CardHeader>
        <CardContent>
          {isLoading ? (
            <div className="flex items-center justify-center py-12">
              <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
            </div>
          ) : (
            <Tabs value={activeSection} onValueChange={setActiveSection}>
              <TabsList className="grid grid-cols-5 mb-6">
                <TabsTrigger value="email" className="flex items-center gap-1" data-testid="tab-email-config">
                  <Mail className="h-4 w-4" />
                  <span className="hidden sm:inline">Email</span>
                </TabsTrigger>
                <TabsTrigger value="whatsapp" className="flex items-center gap-1" data-testid="tab-whatsapp-config">
                  <SiWhatsapp className="h-4 w-4" />
                  <span className="hidden sm:inline">WhatsApp</span>
                </TabsTrigger>
                <TabsTrigger value="appointment" className="flex items-center gap-1" data-testid="tab-appointment-config">
                  <Calendar className="h-4 w-4" />
                  <span className="hidden sm:inline">Booking</span>
                </TabsTrigger>
                <TabsTrigger value="reminder" className="flex items-center gap-1" data-testid="tab-reminder-config">
                  <Bell className="h-4 w-4" />
                  <span className="hidden sm:inline">Reminders</span>
                </TabsTrigger>
                <TabsTrigger value="clinical" className="flex items-center gap-1" data-testid="tab-clinical-config">
                  <Stethoscope className="h-4 w-4" />
                  <span className="hidden sm:inline">Clinical</span>
                </TabsTrigger>
              </TabsList>

              {/* Email Automation Config */}
              <TabsContent value="email" className="space-y-6">
                <div className="flex items-center justify-between p-4 bg-muted/50 rounded-lg">
                  <div className="flex items-center gap-3">
                    <div className="h-10 w-10 rounded-full bg-blue-100 dark:bg-blue-900/30 flex items-center justify-center">
                      <Mail className="h-5 w-5 text-blue-600" />
                    </div>
                    <div>
                      <h3 className="font-medium">Email Automation</h3>
                      <p className="text-sm text-muted-foreground">Auto-classify, reply, and forward emails</p>
                    </div>
                  </div>
                  <Switch
                    checked={emailConfig.is_enabled ?? false}
                    onCheckedChange={(checked) => updateEmailField("is_enabled", checked)}
                    data-testid="switch-email-enabled"
                  />
                </div>

                <div className="grid gap-6 md:grid-cols-2">
                  <div className="space-y-4">
                    <div className="flex items-center justify-between">
                      <div>
                        <Label>Auto-Classify Emails</Label>
                        <p className="text-xs text-muted-foreground">Categorize emails by type (urgent, appointment, prescription, general)</p>
                      </div>
                      <Switch
                        checked={emailConfig.auto_classify ?? true}
                        onCheckedChange={(checked) => updateEmailField("auto_classify", checked)}
                        disabled={!emailConfig.is_enabled}
                        data-testid="switch-email-auto-classify"
                      />
                    </div>

                    <div className="flex items-center justify-between">
                      <div>
                        <Label>Auto-Reply</Label>
                        <p className="text-xs text-muted-foreground">Automatically reply to common inquiries</p>
                      </div>
                      <Switch
                        checked={emailConfig.auto_reply_enabled ?? false}
                        onCheckedChange={(checked) => updateEmailField("auto_reply_enabled", checked)}
                        disabled={!emailConfig.is_enabled}
                        data-testid="switch-email-auto-reply"
                      />
                    </div>

                    <div className="flex items-center justify-between">
                      <div>
                        <Label>Forward Urgent Emails</Label>
                        <p className="text-xs text-muted-foreground">Forward urgent emails to specified address</p>
                      </div>
                      <Switch
                        checked={emailConfig.forward_urgent_enabled ?? true}
                        onCheckedChange={(checked) => updateEmailField("forward_urgent_enabled", checked)}
                        disabled={!emailConfig.is_enabled}
                        data-testid="switch-email-forward-urgent"
                      />
                    </div>
                  </div>

                  <div className="space-y-4">
                    <div>
                      <Label htmlFor="forward-email">Forward Urgent To</Label>
                      <Input
                        id="forward-email"
                        type="email"
                        placeholder="urgent@clinic.com"
                        value={emailConfig.forward_urgent_to || ""}
                        onChange={(e) => updateEmailField("forward_urgent_to", e.target.value)}
                        disabled={!emailConfig.is_enabled || !emailConfig.forward_urgent_enabled}
                        data-testid="input-forward-email"
                      />
                    </div>

                    <div>
                      <Label htmlFor="sync-frequency">Sync Frequency (minutes)</Label>
                      <Select
                        value={String(emailConfig.sync_frequency_minutes || 5)}
                        onValueChange={(value) => updateEmailField("sync_frequency_minutes", parseInt(value))}
                        disabled={!emailConfig.is_enabled}
                      >
                        <SelectTrigger id="sync-frequency" data-testid="select-email-sync-frequency">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="1">Every 1 minute</SelectItem>
                          <SelectItem value="5">Every 5 minutes</SelectItem>
                          <SelectItem value="10">Every 10 minutes</SelectItem>
                          <SelectItem value="15">Every 15 minutes</SelectItem>
                          <SelectItem value="30">Every 30 minutes</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                  </div>
                </div>

                {emailConfig.auto_reply_enabled && (
                  <div className="space-y-2">
                    <Label htmlFor="auto-reply-template">Auto-Reply Template</Label>
                    <Textarea
                      id="auto-reply-template"
                      placeholder="Thank you for contacting Dr. {doctor_name}'s office. We have received your message and will respond within 24 hours..."
                      value={emailConfig.auto_reply_template || ""}
                      onChange={(e) => updateEmailField("auto_reply_template", e.target.value)}
                      className="min-h-[100px]"
                      data-testid="textarea-auto-reply-template"
                    />
                    <p className="text-xs text-muted-foreground">
                      Variables: {"{doctor_name}"}, {"{clinic_name}"}, {"{patient_name}"}
                    </p>
                  </div>
                )}
              </TabsContent>

              {/* WhatsApp Automation Config */}
              <TabsContent value="whatsapp" className="space-y-6">
                <div className="flex items-center justify-between p-4 bg-muted/50 rounded-lg">
                  <div className="flex items-center gap-3">
                    <div className="h-10 w-10 rounded-full bg-green-100 dark:bg-green-900/30 flex items-center justify-center">
                      <SiWhatsapp className="h-5 w-5 text-green-600" />
                    </div>
                    <div>
                      <h3 className="font-medium">WhatsApp Automation</h3>
                      <p className="text-sm text-muted-foreground">Auto-reply and template messages</p>
                    </div>
                  </div>
                  <Switch
                    checked={whatsappConfig.is_enabled ?? false}
                    onCheckedChange={(checked) => updateWhatsappField("is_enabled", checked)}
                    data-testid="switch-whatsapp-enabled"
                  />
                </div>

                <div className="grid gap-6 md:grid-cols-2">
                  <div className="space-y-4">
                    <div className="flex items-center justify-between">
                      <div>
                        <Label>Auto-Reply</Label>
                        <p className="text-xs text-muted-foreground">Automatically reply to messages</p>
                      </div>
                      <Switch
                        checked={whatsappConfig.auto_reply_enabled ?? false}
                        onCheckedChange={(checked) => updateWhatsappField("auto_reply_enabled", checked)}
                        disabled={!whatsappConfig.is_enabled}
                        data-testid="switch-whatsapp-auto-reply"
                      />
                    </div>

                    <div className="flex items-center justify-between">
                      <div>
                        <Label>Business Hours Only</Label>
                        <p className="text-xs text-muted-foreground">Only auto-reply during business hours</p>
                      </div>
                      <Switch
                        checked={whatsappConfig.business_hours_only ?? true}
                        onCheckedChange={(checked) => updateWhatsappField("business_hours_only", checked)}
                        disabled={!whatsappConfig.is_enabled}
                        data-testid="switch-whatsapp-business-hours"
                      />
                    </div>
                  </div>

                  <div className="space-y-4">
                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <Label>Business Hours Start</Label>
                        <Input
                          type="time"
                          value={whatsappConfig.business_hours_start || "09:00"}
                          onChange={(e) => updateWhatsappField("business_hours_start", e.target.value)}
                          disabled={!whatsappConfig.is_enabled}
                          data-testid="input-whatsapp-hours-start"
                        />
                      </div>
                      <div>
                        <Label>Business Hours End</Label>
                        <Input
                          type="time"
                          value={whatsappConfig.business_hours_end || "17:00"}
                          onChange={(e) => updateWhatsappField("business_hours_end", e.target.value)}
                          disabled={!whatsappConfig.is_enabled}
                          data-testid="input-whatsapp-hours-end"
                        />
                      </div>
                    </div>

                    <div>
                      <Label>Sync Frequency (minutes)</Label>
                      <Select
                        value={String(whatsappConfig.sync_frequency_minutes || 5)}
                        onValueChange={(value) => updateWhatsappField("sync_frequency_minutes", parseInt(value))}
                        disabled={!whatsappConfig.is_enabled}
                      >
                        <SelectTrigger data-testid="select-whatsapp-sync-frequency">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="1">Every 1 minute</SelectItem>
                          <SelectItem value="5">Every 5 minutes</SelectItem>
                          <SelectItem value="10">Every 10 minutes</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                  </div>
                </div>

                <div className="space-y-4">
                  <div>
                    <Label htmlFor="greeting-template">Greeting Message Template</Label>
                    <Textarea
                      id="greeting-template"
                      placeholder="Hello! Thank you for contacting Dr. {doctor_name}'s office. How can we help you today?"
                      value={whatsappConfig.greeting_template || ""}
                      onChange={(e) => updateWhatsappField("greeting_template", e.target.value)}
                      disabled={!whatsappConfig.is_enabled}
                      data-testid="textarea-greeting-template"
                    />
                  </div>

                  <div>
                    <Label htmlFor="out-of-hours-template">Out of Hours Message</Label>
                    <Textarea
                      id="out-of-hours-template"
                      placeholder="Thank you for your message. Our office hours are {hours}. We will respond during the next business day."
                      value={whatsappConfig.out_of_hours_template || ""}
                      onChange={(e) => updateWhatsappField("out_of_hours_template", e.target.value)}
                      disabled={!whatsappConfig.is_enabled}
                      data-testid="textarea-out-of-hours-template"
                    />
                  </div>
                </div>
              </TabsContent>

              {/* Appointment Automation Config */}
              <TabsContent value="appointment" className="space-y-6">
                <div className="flex items-center justify-between p-4 bg-muted/50 rounded-lg">
                  <div className="flex items-center gap-3">
                    <div className="h-10 w-10 rounded-full bg-purple-100 dark:bg-purple-900/30 flex items-center justify-center">
                      <Calendar className="h-5 w-5 text-purple-600" />
                    </div>
                    <div>
                      <h3 className="font-medium">Appointment Automation</h3>
                      <p className="text-sm text-muted-foreground">Auto-book, confirm, and sync appointments</p>
                    </div>
                  </div>
                  <Switch
                    checked={appointmentConfig.is_enabled ?? false}
                    onCheckedChange={(checked) => updateAppointmentField("is_enabled", checked)}
                    data-testid="switch-appointment-enabled"
                  />
                </div>

                <div className="grid gap-6 md:grid-cols-2">
                  <div className="space-y-4">
                    <div className="flex items-center justify-between">
                      <div>
                        <Label>Auto-Book Appointments</Label>
                        <p className="text-xs text-muted-foreground">Automatically book appointments from requests</p>
                      </div>
                      <Switch
                        checked={appointmentConfig.auto_book_enabled ?? false}
                        onCheckedChange={(checked) => updateAppointmentField("auto_book_enabled", checked)}
                        disabled={!appointmentConfig.is_enabled}
                        data-testid="switch-auto-book"
                      />
                    </div>

                    <div className="flex items-center justify-between">
                      <div>
                        <Label>Auto-Confirm Appointments</Label>
                        <p className="text-xs text-muted-foreground">Send confirmation after booking</p>
                      </div>
                      <Switch
                        checked={appointmentConfig.auto_confirm_enabled ?? true}
                        onCheckedChange={(checked) => updateAppointmentField("auto_confirm_enabled", checked)}
                        disabled={!appointmentConfig.is_enabled}
                        data-testid="switch-auto-confirm"
                      />
                    </div>

                    <div className="flex items-center justify-between">
                      <div>
                        <Label>Calendar Sync</Label>
                        <p className="text-xs text-muted-foreground">Sync with Google Calendar</p>
                      </div>
                      <Switch
                        checked={appointmentConfig.calendar_sync_enabled ?? true}
                        onCheckedChange={(checked) => updateAppointmentField("calendar_sync_enabled", checked)}
                        disabled={!appointmentConfig.is_enabled}
                        data-testid="switch-calendar-sync"
                      />
                    </div>

                    <div className="flex items-center justify-between">
                      <div>
                        <Label>Appointment Reminders</Label>
                        <p className="text-xs text-muted-foreground">Send reminder before appointments</p>
                      </div>
                      <Switch
                        checked={appointmentConfig.reminder_enabled ?? true}
                        onCheckedChange={(checked) => updateAppointmentField("reminder_enabled", checked)}
                        disabled={!appointmentConfig.is_enabled}
                        data-testid="switch-appointment-reminders"
                      />
                    </div>
                  </div>

                  <div className="space-y-4">
                    <div>
                      <Label>Default Appointment Duration</Label>
                      <Select
                        value={String(appointmentConfig.default_duration_minutes || 30)}
                        onValueChange={(value) => updateAppointmentField("default_duration_minutes", parseInt(value))}
                        disabled={!appointmentConfig.is_enabled}
                      >
                        <SelectTrigger data-testid="select-duration">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="15">15 minutes</SelectItem>
                          <SelectItem value="30">30 minutes</SelectItem>
                          <SelectItem value="45">45 minutes</SelectItem>
                          <SelectItem value="60">60 minutes</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>

                    <div>
                      <Label>Buffer Between Appointments</Label>
                      <Select
                        value={String(appointmentConfig.buffer_minutes || 15)}
                        onValueChange={(value) => updateAppointmentField("buffer_minutes", parseInt(value))}
                        disabled={!appointmentConfig.is_enabled}
                      >
                        <SelectTrigger data-testid="select-buffer">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="0">No buffer</SelectItem>
                          <SelectItem value="5">5 minutes</SelectItem>
                          <SelectItem value="10">10 minutes</SelectItem>
                          <SelectItem value="15">15 minutes</SelectItem>
                          <SelectItem value="30">30 minutes</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>

                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <Label>Available Hours Start</Label>
                        <Input
                          type="time"
                          value={appointmentConfig.available_hours_start || "09:00"}
                          onChange={(e) => updateAppointmentField("available_hours_start", e.target.value)}
                          disabled={!appointmentConfig.is_enabled}
                          data-testid="input-available-start"
                        />
                      </div>
                      <div>
                        <Label>Available Hours End</Label>
                        <Input
                          type="time"
                          value={appointmentConfig.available_hours_end || "17:00"}
                          onChange={(e) => updateAppointmentField("available_hours_end", e.target.value)}
                          disabled={!appointmentConfig.is_enabled}
                          data-testid="input-available-end"
                        />
                      </div>
                    </div>
                  </div>
                </div>
              </TabsContent>

              {/* Reminder Automation Config */}
              <TabsContent value="reminder" className="space-y-6">
                <div className="flex items-center justify-between p-4 bg-muted/50 rounded-lg">
                  <div className="flex items-center gap-3">
                    <div className="h-10 w-10 rounded-full bg-amber-100 dark:bg-amber-900/30 flex items-center justify-center">
                      <Bell className="h-5 w-5 text-amber-600" />
                    </div>
                    <div>
                      <h3 className="font-medium">Reminder Automation</h3>
                      <p className="text-sm text-muted-foreground">Medication, appointment, and follow-up reminders</p>
                    </div>
                  </div>
                  <Switch
                    checked={reminderConfig.is_enabled ?? false}
                    onCheckedChange={(checked) => updateReminderField("is_enabled", checked)}
                    data-testid="switch-reminder-enabled"
                  />
                </div>

                <div className="grid gap-6 md:grid-cols-2">
                  <div className="space-y-4">
                    <h4 className="font-medium text-sm">Reminder Types</h4>
                    <div className="flex items-center justify-between">
                      <div>
                        <Label>Medication Reminders</Label>
                        <p className="text-xs text-muted-foreground">Remind patients to take medications</p>
                      </div>
                      <Switch
                        checked={reminderConfig.medication_reminders_enabled ?? true}
                        onCheckedChange={(checked) => updateReminderField("medication_reminders_enabled", checked)}
                        disabled={!reminderConfig.is_enabled}
                        data-testid="switch-medication-reminders"
                      />
                    </div>

                    <div className="flex items-center justify-between">
                      <div>
                        <Label>Appointment Reminders</Label>
                        <p className="text-xs text-muted-foreground">Remind patients of upcoming appointments</p>
                      </div>
                      <Switch
                        checked={reminderConfig.appointment_reminders_enabled ?? true}
                        onCheckedChange={(checked) => updateReminderField("appointment_reminders_enabled", checked)}
                        disabled={!reminderConfig.is_enabled}
                        data-testid="switch-appointment-reminders"
                      />
                    </div>

                    <div className="flex items-center justify-between">
                      <div>
                        <Label>Follow-up Reminders</Label>
                        <p className="text-xs text-muted-foreground">Remind patients for follow-up visits</p>
                      </div>
                      <Switch
                        checked={reminderConfig.followup_reminders_enabled ?? true}
                        onCheckedChange={(checked) => updateReminderField("followup_reminders_enabled", checked)}
                        disabled={!reminderConfig.is_enabled}
                        data-testid="switch-followup-reminders"
                      />
                    </div>

                    <div className="flex items-center justify-between">
                      <div>
                        <Label>No-Show Follow-up</Label>
                        <p className="text-xs text-muted-foreground">Follow up with patients who missed appointments</p>
                      </div>
                      <Switch
                        checked={reminderConfig.noshow_followup_enabled ?? true}
                        onCheckedChange={(checked) => updateReminderField("noshow_followup_enabled", checked)}
                        disabled={!reminderConfig.is_enabled}
                        data-testid="switch-noshow-followup"
                      />
                    </div>
                  </div>

                  <div className="space-y-4">
                    <h4 className="font-medium text-sm">Delivery Channels</h4>
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <Mail className="h-4 w-4 text-blue-600" />
                        <Label>Email</Label>
                      </div>
                      <Switch
                        checked={reminderConfig.email_enabled ?? true}
                        onCheckedChange={(checked) => updateReminderField("email_enabled", checked)}
                        disabled={!reminderConfig.is_enabled}
                        data-testid="switch-reminder-email"
                      />
                    </div>

                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <SiWhatsapp className="h-4 w-4 text-green-600" />
                        <Label>WhatsApp</Label>
                      </div>
                      <Switch
                        checked={reminderConfig.whatsapp_enabled ?? false}
                        onCheckedChange={(checked) => updateReminderField("whatsapp_enabled", checked)}
                        disabled={!reminderConfig.is_enabled}
                        data-testid="switch-reminder-whatsapp"
                      />
                    </div>

                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <MessageSquare className="h-4 w-4 text-purple-600" />
                        <Label>SMS</Label>
                      </div>
                      <Switch
                        checked={reminderConfig.sms_enabled ?? false}
                        onCheckedChange={(checked) => updateReminderField("sms_enabled", checked)}
                        disabled={!reminderConfig.is_enabled}
                        data-testid="switch-reminder-sms"
                      />
                    </div>

                    <Separator />

                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <Label>Quiet Hours Start</Label>
                        <Input
                          type="time"
                          value={reminderConfig.quiet_hours_start || "21:00"}
                          onChange={(e) => updateReminderField("quiet_hours_start", e.target.value)}
                          disabled={!reminderConfig.is_enabled}
                          data-testid="input-quiet-start"
                        />
                      </div>
                      <div>
                        <Label>Quiet Hours End</Label>
                        <Input
                          type="time"
                          value={reminderConfig.quiet_hours_end || "08:00"}
                          onChange={(e) => updateReminderField("quiet_hours_end", e.target.value)}
                          disabled={!reminderConfig.is_enabled}
                          data-testid="input-quiet-end"
                        />
                      </div>
                    </div>
                    <p className="text-xs text-muted-foreground">No reminders will be sent during quiet hours</p>
                  </div>
                </div>
              </TabsContent>

              {/* Clinical Automation Config */}
              <TabsContent value="clinical" className="space-y-6">
                <div className="flex items-center justify-between p-4 bg-muted/50 rounded-lg">
                  <div className="flex items-center gap-3">
                    <div className="h-10 w-10 rounded-full bg-pink-100 dark:bg-pink-900/30 flex items-center justify-center">
                      <Stethoscope className="h-5 w-5 text-pink-600" />
                    </div>
                    <div>
                      <h3 className="font-medium">Clinical Automation</h3>
                      <p className="text-sm text-muted-foreground">AI-powered diagnosis, prescriptions, and clinical decision support</p>
                    </div>
                  </div>
                  <Switch
                    checked={clinicalConfig.is_enabled ?? false}
                    onCheckedChange={(checked) => updateClinicalField("is_enabled", checked)}
                    data-testid="switch-clinical-enabled"
                  />
                </div>

                {/* AI Diagnosis Assist Section */}
                <Card className="border-2 border-purple-200 dark:border-purple-800">
                  <CardHeader className="pb-3">
                    <div className="flex items-center gap-2">
                      <Brain className="h-5 w-5 text-purple-600" />
                      <CardTitle className="text-base">AI Diagnosis Assist</CardTitle>
                    </div>
                    <CardDescription>Lysa pre-analyzes patient data and suggests diagnoses</CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="flex items-center justify-between">
                      <div>
                        <Label>Auto-Generate SOAP Notes</Label>
                        <p className="text-xs text-muted-foreground">Generate structured SOAP notes from conversations</p>
                      </div>
                      <Switch
                        checked={clinicalConfig.auto_soap_notes ?? true}
                        onCheckedChange={(checked) => updateClinicalField("auto_soap_notes", checked)}
                        disabled={!clinicalConfig.is_enabled}
                        data-testid="switch-auto-soap"
                      />
                    </div>

                    <div className="flex items-center justify-between">
                      <div>
                        <Label>ICD-10 Code Suggestions</Label>
                        <p className="text-xs text-muted-foreground">Automatically suggest ICD-10 codes based on symptoms</p>
                      </div>
                      <Switch
                        checked={clinicalConfig.auto_icd10_suggest ?? true}
                        onCheckedChange={(checked) => updateClinicalField("auto_icd10_suggest", checked)}
                        disabled={!clinicalConfig.is_enabled}
                        data-testid="switch-auto-icd10"
                      />
                    </div>

                    <div className="flex items-center justify-between">
                      <div>
                        <Label>Differential Diagnosis Lists</Label>
                        <p className="text-xs text-muted-foreground">Generate differential diagnosis based on patient data</p>
                      </div>
                      <Switch
                        checked={clinicalConfig.auto_differential_diagnosis ?? true}
                        onCheckedChange={(checked) => updateClinicalField("auto_differential_diagnosis", checked)}
                        disabled={!clinicalConfig.is_enabled}
                        data-testid="switch-auto-differential"
                      />
                    </div>

                    <div className="flex items-center justify-between">
                      <div>
                        <Label>Use Patient History</Label>
                        <p className="text-xs text-muted-foreground">Consider patient medical history for diagnosis suggestions</p>
                      </div>
                      <Switch
                        checked={clinicalConfig.use_patient_history ?? true}
                        onCheckedChange={(checked) => updateClinicalField("use_patient_history", checked)}
                        disabled={!clinicalConfig.is_enabled}
                        data-testid="switch-use-history"
                      />
                    </div>
                  </CardContent>
                </Card>

                {/* Rx Builder Assist Section */}
                <Card className="border-2 border-blue-200 dark:border-blue-800">
                  <CardHeader className="pb-3">
                    <div className="flex items-center gap-2">
                      <Pill className="h-5 w-5 text-blue-600" />
                      <CardTitle className="text-base">Rx Builder Assist</CardTitle>
                    </div>
                    <CardDescription>AI-powered prescription assistance with safety checks</CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="flex items-center justify-between">
                      <div>
                        <Label>Medication Suggestions</Label>
                        <p className="text-xs text-muted-foreground">Suggest medications based on diagnosis</p>
                      </div>
                      <Switch
                        checked={clinicalConfig.prescription_assist_enabled ?? true}
                        onCheckedChange={(checked) => updateClinicalField("prescription_assist_enabled", checked)}
                        disabled={!clinicalConfig.is_enabled}
                        data-testid="switch-prescription-assist"
                      />
                    </div>

                    <div className="flex items-center justify-between">
                      <div>
                        <Label>Drug-Drug Interaction Check</Label>
                        <p className="text-xs text-muted-foreground">Automatically check for medication interactions</p>
                      </div>
                      <Switch
                        checked={clinicalConfig.drug_interaction_check ?? true}
                        onCheckedChange={(checked) => updateClinicalField("drug_interaction_check", checked)}
                        disabled={!clinicalConfig.is_enabled}
                        data-testid="switch-drug-interaction"
                      />
                    </div>

                    <div className="flex items-center justify-between">
                      <div>
                        <Label>Contraindication Alerts</Label>
                        <p className="text-xs text-muted-foreground">Alert for medication contraindications based on patient history</p>
                      </div>
                      <Switch
                        checked={clinicalConfig.contraindication_alerts ?? true}
                        onCheckedChange={(checked) => updateClinicalField("contraindication_alerts", checked)}
                        disabled={!clinicalConfig.is_enabled}
                        data-testid="switch-contraindication"
                      />
                    </div>

                    <div className="flex items-center justify-between">
                      <div>
                        <Label>Auto-Fill Dosage Recommendations</Label>
                        <p className="text-xs text-muted-foreground">Suggest standard dosages based on condition and patient profile</p>
                      </div>
                      <Switch
                        checked={clinicalConfig.auto_dosage_recommendations ?? true}
                        onCheckedChange={(checked) => updateClinicalField("auto_dosage_recommendations", checked)}
                        disabled={!clinicalConfig.is_enabled}
                        data-testid="switch-auto-dosage"
                      />
                    </div>

                    <div className="flex items-center justify-between">
                      <div>
                        <Label>Require Doctor Approval</Label>
                        <p className="text-xs text-muted-foreground">All prescriptions require physician sign-off</p>
                      </div>
                      <Switch
                        checked={clinicalConfig.require_prescription_approval ?? true}
                        onCheckedChange={(checked) => updateClinicalField("require_prescription_approval", checked)}
                        disabled={!clinicalConfig.is_enabled}
                        data-testid="switch-require-approval"
                      />
                    </div>
                  </CardContent>
                </Card>

                {/* Chronic Refill Automation Section */}
                <Card className="border-2 border-green-200 dark:border-green-800">
                  <CardHeader className="pb-3">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <RotateCcw className="h-5 w-5 text-green-600" />
                        <CardTitle className="text-base">Chronic Refill Automation</CardTitle>
                      </div>
                      <Switch
                        checked={clinicalConfig.chronic_refill_enabled ?? false}
                        onCheckedChange={(checked) => updateClinicalField("chronic_refill_enabled", checked)}
                        disabled={!clinicalConfig.is_enabled}
                        data-testid="switch-chronic-refill"
                      />
                    </div>
                    <CardDescription>Auto-generate refill prescriptions for stable chronic patients</CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <Label>Minimum Adherence Threshold</Label>
                        <div className="flex items-center gap-2 mt-1">
                          <Input
                            type="number"
                            min={50}
                            max={100}
                            value={clinicalConfig.chronic_refill_adherence_threshold ?? 80}
                            onChange={(e) => updateClinicalField("chronic_refill_adherence_threshold", parseInt(e.target.value))}
                            disabled={!clinicalConfig.is_enabled || !clinicalConfig.chronic_refill_enabled}
                            className="w-20"
                            data-testid="input-adherence-threshold"
                          />
                          <span className="text-sm text-muted-foreground">%</span>
                        </div>
                        <p className="text-xs text-muted-foreground mt-1">Patient must meet this adherence level</p>
                      </div>
                      <div>
                        <Label>Days Before Expiry to Alert</Label>
                        <div className="flex items-center gap-2 mt-1">
                          <Input
                            type="number"
                            min={1}
                            max={30}
                            value={clinicalConfig.chronic_refill_days_before_expiry ?? 7}
                            onChange={(e) => updateClinicalField("chronic_refill_days_before_expiry", parseInt(e.target.value))}
                            disabled={!clinicalConfig.is_enabled || !clinicalConfig.chronic_refill_enabled}
                            className="w-20"
                            data-testid="input-days-before-expiry"
                          />
                          <span className="text-sm text-muted-foreground">days</span>
                        </div>
                        <p className="text-xs text-muted-foreground mt-1">Alert before prescription expires</p>
                      </div>
                    </div>

                    <div className="flex items-center justify-between">
                      <div>
                        <Label>Require Doctor Approval for Refills</Label>
                        <p className="text-xs text-muted-foreground">Doctor must approve each auto-generated refill</p>
                      </div>
                      <Switch
                        checked={clinicalConfig.chronic_refill_require_approval ?? true}
                        onCheckedChange={(checked) => updateClinicalField("chronic_refill_require_approval", checked)}
                        disabled={!clinicalConfig.is_enabled || !clinicalConfig.chronic_refill_enabled}
                        data-testid="switch-refill-approval"
                      />
                    </div>
                  </CardContent>
                </Card>

                {/* Rx Templates Section */}
                <Card className="border-2 border-orange-200 dark:border-orange-800">
                  <CardHeader className="pb-3">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <FileText className="h-5 w-5 text-orange-600" />
                        <CardTitle className="text-base">Rx Templates</CardTitle>
                      </div>
                      <Button
                        size="sm"
                        onClick={() => {
                          setEditingTemplate(null);
                          setNewTemplate({
                            name: "",
                            condition: "",
                            medication_name: "",
                            dosage: "",
                            frequency: "once_daily",
                            duration: "30 days",
                            route: "oral",
                            instructions: "",
                            is_active: true
                          });
                          setShowTemplateDialog(true);
                        }}
                        disabled={!clinicalConfig.is_enabled}
                        data-testid="button-add-template"
                      >
                        <Plus className="h-4 w-4 mr-1" />
                        Add Template
                      </Button>
                    </div>
                    <CardDescription>Saved prescription templates for common conditions</CardDescription>
                  </CardHeader>
                  <CardContent>
                    {rxTemplates.length === 0 ? (
                      <div className="text-center py-6 text-muted-foreground">
                        <FileText className="h-10 w-10 mx-auto mb-3 opacity-50" />
                        <p>No Rx templates configured</p>
                        <p className="text-xs">Add templates for quick prescription creation</p>
                      </div>
                    ) : (
                      <ScrollArea className="h-[200px]">
                        <div className="space-y-2">
                          {rxTemplates.map((template) => (
                            <div
                              key={template.id}
                              className="flex items-center justify-between p-3 rounded-lg border hover-elevate"
                              data-testid={`rx-template-${template.id}`}
                            >
                              <div className="flex items-center gap-3">
                                <div className={`h-2 w-2 rounded-full ${template.is_active ? 'bg-green-500' : 'bg-gray-400'}`} />
                                <div>
                                  <p className="font-medium text-sm">{template.name}</p>
                                  <p className="text-xs text-muted-foreground">
                                    {template.medication_name} {template.dosage} - {template.condition}
                                  </p>
                                </div>
                              </div>
                              <div className="flex items-center gap-2">
                                <Button
                                  size="sm"
                                  variant="ghost"
                                  onClick={() => {
                                    setEditingTemplate(template);
                                    setNewTemplate(template);
                                    setShowTemplateDialog(true);
                                  }}
                                  data-testid={`button-edit-template-${template.id}`}
                                >
                                  <Settings className="h-4 w-4" />
                                </Button>
                                <Button
                                  size="sm"
                                  variant="ghost"
                                  onClick={() => deleteRxTemplate.mutate(template.id)}
                                  data-testid={`button-delete-template-${template.id}`}
                                >
                                  <X className="h-4 w-4 text-destructive" />
                                </Button>
                              </div>
                            </div>
                          ))}
                        </div>
                      </ScrollArea>
                    )}
                  </CardContent>
                </Card>

                <div className="p-4 bg-amber-50 dark:bg-amber-900/20 rounded-lg border border-amber-200 dark:border-amber-800">
                  <div className="flex items-start gap-3">
                    <Shield className="h-5 w-5 text-amber-600 mt-0.5" />
                    <div>
                      <p className="font-medium text-amber-800 dark:text-amber-200">Clinical Safety Notice</p>
                      <p className="text-sm text-amber-700 dark:text-amber-300">
                        All clinical suggestions are AI-assisted and require physician review. 
                        Lysa is designed to assist, not replace, clinical decision-making.
                      </p>
                    </div>
                  </div>
                </div>
              </TabsContent>
            </Tabs>
          )}
        </CardContent>
      </Card>

      {/* Scheduled Jobs Overview */}
      <Card data-testid="card-scheduled-jobs">
        <CardHeader className="pb-3">
          <CardTitle className="flex items-center gap-2">
            <History className="h-5 w-5" />
            Scheduled Jobs
          </CardTitle>
          <CardDescription>Background tasks running automatically</CardDescription>
        </CardHeader>
        <CardContent>
          {scheduledJobs.length === 0 ? (
            <div className="text-center py-6 text-muted-foreground">
              <Clock className="h-10 w-10 mx-auto mb-3 opacity-50" />
              <p>No scheduled jobs configured</p>
            </div>
          ) : (
            <ScrollArea className="h-[200px]">
              <div className="space-y-2">
                {scheduledJobs.map((job) => (
                  <div
                    key={job.id}
                    className="flex items-center justify-between p-3 rounded-lg border"
                    data-testid={`scheduled-job-${job.id}`}
                  >
                    <div className="flex items-center gap-3">
                      <div className={`h-2 w-2 rounded-full ${job.is_enabled ? 'bg-green-500' : 'bg-gray-400'}`} />
                      <div>
                        <p className="font-medium text-sm">{job.name}</p>
                        <p className="text-xs text-muted-foreground">
                          {job.frequency} · {job.run_count} runs ({job.success_count} success, {job.failure_count} failed)
                        </p>
                      </div>
                    </div>
                    <div className="text-right">
                      {job.next_run_at && (
                        <p className="text-xs text-muted-foreground">
                          Next: {format(parseISO(job.next_run_at), "MMM d, HH:mm")}
                        </p>
                      )}
                      <Badge variant={job.last_run_status === 'completed' ? 'default' : job.last_run_status === 'failed' ? 'destructive' : 'secondary'}>
                        {job.last_run_status || 'pending'}
                      </Badge>
                    </div>
                  </div>
                ))}
              </div>
            </ScrollArea>
          )}
        </CardContent>
      </Card>

      {/* Rx Template Dialog */}
      <Dialog open={showTemplateDialog} onOpenChange={setShowTemplateDialog}>
        <DialogContent className="max-w-2xl">
          <DialogHeader>
            <DialogTitle>{editingTemplate ? "Edit Rx Template" : "Add New Rx Template"}</DialogTitle>
            <DialogDescription>
              Create a prescription template for quick reuse with common conditions
            </DialogDescription>
          </DialogHeader>
          
          <div className="grid gap-4 py-4">
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label htmlFor="template-name">Template Name</Label>
                <Input
                  id="template-name"
                  placeholder="e.g., Hypertension Standard"
                  value={newTemplate.name || ""}
                  onChange={(e) => setNewTemplate({ ...newTemplate, name: e.target.value })}
                  data-testid="input-template-name"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="template-condition">Condition</Label>
                <Input
                  id="template-condition"
                  placeholder="e.g., Hypertension"
                  value={newTemplate.condition || ""}
                  onChange={(e) => setNewTemplate({ ...newTemplate, condition: e.target.value })}
                  data-testid="input-template-condition"
                />
              </div>
            </div>

            <Separator />

            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label htmlFor="template-medication">Medication Name</Label>
                <Input
                  id="template-medication"
                  placeholder="e.g., Lisinopril"
                  value={newTemplate.medication_name || ""}
                  onChange={(e) => setNewTemplate({ ...newTemplate, medication_name: e.target.value })}
                  data-testid="input-template-medication"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="template-dosage">Dosage</Label>
                <Input
                  id="template-dosage"
                  placeholder="e.g., 10mg"
                  value={newTemplate.dosage || ""}
                  onChange={(e) => setNewTemplate({ ...newTemplate, dosage: e.target.value })}
                  data-testid="input-template-dosage"
                />
              </div>
            </div>

            <div className="grid grid-cols-3 gap-4">
              <div className="space-y-2">
                <Label htmlFor="template-frequency">Frequency</Label>
                <Select
                  value={newTemplate.frequency || "once_daily"}
                  onValueChange={(value) => setNewTemplate({ ...newTemplate, frequency: value })}
                >
                  <SelectTrigger data-testid="select-template-frequency">
                    <SelectValue placeholder="Select frequency" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="once_daily">Once daily</SelectItem>
                    <SelectItem value="twice_daily">Twice daily (BID)</SelectItem>
                    <SelectItem value="three_times_daily">Three times daily (TID)</SelectItem>
                    <SelectItem value="four_times_daily">Four times daily (QID)</SelectItem>
                    <SelectItem value="every_4_hours">Every 4 hours</SelectItem>
                    <SelectItem value="every_6_hours">Every 6 hours</SelectItem>
                    <SelectItem value="every_8_hours">Every 8 hours</SelectItem>
                    <SelectItem value="every_12_hours">Every 12 hours</SelectItem>
                    <SelectItem value="as_needed">As needed (PRN)</SelectItem>
                    <SelectItem value="at_bedtime">At bedtime (HS)</SelectItem>
                    <SelectItem value="weekly">Weekly</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-2">
                <Label htmlFor="template-route">Route</Label>
                <Select
                  value={newTemplate.route || "oral"}
                  onValueChange={(value) => setNewTemplate({ ...newTemplate, route: value })}
                >
                  <SelectTrigger data-testid="select-template-route">
                    <SelectValue placeholder="Select route" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="oral">Oral (PO)</SelectItem>
                    <SelectItem value="sublingual">Sublingual (SL)</SelectItem>
                    <SelectItem value="topical">Topical</SelectItem>
                    <SelectItem value="inhalation">Inhalation</SelectItem>
                    <SelectItem value="injection_im">Intramuscular (IM)</SelectItem>
                    <SelectItem value="injection_iv">Intravenous (IV)</SelectItem>
                    <SelectItem value="injection_sc">Subcutaneous (SC)</SelectItem>
                    <SelectItem value="rectal">Rectal (PR)</SelectItem>
                    <SelectItem value="ophthalmic">Ophthalmic (Eye)</SelectItem>
                    <SelectItem value="otic">Otic (Ear)</SelectItem>
                    <SelectItem value="nasal">Nasal</SelectItem>
                    <SelectItem value="transdermal">Transdermal (Patch)</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-2">
                <Label htmlFor="template-duration">Duration</Label>
                <Input
                  id="template-duration"
                  placeholder="e.g., 30 days"
                  value={newTemplate.duration || ""}
                  onChange={(e) => setNewTemplate({ ...newTemplate, duration: e.target.value })}
                  data-testid="input-template-duration"
                />
              </div>
            </div>

            <div className="space-y-2">
              <Label htmlFor="template-instructions">Special Instructions</Label>
              <Textarea
                id="template-instructions"
                placeholder="e.g., Take with food. Avoid grapefruit."
                value={newTemplate.instructions || ""}
                onChange={(e) => setNewTemplate({ ...newTemplate, instructions: e.target.value })}
                rows={2}
                data-testid="textarea-template-instructions"
              />
            </div>

            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Switch
                  checked={newTemplate.is_active ?? true}
                  onCheckedChange={(checked) => setNewTemplate({ ...newTemplate, is_active: checked })}
                  data-testid="switch-template-active"
                />
                <Label>Template Active</Label>
              </div>
            </div>
          </div>

          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => {
                setShowTemplateDialog(false);
                setEditingTemplate(null);
              }}
              data-testid="button-cancel-template"
            >
              Cancel
            </Button>
            <Button
              onClick={() => {
                if (editingTemplate) {
                  saveRxTemplate.mutate({ ...newTemplate, id: editingTemplate.id });
                } else {
                  saveRxTemplate.mutate(newTemplate);
                }
              }}
              disabled={!newTemplate.name || !newTemplate.medication_name || !newTemplate.dosage || saveRxTemplate.isPending}
              data-testid="button-save-template"
            >
              {saveRxTemplate.isPending ? (
                <>
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  Saving...
                </>
              ) : (
                <>
                  <Check className="h-4 w-4 mr-2" />
                  {editingTemplate ? "Update Template" : "Save Template"}
                </>
              )}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
