import { useState, useEffect } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger, DialogFooter } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Separator } from "@/components/ui/separator";
import { 
  Activity, 
  Mail, 
  MessageSquare, 
  Calendar, 
  Bell, 
  CheckCircle2,
  XCircle,
  Clock,
  Loader2,
  RefreshCw,
  ChevronDown,
  Zap,
  TrendingUp,
  Settings,
  Play,
  Pause,
  AlertCircle,
  Webhook,
  Bot
} from "lucide-react";
import { formatDistanceToNow } from "date-fns";
import { apiRequest, queryClient } from "@/lib/queryClient";
import { useToast } from "@/hooks/use-toast";

interface AutomationStatus {
  is_running: boolean;
  current_job?: {
    id: string;
    job_type: string;
    status: string;
  };
  jobs_in_queue: number;
  jobs_today: number;
  jobs_completed_today: number;
  jobs_failed_today: number;
  email_sync_status: string;
  email_last_sync?: string;
  emails_processed_today: number;
  whatsapp_sync_status: string;
  whatsapp_last_sync?: string;
  messages_sent_today: number;
  calendar_sync_status: string;
  calendar_last_sync?: string;
  appointments_booked_today: number;
  reminders_sent_today: number;
  recent_activities: Array<{
    id: string;
    message: string;
    action_type?: string;
    level: string;
    timestamp: string;
  }>;
}

interface DashboardStats {
  total_jobs_today: number;
  completed_jobs_today: number;
  failed_jobs_today: number;
  pending_jobs: number;
  running_jobs: number;
  emails_synced: number;
  emails_classified: number;
  emails_auto_replied: number;
  emails_forwarded: number;
  whatsapp_messages_received: number;
  whatsapp_messages_sent: number;
  whatsapp_auto_replies: number;
  appointments_requested: number;
  appointments_booked: number;
  appointments_confirmed: number;
  reminders_medication: number;
  reminders_appointment: number;
  reminders_followup: number;
  clinical_soap_notes: number;
  clinical_diagnoses: number;
  clinical_prescriptions: number;
  avg_job_duration_ms: number;
  success_rate: number;
  last_email_sync?: string;
  last_whatsapp_sync?: string;
  last_calendar_sync?: string;
}

interface ChannelConfig {
  email: {
    enabled: boolean;
    auto_reply: boolean;
    auto_classify: boolean;
    forward_urgent: boolean;
    sync_interval_minutes: number;
  };
  whatsapp: {
    enabled: boolean;
    auto_reply: boolean;
    business_hours_only: boolean;
    welcome_message?: string;
    away_message?: string;
  };
  calendar: {
    enabled: boolean;
    bidirectional_sync: boolean;
    sync_interval_minutes: number;
    auto_book_appointments: boolean;
  };
}

interface WebhookStatus {
  gmail: {
    connected: boolean;
    watch_expiration?: string;
    last_notification?: string;
  };
  whatsapp: {
    connected: boolean;
    verify_token_set: boolean;
    last_webhook?: string;
  };
}

function ChannelConfigDialog({ 
  channel, 
  children 
}: { 
  channel: 'email' | 'whatsapp' | 'calendar';
  children: React.ReactNode;
}) {
  const { toast } = useToast();
  const [open, setOpen] = useState(false);
  const [saving, setSaving] = useState(false);
  const [config, setConfig] = useState<ChannelConfig[typeof channel] | null>(null);

  const { data: currentConfig } = useQuery<ChannelConfig>({
    queryKey: ["/api/v1/automation/config"],
    enabled: open,
  });

  useEffect(() => {
    if (open && currentConfig && currentConfig[channel]) {
      setConfig(currentConfig[channel]);
    }
  }, [open, currentConfig, channel]);

  const updateConfigMutation = useMutation({
    mutationFn: async (newConfig: Partial<ChannelConfig>) => {
      const response = await apiRequest("/api/v1/automation/config", {
        method: "PATCH",
        json: newConfig,
      });
      return response.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/v1/automation/config"] });
      queryClient.invalidateQueries({ queryKey: ["/api/v1/automation/status"] });
      toast({
        title: "Configuration Updated",
        description: `${channel.charAt(0).toUpperCase() + channel.slice(1)} settings have been saved.`,
      });
      setOpen(false);
    },
    onError: () => {
      toast({
        title: "Error",
        description: "Failed to update configuration. Please try again.",
        variant: "destructive",
      });
    },
  });

  const handleSave = async () => {
    if (!config) return;
    setSaving(true);
    await updateConfigMutation.mutateAsync({ [channel]: config });
    setSaving(false);
  };

  const renderEmailConfig = () => (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <Label htmlFor="email-enabled">Enable Email Automation</Label>
          <p className="text-xs text-muted-foreground">Sync and process emails automatically</p>
        </div>
        <Switch
          id="email-enabled"
          checked={(config as ChannelConfig['email'])?.enabled ?? currentConfig?.email?.enabled ?? false}
          onCheckedChange={(checked) => setConfig(prev => ({ ...(prev ?? currentConfig?.email ?? {}), enabled: checked } as ChannelConfig['email']))}
          data-testid="switch-email-enabled"
        />
      </div>
      <Separator />
      <div className="flex items-center justify-between">
        <div>
          <Label htmlFor="email-auto-reply">Auto-Reply</Label>
          <p className="text-xs text-muted-foreground">AI-powered automatic responses</p>
        </div>
        <Switch
          id="email-auto-reply"
          checked={(config as ChannelConfig['email'])?.auto_reply ?? currentConfig?.email?.auto_reply ?? false}
          onCheckedChange={(checked) => setConfig(prev => ({ ...(prev ?? currentConfig?.email ?? {}), auto_reply: checked } as ChannelConfig['email']))}
          data-testid="switch-email-auto-reply"
        />
      </div>
      <div className="flex items-center justify-between">
        <div>
          <Label htmlFor="email-classify">Auto-Classify</Label>
          <p className="text-xs text-muted-foreground">AI categorization (urgent, patient inquiry, etc.)</p>
        </div>
        <Switch
          id="email-classify"
          checked={(config as ChannelConfig['email'])?.auto_classify ?? currentConfig?.email?.auto_classify ?? true}
          onCheckedChange={(checked) => setConfig(prev => ({ ...(prev ?? currentConfig?.email ?? {}), auto_classify: checked } as ChannelConfig['email']))}
          data-testid="switch-email-classify"
        />
      </div>
      <div className="flex items-center justify-between">
        <div>
          <Label htmlFor="email-forward">Forward Urgent</Label>
          <p className="text-xs text-muted-foreground">Forward critical emails immediately</p>
        </div>
        <Switch
          id="email-forward"
          checked={(config as ChannelConfig['email'])?.forward_urgent ?? currentConfig?.email?.forward_urgent ?? true}
          onCheckedChange={(checked) => setConfig(prev => ({ ...(prev ?? currentConfig?.email ?? {}), forward_urgent: checked } as ChannelConfig['email']))}
          data-testid="switch-email-forward"
        />
      </div>
      <div>
        <Label htmlFor="email-interval">Sync Interval (minutes)</Label>
        <Select
          value={String((config as ChannelConfig['email'])?.sync_interval_minutes ?? currentConfig?.email?.sync_interval_minutes ?? 5)}
          onValueChange={(value) => setConfig(prev => ({ ...(prev ?? currentConfig?.email ?? {}), sync_interval_minutes: parseInt(value) } as ChannelConfig['email']))}
        >
          <SelectTrigger id="email-interval" data-testid="select-email-interval">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="1">1 minute</SelectItem>
            <SelectItem value="5">5 minutes</SelectItem>
            <SelectItem value="15">15 minutes</SelectItem>
            <SelectItem value="30">30 minutes</SelectItem>
          </SelectContent>
        </Select>
      </div>
    </div>
  );

  const renderWhatsAppConfig = () => (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <Label htmlFor="whatsapp-enabled">Enable WhatsApp Automation</Label>
          <p className="text-xs text-muted-foreground">Automatic message handling</p>
        </div>
        <Switch
          id="whatsapp-enabled"
          checked={(config as ChannelConfig['whatsapp'])?.enabled ?? currentConfig?.whatsapp?.enabled ?? false}
          onCheckedChange={(checked) => setConfig(prev => ({ ...(prev ?? currentConfig?.whatsapp ?? {}), enabled: checked } as ChannelConfig['whatsapp']))}
          data-testid="switch-whatsapp-enabled"
        />
      </div>
      <Separator />
      <div className="flex items-center justify-between">
        <div>
          <Label htmlFor="whatsapp-auto-reply">Auto-Reply</Label>
          <p className="text-xs text-muted-foreground">AI-powered message responses</p>
        </div>
        <Switch
          id="whatsapp-auto-reply"
          checked={(config as ChannelConfig['whatsapp'])?.auto_reply ?? currentConfig?.whatsapp?.auto_reply ?? false}
          onCheckedChange={(checked) => setConfig(prev => ({ ...(prev ?? currentConfig?.whatsapp ?? {}), auto_reply: checked } as ChannelConfig['whatsapp']))}
          data-testid="switch-whatsapp-auto-reply"
        />
      </div>
      <div className="flex items-center justify-between">
        <div>
          <Label htmlFor="whatsapp-hours">Business Hours Only</Label>
          <p className="text-xs text-muted-foreground">Limit auto-replies to business hours</p>
        </div>
        <Switch
          id="whatsapp-hours"
          checked={(config as ChannelConfig['whatsapp'])?.business_hours_only ?? currentConfig?.whatsapp?.business_hours_only ?? true}
          onCheckedChange={(checked) => setConfig(prev => ({ ...(prev ?? currentConfig?.whatsapp ?? {}), business_hours_only: checked } as ChannelConfig['whatsapp']))}
          data-testid="switch-whatsapp-hours"
        />
      </div>
      <div>
        <Label htmlFor="whatsapp-welcome">Welcome Message</Label>
        <Textarea
          id="whatsapp-welcome"
          placeholder="Hello! Thank you for contacting our clinic. How can we assist you today?"
          value={(config as ChannelConfig['whatsapp'])?.welcome_message ?? currentConfig?.whatsapp?.welcome_message ?? ""}
          onChange={(e) => setConfig(prev => ({ ...(prev ?? currentConfig?.whatsapp ?? {}), welcome_message: e.target.value } as ChannelConfig['whatsapp']))}
          className="mt-1"
          data-testid="input-whatsapp-welcome"
        />
      </div>
      <div>
        <Label htmlFor="whatsapp-away">Away Message</Label>
        <Textarea
          id="whatsapp-away"
          placeholder="Thank you for your message. Our office is currently closed. We'll respond during business hours."
          value={(config as ChannelConfig['whatsapp'])?.away_message ?? currentConfig?.whatsapp?.away_message ?? ""}
          onChange={(e) => setConfig(prev => ({ ...(prev ?? currentConfig?.whatsapp ?? {}), away_message: e.target.value } as ChannelConfig['whatsapp']))}
          className="mt-1"
          data-testid="input-whatsapp-away"
        />
      </div>
    </div>
  );

  const renderCalendarConfig = () => (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <Label htmlFor="calendar-enabled">Enable Calendar Sync</Label>
          <p className="text-xs text-muted-foreground">Sync with Google Calendar</p>
        </div>
        <Switch
          id="calendar-enabled"
          checked={(config as ChannelConfig['calendar'])?.enabled ?? currentConfig?.calendar?.enabled ?? false}
          onCheckedChange={(checked) => setConfig(prev => ({ ...(prev ?? currentConfig?.calendar ?? {}), enabled: checked } as ChannelConfig['calendar']))}
          data-testid="switch-calendar-enabled"
        />
      </div>
      <Separator />
      <div className="flex items-center justify-between">
        <div>
          <Label htmlFor="calendar-bidirectional">Bidirectional Sync</Label>
          <p className="text-xs text-muted-foreground">Sync changes both ways</p>
        </div>
        <Switch
          id="calendar-bidirectional"
          checked={(config as ChannelConfig['calendar'])?.bidirectional_sync ?? currentConfig?.calendar?.bidirectional_sync ?? true}
          onCheckedChange={(checked) => setConfig(prev => ({ ...(prev ?? currentConfig?.calendar ?? {}), bidirectional_sync: checked } as ChannelConfig['calendar']))}
          data-testid="switch-calendar-bidirectional"
        />
      </div>
      <div className="flex items-center justify-between">
        <div>
          <Label htmlFor="calendar-auto-book">Auto-Book Appointments</Label>
          <p className="text-xs text-muted-foreground">AI processes appointment requests</p>
        </div>
        <Switch
          id="calendar-auto-book"
          checked={(config as ChannelConfig['calendar'])?.auto_book_appointments ?? currentConfig?.calendar?.auto_book_appointments ?? false}
          onCheckedChange={(checked) => setConfig(prev => ({ ...(prev ?? currentConfig?.calendar ?? {}), auto_book_appointments: checked } as ChannelConfig['calendar']))}
          data-testid="switch-calendar-auto-book"
        />
      </div>
      <div>
        <Label htmlFor="calendar-interval">Sync Interval (minutes)</Label>
        <Select
          value={String((config as ChannelConfig['calendar'])?.sync_interval_minutes ?? currentConfig?.calendar?.sync_interval_minutes ?? 15)}
          onValueChange={(value) => setConfig(prev => ({ ...(prev ?? currentConfig?.calendar ?? {}), sync_interval_minutes: parseInt(value) } as ChannelConfig['calendar']))}
        >
          <SelectTrigger id="calendar-interval" data-testid="select-calendar-interval">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="5">5 minutes</SelectItem>
            <SelectItem value="15">15 minutes</SelectItem>
            <SelectItem value="30">30 minutes</SelectItem>
            <SelectItem value="60">1 hour</SelectItem>
          </SelectContent>
        </Select>
      </div>
    </div>
  );

  const channelLabels = {
    email: 'Email',
    whatsapp: 'WhatsApp',
    calendar: 'Calendar',
  };

  const channelIcons = {
    email: <Mail className="h-5 w-5" />,
    whatsapp: <MessageSquare className="h-5 w-5" />,
    calendar: <Calendar className="h-5 w-5" />,
  };

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        {children}
      </DialogTrigger>
      <DialogContent className="sm:max-w-md" data-testid={`dialog-${channel}-config`}>
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            {channelIcons[channel]}
            {channelLabels[channel]} Configuration
          </DialogTitle>
          <DialogDescription>
            Configure automation settings for {channelLabels[channel].toLowerCase()} processing.
          </DialogDescription>
        </DialogHeader>
        <div className="py-4">
          {channel === 'email' && renderEmailConfig()}
          {channel === 'whatsapp' && renderWhatsAppConfig()}
          {channel === 'calendar' && renderCalendarConfig()}
        </div>
        <DialogFooter>
          <Button variant="outline" onClick={() => setOpen(false)} data-testid="button-cancel-config">
            Cancel
          </Button>
          <Button onClick={handleSave} disabled={saving} data-testid="button-save-config">
            {saving ? <Loader2 className="h-4 w-4 animate-spin mr-2" /> : null}
            Save Changes
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

function WebhookStatusPanel() {
  const { data: webhookStatus, isLoading } = useQuery<WebhookStatus>({
    queryKey: ["/api/v1/webhooks/status"],
    refetchInterval: 60000,
  });

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-4">
        <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />
      </div>
    );
  }

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2 text-sm font-medium">
        <Webhook className="h-4 w-4 text-muted-foreground" />
        Webhook Connections
      </div>
      
      <div className="space-y-2">
        <div className="flex items-center justify-between p-2 rounded-md border" data-testid="webhook-gmail">
          <div className="flex items-center gap-2">
            <Mail className="h-4 w-4 text-blue-600" />
            <span className="text-sm">Gmail Pub/Sub</span>
          </div>
          <div className="flex items-center gap-2">
            {webhookStatus?.gmail?.connected ? (
              <>
                <CheckCircle2 className="h-4 w-4 text-green-600" />
                <span className="text-xs text-green-600">Connected</span>
              </>
            ) : (
              <>
                <XCircle className="h-4 w-4 text-muted-foreground" />
                <span className="text-xs text-muted-foreground">Not configured</span>
              </>
            )}
          </div>
        </div>

        <div className="flex items-center justify-between p-2 rounded-md border" data-testid="webhook-whatsapp">
          <div className="flex items-center gap-2">
            <MessageSquare className="h-4 w-4 text-green-600" />
            <span className="text-sm">WhatsApp Cloud API</span>
          </div>
          <div className="flex items-center gap-2">
            {webhookStatus?.whatsapp?.connected ? (
              <>
                <CheckCircle2 className="h-4 w-4 text-green-600" />
                <span className="text-xs text-green-600">Connected</span>
              </>
            ) : (
              <>
                <XCircle className="h-4 w-4 text-muted-foreground" />
                <span className="text-xs text-muted-foreground">Not configured</span>
              </>
            )}
          </div>
        </div>
      </div>

      {webhookStatus?.gmail?.last_notification && (
        <p className="text-xs text-muted-foreground">
          Last Gmail notification: {formatDistanceToNow(new Date(webhookStatus.gmail.last_notification), { addSuffix: true })}
        </p>
      )}
      {webhookStatus?.whatsapp?.last_webhook && (
        <p className="text-xs text-muted-foreground">
          Last WhatsApp webhook: {formatDistanceToNow(new Date(webhookStatus.whatsapp.last_webhook), { addSuffix: true })}
        </p>
      )}
    </div>
  );
}

export function AutomationStatusPanel() {
  const [isExpanded, setIsExpanded] = useState(true);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [activeTab, setActiveTab] = useState("overview");
  const { toast } = useToast();

  const { data: status, isLoading: statusLoading, refetch: refetchStatus } = useQuery<AutomationStatus>({
    queryKey: ["/api/v1/automation/status"],
    refetchInterval: 30000,
  });

  const { data: stats, isLoading: statsLoading } = useQuery<DashboardStats>({
    queryKey: ["/api/v1/automation/dashboard/stats"],
    refetchInterval: 60000,
  });

  const toggleEngineMutation = useMutation({
    mutationFn: async (action: 'start' | 'pause') => {
      const response = await apiRequest(`/api/v1/automation/engine/${action}`, {
        method: "POST",
      });
      return response.json();
    },
    onSuccess: (_, action) => {
      queryClient.invalidateQueries({ queryKey: ["/api/v1/automation/status"] });
      toast({
        title: action === 'start' ? "Engine Started" : "Engine Paused",
        description: action === 'start' 
          ? "Lysa automation engine is now running." 
          : "Lysa automation engine has been paused.",
      });
    },
    onError: () => {
      toast({
        title: "Error",
        description: "Failed to toggle engine state. Please try again.",
        variant: "destructive",
      });
    },
  });

  const triggerSyncMutation = useMutation({
    mutationFn: async (channel: 'email' | 'whatsapp' | 'calendar') => {
      const response = await apiRequest(`/api/v1/automation/sync/${channel}`, {
        method: "POST",
      });
      return response.json();
    },
    onSuccess: (_, channel) => {
      queryClient.invalidateQueries({ queryKey: ["/api/v1/automation/status"] });
      toast({
        title: "Sync Triggered",
        description: `${channel.charAt(0).toUpperCase() + channel.slice(1)} sync has been initiated.`,
      });
    },
    onError: () => {
      toast({
        title: "Sync Failed",
        description: "Failed to trigger sync. Please try again.",
        variant: "destructive",
      });
    },
  });

  const handleRefresh = async () => {
    setIsRefreshing(true);
    await refetchStatus();
    await queryClient.invalidateQueries({ queryKey: ["/api/v1/automation/dashboard/stats"] });
    setTimeout(() => setIsRefreshing(false), 500);
  };

  const getStatusColor = (syncStatus: string) => {
    switch (syncStatus) {
      case "active": return "text-green-600";
      case "disabled": return "text-muted-foreground";
      case "error": return "text-destructive";
      default: return "text-muted-foreground";
    }
  };

  const getStatusIcon = (syncStatus: string) => {
    switch (syncStatus) {
      case "active": return <CheckCircle2 className="h-4 w-4 text-green-600" />;
      case "disabled": return <Clock className="h-4 w-4 text-muted-foreground" />;
      case "error": return <XCircle className="h-4 w-4 text-destructive" />;
      default: return <Clock className="h-4 w-4 text-muted-foreground" />;
    }
  };

  const successRate = stats?.success_rate ?? 0;

  return (
    <Card data-testid="card-automation-status">
      <Collapsible open={isExpanded} onOpenChange={setIsExpanded}>
        <CardHeader className="pb-2">
          <div className="flex items-center justify-between gap-2">
            <div className="flex items-center gap-2">
              <Bot className="h-5 w-5 text-primary" />
              <div>
                <CardTitle className="text-base flex items-center gap-2">
                  Lysa Automation
                  <div className={`h-2 w-2 rounded-full ${status?.is_running ? 'bg-green-500 animate-pulse' : 'bg-muted'}`} />
                </CardTitle>
                <CardDescription>
                  {status?.is_running ? (
                    <span className="text-green-600 font-medium">Active</span>
                  ) : (
                    <span className="text-muted-foreground">Paused</span>
                  )}
                  {status?.jobs_in_queue ? ` · ${status.jobs_in_queue} jobs queued` : null}
                </CardDescription>
              </div>
            </div>
            <div className="flex items-center gap-1">
              <Button 
                variant="ghost" 
                size="icon"
                onClick={() => toggleEngineMutation.mutate(status?.is_running ? 'pause' : 'start')}
                disabled={toggleEngineMutation.isPending}
                data-testid="button-toggle-engine"
              >
                {toggleEngineMutation.isPending ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : status?.is_running ? (
                  <Pause className="h-4 w-4" />
                ) : (
                  <Play className="h-4 w-4" />
                )}
              </Button>
              <Button 
                variant="ghost" 
                size="icon"
                onClick={handleRefresh}
                disabled={isRefreshing}
                data-testid="button-refresh-automation"
              >
                <RefreshCw className={`h-4 w-4 ${isRefreshing ? 'animate-spin' : ''}`} />
              </Button>
              <CollapsibleTrigger asChild>
                <Button variant="ghost" size="icon" data-testid="button-toggle-automation-panel">
                  <ChevronDown className={`h-4 w-4 transition-transform ${isExpanded ? 'rotate-180' : ''}`} />
                </Button>
              </CollapsibleTrigger>
            </div>
          </div>
        </CardHeader>

        <CollapsibleContent>
          <CardContent className="space-y-4">
            {(statusLoading || statsLoading) ? (
              <div className="flex items-center justify-center py-8">
                <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
              </div>
            ) : (
              <Tabs value={activeTab} onValueChange={setActiveTab}>
                <TabsList className="grid w-full grid-cols-3" data-testid="tabs-automation">
                  <TabsTrigger value="overview" data-testid="tab-overview">Overview</TabsTrigger>
                  <TabsTrigger value="channels" data-testid="tab-channels">Channels</TabsTrigger>
                  <TabsTrigger value="activity" data-testid="tab-activity">Activity</TabsTrigger>
                </TabsList>

                <TabsContent value="overview" className="space-y-4">
                  <div className="grid grid-cols-2 gap-3">
                    <div className="p-3 rounded-lg bg-muted/50">
                      <div className="flex items-center justify-between mb-1">
                        <span className="text-xs text-muted-foreground">Today's Jobs</span>
                        <TrendingUp className="h-3 w-3 text-muted-foreground" />
                      </div>
                      <div className="flex items-baseline gap-2">
                        <span className="text-2xl font-bold" data-testid="text-jobs-today">
                          {stats?.completed_jobs_today ?? 0}
                        </span>
                        <span className="text-xs text-muted-foreground">
                          / {stats?.total_jobs_today ?? 0} total
                        </span>
                      </div>
                      {(stats?.total_jobs_today ?? 0) > 0 && (
                        <Progress 
                          value={successRate} 
                          className="h-1 mt-2" 
                          data-testid="progress-success-rate"
                        />
                      )}
                    </div>

                    <div className="p-3 rounded-lg bg-muted/50">
                      <div className="flex items-center justify-between mb-1">
                        <span className="text-xs text-muted-foreground">Success Rate</span>
                        <Zap className="h-3 w-3 text-muted-foreground" />
                      </div>
                      <div className="flex items-baseline gap-2">
                        <span className="text-2xl font-bold" data-testid="text-success-rate">
                          {successRate.toFixed(0)}%
                        </span>
                      </div>
                      <div className="flex gap-2 mt-1 flex-wrap">
                        {(stats?.failed_jobs_today ?? 0) > 0 && (
                          <Badge variant="destructive" className="text-xs">
                            {stats?.failed_jobs_today} failed
                          </Badge>
                        )}
                        {(stats?.pending_jobs ?? 0) > 0 && (
                          <Badge variant="secondary" className="text-xs">
                            {stats?.pending_jobs} pending
                          </Badge>
                        )}
                      </div>
                    </div>
                  </div>

                  <div className="grid grid-cols-3 gap-2">
                    <div className="p-2 rounded-md bg-blue-500/10 text-center">
                      <div className="text-lg font-bold text-blue-600" data-testid="text-emails-processed">
                        {stats?.emails_synced ?? 0}
                      </div>
                      <div className="text-xs text-muted-foreground">Emails</div>
                    </div>
                    <div className="p-2 rounded-md bg-green-500/10 text-center">
                      <div className="text-lg font-bold text-green-600" data-testid="text-messages-sent">
                        {stats?.whatsapp_messages_sent ?? 0}
                      </div>
                      <div className="text-xs text-muted-foreground">Messages</div>
                    </div>
                    <div className="p-2 rounded-md bg-purple-500/10 text-center">
                      <div className="text-lg font-bold text-purple-600" data-testid="text-appointments-booked">
                        {stats?.appointments_booked ?? 0}
                      </div>
                      <div className="text-xs text-muted-foreground">Appts</div>
                    </div>
                  </div>

                  <div className="grid grid-cols-3 gap-2">
                    <div className="p-2 rounded-md bg-amber-500/10 text-center">
                      <div className="text-lg font-bold text-amber-600" data-testid="text-reminders-sent">
                        {(stats?.reminders_medication ?? 0) + (stats?.reminders_appointment ?? 0) + (stats?.reminders_followup ?? 0)}
                      </div>
                      <div className="text-xs text-muted-foreground">Reminders</div>
                    </div>
                    <div className="p-2 rounded-md bg-pink-500/10 text-center">
                      <div className="text-lg font-bold text-pink-600" data-testid="text-clinical-docs">
                        {(stats?.clinical_soap_notes ?? 0) + (stats?.clinical_diagnoses ?? 0)}
                      </div>
                      <div className="text-xs text-muted-foreground">Clinical</div>
                    </div>
                    <div className="p-2 rounded-md bg-cyan-500/10 text-center">
                      <div className="text-lg font-bold text-cyan-600" data-testid="text-auto-replies">
                        {(stats?.emails_auto_replied ?? 0) + (stats?.whatsapp_auto_replies ?? 0)}
                      </div>
                      <div className="text-xs text-muted-foreground">Auto-Reply</div>
                    </div>
                  </div>

                  {status?.current_job && (
                    <div className="p-3 rounded-md bg-primary/10 border border-primary/20" data-testid="current-job">
                      <div className="flex items-center gap-2 mb-1">
                        <Loader2 className="h-4 w-4 animate-spin text-primary" />
                        <span className="text-sm font-medium">Running Job</span>
                      </div>
                      <p className="text-xs text-muted-foreground">
                        {status.current_job.job_type} - {status.current_job.id.slice(0, 8)}
                      </p>
                    </div>
                  )}
                </TabsContent>

                <TabsContent value="channels" className="space-y-4">
                  <div className="space-y-2">
                    <div className="flex items-center justify-between p-3 rounded-md border" data-testid="channel-email">
                      <div className="flex items-center gap-3">
                        <div className="p-2 rounded-md bg-blue-500/10">
                          <Mail className="h-5 w-5 text-blue-600" />
                        </div>
                        <div>
                          <div className="font-medium text-sm">Email</div>
                          <div className="flex items-center gap-2 text-xs text-muted-foreground">
                            {getStatusIcon(status?.email_sync_status ?? "disabled")}
                            <span className={getStatusColor(status?.email_sync_status ?? "disabled")}>
                              {status?.email_sync_status === "active" ? "Active" : "Disabled"}
                            </span>
                            {status?.email_last_sync && (
                              <span>· {formatDistanceToNow(new Date(status.email_last_sync), { addSuffix: true })}</span>
                            )}
                          </div>
                        </div>
                      </div>
                      <div className="flex items-center gap-2">
                        <Button 
                          variant="outline" 
                          size="sm"
                          onClick={() => triggerSyncMutation.mutate('email')}
                          disabled={triggerSyncMutation.isPending}
                          data-testid="button-sync-email"
                        >
                          {triggerSyncMutation.isPending ? <Loader2 className="h-3 w-3 animate-spin" /> : <RefreshCw className="h-3 w-3" />}
                        </Button>
                        <ChannelConfigDialog channel="email">
                          <Button variant="outline" size="sm" data-testid="button-config-email">
                            <Settings className="h-3 w-3" />
                          </Button>
                        </ChannelConfigDialog>
                      </div>
                    </div>

                    <div className="flex items-center justify-between p-3 rounded-md border" data-testid="channel-whatsapp">
                      <div className="flex items-center gap-3">
                        <div className="p-2 rounded-md bg-green-500/10">
                          <MessageSquare className="h-5 w-5 text-green-600" />
                        </div>
                        <div>
                          <div className="font-medium text-sm">WhatsApp</div>
                          <div className="flex items-center gap-2 text-xs text-muted-foreground">
                            {getStatusIcon(status?.whatsapp_sync_status ?? "disabled")}
                            <span className={getStatusColor(status?.whatsapp_sync_status ?? "disabled")}>
                              {status?.whatsapp_sync_status === "active" ? "Active" : "Disabled"}
                            </span>
                            {status?.whatsapp_last_sync && (
                              <span>· {formatDistanceToNow(new Date(status.whatsapp_last_sync), { addSuffix: true })}</span>
                            )}
                          </div>
                        </div>
                      </div>
                      <div className="flex items-center gap-2">
                        <Button 
                          variant="outline" 
                          size="sm"
                          onClick={() => triggerSyncMutation.mutate('whatsapp')}
                          disabled={triggerSyncMutation.isPending}
                          data-testid="button-sync-whatsapp"
                        >
                          {triggerSyncMutation.isPending ? <Loader2 className="h-3 w-3 animate-spin" /> : <RefreshCw className="h-3 w-3" />}
                        </Button>
                        <ChannelConfigDialog channel="whatsapp">
                          <Button variant="outline" size="sm" data-testid="button-config-whatsapp">
                            <Settings className="h-3 w-3" />
                          </Button>
                        </ChannelConfigDialog>
                      </div>
                    </div>

                    <div className="flex items-center justify-between p-3 rounded-md border" data-testid="channel-calendar">
                      <div className="flex items-center gap-3">
                        <div className="p-2 rounded-md bg-purple-500/10">
                          <Calendar className="h-5 w-5 text-purple-600" />
                        </div>
                        <div>
                          <div className="font-medium text-sm">Calendar</div>
                          <div className="flex items-center gap-2 text-xs text-muted-foreground">
                            {getStatusIcon(status?.calendar_sync_status ?? "disabled")}
                            <span className={getStatusColor(status?.calendar_sync_status ?? "disabled")}>
                              {status?.calendar_sync_status === "active" ? "Active" : "Disabled"}
                            </span>
                            {status?.calendar_last_sync && (
                              <span>· {formatDistanceToNow(new Date(status.calendar_last_sync), { addSuffix: true })}</span>
                            )}
                          </div>
                        </div>
                      </div>
                      <div className="flex items-center gap-2">
                        <Button 
                          variant="outline" 
                          size="sm"
                          onClick={() => triggerSyncMutation.mutate('calendar')}
                          disabled={triggerSyncMutation.isPending}
                          data-testid="button-sync-calendar"
                        >
                          {triggerSyncMutation.isPending ? <Loader2 className="h-3 w-3 animate-spin" /> : <RefreshCw className="h-3 w-3" />}
                        </Button>
                        <ChannelConfigDialog channel="calendar">
                          <Button variant="outline" size="sm" data-testid="button-config-calendar">
                            <Settings className="h-3 w-3" />
                          </Button>
                        </ChannelConfigDialog>
                      </div>
                    </div>
                  </div>

                  <Separator />
                  <WebhookStatusPanel />
                </TabsContent>

                <TabsContent value="activity" className="space-y-4">
                  {status?.recent_activities && status.recent_activities.length > 0 ? (
                    <div className="space-y-2 max-h-64 overflow-y-auto">
                      {status.recent_activities.map((activity) => (
                        <div 
                          key={activity.id}
                          className="flex items-start gap-3 p-2 rounded-md bg-muted/30"
                          data-testid={`activity-${activity.id}`}
                        >
                          <Activity className={`h-4 w-4 mt-0.5 ${
                            activity.level === 'error' ? 'text-destructive' :
                            activity.level === 'warning' ? 'text-amber-600' :
                            'text-green-600'
                          }`} />
                          <div className="flex-1 min-w-0">
                            <p className="text-sm">{activity.message}</p>
                            <div className="flex items-center gap-2 mt-1">
                              {activity.action_type && (
                                <Badge variant="outline" className="text-xs">
                                  {activity.action_type}
                                </Badge>
                              )}
                              <span className="text-xs text-muted-foreground">
                                {formatDistanceToNow(new Date(activity.timestamp), { addSuffix: true })}
                              </span>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="flex flex-col items-center justify-center py-8 text-muted-foreground">
                      <Activity className="h-8 w-8 mb-2" />
                      <p className="text-sm">No recent activity</p>
                    </div>
                  )}
                </TabsContent>
              </Tabs>
            )}
          </CardContent>
        </CollapsibleContent>
      </Collapsible>
    </Card>
  );
}

export function AutomationStatusBadge() {
  const { data: status } = useQuery<AutomationStatus>({
    queryKey: ["/api/v1/automation/status"],
    refetchInterval: 60000,
  });

  if (!status) return null;

  return (
    <div className="flex items-center gap-2" data-testid="automation-status-badge">
      <div className={`h-2 w-2 rounded-full ${status.is_running ? 'bg-green-500 animate-pulse' : 'bg-muted'}`} />
      <span className="text-xs text-muted-foreground">
        {status.is_running ? "Lysa Active" : "Lysa Paused"}
      </span>
      {status.jobs_in_queue > 0 && (
        <Badge variant="secondary" className="text-xs">
          {status.jobs_in_queue} queued
        </Badge>
      )}
    </div>
  );
}
