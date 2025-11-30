import { useState, useEffect } from "react";
import { useQuery } from "@tanstack/react-query";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { 
  Activity, 
  Mail, 
  MessageSquare, 
  Calendar, 
  Bell, 
  Stethoscope,
  CheckCircle2,
  XCircle,
  Clock,
  Loader2,
  RefreshCw,
  ChevronDown,
  Zap,
  TrendingUp
} from "lucide-react";
import { formatDistanceToNow } from "date-fns";
import { apiRequest, queryClient } from "@/lib/queryClient";

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

export function AutomationStatusPanel() {
  const [isExpanded, setIsExpanded] = useState(true);
  const [isRefreshing, setIsRefreshing] = useState(false);

  const { data: status, isLoading: statusLoading, refetch: refetchStatus } = useQuery<AutomationStatus>({
    queryKey: ["/api/py/v1/automation/status"],
    refetchInterval: 30000,
  });

  const { data: stats, isLoading: statsLoading } = useQuery<DashboardStats>({
    queryKey: ["/api/py/v1/automation/dashboard/stats"],
    refetchInterval: 60000,
  });

  const handleRefresh = async () => {
    setIsRefreshing(true);
    await refetchStatus();
    await queryClient.invalidateQueries({ queryKey: ["/api/py/v1/automation/dashboard/stats"] });
    setTimeout(() => setIsRefreshing(false), 500);
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case "active": return "text-green-600";
      case "disabled": return "text-muted-foreground";
      case "error": return "text-destructive";
      default: return "text-muted-foreground";
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
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
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <div className={`h-2 w-2 rounded-full ${status?.is_running ? 'bg-green-500 animate-pulse' : 'bg-muted'}`} />
              <CardTitle className="text-base">Automation Engine</CardTitle>
            </div>
            <div className="flex items-center gap-2">
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
          <CardDescription className="flex items-center gap-2">
            {status?.is_running ? (
              <>
                <span className="text-green-600 font-medium">Active</span>
                <span>·</span>
                <span>{status?.jobs_in_queue ?? 0} jobs in queue</span>
              </>
            ) : (
              <span className="text-muted-foreground">Engine paused</span>
            )}
          </CardDescription>
        </CardHeader>

        <CollapsibleContent>
          <CardContent className="space-y-4">
            {(statusLoading || statsLoading) ? (
              <div className="flex items-center justify-center py-8">
                <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
              </div>
            ) : (
              <>
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
                    <div className="flex gap-2 mt-1">
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

                <div className="space-y-2">
                  <h4 className="text-sm font-medium text-muted-foreground">Sync Status</h4>
                  
                  <div className="flex items-center justify-between p-2 rounded-md border" data-testid="status-email-sync">
                    <div className="flex items-center gap-2">
                      <Mail className="h-4 w-4 text-blue-600" />
                      <span className="text-sm">Email Sync</span>
                    </div>
                    <div className="flex items-center gap-2">
                      {getStatusIcon(status?.email_sync_status ?? "disabled")}
                      <span className={`text-sm ${getStatusColor(status?.email_sync_status ?? "disabled")}`}>
                        {status?.email_sync_status === "active" ? "Active" : "Disabled"}
                      </span>
                      {status?.email_last_sync && (
                        <span className="text-xs text-muted-foreground">
                          · {formatDistanceToNow(new Date(status.email_last_sync), { addSuffix: true })}
                        </span>
                      )}
                    </div>
                  </div>

                  <div className="flex items-center justify-between p-2 rounded-md border" data-testid="status-whatsapp-sync">
                    <div className="flex items-center gap-2">
                      <MessageSquare className="h-4 w-4 text-green-600" />
                      <span className="text-sm">WhatsApp</span>
                    </div>
                    <div className="flex items-center gap-2">
                      {getStatusIcon(status?.whatsapp_sync_status ?? "disabled")}
                      <span className={`text-sm ${getStatusColor(status?.whatsapp_sync_status ?? "disabled")}`}>
                        {status?.whatsapp_sync_status === "active" ? "Active" : "Disabled"}
                      </span>
                    </div>
                  </div>

                  <div className="flex items-center justify-between p-2 rounded-md border" data-testid="status-calendar-sync">
                    <div className="flex items-center gap-2">
                      <Calendar className="h-4 w-4 text-purple-600" />
                      <span className="text-sm">Calendar</span>
                    </div>
                    <div className="flex items-center gap-2">
                      {getStatusIcon(status?.calendar_sync_status ?? "disabled")}
                      <span className={`text-sm ${getStatusColor(status?.calendar_sync_status ?? "disabled")}`}>
                        {status?.calendar_sync_status === "active" ? "Active" : "Disabled"}
                      </span>
                    </div>
                  </div>
                </div>

                <div className="space-y-2">
                  <h4 className="text-sm font-medium text-muted-foreground">Today's Activity</h4>
                  
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
                </div>

                {status?.recent_activities && status.recent_activities.length > 0 && (
                  <div className="space-y-2">
                    <h4 className="text-sm font-medium text-muted-foreground">Recent Activity</h4>
                    <div className="space-y-1 max-h-32 overflow-y-auto">
                      {status.recent_activities.slice(0, 5).map((activity) => (
                        <div 
                          key={activity.id}
                          className="flex items-center gap-2 text-xs p-1.5 rounded bg-muted/30"
                          data-testid={`activity-${activity.id}`}
                        >
                          <Activity className={`h-3 w-3 ${
                            activity.level === 'error' ? 'text-destructive' :
                            activity.level === 'warning' ? 'text-amber-600' :
                            'text-green-600'
                          }`} />
                          <span className="flex-1 truncate">{activity.message}</span>
                          <span className="text-muted-foreground">
                            {formatDistanceToNow(new Date(activity.timestamp), { addSuffix: true })}
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

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
              </>
            )}
          </CardContent>
        </CollapsibleContent>
      </Collapsible>
    </Card>
  );
}

export function AutomationStatusBadge() {
  const { data: status } = useQuery<AutomationStatus>({
    queryKey: ["/api/py/v1/automation/status"],
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
