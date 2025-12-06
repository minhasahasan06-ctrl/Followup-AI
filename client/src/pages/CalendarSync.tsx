import { useState } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Switch } from "@/components/ui/switch";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Separator } from "@/components/ui/separator";
import { Skeleton } from "@/components/ui/skeleton";
import { 
  Calendar, 
  Check, 
  X, 
  RefreshCw, 
  Settings, 
  Clock, 
  ArrowLeftRight,
  ExternalLink,
  AlertTriangle,
  CheckCircle2,
  XCircle,
  CalendarDays,
  Loader2,
  Link2,
  Link2Off,
  History
} from "lucide-react";
import { SiGoogle } from "react-icons/si";
import { format } from "date-fns";
import { useToast } from "@/hooks/use-toast";
import { apiRequest, queryClient } from "@/lib/queryClient";

interface ConnectorStatus {
  connected: boolean;
  calendarId?: string;
  calendarName?: string;
  email?: string;
  source: string;
}

interface SyncConfig {
  id?: string;
  syncEnabled?: boolean;
  syncDirection?: string;
  calendarId?: string;
  calendarName?: string;
  lastSyncAt?: string;
  lastSyncStatus?: string;
  lastSyncError?: string;
  totalEventsSynced?: number;
  conflictResolution?: string;
}

interface CalendarEvent {
  id: string;
  summary: string;
  description?: string;
  location?: string;
  start: { dateTime?: string; date?: string };
  end: { dateTime?: string; date?: string };
  status: string;
  hangoutLink?: string;
  attendees?: Array<{
    email: string;
    responseStatus: string;
    displayName?: string;
  }>;
}

interface SyncLog {
  id: string;
  syncType: string;
  syncDirection: string;
  status: string;
  eventsCreated: number;
  eventsUpdated: number;
  eventsDeleted: number;
  conflictsDetected: number;
  durationMs: number;
  error?: string;
  createdAt: string;
}

export default function CalendarSync() {
  const { toast } = useToast();
  const [selectedCalendar, setSelectedCalendar] = useState<string>("primary");

  const { data: connectorStatus, isLoading: connectorLoading, refetch: refetchConnector } = useQuery<ConnectorStatus>({
    queryKey: ["/api/v1/calendar/connector-status"],
    refetchInterval: 30000,
  });

  const { data: syncConfig, isLoading: syncConfigLoading } = useQuery<SyncConfig>({
    queryKey: ["/api/v1/calendar/sync/status"],
  });

  const { data: eventsData, isLoading: eventsLoading, refetch: refetchEvents } = useQuery<{ events: CalendarEvent[] }>({
    queryKey: ["/api/v1/calendar/events", selectedCalendar],
    enabled: connectorStatus?.connected === true,
  });

  const { data: syncLogs = [], isLoading: logsLoading } = useQuery<SyncLog[]>({
    queryKey: ["/api/v1/calendar/sync/logs"],
  });

  const syncMutation = useMutation({
    mutationFn: async () => {
      return apiRequest("/api/v1/calendar/sync-appointments", {
        method: "POST",
        body: JSON.stringify({ calendarId: selectedCalendar })
      });
    },
    onSuccess: (data: any) => {
      toast({
        title: "Sync Complete",
        description: `Synced ${data.synced} appointments. ${data.failed > 0 ? `${data.failed} failed.` : ''}`,
      });
      queryClient.invalidateQueries({ queryKey: ["/api/v1/calendar/events"] });
      queryClient.invalidateQueries({ queryKey: ["/api/v1/calendar/sync/logs"] });
    },
    onError: () => {
      toast({
        title: "Sync Failed",
        description: "Failed to sync appointments to Google Calendar",
        variant: "destructive",
      });
    }
  });

  const updateSettingsMutation = useMutation({
    mutationFn: async (settings: Partial<SyncConfig>) => {
      return apiRequest("/api/v1/calendar/sync/settings", {
        method: "PATCH",
        body: JSON.stringify(settings)
      });
    },
    onSuccess: () => {
      toast({
        title: "Settings Updated",
        description: "Calendar sync settings have been updated",
      });
      queryClient.invalidateQueries({ queryKey: ["/api/v1/calendar/sync/status"] });
    },
    onError: () => {
      toast({
        title: "Error",
        description: "Failed to update settings",
        variant: "destructive",
      });
    }
  });

  const events = eventsData?.events || [];
  const isConnected = connectorStatus?.connected === true;

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between flex-wrap gap-4">
        <div>
          <h1 className="text-3xl font-semibold" data-testid="text-calendar-sync-title">
            Google Calendar Sync
          </h1>
          <p className="text-muted-foreground">
            Connect your Google Calendar to sync appointments bidirectionally
          </p>
        </div>
        <div className="flex items-center gap-2 flex-wrap">
          {isConnected && (
            <Button 
              variant="outline" 
              onClick={() => refetchEvents()}
              data-testid="button-refresh-calendar"
            >
              <RefreshCw className="h-4 w-4 mr-2" />
              Refresh
            </Button>
          )}
        </div>
      </div>

      <div className="grid gap-6 lg:grid-cols-3">
        <Card className="lg:col-span-2" data-testid="card-connection-status">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <SiGoogle className="h-5 w-5" />
              Connection Status
            </CardTitle>
            <CardDescription>
              Connect your Google Calendar account to enable appointment syncing
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {connectorLoading ? (
              <div className="flex items-center gap-4">
                <Skeleton className="h-12 w-12 rounded-full" />
                <div className="space-y-2">
                  <Skeleton className="h-4 w-48" />
                  <Skeleton className="h-3 w-32" />
                </div>
              </div>
            ) : isConnected ? (
              <div className="space-y-4">
                <div className="flex items-center gap-4 p-4 rounded-lg bg-accent/50">
                  <div className="flex h-12 w-12 items-center justify-center rounded-full bg-green-500/20">
                    <CheckCircle2 className="h-6 w-6 text-green-600" />
                  </div>
                  <div className="flex-1">
                    <p className="font-medium">Connected to Google Calendar</p>
                    <p className="text-sm text-muted-foreground">
                      {connectorStatus?.email || connectorStatus?.calendarName || "Primary Calendar"}
                    </p>
                  </div>
                  <Badge variant="secondary">
                    <Check className="h-3 w-3 mr-1" />
                    Active
                  </Badge>
                </div>

                <Separator />

                <div className="grid gap-4 sm:grid-cols-2">
                  <div className="space-y-2">
                    <label className="text-sm font-medium">Sync Direction</label>
                    <Select 
                      value={syncConfig?.syncDirection || "bidirectional"}
                      onValueChange={(value) => updateSettingsMutation.mutate({ syncDirection: value })}
                    >
                      <SelectTrigger data-testid="select-sync-direction">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="bidirectional" data-testid="option-bidirectional">
                          <span className="flex items-center gap-2">
                            <ArrowLeftRight className="h-4 w-4" />
                            Bidirectional
                          </span>
                        </SelectItem>
                        <SelectItem value="to_google" data-testid="option-to-google">
                          <span className="flex items-center gap-2">
                            <ExternalLink className="h-4 w-4" />
                            To Google Only
                          </span>
                        </SelectItem>
                        <SelectItem value="from_google" data-testid="option-from-google">
                          <span className="flex items-center gap-2">
                            <Calendar className="h-4 w-4" />
                            From Google Only
                          </span>
                        </SelectItem>
                      </SelectContent>
                    </Select>
                  </div>

                  <div className="space-y-2">
                    <label className="text-sm font-medium">Conflict Resolution</label>
                    <Select 
                      value={syncConfig?.conflictResolution || "google_wins"}
                      onValueChange={(value) => updateSettingsMutation.mutate({ conflictResolution: value })}
                    >
                      <SelectTrigger data-testid="select-conflict-resolution">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="google_wins" data-testid="option-google-wins">Google Calendar wins</SelectItem>
                        <SelectItem value="local_wins" data-testid="option-local-wins">Local appointments win</SelectItem>
                        <SelectItem value="newest_wins" data-testid="option-newest-wins">Most recent wins</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </div>

                <div className="flex items-center justify-between p-4 rounded-lg border">
                  <div className="space-y-0.5">
                    <div className="font-medium">Auto-sync enabled</div>
                    <div className="text-sm text-muted-foreground">
                      Automatically sync new appointments
                    </div>
                  </div>
                  <Switch 
                    checked={syncConfig?.syncEnabled || false}
                    onCheckedChange={(checked) => updateSettingsMutation.mutate({ syncEnabled: checked })}
                    data-testid="switch-auto-sync"
                  />
                </div>

                <div className="flex items-center gap-2 flex-wrap">
                  <Button 
                    onClick={() => syncMutation.mutate()}
                    disabled={syncMutation.isPending}
                    data-testid="button-sync-now"
                  >
                    {syncMutation.isPending ? (
                      <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                    ) : (
                      <RefreshCw className="h-4 w-4 mr-2" />
                    )}
                    Sync Now
                  </Button>
                </div>

                {syncConfig?.lastSyncAt && (
                  <div className="flex items-center gap-2 text-sm text-muted-foreground">
                    <Clock className="h-4 w-4" />
                    Last synced: {format(new Date(syncConfig.lastSyncAt), "MMM d, yyyy h:mm a")}
                    {syncConfig.lastSyncStatus === "success" && (
                      <Badge variant="outline" className="ml-2">
                        <Check className="h-3 w-3 mr-1" />
                        Success
                      </Badge>
                    )}
                    {syncConfig.lastSyncStatus === "failed" && (
                      <Badge variant="destructive" className="ml-2">
                        <X className="h-3 w-3 mr-1" />
                        Failed
                      </Badge>
                    )}
                  </div>
                )}

                {syncConfig?.lastSyncError && (
                  <Alert variant="destructive">
                    <AlertTriangle className="h-4 w-4" />
                    <AlertTitle>Last sync error</AlertTitle>
                    <AlertDescription>{syncConfig.lastSyncError}</AlertDescription>
                  </Alert>
                )}
              </div>
            ) : (
              <div className="space-y-4">
                <Alert>
                  <Link2Off className="h-4 w-4" />
                  <AlertTitle>Not Connected</AlertTitle>
                  <AlertDescription>
                    Your Google Calendar is not connected yet. Please connect your Google account using the Replit integrations panel to enable calendar sync.
                  </AlertDescription>
                </Alert>

                <div className="p-6 rounded-lg border border-dashed flex flex-col items-center justify-center gap-4 text-center">
                  <div className="flex h-16 w-16 items-center justify-center rounded-full bg-muted">
                    <SiGoogle className="h-8 w-8 text-muted-foreground" />
                  </div>
                  <div>
                    <h3 className="font-medium">Connect Google Calendar</h3>
                    <p className="text-sm text-muted-foreground mt-1">
                      Sync your appointments with your personal Google Calendar
                    </p>
                  </div>
                  <Button 
                    variant="outline" 
                    onClick={() => refetchConnector()}
                    data-testid="button-check-connection"
                  >
                    <RefreshCw className="h-4 w-4 mr-2" />
                    Check Connection Status
                  </Button>
                </div>
              </div>
            )}
          </CardContent>
        </Card>

        <Card data-testid="card-sync-stats">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Settings className="h-5 w-5" />
              Sync Statistics
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div className="text-center p-4 rounded-lg bg-accent/50">
                <div className="text-2xl font-bold" data-testid="text-events-synced">
                  {syncConfig?.totalEventsSynced || 0}
                </div>
                <div className="text-xs text-muted-foreground">Events Synced</div>
              </div>
              <div className="text-center p-4 rounded-lg bg-accent/50">
                <div className="text-2xl font-bold" data-testid="text-upcoming-events">
                  {events.length}
                </div>
                <div className="text-xs text-muted-foreground">Upcoming Events</div>
              </div>
            </div>

            {isConnected && (
              <div className="space-y-2">
                <div className="flex items-center justify-between text-sm">
                  <span className="text-muted-foreground">Connection</span>
                  <Badge variant="secondary">
                    <Link2 className="h-3 w-3 mr-1" />
                    Active
                  </Badge>
                </div>
                <div className="flex items-center justify-between text-sm">
                  <span className="text-muted-foreground">Sync Direction</span>
                  <span className="font-medium">
                    {syncConfig?.syncDirection === "bidirectional" ? "↔ Both Ways" : 
                     syncConfig?.syncDirection === "to_google" ? "→ To Google" : "← From Google"}
                  </span>
                </div>
                <div className="flex items-center justify-between text-sm">
                  <span className="text-muted-foreground">Auto-sync</span>
                  <span className="font-medium">
                    {syncConfig?.syncEnabled ? "Enabled" : "Disabled"}
                  </span>
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {isConnected && (
        <div className="grid gap-6 lg:grid-cols-2">
          <Card data-testid="card-upcoming-events">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <CalendarDays className="h-5 w-5" />
                Upcoming Events
              </CardTitle>
              <CardDescription>
                Events from your connected Google Calendar (next 7 days)
              </CardDescription>
            </CardHeader>
            <CardContent>
              {eventsLoading ? (
                <div className="space-y-4">
                  {[1, 2, 3].map((i) => (
                    <div key={i} className="flex items-start gap-4">
                      <Skeleton className="h-10 w-10 rounded" />
                      <div className="space-y-2 flex-1">
                        <Skeleton className="h-4 w-3/4" />
                        <Skeleton className="h-3 w-1/2" />
                      </div>
                    </div>
                  ))}
                </div>
              ) : events.length === 0 ? (
                <div className="text-center py-8 text-muted-foreground">
                  <Calendar className="h-12 w-12 mx-auto mb-4 opacity-50" />
                  <p>No upcoming events in the next 7 days</p>
                </div>
              ) : (
                <div className="space-y-3 max-h-96 overflow-y-auto">
                  {events.slice(0, 10).map((event) => (
                    <div 
                      key={event.id} 
                      className="flex items-start gap-3 p-3 rounded-lg border hover-elevate"
                      data-testid={`event-${event.id}`}
                    >
                      <div className="flex h-10 w-10 items-center justify-center rounded bg-primary/10 flex-shrink-0">
                        <Calendar className="h-5 w-5 text-primary" />
                      </div>
                      <div className="flex-1 min-w-0">
                        <p className="font-medium truncate">{event.summary || "Untitled Event"}</p>
                        <div className="flex items-center gap-2 text-sm text-muted-foreground mt-1">
                          <Clock className="h-3 w-3" />
                          {event.start?.dateTime ? 
                            format(new Date(event.start.dateTime), "MMM d, h:mm a") : 
                            event.start?.date || "No date"
                          }
                        </div>
                        {event.location && (
                          <p className="text-xs text-muted-foreground truncate mt-1">
                            📍 {event.location}
                          </p>
                        )}
                      </div>
                      {event.hangoutLink && (
                        <a 
                          href={event.hangoutLink} 
                          target="_blank" 
                          rel="noopener noreferrer"
                          className="text-primary"
                          data-testid={`link-event-meet-${event.id}`}
                        >
                          <ExternalLink className="h-4 w-4" />
                        </a>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>

          <Card data-testid="card-sync-history">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <History className="h-5 w-5" />
                Sync History
              </CardTitle>
              <CardDescription>
                Recent synchronization activity
              </CardDescription>
            </CardHeader>
            <CardContent>
              {logsLoading ? (
                <div className="space-y-4">
                  {[1, 2, 3].map((i) => (
                    <div key={i} className="flex items-start gap-4">
                      <Skeleton className="h-8 w-8 rounded-full" />
                      <div className="space-y-2 flex-1">
                        <Skeleton className="h-4 w-1/2" />
                        <Skeleton className="h-3 w-3/4" />
                      </div>
                    </div>
                  ))}
                </div>
              ) : syncLogs.length === 0 ? (
                <div className="text-center py-8 text-muted-foreground">
                  <History className="h-12 w-12 mx-auto mb-4 opacity-50" />
                  <p>No sync history yet</p>
                  <p className="text-sm">Sync your calendar to see activity here</p>
                </div>
              ) : (
                <div className="space-y-3 max-h-96 overflow-y-auto">
                  {syncLogs.slice(0, 10).map((log) => (
                    <div 
                      key={log.id} 
                      className="flex items-start gap-3 p-3 rounded-lg border"
                      data-testid={`sync-log-${log.id}`}
                    >
                      <div className={`flex h-8 w-8 items-center justify-center rounded-full flex-shrink-0 ${
                        log.status === "success" ? "bg-green-500/20" : 
                        log.status === "failed" ? "bg-red-500/20" : "bg-yellow-500/20"
                      }`}>
                        {log.status === "success" ? (
                          <CheckCircle2 className="h-4 w-4 text-green-600" />
                        ) : log.status === "failed" ? (
                          <XCircle className="h-4 w-4 text-red-600" />
                        ) : (
                          <AlertTriangle className="h-4 w-4 text-yellow-600" />
                        )}
                      </div>
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2">
                          <p className="font-medium text-sm">
                            {log.syncType === "full" ? "Full Sync" : 
                             log.syncType === "manual" ? "Manual Sync" : "Incremental Sync"}
                          </p>
                          <Badge variant="outline" className="text-xs">
                            {log.syncDirection}
                          </Badge>
                        </div>
                        <div className="flex items-center gap-3 text-xs text-muted-foreground mt-1">
                          <span>+{log.eventsCreated} created</span>
                          <span>~{log.eventsUpdated} updated</span>
                          <span>-{log.eventsDeleted} deleted</span>
                        </div>
                        <p className="text-xs text-muted-foreground mt-1">
                          {format(new Date(log.createdAt), "MMM d, h:mm a")} • {log.durationMs}ms
                        </p>
                        {log.error && (
                          <p className="text-xs text-destructive mt-1 truncate">
                            {log.error}
                          </p>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      )}

      <Alert>
        <AlertTriangle className="h-4 w-4" />
        <AlertTitle>Important Notice</AlertTitle>
        <AlertDescription>
          Calendar sync uses the platform's Google Calendar connection. Each appointment synced will appear in your connected Google Calendar. 
          Patient information is synced in a HIPAA-compliant manner - only appointment times and titles are shared, no protected health information (PHI) is included in calendar events.
        </AlertDescription>
      </Alert>
    </div>
  );
}
