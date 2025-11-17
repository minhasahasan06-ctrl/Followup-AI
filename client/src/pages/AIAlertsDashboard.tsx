import { useState } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { useToast } from "@/hooks/use-toast";
import { apiRequest, queryClient } from "@/lib/queryClient";
import {
  Bell,
  BellOff,
  Mail,
  MessageSquare,
  CheckCircle2,
  XCircle,
  Clock,
  AlertTriangle,
  Settings,
  Trash2,
  Info
} from "lucide-react";
import { LegalDisclaimer } from "@/components/LegalDisclaimer";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";

interface AlertRule {
  rule_id: number;
  rule_name: string;
  metric_name: string;
  threshold_type: string;
  threshold_value: number;
  severity: string;
  enabled: boolean;
  notification_channels: string[];
}

interface AlertNotification {
  alert_id: number;
  patient_id: string;
  alert_type: string;
  severity: string;
  message: string;
  triggered_at: string;
  acknowledged: boolean;
  acknowledged_at?: string;
  delivery_status: {
    dashboard: boolean;
    email?: boolean;
    sms?: boolean;
  };
}

export default function AIAlertsDashboard() {
  const { toast } = useToast();
  const [selectedRule, setSelectedRule] = useState<number | null>(null);

  const { data: pendingAlerts, isLoading: loadingPending } = useQuery<AlertNotification[]>({
    queryKey: ["/api/v1/alerts/pending"],
  });

  const { data: alertHistory } = useQuery<AlertNotification[]>({
    queryKey: ["/api/v1/alerts/history/me"],
  });

  const { data: alertRules, isLoading: loadingRules } = useQuery<AlertRule[]>({
    queryKey: ["/api/v1/alerts/rules/me"],
  });

  const acknowledgeMutation = useMutation({
    mutationFn: async (alertId: number) => {
      return apiRequest(`/api/v1/alerts/${alertId}/acknowledge`, {
        method: "POST",
      });
    },
    onSuccess: () => {
      toast({
        title: "Alert Acknowledged",
        description: "The alert has been marked as reviewed.",
      });
      queryClient.invalidateQueries({ queryKey: ["/api/v1/alerts/pending"] });
      queryClient.invalidateQueries({ queryKey: ["/api/v1/alerts/history/me"] });
    },
  });

  const toggleRuleMutation = useMutation({
    mutationFn: async ({ ruleId, enabled }: { ruleId: number; enabled: boolean }) => {
      return apiRequest(`/api/v1/alerts/rules/${ruleId}`, {
        method: "PATCH",
        body: JSON.stringify({ enabled }),
        headers: { "Content-Type": "application/json" },
      });
    },
    onSuccess: () => {
      toast({
        title: "Rule Updated",
        description: "Alert rule has been updated successfully.",
      });
      queryClient.invalidateQueries({ queryKey: ["/api/v1/alerts/rules/me"] });
    },
  });

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case "critical":
        return "bg-rose-50 text-rose-700 border-rose-200 dark:bg-rose-950 dark:text-rose-300";
      case "high":
        return "bg-orange-50 text-orange-700 border-orange-200 dark:bg-orange-950 dark:text-orange-300";
      case "medium":
        return "bg-yellow-50 text-yellow-700 border-yellow-200 dark:bg-yellow-950 dark:text-yellow-300";
      case "low":
        return "bg-blue-50 text-blue-700 border-blue-200 dark:bg-blue-950 dark:text-blue-300";
      default:
        return "bg-slate-50 text-slate-700 border-slate-200";
    }
  };

  const getSeverityIcon = (severity: string) => {
    switch (severity) {
      case "critical":
      case "high":
        return <AlertTriangle className="h-4 w-4" />;
      case "medium":
        return <Info className="h-4 w-4" />;
      case "low":
        return <CheckCircle2 className="h-4 w-4" />;
      default:
        return null;
    }
  };

  if (loadingPending || loadingRules) {
    return (
      <div className="container mx-auto p-6 max-w-7xl space-y-6">
        <Skeleton className="h-12 w-96" />
        <Skeleton className="h-48" />
        <Skeleton className="h-96" />
      </div>
    );
  }

  return (
    <div className="container mx-auto p-6 max-w-7xl space-y-6">
      {/* Header */}
      <div className="space-y-2">
        <h1 className="text-4xl font-bold tracking-tight text-foreground flex items-center gap-3" data-testid="text-page-title">
          <Bell className="h-10 w-10 text-primary" />
          AI Alert Management
        </h1>
        <p className="text-muted-foreground leading-relaxed">
          Real-time health change notifications and alert configuration
        </p>
      </div>

      <LegalDisclaimer />

      {/* Pending Alerts */}
      <Card data-testid="card-pending-alerts">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Bell className="h-5 w-5 text-primary" />
            Pending Alerts
            {pendingAlerts && pendingAlerts.length > 0 && (
              <Badge variant="destructive">{pendingAlerts.length}</Badge>
            )}
          </CardTitle>
          <CardDescription>Alerts requiring your attention</CardDescription>
        </CardHeader>
        <CardContent>
          {pendingAlerts && pendingAlerts.length > 0 ? (
            <div className="space-y-3">
              {pendingAlerts.map((alert) => (
                <div
                  key={alert.alert_id}
                  className="p-4 rounded-lg border bg-card"
                  data-testid={`alert-${alert.alert_id}`}
                >
                  <div className="flex items-start justify-between gap-4">
                    <div className="flex-1 space-y-2">
                      <div className="flex items-center gap-2">
                        <Badge className={getSeverityColor(alert.severity)}>
                          {getSeverityIcon(alert.severity)}
                          {alert.severity.toUpperCase()}
                        </Badge>
                        <span className="text-sm text-muted-foreground">
                          {alert.alert_type}
                        </span>
                      </div>
                      <p className="text-sm font-medium">{alert.message}</p>
                      <div className="flex items-center gap-3 text-xs text-muted-foreground">
                        <div className="flex items-center gap-1">
                          <Clock className="h-3 w-3" />
                          {new Date(alert.triggered_at).toLocaleString()}
                        </div>
                        <div className="flex items-center gap-2">
                          {alert.delivery_status.dashboard && (
                            <Badge variant="outline" className="text-xs">Dashboard</Badge>
                          )}
                          {alert.delivery_status.email && (
                            <Badge variant="outline" className="text-xs">
                              <Mail className="h-3 w-3 mr-1" />
                              Email
                            </Badge>
                          )}
                          {alert.delivery_status.sms && (
                            <Badge variant="outline" className="text-xs">
                              <MessageSquare className="h-3 w-3 mr-1" />
                              SMS
                            </Badge>
                          )}
                        </div>
                      </div>
                    </div>
                    <Button
                      size="sm"
                      onClick={() => acknowledgeMutation.mutate(alert.alert_id)}
                      data-testid={`button-acknowledge-${alert.alert_id}`}
                    >
                      <CheckCircle2 className="h-4 w-4 mr-2" />
                      Acknowledge
                    </Button>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center py-12" data-testid="empty-pending">
              <CheckCircle2 className="h-16 w-16 mx-auto text-teal-500 opacity-50 mb-4" />
              <h3 className="text-lg font-semibold mb-2">All Clear!</h3>
              <p className="text-muted-foreground">No pending alerts at this time</p>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Alert Rules Configuration */}
      <Card data-testid="card-alert-rules">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Settings className="h-5 w-5" />
            Alert Rules
          </CardTitle>
          <CardDescription>Configure when and how you receive alerts</CardDescription>
        </CardHeader>
        <CardContent>
          {alertRules && alertRules.length > 0 ? (
            <div className="space-y-4">
              {alertRules.map((rule) => (
                <div
                  key={rule.rule_id}
                  className="p-4 rounded-lg border bg-card"
                  data-testid={`rule-${rule.rule_id}`}
                >
                  <div className="flex items-start justify-between gap-4">
                    <div className="flex-1 space-y-2">
                      <div className="flex items-center gap-2">
                        <h4 className="font-semibold">{rule.rule_name}</h4>
                        <Badge className={getSeverityColor(rule.severity)}>
                          {rule.severity}
                        </Badge>
                      </div>
                      <p className="text-sm text-muted-foreground">
                        {rule.metric_name} {rule.threshold_type} {rule.threshold_value}
                      </p>
                      <div className="flex items-center gap-2">
                        {rule.notification_channels.includes("dashboard") && (
                          <Badge variant="outline" className="text-xs">Dashboard</Badge>
                        )}
                        {rule.notification_channels.includes("email") && (
                          <Badge variant="outline" className="text-xs">
                            <Mail className="h-3 w-3 mr-1" />
                            Email
                          </Badge>
                        )}
                        {rule.notification_channels.includes("sms") && (
                          <Badge variant="outline" className="text-xs">
                            <MessageSquare className="h-3 w-3 mr-1" />
                            SMS
                          </Badge>
                        )}
                      </div>
                    </div>
                    <div className="flex items-center gap-3">
                      <div className="flex items-center gap-2">
                        <Switch
                          checked={rule.enabled}
                          onCheckedChange={(enabled) =>
                            toggleRuleMutation.mutate({ ruleId: rule.rule_id, enabled })
                          }
                          data-testid={`switch-rule-${rule.rule_id}`}
                        />
                        <Label className="text-sm">
                          {rule.enabled ? "Enabled" : "Disabled"}
                        </Label>
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center py-12">
              <Settings className="h-16 w-16 mx-auto text-muted-foreground opacity-50 mb-4" />
              <h3 className="text-lg font-semibold mb-2">No Alert Rules</h3>
              <p className="text-muted-foreground mb-4">Create your first alert rule to get started</p>
              <Button data-testid="button-create-rule">Create Rule</Button>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Alert History */}
      {alertHistory && alertHistory.length > 0 && (
        <Card data-testid="card-alert-history">
          <CardHeader>
            <CardTitle>Alert History</CardTitle>
            <CardDescription>Recent alerts (last 30 days)</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              {alertHistory.slice(0, 10).map((alert) => (
                <div
                  key={alert.alert_id}
                  className="flex items-center justify-between p-3 rounded-lg border bg-card hover-elevate"
                  data-testid={`history-${alert.alert_id}`}
                >
                  <div className="flex items-center gap-3">
                    {alert.acknowledged ? (
                      <CheckCircle2 className="h-4 w-4 text-teal-500" />
                    ) : (
                      <XCircle className="h-4 w-4 text-muted-foreground" />
                    )}
                    <div>
                      <div className="text-sm font-medium">{alert.message}</div>
                      <div className="text-xs text-muted-foreground">
                        {new Date(alert.triggered_at).toLocaleString()}
                      </div>
                    </div>
                  </div>
                  <Badge className={getSeverityColor(alert.severity)}>
                    {alert.severity}
                  </Badge>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Information */}
      <Alert>
        <Info className="h-4 w-4" />
        <AlertTitle>About Alerts</AlertTitle>
        <AlertDescription>
          <p className="mb-2">
            Alerts notify you when the AI detects significant changes in your health patterns based on:
          </p>
          <ul className="list-disc list-inside ml-4 space-y-1">
            <li>Video AI metrics (respiratory rate, skin changes)</li>
            <li>Audio AI metrics (breathing patterns, voice changes)</li>
            <li>Trend analysis (baseline deviations, risk score changes)</li>
          </ul>
          <p className="mt-3 text-sm font-semibold">
            Alerts are wellness notifications - not emergency medical advice. For urgent concerns, contact your healthcare provider or emergency services.
          </p>
        </AlertDescription>
      </Alert>
    </div>
  );
}
