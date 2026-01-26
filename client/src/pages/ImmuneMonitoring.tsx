import { useQuery, useMutation } from "@tanstack/react-query";
import { queryClient, apiRequest } from "@/lib/queryClient";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { useToast } from "@/hooks/use-toast";
import { 
  Activity, 
  Heart, 
  Moon, 
  Zap, 
  AlertTriangle, 
  CheckCircle2, 
  TrendingUp, 
  TrendingDown,
  RefreshCw,
  Shield,
  Watch
} from "lucide-react";
import { format } from "date-fns";

interface ImmuneBiomarker {
  id: number;
  patientId: string;
  source: string;
  measuredAt: Date;
  hrvRmssd?: number;
  restingHeartRate?: number;
  sleepQuality?: number;
  sleepDuration?: number;
  deepSleepDuration?: number;
  steps?: number;
  activeMinutes?: number;
  caloriesBurned?: number;
  stressLevel?: number;
  recoveryScore?: number;
  bodyTemperature?: number;
  spo2?: number;
}

interface ImmuneDigitalTwin {
  id: number;
  patientId: string;
  immuneScore: number;
  infectionRiskLevel: string;
  trend?: string;
  trendDuration?: string;
  predictedRisks?: string[];
  contributingFactors?: string[];
  generatedAt: Date;
}

interface RiskAlert {
  id: number;
  patientId: string;
  alertType: string;
  severity: string;
  message: string;
  triggeredAt: Date;
  status: string;
}

export default function ImmuneMonitoring() {
  const { toast } = useToast();

  const { data: biomarkers, isLoading: biomarkersLoading } = useQuery<ImmuneBiomarker[]>({
    queryKey: ["/api/immune/biomarkers"],
  });

  const { data: digitalTwin, isLoading: twinLoading } = useQuery<ImmuneDigitalTwin>({
    queryKey: ["/api/immune/digital-twin"],
  });

  const { data: alerts, isLoading: alertsLoading } = useQuery<RiskAlert[]>({
    queryKey: ["/api/risk/alerts"],
  });

  const syncBiomarkersMutation = useMutation({
    mutationFn: async () => {
      const res = await apiRequest("/api/immune/biomarkers/sync", { method: "POST", json: {} });
      return await res.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/immune/biomarkers"] });
      queryClient.invalidateQueries({ queryKey: ["/api/immune/digital-twin"] });
      toast({
        title: "Sync Complete",
        description: "Your immune biomarker data has been updated.",
      });
    },
    onError: () => {
      toast({
        title: "Sync Failed",
        description: "Unable to sync wearable data. Please try again.",
        variant: "destructive",
      });
    },
  });

  const dismissAlertMutation = useMutation({
    mutationFn: async (alertId: number) => {
      const res = await apiRequest(`/api/risk/alerts/${alertId}/dismiss`, { method: "POST", json: {} });
      return await res.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/risk/alerts"] });
      toast({
        title: "Alert Dismissed",
        description: "Risk alert has been acknowledged.",
      });
    },
  });

  const getImmuneScoreColor = (score?: number) => {
    if (!score) return "text-muted-foreground";
    if (score >= 70) return "text-green-600 dark:text-green-400";
    if (score >= 40) return "text-yellow-600 dark:text-yellow-400";
    return "text-red-600 dark:text-red-400";
  };

  const getRiskLevelBadge = (level?: string) => {
    if (!level) return null;
    
    const variants: Record<string, "default" | "secondary" | "destructive" | "outline"> = {
      low: "default",
      moderate: "secondary",
      high: "destructive",
      critical: "destructive",
    };

    return (
      <Badge variant={variants[level] || "outline"} data-testid={`badge-risk-${level}`}>
        {level.toUpperCase()}
      </Badge>
    );
  };

  const getTrendIcon = (trend?: string) => {
    if (trend === "improving") return <TrendingUp className="h-5 w-5 text-green-600" />;
    if (trend === "declining") return <TrendingDown className="h-5 w-5 text-red-600" />;
    return null;
  };

  const activeAlerts = alerts?.filter((a: any) => a.status === "active") || [];
  const latestBiomarker = biomarkers?.[0];

  if (biomarkersLoading || twinLoading || alertsLoading) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <div className="text-center">
          <div className="h-8 w-8 animate-spin rounded-full border-4 border-primary border-t-transparent mx-auto mb-4" />
          <p className="text-muted-foreground">Loading immune monitoring data...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-7xl mx-auto space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight" data-testid="heading-immune-monitoring">
            Immune Monitoring
          </h1>
          <p className="text-muted-foreground">
            Real-time immune function tracking powered by AI digital twin technology
          </p>
        </div>
        <Button
          onClick={() => syncBiomarkersMutation.mutate()}
          disabled={syncBiomarkersMutation.isPending}
          data-testid="button-sync-biomarkers"
        >
          <RefreshCw className={`h-4 w-4 mr-2 ${syncBiomarkersMutation.isPending ? "animate-spin" : ""}`} />
          Sync Wearables
        </Button>
      </div>

      {activeAlerts.length > 0 && (
        <Card className="border-red-200 dark:border-red-900 bg-red-50 dark:bg-red-950/20" data-testid="card-active-alerts">
          <CardHeader className="pb-3">
            <div className="flex items-center gap-2">
              <AlertTriangle className="h-5 w-5 text-red-600" />
              <CardTitle className="text-red-900 dark:text-red-100">Active Risk Alerts</CardTitle>
            </div>
          </CardHeader>
          <CardContent className="space-y-3">
            {activeAlerts.map((alert) => (
              <div 
                key={alert.id} 
                className="flex items-start justify-between p-4 bg-white dark:bg-gray-900 rounded-lg"
                data-testid={`alert-${alert.id}`}
              >
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-1">
                    <Badge variant="destructive" data-testid={`badge-alert-severity-${alert.severity}`}>
                      {alert.severity}
                    </Badge>
                    <Badge variant="outline" data-testid={`badge-alert-type-${alert.alertType}`}>
                      {alert.alertType}
                    </Badge>
                  </div>
                  <p className="text-sm font-medium mb-1" data-testid={`text-alert-message-${alert.id}`}>
                    {alert.message}
                  </p>
                  <p className="text-xs text-muted-foreground">
                    {format(new Date(alert.triggeredAt), "MMM d, yyyy 'at' h:mm a")}
                  </p>
                </div>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => dismissAlertMutation.mutate(alert.id)}
                  disabled={dismissAlertMutation.isPending}
                  data-testid={`button-dismiss-alert-${alert.id}`}
                >
                  Dismiss
                </Button>
              </div>
            ))}
          </CardContent>
        </Card>
      )}

      <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
        <Card className="lg:col-span-2" data-testid="card-immune-digital-twin">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Shield className="h-5 w-5 text-primary" />
              Immune Digital Twin
            </CardTitle>
            <CardDescription>
              AI-powered immune function prediction based on real-time biomarkers
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            {digitalTwin ? (
              <>
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-muted-foreground mb-1">Immune Score</p>
                    <div className="flex items-baseline gap-2">
                      <span 
                        className={`text-5xl font-bold ${getImmuneScoreColor(digitalTwin.immuneScore)}`}
                        data-testid="text-immune-score"
                      >
                        {digitalTwin.immuneScore}
                      </span>
                      <span className="text-lg text-muted-foreground">/100</span>
                    </div>
                  </div>
                  <div className="text-right">
                    <p className="text-sm text-muted-foreground mb-1">Risk Level</p>
                    {getRiskLevelBadge(digitalTwin.infectionRiskLevel)}
                  </div>
                </div>

                <div>
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm font-medium">Immune Function</span>
                    <span className="text-sm text-muted-foreground">{digitalTwin.immuneScore}%</span>
                  </div>
                  <Progress 
                    value={digitalTwin.immuneScore} 
                    className="h-3"
                    data-testid="progress-immune-function"
                  />
                </div>

                {digitalTwin.trend && (
                  <div className="flex items-center gap-2 p-3 bg-muted/50 rounded-lg" data-testid="div-immune-trend">
                    {getTrendIcon(digitalTwin.trend)}
                    <div>
                      <p className="text-sm font-medium">
                        {digitalTwin.trend === "improving" ? "Improving" : digitalTwin.trend === "declining" ? "Declining" : "Stable"}
                      </p>
                      <p className="text-xs text-muted-foreground">
                        {digitalTwin.trendDuration || "Recent trend"}
                      </p>
                    </div>
                  </div>
                )}

                {digitalTwin.predictedRisks && digitalTwin.predictedRisks.length > 0 && (
                  <div>
                    <p className="text-sm font-medium mb-2">Predicted Risks</p>
                    <ul className="space-y-1" data-testid="list-predicted-risks">
                      {digitalTwin.predictedRisks.map((risk: string, idx: number) => (
                        <li key={idx} className="text-sm text-muted-foreground flex items-center gap-2">
                          <AlertTriangle className="h-3 w-3" />
                          {risk}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

                {digitalTwin.contributingFactors && digitalTwin.contributingFactors.length > 0 && (
                  <div>
                    <p className="text-sm font-medium mb-2">Contributing Factors</p>
                    <div className="flex flex-wrap gap-2" data-testid="div-contributing-factors">
                      {digitalTwin.contributingFactors.map((factor: string, idx: number) => (
                        <Badge key={idx} variant="outline">
                          {factor}
                        </Badge>
                      ))}
                    </div>
                  </div>
                )}

                <p className="text-xs text-muted-foreground" data-testid="text-last-updated">
                  Last updated: {format(new Date(digitalTwin.generatedAt), "MMM d, yyyy 'at' h:mm a")}
                </p>
              </>
            ) : (
              <div className="text-center py-12">
                <Shield className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
                <p className="text-muted-foreground mb-4">No immune digital twin data available</p>
                <Button onClick={() => syncBiomarkersMutation.mutate()} data-testid="button-sync-to-generate">
                  <RefreshCw className="h-4 w-4 mr-2" />
                  Sync Wearables to Generate
                </Button>
              </div>
            )}
          </CardContent>
        </Card>

        <Card data-testid="card-wearable-status">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Watch className="h-5 w-5" />
              Wearable Status
            </CardTitle>
          </CardHeader>
          <CardContent>
            {latestBiomarker ? (
              <div className="space-y-4">
                <div className="flex items-center gap-2">
                  <CheckCircle2 className="h-5 w-5 text-green-600" />
                  <span className="text-sm font-medium">Connected</span>
                </div>
                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span className="text-muted-foreground">Last Sync</span>
                    <span className="font-medium" data-testid="text-last-sync">
                      {format(new Date(latestBiomarker.measuredAt), "h:mm a")}
                    </span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-muted-foreground">Data Points</span>
                    <span className="font-medium" data-testid="text-data-points">
                      {biomarkers?.length || 0}
                    </span>
                  </div>
                </div>
              </div>
            ) : (
              <div className="text-center py-6">
                <Watch className="h-8 w-8 mx-auto text-muted-foreground mb-2" />
                <p className="text-sm text-muted-foreground mb-3">No wearable data synced</p>
                <Button 
                  variant="outline" 
                  size="sm"
                  onClick={() => window.location.href = "/wearables"}
                  data-testid="button-connect-wearable"
                >
                  Connect Wearable
                </Button>
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-4">
        <Card data-testid="card-biomarker-hrv">
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between">
              <CardTitle className="text-sm font-medium">Heart Rate Variability</CardTitle>
              <Heart className="h-4 w-4 text-muted-foreground" />
            </div>
          </CardHeader>
          <CardContent>
            {latestBiomarker?.hrvRmssd ? (
              <>
                <div className="text-3xl font-bold" data-testid="text-hrv-value">
                  {latestBiomarker.hrvRmssd}
                </div>
                <p className="text-xs text-muted-foreground">ms (RMSSD)</p>
              </>
            ) : (
              <p className="text-sm text-muted-foreground">No data</p>
            )}
          </CardContent>
        </Card>

        <Card data-testid="card-biomarker-sleep">
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between">
              <CardTitle className="text-sm font-medium">Sleep Quality</CardTitle>
              <Moon className="h-4 w-4 text-muted-foreground" />
            </div>
          </CardHeader>
          <CardContent>
            {latestBiomarker?.sleepQuality ? (
              <>
                <div className="text-3xl font-bold" data-testid="text-sleep-value">
                  {latestBiomarker.sleepQuality}
                </div>
                <p className="text-xs text-muted-foreground">
                  {latestBiomarker.sleepDuration ? `${latestBiomarker.sleepDuration} hours` : "Quality score"}
                </p>
              </>
            ) : (
              <p className="text-sm text-muted-foreground">No data</p>
            )}
          </CardContent>
        </Card>

        <Card data-testid="card-biomarker-activity">
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between">
              <CardTitle className="text-sm font-medium">Activity Level</CardTitle>
              <Activity className="h-4 w-4 text-muted-foreground" />
            </div>
          </CardHeader>
          <CardContent>
            {latestBiomarker?.steps ? (
              <>
                <div className="text-3xl font-bold" data-testid="text-activity-value">
                  {latestBiomarker.steps.toLocaleString()}
                </div>
                <p className="text-xs text-muted-foreground">steps today</p>
              </>
            ) : (
              <p className="text-sm text-muted-foreground">No data</p>
            )}
          </CardContent>
        </Card>

        <Card data-testid="card-biomarker-recovery">
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between">
              <CardTitle className="text-sm font-medium">Recovery Score</CardTitle>
              <Zap className="h-4 w-4 text-muted-foreground" />
            </div>
          </CardHeader>
          <CardContent>
            {latestBiomarker?.recoveryScore ? (
              <>
                <div className="text-3xl font-bold" data-testid="text-recovery-value">
                  {latestBiomarker.recoveryScore}
                </div>
                <p className="text-xs text-muted-foreground">out of 100</p>
              </>
            ) : (
              <p className="text-sm text-muted-foreground">No data</p>
            )}
          </CardContent>
        </Card>
      </div>

      {biomarkers && biomarkers.length > 1 && (
        <Card data-testid="card-biomarker-history">
          <CardHeader>
            <CardTitle>Biomarker History</CardTitle>
            <CardDescription>Recent immune health measurements</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {biomarkers.slice(0, 5).map((biomarker) => (
                <div 
                  key={biomarker.id} 
                  className="flex items-center justify-between p-3 border rounded-lg"
                  data-testid={`biomarker-history-${biomarker.id}`}
                >
                  <div>
                    <p className="text-sm font-medium">
                      {format(new Date(biomarker.measuredAt), "MMM d, yyyy")}
                    </p>
                    <div className="flex gap-4 text-xs text-muted-foreground mt-1">
                      <span>HRV: {biomarker.hrvRmssd || "N/A"}</span>
                      <span>Sleep: {biomarker.sleepQuality || "N/A"}</span>
                      <span>Steps: {biomarker.steps?.toLocaleString() || "N/A"}</span>
                    </div>
                  </div>
                  <Badge variant="outline">
                    {biomarker.source || "Unknown"}
                  </Badge>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
