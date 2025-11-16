import { useQuery } from "@tanstack/react-query";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Progress } from "@/components/ui/progress";
import { Separator } from "@/components/ui/separator";
import { 
  Activity, 
  TrendingUp, 
  TrendingDown, 
  AlertTriangle,
  CheckCircle2,
  Info,
  Heart,
  Wind,
  Sparkles,
  Calendar
} from "lucide-react";
import { LegalDisclaimer } from "@/components/LegalDisclaimer";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar } from 'recharts';

interface RiskScore {
  patient_id: string;
  score: number;
  level: string;
  calculated_at: string;
  factors: RiskFactorContribution[];
  total_deviations: number;
  critical_deviations: number;
  moderate_deviations: number;
  recommendation: string;
  action_items: string[];
}

interface RiskFactorContribution {
  metric_name: string;
  z_score: number;
  severity_level: string;
  points: number;
  description: string;
}

interface Deviation {
  id: number;
  metric_name: string;
  measurement_value: number;
  measurement_date: string;
  z_score: number;
  percent_change: number;
  baseline_mean: number;
  baseline_std: number;
  trend_direction: string | null;
  deviation_type: string;
  severity_level: string;
  alert_triggered: boolean;
  alert_message: string | null;
}

interface RiskHistoryPoint {
  date: string;
  score: number;
  level: string;
  deviation_count: number;
}

export default function DeteriorationDashboard() {
  // Fetch current risk score
  const { data: riskScore, isLoading: isLoadingRisk } = useQuery<RiskScore>({
    queryKey: ["/api/v1/risk/score/me"],
  });

  // Fetch risk history (7 days)
  const { data: riskHistory, isLoading: isLoadingHistory } = useQuery<RiskHistoryPoint[]>({
    queryKey: ["/api/v1/risk/history/me"],
  });

  // Fetch recent deviations
  const { data: deviations, isLoading: isLoadingDeviations } = useQuery<Deviation[]>({
    queryKey: ["/api/v1/deviation/me"],
  });

  // Fetch baseline
  const { data: baseline } = useQuery({
    queryKey: ["/api/v1/baseline/current/me"],
  });

  const getRiskLevelColor = (level: string) => {
    switch (level) {
      case "stable":
        return "bg-teal-50 text-teal-700 border-teal-200 dark:bg-teal-950 dark:text-teal-300";
      case "monitoring":
        return "bg-orange-50 text-orange-700 border-orange-200 dark:bg-orange-950 dark:text-orange-300";
      case "urgent":
        return "bg-rose-50 text-rose-700 border-rose-200 dark:bg-rose-950 dark:text-rose-300";
      default:
        return "bg-slate-50 text-slate-700 border-slate-200";
    }
  };

  const getRiskLevelIcon = (level: string) => {
    switch (level) {
      case "stable":
        return <CheckCircle2 className="h-5 w-5" data-testid="icon-stable" />;
      case "monitoring":
        return <Info className="h-5 w-5" data-testid="icon-monitoring" />;
      case "urgent":
        return <AlertTriangle className="h-5 w-5" data-testid="icon-urgent" />;
      default:
        return null;
    }
  };

  const getMetricDisplayName = (metric: string) => {
    const names: Record<string, string> = {
      "respiratory_rate": "Breathing Pattern",
      "pain_facial": "Discomfort Level",
      "pain_self_reported": "Reported Pain",
      "symptom_severity": "Symptom Intensity",
    };
    return names[metric] || metric;
  };

  const getMetricIcon = (metric: string) => {
    switch (metric) {
      case "respiratory_rate":
        return <Wind className="h-4 w-4" />;
      case "pain_facial":
      case "pain_self_reported":
        return <Heart className="h-4 w-4" />;
      case "symptom_severity":
        return <Activity className="h-4 w-4" />;
      default:
        return <Sparkles className="h-4 w-4" />;
    }
  };

  if (isLoadingRisk || isLoadingHistory || isLoadingDeviations) {
    return (
      <div className="container mx-auto p-6 max-w-7xl space-y-6">
        <Skeleton className="h-12 w-96" />
        <div className="grid gap-6 md:grid-cols-3">
          <Skeleton className="h-48" />
          <Skeleton className="h-48" />
          <Skeleton className="h-48" />
        </div>
        <Skeleton className="h-96" />
      </div>
    );
  }

  return (
    <div className="container mx-auto p-6 max-w-7xl space-y-6">
      {/* Header */}
      <div className="space-y-2">
        <h1 className="text-4xl font-bold tracking-tight text-foreground" data-testid="text-page-title">
          Health Change Monitoring
        </h1>
        <p className="text-muted-foreground leading-relaxed">
          Track wellness patterns and changes in your health metrics over time
        </p>
      </div>

      <LegalDisclaimer />

      {/* Risk Score Overview */}
      <div className="grid gap-6 md:grid-cols-3">
        {/* Current Risk Score */}
        <Card className="md:col-span-2" data-testid="card-risk-score">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Activity className="h-5 w-5 text-primary" />
              Current Wellness Score
            </CardTitle>
            <CardDescription>
              Last calculated: {riskScore ? new Date(riskScore.calculated_at).toLocaleString() : 'N/A'}
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {riskScore ? (
              <>
                <div className="flex items-center gap-6">
                  <div className="text-center">
                    <div className="text-6xl font-bold text-primary" data-testid="text-risk-score">
                      {riskScore.score}
                    </div>
                    <div className="text-sm text-muted-foreground">out of 15</div>
                  </div>
                  <div className="flex-1">
                    <Progress value={(riskScore.score / 15) * 100} className="h-4" />
                    <div className="mt-4">
                      <Badge 
                        className={`${getRiskLevelColor(riskScore.level)} flex items-center gap-2 w-fit`}
                        data-testid={`badge-risk-level-${riskScore.level}`}
                      >
                        {getRiskLevelIcon(riskScore.level)}
                        {riskScore.level.charAt(0).toUpperCase() + riskScore.level.slice(1)}
                      </Badge>
                    </div>
                  </div>
                </div>

                <Separator />

                <div className="grid grid-cols-3 gap-4 text-center">
                  <div>
                    <div className="text-2xl font-semibold" data-testid="text-total-deviations">
                      {riskScore.total_deviations}
                    </div>
                    <div className="text-sm text-muted-foreground">Total Changes</div>
                  </div>
                  <div>
                    <div className="text-2xl font-semibold text-orange-600" data-testid="text-moderate-deviations">
                      {riskScore.moderate_deviations}
                    </div>
                    <div className="text-sm text-muted-foreground">Moderate</div>
                  </div>
                  <div>
                    <div className="text-2xl font-semibold text-rose-600" data-testid="text-critical-deviations">
                      {riskScore.critical_deviations}
                    </div>
                    <div className="text-sm text-muted-foreground">Critical</div>
                  </div>
                </div>
              </>
            ) : (
              <div className="text-center py-8 text-muted-foreground">
                <Info className="h-12 w-12 mx-auto mb-4 opacity-50" />
                <p>No risk score data available</p>
                <p className="text-sm mt-2">Continue tracking your health metrics to build your baseline</p>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Quick Summary */}
        <Card data-testid="card-baseline-info">
          <CardHeader>
            <CardTitle className="text-lg">Baseline Status</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            {baseline ? (
              <>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-muted-foreground">Data Points</span>
                  <span className="font-semibold">{baseline.data_points_count || 0}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-muted-foreground">Quality</span>
                  <Badge variant="outline">{baseline.baseline_quality}</Badge>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-muted-foreground">Last Updated</span>
                  <span className="text-sm">
                    {new Date(baseline.created_at).toLocaleDateString()}
                  </span>
                </div>
                <Button 
                  variant="outline" 
                  size="sm" 
                  className="w-full mt-4"
                  data-testid="button-recalculate-baseline"
                >
                  <Calendar className="h-4 w-4 mr-2" />
                  Recalculate
                </Button>
              </>
            ) : (
              <div className="text-sm text-muted-foreground text-center py-4">
                No baseline established yet
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Risk Trend Chart */}
      {riskHistory && riskHistory.length > 0 && (
        <Card data-testid="card-risk-trend">
          <CardHeader>
            <CardTitle>7-Day Wellness Trend</CardTitle>
            <CardDescription>
              Track how your wellness score changes over time
            </CardDescription>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={riskHistory}>
                <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                <XAxis 
                  dataKey="date" 
                  className="text-xs"
                  tickFormatter={(date) => new Date(date).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })}
                />
                <YAxis domain={[0, 15]} className="text-xs" />
                <Tooltip 
                  content={({ active, payload }) => {
                    if (active && payload && payload.length) {
                      const data = payload[0].payload;
                      return (
                        <div className="bg-card border border-border rounded-lg shadow-lg p-3">
                          <p className="font-semibold">{new Date(data.date).toLocaleDateString()}</p>
                          <p className="text-sm">
                            Score: <span className="font-semibold">{data.score}</span>
                          </p>
                          <p className="text-sm">
                            Level: <Badge className={getRiskLevelColor(data.level)} size="sm">{data.level}</Badge>
                          </p>
                          <p className="text-sm text-muted-foreground">
                            {data.deviation_count} change(s)
                          </p>
                        </div>
                      );
                    }
                    return null;
                  }}
                />
                <Legend />
                <Line 
                  type="monotone" 
                  dataKey="score" 
                  stroke="hsl(180, 45%, 45%)" 
                  strokeWidth={2}
                  dot={{ fill: "hsl(180, 45%, 45%)", r: 4 }}
                  name="Wellness Score"
                />
              </LineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      )}

      {/* Recommendations */}
      {riskScore && riskScore.recommendation && (
        <Alert 
          className={riskScore.level === "urgent" ? "border-rose-500 bg-rose-50 dark:bg-rose-950" : ""}
          data-testid="alert-recommendations"
        >
          <AlertTriangle className="h-4 w-4" />
          <AlertTitle className="font-semibold">Wellness Guidance</AlertTitle>
          <AlertDescription>
            <p className="mb-4">{riskScore.recommendation}</p>
            {riskScore.action_items && riskScore.action_items.length > 0 && (
              <ul className="space-y-2 ml-4">
                {riskScore.action_items.map((item, idx) => (
                  <li key={idx} className="text-sm flex items-start gap-2">
                    <CheckCircle2 className="h-4 w-4 mt-0.5 flex-shrink-0 text-teal-600" />
                    <span>{item}</span>
                  </li>
                ))}
              </ul>
            )}
          </AlertDescription>
        </Alert>
      )}

      {/* Risk Factors Breakdown */}
      {riskScore && riskScore.factors && riskScore.factors.length > 0 && (
        <Card data-testid="card-risk-factors">
          <CardHeader>
            <CardTitle>What's Changing</CardTitle>
            <CardDescription>
              Detailed breakdown of pattern changes detected in the last 24 hours
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {riskScore.factors.map((factor, idx) => (
                <div 
                  key={idx} 
                  className="flex items-start gap-4 p-4 rounded-lg border bg-card hover-elevate"
                  data-testid={`factor-${factor.metric_name}`}
                >
                  <div className="flex-shrink-0 mt-1">
                    {getMetricIcon(factor.metric_name)}
                  </div>
                  <div className="flex-1 space-y-2">
                    <div className="flex items-center justify-between">
                      <h4 className="font-semibold">
                        {getMetricDisplayName(factor.metric_name)}
                      </h4>
                      <Badge 
                        variant={factor.severity_level === "critical" ? "destructive" : "secondary"}
                        data-testid={`badge-severity-${factor.severity_level}`}
                      >
                        {factor.severity_level}
                      </Badge>
                    </div>
                    <p className="text-sm text-muted-foreground">{factor.description}</p>
                    <div className="flex items-center gap-4 text-sm">
                      <div>
                        <span className="text-muted-foreground">Z-Score:</span>{" "}
                        <span className="font-mono font-semibold">{factor.z_score.toFixed(2)}</span>
                      </div>
                      <div>
                        <span className="text-muted-foreground">Risk Points:</span>{" "}
                        <span className="font-semibold text-primary">+{factor.points}</span>
                      </div>
                    </div>
                  </div>
                  {factor.z_score > 0 ? (
                    <TrendingUp className="h-5 w-5 text-rose-500 flex-shrink-0" />
                  ) : (
                    <TrendingDown className="h-5 w-5 text-blue-500 flex-shrink-0" />
                  )}
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Recent Deviations */}
      {deviations && deviations.length > 0 && (
        <Card data-testid="card-deviations">
          <CardHeader>
            <CardTitle>Recent Pattern Changes</CardTitle>
            <CardDescription>
              All detected changes in your health patterns (last 7 days)
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {deviations.slice(0, 10).map((deviation) => (
                <div 
                  key={deviation.id}
                  className="flex items-center justify-between p-3 rounded-lg border bg-card hover-elevate"
                  data-testid={`deviation-${deviation.id}`}
                >
                  <div className="flex items-center gap-3 flex-1">
                    <div className="flex-shrink-0">
                      {getMetricIcon(deviation.metric_name)}
                    </div>
                    <div className="flex-1">
                      <div className="font-medium text-sm">
                        {getMetricDisplayName(deviation.metric_name)}
                      </div>
                      <div className="text-xs text-muted-foreground">
                        {new Date(deviation.measurement_date).toLocaleString()}
                      </div>
                    </div>
                  </div>
                  <div className="flex items-center gap-3">
                    <div className="text-right">
                      <div className="text-sm font-mono">
                        z = {deviation.z_score.toFixed(2)}
                      </div>
                      <div className="text-xs text-muted-foreground">
                        {deviation.percent_change > 0 ? "+" : ""}
                        {deviation.percent_change.toFixed(1)}%
                      </div>
                    </div>
                    <Badge 
                      variant={deviation.severity_level === "critical" ? "destructive" : "secondary"}
                      className="flex-shrink-0"
                    >
                      {deviation.severity_level}
                    </Badge>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Empty State */}
      {(!riskScore || !deviations || deviations.length === 0) && (
        <Card className="text-center py-12" data-testid="card-empty-state">
          <CardContent className="space-y-4">
            <Sparkles className="h-16 w-16 mx-auto text-muted-foreground opacity-50" />
            <div>
              <h3 className="text-lg font-semibold mb-2">Getting Started</h3>
              <p className="text-muted-foreground max-w-md mx-auto">
                Continue tracking your health metrics for at least 7 days to establish your personal baseline. 
                Once your baseline is set, we'll start monitoring for changes.
              </p>
            </div>
            <Button variant="default" className="mt-6" data-testid="button-start-tracking">
              Start Tracking
            </Button>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
