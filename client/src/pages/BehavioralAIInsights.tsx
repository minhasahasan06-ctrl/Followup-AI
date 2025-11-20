import { useQuery } from "@tanstack/react-query";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { 
  Brain, 
  TrendingUp, 
  AlertTriangle, 
  Activity, 
  Target,
  Clock,
  BarChart3,
  ArrowLeft
} from "lucide-react";
import { useAuth } from "@/hooks/useAuth";
import { Link } from "wouter";
import { format } from "date-fns";

interface RiskScore {
  composite_risk: number;
  risk_level: string;
  calculated_at: string;
}

interface Trend {
  id: number;
  type: string;
  severity: string;
  detected_at: string;
  clinical_significance: string;
}

interface Alert {
  id: number;
  type: string;
  severity: string;
  title: string;
  message: string;
  triggered_at: string;
}

interface DashboardData {
  status: string;
  risk_score: RiskScore | null;
  recent_trends: Trend[];
  unresolved_alerts: Alert[];
}

const severityColors = {
  critical: "destructive",
  high: "destructive",
  moderate: "default",
  medium: "default",
  low: "secondary",
  normal: "secondary"
} as const;

const riskLevelColors = {
  critical: "bg-red-500",
  high: "bg-orange-500",
  moderate: "bg-yellow-500",
  low: "bg-green-500",
  minimal: "bg-blue-500"
} as const;

export default function BehavioralAIInsights() {
  const { user } = useAuth();
  
  const { data, isLoading } = useQuery<DashboardData>({
    queryKey: [`/api/v1/behavior-ai/dashboard/${user?.id}`],
    enabled: !!user?.id,
  });

  const riskPercentage = data?.risk_score?.composite_risk 
    ? Math.round(data.risk_score.composite_risk * 100) 
    : null;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-4xl font-semibold mb-2" data-testid="text-insights-title">
            Behavioral AI Insight
          </h1>
          <p className="text-muted-foreground">
            AI-powered analysis of your behavioral patterns and health trends
          </p>
        </div>
        <Link href="/">
          <Button variant="outline" className="gap-2" data-testid="button-back-dashboard">
            <ArrowLeft className="h-4 w-4" />
            Back to Dashboard
          </Button>
        </Link>
      </div>

      {isLoading ? (
        <div className="text-center py-12">
          <div className="inline-block h-8 w-8 animate-spin rounded-full border-4 border-solid border-current border-r-transparent" />
          <p className="mt-4 text-muted-foreground">Loading your insights...</p>
        </div>
      ) : (
        <>
          {/* Risk Score Overview */}
          <div className="grid gap-6 md:grid-cols-2">
            <Card className="bg-gradient-to-br from-primary/5 to-primary/10" data-testid="card-risk-score">
              <CardHeader>
                <CardTitle className="flex items-center gap-2" data-testid="text-risk-title">
                  <Target className="h-5 w-5" />
                  Overall Risk Score
                </CardTitle>
              </CardHeader>
              <CardContent>
                {data?.risk_score ? (
                  <div className="space-y-4">
                    <div className="flex items-center justify-between">
                      <div className="text-5xl font-bold" data-testid="value-risk-percentage">
                        {riskPercentage}%
                      </div>
                      <Badge 
                        variant={severityColors[data.risk_score.risk_level as keyof typeof severityColors] || "secondary"}
                        className="text-sm"
                        data-testid="badge-risk-level"
                      >
                        {data.risk_score.risk_level.toUpperCase()}
                      </Badge>
                    </div>
                    
                    <div className="w-full bg-muted rounded-full h-3" data-testid="progress-risk">
                      <div 
                        className={`h-3 rounded-full ${riskLevelColors[data.risk_score.risk_level as keyof typeof riskLevelColors] || 'bg-blue-500'}`}
                        style={{ width: `${riskPercentage}%` }}
                        data-testid="progress-risk-fill"
                      />
                    </div>
                    
                    <div className="flex items-center gap-2 text-sm text-muted-foreground">
                      <Clock className="h-4 w-4" />
                      <span data-testid="text-calculated-at">
                        Last calculated: {format(new Date(data.risk_score.calculated_at), 'MMM d, yyyy h:mm a')}
                      </span>
                    </div>
                  </div>
                ) : (
                  <div className="text-center py-8">
                    <Activity className="h-12 w-12 mx-auto mb-4 opacity-50" />
                    <p className="text-muted-foreground" data-testid="text-no-risk-data">
                      No risk data available yet. Complete your daily check-ins to start tracking.
                    </p>
                  </div>
                )}
              </CardContent>
            </Card>

            <Card data-testid="card-metrics-summary">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <BarChart3 className="h-5 w-5" />
                  Metrics Summary
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  <div className="flex items-center justify-between" data-testid="metric-trends">
                    <span className="text-sm text-muted-foreground">Recent Trends</span>
                    <span className="font-semibold">{data?.recent_trends?.length || 0}</span>
                  </div>
                  <div className="flex items-center justify-between" data-testid="metric-alerts">
                    <span className="text-sm text-muted-foreground">Active Alerts</span>
                    <span className="font-semibold">{data?.unresolved_alerts?.length || 0}</span>
                  </div>
                  <div className="flex items-center justify-between" data-testid="metric-risk-level">
                    <span className="text-sm text-muted-foreground">Risk Level</span>
                    <Badge variant="outline" data-testid="badge-current-risk">
                      {data?.risk_score?.risk_level?.toUpperCase() || 'N/A'}
                    </Badge>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Detailed Tabs */}
          <Tabs defaultValue="trends" className="space-y-6">
            <TabsList>
              <TabsTrigger value="trends" data-testid="tab-trends">
                <TrendingUp className="h-4 w-4 mr-2" />
                Trends
              </TabsTrigger>
              <TabsTrigger value="alerts" data-testid="tab-alerts">
                <AlertTriangle className="h-4 w-4 mr-2" />
                Alerts
              </TabsTrigger>
            </TabsList>

            <TabsContent value="trends">
              <Card data-testid="card-trends">
                <CardHeader>
                  <CardTitle>Recent Deterioration Trends</CardTitle>
                  <p className="text-sm text-muted-foreground">
                    AI-detected patterns in your behavioral and health metrics
                  </p>
                </CardHeader>
                <CardContent>
                  {data?.recent_trends && data.recent_trends.length > 0 ? (
                    <div className="space-y-4">
                      {data.recent_trends.map((trend) => (
                        <div 
                          key={trend.id} 
                          className="flex items-start gap-4 p-4 rounded-lg border bg-card hover-elevate"
                          data-testid={`trend-item-${trend.id}`}
                        >
                          <TrendingUp className="h-5 w-5 text-chart-2 mt-0.5 flex-shrink-0" />
                          <div className="flex-1 space-y-2">
                            <div className="flex items-start justify-between gap-2">
                              <div>
                                <p className="font-medium" data-testid={`text-trend-type-${trend.id}`}>
                                  {trend.type.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                                </p>
                                <p className="text-sm text-muted-foreground mt-1" data-testid={`text-trend-significance-${trend.id}`}>
                                  {trend.clinical_significance}
                                </p>
                              </div>
                              <Badge 
                                variant={severityColors[trend.severity as keyof typeof severityColors] || "secondary"}
                                data-testid={`badge-trend-severity-${trend.id}`}
                              >
                                {trend.severity}
                              </Badge>
                            </div>
                            <div className="flex items-center gap-2 text-xs text-muted-foreground">
                              <Clock className="h-3 w-3" />
                              <span data-testid={`text-trend-detected-${trend.id}`}>
                                Detected {format(new Date(trend.detected_at), 'MMM d, yyyy')}
                              </span>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="text-center py-12">
                      <TrendingUp className="h-12 w-12 mx-auto mb-4 opacity-50" />
                      <p className="text-muted-foreground" data-testid="text-no-trends">
                        No trends detected in the last 7 days
                      </p>
                      <p className="text-sm text-muted-foreground mt-1">
                        Keep logging your daily check-ins for AI analysis
                      </p>
                    </div>
                  )}
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="alerts">
              <Card data-testid="card-alerts">
                <CardHeader>
                  <CardTitle>Active Alerts</CardTitle>
                  <p className="text-sm text-muted-foreground">
                    Important notifications requiring your attention
                  </p>
                </CardHeader>
                <CardContent>
                  {data?.unresolved_alerts && data.unresolved_alerts.length > 0 ? (
                    <div className="space-y-4">
                      {data.unresolved_alerts.map((alert) => (
                        <div 
                          key={alert.id}
                          className="flex items-start gap-4 p-4 rounded-lg border bg-card hover-elevate"
                          data-testid={`alert-item-${alert.id}`}
                        >
                          <AlertTriangle className="h-5 w-5 text-destructive mt-0.5 flex-shrink-0" />
                          <div className="flex-1 space-y-2">
                            <div className="flex items-start justify-between gap-2">
                              <div>
                                <p className="font-medium" data-testid={`text-alert-title-${alert.id}`}>
                                  {alert.title}
                                </p>
                                <p className="text-sm text-muted-foreground mt-1" data-testid={`text-alert-message-${alert.id}`}>
                                  {alert.message}
                                </p>
                              </div>
                              <Badge 
                                variant={severityColors[alert.severity as keyof typeof severityColors] || "destructive"}
                                data-testid={`badge-alert-severity-${alert.id}`}
                              >
                                {alert.severity}
                              </Badge>
                            </div>
                            <div className="flex items-center gap-2 text-xs text-muted-foreground">
                              <Clock className="h-3 w-3" />
                              <span data-testid={`text-alert-triggered-${alert.id}`}>
                                Triggered {format(new Date(alert.triggered_at), 'MMM d, yyyy h:mm a')}
                              </span>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="text-center py-12">
                      <AlertTriangle className="h-12 w-12 mx-auto mb-4 opacity-50" />
                      <p className="text-muted-foreground" data-testid="text-no-alerts">
                        No active alerts
                      </p>
                      <p className="text-sm text-muted-foreground mt-1">
                        You're doing great! Continue with your daily routine
                      </p>
                    </div>
                  )}
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>

          {/* Info Footer */}
          <Card className="bg-muted/50" data-testid="card-info-footer">
            <CardContent className="pt-6">
              <div className="flex items-start gap-3">
                <Brain className="h-5 w-5 text-primary mt-0.5 flex-shrink-0" />
                <div className="text-sm text-muted-foreground">
                  <p className="font-medium text-foreground mb-1">
                    About Behavioral AI Analysis
                  </p>
                  <p>
                    Our AI system analyzes multiple data streams including check-in patterns, 
                    digital biomarkers from your device, cognitive test results, and sentiment 
                    in your journal entries to detect early signs of health deterioration. 
                    This helps you and your care team intervene early when needed.
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
        </>
      )}
    </div>
  );
}
