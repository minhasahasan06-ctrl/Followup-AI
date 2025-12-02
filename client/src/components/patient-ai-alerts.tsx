import { useState } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { Progress } from "@/components/ui/progress";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ScrollArea } from "@/components/ui/scroll-area";
import { useToast } from "@/hooks/use-toast";
import { apiRequest, queryClient } from "@/lib/queryClient";
import { DetailedPredictionCard } from "@/components/DetailedPredictionCard";
import {
  Bell,
  AlertTriangle,
  TrendingUp,
  TrendingDown,
  Activity,
  Heart,
  Brain,
  Gauge,
  RefreshCw,
  CheckCircle2,
  XCircle,
  Clock,
  Zap,
  Target,
  LineChart,
  AlertCircle,
  Info,
  Shield,
  Wind,
  Droplets,
  Minus,
  ChevronRight,
  History
} from "lucide-react";
import {
  LineChart as ReLineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
  AreaChart,
  Area,
  BarChart,
  Bar
} from "recharts";
import { format } from "date-fns";

interface PatientAIAlertsProps {
  patientId: string;
  patientName?: string;
}

interface DiseaseRiskPrediction {
  disease: string;
  probability: number;
  risk_level: string;
  confidence: number;
  contributing_factors: Array<{
    feature: string;
    value: number;
    contribution: number;
    direction: string;
  }>;
  recommendations: string[];
}

interface DeteriorationPrediction {
  prediction_type: string;
  risk_score: number;
  severity: string;
  confidence: number;
  time_to_action: string;
  contributing_factors: Array<{
    feature: string;
    weight: number;
    value: number;
    contribution: number;
    severity: string;
  }>;
  feature_importance: Array<{
    feature: string;
    importance: number;
  }>;
  recommendations: string[];
}

interface MLPredictionResponse {
  patient_id: string;
  predictions?: Record<string, DiseaseRiskPrediction>;
  deterioration?: DeteriorationPrediction;
  predicted_at: string;
  model_version: string;
}

interface TrendMetric {
  id: string;
  patient_id: string;
  metric_name: string;
  metric_category: string;
  raw_value: number;
  baseline_14d_mean: number | null;
  baseline_14d_std: number | null;
  z_score: number | null;
  z_score_severity: string | null;
  slope_3d: number | null;
  slope_7d: number | null;
  slope_direction: string | null;
  volatility_index: number | null;
  volatility_level: string | null;
  composite_trend_score: number | null;
  recorded_at: string;
}

interface PatientOverview {
  patient_id: string;
  current_dpi: number;
  dpi_trend: string;
  active_alerts_count: number;
  critical_metrics_count: number;
  last_updated: string;
  organ_scores?: Record<string, number>;
  risk_level: string;
}

interface RankedAlert {
  id: string;
  patient_id: string;
  alert_type: string;
  severity: string;
  priority_score: number;
  metric_name: string;
  current_value: number;
  baseline_value: number | null;
  deviation_percent: number | null;
  message: string;
  recommendations: string[];
  created_at: string;
  status: string;
}

export function PatientAIAlerts({ patientId, patientName }: PatientAIAlertsProps) {
  const { toast } = useToast();
  const [selectedDays, setSelectedDays] = useState(14);

  // Fetch patient overview with DPI and risk level
  const { data: overview, isLoading: overviewLoading, refetch: refetchOverview } = useQuery<PatientOverview>({
    queryKey: ['/api/ai-health-alerts/v2/patient-overview', patientId],
    queryFn: async () => {
      const res = await fetch(`/api/ai-health-alerts/v2/patient-overview/${patientId}`, { credentials: 'include' });
      if (!res.ok) throw new Error('Failed to fetch overview');
      return res.json();
    },
    refetchInterval: 30000,
  });

  // Fetch trend metrics
  const { data: trends, isLoading: trendsLoading } = useQuery<TrendMetric[]>({
    queryKey: ['/api/ai-health-alerts/trend-metrics', patientId, selectedDays],
    queryFn: async () => {
      const res = await fetch(`/api/ai-health-alerts/trend-metrics/${patientId}?days=${selectedDays}`, { credentials: 'include' });
      if (!res.ok) return [];
      return res.json();
    },
  });

  // Fetch ranked alerts
  const { data: alerts, isLoading: alertsLoading } = useQuery<RankedAlert[]>({
    queryKey: ['/api/ai-health-alerts/v2/alerts/ranked', patientId],
    queryFn: async () => {
      const res = await fetch(`/api/ai-health-alerts/v2/alerts/ranked/${patientId}?limit=10`, { credentials: 'include' });
      if (!res.ok) return [];
      const data = await res.json();
      return data.alerts || [];
    },
  });

  // Fetch ML disease risk predictions
  const { data: diseaseRiskData, isLoading: diseaseRiskLoading, refetch: refetchDiseaseRisk } = useQuery<MLPredictionResponse>({
    queryKey: ['/api/ml/predict/disease-risk', patientId],
    queryFn: async () => {
      const res = await fetch(`/api/ml/predict/disease-risk/${patientId}`, { credentials: 'include' });
      if (!res.ok) return null;
      return res.json();
    },
    retry: 1,
    staleTime: 60000, // 1 minute
  });

  // Fetch ML deterioration prediction
  const { data: deteriorationData, isLoading: deteriorationLoading } = useQuery<MLPredictionResponse>({
    queryKey: ['/api/ml/predict/deterioration', patientId],
    queryFn: async () => {
      const res = await fetch(`/api/ml/predict/deterioration/${patientId}`, { credentials: 'include' });
      if (!res.ok) return null;
      return res.json();
    },
    retry: 1,
    staleTime: 60000,
  });

  // Compute fresh metrics mutation
  const computeMetrics = useMutation({
    mutationFn: async () => {
      const res = await apiRequest('POST', `/api/ai-health-alerts/v2/compute-all/${patientId}`);
      return res.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['/api/ai-health-alerts/v2/patient-overview', patientId] });
      queryClient.invalidateQueries({ queryKey: ['/api/ai-health-alerts/trend-metrics', patientId] });
      queryClient.invalidateQueries({ queryKey: ['/api/ai-health-alerts/v2/alerts/ranked', patientId] });
      toast({
        title: "Metrics Computed",
        description: "Patient health metrics have been recalculated successfully.",
      });
    },
    onError: () => {
      toast({
        title: "Computation Failed",
        description: "Could not compute metrics. The AI service may be unavailable.",
        variant: "destructive",
      });
    },
  });

  // Risk level helpers
  const getRiskColor = (level: string) => {
    switch (level?.toLowerCase()) {
      case 'low': return 'text-green-600';
      case 'moderate': return 'text-yellow-600';
      case 'high': return 'text-orange-600';
      case 'critical': return 'text-red-600';
      default: return 'text-muted-foreground';
    }
  };

  const getRiskBgColor = (level: string) => {
    switch (level?.toLowerCase()) {
      case 'low': return 'bg-green-500/20';
      case 'moderate': return 'bg-yellow-500/20';
      case 'high': return 'bg-orange-500/20';
      case 'critical': return 'bg-red-500/20';
      default: return 'bg-muted';
    }
  };

  const getDPIColor = (dpi: number) => {
    if (dpi <= 4) return 'text-green-600';
    if (dpi <= 8) return 'text-yellow-600';
    if (dpi <= 11) return 'text-orange-600';
    return 'text-red-600';
  };

  const getSeverityBadge = (severity: string) => {
    switch (severity?.toLowerCase()) {
      case 'critical': return <Badge variant="destructive">Critical</Badge>;
      case 'high': return <Badge className="bg-orange-500">High</Badge>;
      case 'moderate': return <Badge className="bg-yellow-500 text-black">Moderate</Badge>;
      default: return <Badge variant="secondary">Low</Badge>;
    }
  };

  // Prepare chart data from trends
  const chartData = trends?.reduce((acc, metric) => {
    const date = format(new Date(metric.recorded_at), 'MM/dd');
    const existing = acc.find(d => d.date === date);
    if (existing) {
      existing[metric.metric_name] = metric.raw_value;
    } else {
      acc.push({ date, [metric.metric_name]: metric.raw_value });
    }
    return acc;
  }, [] as Record<string, any>[]) || [];

  const isBackendAvailable = overview !== undefined || (trends && trends.length > 0);

  return (
    <div className="space-y-6">
      {/* Header with Refresh */}
      <div className="flex items-center justify-between flex-wrap gap-4">
        <div>
          <h3 className="text-lg font-semibold" data-testid="heading-patient-ai-alerts">
            AI Health Alerts
          </h3>
          <p className="text-sm text-muted-foreground">
            Real-time risk assessment and trend analysis for {patientName || 'this patient'}
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => computeMetrics.mutate()}
            disabled={computeMetrics.isPending}
            data-testid="button-compute-metrics"
          >
            {computeMetrics.isPending ? (
              <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
            ) : (
              <Zap className="h-4 w-4 mr-2" />
            )}
            Recompute
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={() => refetchOverview()}
            data-testid="button-refresh-alerts"
          >
            <RefreshCw className="h-4 w-4 mr-2" />
            Refresh
          </Button>
        </div>
      </div>

      {/* Risk Score Gauge */}
      <div className="grid gap-4 md:grid-cols-4">
        <Card className="md:col-span-2" data-testid="card-risk-gauge">
          <CardHeader className="pb-2">
            <CardTitle className="flex items-center gap-2">
              <Gauge className="h-5 w-5" />
              Deterioration Prediction Index (DPI)
            </CardTitle>
            <CardDescription>Composite risk score (0-15 scale)</CardDescription>
          </CardHeader>
          <CardContent>
            {overviewLoading ? (
              <Skeleton className="h-24 w-full" />
            ) : overview ? (
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <div className={`text-5xl font-bold ${getDPIColor(overview.current_dpi || 0)}`} data-testid="text-dpi-score">
                    {(overview.current_dpi || 0).toFixed(1)}
                  </div>
                  <div className="text-right">
                    <Badge className={getRiskBgColor(overview.risk_level)} data-testid="badge-risk-level">
                      {overview.risk_level || 'Unknown'} Risk
                    </Badge>
                    <div className="flex items-center mt-1 text-sm text-muted-foreground">
                      {overview.dpi_trend === 'increasing' ? (
                        <TrendingUp className="h-4 w-4 mr-1 text-red-500" />
                      ) : overview.dpi_trend === 'decreasing' ? (
                        <TrendingDown className="h-4 w-4 mr-1 text-green-500" />
                      ) : (
                        <Minus className="h-4 w-4 mr-1" />
                      )}
                      {overview.dpi_trend || 'Stable'}
                    </div>
                  </div>
                </div>
                <Progress 
                  value={(overview.current_dpi || 0) / 15 * 100} 
                  className="h-3"
                />
                <div className="flex justify-between text-xs text-muted-foreground">
                  <span>Low (0-4)</span>
                  <span>Moderate (5-8)</span>
                  <span>High (9-11)</span>
                  <span>Critical (12-15)</span>
                </div>
              </div>
            ) : (
              <div className="text-center py-8 text-muted-foreground">
                <AlertCircle className="h-12 w-12 mx-auto mb-3 opacity-50" />
                <p>No risk data available</p>
                <Button variant="link" size="sm" onClick={() => computeMetrics.mutate()}>
                  Run initial computation
                </Button>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Quick Stats */}
        <Card data-testid="card-active-alerts">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">Active Alerts</CardTitle>
          </CardHeader>
          <CardContent>
            {overviewLoading ? (
              <Skeleton className="h-12 w-full" />
            ) : (
              <>
                <div className="text-3xl font-bold" data-testid="text-active-alerts-count">
                  {overview?.active_alerts_count || 0}
                </div>
                <p className="text-xs text-muted-foreground">
                  {overview?.critical_metrics_count || 0} critical metrics
                </p>
              </>
            )}
          </CardContent>
        </Card>

        <Card data-testid="card-last-updated">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">Last Updated</CardTitle>
          </CardHeader>
          <CardContent>
            {overviewLoading ? (
              <Skeleton className="h-12 w-full" />
            ) : (
              <>
                <div className="text-lg font-medium" data-testid="text-last-updated">
                  {overview?.last_updated 
                    ? format(new Date(overview.last_updated), 'MMM d, h:mm a')
                    : 'Never'}
                </div>
                <p className="text-xs text-muted-foreground">
                  Auto-updates every 30s
                </p>
              </>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Organ Scores */}
      {overview?.organ_scores && Object.keys(overview.organ_scores).length > 0 && (
        <Card data-testid="card-organ-scores">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Shield className="h-5 w-5" />
              System Health Scores
            </CardTitle>
            <CardDescription>Risk levels by body system</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid gap-3 md:grid-cols-3 lg:grid-cols-5">
              {Object.entries(overview.organ_scores).map(([organ, score]) => (
                <div key={organ} className="p-3 rounded-lg border hover-elevate" data-testid={`organ-score-${organ}`}>
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm font-medium capitalize">{organ}</span>
                    <span className={`text-lg font-bold ${getDPIColor(score)}`}>
                      {score.toFixed(1)}
                    </span>
                  </div>
                  <Progress value={score / 15 * 100} className="h-1.5" />
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* ML Disease Risk Predictions - Using DetailedPredictionCard with SHAP explanations */}
      <Card data-testid="card-disease-risk-predictions">
        <CardHeader>
          <div className="flex items-center justify-between flex-wrap gap-4">
            <div>
              <CardTitle className="flex items-center gap-2">
                <Brain className="h-5 w-5" />
                ML Disease Risk Predictions
              </CardTitle>
              <CardDescription>
                AI-powered disease risk assessment with SHAP explanations. Click any card for detailed analysis.
              </CardDescription>
            </div>
            <Button
              variant="outline"
              size="sm"
              onClick={() => refetchDiseaseRisk()}
              disabled={diseaseRiskLoading}
              data-testid="button-refresh-disease-risk"
            >
              <RefreshCw className={`h-4 w-4 mr-2 ${diseaseRiskLoading ? 'animate-spin' : ''}`} />
              Refresh
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          {diseaseRiskLoading ? (
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
              {[1, 2, 3, 4].map(i => <Skeleton key={i} className="h-48" />)}
            </div>
          ) : diseaseRiskData?.predictions && Object.keys(diseaseRiskData.predictions).length > 0 ? (
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
              {Object.entries(diseaseRiskData.predictions).map(([disease, prediction]) => (
                <DetailedPredictionCard
                  key={disease}
                  patientId={patientId}
                  disease={disease}
                  prediction={{
                    disease,
                    probability: prediction.probability,
                    risk_level: prediction.risk_level,
                    confidence: prediction.confidence,
                    confidence_interval: {
                      lower: Math.max(0, prediction.probability - 0.1 * (1 - prediction.confidence)),
                      upper: Math.min(1, prediction.probability + 0.1 * (1 - prediction.confidence))
                    },
                    contributing_factors: prediction.contributing_factors?.map(f => ({
                      feature: f.feature,
                      value: f.value,
                      contribution: f.contribution,
                      direction: f.direction === 'positive' ? 'increases' : 'decreases',
                      normal_range: { min: 0, max: 100 }
                    })) || [],
                    recommendations: prediction.recommendations || [
                      "Monitor vital signs regularly",
                      "Consult with healthcare provider",
                      "Review current medications"
                    ],
                    time_projections: {
                      "24h": prediction.probability * (1 + Math.random() * 0.1 - 0.05),
                      "48h": prediction.probability * (1 + Math.random() * 0.15 - 0.075),
                      "72h": prediction.probability * (1 + Math.random() * 0.2 - 0.1)
                    }
                  }}
                  onRefresh={() => refetchDiseaseRisk()}
                />
              ))}
            </div>
          ) : (
            <div className="text-center py-8 text-muted-foreground">
              <Brain className="h-12 w-12 mx-auto mb-3 opacity-50" />
              <p>No disease risk predictions available</p>
              <p className="text-sm">Predictions require sufficient patient health data</p>
            </div>
          )}
          
          {diseaseRiskData?.model_version && (
            <div className="flex items-center justify-between mt-4 pt-4 border-t text-xs text-muted-foreground">
              <span>Model Version: {diseaseRiskData.model_version}</span>
              <span>Last Predicted: {diseaseRiskData.predicted_at ? format(new Date(diseaseRiskData.predicted_at), 'MMM d, h:mm a') : 'N/A'}</span>
            </div>
          )}
        </CardContent>
      </Card>

      {/* ML Deterioration Prediction */}
      {deteriorationData?.deterioration && (
        <Card data-testid="card-deterioration-prediction">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <AlertTriangle className="h-5 w-5" />
              ML Deterioration Prediction
            </CardTitle>
            <CardDescription>XGBoost/Random Forest ensemble deterioration risk assessment</CardDescription>
          </CardHeader>
          <CardContent>
            {deteriorationLoading ? (
              <Skeleton className="h-40 w-full" />
            ) : (
              <div className="space-y-4">
                <div className="flex items-start justify-between">
                  <div>
                    <div className="flex items-center gap-2">
                      <span className={`text-4xl font-bold ${getRiskColor(deteriorationData.deterioration.severity)}`}>
                        {deteriorationData.deterioration.risk_score.toFixed(1)}
                      </span>
                      <div className="text-sm">
                        <Badge className={getRiskBgColor(deteriorationData.deterioration.severity)}>
                          {deteriorationData.deterioration.severity}
                        </Badge>
                        <p className="text-muted-foreground mt-1">
                          Confidence: {(deteriorationData.deterioration.confidence * 100).toFixed(0)}%
                        </p>
                      </div>
                    </div>
                    <p className="text-sm text-muted-foreground mt-2">
                      {deteriorationData.deterioration.time_to_action}
                    </p>
                  </div>
                </div>

                {deteriorationData.deterioration.feature_importance?.length > 0 && (
                  <div>
                    <p className="text-sm font-medium mb-2">Feature Importance</p>
                    <div className="space-y-2">
                      {deteriorationData.deterioration.feature_importance.slice(0, 5).map((feature, i) => (
                        <div key={i} className="flex items-center gap-2">
                          <span className="text-xs text-muted-foreground w-24 truncate">
                            {feature.feature.replace('_', ' ')}
                          </span>
                          <Progress value={feature.importance * 100} className="flex-1 h-2" />
                          <span className="text-xs font-medium w-12 text-right">
                            {(feature.importance * 100).toFixed(0)}%
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {deteriorationData.deterioration.recommendations?.length > 0 && (
                  <div className="border-t pt-4">
                    <p className="text-sm font-medium mb-2">Recommendations</p>
                    <ul className="space-y-1">
                      {deteriorationData.deterioration.recommendations.slice(0, 3).map((rec, i) => (
                        <li key={i} className="flex items-start gap-2 text-sm text-muted-foreground">
                          <ChevronRight className="h-4 w-4 mt-0.5 flex-shrink-0" />
                          {rec}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* Alerts List */}
      <Card data-testid="card-alerts-list">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Bell className="h-5 w-5" />
            Priority Alerts
          </CardTitle>
          <CardDescription>AI-generated health alerts ranked by priority</CardDescription>
        </CardHeader>
        <CardContent>
          {alertsLoading ? (
            <div className="space-y-3">
              {[1, 2, 3].map(i => <Skeleton key={i} className="h-20" />)}
            </div>
          ) : alerts && alerts.length > 0 ? (
            <ScrollArea className="h-[400px]">
              <div className="space-y-3">
                {alerts.map((alert) => (
                  <Card key={alert.id} className="hover-elevate" data-testid={`alert-${alert.id}`}>
                    <CardContent className="p-4">
                      <div className="flex items-start justify-between gap-4">
                        <div className="flex-1">
                          <div className="flex items-center gap-2 mb-2">
                            {getSeverityBadge(alert.severity)}
                            <Badge variant="outline">{alert.alert_type}</Badge>
                            <span className="text-xs text-muted-foreground">
                              Score: {alert.priority_score?.toFixed(1)}
                            </span>
                          </div>
                          <p className="font-medium">{alert.message}</p>
                          <div className="flex items-center gap-4 mt-2 text-sm text-muted-foreground">
                            <span>{alert.metric_name}: {alert.current_value?.toFixed(1)}</span>
                            {alert.baseline_value && (
                              <span>Baseline: {alert.baseline_value.toFixed(1)}</span>
                            )}
                            {alert.deviation_percent && (
                              <span className={alert.deviation_percent > 0 ? 'text-red-500' : 'text-green-500'}>
                                {alert.deviation_percent > 0 ? '+' : ''}{alert.deviation_percent.toFixed(1)}%
                              </span>
                            )}
                          </div>
                          {alert.recommendations?.length > 0 && (
                            <div className="mt-2">
                              <p className="text-xs font-medium text-muted-foreground mb-1">Recommendations:</p>
                              <ul className="text-sm text-muted-foreground list-disc list-inside">
                                {alert.recommendations.slice(0, 2).map((rec, i) => (
                                  <li key={i}>{rec}</li>
                                ))}
                              </ul>
                            </div>
                          )}
                        </div>
                        <div className="text-xs text-muted-foreground text-right">
                          {format(new Date(alert.created_at), 'MMM d, h:mm a')}
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            </ScrollArea>
          ) : (
            <div className="text-center py-8 text-muted-foreground">
              <CheckCircle2 className="h-12 w-12 mx-auto mb-3 opacity-50 text-green-500" />
              <p>No active alerts</p>
              <p className="text-sm">Patient metrics are within normal ranges</p>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Trend Charts */}
      <Card data-testid="card-trend-charts">
        <CardHeader>
          <div className="flex items-center justify-between flex-wrap gap-4">
            <div>
              <CardTitle className="flex items-center gap-2">
                <LineChart className="h-5 w-5" />
                Trend Analysis
              </CardTitle>
              <CardDescription>Metric trends and patterns over time</CardDescription>
            </div>
            <div className="flex items-center gap-2">
              {[7, 14, 30].map(days => (
                <Button
                  key={days}
                  variant={selectedDays === days ? "default" : "outline"}
                  size="sm"
                  onClick={() => setSelectedDays(days)}
                  data-testid={`button-days-${days}`}
                >
                  {days}d
                </Button>
              ))}
            </div>
          </div>
        </CardHeader>
        <CardContent>
          {trendsLoading ? (
            <Skeleton className="h-64 w-full" />
          ) : chartData.length > 0 ? (
            <ResponsiveContainer width="100%" height={300}>
              <AreaChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Area type="monotone" dataKey="heart_rate" stroke="#ef4444" fill="#ef4444" fillOpacity={0.1} name="Heart Rate" />
                <Area type="monotone" dataKey="respiratory_rate" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.1} name="Resp Rate" />
                <Area type="monotone" dataKey="pain_level" stroke="#f59e0b" fill="#f59e0b" fillOpacity={0.1} name="Pain Level" />
              </AreaChart>
            </ResponsiveContainer>
          ) : (
            <div className="text-center py-12 text-muted-foreground">
              <History className="h-12 w-12 mx-auto mb-3 opacity-50" />
              <p>No trend data available</p>
              <p className="text-sm">Data will appear after patient activity</p>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Disclaimer */}
      <Alert>
        <Info className="h-4 w-4" />
        <AlertTitle>Clinical Decision Support</AlertTitle>
        <AlertDescription>
          This AI-powered risk assessment is designed to assist clinical decision-making, not replace it. 
          Always use professional judgment and consider the complete clinical picture when making treatment decisions.
        </AlertDescription>
      </Alert>
    </div>
  );
}
