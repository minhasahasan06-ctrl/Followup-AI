import { useState } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Progress } from "@/components/ui/progress";
import { Skeleton } from "@/components/ui/skeleton";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import { useToast } from "@/hooks/use-toast";
import { apiRequest, queryClient } from "@/lib/queryClient";
import { DetailedPredictionCard } from "@/components/DetailedPredictionCard";
import {
  Brain,
  Heart,
  Activity,
  TrendingUp,
  TrendingDown,
  Minus,
  RefreshCw,
  AlertTriangle,
  Shield,
  Droplets,
  Clock,
  Target,
  BarChart3,
  LineChart,
  Users,
  Zap,
  ChevronRight,
  Info,
  CheckCircle2,
} from "lucide-react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as ReTooltip,
  ResponsiveContainer,
  Cell,
  PieChart,
  Pie,
  Legend,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
} from "recharts";
import { format } from "date-fns";

interface ContributingFactor {
  feature: string;
  value: number;
  contribution: number;
  direction: string;
  baseline?: number;
  normal_range?: { min: number; max: number };
}

interface DiseaseRiskPrediction {
  disease: string;
  probability: number;
  risk_level: string;
  confidence: number;
  confidence_interval?: { lower: number; upper: number };
  contributing_factors: ContributingFactor[];
  recommendations: string[];
  time_projections?: {
    "24h": number;
    "48h": number;
    "72h": number;
  };
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

interface TimeSeriesForecast {
  metric: string;
  current_value: number;
  forecasts: {
    "24h": { value: number; confidence_interval: { lower: number; upper: number } };
    "48h": { value: number; confidence_interval: { lower: number; upper: number } };
    "72h": { value: number; confidence_interval: { lower: number; upper: number } };
  };
  trend_direction: string;
  trend_strength: number;
}

interface PatientSegment {
  segment_id: string;
  segment_name: string;
  description: string;
  risk_profile: string;
  characteristics: string[];
  similar_patient_count: number;
  membership_score: number;
}

interface MLPredictionResponse {
  patient_id: string;
  predictions?: Record<string, DiseaseRiskPrediction>;
  deterioration?: DeteriorationPrediction;
  time_series?: TimeSeriesForecast[];
  patient_segment?: PatientSegment;
  predicted_at: string;
  model_version: string;
}

interface MLInsightsPanelProps {
  patientId: string;
  patientName?: string;
  isDoctor?: boolean;
}

const getRiskColor = (level: string) => {
  switch (level?.toLowerCase()) {
    case "critical":
      return "text-red-600 dark:text-red-400";
    case "high":
      return "text-orange-600 dark:text-orange-400";
    case "moderate":
      return "text-yellow-600 dark:text-yellow-400";
    case "low":
      return "text-green-600 dark:text-green-400";
    default:
      return "text-muted-foreground";
  }
};

const getRiskBgColor = (level: string) => {
  switch (level?.toLowerCase()) {
    case "critical":
      return "bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-300";
    case "high":
      return "bg-orange-100 text-orange-800 dark:bg-orange-900/30 dark:text-orange-300";
    case "moderate":
      return "bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-300";
    case "low":
      return "bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300";
    default:
      return "";
  }
};

const getDiseaseIcon = (disease: string) => {
  switch (disease) {
    case "stroke":
      return <Brain className="h-5 w-5 text-purple-500" />;
    case "sepsis":
      return <Droplets className="h-5 w-5 text-red-500" />;
    case "diabetes":
      return <Activity className="h-5 w-5 text-blue-500" />;
    case "heart_disease":
      return <Heart className="h-5 w-5 text-pink-500" />;
    default:
      return <Shield className="h-5 w-5" />;
  }
};

function OverviewRiskGauge({ 
  diseaseRisks,
  deterioration,
  isLoading 
}: { 
  diseaseRisks?: Record<string, DiseaseRiskPrediction>;
  deterioration?: DeteriorationPrediction;
  isLoading: boolean;
}) {
  if (isLoading) {
    return (
      <Card data-testid="card-overview-risk">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Target className="h-5 w-5" />
            Risk Overview
          </CardTitle>
        </CardHeader>
        <CardContent>
          <Skeleton className="h-24 w-full" />
        </CardContent>
      </Card>
    );
  }

  const risks = diseaseRisks ? Object.values(diseaseRisks) : [];
  const maxRisk = risks.reduce((max, r) => r.probability > max ? r.probability : max, 0);
  const avgRisk = risks.length > 0 ? risks.reduce((sum, r) => sum + r.probability, 0) / risks.length : 0;
  const deteriorationRisk = deterioration?.risk_score ?? 0;

  const overallRisk = Math.max(maxRisk, deteriorationRisk / 15);
  const overallLevel = overallRisk > 0.75 ? "critical" : overallRisk > 0.5 ? "high" : overallRisk > 0.25 ? "moderate" : "low";

  const radarData = [
    { category: "Cardiovascular", value: (diseaseRisks?.heart_disease?.probability ?? 0) * 100 },
    { category: "Stroke", value: (diseaseRisks?.stroke?.probability ?? 0) * 100 },
    { category: "Diabetes", value: (diseaseRisks?.diabetes?.probability ?? 0) * 100 },
    { category: "Sepsis", value: (diseaseRisks?.sepsis?.probability ?? 0) * 100 },
    { category: "Deterioration", value: (deteriorationRisk / 15) * 100 },
  ];

  return (
    <Card data-testid="card-overview-risk">
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2">
            <Target className="h-5 w-5" />
            Risk Overview
          </CardTitle>
          <Badge className={getRiskBgColor(overallLevel)}>
            {overallLevel}
          </Badge>
        </div>
        <CardDescription>Combined risk assessment from all ML models</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="grid md:grid-cols-2 gap-4">
          <div className="space-y-4">
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span className="text-muted-foreground">Overall Risk Score</span>
                <span className={`font-bold ${getRiskColor(overallLevel)}`}>
                  {(overallRisk * 100).toFixed(0)}%
                </span>
              </div>
              <Progress value={overallRisk * 100} className="h-3" />
            </div>
            
            <div className="grid grid-cols-2 gap-2 text-sm">
              <div className="p-2 rounded-lg bg-muted/50">
                <p className="text-xs text-muted-foreground">Highest Disease Risk</p>
                <p className="font-semibold">{(maxRisk * 100).toFixed(0)}%</p>
              </div>
              <div className="p-2 rounded-lg bg-muted/50">
                <p className="text-xs text-muted-foreground">Average Risk</p>
                <p className="font-semibold">{(avgRisk * 100).toFixed(0)}%</p>
              </div>
              <div className="p-2 rounded-lg bg-muted/50">
                <p className="text-xs text-muted-foreground">Deterioration Score</p>
                <p className="font-semibold">{deteriorationRisk.toFixed(1)}/15</p>
              </div>
              <div className="p-2 rounded-lg bg-muted/50">
                <p className="text-xs text-muted-foreground">Active Predictions</p>
                <p className="font-semibold">{risks.length + (deterioration ? 1 : 0)}</p>
              </div>
            </div>
          </div>

          <div className="h-48">
            <ResponsiveContainer width="100%" height="100%">
              <RadarChart data={radarData}>
                <PolarGrid />
                <PolarAngleAxis dataKey="category" tick={{ fontSize: 10 }} />
                <PolarRadiusAxis domain={[0, 100]} tick={{ fontSize: 9 }} />
                <Radar
                  name="Risk %"
                  dataKey="value"
                  stroke="#ef4444"
                  fill="#ef4444"
                  fillOpacity={0.3}
                />
              </RadarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

function DeteriorationPanel({ 
  deterioration, 
  isLoading,
  onRefresh 
}: { 
  deterioration?: DeteriorationPrediction;
  isLoading: boolean;
  onRefresh: () => void;
}) {
  if (isLoading) {
    return (
      <div className="space-y-4">
        <Skeleton className="h-32 w-full" />
        <Skeleton className="h-64 w-full" />
      </div>
    );
  }

  if (!deterioration) {
    return (
      <Card>
        <CardContent className="py-12 text-center">
          <Activity className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
          <h3 className="text-lg font-semibold mb-2">No Deterioration Data</h3>
          <p className="text-muted-foreground max-w-md mx-auto mb-4">
            Deterioration prediction requires sufficient health data. 
            Continue logging daily health information to enable this feature.
          </p>
          <Button variant="outline" onClick={onRefresh}>
            <RefreshCw className="h-4 w-4 mr-2" />
            Check Again
          </Button>
        </CardContent>
      </Card>
    );
  }

  const importanceData = deterioration.feature_importance?.slice(0, 8).map(f => ({
    feature: f.feature.replace(/_/g, " "),
    importance: Math.abs(f.importance) * 100,
  })) || [];

  const contributingFactors = deterioration.contributing_factors?.slice(0, 6) || [];

  return (
    <div className="space-y-4">
      <Card data-testid="card-deterioration-score">
        <CardHeader className="pb-2">
          <div className="flex items-center justify-between">
            <CardTitle className="flex items-center gap-2">
              <AlertTriangle className="h-5 w-5" />
              Deterioration Risk Score
            </CardTitle>
            <Button variant="ghost" size="sm" onClick={onRefresh}>
              <RefreshCw className="h-4 w-4" />
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-between mb-4">
            <div>
              <p className={`text-4xl font-bold ${getRiskColor(deterioration.severity)}`}>
                {deterioration.risk_score.toFixed(1)}
              </p>
              <p className="text-sm text-muted-foreground">out of 15</p>
            </div>
            <div className="text-right">
              <Badge className={getRiskBgColor(deterioration.severity)}>
                {deterioration.severity}
              </Badge>
              <p className="text-xs text-muted-foreground mt-1">
                Confidence: {(deterioration.confidence * 100).toFixed(0)}%
              </p>
            </div>
          </div>
          
          <Progress value={(deterioration.risk_score / 15) * 100} className="h-3 mb-4" />

          {deterioration.time_to_action && (
            <div className="flex items-center gap-2 p-2 rounded-lg bg-muted/50">
              <Clock className="h-4 w-4 text-muted-foreground" />
              <span className="text-sm">
                Recommended action window: <strong>{deterioration.time_to_action}</strong>
              </span>
            </div>
          )}
        </CardContent>
      </Card>

      <div className="grid md:grid-cols-2 gap-4">
        <Card data-testid="card-feature-importance">
          <CardHeader className="pb-2">
            <CardTitle className="text-base flex items-center gap-2">
              <BarChart3 className="h-4 w-4" />
              Feature Importance
            </CardTitle>
            <CardDescription>Key factors driving the deterioration prediction</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={importanceData}
                  layout="vertical"
                  margin={{ top: 0, right: 10, left: 70, bottom: 0 }}
                >
                  <CartesianGrid strokeDasharray="3 3" horizontal={true} vertical={false} />
                  <XAxis type="number" domain={[0, 100]} tickFormatter={(v) => `${v}%`} />
                  <YAxis type="category" dataKey="feature" tick={{ fontSize: 11 }} width={65} />
                  <ReTooltip
                    formatter={(value: number) => [`${value.toFixed(1)}%`, "Importance"]}
                  />
                  <Bar dataKey="importance" fill="#6366f1" radius={[0, 4, 4, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        <Card data-testid="card-contributing-factors">
          <CardHeader className="pb-2">
            <CardTitle className="text-base flex items-center gap-2">
              <Activity className="h-4 w-4" />
              Contributing Factors
            </CardTitle>
            <CardDescription>Current values affecting risk</CardDescription>
          </CardHeader>
          <CardContent>
            <ScrollArea className="h-64">
              <div className="space-y-2">
                {contributingFactors.map((factor, i) => (
                  <div 
                    key={i} 
                    className="flex items-center justify-between p-2 rounded-lg bg-muted/30 hover-elevate"
                  >
                    <div className="flex-1">
                      <p className="text-sm font-medium capitalize">
                        {factor.feature.replace(/_/g, " ")}
                      </p>
                      <p className="text-xs text-muted-foreground">
                        Current: {factor.value?.toFixed(2) ?? "N/A"}
                      </p>
                    </div>
                    <div className="flex items-center gap-2">
                      <Badge
                        className={
                          factor.severity === "high" || factor.severity === "critical"
                            ? "bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-300"
                            : factor.severity === "moderate"
                            ? "bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-300"
                            : "bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300"
                        }
                      >
                        {factor.severity}
                      </Badge>
                    </div>
                  </div>
                ))}
              </div>
            </ScrollArea>
          </CardContent>
        </Card>
      </div>

      {deterioration.recommendations && deterioration.recommendations.length > 0 && (
        <Card data-testid="card-deterioration-recommendations">
          <CardHeader className="pb-2">
            <CardTitle className="text-base flex items-center gap-2">
              <CheckCircle2 className="h-4 w-4" />
              Recommendations
            </CardTitle>
          </CardHeader>
          <CardContent>
            <ul className="space-y-2">
              {deterioration.recommendations.map((rec, i) => (
                <li key={i} className="flex items-start gap-2 text-sm">
                  <ChevronRight className="h-4 w-4 mt-0.5 flex-shrink-0 text-primary" />
                  <span>{rec}</span>
                </li>
              ))}
            </ul>
          </CardContent>
        </Card>
      )}
    </div>
  );
}

function TimeSeriesPanel({ 
  forecasts, 
  isLoading 
}: { 
  forecasts?: TimeSeriesForecast[];
  isLoading: boolean;
}) {
  if (isLoading) {
    return <Skeleton className="h-64 w-full" />;
  }

  if (!forecasts || forecasts.length === 0) {
    return (
      <Card>
        <CardContent className="py-12 text-center">
          <LineChart className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
          <h3 className="text-lg font-semibold mb-2">Time-Series Forecasts</h3>
          <p className="text-muted-foreground max-w-md mx-auto">
            LSTM-based vital sign forecasting will be available once sufficient 
            time-series data is collected. Continue logging daily vitals to enable predictions.
          </p>
          <Badge variant="secondary" className="mt-4">Coming Soon</Badge>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="grid md:grid-cols-2 gap-4">
      {forecasts.map((forecast, i) => (
        <Card key={i} data-testid={`card-forecast-${forecast.metric}`}>
          <CardHeader className="pb-2">
            <div className="flex items-center justify-between">
              <CardTitle className="text-base capitalize">
                {forecast.metric.replace(/_/g, " ")} Forecast
              </CardTitle>
              <Badge variant={forecast.trend_direction === "increasing" ? "destructive" : forecast.trend_direction === "decreasing" ? "default" : "secondary"}>
                {forecast.trend_direction === "increasing" ? (
                  <TrendingUp className="h-3 w-3 mr-1" />
                ) : forecast.trend_direction === "decreasing" ? (
                  <TrendingDown className="h-3 w-3 mr-1" />
                ) : (
                  <Minus className="h-3 w-3 mr-1" />
                )}
                {forecast.trend_direction}
              </Badge>
            </div>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <span className="text-sm text-muted-foreground">Current Value</span>
                <span className="text-xl font-bold">{forecast.current_value.toFixed(1)}</span>
              </div>
              <div className="grid grid-cols-3 gap-2">
                {(["24h", "48h", "72h"] as const).map((time) => {
                  const f = forecast.forecasts[time];
                  return (
                    <div key={time} className="text-center p-2 rounded-md bg-muted/50">
                      <p className="text-xs text-muted-foreground">{time}</p>
                      <p className="text-sm font-semibold">{f.value.toFixed(1)}</p>
                      <p className="text-xs text-muted-foreground">
                        {f.confidence_interval.lower.toFixed(1)}-{f.confidence_interval.upper.toFixed(1)}
                      </p>
                    </div>
                  );
                })}
              </div>
            </div>
          </CardContent>
        </Card>
      ))}
    </div>
  );
}

function SegmentationPanel({ 
  segment, 
  isLoading 
}: { 
  segment?: PatientSegment;
  isLoading: boolean;
}) {
  if (isLoading) {
    return <Skeleton className="h-64 w-full" />;
  }

  if (!segment) {
    return (
      <Card>
        <CardContent className="py-12 text-center">
          <Users className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
          <h3 className="text-lg font-semibold mb-2">Patient Segmentation</h3>
          <p className="text-muted-foreground max-w-md mx-auto">
            K-Means clustering for patient phenotyping will identify similar patient 
            profiles and risk patterns. This feature requires population-level data analysis.
          </p>
          <Badge variant="secondary" className="mt-4">Coming Soon</Badge>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card data-testid="card-patient-segment">
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2">
            <Users className="h-5 w-5" />
            Patient Segment
          </CardTitle>
          <Badge className={getRiskBgColor(segment.risk_profile)}>
            {segment.risk_profile} Risk
          </Badge>
        </div>
        <CardDescription>{segment.description}</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          <div className="flex items-center justify-between p-3 rounded-lg bg-muted/50">
            <div>
              <p className="font-medium">{segment.segment_name}</p>
              <p className="text-sm text-muted-foreground">Segment ID: {segment.segment_id}</p>
            </div>
            <div className="text-right">
              <p className="text-sm text-muted-foreground">Membership Score</p>
              <p className="font-bold">{(segment.membership_score * 100).toFixed(0)}%</p>
            </div>
          </div>

          <div>
            <p className="text-sm font-medium mb-2">Segment Characteristics</p>
            <div className="flex flex-wrap gap-2">
              {segment.characteristics.map((char, i) => (
                <Badge key={i} variant="outline">{char}</Badge>
              ))}
            </div>
          </div>

          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <Users className="h-4 w-4" />
            <span>{segment.similar_patient_count} similar patients in this segment</span>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

export function MLInsightsPanel({ patientId, patientName, isDoctor = false }: MLInsightsPanelProps) {
  const { toast } = useToast();
  const [activeTab, setActiveTab] = useState("overview");

  const { data: diseaseRiskData, isLoading: diseaseRiskLoading, refetch: refetchDiseaseRisk } = useQuery<MLPredictionResponse>({
    queryKey: ['/api/ml/predict/disease-risk', patientId],
    queryFn: async () => {
      const res = await fetch(`/api/ml/predict/disease-risk/${patientId}`, { credentials: 'include' });
      if (!res.ok) return null;
      return res.json();
    },
    retry: 1,
    staleTime: 60000,
  });

  const { data: deteriorationData, isLoading: deteriorationLoading, refetch: refetchDeterioration } = useQuery<MLPredictionResponse>({
    queryKey: ['/api/ml/predict/deterioration', patientId],
    queryFn: async () => {
      const res = await fetch(`/api/ml/predict/deterioration/${patientId}`, { credentials: 'include' });
      if (!res.ok) return null;
      return res.json();
    },
    retry: 1,
    staleTime: 60000,
  });

  const { data: timeSeriesData, isLoading: timeSeriesLoading } = useQuery<{ forecasts: TimeSeriesForecast[] }>({
    queryKey: ['/api/ml/predict/time-series', patientId],
    queryFn: async () => {
      const res = await fetch(`/api/ml/predict/time-series/${patientId}`, { credentials: 'include' });
      if (!res.ok) return { forecasts: [] };
      return res.json();
    },
    retry: 1,
    staleTime: 60000,
  });

  const { data: segmentData, isLoading: segmentLoading } = useQuery<{ segment: PatientSegment }>({
    queryKey: ['/api/ml/predict/patient-segments', patientId],
    queryFn: async () => {
      const res = await fetch(`/api/ml/predict/patient-segments/${patientId}`, { credentials: 'include' });
      if (!res.ok) return { segment: null };
      return res.json();
    },
    retry: 1,
    staleTime: 300000,
  });

  const refreshAllMutation = useMutation({
    mutationFn: async () => {
      await Promise.all([
        refetchDiseaseRisk(),
        refetchDeterioration(),
      ]);
    },
    onSuccess: () => {
      toast({
        title: "Predictions Refreshed",
        description: "All ML predictions have been updated.",
      });
    },
  });

  const isLoading = diseaseRiskLoading || deteriorationLoading;

  return (
    <div className="space-y-6" data-testid="ml-insights-panel">
      <div className="flex items-center justify-between flex-wrap gap-4">
        <div>
          <h3 className="text-lg font-semibold" data-testid="heading-ml-insights">
            ML Insights
          </h3>
          <p className="text-sm text-muted-foreground">
            Comprehensive AI-powered health predictions for {patientName || 'this patient'}
          </p>
        </div>
        <Button
          variant="outline"
          size="sm"
          onClick={() => refreshAllMutation.mutate()}
          disabled={refreshAllMutation.isPending}
          data-testid="button-refresh-ml"
        >
          {refreshAllMutation.isPending ? (
            <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
          ) : (
            <Zap className="h-4 w-4 mr-2" />
          )}
          Refresh All
        </Button>
      </div>

      <OverviewRiskGauge
        diseaseRisks={diseaseRiskData?.predictions}
        deterioration={deteriorationData?.deterioration}
        isLoading={isLoading}
      />

      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="disease-risk" data-testid="tab-disease-risk">
            <Heart className="h-4 w-4 mr-2" />
            Disease Risk
          </TabsTrigger>
          <TabsTrigger value="deterioration" data-testid="tab-deterioration">
            <AlertTriangle className="h-4 w-4 mr-2" />
            Deterioration
          </TabsTrigger>
          <TabsTrigger value="time-series" data-testid="tab-time-series">
            <LineChart className="h-4 w-4 mr-2" />
            Forecasts
          </TabsTrigger>
          <TabsTrigger value="segmentation" data-testid="tab-segmentation">
            <Users className="h-4 w-4 mr-2" />
            Segmentation
          </TabsTrigger>
        </TabsList>

        <TabsContent value="disease-risk" className="mt-4">
          {diseaseRiskLoading ? (
            <div className="grid md:grid-cols-2 gap-4">
              <Skeleton className="h-48" />
              <Skeleton className="h-48" />
              <Skeleton className="h-48" />
              <Skeleton className="h-48" />
            </div>
          ) : diseaseRiskData?.predictions && Object.keys(diseaseRiskData.predictions).length > 0 ? (
            <div className="grid md:grid-cols-2 gap-4">
              {Object.entries(diseaseRiskData.predictions).map(([disease, prediction]) => (
                <DetailedPredictionCard
                  key={disease}
                  patientId={patientId}
                  disease={disease}
                  prediction={prediction}
                  onRefresh={() => refetchDiseaseRisk()}
                />
              ))}
            </div>
          ) : (
            <Card>
              <CardContent className="py-12 text-center">
                <Shield className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
                <h3 className="text-lg font-semibold mb-2">No Disease Risk Data</h3>
                <p className="text-muted-foreground max-w-md mx-auto">
                  Disease risk predictions require health data. Continue logging 
                  vitals and health information to enable personalized risk assessment.
                </p>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        <TabsContent value="deterioration" className="mt-4">
          <DeteriorationPanel
            deterioration={deteriorationData?.deterioration}
            isLoading={deteriorationLoading}
            onRefresh={() => refetchDeterioration()}
          />
        </TabsContent>

        <TabsContent value="time-series" className="mt-4">
          <TimeSeriesPanel
            forecasts={timeSeriesData?.forecasts}
            isLoading={timeSeriesLoading}
          />
        </TabsContent>

        <TabsContent value="segmentation" className="mt-4">
          <SegmentationPanel
            segment={segmentData?.segment}
            isLoading={segmentLoading}
          />
        </TabsContent>
      </Tabs>

      {isDoctor && (
        <Card className="border-dashed" data-testid="card-ml-model-info">
          <CardContent className="py-4">
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
              <Info className="h-4 w-4" />
              <span>
                ML models: Disease Risk (Logistic Regression + XGBoost), Deterioration (Ensemble), 
                Time-Series (LSTM), Segmentation (K-Means). Last updated: {diseaseRiskData?.predicted_at ? format(new Date(diseaseRiskData.predicted_at), "PPp") : "N/A"}
              </span>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
