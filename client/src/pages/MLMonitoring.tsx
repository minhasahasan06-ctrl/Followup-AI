import { useState } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Textarea } from "@/components/ui/textarea";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Separator } from "@/components/ui/separator";
import { Skeleton } from "@/components/ui/skeleton";
import { 
  Activity, 
  Zap, 
  Database, 
  TrendingUp, 
  Clock, 
  CheckCircle2,
  Brain,
  Stethoscope,
  AlertTriangle,
  Play,
  Loader2,
  FileText,
  Pill,
  HeartPulse,
  Thermometer,
  Wind,
  Droplets,
  AlertCircle,
  RefreshCw,
  Beaker,
  Target
} from "lucide-react";
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, PieChart, Pie, Cell } from "recharts";
import { apiRequest, queryClient } from "@/lib/queryClient";
import { useToast } from "@/hooks/use-toast";

interface MLStats {
  total_predictions: number;
  predictions_today: number;
  cache_hit_rate_percent: number;
  active_models: number;
  avg_inference_time_ms: number;
  redis_enabled: boolean;
  backend_status?: string;
}

interface MLModel {
  id: number;
  name: string;
  version: string;
  type: string;
  task: string;
  is_deployed: boolean;
  metrics: any;
}

interface ModelPerformance {
  model_name: string;
  model_version: string;
  time_window_hours: number;
  metrics: Record<string, Array<{
    value: number;
    unit: string;
    measured_at: string;
    aggregation: string;
  }>>;
}

interface PredictionHistory {
  predictions: Array<{
    id: number;
    type: string;
    result: any;
    confidence: number;
    predicted_at: string;
    inference_time_ms: number;
    cache_hit: boolean;
  }>;
  total: number;
}

interface SymptomEntity {
  text: string;
  label: string;
  start: number;
  end: number;
  confidence: number;
}

interface SymptomAnalysisResult {
  entities: SymptomEntity[];
  symptoms: string[];
  conditions: string[];
  medications: string[];
  body_parts: string[];
  severity_indicators: string[];
  temporal_info: string[];
  inference_time_ms: number;
  model_used: string;
}

interface DeteriorationResult {
  risk_score: number;
  risk_level: string;
  confidence: number;
  contributing_factors: Array<{
    name: string;
    weight: number;
    status: string;
  }>;
  recommendations: string[];
  inference_time_ms: number;
}

const ENTITY_COLORS: Record<string, string> = {
  SYMPTOM: "#ef4444",
  CONDITION: "#f97316",
  MEDICATION: "#22c55e",
  BODY_PART: "#3b82f6",
  SEVERITY: "#a855f7",
  TEMPORAL: "#6366f1",
  DEFAULT: "#71717a"
};

export default function MLMonitoring() {
  const { toast } = useToast();
  const [activeTab, setActiveTab] = useState("overview");
  
  // Symptom Analysis state
  const [symptomText, setSymptomText] = useState("");
  const [includeContext, setIncludeContext] = useState(false);
  const [analysisResult, setAnalysisResult] = useState<SymptomAnalysisResult | null>(null);
  
  // Deterioration Prediction state
  const [deteriorationMetrics, setDeteriorationMetrics] = useState({
    heart_rate: 75,
    blood_pressure_systolic: 120,
    blood_pressure_diastolic: 80,
    respiratory_rate: 16,
    temperature: 98.6,
    oxygen_saturation: 98,
    pain_level: 3,
    fatigue_level: 4,
    sleep_quality: 7
  });
  const [deteriorationResult, setDeteriorationResult] = useState<DeteriorationResult | null>(null);
  
  // Generic Prediction state
  const [selectedModel, setSelectedModel] = useState<string>("");
  const [genericInput, setGenericInput] = useState("");
  const [genericResult, setGenericResult] = useState<any>(null);

  // Fetch ML system statistics
  const { data: stats, isLoading: statsLoading, refetch: refetchStats } = useQuery<MLStats>({
    queryKey: ["/api/v1/ml/stats"],
    refetchInterval: 10000
  });

  // Fetch available models
  const { data: modelsData, isLoading: modelsLoading } = useQuery<{ models: MLModel[] }>({
    queryKey: ["/api/v1/ml/models"]
  });

  // Fetch prediction history
  const { data: historyData } = useQuery<PredictionHistory>({
    queryKey: ["/api/v1/ml/predictions/history"],
    enabled: !!stats
  });

  // Fetch model performance (example for clinical_ner model)
  const { data: performanceData } = useQuery<ModelPerformance>({
    queryKey: ["/api/v1/ml/models/clinical_ner/performance"],
    enabled: !!modelsData?.models?.some(m => m.name === "clinical_ner")
  });

  // Symptom Analysis mutation
  const symptomAnalysisMutation = useMutation({
    mutationFn: async (data: { text: string; include_context: boolean }) => {
      return apiRequest("/api/v1/ml/predict/symptom-analysis", {
        method: "POST",
        body: JSON.stringify(data)
      });
    },
    onSuccess: (data: SymptomAnalysisResult) => {
      setAnalysisResult(data);
      toast({
        title: "Analysis Complete",
        description: `Found ${data.entities?.length || 0} entities in ${data.inference_time_ms?.toFixed(0) || 0}ms`,
      });
      queryClient.invalidateQueries({ queryKey: ["/api/v1/ml/predictions/history"] });
      queryClient.invalidateQueries({ queryKey: ["/api/v1/ml/stats"] });
    },
    onError: (error: any) => {
      toast({
        title: "Analysis Failed",
        description: error.message || "Failed to analyze symptoms. ML service may be unavailable.",
        variant: "destructive",
      });
    }
  });

  // Deterioration Prediction mutation
  const deteriorationMutation = useMutation({
    mutationFn: async (data: { metrics: typeof deteriorationMetrics }) => {
      return apiRequest("/api/v1/ml/predict/deterioration", {
        method: "POST",
        body: JSON.stringify(data)
      });
    },
    onSuccess: (data: DeteriorationResult) => {
      setDeteriorationResult(data);
      toast({
        title: "Prediction Complete",
        description: `Risk Level: ${data.risk_level} (Score: ${data.risk_score?.toFixed(1) || 0})`,
      });
      queryClient.invalidateQueries({ queryKey: ["/api/v1/ml/predictions/history"] });
      queryClient.invalidateQueries({ queryKey: ["/api/v1/ml/stats"] });
    },
    onError: (error: any) => {
      toast({
        title: "Prediction Failed",
        description: error.message || "Failed to predict deterioration risk. ML service may be unavailable.",
        variant: "destructive",
      });
    }
  });

  // Generic Prediction mutation
  const genericPredictionMutation = useMutation({
    mutationFn: async (data: { model: string; input: any }) => {
      return apiRequest("/api/v1/ml/predict", {
        method: "POST",
        body: JSON.stringify(data)
      });
    },
    onSuccess: (data: any) => {
      setGenericResult(data);
      toast({
        title: "Prediction Complete",
        description: `Inference time: ${data.inference_time_ms?.toFixed(0) || 0}ms`,
      });
      queryClient.invalidateQueries({ queryKey: ["/api/v1/ml/predictions/history"] });
      queryClient.invalidateQueries({ queryKey: ["/api/v1/ml/stats"] });
    },
    onError: (error: any) => {
      toast({
        title: "Prediction Failed",
        description: error.message || "Failed to run prediction. Check model and input.",
        variant: "destructive",
      });
    }
  });

  const models = modelsData?.models || [];
  const predictions = historyData?.predictions || [];
  const isBackendAvailable = stats?.backend_status !== 'unavailable';

  // Transform performance data for charts
  const latencyData = performanceData?.metrics?.latency?.map(m => ({
    time: new Date(m.measured_at).toLocaleTimeString(),
    latency: m.value
  })) || [];

  const accuracyData = performanceData?.metrics?.accuracy?.map(m => ({
    time: new Date(m.measured_at).toLocaleTimeString(),
    accuracy: m.value
  })) || [];

  // Entity type distribution for pie chart
  const entityDistribution = analysisResult?.entities?.reduce((acc, entity) => {
    acc[entity.label] = (acc[entity.label] || 0) + 1;
    return acc;
  }, {} as Record<string, number>) || {};

  const pieData = Object.entries(entityDistribution).map(([name, value]) => ({
    name,
    value,
    color: ENTITY_COLORS[name] || ENTITY_COLORS.DEFAULT
  }));

  // Risk level colors
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

  return (
    <div className="container mx-auto p-6 space-y-6">
      <div className="flex justify-between items-center flex-wrap gap-4">
        <div>
          <h1 className="text-3xl font-bold" data-testid="heading-ml-monitoring">ML Inference Dashboard</h1>
          <p className="text-muted-foreground">Machine learning model management and prediction tools</p>
        </div>
        <div className="flex items-center gap-2 flex-wrap">
          <Badge 
            variant={isBackendAvailable ? "default" : "destructive"} 
            data-testid="badge-backend-status"
          >
            {isBackendAvailable ? "Backend Online" : "Backend Offline"}
          </Badge>
          <Badge variant={stats?.redis_enabled ? "default" : "secondary"} data-testid="badge-redis-status">
            {stats?.redis_enabled ? "Redis Enabled" : "No Cache"}
          </Badge>
          <Button 
            variant="outline" 
            size="sm" 
            onClick={() => refetchStats()}
            data-testid="button-refresh-stats"
          >
            <RefreshCw className="h-4 w-4 mr-2" />
            Refresh
          </Button>
        </div>
      </div>

      {!isBackendAvailable && (
        <Alert variant="destructive" data-testid="alert-backend-offline">
          <AlertCircle className="h-4 w-4" />
          <AlertTitle>ML Backend Unavailable</AlertTitle>
          <AlertDescription>
            The Python ML backend is not responding. Prediction features will not work until the service is restored.
            This may be due to the heavy ML models (TensorFlow, MediaPipe) taking time to load at startup.
          </AlertDescription>
        </Alert>
      )}

      {/* Statistics Overview */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-5">
        <Card data-testid="card-total-predictions">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Predictions</CardTitle>
            <Database className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            {statsLoading ? (
              <Skeleton className="h-8 w-24" />
            ) : (
              <>
                <div className="text-2xl font-bold" data-testid="text-total-predictions">
                  {stats?.total_predictions?.toLocaleString() || 0}
                </div>
                <p className="text-xs text-muted-foreground">
                  {stats?.predictions_today || 0} today
                </p>
              </>
            )}
          </CardContent>
        </Card>

        <Card data-testid="card-cache-rate">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Cache Hit Rate</CardTitle>
            <Zap className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            {statsLoading ? (
              <Skeleton className="h-8 w-24" />
            ) : (
              <>
                <div className="text-2xl font-bold" data-testid="text-cache-rate">
                  {(stats?.cache_hit_rate_percent || 0).toFixed(1)}%
                </div>
                <p className="text-xs text-muted-foreground">
                  Redis caching efficiency
                </p>
              </>
            )}
          </CardContent>
        </Card>

        <Card data-testid="card-active-models">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Active Models</CardTitle>
            <Activity className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            {statsLoading ? (
              <Skeleton className="h-8 w-24" />
            ) : (
              <>
                <div className="text-2xl font-bold" data-testid="text-active-models">
                  {stats?.active_models || 0}
                </div>
                <p className="text-xs text-muted-foreground">
                  {models.filter(m => m.is_deployed).length} deployed
                </p>
              </>
            )}
          </CardContent>
        </Card>

        <Card data-testid="card-avg-latency">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Avg Latency</CardTitle>
            <Clock className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            {statsLoading ? (
              <Skeleton className="h-8 w-24" />
            ) : (
              <>
                <div className="text-2xl font-bold" data-testid="text-avg-latency">
                  {(stats?.avg_inference_time_ms || 0).toFixed(0)}ms
                </div>
                <p className="text-xs text-muted-foreground">
                  Average inference time
                </p>
              </>
            )}
          </CardContent>
        </Card>

        <Card data-testid="card-system-status">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">System Status</CardTitle>
            {isBackendAvailable ? (
              <CheckCircle2 className="h-4 w-4 text-green-500" />
            ) : (
              <AlertTriangle className="h-4 w-4 text-red-500" />
            )}
          </CardHeader>
          <CardContent>
            <div className={`text-2xl font-bold ${isBackendAvailable ? 'text-green-600' : 'text-red-600'}`} data-testid="text-system-status">
              {isBackendAvailable ? 'Healthy' : 'Offline'}
            </div>
            <p className="text-xs text-muted-foreground">
              {isBackendAvailable ? 'All systems operational' : 'Backend unavailable'}
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Main Tabs */}
      <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-4">
        <TabsList className="flex-wrap">
          <TabsTrigger value="overview" data-testid="tab-overview">
            <Activity className="h-4 w-4 mr-2" />
            Overview
          </TabsTrigger>
          <TabsTrigger value="symptom-analysis" data-testid="tab-symptom-analysis">
            <Stethoscope className="h-4 w-4 mr-2" />
            Symptom Analysis
          </TabsTrigger>
          <TabsTrigger value="deterioration" data-testid="tab-deterioration">
            <TrendingUp className="h-4 w-4 mr-2" />
            Deterioration Prediction
          </TabsTrigger>
          <TabsTrigger value="generic" data-testid="tab-generic">
            <Beaker className="h-4 w-4 mr-2" />
            Generic Prediction
          </TabsTrigger>
          <TabsTrigger value="models" data-testid="tab-models">
            <Brain className="h-4 w-4 mr-2" />
            Models
          </TabsTrigger>
          <TabsTrigger value="history" data-testid="tab-history">
            <Clock className="h-4 w-4 mr-2" />
            History
          </TabsTrigger>
        </TabsList>

        {/* Overview Tab */}
        <TabsContent value="overview" className="space-y-4">
          <div className="grid gap-4 md:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle>Inference Latency</CardTitle>
                <CardDescription>Average response time over last 24 hours</CardDescription>
              </CardHeader>
              <CardContent>
                {latencyData.length > 0 ? (
                  <ResponsiveContainer width="100%" height={300}>
                    <LineChart data={latencyData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="time" />
                      <YAxis />
                      <Tooltip />
                      <Legend />
                      <Line type="monotone" dataKey="latency" stroke="#8884d8" name="Latency (ms)" />
                    </LineChart>
                  </ResponsiveContainer>
                ) : (
                  <div className="flex flex-col items-center justify-center py-20 text-muted-foreground">
                    <Clock className="h-12 w-12 mb-4 opacity-50" />
                    <p>No performance data available yet</p>
                    <p className="text-sm">Run some predictions to generate metrics</p>
                  </div>
                )}
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Model Accuracy</CardTitle>
                <CardDescription>Prediction accuracy over time</CardDescription>
              </CardHeader>
              <CardContent>
                {accuracyData.length > 0 ? (
                  <ResponsiveContainer width="100%" height={300}>
                    <LineChart data={accuracyData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="time" />
                      <YAxis domain={[0, 100]} />
                      <Tooltip />
                      <Legend />
                      <Line type="monotone" dataKey="accuracy" stroke="#82ca9d" name="Accuracy (%)" />
                    </LineChart>
                  </ResponsiveContainer>
                ) : (
                  <div className="flex flex-col items-center justify-center py-20 text-muted-foreground">
                    <Target className="h-12 w-12 mb-4 opacity-50" />
                    <p>No accuracy data available yet</p>
                    <p className="text-sm">Run predictions to track accuracy</p>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        {/* Symptom Analysis Tab */}
        <TabsContent value="symptom-analysis" className="space-y-4">
          <div className="grid gap-6 lg:grid-cols-2">
            <Card data-testid="card-symptom-input">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Stethoscope className="h-5 w-5" />
                  Clinical-BERT Symptom Analysis
                </CardTitle>
                <CardDescription>
                  Extract medical entities from clinical text using NER (Named Entity Recognition)
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="symptom-text">Clinical Text</Label>
                  <Textarea
                    id="symptom-text"
                    placeholder="Enter clinical text to analyze. Example: 'Patient presents with severe headache and fever of 101.5°F for 3 days. Reports mild nausea and fatigue. Currently taking ibuprofen 400mg twice daily.'"
                    value={symptomText}
                    onChange={(e) => setSymptomText(e.target.value)}
                    rows={6}
                    className="resize-none"
                    data-testid="input-symptom-text"
                  />
                </div>

                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-2">
                    <Switch
                      id="include-context"
                      checked={includeContext}
                      onCheckedChange={setIncludeContext}
                      data-testid="switch-include-context"
                    />
                    <Label htmlFor="include-context">Include medical context</Label>
                  </div>
                </div>

                <Button
                  onClick={() => symptomAnalysisMutation.mutate({ text: symptomText, include_context: includeContext })}
                  disabled={!symptomText.trim() || symptomAnalysisMutation.isPending || !isBackendAvailable}
                  className="w-full"
                  data-testid="button-analyze-symptoms"
                >
                  {symptomAnalysisMutation.isPending ? (
                    <>
                      <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                      Analyzing...
                    </>
                  ) : (
                    <>
                      <Play className="h-4 w-4 mr-2" />
                      Analyze Symptoms
                    </>
                  )}
                </Button>

                {!isBackendAvailable && (
                  <p className="text-sm text-muted-foreground text-center">
                    ML backend must be online to run analysis
                  </p>
                )}
              </CardContent>
            </Card>

            <Card data-testid="card-symptom-results">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <FileText className="h-5 w-5" />
                  Analysis Results
                </CardTitle>
                <CardDescription>
                  Extracted medical entities and classifications
                </CardDescription>
              </CardHeader>
              <CardContent>
                {analysisResult ? (
                  <div className="space-y-4">
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-muted-foreground">Model: {analysisResult.model_used}</span>
                      <Badge variant="outline">{analysisResult.inference_time_ms?.toFixed(0)}ms</Badge>
                    </div>

                    <Separator />

                    {/* Entity Distribution Chart */}
                    {pieData.length > 0 && (
                      <div className="h-48">
                        <ResponsiveContainer width="100%" height="100%">
                          <PieChart>
                            <Pie
                              data={pieData}
                              cx="50%"
                              cy="50%"
                              innerRadius={40}
                              outerRadius={70}
                              paddingAngle={2}
                              dataKey="value"
                            >
                              {pieData.map((entry, index) => (
                                <Cell key={`cell-${index}`} fill={entry.color} />
                              ))}
                            </Pie>
                            <Tooltip />
                            <Legend />
                          </PieChart>
                        </ResponsiveContainer>
                      </div>
                    )}

                    <Separator />

                    {/* Entity Tags */}
                    <div className="space-y-3">
                      {analysisResult.symptoms?.length > 0 && (
                        <div>
                          <Label className="text-xs uppercase text-muted-foreground">Symptoms</Label>
                          <div className="flex flex-wrap gap-2 mt-1">
                            {analysisResult.symptoms.map((symptom, i) => (
                              <Badge key={i} variant="secondary" style={{ backgroundColor: ENTITY_COLORS.SYMPTOM + '20', color: ENTITY_COLORS.SYMPTOM }}>
                                {symptom}
                              </Badge>
                            ))}
                          </div>
                        </div>
                      )}

                      {analysisResult.conditions?.length > 0 && (
                        <div>
                          <Label className="text-xs uppercase text-muted-foreground">Conditions</Label>
                          <div className="flex flex-wrap gap-2 mt-1">
                            {analysisResult.conditions.map((condition, i) => (
                              <Badge key={i} variant="secondary" style={{ backgroundColor: ENTITY_COLORS.CONDITION + '20', color: ENTITY_COLORS.CONDITION }}>
                                {condition}
                              </Badge>
                            ))}
                          </div>
                        </div>
                      )}

                      {analysisResult.medications?.length > 0 && (
                        <div>
                          <Label className="text-xs uppercase text-muted-foreground">Medications</Label>
                          <div className="flex flex-wrap gap-2 mt-1">
                            {analysisResult.medications.map((med, i) => (
                              <Badge key={i} variant="secondary" style={{ backgroundColor: ENTITY_COLORS.MEDICATION + '20', color: ENTITY_COLORS.MEDICATION }}>
                                <Pill className="h-3 w-3 mr-1" />
                                {med}
                              </Badge>
                            ))}
                          </div>
                        </div>
                      )}

                      {analysisResult.body_parts?.length > 0 && (
                        <div>
                          <Label className="text-xs uppercase text-muted-foreground">Body Parts</Label>
                          <div className="flex flex-wrap gap-2 mt-1">
                            {analysisResult.body_parts.map((part, i) => (
                              <Badge key={i} variant="secondary" style={{ backgroundColor: ENTITY_COLORS.BODY_PART + '20', color: ENTITY_COLORS.BODY_PART }}>
                                {part}
                              </Badge>
                            ))}
                          </div>
                        </div>
                      )}

                      {analysisResult.severity_indicators?.length > 0 && (
                        <div>
                          <Label className="text-xs uppercase text-muted-foreground">Severity</Label>
                          <div className="flex flex-wrap gap-2 mt-1">
                            {analysisResult.severity_indicators.map((sev, i) => (
                              <Badge key={i} variant="secondary" style={{ backgroundColor: ENTITY_COLORS.SEVERITY + '20', color: ENTITY_COLORS.SEVERITY }}>
                                {sev}
                              </Badge>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                ) : (
                  <div className="flex flex-col items-center justify-center py-12 text-muted-foreground">
                    <Stethoscope className="h-12 w-12 mb-4 opacity-50" />
                    <p>No analysis results yet</p>
                    <p className="text-sm">Enter text and click Analyze to extract medical entities</p>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        {/* Deterioration Prediction Tab */}
        <TabsContent value="deterioration" className="space-y-4">
          <div className="grid gap-6 lg:grid-cols-2">
            <Card data-testid="card-deterioration-input">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <TrendingUp className="h-5 w-5" />
                  Patient Metrics Input
                </CardTitle>
                <CardDescription>
                  Enter current patient vitals and wellness metrics for risk assessment
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="grid gap-4 sm:grid-cols-2">
                  {/* Vital Signs */}
                  <div className="space-y-2">
                    <Label className="flex items-center gap-2">
                      <HeartPulse className="h-4 w-4" />
                      Heart Rate (bpm)
                    </Label>
                    <div className="flex items-center gap-4">
                      <Slider
                        value={[deteriorationMetrics.heart_rate]}
                        onValueChange={([v]) => setDeteriorationMetrics(m => ({ ...m, heart_rate: v }))}
                        min={40}
                        max={180}
                        step={1}
                        className="flex-1"
                        data-testid="slider-heart-rate"
                      />
                      <span className="w-12 text-right font-medium">{deteriorationMetrics.heart_rate}</span>
                    </div>
                  </div>

                  <div className="space-y-2">
                    <Label className="flex items-center gap-2">
                      <Activity className="h-4 w-4" />
                      Blood Pressure (mmHg)
                    </Label>
                    <div className="flex items-center gap-2">
                      <Input
                        type="number"
                        value={deteriorationMetrics.blood_pressure_systolic}
                        onChange={(e) => setDeteriorationMetrics(m => ({ ...m, blood_pressure_systolic: Number(e.target.value) }))}
                        className="w-20"
                        data-testid="input-bp-systolic"
                      />
                      <span>/</span>
                      <Input
                        type="number"
                        value={deteriorationMetrics.blood_pressure_diastolic}
                        onChange={(e) => setDeteriorationMetrics(m => ({ ...m, blood_pressure_diastolic: Number(e.target.value) }))}
                        className="w-20"
                        data-testid="input-bp-diastolic"
                      />
                    </div>
                  </div>

                  <div className="space-y-2">
                    <Label className="flex items-center gap-2">
                      <Wind className="h-4 w-4" />
                      Respiratory Rate (/min)
                    </Label>
                    <div className="flex items-center gap-4">
                      <Slider
                        value={[deteriorationMetrics.respiratory_rate]}
                        onValueChange={([v]) => setDeteriorationMetrics(m => ({ ...m, respiratory_rate: v }))}
                        min={8}
                        max={40}
                        step={1}
                        className="flex-1"
                        data-testid="slider-respiratory-rate"
                      />
                      <span className="w-12 text-right font-medium">{deteriorationMetrics.respiratory_rate}</span>
                    </div>
                  </div>

                  <div className="space-y-2">
                    <Label className="flex items-center gap-2">
                      <Thermometer className="h-4 w-4" />
                      Temperature (°F)
                    </Label>
                    <div className="flex items-center gap-4">
                      <Slider
                        value={[deteriorationMetrics.temperature]}
                        onValueChange={([v]) => setDeteriorationMetrics(m => ({ ...m, temperature: v }))}
                        min={95}
                        max={105}
                        step={0.1}
                        className="flex-1"
                        data-testid="slider-temperature"
                      />
                      <span className="w-16 text-right font-medium">{deteriorationMetrics.temperature.toFixed(1)}</span>
                    </div>
                  </div>

                  <div className="space-y-2">
                    <Label className="flex items-center gap-2">
                      <Droplets className="h-4 w-4" />
                      Oxygen Saturation (%)
                    </Label>
                    <div className="flex items-center gap-4">
                      <Slider
                        value={[deteriorationMetrics.oxygen_saturation]}
                        onValueChange={([v]) => setDeteriorationMetrics(m => ({ ...m, oxygen_saturation: v }))}
                        min={70}
                        max={100}
                        step={1}
                        className="flex-1"
                        data-testid="slider-oxygen"
                      />
                      <span className="w-12 text-right font-medium">{deteriorationMetrics.oxygen_saturation}%</span>
                    </div>
                  </div>

                  <div className="space-y-2">
                    <Label className="flex items-center gap-2">
                      <AlertTriangle className="h-4 w-4" />
                      Pain Level (0-10)
                    </Label>
                    <div className="flex items-center gap-4">
                      <Slider
                        value={[deteriorationMetrics.pain_level]}
                        onValueChange={([v]) => setDeteriorationMetrics(m => ({ ...m, pain_level: v }))}
                        min={0}
                        max={10}
                        step={1}
                        className="flex-1"
                        data-testid="slider-pain"
                      />
                      <span className="w-12 text-right font-medium">{deteriorationMetrics.pain_level}</span>
                    </div>
                  </div>
                </div>

                <Separator />

                <Button
                  onClick={() => deteriorationMutation.mutate({ metrics: deteriorationMetrics })}
                  disabled={deteriorationMutation.isPending || !isBackendAvailable}
                  className="w-full"
                  data-testid="button-predict-deterioration"
                >
                  {deteriorationMutation.isPending ? (
                    <>
                      <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                      Predicting...
                    </>
                  ) : (
                    <>
                      <TrendingUp className="h-4 w-4 mr-2" />
                      Predict Deterioration Risk
                    </>
                  )}
                </Button>
              </CardContent>
            </Card>

            <Card data-testid="card-deterioration-results">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Target className="h-5 w-5" />
                  Risk Assessment Results
                </CardTitle>
                <CardDescription>
                  ML-predicted deterioration risk and contributing factors
                </CardDescription>
              </CardHeader>
              <CardContent>
                {deteriorationResult ? (
                  <div className="space-y-6">
                    {/* Risk Score Display */}
                    <div className={`p-6 rounded-lg text-center ${getRiskBgColor(deteriorationResult.risk_level)}`}>
                      <div className={`text-4xl font-bold ${getRiskColor(deteriorationResult.risk_level)}`} data-testid="text-risk-score">
                        {deteriorationResult.risk_score?.toFixed(1)}
                      </div>
                      <div className={`text-lg font-medium mt-1 ${getRiskColor(deteriorationResult.risk_level)}`}>
                        {deteriorationResult.risk_level} Risk
                      </div>
                      <div className="text-sm text-muted-foreground mt-2">
                        Confidence: {(deteriorationResult.confidence * 100).toFixed(0)}%
                      </div>
                    </div>

                    <Separator />

                    {/* Contributing Factors */}
                    {deteriorationResult.contributing_factors?.length > 0 && (
                      <div className="space-y-3">
                        <Label className="text-sm font-medium">Contributing Factors</Label>
                        {deteriorationResult.contributing_factors.map((factor, i) => (
                          <div key={i} className="flex items-center justify-between p-2 rounded border">
                            <span className="font-medium">{factor.name}</span>
                            <div className="flex items-center gap-2">
                              <Badge variant={factor.status === 'normal' ? 'secondary' : factor.status === 'elevated' ? 'default' : 'destructive'}>
                                {factor.status}
                              </Badge>
                              <span className="text-sm text-muted-foreground">
                                Weight: {(factor.weight * 100).toFixed(0)}%
                              </span>
                            </div>
                          </div>
                        ))}
                      </div>
                    )}

                    {/* Recommendations */}
                    {deteriorationResult.recommendations?.length > 0 && (
                      <div className="space-y-2">
                        <Label className="text-sm font-medium">Recommendations</Label>
                        <ul className="space-y-1">
                          {deteriorationResult.recommendations.map((rec, i) => (
                            <li key={i} className="text-sm text-muted-foreground flex items-start gap-2">
                              <CheckCircle2 className="h-4 w-4 text-green-500 mt-0.5 flex-shrink-0" />
                              {rec}
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}

                    <div className="text-xs text-muted-foreground text-center">
                      Inference time: {deteriorationResult.inference_time_ms?.toFixed(0)}ms
                    </div>
                  </div>
                ) : (
                  <div className="flex flex-col items-center justify-center py-12 text-muted-foreground">
                    <TrendingUp className="h-12 w-12 mb-4 opacity-50" />
                    <p>No prediction results yet</p>
                    <p className="text-sm">Adjust metrics and click Predict to assess risk</p>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        {/* Generic Prediction Tab */}
        <TabsContent value="generic" className="space-y-4">
          <Card data-testid="card-generic-prediction">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Beaker className="h-5 w-5" />
                Generic ML Prediction Interface
              </CardTitle>
              <CardDescription>
                Select a model and provide input data for testing predictions
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid gap-4 md:grid-cols-2">
                <div className="space-y-2">
                  <Label htmlFor="model-select">Select Model</Label>
                  <Select value={selectedModel} onValueChange={setSelectedModel}>
                    <SelectTrigger data-testid="select-model">
                      <SelectValue placeholder="Choose a model..." />
                    </SelectTrigger>
                    <SelectContent>
                      {models.map((model) => (
                        <SelectItem key={model.id} value={model.name} data-testid={`option-model-${model.name}`}>
                          {model.name} (v{model.version})
                        </SelectItem>
                      ))}
                      {models.length === 0 && (
                        <SelectItem value="none" disabled>No models available</SelectItem>
                      )}
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-2">
                  <Label>Model Info</Label>
                  {selectedModel && models.find(m => m.name === selectedModel) ? (
                    <div className="p-3 rounded-lg border text-sm">
                      <p><strong>Type:</strong> {models.find(m => m.name === selectedModel)?.type}</p>
                      <p><strong>Task:</strong> {models.find(m => m.name === selectedModel)?.task}</p>
                      <p><strong>Status:</strong> {models.find(m => m.name === selectedModel)?.is_deployed ? 'Deployed' : 'Not Deployed'}</p>
                    </div>
                  ) : (
                    <div className="p-3 rounded-lg border text-sm text-muted-foreground">
                      Select a model to see details
                    </div>
                  )}
                </div>
              </div>

              <div className="space-y-2">
                <Label htmlFor="generic-input">Input Data (JSON)</Label>
                <Textarea
                  id="generic-input"
                  placeholder='{"text": "your input data here"}'
                  value={genericInput}
                  onChange={(e) => setGenericInput(e.target.value)}
                  rows={4}
                  className="font-mono text-sm"
                  data-testid="input-generic-data"
                />
              </div>

              <Button
                onClick={() => {
                  try {
                    const input = JSON.parse(genericInput || '{}');
                    genericPredictionMutation.mutate({ model: selectedModel, input });
                  } catch (e) {
                    toast({
                      title: "Invalid JSON",
                      description: "Please enter valid JSON input data",
                      variant: "destructive",
                    });
                  }
                }}
                disabled={!selectedModel || !genericInput.trim() || genericPredictionMutation.isPending || !isBackendAvailable}
                data-testid="button-run-prediction"
              >
                {genericPredictionMutation.isPending ? (
                  <>
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                    Running...
                  </>
                ) : (
                  <>
                    <Play className="h-4 w-4 mr-2" />
                    Run Prediction
                  </>
                )}
              </Button>

              {genericResult && (
                <div className="space-y-2">
                  <Label>Prediction Result</Label>
                  <pre className="p-4 rounded-lg bg-muted overflow-auto text-sm max-h-64" data-testid="text-generic-result">
                    {JSON.stringify(genericResult, null, 2)}
                  </pre>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Models Tab */}
        <TabsContent value="models" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Deployed ML Models</CardTitle>
              <CardDescription>All machine learning models registered in the system</CardDescription>
            </CardHeader>
            <CardContent>
              {modelsLoading ? (
                <div className="space-y-4">
                  {[1, 2, 3].map((i) => (
                    <Skeleton key={i} className="h-20" />
                  ))}
                </div>
              ) : models.length > 0 ? (
                <div className="space-y-4">
                  {models.map((model) => (
                    <div
                      key={model.id}
                      className="flex items-center justify-between p-4 border rounded-lg hover-elevate"
                      data-testid={`model-card-${model.name}`}
                    >
                      <div className="space-y-1">
                        <div className="flex items-center gap-2">
                          <h3 className="font-semibold" data-testid={`text-model-name-${model.name}`}>
                            {model.name}
                          </h3>
                          <Badge variant={model.is_deployed ? "default" : "secondary"}>
                            v{model.version}
                          </Badge>
                        </div>
                        <p className="text-sm text-muted-foreground">
                          {model.type} • {model.task}
                        </p>
                        {model.metrics && (
                          <p className="text-xs text-muted-foreground">
                            Accuracy: {(model.metrics.accuracy * 100).toFixed(1)}%
                          </p>
                        )}
                      </div>
                      <Badge variant={model.is_deployed ? "default" : "outline"}>
                        {model.is_deployed ? "Deployed" : "Inactive"}
                      </Badge>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="flex flex-col items-center justify-center py-12 text-muted-foreground">
                  <Brain className="h-12 w-12 mb-4 opacity-50" />
                  <p>No models loaded</p>
                  <p className="text-sm">Start the Python backend to load ML models</p>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* History Tab */}
        <TabsContent value="history" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Recent Predictions</CardTitle>
              <CardDescription>Last {predictions.length} ML predictions</CardDescription>
            </CardHeader>
            <CardContent>
              {predictions.length > 0 ? (
                <div className="space-y-3">
                  {predictions.map((pred) => (
                    <div
                      key={pred.id}
                      className="flex items-center justify-between p-3 border rounded-lg hover-elevate"
                      data-testid={`prediction-${pred.id}`}
                    >
                      <div className="space-y-1 flex-1">
                        <div className="flex items-center gap-2">
                          <span className="font-medium">{pred.type}</span>
                          {pred.cache_hit && (
                            <Badge variant="secondary" className="text-xs">
                              <Zap className="h-3 w-3 mr-1" />
                              Cached
                            </Badge>
                          )}
                        </div>
                        <p className="text-xs text-muted-foreground">
                          {new Date(pred.predicted_at).toLocaleString()} • {pred.inference_time_ms?.toFixed(0)}ms
                        </p>
                      </div>
                      <div className="text-right">
                        {pred.confidence && (
                          <p className="text-sm font-medium">
                            {(pred.confidence * 100).toFixed(1)}% confident
                          </p>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="flex flex-col items-center justify-center py-12 text-muted-foreground">
                  <Clock className="h-12 w-12 mb-4 opacity-50" />
                  <p>No predictions yet</p>
                  <p className="text-sm">Use the prediction tools to generate history</p>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      {/* System Information */}
      <Card>
        <CardHeader>
          <CardTitle>System Configuration</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid gap-3 md:grid-cols-4">
            <div>
              <p className="text-sm font-medium">Redis Caching</p>
              <p className="text-sm text-muted-foreground">
                {stats?.redis_enabled ? "Enabled (10-100x speedup)" : "Disabled"}
              </p>
            </div>
            <div>
              <p className="text-sm font-medium">Async Inference</p>
              <p className="text-sm text-muted-foreground">Thread pool with 4 workers</p>
            </div>
            <div>
              <p className="text-sm font-medium">HIPAA Audit Logging</p>
              <p className="text-sm text-muted-foreground">All predictions logged</p>
            </div>
            <div>
              <p className="text-sm font-medium">Backend Status</p>
              <p className={`text-sm ${isBackendAvailable ? 'text-green-600' : 'text-red-600'}`}>
                {isBackendAvailable ? 'Online' : 'Offline'}
              </p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
