import { useState } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { Separator } from "@/components/ui/separator";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { ScrollArea } from "@/components/ui/scroll-area";
import { useToast } from "@/hooks/use-toast";
import { apiRequest, queryClient } from "@/lib/queryClient";
import {
  Brain,
  Stethoscope,
  TrendingUp,
  TrendingDown,
  AlertTriangle,
  Play,
  Loader2,
  Pill,
  HeartPulse,
  Wind,
  Thermometer,
  Activity,
  AlertCircle,
  CheckCircle2,
  Info,
  Zap,
  RefreshCw,
  Clock,
  FileText,
  BarChart3,
  LineChart as LineChartIcon,
  ArrowUp,
  ArrowDown,
  Minus
} from "lucide-react";
import {
  PieChart,
  Pie,
  Cell,
  ResponsiveContainer,
  Tooltip,
  Legend,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  AreaChart,
  Area
} from "recharts";
import { format } from "date-fns";

interface PatientMLToolsProps {
  patientId: string;
  patientName?: string;
}

interface SymptomAnalysisResult {
  symptoms: string[];
  conditions: string[];
  medications: string[];
  body_parts: string[];
  severity_indicators: string[];
  entities: { text: string; label: string; confidence: number }[];
  model_used: string;
  inference_time_ms: number;
}

interface DeteriorationResult {
  risk_score: number;
  risk_level: string;
  confidence: number;
  contributing_factors: { factor: string; weight: number; value: number }[];
  recommendations: string[];
  time_to_action: string;
  model_used: string;
  inference_time_ms: number;
}

interface TimeSeriesForecast {
  metric: string;
  historical: Array<{
    timestamp: string;
    value: number;
  }>;
  predictions: Array<{
    timestamp: string;
    value: number;
    lower_bound: number;
    upper_bound: number;
    confidence: number;
  }>;
  trend: string;
  anomalies: Array<{
    timestamp: string;
    value: number;
    z_score: number;
    severity: string;
  }>;
  model_used: string;
  mape: number;
}

interface TimeSeriesResponse {
  patient_id: string;
  forecasts: Record<string, TimeSeriesForecast>;
  predicted_at: string;
  horizon_hours: number;
  model_version: string;
}

const ENTITY_COLORS: Record<string, string> = {
  SYMPTOM: '#ef4444',
  CONDITION: '#f59e0b',
  MEDICATION: '#3b82f6',
  BODY_PART: '#8b5cf6',
  SEVERITY: '#ec4899',
  DEFAULT: '#6b7280'
};

export function PatientMLTools({ patientId, patientName }: PatientMLToolsProps) {
  const { toast } = useToast();
  
  // Symptom Analysis State
  const [symptomText, setSymptomText] = useState("");
  const [includeContext, setIncludeContext] = useState(true);
  const [analysisResult, setAnalysisResult] = useState<SymptomAnalysisResult | null>(null);

  // Deterioration Prediction State
  const [metrics, setMetrics] = useState({
    heart_rate: 72,
    blood_pressure_systolic: 120,
    blood_pressure_diastolic: 80,
    respiratory_rate: 16,
    temperature: 98.6,
    oxygen_saturation: 98,
    pain_level: 3,
    fatigue_score: 4,
  });
  const [deteriorationResult, setDeteriorationResult] = useState<DeteriorationResult | null>(null);
  const [forecastHorizon, setForecastHorizon] = useState(24);
  const [selectedMetric, setSelectedMetric] = useState<string>("heart_rate");

  // Fetch time-series forecast
  const { data: timeSeriesData, isLoading: timeSeriesLoading, refetch: refetchTimeSeries } = useQuery<TimeSeriesResponse>({
    queryKey: ['/api/ml/predict/time-series-forecast', patientId, forecastHorizon],
    queryFn: async () => {
      const res = await fetch(`/api/ml/predict/time-series-forecast/${patientId}?horizon_hours=${forecastHorizon}`, { credentials: 'include' });
      if (!res.ok) return null;
      return res.json();
    },
    retry: 1,
    staleTime: 120000, // 2 minutes
  });

  // Check if ML backend is available
  const { data: mlStats } = useQuery({
    queryKey: ['/api/v1/ml/stats'],
    refetchInterval: 30000,
  });

  const isBackendAvailable = mlStats?.backend_status !== 'unavailable';

  // Symptom Analysis Mutation
  const symptomAnalysisMutation = useMutation({
    mutationFn: async (data: { text: string; include_context: boolean }) => {
      const res = await apiRequest('/api/v1/ml/predict/symptom-analysis', {
        method: 'POST',
        json: { text: data.text, include_context: data.include_context, patient_id: patientId }
      });
      return res.json();
    },
    onSuccess: (data) => {
      setAnalysisResult(data);
      toast({
        title: "Analysis Complete",
        description: `Extracted ${data.entities?.length || 0} medical entities in ${data.inference_time_ms?.toFixed(0)}ms`,
      });
    },
    onError: () => {
      toast({
        title: "Analysis Failed",
        description: "Could not analyze symptoms. ML service may be unavailable.",
        variant: "destructive",
      });
    },
  });

  // Deterioration Prediction Mutation
  const deteriorationMutation = useMutation({
    mutationFn: async (data: { patient_id: string; metrics: typeof metrics }) => {
      const res = await apiRequest('/api/v1/ml/predict/deterioration', {
        method: 'POST',
        json: { patient_id: data.patient_id, metrics: data.metrics, time_window_days: 7 }
      });
      return res.json();
    },
    onSuccess: (data) => {
      setDeteriorationResult(data);
      toast({
        title: "Prediction Complete",
        description: `Risk level: ${data.risk_level} (${(data.confidence * 100).toFixed(0)}% confidence)`,
      });
    },
    onError: () => {
      toast({
        title: "Prediction Failed",
        description: "Could not predict deterioration risk. ML service may be unavailable.",
        variant: "destructive",
      });
    },
  });

  // Entity distribution for pie chart
  const entityDistribution = analysisResult?.entities?.reduce((acc, entity) => {
    acc[entity.label] = (acc[entity.label] || 0) + 1;
    return acc;
  }, {} as Record<string, number>) || {};

  const pieData = Object.entries(entityDistribution).map(([name, value]) => ({
    name,
    value,
    color: ENTITY_COLORS[name] || ENTITY_COLORS.DEFAULT
  }));

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

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between flex-wrap gap-4">
        <div>
          <h3 className="text-lg font-semibold" data-testid="heading-patient-ml-tools">
            ML Prediction Tools
          </h3>
          <p className="text-sm text-muted-foreground">
            AI-powered analysis and prediction for {patientName || 'this patient'}
          </p>
        </div>
        <Badge 
          variant={isBackendAvailable ? "default" : "destructive"} 
          data-testid="badge-ml-status"
        >
          {isBackendAvailable ? "ML Service Online" : "ML Service Offline"}
        </Badge>
      </div>

      {!isBackendAvailable && (
        <Alert variant="destructive" data-testid="alert-ml-offline">
          <AlertCircle className="h-4 w-4" />
          <AlertTitle>ML Service Unavailable</AlertTitle>
          <AlertDescription>
            The Python ML backend is not responding. Prediction tools will not work until the service is restored.
            This may be due to heavy ML models (TensorFlow, MediaPipe) taking time to load at startup.
          </AlertDescription>
        </Alert>
      )}

      <div className="grid gap-6 lg:grid-cols-2">
        {/* Symptom Analysis Tool */}
        <Card data-testid="card-symptom-analysis">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Stethoscope className="h-5 w-5" />
              Symptom Analysis
            </CardTitle>
            <CardDescription>
              Extract medical entities from clinical notes using Clinical-BERT NER
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="symptom-text">Clinical Notes / Patient Description</Label>
              <Textarea
                id="symptom-text"
                placeholder="Enter clinical notes or patient-reported symptoms. Example: 'Patient reports severe headache and fever of 101.5°F for 3 days. Mild nausea and fatigue. Currently taking ibuprofen 400mg twice daily.'"
                value={symptomText}
                onChange={(e) => setSymptomText(e.target.value)}
                rows={4}
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
                  Analyze for {patientName?.split(' ')[0] || 'Patient'}
                </>
              )}
            </Button>

            {analysisResult && (
              <div className="space-y-4 pt-4 border-t">
                <div className="flex items-center justify-between text-sm">
                  <span className="text-muted-foreground">Model: {analysisResult.model_used}</span>
                  <Badge variant="outline">{analysisResult.inference_time_ms?.toFixed(0)}ms</Badge>
                </div>

                {pieData.length > 0 && (
                  <div className="h-40">
                    <ResponsiveContainer width="100%" height="100%">
                      <PieChart>
                        <Pie
                          data={pieData}
                          cx="50%"
                          cy="50%"
                          innerRadius={30}
                          outerRadius={55}
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

                <div className="space-y-3">
                  {analysisResult.symptoms?.length > 0 && (
                    <div>
                      <Label className="text-xs uppercase text-muted-foreground">Symptoms</Label>
                      <div className="flex flex-wrap gap-1 mt-1">
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
                      <div className="flex flex-wrap gap-1 mt-1">
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
                      <div className="flex flex-wrap gap-1 mt-1">
                        {analysisResult.medications.map((med, i) => (
                          <Badge key={i} variant="secondary" style={{ backgroundColor: ENTITY_COLORS.MEDICATION + '20', color: ENTITY_COLORS.MEDICATION }}>
                            <Pill className="h-3 w-3 mr-1" />
                            {med}
                          </Badge>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Deterioration Prediction Tool */}
        <Card data-testid="card-deterioration-prediction">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <TrendingUp className="h-5 w-5" />
              Deterioration Prediction
            </CardTitle>
            <CardDescription>
              Predict health deterioration risk based on current metrics
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid gap-4 grid-cols-2">
              <div className="space-y-2">
                <Label className="flex items-center gap-2">
                  <HeartPulse className="h-4 w-4" />
                  Heart Rate (bpm)
                </Label>
                <div className="flex items-center gap-2">
                  <Slider
                    value={[metrics.heart_rate]}
                    onValueChange={([v]) => setMetrics(m => ({ ...m, heart_rate: v }))}
                    min={40}
                    max={180}
                    step={1}
                    className="flex-1"
                    data-testid="slider-heart-rate"
                  />
                  <span className="w-10 text-right text-sm font-medium">{metrics.heart_rate}</span>
                </div>
              </div>

              <div className="space-y-2">
                <Label className="flex items-center gap-2">
                  <Wind className="h-4 w-4" />
                  Respiratory Rate
                </Label>
                <div className="flex items-center gap-2">
                  <Slider
                    value={[metrics.respiratory_rate]}
                    onValueChange={([v]) => setMetrics(m => ({ ...m, respiratory_rate: v }))}
                    min={8}
                    max={40}
                    step={1}
                    className="flex-1"
                    data-testid="slider-respiratory-rate"
                  />
                  <span className="w-10 text-right text-sm font-medium">{metrics.respiratory_rate}</span>
                </div>
              </div>

              <div className="space-y-2">
                <Label className="flex items-center gap-2">
                  <Thermometer className="h-4 w-4" />
                  Temperature (°F)
                </Label>
                <Input
                  type="number"
                  value={metrics.temperature}
                  onChange={(e) => setMetrics(m => ({ ...m, temperature: Number(e.target.value) }))}
                  step={0.1}
                  data-testid="input-temperature"
                />
              </div>

              <div className="space-y-2">
                <Label className="flex items-center gap-2">
                  <Activity className="h-4 w-4" />
                  O2 Saturation (%)
                </Label>
                <div className="flex items-center gap-2">
                  <Slider
                    value={[metrics.oxygen_saturation]}
                    onValueChange={([v]) => setMetrics(m => ({ ...m, oxygen_saturation: v }))}
                    min={80}
                    max={100}
                    step={1}
                    className="flex-1"
                    data-testid="slider-oxygen"
                  />
                  <span className="w-10 text-right text-sm font-medium">{metrics.oxygen_saturation}</span>
                </div>
              </div>

              <div className="space-y-2">
                <Label>Pain Level (0-10)</Label>
                <div className="flex items-center gap-2">
                  <Slider
                    value={[metrics.pain_level]}
                    onValueChange={([v]) => setMetrics(m => ({ ...m, pain_level: v }))}
                    min={0}
                    max={10}
                    step={1}
                    className="flex-1"
                    data-testid="slider-pain"
                  />
                  <span className="w-10 text-right text-sm font-medium">{metrics.pain_level}</span>
                </div>
              </div>

              <div className="space-y-2">
                <Label>Fatigue Score (0-10)</Label>
                <div className="flex items-center gap-2">
                  <Slider
                    value={[metrics.fatigue_score]}
                    onValueChange={([v]) => setMetrics(m => ({ ...m, fatigue_score: v }))}
                    min={0}
                    max={10}
                    step={1}
                    className="flex-1"
                    data-testid="slider-fatigue"
                  />
                  <span className="w-10 text-right text-sm font-medium">{metrics.fatigue_score}</span>
                </div>
              </div>
            </div>

            <div className="space-y-2">
              <Label>Blood Pressure (mmHg)</Label>
              <div className="flex items-center gap-2">
                <Input
                  type="number"
                  value={metrics.blood_pressure_systolic}
                  onChange={(e) => setMetrics(m => ({ ...m, blood_pressure_systolic: Number(e.target.value) }))}
                  className="w-20"
                  data-testid="input-bp-systolic"
                />
                <span>/</span>
                <Input
                  type="number"
                  value={metrics.blood_pressure_diastolic}
                  onChange={(e) => setMetrics(m => ({ ...m, blood_pressure_diastolic: Number(e.target.value) }))}
                  className="w-20"
                  data-testid="input-bp-diastolic"
                />
              </div>
            </div>

            <Button
              onClick={() => deteriorationMutation.mutate({ patient_id: patientId, metrics })}
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
                  <Brain className="h-4 w-4 mr-2" />
                  Predict Risk for {patientName?.split(' ')[0] || 'Patient'}
                </>
              )}
            </Button>

            {deteriorationResult && (
              <div className="space-y-4 pt-4 border-t">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-muted-foreground">Risk Score</p>
                    <p className={`text-3xl font-bold ${getRiskColor(deteriorationResult.risk_level)}`} data-testid="text-risk-score">
                      {deteriorationResult.risk_score.toFixed(1)}
                    </p>
                  </div>
                  <div className="text-right">
                    <Badge className={getRiskBgColor(deteriorationResult.risk_level)} data-testid="badge-risk-level">
                      {deteriorationResult.risk_level} Risk
                    </Badge>
                    <p className="text-sm text-muted-foreground mt-1">
                      {(deteriorationResult.confidence * 100).toFixed(0)}% confidence
                    </p>
                  </div>
                </div>

                {deteriorationResult.time_to_action && (
                  <Alert>
                    <Clock className="h-4 w-4" />
                    <AlertTitle>Time to Action</AlertTitle>
                    <AlertDescription>{deteriorationResult.time_to_action}</AlertDescription>
                  </Alert>
                )}

                {deteriorationResult.contributing_factors?.length > 0 && (
                  <div>
                    <Label className="text-xs uppercase text-muted-foreground mb-2 block">Contributing Factors</Label>
                    <div className="space-y-2">
                      {deteriorationResult.contributing_factors.map((factor, i) => (
                        <div key={i} className="flex items-center justify-between p-2 rounded-lg border">
                          <span className="text-sm">{factor.factor}</span>
                          <div className="flex items-center gap-2">
                            <span className="text-sm text-muted-foreground">Value: {factor.value?.toFixed(1)}</span>
                            <Badge variant="outline">Weight: {(factor.weight * 100).toFixed(0)}%</Badge>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {deteriorationResult.recommendations?.length > 0 && (
                  <div>
                    <Label className="text-xs uppercase text-muted-foreground mb-2 block">Recommendations</Label>
                    <ul className="space-y-1 text-sm">
                      {deteriorationResult.recommendations.map((rec, i) => (
                        <li key={i} className="flex items-start gap-2">
                          <CheckCircle2 className="h-4 w-4 mt-0.5 text-green-500 flex-shrink-0" />
                          <span>{rec}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* LSTM Time-Series Forecast Section */}
      <Card data-testid="card-time-series-forecast">
        <CardHeader>
          <div className="flex items-center justify-between flex-wrap gap-4">
            <div>
              <CardTitle className="flex items-center gap-2">
                <LineChartIcon className="h-5 w-5" />
                LSTM Time-Series Forecast
              </CardTitle>
              <CardDescription>
                Deep learning vital sign predictions with confidence intervals
              </CardDescription>
            </div>
            <div className="flex items-center gap-2">
              <div className="flex items-center gap-2 border rounded-md p-1">
                {[12, 24, 48].map(hours => (
                  <Button
                    key={hours}
                    variant={forecastHorizon === hours ? "default" : "ghost"}
                    size="sm"
                    onClick={() => setForecastHorizon(hours)}
                    data-testid={`button-horizon-${hours}`}
                  >
                    {hours}h
                  </Button>
                ))}
              </div>
              <Button
                variant="outline"
                size="sm"
                onClick={() => refetchTimeSeries()}
                disabled={timeSeriesLoading}
                data-testid="button-refresh-forecast"
              >
                <RefreshCw className={`h-4 w-4 mr-2 ${timeSeriesLoading ? 'animate-spin' : ''}`} />
                Refresh
              </Button>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          {timeSeriesLoading ? (
            <div className="space-y-4">
              <Skeleton className="h-64 w-full" />
              <div className="grid gap-4 md:grid-cols-3">
                {[1, 2, 3].map(i => <Skeleton key={i} className="h-24" />)}
              </div>
            </div>
          ) : timeSeriesData?.forecasts && Object.keys(timeSeriesData.forecasts).length > 0 ? (
            <div className="space-y-6">
              {/* Metric Selector */}
              <div className="flex items-center gap-2 flex-wrap">
                {Object.keys(timeSeriesData.forecasts).map(metric => (
                  <Button
                    key={metric}
                    variant={selectedMetric === metric ? "default" : "outline"}
                    size="sm"
                    onClick={() => setSelectedMetric(metric)}
                    data-testid={`button-metric-${metric}`}
                    className="capitalize"
                  >
                    {metric === 'heart_rate' && <HeartPulse className="h-4 w-4 mr-1" />}
                    {metric === 'respiratory_rate' && <Wind className="h-4 w-4 mr-1" />}
                    {metric === 'oxygen_saturation' && <Activity className="h-4 w-4 mr-1" />}
                    {metric === 'blood_pressure' && <BarChart3 className="h-4 w-4 mr-1" />}
                    {metric.replace('_', ' ')}
                  </Button>
                ))}
              </div>

              {/* Selected Metric Chart */}
              {timeSeriesData.forecasts[selectedMetric] && (
                <div className="space-y-4">
                  <div className="h-64">
                    <ResponsiveContainer width="100%" height="100%">
                      <AreaChart
                        data={[
                          ...timeSeriesData.forecasts[selectedMetric].historical.map(h => ({
                            time: format(new Date(h.timestamp), 'MM/dd HH:mm'),
                            value: h.value,
                            type: 'historical'
                          })),
                          ...timeSeriesData.forecasts[selectedMetric].predictions.map(p => ({
                            time: format(new Date(p.timestamp), 'MM/dd HH:mm'),
                            predicted: p.value,
                            lower: p.lower_bound,
                            upper: p.upper_bound,
                            type: 'prediction'
                          }))
                        ]}
                      >
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="time" tick={{ fontSize: 11 }} />
                        <YAxis />
                        <Tooltip />
                        <Legend />
                        <Line 
                          type="monotone" 
                          dataKey="value" 
                          stroke="#3b82f6" 
                          strokeWidth={2}
                          dot={false}
                          name="Historical"
                        />
                        <Area
                          type="monotone"
                          dataKey="upper"
                          stroke="transparent"
                          fill="#10b981"
                          fillOpacity={0.1}
                          name="Upper Bound"
                        />
                        <Area
                          type="monotone"
                          dataKey="lower"
                          stroke="transparent"
                          fill="#10b981"
                          fillOpacity={0.1}
                          name="Lower Bound"
                        />
                        <Line 
                          type="monotone" 
                          dataKey="predicted" 
                          stroke="#10b981" 
                          strokeWidth={2}
                          strokeDasharray="5 5"
                          dot={false}
                          name="Predicted"
                        />
                      </AreaChart>
                    </ResponsiveContainer>
                  </div>

                  {/* Metric Summary Cards */}
                  <div className="grid gap-4 md:grid-cols-4">
                    <Card className="hover-elevate" data-testid="card-trend-indicator">
                      <CardContent className="pt-4">
                        <div className="flex items-center justify-between">
                          <div>
                            <p className="text-sm text-muted-foreground">Trend</p>
                            <div className="flex items-center gap-1 mt-1">
                              {timeSeriesData.forecasts[selectedMetric].trend === 'increasing' && (
                                <>
                                  <ArrowUp className="h-5 w-5 text-red-500" />
                                  <span className="font-medium text-red-500">Increasing</span>
                                </>
                              )}
                              {timeSeriesData.forecasts[selectedMetric].trend === 'decreasing' && (
                                <>
                                  <ArrowDown className="h-5 w-5 text-green-500" />
                                  <span className="font-medium text-green-500">Decreasing</span>
                                </>
                              )}
                              {timeSeriesData.forecasts[selectedMetric].trend === 'stable' && (
                                <>
                                  <Minus className="h-5 w-5 text-blue-500" />
                                  <span className="font-medium text-blue-500">Stable</span>
                                </>
                              )}
                            </div>
                          </div>
                        </div>
                      </CardContent>
                    </Card>

                    <Card className="hover-elevate" data-testid="card-model-accuracy">
                      <CardContent className="pt-4">
                        <div>
                          <p className="text-sm text-muted-foreground">Model Accuracy</p>
                          <p className="text-xl font-bold text-green-600">
                            {((1 - (timeSeriesData.forecasts[selectedMetric].mape || 0)) * 100).toFixed(1)}%
                          </p>
                          <p className="text-xs text-muted-foreground">MAPE: {((timeSeriesData.forecasts[selectedMetric].mape || 0) * 100).toFixed(2)}%</p>
                        </div>
                      </CardContent>
                    </Card>

                    <Card className="hover-elevate" data-testid="card-anomalies">
                      <CardContent className="pt-4">
                        <div>
                          <p className="text-sm text-muted-foreground">Anomalies Detected</p>
                          <p className="text-xl font-bold">
                            {timeSeriesData.forecasts[selectedMetric].anomalies?.length || 0}
                          </p>
                          {timeSeriesData.forecasts[selectedMetric].anomalies?.length > 0 && (
                            <Badge variant="destructive" className="mt-1">Requires Attention</Badge>
                          )}
                        </div>
                      </CardContent>
                    </Card>

                    <Card className="hover-elevate" data-testid="card-model-info">
                      <CardContent className="pt-4">
                        <div>
                          <p className="text-sm text-muted-foreground">Model</p>
                          <p className="font-medium">{timeSeriesData.forecasts[selectedMetric].model_used || 'LSTM-RNN'}</p>
                          <p className="text-xs text-muted-foreground mt-1">
                            Horizon: {timeSeriesData.horizon_hours}h
                          </p>
                        </div>
                      </CardContent>
                    </Card>
                  </div>

                  {/* Anomaly Details */}
                  {timeSeriesData.forecasts[selectedMetric].anomalies?.length > 0 && (
                    <div className="border-t pt-4">
                      <h4 className="font-medium mb-2 flex items-center gap-2">
                        <AlertTriangle className="h-4 w-4 text-yellow-500" />
                        Detected Anomalies
                      </h4>
                      <div className="space-y-2">
                        {timeSeriesData.forecasts[selectedMetric].anomalies.slice(0, 5).map((anomaly, i) => (
                          <div key={i} className="flex items-center justify-between p-2 rounded-lg border">
                            <div className="flex items-center gap-2">
                              <Badge 
                                variant={anomaly.severity === 'critical' ? 'destructive' : 'secondary'}
                              >
                                {anomaly.severity}
                              </Badge>
                              <span className="text-sm">
                                {format(new Date(anomaly.timestamp), 'MMM d, HH:mm')}
                              </span>
                            </div>
                            <div className="text-sm text-muted-foreground">
                              Value: {anomaly.value.toFixed(1)} | Z-Score: {anomaly.z_score.toFixed(2)}
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}

              {/* Model Version Footer */}
              <div className="flex items-center justify-between pt-4 border-t text-xs text-muted-foreground">
                <span>Model Version: {timeSeriesData.model_version}</span>
                <span>Last Predicted: {timeSeriesData.predicted_at ? format(new Date(timeSeriesData.predicted_at), 'MMM d, h:mm a') : 'N/A'}</span>
              </div>
            </div>
          ) : (
            <div className="text-center py-12 text-muted-foreground">
              <LineChartIcon className="h-12 w-12 mx-auto mb-3 opacity-50" />
              <p>No time-series forecast available</p>
              <p className="text-sm">Forecasts require sufficient historical vital sign data</p>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Disclaimer */}
      <Alert>
        <Info className="h-4 w-4" />
        <AlertTitle>AI Prediction Disclaimer</AlertTitle>
        <AlertDescription>
          These ML predictions are probabilistic estimates based on available data. They should be used as 
          supplementary clinical decision support tools, not as definitive diagnoses. Always correlate with 
          clinical examination, patient history, and professional medical judgment.
        </AlertDescription>
      </Alert>
    </div>
  );
}
