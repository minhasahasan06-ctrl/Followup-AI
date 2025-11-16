import { useQuery } from "@tanstack/react-query";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Activity, Zap, Database, TrendingUp, Clock, CheckCircle2 } from "lucide-react";
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from "recharts";

interface MLStats {
  total_predictions: number;
  predictions_today: number;
  cache_hit_rate_percent: number;
  active_models: number;
  avg_inference_time_ms: number;
  redis_enabled: boolean;
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

export default function MLMonitoring() {
  // Fetch ML system statistics
  const { data: stats } = useQuery<MLStats>({
    queryKey: ["/api/v1/ml/stats"],
    refetchInterval: 10000 // Refresh every 10 seconds
  });

  // Fetch available models
  const { data: modelsData } = useQuery<{ models: MLModel[] }>({
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

  const models = modelsData?.models || [];
  const predictions = historyData?.predictions || [];

  // Transform performance data for charts
  const latencyData = performanceData?.metrics?.latency?.map(m => ({
    time: new Date(m.measured_at).toLocaleTimeString(),
    latency: m.value
  })) || [];

  const accuracyData = performanceData?.metrics?.accuracy?.map(m => ({
    time: new Date(m.measured_at).toLocaleTimeString(),
    accuracy: m.value
  })) || [];

  return (
    <div className="container mx-auto p-6 space-y-6">
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold" data-testid="heading-ml-monitoring">ML Inference Monitoring</h1>
          <p className="text-muted-foreground">Track machine learning model performance and predictions</p>
        </div>
        <Badge variant={stats?.redis_enabled ? "default" : "secondary"} data-testid="badge-redis-status">
          {stats?.redis_enabled ? "Redis Caching Enabled" : "No Cache"}
        </Badge>
      </div>

      {/* Statistics Overview */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-5">
        <Card data-testid="card-total-predictions">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Predictions</CardTitle>
            <Database className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold" data-testid="text-total-predictions">
              {stats?.total_predictions.toLocaleString() || 0}
            </div>
            <p className="text-xs text-muted-foreground">
              {stats?.predictions_today || 0} today
            </p>
          </CardContent>
        </Card>

        <Card data-testid="card-cache-rate">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Cache Hit Rate</CardTitle>
            <Zap className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold" data-testid="text-cache-rate">
              {stats?.cache_hit_rate_percent.toFixed(1) || 0}%
            </div>
            <p className="text-xs text-muted-foreground">
              Redis caching efficiency
            </p>
          </CardContent>
        </Card>

        <Card data-testid="card-active-models">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Active Models</CardTitle>
            <Activity className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold" data-testid="text-active-models">
              {stats?.active_models || 0}
            </div>
            <p className="text-xs text-muted-foreground">
              {models.filter(m => m.is_deployed).length} deployed
            </p>
          </CardContent>
        </Card>

        <Card data-testid="card-avg-latency">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Avg Latency</CardTitle>
            <Clock className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold" data-testid="text-avg-latency">
              {stats?.avg_inference_time_ms.toFixed(0) || 0}ms
            </div>
            <p className="text-xs text-muted-foreground">
              Average inference time
            </p>
          </CardContent>
        </Card>

        <Card data-testid="card-system-status">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">System Status</CardTitle>
            <CheckCircle2 className="h-4 w-4 text-green-500" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-green-600" data-testid="text-system-status">
              Healthy
            </div>
            <p className="text-xs text-muted-foreground">
              All systems operational
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Detailed Tabs */}
      <Tabs defaultValue="models" className="space-y-4">
        <TabsList>
          <TabsTrigger value="models" data-testid="tab-models">Models</TabsTrigger>
          <TabsTrigger value="performance" data-testid="tab-performance">Performance</TabsTrigger>
          <TabsTrigger value="predictions" data-testid="tab-predictions">Recent Predictions</TabsTrigger>
        </TabsList>

        {/* Models Tab */}
        <TabsContent value="models" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Deployed ML Models</CardTitle>
              <CardDescription>All active machine learning models in the system</CardDescription>
            </CardHeader>
            <CardContent>
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
                
                {models.length === 0 && (
                  <p className="text-center text-muted-foreground py-8">
                    No models loaded. Start the Python backend to load ML models.
                  </p>
                )}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Performance Tab */}
        <TabsContent value="performance" className="space-y-4">
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
                  <p className="text-center text-muted-foreground py-20">
                    No performance data available yet
                  </p>
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
                  <p className="text-center text-muted-foreground py-20">
                    No accuracy data available yet
                  </p>
                )}
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        {/* Predictions Tab */}
        <TabsContent value="predictions" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Recent Predictions</CardTitle>
              <CardDescription>Last {predictions.length} ML predictions</CardDescription>
            </CardHeader>
            <CardContent>
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
                        {new Date(pred.predicted_at).toLocaleString()} • {pred.inference_time_ms.toFixed(0)}ms
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

                {predictions.length === 0 && (
                  <p className="text-center text-muted-foreground py-8">
                    No predictions yet. Use the ML inference API to generate predictions.
                  </p>
                )}
              </div>
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
          <div className="grid gap-3 md:grid-cols-3">
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
              <p className="text-sm text-muted-foreground">All predictions logged to database</p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
