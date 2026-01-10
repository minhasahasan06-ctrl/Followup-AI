import { useState } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Skeleton } from "@/components/ui/skeleton";
import { ScrollArea } from "@/components/ui/scroll-area";
import { 
  Activity, 
  Brain,
  Shield,
  TrendingUp,
  TrendingDown,
  AlertTriangle,
  CheckCircle2,
  Lock,
  FileText,
  Users,
  BarChart3,
  RefreshCw,
  Loader2,
  Search,
  Eye,
  Clock,
  Hash
} from "lucide-react";
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, PieChart, Pie, Cell } from "recharts";
import { apiRequest, queryClient } from "@/lib/queryClient";
import { useToast } from "@/hooks/use-toast";
import { useAuth } from "@/contexts/AuthContext";

interface TinkerHealth {
  enabled: boolean;
  api_connected: boolean;
  privacy_firewall_active: boolean;
  k_anonymity_threshold: number;
  circuit_breaker_state: string;
  uptime_hours: number;
  last_request_timestamp: string | null;
  version: string;
}

interface CohortAnalysisResult {
  cohort_hash: string;
  sample_size: number;
  k_anonymity_verified: boolean;
  insights: {
    risk_distribution: Record<string, number>;
    age_buckets: Record<string, number>;
    top_conditions: string[];
    feature_importance?: Record<string, number>;
  };
  privacy_metadata: {
    phi_fields_stripped: number;
    hash_salt_applied: boolean;
    timestamp: string;
  };
}

interface DriftCheckResult {
  drift_detected: boolean;
  drift_score: number;
  psi_values: Record<string, number>;
  affected_features: string[];
  recommendations: string[];
  timestamp: string;
}

interface DriftStatus {
  enabled: boolean;
  drift_detected: boolean;
  last_check: string | null;
  active_alerts?: number;
  recent_runs: Array<{
    id: number;
    model_id: string;
    drift_detected: boolean;
    psi_score: number | null;
    created_at: string | null;
  }>;
}

interface PrivacyStats {
  total_requests: number;
  phi_detections_blocked: number;
  k_anonymity_rejections: number;
  successful_analyses: number;
  privacy_score: number;
  last_audit_timestamp: string | null;
}

interface AuditLogEntry {
  id: number;
  action: string;
  user_hash: string;
  timestamp: string;
  k_anon_verified: boolean;
  success: boolean;
  details: Record<string, unknown>;
}

const DRIFT_COLORS = {
  low: "#22c55e",
  medium: "#f97316",
  high: "#ef4444"
};

const PIE_COLORS = ["#0d9488", "#14b8a6", "#2dd4bf", "#5eead4", "#99f6e4"];

export default function TinkerDashboard() {
  const { toast } = useToast();
  const { user } = useAuth();
  const [activeTab, setActiveTab] = useState("overview");
  const [cohortConditions, setCohortConditions] = useState("");
  const [analysisResult, setAnalysisResult] = useState<CohortAnalysisResult | null>(null);

  const isAdminOrDoctor = user?.role === "admin" || user?.role === "doctor";

  const { data: health, isLoading: healthLoading, isError: healthError, refetch: refetchHealth } = useQuery<TinkerHealth>({
    queryKey: ["/api/v1/tinker/health"],
    refetchInterval: 30000,
    retry: 1
  });

  const { data: privacyStats, isLoading: privacyLoading, isError: privacyError } = useQuery<PrivacyStats>({
    queryKey: ["/api/v1/tinker/privacy/stats"],
    enabled: isAdminOrDoctor && !!health,
    retry: 1
  });

  const { data: auditLogs, isLoading: auditLoading, isError: auditError } = useQuery<{ entries: AuditLogEntry[], total: number }>({
    queryKey: ["/api/v1/tinker/audits"],
    enabled: isAdminOrDoctor && !!health,
    retry: 1
  });

  const { data: driftStatus, isLoading: driftLoading, isError: driftError, refetch: refetchDrift } = useQuery<DriftStatus>({
    queryKey: ["/api/v1/tinker/drift/status"],
    enabled: health?.enabled ?? false,
    retry: 1
  });

  const cohortMutation = useMutation({
    mutationFn: async (conditions: string[]) => {
      const response = await apiRequest("/api/v1/tinker/cohort/analyze", { 
        method: "POST",
        json: { conditions, include_risk_distribution: true }
      });
      if (!response.ok) {
        throw new Error("Cohort analysis failed");
      }
      return response.json() as Promise<CohortAnalysisResult>;
    },
    onSuccess: (data) => {
      setAnalysisResult(data);
      toast({
        title: "Cohort Analysis Complete",
        description: `Analyzed ${data.sample_size} records with k-anonymity verified`,
      });
    },
    onError: (error: Error) => {
      toast({
        title: "Analysis Failed",
        description: error.message,
        variant: "destructive"
      });
    }
  });

  const handleCohortAnalysis = () => {
    const conditions = cohortConditions.split(",").map(c => c.trim()).filter(Boolean);
    if (conditions.length === 0) {
      toast({
        title: "Input Required",
        description: "Please enter at least one condition to analyze",
        variant: "destructive"
      });
      return;
    }
    cohortMutation.mutate(conditions);
  };

  const getDriftSeverity = (score: number) => {
    if (score < 0.1) return "low";
    if (score < 0.25) return "medium";
    return "high";
  };

  if (healthLoading) {
    return (
      <div className="p-6 space-y-6">
        <Skeleton className="h-10 w-64" />
        <div className="grid gap-4 md:grid-cols-4">
          {[1, 2, 3, 4].map((i) => (
            <Skeleton key={i} className="h-32" />
          ))}
        </div>
      </div>
    );
  }

  if (healthError) {
    return (
      <div className="p-6 space-y-6">
        <div className="flex items-center gap-3">
          <Brain className="h-8 w-8 text-primary" />
          <h1 className="text-3xl font-bold">Tinker Thinking Machine</h1>
        </div>
        <Alert variant="destructive">
          <AlertTriangle className="h-4 w-4" />
          <AlertTitle>Connection Error</AlertTitle>
          <AlertDescription>
            Unable to connect to Tinker service. Please check your network connection and try again.
          </AlertDescription>
        </Alert>
        <Button onClick={() => refetchHealth()} data-testid="button-retry-health">
          <RefreshCw className="h-4 w-4 mr-2" />
          Retry Connection
        </Button>
      </div>
    );
  }

  if (!health?.enabled) {
    return (
      <div className="p-6 space-y-6">
        <div className="flex items-center gap-3">
          <Brain className="h-8 w-8 text-primary" />
          <h1 className="text-3xl font-bold">Tinker Thinking Machine</h1>
        </div>
        <Alert>
          <AlertTriangle className="h-4 w-4" />
          <AlertTitle>Service Disabled</AlertTitle>
          <AlertDescription>
            Tinker integration is currently disabled. Contact your administrator to enable 
            TINKER_ENABLED and provide a valid TINKER_API_KEY to activate AI analysis features.
          </AlertDescription>
        </Alert>
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Shield className="h-5 w-5 text-primary" />
              Privacy Protection Active
            </CardTitle>
            <CardDescription>
              Even when disabled, the privacy firewall remains active to protect PHI
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid gap-4 md:grid-cols-3">
              <div className="flex items-center gap-3 p-4 rounded-lg bg-muted/50">
                <Lock className="h-5 w-5 text-green-600" />
                <div>
                  <p className="font-medium">PHI Protection</p>
                  <p className="text-sm text-muted-foreground">Always enforced</p>
                </div>
              </div>
              <div className="flex items-center gap-3 p-4 rounded-lg bg-muted/50">
                <Users className="h-5 w-5 text-blue-600" />
                <div>
                  <p className="font-medium">K-Anonymity Threshold</p>
                  <p className="text-sm text-muted-foreground">k ≥ 25 required</p>
                </div>
              </div>
              <div className="flex items-center gap-3 p-4 rounded-lg bg-muted/50">
                <Hash className="h-5 w-5 text-purple-600" />
                <div>
                  <p className="font-medium">SHA256 Hashing</p>
                  <p className="text-sm text-muted-foreground">Salted identifiers</p>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Brain className="h-8 w-8 text-primary" />
          <div>
            <h1 className="text-3xl font-bold">Tinker Thinking Machine</h1>
            <p className="text-muted-foreground">HIPAA-Compliant AI Analysis Platform</p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <Badge variant={health.api_connected ? "default" : "destructive"} data-testid="status-api">
            {health.api_connected ? "Connected" : "Disconnected"}
          </Badge>
          <Badge variant="outline" className="bg-green-50 text-green-700 dark:bg-green-950 dark:text-green-300" data-testid="status-privacy">
            <Lock className="h-3 w-3 mr-1" />
            Privacy Firewall Active
          </Badge>
          <Button variant="ghost" size="icon" onClick={() => refetchHealth()} data-testid="button-refresh-health">
            <RefreshCw className="h-4 w-4" />
          </Button>
        </div>
      </div>

      <div className="grid gap-4 md:grid-cols-4">
        <Card data-testid="card-circuit-breaker">
          <CardHeader className="flex flex-row items-center justify-between gap-2 space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Circuit Breaker</CardTitle>
            <Activity className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold capitalize">{health.circuit_breaker_state ?? 'Unknown'}</div>
            <p className="text-xs text-muted-foreground">System protection status</p>
          </CardContent>
        </Card>

        <Card data-testid="card-k-anonymity">
          <CardHeader className="flex flex-row items-center justify-between gap-2 space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">K-Anonymity</CardTitle>
            <Users className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">k ≥ {health.k_anonymity_threshold ?? 25}</div>
            <p className="text-xs text-muted-foreground">Minimum cohort size</p>
          </CardContent>
        </Card>

        <Card data-testid="card-uptime">
          <CardHeader className="flex flex-row items-center justify-between gap-2 space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Uptime</CardTitle>
            <Clock className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{(health.uptime_hours ?? 0).toFixed(1)}h</div>
            <p className="text-xs text-muted-foreground">Current session</p>
          </CardContent>
        </Card>

        <Card data-testid="card-version">
          <CardHeader className="flex flex-row items-center justify-between gap-2 space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Version</CardTitle>
            <BarChart3 className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{health.version ?? '1.0.0'}</div>
            <p className="text-xs text-muted-foreground">NON-BAA Mode</p>
          </CardContent>
        </Card>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-4">
        <TabsList data-testid="tabs-tinker">
          <TabsTrigger value="overview" data-testid="tab-overview">Overview</TabsTrigger>
          <TabsTrigger value="cohort" data-testid="tab-cohort">Cohort Analysis</TabsTrigger>
          <TabsTrigger value="drift" data-testid="tab-drift">Drift Monitoring</TabsTrigger>
          {isAdminOrDoctor && (
            <>
              <TabsTrigger value="privacy" data-testid="tab-privacy">Privacy Stats</TabsTrigger>
              <TabsTrigger value="audits" data-testid="tab-audits">Audit Logs</TabsTrigger>
            </>
          )}
        </TabsList>

        <TabsContent value="overview" className="space-y-4">
          <div className="grid gap-4 md:grid-cols-2">
            <Card data-testid="card-privacy-architecture">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Shield className="h-5 w-5 text-primary" />
                  Privacy Architecture
                </CardTitle>
                <CardDescription>
                  Multi-layer protection for HIPAA compliance
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-3">
                  <div className="flex items-center justify-between p-3 rounded-lg bg-muted/50">
                    <div className="flex items-center gap-2">
                      <CheckCircle2 className="h-4 w-4 text-green-600" />
                      <span className="text-sm">PHI Detection (18+ patterns)</span>
                    </div>
                    <Badge variant="outline">Active</Badge>
                  </div>
                  <div className="flex items-center justify-between p-3 rounded-lg bg-muted/50">
                    <div className="flex items-center gap-2">
                      <CheckCircle2 className="h-4 w-4 text-green-600" />
                      <span className="text-sm">SHA256 Salted Hashing</span>
                    </div>
                    <Badge variant="outline">Active</Badge>
                  </div>
                  <div className="flex items-center justify-between p-3 rounded-lg bg-muted/50">
                    <div className="flex items-center gap-2">
                      <CheckCircle2 className="h-4 w-4 text-green-600" />
                      <span className="text-sm">Data Bucketing (age, vitals)</span>
                    </div>
                    <Badge variant="outline">Active</Badge>
                  </div>
                  <div className="flex items-center justify-between p-3 rounded-lg bg-muted/50">
                    <div className="flex items-center gap-2">
                      <CheckCircle2 className="h-4 w-4 text-green-600" />
                      <span className="text-sm">K-Anonymity Enforcement</span>
                    </div>
                    <Badge variant="outline">k ≥ 25</Badge>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card data-testid="card-data-flow">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Brain className="h-5 w-5 text-primary" />
                  Data Flow
                </CardTitle>
                <CardDescription>
                  How patient data is processed
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  <div className="flex items-center gap-2 text-sm">
                    <div className="w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center text-primary font-medium">1</div>
                    <span>Raw data enters privacy firewall</span>
                  </div>
                  <div className="h-4 w-px bg-border ml-4" />
                  <div className="flex items-center gap-2 text-sm">
                    <div className="w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center text-primary font-medium">2</div>
                    <span>PHI fields stripped and hashed</span>
                  </div>
                  <div className="h-4 w-px bg-border ml-4" />
                  <div className="flex items-center gap-2 text-sm">
                    <div className="w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center text-primary font-medium">3</div>
                    <span>Values bucketed for anonymity</span>
                  </div>
                  <div className="h-4 w-px bg-border ml-4" />
                  <div className="flex items-center gap-2 text-sm">
                    <div className="w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center text-primary font-medium">4</div>
                    <span>K-anonymity verified (k ≥ 25)</span>
                  </div>
                  <div className="h-4 w-px bg-border ml-4" />
                  <div className="flex items-center gap-2 text-sm">
                    <div className="w-8 h-8 rounded-full bg-green-600/10 flex items-center justify-center text-green-600 font-medium">5</div>
                    <span>Safe payload sent to Tinker API</span>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          <Alert className="bg-blue-50 dark:bg-blue-950 border-blue-200 dark:border-blue-800">
            <Eye className="h-4 w-4 text-blue-600" />
            <AlertTitle className="text-blue-800 dark:text-blue-200">NON-BAA Mode Active</AlertTitle>
            <AlertDescription className="text-blue-700 dark:text-blue-300">
              Tinker integration operates without Business Associate Agreement. Enhanced privacy 
              protections are enforced: no PHI leaves this system, only hashed identifiers, 
              bucketed values, and k-anonymized aggregates are transmitted.
            </AlertDescription>
          </Alert>
        </TabsContent>

        <TabsContent value="cohort" className="space-y-4">
          <Card data-testid="card-cohort-analysis">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Users className="h-5 w-5 text-primary" />
                Cohort Analysis
              </CardTitle>
              <CardDescription>
                Analyze patient cohorts with k-anonymity protection (minimum 25 patients required)
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="conditions">Conditions (comma-separated)</Label>
                <div className="flex gap-2">
                  <Input
                    id="conditions"
                    placeholder="e.g., hypertension, diabetes, asthma"
                    value={cohortConditions}
                    onChange={(e) => setCohortConditions(e.target.value)}
                    data-testid="input-conditions"
                  />
                  <Button 
                    onClick={handleCohortAnalysis}
                    disabled={cohortMutation.isPending}
                    data-testid="button-analyze"
                  >
                    {cohortMutation.isPending ? (
                      <Loader2 className="h-4 w-4 animate-spin mr-2" />
                    ) : (
                      <Search className="h-4 w-4 mr-2" />
                    )}
                    Analyze
                  </Button>
                </div>
                <p className="text-xs text-muted-foreground">
                  Analysis will only proceed if cohort meets k-anonymity threshold
                </p>
              </div>

              {analysisResult && (
                <div className="mt-6 space-y-4" data-testid="result-cohort">
                  <div className="flex items-center justify-between">
                    <h3 className="font-semibold">Analysis Results</h3>
                    <Badge variant={analysisResult.k_anonymity_verified ? "default" : "destructive"}>
                      {analysisResult.k_anonymity_verified ? "K-Anonymity Verified" : "Privacy Check Failed"}
                    </Badge>
                  </div>

                  <div className="grid gap-4 md:grid-cols-3">
                    <div className="p-4 rounded-lg bg-muted/50">
                      <p className="text-sm text-muted-foreground">Sample Size</p>
                      <p className="text-2xl font-bold">{analysisResult.sample_size}</p>
                    </div>
                    <div className="p-4 rounded-lg bg-muted/50">
                      <p className="text-sm text-muted-foreground">PHI Fields Stripped</p>
                      <p className="text-2xl font-bold">{analysisResult.privacy_metadata.phi_fields_stripped}</p>
                    </div>
                    <div className="p-4 rounded-lg bg-muted/50">
                      <p className="text-sm text-muted-foreground">Cohort Hash</p>
                      <p className="text-sm font-mono truncate">{analysisResult.cohort_hash}</p>
                    </div>
                  </div>

                  {analysisResult.insights.risk_distribution && (
                    <div className="h-64">
                      <ResponsiveContainer width="100%" height="100%">
                        <PieChart>
                          <Pie
                            data={Object.entries(analysisResult.insights.risk_distribution).map(([name, value]) => ({ name, value }))}
                            cx="50%"
                            cy="50%"
                            outerRadius={80}
                            dataKey="value"
                            label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                          >
                            {Object.keys(analysisResult.insights.risk_distribution).map((_, index) => (
                              <Cell key={`cell-${index}`} fill={PIE_COLORS[index % PIE_COLORS.length]} />
                            ))}
                          </Pie>
                          <Tooltip />
                          <Legend />
                        </PieChart>
                      </ResponsiveContainer>
                    </div>
                  )}

                  {analysisResult.insights.top_conditions && (
                    <div className="flex flex-wrap gap-2">
                      <span className="text-sm text-muted-foreground">Top Conditions:</span>
                      {analysisResult.insights.top_conditions.map((condition) => (
                        <Badge key={condition} variant="outline">{condition}</Badge>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="drift" className="space-y-4">
          <Card data-testid="card-drift-monitoring">
            <CardHeader className="flex flex-row items-center justify-between">
              <div>
                <CardTitle className="flex items-center gap-2">
                  <TrendingUp className="h-5 w-5 text-primary" />
                  Drift Monitoring
                </CardTitle>
                <CardDescription>
                  Monitor data distribution changes over time
                </CardDescription>
              </div>
              <Button variant="outline" size="sm" onClick={() => refetchDrift()} data-testid="button-refresh-drift">
                <RefreshCw className="h-4 w-4 mr-2" />
                Refresh
              </Button>
            </CardHeader>
            <CardContent>
              {driftLoading ? (
                <div className="space-y-4">
                  <Skeleton className="h-8 w-full" />
                  <Skeleton className="h-64 w-full" />
                </div>
              ) : driftError ? (
                <Alert variant="destructive">
                  <AlertTriangle className="h-4 w-4" />
                  <AlertTitle>Error Loading Drift Data</AlertTitle>
                  <AlertDescription>
                    Failed to check for data drift. Ensure Tinker is enabled and try again.
                  </AlertDescription>
                </Alert>
              ) : driftStatus ? (
                <div className="space-y-4">
                  <div className="flex items-center justify-between p-4 rounded-lg" style={{
                    backgroundColor: driftStatus.drift_detected 
                      ? "rgba(239, 68, 68, 0.1)" 
                      : "rgba(34, 197, 94, 0.1)"
                  }}>
                    <div className="flex items-center gap-2">
                      {driftStatus.drift_detected ? (
                        <AlertTriangle className="h-5 w-5 text-red-600" />
                      ) : (
                        <CheckCircle2 className="h-5 w-5 text-green-600" />
                      )}
                      <span className="font-medium">
                        {driftStatus.drift_detected ? "Drift Detected" : "No Drift Detected"}
                      </span>
                    </div>
                    {driftStatus.active_alerts !== undefined && driftStatus.active_alerts > 0 && (
                      <Badge variant="destructive">
                        {driftStatus.active_alerts} Active Alert{driftStatus.active_alerts > 1 ? 's' : ''}
                      </Badge>
                    )}
                  </div>

                  {driftStatus.last_check && (
                    <div className="text-sm text-muted-foreground">
                      Last checked: {new Date(driftStatus.last_check).toLocaleString()}
                    </div>
                  )}

                  {driftStatus.recent_runs.length > 0 && (
                    <div className="h-64">
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={driftStatus.recent_runs.map((run) => ({
                          model: run.model_id || 'Unknown',
                          psi: run.psi_score ?? 0,
                          threshold: 0.1
                        }))}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey="model" />
                          <YAxis />
                          <Tooltip />
                          <Legend />
                          <Bar dataKey="psi" fill="#0d9488" name="PSI Score" />
                          <Bar dataKey="threshold" fill="#f97316" name="Threshold" />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  )}

                  {driftStatus.recent_runs.length > 0 && (
                    <div className="space-y-2">
                      <h4 className="font-medium text-sm">Recent Drift Runs</h4>
                      <div className="space-y-2">
                        {driftStatus.recent_runs.map((run) => (
                          <div key={run.id} className="flex items-center justify-between p-2 rounded border">
                            <div className="flex items-center gap-2">
                              {run.drift_detected ? (
                                <AlertTriangle className="h-4 w-4 text-red-500" />
                              ) : (
                                <CheckCircle2 className="h-4 w-4 text-green-500" />
                              )}
                              <span className="text-sm">{run.model_id}</span>
                            </div>
                            <div className="flex items-center gap-2">
                              {run.psi_score !== null && (
                                <Badge variant="outline">PSI: {run.psi_score.toFixed(3)}</Badge>
                              )}
                              {run.created_at && (
                                <span className="text-xs text-muted-foreground">
                                  {new Date(run.created_at).toLocaleDateString()}
                                </span>
                              )}
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              ) : (
                <div className="text-center py-8 text-muted-foreground">
                  <TrendingDown className="h-12 w-12 mx-auto mb-4 opacity-50" />
                  <p>No drift data available</p>
                  <p className="text-sm">Enable Tinker and configure drift monitoring</p>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {isAdminOrDoctor && (
          <TabsContent value="privacy" className="space-y-4">
            <Card data-testid="card-privacy-stats">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Lock className="h-5 w-5 text-primary" />
                  Privacy Statistics
                </CardTitle>
                <CardDescription>
                  HIPAA compliance and privacy enforcement metrics
                </CardDescription>
              </CardHeader>
              <CardContent>
                {privacyLoading ? (
                  <div className="grid gap-4 md:grid-cols-4">
                    {[1, 2, 3, 4].map((i) => (
                      <Skeleton key={i} className="h-24" />
                    ))}
                  </div>
                ) : privacyError ? (
                  <Alert variant="destructive">
                    <AlertTriangle className="h-4 w-4" />
                    <AlertTitle>Failed to Load</AlertTitle>
                    <AlertDescription>
                      Unable to retrieve privacy statistics. Please try again later.
                    </AlertDescription>
                  </Alert>
                ) : privacyStats ? (
                  <div className="space-y-6">
                    <div className="grid gap-4 md:grid-cols-4">
                      <div className="p-4 rounded-lg bg-muted/50 text-center">
                        <p className="text-3xl font-bold text-primary">{privacyStats.total_requests}</p>
                        <p className="text-sm text-muted-foreground">Total Requests</p>
                      </div>
                      <div className="p-4 rounded-lg bg-green-50 dark:bg-green-950 text-center">
                        <p className="text-3xl font-bold text-green-600">{privacyStats.successful_analyses}</p>
                        <p className="text-sm text-muted-foreground">Successful Analyses</p>
                      </div>
                      <div className="p-4 rounded-lg bg-red-50 dark:bg-red-950 text-center">
                        <p className="text-3xl font-bold text-red-600">{privacyStats.phi_detections_blocked}</p>
                        <p className="text-sm text-muted-foreground">PHI Blocked</p>
                      </div>
                      <div className="p-4 rounded-lg bg-orange-50 dark:bg-orange-950 text-center">
                        <p className="text-3xl font-bold text-orange-600">{privacyStats.k_anonymity_rejections}</p>
                        <p className="text-sm text-muted-foreground">K-Anon Rejections</p>
                      </div>
                    </div>

                    <div className="p-4 rounded-lg border">
                      <div className="flex items-center justify-between mb-2">
                        <span className="font-medium">Privacy Score</span>
                        <span className="text-2xl font-bold">{privacyStats.privacy_score}%</span>
                      </div>
                      <div className="w-full h-3 bg-muted rounded-full overflow-hidden">
                        <div 
                          className="h-full bg-primary transition-all"
                          style={{ width: `${privacyStats.privacy_score}%` }}
                        />
                      </div>
                      <p className="text-xs text-muted-foreground mt-2">
                        Based on PHI protection rate and k-anonymity compliance
                      </p>
                    </div>
                  </div>
                ) : (
                  <p className="text-muted-foreground">No privacy statistics available</p>
                )}
              </CardContent>
            </Card>
          </TabsContent>
        )}

        {isAdminOrDoctor && (
          <TabsContent value="audits" className="space-y-4">
            <Card data-testid="card-audit-logs">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <FileText className="h-5 w-5 text-primary" />
                  Audit Logs
                </CardTitle>
                <CardDescription>
                  HIPAA-compliant audit trail with SHA256 hashed identifiers
                </CardDescription>
              </CardHeader>
              <CardContent>
                {auditLoading ? (
                  <div className="space-y-2">
                    {[1, 2, 3, 4, 5].map((i) => (
                      <Skeleton key={i} className="h-16" />
                    ))}
                  </div>
                ) : auditError ? (
                  <Alert variant="destructive">
                    <AlertTriangle className="h-4 w-4" />
                    <AlertTitle>Error Loading Audit Logs</AlertTitle>
                    <AlertDescription>
                      Failed to fetch audit logs. Please check your connection and try again.
                    </AlertDescription>
                  </Alert>
                ) : auditLogs?.entries && auditLogs.entries.length > 0 ? (
                  <ScrollArea className="h-96">
                    <div className="space-y-2">
                      {auditLogs.entries.map((entry) => (
                        <div 
                          key={entry.id}
                          className="p-3 rounded-lg border flex items-center justify-between"
                          data-testid={`audit-entry-${entry.id}`}
                        >
                          <div className="flex items-center gap-3">
                            {entry.success ? (
                              <CheckCircle2 className="h-4 w-4 text-green-600" />
                            ) : (
                              <AlertTriangle className="h-4 w-4 text-red-600" />
                            )}
                            <div>
                              <p className="font-medium text-sm">{entry.action}</p>
                              <p className="text-xs text-muted-foreground font-mono">
                                User: {entry.user_hash.slice(0, 16)}...
                              </p>
                            </div>
                          </div>
                          <div className="text-right">
                            <Badge variant={entry.k_anon_verified ? "outline" : "secondary"} className="text-xs">
                              {entry.k_anon_verified ? "K-Anon Verified" : "Not Verified"}
                            </Badge>
                            <p className="text-xs text-muted-foreground mt-1">
                              {new Date(entry.timestamp).toLocaleString()}
                            </p>
                          </div>
                        </div>
                      ))}
                    </div>
                  </ScrollArea>
                ) : (
                  <div className="text-center py-8 text-muted-foreground">
                    <FileText className="h-12 w-12 mx-auto mb-4 opacity-50" />
                    <p>No audit logs available</p>
                  </div>
                )}
              </CardContent>
            </Card>
          </TabsContent>
        )}
      </Tabs>
    </div>
  );
}
