import { useState } from 'react';
import { useQuery, useMutation } from '@tanstack/react-query';
import { queryClient, apiRequest } from '@/lib/queryClient';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Skeleton } from '@/components/ui/skeleton';
import { Progress } from '@/components/ui/progress';
import { useToast } from '@/hooks/use-toast';
import {
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
} from '@/components/ui/tabs';
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
} from 'recharts';
import {
  Activity,
  AlertTriangle,
  CheckCircle2,
  Clock,
  RefreshCw,
  Settings,
  Users,
  TrendingUp,
  BarChart3,
  Bell,
  Shield,
  Brain,
  Loader2,
  Save,
  PieChart as PieChartIcon,
} from 'lucide-react';

interface SystemHealth {
  status: string;
  uptime_hours: number;
  active_patients_24h: number;
  total_patients: number;
  signals_processed_1h: number;
  tasks_generated_24h: number;
  alerts_triggered_24h: number;
  notifications_sent_24h: number;
  last_inference_run: string | null;
  wellness_disclaimer: string;
}

interface EngagementMetrics {
  tasks: {
    total: number;
    completed: number;
    completion_rate: number;
    avg_completion_time_hours: number;
    by_type: Record<string, number>;
    by_priority: Record<string, number>;
  };
  notifications: {
    total_sent: number;
    total_read: number;
    read_rate: number;
    by_channel: Record<string, number>;
  };
  privacy_note: string;
  wellness_disclaimer: string;
}

interface ModelPerformance {
  models: Array<{
    name: string;
    version: string;
    accuracy: number;
    precision: number;
    recall: number;
    f1_score: number;
    auc_roc: number;
    drift_detected: boolean;
    last_trained: string;
    predictions_count: number;
  }>;
  drift_alerts_active: number;
  wellness_disclaimer: string;
}

interface CohortData {
  cohorts: Array<{
    name: string;
    patient_count: number;
    description: string;
    avg_risk_score: number;
    avg_engagement: number;
  }>;
  risk_distribution: Record<string, number>;
  total_cohorts: number;
  privacy_note: string;
  wellness_disclaimer: string;
}

interface TriggerAnalytics {
  by_type: Record<string, number>;
  by_severity: Record<string, number>;
  total_triggers: number;
  privacy_note: string;
  wellness_disclaimer: string;
}

interface Configuration {
  key: string;
  value: any;
  category: string;
  description: string;
  min_value?: number;
  max_value?: number;
}

interface ConfigurationsData {
  configurations: Configuration[];
  total: number;
  wellness_disclaimer: string;
}

const CHART_COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899'];

export function AutopilotAdminDashboard() {
  const { toast } = useToast();
  const [activeTab, setActiveTab] = useState('overview');
  const [analysisDays, setAnalysisDays] = useState('30');
  const [configEdits, setConfigEdits] = useState<Record<string, any>>({});

  const { data: healthData, isLoading: healthLoading, refetch: refetchHealth } = useQuery<SystemHealth>({
    queryKey: ['/api/autopilot/admin/health'],
    refetchInterval: 30000,
  });

  const { data: engagementData, isLoading: engagementLoading, refetch: refetchEngagement } = useQuery<EngagementMetrics>({
    queryKey: ['/api/autopilot/admin/engagement', { days: analysisDays }],
  });

  const { data: modelsData, isLoading: modelsLoading, refetch: refetchModels } = useQuery<ModelPerformance>({
    queryKey: ['/api/autopilot/admin/models/performance', { days: analysisDays }],
  });

  const { data: cohortsData, isLoading: cohortsLoading, refetch: refetchCohorts } = useQuery<CohortData>({
    queryKey: ['/api/autopilot/admin/cohorts'],
  });

  const { data: triggersData, isLoading: triggersLoading, refetch: refetchTriggers } = useQuery<TriggerAnalytics>({
    queryKey: ['/api/autopilot/admin/triggers', { days: analysisDays }],
  });

  const { data: configsData, isLoading: configsLoading, refetch: refetchConfigs } = useQuery<ConfigurationsData>({
    queryKey: ['/api/autopilot/admin/configurations'],
  });

  const updateConfigMutation = useMutation({
    mutationFn: async (payload: { config_key: string; config_value: any }) => {
      return await apiRequest('/api/autopilot/admin/configurations', {
        method: 'PUT',
        body: JSON.stringify(payload),
      });
    },
    onSuccess: () => {
      toast({ title: 'Configuration Updated', description: 'The setting has been saved successfully.' });
      queryClient.invalidateQueries({ queryKey: ['/api/autopilot/admin/configurations'] });
      setConfigEdits({});
    },
    onError: (error: Error) => {
      toast({ title: 'Update Failed', description: error.message, variant: 'destructive' });
    },
  });

  const handleRefreshAll = () => {
    refetchHealth();
    refetchEngagement();
    refetchModels();
    refetchCohorts();
    refetchTriggers();
    refetchConfigs();
    toast({ title: 'Refreshing', description: 'Dashboard data is being refreshed.' });
  };

  const handleConfigSave = (key: string, value: any) => {
    updateConfigMutation.mutate({ config_key: key, config_value: value });
  };

  const riskPieData = cohortsData?.risk_distribution
    ? Object.entries(cohortsData.risk_distribution).map(([name, value]) => ({
        name,
        value,
      }))
    : [];

  const triggerTypeData = triggersData?.by_type
    ? Object.entries(triggersData.by_type).map(([name, value]) => ({
        name: name.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase()),
        value,
      }))
    : [];

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between gap-4 flex-wrap">
        <div>
          <h2 className="text-2xl font-bold" data-testid="text-admin-title">Autopilot Admin Dashboard</h2>
          <p className="text-muted-foreground">System monitoring, analytics, and configuration management</p>
        </div>
        <div className="flex items-center gap-2">
          <Select value={analysisDays} onValueChange={setAnalysisDays}>
            <SelectTrigger className="w-32" data-testid="select-analysis-days">
              <SelectValue placeholder="Period" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="7">7 Days</SelectItem>
              <SelectItem value="30">30 Days</SelectItem>
              <SelectItem value="90">90 Days</SelectItem>
            </SelectContent>
          </Select>
          <Button variant="outline" onClick={handleRefreshAll} data-testid="button-refresh-all">
            <RefreshCw className="h-4 w-4 mr-2" />
            Refresh
          </Button>
        </div>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-4">
        <TabsList className="grid w-full grid-cols-5">
          <TabsTrigger value="overview" data-testid="tab-overview">
            <Activity className="h-4 w-4 mr-2" />
            Overview
          </TabsTrigger>
          <TabsTrigger value="engagement" data-testid="tab-engagement">
            <Users className="h-4 w-4 mr-2" />
            Engagement
          </TabsTrigger>
          <TabsTrigger value="models" data-testid="tab-models">
            <Brain className="h-4 w-4 mr-2" />
            ML Models
          </TabsTrigger>
          <TabsTrigger value="cohorts" data-testid="tab-cohorts">
            <PieChartIcon className="h-4 w-4 mr-2" />
            Cohorts
          </TabsTrigger>
          <TabsTrigger value="config" data-testid="tab-config">
            <Settings className="h-4 w-4 mr-2" />
            Config
          </TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="space-y-6">
          {healthLoading ? (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              {[1, 2, 3, 4].map((i) => (
                <Skeleton key={i} className="h-32 w-full" />
              ))}
            </div>
          ) : healthData ? (
            <>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                <Card>
                  <CardHeader className="pb-2">
                    <CardTitle className="text-sm font-medium text-muted-foreground">System Status</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="flex items-center gap-2">
                      {healthData.status === 'healthy' ? (
                        <CheckCircle2 className="h-8 w-8 text-green-500" />
                      ) : (
                        <AlertTriangle className="h-8 w-8 text-yellow-500" />
                      )}
                      <div>
                        <p className="text-2xl font-bold capitalize" data-testid="text-system-status">
                          {healthData.status}
                        </p>
                        <p className="text-xs text-muted-foreground">
                          Uptime: {healthData.uptime_hours.toFixed(1)}h
                        </p>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader className="pb-2">
                    <CardTitle className="text-sm font-medium text-muted-foreground">Active Patients</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="flex items-center gap-2">
                      <Users className="h-8 w-8 text-blue-500" />
                      <div>
                        <p className="text-2xl font-bold" data-testid="text-active-patients">
                          {healthData.active_patients_24h}
                        </p>
                        <p className="text-xs text-muted-foreground">
                          of {healthData.total_patients} total
                        </p>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader className="pb-2">
                    <CardTitle className="text-sm font-medium text-muted-foreground">Signals Processed</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="flex items-center gap-2">
                      <Activity className="h-8 w-8 text-purple-500" />
                      <div>
                        <p className="text-2xl font-bold" data-testid="text-signals-processed">
                          {healthData.signals_processed_1h}
                        </p>
                        <p className="text-xs text-muted-foreground">last hour</p>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader className="pb-2">
                    <CardTitle className="text-sm font-medium text-muted-foreground">Alerts Today</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="flex items-center gap-2">
                      <Bell className="h-8 w-8 text-orange-500" />
                      <div>
                        <p className="text-2xl font-bold" data-testid="text-alerts-today">
                          {healthData.alerts_triggered_24h}
                        </p>
                        <p className="text-xs text-muted-foreground">
                          {healthData.notifications_sent_24h} notifications sent
                        </p>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                <Card>
                  <CardHeader>
                    <CardTitle className="text-lg flex items-center gap-2">
                      <BarChart3 className="h-5 w-5" />
                      Trigger Distribution
                    </CardTitle>
                    <CardDescription>Triggers by type over the selected period</CardDescription>
                  </CardHeader>
                  <CardContent>
                    {triggersLoading ? (
                      <Skeleton className="h-64 w-full" />
                    ) : triggerTypeData.length > 0 ? (
                      <ResponsiveContainer width="100%" height={250}>
                        <BarChart data={triggerTypeData}>
                          <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                          <XAxis dataKey="name" tick={{ fontSize: 10 }} angle={-45} textAnchor="end" height={80} />
                          <YAxis />
                          <Tooltip
                            contentStyle={{
                              backgroundColor: 'hsl(var(--popover))',
                              border: '1px solid hsl(var(--border))',
                              borderRadius: '8px',
                            }}
                          />
                          <Bar dataKey="value" fill="#3b82f6" radius={[4, 4, 0, 0]} />
                        </BarChart>
                      </ResponsiveContainer>
                    ) : (
                      <div className="h-64 flex items-center justify-center text-muted-foreground">
                        No trigger data available
                      </div>
                    )}
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <CardTitle className="text-lg flex items-center gap-2">
                      <Shield className="h-5 w-5" />
                      Risk Distribution
                    </CardTitle>
                    <CardDescription>Patient distribution by risk level</CardDescription>
                  </CardHeader>
                  <CardContent>
                    {cohortsLoading ? (
                      <Skeleton className="h-64 w-full" />
                    ) : riskPieData.length > 0 ? (
                      <ResponsiveContainer width="100%" height={250}>
                        <PieChart>
                          <Pie
                            data={riskPieData}
                            cx="50%"
                            cy="50%"
                            labelLine={false}
                            label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                            outerRadius={80}
                            dataKey="value"
                          >
                            {riskPieData.map((_, index) => (
                              <Cell key={`cell-${index}`} fill={CHART_COLORS[index % CHART_COLORS.length]} />
                            ))}
                          </Pie>
                          <Tooltip
                            contentStyle={{
                              backgroundColor: 'hsl(var(--popover))',
                              border: '1px solid hsl(var(--border))',
                              borderRadius: '8px',
                            }}
                          />
                        </PieChart>
                      </ResponsiveContainer>
                    ) : (
                      <div className="h-64 flex items-center justify-center text-muted-foreground">
                        No cohort data available
                      </div>
                    )}
                  </CardContent>
                </Card>
              </div>
            </>
          ) : (
            <Card>
              <CardContent className="p-8 text-center">
                <AlertTriangle className="h-12 w-12 mx-auto mb-4 text-yellow-500" />
                <p className="text-muted-foreground">Unable to load system health data</p>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        <TabsContent value="engagement" className="space-y-6">
          {engagementLoading ? (
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {[1, 2, 3].map((i) => (
                <Skeleton key={i} className="h-32 w-full" />
              ))}
            </div>
          ) : engagementData ? (
            <>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <Card>
                  <CardHeader className="pb-2">
                    <CardTitle className="text-sm font-medium text-muted-foreground">Task Completion Rate</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-2">
                      <p className="text-3xl font-bold" data-testid="text-completion-rate">
                        {(engagementData.tasks.completion_rate * 100).toFixed(1)}%
                      </p>
                      <Progress value={engagementData.tasks.completion_rate * 100} />
                      <p className="text-xs text-muted-foreground">
                        {engagementData.tasks.completed} of {engagementData.tasks.total} tasks completed
                      </p>
                    </div>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader className="pb-2">
                    <CardTitle className="text-sm font-medium text-muted-foreground">Notification Read Rate</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-2">
                      <p className="text-3xl font-bold" data-testid="text-read-rate">
                        {(engagementData.notifications.read_rate * 100).toFixed(1)}%
                      </p>
                      <Progress value={engagementData.notifications.read_rate * 100} />
                      <p className="text-xs text-muted-foreground">
                        {engagementData.notifications.total_read} of {engagementData.notifications.total_sent} read
                      </p>
                    </div>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader className="pb-2">
                    <CardTitle className="text-sm font-medium text-muted-foreground">Avg Completion Time</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="flex items-center gap-2">
                      <Clock className="h-8 w-8 text-blue-500" />
                      <div>
                        <p className="text-3xl font-bold" data-testid="text-avg-time">
                          {engagementData.tasks.avg_completion_time_hours.toFixed(1)}h
                        </p>
                        <p className="text-xs text-muted-foreground">average response time</p>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                <Card>
                  <CardHeader>
                    <CardTitle className="text-lg">Tasks by Type</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-3">
                      {Object.entries(engagementData.tasks.by_type).map(([type, count]) => (
                        <div key={type} className="flex items-center justify-between">
                          <span className="text-sm capitalize">{type.replace(/_/g, ' ')}</span>
                          <Badge variant="secondary">{count}</Badge>
                        </div>
                      ))}
                      {Object.keys(engagementData.tasks.by_type).length === 0 && (
                        <p className="text-muted-foreground text-sm">No task type data available</p>
                      )}
                    </div>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <CardTitle className="text-lg">Notifications by Channel</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-3">
                      {Object.entries(engagementData.notifications.by_channel).map(([channel, count]) => (
                        <div key={channel} className="flex items-center justify-between">
                          <span className="text-sm capitalize">{channel}</span>
                          <Badge variant="secondary">{count}</Badge>
                        </div>
                      ))}
                      {Object.keys(engagementData.notifications.by_channel).length === 0 && (
                        <p className="text-muted-foreground text-sm">No channel data available</p>
                      )}
                    </div>
                  </CardContent>
                </Card>
              </div>
            </>
          ) : (
            <Card>
              <CardContent className="p-8 text-center">
                <AlertTriangle className="h-12 w-12 mx-auto mb-4 text-yellow-500" />
                <p className="text-muted-foreground">Unable to load engagement data</p>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        <TabsContent value="models" className="space-y-6">
          {modelsLoading ? (
            <div className="space-y-4">
              {[1, 2, 3].map((i) => (
                <Skeleton key={i} className="h-24 w-full" />
              ))}
            </div>
          ) : modelsData ? (
            <>
              {modelsData.drift_alerts_active > 0 && (
                <Card className="border-yellow-500/50 bg-yellow-500/10">
                  <CardContent className="p-4 flex items-center gap-3">
                    <AlertTriangle className="h-6 w-6 text-yellow-500" />
                    <div>
                      <p className="font-medium">Model Drift Detected</p>
                      <p className="text-sm text-muted-foreground">
                        {modelsData.drift_alerts_active} model(s) showing performance degradation
                      </p>
                    </div>
                  </CardContent>
                </Card>
              )}

              <div className="space-y-4">
                {modelsData.models.map((model) => (
                  <Card key={model.name}>
                    <CardHeader className="pb-2">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <Brain className="h-5 w-5" />
                          <CardTitle className="text-lg">{model.name}</CardTitle>
                          <Badge variant="secondary" className="text-xs">v{model.version}</Badge>
                        </div>
                        {model.drift_detected && (
                          <Badge variant="destructive" className="text-xs">
                            <AlertTriangle className="h-3 w-3 mr-1" />
                            Drift Detected
                          </Badge>
                        )}
                      </div>
                    </CardHeader>
                    <CardContent>
                      <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
                        <div>
                          <p className="text-xs text-muted-foreground">Accuracy</p>
                          <p className="text-xl font-semibold">{(model.accuracy * 100).toFixed(1)}%</p>
                        </div>
                        <div>
                          <p className="text-xs text-muted-foreground">Precision</p>
                          <p className="text-xl font-semibold">{(model.precision * 100).toFixed(1)}%</p>
                        </div>
                        <div>
                          <p className="text-xs text-muted-foreground">Recall</p>
                          <p className="text-xl font-semibold">{(model.recall * 100).toFixed(1)}%</p>
                        </div>
                        <div>
                          <p className="text-xs text-muted-foreground">F1 Score</p>
                          <p className="text-xl font-semibold">{(model.f1_score * 100).toFixed(1)}%</p>
                        </div>
                        <div>
                          <p className="text-xs text-muted-foreground">AUC-ROC</p>
                          <p className="text-xl font-semibold">{(model.auc_roc * 100).toFixed(1)}%</p>
                        </div>
                      </div>
                      <div className="mt-4 pt-4 border-t flex items-center justify-between text-sm text-muted-foreground">
                        <span>Last trained: {new Date(model.last_trained).toLocaleDateString()}</span>
                        <span>{model.predictions_count.toLocaleString()} predictions</span>
                      </div>
                    </CardContent>
                  </Card>
                ))}

                {modelsData.models.length === 0 && (
                  <Card>
                    <CardContent className="p-8 text-center">
                      <Brain className="h-12 w-12 mx-auto mb-4 text-muted-foreground" />
                      <p className="text-muted-foreground">No ML models registered yet</p>
                    </CardContent>
                  </Card>
                )}
              </div>
            </>
          ) : (
            <Card>
              <CardContent className="p-8 text-center">
                <AlertTriangle className="h-12 w-12 mx-auto mb-4 text-yellow-500" />
                <p className="text-muted-foreground">Unable to load model performance data</p>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        <TabsContent value="cohorts" className="space-y-6">
          {cohortsLoading ? (
            <div className="space-y-4">
              {[1, 2, 3].map((i) => (
                <Skeleton key={i} className="h-24 w-full" />
              ))}
            </div>
          ) : cohortsData ? (
            <>
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                <Card>
                  <CardHeader>
                    <CardTitle className="text-lg flex items-center gap-2">
                      <Shield className="h-5 w-5" />
                      Risk Distribution
                    </CardTitle>
                    <CardDescription>Patients grouped by risk level (privacy protected)</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <ResponsiveContainer width="100%" height={250}>
                      <PieChart>
                        <Pie
                          data={riskPieData}
                          cx="50%"
                          cy="50%"
                          labelLine={false}
                          label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                          outerRadius={80}
                          dataKey="value"
                        >
                          {riskPieData.map((_, index) => (
                            <Cell key={`cell-${index}`} fill={CHART_COLORS[index % CHART_COLORS.length]} />
                          ))}
                        </Pie>
                        <Tooltip
                          contentStyle={{
                            backgroundColor: 'hsl(var(--popover))',
                            border: '1px solid hsl(var(--border))',
                            borderRadius: '8px',
                          }}
                        />
                      </PieChart>
                    </ResponsiveContainer>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <CardTitle className="text-lg">Cohort Summary</CardTitle>
                    <CardDescription>{cohortsData.total_cohorts} patient cohorts identified</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-4">
                      {cohortsData.cohorts.map((cohort) => (
                        <div key={cohort.name} className="p-3 rounded-lg bg-muted/50">
                          <div className="flex items-center justify-between mb-1">
                            <span className="font-medium">{cohort.name}</span>
                            <Badge variant="secondary">{cohort.patient_count} patients</Badge>
                          </div>
                          <p className="text-xs text-muted-foreground mb-2">{cohort.description}</p>
                          <div className="flex gap-4 text-xs">
                            <span>Avg Risk: {cohort.avg_risk_score.toFixed(1)}</span>
                            <span>Engagement: {(cohort.avg_engagement * 100).toFixed(0)}%</span>
                          </div>
                        </div>
                      ))}
                      {cohortsData.cohorts.length === 0 && (
                        <p className="text-muted-foreground text-sm text-center py-4">
                          No cohorts available (minimum 10 patients required per cohort)
                        </p>
                      )}
                    </div>
                  </CardContent>
                </Card>
              </div>

              {cohortsData.privacy_note && (
                <Card className="bg-muted/30">
                  <CardContent className="p-4 flex items-center gap-3">
                    <Shield className="h-5 w-5 text-blue-500" />
                    <p className="text-sm text-muted-foreground">{cohortsData.privacy_note}</p>
                  </CardContent>
                </Card>
              )}
            </>
          ) : (
            <Card>
              <CardContent className="p-8 text-center">
                <AlertTriangle className="h-12 w-12 mx-auto mb-4 text-yellow-500" />
                <p className="text-muted-foreground">Unable to load cohort data</p>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        <TabsContent value="config" className="space-y-6">
          {configsLoading ? (
            <div className="space-y-4">
              {[1, 2, 3].map((i) => (
                <Skeleton key={i} className="h-20 w-full" />
              ))}
            </div>
          ) : configsData ? (
            <div className="space-y-4">
              {configsData.configurations.map((config) => (
                <Card key={config.key}>
                  <CardContent className="p-4">
                    <div className="flex items-start justify-between gap-4">
                      <div className="flex-1">
                        <div className="flex items-center gap-2 mb-1">
                          <span className="font-medium">{config.key}</span>
                          <Badge variant="outline" className="text-xs">{config.category}</Badge>
                        </div>
                        <p className="text-sm text-muted-foreground mb-2">{config.description}</p>
                        {(config.min_value !== undefined || config.max_value !== undefined) && (
                          <p className="text-xs text-muted-foreground">
                            Range: {config.min_value ?? '-∞'} to {config.max_value ?? '∞'}
                          </p>
                        )}
                      </div>
                      <div className="flex items-center gap-2">
                        <Input
                          type={typeof config.value === 'number' ? 'number' : 'text'}
                          value={configEdits[config.key] ?? config.value}
                          onChange={(e) => setConfigEdits({
                            ...configEdits,
                            [config.key]: typeof config.value === 'number'
                              ? parseFloat(e.target.value)
                              : e.target.value
                          })}
                          className="w-32"
                          data-testid={`input-config-${config.key}`}
                        />
                        <Button
                          size="icon"
                          variant="outline"
                          disabled={configEdits[config.key] === undefined || updateConfigMutation.isPending}
                          onClick={() => handleConfigSave(config.key, configEdits[config.key])}
                          data-testid={`button-save-${config.key}`}
                        >
                          {updateConfigMutation.isPending ? (
                            <Loader2 className="h-4 w-4 animate-spin" />
                          ) : (
                            <Save className="h-4 w-4" />
                          )}
                        </Button>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))}

              {configsData.configurations.length === 0 && (
                <Card>
                  <CardContent className="p-8 text-center">
                    <Settings className="h-12 w-12 mx-auto mb-4 text-muted-foreground" />
                    <p className="text-muted-foreground">No configurations available</p>
                  </CardContent>
                </Card>
              )}
            </div>
          ) : (
            <Card>
              <CardContent className="p-8 text-center">
                <AlertTriangle className="h-12 w-12 mx-auto mb-4 text-yellow-500" />
                <p className="text-muted-foreground">Unable to load configuration data</p>
              </CardContent>
            </Card>
          )}
        </TabsContent>
      </Tabs>

      <Card className="bg-muted/30">
        <CardContent className="p-4 flex items-center gap-3">
          <Shield className="h-5 w-5 text-blue-500 flex-shrink-0" />
          <p className="text-xs text-muted-foreground">
            {healthData?.wellness_disclaimer || 
             "This information is for wellness monitoring only and is not a substitute for professional medical advice, diagnosis, or treatment."}
          </p>
        </CardContent>
      </Card>
    </div>
  );
}

export default AutopilotAdminDashboard;
