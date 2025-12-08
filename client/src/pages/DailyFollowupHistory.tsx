import { useState, useMemo } from 'react';
import { useQuery, useMutation } from '@tanstack/react-query';
import { Link } from 'wouter';
import { queryClient, apiRequest } from '@/lib/queryClient';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Progress } from '@/components/ui/progress';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Checkbox } from '@/components/ui/checkbox';
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible';
import { useToast } from '@/hooks/use-toast';
import { DeviceDataManager } from '@/components/DeviceDataManager';
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
  Legend,
  PieChart,
  Pie,
  Cell,
} from 'recharts';
import {
  Activity,
  AlertCircle,
  BarChart3,
  Brain,
  Calendar,
  Camera,
  CheckCircle2,
  ChevronRight,
  Clock,
  Eye,
  FileText,
  Hand,
  Heart,
  Loader2,
  Mic,
  Palette,
  Play,
  Smile,
  Thermometer,
  TrendingDown,
  TrendingUp,
  User,
  Video,
  Wind,
  Zap,
  Footprints,
  MessageSquare,
  AlertTriangle,
  Target,
  ChevronDown,
  ChevronUp,
  Shield,
  Bell,
  ListChecks,
  Sparkles,
  RefreshCw,
} from 'lucide-react';
import { useAuth } from '@/contexts/AuthContext';
import { format, subDays, subMonths, subYears, isWithinInterval, startOfDay, endOfDay, parseISO } from 'date-fns';

const TIME_RANGES = [
  { value: '7', label: '7 Days', days: 7 },
  { value: '15', label: '15 Days', days: 15 },
  { value: '30', label: '30 Days', days: 30 },
  { value: '90', label: '3 Months', days: 90 },
  { value: '180', label: '6 Months', days: 180 },
  { value: '365', label: '1 Year', days: 365 },
  { value: 'all', label: 'All Time', days: Infinity },
];

const CHART_COLORS = {
  primary: 'hsl(var(--primary))',
  secondary: 'hsl(var(--chart-2))',
  tertiary: 'hsl(var(--chart-3))',
  quaternary: 'hsl(var(--chart-4))',
  quinary: 'hsl(var(--chart-5))',
};

function LegalDisclaimer() {
  return (
    <Alert className="border-amber-500/50 bg-amber-500/5">
      <AlertCircle className="h-4 w-4 text-amber-500" />
      <AlertTitle className="text-amber-700 dark:text-amber-400 font-semibold">
        Wellness Monitoring - Not Medical Advice
      </AlertTitle>
      <AlertDescription className="text-amber-600 dark:text-amber-300 text-sm">
        This data is for personal wellness tracking only. Consult your healthcare provider for medical decisions.
      </AlertDescription>
    </Alert>
  );
}

function EmptyStatePrompt({ 
  icon: Icon, 
  title, 
  description, 
  actionLabel, 
  actionHref 
}: { 
  icon: any; 
  title: string; 
  description: string; 
  actionLabel: string; 
  actionHref: string;
}) {
  return (
    <Card className="border-dashed">
      <CardContent className="p-8 text-center">
        <div className="mx-auto w-12 h-12 rounded-full bg-muted flex items-center justify-center mb-4">
          <Icon className="h-6 w-6 text-muted-foreground" />
        </div>
        <h3 className="font-semibold text-lg mb-2">{title}</h3>
        <p className="text-muted-foreground text-sm mb-4 max-w-md mx-auto">{description}</p>
        <Link href={actionHref}>
          <Button className="gap-2" data-testid="button-start-tracking">
            <Play className="h-4 w-4" />
            {actionLabel}
          </Button>
        </Link>
      </CardContent>
    </Card>
  );
}

function StatCard({ label, value, unit, icon: Icon, trend, color = "text-primary" }: {
  label: string;
  value: string | number;
  unit?: string;
  icon: any;
  trend?: 'up' | 'down' | 'stable';
  color?: string;
}) {
  return (
    <div className="p-4 rounded-lg border bg-card">
      <div className="flex items-center justify-between mb-2">
        <Icon className={`h-4 w-4 ${color}`} />
        {trend && (
          <div className={`flex items-center gap-1 text-xs ${
            trend === 'up' ? 'text-chart-2' : trend === 'down' ? 'text-destructive' : 'text-muted-foreground'
          }`}>
            {trend === 'up' ? <TrendingUp className="h-3 w-3" /> : 
             trend === 'down' ? <TrendingDown className="h-3 w-3" /> : null}
          </div>
        )}
      </div>
      <div className="text-2xl font-bold">{value}{unit && <span className="text-sm font-normal text-muted-foreground ml-1">{unit}</span>}</div>
      <div className="text-xs text-muted-foreground">{label}</div>
    </div>
  );
}

interface AutopilotState {
  patient_id: string;
  risk_score: number;
  risk_state: string;
  risk_components: Record<string, number>;
  top_risk_components: Array<{ name: string; value: number }>;
  next_followup_at: string | null;
  last_updated: string | null;
  last_checkin_at: string | null;
  model_version: string;
  confidence: number;
}

interface AutopilotTask {
  id: string;
  task_description: string;
  priority: string;
  trigger_type: string;
  completed: boolean;
  created_at: string;
}

interface AutopilotData {
  patient_state: AutopilotState;
  today_tasks: AutopilotTask[];
  pending_task_count: number;
  has_urgent_tasks: boolean;
  wellness_disclaimer: string;
}

function getRiskColor(score: number): string {
  if (score <= 3) return 'text-chart-2';
  if (score <= 6) return 'text-yellow-500';
  if (score <= 9) return 'text-orange-500';
  return 'text-destructive';
}

function getRiskBgColor(score: number): string {
  if (score <= 3) return 'bg-chart-2/10 border-chart-2/30';
  if (score <= 6) return 'bg-yellow-500/10 border-yellow-500/30';
  if (score <= 9) return 'bg-orange-500/10 border-orange-500/30';
  return 'bg-destructive/10 border-destructive/30';
}

function getPriorityBadge(priority: string) {
  switch (priority?.toLowerCase()) {
    case 'critical':
    case 'high':
      return <Badge variant="destructive" className="text-xs">Urgent</Badge>;
    case 'medium':
      return <Badge className="bg-yellow-500/20 text-yellow-600 text-xs">Medium</Badge>;
    default:
      return <Badge variant="secondary" className="text-xs">Normal</Badge>;
  }
}

function AutopilotSection({ 
  data, 
  isLoading,
  isError, 
  onRefresh, 
  onTaskComplete,
  isRefreshing,
  isBackendStarting,
  retryCount
}: { 
  data: AutopilotData | undefined;
  isLoading: boolean;
  isError?: boolean;
  onRefresh: () => void;
  onTaskComplete: (taskId: string) => void;
  isRefreshing: boolean;
  isBackendStarting?: boolean;
  retryCount?: number;
}) {
  const [isExpanded, setIsExpanded] = useState(false);
  
  if (isLoading || isBackendStarting) {
    return (
      <Card className="border-primary/30 bg-primary/5">
        <CardContent className="p-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-full bg-primary/10">
                <Loader2 className="h-5 w-5 animate-spin text-primary" />
              </div>
              <div>
                <p className="font-medium">
                  {isBackendStarting ? 'Starting Secure Services' : 'Loading Autopilot'}
                </p>
                <p className="text-sm text-muted-foreground">
                  {isBackendStarting 
                    ? 'Initializing HIPAA-compliant AI models. This may take up to 45 seconds on first load.'
                    : 'Loading your personalized follow-up status...'}
                </p>
                {retryCount && retryCount > 0 && (
                  <p className="text-xs text-muted-foreground mt-1">
                    Connection attempt {retryCount}/5...
                  </p>
                )}
              </div>
            </div>
            <Shield className="h-6 w-6 text-primary/50" />
          </div>
          {isBackendStarting && (
            <Progress value={(retryCount || 1) * 20} className="mt-4 h-1" />
          )}
        </CardContent>
      </Card>
    );
  }

  if (isError && !isBackendStarting) {
    return (
      <Card className="border-destructive/50">
        <CardContent className="p-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-full bg-destructive/10">
                <AlertTriangle className="h-5 w-5 text-destructive" />
              </div>
              <div>
                <p className="font-medium">Autopilot Unavailable</p>
                <p className="text-sm text-muted-foreground">
                  Unable to load your personalized follow-up data. Please try again.
                </p>
              </div>
            </div>
            <Button variant="outline" size="sm" onClick={onRefresh} data-testid="button-retry-autopilot">
              <RefreshCw className={`h-4 w-4 mr-2 ${isRefreshing ? 'animate-spin' : ''}`} />
              Retry
            </Button>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (!data) {
    return (
      <Card className="border-dashed">
        <CardContent className="p-6">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-full bg-muted">
              <Target className="h-5 w-5 text-muted-foreground" />
            </div>
            <div>
              <p className="font-medium">Autopilot Initializing</p>
              <p className="text-sm text-muted-foreground">
                Your personalized follow-up system is learning from your health data.
              </p>
            </div>
          </div>
        </CardContent>
      </Card>
    );
  }

  const { patient_state, today_tasks, pending_task_count, has_urgent_tasks } = data;
  const riskScore = patient_state?.risk_score || 0;
  const riskState = patient_state?.risk_state || 'Stable';

  return (
    <Collapsible open={isExpanded} onOpenChange={setIsExpanded}>
      <Card className={`${getRiskBgColor(riskScore)} transition-colors`}>
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className={`p-2 rounded-full ${riskScore <= 3 ? 'bg-chart-2/20' : riskScore <= 6 ? 'bg-yellow-500/20' : 'bg-destructive/20'}`}>
                <Sparkles className={`h-5 w-5 ${getRiskColor(riskScore)}`} />
              </div>
              <div>
                <CardTitle className="flex items-center gap-2" data-testid="text-autopilot-title">
                  Followup Autopilot
                  {has_urgent_tasks && (
                    <Badge variant="destructive" className="text-xs animate-pulse">
                      <Bell className="h-3 w-3 mr-1" />
                      Urgent
                    </Badge>
                  )}
                </CardTitle>
                <CardDescription>
                  ML-powered adaptive follow-up system
                </CardDescription>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <Button 
                variant="ghost" 
                size="icon"
                onClick={(e) => { e.stopPropagation(); onRefresh(); }}
                disabled={isRefreshing}
                data-testid="button-refresh-autopilot"
              >
                <RefreshCw className={`h-4 w-4 ${isRefreshing ? 'animate-spin' : ''}`} />
              </Button>
              <CollapsibleTrigger asChild>
                <Button variant="ghost" size="icon" data-testid="button-expand-autopilot">
                  {isExpanded ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
                </Button>
              </CollapsibleTrigger>
            </div>
          </div>
        </CardHeader>
        
        <CardContent className="space-y-4">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="p-3 rounded-lg bg-background/50 border">
              <div className="flex items-center gap-2 mb-1">
                <Target className={`h-4 w-4 ${getRiskColor(riskScore)}`} />
                <span className="text-xs text-muted-foreground">Risk Score</span>
              </div>
              <div className={`text-2xl font-bold ${getRiskColor(riskScore)}`} data-testid="text-risk-score">
                {riskScore.toFixed(1)}
                <span className="text-sm font-normal text-muted-foreground">/15</span>
              </div>
            </div>
            
            <div className="p-3 rounded-lg bg-background/50 border">
              <div className="flex items-center gap-2 mb-1">
                <Shield className="h-4 w-4 text-primary" />
                <span className="text-xs text-muted-foreground">Status</span>
              </div>
              <div className="text-lg font-semibold" data-testid="text-risk-state">{riskState}</div>
            </div>
            
            <div className="p-3 rounded-lg bg-background/50 border">
              <div className="flex items-center gap-2 mb-1">
                <ListChecks className="h-4 w-4 text-primary" />
                <span className="text-xs text-muted-foreground">Pending Tasks</span>
              </div>
              <div className="text-lg font-semibold" data-testid="text-pending-tasks">{pending_task_count}</div>
            </div>
            
            <div className="p-3 rounded-lg bg-background/50 border">
              <div className="flex items-center gap-2 mb-1">
                <Clock className="h-4 w-4 text-primary" />
                <span className="text-xs text-muted-foreground">Next Follow-up</span>
              </div>
              <div className="text-sm font-medium" data-testid="text-next-followup">
                {patient_state?.next_followup_at 
                  ? format(new Date(patient_state.next_followup_at), 'MMM d, h:mm a')
                  : 'Not scheduled'}
              </div>
            </div>
          </div>

          {patient_state?.top_risk_components?.length > 0 && (
            <div className="p-3 rounded-lg bg-background/50 border">
              <p className="text-xs text-muted-foreground mb-2">Top Contributing Factors</p>
              <div className="flex flex-wrap gap-2">
                {patient_state.top_risk_components.map((component, idx) => (
                  <Badge key={idx} variant="outline" className="text-xs">
                    {component.name}: {component.value.toFixed(0)}%
                  </Badge>
                ))}
              </div>
            </div>
          )}

          {today_tasks?.length > 0 && (
            <div className="space-y-2">
              <p className="text-sm font-medium flex items-center gap-2">
                <ListChecks className="h-4 w-4" />
                Today's Adaptive Tasks
              </p>
              <div className="space-y-2">
                {today_tasks.slice(0, isExpanded ? undefined : 3).map((task) => (
                  <div 
                    key={task.id} 
                    className={`flex items-start gap-3 p-3 rounded-lg border bg-background/50 ${
                      task.completed ? 'opacity-60' : ''
                    }`}
                    data-testid={`task-item-${task.id}`}
                  >
                    <Checkbox
                      checked={task.completed}
                      onCheckedChange={() => !task.completed && onTaskComplete(task.id)}
                      disabled={task.completed}
                      data-testid={`checkbox-task-${task.id}`}
                    />
                    <div className="flex-1 min-w-0">
                      <p className={`text-sm ${task.completed ? 'line-through text-muted-foreground' : ''}`}>
                        {task.task_description}
                      </p>
                      <div className="flex items-center gap-2 mt-1">
                        {getPriorityBadge(task.priority)}
                        <span className="text-xs text-muted-foreground">
                          {task.trigger_type?.replace(/_/g, ' ')}
                        </span>
                      </div>
                    </div>
                    {task.completed && (
                      <CheckCircle2 className="h-4 w-4 text-chart-2 shrink-0" />
                    )}
                  </div>
                ))}
              </div>
              {today_tasks.length > 3 && !isExpanded && (
                <Button 
                  variant="ghost" 
                  size="sm" 
                  className="w-full text-xs"
                  onClick={() => setIsExpanded(true)}
                  data-testid="button-show-more-tasks"
                >
                  Show {today_tasks.length - 3} more tasks
                </Button>
              )}
            </div>
          )}

          <CollapsibleContent className="space-y-4 pt-2">
            {patient_state?.risk_components && Object.keys(patient_state.risk_components).length > 0 && (
              <div className="p-4 rounded-lg bg-background/50 border">
                <p className="text-sm font-medium mb-3">Risk Breakdown by Category</p>
                <div className="space-y-3">
                  {Object.entries(patient_state.risk_components)
                    .sort(([,a], [,b]) => b - a)
                    .map(([category, value]) => (
                      <div key={category} className="space-y-1">
                        <div className="flex items-center justify-between text-sm">
                          <span className="capitalize">{category.replace(/_/g, ' ')}</span>
                          <span className={getRiskColor(value / 6.67)}>{value.toFixed(0)}%</span>
                        </div>
                        <Progress value={value} className="h-2" />
                      </div>
                    ))}
                </div>
              </div>
            )}
            
            <div className="p-3 rounded-lg bg-background/50 border">
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div>
                  <span className="text-muted-foreground">Model Version:</span>
                  <span className="ml-2 font-mono">{patient_state?.model_version || '1.0.0'}</span>
                </div>
                <div>
                  <span className="text-muted-foreground">Confidence:</span>
                  <span className="ml-2">{((patient_state?.confidence || 0.5) * 100).toFixed(0)}%</span>
                </div>
                <div>
                  <span className="text-muted-foreground">Last Updated:</span>
                  <span className="ml-2">
                    {patient_state?.last_updated 
                      ? format(new Date(patient_state.last_updated), 'MMM d, h:mm a')
                      : 'N/A'}
                  </span>
                </div>
                <div>
                  <span className="text-muted-foreground">Last Check-in:</span>
                  <span className="ml-2">
                    {patient_state?.last_checkin_at 
                      ? format(new Date(patient_state.last_checkin_at), 'MMM d, h:mm a')
                      : 'N/A'}
                  </span>
                </div>
              </div>
            </div>
          </CollapsibleContent>
        </CardContent>
      </Card>
    </Collapsible>
  );
}

export default function DailyFollowupHistory() {
  const { user } = useAuth();
  const [timeRange, setTimeRange] = useState('30');
  const [activeTab, setActiveTab] = useState('device');

  const selectedRange = TIME_RANGES.find(r => r.value === timeRange) || TIME_RANGES[2];
  const daysLimit = selectedRange.days === Infinity ? 3650 : selectedRange.days;

  const { data: deviceHistory, isLoading: deviceLoading } = useQuery<any[]>({
    queryKey: ['/api/daily-followup/history', { limit: daysLimit }],
    enabled: !!user,
  });

  const { data: symptomFeed, isLoading: symptomsLoading } = useQuery<any[]>({
    queryKey: ['/api/symptom-checkin/feed/unified', { days: daysLimit }],
    enabled: !!user,
  });

  const { data: paintrackSessions, isLoading: paintrackLoading } = useQuery<any[]>({
    queryKey: ['/api/paintrack/sessions', { limit: daysLimit }],
    enabled: !!user,
  });

  const { data: mentalHealthHistory, isLoading: mentalHealthLoading } = useQuery<any>({
    queryKey: ['/api/v1/mental-health/questionnaires/history', { limit: daysLimit }],
    enabled: !!user,
  });

  const { data: voiceFollowups, isLoading: audioLoading } = useQuery<any[]>({
    queryKey: ['/api/voice-followup/recent', { limit: daysLimit }],
    enabled: !!user,
  });

  const { data: videoMetrics, isLoading: videoMetricsLoading } = useQuery<any>({
    queryKey: ['/api/video-ai/latest-metrics'],
    enabled: !!user,
  });

  const { data: videoSessions, isLoading: videoSessionsLoading } = useQuery<any[]>({
    queryKey: ['/api/video-ai/exam-sessions', { days: daysLimit }],
    enabled: !!user,
  });

  const videoLoading = videoMetricsLoading || videoSessionsLoading;

  const patientId = user?.id || user?.sub || 'demo-patient';
  const { toast } = useToast();
  const [isRefreshingAutopilot, setIsRefreshingAutopilot] = useState(false);

  const { data: autopilotData, isLoading: autopilotLoading, isError: autopilotError, refetch: refetchAutopilot, isFetching, status, fetchStatus } = useQuery<AutopilotData>({
    queryKey: [`/api/v1/followup-autopilot/patients/${patientId}/autopilot`],
    enabled: !!user && !!patientId && patientId !== 'demo-patient',
    retry: 5,
    retryDelay: (attemptIndex) => Math.min(2000 * Math.pow(1.5, attemptIndex), 15000),
    staleTime: 30000,
    gcTime: 60000,
  });

  const isAutopilotStarting = status === 'pending' && fetchStatus === 'fetching' && !autopilotData;
  const autopilotRetryCount = 0;

  const completeTaskMutation = useMutation({
    mutationFn: async (taskId: string) => {
      return apiRequest('POST', `/api/v1/followup-autopilot/patients/${patientId}/tasks/${taskId}/complete`, {});
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: [`/api/v1/followup-autopilot/patients/${patientId}/autopilot`] });
      toast({
        title: "Task Completed",
        description: "Great job completing your follow-up task!",
      });
    },
    onError: () => {
      toast({
        title: "Error",
        description: "Failed to complete task. Please try again.",
        variant: "destructive",
      });
    },
  });

  const handleRefreshAutopilot = async () => {
    setIsRefreshingAutopilot(true);
    try {
      await refetchAutopilot();
    } finally {
      setIsRefreshingAutopilot(false);
    }
  };

  const handleTaskComplete = (taskId: string) => {
    completeTaskMutation.mutate(taskId);
  };

  const filterByTimeRange = (data: any[], dateField: string = 'createdAt') => {
    if (!data || selectedRange.days === Infinity) return data || [];
    const cutoffDate = subDays(new Date(), selectedRange.days);
    return data.filter(item => {
      const itemDate = new Date(item[dateField] || item.timestamp || item.completed_at);
      return itemDate >= cutoffDate;
    });
  };

  const filteredDeviceHistory = useMemo(() => filterByTimeRange(deviceHistory || []), [deviceHistory, timeRange]);
  const filteredSymptoms = useMemo(() => {
    const filtered = filterByTimeRange(symptomFeed || [], 'timestamp');
    return filtered.filter((s: any) => s.dataSource === 'patient-reported');
  }, [symptomFeed, timeRange]);
  const filteredPaintrack = useMemo(() => filterByTimeRange(paintrackSessions || []), [paintrackSessions, timeRange]);
  const filteredMentalHealth = useMemo(() => {
    const history = mentalHealthHistory?.history || [];
    return filterByTimeRange(history, 'completed_at');
  }, [mentalHealthHistory, timeRange]);
  const filteredVoice = useMemo(() => filterByTimeRange(voiceFollowups || []), [voiceFollowups, timeRange]);
  const filteredVideoSessions = useMemo(() => filterByTimeRange(videoSessions || [], 'started_at'), [videoSessions, timeRange]);

  const hasAnyData = (filteredDeviceHistory?.length > 0) || 
                     (filteredSymptoms?.length > 0) || 
                     (filteredPaintrack?.length > 0) || 
                     (filteredMentalHealth?.length > 0) ||
                     (filteredVoice?.length > 0) ||
                     (filteredVideoSessions?.length > 0) ||
                     videoMetrics;

  const isNewPatient = !hasAnyData && !deviceLoading && !symptomsLoading && !paintrackLoading && !mentalHealthLoading;

  const deviceChartData = useMemo(() => {
    if (!filteredDeviceHistory?.length) return [];
    return filteredDeviceHistory
      .slice()
      .reverse()
      .map((d: any) => ({
        date: format(new Date(d.createdAt || d.date), 'MMM d'),
        heartRate: d.heartRate,
        spo2: d.oxygenSaturation,
        temperature: d.temperature,
        steps: d.steps,
      }));
  }, [filteredDeviceHistory]);

  const symptomChartData = useMemo(() => {
    if (!filteredSymptoms?.length) return [];
    const grouped: Record<string, any> = {};
    filteredSymptoms.forEach((s: any) => {
      const date = format(new Date(s.timestamp), 'MMM d');
      if (!grouped[date]) {
        grouped[date] = { date, painTotal: 0, fatigueTotal: 0, count: 0 };
      }
      grouped[date].painTotal += s.painLevel || 0;
      grouped[date].fatigueTotal += s.fatigueLevel || 0;
      grouped[date].count++;
    });
    return Object.values(grouped).map((g: any) => ({
      date: g.date,
      painLevel: g.count > 0 ? (g.painTotal / g.count).toFixed(1) : 0,
      fatigueLevel: g.count > 0 ? (g.fatigueTotal / g.count).toFixed(1) : 0,
    }));
  }, [filteredSymptoms]);

  const painChartData = useMemo(() => {
    if (!filteredPaintrack?.length) return [];
    return filteredPaintrack
      .slice()
      .reverse()
      .map((p: any) => ({
        date: format(new Date(p.createdAt), 'MMM d'),
        vas: p.patientVas,
        joint: p.joint,
      }));
  }, [filteredPaintrack]);

  const videoSessionsChartData = useMemo(() => {
    if (!filteredVideoSessions?.length) return [];
    return filteredVideoSessions
      .slice()
      .reverse()
      .map((s: any) => ({
        date: format(new Date(s.started_at), 'MMM d'),
        completed: s.completed_segments || 0,
        total: s.total_segments || 7,
        status: s.status,
      }));
  }, [filteredVideoSessions]);

  const mentalHealthChartData = useMemo(() => {
    if (!filteredMentalHealth?.length) return [];
    return filteredMentalHealth
      .slice()
      .reverse()
      .map((m: any) => ({
        date: format(new Date(m.completed_at), 'MMM d'),
        score: m.total_score,
        type: m.questionnaire_type,
        severity: m.severity_level,
      }));
  }, [filteredMentalHealth]);

  const audioChartData = useMemo(() => {
    if (!filteredVoice?.length) return [];
    return filteredVoice
      .slice()
      .reverse()
      .map((v: any) => ({
        date: format(new Date(v.createdAt), 'MMM d'),
        empathy: v.empathyLevel || 0,
        symptoms: v.extractedSymptoms?.length || 0,
        needsFollowup: v.needsFollowup ? 1 : 0,
      }));
  }, [filteredVoice]);

  const calculateTrendFromValues = (values: number[]): 'up' | 'down' | 'stable' => {
    if (!values?.length || values.length < 2) return 'stable';
    const recentCount = Math.max(1, Math.min(7, Math.ceil(values.length / 2)));
    const olderCount = Math.max(1, values.length - recentCount);
    const recentAvg = values.slice(-recentCount).reduce((a, b) => a + b, 0) / recentCount;
    const olderAvg = values.slice(0, olderCount).reduce((a, b) => a + b, 0) / olderCount;
    if (olderAvg === 0) return recentAvg > 0 ? 'up' : 'stable';
    return recentAvg > olderAvg * 1.05 ? 'up' : recentAvg < olderAvg * 0.95 ? 'down' : 'stable';
  };

  const videoCompletionTrend = useMemo(() => {
    const values = (filteredVideoSessions || []).map((s: any) => s.completed_segments).filter((v: any) => v != null);
    return calculateTrendFromValues(values);
  }, [filteredVideoSessions]);
  
  const audioEmpathyTrend = useMemo(() => {
    const values = (filteredVoice || []).map((v: any) => v.empathyLevel).filter((v: any) => v != null && !isNaN(v));
    return calculateTrendFromValues(values);
  }, [filteredVoice]);
  
  const mentalHealthTrendCalc = useMemo(() => {
    const values = (filteredMentalHealth || []).map((m: any) => m.total_score).filter((v: any) => v != null);
    return calculateTrendFromValues(values);
  }, [filteredMentalHealth]);

  const getSeverityColor = (severity: string) => {
    switch (severity?.toLowerCase()) {
      case 'none':
      case 'minimal':
        return 'bg-chart-2/20 text-chart-2';
      case 'mild':
        return 'bg-yellow-500/20 text-yellow-600';
      case 'moderate':
        return 'bg-orange-500/20 text-orange-600';
      case 'moderately_severe':
      case 'severe':
        return 'bg-destructive/20 text-destructive';
      default:
        return 'bg-muted text-muted-foreground';
    }
  };

  const calculateStats = (data: any[], field: string) => {
    if (!data?.length) return { avg: 0, min: 0, max: 0, trend: 'stable' as const };
    const values = data.map(d => d[field]).filter(v => v != null);
    if (!values.length) return { avg: 0, min: 0, max: 0, trend: 'stable' as const };
    const avg = values.reduce((a, b) => a + b, 0) / values.length;
    const min = Math.min(...values);
    const max = Math.max(...values);
    const recentAvg = values.slice(-7).reduce((a, b) => a + b, 0) / Math.min(7, values.length);
    const olderAvg = values.slice(0, -7).reduce((a, b) => a + b, 0) / Math.max(1, values.length - 7);
    const trend = recentAvg > olderAvg * 1.05 ? 'up' : recentAvg < olderAvg * 0.95 ? 'down' : 'stable';
    return { avg: avg.toFixed(1), min, max, trend: trend as 'up' | 'down' | 'stable' };
  };

  if (!user) {
    return (
      <div className="container mx-auto p-6 max-w-7xl">
        <Card>
          <CardContent className="p-12 text-center">
            <AlertCircle className="h-12 w-12 mx-auto mb-4 text-muted-foreground" />
            <h2 className="text-xl font-semibold mb-2">Sign In Required</h2>
            <p className="text-muted-foreground mb-4">Please sign in to view your health history.</p>
            <Link href="/auth">
              <Button data-testid="button-sign-in">Sign In</Button>
            </Link>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="container mx-auto p-6 max-w-7xl space-y-6">
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
        <div className="space-y-1">
          <h1 className="text-3xl font-bold tracking-tight flex items-center gap-3" data-testid="text-page-title">
            <Activity className="h-8 w-8 text-primary" />
            Daily Follow-up History
          </h1>
          <p className="text-muted-foreground">
            Track your health trends over time across all wellness categories
          </p>
        </div>
        <div className="flex items-center gap-3">
          <Select value={timeRange} onValueChange={setTimeRange}>
            <SelectTrigger className="w-[140px]" data-testid="select-time-range">
              <Calendar className="h-4 w-4 mr-2" />
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {TIME_RANGES.map(range => (
                <SelectItem key={range.value} value={range.value}>
                  {range.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </div>

      <LegalDisclaimer />

      <AutopilotSection
        data={autopilotData}
        isLoading={autopilotLoading}
        isError={autopilotError}
        onRefresh={handleRefreshAutopilot}
        onTaskComplete={handleTaskComplete}
        isRefreshing={isRefreshingAutopilot || completeTaskMutation.isPending}
        isBackendStarting={isAutopilotStarting}
        retryCount={autopilotRetryCount}
      />

      {isNewPatient && (
        <Card className="border-primary/20 bg-primary/5">
          <CardContent className="p-8 text-center">
            <div className="mx-auto w-16 h-16 rounded-full bg-primary/10 flex items-center justify-center mb-4">
              <Heart className="h-8 w-8 text-primary" />
            </div>
            <h2 className="text-2xl font-bold mb-2">Welcome to Your Health Journey</h2>
            <p className="text-muted-foreground mb-6 max-w-lg mx-auto">
              Start tracking your daily wellness to build a comprehensive picture of your health. 
              Complete activities in any tab below to begin seeing your trends and insights.
            </p>
            <Link href="/">
              <Button size="lg" className="gap-2" data-testid="button-go-to-dashboard">
                <Play className="h-5 w-5" />
                Go to Dashboard to Start
              </Button>
            </Link>
          </CardContent>
        </Card>
      )}

      <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-4">
        <TabsList className="w-full flex gap-2 h-auto p-2 flex-wrap">
          <TabsTrigger value="device" data-testid="tab-device-history" className="flex-1 min-w-[100px]">
            <Heart className="h-3 w-3 mr-1" />
            Device Data
          </TabsTrigger>
          <TabsTrigger value="symptoms" data-testid="tab-symptoms-history" className="flex-1 min-w-[100px]">
            <Activity className="h-3 w-3 mr-1" />
            Symptoms
          </TabsTrigger>
          <TabsTrigger value="video-ai" data-testid="tab-video-history" className="flex-1 min-w-[100px]">
            <Video className="h-3 w-3 mr-1" />
            Video AI
          </TabsTrigger>
          <TabsTrigger value="audio-ai" data-testid="tab-audio-history" className="flex-1 min-w-[100px]">
            <Mic className="h-3 w-3 mr-1" />
            Audio AI
          </TabsTrigger>
          <TabsTrigger value="paintrack" data-testid="tab-paintrack-history" className="flex-1 min-w-[100px]">
            <Zap className="h-3 w-3 mr-1" />
            PainTrack
          </TabsTrigger>
          <TabsTrigger value="mental-health" data-testid="tab-mental-health-history" className="flex-1 min-w-[100px]">
            <Brain className="h-3 w-3 mr-1" />
            Mental Health
          </TabsTrigger>
        </TabsList>

        <TabsContent value="device" className="space-y-4">
          <Tabs defaultValue="medical-devices" className="w-full">
            <TabsList className="grid w-full grid-cols-2 max-w-md">
              <TabsTrigger value="medical-devices" data-testid="subtab-medical-devices">
                <Heart className="h-3 w-3 mr-1" />
                Medical Devices
              </TabsTrigger>
              <TabsTrigger value="wearable-trends" data-testid="subtab-wearable-trends">
                <BarChart3 className="h-3 w-3 mr-1" />
                Wearable Trends
              </TabsTrigger>
            </TabsList>

            <TabsContent value="medical-devices" className="mt-4">
              <DeviceDataManager />
            </TabsContent>

            <TabsContent value="wearable-trends" className="mt-4 space-y-4">
              {deviceLoading ? (
                <Card><CardContent className="p-8 text-center"><Loader2 className="h-8 w-8 animate-spin mx-auto" /></CardContent></Card>
              ) : filteredDeviceHistory?.length > 0 ? (
                <>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <StatCard 
                      label="Avg Heart Rate" 
                      value={calculateStats(filteredDeviceHistory, 'heartRate').avg} 
                      unit="bpm"
                      icon={Heart}
                      trend={calculateStats(filteredDeviceHistory, 'heartRate').trend}
                      color="text-rose-500"
                    />
                    <StatCard 
                      label="Avg SpO2" 
                      value={calculateStats(filteredDeviceHistory, 'oxygenSaturation').avg} 
                      unit="%"
                      icon={Wind}
                      trend={calculateStats(filteredDeviceHistory, 'oxygenSaturation').trend}
                      color="text-blue-500"
                    />
                    <StatCard 
                      label="Avg Temperature" 
                      value={calculateStats(filteredDeviceHistory, 'temperature').avg} 
                      unit="°F"
                      icon={Thermometer}
                      trend={calculateStats(filteredDeviceHistory, 'temperature').trend}
                      color="text-orange-500"
                    />
                    <StatCard 
                      label="Total Steps" 
                      value={filteredDeviceHistory.reduce((sum, d) => sum + (d.steps || 0), 0).toLocaleString()} 
                      icon={Footprints}
                      color="text-chart-2"
                    />
                  </div>

                  <Card>
                    <CardHeader>
                      <CardTitle className="flex items-center gap-2">
                        <BarChart3 className="h-5 w-5" />
                        Device Data Trends
                      </CardTitle>
                      <CardDescription>
                        {filteredDeviceHistory.length} data points over {selectedRange.label.toLowerCase()}
                      </CardDescription>
                    </CardHeader>
                    <CardContent>
                      <div className="h-[300px]">
                        <ResponsiveContainer width="100%" height="100%">
                          <LineChart data={deviceChartData}>
                            <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                            <XAxis dataKey="date" className="text-xs" />
                            <YAxis className="text-xs" />
                            <Tooltip 
                              contentStyle={{ 
                                backgroundColor: 'hsl(var(--card))', 
                                border: '1px solid hsl(var(--border))',
                                borderRadius: '8px'
                              }} 
                            />
                            <Legend />
                            <Line type="monotone" dataKey="heartRate" name="Heart Rate (bpm)" stroke="hsl(var(--chart-1))" strokeWidth={2} dot={false} />
                            <Line type="monotone" dataKey="spo2" name="SpO2 (%)" stroke="hsl(var(--chart-2))" strokeWidth={2} dot={false} />
                          </LineChart>
                        </ResponsiveContainer>
                      </div>
                    </CardContent>
                  </Card>

                  <Card>
                    <CardHeader>
                      <CardTitle>Recent Entries</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <ScrollArea className="h-[200px]">
                        <div className="space-y-2">
                          {filteredDeviceHistory.slice(0, 10).map((entry: any, idx: number) => (
                            <div key={idx} className="flex items-center justify-between p-3 rounded-lg bg-muted/50">
                              <div className="flex items-center gap-3">
                                <Clock className="h-4 w-4 text-muted-foreground" />
                                <span className="text-sm">{format(new Date(entry.createdAt || entry.date), 'MMM d, yyyy h:mm a')}</span>
                              </div>
                              <div className="flex items-center gap-4 text-sm">
                                {entry.heartRate && <span><Heart className="h-3 w-3 inline mr-1 text-rose-500" />{entry.heartRate} bpm</span>}
                                {entry.oxygenSaturation && <span><Wind className="h-3 w-3 inline mr-1 text-blue-500" />{entry.oxygenSaturation}%</span>}
                                {entry.steps && <span><Footprints className="h-3 w-3 inline mr-1 text-chart-2" />{entry.steps.toLocaleString()}</span>}
                              </div>
                            </div>
                          ))}
                        </div>
                      </ScrollArea>
                    </CardContent>
                  </Card>
                </>
              ) : (
                <EmptyStatePrompt
                  icon={Heart}
                  title="No Wearable Data Yet"
                  description="Sync your wearable device to start tracking heart rate, SpO2, temperature, and activity levels."
                  actionLabel="Connect Wearable"
                  actionHref="/wearables"
                />
              )}
            </TabsContent>
          </Tabs>
        </TabsContent>

        <TabsContent value="symptoms" className="space-y-4">
          {symptomsLoading ? (
            <Card><CardContent className="p-8 text-center"><Loader2 className="h-8 w-8 animate-spin mx-auto" /></CardContent></Card>
          ) : filteredSymptoms?.length > 0 ? (
            <>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <StatCard 
                  label="Avg Pain Level" 
                  value={calculateStats(filteredSymptoms, 'painLevel').avg} 
                  unit="/10"
                  icon={Zap}
                  trend={calculateStats(filteredSymptoms, 'painLevel').trend}
                  color="text-orange-500"
                />
                <StatCard 
                  label="Avg Fatigue" 
                  value={calculateStats(filteredSymptoms, 'fatigueLevel').avg} 
                  unit="/10"
                  icon={Activity}
                  trend={calculateStats(filteredSymptoms, 'fatigueLevel').trend}
                  color="text-yellow-500"
                />
                <StatCard 
                  label="Check-ins" 
                  value={filteredSymptoms.length} 
                  icon={CheckCircle2}
                  color="text-chart-2"
                />
                <StatCard 
                  label="Avg Mood" 
                  value={(() => {
                    const moodValues = filteredSymptoms.filter((s: any) => s.mood).map((s: any) => {
                      const moodMap: Record<string, number> = { poor: 1, fair: 2, good: 3, great: 4, excellent: 5 };
                      return moodMap[s.mood?.toLowerCase()] || 3;
                    });
                    return moodValues.length > 0 ? (moodValues.reduce((a, b) => a + b, 0) / moodValues.length).toFixed(1) : '-';
                  })()} 
                  unit="/5"
                  icon={Smile}
                  color="text-primary"
                />
              </div>

              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <BarChart3 className="h-5 w-5" />
                    Symptom Trends
                  </CardTitle>
                  <CardDescription>
                    Pain and fatigue levels over {selectedRange.label.toLowerCase()}
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="h-[300px]">
                    <ResponsiveContainer width="100%" height="100%">
                      <AreaChart data={symptomChartData}>
                        <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                        <XAxis dataKey="date" className="text-xs" />
                        <YAxis domain={[0, 10]} className="text-xs" />
                        <Tooltip 
                          contentStyle={{ 
                            backgroundColor: 'hsl(var(--card))', 
                            border: '1px solid hsl(var(--border))',
                            borderRadius: '8px'
                          }} 
                        />
                        <Legend />
                        <Area type="monotone" dataKey="painLevel" name="Pain Level" stroke="hsl(var(--chart-1))" fill="hsl(var(--chart-1) / 0.2)" />
                        <Area type="monotone" dataKey="fatigueLevel" name="Fatigue Level" stroke="hsl(var(--chart-3))" fill="hsl(var(--chart-3) / 0.2)" />
                      </AreaChart>
                    </ResponsiveContainer>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Recent Symptom Check-ins</CardTitle>
                </CardHeader>
                <CardContent>
                  <ScrollArea className="h-[250px]">
                    <div className="space-y-3">
                      {filteredSymptoms.slice(0, 15).map((symptom: any, idx: number) => (
                        <div key={idx} className="p-3 rounded-lg border bg-card">
                          <div className="flex items-center justify-between mb-2">
                            <span className="text-sm font-medium">{format(new Date(symptom.timestamp), 'MMM d, yyyy h:mm a')}</span>
                            <Badge variant="secondary" className="text-xs">Patient-reported</Badge>
                          </div>
                          <div className="grid grid-cols-3 gap-2 text-sm">
                            {symptom.painLevel != null && (
                              <div><span className="text-muted-foreground">Pain:</span> {symptom.painLevel}/10</div>
                            )}
                            {symptom.fatigueLevel != null && (
                              <div><span className="text-muted-foreground">Fatigue:</span> {symptom.fatigueLevel}/10</div>
                            )}
                            {symptom.mood && (
                              <div><span className="text-muted-foreground">Mood:</span> {symptom.mood}</div>
                            )}
                          </div>
                          {symptom.note && (
                            <p className="text-xs text-muted-foreground mt-2 italic">"{symptom.note}"</p>
                          )}
                        </div>
                      ))}
                    </div>
                  </ScrollArea>
                </CardContent>
              </Card>
            </>
          ) : (
            <EmptyStatePrompt
              icon={Activity}
              title="No Symptom Check-ins Yet"
              description="Log your daily symptoms including pain, fatigue, mood, and other health indicators to track patterns over time."
              actionLabel="Start Symptom Check-in"
              actionHref="/"
            />
          )}
        </TabsContent>

        <TabsContent value="video-ai" className="space-y-4">
          {videoLoading ? (
            <Card><CardContent className="p-8 text-center"><Loader2 className="h-8 w-8 animate-spin mx-auto" /></CardContent></Card>
          ) : (filteredVideoSessions?.length > 0 || videoMetrics) ? (
            <>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <StatCard 
                  label="Total Examinations" 
                  value={filteredVideoSessions?.length || 0} 
                  icon={Video}
                  color="text-primary"
                />
                <StatCard 
                  label="Completed" 
                  value={filteredVideoSessions?.filter((s: any) => s.status === 'completed').length || 0} 
                  icon={CheckCircle2}
                  color="text-chart-2"
                />
                <StatCard 
                  label="Avg Segments" 
                  value={filteredVideoSessions?.length > 0 
                    ? (filteredVideoSessions.reduce((sum: number, s: any) => sum + (s.completed_segments || 0), 0) / filteredVideoSessions.length).toFixed(1)
                    : '0'
                  } 
                  unit="/7"
                  icon={Camera}
                  trend={videoCompletionTrend}
                  color="text-blue-500"
                />
                <StatCard 
                  label="This Week" 
                  value={filteredVideoSessions?.filter((s: any) => {
                    const sessionDate = new Date(s.started_at);
                    const weekAgo = subDays(new Date(), 7);
                    return sessionDate >= weekAgo;
                  }).length || 0} 
                  icon={Calendar}
                  color="text-purple-500"
                />
              </div>

              {filteredVideoSessions?.length > 1 && (
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <BarChart3 className="h-5 w-5" />
                      Examination Completion Trends
                    </CardTitle>
                    <CardDescription>
                      Segments completed per examination over {selectedRange.label.toLowerCase()}
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="h-[250px]">
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={videoSessionsChartData}>
                          <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                          <XAxis dataKey="date" className="text-xs" />
                          <YAxis domain={[0, 7]} className="text-xs" />
                          <Tooltip 
                            contentStyle={{ 
                              backgroundColor: 'hsl(var(--card))', 
                              border: '1px solid hsl(var(--border))',
                              borderRadius: '8px'
                            }}
                            formatter={(value: any, name: any) => [
                              `${value}/7 segments`,
                              'Completed'
                            ]}
                          />
                          <Bar dataKey="completed" name="Completed Segments" fill="hsl(var(--chart-2))" radius={[4, 4, 0, 0]} />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </CardContent>
                </Card>
              )}

              {videoMetrics && (
                <Card>
                  <CardHeader>
                    <div className="flex items-center justify-between">
                      <div>
                        <CardTitle className="flex items-center gap-2">
                          <Video className="h-5 w-5 text-primary" />
                          Latest AI Metrics
                        </CardTitle>
                        <CardDescription>
                          Last examination: {videoMetrics.created_at ? format(new Date(videoMetrics.created_at), 'MMM d, yyyy h:mm a') : 'N/A'}
                        </CardDescription>
                      </div>
                      <Link href="/ai-video">
                        <Button variant="outline" size="sm" className="gap-2" data-testid="button-view-full-video-analysis">
                          Full Analysis
                          <ChevronRight className="h-4 w-4" />
                        </Button>
                      </Link>
                    </div>
                  </CardHeader>
                  <CardContent>
                    <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                      {videoMetrics.respiratory_rate_bpm && (
                        <div className="p-3 rounded-lg border bg-muted/30">
                          <div className="flex items-center gap-2 mb-1">
                            <Wind className="h-3 w-3 text-blue-500" />
                            <span className="text-xs text-muted-foreground">Respiratory</span>
                          </div>
                          <div className="text-lg font-bold">{videoMetrics.respiratory_rate_bpm.toFixed(1)} bpm</div>
                        </div>
                      )}
                      {videoMetrics.skin_pallor_score != null && (
                        <div className="p-3 rounded-lg border bg-muted/30">
                          <div className="flex items-center gap-2 mb-1">
                            <Palette className="h-3 w-3 text-amber-500" />
                            <span className="text-xs text-muted-foreground">Pallor</span>
                          </div>
                          <div className="text-lg font-bold">{videoMetrics.skin_pallor_score.toFixed(1)}/100</div>
                        </div>
                      )}
                      {videoMetrics.jaundice_risk_level && (
                        <div className="p-3 rounded-lg border bg-muted/30">
                          <div className="flex items-center gap-2 mb-1">
                            <Eye className="h-3 w-3 text-yellow-500" />
                            <span className="text-xs text-muted-foreground">Jaundice</span>
                          </div>
                          <div className="text-lg font-bold capitalize">{videoMetrics.jaundice_risk_level}</div>
                        </div>
                      )}
                      {videoMetrics.facial_swelling_score != null && (
                        <div className="p-3 rounded-lg border bg-muted/30">
                          <div className="flex items-center gap-2 mb-1">
                            <User className="h-3 w-3 text-rose-500" />
                            <span className="text-xs text-muted-foreground">Swelling</span>
                          </div>
                          <div className="text-lg font-bold">{videoMetrics.facial_swelling_score.toFixed(1)}/100</div>
                        </div>
                      )}
                      {videoMetrics.tremor_detected !== undefined && (
                        <div className="p-3 rounded-lg border bg-muted/30">
                          <div className="flex items-center gap-2 mb-1">
                            <Hand className="h-3 w-3 text-purple-500" />
                            <span className="text-xs text-muted-foreground">Tremor</span>
                          </div>
                          <div className="text-lg font-bold">{videoMetrics.tremor_detected ? 'Detected' : 'None'}</div>
                        </div>
                      )}
                      {videoMetrics.tongue_coating_detected !== undefined && (
                        <div className="p-3 rounded-lg border bg-muted/30">
                          <div className="flex items-center gap-2 mb-1">
                            <MessageSquare className="h-3 w-3 text-pink-500" />
                            <span className="text-xs text-muted-foreground">Tongue</span>
                          </div>
                          <div className="text-lg font-bold">{videoMetrics.tongue_coating_detected ? 'Coating' : 'Normal'}</div>
                        </div>
                      )}
                    </div>
                  </CardContent>
                </Card>
              )}

              <Card>
                <CardHeader>
                  <CardTitle>Examination History</CardTitle>
                  <CardDescription>{filteredVideoSessions?.length || 0} examinations in {selectedRange.label.toLowerCase()}</CardDescription>
                </CardHeader>
                <CardContent>
                  <ScrollArea className="h-[200px]">
                    <div className="space-y-2">
                      {filteredVideoSessions?.map((session: any, idx: number) => (
                        <div key={idx} className="flex items-center justify-between p-3 rounded-lg border bg-card">
                          <div className="flex items-center gap-3">
                            <div className={`p-2 rounded-full ${session.status === 'completed' ? 'bg-chart-2/20' : 'bg-muted'}`}>
                              {session.status === 'completed' ? (
                                <CheckCircle2 className="h-4 w-4 text-chart-2" />
                              ) : (
                                <Clock className="h-4 w-4 text-muted-foreground" />
                              )}
                            </div>
                            <div>
                              <div className="text-sm font-medium">{format(new Date(session.started_at), 'MMM d, yyyy h:mm a')}</div>
                              <div className="text-xs text-muted-foreground">
                                {session.completed_segments}/{session.total_segments} segments
                              </div>
                            </div>
                          </div>
                          <Badge variant={session.status === 'completed' ? 'default' : 'secondary'} className="text-xs capitalize">
                            {session.status}
                          </Badge>
                        </div>
                      ))}
                      {(!filteredVideoSessions || filteredVideoSessions.length === 0) && (
                        <div className="text-center py-4 text-muted-foreground text-sm">
                          No examination sessions in selected time range
                        </div>
                      )}
                    </div>
                  </ScrollArea>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Start New Video Examination</CardTitle>
                  <CardDescription>Record a 7-stage AI-guided video examination to track changes</CardDescription>
                </CardHeader>
                <CardContent>
                  <Link href="/daily-followup/video-exam">
                    <Button className="w-full gap-2" data-testid="button-start-video-exam">
                      <Camera className="h-4 w-4" />
                      Start Video Examination
                    </Button>
                  </Link>
                </CardContent>
              </Card>
            </>
          ) : (
            <EmptyStatePrompt
              icon={Video}
              title="No Video AI Analysis Yet"
              description="Complete a guided video examination to get AI-powered analysis of respiratory rate, skin pallor, jaundice indicators, and more."
              actionLabel="Start Video Examination"
              actionHref="/daily-followup/video-exam"
            />
          )}
        </TabsContent>

        <TabsContent value="audio-ai" className="space-y-4">
          {audioLoading ? (
            <Card><CardContent className="p-8 text-center"><Loader2 className="h-8 w-8 animate-spin mx-auto" /></CardContent></Card>
          ) : filteredVoice?.length > 0 ? (
            <>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <StatCard 
                  label="Voice Recordings" 
                  value={filteredVoice.length} 
                  icon={Mic}
                  color="text-purple-500"
                />
                <StatCard 
                  label="Follow-ups Needed" 
                  value={filteredVoice.filter((v: any) => v.needsFollowup).length} 
                  icon={AlertTriangle}
                  color="text-amber-500"
                />
                <StatCard 
                  label="Avg Empathy Level" 
                  value={(() => {
                    const levels = filteredVoice.filter((v: any) => v.empathyLevel).map((v: any) => v.empathyLevel);
                    return levels.length > 0 ? (levels.reduce((a: number, b: number) => a + b, 0) / levels.length).toFixed(1) : '-';
                  })()} 
                  unit="/10"
                  icon={Heart}
                  trend={audioEmpathyTrend}
                  color="text-rose-500"
                />
                <StatCard 
                  label="Symptoms Extracted" 
                  value={filteredVoice.reduce((sum: number, v: any) => sum + (v.extractedSymptoms?.length || 0), 0)} 
                  icon={Activity}
                  color="text-primary"
                />
              </div>

              {audioChartData.length > 1 && (
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <BarChart3 className="h-5 w-5" />
                      Voice Analysis Trends
                    </CardTitle>
                    <CardDescription>
                      Empathy levels and symptoms extracted over {selectedRange.label.toLowerCase()}
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="h-[300px]">
                      <ResponsiveContainer width="100%" height="100%">
                        <AreaChart data={audioChartData}>
                          <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                          <XAxis dataKey="date" className="text-xs" />
                          <YAxis className="text-xs" />
                          <Tooltip 
                            contentStyle={{ 
                              backgroundColor: 'hsl(var(--card))', 
                              border: '1px solid hsl(var(--border))',
                              borderRadius: '8px'
                            }} 
                          />
                          <Legend />
                          <Area type="monotone" dataKey="empathy" name="Empathy Level" stroke="hsl(var(--chart-4))" fill="hsl(var(--chart-4) / 0.2)" />
                          <Area type="monotone" dataKey="symptoms" name="Symptoms Extracted" stroke="hsl(var(--chart-2))" fill="hsl(var(--chart-2) / 0.2)" />
                        </AreaChart>
                      </ResponsiveContainer>
                    </div>
                  </CardContent>
                </Card>
              )}

              <Card>
                <CardHeader>
                  <CardTitle>Voice Follow-up History</CardTitle>
                  <CardDescription>{filteredVoice.length} recordings over {selectedRange.label.toLowerCase()}</CardDescription>
                </CardHeader>
                <CardContent>
                  <ScrollArea className="h-[300px]">
                    <div className="space-y-3">
                      {filteredVoice.map((voice: any, idx: number) => (
                        <div key={idx} className="p-4 rounded-lg border bg-card">
                          <div className="flex items-center justify-between mb-3">
                            <span className="text-sm font-medium">{format(new Date(voice.createdAt), 'MMM d, yyyy h:mm a')}</span>
                            {voice.needsFollowup && (
                              <Badge variant="destructive" className="text-xs">Needs Follow-up</Badge>
                            )}
                          </div>
                          {voice.conversationSummary && (
                            <p className="text-sm mb-3">{voice.conversationSummary}</p>
                          )}
                          <div className="grid grid-cols-2 gap-2 text-xs">
                            {voice.extractedMood && (
                              <div><span className="text-muted-foreground">Mood:</span> {voice.extractedMood}</div>
                            )}
                            {voice.empathyLevel && (
                              <div><span className="text-muted-foreground">Empathy:</span> {voice.empathyLevel}/10</div>
                            )}
                          </div>
                          {voice.extractedSymptoms?.length > 0 && (
                            <div className="flex flex-wrap gap-1 mt-2">
                              {voice.extractedSymptoms.map((symptom: string, sIdx: number) => (
                                <Badge key={sIdx} variant="secondary" className="text-xs">{symptom}</Badge>
                              ))}
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  </ScrollArea>
                </CardContent>
              </Card>
            </>
          ) : (
            <EmptyStatePrompt
              icon={Mic}
              title="No Audio AI Analysis Yet"
              description="Record voice follow-ups to let AI analyze your speech patterns, extract symptoms, and track your emotional well-being."
              actionLabel="Start Voice Recording"
              actionHref="/"
            />
          )}
        </TabsContent>

        <TabsContent value="paintrack" className="space-y-4">
          {paintrackLoading ? (
            <Card><CardContent className="p-8 text-center"><Loader2 className="h-8 w-8 animate-spin mx-auto" /></CardContent></Card>
          ) : filteredPaintrack?.length > 0 ? (
            <>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <StatCard 
                  label="Avg Pain (VAS)" 
                  value={calculateStats(filteredPaintrack, 'patientVas').avg} 
                  unit="/10"
                  icon={Zap}
                  trend={calculateStats(filteredPaintrack, 'patientVas').trend}
                  color="text-orange-500"
                />
                <StatCard 
                  label="Sessions" 
                  value={filteredPaintrack.length} 
                  icon={FileText}
                  color="text-primary"
                />
                <StatCard 
                  label="Max Pain" 
                  value={calculateStats(filteredPaintrack, 'patientVas').max} 
                  unit="/10"
                  icon={TrendingUp}
                  color="text-destructive"
                />
                <StatCard 
                  label="Min Pain" 
                  value={calculateStats(filteredPaintrack, 'patientVas').min} 
                  unit="/10"
                  icon={TrendingDown}
                  color="text-chart-2"
                />
              </div>

              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <BarChart3 className="h-5 w-5" />
                    Pain Level Trends
                  </CardTitle>
                  <CardDescription>
                    VAS scores over {selectedRange.label.toLowerCase()}
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="h-[300px]">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={painChartData}>
                        <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                        <XAxis dataKey="date" className="text-xs" />
                        <YAxis domain={[0, 10]} className="text-xs" />
                        <Tooltip 
                          contentStyle={{ 
                            backgroundColor: 'hsl(var(--card))', 
                            border: '1px solid hsl(var(--border))',
                            borderRadius: '8px'
                          }}
                          formatter={(value: any, name: any, props: any) => [
                            `${value}/10 (${props.payload.joint})`,
                            'Pain Level'
                          ]}
                        />
                        <Bar dataKey="vas" name="Pain Level" fill="hsl(var(--chart-1))" radius={[4, 4, 0, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Recent Pain Assessments</CardTitle>
                </CardHeader>
                <CardContent>
                  <ScrollArea className="h-[250px]">
                    <div className="space-y-3">
                      {filteredPaintrack.slice(0, 15).map((session: any, idx: number) => (
                        <div key={idx} className="p-3 rounded-lg border bg-card">
                          <div className="flex items-center justify-between mb-2">
                            <div className="flex items-center gap-2">
                              <Zap className="h-4 w-4 text-orange-500" />
                              <span className="font-medium capitalize">{session.laterality} {session.joint}</span>
                            </div>
                            <div className="flex items-center gap-2">
                              <span className="text-2xl font-bold">{session.patientVas}</span>
                              <span className="text-sm text-muted-foreground">/10</span>
                            </div>
                          </div>
                          <div className="flex items-center justify-between text-sm text-muted-foreground">
                            <span>{format(new Date(session.createdAt), 'MMM d, yyyy h:mm a')}</span>
                            <span className="capitalize">{session.module}</span>
                          </div>
                          {session.patientNotes && (
                            <p className="text-xs text-muted-foreground mt-2 italic">"{session.patientNotes}"</p>
                          )}
                        </div>
                      ))}
                    </div>
                  </ScrollArea>
                </CardContent>
              </Card>
            </>
          ) : (
            <EmptyStatePrompt
              icon={Zap}
              title="No Pain Tracking Data Yet"
              description="Track chronic pain with detailed assessments including joint selection, medication tracking, and video documentation."
              actionLabel="Start Pain Assessment"
              actionHref="/"
            />
          )}
        </TabsContent>

        <TabsContent value="mental-health" className="space-y-4">
          {mentalHealthLoading ? (
            <Card><CardContent className="p-8 text-center"><Loader2 className="h-8 w-8 animate-spin mx-auto" /></CardContent></Card>
          ) : filteredMentalHealth?.length > 0 ? (
            <>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <StatCard 
                  label="Assessments" 
                  value={filteredMentalHealth.length} 
                  icon={Brain}
                  color="text-purple-500"
                />
                <StatCard 
                  label="Avg Score" 
                  value={calculateStats(filteredMentalHealth, 'total_score').avg} 
                  icon={BarChart3}
                  trend={mentalHealthTrendCalc}
                  color="text-primary"
                />
                <StatCard 
                  label="PHQ-9 Count" 
                  value={filteredMentalHealth.filter((m: any) => m.questionnaire_type === 'PHQ-9').length} 
                  icon={FileText}
                  color="text-blue-500"
                />
                <StatCard 
                  label="GAD-7 Count" 
                  value={filteredMentalHealth.filter((m: any) => m.questionnaire_type === 'GAD-7').length} 
                  icon={FileText}
                  color="text-amber-500"
                />
              </div>

              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <BarChart3 className="h-5 w-5" />
                    Mental Health Score Trends
                  </CardTitle>
                  <CardDescription>
                    Questionnaire scores over {selectedRange.label.toLowerCase()}
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="h-[300px]">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={mentalHealthChartData}>
                        <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                        <XAxis dataKey="date" className="text-xs" />
                        <YAxis className="text-xs" />
                        <Tooltip 
                          contentStyle={{ 
                            backgroundColor: 'hsl(var(--card))', 
                            border: '1px solid hsl(var(--border))',
                            borderRadius: '8px'
                          }}
                          formatter={(value: any, name: any, props: any) => [
                            `${value} (${props.payload.type})`,
                            'Score'
                          ]}
                        />
                        <Legend />
                        <Line type="monotone" dataKey="score" name="Score" stroke="hsl(var(--chart-4))" strokeWidth={2} dot />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Assessment History</CardTitle>
                </CardHeader>
                <CardContent>
                  <ScrollArea className="h-[250px]">
                    <div className="space-y-3">
                      {filteredMentalHealth.map((assessment: any, idx: number) => (
                        <div key={idx} className="p-3 rounded-lg border bg-card">
                          <div className="flex items-center justify-between mb-2">
                            <div className="flex items-center gap-2">
                              <Brain className="h-4 w-4 text-purple-500" />
                              <span className="font-medium">{assessment.questionnaire_type}</span>
                            </div>
                            <div className="flex items-center gap-2">
                              <span className="text-2xl font-bold">{assessment.total_score}</span>
                              <span className="text-sm text-muted-foreground">/{assessment.max_score}</span>
                            </div>
                          </div>
                          <div className="flex items-center justify-between">
                            <span className="text-sm text-muted-foreground">
                              {format(new Date(assessment.completed_at), 'MMM d, yyyy h:mm a')}
                            </span>
                            <Badge className={getSeverityColor(assessment.severity_level)}>
                              {assessment.severity_level?.replace(/_/g, ' ')}
                            </Badge>
                          </div>
                          {assessment.crisis_detected && (
                            <Alert className="mt-2 p-2 border-destructive/50 bg-destructive/5">
                              <AlertTriangle className="h-3 w-3 text-destructive" />
                              <AlertDescription className="text-xs text-destructive">
                                Crisis indicators were detected in this assessment
                              </AlertDescription>
                            </Alert>
                          )}
                        </div>
                      ))}
                    </div>
                  </ScrollArea>
                </CardContent>
              </Card>

              <Link href="/mental-health">
                <Card className="hover-elevate cursor-pointer">
                  <CardContent className="p-4 flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <Brain className="h-5 w-5 text-primary" />
                      <div>
                        <p className="font-medium">View Full Mental Health Dashboard</p>
                        <p className="text-sm text-muted-foreground">Access detailed analysis, clusters, and AI insights</p>
                      </div>
                    </div>
                    <ChevronRight className="h-5 w-5 text-muted-foreground" />
                  </CardContent>
                </Card>
              </Link>
            </>
          ) : (
            <EmptyStatePrompt
              icon={Brain}
              title="No Mental Health Assessments Yet"
              description="Complete PHQ-9, GAD-7, or PSS-10 questionnaires to track your mental wellness indicators over time."
              actionLabel="Start Assessment"
              actionHref="/"
            />
          )}
        </TabsContent>
      </Tabs>
    </div>
  );
}
