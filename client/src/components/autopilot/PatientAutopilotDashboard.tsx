import { useState } from 'react';
import { useQuery, useMutation } from '@tanstack/react-query';
import { queryClient, apiRequest } from '@/lib/queryClient';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Switch } from '@/components/ui/switch';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Skeleton } from '@/components/ui/skeleton';
import { Progress } from '@/components/ui/progress';
import { Separator } from '@/components/ui/separator';
import { useToast } from '@/hooks/use-toast';
import {
  Activity,
  Bell,
  BellOff,
  CheckCircle2,
  Clock,
  AlertTriangle,
  AlertCircle,
  TrendingUp,
  TrendingDown,
  Heart,
  Brain,
  Pill,
  Shield,
  RefreshCw,
  Loader2,
  Info,
  ChevronRight,
  Zap,
  Calendar,
  Settings,
} from 'lucide-react';
import { format, formatDistanceToNow } from 'date-fns';
import { RiskHistoryChart, TriggerEventsCard, NotificationsCard } from './AutopilotInsightsTab';

interface PatientState {
  risk_score: number;
  risk_state: string;
  risk_components: Record<string, number>;
  next_followup_at: string | null;
  last_updated: string | null;
  confidence: number;
}

interface Task {
  id: string;
  task_type: string;
  priority: string;
  status: string;
  title: string;
  description: string;
  scheduled_at: string;
  created_at: string;
}

interface Trigger {
  id: string;
  name: string;
  severity: string;
  created_at: string;
}

interface Recommendation {
  type: string;
  priority: string;
  message: string;
  action: string;
}

interface DashboardSummary {
  patient_id: string;
  state: PatientState;
  tasks: {
    pending: Task[];
    count: number;
    has_urgent: boolean;
  };
  recent_triggers: Trigger[];
  signal_summary: Record<string, { count: number; avg_score: number }>;
  recommendations: Recommendation[];
  wellness_disclaimer: string;
}

interface NotificationPreferences {
  in_app_enabled: boolean;
  push_enabled: boolean;
  email_enabled: boolean;
  sms_enabled: boolean;
  health_alerts_enabled: boolean;
  med_reminders_enabled: boolean;
  daily_followups_enabled: boolean;
  preferred_contact_hour: number | null;
  quiet_hours_start: number;
  quiet_hours_end: number;
  urgency_threshold: string;
}

function getRiskStateColor(state: string): string {
  switch (state?.toLowerCase()) {
    case 'critical': return 'text-red-600 dark:text-red-400';
    case 'worsening': return 'text-orange-600 dark:text-orange-400';
    case 'atrisk': return 'text-yellow-600 dark:text-yellow-400';
    default: return 'text-green-600 dark:text-green-400';
  }
}

function getRiskStateBadge(state: string) {
  switch (state?.toLowerCase()) {
    case 'critical':
      return <Badge variant="destructive" data-testid="badge-risk-critical"><AlertTriangle className="h-3 w-3 mr-1" />Critical</Badge>;
    case 'worsening':
      return <Badge className="bg-orange-500/20 text-orange-600 dark:text-orange-400" data-testid="badge-risk-worsening"><AlertCircle className="h-3 w-3 mr-1" />Worsening</Badge>;
    case 'atrisk':
      return <Badge className="bg-yellow-500/20 text-yellow-600 dark:text-yellow-400" data-testid="badge-risk-atrisk"><Info className="h-3 w-3 mr-1" />At Risk</Badge>;
    default:
      return <Badge className="bg-green-500/20 text-green-600 dark:text-green-400" data-testid="badge-risk-stable"><Shield className="h-3 w-3 mr-1" />Stable</Badge>;
  }
}

function getPriorityBadge(priority: string) {
  switch (priority?.toLowerCase()) {
    case 'critical':
      return <Badge variant="destructive" className="text-xs">Critical</Badge>;
    case 'high':
      return <Badge className="bg-orange-500/20 text-orange-600 dark:text-orange-400 text-xs">High</Badge>;
    case 'medium':
      return <Badge className="bg-blue-500/20 text-blue-600 dark:text-blue-400 text-xs">Medium</Badge>;
    default:
      return <Badge variant="secondary" className="text-xs">Low</Badge>;
  }
}

function getComponentIcon(name: string) {
  switch (name.toLowerCase()) {
    case 'clinical': return <Heart className="h-4 w-4" />;
    case 'mental_health': return <Brain className="h-4 w-4" />;
    case 'adherence': return <Pill className="h-4 w-4" />;
    case 'anomaly': return <Zap className="h-4 w-4" />;
    default: return <Activity className="h-4 w-4" />;
  }
}

function RiskOverviewCard({ state, onRefresh, isRefreshing }: { 
  state: PatientState; 
  onRefresh: () => void;
  isRefreshing: boolean;
}) {
  const riskScore = state?.risk_score || 0;
  const riskState = state?.risk_state || 'Stable';
  const components = state?.risk_components || {};
  
  return (
    <Card data-testid="card-risk-overview">
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between gap-2">
          <CardTitle className="flex items-center gap-2">
            <Activity className="h-5 w-5" />
            Wellness Status
          </CardTitle>
          <Button 
            variant="ghost" 
            size="icon" 
            onClick={onRefresh}
            disabled={isRefreshing}
            data-testid="button-refresh-status"
          >
            {isRefreshing ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <RefreshCw className="h-4 w-4" />
            )}
          </Button>
        </div>
        <CardDescription>
          AI-powered wellness monitoring (not medical advice)
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-sm text-muted-foreground">Risk Score</p>
            <p className={`text-3xl font-bold ${getRiskStateColor(riskState)}`} data-testid="text-risk-score">
              {riskScore.toFixed(0)}
            </p>
          </div>
          <div className="text-right">
            {getRiskStateBadge(riskState)}
            {state?.confidence && (
              <p className="text-xs text-muted-foreground mt-1">
                {(state.confidence * 100).toFixed(0)}% confidence
              </p>
            )}
          </div>
        </div>
        
        <Progress 
          value={Math.min(riskScore, 100)} 
          className="h-2"
          data-testid="progress-risk-score"
        />
        
        <div className="grid grid-cols-2 gap-3">
          {Object.entries(components).slice(0, 4).map(([name, value]) => (
            <div key={name} className="flex items-center gap-2 text-sm">
              {getComponentIcon(name)}
              <span className="text-muted-foreground capitalize">
                {name.replace('_', ' ')}:
              </span>
              <span className="font-medium">{typeof value === 'number' ? value.toFixed(0) : value}%</span>
            </div>
          ))}
        </div>
        
        {state?.next_followup_at && (
          <div className="flex items-center gap-2 text-sm text-muted-foreground pt-2 border-t">
            <Calendar className="h-4 w-4" />
            <span>Next check-in: {formatDistanceToNow(new Date(state.next_followup_at), { addSuffix: true })}</span>
          </div>
        )}
        
        {state?.last_updated && (
          <p className="text-xs text-muted-foreground">
            Last updated: {format(new Date(state.last_updated), 'MMM d, h:mm a')}
          </p>
        )}
      </CardContent>
    </Card>
  );
}

function TasksCard({ tasks, onComplete }: { 
  tasks: Task[]; 
  onComplete: (taskId: string) => void;
}) {
  if (!tasks || tasks.length === 0) {
    return (
      <Card data-testid="card-tasks-empty">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-base">
            <CheckCircle2 className="h-5 w-5" />
            Today's Tasks
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center py-6 text-muted-foreground">
            <CheckCircle2 className="h-10 w-10 mx-auto mb-2 text-green-500" />
            <p>All caught up!</p>
            <p className="text-sm">No pending wellness tasks</p>
          </div>
        </CardContent>
      </Card>
    );
  }
  
  return (
    <Card data-testid="card-tasks">
      <CardHeader className="pb-2">
        <CardTitle className="flex items-center justify-between text-base">
          <span className="flex items-center gap-2">
            <Clock className="h-5 w-5" />
            Today's Tasks
          </span>
          <Badge variant="secondary">{tasks.length}</Badge>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <ScrollArea className="h-[200px]">
          <div className="space-y-3">
            {tasks.map((task) => (
              <div 
                key={task.id} 
                className="flex items-start justify-between gap-2 p-3 rounded-lg border bg-card"
                data-testid={`task-item-${task.id}`}
              >
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-1">
                    {getPriorityBadge(task.priority)}
                    <span className="text-sm font-medium truncate">
                      {task.title || task.task_type.replace(/_/g, ' ')}
                    </span>
                  </div>
                  {task.description && (
                    <p className="text-xs text-muted-foreground line-clamp-2">
                      {task.description}
                    </p>
                  )}
                </div>
                <Button 
                  size="sm" 
                  variant="outline"
                  onClick={() => onComplete(task.id)}
                  data-testid={`button-complete-task-${task.id}`}
                >
                  <CheckCircle2 className="h-4 w-4" />
                </Button>
              </div>
            ))}
          </div>
        </ScrollArea>
      </CardContent>
    </Card>
  );
}

function RecommendationsCard({ recommendations }: { recommendations: Recommendation[] }) {
  if (!recommendations || recommendations.length === 0) return null;
  
  return (
    <Card data-testid="card-recommendations">
      <CardHeader className="pb-2">
        <CardTitle className="flex items-center gap-2 text-base">
          <TrendingUp className="h-5 w-5" />
          Recommendations
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-2">
          {recommendations.map((rec, idx) => (
            <div 
              key={idx}
              className="flex items-center justify-between p-3 rounded-lg border bg-card hover-elevate cursor-pointer"
              data-testid={`recommendation-${idx}`}
            >
              <div className="flex items-center gap-3">
                {rec.type === 'wellness_check' && <Activity className="h-5 w-5 text-blue-500" />}
                {rec.type === 'medication' && <Pill className="h-5 w-5 text-purple-500" />}
                {rec.type === 'wellness' && <Brain className="h-5 w-5 text-green-500" />}
                <span className="text-sm">{rec.message}</span>
              </div>
              <ChevronRight className="h-4 w-4 text-muted-foreground" />
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}

function SignalSummaryCard({ signals }: { signals: Record<string, { count: number; avg_score: number }> }) {
  if (!signals || Object.keys(signals).length === 0) return null;
  
  const signalLabels: Record<string, string> = {
    device: 'Device Data',
    symptom: 'Symptoms',
    video: 'Video Check-in',
    audio: 'Voice Analysis',
    pain: 'Pain Tracking',
    mental: 'Mental Health',
    meds: 'Medications',
    environment: 'Environment',
    exposure: 'Exposures',
  };
  
  return (
    <Card data-testid="card-signal-summary">
      <CardHeader className="pb-2">
        <CardTitle className="flex items-center gap-2 text-base">
          <Zap className="h-5 w-5" />
          Recent Activity
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-2 gap-2">
          {Object.entries(signals).map(([category, data]) => (
            <div 
              key={category}
              className="flex items-center justify-between p-2 rounded border bg-muted/30"
              data-testid={`signal-${category}`}
            >
              <span className="text-xs text-muted-foreground">
                {signalLabels[category] || category}
              </span>
              <Badge variant="secondary" className="text-xs">
                {data.count}
              </Badge>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}

function PreferencesPanel({ patientId }: { patientId: string }) {
  const { toast } = useToast();
  
  const { data, isLoading } = useQuery<{ preferences: NotificationPreferences }>({
    queryKey: ['/api/v1/followup-autopilot/patients', patientId, 'preferences'],
    queryFn: async () => {
      const res = await fetch(`/api/v1/followup-autopilot/patients/${encodeURIComponent(patientId)}/preferences`);
      if (!res.ok) throw new Error('Failed to fetch preferences');
      return res.json();
    },
    enabled: !!patientId,
  });
  
  const updatePrefs = useMutation({
    mutationFn: async (prefs: Partial<NotificationPreferences>) => {
      return apiRequest('PATCH', `/api/v1/followup-autopilot/patients/${encodeURIComponent(patientId)}/preferences`, prefs);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['/api/v1/followup-autopilot/patients', patientId, 'preferences'] });
      toast({ title: 'Preferences updated' });
    },
    onError: () => {
      toast({ title: 'Failed to update preferences', variant: 'destructive' });
    },
  });
  
  if (isLoading) {
    return (
      <div className="space-y-4">
        <Skeleton className="h-12 w-full" />
        <Skeleton className="h-12 w-full" />
        <Skeleton className="h-12 w-full" />
      </div>
    );
  }
  
  const prefs = data?.preferences || {
    in_app_enabled: true,
    push_enabled: true,
    email_enabled: true,
    sms_enabled: false,
    preferred_contact_hour: null,
  };
  
  return (
    <div className="space-y-6" data-testid="panel-preferences">
      <div>
        <h3 className="text-lg font-semibold mb-4">Notification Channels</h3>
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Bell className="h-4 w-4 text-muted-foreground" />
              <Label htmlFor="in-app">In-App Notifications</Label>
            </div>
            <Switch 
              id="in-app"
              checked={prefs.in_app_enabled}
              onCheckedChange={(checked) => updatePrefs.mutate({ in_app_enabled: checked })}
              data-testid="switch-in-app"
            />
          </div>
          
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Bell className="h-4 w-4 text-muted-foreground" />
              <Label htmlFor="push">Push Notifications</Label>
            </div>
            <Switch 
              id="push"
              checked={prefs.push_enabled}
              onCheckedChange={(checked) => updatePrefs.mutate({ push_enabled: checked })}
              data-testid="switch-push"
            />
          </div>
          
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Bell className="h-4 w-4 text-muted-foreground" />
              <Label htmlFor="email">Email Notifications</Label>
            </div>
            <Switch 
              id="email"
              checked={prefs.email_enabled}
              onCheckedChange={(checked) => updatePrefs.mutate({ email_enabled: checked })}
              data-testid="switch-email"
            />
          </div>
          
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Bell className="h-4 w-4 text-muted-foreground" />
              <Label htmlFor="sms">SMS Notifications</Label>
            </div>
            <Switch 
              id="sms"
              checked={prefs.sms_enabled}
              onCheckedChange={(checked) => updatePrefs.mutate({ sms_enabled: checked })}
              data-testid="switch-sms"
            />
          </div>
        </div>
      </div>
      
      <Separator />
      
      <div>
        <h3 className="text-lg font-semibold mb-4">Contact Preferences</h3>
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <Label htmlFor="contact-hour">Preferred Contact Hour</Label>
            <Select
              value={prefs.preferred_contact_hour?.toString() || 'any'}
              onValueChange={(value) => {
                const hour = value === 'any' ? null : parseInt(value, 10);
                updatePrefs.mutate({ preferred_contact_hour: hour });
              }}
            >
              <SelectTrigger className="w-[140px]" data-testid="select-contact-hour">
                <SelectValue placeholder="Any time" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="any">Any time</SelectItem>
                <SelectItem value="8">8:00 AM</SelectItem>
                <SelectItem value="9">9:00 AM</SelectItem>
                <SelectItem value="10">10:00 AM</SelectItem>
                <SelectItem value="12">12:00 PM</SelectItem>
                <SelectItem value="14">2:00 PM</SelectItem>
                <SelectItem value="16">4:00 PM</SelectItem>
                <SelectItem value="18">6:00 PM</SelectItem>
                <SelectItem value="20">8:00 PM</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </div>
      </div>
    </div>
  );
}

export function PatientAutopilotDashboard({ patientId }: { patientId: string }) {
  const { toast } = useToast();
  const [activeTab, setActiveTab] = useState('overview');
  
  const { data, isLoading, isError, refetch, isFetching } = useQuery<DashboardSummary>({
    queryKey: ['/api/v1/followup-autopilot/patients', patientId, 'summary'],
    queryFn: async () => {
      const res = await fetch(`/api/v1/followup-autopilot/patients/${encodeURIComponent(patientId)}/summary`);
      if (!res.ok) throw new Error('Failed to fetch dashboard');
      return res.json();
    },
    enabled: !!patientId,
    staleTime: 60000,
    refetchInterval: 300000,
  });
  
  const completeTask = useMutation({
    mutationFn: async (taskId: string) => {
      return apiRequest('POST', `/api/v1/followup-autopilot/patients/${encodeURIComponent(patientId)}/tasks/${encodeURIComponent(taskId)}/complete`, {});
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['/api/v1/followup-autopilot/patients', patientId, 'summary'] });
      toast({ title: 'Task completed' });
    },
    onError: () => {
      toast({ title: 'Failed to complete task', variant: 'destructive' });
    },
  });
  
  if (isLoading) {
    return (
      <div className="space-y-4" data-testid="dashboard-loading">
        <Skeleton className="h-[200px] w-full" />
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <Skeleton className="h-[250px]" />
          <Skeleton className="h-[250px]" />
        </div>
      </div>
    );
  }
  
  if (isError) {
    return (
      <Card className="border-destructive/50" data-testid="dashboard-error">
        <CardContent className="p-6 text-center">
          <AlertTriangle className="h-10 w-10 text-destructive mx-auto mb-2" />
          <p className="font-medium">Unable to load wellness dashboard</p>
          <p className="text-sm text-muted-foreground mb-4">Please try again</p>
          <Button variant="outline" onClick={() => refetch()} data-testid="button-retry">
            <RefreshCw className="h-4 w-4 mr-2" />
            Retry
          </Button>
        </CardContent>
      </Card>
    );
  }
  
  const summary = data!;
  
  return (
    <div className="space-y-4" data-testid="patient-autopilot-dashboard">
      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="overview" data-testid="tab-overview">
            <Activity className="h-4 w-4 mr-2" />
            Overview
          </TabsTrigger>
          <TabsTrigger value="history" data-testid="tab-history">
            <TrendingUp className="h-4 w-4 mr-2" />
            History
          </TabsTrigger>
          <TabsTrigger value="settings" data-testid="tab-settings">
            <Settings className="h-4 w-4 mr-2" />
            Settings
          </TabsTrigger>
        </TabsList>
        
        <TabsContent value="overview" className="space-y-4 mt-4">
          <RiskOverviewCard 
            state={summary.state} 
            onRefresh={() => refetch()}
            isRefreshing={isFetching}
          />
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <TasksCard 
              tasks={summary.tasks.pending} 
              onComplete={(taskId) => completeTask.mutate(taskId)}
            />
            <RecommendationsCard recommendations={summary.recommendations} />
          </div>
          
          <SignalSummaryCard signals={summary.signal_summary} />
          
          {summary.recent_triggers.length > 0 && (
            <Card data-testid="card-recent-alerts">
              <CardHeader className="pb-2">
                <CardTitle className="flex items-center gap-2 text-base">
                  <AlertCircle className="h-5 w-5" />
                  Recent Alerts
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  {summary.recent_triggers.slice(0, 3).map((trigger) => (
                    <div 
                      key={trigger.id}
                      className="flex items-center justify-between p-2 rounded border"
                      data-testid={`alert-${trigger.id}`}
                    >
                      <span className="text-sm">
                        {trigger.name.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}
                      </span>
                      <span className="text-xs text-muted-foreground">
                        {formatDistanceToNow(new Date(trigger.created_at), { addSuffix: true })}
                      </span>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}
          
          <p className="text-xs text-muted-foreground text-center pt-2">
            {summary.wellness_disclaimer}
          </p>
        </TabsContent>
        
        <TabsContent value="history" className="mt-4">
          <RiskHistoryChart patientId={patientId} />
        </TabsContent>
        
        <TabsContent value="settings" className="mt-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Settings className="h-5 w-5" />
                Notification Preferences
              </CardTitle>
              <CardDescription>
                Customize how and when you receive wellness updates
              </CardDescription>
            </CardHeader>
            <CardContent>
              <PreferencesPanel patientId={patientId} />
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}

export default PatientAutopilotDashboard;
