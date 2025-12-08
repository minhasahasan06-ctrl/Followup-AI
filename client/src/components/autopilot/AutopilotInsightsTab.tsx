import { useState } from 'react';
import { useQuery, useMutation } from '@tanstack/react-query';
import { queryClient, apiRequest } from '@/lib/queryClient';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Skeleton } from '@/components/ui/skeleton';
import { useToast } from '@/hooks/use-toast';
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from 'recharts';
import {
  TrendingUp,
  TrendingDown,
  AlertTriangle,
  AlertCircle,
  Info,
  Clock,
  Bell,
  BellOff,
  CheckCircle2,
  Loader2,
  RefreshCw,
  BarChart3,
  Activity,
  Shield,
} from 'lucide-react';
import { format } from 'date-fns';

interface RiskHistoryPoint {
  date: string;
  risk_score: number;
  avg_pain: number;
  avg_fatigue: number;
  avg_mood: number;
  mh_score: number;
  med_adherence_7d: number;
  env_risk_score: number;
  engagement_rate_14d: number;
}

interface RiskHistoryData {
  patient_id: string;
  current_state: {
    risk_score: number;
    risk_state: string;
    risk_components: Record<string, number>;
    last_updated: string | null;
  } | null;
  history: RiskHistoryPoint[];
  period_days: number;
  wellness_disclaimer: string;
}

interface TriggerEvent {
  id: string;
  name: string;
  severity: string;
  context: Record<string, any>;
  created_at: string;
}

interface TriggerData {
  patient_id: string;
  triggers: TriggerEvent[];
  count: number;
  severity_counts: Record<string, number>;
  period_days: number;
  wellness_disclaimer: string;
}

interface Notification {
  id: string;
  title: string;
  body: string;
  channel: string;
  priority: string;
  status: string;
  created_at: string;
  is_read: boolean;
}

interface NotificationData {
  notifications: Notification[];
  count: number;
}

const HISTORY_RANGES = [
  { value: '7', label: '7 Days', days: 7 },
  { value: '30', label: '30 Days', days: 30 },
  { value: '90', label: '90 Days', days: 90 },
];

function getSeverityBadge(severity: string) {
  switch (severity?.toLowerCase()) {
    case 'alert':
      return <Badge variant="destructive" className="text-xs"><AlertTriangle className="h-3 w-3 mr-1" />Alert</Badge>;
    case 'warning':
      return <Badge className="bg-yellow-500/20 text-yellow-600 dark:text-yellow-400 text-xs"><AlertCircle className="h-3 w-3 mr-1" />Warning</Badge>;
    default:
      return <Badge variant="secondary" className="text-xs"><Info className="h-3 w-3 mr-1" />Info</Badge>;
  }
}

function formatTriggerName(name: string): string {
  return name
    .replace(/_/g, ' ')
    .replace(/\b\w/g, c => c.toUpperCase());
}

export function RiskHistoryChart({ patientId }: { patientId: string }) {
  const [historyDays, setHistoryDays] = useState('30');
  
  const { data, isLoading, isError, refetch } = useQuery<RiskHistoryData>({
    queryKey: ['/api/v1/followup-autopilot/patients', patientId, 'history', { days: historyDays }],
    queryFn: async () => {
      const res = await fetch(`/api/v1/followup-autopilot/patients/${encodeURIComponent(patientId)}/history?days=${historyDays}`);
      if (!res.ok) throw new Error('Failed to fetch history');
      return res.json();
    },
    enabled: !!patientId,
    staleTime: 60000,
  });

  if (isLoading) {
    return (
      <Card>
        <CardHeader>
          <Skeleton className="h-6 w-48" />
          <Skeleton className="h-4 w-64 mt-2" />
        </CardHeader>
        <CardContent>
          <Skeleton className="h-[300px] w-full" />
        </CardContent>
      </Card>
    );
  }

  if (isError) {
    return (
      <Card className="border-destructive/50">
        <CardContent className="p-6 text-center">
          <AlertTriangle className="h-8 w-8 text-destructive mx-auto mb-2" />
          <p className="text-sm text-muted-foreground">Unable to load risk history</p>
          <Button variant="outline" size="sm" onClick={() => refetch()} className="mt-3">
            <RefreshCw className="h-4 w-4 mr-2" />Retry
          </Button>
        </CardContent>
      </Card>
    );
  }

  const chartData = (data?.history || []).map(h => ({
    ...h,
    date: format(new Date(h.date), 'MMM d'),
    risk_score: h.risk_score || 0,
  })).reverse();

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between flex-wrap gap-2">
          <div>
            <CardTitle className="flex items-center gap-2">
              <BarChart3 className="h-5 w-5" />
              Risk Score History
            </CardTitle>
            <CardDescription>
              Track your wellness risk score trends over time
            </CardDescription>
          </div>
          <Select value={historyDays} onValueChange={setHistoryDays}>
            <SelectTrigger className="w-[120px]" data-testid="select-history-range">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {HISTORY_RANGES.map(r => (
                <SelectItem key={r.value} value={r.value}>{r.label}</SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </CardHeader>
      <CardContent>
        {chartData.length > 0 ? (
          <div className="h-[300px]" data-testid="chart-risk-history">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={chartData}>
                <defs>
                  <linearGradient id="riskGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="hsl(var(--chart-1))" stopOpacity={0.3}/>
                    <stop offset="95%" stopColor="hsl(var(--chart-1))" stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                <XAxis dataKey="date" className="text-xs" />
                <YAxis domain={[0, 15]} className="text-xs" />
                <Tooltip 
                  contentStyle={{ 
                    backgroundColor: 'hsl(var(--card))', 
                    border: '1px solid hsl(var(--border))',
                    borderRadius: '8px'
                  }}
                  formatter={(value: number, name: string) => {
                    if (name === 'Risk Score') return [`${value.toFixed(1)}/15`, name];
                    return [value, name];
                  }}
                />
                <Legend />
                <Area 
                  type="monotone" 
                  dataKey="risk_score" 
                  name="Risk Score" 
                  stroke="hsl(var(--chart-1))" 
                  fill="url(#riskGradient)"
                  strokeWidth={2}
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        ) : (
          <div className="h-[200px] flex items-center justify-center text-muted-foreground">
            <div className="text-center">
              <Activity className="h-8 w-8 mx-auto mb-2 opacity-50" />
              <p>No history data available yet</p>
              <p className="text-xs mt-1">Complete daily check-ins to build your history</p>
            </div>
          </div>
        )}
        
        {data?.current_state && (
          <div className="mt-4 p-3 rounded-lg bg-muted/50 border">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
              <div>
                <span className="text-muted-foreground">Current Score:</span>
                <span className="ml-2 font-bold">{data.current_state.risk_score?.toFixed(1) || 'N/A'}/15</span>
              </div>
              <div>
                <span className="text-muted-foreground">State:</span>
                <span className="ml-2 font-medium">{data.current_state.risk_state || 'Unknown'}</span>
              </div>
              <div>
                <span className="text-muted-foreground">Data Points:</span>
                <span className="ml-2">{chartData.length}</span>
              </div>
              <div>
                <span className="text-muted-foreground">Period:</span>
                <span className="ml-2">{data.period_days} days</span>
              </div>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

export function TriggerEventsLog({ patientId }: { patientId: string }) {
  const [triggerDays, setTriggerDays] = useState('14');
  const [severityFilter, setSeverityFilter] = useState<string>('all');
  
  const { data, isLoading, isError, refetch } = useQuery<TriggerData>({
    queryKey: ['/api/v1/followup-autopilot/patients', patientId, 'triggers', { days: triggerDays, severity: severityFilter }],
    queryFn: async () => {
      const params = new URLSearchParams({ days: triggerDays });
      if (severityFilter !== 'all') params.append('severity', severityFilter);
      const res = await fetch(`/api/v1/followup-autopilot/patients/${encodeURIComponent(patientId)}/triggers?${params}`);
      if (!res.ok) throw new Error('Failed to fetch triggers');
      return res.json();
    },
    enabled: !!patientId,
    staleTime: 30000,
  });

  if (isLoading) {
    return (
      <Card>
        <CardHeader>
          <Skeleton className="h-6 w-48" />
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {[1,2,3].map(i => <Skeleton key={i} className="h-16 w-full" />)}
          </div>
        </CardContent>
      </Card>
    );
  }

  if (isError) {
    return (
      <Card className="border-destructive/50">
        <CardContent className="p-6 text-center">
          <AlertTriangle className="h-8 w-8 text-destructive mx-auto mb-2" />
          <p className="text-sm text-muted-foreground">Unable to load trigger events</p>
          <Button variant="outline" size="sm" onClick={() => refetch()} className="mt-3">
            <RefreshCw className="h-4 w-4 mr-2" />Retry
          </Button>
        </CardContent>
      </Card>
    );
  }

  const triggers = data?.triggers || [];
  const severityCounts = data?.severity_counts || {};

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between flex-wrap gap-2">
          <div>
            <CardTitle className="flex items-center gap-2">
              <AlertCircle className="h-5 w-5" />
              Trigger Events Log
            </CardTitle>
            <CardDescription>
              Events that triggered wellness follow-ups
            </CardDescription>
          </div>
          <div className="flex items-center gap-2">
            <Select value={severityFilter} onValueChange={setSeverityFilter}>
              <SelectTrigger className="w-[100px]" data-testid="select-severity-filter">
                <SelectValue placeholder="All" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All</SelectItem>
                <SelectItem value="info">Info</SelectItem>
                <SelectItem value="warning">Warning</SelectItem>
                <SelectItem value="alert">Alert</SelectItem>
              </SelectContent>
            </Select>
            <Select value={triggerDays} onValueChange={setTriggerDays}>
              <SelectTrigger className="w-[100px]" data-testid="select-trigger-days">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {HISTORY_RANGES.map(r => (
                  <SelectItem key={r.value} value={r.value}>{r.label}</SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </div>
        {Object.keys(severityCounts).length > 0 && (
          <div className="flex items-center gap-2 mt-2">
            {Object.entries(severityCounts).map(([sev, count]) => (
              <Badge key={sev} variant="outline" className="text-xs">
                {sev}: {count}
              </Badge>
            ))}
          </div>
        )}
      </CardHeader>
      <CardContent>
        {triggers.length > 0 ? (
          <ScrollArea className="h-[300px]" data-testid="list-trigger-events">
            <div className="space-y-2">
              {triggers.map((trigger) => (
                <div 
                  key={trigger.id} 
                  className="p-3 rounded-lg border bg-card hover-elevate"
                  data-testid={`trigger-event-${trigger.id}`}
                >
                  <div className="flex items-start justify-between gap-2">
                    <div className="flex-1">
                      <div className="flex items-center gap-2 mb-1">
                        {getSeverityBadge(trigger.severity)}
                        <span className="font-medium text-sm">
                          {formatTriggerName(trigger.name)}
                        </span>
                      </div>
                      <div className="flex items-center gap-2 text-xs text-muted-foreground">
                        <Clock className="h-3 w-3" />
                        {trigger.created_at 
                          ? format(new Date(trigger.created_at), 'MMM d, yyyy h:mm a')
                          : 'Unknown time'}
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </ScrollArea>
        ) : (
          <div className="h-[200px] flex items-center justify-center text-muted-foreground">
            <div className="text-center">
              <Shield className="h-8 w-8 mx-auto mb-2 opacity-50" />
              <p>No trigger events in this period</p>
              <p className="text-xs mt-1">This means your wellness indicators are stable</p>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

export function NotificationsBell({ patientId }: { patientId: string }) {
  const [isOpen, setIsOpen] = useState(false);
  const { toast } = useToast();
  
  const { data, isLoading, refetch } = useQuery<NotificationData>({
    queryKey: ['/api/v1/followup-autopilot/patients', patientId, 'notifications'],
    queryFn: async () => {
      const res = await fetch(`/api/v1/followup-autopilot/patients/${encodeURIComponent(patientId)}/notifications`);
      if (!res.ok) throw new Error('Failed to fetch notifications');
      return res.json();
    },
    enabled: !!patientId,
    staleTime: 30000,
    refetchInterval: 60000,
  });

  const markReadMutation = useMutation({
    mutationFn: async (notificationId: string) => {
      return apiRequest('POST', `/api/v1/followup-autopilot/patients/${patientId}/notifications/${notificationId}/read`, {});
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ 
        queryKey: ['/api/v1/followup-autopilot/patients', patientId, 'notifications'] 
      });
    },
    onError: () => {
      toast({
        title: "Error",
        description: "Failed to mark notification as read",
        variant: "destructive",
      });
    },
  });

  const notifications = data?.notifications || [];
  const unreadCount = notifications.filter(n => !n.is_read && n.status === 'pending').length;

  return (
    <div className="relative">
      <Button 
        variant="ghost" 
        size="icon" 
        onClick={() => setIsOpen(!isOpen)}
        className="relative"
        data-testid="button-notifications-bell"
      >
        {isLoading ? (
          <Loader2 className="h-5 w-5 animate-spin" />
        ) : unreadCount > 0 ? (
          <Bell className="h-5 w-5" />
        ) : (
          <BellOff className="h-5 w-5 text-muted-foreground" />
        )}
        {unreadCount > 0 && (
          <span className="absolute -top-1 -right-1 h-5 w-5 rounded-full bg-destructive text-destructive-foreground text-xs flex items-center justify-center animate-pulse">
            {unreadCount > 9 ? '9+' : unreadCount}
          </span>
        )}
      </Button>

      {isOpen && (
        <>
          <div 
            className="fixed inset-0 z-40" 
            onClick={() => setIsOpen(false)} 
          />
          <Card className="absolute right-0 top-12 z-50 w-80 shadow-lg" data-testid="dropdown-notifications">
            <CardHeader className="pb-2">
              <div className="flex items-center justify-between">
                <CardTitle className="text-base">Notifications</CardTitle>
                <Button 
                  variant="ghost" 
                  size="sm" 
                  onClick={() => refetch()}
                  data-testid="button-refresh-notifications"
                >
                  <RefreshCw className="h-4 w-4" />
                </Button>
              </div>
            </CardHeader>
            <CardContent className="p-0">
              <ScrollArea className="h-[300px]">
                {notifications.length > 0 ? (
                  <div className="divide-y">
                    {notifications.map((notif) => (
                      <div 
                        key={notif.id}
                        className={`p-3 hover:bg-muted/50 cursor-pointer ${
                          notif.is_read ? 'opacity-60' : ''
                        }`}
                        onClick={() => {
                          if (!notif.is_read) {
                            markReadMutation.mutate(notif.id);
                          }
                        }}
                        data-testid={`notification-item-${notif.id}`}
                      >
                        <div className="flex items-start gap-2">
                          <div className={`mt-1 h-2 w-2 rounded-full flex-shrink-0 ${
                            notif.is_read ? 'bg-muted' : 'bg-primary'
                          }`} />
                          <div className="flex-1 min-w-0">
                            <p className="text-sm font-medium truncate">{notif.title}</p>
                            <p className="text-xs text-muted-foreground line-clamp-2">{notif.body}</p>
                            <p className="text-xs text-muted-foreground mt-1">
                              {notif.created_at 
                                ? format(new Date(notif.created_at), 'MMM d, h:mm a')
                                : 'Unknown'}
                            </p>
                          </div>
                          {!notif.is_read && (
                            <CheckCircle2 className="h-4 w-4 text-muted-foreground hover:text-primary flex-shrink-0" />
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="p-6 text-center text-muted-foreground">
                    <Bell className="h-8 w-8 mx-auto mb-2 opacity-50" />
                    <p className="text-sm">No notifications</p>
                  </div>
                )}
              </ScrollArea>
            </CardContent>
          </Card>
        </>
      )}
    </div>
  );
}

export function AutopilotInsightsTab({ patientId }: { patientId: string }) {
  return (
    <div className="space-y-4">
      <RiskHistoryChart patientId={patientId} />
      <TriggerEventsLog patientId={patientId} />
    </div>
  );
}
