import { useState } from 'react';
import { useQuery, useMutation } from '@tanstack/react-query';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Separator } from '@/components/ui/separator';
import { apiRequest, queryClient } from '@/lib/queryClient';
import { useToast } from '@/hooks/use-toast';
import { 
  Bell, AlertTriangle, AlertCircle, Info, CheckCircle, Clock,
  Filter, Search, Eye, Check, X, TrendingUp, TrendingDown,
  Activity, User, Beaker, ChevronRight, BarChart3
} from 'lucide-react';
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, 
  ResponsiveContainer, ReferenceLine 
} from 'recharts';

interface ResearchAlert {
  id: string;
  type: 'deterioration' | 'enrollment' | 'compliance' | 'data_quality' | 'analysis' | 'threshold';
  severity: 'critical' | 'high' | 'medium' | 'low';
  status: 'active' | 'acknowledged' | 'resolved';
  title: string;
  message: string;
  patientId?: string;
  studyId?: string;
  metadata?: {
    riskScore?: number;
    threshold?: number;
    previousValue?: number;
    currentValue?: number;
    topFeatures?: { name: string; importance: number }[];
    trend?: { date: string; value: number }[];
  };
  createdAt: string;
  acknowledgedAt?: string;
  acknowledgedBy?: string;
  resolvedAt?: string;
  resolvedBy?: string;
}

const SEVERITY_CONFIG = {
  critical: { 
    icon: AlertCircle, 
    color: 'text-red-500', 
    bg: 'bg-red-500/10', 
    badge: 'destructive' as const,
    label: 'Critical'
  },
  high: { 
    icon: AlertTriangle, 
    color: 'text-amber-500', 
    bg: 'bg-amber-500/10', 
    badge: 'default' as const,
    label: 'High'
  },
  medium: { 
    icon: Info, 
    color: 'text-blue-500', 
    bg: 'bg-blue-500/10', 
    badge: 'secondary' as const,
    label: 'Medium'
  },
  low: { 
    icon: Bell, 
    color: 'text-gray-500', 
    bg: 'bg-gray-500/10', 
    badge: 'outline' as const,
    label: 'Low'
  },
};

const TYPE_CONFIG = {
  deterioration: { label: 'Deterioration', icon: TrendingDown, color: 'text-red-500' },
  enrollment: { label: 'Enrollment', icon: User, color: 'text-blue-500' },
  compliance: { label: 'Compliance', icon: CheckCircle, color: 'text-amber-500' },
  data_quality: { label: 'Data Quality', icon: BarChart3, color: 'text-purple-500' },
  analysis: { label: 'Analysis', icon: Beaker, color: 'text-emerald-500' },
  threshold: { label: 'Threshold', icon: Activity, color: 'text-orange-500' },
};

const mockAlerts: ResearchAlert[] = [
  {
    id: '1',
    type: 'deterioration',
    severity: 'critical',
    status: 'active',
    title: 'High Deterioration Risk Detected',
    message: 'Patient shows significantly elevated deterioration index with risk score above threshold.',
    patientId: 'P001',
    metadata: {
      riskScore: 12.5,
      threshold: 10,
      previousValue: 8.2,
      currentValue: 12.5,
      topFeatures: [
        { name: 'Heart rate variability', importance: 0.35 },
        { name: 'Medication adherence', importance: 0.25 },
        { name: 'Sleep quality', importance: 0.20 },
      ],
      trend: [
        { date: 'Day 1', value: 6.5 },
        { date: 'Day 2', value: 7.2 },
        { date: 'Day 3', value: 8.0 },
        { date: 'Day 4', value: 8.2 },
        { date: 'Day 5', value: 10.1 },
        { date: 'Day 6', value: 11.8 },
        { date: 'Day 7', value: 12.5 },
      ],
    },
    createdAt: new Date(Date.now() - 2 * 60 * 60 * 1000).toISOString(),
  },
  {
    id: '2',
    type: 'enrollment',
    severity: 'medium',
    status: 'active',
    title: 'Enrollment Below Target',
    message: 'Study enrollment is 15% below projected milestone for this period.',
    studyId: 'S001',
    metadata: {
      currentValue: 45,
      threshold: 53,
    },
    createdAt: new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString(),
  },
  {
    id: '3',
    type: 'compliance',
    severity: 'high',
    status: 'acknowledged',
    title: 'Low Follow-up Compliance',
    message: '3 patients have missed their scheduled follow-up visits in the past week.',
    studyId: 'S002',
    acknowledgedAt: new Date(Date.now() - 12 * 60 * 60 * 1000).toISOString(),
    acknowledgedBy: 'Dr. Smith',
    createdAt: new Date(Date.now() - 48 * 60 * 60 * 1000).toISOString(),
  },
  {
    id: '4',
    type: 'data_quality',
    severity: 'low',
    status: 'resolved',
    title: 'Missing Lab Values',
    message: 'Some lab result records are incomplete for 5 patients.',
    resolvedAt: new Date(Date.now() - 6 * 60 * 60 * 1000).toISOString(),
    resolvedBy: 'System',
    createdAt: new Date(Date.now() - 72 * 60 * 60 * 1000).toISOString(),
  },
];

export function AlertsTab() {
  const { toast } = useToast();
  const [selectedAlert, setSelectedAlert] = useState<ResearchAlert | null>(null);
  const [detailsOpen, setDetailsOpen] = useState(false);
  const [typeFilter, setTypeFilter] = useState<string>('all');
  const [severityFilter, setSeverityFilter] = useState<string>('all');
  const [statusFilter, setStatusFilter] = useState<string>('active');
  const [searchQuery, setSearchQuery] = useState('');

  const { data: alertsData, isLoading } = useQuery<ResearchAlert[]>({
    queryKey: ['/api/v1/research-center/alerts', { type: typeFilter, severity: severityFilter, status: statusFilter }],
  });
  
  const alerts = alertsData && alertsData.length > 0 ? alertsData : mockAlerts;

  const acknowledgeMutation = useMutation({
    mutationFn: async (alertId: string) => {
      const response = await apiRequest('POST', `/api/v1/research-center/alerts/${alertId}/acknowledge`, {});
      return response.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['/api/v1/research-center/alerts'] });
      toast({ title: 'Alert Acknowledged', description: 'The alert has been marked as acknowledged.' });
    },
    onError: (error: Error) => {
      toast({ title: 'Error', description: error.message, variant: 'destructive' });
    },
  });

  const resolveMutation = useMutation({
    mutationFn: async (alertId: string) => {
      const response = await apiRequest('POST', `/api/v1/research-center/alerts/${alertId}/resolve`, {});
      return response.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['/api/v1/research-center/alerts'] });
      setDetailsOpen(false);
      toast({ title: 'Alert Resolved', description: 'The alert has been marked as resolved.' });
    },
    onError: (error: Error) => {
      toast({ title: 'Error', description: error.message, variant: 'destructive' });
    },
  });

  const filteredAlerts = alerts.filter(alert => {
    if (typeFilter !== 'all' && alert.type !== typeFilter) return false;
    if (severityFilter !== 'all' && alert.severity !== severityFilter) return false;
    if (statusFilter !== 'all' && alert.status !== statusFilter) return false;
    if (searchQuery) {
      const query = searchQuery.toLowerCase();
      return alert.title.toLowerCase().includes(query) || 
             alert.message.toLowerCase().includes(query);
    }
    return true;
  });

  const alertCounts = {
    critical: alerts.filter(a => a.severity === 'critical' && a.status === 'active').length,
    high: alerts.filter(a => a.severity === 'high' && a.status === 'active').length,
    medium: alerts.filter(a => a.severity === 'medium' && a.status === 'active').length,
    low: alerts.filter(a => a.severity === 'low' && a.status === 'active').length,
  };

  const renderSHAPExplanation = (features?: { name: string; importance: number }[]) => {
    if (!features || features.length === 0) return null;
    
    return (
      <div className="space-y-2">
        <Label className="text-sm">Top Contributing Factors (SHAP)</Label>
        <div className="space-y-2">
          {features.map((feature, idx) => (
            <div key={idx} className="flex items-center gap-3">
              <div className="flex-1">
                <div className="flex items-center justify-between text-sm">
                  <span>{feature.name}</span>
                  <span className="text-muted-foreground">{(feature.importance * 100).toFixed(0)}%</span>
                </div>
                <div className="h-2 bg-muted rounded-full mt-1 overflow-hidden">
                  <div 
                    className="h-full bg-primary rounded-full transition-all"
                    style={{ width: `${feature.importance * 100}%` }}
                  />
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  };

  const renderTrendChart = (trend?: { date: string; value: number }[], threshold?: number) => {
    if (!trend || trend.length === 0) return null;
    
    return (
      <div className="space-y-2">
        <Label className="text-sm">Risk Score Trend</Label>
        <ResponsiveContainer width="100%" height={150}>
          <LineChart data={trend}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="date" fontSize={10} />
            <YAxis fontSize={10} domain={[0, 'auto']} />
            <Tooltip />
            <Line 
              type="monotone" 
              dataKey="value" 
              stroke="#8b5cf6" 
              strokeWidth={2}
              dot={{ r: 3 }}
            />
            {threshold && (
              <ReferenceLine 
                y={threshold} 
                stroke="#ef4444" 
                strokeDasharray="5 5"
                label={{ value: 'Threshold', position: 'right', fontSize: 10 }}
              />
            )}
          </LineChart>
        </ResponsiveContainer>
      </div>
    );
  };

  return (
    <div className="space-y-4" data-testid="alerts-tab">
      <div className="grid gap-4 md:grid-cols-4">
        {Object.entries(SEVERITY_CONFIG).map(([key, config]) => {
          const Icon = config.icon;
          const count = alertCounts[key as keyof typeof alertCounts];
          return (
            <Card 
              key={key} 
              className={`hover-elevate cursor-pointer ${severityFilter === key ? 'ring-2 ring-primary' : ''}`}
              onClick={() => setSeverityFilter(severityFilter === key ? 'all' : key)}
              data-testid={`card-severity-${key}`}
            >
              <CardContent className="pt-4">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-muted-foreground">{config.label}</p>
                    <p className="text-2xl font-bold">{count}</p>
                  </div>
                  <div className={`flex h-10 w-10 items-center justify-center rounded-full ${config.bg}`}>
                    <Icon className={`h-5 w-5 ${config.color}`} />
                  </div>
                </div>
              </CardContent>
            </Card>
          );
        })}
      </div>

      <Card>
        <CardHeader className="pb-2">
          <div className="flex items-center justify-between gap-4 flex-wrap">
            <CardTitle className="flex items-center gap-2 text-base">
              <Bell className="h-4 w-4" />
              Research Alerts
            </CardTitle>
            <div className="flex items-center gap-2 flex-wrap">
              <div className="relative">
                <Search className="absolute left-2.5 top-2.5 h-4 w-4 text-muted-foreground" />
                <Input
                  placeholder="Search alerts..."
                  className="pl-8 w-48"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  data-testid="input-search-alerts"
                />
              </div>
              <Select value={typeFilter} onValueChange={setTypeFilter}>
                <SelectTrigger className="w-36" data-testid="select-type-filter">
                  <SelectValue placeholder="All Types" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Types</SelectItem>
                  {Object.entries(TYPE_CONFIG).map(([key, config]) => (
                    <SelectItem key={key} value={key}>{config.label}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <Select value={statusFilter} onValueChange={setStatusFilter}>
                <SelectTrigger className="w-36" data-testid="select-status-filter">
                  <SelectValue placeholder="Active" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All Status</SelectItem>
                  <SelectItem value="active">Active</SelectItem>
                  <SelectItem value="acknowledged">Acknowledged</SelectItem>
                  <SelectItem value="resolved">Resolved</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          {isLoading ? (
            <div className="space-y-3">
              {[1, 2, 3].map((i) => (
                <div key={i} className="h-20 bg-muted animate-pulse rounded-lg" />
              ))}
            </div>
          ) : filteredAlerts.length > 0 ? (
            <ScrollArea className="h-[400px]">
              <div className="space-y-3">
                {filteredAlerts.map((alert) => {
                  const severityConfig = SEVERITY_CONFIG[alert.severity];
                  const typeConfig = TYPE_CONFIG[alert.type];
                  const SeverityIcon = severityConfig.icon;
                  const TypeIcon = typeConfig.icon;
                  
                  return (
                    <div 
                      key={alert.id}
                      className={`flex items-start gap-4 p-4 border rounded-lg hover-elevate ${severityConfig.bg}`}
                      data-testid={`alert-${alert.id}`}
                    >
                      <div className={`flex h-10 w-10 items-center justify-center rounded-full bg-background`}>
                        <SeverityIcon className={`h-5 w-5 ${severityConfig.color}`} />
                      </div>
                      <div className="flex-1 min-w-0">
                        <div className="flex items-start justify-between gap-2">
                          <div>
                            <div className="flex items-center gap-2">
                              <h4 className="font-medium">{alert.title}</h4>
                              <Badge variant={severityConfig.badge}>{severityConfig.label}</Badge>
                              <Badge variant="outline" className="text-xs">
                                <TypeIcon className="h-3 w-3 mr-1" />
                                {typeConfig.label}
                              </Badge>
                            </div>
                            <p className="text-sm text-muted-foreground mt-1 line-clamp-2">
                              {alert.message}
                            </p>
                          </div>
                          <div className="flex items-center gap-2 flex-shrink-0">
                            {alert.status === 'active' && (
                              <>
                                <Button 
                                  variant="outline" 
                                  size="sm"
                                  onClick={() => acknowledgeMutation.mutate(alert.id)}
                                  data-testid={`button-acknowledge-${alert.id}`}
                                >
                                  <Check className="h-3 w-3 mr-1" />
                                  Acknowledge
                                </Button>
                              </>
                            )}
                            <Button 
                              variant="ghost" 
                              size="icon"
                              onClick={() => {
                                setSelectedAlert(alert);
                                setDetailsOpen(true);
                              }}
                              data-testid={`button-view-alert-${alert.id}`}
                            >
                              <Eye className="h-4 w-4" />
                            </Button>
                          </div>
                        </div>
                        <div className="flex items-center gap-4 mt-2 text-xs text-muted-foreground">
                          <span className="flex items-center gap-1">
                            <Clock className="h-3 w-3" />
                            {new Date(alert.createdAt).toLocaleString()}
                          </span>
                          {alert.patientId && (
                            <span className="flex items-center gap-1">
                              <User className="h-3 w-3" />
                              Patient: {alert.patientId}
                            </span>
                          )}
                          {alert.studyId && (
                            <span className="flex items-center gap-1">
                              <Beaker className="h-3 w-3" />
                              Study: {alert.studyId}
                            </span>
                          )}
                          {alert.status !== 'active' && (
                            <Badge variant="secondary" className="text-xs">
                              {alert.status === 'acknowledged' ? 'Acknowledged' : 'Resolved'}
                            </Badge>
                          )}
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </ScrollArea>
          ) : (
            <div className="text-center py-12">
              <CheckCircle className="h-12 w-12 mx-auto text-emerald-500 opacity-50 mb-4" />
              <h3 className="text-sm font-medium mb-2">No Alerts</h3>
              <p className="text-xs text-muted-foreground">
                No alerts match your current filter criteria
              </p>
            </div>
          )}
        </CardContent>
      </Card>

      <Dialog open={detailsOpen} onOpenChange={setDetailsOpen}>
        <DialogContent className="max-w-2xl">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              {selectedAlert && (
                <>
                  {(() => {
                    const Icon = SEVERITY_CONFIG[selectedAlert.severity].icon;
                    return <Icon className={`h-5 w-5 ${SEVERITY_CONFIG[selectedAlert.severity].color}`} />;
                  })()}
                  {selectedAlert.title}
                </>
              )}
            </DialogTitle>
            <DialogDescription>
              Alert details, trend analysis, and recommended actions
            </DialogDescription>
          </DialogHeader>

          {selectedAlert && (
            <div className="space-y-4">
              <div className="flex items-center gap-2 flex-wrap">
                <Badge variant={SEVERITY_CONFIG[selectedAlert.severity].badge}>
                  {SEVERITY_CONFIG[selectedAlert.severity].label}
                </Badge>
                <Badge variant="outline">
                  {TYPE_CONFIG[selectedAlert.type].label}
                </Badge>
                <Badge variant={selectedAlert.status === 'active' ? 'default' : 'secondary'}>
                  {selectedAlert.status.charAt(0).toUpperCase() + selectedAlert.status.slice(1)}
                </Badge>
              </div>

              <p className="text-sm">{selectedAlert.message}</p>

              <Separator />

              {selectedAlert.metadata && (
                <div className="space-y-4">
                  {selectedAlert.metadata.currentValue !== undefined && (
                    <div className="grid gap-4 md:grid-cols-3">
                      <Card>
                        <CardContent className="pt-4">
                          <p className="text-xs text-muted-foreground">Current Value</p>
                          <p className="text-xl font-bold">{selectedAlert.metadata.currentValue}</p>
                        </CardContent>
                      </Card>
                      {selectedAlert.metadata.threshold !== undefined && (
                        <Card>
                          <CardContent className="pt-4">
                            <p className="text-xs text-muted-foreground">Threshold</p>
                            <p className="text-xl font-bold text-amber-500">{selectedAlert.metadata.threshold}</p>
                          </CardContent>
                        </Card>
                      )}
                      {selectedAlert.metadata.previousValue !== undefined && (
                        <Card>
                          <CardContent className="pt-4">
                            <p className="text-xs text-muted-foreground">Previous Value</p>
                            <p className="text-xl font-bold text-muted-foreground">
                              {selectedAlert.metadata.previousValue}
                            </p>
                          </CardContent>
                        </Card>
                      )}
                    </div>
                  )}

                  {renderTrendChart(selectedAlert.metadata.trend, selectedAlert.metadata.threshold)}
                  {renderSHAPExplanation(selectedAlert.metadata.topFeatures)}
                </div>
              )}

              <div className="text-xs text-muted-foreground space-y-1">
                <p>Created: {new Date(selectedAlert.createdAt).toLocaleString()}</p>
                {selectedAlert.acknowledgedAt && (
                  <p>Acknowledged: {new Date(selectedAlert.acknowledgedAt).toLocaleString()} by {selectedAlert.acknowledgedBy}</p>
                )}
                {selectedAlert.resolvedAt && (
                  <p>Resolved: {new Date(selectedAlert.resolvedAt).toLocaleString()} by {selectedAlert.resolvedBy}</p>
                )}
              </div>
            </div>
          )}

          <DialogFooter>
            {selectedAlert?.status === 'active' && (
              <>
                <Button 
                  variant="outline"
                  onClick={() => acknowledgeMutation.mutate(selectedAlert.id)}
                  disabled={acknowledgeMutation.isPending}
                >
                  <Check className="h-4 w-4 mr-2" />
                  Acknowledge
                </Button>
                <Button 
                  onClick={() => resolveMutation.mutate(selectedAlert.id)}
                  disabled={resolveMutation.isPending}
                >
                  <CheckCircle className="h-4 w-4 mr-2" />
                  Resolve Alert
                </Button>
              </>
            )}
            {selectedAlert?.status === 'acknowledged' && (
              <Button 
                onClick={() => resolveMutation.mutate(selectedAlert.id)}
                disabled={resolveMutation.isPending}
              >
                <CheckCircle className="h-4 w-4 mr-2" />
                Resolve Alert
              </Button>
            )}
            <Button variant="outline" onClick={() => setDetailsOpen(false)}>
              Close
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
