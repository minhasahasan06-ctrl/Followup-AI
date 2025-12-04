import { useState, useRef } from 'react';
import * as React from 'react';
import { useQuery, useMutation } from '@tanstack/react-query';
import { queryClient } from '@/lib/queryClient';
import { useToast } from '@/hooks/use-toast';
import { Link } from 'wouter';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Label } from '@/components/ui/label';
import { Skeleton } from '@/components/ui/skeleton';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import {
  Video,
  Camera,
  Play,
  CheckCircle2,
  AlertCircle,
  Info,
  BarChart3,
  Loader2,
  Wind,
  Hand,
  Eye,
  User,
  Activity,
  MessageSquare,
  XCircle,
  Palette,
  Users,
  Zap,
  Smile,
  ExternalLink,
  Vibrate,
  Footprints,
  Droplets,
  TrendingUp,
  TrendingDown,
  Minus,
  Watch,
  Heart,
  Thermometer,
  Gauge,
  Moon,
  Flame,
  RefreshCw,
  Calculator,
  Clock,
  Mic,
  Square,
  Upload,
  FileText,
  Stethoscope,
  Pill,
} from 'lucide-react';
import { ExamPrepStep } from '@/components/ExamPrepStep';
import { VideoRecorder } from '@/components/VideoRecorder';
import { useGuidedExamWorkflow } from '@/hooks/useGuidedExamWorkflow';
import { useAuth } from '@/contexts/AuthContext';

interface TremorDashboard {
  patient_id: string;
  latest_tremor: {
    tremor_index?: number;
    dominant_frequency_hz?: number;
    parkinsonian_likelihood?: number;
    essential_tremor_likelihood?: number;
    created_at?: string;
  } | null;
  trend: {
    status: 'stable' | 'increasing' | 'decreasing' | 'insufficient_data';
    avg_tremor_index_7days: number;
    recordings_count_7days: number;
  };
}

interface GaitSession {
  session_id: number;
  status: string;
  total_strides: number;
  abnormality_score: number;
  created_at: string;
}

interface EdemaMetrics {
  swelling_detected: boolean;
  swelling_severity: string;
  overall_expansion_percent: number;
  analyzed_at: string;
}

interface DeviceConnection {
  id: string;
  vendor_id: string;
  device_type: string;
  device_name: string;
  status: string;
  last_sync_at: string | null;
  data_types: string[];
}

interface DeviceReading {
  data_type: string;
  value: number;
  unit: string;
  timestamp: string;
  source_device?: string;
}

interface HealthSectionAnalytics {
  section: string;
  risk_score: number;
  trend: 'stable' | 'improving' | 'declining' | 'unknown';
  deterioration_index: number;
  stability_score: number;
  last_updated: string;
  readings_count: number;
}

interface DeviationRecord {
  id: number;
  patient_id: string;
  baseline_id: number;
  metric_name: string;
  measurement_value: number;
  measurement_date: string;
  z_score: number;
  percent_change: number;
  baseline_mean: number;
  baseline_std: number;
  trend_3day_slope: number | null;
  trend_7day_slope: number | null;
  trend_direction: string | null;
  deviation_type: string;
  severity_level: string;
  alert_triggered: boolean;
  alert_message: string | null;
  created_at: string;
}

interface DeviationSummary {
  metric_name: string;
  total_deviations: number;
  critical_count: number;
  moderate_count: number;
  recent_z_score: number | null;
  trend_direction: string | null;
}

interface BaselineStats {
  mean: number | null;
  std: number | null;
  min: number | null;
  max: number | null;
  n_samples: number;
}

interface BaselineInfo {
  id: number;
  patient_id: string;
  baseline_start_date: string;
  baseline_end_date: string;
  data_points_count: number;
  pain_facial: BaselineStats;
  pain_self_reported: BaselineStats;
  respiratory_rate: BaselineStats;
  symptom_severity: BaselineStats;
  baseline_quality: string | null;
  is_current: boolean;
  created_at: string;
}

interface DeviceHealthData {
  connections: DeviceConnection[];
  latest_readings: DeviceReading[];
  health_analytics: HealthSectionAnalytics[];
  last_sync: string | null;
}

function DeviceDataTab({ patientId }: { patientId: string }) {
  const { data: deviceData, isLoading, error, refetch } = useQuery<DeviceHealthData>({
    queryKey: ['/api/v1/devices/daily-summary', patientId],
    queryFn: async () => {
      const [connectionsRes, analyticsRes] = await Promise.all([
        fetch('/api/v1/devices/connections'),
        fetch('/api/v1/devices/health-analytics'),
      ]);
      
      if (connectionsRes.status === 401 || analyticsRes.status === 401) {
        return {
          connections: [],
          latest_readings: [],
          health_analytics: [],
          last_sync: null,
        };
      }
      
      const connections = connectionsRes.ok ? await connectionsRes.json() : { devices: [] };
      const analytics = analyticsRes.ok ? await analyticsRes.json() : { sections: [] };
      
      return {
        connections: connections.devices || [],
        latest_readings: [],
        health_analytics: analytics.sections || [],
        last_sync: connections.devices?.[0]?.last_sync_at || null,
      };
    },
    enabled: !!patientId,
    retry: false,
  });

  const formatValue = (value: number, unit: string) => {
    if (unit === 'bpm' || unit === 'mmHg' || unit === 'mg/dL') {
      return `${Math.round(value)} ${unit}`;
    }
    if (unit === '%') {
      return `${value.toFixed(1)}%`;
    }
    if (unit === 'steps') {
      return value.toLocaleString();
    }
    return `${value} ${unit}`;
  };

  const getTrendIcon = (trend: string) => {
    switch (trend) {
      case 'improving': return <TrendingUp className="h-4 w-4 text-green-500" />;
      case 'declining': return <TrendingDown className="h-4 w-4 text-red-500" />;
      default: return <Minus className="h-4 w-4 text-muted-foreground" />;
    }
  };

  const getRiskColor = (score: number) => {
    if (score < 3) return 'text-green-500';
    if (score < 6) return 'text-yellow-500';
    if (score < 9) return 'text-orange-500';
    return 'text-red-500';
  };

  const getDeviceIcon = (deviceType: string) => {
    switch (deviceType) {
      case 'smartwatch': return <Watch className="h-5 w-5" />;
      case 'bp_monitor': return <Gauge className="h-5 w-5" />;
      case 'glucose_meter': return <Droplets className="h-5 w-5" />;
      case 'thermometer': return <Thermometer className="h-5 w-5" />;
      case 'pulse_oximeter': return <Heart className="h-5 w-5" />;
      default: return <Activity className="h-5 w-5" />;
    }
  };

  if (isLoading) {
    return (
      <div className="space-y-4">
        <Skeleton className="h-32 w-full" />
        <Skeleton className="h-48 w-full" />
        <Skeleton className="h-24 w-full" />
      </div>
    );
  }

  const hasDevices = deviceData?.connections && deviceData.connections.length > 0;
  const hasAnalytics = deviceData?.health_analytics && deviceData.health_analytics.length > 0;

  if (!hasDevices) {
    return (
      <Card className="border-dashed">
        <CardContent className="p-8 text-center">
          <Watch className="h-12 w-12 mx-auto mb-4 text-muted-foreground" />
          <h3 className="text-lg font-semibold mb-2">No Devices Connected</h3>
          <p className="text-muted-foreground mb-4">
            Connect your wearable devices to see real-time health metrics
          </p>
          <Link href="/device-connect">
            <Button data-testid="button-connect-device">
              <Watch className="h-4 w-4 mr-2" />
              Connect Device
            </Button>
          </Link>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-lg font-semibold">Connected Devices</h3>
          <p className="text-sm text-muted-foreground">
            {deviceData.connections.length} device{deviceData.connections.length !== 1 ? 's' : ''} synced
            {deviceData.last_sync && ` • Last sync: ${new Date(deviceData.last_sync).toLocaleTimeString()}`}
          </p>
        </div>
        <Button variant="outline" size="sm" onClick={() => refetch()} data-testid="button-refresh-devices">
          <RefreshCw className="h-4 w-4 mr-2" />
          Sync Now
        </Button>
      </div>

      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
        {deviceData.connections.map((device) => (
          <Card key={device.id} className="hover-elevate">
            <CardContent className="p-4">
              <div className="flex items-start justify-between mb-3">
                <div className="flex items-center gap-3">
                  <div className="p-2 rounded-lg bg-primary/10 text-primary">
                    {getDeviceIcon(device.device_type)}
                  </div>
                  <div>
                    <p className="font-medium">{device.device_name}</p>
                    <p className="text-xs text-muted-foreground capitalize">
                      {device.vendor_id.replace('_', ' ')}
                    </p>
                  </div>
                </div>
                <Badge variant={device.status === 'active' ? 'default' : 'secondary'}>
                  {device.status}
                </Badge>
              </div>
              {device.last_sync_at && (
                <p className="text-xs text-muted-foreground">
                  Last synced: {new Date(device.last_sync_at).toLocaleString()}
                </p>
              )}
            </CardContent>
          </Card>
        ))}
      </div>

      {hasAnalytics && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Activity className="h-5 w-5 text-primary" />
              Health Section Analytics
            </CardTitle>
            <CardDescription>
              AI-powered risk analysis from your device data
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid gap-4 md:grid-cols-2">
              {deviceData.health_analytics.map((section) => (
                <div key={section.section} className="p-4 rounded-lg bg-muted/50">
                  <div className="flex items-center justify-between mb-2">
                    <span className="font-medium capitalize">{section.section.replace('_', ' ')}</span>
                    <div className="flex items-center gap-2">
                      {getTrendIcon(section.trend)}
                      <span className={`text-sm font-semibold ${getRiskColor(section.risk_score)}`}>
                        Risk: {section.risk_score.toFixed(1)}
                      </span>
                    </div>
                  </div>
                  <div className="grid grid-cols-2 gap-2 text-sm">
                    <div>
                      <span className="text-muted-foreground">Stability:</span>
                      <span className="ml-1 font-medium">{(section.stability_score * 100).toFixed(0)}%</span>
                    </div>
                    <div>
                      <span className="text-muted-foreground">Deterioration:</span>
                      <span className="ml-1 font-medium">{section.deterioration_index.toFixed(2)}</span>
                    </div>
                  </div>
                  <Progress value={section.stability_score * 100} className="mt-2 h-1" />
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      <Card>
        <CardContent className="p-4">
          <div className="flex items-center justify-between">
            <div className="space-y-1">
              <h4 className="font-medium">Manage Your Devices</h4>
              <p className="text-sm text-muted-foreground">
                Add new devices, update settings, or view detailed metrics
              </p>
            </div>
            <Link href="/device-connect">
              <Button variant="outline" className="gap-2" data-testid="button-manage-devices">
                <Watch className="h-4 w-4" />
                Device Settings
                <ExternalLink className="h-3 w-3" />
              </Button>
            </Link>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

function TremorGaitEdemaInsights24h({ patientId }: { patientId: string }) {
  const { data: tremorDashboard, isLoading: tremorLoading } = useQuery<TremorDashboard>({
    queryKey: ['/api/v1/tremor/dashboard', patientId],
    queryFn: async () => {
      const response = await fetch(`/api/v1/tremor/dashboard/${patientId}`);
      if (!response.ok) {
        if (response.status === 404 || response.status === 502) return null;
        throw new Error('Failed to fetch tremor data');
      }
      return response.json();
    },
    enabled: !!patientId,
  });

  const { data: gaitSessions, isLoading: gaitLoading } = useQuery<{ sessions: GaitSession[] }>({
    queryKey: ['/api/v1/gait-analysis/sessions', patientId],
    queryFn: async () => {
      const response = await fetch(`/api/v1/gait-analysis/sessions/${patientId}?limit=1`);
      if (!response.ok) {
        if (response.status === 404 || response.status === 502) return { sessions: [] };
        throw new Error('Failed to fetch gait data');
      }
      return response.json();
    },
    enabled: !!patientId,
  });

  const { data: edemaMetrics, isLoading: edemaLoading } = useQuery<EdemaMetrics[]>({
    queryKey: ['/api/v1/edema/metrics', patientId],
    queryFn: async () => {
      const response = await fetch(`/api/v1/edema/metrics/${patientId}?limit=1`);
      if (!response.ok) {
        if (response.status === 404 || response.status === 502) return [];
        throw new Error('Failed to fetch edema data');
      }
      return response.json();
    },
    enabled: !!patientId,
  });

  const isLoading = tremorLoading || gaitLoading || edemaLoading;
  const latestTremor = tremorDashboard?.latest_tremor;
  const latestGait = gaitSessions?.sessions?.[0];
  const latestEdema = edemaMetrics?.[0];

  const is24h = (dateStr?: string) => {
    if (!dateStr) return false;
    return (Date.now() - new Date(dateStr).getTime()) < 24 * 60 * 60 * 1000;
  };

  const hasTremorToday = latestTremor && is24h(latestTremor.created_at);
  const hasGaitToday = latestGait && latestGait.status === 'completed' && is24h(latestGait.created_at);
  const hasEdemaToday = latestEdema && is24h(latestEdema.analyzed_at);

  if (isLoading) {
    return (
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-base">Advanced AI Analysis (24h)</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-3 gap-3">
            <Skeleton className="h-24" />
            <Skeleton className="h-24" />
            <Skeleton className="h-24" />
          </div>
        </CardContent>
      </Card>
    );
  }

  if (!hasTremorToday && !hasGaitToday && !hasEdemaToday) {
    return null;
  }

  const getTremorSeverity = (index: number) => {
    if (index < 2) return { label: 'Normal', color: 'text-green-500' };
    if (index < 4) return { label: 'Mild', color: 'text-yellow-500' };
    if (index < 6) return { label: 'Moderate', color: 'text-orange-500' };
    return { label: 'Significant', color: 'text-red-500' };
  };

  const getEdemaStatus = (severity: string, detected: boolean) => {
    if (!detected) return { label: 'None', color: 'text-green-500' };
    switch (severity?.toLowerCase()) {
      case 'mild': return { label: 'Mild', color: 'text-yellow-500' };
      case 'moderate': return { label: 'Moderate', color: 'text-orange-500' };
      case 'severe': return { label: 'Significant', color: 'text-red-500' };
      default: return { label: 'Detected', color: 'text-yellow-500' };
    }
  };

  const getFallRisk = (score: number) => {
    if (score < 0.3) return { label: 'Low', color: 'text-green-500' };
    if (score < 0.6) return { label: 'Moderate', color: 'text-yellow-500' };
    return { label: 'Elevated', color: 'text-red-500' };
  };

  return (
    <Card data-testid="card-24h-advanced-insights">
      <CardHeader className="pb-2">
        <CardTitle className="text-base flex items-center gap-2">
          <Activity className="h-4 w-4 text-primary" />
          Advanced AI Analysis (Last 24 Hours)
        </CardTitle>
        <CardDescription className="text-xs">
          Tremor, Gait, and Edema patterns from your latest examination
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-3 gap-3">
          {/* Tremor */}
          <div className={`p-3 rounded-lg border ${hasTremorToday ? 'bg-muted/50' : 'bg-muted/20'}`}>
            <div className="flex items-center gap-2 mb-2">
              <Vibrate className="h-4 w-4 text-purple-500" />
              <span className="text-sm font-medium">Tremor</span>
            </div>
            {hasTremorToday && latestTremor ? (
              <div className="space-y-1">
                <div className="flex items-center justify-between">
                  <span className="text-xs text-muted-foreground">Index</span>
                  <span className={`text-sm font-bold ${getTremorSeverity(latestTremor.tremor_index || 0).color}`}>
                    {(latestTremor.tremor_index || 0).toFixed(1)}/10
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-xs text-muted-foreground">Status</span>
                  <Badge variant="outline" className="text-xs px-1">
                    {getTremorSeverity(latestTremor.tremor_index || 0).label}
                  </Badge>
                </div>
                {tremorDashboard?.trend?.status !== 'insufficient_data' && (
                  <div className="flex items-center gap-1 text-xs pt-1">
                    {tremorDashboard.trend.status === 'increasing' && <TrendingUp className="h-3 w-3 text-orange-500" />}
                    {tremorDashboard.trend.status === 'decreasing' && <TrendingDown className="h-3 w-3 text-green-500" />}
                    {tremorDashboard.trend.status === 'stable' && <Minus className="h-3 w-3 text-blue-500" />}
                    <span className="text-muted-foreground capitalize">{tremorDashboard.trend.status}</span>
                  </div>
                )}
              </div>
            ) : (
              <div className="text-center py-2">
                <span className="text-xs text-muted-foreground">No data today</span>
              </div>
            )}
          </div>

          {/* Gait */}
          <div className={`p-3 rounded-lg border ${hasGaitToday ? 'bg-muted/50' : 'bg-muted/20'}`}>
            <div className="flex items-center gap-2 mb-2">
              <Footprints className="h-4 w-4 text-blue-500" />
              <span className="text-sm font-medium">Gait</span>
            </div>
            {hasGaitToday && latestGait ? (
              <div className="space-y-1">
                <div className="flex items-center justify-between">
                  <span className="text-xs text-muted-foreground">Strides</span>
                  <span className="text-sm font-bold">{latestGait.total_strides}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-xs text-muted-foreground">Fall Risk</span>
                  <Badge variant="outline" className={`text-xs px-1 ${getFallRisk(latestGait.abnormality_score).color}`}>
                    {getFallRisk(latestGait.abnormality_score).label}
                  </Badge>
                </div>
              </div>
            ) : (
              <div className="text-center py-2">
                <span className="text-xs text-muted-foreground">No data today</span>
              </div>
            )}
          </div>

          {/* Edema */}
          <div className={`p-3 rounded-lg border ${hasEdemaToday ? 'bg-muted/50' : 'bg-muted/20'}`}>
            <div className="flex items-center gap-2 mb-2">
              <Droplets className="h-4 w-4 text-cyan-500" />
              <span className="text-sm font-medium">Edema</span>
            </div>
            {hasEdemaToday && latestEdema ? (
              <div className="space-y-1">
                <div className="flex items-center justify-between">
                  <span className="text-xs text-muted-foreground">Expansion</span>
                  <span className="text-sm font-bold">{latestEdema.overall_expansion_percent?.toFixed(1) || 0}%</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-xs text-muted-foreground">Status</span>
                  <Badge variant="outline" className={`text-xs px-1 ${getEdemaStatus(latestEdema.swelling_severity, latestEdema.swelling_detected).color}`}>
                    {getEdemaStatus(latestEdema.swelling_severity, latestEdema.swelling_detected).label}
                  </Badge>
                </div>
              </div>
            ) : (
              <div className="text-center py-2">
                <span className="text-xs text-muted-foreground">No data today</span>
              </div>
            )}
          </div>
        </div>

        <p className="text-xs text-muted-foreground mt-3 italic text-center">
          Wellness indicators only - discuss changes with your healthcare provider
        </p>
      </CardContent>
    </Card>
  );
}

function DeviationAnalysisTab() {
  const { toast } = useToast();

  const { data: deviationSummary, isLoading: loadingSummary, error: summaryError, refetch: refetchSummary } = useQuery<DeviationSummary[]>({
    queryKey: ['/api/v1/deviation/summary/me'],
    queryFn: async () => {
      const response = await fetch('/api/v1/deviation/summary/me?days=7');
      if (!response.ok) {
        if (response.status === 404 || response.status === 403) return [];
        if (response.status === 502) {
          throw new Error('Health analysis service temporarily unavailable');
        }
        throw new Error('Failed to fetch deviation summary');
      }
      return response.json();
    },
    retry: 1,
  });

  const { data: recentDeviations, isLoading: loadingDeviations, error: deviationsError, refetch: refetchDeviations } = useQuery<DeviationRecord[]>({
    queryKey: ['/api/v1/deviation/me'],
    queryFn: async () => {
      const response = await fetch('/api/v1/deviation/me?days=7&alert_only=true');
      if (!response.ok) {
        if (response.status === 404 || response.status === 403) return [];
        if (response.status === 502) {
          throw new Error('Health analysis service temporarily unavailable');
        }
        throw new Error('Failed to fetch deviations');
      }
      return response.json();
    },
    retry: 1,
  });

  const { data: currentBaseline, isLoading: loadingBaseline, refetch: refetchBaseline } = useQuery<BaselineInfo | null>({
    queryKey: ['/api/v1/baseline/current/me'],
    queryFn: async () => {
      const response = await fetch('/api/v1/baseline/current/me');
      if (!response.ok) {
        if (response.status === 404 || response.status === 403 || response.status === 502) return null;
        throw new Error('Failed to fetch baseline');
      }
      return response.json();
    },
    retry: 1,
  });

  const recalculateBaselineMutation = useMutation({
    mutationFn: async () => {
      const response = await fetch('/api/v1/baseline/calculate/me', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
      });
      if (!response.ok) {
        const error = await response.json().catch(() => ({ message: 'Failed to recalculate baseline' }));
        throw new Error(error.message || error.detail || 'Failed to recalculate baseline');
      }
      return response.json();
    },
    onSuccess: () => {
      toast({
        title: "Baseline Recalculated",
        description: "Your personal health baseline has been updated with the latest data.",
      });
      refetchBaseline();
      refetchSummary();
      refetchDeviations();
      queryClient.invalidateQueries({ queryKey: ['/api/v1/deviation'] });
    },
    onError: (error: Error) => {
      toast({
        title: "Recalculation Failed",
        description: error.message || "Unable to recalculate baseline. Please try again.",
        variant: "destructive",
      });
    },
  });

  const isLoading = loadingSummary || loadingDeviations || loadingBaseline;
  const hasError = summaryError || deviationsError;
  const hasDeviations = deviationSummary && deviationSummary.length > 0;
  const criticalCount = deviationSummary?.reduce((sum, d) => sum + d.critical_count, 0) || 0;
  const moderateCount = deviationSummary?.reduce((sum, d) => sum + d.moderate_count, 0) || 0;
  const totalAlerts = recentDeviations?.filter(d => d.alert_triggered).length || 0;

  const getSeverityColor = (severity: string) => {
    switch (severity?.toLowerCase()) {
      case 'critical': return 'text-red-600 bg-red-100 dark:bg-red-900/30';
      case 'moderate': return 'text-amber-600 bg-amber-100 dark:bg-amber-900/30';
      case 'mild': return 'text-yellow-600 bg-yellow-100 dark:bg-yellow-900/30';
      default: return 'text-muted-foreground bg-muted/50';
    }
  };

  const getDeviationTypeIcon = (type: string) => {
    switch (type?.toLowerCase()) {
      case 'pain_facial':
      case 'pain_self_reported':
        return <Zap className="h-4 w-4 text-orange-500" />;
      case 'respiratory_rate':
        return <Wind className="h-4 w-4 text-blue-500" />;
      case 'heart_rate':
        return <Heart className="h-4 w-4 text-red-500" />;
      case 'symptom_severity':
        return <Activity className="h-4 w-4 text-purple-500" />;
      case 'temperature':
        return <Thermometer className="h-4 w-4 text-orange-500" />;
      default:
        return <Gauge className="h-4 w-4 text-primary" />;
    }
  };

  const formatMetricName = (name: string) => {
    return name?.replace(/_/g, ' ')
      .replace(/\b\w/g, (c) => c.toUpperCase());
  };

  const getTrendIcon = (direction: string | null) => {
    if (!direction) return null;
    switch (direction.toLowerCase()) {
      case 'increasing': return <TrendingUp className="h-3 w-3 text-red-500" />;
      case 'decreasing': return <TrendingDown className="h-3 w-3 text-green-500" />;
      case 'stable': return <Minus className="h-3 w-3 text-blue-500" />;
      default: return null;
    }
  };

  if (isLoading) {
    return (
      <Card data-testid="card-deviation-loading">
        <CardHeader>
          <Skeleton className="h-6 w-48" data-testid="skeleton-title" />
          <Skeleton className="h-4 w-64 mt-1" data-testid="skeleton-description" />
        </CardHeader>
        <CardContent>
          <div className="grid gap-3 md:grid-cols-2" data-testid="skeleton-grid">
            <Skeleton className="h-24" />
            <Skeleton className="h-24" />
            <Skeleton className="h-24" />
            <Skeleton className="h-24" />
          </div>
        </CardContent>
      </Card>
    );
  }

  if (hasError) {
    return (
      <Card data-testid="card-deviation-error">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <AlertCircle className="h-5 w-5 text-amber-500" />
            Deviation Analysis
          </CardTitle>
          <CardDescription>
            Unable to load deviation data at this time
          </CardDescription>
        </CardHeader>
        <CardContent>
          <Alert className="border-amber-500/50 bg-amber-500/5" data-testid="alert-deviation-error">
            <AlertCircle className="h-4 w-4 text-amber-500" />
            <AlertTitle className="text-amber-700 dark:text-amber-400">
              Service Temporarily Unavailable
            </AlertTitle>
            <AlertDescription className="text-amber-600 dark:text-amber-300">
              {(summaryError as Error)?.message || (deviationsError as Error)?.message || 'Unable to load deviation analysis. Please try again later.'}
            </AlertDescription>
          </Alert>
          <div className="mt-4 flex justify-center">
            <Button variant="outline" onClick={() => refetchSummary()} data-testid="button-retry-deviations">
              <RefreshCw className="h-4 w-4 mr-2" />
              Try Again
            </Button>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card data-testid="card-deviation-analysis">
      <CardHeader>
        <div className="flex items-center justify-between flex-wrap gap-2">
          <div>
            <CardTitle className="flex items-center gap-2">
              <BarChart3 className="h-5 w-5 text-primary" />
              Deviation Analysis
              {criticalCount > 0 && (
                <Badge variant="destructive" data-testid="badge-critical-deviations">
                  {criticalCount} Critical
                </Badge>
              )}
              {moderateCount > 0 && (
                <Badge variant="secondary" className="bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-300" data-testid="badge-moderate-deviations">
                  {moderateCount} Moderate
                </Badge>
              )}
            </CardTitle>
            <CardDescription>
              Comparing your recent metrics to your personal health baseline (7-day window)
            </CardDescription>
          </div>
          <div className="flex items-center gap-2">
            <Button 
              variant="outline" 
              size="sm" 
              onClick={() => recalculateBaselineMutation.mutate()} 
              disabled={recalculateBaselineMutation.isPending}
              data-testid="button-recalculate-baseline"
            >
              {recalculateBaselineMutation.isPending ? (
                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
              ) : (
                <Calculator className="h-4 w-4 mr-2" />
              )}
              Recalculate Baseline
            </Button>
            <Button variant="outline" size="sm" onClick={() => { refetchSummary(); refetchDeviations(); refetchBaseline(); }} data-testid="button-refresh-deviations">
              <RefreshCw className="h-4 w-4 mr-2" />
              Refresh
            </Button>
          </div>
        </div>

        {/* Baseline Status Section */}
        {currentBaseline && (
          <div className="mt-4 p-3 bg-muted/50 rounded-lg border" data-testid="section-baseline-status">
            <div className="flex items-center justify-between flex-wrap gap-2">
              <div className="flex items-center gap-2">
                <Clock className="h-4 w-4 text-muted-foreground" />
                <span className="text-sm text-muted-foreground">Current Baseline:</span>
                <span className="text-sm font-medium" data-testid="text-baseline-dates">
                  {new Date(currentBaseline.baseline_start_date).toLocaleDateString()} — {new Date(currentBaseline.baseline_end_date).toLocaleDateString()}
                </span>
              </div>
              <div className="flex items-center gap-4 text-xs text-muted-foreground">
                <span data-testid="text-baseline-datapoints">
                  {currentBaseline.data_points_count} data points
                </span>
                {currentBaseline.baseline_quality && (
                  <Badge 
                    variant="outline" 
                    className={`text-xs ${
                      currentBaseline.baseline_quality === 'high' ? 'border-green-500 text-green-600' :
                      currentBaseline.baseline_quality === 'medium' ? 'border-amber-500 text-amber-600' :
                      'border-muted-foreground'
                    }`}
                    data-testid="badge-baseline-quality"
                  >
                    {currentBaseline.baseline_quality} quality
                  </Badge>
                )}
                <span data-testid="text-baseline-created">
                  Last updated: {new Date(currentBaseline.created_at).toLocaleDateString()}
                </span>
              </div>
            </div>
          </div>
        )}

        {!currentBaseline && !loadingBaseline && (
          <div className="mt-4 p-3 bg-amber-50 dark:bg-amber-900/20 rounded-lg border border-amber-200 dark:border-amber-800" data-testid="section-no-baseline">
            <div className="flex items-center gap-2">
              <AlertCircle className="h-4 w-4 text-amber-500" />
              <span className="text-sm text-amber-700 dark:text-amber-400">
                No baseline established yet. Click "Recalculate Baseline" to create your personal health baseline from your recent measurements.
              </span>
            </div>
          </div>
        )}
      </CardHeader>
      <CardContent>
        {!hasDeviations ? (
          <div className="flex flex-col items-center justify-center py-8" data-testid="empty-deviation-state">
            <CheckCircle2 className="h-12 w-12 text-green-500 mb-3" />
            <h4 className="font-semibold text-lg">All Metrics Within Baseline</h4>
            <p className="text-sm text-muted-foreground text-center max-w-md mt-1">
              Your recent health measurements are consistent with your personal baseline.
              Continue tracking to maintain awareness of your health patterns.
            </p>
          </div>
        ) : (
          <div className="space-y-4">
            {/* Deviation Summary Grid */}
            <div className="grid gap-3 md:grid-cols-2 lg:grid-cols-3">
              {deviationSummary?.map((summary) => (
                <div
                  key={summary.metric_name}
                  className={`p-4 rounded-lg border ${
                    summary.critical_count > 0 
                      ? 'border-red-500/50 bg-red-50/50 dark:bg-red-900/20' 
                      : summary.moderate_count > 0 
                        ? 'border-amber-500/50 bg-amber-50/50 dark:bg-amber-900/20'
                        : 'border-border bg-muted/30'
                  }`}
                  data-testid={`deviation-metric-${summary.metric_name}`}
                >
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-2">
                      {getDeviationTypeIcon(summary.metric_name)}
                      <span className="font-medium text-sm">{formatMetricName(summary.metric_name)}</span>
                    </div>
                    {summary.critical_count > 0 && (
                      <Badge variant="destructive" className="text-xs">
                        {summary.critical_count} critical
                      </Badge>
                    )}
                  </div>
                  
                  <div className="grid grid-cols-2 gap-2 text-xs">
                    <div>
                      <span className="text-muted-foreground">Total Deviations:</span>
                      <span className="ml-1 font-semibold">{summary.total_deviations}</span>
                    </div>
                    {summary.recent_z_score !== null && (
                      <div>
                        <span className="text-muted-foreground">Z-Score:</span>
                        <span className={`ml-1 font-semibold ${
                          Math.abs(summary.recent_z_score) > 2 ? 'text-red-600' :
                          Math.abs(summary.recent_z_score) > 1 ? 'text-amber-600' : ''
                        }`}>
                          {summary.recent_z_score.toFixed(2)}σ
                        </span>
                      </div>
                    )}
                  </div>
                  
                  {summary.trend_direction && (
                    <div className="flex items-center gap-1 text-xs mt-2 pt-2 border-t">
                      {getTrendIcon(summary.trend_direction)}
                      <span className="text-muted-foreground capitalize">
                        Trend: {summary.trend_direction}
                      </span>
                    </div>
                  )}
                </div>
              ))}
            </div>

            {/* Recent Deviation Alerts Timeline */}
            {recentDeviations && recentDeviations.length > 0 && (
              <div className="mt-4" data-testid="deviation-timeline">
                <h4 className="font-semibold text-sm mb-3 flex items-center gap-2">
                  <AlertCircle className="h-4 w-4 text-amber-500" />
                  Recent Deviation Alerts ({totalAlerts})
                </h4>
                <div className="max-h-[200px] overflow-y-auto pr-1 space-y-2">
                  {recentDeviations.filter(d => d.alert_triggered).slice(0, 10).map((deviation) => (
                    <div
                      key={deviation.id}
                      className={`p-3 rounded-lg border ${getSeverityColor(deviation.severity_level)} flex items-start justify-between`}
                      data-testid={`deviation-alert-${deviation.id}`}
                    >
                      <div className="flex items-start gap-2">
                        {getDeviationTypeIcon(deviation.metric_name)}
                        <div>
                          <p className="font-medium text-sm">{formatMetricName(deviation.metric_name)}</p>
                          <p className="text-xs mt-0.5">
                            {deviation.alert_message || 
                              `${deviation.percent_change >= 0 ? '+' : ''}${deviation.percent_change.toFixed(1)}% from baseline`
                            }
                          </p>
                          <p className="text-xs text-muted-foreground mt-1">
                            Value: {deviation.measurement_value.toFixed(1)} | 
                            Baseline: {deviation.baseline_mean.toFixed(1)} ± {deviation.baseline_std.toFixed(1)}
                          </p>
                        </div>
                      </div>
                      <div className="text-right">
                        <Badge variant="outline" className="text-xs capitalize">
                          {deviation.severity_level}
                        </Badge>
                        <p className="text-xs text-muted-foreground mt-1">
                          {new Date(deviation.measurement_date).toLocaleDateString()}
                        </p>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        <p className="text-xs text-muted-foreground mt-4 italic text-center">
          Deviations indicate changes from your personal baseline - not necessarily concerning.
          Discuss significant changes with your healthcare provider.
        </p>
      </CardContent>
    </Card>
  );
}

interface VoiceFollowupResult {
  id: number;
  transcription: string;
  response: string;
  extractedSymptoms: Array<{
    symptom: string;
    severity: string;
    duration?: string;
  }>;
  extractedVitals: Array<{
    type: string;
    value: string;
    unit?: string;
  }>;
  recommendedActions: string[];
  conversationSummary?: string;
  empathyLevel?: string;
  concernsRaised: boolean;
  needsFollowup: boolean;
  createdAt: string;
}

function VoiceFollowupTab() {
  const { toast } = useToast();
  const [isRecording, setIsRecording] = useState(false);
  const [recordingTime, setRecordingTime] = useState(0);
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [transcription, setTranscription] = useState<string>('');
  const [isEditing, setIsEditing] = useState(false);
  
  const mediaRecorderRef = React.useRef<MediaRecorder | null>(null);
  const audioChunksRef = React.useRef<Blob[]>([]);
  const timerRef = React.useRef<NodeJS.Timeout | null>(null);
  const maxDuration = 120; // 2 minutes

  const { data: recentFollowups, isLoading: loadingHistory, refetch: refetchHistory } = useQuery<VoiceFollowupResult[]>({
    queryKey: ['/api/voice-followup/recent'],
    queryFn: async () => {
      const response = await fetch('/api/voice-followup/recent?limit=5');
      if (!response.ok) {
        if (response.status === 401 || response.status === 404) return [];
        throw new Error('Failed to fetch voice history');
      }
      return response.json();
    },
    retry: 1,
  });

  const uploadMutation = useMutation({
    mutationFn: async ({ audio, transcript }: { audio: Blob; transcript?: string }) => {
      const formData = new FormData();
      formData.append('audio', audio, `voice-followup-${Date.now()}.webm`);
      if (transcript) {
        formData.append('transcription', transcript);
      }
      
      const response = await fetch('/api/voice-followup/upload', {
        method: 'POST',
        body: formData,
        credentials: 'include',
      });

      if (!response.ok) {
        const error = await response.json().catch(() => ({ message: 'Upload failed' }));
        throw new Error(error.message || 'Failed to process voice recording');
      }

      return response.json();
    },
    onSuccess: (data) => {
      toast({
        title: "Voice Check-in Complete",
        description: "Your recording has been processed and analyzed.",
      });
      setTranscription(data.transcription || '');
      refetchHistory();
      queryClient.invalidateQueries({ queryKey: ['/api/voice-followup'] });
    },
    onError: (error: Error) => {
      toast({
        title: "Processing Failed",
        description: error.message || "Unable to process your recording. Please try again.",
        variant: "destructive",
      });
    },
  });

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });

      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = () => {
        const blob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
        setAudioBlob(blob);
        setAudioUrl(URL.createObjectURL(blob));
        stream.getTracks().forEach((track) => track.stop());
      };

      mediaRecorder.start();
      setIsRecording(true);
      setRecordingTime(0);
      setTranscription('');

      timerRef.current = setInterval(() => {
        setRecordingTime((prev) => {
          const newTime = prev + 1;
          if (newTime >= maxDuration) {
            stopRecording();
            return maxDuration;
          }
          return newTime;
        });
      }, 1000);
    } catch (error) {
      console.error('Error accessing microphone:', error);
      toast({
        title: "Microphone Access Denied",
        description: "Please allow microphone access to record voice messages.",
        variant: "destructive",
      });
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      
      if (timerRef.current) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }
    }
  };

  const handleUpload = () => {
    if (audioBlob) {
      uploadMutation.mutate({ 
        audio: audioBlob, 
        transcript: isEditing ? transcription : undefined 
      });
    }
  };

  const handleReset = () => {
    setAudioBlob(null);
    if (audioUrl) {
      URL.revokeObjectURL(audioUrl);
    }
    setAudioUrl(null);
    setRecordingTime(0);
    setTranscription('');
    setIsEditing(false);
  };

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const progressPercentage = (recordingTime / maxDuration) * 100;

  return (
    <div className="space-y-6">
      {/* Voice Recording Card */}
      <Card data-testid="card-voice-recording">
        <CardHeader>
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-full bg-gradient-to-br from-pink-400 to-red-500 text-white">
              <Mic className="h-5 w-5" />
            </div>
            <div>
              <CardTitle>Voice Check-in</CardTitle>
              <CardDescription>
                Speak naturally about how you're feeling - our AI will extract symptoms and vitals
              </CardDescription>
            </div>
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Initial State - Start Recording */}
          {!audioBlob && !isRecording && (
            <div className="text-center py-8">
              <Button
                size="lg"
                onClick={startRecording}
                className="h-20 w-20 rounded-full"
                data-testid="button-start-voice-recording"
              >
                <Mic className="h-8 w-8" />
              </Button>
              <p className="text-sm text-muted-foreground mt-4">
                Tap to start recording (up to {maxDuration / 60} minutes)
              </p>
              <p className="text-xs text-muted-foreground mt-2">
                Example: "I've been having headaches for the past two days, my temperature feels normal, 
                and I slept okay last night. My energy level is about a 6 out of 10."
              </p>
            </div>
          )}

          {/* Recording State */}
          {isRecording && (
            <div className="space-y-4">
              <div className="text-center py-4">
                <div className="flex items-center justify-center gap-3 mb-4">
                  <div className="h-4 w-4 rounded-full bg-red-500 animate-pulse" />
                  <span className="text-3xl font-mono font-bold" data-testid="text-voice-recording-time">
                    {formatTime(recordingTime)}
                  </span>
                </div>
                <Progress value={progressPercentage} className="h-2" />
                <p className="text-xs text-muted-foreground mt-2">
                  {formatTime(maxDuration - recordingTime)} remaining
                </p>
              </div>

              <div className="flex justify-center">
                <Button
                  variant="destructive"
                  size="lg"
                  onClick={stopRecording}
                  className="h-20 w-20 rounded-full"
                  data-testid="button-stop-voice-recording"
                >
                  <Square className="h-8 w-8" />
                </Button>
              </div>

              <p className="text-center text-sm text-muted-foreground">
                Speak clearly about your symptoms, how you're feeling, and any vital signs you've measured
              </p>
            </div>
          )}

          {/* Review State - Audio recorded */}
          {audioBlob && !uploadMutation.isSuccess && (
            <div className="space-y-4">
              <div className="p-4 bg-muted rounded-lg">
                <p className="text-sm font-medium mb-2 flex items-center gap-2">
                  <CheckCircle2 className="h-4 w-4 text-green-500" />
                  Recording Complete
                </p>
                <audio src={audioUrl || ''} controls className="w-full" data-testid="audio-voice-player" />
                <p className="text-xs text-muted-foreground mt-2">
                  Duration: {formatTime(recordingTime)}
                </p>
              </div>

              {/* Transcription Edit Option */}
              {transcription && (
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <Label className="flex items-center gap-2">
                      <FileText className="h-4 w-4" />
                      Transcription
                    </Label>
                    <Button 
                      variant="ghost" 
                      size="sm" 
                      onClick={() => setIsEditing(!isEditing)}
                      data-testid="button-edit-transcription"
                    >
                      {isEditing ? 'Done Editing' : 'Edit'}
                    </Button>
                  </div>
                  {isEditing ? (
                    <Textarea 
                      value={transcription}
                      onChange={(e) => setTranscription(e.target.value)}
                      className="min-h-[100px]"
                      data-testid="textarea-transcription"
                    />
                  ) : (
                    <p className="text-sm p-3 bg-muted rounded-lg" data-testid="text-transcription-preview">
                      {transcription}
                    </p>
                  )}
                </div>
              )}

              <div className="flex gap-2">
                <Button
                  onClick={handleUpload}
                  disabled={uploadMutation.isPending}
                  className="flex-1"
                  data-testid="button-submit-voice"
                >
                  {uploadMutation.isPending ? (
                    <>
                      <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                      Processing...
                    </>
                  ) : (
                    <>
                      <Upload className="w-4 h-4 mr-2" />
                      Submit Voice Check-in
                    </>
                  )}
                </Button>
                <Button
                  variant="outline"
                  onClick={handleReset}
                  disabled={uploadMutation.isPending}
                  data-testid="button-rerecord-voice"
                >
                  Re-record
                </Button>
              </div>
            </div>
          )}

          {/* Success State - Show Results */}
          {uploadMutation.isSuccess && uploadMutation.data && (
            <div className="space-y-4">
              <Alert className="border-green-500/50 bg-green-500/5">
                <CheckCircle2 className="h-4 w-4 text-green-500" />
                <AlertTitle className="text-green-700 dark:text-green-400">
                  Voice Check-in Processed
                </AlertTitle>
                <AlertDescription className="text-green-600 dark:text-green-300">
                  {uploadMutation.data.conversationSummary || 'Your voice message has been analyzed successfully.'}
                </AlertDescription>
              </Alert>

              {/* AI Response */}
              {uploadMutation.data.response && (
                <Card className="border-primary/20">
                  <CardHeader className="pb-2">
                    <CardTitle className="text-base flex items-center gap-2">
                      <Heart className="h-4 w-4 text-pink-500" />
                      Health Companion Response
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="text-sm leading-relaxed" data-testid="text-ai-response">
                      {uploadMutation.data.response}
                    </p>
                  </CardContent>
                </Card>
              )}

              {/* Extracted Symptoms */}
              {uploadMutation.data.extractedSymptoms && uploadMutation.data.extractedSymptoms.length > 0 && (
                <div className="p-4 bg-muted rounded-lg">
                  <p className="font-medium text-sm mb-2 flex items-center gap-2">
                    <Stethoscope className="h-4 w-4 text-blue-500" />
                    Extracted Symptoms
                  </p>
                  <div className="flex flex-wrap gap-2">
                    {uploadMutation.data.extractedSymptoms.map((symptom: any, idx: number) => (
                      <Badge key={idx} variant="secondary" data-testid={`badge-symptom-${idx}`}>
                        {symptom.symptom}
                        {symptom.severity && ` (${symptom.severity})`}
                      </Badge>
                    ))}
                  </div>
                </div>
              )}

              {/* Extracted Vitals */}
              {uploadMutation.data.extractedVitals && uploadMutation.data.extractedVitals.length > 0 && (
                <div className="p-4 bg-muted rounded-lg">
                  <p className="font-medium text-sm mb-2 flex items-center gap-2">
                    <Activity className="h-4 w-4 text-green-500" />
                    Mentioned Vitals
                  </p>
                  <div className="grid grid-cols-2 gap-2">
                    {uploadMutation.data.extractedVitals.map((vital: any, idx: number) => (
                      <div key={idx} className="text-sm p-2 bg-background rounded" data-testid={`vital-${idx}`}>
                        <span className="text-muted-foreground">{vital.type}:</span>{' '}
                        <span className="font-medium">{vital.value} {vital.unit || ''}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Recommended Actions */}
              {uploadMutation.data.recommendedActions && uploadMutation.data.recommendedActions.length > 0 && (
                <div className="p-4 bg-blue-50 dark:bg-blue-950/30 rounded-lg">
                  <p className="font-medium text-sm mb-2 flex items-center gap-2">
                    <Pill className="h-4 w-4 text-blue-500" />
                    Suggested Actions
                  </p>
                  <ul className="list-disc list-inside space-y-1 text-sm">
                    {uploadMutation.data.recommendedActions.map((action: string, idx: number) => (
                      <li key={idx} className="text-muted-foreground" data-testid={`action-${idx}`}>
                        {action}
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {/* Concern Badges */}
              <div className="flex flex-wrap gap-2">
                {uploadMutation.data.empathyLevel && (
                  <Badge variant="outline" className="capitalize">
                    {uploadMutation.data.empathyLevel}
                  </Badge>
                )}
                {uploadMutation.data.concernsRaised && (
                  <Badge variant="outline" className="bg-yellow-50 dark:bg-yellow-950 text-yellow-700 dark:text-yellow-300">
                    <AlertCircle className="w-3 h-3 mr-1" />
                    Concerns Noted
                  </Badge>
                )}
                {uploadMutation.data.needsFollowup && (
                  <Badge variant="outline" className="bg-blue-50 dark:bg-blue-950 text-blue-700 dark:text-blue-300">
                    Follow-up Recommended
                  </Badge>
                )}
              </div>

              <Button onClick={handleReset} variant="outline" className="w-full" data-testid="button-new-voice-recording">
                <Mic className="h-4 w-4 mr-2" />
                Record Another Check-in
              </Button>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Recent Voice Check-ins History */}
      <Card data-testid="card-voice-history">
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="flex items-center gap-2">
                <Clock className="h-5 w-5 text-muted-foreground" />
                Recent Voice Check-ins
              </CardTitle>
              <CardDescription>
                Your latest voice recordings and extracted health data
              </CardDescription>
            </div>
            <Button variant="outline" size="sm" onClick={() => refetchHistory()} data-testid="button-refresh-voice-history">
              <RefreshCw className="h-4 w-4" />
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          {loadingHistory ? (
            <div className="space-y-3">
              <Skeleton className="h-20 w-full" />
              <Skeleton className="h-20 w-full" />
            </div>
          ) : recentFollowups && recentFollowups.length > 0 ? (
            <div className="space-y-3">
              {recentFollowups.map((followup, idx) => (
                <div 
                  key={followup.id || idx} 
                  className="p-4 border rounded-lg hover-elevate"
                  data-testid={`voice-history-${idx}`}
                >
                  <div className="flex items-start justify-between mb-2">
                    <div className="flex items-center gap-2">
                      <Mic className="h-4 w-4 text-primary" />
                      <span className="text-sm font-medium">
                        {new Date(followup.createdAt).toLocaleDateString()} at{' '}
                        {new Date(followup.createdAt).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                      </span>
                    </div>
                    <div className="flex gap-1">
                      {followup.concernsRaised && (
                        <Badge variant="outline" className="text-xs bg-yellow-50 dark:bg-yellow-950">
                          Concerns
                        </Badge>
                      )}
                      {followup.needsFollowup && (
                        <Badge variant="outline" className="text-xs bg-blue-50 dark:bg-blue-950">
                          Follow-up
                        </Badge>
                      )}
                    </div>
                  </div>
                  
                  {followup.conversationSummary && (
                    <p className="text-sm text-muted-foreground mb-2 line-clamp-2">
                      {followup.conversationSummary}
                    </p>
                  )}
                  
                  {followup.extractedSymptoms && followup.extractedSymptoms.length > 0 && (
                    <div className="flex flex-wrap gap-1">
                      {followup.extractedSymptoms.slice(0, 4).map((s, sidx) => (
                        <Badge key={sidx} variant="secondary" className="text-xs">
                          {s.symptom}
                        </Badge>
                      ))}
                      {followup.extractedSymptoms.length > 4 && (
                        <Badge variant="outline" className="text-xs">
                          +{followup.extractedSymptoms.length - 4} more
                        </Badge>
                      )}
                    </div>
                  )}
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center py-8">
              <Mic className="h-12 w-12 mx-auto mb-4 text-muted-foreground/50" />
              <p className="text-muted-foreground">No voice check-ins yet</p>
              <p className="text-sm text-muted-foreground">
                Record your first voice check-in to see your history here
              </p>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}

function LegalDisclaimer() {
  return (
    <Alert className="border-amber-500/50 bg-amber-500/5">
      <AlertCircle className="h-4 w-4 text-amber-500" />
      <AlertTitle className="text-amber-700 dark:text-amber-400 font-semibold">
        Wellness Monitoring Platform - Not a Medical Device
      </AlertTitle>
      <AlertDescription className="text-amber-600 dark:text-amber-300 text-sm">
        This platform is not a medical device and is not intended to diagnose, treat, cure, or prevent any disease.
        All information provided is for wellness monitoring and personal tracking purposes only.
        Please discuss any changes or concerns with your healthcare provider.
      </AlertDescription>
    </Alert>
  );
}

export default function DailyFollowup() {
  const { user } = useAuth();
  const [customLocation, setCustomLocation] = useState('');
  const [customDescription, setCustomDescription] = useState('');

  const {
    sessionId,
    currentStep,
    currentStepIndex,
    totalSteps,
    workflowState,
    examSteps,
    progress,
    completedSteps,
    startExam,
    startRecording,
    completeRecording,
    skipCurrentStep,
    resetWorkflow,
    isStarting,
    isUploading,
  } = useGuidedExamWorkflow();

  // Fetch today's video metrics (last 24 hours)
  const { data: todayMetrics } = useQuery<any>({
    queryKey: ["/api/video-ai/latest-metrics"],
    enabled: !!user && (workflowState === 'idle' || workflowState === 'completed'),
  });

  const handleRecordingComplete = (videoBlob: Blob, durationSeconds: number) => {
    if (currentStep?.examType === 'custom') {
      completeRecording(videoBlob, durationSeconds, {
        location: customLocation,
        description: customDescription,
      });
      setCustomLocation('');
      setCustomDescription('');
    } else {
      completeRecording(videoBlob, durationSeconds);
    }
  };

  // Check if metrics are from today (last 24 hours)
  const hasMetricsToday = todayMetrics && todayMetrics.created_at && 
    (Date.now() - new Date(todayMetrics.created_at).getTime()) < 24 * 60 * 60 * 1000;

  // Active examination workflow
  if (workflowState !== 'idle') {
    return (
      <div className="container mx-auto p-6 max-w-5xl space-y-6">
        {/* Progress Header */}
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h1 className="text-3xl font-bold tracking-tight" data-testid="text-exam-title">
              Guided Examination
            </h1>
            <Badge variant="secondary" className="text-sm">
              Step {currentStepIndex + 1} of {totalSteps}
            </Badge>
          </div>

          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <span className="text-muted-foreground">Overall Progress</span>
              <span className="font-medium">{completedSteps} / {totalSteps} exams</span>
            </div>
            <Progress value={progress} className="h-3" data-testid="progress-overall" />
          </div>
        </div>

        {/* Exam Steps Progress Indicators */}
        <div className="grid grid-cols-7 gap-2">
          {examSteps.map((step, idx) => {
            const Icon = {
              respiratory: Wind,
              skin_pallor: Hand,
              eye_sclera: Eye,
              swelling: User,
              tremor: Activity,
              tongue: MessageSquare,
              custom: AlertCircle,
            }[step.examType] || Video;

            const isActive = idx === currentStepIndex;
            const isCompleted = step.segmentId || step.skipped;
            const isSkipped = step.skipped;

            return (
              <div
                key={step.examType}
                className={`flex flex-col items-center gap-2 p-3 rounded-lg border transition-all ${
                  isActive ? 'bg-primary/10 border-primary' : 
                  isCompleted ? 'bg-muted border-muted' : 
                  'bg-card'
                }`}
              >
                <div className={`relative ${isSkipped ? 'opacity-50' : ''}`}>
                  <Icon className={`h-5 w-5 ${
                    isActive ? 'text-primary' : 
                    isCompleted ? 'text-muted-foreground' : 
                    'text-muted-foreground/50'
                  }`} />
                  {isCompleted && !isSkipped && (
                    <CheckCircle2 className="h-3 w-3 text-chart-2 absolute -top-1 -right-1" />
                  )}
                  {isSkipped && (
                    <XCircle className="h-3 w-3 text-muted-foreground absolute -top-1 -right-1" />
                  )}
                </div>
                <span className="text-xs text-center font-medium line-clamp-2">
                  {step.examType.replace('_', ' ')}
                </span>
              </div>
            );
          })}
        </div>

        {/* Current Step Content */}
        {workflowState === 'prep' && currentStep && (
          <ExamPrepStep
            examType={currentStep.examType}
            sequenceNumber={currentStepIndex + 1}
            totalExams={totalSteps}
            onStartRecording={startRecording}
            onSkipExam={skipCurrentStep}
          />
        )}

        {workflowState === 'recording' && currentStep && (
          <div className="space-y-4">
            {currentStep.examType === 'custom' && !customLocation && (
              <Card>
                <CardHeader>
                  <CardTitle>Describe the Area of Concern</CardTitle>
                  <CardDescription>Help us understand what you want to examine</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="space-y-2">
                    <Label htmlFor="location">Location on Body</Label>
                    <Input
                      id="location"
                      placeholder="e.g., Left forearm, Right ankle, Upper back"
                      value={customLocation}
                      onChange={(e) => setCustomLocation(e.target.value)}
                      data-testid="input-custom-location"
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="description">What to Look For</Label>
                    <Textarea
                      id="description"
                      placeholder="e.g., Red rash, unusual discoloration, new mole, swelling"
                      value={customDescription}
                      onChange={(e) => setCustomDescription(e.target.value)}
                      rows={3}
                      data-testid="input-custom-description"
                    />
                  </div>
                </CardContent>
              </Card>
            )}

            {(currentStep.examType !== 'custom' || customLocation) && (
              <VideoRecorder
                examType={currentStep.examType}
                onRecordingComplete={handleRecordingComplete}
                onCancel={() => resetWorkflow()}
                maxDurationSeconds={60}
                autoStopAfter={currentStep.examType === 'respiratory' ? 30 : undefined}
              />
            )}
          </div>
        )}

        {workflowState === 'uploading' && (
          <Card>
            <CardContent className="p-12 text-center">
              <Loader2 className="h-12 w-12 animate-spin mx-auto mb-4 text-primary" />
              <h3 className="text-lg font-semibold mb-2">Uploading & Analyzing...</h3>
              <p className="text-muted-foreground">
                Securely uploading your recording and running AI analysis
              </p>
            </CardContent>
          </Card>
        )}

        {workflowState === 'completed' && (
          <Card className="border-chart-2">
            <CardContent className="p-12 text-center">
              <CheckCircle2 className="h-16 w-16 mx-auto mb-4 text-chart-2" />
              <h2 className="text-2xl font-bold mb-2">Examination Complete!</h2>
              <p className="text-muted-foreground mb-6">
                All recordings have been processed. View your analysis below.
              </p>
              <Button onClick={resetWorkflow} size="lg" data-testid="button-start-new">
                View Today's Results
              </Button>
            </CardContent>
          </Card>
        )}
      </div>
    );
  }

  // Idle state - show today's metrics or start examination
  return (
    <div className="container mx-auto p-6 max-w-7xl space-y-6">
      {/* Header */}
      <div className="space-y-2">
        <h1 className="text-4xl font-bold tracking-tight text-foreground flex items-center gap-3" data-testid="text-page-title">
          <Activity className="h-10 w-10 text-primary" />
          Daily Followup
        </h1>
        <p className="text-muted-foreground leading-relaxed">
          Complete your daily video examination and track your health metrics
        </p>
      </div>

      <LegalDisclaimer />

      {/* Tabs for Examination, Device Data, and Deviations */}
      <Tabs defaultValue="examination" className="w-full">
        <TabsList className="grid w-full grid-cols-4 max-w-2xl">
          <TabsTrigger value="examination" className="gap-2" data-testid="tab-examination">
            <Video className="h-4 w-4" />
            Examinations
          </TabsTrigger>
          <TabsTrigger value="voice" className="gap-2" data-testid="tab-voice">
            <Mic className="h-4 w-4" />
            Voice
          </TabsTrigger>
          <TabsTrigger value="devices" className="gap-2" data-testid="tab-devices">
            <Watch className="h-4 w-4" />
            Device Data
          </TabsTrigger>
          <TabsTrigger value="deviations" className="gap-2" data-testid="tab-deviations">
            <BarChart3 className="h-4 w-4" />
            Deviations
          </TabsTrigger>
        </TabsList>

        <TabsContent value="examination" className="mt-6">
          {/* Today's Metrics or Start Examination */}
          {hasMetricsToday ? (
            <div className="space-y-6">
          {/* Header with Link to Full Analysis */}
          <Card className="border-primary/20">
            <CardHeader>
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle className="flex items-center gap-2">
                    <CheckCircle2 className="h-6 w-6 text-chart-2" />
                    Today's Video AI Metrics
                  </CardTitle>
                  <CardDescription>
                    Last examination: {new Date(todayMetrics.created_at).toLocaleString()}
                  </CardDescription>
                </div>
                <Link href="/ai-video">
                  <Button variant="outline" size="sm" className="gap-2" data-testid="button-view-full-analysis">
                    <BarChart3 className="h-4 w-4" />
                    Full Analysis
                    <ExternalLink className="h-3 w-3" />
                  </Button>
                </Link>
              </div>
            </CardHeader>
            <CardContent className="space-y-4">
              {/* ML Metrics Grid */}
              <div className="grid grid-cols-2 gap-3">
                {todayMetrics.respiratory_rate_bpm && (
                  <div className="flex items-center justify-between p-3 rounded-md bg-muted/50">
                    <div className="flex items-center gap-2">
                      <Wind className="h-4 w-4 text-blue-500" />
                      <span className="text-sm text-muted-foreground">Respiratory Rate</span>
                    </div>
                    <span className="font-medium">{todayMetrics.respiratory_rate_bpm.toFixed(1)} bpm</span>
                  </div>
                )}
                
                {todayMetrics.skin_pallor_score !== null && todayMetrics.skin_pallor_score !== undefined && (
                  <div className="flex items-center justify-between p-3 rounded-md bg-muted/50">
                    <div className="flex items-center gap-2">
                      <Palette className="h-4 w-4 text-amber-500" />
                      <span className="text-sm text-muted-foreground">Skin Pallor</span>
                    </div>
                    <span className="font-medium">{todayMetrics.skin_pallor_score.toFixed(1)}/100</span>
                  </div>
                )}
                
                {todayMetrics.jaundice_risk_level && (
                  <div className="flex items-center justify-between p-3 rounded-md bg-muted/50">
                    <div className="flex items-center gap-2">
                      <Eye className="h-4 w-4 text-yellow-500" />
                      <span className="text-sm text-muted-foreground">Jaundice Risk</span>
                    </div>
                    <span className="font-medium capitalize">{todayMetrics.jaundice_risk_level}</span>
                  </div>
                )}
                
                {todayMetrics.facial_swelling_score !== null && todayMetrics.facial_swelling_score !== undefined && (
                  <div className="flex items-center justify-between p-3 rounded-md bg-muted/50">
                    <div className="flex items-center gap-2">
                      <Users className="h-4 w-4 text-rose-500" />
                      <span className="text-sm text-muted-foreground">Facial Swelling</span>
                    </div>
                    <span className="font-medium">{todayMetrics.facial_swelling_score.toFixed(1)}/100</span>
                  </div>
                )}
                
                {todayMetrics.head_stability_score !== null && todayMetrics.head_stability_score !== undefined && (
                  <div className="flex items-center justify-between p-3 rounded-md bg-muted/50">
                    <div className="flex items-center gap-2">
                      <Zap className="h-4 w-4 text-purple-500" />
                      <span className="text-sm text-muted-foreground">Tremor Detection</span>
                    </div>
                    <span className="font-medium">{todayMetrics.tremor_detected ? 'Detected' : 'None'}</span>
                  </div>
                )}
                
                {todayMetrics.tongue_color_index !== null && todayMetrics.tongue_color_index !== undefined && (
                  <div className="flex items-center justify-between p-3 rounded-md bg-muted/50">
                    <div className="flex items-center gap-2">
                      <Smile className="h-4 w-4 text-pink-500" />
                      <span className="text-sm text-muted-foreground">Tongue Analysis</span>
                    </div>
                    <span className="font-medium">{todayMetrics.tongue_coating_detected ? 'Coating' : 'Normal'}</span>
                  </div>
                )}
              </div>

              {/* Quality Metrics */}
              {(todayMetrics.lighting_quality_score || todayMetrics.frames_analyzed || todayMetrics.face_detection_confidence) && (
                <div className="rounded-md border bg-muted/30 p-3 space-y-2">
                  <p className="text-sm font-medium">Analysis Quality</p>
                  <div className="grid grid-cols-3 gap-3 text-sm">
                    {todayMetrics.lighting_quality_score && (
                      <div>
                        <span className="text-muted-foreground">Lighting: </span>
                        <span className="font-medium">{todayMetrics.lighting_quality_score.toFixed(0)}/100</span>
                      </div>
                    )}
                    {todayMetrics.frames_analyzed && (
                      <div>
                        <span className="text-muted-foreground">Frames: </span>
                        <span className="font-medium">{todayMetrics.frames_analyzed}</span>
                      </div>
                    )}
                    {todayMetrics.face_detection_confidence && (
                      <div>
                        <span className="text-muted-foreground">Confidence: </span>
                        <span className="font-medium">{(todayMetrics.face_detection_confidence * 100).toFixed(0)}%</span>
                      </div>
                    )}
                  </div>
                </div>
              )}
            </CardContent>
          </Card>

          {/* Advanced AI Analysis - Tremor, Gait, Edema */}
          {user?.id && (
            <TremorGaitEdemaInsights24h patientId={user.id} />
          )}

          {/* Option to do another examination */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Camera className="h-5 w-5 text-primary" />
                Record Another Examination
              </CardTitle>
              <CardDescription>
                Track changes throughout the day or document specific concerns
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Button
                onClick={startExam}
                disabled={isStarting}
                className="w-full gap-2"
                data-testid="button-start-exam"
              >
                {isStarting ? (
                  <>
                    <Loader2 className="h-5 w-5 animate-spin" />
                    Starting...
                  </>
                ) : (
                  <>
                    <Play className="h-5 w-5" />
                    Start New Examination
                  </>
                )}
              </Button>
            </CardContent>
          </Card>
        </div>
      ) : (
        // No metrics today - show start examination card
        <Card className="border-primary/20">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Camera className="h-6 w-6 text-primary" />
              Start Your Daily Examination
            </CardTitle>
            <CardDescription>
              Follow step-by-step instructions to record 7 different health examinations using your camera
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            <div className="grid gap-4 md:grid-cols-2">
              <div className="space-y-3">
                <h3 className="font-semibold flex items-center gap-2">
                  <Wind className="h-4 w-4 text-blue-500" />
                  What We'll Examine
                </h3>
                <ul className="space-y-2 text-sm text-muted-foreground">
                  <li className="flex items-start gap-2">
                    <CheckCircle2 className="h-4 w-4 text-chart-2 flex-shrink-0 mt-0.5" />
                    <span>Respiratory rate, movement & pattern</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <CheckCircle2 className="h-4 w-4 text-chart-2 flex-shrink-0 mt-0.5" />
                    <span>Skin pallor & color changes</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <CheckCircle2 className="h-4 w-4 text-chart-2 flex-shrink-0 mt-0.5" />
                    <span>Eye sclera yellowness (jaundice)</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <CheckCircle2 className="h-4 w-4 text-chart-2 flex-shrink-0 mt-0.5" />
                    <span>Facial & leg swelling detection</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <CheckCircle2 className="h-4 w-4 text-chart-2 flex-shrink-0 mt-0.5" />
                    <span>Tremor & involuntary movement</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <CheckCircle2 className="h-4 w-4 text-chart-2 flex-shrink-0 mt-0.5" />
                    <span>Tongue color & coating analysis</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <CheckCircle2 className="h-4 w-4 text-chart-2 flex-shrink-0 mt-0.5" />
                    <span>Custom abnormality documentation</span>
                  </li>
                </ul>
              </div>

              <div className="space-y-3">
                <h3 className="font-semibold flex items-center gap-2">
                  <Info className="h-4 w-4 text-primary" />
                  How It Works
                </h3>
                <ul className="space-y-2 text-sm text-muted-foreground">
                  <li className="flex items-start gap-2">
                    <span className="flex-shrink-0 w-6 h-6 rounded-full bg-primary/10 text-primary flex items-center justify-center text-xs font-medium">1</span>
                    <span>Read preparation instructions for each exam</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="flex-shrink-0 w-6 h-6 rounded-full bg-primary/10 text-primary flex items-center justify-center text-xs font-medium">2</span>
                    <span>30-second countdown to prepare (skippable)</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="flex-shrink-0 w-6 h-6 rounded-full bg-primary/10 text-primary flex items-center justify-center text-xs font-medium">3</span>
                    <span>Record live video using your camera</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="flex-shrink-0 w-6 h-6 rounded-full bg-primary/10 text-primary flex items-center justify-center text-xs font-medium">4</span>
                    <span>AI analyzes each recording automatically</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="flex-shrink-0 w-6 h-6 rounded-full bg-primary/10 text-primary flex items-center justify-center text-xs font-medium">5</span>
                    <span>View detailed insights and trends</span>
                  </li>
                </ul>
              </div>
            </div>

            <Button
              onClick={startExam}
              disabled={isStarting}
              size="lg"
              className="w-full"
              data-testid="button-start-exam"
            >
              {isStarting ? (
                <>
                  <Loader2 className="h-5 w-5 mr-2 animate-spin" />
                  Starting Examination...
                </>
              ) : (
                <>
                  <Play className="h-5 w-5 mr-2" />
                  Start Guided Examination
                </>
              )}
            </Button>

            <p className="text-xs text-center text-muted-foreground">
              You can skip any examination step. Camera access is required.
            </p>
          </CardContent>
        </Card>
      )}

          {/* Link to history page */}
          <Card>
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div className="space-y-1">
                  <h3 className="font-semibold">View All Health History</h3>
                  <p className="text-sm text-muted-foreground">
                    Access detailed trends across all wellness categories
                  </p>
                </div>
                <Link href="/daily-followup">
                  <Button variant="outline" className="gap-2" data-testid="button-health-history">
                    <BarChart3 className="h-4 w-4" />
                    Health History
                    <ExternalLink className="h-3 w-3" />
                  </Button>
                </Link>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="voice" className="mt-6">
          <div className="space-y-4">
            <Alert className="border-pink-500/50 bg-pink-500/5" data-testid="alert-voice-disclaimer">
              <Mic className="h-4 w-4 text-pink-500" />
              <AlertTitle className="text-pink-700 dark:text-pink-400 font-semibold">
                Voice-Based Health Check-in
              </AlertTitle>
              <AlertDescription className="text-pink-600 dark:text-pink-300 text-sm">
                Speak naturally about how you're feeling. Our AI will transcribe your message and extract 
                symptoms, vitals, and health information to track your wellness journey.
              </AlertDescription>
            </Alert>
            <VoiceFollowupTab />
          </div>
        </TabsContent>

        <TabsContent value="devices" className="mt-6">
          <DeviceDataTab patientId={user?.id || ''} />
        </TabsContent>

        <TabsContent value="deviations" className="mt-6">
          <div className="space-y-4">
            <Alert className="border-blue-500/50 bg-blue-500/5" data-testid="alert-deviation-disclaimer">
              <Info className="h-4 w-4 text-blue-500" />
              <AlertTitle className="text-blue-700 dark:text-blue-400 font-semibold">
                Wellness Pattern Tracking
              </AlertTitle>
              <AlertDescription className="text-blue-600 dark:text-blue-300 text-sm">
                Deviations show how your recent measurements compare to your personal baseline patterns.
                Changes from baseline are normal and not necessarily concerning. Consult your healthcare provider about any health questions.
              </AlertDescription>
            </Alert>
            <DeviationAnalysisTab />
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
}
