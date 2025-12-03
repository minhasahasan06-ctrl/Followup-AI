import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
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

      {/* Tabs for Examination and Device Data */}
      <Tabs defaultValue="examination" className="w-full">
        <TabsList className="grid w-full grid-cols-2 max-w-md">
          <TabsTrigger value="examination" className="gap-2" data-testid="tab-examination">
            <Video className="h-4 w-4" />
            Examinations
          </TabsTrigger>
          <TabsTrigger value="devices" className="gap-2" data-testid="tab-devices">
            <Watch className="h-4 w-4" />
            Device Data
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

        <TabsContent value="devices" className="mt-6">
          <DeviceDataTab patientId={user?.id || ''} />
        </TabsContent>
      </Tabs>
    </div>
  );
}
