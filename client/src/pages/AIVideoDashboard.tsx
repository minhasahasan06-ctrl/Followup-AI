import { useState, useEffect, useRef, useCallback } from 'react';
import { useQuery, useMutation } from '@tanstack/react-query';
import { Card, CardContent, CardHeader, CardTitle, CardDescription, CardFooter } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Label } from '@/components/ui/label';
import { Skeleton } from '@/components/ui/skeleton';
import {
  Video,
  Camera,
  Play,
  CheckCircle2,
  Clock,
  AlertCircle,
  Info,
  BarChart3,
  Calendar,
  Loader2,
  Wind,
  Hand,
  Eye,
  User,
  Activity,
  MessageSquare,
  XCircle,
  Vibrate,
  Footprints,
  Droplets,
  TrendingUp,
  TrendingDown,
  Minus,
  Upload,
  Target,
  Smartphone,
  AlertTriangle,
  FileVideo,
  History,
  RefreshCw,
} from 'lucide-react';
import { ExamPrepStep } from '@/components/ExamPrepStep';
import { VideoRecorder } from '@/components/VideoRecorder';
import { useGuidedExamWorkflow } from '@/hooks/useGuidedExamWorkflow';
import { useToast } from '@/hooks/use-toast';
import { useAuth } from '@/hooks/useAuth';
import { queryClient } from '@/lib/queryClient';
import { format } from 'date-fns';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Area,
  AreaChart,
} from 'recharts';

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

interface TremorAnalysis {
  patient_id: string;
  has_data: boolean;
  tremor_index?: number;
  tremor_detected?: boolean;
  dominant_frequency_hz?: number;
  tremor_amplitude_mg?: number;
  parkinsonian_likelihood?: number;
  essential_tremor_likelihood?: number;
  physiological_tremor?: boolean;
  created_at?: string;
}

interface TremorDashboard {
  patient_id: string;
  latest_tremor: TremorAnalysis | null;
  trend: {
    status: 'stable' | 'increasing' | 'decreasing' | 'insufficient_data';
    avg_tremor_index_7days: number;
    recordings_count_7days: number;
  };
  history_7days: Array<{
    tremor_index: number;
    tremor_detected: boolean;
    created_at: string;
  }>;
}

interface GaitSession {
  session_id: number;
  created_at: string;
  status: string;
  duration_seconds: number;
  total_strides: number;
  walking_detected: boolean;
  abnormality_detected: boolean;
  abnormality_score: number;
  quality_score: number;
}

interface GaitMetrics {
  session_id: number;
  patient_id: string;
  temporal: {
    stride_time_avg_sec: number;
    cadence_steps_per_min: number;
    walking_speed_m_per_sec: number;
  };
  symmetry_stability: {
    overall_symmetry: number;
    balance_confidence_score: number;
    stride_time_variability_percent: number;
  };
  clinical_risks: {
    fall_risk_score: number;
    parkinson_indicators: number;
    neuropathy_indicators: number;
    pain_indicators: number;
  };
}

interface EdemaMetrics {
  id: number;
  session_id: number;
  analyzed_at: string;
  swelling_detected: boolean;
  swelling_severity: string;
  overall_expansion_percent: number;
  regions_affected: number;
  regional_analysis: {
    face_upper_body: { swelling_detected: boolean; expansion_percent: number };
    legs_feet: { swelling_detected: boolean; expansion_percent: number };
    asymmetry_detected: boolean;
    asymmetry_difference: number;
  };
}

function TremorInsightsCard({ patientId }: { patientId: string }) {
  const { data: tremorDashboard, isLoading } = useQuery<TremorDashboard>({
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

  if (isLoading) {
    return (
      <Card data-testid="card-tremor-insights">
        <CardHeader className="pb-2">
          <CardTitle className="flex items-center gap-2 text-base">
            <Vibrate className="h-4 w-4 text-purple-500" />
            Tremor Analysis
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-2">
            <Skeleton className="h-8 w-full" />
            <Skeleton className="h-16 w-full" />
          </div>
        </CardContent>
      </Card>
    );
  }

  if (!tremorDashboard?.latest_tremor) {
    return (
      <Card data-testid="card-tremor-insights">
        <CardHeader className="pb-2">
          <CardTitle className="flex items-center gap-2 text-base">
            <Vibrate className="h-4 w-4 text-purple-500" />
            Tremor Analysis
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center py-4">
            <Smartphone className="h-8 w-8 mx-auto text-muted-foreground opacity-50 mb-2" />
            <p className="text-sm text-muted-foreground">No tremor data recorded yet</p>
            <p className="text-xs text-muted-foreground mt-1">
              Complete the tremor examination to see analysis
            </p>
          </div>
        </CardContent>
      </Card>
    );
  }

  const latest = tremorDashboard.latest_tremor;
  const trend = tremorDashboard.trend;

  const getTremorSeverity = (index: number) => {
    if (index < 2) return { label: 'Normal', color: 'text-green-500 bg-green-100 dark:bg-green-900/30' };
    if (index < 4) return { label: 'Mild', color: 'text-yellow-500 bg-yellow-100 dark:bg-yellow-900/30' };
    if (index < 6) return { label: 'Moderate', color: 'text-orange-500 bg-orange-100 dark:bg-orange-900/30' };
    return { label: 'Significant', color: 'text-red-500 bg-red-100 dark:bg-red-900/30' };
  };

  const severity = getTremorSeverity(latest.tremor_index || 0);

  return (
    <Card data-testid="card-tremor-insights">
      <CardHeader className="pb-2">
        <CardTitle className="flex items-center justify-between gap-2 text-base">
          <div className="flex items-center gap-2">
            <Vibrate className="h-4 w-4 text-purple-500" />
            Tremor Analysis
          </div>
          <Badge className={`text-xs ${severity.color}`}>
            {severity.label}
          </Badge>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="grid grid-cols-2 gap-3">
          <div className="p-2 rounded-lg bg-muted/50">
            <p className="text-xs text-muted-foreground">Tremor Index</p>
            <p className="text-lg font-bold" data-testid="text-tremor-index">
              {(latest.tremor_index || 0).toFixed(1)}/10
            </p>
          </div>
          <div className="p-2 rounded-lg bg-muted/50">
            <p className="text-xs text-muted-foreground">Frequency</p>
            <p className="text-lg font-bold" data-testid="text-tremor-frequency">
              {(latest.dominant_frequency_hz || 0).toFixed(1)} Hz
            </p>
          </div>
        </div>

        {(latest.parkinsonian_likelihood !== undefined || latest.essential_tremor_likelihood !== undefined) && (
          <div className="space-y-2 text-xs">
            <p className="font-medium text-muted-foreground">Pattern Indicators (Wellness Only)</p>
            <div className="grid grid-cols-2 gap-2">
              {latest.parkinsonian_likelihood !== undefined && (
                <div className="flex items-center justify-between p-1.5 rounded bg-muted/30">
                  <span>Resting tremor pattern</span>
                  <span className="font-medium">{(latest.parkinsonian_likelihood * 100).toFixed(0)}%</span>
                </div>
              )}
              {latest.essential_tremor_likelihood !== undefined && (
                <div className="flex items-center justify-between p-1.5 rounded bg-muted/30">
                  <span>Action tremor pattern</span>
                  <span className="font-medium">{(latest.essential_tremor_likelihood * 100).toFixed(0)}%</span>
                </div>
              )}
            </div>
          </div>
        )}

        {trend.status !== 'insufficient_data' && (
          <div className="flex items-center gap-2 text-xs">
            <span className="text-muted-foreground">7-day trend:</span>
            {trend.status === 'increasing' && (
              <Badge variant="outline" className="text-orange-500 border-orange-300">
                <TrendingUp className="h-3 w-3 mr-1" /> Increasing
              </Badge>
            )}
            {trend.status === 'decreasing' && (
              <Badge variant="outline" className="text-green-500 border-green-300">
                <TrendingDown className="h-3 w-3 mr-1" /> Decreasing
              </Badge>
            )}
            {trend.status === 'stable' && (
              <Badge variant="outline" className="text-blue-500 border-blue-300">
                <Minus className="h-3 w-3 mr-1" /> Stable
              </Badge>
            )}
            <span className="text-muted-foreground">({trend.recordings_count_7days} recordings)</span>
          </div>
        )}

        <p className="text-xs text-muted-foreground italic">
          Wellness monitoring indicator only - discuss changes with your healthcare provider
        </p>
      </CardContent>
    </Card>
  );
}

function GaitInsightsCard({ patientId }: { patientId: string }) {
  const { data: gaitSessions, isLoading } = useQuery<{ sessions: GaitSession[] }>({
    queryKey: ['/api/v1/gait-analysis/sessions', patientId],
    queryFn: async () => {
      const response = await fetch(`/api/v1/gait-analysis/sessions/${patientId}?limit=5`);
      if (!response.ok) {
        if (response.status === 404 || response.status === 502) return { sessions: [] };
        throw new Error('Failed to fetch gait data');
      }
      return response.json();
    },
    enabled: !!patientId,
  });

  const latestSession = gaitSessions?.sessions?.[0];

  const { data: gaitMetrics } = useQuery<GaitMetrics>({
    queryKey: ['/api/v1/gait-analysis/metrics', latestSession?.session_id],
    queryFn: async () => {
      const response = await fetch(`/api/v1/gait-analysis/metrics/${latestSession?.session_id}`);
      if (!response.ok) return null;
      return response.json();
    },
    enabled: !!latestSession?.session_id && latestSession?.status === 'completed',
  });

  if (isLoading) {
    return (
      <Card data-testid="card-gait-insights">
        <CardHeader className="pb-2">
          <CardTitle className="flex items-center gap-2 text-base">
            <Footprints className="h-4 w-4 text-blue-500" />
            Gait Analysis
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-2">
            <Skeleton className="h-8 w-full" />
            <Skeleton className="h-16 w-full" />
          </div>
        </CardContent>
      </Card>
    );
  }

  if (!latestSession || latestSession.status !== 'completed') {
    return (
      <Card data-testid="card-gait-insights">
        <CardHeader className="pb-2">
          <CardTitle className="flex items-center gap-2 text-base">
            <Footprints className="h-4 w-4 text-blue-500" />
            Gait Analysis
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center py-4">
            <FileVideo className="h-8 w-8 mx-auto text-muted-foreground opacity-50 mb-2" />
            <p className="text-sm text-muted-foreground">No gait analysis available</p>
            <p className="text-xs text-muted-foreground mt-1">
              Upload a walking video to analyze your gait patterns
            </p>
          </div>
        </CardContent>
      </Card>
    );
  }

  const getFallRiskLevel = (score: number) => {
    if (score < 0.3) return { label: 'Low Risk', color: 'text-green-500 bg-green-100 dark:bg-green-900/30' };
    if (score < 0.6) return { label: 'Moderate', color: 'text-yellow-500 bg-yellow-100 dark:bg-yellow-900/30' };
    return { label: 'Elevated', color: 'text-red-500 bg-red-100 dark:bg-red-900/30' };
  };

  const fallRisk = getFallRiskLevel(gaitMetrics?.clinical_risks?.fall_risk_score || latestSession.abnormality_score || 0);

  return (
    <Card data-testid="card-gait-insights">
      <CardHeader className="pb-2">
        <CardTitle className="flex items-center justify-between gap-2 text-base">
          <div className="flex items-center gap-2">
            <Footprints className="h-4 w-4 text-blue-500" />
            Gait Analysis
          </div>
          <Badge className={`text-xs ${fallRisk.color}`}>
            {fallRisk.label}
          </Badge>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="grid grid-cols-3 gap-2">
          <div className="p-2 rounded-lg bg-muted/50 text-center">
            <p className="text-xs text-muted-foreground">Strides</p>
            <p className="text-lg font-bold" data-testid="text-gait-strides">
              {latestSession.total_strides || 0}
            </p>
          </div>
          <div className="p-2 rounded-lg bg-muted/50 text-center">
            <p className="text-xs text-muted-foreground">Cadence</p>
            <p className="text-lg font-bold" data-testid="text-gait-cadence">
              {gaitMetrics?.temporal?.cadence_steps_per_min?.toFixed(0) || '—'}
            </p>
            <p className="text-xs text-muted-foreground">steps/min</p>
          </div>
          <div className="p-2 rounded-lg bg-muted/50 text-center">
            <p className="text-xs text-muted-foreground">Symmetry</p>
            <p className="text-lg font-bold" data-testid="text-gait-symmetry">
              {gaitMetrics?.symmetry_stability?.overall_symmetry 
                ? `${(gaitMetrics.symmetry_stability.overall_symmetry * 100).toFixed(0)}%` 
                : '—'}
            </p>
          </div>
        </div>

        {gaitMetrics?.clinical_risks && (
          <div className="space-y-2 text-xs">
            <p className="font-medium text-muted-foreground">Clinical Indicators (Wellness Only)</p>
            <div className="grid grid-cols-2 gap-2">
              <div className="flex items-center justify-between p-1.5 rounded bg-muted/30">
                <span>Fall risk indicator</span>
                <span className="font-medium">{(gaitMetrics.clinical_risks.fall_risk_score * 100).toFixed(0)}%</span>
              </div>
              <div className="flex items-center justify-between p-1.5 rounded bg-muted/30">
                <span>Balance confidence</span>
                <span className="font-medium">{((gaitMetrics.symmetry_stability?.balance_confidence_score || 0) * 100).toFixed(0)}%</span>
              </div>
            </div>
          </div>
        )}

        <p className="text-xs text-muted-foreground italic">
          Last analyzed: {format(new Date(latestSession.created_at), 'MMM d, h:mm a')}
        </p>
      </CardContent>
    </Card>
  );
}

function EdemaInsightsCard({ patientId }: { patientId: string }) {
  const { data: edemaMetrics, isLoading } = useQuery<EdemaMetrics[]>({
    queryKey: ['/api/v1/edema/metrics', patientId],
    queryFn: async () => {
      const response = await fetch(`/api/v1/edema/metrics/${patientId}?limit=5`);
      if (!response.ok) {
        if (response.status === 404 || response.status === 502) return [];
        throw new Error('Failed to fetch edema data');
      }
      return response.json();
    },
    enabled: !!patientId,
  });

  if (isLoading) {
    return (
      <Card data-testid="card-edema-insights">
        <CardHeader className="pb-2">
          <CardTitle className="flex items-center gap-2 text-base">
            <Droplets className="h-4 w-4 text-cyan-500" />
            Edema Analysis
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-2">
            <Skeleton className="h-8 w-full" />
            <Skeleton className="h-16 w-full" />
          </div>
        </CardContent>
      </Card>
    );
  }

  const latest = edemaMetrics?.[0];

  if (!latest) {
    return (
      <Card data-testid="card-edema-insights">
        <CardHeader className="pb-2">
          <CardTitle className="flex items-center gap-2 text-base">
            <Droplets className="h-4 w-4 text-cyan-500" />
            Edema Analysis
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center py-4">
            <Target className="h-8 w-8 mx-auto text-muted-foreground opacity-50 mb-2" />
            <p className="text-sm text-muted-foreground">No edema analysis available</p>
            <p className="text-xs text-muted-foreground mt-1">
              Complete the swelling examination to see analysis
            </p>
          </div>
        </CardContent>
      </Card>
    );
  }

  const getSeverityBadge = (severity: string) => {
    switch (severity?.toLowerCase()) {
      case 'none':
        return { label: 'None Detected', color: 'text-green-500 bg-green-100 dark:bg-green-900/30' };
      case 'mild':
        return { label: 'Mild', color: 'text-yellow-500 bg-yellow-100 dark:bg-yellow-900/30' };
      case 'moderate':
        return { label: 'Moderate', color: 'text-orange-500 bg-orange-100 dark:bg-orange-900/30' };
      case 'severe':
        return { label: 'Significant', color: 'text-red-500 bg-red-100 dark:bg-red-900/30' };
      default:
        return { label: latest.swelling_detected ? 'Detected' : 'None', color: latest.swelling_detected ? 'text-yellow-500 bg-yellow-100' : 'text-green-500 bg-green-100' };
    }
  };

  const severityBadge = getSeverityBadge(latest.swelling_severity);

  return (
    <Card data-testid="card-edema-insights">
      <CardHeader className="pb-2">
        <CardTitle className="flex items-center justify-between gap-2 text-base">
          <div className="flex items-center gap-2">
            <Droplets className="h-4 w-4 text-cyan-500" />
            Edema Analysis
          </div>
          <Badge className={`text-xs ${severityBadge.color}`}>
            {severityBadge.label}
          </Badge>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="grid grid-cols-2 gap-3">
          <div className="p-2 rounded-lg bg-muted/50">
            <p className="text-xs text-muted-foreground">Overall Expansion</p>
            <p className="text-lg font-bold" data-testid="text-edema-expansion">
              {latest.overall_expansion_percent?.toFixed(1) || 0}%
            </p>
          </div>
          <div className="p-2 rounded-lg bg-muted/50">
            <p className="text-xs text-muted-foreground">Regions Affected</p>
            <p className="text-lg font-bold" data-testid="text-edema-regions">
              {latest.regions_affected || 0}
            </p>
          </div>
        </div>

        {latest.regional_analysis && (
          <div className="space-y-2 text-xs">
            <p className="font-medium text-muted-foreground">Regional Breakdown</p>
            <div className="space-y-1">
              <div className="flex items-center justify-between p-1.5 rounded bg-muted/30">
                <span className="flex items-center gap-1">
                  <User className="h-3 w-3" /> Face/Upper Body
                </span>
                <span className={`font-medium ${latest.regional_analysis.face_upper_body?.swelling_detected ? 'text-orange-500' : 'text-green-500'}`}>
                  {latest.regional_analysis.face_upper_body?.swelling_detected ? 'Detected' : 'Normal'}
                </span>
              </div>
              <div className="flex items-center justify-between p-1.5 rounded bg-muted/30">
                <span className="flex items-center gap-1">
                  <Footprints className="h-3 w-3" /> Legs/Feet
                </span>
                <span className={`font-medium ${latest.regional_analysis.legs_feet?.swelling_detected ? 'text-orange-500' : 'text-green-500'}`}>
                  {latest.regional_analysis.legs_feet?.swelling_detected ? 'Detected' : 'Normal'}
                </span>
              </div>
              {latest.regional_analysis.asymmetry_detected && (
                <div className="flex items-center gap-1 p-1.5 rounded bg-amber-100 dark:bg-amber-900/30 text-amber-700 dark:text-amber-400">
                  <AlertTriangle className="h-3 w-3" />
                  <span>Asymmetry detected ({latest.regional_analysis.asymmetry_difference?.toFixed(1)}% difference)</span>
                </div>
              )}
            </div>
          </div>
        )}

        <p className="text-xs text-muted-foreground italic">
          Last analyzed: {format(new Date(latest.analyzed_at), 'MMM d, h:mm a')}
        </p>
      </CardContent>
    </Card>
  );
}

function VideoAIInsightsPanel({ patientId }: { patientId: string }) {
  return (
    <div className="grid gap-4 md:grid-cols-3" data-testid="panel-video-ai-insights">
      <TremorInsightsCard patientId={patientId} />
      <GaitInsightsCard patientId={patientId} />
      <EdemaInsightsCard patientId={patientId} />
    </div>
  );
}

export default function AIVideoDashboard() {
  const { user } = useAuth();
  const patientId = user?.id || 'demo-patient';
  
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

  // Fetch recent sessions (last 15 days)
  const { data: recentSessions } = useQuery({
    queryKey: ['/api/v1/video-ai/exam-sessions', { days: 15 }],
    enabled: workflowState === 'idle' || workflowState === 'completed',
  });

  // Fetch AI analysis insights
  const { data: analysisInsights } = useQuery({
    queryKey: ['/api/v1/video-ai/analysis/insights', { sessionId }],
    enabled: !!sessionId && workflowState === 'completed',
  });

  const handleRecordingComplete = (videoBlob: Blob, durationSeconds: number) => {
    if (currentStep?.examType === 'custom') {
      completeRecording(videoBlob, durationSeconds, {
        location: customLocation,
        description: customDescription,
      });
      // Reset custom fields
      setCustomLocation('');
      setCustomDescription('');
    } else {
      completeRecording(videoBlob, durationSeconds);
    }
  };

  // Idle state - show welcome screen
  if (workflowState === 'idle') {
    return (
      <div className="container mx-auto p-6 max-w-7xl space-y-6">
        {/* Header */}
        <div className="space-y-2">
          <h1 className="text-4xl font-bold tracking-tight text-foreground flex items-center gap-3" data-testid="text-page-title">
            <Video className="h-10 w-10 text-primary" />
            Guided Video Examination
          </h1>
          <p className="text-muted-foreground leading-relaxed">
            AI-powered guided examination for comprehensive health monitoring
          </p>
        </div>

        <LegalDisclaimer />

        {/* Welcome Card */}
        <Card className="border-primary/20">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Camera className="h-6 w-6 text-primary" />
              Start Your Guided Examination
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
                    <span>Skin pallor & color changes (palms/soles/nails/nail beds) for anaemia, nicotine stains, burns, or any other abnormality</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <CheckCircle2 className="h-4 w-4 text-chart-2 flex-shrink-0 mt-0.5" />
                    <span>Eye sclera yellowness (jaundice indicator)</span>
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

        {/* Recent Sessions & Analysis - Bottom Panels */}
        <Tabs defaultValue="recent" className="w-full">
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="recent">
              <Calendar className="h-4 w-4 mr-2" />
              Last 15 Days
            </TabsTrigger>
            <TabsTrigger value="insights">
              <BarChart3 className="h-4 w-4 mr-2" />
              AI Analysis
            </TabsTrigger>
          </TabsList>

          <TabsContent value="recent" className="space-y-3 mt-4">
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Recent Examination Sessions</CardTitle>
                <CardDescription>Your video recordings from the past 15 days</CardDescription>
              </CardHeader>
              <CardContent>
                {recentSessions && recentSessions.length > 0 ? (
                  <div className="space-y-2">
                    {recentSessions.map((session: any) => (
                      <div
                        key={session.id}
                        className="flex items-center justify-between p-3 rounded-lg border bg-card hover-elevate"
                      >
                        <div className="flex items-center gap-3">
                          <Video className="h-5 w-5 text-muted-foreground" />
                          <div>
                            <div className="font-medium text-sm">
                              {new Date(session.started_at).toLocaleDateString()}
                            </div>
                            <div className="text-xs text-muted-foreground">
                              {session.completed_segments} of {session.total_segments} exams completed
                            </div>
                          </div>
                        </div>
                        <Badge variant={session.status === 'completed' ? 'default' : 'secondary'}>
                          {session.status}
                        </Badge>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="text-center py-12">
                    <Video className="h-16 w-16 mx-auto text-muted-foreground opacity-50 mb-4" />
                    <h3 className="text-lg font-semibold mb-2">No Sessions Yet</h3>
                    <p className="text-muted-foreground max-w-md mx-auto">
                      Start your first guided examination to begin tracking your health patterns
                    </p>
                  </div>
                )}
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="insights" className="space-y-4 mt-4">
            <div className="space-y-2">
              <h3 className="text-lg font-semibold flex items-center gap-2">
                <BarChart3 className="h-5 w-5 text-primary" />
                AI Video Analysis Insights
              </h3>
              <p className="text-sm text-muted-foreground">
                Tremor, Gait, and Edema analysis from your examinations
              </p>
            </div>
            <VideoAIInsightsPanel patientId={patientId} />
            <Alert className="bg-blue-50 dark:bg-blue-900/20 border-blue-200">
              <Info className="h-4 w-4 text-blue-500" />
              <AlertDescription className="text-xs">
                These insights are for wellness monitoring only and are not medical diagnoses. 
                Always discuss any concerns with your healthcare provider.
              </AlertDescription>
            </Alert>
          </TabsContent>
        </Tabs>
      </div>
    );
  }

  // Active examination workflow
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
          {/* Custom exam input */}
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
        <div className="space-y-6">
          <Card className="border-chart-2">
            <CardContent className="p-8 text-center">
              <CheckCircle2 className="h-12 w-12 mx-auto mb-4 text-chart-2" />
              <h2 className="text-2xl font-bold mb-2">Examination Complete!</h2>
              <p className="text-muted-foreground mb-4">
                All recordings have been processed. Your AI analysis is ready.
              </p>
              <Button onClick={resetWorkflow} size="lg" data-testid="button-start-new">
                Start New Examination
              </Button>
            </CardContent>
          </Card>

          <div className="space-y-3">
            <h3 className="text-lg font-semibold flex items-center gap-2">
              <BarChart3 className="h-5 w-5 text-primary" />
              Your AI Analysis Results
            </h3>
            <p className="text-sm text-muted-foreground">
              Tremor, Gait, and Edema analysis from your examination
            </p>
          </div>
          
          <VideoAIInsightsPanel patientId={patientId} />

          <Alert className="bg-blue-50 dark:bg-blue-900/20 border-blue-200">
            <Info className="h-4 w-4 text-blue-500" />
            <AlertDescription className="text-xs">
              These insights are for wellness monitoring only and are not medical diagnoses. 
              Always discuss any changes with your healthcare provider.
            </AlertDescription>
          </Alert>
        </div>
      )}
    </div>
  );
}
