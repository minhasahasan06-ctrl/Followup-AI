import { useState, useMemo } from "react";
import { Link } from "wouter";
import { useQuery, useMutation } from "@tanstack/react-query";
import { Card, CardContent, CardDescription, CardHeader, CardTitle, CardFooter } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Progress } from "@/components/ui/progress";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { useToast } from "@/hooks/use-toast";
import { apiRequest, queryClient } from "@/lib/queryClient";
import { useAuth } from "@/hooks/useAuth";
import { DetailedPredictionCard } from "@/components/DetailedPredictionCard";
import {
  Bell,
  CheckCircle2,
  XCircle,
  Clock,
  AlertTriangle,
  TrendingUp,
  TrendingDown,
  Minus,
  Activity,
  Heart,
  Brain,
  Zap,
  LineChart,
  BarChart3,
  AlertCircle,
  RefreshCw,
  ChevronRight,
  Info,
  Shield,
  Sparkles,
  Target,
  Calendar,
  Wind,
  Droplets,
  Footprints,
  Gauge,
  History,
  Moon,
  Link2
} from "lucide-react";
import { LegalDisclaimer } from "@/components/LegalDisclaimer";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import {
  LineChart as ReLineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
  Area,
  AreaChart,
  Bar,
  BarChart
} from "recharts";
import { format } from "date-fns";

interface TrendMetric {
  id: string;
  patient_id: string;
  metric_name: string;
  metric_category: string;
  raw_value: number;
  baseline_14d_mean: number | null;
  baseline_14d_std: number | null;
  z_score: number | null;
  z_score_severity: string | null;
  slope_3d: number | null;
  slope_7d: number | null;
  slope_14d: number | null;
  slope_direction: string | null;
  volatility_index: number | null;
  volatility_level: string | null;
  composite_trend_score: number | null;
  recorded_at: string;
  computed_at: string;
}

interface EngagementMetric {
  id: string;
  patient_id: string;
  period_start: string;
  period_end: string;
  adherence_score: number;
  checkins_completed: number;
  checkins_expected: number;
  captures_completed: number;
  surveys_completed: number;
  engagement_score: number;
  engagement_trend: string;
  engagement_drop_14d: number | null;
  current_streak: number;
  longest_streak: number;
  alerts_generated: number;
  alerts_dismissed: number;
  computed_at: string;
}

interface QolMetric {
  id: string;
  patient_id: string;
  wellness_index: number;
  wellness_components: Record<string, number> | null;
  wellness_trend: string;
  functional_status: number;
  functional_components: Record<string, any> | null;
  selfcare_score: number;
  selfcare_components: Record<string, number> | null;
  stability_score: number;
  behavior_patterns: Record<string, number> | null;
  recorded_at: string;
  computed_at: string;
}

interface HealthAlert {
  id: string;
  patient_id: string;
  alert_type: string;
  alert_category: string;
  severity: string;
  priority: number;
  escalation_probability: number | null;
  title: string;
  message: string;
  disclaimer: string;
  contributing_metrics: any[] | null;
  trigger_rule: string | null;
  status: string;
  acknowledged_by: string | null;
  acknowledged_at: string | null;
  dismissed_by: string | null;
  dismissed_at: string | null;
  dismiss_reason: string | null;
  clinician_notes: string | null;
  created_at: string;
}

interface PatientOverview {
  patient_id: string;
  trend_metrics: any[];
  engagement: {
    adherence_score: number | null;
    engagement_score: number | null;
    trend: string | null;
    current_streak: number;
    checkins_completed: number;
    computed_at: string | null;
  } | null;
  quality_of_life: {
    wellness_index: number | null;
    wellness_trend: string | null;
    functional_status: number | null;
    selfcare_score: number | null;
    stability_score: number | null;
    behavior_patterns: Record<string, number> | null;
    recorded_at: string | null;
  } | null;
  active_alerts: any[];
  total_active_alerts: number;
}

interface OrganScore {
  organ_group: string;
  organ_name: string;
  normalized_score: number;
  color_bucket: string;
  contributing_metrics: Array<{
    metric_name: string;
    z_score: number;
    weight: number;
    contribution: number;
  }>;
  confidence: number;
}

interface OrganScoreResult {
  patient_id: string;
  computed_at: string;
  organ_scores: Record<string, OrganScore>;
  overall_status: string;
}

interface DPIResult {
  patient_id: string;
  computed_at: string;
  dpi_raw: number;
  dpi_normalized: number;
  dpi_bucket: string;
  bucket_changed: boolean;
  previous_bucket: string | null;
  organ_contributions: Record<string, number>;
  explainability: {
    top_contributors: string[];
    trend_direction: string;
    confidence: number;
  };
}

interface MLPrediction {
  patient_id: string;
  ensemble_score: number;
  ensemble_confidence: number;
  statistical_weight: number;
  ml_weight: number;
  trend_direction: string;
  risk_trajectory: string;
  predictions: Record<string, {
    horizon: string;
    deterioration_probability: number;
    confidence: number;
    risk_level: string;
    feature_importance?: Record<string, number>;
  }>;
  computed_at: string;
  model_version: string;
  disclaimer: string;
  status?: string;
}

interface RiskFactor {
  metric_name: string;
  z_score: number;
  severity_level: string;
  points: number;
  description: string;
}

interface RiskScoreData {
  patient_id: string;
  score: number;
  level: string;
  calculated_at: string;
  factors: RiskFactor[];
  total_deviations: number;
  critical_deviations: number;
  moderate_deviations: number;
  recommendation: string;
  action_items: string[];
}

interface RiskHistoryEntry {
  date: string;
  score: number;
  level: string;
  deviation_count: number;
}

interface BaselineStats {
  mean: number | null;
  std: number | null;
  min_value: number | null;
  max_value: number | null;
}

interface BaselineData {
  id: number;
  patient_id: string;
  baseline_start_date: string;
  baseline_end_date: string;
  data_points_count: number;
  pain_facial: BaselineStats;
  pain_self_reported: BaselineStats;
  respiratory_rate: BaselineStats;
  symptom_severity: BaselineStats;
  activity_impact_rate: number | null;
  baseline_quality: string | null;
  is_current: boolean;
  created_at: string;
}

interface DiseaseRiskPrediction {
  disease: string;
  probability: number;
  risk_level: string;
  confidence: number;
  confidence_interval?: { lower: number; upper: number };
  contributing_factors: Array<{
    feature: string;
    value: number;
    contribution: number;
    direction: string;
    baseline?: number;
    normal_range?: { min: number; max: number };
  }>;
  recommendations: string[];
  time_projections?: {
    "24h": number;
    "48h": number;
    "72h": number;
  };
}

interface MLDiseaseRiskResponse {
  patient_id: string;
  predictions?: Record<string, DiseaseRiskPrediction>;
  predicted_at: string;
  model_version: string;
}

interface DeviceHealthSection {
  deterioration_index: number;
  risk_score: number;
  risk_level: string;
  trend: string;
  stability_score: number;
  data_coverage: number;
  anomalies_detected: number;
  alert_triggered: boolean;
}

interface DeviceAnalyticsData {
  patient_id: string;
  generated_at: string;
  overall_risk_score: number;
  overall_trend: string;
  critical_alerts: string[];
  recommendations: string[];
  sections: Record<string, DeviceHealthSection>;
}

const HEALTH_SECTION_CONFIG: Record<string, { 
  label: string; 
  icon: typeof Heart; 
  color: string;
  description: string;
}> = {
  cardiovascular: { 
    label: "Cardiovascular", 
    icon: Heart, 
    color: "text-rose-500",
    description: "Heart rate, blood pressure, HRV"
  },
  respiratory: { 
    label: "Respiratory", 
    icon: Wind, 
    color: "text-blue-500",
    description: "SpO2, respiratory rate, breathing patterns"
  },
  metabolic: { 
    label: "Metabolic", 
    icon: Droplets, 
    color: "text-amber-500",
    description: "Blood glucose, temperature, hydration"
  },
  activity: { 
    label: "Activity", 
    icon: Footprints, 
    color: "text-green-500",
    description: "Steps, active minutes, exercise"
  },
  sleep: { 
    label: "Sleep", 
    icon: Moon, 
    color: "text-purple-500",
    description: "Sleep quality, duration, stages"
  },
  body_composition: { 
    label: "Body Composition", 
    icon: Gauge, 
    color: "text-cyan-500",
    description: "Weight, BMI, body fat"
  },
};

const CHART_COLORS = {
  primary: 'hsl(var(--primary))',
  secondary: 'hsl(var(--chart-2))',
  tertiary: 'hsl(var(--chart-3))',
  quaternary: 'hsl(var(--chart-4))',
  quinary: 'hsl(var(--chart-5))',
};

function StatCard({ 
  title, 
  value, 
  subtitle, 
  trend, 
  icon: Icon, 
  color = "text-primary",
  testId 
}: { 
  title: string; 
  value: string | number; 
  subtitle?: string; 
  trend?: 'up' | 'down' | 'stable'; 
  icon: any;
  color?: string;
  testId?: string;
}) {
  const TrendIcon = trend === 'up' ? TrendingUp : trend === 'down' ? TrendingDown : Minus;
  const trendColor = trend === 'up' ? 'text-green-500' : trend === 'down' ? 'text-red-500' : 'text-muted-foreground';
  
  return (
    <Card data-testid={testId}>
      <CardContent className="p-4">
        <div className="flex items-center justify-between">
          <div className="space-y-1">
            <p className="text-sm text-muted-foreground">{title}</p>
            <p className="text-2xl font-bold">{value}</p>
            {subtitle && (
              <div className="flex items-center gap-1">
                {trend && <TrendIcon className={`h-3 w-3 ${trendColor}`} />}
                <p className="text-xs text-muted-foreground">{subtitle}</p>
              </div>
            )}
          </div>
          <div className={`p-2 rounded-full bg-muted ${color}`}>
            <Icon className="h-5 w-5" />
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

function AlertCard({ 
  alert, 
  onAcknowledge, 
  onDismiss, 
  isUpdating 
}: { 
  alert: HealthAlert; 
  onAcknowledge: (id: string) => void;
  onDismiss: (id: string, reason: string, notes?: string) => void;
  isUpdating: boolean;
}) {
  const [dismissDialogOpen, setDismissDialogOpen] = useState(false);
  const [dismissReason, setDismissReason] = useState("");
  const [clinicianNotes, setClinicianNotes] = useState("");

  const getSeverityStyles = (severity: string) => {
    switch (severity) {
      case "critical":
        return "border-l-4 border-l-rose-500 bg-rose-50/50 dark:bg-rose-950/20";
      case "high":
        return "border-l-4 border-l-orange-500 bg-orange-50/50 dark:bg-orange-950/20";
      case "moderate":
        return "border-l-4 border-l-yellow-500 bg-yellow-50/50 dark:bg-yellow-950/20";
      default:
        return "border-l-4 border-l-blue-500 bg-blue-50/50 dark:bg-blue-950/20";
    }
  };

  const getSeverityBadge = (severity: string) => {
    const colors: Record<string, string> = {
      critical: "bg-rose-100 text-rose-700 dark:bg-rose-900 dark:text-rose-300",
      high: "bg-orange-100 text-orange-700 dark:bg-orange-900 dark:text-orange-300",
      moderate: "bg-yellow-100 text-yellow-700 dark:bg-yellow-900 dark:text-yellow-300",
      low: "bg-blue-100 text-blue-700 dark:bg-blue-900 dark:text-blue-300",
    };
    return colors[severity] || colors.low;
  };

  const handleDismiss = () => {
    onDismiss(alert.id, dismissReason, clinicianNotes);
    setDismissDialogOpen(false);
    setDismissReason("");
    setClinicianNotes("");
  };

  return (
    <Card className={getSeverityStyles(alert.severity)} data-testid={`alert-card-${alert.id}`}>
      <CardContent className="p-4">
        <div className="flex flex-col gap-3">
          <div className="flex items-start justify-between gap-4">
            <div className="flex-1 space-y-2">
              <div className="flex items-center gap-2 flex-wrap">
                <Badge className={getSeverityBadge(alert.severity)}>
                  <AlertTriangle className="h-3 w-3 mr-1" />
                  {alert.severity.toUpperCase()}
                </Badge>
                <Badge variant="outline" className="text-xs">
                  {alert.alert_type}
                </Badge>
                {alert.escalation_probability && alert.escalation_probability > 0.5 && (
                  <Badge variant="secondary" className="text-xs">
                    <Zap className="h-3 w-3 mr-1" />
                    {(alert.escalation_probability * 100).toFixed(0)}% escalation risk
                  </Badge>
                )}
              </div>
              <h4 className="font-semibold">{alert.title}</h4>
              <p className="text-sm text-muted-foreground">{alert.message}</p>
              <p className="text-xs text-muted-foreground/70 italic">{alert.disclaimer}</p>
              <div className="flex items-center gap-3 text-xs text-muted-foreground">
                <div className="flex items-center gap-1">
                  <Clock className="h-3 w-3" />
                  {format(new Date(alert.created_at), "MMM d, h:mm a")}
                </div>
                {alert.trigger_rule && (
                  <Badge variant="outline" className="text-xs">
                    Rule: {alert.trigger_rule}
                  </Badge>
                )}
              </div>
            </div>
          </div>
          
          <div className="flex items-center gap-2 justify-end flex-wrap">
            {alert.status === "new" && (
              <Button
                size="sm"
                variant="outline"
                onClick={() => onAcknowledge(alert.id)}
                disabled={isUpdating}
                data-testid={`button-acknowledge-${alert.id}`}
              >
                <CheckCircle2 className="h-4 w-4 mr-2" />
                Acknowledge
              </Button>
            )}
            
            <Dialog open={dismissDialogOpen} onOpenChange={setDismissDialogOpen}>
              <DialogTrigger asChild>
                <Button
                  size="sm"
                  variant="ghost"
                  disabled={isUpdating}
                  data-testid={`button-dismiss-${alert.id}`}
                >
                  <XCircle className="h-4 w-4 mr-2" />
                  Dismiss
                </Button>
              </DialogTrigger>
              <DialogContent>
                <DialogHeader>
                  <DialogTitle>Dismiss Alert</DialogTitle>
                  <DialogDescription>
                    Please provide a reason for dismissing this alert. This helps improve future alert accuracy.
                  </DialogDescription>
                </DialogHeader>
                <div className="space-y-4 py-4">
                  <div className="space-y-2">
                    <Label htmlFor="dismiss-reason">Reason for Dismissal</Label>
                    <Textarea
                      id="dismiss-reason"
                      placeholder="e.g., False positive, Patient contacted, Already addressed..."
                      value={dismissReason}
                      onChange={(e) => setDismissReason(e.target.value)}
                      data-testid="input-dismiss-reason"
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="clinician-notes">Notes (Optional)</Label>
                    <Textarea
                      id="clinician-notes"
                      placeholder="Additional notes..."
                      value={clinicianNotes}
                      onChange={(e) => setClinicianNotes(e.target.value)}
                      data-testid="input-clinician-notes"
                    />
                  </div>
                </div>
                <DialogFooter>
                  <Button variant="outline" onClick={() => setDismissDialogOpen(false)}>
                    Cancel
                  </Button>
                  <Button onClick={handleDismiss} disabled={!dismissReason.trim()}>
                    Dismiss Alert
                  </Button>
                </DialogFooter>
              </DialogContent>
            </Dialog>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

function TrendMetricsChart({ metrics }: { metrics: TrendMetric[] }) {
  const chartData = useMemo(() => {
    if (!metrics?.length) return [];
    
    const grouped: Record<string, any> = {};
    metrics.forEach(m => {
      const date = format(new Date(m.recorded_at), "MMM d");
      if (!grouped[date]) {
        grouped[date] = { date };
      }
      grouped[date][m.metric_name] = m.raw_value;
      grouped[date][`${m.metric_name}_z`] = m.z_score;
    });
    
    return Object.values(grouped).reverse();
  }, [metrics]);

  if (!chartData.length) {
    return (
      <div className="text-center py-8">
        <LineChart className="h-12 w-12 mx-auto text-muted-foreground opacity-50 mb-2" />
        <p className="text-sm text-muted-foreground">No trend data available</p>
      </div>
    );
  }

  return (
    <ResponsiveContainer width="100%" height={250}>
      <AreaChart data={chartData}>
        <defs>
          <linearGradient id="colorPain" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor={CHART_COLORS.primary} stopOpacity={0.3}/>
            <stop offset="95%" stopColor={CHART_COLORS.primary} stopOpacity={0}/>
          </linearGradient>
          <linearGradient id="colorFatigue" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor={CHART_COLORS.secondary} stopOpacity={0.3}/>
            <stop offset="95%" stopColor={CHART_COLORS.secondary} stopOpacity={0}/>
          </linearGradient>
        </defs>
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
        <Area 
          type="monotone" 
          dataKey="pain_level" 
          name="Pain Level" 
          stroke={CHART_COLORS.primary} 
          fill="url(#colorPain)" 
        />
        <Area 
          type="monotone" 
          dataKey="fatigue_level" 
          name="Fatigue Level" 
          stroke={CHART_COLORS.secondary} 
          fill="url(#colorFatigue)" 
        />
        <Area 
          type="monotone" 
          dataKey="mood" 
          name="Mood" 
          stroke={CHART_COLORS.tertiary} 
          fillOpacity={0.3}
          fill={CHART_COLORS.tertiary}
        />
      </AreaChart>
    </ResponsiveContainer>
  );
}

function QoLChart({ qolMetrics }: { qolMetrics: QolMetric[] }) {
  const chartData = useMemo(() => {
    if (!qolMetrics?.length) return [];
    
    return qolMetrics.slice(0, 14).reverse().map(q => ({
      date: format(new Date(q.recorded_at), "MMM d"),
      wellness: q.wellness_index,
      functional: q.functional_status,
      selfcare: q.selfcare_score,
      stability: q.stability_score
    }));
  }, [qolMetrics]);

  if (!chartData.length) {
    return (
      <div className="text-center py-8">
        <Heart className="h-12 w-12 mx-auto text-muted-foreground opacity-50 mb-2" />
        <p className="text-sm text-muted-foreground">No quality of life data available</p>
      </div>
    );
  }

  return (
    <ResponsiveContainer width="100%" height={250}>
      <ReLineChart data={chartData}>
        <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
        <XAxis dataKey="date" className="text-xs" />
        <YAxis domain={[0, 100]} className="text-xs" />
        <Tooltip 
          contentStyle={{ 
            backgroundColor: 'hsl(var(--card))', 
            border: '1px solid hsl(var(--border))',
            borderRadius: '8px'
          }} 
        />
        <Legend />
        <Line type="monotone" dataKey="wellness" name="Wellness Index" stroke={CHART_COLORS.primary} strokeWidth={2} dot={false} />
        <Line type="monotone" dataKey="functional" name="Functional Status" stroke={CHART_COLORS.secondary} strokeWidth={2} dot={false} />
        <Line type="monotone" dataKey="selfcare" name="Self-Care Score" stroke={CHART_COLORS.tertiary} strokeWidth={2} dot={false} />
        <Line type="monotone" dataKey="stability" name="Stability Score" stroke={CHART_COLORS.quaternary} strokeWidth={2} dot={false} />
      </ReLineChart>
    </ResponsiveContainer>
  );
}

function EngagementChart({ engagementMetrics }: { engagementMetrics: EngagementMetric[] }) {
  const chartData = useMemo(() => {
    if (!engagementMetrics?.length) return [];
    
    return engagementMetrics.slice(0, 10).reverse().map(e => ({
      date: format(new Date(e.period_end), "MMM d"),
      adherence: e.adherence_score,
      engagement: e.engagement_score,
      checkins: e.checkins_completed,
      streak: e.current_streak
    }));
  }, [engagementMetrics]);

  if (!chartData.length) {
    return (
      <div className="text-center py-8">
        <Target className="h-12 w-12 mx-auto text-muted-foreground opacity-50 mb-2" />
        <p className="text-sm text-muted-foreground">No engagement data available</p>
      </div>
    );
  }

  return (
    <ResponsiveContainer width="100%" height={250}>
      <BarChart data={chartData}>
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
        <Bar dataKey="adherence" name="Adherence %" fill={CHART_COLORS.primary} radius={[4, 4, 0, 0]} />
        <Bar dataKey="engagement" name="Engagement Score" fill={CHART_COLORS.secondary} radius={[4, 4, 0, 0]} />
        <Bar dataKey="checkins" name="Check-ins" fill={CHART_COLORS.tertiary} radius={[4, 4, 0, 0]} />
      </BarChart>
    </ResponsiveContainer>
  );
}

function DPIGauge({ dpi }: { dpi: DPIResult | null }) {
  const getBucketColor = (bucket: string) => {
    switch (bucket) {
      case 'green': return 'text-green-500 bg-green-100 dark:bg-green-900/30';
      case 'yellow': return 'text-yellow-500 bg-yellow-100 dark:bg-yellow-900/30';
      case 'orange': return 'text-orange-500 bg-orange-100 dark:bg-orange-900/30';
      case 'red': return 'text-red-500 bg-red-100 dark:bg-red-900/30';
      default: return 'text-muted-foreground bg-muted';
    }
  };

  const getBucketLabel = (bucket: string) => {
    switch (bucket) {
      case 'green': return 'Stable';
      case 'yellow': return 'Elevated';
      case 'orange': return 'Warning';
      case 'red': return 'Critical';
      default: return 'Unknown';
    }
  };

  if (!dpi) {
    return (
      <div className="text-center py-8">
        <Gauge className="h-12 w-12 mx-auto text-muted-foreground opacity-50 mb-2" />
        <p className="text-sm text-muted-foreground">No DPI data available</p>
        <p className="text-xs text-muted-foreground mt-1">Complete check-ins to generate your health index</p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-center">
        <div className={`relative w-32 h-32 rounded-full flex items-center justify-center ${getBucketColor(dpi.dpi_bucket)}`}>
          <div className="text-center">
            <p className="text-3xl font-bold">{dpi.dpi_normalized.toFixed(0)}</p>
            <p className="text-xs font-medium">{getBucketLabel(dpi.dpi_bucket)}</p>
          </div>
        </div>
      </div>
      
      {dpi.bucket_changed && dpi.previous_bucket && (
        <Alert className="bg-amber-50 dark:bg-amber-900/20 border-amber-200">
          <AlertTriangle className="h-4 w-4 text-amber-500" />
          <AlertDescription className="text-sm">
            Status changed from <span className="font-medium capitalize">{dpi.previous_bucket}</span> to{' '}
            <span className="font-medium capitalize">{dpi.dpi_bucket}</span>
          </AlertDescription>
        </Alert>
      )}

      {dpi.explainability?.top_contributors?.length > 0 && (
        <div className="space-y-2">
          <p className="text-xs font-medium text-muted-foreground">Top Contributing Factors:</p>
          <div className="flex flex-wrap gap-1">
            {dpi.explainability.top_contributors.slice(0, 3).map((contributor, i) => (
              <Badge key={i} variant="outline" className="text-xs capitalize">
                {contributor.replace(/_/g, ' ')}
              </Badge>
            ))}
          </div>
        </div>
      )}

      <p className="text-xs text-muted-foreground text-center italic">
        Deterioration Prediction Index - observational indicator only
      </p>
    </div>
  );
}

function OrganScoresPanel({ organScores }: { organScores: OrganScoreResult | null }) {
  const getOrganIcon = (organ: string) => {
    switch (organ) {
      case 'respiratory': return Wind;
      case 'cardio_fluid': return Droplets;
      case 'hepatic_hematologic': return Heart;
      case 'mobility': return Footprints;
      case 'cognitive_behavioral': return Brain;
      default: return Activity;
    }
  };

  const getBucketStyles = (bucket: string) => {
    switch (bucket) {
      case 'green': return 'border-green-500 bg-green-50 dark:bg-green-900/20';
      case 'yellow': return 'border-yellow-500 bg-yellow-50 dark:bg-yellow-900/20';
      case 'orange': return 'border-orange-500 bg-orange-50 dark:bg-orange-900/20';
      case 'red': return 'border-red-500 bg-red-50 dark:bg-red-900/20';
      default: return 'border-muted bg-muted/50';
    }
  };

  const getProgressColor = (bucket: string) => {
    switch (bucket) {
      case 'green': return 'bg-green-500';
      case 'yellow': return 'bg-yellow-500';
      case 'orange': return 'bg-orange-500';
      case 'red': return 'bg-red-500';
      default: return 'bg-muted';
    }
  };

  if (!organScores?.organ_scores) {
    return (
      <div className="text-center py-8">
        <Activity className="h-12 w-12 mx-auto text-muted-foreground opacity-50 mb-2" />
        <p className="text-sm text-muted-foreground">No organ system data available</p>
        <p className="text-xs text-muted-foreground mt-1">Complete check-ins to see organ-level insights</p>
      </div>
    );
  }

  return (
    <div className="space-y-3">
      {Object.entries(organScores.organ_scores).map(([key, score]) => {
        const Icon = getOrganIcon(key);
        return (
          <div 
            key={key} 
            className={`p-3 rounded-lg border-l-4 ${getBucketStyles(score.color_bucket)}`}
            data-testid={`organ-score-${key}`}
          >
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                <Icon className="h-4 w-4" />
                <span className="font-medium text-sm">{score.organ_name}</span>
              </div>
              <Badge variant="outline" className="text-xs capitalize">
                {score.color_bucket}
              </Badge>
            </div>
            <div className="space-y-1">
              <div className="flex justify-between text-xs text-muted-foreground">
                <span>Score</span>
                <span>{score.normalized_score.toFixed(0)}/100</span>
              </div>
              <div className="h-2 bg-muted rounded-full overflow-hidden">
                <div 
                  className={`h-full ${getProgressColor(score.color_bucket)} transition-all`}
                  style={{ width: `${Math.min(score.normalized_score, 100)}%` }}
                />
              </div>
            </div>
          </div>
        );
      })}
      <p className="text-xs text-muted-foreground text-center italic mt-4">
        Organ-level observational indicators - NOT diagnostic assessments
      </p>
    </div>
  );
}

function MLPredictionsPanel({ prediction }: { prediction: MLPrediction | null }) {
  const getRiskColor = (riskLevel: string) => {
    switch (riskLevel.toLowerCase()) {
      case 'low': return 'text-green-500 bg-green-100 dark:bg-green-900/30';
      case 'moderate': return 'text-yellow-500 bg-yellow-100 dark:bg-yellow-900/30';
      case 'high': return 'text-orange-500 bg-orange-100 dark:bg-orange-900/30';
      case 'critical': return 'text-red-500 bg-red-100 dark:bg-red-900/30';
      default: return 'text-muted-foreground bg-muted';
    }
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return 'text-green-500';
    if (confidence >= 0.6) return 'text-yellow-500';
    if (confidence >= 0.4) return 'text-orange-500';
    return 'text-red-500';
  };

  const getTrendIcon = (trend: string) => {
    switch (trend.toLowerCase()) {
      case 'improving': return TrendingDown;
      case 'deteriorating': return TrendingUp;
      case 'stable': return Minus;
      default: return Activity;
    }
  };

  const formatHorizon = (horizon: string) => {
    const hours = horizon.replace('h', '').replace('_hours', '');
    return `${hours}h`;
  };

  if (!prediction || prediction.status === 'no_predictions' || prediction.status === 'insufficient_data') {
    return (
      <div className="text-center py-8">
        <Sparkles className="h-12 w-12 mx-auto text-muted-foreground opacity-50 mb-2" />
        <p className="text-sm text-muted-foreground">No ML predictions available</p>
        <p className="text-xs text-muted-foreground mt-1">
          {prediction?.status === 'insufficient_data' 
            ? 'Insufficient vital signs data for prediction'
            : 'Complete more check-ins to enable predictive analysis'}
        </p>
      </div>
    );
  }

  const TrendIconComponent = getTrendIcon(prediction.trend_direction);
  const horizonEntries = Object.entries(prediction.predictions || {}).sort((a, b) => {
    const aHours = parseInt(a[0].replace('h', '').replace('_hours', ''));
    const bHours = parseInt(b[0].replace('h', '').replace('_hours', ''));
    return aHours - bHours;
  });

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <Sparkles className="h-4 w-4 text-primary" />
          <span className="text-sm font-medium">ML Deterioration Prediction</span>
        </div>
        <Badge variant="outline" className="text-xs">
          v{prediction.model_version || '1.0'}
        </Badge>
      </div>

      <div className="grid grid-cols-2 gap-3">
        <div className={`p-3 rounded-lg ${getRiskColor(prediction.risk_trajectory)}`}>
          <p className="text-xs text-muted-foreground mb-1">Ensemble Risk Score</p>
          <p className="text-2xl font-bold">{(prediction.ensemble_score * 100).toFixed(0)}%</p>
          <div className="flex items-center gap-1 mt-1">
            <TrendIconComponent className="h-3 w-3" />
            <span className="text-xs capitalize">{prediction.trend_direction}</span>
          </div>
        </div>
        <div className="p-3 rounded-lg bg-muted/50">
          <p className="text-xs text-muted-foreground mb-1">Confidence</p>
          <p className={`text-2xl font-bold ${getConfidenceColor(prediction.ensemble_confidence)}`}>
            {(prediction.ensemble_confidence * 100).toFixed(0)}%
          </p>
          <p className="text-xs text-muted-foreground mt-1">
            ML: {(prediction.ml_weight * 100).toFixed(0)}% | Stat: {(prediction.statistical_weight * 100).toFixed(0)}%
          </p>
        </div>
      </div>

      <div className="space-y-2">
        <p className="text-xs font-medium text-muted-foreground">Prediction Horizons</p>
        <div className="grid grid-cols-4 gap-2">
          {horizonEntries.map(([horizon, pred]) => (
            <div 
              key={horizon}
              className={`p-2 rounded-lg text-center ${getRiskColor(pred.risk_level)}`}
              data-testid={`prediction-${horizon}`}
            >
              <p className="text-xs font-medium">{formatHorizon(horizon)}</p>
              <p className="text-lg font-bold">{(pred.deterioration_probability * 100).toFixed(0)}%</p>
              <p className={`text-xs ${getConfidenceColor(pred.confidence)}`}>
                ±{((1 - pred.confidence) * 100).toFixed(0)}%
              </p>
            </div>
          ))}
        </div>
      </div>

      <Alert className="bg-blue-50 dark:bg-blue-900/20 border-blue-200">
        <Info className="h-4 w-4 text-blue-500" />
        <AlertDescription className="text-xs">
          {prediction.disclaimer || 'ML predictions are observational indicators only. Not a diagnosis.'}
        </AlertDescription>
      </Alert>

      <p className="text-xs text-muted-foreground text-center italic">
        Last computed: {prediction.computed_at ? format(new Date(prediction.computed_at), 'MMM d, h:mm a') : 'Unknown'}
      </p>
    </div>
  );
}

function RiskScoreGauge({ riskScore, isLoading }: { riskScore: RiskScoreData | null; isLoading: boolean }) {
  const getLevelStyles = (level: string) => {
    switch (level) {
      case 'stable': return 'text-green-500 bg-green-100 dark:bg-green-900/30 border-green-500';
      case 'monitoring': return 'text-yellow-500 bg-yellow-100 dark:bg-yellow-900/30 border-yellow-500';
      case 'urgent': return 'text-red-500 bg-red-100 dark:bg-red-900/30 border-red-500';
      default: return 'text-muted-foreground bg-muted border-muted';
    }
  };

  const getLevelLabel = (level: string) => {
    switch (level) {
      case 'stable': return 'Stable';
      case 'monitoring': return 'Monitoring';
      case 'urgent': return 'Urgent';
      default: return 'Unknown';
    }
  };

  const getScoreColor = (score: number) => {
    if (score <= 2) return 'text-green-500';
    if (score <= 5) return 'text-yellow-500';
    return 'text-red-500';
  };

  const getProgressColor = (score: number) => {
    if (score <= 2) return 'bg-green-500';
    if (score <= 5) return 'bg-yellow-500';
    return 'bg-red-500';
  };

  if (isLoading) {
    return (
      <div className="space-y-4">
        <Skeleton className="h-32 w-32 rounded-full mx-auto" />
        <Skeleton className="h-4 w-full" />
        <Skeleton className="h-4 w-3/4" />
      </div>
    );
  }

  if (!riskScore) {
    return (
      <div className="text-center py-8">
        <Shield className="h-12 w-12 mx-auto text-muted-foreground opacity-50 mb-2" />
        <p className="text-sm text-muted-foreground">No risk score data available</p>
        <p className="text-xs text-muted-foreground mt-1">Complete daily check-ins to calculate your risk score</p>
      </div>
    );
  }

  return (
    <div className="space-y-4" data-testid="risk-score-gauge">
      <div className="flex items-center justify-center">
        <div className={`relative w-36 h-36 rounded-full flex items-center justify-center border-4 ${getLevelStyles(riskScore.level)}`}>
          <div className="text-center">
            <p className={`text-4xl font-bold ${getScoreColor(riskScore.score)}`} data-testid="text-risk-score">
              {riskScore.score}
            </p>
            <p className="text-xs font-medium">/15</p>
            <Badge className={`mt-1 ${getLevelStyles(riskScore.level)}`} data-testid="badge-risk-level">
              {getLevelLabel(riskScore.level)}
            </Badge>
          </div>
        </div>
      </div>

      <div className="space-y-2">
        <div className="flex justify-between text-xs text-muted-foreground">
          <span>Composite Risk Score</span>
          <span>{riskScore.score}/15</span>
        </div>
        <div className="h-3 bg-muted rounded-full overflow-hidden">
          <div 
            className={`h-full ${getProgressColor(riskScore.score)} transition-all duration-500`}
            style={{ width: `${(riskScore.score / 15) * 100}%` }}
          />
        </div>
        <div className="flex justify-between text-xs text-muted-foreground">
          <span className="text-green-500">Stable (0-2)</span>
          <span className="text-yellow-500">Monitoring (3-5)</span>
          <span className="text-red-500">Urgent (6-15)</span>
        </div>
      </div>

      <div className="grid grid-cols-3 gap-2 text-center">
        <div className="p-2 rounded-lg bg-muted/50">
          <p className="text-xs text-muted-foreground">Total Deviations</p>
          <p className="text-lg font-semibold" data-testid="text-total-deviations">{riskScore.total_deviations}</p>
        </div>
        <div className="p-2 rounded-lg bg-red-50 dark:bg-red-900/20">
          <p className="text-xs text-muted-foreground">Critical</p>
          <p className="text-lg font-semibold text-red-500" data-testid="text-critical-deviations">{riskScore.critical_deviations}</p>
        </div>
        <div className="p-2 rounded-lg bg-yellow-50 dark:bg-yellow-900/20">
          <p className="text-xs text-muted-foreground">Moderate</p>
          <p className="text-lg font-semibold text-yellow-500" data-testid="text-moderate-deviations">{riskScore.moderate_deviations}</p>
        </div>
      </div>

      {riskScore.factors.length > 0 && (
        <div className="space-y-2">
          <p className="text-xs font-medium text-muted-foreground">Contributing Factors:</p>
          <div className="space-y-2 max-h-48 overflow-y-auto">
            {riskScore.factors.map((factor, idx) => (
              <div 
                key={idx}
                className={`p-2 rounded-lg text-xs ${
                  factor.severity_level === 'critical' 
                    ? 'bg-red-50 dark:bg-red-900/20 border-l-2 border-red-500'
                    : factor.severity_level === 'moderate'
                    ? 'bg-yellow-50 dark:bg-yellow-900/20 border-l-2 border-yellow-500'
                    : 'bg-muted/50'
                }`}
                data-testid={`risk-factor-${idx}`}
              >
                <div className="flex items-center justify-between gap-2 flex-wrap">
                  <span className="font-medium capitalize">{factor.metric_name.replace(/_/g, ' ')}</span>
                  <Badge variant="outline" className="text-xs">
                    +{factor.points} pts
                  </Badge>
                </div>
                <p className="text-muted-foreground mt-1">{factor.description}</p>
                <div className="flex items-center gap-2 mt-1">
                  <span className="text-muted-foreground">Z-Score:</span>
                  <span className={factor.z_score > 2 ? 'text-red-500' : factor.z_score < -1.5 ? 'text-blue-500' : ''}>
                    {factor.z_score > 0 ? '+' : ''}{factor.z_score.toFixed(2)}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {riskScore.recommendation && (
        <Alert className={`${
          riskScore.level === 'urgent' 
            ? 'bg-red-50 dark:bg-red-900/20 border-red-200'
            : riskScore.level === 'monitoring'
            ? 'bg-yellow-50 dark:bg-yellow-900/20 border-yellow-200'
            : 'bg-green-50 dark:bg-green-900/20 border-green-200'
        }`}>
          <AlertCircle className={`h-4 w-4 ${
            riskScore.level === 'urgent' ? 'text-red-500' : 
            riskScore.level === 'monitoring' ? 'text-yellow-500' : 'text-green-500'
          }`} />
          <AlertTitle className="text-sm">Recommendation</AlertTitle>
          <AlertDescription className="text-xs">
            {riskScore.recommendation}
          </AlertDescription>
        </Alert>
      )}

      {riskScore.action_items && riskScore.action_items.length > 0 && (
        <div className="space-y-2">
          <p className="text-xs font-medium text-muted-foreground">Suggested Actions:</p>
          <ul className="text-xs space-y-1 list-disc list-inside text-muted-foreground">
            {riskScore.action_items.map((item, idx) => (
              <li key={idx} data-testid={`action-item-${idx}`}>{item}</li>
            ))}
          </ul>
        </div>
      )}

      <p className="text-xs text-muted-foreground text-center italic">
        Wellness monitoring indicator only - NOT a medical diagnosis. 
        {riskScore.calculated_at && ` Last calculated: ${format(new Date(riskScore.calculated_at), 'MMM d, h:mm a')}`}
      </p>
    </div>
  );
}

function RiskHistoryChart({ history, isLoading }: { history: RiskHistoryEntry[]; isLoading: boolean }) {
  const chartData = useMemo(() => {
    if (!history?.length) return [];
    return history.map(h => ({
      date: format(new Date(h.date), "MMM d"),
      score: h.score,
      deviations: h.deviation_count,
      level: h.level
    }));
  }, [history]);

  if (isLoading) {
    return <Skeleton className="h-64 w-full" />;
  }

  if (!chartData.length) {
    return (
      <div className="text-center py-8">
        <BarChart3 className="h-12 w-12 mx-auto text-muted-foreground opacity-50 mb-2" />
        <p className="text-sm text-muted-foreground">No risk history available</p>
        <p className="text-xs text-muted-foreground mt-1">Risk scores will be tracked over time</p>
      </div>
    );
  }

  return (
    <div className="space-y-4" data-testid="risk-history-chart">
      <ResponsiveContainer width="100%" height={250}>
        <AreaChart data={chartData}>
          <defs>
            <linearGradient id="riskGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="hsl(var(--destructive))" stopOpacity={0.3}/>
              <stop offset="95%" stopColor="hsl(var(--destructive))" stopOpacity={0}/>
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
          <XAxis dataKey="date" className="text-xs" />
          <YAxis domain={[0, 15]} className="text-xs" ticks={[0, 2, 5, 10, 15]} />
          <Tooltip 
            contentStyle={{ 
              backgroundColor: 'hsl(var(--card))', 
              border: '1px solid hsl(var(--border))',
              borderRadius: '8px'
            }}
            formatter={(value: number, name: string) => {
              if (name === 'score') {
                const level = value <= 2 ? 'Stable' : value <= 5 ? 'Monitoring' : 'Urgent';
                return [`${value}/15 (${level})`, 'Risk Score'];
              }
              return [value, 'Deviations'];
            }}
          />
          <Legend />
          <Area 
            type="monotone" 
            dataKey="score" 
            name="Risk Score" 
            stroke="hsl(var(--destructive))" 
            fill="url(#riskGradient)" 
            strokeWidth={2}
          />
          <Area 
            type="monotone" 
            dataKey="deviations" 
            name="Deviations" 
            stroke={CHART_COLORS.secondary} 
            fillOpacity={0.2}
            fill={CHART_COLORS.secondary}
          />
        </AreaChart>
      </ResponsiveContainer>
      
      <div className="flex items-center justify-center gap-4 text-xs">
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 rounded-full bg-green-500" />
          <span>Stable (0-2)</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 rounded-full bg-yellow-500" />
          <span>Monitoring (3-5)</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 rounded-full bg-red-500" />
          <span>Urgent (6-15)</span>
        </div>
      </div>

      <p className="text-xs text-muted-foreground text-center italic">
        7-day risk score trend - wellness monitoring indicator only
      </p>
    </div>
  );
}

function BaselinePanel({ 
  baseline, 
  isLoading, 
  onRecalculate, 
  isRecalculating 
}: { 
  baseline: BaselineData | null; 
  isLoading: boolean;
  onRecalculate?: () => void;
  isRecalculating?: boolean;
}) {
  if (isLoading) {
    return (
      <div className="space-y-4">
        <Skeleton className="h-8 w-full" />
        <Skeleton className="h-24" />
        <Skeleton className="h-24" />
      </div>
    );
  }

  if (!baseline) {
    return (
      <div className="text-center py-8">
        <Target className="h-12 w-12 mx-auto text-muted-foreground opacity-50 mb-2" />
        <p className="text-sm text-muted-foreground">No baseline data available</p>
        <p className="text-xs text-muted-foreground mt-1">Complete at least 7 days of check-ins to establish your baseline</p>
        {onRecalculate && (
          <Button
            variant="outline"
            size="sm"
            onClick={onRecalculate}
            disabled={isRecalculating}
            className="mt-4"
            data-testid="button-create-baseline"
          >
            <RefreshCw className={`h-4 w-4 mr-2 ${isRecalculating ? 'animate-spin' : ''}`} />
            {isRecalculating ? 'Calculating...' : 'Calculate Baseline'}
          </Button>
        )}
      </div>
    );
  }

  const formatStat = (stat: BaselineStats | null, unit: string = '') => {
    if (!stat?.mean) return '—';
    return `${stat.mean.toFixed(1)}${unit} ±${(stat.std || 0).toFixed(1)}`;
  };

  const getQualityColor = (quality: string | null) => {
    switch (quality) {
      case 'excellent': return 'text-green-500 bg-green-100 dark:bg-green-900/30';
      case 'good': return 'text-blue-500 bg-blue-100 dark:bg-blue-900/30';
      case 'fair': return 'text-yellow-500 bg-yellow-100 dark:bg-yellow-900/30';
      case 'poor': return 'text-red-500 bg-red-100 dark:bg-red-900/30';
      default: return 'text-muted-foreground bg-muted';
    }
  };

  return (
    <div className="space-y-4" data-testid="baseline-panel">
      <div className="flex items-center justify-between gap-2 flex-wrap">
        <div className="flex items-center gap-2">
          <Target className="h-5 w-5 text-primary" />
          <span className="font-medium">Your Health Baseline</span>
        </div>
        <div className="flex items-center gap-2">
          <Badge variant="outline" className="text-xs">
            {baseline.data_points_count} data points
          </Badge>
          {baseline.baseline_quality && (
            <Badge className={`text-xs ${getQualityColor(baseline.baseline_quality)}`}>
              {baseline.baseline_quality.toUpperCase()} quality
            </Badge>
          )}
        </div>
      </div>

      <div className="text-xs text-muted-foreground">
        Baseline period: {format(new Date(baseline.baseline_start_date), 'MMM d')} - {format(new Date(baseline.baseline_end_date), 'MMM d, yyyy')}
      </div>

      <div className="grid grid-cols-2 gap-3">
        <div className="p-3 rounded-lg bg-muted/50">
          <div className="flex items-center gap-2 mb-2">
            <Wind className="h-4 w-4 text-blue-500" />
            <span className="text-sm font-medium">Respiratory Rate</span>
          </div>
          <p className="text-lg font-bold" data-testid="baseline-respiratory">
            {formatStat(baseline.respiratory_rate, ' bpm')}
          </p>
          {baseline.respiratory_rate?.min_value && baseline.respiratory_rate?.max_value && (
            <p className="text-xs text-muted-foreground">
              Range: {baseline.respiratory_rate.min_value.toFixed(0)}-{baseline.respiratory_rate.max_value.toFixed(0)} bpm
            </p>
          )}
        </div>

        <div className="p-3 rounded-lg bg-muted/50">
          <div className="flex items-center gap-2 mb-2">
            <Activity className="h-4 w-4 text-rose-500" />
            <span className="text-sm font-medium">Discomfort Level</span>
          </div>
          <p className="text-lg font-bold" data-testid="baseline-pain">
            {formatStat(baseline.pain_facial, '/10')}
          </p>
          {baseline.pain_facial?.min_value !== null && baseline.pain_facial?.max_value !== null && (
            <p className="text-xs text-muted-foreground">
              Range: {baseline.pain_facial.min_value?.toFixed(1)}-{baseline.pain_facial.max_value?.toFixed(1)}
            </p>
          )}
        </div>

        <div className="p-3 rounded-lg bg-muted/50">
          <div className="flex items-center gap-2 mb-2">
            <AlertCircle className="h-4 w-4 text-orange-500" />
            <span className="text-sm font-medium">Symptom Severity</span>
          </div>
          <p className="text-lg font-bold" data-testid="baseline-symptom">
            {formatStat(baseline.symptom_severity, '/10')}
          </p>
          {baseline.symptom_severity?.min_value !== null && baseline.symptom_severity?.max_value !== null && (
            <p className="text-xs text-muted-foreground">
              Range: {baseline.symptom_severity.min_value?.toFixed(1)}-{baseline.symptom_severity.max_value?.toFixed(1)}
            </p>
          )}
        </div>

        <div className="p-3 rounded-lg bg-muted/50">
          <div className="flex items-center gap-2 mb-2">
            <Heart className="h-4 w-4 text-purple-500" />
            <span className="text-sm font-medium">Self-Reported Pain</span>
          </div>
          <p className="text-lg font-bold" data-testid="baseline-self-pain">
            {formatStat(baseline.pain_self_reported, '/10')}
          </p>
          {baseline.pain_self_reported?.min_value !== null && baseline.pain_self_reported?.max_value !== null && (
            <p className="text-xs text-muted-foreground">
              Range: {baseline.pain_self_reported.min_value?.toFixed(1)}-{baseline.pain_self_reported.max_value?.toFixed(1)}
            </p>
          )}
        </div>
      </div>

      {baseline.activity_impact_rate !== null && (
        <div className="p-3 rounded-lg bg-muted/50">
          <div className="flex items-center justify-between gap-2">
            <div className="flex items-center gap-2">
              <Footprints className="h-4 w-4 text-green-500" />
              <span className="text-sm font-medium">Activity Impact Rate</span>
            </div>
            <span className="text-lg font-bold" data-testid="baseline-activity">
              {(baseline.activity_impact_rate * 100).toFixed(0)}%
            </span>
          </div>
          <Progress value={baseline.activity_impact_rate * 100} className="mt-2" />
        </div>
      )}

      <Alert className="bg-blue-50 dark:bg-blue-900/20 border-blue-200">
        <Info className="h-4 w-4 text-blue-500" />
        <AlertDescription className="text-xs">
          Your baseline represents your typical health patterns over the past 7 days. 
          Deviations from baseline help identify wellness changes early.
        </AlertDescription>
      </Alert>

      <div className="flex items-center justify-between gap-2 flex-wrap">
        <p className="text-xs text-muted-foreground italic">
          Baseline established: {format(new Date(baseline.created_at), 'MMM d, yyyy h:mm a')}
        </p>
        {onRecalculate && (
          <Button
            variant="outline"
            size="sm"
            onClick={onRecalculate}
            disabled={isRecalculating}
            data-testid="button-recalculate-baseline"
          >
            <RefreshCw className={`h-3 w-3 mr-1 ${isRecalculating ? 'animate-spin' : ''}`} />
            {isRecalculating ? 'Updating...' : 'Recalculate'}
          </Button>
        )}
      </div>
    </div>
  );
}

function DeviceHealthSectionPanel({ 
  analytics, 
  isLoading 
}: { 
  analytics: DeviceAnalyticsData | null; 
  isLoading: boolean;
}) {
  const getRiskLevelColor = (level: string) => {
    switch (level) {
      case 'critical': return 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400';
      case 'high': return 'bg-orange-100 text-orange-700 dark:bg-orange-900/30 dark:text-orange-400';
      case 'moderate': return 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400';
      case 'low': return 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400';
      default: return 'bg-muted text-muted-foreground';
    }
  };

  const getTrendIcon = (trend: string) => {
    switch (trend) {
      case 'improving': return { icon: TrendingUp, color: 'text-green-500' };
      case 'worsening':
      case 'declining': return { icon: TrendingDown, color: 'text-red-500' };
      default: return { icon: Minus, color: 'text-muted-foreground' };
    }
  };

  const getDeteriorationColor = (index: number) => {
    if (index >= 0.7) return 'text-red-500';
    if (index >= 0.4) return 'text-yellow-500';
    return 'text-green-500';
  };

  if (isLoading) {
    return (
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {Array(6).fill(0).map((_, i) => (
          <Skeleton key={i} className="h-48" />
        ))}
      </div>
    );
  }

  if (!analytics || !analytics.sections || Object.keys(analytics.sections).length === 0) {
    return (
      <div className="text-center py-12">
        <Activity className="h-16 w-16 mx-auto text-muted-foreground opacity-50 mb-4" />
        <h3 className="text-lg font-semibold mb-2">No Device Data Available</h3>
        <p className="text-muted-foreground mb-4">
          Connect medical devices to view health section analytics.
        </p>
        <Button asChild variant="outline">
          <Link href="/device-connect">
            <Link2 className="h-4 w-4 mr-2" />
            Connect Devices
          </Link>
        </Button>
      </div>
    );
  }

  return (
    <div className="space-y-6" data-testid="device-health-sections">
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card className="col-span-1">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div className="space-y-1">
                <p className="text-sm text-muted-foreground">Overall Device Risk</p>
                <p className="text-3xl font-bold" data-testid="text-overall-risk">
                  {(analytics.overall_risk_score * 100).toFixed(0)}%
                </p>
              </div>
              <div className={`p-3 rounded-full ${
                analytics.overall_risk_score >= 0.7 ? 'bg-red-100 dark:bg-red-900/30' :
                analytics.overall_risk_score >= 0.4 ? 'bg-yellow-100 dark:bg-yellow-900/30' :
                'bg-green-100 dark:bg-green-900/30'
              }`}>
                <Shield className={`h-6 w-6 ${
                  analytics.overall_risk_score >= 0.7 ? 'text-red-500' :
                  analytics.overall_risk_score >= 0.4 ? 'text-yellow-500' :
                  'text-green-500'
                }`} />
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="col-span-1">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div className="space-y-1">
                <p className="text-sm text-muted-foreground">Overall Trend</p>
                <div className="flex items-center gap-2">
                  {(() => {
                    const { icon: TrendIconComponent, color } = getTrendIcon(analytics.overall_trend);
                    return (
                      <>
                        <TrendIconComponent className={`h-6 w-6 ${color}`} />
                        <p className="text-lg font-semibold capitalize" data-testid="text-overall-trend">
                          {analytics.overall_trend}
                        </p>
                      </>
                    );
                  })()}
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="col-span-1">
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div className="space-y-1">
                <p className="text-sm text-muted-foreground">Critical Alerts</p>
                <p className="text-3xl font-bold" data-testid="text-critical-count">
                  {analytics.critical_alerts?.length || 0}
                </p>
              </div>
              <div className={`p-3 rounded-full ${
                (analytics.critical_alerts?.length || 0) > 0 
                  ? 'bg-red-100 dark:bg-red-900/30' 
                  : 'bg-green-100 dark:bg-green-900/30'
              }`}>
                <AlertTriangle className={`h-6 w-6 ${
                  (analytics.critical_alerts?.length || 0) > 0 ? 'text-red-500' : 'text-green-500'
                }`} />
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {Object.entries(analytics.sections).map(([sectionKey, section]) => {
          const config = HEALTH_SECTION_CONFIG[sectionKey] || {
            label: sectionKey.replace(/_/g, ' '),
            icon: Activity,
            color: 'text-primary',
            description: ''
          };
          const SectionIcon = config.icon;
          const { icon: TrendIconComponent, color: trendColor } = getTrendIcon(section.trend);

          return (
            <Card key={sectionKey} className="hover-elevate" data-testid={`section-card-${sectionKey}`}>
              <CardHeader className="pb-2">
                <div className="flex items-center justify-between gap-2">
                  <div className="flex items-center gap-2">
                    <div className={`p-2 rounded-lg bg-muted`}>
                      <SectionIcon className={`h-5 w-5 ${config.color}`} />
                    </div>
                    <div>
                      <CardTitle className="text-base">{config.label}</CardTitle>
                      <CardDescription className="text-xs">{config.description}</CardDescription>
                    </div>
                  </div>
                  {section.alert_triggered && (
                    <Badge variant="destructive" className="text-xs">
                      <AlertCircle className="h-3 w-3 mr-1" />
                      Alert
                    </Badge>
                  )}
                </div>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="grid grid-cols-2 gap-3">
                  <div className="p-2 rounded-lg bg-muted/50">
                    <p className="text-xs text-muted-foreground">Risk Score</p>
                    <div className="flex items-center gap-2 mt-1">
                      <p className="text-lg font-bold">{(section.risk_score * 100).toFixed(0)}%</p>
                      <Badge className={`text-xs ${getRiskLevelColor(section.risk_level)}`}>
                        {section.risk_level}
                      </Badge>
                    </div>
                  </div>
                  <div className="p-2 rounded-lg bg-muted/50">
                    <p className="text-xs text-muted-foreground">Deterioration</p>
                    <p className={`text-lg font-bold ${getDeteriorationColor(section.deterioration_index)}`}>
                      {(section.deterioration_index * 100).toFixed(0)}%
                    </p>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-3">
                  <div className="p-2 rounded-lg bg-muted/50">
                    <p className="text-xs text-muted-foreground">Trend</p>
                    <div className="flex items-center gap-1 mt-1">
                      <TrendIconComponent className={`h-4 w-4 ${trendColor}`} />
                      <p className="text-sm font-medium capitalize">{section.trend}</p>
                    </div>
                  </div>
                  <div className="p-2 rounded-lg bg-muted/50">
                    <p className="text-xs text-muted-foreground">Stability</p>
                    <div className="flex items-center gap-1 mt-1">
                      <Progress value={section.stability_score * 100} className="h-2 flex-1" />
                      <p className="text-sm font-medium">{(section.stability_score * 100).toFixed(0)}%</p>
                    </div>
                  </div>
                </div>

                <div className="flex items-center justify-between text-xs text-muted-foreground pt-2 border-t">
                  <div className="flex items-center gap-1">
                    <Activity className="h-3 w-3" />
                    <span>Coverage: {(section.data_coverage * 100).toFixed(0)}%</span>
                  </div>
                  {section.anomalies_detected > 0 && (
                    <Badge variant="outline" className="text-xs">
                      {section.anomalies_detected} anomalies
                    </Badge>
                  )}
                </div>
              </CardContent>
            </Card>
          );
        })}
      </div>

      {analytics.recommendations && analytics.recommendations.length > 0 && (
        <Alert className="bg-blue-50 dark:bg-blue-900/20 border-blue-200">
          <Info className="h-4 w-4 text-blue-500" />
          <AlertTitle>Device-Based Recommendations</AlertTitle>
          <AlertDescription>
            <ul className="list-disc list-inside mt-2 space-y-1 text-sm">
              {analytics.recommendations.map((rec, idx) => (
                <li key={idx}>{rec}</li>
              ))}
            </ul>
          </AlertDescription>
        </Alert>
      )}

      {analytics.critical_alerts && analytics.critical_alerts.length > 0 && (
        <Alert variant="destructive">
          <AlertTriangle className="h-4 w-4" />
          <AlertTitle>Critical Device Alerts</AlertTitle>
          <AlertDescription>
            <ul className="list-disc list-inside mt-2 space-y-1 text-sm">
              {analytics.critical_alerts.map((alert, idx) => (
                <li key={idx}>{alert}</li>
              ))}
            </ul>
          </AlertDescription>
        </Alert>
      )}

      <p className="text-xs text-muted-foreground text-center italic">
        Device analytics based on connected medical devices. 
        {analytics.generated_at && ` Last updated: ${format(new Date(analytics.generated_at), 'MMM d, h:mm a')}`}
      </p>
    </div>
  );
}

export default function AIAlertsDashboard() {
  const { toast } = useToast();
  const { user } = useAuth();
  const [isComputing, setIsComputing] = useState(false);

  const patientId = user?.id || "demo-patient";

  const { data: overview, isLoading: loadingOverview, refetch: refetchOverview } = useQuery<PatientOverview>({
    queryKey: ['/api/ai-health-alerts/patient-overview', patientId],
    queryFn: async () => {
      const response = await fetch(`/api/ai-health-alerts/patient-overview/${patientId}`);
      if (!response.ok) {
        if (response.status === 404) return null;
        throw new Error('Failed to fetch overview');
      }
      return response.json();
    },
    enabled: !!patientId,
  });

  const { data: dpiData, isLoading: loadingDpi, refetch: refetchDpi } = useQuery<DPIResult>({
    queryKey: ['/api/ai-health-alerts/v2/dpi', patientId],
    queryFn: async () => {
      const response = await fetch(`/api/ai-health-alerts/v2/dpi/${patientId}`);
      if (!response.ok) {
        if (response.status === 404 || response.status === 502) return null;
        throw new Error('Failed to fetch DPI');
      }
      return response.json();
    },
    enabled: !!patientId,
  });

  const { data: organScores, isLoading: loadingOrgans, refetch: refetchOrgans } = useQuery<OrganScoreResult>({
    queryKey: ['/api/ai-health-alerts/v2/organ-scores', patientId],
    queryFn: async () => {
      const response = await fetch(`/api/ai-health-alerts/v2/organ-scores/${patientId}`);
      if (!response.ok) {
        if (response.status === 404 || response.status === 502) return null;
        throw new Error('Failed to fetch organ scores');
      }
      return response.json();
    },
    enabled: !!patientId,
  });

  const { data: alerts, isLoading: loadingAlerts, refetch: refetchAlerts } = useQuery<HealthAlert[]>({
    queryKey: ['/api/ai-health-alerts/alerts', patientId],
    queryFn: async () => {
      const response = await fetch(`/api/ai-health-alerts/alerts/${patientId}?limit=50`);
      if (!response.ok) {
        if (response.status === 404) return [];
        throw new Error('Failed to fetch alerts');
      }
      return response.json();
    },
    enabled: !!patientId,
  });

  const { data: trendMetrics, isLoading: loadingTrends } = useQuery<TrendMetric[]>({
    queryKey: ['/api/ai-health-alerts/trend-metrics', patientId],
    queryFn: async () => {
      const response = await fetch(`/api/ai-health-alerts/trend-metrics/${patientId}?days=14`);
      if (!response.ok) {
        if (response.status === 404) return [];
        throw new Error('Failed to fetch trends');
      }
      return response.json();
    },
    enabled: !!patientId,
  });

  const { data: engagementMetrics, isLoading: loadingEngagement } = useQuery<EngagementMetric[]>({
    queryKey: ['/api/ai-health-alerts/engagement-metrics', patientId],
    queryFn: async () => {
      const response = await fetch(`/api/ai-health-alerts/engagement-metrics/${patientId}?days=30`);
      if (!response.ok) {
        if (response.status === 404) return [];
        throw new Error('Failed to fetch engagement');
      }
      return response.json();
    },
    enabled: !!patientId,
  });

  const { data: qolMetrics, isLoading: loadingQol } = useQuery<QolMetric[]>({
    queryKey: ['/api/ai-health-alerts/qol-metrics', patientId],
    queryFn: async () => {
      const response = await fetch(`/api/ai-health-alerts/qol-metrics/${patientId}?days=30`);
      if (!response.ok) {
        if (response.status === 404) return [];
        throw new Error('Failed to fetch QoL');
      }
      return response.json();
    },
    enabled: !!patientId,
  });

  const { data: mlPrediction, isLoading: loadingPrediction, refetch: refetchPrediction } = useQuery<MLPrediction>({
    queryKey: ['/api/ai-health-alerts/v2/predictions', patientId],
    queryFn: async () => {
      const response = await fetch(`/api/ai-health-alerts/v2/predictions/${patientId}`);
      if (!response.ok) {
        if (response.status === 404 || response.status === 502) return null;
        throw new Error('Failed to fetch ML predictions');
      }
      return response.json();
    },
    enabled: !!patientId,
  });

  const { data: riskScore, isLoading: loadingRiskScore, refetch: refetchRiskScore } = useQuery<RiskScoreData>({
    queryKey: ['/api/v1/risk/score/me', patientId],
    queryFn: async () => {
      const response = await fetch(`/api/v1/risk/score/me`);
      if (!response.ok) {
        if (response.status === 404 || response.status === 502) return null;
        throw new Error('Failed to fetch risk score');
      }
      return response.json();
    },
    enabled: !!patientId,
  });

  const { data: riskHistory, isLoading: loadingRiskHistory } = useQuery<RiskHistoryEntry[]>({
    queryKey: ['/api/v1/risk/history/me', patientId],
    queryFn: async () => {
      const response = await fetch(`/api/v1/risk/history/me?days=7`);
      if (!response.ok) {
        if (response.status === 404 || response.status === 502) return [];
        throw new Error('Failed to fetch risk history');
      }
      return response.json();
    },
    enabled: !!patientId,
  });

  const { data: baseline, isLoading: loadingBaseline, refetch: refetchBaseline } = useQuery<BaselineData>({
    queryKey: ['/api/v1/baseline/current/me', patientId],
    queryFn: async () => {
      const response = await fetch(`/api/v1/baseline/current/me`);
      if (!response.ok) {
        if (response.status === 404 || response.status === 502) return null;
        throw new Error('Failed to fetch baseline');
      }
      return response.json();
    },
    enabled: !!patientId,
  });

  const { data: baselineHistory, isLoading: loadingBaselineHistory, refetch: refetchBaselineHistory } = useQuery<BaselineData[]>({
    queryKey: ['/api/v1/baseline/history/me', patientId],
    queryFn: async () => {
      const response = await fetch(`/api/v1/baseline/history/me?limit=10`);
      if (!response.ok) {
        if (response.status === 404 || response.status === 502) return [];
        throw new Error('Failed to fetch baseline history');
      }
      return response.json();
    },
    enabled: !!patientId,
  });

  const { data: diseaseRiskData, isLoading: loadingDiseaseRisk, refetch: refetchDiseaseRisk } = useQuery<MLDiseaseRiskResponse>({
    queryKey: ['/api/ml/predict/disease-risk', patientId],
    queryFn: async () => {
      const response = await fetch(`/api/ml/predict/disease-risk/${patientId}`, { credentials: 'include' });
      if (!response.ok) {
        if (response.status === 404 || response.status === 502) return null;
        throw new Error('Failed to fetch disease risk predictions');
      }
      return response.json();
    },
    enabled: !!patientId,
    staleTime: 60000,
  });

  const { data: deviceAnalytics, isLoading: loadingDeviceAnalytics, refetch: refetchDeviceAnalytics } = useQuery<DeviceAnalyticsData>({
    queryKey: ['/api/ai-health-alerts/device-analytics', patientId],
    queryFn: async () => {
      const response = await fetch(`/api/ai-health-alerts/device-analytics/${patientId}?days=7`);
      if (!response.ok) {
        if (response.status === 404 || response.status === 502 || response.status === 503) return null;
        throw new Error('Failed to fetch device analytics');
      }
      return response.json();
    },
    enabled: !!patientId,
    staleTime: 60000,
  });

  const recalculateBaselineMutation = useMutation({
    mutationFn: async () => {
      const response = await fetch('/api/v1/baseline/calculate/me', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
      });
      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Failed to recalculate baseline');
      }
      return response.json();
    },
    onSuccess: () => {
      toast({
        title: "Baseline Recalculated",
        description: "Your health baseline has been updated with the latest data.",
      });
      refetchBaseline();
      refetchBaselineHistory();
      refetchRiskScore();
      queryClient.invalidateQueries({ queryKey: ['/api/v1/baseline'] });
    },
    onError: (error: Error) => {
      toast({
        title: "Baseline Calculation Failed",
        description: error.message || "Unable to calculate baseline. You may need more health data.",
        variant: "destructive",
      });
    },
  });

  const acknowledgeMutation = useMutation({
    mutationFn: async (alertId: string) => {
      const response = await fetch(`/api/ai-health-alerts/alerts/${alertId}?clinician_id=${patientId}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ status: "acknowledged" }),
      });
      if (!response.ok) throw new Error('Failed to acknowledge alert');
      return response.json();
    },
    onSuccess: () => {
      toast({
        title: "Alert Acknowledged",
        description: "The alert has been marked as reviewed.",
      });
      refetchAlerts();
      refetchOverview();
    },
  });

  const dismissMutation = useMutation({
    mutationFn: async ({ alertId, reason, notes }: { alertId: string; reason: string; notes?: string }) => {
      const response = await fetch(`/api/ai-health-alerts/alerts/${alertId}?clinician_id=${patientId}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ 
          status: "dismissed", 
          dismiss_reason: reason,
          clinician_notes: notes 
        }),
      });
      if (!response.ok) throw new Error('Failed to dismiss alert');
      return response.json();
    },
    onSuccess: () => {
      toast({
        title: "Alert Dismissed",
        description: "The alert has been dismissed with your notes.",
      });
      refetchAlerts();
      refetchOverview();
    },
  });

  const handleComputeMetrics = async () => {
    setIsComputing(true);
    try {
      const response = await fetch(`/api/ai-health-alerts/v2/compute-all/${patientId}`, {
        method: "POST",
      });
      if (!response.ok) throw new Error('Failed to compute metrics');
      
      toast({
        title: "Metrics Computed",
        description: "All health metrics including DPI and organ scores have been recalculated.",
      });
      
      refetchOverview();
      refetchAlerts();
      refetchDpi();
      refetchOrgans();
      refetchRiskScore();
      refetchBaseline();
      queryClient.invalidateQueries({ queryKey: ['/api/ai-health-alerts'] });
      queryClient.invalidateQueries({ queryKey: ['/api/v1/risk'] });
      queryClient.invalidateQueries({ queryKey: ['/api/v1/baseline'] });
    } catch (error) {
      toast({
        title: "Computation Failed",
        description: "Unable to compute metrics. Please try again.",
        variant: "destructive",
      });
    } finally {
      setIsComputing(false);
    }
  };

  const activeAlerts = useMemo(() => {
    return (alerts || []).filter(a => a.status === "new" || a.status === "acknowledged");
  }, [alerts]);

  const dismissedAlerts = useMemo(() => {
    return (alerts || []).filter(a => a.status === "dismissed");
  }, [alerts]);

  const criticalAlerts = activeAlerts.filter(a => a.severity === "critical");
  const highAlerts = activeAlerts.filter(a => a.severity === "high");

  const latestEngagement = engagementMetrics?.[0];
  const latestQol = qolMetrics?.[0];

  const isLoading = loadingOverview || loadingAlerts || loadingTrends || loadingEngagement || loadingQol || loadingDpi || loadingOrgans;

  if (isLoading) {
    return (
      <div className="container mx-auto p-6 max-w-7xl space-y-6">
        <Skeleton className="h-12 w-96" />
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <Skeleton className="h-24" />
          <Skeleton className="h-24" />
          <Skeleton className="h-24" />
          <Skeleton className="h-24" />
        </div>
        <Skeleton className="h-96" />
      </div>
    );
  }

  return (
    <div className="container mx-auto p-6 max-w-7xl space-y-6">
      <div className="flex items-start justify-between gap-4 flex-wrap">
        <div className="space-y-2">
          <h1 className="text-4xl font-bold tracking-tight text-foreground flex items-center gap-3" data-testid="text-page-title">
            <Bell className="h-10 w-10 text-primary" />
            AI Health Alert Engine
          </h1>
          <p className="text-muted-foreground leading-relaxed">
            Comprehensive trend detection, engagement monitoring, and quality of life insights
          </p>
        </div>
        <Button 
          onClick={handleComputeMetrics} 
          disabled={isComputing}
          data-testid="button-compute-metrics"
        >
          <RefreshCw className={`h-4 w-4 mr-2 ${isComputing ? 'animate-spin' : ''}`} />
          {isComputing ? 'Computing...' : 'Recompute Metrics'}
        </Button>
      </div>

      <LegalDisclaimer />

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4">
        <StatCard
          title="Health Index"
          value={dpiData?.dpi_normalized?.toFixed(0) || "—"}
          subtitle={dpiData?.dpi_bucket ? `Status: ${dpiData.dpi_bucket}` : "No data"}
          trend={dpiData?.explainability?.trend_direction === "improving" ? 'up' : dpiData?.explainability?.trend_direction === "declining" ? 'down' : 'stable'}
          icon={Gauge}
          color={dpiData?.dpi_bucket === "red" ? "text-red-500" : dpiData?.dpi_bucket === "orange" ? "text-orange-500" : dpiData?.dpi_bucket === "yellow" ? "text-yellow-500" : "text-green-500"}
          testId="stat-dpi"
        />
        <StatCard
          title="Active Alerts"
          value={activeAlerts.length}
          subtitle={criticalAlerts.length > 0 ? `${criticalAlerts.length} critical` : "None critical"}
          trend={activeAlerts.length > 5 ? 'up' : 'stable'}
          icon={Bell}
          color={criticalAlerts.length > 0 ? "text-rose-500" : "text-primary"}
          testId="stat-active-alerts"
        />
        <StatCard
          title="Wellness Index"
          value={latestQol?.wellness_index?.toFixed(0) || "—"}
          subtitle={latestQol?.wellness_trend || "No data"}
          trend={latestQol?.wellness_trend === "improving" ? 'up' : latestQol?.wellness_trend === "declining" ? 'down' : 'stable'}
          icon={Heart}
          color="text-rose-500"
          testId="stat-wellness-index"
        />
        <StatCard
          title="Adherence Score"
          value={latestEngagement?.adherence_score ? `${latestEngagement.adherence_score.toFixed(0)}%` : "—"}
          subtitle={`${latestEngagement?.current_streak || 0} day streak`}
          trend={latestEngagement?.engagement_trend === "improving" ? 'up' : latestEngagement?.engagement_trend === "declining" ? 'down' : 'stable'}
          icon={Target}
          color="text-blue-500"
          testId="stat-adherence"
        />
        <StatCard
          title="Stability Score"
          value={latestQol?.stability_score?.toFixed(0) || "—"}
          subtitle="Daily consistency"
          trend={latestQol?.stability_score && latestQol.stability_score > 70 ? 'up' : latestQol?.stability_score && latestQol.stability_score < 50 ? 'down' : 'stable'}
          icon={Activity}
          color="text-green-500"
          testId="stat-stability"
        />
      </div>

      <Card className="border-primary/30 bg-gradient-to-r from-primary/5 to-primary/10" data-testid="card-ml-insights-cta">
        <CardContent className="flex items-center justify-between py-4 flex-wrap gap-4">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-primary/20">
              <Brain className="h-6 w-6 text-primary" />
            </div>
            <div>
              <h4 className="font-semibold">Advanced ML Insights</h4>
              <p className="text-sm text-muted-foreground">
                Deep-dive into disease risk, deterioration predictions, time-series forecasts, and patient segmentation
              </p>
            </div>
          </div>
          <Button 
            asChild
            data-testid="button-view-ml-insights"
          >
            <Link href="/ml-insights">
              <Sparkles className="h-4 w-4 mr-2" />
              View ML Insights
            </Link>
          </Button>
        </CardContent>
      </Card>

      <Tabs defaultValue="alerts" className="w-full">
        <TabsList className="grid w-full grid-cols-6">
          <TabsTrigger value="alerts" className="flex items-center gap-2" data-testid="tab-alerts">
            <Bell className="h-4 w-4" />
            Alerts {activeAlerts.length > 0 && <Badge variant="destructive" className="ml-1">{activeAlerts.length}</Badge>}
          </TabsTrigger>
          <TabsTrigger value="devices" className="flex items-center gap-2" data-testid="tab-devices">
            <Activity className="h-4 w-4" />
            Device Health
          </TabsTrigger>
          <TabsTrigger value="risk" className="flex items-center gap-2" data-testid="tab-risk">
            <Shield className="h-4 w-4" />
            Risk Score
            {riskScore && riskScore.level === 'urgent' && (
              <Badge variant="destructive" className="ml-1">{riskScore.score}</Badge>
            )}
          </TabsTrigger>
          <TabsTrigger value="trends" className="flex items-center gap-2" data-testid="tab-trends">
            <LineChart className="h-4 w-4" />
            Trends
          </TabsTrigger>
          <TabsTrigger value="engagement" className="flex items-center gap-2" data-testid="tab-engagement">
            <Target className="h-4 w-4" />
            Engagement
          </TabsTrigger>
          <TabsTrigger value="wellness" className="flex items-center gap-2" data-testid="tab-wellness">
            <Heart className="h-4 w-4" />
            Quality of Life
          </TabsTrigger>
        </TabsList>

        <TabsContent value="alerts" className="space-y-4 mt-4">
          {criticalAlerts.length > 0 && (
            <Alert variant="destructive">
              <AlertTriangle className="h-4 w-4" />
              <AlertTitle>Critical Alerts Require Attention</AlertTitle>
              <AlertDescription>
                You have {criticalAlerts.length} critical alert(s) that should be reviewed immediately.
              </AlertDescription>
            </Alert>
          )}

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
            <div className="lg:col-span-2 space-y-4">
              <Card data-testid="card-pending-alerts">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Bell className="h-5 w-5 text-primary" />
                    Active Alerts
                    {activeAlerts.length > 0 && (
                      <Badge variant="destructive">{activeAlerts.length}</Badge>
                    )}
                  </CardTitle>
                  <CardDescription>Alerts requiring review or acknowledgment</CardDescription>
                </CardHeader>
                <CardContent>
                  {activeAlerts.length > 0 ? (
                    <div className="space-y-3">
                      {activeAlerts.map((alert) => (
                        <AlertCard
                          key={alert.id}
                          alert={alert}
                          onAcknowledge={(id) => acknowledgeMutation.mutate(id)}
                          onDismiss={(id, reason, notes) => dismissMutation.mutate({ alertId: id, reason, notes })}
                          isUpdating={acknowledgeMutation.isPending || dismissMutation.isPending}
                        />
                      ))}
                    </div>
                  ) : (
                    <div className="text-center py-12" data-testid="empty-alerts">
                      <CheckCircle2 className="h-16 w-16 mx-auto text-green-500 opacity-50 mb-4" />
                      <h3 className="text-lg font-semibold mb-2">All Clear!</h3>
                      <p className="text-muted-foreground">No active alerts at this time. Your health patterns are stable.</p>
                    </div>
                  )}
                </CardContent>
              </Card>
            </div>

            <div className="space-y-4">
              <Card data-testid="card-dpi">
                <CardHeader className="pb-2">
                  <CardTitle className="flex items-center gap-2 text-base">
                    <Gauge className="h-4 w-4 text-primary" />
                    Deterioration Index
                  </CardTitle>
                  <CardDescription className="text-xs">Composite health status indicator</CardDescription>
                </CardHeader>
                <CardContent>
                  <DPIGauge dpi={dpiData || null} />
                </CardContent>
              </Card>

              <Card data-testid="card-organ-scores">
                <CardHeader className="pb-2">
                  <CardTitle className="flex items-center gap-2 text-base">
                    <Activity className="h-4 w-4 text-primary" />
                    System Scores
                  </CardTitle>
                  <CardDescription className="text-xs">Organ-level health indicators</CardDescription>
                </CardHeader>
                <CardContent>
                  <OrganScoresPanel organScores={organScores || null} />
                </CardContent>
              </Card>

              <Card data-testid="card-ml-predictions">
                <CardHeader className="pb-2">
                  <CardTitle className="flex items-center gap-2 text-base">
                    <Sparkles className="h-4 w-4 text-primary" />
                    AI Predictions
                  </CardTitle>
                  <CardDescription className="text-xs">Deep learning deterioration forecasts</CardDescription>
                </CardHeader>
                <CardContent>
                  {loadingPrediction ? (
                    <div className="space-y-3">
                      <Skeleton className="h-24 w-full" />
                      <Skeleton className="h-16 w-full" />
                    </div>
                  ) : (
                    <MLPredictionsPanel prediction={mlPrediction || null} />
                  )}
                </CardContent>
              </Card>
            </div>
          </div>

          <Card data-testid="card-disease-risk-predictions">
            <CardHeader>
              <div className="flex items-center justify-between gap-2 flex-wrap">
                <div className="flex items-center gap-2">
                  <Brain className="h-5 w-5 text-purple-500" />
                  <div>
                    <CardTitle>ML Disease Risk Predictions</CardTitle>
                    <CardDescription>
                      AI-powered predictions with SHAP explainability for disease risk assessment
                    </CardDescription>
                  </div>
                </div>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => refetchDiseaseRisk()}
                  disabled={loadingDiseaseRisk}
                  data-testid="button-refresh-disease-risk"
                >
                  <RefreshCw className={`h-4 w-4 mr-2 ${loadingDiseaseRisk ? 'animate-spin' : ''}`} />
                  Refresh
                </Button>
              </div>
            </CardHeader>
            <CardContent>
              {loadingDiseaseRisk ? (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                  {[1, 2, 3, 4].map((i) => (
                    <Skeleton key={i} className="h-48 w-full" />
                  ))}
                </div>
              ) : diseaseRiskData?.predictions && Object.keys(diseaseRiskData.predictions).length > 0 ? (
                <div className="space-y-4">
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                    {Object.entries(diseaseRiskData.predictions).map(([disease, prediction]) => (
                      <DetailedPredictionCard
                        key={disease}
                        patientId={patientId}
                        disease={disease}
                        prediction={prediction}
                        onRefresh={() => refetchDiseaseRisk()}
                      />
                    ))}
                  </div>
                  <Alert className="bg-blue-50 dark:bg-blue-900/20 border-blue-200">
                    <Info className="h-4 w-4 text-blue-500" />
                    <AlertDescription className="text-xs">
                      Click on any disease card to view detailed SHAP explanations showing how each factor 
                      contributes to your risk score. These are AI-assisted wellness insights, not medical diagnoses.
                    </AlertDescription>
                  </Alert>
                  {diseaseRiskData.predicted_at && (
                    <p className="text-xs text-muted-foreground text-right">
                      Last updated: {format(new Date(diseaseRiskData.predicted_at), "MMM d, yyyy h:mm a")}
                      {diseaseRiskData.model_version && ` • Model v${diseaseRiskData.model_version}`}
                    </p>
                  )}
                </div>
              ) : (
                <div className="text-center py-8" data-testid="empty-disease-risk">
                  <Brain className="h-12 w-12 mx-auto text-muted-foreground opacity-50 mb-4" />
                  <h3 className="text-lg font-semibold mb-2">No Predictions Available</h3>
                  <p className="text-muted-foreground text-sm max-w-md mx-auto">
                    Disease risk predictions require sufficient health data. Continue logging your daily health 
                    information to enable AI-powered risk analysis.
                  </p>
                </div>
              )}
            </CardContent>
          </Card>

          {dismissedAlerts.length > 0 && (
            <Card data-testid="card-dismissed-alerts">
              <CardHeader>
                <CardTitle className="text-lg">Recently Dismissed</CardTitle>
                <CardDescription>Alerts that have been reviewed and dismissed</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  {dismissedAlerts.slice(0, 5).map((alert) => (
                    <div
                      key={alert.id}
                      className="flex items-center justify-between p-3 rounded-lg border bg-muted/50"
                      data-testid={`dismissed-alert-${alert.id}`}
                    >
                      <div className="flex items-center gap-3">
                        <XCircle className="h-4 w-4 text-muted-foreground" />
                        <div>
                          <div className="text-sm font-medium">{alert.title}</div>
                          <div className="text-xs text-muted-foreground">
                            {format(new Date(alert.dismissed_at || alert.created_at), "MMM d, h:mm a")}
                            {alert.dismiss_reason && ` • ${alert.dismiss_reason}`}
                          </div>
                        </div>
                      </div>
                      <Badge variant="outline">{alert.severity}</Badge>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        <TabsContent value="devices" className="space-y-4 mt-4">
          <Card data-testid="card-device-health">
            <CardHeader>
              <div className="flex items-center justify-between gap-2 flex-wrap">
                <div>
                  <CardTitle className="flex items-center gap-2">
                    <Activity className="h-5 w-5 text-primary" />
                    Device Health Analytics
                  </CardTitle>
                  <CardDescription>
                    Health metrics derived from your connected medical devices
                  </CardDescription>
                </div>
                <div className="flex items-center gap-2">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => refetchDeviceAnalytics()}
                    disabled={loadingDeviceAnalytics}
                    data-testid="button-refresh-device-analytics"
                  >
                    <RefreshCw className={`h-4 w-4 mr-1 ${loadingDeviceAnalytics ? 'animate-spin' : ''}`} />
                    Refresh
                  </Button>
                  <Button asChild size="sm" data-testid="button-manage-devices">
                    <Link href="/device-connect">
                      <Link2 className="h-4 w-4 mr-1" />
                      Manage Devices
                    </Link>
                  </Button>
                </div>
              </div>
            </CardHeader>
            <CardContent>
              <DeviceHealthSectionPanel 
                analytics={deviceAnalytics || null} 
                isLoading={loadingDeviceAnalytics} 
              />
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="risk" className="space-y-4 mt-4">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
            <div className="lg:col-span-2 space-y-4">
              <Card data-testid="card-risk-score">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Shield className="h-5 w-5 text-primary" />
                    Composite Risk Score
                  </CardTitle>
                  <CardDescription>
                    Health change indicator based on baseline deviations (0-15 scale)
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <RiskScoreGauge riskScore={riskScore || null} isLoading={loadingRiskScore} />
                </CardContent>
              </Card>

              <Card data-testid="card-risk-history">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <BarChart3 className="h-5 w-5 text-primary" />
                    Risk Score History (7 Days)
                  </CardTitle>
                  <CardDescription>
                    Track how your risk score has changed over time
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <RiskHistoryChart history={riskHistory || []} isLoading={loadingRiskHistory} />
                </CardContent>
              </Card>
            </div>

            <div className="space-y-4">
              <Card data-testid="card-baseline">
                <CardHeader className="pb-2">
                  <CardTitle className="flex items-center gap-2 text-base">
                    <Target className="h-4 w-4 text-primary" />
                    Your Health Baseline
                  </CardTitle>
                  <CardDescription className="text-xs">
                    7-day rolling average reference values
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <BaselinePanel 
                    baseline={baseline || null} 
                    isLoading={loadingBaseline}
                    onRecalculate={() => recalculateBaselineMutation.mutate()}
                    isRecalculating={recalculateBaselineMutation.isPending}
                  />
                </CardContent>
              </Card>

              {baselineHistory && baselineHistory.length > 1 && (
                <Card data-testid="card-baseline-history">
                  <CardHeader className="pb-2">
                    <CardTitle className="flex items-center gap-2 text-base">
                      <History className="h-4 w-4 text-primary" />
                      Baseline History
                    </CardTitle>
                    <CardDescription className="text-xs">
                      How your baseline has evolved
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-2">
                      {baselineHistory.slice(0, 5).map((b, index) => (
                        <div 
                          key={b.id}
                          className={`p-2 rounded-lg border text-xs ${index === 0 ? 'bg-primary/5 border-primary/20' : 'bg-muted/30'}`}
                          data-testid={`baseline-history-item-${b.id}`}
                        >
                          <div className="flex items-center justify-between gap-2 flex-wrap">
                            <div className="flex items-center gap-2">
                              <span className="font-medium">
                                {format(new Date(b.baseline_start_date), 'MMM d')} - {format(new Date(b.baseline_end_date), 'MMM d')}
                              </span>
                              {index === 0 && (
                                <Badge variant="outline" className="text-xs py-0 px-1">
                                  Current
                                </Badge>
                              )}
                            </div>
                            <div className="flex items-center gap-2">
                              <span className="text-muted-foreground">{b.data_points_count} pts</span>
                              {b.baseline_quality && (
                                <Badge 
                                  variant="outline" 
                                  className={`text-xs py-0 px-1 ${
                                    b.baseline_quality === 'excellent' ? 'text-green-600 border-green-300' :
                                    b.baseline_quality === 'good' ? 'text-blue-600 border-blue-300' :
                                    b.baseline_quality === 'fair' ? 'text-yellow-600 border-yellow-300' :
                                    'text-red-600 border-red-300'
                                  }`}
                                >
                                  {b.baseline_quality}
                                </Badge>
                              )}
                            </div>
                          </div>
                          <div className="grid grid-cols-2 gap-2 mt-1 text-muted-foreground">
                            <span>RR: {b.respiratory_rate?.mean?.toFixed(1) || '—'} bpm</span>
                            <span>Pain: {b.pain_facial?.mean?.toFixed(1) || '—'}/10</span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              )}

              <Card data-testid="card-risk-info">
                <CardHeader className="pb-2">
                  <CardTitle className="flex items-center gap-2 text-base">
                    <Info className="h-4 w-4 text-blue-500" />
                    How Risk Scoring Works
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div className="space-y-2 text-xs text-muted-foreground">
                    <p>Your risk score is calculated based on deviations from your personal baseline:</p>
                    <ul className="list-disc list-inside space-y-1">
                      <li><strong>Z-Score Analysis:</strong> Measures how far current values deviate from your normal range</li>
                      <li><strong>Weighted Scoring:</strong> Different metrics carry different weights based on clinical significance</li>
                      <li><strong>24-Hour Window:</strong> Uses the most recent 24 hours of data</li>
                    </ul>
                  </div>
                  <div className="space-y-2">
                    <p className="text-xs font-medium text-muted-foreground">Score Interpretation:</p>
                    <div className="space-y-1">
                      <div className="flex items-center gap-2 text-xs">
                        <div className="w-3 h-3 rounded-full bg-green-500" />
                        <span><strong>0-2 (Stable):</strong> Your patterns are within normal range</span>
                      </div>
                      <div className="flex items-center gap-2 text-xs">
                        <div className="w-3 h-3 rounded-full bg-yellow-500" />
                        <span><strong>3-5 (Monitoring):</strong> Some changes detected, increased awareness recommended</span>
                      </div>
                      <div className="flex items-center gap-2 text-xs">
                        <div className="w-3 h-3 rounded-full bg-red-500" />
                        <span><strong>6-15 (Urgent):</strong> Significant changes, discuss with healthcare provider</span>
                      </div>
                    </div>
                  </div>
                  <Alert className="bg-blue-50 dark:bg-blue-900/20 border-blue-200">
                    <Info className="h-4 w-4 text-blue-500" />
                    <AlertDescription className="text-xs">
                      This is a wellness monitoring tool for change detection only. 
                      It is NOT a medical diagnosis. Always consult your healthcare provider 
                      for medical advice.
                    </AlertDescription>
                  </Alert>
                </CardContent>
              </Card>
            </div>
          </div>
        </TabsContent>

        <TabsContent value="trends" className="space-y-4 mt-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <Card data-testid="card-trend-chart">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <LineChart className="h-5 w-5" />
                  Symptom Trends (14 Days)
                </CardTitle>
                <CardDescription>Z-score analysis and rolling slopes</CardDescription>
              </CardHeader>
              <CardContent>
                <TrendMetricsChart metrics={trendMetrics || []} />
              </CardContent>
            </Card>

            <Card data-testid="card-z-scores">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <BarChart3 className="h-5 w-5" />
                  Current Z-Scores
                </CardTitle>
                <CardDescription>Deviation from 14-day baseline</CardDescription>
              </CardHeader>
              <CardContent>
                {trendMetrics && trendMetrics.length > 0 ? (
                  <div className="space-y-3">
                    {Array.from(new Set(trendMetrics.map(m => m.metric_name))).slice(0, 6).map(metricName => {
                      const metric = trendMetrics.find(m => m.metric_name === metricName);
                      if (!metric) return null;
                      
                      const zScore = metric.z_score || 0;
                      const absZ = Math.abs(zScore);
                      const color = absZ >= 2.5 ? "bg-rose-500" : absZ >= 2 ? "bg-orange-500" : absZ >= 1 ? "bg-yellow-500" : "bg-green-500";
                      
                      return (
                        <div key={metricName} className="space-y-1">
                          <div className="flex justify-between text-sm">
                            <span className="capitalize">{metricName.replace('_', ' ')}</span>
                            <span className={`font-mono ${absZ >= 2.5 ? 'text-rose-500' : absZ >= 2 ? 'text-orange-500' : ''}`}>
                              z={zScore?.toFixed(2)}
                            </span>
                          </div>
                          <Progress 
                            value={Math.min(absZ * 33, 100)} 
                            className="h-2"
                          />
                        </div>
                      );
                    })}
                  </div>
                ) : (
                  <div className="text-center py-8">
                    <BarChart3 className="h-12 w-12 mx-auto text-muted-foreground opacity-50 mb-2" />
                    <p className="text-sm text-muted-foreground">No z-score data available yet</p>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>

          <Card data-testid="card-volatility">
            <CardHeader>
              <CardTitle>Volatility & Slope Analysis</CardTitle>
              <CardDescription>Stability indicators across tracked metrics</CardDescription>
            </CardHeader>
            <CardContent>
              {trendMetrics && trendMetrics.length > 0 ? (
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  {Array.from(new Set(trendMetrics.map(m => m.metric_name))).slice(0, 4).map(metricName => {
                    const metric = trendMetrics.find(m => m.metric_name === metricName);
                    if (!metric) return null;
                    
                    return (
                      <div key={metricName} className="p-4 rounded-lg border bg-card">
                        <h4 className="font-medium text-sm capitalize mb-2">{metricName.replace('_', ' ')}</h4>
                        <div className="space-y-1 text-xs">
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">Volatility:</span>
                            <Badge variant="outline" className="text-xs">
                              {metric.volatility_level || 'N/A'}
                            </Badge>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">7d Slope:</span>
                            <span className={metric.slope_7d && metric.slope_7d < -0.5 ? 'text-rose-500' : ''}>
                              {metric.slope_7d?.toFixed(3) || 'N/A'}
                            </span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">Direction:</span>
                            <span className="flex items-center gap-1">
                              {metric.slope_direction === 'increasing' && <TrendingUp className="h-3 w-3 text-green-500" />}
                              {metric.slope_direction === 'decreasing' && <TrendingDown className="h-3 w-3 text-red-500" />}
                              {metric.slope_direction === 'stable' && <Minus className="h-3 w-3 text-muted-foreground" />}
                              {metric.slope_direction || 'N/A'}
                            </span>
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              ) : (
                <div className="text-center py-8">
                  <Activity className="h-12 w-12 mx-auto text-muted-foreground opacity-50 mb-2" />
                  <p className="text-sm text-muted-foreground">Complete more check-ins to see volatility analysis</p>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="engagement" className="space-y-4 mt-4">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <Card>
              <CardContent className="p-4">
                <div className="flex items-center gap-3">
                  <div className="p-2 rounded-full bg-blue-100 dark:bg-blue-900">
                    <Target className="h-5 w-5 text-blue-500" />
                  </div>
                  <div>
                    <p className="text-sm text-muted-foreground">Check-ins Completed</p>
                    <p className="text-2xl font-bold">{latestEngagement?.checkins_completed || 0}</p>
                  </div>
                </div>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="p-4">
                <div className="flex items-center gap-3">
                  <div className="p-2 rounded-full bg-green-100 dark:bg-green-900">
                    <Sparkles className="h-5 w-5 text-green-500" />
                  </div>
                  <div>
                    <p className="text-sm text-muted-foreground">Current Streak</p>
                    <p className="text-2xl font-bold">{latestEngagement?.current_streak || 0} days</p>
                  </div>
                </div>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="p-4">
                <div className="flex items-center gap-3">
                  <div className="p-2 rounded-full bg-purple-100 dark:bg-purple-900">
                    <Calendar className="h-5 w-5 text-purple-500" />
                  </div>
                  <div>
                    <p className="text-sm text-muted-foreground">Surveys Completed</p>
                    <p className="text-2xl font-bold">{latestEngagement?.surveys_completed || 0}</p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          <Card data-testid="card-engagement-chart">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <BarChart3 className="h-5 w-5" />
                Engagement History
              </CardTitle>
              <CardDescription>Adherence and engagement over time</CardDescription>
            </CardHeader>
            <CardContent>
              <EngagementChart engagementMetrics={engagementMetrics || []} />
            </CardContent>
          </Card>

          {latestEngagement?.engagement_drop_14d && latestEngagement.engagement_drop_14d > 20 && (
            <Alert>
              <AlertTriangle className="h-4 w-4" />
              <AlertTitle>Engagement Drop Detected</AlertTitle>
              <AlertDescription>
                Your engagement has dropped by {latestEngagement.engagement_drop_14d.toFixed(1)}% compared to your 14-day baseline.
                Regular check-ins help maintain accurate health monitoring.
              </AlertDescription>
            </Alert>
          )}
        </TabsContent>

        <TabsContent value="wellness" className="space-y-4 mt-4">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <Card>
              <CardContent className="p-4">
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-muted-foreground">Wellness Index</span>
                    <Heart className="h-4 w-4 text-rose-500" />
                  </div>
                  <Progress value={latestQol?.wellness_index || 0} className="h-2" />
                  <p className="text-xl font-bold">{latestQol?.wellness_index?.toFixed(0) || 0}/100</p>
                </div>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="p-4">
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-muted-foreground">Functional Status</span>
                    <Activity className="h-4 w-4 text-blue-500" />
                  </div>
                  <Progress value={latestQol?.functional_status || 0} className="h-2" />
                  <p className="text-xl font-bold">{latestQol?.functional_status?.toFixed(0) || 0}/100</p>
                </div>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="p-4">
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-muted-foreground">Self-Care Score</span>
                    <Shield className="h-4 w-4 text-green-500" />
                  </div>
                  <Progress value={latestQol?.selfcare_score || 0} className="h-2" />
                  <p className="text-xl font-bold">{latestQol?.selfcare_score?.toFixed(0) || 0}/100</p>
                </div>
              </CardContent>
            </Card>
            <Card>
              <CardContent className="p-4">
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-muted-foreground">Stability Score</span>
                    <Brain className="h-4 w-4 text-purple-500" />
                  </div>
                  <Progress value={latestQol?.stability_score || 0} className="h-2" />
                  <p className="text-xl font-bold">{latestQol?.stability_score?.toFixed(0) || 0}/100</p>
                </div>
              </CardContent>
            </Card>
          </div>

          <Card data-testid="card-qol-chart">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <LineChart className="h-5 w-5" />
                Quality of Life Trends
              </CardTitle>
              <CardDescription>Wellness, functional status, and self-care over time</CardDescription>
            </CardHeader>
            <CardContent>
              <QoLChart qolMetrics={qolMetrics || []} />
            </CardContent>
          </Card>

          {latestQol?.behavior_patterns && (
            <Card data-testid="card-behavior-patterns">
              <CardHeader>
                <CardTitle>Behavioral Pattern Indicators</CardTitle>
                <CardDescription>Statistical patterns observed in your data (NOT medical diagnoses)</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  {Object.entries(latestQol.behavior_patterns).map(([key, value]) => (
                    <div key={key} className="p-4 rounded-lg border bg-card">
                      <h4 className="text-sm font-medium capitalize mb-2">
                        {key.replace(/([A-Z])/g, ' $1').replace('Like Pattern', '').trim()}
                      </h4>
                      <div className="space-y-2">
                        <Progress value={value as number} className="h-2" />
                        <p className="text-lg font-bold">{(value as number).toFixed(0)}/100</p>
                      </div>
                    </div>
                  ))}
                </div>
                <p className="text-xs text-muted-foreground mt-4 italic">
                  These are observational statistical patterns and should not be interpreted as medical indicators.
                  Always consult your healthcare provider for clinical interpretation.
                </p>
              </CardContent>
            </Card>
          )}
        </TabsContent>
      </Tabs>

      <Alert>
        <Info className="h-4 w-4" />
        <AlertTitle>About the AI Health Alert Engine</AlertTitle>
        <AlertDescription>
          <p className="mb-2">
            This system analyzes your health data patterns using statistical methods including:
          </p>
          <ul className="list-disc list-inside ml-4 space-y-1 text-sm">
            <li><strong>Deterioration Index (DPI):</strong> Composite score combining all organ system indicators</li>
            <li><strong>Organ System Scoring:</strong> Individual scores for Respiratory, Cardio/Fluid, Hepatic, Mobility, and Cognitive systems</li>
            <li><strong>Z-Score Analysis:</strong> Detects deviations from your personal 14-day baseline</li>
            <li><strong>Rolling Slopes:</strong> Identifies trends over 3, 7, and 14-day windows</li>
            <li><strong>Color-Coded Buckets:</strong> Green (stable), Yellow (elevated), Orange (warning), Red (critical)</li>
          </ul>
          <p className="mt-3 text-sm font-semibold">
            All alerts are observational pattern notifications — NOT medical diagnoses or emergency alerts.
            For urgent health concerns, contact your healthcare provider or emergency services.
          </p>
        </AlertDescription>
      </Alert>
    </div>
  );
}
