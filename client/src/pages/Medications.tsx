import { useState } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Progress } from "@/components/ui/progress";
import { Slider } from "@/components/ui/slider";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Skeleton } from "@/components/ui/skeleton";
import { DrugInteractionAlert } from "@/components/DrugInteractionAlert";
import { useToast } from "@/hooks/use-toast";
import { apiRequest, queryClient } from "@/lib/queryClient";
import { useAuth } from "@/contexts/AuthContext";
import {
  Pill,
  AlertTriangle,
  TrendingUp,
  TrendingDown,
  Clock,
  Activity,
  Sparkles,
  RefreshCw,
  Info,
  ChevronRight,
  Check,
  Settings,
  Sliders,
  CheckCircle2,
  XCircle,
  AlertCircle,
  Calendar,
  Search,
  Loader2,
  Shield,
  FileText,
  Plus,
  Users,
  Infinity,
  Timer,
  PlayCircle,
  PauseCircle,
  Stethoscope,
  Ban,
} from "lucide-react";
import { format, differenceInDays, addDays } from "date-fns";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
  DialogFooter,
} from "@/components/ui/dialog";

// Type definitions
interface Correlation {
  id: number;
  medication_name: string;
  symptom_name: string;
  correlation_strength: string;
  confidence_score: number;
  time_to_onset_hours: number;
  symptom_onset_date: string;
  medication_change_date: string;
  temporal_pattern: string | null;
  ai_reasoning: string | null;
  patient_impact: string;
  action_recommended: string;
  analysis_date: string;
}

interface MedicationGroup {
  medication_name: string;
  dosage: string;
  total_correlations: number;
  strong_correlations: number;
  correlations: Correlation[];
}

interface EffectsSummary {
  patient_id: string;
  analysis_period_days: number;
  total_correlations_found: number;
  strong_correlations_count: number;
  medications_analyzed: number;
  medication_groups: MedicationGroup[];
  summary_generated_at: string;
  recommendations: string;
}

interface AdherenceTrendPoint {
  date: string;
  adherenceRate: number;
}

interface RegimenRisk {
  level: "low" | "moderate" | "high" | "unknown";
  rationale: string;
}

interface MissedDoseEscalation {
  count: number;
  severity: "none" | "warning" | "critical";
}

interface MedicationAdherenceData {
  currentAdherenceRate: number | null;
  sevenDayTrend: AdherenceTrendPoint[];
  regimenRisk: RegimenRisk;
  missedDoseEscalation: MissedDoseEscalation;
}

// Chronic care medication types
interface MedicationConflict {
  id: string;
  patientId: string;
  medication1Id: string;
  medication2Id: string;
  doctor1Id: string;
  doctor2Id: string;
  specialty1: string;
  specialty2: string;
  conflictType: string;
  severity: string;
  description: string;
  status: 'pending' | 'resolved';
  doctor1Response?: string;
  doctor2Response?: string;
  resolution?: string;
  createdAt: string;
}

interface ChronicMedication {
  id: string;
  name: string;
  dosage: string;
  frequency: string;
  specialty?: string;
  isContinuous?: boolean;
  durationDays?: number;
  intendedStartDate?: string;
  actualStartDate?: string;
  computedEndDate?: string;
  status: string;
  conflictStatus?: string;
  prescribedBy?: string;
  prescribedByName?: string;
  active: boolean;
}

// Helper function to calculate remaining days
function getRemainingDays(endDate: string | undefined | null): number | null {
  if (!endDate) return null;
  const end = new Date(endDate);
  const today = new Date();
  return differenceInDays(end, today);
}

// Helper function to format duration display
function formatDuration(med: ChronicMedication): string {
  if (med.isContinuous) {
    return "Continuous";
  }
  
  const remaining = getRemainingDays(med.computedEndDate);
  if (remaining === null) {
    return med.durationDays ? `${med.durationDays} day course` : "Duration not set";
  }
  
  if (remaining < 0) {
    return "Completed";
  }
  
  if (remaining === 0) {
    return "Last day";
  }
  
  return `${remaining} day${remaining !== 1 ? 's' : ''} left`;
}

// Specialty display helper
function getSpecialtyColor(specialty: string): string {
  const colors: Record<string, string> = {
    cardiology: "bg-red-100 text-red-700 dark:bg-red-900 dark:text-red-300",
    oncology: "bg-purple-100 text-purple-700 dark:bg-purple-900 dark:text-purple-300",
    neurology: "bg-blue-100 text-blue-700 dark:bg-blue-900 dark:text-blue-300",
    rheumatology: "bg-amber-100 text-amber-700 dark:bg-amber-900 dark:text-amber-300",
    immunology: "bg-green-100 text-green-700 dark:bg-green-900 dark:text-green-300",
    endocrinology: "bg-cyan-100 text-cyan-700 dark:bg-cyan-900 dark:text-cyan-300",
    gastroenterology: "bg-orange-100 text-orange-700 dark:bg-orange-900 dark:text-orange-300",
    pulmonology: "bg-sky-100 text-sky-700 dark:bg-sky-900 dark:text-sky-300",
    nephrology: "bg-teal-100 text-teal-700 dark:bg-teal-900 dark:text-teal-300",
    psychiatry: "bg-indigo-100 text-indigo-700 dark:bg-indigo-900 dark:text-indigo-300",
    "general medicine": "bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300",
    unspecified: "bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300",
  };
  return colors[specialty?.toLowerCase()] || colors.unspecified;
}

export default function Medications() {
  const { toast } = useToast();
  const { user } = useAuth();
  
  // Side Effects state
  const [selectedMedicationIdx, setSelectedMedicationIdx] = useState<number>(0);
  const [daysBack, setDaysBack] = useState(90);
  const [minConfidence, setMinConfidence] = useState(0.4);
  const [showAnalysisConfig, setShowAnalysisConfig] = useState(false);

  // Interaction Checker state
  const [drugToCheck, setDrugToCheck] = useState("");
  const [checking, setChecking] = useState(false);
  const [checkResults, setCheckResults] = useState<any>(null);

  // Fetch medication effects summary
  const { data: effectsSummary, isLoading: effectsLoading } = useQuery<EffectsSummary>({
    queryKey: [`/api/v1/medication-side-effects/summary/me`, { days_back: daysBack }],
    queryFn: async () => {
      const res = await fetch(`/api/v1/medication-side-effects/summary/me?days_back=${daysBack}`, {
        credentials: 'include'
      });
      if (!res.ok) throw new Error("Failed to load medication effects");
      return res.json();
    }
  });

  // Fetch adherence data
  const { data: adherenceData, isLoading: adherenceLoading } = useQuery<MedicationAdherenceData>({
    queryKey: [`/python/api/v1/behavior-ai/medication-adherence/${user?.id}`],
    enabled: !!user?.id,
  });

  // Fetch interaction alerts
  const { data: activeAlerts, isLoading: alertsLoading } = useQuery<any[]>({
    queryKey: ['/api/drug-interactions/alerts'],
  });

  const { data: allAlerts } = useQuery<any[]>({
    queryKey: ['/api/drug-interactions/alerts/all'],
  });

  // Mutations
  const analyzeCorrelations = useMutation({
    mutationFn: async (params: { days_back: number; min_confidence: number }) => {
      return await apiRequest(`/api/v1/medication-side-effects/analyze/me`, {
        method: "POST",
        body: JSON.stringify(params)
      });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ 
        queryKey: [`/api/v1/medication-side-effects/summary/me`],
        exact: false,
        refetchType: 'active'
      });
      toast({
        title: "Analysis Complete",
        description: "Your medication effects have been analyzed successfully."
      });
    },
    onError: (error: any) => {
      toast({
        title: "Analysis Failed",
        description: error.message || "Failed to analyze correlations. Please try again.",
        variant: "destructive"
      });
    }
  });

  const acknowledgeMutation = useMutation({
    mutationFn: async (alertId: string) => {
      return apiRequest('POST', `/api/drug-interactions/alerts/${alertId}/acknowledge`, {});
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['/api/drug-interactions/alerts'] });
      queryClient.invalidateQueries({ queryKey: ['/api/drug-interactions/alerts/all'] });
      toast({
        title: "Alert acknowledged",
        description: "You've acknowledged this drug interaction warning.",
      });
    },
  });

  const handleCheckDrug = async () => {
    if (!drugToCheck.trim()) {
      toast({
        title: "Drug name required",
        description: "Please enter a medication name to check for interactions.",
        variant: "destructive",
      });
      return;
    }

    setChecking(true);
    try {
      const result: any = await apiRequest('POST', '/api/drug-interactions/analyze', {
        drugName: drugToCheck,
      });
      setCheckResults(result);
      
      if (result.hasBlockingInteraction) {
        toast({
          title: "Severe interaction detected",
          description: "This medication has a severe interaction with your current medications.",
          variant: "destructive",
        });
      } else if (result.hasInteractions) {
        toast({
          title: "Interactions found",
          description: `Found ${result.interactions.length} potential interaction(s).`,
        });
      } else {
        toast({
          title: "No interactions found",
          description: "This medication appears safe to take with your current medications.",
        });
      }
    } catch (error) {
      toast({
        title: "Analysis failed",
        description: "Failed to analyze drug interactions. Please try again.",
        variant: "destructive",
      });
    } finally {
      setChecking(false);
    }
  };

  // Helper functions
  const getStrengthBadge = (strength: string) => {
    switch (strength.toUpperCase()) {
      case "STRONG":
        return <Badge className="bg-rose-500 hover:bg-rose-600">
          <AlertTriangle className="w-3 h-3 mr-1" />
          Strong
        </Badge>;
      case "LIKELY":
        return <Badge className="bg-coral-500 hover:bg-coral-600">
          <TrendingUp className="w-3 h-3 mr-1" />
          Likely
        </Badge>;
      case "POSSIBLE":
        return <Badge variant="secondary">Possible</Badge>;
      default:
        return <Badge variant="outline">Weak</Badge>;
    }
  };

  const getRiskColor = (level: string) => {
    switch (level) {
      case "high":
        return "destructive";
      case "moderate":
        return "default";
      case "low":
        return "secondary";
      default:
        return "outline";
    }
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case "critical":
        return "destructive";
      case "warning":
        return "default";
      default:
        return "secondary";
    }
  };

  const formatAdherenceRate = (rate: number | null) => {
    if (rate === null) return "No data";
    return `${(rate * 100).toFixed(1)}%`;
  };

  const severeAlertsCount = activeAlerts?.filter(a => a.severityLevel === 'severe').length || 0;
  const moderateAlertsCount = activeAlerts?.filter(a => a.severityLevel === 'moderate').length || 0;

  // Get all unique medications across features
  const allMedications = effectsSummary?.medication_groups.map(g => g.medication_name) || [];

  return (
    <div className="h-full overflow-auto p-6">
      <div className="max-w-7xl mx-auto space-y-6">
        <div>
          <h1 className="text-3xl font-bold mb-2">Medication Management</h1>
          <p className="text-muted-foreground">
            Comprehensive AI-powered medication tracking, adherence monitoring, side effect analysis, and interaction detection
          </p>
        </div>

        {/* Summary Cards */}
        <div className="grid gap-4 md:grid-cols-4">
          <Card>
            <CardHeader className="flex flex-row items-center justify-between gap-2 space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Active Medications</CardTitle>
              <Pill className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold" data-testid="count-medications">
                {allMedications.length}
              </div>
              <p className="text-xs text-muted-foreground">Currently tracked</p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between gap-2 space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Adherence Rate</CardTitle>
              <CheckCircle2 className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold" data-testid="adherence-rate">
                {adherenceLoading ? "..." : formatAdherenceRate(adherenceData?.currentAdherenceRate ?? null)}
              </div>
              <p className="text-xs text-muted-foreground">Last 7 days</p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between gap-2 space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Side Effects</CardTitle>
              <Activity className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold" data-testid="count-side-effects">
                {effectsSummary?.strong_correlations_count || 0}
              </div>
              <p className="text-xs text-muted-foreground">Strong correlations</p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between gap-2 space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Interaction Alerts</CardTitle>
              <AlertTriangle className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold" data-testid="count-alerts">
                {activeAlerts?.length || 0}
              </div>
              <p className="text-xs text-muted-foreground">
                {severeAlertsCount > 0 ? `${severeAlertsCount} severe` : "All safe"}
              </p>
            </CardContent>
          </Card>
        </div>

        <Tabs defaultValue="overview" className="space-y-6">
          <TabsList className="grid w-full max-w-4xl grid-cols-6">
            <TabsTrigger value="overview" data-testid="tab-overview">
              <Pill className="h-4 w-4 mr-2" />
              Overview
            </TabsTrigger>
            <TabsTrigger value="timeline" data-testid="tab-timeline">
              <Clock className="h-4 w-4 mr-2" />
              Timeline
            </TabsTrigger>
            <TabsTrigger value="adherence" data-testid="tab-adherence">
              <Calendar className="h-4 w-4 mr-2" />
              Adherence
            </TabsTrigger>
            <TabsTrigger value="side-effects" data-testid="tab-side-effects">
              <Activity className="h-4 w-4 mr-2" />
              Side Effects
            </TabsTrigger>
            <TabsTrigger value="interactions" data-testid="tab-interactions">
              <Shield className="h-4 w-4 mr-2" />
              Interactions
            </TabsTrigger>
            <TabsTrigger value="requests" data-testid="tab-requests">
              <FileText className="h-4 w-4 mr-2" />
              Requests
            </TabsTrigger>
          </TabsList>

          {/* Overview Tab */}
          <TabsContent value="overview" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>Your Medications</CardTitle>
                <CardDescription>
                  Overview of all tracked medications with integrated insights
                </CardDescription>
              </CardHeader>
              <CardContent>
                {effectsLoading ? (
                  <div className="space-y-3">
                    {[1, 2, 3].map(i => <Skeleton key={i} className="h-20 w-full" />)}
                  </div>
                ) : allMedications.length === 0 ? (
                  <div className="text-center py-12 text-muted-foreground">
                    <Pill className="h-12 w-12 mx-auto mb-3 opacity-50" />
                    <p>No medications tracked yet</p>
                    <p className="text-sm mt-1">Start tracking medications to monitor adherence and side effects</p>
                  </div>
                ) : (
                  <div className="space-y-3">
                    {effectsSummary?.medication_groups.map((med, idx) => {
                      const hasInteractions = activeAlerts?.some(
                        a => a.drug1Name === med.medication_name || a.drug2Name === med.medication_name
                      );
                      const hasSideEffects = med.strong_correlations > 0;

                      return (
                        <Card key={idx} className="hover-elevate" data-testid={`card-medication-${idx}`}>
                          <CardHeader className="pb-3">
                            <div className="flex items-start justify-between">
                              <div className="flex-1">
                                <h3 className="font-semibold text-lg">{med.medication_name}</h3>
                                <p className="text-sm text-muted-foreground">{med.dosage}</p>
                              </div>
                              <div className="flex gap-2">
                                {hasInteractions && (
                                  <Badge variant="destructive" className="text-xs">
                                    <AlertTriangle className="h-3 w-3 mr-1" />
                                    Interaction
                                  </Badge>
                                )}
                                {hasSideEffects && (
                                  <Badge variant="secondary" className="text-xs">
                                    <Activity className="h-3 w-3 mr-1" />
                                    {med.strong_correlations} Effects
                                  </Badge>
                                )}
                              </div>
                            </div>
                          </CardHeader>
                          {(hasSideEffects || hasInteractions) && (
                            <CardContent className="pt-0">
                              <div className="space-y-2 text-sm">
                                {hasSideEffects && (
                                  <div className="flex items-center gap-2 text-muted-foreground">
                                    <Activity className="h-4 w-4" />
                                    <span>{med.total_correlations} potential side effects detected</span>
                                  </div>
                                )}
                                {hasInteractions && (
                                  <div className="flex items-center gap-2 text-destructive">
                                    <AlertTriangle className="h-4 w-4" />
                                    <span>Has drug interactions - review Interactions tab</span>
                                  </div>
                                )}
                              </div>
                            </CardContent>
                          )}
                        </Card>
                      );
                    })}
                  </div>
                )}
              </CardContent>
            </Card>

            {/* Quick Actions */}
            <div className="grid gap-4 md:grid-cols-2">
              <Card className="hover-elevate cursor-pointer" onClick={() => document.querySelector('[value="adherence"]')?.dispatchEvent(new MouseEvent('click', { bubbles: true }))}>
                <CardHeader>
                  <CardTitle className="text-lg flex items-center gap-2">
                    <Calendar className="h-5 w-5" />
                    Check Adherence
                  </CardTitle>
                  <CardDescription>
                    View your medication compliance trends
                  </CardDescription>
                </CardHeader>
              </Card>

              <Card className="hover-elevate cursor-pointer" onClick={() => document.querySelector('[value="interactions"]')?.dispatchEvent(new MouseEvent('click', { bubbles: true }))}>
                <CardHeader>
                  <CardTitle className="text-lg flex items-center gap-2">
                    <Shield className="h-5 w-5" />
                    Check New Medication
                  </CardTitle>
                  <CardDescription>
                    Verify safety before adding a new drug
                  </CardDescription>
                </CardHeader>
              </Card>
            </div>
          </TabsContent>

          {/* Adherence Tab */}
          <TabsContent value="adherence" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Pill className="h-5 w-5" />
                  Current Adherence Rate
                </CardTitle>
                <CardDescription>Your medication compliance over the last 7 days</CardDescription>
              </CardHeader>
              <CardContent>
                {adherenceLoading ? (
                  <Skeleton className="h-24 w-full" />
                ) : (
                  <div className="space-y-4">
                    <div className="text-5xl font-bold" data-testid="text-adherence-rate">
                      {formatAdherenceRate(adherenceData?.currentAdherenceRate ?? null)}
                    </div>
                    {adherenceData?.currentAdherenceRate !== null && adherenceData?.currentAdherenceRate !== undefined && (
                      <Progress
                        value={(adherenceData?.currentAdherenceRate ?? 0) * 100}
                        className="h-3"
                      />
                    )}
                    <p className="text-sm text-muted-foreground">
                      {adherenceData?.currentAdherenceRate && adherenceData?.currentAdherenceRate >= 0.9
                        ? "Excellent adherence! Keep it up."
                        : adherenceData?.currentAdherenceRate && adherenceData?.currentAdherenceRate >= 0.7
                        ? "Good adherence, but there's room for improvement."
                        : "Low adherence detected. Consider setting medication reminders."}
                    </p>
                  </div>
                )}
              </CardContent>
            </Card>

            <div className="grid gap-4 md:grid-cols-2">
              <Card>
                <CardHeader>
                  <CardTitle className="text-lg">Regimen Risk</CardTitle>
                  <CardDescription>Interaction risk assessment</CardDescription>
                </CardHeader>
                <CardContent>
                  {adherenceLoading ? (
                    <Skeleton className="h-16 w-full" />
                  ) : (
                    <div className="space-y-3">
                      <Badge variant={getRiskColor(adherenceData?.regimenRisk.level || "unknown") as any} className="text-sm">
                        {adherenceData?.regimenRisk.level.toUpperCase() || "UNKNOWN"} RISK
                      </Badge>
                      <p className="text-sm text-muted-foreground">
                        {adherenceData?.regimenRisk.rationale || "No risk assessment available"}
                      </p>
                    </div>
                  )}
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle className="text-lg">Missed Doses</CardTitle>
                  <CardDescription>Recent tracking period</CardDescription>
                </CardHeader>
                <CardContent>
                  {adherenceLoading ? (
                    <Skeleton className="h-16 w-full" />
                  ) : (
                    <div className="space-y-3">
                      <div className="text-3xl font-bold">
                        {adherenceData?.missedDoseEscalation.count || 0}
                      </div>
                      <Badge variant={getSeverityColor(adherenceData?.missedDoseEscalation.severity || "none") as any}>
                        {adherenceData?.missedDoseEscalation.severity.toUpperCase() || "NONE"}
                      </Badge>
                    </div>
                  )}
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          {/* Side Effects Tab */}
          <TabsContent value="side-effects" className="space-y-6">
            <Card>
              <CardHeader className="flex flex-row items-center justify-between">
                <div>
                  <CardTitle>AI Side Effect Analysis</CardTitle>
                  <CardDescription>
                    Temporal correlation analysis between medications and symptoms
                  </CardDescription>
                </div>
                <Dialog open={showAnalysisConfig} onOpenChange={setShowAnalysisConfig}>
                  <DialogTrigger asChild>
                    <Button variant="outline" size="sm">
                      <Settings className="h-4 w-4 mr-2" />
                      Configure
                    </Button>
                  </DialogTrigger>
                  <DialogContent>
                    <DialogHeader>
                      <DialogTitle>Analysis Configuration</DialogTitle>
                      <DialogDescription>
                        Adjust analysis parameters for medication side effect detection
                      </DialogDescription>
                    </DialogHeader>
                    <div className="space-y-4 py-4">
                      <div className="space-y-2">
                        <Label>Analysis Period: {daysBack} days</Label>
                        <Slider
                          value={[daysBack]}
                          onValueChange={(val) => setDaysBack(val[0])}
                          min={30}
                          max={180}
                          step={30}
                        />
                      </div>
                      <div className="space-y-2">
                        <Label>Minimum Confidence: {(minConfidence * 100).toFixed(0)}%</Label>
                        <Slider
                          value={[minConfidence * 100]}
                          onValueChange={(val) => setMinConfidence(val[0] / 100)}
                          min={20}
                          max={90}
                          step={10}
                        />
                      </div>
                    </div>
                    <DialogFooter>
                      <Button
                        onClick={() => {
                          analyzeCorrelations.mutate({ days_back: daysBack, min_confidence: minConfidence });
                          setShowAnalysisConfig(false);
                        }}
                        disabled={analyzeCorrelations.isPending}
                      >
                        {analyzeCorrelations.isPending ? "Analyzing..." : "Run Analysis"}
                      </Button>
                    </DialogFooter>
                  </DialogContent>
                </Dialog>
              </CardHeader>
              <CardContent>
                {effectsLoading ? (
                  <Skeleton className="h-64 w-full" />
                ) : effectsSummary && effectsSummary.medication_groups.length > 0 ? (
                  <ScrollArea className="h-[600px]">
                    <div className="space-y-4">
                      {effectsSummary.medication_groups.map((med, idx) => (
                        <Card key={idx} className="hover-elevate">
                          <CardHeader>
                            <div className="flex items-start justify-between">
                              <div>
                                <CardTitle className="text-lg">{med.medication_name}</CardTitle>
                                <CardDescription>{med.dosage}</CardDescription>
                              </div>
                              <Badge variant="secondary">
                                {med.total_correlations} correlation{med.total_correlations !== 1 ? 's' : ''}
                              </Badge>
                            </div>
                          </CardHeader>
                          <CardContent className="space-y-3">
                            {med.correlations.map((corr, cIdx) => (
                              <div key={cIdx} className="p-3 border rounded-lg space-y-2">
                                <div className="flex items-start justify-between">
                                  <div className="flex-1">
                                    <p className="font-medium">{corr.symptom_name}</p>
                                    <p className="text-sm text-muted-foreground">
                                      Onset: ~{corr.time_to_onset_hours}h after dose
                                    </p>
                                  </div>
                                  {getStrengthBadge(corr.correlation_strength)}
                                </div>
                                {corr.ai_reasoning && (
                                  <p className="text-sm text-muted-foreground">
                                    <Sparkles className="h-3 w-3 inline mr-1" />
                                    {corr.ai_reasoning}
                                  </p>
                                )}
                                <div className="flex items-center gap-2 text-xs text-muted-foreground">
                                  <span>Confidence: {(corr.confidence_score * 100).toFixed(0)}%</span>
                                  <span>•</span>
                                  <span>{corr.patient_impact}</span>
                                </div>
                              </div>
                            ))}
                          </CardContent>
                        </Card>
                      ))}
                    </div>
                  </ScrollArea>
                ) : (
                  <div className="text-center py-12 text-muted-foreground">
                    <Activity className="h-12 w-12 mx-auto mb-3 opacity-50" />
                    <p>No side effect correlations detected</p>
                    <p className="text-sm mt-1">Continue tracking symptoms to enable AI analysis</p>
                  </div>
                )}
              </CardContent>
            </Card>
          </TabsContent>

          {/* Interactions Tab */}
          <TabsContent value="interactions" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>Check New Medication</CardTitle>
                <CardDescription>
                  Verify safety before adding a new medication to your regimen
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex gap-2">
                  <div className="flex-1">
                    <Label htmlFor="drug-name">Medication Name</Label>
                    <Input
                      id="drug-name"
                      placeholder="e.g., Aspirin, Warfarin..."
                      value={drugToCheck}
                      onChange={(e) => setDrugToCheck(e.target.value)}
                      onKeyDown={(e) => e.key === 'Enter' && handleCheckDrug()}
                      data-testid="input-drug-name"
                    />
                  </div>
                  <div className="flex items-end">
                    <Button
                      onClick={handleCheckDrug}
                      disabled={checking}
                      data-testid="button-check-drug"
                    >
                      {checking ? (
                        <>
                          <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                          Checking...
                        </>
                      ) : (
                        <>
                          <Search className="h-4 w-4 mr-2" />
                          Check Safety
                        </>
                      )}
                    </Button>
                  </div>
                </div>

                {checkResults && (
                  <Alert variant={checkResults.hasBlockingInteraction ? "destructive" : "default"}>
                    <AlertTriangle className="h-4 w-4" />
                    <AlertTitle>
                      {checkResults.hasBlockingInteraction
                        ? "⚠️ Severe Interaction Detected"
                        : checkResults.hasInteractions
                        ? "Interactions Found"
                        : "✓ Safe to Take"}
                    </AlertTitle>
                    <AlertDescription>
                      {checkResults.hasBlockingInteraction
                        ? "Do NOT take this medication with your current regimen. Consult your doctor immediately."
                        : checkResults.hasInteractions
                        ? `Found ${checkResults.interactions.length} potential interaction(s). Review details below.`
                        : "This medication appears safe with your current medications."}
                    </AlertDescription>
                  </Alert>
                )}
              </CardContent>
            </Card>

            {/* Active Alerts */}
            <Card>
              <CardHeader>
                <CardTitle>Active Interaction Alerts</CardTitle>
                <CardDescription>
                  {activeAlerts?.length || 0} alert{activeAlerts?.length !== 1 ? 's' : ''} requiring attention
                </CardDescription>
              </CardHeader>
              <CardContent>
                {alertsLoading ? (
                  <Skeleton className="h-48 w-full" />
                ) : activeAlerts && activeAlerts.length > 0 ? (
                  <div className="space-y-3">
                    {activeAlerts.map((alert) => (
                      <DrugInteractionAlert
                        key={alert.id}
                        alert={alert}
                        onAcknowledge={() => acknowledgeMutation.mutate(alert.id)}
                      />
                    ))}
                  </div>
                ) : (
                  <div className="text-center py-12 text-muted-foreground">
                    <CheckCircle2 className="h-12 w-12 mx-auto mb-3 text-green-500 opacity-50" />
                    <p className="font-medium">No Active Alerts</p>
                    <p className="text-sm mt-1">Your current medications have no known dangerous interactions</p>
                  </div>
                )}
              </CardContent>
            </Card>

            {/* Summary Cards */}
            <div className="grid gap-4 md:grid-cols-3">
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm font-medium">Severe</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold text-destructive">{severeAlertsCount}</div>
                </CardContent>
              </Card>
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm font-medium">Moderate</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">{moderateAlertsCount}</div>
                </CardContent>
              </Card>
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm font-medium">Total Tracked</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">{allAlerts?.length || 0}</div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          {/* Timeline Tab */}
          <TabsContent value="timeline" className="space-y-6">
            <MedicationTimeline />
          </TabsContent>

          {/* Requests Tab */}
          <TabsContent value="requests" className="space-y-6">
            <DosageChangeRequests />
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}

// Medication Timeline Component
function MedicationTimeline() {
  const { toast } = useToast();
  const { data: changelog, isLoading } = useQuery<any[]>({
    queryKey: ['/api/medications/changelog/all'],
  });

  const { data: pendingMeds } = useQuery<any[]>({
    queryKey: ['/api/medications/pending-confirmation'],
  });

  const { data: inactiveMeds } = useQuery<any[]>({
    queryKey: ['/api/medications/inactive'],
  });

  const confirmMutation = useMutation({
    mutationFn: async (id: string) => {
      return apiRequest('POST', `/api/medications/${id}/confirm`, {});
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['/api/medications'] });
      queryClient.invalidateQueries({ queryKey: ['/api/medications/pending-confirmation'] });
      queryClient.invalidateQueries({ queryKey: ['/api/medications/changelog/all'] });
      toast({
        title: "Medication confirmed",
        description: "The medication has been added to your active list.",
      });
    },
  });

  const discontinueMutation = useMutation({
    mutationFn: async ({ id, reason }: { id: string; reason: string }) => {
      return apiRequest('POST', `/api/medications/${id}/discontinue`, { reason });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['/api/medications'] });
      queryClient.invalidateQueries({ queryKey: ['/api/medications/inactive'] });
      queryClient.invalidateQueries({ queryKey: ['/api/medications/changelog/all'] });
      toast({
        title: "Medication discontinued",
        description: "The medication has been marked as inactive.",
      });
    },
  });

  const reactivateMutation = useMutation({
    mutationFn: async (id: string) => {
      return apiRequest('POST', `/api/medications/${id}/reactivate`, {});
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['/api/medications'] });
      queryClient.invalidateQueries({ queryKey: ['/api/medications/inactive'] });
      queryClient.invalidateQueries({ queryKey: ['/api/medications/changelog/all'] });
      toast({
        title: "Medication reactivated",
        description: "The medication has been added back to your active list.",
      });
    },
  });

  const getChangeTypeIcon = (changeType: string) => {
    switch (changeType) {
      case 'added':
        return <Plus className="h-4 w-4 text-green-500" />;
      case 'discontinued':
        return <XCircle className="h-4 w-4 text-red-500" />;
      case 'reactivated':
        return <CheckCircle2 className="h-4 w-4 text-green-500" />;
      case 'dosage_changed':
        return <Sliders className="h-4 w-4 text-blue-500" />;
      default:
        return <Activity className="h-4 w-4" />;
    }
  };

  const getChangeTypeBadge = (changeType: string) => {
    switch (changeType) {
      case 'added':
        return <Badge variant="secondary">Added</Badge>;
      case 'discontinued':
        return <Badge variant="destructive">Discontinued</Badge>;
      case 'reactivated':
        return <Badge variant="secondary">Reactivated</Badge>;
      case 'dosage_changed':
        return <Badge>Dosage Changed</Badge>;
      default:
        return <Badge variant="outline">{changeType}</Badge>;
    }
  };

  return (
    <div className="space-y-6">
      {/* Pending Confirmation */}
      {pendingMeds && pendingMeds.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <AlertCircle className="h-5 w-5 text-amber-500" />
              Pending Confirmation
            </CardTitle>
            <CardDescription>
              These medications were auto-detected from your medical files and need confirmation
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {pendingMeds.map((med) => (
                <Card key={med.id} className="hover-elevate" data-testid={`card-pending-${med.id}`}>
                  <CardContent className="pt-6">
                    <div className="flex items-start justify-between">
                      <div className="flex-1">
                        <h4 className="font-semibold">{med.name}</h4>
                        <p className="text-sm text-muted-foreground">
                          {med.dosage} • {med.frequency}
                        </p>
                        {med.sourceDocumentId && (
                          <p className="text-xs text-muted-foreground mt-1">
                            Detected from medical file
                          </p>
                        )}
                      </div>
                      <div className="flex gap-2">
                        <Button
                          size="sm"
                          onClick={() => confirmMutation.mutate(med.id)}
                          disabled={confirmMutation.isPending}
                          data-testid={`button-confirm-${med.id}`}
                        >
                          <Check className="h-4 w-4 mr-1" />
                          Confirm
                        </Button>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Inactive Medications */}
      {inactiveMeds && inactiveMeds.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <XCircle className="h-5 w-5 text-muted-foreground" />
              Inactive Medications
            </CardTitle>
            <CardDescription>
              Previously discontinued medications
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {inactiveMeds.map((med) => (
                <Card key={med.id} className="opacity-60 hover-elevate" data-testid={`card-inactive-${med.id}`}>
                  <CardContent className="pt-6">
                    <div className="flex items-start justify-between">
                      <div className="flex-1">
                        <h4 className="font-semibold">{med.name}</h4>
                        <p className="text-sm text-muted-foreground">
                          {med.dosage} • {med.frequency}
                        </p>
                        {med.discontinuationReason && (
                          <p className="text-xs text-muted-foreground mt-1">
                            Reason: {med.discontinuationReason}
                          </p>
                        )}
                        {med.discontinuedAt && (
                          <p className="text-xs text-muted-foreground">
                            Discontinued: {format(new Date(med.discontinuedAt), 'MMM d, yyyy')}
                          </p>
                        )}
                      </div>
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={() => reactivateMutation.mutate(med.id)}
                        disabled={reactivateMutation.isPending}
                        data-testid={`button-reactivate-${med.id}`}
                      >
                        <RefreshCw className="h-4 w-4 mr-1" />
                        Reactivate
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Change History */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Clock className="h-5 w-5" />
            Medication History
          </CardTitle>
          <CardDescription>
            Complete timeline of all medication changes (HIPAA audit trail)
          </CardDescription>
        </CardHeader>
        <CardContent>
          {isLoading ? (
            <div className="space-y-3">
              {[1, 2, 3].map(i => <Skeleton key={i} className="h-20 w-full" />)}
            </div>
          ) : changelog && changelog.length > 0 ? (
            <ScrollArea className="h-[500px]">
              <div className="space-y-4">
                {changelog.map((log, idx) => (
                  <div key={log.id} className="flex gap-4" data-testid={`log-entry-${idx}`}>
                    <div className="flex flex-col items-center">
                      <div className="rounded-full bg-muted p-2">
                        {getChangeTypeIcon(log.changeType)}
                      </div>
                      {idx < changelog.length - 1 && (
                        <div className="h-full w-px bg-border mt-2" />
                      )}
                    </div>
                    <div className="flex-1 pb-4">
                      <div className="flex items-start justify-between mb-1">
                        <div className="flex items-center gap-2">
                          {getChangeTypeBadge(log.changeType)}
                          <span className="text-sm text-muted-foreground">
                            {format(new Date(log.changedAt), 'MMM d, yyyy h:mm a')}
                          </span>
                        </div>
                        <Badge variant="outline" className="text-xs">
                          by {log.changedBy}
                        </Badge>
                      </div>
                      <p className="text-sm font-medium">{log.changeReason}</p>
                      {log.changeType === 'dosage_changed' && (
                        <p className="text-sm text-muted-foreground mt-1">
                          {log.oldDosage} → {log.newDosage}
                          {log.oldFrequency !== log.newFrequency && ` • ${log.oldFrequency} → ${log.newFrequency}`}
                        </p>
                      )}
                      {log.notes && (
                        <p className="text-sm text-muted-foreground mt-1 italic">
                          Note: {log.notes}
                        </p>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </ScrollArea>
          ) : (
            <div className="text-center py-12 text-muted-foreground">
              <Clock className="h-12 w-12 mx-auto mb-3 opacity-50" />
              <p>No medication history yet</p>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}

// Dosage Change Requests Component
function DosageChangeRequests() {
  const { toast } = useToast();
  const { user } = useAuth();
  const [showRequestDialog, setShowRequestDialog] = useState(false);
  const [selectedMed, setSelectedMed] = useState<any>(null);
  const [requestReason, setRequestReason] = useState("");
  const [newDosage, setNewDosage] = useState("");
  const [newFrequency, setNewFrequency] = useState("");

  const { data: activeMeds } = useQuery<any[]>({
    queryKey: ['/api/medications'],
  });

  const { data: myRequests, isLoading } = useQuery<any[]>({
    queryKey: ['/api/dosage-change-requests'],
  });

  const { data: pendingRequests, isLoading: loadingPending } = useQuery<any[]>({
    queryKey: ['/api/dosage-change-requests/pending'],
    enabled: user?.role === 'doctor',
  });

  const createRequestMutation = useMutation({
    mutationFn: async (data: any) => {
      return apiRequest('POST', '/api/dosage-change-requests', data);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['/api/dosage-change-requests'] });
      setShowRequestDialog(false);
      setSelectedMed(null);
      setRequestReason("");
      setNewDosage("");
      setNewFrequency("");
      toast({
        title: "Request submitted",
        description: "Your dosage change request has been sent to your doctor for approval.",
      });
    },
  });

  const approveMutation = useMutation({
    mutationFn: async ({ id, notes }: { id: string; notes?: string }) => {
      return apiRequest('POST', `/api/dosage-change-requests/${id}/approve`, { notes });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['/api/dosage-change-requests'] });
      queryClient.invalidateQueries({ queryKey: ['/api/dosage-change-requests/pending'] });
      queryClient.invalidateQueries({ queryKey: ['/api/medications'] });
      toast({
        title: "Request approved",
        description: "The dosage change has been applied to the patient's medication.",
      });
    },
  });

  const rejectMutation = useMutation({
    mutationFn: async ({ id, notes }: { id: string; notes: string }) => {
      return apiRequest('POST', `/api/dosage-change-requests/${id}/reject`, { notes });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['/api/dosage-change-requests'] });
      queryClient.invalidateQueries({ queryKey: ['/api/dosage-change-requests/pending'] });
      toast({
        title: "Request rejected",
        description: "The dosage change request has been declined.",
      });
    },
  });

  const handleSubmitRequest = () => {
    if (!selectedMed || !newDosage || !requestReason.trim()) {
      toast({
        title: "Missing information",
        description: "Please provide dosage, frequency, and reason for the change.",
        variant: "destructive",
      });
      return;
    }

    createRequestMutation.mutate({
      medicationId: selectedMed.id,
      currentDosage: selectedMed.dosage,
      currentFrequency: selectedMed.frequency,
      requestedDosage: newDosage,
      requestedFrequency: newFrequency || selectedMed.frequency,
      requestReason,
      requestType: 'dosage_change',
    });
  };

  const getStatusBadge = (status: string) => {
    switch (status) {
      case 'pending':
        return <Badge variant="secondary">Pending</Badge>;
      case 'approved':
        return <Badge className="bg-green-500 hover:bg-green-600">Approved</Badge>;
      case 'rejected':
        return <Badge variant="destructive">Rejected</Badge>;
      default:
        return <Badge variant="outline">{status}</Badge>;
    }
  };

  return (
    <div className="space-y-6">
      {/* Create Request Button */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle>Dosage Change Requests</CardTitle>
              <CardDescription>
                Request changes to your medication dosages (requires doctor approval)
              </CardDescription>
            </div>
            <Button onClick={() => setShowRequestDialog(true)} data-testid="button-new-request">
              <Plus className="h-4 w-4 mr-2" />
              New Request
            </Button>
          </div>
        </CardHeader>
      </Card>

      {/* My Requests */}
      <Card>
        <CardHeader>
          <CardTitle>My Requests</CardTitle>
        </CardHeader>
        <CardContent>
          {isLoading ? (
            <div className="space-y-3">
              {[1, 2].map(i => <Skeleton key={i} className="h-20 w-full" />)}
            </div>
          ) : myRequests && myRequests.length > 0 ? (
            <div className="space-y-3">
              {myRequests.map((req) => (
                <Card key={req.id} className="hover-elevate" data-testid={`request-${req.id}`}>
                  <CardContent className="pt-6">
                    <div className="flex items-start justify-between mb-2">
                      <div className="flex-1">
                        <div className="flex items-center gap-2 mb-1">
                          <h4 className="font-semibold">Medication ID: {req.medicationId}</h4>
                          {getStatusBadge(req.status)}
                        </div>
                        <p className="text-sm text-muted-foreground">
                          {req.currentDosage} → {req.requestedDosage}
                          {req.currentFrequency !== req.requestedFrequency && 
                            ` • ${req.currentFrequency} → ${req.requestedFrequency}`}
                        </p>
                      </div>
                      <span className="text-xs text-muted-foreground">
                        {format(new Date(req.requestedAt), 'MMM d, yyyy')}
                      </span>
                    </div>
                    <p className="text-sm mt-2">
                      <span className="font-medium">Reason: </span>
                      {req.requestReason}
                    </p>
                    {req.status === 'rejected' && req.reviewNotes && (
                      <Alert variant="destructive" className="mt-3">
                        <AlertTriangle className="h-4 w-4" />
                        <AlertDescription>
                          <span className="font-medium">Doctor's response: </span>
                          {req.reviewNotes}
                        </AlertDescription>
                      </Alert>
                    )}
                    {req.status === 'approved' && req.reviewNotes && (
                      <Alert className="mt-3">
                        <CheckCircle2 className="h-4 w-4" />
                        <AlertDescription>
                          <span className="font-medium">Doctor's note: </span>
                          {req.reviewNotes}
                        </AlertDescription>
                      </Alert>
                    )}
                  </CardContent>
                </Card>
              ))}
            </div>
          ) : (
            <div className="text-center py-12 text-muted-foreground">
              <FileText className="h-12 w-12 mx-auto mb-3 opacity-50" />
              <p>No requests yet</p>
              <p className="text-sm mt-1">Create a request to change your medication dosage</p>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Pending Requests for Doctors */}
      {user?.role === 'doctor' && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <AlertCircle className="h-5 w-5 text-amber-500" />
              Pending Approvals
            </CardTitle>
            <CardDescription>
              Patient dosage change requests awaiting your review
            </CardDescription>
          </CardHeader>
          <CardContent>
            {loadingPending ? (
              <div className="space-y-3">
                {[1, 2].map(i => <Skeleton key={i} className="h-24 w-full" />)}
              </div>
            ) : pendingRequests && pendingRequests.length > 0 ? (
              <div className="space-y-3">
                {pendingRequests.map((req) => (
                  <Card key={req.id} className="hover-elevate" data-testid={`pending-request-${req.id}`}>
                    <CardContent className="pt-6">
                      <div className="space-y-3">
                        <div className="flex items-start justify-between">
                          <div className="flex-1">
                            <h4 className="font-semibold">Patient: {req.patientId}</h4>
                            <p className="text-sm text-muted-foreground">
                              Medication ID: {req.medicationId}
                            </p>
                            <p className="text-sm mt-1">
                              {req.currentDosage} → {req.requestedDosage}
                              {req.currentFrequency !== req.requestedFrequency && 
                                ` • ${req.currentFrequency} → ${req.requestedFrequency}`}
                            </p>
                          </div>
                          <span className="text-xs text-muted-foreground">
                            {format(new Date(req.requestedAt), 'MMM d, yyyy')}
                          </span>
                        </div>
                        <p className="text-sm">
                          <span className="font-medium">Patient's reason: </span>
                          {req.requestReason}
                        </p>
                        <div className="flex gap-2 pt-2">
                          <Button
                            size="sm"
                            onClick={() => approveMutation.mutate({ id: req.id })}
                            disabled={approveMutation.isPending}
                            data-testid={`button-approve-${req.id}`}
                          >
                            <Check className="h-4 w-4 mr-1" />
                            Approve
                          </Button>
                          <Button
                            size="sm"
                            variant="destructive"
                            onClick={() => {
                              const notes = prompt("Rejection reason (optional):");
                              if (notes !== null) {
                                rejectMutation.mutate({ id: req.id, notes: notes || "No reason provided" });
                              }
                            }}
                            disabled={rejectMutation.isPending}
                            data-testid={`button-reject-${req.id}`}
                          >
                            <XCircle className="h-4 w-4 mr-1" />
                            Reject
                          </Button>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            ) : (
              <div className="text-center py-12 text-muted-foreground">
                <CheckCircle2 className="h-12 w-12 mx-auto mb-3 text-green-500 opacity-50" />
                <p>No pending requests</p>
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* Request Dialog */}
      <Dialog open={showRequestDialog} onOpenChange={setShowRequestDialog}>
        <DialogContent data-testid="dialog-dosage-change">
          <DialogHeader>
            <DialogTitle>Request Dosage Change</DialogTitle>
            <DialogDescription>
              Submit a request to change your medication dosage. Your doctor will review and approve or reject the request.
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4">
            <div>
              <Label>Select Medication</Label>
              <select
                className="w-full mt-1 p-2 border rounded-md"
                value={selectedMed?.id || ''}
                onChange={(e) => {
                  const med = activeMeds?.find(m => m.id === e.target.value);
                  setSelectedMed(med);
                  setNewDosage(med?.dosage || '');
                  setNewFrequency(med?.frequency || '');
                }}
                data-testid="select-medication"
              >
                <option value="">-- Select medication --</option>
                {activeMeds?.map((med) => (
                  <option key={med.id} value={med.id}>
                    {med.name} ({med.dosage})
                  </option>
                ))}
              </select>
            </div>
            {selectedMed && (
              <>
                <div>
                  <Label>Current Dosage</Label>
                  <Input value={selectedMed.dosage} disabled />
                </div>
                <div>
                  <Label>New Dosage</Label>
                  <Input
                    value={newDosage}
                    onChange={(e) => setNewDosage(e.target.value)}
                    placeholder="e.g., 20mg"
                    data-testid="input-new-dosage"
                  />
                </div>
                <div>
                  <Label>New Frequency (optional)</Label>
                  <Input
                    value={newFrequency}
                    onChange={(e) => setNewFrequency(e.target.value)}
                    placeholder={`Current: ${selectedMed.frequency}`}
                    data-testid="input-new-frequency"
                  />
                </div>
                <div>
                  <Label>Reason for Change</Label>
                  <textarea
                    className="w-full mt-1 p-2 border rounded-md"
                    rows={3}
                    value={requestReason}
                    onChange={(e) => setRequestReason(e.target.value)}
                    placeholder="Explain why you want to change the dosage..."
                    data-testid="textarea-reason"
                  />
                </div>
              </>
            )}
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setShowRequestDialog(false)}>
              Cancel
            </Button>
            <Button
              onClick={handleSubmitRequest}
              disabled={createRequestMutation.isPending || !selectedMed}
              data-testid="button-submit-request"
            >
              {createRequestMutation.isPending && <Loader2 className="h-4 w-4 mr-2 animate-spin" />}
              Submit Request
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
