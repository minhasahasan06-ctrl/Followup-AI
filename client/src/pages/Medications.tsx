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
} from "lucide-react";
import { format } from "date-fns";
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
          <TabsList className="grid w-full max-w-3xl grid-cols-4">
            <TabsTrigger value="overview" data-testid="tab-overview">
              <Pill className="h-4 w-4 mr-2" />
              Overview
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
        </Tabs>
      </div>
    </div>
  );
}
