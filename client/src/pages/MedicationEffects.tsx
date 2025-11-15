import { useState } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Skeleton } from "@/components/ui/skeleton";
import { Slider } from "@/components/ui/slider";
import { Label } from "@/components/ui/label";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { useToast } from "@/hooks/use-toast";
import { apiRequest, queryClient } from "@/lib/queryClient";
import {
  Pill,
  AlertTriangle,
  TrendingUp,
  Clock,
  Activity,
  Sparkles,
  RefreshCw,
  Info,
  ChevronRight,
  Check,
  Settings,
  Sliders,
} from "lucide-react";
import { format } from "date-fns";

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

export default function MedicationEffects() {
  const { toast } = useToast();
  const [selectedMedicationIdx, setSelectedMedicationIdx] = useState<number>(0); // FIX: Default to first medication
  const [daysBack, setDaysBack] = useState(90);
  const [minConfidence, setMinConfidence] = useState(0.4);
  const [showAnalysisConfig, setShowAnalysisConfig] = useState(false);

  // Fetch effects summary
  const { data: summary, isLoading } = useQuery<EffectsSummary>({
    queryKey: [`/api/v1/medication-side-effects/summary/me`, { days_back: daysBack }],
    queryFn: async () => {
      const res = await fetch(`/api/v1/medication-side-effects/summary/me?days_back=${daysBack}`, {
        credentials: 'include'
      });
      if (!res.ok) throw new Error("Failed to load medication effects");
      return res.json();
    }
  });

  // Trigger new analysis
  const analyzeCorrelations = useMutation({
    mutationFn: async (params: { days_back: number; min_confidence: number }) => {
      return await apiRequest(`/api/v1/medication-side-effects/analyze/me`, {
        method: "POST",
        body: JSON.stringify(params)
      });
    },
    onSuccess: () => {
      // FIX: Invalidate all summary queries (exact: false to match parameterized keys)
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

  const getStrengthBadge = (strength: string) => {
    switch (strength.toUpperCase()) {
      case "STRONG":
        return <Badge className="bg-rose-500 hover:bg-rose-600" data-testid={`badge-strength-strong`}>
          <AlertTriangle className="w-3 h-3 mr-1" />
          Strong Correlation
        </Badge>;
      case "LIKELY":
        return <Badge className="bg-coral-500 hover:bg-coral-600" data-testid={`badge-strength-likely`}>
          <TrendingUp className="w-3 h-3 mr-1" />
          Likely Correlation
        </Badge>;
      case "POSSIBLE":
        return <Badge variant="outline" className="border-teal-400 text-teal-700" data-testid={`badge-strength-possible`}>
          <Info className="w-3 h-3 mr-1" />
          Possible Correlation
        </Badge>;
      case "UNLIKELY":
        return <Badge variant="outline" className="text-slate-600" data-testid={`badge-strength-unlikely`}>
          Unlikely
        </Badge>;
      default:
        return <Badge variant="outline" data-testid={`badge-strength-${strength.toLowerCase()}`}>{strength}</Badge>;
    }
  };

  const getImpactColor = (impact: string) => {
    switch (impact.toLowerCase()) {
      case "severe": return "text-rose-600 font-semibold";
      case "moderate": return "text-coral-600 font-medium";
      case "mild": return "text-teal-600";
      default: return "text-slate-600";
    }
  };

  if (isLoading) {
    return (
      <div className="container mx-auto p-6 max-w-7xl" data-testid="container-medication-effects">
        <div className="space-y-6">
          <Skeleton className="h-12 w-3/4" />
          <Skeleton className="h-32 w-full" />
          <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
            <Skeleton className="h-64" />
            <Skeleton className="h-64" />
            <Skeleton className="h-64" />
          </div>
        </div>
      </div>
    );
  }

  if (!summary || summary.medication_groups.length === 0) {
    return (
      <div className="container mx-auto p-6 max-w-7xl" data-testid="container-medication-effects">
        <Card className="p-8">
          <div className="text-center space-y-4">
            <Pill className="w-16 h-16 mx-auto text-slate-400" />
            <h2 className="text-2xl font-semibold">No Medication Effects Found</h2>
            <p className="text-slate-600 max-w-md mx-auto">
              Start tracking your medications and symptoms to see AI-powered correlation analysis here.
            </p>
            <Button
              onClick={() => analyzeCorrelations.mutate({ days_back: daysBack, min_confidence: 0.4 })}
              disabled={analyzeCorrelations.isPending}
              data-testid="button-analyze-now"
              className="bg-teal-600 hover:bg-teal-700"
            >
              {analyzeCorrelations.isPending ? (
                <>
                  <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
                  Analyzing...
                </>
              ) : (
                <>
                  <Sparkles className="w-4 h-4 mr-2" />
                  Analyze Now
                </>
              )}
            </Button>
          </div>
        </Card>
      </div>
    );
  }

  // FIX: Default to first medication if available
  const selectedMedication = summary.medication_groups[selectedMedicationIdx] || null;

  return (
    <div className="container mx-auto p-6 max-w-7xl" data-testid="container-medication-effects">
      {/* Header with Analysis Config - FIX: Add configuration UI */}
      <div className="mb-8 flex items-start justify-between">
        <div className="space-y-2">
          <h1 className="text-4xl font-bold tracking-tight">Medication Effects Analysis</h1>
          <p className="text-slate-600 text-lg">
            AI-powered insights into how your medications correlate with your symptoms
          </p>
        </div>
        
        <Dialog open={showAnalysisConfig} onOpenChange={setShowAnalysisConfig}>
          <DialogTrigger asChild>
            <Button variant="outline" data-testid="button-configure-analysis">
              <Settings className="w-4 h-4 mr-2" />
              Configure Analysis
            </Button>
          </DialogTrigger>
          <DialogContent className="sm:max-w-md">
            <DialogHeader>
              <DialogTitle>Analysis Configuration</DialogTitle>
              <DialogDescription>
                Customize the correlation analysis parameters
              </DialogDescription>
            </DialogHeader>
            <div className="space-y-6 py-4">
              <div className="space-y-2">
                <Label htmlFor="days-back-slider">Analysis Period: {daysBack} days</Label>
                <Slider
                  id="days-back-slider"
                  min={7}
                  max={365}
                  step={7}
                  value={[daysBack]}
                  onValueChange={(value) => setDaysBack(value[0])}
                  data-testid="slider-days-back"
                />
                <p className="text-xs text-slate-600">
                  Analyze medications and symptoms from the last {daysBack} days
                </p>
              </div>
              
              <div className="space-y-2">
                <Label htmlFor="confidence-slider">
                  Minimum Confidence: {(minConfidence * 100).toFixed(0)}%
                </Label>
                <Slider
                  id="confidence-slider"
                  min={0}
                  max={100}
                  step={5}
                  value={[minConfidence * 100]}
                  onValueChange={(value) => setMinConfidence(value[0] / 100)}
                  data-testid="slider-min-confidence"
                />
                <p className="text-xs text-slate-600">
                  Only show correlations with at least {(minConfidence * 100).toFixed(0)}% confidence
                </p>
              </div>

              <Button
                onClick={() => {
                  analyzeCorrelations.mutate({ days_back: daysBack, min_confidence: minConfidence });
                  setShowAnalysisConfig(false);
                }}
                disabled={analyzeCorrelations.isPending}
                className="w-full bg-teal-600 hover:bg-teal-700"
                data-testid="button-apply-and-analyze"
              >
                {analyzeCorrelations.isPending ? (
                  <>
                    <RefreshCw className="w-4 h-4 mr-2 animate-spin" />
                    Analyzing...
                  </>
                ) : (
                  <>
                    <Sparkles className="w-4 h-4 mr-2" />
                    Apply & Analyze
                  </>
                )}
              </Button>
            </div>
          </DialogContent>
        </Dialog>
      </div>

      {/* Summary Stats */}
      <div className="grid gap-6 md:grid-cols-4 mb-8">
        <Card data-testid="card-stat-medications">
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium text-slate-600">Medications Analyzed</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold text-teal-600">
              {summary.medications_analyzed}
            </div>
          </CardContent>
        </Card>

        <Card data-testid="card-stat-correlations">
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium text-slate-600">Total Correlations</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold">{summary.total_correlations_found}</div>
          </CardContent>
        </Card>

        <Card data-testid="card-stat-strong">
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium text-slate-600">Strong Correlations</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold text-rose-600">
              {summary.strong_correlations_count}
            </div>
          </CardContent>
        </Card>

        <Card data-testid="card-stat-period">
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium text-slate-600">Analysis Period</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold">{summary.analysis_period_days} days</div>
          </CardContent>
        </Card>
      </div>

      {/* Main Content: Tabs */}
      <Tabs defaultValue="overview" className="space-y-6">
        <TabsList data-testid="tabs-main">
          <TabsTrigger value="overview" data-testid="tab-overview">
            <Activity className="w-4 h-4 mr-2" />
            Overview
          </TabsTrigger>
          <TabsTrigger value="by-medication" data-testid="tab-by-medication">
            <Pill className="w-4 h-4 mr-2" />
            By Medication
          </TabsTrigger>
          <TabsTrigger value="recommendations" data-testid="tab-recommendations">
            <Sparkles className="w-4 h-4 mr-2" />
            Recommendations
          </TabsTrigger>
        </TabsList>

        {/* Overview Tab */}
        <TabsContent value="overview" className="space-y-6">
          <div className="grid gap-6 lg:grid-cols-2">
            {summary.medication_groups.map((medGroup, idx) => (
              <Card
                key={idx}
                className="hover-elevate cursor-pointer"
                onClick={() => {
                  setSelectedMedicationIdx(idx);
                  // Switch to By Medication tab to show details - FIX: Make tab switching visible
                  const tabTrigger = document.querySelector('[data-testid="tab-by-medication"]') as HTMLElement;
                  tabTrigger?.click();
                }}
                data-testid={`card-medication-${idx}`}
              >
                <CardHeader>
                  <div className="flex items-start justify-between">
                    <div className="space-y-1">
                      <CardTitle className="text-xl flex items-center gap-2">
                        <Pill className="w-5 h-5 text-teal-600" />
                        {medGroup.medication_name}
                      </CardTitle>
                      <CardDescription>{medGroup.dosage}</CardDescription>
                    </div>
                    <ChevronRight className="w-5 h-5 text-slate-400" />
                  </div>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="flex items-center gap-6">
                    <div>
                      <div className="text-2xl font-bold text-slate-900">
                        {medGroup.total_correlations}
                      </div>
                      <div className="text-sm text-slate-600">Total Correlations</div>
                    </div>
                    {medGroup.strong_correlations > 0 && (
                      <div>
                        <div className="text-2xl font-bold text-rose-600">
                          {medGroup.strong_correlations}
                        </div>
                        <div className="text-sm text-slate-600">Strong</div>
                      </div>
                    )}
                  </div>

                  {/* Top correlations preview */}
                  <div className="space-y-2">
                    {medGroup.correlations.slice(0, 3).map((corr) => (
                      <div
                        key={corr.id}
                        className="flex items-center justify-between text-sm p-2 rounded bg-slate-50"
                        data-testid={`correlation-preview-${corr.id}`}
                      >
                        <span className="font-medium">{corr.symptom_name}</span>
                        <span className={getImpactColor(corr.patient_impact)}>
                          {corr.patient_impact}
                        </span>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>

        {/* By Medication Tab */}
        <TabsContent value="by-medication" className="space-y-6">
          <div className="grid gap-6 lg:grid-cols-3">
            {/* Medication List */}
            <Card className="lg:col-span-1">
              <CardHeader>
                <CardTitle>Medications</CardTitle>
                <CardDescription>Select to view correlations</CardDescription>
              </CardHeader>
              <CardContent>
                <ScrollArea className="h-[500px]">
                  <div className="space-y-2">
                    {summary.medication_groups.map((medGroup, idx) => (
                      <Button
                        key={idx}
                        variant={selectedMedicationIdx === idx ? "default" : "outline"}
                        className="w-full justify-start"
                        onClick={() => setSelectedMedicationIdx(idx)}
                        data-testid={`button-select-medication-${idx}`}
                      >
                        <Pill className="w-4 h-4 mr-2" />
                        <div className="flex-1 text-left">
                          <div className="font-medium">{medGroup.medication_name}</div>
                          <div className="text-xs opacity-70">{medGroup.total_correlations} correlations</div>
                        </div>
                      </Button>
                    ))}
                  </div>
                </ScrollArea>
              </CardContent>
            </Card>

            {/* Correlation Details */}
            <Card className="lg:col-span-2">
              <CardHeader>
                <CardTitle>
                  {selectedMedication ? selectedMedication.medication_name : "Select a Medication"}
                </CardTitle>
                <CardDescription>
                  {selectedMedication && `${selectedMedication.dosage} - ${selectedMedication.total_correlations} correlations found`}
                </CardDescription>
              </CardHeader>
              <CardContent>
                {selectedMedication ? (
                  <ScrollArea className="h-[500px]">
                    <div className="space-y-4">
                      {selectedMedication.correlations.map((corr) => (
                        <Card
                          key={corr.id}
                          className="border-l-4 border-l-teal-600"
                          data-testid={`card-correlation-${corr.id}`}
                        >
                          <CardHeader className="pb-3">
                            <div className="flex items-start justify-between">
                              <CardTitle className="text-lg">{corr.symptom_name}</CardTitle>
                              {getStrengthBadge(corr.correlation_strength)}
                            </div>
                          </CardHeader>
                          <CardContent className="space-y-3">
                            <div className="grid grid-cols-2 gap-4 text-sm">
                              <div>
                                <div className="text-slate-600">Confidence Score</div>
                                <div className="font-semibold text-lg">
                                  {(corr.confidence_score * 100).toFixed(0)}%
                                </div>
                              </div>
                              <div>
                                <div className="text-slate-600">Time to Onset</div>
                                <div className="font-semibold flex items-center gap-1">
                                  <Clock className="w-4 h-4" />
                                  {corr.time_to_onset_hours}h
                                </div>
                              </div>
                            </div>

                            <div>
                              <div className="text-slate-600 text-sm mb-1">Patient Impact</div>
                              <div className={`text-lg ${getImpactColor(corr.patient_impact)}`}>
                                {corr.patient_impact.charAt(0).toUpperCase() + corr.patient_impact.slice(1)}
                              </div>
                            </div>

                            {corr.temporal_pattern && (
                              <div className="p-3 bg-slate-50 rounded">
                                <div className="text-slate-600 text-sm mb-1">Pattern</div>
                                <div className="text-sm">{corr.temporal_pattern}</div>
                              </div>
                            )}

                            {corr.ai_reasoning && (
                              <div className="p-3 bg-teal-50 rounded">
                                <div className="text-teal-800 text-sm font-medium mb-1 flex items-center gap-1">
                                  <Sparkles className="w-3 h-3" />
                                  AI Reasoning
                                </div>
                                <div className="text-sm text-teal-900">{corr.ai_reasoning}</div>
                              </div>
                            )}

                            <div className="pt-2 border-t">
                              <div className="text-slate-600 text-sm mb-1">Recommended Action</div>
                              <div className="text-sm">{corr.action_recommended}</div>
                            </div>

                            <div className="text-xs text-slate-500">
                              Analyzed: {format(new Date(corr.analysis_date), "PPp")}
                            </div>
                          </CardContent>
                        </Card>
                      ))}
                    </div>
                  </ScrollArea>
                ) : (
                  <div className="flex items-center justify-center h-[500px] text-slate-500">
                    <div className="text-center space-y-2">
                      <Pill className="w-12 h-12 mx-auto opacity-30" />
                      <p>Select a medication to view correlation details</p>
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        {/* Recommendations Tab - FIX: Structured action cards */}
        <TabsContent value="recommendations" className="space-y-6">
          <div className="grid gap-6 lg:grid-cols-2">
            {/* AI Recommendations */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Sparkles className="w-5 h-5 text-teal-600" />
                  AI Recommendations
                </CardTitle>
                <CardDescription>
                  Based on {summary.analysis_period_days} days of analysis
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="p-6 bg-teal-50 rounded-lg border border-teal-200">
                  <p className="text-teal-900 leading-relaxed">
                    {summary.recommendations}
                  </p>
                </div>
              </CardContent>
            </Card>

            {/* Priority Actions */}
            <Card>
              <CardHeader>
                <CardTitle>Priority Actions</CardTitle>
                <CardDescription>Steps you should take based on this analysis</CardDescription>
              </CardHeader>
              <CardContent className="space-y-3">
                {summary.strong_correlations_count > 0 && (
                  <Card className="border-l-4 border-l-rose-600" data-testid="action-review-strong">
                    <CardContent className="pt-4 pb-4">
                      <div className="flex items-start gap-3">
                        <AlertTriangle className="w-5 h-5 text-rose-600 flex-shrink-0 mt-0.5" />
                        <div className="flex-1">
                          <div className="font-semibold text-rose-900 mb-1">
                            Review Strong Correlations
                          </div>
                          <div className="text-sm text-rose-800">
                            {summary.strong_correlations_count} strong correlation(s) require attention. 
                            Schedule a consultation with your healthcare provider.
                          </div>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                )}

                <Card className="border-l-4 border-l-teal-600" data-testid="action-continue-tracking">
                  <CardContent className="pt-4 pb-4">
                    <div className="flex items-start gap-3">
                      <Activity className="w-5 h-5 text-teal-600 flex-shrink-0 mt-0.5" />
                      <div className="flex-1">
                        <div className="font-semibold mb-1">Continue Tracking</div>
                        <div className="text-sm text-slate-700">
                          Keep logging your symptoms and medications daily for more accurate analysis.
                        </div>
                      </div>
                      <Check className="w-5 h-5 text-teal-600" />
                    </div>
                  </CardContent>
                </Card>

                <Card className="border-l-4 border-l-teal-600" data-testid="action-regular-analysis">
                  <CardContent className="pt-4 pb-4">
                    <div className="flex items-start gap-3">
                      <Clock className="w-5 h-5 text-teal-600 flex-shrink-0 mt-0.5" />
                      <div className="flex-1">
                        <div className="font-semibold mb-1">Schedule Regular Analysis</div>
                        <div className="text-sm text-slate-700">
                          Run this analysis weekly to track changes and identify new patterns.
                        </div>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                <Card className="border-l-4 border-l-teal-600" data-testid="action-discuss-doctor">
                  <CardContent className="pt-4 pb-4">
                    <div className="flex items-start gap-3">
                      <Pill className="w-5 h-5 text-teal-600 flex-shrink-0 mt-0.5" />
                      <div className="flex-1">
                        <div className="font-semibold mb-1">Discuss With Doctor</div>
                        <div className="text-sm text-slate-700">
                          Share these correlation insights during your next consultation for personalized guidance.
                        </div>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </CardContent>
            </Card>
          </div>

          {/* Re-analyze Button */}
          <Card>
            <CardContent className="pt-6">
              <div className="flex items-center justify-between">
                <div>
                  <div className="font-semibold mb-1">Run New Analysis</div>
                  <div className="text-sm text-slate-600">
                    Updated medication or symptom data? Re-analyze to get fresh insights.
                  </div>
                </div>
                <Button
                  onClick={() => setShowAnalysisConfig(true)}
                  data-testid="button-open-config"
                  className="bg-teal-600 hover:bg-teal-700"
                >
                  <Sliders className="w-4 h-4 mr-2" />
                  Configure & Analyze
                </Button>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
