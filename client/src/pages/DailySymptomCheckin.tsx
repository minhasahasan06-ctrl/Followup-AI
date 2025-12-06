import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Separator } from "@/components/ui/separator";
import { ScrollArea } from "@/components/ui/scroll-area";
import { useQuery, useMutation } from "@tanstack/react-query";
import { apiRequest, queryClient } from "@/lib/queryClient";
import { useToast } from "@/hooks/use-toast";
import { 
  Activity, 
  Brain, 
  Heart, 
  Mic, 
  Video, 
  Calendar,
  TrendingUp,
  AlertCircle,
  CheckCircle2,
  Info,
  Loader2,
  BarChart3,
  FileText
} from "lucide-react";

export default function DailySymptomCheckin() {
  const { toast } = useToast();
  const [activeView, setActiveView] = useState<"checkin" | "history" | "trends">("checkin");

  return (
    <div className="container mx-auto p-6 space-y-6">
      {/* Page Header */}
      <div className="space-y-2">
        <h1 className="text-3xl font-bold" data-testid="text-page-title">Daily Symptom Check-In</h1>
        <p className="text-muted-foreground" data-testid="text-page-description">
          Track your symptoms, patterns, and wellness metrics for comprehensive health monitoring
        </p>
      </div>

      {/* Persistent Disclaimer */}
      <Alert data-testid="alert-disclaimer">
        <Info className="h-4 w-4" />
        <AlertDescription>
          <strong>Important:</strong> SymptomTrack collects self-reported and observational data. 
          It is not a diagnostic tool and all insights require clinician interpretation.
        </AlertDescription>
      </Alert>

      {/* View Switcher */}
      <div className="flex gap-2" data-testid="symptom-view-switcher">
        <Button
          variant={activeView === "checkin" ? "default" : "outline"}
          onClick={() => setActiveView("checkin")}
          data-testid="button-view-checkin"
          className="flex items-center gap-2"
        >
          <CheckCircle2 className="h-4 w-4" />
          Daily Check-In
        </Button>
        <Button
          variant={activeView === "history" ? "default" : "outline"}
          onClick={() => setActiveView("history")}
          data-testid="button-view-history"
          className="flex items-center gap-2"
        >
          <Calendar className="h-4 w-4" />
          History
        </Button>
        <Button
          variant={activeView === "trends" ? "default" : "outline"}
          onClick={() => setActiveView("trends")}
          data-testid="button-view-trends"
          className="flex items-center gap-2"
        >
          <TrendingUp className="h-4 w-4" />
          Trends & Analytics
        </Button>
      </div>

      {/* Content Area */}
      {activeView === "checkin" && <DailyCheckinFlow />}
      {activeView === "history" && <SymptomHistory />}
      {activeView === "trends" && <SymptomTrendsAnalytics />}
    </div>
  );
}

// ============================================================================
// DAILY CHECK-IN FLOW
// ============================================================================
function DailyCheckinFlow() {
  const { toast } = useToast();
  
  // Form state
  const [painLevel, setPainLevel] = useState<number>(0);
  const [fatigueLevel, setFatigueLevel] = useState<number>(0);
  const [breathlessnessLevel, setBreathlessnessLevel] = useState<number>(0);
  const [sleepQuality, setSleepQuality] = useState<number>(5);
  const [mood, setMood] = useState<string>("okay");
  const [mobilityScore, setMobilityScore] = useState<number>(5);
  const [medicationsTaken, setMedicationsTaken] = useState<boolean>(true);
  const [selectedSymptoms, setSelectedSymptoms] = useState<string[]>([]);
  const [selectedTriggers, setSelectedTriggers] = useState<string[]>([]);
  const [note, setNote] = useState<string>("");

  // Symptom options
  const symptomOptions = [
    "headache", "nausea", "dizziness", "fatigue", "fever", "chills",
    "cough", "shortness of breath", "chest pain", "muscle aches",
    "joint pain", "swelling", "rash", "confusion", "anxiety"
  ];

  const triggerOptions = [
    "physical activity", "stress", "weather change", "lack of sleep",
    "missed medication", "dietary changes", "dehydration", "allergens"
  ];

  const moodOptions = [
    { value: "great", label: "Great", color: "text-green-600" },
    { value: "good", label: "Good", color: "text-green-500" },
    { value: "okay", label: "Okay", color: "text-yellow-600" },
    { value: "low", label: "Low", color: "text-orange-600" },
    { value: "very_low", label: "Very Low", color: "text-red-600" }
  ];

  // Create check-in mutation
  const createCheckinMutation = useMutation({
    mutationFn: async (data: any) => {
      return apiRequest("/api/symptom-checkin/checkin", {
        method: "POST",
        body: JSON.stringify(data)
      });
    },
    onSuccess: () => {
      toast({
        title: "Check-in saved",
        description: "Your daily symptom check-in has been recorded."
      });
      // Reset form
      setPainLevel(0);
      setFatigueLevel(0);
      setBreathlessnessLevel(0);
      setSleepQuality(5);
      setMood("okay");
      setMobilityScore(5);
      setMedicationsTaken(true);
      setSelectedSymptoms([]);
      setSelectedTriggers([]);
      setNote("");
      queryClient.invalidateQueries({ queryKey: ["/api/symptom-checkin/checkins/recent"] });
    },
    onError: (error: any) => {
      toast({
        title: "Error saving check-in",
        description: error.message,
        variant: "destructive"
      });
    }
  });

  const handleSubmit = () => {
    createCheckinMutation.mutate({
      painLevel: painLevel > 0 ? painLevel : null,
      fatigueLevel: fatigueLevel > 0 ? fatigueLevel : null,
      breathlessnessLevel: breathlessnessLevel > 0 ? breathlessnessLevel : null,
      sleepQuality,
      mood,
      mobilityScore,
      medicationsTaken,
      symptoms: selectedSymptoms,
      triggers: selectedTriggers,
      note: note.trim() || null
    });
  };

  const toggleSymptom = (symptom: string) => {
    setSelectedSymptoms(prev =>
      prev.includes(symptom)
        ? prev.filter(s => s !== symptom)
        : [...prev, symptom]
    );
  };

  const toggleTrigger = (trigger: string) => {
    setSelectedTriggers(prev =>
      prev.includes(trigger)
        ? prev.filter(t => t !== trigger)
        : [...prev, trigger]
    );
  };

  return (
    <Card data-testid="card-daily-checkin">
      <CardHeader>
        <CardTitle>Daily Check-In</CardTitle>
        <CardDescription>
          How are you feeling today? All responses are observational and patient-reported.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Pain Level */}
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <Label htmlFor="pain-slider" data-testid="label-pain-level">
              Pain Level (Self-Reported VAS): {painLevel}/10
            </Label>
            <Badge variant="outline" data-testid="badge-pain-value">{painLevel}</Badge>
          </div>
          <Slider
            id="pain-slider"
            data-testid="slider-pain-level"
            value={[painLevel]}
            onValueChange={(v) => setPainLevel(v[0])}
            max={10}
            step={1}
            className="w-full"
          />
          <p className="text-xs text-muted-foreground">
            0 = No pain, 10 = Worst pain imaginable
          </p>
        </div>

        <Separator />

        {/* Fatigue Level */}
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <Label htmlFor="fatigue-slider" data-testid="label-fatigue-level">
              Fatigue Level: {fatigueLevel}/10
            </Label>
            <Badge variant="outline" data-testid="badge-fatigue-value">{fatigueLevel}</Badge>
          </div>
          <Slider
            id="fatigue-slider"
            data-testid="slider-fatigue-level"
            value={[fatigueLevel]}
            onValueChange={(v) => setFatigueLevel(v[0])}
            max={10}
            step={1}
            className="w-full"
          />
          <p className="text-xs text-muted-foreground">
            0 = No fatigue, 10 = Completely exhausted
          </p>
        </div>

        <Separator />

        {/* Breathlessness */}
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <Label htmlFor="breathlessness-slider" data-testid="label-breathlessness">
              Breathlessness: {breathlessnessLevel}/10
            </Label>
            <Badge variant="outline" data-testid="badge-breathlessness-value">{breathlessnessLevel}</Badge>
          </div>
          <Slider
            id="breathlessness-slider"
            data-testid="slider-breathlessness"
            value={[breathlessnessLevel]}
            onValueChange={(v) => setBreathlessnessLevel(v[0])}
            max={10}
            step={1}
            className="w-full"
          />
          <p className="text-xs text-muted-foreground">
            0 = Normal breathing, 10 = Severe shortness of breath
          </p>
        </div>

        <Separator />

        {/* Sleep Quality */}
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <Label htmlFor="sleep-slider" data-testid="label-sleep-quality">
              Sleep Quality: {sleepQuality}/10
            </Label>
            <Badge variant="outline" data-testid="badge-sleep-value">{sleepQuality}</Badge>
          </div>
          <Slider
            id="sleep-slider"
            data-testid="slider-sleep-quality"
            value={[sleepQuality]}
            onValueChange={(v) => setSleepQuality(v[0])}
            max={10}
            step={1}
            className="w-full"
          />
          <p className="text-xs text-muted-foreground">
            0 = Terrible sleep, 10 = Perfect rest
          </p>
        </div>

        <Separator />

        {/* Mood */}
        <div className="space-y-3">
          <Label data-testid="label-mood">Mood</Label>
          <div className="flex gap-2 flex-wrap">
            {moodOptions.map((option) => (
              <Button
                key={option.value}
                variant={mood === option.value ? "default" : "outline"}
                onClick={() => setMood(option.value)}
                data-testid={`button-mood-${option.value}`}
              >
                {option.label}
              </Button>
            ))}
          </div>
        </div>

        <Separator />

        {/* Mobility Score */}
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <Label htmlFor="mobility-slider" data-testid="label-mobility">
              Mobility: {mobilityScore}/10
            </Label>
            <Badge variant="outline" data-testid="badge-mobility-value">{mobilityScore}</Badge>
          </div>
          <Slider
            id="mobility-slider"
            data-testid="slider-mobility"
            value={[mobilityScore]}
            onValueChange={(v) => setMobilityScore(v[0])}
            max={10}
            step={1}
            className="w-full"
          />
          <p className="text-xs text-muted-foreground">
            0 = Cannot move, 10 = Full mobility
          </p>
        </div>

        <Separator />

        {/* Medications Taken */}
        <div className="flex items-center gap-3">
          <Label htmlFor="medications-taken" data-testid="label-medications">
            Did you take your medications as prescribed today?
          </Label>
          <div className="flex gap-2">
            <Button
              variant={medicationsTaken ? "default" : "outline"}
              onClick={() => setMedicationsTaken(true)}
              data-testid="button-medications-yes"
            >
              Yes
            </Button>
            <Button
              variant={!medicationsTaken ? "default" : "outline"}
              onClick={() => setMedicationsTaken(false)}
              data-testid="button-medications-no"
            >
              No
            </Button>
          </div>
        </div>

        <Separator />

        {/* Symptoms */}
        <div className="space-y-3">
          <Label data-testid="label-symptoms">Symptoms Experienced Today</Label>
          <div className="flex gap-2 flex-wrap">
            {symptomOptions.map((symptom) => (
              <Badge
                key={symptom}
                variant={selectedSymptoms.includes(symptom) ? "default" : "outline"}
                className="cursor-pointer hover-elevate active-elevate-2"
                onClick={() => toggleSymptom(symptom)}
                data-testid={`badge-symptom-${symptom.replace(/\s+/g, "-")}`}
              >
                {symptom}
              </Badge>
            ))}
          </div>
        </div>

        <Separator />

        {/* Triggers */}
        <div className="space-y-3">
          <Label data-testid="label-triggers">Possible Triggers</Label>
          <div className="flex gap-2 flex-wrap">
            {triggerOptions.map((trigger) => (
              <Badge
                key={trigger}
                variant={selectedTriggers.includes(trigger) ? "default" : "outline"}
                className="cursor-pointer hover-elevate active-elevate-2"
                onClick={() => toggleTrigger(trigger)}
                data-testid={`badge-trigger-${trigger.replace(/\s+/g, "-")}`}
              >
                {trigger}
              </Badge>
            ))}
          </div>
        </div>

        <Separator />

        {/* Notes */}
        <div className="space-y-3">
          <Label htmlFor="note" data-testid="label-note">Additional Notes (Optional)</Label>
          <Textarea
            id="note"
            data-testid="textarea-note"
            value={note}
            onChange={(e) => setNote(e.target.value)}
            placeholder="Any additional details about how you're feeling today..."
            rows={4}
          />
        </div>

        {/* Submit Button */}
        <Button
          onClick={handleSubmit}
          disabled={createCheckinMutation.isPending}
          data-testid="button-submit-checkin"
          className="w-full"
        >
          {createCheckinMutation.isPending && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
          Save Check-In
        </Button>
      </CardContent>
    </Card>
  );
}

// ============================================================================
// SYMPTOM HISTORY
// ============================================================================
function SymptomHistory() {
  const [daysToShow, setDaysToShow] = useState(7);
  
  const { data: checkins, isLoading } = useQuery({
    queryKey: ["/api/symptom-checkin/checkins/recent", daysToShow],
    queryFn: async () => {
      const response = await fetch(`/api/symptom-checkin/checkins/recent?days=${daysToShow}`, {
        credentials: "include"
      });
      if (!response.ok) throw new Error("Failed to fetch check-ins");
      return response.json();
    }
  });

  return (
    <Card data-testid="card-symptom-history">
      <CardHeader>
        <div className="flex items-center justify-between flex-wrap gap-4">
          <div>
            <CardTitle>Check-In History</CardTitle>
            <CardDescription>View your past symptom reports</CardDescription>
          </div>
          <div className="flex gap-2">
            <Button
              variant={daysToShow === 7 ? "default" : "outline"}
              onClick={() => setDaysToShow(7)}
              data-testid="button-days-7"
            >
              7 days
            </Button>
            <Button
              variant={daysToShow === 30 ? "default" : "outline"}
              onClick={() => setDaysToShow(30)}
              data-testid="button-days-30"
            >
              30 days
            </Button>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        {isLoading ? (
          <div className="flex items-center justify-center py-8">
            <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
          </div>
        ) : checkins && checkins.length > 0 ? (
          <ScrollArea className="h-[500px]">
            <div className="space-y-4">
              {checkins.map((checkin: any) => (
                <Card key={checkin.id} data-testid={`card-checkin-${checkin.id}`}>
                  <CardHeader>
                    <CardTitle className="text-sm">
                      {new Date(checkin.timestamp).toLocaleDateString()} at{" "}
                      {new Date(checkin.timestamp).toLocaleTimeString()}
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-2 text-sm">
                    {checkin.painLevel !== null && (
                      <p data-testid="text-history-pain">Pain: {checkin.painLevel}/10</p>
                    )}
                    {checkin.fatigueLevel !== null && (
                      <p data-testid="text-history-fatigue">Fatigue: {checkin.fatigueLevel}/10</p>
                    )}
                    {checkin.symptoms && checkin.symptoms.length > 0 && (
                      <div>
                        <strong>Symptoms:</strong>{" "}
                        {checkin.symptoms.map((s: string) => (
                          <Badge key={s} variant="outline" className="mr-1">
                            {s}
                          </Badge>
                        ))}
                      </div>
                    )}
                  </CardContent>
                </Card>
              ))}
            </div>
          </ScrollArea>
        ) : (
          <div className="text-center py-8 text-muted-foreground" data-testid="text-no-history">
            No check-ins recorded yet. Complete your first daily check-in to get started.
          </div>
        )}
      </CardContent>
    </Card>
  );
}

// ============================================================================
// SYMPTOM TRENDS & ANALYTICS
// ============================================================================
function SymptomTrendsAnalytics() {
  const { toast } = useToast();
  const [selectedPeriod, setSelectedPeriod] = useState<"3day" | "7day" | "15day" | "30day">("7day");

  const generateReportMutation = useMutation({
    mutationFn: async (reportType: string) => {
      return apiRequest("/api/symptom-checkin/trend-report", {
        method: "POST",
        body: JSON.stringify({ reportType })
      });
    },
    onSuccess: (data) => {
      toast({
        title: "Trend report generated",
        description: "Your symptom trend analysis is ready."
      });
    },
    onError: (error: any) => {
      toast({
        title: "Error generating report",
        description: error.message,
        variant: "destructive"
      });
    }
  });

  const handleGenerateReport = () => {
    generateReportMutation.mutate(selectedPeriod);
  };

  return (
    <div className="space-y-4">
      {/* Report Period Selector */}
      <Card data-testid="card-trend-selector">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <BarChart3 className="h-5 w-5" />
            Symptom Trend Analysis
          </CardTitle>
          <CardDescription>
            Generate ML-based observational reports for clinician review
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex gap-2 flex-wrap">
            <Button
              variant={selectedPeriod === "3day" ? "default" : "outline"}
              onClick={() => setSelectedPeriod("3day")}
              data-testid="button-period-3day"
            >
              3-Day Snapshot
            </Button>
            <Button
              variant={selectedPeriod === "7day" ? "default" : "outline"}
              onClick={() => setSelectedPeriod("7day")}
              data-testid="button-period-7day"
            >
              7-Day Overview
            </Button>
            <Button
              variant={selectedPeriod === "15day" ? "default" : "outline"}
              onClick={() => setSelectedPeriod("15day")}
              data-testid="button-period-15day"
            >
              15-Day Trends
            </Button>
            <Button
              variant={selectedPeriod === "30day" ? "default" : "outline"}
              onClick={() => setSelectedPeriod("30day")}
              data-testid="button-period-30day"
            >
              30-Day Analysis
            </Button>
          </div>

          <Button
            onClick={handleGenerateReport}
            disabled={generateReportMutation.isPending}
            data-testid="button-generate-report"
            className="w-full"
          >
            {generateReportMutation.isPending && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
            <FileText className="mr-2 h-4 w-4" />
            Generate Trend Report
          </Button>
        </CardContent>
      </Card>

      {/* Report Results */}
      {generateReportMutation.data && (
        <Card data-testid="card-trend-report">
          <CardHeader>
            <CardTitle>Observational Trend Report</CardTitle>
            <CardDescription>
              {new Date(generateReportMutation.data.periodStart).toLocaleDateString()} -{" "}
              {new Date(generateReportMutation.data.periodEnd).toLocaleDateString()}
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* Disclaimer */}
            <Alert data-testid="alert-report-disclaimer">
              <AlertCircle className="h-4 w-4" />
              <AlertDescription>
                This report contains observational data only. It is NOT a diagnostic tool and 
                requires professional clinician interpretation.
              </AlertDescription>
            </Alert>

            {/* Clinician Summary */}
            <div className="space-y-2">
              <h3 className="font-semibold" data-testid="text-summary-title">Clinician Summary</h3>
              <div className="bg-muted p-4 rounded-md">
                <pre className="whitespace-pre-wrap text-sm font-mono" data-testid="text-summary-content">
                  {generateReportMutation.data.clinicianSummary}
                </pre>
              </div>
            </div>

            {/* Metrics Summary */}
            {generateReportMutation.data.aggregatedMetrics && (
              <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
                {generateReportMutation.data.aggregatedMetrics.avgPainLevel !== null && (
                  <Card>
                    <CardHeader className="pb-2">
                      <CardTitle className="text-sm">Avg Pain Level</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <p className="text-2xl font-bold" data-testid="metric-avg-pain">
                        {generateReportMutation.data.aggregatedMetrics.avgPainLevel}/10
                      </p>
                    </CardContent>
                  </Card>
                )}
                {generateReportMutation.data.aggregatedMetrics.avgFatigueLevel !== null && (
                  <Card>
                    <CardHeader className="pb-2">
                      <CardTitle className="text-sm">Avg Fatigue</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <p className="text-2xl font-bold" data-testid="metric-avg-fatigue">
                        {generateReportMutation.data.aggregatedMetrics.avgFatigueLevel}/10
                      </p>
                    </CardContent>
                  </Card>
                )}
                {generateReportMutation.data.aggregatedMetrics.avgSleepQuality !== null && (
                  <Card>
                    <CardHeader className="pb-2">
                      <CardTitle className="text-sm">Avg Sleep Quality</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <p className="text-2xl font-bold" data-testid="metric-avg-sleep">
                        {generateReportMutation.data.aggregatedMetrics.avgSleepQuality}/10
                      </p>
                    </CardContent>
                  </Card>
                )}
              </div>
            )}
          </CardContent>
        </Card>
      )}
    </div>
  );
}
