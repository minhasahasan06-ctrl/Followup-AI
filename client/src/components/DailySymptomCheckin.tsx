import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { useMutation, useQuery } from "@tanstack/react-query";
import { apiRequest, queryClient } from "@/lib/queryClient";
import { useToast } from "@/hooks/use-toast";
import { CheckCircle2, Loader2, Info, Calendar } from "lucide-react";
import { format } from "date-fns";

export function DailySymptomCheckin() {
  const { toast } = useToast();
  
  // Fetch recent check-ins
  const { data: recentCheckins, isLoading: isLoadingCheckins, error: checkinsError } = useQuery({
    queryKey: ["/api/symptom-checkin/history"],
    queryFn: () => apiRequest("/api/symptom-checkin/history?days=7")
  });
  
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
      // Invalidate all symptom-related queries to refresh UI
      queryClient.invalidateQueries({ queryKey: ["/api/symptom-checkin/history"] });
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
      painLevel, // Pain VAS ALWAYS reported (0-10), never null - legal requirement
      fatigueLevel: fatigueLevel > 0 ? fatigueLevel : null,
      breathlessnessLevel: breathlessnessLevel > 0 ? breathlessnessLevel : null,
      sleepQuality,
      mood,
      mobilityScore,
      medicationsTaken,
      symptoms: selectedSymptoms,
      triggers: selectedTriggers,
      note: note.trim() || null,
      deviceType: "web"
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
    <div className="space-y-4">
      {/* Persistent Disclaimer */}
      <Alert data-testid="alert-disclaimer">
        <Info className="h-4 w-4" />
        <AlertDescription className="text-xs">
          <strong>Important:</strong> All responses are patient-reported and observational data. 
          This is not a diagnostic tool and all insights require clinician interpretation.
        </AlertDescription>
      </Alert>

      {/* Pain Level */}
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <Label htmlFor="pain-slider" className="text-sm" data-testid="label-pain-level">
            Pain Level (VAS): {painLevel}/10
          </Label>
          <Badge variant="outline" className="text-xs" data-testid="badge-pain-value">{painLevel}</Badge>
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
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <Label htmlFor="fatigue-slider" className="text-sm" data-testid="label-fatigue-level">
            Fatigue: {fatigueLevel}/10
          </Label>
          <Badge variant="outline" className="text-xs" data-testid="badge-fatigue-value">{fatigueLevel}</Badge>
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
      </div>

      <Separator />

      {/* Breathlessness */}
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <Label htmlFor="breathlessness-slider" className="text-sm" data-testid="label-breathlessness">
            Breathlessness: {breathlessnessLevel}/10
          </Label>
          <Badge variant="outline" className="text-xs" data-testid="badge-breathlessness-value">{breathlessnessLevel}</Badge>
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
      </div>

      <Separator />

      {/* Sleep Quality */}
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <Label htmlFor="sleep-slider" className="text-sm" data-testid="label-sleep-quality">
            Sleep Quality: {sleepQuality}/10
          </Label>
          <Badge variant="outline" className="text-xs" data-testid="badge-sleep-value">{sleepQuality}</Badge>
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
      </div>

      <Separator />

      {/* Mood */}
      <div className="space-y-2">
        <Label className="text-sm" data-testid="label-mood">Mood</Label>
        <div className="flex gap-2 flex-wrap">
          {moodOptions.map((option) => (
            <Button
              key={option.value}
              variant={mood === option.value ? "default" : "outline"}
              size="sm"
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
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <Label htmlFor="mobility-slider" className="text-sm" data-testid="label-mobility">
            Mobility: {mobilityScore}/10
          </Label>
          <Badge variant="outline" className="text-xs" data-testid="badge-mobility-value">{mobilityScore}</Badge>
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
      </div>

      <Separator />

      {/* Medications Taken */}
      <div className="space-y-2">
        <Label className="text-sm" data-testid="label-medications">
          Medications taken as prescribed?
        </Label>
        <div className="flex gap-2">
          <Button
            variant={medicationsTaken ? "default" : "outline"}
            size="sm"
            onClick={() => setMedicationsTaken(true)}
            data-testid="button-medications-yes"
          >
            Yes
          </Button>
          <Button
            variant={!medicationsTaken ? "default" : "outline"}
            size="sm"
            onClick={() => setMedicationsTaken(false)}
            data-testid="button-medications-no"
          >
            No
          </Button>
        </div>
      </div>

      <Separator />

      {/* Symptoms */}
      <div className="space-y-2">
        <Label className="text-sm" data-testid="label-symptoms">Symptoms Experienced Today</Label>
        <div className="flex gap-1 flex-wrap">
          {symptomOptions.map((symptom) => (
            <Badge
              key={symptom}
              variant={selectedSymptoms.includes(symptom) ? "default" : "outline"}
              className="cursor-pointer hover-elevate active-elevate-2 text-xs"
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
      <div className="space-y-2">
        <Label className="text-sm" data-testid="label-triggers">Possible Triggers</Label>
        <div className="flex gap-1 flex-wrap">
          {triggerOptions.map((trigger) => (
            <Badge
              key={trigger}
              variant={selectedTriggers.includes(trigger) ? "default" : "outline"}
              className="cursor-pointer hover-elevate active-elevate-2 text-xs"
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
      <div className="space-y-2">
        <Label htmlFor="note" className="text-sm" data-testid="label-note">Additional Notes (Optional)</Label>
        <Textarea
          id="note"
          data-testid="textarea-note"
          value={note}
          onChange={(e) => setNote(e.target.value)}
          placeholder="Any additional details about how you're feeling today..."
          rows={3}
          className="text-sm"
        />
      </div>

      {/* Submit Button */}
      <Button
        onClick={handleSubmit}
        disabled={createCheckinMutation.isPending}
        className="w-full"
        data-testid="button-submit-checkin"
      >
        {createCheckinMutation.isPending ? (
          <>
            <Loader2 className="mr-2 h-4 w-4 animate-spin" />
            Saving...
          </>
        ) : (
          <>
            <CheckCircle2 className="mr-2 h-4 w-4" />
            Save Check-In
          </>
        )}
      </Button>

      <p className="text-xs text-center text-muted-foreground">
        All data is patient-reported and observational. Not a diagnostic tool.
      </p>

      {/* Recent Check-ins */}
      <Separator />
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <Label className="text-sm font-medium">Recent Check-Ins (Last 7 Days)</Label>
          {recentCheckins && recentCheckins.length > 0 && (
            <Badge variant="secondary" className="text-xs">{recentCheckins.length}</Badge>
          )}
        </div>
        
        {/* Persistent Disclaimer for Historical Data */}
        <Alert className="bg-muted/30" data-testid="alert-history-disclaimer">
          <Info className="h-3 w-3" />
          <AlertDescription className="text-xs">
            All historical check-ins are patient-reported observational data. Not diagnostic. Requires clinician interpretation.
          </AlertDescription>
        </Alert>
        
        {isLoadingCheckins ? (
          <div className="flex items-center justify-center py-8">
            <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
            <span className="ml-2 text-sm text-muted-foreground">Loading check-ins...</span>
          </div>
        ) : checkinsError ? (
          <Alert variant="destructive">
            <AlertDescription className="text-xs">
              Failed to load check-in history. Please try again later.
            </AlertDescription>
          </Alert>
        ) : !recentCheckins || recentCheckins.length === 0 ? (
          <div className="text-center py-8">
            <p className="text-sm text-muted-foreground">No check-ins recorded in the last 7 days.</p>
            <p className="text-xs text-muted-foreground mt-1">Submit your first check-in above to start tracking.</p>
          </div>
        ) : (
          <div className="space-y-2 max-h-60 overflow-y-auto">
            {recentCheckins.slice(0, 5).map((checkin: any) => (
              <div key={checkin.id} className="rounded-md border bg-muted/30 p-3 space-y-2">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Calendar className="h-3 w-3 text-muted-foreground" />
                    <span className="text-xs font-medium">
                      {format(new Date(checkin.timestamp), "MMM d, h:mm a")}
                    </span>
                  </div>
                  <div className="flex gap-1">
                    {checkin.medicationsTaken && (
                      <Badge variant="outline" className="text-xs">Meds Taken</Badge>
                    )}
                  </div>
                </div>
                
                <div className="grid grid-cols-3 gap-2 text-xs">
                  <div>
                    <span className="text-muted-foreground">Pain: </span>
                    <span className="font-medium">{checkin.painLevel}/10</span>
                  </div>
                  <div>
                    <span className="text-muted-foreground">Fatigue: </span>
                    <span className="font-medium">{checkin.fatigueLevel || 0}/10</span>
                  </div>
                  <div>
                    <span className="text-muted-foreground">Sleep: </span>
                    <span className="font-medium">{checkin.sleepQuality}/10</span>
                  </div>
                </div>
                
                {checkin.note && (
                  <p className="text-xs text-muted-foreground line-clamp-2">{checkin.note}</p>
                )}
                
                {checkin.symptoms && checkin.symptoms.length > 0 && (
                  <div className="flex gap-1 flex-wrap">
                    {checkin.symptoms.slice(0, 3).map((symptom: string) => (
                      <Badge key={symptom} variant="secondary" className="text-xs">
                        {symptom}
                      </Badge>
                    ))}
                    {checkin.symptoms.length > 3 && (
                      <Badge variant="secondary" className="text-xs">
                        +{checkin.symptoms.length - 3} more
                      </Badge>
                    )}
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
