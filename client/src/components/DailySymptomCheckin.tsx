import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Form, FormControl, FormField, FormItem, FormLabel, FormMessage } from "@/components/ui/form";
import { useMutation, useQuery } from "@tanstack/react-query";
import { apiRequest, queryClient } from "@/lib/queryClient";
import { useToast } from "@/hooks/use-toast";
import { CheckCircle2, Loader2, Info, Calendar } from "lucide-react";
import { format } from "date-fns";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { dailyCheckinFormSchema } from "@shared/schema";
import type { z } from "zod";

type FormValues = z.infer<typeof dailyCheckinFormSchema>;

export function DailySymptomCheckin() {
  const { toast } = useToast();
  
  // Fetch unified symptom feed (patient-reported + AI-extracted from Agent Clona)
  const { data: unifiedFeed, isLoading: isLoadingFeed, error: feedError } = useQuery({
    queryKey: ["/api/symptom-checkin/feed/unified"]
  });

  // Form setup with validation
  const form = useForm<FormValues>({
    resolver: zodResolver(dailyCheckinFormSchema),
    defaultValues: {
      painLevel: 0,
      fatigueLevel: 0,
      breathlessnessLevel: 0,
      sleepQuality: 5,
      mood: "okay",
      mobilityScore: 5,
      medicationsTaken: true,
      symptoms: [],
      triggers: [],
      note: "",
    },
  });

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
    { value: "great" as const, label: "Great", color: "text-green-600" },
    { value: "good" as const, label: "Good", color: "text-green-500" },
    { value: "okay" as const, label: "Okay", color: "text-yellow-600" },
    { value: "low" as const, label: "Low", color: "text-orange-600" },
    { value: "very_low" as const, label: "Very Low", color: "text-red-600" }
  ];

  // Create check-in mutation
  const createCheckinMutation = useMutation({
    mutationFn: async (data: any) => {
      return apiRequest("/api/symptom-checkin/checkin", {
        method: "POST",
        body: JSON.stringify({ ...data, deviceType: "web" })
      });
    },
    onSuccess: () => {
      toast({
        title: "Check-in saved",
        description: "Your daily symptom check-in has been recorded."
      });
      // Reset form to default values
      form.reset();
      // Invalidate unified feed to refresh UI
      queryClient.invalidateQueries({ queryKey: ["/api/symptom-checkin/feed/unified"] });
    },
    onError: (error: any) => {
      toast({
        title: "Error saving check-in",
        description: error.message,
        variant: "destructive"
      });
    }
  });

  const onSubmit = (data: FormValues) => {
    createCheckinMutation.mutate(data);
  };

  const toggleSymptom = (symptom: string) => {
    const current = form.getValues("symptoms");
    const updated = current.includes(symptom)
      ? current.filter(s => s !== symptom)
      : [...current, symptom];
    form.setValue("symptoms", updated);
  };

  const toggleTrigger = (trigger: string) => {
    const current = form.getValues("triggers");
    const updated = current.includes(trigger)
      ? current.filter(t => t !== trigger)
      : [...current, trigger];
    form.setValue("triggers", updated);
  };

  return (
    <Form {...form}>
      <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-4">
        {/* Persistent Disclaimer */}
        <Alert data-testid="alert-disclaimer">
          <Info className="h-4 w-4" />
          <AlertDescription className="text-xs">
            <strong>Important:</strong> All responses are patient-reported and observational data. 
            This is not a diagnostic tool and all insights require clinician interpretation.
          </AlertDescription>
        </Alert>

        {/* Pain Level */}
        <FormField
          control={form.control}
          name="painLevel"
          render={({ field }) => (
            <FormItem>
              <div className="flex items-center justify-between">
                <FormLabel className="text-sm" data-testid="label-pain-level">
                  Pain Level (VAS): {field.value}/10
                </FormLabel>
                <Badge variant="outline" className="text-xs" data-testid="badge-pain-value">{field.value}</Badge>
              </div>
              <FormControl>
                <Slider
                  data-testid="slider-pain-level"
                  value={[field.value]}
                  onValueChange={(v) => field.onChange(v[0])}
                  max={10}
                  step={1}
                  className="w-full"
                />
              </FormControl>
              <p className="text-xs text-muted-foreground">
                0 = No pain, 10 = Worst pain imaginable
              </p>
              <FormMessage />
            </FormItem>
          )}
        />

        <Separator />

        {/* Fatigue Level */}
        <FormField
          control={form.control}
          name="fatigueLevel"
          render={({ field }) => (
            <FormItem>
              <div className="flex items-center justify-between">
                <FormLabel className="text-sm" data-testid="label-fatigue-level">
                  Fatigue: {field.value}/10
                </FormLabel>
                <Badge variant="outline" className="text-xs" data-testid="badge-fatigue-value">{field.value}</Badge>
              </div>
              <FormControl>
                <Slider
                  data-testid="slider-fatigue-level"
                  value={[field.value]}
                  onValueChange={(v) => field.onChange(v[0])}
                  max={10}
                  step={1}
                  className="w-full"
                />
              </FormControl>
              <FormMessage />
            </FormItem>
          )}
        />

        <Separator />

        {/* Breathlessness */}
        <FormField
          control={form.control}
          name="breathlessnessLevel"
          render={({ field }) => (
            <FormItem>
              <div className="flex items-center justify-between">
                <FormLabel className="text-sm" data-testid="label-breathlessness">
                  Breathlessness: {field.value}/10
                </FormLabel>
                <Badge variant="outline" className="text-xs" data-testid="badge-breathlessness-value">{field.value}</Badge>
              </div>
              <FormControl>
                <Slider
                  data-testid="slider-breathlessness"
                  value={[field.value]}
                  onValueChange={(v) => field.onChange(v[0])}
                  max={10}
                  step={1}
                  className="w-full"
                />
              </FormControl>
              <FormMessage />
            </FormItem>
          )}
        />

        <Separator />

        {/* Sleep Quality */}
        <FormField
          control={form.control}
          name="sleepQuality"
          render={({ field }) => (
            <FormItem>
              <div className="flex items-center justify-between">
                <FormLabel className="text-sm" data-testid="label-sleep-quality">
                  Sleep Quality: {field.value}/10
                </FormLabel>
                <Badge variant="outline" className="text-xs" data-testid="badge-sleep-value">{field.value}</Badge>
              </div>
              <FormControl>
                <Slider
                  data-testid="slider-sleep-quality"
                  value={[field.value]}
                  onValueChange={(v) => field.onChange(v[0])}
                  max={10}
                  step={1}
                  className="w-full"
                />
              </FormControl>
              <FormMessage />
            </FormItem>
          )}
        />

        <Separator />

        {/* Mood */}
        <FormField
          control={form.control}
          name="mood"
          render={({ field }) => (
            <FormItem>
              <FormLabel className="text-sm" data-testid="label-mood">Mood</FormLabel>
              <FormControl>
                <div className="flex gap-2 flex-wrap">
                  {moodOptions.map((option) => (
                    <Button
                      key={option.value}
                      type="button"
                      variant={field.value === option.value ? "default" : "outline"}
                      size="sm"
                      onClick={() => field.onChange(option.value)}
                      data-testid={`button-mood-${option.value}`}
                    >
                      {option.label}
                    </Button>
                  ))}
                </div>
              </FormControl>
              <FormMessage />
            </FormItem>
          )}
        />

        <Separator />

        {/* Mobility Score */}
        <FormField
          control={form.control}
          name="mobilityScore"
          render={({ field }) => (
            <FormItem>
              <div className="flex items-center justify-between">
                <FormLabel className="text-sm" data-testid="label-mobility">
                  Mobility: {field.value}/10
                </FormLabel>
                <Badge variant="outline" className="text-xs" data-testid="badge-mobility-value">{field.value}</Badge>
              </div>
              <FormControl>
                <Slider
                  data-testid="slider-mobility"
                  value={[field.value]}
                  onValueChange={(v) => field.onChange(v[0])}
                  max={10}
                  step={1}
                  className="w-full"
                />
              </FormControl>
              <FormMessage />
            </FormItem>
          )}
        />

        <Separator />

        {/* Medications Taken */}
        <FormField
          control={form.control}
          name="medicationsTaken"
          render={({ field }) => (
            <FormItem>
              <FormLabel className="text-sm" data-testid="label-medications">
                Medications taken as prescribed?
              </FormLabel>
              <FormControl>
                <div className="flex gap-2">
                  <Button
                    type="button"
                    variant={field.value ? "default" : "outline"}
                    size="sm"
                    onClick={() => field.onChange(true)}
                    data-testid="button-medications-yes"
                  >
                    Yes
                  </Button>
                  <Button
                    type="button"
                    variant={!field.value ? "default" : "outline"}
                    size="sm"
                    onClick={() => field.onChange(false)}
                    data-testid="button-medications-no"
                  >
                    No
                  </Button>
                </div>
              </FormControl>
              <FormMessage />
            </FormItem>
          )}
        />

        <Separator />

        {/* Symptoms */}
        <FormField
          control={form.control}
          name="symptoms"
          render={({ field }) => (
            <FormItem>
              <FormLabel className="text-sm" data-testid="label-symptoms">Symptoms Experienced Today</FormLabel>
              <FormControl>
                <div className="flex gap-1 flex-wrap">
                  {symptomOptions.map((symptom) => (
                    <Badge
                      key={symptom}
                      variant={field.value.includes(symptom) ? "default" : "outline"}
                      className="cursor-pointer hover-elevate active-elevate-2 text-xs"
                      onClick={() => toggleSymptom(symptom)}
                      data-testid={`badge-symptom-${symptom.replace(/\s+/g, "-")}`}
                    >
                      {symptom}
                    </Badge>
                  ))}
                </div>
              </FormControl>
              <FormMessage />
            </FormItem>
          )}
        />

        <Separator />

        {/* Triggers */}
        <FormField
          control={form.control}
          name="triggers"
          render={({ field }) => (
            <FormItem>
              <FormLabel className="text-sm" data-testid="label-triggers">Possible Triggers</FormLabel>
              <FormControl>
                <div className="flex gap-1 flex-wrap">
                  {triggerOptions.map((trigger) => (
                    <Badge
                      key={trigger}
                      variant={field.value.includes(trigger) ? "default" : "outline"}
                      className="cursor-pointer hover-elevate active-elevate-2 text-xs"
                      onClick={() => toggleTrigger(trigger)}
                      data-testid={`badge-trigger-${trigger.replace(/\s+/g, "-")}`}
                    >
                      {trigger}
                    </Badge>
                  ))}
                </div>
              </FormControl>
              <FormMessage />
            </FormItem>
          )}
        />

        <Separator />

        {/* Notes */}
        <FormField
          control={form.control}
          name="note"
          render={({ field }) => (
            <FormItem>
              <FormLabel htmlFor="note" className="text-sm" data-testid="label-note">Additional Notes (Optional)</FormLabel>
              <FormControl>
                <Textarea
                  {...field}
                  id="note"
                  data-testid="textarea-note"
                  placeholder="Any additional details about how you're feeling today..."
                  rows={3}
                  className="text-sm"
                />
              </FormControl>
              <FormMessage />
            </FormItem>
          )}
        />

        {/* Submit Button */}
        <Button
          type="submit"
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

        {/* Unified Symptom Feed (Patient-reported + AI-extracted) */}
        <Separator />
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <Label className="text-sm font-medium">Symptom Timeline (Last 7 Days)</Label>
            {unifiedFeed && unifiedFeed.length > 0 && (
              <Badge variant="secondary" className="text-xs">{unifiedFeed.length}</Badge>
            )}
          </div>
          
          {/* Persistent Disclaimer for Historical Data */}
          <Alert className="bg-muted/30" data-testid="alert-history-disclaimer">
            <Info className="h-3 w-3" />
            <AlertDescription className="text-xs">
              Timeline combines patient-reported check-ins and AI-observed symptoms from Agent Clona. All data is observational. Not diagnostic. Requires clinician interpretation.
            </AlertDescription>
          </Alert>
          
          {isLoadingFeed ? (
            <div className="flex items-center justify-center py-8">
              <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
              <span className="ml-2 text-sm text-muted-foreground">Loading symptom feed...</span>
            </div>
          ) : feedError ? (
            <Alert variant="destructive">
              <AlertDescription className="text-xs">
                Failed to load symptom feed. Please try again later.
              </AlertDescription>
            </Alert>
          ) : !unifiedFeed || unifiedFeed.length === 0 ? (
            <div className="text-center py-8">
              <p className="text-sm text-muted-foreground">No symptom data recorded in the last 7 days.</p>
              <p className="text-xs text-muted-foreground mt-1">Submit a check-in above or chat with Agent Clona to start tracking.</p>
            </div>
          ) : (
            <div className="space-y-2 max-h-96 overflow-y-auto">
              {unifiedFeed.slice(0, 10).map((item: any) => {
                const timestamp = item.timestamp ? new Date(item.timestamp) : new Date();
                const isValidDate = timestamp instanceof Date && !isNaN(timestamp.getTime());
                
                return (
                <div key={item.id} className="rounded-md border bg-muted/30 p-3 space-y-2">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <Calendar className="h-3 w-3 text-muted-foreground" />
                      <span className="text-xs font-medium">
                        {isValidDate ? format(timestamp, "MMM d, h:mm a") : "Invalid date"}
                      </span>
                    </div>
                    <div className="flex gap-1">
                      {/* Data Source Badge - Critical for HIPAA compliance */}
                      <Badge 
                        variant={item.dataSource === "patient-reported" ? "outline" : "secondary"} 
                        className="text-xs" 
                        data-testid={`badge-observational-${item.id}`}
                      >
                        {item.observationalLabel}
                      </Badge>
                      {item.confidence && (
                        <Badge variant="outline" className="text-xs">
                          {Math.round(item.confidence * 100)}% confidence
                        </Badge>
                      )}
                      {item.medicationsTaken && (
                        <Badge variant="outline" className="text-xs">Meds Taken</Badge>
                      )}
                    </div>
                  </div>
                  
                  {/* Patient-Reported Check-In */}
                  {item.dataSource === "patient-reported" && (
                    <>
                      <div className="grid grid-cols-3 gap-2 text-xs">
                        <div>
                          <span className="text-muted-foreground">Pain: </span>
                          <span className="font-medium">{item.painLevel}/10</span>
                        </div>
                        <div>
                          <span className="text-muted-foreground">Fatigue: </span>
                          <span className="font-medium">{item.fatigueLevel || 0}/10</span>
                        </div>
                        <div>
                          <span className="text-muted-foreground">Sleep: </span>
                          <span className="font-medium">{item.sleepQuality}/10</span>
                        </div>
                      </div>
                      
                      {item.note && (
                        <p className="text-xs text-muted-foreground line-clamp-2">{item.note}</p>
                      )}
                      
                      {item.symptoms && item.symptoms.length > 0 && (
                        <div className="flex gap-1 flex-wrap">
                          {item.symptoms.slice(0, 3).map((symptom: string) => (
                            <Badge key={symptom} variant="secondary" className="text-xs">
                              {symptom}
                            </Badge>
                          ))}
                          {item.symptoms.length > 3 && (
                            <Badge variant="secondary" className="text-xs">
                              +{item.symptoms.length - 3} more
                            </Badge>
                          )}
                        </div>
                      )}
                    </>
                  )}
                  
                  {/* AI-Extracted Symptoms from Agent Clona */}
                  {item.dataSource === "ai-extracted" && (
                    <>
                      {item.symptomTypes && item.symptomTypes.length > 0 && (
                        <div className="space-y-1">
                          <div className="text-xs font-medium text-muted-foreground">AI-detected symptoms:</div>
                          <div className="flex gap-1 flex-wrap">
                            {item.symptomTypes.map((type: string) => (
                              <Badge key={type} variant="default" className="text-xs bg-blue-100 text-blue-900 dark:bg-blue-900 dark:text-blue-100">
                                {type}
                              </Badge>
                            ))}
                          </div>
                        </div>
                      )}
                      
                      {item.locations && item.locations.length > 0 && (
                        <div className="text-xs">
                          <span className="text-muted-foreground">Locations: </span>
                          <span className="font-medium">{item.locations.join(", ")}</span>
                        </div>
                      )}
                      
                      {item.intensityMentions && item.intensityMentions.length > 0 && (
                        <div className="text-xs">
                          <span className="text-muted-foreground">Intensity: </span>
                          <span className="font-medium">{item.intensityMentions.join(", ")}</span>
                        </div>
                      )}
                      
                      {item.temporalInfo && (
                        <div className="text-xs">
                          <span className="text-muted-foreground">Timing: </span>
                          <span className="font-medium">{item.temporalInfo}</span>
                        </div>
                      )}
                      
                      {item.sessionId && (
                        <div className="text-xs text-muted-foreground">
                          From chat session: {item.sessionId.slice(0, 8)}...
                        </div>
                      )}
                    </>
                  )}
                </div>
                );
              })}
            </div>
          )}
        </div>
      </form>
    </Form>
  );
}
