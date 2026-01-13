import { useState, useEffect } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { apiRequest, queryClient } from "@/lib/queryClient";
import { Card, CardContent, CardDescription, CardHeader, CardTitle, CardFooter } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Progress } from "@/components/ui/progress";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { useToast } from "@/hooks/use-toast";
import { 
  Brain, ChevronRight, ChevronLeft, Save, CheckCircle2, 
  AlertTriangle, Phone, MessageSquare, ExternalLink, Loader2
} from "lucide-react";

interface CBTPrompt {
  id: string;
  prompt: string;
  help_text: string;
}

interface CBTSessionData {
  id: string;
  session_type: string;
  prompts: CBTPrompt[];
  created_at: string;
}

interface CrisisResources {
  national_suicide_prevention: string;
  crisis_text_line: string;
  international: string;
  emergency: string;
}

interface CBTSessionPanelProps {
  patientId?: string;
}

export function CBTSessionPanel({ patientId }: CBTSessionPanelProps) {
  const { toast } = useToast();
  const [currentStep, setCurrentStep] = useState(0);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [responses, setResponses] = useState<Record<string, string>>({});
  const [distressBefore, setDistressBefore] = useState(50);
  const [distressAfter, setDistressAfter] = useState<number | null>(null);
  const [crisisDetected, setCrisisDetected] = useState(false);
  const [crisisResources, setCrisisResources] = useState<CrisisResources | null>(null);

  const { data: prompts, isLoading: loadingPrompts } = useQuery<{ prompts: CBTPrompt[]; crisis_resources: CrisisResources }>({
    queryKey: ["/api/v1/cbt/prompts"],
    staleTime: 24 * 60 * 60 * 1000,
  });

  const createSessionMutation = useMutation({
    mutationFn: async () => {
      const url = `/api/v1/cbt/patient/${patientId || "me"}/sessions`;
      const res = await apiRequest(url, {
        method: "POST",
        json: { session_type: "thought_record" }
      });
      return await res.json() as CBTSessionData;
    },
    onSuccess: (data: CBTSessionData) => {
      setSessionId(data.id);
      setCurrentStep(0);
      setResponses({});
      setDistressBefore(50);
      setDistressAfter(null);
      setCrisisDetected(false);
    },
    onError: (error: Error) => {
      toast({
        title: "Error",
        description: error.message || "Failed to start session",
        variant: "destructive",
      });
    },
  });

  const updateSessionMutation = useMutation({
    mutationFn: async (updates: Record<string, any>) => {
      if (!sessionId) throw new Error("No session active");
      const url = `/api/v1/cbt/patient/${patientId || "me"}/sessions/${sessionId}`;
      const res = await apiRequest(url, { method: "PATCH", json: updates });
      return await res.json() as { crisis_detected?: boolean; crisis_resources?: CrisisResources };
    },
    onSuccess: (data: { crisis_detected?: boolean; crisis_resources?: CrisisResources }) => {
      if (data.crisis_detected) {
        setCrisisDetected(true);
        setCrisisResources(data.crisis_resources || prompts?.crisis_resources || null);
      }
    },
    onError: (error: Error) => {
      toast({
        title: "Error saving",
        description: error.message || "Failed to save your response",
        variant: "destructive",
      });
    },
  });

  const completeSessionMutation = useMutation({
    mutationFn: async () => {
      if (!sessionId) throw new Error("No session active");
      const url = `/api/v1/cbt/patient/${patientId || "me"}/sessions/${sessionId}/complete`;
      const res = await apiRequest(url, { method: "POST", json: {} });
      return await res.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/v1/cbt/patient", patientId || "me", "sessions"] });
      toast({
        title: "Session completed",
        description: "Great work! Your thought record has been saved.",
      });
      setSessionId(null);
      setCurrentStep(0);
      setResponses({});
    },
    onError: (error: Error) => {
      toast({
        title: "Error",
        description: error.message || "Failed to complete session",
        variant: "destructive",
      });
    },
  });

  const currentPrompt = prompts?.prompts?.[currentStep];
  const totalSteps = prompts?.prompts?.length || 7;
  const progress = ((currentStep + 1) / totalSteps) * 100;

  const handleResponseChange = (value: string) => {
    if (!currentPrompt) return;
    setResponses(prev => ({ ...prev, [currentPrompt.id]: value }));
  };

  const handleNext = async () => {
    if (!currentPrompt) return;
    
    const fieldMap: Record<string, string> = {
      situation: "situation",
      automatic_thoughts: "automatic_thoughts",
      emotions: "emotions",
      evidence_for: "evidence_for",
      evidence_against: "evidence_against",
      balanced_thought: "balanced_thought",
      action_plan: "action_plan",
    };

    const fieldKey = fieldMap[currentPrompt.id];
    if (fieldKey && responses[currentPrompt.id]) {
      await updateSessionMutation.mutateAsync({
        [fieldKey]: currentPrompt.id === "emotions" 
          ? { text: responses[currentPrompt.id] }
          : responses[currentPrompt.id]
      });
    }

    if (currentStep < totalSteps - 1) {
      setCurrentStep(prev => prev + 1);
    }
  };

  const handlePrevious = () => {
    if (currentStep > 0) {
      setCurrentStep(prev => prev - 1);
    }
  };

  const handleComplete = async () => {
    if (distressAfter !== null) {
      await updateSessionMutation.mutateAsync({
        distress_before: distressBefore,
        distress_after: distressAfter
      });
    }
    await completeSessionMutation.mutateAsync();
  };

  if (!sessionId) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Brain className="h-5 w-5" />
            CBT Thought Record
          </CardTitle>
          <CardDescription>
            Work through difficult thoughts using structured cognitive behavioral therapy techniques
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="p-4 bg-muted/50 rounded-lg space-y-2">
            <p className="text-sm font-medium">What is a Thought Record?</p>
            <p className="text-sm text-muted-foreground">
              A thought record helps you identify and challenge unhelpful thinking patterns. 
              By examining your thoughts objectively, you can develop more balanced perspectives 
              and reduce emotional distress.
            </p>
          </div>

          <div className="space-y-2">
            <Label>How distressed are you feeling right now? (0-100)</Label>
            <Slider
              value={[distressBefore]}
              onValueChange={([value]) => setDistressBefore(value)}
              max={100}
              step={1}
              className="w-full"
            />
            <div className="flex justify-between text-xs text-muted-foreground">
              <span>No distress (0)</span>
              <span className="font-medium">{distressBefore}</span>
              <span>Extreme distress (100)</span>
            </div>
          </div>
        </CardContent>
        <CardFooter>
          <Button 
            onClick={() => createSessionMutation.mutate()}
            disabled={createSessionMutation.isPending}
            className="w-full"
            data-testid="button-start-cbt-session"
          >
            {createSessionMutation.isPending ? (
              <Loader2 className="h-4 w-4 mr-2 animate-spin" />
            ) : (
              <Brain className="h-4 w-4 mr-2" />
            )}
            Start Thought Record
          </Button>
        </CardFooter>
      </Card>
    );
  }

  if (crisisDetected && crisisResources) {
    return (
      <Card className="border-red-200 dark:border-red-900">
        <CardHeader className="bg-red-50 dark:bg-red-950/30">
          <CardTitle className="flex items-center gap-2 text-red-700 dark:text-red-400">
            <AlertTriangle className="h-5 w-5" />
            We're Here to Help
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4 pt-4">
          <Alert variant="destructive">
            <AlertTriangle className="h-4 w-4" />
            <AlertTitle>Your safety matters</AlertTitle>
            <AlertDescription>
              Based on what you've shared, we want to make sure you have support. 
              Your care team has been notified. Please reach out to one of these resources if you need immediate help.
            </AlertDescription>
          </Alert>

          <div className="space-y-3">
            <a 
              href="tel:988" 
              className="flex items-center gap-3 p-4 border rounded-lg hover:bg-muted/50 transition-colors"
            >
              <Phone className="h-5 w-5 text-red-600" />
              <div>
                <p className="font-medium">988 Suicide & Crisis Lifeline</p>
                <p className="text-sm text-muted-foreground">Call or text 988</p>
              </div>
            </a>

            <a 
              href="sms:741741&body=HOME" 
              className="flex items-center gap-3 p-4 border rounded-lg hover:bg-muted/50 transition-colors"
            >
              <MessageSquare className="h-5 w-5 text-blue-600" />
              <div>
                <p className="font-medium">Crisis Text Line</p>
                <p className="text-sm text-muted-foreground">Text HOME to 741741</p>
              </div>
            </a>

            <a 
              href="https://findahelpline.com" 
              target="_blank" 
              rel="noopener noreferrer"
              className="flex items-center gap-3 p-4 border rounded-lg hover:bg-muted/50 transition-colors"
            >
              <ExternalLink className="h-5 w-5 text-green-600" />
              <div>
                <p className="font-medium">Find a Helpline</p>
                <p className="text-sm text-muted-foreground">International crisis resources</p>
              </div>
            </a>
          </div>

          <Separator />

          <Button
            variant="outline"
            onClick={() => setCrisisDetected(false)}
            className="w-full"
          >
            Continue with Thought Record
          </Button>
        </CardContent>
      </Card>
    );
  }

  if (currentStep === totalSteps - 1 && Object.keys(responses).length >= totalSteps - 1) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <CheckCircle2 className="h-5 w-5 text-green-600" />
            Review & Complete
          </CardTitle>
          <CardDescription>
            Excellent work! Let's see how you're feeling now.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <Label>How distressed are you feeling now? (0-100)</Label>
            <Slider
              value={[distressAfter ?? distressBefore]}
              onValueChange={([value]) => setDistressAfter(value)}
              max={100}
              step={1}
              className="w-full"
            />
            <div className="flex justify-between text-xs text-muted-foreground">
              <span>No distress (0)</span>
              <span className="font-medium">{distressAfter ?? distressBefore}</span>
              <span>Extreme distress (100)</span>
            </div>
          </div>

          {distressAfter !== null && distressAfter < distressBefore && (
            <Alert className="bg-green-50 dark:bg-green-950/30 border-green-200">
              <CheckCircle2 className="h-4 w-4 text-green-600" />
              <AlertDescription>
                Your distress decreased by {distressBefore - distressAfter} points. Great progress!
              </AlertDescription>
            </Alert>
          )}
        </CardContent>
        <CardFooter className="flex gap-2">
          <Button variant="outline" onClick={handlePrevious}>
            <ChevronLeft className="h-4 w-4 mr-2" />
            Back
          </Button>
          <Button 
            onClick={handleComplete}
            disabled={completeSessionMutation.isPending}
            className="flex-1"
            data-testid="button-complete-cbt-session"
          >
            {completeSessionMutation.isPending ? (
              <Loader2 className="h-4 w-4 mr-2 animate-spin" />
            ) : (
              <CheckCircle2 className="h-4 w-4 mr-2" />
            )}
            Complete Session
          </Button>
        </CardFooter>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between mb-2">
          <Badge variant="outline">Step {currentStep + 1} of {totalSteps}</Badge>
          <span className="text-sm text-muted-foreground">{Math.round(progress)}% complete</span>
        </div>
        <Progress value={progress} className="h-2" />
        <CardTitle className="mt-4">{currentPrompt?.prompt}</CardTitle>
        {currentPrompt?.help_text && (
          <CardDescription>{currentPrompt.help_text}</CardDescription>
        )}
      </CardHeader>
      <CardContent>
        <Textarea
          placeholder="Type your response here..."
          value={responses[currentPrompt?.id || ""] || ""}
          onChange={(e) => handleResponseChange(e.target.value)}
          className="min-h-[150px]"
          data-testid={`textarea-cbt-${currentPrompt?.id}`}
        />
      </CardContent>
      <CardFooter className="flex justify-between">
        <Button 
          variant="outline" 
          onClick={handlePrevious}
          disabled={currentStep === 0}
        >
          <ChevronLeft className="h-4 w-4 mr-2" />
          Previous
        </Button>
        <Button 
          onClick={handleNext}
          disabled={updateSessionMutation.isPending || !responses[currentPrompt?.id || ""]}
          data-testid="button-cbt-next"
        >
          {updateSessionMutation.isPending ? (
            <Loader2 className="h-4 w-4 mr-2 animate-spin" />
          ) : (
            <>
              Next
              <ChevronRight className="h-4 w-4 ml-2" />
            </>
          )}
        </Button>
      </CardFooter>
    </Card>
  );
}

export default CBTSessionPanel;
