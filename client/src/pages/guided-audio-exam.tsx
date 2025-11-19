import { useState, useRef, useEffect } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { useToast } from "@/hooks/use-toast";
import { apiRequest, queryClient } from "@/lib/queryClient";
import { useAuth } from "@/contexts/AuthContext";
import { Mic, MicOff, Play, Pause, CheckCircle2, AlertCircle, Volume2 } from "lucide-react";

type AudioStage = "breathing" | "coughing" | "speaking" | "reading";

interface AudioSession {
  session_id: string;
  status: string;
  current_stage: AudioStage | null;
  prep_time_seconds: number;
  prioritized_stages: AudioStage[];
}

interface StageInfo {
  name: string;
  icon: React.ReactNode;
  description: string;
  instructions: string;
  duration: number;
}

interface AudioAnalysisResults {
  session_id: string;
  yamnet_available: boolean;
  top_audio_event: string | null;
  cough_probability_ml: number;
  speech_probability_ml: number;
  breathing_probability_ml: number;
  wheeze_probability_ml: number;
  speech_fluency_score: number | null;
  voice_weakness_index: number | null;
  pause_frequency_per_minute: number | null;
  vocal_amplitude_db: number | null;
  breath_rate_per_minute: number | null;
  wheeze_detected: boolean;
  cough_count: number;
  analysis_confidence: number;
  recommendations: string[];
}

const STAGE_INFO: Record<AudioStage, StageInfo> = {
  breathing: {
    name: "Breathing",
    icon: <Volume2 className="w-5 h-5" />,
    description: "Deep Breathing Exercise",
    instructions: "Take 5-6 deep breaths. Breathe in slowly through your nose, then exhale through your mouth. Try to breathe naturally and deeply.",
    duration: 30
  },
  coughing: {
    name: "Coughing",
    icon: <Volume2 className="w-5 h-5" />,
    description: "Voluntary Cough",
    instructions: "Cough 2-3 times as you normally would. This helps us detect any changes in your respiratory patterns.",
    duration: 15
  },
  speaking: {
    name: "Speaking",
    icon: <Mic className="w-5 h-5" />,
    description: "Free Speech",
    instructions: "Speak naturally for 30 seconds. You can describe your day, talk about a hobby, or tell a short story. Speak at a comfortable pace.",
    duration: 30
  },
  reading: {
    name: "Reading",
    icon: <Mic className="w-5 h-5" />,
    description: "Reading Passage",
    instructions: "Read this passage aloud at your normal speaking pace:\n\n\"When the sunlight strikes raindrops in the air, they act as a prism and form a rainbow. The rainbow is a division of white light into many beautiful colors.\"",
    duration: 40
  }
};

export default function GuidedAudioExam() {
  const { toast } = useToast();
  const { user } = useAuth();
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [currentStage, setCurrentStage] = useState<AudioStage | null>(null);
  const [isRecording, setIsRecording] = useState(false);
  const [recordingTime, setRecordingTime] = useState(0);
  const [prepCountdown, setPrepCountdown] = useState(30);
  const [showPrep, setShowPrep] = useState(false);
  const [completedStages, setCompletedStages] = useState<Set<AudioStage>>(new Set());
  const [examComplete, setExamComplete] = useState(false);
  
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<NodeJS.Timeout | null>(null);

  // Create examination session
  const createSessionMutation = useMutation({
    mutationFn: async () => {
      // FIXED: Use real authenticated user ID instead of mock
      if (!user?.id) {
        throw new Error("User not authenticated");
      }
      
      const response = await apiRequest("/api/v1/guided-audio-exam/sessions", {
        method: "POST",
        json: {
          patient_id: String(user.id),
          device_info: {
            userAgent: navigator.userAgent,
            platform: navigator.platform
          }
        }
      });
      return await response.json();
    },
    onSuccess: (data) => {
      setSessionId(data.session_id);
      setCurrentStage(data.current_stage);
      toast({
        title: "Session Created",
        description: "Guided audio examination session started successfully"
      });
    },
    onError: (error: Error) => {
      toast({
        variant: "destructive",
        title: "Session Creation Failed",
        description: error.message
      });
    }
  });

  // Upload audio segment
  const uploadAudioMutation = useMutation({
    mutationFn: async ({ stage, audioBlob }: { stage: AudioStage; audioBlob: Blob }) => {
      // Convert blob to base64
      const base64Audio = await new Promise<string>((resolve) => {
        const reader = new FileReader();
        reader.onloadend = () => {
          const base64 = reader.result as string;
          const base64Data = base64.split(",")[1];
          resolve(base64Data);
        };
        reader.readAsDataURL(audioBlob);
      });

      const response = await apiRequest(`/api/v1/guided-audio-exam/sessions/${sessionId}/upload`, {
        method: "POST",
        json: {
          stage,
          audio_base64: base64Audio,
          duration_seconds: recordingTime
        }
      });
      return await response.json();
    },
    onSuccess: (data: any, variables) => {
      setCompletedStages(prev => new Set(prev).add(variables.stage));
      
      if (data.next_stage) {
        setCurrentStage(data.next_stage as AudioStage);
        toast({
          title: "Stage Complete",
          description: `${STAGE_INFO[variables.stage].name} recorded. Moving to ${STAGE_INFO[data.next_stage as AudioStage].name}.`
        });
      } else {
        toast({
          title: "All Stages Complete",
          description: "Completing examination and running AI analysis..."
        });
        completeSessionMutation.mutate();
      }
    },
    onError: (error: Error) => {
      toast({
        variant: "destructive",
        title: "Upload Failed",
        description: error.message
      });
    }
  });

  // Complete examination session
  const completeSessionMutation = useMutation({
    mutationFn: async () => {
      const response = await apiRequest(`/api/v1/guided-audio-exam/sessions/${sessionId}/complete`, {
        method: "POST"
      });
      return await response.json();
    },
    onSuccess: () => {
      setExamComplete(true);
      queryClient.invalidateQueries({ queryKey: ["/api/v1/guided-audio-exam/sessions", sessionId, "results"] });
      toast({
        title: "Examination Complete",
        description: "AI analysis completed successfully. View your results below."
      });
    },
    onError: (error: Error) => {
      toast({
        variant: "destructive",
        title: "Analysis Failed",
        description: error.message
      });
    }
  });

  // Fetch results after completion
  const { data: analysisResults, isLoading: resultsLoading } = useQuery<AudioAnalysisResults>({
    queryKey: ["/api/v1/guided-audio-exam/sessions", sessionId, "results"],
    enabled: examComplete && !!sessionId,
    retry: 2
  });

  // Prep countdown timer
  useEffect(() => {
    if (showPrep && prepCountdown > 0) {
      const timer = setTimeout(() => {
        setPrepCountdown(prev => prev - 1);
      }, 1000);
      return () => clearTimeout(timer);
    } else if (showPrep && prepCountdown === 0) {
      setShowPrep(false);
      setPrepCountdown(30);
      startRecording();
    }
  }, [showPrep, prepCountdown]);

  // Recording timer
  useEffect(() => {
    if (isRecording) {
      timerRef.current = setInterval(() => {
        setRecordingTime(prev => prev + 1);
      }, 1000);
    } else if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [isRecording]);

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: "audio/wav" });
        if (currentStage) {
          uploadAudioMutation.mutate({ stage: currentStage, audioBlob });
        }
        
        stream.getTracks().forEach(track => track.stop());
      };

      mediaRecorder.start();
      setIsRecording(true);
      setRecordingTime(0);
    } catch (error) {
      toast({
        variant: "destructive",
        title: "Microphone Access Denied",
        description: "Please allow microphone access to continue."
      });
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  };

  const handleStageStart = () => {
    setShowPrep(true);
  };

  const handleStartExam = () => {
    createSessionMutation.mutate();
  };

  const progressPercent = (completedStages.size / 4) * 100;

  // FIXED: Authentication gate
  if (!user) {
    return (
      <div className="container mx-auto p-6 max-w-4xl">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <AlertCircle className="w-6 h-6 text-destructive" />
              Authentication Required
            </CardTitle>
            <CardDescription>
              You must be logged in to access the guided audio examination.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-muted-foreground">
              Please log in to continue with your personalized audio wellness monitoring.
            </p>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="container mx-auto p-6 max-w-4xl">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Mic className="w-6 h-6" />
            Guided Audio Examination
          </CardTitle>
          <CardDescription>
            AI-powered audio wellness monitoring with respiratory and neurological assessment
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          {!sessionId && !examComplete && (
            <div className="space-y-4">
              <div className="bg-muted p-4 rounded-lg">
                <h3 className="font-semibold mb-2">What to Expect</h3>
                <ul className="space-y-2 text-sm">
                  <li className="flex items-start gap-2">
                    <CheckCircle2 className="w-4 h-4 mt-0.5 text-primary" />
                    <span><strong>4 Audio Stages:</strong> Breathing, Coughing, Speaking, Reading</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <CheckCircle2 className="w-4 h-4 mt-0.5 text-primary" />
                    <span><strong>30-Second Prep:</strong> Instructions before each stage</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <CheckCircle2 className="w-4 h-4 mt-0.5 text-primary" />
                    <span><strong>AI Analysis:</strong> YAMNet ML + neurological metrics</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <CheckCircle2 className="w-4 h-4 mt-0.5 text-primary" />
                    <span><strong>Quiet Environment:</strong> Find a quiet space for best results</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <CheckCircle2 className="w-4 h-4 mt-0.5 text-primary" />
                    <span><strong>Personalized:</strong> Examination prioritized for your health conditions</span>
                  </li>
                </ul>
              </div>
              
              <Button 
                onClick={handleStartExam}
                disabled={createSessionMutation.isPending || !user}
                size="lg"
                className="w-full"
                data-testid="button-start-exam"
              >
                {createSessionMutation.isPending ? "Creating Session..." : "Start Audio Examination"}
              </Button>
            </div>
          )}

          {sessionId && !examComplete && (
            <>
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium">Progress</span>
                  <span className="text-sm text-muted-foreground">
                    {completedStages.size} / 4 stages complete
                  </span>
                </div>
                <Progress value={progressPercent} />
              </div>

              <div className="grid grid-cols-4 gap-2">
                {(["breathing", "coughing", "speaking", "reading"] as AudioStage[]).map((stage) => (
                  <Badge 
                    key={stage}
                    variant={completedStages.has(stage) ? "default" : "outline"}
                    className="justify-center"
                  >
                    {completedStages.has(stage) && <CheckCircle2 className="w-3 h-3 mr-1" />}
                    {STAGE_INFO[stage].name}
                  </Badge>
                ))}
              </div>

              {currentStage && !showPrep && !isRecording && (
                <Card className="bg-primary/5 border-primary/20">
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2 text-lg">
                      {STAGE_INFO[currentStage].icon}
                      {STAGE_INFO[currentStage].name} Stage
                    </CardTitle>
                    <CardDescription>
                      {STAGE_INFO[currentStage].description}
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="bg-background p-4 rounded-lg">
                      <p className="text-sm whitespace-pre-line">
                        {STAGE_INFO[currentStage].instructions}
                      </p>
                    </div>
                    <Button 
                      onClick={handleStageStart}
                      className="w-full"
                      data-testid={`button-start-${currentStage}`}
                    >
                      Begin {STAGE_INFO[currentStage].name}
                    </Button>
                  </CardContent>
                </Card>
              )}

              {showPrep && (
                <Card className="bg-primary text-primary-foreground">
                  <CardContent className="pt-6">
                    <div className="text-center space-y-4">
                      <div className="text-6xl font-bold">{prepCountdown}</div>
                      <p className="text-lg">Get ready to record...</p>
                      <p className="text-sm opacity-90">
                        {currentStage && STAGE_INFO[currentStage].description}
                      </p>
                    </div>
                  </CardContent>
                </Card>
              )}

              {isRecording && currentStage && (
                <Card className="bg-destructive/10 border-destructive/30">
                  <CardContent className="pt-6">
                    <div className="text-center space-y-4">
                      <div className="flex justify-center">
                        <div className="w-20 h-20 bg-destructive rounded-full flex items-center justify-center animate-pulse">
                          <Mic className="w-10 h-10 text-destructive-foreground" />
                        </div>
                      </div>
                      <div className="text-4xl font-bold">{recordingTime}s</div>
                      <p className="text-lg font-semibold">Recording {STAGE_INFO[currentStage].name}...</p>
                      <Button 
                        onClick={stopRecording}
                        variant="destructive"
                        size="lg"
                        disabled={uploadAudioMutation.isPending}
                        data-testid="button-stop-recording"
                      >
                        {uploadAudioMutation.isPending ? "Uploading..." : "Stop & Continue"}
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              )}
            </>
          )}

          {examComplete && (
            <Card className="bg-primary/5 border-primary/20">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <CheckCircle2 className="w-6 h-6 text-primary" />
                  Examination Complete!
                </CardTitle>
                <CardDescription>
                  AI analysis has been completed. Your audio wellness metrics are ready.
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div className="bg-background p-4 rounded-lg">
                    <div className="text-sm text-muted-foreground">Stages Completed</div>
                    <div className="text-2xl font-bold">{completedStages.size}/4</div>
                  </div>
                  <div className="bg-background p-4 rounded-lg">
                    <div className="text-sm text-muted-foreground">Analysis Status</div>
                    <div className="text-2xl font-bold text-primary">Complete</div>
                  </div>
                </div>
                
                {analysisResults && (
                  <div className="space-y-4">
                    <div className="bg-muted p-4 rounded-lg space-y-3">
                      <h4 className="font-semibold">AI Analysis Results</h4>
                      
                      <div className="grid grid-cols-2 gap-3">
                        <div className="bg-background p-3 rounded">
                          <div className="text-xs text-muted-foreground">Speech Fluency</div>
                          <div className="text-lg font-bold">{analysisResults.speech_fluency_score?.toFixed(1) || "N/A"}/100</div>
                        </div>
                        <div className="bg-background p-3 rounded">
                          <div className="text-xs text-muted-foreground">Voice Weakness</div>
                          <div className="text-lg font-bold">{analysisResults.voice_weakness_index?.toFixed(1) || "N/A"}/100</div>
                        </div>
                        <div className="bg-background p-3 rounded">
                          <div className="text-xs text-muted-foreground">Breath Rate</div>
                          <div className="text-lg font-bold">{analysisResults.breath_rate_per_minute?.toFixed(0) || "N/A"} /min</div>
                        </div>
                        <div className="bg-background p-3 rounded">
                          <div className="text-xs text-muted-foreground">Cough Count</div>
                          <div className="text-lg font-bold">{analysisResults.cough_count || 0}</div>
                        </div>
                      </div>
                      
                      <div className="text-sm space-y-2">
                        <div className="flex items-center justify-between">
                          <span className="text-muted-foreground">ML Cough Probability:</span>
                          <span className="font-semibold">{(analysisResults.cough_probability_ml * 100).toFixed(1)}%</span>
                        </div>
                        <div className="flex items-center justify-between">
                          <span className="text-muted-foreground">ML Speech Probability:</span>
                          <span className="font-semibold">{(analysisResults.speech_probability_ml * 100).toFixed(1)}%</span>
                        </div>
                        <div className="flex items-center justify-between">
                          <span className="text-muted-foreground">Wheeze Detected:</span>
                          <Badge variant={analysisResults.wheeze_detected ? "destructive" : "default"}>
                            {analysisResults.wheeze_detected ? "Yes" : "No"}
                          </Badge>
                        </div>
                      </div>
                    </div>

                    {analysisResults.recommendations && analysisResults.recommendations.length > 0 && (
                      <div className="bg-muted p-4 rounded-lg space-y-2">
                        <h4 className="font-semibold">Recommendations:</h4>
                        <ul className="text-sm space-y-1">
                          {analysisResults.recommendations.map((rec: string, idx: number) => (
                            <li key={idx}>✓ {rec}</li>
                          ))}
                        </ul>
                      </div>
                    )}
                  </div>
                )}

                <Button 
                  onClick={() => window.location.reload()}
                  variant="outline"
                  className="w-full"
                  data-testid="button-new-exam"
                >
                  Start New Examination
                </Button>
              </CardContent>
            </Card>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
