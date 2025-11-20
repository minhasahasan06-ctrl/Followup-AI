import { HealthMetricCard } from "@/components/HealthMetricCard";
import { MedicationCard } from "@/components/MedicationCard";
import { FollowUpCard } from "@/components/FollowUpCard";
import { ReminderCard } from "@/components/ReminderCard";
import { EmergencyAlert } from "@/components/EmergencyAlert";
import DynamicWelcome from "@/components/DynamicWelcome";
import { LegalDisclaimer } from "@/components/LegalDisclaimer";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Heart, Activity, Droplet, Moon, TrendingUp, Calendar, CheckCircle, Brain, Video, Eye, Hand, Smile, Play, Wind, Palette, Zap, Users, Mic, Volume2, MicOff, Pause, CheckCircle2, AlertCircle, Camera, TrendingDown, Minus, User, Info, GitCompare } from "lucide-react";
import { useState, useRef, useEffect } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { useAuth } from "@/hooks/useAuth";
import { Link } from "wouter";
import type { DailyFollowup, Medication, DynamicTask, BehavioralInsight } from "@shared/schema";
import { useToast } from "@/hooks/use-toast";
import { apiRequest, queryClient } from "@/lib/queryClient";
import { Progress } from "@/components/ui/progress";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { format } from "date-fns";

// Audio examination types
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

// Symptom Journal types
interface SymptomMeasurement {
  id: number;
  body_area: string;
  created_at: string;
  color_change_percent: number | null;
  area_change_percent: number | null;
  respiratory_rate_bpm: number | null;
  ai_observations: string;
  detected_changes: string[];
  image_url?: string;
  alerts: Array<{
    severity: string;
    title: string;
    message: string;
    change_percent: number;
  }>;
}

const BODY_AREAS = [
  { value: "legs", label: "Legs (Swelling)", icon: Activity },
  { value: "face", label: "Face (Color/Swelling)", icon: User },
  { value: "eyes", label: "Eyes (Redness/Discharge)", icon: Eye },
  { value: "chest", label: "Breathing (Chest)", icon: Heart },
];

const STAGE_INFO: Record<AudioStage, StageInfo> = {
  breathing: {
    name: "Breathing",
    icon: <Volume2 className="w-4 h-4" />,
    description: "Deep Breathing Exercise",
    instructions: "Take 5-6 deep breaths. Breathe in slowly through your nose, then exhale through your mouth. Try to breathe naturally and deeply.",
    duration: 30
  },
  coughing: {
    name: "Coughing",
    icon: <Volume2 className="w-4 h-4" />,
    description: "Voluntary Cough",
    instructions: "Cough 2-3 times as you normally would. This helps us detect any changes in your respiratory patterns.",
    duration: 15
  },
  speaking: {
    name: "Speaking",
    icon: <Mic className="w-4 h-4" />,
    description: "Free Speech",
    instructions: "Speak naturally for 30 seconds. You can describe your day, talk about a hobby, or tell a short story. Speak at a comfortable pace.",
    duration: 30
  },
  reading: {
    name: "Reading",
    icon: <Mic className="w-4 h-4" />,
    description: "Reading Passage",
    instructions: "Read this passage aloud at your normal speaking pace:\n\n\"When the sunlight strikes raindrops in the air, they act as a prism and form a rainbow. The rainbow is a division of white light into many beautiful colors.\"",
    duration: 40
  }
};

export default function Dashboard() {
  const [showEmergency, setShowEmergency] = useState(false);
  const { user } = useAuth();
  const { toast } = useToast();

  // Audio examination state
  const [audioSessionId, setAudioSessionId] = useState<string | null>(null);
  const [currentAudioStage, setCurrentAudioStage] = useState<AudioStage | null>(null);
  const [isRecording, setIsRecording] = useState(false);
  const [recordingTime, setRecordingTime] = useState(0);
  const [prepCountdown, setPrepCountdown] = useState(30);
  const [showPrep, setShowPrep] = useState(false);
  const [completedStages, setCompletedStages] = useState<Set<AudioStage>>(new Set());
  const [examComplete, setExamComplete] = useState(false);
  
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<NodeJS.Timeout | null>(null);

  // Symptom Journal state
  const [selectedBodyArea, setSelectedBodyArea] = useState<string>("legs");
  const [symptomActiveTab, setSymptomActiveTab] = useState("capture");
  const [capturedImage, setCapturedImage] = useState<string | null>(null);
  const [patientNotes, setPatientNotes] = useState("");
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [countdown, setCountdown] = useState(0);
  const [isRecordingSymptom, setIsRecordingSymptom] = useState(false);
  const [isRecordingVideo, setIsRecordingVideo] = useState(false);
  const [recordedVideoBlob, setRecordedVideoBlob] = useState<Blob | null>(null);
  const [respiratoryResult, setRespiratoryResult] = useState<any | null>(null);
  const [showComparison, setShowComparison] = useState(false);
  const [selectedMeasurement1, setSelectedMeasurement1] = useState<number | null>(null);
  const [selectedMeasurement2, setSelectedMeasurement2] = useState<number | null>(null);
  
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const symptomMediaRecorderRef = useRef<MediaRecorder | null>(null);
  const recordedChunksRef = useRef<Blob[]>([]);

  const { data: todayFollowup } = useQuery<DailyFollowup>({
    queryKey: ["/api/daily-followup/today"],
  });

  const { data: medications } = useQuery<Medication[]>({
    queryKey: ["/api/medications"],
  });

  const { data: tasks } = useQuery<DynamicTask[]>({
    queryKey: ["/api/tasks"],
  });

  const { data: insights } = useQuery<BehavioralInsight[]>({
    queryKey: ["/api/behavioral-insights"],
  });

  // Fetch latest video AI metrics via Express proxy
  const { data: latestVideoMetrics } = useQuery<any>({
    queryKey: ["/api/video-ai/latest-metrics"],
    enabled: !!user,
  });

  // Check if metrics are from today (12am-12am)
  const isMetricsFromToday = latestVideoMetrics?.created_at 
    ? new Date(latestVideoMetrics.created_at).toDateString() === new Date().toDateString()
    : false;

  // Symptom Journal queries
  const measurementsUrl = `/api/v1/symptom-journal/measurements/recent?${new URLSearchParams({
    ...(selectedBodyArea && { body_area: selectedBodyArea }),
    days: "30"
  })}`;
  
  const { data: recentData, isLoading: loadingRecent } = useQuery<{ measurements: SymptomMeasurement[] }>({
    queryKey: [measurementsUrl],
  });

  const { data: alertsData } = useQuery<{ alerts: any[] }>({
    queryKey: ["/api/v1/symptom-journal/alerts?acknowledged=false"],
  });

  const trendsUrl = selectedBodyArea 
    ? `/api/v1/symptom-journal/trends?${new URLSearchParams({ body_area: selectedBodyArea, days: "30" })}`
    : null;
    
  const { data: trendsData } = useQuery({
    queryKey: [trendsUrl || "/api/v1/symptom-journal/trends/disabled"],
    enabled: !!selectedBodyArea && !!trendsUrl,
  });

  const comparisonUrl = selectedMeasurement1 && selectedMeasurement2 && selectedBodyArea
    ? `/api/v1/symptom-journal/compare?${new URLSearchParams({
        body_area: selectedBodyArea,
        measurement_id_1: selectedMeasurement1.toString(),
        measurement_id_2: selectedMeasurement2.toString(),
      })}`
    : null;

  const { data: comparisonData, isLoading: comparisonLoading } = useQuery({
    queryKey: [comparisonUrl || "/api/v1/symptom-journal/compare/disabled"],
    enabled: !!comparisonUrl,
  });

  // Audio examination mutations
  const createAudioSessionMutation = useMutation({
    mutationFn: async () => {
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
      setAudioSessionId(data.session_id);
      setCurrentAudioStage(data.current_stage);
      toast({
        title: "Session Created",
        description: "Audio examination session started successfully"
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

  const uploadAudioMutation = useMutation({
    mutationFn: async ({ stage, audioBlob }: { stage: AudioStage; audioBlob: Blob }) => {
      const base64Audio = await new Promise<string>((resolve) => {
        const reader = new FileReader();
        reader.onloadend = () => {
          const base64 = reader.result as string;
          const base64Data = base64.split(",")[1];
          resolve(base64Data);
        };
        reader.readAsDataURL(audioBlob);
      });

      const response = await apiRequest(`/api/v1/guided-audio-exam/sessions/${audioSessionId}/upload`, {
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
        setCurrentAudioStage(data.next_stage as AudioStage);
        toast({
          title: "Stage Complete",
          description: `${STAGE_INFO[variables.stage].name} recorded. Moving to ${STAGE_INFO[data.next_stage as AudioStage].name}.`
        });
      } else {
        toast({
          title: "All Stages Complete",
          description: "Completing examination and running AI analysis..."
        });
        completeAudioSessionMutation.mutate();
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

  const completeAudioSessionMutation = useMutation({
    mutationFn: async () => {
      const response = await apiRequest(`/api/v1/guided-audio-exam/sessions/${audioSessionId}/complete`, {
        method: "POST"
      });
      return await response.json();
    },
    onSuccess: () => {
      setExamComplete(true);
      queryClient.invalidateQueries({ queryKey: ["/api/v1/guided-audio-exam/sessions", audioSessionId, "results"] });
      toast({
        title: "Examination Complete",
        description: "AI analysis completed successfully. View your results in the Audio AI tab."
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

  // Fetch audio analysis results after completion
  const { data: audioAnalysisResults, isLoading: audioResultsLoading } = useQuery<AudioAnalysisResults>({
    queryKey: ["/api/v1/guided-audio-exam/sessions", audioSessionId, "results"],
    enabled: examComplete && !!audioSessionId,
    retry: 2
  });

  // Prep countdown timer effect
  useEffect(() => {
    if (showPrep && prepCountdown > 0) {
      const timer = setTimeout(() => {
        setPrepCountdown(prev => prev - 1);
      }, 1000);
      return () => clearTimeout(timer);
    } else if (showPrep && prepCountdown === 0) {
      setShowPrep(false);
      setPrepCountdown(30);
      startAudioRecording();
    }
  }, [showPrep, prepCountdown]);

  // Recording timer effect
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

  // Audio recording functions
  const startAudioRecording = async () => {
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
        if (currentAudioStage) {
          uploadAudioMutation.mutate({ stage: currentAudioStage, audioBlob });
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

  const stopAudioRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  };

  const handleAudioStageStart = () => {
    setShowPrep(true);
  };

  const handleStartAudioExam = () => {
    createAudioSessionMutation.mutate();
  };

  const handleResetAudioExam = () => {
    setAudioSessionId(null);
    setCurrentAudioStage(null);
    setIsRecording(false);
    setRecordingTime(0);
    setPrepCountdown(30);
    setShowPrep(false);
    setCompletedStages(new Set());
    setExamComplete(false);
  };

  const audioProgressPercent = (completedStages.size / 4) * 100;

  return (
    <div className="space-y-6">
      {showEmergency && (
        <EmergencyAlert
          symptoms={[
            "Severe chest pain or pressure",
            "Difficulty breathing",
            "High fever (103°F+) with confusion",
          ]}
          onDismiss={() => setShowEmergency(false)}
        />
      )}

      {/* Dynamic Welcome Screen */}
      <DynamicWelcome userName={user?.firstName} />

      {/* Legal Disclaimer */}
      <LegalDisclaimer variant="wellness" />

      <div>
        <h2 className="text-2xl font-semibold mb-2" data-testid="text-health-summary">Today's Health Summary</h2>
        <p className="text-muted-foreground" data-testid="text-summary-subtitle">Your daily metrics and activities</p>
      </div>

      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4" data-testid="grid-health-metrics">
        <HealthMetricCard
          title="Heart Rate"
          value="72"
          unit="bpm"
          icon={Heart}
          status="normal"
          trend="down"
          trendValue="3%"
          lastUpdated="2 mins ago"
          testId="card-metric-heart-rate"
        />
        <HealthMetricCard
          title="Steps Today"
          value="3,542"
          icon={Activity}
          status="normal"
          trend="up"
          trendValue="12%"
          lastUpdated="5 mins ago"
          testId="card-metric-steps"
        />
        <HealthMetricCard
          title="Water Intake"
          value="1.2"
          unit="L"
          icon={Droplet}
          status="warning"
          lastUpdated="1 hour ago"
          testId="card-metric-water"
        />
        <HealthMetricCard
          title="Sleep Quality"
          value="7.5"
          unit="hrs"
          icon={Moon}
          status="normal"
          trend="up"
          trendValue="0.5h"
          lastUpdated="Today"
          testId="card-metric-sleep"
        />
      </div>

      <div className="grid gap-6 lg:grid-cols-3">
        <div className="lg:col-span-2 space-y-6">
          <Card data-testid="card-todays-medications">
            <CardHeader className="flex flex-row items-center justify-between gap-2 space-y-0">
              <CardTitle data-testid="text-medications-title">Today's Medications</CardTitle>
              <Button variant="ghost" size="sm" data-testid="button-view-all-medications">
                View All
              </Button>
            </CardHeader>
            <CardContent className="grid gap-4 md:grid-cols-2">
              <MedicationCard
                name="Vitamin D3"
                dosage="2000 IU"
                frequency="Once daily"
                nextDose="2:00 PM"
                status="pending"
                isOTC={true}
                testId="card-medication-vitamin-d3"
              />
              <MedicationCard
                name="Prednisone"
                dosage="10mg"
                frequency="Twice daily"
                nextDose="Taken at 8:00 AM"
                status="taken"
                aiSuggestion="Consider reducing to 5mg based on recent improvements"
                testId="card-medication-prednisone"
              />
            </CardContent>
          </Card>

          <Card data-testid="card-daily-followup">
            <CardHeader>
              <CardTitle data-testid="text-followup-title">Daily Follow-up</CardTitle>
            </CardHeader>
            <CardContent>
              <Tabs defaultValue="device" className="space-y-4">
                <TabsList className="grid w-full grid-cols-4">
                  <TabsTrigger value="device" data-testid="tab-device">Device Data</TabsTrigger>
                  <TabsTrigger value="symptom-journal" data-testid="tab-symptom-journal">Symptoms</TabsTrigger>
                  <TabsTrigger value="video-ai" data-testid="tab-video-ai">Video AI</TabsTrigger>
                  <TabsTrigger value="audio-ai" data-testid="tab-audio-ai">Audio AI</TabsTrigger>
                </TabsList>
                <TabsContent value="device" className="space-y-3">
                  <div className="grid grid-cols-2 gap-3 text-sm">
                    <div data-testid="data-heart-rate">
                      <span className="text-muted-foreground">Heart Rate: </span>
                      <span className="font-medium" data-testid="value-heart-rate">{todayFollowup?.heartRate || "--"} bpm</span>
                    </div>
                    <div data-testid="data-spo2">
                      <span className="text-muted-foreground">SpO2: </span>
                      <span className="font-medium" data-testid="value-spo2">{todayFollowup?.oxygenSaturation || "--"}%</span>
                    </div>
                    <div data-testid="data-temperature">
                      <span className="text-muted-foreground">Temp: </span>
                      <span className="font-medium" data-testid="value-temperature">{todayFollowup?.temperature || "--"}°F</span>
                    </div>
                    <div data-testid="data-steps">
                      <span className="text-muted-foreground">Steps: </span>
                      <span className="font-medium" data-testid="value-steps">{todayFollowup?.stepsCount || "--"}</span>
                    </div>
                  </div>
                </TabsContent>
                <TabsContent value="symptom-journal" className="space-y-3">
                  {/* Symptom Journal - Compact View */}
                  <div className="space-y-3">
                    {/* Active Alerts Banner */}
                    {alertsData && alertsData.alerts.length > 0 && (
                      <Alert variant="destructive" className="text-xs">
                        <AlertCircle className="h-3 w-3" />
                        <AlertDescription>
                          {alertsData.alerts.length} active alert{alertsData.alerts.length > 1 ? 's' : ''} detected
                        </AlertDescription>
                      </Alert>
                    )}

                    {/* Recent Measurements */}
                    {loadingRecent ? (
                      <p className="text-xs text-muted-foreground">Loading measurements...</p>
                    ) : !recentData || recentData.measurements.length === 0 ? (
                      <div className="rounded-lg border bg-gradient-to-br from-primary/5 to-primary/10 p-3 space-y-2">
                        <p className="text-xs font-medium">Visual Symptom Tracking</p>
                        <p className="text-xs text-muted-foreground">
                          Track changes in legs, face, eyes, or chest over time with AI analysis.
                        </p>
                        <Button 
                          size="sm" 
                          className="w-full text-xs"
                          onClick={() => window.location.href = '/symptom-journal'}
                          data-testid="button-start-symptom-tracking"
                        >
                          <Camera className="mr-1 h-3 w-3" />
                          Start Tracking
                        </Button>
                      </div>
                    ) : (
                      <div className="space-y-2">
                        <div className="flex items-center justify-between">
                          <p className="text-xs font-medium">Latest Measurements</p>
                          <Button 
                            variant="outline"
                            size="sm"
                            className="h-6 text-xs"
                            onClick={() => window.location.href = '/symptom-journal'}
                            data-testid="button-view-all-symptoms"
                          >
                            View All
                          </Button>
                        </div>
                        
                        {recentData.measurements.slice(0, 2).map((measurement) => (
                          <div key={measurement.id} className="rounded-md border bg-muted/30 p-2 space-y-1">
                            <div className="flex items-center justify-between">
                              <Badge variant="secondary" className="text-xs">{measurement.body_area}</Badge>
                              <span className="text-xs text-muted-foreground">
                                {format(new Date(measurement.created_at), "MMM d, h:mm a")}
                              </span>
                            </div>
                            {measurement.ai_observations && (
                              <p className="text-xs line-clamp-2">{measurement.ai_observations}</p>
                            )}
                            <div className="flex gap-2 text-xs">
                              {measurement.color_change_percent !== null && (
                                <div className="flex items-center gap-0.5">
                                  <strong>Color:</strong>
                                  {measurement.color_change_percent > 0 ? (
                                    <TrendingUp className="h-2 w-2 text-orange-500" />
                                  ) : measurement.color_change_percent < 0 ? (
                                    <TrendingDown className="h-2 w-2 text-blue-500" />
                                  ) : (
                                    <Minus className="h-2 w-2" />
                                  )}
                                  <span>{Math.abs(measurement.color_change_percent).toFixed(1)}%</span>
                                </div>
                              )}
                              {measurement.respiratory_rate_bpm && (
                                <div>
                                  <strong>Breathing:</strong> {measurement.respiratory_rate_bpm} /min
                                </div>
                              )}
                            </div>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                </TabsContent>
                <TabsContent value="video-ai" className="space-y-3">
                  {isMetricsFromToday && latestVideoMetrics ? (
                    <>
                      {/* Today's Video AI Metrics */}
                      <div className="space-y-3">
                        <div className="flex items-center justify-between">
                          <p className="text-sm font-medium">Today's AI Analysis</p>
                          <Badge variant="secondary" className="text-xs">
                            {new Date(latestVideoMetrics.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                          </Badge>
                        </div>
                        
                        <div className="grid grid-cols-2 gap-2 text-xs">
                          {latestVideoMetrics.respiratory_rate_bpm && (
                            <div className="flex items-center justify-between p-2 rounded-md bg-muted/50">
                              <div className="flex items-center gap-1.5">
                                <Wind className="h-3 w-3 text-blue-500" />
                                <span className="text-muted-foreground">Respiratory</span>
                              </div>
                              <span className="font-medium">{latestVideoMetrics.respiratory_rate_bpm.toFixed(1)} bpm</span>
                            </div>
                          )}
                          
                          {latestVideoMetrics.skin_pallor_score !== null && latestVideoMetrics.skin_pallor_score !== undefined && (
                            <div className="flex items-center justify-between p-2 rounded-md bg-muted/50">
                              <div className="flex items-center gap-1.5">
                                <Palette className="h-3 w-3 text-amber-500" />
                                <span className="text-muted-foreground">Skin Pallor</span>
                              </div>
                              <span className="font-medium">{latestVideoMetrics.skin_pallor_score.toFixed(1)}/100</span>
                            </div>
                          )}
                          
                          {latestVideoMetrics.jaundice_risk_level && (
                            <div className="flex items-center justify-between p-2 rounded-md bg-muted/50">
                              <div className="flex items-center gap-1.5">
                                <Eye className="h-3 w-3 text-yellow-500" />
                                <span className="text-muted-foreground">Jaundice</span>
                              </div>
                              <span className="font-medium capitalize">{latestVideoMetrics.jaundice_risk_level}</span>
                            </div>
                          )}
                          
                          {latestVideoMetrics.facial_swelling_score !== null && latestVideoMetrics.facial_swelling_score !== undefined && (
                            <div className="flex items-center justify-between p-2 rounded-md bg-muted/50">
                              <div className="flex items-center gap-1.5">
                                <Users className="h-3 w-3 text-rose-500" />
                                <span className="text-muted-foreground">Swelling</span>
                              </div>
                              <span className="font-medium">{latestVideoMetrics.facial_swelling_score.toFixed(1)}/100</span>
                            </div>
                          )}
                          
                          {((latestVideoMetrics.tremor_detected !== null && latestVideoMetrics.tremor_detected !== undefined) || (latestVideoMetrics.head_stability_score !== null && latestVideoMetrics.head_stability_score !== undefined)) && (
                            <div className="flex items-center justify-between p-2 rounded-md bg-muted/50">
                              <div className="flex items-center gap-1.5">
                                <Zap className="h-3 w-3 text-purple-500" />
                                <span className="text-muted-foreground">Tremor</span>
                              </div>
                              <span className="font-medium">
                                {latestVideoMetrics.tremor_detected !== null && latestVideoMetrics.tremor_detected !== undefined 
                                  ? (latestVideoMetrics.tremor_detected ? 'Detected' : 'None')
                                  : latestVideoMetrics.head_stability_score !== null && latestVideoMetrics.head_stability_score !== undefined
                                  ? `Score: ${latestVideoMetrics.head_stability_score.toFixed(1)}`
                                  : 'N/A'
                                }
                              </span>
                            </div>
                          )}
                          
                          {latestVideoMetrics.tongue_color_index !== null && latestVideoMetrics.tongue_color_index !== undefined && (
                            <div className="flex items-center justify-between p-2 rounded-md bg-muted/50">
                              <div className="flex items-center gap-1.5">
                                <Smile className="h-3 w-3 text-pink-500" />
                                <span className="text-muted-foreground">Tongue</span>
                              </div>
                              <span className="font-medium">{latestVideoMetrics.tongue_coating_detected ? 'Coating' : 'Normal'}</span>
                            </div>
                          )}
                        </div>
                        
                        {/* Quality Metrics */}
                        {(latestVideoMetrics.lighting_quality_score || latestVideoMetrics.frames_analyzed || latestVideoMetrics.face_detection_confidence) && (
                          <div className="rounded-md border bg-muted/30 p-2 space-y-1">
                            <p className="text-xs font-medium">Analysis Quality</p>
                            <div className="grid grid-cols-3 gap-2 text-xs">
                              {latestVideoMetrics.lighting_quality_score && (
                                <div>
                                  <span className="text-muted-foreground">Lighting: </span>
                                  <span className="font-medium">{latestVideoMetrics.lighting_quality_score.toFixed(0)}/100</span>
                                </div>
                              )}
                              {latestVideoMetrics.frames_analyzed && (
                                <div>
                                  <span className="text-muted-foreground">Frames: </span>
                                  <span className="font-medium">{latestVideoMetrics.frames_analyzed}</span>
                                </div>
                              )}
                              {latestVideoMetrics.face_detection_confidence && (
                                <div>
                                  <span className="text-muted-foreground">Confidence: </span>
                                  <span className="font-medium">{(latestVideoMetrics.face_detection_confidence * 100).toFixed(0)}%</span>
                                </div>
                              )}
                            </div>
                          </div>
                        )}
                        
                        <Link href="/guided-video-exam">
                          <Button size="sm" variant="outline" className="w-full gap-2" data-testid="button-new-exam">
                            <Play className="h-3 w-3" />
                            Take Another Exam
                          </Button>
                        </Link>
                      </div>
                    </>
                  ) : (
                    <>
                      {/* No exam taken today - Prompt to take exam */}
                      <div className="rounded-lg border bg-gradient-to-br from-primary/5 to-primary/10 p-4 space-y-3">
                        <div className="flex items-center gap-2">
                          <div className="rounded-full bg-primary/10 p-2">
                            <Video className="h-4 w-4 text-primary" />
                          </div>
                          <div>
                            <p className="font-medium text-sm">No Examination Today</p>
                            <p className="text-xs text-muted-foreground">Complete your daily video exam</p>
                          </div>
                        </div>

                        <Link href="/guided-video-exam">
                          <Button size="sm" className="w-full gap-2" data-testid="button-start-video-exam">
                            <Play className="h-4 w-4" />
                            Start Video Examination
                          </Button>
                        </Link>
                        
                        <p className="text-xs text-center text-muted-foreground">
                          7-stage AI analysis • ~5-7 minutes • Camera required
                        </p>
                      </div>
                      
                      <div className="text-xs text-muted-foreground" data-testid="text-video-info">
                        <p className="font-medium mb-1">AI tracks:</p>
                        <ul className="space-y-0.5 ml-4 list-disc">
                          <li>Respiratory rate & breathing</li>
                          <li>Skin pallor (anemia detection)</li>
                          <li>Eye sclera yellowness (jaundice)</li>
                          <li>Facial swelling & edema</li>
                          <li>Head/hand tremor patterns</li>
                          <li>Tongue color & coating</li>
                        </ul>
                      </div>
                    </>
                  )}
                </TabsContent>
                <TabsContent value="audio-ai" className="space-y-4">
                  {/* Completed Examination Results */}
                  {examComplete && audioAnalysisResults ? (
                    <div className="space-y-4">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <CheckCircle2 className="h-5 w-5 text-primary" />
                          <p className="font-medium text-sm">Examination Complete</p>
                        </div>
                        <Button size="sm" variant="outline" onClick={handleResetAudioExam} data-testid="button-new-audio-exam">
                          New Exam
                        </Button>
                      </div>

                      <div className="grid grid-cols-2 gap-3 text-sm">
                        <div className="bg-muted/50 p-3 rounded-md">
                          <div className="text-xs text-muted-foreground">Speech Fluency</div>
                          <div className="text-lg font-bold">{audioAnalysisResults.speech_fluency_score?.toFixed(1) || "N/A"}/100</div>
                        </div>
                        <div className="bg-muted/50 p-3 rounded-md">
                          <div className="text-xs text-muted-foreground">Voice Weakness</div>
                          <div className="text-lg font-bold">{audioAnalysisResults.voice_weakness_index?.toFixed(1) || "N/A"}/100</div>
                        </div>
                        <div className="bg-muted/50 p-3 rounded-md">
                          <div className="text-xs text-muted-foreground">Breath Rate</div>
                          <div className="text-lg font-bold">{audioAnalysisResults.breath_rate_per_minute?.toFixed(0) || "N/A"} /min</div>
                        </div>
                        <div className="bg-muted/50 p-3 rounded-md">
                          <div className="text-xs text-muted-foreground">Cough Count</div>
                          <div className="text-lg font-bold">{audioAnalysisResults.cough_count || 0}</div>
                        </div>
                      </div>

                      <div className="rounded-lg border bg-muted/30 p-3 space-y-2">
                        <p className="text-xs font-medium">ML Analysis</p>
                        <div className="space-y-1 text-xs">
                          <div className="flex items-center justify-between">
                            <span className="text-muted-foreground">Cough Probability:</span>
                            <span className="font-semibold">{(audioAnalysisResults.cough_probability_ml * 100).toFixed(1)}%</span>
                          </div>
                          <div className="flex items-center justify-between">
                            <span className="text-muted-foreground">Speech Probability:</span>
                            <span className="font-semibold">{(audioAnalysisResults.speech_probability_ml * 100).toFixed(1)}%</span>
                          </div>
                          <div className="flex items-center justify-between">
                            <span className="text-muted-foreground">Wheeze Detected:</span>
                            <Badge variant={audioAnalysisResults.wheeze_detected ? "destructive" : "secondary"} className="text-xs">
                              {audioAnalysisResults.wheeze_detected ? "Yes" : "No"}
                            </Badge>
                          </div>
                        </div>
                      </div>

                      {audioAnalysisResults.recommendations && audioAnalysisResults.recommendations.length > 0 && (
                        <div className="rounded-lg border bg-primary/5 p-3 space-y-2">
                          <p className="text-xs font-medium">Recommendations:</p>
                          <ul className="text-xs space-y-1">
                            {audioAnalysisResults.recommendations.map((rec: string, idx: number) => (
                              <li key={idx} className="flex items-start gap-2">
                                <CheckCircle2 className="h-3 w-3 mt-0.5 flex-shrink-0 text-primary" />
                                <span>{rec}</span>
                              </li>
                            ))}
                          </ul>
                        </div>
                      )}
                    </div>
                  ) : audioSessionId && !examComplete ? (
                    /* Active Examination Workflow */
                    <div className="space-y-4">
                      {/* Progress Bar */}
                      <div className="space-y-2">
                        <div className="flex items-center justify-between text-xs">
                          <span className="font-medium">Progress</span>
                          <span className="text-muted-foreground">{completedStages.size} / 4 stages</span>
                        </div>
                        <Progress value={audioProgressPercent} />
                      </div>

                      {/* Stage Badges */}
                      <div className="grid grid-cols-4 gap-1">
                        {(["breathing", "coughing", "speaking", "reading"] as AudioStage[]).map((stage) => (
                          <Badge 
                            key={stage}
                            variant={completedStages.has(stage) ? "default" : "outline"}
                            className="justify-center text-xs py-1"
                          >
                            {completedStages.has(stage) && <CheckCircle2 className="w-2 h-2 mr-1" />}
                            {STAGE_INFO[stage].name}
                          </Badge>
                        ))}
                      </div>

                      {/* Current Stage Instructions */}
                      {currentAudioStage && !showPrep && !isRecording && (
                        <div className="rounded-lg border bg-primary/5 p-3 space-y-3">
                          <div className="flex items-center gap-2">
                            {STAGE_INFO[currentAudioStage].icon}
                            <p className="font-medium text-sm">{STAGE_INFO[currentAudioStage].name} Stage</p>
                          </div>
                          <div className="bg-background p-3 rounded text-xs whitespace-pre-line">
                            {STAGE_INFO[currentAudioStage].instructions}
                          </div>
                          <Button 
                            onClick={handleAudioStageStart}
                            size="sm"
                            className="w-full"
                            data-testid={`button-start-${currentAudioStage}`}
                          >
                            Begin {STAGE_INFO[currentAudioStage].name}
                          </Button>
                        </div>
                      )}

                      {/* Prep Countdown */}
                      {showPrep && (
                        <div className="rounded-lg bg-primary text-primary-foreground p-6 text-center space-y-3">
                          <div className="text-4xl font-bold">{prepCountdown}</div>
                          <p className="text-sm">Get ready to record...</p>
                          <p className="text-xs opacity-90">
                            {currentAudioStage && STAGE_INFO[currentAudioStage].description}
                          </p>
                        </div>
                      )}

                      {/* Recording State */}
                      {isRecording && currentAudioStage && (
                        <div className="rounded-lg border border-destructive/30 bg-destructive/10 p-4 space-y-4">
                          <div className="flex justify-center">
                            <div className="w-16 h-16 bg-destructive rounded-full flex items-center justify-center animate-pulse">
                              <Mic className="w-8 h-8 text-destructive-foreground" />
                            </div>
                          </div>
                          <div className="text-3xl font-bold text-center">{recordingTime}s</div>
                          <p className="text-center font-semibold">Recording {STAGE_INFO[currentAudioStage].name}...</p>
                          <Button 
                            onClick={stopAudioRecording}
                            variant="destructive"
                            size="sm"
                            className="w-full"
                            disabled={uploadAudioMutation.isPending}
                            data-testid="button-stop-recording"
                          >
                            {uploadAudioMutation.isPending ? "Uploading..." : "Stop & Continue"}
                          </Button>
                        </div>
                      )}
                    </div>
                  ) : (
                    /* Not Started State */
                    <div className="space-y-3">
                      <div className="rounded-lg border bg-gradient-to-br from-primary/5 to-primary/10 p-4 space-y-3">
                        <div className="flex items-center gap-2">
                          <div className="rounded-full bg-primary/10 p-2">
                            <Mic className="h-4 w-4 text-primary" />
                          </div>
                          <div>
                            <p className="font-medium text-sm">AI Audio Examination</p>
                            <p className="text-xs text-muted-foreground">4-stage guided examination</p>
                          </div>
                        </div>

                        <div className="bg-background/50 p-3 rounded space-y-2">
                          <p className="text-xs font-medium">What to Expect:</p>
                          <ul className="text-xs space-y-1 text-muted-foreground">
                            <li className="flex items-start gap-2">
                              <CheckCircle2 className="w-3 h-3 mt-0.5 flex-shrink-0" />
                              <span>Breathing, Coughing, Speaking, Reading</span>
                            </li>
                            <li className="flex items-start gap-2">
                              <CheckCircle2 className="w-3 h-3 mt-0.5 flex-shrink-0" />
                              <span>30-second prep before each stage</span>
                            </li>
                            <li className="flex items-start gap-2">
                              <CheckCircle2 className="w-3 h-3 mt-0.5 flex-shrink-0" />
                              <span>YAMNet ML + neurological metrics</span>
                            </li>
                            <li className="flex items-start gap-2">
                              <CheckCircle2 className="w-3 h-3 mt-0.5 flex-shrink-0" />
                              <span>~2-3 minutes total</span>
                            </li>
                          </ul>
                        </div>
                        
                        <Button 
                          onClick={handleStartAudioExam}
                          disabled={createAudioSessionMutation.isPending || !user}
                          size="sm"
                          className="w-full gap-2"
                          data-testid="button-start-audio-exam"
                        >
                          {createAudioSessionMutation.isPending ? "Creating Session..." : (
                            <>
                              <Play className="h-3 w-3" />
                              Start Audio Examination
                            </>
                          )}
                        </Button>
                        
                        <p className="text-xs text-center text-muted-foreground">
                          Microphone access required • Find a quiet space
                        </p>
                      </div>
                      
                      <div className="text-xs text-muted-foreground">
                        <p className="font-medium mb-1">AI tracks:</p>
                        <ul className="space-y-0.5 ml-4 list-disc">
                          <li>Breath cycles & respiratory rate</li>
                          <li>Speech pace & fluency</li>
                          <li>Cough & wheeze detection</li>
                          <li>Voice hoarseness & fatigue</li>
                          <li>Pause patterns & neurological markers</li>
                        </ul>
                      </div>
                    </div>
                  )}
                </TabsContent>
              </Tabs>
            </CardContent>
          </Card>

          <Card data-testid="card-dynamic-tasks">
            <CardHeader className="flex flex-row items-center justify-between gap-2 space-y-0">
              <CardTitle data-testid="text-tasks-title">Dynamic Tasks</CardTitle>
              <Badge variant="secondary" data-testid="badge-tasks-pending">{tasks?.filter(t => !t.completed).length || 0} pending</Badge>
            </CardHeader>
            <CardContent>
              {tasks && tasks.length > 0 ? (
                <div className="space-y-2">
                  {tasks.slice(0, 5).map((task) => (
                    <div key={task.id} className="flex items-center gap-2 p-2 rounded-md hover-elevate" data-testid={`task-item-${task.id}`}>
                      {task.completed ? (
                        <CheckCircle className="h-4 w-4 text-chart-2 flex-shrink-0" data-testid={`icon-task-completed-${task.id}`} />
                      ) : (
                        <div className="h-4 w-4 rounded-full border-2 flex-shrink-0" data-testid={`icon-task-pending-${task.id}`} />
                      )}
                      <div className="flex-1 min-w-0">
                        <p className="text-sm font-medium truncate" data-testid={`text-task-title-${task.id}`}>{task.title}</p>
                        {task.description && (
                          <p className="text-xs text-muted-foreground truncate" data-testid={`text-task-description-${task.id}`}>{task.description}</p>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-sm text-muted-foreground" data-testid="text-no-tasks">No tasks for today</p>
              )}
            </CardContent>
          </Card>
        </div>

        <div className="space-y-6">
          <Card data-testid="card-reminders">
            <CardHeader>
              <CardTitle className="flex items-center gap-2" data-testid="text-reminders-title">
                <Calendar className="h-5 w-5" />
                Reminders
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <ReminderCard
                type="water"
                title="Drink Water"
                time="2:00 PM"
                description="You've had 4 glasses today. Goal: 8 glasses"
                testId="reminder-water"
              />
              <ReminderCard
                type="exercise"
                title="Gentle Stretching"
                time="3:00 PM"
                description="15-minute low-impact session"
                testId="reminder-exercise"
              />
            </CardContent>
          </Card>

          <Card className="bg-gradient-to-br from-primary/5 to-primary/10" data-testid="card-behavioral-insights">
            <CardHeader>
              <CardTitle className="text-base flex items-center gap-2" data-testid="text-insights-title">
                <Brain className="h-5 w-5" />
                Behavioral AI Insight
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              {insights && insights.length > 0 ? (
                insights.slice(0, 2).map((insight, idx) => (
                  <div key={idx} className="flex items-start gap-2" data-testid={`insight-item-${idx}`}>
                    <TrendingUp className="h-4 w-4 text-chart-2 mt-0.5 flex-shrink-0" />
                    <div className="text-sm">
                      <p className="font-medium mb-1" data-testid={`text-stress-level-${idx}`}>Stress Level: {insight.stressScore}/10</p>
                      <p className="text-muted-foreground" data-testid={`text-activity-level-${idx}`}>Activity: {insight.activityLevel}</p>
                    </div>
                  </div>
                ))
              ) : (
                <p className="text-sm text-muted-foreground" data-testid="text-insight-placeholder">
                  Complete your daily check-ins to unlock AI-powered behavioral insights and health trend detection.
                </p>
              )}
              <Link href="/behavioral-ai-insights">
                <Button variant="outline" size="sm" className="w-full" data-testid="button-view-insights">
                  View Full Insights
                </Button>
              </Link>
            </CardContent>
          </Card>

          <Card data-testid="card-medication-adherence">
            <CardHeader>
              <CardTitle className="text-base" data-testid="text-adherence-title">Medication Adherence</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                <div className="flex items-center justify-between text-sm">
                  <span className="text-muted-foreground">This Week</span>
                  <span className="font-semibold text-chart-2" data-testid="value-adherence-percentage">92%</span>
                </div>
                <div className="w-full bg-muted rounded-full h-2" data-testid="progress-adherence">
                  <div className="bg-chart-2 h-2 rounded-full" style={{ width: "92%" }} data-testid="progress-adherence-fill" />
                </div>
                <p className="text-xs text-muted-foreground" data-testid="text-active-medications">
                  {medications?.filter(m => m.active).length || 0} active medications
                </p>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
