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
import { Heart, Activity, Droplet, Moon, TrendingUp, Calendar, CheckCircle, Brain, Video, Eye, Hand, Smile, Play, Wind, Palette, Zap, Users, Mic, Volume2, MicOff, Pause, CheckCircle2, AlertCircle, Camera, TrendingDown, Minus, User, Info, GitCompare, ClipboardList, AlertTriangle, Check, FileText, Phone, Loader2 } from "lucide-react";
import { useState, useRef, useEffect } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { useAuth } from "@/hooks/useAuth";
import { Link } from "wouter";
import type { DailyFollowup, Medication, DynamicTask, BehavioralInsight, PaintrackSession } from "@shared/schema";
import { useToast } from "@/hooks/use-toast";
import { apiRequest, queryClient } from "@/lib/queryClient";
import { Progress } from "@/components/ui/progress";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
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

// Mental Health types
interface QuestionnaireTemplate {
  type: string;
  full_name: string;
  description: string;
  public_domain: boolean;
  timeframe: string;
  instructions: string;
  response_options: Array<{
    value: number;
    label: string;
  }>;
  questions: Array<{
    id: string;
    text: string;
    cluster: string;
    crisis_flag?: boolean;
    reverse_scored?: boolean;
  }>;
  scoring: {
    max_score: number;
    severity_levels: Array<{
      range: number[];
      level: string;
      description: string;
    }>;
  };
}

interface QuestionnaireResponse {
  response_id: string;
  questionnaire_type: string;
  score: {
    total_score: number;
    max_score: number;
    severity_level: string;
    severity_description: string;
    cluster_scores: Record<string, any>;
    neutral_summary: string;
    key_observations: string[];
  };
  crisis_intervention?: {
    crisis_detected: boolean;
    crisis_severity: string;
    intervention_message: string;
    crisis_hotlines: Array<{
      name: string;
      phone?: string;
      sms?: string;
      description: string;
      website: string;
    }>;
    next_steps: string[];
  };
  analysis_id?: string;
}

interface MentalHealthHistoryItem {
  response_id: string;
  questionnaire_type: string;
  completed_at: string;
  total_score: number;
  max_score: number;
  severity_level: string;
  crisis_detected: boolean;
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

// Daily Wellness Check questions (for TODAY)
const DAILY_WELLNESS_QUESTIONS = [
  {
    id: "mood",
    text: "How would you describe your mood today?",
    options: [
      { value: 0, label: "Very low" },
      { value: 1, label: "Low" },
      { value: 2, label: "Neutral" },
      { value: 3, label: "Good" },
      { value: 4, label: "Very good" },
    ]
  },
  {
    id: "anxiety",
    text: "How anxious have you felt today?",
    options: [
      { value: 0, label: "Not at all" },
      { value: 1, label: "Slightly" },
      { value: 2, label: "Moderately" },
      { value: 3, label: "Very" },
      { value: 4, label: "Extremely" },
    ]
  },
  {
    id: "stress",
    text: "How stressed have you felt today?",
    options: [
      { value: 0, label: "Not at all" },
      { value: 1, label: "Slightly" },
      { value: 2, label: "Moderately" },
      { value: 3, label: "Very" },
      { value: 4, label: "Extremely" },
    ]
  },
  {
    id: "energy",
    text: "How would you rate your energy level today?",
    options: [
      { value: 0, label: "Very low" },
      { value: 1, label: "Low" },
      { value: 2, label: "Moderate" },
      { value: 3, label: "High" },
      { value: 4, label: "Very high" },
    ]
  },
  {
    id: "sleep",
    text: "How did you sleep last night?",
    options: [
      { value: 0, label: "Very poor" },
      { value: 1, label: "Poor" },
      { value: 2, label: "Fair" },
      { value: 3, label: "Good" },
      { value: 4, label: "Excellent" },
    ]
  },
];

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

  // Mental Health state
  const [selectedQuestionnaire, setSelectedQuestionnaire] = useState<QuestionnaireTemplate | null>(null);
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0);
  const [mentalHealthResponses, setMentalHealthResponses] = useState<Record<string, { response: number; response_text: string }>>({});
  const [isSubmittingMentalHealth, setIsSubmittingMentalHealth] = useState(false);
  const [completedMentalHealthResponse, setCompletedMentalHealthResponse] = useState<QuestionnaireResponse | null>(null);
  const [mentalHealthStartTime, setMentalHealthStartTime] = useState<number | null>(null);

  // Daily Wellness Check state
  const [showDailyWellness, setShowDailyWellness] = useState(false);
  const [dailyWellnessResponses, setDailyWellnessResponses] = useState<Record<string, number>>({});
  const [completedDailyWellness, setCompletedDailyWellness] = useState(false);

  // PainTrack state
  const [paintrackStep, setPaintrackStep] = useState<'select-module' | 'select-joint' | 'instructions' | 'recording' | 'pain-report' | 'complete'>('select-module');
  const [selectedModule, setSelectedModule] = useState<string | null>(null);
  const [selectedJoint, setSelectedJoint] = useState<string | null>(null);
  const [selectedLaterality, setSelectedLaterality] = useState<'left' | 'right' | 'bilateral' | null>(null);
  const [painVAS, setPainVAS] = useState<number>(5);
  const [painNotes, setPainNotes] = useState<string>("");
  const [medicationTaken, setMedicationTaken] = useState<boolean>(false);
  const [medicationDetails, setMedicationDetails] = useState<string>("");
  
  // Dual-camera capture state
  const [isRecordingPain, setIsRecordingPain] = useState<boolean>(false);
  const [painRecordingTime, setPainRecordingTime] = useState<number>(0);
  const [frontCameraStream, setFrontCameraStream] = useState<MediaStream | null>(null);
  const [backCameraStream, setBackCameraStream] = useState<MediaStream | null>(null);
  const [frontVideoBlob, setFrontVideoBlob] = useState<Blob | null>(null);
  const [backVideoBlob, setBackVideoBlob] = useState<Blob | null>(null);
  const [dualCameraSupported, setDualCameraSupported] = useState<boolean>(false);
  const frontVideoRef = useRef<HTMLVideoElement>(null);
  const backVideoRef = useRef<HTMLVideoElement>(null);
  const painTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const frontMediaRecorderRef = useRef<MediaRecorder | null>(null);
  const backMediaRecorderRef = useRef<MediaRecorder | null>(null);
  const frontChunksRef = useRef<Blob[]>([]);
  const backChunksRef = useRef<Blob[]>([]);

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

  // PainTrack mutations
  const createPaintrackSessionMutation = useMutation({
    mutationFn: async (sessionData: {
      module: string;
      joint: string;
      laterality: string | null;
      patientVas: number;
      patientNotes: string;
      medicationTaken: boolean;
      medicationDetails: string;
    }) => {
      return await apiRequest("/api/paintrack/sessions", {
        method: "POST",
        body: JSON.stringify(sessionData),
        headers: {
          "Content-Type": "application/json",
        },
      });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/paintrack/sessions"] });
      toast({
        title: "Session Saved!",
        description: `Pain level ${painVAS}/10 recorded for ${selectedJoint}`,
      });
    },
    onError: (error: Error) => {
      toast({
        variant: "destructive",
        title: "Failed to Save Session",
        description: error.message
      });
    }
  });

  // PainTrack sessions query
  const { data: paintrackSessions, isLoading: paintrackSessionsLoading } = useQuery<PaintrackSession[]>({
    queryKey: ["/api/paintrack/sessions"],
    enabled: !!user,
  });

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

  // Mental Health queries
  const { data: questionnairesData, isLoading: isLoadingQuestionnaires, isError: isErrorQuestionnaires, error: questionnaireError } = useQuery({
    queryKey: ["/api/v1/mental-health/questionnaires"],
  });

  const { data: mentalHealthHistoryData, refetch: refetchMentalHealthHistory } = useQuery({
    queryKey: ["/api/v1/mental-health/history"],
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

  // PainTrack dual-camera recording functions
  const startDualCameraRecording = async () => {
    try {
      // Reset dual-camera status at start of each session
      setDualCameraSupported(false);
      
      // Check MediaRecorder API availability
      if (typeof MediaRecorder === 'undefined') {
        throw new Error('Video recording is not supported on this browser. Please use Chrome, Firefox, or Safari 14.1+');
      }
      
      // Determine best MIME type with fallback chain
      let mimeType: string;
      const isIOS = /iPhone|iPad|iPod/.test(navigator.userAgent);
      
      // Test MIME types in order of preference
      if (MediaRecorder.isTypeSupported('video/mp4;codecs=h264')) {
        mimeType = 'video/mp4;codecs=h264';
      } else if (MediaRecorder.isTypeSupported('video/webm;codecs=vp9')) {
        mimeType = 'video/webm;codecs=vp9';
      } else if (MediaRecorder.isTypeSupported('video/webm;codecs=vp8')) {
        mimeType = 'video/webm;codecs=vp8';
      } else if (MediaRecorder.isTypeSupported('video/webm')) {
        mimeType = 'video/webm';
      } else if (MediaRecorder.isTypeSupported('video/mp4')) {
        mimeType = 'video/mp4';
      } else {
        throw new Error('MediaRecorder not supported on this device. Please use a modern browser.');
      }
      
      console.log(`[PainTrack] Selected MIME type: ${mimeType}, iOS: ${isIOS}`);

      // Front camera (user-facing)
      const frontConstraints: MediaStreamConstraints = {
        video: { facingMode: 'user', width: 1280, height: 720 },
        audio: false
      };

      // Back camera (environment-facing for joint)
      const backConstraints: MediaStreamConstraints = {
        video: { facingMode: 'environment', width: 1280, height: 720 },
        audio: false
      };

      // Attempt to get both streams
      const frontStream = await navigator.mediaDevices.getUserMedia(frontConstraints);
      setFrontCameraStream(frontStream);
      
      if (frontVideoRef.current) {
        frontVideoRef.current.srcObject = frontStream;
      }

      // Try to get back camera (may not be supported on all devices)
      let backStream: MediaStream | null = null;
      try {
        backStream = await navigator.mediaDevices.getUserMedia(backConstraints);
        setBackCameraStream(backStream);
        
        if (backVideoRef.current) {
          backVideoRef.current.srcObject = backStream;
        }
      } catch (backCameraError) {
        console.warn("Back camera not available, using front camera only:", backCameraError);
        toast({
          title: "Single Camera Mode",
          description: "Using front camera only. Back camera not available on this device.",
        });
      }

      // Start recording on both streams using refs for stability
      frontChunksRef.current = [];
      const frontRecorder = new MediaRecorder(frontStream, { mimeType });
      
      frontRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          frontChunksRef.current.push(event.data);
        }
      };

      frontRecorder.onstop = () => {
        const blob = new Blob(frontChunksRef.current, { type: mimeType });
        setFrontVideoBlob(blob);
        frontChunksRef.current = [];
      };

      frontRecorder.onerror = (event) => {
        console.error('[PainTrack] Front camera recording error:', event);
        toast({
          variant: "destructive",
          title: "Recording Error",
          description: "Front camera recording failed. Please try again.",
        });
      };

      frontRecorder.start();
      frontMediaRecorderRef.current = frontRecorder;

      // Record back camera if available (use local backStream variable, not state)
      if (backStream) {
        backChunksRef.current = [];
        const backRecorder = new MediaRecorder(backStream, { mimeType });
        
        backRecorder.ondataavailable = (event) => {
          if (event.data.size > 0) {
            backChunksRef.current.push(event.data);
          }
        };

        backRecorder.onstop = () => {
          const blob = new Blob(backChunksRef.current, { type: mimeType });
          // Only set blob if we have chunks (confirms successful recording)
          if (backChunksRef.current.length > 0) {
            setBackVideoBlob(blob);
          } else {
            console.warn('[PainTrack] Back camera stopped with no data');
            setDualCameraSupported(false);
          }
          backChunksRef.current = [];
        };

        backRecorder.onerror = (event) => {
          console.error('[PainTrack] Back camera recording error:', event);
          setDualCameraSupported(false);
          toast({
            title: "Single Camera Mode",
            description: "Back camera failed. Continuing with front camera only.",
          });
        };

        try {
          backRecorder.start();
          backMediaRecorderRef.current = backRecorder;
          
          // Set dualCameraSupported only after back camera successfully started
          setDualCameraSupported(true);
          console.log('[PainTrack] Dual-camera recording started successfully');
        } catch (startError) {
          console.error('[PainTrack] Failed to start back camera recorder:', startError);
          setDualCameraSupported(false);
        }
      } else {
        // Back camera not available
        setDualCameraSupported(false);
        console.log('[PainTrack] Single-camera mode (back camera unavailable)');
      }

      setIsRecordingPain(true);
      setPainRecordingTime(0);

      // Auto-stop after 20 seconds
      setTimeout(() => {
        stopDualCameraRecording();
      }, 20000);

    } catch (error) {
      console.error("Camera access error:", error);
      toast({
        variant: "destructive",
        title: "Camera Access Denied",
        description: "Please allow camera access to record your movement.",
      });
    }
  };

  const stopDualCameraRecording = () => {
    // Use refs directly to ensure we have stable references
    const frontRec = frontMediaRecorderRef.current;
    const backRec = backMediaRecorderRef.current;
    
    if (frontRec && frontRec.state !== 'inactive') {
      frontRec.stop();
    }
    if (backRec && backRec.state !== 'inactive') {
      backRec.stop();
    }

    // Stop all tracks
    frontCameraStream?.getTracks().forEach(track => track.stop());
    backCameraStream?.getTracks().forEach(track => track.stop());

    setIsRecordingPain(false);
    
    if (painTimerRef.current) {
      clearInterval(painTimerRef.current);
      painTimerRef.current = null;
    }

    // Wait for blob creation before moving to next step
    // DO NOT clear refs here - they're needed for blob creation in onstop handlers
    setTimeout(() => {
      setPaintrackStep('pain-report');
    }, 1000); // Increased to 1s to ensure blobs are ready
  };

  // Timer effect for pain recording
  useEffect(() => {
    if (isRecordingPain) {
      painTimerRef.current = setInterval(() => {
        setPainRecordingTime(prev => prev + 1);
      }, 1000);
    } else if (painTimerRef.current) {
      clearInterval(painTimerRef.current);
      painTimerRef.current = null;
    }
    return () => {
      if (painTimerRef.current) clearInterval(painTimerRef.current);
    };
  }, [isRecordingPain]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      frontCameraStream?.getTracks().forEach(track => track.stop());
      backCameraStream?.getTracks().forEach(track => track.stop());
    };
  }, [frontCameraStream, backCameraStream]);

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

  // Mental Health handlers
  const handleStartQuestionnaire = (template: QuestionnaireTemplate) => {
    setSelectedQuestionnaire(template);
    setCurrentQuestionIndex(0);
    setMentalHealthResponses({});
    setCompletedMentalHealthResponse(null);
    setMentalHealthStartTime(Date.now());
  };

  const handleMentalHealthResponseSelect = (questionId: string, value: number, label: string) => {
    setMentalHealthResponses(prev => ({
      ...prev,
      [questionId]: { response: value, response_text: label }
    }));
  };

  const handleMentalHealthNext = () => {
    if (selectedQuestionnaire && currentQuestionIndex < selectedQuestionnaire.questions.length - 1) {
      setCurrentQuestionIndex(currentQuestionIndex + 1);
    }
  };

  const handleMentalHealthPrevious = () => {
    if (currentQuestionIndex > 0) {
      setCurrentQuestionIndex(currentQuestionIndex - 1);
    }
  };

  const handleSubmitMentalHealth = async () => {
    if (!selectedQuestionnaire) return;

    const allQuestionsAnswered = selectedQuestionnaire.questions.every(q => mentalHealthResponses[q.id]);
    if (!allQuestionsAnswered) {
      toast({
        title: "Incomplete Questionnaire",
        description: "Please answer all questions before submitting.",
        variant: "destructive",
      });
      return;
    }

    setIsSubmittingMentalHealth(true);

    try {
      const durationSeconds = mentalHealthStartTime ? Math.round((Date.now() - mentalHealthStartTime) / 1000) : null;

      const formattedResponses = selectedQuestionnaire.questions.map(q => ({
        question_id: q.id,
        question_text: q.text,
        response: mentalHealthResponses[q.id].response,
        response_text: mentalHealthResponses[q.id].response_text,
      }));

      const result = await apiRequest<QuestionnaireResponse>("/api/v1/mental-health/submit", {
        method: "POST",
        body: JSON.stringify({
          questionnaire_type: selectedQuestionnaire.type,
          responses: formattedResponses,
          duration_seconds: durationSeconds,
          allow_storage: true,
          allow_clinical_sharing: true,
        }),
        headers: {
          "Content-Type": "application/json",
        },
      });

      setCompletedMentalHealthResponse(result);
      refetchMentalHealthHistory();
      
      toast({
        title: "Questionnaire Submitted",
        description: "Your responses have been recorded and analyzed.",
      });

    } catch (error: any) {
      toast({
        title: "Submission Failed",
        description: error.message || "Failed to submit questionnaire. Please try again.",
        variant: "destructive",
      });
    } finally {
      setIsSubmittingMentalHealth(false);
    }
  };

  const handleStartNewMentalHealth = () => {
    setSelectedQuestionnaire(null);
    setCurrentQuestionIndex(0);
    setMentalHealthResponses({});
    setCompletedMentalHealthResponse(null);
    setMentalHealthStartTime(null);
  };

  // Daily Wellness Check handlers
  const handleStartDailyWellness = () => {
    setShowDailyWellness(true);
    setDailyWellnessResponses({});
    setCompletedDailyWellness(false);
  };

  const handleDailyWellnessResponse = (questionId: string, value: number) => {
    setDailyWellnessResponses(prev => ({
      ...prev,
      [questionId]: value
    }));
  };

  const handleSubmitDailyWellness = () => {
    const allAnswered = DAILY_WELLNESS_QUESTIONS.every(q => 
      dailyWellnessResponses[q.id] !== undefined
    );
    
    if (!allAnswered) {
      toast({
        title: "Incomplete",
        description: "Please answer all questions before submitting.",
        variant: "destructive",
      });
      return;
    }
    
    setCompletedDailyWellness(true);
    
    toast({
      title: "Daily Wellness Check Complete!",
      description: "Your responses have been recorded.",
    });
  };

  const handleCloseDailyWellness = () => {
    setShowDailyWellness(false);
    setDailyWellnessResponses({});
    setCompletedDailyWellness(false);
  };

  const getSeverityColor = (level: string) => {
    const colors: Record<string, string> = {
      minimal: "bg-green-500/10 text-green-700 dark:text-green-400",
      low: "bg-green-500/10 text-green-700 dark:text-green-400",
      mild: "bg-yellow-500/10 text-yellow-700 dark:text-yellow-400",
      moderate: "bg-orange-500/10 text-orange-700 dark:text-orange-400",
      moderately_severe: "bg-red-500/10 text-red-700 dark:text-red-400",
      severe: "bg-red-600/10 text-red-700 dark:text-red-400",
      high: "bg-red-500/10 text-red-700 dark:text-red-400",
    };
    return colors[level] || "bg-gray-500/10 text-gray-700 dark:text-gray-400";
  };

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
                <TabsList className="grid w-full grid-cols-6 gap-1">
                  <TabsTrigger value="device" data-testid="tab-device">Device Data</TabsTrigger>
                  <TabsTrigger value="symptom-journal" data-testid="tab-symptom-journal">Symptoms</TabsTrigger>
                  <TabsTrigger value="video-ai" data-testid="tab-video-ai">Video AI</TabsTrigger>
                  <TabsTrigger value="audio-ai" data-testid="tab-audio-ai">Audio AI</TabsTrigger>
                  <TabsTrigger value="paintrack" data-testid="tab-paintrack">
                    <Activity className="w-3 h-3 mr-1" />
                    PainTrack
                  </TabsTrigger>
                  <TabsTrigger value="mental-health" data-testid="tab-mental-health">
                    <Brain className="w-3 h-3 mr-1" />
                    Mental Health
                  </TabsTrigger>
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

                {/* PainTrack Tab */}
                <TabsContent value="paintrack" className="space-y-3">
                  {/* Recent Sessions History */}
                  {paintrackSessions && Array.isArray(paintrackSessions) && paintrackSessions.length > 0 && paintrackStep === 'select-module' && (
                    <div className="p-3 rounded-md border bg-muted/30" data-testid="paintrack-session-history">
                      <h3 className="font-semibold text-xs mb-2">Recent Sessions</h3>
                      {paintrackSessionsLoading ? (
                        <p className="text-xs text-muted-foreground">Loading sessions...</p>
                      ) : (
                        <div className="space-y-2">
                          {paintrackSessions.slice(0, 3).map((session: PaintrackSession) => (
                            <div key={session.id} className="p-2 rounded-md bg-background border text-xs" data-testid={`paintrack-session-${session.id}`}>
                              <div className="flex items-center justify-between">
                                <div className="flex items-center gap-2">
                                  <Activity className="h-3 w-3 text-muted-foreground" />
                                  <span className="font-medium capitalize">{session.laterality || ''} {session.joint}</span>
                                </div>
                                <Badge variant="outline" className="text-xs">
                                  {session.patientVas}/10
                                </Badge>
                              </div>
                              <p className="text-xs text-muted-foreground mt-1">
                                {session.module} • {new Date(session.createdAt).toLocaleDateString()}
                              </p>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  )}

                  {paintrackStep === 'select-module' && (
                    <div className="space-y-4" data-testid="paintrack-module-selection">
                      <div>
                        <h3 className="font-semibold text-sm mb-1">PainTrack - Select Module</h3>
                        <p className="text-xs text-muted-foreground">
                          Choose the type of pain tracking you need
                        </p>
                      </div>

                      <div className="grid gap-3">
                        {['ArthroTrack', 'MuscleTrack', 'PostOpTrack'].map((module) => (
                          <div
                            key={module}
                            className="p-3 rounded-md border hover-elevate cursor-pointer"
                            onClick={() => {
                              setSelectedModule(module);
                              setPaintrackStep('select-joint');
                            }}
                            data-testid={`card-module-${module.toLowerCase()}`}
                          >
                            <div className="flex items-center gap-2">
                              <Activity className="h-4 w-4 text-primary" />
                              <h4 className="font-medium text-sm">{module}</h4>
                            </div>
                            <p className="text-xs text-muted-foreground mt-1">
                              {module === 'ArthroTrack' && 'Track arthritis pain across joints'}
                              {module === 'MuscleTrack' && 'Monitor muscle pain and strain'}
                              {module === 'PostOpTrack' && 'Post-surgery recovery tracking'}
                            </p>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {paintrackStep === 'select-joint' && (
                    <div className="space-y-4" data-testid="paintrack-joint-selection">
                      <div className="flex items-center justify-between">
                        <div>
                          <h3 className="font-semibold text-sm mb-1">Select Joint</h3>
                          <p className="text-xs text-muted-foreground">
                            Module: {selectedModule}
                          </p>
                        </div>
                        <Button
                          onClick={() => setPaintrackStep('select-module')}
                          variant="ghost"
                          size="sm"
                          data-testid="button-back-to-modules"
                        >
                          Back
                        </Button>
                      </div>

                      <div className="grid grid-cols-2 gap-2">
                        {['Knee', 'Hip', 'Shoulder', 'Elbow', 'Wrist', 'Ankle'].map((joint) => (
                          <Button
                            key={joint}
                            variant="outline"
                            className="h-auto p-3 flex-col gap-1"
                            onClick={() => {
                              setSelectedJoint(joint.toLowerCase());
                              setPaintrackStep('instructions');
                            }}
                            data-testid={`button-joint-${joint.toLowerCase()}`}
                          >
                            <Activity className="h-4 w-4" />
                            <span className="text-xs">{joint}</span>
                          </Button>
                        ))}
                      </div>

                      {selectedJoint && (
                        <div className="space-y-2">
                          <Label className="text-sm">Which side?</Label>
                          <RadioGroup
                            value={selectedLaterality || ''}
                            onValueChange={(value) => setSelectedLaterality(value as 'left' | 'right' | 'bilateral')}
                            data-testid="radio-group-laterality"
                          >
                            {['left', 'right', 'bilateral'].map((side) => (
                              <div key={side} className="flex items-center space-x-2">
                                <RadioGroupItem
                                  value={side}
                                  id={`laterality-${side}`}
                                  data-testid={`radio-laterality-${side}`}
                                />
                                <Label
                                  htmlFor={`laterality-${side}`}
                                  className="text-sm font-normal cursor-pointer capitalize"
                                >
                                  {side}
                                </Label>
                              </div>
                            ))}
                          </RadioGroup>
                        </div>
                      )}
                    </div>
                  )}

                  {paintrackStep === 'instructions' && (
                    <div className="space-y-4" data-testid="paintrack-instructions">
                      <div className="flex items-center justify-between">
                        <h3 className="font-semibold text-sm">Recording Instructions</h3>
                        <Button
                          onClick={() => setPaintrackStep('select-joint')}
                          variant="ghost"
                          size="sm"
                          data-testid="button-back-to-joints"
                        >
                          Back
                        </Button>
                      </div>

                      <Alert data-testid="alert-instructions">
                        <Info className="h-4 w-4" />
                        <AlertTitle className="text-sm">Dual-Camera Recording</AlertTitle>
                        <AlertDescription className="text-xs space-y-2">
                          <p>We'll record two simultaneous videos:</p>
                          <ul className="list-disc ml-4 space-y-1">
                            <li>Front camera: Your face (for pain indicators)</li>
                            <li>Back camera: The {selectedJoint} joint</li>
                          </ul>
                        </AlertDescription>
                      </Alert>

                      <div className="space-y-2 text-xs" data-testid="prep-checklist">
                        <p className="font-medium">Before you start:</p>
                        <ul className="space-y-1 ml-4 list-disc text-muted-foreground">
                          <li data-testid="prep-item-lighting">Ensure good lighting on both face and joint</li>
                          <li data-testid="prep-item-positioning">Position camera to see full joint movement</li>
                          <li data-testid="prep-item-stability">Keep camera stable (use a stand if possible)</li>
                          <li data-testid="prep-item-duration">Recording will last 15-20 seconds</li>
                        </ul>
                      </div>

                      <div className="p-3 rounded-md bg-muted/30 border text-xs space-y-1" data-testid="movement-instructions">
                        <p className="font-medium">Guided Movement:</p>
                        <p className="text-muted-foreground" data-testid={`movement-guidance-${selectedJoint}`}>
                          {selectedJoint === 'knee' && 'Fully extend your knee, then slowly flex it'}
                          {selectedJoint === 'hip' && 'Rotate your hip through full range of motion'}
                          {selectedJoint === 'shoulder' && 'Raise arm overhead, then lower slowly'}
                          {selectedJoint === 'elbow' && 'Fully extend elbow, then flex slowly'}
                          {selectedJoint === 'wrist' && 'Rotate wrist through full range'}
                          {selectedJoint === 'ankle' && 'Point toe down, then flex up'}
                        </p>
                      </div>

                      <Button
                        onClick={() => {
                          setPaintrackStep('recording');
                          startDualCameraRecording();
                        }}
                        className="w-full gap-2"
                        data-testid="button-start-recording"
                      >
                        <Camera className="h-4 w-4" />
                        Start Dual-Camera Recording
                      </Button>
                    </div>
                  )}

                  {paintrackStep === 'recording' && (
                    <div className="space-y-4" data-testid="paintrack-recording">
                      <div className="flex items-center justify-between">
                        <h3 className="font-semibold text-sm">Recording in Progress</h3>
                        <Badge variant="outline" className="text-xs">
                          {Math.floor(painRecordingTime / 60)}:{(painRecordingTime % 60).toString().padStart(2, '0')}
                        </Badge>
                      </div>

                      {dualCameraSupported && (
                        <Alert data-testid="alert-iphone17-detected">
                          <Info className="h-4 w-4" />
                          <AlertDescription className="text-xs">
                            iPhone 17 Pro detected - Dual-camera recording active
                          </AlertDescription>
                        </Alert>
                      )}

                      <div className="grid grid-cols-2 gap-2">
                        <div className="space-y-1">
                          <p className="text-xs font-medium">Front Camera (Face)</p>
                          <video
                            ref={frontVideoRef}
                            autoPlay
                            playsInline
                            muted
                            className="w-full rounded-md border bg-black aspect-video"
                            data-testid="video-front-camera"
                          />
                        </div>
                        <div className="space-y-1">
                          <p className="text-xs font-medium">Back Camera (Joint)</p>
                          <video
                            ref={backVideoRef}
                            autoPlay
                            playsInline
                            muted
                            className="w-full rounded-md border bg-black aspect-video"
                            data-testid="video-back-camera"
                          />
                          {!backCameraStream && (
                            <p className="text-xs text-muted-foreground">
                              Single camera mode
                            </p>
                          )}
                        </div>
                      </div>

                      <div className="p-3 rounded-md bg-muted/30 border text-xs" data-testid="movement-guidance-active">
                        <p className="font-medium mb-1">Perform movement:</p>
                        <p className="text-muted-foreground">
                          {selectedJoint === 'knee' && 'Fully extend your knee, then slowly flex it'}
                          {selectedJoint === 'hip' && 'Rotate your hip through full range of motion'}
                          {selectedJoint === 'shoulder' && 'Raise arm overhead, then lower slowly'}
                          {selectedJoint === 'elbow' && 'Fully extend elbow, then flex slowly'}
                          {selectedJoint === 'wrist' && 'Rotate wrist through full range'}
                          {selectedJoint === 'ankle' && 'Point toe down, then flex up'}
                        </p>
                      </div>

                      {isRecordingPain && (
                        <div className="flex items-center justify-center gap-2">
                          <div className="h-3 w-3 rounded-full bg-red-500 animate-pulse" />
                          <span className="text-sm font-medium">Recording...</span>
                        </div>
                      )}

                      <Button
                        onClick={stopDualCameraRecording}
                        variant="outline"
                        className="w-full"
                        disabled={!isRecordingPain}
                        data-testid="button-stop-recording"
                      >
                        Stop Recording
                      </Button>
                    </div>
                  )}

                  {paintrackStep === 'pain-report' && (
                    <div className="space-y-4" data-testid="paintrack-pain-report">
                      <h3 className="font-semibold text-sm">Self-Report Pain Level</h3>

                      <div className="space-y-3">
                        <div>
                          <Label className="text-sm">Pain Level (0-10)</Label>
                          <div className="flex items-center gap-3 mt-2">
                            <span className="text-xs text-muted-foreground">0</span>
                            <input
                              type="range"
                              min="0"
                              max="10"
                              value={painVAS}
                              onChange={(e) => setPainVAS(parseInt(e.target.value))}
                              className="flex-1"
                              data-testid="slider-pain-vas"
                            />
                            <span className="text-xs text-muted-foreground">10</span>
                          </div>
                          <div className="flex justify-between mt-1 text-xs text-muted-foreground">
                            <span>No pain</span>
                            <Badge variant="outline" className="font-bold text-base">
                              {painVAS}
                            </Badge>
                            <span>Worst pain</span>
                          </div>
                        </div>

                        <div>
                          <Label className="text-sm">Additional Notes (Optional)</Label>
                          <textarea
                            value={painNotes}
                            onChange={(e) => setPainNotes(e.target.value)}
                            className="w-full mt-1 p-2 text-sm border rounded-md"
                            rows={3}
                            placeholder="Describe your pain, triggers, or any other details..."
                            data-testid="textarea-pain-notes"
                          />
                        </div>

                        <div className="flex items-center space-x-2">
                          <input
                            type="checkbox"
                            checked={medicationTaken}
                            onChange={(e) => setMedicationTaken(e.target.checked)}
                            id="medication-taken"
                            data-testid="checkbox-medication-taken"
                            className="rounded"
                          />
                          <Label htmlFor="medication-taken" className="text-sm cursor-pointer">
                            I took pain medication today
                          </Label>
                        </div>

                        {medicationTaken && (
                          <div>
                            <Label className="text-sm">Medication Details</Label>
                            <input
                              type="text"
                              value={medicationDetails}
                              onChange={(e) => setMedicationDetails(e.target.value)}
                              className="w-full mt-1 p-2 text-sm border rounded-md"
                              placeholder="e.g., Ibuprofen 400mg at 9am"
                              data-testid="input-medication-details"
                            />
                          </div>
                        )}
                      </div>

                      <Button
                        onClick={() => {
                          if (!selectedModule || !selectedJoint) {
                            toast({
                              variant: "destructive",
                              title: "Missing Information",
                              description: "Module and joint are required",
                            });
                            return;
                          }
                          
                          // Calculate video metadata
                          const hasVideos = frontVideoBlob !== null;
                          const hasDualCamera = backVideoBlob !== null;
                          const recordingDuration = painRecordingTime;
                          
                          // S3 Upload Implementation
                          const uploadVideosAndSubmit = async () => {
                            let frontVideoUrl: string | undefined;
                            let jointVideoUrl: string | undefined;
                            let frontUploadSuccess = false;
                            let backUploadSuccess = false;

                            try {
                              // Upload front camera video if available
                              if (frontVideoBlob) {
                                const frontFormData = new FormData();
                                frontFormData.append('video', frontVideoBlob, `paintrack-front-${Date.now()}.webm`);
                                frontFormData.append('videoType', 'front');
                                frontFormData.append('module', selectedModule);
                                frontFormData.append('joint', selectedJoint || '');

                                const frontUpload = await fetch('/api/paintrack/upload-video', {
                                  method: 'POST',
                                  body: frontFormData,
                                });

                                if (frontUpload.ok) {
                                  const frontData = await frontUpload.json();
                                  frontVideoUrl = frontData.videoUrl;
                                  frontUploadSuccess = true;
                                } else {
                                  console.warn("Front camera upload failed, proceeding without video");
                                }
                              }

                              // Upload back camera video if available
                              if (backVideoBlob) {
                                const backFormData = new FormData();
                                backFormData.append('video', backVideoBlob, `paintrack-back-${Date.now()}.webm`);
                                backFormData.append('videoType', 'back');
                                backFormData.append('module', selectedModule);
                                backFormData.append('joint', selectedJoint || '');

                                const backUpload = await fetch('/api/paintrack/upload-video', {
                                  method: 'POST',
                                  body: backFormData,
                                });

                                if (backUpload.ok) {
                                  const backData = await backUpload.json();
                                  jointVideoUrl = backData.videoUrl;
                                  backUploadSuccess = true;
                                } else {
                                  console.warn("Back camera upload failed, proceeding without joint video");
                                }
                              }
                            } catch (uploadError) {
                              console.error("Video upload error:", uploadError);
                              toast({
                                variant: "destructive",
                                title: "Video Upload Issue",
                                description: "Videos could not be uploaded, but your pain report will still be saved.",
                              });
                            }

                            // Submit session with video URLs or metadata only
                            // dualCameraSupported is true ONLY if BOTH uploads succeeded
                            const actualDualCameraSuccess = frontUploadSuccess && backUploadSuccess;
                            
                            console.log(`[PainTrack] Upload summary - Front: ${frontUploadSuccess}, Back: ${backUploadSuccess}, Dual: ${actualDualCameraSuccess}`);
                            
                            createPaintrackSessionMutation.mutate({
                              module: selectedModule,
                              joint: selectedJoint,
                              laterality: selectedLaterality,
                              patientVas: painVAS,
                              patientNotes: painNotes,
                              medicationTaken,
                              medicationDetails,
                              recordingDuration: hasVideos ? recordingDuration : undefined,
                              dualCameraSupported: actualDualCameraSuccess, // TRUE only if both uploaded
                              frontVideoUrl,
                              jointVideoUrl,
                            });
                          };

                          // Execute upload and submission
                          uploadVideosAndSubmit();
                          setPaintrackStep('complete');
                        }}
                        className="w-full"
                        disabled={createPaintrackSessionMutation.isPending}
                        data-testid="button-submit-pain-report"
                      >
                        {createPaintrackSessionMutation.isPending ? "Saving..." : "Submit Report"}
                      </Button>
                    </div>
                  )}

                  {paintrackStep === 'complete' && (
                    <div className="space-y-4" data-testid="paintrack-complete">
                      <div className="flex items-center gap-2">
                        <CheckCircle className="h-5 w-5 text-green-600" />
                        <h3 className="font-semibold">Session Complete!</h3>
                      </div>

                      <div className="p-3 rounded-md bg-muted/30 border space-y-2 text-sm">
                        <div className="grid grid-cols-2 gap-2">
                          <div>
                            <p className="text-xs text-muted-foreground">Module</p>
                            <p className="font-medium">{selectedModule}</p>
                          </div>
                          <div>
                            <p className="text-xs text-muted-foreground">Joint</p>
                            <p className="font-medium capitalize">{selectedLaterality} {selectedJoint}</p>
                          </div>
                          <div>
                            <p className="text-xs text-muted-foreground">Pain Level</p>
                            <p className="font-medium">{painVAS}/10</p>
                          </div>
                          {medicationTaken && (
                            <div>
                              <p className="text-xs text-muted-foreground">Medication</p>
                              <p className="font-medium text-xs">{medicationDetails || 'Yes'}</p>
                            </div>
                          )}
                        </div>
                      </div>

                      {frontVideoBlob && (
                        <Alert data-testid="alert-videos-captured">
                          <CheckCircle className="h-4 w-4" />
                          <AlertDescription className="text-xs">
                            ✓ Video captured ({backVideoBlob ? 'Dual-camera' : 'Single camera'}) - {painRecordingTime}s recording
                          </AlertDescription>
                        </Alert>
                      )}

                      <p className="text-xs text-muted-foreground">
                        Your self-reported pain level has been recorded. Optional video analysis (if recorded) will extract technical movement metrics only - pain is always based on your self-report.
                      </p>

                      <Button
                        onClick={() => {
                          setPaintrackStep('select-module');
                          setSelectedModule(null);
                          setSelectedJoint(null);
                          setSelectedLaterality(null);
                          setPainVAS(5);
                          setPainNotes('');
                          setMedicationTaken(false);
                          setMedicationDetails('');
                        }}
                        className="w-full"
                        data-testid="button-new-session"
                      >
                        New Session
                      </Button>
                    </div>
                  )}
                </TabsContent>

                {/* Mental Health AI Tab */}
                <TabsContent value="mental-health" className="space-y-3">
                  {/* Daily Wellness Check */}
                  {!showDailyWellness && !selectedQuestionnaire && !completedMentalHealthResponse && (
                    <div className="p-4 rounded-md border bg-gradient-to-br from-blue-50 to-indigo-50 dark:from-blue-950/20 dark:to-indigo-950/20 hover-elevate" data-testid="card-daily-wellness">
                      <div className="flex items-start justify-between gap-3">
                        <div className="flex-1">
                          <div className="flex items-center gap-2 mb-2">
                            <Heart className="h-4 w-4 text-blue-600" />
                            <h3 className="font-semibold text-sm">Daily Wellness Check</h3>
                            <Badge variant="outline" className="text-xs">Quick • 1 min</Badge>
                          </div>
                          <p className="text-xs text-muted-foreground mb-3">
                            Quick check-in about how you're feeling <strong>today</strong>. Track your daily mood, anxiety, stress, energy, and sleep.
                          </p>
                          <Button
                            onClick={handleStartDailyWellness}
                            size="sm"
                            className="w-full gap-2"
                            data-testid="button-start-daily-wellness"
                          >
                            <CheckCircle className="h-3 w-3" />
                            Start Daily Check-In
                          </Button>
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Daily Wellness Form */}
                  {showDailyWellness && !completedDailyWellness && (
                    <div className="space-y-4" data-testid="daily-wellness-form">
                      <div className="flex items-center justify-between">
                        <h3 className="font-semibold text-sm">Daily Wellness Check</h3>
                        <Button
                          onClick={handleCloseDailyWellness}
                          variant="ghost"
                          size="sm"
                          data-testid="button-close-daily-wellness"
                        >
                          Cancel
                        </Button>
                      </div>

                      <div className="space-y-4">
                        {DAILY_WELLNESS_QUESTIONS.map((question, idx) => (
                          <div key={question.id} className="space-y-2">
                            <Label className="text-sm font-medium">
                              {idx + 1}. {question.text}
                            </Label>
                            <RadioGroup
                              value={dailyWellnessResponses[question.id]?.toString()}
                              onValueChange={(value) => handleDailyWellnessResponse(question.id, parseInt(value))}
                              data-testid={`radio-group-${question.id}`}
                            >
                              {question.options.map((option) => (
                                <div key={option.value} className="flex items-center space-x-2">
                                  <RadioGroupItem
                                    value={option.value.toString()}
                                    id={`${question.id}-${option.value}`}
                                    data-testid={`radio-${question.id}-${option.value}`}
                                  />
                                  <Label
                                    htmlFor={`${question.id}-${option.value}`}
                                    className="text-sm font-normal cursor-pointer"
                                  >
                                    {option.label}
                                  </Label>
                                </div>
                              ))}
                            </RadioGroup>
                          </div>
                        ))}
                      </div>

                      <Button
                        onClick={handleSubmitDailyWellness}
                        className="w-full"
                        disabled={Object.keys(dailyWellnessResponses).length < DAILY_WELLNESS_QUESTIONS.length}
                        data-testid="button-submit-daily-wellness"
                      >
                        Submit Daily Check-In
                      </Button>
                    </div>
                  )}

                  {/* Daily Wellness Completion */}
                  {completedDailyWellness && (
                    <div className="space-y-4" data-testid="daily-wellness-complete">
                      <div className="flex items-center justify-between">
                        <h3 className="text-lg font-semibold flex items-center gap-2">
                          <CheckCircle className="h-5 w-5 text-green-600" />
                          Daily Check-In Complete
                        </h3>
                        <Button
                          onClick={handleCloseDailyWellness}
                          variant="outline"
                          size="sm"
                          data-testid="button-done-daily-wellness"
                        >
                          Done
                        </Button>
                      </div>

                      <div className="p-3 rounded-md bg-muted/30 border text-sm space-y-2">
                        <p className="text-muted-foreground">
                          Thank you for completing today's wellness check! Your responses help track patterns and identify changes over time.
                        </p>
                      </div>
                    </div>
                  )}

                  {isLoadingQuestionnaires ? (
                    <div className="flex items-center justify-center py-8" data-testid="loading-questionnaires">
                      <Loader2 className="h-6 w-6 animate-spin" />
                    </div>
                  ) : isErrorQuestionnaires ? (
                    <Alert variant="destructive" data-testid="alert-questionnaires-error">
                      <AlertCircle className="h-4 w-4" />
                      <AlertTitle>Error Loading Questionnaires</AlertTitle>
                      <AlertDescription>
                        {questionnaireError?.message || "Failed to load mental health questionnaires. Please try again."}
                      </AlertDescription>
                    </Alert>
                  ) : completedMentalHealthResponse ? (
                    <div className="space-y-4" data-testid="mental-health-results">
                      <div className="flex items-center justify-between">
                        <h3 className="text-lg font-semibold flex items-center gap-2">
                          <CheckCircle className="h-5 w-5 text-green-600" />
                          Assessment Complete
                        </h3>
                        <Button
                          onClick={handleStartNewMentalHealth}
                          variant="outline"
                          size="sm"
                          data-testid="button-start-new-assessment"
                        >
                          New Assessment
                        </Button>
                      </div>

                      {completedMentalHealthResponse.crisis_intervention?.crisis_detected && (
                        <Alert variant="destructive" className="border-red-600" data-testid="alert-crisis-detected">
                          <AlertTriangle className="h-4 w-4" />
                          <AlertTitle>Immediate Support Available</AlertTitle>
                          <AlertDescription className="space-y-2">
                            <p>{completedMentalHealthResponse.crisis_intervention.intervention_message}</p>
                            <div className="mt-3 space-y-2">
                              {completedMentalHealthResponse.crisis_intervention.crisis_hotlines.map((hotline, idx) => (
                                <div key={idx} className="flex items-start gap-2 text-sm">
                                  <Phone className="h-3 w-3 mt-0.5 flex-shrink-0" />
                                  <div>
                                    <strong>{hotline.name}:</strong> {hotline.phone || hotline.sms}
                                    <p className="text-xs opacity-90">{hotline.description}</p>
                                  </div>
                                </div>
                              ))}
                            </div>
                          </AlertDescription>
                        </Alert>
                      )}

                      <div className="grid gap-3">
                        <div className="p-3 rounded-md bg-card border">
                          <div className="flex items-center justify-between mb-2">
                            <span className="text-sm font-medium">Overall Score</span>
                            <Badge className={getSeverityColor(completedMentalHealthResponse.score.severity_level)}>
                              {completedMentalHealthResponse.score.severity_level.replace(/_/g, ' ')}
                            </Badge>
                          </div>
                          <div className="flex items-baseline gap-2">
                            <span className="text-2xl font-bold">{completedMentalHealthResponse.score.total_score}</span>
                            <span className="text-sm text-muted-foreground">/ {completedMentalHealthResponse.score.max_score}</span>
                          </div>
                          <p className="text-xs text-muted-foreground mt-1">
                            {completedMentalHealthResponse.score.severity_description}
                          </p>
                        </div>

                        <div className="p-3 rounded-md bg-muted/30 border text-sm space-y-2">
                          <p className="font-medium flex items-center gap-2">
                            <Brain className="h-3 w-3" />
                            AI Analysis Summary
                          </p>
                          <p className="text-muted-foreground text-xs leading-relaxed">
                            {completedMentalHealthResponse.score.neutral_summary}
                          </p>
                          {completedMentalHealthResponse.score.key_observations.length > 0 && (
                            <div className="mt-2 space-y-1">
                              <p className="text-xs font-medium">Key Observations:</p>
                              <ul className="space-y-0.5">
                                {completedMentalHealthResponse.score.key_observations.map((obs, idx) => (
                                  <li key={idx} className="text-xs text-muted-foreground flex items-start gap-2">
                                    <Check className="h-2 w-2 mt-0.5 flex-shrink-0" />
                                    <span>{obs}</span>
                                  </li>
                                ))}
                              </ul>
                            </div>
                          )}
                        </div>

                        {mentalHealthHistoryData?.history && mentalHealthHistoryData.history.length > 1 && (
                          <div className="p-3 rounded-md bg-muted/30 border">
                            <p className="text-sm font-medium mb-2">Recent History</p>
                            <div className="space-y-1">
                              {mentalHealthHistoryData.history.slice(0, 3).map((item: MentalHealthHistoryItem) => (
                                <div key={item.response_id} className="flex items-center justify-between text-xs">
                                  <span className="text-muted-foreground">
                                    {format(new Date(item.completed_at), 'MMM d, yyyy')}
                                  </span>
                                  <Badge variant="outline" className={getSeverityColor(item.severity_level)}>
                                    {item.total_score}/{item.max_score}
                                  </Badge>
                                </div>
                              ))}
                            </div>
                            <Link href="/mental-health">
                              <Button variant="link" size="sm" className="p-0 h-auto text-xs mt-2">
                                View Full History →
                              </Button>
                            </Link>
                          </div>
                        )}
                      </div>
                    </div>
                  ) : selectedQuestionnaire ? (
                    <div className="space-y-4" data-testid="questionnaire-active">
                      <div className="flex items-center justify-between">
                        <div>
                          <h3 className="font-semibold text-sm">{selectedQuestionnaire.full_name}</h3>
                          <p className="text-xs text-muted-foreground">{selectedQuestionnaire.description}</p>
                        </div>
                        <Button
                          onClick={handleStartNewMentalHealth}
                          variant="ghost"
                          size="sm"
                          data-testid="button-cancel-questionnaire"
                        >
                          Cancel
                        </Button>
                      </div>

                      <div className="space-y-1">
                        <div className="flex items-center justify-between text-xs text-muted-foreground">
                          <span>Question {currentQuestionIndex + 1} of {selectedQuestionnaire.questions.length}</span>
                          <span>{Math.round(((currentQuestionIndex + 1) / selectedQuestionnaire.questions.length) * 100)}%</span>
                        </div>
                        <Progress value={((currentQuestionIndex + 1) / selectedQuestionnaire.questions.length) * 100} />
                      </div>

                      <ScrollArea className="h-[300px]">
                        <div className="space-y-4 pr-4">
                          <div className="p-4 rounded-md bg-muted/30 border">
                            <p className="text-sm font-medium leading-relaxed">
                              {selectedQuestionnaire.questions[currentQuestionIndex].text}
                            </p>
                            <p className="text-xs text-muted-foreground mt-2">
                              {selectedQuestionnaire.timeframe}
                            </p>
                          </div>

                          <RadioGroup
                            value={mentalHealthResponses[selectedQuestionnaire.questions[currentQuestionIndex].id]?.response.toString() || ""}
                            onValueChange={(value) => {
                              const option = selectedQuestionnaire.response_options.find(o => o.value.toString() === value);
                              if (option) {
                                handleMentalHealthResponseSelect(
                                  selectedQuestionnaire.questions[currentQuestionIndex].id,
                                  option.value,
                                  option.label
                                );
                              }
                            }}
                            className="space-y-2"
                          >
                            {selectedQuestionnaire.response_options.map((option) => (
                              <div
                                key={option.value}
                                className="flex items-center space-x-2 p-2 rounded-md hover-elevate border"
                                data-testid={`radio-option-${option.value}`}
                              >
                                <RadioGroupItem value={option.value.toString()} id={`option-${option.value}`} />
                                <Label htmlFor={`option-${option.value}`} className="flex-1 text-sm cursor-pointer">
                                  {option.label}
                                </Label>
                              </div>
                            ))}
                          </RadioGroup>
                        </div>
                      </ScrollArea>

                      <div className="flex items-center justify-between gap-2 pt-2">
                        <Button
                          onClick={handleMentalHealthPrevious}
                          disabled={currentQuestionIndex === 0}
                          variant="outline"
                          size="sm"
                          data-testid="button-previous-question"
                        >
                          Previous
                        </Button>
                        
                        {currentQuestionIndex === selectedQuestionnaire.questions.length - 1 ? (
                          <Button
                            onClick={handleSubmitMentalHealth}
                            disabled={!selectedQuestionnaire.questions.every(q => mentalHealthResponses[q.id]) || isSubmittingMentalHealth}
                            size="sm"
                            className="gap-2"
                            data-testid="button-submit-questionnaire"
                          >
                            {isSubmittingMentalHealth ? (
                              <>
                                <Loader2 className="h-3 w-3 animate-spin" />
                                Analyzing...
                              </>
                            ) : (
                              <>
                                <CheckCircle2 className="h-3 w-3" />
                                Submit & Analyze
                              </>
                            )}
                          </Button>
                        ) : (
                          <Button
                            onClick={handleMentalHealthNext}
                            disabled={!mentalHealthResponses[selectedQuestionnaire.questions[currentQuestionIndex].id]}
                            size="sm"
                            data-testid="button-next-question"
                          >
                            Next
                          </Button>
                        )}
                      </div>
                    </div>
                  ) : (
                    <div className="space-y-4" data-testid="questionnaire-selection">
                      <div>
                        <h3 className="font-semibold text-sm mb-1">Mental Health Screening</h3>
                        <p className="text-xs text-muted-foreground">
                          Select a validated questionnaire to assess your mental wellbeing
                        </p>
                      </div>

                      <div className="grid gap-3">
                        {questionnairesData?.questionnaires?.map((template: QuestionnaireTemplate) => (
                          <div
                            key={template.type}
                            className="p-3 rounded-md border hover-elevate cursor-pointer"
                            onClick={() => handleStartQuestionnaire(template)}
                            data-testid={`card-questionnaire-${template.type}`}
                          >
                            <div className="flex items-start justify-between gap-2">
                              <div className="flex-1">
                                <div className="flex items-center gap-2">
                                  <ClipboardList className="h-4 w-4" />
                                  <h4 className="font-medium text-sm">{template.full_name}</h4>
                                </div>
                                <p className="text-xs text-muted-foreground mt-1">{template.description}</p>
                                <div className="flex items-center gap-3 mt-2 text-xs text-muted-foreground">
                                  <span>{template.questions.length} questions</span>
                                  <span>•</span>
                                  <span>~2-3 min</span>
                                  {template.public_domain && (
                                    <>
                                      <span>•</span>
                                      <Badge variant="outline" className="h-5 text-xs">Public Domain</Badge>
                                    </>
                                  )}
                                </div>
                              </div>
                              <FileText className="h-4 w-4 text-muted-foreground" />
                            </div>
                          </div>
                        ))}
                      </div>

                      <Alert className="text-xs" data-testid="alert-mental-health-disclaimer">
                        <Info className="h-3 w-3" />
                        <AlertDescription>
                          These are screening tools, not diagnostic assessments. Results are analyzed by AI for pattern detection and stored securely for trend tracking.
                        </AlertDescription>
                      </Alert>

                      {mentalHealthHistoryData?.history && mentalHealthHistoryData.history.length > 0 && (
                        <div className="p-3 rounded-md bg-muted/30 border">
                          <p className="text-sm font-medium mb-2">Recent Assessments</p>
                          <div className="space-y-1">
                            {mentalHealthHistoryData.history.slice(0, 2).map((item: MentalHealthHistoryItem) => (
                              <div key={item.response_id} className="flex items-center justify-between text-xs">
                                <span className="text-muted-foreground">
                                  {format(new Date(item.completed_at), 'MMM d, h:mm a')}
                                </span>
                                <div className="flex items-center gap-2">
                                  <Badge variant="outline" className={getSeverityColor(item.severity_level)}>
                                    {item.total_score}/{item.max_score}
                                  </Badge>
                                  {item.crisis_detected && (
                                    <AlertTriangle className="h-3 w-3 text-red-600" />
                                  )}
                                </div>
                              </div>
                            ))}
                          </div>
                          <Link href="/mental-health">
                            <Button variant="link" size="sm" className="p-0 h-auto text-xs mt-2">
                              View All History →
                            </Button>
                          </Link>
                        </div>
                      )}
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
