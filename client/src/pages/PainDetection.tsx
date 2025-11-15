import { useState, useRef, useEffect } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import * as faceLandmarksDetection from "@tensorflow-models/face-landmarks-detection";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Checkbox } from "@/components/ui/checkbox";
import { useToast } from "@/hooks/use-toast";
import { queryClient, apiRequest } from "@/lib/queryClient";
import { Camera, AlertCircle, TrendingUp, TrendingDown, Minus } from "lucide-react";
import { format } from "date-fns";

interface FacialMetrics {
  left_eyebrow_angle: number;
  right_eyebrow_angle: number;
  eyebrow_asymmetry: number;
  left_nasolabial_tension: number;
  right_nasolabial_tension: number;
  forehead_contractions: number;
  eye_contractions: number;
  mouth_contractions: number;
  grimace_intensity: number;
  grimace_duration_ms: number;
  facial_landmarks: any;
  recording_quality: string;
}

interface PainMeasurement {
  id: number;
  facial_stress_score: number;
  pain_severity_estimate: string;
  change_from_previous: number | null;
  created_at: string;
  alert_message?: string;
}

export default function PainDetection() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isRecording, setIsRecording] = useState(false);
  const [countdown, setCountdown] = useState(10);
  const [detector, setDetector] = useState<any>(null);
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [facialMetrics, setFacialMetrics] = useState<FacialMetrics | null>(null);
  const [showQuestionnaire, setShowQuestionnaire] = useState(false);
  const [latestMeasurement, setLatestMeasurement] = useState<PainMeasurement | null>(null);
  
  // Facial analysis state
  const [contractionCounts, setContractionCounts] = useState({
    forehead: 0,
    eye: 0,
    mouth: 0
  });
  const [previousLandmarks, setPreviousLandmarks] = useState<any>(null);
  const [grimaceStartTime, setGrimaceStartTime] = useState<number | null>(null);
  const [maxGrimaceIntensity, setMaxGrimaceIntensity] = useState(0);
  
  // Questionnaire state
  const [painLevel, setPainLevel] = useState(5);
  const [painLocation, setPainLocation] = useState("");
  const [painType, setPainType] = useState("");
  const [painDuration, setPainDuration] = useState("");
  const [painTriggers, setPainTriggers] = useState("");
  const [affectsSleep, setAffectsSleep] = useState(false);
  const [affectsActivities, setAffectsActivities] = useState(false);
  const [affectsMood, setAffectsMood] = useState(false);
  const [medicationTaken, setMedicationTaken] = useState(false);
  const [medicationNames, setMedicationNames] = useState("");
  const [medicationEffectiveness, setMedicationEffectiveness] = useState("");
  const [additionalNotes, setAdditionalNotes] = useState("");
  
  const { toast } = useToast();

  // Fetch recent measurements
  const { data: recentMeasurements } = useQuery<{ measurements: PainMeasurement[] }>({
    queryKey: ["/api/v1/pain-tracking/measurements/recent"],
    queryFn: async () => {
      const res = await fetch("/api/v1/pain-tracking/measurements/recent?days=7");
      return res.json();
    },
  });

  // Fetch current trend
  const { data: currentTrend } = useQuery({
    queryKey: ["/api/v1/pain-tracking/trends/current"],
  });

  // Submit measurement mutation
  const submitMeasurement = useMutation({
    mutationFn: async (metrics: FacialMetrics) => {
      const res = await apiRequest("/api/v1/pain-tracking/measurements", {
        method: "POST",
        body: JSON.stringify({ facial_metrics: metrics }),
      });
      return res.json();
    },
    onSuccess: (data) => {
      setLatestMeasurement(data);
      setShowQuestionnaire(true);
      queryClient.invalidateQueries({ queryKey: ["/api/v1/pain-tracking/measurements/recent"] });
      queryClient.invalidateQueries({ queryKey: ["/api/v1/pain-tracking/trends/current"] });
      
      if (data.alert_message) {
        toast({
          title: "Pain Alert",
          description: data.alert_message,
          variant: "destructive",
        });
      }
    },
  });

  // Submit questionnaire mutation
  const submitQuestionnaire = useMutation({
    mutationFn: async (data: any) => {
      if (!latestMeasurement) throw new Error("No measurement found");
      return await apiRequest(`/api/v1/pain-tracking/measurements/${latestMeasurement.id}/questionnaire`, {
        method: "POST",
        body: JSON.stringify(data),
      });
    },
    onSuccess: () => {
      toast({
        title: "Complete",
        description: "Pain assessment completed successfully",
      });
      setShowQuestionnaire(false);
      resetQuestionnaire();
    },
  });

  // Initialize face detector
  useEffect(() => {
    const loadModel = async () => {
      try {
        const model = faceLandmarksDetection.SupportedModels.MediaPipeFaceMesh;
        const detectorConfig: faceLandmarksDetection.MediaPipeFaceMeshMediaPipeModelConfig = {
          runtime: "mediapipe",
          solutionPath: "https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh",
          refineLandmarks: true,
        };
        const faceDetector = await faceLandmarksDetection.createDetector(model, detectorConfig);
        setDetector(faceDetector);
      } catch (error) {
        console.error("Error loading face detector:", error);
        toast({
          title: "Error",
          description: "Failed to load facial detection model",
          variant: "destructive",
        });
      }
    };
    loadModel();
  }, []);

  const startCamera = async () => {
    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, facingMode: "user" },
      });
      
      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream;
        videoRef.current.play();
      }
      
      setStream(mediaStream);
    } catch (error) {
      console.error("Error accessing camera:", error);
      toast({
        title: "Camera Error",
        description: "Could not access camera. Please grant camera permissions.",
        variant: "destructive",
      });
    }
  };

  const stopCamera = () => {
    if (stream) {
      stream.getTracks().forEach(track => track.stop());
      setStream(null);
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
  };

  const calculateEyebrowAngle = (landmarks: any, side: "left" | "right") => {
    // Simplified eyebrow angle calculation
    // In reality, we'd use specific landmark points for eyebrows
    const indices = side === "left" ? [70, 63] : [300, 293];
    const p1 = landmarks.keypoints[indices[0]];
    const p2 = landmarks.keypoints[indices[1]];
    
    const angle = Math.atan2(p2.y - p1.y, p2.x - p1.x) * (180 / Math.PI);
    return Math.abs(angle);
  };

  const calculateNasolabialTension = (landmarks: any, side: "left" | "right") => {
    // Simplified nasolabial fold tension (0-1 scale)
    // In production, we'd use specific facial mesh points
    const baseIndices = side === "left" ? [205, 50] : [425, 280];
    const p1 = landmarks.keypoints[baseIndices[0]];
    const p2 = landmarks.keypoints[baseIndices[1]];
    
    const distance = Math.sqrt(Math.pow(p2.x - p1.x, 2) + Math.pow(p2.y - p1.y, 2));
    // Normalize to 0-1 scale (adjust threshold based on testing)
    return Math.min(distance / 100, 1.0);
  };

  const detectMicroContractions = (currentLandmarks: any, previousLandmarks: any) => {
    if (!previousLandmarks) return { forehead: 0, eye: 0, mouth: 0 };
    
    // Detect small movements in facial regions
    let forehead = 0, eye = 0, mouth = 0;
    
    // Forehead region (simplified)
    const foreheadIndices = [10, 67, 69, 104, 108, 151, 337, 338];
    foreheadIndices.forEach(idx => {
      const curr = currentLandmarks.keypoints[idx];
      const prev = previousLandmarks.keypoints[idx];
      const movement = Math.sqrt(Math.pow(curr.x - prev.x, 2) + Math.pow(curr.y - prev.y, 2));
      if (movement > 2) forehead++;
    });
    
    // Eye region
    const eyeIndices = [33, 133, 160, 159, 158, 157, 173, 263, 362, 386, 385, 384, 398];
    eyeIndices.forEach(idx => {
      const curr = currentLandmarks.keypoints[idx];
      const prev = previousLandmarks.keypoints[idx];
      const movement = Math.sqrt(Math.pow(curr.x - prev.x, 2) + Math.pow(curr.y - prev.y, 2));
      if (movement > 1.5) eye++;
    });
    
    // Mouth region
    const mouthIndices = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291];
    mouthIndices.forEach(idx => {
      const curr = currentLandmarks.keypoints[idx];
      const prev = previousLandmarks.keypoints[idx];
      const movement = Math.sqrt(Math.pow(curr.x - prev.x, 2) + Math.pow(curr.y - prev.y, 2));
      if (movement > 2) mouth++;
    });
    
    return { forehead, eye, mouth };
  };

  const calculateGrimaceIntensity = (landmarks: any) => {
    // Simplified grimace detection based on mouth corners and lip tightness
    const mouthLeft = landmarks.keypoints[61];
    const mouthRight = landmarks.keypoints[291];
    const upperLip = landmarks.keypoints[13];
    const lowerLip = landmarks.keypoints[14];
    
    const mouthWidth = Math.sqrt(Math.pow(mouthRight.x - mouthLeft.x, 2) + Math.pow(mouthRight.y - mouthLeft.y, 2));
    const lipGap = Math.abs(lowerLip.y - upperLip.y);
    
    // Grimacing typically shows tightened mouth with specific width/height ratio
    const grimaceRatio = mouthWidth / (lipGap + 1);
    const intensity = Math.min(grimaceRatio / 10, 1.0);
    
    return intensity;
  };

  const analyzeFace = async () => {
    if (!detector || !videoRef.current) return;
    
    const faces = await detector.estimateFaces(videoRef.current);
    
    if (faces && faces.length > 0) {
      const face = faces[0];
      
      // Draw landmarks on canvas
      if (canvasRef.current) {
        const ctx = canvasRef.current.getContext("2d");
        if (ctx && videoRef.current) {
          ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
          ctx.drawImage(videoRef.current, 0, 0, canvasRef.current.width, canvasRef.current.height);
          
          // Draw keypoints
          face.keypoints.forEach((point: any) => {
            ctx.fillStyle = "#00FF00";
            ctx.beginPath();
            ctx.arc(point.x, point.y, 1, 0, 2 * Math.PI);
            ctx.fill();
          });
        }
      }
      
      // Calculate metrics
      const leftEyebrow = calculateEyebrowAngle(face, "left");
      const rightEyebrow = calculateEyebrowAngle(face, "right");
      const eyebrowAsymmetry = Math.abs(leftEyebrow - rightEyebrow) / 90; // Normalize
      
      const leftNasolabial = calculateNasolabialTension(face, "left");
      const rightNasolabial = calculateNasolabialTension(face, "right");
      
      const grimaceIntensity = calculateGrimaceIntensity(face);
      
      // Track grimacing duration
      if (grimaceIntensity > 0.5) {
        if (grimaceStartTime === null) {
          setGrimaceStartTime(Date.now());
        }
        setMaxGrimaceIntensity(Math.max(maxGrimaceIntensity, grimaceIntensity));
      } else {
        setGrimaceStartTime(null);
      }
      
      // Detect micro contractions
      if (previousLandmarks) {
        const contractions = detectMicroContractions(face, previousLandmarks);
        setContractionCounts(prev => ({
          forehead: prev.forehead + contractions.forehead,
          eye: prev.eye + contractions.eye,
          mouth: prev.mouth + contractions.mouth
        }));
      }
      
      setPreviousLandmarks(face);
      
      // Update metrics for final submission
      const currentMetrics: FacialMetrics = {
        left_eyebrow_angle: leftEyebrow,
        right_eyebrow_angle: rightEyebrow,
        eyebrow_asymmetry: eyebrowAsymmetry,
        left_nasolabial_tension: leftNasolabial,
        right_nasolabial_tension: rightNasolabial,
        forehead_contractions: contractionCounts.forehead,
        eye_contractions: contractionCounts.eye,
        mouth_contractions: contractionCounts.mouth,
        grimace_intensity: maxGrimaceIntensity,
        grimace_duration_ms: grimaceStartTime ? Date.now() - grimaceStartTime : 0,
        facial_landmarks: face.keypoints,
        recording_quality: "good"
      };
      
      setFacialMetrics(currentMetrics);
    }
  };

  const startRecording = async () => {
    await startCamera();
    setIsRecording(true);
    setCountdown(10);
    setContractionCounts({ forehead: 0, eye: 0, mouth: 0 });
    setPreviousLandmarks(null);
    setGrimaceStartTime(null);
    setMaxGrimaceIntensity(0);
    
    // Start countdown and facial analysis
    const interval = setInterval(async () => {
      await analyzeFace();
    }, 100); // Analyze every 100ms
    
    const countdownInterval = setInterval(() => {
      setCountdown(prev => {
        if (prev <= 1) {
          clearInterval(interval);
          clearInterval(countdownInterval);
          finishRecording();
          return 0;
        }
        return prev - 1;
      });
    }, 1000);
  };

  const finishRecording = () => {
    setIsRecording(false);
    stopCamera();
    
    if (facialMetrics) {
      submitMeasurement.mutate(facialMetrics);
    }
  };

  const handleSubmitQuestionnaire = () => {
    submitQuestionnaire.mutate({
      pain_level_self_reported: painLevel,
      pain_location: painLocation,
      pain_type: painType,
      pain_duration: painDuration,
      pain_triggers: painTriggers,
      affects_sleep: affectsSleep,
      affects_daily_activities: affectsActivities,
      affects_mood: affectsMood,
      pain_medication_taken: medicationTaken,
      medication_names: medicationNames,
      medication_effectiveness: medicationEffectiveness,
      additional_notes: additionalNotes,
    });
  };

  const resetQuestionnaire = () => {
    setPainLevel(5);
    setPainLocation("");
    setPainType("");
    setPainDuration("");
    setPainTriggers("");
    setAffectsSleep(false);
    setAffectsActivities(false);
    setAffectsMood(false);
    setMedicationTaken(false);
    setMedicationNames("");
    setMedicationEffectiveness("");
    setAdditionalNotes("");
  };

  const getSeverityBadge = (severity: string) => {
    const variants: Record<string, "default" | "secondary" | "destructive"> = {
      low: "secondary",
      moderate: "default",
      severe: "destructive",
    };
    return <Badge variant={variants[severity] || "default"}>{severity}</Badge>;
  };

  const getTrendIcon = (changePercent: number | null) => {
    if (!changePercent) return <Minus className="h-4 w-4" />;
    if (changePercent > 0) return <TrendingUp className="h-4 w-4 text-red-500" />;
    return <TrendingDown className="h-4 w-4 text-green-500" />;
  };

  return (
    <div className="container mx-auto p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Pain Detection</h1>
          <p className="text-muted-foreground">Track pain progression through facial analysis</p>
        </div>
      </div>

      {/* Current Trend Summary */}
      {currentTrend && (
        <Card data-testid="card-current-trend">
          <CardHeader>
            <CardTitle>Weekly Trend</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div>
                <p className="text-sm text-muted-foreground">Average Stress Score</p>
                <p className="text-2xl font-bold">{currentTrend.current_week_avg?.toFixed(1) || "N/A"}</p>
              </div>
              <div>
                <p className="text-sm text-muted-foreground">Change from Last Week</p>
                <div className="flex items-center gap-2">
                  <p className="text-2xl font-bold">
                    {currentTrend.change_percent ? `${currentTrend.change_percent > 0 ? "+" : ""}${currentTrend.change_percent.toFixed(1)}%` : "N/A"}
                  </p>
                  {getTrendIcon(currentTrend.change_percent)}
                </div>
              </div>
              <div>
                <p className="text-sm text-muted-foreground">Trend Direction</p>
                <Badge variant={currentTrend.trend_direction === "improving" ? "secondary" : currentTrend.trend_direction === "worsening" ? "destructive" : "default"}>
                  {currentTrend.trend_direction}
                </Badge>
              </div>
            </div>
            {currentTrend.requires_attention && (
              <div className="mt-4 p-3 bg-red-50 dark:bg-red-950 rounded-md flex items-center gap-2">
                <AlertCircle className="h-4 w-4 text-red-600" />
                <p className="text-sm text-red-600 dark:text-red-400">
                  Your pain levels appear elevated. Consider contacting your healthcare provider.
                </p>
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* Camera Interface */}
      {!showQuestionnaire && (
        <Card data-testid="card-camera-interface">
          <CardHeader>
            <CardTitle>Daily Pain Assessment</CardTitle>
            <p className="text-sm text-muted-foreground">
              Simply look at the camera for 10 seconds. We'll analyze micro-expressions to detect pain indicators.
            </p>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="relative w-full max-w-2xl mx-auto">
              <video
                ref={videoRef}
                className="w-full rounded-lg"
                style={{ display: isRecording ? "block" : "none" }}
                data-testid="video-camera-feed"
              />
              <canvas
                ref={canvasRef}
                width={640}
                height={480}
                className="absolute top-0 left-0 w-full h-full rounded-lg"
                style={{ display: isRecording ? "block" : "none" }}
              />
              
              {!isRecording && (
                <div className="flex flex-col items-center justify-center p-12 bg-muted rounded-lg">
                  <Camera className="h-24 w-24 text-muted-foreground mb-4" />
                  <p className="text-lg font-medium mb-2">Ready to start your daily check-in?</p>
                  <p className="text-sm text-muted-foreground mb-4">
                    This will take only 10 seconds
                  </p>
                  <Button
                    onClick={startRecording}
                    disabled={!detector || submitMeasurement.isPending}
                    data-testid="button-start-recording"
                  >
                    <Camera className="mr-2 h-4 w-4" />
                    Start Recording
                  </Button>
                </div>
              )}
              
              {isRecording && (
                <div className="mt-4">
                  <div className="flex items-center justify-between mb-2">
                    <p className="text-sm font-medium">Recording... {countdown}s remaining</p>
                    <p className="text-sm text-muted-foreground">Please look directly at the camera</p>
                  </div>
                  <Progress value={(10 - countdown) * 10} data-testid="progress-recording" />
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Questionnaire */}
      {showQuestionnaire && latestMeasurement && (
        <Card data-testid="card-questionnaire">
          <CardHeader>
            <CardTitle>Pain Assessment Questionnaire</CardTitle>
            <p className="text-sm text-muted-foreground">
              Please answer a few questions about your pain experience
            </p>
          </CardHeader>
          <CardContent className="space-y-6">
            {/* Latest measurement result */}
            <div className="p-4 bg-muted rounded-lg space-y-2">
              <div className="flex items-center justify-between">
                <p className="font-medium">Facial Stress Score</p>
                <p className="text-2xl font-bold" data-testid="text-stress-score">
                  {latestMeasurement.facial_stress_score?.toFixed(1)}
                </p>
              </div>
              <div className="flex items-center justify-between">
                <p className="text-sm text-muted-foreground">Pain Severity</p>
                {getSeverityBadge(latestMeasurement.pain_severity_estimate)}
              </div>
              {latestMeasurement.change_from_previous !== null && (
                <div className="flex items-center justify-between">
                  <p className="text-sm text-muted-foreground">Change from last check-in</p>
                  <div className="flex items-center gap-2">
                    <p className="text-sm font-medium">
                      {latestMeasurement.change_from_previous > 0 ? "+" : ""}
                      {latestMeasurement.change_from_previous.toFixed(1)}%
                    </p>
                    {getTrendIcon(latestMeasurement.change_from_previous)}
                  </div>
                </div>
              )}
            </div>

            {/* Pain level */}
            <div className="space-y-2">
              <Label htmlFor="pain-level">Pain Level (0-10)</Label>
              <div className="flex items-center gap-4">
                <input
                  type="range"
                  id="pain-level"
                  min="0"
                  max="10"
                  value={painLevel}
                  onChange={(e) => setPainLevel(parseInt(e.target.value))}
                  className="flex-1"
                  data-testid="input-pain-level"
                />
                <span className="text-2xl font-bold w-12 text-center" data-testid="text-pain-level">
                  {painLevel}
                </span>
              </div>
              <p className="text-xs text-muted-foreground">0 = No pain, 10 = Worst pain imaginable</p>
            </div>

            {/* Pain location */}
            <div className="space-y-2">
              <Label htmlFor="pain-location">Where is your pain located?</Label>
              <Textarea
                id="pain-location"
                placeholder="e.g., lower back, joints, headache"
                value={painLocation}
                onChange={(e) => setPainLocation(e.target.value)}
                data-testid="input-pain-location"
              />
            </div>

            {/* Pain type */}
            <div className="space-y-2">
              <Label htmlFor="pain-type">What type of pain is it?</Label>
              <Select value={painType} onValueChange={setPainType}>
                <SelectTrigger data-testid="select-pain-type">
                  <SelectValue placeholder="Select pain type" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="sharp">Sharp</SelectItem>
                  <SelectItem value="dull">Dull</SelectItem>
                  <SelectItem value="throbbing">Throbbing</SelectItem>
                  <SelectItem value="burning">Burning</SelectItem>
                  <SelectItem value="aching">Aching</SelectItem>
                  <SelectItem value="stabbing">Stabbing</SelectItem>
                </SelectContent>
              </Select>
            </div>

            {/* Pain duration */}
            <div className="space-y-2">
              <Label htmlFor="pain-duration">How long have you had this pain?</Label>
              <Select value={painDuration} onValueChange={setPainDuration}>
                <SelectTrigger data-testid="select-pain-duration">
                  <SelectValue placeholder="Select duration" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="less_than_hour">Less than 1 hour</SelectItem>
                  <SelectItem value="few_hours">A few hours</SelectItem>
                  <SelectItem value="all_day">All day</SelectItem>
                  <SelectItem value="few_days">A few days</SelectItem>
                  <SelectItem value="week_plus">A week or more</SelectItem>
                </SelectContent>
              </Select>
            </div>

            {/* Pain triggers */}
            <div className="space-y-2">
              <Label htmlFor="pain-triggers">What triggers or worsens your pain?</Label>
              <Textarea
                id="pain-triggers"
                placeholder="e.g., movement, stress, weather changes"
                value={painTriggers}
                onChange={(e) => setPainTriggers(e.target.value)}
                data-testid="input-pain-triggers"
              />
            </div>

            {/* Impact checkboxes */}
            <div className="space-y-3">
              <Label>How does this pain affect you?</Label>
              <div className="space-y-2">
                <div className="flex items-center space-x-2">
                  <Checkbox
                    id="affects-sleep"
                    checked={affectsSleep}
                    onCheckedChange={(checked) => setAffectsSleep(checked as boolean)}
                    data-testid="checkbox-affects-sleep"
                  />
                  <label htmlFor="affects-sleep" className="text-sm cursor-pointer">
                    Affects my sleep
                  </label>
                </div>
                <div className="flex items-center space-x-2">
                  <Checkbox
                    id="affects-activities"
                    checked={affectsActivities}
                    onCheckedChange={(checked) => setAffectsActivities(checked as boolean)}
                    data-testid="checkbox-affects-activities"
                  />
                  <label htmlFor="affects-activities" className="text-sm cursor-pointer">
                    Affects my daily activities
                  </label>
                </div>
                <div className="flex items-center space-x-2">
                  <Checkbox
                    id="affects-mood"
                    checked={affectsMood}
                    onCheckedChange={(checked) => setAffectsMood(checked as boolean)}
                    data-testid="checkbox-affects-mood"
                  />
                  <label htmlFor="affects-mood" className="text-sm cursor-pointer">
                    Affects my mood
                  </label>
                </div>
              </div>
            </div>

            {/* Medication */}
            <div className="space-y-3">
              <div className="flex items-center space-x-2">
                <Checkbox
                  id="medication-taken"
                  checked={medicationTaken}
                  onCheckedChange={(checked) => setMedicationTaken(checked as boolean)}
                  data-testid="checkbox-medication-taken"
                />
                <label htmlFor="medication-taken" className="text-sm font-medium cursor-pointer">
                  I took pain medication today
                </label>
              </div>
              
              {medicationTaken && (
                <div className="space-y-3 ml-6">
                  <div className="space-y-2">
                    <Label htmlFor="medication-names">What medication did you take?</Label>
                    <Textarea
                      id="medication-names"
                      placeholder="e.g., Ibuprofen 400mg"
                      value={medicationNames}
                      onChange={(e) => setMedicationNames(e.target.value)}
                      data-testid="input-medication-names"
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="medication-effectiveness">How effective was it?</Label>
                    <Select value={medicationEffectiveness} onValueChange={setMedicationEffectiveness}>
                      <SelectTrigger data-testid="select-medication-effectiveness">
                        <SelectValue placeholder="Select effectiveness" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="very_effective">Very effective</SelectItem>
                        <SelectItem value="somewhat_effective">Somewhat effective</SelectItem>
                        <SelectItem value="not_effective">Not effective</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </div>
              )}
            </div>

            {/* Additional notes */}
            <div className="space-y-2">
              <Label htmlFor="additional-notes">Additional notes (optional)</Label>
              <Textarea
                id="additional-notes"
                placeholder="Any other details you'd like to share..."
                value={additionalNotes}
                onChange={(e) => setAdditionalNotes(e.target.value)}
                rows={3}
                data-testid="input-additional-notes"
              />
            </div>

            <div className="flex gap-3">
              <Button
                onClick={handleSubmitQuestionnaire}
                disabled={submitQuestionnaire.isPending}
                className="flex-1"
                data-testid="button-submit-questionnaire"
              >
                Submit Assessment
              </Button>
              <Button
                onClick={() => setShowQuestionnaire(false)}
                variant="outline"
                data-testid="button-skip-questionnaire"
              >
                Skip for Now
              </Button>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Recent Measurements */}
      {recentMeasurements && recentMeasurements.length > 0 && !showQuestionnaire && (
        <Card data-testid="card-recent-measurements">
          <CardHeader>
            <CardTitle>Recent Measurements</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {recentMeasurements.map((measurement) => (
                <div
                  key={measurement.id}
                  className="flex items-center justify-between p-3 border rounded-lg"
                  data-testid={`measurement-${measurement.id}`}
                >
                  <div>
                    <p className="font-medium">
                      {format(new Date(measurement.created_at), "MMM d, yyyy 'at' h:mm a")}
                    </p>
                    <div className="flex items-center gap-2 mt-1">
                      {getSeverityBadge(measurement.pain_severity_estimate)}
                      {measurement.change_from_previous !== null && (
                        <span className="text-sm text-muted-foreground">
                          {measurement.change_from_previous > 0 ? "+" : ""}
                          {measurement.change_from_previous.toFixed(1)}%
                        </span>
                      )}
                    </div>
                  </div>
                  <div className="text-right">
                    <p className="text-2xl font-bold">{measurement.facial_stress_score?.toFixed(1)}</p>
                    <p className="text-xs text-muted-foreground">Stress Score</p>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
