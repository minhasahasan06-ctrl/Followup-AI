import { useState, useRef, useEffect } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { useToast } from "@/hooks/use-toast";
import { queryClient, apiRequest } from "@/lib/queryClient";
import { 
  Camera, 
  AlertCircle, 
  TrendingUp, 
  TrendingDown, 
  Minus, 
  Eye, 
  User, 
  Activity,
  Heart,
  Info,
  GitCompare
} from "lucide-react";
import { format } from "date-fns";

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

export default function SymptomJournal() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const streamRef = useRef<MediaStream | null>(null); // Ref to track stream for cleanup
  
  const [selectedBodyArea, setSelectedBodyArea] = useState<string>("legs");
  const [isRecording, setIsRecording] = useState(false);
  const [countdown, setCountdown] = useState(0);
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [capturedImage, setCapturedImage] = useState<string | null>(null);
  const [patientNotes, setPatientNotes] = useState("");
  const [activeTab, setActiveTab] = useState("capture");
  
  // Respiratory analysis state
  const [isRecordingVideo, setIsRecordingVideo] = useState(false);
  const [recordedVideoBlob, setRecordedVideoBlob] = useState<Blob | null>(null);
  const [respiratoryResult, setRespiratoryResult] = useState<any | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const recordedChunksRef = useRef<Blob[]>([]);
  
  // Comparison view state
  const [showComparison, setShowComparison] = useState(false);
  const [selectedMeasurement1, setSelectedMeasurement1] = useState<number | null>(null);
  const [selectedMeasurement2, setSelectedMeasurement2] = useState<number | null>(null);
  
  const { toast } = useToast();

  // Fetch recent measurements
  const measurementsUrl = `/api/v1/symptom-journal/measurements/recent?${new URLSearchParams({
    ...(selectedBodyArea && { body_area: selectedBodyArea }),
    days: "30"
  })}`;
  
  const { data: recentData, isLoading: loadingRecent } = useQuery<{ measurements: SymptomMeasurement[] }>({
    queryKey: [measurementsUrl],
  });

  // Fetch alerts
  const { data: alertsData } = useQuery<{ alerts: any[] }>({
    queryKey: ["/api/v1/symptom-journal/alerts?acknowledged=false"],
  });

  // Fetch trends
  const trendsUrl = selectedBodyArea 
    ? `/api/v1/symptom-journal/trends?${new URLSearchParams({ body_area: selectedBodyArea, days: "30" })}`
    : null;
    
  const { data: trendsData } = useQuery({
    queryKey: [trendsUrl || "/api/v1/symptom-journal/trends/disabled"],
    enabled: !!selectedBodyArea && !!trendsUrl,
  });

  // Fetch comparison data
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

  // Analyze respiratory rate mutation
  const analyzeRespiratoryRate = useMutation({
    mutationFn: async ({ videoFrames, duration }: { videoFrames: string[]; duration: number }) => {
      const res = await apiRequest("/api/v1/symptom-journal/analyze-respiratory", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          video_frames: videoFrames,
          duration_seconds: duration
        }),
      });
      return res.json();
    },
    onSuccess: (data) => {
      setRespiratoryResult(data);
      toast({
        title: "Respiratory Analysis Complete",
        description: data.respiratory_analysis?.estimated_bpm 
          ? `Estimated: ${data.respiratory_analysis.estimated_bpm} breaths/min`
          : "Analysis complete. Check results below.",
      });
    },
    onError: (error: any) => {
      toast({
        title: "Analysis Failed",
        description: error.message || "Failed to analyze respiratory rate",
        variant: "destructive",
      });
    },
  });

  // Upload symptom image mutation
  const uploadSymptomImage = useMutation({
    mutationFn: async ({ imageBlob, notes }: { imageBlob: Blob; notes: string }) => {
      const formData = new FormData();
      formData.append("file", imageBlob, `symptom-${selectedBodyArea}-${Date.now()}.jpg`);
      formData.append("body_area", selectedBodyArea);
      if (notes) formData.append("patient_notes", notes);

      const res = await apiRequest("/api/v1/symptom-journal/upload", {
        method: "POST",
        body: formData,
      });

      return res.json();
    },
    onSuccess: (data) => {
      toast({
        title: "Symptom Recorded",
        description: data.ai_observations ? 
          "AI analysis complete. Check the observations tab." : 
          "Symptom image uploaded successfully.",
      });

      // Show any alerts
      if (data.alerts && data.alerts.length > 0) {
        data.alerts.forEach((alert: any) => {
          toast({
            title: alert.title,
            description: alert.message,
            variant: alert.severity === "high" ? "destructive" : "default",
          });
        });
      }

      // Clear form
      setCapturedImage(null);
      setPatientNotes("");
      stopCamera();

      // Refresh data (invalidate all related queries)
      queryClient.invalidateQueries({ 
        predicate: (query) => 
          query.queryKey[0]?.toString().includes("/api/v1/symptom-journal")
      });

      // Switch to history tab
      setActiveTab("history");
    },
    onError: (error: any) => {
      toast({
        title: "Upload Failed",
        description: error.message || "Failed to upload symptom image",
        variant: "destructive",
      });
    },
  });

  // Start camera
  const startCamera = async () => {
    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, facingMode: "user" },
      });
      
      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream;
        videoRef.current.play();
      }
      
      streamRef.current = mediaStream; // Store in ref for cleanup
      setStream(mediaStream);
      setIsRecording(true);
    } catch (error) {
      console.error("Error accessing camera:", error);
      toast({
        title: "Camera Error",
        description: "Failed to access camera. Please check permissions.",
        variant: "destructive",
      });
    }
  };

  // Stop camera
  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => {
        track.stop();
        track.enabled = false;
      });
      streamRef.current = null;
    }
    if (stream) {
      stream.getTracks().forEach(track => {
        track.stop();
        track.enabled = false;
      });
    }
    setStream(null);
    setIsRecording(false);
    setCountdown(0);
  };

  // Capture photo
  const capturePhoto = () => {
    if (!videoRef.current || !canvasRef.current) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const context = canvas.getContext("2d");
    
    if (!context) return;

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    context.drawImage(video, 0, 0);

    canvas.toBlob((blob) => {
      if (blob) {
        const imageUrl = URL.createObjectURL(blob);
        setCapturedImage(imageUrl);
        stopCamera();
      }
    }, "image/jpeg", 0.95);
  };

  // Handle file upload
  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e) => {
      const result = e.target?.result as string;
      setCapturedImage(result);
    };
    reader.readAsDataURL(file);
  };

  // Submit symptom
  const handleSubmit = async () => {
    if (!capturedImage) {
      toast({
        title: "No Image",
        description: "Please capture or upload an image first",
        variant: "destructive",
      });
      return;
    }

    // Convert data URL to blob
    const response = await fetch(capturedImage);
    const blob = await response.blob();

    uploadSymptomImage.mutate({ imageBlob: blob, notes: patientNotes });
  };

  // Start video recording for respiratory analysis
  const startVideoRecording = async () => {
    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, facingMode: "user" },
        audio: false
      });
      
      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream;
        videoRef.current.play();
      }
      
      streamRef.current = mediaStream;
      setStream(mediaStream);
      
      // Create MediaRecorder
      recordedChunksRef.current = [];
      const recorder = new MediaRecorder(mediaStream, {
        mimeType: 'video/webm;codecs=vp8'
      });
      
      recorder.ondataavailable = (event) => {
        if (event.data && event.data.size > 0) {
          recordedChunksRef.current.push(event.data);
        }
      };
      
      recorder.onstop = () => {
        const blob = new Blob(recordedChunksRef.current, { type: 'video/webm' });
        setRecordedVideoBlob(blob);
        processVideoForRespiratoryAnalysis(blob);
      };
      
      mediaRecorderRef.current = recorder;
      recorder.start();
      setIsRecordingVideo(true);
      
      // Auto-stop after 10 seconds
      setTimeout(() => {
        if (mediaRecorderRef.current && mediaRecorderRef.current.state === "recording") {
          stopVideoRecording();
        }
      }, 10000);
      
    } catch (error) {
      console.error("Error accessing camera for video:", error);
      toast({
        title: "Camera Error",
        description: "Failed to access camera for video recording.",
        variant: "destructive",
      });
    }
  };

  // Stop video recording
  const stopVideoRecording = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === "recording") {
      mediaRecorderRef.current.stop();
    }
    setIsRecordingVideo(false);
    stopCamera();
  };

  // Extract frames from video and analyze
  const processVideoForRespiratoryAnalysis = async (videoBlob: Blob) => {
    try {
      const video = document.createElement('video');
      video.src = URL.createObjectURL(videoBlob);
      video.muted = true;
      
      await new Promise((resolve) => {
        video.onloadedmetadata = resolve;
      });
      
      const duration = video.duration;
      const framesToExtract = 10; // Extract 10 frames
      const frameInterval = duration / framesToExtract;
      const frames: string[] = [];
      
      const canvas = document.createElement('canvas');
      canvas.width = 320; // Smaller size for API efficiency
      canvas.height = 240;
      const ctx = canvas.getContext('2d');
      
      for (let i = 0; i < framesToExtract; i++) {
        video.currentTime = i * frameInterval;
        await new Promise((resolve) => {
          video.onseeked = resolve;
        });
        
        if (ctx) {
          ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
          const base64Frame = canvas.toDataURL('image/jpeg', 0.7).split(',')[1];
          frames.push(base64Frame);
        }
      }
      
      URL.revokeObjectURL(video.src);
      
      // Send to backend for analysis
      analyzeRespiratoryRate.mutate({
        videoFrames: frames,
        duration: duration
      });
      
    } catch (error) {
      console.error("Error processing video:", error);
      toast({
        title: "Processing Error",
        description: "Failed to process video for analysis",
        variant: "destructive",
      });
    }
  };

  // HIPAA Privacy: Cleanup camera stream on unmount
  useEffect(() => {
    return () => {
      // Use ref to access current stream value at unmount time
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => {
          track.stop();
          track.enabled = false;
        });
        streamRef.current = null;
      }
    };
  }, []);
  
  // Also cleanup when switching away from capture tab
  useEffect(() => {
    if (activeTab !== "capture" && stream) {
      stopCamera();
    }
  }, [activeTab, stream]);

  const selectedAreaIcon = BODY_AREAS.find(a => a.value === selectedBodyArea)?.icon || Activity;
  const SelectedIcon = selectedAreaIcon;

  return (
    <div className="container mx-auto p-6 max-w-6xl">
      {/* Header */}
      <div className="mb-6">
        <h1 className="text-3xl font-bold mb-2" data-testid="text-page-title">Symptom Journal</h1>
        <p className="text-muted-foreground">
          Track visual changes in different body areas. This is for monitoring only - not for medical diagnosis.
        </p>
      </div>

      {/* Safety Disclaimer */}
      <Alert className="mb-6">
        <Info className="h-4 w-4" />
        <AlertDescription>
          <strong>Important:</strong> This tool monitors changes over time and does not provide medical diagnoses. 
          If you notice concerning changes or experience severe symptoms, contact your healthcare provider immediately.
        </AlertDescription>
      </Alert>

      {/* Unacknowledged Alerts */}
      {alertsData && alertsData.alerts.length > 0 && (
        <Alert variant="destructive" className="mb-6">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>
            You have {alertsData.alerts.length} unacknowledged alert(s). Check the Alerts tab for details.
          </AlertDescription>
        </Alert>
      )}

      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="grid w-full grid-cols-4" data-testid="tabs-navigation">
          <TabsTrigger value="capture" data-testid="tab-capture">Capture</TabsTrigger>
          <TabsTrigger value="history" data-testid="tab-history">History</TabsTrigger>
          <TabsTrigger value="trends" data-testid="tab-trends">Trends</TabsTrigger>
          <TabsTrigger value="alerts" data-testid="tab-alerts">
            Alerts {alertsData && alertsData.alerts.length > 0 && (
              <Badge variant="destructive" className="ml-2">{alertsData.alerts.length}</Badge>
            )}
          </TabsTrigger>
        </TabsList>

        {/* Capture Tab */}
        <TabsContent value="capture" className="mt-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <SelectedIcon className="h-5 w-5" />
                Record Symptom
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              {/* Body Area Selection */}
              <div>
                <Label htmlFor="body-area">Select Body Area</Label>
                <Select value={selectedBodyArea} onValueChange={setSelectedBodyArea}>
                  <SelectTrigger id="body-area" data-testid="select-body-area">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {BODY_AREAS.map((area) => (
                      <SelectItem key={area.value} value={area.value} data-testid={`option-${area.value}`}>
                        <div className="flex items-center gap-2">
                          <area.icon className="h-4 w-4" />
                          {area.label}
                        </div>
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              {/* Respiratory Rate Analysis (Chest area only) */}
              {selectedBodyArea === "chest" && (
                <Card className="bg-muted/50">
                  <CardContent className="pt-6 space-y-4">
                    <div className="flex items-center gap-2 text-sm font-medium">
                      <Heart className="h-4 w-4" />
                      Respiratory Rate Analysis
                    </div>
                    
                    {!recordedVideoBlob && !respiratoryResult ? (
                      <div>
                        <p className="text-sm text-muted-foreground mb-3">
                          Record a 10-second video of your chest breathing normally. AI will estimate your respiratory rate.
                        </p>
                        
                        {!isRecordingVideo ? (
                          <Button 
                            onClick={startVideoRecording} 
                            className="w-full"
                            data-testid="button-start-respiratory-recording"
                          >
                            <Camera className="mr-2 h-4 w-4" />
                            Start Respiratory Recording (10s)
                          </Button>
                        ) : (
                          <div className="space-y-3">
                            <video
                              ref={videoRef}
                              className="w-full rounded-lg border"
                              autoPlay
                              playsInline
                              muted
                              data-testid="video-respiratory-preview"
                            />
                            <div className="flex gap-2">
                              <Button 
                                onClick={stopVideoRecording}
                                variant="destructive"
                                className="flex-1"
                                data-testid="button-stop-respiratory-recording"
                              >
                                Stop Recording
                              </Button>
                            </div>
                            <Alert>
                              <Info className="h-4 w-4" />
                              <AlertDescription>
                                Recording will auto-stop after 10 seconds. Breathe normally.
                              </AlertDescription>
                            </Alert>
                          </div>
                        )}
                      </div>
                    ) : respiratoryResult ? (
                      <div className="space-y-4">
                        <Alert className={
                          respiratoryResult.pattern_assessment?.rate_category === "within_typical_range" 
                            ? "bg-green-50 border-green-200" 
                            : "bg-yellow-50 border-yellow-200"
                        }>
                          <Heart className="h-4 w-4" />
                          <AlertDescription>
                            <div className="space-y-2">
                              <div className="text-lg font-semibold">
                                {respiratoryResult.respiratory_analysis?.estimated_bpm || "N/A"} breaths/min
                              </div>
                              <div className="text-sm">
                                {respiratoryResult.respiratory_analysis?.observations}
                              </div>
                              {respiratoryResult.pattern_assessment && (
                                <div className="text-sm mt-2">
                                  <strong>{respiratoryResult.pattern_assessment.observation}</strong>
                                  <br />
                                  {respiratoryResult.pattern_assessment.monitoring_note}
                                </div>
                              )}
                            </div>
                          </AlertDescription>
                        </Alert>
                        
                        {respiratoryResult.trend_analysis && (
                          <div className="text-sm space-y-1">
                            <div className="font-medium">Trend Analysis:</div>
                            <div>{respiratoryResult.trend_analysis.observation}</div>
                          </div>
                        )}
                        
                        <Button
                          variant="outline"
                          onClick={() => {
                            setRecordedVideoBlob(null);
                            setRespiratoryResult(null);
                          }}
                          className="w-full"
                          data-testid="button-record-again"
                        >
                          Record Again
                        </Button>
                      </div>
                    ) : analyzeRespiratoryRate.isPending && (
                      <div className="text-center py-4">
                        <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-primary" data-testid="spinner-analyzing"></div>
                        <p className="mt-2 text-sm text-muted-foreground">Analyzing respiratory rate...</p>
                      </div>
                    )}
                  </CardContent>
                </Card>
              )}

              {/* Camera/Upload Section */}
              {!capturedImage ? (
                <div className="space-y-4">
                  {/* Video Preview */}
                  {isRecording ? (
                    <div className="relative">
                      <video
                        ref={videoRef}
                        className="w-full rounded-lg border"
                        autoPlay
                        playsInline
                        muted
                        data-testid="video-preview"
                      />
                      <canvas ref={canvasRef} className="hidden" />
                      
                      <div className="mt-4 flex gap-2 justify-center">
                        <Button onClick={capturePhoto} data-testid="button-capture-photo">
                          <Camera className="mr-2 h-4 w-4" />
                          Capture Photo
                        </Button>
                        <Button variant="outline" onClick={stopCamera} data-testid="button-stop-camera">
                          Stop Camera
                        </Button>
                      </div>
                    </div>
                  ) : (
                    <div className="space-y-2">
                      <Button onClick={startCamera} className="w-full" data-testid="button-start-camera">
                        <Camera className="mr-2 h-4 w-4" />
                        Start Camera
                      </Button>
                      
                      <div className="text-center text-muted-foreground">or</div>
                      
                      <div>
                        <input
                          ref={fileInputRef}
                          type="file"
                          accept="image/*"
                          onChange={handleFileUpload}
                          className="hidden"
                          data-testid="input-file-upload"
                        />
                        <Button
                          variant="outline"
                          onClick={() => fileInputRef.current?.click()}
                          className="w-full"
                          data-testid="button-upload-image"
                        >
                          Upload Image
                        </Button>
                      </div>
                    </div>
                  )}
                </div>
              ) : (
                <div className="space-y-4">
                  {/* Preview */}
                  <div>
                    <img 
                      src={capturedImage} 
                      alt="Captured symptom" 
                      className="w-full rounded-lg border"
                      data-testid="img-captured-preview"
                    />
                  </div>

                  {/* Notes */}
                  <div>
                    <Label htmlFor="notes">Notes (Optional)</Label>
                    <Textarea
                      id="notes"
                      placeholder="Describe what you're experiencing (e.g., 'Legs feel more swollen today', 'Face looks more flushed')"
                      value={patientNotes}
                      onChange={(e) => setPatientNotes(e.target.value)}
                      rows={4}
                      data-testid="textarea-notes"
                    />
                  </div>

                  {/* Action Buttons */}
                  <div className="flex gap-2">
                    <Button
                      onClick={handleSubmit}
                      disabled={uploadSymptomImage.isPending}
                      className="flex-1"
                      data-testid="button-submit-symptom"
                    >
                      {uploadSymptomImage.isPending ? "Analyzing..." : "Submit Symptom"}
                    </Button>
                    <Button
                      variant="outline"
                      onClick={() => {
                        setCapturedImage(null);
                        setPatientNotes("");
                      }}
                      data-testid="button-retake"
                    >
                      Retake
                    </Button>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* History Tab */}
        <TabsContent value="history" className="mt-6">
          {/* Comparison View Section */}
          {showComparison && recentData && recentData.measurements.length >= 2 && (
            <Card className="mb-6">
              <CardHeader>
                <div className="flex items-center justify-between">
                  <CardTitle className="flex items-center gap-2">
                    <GitCompare className="h-5 w-5" />
                    Compare Measurements
                  </CardTitle>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => {
                      setShowComparison(false);
                      setSelectedMeasurement1(null);
                      setSelectedMeasurement2(null);
                    }}
                    data-testid="button-close-comparison"
                  >
                    Close
                  </Button>
                </div>
              </CardHeader>
              <CardContent className="space-y-6">
                {/* Measurement Selection */}
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <Label htmlFor="measurement-1">First Measurement</Label>
                    <Select 
                      value={selectedMeasurement1?.toString() || ""} 
                      onValueChange={(val) => setSelectedMeasurement1(parseInt(val))}
                    >
                      <SelectTrigger id="measurement-1" data-testid="select-measurement-1">
                        <SelectValue placeholder="Select measurement" />
                      </SelectTrigger>
                      <SelectContent>
                        {recentData.measurements.map((m) => (
                          <SelectItem 
                            key={m.id} 
                            value={m.id.toString()}
                            disabled={m.id === selectedMeasurement2}
                            data-testid={`option-m1-${m.id}`}
                          >
                            {format(new Date(m.created_at), "MMM d, yyyy h:mm a")}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>

                  <div>
                    <Label htmlFor="measurement-2">Second Measurement</Label>
                    <Select 
                      value={selectedMeasurement2?.toString() || ""} 
                      onValueChange={(val) => setSelectedMeasurement2(parseInt(val))}
                    >
                      <SelectTrigger id="measurement-2" data-testid="select-measurement-2">
                        <SelectValue placeholder="Select measurement" />
                      </SelectTrigger>
                      <SelectContent>
                        {recentData.measurements.map((m) => (
                          <SelectItem 
                            key={m.id} 
                            value={m.id.toString()}
                            disabled={m.id === selectedMeasurement1}
                            data-testid={`option-m2-${m.id}`}
                          >
                            {format(new Date(m.created_at), "MMM d, yyyy h:mm a")}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                </div>

                {/* Comparison Results */}
                {comparisonLoading && (
                  <div className="text-center py-8" data-testid="spinner-comparison-loading">
                    <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
                    <p className="mt-2 text-sm text-muted-foreground">Loading comparison...</p>
                  </div>
                )}

                {selectedMeasurement1 && selectedMeasurement2 && !comparisonData && !comparisonLoading && (
                  <Alert variant="destructive" data-testid="alert-comparison-error">
                    <AlertCircle className="h-4 w-4" />
                    <AlertDescription>
                      Failed to load comparison data. Please try again or select different measurements.
                    </AlertDescription>
                  </Alert>
                )}

                {comparisonData && !comparisonLoading && comparisonData.comparison && (
                  <div className="space-y-6">
                    {/* Side-by-Side Images */}
                    <div className="grid grid-cols-2 gap-4">
                      <Card>
                        <CardContent className="p-4">
                          <div className="text-sm text-muted-foreground mb-2" data-testid="text-date-m1">
                            {format(new Date(comparisonData.comparison.measurement_1.date), "MMM d, yyyy h:mm a")}
                          </div>
                          {comparisonData.comparison.measurement_1.image_url && (
                            <img
                              src={comparisonData.comparison.measurement_1.image_url}
                              alt="Measurement 1"
                              className="w-full rounded border"
                              data-testid="img-comparison-1"
                            />
                          )}
                          {comparisonData.comparison.measurement_1.ai_observations && (
                            <div className="mt-3 text-sm" data-testid="text-observations-m1">
                              <strong>AI Notes:</strong> {comparisonData.comparison.measurement_1.ai_observations}
                            </div>
                          )}
                        </CardContent>
                      </Card>

                      <Card>
                        <CardContent className="p-4">
                          <div className="text-sm text-muted-foreground mb-2" data-testid="text-date-m2">
                            {format(new Date(comparisonData.comparison.measurement_2.date), "MMM d, yyyy h:mm a")}
                          </div>
                          {comparisonData.comparison.measurement_2.image_url && (
                            <img
                              src={comparisonData.comparison.measurement_2.image_url}
                              alt="Measurement 2"
                              className="w-full rounded border"
                              data-testid="img-comparison-2"
                            />
                          )}
                          {comparisonData.comparison.measurement_2.ai_observations && (
                            <div className="mt-3 text-sm" data-testid="text-observations-m2">
                              <strong>AI Notes:</strong> {comparisonData.comparison.measurement_2.ai_observations}
                            </div>
                          )}
                        </CardContent>
                      </Card>
                    </div>

                    {/* Change Metrics */}
                    <Alert data-testid="alert-change-summary">
                      <Activity className="h-4 w-4" />
                      <AlertDescription>
                        <div className="space-y-2">
                          <div className="font-semibold">
                            Changes over {comparisonData.comparison.changes.days_between} days
                          </div>
                          
                          <div className="grid grid-cols-2 gap-4 mt-2">
                            {comparisonData.comparison.changes.color_change !== null && (
                              <div className="flex items-center gap-2" data-testid="text-color-change">
                                <strong>Color:</strong>
                                <span className={
                                  Math.abs(comparisonData.comparison.changes.color_change) > 10 
                                    ? "text-orange-600 font-semibold" 
                                    : ""
                                }>
                                  {comparisonData.comparison.changes.color_change > 0 ? "+" : ""}
                                  {comparisonData.comparison.changes.color_change.toFixed(1)}%
                                </span>
                              </div>
                            )}
                            
                            {comparisonData.comparison.changes.area_change !== null && (
                              <div className="flex items-center gap-2" data-testid="text-area-change">
                                <strong>Area:</strong>
                                <span className={
                                  Math.abs(comparisonData.comparison.changes.area_change) > 15 
                                    ? "text-orange-600 font-semibold" 
                                    : ""
                                }>
                                  {comparisonData.comparison.changes.area_change > 0 ? "+" : ""}
                                  {comparisonData.comparison.changes.area_change.toFixed(1)}%
                                </span>
                              </div>
                            )}
                            
                            {comparisonData.comparison.changes.respiratory_rate_change !== null && (
                              <div className="flex items-center gap-2" data-testid="text-respiratory-change">
                                <strong>Respiratory Rate:</strong>
                                <span>
                                  {comparisonData.comparison.changes.respiratory_rate_change > 0 ? "+" : ""}
                                  {comparisonData.comparison.changes.respiratory_rate_change} breaths/min
                                </span>
                              </div>
                            )}
                          </div>
                        </div>
                      </AlertDescription>
                    </Alert>
                  </div>
                )}
              </CardContent>
            </Card>
          )}

          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle>Recent Measurements</CardTitle>
                {recentData && recentData.measurements.length >= 2 && !showComparison && (
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setShowComparison(true)}
                    data-testid="button-show-comparison"
                  >
                    <GitCompare className="mr-2 h-4 w-4" />
                    Compare
                  </Button>
                )}
              </div>
            </CardHeader>
            <CardContent>
              {loadingRecent ? (
                <p className="text-muted-foreground" data-testid="text-loading">Loading...</p>
              ) : !recentData || recentData.measurements.length === 0 ? (
                <p className="text-muted-foreground" data-testid="text-no-measurements">
                  No measurements recorded yet. Start by capturing your first symptom.
                </p>
              ) : (
                <div className="space-y-4">
                  {recentData.measurements.map((measurement) => (
                    <Card key={measurement.id} data-testid={`card-measurement-${measurement.id}`}>
                      <CardContent className="p-4">
                        <div className="flex gap-4">
                          {/* Image */}
                          {measurement.image_url && (
                            <div className="flex-shrink-0">
                              <img
                                src={measurement.image_url}
                                alt={`${measurement.body_area} symptom`}
                                className="w-32 h-32 object-cover rounded border"
                                data-testid={`img-measurement-${measurement.id}`}
                              />
                            </div>
                          )}

                          {/* Details */}
                          <div className="flex-1 space-y-2">
                            <div className="flex items-center justify-between">
                              <div className="flex items-center gap-2">
                                <Badge data-testid={`badge-body-area-${measurement.id}`}>
                                  {measurement.body_area}
                                </Badge>
                                <span className="text-sm text-muted-foreground" data-testid={`text-date-${measurement.id}`}>
                                  {format(new Date(measurement.created_at), "MMM d, yyyy 'at' h:mm a")}
                                </span>
                              </div>
                            </div>

                            {/* AI Observations */}
                            {measurement.ai_observations && (
                              <div className="text-sm" data-testid={`text-ai-observations-${measurement.id}`}>
                                <strong>AI Observations:</strong> {measurement.ai_observations}
                              </div>
                            )}

                            {/* Changes */}
                            <div className="flex gap-4 text-sm">
                              {measurement.color_change_percent !== null && (
                                <div className="flex items-center gap-1" data-testid={`text-color-change-${measurement.id}`}>
                                  <strong>Color:</strong> 
                                  {measurement.color_change_percent > 0 ? (
                                    <TrendingUp className="h-3 w-3 text-orange-500" />
                                  ) : measurement.color_change_percent < 0 ? (
                                    <TrendingDown className="h-3 w-3 text-blue-500" />
                                  ) : (
                                    <Minus className="h-3 w-3" />
                                  )}
                                  {Math.abs(measurement.color_change_percent).toFixed(1)}%
                                </div>
                              )}
                              
                              {measurement.area_change_percent !== null && (
                                <div className="flex items-center gap-1" data-testid={`text-area-change-${measurement.id}`}>
                                  <strong>Area:</strong> 
                                  {measurement.area_change_percent > 0 ? (
                                    <TrendingUp className="h-3 w-3 text-orange-500" />
                                  ) : measurement.area_change_percent < 0 ? (
                                    <TrendingDown className="h-3 w-3 text-blue-500" />
                                  ) : (
                                    <Minus className="h-3 w-3" />
                                  )}
                                  {Math.abs(measurement.area_change_percent).toFixed(1)}%
                                </div>
                              )}

                              {measurement.respiratory_rate_bpm && (
                                <div data-testid={`text-respiratory-rate-${measurement.id}`}>
                                  <strong>Breathing:</strong> {measurement.respiratory_rate_bpm} breaths/min
                                </div>
                              )}
                            </div>

                            {/* Alerts */}
                            {measurement.alerts && measurement.alerts.length > 0 && (
                              <div className="space-y-1">
                                {measurement.alerts.map((alert, idx) => (
                                  <Alert 
                                    key={idx} 
                                    variant={alert.severity === "high" ? "destructive" : "default"}
                                    data-testid={`alert-${measurement.id}-${idx}`}
                                  >
                                    <AlertCircle className="h-4 w-4" />
                                    <AlertDescription>
                                      <strong>{alert.title}:</strong> {alert.message}
                                    </AlertDescription>
                                  </Alert>
                                ))}
                              </div>
                            )}
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Trends Tab */}
        <TabsContent value="trends" className="mt-6">
          <Card>
            <CardHeader>
              <CardTitle>Trends for {selectedBodyArea}</CardTitle>
            </CardHeader>
            <CardContent>
              {trendsData && trendsData.trend_data && trendsData.trend_data.length > 0 ? (
                <div className="space-y-4">
                  <p className="text-sm text-muted-foreground">
                    {trendsData.measurement_count} measurements over the last 30 days
                  </p>
                  
                  <div className="text-sm text-muted-foreground">
                    Detailed trend visualization coming soon. For now, view individual measurements in the History tab.
                  </div>
                </div>
              ) : (
                <p className="text-muted-foreground" data-testid="text-no-trends">
                  No trend data available for {selectedBodyArea}. Record more measurements to see trends.
                </p>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Alerts Tab */}
        <TabsContent value="alerts" className="mt-6">
          <Card>
            <CardHeader>
              <CardTitle>Active Alerts</CardTitle>
            </CardHeader>
            <CardContent>
              {!alertsData || alertsData.alerts.length === 0 ? (
                <p className="text-muted-foreground" data-testid="text-no-alerts">
                  No active alerts. We'll notify you if we detect significant changes.
                </p>
              ) : (
                <div className="space-y-3">
                  {alertsData.alerts.map((alert) => (
                    <Alert 
                      key={alert.id} 
                      variant={alert.severity === "high" ? "destructive" : "default"}
                      data-testid={`alert-card-${alert.id}`}
                    >
                      <AlertCircle className="h-4 w-4" />
                      <AlertDescription>
                        <div className="space-y-2">
                          <div>
                            <strong>{alert.title}</strong>
                            <p className="text-sm mt-1">{alert.message}</p>
                          </div>
                          <div className="flex items-center gap-2 text-xs text-muted-foreground">
                            <Badge variant="outline" data-testid={`badge-body-area-${alert.id}`}>
                              {alert.body_area}
                            </Badge>
                            <span data-testid={`text-alert-date-${alert.id}`}>
                              {format(new Date(alert.created_at), "MMM d, yyyy")}
                            </span>
                          </div>
                          <Button
                            size="sm"
                            variant="outline"
                            onClick={async () => {
                              try {
                                await apiRequest(`/api/v1/symptom-journal/alerts/${alert.id}/acknowledge`, {
                                  method: "POST",
                                });
                                queryClient.invalidateQueries({ queryKey: ["/api/v1/symptom-journal/alerts"] });
                                toast({
                                  title: "Alert Acknowledged",
                                  description: "This alert has been marked as read.",
                                });
                              } catch (error) {
                                toast({
                                  title: "Error",
                                  description: "Failed to acknowledge alert",
                                  variant: "destructive",
                                });
                              }
                            }}
                            data-testid={`button-acknowledge-${alert.id}`}
                          >
                            Acknowledge
                          </Button>
                        </div>
                      </AlertDescription>
                    </Alert>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
