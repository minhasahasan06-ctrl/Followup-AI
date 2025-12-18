import { useState, useRef, useCallback, useEffect } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { apiRequest, queryClient } from "@/lib/queryClient";
import { Card, CardContent, CardHeader, CardTitle, CardFooter } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { useToast } from "@/hooks/use-toast";
import {
  Camera,
  ChevronLeft,
  ChevronRight,
  Check,
  X,
  AlertCircle,
  Eye,
  Hand,
  CircleDot,
  Smile,
  Activity,
  Upload,
  Loader2,
  RefreshCw,
  Sparkles,
  ImageIcon,
  CheckCircle2,
  Zap
} from "lucide-react";

interface ExamStage {
  id: string;
  name: string;
  icon: React.ReactNode;
  instructions: string[];
  examType: string;
}

const EXAM_STAGES: ExamStage[] = [
  {
    id: "eyes",
    name: "Eye Examination",
    icon: <Eye className="h-5 w-5" />,
    instructions: [
      "Look directly at the camera",
      "Keep eyes open wide",
      "Ensure good lighting on your face",
      "Remove glasses if wearing"
    ],
    examType: "eye"
  },
  {
    id: "palm",
    name: "Palm Examination",
    icon: <Hand className="h-5 w-5" />,
    instructions: [
      "Hold palm flat toward camera",
      "Spread fingers slightly apart",
      "Ensure palm is well-lit",
      "Keep hand steady for capture"
    ],
    examType: "palm"
  },
  {
    id: "tongue",
    name: "Tongue Examination",
    icon: <CircleDot className="h-5 w-5" />,
    instructions: [
      "Open mouth wide and stick out tongue",
      "Ensure adequate lighting",
      "Keep tongue extended for 3 seconds",
      "Breathe normally"
    ],
    examType: "tongue"
  },
  {
    id: "lips",
    name: "Lip Examination",
    icon: <Smile className="h-5 w-5" />,
    instructions: [
      "Close mouth naturally",
      "Face camera directly",
      "Ensure lips are visible",
      "Good lighting on face"
    ],
    examType: "lips"
  }
];

interface ExamGuidanceWizardProps {
  patientId: string;
  onComplete?: (sessionId: string) => void;
  onCancel?: () => void;
}

interface SessionData {
  session_id: string;
  patient_id: string;
  status: string;
  created_at: string;
}

interface StageCapture {
  stage: string;
  imageData: string | null;
  qualityScore: number | null;
  isAcceptable: boolean;
  uploaded: boolean;
  s3Key: string | null;
}

export function ExamGuidanceWizard({ patientId, onComplete, onCancel }: ExamGuidanceWizardProps) {
  const { toast } = useToast();
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  
  const [currentStageIndex, setCurrentStageIndex] = useState(0);
  const [session, setSession] = useState<SessionData | null>(null);
  const [cameraActive, setCameraActive] = useState(false);
  const [captures, setCaptures] = useState<Record<string, StageCapture>>({});
  const [isCapturing, setIsCapturing] = useState(false);
  const [isCheckingQuality, setIsCheckingQuality] = useState(false);

  const currentStage = EXAM_STAGES[currentStageIndex];
  const progress = ((Object.values(captures).filter(c => c.uploaded).length) / EXAM_STAGES.length) * 100;

  const createSessionMutation = useMutation({
    mutationFn: async () => {
      const response = await apiRequest("POST", "/api/video/exam-sessions", {
        patient_id: patientId
      });
      return response.json();
    },
    onSuccess: (data: SessionData) => {
      setSession(data);
      toast({
        title: "Exam Session Started",
        description: "Camera-guided examination is ready to begin.",
      });
    },
    onError: (error: Error) => {
      toast({
        title: "Failed to Start Session",
        description: error.message,
        variant: "destructive"
      });
    }
  });

  const uploadStageMutation = useMutation({
    mutationFn: async ({ stage, imageData }: { stage: string; imageData: string }) => {
      if (!session) throw new Error("No active session");
      
      const uploadUrlRes = await apiRequest("POST", `/api/video/exam-sessions/${session.session_id}/upload-url`, {
        stage,
        content_type: "image/jpeg",
        file_extension: "jpg"
      });
      const uploadData = await uploadUrlRes.json();
      
      const base64Data = imageData.split(",")[1] || imageData;
      const binaryData = atob(base64Data);
      const bytes = new Uint8Array(binaryData.length);
      for (let i = 0; i < binaryData.length; i++) {
        bytes[i] = binaryData.charCodeAt(i);
      }
      const blob = new Blob([bytes], { type: "image/jpeg" });
      
      if (uploadData.upload_url && uploadData.upload_url !== "local_fallback") {
        await fetch(uploadData.upload_url, {
          method: "PUT",
          body: blob,
          headers: { "Content-Type": "image/jpeg" }
        });
      }
      
      await apiRequest("POST", `/api/video/exam-sessions/${session.session_id}/complete-stage`, {
        stage,
        s3_key: uploadData.s3_key,
        quality_score: captures[stage]?.qualityScore || 70
      });
      
      return { stage, s3_key: uploadData.s3_key };
    },
    onSuccess: ({ stage, s3_key }) => {
      setCaptures(prev => ({
        ...prev,
        [stage]: { ...prev[stage], uploaded: true, s3Key: s3_key }
      }));
      toast({
        title: "Image Uploaded",
        description: `${EXAM_STAGES.find(s => s.id === stage)?.name || stage} capture saved.`
      });
    },
    onError: (error: Error) => {
      toast({
        title: "Upload Failed",
        description: error.message,
        variant: "destructive"
      });
    }
  });

  const completeSessionMutation = useMutation({
    mutationFn: async () => {
      if (!session) throw new Error("No active session");
      const response = await apiRequest("POST", `/api/video/exam-sessions/${session.session_id}/complete`);
      return response.json();
    },
    onSuccess: (data) => {
      toast({
        title: "Exam Completed",
        description: `${data.completed_stages}/${data.total_stages} stages completed successfully.`
      });
      onComplete?.(session!.session_id);
    },
    onError: (error: Error) => {
      toast({
        title: "Failed to Complete Session",
        description: error.message,
        variant: "destructive"
      });
    }
  });

  const checkQualityMutation = useMutation({
    mutationFn: async (imageData: string) => {
      const base64Data = imageData.split(",")[1] || imageData;
      const response = await apiRequest("POST", "/api/video/quality-check", {
        image_data: base64Data
      });
      return response.json();
    }
  });

  const startCamera = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: "user",
          width: { ideal: 1280 },
          height: { ideal: 720 }
        }
      });
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
        setCameraActive(true);
      }
    } catch (err) {
      toast({
        title: "Camera Access Denied",
        description: "Please allow camera access to continue with the examination.",
        variant: "destructive"
      });
    }
  }, [toast]);

  const stopCamera = useCallback(() => {
    if (videoRef.current?.srcObject) {
      const tracks = (videoRef.current.srcObject as MediaStream).getTracks();
      tracks.forEach(track => track.stop());
      videoRef.current.srcObject = null;
      setCameraActive(false);
    }
  }, []);

  const captureImage = useCallback(async () => {
    if (!videoRef.current || !canvasRef.current || !currentStage) return;
    
    setIsCapturing(true);
    
    const video = videoRef.current;
    const canvas = canvasRef.current;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    
    const ctx = canvas.getContext("2d");
    if (!ctx) {
      setIsCapturing(false);
      return;
    }
    
    ctx.drawImage(video, 0, 0);
    const imageData = canvas.toDataURL("image/jpeg", 0.9);
    
    setIsCheckingQuality(true);
    
    try {
      const qualityResult = await checkQualityMutation.mutateAsync(imageData);
      
      setCaptures(prev => ({
        ...prev,
        [currentStage.id]: {
          stage: currentStage.id,
          imageData,
          qualityScore: qualityResult.quality_score,
          isAcceptable: qualityResult.is_acceptable,
          uploaded: false,
          s3Key: null
        }
      }));
      
      if (!qualityResult.is_acceptable) {
        toast({
          title: "Image Quality Issue",
          description: qualityResult.recommendations?.join(". ") || "Please retake the image with better lighting.",
          variant: "destructive"
        });
      }
    } catch (error) {
      setCaptures(prev => ({
        ...prev,
        [currentStage.id]: {
          stage: currentStage.id,
          imageData,
          qualityScore: 70,
          isAcceptable: true,
          uploaded: false,
          s3Key: null
        }
      }));
    }
    
    setIsCheckingQuality(false);
    setIsCapturing(false);
  }, [currentStage, checkQualityMutation, toast]);

  const uploadCurrentCapture = useCallback(async () => {
    const capture = captures[currentStage.id];
    if (!capture?.imageData) return;
    
    await uploadStageMutation.mutateAsync({
      stage: currentStage.id,
      imageData: capture.imageData
    });
  }, [captures, currentStage, uploadStageMutation]);

  const goToNextStage = useCallback(() => {
    if (currentStageIndex < EXAM_STAGES.length - 1) {
      setCurrentStageIndex(prev => prev + 1);
    }
  }, [currentStageIndex]);

  const goToPreviousStage = useCallback(() => {
    if (currentStageIndex > 0) {
      setCurrentStageIndex(prev => prev - 1);
    }
  }, [currentStageIndex]);

  const handleComplete = useCallback(async () => {
    await completeSessionMutation.mutateAsync();
  }, [completeSessionMutation]);

  useEffect(() => {
    if (!session) {
      createSessionMutation.mutate();
    }
  }, []);

  useEffect(() => {
    return () => {
      stopCamera();
    };
  }, [stopCamera]);

  const currentCapture = captures[currentStage?.id];
  const allStagesComplete = EXAM_STAGES.every(stage => captures[stage.id]?.uploaded);

  return (
    <Card className="w-full max-w-4xl mx-auto">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between gap-2">
          <div className="flex items-center gap-2">
            <Activity className="h-5 w-5 text-primary" />
            <CardTitle className="text-lg">Health Care Exam Capture</CardTitle>
          </div>
          <Button variant="ghost" size="icon" onClick={onCancel} data-testid="button-cancel-exam">
            <X className="h-4 w-4" />
          </Button>
        </div>
        <Progress value={progress} className="h-2 mt-3" data-testid="progress-exam" />
        <div className="flex justify-between text-xs text-muted-foreground mt-1">
          <span>Stage {currentStageIndex + 1} of {EXAM_STAGES.length}</span>
          <span>{Math.round(progress)}% Complete</span>
        </div>
      </CardHeader>

      <CardContent>
        <div className="flex gap-2 mb-4 overflow-x-auto pb-2">
          {EXAM_STAGES.map((stage, index) => {
            const stageCapture = captures[stage.id];
            const isComplete = stageCapture?.uploaded;
            const isCurrent = index === currentStageIndex;
            
            return (
              <Button
                key={stage.id}
                variant={isCurrent ? "default" : isComplete ? "secondary" : "outline"}
                size="sm"
                className="flex items-center gap-2 min-w-fit"
                onClick={() => setCurrentStageIndex(index)}
                data-testid={`button-stage-${stage.id}`}
              >
                {isComplete ? (
                  <CheckCircle2 className="h-4 w-4 text-green-500" />
                ) : (
                  stage.icon
                )}
                <span className="hidden sm:inline">{stage.name}</span>
              </Button>
            );
          })}
        </div>

        <div className="grid md:grid-cols-2 gap-4">
          <div className="space-y-4">
            <Card>
              <CardHeader className="py-3">
                <div className="flex items-center gap-2">
                  {currentStage.icon}
                  <CardTitle className="text-base">{currentStage.name}</CardTitle>
                </div>
              </CardHeader>
              <CardContent className="py-0 pb-4">
                <ul className="space-y-2">
                  {currentStage.instructions.map((instruction, i) => (
                    <li key={i} className="flex items-start gap-2 text-sm text-muted-foreground">
                      <Check className="h-4 w-4 text-primary mt-0.5 shrink-0" />
                      {instruction}
                    </li>
                  ))}
                </ul>
              </CardContent>
            </Card>

            {currentCapture?.imageData && (
              <Card>
                <CardHeader className="py-3">
                  <div className="flex items-center justify-between">
                    <CardTitle className="text-base">Captured Image</CardTitle>
                    {currentCapture.qualityScore && (
                      <Badge variant={currentCapture.isAcceptable ? "default" : "destructive"}>
                        Quality: {Math.round(currentCapture.qualityScore)}%
                      </Badge>
                    )}
                  </div>
                </CardHeader>
                <CardContent className="py-0 pb-4">
                  <img
                    src={currentCapture.imageData}
                    alt="Captured"
                    className="w-full rounded-md border"
                    data-testid="img-captured"
                  />
                  <div className="flex gap-2 mt-3">
                    <Button
                      variant="outline"
                      size="sm"
                      className="flex-1"
                      onClick={() => setCaptures(prev => {
                        const newCaptures = { ...prev };
                        delete newCaptures[currentStage.id];
                        return newCaptures;
                      })}
                      data-testid="button-retake"
                    >
                      <RefreshCw className="h-4 w-4 mr-1" />
                      Retake
                    </Button>
                    {!currentCapture.uploaded && (
                      <Button
                        size="sm"
                        className="flex-1"
                        onClick={uploadCurrentCapture}
                        disabled={uploadStageMutation.isPending}
                        data-testid="button-upload-stage"
                      >
                        {uploadStageMutation.isPending ? (
                          <Loader2 className="h-4 w-4 mr-1 animate-spin" />
                        ) : (
                          <Upload className="h-4 w-4 mr-1" />
                        )}
                        Save
                      </Button>
                    )}
                    {currentCapture.uploaded && (
                      <Badge variant="secondary" className="flex items-center gap-1">
                        <CheckCircle2 className="h-3 w-3" />
                        Saved
                      </Badge>
                    )}
                  </div>
                </CardContent>
              </Card>
            )}
          </div>

          <div className="space-y-4">
            <Card className="relative overflow-hidden">
              <CardContent className="p-0">
                {!cameraActive ? (
                  <div className="aspect-video bg-muted flex flex-col items-center justify-center gap-4">
                    <Camera className="h-12 w-12 text-muted-foreground" />
                    <Button onClick={startCamera} data-testid="button-start-camera">
                      <Camera className="h-4 w-4 mr-2" />
                      Start Camera
                    </Button>
                  </div>
                ) : (
                  <div className="relative">
                    <video
                      ref={videoRef}
                      className="w-full aspect-video object-cover"
                      autoPlay
                      playsInline
                      muted
                      data-testid="video-camera-feed"
                    />
                    <canvas ref={canvasRef} className="hidden" />
                    
                    <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                      {currentStage.id === "eyes" && (
                        <div className="w-48 h-32 border-2 border-dashed border-white/50 rounded-full" />
                      )}
                      {currentStage.id === "palm" && (
                        <div className="w-40 h-48 border-2 border-dashed border-white/50 rounded-lg" />
                      )}
                      {currentStage.id === "tongue" && (
                        <div className="w-32 h-24 border-2 border-dashed border-white/50 rounded-full" />
                      )}
                      {currentStage.id === "lips" && (
                        <div className="w-36 h-20 border-2 border-dashed border-white/50 rounded-full" />
                      )}
                    </div>

                    {isCheckingQuality && (
                      <div className="absolute inset-0 bg-black/50 flex items-center justify-center">
                        <div className="flex items-center gap-2 text-white">
                          <Loader2 className="h-5 w-5 animate-spin" />
                          Checking quality...
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </CardContent>
            </Card>

            {cameraActive && (
              <div className="flex justify-center gap-3">
                <Button
                  size="lg"
                  onClick={captureImage}
                  disabled={isCapturing || isCheckingQuality}
                  data-testid="button-capture"
                >
                  {isCapturing ? (
                    <Loader2 className="h-5 w-5 animate-spin" />
                  ) : (
                    <Camera className="h-5 w-5" />
                  )}
                  <span className="ml-2">Capture</span>
                </Button>
                <Button variant="outline" size="lg" onClick={stopCamera} data-testid="button-stop-camera">
                  <X className="h-5 w-5" />
                </Button>
              </div>
            )}
          </div>
        </div>
      </CardContent>

      <CardFooter className="flex justify-between pt-4 border-t">
        <Button
          variant="outline"
          onClick={goToPreviousStage}
          disabled={currentStageIndex === 0}
          data-testid="button-prev-stage"
        >
          <ChevronLeft className="h-4 w-4 mr-1" />
          Previous
        </Button>

        <div className="flex gap-2">
          {!allStagesComplete && (
            <Button
              variant="outline"
              onClick={goToNextStage}
              disabled={currentStageIndex === EXAM_STAGES.length - 1}
              data-testid="button-next-stage"
            >
              Next
              <ChevronRight className="h-4 w-4 ml-1" />
            </Button>
          )}
          
          {allStagesComplete && (
            <Button
              onClick={handleComplete}
              disabled={completeSessionMutation.isPending}
              data-testid="button-complete-exam"
            >
              {completeSessionMutation.isPending ? (
                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
              ) : (
                <Sparkles className="h-4 w-4 mr-2" />
              )}
              Complete Exam
            </Button>
          )}
        </div>
      </CardFooter>
    </Card>
  );
}
