import { useState, useRef, useEffect } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { useToast } from "@/hooks/use-toast";
import { Camera, Check, AlertCircle, Loader2, Play, Square } from "lucide-react";
import { apiRequest, queryClient } from "@/lib/queryClient";

export default function ExamCoach() {
  const { toast } = useToast();
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  
  const [activeTab, setActiveTab] = useState("select");
  const [selectedExam, setSelectedExam] = useState<string | null>(null);
  const [currentSession, setCurrentSession] = useState<any>(null);
  const [currentStep, setCurrentStep] = useState<any>(null);
  const [cameraActive, setCameraActive] = useState(false);
  const [coachingFeedback, setCoachingFeedback] = useState<any>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [capturedImage, setCapturedImage] = useState<string | null>(null);
  
  // Fetch available exam protocols
  const { data: protocols } = useQuery({
    queryKey: ["/api/v1/exam-coach/protocols"],
  });
  
  // Start session mutation
  const startSession = useMutation({
    mutationFn: async (examType: string) => {
      return await apiRequest("/api/v1/exam-coach/start-session", {
        method: "POST",
        json: { exam_type: examType }
      });
    },
    onSuccess: (data) => {
      setCurrentSession(data);
      setCurrentStep(data.current_step);
      setActiveTab("coaching");
      toast({
        title: "Session Started",
        description: `${data.protocol.name} coaching session begun`
      });
    }
  });
  
  // Analyze frame mutation
  const analyzeFrame = useMutation({
    mutationFn: async (frameData: string) => {
      return await apiRequest("/api/v1/exam-coach/analyze-frame", {
        method: "POST",
        json: {
          session_id: currentSession.session_id,
          step_id: currentStep.step_id,
          frame_data: frameData,
          current_instruction: currentStep.instruction
        }
      });
    },
    onSuccess: (data) => {
      setCoachingFeedback(data.analysis);
      
      // Speak voice guidance
      if (data.analysis.voice_guidance && 'speechSynthesis' in window) {
        const utterance = new SpeechSynthesisUtterance(data.analysis.voice_guidance);
        speechSynthesis.speak(utterance);
      }
      
      if (data.analysis.is_ready) {
        toast({
          title: "Ready to Capture!",
          description: data.analysis.readiness_message,
        });
      }
    }
  });
  
  // Complete step mutation
  const completeStep = useMutation({
    mutationFn: async (imageData: string) => {
      const blob = await fetch(imageData).then(r => r.blob());
      const formData = new FormData();
      formData.append('session_id', currentSession.session_id.toString());
      formData.append('step_id', currentStep.step_id.toString());
      formData.append('image_file', blob, 'capture.jpg');
      
      return await apiRequest("/api/v1/exam-coach/complete-step", {
        method: "POST",
        body: formData
      });
    },
    onSuccess: (data) => {
      toast({
        title: "Step Complete!",
        description: `${data.completed_steps} of ${data.total_steps} steps completed`
      });
      
      setCapturedImage(null);
      setCoachingFeedback(null);
      
      if (data.session_status === "completed") {
        stopCamera();
        setActiveTab("complete");
        toast({
          title: "Exam Complete!",
          description: "Great job! Your exam has been saved.",
        });
      } else {
        setCurrentStep(data.next_step);
      }
    }
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
      
      streamRef.current = mediaStream;
      setCameraActive(true);
      
      // Start periodic frame analysis
      startFrameAnalysis();
      
    } catch (error) {
      console.error("Camera error:", error);
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
    setCameraActive(false);
    setIsAnalyzing(false);
  };
  
  // Start frame analysis
  const startFrameAnalysis = () => {
    const interval = setInterval(() => {
      if (!cameraActive || !videoRef.current || isAnalyzing) return;
      
      // Capture frame
      const canvas = canvasRef.current;
      if (!canvas) return;
      
      const ctx = canvas.getContext('2d');
      if (!ctx) return;
      
      canvas.width = videoRef.current.videoWidth;
      canvas.height = videoRef.current.videoHeight;
      ctx.drawImage(videoRef.current, 0, 0);
      
      const frameData = canvas.toDataURL('image/jpeg', 0.8);
      setIsAnalyzing(true);
      
      // Analyze frame
      analyzeFrame.mutate(frameData, {
        onSettled: () => setIsAnalyzing(false)
      });
      
    }, 3000); // Analyze every 3 seconds
    
    return interval;
  };
  
  // Capture photo
  const capturePhoto = () => {
    if (!videoRef.current || !canvasRef.current) return;
    
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    canvas.width = videoRef.current.videoWidth;
    canvas.height = videoRef.current.videoHeight;
    ctx.drawImage(videoRef.current, 0, 0);
    
    const imageData = canvas.toDataURL('image/jpeg', 0.9);
    setCapturedImage(imageData);
  };
  
  // Submit captured image
  const submitImage = () => {
    if (!capturedImage) return;
    completeStep.mutate(capturedImage);
  };
  
  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => {
          track.stop();
          track.enabled = false;
        });
        streamRef.current = null;
      }
    };
  }, []);
  
  return (
    <div className="container mx-auto p-6">
      <Card>
        <CardHeader>
          <CardTitle>Home Clinical Exam Coach</CardTitle>
          <CardDescription>
            AI-powered guidance for proper self-examination techniques
          </CardDescription>
        </CardHeader>
        
        <CardContent>
          <Tabs value={activeTab} onValueChange={setActiveTab}>
            <TabsList className="grid w-full grid-cols-3" data-testid="tabs-exam-coach">
              <TabsTrigger value="select" data-testid="tab-select-exam">Select Exam</TabsTrigger>
              <TabsTrigger value="coaching" disabled={!currentSession} data-testid="tab-coaching">Coaching</TabsTrigger>
              <TabsTrigger value="complete" disabled={!currentSession} data-testid="tab-complete">Complete</TabsTrigger>
            </TabsList>
            
            {/* Select Exam Tab */}
            <TabsContent value="select" className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {protocols?.protocols?.map((protocol: any) => (
                  <Card
                    key={protocol.exam_type}
                    className={`cursor-pointer hover-elevate ${
                      selectedExam === protocol.exam_type ? 'border-primary' : ''
                    }`}
                    onClick={() => setSelectedExam(protocol.exam_type)}
                    data-testid={`card-exam-${protocol.exam_type}`}
                  >
                    <CardHeader>
                      <CardTitle className="text-lg">{protocol.name}</CardTitle>
                      <CardDescription>{protocol.description}</CardDescription>
                    </CardHeader>
                    <CardContent>
                      <Badge data-testid={`badge-steps-${protocol.exam_type}`}>
                        {protocol.total_steps} steps
                      </Badge>
                    </CardContent>
                  </Card>
                ))}
              </div>
              
              <Button
                onClick={() => selectedExam && startSession.mutate(selectedExam)}
                disabled={!selectedExam || startSession.isPending}
                className="w-full"
                data-testid="button-start-exam"
              >
                {startSession.isPending ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Starting...
                  </>
                ) : (
                  <>
                    <Play className="mr-2 h-4 w-4" />
                    Start Exam
                  </>
                )}
              </Button>
            </TabsContent>
            
            {/* Coaching Tab */}
            <TabsContent value="coaching" className="space-y-4">
              {currentStep && (
                <>
                  {/* Progress */}
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span data-testid="text-step-progress">
                        Step {currentStep.step_number} of {currentSession?.protocol?.steps?.length || 0}
                      </span>
                      <span>{Math.round((currentStep.step_number / (currentSession?.protocol?.steps?.length || 1)) * 100)}%</span>
                    </div>
                    <Progress
                      value={(currentStep.step_number / (currentSession?.protocol?.steps?.length || 1)) * 100}
                      data-testid="progress-exam"
                    />
                  </div>
                  
                  {/* Instruction */}
                  <Card className="bg-primary/5">
                    <CardHeader>
                      <CardTitle className="text-lg">Current Instruction</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <p className="text-lg font-medium" data-testid="text-instruction">
                        {currentStep.instruction}
                      </p>
                      
                      {/* Coaching Hints */}
                      {currentStep.hints && currentStep.hints.length > 0 && (
                        <div className="mt-4 space-y-2">
                          <p className="text-sm font-medium">Tips:</p>
                          <ul className="list-disc list-inside space-y-1">
                            {currentStep.hints.map((hint: string, idx: number) => (
                              <li key={idx} className="text-sm text-muted-foreground">
                                {hint}
                              </li>
                            ))}
                          </ul>
                        </div>
                      )}
                    </CardContent>
                  </Card>
                  
                  {/* Camera View */}
                  {!capturedImage ? (
                    <Card>
                      <CardContent className="p-6">
                        {cameraActive ? (
                          <div className="space-y-4">
                            <div className="relative bg-black rounded-lg overflow-hidden">
                              <video
                                ref={videoRef}
                                autoPlay
                                playsInline
                                muted
                                className="w-full"
                                data-testid="video-camera-feed"
                              />
                              <canvas ref={canvasRef} className="hidden" />
                              
                              {/* Real-time Coaching Overlay */}
                              {coachingFeedback && (
                                <div className="absolute top-4 left-4 right-4 space-y-2">
                                  {coachingFeedback.is_ready ? (
                                    <Badge className="bg-green-500" data-testid="badge-ready">
                                      <Check className="mr-1 h-3 w-3" />
                                      {coachingFeedback.readiness_message}
                                    </Badge>
                                  ) : (
                                    <div className="space-y-2">
                                      {coachingFeedback.lighting_feedback && (
                                        <Badge variant="secondary" data-testid="badge-lighting-feedback">
                                          {coachingFeedback.lighting_feedback}
                                        </Badge>
                                      )}
                                      {coachingFeedback.angle_feedback && (
                                        <Badge variant="secondary" data-testid="badge-angle-feedback">
                                          {coachingFeedback.angle_feedback}
                                        </Badge>
                                      )}
                                      {coachingFeedback.distance_feedback && (
                                        <Badge variant="secondary" data-testid="badge-distance-feedback">
                                          {coachingFeedback.distance_feedback}
                                        </Badge>
                                      )}
                                    </div>
                                  )}
                                </div>
                              )}
                            </div>
                            
                            <div className="flex gap-2">
                              <Button
                                onClick={capturePhoto}
                                disabled={!coachingFeedback?.is_ready}
                                className="flex-1"
                                data-testid="button-capture"
                              >
                                <Camera className="mr-2 h-4 w-4" />
                                Capture Photo
                              </Button>
                              <Button
                                variant="outline"
                                onClick={stopCamera}
                                data-testid="button-stop"
                              >
                                <Square className="mr-2 h-4 w-4" />
                                Stop
                              </Button>
                            </div>
                          </div>
                        ) : (
                          <Button
                            onClick={startCamera}
                            className="w-full"
                            data-testid="button-start-camera"
                          >
                            <Camera className="mr-2 h-4 w-4" />
                            Start Camera
                          </Button>
                        )}
                      </CardContent>
                    </Card>
                  ) : (
                    <Card>
                      <CardContent className="p-6 space-y-4">
                        <img
                          src={capturedImage}
                          alt="Captured"
                          className="w-full rounded-lg"
                          data-testid="img-captured-preview"
                        />
                        <div className="flex gap-2">
                          <Button
                            onClick={submitImage}
                            disabled={completeStep.isPending}
                            className="flex-1"
                            data-testid="button-submit-step"
                          >
                            {completeStep.isPending ? (
                              <>
                                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                                Submitting...
                              </>
                            ) : (
                              <>
                                <Check className="mr-2 h-4 w-4" />
                                Submit & Continue
                              </>
                            )}
                          </Button>
                          <Button
                            variant="outline"
                            onClick={() => setCapturedImage(null)}
                            data-testid="button-retake"
                          >
                            Retake
                          </Button>
                        </div>
                      </CardContent>
                    </Card>
                  )}
                </>
              )}
            </TabsContent>
            
            {/* Complete Tab */}
            <TabsContent value="complete" className="space-y-4">
              <Card className="bg-green-50 dark:bg-green-950">
                <CardContent className="p-6 text-center">
                  <Check className="h-16 w-16 text-green-500 mx-auto mb-4" />
                  <h3 className="text-2xl font-bold mb-2">Exam Complete!</h3>
                  <p className="text-muted-foreground">
                    Your exam has been saved and is ready for doctor review.
                  </p>
                </CardContent>
              </Card>
              
              <Button
                onClick={() => {
                  setCurrentSession(null);
                  setCurrentStep(null);
                  setActiveTab("select");
                }}
                className="w-full"
                data-testid="button-start-new-exam"
              >
                Start New Exam
              </Button>
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>
    </div>
  );
}
