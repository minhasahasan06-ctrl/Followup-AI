import { useState, useEffect, useRef } from 'react';
import { Camera, Eye, Hand, Smile, CheckCircle2, AlertCircle, Clock, Video, Loader2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { useToast } from '@/hooks/use-toast';
import { useMutation, useQuery } from '@tanstack/react-query';
import { apiRequest, queryClient } from '@/lib/queryClient';
import { useAuth } from '@/contexts/AuthContext';

type ExamStage = 'eyes' | 'palm' | 'tongue' | 'lips';
type WorkflowState = 'idle' | 'prep' | 'capture' | 'processing' | 'completed' | 'error';

interface StageConfig {
  id: ExamStage;
  title: string;
  icon: typeof Eye;
  instructions: string[];
  captureInstructions: string;
  prepDuration: number; // seconds
}

const EXAM_STAGES: StageConfig[] = [
  {
    id: 'eyes',
    title: 'Eye Examination',
    icon: Eye,
    instructions: [
      'Position your face 12-18 inches from camera',
      'Look directly at the camera',
      'Ensure good lighting on your face',
      'Remove glasses if wearing them',
      'We will analyze sclera color for jaundice detection'
    ],
    captureInstructions: 'Look straight ahead. Hold still for 3 seconds.',
    prepDuration: 30
  },
  {
    id: 'palm',
    title: 'Palm Examination', 
    icon: Hand,
    instructions: [
      'Hold your palm flat facing the camera',
      'Position palm 8-12 inches from camera',
      'Ensure even lighting across palm',
      'Keep fingers together and straight',
      'We will analyze palm color for anemia detection'
    ],
    captureInstructions: 'Hold palm steady. Keep it centered and flat.',
    prepDuration: 30
  },
  {
    id: 'tongue',
    title: 'Tongue Examination',
    icon: Smile,
    instructions: [
      'Open mouth wide and stick out tongue',
      'Position face 8-12 inches from camera',
      'Ensure tongue is fully extended',
      'Avoid shadows on tongue surface',
      'We will analyze tongue color and coating'
    ],
    captureInstructions: 'Stick out tongue fully. Hold position for 3 seconds.',
    prepDuration: 30
  },
  {
    id: 'lips',
    title: 'Lip Examination',
    icon: Smile,
    instructions: [
      'Position face 12-18 inches from camera',
      'Relax lips naturally (closed)',
      'Ensure good lighting on lips',
      'Remove lipstick or lip gloss',
      'We will analyze lip color and hydration'
    ],
    captureInstructions: 'Keep lips relaxed and closed. Hold still.',
    prepDuration: 30
  }
];

export default function GuidedVideoExam() {
  const { user } = useAuth();
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [currentStageIndex, setCurrentStageIndex] = useState(0);
  const [workflowState, setWorkflowState] = useState<WorkflowState>('idle');
  const [prepCountdown, setPrepCountdown] = useState(30);
  const [captureCountdown, setCaptureCountdown] = useState(3);
  const [cameraStream, setCameraStream] = useState<MediaStream | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const { toast } = useToast();
  
  // Get patient ID from auth context
  const patientId = user?.id || 'unknown';

  const currentStage = EXAM_STAGES[currentStageIndex];
  const progress = ((currentStageIndex) / EXAM_STAGES.length) * 100;

  // Create session mutation
  const createSessionMutation = useMutation({
    mutationFn: async () => {
      const response = await apiRequest<{ session_id: string }>('/api/v1/guided-exam/sessions', {
        method: 'POST',
        body: JSON.stringify({
          patient_id: patientId,
          device_info: {
            userAgent: navigator.userAgent,
            screenResolution: `${window.screen.width}x${window.screen.height}`
          }
        }),
        headers: { 'Content-Type': 'application/json' }
      });
      return response;
    },
    onSuccess: (data) => {
      setSessionId(data.session_id);
      setWorkflowState('prep');
      toast({
        title: 'Examination Started',
        description: 'Follow the step-by-step instructions for each stage.'
      });
    },
    onError: (error: Error) => {
      toast({
        title: 'Failed to Start',
        description: error.message,
        variant: 'destructive'
      });
    }
  });

  // Capture frame mutation
  const captureFrameMutation = useMutation({
    mutationFn: async ({ stage, frameBase64 }: { stage: ExamStage; frameBase64: string }) => {
      if (!sessionId) throw new Error('No session ID');
      
      return await apiRequest(`/api/v1/guided-exam/sessions/${sessionId}/capture`, {
        method: 'POST',
        body: JSON.stringify({ stage, frame_base64: frameBase64 }),
        headers: { 'Content-Type': 'application/json' }
      });
    },
    onSuccess: (data: any) => {
      if (data.next_stage) {
        // Move to next stage
        setCurrentStageIndex(prev => prev + 1);
        setWorkflowState('prep');
        setPrepCountdown(30);
      } else {
        // All stages complete, trigger analysis
        completeExamMutation.mutate();
      }
    },
    onError: (error: Error) => {
      toast({
        title: 'Capture Failed',
        description: error.message,
        variant: 'destructive'
      });
      setWorkflowState('error');
    }
  });

  // Complete exam mutation
  const completeExamMutation = useMutation({
    mutationFn: async () => {
      if (!sessionId) throw new Error('No session ID');
      
      return await apiRequest(`/api/v1/guided-exam/sessions/${sessionId}/complete`, {
        method: 'POST'
      });
    },
    onSuccess: () => {
      setWorkflowState('completed');
      toast({
        title: 'Analysis Complete',
        description: 'Your examination has been analyzed successfully.'
      });
      // FIX ISSUE A: Explicitly invalidate results query to trigger auto-refresh
      queryClient.invalidateQueries({ 
        queryKey: [`/api/v1/guided-exam/sessions/${sessionId}/results`] 
      });
      queryClient.invalidateQueries({ queryKey: ['/api/v1/guided-exam/sessions'] });
    },
    onError: (error: Error) => {
      toast({
        title: 'Analysis Failed',
        description: error.message,
        variant: 'destructive'
      });
      setWorkflowState('error');
    }
  });

  // Get results query
  const { data: results, isLoading: resultsLoading } = useQuery({
    queryKey: [`/api/v1/guided-exam/sessions/${sessionId}/results`],
    enabled: workflowState === 'completed' && !!sessionId
  });
  
  // Get personalized configs (for display purposes)
  const { data: personalizedConfig } = useQuery({
    queryKey: [`/api/v1/guided-exam/personalized-config`, patientId],
    enabled: !!patientId && workflowState === 'idle'
  });

  // Prep countdown timer
  useEffect(() => {
    if (workflowState === 'prep' && prepCountdown > 0) {
      const timer = setTimeout(() => setPrepCountdown(prev => prev - 1), 1000);
      return () => clearTimeout(timer);
    } else if (workflowState === 'prep' && prepCountdown === 0) {
      startCapture();
    }
  }, [workflowState, prepCountdown]);

  // Capture countdown timer
  useEffect(() => {
    if (workflowState === 'capture' && captureCountdown > 0) {
      const timer = setTimeout(() => setCaptureCountdown(prev => prev - 1), 1000);
      return () => clearTimeout(timer);
    } else if (workflowState === 'capture' && captureCountdown === 0) {
      captureFrame();
    }
  }, [workflowState, captureCountdown]);

  // Start camera when entering capture state
  const startCapture = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 1280, height: 720, facingMode: 'user' },
        audio: false
      });
      
      setCameraStream(stream);
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
      
      setWorkflowState('capture');
      setCaptureCountdown(3);
    } catch (error) {
      toast({
        title: 'Camera Error',
        description: 'Failed to access camera. Please grant camera permissions.',
        variant: 'destructive'
      });
      setWorkflowState('error');
    }
  };

  // Capture frame from video
  const captureFrame = () => {
    if (!videoRef.current || !canvasRef.current) return;
    
    const canvas = canvasRef.current;
    const video = videoRef.current;
    
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    ctx.drawImage(video, 0, 0);
    
    // Convert to base64
    const frameBase64 = canvas.toDataURL('image/jpeg', 0.9);
    
    // Stop camera
    if (cameraStream) {
      cameraStream.getTracks().forEach(track => track.stop());
      setCameraStream(null);
    }
    
    // Upload frame
    setWorkflowState('processing');
    captureFrameMutation.mutate({
      stage: currentStage.id,
      frameBase64
    });
  };

  // Skip to manual review
  const skipPrepCountdown = () => {
    setPrepCountdown(0);
  };

  // Start exam
  const startExam = () => {
    setCurrentStageIndex(0);
    setPrepCountdown(30);
    createSessionMutation.mutate();
  };

  // Reset exam
  const resetExam = () => {
    setSessionId(null);
    setCurrentStageIndex(0);
    setWorkflowState('idle');
    setPrepCountdown(30);
    setCaptureCountdown(3);
    if (cameraStream) {
      cameraStream.getTracks().forEach(track => track.stop());
      setCameraStream(null);
    }
  };

  // Render idle state
  if (workflowState === 'idle') {
    return (
      <div className="container mx-auto p-6 max-w-4xl space-y-6">
        <div className="space-y-2">
          <h1 className="text-4xl font-bold flex items-center gap-3" data-testid="text-page-title">
            <Video className="h-10 w-10 text-primary" />
            Guided Video Examination
          </h1>
          <p className="text-muted-foreground">
            Complete 4-stage health examination with AI-powered analysis
          </p>
        </div>

        <Card>
          <CardHeader>
            <CardTitle>Examination Overview</CardTitle>
            <CardDescription>
              This guided examination captures specific body regions to detect health changes
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {EXAM_STAGES.map((stage, index) => (
              <div key={stage.id} className="flex items-start gap-3 p-3 rounded-lg bg-muted/30">
                <div className="p-2 rounded-lg bg-primary/10">
                  <stage.icon className="h-5 w-5 text-primary" />
                </div>
                <div className="flex-1">
                  <h3 className="font-semibold">{index + 1}. {stage.title}</h3>
                  <p className="text-sm text-muted-foreground mt-1">
                    {stage.instructions[stage.instructions.length - 1]}
                  </p>
                </div>
              </div>
            ))}

            <div className="mt-6 p-4 bg-amber-500/10 border border-amber-500/20 rounded-lg">
              <p className="text-sm text-amber-900 dark:text-amber-100 flex items-center gap-2">
                <AlertCircle className="h-4 w-4" />
                <span>
                  <strong>Privacy Notice:</strong> All images are encrypted and used only for health monitoring. 
                  This is not a diagnostic tool. Consult your doctor for medical advice.
                </span>
              </p>
            </div>

            <Button 
              onClick={startExam} 
              disabled={createSessionMutation.isPending}
              className="w-full"
              size="lg"
              data-testid="button-start-exam"
            >
              {createSessionMutation.isPending ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Starting...
                </>
              ) : (
                <>
                  <Camera className="mr-2 h-4 w-4" />
                  Start Examination
                </>
              )}
            </Button>
          </CardContent>
        </Card>
      </div>
    );
  }

  // Render prep state
  if (workflowState === 'prep') {
    const StageIcon = currentStage.icon;
    
    return (
      <div className="container mx-auto p-6 max-w-4xl space-y-6">
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <h1 className="text-2xl font-bold">
              Stage {currentStageIndex + 1} of {EXAM_STAGES.length}: {currentStage.title}
            </h1>
            <span className="text-sm text-muted-foreground">Session: {sessionId?.slice(0, 8)}</span>
          </div>
          <Progress value={progress} className="h-2" />
        </div>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-3">
              <StageIcon className="h-6 w-6 text-primary" />
              Preparation Instructions
            </CardTitle>
            <CardDescription>
              Please review and follow these instructions carefully
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            <div className="space-y-3">
              {currentStage.instructions.map((instruction, index) => (
                <div key={index} className="flex items-start gap-3">
                  <div className="flex-shrink-0 w-6 h-6 rounded-full bg-primary/10 flex items-center justify-center text-primary text-sm font-semibold">
                    {index + 1}
                  </div>
                  <p className="text-sm pt-0.5">{instruction}</p>
                </div>
              ))}
            </div>

            <div className="p-6 bg-primary/5 rounded-lg text-center space-y-3">
              <Clock className="h-12 w-12 mx-auto text-primary" />
              <div>
                <p className="text-sm text-muted-foreground">Camera will start in</p>
                <p className="text-5xl font-bold text-primary">{prepCountdown}</p>
                <p className="text-sm text-muted-foreground mt-1">seconds</p>
              </div>
            </div>

            <div className="flex gap-3">
              <Button 
                onClick={skipPrepCountdown}
                variant="outline"
                className="flex-1"
                data-testid="button-skip-prep"
              >
                I'm Ready - Skip Countdown
              </Button>
              <Button 
                onClick={resetExam}
                variant="ghost"
                data-testid="button-cancel-exam"
              >
                Cancel Exam
              </Button>
            </div>
          </CardContent>
        </Card>
      </div>
    );
  }

  // Render capture state
  if (workflowState === 'capture') {
    return (
      <div className="container mx-auto p-6 max-w-4xl space-y-6">
        <div className="space-y-2">
          <h1 className="text-2xl font-bold">
            Capturing: {currentStage.title}
          </h1>
          <Progress value={progress} className="h-2" />
        </div>

        <Card>
          <CardContent className="p-6 space-y-4">
            <div className="relative aspect-video bg-black rounded-lg overflow-hidden">
              <video
                ref={videoRef}
                autoPlay
                playsInline
                muted
                className="w-full h-full object-cover"
              />
              
              {/* Countdown overlay */}
              <div className="absolute inset-0 flex items-center justify-center bg-black/50">
                <div className="text-center space-y-2">
                  <p className="text-white text-xl">{currentStage.captureInstructions}</p>
                  <p className="text-white text-6xl font-bold">{captureCountdown}</p>
                </div>
              </div>
            </div>

            <p className="text-center text-sm text-muted-foreground">
              Hold position... Capturing in {captureCountdown} seconds
            </p>
          </CardContent>
        </Card>

        {/* Hidden canvas for frame capture */}
        <canvas ref={canvasRef} className="hidden" />
      </div>
    );
  }

  // Render processing state
  if (workflowState === 'processing' || completeExamMutation.isPending) {
    return (
      <div className="container mx-auto p-6 max-w-4xl">
        <Card>
          <CardContent className="p-12 text-center space-y-4">
            <Loader2 className="h-16 w-16 mx-auto animate-spin text-primary" />
            <h2 className="text-2xl font-bold">Processing</h2>
            <p className="text-muted-foreground">
              {completeExamMutation.isPending 
                ? 'Analyzing all captured frames with AI...' 
                : 'Uploading captured frame...'}
            </p>
          </CardContent>
        </Card>
      </div>
    );
  }

  // Render completed state
  if (workflowState === 'completed') {
    return (
      <div className="container mx-auto p-6 max-w-4xl space-y-6">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-3 text-green-600">
              <CheckCircle2 className="h-6 w-6" />
              Examination Complete
            </CardTitle>
            <CardDescription>
              Your guided examination has been analyzed successfully
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {resultsLoading && (
              <div className="text-center py-8">
                <Loader2 className="h-8 w-8 mx-auto animate-spin text-primary" />
                <p className="text-sm text-muted-foreground mt-2">Loading results...</p>
              </div>
            )}
            
            {results && !resultsLoading && (
              <div className="space-y-4">
                <h3 className="font-semibold text-lg">Clinical Metrics Analysis:</h3>
                
                <div className="grid md:grid-cols-2 gap-3">
                  {/* Scleral Analysis */}
                  {typeof results.scleral_chromaticity_index === 'number' && (
                    <div className="p-4 bg-muted/30 rounded-lg border border-border">
                      <div className="flex items-center gap-2 mb-2">
                        <Eye className="h-4 w-4 text-primary" />
                        <p className="text-sm font-medium">Scleral Chromaticity</p>
                      </div>
                      <p className="text-3xl font-bold text-primary">{results.scleral_chromaticity_index.toFixed(1)}</p>
                      <p className="text-xs text-muted-foreground mt-1">
                        Jaundice Detection • Normal: 0-20
                      </p>
                      {results.sclera_b_channel && (
                        <p className="text-xs text-muted-foreground mt-1">
                          b* channel: {results.sclera_b_channel.toFixed(1)}
                        </p>
                      )}
                    </div>
                  )}

                  {/* Conjunctival Pallor */}
                  {typeof results.conjunctival_pallor_index === 'number' && (
                    <div className="p-4 bg-muted/30 rounded-lg border border-border">
                      <div className="flex items-center gap-2 mb-2">
                        <Hand className="h-4 w-4 text-primary" />
                        <p className="text-sm font-medium">Conjunctival Pallor Index</p>
                      </div>
                      <p className="text-3xl font-bold text-primary">{results.conjunctival_pallor_index.toFixed(1)}</p>
                      <p className="text-xs text-muted-foreground mt-1">
                        Anemia Detection • Normal: 45-100
                      </p>
                      {results.palmar_redness_a && (
                        <p className="text-xs text-muted-foreground mt-1">
                          a* redness: {results.palmar_redness_a.toFixed(1)}
                        </p>
                      )}
                    </div>
                  )}

                  {/* Tongue Analysis */}
                  {typeof results.tongue_color_index === 'number' && (
                    <div className="p-4 bg-muted/30 rounded-lg border border-border">
                      <div className="flex items-center gap-2 mb-2">
                        <Smile className="h-4 w-4 text-primary" />
                        <p className="text-sm font-medium">Tongue Color Index</p>
                      </div>
                      <p className="text-3xl font-bold text-primary">{results.tongue_color_index.toFixed(1)}</p>
                      {results.tongue_coating_detected && (
                        <p className="text-xs text-amber-600 font-medium mt-1">
                          ⚠️ Coating: {results.tongue_coating_color || 'detected'}
                        </p>
                      )}
                      {results.tongue_coating_thickness !== null && (
                        <p className="text-xs text-muted-foreground mt-1">
                          Thickness: {results.tongue_coating_thickness.toFixed(0)}%
                        </p>
                      )}
                    </div>
                  )}

                  {/* Lip Hydration */}
                  {typeof results.lip_hydration_score === 'number' && (
                    <div className="p-4 bg-muted/30 rounded-lg border border-border">
                      <div className="flex items-center gap-2 mb-2">
                        <Smile className="h-4 w-4 text-primary" />
                        <p className="text-sm font-medium">Lip Hydration Score</p>
                      </div>
                      <p className="text-3xl font-bold text-primary">{results.lip_hydration_score.toFixed(1)}</p>
                      {results.lip_cyanosis_detected && (
                        <p className="text-xs text-red-600 font-medium mt-1">⚠️ Cyanosis detected</p>
                      )}
                      {results.lip_dryness_score !== null && (
                        <p className="text-xs text-muted-foreground mt-1">
                          Dryness: {results.lip_dryness_score.toFixed(0)}%
                        </p>
                      )}
                    </div>
                  )}
                </div>
                
                {/* Additional LAB Color Metrics */}
                <details className="mt-4">
                  <summary className="cursor-pointer text-sm font-medium text-muted-foreground hover:text-foreground">
                    View Detailed LAB Color Analysis (31 metrics)
                  </summary>
                  <div className="mt-3 p-4 bg-muted/20 rounded-lg space-y-2 text-sm">
                    {results.sclera_l_channel && <p>Sclera L*: {results.sclera_l_channel.toFixed(1)}</p>}
                    {results.sclera_a_channel && <p>Sclera a*: {results.sclera_a_channel.toFixed(1)}</p>}
                    {results.palmar_l_channel && <p>Palm L*: {results.palmar_l_channel.toFixed(1)}</p>}
                    {results.palmar_perfusion_index && <p>Palmar Perfusion: {results.palmar_perfusion_index.toFixed(1)}</p>}
                    {results.tongue_l_channel && <p>Tongue L*: {results.tongue_l_channel.toFixed(1)}</p>}
                    {results.lip_l_channel && <p>Lip L*: {results.lip_l_channel.toFixed(1)}</p>}
                    <p className="text-xs text-muted-foreground italic mt-2">
                      LAB color space provides perceptually-uniform clinical measurements
                    </p>
                  </div>
                </details>
              </div>
            )}
            
            {!results && !resultsLoading && (
              <p className="text-sm text-muted-foreground text-center py-4">
                No results available yet. Please try again.
              </p>
            )}

            <div className="mt-6 p-4 bg-amber-500/10 border border-amber-500/20 rounded-lg">
              <p className="text-sm text-amber-900 dark:text-amber-100">
                <strong>Important:</strong> These metrics are for wellness monitoring only and are not diagnostic. 
                Consult your healthcare provider for medical interpretation.
              </p>
            </div>

            <Button onClick={resetExam} className="w-full" data-testid="button-new-exam">
              Start New Examination
            </Button>
          </CardContent>
        </Card>
      </div>
    );
  }

  // Render error state
  if (workflowState === 'error') {
    return (
      <div className="container mx-auto p-6 max-w-4xl">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-3 text-destructive">
              <AlertCircle className="h-6 w-6" />
              Examination Error
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <p className="text-muted-foreground">
              An error occurred during the examination. Please try again.
            </p>
            <Button onClick={resetExam} className="w-full">
              Start Over
            </Button>
          </CardContent>
        </Card>
      </div>
    );
  }

  return null;
}
