import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Link } from 'wouter';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Label } from '@/components/ui/label';
import {
  Video,
  Camera,
  Play,
  CheckCircle2,
  AlertCircle,
  Info,
  BarChart3,
  Loader2,
  Wind,
  Hand,
  Eye,
  User,
  Activity,
  MessageSquare,
  XCircle,
  Palette,
  Users,
  Zap,
  Smile,
  ExternalLink,
} from 'lucide-react';
import { ExamPrepStep } from '@/components/ExamPrepStep';
import { VideoRecorder } from '@/components/VideoRecorder';
import { useGuidedExamWorkflow } from '@/hooks/useGuidedExamWorkflow';
import { useAuth } from '@/contexts/AuthContext';

function LegalDisclaimer() {
  return (
    <Alert className="border-amber-500/50 bg-amber-500/5">
      <AlertCircle className="h-4 w-4 text-amber-500" />
      <AlertTitle className="text-amber-700 dark:text-amber-400 font-semibold">
        Wellness Monitoring Platform - Not a Medical Device
      </AlertTitle>
      <AlertDescription className="text-amber-600 dark:text-amber-300 text-sm">
        This platform is not a medical device and is not intended to diagnose, treat, cure, or prevent any disease.
        All information provided is for wellness monitoring and personal tracking purposes only.
        Please discuss any changes or concerns with your healthcare provider.
      </AlertDescription>
    </Alert>
  );
}

export default function DailyFollowup() {
  const { user } = useAuth();
  const [customLocation, setCustomLocation] = useState('');
  const [customDescription, setCustomDescription] = useState('');

  const {
    sessionId,
    currentStep,
    currentStepIndex,
    totalSteps,
    workflowState,
    examSteps,
    progress,
    completedSteps,
    startExam,
    startRecording,
    completeRecording,
    skipCurrentStep,
    resetWorkflow,
    isStarting,
    isUploading,
  } = useGuidedExamWorkflow();

  // Fetch today's video metrics (last 24 hours)
  const { data: todayMetrics } = useQuery<any>({
    queryKey: ["/api/video-ai/latest-metrics"],
    enabled: !!user && (workflowState === 'idle' || workflowState === 'completed'),
  });

  const handleRecordingComplete = (videoBlob: Blob, durationSeconds: number) => {
    if (currentStep?.examType === 'custom') {
      completeRecording(videoBlob, durationSeconds, {
        location: customLocation,
        description: customDescription,
      });
      setCustomLocation('');
      setCustomDescription('');
    } else {
      completeRecording(videoBlob, durationSeconds);
    }
  };

  // Check if metrics are from today (last 24 hours)
  const hasMetricsToday = todayMetrics && todayMetrics.created_at && 
    (Date.now() - new Date(todayMetrics.created_at).getTime()) < 24 * 60 * 60 * 1000;

  // Active examination workflow
  if (workflowState !== 'idle') {
    return (
      <div className="container mx-auto p-6 max-w-5xl space-y-6">
        {/* Progress Header */}
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h1 className="text-3xl font-bold tracking-tight" data-testid="text-exam-title">
              Guided Examination
            </h1>
            <Badge variant="secondary" className="text-sm">
              Step {currentStepIndex + 1} of {totalSteps}
            </Badge>
          </div>

          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <span className="text-muted-foreground">Overall Progress</span>
              <span className="font-medium">{completedSteps} / {totalSteps} exams</span>
            </div>
            <Progress value={progress} className="h-3" data-testid="progress-overall" />
          </div>
        </div>

        {/* Exam Steps Progress Indicators */}
        <div className="grid grid-cols-7 gap-2">
          {examSteps.map((step, idx) => {
            const Icon = {
              respiratory: Wind,
              skin_pallor: Hand,
              eye_sclera: Eye,
              swelling: User,
              tremor: Activity,
              tongue: MessageSquare,
              custom: AlertCircle,
            }[step.examType] || Video;

            const isActive = idx === currentStepIndex;
            const isCompleted = step.segmentId || step.skipped;
            const isSkipped = step.skipped;

            return (
              <div
                key={step.examType}
                className={`flex flex-col items-center gap-2 p-3 rounded-lg border transition-all ${
                  isActive ? 'bg-primary/10 border-primary' : 
                  isCompleted ? 'bg-muted border-muted' : 
                  'bg-card'
                }`}
              >
                <div className={`relative ${isSkipped ? 'opacity-50' : ''}`}>
                  <Icon className={`h-5 w-5 ${
                    isActive ? 'text-primary' : 
                    isCompleted ? 'text-muted-foreground' : 
                    'text-muted-foreground/50'
                  }`} />
                  {isCompleted && !isSkipped && (
                    <CheckCircle2 className="h-3 w-3 text-chart-2 absolute -top-1 -right-1" />
                  )}
                  {isSkipped && (
                    <XCircle className="h-3 w-3 text-muted-foreground absolute -top-1 -right-1" />
                  )}
                </div>
                <span className="text-xs text-center font-medium line-clamp-2">
                  {step.examType.replace('_', ' ')}
                </span>
              </div>
            );
          })}
        </div>

        {/* Current Step Content */}
        {workflowState === 'prep' && currentStep && (
          <ExamPrepStep
            examType={currentStep.examType}
            sequenceNumber={currentStepIndex + 1}
            totalExams={totalSteps}
            onStartRecording={startRecording}
            onSkipExam={skipCurrentStep}
          />
        )}

        {workflowState === 'recording' && currentStep && (
          <div className="space-y-4">
            {currentStep.examType === 'custom' && !customLocation && (
              <Card>
                <CardHeader>
                  <CardTitle>Describe the Area of Concern</CardTitle>
                  <CardDescription>Help us understand what you want to examine</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="space-y-2">
                    <Label htmlFor="location">Location on Body</Label>
                    <Input
                      id="location"
                      placeholder="e.g., Left forearm, Right ankle, Upper back"
                      value={customLocation}
                      onChange={(e) => setCustomLocation(e.target.value)}
                      data-testid="input-custom-location"
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="description">What to Look For</Label>
                    <Textarea
                      id="description"
                      placeholder="e.g., Red rash, unusual discoloration, new mole, swelling"
                      value={customDescription}
                      onChange={(e) => setCustomDescription(e.target.value)}
                      rows={3}
                      data-testid="input-custom-description"
                    />
                  </div>
                </CardContent>
              </Card>
            )}

            {(currentStep.examType !== 'custom' || customLocation) && (
              <VideoRecorder
                examType={currentStep.examType}
                onRecordingComplete={handleRecordingComplete}
                onCancel={() => resetWorkflow()}
                maxDurationSeconds={60}
                autoStopAfter={currentStep.examType === 'respiratory' ? 30 : undefined}
              />
            )}
          </div>
        )}

        {workflowState === 'uploading' && (
          <Card>
            <CardContent className="p-12 text-center">
              <Loader2 className="h-12 w-12 animate-spin mx-auto mb-4 text-primary" />
              <h3 className="text-lg font-semibold mb-2">Uploading & Analyzing...</h3>
              <p className="text-muted-foreground">
                Securely uploading your recording and running AI analysis
              </p>
            </CardContent>
          </Card>
        )}

        {workflowState === 'completed' && (
          <Card className="border-chart-2">
            <CardContent className="p-12 text-center">
              <CheckCircle2 className="h-16 w-16 mx-auto mb-4 text-chart-2" />
              <h2 className="text-2xl font-bold mb-2">Examination Complete!</h2>
              <p className="text-muted-foreground mb-6">
                All recordings have been processed. View your analysis below.
              </p>
              <Button onClick={resetWorkflow} size="lg" data-testid="button-start-new">
                View Today's Results
              </Button>
            </CardContent>
          </Card>
        )}
      </div>
    );
  }

  // Idle state - show today's metrics or start examination
  return (
    <div className="container mx-auto p-6 max-w-7xl space-y-6">
      {/* Header */}
      <div className="space-y-2">
        <h1 className="text-4xl font-bold tracking-tight text-foreground flex items-center gap-3" data-testid="text-page-title">
          <Activity className="h-10 w-10 text-primary" />
          Daily Followup
        </h1>
        <p className="text-muted-foreground leading-relaxed">
          Complete your daily video examination and track your health metrics
        </p>
      </div>

      <LegalDisclaimer />

      {/* Today's Metrics or Start Examination */}
      {hasMetricsToday ? (
        <div className="space-y-6">
          {/* Header with Link to Full Analysis */}
          <Card className="border-primary/20">
            <CardHeader>
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle className="flex items-center gap-2">
                    <CheckCircle2 className="h-6 w-6 text-chart-2" />
                    Today's Video AI Metrics
                  </CardTitle>
                  <CardDescription>
                    Last examination: {new Date(todayMetrics.created_at).toLocaleString()}
                  </CardDescription>
                </div>
                <Link href="/ai-video">
                  <Button variant="outline" size="sm" className="gap-2" data-testid="button-view-full-analysis">
                    <BarChart3 className="h-4 w-4" />
                    Full Analysis
                    <ExternalLink className="h-3 w-3" />
                  </Button>
                </Link>
              </div>
            </CardHeader>
            <CardContent className="space-y-4">
              {/* ML Metrics Grid */}
              <div className="grid grid-cols-2 gap-3">
                {todayMetrics.respiratory_rate_bpm && (
                  <div className="flex items-center justify-between p-3 rounded-md bg-muted/50">
                    <div className="flex items-center gap-2">
                      <Wind className="h-4 w-4 text-blue-500" />
                      <span className="text-sm text-muted-foreground">Respiratory Rate</span>
                    </div>
                    <span className="font-medium">{todayMetrics.respiratory_rate_bpm.toFixed(1)} bpm</span>
                  </div>
                )}
                
                {todayMetrics.skin_pallor_score !== null && todayMetrics.skin_pallor_score !== undefined && (
                  <div className="flex items-center justify-between p-3 rounded-md bg-muted/50">
                    <div className="flex items-center gap-2">
                      <Palette className="h-4 w-4 text-amber-500" />
                      <span className="text-sm text-muted-foreground">Skin Pallor</span>
                    </div>
                    <span className="font-medium">{todayMetrics.skin_pallor_score.toFixed(1)}/100</span>
                  </div>
                )}
                
                {todayMetrics.jaundice_risk_level && (
                  <div className="flex items-center justify-between p-3 rounded-md bg-muted/50">
                    <div className="flex items-center gap-2">
                      <Eye className="h-4 w-4 text-yellow-500" />
                      <span className="text-sm text-muted-foreground">Jaundice Risk</span>
                    </div>
                    <span className="font-medium capitalize">{todayMetrics.jaundice_risk_level}</span>
                  </div>
                )}
                
                {todayMetrics.facial_swelling_score !== null && todayMetrics.facial_swelling_score !== undefined && (
                  <div className="flex items-center justify-between p-3 rounded-md bg-muted/50">
                    <div className="flex items-center gap-2">
                      <Users className="h-4 w-4 text-rose-500" />
                      <span className="text-sm text-muted-foreground">Facial Swelling</span>
                    </div>
                    <span className="font-medium">{todayMetrics.facial_swelling_score.toFixed(1)}/100</span>
                  </div>
                )}
                
                {todayMetrics.head_stability_score !== null && todayMetrics.head_stability_score !== undefined && (
                  <div className="flex items-center justify-between p-3 rounded-md bg-muted/50">
                    <div className="flex items-center gap-2">
                      <Zap className="h-4 w-4 text-purple-500" />
                      <span className="text-sm text-muted-foreground">Tremor Detection</span>
                    </div>
                    <span className="font-medium">{todayMetrics.tremor_detected ? 'Detected' : 'None'}</span>
                  </div>
                )}
                
                {todayMetrics.tongue_color_index !== null && todayMetrics.tongue_color_index !== undefined && (
                  <div className="flex items-center justify-between p-3 rounded-md bg-muted/50">
                    <div className="flex items-center gap-2">
                      <Smile className="h-4 w-4 text-pink-500" />
                      <span className="text-sm text-muted-foreground">Tongue Analysis</span>
                    </div>
                    <span className="font-medium">{todayMetrics.tongue_coating_detected ? 'Coating' : 'Normal'}</span>
                  </div>
                )}
              </div>

              {/* Quality Metrics */}
              {(todayMetrics.lighting_quality_score || todayMetrics.frames_analyzed || todayMetrics.face_detection_confidence) && (
                <div className="rounded-md border bg-muted/30 p-3 space-y-2">
                  <p className="text-sm font-medium">Analysis Quality</p>
                  <div className="grid grid-cols-3 gap-3 text-sm">
                    {todayMetrics.lighting_quality_score && (
                      <div>
                        <span className="text-muted-foreground">Lighting: </span>
                        <span className="font-medium">{todayMetrics.lighting_quality_score.toFixed(0)}/100</span>
                      </div>
                    )}
                    {todayMetrics.frames_analyzed && (
                      <div>
                        <span className="text-muted-foreground">Frames: </span>
                        <span className="font-medium">{todayMetrics.frames_analyzed}</span>
                      </div>
                    )}
                    {todayMetrics.face_detection_confidence && (
                      <div>
                        <span className="text-muted-foreground">Confidence: </span>
                        <span className="font-medium">{(todayMetrics.face_detection_confidence * 100).toFixed(0)}%</span>
                      </div>
                    )}
                  </div>
                </div>
              )}
            </CardContent>
          </Card>

          {/* Option to do another examination */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Camera className="h-5 w-5 text-primary" />
                Record Another Examination
              </CardTitle>
              <CardDescription>
                Track changes throughout the day or document specific concerns
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Button
                onClick={startExam}
                disabled={isStarting}
                className="w-full gap-2"
                data-testid="button-start-exam"
              >
                {isStarting ? (
                  <>
                    <Loader2 className="h-5 w-5 animate-spin" />
                    Starting...
                  </>
                ) : (
                  <>
                    <Play className="h-5 w-5" />
                    Start New Examination
                  </>
                )}
              </Button>
            </CardContent>
          </Card>
        </div>
      ) : (
        // No metrics today - show start examination card
        <Card className="border-primary/20">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Camera className="h-6 w-6 text-primary" />
              Start Your Daily Examination
            </CardTitle>
            <CardDescription>
              Follow step-by-step instructions to record 7 different health examinations using your camera
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            <div className="grid gap-4 md:grid-cols-2">
              <div className="space-y-3">
                <h3 className="font-semibold flex items-center gap-2">
                  <Wind className="h-4 w-4 text-blue-500" />
                  What We'll Examine
                </h3>
                <ul className="space-y-2 text-sm text-muted-foreground">
                  <li className="flex items-start gap-2">
                    <CheckCircle2 className="h-4 w-4 text-chart-2 flex-shrink-0 mt-0.5" />
                    <span>Respiratory rate, movement & pattern</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <CheckCircle2 className="h-4 w-4 text-chart-2 flex-shrink-0 mt-0.5" />
                    <span>Skin pallor & color changes</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <CheckCircle2 className="h-4 w-4 text-chart-2 flex-shrink-0 mt-0.5" />
                    <span>Eye sclera yellowness (jaundice)</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <CheckCircle2 className="h-4 w-4 text-chart-2 flex-shrink-0 mt-0.5" />
                    <span>Facial & leg swelling detection</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <CheckCircle2 className="h-4 w-4 text-chart-2 flex-shrink-0 mt-0.5" />
                    <span>Tremor & involuntary movement</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <CheckCircle2 className="h-4 w-4 text-chart-2 flex-shrink-0 mt-0.5" />
                    <span>Tongue color & coating analysis</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <CheckCircle2 className="h-4 w-4 text-chart-2 flex-shrink-0 mt-0.5" />
                    <span>Custom abnormality documentation</span>
                  </li>
                </ul>
              </div>

              <div className="space-y-3">
                <h3 className="font-semibold flex items-center gap-2">
                  <Info className="h-4 w-4 text-primary" />
                  How It Works
                </h3>
                <ul className="space-y-2 text-sm text-muted-foreground">
                  <li className="flex items-start gap-2">
                    <span className="flex-shrink-0 w-6 h-6 rounded-full bg-primary/10 text-primary flex items-center justify-center text-xs font-medium">1</span>
                    <span>Read preparation instructions for each exam</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="flex-shrink-0 w-6 h-6 rounded-full bg-primary/10 text-primary flex items-center justify-center text-xs font-medium">2</span>
                    <span>30-second countdown to prepare (skippable)</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="flex-shrink-0 w-6 h-6 rounded-full bg-primary/10 text-primary flex items-center justify-center text-xs font-medium">3</span>
                    <span>Record live video using your camera</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="flex-shrink-0 w-6 h-6 rounded-full bg-primary/10 text-primary flex items-center justify-center text-xs font-medium">4</span>
                    <span>AI analyzes each recording automatically</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="flex-shrink-0 w-6 h-6 rounded-full bg-primary/10 text-primary flex items-center justify-center text-xs font-medium">5</span>
                    <span>View detailed insights and trends</span>
                  </li>
                </ul>
              </div>
            </div>

            <Button
              onClick={startExam}
              disabled={isStarting}
              size="lg"
              className="w-full"
              data-testid="button-start-exam"
            >
              {isStarting ? (
                <>
                  <Loader2 className="h-5 w-5 mr-2 animate-spin" />
                  Starting Examination...
                </>
              ) : (
                <>
                  <Play className="h-5 w-5 mr-2" />
                  Start Guided Examination
                </>
              )}
            </Button>

            <p className="text-xs text-center text-muted-foreground">
              You can skip any examination step. Camera access is required.
            </p>
          </CardContent>
        </Card>
      )}

      {/* Link to full analysis page */}
      <Card>
        <CardContent className="p-6">
          <div className="flex items-center justify-between">
            <div className="space-y-1">
              <h3 className="font-semibold">View Complete Analysis & History</h3>
              <p className="text-sm text-muted-foreground">
                Access detailed AI insights, trends, and your examination history
              </p>
            </div>
            <Link href="/ai-video">
              <Button variant="outline" className="gap-2" data-testid="button-full-analysis-page">
                <BarChart3 className="h-4 w-4" />
                Full Analysis Page
                <ExternalLink className="h-3 w-3" />
              </Button>
            </Link>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
