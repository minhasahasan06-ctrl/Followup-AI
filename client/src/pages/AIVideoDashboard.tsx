import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Label } from '@/components/ui/label';
import {
  Video,
  Camera,
  Play,
  CheckCircle2,
  Clock,
  AlertCircle,
  Info,
  BarChart3,
  Calendar,
  Loader2,
  Wind,
  Hand,
  Eye,
  User,
  Activity,
  MessageSquare,
  XCircle,
} from 'lucide-react';
import { ExamPrepStep } from '@/components/ExamPrepStep';
import { VideoRecorder } from '@/components/VideoRecorder';
import { useGuidedExamWorkflow } from '@/hooks/useGuidedExamWorkflow';

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

export default function AIVideoDashboard() {
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

  // Fetch recent sessions (last 15 days)
  const { data: recentSessions } = useQuery({
    queryKey: ['/api/v1/video-ai/exam-sessions', { days: 15 }],
    enabled: workflowState === 'idle' || workflowState === 'completed',
  });

  // Fetch AI analysis insights
  const { data: analysisInsights } = useQuery({
    queryKey: ['/api/v1/video-ai/analysis/insights', { sessionId }],
    enabled: !!sessionId && workflowState === 'completed',
  });

  const handleRecordingComplete = (videoBlob: Blob, durationSeconds: number) => {
    if (currentStep?.examType === 'custom') {
      completeRecording(videoBlob, durationSeconds, {
        location: customLocation,
        description: customDescription,
      });
      // Reset custom fields
      setCustomLocation('');
      setCustomDescription('');
    } else {
      completeRecording(videoBlob, durationSeconds);
    }
  };

  // Idle state - show welcome screen
  if (workflowState === 'idle') {
    return (
      <div className="container mx-auto p-6 max-w-7xl space-y-6">
        {/* Header */}
        <div className="space-y-2">
          <h1 className="text-4xl font-bold tracking-tight text-foreground flex items-center gap-3" data-testid="text-page-title">
            <Video className="h-10 w-10 text-primary" />
            Guided Video Examination
          </h1>
          <p className="text-muted-foreground leading-relaxed">
            AI-powered guided examination for comprehensive health monitoring
          </p>
        </div>

        <LegalDisclaimer />

        {/* Welcome Card */}
        <Card className="border-primary/20">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Camera className="h-6 w-6 text-primary" />
              Start Your Guided Examination
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
                    <span>Skin pallor & color changes (palms/soles)</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <CheckCircle2 className="h-4 w-4 text-chart-2 flex-shrink-0 mt-0.5" />
                    <span>Eye sclera yellowness (jaundice indicator)</span>
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

        {/* Recent Sessions & Analysis - Bottom Panels */}
        <Tabs defaultValue="recent" className="w-full">
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="recent">
              <Calendar className="h-4 w-4 mr-2" />
              Last 15 Days
            </TabsTrigger>
            <TabsTrigger value="insights">
              <BarChart3 className="h-4 w-4 mr-2" />
              AI Analysis
            </TabsTrigger>
          </TabsList>

          <TabsContent value="recent" className="space-y-3 mt-4">
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Recent Examination Sessions</CardTitle>
                <CardDescription>Your video recordings from the past 15 days</CardDescription>
              </CardHeader>
              <CardContent>
                {recentSessions && recentSessions.length > 0 ? (
                  <div className="space-y-2">
                    {recentSessions.map((session: any) => (
                      <div
                        key={session.id}
                        className="flex items-center justify-between p-3 rounded-lg border bg-card hover-elevate"
                      >
                        <div className="flex items-center gap-3">
                          <Video className="h-5 w-5 text-muted-foreground" />
                          <div>
                            <div className="font-medium text-sm">
                              {new Date(session.started_at).toLocaleDateString()}
                            </div>
                            <div className="text-xs text-muted-foreground">
                              {session.completed_segments} of {session.total_segments} exams completed
                            </div>
                          </div>
                        </div>
                        <Badge variant={session.status === 'completed' ? 'default' : 'secondary'}>
                          {session.status}
                        </Badge>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="text-center py-12">
                    <Video className="h-16 w-16 mx-auto text-muted-foreground opacity-50 mb-4" />
                    <h3 className="text-lg font-semibold mb-2">No Sessions Yet</h3>
                    <p className="text-muted-foreground max-w-md mx-auto">
                      Start your first guided examination to begin tracking your health patterns
                    </p>
                  </div>
                )}
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="insights" className="space-y-3 mt-4">
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">AI Video Analysis Insights</CardTitle>
                <CardDescription>Detailed examination reports with trends (Today, 7 days, 30 days, 1 year)</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="text-center py-12">
                  <BarChart3 className="h-16 w-16 mx-auto text-muted-foreground opacity-50 mb-4" />
                  <h3 className="text-lg font-semibold mb-2">Analysis Coming Soon</h3>
                  <p className="text-muted-foreground max-w-md mx-auto">
                    Complete your first examination to view AI-powered insights and trends
                  </p>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    );
  }

  // Active examination workflow
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
          {/* Custom exam input */}
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
              Start New Examination
            </Button>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
