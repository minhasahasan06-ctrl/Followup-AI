import { useState } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { Progress } from "@/components/ui/progress";
import { useToast } from "@/hooks/use-toast";
import { apiRequest, queryClient } from "@/lib/queryClient";
import {
  Mic,
  Activity,
  Upload,
  Waves,
  Wind,
  AlertCircle,
  Info,
  TrendingUp,
  Volume2,
  Sparkles,
  Play
} from "lucide-react";
import { LegalDisclaimer } from "@/components/LegalDisclaimer";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Link } from "wouter";

interface AudioMetrics {
  session_id: number;
  breath_cycles_per_min: number;
  speech_pace_wpm: number;
  cough_detected: boolean;
  wheeze_detected: boolean;
  voice_hoarseness_score: number;
  confidence: number;
  quality_score: number;
  processing_time: number;
  timestamp: string;
}

interface AudioSession {
  session_id: number;
  s3_key: string;
  processing_status: string;
  upload_timestamp: string;
  metrics?: AudioMetrics;
}

export default function AIAudioDashboard() {
  const { toast } = useToast();
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState(false);

  const { data: sessions, isLoading } = useQuery<AudioSession[]>({
    queryKey: ["/api/v1/audio-ai/sessions/me"],
  });

  const { data: latestMetrics } = useQuery<AudioMetrics>({
    queryKey: ["/api/v1/audio-ai/metrics/latest/me"],
  });

  const uploadMutation = useMutation({
    mutationFn: async (file: File) => {
      const formData = new FormData();
      formData.append("audio_file", file);
      
      return apiRequest("/api/v1/audio-ai/upload", {
        method: "POST",
        body: formData,
      });
    },
    onSuccess: () => {
      toast({
        title: "Audio Uploaded",
        description: "Your audio is being analyzed. Results will appear shortly.",
      });
      queryClient.invalidateQueries({ queryKey: ["/api/v1/audio-ai/sessions/me"] });
      setSelectedFile(null);
      setIsUploading(false);
    },
    onError: () => {
      toast({
        title: "Upload Failed",
        description: "There was an error uploading your audio. Please try again.",
        variant: "destructive",
      });
      setIsUploading(false);
    },
  });

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      if (file.type.startsWith("audio/")) {
        setSelectedFile(file);
      } else {
        toast({
          title: "Invalid File",
          description: "Please select an audio file.",
          variant: "destructive",
        });
      }
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) return;
    setIsUploading(true);
    uploadMutation.mutate(selectedFile);
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case "completed":
        return "bg-teal-50 text-teal-700 border-teal-200 dark:bg-teal-950 dark:text-teal-300";
      case "processing":
        return "bg-blue-50 text-blue-700 border-blue-200 dark:bg-blue-950 dark:text-blue-300";
      case "failed":
        return "bg-rose-50 text-rose-700 border-rose-200 dark:bg-rose-950 dark:text-rose-300";
      default:
        return "bg-slate-50 text-slate-700 border-slate-200";
    }
  };

  if (isLoading) {
    return (
      <div className="container mx-auto p-6 max-w-7xl space-y-6">
        <Skeleton className="h-12 w-96" />
        <div className="grid gap-6 md:grid-cols-4">
          <Skeleton className="h-48" />
          <Skeleton className="h-48" />
          <Skeleton className="h-48" />
          <Skeleton className="h-48" />
        </div>
      </div>
    );
  }

  return (
    <div className="container mx-auto p-6 max-w-7xl space-y-6">
      {/* Header */}
      <div className="space-y-2">
        <h1 className="text-4xl font-bold tracking-tight text-foreground flex items-center gap-3" data-testid="text-page-title">
          <Mic className="h-10 w-10 text-primary" />
          Audio AI Analysis
        </h1>
        <p className="text-muted-foreground leading-relaxed">
          Start live AI-guided examinations with real-time voice directions, or view your analysis history
        </p>
      </div>

      <LegalDisclaimer />

      {/* Live Examination Section */}
      <Card className="bg-gradient-to-br from-primary/5 to-primary/10" data-testid="card-live-exam">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Mic className="h-5 w-5" />
            Live Audio Examination
          </CardTitle>
          <CardDescription>
            AI-guided examination with real-time directions for breathing, coughing, speaking, and reading
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="rounded-lg border bg-background/50 p-4 space-y-3">
            <div className="flex items-center gap-2">
              <div className="rounded-full bg-primary/10 p-2">
                <Play className="h-4 w-4 text-primary" />
              </div>
              <div>
                <p className="font-medium text-sm">4-Stage Audio Examination</p>
                <p className="text-xs text-muted-foreground">Breathing • Coughing • Speaking • Reading</p>
              </div>
            </div>
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
              <Volume2 className="h-4 w-4" />
              <span>Voice-guided instructions • Real-time recording • AI analysis</span>
            </div>
          </div>
          
          <Link href="/guided-audio-exam">
            <Button className="w-full gap-2" size="lg" data-testid="button-start-live-exam">
              <Play className="h-5 w-5" />
              Start Live Examination
            </Button>
          </Link>
          
          <p className="text-xs text-center text-muted-foreground">
            Allow microphone access when prompted. The examination takes ~2-3 minutes.
          </p>
        </CardContent>
      </Card>

      {/* Latest Metrics Overview */}
      {latestMetrics && (
        <div className="grid gap-6 md:grid-cols-4">
          <Card data-testid="card-breath-cycles">
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium flex items-center gap-2">
                <Wind className="h-4 w-4 text-blue-500" />
                Breath Cycles
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold" data-testid="text-breath-cycles">
                {latestMetrics.breath_cycles_per_min.toFixed(1)}
              </div>
              <p className="text-xs text-muted-foreground mt-1">cycles/min</p>
            </CardContent>
          </Card>

          <Card data-testid="card-speech-pace">
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium flex items-center gap-2">
                <Volume2 className="h-4 w-4 text-purple-500" />
                Speech Pace
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold" data-testid="text-speech-pace">
                {latestMetrics.speech_pace_wpm.toFixed(0)}
              </div>
              <p className="text-xs text-muted-foreground mt-1">words/min</p>
            </CardContent>
          </Card>

          <Card data-testid="card-voice-quality">
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium flex items-center gap-2">
                <Waves className="h-4 w-4 text-amber-500" />
                Voice Hoarseness
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold" data-testid="text-voice-hoarseness">
                {(latestMetrics.voice_hoarseness_score * 100).toFixed(0)}%
              </div>
              <Progress value={latestMetrics.voice_hoarseness_score * 100} className="mt-2" />
            </CardContent>
          </Card>

          <Card data-testid="card-respiratory-alerts">
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium flex items-center gap-2">
                <AlertCircle className="h-4 w-4 text-rose-500" />
                Respiratory Alerts
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-sm">Cough</span>
                <Badge variant={latestMetrics.cough_detected ? "destructive" : "secondary"}>
                  {latestMetrics.cough_detected ? "Detected" : "None"}
                </Badge>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm">Wheeze</span>
                <Badge variant={latestMetrics.wheeze_detected ? "destructive" : "secondary"}>
                  {latestMetrics.wheeze_detected ? "Detected" : "None"}
                </Badge>
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Recent Sessions */}
      <Card data-testid="card-sessions">
        <CardHeader>
          <CardTitle>Analysis History</CardTitle>
          <CardDescription>Your recent audio analysis sessions</CardDescription>
        </CardHeader>
        <CardContent>
          {sessions && sessions.length > 0 ? (
            <div className="space-y-3">
              {sessions.map((session) => (
                <div
                  key={session.session_id}
                  className="flex items-center justify-between p-4 rounded-lg border bg-card hover-elevate"
                  data-testid={`session-${session.session_id}`}
                >
                  <div className="flex items-center gap-3">
                    <Mic className="h-5 w-5 text-muted-foreground" />
                    <div>
                      <div className="font-medium text-sm">
                        Session #{session.session_id}
                      </div>
                      <div className="text-xs text-muted-foreground">
                        {new Date(session.upload_timestamp).toLocaleString()}
                      </div>
                    </div>
                  </div>
                  <div className="flex items-center gap-3">
                    {session.metrics && (
                      <div className="text-sm text-muted-foreground">
                        {session.metrics.breath_cycles_per_min.toFixed(1)} cycles/min
                      </div>
                    )}
                    <Badge className={getStatusColor(session.processing_status)}>
                      {session.processing_status}
                    </Badge>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center py-12" data-testid="empty-state">
              <Mic className="h-16 w-16 mx-auto text-muted-foreground opacity-50 mb-4" />
              <h3 className="text-lg font-semibold mb-2">No Sessions Yet</h3>
              <p className="text-muted-foreground max-w-md mx-auto">
                Upload your first audio recording to start tracking respiratory health patterns
              </p>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Information Alert */}
      <Alert data-testid="alert-info">
        <Info className="h-4 w-4" />
        <AlertTitle>How It Works</AlertTitle>
        <AlertDescription className="space-y-2">
          <p>Our AI analyzes your audio to extract:</p>
          <ul className="list-disc list-inside ml-4 space-y-1">
            <li>Breath cycles and respiratory rate patterns</li>
            <li>Speech pace and articulation changes</li>
            <li>Cough detection and frequency</li>
            <li>Wheeze detection (high-frequency respiratory sounds)</li>
            <li>Voice quality and hoarseness indicators</li>
          </ul>
          <p className="mt-3 text-sm font-semibold">
            This is wellness monitoring only - not medical diagnosis. Discuss concerns with your healthcare provider.
          </p>
        </AlertDescription>
      </Alert>
    </div>
  );
}
