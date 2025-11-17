import { useState } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { Progress } from "@/components/ui/progress";
import { Separator } from "@/components/ui/separator";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { useToast } from "@/hooks/use-toast";
import { apiRequest, queryClient } from "@/lib/queryClient";
import {
  Video,
  Activity,
  Eye,
  TrendingUp,
  Upload,
  PlayCircle,
  CheckCircle2,
  AlertTriangle,
  Info,
  Wind,
  Heart,
  Sparkles
} from "lucide-react";
import { LegalDisclaimer } from "@/components/LegalDisclaimer";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

interface VideoMetrics {
  session_id: number;
  respiratory_rate: number;
  skin_pallor_score: number;
  sclera_yellowness: number;
  facial_swelling_score: number;
  head_tremor_detected: boolean;
  confidence: number;
  quality_score: number;
  processing_time: number;
  timestamp: string;
}

interface VideoSession {
  session_id: number;
  s3_key: string;
  processing_status: string;
  upload_timestamp: string;
  metrics?: VideoMetrics;
}

export default function AIVideoDashboard() {
  const { toast } = useToast();
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState(false);

  // Fetch recent video sessions
  const { data: sessions, isLoading } = useQuery<VideoSession[]>({
    queryKey: ["/api/v1/video-ai/sessions/me"],
  });

  // Fetch latest metrics
  const { data: latestMetrics } = useQuery<VideoMetrics>({
    queryKey: ["/api/v1/video-ai/metrics/latest/me"],
  });

  const uploadMutation = useMutation({
    mutationFn: async (file: File) => {
      const formData = new FormData();
      formData.append("video_file", file);
      
      return apiRequest("/api/v1/video-ai/upload", {
        method: "POST",
        body: formData,
      });
    },
    onSuccess: () => {
      toast({
        title: "Video Uploaded",
        description: "Your video is being analyzed. Results will appear shortly.",
      });
      queryClient.invalidateQueries({ queryKey: ["/api/v1/video-ai/sessions/me"] });
      setSelectedFile(null);
      setIsUploading(false);
    },
    onError: () => {
      toast({
        title: "Upload Failed",
        description: "There was an error uploading your video. Please try again.",
        variant: "destructive",
      });
      setIsUploading(false);
    },
  });

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      if (file.type.startsWith("video/")) {
        setSelectedFile(file);
      } else {
        toast({
          title: "Invalid File",
          description: "Please select a video file.",
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
        <div className="grid gap-6 md:grid-cols-3">
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
          <Video className="h-10 w-10 text-primary" />
          Video AI Analysis
        </h1>
        <p className="text-muted-foreground leading-relaxed">
          AI-powered video analysis for respiratory rate, skin changes, and wellness monitoring
        </p>
      </div>

      <LegalDisclaimer />

      {/* Upload Section */}
      <Card data-testid="card-upload">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Upload className="h-5 w-5" />
            Upload Video for Analysis
          </CardTitle>
          <CardDescription>
            Record a 30-60 second video showing your face clearly in good lighting
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex items-center gap-4">
            <input
              type="file"
              accept="video/*"
              onChange={handleFileSelect}
              className="flex-1"
              data-testid="input-video-file"
            />
            <Button
              onClick={handleUpload}
              disabled={!selectedFile || isUploading}
              data-testid="button-upload-video"
            >
              {isUploading ? (
                <>
                  <Activity className="h-4 w-4 mr-2 animate-spin" />
                  Uploading...
                </>
              ) : (
                <>
                  <Upload className="h-4 w-4 mr-2" />
                  Upload & Analyze
                </>
              )}
            </Button>
          </div>
          {selectedFile && (
            <div className="text-sm text-muted-foreground">
              Selected: {selectedFile.name} ({(selectedFile.size / 1024 / 1024).toFixed(2)} MB)
            </div>
          )}
        </CardContent>
      </Card>

      {/* Latest Metrics Overview */}
      {latestMetrics && (
        <div className="grid gap-6 md:grid-cols-4">
          <Card data-testid="card-respiratory-rate">
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium flex items-center gap-2">
                <Wind className="h-4 w-4 text-blue-500" />
                Respiratory Rate
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold" data-testid="text-respiratory-rate">
                {latestMetrics.respiratory_rate.toFixed(1)}
              </div>
              <p className="text-xs text-muted-foreground mt-1">breaths/min</p>
            </CardContent>
          </Card>

          <Card data-testid="card-skin-pallor">
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium flex items-center gap-2">
                <Heart className="h-4 w-4 text-rose-500" />
                Skin Pallor
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold" data-testid="text-skin-pallor">
                {(latestMetrics.skin_pallor_score * 100).toFixed(0)}%
              </div>
              <Progress value={latestMetrics.skin_pallor_score * 100} className="mt-2" />
            </CardContent>
          </Card>

          <Card data-testid="card-sclera">
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium flex items-center gap-2">
                <Eye className="h-4 w-4 text-amber-500" />
                Eye Sclera
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold" data-testid="text-sclera">
                {(latestMetrics.sclera_yellowness * 100).toFixed(0)}%
              </div>
              <Progress value={latestMetrics.sclera_yellowness * 100} className="mt-2" />
            </CardContent>
          </Card>

          <Card data-testid="card-quality">
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium flex items-center gap-2">
                <Sparkles className="h-4 w-4 text-purple-500" />
                Analysis Quality
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold" data-testid="text-quality">
                {(latestMetrics.quality_score * 100).toFixed(0)}%
              </div>
              <Progress value={latestMetrics.quality_score * 100} className="mt-2" />
            </CardContent>
          </Card>
        </div>
      )}

      {/* Recent Sessions */}
      <Card data-testid="card-sessions">
        <CardHeader>
          <CardTitle>Analysis History</CardTitle>
          <CardDescription>Your recent video analysis sessions</CardDescription>
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
                    <PlayCircle className="h-5 w-5 text-muted-foreground" />
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
                        {session.metrics.respiratory_rate.toFixed(1)} bpm
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
              <Video className="h-16 w-16 mx-auto text-muted-foreground opacity-50 mb-4" />
              <h3 className="text-lg font-semibold mb-2">No Sessions Yet</h3>
              <p className="text-muted-foreground max-w-md mx-auto">
                Upload your first video to start tracking your health patterns with AI
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
          <p>Our AI analyzes your video to extract:</p>
          <ul className="list-disc list-inside ml-4 space-y-1">
            <li>Respiratory rate from subtle chest/facial movements</li>
            <li>Skin pallor and color changes</li>
            <li>Eye sclera yellowness (jaundice indicator)</li>
            <li>Facial swelling patterns</li>
            <li>Head tremor or movement abnormalities</li>
          </ul>
          <p className="mt-3 text-sm font-semibold">
            This is wellness monitoring only - not medical diagnosis. Discuss concerns with your healthcare provider.
          </p>
        </AlertDescription>
      </Alert>
    </div>
  );
}
