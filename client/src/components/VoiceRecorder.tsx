import { useState, useRef, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Mic, Square, Upload, Loader2, Heart, AlertCircle } from "lucide-react";
import { Progress } from "@/components/ui/progress";
import { useMutation } from "@tanstack/react-query";
import { apiRequest } from "@/lib/queryClient";
import { useToast } from "@/hooks/use-toast";

interface VoiceRecorderProps {
  onUploadSuccess?: (result: any) => void;
  maxDuration?: number; // seconds
  showResponse?: boolean;
}

export function VoiceRecorder({
  onUploadSuccess,
  maxDuration = 60,
  showResponse = true,
}: VoiceRecorderProps) {
  const { toast } = useToast();
  const [isRecording, setIsRecording] = useState(false);
  const [recordingTime, setRecordingTime] = useState(0);
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [uploadResult, setUploadResult] = useState<any>(null);
  
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<NodeJS.Timeout | null>(null);

  const uploadMutation = useMutation({
    mutationFn: async (audioBlob: Blob) => {
      const formData = new FormData();
      formData.append("audio", audioBlob, `voice-followup-${Date.now()}.webm`);
      
      const response = await fetch("/api/voice-followup/upload", {
        method: "POST",
        body: formData,
        credentials: "include",
      });

      if (!response.ok) {
        throw new Error("Upload failed");
      }

      return response.json();
    },
    onSuccess: (data) => {
      setUploadResult(data);
      if (onUploadSuccess) {
        onUploadSuccess(data);
      }
      toast({
        title: "Voice Log Saved",
        description: "Your health companion has processed your recording.",
      });
    },
    onError: () => {
      toast({
        title: "Upload Failed",
        description: "Unable to process your recording. Please try again.",
        variant: "destructive",
      });
    },
  });

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: "audio/webm",
      });

      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = () => {
        const blob = new Blob(audioChunksRef.current, { type: "audio/webm" });
        setAudioBlob(blob);
        setAudioUrl(URL.createObjectURL(blob));
        
        // Stop all tracks
        stream.getTracks().forEach((track) => track.stop());
      };

      mediaRecorder.start();
      setIsRecording(true);
      setRecordingTime(0);
      setUploadResult(null);

      // Start timer
      timerRef.current = setInterval(() => {
        setRecordingTime((prev) => {
          const newTime = prev + 1;
          if (newTime >= maxDuration) {
            stopRecording();
            return maxDuration;
          }
          return newTime;
        });
      }, 1000);
    } catch (error) {
      console.error("Error accessing microphone:", error);
      toast({
        title: "Microphone Access Denied",
        description: "Please allow microphone access to record voice messages.",
        variant: "destructive",
      });
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== "inactive") {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      
      if (timerRef.current) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }
    }
  };

  const handleUpload = () => {
    if (audioBlob) {
      uploadMutation.mutate(audioBlob);
    }
  };

  const handleReset = () => {
    setAudioBlob(null);
    setAudioUrl(null);
    setRecordingTime(0);
    setUploadResult(null);
    if (audioUrl) {
      URL.revokeObjectURL(audioUrl);
    }
  };

  useEffect(() => {
    return () => {
      if (timerRef.current) {
        clearInterval(timerRef.current);
      }
      if (audioUrl) {
        URL.revokeObjectURL(audioUrl);
      }
    };
  }, [audioUrl]);

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, "0")}`;
  };

  const progressPercentage = (recordingTime / maxDuration) * 100;

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-full bg-gradient-to-br from-pink-400 to-red-500 text-white">
              <Mic className="h-5 w-5" />
            </div>
            <div>
              <CardTitle>Voice Check-in</CardTitle>
              <CardDescription>
                Record a quick 1-minute voice message about how you're feeling
              </CardDescription>
            </div>
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          {!audioBlob && !isRecording && (
            <div className="text-center py-8">
              <Button
                size="lg"
                onClick={startRecording}
                className="h-16 w-16 rounded-full"
                data-testid="button-start-recording"
              >
                <Mic className="h-6 w-6" />
              </Button>
              <p className="text-sm text-muted-foreground mt-4">
                Tap to start recording (up to {maxDuration} seconds)
              </p>
            </div>
          )}

          {isRecording && (
            <div className="space-y-4">
              <div className="text-center py-4">
                <div className="flex items-center justify-center gap-3 mb-4">
                  <div className="h-3 w-3 rounded-full bg-red-500 animate-pulse" />
                  <span className="text-2xl font-mono font-bold" data-testid="text-recording-time">
                    {formatTime(recordingTime)}
                  </span>
                </div>
                <Progress value={progressPercentage} className="h-2" />
                <p className="text-xs text-muted-foreground mt-2">
                  {maxDuration - recordingTime} seconds remaining
                </p>
              </div>

              <div className="flex justify-center">
                <Button
                  variant="destructive"
                  size="lg"
                  onClick={stopRecording}
                  className="h-16 w-16 rounded-full"
                  data-testid="button-stop-recording"
                >
                  <Square className="h-6 w-6" />
                </Button>
              </div>
            </div>
          )}

          {audioBlob && !uploadResult && (
            <div className="space-y-4">
              <div className="p-4 bg-muted rounded-lg">
                <p className="text-sm font-medium mb-2">Recording Complete</p>
                <audio src={audioUrl || ""} controls className="w-full" data-testid="audio-player" />
                <p className="text-xs text-muted-foreground mt-2">
                  Duration: {formatTime(recordingTime)}
                </p>
              </div>

              <div className="flex gap-2">
                <Button
                  onClick={handleUpload}
                  disabled={uploadMutation.isPending}
                  className="flex-1"
                  data-testid="button-upload"
                >
                  {uploadMutation.isPending ? (
                    <>
                      <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                      Processing...
                    </>
                  ) : (
                    <>
                      <Upload className="w-4 h-4 mr-2" />
                      Submit Voice Log
                    </>
                  )}
                </Button>
                <Button
                  variant="outline"
                  onClick={handleReset}
                  disabled={uploadMutation.isPending}
                  data-testid="button-reset"
                >
                  Re-record
                </Button>
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {showResponse && uploadResult && (
        <Card className="border-green-200 dark:border-green-800">
          <CardHeader className="pb-3">
            <div className="flex items-center gap-2">
              <Heart className="w-5 h-5 text-green-600 dark:text-green-400" />
              <CardTitle className="text-base">Your Health Companion's Response</CardTitle>
            </div>
          </CardHeader>
          <CardContent className="space-y-4">
            <div>
              <p className="text-sm leading-relaxed mb-3" data-testid="text-ai-response">
                {uploadResult.response}
              </p>
            </div>

            <div className="flex flex-wrap items-center gap-2">
              <Badge variant="outline" className="capitalize">
                {uploadResult.empathyLevel}
              </Badge>
              {uploadResult.concernsRaised && (
                <Badge variant="outline" className="bg-yellow-50 dark:bg-yellow-950">
                  <AlertCircle className="w-3 h-3 mr-1" />
                  Concerns Noted
                </Badge>
              )}
              {uploadResult.needsFollowup && (
                <Badge variant="outline" className="bg-blue-50 dark:bg-blue-950">
                  Follow-up Recommended
                </Badge>
              )}
            </div>

            {uploadResult.conversationSummary && (
              <div className="p-3 bg-muted rounded-lg text-sm">
                <p className="font-medium mb-1">Summary:</p>
                <p className="text-muted-foreground">{uploadResult.conversationSummary}</p>
              </div>
            )}

            {uploadResult.extractedSymptoms && uploadResult.extractedSymptoms.length > 0 && (
              <div className="p-3 bg-muted rounded-lg text-sm">
                <p className="font-medium mb-2">Symptoms Mentioned:</p>
                <div className="flex flex-wrap gap-1">
                  {uploadResult.extractedSymptoms.map((symptom: any, idx: number) => (
                    <Badge key={idx} variant="secondary" className="text-xs">
                      {symptom.symptom} ({symptom.severity})
                    </Badge>
                  ))}
                </div>
              </div>
            )}

            {uploadResult.recommendedActions && uploadResult.recommendedActions.length > 0 && (
              <div className="p-3 bg-blue-50 dark:bg-blue-950 rounded-lg text-sm">
                <p className="font-medium mb-2">Recommended Actions:</p>
                <ul className="list-disc list-inside space-y-1">
                  {uploadResult.recommendedActions.map((action: string, idx: number) => (
                    <li key={idx} className="text-muted-foreground">
                      {action}
                    </li>
                  ))}
                </ul>
              </div>
            )}

            <Button onClick={handleReset} variant="outline" className="w-full" data-testid="button-new-recording">
              Record Another Message
            </Button>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
