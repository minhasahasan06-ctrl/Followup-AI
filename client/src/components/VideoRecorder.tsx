import { useState, useRef, useEffect } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Progress } from "@/components/ui/progress";
import { apiRequest } from "@/lib/queryClient";
import { 
  Camera,
  CameraOff,
  Circle,
  Square,
  AlertTriangle,
  Loader2
} from "lucide-react";

interface VideoRecorderProps {
  examType: string;
  onRecordingComplete: (videoBlob: Blob, durationSeconds: number) => void;
  onCancel: () => void;
  maxDurationSeconds?: number;
  autoStopAfter?: number; // Auto-stop recording after N seconds
}

export function VideoRecorder({
  examType,
  onRecordingComplete,
  onCancel,
  maxDurationSeconds = 60,
  autoStopAfter
}: VideoRecorderProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const startTimeRef = useRef<number>(0);

  const [cameraStatus, setCameraStatus] = useState<'idle' | 'requesting' | 'ready' | 'error'>('idle');
  const [recordingStatus, setRecordingStatus] = useState<'idle' | 'recording' | 'stopped'>('idle');
  const [error, setError] = useState<string>('');
  const [recordingTime, setRecordingTime] = useState(0);

  // HIPAA Audit: Log camera access to backend
  const logCameraAccess = async (status: 'granted' | 'denied', errorMsg?: string) => {
    try {
      const formData = new FormData();
      formData.append('status', status);
      formData.append('exam_type', examType);
      if (errorMsg) formData.append('error_message', errorMsg);
      
      await apiRequest('/api/v1/video-ai/exam-sessions/audit/camera-access', {
        method: 'POST',
        body: formData,
      });
    } catch (error) {
      console.error('Failed to log camera access:', error);
      // Don't block the workflow if audit logging fails
    }
  };

  // Request camera access
  const startCamera = async () => {
    setCameraStatus('requesting');
    setError('');

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 },
          facingMode: 'user'
        },
        audio: false // Video only for privacy
      });

      mediaStreamRef.current = stream;
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }

      setCameraStatus('ready');
      
      // HIPAA Compliance: Log camera access granted
      await logCameraAccess('granted');
    } catch (err) {
      console.error('Camera access error:', err);
      setCameraStatus('error');
      
      let errorMessage = 'An unexpected error occurred while accessing the camera.';
      
      if (err instanceof DOMException) {
        if (err.name === 'NotAllowedError') {
          errorMessage = 'Camera access denied. Please allow camera permissions in your browser settings.';
        } else if (err.name === 'NotFoundError') {
          errorMessage = 'No camera found. Please connect a camera and try again.';
        } else {
          errorMessage = 'Failed to access camera. Please check your browser permissions.';
        }
      }
      
      setError(errorMessage);
      
      // HIPAA Compliance: Log camera access denied
      await logCameraAccess('denied', errorMessage);
    }
  };

  // Stop camera and cleanup
  const stopCamera = () => {
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach(track => track.stop());
      mediaStreamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setCameraStatus('idle');
  };

  // Start recording
  const startRecording = () => {
    if (!mediaStreamRef.current) return;

    try {
      const options = { mimeType: 'video/webm;codecs=vp9' };
      let mediaRecorder: MediaRecorder;

      // Try vp9 first, fallback to vp8, then default
      try {
        mediaRecorder = new MediaRecorder(mediaStreamRef.current, options);
      } catch {
        try {
          mediaRecorder = new MediaRecorder(mediaStreamRef.current, { mimeType: 'video/webm;codecs=vp8' });
        } catch {
          mediaRecorder = new MediaRecorder(mediaStreamRef.current);
        }
      }

      chunksRef.current = [];
      startTimeRef.current = Date.now();

      mediaRecorder.ondataavailable = (event) => {
        if (event.data && event.data.size > 0) {
          chunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: 'video/webm' });
        const durationSeconds = Math.round((Date.now() - startTimeRef.current) / 1000);
        onRecordingComplete(blob, durationSeconds);
        stopCamera();
      };

      mediaRecorderRef.current = mediaRecorder;
      mediaRecorder.start(100); // Collect data every 100ms
      setRecordingStatus('recording');
    } catch (err) {
      console.error('Recording start error:', err);
      setError('Failed to start recording. Please try again.');
    }
  };

  // Stop recording
  const stopRecording = () => {
    if (mediaRecorderRef.current && recordingStatus === 'recording') {
      mediaRecorderRef.current.stop();
      setRecordingStatus('stopped');
    }
  };

  // Recording timer
  useEffect(() => {
    if (recordingStatus !== 'recording') return;

    const interval = setInterval(() => {
      const elapsed = Math.floor((Date.now() - startTimeRef.current) / 1000);
      setRecordingTime(elapsed);

      // Auto-stop if duration limit reached
      if (elapsed >= maxDurationSeconds) {
        stopRecording();
      }

      // Auto-stop after specific duration (for exams with fixed time)
      if (autoStopAfter && elapsed >= autoStopAfter) {
        stopRecording();
      }
    }, 100);

    return () => clearInterval(interval);
  }, [recordingStatus, maxDurationSeconds, autoStopAfter]);

  // Start camera on mount
  useEffect(() => {
    startCamera();
    return () => {
      stopCamera();
    };
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (mediaRecorderRef.current && recordingStatus === 'recording') {
        mediaRecorderRef.current.stop();
      }
      stopCamera();
    };
  }, []);

  const progress = (recordingTime / maxDurationSeconds) * 100;

  return (
    <div className="space-y-4">
      {/* Camera Feed */}
      <Card className="overflow-hidden" data-testid="card-camera-feed">
        <CardContent className="p-0 relative">
          <div className="relative bg-black aspect-video">
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              className="w-full h-full object-cover"
              data-testid="video-preview"
            />

            {/* Recording Indicator */}
            {recordingStatus === 'recording' && (
              <div className="absolute top-4 left-4 flex items-center gap-2 bg-red-500 text-white px-3 py-1.5 rounded-full animate-pulse">
                <Circle className="h-3 w-3 fill-current" />
                <span className="text-sm font-medium">REC {recordingTime}s</span>
              </div>
            )}

            {/* Camera Status Overlay */}
            {cameraStatus !== 'ready' && (
              <div className="absolute inset-0 flex items-center justify-center bg-black/80">
                {cameraStatus === 'requesting' && (
                  <div className="text-center text-white">
                    <Loader2 className="h-12 w-12 animate-spin mx-auto mb-4" />
                    <p>Requesting camera access...</p>
                  </div>
                )}
                {cameraStatus === 'error' && (
                  <div className="text-center text-white p-6">
                    <CameraOff className="h-12 w-12 mx-auto mb-4 text-red-400" />
                    <p className="text-red-400 font-medium">Camera Error</p>
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Recording Progress */}
          {recordingStatus === 'recording' && (
            <div className="p-4 bg-muted">
              <div className="flex items-center justify-between text-sm mb-2">
                <span>Recording {examType} examination...</span>
                <span className="font-medium">
                  {recordingTime}s / {maxDurationSeconds}s
                </span>
              </div>
              <Progress value={progress} className="h-2" />
              {autoStopAfter && (
                <p className="text-xs text-muted-foreground mt-2">
                  Recording will stop automatically after {autoStopAfter} seconds
                </p>
              )}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Error Alert */}
      {error && (
        <Alert variant="destructive" data-testid="alert-error">
          <AlertTriangle className="h-4 w-4" />
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {/* Controls */}
      <div className="flex gap-3">
        {recordingStatus === 'idle' && cameraStatus === 'ready' && (
          <Button
            onClick={startRecording}
            size="lg"
            className="flex-1"
            data-testid="button-start-recording"
          >
            <Circle className="h-5 w-5 mr-2" />
            Start Recording
          </Button>
        )}

        {recordingStatus === 'recording' && (
          <Button
            onClick={stopRecording}
            size="lg"
            variant="destructive"
            className="flex-1"
            data-testid="button-stop-recording"
          >
            <Square className="h-5 w-5 mr-2" />
            Stop Recording
          </Button>
        )}

        <Button
          onClick={() => {
            stopCamera();
            onCancel();
          }}
          variant="outline"
          size="lg"
          data-testid="button-cancel-recording"
        >
          Cancel
        </Button>
      </div>

      {/* HIPAA Notice */}
      <div className="text-xs text-center text-muted-foreground p-3 bg-muted/30 rounded-lg">
        <AlertTriangle className="h-3 w-3 inline mr-1" />
        Your video is encrypted and securely stored. Camera access is required for this examination.
      </div>
    </div>
  );
}
