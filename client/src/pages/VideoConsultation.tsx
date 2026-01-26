import { useState, useCallback } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { useParams, useLocation } from "wouter";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Video, VideoOff, Mic, MicOff, PhoneOff, Maximize2, User, Shield, Loader2 } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import { apiRequest } from "@/lib/queryClient";
import { useAuth } from "@/hooks/useAuth";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";

interface VideoSession {
  room_name: string;
  room_url: string;
  access_token: string;
  expires_at: string;
  config: {
    chat_enabled: boolean;
    recording_enabled: boolean;
    max_duration_minutes: number;
  };
}

interface ConsultationDetails {
  id: number;
  doctor_id: string;
  patient_id: string;
  status: string;
  scheduled_for?: string;
}

export default function VideoConsultation() {
  const { consultationId } = useParams<{ consultationId: string }>();
  const { user } = useAuth();
  const { toast } = useToast();
  const [, setLocation] = useLocation();
  
  const [sessionActive, setSessionActive] = useState(false);
  const [videoSession, setVideoSession] = useState<VideoSession | null>(null);
  const [isVideoOn, setIsVideoOn] = useState(true);
  const [isAudioOn, setIsAudioOn] = useState(true);
  const [isFullscreen, setIsFullscreen] = useState(false);

  const { data: consultationDetails, isLoading: loadingDetails } = useQuery<ConsultationDetails>({
    queryKey: [`/api/v1/consultations/${consultationId}/details`],
    enabled: !!consultationId,
  });

  const startSessionMutation = useMutation({
    mutationFn: async () => {
      const doctorId = user?.role === 'doctor' ? user.id : consultationDetails?.doctor_id;
      if (!doctorId) {
        throw new Error('Could not determine doctor ID for consultation');
      }
      const res = await apiRequest(`/api/consultations/${consultationId}/start-video`, {
        method: "POST",
        json: {
          doctor_id: doctorId,
          duration_minutes: 60,
          enable_recording: false
        }
      });
      return await res.json() as VideoSession;
    },
    onSuccess: (data) => {
      setVideoSession(data);
      setSessionActive(true);
      toast({
        title: "Video session started",
        description: "HIPAA-compliant video room is ready",
      });
    },
    onError: (error: Error) => {
      toast({
        title: "Failed to start video",
        description: error.message || "Could not start video consultation",
        variant: "destructive",
      });
    },
  });

  const endSessionMutation = useMutation({
    mutationFn: async () => {
      if (!videoSession?.room_name) return;
      await apiRequest(`/api/consultations/${consultationId}/end-video?room_name=${videoSession.room_name}`, { method: "DELETE" });
    },
    onSuccess: () => {
      setVideoSession(null);
      setSessionActive(false);
      toast({
        title: "Session ended",
        description: "Video consultation has been ended securely",
      });
      setLocation("/consultation-requests");
    },
    onError: (error: Error) => {
      toast({
        title: "Error ending session",
        description: error.message,
        variant: "destructive",
      });
    },
  });

  const handleStartSession = useCallback(() => {
    startSessionMutation.mutate();
  }, [startSessionMutation]);

  const handleEndSession = useCallback(() => {
    endSessionMutation.mutate();
  }, [endSessionMutation]);

  const handleToggleVideo = () => setIsVideoOn(!isVideoOn);
  const handleToggleAudio = () => setIsAudioOn(!isAudioOn);
  const handleToggleFullscreen = () => setIsFullscreen(!isFullscreen);

  if (!consultationId) {
    return (
      <div className="flex items-center justify-center h-screen">
        <Card>
          <CardHeader>
            <CardTitle>Invalid Consultation</CardTitle>
            <CardDescription>No consultation ID provided</CardDescription>
          </CardHeader>
        </Card>
      </div>
    );
  }

  return (
    <div className={`${isFullscreen ? 'fixed inset-0 z-50 bg-background' : 'space-y-6'}`}>
      {!isFullscreen && (
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-4xl font-semibold mb-2">Video Consultation</h1>
            <p className="text-muted-foreground">HIPAA-compliant secure video session</p>
          </div>
          <Badge variant="outline" className="gap-1">
            <Shield className="h-3 w-3" />
            End-to-End Encrypted
          </Badge>
        </div>
      )}

      {!sessionActive ? (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Video className="h-5 w-5" />
              Start Video Consultation
            </CardTitle>
            <CardDescription>
              Begin a secure, HIPAA-compliant video session with your {user?.role === 'doctor' ? 'patient' : 'doctor'}
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <Alert>
              <Shield className="h-4 w-4" />
              <AlertTitle>HIPAA Compliance</AlertTitle>
              <AlertDescription>
                This video session is protected by end-to-end encryption. No PHI is stored in video room URLs.
                Recording is disabled by default. Daily.co maintains a signed BAA for HIPAA compliance.
              </AlertDescription>
            </Alert>

            <div className="flex items-center gap-4 p-4 border rounded-lg">
              <div className="h-12 w-12 rounded-full bg-muted flex items-center justify-center">
                <User className="h-6 w-6" />
              </div>
              <div>
                <p className="font-medium">Consultation #{consultationId}</p>
                <p className="text-sm text-muted-foreground">Ready to start video session</p>
              </div>
            </div>

            <div className="flex gap-2">
              <Button
                onClick={handleStartSession}
                disabled={startSessionMutation.isPending || (user?.role !== 'doctor' && loadingDetails)}
                className="flex-1"
                data-testid="button-start-video"
              >
                {loadingDetails && user?.role !== 'doctor' ? (
                  <>
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                    Loading...
                  </>
                ) : (
                  <>
                    <Video className="h-4 w-4 mr-2" />
                    {startSessionMutation.isPending ? "Starting..." : "Start Video Call"}
                  </>
                )}
              </Button>
              <Button
                variant="outline"
                onClick={() => setLocation("/consultation-requests")}
                data-testid="button-back"
              >
                Back
              </Button>
            </div>
          </CardContent>
        </Card>
      ) : (
        <div className={`${isFullscreen ? 'h-full' : ''}`}>
          <Card className={`${isFullscreen ? 'h-full rounded-none' : ''}`}>
            <CardContent className={`p-0 ${isFullscreen ? 'h-full' : ''}`}>
              <div className={`relative ${isFullscreen ? 'h-full' : 'aspect-video'} bg-slate-900 rounded-t-lg overflow-hidden`}>
                {videoSession?.room_url ? (
                  <iframe
                    src={`${videoSession.room_url}?t=${videoSession.access_token}`}
                    allow="camera; microphone; fullscreen; speaker; display-capture"
                    className="w-full h-full border-0"
                    data-testid="video-frame"
                  />
                ) : (
                  <div className="flex items-center justify-center h-full text-white">
                    <Video className="h-12 w-12 animate-pulse" />
                  </div>
                )}

                <div className="absolute bottom-4 left-1/2 transform -translate-x-1/2 flex items-center gap-2 bg-black/50 backdrop-blur-sm rounded-full p-2">
                  <Button
                    size="icon"
                    variant={isVideoOn ? "secondary" : "destructive"}
                    onClick={handleToggleVideo}
                    data-testid="button-toggle-video"
                  >
                    {isVideoOn ? <Video className="h-4 w-4" /> : <VideoOff className="h-4 w-4" />}
                  </Button>
                  <Button
                    size="icon"
                    variant={isAudioOn ? "secondary" : "destructive"}
                    onClick={handleToggleAudio}
                    data-testid="button-toggle-audio"
                  >
                    {isAudioOn ? <Mic className="h-4 w-4" /> : <MicOff className="h-4 w-4" />}
                  </Button>
                  <Button
                    size="icon"
                    variant="destructive"
                    onClick={handleEndSession}
                    disabled={endSessionMutation.isPending}
                    data-testid="button-end-call"
                  >
                    <PhoneOff className="h-4 w-4" />
                  </Button>
                  <Button
                    size="icon"
                    variant="secondary"
                    onClick={handleToggleFullscreen}
                    data-testid="button-fullscreen"
                  >
                    <Maximize2 className="h-4 w-4" />
                  </Button>
                </div>
              </div>

              {!isFullscreen && (
                <div className="p-4 border-t">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-4">
                      <Badge variant="outline" className="gap-1">
                        <Shield className="h-3 w-3" />
                        Encrypted
                      </Badge>
                      {videoSession?.config.chat_enabled && (
                        <Badge variant="secondary">Chat Enabled</Badge>
                      )}
                    </div>
                    <p className="text-sm text-muted-foreground">
                      Room: {videoSession?.room_name}
                    </p>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
}
