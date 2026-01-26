import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import {
  Mic,
  MicOff,
  Video,
  VideoOff,
  PhoneOff,
  Monitor,
  MonitorOff,
  Maximize2,
  Minimize2,
  Settings,
  Users,
  Signal,
  Loader2,
  AlertCircle,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { useState } from "react";
import type { CallState, CallParticipant, CallQuality } from "@/hooks/useDailyCall";

interface VideoConsoleProps {
  callState: CallState;
  participants: CallParticipant[];
  duration: number;
  quality: CallQuality;
  isMuted: boolean;
  isCameraOff: boolean;
  isScreenSharing: boolean;
  roomUrl: string | null;
  error: string | null;
  patientName?: string;
  onToggleMute: () => void;
  onToggleCamera: () => void;
  onToggleScreenShare: () => void;
  onEndCall: () => void;
}

function formatDuration(seconds: number): string {
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const secs = seconds % 60;

  if (hours > 0) {
    return `${hours}:${minutes.toString().padStart(2, "0")}:${secs.toString().padStart(2, "0")}`;
  }
  return `${minutes}:${secs.toString().padStart(2, "0")}`;
}

function getQualityColor(quality: "good" | "fair" | "poor"): string {
  switch (quality) {
    case "good":
      return "text-green-500";
    case "fair":
      return "text-yellow-500";
    case "poor":
      return "text-red-500";
  }
}

function getQualityBars(quality: "good" | "fair" | "poor"): number {
  switch (quality) {
    case "good":
      return 4;
    case "fair":
      return 2;
    case "poor":
      return 1;
  }
}

export function VideoConsole({
  callState,
  participants,
  duration,
  quality,
  isMuted,
  isCameraOff,
  isScreenSharing,
  roomUrl,
  error,
  patientName = "Patient",
  onToggleMute,
  onToggleCamera,
  onToggleScreenShare,
  onEndCall,
}: VideoConsoleProps) {
  const [isFullscreen, setIsFullscreen] = useState(false);
  const isActive = callState === "connected" || callState === "joining";

  if (!isActive && callState !== "leaving") return null;

  const remoteParticipants = participants.filter(p => !p.isLocal);

  return (
    <div
      className={cn(
        "fixed z-50 bg-background border rounded-lg shadow-2xl overflow-hidden",
        isFullscreen
          ? "inset-0 rounded-none"
          : "bottom-4 right-4 w-[480px] h-[360px]"
      )}
      data-testid="video-console"
    >
      <div className="relative h-full flex flex-col">
        <div className="absolute top-0 left-0 right-0 z-10 flex items-center justify-between p-3 bg-gradient-to-b from-black/70 to-transparent">
          <div className="flex items-center gap-2">
            <Badge variant="secondary" className="gap-1 bg-black/50 text-white border-0">
              <Users className="h-3 w-3" />
              {participants.length}
            </Badge>
            <Badge variant="secondary" className="bg-black/50 text-white border-0">
              {formatDuration(duration)}
            </Badge>
            <div className={cn("flex items-center gap-0.5", getQualityColor(quality.video))}>
              <Signal className="h-3 w-3" />
              {[...Array(4)].map((_, i) => (
                <div
                  key={i}
                  className={cn(
                    "w-0.5 rounded-sm",
                    i < getQualityBars(quality.video)
                      ? getQualityColor(quality.video).replace("text-", "bg-")
                      : "bg-gray-400"
                  )}
                  style={{ height: `${(i + 1) * 3}px` }}
                />
              ))}
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Button
              variant="ghost"
              size="icon"
              className="h-8 w-8 text-white hover:bg-white/20"
              onClick={() => setIsFullscreen(prev => !prev)}
              data-testid="button-fullscreen"
            >
              {isFullscreen ? (
                <Minimize2 className="h-4 w-4" />
              ) : (
                <Maximize2 className="h-4 w-4" />
              )}
            </Button>
          </div>
        </div>

        <div className="flex-1 bg-gray-900 flex items-center justify-center">
          {callState === "joining" ? (
            <div className="text-center text-white">
              <Loader2 className="h-8 w-8 animate-spin mx-auto mb-2" />
              <p className="text-sm">Connecting...</p>
            </div>
          ) : error ? (
            <div className="text-center text-white">
              <AlertCircle className="h-8 w-8 text-red-500 mx-auto mb-2" />
              <p className="text-sm text-red-400">{error}</p>
            </div>
          ) : roomUrl ? (
            <div className="w-full h-full relative">
              <div className="absolute inset-0 flex items-center justify-center">
                <Avatar className="h-24 w-24">
                  <AvatarImage src="" />
                  <AvatarFallback className="text-2xl bg-primary text-primary-foreground">
                    {patientName.charAt(0).toUpperCase()}
                  </AvatarFallback>
                </Avatar>
              </div>
              {remoteParticipants.length === 0 && (
                <div className="absolute bottom-16 left-1/2 -translate-x-1/2 text-white text-sm">
                  Waiting for {patientName} to join...
                </div>
              )}
              <div className="absolute bottom-20 right-4 w-32 h-24 bg-gray-800 rounded-lg overflow-hidden border-2 border-gray-700">
                <div className="w-full h-full flex items-center justify-center">
                  {isCameraOff ? (
                    <div className="text-center text-gray-400">
                      <VideoOff className="h-6 w-6 mx-auto mb-1" />
                      <span className="text-xs">Camera Off</span>
                    </div>
                  ) : (
                    <Avatar className="h-12 w-12">
                      <AvatarFallback className="bg-blue-600 text-white">You</AvatarFallback>
                    </Avatar>
                  )}
                </div>
              </div>
            </div>
          ) : null}
        </div>

        <div className="absolute bottom-0 left-0 right-0 z-10 flex items-center justify-center gap-3 p-4 bg-gradient-to-t from-black/70 to-transparent">
          <Button
            variant={isMuted ? "destructive" : "secondary"}
            size="icon"
            className="h-12 w-12 rounded-full"
            onClick={onToggleMute}
            data-testid="button-console-mute"
          >
            {isMuted ? (
              <MicOff className="h-5 w-5" />
            ) : (
              <Mic className="h-5 w-5" />
            )}
          </Button>

          <Button
            variant={isCameraOff ? "destructive" : "secondary"}
            size="icon"
            className="h-12 w-12 rounded-full"
            onClick={onToggleCamera}
            data-testid="button-console-camera"
          >
            {isCameraOff ? (
              <VideoOff className="h-5 w-5" />
            ) : (
              <Video className="h-5 w-5" />
            )}
          </Button>

          <Button
            variant={isScreenSharing ? "default" : "secondary"}
            size="icon"
            className="h-12 w-12 rounded-full"
            onClick={onToggleScreenShare}
            data-testid="button-console-screen"
          >
            {isScreenSharing ? (
              <MonitorOff className="h-5 w-5" />
            ) : (
              <Monitor className="h-5 w-5" />
            )}
          </Button>

          <Button
            variant="destructive"
            size="icon"
            className="h-12 w-12 rounded-full"
            onClick={onEndCall}
            data-testid="button-console-end"
          >
            <PhoneOff className="h-5 w-5" />
          </Button>

          <Button
            variant="secondary"
            size="icon"
            className="h-12 w-12 rounded-full"
            data-testid="button-console-settings"
          >
            <Settings className="h-5 w-5" />
          </Button>
        </div>
      </div>
    </div>
  );
}
