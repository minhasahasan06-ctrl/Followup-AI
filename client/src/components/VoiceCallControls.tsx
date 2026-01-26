import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { 
  Mic, 
  MicOff, 
  PhoneOff, 
  Volume2,
  Loader2,
  AlertCircle
} from "lucide-react";
import { cn } from "@/lib/utils";
import type { VoiceSessionState } from "@/hooks/useVoiceSession";

interface VoiceCallControlsProps {
  sessionState: VoiceSessionState;
  isMuted: boolean;
  agentName: string;
  error: string | null;
  onToggleMute: () => void;
  onEndCall: () => void;
  onInterrupt: () => void;
}

export function VoiceCallControls({
  sessionState,
  isMuted,
  agentName,
  error,
  onToggleMute,
  onEndCall,
  onInterrupt,
}: VoiceCallControlsProps) {
  const isActive = sessionState !== "idle" && sessionState !== "ended";

  if (!isActive) return null;

  const getStateLabel = () => {
    switch (sessionState) {
      case "connecting":
        return "Connecting...";
      case "connected":
        return "Connected";
      case "speaking":
        return `${agentName} is speaking`;
      case "listening":
        return "Listening...";
      case "error":
        return "Connection Error";
      default:
        return "";
    }
  };

  const getStateColor = () => {
    switch (sessionState) {
      case "connecting":
        return "bg-yellow-500";
      case "connected":
      case "listening":
        return "bg-green-500";
      case "speaking":
        return "bg-blue-500";
      case "error":
        return "bg-red-500";
      default:
        return "bg-gray-500";
    }
  };

  return (
    <Card className="fixed bottom-24 left-1/2 -translate-x-1/2 z-50 p-4 shadow-lg bg-background/95 backdrop-blur border-2">
      <div className="flex items-center gap-4">
        <div className="flex items-center gap-2">
          <div className={cn("w-3 h-3 rounded-full animate-pulse", getStateColor())} />
          <span className="font-medium text-sm" data-testid="text-call-state">{getStateLabel()}</span>
        </div>

        {sessionState === "speaking" && (
          <Badge variant="secondary" className="gap-1">
            <Volume2 className="h-3 w-3" />
            Speaking
          </Badge>
        )}

        {error && (
          <Badge variant="destructive" className="gap-1">
            <AlertCircle className="h-3 w-3" />
            {error}
          </Badge>
        )}

        <div className="flex items-center gap-2 ml-4">
          {sessionState === "speaking" && (
            <Button
              variant="outline"
              size="sm"
              onClick={onInterrupt}
              data-testid="button-interrupt"
            >
              Interrupt
            </Button>
          )}

          <Button
            variant={isMuted ? "destructive" : "outline"}
            size="icon"
            onClick={onToggleMute}
            disabled={sessionState === "connecting"}
            data-testid="button-toggle-mute"
          >
            {isMuted ? (
              <MicOff className="h-4 w-4" />
            ) : (
              <Mic className="h-4 w-4" />
            )}
          </Button>

          <Button
            variant="destructive"
            size="icon"
            onClick={onEndCall}
            disabled={sessionState === "connecting"}
            data-testid="button-end-call"
          >
            {sessionState === "connecting" ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <PhoneOff className="h-4 w-4" />
            )}
          </Button>
        </div>
      </div>
    </Card>
  );
}
