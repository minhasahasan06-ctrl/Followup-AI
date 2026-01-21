import { useState, useCallback, useEffect, useRef } from "react";
import { useMutation } from "@tanstack/react-query";
import { apiRequest } from "@/lib/queryClient";

export type CallState = "idle" | "joining" | "connected" | "leaving" | "error";

export interface CallParticipant {
  id: string;
  name: string;
  isLocal: boolean;
  hasVideo: boolean;
  hasAudio: boolean;
  isSpeaking: boolean;
}

export interface CallQuality {
  video: "good" | "fair" | "poor";
  audio: "good" | "fair" | "poor";
  bitrate: number;
}

interface DailyRoom {
  url: string;
  name: string;
  token?: string;
}

interface UseDailyCallReturn {
  callState: CallState;
  participants: CallParticipant[];
  localParticipant: CallParticipant | null;
  duration: number;
  quality: CallQuality;
  isMuted: boolean;
  isCameraOff: boolean;
  isScreenSharing: boolean;
  error: string | null;
  roomUrl: string | null;
  joinCall: (roomName: string, patientId?: string) => Promise<void>;
  leaveCall: () => Promise<void>;
  toggleMute: () => void;
  toggleCamera: () => void;
  toggleScreenShare: () => void;
}

export function useDailyCall(): UseDailyCallReturn {
  const [callState, setCallState] = useState<CallState>("idle");
  const [participants, setParticipants] = useState<CallParticipant[]>([]);
  const [duration, setDuration] = useState(0);
  const [quality, setQuality] = useState<CallQuality>({
    video: "good",
    audio: "good",
    bitrate: 0,
  });
  const [isMuted, setIsMuted] = useState(false);
  const [isCameraOff, setIsCameraOff] = useState(false);
  const [isScreenSharing, setIsScreenSharing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [roomUrl, setRoomUrl] = useState<string | null>(null);

  const callFrameRef = useRef<HTMLIFrameElement | null>(null);
  const durationIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const startTimeRef = useRef<number | null>(null);

  const createRoomMutation = useMutation({
    mutationFn: async ({ roomName, patientId }: { roomName: string; patientId?: string }) => {
      const response = await apiRequest("POST", "/api/daily/rooms", {
        room_name: roomName,
        patient_id: patientId,
        privacy: "private",
        enable_recording: true,
      });
      return response.json() as Promise<DailyRoom>;
    },
  });

  const endRoomMutation = useMutation({
    mutationFn: async (roomName: string) => {
      await apiRequest("DELETE", `/api/daily/rooms/${roomName}`);
    },
  });

  const cleanup = useCallback(() => {
    if (durationIntervalRef.current) {
      clearInterval(durationIntervalRef.current);
      durationIntervalRef.current = null;
    }
    if (callFrameRef.current) {
      callFrameRef.current.remove();
      callFrameRef.current = null;
    }
    startTimeRef.current = null;
    setDuration(0);
  }, []);

  useEffect(() => {
    return cleanup;
  }, [cleanup]);

  const joinCall = useCallback(async (roomName: string, patientId?: string) => {
    try {
      setError(null);
      setCallState("joining");

      const room = await createRoomMutation.mutateAsync({ roomName, patientId });
      setRoomUrl(room.url);

      startTimeRef.current = Date.now();
      durationIntervalRef.current = setInterval(() => {
        if (startTimeRef.current) {
          setDuration(Math.floor((Date.now() - startTimeRef.current) / 1000));
        }
      }, 1000);

      setCallState("connected");
      
      setParticipants([
        {
          id: "local",
          name: "You",
          isLocal: true,
          hasVideo: !isCameraOff,
          hasAudio: !isMuted,
          isSpeaking: false,
        },
      ]);

    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : "Failed to join call";
      setError(errorMessage);
      setCallState("error");
    }
  }, [createRoomMutation, isCameraOff, isMuted]);

  const leaveCall = useCallback(async () => {
    setCallState("leaving");
    
    try {
      if (roomUrl) {
        const roomName = roomUrl.split("/").pop();
        if (roomName) {
          await endRoomMutation.mutateAsync(roomName);
        }
      }
    } catch {
    }

    cleanup();
    setRoomUrl(null);
    setParticipants([]);
    setCallState("idle");
  }, [roomUrl, endRoomMutation, cleanup]);

  const toggleMute = useCallback(() => {
    setIsMuted(prev => {
      const newValue = !prev;
      setParticipants(current =>
        current.map(p =>
          p.isLocal ? { ...p, hasAudio: !newValue } : p
        )
      );
      return newValue;
    });
  }, []);

  const toggleCamera = useCallback(() => {
    setIsCameraOff(prev => {
      const newValue = !prev;
      setParticipants(current =>
        current.map(p =>
          p.isLocal ? { ...p, hasVideo: !newValue } : p
        )
      );
      return newValue;
    });
  }, []);

  const toggleScreenShare = useCallback(() => {
    setIsScreenSharing(prev => !prev);
  }, []);

  const localParticipant = participants.find(p => p.isLocal) ?? null;

  return {
    callState,
    participants,
    localParticipant,
    duration,
    quality,
    isMuted,
    isCameraOff,
    isScreenSharing,
    error,
    roomUrl,
    joinCall,
    leaveCall,
    toggleMute,
    toggleCamera,
    toggleScreenShare,
  };
}
