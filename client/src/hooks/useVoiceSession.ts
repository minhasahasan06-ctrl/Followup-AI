import { useState, useRef, useCallback, useEffect } from "react";
import { useMutation } from "@tanstack/react-query";
import { apiRequest } from "@/lib/queryClient";

export type VoiceSessionState = "idle" | "connecting" | "connected" | "speaking" | "listening" | "error" | "ended";

interface VoiceSession {
  session_id: string;
  state: string;
  agent_type: string;
  created_at: string;
}

interface VoiceSessionHook {
  sessionState: VoiceSessionState;
  sessionId: string | null;
  isMuted: boolean;
  isConnecting: boolean;
  error: string | null;
  startSession: (agentType: "clona" | "lysa") => Promise<void>;
  endSession: () => Promise<void>;
  toggleMute: () => void;
  interrupt: () => Promise<void>;
}

export function useVoiceSession(): VoiceSessionHook {
  const [sessionState, setSessionState] = useState<VoiceSessionState>("idle");
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [isMuted, setIsMuted] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  const wsRef = useRef<WebSocket | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);

  const cleanup = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach(track => track.stop());
      mediaStreamRef.current = null;
    }
    if (audioContextRef.current) {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }
  }, []);

  useEffect(() => {
    return cleanup;
  }, [cleanup]);

  const createSessionMutation = useMutation({
    mutationFn: async (agentType: "clona" | "lysa") => {
      const response = await apiRequest("POST", "/api/voice/sessions", {
        agent_type: agentType,
      });
      return response.json() as Promise<VoiceSession>;
    },
  });

  const endSessionMutation = useMutation({
    mutationFn: async (id: string) => {
      await apiRequest("DELETE", `/api/voice/sessions/${id}`);
    },
  });

  const interruptMutation = useMutation({
    mutationFn: async (id: string) => {
      await apiRequest("POST", `/api/voice/sessions/${id}/interrupt`);
    },
  });

  const startSession = useCallback(async (agentType: "clona" | "lysa") => {
    try {
      setError(null);
      setSessionState("connecting");

      const session = await createSessionMutation.mutateAsync(agentType);
      setSessionId(session.session_id);

      try {
        mediaStreamRef.current = await navigator.mediaDevices.getUserMedia({ 
          audio: {
            echoCancellation: true,
            noiseSuppression: true,
            autoGainControl: true,
          } 
        });
      } catch (mediaError) {
        setError("Microphone access denied. Please allow microphone access to use voice features.");
        setSessionState("error");
        return;
      }

      const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
      const wsUrl = `${protocol}//${window.location.host}/api/voice/ws/${session.session_id}`;
      
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        setSessionState("connected");
        startAudioStreaming();
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          handleWebSocketMessage(data);
        } catch {
          if (event.data instanceof Blob) {
            playAudioResponse(event.data);
          }
        }
      };

      ws.onerror = () => {
        setError("Voice connection error");
        setSessionState("error");
      };

      ws.onclose = () => {
        if (sessionState !== "ended") {
          setSessionState("idle");
        }
        cleanup();
      };

    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : "Failed to start voice session";
      setError(errorMessage);
      setSessionState("error");
    }
  }, [createSessionMutation, cleanup, sessionState]);

  const handleWebSocketMessage = useCallback((data: Record<string, unknown>) => {
    switch (data.type) {
      case "state_change":
        if (data.state === "speaking") {
          setSessionState("speaking");
        } else if (data.state === "listening") {
          setSessionState("listening");
        }
        break;
      case "transcript":
        break;
      case "action_card":
        break;
      case "error":
        setError(data.message as string);
        break;
    }
  }, []);

  const startAudioStreaming = useCallback(() => {
    if (!mediaStreamRef.current || !wsRef.current) return;

    audioContextRef.current = new AudioContext({ sampleRate: 16000 });
    const source = audioContextRef.current.createMediaStreamSource(mediaStreamRef.current);
    const processor = audioContextRef.current.createScriptProcessor(4096, 1, 1);

    processor.onaudioprocess = (e) => {
      if (wsRef.current?.readyState === WebSocket.OPEN && !isMuted) {
        const audioData = e.inputBuffer.getChannelData(0);
        const int16Data = new Int16Array(audioData.length);
        for (let i = 0; i < audioData.length; i++) {
          int16Data[i] = Math.max(-32768, Math.min(32767, audioData[i] * 32768));
        }
        wsRef.current.send(int16Data.buffer);
      }
    };

    source.connect(processor);
    processor.connect(audioContextRef.current.destination);
  }, [isMuted]);

  const playAudioResponse = useCallback(async (audioBlob: Blob) => {
    const audioUrl = URL.createObjectURL(audioBlob);
    const audio = new Audio(audioUrl);
    audio.onended = () => URL.revokeObjectURL(audioUrl);
    await audio.play();
  }, []);

  const endSession = useCallback(async () => {
    if (sessionId) {
      try {
        await endSessionMutation.mutateAsync(sessionId);
      } catch {
      }
    }
    setSessionState("ended");
    setSessionId(null);
    cleanup();
    setTimeout(() => setSessionState("idle"), 100);
  }, [sessionId, endSessionMutation, cleanup]);

  const toggleMute = useCallback(() => {
    setIsMuted(prev => !prev);
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getAudioTracks().forEach(track => {
        track.enabled = isMuted;
      });
    }
  }, [isMuted]);

  const interrupt = useCallback(async () => {
    if (sessionId) {
      await interruptMutation.mutateAsync(sessionId);
    }
  }, [sessionId, interruptMutation]);

  return {
    sessionState,
    sessionId,
    isMuted,
    isConnecting: sessionState === "connecting",
    error,
    startSession,
    endSession,
    toggleMute,
    interrupt,
  };
}
