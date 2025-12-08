import { useMutation } from '@tanstack/react-query';
import { apiRequest } from '@/lib/queryClient';
import { useAuth } from '@/contexts/AuthContext';

export type SignalCategory = 
  | 'device' 
  | 'symptom' 
  | 'video' 
  | 'audio' 
  | 'pain' 
  | 'mental' 
  | 'environment' 
  | 'meds'
  | 'exposure';

interface SignalPayload {
  category: SignalCategory;
  source: string;
  raw_payload: Record<string, any>;
  ml_score?: number;
  signal_time?: string;
}

interface UseAutopilotSignalOptions {
  onSuccess?: () => void;
  onError?: (error: Error) => void;
}

export function useAutopilotSignal(options?: UseAutopilotSignalOptions) {
  const { user } = useAuth();
  const patientId = user?.id || user?.sub || 'demo-patient';

  const sendSignalMutation = useMutation({
    mutationFn: async (signal: SignalPayload) => {
      if (!patientId || patientId === 'demo-patient') {
        console.warn('[Autopilot] Skipping signal send for demo patient');
        return { signal_id: null, status: 'skipped' };
      }

      return apiRequest(
        'POST',
        `/api/v1/followup-autopilot/patients/${encodeURIComponent(patientId)}/signals`,
        {
          category: signal.category,
          source: signal.source,
          raw_payload: signal.raw_payload,
          ml_score: signal.ml_score,
          signal_time: signal.signal_time || new Date().toISOString(),
        }
      );
    },
    onSuccess: () => {
      options?.onSuccess?.();
    },
    onError: (error: Error) => {
      console.error('[Autopilot] Failed to send signal:', error);
      options?.onError?.(error);
    },
  });

  const sendBatchSignalsMutation = useMutation({
    mutationFn: async (signals: SignalPayload[]) => {
      if (!patientId || patientId === 'demo-patient') {
        console.warn('[Autopilot] Skipping batch signal send for demo patient');
        return { signal_ids: [], count: 0 };
      }

      return apiRequest(
        'POST',
        `/api/v1/followup-autopilot/patients/${encodeURIComponent(patientId)}/signals/batch`,
        {
          signals: signals.map(s => ({
            category: s.category,
            source: s.source,
            raw_payload: s.raw_payload,
            ml_score: s.ml_score,
            signal_time: s.signal_time || new Date().toISOString(),
          })),
        }
      );
    },
    onSuccess: () => {
      options?.onSuccess?.();
    },
    onError: (error: Error) => {
      console.error('[Autopilot] Failed to send batch signals:', error);
      options?.onError?.(error);
    },
  });

  const sendDeviceSignal = (data: {
    heartRate?: number;
    oxygenSaturation?: number;
    temperature?: number;
    steps?: number;
    sleepHours?: number;
    source?: string;
  }) => {
    sendSignalMutation.mutate({
      category: 'device',
      source: data.source || 'wearable_sync',
      raw_payload: data,
      ml_score: undefined,
    });
  };

  const sendSymptomSignal = (data: {
    painLevel?: number;
    fatigueLevel?: number;
    symptoms?: string[];
    notes?: string;
  }) => {
    const mlScore = data.painLevel && data.fatigueLevel 
      ? Math.min(1, (data.painLevel + data.fatigueLevel) / 20) 
      : undefined;

    sendSignalMutation.mutate({
      category: 'symptom',
      source: 'patient_reported',
      raw_payload: data,
      ml_score: mlScore,
    });
  };

  const sendVideoSignal = (data: {
    sessionId?: string;
    respiratoryRisk?: number;
    tremorIndex?: number;
    gaitScore?: number;
    analysisResults?: Record<string, any>;
  }) => {
    sendSignalMutation.mutate({
      category: 'video',
      source: 'video_exam',
      raw_payload: data,
      ml_score: data.respiratoryRisk,
    });
  };

  const sendAudioSignal = (data: {
    sessionId?: string;
    emotionScore?: number;
    stressLevel?: number;
    extractedSymptoms?: string[];
    analysisResults?: Record<string, any>;
  }) => {
    sendSignalMutation.mutate({
      category: 'audio',
      source: 'voice_analysis',
      raw_payload: data,
      ml_score: data.emotionScore,
    });
  };

  const sendPainSignal = (data: {
    vasScore: number;
    joint?: string;
    duration?: string;
    medication?: string;
  }) => {
    sendSignalMutation.mutate({
      category: 'pain',
      source: 'paintrack',
      raw_payload: data,
      ml_score: data.vasScore / 10,
    });
  };

  const sendMentalHealthSignal = (data: {
    questionnaireType: string;
    totalScore: number;
    maxScore: number;
    severityLevel?: string;
    responses?: Record<string, any>;
  }) => {
    const normalizedScore = data.totalScore / data.maxScore;

    sendSignalMutation.mutate({
      category: 'mental',
      source: 'mental_health_screening',
      raw_payload: data,
      ml_score: normalizedScore,
    });
  };

  const sendMedicationSignal = (data: {
    medicationId?: string;
    medicationName: string;
    action: 'taken' | 'missed' | 'skipped' | 'late';
    scheduledTime?: string;
    actualTime?: string;
  }) => {
    const adherenceScore = data.action === 'taken' ? 1.0 
      : data.action === 'late' ? 0.7 
      : 0.0;

    sendSignalMutation.mutate({
      category: 'meds',
      source: 'medication_tracker',
      raw_payload: data,
      ml_score: adherenceScore,
    });
  };

  const sendEnvironmentSignal = (data: {
    aqi?: number;
    pollenIndex?: number;
    temperature?: number;
    humidity?: number;
    location?: string;
  }) => {
    const envRisk = Math.min(1, ((data.aqi || 0) / 300 + (data.pollenIndex || 0) / 12) / 2);

    sendSignalMutation.mutate({
      category: 'environment',
      source: 'environmental_monitor',
      raw_payload: data,
      ml_score: envRisk,
    });
  };

  const sendExposureSignal = (data: {
    exposureType: string;
    riskLevel?: number;
    location?: string;
    notes?: string;
  }) => {
    sendSignalMutation.mutate({
      category: 'exposure',
      source: 'risk_exposure_report',
      raw_payload: data,
      ml_score: data.riskLevel,
    });
  };

  return {
    sendSignal: sendSignalMutation.mutate,
    sendBatchSignals: sendBatchSignalsMutation.mutate,
    sendDeviceSignal,
    sendSymptomSignal,
    sendVideoSignal,
    sendAudioSignal,
    sendPainSignal,
    sendMentalHealthSignal,
    sendMedicationSignal,
    sendEnvironmentSignal,
    sendExposureSignal,
    isPending: sendSignalMutation.isPending || sendBatchSignalsMutation.isPending,
    isError: sendSignalMutation.isError || sendBatchSignalsMutation.isError,
  };
}
