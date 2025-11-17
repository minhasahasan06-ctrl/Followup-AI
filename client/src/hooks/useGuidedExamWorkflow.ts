import { useState, useCallback } from 'react';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { apiRequest } from '@/lib/queryClient';
import { useToast } from '@/hooks/use-toast';

export type ExamType = 
  | 'respiratory'
  | 'skin_pallor'
  | 'eye_sclera'
  | 'swelling'
  | 'tremor'
  | 'tongue'
  | 'custom';

export type WorkflowState = 'idle' | 'prep' | 'recording' | 'uploading' | 'completed';

interface ExamStep {
  examType: ExamType;
  sequenceOrder: number;
  skipped: boolean;
  videoBlob?: Blob;
  durationSeconds?: number;
  segmentId?: string;
  customLocation?: string;
  customDescription?: string;
}

const EXAM_SEQUENCE: ExamType[] = [
  'respiratory',
  'skin_pallor',
  'eye_sclera',
  'swelling',
  'tremor',
  'tongue',
  'custom'
];

export function useGuidedExamWorkflow() {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [currentStepIndex, setCurrentStepIndex] = useState(0);
  const [workflowState, setWorkflowState] = useState<WorkflowState>('idle');
  const [examSteps, setExamSteps] = useState<ExamStep[]>(
    EXAM_SEQUENCE.map((examType, index) => ({
      examType,
      sequenceOrder: index + 1,
      skipped: false,
    }))
  );
  const { toast } = useToast();
  const queryClient = useQueryClient();

  // Start new exam session
  const startSessionMutation = useMutation({
    mutationFn: async () => {
      return await apiRequest<{ session_id: string }>('/api/v1/video-ai/exam-sessions/start', {
        method: 'POST',
      });
    },
    onSuccess: (data) => {
      setSessionId(data.session_id);
      setCurrentStepIndex(0);
      setWorkflowState('prep');
      toast({
        title: 'Examination Started',
        description: 'Follow the guided instructions for each examination step.',
      });
    },
    onError: (error: Error) => {
      toast({
        title: 'Failed to Start Examination',
        description: error.message,
        variant: 'destructive',
      });
    },
  });

  // Upload video segment
  const uploadSegmentMutation = useMutation({
    mutationFn: async ({
      videoBlob,
      examType,
      sequenceOrder,
      durationSeconds,
      customLocation,
      customDescription,
    }: {
      videoBlob: Blob;
      examType: ExamType;
      sequenceOrder: number;
      durationSeconds: number;
      customLocation?: string;
      customDescription?: string;
    }) => {
      const formData = new FormData();
      formData.append('file', videoBlob, `${examType}_${Date.now()}.webm`);
      formData.append('session_id', sessionId!);
      formData.append('exam_type', examType);
      formData.append('sequence_order', sequenceOrder.toString());
      formData.append('duration_seconds', durationSeconds.toString());
      if (customLocation) formData.append('custom_location', customLocation);
      if (customDescription) formData.append('custom_description', customDescription);

      return await apiRequest<{ segment_id: string }>('/api/v1/video-ai/exam-sessions/upload-segment', {
        method: 'POST',
        body: formData,
      });
    },
    onSuccess: (data, variables) => {
      const { examType, sequenceOrder, durationSeconds } = variables;
      
      // Update exam step with segment ID
      setExamSteps(prev =>
        prev.map(step =>
          step.sequenceOrder === sequenceOrder
            ? { ...step, segmentId: data.segment_id, durationSeconds }
            : step
        )
      );

      toast({
        title: 'Recording Uploaded',
        description: `${examType} examination uploaded successfully.`,
      });

      // Move to next step
      moveToNextStep();
    },
    onError: (error: Error) => {
      toast({
        title: 'Upload Failed',
        description: error.message,
        variant: 'destructive',
      });
      setWorkflowState('prep');
    },
  });

  // Complete exam session
  const completeSessionMutation = useMutation({
    mutationFn: async () => {
      return await apiRequest(`/api/v1/video-ai/exam-sessions/${sessionId}/complete`, {
        method: 'POST',
      });
    },
    onSuccess: () => {
      setWorkflowState('completed');
      queryClient.invalidateQueries({ queryKey: ['/api/v1/video-ai/exam-sessions'] });
      toast({
        title: 'Examination Completed',
        description: 'All recordings have been processed. View your analysis below.',
      });
    },
    onError: (error: Error) => {
      toast({
        title: 'Failed to Complete Session',
        description: error.message,
        variant: 'destructive',
      });
    },
  });

  // Workflow control functions
  const startExam = useCallback(() => {
    startSessionMutation.mutate();
  }, []);

  const startRecording = useCallback(() => {
    setWorkflowState('recording');
  }, []);

  const completeRecording = useCallback(
    (videoBlob: Blob, durationSeconds: number, customData?: { location?: string; description?: string }) => {
      const currentStep = examSteps[currentStepIndex];
      setWorkflowState('uploading');

      uploadSegmentMutation.mutate({
        videoBlob,
        examType: currentStep.examType,
        sequenceOrder: currentStep.sequenceOrder,
        durationSeconds,
        customLocation: customData?.location,
        customDescription: customData?.description,
      });
    },
    [currentStepIndex, examSteps, sessionId]
  );

  const skipCurrentStep = useCallback(() => {
    const currentStep = examSteps[currentStepIndex];
    
    // Mark as skipped
    setExamSteps(prev =>
      prev.map(step =>
        step.sequenceOrder === currentStep.sequenceOrder
          ? { ...step, skipped: true }
          : step
      )
    );

    toast({
      title: 'Examination Skipped',
      description: `${currentStep.examType} examination has been skipped.`,
    });

    moveToNextStep();
  }, [currentStepIndex, examSteps]);

  const moveToNextStep = useCallback(() => {
    if (currentStepIndex < EXAM_SEQUENCE.length - 1) {
      setCurrentStepIndex(prev => prev + 1);
      setWorkflowState('prep');
    } else {
      // All steps completed, finalize session
      completeSessionMutation.mutate();
    }
  }, [currentStepIndex]);

  const resetWorkflow = useCallback(() => {
    setSessionId(null);
    setCurrentStepIndex(0);
    setWorkflowState('idle');
    setExamSteps(
      EXAM_SEQUENCE.map((examType, index) => ({
        examType,
        sequenceOrder: index + 1,
        skipped: false,
      }))
    );
  }, []);

  const currentStep = examSteps[currentStepIndex];
  const totalSteps = EXAM_SEQUENCE.length;
  const completedSteps = examSteps.filter(s => s.segmentId || s.skipped).length;
  const progress = (completedSteps / totalSteps) * 100;

  return {
    // State
    sessionId,
    currentStep,
    currentStepIndex,
    totalSteps,
    workflowState,
    examSteps,
    progress,
    completedSteps,

    // Actions
    startExam,
    startRecording,
    completeRecording,
    skipCurrentStep,
    resetWorkflow,

    // Mutation states
    isStarting: startSessionMutation.isPending,
    isUploading: uploadSegmentMutation.isPending,
    isCompleting: completeSessionMutation.isPending,
  };
}
