import { useMutation, useQuery } from '@tanstack/react-query';
import { apiRequest, queryClient } from '@/lib/queryClient';

export interface AIQuestion {
  id: string;
  text: string;
  type: 'text' | 'scale' | 'select' | 'multiselect' | 'boolean';
  options?: string[];
  min?: number;
  max?: number;
  category?: string;
  required?: boolean;
}

export interface NextQuestionsResponse {
  experience_id: string;
  questions: AIQuestion[];
  generated_at: string;
  is_stub: boolean;
}

export interface AutopilotTemplate {
  id: string;
  name: string;
  description: string;
  category: string;
  frequency?: string;
  duration?: string;
  priority?: 'high' | 'medium' | 'low';
  actions?: string[];
}

export interface AutopilotPlanResponse {
  experience_id: string;
  templates: AutopilotTemplate[];
  generated_at: string;
  is_stub: boolean;
}

export interface HabitSuggestion {
  id: string;
  name: string;
  description: string;
  category: string;
  frequency: string;
  difficulty: 'easy' | 'medium' | 'hard';
  benefits?: string[];
  tips?: string[];
}

export interface HabitSuggestionsResponse {
  experience_id: string;
  habits: HabitSuggestion[];
  suggestions: HabitSuggestion[];
  generated_at: string;
  is_stub: boolean;
}

export interface FeedbackResponse {
  feedback_id: string;
  experience_id: string;
  rating: string;
  received_at: string;
}

export function useNextQuestions(patientId: string, enabled = true) {
  return useQuery<NextQuestionsResponse>({
    queryKey: ['/api/patient/ai/next-questions', patientId],
    queryFn: async () => {
      const response = await apiRequest('/api/patient/ai/next-questions', {
        method: 'POST',
        body: JSON.stringify({ patient_id: patientId }),
      });
      return response;
    },
    enabled: enabled && !!patientId,
    staleTime: 1000 * 60 * 5,
    retry: 1,
  });
}

export function useAutopilotPlan(patientId: string) {
  return useMutation<AutopilotPlanResponse, Error, { patient_data?: Record<string, unknown> }>({
    mutationFn: async (data) => {
      const response = await apiRequest('/api/patient/ai/autopilot-plan', {
        method: 'POST',
        body: JSON.stringify({ patient_id: patientId, ...data }),
      });
      return response;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['/api/patient/ai/autopilot-plan', patientId] });
    },
  });
}

export function useHabitSuggestions(patientId: string) {
  return useMutation<HabitSuggestionsResponse, Error, { patient_data?: Record<string, unknown> }>({
    mutationFn: async (data) => {
      const response = await apiRequest('/api/patient/ai/habit-suggestions', {
        method: 'POST',
        body: JSON.stringify({ patient_id: patientId, ...data }),
      });
      return {
        ...response,
        suggestions: response.habits || response.suggestions || [],
      };
    },
  });
}

export function useAIFeedback() {
  return useMutation<FeedbackResponse, Error, {
    experience_id: string;
    patient_id: string;
    rating: 'helpful' | 'not_helpful' | 'neutral';
    reason_code?: string;
    additional_context?: string;
  }>({
    mutationFn: async (data) => {
      const response = await apiRequest('/api/patient/ai/feedback', {
        method: 'POST',
        body: JSON.stringify(data),
      });
      return response;
    },
  });
}
