import { useMutation, useQuery } from '@tanstack/react-query';
import { apiRequest, queryClient } from '@/lib/queryClient';

export interface CohortTranslation {
  cohort_id: string;
  sql_filter: string;
  dsl_json: Record<string, unknown>;
  patient_count: number;
  k_anonymous: boolean;
  preview: Array<{ patient_id_hash: string; age_bucket: string; condition_bucket: string }>;
  warnings?: string[];
}

export interface CohortTranslateResponse {
  experience_id: string;
  cohort: CohortTranslation;
  generated_at: string;
  is_stub: boolean;
}

export interface StudyProtocol {
  id: string;
  title: string;
  description: string;
  study_type: string;
  primary_endpoints: string[];
  secondary_endpoints?: string[];
  inclusion_criteria: string[];
  exclusion_criteria: string[];
  data_collection: string[];
  timeline?: string;
  sample_size_estimate?: number;
}

export interface StudyProtocolResponse {
  experience_id: string;
  protocol: StudyProtocol;
  generated_at: string;
  is_stub: boolean;
}

export interface TrialEmulation {
  trial_id: string;
  treatment_arm: { size: number; outcomes: Record<string, number> };
  control_arm: { size: number; outcomes: Record<string, number> };
  effect_estimate: number;
  confidence_interval: [number, number];
  p_value: number;
  balance_metrics: Record<string, number>;
}

export interface TrialEmulationResponse {
  experience_id: string;
  emulation: TrialEmulation;
  generated_at: string;
  is_stub: boolean;
}

export function useCohortTranslate() {
  return useMutation<CohortTranslateResponse, Error, {
    natural_language: string;
    doctor_id: string;
  }>({
    mutationFn: async (data) => {
      const response = await apiRequest('/api/research/ai/translate-cohort', {
        method: 'POST',
        body: JSON.stringify(data),
      });
      return response;
    },
  });
}

export function useGenerateProtocol() {
  return useMutation<StudyProtocolResponse, Error, {
    cohort_id: string;
    doctor_id: string;
    study_objective: string;
    study_type?: string;
  }>({
    mutationFn: async (data) => {
      const response = await apiRequest('/api/research/ai/study-protocol', {
        method: 'POST',
        body: JSON.stringify(data),
      });
      return response;
    },
  });
}

export function useTrialEmulation() {
  return useMutation<TrialEmulationResponse, Error, {
    cohort_id: string;
    doctor_id: string;
    treatment_variable: string;
    outcome_variable: string;
    covariates?: string[];
  }>({
    mutationFn: async (data) => {
      const response = await apiRequest('/api/research/ai/emulate-trial', {
        method: 'POST',
        body: JSON.stringify(data),
      });
      return response;
    },
  });
}

export function useReproducibilityPack(studyId: string, enabled = true) {
  return useQuery<{
    study_id: string;
    code_hash: string;
    data_hash: string;
    env_hash: string;
    created_at: string;
    artifacts: Array<{ name: string; hash: string; size: number }>;
  }>({
    queryKey: ['/api/research/reproducibility-pack', studyId],
    enabled: enabled && !!studyId,
  });
}
