import { useMutation, useQuery } from '@tanstack/react-query';
import { apiRequest, queryClient } from '@/lib/queryClient';

export interface JobComposition {
  job_id: string;
  job_name: string;
  model_type: string;
  hyperparameters: Record<string, unknown>;
  data_pipeline: {
    preprocessing: string[];
    feature_engineering: string[];
    validation_strategy: string;
  };
  estimated_duration: string;
  resource_requirements: Record<string, unknown>;
}

export interface JobCompositionResponse {
  experience_id: string;
  job: JobComposition;
  generated_at: string;
  is_stub: boolean;
}

export interface ValidationReport {
  job_id: string;
  temporal_split: {
    train_start: string;
    train_end: string;
    test_start: string;
    test_end: string;
    gap_days: number;
  };
  leakage_report: {
    feature_leakage: Array<{ feature: string; risk: string; reason: string }>;
    target_leakage: boolean;
    data_leakage: boolean;
    overall_risk: 'low' | 'medium' | 'high';
  };
  train_test_distribution: {
    feature_drifts: Array<{ feature: string; psi: number; status: string }>;
    label_distribution: { train: Record<string, number>; test: Record<string, number> };
  };
}

export interface CalibrationReport {
  job_id: string;
  calibration_curve: Array<{ bin: number; predicted: number; actual: number; count: number }>;
  brier_score: number;
  expected_calibration_error: number;
  reliability_diagram_url?: string;
  threshold_analysis: Array<{
    threshold: number;
    precision: number;
    recall: number;
    f1: number;
    specificity: number;
  }>;
  optimal_threshold: number;
  optimal_metrics: { precision: number; recall: number; f1: number };
}

export interface GovernancePack {
  job_id: string;
  model_card: {
    name: string;
    version: string;
    description: string;
    intended_use: string;
    limitations: string[];
    ethical_considerations: string[];
    performance_metrics: Record<string, number>;
    training_data_summary: string;
  };
  datasheet: {
    data_source: string;
    collection_process: string;
    preprocessing: string[];
    privacy_considerations: string[];
    k_anonymity_threshold: number;
  };
  reproducibility_pack: {
    code_hash: string;
    data_hash: string;
    env_hash: string;
    random_seed: number;
  };
  deploy_gate: {
    passed: boolean;
    checks: Array<{ name: string; status: 'pass' | 'fail' | 'warn'; message: string }>;
    blocking_issues: string[];
  };
}

export interface DriftReport {
  model_id: string;
  report_date: string;
  feature_drifts: Array<{
    feature: string;
    psi: number;
    kl_divergence: number;
    status: 'ok' | 'warning' | 'critical';
    baseline_mean: number;
    current_mean: number;
  }>;
  prediction_drift: {
    psi: number;
    status: string;
    baseline_distribution: Record<string, number>;
    current_distribution: Record<string, number>;
  };
  alerts: Array<{
    id: string;
    severity: 'info' | 'warning' | 'critical';
    message: string;
    feature?: string;
    created_at: string;
  }>;
}

export interface FeedbackDataset {
  id: string;
  name: string;
  source: 'patient' | 'doctor' | 'both';
  record_count: number;
  feedback_types: string[];
  date_range: { start: string; end: string };
  status: 'building' | 'ready' | 'failed';
  created_at: string;
}

export function useComposeJob() {
  return useMutation<JobCompositionResponse, Error, {
    model_type: string;
    objective: string;
    data_sources: string[];
    constraints?: Record<string, unknown>;
  }>({
    mutationFn: async (data) => {
      const response = await apiRequest('/api/ml/ai/compose-job', {
        method: 'POST',
        body: JSON.stringify(data),
      });
      return response;
    },
  });
}

export function useValidationReport(jobId: string, enabled = true) {
  return useQuery<ValidationReport>({
    queryKey: ['/api/ml/validation', jobId],
    queryFn: async () => {
      const response = await fetch(`/api/v1/ml-training/jobs/${jobId}/validation`);
      if (!response.ok) throw new Error('Failed to fetch validation report');
      return response.json();
    },
    enabled: enabled && !!jobId,
  });
}

export function useCalibrationReport(jobId: string, enabled = true) {
  return useQuery<CalibrationReport>({
    queryKey: ['/api/ml/calibration', jobId],
    queryFn: async () => {
      const response = await fetch(`/api/v1/ml-training/jobs/${jobId}/calibration`);
      if (!response.ok) throw new Error('Failed to fetch calibration report');
      return response.json();
    },
    enabled: enabled && !!jobId,
  });
}

export function useGovernancePack(jobId: string, enabled = true) {
  return useQuery<GovernancePack>({
    queryKey: ['/api/ml/governance', jobId],
    queryFn: async () => {
      const response = await fetch(`/api/v1/ml-training/jobs/${jobId}/governance`);
      if (!response.ok) throw new Error('Failed to fetch governance pack');
      return response.json();
    },
    enabled: enabled && !!jobId,
  });
}

export function useDriftReport(modelId: string, enabled = true) {
  return useQuery<DriftReport>({
    queryKey: ['/api/ml/drift', modelId],
    queryFn: async () => {
      const response = await fetch(`/api/v1/ml-training/models/${modelId}/drift`);
      if (!response.ok) throw new Error('Failed to fetch drift report');
      return response.json();
    },
    enabled: enabled && !!modelId,
  });
}

export function useFeedbackDatasets() {
  return useQuery<FeedbackDataset[]>({
    queryKey: ['/api/ml/feedback-datasets'],
    queryFn: async () => {
      const response = await fetch('/api/v1/ml-training/feedback-datasets');
      if (!response.ok) throw new Error('Failed to fetch feedback datasets');
      return response.json();
    },
  });
}

export function useBuildFeedbackDataset() {
  return useMutation<FeedbackDataset, Error, {
    name: string;
    source: 'patient' | 'doctor' | 'both';
    feedback_types: string[];
    date_range?: { start: string; end: string };
  }>({
    mutationFn: async (data) => {
      const response = await apiRequest('/api/v1/ml-training/feedback-datasets', {
        method: 'POST',
        body: JSON.stringify(data),
      });
      return response;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['/api/ml/feedback-datasets'] });
    },
  });
}
