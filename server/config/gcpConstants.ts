/**
 * Centralized Google Cloud Platform Configuration Constants
 * 
 * All GCP configuration is centralized here for the DRY principle.
 * Uses environment variables with sensible defaults for development.
 */

export const GCP_CONFIG = {
  PROJECT_ID: process.env.GCP_PROJECT_ID || process.env.GOOGLE_CLOUD_PROJECT || "",
  REGION: process.env.GCP_REGION || "us-central1",
  LOCATION: process.env.GCP_LOCATION || "us-central1",
  
  STORAGE: {
    BUCKET_NAME: process.env.GCS_BUCKET_NAME || "",
    SYMPTOM_IMAGES_PREFIX: "symptom-journal",
    EXAM_FRAMES_PREFIX: "guided-exam",
    RESEARCH_ARTIFACTS_PREFIX: "research-artifacts",
    ML_MODELS_PREFIX: "ml-models",
    SIGNED_URL_EXPIRATION: 3600,
  },
  
  KMS: {
    KEY_RING: process.env.GCP_KMS_KEY_RING || "followup-ai-keyring",
    CRYPTO_KEY: process.env.GCP_KMS_CRYPTO_KEY || "hipaa-encryption-key",
    LOCATION: process.env.GCP_KMS_LOCATION || "us-central1",
  },
  
  HEALTHCARE: {
    DATASET_ID: process.env.GCP_HEALTHCARE_DATASET || "followup-ai-dataset",
    FHIR_STORE_ID: process.env.GCP_FHIR_STORE || "fhir-store",
    DICOM_STORE_ID: process.env.GCP_DICOM_STORE || "dicom-store",
    LOCATION: process.env.GCP_HEALTHCARE_LOCATION || "us-central1",
  },
  
  DOCUMENT_AI: {
    PROCESSOR_ID: process.env.GCP_DOCUMENT_AI_PROCESSOR || "",
    HEALTHCARE_PROCESSOR_ID: process.env.GCP_HEALTHCARE_DOCUMENT_PROCESSOR || "",
    LOCATION: process.env.GCP_DOCUMENT_AI_LOCATION || "us",
  },
  
  HEALTHCARE_NLP: {
    ENDPOINT: "healthcare.googleapis.com",
    API_VERSION: "v1",
  },
} as const;

export const GCP_ERROR_CODES = {
  NOT_FOUND: 404,
  PERMISSION_DENIED: 403,
  INVALID_ARGUMENT: 400,
  UNAUTHENTICATED: 401,
  RESOURCE_EXHAUSTED: 429,
  INTERNAL: 500,
  UNAVAILABLE: 503,
} as const;

export const HIPAA_AUDIT_ACTIONS = {
  FILE_UPLOAD: "FILE_UPLOAD",
  FILE_DOWNLOAD: "FILE_DOWNLOAD",
  FILE_DELETE: "FILE_DELETE",
  PHI_ACCESS: "PHI_ACCESS",
  PHI_DETECTION: "PHI_DETECTION",
  FHIR_READ: "FHIR_READ",
  FHIR_WRITE: "FHIR_WRITE",
  DICOM_ACCESS: "DICOM_ACCESS",
  ENCRYPTION: "ENCRYPTION",
  DECRYPTION: "DECRYPTION",
} as const;

export function isGCPConfigured(): boolean {
  return !!(GCP_CONFIG.PROJECT_ID && GCP_CONFIG.STORAGE.BUCKET_NAME);
}

export function isHealthcareConfigured(): boolean {
  return !!(GCP_CONFIG.PROJECT_ID && GCP_CONFIG.HEALTHCARE.DATASET_ID);
}

export function isKMSConfigured(): boolean {
  return !!(GCP_CONFIG.PROJECT_ID && GCP_CONFIG.KMS.KEY_RING && GCP_CONFIG.KMS.CRYPTO_KEY);
}

export function isDocumentAIConfigured(): boolean {
  return !!(GCP_CONFIG.PROJECT_ID && GCP_CONFIG.DOCUMENT_AI.PROCESSOR_ID);
}
