"""
Google Cloud Platform Configuration Constants (Python)

Centralized GCP configuration for all Python services.
Mirrors server/config/gcpConstants.ts for consistency.
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class StorageConfig:
    BUCKET_NAME: str = os.getenv("GCS_BUCKET_NAME", "")
    SYMPTOM_IMAGES_PREFIX: str = "symptom-journal"
    EXAM_FRAMES_PREFIX: str = "guided-exam"
    RESEARCH_ARTIFACTS_PREFIX: str = "research-artifacts"
    ML_MODELS_PREFIX: str = "ml-models"
    SIGNED_URL_EXPIRATION: int = 3600


@dataclass(frozen=True)
class KMSConfig:
    KEY_RING: str = os.getenv("GCP_KMS_KEY_RING", "followup-ai-keyring")
    CRYPTO_KEY: str = os.getenv("GCP_KMS_CRYPTO_KEY", "hipaa-encryption-key")
    LOCATION: str = os.getenv("GCP_KMS_LOCATION", "us-central1")


@dataclass(frozen=True)
class HealthcareConfig:
    DATASET_ID: str = os.getenv("GCP_HEALTHCARE_DATASET", "followup-ai-dataset")
    FHIR_STORE_ID: str = os.getenv("GCP_FHIR_STORE", "fhir-store")
    DICOM_STORE_ID: str = os.getenv("GCP_DICOM_STORE", "dicom-store")
    LOCATION: str = os.getenv("GCP_HEALTHCARE_LOCATION", "us-central1")


@dataclass(frozen=True)
class DocumentAIConfig:
    PROCESSOR_ID: str = os.getenv("GCP_DOCUMENT_AI_PROCESSOR", "")
    HEALTHCARE_PROCESSOR_ID: str = os.getenv("GCP_HEALTHCARE_DOCUMENT_PROCESSOR", "")
    LOCATION: str = os.getenv("GCP_DOCUMENT_AI_LOCATION", "us")


@dataclass(frozen=True)
class GCPConfig:
    PROJECT_ID: str = os.getenv("GCP_PROJECT_ID", os.getenv("GOOGLE_CLOUD_PROJECT", ""))
    REGION: str = os.getenv("GCP_REGION", "us-central1")
    LOCATION: str = os.getenv("GCP_LOCATION", "us-central1")
    STORAGE: StorageConfig = StorageConfig()
    KMS: KMSConfig = KMSConfig()
    HEALTHCARE: HealthcareConfig = HealthcareConfig()
    DOCUMENT_AI: DocumentAIConfig = DocumentAIConfig()


GCP_CONFIG = GCPConfig()


HIPAA_AUDIT_ACTIONS = {
    "FILE_UPLOAD": "FILE_UPLOAD",
    "FILE_DOWNLOAD": "FILE_DOWNLOAD",
    "FILE_DELETE": "FILE_DELETE",
    "PHI_ACCESS": "PHI_ACCESS",
    "PHI_DETECTION": "PHI_DETECTION",
    "FHIR_READ": "FHIR_READ",
    "FHIR_WRITE": "FHIR_WRITE",
    "DICOM_ACCESS": "DICOM_ACCESS",
    "ENCRYPTION": "ENCRYPTION",
    "DECRYPTION": "DECRYPTION",
}


def is_gcp_configured() -> bool:
    """Check if basic GCP configuration is available."""
    return bool(GCP_CONFIG.PROJECT_ID and GCP_CONFIG.STORAGE.BUCKET_NAME)


def is_healthcare_configured() -> bool:
    """Check if Healthcare API is configured."""
    return bool(GCP_CONFIG.PROJECT_ID and GCP_CONFIG.HEALTHCARE.DATASET_ID)


def is_kms_configured() -> bool:
    """Check if Cloud KMS is configured."""
    return bool(GCP_CONFIG.PROJECT_ID and GCP_CONFIG.KMS.KEY_RING and GCP_CONFIG.KMS.CRYPTO_KEY)


def is_document_ai_configured() -> bool:
    """Check if Document AI is configured."""
    return bool(GCP_CONFIG.PROJECT_ID and GCP_CONFIG.DOCUMENT_AI.PROCESSOR_ID)


def get_kms_key_path() -> str:
    """Get the full Cloud KMS key path."""
    return (
        f"projects/{GCP_CONFIG.PROJECT_ID}/locations/{GCP_CONFIG.KMS.LOCATION}/"
        f"keyRings/{GCP_CONFIG.KMS.KEY_RING}/cryptoKeys/{GCP_CONFIG.KMS.CRYPTO_KEY}"
    )


def get_healthcare_dataset_path() -> str:
    """Get the Healthcare API dataset path."""
    return (
        f"projects/{GCP_CONFIG.PROJECT_ID}/locations/{GCP_CONFIG.HEALTHCARE.LOCATION}/"
        f"datasets/{GCP_CONFIG.HEALTHCARE.DATASET_ID}"
    )


def get_fhir_store_path() -> str:
    """Get the FHIR store path."""
    return f"{get_healthcare_dataset_path()}/fhirStores/{GCP_CONFIG.HEALTHCARE.FHIR_STORE_ID}"


def get_dicom_store_path() -> str:
    """Get the DICOM store path."""
    return f"{get_healthcare_dataset_path()}/dicomStores/{GCP_CONFIG.HEALTHCARE.DICOM_STORE_ID}"
