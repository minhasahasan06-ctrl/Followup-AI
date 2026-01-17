/**
 * Google Cloud Platform Client Initialization
 * 
 * Centralized initialization of all GCP clients:
 * - Cloud Storage (GCS)
 * - Cloud KMS
 * - Healthcare API (FHIR, DICOM, NLP)
 * - Document AI
 * 
 * Authentication via GOOGLE_APPLICATION_CREDENTIALS or Application Default Credentials
 */

import { Storage } from "@google-cloud/storage";
import { KeyManagementServiceClient } from "@google-cloud/kms";
import { GCP_CONFIG, isGCPConfigured, isKMSConfigured } from "./config/gcpConstants";

class GCPClientManager {
  private static instance: GCPClientManager;
  private storageClient: Storage | null = null;
  private kmsClient: KeyManagementServiceClient | null = null;
  private initialized = false;

  private constructor() {
    this.initialize();
  }

  private initialize(): void {
    try {
      if (!process.env.GOOGLE_APPLICATION_CREDENTIALS && !process.env.GCP_PROJECT_ID) {
        console.warn("[GCP] Warning: GOOGLE_APPLICATION_CREDENTIALS or GCP_PROJECT_ID not set. GCP features will use local fallback.");
        return;
      }

      const clientConfig = GCP_CONFIG.PROJECT_ID ? { projectId: GCP_CONFIG.PROJECT_ID } : {};

      this.storageClient = new Storage(clientConfig);
      console.log("[GCP] Storage client initialized");

      if (isKMSConfigured()) {
        this.kmsClient = new KeyManagementServiceClient(clientConfig);
        console.log("[GCP] KMS client initialized");
      }

      this.initialized = true;
      console.log(`[GCP] All clients initialized for project: ${GCP_CONFIG.PROJECT_ID || "default"}`);
    } catch (error: any) {
      console.error("[GCP] Failed to initialize clients:", error.message);
      this.initialized = false;
    }
  }

  public static getInstance(): GCPClientManager {
    if (!GCPClientManager.instance) {
      GCPClientManager.instance = new GCPClientManager();
    }
    return GCPClientManager.instance;
  }

  public getStorageClient(): Storage {
    if (!this.storageClient) {
      throw new Error("GCP Storage client not initialized. Check GOOGLE_APPLICATION_CREDENTIALS.");
    }
    return this.storageClient;
  }

  public getKMSClient(): KeyManagementServiceClient {
    if (!this.kmsClient) {
      throw new Error("GCP KMS client not initialized. Check GCP_KMS_* environment variables.");
    }
    return this.kmsClient;
  }

  public isInitialized(): boolean {
    return this.initialized;
  }

  public isStorageAvailable(): boolean {
    return this.storageClient !== null;
  }

  public isKMSAvailable(): boolean {
    return this.kmsClient !== null;
  }

  public getBucket(bucketName?: string) {
    const storage = this.getStorageClient();
    return storage.bucket(bucketName || GCP_CONFIG.STORAGE.BUCKET_NAME);
  }

  public getKMSKeyPath(): string {
    return `projects/${GCP_CONFIG.PROJECT_ID}/locations/${GCP_CONFIG.KMS.LOCATION}/keyRings/${GCP_CONFIG.KMS.KEY_RING}/cryptoKeys/${GCP_CONFIG.KMS.CRYPTO_KEY}`;
  }

  public getHealthcareDatasetPath(): string {
    return `projects/${GCP_CONFIG.PROJECT_ID}/locations/${GCP_CONFIG.HEALTHCARE.LOCATION}/datasets/${GCP_CONFIG.HEALTHCARE.DATASET_ID}`;
  }

  public getFHIRStorePath(): string {
    return `${this.getHealthcareDatasetPath()}/fhirStores/${GCP_CONFIG.HEALTHCARE.FHIR_STORE_ID}`;
  }

  public getDICOMStorePath(): string {
    return `${this.getHealthcareDatasetPath()}/dicomStores/${GCP_CONFIG.HEALTHCARE.DICOM_STORE_ID}`;
  }
}

export const gcpManager = GCPClientManager.getInstance();

export function getStorageClient(): Storage {
  return gcpManager.getStorageClient();
}

export function getKMSClient(): KeyManagementServiceClient {
  return gcpManager.getKMSClient();
}

export function isGCPAvailable(): boolean {
  return gcpManager.isInitialized();
}

export function isStorageAvailable(): boolean {
  return gcpManager.isStorageAvailable();
}

export function isKMSAvailable(): boolean {
  return gcpManager.isKMSAvailable();
}

export { GCP_CONFIG } from "./config/gcpConstants";
