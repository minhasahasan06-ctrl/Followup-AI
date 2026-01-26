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

function parseGCSCredentials(): { credentials: any; projectId: string } | null {
  let credentialsJson = process.env.GCS_SERVICE_ACCOUNT_KEY;
  if (!credentialsJson) {
    console.log("Could not parse GCS credentials from env var, trying file fallback...");
    return null;
  }
  
  try {
    const lastBraceIndex = credentialsJson.lastIndexOf('}');
    if (lastBraceIndex !== -1 && lastBraceIndex < credentialsJson.length - 1) {
      credentialsJson = credentialsJson.substring(0, lastBraceIndex + 1);
    }
    
    const credentials = JSON.parse(credentialsJson);
    const projectId = credentials.project_id;
    console.log(`[GCP] Credentials loaded from GCS_SERVICE_ACCOUNT_KEY for project: ${projectId}`);
    return { credentials, projectId };
  } catch (error: any) {
    console.error("[GCP] Failed to parse GCS_SERVICE_ACCOUNT_KEY:", error.message);
    return null;
  }
}

class GCPClientManager {
  private static instance: GCPClientManager;
  private storageClient: Storage | null = null;
  private kmsClient: KeyManagementServiceClient | null = null;
  private initialized = false;
  private initAttempted = false;
  private projectId: string | null = null;

  private constructor() {
  }
  
  private ensureInitialized(): void {
    if (!this.initAttempted) {
      this.initAttempted = true;
      this.initialize();
    }
  }

  private initialize(): void {
    try {
      const parsedCredentials = parseGCSCredentials();
      
      if (parsedCredentials) {
        this.projectId = parsedCredentials.projectId;
        const clientConfig = {
          projectId: parsedCredentials.projectId,
          credentials: parsedCredentials.credentials,
        };
        
        this.storageClient = new Storage(clientConfig);
        console.log("[GCP] Storage client initialized with service account credentials");

        if (isKMSConfigured()) {
          this.kmsClient = new KeyManagementServiceClient(clientConfig);
          console.log("[GCP] KMS client initialized");
        }

        this.initialized = true;
        console.log(`[GCP] All clients initialized for project: ${this.projectId}`);
        return;
      }
      
      if (process.env.GOOGLE_APPLICATION_CREDENTIALS || process.env.GCP_PROJECT_ID) {
        const clientConfig = GCP_CONFIG.PROJECT_ID ? { projectId: GCP_CONFIG.PROJECT_ID } : {};
        this.projectId = GCP_CONFIG.PROJECT_ID || null;

        this.storageClient = new Storage(clientConfig);
        console.log("[GCP] Storage client initialized with ADC");

        if (isKMSConfigured()) {
          this.kmsClient = new KeyManagementServiceClient(clientConfig);
          console.log("[GCP] KMS client initialized");
        }

        this.initialized = true;
        console.log(`[GCP] All clients initialized for project: ${this.projectId || "default"}`);
        return;
      }
      
      console.log("GCS credentials not available - GCS operations may fail");
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
    this.ensureInitialized();
    if (!this.storageClient) {
      throw new Error("GCP Storage client not initialized. Check GCS_SERVICE_ACCOUNT_KEY or GOOGLE_APPLICATION_CREDENTIALS.");
    }
    return this.storageClient;
  }

  public getKMSClient(): KeyManagementServiceClient {
    this.ensureInitialized();
    if (!this.kmsClient) {
      throw new Error("GCP KMS client not initialized. Check GCP_KMS_* environment variables.");
    }
    return this.kmsClient;
  }

  public isInitialized(): boolean {
    this.ensureInitialized();
    return this.initialized;
  }

  public isStorageAvailable(): boolean {
    this.ensureInitialized();
    return this.storageClient !== null;
  }

  public isKMSAvailable(): boolean {
    this.ensureInitialized();
    return this.kmsClient !== null;
  }

  public getBucket(bucketName?: string) {
    this.ensureInitialized();
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
