/**
 * Google Cloud Healthcare API FHIR/DICOM Service
 * 
 * Replaces AWS HealthLake and Medical Imaging with Google's Healthcare API:
 * - FHIR R4 resource management
 * - DICOM store operations
 * - Patient data management
 */

import { GoogleAuth } from "google-auth-library";
import { GCP_CONFIG, isHealthcareConfigured } from "../config/gcpConstants";
import { gcpErrorHandler } from "../utils/gcpErrorHandler";

export interface FHIRResource {
  resourceType: string;
  id?: string;
  [key: string]: any;
}

export interface FHIRSearchResult {
  resourceType: "Bundle";
  type: "searchset";
  total: number;
  entry?: Array<{
    resource: FHIRResource;
    fullUrl?: string;
  }>;
}

export interface DICOMStudy {
  studyInstanceUID: string;
  patientId?: string;
  studyDate?: string;
  studyDescription?: string;
  modality?: string;
  seriesCount?: number;
}

class GCPFHIRService {
  private auth: GoogleAuth;
  private accessToken: string | null = null;
  private tokenExpiry: number = 0;
  private isConfigured: boolean;
  private baseUrl: string;

  constructor() {
    this.isConfigured = isHealthcareConfigured();
    this.auth = new GoogleAuth({
      scopes: ["https://www.googleapis.com/auth/cloud-healthcare"],
    });

    const datasetPath = `projects/${GCP_CONFIG.PROJECT_ID}/locations/${GCP_CONFIG.HEALTHCARE.LOCATION}/datasets/${GCP_CONFIG.HEALTHCARE.DATASET_ID}`;
    this.baseUrl = `https://healthcare.googleapis.com/v1/${datasetPath}`;

    if (this.isConfigured) {
      console.log("[FHIR Service] Initialized");
    } else {
      console.warn("[FHIR Service] Healthcare API not configured");
    }
  }

  private async getAccessToken(): Promise<string> {
    if (this.accessToken && Date.now() < this.tokenExpiry) {
      return this.accessToken;
    }

    const client = await this.auth.getClient();
    const tokenResponse = await client.getAccessToken();
    this.accessToken = tokenResponse.token || "";
    this.tokenExpiry = Date.now() + 3500 * 1000;
    return this.accessToken;
  }

  private getFHIRStoreUrl(): string {
    return `${this.baseUrl}/fhirStores/${GCP_CONFIG.HEALTHCARE.FHIR_STORE_ID}/fhir`;
  }

  private getDICOMStoreUrl(): string {
    return `${this.baseUrl}/dicomStores/${GCP_CONFIG.HEALTHCARE.DICOM_STORE_ID}`;
  }

  async createResource(resource: FHIRResource): Promise<FHIRResource> {
    if (!this.isConfigured) {
      throw new Error("FHIR service not configured");
    }

    gcpErrorHandler.createAuditLog("FHIR_WRITE", "fhir", true, {
      resourceId: resource.id,
      metadata: { resourceType: resource.resourceType },
    });

    try {
      const token = await this.getAccessToken();
      const url = `${this.getFHIRStoreUrl()}/${resource.resourceType}`;

      const response = await fetch(url, {
        method: "POST",
        headers: {
          "Authorization": `Bearer ${token}`,
          "Content-Type": "application/fhir+json",
        },
        body: JSON.stringify(resource),
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`FHIR create error: ${response.status} - ${errorText}`);
      }

      return response.json();
    } catch (error: any) {
      gcpErrorHandler.handleHealthcareError(error, "createResource");
      throw error;
    }
  }

  async readResource(resourceType: string, resourceId: string): Promise<FHIRResource | null> {
    if (!this.isConfigured) {
      throw new Error("FHIR service not configured");
    }

    gcpErrorHandler.createAuditLog("FHIR_READ", "fhir", true, {
      resourceId,
      metadata: { resourceType },
    });

    try {
      const token = await this.getAccessToken();
      const url = `${this.getFHIRStoreUrl()}/${resourceType}/${resourceId}`;

      const response = await fetch(url, {
        method: "GET",
        headers: {
          "Authorization": `Bearer ${token}`,
          "Accept": "application/fhir+json",
        },
      });

      if (response.status === 404) {
        return null;
      }

      if (!response.ok) {
        throw new Error(`FHIR read error: ${response.status}`);
      }

      return response.json();
    } catch (error: any) {
      gcpErrorHandler.handleHealthcareError(error, "readResource");
      throw error;
    }
  }

  async updateResource(resource: FHIRResource): Promise<FHIRResource> {
    if (!this.isConfigured || !resource.id) {
      throw new Error("FHIR service not configured or resource ID missing");
    }

    gcpErrorHandler.createAuditLog("FHIR_WRITE", "fhir", true, {
      resourceId: resource.id,
      metadata: { resourceType: resource.resourceType },
    });

    try {
      const token = await this.getAccessToken();
      const url = `${this.getFHIRStoreUrl()}/${resource.resourceType}/${resource.id}`;

      const response = await fetch(url, {
        method: "PUT",
        headers: {
          "Authorization": `Bearer ${token}`,
          "Content-Type": "application/fhir+json",
        },
        body: JSON.stringify(resource),
      });

      if (!response.ok) {
        throw new Error(`FHIR update error: ${response.status}`);
      }

      return response.json();
    } catch (error: any) {
      gcpErrorHandler.handleHealthcareError(error, "updateResource");
      throw error;
    }
  }

  async searchResources(
    resourceType: string,
    params: Record<string, string> = {}
  ): Promise<FHIRSearchResult> {
    if (!this.isConfigured) {
      return { resourceType: "Bundle", type: "searchset", total: 0, entry: [] };
    }

    gcpErrorHandler.createAuditLog("FHIR_READ", "fhir", true, {
      metadata: { resourceType, searchParams: Object.keys(params) },
    });

    try {
      const token = await this.getAccessToken();
      const searchParams = new URLSearchParams(params);
      const url = `${this.getFHIRStoreUrl()}/${resourceType}?${searchParams.toString()}`;

      const response = await fetch(url, {
        method: "GET",
        headers: {
          "Authorization": `Bearer ${token}`,
          "Accept": "application/fhir+json",
        },
      });

      if (!response.ok) {
        throw new Error(`FHIR search error: ${response.status}`);
      }

      return response.json();
    } catch (error: any) {
      gcpErrorHandler.handleHealthcareError(error, "searchResources");
      return { resourceType: "Bundle", type: "searchset", total: 0, entry: [] };
    }
  }

  async deleteResource(resourceType: string, resourceId: string): Promise<boolean> {
    if (!this.isConfigured) {
      throw new Error("FHIR service not configured");
    }

    gcpErrorHandler.createAuditLog("FHIR_WRITE", "fhir", true, {
      resourceId,
      metadata: { resourceType, action: "delete" },
    });

    try {
      const token = await this.getAccessToken();
      const url = `${this.getFHIRStoreUrl()}/${resourceType}/${resourceId}`;

      const response = await fetch(url, {
        method: "DELETE",
        headers: {
          "Authorization": `Bearer ${token}`,
        },
      });

      return response.ok;
    } catch (error: any) {
      gcpErrorHandler.handleHealthcareError(error, "deleteResource");
      return false;
    }
  }

  async searchDICOMStudies(patientId?: string): Promise<DICOMStudy[]> {
    if (!this.isConfigured) {
      return [];
    }

    gcpErrorHandler.createAuditLog("DICOM_ACCESS", "dicom", true, {
      metadata: { patientId },
    });

    try {
      const token = await this.getAccessToken();
      let url = `${this.getDICOMStoreUrl()}/dicomWeb/studies`;
      
      if (patientId) {
        url += `?PatientID=${encodeURIComponent(patientId)}`;
      }

      const response = await fetch(url, {
        method: "GET",
        headers: {
          "Authorization": `Bearer ${token}`,
          "Accept": "application/dicom+json",
        },
      });

      if (!response.ok) {
        throw new Error(`DICOM search error: ${response.status}`);
      }

      const studies = await response.json();
      return this.parseDICOMStudies(studies);
    } catch (error: any) {
      gcpErrorHandler.handleHealthcareError(error, "searchDICOMStudies");
      return [];
    }
  }

  private parseDICOMStudies(studies: any[]): DICOMStudy[] {
    return studies.map((study) => ({
      studyInstanceUID: study["0020000D"]?.Value?.[0] || "",
      patientId: study["00100020"]?.Value?.[0],
      studyDate: study["00080020"]?.Value?.[0],
      studyDescription: study["00081030"]?.Value?.[0],
      modality: study["00080060"]?.Value?.[0],
    }));
  }

  async getDICOMStudyMetadata(studyInstanceUID: string): Promise<any> {
    if (!this.isConfigured) {
      return null;
    }

    gcpErrorHandler.createAuditLog("DICOM_ACCESS", "dicom", true, {
      resourceId: studyInstanceUID,
    });

    try {
      const token = await this.getAccessToken();
      const url = `${this.getDICOMStoreUrl()}/dicomWeb/studies/${studyInstanceUID}/metadata`;

      const response = await fetch(url, {
        method: "GET",
        headers: {
          "Authorization": `Bearer ${token}`,
          "Accept": "application/dicom+json",
        },
      });

      if (!response.ok) {
        return null;
      }

      return response.json();
    } catch (error: any) {
      gcpErrorHandler.handleHealthcareError(error, "getDICOMStudyMetadata");
      return null;
    }
  }

  isUsingHealthcareAPI(): boolean {
    return this.isConfigured;
  }
}

export const fhirService = new GCPFHIRService();
export default fhirService;
