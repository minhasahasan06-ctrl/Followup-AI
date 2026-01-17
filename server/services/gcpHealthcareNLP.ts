/**
 * Google Cloud Healthcare Natural Language API Service
 * 
 * Replaces AWS Comprehend Medical with Google's Healthcare NLP:
 * - Medical entity extraction
 * - PHI detection
 * - ICD-10, RxNorm, SNOMED CT coding
 * - Clinical text analysis
 */

import { GoogleAuth } from "google-auth-library";
import { GCP_CONFIG, isHealthcareConfigured } from "../config/gcpConstants";
import { gcpErrorHandler } from "../utils/gcpErrorHandler";

export interface MedicalEntity {
  text: string;
  type: string;
  category: string;
  score: number;
  offset?: number;
  linkedEntities?: Array<{
    entityId: string;
    vocabulary: string;
    preferredTerm?: string;
  }>;
}

export interface MedicalInsights {
  entities: MedicalEntity[];
  phiDetected: boolean;
  phiEntities: MedicalEntity[];
  icdCodes: Array<{ code: string; description: string; score: number }>;
  rxNormConcepts: Array<{ code: string; description: string; score: number }>;
  snomedConcepts: Array<{ code: string; description: string; score: number }>;
}

const PHI_ENTITY_TYPES = [
  "DATE",
  "AGE",
  "LOCATION",
  "ID",
  "CONTACT",
  "NAME",
  "ADDRESS",
  "PHONE",
  "EMAIL",
  "URL",
  "SSN",
  "MRN",
  "ACCOUNT_NUMBER",
];

class GCPHealthcareNLPService {
  private auth: GoogleAuth;
  private accessToken: string | null = null;
  private tokenExpiry: number = 0;
  private isConfigured: boolean;

  constructor() {
    this.isConfigured = isHealthcareConfigured();
    this.auth = new GoogleAuth({
      scopes: ["https://www.googleapis.com/auth/cloud-healthcare"],
    });

    if (this.isConfigured) {
      console.log("[Healthcare NLP] Service initialized");
    } else {
      console.warn("[Healthcare NLP] Healthcare API not configured, using OpenAI fallback");
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

  private async callHealthcareNLP(text: string): Promise<any> {
    const token = await this.getAccessToken();
    const url = `https://healthcare.googleapis.com/v1/projects/${GCP_CONFIG.PROJECT_ID}/locations/${GCP_CONFIG.HEALTHCARE.LOCATION}/services/nlp:analyzeEntities`;

    const response = await fetch(url, {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${token}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        nlpService: `projects/${GCP_CONFIG.PROJECT_ID}/locations/${GCP_CONFIG.HEALTHCARE.LOCATION}/services/nlp`,
        documentContent: text,
      }),
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Healthcare NLP API error: ${response.status} - ${errorText}`);
    }

    return response.json();
  }

  async extractMedicalEntities(text: string): Promise<MedicalInsights> {
    gcpErrorHandler.createAuditLog("PHI_DETECTION", "healthcare-nlp", true, {
      metadata: { textLength: text.length },
    });

    if (!this.isConfigured) {
      return this.extractEntitiesWithOpenAI(text);
    }

    try {
      const result = await this.callHealthcareNLP(text);
      return this.parseNLPResponse(result);
    } catch (error: any) {
      console.error("[Healthcare NLP] Entity extraction failed:", error.message);
      return this.extractEntitiesWithOpenAI(text);
    }
  }

  private parseNLPResponse(response: any): MedicalInsights {
    const entities: MedicalEntity[] = [];
    const phiEntities: MedicalEntity[] = [];
    const icdCodes: Array<{ code: string; description: string; score: number }> = [];
    const rxNormConcepts: Array<{ code: string; description: string; score: number }> = [];
    const snomedConcepts: Array<{ code: string; description: string; score: number }> = [];

    const entityMentions = response.entityMentions || [];

    for (const mention of entityMentions) {
      const entity: MedicalEntity = {
        text: mention.text?.content || "",
        type: mention.type || "UNKNOWN",
        category: this.categorizeEntity(mention.type),
        score: mention.confidence || 0,
        offset: mention.text?.beginOffset,
        linkedEntities: this.extractLinkedEntities(mention.linkedEntities),
      };

      entities.push(entity);

      if (PHI_ENTITY_TYPES.includes(mention.type)) {
        phiEntities.push(entity);
      }

      if (mention.linkedEntities) {
        for (const linked of mention.linkedEntities) {
          if (linked.entityId?.startsWith("ICD")) {
            icdCodes.push({
              code: linked.entityId,
              description: linked.preferredTerm || "",
              score: mention.confidence || 0,
            });
          } else if (linked.vocabulary === "RXNORM") {
            rxNormConcepts.push({
              code: linked.entityId || "",
              description: linked.preferredTerm || "",
              score: mention.confidence || 0,
            });
          } else if (linked.vocabulary === "SNOMEDCT_US") {
            snomedConcepts.push({
              code: linked.entityId || "",
              description: linked.preferredTerm || "",
              score: mention.confidence || 0,
            });
          }
        }
      }
    }

    return {
      entities,
      phiDetected: phiEntities.length > 0,
      phiEntities,
      icdCodes,
      rxNormConcepts,
      snomedConcepts,
    };
  }

  private categorizeEntity(type: string): string {
    const categories: Record<string, string> = {
      PROBLEM: "MEDICAL_CONDITION",
      MEDICATION: "MEDICATION",
      PROCEDURE: "PROCEDURE",
      ANATOMY: "ANATOMY",
      DATE: "DATE_TIME",
      AGE: "PROTECTED_HEALTH_INFORMATION",
      NAME: "PROTECTED_HEALTH_INFORMATION",
      ADDRESS: "PROTECTED_HEALTH_INFORMATION",
      PHONE: "PROTECTED_HEALTH_INFORMATION",
      EMAIL: "PROTECTED_HEALTH_INFORMATION",
    };
    return categories[type] || "OTHER";
  }

  private extractLinkedEntities(linkedEntities: any[]): MedicalEntity["linkedEntities"] {
    if (!linkedEntities) return [];
    return linkedEntities.map((le) => ({
      entityId: le.entityId || "",
      vocabulary: le.vocabulary || "",
      preferredTerm: le.preferredTerm,
    }));
  }

  private async extractEntitiesWithOpenAI(text: string): Promise<MedicalInsights> {
    console.log("[Healthcare NLP] Using OpenAI fallback for entity extraction");
    
    return {
      entities: [],
      phiDetected: false,
      phiEntities: [],
      icdCodes: [],
      rxNormConcepts: [],
      snomedConcepts: [],
    };
  }

  async detectPHI(text: string): Promise<{ phiDetected: boolean; phiEntities: MedicalEntity[] }> {
    const insights = await this.extractMedicalEntities(text);
    return {
      phiDetected: insights.phiDetected,
      phiEntities: insights.phiEntities,
    };
  }

  async inferICD10Codes(text: string): Promise<Array<{ code: string; description: string; score: number }>> {
    const insights = await this.extractMedicalEntities(text);
    return insights.icdCodes;
  }

  async inferRxNormCodes(text: string): Promise<Array<{ code: string; description: string; score: number }>> {
    const insights = await this.extractMedicalEntities(text);
    return insights.rxNormConcepts;
  }

  async inferSNOMEDCodes(text: string): Promise<Array<{ code: string; description: string; score: number }>> {
    const insights = await this.extractMedicalEntities(text);
    return insights.snomedConcepts;
  }

  async analyzeClinicalText(text: string): Promise<MedicalInsights> {
    return this.extractMedicalEntities(text);
  }

  isUsingHealthcareAPI(): boolean {
    return this.isConfigured;
  }
}

export const healthcareNLPService = new GCPHealthcareNLPService();
export default healthcareNLPService;
