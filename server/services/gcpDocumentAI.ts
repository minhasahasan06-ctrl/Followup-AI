/**
 * Google Cloud Document AI Service
 * 
 * Replaces AWS Textract with Document AI Healthcare processors:
 * - OCR for medical documents
 * - Form field extraction
 * - Healthcare document parsing
 */

import { GoogleAuth } from "google-auth-library";
import { GCP_CONFIG, isDocumentAIConfigured } from "../config/gcpConstants";
import { gcpErrorHandler } from "../utils/gcpErrorHandler";

export interface ExtractedField {
  fieldName: string;
  fieldValue: string;
  confidence: number;
  boundingBox?: {
    left: number;
    top: number;
    width: number;
    height: number;
  };
}

export interface ExtractedTable {
  rows: string[][];
  headers?: string[];
  confidence: number;
}

export interface DocumentExtractionResult {
  text: string;
  fields: ExtractedField[];
  tables: ExtractedTable[];
  pages: number;
  confidence: number;
  mimeType: string;
}

class GCPDocumentAIService {
  private auth: GoogleAuth;
  private accessToken: string | null = null;
  private tokenExpiry: number = 0;
  private isConfigured: boolean;

  constructor() {
    this.isConfigured = isDocumentAIConfigured();
    this.auth = new GoogleAuth({
      scopes: ["https://www.googleapis.com/auth/cloud-platform"],
    });

    if (this.isConfigured) {
      console.log("[Document AI] Service initialized");
    } else {
      console.warn("[Document AI] Not configured, OCR features will be limited");
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

  async processDocument(
    documentContent: Buffer,
    mimeType: string = "application/pdf"
  ): Promise<DocumentExtractionResult> {
    if (!this.isConfigured) {
      console.warn("[Document AI] Not configured, returning empty result");
      return {
        text: "",
        fields: [],
        tables: [],
        pages: 0,
        confidence: 0,
        mimeType,
      };
    }

    try {
      const token = await this.getAccessToken();
      const processorPath = `projects/${GCP_CONFIG.PROJECT_ID}/locations/${GCP_CONFIG.DOCUMENT_AI.LOCATION}/processors/${GCP_CONFIG.DOCUMENT_AI.PROCESSOR_ID}`;
      const url = `https://${GCP_CONFIG.DOCUMENT_AI.LOCATION}-documentai.googleapis.com/v1/${processorPath}:process`;

      const requestBody = {
        rawDocument: {
          content: documentContent.toString("base64"),
          mimeType,
        },
      };

      const response = await fetch(url, {
        method: "POST",
        headers: {
          "Authorization": `Bearer ${token}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify(requestBody),
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Document AI error: ${response.status} - ${errorText}`);
      }

      const result = await response.json();
      return this.parseDocumentResponse(result, mimeType);
    } catch (error: any) {
      gcpErrorHandler.handleDocumentAIError(error, "processDocument");
      return {
        text: "",
        fields: [],
        tables: [],
        pages: 0,
        confidence: 0,
        mimeType,
      };
    }
  }

  private parseDocumentResponse(response: any, mimeType: string): DocumentExtractionResult {
    const document = response.document || {};
    const text = document.text || "";
    const pages = document.pages || [];

    const fields: ExtractedField[] = [];
    const tables: ExtractedTable[] = [];
    let totalConfidence = 0;
    let confidenceCount = 0;

    for (const page of pages) {
      if (page.formFields) {
        for (const field of page.formFields) {
          const fieldName = this.extractTextFromLayout(field.fieldName, text);
          const fieldValue = this.extractTextFromLayout(field.fieldValue, text);
          const confidence = field.fieldName?.confidence || 0;

          fields.push({
            fieldName,
            fieldValue,
            confidence,
            boundingBox: this.extractBoundingBox(field.fieldName?.boundingPoly),
          });

          totalConfidence += confidence;
          confidenceCount++;
        }
      }

      if (page.tables) {
        for (const table of page.tables) {
          const tableData: string[][] = [];
          const headers: string[] = [];

          if (table.headerRows) {
            for (const headerRow of table.headerRows) {
              const row: string[] = [];
              for (const cell of headerRow.cells || []) {
                row.push(this.extractTextFromLayout(cell.layout, text));
              }
              headers.push(...row);
            }
          }

          if (table.bodyRows) {
            for (const bodyRow of table.bodyRows) {
              const row: string[] = [];
              for (const cell of bodyRow.cells || []) {
                row.push(this.extractTextFromLayout(cell.layout, text));
              }
              tableData.push(row);
            }
          }

          tables.push({
            rows: tableData,
            headers: headers.length > 0 ? headers : undefined,
            confidence: 0.9,
          });
        }
      }
    }

    return {
      text,
      fields,
      tables,
      pages: pages.length,
      confidence: confidenceCount > 0 ? totalConfidence / confidenceCount : 0,
      mimeType,
    };
  }

  private extractTextFromLayout(layout: any, fullText: string): string {
    if (!layout?.textAnchor?.textSegments) return "";

    let result = "";
    for (const segment of layout.textAnchor.textSegments) {
      const start = parseInt(segment.startIndex || "0");
      const end = parseInt(segment.endIndex || "0");
      result += fullText.substring(start, end);
    }
    return result.trim();
  }

  private extractBoundingBox(boundingPoly: any): ExtractedField["boundingBox"] | undefined {
    if (!boundingPoly?.normalizedVertices || boundingPoly.normalizedVertices.length < 4) {
      return undefined;
    }

    const vertices = boundingPoly.normalizedVertices;
    return {
      left: vertices[0].x || 0,
      top: vertices[0].y || 0,
      width: (vertices[1].x || 0) - (vertices[0].x || 0),
      height: (vertices[2].y || 0) - (vertices[0].y || 0),
    };
  }

  async processHealthcareDocument(
    documentContent: Buffer,
    mimeType: string = "application/pdf"
  ): Promise<DocumentExtractionResult> {
    if (!GCP_CONFIG.DOCUMENT_AI.HEALTHCARE_PROCESSOR_ID) {
      return this.processDocument(documentContent, mimeType);
    }

    try {
      const token = await this.getAccessToken();
      const processorPath = `projects/${GCP_CONFIG.PROJECT_ID}/locations/${GCP_CONFIG.DOCUMENT_AI.LOCATION}/processors/${GCP_CONFIG.DOCUMENT_AI.HEALTHCARE_PROCESSOR_ID}`;
      const url = `https://${GCP_CONFIG.DOCUMENT_AI.LOCATION}-documentai.googleapis.com/v1/${processorPath}:process`;

      const requestBody = {
        rawDocument: {
          content: documentContent.toString("base64"),
          mimeType,
        },
      };

      const response = await fetch(url, {
        method: "POST",
        headers: {
          "Authorization": `Bearer ${token}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify(requestBody),
      });

      if (!response.ok) {
        throw new Error(`Healthcare Document AI error: ${response.status}`);
      }

      const result = await response.json();
      return this.parseDocumentResponse(result, mimeType);
    } catch (error: any) {
      console.error("[Document AI] Healthcare processing failed, falling back to standard:", error.message);
      return this.processDocument(documentContent, mimeType);
    }
  }

  async extractTextFromImage(imageContent: Buffer, mimeType: string = "image/png"): Promise<string> {
    const result = await this.processDocument(imageContent, mimeType);
    return result.text;
  }

  isUsingDocumentAI(): boolean {
    return this.isConfigured;
  }
}

export const documentAIService = new GCPDocumentAIService();
export default documentAIService;
