import {
  DetectEntitiesV2Command,
  DetectPHICommand,
  InferICD10CMCommand,
  InferRxNormCommand,
  InferSNOMEDCTCommand,
  type Entity,
} from "@aws-sdk/client-comprehendmedical";
import {
  SearchImageSetsCommand,
  GetImageSetMetadataCommand,
} from "@aws-sdk/client-medical-imaging";
import {
  ListSequenceStoresCommand,
  ListReadSetsCommand,
} from "@aws-sdk/client-omics";
import {
  comprehendMedicalClient,
  medicalImagingClient,
  omicsClient,
} from "./aws";

interface MedicalEntity {
  text: string;
  type: string;
  category: string;
  score: number;
  traits?: Array<{ name: string; score: number }>;
  attributes?: Array<{ text: string; type: string; score: number }>;
}

interface MedicalInsights {
  entities: MedicalEntity[];
  icdCodes?: Array<{ code: string; description: string; score: number }>;
  rxNormConcepts?: Array<{ code: string; description: string; score: number }>;
  snomedConcepts?: Array<{ code: string; description: string; score: number }>;
  phiDetected?: boolean;
  phiEntities?: MedicalEntity[];
}


export class AWSHealthcareService {
  private static MEDICAL_IMAGING_DATASTORE_ID = process.env.AWS_MEDICAL_IMAGING_DATASTORE_ID;

  /**
   * Extract medical entities from clinical text using Amazon Comprehend Medical
   * Replaces simple keyword matching with AI-powered medical NLP
   */
  static async extractMedicalEntities(text: string): Promise<MedicalInsights> {
    try {
      const command = new DetectEntitiesV2Command({ Text: text });
      const response = await comprehendMedicalClient.send(command);

      const entities: MedicalEntity[] =
        response.Entities?.map((entity: Entity) => ({
          text: entity.Text || "",
          type: entity.Type || "",
          category: entity.Category || "",
          score: entity.Score || 0,
          traits: entity.Traits?.map((trait) => ({
            name: trait.Name || "",
            score: trait.Score || 0,
          })),
          attributes: entity.Attributes?.map((attr) => ({
            text: attr.Text || "",
            type: attr.Type || "",
            score: attr.Score || 0,
          })),
        })) || [];

      return {
        entities,
        phiDetected: false,
      };
    } catch (error) {
      console.error("Comprehend Medical entity extraction failed:", error);
      return { entities: [], phiDetected: false };
    }
  }

  /**
   * Detect Protected Health Information (PHI) in text
   * Critical for HIPAA compliance
   */
  static async detectPHI(text: string): Promise<{ phiDetected: boolean; phiEntities: MedicalEntity[] }> {
    try {
      const command = new DetectPHICommand({ Text: text });
      const response = await comprehendMedicalClient.send(command);

      const phiEntities: MedicalEntity[] =
        response.Entities?.map((entity: Entity) => ({
          text: entity.Text || "",
          type: entity.Type || "",
          category: entity.Category || "",
          score: entity.Score || 0,
        })) || [];

      return {
        phiDetected: phiEntities.length > 0,
        phiEntities,
      };
    } catch (error) {
      console.error("PHI detection failed:", error);
      return { phiDetected: false, phiEntities: [] };
    }
  }

  /**
   * Infer ICD-10-CM diagnosis codes from clinical text
   * Useful for automated medical coding
   */
  static async inferICD10Codes(text: string) {
    try {
      const command = new InferICD10CMCommand({ Text: text });
      const response = await comprehendMedicalClient.send(command);

      return (
        response.Entities?.map((entity) => ({
          text: entity.Text || "",
          code: entity.ICD10CMConcepts?.[0]?.Code || "",
          description: entity.ICD10CMConcepts?.[0]?.Description || "",
          score: entity.ICD10CMConcepts?.[0]?.Score || 0,
        })) || []
      );
    } catch (error) {
      console.error("ICD-10 inference failed:", error);
      return [];
    }
  }

  /**
   * Infer RxNorm medication codes from clinical text
   * Identifies medications mentioned in conversations
   */
  static async inferRxNormCodes(text: string) {
    try {
      const command = new InferRxNormCommand({ Text: text });
      const response = await comprehendMedicalClient.send(command);

      return (
        response.Entities?.map((entity) => ({
          text: entity.Text || "",
          code: entity.RxNormConcepts?.[0]?.Code || "",
          description: entity.RxNormConcepts?.[0]?.Description || "",
          score: entity.RxNormConcepts?.[0]?.Score || 0,
        })) || []
      );
    } catch (error) {
      console.error("RxNorm inference failed:", error);
      return [];
    }
  }

  /**
   * Infer SNOMED CT medical concepts from clinical text
   * Provides standardized medical terminology
   */
  static async inferSNOMEDCodes(text: string) {
    try {
      const command = new InferSNOMEDCTCommand({ Text: text });
      const response = await comprehendMedicalClient.send(command);

      return (
        response.Entities?.flatMap(
          (entity) =>
            entity.SNOMEDCTConcepts?.map((concept) => ({
              text: entity.Text || "",
              code: concept.Code || "",
              description: concept.Description || "",
              score: concept.Score || 0,
            })) || []
        ) || []
      );
    } catch (error) {
      console.error("SNOMED CT inference failed:", error);
      return [];
    }
  }

  /**
   * Comprehensive medical analysis combining all Comprehend Medical capabilities
   * Used by Agent Clona for advanced medical understanding
   */
  static async analyzeClinicalText(text: string): Promise<MedicalInsights> {
    try {
      const [entities, phi, icdCodes, rxNormCodes, snomedConcepts] = await Promise.all([
        this.extractMedicalEntities(text),
        this.detectPHI(text),
        this.inferICD10Codes(text),
        this.inferRxNormCodes(text),
        this.inferSNOMEDCodes(text),
      ]);

      return {
        ...entities,
        phiDetected: phi.phiDetected,
        phiEntities: phi.phiEntities,
        icdCodes,
        rxNormConcepts: rxNormCodes,
        snomedConcepts,
      };
    } catch (error) {
      console.error("Clinical text analysis failed:", error);
      return { entities: [], phiDetected: false };
    }
  }

  /**
   * AWS HealthLake integration uses FHIR R4 REST APIs (not SDK commands)
   * For production implementation, use HTTP requests with AWS Signature Version 4
   * Reference: https://docs.aws.amazon.com/healthlake/latest/devguide/crud-healthlake.html
   * 
   * Note: This requires custom HTTP request signing and is complex to implement.
   * For now, we focus on Comprehend Medical which provides immediate value.
   */

  /**
   * Search medical imaging data in AWS HealthImaging
   * Useful for analyzing radiology reports, CT scans, MRI, X-rays
   */
  static async searchMedicalImages(patientId: string): Promise<any[]> {
    if (!this.MEDICAL_IMAGING_DATASTORE_ID) {
      console.warn("Medical Imaging datastore ID not configured");
      return [];
    }

    try {
      const command = new SearchImageSetsCommand({
        datastoreId: this.MEDICAL_IMAGING_DATASTORE_ID,
        searchCriteria: {
          filters: [
            {
              values: [{ DICOMPatientId: patientId }],
              operator: "EQUAL",
            },
          ],
        },
      });

      const response = await medicalImagingClient.send(command);
      return response.imageSetsMetadataSummaries || [];
    } catch (error) {
      console.error("Medical image search failed:", error);
      return [];
    }
  }

  /**
   * Get medical image metadata from AWS HealthImaging
   */
  static async getImageMetadata(imageSetId: string): Promise<any> {
    if (!this.MEDICAL_IMAGING_DATASTORE_ID) {
      console.warn("Medical Imaging datastore ID not configured");
      return null;
    }

    try {
      const command = new GetImageSetMetadataCommand({
        datastoreId: this.MEDICAL_IMAGING_DATASTORE_ID,
        imageSetId,
      });

      const response = await medicalImagingClient.send(command);
      return response.imageSetMetadataBlob;
    } catch (error) {
      console.error("Image metadata retrieval failed:", error);
      return null;
    }
  }

  /**
   * List genomic sequence stores in AWS HealthOmics
   * Foundation for genomic analysis capabilities
   */
  static async listGenomicStores(): Promise<any[]> {
    try {
      const command = new ListSequenceStoresCommand({});
      const response = await omicsClient.send(command);
      return response.sequenceStores || [];
    } catch (error) {
      console.error("Genomic store listing failed:", error);
      return [];
    }
  }

  /**
   * List genomic read sets for a patient
   * Enables personalized medicine based on genomic data
   */
  static async listGenomicReadSets(sequenceStoreId: string, patientId?: string): Promise<any[]> {
    try {
      const command = new ListReadSetsCommand({
        sequenceStoreId,
        filter: patientId ? { subjectId: patientId } : undefined,
      });

      const response = await omicsClient.send(command);
      return response.readSets || [];
    } catch (error) {
      console.error("Genomic read set listing failed:", error);
      return [];
    }
  }
}

export default AWSHealthcareService;
