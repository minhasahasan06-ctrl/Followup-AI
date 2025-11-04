import { S3Client } from "@aws-sdk/client-s3";
import { TextractClient } from "@aws-sdk/client-textract";
import { ComprehendMedicalClient } from "@aws-sdk/client-comprehendmedical";
import { HealthLakeClient } from "@aws-sdk/client-healthlake";
import { MedicalImagingClient } from "@aws-sdk/client-medical-imaging";

const AWS_REGION = process.env.AWS_REGION || "us-east-1";

const awsConfig = {
  region: AWS_REGION,
  credentials: {
    accessKeyId: process.env.AWS_ACCESS_KEY_ID || "",
    secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY || "",
  },
};

export const s3Client = new S3Client(awsConfig);

export const textractClient = new TextractClient(awsConfig);

export const comprehendMedicalClient = new ComprehendMedicalClient(awsConfig);

export const healthLakeClient = new HealthLakeClient(awsConfig);

export const medicalImagingClient = new MedicalImagingClient(awsConfig);

export const AWS_S3_BUCKET = process.env.AWS_S3_BUCKET_NAME || "";
