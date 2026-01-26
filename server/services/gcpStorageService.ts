/**
 * Google Cloud Storage Service
 * 
 * Production-grade file storage with:
 * - Signed URL generation for secure access
 * - Cloud KMS encryption for HIPAA compliance
 * - Local filesystem fallback for development
 * - Zod validation for all inputs
 * - Async operations to prevent blocking
 */

import { z } from "zod";
import { Bucket, File, GetSignedUrlConfig } from "@google-cloud/storage";
import { gcpManager, isStorageAvailable, GCP_CONFIG } from "../gcpConfig";
import { gcpErrorHandler } from "../utils/gcpErrorHandler";
import * as fs from "fs/promises";
import * as path from "path";

const UploadOptionsSchema = z.object({
  contentType: z.string().default("application/octet-stream"),
  metadata: z.record(z.string()).optional(),
  encrypt: z.boolean().default(true),
  folder: z.string().optional(),
  filename: z.string().optional(),
});

const DownloadOptionsSchema = z.object({
  generateSignedUrl: z.boolean().default(false),
  expirationSeconds: z.number().default(3600),
});

export type UploadOptions = z.infer<typeof UploadOptionsSchema>;
export type DownloadOptions = z.infer<typeof DownloadOptionsSchema>;

export interface UploadResult {
  bucket: string;
  key: string;
  url: string;
  uri: string;
  size: number;
  contentType: string;
  encrypted: boolean;
}

export interface DownloadResult {
  data: Buffer;
  contentType: string;
  size: number;
  metadata?: Record<string, string>;
}

class GCPStorageService {
  private bucket: Bucket | null = null;
  private useGCS: boolean;
  private localStorageDir: string = "tmp/storage";

  constructor() {
    this.useGCS = isStorageAvailable() && !!GCP_CONFIG.STORAGE.BUCKET_NAME;
    
    if (this.useGCS) {
      try {
        this.bucket = gcpManager.getBucket();
        console.log(`[GCS] Storage service initialized with bucket: ${GCP_CONFIG.STORAGE.BUCKET_NAME}`);
      } catch (error) {
        console.warn("[GCS] Failed to initialize bucket, using local fallback");
        this.useGCS = false;
      }
    }

    if (!this.useGCS) {
      console.warn("[GCS] Using local filesystem storage fallback");
      this.ensureLocalStorageDir();
    }
  }

  private async ensureLocalStorageDir(): Promise<void> {
    try {
      await fs.mkdir(this.localStorageDir, { recursive: true });
    } catch (error) {
      console.error("[GCS] Failed to create local storage directory:", error);
    }
  }

  async uploadFile(
    fileData: Buffer,
    key: string,
    options: Partial<UploadOptions> = {}
  ): Promise<UploadResult> {
    const validatedOptions = UploadOptionsSchema.parse(options);
    const finalKey = validatedOptions.folder ? `${validatedOptions.folder}/${key}` : key;

    gcpErrorHandler.createAuditLog("FILE_UPLOAD", "storage", true, {
      resourceId: finalKey,
      metadata: { contentType: validatedOptions.contentType, size: fileData.length },
    });

    if (this.useGCS && this.bucket) {
      return this.uploadToGCS(fileData, finalKey, validatedOptions);
    } else {
      return this.uploadToLocal(fileData, finalKey, validatedOptions);
    }
  }

  private async uploadToGCS(
    fileData: Buffer,
    key: string,
    options: UploadOptions
  ): Promise<UploadResult> {
    const file = this.bucket!.file(key);

    const metadata: Record<string, any> = {
      contentType: options.contentType,
      metadata: options.metadata || {},
    };

    if (options.encrypt && GCP_CONFIG.KMS.KEY_RING) {
      metadata.kmsKeyName = gcpManager.getKMSKeyPath();
    }

    await file.save(fileData, {
      metadata,
      resumable: fileData.length > 5 * 1024 * 1024,
    });

    const [fileMetadata] = await file.getMetadata();

    const signedUrl = await this.generateSignedUrl(key, "read", GCP_CONFIG.STORAGE.SIGNED_URL_EXPIRATION);

    return {
      bucket: GCP_CONFIG.STORAGE.BUCKET_NAME,
      key,
      url: signedUrl || `gs://${GCP_CONFIG.STORAGE.BUCKET_NAME}/${key}`,
      uri: `gs://${GCP_CONFIG.STORAGE.BUCKET_NAME}/${key}`,
      size: fileData.length,
      contentType: options.contentType,
      encrypted: !!metadata.kmsKeyName,
    };
  }

  private async uploadToLocal(
    fileData: Buffer,
    key: string,
    options: UploadOptions
  ): Promise<UploadResult> {
    const localFilename = key.replace(/\//g, "_");
    const localPath = path.join(this.localStorageDir, localFilename);

    await this.ensureLocalStorageDir();
    await fs.writeFile(localPath, fileData);

    const absolutePath = path.resolve(localPath);
    
    return {
      bucket: "local",
      key,
      url: `file://${absolutePath}`,
      uri: `file://${absolutePath}`,
      size: fileData.length,
      contentType: options.contentType,
      encrypted: false,
    };
  }

  async downloadFile(
    uri: string,
    options: Partial<DownloadOptions> = {}
  ): Promise<DownloadResult> {
    const validatedOptions = DownloadOptionsSchema.parse(options);

    gcpErrorHandler.createAuditLog("FILE_DOWNLOAD", "storage", true, {
      resourceId: uri,
    });

    if (uri.startsWith("gs://")) {
      return this.downloadFromGCS(uri);
    } else if (uri.startsWith("file://")) {
      return this.downloadFromLocal(uri);
    } else {
      throw new Error(`Invalid URI format: ${uri}`);
    }
  }

  private async downloadFromGCS(uri: string): Promise<DownloadResult> {
    const match = uri.match(/^gs:\/\/([^/]+)\/(.+)$/);
    if (!match) {
      throw new Error(`Invalid GCS URI: ${uri}`);
    }

    const [, bucketName, key] = match;
    const bucket = gcpManager.getStorageClient().bucket(bucketName);
    const file = bucket.file(key);

    const [data] = await file.download();
    const [metadata] = await file.getMetadata();

    return {
      data,
      contentType: metadata.contentType || "application/octet-stream",
      size: data.length,
      metadata: metadata.metadata as Record<string, string>,
    };
  }

  private async downloadFromLocal(uri: string): Promise<DownloadResult> {
    const localPath = uri.replace("file://", "");
    const data = await fs.readFile(localPath);

    return {
      data,
      contentType: "application/octet-stream",
      size: data.length,
    };
  }

  async generateSignedUrl(
    key: string,
    action: "read" | "write" = "read",
    expirationSeconds: number = 3600
  ): Promise<string | null> {
    if (!this.useGCS || !this.bucket) {
      console.warn("[GCS] Signed URLs not available in local storage mode");
      return null;
    }

    try {
      const file = this.bucket.file(key);
      const config: GetSignedUrlConfig = {
        version: "v4",
        action: action === "read" ? "read" : "write",
        expires: Date.now() + expirationSeconds * 1000,
      };

      const [url] = await file.getSignedUrl(config);
      return url;
    } catch (error: any) {
      console.error("[GCS] Failed to generate signed URL:", error.message);
      return null;
    }
  }

  async deleteFile(uri: string): Promise<boolean> {
    gcpErrorHandler.createAuditLog("FILE_DELETE", "storage", true, {
      resourceId: uri,
    });

    if (uri.startsWith("gs://")) {
      const match = uri.match(/^gs:\/\/([^/]+)\/(.+)$/);
      if (!match) return false;

      const [, bucketName, key] = match;
      const bucket = gcpManager.getStorageClient().bucket(bucketName);
      const file = bucket.file(key);

      try {
        await file.delete();
        return true;
      } catch (error) {
        console.error("[GCS] Delete failed:", error);
        return false;
      }
    } else if (uri.startsWith("file://")) {
      const localPath = uri.replace("file://", "");
      try {
        await fs.unlink(localPath);
        return true;
      } catch (error) {
        console.error("[GCS] Local delete failed:", error);
        return false;
      }
    }

    return false;
  }

  async listFiles(prefix: string): Promise<string[]> {
    if (!this.useGCS || !this.bucket) {
      try {
        const files = await fs.readdir(this.localStorageDir);
        return files
          .filter((f) => f.startsWith(prefix.replace(/\//g, "_")))
          .map((f) => `file://${path.resolve(this.localStorageDir, f)}`);
      } catch {
        return [];
      }
    }

    try {
      const [files] = await this.bucket.getFiles({ prefix });
      return files.map((f) => `gs://${GCP_CONFIG.STORAGE.BUCKET_NAME}/${f.name}`);
    } catch (error) {
      console.error("[GCS] List files failed:", error);
      return [];
    }
  }

  async fileExists(uri: string): Promise<boolean> {
    if (uri.startsWith("gs://")) {
      const match = uri.match(/^gs:\/\/([^/]+)\/(.+)$/);
      if (!match) return false;

      const [, bucketName, key] = match;
      const bucket = gcpManager.getStorageClient().bucket(bucketName);
      const file = bucket.file(key);

      try {
        const [exists] = await file.exists();
        return exists;
      } catch {
        return false;
      }
    } else if (uri.startsWith("file://")) {
      const localPath = uri.replace("file://", "");
      try {
        await fs.access(localPath);
        return true;
      } catch {
        return false;
      }
    }

    return false;
  }

  isUsingGCS(): boolean {
    return this.useGCS;
  }
}

export const gcsService = new GCPStorageService();
export default gcsService;
