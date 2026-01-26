import { Storage, Bucket } from "@google-cloud/storage";
import * as fs from "fs";
import * as path from "path";

let storage: Storage;
let bucket: Bucket;

const GCS_BUCKET_NAME = process.env.GCS_BUCKET_NAME || "";
const GCS_PROJECT_ID = process.env.GCS_PROJECT_ID || "";

function loadCredentialsFromFile(): object | null {
  // Try to load from attached_assets directory (fallback for when secret isn't properly set)
  const possiblePaths = [
    "attached_assets/followupai-medlm-prod-9355f1756964_1765725040068.json",
    "gcs-credentials.json",
  ];
  
  for (const filePath of possiblePaths) {
    try {
      if (fs.existsSync(filePath)) {
        const content = fs.readFileSync(filePath, "utf-8");
        const parsed = JSON.parse(content);
        console.log(`✅ GCS credentials loaded from file: ${filePath}`);
        return parsed;
      }
    } catch (err) {
      // Continue to next path
    }
  }
  return null;
}

let initAttempted = false;

function initializeGCS(): void {
  if (storage) return;
  if (initAttempted) return;
  initAttempted = true;

  let credentials = process.env.GCS_SERVICE_ACCOUNT_KEY;
  let parsedCredentials: object | null = null;
  
  if (credentials) {
    try {
      let credentialsToUse = credentials.trim();
      
      if (credentialsToUse.startsWith('"') || credentialsToUse.startsWith("'")) {
        credentialsToUse = credentialsToUse.slice(1, -1);
      }
      
      credentialsToUse = credentialsToUse.replace(/\\\\n/g, '\\n');
      
      const lastBraceIndex = credentialsToUse.lastIndexOf('}');
      if (lastBraceIndex !== -1 && lastBraceIndex < credentialsToUse.length - 1) {
        credentialsToUse = credentialsToUse.substring(0, lastBraceIndex + 1);
      }
      
      parsedCredentials = JSON.parse(credentialsToUse);
      console.log(`[GCS] Credentials loaded for project: ${(parsedCredentials as any).project_id}`);
    } catch (error) {
      console.warn("[GCS] Could not parse credentials from env var, trying file fallback...");
      parsedCredentials = loadCredentialsFromFile();
    }
  } else {
    parsedCredentials = loadCredentialsFromFile();
  }
  
  if (parsedCredentials) {
    storage = new Storage({
      projectId: GCS_PROJECT_ID || (parsedCredentials as any).project_id,
      credentials: parsedCredentials,
    });
    console.log("[GCS] Storage client initialized successfully");
  } else {
    console.warn("[GCS] Credentials not available - storage operations may fail");
    storage = new Storage({
      projectId: GCS_PROJECT_ID,
    });
  }

  if (GCS_BUCKET_NAME) {
    bucket = storage.bucket(GCS_BUCKET_NAME);
  }
}

export function getStorage(): Storage {
  if (!storage) {
    initializeGCS();
  }
  return storage;
}

export function getBucket(): Bucket {
  if (!bucket) {
    if (!GCS_BUCKET_NAME) {
      throw new Error("GCS_BUCKET_NAME environment variable is not set");
    }
    initializeGCS();
  }
  return bucket;
}

export interface UploadOptions {
  key: string;
  body: Buffer | string;
  contentType?: string;
  metadata?: Record<string, string>;
}

export async function uploadFile(options: UploadOptions): Promise<string> {
  const { key, body, contentType, metadata } = options;
  const file = getBucket().file(key);

  await file.save(body instanceof Buffer ? body : Buffer.from(body), {
    contentType: contentType || "application/octet-stream",
    metadata: metadata ? { metadata } : undefined,
    resumable: false,
  });

  return `gs://${GCS_BUCKET_NAME}/${key}`;
}

export async function uploadFileWithStream(options: UploadOptions): Promise<string> {
  const { key, body, contentType, metadata } = options;
  const file = getBucket().file(key);

  return new Promise((resolve, reject) => {
    const stream = file.createWriteStream({
      contentType: contentType || "application/octet-stream",
      metadata: metadata ? { metadata } : undefined,
      resumable: true,
    });

    stream.on("error", reject);
    stream.on("finish", () => {
      resolve(`gs://${GCS_BUCKET_NAME}/${key}`);
    });

    if (body instanceof Buffer) {
      stream.end(body);
    } else {
      stream.end(Buffer.from(body));
    }
  });
}

export async function downloadFile(key: string): Promise<Buffer> {
  const file = getBucket().file(key);
  const [contents] = await file.download();
  return contents;
}

export async function getSignedUrl(
  key: string,
  expirationMinutes: number = 60
): Promise<string> {
  const file = getBucket().file(key);
  const [url] = await file.getSignedUrl({
    action: "read",
    expires: Date.now() + expirationMinutes * 60 * 1000,
  });
  return url;
}

export async function getSignedUploadUrl(
  key: string,
  contentType: string,
  expirationMinutes: number = 60
): Promise<string> {
  const file = getBucket().file(key);
  const [url] = await file.getSignedUrl({
    action: "write",
    expires: Date.now() + expirationMinutes * 60 * 1000,
    contentType,
  });
  return url;
}

export async function deleteFile(key: string): Promise<void> {
  const file = getBucket().file(key);
  await file.delete({ ignoreNotFound: true });
}

export async function fileExists(key: string): Promise<boolean> {
  const file = getBucket().file(key);
  const [exists] = await file.exists();
  return exists;
}

export async function listFiles(prefix: string): Promise<string[]> {
  const [files] = await getBucket().getFiles({ prefix });
  return files.map((file) => file.name);
}

export async function copyFile(sourceKey: string, destKey: string): Promise<void> {
  const sourceFile = getBucket().file(sourceKey);
  const destFile = getBucket().file(destKey);
  await sourceFile.copy(destFile);
}

export function getPublicUrl(key: string): string {
  return `https://storage.googleapis.com/${GCS_BUCKET_NAME}/${key}`;
}

export { GCS_BUCKET_NAME, GCS_PROJECT_ID };
