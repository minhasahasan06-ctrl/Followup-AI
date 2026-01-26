/**
 * Google Cloud KMS Service
 * 
 * HIPAA-compliant encryption/decryption using Cloud KMS.
 * Replaces AWS KMS for all encryption operations.
 */

import { gcpManager, isKMSAvailable, GCP_CONFIG } from "../gcpConfig";
import { gcpErrorHandler } from "../utils/gcpErrorHandler";
import * as crypto from "crypto";

export interface EncryptResult {
  ciphertext: string;
  keyVersion: string;
}

export interface DecryptResult {
  plaintext: Buffer;
}

class GCPKMSService {
  private useKMS: boolean;

  constructor() {
    this.useKMS = isKMSAvailable();
    
    if (this.useKMS) {
      console.log("[GCP KMS] Service initialized");
    } else {
      console.warn("[GCP KMS] KMS not configured, using local encryption fallback");
    }
  }

  async encrypt(plaintext: Buffer): Promise<EncryptResult> {
    gcpErrorHandler.createAuditLog("ENCRYPTION", "kms", true, {
      metadata: { size: plaintext.length },
    });

    if (this.useKMS) {
      return this.encryptWithKMS(plaintext);
    } else {
      return this.encryptLocal(plaintext);
    }
  }

  private async encryptWithKMS(plaintext: Buffer): Promise<EncryptResult> {
    try {
      const kmsClient = gcpManager.getKMSClient();
      const keyPath = gcpManager.getKMSKeyPath();

      const [result] = await kmsClient.encrypt({
        name: keyPath,
        plaintext: plaintext,
      });

      return {
        ciphertext: Buffer.from(result.ciphertext as Uint8Array).toString("base64"),
        keyVersion: result.name || keyPath,
      };
    } catch (error: any) {
      gcpErrorHandler.handleKMSError(error, "encrypt");
      throw error;
    }
  }

  private async encryptLocal(plaintext: Buffer): Promise<EncryptResult> {
    const key = this.getLocalEncryptionKey();
    const iv = crypto.randomBytes(16);
    const cipher = crypto.createCipheriv("aes-256-gcm", key, iv);
    
    const encrypted = Buffer.concat([cipher.update(plaintext), cipher.final()]);
    const authTag = cipher.getAuthTag();

    const combined = Buffer.concat([iv, authTag, encrypted]);

    return {
      ciphertext: combined.toString("base64"),
      keyVersion: "local-dev-key",
    };
  }

  async decrypt(ciphertext: string): Promise<DecryptResult> {
    gcpErrorHandler.createAuditLog("DECRYPTION", "kms", true, {});

    if (this.useKMS) {
      return this.decryptWithKMS(ciphertext);
    } else {
      return this.decryptLocal(ciphertext);
    }
  }

  private async decryptWithKMS(ciphertext: string): Promise<DecryptResult> {
    try {
      const kmsClient = gcpManager.getKMSClient();
      const keyPath = gcpManager.getKMSKeyPath();

      const [result] = await kmsClient.decrypt({
        name: keyPath,
        ciphertext: Buffer.from(ciphertext, "base64"),
      });

      return {
        plaintext: Buffer.from(result.plaintext as Uint8Array),
      };
    } catch (error: any) {
      gcpErrorHandler.handleKMSError(error, "decrypt");
      throw error;
    }
  }

  private async decryptLocal(ciphertext: string): Promise<DecryptResult> {
    const key = this.getLocalEncryptionKey();
    const combined = Buffer.from(ciphertext, "base64");

    const iv = combined.subarray(0, 16);
    const authTag = combined.subarray(16, 32);
    const encrypted = combined.subarray(32);

    const decipher = crypto.createDecipheriv("aes-256-gcm", key, iv);
    decipher.setAuthTag(authTag);

    const decrypted = Buffer.concat([decipher.update(encrypted), decipher.final()]);

    return { plaintext: decrypted };
  }

  private getLocalEncryptionKey(): Buffer {
    const envKey = process.env.LOCAL_ENCRYPTION_KEY;
    if (envKey && envKey.length >= 32) {
      return Buffer.from(envKey.substring(0, 32));
    }
    return crypto.scryptSync("followup-ai-dev-key", "salt", 32);
  }

  async generateDataKey(): Promise<{ plaintext: Buffer; encrypted: string }> {
    const plaintext = crypto.randomBytes(32);
    const { ciphertext } = await this.encrypt(plaintext);

    return {
      plaintext,
      encrypted: ciphertext,
    };
  }

  isUsingKMS(): boolean {
    return this.useKMS;
  }
}

export const kmsService = new GCPKMSService();
export default kmsService;
