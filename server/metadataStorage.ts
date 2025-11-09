import crypto from 'crypto';

interface UserMetadata {
  email: string;
  cognitoSub: string;
  firstName: string;
  lastName: string;
  phoneNumber: string;
  role: 'patient' | 'doctor';
  ehrImportMethod?: string;
  ehrPlatform?: string;
  organization?: string;
  medicalLicenseNumber?: string;
  licenseCountry?: string;
  kycPhotoUrl?: string;
  googleDriveApplicationUrl?: string;
  expiresAt: number;
}

interface PhoneVerificationMetadata {
  email: string;
  phoneNumber: string;
  hashedCode: string;
  expiresAt: number;
}

class MetadataStorage {
  private userMetadata: Map<string, UserMetadata> = new Map();
  private phoneVerification: Map<string, PhoneVerificationMetadata> = new Map();
  private readonly TTL = 24 * 60 * 60 * 1000; // 24 hours
  private readonly PHONE_CODE_TTL = 10 * 60 * 1000; // 10 minutes

  constructor() {
    // Cleanup expired data every hour
    setInterval(() => this.cleanup(), 60 * 60 * 1000);
  }

  private cleanup() {
    const now = Date.now();
    
    for (const [email, data] of this.userMetadata.entries()) {
      if (data.expiresAt < now) {
        this.userMetadata.delete(email);
        console.log(`[METADATA] Purged expired metadata for ${email}`);
      }
    }
    
    for (const [email, data] of this.phoneVerification.entries()) {
      if (data.expiresAt < now) {
        this.phoneVerification.delete(email);
        console.log(`[METADATA] Purged expired phone verification for ${email}`);
      }
    }
  }

  setUserMetadata(email: string, data: Omit<UserMetadata, 'email' | 'expiresAt'>) {
    this.userMetadata.set(email, {
      email,
      ...data,
      expiresAt: Date.now() + this.TTL,
    });
    console.log(`[METADATA] Stored metadata for ${email}`);
  }

  getUserMetadata(email: string): UserMetadata | null {
    const data = this.userMetadata.get(email);
    if (!data) return null;
    
    if (data.expiresAt < Date.now()) {
      this.userMetadata.delete(email);
      return null;
    }
    
    return data;
  }

  deleteUserMetadata(email: string) {
    this.userMetadata.delete(email);
    console.log(`[METADATA] Deleted metadata for ${email}`);
  }

  hashPhoneCode(code: string): string {
    return crypto.createHash('sha256').update(code).digest('hex');
  }

  setPhoneVerification(email: string, phoneNumber: string, code: string) {
    const hashedCode = this.hashPhoneCode(code);
    this.phoneVerification.set(email, {
      email,
      phoneNumber,
      hashedCode,
      expiresAt: Date.now() + this.PHONE_CODE_TTL,
    });
    console.log(`[METADATA] Stored phone verification for ${email}`);
  }

  verifyPhoneCode(email: string, code: string): { valid: boolean; phoneNumber?: string } {
    const data = this.phoneVerification.get(email);
    if (!data) {
      return { valid: false };
    }
    
    if (data.expiresAt < Date.now()) {
      this.phoneVerification.delete(email);
      return { valid: false };
    }
    
    const hashedCode = this.hashPhoneCode(code);
    const valid = hashedCode === data.hashedCode;
    
    if (valid) {
      this.phoneVerification.delete(email);
      return { valid: true, phoneNumber: data.phoneNumber };
    }
    
    return { valid: false };
  }

  getPhoneNumber(email: string): string | null {
    const data = this.phoneVerification.get(email);
    return data?.phoneNumber || null;
  }
}

export const metadataStorage = new MetadataStorage();
