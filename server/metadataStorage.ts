import bcrypt from 'bcryptjs';

interface UserMetadata {
  email: string;
  cognitoSub: string;
  cognitoUsername: string;
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
  verificationCode?: string;
  verificationCodeExpires?: number;
  expiresAt: number;
}

interface PhoneVerificationMetadata {
  email: string;
  phoneNumber: string;
  hashedCode: string;
  expiresAt: number;
}

interface EmailVerificationMetadata {
  email: string;
  hashedCode: string;
  expiresAt: number;
}

class MetadataStorage {
  private userMetadata: Map<string, UserMetadata> = new Map();
  private phoneVerification: Map<string, PhoneVerificationMetadata> = new Map();
  private emailVerification: Map<string, EmailVerificationMetadata> = new Map();
  private readonly TTL = 24 * 60 * 60 * 1000; // 24 hours
  private readonly EMAIL_CODE_TTL = 24 * 60 * 60 * 1000; // 24 hours
  private readonly PHONE_CODE_TTL = 15 * 60 * 1000; // 15 minutes
  private readonly EMAIL_CODE_TTL = 24 * 60 * 60 * 1000; // 24 hours

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
    
    for (const [email, data] of this.emailVerification.entries()) {
      if (data.expiresAt < now) {
        this.emailVerification.delete(email);
        console.log(`[METADATA] Purged expired email verification for ${email}`);
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


  async setPhoneVerification(email: string, phoneNumber: string, code: string) {
    const hashedCode = await bcrypt.hash(code, 10);
    this.phoneVerification.set(email, {
      email,
      phoneNumber,
      hashedCode,
      expiresAt: Date.now() + this.PHONE_CODE_TTL,
    });
    console.log(`[METADATA] Stored phone verification for ${email}`);
  }

  async verifyPhoneCode(email: string, code: string): Promise<{ valid: boolean; phoneNumber?: string }> {
    const data = this.phoneVerification.get(email);
    if (!data) {
      return { valid: false };
    }
    
    if (data.expiresAt < Date.now()) {
      this.phoneVerification.delete(email);
      return { valid: false };
    }
    
    const valid = await bcrypt.compare(code, data.hashedCode);
    
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

  async setEmailVerification(email: string, code: string) {
    const hashedCode = await bcrypt.hash(code, 10);
    this.emailVerification.set(email, {
      email,
      hashedCode,
      expiresAt: Date.now() + this.EMAIL_CODE_TTL,
    });
    console.log(`[METADATA] Stored email verification for ${email}`);
  }

  async verifyEmailCode(email: string, code: string): Promise<{ valid: boolean }> {
    const data = this.emailVerification.get(email);
    if (!data) {
      return { valid: false };
    }
    
    if (data.expiresAt < Date.now()) {
      this.emailVerification.delete(email);
      return { valid: false };
    }
    
    const valid = await bcrypt.compare(code, data.hashedCode);
    
    if (valid) {
      this.emailVerification.delete(email);
      return { valid: true };
    }
    
    return { valid: false };
  }

  deleteEmailVerification(email: string) {
    this.emailVerification.delete(email);
    console.log(`[METADATA] Deleted email verification for ${email}`);
  }
}

export const metadataStorage = new MetadataStorage();
