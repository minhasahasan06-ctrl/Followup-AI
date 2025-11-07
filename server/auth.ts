import bcrypt from "bcryptjs";
import crypto from "crypto";
import { type Express, type RequestHandler } from "express";
import session from "express-session";
import connectPg from "connect-pg-simple";
import { storage } from "./storage";
import { Resend } from 'resend';

// Extend Express Request to include user
declare global {
  namespace Express {
    interface Request {
      user?: {
        id: string;
        email: string;
        role: string;
      };
    }
  }
}

function getReplitConnectorAuth() {
  const hostname = process.env.REPLIT_CONNECTORS_HOSTNAME;
  const xReplitToken = process.env.REPL_IDENTITY
    ? `repl ${process.env.REPL_IDENTITY}`
    : process.env.WEB_REPL_RENEWAL
    ? `depl ${process.env.WEB_REPL_RENEWAL}`
    : null;

  if (!hostname || !xReplitToken) {
    return null;
  }

  return { hostname, xReplitToken };
}

export function isEmailVerificationConfigured(): boolean {
  // Consider configured if either Replit Connector is available or ENV has API key
  return (
    !!process.env.RESEND_API_KEY ||
    getReplitConnectorAuth() !== null
  );
}

export function isEmailVerificationRequired(): boolean {
  const flag = process.env.EMAIL_VERIFICATION_REQUIRED?.toLowerCase();
  if (flag === "true") {
    return true;
  }
  if (flag === "false") {
    return false;
  }
  return process.env.NODE_ENV === "production" && isEmailVerificationConfigured();
}

// Session configuration
export function getSession(maxAge?: number) {
  const sessionTtl = maxAge || (1 * 24 * 60 * 60 * 1000); // Default 1 day in milliseconds

  // Detect if running in Replit environment (iframe context requires third-party cookies)
  const isReplit = !!process.env.REPL_ID || !!process.env.REPLIT_DEV_DOMAIN;

  // Prefer Postgres-backed sessions when DATABASE_URL is configured; otherwise fall back to in-memory store.
  // The in-memory store is only suitable for development and will not persist across restarts.
  let store: session.Store | undefined;
  if (process.env.DATABASE_URL) {
    const PgStore = connectPg(session);
    store = new PgStore({
      conString: process.env.DATABASE_URL,
      createTableIfMissing: true,
      tableName: "sessions",
    });
  } else {
    // eslint-disable-next-line no-console
    console.warn("[SESSION] DATABASE_URL not set – using MemoryStore (dev only). Sessions won't persist across restarts.");
    store = new session.MemoryStore();
  }

  // Replit serves apps in an iframe, creating third-party cookie context
  // Must use sameSite: "none" with secure: true for cookies to work in iframe
  const cookieConfig = isReplit ? {
    httpOnly: true,
    secure: true, // Required for sameSite: none (Replit uses HTTPS proxy)
    sameSite: "none" as const, // Required for third-party cookie context (iframe)
    maxAge: sessionTtl,
    path: '/',
  } : {
    httpOnly: true,
    secure: false, // Local development uses HTTP
    sameSite: "lax" as const,
    maxAge: sessionTtl,
    path: '/',
  };

  const sessionConfig = session({
    secret: process.env.SESSION_SECRET || "dev-insecure-secret",
    store,
    resave: false,
    saveUninitialized: false,
    name: 'followupai.sid',
    proxy: true, // Trust the Replit proxy
    cookie: cookieConfig,
  });
  
  // Log session configuration
  console.log(`[SESSION] Environment: ${isReplit ? 'Replit (iframe)' : 'Local'}, secure: ${cookieConfig.secure}, sameSite: ${cookieConfig.sameSite}, store: ${store ? store.constructor.name : 'none'}`);
  
  return sessionConfig;
}

// Resend client helper
let connectionSettings: any;

async function getCredentials() {
  // Primary: allow standard ENV configuration
  const apiKeyFromEnv = process.env.RESEND_API_KEY;
  const fromEmailFromEnv = process.env.RESEND_FROM_EMAIL;
  if (apiKeyFromEnv) {
    return { apiKey: apiKeyFromEnv, fromEmail: fromEmailFromEnv };
  }

  // Fallback: Replit Connector
  const connectorAuth = getReplitConnectorAuth();
  if (!connectorAuth) {
    throw new Error('Resend credentials are not configured (set RESEND_API_KEY or Replit Connector)');
  }

  const { hostname, xReplitToken } = connectorAuth;

  connectionSettings = await fetch(
    'https://' + hostname + '/api/v2/connection?include_secrets=true&connector_names=resend',
    {
      headers: {
        'Accept': 'application/json',
        'X_REPLIT_TOKEN': xReplitToken
      }
    }
  ).then(res => res.json()).then(data => data.items?.[0]);

  if (!connectionSettings || (!connectionSettings.settings.api_key)) {
    throw new Error('Resend not connected');
  }
  return { apiKey: connectionSettings.settings.api_key, fromEmail: connectionSettings.settings.from_email };
}

async function getUncachableResendClient() {
  const credentials = await getCredentials();
  return {
    client: new Resend(credentials.apiKey),
    fromEmail: credentials.fromEmail
  };
}

// Generate verification token
export function generateToken(): string {
  return crypto.randomBytes(32).toString("hex");
}

// Hash password
export async function hashPassword(password: string): Promise<string> {
  const salt = await bcrypt.genSalt(10);
  return bcrypt.hash(password, salt);
}

// Compare password
export async function comparePassword(password: string, hash: string): Promise<boolean> {
  return bcrypt.compare(password, hash);
}

// Send verification email
export async function sendVerificationEmail(email: string, token: string, firstName: string) {
  if (!isEmailVerificationRequired()) {
    console.log(`[EMAIL] Skipping verification email for ${email}; verification is disabled in this environment.`);
    return;
  }

  if (!isEmailVerificationConfigured()) {
    const verificationUrl = `${process.env.APP_URL || 'http://localhost:5000'}/verify-email?token=${token}`;
    console.warn(`[EMAIL] Verification required but email provider not configured. Manual verify URL for ${email}: ${verificationUrl}`);
    return;
  }

  console.log(`[EMAIL] Attempting to send verification email to: ${email}`);
  
  try {
    const { client, fromEmail } = await getUncachableResendClient();
    console.log(`[EMAIL] Resend client initialized. From email: ${fromEmail || 'onboarding@resend.dev (fallback)'}`);
    
    const verificationUrl = `${process.env.APP_URL || 'http://localhost:5000'}/verify-email?token=${token}`;
    
    // Use custom domain if configured and verified, otherwise use test domain
    const senderEmail = (fromEmail && !fromEmail.includes('resend.dev')) 
      ? fromEmail 
      : 'Followup AI <onboarding@resend.dev>';
    
    const result = await client.emails.send({
      from: senderEmail,
      to: email,
      subject: "Verify Your Followup AI Account",
      html: `
        <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
          <h2 style="color: #6366f1;">Welcome to Followup AI, ${firstName}!</h2>
          <p>Thank you for signing up. Please verify your email address by clicking the link below:</p>
          <a href="${verificationUrl}" style="display: inline-block; padding: 12px 24px; background-color: #6366f1; color: white; text-decoration: none; border-radius: 4px; margin: 16px 0;">
            Verify Email Address
          </a>
          <p>Or copy and paste this link into your browser:</p>
          <p style="color: #666; font-size: 14px;">${verificationUrl}</p>
          <p style="margin-top: 24px; color: #666; font-size: 12px;">
            This link will expire in 24 hours. If you didn't create an account, please ignore this email.
          </p>
        </div>
      `,
    });
    
    console.log(`[EMAIL] ✓ Verification email sent successfully!`, result);
  } catch (error: any) {
    console.error("[EMAIL] ✗ Error sending verification email:", error);
    console.error("[EMAIL] Error details:", JSON.stringify(error, null, 2));
    throw error;
  }
}

// Send password reset email
export async function sendPasswordResetEmail(email: string, token: string, firstName: string) {
  // If email integration isn't configured, log the reset URL as a fallback
  if (!isEmailVerificationConfigured()) {
    const resetUrl = `${process.env.APP_URL || 'http://localhost:5000'}/reset-password?token=${token}`;
    console.warn(`[EMAIL] Resend not configured. Password reset URL for ${email}: ${resetUrl}`);
    return;
  }

  const { client, fromEmail } = await getUncachableResendClient();
  const resetUrl = `${process.env.APP_URL || 'http://localhost:5000'}/reset-password?token=${token}`;
  
  try {
    // Use custom domain if configured and verified, otherwise use test domain
    const senderEmail = (fromEmail && !fromEmail.includes('resend.dev')) 
      ? fromEmail 
      : 'Followup AI <onboarding@resend.dev>';
    
    await client.emails.send({
      from: senderEmail,
      to: email,
      subject: "Reset Your Followup AI Password",
      html: `
        <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
          <h2 style="color: #6366f1;">Password Reset Request</h2>
          <p>Hello ${firstName},</p>
          <p>We received a request to reset your password. Click the link below to create a new password:</p>
          <a href="${resetUrl}" style="display: inline-block; padding: 12px 24px; background-color: #6366f1; color: white; text-decoration: none; border-radius: 4px; margin: 16px 0;">
            Reset Password
          </a>
          <p>Or copy and paste this link into your browser:</p>
          <p style="color: #666; font-size: 14px;">${resetUrl}</p>
          <p style="margin-top: 24px; color: #666; font-size: 12px;">
            This link will expire in 1 hour. If you didn't request a password reset, please ignore this email.
          </p>
        </div>
      `,
    });
  } catch (error) {
    console.error("Error sending password reset email:", error);
    throw error;
  }
}

// Authentication middleware
export const isAuthenticated: RequestHandler = async (req, res, next) => {
  // Debug logging (only for auth-related endpoints to reduce noise)
  const isAuthEndpoint = req.path.includes('/api/auth/') || req.path.includes('/api/patient/') || req.path.includes('/api/doctor/');
  
  if (isAuthEndpoint) {
    console.log(`[AUTH] ${req.method} ${req.path} - Session ID: ${req.sessionID}, userId: ${(req.session as any)?.userId || 'none'}`);
  }
  
  if (req.session && (req.session as any).userId) {
    const userId = (req.session as any).userId;
    const user = await storage.getUser(userId);
    
    if (user) {
      req.user = {
        id: user.id,
        email: user.email!,
        role: user.role,
      };
      if (isAuthEndpoint) {
        console.log(`[AUTH] ✓ Authenticated: ${user.email} (${user.role})`);
      }
      return next();
    } else {
      console.log(`[AUTH] ✗ User not found for userId: ${userId}`);
    }
  } else {
    if (isAuthEndpoint) {
      console.log(`[AUTH] ✗ Unauthorized - no session or userId`);
      console.log(`[AUTH] Cookie header present: ${!!req.headers.cookie}`);
    }
  }
  
  return res.status(401).json({ message: "Unauthorized" });
};

// Doctor role middleware
export const isDoctor: RequestHandler = async (req, res, next) => {
  await isAuthenticated(req, res, () => {
    if (req.user?.role === "doctor") {
      return next();
    }
    return res.status(403).json({ message: "Forbidden: Doctor access required" });
  });
};

// Setup auth routes
export function setupAuth(app: Express) {
  app.set("trust proxy", 1);
  app.use(getSession());
}
