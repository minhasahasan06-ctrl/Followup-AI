import bcrypt from "bcryptjs";
import crypto from "crypto";
import { type Express, type RequestHandler } from "express";
import session from "express-session";
import connectPg from "connect-pg-simple";
import { storage } from "./storage";
import nodemailer from "nodemailer";

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

// Session configuration
export function getSession(maxAge?: number) {
  const isProduction = process.env.NODE_ENV === "production";
  const sessionTtl = maxAge || (1 * 24 * 60 * 60 * 1000); // Default 1 day in milliseconds
  const pgStore = connectPg(session);
  const sessionStore = new pgStore({
    conString: process.env.DATABASE_URL,
    createTableIfMissing: false,
    // Don't set a fixed ttl - let it use the session's cookie.maxAge automatically
    tableName: "sessions",
  });
  return session({
    secret: process.env.SESSION_SECRET!,
    store: sessionStore,
    resave: false,
    saveUninitialized: false,
    cookie: {
      httpOnly: true,
      secure: isProduction, // Only secure in production (HTTPS)
      sameSite: isProduction ? "strict" : "lax",
      maxAge: sessionTtl, // cookie maxAge is in milliseconds
    },
  });
}

// Email transporter (using nodemailer)
const createEmailTransporter = () => {
  // In development, you can use a service like Ethereal or configure SMTP
  return nodemailer.createTransport({
    host: process.env.SMTP_HOST || "smtp.gmail.com",
    port: parseInt(process.env.SMTP_PORT || "587"),
    secure: false,
    auth: {
      user: process.env.SMTP_USER,
      pass: process.env.SMTP_PASS,
    },
  });
};

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
  const transporter = createEmailTransporter();
  const verificationUrl = `${process.env.APP_URL || 'https://localhost:5000'}/verify-email?token=${token}`;
  
  const mailOptions = {
    from: process.env.SMTP_FROM || '"Followup AI" <noreply@followupai.com>',
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
  };

  try {
    await transporter.sendMail(mailOptions);
  } catch (error) {
    console.error("Error sending verification email:", error);
    throw error;
  }
}

// Send password reset email
export async function sendPasswordResetEmail(email: string, token: string, firstName: string) {
  const transporter = createEmailTransporter();
  const resetUrl = `${process.env.APP_URL || 'https://localhost:5000'}/reset-password?token=${token}`;
  
  const mailOptions = {
    from: process.env.SMTP_FROM || '"Followup AI" <noreply@followupai.com>',
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
  };

  try {
    await transporter.sendMail(mailOptions);
  } catch (error) {
    console.error("Error sending password reset email:", error);
    throw error;
  }
}

// Authentication middleware
export const isAuthenticated: RequestHandler = async (req, res, next) => {
  if (req.session && (req.session as any).userId) {
    const userId = (req.session as any).userId;
    const user = await storage.getUser(userId);
    
    if (user) {
      req.user = {
        id: user.id,
        email: user.email!,
        role: user.role,
      };
      return next();
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
