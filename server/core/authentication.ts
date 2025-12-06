/**
 * Unified Authentication & Authorization Module - HIPAA-Compliant
 * Consolidates all server-side authentication logic
 * 
 * SECURITY REQUIREMENTS:
 * - JWT verification using AWS Cognito JWKS
 * - Comprehensive audit logging
 * - Secure error handling
 * - No information leakage
 */

import { type RequestHandler, Request, Response, NextFunction } from "express";
import {
  CognitoIdentityProviderClient,
  SignUpCommand,
  InitiateAuthCommand,
  ConfirmSignUpCommand,
  ResendConfirmationCodeCommand,
  ForgotPasswordCommand,
  ConfirmForgotPasswordCommand,
  AdminUpdateUserAttributesCommand,
  AdminConfirmSignUpCommand,
  GetUserCommand,
  DescribeUserPoolCommand,
} from "@aws-sdk/client-cognito-identity-provider";
import jwt from "jsonwebtoken";
import jwksClient from "jwks-rsa";
import type { JwtPayload } from "jsonwebtoken";
import crypto from "crypto";
import { storage } from "../storage";

const REGION = process.env.AWS_COGNITO_REGION!;
const USER_POOL_ID = process.env.AWS_COGNITO_USER_POOL_ID!;
const CLIENT_ID = process.env.AWS_COGNITO_CLIENT_ID!;
const CLIENT_SECRET = process.env.AWS_COGNITO_CLIENT_SECRET!;
const IS_PRODUCTION = process.env.NODE_ENV === "production";

// Extend Express Request to include user
declare global {
  namespace Express {
    interface Request {
      user?: {
        id: string;
        email: string;
        role: string;
      };
      userId?: string;
      cognitoUsername?: string;
    }
  }
}

// Compute SECRET_HASH for Cognito API calls
function computeSecretHash(username: string): string {
  const message = username + CLIENT_ID;
  const hmac = crypto.createHmac('sha256', CLIENT_SECRET);
  hmac.update(message);
  return hmac.digest('base64');
}

export const cognitoClient = new CognitoIdentityProviderClient({
  region: REGION,
  credentials: {
    accessKeyId: process.env.AWS_ACCESS_KEY_ID!,
    secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY!,
  },
});

// JWKS client for JWT verification
const jwks = jwksClient({
  jwksUri: `https://cognito-idp.${REGION}.amazonaws.com/${USER_POOL_ID}/.well-known/jwks.json`,
  cache: true,
  rateLimit: true,
  cacheMaxAge: 3600000, // 1 hour
});

function getSigningKey(kid: string): Promise<string> {
  return new Promise((resolve, reject) => {
    jwks.getSigningKey(kid, (err, key) => {
      if (err) {
        reject(err);
      } else {
        const signingKey = key?.getPublicKey();
        resolve(signingKey!);
      }
    });
  });
}

/**
 * Verify JWT token with comprehensive security checks
 */
export async function verifyToken(token: string): Promise<JwtPayload> {
  // Check configuration
  if (!REGION || !USER_POOL_ID || !CLIENT_ID) {
    if (IS_PRODUCTION) {
      throw new Error("AWS Cognito not configured in production");
    }
    // Development mode - require secure secret
    const devSecret = process.env.DEV_MODE_SECRET;
    if (!devSecret || devSecret.length < 32) {
      throw new Error("Development mode requires DEV_MODE_SECRET (min 32 chars)");
    }
    // Verify with dev secret
    return jwt.verify(token, devSecret) as JwtPayload;
  }

  const decoded = jwt.decode(token, { complete: true });
  if (!decoded || !decoded.header.kid) {
    throw new Error("Invalid token format");
  }

  const signingKey = await getSigningKey(decoded.header.kid);
  
  return new Promise((resolve, reject) => {
    jwt.verify(
      token,
      signingKey,
      {
        algorithms: ["RS256"],
        issuer: `https://cognito-idp.${REGION}.amazonaws.com/${USER_POOL_ID}`,
        audience: CLIENT_ID,
      },
      (err, decoded) => {
        if (err) {
          reject(err);
          return;
        }
        
        const payload = decoded as JwtPayload;
        
        // Validate token_use claim
        if (payload.token_use !== 'id' && payload.token_use !== 'access') {
          reject(new Error('Invalid token type'));
          return;
        }
        
        resolve(payload);
      }
    );
  });
}

/**
 * Middleware to verify JWT token and attach user to request
 * HIPAA-compliant with audit logging
 */
export const isAuthenticated: RequestHandler = async (req, res, next) => {
  try {
    const authHeader = req.headers.authorization;
    if (!authHeader || !authHeader.startsWith("Bearer ")) {
      // Log failed authentication attempt
      console.log(`[AUTH] Unauthorized - missing token from ${req.ip}`);
      return res.status(401).json({ message: "Unauthorized" });
    }

    const token = authHeader.substring(7);
    const decoded = await verifyToken(token);
    
    // Attach user ID (Cognito sub) to request
    req.userId = decoded.sub;
    req.cognitoUsername = decoded["cognito:username"] as string;
    
    // Get user from database for role information
    if (decoded.sub) {
      const user = await storage.getUser(decoded.sub);
      if (user) {
        req.user = {
          id: user.id,
          email: user.email!,
          role: user.role,
        };
      }
    }
    
    next();
  } catch (error) {
    // Log error securely without exposing details
    console.error(`[AUTH] Authentication failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    return res.status(401).json({ message: "Unauthorized" });
  }
};

/**
 * Middleware to verify doctor role
 */
export const isDoctor: RequestHandler = async (req, res, next) => {
  try {
    const userId = req.userId || req.user?.id;
    if (!userId) {
      return res.status(401).json({ message: "Unauthorized" });
    }

    const user = await storage.getUser(userId);
    if (!user || user.role !== "doctor") {
      console.log(`[AUTH] Access denied - user ${userId} is not a doctor`);
      return res.status(403).json({ message: "Access denied. Doctor role required." });
    }

    req.user = {
      id: user.id,
      email: user.email!,
      role: user.role,
    };

    next();
  } catch (error) {
    console.error("Doctor auth error:", error);
    return res.status(403).json({ message: "Access denied" });
  }
};

/**
 * Middleware to verify patient role
 */
export const isPatient: RequestHandler = async (req, res, next) => {
  try {
    const userId = req.userId || req.user?.id;
    if (!userId) {
      return res.status(401).json({ message: "Unauthorized" });
    }

    const user = await storage.getUser(userId);
    if (!user || user.role !== "patient") {
      console.log(`[AUTH] Access denied - user ${userId} is not a patient`);
      return res.status(403).json({ message: "Access denied. Patient role required." });
    }

    req.user = {
      id: user.id,
      email: user.email!,
      role: user.role,
    };

    next();
  } catch (error) {
    console.error("Patient auth error:", error);
    return res.status(403).json({ message: "Access denied" });
  }
};

// Export Cognito functions (keeping existing API)
export async function signUp(
  email: string,
  password: string,
  firstName: string,
  lastName: string,
  role: "patient" | "doctor",
  phoneNumber?: string
) {
  const username = crypto.randomUUID();
  
  const command = new SignUpCommand({
    ClientId: CLIENT_ID,
    Username: username,
    Password: password,
    SecretHash: computeSecretHash(username),
    UserAttributes: [
      { Name: "email", Value: email },
      { Name: "name", Value: `${firstName} ${lastName}` },
      { Name: "phone_number", Value: phoneNumber || "+10000000000" },
      { Name: "birthdate", Value: "1990-01-01" },
      { Name: "gender", Value: "prefer not to say" },
      { Name: "zoneinfo", Value: "America/Los_Angeles" },
      { Name: "profile", Value: "https://followupai.com/profile/pending" },
      { Name: "address", Value: "United States" },
      { Name: "given_name", Value: firstName },
      { Name: "family_name", Value: lastName },
    ],
  });

  try {
    const response = await cognitoClient.send(command);
    console.log(`[COGNITO] User signed up: ${email} (username: ${username})`);
    return { ...response, username };
  } catch (error: any) {
    console.error(`[COGNITO] Signup error for ${email}:`, error.message);
    throw error;
  }
}

export async function confirmSignUp(email: string, code: string, username?: string) {
  const cognitoUsername = username || email;
  
  const command = new ConfirmSignUpCommand({
    ClientId: CLIENT_ID,
    Username: cognitoUsername,
    ConfirmationCode: code,
    SecretHash: computeSecretHash(cognitoUsername),
  });

  try {
    const response = await cognitoClient.send(command);
    console.log(`[COGNITO] Email confirmed: ${email}`);
    return response;
  } catch (error: any) {
    console.error(`[COGNITO] Confirmation error for ${email}:`, error.message);
    throw error;
  }
}

export async function signIn(email: string, password: string) {
  const command = new InitiateAuthCommand({
    ClientId: CLIENT_ID,
    AuthFlow: "USER_PASSWORD_AUTH",
    AuthParameters: {
      USERNAME: email,
      PASSWORD: password,
      SECRET_HASH: computeSecretHash(email),
    },
  });

  const response = await cognitoClient.send(command);
  return response.AuthenticationResult;
}

export async function forgotPassword(email: string) {
  const command = new ForgotPasswordCommand({
    ClientId: CLIENT_ID,
    Username: email,
    SecretHash: computeSecretHash(email),
  });

  const response = await cognitoClient.send(command);
  return response;
}

export async function confirmForgotPassword(
  email: string,
  code: string,
  newPassword: string
) {
  const command = new ConfirmForgotPasswordCommand({
    ClientId: CLIENT_ID,
    Username: email,
    ConfirmationCode: code,
    Password: newPassword,
    SecretHash: computeSecretHash(email),
  });

  const response = await cognitoClient.send(command);
  return response;
}

// Re-export other Cognito functions from cognitoAuth.ts for backward compatibility
export {
  adminConfirmSignUp,
  resendConfirmationCode,
  adminConfirmUser,
  getUserInfo,
  describeUserPoolSchema,
} from "../cognitoAuth";
