import {
  CognitoIdentityProviderClient,
  SignUpCommand,
  InitiateAuthCommand,
  ConfirmSignUpCommand,
  ResendConfirmationCodeCommand,
  ForgotPasswordCommand,
  ConfirmForgotPasswordCommand,
  AdminUpdateUserAttributesCommand,
  AdminSetUserPasswordCommand,
  GetUserCommand,
} from "@aws-sdk/client-cognito-identity-provider";
import type { Request, Response, NextFunction, RequestHandler } from "express";
import jwt from "jsonwebtoken";
import jwksClient from "jwks-rsa";
import type { JwtPayload } from "jsonwebtoken";
import crypto from "crypto";
import { storage } from "./storage";

const REGION = process.env.AWS_COGNITO_REGION!;
const USER_POOL_ID = process.env.AWS_COGNITO_USER_POOL_ID!;
const CLIENT_ID = process.env.AWS_COGNITO_CLIENT_ID!;
const CLIENT_SECRET = process.env.AWS_COGNITO_CLIENT_SECRET!;

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

export async function verifyToken(token: string): Promise<JwtPayload> {
  const decoded = jwt.decode(token, { complete: true });
  if (!decoded || !decoded.header.kid) {
    throw new Error("Invalid token");
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
        
        // Validate token_use claim (must be 'id' for ID tokens)
        if (payload.token_use !== 'id') {
          reject(new Error('Invalid token type - expected ID token'));
          return;
        }
        
        // Validate audience matches our client ID
        if (payload.aud !== CLIENT_ID) {
          reject(new Error('Invalid token audience'));
          return;
        }
        
        resolve(payload);
      }
    );
  });
}

// Middleware to verify JWT token and attach user to request
export const isAuthenticated: RequestHandler = async (req, res, next) => {
  try {
    const authHeader = req.headers.authorization;
    if (!authHeader || !authHeader.startsWith("Bearer ")) {
      return res.status(401).json({ message: "Unauthorized" });
    }

    const token = authHeader.substring(7);
    const decoded = await verifyToken(token);
    
    // Attach user ID (Cognito sub) to request
    (req as any).userId = decoded.sub;
    (req as any).cognitoUsername = decoded["cognito:username"];
    
    next();
  } catch (error) {
    console.error("Auth error:", error);
    return res.status(401).json({ message: "Unauthorized" });
  }
};

// Middleware to verify doctor role
export const isDoctor: RequestHandler = async (req, res, next) => {
  try {
    const userId = (req as any).userId;
    if (!userId) {
      return res.status(401).json({ message: "Unauthorized" });
    }

    const user = await storage.getUser(userId);
    if (!user || user.role !== "doctor") {
      return res.status(403).json({ message: "Access denied. Doctor role required." });
    }

    next();
  } catch (error) {
    console.error("Doctor auth error:", error);
    return res.status(403).json({ message: "Access denied" });
  }
};

// Sign up a new user
export async function signUp(
  email: string,
  password: string,
  firstName: string,
  lastName: string,
  role: "patient" | "doctor",
  phoneNumber?: string
) {
  // Generate unique username (Cognito requires non-email username when email alias is configured)
  const username = crypto.randomUUID();
  
  const command = new SignUpCommand({
    ClientId: CLIENT_ID,
    Username: username,
    Password: password,
    SecretHash: computeSecretHash(username),
    UserAttributes: [
      { Name: "email", Value: email },
      { Name: "given_name", Value: firstName },
      { Name: "family_name", Value: lastName },
      { Name: "name", Value: `${firstName} ${lastName}` },
      // Standard OIDC attributes
      { Name: "phone_number", Value: phoneNumber || "+10000000000" },
      { Name: "birthdate", Value: "1990-01-01" },
      { Name: "gender", Value: "Not specified" },
      { Name: "zoneinfo", Value: "UTC" },
      // Custom attributes required by this specific Cognito User Pool (error shows these are required)
      { Name: "custom:profileUrl", Value: "https://example.com/profile" },
      { Name: "custom:addresses", Value: "US" },
      { Name: "custom:phoneNumbers", Value: phoneNumber || "+10000000000" },
      { Name: "custom:timezone", Value: "America/Los_Angeles" },
      // Note: Role is tracked in our database, not in Cognito
    ],
  });

  const response = await cognitoClient.send(command);
  return { ...response, username }; // Return username for metadata storage
}

// Confirm sign up with verification code
export async function confirmSignUp(email: string, code: string) {
  const command = new ConfirmSignUpCommand({
    ClientId: CLIENT_ID,
    Username: email,
    ConfirmationCode: code,
    SecretHash: computeSecretHash(email),
  });

  const response = await cognitoClient.send(command);
  return response;
}

// Resend verification code
export async function resendConfirmationCode(email: string) {
  const command = new ResendConfirmationCodeCommand({
    ClientId: CLIENT_ID,
    Username: email,
    SecretHash: computeSecretHash(email),
  });

  const response = await cognitoClient.send(command);
  return response;
}

// Sign in user
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

// Forgot password - send reset code
export async function forgotPassword(email: string) {
  const command = new ForgotPasswordCommand({
    ClientId: CLIENT_ID,
    Username: email,
    SecretHash: computeSecretHash(email),
  });

  const response = await cognitoClient.send(command);
  return response;
}

// Confirm forgot password with code
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

// Get user info from access token
export async function getUserInfo(accessToken: string) {
  const command = new GetUserCommand({
    AccessToken: accessToken,
  });

  const response = await cognitoClient.send(command);
  return response;
}
