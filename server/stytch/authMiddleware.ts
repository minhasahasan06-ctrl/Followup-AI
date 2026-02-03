/**
 * Stytch Consumer Authentication Middleware
 * 
 * Uses Stytch CONSUMER SDK for session validation:
 * - sessions.authenticate() - Validates session tokens from HttpOnly cookies
 * - RBAC via user.trusted_metadata.role (admin, doctor, patient)
 * - No B2B organization logic - individual user sessions only
 * - M2M token validation for service-to-service authentication
 */
import { Request, Response, NextFunction, RequestHandler } from "express";
import { getStytchClient, isStytchConfigured, validateM2MToken } from "./stytchClient";
import { storage } from "../storage";

export interface StytchUser {
  id: string;
  email: string;
  phone?: string;
  role: "doctor" | "patient" | "admin";
  stytchUserId: string;
  sessionToken?: string;
}

declare global {
  namespace Express {
    interface Request {
      stytchUser?: StytchUser;
      m2mClient?: {
        clientId: string;
        scopes: string[];
      };
    }
  }
}

const SESSION_COOKIE_NAME = "stytch_session_token";
const SESSION_DURATION_MINUTES = 60 * 24;

/**
 * Get cookie options for session tokens
 * 
 * Cross-domain auth (Vercel → Cloud Run):
 * - SameSite=None + Secure required for cross-origin cookies
 * - Only use SameSite=None when CORS_ORIGINS is configured (indicates cross-domain setup)
 * 
 * Same-origin auth (development):
 * - SameSite=Lax for CSRF protection
 * - Secure only in production (localhost doesn't use HTTPS)
 */
export function getSessionCookieOptions(isProduction: boolean) {
  // Check if cross-domain auth is configured
  const corsOrigins = process.env.CORS_ORIGINS?.split(',').map(o => o.trim()).filter(Boolean) || [];
  const isCrossDomain = corsOrigins.length > 0;
  
  return {
    httpOnly: true,
    // SameSite=None requires Secure=true, and is needed for cross-domain cookies
    secure: isProduction || isCrossDomain,
    // Use None for cross-domain, Lax for same-origin (better CSRF protection)
    sameSite: isCrossDomain ? "none" as const : "lax" as const,
    maxAge: SESSION_DURATION_MINUTES * 60 * 1000,
    path: "/",
  };
}

export const requireAuth: RequestHandler = async (req, res, next) => {
  // Check for Express session-based auth first (dev login)
  const session = (req as any).session;
  console.log(`[AUTH] Session check - hasSession: ${!!session}, userId: ${session?.userId}, hasUser: ${!!session?.user}`);
  
  if (session?.user && session?.userId) {
    const sessionUser = session.user;
    console.log(`[AUTH] Dev session found for: ${sessionUser.email}`);
    (req as any).stytchUser = {
      id: sessionUser.id,
      email: sessionUser.email,
      phone: sessionUser.phone,
      role: sessionUser.role || 'patient',
      firstName: sessionUser.firstName,
      lastName: sessionUser.lastName,
    };
    (req as any).user = sessionUser;
    return next();
  }

  if (!isStytchConfigured()) {
    console.warn("[STYTCH] Auth not configured, allowing request in dev mode");
    if (process.env.NODE_ENV !== "production") {
      return next();
    }
    return res.status(503).json({ error: "Authentication service not configured" });
  }

  const sessionToken = req.cookies?.[SESSION_COOKIE_NAME];
  
  if (!sessionToken) {
    return res.status(401).json({ error: "No session token provided" });
  }

  try {
    const client = getStytchClient();
    const authResponse = await client.sessions.authenticate({
      session_token: sessionToken,
      session_duration_minutes: SESSION_DURATION_MINUTES,
    });

    const stytchUserId = authResponse.session.user_id;
    const userEmail = authResponse.user.emails?.[0]?.email || "";
    const userPhone = authResponse.user.phone_numbers?.[0]?.phone_number;

    const trustedMetadata = authResponse.user.trusted_metadata as Record<string, any> || {};
    const role = (trustedMetadata.role as "doctor" | "patient" | "admin") || "patient";

    let dbUser = await storage.getUserByEmail(userEmail);
    if (!dbUser && userEmail) {
      dbUser = await storage.createUser({
        email: userEmail,
        firstName: authResponse.user.name?.first_name || "",
        lastName: authResponse.user.name?.last_name || "",
        role: role,
        phone: userPhone,
      });
    }

    req.stytchUser = {
      id: dbUser?.id || stytchUserId,
      email: userEmail,
      phone: userPhone,
      role: role,
      stytchUserId: stytchUserId,
      sessionToken: authResponse.session_token,
    };

    if (authResponse.session_token !== sessionToken) {
      const isProduction = process.env.NODE_ENV === "production";
      res.cookie(SESSION_COOKIE_NAME, authResponse.session_token, getSessionCookieOptions(isProduction));
    }

    next();
  } catch (error: any) {
    console.error("[STYTCH] Session authentication failed:", error.message);
    res.clearCookie(SESSION_COOKIE_NAME);
    return res.status(401).json({ error: "Invalid or expired session" });
  }
};

export function requireRole(...allowedRoles: Array<"doctor" | "patient" | "admin">): RequestHandler {
  return async (req, res, next) => {
    if (!req.stytchUser) {
      return res.status(401).json({ error: "Authentication required" });
    }

    if (!allowedRoles.includes(req.stytchUser.role)) {
      return res.status(403).json({
        error: `Access denied. Required role: ${allowedRoles.join(" or ")}`,
      });
    }

    next();
  };
}

export const requireDoctor: RequestHandler = requireRole("doctor", "admin");
export const requirePatient: RequestHandler = requireRole("patient");
export const requireAdmin: RequestHandler = requireRole("admin");

export const requireM2MAuth = (requiredScopes: string[] = []): RequestHandler => {
  return async (req, res, next) => {
    const authHeader = req.headers.authorization;
    
    if (!authHeader?.startsWith("Bearer ")) {
      return res.status(401).json({ error: "Missing Bearer token" });
    }

    const token = authHeader.slice(7);

    try {
      const result = await validateM2MToken(token, requiredScopes);
      req.m2mClient = {
        clientId: result.clientId,
        scopes: result.scopes,
      };
      next();
    } catch (error: any) {
      console.error("[STYTCH] M2M token validation failed:", error.message);
      
      if (error.error_type === "insufficient_scopes") {
        return res.status(403).json({
          error: "Insufficient permissions",
          required_scopes: requiredScopes,
        });
      }
      
      return res.status(401).json({ error: "Invalid M2M token" });
    }
  };
};

export const optionalAuth: RequestHandler = async (req, res, next) => {
  const sessionToken = req.cookies?.[SESSION_COOKIE_NAME];
  
  if (!sessionToken || !isStytchConfigured()) {
    return next();
  }

  try {
    const client = getStytchClient();
    const authResponse = await client.sessions.authenticate({
      session_token: sessionToken,
      session_duration_minutes: SESSION_DURATION_MINUTES,
    });

    const stytchUserId = authResponse.session.user_id;
    const userEmail = authResponse.user.emails?.[0]?.email || "";
    const userPhone = authResponse.user.phone_numbers?.[0]?.phone_number;
    const trustedMetadata = authResponse.user.trusted_metadata as Record<string, any> || {};
    const role = (trustedMetadata.role as "doctor" | "patient" | "admin") || "patient";

    const dbUser = await storage.getUserByEmail(userEmail);

    req.stytchUser = {
      id: dbUser?.id || stytchUserId,
      email: userEmail,
      phone: userPhone,
      role: role,
      stytchUserId: stytchUserId,
      sessionToken: authResponse.session_token,
    };

    if (authResponse.session_token !== sessionToken) {
      const isProduction = process.env.NODE_ENV === "production";
      res.cookie(SESSION_COOKIE_NAME, authResponse.session_token, getSessionCookieOptions(isProduction));
    }
  } catch (error) {
    res.clearCookie(SESSION_COOKIE_NAME);
  }

  next();
};

// Note: devBypassAuth removed for production security - use requireAuth instead
