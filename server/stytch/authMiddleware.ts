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

export function getSessionCookieOptions(isProduction: boolean) {
  return {
    httpOnly: true,
    secure: isProduction,
    sameSite: "lax" as const,
    maxAge: SESSION_DURATION_MINUTES * 60 * 1000,
    path: "/",
  };
}

export const requireAuth: RequestHandler = async (req, res, next) => {
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

export const devBypassAuth: RequestHandler = async (req, res, next) => {
  if (process.env.NODE_ENV === "production") {
    return requireAuth(req, res, next);
  }

  const sessionToken = req.cookies?.[SESSION_COOKIE_NAME];
  
  if (sessionToken && isStytchConfigured()) {
    return requireAuth(req, res, next);
  }

  const patientId = req.params.patientId || req.query.patientId;
  if (patientId && String(patientId).startsWith("dev-")) {
    req.stytchUser = {
      id: String(patientId),
      email: "dev@followup.ai",
      role: "patient",
      stytchUserId: String(patientId),
    };
    console.log(`[STYTCH] Dev bypass for: ${patientId}`);
    return next();
  }

  return res.status(401).json({ error: "Authentication required" });
};
