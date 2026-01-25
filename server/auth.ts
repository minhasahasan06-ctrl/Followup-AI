import { type RequestHandler } from "express";
import session from "express-session";
import connectPg from "connect-pg-simple";
import { storage } from "./storage";
import { getStytchClient, isStytchConfigured } from "./stytch/stytchClient";

const DEFAULT_SESSION_SECRET = "dev-insecure-secret";
const STYTCH_SESSION_COOKIE = "stytch_session_token";
const SESSION_DURATION_MINUTES = 60 * 24;
const ALLOWED_ROLES = ["doctor", "patient", "admin"] as const;
type AllowedRole = typeof ALLOWED_ROLES[number];

declare global {
  namespace Express {
    interface Request {
      user?: {
        id: string;
        email: string;
        role: string;
        stytchUserId?: string;
      };
    }
  }
}

function validateRole(role: string | undefined): AllowedRole {
  if (role && ALLOWED_ROLES.includes(role as AllowedRole)) {
    return role as AllowedRole;
  }
  return "patient";
}

function logAuthEvent(event: string, details: Record<string, any>) {
  const timestamp = new Date().toISOString();
  console.log(`[AUTH-AUDIT] ${timestamp} ${event}:`, JSON.stringify(details));
}

export function getSession(maxAge?: number) {
  const isProduction = process.env.NODE_ENV === "production";
  const sessionTtl = maxAge || (1 * 24 * 60 * 60 * 1000);

  const configuredSecret = process.env.SESSION_SECRET || DEFAULT_SESSION_SECRET;
  if (isProduction && configuredSecret === DEFAULT_SESSION_SECRET) {
    throw new Error("SESSION_SECRET must be configured in production environments");
  }

  let store: session.Store | undefined;
  if (process.env.DATABASE_URL) {
    const PgStore = connectPg(session);
    store = new PgStore({
      conString: process.env.DATABASE_URL,
      createTableIfMissing: true,
      tableName: "sessions",
    });
  } else {
    console.warn("[SESSION] DATABASE_URL not set – using MemoryStore (dev only). Sessions won't persist across restarts.");
    store = new session.MemoryStore();
  }

  const sessionConfig = session({
    secret: configuredSecret,
    store,
    resave: false,
    saveUninitialized: false,
    name: 'followupai.sid',
    proxy: true,
    cookie: {
      httpOnly: true,
      secure: isProduction,
      sameSite: "lax",
      maxAge: sessionTtl,
      domain: undefined,
      path: '/',
    },
  });
  
  console.log(`[SESSION] Session configured - secure: ${isProduction}, sameSite: lax, store: ${store ? store.constructor.name : 'none'}`);
  
  return sessionConfig;
}

async function authenticateWithStytch(req: any, res: any): Promise<boolean> {
  const sessionToken = req.cookies?.[STYTCH_SESSION_COOKIE];
  
  if (!sessionToken || !isStytchConfigured()) {
    return false;
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
    const role = validateRole(trustedMetadata.role);

    let dbUser = await storage.getUserByEmail(userEmail);
    if (!dbUser && userEmail) {
      dbUser = await storage.createUser({
        email: userEmail,
        firstName: authResponse.user.name?.first_name || "",
        lastName: authResponse.user.name?.last_name || "",
        role: role === "admin" ? "doctor" : role,
        phone: userPhone,
      });
      
      logAuthEvent("USER_CREATED", { 
        email: userEmail, 
        role, 
        stytchUserId,
        source: "stytch_session" 
      });
    }

    req.user = {
      id: dbUser?.id || stytchUserId,
      email: userEmail,
      role: role,
      stytchUserId: stytchUserId,
    };

    if (authResponse.session_token !== sessionToken) {
      const isProduction = process.env.NODE_ENV === "production";
      res.cookie(STYTCH_SESSION_COOKIE, authResponse.session_token, {
        httpOnly: true,
        secure: isProduction,
        sameSite: "lax" as const,
        maxAge: SESSION_DURATION_MINUTES * 60 * 1000,
        path: "/",
      });
    }

    return true;
  } catch (error: any) {
    logAuthEvent("SESSION_VALIDATION_FAILED", { 
      error: error.message,
      ip: req.ip,
      userAgent: req.get('user-agent') 
    });
    res.clearCookie(STYTCH_SESSION_COOKIE);
    return false;
  }
}

async function authenticateWithLegacySession(req: any): Promise<boolean> {
  const isProduction = process.env.NODE_ENV === 'production';
  const legacyEnabled = process.env.LEGACY_SESSION_ENABLED === 'true';
  
  if (isProduction && !legacyEnabled) {
    return false;
  }
  
  if (isStytchConfigured() && !legacyEnabled) {
    return false;
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
      
      if (!isProduction) {
        console.warn(`[AUTH] Legacy session fallback used for user: ${user.email}`);
      }
      
      return true;
    }
  }
  return false;
}

export const isAuthenticated: RequestHandler = async (req, res, next) => {
  if (await authenticateWithStytch(req, res)) {
    return next();
  }

  if (await authenticateWithLegacySession(req)) {
    return next();
  }
  
  logAuthEvent("AUTH_FAILED", { 
    path: req.path, 
    ip: req.ip,
    hasStytchCookie: !!req.cookies?.[STYTCH_SESSION_COOKIE],
    hasLegacySession: !!(req.session as any)?.userId
  });
  
  return res.status(401).json({ message: "Unauthorized" });
};

function requireRole(role: AllowedRole): RequestHandler {
  return async (req, res, next) => {
    try {
      const userRole = req.user?.role;
      if (!userRole) {
        return res.status(401).json({ message: "Unauthorized" });
      }

      if (userRole !== role && userRole !== "admin") {
        logAuthEvent("ACCESS_DENIED", { 
          userId: req.user?.id,
          email: req.user?.email,
          userRole,
          requiredRole: role,
          path: req.path 
        });
        return res.status(403).json({ message: `Access denied. ${role.charAt(0).toUpperCase()}${role.slice(1)} role required.` });
      }

      next();
    } catch (error) {
      console.error(`${role} auth error:`, error);
      return res.status(403).json({ message: "Access denied" });
    }
  };
}

export const isDoctor = requireRole("doctor");
export const isPatient = requireRole("patient");
export const isAdmin = requireRole("admin");

// Note: isAuthenticatedOrDevBypass removed for production - use isAuthenticated instead
