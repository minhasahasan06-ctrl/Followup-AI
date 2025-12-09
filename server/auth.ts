import { type RequestHandler } from "express";
import session from "express-session";
import connectPg from "connect-pg-simple";
import { storage } from "./storage";

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
    console.warn("[SESSION] DATABASE_URL not set – using MemoryStore (dev only). Sessions won't persist across restarts.");
    store = new session.MemoryStore();
  }

  const sessionConfig = session({
    secret: process.env.SESSION_SECRET || "dev-insecure-secret",
    store,
    resave: false,
    saveUninitialized: false, // Only save sessions that have been modified (e.g., have userId set)
    name: 'followupai.sid',
    proxy: true,
    cookie: {
      httpOnly: true,
      // Use Secure cookies in production (HTTPS) to ensure browsers accept the session
      secure: isProduction,
      sameSite: "lax",
      maxAge: sessionTtl,
      domain: undefined,
      path: '/',
    },
  });
  
  // Log session configuration
  console.log(`[SESSION] Session configured - secure: ${isProduction}, sameSite: lax, store: ${store ? store.constructor.name : 'none'}`);
  
  return sessionConfig;
}

// Authentication middleware - checks session for userId and sets req.user
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
    } else {
      console.log(`[AUTH] ✗ User not found for userId: ${userId}`);
    }
  } else {
    console.log(`[AUTH] ✗ Unauthorized - no session or userId`);
  }
  
  return res.status(401).json({ message: "Unauthorized" });
};

// Doctor role middleware
export const isDoctor: RequestHandler = async (req, res, next) => {
  try {
    const userId = req.user?.id;
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

// Patient role middleware
export const isPatient: RequestHandler = async (req, res, next) => {
  try {
    const userId = req.user?.id;
    if (!userId) {
      return res.status(401).json({ message: "Unauthorized" });
    }

    const user = await storage.getUser(userId);
    if (!user || user.role !== "patient") {
      return res.status(403).json({ message: "Access denied. Patient role required." });
    }

    next();
  } catch (error) {
    console.error("Patient auth error:", error);
    return res.status(403).json({ message: "Access denied" });
  }
};

// Authentication middleware with dev bypass for autopilot routes
// In development, allows dev-patient-* pattern IDs to pass through when session auth fails
export const isAuthenticatedOrDevBypass: RequestHandler = async (req, res, next) => {
  // First try normal session authentication
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
  
  // Dev bypass: In non-production, allow requests with dev-patient-* patientId in route params
  const isProduction = process.env.NODE_ENV === 'production';
  if (!isProduction) {
    const patientId = req.params.patientId;
    if (patientId && patientId.startsWith('dev-patient-')) {
      // Set up a dev user context
      req.user = {
        id: patientId,
        email: 'dev-patient@followup.ai',
        role: 'patient',
      };
      console.log(`[AUTH] Dev bypass for autopilot route: ${patientId}`);
      return next();
    }
  }
  
  console.log(`[AUTH] ✗ Unauthorized - no session or dev bypass available`);
  return res.status(401).json({ message: "Unauthorized" });
};
