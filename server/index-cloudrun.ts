/**
 * Express Server Entry Point for Cloud Run - MINIMAL VERSION
 * 
 * This is a minimal Express server that:
 * 1. Handles Stytch authentication (magic links, SMS OTP)
 * 2. Acts as the API gateway for the frontend
 * 
 * Stripped down to only include auth routes - no heavy dependencies
 */
import express, { type Request, Response, NextFunction } from "express";
import cookieParser from "cookie-parser";
import cors from "cors";
import session from "express-session";
import { createServer } from "http";
import { stytchAuthRoutes, isStytchConfigured } from "./stytch";

console.log('[STARTUP] Starting minimal Express server for Cloud Run...');

const app = express();

// Trust proxy for Cloud Run (needed for secure cookies)
app.set('trust proxy', 1);

// CORS configuration for cross-domain frontend
const corsOrigins = process.env.CORS_ORIGINS?.split(',').map(o => o.trim()).filter(Boolean) || [];
console.log('[CORS] Allowed origins:', corsOrigins.length > 0 ? corsOrigins.join(', ') : 'none configured');

app.use(cors({
  origin: (origin, callback) => {
    if (!origin) {
      return callback(null, true);
    }
    if (corsOrigins.includes(origin)) {
      return callback(null, true);
    }
    console.warn(`[CORS] Rejected origin: ${origin}`);
    return callback(new Error('Not allowed by CORS'), false);
  },
  credentials: true,
  methods: ['GET', 'POST', 'PUT', 'PATCH', 'DELETE', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization', 'X-Requested-With'],
  exposedHeaders: ['Set-Cookie'],
}));

// Body parsing
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true, limit: '10mb' }));
app.use(cookieParser());

// Session configuration for cross-domain
const isProduction = process.env.NODE_ENV === 'production';
const hasCorsOrigins = corsOrigins.length > 0;

app.use(session({
  secret: process.env.SESSION_SECRET || 'dev-secret-change-in-production',
  resave: false,
  saveUninitialized: false,
  cookie: {
    secure: isProduction,
    httpOnly: true,
    sameSite: hasCorsOrigins ? 'none' : 'lax',
    maxAge: 24 * 60 * 60 * 1000,
  },
}));

console.log(`[SESSION] Configured - secure: ${isProduction}, sameSite: ${hasCorsOrigins ? 'none' : 'lax'}`);

// Health check endpoint - respond immediately
app.get('/health', (_req: Request, res: Response) => {
  res.json({ 
    status: 'healthy', 
    service: 'followupai-express',
    stytch: isStytchConfigured() ? 'configured' : 'not configured',
    timestamp: new Date().toISOString()
  });
});

// Startup probe for Cloud Run - respond immediately
app.get('/startup', (_req: Request, res: Response) => {
  res.json({ status: 'ready', service: 'followupai-express' });
});

// Root endpoint
app.get('/', (_req: Request, res: Response) => {
  res.json({ 
    service: 'followupai-express',
    version: '1.0.0',
    status: 'running'
  });
});

// Register Stytch auth routes
app.use('/api/auth', stytchAuthRoutes);
console.log('[AUTH] Stytch authentication routes registered at /api/auth/*');

// Error handler
app.use((err: any, _req: Request, res: Response, _next: NextFunction) => {
  console.error('[ERROR]', err.message || err);
  res.status(err.status || 500).json({ 
    error: err.message || 'Internal server error' 
  });
});

// 404 handler for API routes
app.use('/api/*', (_req: Request, res: Response) => {
  res.status(404).json({ error: 'API endpoint not found' });
});

// Start server
const PORT = parseInt(process.env.PORT || '8080', 10);
const server = createServer(app);

server.listen(PORT, '0.0.0.0', () => {
  console.log(`[EXPRESS] Server running on port ${PORT}`);
  console.log(`[STYTCH] ${isStytchConfigured() ? 'Configured and ready' : 'Not configured'}`);
});

// Handle graceful shutdown
process.on('SIGTERM', () => {
  console.log('[SHUTDOWN] Received SIGTERM, shutting down gracefully...');
  server.close(() => {
    console.log('[SHUTDOWN] Server closed');
    process.exit(0);
  });
});
