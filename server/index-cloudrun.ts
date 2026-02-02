/**
 * Express Server Entry Point for Cloud Run
 * 
 * This is a standalone Express server that:
 * 1. Handles Stytch authentication (magic links, SMS OTP)
 * 2. Proxies requests to Python FastAPI backend on Cloud Run
 * 3. Serves as the API gateway for the frontend
 * 
 * Does NOT start the Python backend (runs separately on Cloud Run)
 */
import express, { type Request, Response, NextFunction } from "express";
import cookieParser from "cookie-parser";
import cors from "cors";
import session from "express-session";
import { registerRoutes } from "./routes";
import { log } from "./vite";
import { seedDatabase } from "./seed";
import { isStytchConfigured } from "./stytch";
import { runConfigGuard } from "./config_guard";

// HIPAA Config Guard
try {
  runConfigGuard(true);
} catch (error) {
  console.error('[STARTUP] Config guard failed - exiting');
  process.exit(1);
}

const app = express();

// Trust proxy for Cloud Run (needed for secure cookies)
app.set('trust proxy', 1);

// CORS configuration for cross-domain frontend
const corsOrigins = process.env.CORS_ORIGINS?.split(',').map(o => o.trim()).filter(Boolean) || [];
console.log('[CORS] Allowed origins:', corsOrigins.length > 0 ? corsOrigins.join(', ') : 'none configured');

app.use(cors({
  origin: (origin, callback) => {
    // Allow requests with no origin (mobile apps, Postman, etc.)
    if (!origin) {
      return callback(null, true);
    }
    // Check if origin is in allowed list
    if (corsOrigins.includes(origin)) {
      return callback(null, true);
    }
    // Reject other origins in production
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
    maxAge: 24 * 60 * 60 * 1000, // 24 hours
  },
}));

console.log(`[SESSION] Configured - secure: ${isProduction}, sameSite: ${hasCorsOrigins ? 'none' : 'lax'}`);

// Health check endpoint
app.get('/health', (_req: Request, res: Response) => {
  res.json({ 
    status: 'healthy', 
    service: 'followupai-express',
    stytch: isStytchConfigured() ? 'configured' : 'not configured',
    timestamp: new Date().toISOString()
  });
});

// Startup probe for Cloud Run
app.get('/startup', (_req: Request, res: Response) => {
  res.json({ status: 'ready', service: 'followupai-express' });
});

// Start server
async function startServer() {
  console.log('[STARTUP] Initializing Express server for Cloud Run...');
  
  // Seed database
  try {
    await seedDatabase();
    console.log('[STARTUP] Database seeded');
  } catch (error) {
    console.warn('[STARTUP] Database seed skipped:', error);
  }
  
  // Register API routes (including Stytch auth)
  const server = await registerRoutes(app);
  
  // Error handler
  app.use((err: any, _req: Request, res: Response, _next: NextFunction) => {
    console.error('[ERROR]', err);
    res.status(err.status || 500).json({ 
      error: err.message || 'Internal server error' 
    });
  });
  
  // 404 handler for API routes
  app.use('/api/*', (_req: Request, res: Response) => {
    res.status(404).json({ error: 'API endpoint not found' });
  });
  
  const PORT = parseInt(process.env.PORT || '8080', 10);
  
  server.listen(PORT, '0.0.0.0', () => {
    console.log(`[EXPRESS] Server running on port ${PORT}`);
    console.log(`[STYTCH] ${isStytchConfigured() ? 'Configured and ready' : 'Not configured'}`);
  });
}

startServer().catch((error) => {
  console.error('[FATAL] Failed to start server:', error);
  process.exit(1);
});
