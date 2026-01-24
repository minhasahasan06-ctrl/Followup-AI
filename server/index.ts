import express, { type Request, Response, NextFunction } from "express";
import { spawn, ChildProcess } from "child_process";
import cookieParser from "cookie-parser";
import { registerRoutes } from "./routes";
import { setupVite, serveStatic, log } from "./vite";
import { seedDatabase } from "./seed";
import { isStytchConfigured } from "./stytch";
import { runConfigGuard } from "./config_guard";
import { safeLogger } from "./safe_logger";

// HIPAA Config Guard - Run BEFORE any other initialization
// This must be the first thing that runs to ensure we're in a safe environment
try {
  runConfigGuard(true);
} catch (error) {
  console.error('[STARTUP] Config guard failed - exiting');
  process.exit(1);
}

// Python backend configuration
const PYTHON_PORT = 8000;
const PYTHON_HEALTH_URL = `http://localhost:${PYTHON_PORT}/`;
const PYTHON_STARTUP_TIMEOUT_MS = 120000; // 2 minutes max wait
const PYTHON_HEALTH_POLL_INTERVAL_MS = 500; // Check every 500ms

let pythonProcess: ChildProcess | null = null;

// Start Python FastAPI backend on port 8000
function startPythonBackend(): ChildProcess {
  const proc = spawn('python', ['-m', 'uvicorn', 'app.main:app', '--host', '0.0.0.0', '--port', String(PYTHON_PORT)], {
    cwd: process.cwd(),
    stdio: ['ignore', 'pipe', 'pipe'],
    env: { ...process.env, PYTHONUNBUFFERED: '1' }
  });

  proc.stdout?.on('data', (data: Buffer) => {
    const msg = data.toString().trim();
    if (msg) log(`[Python] ${msg}`);
  });

  proc.stderr?.on('data', (data: Buffer) => {
    const msg = data.toString().trim();
    // Filter out verbose TensorFlow/CUDA warnings
    if (msg && !msg.includes('computation placer') && !msg.includes('cuDNN') && !msg.includes('cuBLAS')) {
      log(`[Python] ${msg}`);
    }
  });

  proc.on('error', (err) => {
    log(`[Python] Failed to start: ${err.message}`);
  });

  proc.on('exit', (code) => {
    log(`[Python] Process exited with code ${code}`);
    // Attempt restart after 5 seconds if it crashes (only if not intentional shutdown)
    if (code !== 0 && pythonProcess === proc) {
      setTimeout(() => {
        log('[Python] Attempting restart...');
        pythonProcess = startPythonBackend();
      }, 5000);
    }
  });

  log('[Python] Starting FastAPI backend on port 8000...');
  return proc;
}

// Wait for Python backend to be healthy before continuing
async function waitForPythonBackend(): Promise<boolean> {
  const startTime = Date.now();
  log('[Python] Waiting for FastAPI backend to be ready...');
  
  while (Date.now() - startTime < PYTHON_STARTUP_TIMEOUT_MS) {
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 2000);
      
      const response = await fetch(PYTHON_HEALTH_URL, {
        method: 'GET',
        signal: controller.signal,
      });
      
      clearTimeout(timeoutId);
      
      if (response.ok) {
        const elapsed = Math.round((Date.now() - startTime) / 1000);
        log(`[Python] FastAPI backend ready after ${elapsed}s`);
        return true;
      }
    } catch {
      // Connection refused or timeout - Python not ready yet
    }
    
    // Wait before next poll
    await new Promise(resolve => setTimeout(resolve, PYTHON_HEALTH_POLL_INTERVAL_MS));
  }
  
  log('[Python] WARNING: FastAPI backend did not become ready within timeout');
  return false;
}

// Initialize Python backend (start + wait for ready)
async function initializePythonBackend(): Promise<void> {
  if (process.env.NODE_ENV === 'development') {
    pythonProcess = startPythonBackend();
    await waitForPythonBackend();
  }
}

const app = express();
// Trust the first proxy so secure cookies are set correctly when running behind a load balancer
app.set("trust proxy", 1);
app.use(cookieParser());
app.use(express.json());
app.use(express.urlencoded({ extended: false }));

// Log Stytch configuration status
if (isStytchConfigured()) {
  log("[STYTCH] Authentication service configured and ready");
} else {
  log("[STYTCH] Warning: STYTCH_PROJECT_ID or STYTCH_SECRET not set - auth features disabled");
}

// HIPAA-compliant request logging middleware
// Uses safe_logger to redact PHI from response bodies before logging
app.use((req, res, next) => {
  const start = Date.now();
  const path = req.path;
  let capturedJsonResponse: Record<string, any> | undefined = undefined;

  const originalResJson = res.json;
  res.json = function (bodyJson, ...args) {
    capturedJsonResponse = bodyJson;
    return originalResJson.apply(res, [bodyJson, ...args]);
  };

  res.on("finish", () => {
    const duration = Date.now() - start;
    if (path.startsWith("/api")) {
      // HIPAA: Log request info without response body to prevent PHI leakage
      // Response bodies may contain patient data and should never be logged
      const logLine = `${req.method} ${path} ${res.statusCode} in ${duration}ms`;
      safeLogger.info(logLine);
    }
  });

  next();
});

(async () => {
  // Start Python backend in background (don't wait for it to be ready)
  // Express will start immediately; Python routes will work once FastAPI is ready
  if (process.env.NODE_ENV === 'development') {
    pythonProcess = startPythonBackend();
    // Check for Python readiness in background without blocking
    waitForPythonBackend().then(ready => {
      if (!ready) {
        log('[Python] Backend started but may still be loading ML models...');
      }
    });
  }
  
  // Auto-seed database with sample data for fully functional features
  try {
    await seedDatabase();
    log("Database seed completed");
  } catch (error) {
    log("Seed skipped or failed (non-critical)");
  }

  const server = await registerRoutes(app);

  app.use((err: any, _req: Request, res: Response, _next: NextFunction) => {
    const status = err.status || err.statusCode || 500;
    const message = err.message || "Internal Server Error";

    res.status(status).json({ message });
    throw err;
  });

  // importantly only setup vite in development and after
  // setting up all the other routes so the catch-all route
  // doesn't interfere with the other routes
  if (app.get("env") === "development") {
    await setupVite(app, server);
  } else {
    serveStatic(app);
  }

  // ALWAYS serve the app on the port specified in the environment variable PORT
  // Other ports are firewalled. Default to 5000 if not specified.
  // this serves both the API and the client.
  // It is the only port that is not firewalled.
  const port = parseInt(process.env.PORT || '5000', 10);
  server.listen({
    port,
    host: "0.0.0.0",
    reusePort: true,
  }, () => {
    log(`serving on port ${port}`);
  });
})();
