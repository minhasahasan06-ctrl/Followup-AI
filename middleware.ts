/**
 * Vercel Edge Middleware for HTTP Basic Auth Protection
 * 
 * This middleware uses the standard Web Request/Response API (not Next.js APIs)
 * to be compatible with Vercel static SPA deployments (Vite, React, etc.)
 * 
 * Set BASIC_AUTH_USER and BASIC_AUTH_PASSWORD in Vercel environment variables
 * to enable password protection. Leave empty to disable.
 */

const REALM = "Followup AI";

// Paths to exclude from basic auth
const EXCLUDED_PATHS = [
  '/api/',
  '/_next/',
  '/assets/',
  '/favicon.ico',
  '/robots.txt',
  '/sitemap.xml',
  '/manifest.webmanifest',
  '/vite.svg',
];

// File extensions to exclude
const EXCLUDED_EXTENSIONS = [
  '.css', '.js', '.map', '.png', '.jpg', '.jpeg', '.gif', '.svg', 
  '.webp', '.ico', '.woff', '.woff2', '.ttf', '.eot', '.json'
];

/**
 * Constant-time string comparison to prevent timing attacks
 */
function constantTimeCompare(a: string, b: string): boolean {
  if (a.length !== b.length) {
    return false;
  }
  
  let result = 0;
  for (let i = 0; i < a.length; i++) {
    result |= a.charCodeAt(i) ^ b.charCodeAt(i);
  }
  return result === 0;
}

/**
 * Parse Basic Auth header and extract credentials
 */
function parseBasicAuth(
  authHeader: string | null
): { username: string; password: string } | null {
  if (!authHeader || !authHeader.startsWith("Basic ")) {
    return null;
  }

  try {
    const base64Credentials = authHeader.slice(6);
    const credentials = atob(base64Credentials);
    const separatorIndex = credentials.indexOf(":");
    
    if (separatorIndex === -1) {
      return null;
    }

    return {
      username: credentials.slice(0, separatorIndex),
      password: credentials.slice(separatorIndex + 1),
    };
  } catch {
    return null;
  }
}

/**
 * Check if the request path should be excluded from basic auth
 */
function shouldExcludePath(pathname: string): boolean {
  // Check excluded path prefixes
  for (const path of EXCLUDED_PATHS) {
    if (pathname.startsWith(path)) {
      return true;
    }
  }
  
  // Check excluded file extensions
  for (const ext of EXCLUDED_EXTENSIONS) {
    if (pathname.endsWith(ext)) {
      return true;
    }
  }
  
  return false;
}

/**
 * Create 401 Unauthorized response with WWW-Authenticate header
 */
function createUnauthorizedResponse(): Response {
  return new Response("Authentication required", {
    status: 401,
    headers: {
      "WWW-Authenticate": `Basic realm="${REALM}", charset="UTF-8"`,
      "Content-Type": "text/plain",
    },
  });
}

/**
 * Vercel Edge Middleware handler
 * Uses standard Web Request/Response API for static site compatibility
 */
export default function middleware(request: Request): Response | undefined {
  const url = new URL(request.url);
  const pathname = url.pathname;
  
  // Skip excluded paths (static assets, API routes, etc.)
  if (shouldExcludePath(pathname)) {
    return undefined; // Continue to origin
  }

  // Get credentials from environment
  const expectedUsername = process.env.BASIC_AUTH_USER;
  const expectedPassword = process.env.BASIC_AUTH_PASSWORD;

  // If no credentials configured, skip basic auth
  if (!expectedUsername || !expectedPassword) {
    return undefined; // Continue to origin
  }

  // Parse authorization header
  const authHeader = request.headers.get("authorization");
  const credentials = parseBasicAuth(authHeader);

  // No credentials provided
  if (!credentials) {
    return createUnauthorizedResponse();
  }

  // Validate credentials with constant-time comparison
  const isValidUsername = constantTimeCompare(
    credentials.username,
    expectedUsername
  );
  const isValidPassword = constantTimeCompare(
    credentials.password,
    expectedPassword
  );

  if (!isValidUsername || !isValidPassword) {
    return createUnauthorizedResponse();
  }

  // Credentials valid, continue to origin
  return undefined;
}

/**
 * Vercel Edge Config
 * Specifies which routes this middleware should run on
 */
export const config = {
  matcher: [
    /*
     * Match all request paths except:
     * - api (API routes)
     * - _next/static (static files)
     * - _next/image (image optimization)
     * - assets (static assets)
     * - favicon.ico, robots.txt, sitemap.xml
     * - Static file extensions (css, js, images, fonts, etc.)
     */
    "/((?!api|_next/static|_next/image|assets|favicon\\.ico|robots\\.txt|sitemap\\.xml|manifest\\.webmanifest|vite\\.svg|.*\\.(?:css|js|map|png|jpg|jpeg|gif|svg|webp|ico|woff|woff2|ttf|eot|json)).*)",
  ],
};
