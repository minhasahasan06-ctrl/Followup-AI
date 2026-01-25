/**
 * Cloud Run Authenticated Proxy
 * 
 * Provides a secure proxy layer that:
 * 1. Validates Stytch session tokens (user authentication)
 * 2. Generates Google ID tokens for Cloud Run access (service authentication)
 * 3. Forwards requests to Cloud Run with proper headers
 * 
 * This bridges the gap between browser-based Stytch auth and 
 * Cloud Run's IAM authentication requirements.
 */
import { Router, Request, Response, NextFunction } from "express";
import { requireAuth, StytchUser } from "./stytch";
import { fetchFromCloudRun, getPythonBackendUrl } from "./cloudRunAuth";

const router = Router();

const CLOUD_RUN_URL = process.env.CLOUD_RUN_URL || getPythonBackendUrl();

interface ProxyRequestOptions {
  method: string;
  headers: Record<string, string>;
  body?: string;
}

function buildProxyHeaders(req: Request): Record<string, string> {
  const headers: Record<string, string> = {
    "Content-Type": req.get("Content-Type") || "application/json",
  };

  if (req.stytchUser) {
    headers["X-Stytch-User-Id"] = req.stytchUser.stytchUserId;
    headers["X-User-Email"] = req.stytchUser.email;
    headers["X-User-Role"] = req.stytchUser.role;
    if (req.stytchUser.id) {
      headers["X-User-Id"] = req.stytchUser.id;
    }
  }

  const forwardHeaders = [
    "accept",
    "accept-language",
    "cache-control",
    "x-request-id",
    "x-correlation-id",
  ];

  for (const header of forwardHeaders) {
    const value = req.get(header);
    if (value) {
      headers[header] = value;
    }
  }

  return headers;
}

async function proxyToCloudRun(
  req: Request,
  res: Response,
  targetPath: string
): Promise<void> {
  const queryString = req.url.includes("?") ? req.url.split("?")[1] : "";
  const fullUrl = `${CLOUD_RUN_URL}${targetPath}${queryString ? `?${queryString}` : ""}`;

  const options: ProxyRequestOptions = {
    method: req.method,
    headers: buildProxyHeaders(req),
  };

  if (["POST", "PUT", "PATCH"].includes(req.method) && req.body) {
    options.body = typeof req.body === "string" ? req.body : JSON.stringify(req.body);
  }

  try {
    const response = await fetchFromCloudRun(fullUrl, options);

    res.status(response.status);

    const headersToForward = ["content-type", "x-request-id", "x-correlation-id"];
    for (const header of headersToForward) {
      const value = response.headers.get(header);
      if (value) {
        res.setHeader(header, value);
      }
    }

    const contentType = response.headers.get("content-type") || "";
    if (contentType.includes("application/json")) {
      const data = await response.json();
      res.json(data);
    } else {
      const text = await response.text();
      res.send(text);
    }
  } catch (error) {
    console.error("[CloudRunProxy] Request failed:", error);
    res.status(502).json({
      error: "Backend service unavailable",
      message: error instanceof Error ? error.message : "Unknown error",
    });
  }
}

router.all("/*", requireAuth, async (req: Request, res: Response) => {
  const targetPath = req.path;
  await proxyToCloudRun(req, res, `/api${targetPath}`);
});

export { router as cloudRunProxyRouter };
export { proxyToCloudRun };
