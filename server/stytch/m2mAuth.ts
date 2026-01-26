import { getM2MAccessToken, validateM2MToken } from "./stytchClient";

const M2M_CLIENT_ID = process.env.STYTCH_M2M_CLIENT_ID;
const M2M_CLIENT_SECRET = process.env.STYTCH_M2M_CLIENT_SECRET;

export const M2M_SCOPES = {
  READ_USERS: "read:users",
  WRITE_USERS: "write:users",
  READ_HEALTH: "read:health",
  WRITE_HEALTH: "write:health",
  READ_ML: "read:ml",
  WRITE_ML: "write:ml",
  ADMIN: "admin:all",
} as const;

export type M2MScope = (typeof M2M_SCOPES)[keyof typeof M2M_SCOPES];

export async function getInternalServiceToken(scopes: M2MScope[] = []): Promise<string | null> {
  if (!M2M_CLIENT_ID || !M2M_CLIENT_SECRET) {
    console.warn("[M2M] M2M credentials not configured (STYTCH_M2M_CLIENT_ID, STYTCH_M2M_CLIENT_SECRET)");
    return null;
  }

  try {
    const token = await getM2MAccessToken(M2M_CLIENT_ID, M2M_CLIENT_SECRET, scopes);
    return token;
  } catch (error: any) {
    console.error("[M2M] Failed to get internal service token:", error.message);
    return null;
  }
}

export async function validateServiceToken(
  token: string,
  requiredScopes: M2MScope[] = []
): Promise<{
  valid: boolean;
  clientId?: string;
  scopes?: string[];
  error?: string;
}> {
  try {
    const result = await validateM2MToken(token, requiredScopes);
    return {
      valid: true,
      clientId: result.clientId,
      scopes: result.scopes,
    };
  } catch (error: any) {
    return {
      valid: false,
      error: error.error_type || error.message,
    };
  }
}

export async function callFastAPIWithM2M(
  endpoint: string,
  options: {
    method?: string;
    body?: any;
    scopes?: M2MScope[];
    headers?: Record<string, string>;
  } = {}
): Promise<Response> {
  const { method = "GET", body, scopes = [M2M_SCOPES.READ_HEALTH], headers = {} } = options;

  const token = await getInternalServiceToken(scopes);
  
  const fastApiUrl = process.env.FASTAPI_URL || "http://localhost:8000";
  const url = `${fastApiUrl}${endpoint}`;

  const requestHeaders: Record<string, string> = {
    ...headers,
    "Content-Type": "application/json",
  };

  if (token) {
    requestHeaders["Authorization"] = `Bearer ${token}`;
  } else {
    const devSecret = process.env.DEV_MODE_SECRET;
    if (devSecret) {
      requestHeaders["Authorization"] = `Bearer ${devSecret}`;
    }
  }

  const response = await fetch(url, {
    method,
    headers: requestHeaders,
    body: body ? JSON.stringify(body) : undefined,
  });

  return response;
}

export function createM2MAuthHeaders(token: string): Record<string, string> {
  return {
    Authorization: `Bearer ${token}`,
    "Content-Type": "application/json",
  };
}

export interface M2MClientConfig {
  clientId: string;
  clientSecret: string;
  allowedScopes: M2MScope[];
  description: string;
}

export const REGISTERED_M2M_CLIENTS: Record<string, M2MClientConfig> = {
  fastapi_backend: {
    clientId: process.env.STYTCH_M2M_CLIENT_ID || "",
    clientSecret: process.env.STYTCH_M2M_CLIENT_SECRET || "",
    allowedScopes: [
      M2M_SCOPES.READ_USERS,
      M2M_SCOPES.READ_HEALTH,
      M2M_SCOPES.WRITE_HEALTH,
      M2M_SCOPES.READ_ML,
      M2M_SCOPES.WRITE_ML,
    ],
    description: "FastAPI backend service for ML and health analysis",
  },
};

export async function getClientToken(
  clientName: keyof typeof REGISTERED_M2M_CLIENTS,
  requestedScopes?: M2MScope[]
): Promise<string | null> {
  const client = REGISTERED_M2M_CLIENTS[clientName];
  
  if (!client || !client.clientId || !client.clientSecret) {
    console.warn(`[M2M] Client '${clientName}' not configured`);
    return null;
  }

  const scopes = requestedScopes || client.allowedScopes;
  
  const validScopes = scopes.filter((s) => client.allowedScopes.includes(s));
  if (validScopes.length !== scopes.length) {
    console.warn(`[M2M] Some requested scopes not allowed for client '${clientName}'`);
  }

  try {
    return await getM2MAccessToken(client.clientId, client.clientSecret, validScopes);
  } catch (error: any) {
    console.error(`[M2M] Failed to get token for client '${clientName}':`, error.message);
    return null;
  }
}
