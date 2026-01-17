import * as stytch from "stytch";

const STYTCH_PROJECT_ID = process.env.STYTCH_PROJECT_ID;
const STYTCH_SECRET = process.env.STYTCH_SECRET;

if (!STYTCH_PROJECT_ID || !STYTCH_SECRET) {
  console.warn("[STYTCH] Warning: STYTCH_PROJECT_ID or STYTCH_SECRET not configured. Auth features will be disabled.");
}

class StytchClientManager {
  private static instance: StytchClientManager;
  private consumerClient: stytch.Client | null = null;
  private m2mTokenCache: Map<string, { token: string; expiresAt: number }> = new Map();

  private constructor() {
    if (STYTCH_PROJECT_ID && STYTCH_SECRET) {
      this.consumerClient = new stytch.Client({
        project_id: STYTCH_PROJECT_ID,
        secret: STYTCH_SECRET,
      });
      console.log("[STYTCH] Client initialized successfully");
    }
  }

  public static getInstance(): StytchClientManager {
    if (!StytchClientManager.instance) {
      StytchClientManager.instance = new StytchClientManager();
    }
    return StytchClientManager.instance;
  }

  public getClient(): stytch.Client {
    if (!this.consumerClient) {
      throw new Error("Stytch client not initialized. Check STYTCH_PROJECT_ID and STYTCH_SECRET.");
    }
    return this.consumerClient;
  }

  public isConfigured(): boolean {
    return this.consumerClient !== null;
  }

  public async getM2MToken(
    clientId: string,
    clientSecret: string,
    scopes: string[] = []
  ): Promise<string> {
    const cacheKey = `${clientId}:${scopes.sort().join(",")}`;
    const cached = this.m2mTokenCache.get(cacheKey);

    if (cached && cached.expiresAt > Date.now() + 60000) {
      return cached.token;
    }

    const client = this.getClient();
    const response = await client.m2m.token({
      client_id: clientId,
      client_secret: clientSecret,
      scopes,
    });

    const expiresAt = Date.now() + (response.expires_in * 1000);
    this.m2mTokenCache.set(cacheKey, {
      token: response.access_token,
      expiresAt,
    });

    return response.access_token;
  }

  public async authenticateM2MToken(
    accessToken: string,
    requiredScopes: string[] = []
  ): Promise<{
    clientId: string;
    scopes: string[];
    customClaims?: Record<string, any>;
  }> {
    const client = this.getClient();
    const response = await client.m2m.authenticateToken({
      access_token: accessToken,
      required_scopes: requiredScopes,
    });

    return {
      clientId: response.client_id,
      scopes: response.scopes,
      customClaims: response.custom_claims,
    };
  }

  public clearM2MTokenCache(): void {
    this.m2mTokenCache.clear();
  }
}

export const stytchManager = StytchClientManager.getInstance();

export function getStytchClient(): stytch.Client {
  return stytchManager.getClient();
}

export function isStytchConfigured(): boolean {
  return stytchManager.isConfigured();
}

export async function getM2MAccessToken(
  clientId: string,
  clientSecret: string,
  scopes: string[] = []
): Promise<string> {
  return stytchManager.getM2MToken(clientId, clientSecret, scopes);
}

export async function validateM2MToken(
  accessToken: string,
  requiredScopes: string[] = []
): Promise<{
  clientId: string;
  scopes: string[];
  customClaims?: Record<string, any>;
}> {
  return stytchManager.authenticateM2MToken(accessToken, requiredScopes);
}
