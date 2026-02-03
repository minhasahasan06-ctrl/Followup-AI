/**
 * Stytch Authentication Routes for Cloud Run
 * 
 * Minimal version that uses authStorage instead of the full storage module.
 * This reduces dependencies and speeds up container startup.
 */
import { Router, Request, Response } from "express";
import { getStytchClient, isStytchConfigured } from "./stytchClient";
import { getSessionCookieOptions, requireAuth } from "./authMiddlewareCloudRun";
import { authStorage } from "./authStorage";

const router = Router();

const SESSION_COOKIE_NAME = "stytch_session_token";
const SESSION_DURATION_MINUTES = 60 * 24;

// Magic Link - Send
router.post("/magic-link/send", async (req: Request, res: Response) => {
  if (!isStytchConfigured()) {
    return res.status(503).json({ error: "Authentication service not configured" });
  }

  const { email, role = "patient" } = req.body;

  if (!email || typeof email !== "string") {
    return res.status(400).json({ error: "Email is required" });
  }

  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  if (!emailRegex.test(email)) {
    return res.status(400).json({ error: "Invalid email format" });
  }

  if (!["doctor", "patient", "admin"].includes(role)) {
    return res.status(400).json({ error: "Invalid role. Must be doctor, patient, or admin" });
  }

  try {
    const client = getStytchClient();
    const baseUrl = process.env.APP_URL || `${req.protocol}://${req.get("host")}`;

    const response = await client.magicLinks.email.loginOrCreate({
      email,
      login_magic_link_url: `${baseUrl}/auth/magic-link/callback`,
      signup_magic_link_url: `${baseUrl}/auth/magic-link/callback`,
      login_expiration_minutes: 30,
      signup_expiration_minutes: 30,
      create_user_as_pending: false,
    });

    if (response.user_created) {
      await client.users.update({
        user_id: response.user_id,
        trusted_metadata: { role },
      });
    }

    console.log(`[STYTCH] Magic link sent to ${email}`);
    res.json({
      success: true,
      message: "Magic link sent to your email",
      userCreated: response.user_created,
    });
  } catch (error: any) {
    console.error("[STYTCH] Magic link send error:", error.message);
    res.status(500).json({ error: "Failed to send magic link" });
  }
});

// Magic Link - Authenticate
router.post("/magic-link/authenticate", async (req: Request, res: Response) => {
  if (!isStytchConfigured()) {
    return res.status(503).json({ error: "Authentication service not configured" });
  }

  const { token } = req.body;

  if (!token || typeof token !== "string") {
    return res.status(400).json({ error: "Token is required" });
  }

  try {
    const client = getStytchClient();
    const response = await client.magicLinks.authenticate({
      token,
      session_duration_minutes: SESSION_DURATION_MINUTES,
    });

    const userEmail = response.user.emails?.[0]?.email || "";
    const trustedMetadata = response.user.trusted_metadata as Record<string, any> || {};
    const role = trustedMetadata.role || "patient";

    let dbUser = await authStorage.getUserByEmail(userEmail);
    if (!dbUser && userEmail) {
      dbUser = await authStorage.createUser({
        email: userEmail,
        firstName: response.user.name?.first_name || "",
        lastName: response.user.name?.last_name || "",
        role: role,
      });
    }

    const isProduction = process.env.NODE_ENV === "production";
    res.cookie(SESSION_COOKIE_NAME, response.session_token, getSessionCookieOptions(isProduction));

    console.log(`[STYTCH] Magic link authenticated for ${userEmail}`);
    res.json({
      success: true,
      user: {
        id: dbUser?.id || response.user.user_id,
        email: userEmail,
        role: role,
        firstName: response.user.name?.first_name,
        lastName: response.user.name?.last_name,
      },
    });
  } catch (error: any) {
    console.error("[STYTCH] Magic link auth error:", error.message);
    res.status(401).json({ error: "Invalid or expired magic link" });
  }
});

// SMS OTP - Send
router.post("/sms-otp/send", async (req: Request, res: Response) => {
  if (!isStytchConfigured()) {
    return res.status(503).json({ error: "Authentication service not configured" });
  }

  const { phone, phone_number, role = "patient" } = req.body;
  const phoneValue = phone || phone_number;

  if (!phoneValue || typeof phoneValue !== "string") {
    return res.status(400).json({ error: "Phone number is required" });
  }

  const phoneRegex = /^\+[1-9]\d{1,14}$/;
  if (!phoneRegex.test(phoneValue)) {
    return res.status(400).json({ error: "Invalid phone format. Use E.164 format (e.g., +14155551234)" });
  }

  if (!["doctor", "patient", "admin"].includes(role)) {
    return res.status(400).json({ error: "Invalid role. Must be doctor, patient, or admin" });
  }

  try {
    const client = getStytchClient();
    const response = await client.otps.sms.loginOrCreate({
      phone_number: phoneValue,
      expiration_minutes: 10,
      create_user_as_pending: false,
    });

    if (response.user_created) {
      await client.users.update({
        user_id: response.user_id,
        trusted_metadata: { role },
      });
    }

    console.log(`[STYTCH] SMS OTP sent to ${phoneValue}`);
    res.json({
      success: true,
      message: "Verification code sent to your phone",
      userCreated: response.user_created,
      phoneId: response.phone_id,
    });
  } catch (error: any) {
    console.error("[STYTCH] SMS OTP send error:", error.message);
    res.status(500).json({ error: "Failed to send verification code" });
  }
});

// SMS OTP - Verify
router.post("/sms-otp/verify", async (req: Request, res: Response) => {
  if (!isStytchConfigured()) {
    return res.status(503).json({ error: "Authentication service not configured" });
  }

  const { phone, phone_number, code } = req.body;
  const phoneValue = phone || phone_number;

  if (!phoneValue || typeof phoneValue !== "string") {
    return res.status(400).json({ error: "Phone number is required" });
  }

  if (!code || typeof code !== "string") {
    return res.status(400).json({ error: "Verification code is required" });
  }

  try {
    const client = getStytchClient();
    const response = await client.otps.sms.authenticate({
      phone_number: phoneValue,
      code,
      session_duration_minutes: SESSION_DURATION_MINUTES,
    });

    const userPhone = response.user.phone_numbers?.[0]?.phone_number || phoneValue;
    const userEmail = response.user.emails?.[0]?.email || "";
    const trustedMetadata = response.user.trusted_metadata as Record<string, any> || {};
    const role = trustedMetadata.role || "patient";

    let dbUser = userEmail ? await authStorage.getUserByEmail(userEmail) : null;
    if (!dbUser) {
      dbUser = await authStorage.createUser({
        email: userEmail || `${userPhone.replace(/\+/g, "")}@phone.user`,
        firstName: response.user.name?.first_name || "",
        lastName: response.user.name?.last_name || "",
        role: role,
        phone: userPhone,
      });
    }

    const isProduction = process.env.NODE_ENV === "production";
    res.cookie(SESSION_COOKIE_NAME, response.session_token, getSessionCookieOptions(isProduction));

    console.log(`[STYTCH] SMS OTP verified for ${userPhone}`);
    res.json({
      success: true,
      user: {
        id: dbUser?.id || response.user.user_id,
        email: userEmail,
        phone: userPhone,
        role: role,
        firstName: response.user.name?.first_name,
        lastName: response.user.name?.last_name,
      },
    });
  } catch (error: any) {
    console.error("[STYTCH] SMS OTP verify error:", error.message);
    res.status(401).json({ error: "Invalid or expired verification code" });
  }
});

// Session - Get current user
router.get("/session", requireAuth, async (req: Request, res: Response) => {
  const dbUser = await authStorage.getUserByEmail(req.stytchUser?.email || "");

  res.json({
    user: {
      id: dbUser?.id || req.stytchUser?.user_id,
      email: req.stytchUser?.email,
      role: req.stytchUser?.role || "patient",
      firstName: dbUser?.firstName || req.stytchUser?.name?.first_name,
      lastName: dbUser?.lastName || req.stytchUser?.name?.last_name,
    },
  });
});

// Session - Logout
router.post("/logout", async (req: Request, res: Response) => {
  const sessionToken = req.cookies[SESSION_COOKIE_NAME];

  if (sessionToken && isStytchConfigured()) {
    try {
      const client = getStytchClient();
      await client.sessions.revoke({ session_token: sessionToken });
    } catch (error: any) {
      console.warn("[STYTCH] Session revoke warning:", error.message);
    }
  }

  res.clearCookie(SESSION_COOKIE_NAME, {
    httpOnly: true,
    secure: process.env.NODE_ENV === "production",
    sameSite: process.env.CORS_ORIGINS ? "none" : "lax",
  });

  res.json({ success: true, message: "Logged out successfully" });
});

// Status check
router.get("/status", (_req: Request, res: Response) => {
  res.json({
    stytch: isStytchConfigured() ? "configured" : "not configured",
    service: "followupai-express-cloudrun",
  });
});

export const stytchAuthRoutesCloudRun = router;
