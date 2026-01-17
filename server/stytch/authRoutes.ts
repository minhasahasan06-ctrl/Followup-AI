import { Router, Request, Response } from "express";
import { getStytchClient, isStytchConfigured } from "./stytchClient";
import { getSessionCookieOptions, requireAuth } from "./authMiddleware";
import { storage } from "../storage";

const router = Router();

const SESSION_COOKIE_NAME = "stytch_session_token";
const SESSION_DURATION_MINUTES = 60 * 24;

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

    let dbUser = await storage.getUserByEmail(userEmail);
    if (!dbUser && userEmail) {
      dbUser = await storage.createUser({
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

    console.log(`[STYTCH] SMS OTP sent to ${phone}`);
    res.json({
      success: true,
      message: "Verification code sent via SMS",
      phoneId: response.phone_id,
      userCreated: response.user_created,
    });
  } catch (error: any) {
    console.error("[STYTCH] SMS OTP send error:", error.message);
    res.status(500).json({ error: "Failed to send verification code" });
  }
});

router.post("/sms-otp/authenticate", async (req: Request, res: Response) => {
  if (!isStytchConfigured()) {
    return res.status(503).json({ error: "Authentication service not configured" });
  }

  const { phone, code } = req.body;

  if (!phone || typeof phone !== "string") {
    return res.status(400).json({ error: "Phone number is required" });
  }

  if (!code || typeof code !== "string") {
    return res.status(400).json({ error: "Verification code is required" });
  }

  try {
    const client = getStytchClient();
    const response = await client.otps.sms.authenticate({
      phone_number: phone,
      code,
      session_duration_minutes: SESSION_DURATION_MINUTES,
    });

    const userPhone = response.user.phone_numbers?.[0]?.phone_number || phone;
    const userEmail = response.user.emails?.[0]?.email;
    const trustedMetadata = response.user.trusted_metadata as Record<string, any> || {};
    const role = trustedMetadata.role || "patient";

    let dbUser = userEmail ? await storage.getUserByEmail(userEmail) : null;
    if (!dbUser) {
      dbUser = await storage.createUser({
        email: userEmail || `${response.user.user_id}@phone.stytch.local`,
        firstName: response.user.name?.first_name || "",
        lastName: response.user.name?.last_name || "",
        role: role,
        phone: userPhone,
      });
    }

    const isProduction = process.env.NODE_ENV === "production";
    res.cookie(SESSION_COOKIE_NAME, response.session_token, getSessionCookieOptions(isProduction));

    console.log(`[STYTCH] SMS OTP authenticated for ${phone}`);
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
    console.error("[STYTCH] SMS OTP auth error:", error.message);
    res.status(401).json({ error: "Invalid or expired verification code" });
  }
});

router.post("/session/refresh", requireAuth, async (req: Request, res: Response) => {
  res.json({
    success: true,
    user: {
      id: req.stytchUser?.id,
      email: req.stytchUser?.email,
      role: req.stytchUser?.role,
    },
  });
});

router.get("/session", async (req: Request, res: Response) => {
  const sessionToken = req.cookies?.[SESSION_COOKIE_NAME];
  
  if (!sessionToken) {
    return res.status(401).json({ authenticated: false, error: "No session token" });
  }
  
  if (!isStytchConfigured()) {
    return res.status(503).json({ authenticated: false, error: "Authentication service not configured" });
  }
  
  try {
    const client = getStytchClient();
    const authResponse = await client.sessions.authenticate({
      session_token: sessionToken,
      session_duration_minutes: SESSION_DURATION_MINUTES,
    });
    
    const userEmail = authResponse.user.emails?.[0]?.email || "";
    const trustedMetadata = authResponse.user.trusted_metadata as Record<string, any> || {};
    const role = trustedMetadata.role || "patient";
    
    res.json({
      authenticated: true,
      user: {
        id: authResponse.session.user_id,
        email: userEmail,
        role: role,
      },
    });
  } catch (error: any) {
    res.clearCookie(SESSION_COOKIE_NAME);
    return res.status(401).json({ authenticated: false, error: "Invalid or expired session" });
  }
});

router.get("/session/me", requireAuth, async (req: Request, res: Response) => {
  const dbUser = await storage.getUserByEmail(req.stytchUser?.email || "");
  
  res.json({
    user: {
      id: req.stytchUser?.id,
      email: req.stytchUser?.email,
      phone: req.stytchUser?.phone,
      role: req.stytchUser?.role,
      firstName: dbUser?.firstName,
      lastName: dbUser?.lastName,
    },
  });
});

router.post("/logout", async (req: Request, res: Response) => {
  const sessionToken = req.cookies?.[SESSION_COOKIE_NAME];

  if (sessionToken && isStytchConfigured()) {
    try {
      const client = getStytchClient();
      await client.sessions.revoke({ session_token: sessionToken });
      console.log("[STYTCH] Session revoked");
    } catch (error: any) {
      console.warn("[STYTCH] Session revoke warning:", error.message);
    }
  }

  res.clearCookie(SESSION_COOKIE_NAME);
  res.json({ success: true, message: "Logged out successfully" });
});

router.post("/user/update-role", requireAuth, async (req: Request, res: Response) => {
  if (req.stytchUser?.role !== "admin") {
    return res.status(403).json({ error: "Admin access required" });
  }

  const { userId, role } = req.body;

  if (!userId || !role) {
    return res.status(400).json({ error: "userId and role are required" });
  }

  if (!["doctor", "patient", "admin"].includes(role)) {
    return res.status(400).json({ error: "Invalid role" });
  }

  try {
    const client = getStytchClient();
    await client.users.update({
      user_id: userId,
      trusted_metadata: { role },
    });

    console.log(`[STYTCH] Role updated for ${userId} to ${role}`);
    res.json({ success: true, message: "Role updated successfully" });
  } catch (error: any) {
    console.error("[STYTCH] Role update error:", error.message);
    res.status(500).json({ error: "Failed to update role" });
  }
});

export const stytchAuthRoutes = router;
