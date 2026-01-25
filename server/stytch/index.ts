export {
  stytchManager,
  getStytchClient,
  isStytchConfigured,
  getM2MAccessToken,
  validateM2MToken,
} from "./stytchClient";

export {
  requireAuth,
  requireRole,
  requireDoctor,
  requirePatient,
  requireAdmin,
  requireM2MAuth,
  optionalAuth,
  getSessionCookieOptions,
  type StytchUser,
} from "./authMiddleware";

export { stytchAuthRoutes } from "./authRoutes";

export {
  sendSMS,
  sendEmail,
  sendVerificationSMS,
  sendVerificationEmail,
  sendPasswordResetEmail,
  sendWelcomeEmail,
  sendWelcomeSMS,
  sendMedicationReminder,
  sendAppointmentConfirmation,
  sendAppointmentReminder,
  sendAppointmentConfirmationEmail,
  sendEmergencyAlert,
  sendLabResultsNotificationEmail,
} from "./messagingService";

export {
  M2M_SCOPES,
  getInternalServiceToken,
  validateServiceToken,
  callFastAPIWithM2M,
  createM2MAuthHeaders,
  getClientToken,
  REGISTERED_M2M_CLIENTS,
  type M2MScope,
  type M2MClientConfig,
} from "./m2mAuth";
