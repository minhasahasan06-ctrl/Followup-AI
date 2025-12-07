import { sql } from "drizzle-orm";
import {
  pgTable,
  text,
  varchar,
  timestamp,
  jsonb,
  integer,
  boolean,
  index,
  decimal,
} from "drizzle-orm/pg-core";
import { createInsertSchema } from "drizzle-zod";
import { z } from "zod";

// Users table with role-based access (AWS Cognito)
export const users = pgTable("users", {
  // AWS Cognito fields - ID is Cognito sub (UUID format)
  id: varchar("id").primaryKey(),
  email: varchar("email").unique().notNull(),
  firstName: varchar("first_name").notNull(),
  lastName: varchar("last_name").notNull(),
  profileImageUrl: varchar("profile_image_url"),
  emailVerified: boolean("email_verified").default(false),
  verificationToken: varchar("verification_token"),
  verificationTokenExpires: timestamp("verification_token_expires"),
  resetToken: varchar("reset_token"),
  resetTokenExpires: timestamp("reset_token_expires"),
  
  // Application-specific fields
  role: varchar("role", { length: 10 }).notNull().default("patient"), // 'patient' or 'doctor'
  
  // Doctor-specific fields
  organization: varchar("organization"), // Hospital/clinic name for doctors
  medicalLicenseNumber: varchar("medical_license_number"),
  licenseCountry: varchar("license_country"), // Country where license was issued
  licenseVerified: boolean("license_verified").default(false),
  kycPhotoUrl: varchar("kyc_photo_url"), // KYC document/photo
  
  // Patient-specific fields
  ehrPlatform: varchar("ehr_platform"), // Patient's chosen EHR platform
  ehrImportMethod: varchar("ehr_import_method"), // 'manual', 'hospital', 'platform'
  
  // Phone number for SMS and verification
  phoneNumber: varchar("phone_number"),
  phoneVerified: boolean("phone_verified").default(false),
  phoneVerificationCode: varchar("phone_verification_code"),
  phoneVerificationExpires: timestamp("phone_verification_expires"),
  
  // Doctor application verification
  googleDriveApplicationUrl: varchar("google_drive_application_url"),
  adminVerified: boolean("admin_verified").default(false),
  adminVerifiedAt: timestamp("admin_verified_at"),
  adminVerifiedBy: varchar("admin_verified_by"),
  
  // SMS preferences
  smsNotificationsEnabled: boolean("sms_notifications_enabled").default(true),
  smsMedicationReminders: boolean("sms_medication_reminders").default(true),
  smsAppointmentReminders: boolean("sms_appointment_reminders").default(true),
  smsDailyFollowups: boolean("sms_daily_followups").default(true),
  smsHealthAlerts: boolean("sms_health_alerts").default(true),
  
  // Terms and conditions
  termsAccepted: boolean("terms_accepted").default(false),
  termsAcceptedAt: timestamp("terms_accepted_at"),
  
  // Stripe integration for subscriptions
  stripeCustomerId: varchar("stripe_customer_id").unique(),
  stripeSubscriptionId: varchar("stripe_subscription_id"),
  subscriptionStatus: varchar("subscription_status"), // 'active', 'past_due', 'canceled', 'trialing'
  trialEndsAt: timestamp("trial_ends_at"),
  
  // Credit balance (for consultations)
  creditBalance: integer("credit_balance").default(0),
  
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
});

export const insertUserSchema = createInsertSchema(users).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export type InsertUser = z.infer<typeof insertUserSchema>;
export type User = typeof users.$inferSelect;
export type UpsertUser = typeof users.$inferInsert;

// Patient profiles
export const patientProfiles = pgTable("patient_profiles", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  userId: varchar("user_id").notNull().references(() => users.id),
  
  // Unique Followup AI Patient ID (e.g., FAI-ABC123) - generated on profile creation
  followupPatientId: varchar("followup_patient_id").unique(),
  
  dateOfBirth: timestamp("date_of_birth"),
  address: text("address"),
  city: varchar("city"),
  state: varchar("state"),
  zipCode: varchar("zip_code"),
  country: varchar("country").default("USA"),
  immunocompromisedCondition: text("immunocompromised_condition"),
  comorbidities: jsonb("comorbidities").$type<string[]>(),
  allergies: jsonb("allergies").$type<string[]>(),
  currentMedications: jsonb("current_medications").$type<string[]>(),
  ehrSyncEnabled: boolean("ehr_sync_enabled").default(false),
  ehrLastSync: timestamp("ehr_last_sync"),
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
});

export const insertPatientProfileSchema = createInsertSchema(patientProfiles).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export type InsertPatientProfile = z.infer<typeof insertPatientProfileSchema>;
export type PatientProfile = typeof patientProfiles.$inferSelect;

// Doctor profiles
export const doctorProfiles = pgTable("doctor_profiles", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  userId: varchar("user_id").notNull().references(() => users.id),
  linkedinProfileUrl: varchar("linkedin_profile_url"),
  specialties: jsonb("specialties").$type<string[]>(),
  education: jsonb("education").$type<Array<{ institution: string; degree: string; year: number }>>(),
  certifications: jsonb("certifications").$type<string[]>(),
  publications: jsonb("publications").$type<string[]>(),
  researchInterests: jsonb("research_interests").$type<string[]>(),
  bio: text("bio"),
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
});

export const insertDoctorProfileSchema = createInsertSchema(doctorProfiles).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export type InsertDoctorProfile = z.infer<typeof insertDoctorProfileSchema>;
export type DoctorProfile = typeof doctorProfiles.$inferSelect;

// Daily follow-ups
export const dailyFollowups = pgTable("daily_followups", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  patientId: varchar("patient_id").notNull().references(() => users.id),
  date: timestamp("date").notNull().defaultNow(),
  heartRate: integer("heart_rate"),
  bloodPressureSystolic: integer("blood_pressure_systolic"),
  bloodPressureDiastolic: integer("blood_pressure_diastolic"),
  bloodGlucose: decimal("blood_glucose", { precision: 5, scale: 1 }),
  oxygenSaturation: integer("oxygen_saturation"),
  temperature: decimal("temperature", { precision: 4, scale: 1 }),
  respiratoryRate: integer("respiratory_rate"),
  respiratorySoundAnalysis: jsonb("respiratory_sound_analysis").$type<{ quality: string; abnormalities: string[] }>(),
  anemiaDetected: boolean("anemia_detected"),
  jaundiceDetected: boolean("jaundice_detected"),
  edemaDetected: boolean("edema_detected"),
  skinConditions: jsonb("skin_conditions").$type<Array<{ condition: string; confidence: number; location: string }>>(),
  bowelMovements: integer("bowel_movements"),
  bowelConsistency: varchar("bowel_consistency"),
  urineFrequency: integer("urine_frequency"),
  urineColor: varchar("urine_color"),
  waterIntake: decimal("water_intake", { precision: 4, scale: 1 }),
  stepsCount: integer("steps_count"),
  sleepHours: decimal("sleep_hours", { precision: 3, scale: 1 }),
  completed: boolean("completed").default(false),
  deviceDataSource: varchar("device_data_source"),
  cameraAssessmentCompleted: boolean("camera_assessment_completed").default(false),
  createdAt: timestamp("created_at").defaultNow(),
});

export const insertDailyFollowupSchema = createInsertSchema(dailyFollowups).omit({
  id: true,
  createdAt: true,
});

export type InsertDailyFollowup = z.infer<typeof insertDailyFollowupSchema>;
export type DailyFollowup = typeof dailyFollowups.$inferSelect;

// Voice-based daily followups - Quick 1min voice logs with AI extraction
export const voiceFollowups = pgTable("voice_followups", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  patientId: varchar("patient_id").notNull().references(() => users.id),
  audioFileUrl: varchar("audio_file_url").notNull(), // S3 URL
  audioFileName: varchar("audio_file_name").notNull(),
  audioFileSize: integer("audio_file_size"), // bytes
  audioDuration: integer("audio_duration"), // seconds
  transcription: text("transcription").notNull(),
  
  // AI-extracted health data
  extractedSymptoms: jsonb("extracted_symptoms").$type<Array<{ symptom: string; severity: string; confidence: number }>>(),
  extractedMood: varchar("extracted_mood"), // 'positive', 'neutral', 'anxious', 'stressed', 'depressed'
  moodScore: decimal("mood_score", { precision: 3, scale: 2 }), // -1.0 to 1.0
  medicationAdherence: jsonb("medication_adherence").$type<Array<{ medication: string; taken: boolean; time?: string }>>(),
  extractedMetrics: jsonb("extracted_metrics").$type<{
    heartRate?: number;
    bloodPressure?: string;
    temperature?: number;
    sleepHours?: number;
    stepsCount?: number;
    waterIntake?: number;
  }>(),
  
  // AI analysis
  sentimentScore: decimal("sentiment_score", { precision: 3, scale: 2 }), // -1.0 to 1.0
  empathyLevel: varchar("empathy_level").notNull(), // 'supportive', 'empathetic', 'encouraging', 'concerned'
  concernsRaised: boolean("concerns_raised").default(false),
  concernsSummary: text("concerns_summary"),
  aiResponse: text("ai_response").notNull(), // Empathetic AI-generated response
  conversationSummary: text("conversation_summary"),
  
  // Follow-up recommendations
  needsFollowup: boolean("needs_followup").default(false),
  followupReason: text("followup_reason"),
  recommendedActions: jsonb("recommended_actions").$type<string[]>(),
  
  // Link to structured daily followup
  dailyFollowupId: varchar("daily_followup_id").references(() => dailyFollowups.id),
  
  createdAt: timestamp("created_at").defaultNow(),
});

export const insertVoiceFollowupSchema = createInsertSchema(voiceFollowups).omit({
  id: true,
  createdAt: true,
});

export type InsertVoiceFollowup = z.infer<typeof insertVoiceFollowupSchema>;
export type VoiceFollowup = typeof voiceFollowups.$inferSelect;

// Chat sessions - groups related messages together as medical history
export const chatSessions = pgTable("chat_sessions", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  patientId: varchar("patient_id").notNull().references(() => users.id),
  agentType: varchar("agent_type", { length: 20 }).notNull(), // 'clona' or 'lysa'
  contextPatientId: varchar("context_patient_id").references(() => users.id), // For Lysa: patient being discussed
  sessionTitle: varchar("session_title"), // Auto-generated or user-set title
  startedAt: timestamp("started_at").notNull().defaultNow(),
  endedAt: timestamp("ended_at"),
  messageCount: integer("message_count").default(0),
  
  // Medical History Metadata
  symptomsDiscussed: text("symptoms_discussed").array(), // List of symptoms mentioned
  healthInsights: jsonb("health_insights").$type<{
    keySymptoms?: string[];
    recommendations?: string[];
    concerns?: string[];
    vitalSigns?: { type: string; value: string; status: string }[];
    medications?: string[];
  }>(),
  aiSummary: text("ai_summary"), // AI-generated summary of the session
  urgencyLevel: varchar("urgency_level"), // 'low', 'moderate', 'high', 'critical'
  alertsGenerated: boolean("alerts_generated").default(false),
  
  // Doctor Review
  reviewedByDoctor: boolean("reviewed_by_doctor").default(false),
  reviewedBy: varchar("reviewed_by").references(() => users.id),
  reviewedAt: timestamp("reviewed_at"),
  doctorNotes: text("doctor_notes"),
  
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
});

export const insertChatSessionSchema = createInsertSchema(chatSessions).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export type InsertChatSession = z.infer<typeof insertChatSessionSchema>;
export type ChatSession = typeof chatSessions.$inferSelect;

// Chat messages - now linked to sessions
export const chatMessages = pgTable("chat_messages", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  sessionId: varchar("session_id").notNull().references(() => chatSessions.id),
  userId: varchar("user_id").notNull().references(() => users.id),
  role: varchar("role", { length: 10 }).notNull(), // 'user' or 'assistant'
  content: text("content").notNull(),
  agentType: varchar("agent_type", { length: 20 }), // 'clona' or 'lysa'
  medicalEntities: jsonb("medical_entities").$type<Array<{ text: string; type: string }>>(),
  patientContextId: varchar("patient_context_id"), // For doctor reviewing patient
  createdAt: timestamp("created_at").defaultNow(),
});

export const insertChatMessageSchema = createInsertSchema(chatMessages).omit({
  id: true,
  createdAt: true,
});

export type InsertChatMessage = z.infer<typeof insertChatMessageSchema>;
export type ChatMessage = typeof chatMessages.$inferSelect;

// Medications - Extended for complete lifecycle management with chronic care support
export const medications = pgTable("medications", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  patientId: varchar("patient_id").notNull().references(() => users.id),
  name: varchar("name").notNull(),
  dosage: varchar("dosage").notNull(),
  frequency: varchar("frequency").notNull(),
  isOTC: boolean("is_otc").default(false),
  aiSuggestion: text("ai_suggestion"),
  startDate: timestamp("start_date").defaultNow(),
  endDate: timestamp("end_date"),
  active: boolean("active").default(true),
  // Foreign key to drugs table for standardized drug mapping
  drugId: varchar("drug_id").references(() => drugs.id),
  // RxNorm concept unique identifier (RxCUI) for direct API mapping
  rxcui: varchar("rxcui"),
  
  // Medication source tracking
  source: varchar("source").notNull().default("manual"), // 'manual', 'document', 'prescription'
  sourceDocumentId: varchar("source_document_id"), // Link to medical document if from upload
  sourcePrescriptionId: varchar("source_prescription_id"), // Link to prescription if from doctor
  
  // Status and confirmation tracking - Extended for chronic care
  status: varchar("status").notNull().default("active"), // 'draft', 'scheduled', 'active', 'completed', 'expired', 'superseded', 'conflict_hold', 'discontinued'
  confirmedAt: timestamp("confirmed_at"),
  confirmedBy: varchar("confirmed_by"), // User ID who confirmed
  addedBy: varchar("added_by").notNull().default("patient"), // 'patient', 'doctor', 'system'
  autoDetected: boolean("auto_detected").default(false), // True if extracted from document
  
  // Chronic care - Duration and continuous medication tracking
  isContinuous: boolean("is_continuous").default(false), // True for ongoing/indefinite medications
  intendedStartDate: timestamp("intended_start_date"), // When patient plans to start
  actualStartDate: timestamp("actual_start_date"), // When patient confirmed they started
  durationDays: integer("duration_days"), // Null for continuous medications
  computedEndDate: timestamp("computed_end_date"), // Auto-calculated: actualStartDate + durationDays
  
  // Multi-specialty tracking
  specialty: varchar("specialty"), // 'cardiology', 'oncology', 'neurology', etc.
  prescribingDoctorId: varchar("prescribing_doctor_id").references(() => users.id),
  
  // Supersession tracking - same specialty replacement
  supersededBy: varchar("superseded_by"), // ID of the new medication that replaced this one
  supersededAt: timestamp("superseded_at"),
  supersessionReason: text("supersession_reason"),
  
  // Cross-specialty conflict tracking
  conflictGroupId: varchar("conflict_group_id"), // Links medications in same conflict
  conflictStatus: varchar("conflict_status"), // 'pending', 'resolved', 'escalated'
  conflictDetectedAt: timestamp("conflict_detected_at"),
  conflictResolvedAt: timestamp("conflict_resolved_at"),
  conflictResolution: text("conflict_resolution"), // JSON with resolution details
  
  // Reminder settings
  remindersEnabled: boolean("reminders_enabled").default(true),
  reminderOffsets: jsonb("reminder_offsets").$type<number[]>(), // Hours before end to remind, e.g., [24, 48, 168]
  lastReminderSentAt: timestamp("last_reminder_sent_at"),
  
  // Adherence tracking
  adherenceLog: jsonb("adherence_log").$type<Array<{ date: string; taken: boolean; time?: string; notes?: string }>>(),
  
  // Discontinuation tracking
  discontinuedAt: timestamp("discontinued_at"),
  discontinuedBy: varchar("discontinued_by"), // User ID who discontinued
  discontinuationReason: text("discontinuation_reason"),
  replacementMedicationId: varchar("replacement_medication_id"), // If switched to different medication
  
  // Archive tracking - links to medical document record when archived
  archivedAt: timestamp("archived_at"),
  archiveDocumentId: varchar("archive_document_id"), // Link to medical_documents record when archived
  archiveReason: varchar("archive_reason"), // 'completed', 'superseded', 'expired', 'discontinued', 'conflict_resolved'
  
  // Lifecycle metadata for patient records integration
  lifecycleStage: varchar("lifecycle_stage").default("active"), // 'pending', 'active', 'ending_soon', 'completed', 'archived'
  lastLifecycleUpdate: timestamp("last_lifecycle_update"),
  
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
});

export const insertMedicationSchema = createInsertSchema(medications).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export type InsertMedication = z.infer<typeof insertMedicationSchema>;
export type Medication = typeof medications.$inferSelect;

// Prescriptions - Doctor-created prescriptions for patients with chronic care support
export const prescriptions = pgTable("prescriptions", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  patientId: varchar("patient_id").notNull().references(() => users.id),
  doctorId: varchar("doctor_id").notNull().references(() => users.id),
  
  // Medication details
  drugId: varchar("drug_id").references(() => drugs.id),
  rxcui: varchar("rxcui"), // RxNorm code
  medicationName: varchar("medication_name").notNull(),
  dosage: varchar("dosage").notNull(),
  frequency: varchar("frequency").notNull(),
  dosageInstructions: text("dosage_instructions"), // "Take with food", "Avoid alcohol", etc.
  
  // Prescription details
  quantity: integer("quantity"), // Number of pills/doses
  refills: integer("refills").default(0),
  startDate: timestamp("start_date").defaultNow(),
  expirationDate: timestamp("expiration_date"), // When prescription expires
  
  // Chronic care - Duration and continuous tracking
  isContinuous: boolean("is_continuous").default(false), // True for ongoing medications
  durationDays: integer("duration_days"), // Null for continuous medications
  intendedStartDate: timestamp("intended_start_date"), // When patient should start
  
  // Multi-specialty tracking
  specialty: varchar("specialty"), // 'cardiology', 'oncology', 'neurology', etc.
  
  // Supersession tracking - same specialty replaces previous
  supersedes: varchar("supersedes"), // ID of the prescription this one replaces
  supersededBy: varchar("superseded_by"), // ID of prescription that replaced this
  supersededAt: timestamp("superseded_at"),
  
  // Cross-specialty conflict tracking
  conflictGroupId: varchar("conflict_group_id"), // Links prescriptions in same conflict
  hasConflict: boolean("has_conflict").default(false),
  conflictDetectedAt: timestamp("conflict_detected_at"),
  conflictResolvedAt: timestamp("conflict_resolved_at"),
  
  // Status and tracking
  status: varchar("status").notNull().default("sent"), // 'sent', 'acknowledged', 'filled', 'active', 'completed', 'expired', 'superseded', 'conflict_hold'
  acknowledgedAt: timestamp("acknowledged_at"),
  acknowledgedBy: varchar("acknowledged_by"), // Patient who acknowledged
  
  // Link to medical document (generated prescription PDF)
  documentId: varchar("document_id"), // Reference to medical_documents
  
  // Link to created medication
  medicationId: varchar("medication_id").references(() => medications.id),
  
  // Audit
  notes: text("notes"), // Doctor's notes about prescription
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
});

// Cross-Specialty Medication Conflicts - Tracks conflicts between different specialists
export const medicationConflicts = pgTable("medication_conflicts", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  patientId: varchar("patient_id").notNull().references(() => users.id),
  conflictGroupId: varchar("conflict_group_id").notNull(), // Groups related medications
  
  // Conflicting medications/prescriptions
  medication1Id: varchar("medication1_id").references(() => medications.id),
  medication2Id: varchar("medication2_id").references(() => medications.id),
  prescription1Id: varchar("prescription1_id"),
  prescription2Id: varchar("prescription2_id"),
  
  // Doctors involved
  doctor1Id: varchar("doctor1_id").notNull().references(() => users.id),
  doctor2Id: varchar("doctor2_id").notNull().references(() => users.id),
  specialty1: varchar("specialty1").notNull(),
  specialty2: varchar("specialty2").notNull(),
  
  // Conflict details
  conflictType: varchar("conflict_type").notNull(), // 'drug_interaction', 'duplicate_therapy', 'dosage_overlap', 'contraindication'
  severity: varchar("severity").notNull(), // 'low', 'moderate', 'high', 'critical'
  description: text("description").notNull(),
  detectedReason: text("detected_reason"), // Why conflict was flagged
  
  // Status tracking
  status: varchar("status").notNull().default("pending"), // 'pending', 'doctor1_reviewed', 'doctor2_reviewed', 'resolved', 'escalated'
  
  // Doctor responses
  doctor1Response: text("doctor1_response"),
  doctor1RespondedAt: timestamp("doctor1_responded_at"),
  doctor1Action: varchar("doctor1_action"), // 'keep', 'modify', 'cancel', 'defer'
  doctor2Response: text("doctor2_response"),
  doctor2RespondedAt: timestamp("doctor2_responded_at"),
  doctor2Action: varchar("doctor2_action"), // 'keep', 'modify', 'cancel', 'defer'
  
  // Resolution
  resolution: varchar("resolution"), // 'keep_both', 'modify_one', 'cancel_one', 'cancel_both', 'escalated'
  resolutionDetails: text("resolution_details"),
  resolvedBy: varchar("resolved_by"), // User ID who resolved
  resolvedAt: timestamp("resolved_at"),
  
  // Notifications
  doctor1NotifiedAt: timestamp("doctor1_notified_at"),
  doctor2NotifiedAt: timestamp("doctor2_notified_at"),
  patientNotifiedAt: timestamp("patient_notified_at"),
  
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
});

export const insertMedicationConflictSchema = createInsertSchema(medicationConflicts).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export type InsertMedicationConflict = z.infer<typeof insertMedicationConflictSchema>;
export type MedicationConflict = typeof medicationConflicts.$inferSelect;

export const insertPrescriptionSchema = createInsertSchema(prescriptions).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export type InsertPrescription = z.infer<typeof insertPrescriptionSchema>;
export type Prescription = typeof prescriptions.$inferSelect;

// Medication Change Log - Complete audit trail of all medication changes
export const medicationChangeLog = pgTable("medication_change_log", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  medicationId: varchar("medication_id").notNull().references(() => medications.id),
  patientId: varchar("patient_id").notNull().references(() => users.id),
  
  // Change type and metadata
  changeType: varchar("change_type").notNull(), // 'added', 'dosage_changed', 'frequency_changed', 'discontinued', 'reactivated'
  changedBy: varchar("changed_by").notNull(), // 'patient', 'doctor', 'system'
  changedByUserId: varchar("changed_by_user_id").notNull(), // User ID who made the change
  
  // For dosage/frequency changes
  oldDosage: varchar("old_dosage"),
  newDosage: varchar("new_dosage"),
  oldFrequency: varchar("old_frequency"),
  newFrequency: varchar("new_frequency"),
  
  // For discontinuation
  discontinuationReason: text("discontinuation_reason"),
  replacementMedicationId: varchar("replacement_medication_id"),
  
  // Additional context
  changeReason: text("change_reason"), // Why the change was made
  notes: text("notes"),
  
  // HIPAA audit tracking
  ipAddress: varchar("ip_address"),
  userAgent: text("user_agent"),
  
  createdAt: timestamp("created_at").defaultNow(),
});

export const insertMedicationChangeLogSchema = createInsertSchema(medicationChangeLog).omit({
  id: true,
  createdAt: true,
});

export type InsertMedicationChangeLog = z.infer<typeof insertMedicationChangeLogSchema>;
export type MedicationChangeLog = typeof medicationChangeLog.$inferSelect;

// Dosage Change Requests - Patient-initiated requests pending doctor approval
export const dosageChangeRequests = pgTable("dosage_change_requests", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  medicationId: varchar("medication_id").notNull().references(() => medications.id),
  patientId: varchar("patient_id").notNull().references(() => users.id),
  
  // Requested changes
  requestType: varchar("request_type").notNull(), // 'dosage_change', 'frequency_change', 'scheduled_change'
  currentDosage: varchar("current_dosage").notNull(),
  requestedDosage: varchar("requested_dosage").notNull(),
  currentFrequency: varchar("current_frequency").notNull(),
  requestedFrequency: varchar("requested_frequency").notNull(),
  
  // Scheduled changes (for titration)
  scheduledChangeDate: timestamp("scheduled_change_date"), // When to apply the change
  
  // Patient's reasoning
  requestReason: text("request_reason").notNull(), // Why patient is requesting change
  additionalNotes: text("additional_notes"),
  
  // Doctor review
  status: varchar("status").notNull().default("pending"), // 'pending', 'approved', 'rejected'
  reviewedByDoctorId: varchar("reviewed_by_doctor_id").references(() => users.id),
  reviewedAt: timestamp("reviewed_at"),
  doctorNotes: text("doctor_notes"), // Doctor's notes on approval/rejection
  
  // Notification tracking
  doctorNotifiedAt: timestamp("doctor_notified_at"),
  patientNotifiedAt: timestamp("patient_notified_at"),
  
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
});

export const insertDosageChangeRequestSchema = createInsertSchema(dosageChangeRequests).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export type InsertDosageChangeRequest = z.infer<typeof insertDosageChangeRequestSchema>;
export type DosageChangeRequest = typeof dosageChangeRequests.$inferSelect;

// Medication-Drug matching audit table
// Tracks how medications are mapped to standardized drugs with confidence scores
export const medicationDrugMatches = pgTable("medication_drug_matches", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  medicationId: varchar("medication_id").notNull().references(() => medications.id),
  drugId: varchar("drug_id").notNull().references(() => drugs.id),
  matchSource: varchar("match_source").notNull(), // 'rxnorm_exact', 'rxnorm_approximate', 'manual', 'openfda'
  confidenceScore: decimal("confidence_score").notNull(), // 0.0-1.0
  matchedBy: varchar("matched_by"), // User ID who confirmed the match
  matchedAt: timestamp("matched_at").defaultNow(),
  isActive: boolean("is_active").default(true), // Allow multiple candidate matches
  matchMetadata: jsonb("match_metadata").$type<{
    rxcuiCandidates?: string[];
    searchTerm?: string;
    apiResponse?: any;
  }>(),
  createdAt: timestamp("created_at").defaultNow(),
});

export const insertMedicationDrugMatchSchema = createInsertSchema(medicationDrugMatches).omit({
  id: true,
  createdAt: true,
  matchedAt: true,
});

export type InsertMedicationDrugMatch = z.infer<typeof insertMedicationDrugMatchSchema>;
export type MedicationDrugMatch = typeof medicationDrugMatches.$inferSelect;

// Drug knowledge base - Comprehensive drug information
export const drugs = pgTable("drugs", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  // RxNorm standardized identifiers
  rxcui: varchar("rxcui").unique(), // RxNorm Concept Unique Identifier
  name: varchar("name").notNull(),
  genericName: varchar("generic_name"),
  brandNames: jsonb("brand_names").$type<string[]>(),
  drugClass: varchar("drug_class"),
  mechanismOfAction: text("mechanism_of_action"),
  molecularFormula: varchar("molecular_formula"),
  molecularWeight: decimal("molecular_weight"),
  fdaApproved: boolean("fda_approved").default(true),
  blackBoxWarning: boolean("black_box_warning").default(false),
  blackBoxWarningText: text("black_box_warning_text"),
  therapeuticCategories: jsonb("therapeutic_categories").$type<string[]>(),
  commonSideEffects: jsonb("common_side_effects").$type<string[]>(),
  seriousSideEffects: jsonb("serious_side_effects").$type<string[]>(),
  contraindicationsForImmuneSuppressed: text("contraindications_for_immune_suppressed"),
  halfLife: varchar("half_life"),
  metabolismPathway: text("metabolism_pathway"),
  cytochromeP450Enzymes: jsonb("cytochrome_p450_enzymes").$type<string[]>(), // CYP2D6, CYP3A4, etc.
  renalExcretion: boolean("renal_excretion").default(false),
  hepaticMetabolism: boolean("hepatic_metabolism").default(false),
  proteinBinding: decimal("protein_binding"), // Percentage
  bioavailability: decimal("bioavailability"), // Percentage
  peakPlasmaTime: varchar("peak_plasma_time"),
  immunocompromisedSafety: varchar("immunocompromised_safety"), // 'safe', 'caution', 'avoid'
  pregnancyCategory: varchar("pregnancy_category"),
  // Data versioning and provenance
  dataSource: varchar("data_source").default("rxnorm"), // 'rxnorm', 'openfda', 'manual'
  dataVersion: varchar("data_version"), // RxNorm release date (YYYYMMDD format)
  lastUpdated: timestamp("last_updated").defaultNow(),
  createdAt: timestamp("created_at").defaultNow(),
});

export const insertDrugSchema = createInsertSchema(drugs).omit({
  id: true,
  createdAt: true,
  lastUpdated: true,
});

export type InsertDrug = z.infer<typeof insertDrugSchema>;
export type Drug = typeof drugs.$inferSelect;

// Drug-drug interactions with AI-powered analysis
export const drugInteractions = pgTable("drug_interactions", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  drug1Id: varchar("drug1_id").notNull().references(() => drugs.id),
  drug2Id: varchar("drug2_id").notNull().references(() => drugs.id),
  severityLevel: varchar("severity_level").notNull(), // 'severe', 'moderate', 'minor'
  interactionType: varchar("interaction_type").notNull(), // 'pharmacokinetic', 'pharmacodynamic', 'synergistic', 'antagonistic'
  mechanismDescription: text("mechanism_description").notNull(),
  clinicalEffects: text("clinical_effects").notNull(),
  managementRecommendations: text("management_recommendations").notNull(),
  alternativeSuggestions: jsonb("alternative_suggestions").$type<string[]>(),
  onsetTimeframe: varchar("onset_timeframe"), // 'immediate', 'hours', 'days', 'weeks'
  riskForImmunocompromised: varchar("risk_for_immunocompromised"), // 'high', 'medium', 'low'
  requiresMonitoring: boolean("requires_monitoring").default(false),
  monitoringParameters: jsonb("monitoring_parameters").$type<string[]>(),
  evidenceLevel: varchar("evidence_level"), // 'proven', 'probable', 'theoretical'
  literatureReferences: jsonb("literature_references").$type<Array<{ title: string; url: string; year: number }>>(),
  fdaWarning: boolean("fda_warning").default(false),
  aiAnalysisConfidence: decimal("ai_analysis_confidence"), // 0-100% confidence score
  detectedByGNN: boolean("detected_by_gnn").default(false), // Graph Neural Network detection
  detectedByNLP: boolean("detected_by_nlp").default(false), // Natural Language Processing detection
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
});

export const insertDrugInteractionSchema = createInsertSchema(drugInteractions).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export type InsertDrugInteraction = z.infer<typeof insertDrugInteractionSchema>;
export type DrugInteraction = typeof drugInteractions.$inferSelect;

// Patient-specific interaction alerts
export const interactionAlerts = pgTable("interaction_alerts", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  patientId: varchar("patient_id").notNull().references(() => users.id),
  medication1Id: varchar("medication1_id").notNull().references(() => medications.id),
  medication2Id: varchar("medication2_id").notNull().references(() => medications.id),
  interactionId: varchar("interaction_id").notNull().references(() => drugInteractions.id),
  alertStatus: varchar("alert_status").notNull().default("active"), // 'active', 'acknowledged', 'overridden', 'resolved'
  acknowledgedBy: varchar("acknowledged_by"), // User ID who acknowledged
  acknowledgedAt: timestamp("acknowledged_at"),
  overrideReason: text("override_reason"),
  overrideBy: varchar("override_by"), // Doctor ID who overrode
  overrideAt: timestamp("override_at"),
  notifiedPatient: boolean("notified_patient").default(false),
  notifiedDoctor: boolean("notified_doctor").default(false),
  smsAlertSent: boolean("sms_alert_sent").default(false),
  emailAlertSent: boolean("email_alert_sent").default(false),
  clonaMentioned: boolean("clona_mentioned").default(false), // Mentioned in Agent Clona chat
  criticalityScore: integer("criticality_score"), // 1-10 scale based on patient profile
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
});

export const insertInteractionAlertSchema = createInsertSchema(interactionAlerts).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export type InsertInteractionAlert = z.infer<typeof insertInteractionAlertSchema>;
export type InteractionAlert = typeof interactionAlerts.$inferSelect;

// Pharmacogenomic profiles for personalized drug-gene interactions
export const pharmacogenomicProfiles = pgTable("pharmacogenomic_profiles", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  patientId: varchar("patient_id").notNull().unique().references(() => users.id),
  testProvider: varchar("test_provider"), // '23andMe', 'AncestryDNA', 'medical_lab', etc.
  testDate: timestamp("test_date"),
  cypEnzymes: jsonb("cyp_enzymes").$type<Array<{ enzyme: string; genotype: string; phenotype: string; activity: string }>>(), // CYP2D6, CYP3A4, etc.
  slcoTransporters: jsonb("slco_transporters").$type<Array<{ transporter: string; genotype: string; activity: string }>>(),
  hlaAlleles: jsonb("hla_alleles").$type<Array<{ allele: string; present: boolean; riskLevel: string }>>(), // HLA-B*57:01, etc.
  vkorc1Genotype: varchar("vkorc1_genotype"), // For warfarin dosing
  tpmtGenotype: varchar("tpmt_genotype"), // For azathioprine/mercaptopurine
  dpydGenotype: varchar("dpyd_genotype"), // For 5-FU/capecitabine
  g6pdDeficiency: boolean("g6pd_deficiency").default(false),
  otherPharmacogenes: jsonb("other_pharmacogenes").$type<Array<{ gene: string; variants: string[]; clinicalSignificance: string }>>(),
  ethnicBackground: varchar("ethnic_background"), // Important for pharmacogenomics
  consentForResearch: boolean("consent_for_research").default(false),
  dataSource: varchar("data_source"), // 'upload', 'ehr', 'lab_integration'
  rawDataFileUrl: varchar("raw_data_file_url"), // S3 URL for uploaded genetic data
  interpreterNotes: text("interpreter_notes"),
  lastUpdated: timestamp("last_updated").defaultNow(),
  createdAt: timestamp("created_at").defaultNow(),
});

export const insertPharmacogenomicProfileSchema = createInsertSchema(pharmacogenomicProfiles).omit({
  id: true,
  createdAt: true,
  lastUpdated: true,
});

export type InsertPharmacogenomicProfile = z.infer<typeof insertPharmacogenomicProfileSchema>;
export type PharmacogenomicProfile = typeof pharmacogenomicProfiles.$inferSelect;

// Drug-gene interactions for pharmacogenomic warnings
export const drugGeneInteractions = pgTable("drug_gene_interactions", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  drugId: varchar("drug_id").notNull().references(() => drugs.id),
  gene: varchar("gene").notNull(), // 'CYP2D6', 'HLA-B*57:01', etc.
  allele: varchar("allele"), // Specific allele if applicable
  phenotype: varchar("phenotype"), // 'poor metabolizer', 'rapid metabolizer', etc.
  clinicalEffect: text("clinical_effect").notNull(),
  recommendation: text("recommendation").notNull(),
  dosageAdjustment: text("dosage_adjustment"),
  alternativeDrug: varchar("alternative_drug"),
  riskLevel: varchar("risk_level").notNull(), // 'high', 'moderate', 'low'
  fdaGuideline: boolean("fda_guideline").default(false),
  cpicGuideline: boolean("cpic_guideline").default(false), // Clinical Pharmacogenetics Implementation Consortium
  evidence: text("evidence"),
  references: jsonb("references").$type<Array<{ title: string; url: string }>>(),
  createdAt: timestamp("created_at").defaultNow(),
});

export const insertDrugGeneInteractionSchema = createInsertSchema(drugGeneInteractions).omit({
  id: true,
  createdAt: true,
});

export type InsertDrugGeneInteraction = z.infer<typeof insertDrugGeneInteractionSchema>;
export type DrugGeneInteraction = typeof drugGeneInteractions.$inferSelect;

// ==================== ADAPTIVE MEDICATION & NUTRITION INSIGHTS ====================

// Medication schedules - When medications should be taken
export const medicationSchedules = pgTable("medication_schedules", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  medicationId: varchar("medication_id").notNull().references(() => medications.id),
  patientId: varchar("patient_id").notNull().references(() => users.id),
  timeOfDay: varchar("time_of_day").notNull(), // '08:00', '12:00', '20:00', etc.
  withFood: boolean("with_food").default(false),
  withWater: boolean("with_water").default(true),
  specialInstructions: text("special_instructions"),
  aiOptimized: boolean("ai_optimized").default(false), // AI-suggested optimal timing
  aiReasoning: text("ai_reasoning"), // Why AI suggested this timing
  reminderEnabled: boolean("reminder_enabled").default(true),
  smsReminderSent: boolean("sms_reminder_sent").default(false),
  active: boolean("active").default(true),
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
});

export const insertMedicationScheduleSchema = createInsertSchema(medicationSchedules).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export type InsertMedicationSchedule = z.infer<typeof insertMedicationScheduleSchema>;
export type MedicationSchedule = typeof medicationSchedules.$inferSelect;

// Medication adherence tracking
export const medicationAdherence = pgTable("medication_adherence", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  medicationId: varchar("medication_id").notNull().references(() => medications.id),
  scheduleId: varchar("schedule_id").references(() => medicationSchedules.id),
  patientId: varchar("patient_id").notNull().references(() => users.id),
  scheduledTime: timestamp("scheduled_time").notNull(),
  takenAt: timestamp("taken_at"),
  status: varchar("status").notNull().default("pending"), // 'pending', 'taken', 'missed', 'skipped'
  skipReason: text("skip_reason"),
  sideEffects: text("side_effects"),
  effectivenessRating: integer("effectiveness_rating"), // 1-10 scale
  loggedBy: varchar("logged_by").default("patient"), // 'patient', 'companion', 'ai'
  companionChecked: boolean("companion_checked").default(false), // Checked in Health Companion Mode
  createdAt: timestamp("created_at").defaultNow(),
});

export const insertMedicationAdherenceSchema = createInsertSchema(medicationAdherence).omit({
  id: true,
  createdAt: true,
});

export type InsertMedicationAdherence = z.infer<typeof insertMedicationAdherenceSchema>;
export type MedicationAdherence = typeof medicationAdherence.$inferSelect;

// Dietary preferences and restrictions
export const dietaryPreferences = pgTable("dietary_preferences", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  patientId: varchar("patient_id").notNull().unique().references(() => users.id),
  dietType: varchar("diet_type"), // 'omnivore', 'vegetarian', 'vegan', 'pescatarian', 'keto', 'paleo', etc.
  allergies: jsonb("allergies").$type<string[]>(), // Food allergies
  intolerances: jsonb("intolerances").$type<string[]>(), // Lactose, gluten, etc.
  dislikes: jsonb("dislikes").$type<string[]>(), // Foods to avoid
  culturalRestrictions: jsonb("cultural_restrictions").$type<string[]>(),
  religiousRestrictions: jsonb("religious_restrictions").$type<string[]>(),
  calorieTarget: integer("calorie_target"), // Daily calorie goal
  proteinTarget: integer("protein_target"), // Grams per day
  carbTarget: integer("carb_target"), // Grams per day
  fatTarget: integer("fat_target"), // Grams per day
  immuneBoostingFocus: boolean("immune_boosting_focus").default(true), // Focus on immune-boosting foods
  preferredCuisines: jsonb("preferred_cuisines").$type<string[]>(),
  cookingSkillLevel: varchar("cooking_skill_level"), // 'beginner', 'intermediate', 'advanced'
  mealsPerDay: integer("meals_per_day").default(3),
  snacksPerDay: integer("snacks_per_day").default(2),
  updatedAt: timestamp("updated_at").defaultNow(),
  createdAt: timestamp("created_at").defaultNow(),
});

export const insertDietaryPreferenceSchema = createInsertSchema(dietaryPreferences).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export type InsertDietaryPreference = z.infer<typeof insertDietaryPreferenceSchema>;
export type DietaryPreference = typeof dietaryPreferences.$inferSelect;

// AI-generated meal plans
export const mealPlans = pgTable("meal_plans", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  patientId: varchar("patient_id").notNull().references(() => users.id),
  weekStartDate: timestamp("week_start_date").notNull(), // Start of the week for this plan
  planName: varchar("plan_name").notNull(), // "Immune-Boosting Week Plan"
  aiGeneratedSummary: text("ai_generated_summary"), // AI explanation of the plan
  totalCalories: integer("total_calories"),
  focusAreas: jsonb("focus_areas").$type<string[]>(), // ['immune_support', 'energy', 'inflammation_reduction']
  considersMedications: boolean("considers_medications").default(true), // Plan considers medication timing
  active: boolean("active").default(true),
  generatedAt: timestamp("generated_at").defaultNow(),
  createdAt: timestamp("created_at").defaultNow(),
});

export const insertMealPlanSchema = createInsertSchema(mealPlans).omit({
  id: true,
  createdAt: true,
  generatedAt: true,
});

export type InsertMealPlan = z.infer<typeof insertMealPlanSchema>;
export type MealPlan = typeof mealPlans.$inferSelect;

// Individual meals (planned or logged)
export const meals = pgTable("meals", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  patientId: varchar("patient_id").notNull().references(() => users.id),
  mealPlanId: varchar("meal_plan_id").references(() => mealPlans.id), // If part of a plan
  mealType: varchar("meal_type").notNull(), // 'breakfast', 'lunch', 'dinner', 'snack'
  mealName: varchar("meal_name").notNull(),
  description: text("description"),
  ingredients: jsonb("ingredients").$type<Array<{ name: string; amount: string; unit: string }>>(),
  recipeSuggestion: text("recipe_suggestion"), // AI-generated recipe
  scheduledTime: timestamp("scheduled_time"),
  actualTime: timestamp("actual_time"),
  status: varchar("status").default("planned"), // 'planned', 'eaten', 'skipped'
  photoUrl: varchar("photo_url"), // S3 URL for meal photo
  aiNutritionAnalysis: text("ai_nutrition_analysis"), // AI-generated nutrition insights
  immuneBenefits: jsonb("immune_benefits").$type<string[]>(), // ['high_vitamin_c', 'probiotic', 'anti_inflammatory']
  companionLogged: boolean("companion_logged").default(false), // Logged via Health Companion Mode
  createdAt: timestamp("created_at").defaultNow(),
});

export const insertMealSchema = createInsertSchema(meals).omit({
  id: true,
  createdAt: true,
});

export type InsertMeal = z.infer<typeof insertMealSchema>;
export type Meal = typeof meals.$inferSelect;

// Nutrition entries for meals
export const nutritionEntries = pgTable("nutrition_entries", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  mealId: varchar("meal_id").notNull().references(() => meals.id),
  patientId: varchar("patient_id").notNull().references(() => users.id),
  calories: integer("calories"),
  protein: decimal("protein"), // Grams
  carbs: decimal("carbs"), // Grams
  fat: decimal("fat"), // Grams
  fiber: decimal("fiber"), // Grams
  sugar: decimal("sugar"), // Grams
  sodium: decimal("sodium"), // Milligrams
  vitaminC: decimal("vitamin_c"), // Milligrams
  vitaminD: decimal("vitamin_d"), // Micrograms
  zinc: decimal("zinc"), // Milligrams
  iron: decimal("iron"), // Milligrams
  calcium: decimal("calcium"), // Milligrams
  omega3: decimal("omega_3"), // Grams
  antioxidantScore: integer("antioxidant_score"), // 1-100 scale
  immuneSupportScore: integer("immune_support_score"), // 1-100 AI-calculated
  dataSource: varchar("data_source").default("ai_estimation"), // 'ai_estimation', 'manual', 'api'
  createdAt: timestamp("created_at").defaultNow(),
});

export const insertNutritionEntrySchema = createInsertSchema(nutritionEntries).omit({
  id: true,
  createdAt: true,
});

export type InsertNutritionEntry = z.infer<typeof insertNutritionEntrySchema>;
export type NutritionEntry = typeof nutritionEntries.$inferSelect;

// ==================== HEALTH COMPANION MODE ====================

// Natural conversational check-ins
export const companionCheckIns = pgTable("companion_check_ins", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  patientId: varchar("patient_id").notNull().references(() => users.id),
  checkInType: varchar("check_in_type").notNull(), // 'morning', 'midday', 'evening', 'medication', 'meal', 'mood', 'symptom', 'spontaneous'
  conversationSummary: text("conversation_summary").notNull(), // AI-generated summary
  naturalLanguageInput: text("natural_language_input"), // What the patient said
  extractedData: jsonb("extracted_data").$type<{
    mood?: string;
    energy?: number;
    symptoms?: string[];
    medications?: Array<{ name: string; taken: boolean }>;
    meals?: Array<{ type: string; description: string }>;
    concerns?: string[];
  }>(), // AI-extracted structured data
  empathyLevel: varchar("empathy_level").default("supportive"), // 'supportive', 'encouraging', 'urgent', 'celebratory'
  aiResponse: text("ai_response").notNull(), // What companion said back
  sentimentScore: decimal("sentiment_score"), // -1 to 1 (negative to positive)
  concernsRaised: boolean("concerns_raised").default(false),
  needsFollowup: boolean("needs_followup").default(false),
  followupReason: text("followup_reason"),
  notifiedDoctor: boolean("notified_doctor").default(false),
  sessionDuration: integer("session_duration"), // Seconds
  interactionCount: integer("interaction_count").default(1), // Number of back-and-forth messages
  checkedInAt: timestamp("checked_in_at").defaultNow(),
  createdAt: timestamp("created_at").defaultNow(),
});

export const insertCompanionCheckInSchema = createInsertSchema(companionCheckIns).omit({
  id: true,
  createdAt: true,
});

export type InsertCompanionCheckIn = z.infer<typeof insertCompanionCheckInSchema>;
export type CompanionCheckIn = typeof companionCheckIns.$inferSelect;

// Companion engagement tracking
export const companionEngagement = pgTable("companion_engagement", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  patientId: varchar("patient_id").notNull().unique().references(() => users.id),
  totalCheckIns: integer("total_check_ins").default(0),
  currentStreak: integer("current_streak").default(0), // Days in a row
  longestStreak: integer("longest_streak").default(0),
  lastCheckInDate: timestamp("last_check_in_date"),
  favoriteCheckInType: varchar("favorite_check_in_type"),
  avgSentimentScore: decimal("avg_sentiment_score"),
  totalConcernsRaised: integer("total_concerns_raised").default(0),
  companionPersonality: varchar("companion_personality").default("empathetic"), // 'empathetic', 'motivational', 'clinical', 'friend'
  preferredTone: varchar("preferred_tone").default("warm"), // 'warm', 'professional', 'casual', 'cheerful'
  notificationPreference: varchar("notification_preference").default("gentle"), // 'gentle', 'standard', 'none'
  bestCheckInTime: varchar("best_check_in_time"), // '09:00', '14:00', etc. AI-learned
  engagementScore: integer("engagement_score").default(50), // 1-100 based on usage patterns
  updatedAt: timestamp("updated_at").defaultNow(),
  createdAt: timestamp("created_at").defaultNow(),
});

export const insertCompanionEngagementSchema = createInsertSchema(companionEngagement).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export type InsertCompanionEngagement = z.infer<typeof insertCompanionEngagementSchema>;
export type CompanionEngagement = typeof companionEngagement.$inferSelect;

// Dynamic tasks
export const dynamicTasks = pgTable("dynamic_tasks", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  patientId: varchar("patient_id").notNull().references(() => users.id),
  title: varchar("title").notNull(),
  description: text("description"),
  taskType: varchar("task_type"), // 'water', 'exercise', 'medication', 'temperature', etc.
  frequency: varchar("frequency"), // '8hourly', 'daily', 'weekly', etc.
  dueDate: timestamp("due_date"),
  completed: boolean("completed").default(false),
  completedAt: timestamp("completed_at"),
  generatedBy: varchar("generated_by"), // 'ai', 'doctor', 'system'
  relatedCondition: varchar("related_condition"),
  createdAt: timestamp("created_at").defaultNow(),
});

export const insertDynamicTaskSchema = createInsertSchema(dynamicTasks).omit({
  id: true,
  createdAt: true,
});

export type InsertDynamicTask = z.infer<typeof insertDynamicTaskSchema>;
export type DynamicTask = typeof dynamicTasks.$inferSelect;

// Auto journals
export const autoJournals = pgTable("auto_journals", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  patientId: varchar("patient_id").notNull().references(() => users.id),
  date: timestamp("date").notNull().defaultNow(),
  content: text("content").notNull(),
  summary: text("summary"),
  mood: varchar("mood"),
  stressLevel: integer("stress_level"), // 1-10 scale
  generatedFromData: jsonb("generated_from_data").$type<{ followup: boolean; symptoms: boolean; wearables: boolean }>(),
  createdAt: timestamp("created_at").defaultNow(),
});

export const insertAutoJournalSchema = createInsertSchema(autoJournals).omit({
  id: true,
  createdAt: true,
});

export type InsertAutoJournal = z.infer<typeof insertAutoJournalSchema>;
export type AutoJournal = typeof autoJournals.$inferSelect;

// Calm activities
export const calmActivities = pgTable("calm_activities", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  patientId: varchar("patient_id").notNull().references(() => users.id),
  activityType: varchar("activity_type").notNull(), // 'meditation', 'breathing', 'music', 'nature_sounds'
  activityName: varchar("activity_name").notNull(),
  duration: integer("duration"), // minutes
  effectiveness: integer("effectiveness"), // 1-10 scale
  timesUsed: integer("times_used").default(0),
  lastUsed: timestamp("last_used"),
  createdAt: timestamp("created_at").defaultNow(),
});

export const insertCalmActivitySchema = createInsertSchema(calmActivities).omit({
  id: true,
  createdAt: true,
});

export type InsertCalmActivity = z.infer<typeof insertCalmActivitySchema>;
export type CalmActivity = typeof calmActivities.$inferSelect;

// Behavioral insights
export const behavioralInsights = pgTable("behavioral_insights", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  patientId: varchar("patient_id").notNull().references(() => users.id),
  date: timestamp("date").notNull().defaultNow(),
  stressScore: integer("stress_score"), // 1-10 scale
  joyScore: integer("joy_score"), // 1-10 scale
  appUsageDuration: integer("app_usage_duration"), // minutes
  interactionSpeed: varchar("interaction_speed"), // 'slow', 'normal', 'fast'
  sentimentAnalysis: jsonb("sentiment_analysis").$type<{ positive: number; negative: number; neutral: number }>(),
  sleepQuality: integer("sleep_quality"), // 1-10 scale from wearables
  activityLevel: varchar("activity_level"), // 'low', 'moderate', 'high'
  createdAt: timestamp("created_at").defaultNow(),
});

export const insertBehavioralInsightSchema = createInsertSchema(behavioralInsights).omit({
  id: true,
  createdAt: true,
});

export type InsertBehavioralInsight = z.infer<typeof insertBehavioralInsightSchema>;
export type BehavioralInsight = typeof behavioralInsights.$inferSelect;

// Research consents
export const researchConsents = pgTable("research_consents", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  patientId: varchar("patient_id").notNull().references(() => users.id),
  consentGiven: boolean("consent_given").default(false),
  consentDate: timestamp("consent_date"),
  verifiedBy: varchar("verified_by").references(() => users.id),
  verificationDate: timestamp("verification_date"),
  dataAnonymized: boolean("data_anonymized").default(false),
  status: varchar("status").default("pending"), // 'pending', 'approved', 'rejected'
  createdAt: timestamp("created_at").defaultNow(),
});

export const insertResearchConsentSchema = createInsertSchema(researchConsents).omit({
  id: true,
  createdAt: true,
});

export type InsertResearchConsent = z.infer<typeof insertResearchConsentSchema>;
export type ResearchConsent = typeof researchConsents.$inferSelect;

// AI research reports
export const aiResearchReports = pgTable("ai_research_reports", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  createdBy: varchar("created_by").notNull().references(() => users.id),
  title: varchar("title").notNull(),
  summary: text("summary").notNull(),
  findings: jsonb("findings").$type<Array<{ finding: string; significance: string; confidence: number }>>(),
  visualizations: jsonb("visualizations").$type<Array<{ type: string; data: any; title: string }>>(),
  patientCohortSize: integer("patient_cohort_size"),
  analysisType: varchar("analysis_type"), // 'correlation', 'regression', 'survival', 'pattern'
  publishReady: boolean("publish_ready").default(false),
  createdAt: timestamp("created_at").defaultNow(),
});

export const insertAIResearchReportSchema = createInsertSchema(aiResearchReports).omit({
  id: true,
  createdAt: true,
});

export type InsertAIResearchReport = z.infer<typeof insertAIResearchReportSchema>;
export type AIResearchReport = typeof aiResearchReports.$inferSelect;

// Educational content progress
export const educationalProgress = pgTable("educational_progress", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  patientId: varchar("patient_id").notNull().references(() => users.id),
  moduleId: varchar("module_id").notNull(),
  moduleName: varchar("module_name").notNull(),
  category: varchar("category"), // 'immunocompromised', 'diabetes', 'copd', etc.
  progress: integer("progress").default(0), // 0-100%
  completed: boolean("completed").default(false),
  completedAt: timestamp("completed_at"),
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
});

export const insertEducationalProgressSchema = createInsertSchema(educationalProgress).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export type InsertEducationalProgress = z.infer<typeof insertEducationalProgressSchema>;
export type EducationalProgress = typeof educationalProgress.$inferSelect;

// Psychological counseling sessions
export const counselingSessions = pgTable("counseling_sessions", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  userId: varchar("user_id").notNull().references(() => users.id),
  sessionDate: timestamp("session_date").notNull(),
  duration: integer("duration"), // minutes
  counselorName: varchar("counselor_name"),
  sessionType: varchar("session_type"), // 'individual', 'group', 'family', 'emergency'
  sessionMode: varchar("session_mode"), // 'in-person', 'video', 'phone'
  mainConcerns: text("main_concerns").array(),
  sessionNotes: text("session_notes"),
  followUpRequired: boolean("follow_up_required").default(false),
  followUpDate: timestamp("follow_up_date"),
  status: varchar("status").default("scheduled"), // 'scheduled', 'completed', 'cancelled', 'no-show'
  effectiveness: integer("effectiveness"), // 1-10 scale, rated after session
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
});

export const insertCounselingSessionSchema = createInsertSchema(counselingSessions).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export type InsertCounselingSession = z.infer<typeof insertCounselingSessionSchema>;
export type CounselingSession = typeof counselingSessions.$inferSelect;

// Training datasets from public sources (PubMed, PhysioNet, Kaggle)
export const trainingDatasets = pgTable("training_datasets", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  uploadedBy: varchar("uploaded_by").notNull().references(() => users.id),
  source: varchar("source").notNull(), // 'pubmed', 'physionet', 'kaggle', 'who'
  sourceId: varchar("source_id"), // External ID from source platform
  title: varchar("title").notNull(),
  description: text("description"),
  dataType: varchar("data_type"), // 'ecg', 'clinical_notes', 'lab_results', 'imaging', 'research_paper', etc.
  recordCount: integer("record_count"),
  fileSize: integer("file_size"), // bytes
  filePath: varchar("file_path"), // local storage path
  metadata: jsonb("metadata").$type<{ 
    authors?: string[];
    publicationDate?: string;
    doi?: string;
    tags?: string[];
    version?: string;
    license?: string;
  }>(),
  downloadStatus: varchar("download_status").default("pending"), // 'pending', 'downloading', 'completed', 'failed'
  usedForTraining: boolean("used_for_training").default(false),
  createdAt: timestamp("created_at").defaultNow(),
});

export const insertTrainingDatasetSchema = createInsertSchema(trainingDatasets).omit({
  id: true,
  createdAt: true,
});

export type InsertTrainingDataset = z.infer<typeof insertTrainingDatasetSchema>;
export type TrainingDataset = typeof trainingDatasets.$inferSelect;

// Health insight consent for third-party app integration
export const healthInsightConsents = pgTable("health_insight_consents", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  userId: varchar("user_id").notNull().references(() => users.id),
  appName: varchar("app_name").notNull(), // e.g., "Fitbit", "Apple Health", "Google Fit", "MyFitnessPal"
  appCategory: varchar("app_category"), // 'fitness', 'nutrition', 'mental_health', 'sleep', 'medication'
  consentGranted: boolean("consent_granted").notNull().default(true),
  dataTypes: jsonb("data_types").$type<string[]>(), // ['heart_rate', 'steps', 'sleep', 'weight', 'blood_pressure']
  purpose: text("purpose"), // Why the consent is needed
  sharingFrequency: varchar("sharing_frequency").default("real-time"), // 'real-time', 'daily', 'weekly'
  expiryDate: timestamp("expiry_date"), // When consent expires (null = no expiry)
  revokedAt: timestamp("revoked_at"),
  revokedReason: text("revoked_reason"),
  lastSyncedAt: timestamp("last_synced_at"),
  syncStatus: varchar("sync_status").default("active"), // 'active', 'paused', 'error', 'revoked'
  metadata: jsonb("metadata").$type<{
    apiVersion?: string;
    permissions?: string[];
    scopes?: string[];
  }>(),
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
});

export const insertHealthInsightConsentSchema = createInsertSchema(healthInsightConsents).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export type InsertHealthInsightConsent = z.infer<typeof insertHealthInsightConsentSchema>;
export type HealthInsightConsent = typeof healthInsightConsents.$inferSelect;

// Consultation requests and sessions
export const consultations = pgTable("consultations", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  patientId: varchar("patient_id").notNull().references(() => users.id),
  doctorId: varchar("doctor_id").references(() => users.id),
  sessionId: varchar("session_id").references(() => chatSessions.id), // Link to chat session that prompted consultation
  
  // Request details
  requestReason: text("request_reason").notNull(),
  urgencyLevel: varchar("urgency_level").default("normal"), // 'low', 'normal', 'high', 'urgent'
  status: varchar("status").default("pending"), // 'pending', 'accepted', 'in_progress', 'completed', 'cancelled', 'rejected'
  
  // Financial
  creditsCharged: integer("credits_charged").default(20), // 20 credits = 1 consultation
  creditsPaid: boolean("credits_paid").default(false),
  
  // Session timing
  duration: integer("duration").default(10), // minutes (default 10min)
  scheduledAt: timestamp("scheduled_at"),
  startedAt: timestamp("started_at"),
  completedAt: timestamp("completed_at"),
  
  // Notes and outcomes
  doctorNotes: text("doctor_notes"),
  recommendations: text("recommendations").array(),
  followUpRequired: boolean("follow_up_required").default(false),
  followUpDate: timestamp("follow_up_date"),
  
  // Ratings
  patientRating: integer("patient_rating"), // 1-5 stars
  patientFeedback: text("patient_feedback"),
  
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
});

export const insertConsultationSchema = createInsertSchema(consultations).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export type InsertConsultation = z.infer<typeof insertConsultationSchema>;
export type Consultation = typeof consultations.$inferSelect;

// Doctor-to-doctor consultation consents
export const doctorConsultationConsents = pgTable("doctor_consultation_consents", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  patientId: varchar("patient_id").notNull().references(() => users.id),
  requestingDoctorId: varchar("requesting_doctor_id").notNull().references(() => users.id),
  consultingDoctorId: varchar("consulting_doctor_id").notNull().references(() => users.id),
  
  // Consent details
  purpose: text("purpose").notNull(), // Why the consultation is needed
  dataToShare: jsonb("data_to_share").$type<{
    medicalHistory?: boolean;
    labResults?: boolean;
    medications?: boolean;
    chatSessions?: boolean;
    specificSessionIds?: string[];
  }>(),
  
  // Consent status
  consentStatus: varchar("consent_status").default("pending"), // 'pending', 'approved', 'denied', 'revoked'
  consentGrantedAt: timestamp("consent_granted_at"),
  consentRevokedAt: timestamp("consent_revoked_at"),
  revokeReason: text("revoke_reason"),
  
  // Consultation details
  consultationCompleted: boolean("consultation_completed").default(false),
  consultationDate: timestamp("consultation_date"),
  consultationNotes: text("consultation_notes"),
  
  // Expiry
  expiresAt: timestamp("expires_at"), // Auto-revoke after certain time
  
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
});

export const insertDoctorConsultationConsentSchema = createInsertSchema(doctorConsultationConsents).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export type InsertDoctorConsultationConsent = z.infer<typeof insertDoctorConsultationConsentSchema>;
export type DoctorConsultationConsent = typeof doctorConsultationConsents.$inferSelect;

// Credit transactions log
export const creditTransactions = pgTable("credit_transactions", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  userId: varchar("user_id").notNull().references(() => users.id),
  
  transactionType: varchar("transaction_type").notNull(), // 'earned', 'spent', 'purchased', 'refunded', 'withdrawn'
  amount: integer("amount").notNull(), // Can be negative for spending
  balanceAfter: integer("balance_after").notNull(),
  
  // References
  consultationId: varchar("consultation_id").references(() => consultations.id),
  stripePaymentId: varchar("stripe_payment_id"), // For withdrawal transactions
  
  description: text("description"),
  metadata: jsonb("metadata").$type<{
    subscriptionPeriod?: string;
    withdrawalAmount?: number;
    patientName?: string;
  }>(),
  
  createdAt: timestamp("created_at").defaultNow(),
});

export const insertCreditTransactionSchema = createInsertSchema(creditTransactions).omit({
  id: true,
  createdAt: true,
});

export type InsertCreditTransaction = z.infer<typeof insertCreditTransactionSchema>;
export type CreditTransaction = typeof creditTransactions.$inferSelect;

// EHR System Connections (Epic, Oracle Cerner, Athena Health, etc.)
export const ehrConnections = pgTable("ehr_connections", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  userId: varchar("user_id").notNull().references(() => users.id),
  
  // EHR Platform details
  ehrSystem: varchar("ehr_system").notNull(), // 'epic', 'cerner', 'athena', 'eclinicalworks', 'allscripts', 'advancedmd', 'meditech', 'nextgen', 'drchrono'
  ehrSystemName: varchar("ehr_system_name").notNull(), // Display name
  
  // Connection details
  connectionStatus: varchar("connection_status").default("pending"), // 'pending', 'connected', 'disconnected', 'error'
  accessToken: varchar("access_token"), // OAuth token (encrypted)
  refreshToken: varchar("refresh_token"), // OAuth refresh token (encrypted)
  tokenExpiresAt: timestamp("token_expires_at"),
  
  // Patient identifiers in the EHR system
  patientExternalId: varchar("patient_external_id"), // Patient ID in EHR system
  facilityId: varchar("facility_id"), // Hospital/clinic ID
  facilityName: varchar("facility_name"),
  
  // Sync settings
  autoSync: boolean("auto_sync").default(true),
  syncFrequency: varchar("sync_frequency").default("daily"), // 'real-time', 'hourly', 'daily', 'weekly'
  lastSyncedAt: timestamp("last_synced_at"),
  lastSyncStatus: varchar("last_sync_status"), // 'success', 'partial', 'failed'
  lastSyncError: text("last_sync_error"),
  
  // Data types synced
  syncedDataTypes: jsonb("synced_data_types").$type<string[]>(), // ['vitals', 'medications', 'lab_results', 'allergies', 'immunizations', 'conditions', 'procedures']
  
  // Metadata
  metadata: jsonb("metadata").$type<{
    ehrVersion?: string;
    fhirVersion?: string;
    apiEndpoint?: string;
    scopes?: string[];
  }>(),
  
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
});

export const insertEhrConnectionSchema = createInsertSchema(ehrConnections).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export type InsertEhrConnection = z.infer<typeof insertEhrConnectionSchema>;
export type EhrConnection = typeof ehrConnections.$inferSelect;

// Wearable Device Integrations (Amazfit, Garmin, Whoop, Samsung, Eko)
export const wearableIntegrations = pgTable("wearable_integrations", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  userId: varchar("user_id").notNull().references(() => users.id),
  
  // Device details
  deviceType: varchar("device_type").notNull(), // 'amazfit', 'garmin', 'whoop', 'samsung', 'eko'
  deviceName: varchar("device_name").notNull(), // Display name
  deviceModel: varchar("device_model"), // Specific model (e.g., "Amazfit GTR 3", "Garmin Forerunner 945")
  
  // Connection details
  connectionStatus: varchar("connection_status").default("pending"), // 'pending', 'connected', 'disconnected', 'error'
  accessToken: varchar("access_token"), // OAuth token (encrypted)
  refreshToken: varchar("refresh_token"), // OAuth refresh token (encrypted)
  tokenExpiresAt: timestamp("token_expires_at"),
  
  // Device identifiers
  deviceId: varchar("device_id"), // Unique device identifier from manufacturer
  
  // Sync settings
  autoSync: boolean("auto_sync").default(true),
  syncFrequency: varchar("sync_frequency").default("real-time"), // 'real-time', 'hourly', 'daily'
  lastSyncedAt: timestamp("last_synced_at"),
  lastSyncStatus: varchar("last_sync_status"), // 'success', 'partial', 'failed'
  lastSyncError: text("last_sync_error"),
  
  // Data types tracked
  trackedMetrics: jsonb("tracked_metrics").$type<string[]>(), // ['heart_rate', 'steps', 'sleep', 'spo2', 'temperature', 'respiratory_rate', 'stress', 'hrv', 'ecg']
  
  // Battery and device health
  batteryLevel: integer("battery_level"), // 0-100%
  lastBatterySyncedAt: timestamp("last_battery_synced_at"),
  firmwareVersion: varchar("firmware_version"),
  
  // Metadata
  metadata: jsonb("metadata").$type<{
    apiVersion?: string;
    permissions?: string[];
    scopes?: string[];
    pairedAt?: string;
  }>(),
  
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
});

export const insertWearableIntegrationSchema = createInsertSchema(wearableIntegrations).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export type InsertWearableIntegration = z.infer<typeof insertWearableIntegrationSchema>;
export type WearableIntegration = typeof wearableIntegrations.$inferSelect;

// Immune Biomarkers - Digital biomarkers from wearables for immune monitoring
export const immuneBiomarkers = pgTable("immune_biomarkers", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  userId: varchar("user_id").notNull().references(() => users.id),
  
  // Data source
  dataSource: varchar("data_source").notNull(), // 'fitbit', 'apple_health', 'google_fit', 'garmin', 'amazfit', 'manual'
  wearableIntegrationId: varchar("wearable_integration_id").references(() => wearableIntegrations.id),
  
  // Timestamp of measurement
  measuredAt: timestamp("measured_at").notNull(),
  
  // Heart Rate Variability (key immune indicator)
  hrvRmssd: decimal("hrv_rmssd"), // RMSSD in milliseconds (higher = better recovery)
  hrvSdnn: decimal("hrv_sdnn"), // SDNN in milliseconds
  restingHeartRate: integer("resting_heart_rate"), // BPM
  
  // Sleep quality (critical for immune function)
  sleepDuration: decimal("sleep_duration"), // Hours
  deepSleepDuration: decimal("deep_sleep_duration"), // Hours
  remSleepDuration: decimal("rem_sleep_duration"), // Hours
  sleepQuality: integer("sleep_quality"), // 1-100 score
  sleepEfficiency: decimal("sleep_efficiency"), // Percentage
  
  // Activity and stress
  stepsCount: integer("steps_count"),
  activeMinutes: integer("active_minutes"),
  caloriesBurned: integer("calories_burned"),
  stressLevel: integer("stress_level"), // 1-100 scale
  
  // Temperature (early infection indicator)
  bodyTemperature: decimal("body_temperature"), // Celsius
  skinTemperature: decimal("skin_temperature"), // Celsius
  
  // Respiratory
  respiratoryRate: decimal("respiratory_rate"), // Breaths per minute
  oxygenSaturation: decimal("oxygen_saturation"), // SpO2 percentage
  
  // Recovery metrics
  recoveryScore: integer("recovery_score"), // 1-100 composite score
  readinessScore: integer("readiness_score"), // 1-100 readiness for activity
  
  // Metadata
  metadata: jsonb("metadata").$type<{
    deviceModel?: string;
    firmwareVersion?: string;
    confidenceScore?: number;
    isManualEntry?: boolean;
  }>(),
  
  createdAt: timestamp("created_at").defaultNow(),
});

export const insertImmuneBiomarkerSchema = createInsertSchema(immuneBiomarkers).omit({
  id: true,
  createdAt: true,
});

export type InsertImmuneBiomarker = z.infer<typeof insertImmuneBiomarkerSchema>;
export type ImmuneBiomarker = typeof immuneBiomarkers.$inferSelect;

// Immune Digital Twins - AI models predicting immune function status
export const immuneDigitalTwins = pgTable("immune_digital_twins", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  userId: varchar("user_id").notNull().references(() => users.id),
  
  // Prediction timestamp
  predictedAt: timestamp("predicted_at").notNull(),
  predictionWindow: varchar("prediction_window").notNull(), // 'current', '24h', '48h', '7d'
  
  // Immune function score (0-100, where 100 is optimal)
  immuneScore: integer("immune_score").notNull(),
  immuneScoreTrend: varchar("immune_score_trend"), // 'improving', 'stable', 'declining', 'critical'
  
  // Component scores
  recoveryCapacityScore: integer("recovery_capacity_score"), // How well recovering from stress
  infectionResistanceScore: integer("infection_resistance_score"), // Resistance to pathogens
  inflammationScore: integer("inflammation_score"), // Systemic inflammation level
  stressResponseScore: integer("stress_response_score"), // Stress adaptation
  
  // Risk assessment
  infectionRisk: varchar("infection_risk").notNull(), // 'low', 'moderate', 'high', 'critical'
  hospitalAdmissionRisk: varchar("hospitalization_risk"), // Probability of hospitalization
  
  // Contributing factors (for explainability)
  contributingFactors: jsonb("contributing_factors").$type<{
    factor: string;
    impact: 'positive' | 'negative';
    strength: number; // 1-10
  }[]>(),
  
  // Biomarker inputs used for prediction
  biomarkerIds: jsonb("biomarker_ids").$type<string[]>(), // References to immuneBiomarkers
  
  // Model information
  modelVersion: varchar("model_version").notNull(),
  confidenceScore: decimal("confidence_score"), // 0-1 model confidence
  
  // AI-generated insights
  aiInsights: text("ai_insights"), // Natural language explanation
  recommendations: jsonb("recommendations").$type<string[]>(), // Actionable recommendations
  
  createdAt: timestamp("created_at").defaultNow(),
});

export const insertImmuneDigitalTwinSchema = createInsertSchema(immuneDigitalTwins).omit({
  id: true,
  createdAt: true,
});

export type InsertImmuneDigitalTwin = z.infer<typeof insertImmuneDigitalTwinSchema>;
export type ImmuneDigitalTwin = typeof immuneDigitalTwins.$inferSelect;

// Environmental Risk Data - Real-time pathogen and environmental threats
export const environmentalRiskData = pgTable("environmental_risk_data", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  
  // Location data
  latitude: decimal("latitude").notNull(),
  longitude: decimal("longitude").notNull(),
  locationName: varchar("location_name"), // City, neighborhood
  zipCode: varchar("zip_code"),
  country: varchar("country").default("USA"),
  
  // Timestamp
  measuredAt: timestamp("measured_at").notNull(),
  
  // Air Quality Index (AQI)
  aqi: integer("aqi"), // 0-500 scale
  aqiCategory: varchar("aqi_category"), // 'good', 'moderate', 'unhealthy_sensitive', 'unhealthy', 'very_unhealthy', 'hazardous'
  pm25: decimal("pm25"), // PM2.5 particulate matter (µg/m³)
  pm10: decimal("pm10"), // PM10 particulate matter (µg/m³)
  ozone: decimal("ozone"), // O3 concentration
  no2: decimal("no2"), // Nitrogen dioxide
  so2: decimal("so2"), // Sulfur dioxide
  co: decimal("co"), // Carbon monoxide
  
  // Pollen and allergens
  pollenCount: integer("pollen_count"), // 0-12 scale
  pollenTypes: jsonb("pollen_types").$type<string[]>(), // ['tree', 'grass', 'weed']
  moldSporeCount: integer("mold_spore_count"),
  
  // Wastewater surveillance (pathogen detection)
  wastewaterViralLoad: decimal("wastewater_viral_load"), // RNA copies/L
  detectedPathogens: jsonb("detected_pathogens").$type<{
    pathogen: string; // 'covid-19', 'influenza', 'rsv', 'norovirus'
    concentration: number;
    trend: 'increasing' | 'stable' | 'decreasing';
  }[]>(),
  
  // Outbreak tracking
  localOutbreaks: jsonb("local_outbreaks").$type<{
    disease: string;
    caseCount: number;
    severity: 'low' | 'moderate' | 'high';
    radius: number; // miles
  }[]>(),
  
  // Weather conditions (affect immune function)
  temperature: decimal("temperature"), // Celsius
  humidity: decimal("humidity"), // Percentage
  uvIndex: integer("uv_index"), // 0-11+
  
  // Risk scores
  overallRiskScore: integer("overall_risk_score"), // 0-100 composite
  immunocompromisedRisk: varchar("immunocompromised_risk"), // 'low', 'moderate', 'high', 'critical'
  
  // Data sources
  dataSources: jsonb("data_sources").$type<{
    aqi?: string; // 'airnow', 'iqair', 'purple_air'
    wastewater?: string; // 'biobot', 'wastewater_scan', 'cdc'
    outbreak?: string; // 'cdc', 'who', 'local_health_dept'
  }>(),
  
  createdAt: timestamp("created_at").defaultNow(),
});

export const insertEnvironmentalRiskDataSchema = createInsertSchema(environmentalRiskData).omit({
  id: true,
  createdAt: true,
});

export type InsertEnvironmentalRiskData = z.infer<typeof insertEnvironmentalRiskDataSchema>;
export type EnvironmentalRiskData = typeof environmentalRiskData.$inferSelect;

// Risk Alerts - Real-time alerts for immune decline and environmental threats
export const riskAlerts = pgTable("risk_alerts", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  userId: varchar("user_id").notNull().references(() => users.id),
  
  // Alert type
  alertType: varchar("alert_type").notNull(), // 'immune_decline', 'environmental_risk', 'combined'
  severity: varchar("severity").notNull(), // 'low', 'moderate', 'high', 'critical'
  priority: integer("priority").notNull(), // 1-10 (10 = most urgent)
  
  // Alert details
  title: varchar("title").notNull(),
  message: text("message").notNull(),
  aiExplanation: text("ai_explanation"), // Agent Clona's empathetic explanation
  
  // Related data
  immuneDigitalTwinId: varchar("immune_digital_twin_id").references(() => immuneDigitalTwins.id),
  environmentalRiskDataId: varchar("environmental_risk_data_id").references(() => environmentalRiskData.id),
  
  // Recommendations
  recommendations: jsonb("recommendations").$type<{
    action: string;
    urgency: 'immediate' | 'today' | 'this_week';
    category: 'medical' | 'lifestyle' | 'environmental';
  }[]>(),
  
  // User interaction
  status: varchar("status").default("active"), // 'active', 'acknowledged', 'resolved', 'dismissed'
  acknowledgedAt: timestamp("acknowledged_at"),
  resolvedAt: timestamp("resolved_at"),
  
  // Notification tracking
  smsNotificationSent: boolean("sms_notification_sent").default(false),
  smsNotificationSentAt: timestamp("sms_notification_sent_at"),
  emailNotificationSent: boolean("email_notification_sent").default(false),
  emailNotificationSentAt: timestamp("email_notification_sent_at"),
  pushNotificationSent: boolean("push_notification_sent").default(false),
  pushNotificationSentAt: timestamp("push_notification_sent_at"),
  
  // Expiry
  expiresAt: timestamp("expires_at"),
  
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
});

export const insertRiskAlertSchema = createInsertSchema(riskAlerts).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export type InsertRiskAlert = z.infer<typeof insertRiskAlertSchema>;
export type RiskAlert = typeof riskAlerts.$inferSelect;

// Referral System (for 1-month free trial incentive)
export const referrals = pgTable("referrals", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  
  // Referrer (person who shares the link)
  referrerId: varchar("referrer_id").notNull().references(() => users.id),
  referrerType: varchar("referrer_type").notNull(), // 'patient' or 'doctor'
  
  // Referee (person who signs up via the link)
  refereeId: varchar("referee_id").references(() => users.id),
  refereeType: varchar("referee_type"), // 'patient' or 'doctor'
  refereeEmail: varchar("referee_email"), // Captured when they click the link
  
  // Referral code and tracking
  referralCode: varchar("referral_code").unique().notNull(), // Unique code like "REF-ABC123"
  referralLink: varchar("referral_link").notNull(), // Full URL with code
  
  // Status tracking
  status: varchar("status").default("pending"), // 'pending', 'signed_up', 'trial_activated', 'trial_completed', 'expired'
  clickedAt: timestamp("clicked_at"), // When referee clicked the link
  signedUpAt: timestamp("signed_up_at"), // When referee completed signup
  trialActivatedAt: timestamp("trial_activated_at"), // When both got their free month
  
  // Trial benefits
  referrerTrialExtended: boolean("referrer_trial_extended").default(false), // 1-month free trial given to referrer
  refereeTrialGranted: boolean("referee_trial_granted").default(false), // 1-month free trial given to referee
  
  // Tracking
  ipAddress: varchar("ip_address"), // For fraud detection
  userAgent: varchar("user_agent"),
  utmSource: varchar("utm_source"),
  utmMedium: varchar("utm_medium"),
  utmCampaign: varchar("utm_campaign"),
  
  // Expiry
  expiresAt: timestamp("expires_at"), // Referral link expiry (e.g., 90 days)
  
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
});

export const insertReferralSchema = createInsertSchema(referrals).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export type InsertReferral = z.infer<typeof insertReferralSchema>;
export type Referral = typeof referrals.$inferSelect;

// Two-Factor Authentication (2FA) Settings
export const twoFactorAuth = pgTable("two_factor_auth", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  userId: varchar("user_id").unique().notNull().references(() => users.id),
  
  // TOTP settings
  totpSecret: varchar("totp_secret").notNull(), // Base32 encoded secret for TOTP
  enabled: boolean("enabled").default(false),
  
  // Backup codes (hashed)
  backupCodes: jsonb("backup_codes").$type<string[]>(), // Array of hashed backup codes
  
  // Tracking
  enabledAt: timestamp("enabled_at"),
  lastUsedAt: timestamp("last_used_at"),
  
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
});

export const insertTwoFactorAuthSchema = createInsertSchema(twoFactorAuth).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export type InsertTwoFactorAuth = z.infer<typeof insertTwoFactorAuthSchema>;
export type TwoFactorAuth = typeof twoFactorAuth.$inferSelect;

// Admin ML Training Hub TOTP Access Control
export const adminTotpSecrets = pgTable("admin_totp_secrets", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  
  // Secret name/identifier (e.g., "ml-training-hub")
  secretName: varchar("secret_name").unique().notNull(),
  
  // TOTP settings
  totpSecret: varchar("totp_secret").notNull(),
  enabled: boolean("enabled").default(false),
  
  // Setup tracking
  setupCompletedAt: timestamp("setup_completed_at"),
  setupCompletedBy: varchar("setup_completed_by"),
  
  // Usage tracking
  lastVerifiedAt: timestamp("last_verified_at"),
  lastVerifiedBy: varchar("last_verified_by"),
  verificationCount: integer("verification_count").default(0),
  
  // Security
  failedAttempts: integer("failed_attempts").default(0),
  lockedUntil: timestamp("locked_until"),
  
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
});

export const insertAdminTotpSecretSchema = createInsertSchema(adminTotpSecrets).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export type InsertAdminTotpSecret = z.infer<typeof insertAdminTotpSecretSchema>;
export type AdminTotpSecret = typeof adminTotpSecrets.$inferSelect;

// Medical Documents (OCR extraction from uploaded files)
export const medicalDocuments = pgTable("medical_documents", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  userId: varchar("user_id").notNull().references(() => users.id),
  
  // File information
  fileName: varchar("file_name").notNull(),
  fileType: varchar("file_type").notNull(), // 'pdf', 'image', etc.
  fileSize: integer("file_size"), // in bytes
  fileUrl: text("file_url").notNull(), // Storage URL
  
  // OCR extracted data
  extractedText: text("extracted_text"), // Full text extraction
  extractedData: jsonb("extracted_data").$type<{
    patientName?: string;
    dateOfBirth?: string;
    diagnosis?: string[];
    medications?: string[];
    labResults?: { test: string; value: string; unit?: string }[];
    vitalSigns?: { type: string; value: string; unit?: string }[];
    allergies?: string[];
    procedures?: string[];
    notes?: string;
  }>(), // Structured extracted medical data
  
  // Metadata
  documentType: varchar("document_type"), // 'lab_report', 'prescription', 'imaging', 'discharge_summary', 'other'
  documentDate: timestamp("document_date"),
  processingStatus: varchar("processing_status").default("pending"), // 'pending', 'processing', 'completed', 'failed'
  errorMessage: text("error_message"),
  
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
});

export const insertMedicalDocumentSchema = createInsertSchema(medicalDocuments).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export type InsertMedicalDocument = z.infer<typeof insertMedicalDocumentSchema>;
export type MedicalDocument = typeof medicalDocuments.$inferSelect;

// Medical Histories (structured symptom data from OPQRST questioning)
export const medicalHistories = pgTable("medical_histories", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  sessionId: varchar("session_id").notNull().references(() => chatSessions.id),
  userId: varchar("user_id").notNull().references(() => users.id),
  
  // Chief complaint
  chiefComplaint: text("chief_complaint").notNull(),
  
  // OPQRST details
  onset: text("onset"), // When did it start, what were you doing
  provocation: text("provocation"), // What makes it better/worse
  quality: text("quality"), // Description of how it feels
  region: text("region"), // Where is it, does it radiate
  severity: integer("severity"), // Scale 1-10
  timing: text("timing"), // Duration, frequency, patterns
  
  // Additional history
  associatedSymptoms: jsonb("associated_symptoms").$type<string[]>(),
  pastMedicalHistory: text("past_medical_history"),
  currentMedications: jsonb("current_medications").$type<string[]>(),
  allergies: jsonb("allergies").$type<string[]>(),
  recentChanges: text("recent_changes"), // Travel, diet, stress, etc.
  impactOnLife: text("impact_on_life"), // Effect on daily activities
  
  // Summary
  historyComplete: boolean("history_complete").default(false),
  
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
});

export const insertMedicalHistorySchema = createInsertSchema(medicalHistories).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export type InsertMedicalHistory = z.infer<typeof insertMedicalHistorySchema>;
export type MedicalHistory = typeof medicalHistories.$inferSelect;

// Differential Diagnoses (AI-generated from medical history)
export const differentialDiagnoses = pgTable("differential_diagnoses", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  medicalHistoryId: varchar("medical_history_id").notNull().references(() => medicalHistories.id),
  sessionId: varchar("session_id").notNull().references(() => chatSessions.id),
  userId: varchar("user_id").notNull().references(() => users.id),
  
  // Differential diagnosis list (ordered by likelihood)
  diagnoses: jsonb("diagnoses").$type<Array<{
    condition: string;
    likelihood: 'high' | 'moderate' | 'low';
    reasoning: string;
    redFlags?: string[];
    recommendedActions?: string[];
  }>>(),
  
  // Summary and recommendations
  summary: text("summary"), // Patient-friendly explanation
  immediateActions: jsonb("immediate_actions").$type<string[]>(),
  followUpRecommendations: text("follow_up_recommendations"),
  urgencyLevel: varchar("urgency_level"), // 'emergency', 'urgent', 'routine', 'monitor'
  
  // AI metadata
  generatedBy: varchar("generated_by").default("agent_clona"), // Which AI generated this
  confidence: decimal("confidence", { precision: 3, scale: 2 }), // 0.00 to 1.00
  
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
});

export const insertDifferentialDiagnosisSchema = createInsertSchema(differentialDiagnoses).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export type InsertDifferentialDiagnosis = z.infer<typeof insertDifferentialDiagnosisSchema>;
export type DifferentialDiagnosis = typeof differentialDiagnoses.$inferSelect;

// User Settings (language, theme, voice, data permissions)
export const userSettings = pgTable("user_settings", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  userId: varchar("user_id").unique().notNull().references(() => users.id),
  
  // Language preferences
  language: varchar("language").default("en"), // 'en', 'es', 'fr', 'de', 'zh', 'ja', 'ar', 'hi', etc.
  autoDetectLanguage: boolean("auto_detect_language").default(true),
  
  // Theme preferences
  theme: varchar("theme").default("system"), // 'light', 'dark', 'system'
  
  // Voice preferences
  voiceEnabled: boolean("voice_enabled").default(false),
  voiceGender: varchar("voice_gender").default("female"), // 'male', 'female'
  voiceSpeed: decimal("voice_speed", { precision: 2, scale: 1 }).default("1.0"), // 0.5 to 2.0
  voiceLanguage: varchar("voice_language"), // Can differ from text language
  autoPlayVoice: boolean("auto_play_voice").default(false),
  
  // Data permissions
  shareDataWithResearch: boolean("share_data_with_research").default(false),
  shareDataWithDoctors: boolean("share_data_with_doctors").default(true),
  allowThirdPartyIntegrations: boolean("allow_third_party_integrations").default(false),
  allowAIAnalysis: boolean("allow_ai_analysis").default(true),
  
  // Notification preferences
  emailNotifications: boolean("email_notifications").default(true),
  pushNotifications: boolean("push_notifications").default(true),
  
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
});

export const insertUserSettingsSchema = createInsertSchema(userSettings).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export type InsertUserSettings = z.infer<typeof insertUserSettingsSchema>;
export type UserSettings = typeof userSettings.$inferSelect;

// Multi-Condition Correlation Engine
export const correlationPatterns = pgTable("correlation_patterns", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  userId: varchar("user_id").notNull().references(() => users.id),
  
  // Pattern metadata
  patternName: varchar("pattern_name").notNull(), // e.g., "High stress → Poor sleep → Immune decline"
  patternType: varchar("pattern_type").notNull(), // 'positive', 'negative', 'neutral'
  
  // Correlated factors
  factors: jsonb("factors").$type<Array<{
    type: string; // 'medication', 'environment', 'mood', 'sleep', 'biomarker', 'symptom'
    name: string;
    value?: any;
    timestamp?: string;
  }>>(),
  
  // Statistical analysis
  correlationStrength: decimal("correlation_strength", { precision: 3, scale: 2 }), // -1.00 to 1.00
  confidence: decimal("confidence", { precision: 3, scale: 2 }), // 0.00 to 1.00
  sampleSize: integer("sample_size"), // Number of data points analyzed
  
  // Time-based analysis
  timeWindow: varchar("time_window"), // e.g., "7 days", "30 days", "6 months"
  firstObserved: timestamp("first_observed"),
  lastObserved: timestamp("last_observed"),
  frequency: integer("frequency"), // How often this pattern occurs
  
  // AI-generated insights
  insight: text("insight"), // Human-readable explanation
  recommendation: text("recommendation"), // Actionable advice
  severity: varchar("severity"), // 'low', 'moderate', 'high', 'critical'
  
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
});

export const insertCorrelationPatternSchema = createInsertSchema(correlationPatterns).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export type InsertCorrelationPattern = z.infer<typeof insertCorrelationPatternSchema>;
export type CorrelationPattern = typeof correlationPatterns.$inferSelect;

// Genomic Risk Profiling
export const geneticVariants = pgTable("genetic_variants", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  userId: varchar("user_id").notNull().references(() => users.id),
  
  // Genetic information
  gene: varchar("gene").notNull(), // e.g., "CYP2D6", "CYP3A4", "SLCO1B1"
  variant: varchar("variant").notNull(), // e.g., "*1/*2", "rs4149056"
  genotype: varchar("genotype"), // e.g., "AA", "AG", "GG"
  phenotype: varchar("phenotype"), // e.g., "poor metabolizer", "normal metabolizer", "ultrarapid metabolizer"
  
  // Clinical significance
  clinicalSignificance: varchar("clinical_significance"), // 'pathogenic', 'likely_pathogenic', 'uncertain', 'likely_benign', 'benign'
  affectedDrugs: jsonb("affected_drugs").$type<string[]>(), // Drugs affected by this variant
  
  // Risk assessment
  riskLevel: varchar("risk_level"), // 'low', 'moderate', 'high', 'critical'
  recommendations: text("recommendations"),
  
  // Source information
  testingProvider: varchar("testing_provider"), // e.g., "23andMe", "AncestryDNA", "Hospital Lab"
  testDate: timestamp("test_date"),
  reportUrl: varchar("report_url"), // S3 URL for uploaded report
  
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
});

export const insertGeneticVariantSchema = createInsertSchema(geneticVariants).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export type InsertGeneticVariant = z.infer<typeof insertGeneticVariantSchema>;
export type GeneticVariant = typeof geneticVariants.$inferSelect;

export const pharmacogenomicReports = pgTable("pharmacogenomic_reports", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  userId: varchar("user_id").notNull().references(() => users.id),
  
  // Report metadata
  reportName: varchar("report_name").notNull(),
  testingProvider: varchar("testing_provider").notNull(),
  testDate: timestamp("test_date").notNull(),
  reportFileUrl: varchar("report_file_url"), // S3 URL for full report
  
  // Extracted insights
  processedVariants: integer("processed_variants"), // Count of variants analyzed
  highRiskDrugs: jsonb("high_risk_drugs").$type<string[]>(),
  recommendations: jsonb("recommendations").$type<Array<{
    medication: string;
    recommendation: string;
    reasoning: string;
  }>>(),
  
  // Processing status
  processingStatus: varchar("processing_status").default("pending"), // 'pending', 'processing', 'completed', 'error'
  processingError: text("processing_error"),
  
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
});

export const insertPharmacogenomicReportSchema = createInsertSchema(pharmacogenomicReports).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export type InsertPharmacogenomicReport = z.infer<typeof insertPharmacogenomicReportSchema>;
export type PharmacogenomicReport = typeof pharmacogenomicReports.$inferSelect;

// Clinical Trial Matchmaking
export const clinicalTrials = pgTable("clinical_trials", {
  id: varchar("id").primaryKey(), // NCT number from ClinicalTrials.gov
  
  // Basic trial information
  title: text("title").notNull(),
  officialTitle: text("official_title"),
  briefSummary: text("brief_summary"),
  detailedDescription: text("detailed_description"),
  
  // Trial status
  status: varchar("status").notNull(), // 'recruiting', 'active', 'completed', 'suspended', 'terminated'
  phase: varchar("phase"), // 'Phase 1', 'Phase 2', 'Phase 3', 'Phase 4', 'N/A'
  studyType: varchar("study_type"), // 'Interventional', 'Observational'
  
  // Conditions and interventions
  conditions: jsonb("conditions").$type<string[]>(),
  interventions: jsonb("interventions").$type<Array<{
    type: string;
    name: string;
  }>>(),
  
  // Eligibility
  eligibilityCriteria: text("eligibility_criteria"),
  minAge: integer("min_age"), // Age in years
  maxAge: integer("max_age"),
  gender: varchar("gender"), // 'All', 'Male', 'Female'
  healthyVolunteers: boolean("healthy_volunteers"),
  
  // Location and contact
  locations: jsonb("locations").$type<Array<{
    facility: string;
    city: string;
    state: string;
    country: string;
    zipCode?: string;
    status?: string;
    contacts?: Array<{
      name: string;
      phone?: string;
      email?: string;
    }>;
  }>>(),
  
  // Sponsor and dates
  sponsor: varchar("sponsor"),
  collaborators: jsonb("collaborators").$type<string[]>(),
  startDate: timestamp("start_date"),
  completionDate: timestamp("completion_date"),
  
  // URLs
  clinicalTrialsGovUrl: varchar("clinical_trials_gov_url"),
  
  // Metadata
  lastUpdated: timestamp("last_updated"),
  createdAt: timestamp("created_at").defaultNow(),
});

export const insertClinicalTrialSchema = createInsertSchema(clinicalTrials).omit({
  createdAt: true,
});

export type InsertClinicalTrial = z.infer<typeof insertClinicalTrialSchema>;
export type ClinicalTrial = typeof clinicalTrials.$inferSelect;

export const trialMatchScores = pgTable("trial_match_scores", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  userId: varchar("user_id").notNull().references(() => users.id),
  trialId: varchar("trial_id").notNull().references(() => clinicalTrials.id),
  
  // Match scoring
  overallScore: decimal("overall_score", { precision: 3, scale: 2 }), // 0.00 to 1.00
  conditionMatch: decimal("condition_match", { precision: 3, scale: 2 }),
  locationMatch: decimal("location_match", { precision: 3, scale: 2 }),
  eligibilityMatch: decimal("eligibility_match", { precision: 3, scale: 2 }),
  
  // Detailed analysis
  matchingCriteria: jsonb("matching_criteria").$type<Array<{
    criterion: string;
    matched: boolean;
    reasoning: string;
  }>>(),
  disqualifyingFactors: jsonb("disqualifying_factors").$type<string[]>(),
  
  // Distance calculation
  nearestLocationDistance: decimal("nearest_location_distance", { precision: 6, scale: 2 }), // miles
  nearestLocationCity: varchar("nearest_location_city"),
  nearestLocationState: varchar("nearest_location_state"),
  
  // AI-generated insights
  recommendation: varchar("recommendation"), // 'highly_recommended', 'recommended', 'consider', 'not_recommended'
  reasoning: text("reasoning"),
  
  // User actions
  status: varchar("status").default("new"), // 'new', 'interested', 'contacted', 'enrolled', 'declined'
  notes: text("notes"),
  
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
});

export const insertTrialMatchScoreSchema = createInsertSchema(trialMatchScores).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export type InsertTrialMatchScore = z.infer<typeof insertTrialMatchScoreSchema>;
export type TrialMatchScore = typeof trialMatchScores.$inferSelect;

// Predictive Hospitalization Prevention
export const deteriorationPredictions = pgTable("deterioration_predictions", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  userId: varchar("user_id").notNull().references(() => users.id),
  
  // Prediction metadata
  predictionDate: timestamp("prediction_date").notNull().defaultNow(),
  predictionHorizon: integer("prediction_horizon").default(72), // hours (72-hour default)
  
  // Risk assessment
  hospitalizationRisk: decimal("hospitalization_risk", { precision: 3, scale: 2 }).notNull(), // 0.00 to 1.00
  riskLevel: varchar("risk_level").notNull(), // 'low', 'moderate', 'high', 'critical'
  riskTrend: varchar("risk_trend"), // 'improving', 'stable', 'worsening', 'rapidly_worsening'
  
  // Contributing factors
  primaryRiskFactors: jsonb("primary_risk_factors").$type<Array<{
    factor: string;
    type: string; // 'biomarker', 'symptom', 'medication', 'environmental', 'behavioral'
    value: any;
    impact: 'high' | 'moderate' | 'low';
    trend: 'improving' | 'stable' | 'worsening';
  }>>(),
  
  // Biomarker-based indicators
  immuneScoreTrend: decimal("immune_score_trend", { precision: 5, scale: 2 }), // Change in immune score
  hrvTrend: decimal("hrv_trend", { precision: 5, scale: 2 }), // Change in HRV
  sleepQualityTrend: decimal("sleep_quality_trend", { precision: 5, scale: 2 }),
  restingHeartRateTrend: decimal("resting_heart_rate_trend", { precision: 5, scale: 2 }),
  temperatureTrend: decimal("temperature_trend", { precision: 5, scale: 2 }),
  
  // Environmental factors
  environmentalRiskScore: decimal("environmental_risk_score", { precision: 3, scale: 2 }), // 0.00 to 1.00
  airQualityImpact: varchar("air_quality_impact"), // 'none', 'low', 'moderate', 'high'
  pathogenExposureRisk: varchar("pathogen_exposure_risk"), // 'low', 'moderate', 'high'
  
  // AI-generated insights
  earlyWarningSignals: jsonb("early_warning_signals").$type<string[]>(),
  preventionRecommendations: jsonb("prevention_recommendations").$type<Array<{
    action: string;
    priority: 'immediate' | 'high' | 'moderate' | 'low';
    category: string; // 'medication', 'lifestyle', 'environmental', 'medical_attention'
    reasoning: string;
  }>>(),
  
  // Clinical guidance
  recommendedActions: jsonb("recommended_actions").$type<Array<{
    action: string;
    urgency: 'immediate' | 'within_24h' | 'within_72h' | 'routine';
    description: string;
  }>>(),
  shouldContactDoctor: boolean("should_contact_doctor").default(false),
  shouldSeekEmergencyCare: boolean("should_seek_emergency_care").default(false),
  
  // Model metadata
  modelVersion: varchar("model_version"),
  confidence: decimal("confidence", { precision: 3, scale: 2 }), // 0.00 to 1.00
  dataQualityScore: decimal("data_quality_score", { precision: 3, scale: 2 }), // 0.00 to 1.00
  
  // Outcome tracking
  actualOutcome: varchar("actual_outcome"), // 'prevented', 'hospitalized', 'false_alarm', null if too early
  outcomeNotes: text("outcome_notes"),
  outcomeDate: timestamp("outcome_date"),
  
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
});

export const insertDeteriorationPredictionSchema = createInsertSchema(deteriorationPredictions).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export type InsertDeteriorationPrediction = z.infer<typeof insertDeteriorationPredictionSchema>;
export type DeteriorationPrediction = typeof deteriorationPredictions.$inferSelect;

// ============== REINFORCEMENT LEARNING & RECOMMENDATION SYSTEM ==============

// User Preference Learning Profiles (for personalized AI responses)
export const userLearningProfiles = pgTable("user_learning_profiles", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  userId: varchar("user_id").notNull().references(() => users.id),
  agentType: varchar("agent_type").notNull(), // 'clona' or 'lysa'
  
  // Learned conversation preferences
  preferredTone: varchar("preferred_tone"), // 'warm', 'professional', 'empathetic', 'motivational'
  preferredResponseLength: varchar("preferred_response_length"), // 'brief', 'moderate', 'detailed'
  preferredTopics: jsonb("preferred_topics").$type<string[]>(), // Topics user engages with most
  avoidedTopics: jsonb("avoided_topics").$type<string[]>(), // Topics user avoids
  
  // Interaction patterns (learned from behavior)
  averageSessionDuration: integer("average_session_duration"), // minutes
  preferredTimeOfDay: varchar("preferred_time_of_day"), // 'morning', 'afternoon', 'evening', 'night'
  interactionFrequency: decimal("interaction_frequency", { precision: 4, scale: 2 }), // times per day
  
  // Health-specific preferences (Agent Clona)
  favoriteHealthActivities: jsonb("favorite_health_activities").$type<string[]>(),
  strugglingAreas: jsonb("struggling_areas").$type<string[]>(), // Areas needing more support
  motivationalStyle: varchar("motivational_style"), // 'encouragement', 'challenge', 'factual', 'celebration'
  
  // Clinical preferences (Assistant Lysa)
  preferredDiagnosticApproach: varchar("preferred_diagnostic_approach"), // 'algorithmic', 'evidence_based', 'differential'
  specialtyFocus: jsonb("specialty_focus").$type<string[]>(),
  researchInterests: jsonb("research_interests").$type<string[]>(),
  
  // ML model state
  modelVersion: varchar("model_version"),
  embeddingVector: jsonb("embedding_vector").$type<number[]>(), // User preference embedding (TensorFlow)
  lastTrainingDate: timestamp("last_training_date"),
  totalInteractions: integer("total_interactions").default(0),
  
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
});

export const insertUserLearningProfileSchema = createInsertSchema(userLearningProfiles).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export type InsertUserLearningProfile = z.infer<typeof insertUserLearningProfileSchema>;
export type UserLearningProfile = typeof userLearningProfiles.$inferSelect;

// Habit Tracking System
export const habits = pgTable("habits", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  userId: varchar("user_id").notNull().references(() => users.id),
  
  // Habit details
  name: varchar("name").notNull(), // "Morning meditation", "Evening walk", "Medication adherence"
  description: text("description"),
  category: varchar("category").notNull(), // 'health', 'medication', 'exercise', 'wellness', 'nutrition', 'sleep'
  
  // Tracking configuration
  frequency: varchar("frequency").notNull(), // 'daily', 'weekly', 'custom'
  targetDaysPerWeek: integer("target_days_per_week"),
  reminderEnabled: boolean("reminder_enabled").default(true),
  reminderTime: varchar("reminder_time"), // "08:00", "20:00"
  
  // Current status
  isActive: boolean("is_active").default(true),
  currentStreak: integer("current_streak").default(0), // days
  longestStreak: integer("longest_streak").default(0), // days
  totalCompletions: integer("total_completions").default(0),
  
  // AI recommendations
  recommendedBy: varchar("recommended_by"), // 'clona', 'lysa', 'user', 'ml_model'
  recommendationReason: text("recommendation_reason"),
  
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
});

export const insertHabitSchema = createInsertSchema(habits).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export type InsertHabit = z.infer<typeof insertHabitSchema>;
export type Habit = typeof habits.$inferSelect;

// Habit Completion Logs
export const habitCompletions = pgTable("habit_completions", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  habitId: varchar("habit_id").notNull().references(() => habits.id),
  userId: varchar("user_id").notNull().references(() => users.id),
  
  completionDate: timestamp("completion_date").notNull().defaultNow(),
  completed: boolean("completed").notNull().default(true),
  
  // Optional details
  notes: text("notes"),
  mood: varchar("mood"), // 'great', 'good', 'okay', 'struggling'
  difficultyLevel: integer("difficulty_level"), // 1-5 scale
  
  // AI feedback
  aiFeedback: text("ai_feedback"), // Encouraging message from Agent Clona
  
  createdAt: timestamp("created_at").defaultNow(),
});

export const insertHabitCompletionSchema = createInsertSchema(habitCompletions).omit({
  id: true,
  createdAt: true,
});

export type InsertHabitCompletion = z.infer<typeof insertHabitCompletionSchema>;
export type HabitCompletion = typeof habitCompletions.$inferSelect;

// ============================================
// COMPREHENSIVE HABIT TRACKER SYSTEM
// Features: Routines, Triggers, Quit Plans, Mood Tracking, 
// Social Accountability, CBT Flows, Gamification, AI Insights
// ============================================

// Habit Routines - Daily routine builder with time, location, micro-steps
export const habitRoutines = pgTable("habit_routines", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  habitId: varchar("habit_id").notNull().references(() => habits.id, { onDelete: "cascade" }),
  userId: varchar("user_id").notNull().references(() => users.id),
  
  // Time configuration
  scheduledTime: varchar("scheduled_time"), // "08:00", "14:30"
  duration: integer("duration"), // expected duration in minutes
  timeFlexibility: varchar("time_flexibility"), // 'strict', 'flexible', 'anytime'
  
  // Location
  location: varchar("location"), // "home", "gym", "office", "outdoors"
  locationDetails: text("location_details"),
  
  // Triggers and cues
  triggerCue: varchar("trigger_cue"), // What triggers starting this habit
  stackedAfter: varchar("stacked_after").references(() => habits.id), // Habit stacking
  
  // Context
  dayOfWeek: jsonb("day_of_week").$type<string[]>(), // ['monday', 'tuesday', ...]
  isWeekendOnly: boolean("is_weekend_only").default(false),
  
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
});

export const insertHabitRoutineSchema = createInsertSchema(habitRoutines).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export type InsertHabitRoutine = z.infer<typeof insertHabitRoutineSchema>;
export type HabitRoutine = typeof habitRoutines.$inferSelect;

// Habit Micro-Steps - Break down habits into small achievable steps
export const habitMicroSteps = pgTable("habit_micro_steps", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  habitId: varchar("habit_id").notNull().references(() => habits.id, { onDelete: "cascade" }),
  
  stepOrder: integer("step_order").notNull(),
  title: varchar("title").notNull(),
  description: text("description"),
  estimatedMinutes: integer("estimated_minutes"),
  
  // Progress tracking
  isRequired: boolean("is_required").default(true),
  completionCount: integer("completion_count").default(0),
  
  createdAt: timestamp("created_at").defaultNow(),
});

export const insertHabitMicroStepSchema = createInsertSchema(habitMicroSteps).omit({
  id: true,
  createdAt: true,
});

export type InsertHabitMicroStep = z.infer<typeof insertHabitMicroStepSchema>;
export type HabitMicroStep = typeof habitMicroSteps.$inferSelect;

// Habit Reminders - Smart notification scheduling
export const habitReminders = pgTable("habit_reminders", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  habitId: varchar("habit_id").notNull().references(() => habits.id, { onDelete: "cascade" }),
  userId: varchar("user_id").notNull().references(() => users.id),
  
  // Reminder configuration
  reminderType: varchar("reminder_type").notNull(), // 'push', 'email', 'in_app', 'sms'
  scheduledTime: varchar("scheduled_time").notNull(), // "08:00"
  message: text("message"),
  
  // Smart timing
  adaptiveEnabled: boolean("adaptive_enabled").default(true),
  learnedBestTime: varchar("learned_best_time"), // ML-learned optimal time
  
  // Status
  isActive: boolean("is_active").default(true),
  snoozeUntil: timestamp("snooze_until"),
  lastSentAt: timestamp("last_sent_at"),
  
  // Effectiveness tracking
  timesDelivered: integer("times_delivered").default(0),
  timesActedOn: integer("times_acted_on").default(0),
  
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
});

export const insertHabitReminderSchema = createInsertSchema(habitReminders).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export type InsertHabitReminder = z.infer<typeof insertHabitReminderSchema>;
export type HabitReminder = typeof habitReminders.$inferSelect;

// Habit AI Triggers - Pattern detection for skipped habits
export const habitAiTriggers = pgTable("habit_ai_triggers", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  userId: varchar("user_id").notNull().references(() => users.id),
  habitId: varchar("habit_id").references(() => habits.id),
  
  // Detected pattern
  triggerType: varchar("trigger_type").notNull(), // 'sleep', 'mood', 'time', 'weather', 'workload', 'social'
  pattern: text("pattern").notNull(), // "You skip morning habit when sleep < 6 hrs"
  
  // Pattern details
  correlatedFactor: varchar("correlated_factor"), // 'sleep_hours', 'mood_score', 'day_of_week'
  correlationStrength: decimal("correlation_strength", { precision: 3, scale: 2 }), // -1 to 1
  confidence: decimal("confidence", { precision: 3, scale: 2 }), // 0 to 1
  
  // Supporting data
  dataPoints: integer("data_points").default(0),
  samplePeriodDays: integer("sample_period_days"),
  
  // User interaction
  acknowledged: boolean("acknowledged").default(false),
  helpful: boolean("helpful"),
  userNotes: text("user_notes"),
  
  // Status
  isActive: boolean("is_active").default(true),
  lastDetectedAt: timestamp("last_detected_at"),
  
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
});

export const insertHabitAiTriggerSchema = createInsertSchema(habitAiTriggers).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export type InsertHabitAiTrigger = z.infer<typeof insertHabitAiTriggerSchema>;
export type HabitAiTrigger = typeof habitAiTriggers.$inferSelect;

// Quit Plans - Addiction-mode for bad habit control
export const habitQuitPlans = pgTable("habit_quit_plans", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  userId: varchar("user_id").notNull().references(() => users.id),
  
  // Target behavior to quit
  habitName: varchar("habit_name").notNull(), // "smoking", "excessive social media", etc.
  category: varchar("category"), // 'substance', 'behavioral', 'food', 'other'
  
  // Quit strategy
  quitMethod: varchar("quit_method"), // 'cold_turkey', 'gradual_reduction', 'replacement'
  targetQuitDate: timestamp("target_quit_date"),
  dailyLimit: integer("daily_limit"), // For gradual reduction
  
  // Harm reduction steps
  harmReductionSteps: jsonb("harm_reduction_steps").$type<Array<{
    step: string;
    order: number;
    completed: boolean;
  }>>(),
  
  // Motivation
  reasonsToQuit: jsonb("reasons_to_quit").$type<string[]>(),
  moneySavedPerDay: decimal("money_saved_per_day", { precision: 10, scale: 2 }),
  
  // Progress
  startDate: timestamp("start_date"),
  daysClean: integer("days_clean").default(0),
  longestStreak: integer("longest_streak").default(0),
  totalRelapses: integer("total_relapses").default(0),
  
  // Status
  status: varchar("status").default("active"), // 'active', 'paused', 'completed', 'abandoned'
  
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
});

export const insertHabitQuitPlanSchema = createInsertSchema(habitQuitPlans).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export type InsertHabitQuitPlan = z.infer<typeof insertHabitQuitPlanSchema>;
export type HabitQuitPlan = typeof habitQuitPlans.$inferSelect;

// Cravings Log - Track cravings and urges
export const habitCravingsLog = pgTable("habit_cravings_log", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  quitPlanId: varchar("quit_plan_id").notNull().references(() => habitQuitPlans.id, { onDelete: "cascade" }),
  userId: varchar("user_id").notNull().references(() => users.id),
  
  // Craving details
  intensity: integer("intensity").notNull(), // 1-10 scale
  duration: integer("duration"), // minutes
  trigger: varchar("trigger"), // 'stress', 'boredom', 'social', 'habit_cue', 'emotional'
  triggerDetails: text("trigger_details"),
  
  // Response
  copingStrategyUsed: varchar("coping_strategy_used"),
  overcame: boolean("overcame").default(false),
  notes: text("notes"),
  
  // Context
  location: varchar("location"),
  timeOfDay: varchar("time_of_day"),
  mood: varchar("mood"),
  
  occurredAt: timestamp("occurred_at").defaultNow(),
  createdAt: timestamp("created_at").defaultNow(),
});

export const insertHabitCravingsLogSchema = createInsertSchema(habitCravingsLog).omit({
  id: true,
  createdAt: true,
});

export type InsertHabitCravingsLog = z.infer<typeof insertHabitCravingsLogSchema>;
export type HabitCravingsLog = typeof habitCravingsLog.$inferSelect;

// Relapse Log - Track setbacks for quit plans
export const habitRelapseLog = pgTable("habit_relapse_log", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  quitPlanId: varchar("quit_plan_id").notNull().references(() => habitQuitPlans.id, { onDelete: "cascade" }),
  userId: varchar("user_id").notNull().references(() => users.id),
  
  // Relapse details
  severity: varchar("severity"), // 'minor_slip', 'moderate', 'full_relapse'
  quantity: varchar("quantity"), // "2 cigarettes", "1 hour social media"
  trigger: varchar("trigger"),
  emotionalState: varchar("emotional_state"),
  
  // Reflection
  whatHappened: text("what_happened"),
  whatLearnedFromThis: text("what_learned"),
  planToPrevent: text("plan_to_prevent"),
  
  // Reset tracking
  streakDaysLost: integer("streak_days_lost"),
  
  occurredAt: timestamp("occurred_at").defaultNow(),
  createdAt: timestamp("created_at").defaultNow(),
});

export const insertHabitRelapseLogSchema = createInsertSchema(habitRelapseLog).omit({
  id: true,
  createdAt: true,
});

export type InsertHabitRelapseLog = z.infer<typeof insertHabitRelapseLogSchema>;
export type HabitRelapseLog = typeof habitRelapseLog.$inferSelect;

// Habit Mood Entries - Emotion tracking with sentiment analysis
export const habitMoodEntries = pgTable("habit_mood_entries", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  userId: varchar("user_id").notNull().references(() => users.id),
  
  // Mood data
  moodScore: integer("mood_score").notNull(), // 1-10 scale
  moodLabel: varchar("mood_label"), // 'happy', 'anxious', 'angry', 'neutral', 'sad', 'excited'
  energyLevel: integer("energy_level"), // 1-10
  stressLevel: integer("stress_level"), // 1-10
  
  // Journal entry
  journalText: text("journal_text"),
  
  // AI-extracted insights
  sentimentScore: decimal("sentiment_score", { precision: 3, scale: 2 }), // -1 to 1
  extractedEmotions: jsonb("extracted_emotions").$type<string[]>(),
  extractedThemes: jsonb("extracted_themes").$type<string[]>(),
  
  // Context
  associatedHabitId: varchar("associated_habit_id").references(() => habits.id),
  contextTags: jsonb("context_tags").$type<string[]>(), // 'work', 'family', 'health', etc.
  
  recordedAt: timestamp("recorded_at").defaultNow(),
  createdAt: timestamp("created_at").defaultNow(),
});

export const insertHabitMoodEntrySchema = createInsertSchema(habitMoodEntries).omit({
  id: true,
  createdAt: true,
});

export type InsertHabitMoodEntry = z.infer<typeof insertHabitMoodEntrySchema>;
export type HabitMoodEntry = typeof habitMoodEntries.$inferSelect;

// Smart Journals - Weekly AI-summarized journal entries
export const habitJournals = pgTable("habit_journals", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  userId: varchar("user_id").notNull().references(() => users.id),
  
  // Entry details
  title: varchar("title"),
  content: text("content").notNull(),
  entryType: varchar("entry_type"), // 'daily', 'reflection', 'gratitude', 'goal_setting'
  
  // AI analysis
  aiSummary: text("ai_summary"),
  highlights: jsonb("highlights").$type<string[]>(), // 3 key highlights
  risks: jsonb("risks").$type<string[]>(), // 2 identified risks
  recommendations: jsonb("recommendations").$type<string[]>(), // 2 recommendations
  sentimentTrend: varchar("sentiment_trend"), // 'improving', 'stable', 'declining'
  
  // Tags and categories
  tags: jsonb("tags").$type<string[]>(),
  mood: varchar("mood"),
  
  // Weekly summary flag
  isWeeklySummary: boolean("is_weekly_summary").default(false),
  weekStartDate: timestamp("week_start_date"),
  
  recordedAt: timestamp("recorded_at").defaultNow(),
  createdAt: timestamp("created_at").defaultNow(),
});

export const insertHabitJournalSchema = createInsertSchema(habitJournals).omit({
  id: true,
  createdAt: true,
});

export type InsertHabitJournal = z.infer<typeof insertHabitJournalSchema>;
export type HabitJournal = typeof habitJournals.$inferSelect;

// Social Accountability - Buddy system
export const habitBuddies = pgTable("habit_buddies", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  userId: varchar("user_id").notNull().references(() => users.id),
  buddyUserId: varchar("buddy_user_id").notNull().references(() => users.id),
  
  // Relationship status
  status: varchar("status").default("pending"), // 'pending', 'active', 'blocked', 'removed'
  initiatedBy: varchar("initiated_by").references(() => users.id),
  
  // Visibility settings
  shareStreak: boolean("share_streak").default(true),
  shareCompletions: boolean("share_completions").default(true),
  shareMood: boolean("share_mood").default(false),
  
  // Shared habits
  sharedHabitIds: jsonb("shared_habit_ids").$type<string[]>(),
  
  // Engagement
  encouragementsSent: integer("encouragements_sent").default(0),
  encouragementsReceived: integer("encouragements_received").default(0),
  lastInteraction: timestamp("last_interaction"),
  
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
});

export const insertHabitBuddySchema = createInsertSchema(habitBuddies).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export type InsertHabitBuddy = z.infer<typeof insertHabitBuddySchema>;
export type HabitBuddy = typeof habitBuddies.$inferSelect;

// Encouragement Messages between buddies
export const habitEncouragements = pgTable("habit_encouragements", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  fromUserId: varchar("from_user_id").notNull().references(() => users.id),
  toUserId: varchar("to_user_id").notNull().references(() => users.id),
  
  // Message
  messageType: varchar("message_type"), // 'congrats', 'support', 'challenge', 'reminder'
  message: text("message").notNull(),
  prebuiltMessageId: varchar("prebuilt_message_id"), // For pre-built messages
  
  // Context
  relatedHabitId: varchar("related_habit_id").references(() => habits.id),
  relatedAchievement: varchar("related_achievement"),
  
  // Status
  read: boolean("read").default(false),
  readAt: timestamp("read_at"),
  
  createdAt: timestamp("created_at").defaultNow(),
});

export const insertHabitEncouragementSchema = createInsertSchema(habitEncouragements).omit({
  id: true,
  createdAt: true,
});

export type InsertHabitEncouragement = z.infer<typeof insertHabitEncouragementSchema>;
export type HabitEncouragement = typeof habitEncouragements.$inferSelect;

// CBT Sessions - Guided interventions
export const habitCbtSessions = pgTable("habit_cbt_sessions", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  userId: varchar("user_id").notNull().references(() => users.id),
  
  // Session type
  sessionType: varchar("session_type").notNull(), // 'urge_surfing', 'reframe_thought', 'grounding', 'breathing'
  title: varchar("title").notNull(),
  
  // Progress through steps
  currentStep: integer("current_step").default(1),
  totalSteps: integer("total_steps").notNull(),
  stepResponses: jsonb("step_responses").$type<Array<{
    step: number;
    prompt: string;
    response: string;
    timestamp: string;
  }>>(),
  
  // Completion
  completed: boolean("completed").default(false),
  completedAt: timestamp("completed_at"),
  
  // Effectiveness
  preSessionMood: integer("pre_session_mood"),
  postSessionMood: integer("post_session_mood"),
  helpfulRating: integer("helpful_rating"), // 1-5
  notes: text("notes"),
  
  // Related context
  relatedHabitId: varchar("related_habit_id").references(() => habits.id),
  relatedQuitPlanId: varchar("related_quit_plan_id").references(() => habitQuitPlans.id),
  
  startedAt: timestamp("started_at").defaultNow(),
  createdAt: timestamp("created_at").defaultNow(),
});

export const insertHabitCbtSessionSchema = createInsertSchema(habitCbtSessions).omit({
  id: true,
  createdAt: true,
});

export type InsertHabitCbtSession = z.infer<typeof insertHabitCbtSessionSchema>;
export type HabitCbtSession = typeof habitCbtSessions.$inferSelect;

// Visual Rewards - Gamification with growth visualization
export const habitRewards = pgTable("habit_rewards", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  userId: varchar("user_id").notNull().references(() => users.id),
  
  // Reward visualization
  rewardType: varchar("reward_type").default("sunflower"), // 'sunflower', 'tree', 'garden', 'galaxy'
  currentLevel: integer("current_level").default(1),
  growthStage: varchar("growth_stage").default("seed"), // 'seed', 'sprout', 'growing', 'blooming', 'flourishing'
  
  // Progress metrics
  totalPoints: integer("total_points").default(0),
  streakBonus: integer("streak_bonus").default(0),
  completionPoints: integer("completion_points").default(0),
  
  // Visual state
  visualState: jsonb("visual_state").$type<{
    petals?: number;
    leaves?: number;
    flowers?: number;
    height?: number;
    color?: string;
    accessories?: string[];
  }>(),
  
  // Achievements unlocked
  unlockedBadges: jsonb("unlocked_badges").$type<string[]>(),
  unlockedThemes: jsonb("unlocked_themes").$type<string[]>(),
  
  // Stats
  daysActive: integer("days_active").default(0),
  perfectDays: integer("perfect_days").default(0),
  
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
});

export const insertHabitRewardSchema = createInsertSchema(habitRewards).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export type InsertHabitReward = z.infer<typeof insertHabitRewardSchema>;
export type HabitReward = typeof habitRewards.$inferSelect;

// AI Coach Conversations - Chat history with habit coach
export const habitCoachChats = pgTable("habit_coach_chats", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  userId: varchar("user_id").notNull().references(() => users.id),
  
  // Conversation
  sessionId: varchar("session_id").notNull(),
  role: varchar("role").notNull(), // 'user', 'assistant'
  content: text("content").notNull(),
  
  // Context
  relatedHabitId: varchar("related_habit_id").references(() => habits.id),
  relatedQuitPlanId: varchar("related_quit_plan_id").references(() => habitQuitPlans.id),
  
  // AI metadata
  coachPersonality: varchar("coach_personality"), // 'supportive', 'challenging', 'analytical'
  responseType: varchar("response_type"), // 'encouragement', 'tip', 'cbt_technique', 'reflection'
  
  createdAt: timestamp("created_at").defaultNow(),
});

export const insertHabitCoachChatSchema = createInsertSchema(habitCoachChats).omit({
  id: true,
  createdAt: true,
});

export type InsertHabitCoachChat = z.infer<typeof insertHabitCoachChatSchema>;
export type HabitCoachChat = typeof habitCoachChats.$inferSelect;

// Risk Alerts - Preventive alerts for high-risk days
export const habitRiskAlerts = pgTable("habit_risk_alerts", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  userId: varchar("user_id").notNull().references(() => users.id),
  
  // Alert details
  alertType: varchar("alert_type").notNull(), // 'high_risk_day', 'streak_at_risk', 'pattern_detected', 'relapse_warning'
  severity: varchar("severity").notNull(), // 'low', 'medium', 'high', 'critical'
  title: varchar("title").notNull(),
  message: text("message").notNull(),
  
  // Risk factors
  riskScore: decimal("risk_score", { precision: 3, scale: 2 }), // 0-1
  contributingFactors: jsonb("contributing_factors").$type<Array<{
    factor: string;
    weight: number;
    value: string;
  }>>(),
  
  // Related context
  relatedHabitIds: jsonb("related_habit_ids").$type<string[]>(),
  relatedQuitPlanId: varchar("related_quit_plan_id").references(() => habitQuitPlans.id),
  
  // Recommended actions
  suggestedActions: jsonb("suggested_actions").$type<string[]>(),
  
  // Status
  status: varchar("status").default("active"), // 'active', 'acknowledged', 'dismissed', 'resolved'
  acknowledgedAt: timestamp("acknowledged_at"),
  
  // Timing
  predictedFor: timestamp("predicted_for"), // When the risk is predicted for
  expiresAt: timestamp("expires_at"),
  
  createdAt: timestamp("created_at").defaultNow(),
});

export const insertHabitRiskAlertSchema = createInsertSchema(habitRiskAlerts).omit({
  id: true,
  createdAt: true,
});

export type InsertHabitRiskAlert = z.infer<typeof insertHabitRiskAlertSchema>;
export type HabitRiskAlert = typeof habitRiskAlerts.$inferSelect;

// Dynamic AI Recommendations for habits
export const habitAiRecommendations = pgTable("habit_ai_recommendations", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  userId: varchar("user_id").notNull().references(() => users.id),
  habitId: varchar("habit_id").references(() => habits.id),
  
  // Recommendation type
  recommendationType: varchar("recommendation_type").notNull(), // 'difficulty_adjustment', 'time_change', 'micro_step', 'pause', 'celebrate'
  title: varchar("title").notNull(),
  description: text("description").notNull(),
  
  // Analysis basis
  basedOnCompletionRate: decimal("based_on_completion_rate", { precision: 3, scale: 2 }),
  basedOnStreak: integer("based_on_streak"),
  basedOnMoodTrend: varchar("based_on_mood_trend"),
  
  // Specific suggestion
  suggestedChange: jsonb("suggested_change").$type<{
    type: string;
    from?: string | number;
    to?: string | number;
    reason: string;
  }>(),
  
  // Confidence and priority
  confidence: decimal("confidence", { precision: 3, scale: 2 }),
  priority: varchar("priority"), // 'high', 'medium', 'low'
  
  // Status
  status: varchar("status").default("pending"), // 'pending', 'accepted', 'declined', 'expired'
  userResponse: text("user_response"),
  
  expiresAt: timestamp("expires_at"),
  createdAt: timestamp("created_at").defaultNow(),
});

export const insertHabitAiRecommendationSchema = createInsertSchema(habitAiRecommendations).omit({
  id: true,
  createdAt: true,
});

export type InsertHabitAiRecommendation = z.infer<typeof insertHabitAiRecommendationSchema>;
export type HabitAiRecommendation = typeof habitAiRecommendations.$inferSelect;

// ============================================
// END COMPREHENSIVE HABIT TRACKER SYSTEM
// ============================================

// Milestones & Achievements (Positive Reinforcement)
export const milestones = pgTable("milestones", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  userId: varchar("user_id").notNull().references(() => users.id),
  
  // Milestone details
  type: varchar("type").notNull(), // 'streak', 'total_completions', 'health_goal', 'engagement', 'wellness'
  name: varchar("name").notNull(), // "7-Day Streak", "100 Days Strong", "First Month Complete"
  description: text("description"),
  icon: varchar("icon"), // emoji or icon name
  
  // Achievement criteria
  category: varchar("category"), // 'habit', 'health', 'engagement', 'learning', 'doctor_wellness'
  targetValue: integer("target_value"),
  currentValue: integer("current_value").default(0),
  
  // Status
  achieved: boolean("achieved").default(false),
  achievedAt: timestamp("achieved_at"),
  celebrated: boolean("celebrated").default(false), // Whether user saw celebration
  
  // Rewards
  rewardPoints: integer("reward_points").default(0),
  rewardMessage: text("reward_message"),
  
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
});

export const insertMilestoneSchema = createInsertSchema(milestones).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export type InsertMilestone = z.infer<typeof insertMilestoneSchema>;
export type Milestone = typeof milestones.$inferSelect;

// ML-Generated Recommendations
export const mlRecommendations = pgTable("ml_recommendations", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  userId: varchar("user_id").notNull().references(() => users.id),
  agentType: varchar("agent_type").notNull(), // 'clona' or 'lysa'
  
  // Recommendation details
  type: varchar("type").notNull(), // 'habit', 'activity', 'health_tip', 'diagnostic', 'treatment', 'research'
  category: varchar("category"), // 'wellness', 'nutrition', 'exercise', 'medication', 'clinical', 'research'
  title: varchar("title").notNull(),
  description: text("description").notNull(),
  
  // ML model metadata
  modelVersion: varchar("model_version"),
  confidenceScore: decimal("confidence_score", { precision: 3, scale: 2 }), // 0.00 to 1.00
  personalizationScore: decimal("personalization_score", { precision: 3, scale: 2 }), // How personalized (0-1)
  
  // Recommendation reasoning
  reasoning: text("reasoning"), // Why this was recommended
  basedOnFactors: jsonb("based_on_factors").$type<Array<{
    factor: string;
    weight: number;
    source: string; // 'habit_history', 'conversation_data', 'health_metrics', 'research_interest'
  }>>(),
  
  // User interaction
  status: varchar("status").default("pending"), // 'pending', 'accepted', 'declined', 'completed', 'dismissed'
  userFeedback: varchar("user_feedback"), // 'helpful', 'not_helpful', 'irrelevant'
  userNotes: text("user_notes"),
  
  // Reinforcement learning reward signal
  rewardValue: decimal("reward_value", { precision: 5, scale: 2 }), // Calculated reward (positive/negative)
  
  // Priority and timing
  priority: varchar("priority"), // 'high', 'medium', 'low'
  expiresAt: timestamp("expires_at"),
  
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
});

export const insertMLRecommendationSchema = createInsertSchema(mlRecommendations).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export type InsertMLRecommendation = z.infer<typeof insertMLRecommendationSchema>;
export type MLRecommendation = typeof mlRecommendations.$inferSelect;

// Reinforcement Learning Rewards & Feedback
export const rlRewards = pgTable("rl_rewards", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  userId: varchar("user_id").notNull().references(() => users.id),
  agentType: varchar("agent_type").notNull(), // 'clona' or 'lysa'
  
  // State-action-reward data for RL
  state: jsonb("state").$type<{
    userContext: Record<string, any>;
    conversationContext: string[];
    healthMetrics: Record<string, any>;
    recentActions: string[];
  }>(),
  
  action: jsonb("action").$type<{
    type: string;
    content: string;
    parameters: Record<string, any>;
  }>(),
  
  reward: decimal("reward", { precision: 5, scale: 2 }).notNull(), // Positive or negative reward
  
  // Reward calculation factors
  rewardType: varchar("reward_type"), // 'engagement', 'completion', 'satisfaction', 'health_outcome'
  rewardFactors: jsonb("reward_factors").$type<Array<{
    factor: string;
    value: number;
    weight: number;
  }>>(),
  
  // Learning metadata
  episodeId: varchar("episode_id"), // Groups related interactions
  stepNumber: integer("step_number"), // Step in episode
  
  // Model update flag
  usedForTraining: boolean("used_for_training").default(false),
  trainingBatchId: varchar("training_batch_id"),
  
  createdAt: timestamp("created_at").defaultNow(),
});

export const insertRLRewardSchema = createInsertSchema(rlRewards).omit({
  id: true,
  createdAt: true,
});

export type InsertRLReward = z.infer<typeof insertRLRewardSchema>;
export type RLReward = typeof rlRewards.$inferSelect;

// Daily Engagement Tracking
export const dailyEngagement = pgTable("daily_engagement", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  userId: varchar("user_id").notNull().references(() => users.id),
  date: timestamp("date").notNull(),
  
  // Agent Clona engagement (patients)
  clonaInteractions: integer("clona_interactions").default(0),
  clonaSessionDuration: integer("clona_session_duration").default(0), // seconds
  clonaTopics: jsonb("clona_topics").$type<string[]>(),
  clonaSentiment: decimal("clona_sentiment", { precision: 3, scale: 2 }), // -1.00 to 1.00
  
  // Assistant Lysa engagement (doctors)
  lysaInteractions: integer("lysa_interactions").default(0),
  lysaSessionDuration: integer("lysa_session_duration").default(0), // seconds
  lysaTopics: jsonb("lysa_topics").$type<string[]>(),
  
  // Habit completions
  habitsCompleted: integer("habits_completed").default(0),
  habitsSkipped: integer("habits_skipped").default(0),
  
  // Health metrics logged
  healthMetricsLogged: integer("health_metrics_logged").default(0),
  
  // Overall engagement score (0-100)
  engagementScore: integer("engagement_score"),
  
  // Streak tracking
  isStreakDay: boolean("is_streak_day").default(false),
  streakCount: integer("streak_count").default(0),
  
  createdAt: timestamp("created_at").defaultNow(),
});

export const insertDailyEngagementSchema = createInsertSchema(dailyEngagement).omit({
  id: true,
  createdAt: true,
});

export type InsertDailyEngagement = z.infer<typeof insertDailyEngagementSchema>;
export type DailyEngagement = typeof dailyEngagement.$inferSelect;

// Doctor Wellness Tracking (for Assistant Lysa)
export const doctorWellness = pgTable("doctor_wellness", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  doctorId: varchar("doctor_id").notNull().references(() => users.id),
  
  date: timestamp("date").notNull().defaultNow(),
  
  // Wellness activities
  meditationMinutes: integer("meditation_minutes").default(0),
  exerciseMinutes: integer("exercise_minutes").default(0),
  sleepHours: decimal("sleep_hours", { precision: 3, scale: 1 }),
  
  // Stress & burnout tracking
  stressLevel: integer("stress_level"), // 1-10 scale
  burnoutScore: integer("burnout_score"), // 1-100
  workloadLevel: varchar("workload_level"), // 'light', 'moderate', 'heavy', 'overwhelming'
  
  // Professional development
  researchTimeMinutes: integer("research_time_minutes").default(0),
  learningActivities: jsonb("learning_activities").$type<string[]>(),
  
  // Patient care quality indicators
  patientsSeenToday: integer("patients_seen_today").default(0),
  averageConsultationTime: integer("average_consultation_time"), // minutes
  
  // Mood tracking
  mood: varchar("mood"), // 'excellent', 'good', 'okay', 'struggling', 'burnout'
  energyLevel: integer("energy_level"), // 1-10
  
  // AI insights
  lysaRecommendations: jsonb("lysa_recommendations").$type<string[]>(),
  wellnessScore: integer("wellness_score"), // 0-100 (AI-calculated)
  
  notes: text("notes"),
  
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
});

export const insertDoctorWellnessSchema = createInsertSchema(doctorWellness).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export type InsertDoctorWellness = z.infer<typeof insertDoctorWellnessSchema>;
export type DoctorWellness = typeof doctorWellness.$inferSelect;

// ============================================================================
// RECEPTIONIST & ASSISTANT LYSA FEATURES
// ============================================================================

// Triage assessment structure for symptom urgency evaluation
export type TriageAssessment = {
  urgencyScore: number;
  recommendedTimeframe: string;
  redFlags: string[];
  confidence: number;
  assessedAt: string;
  assessedBy: 'ai' | 'rule-based' | 'doctor';
};

// Appointments - Schedule and manage patient appointments
export const appointments = pgTable("appointments", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  
  // Participants
  doctorId: varchar("doctor_id").notNull().references(() => users.id),
  patientId: varchar("patient_id").references(() => users.id), // Nullable for external patients
  patientName: varchar("patient_name"), // For external patients not in system
  patientEmail: varchar("patient_email"),
  patientPhone: varchar("patient_phone"),
  
  // Appointment details
  title: varchar("title").notNull(),
  description: text("description"),
  appointmentType: varchar("appointment_type").notNull(), // 'consultation', 'followup', 'emergency', 'virtual', 'in-person'
  
  // Scheduling
  startTime: timestamp("start_time").notNull(),
  endTime: timestamp("end_time").notNull(),
  duration: integer("duration").notNull(), // minutes
  
  // Location (for in-person appointments)
  location: varchar("location"),
  roomNumber: varchar("room_number"),
  
  // Virtual meeting details
  meetingLink: varchar("meeting_link"),
  meetingPlatform: varchar("meeting_platform"), // 'zoom', 'meet', 'teams'
  
  // Status tracking
  status: varchar("status").notNull().default("scheduled"), // 'scheduled', 'confirmed', 'checked-in', 'in-progress', 'completed', 'cancelled', 'no-show', 'rescheduled'
  confirmationStatus: varchar("confirmation_status").default("pending"), // 'pending', 'confirmed', 'declined'
  confirmedAt: timestamp("confirmed_at"),
  
  // Cancellation/rescheduling
  cancelledBy: varchar("cancelled_by"), // userId who cancelled
  cancellationReason: text("cancellation_reason"),
  cancelledAt: timestamp("cancelled_at"),
  rescheduledFrom: varchar("rescheduled_from"), // original appointment ID
  
  // AI booking (if booked through Assistant Lysa)
  bookedByAI: boolean("booked_by_ai").default(false),
  aiBookingContext: jsonb("ai_booking_context").$type<{ intent: string; confidence: number; extractedInfo: any }>(),
  
  // Booking method tracking - how the appointment was booked
  bookedByMethod: varchar("booked_by_method").default("online"), // 'whatsapp', 'email', 'phone', 'online', 'walk-in'
  
  // Symptom triage (AI-powered urgency assessment)
  symptoms: text("symptoms"),
  urgencyLevel: varchar("urgency_level").default("routine"), // 'emergency', 'urgent', 'routine', 'non-urgent'
  triageAssessment: jsonb("triage_assessment").$type<TriageAssessment>(),
  triageAssessedAt: timestamp("triage_assessed_at"),
  clinicianOverride: boolean("clinician_override").default(false),
  clinicianOverrideReason: text("clinician_override_reason"),
  clinicianOverrideBy: varchar("clinician_override_by"),
  
  // Notes
  doctorNotes: text("doctor_notes"),
  patientNotes: text("patient_notes"),
  
  // Google Calendar integration
  googleCalendarEventId: varchar("google_calendar_event_id"),
  
  // Reminders sent
  remindersSent: jsonb("reminders_sent").$type<Array<{ type: string; sentAt: string }>>(),
  
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
}, (table) => ({
  doctorTimeIdx: index("appointments_doctor_time_idx").on(table.doctorId, table.startTime),
  patientTimeIdx: index("appointments_patient_time_idx").on(table.patientId, table.startTime),
  statusIdx: index("appointments_status_idx").on(table.status),
  urgencyQueueIdx: index("appointments_urgency_queue_idx").on(table.doctorId, table.urgencyLevel, table.startTime),
}));

export const insertAppointmentSchema = createInsertSchema(appointments).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export type InsertAppointment = z.infer<typeof insertAppointmentSchema>;
export type Appointment = typeof appointments.$inferSelect;

// Appointment Triage Logs - Audit trail for symptom triage assessments
export const appointmentTriageLogs = pgTable("appointment_triage_logs", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  appointmentId: varchar("appointment_id").references(() => appointments.id),
  patientId: varchar("patient_id").notNull().references(() => users.id),
  
  // Input
  symptoms: text("symptoms").notNull(),
  patientSelfAssessment: varchar("patient_self_assessment"), // Patient's own urgency estimate
  
  // AI Assessment
  urgencyLevel: varchar("urgency_level").notNull(), // 'emergency', 'urgent', 'routine', 'followup'
  urgencyScore: integer("urgency_score").notNull(), // 0-100
  recommendedTimeframe: varchar("recommended_timeframe").notNull(),
  redFlags: text("red_flags").array(),
  confidence: decimal("confidence", { precision: 3, scale: 2 }).notNull(), // 0.00-1.00
  assessmentMethod: varchar("assessment_method").notNull(), // 'ai', 'rule-based', 'hybrid'
  
  // Clinician Review
  clinicianReviewed: boolean("clinician_reviewed").default(false),
  clinicianAgreed: boolean("clinician_agreed"),
  clinicianOverrideLevel: varchar("clinician_override_level"),
  clinicianOverrideReason: text("clinician_override_reason"),
  reviewedBy: varchar("reviewed_by").references(() => users.id),
  reviewedAt: timestamp("reviewed_at"),
  
  // Risk Alert Integration
  riskAlertCreated: boolean("risk_alert_created").default(false),
  riskAlertId: varchar("risk_alert_id"),
  
  // Metadata
  modelVersion: varchar("model_version"), // e.g., "gpt-4o-2024-11-20"
  processingTimeMs: integer("processing_time_ms"),
  
  createdAt: timestamp("created_at").defaultNow(),
}, (table) => ({
  patientIdx: index("triage_logs_patient_idx").on(table.patientId, table.createdAt),
  urgencyIdx: index("triage_logs_urgency_idx").on(table.urgencyLevel, table.createdAt),
}));

export const insertAppointmentTriageLogSchema = createInsertSchema(appointmentTriageLogs).omit({
  id: true,
  createdAt: true,
});

export type InsertAppointmentTriageLog = z.infer<typeof insertAppointmentTriageLogSchema>;
export type AppointmentTriageLog = typeof appointmentTriageLogs.$inferSelect;

// Doctor Availability - Define doctor's working hours and availability
export const doctorAvailability = pgTable("doctor_availability", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  doctorId: varchar("doctor_id").notNull().references(() => users.id),
  
  // Recurring schedule
  dayOfWeek: integer("day_of_week").notNull(), // 0=Sunday, 1=Monday, ..., 6=Saturday
  startTime: varchar("start_time").notNull(), // "09:00"
  endTime: varchar("end_time").notNull(), // "17:00"
  
  // Override specific dates
  specificDate: timestamp("specific_date"), // For one-time availability changes
  isAvailable: boolean("is_available").default(true), // false for blocked time
  
  // Break times
  breakStart: varchar("break_start"), // "12:00"
  breakEnd: varchar("break_end"), // "13:00"
  
  // Slot configuration
  slotDuration: integer("slot_duration").default(30), // minutes per appointment
  bufferBetweenSlots: integer("buffer_between_slots").default(5), // minutes
  maxSlotsPerDay: integer("max_slots_per_day"),
  
  // Appointment types allowed during this time
  allowedTypes: jsonb("allowed_types").$type<string[]>(), // ['consultation', 'followup']
  
  // Reason for unavailability
  reason: text("reason"), // "Vacation", "Conference", etc.
  
  // Recurring pattern
  isRecurring: boolean("is_recurring").default(true),
  validFrom: timestamp("valid_from").notNull(),
  validUntil: timestamp("valid_until"),
  
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
}, (table) => ({
  doctorDayIdx: index("doctor_availability_doctor_day_idx").on(table.doctorId, table.dayOfWeek),
  dateIdx: index("doctor_availability_date_idx").on(table.specificDate),
}));

export const insertDoctorAvailabilitySchema = createInsertSchema(doctorAvailability).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export type InsertDoctorAvailability = z.infer<typeof insertDoctorAvailabilitySchema>;
export type DoctorAvailability = typeof doctorAvailability.$inferSelect;

// Email Threads - Organize email conversations
export const emailThreads = pgTable("email_threads", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  
  // Doctor who owns this thread
  doctorId: varchar("doctor_id").notNull().references(() => users.id),
  
  // Thread participants
  patientId: varchar("patient_id").references(() => users.id), // Nullable for external contacts
  externalEmail: varchar("external_email"), // For emails not from registered patients
  
  // Thread metadata
  subject: varchar("subject").notNull(),
  snippet: text("snippet"), // Preview of first message
  
  // Categorization (AI-powered)
  category: varchar("category"), // 'appointment', 'inquiry', 'followup', 'urgent', 'administrative'
  priority: varchar("priority").default("normal"), // 'low', 'normal', 'high', 'urgent'
  
  // Status
  status: varchar("status").default("active"), // 'active', 'archived', 'spam', 'trash'
  isRead: boolean("is_read").default(false),
  isStarred: boolean("is_starred").default(false),
  
  // AI assistance
  aiSuggestedReply: text("ai_suggested_reply"),
  aiCategory: varchar("ai_category"), // AI's categorization
  aiUrgency: varchar("ai_urgency"), // AI-detected urgency level
  aiExtractedInfo: jsonb("ai_extracted_info").$type<{ action: string; date?: string; appointment?: any }>(),
  
  // Gmail integration
  gmailThreadId: varchar("gmail_thread_id"),
  
  // Tracking
  lastMessageAt: timestamp("last_message_at"),
  messageCount: integer("message_count").default(0),
  
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
}, (table) => ({
  doctorIdx: index("email_threads_doctor_idx").on(table.doctorId),
  patientIdx: index("email_threads_patient_idx").on(table.patientId),
  statusIdx: index("email_threads_status_idx").on(table.status),
  categoryIdx: index("email_threads_category_idx").on(table.category),
}));

export const insertEmailThreadSchema = createInsertSchema(emailThreads).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export type InsertEmailThread = z.infer<typeof insertEmailThreadSchema>;
export type EmailThread = typeof emailThreads.$inferSelect;

// Email Messages - Individual emails within threads
export const emailMessages = pgTable("email_messages", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  threadId: varchar("thread_id").notNull().references(() => emailThreads.id, { onDelete: "cascade" }),
  
  // Sender/recipient
  fromEmail: varchar("from_email").notNull(),
  fromName: varchar("from_name"),
  toEmail: jsonb("to_email").$type<string[]>().notNull(),
  ccEmail: jsonb("cc_email").$type<string[]>(),
  bccEmail: jsonb("bcc_email").$type<string[]>(),
  
  // Content
  subject: varchar("subject"),
  body: text("body").notNull(),
  bodyHtml: text("body_html"),
  
  // Metadata
  isFromDoctor: boolean("is_from_doctor").default(false),
  isDraft: boolean("is_draft").default(false),
  isSent: boolean("is_sent").default(false),
  
  // AI generation
  generatedByAI: boolean("generated_by_ai").default(false),
  aiPrompt: text("ai_prompt"), // Original prompt if AI-generated
  aiTone: varchar("ai_tone"), // 'professional', 'friendly', 'urgent'
  
  // Attachments
  attachments: jsonb("attachments").$type<Array<{ name: string; url: string; size: number; type: string }>>(),
  
  // Gmail integration
  gmailMessageId: varchar("gmail_message_id"),
  
  // Delivery tracking
  sentAt: timestamp("sent_at"),
  deliveredAt: timestamp("delivered_at"),
  readAt: timestamp("read_at"),
  
  createdAt: timestamp("created_at").defaultNow(),
}, (table) => ({
  threadIdx: index("email_messages_thread_idx").on(table.threadId),
  sentIdx: index("email_messages_sent_idx").on(table.sentAt),
}));

export const insertEmailMessageSchema = createInsertSchema(emailMessages).omit({
  id: true,
  createdAt: true,
});

export type InsertEmailMessage = z.infer<typeof insertEmailMessageSchema>;
export type EmailMessage = typeof emailMessages.$inferSelect;

// Call Logs - Track phone calls and voicemails
export const callLogs = pgTable("call_logs", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  
  // Doctor who owns this call
  doctorId: varchar("doctor_id").notNull().references(() => users.id),
  
  // Caller information
  patientId: varchar("patient_id").references(() => users.id), // Nullable for unknown callers
  callerPhone: varchar("caller_phone").notNull(),
  callerName: varchar("caller_name"),
  
  // Call details
  direction: varchar("direction").notNull(), // 'inbound', 'outbound'
  callType: varchar("call_type").notNull(), // 'appointment', 'inquiry', 'emergency', 'followup', 'voicemail'
  
  // Timing
  startTime: timestamp("start_time").notNull(),
  endTime: timestamp("end_time"),
  duration: integer("duration"), // seconds
  
  // Status
  status: varchar("status").notNull(), // 'answered', 'missed', 'busy', 'failed', 'voicemail'
  isCallback: boolean("is_callback").default(false),
  callbackScheduledFor: timestamp("callback_scheduled_for"),
  
  // Twilio integration
  twilioCallSid: varchar("twilio_call_sid"),
  recordingUrl: varchar("recording_url"),
  recordingDuration: integer("recording_duration"), // seconds
  
  // Voicemail & transcription
  voicemailUrl: varchar("voicemail_url"),
  transcription: text("transcription"),
  transcriptionConfidence: decimal("transcription_confidence", { precision: 3, scale: 2 }),
  
  // AI analysis
  aiSummary: text("ai_summary"),
  aiIntent: varchar("ai_intent"), // 'book_appointment', 'question', 'emergency', 'followup'
  aiSentiment: varchar("ai_sentiment"), // 'positive', 'neutral', 'negative', 'urgent'
  aiExtractedInfo: jsonb("ai_extracted_info").$type<{ 
    reason?: string; 
    appointmentRequest?: { date: string; time: string }; 
    urgency?: string;
    actionItems?: string[];
  }>(),
  
  // Notes
  notes: text("notes"),
  tags: jsonb("tags").$type<string[]>(),
  
  // Follow-up
  requiresFollowup: boolean("requires_followup").default(false),
  followupCompletedAt: timestamp("followup_completed_at"),
  
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
}, (table) => ({
  doctorIdx: index("call_logs_doctor_idx").on(table.doctorId),
  patientIdx: index("call_logs_patient_idx").on(table.patientId),
  statusIdx: index("call_logs_status_idx").on(table.status),
  timeIdx: index("call_logs_time_idx").on(table.startTime),
}));

export const insertCallLogSchema = createInsertSchema(callLogs).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export type InsertCallLog = z.infer<typeof insertCallLogSchema>;
export type CallLog = typeof callLogs.$inferSelect;

// Appointment Reminders - Track reminder delivery
export const appointmentReminders = pgTable("appointment_reminders", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  appointmentId: varchar("appointment_id").notNull().references(() => appointments.id, { onDelete: "cascade" }),
  
  // Reminder configuration
  reminderType: varchar("reminder_type").notNull(), // 'sms', 'email', 'voice'
  scheduledFor: timestamp("scheduled_for").notNull(),
  timingType: varchar("timing_type").notNull(), // '24h', '1h', '15min'
  
  // Delivery status
  status: varchar("status").default("pending"), // 'pending', 'sent', 'delivered', 'failed', 'cancelled'
  sentAt: timestamp("sent_at"),
  deliveredAt: timestamp("delivered_at"),
  
  // Response tracking
  confirmed: boolean("confirmed").default(false),
  confirmedAt: timestamp("confirmed_at"),
  declined: boolean("declined").default(false),
  declinedAt: timestamp("declined_at"),
  
  // Content
  messageContent: text("message_content"),
  
  // Delivery details
  twilioMessageSid: varchar("twilio_message_sid"), // For SMS
  sesMessageId: varchar("ses_message_id"), // For email
  twilioCallSid: varchar("twilio_call_sid"), // For voice
  
  // Error tracking
  error: text("error"),
  retryCount: integer("retry_count").default(0),
  
  createdAt: timestamp("created_at").defaultNow(),
}, (table) => ({
  appointmentIdx: index("appointment_reminders_appointment_idx").on(table.appointmentId),
  scheduledIdx: index("appointment_reminders_scheduled_idx").on(table.scheduledFor),
  statusIdx: index("appointment_reminders_status_idx").on(table.status),
}));

export const insertAppointmentReminderSchema = createInsertSchema(appointmentReminders).omit({
  id: true,
  createdAt: true,
});

export type InsertAppointmentReminder = z.infer<typeof insertAppointmentReminderSchema>;
export type AppointmentReminder = typeof appointmentReminders.$inferSelect;

// Google Calendar sync tracking (for doctor's calendars)
export const googleCalendarSync = pgTable("google_calendar_sync", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  doctorId: varchar("doctor_id").notNull().unique().references(() => users.id),
  
  // OAuth tokens (encrypted in production)
  accessToken: text("access_token"),
  refreshToken: text("refresh_token"),
  tokenExpiry: timestamp("token_expiry"),
  
  // Calendar details
  calendarId: varchar("calendar_id"), // Google Calendar ID (usually email)
  calendarName: varchar("calendar_name"),
  
  // Sync status
  syncEnabled: boolean("sync_enabled").default(true),
  syncDirection: varchar("sync_direction").default("bidirectional"), // 'to_google', 'from_google', 'bidirectional'
  lastSyncAt: timestamp("last_sync_at"),
  lastSyncStatus: varchar("last_sync_status"), // 'success', 'partial', 'failed'
  lastSyncError: text("last_sync_error"),
  
  // Incremental sync tokens (for efficient syncing)
  syncToken: text("sync_token"), // Google Calendar sync token for incremental updates
  pageToken: text("page_token"), // For paginated results
  
  // Sync statistics
  totalEventsSynced: integer("total_events_synced").default(0),
  lastEventSyncedAt: timestamp("last_event_synced_at"),
  
  // Conflict resolution
  conflictResolution: varchar("conflict_resolution").default("google_wins"), // 'google_wins', 'local_wins', 'manual'
  
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
}, (table) => ({
  doctorIdx: index("google_calendar_sync_doctor_idx").on(table.doctorId),
}));

export const insertGoogleCalendarSyncSchema = createInsertSchema(googleCalendarSync).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export type InsertGoogleCalendarSync = z.infer<typeof insertGoogleCalendarSyncSchema>;
export type GoogleCalendarSync = typeof googleCalendarSync.$inferSelect;

// Google Calendar sync logs (audit trail)
export const googleCalendarSyncLogs = pgTable("google_calendar_sync_logs", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  doctorId: varchar("doctor_id").notNull().references(() => users.id),
  syncId: varchar("sync_id").references(() => googleCalendarSync.id),
  
  // Sync details
  syncType: varchar("sync_type").notNull(), // 'full', 'incremental', 'manual'
  syncDirection: varchar("sync_direction").notNull(), // 'to_google', 'from_google', 'bidirectional'
  
  // Results
  status: varchar("status").notNull(), // 'success', 'partial', 'failed'
  eventsCreated: integer("events_created").default(0),
  eventsUpdated: integer("events_updated").default(0),
  eventsDeleted: integer("events_deleted").default(0),
  conflictsDetected: integer("conflicts_detected").default(0),
  
  // Error tracking
  error: text("error"),
  errorDetails: jsonb("error_details"),
  
  // Performance
  durationMs: integer("duration_ms"),
  
  createdAt: timestamp("created_at").defaultNow(),
}, (table) => ({
  doctorIdx: index("google_calendar_sync_logs_doctor_idx").on(table.doctorId),
  createdAtIdx: index("google_calendar_sync_logs_created_idx").on(table.createdAt),
}));

export const insertGoogleCalendarSyncLogSchema = createInsertSchema(googleCalendarSyncLogs).omit({
  id: true,
  createdAt: true,
});

export type InsertGoogleCalendarSyncLog = z.infer<typeof insertGoogleCalendarSyncLogSchema>;
export type GoogleCalendarSyncLog = typeof googleCalendarSyncLogs.$inferSelect;

// ===== GMAIL SYNC (HIPAA CRITICAL) =====
// SECURITY REQUIREMENTS FOR PRODUCTION:
// 1. OAuth tokens MUST be encrypted at rest (AWS KMS envelope encryption)
// 2. Requires Google Workspace BAA - NOT regular Gmail
// 3. PHI detection/redaction required before syncing email content
// 4. Patient consent workflow required before syncing any emails
// 5. Immutable audit trail for all sync/access events
// 6. Token rotation policies and revocation workflows
// 7. Role-based access control enforcement
// 8. Domain validation to ensure Google Workspace account
export const gmailSync = pgTable("gmail_sync", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  doctorId: varchar("doctor_id").notNull().unique().references(() => users.id),
  
  // OAuth tokens - TODO: ENCRYPT WITH AWS KMS IN PRODUCTION
  accessToken: text("access_token"),
  refreshToken: text("refresh_token"),
  tokenExpiry: timestamp("token_expiry"),
  tokenScopes: text("token_scopes").array(), // Track granted scopes for audit
  
  // Google Workspace validation
  googleWorkspaceDomain: varchar("google_workspace_domain"), // Must be validated
  googleWorkspaceBaaConfirmed: boolean("google_workspace_baa_confirmed").default(false),
  
  // Sync configuration
  syncEnabled: boolean("sync_enabled").default(false), // Default OFF for safety
  lastSyncAt: timestamp("last_sync_at"),
  lastSyncStatus: varchar("last_sync_status"),
  lastSyncError: text("last_sync_error"),
  
  // Sync statistics
  totalEmailsSynced: integer("total_emails_synced").default(0),
  
  // PHI protection
  phiRedactionEnabled: boolean("phi_redaction_enabled").default(true),
  consentConfirmed: boolean("consent_confirmed").default(false),
  
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
}, (table) => ({
  doctorIdx: index("gmail_sync_doctor_idx").on(table.doctorId),
}));

export const insertGmailSyncSchema = createInsertSchema(gmailSync).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export type InsertGmailSync = z.infer<typeof insertGmailSyncSchema>;
export type GmailSync = typeof gmailSync.$inferSelect;

// Gmail sync audit logs (HIPAA immutable audit trail)
export const gmailSyncLogs = pgTable("gmail_sync_logs", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  doctorId: varchar("doctor_id").notNull().references(() => users.id),
  syncId: varchar("sync_id").references(() => gmailSync.id),
  
  // Sync details
  action: varchar("action").notNull(), // 'sync', 'send', 'access', 'token_refresh'
  status: varchar("status").notNull(),
  emailsFetched: integer("emails_fetched").default(0),
  phiDetected: boolean("phi_detected").default(false),
  
  // Error tracking
  error: text("error"),
  errorDetails: jsonb("error_details"),
  
  // Audit metadata
  ipAddress: varchar("ip_address"),
  userAgent: text("user_agent"),
  
  createdAt: timestamp("created_at").defaultNow(),
}, (table) => ({
  doctorIdx: index("gmail_sync_logs_doctor_idx").on(table.doctorId),
  createdAtIdx: index("gmail_sync_logs_created_idx").on(table.createdAt),
  actionIdx: index("gmail_sync_logs_action_idx").on(table.action),
}));

export const insertGmailSyncLogSchema = createInsertSchema(gmailSyncLogs).omit({
  id: true,
  createdAt: true,
});

export type InsertGmailSyncLog = z.infer<typeof insertGmailSyncLogSchema>;
export type GmailSyncLog = typeof gmailSyncLogs.$inferSelect;

// Video Exam Sessions - Guided live video examination workflow
export const videoExamSessions = pgTable("video_exam_sessions", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  patientId: varchar("patient_id").notNull().references(() => users.id),
  
  // Session timing
  startedAt: timestamp("started_at").defaultNow(),
  completedAt: timestamp("completed_at"),
  
  // Combined recording storage (if all segments are merged)
  combinedS3Key: varchar("combined_s3_key"),
  combinedS3Bucket: varchar("combined_s3_bucket"),
  combinedKmsKeyId: varchar("combined_kms_key_id"),
  combinedFileSizeBytes: integer("combined_file_size_bytes"),
  
  // Combined analysis reference
  combinedAnalysisId: varchar("combined_analysis_id"),
  
  // Session status: 'in_progress', 'completed', 'abandoned'
  status: varchar("status").notNull().default("in_progress"),
  
  // Metadata
  totalSegments: integer("total_segments").default(0),
  completedSegments: integer("completed_segments").default(0),
  skippedSegments: integer("skipped_segments").default(0),
  totalDurationSeconds: integer("total_duration_seconds").default(0),
  
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
}, (table) => ({
  patientIdx: index("video_exam_sessions_patient_idx").on(table.patientId),
  statusIdx: index("video_exam_sessions_status_idx").on(table.status),
  createdAtIdx: index("video_exam_sessions_created_idx").on(table.createdAt),
}));

export const insertVideoExamSessionSchema = createInsertSchema(videoExamSessions).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export type InsertVideoExamSession = z.infer<typeof insertVideoExamSessionSchema>;
export type VideoExamSession = typeof videoExamSessions.$inferSelect;

// Video Exam Segments - Individual examination recordings
export const videoExamSegments = pgTable("video_exam_segments", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  sessionId: varchar("session_id").notNull().references(() => videoExamSessions.id, { onDelete: "cascade" }),
  
  // Exam type: 'respiratory', 'skin_pallor', 'eye_sclera', 'swelling', 'tremor', 'tongue', 'custom'
  examType: varchar("exam_type").notNull(),
  sequenceOrder: integer("sequence_order").notNull(), // 1-7
  
  // Recording status
  skipped: boolean("skipped").default(false),
  prepDurationSeconds: integer("prep_duration_seconds").default(30),
  
  // Timing
  captureStartedAt: timestamp("capture_started_at"),
  captureEndedAt: timestamp("capture_ended_at"),
  durationSeconds: integer("duration_seconds"),
  
  // S3 storage (encrypted with KMS)
  s3Key: varchar("s3_key"),
  s3Bucket: varchar("s3_bucket"),
  kmsKeyId: varchar("kms_key_id"),
  fileSizeBytes: integer("file_size_bytes"),
  
  // AI Analysis reference
  analysisId: varchar("analysis_id"),
  
  // Processing status: 'pending', 'processing', 'completed', 'failed', 'skipped'
  status: varchar("status").notNull().default("pending"),
  
  // Custom abnormality description (for examType='custom')
  customLocation: text("custom_location"),
  customDescription: text("custom_description"),
  
  // Audit metadata
  uploadedBy: varchar("uploaded_by").references(() => users.id),
  
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
}, (table) => ({
  sessionIdx: index("video_exam_segments_session_idx").on(table.sessionId),
  examTypeIdx: index("video_exam_segments_exam_type_idx").on(table.examType),
  statusIdx: index("video_exam_segments_status_idx").on(table.status),
}));

export const insertVideoExamSegmentSchema = createInsertSchema(videoExamSegments).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export type InsertVideoExamSegment = z.infer<typeof insertVideoExamSegmentSchema>;
export type VideoExamSegment = typeof videoExamSegments.$inferSelect;

// ===========================================================================================
// BEHAVIOR AI ANALYSIS SYSTEM - Complete Multi-Modal Deterioration Detection
// ===========================================================================================

// Daily check-ins for behavioral pattern tracking
export const behaviorCheckins = pgTable("behavior_checkins", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  patientId: varchar("patient_id").notNull().references(() => users.id),
  
  // Check-in timing metadata
  scheduledTime: timestamp("scheduled_time"), // Expected check-in time
  completedAt: timestamp("completed_at"), // Actual completion time
  responseLatencyMinutes: integer("response_latency_minutes"), // Delay in minutes
  skipped: boolean("skipped").default(false),
  skipReason: text("skip_reason"), // Free text: "too tired", "not today", etc.
  
  // Self-reported data
  symptomSeverity: integer("symptom_severity"), // 1-10 scale
  symptomDescription: text("symptom_description"),
  painLevel: integer("pain_level"), // 1-10 scale
  medicationTaken: boolean("medication_taken").default(false),
  medicationSkippedReason: text("medication_skipped_reason"),
  
  // Engagement metrics
  sessionDurationSeconds: integer("session_duration_seconds"),
  interactionCount: integer("interaction_count"), // Number of taps/clicks
  
  // Avoidance pattern detection
  avoidanceLanguageDetected: boolean("avoidance_language_detected").default(false),
  avoidancePhrases: jsonb("avoidance_phrases").$type<string[]>(),
  
  // Sentiment from free text
  sentimentPolarity: decimal("sentiment_polarity", { precision: 5, scale: 3 }), // -1 to +1
  
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
}, (table) => ({
  patientIdx: index("behavior_checkins_patient_idx").on(table.patientId),
  completedIdx: index("behavior_checkins_completed_idx").on(table.completedAt),
  skippedIdx: index("behavior_checkins_skipped_idx").on(table.skipped),
}));

export const insertBehaviorCheckinSchema = createInsertSchema(behaviorCheckins).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export type InsertBehaviorCheckin = z.infer<typeof insertBehaviorCheckinSchema>;
export type BehaviorCheckin = typeof behaviorCheckins.$inferSelect;

// Aggregated behavioral metrics (daily rollup)
export const behaviorMetrics = pgTable("behavior_metrics", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  patientId: varchar("patient_id").notNull().references(() => users.id),
  date: timestamp("date").notNull(), // Date of metrics
  
  // Check-in consistency
  checkinTimeConsistencyScore: decimal("checkin_time_consistency_score", { precision: 5, scale: 3 }), // 0-1 (1=very consistent)
  checkinCompletionRate: decimal("checkin_completion_rate", { precision: 5, scale: 3 }), // % completed vs scheduled
  avgResponseLatencyMinutes: decimal("avg_response_latency_minutes", { precision: 8, scale: 2 }),
  skippedCheckinsCount: integer("skipped_checkins_count").default(0),
  
  // Routine deviation
  routineDeviationScore: decimal("routine_deviation_score", { precision: 5, scale: 3 }), // 0-1 (1=major deviation)
  
  // Medication adherence
  medicationAdherenceRate: decimal("medication_adherence_rate", { precision: 5, scale: 3 }), // 0-1
  medicationSkipsCount: integer("medication_skips_count").default(0),
  
  // App engagement
  appEngagementDurationMinutes: decimal("app_engagement_duration_minutes", { precision: 8, scale: 2 }),
  appOpenCount: integer("app_open_count").default(0),
  
  // Avoidance patterns
  avoidancePatternsDetected: boolean("avoidance_patterns_detected").default(false),
  avoidanceCount: integer("avoidance_count").default(0),
  avoidancePhrases: jsonb("avoidance_phrases").$type<string[]>(),
  
  // Tone change detection
  avgSentimentPolarity: decimal("avg_sentiment_polarity", { precision: 5, scale: 3 }),
  sentimentTrendSlope: decimal("sentiment_trend_slope", { precision: 8, scale: 5 }), // Rate of sentiment decline
  
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
}, (table) => ({
  patientDateIdx: index("behavior_metrics_patient_date_idx").on(table.patientId, table.date),
}));

export const insertBehaviorMetricSchema = createInsertSchema(behaviorMetrics).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export type InsertBehaviorMetric = z.infer<typeof insertBehaviorMetricSchema>;
export type BehaviorMetric = typeof behaviorMetrics.$inferSelect;

// Digital biomarkers from phone/wearable data
export const digitalBiomarkers = pgTable("digital_biomarkers", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  patientId: varchar("patient_id").notNull().references(() => users.id),
  date: timestamp("date").notNull(),
  
  // Activity metrics
  dailyStepCount: integer("daily_step_count"),
  stepTrend7Day: decimal("step_trend_7day", { precision: 8, scale: 2 }), // Slope of 7-day trend
  activityBurstCount: integer("activity_burst_count"), // Sudden bursts of activity
  sedentaryDurationMinutes: integer("sedentary_duration_minutes"),
  movementVariabilityScore: decimal("movement_variability_score", { precision: 5, scale: 3 }), // 0-1
  
  // Circadian rhythm
  circadianRhythmStability: decimal("circadian_rhythm_stability", { precision: 5, scale: 3 }), // 0-1
  sleepWakeIrregularityMinutes: integer("sleep_wake_irregularity_minutes"),
  dailyPeakActivityTime: varchar("daily_peak_activity_time"), // HH:MM format
  
  // Phone usage patterns
  phoneUsageIrregularity: decimal("phone_usage_irregularity", { precision: 5, scale: 3 }), // 0-1
  nightPhoneInteractionCount: integer("night_phone_interaction_count"), // 10pm-6am
  screenOnDurationMinutes: integer("screen_on_duration_minutes"),
  
  // Mobility changes
  mobilityDropDetected: boolean("mobility_drop_detected").default(false),
  mobilityChangePercent: decimal("mobility_change_percent", { precision: 6, scale: 2 }), // % change from baseline
  
  // Raw accelerometer stats
  accelerometerStdDev: decimal("accelerometer_std_dev", { precision: 10, scale: 5 }),
  accelerometerMeanMagnitude: decimal("accelerometer_mean_magnitude", { precision: 10, scale: 5 }),
  
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
}, (table) => ({
  patientDateIdx: index("digital_biomarkers_patient_date_idx").on(table.patientId, table.date),
  mobilityDropIdx: index("digital_biomarkers_mobility_drop_idx").on(table.mobilityDropDetected),
}));

export const insertDigitalBiomarkerSchema = createInsertSchema(digitalBiomarkers).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export type InsertDigitalBiomarker = z.infer<typeof insertDigitalBiomarkerSchema>;
export type DigitalBiomarker = typeof digitalBiomarkers.$inferSelect;

// Cognitive test results (weekly micro-tests)
export const cognitiveTests = pgTable("cognitive_tests", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  patientId: varchar("patient_id").notNull().references(() => users.id),
  testType: varchar("test_type").notNull(), // 'reaction_time', 'tapping', 'memory', 'pattern_recall', 'instruction_follow'
  
  // Test metadata
  startedAt: timestamp("started_at").notNull(),
  completedAt: timestamp("completed_at"),
  durationSeconds: integer("duration_seconds"),
  
  // Performance metrics
  reactionTimeMs: integer("reaction_time_ms"),
  tappingSpeed: decimal("tapping_speed", { precision: 6, scale: 2 }), // taps/second
  errorRate: decimal("error_rate", { precision: 5, scale: 3 }), // 0-1
  memoryScore: decimal("memory_score", { precision: 5, scale: 3 }), // 0-1
  patternRecallAccuracy: decimal("pattern_recall_accuracy", { precision: 5, scale: 3 }), // 0-1
  instructionAccuracy: decimal("instruction_accuracy", { precision: 5, scale: 3 }), // 0-1
  
  // Detailed results
  rawResults: jsonb("raw_results").$type<Record<string, any>>(),
  
  // Drift detection (vs patient baseline)
  baselineDeviation: decimal("baseline_deviation", { precision: 6, scale: 3 }), // Z-score
  anomalyDetected: boolean("anomaly_detected").default(false),
  
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
}, (table) => ({
  patientTypeIdx: index("cognitive_tests_patient_type_idx").on(table.patientId, table.testType),
  anomalyIdx: index("cognitive_tests_anomaly_idx").on(table.anomalyDetected),
}));

export const insertCognitiveTestSchema = createInsertSchema(cognitiveTests).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export type InsertCognitiveTest = z.infer<typeof insertCognitiveTestSchema>;
export type CognitiveTest = typeof cognitiveTests.$inferSelect;

// Sentiment/language analysis from text inputs
export const sentimentAnalysis = pgTable("sentiment_analysis", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  patientId: varchar("patient_id").notNull().references(() => users.id),
  sourceType: varchar("source_type").notNull(), // 'checkin', 'symptom_journal', 'chat', 'audio_transcript'
  sourceId: varchar("source_id"), // Reference to source record
  
  // Raw text
  textContent: text("text_content").notNull(),
  analyzedAt: timestamp("analyzed_at").notNull(),
  
  // Sentiment metrics (DistilBERT)
  sentimentPolarity: decimal("sentiment_polarity", { precision: 5, scale: 3 }), // -1 to +1
  sentimentLabel: varchar("sentiment_label"), // 'positive', 'neutral', 'negative'
  sentimentConfidence: decimal("sentiment_confidence", { precision: 5, scale: 3 }),
  
  // Language biomarkers
  messageLengthChars: integer("message_length_chars"),
  wordCount: integer("word_count"),
  lexicalComplexity: decimal("lexical_complexity", { precision: 5, scale: 3 }), // 0-1
  negativityRatio: decimal("negativity_ratio", { precision: 5, scale: 3 }), // % negative words
  
  // Stress/help-seeking keywords
  stressKeywordCount: integer("stress_keyword_count").default(0),
  stressKeywords: jsonb("stress_keywords").$type<string[]>(),
  helpSeekingDetected: boolean("help_seeking_detected").default(false),
  helpSeekingPhrases: jsonb("help_seeking_phrases").$type<string[]>(),
  
  // Hesitation markers
  hesitationCount: integer("hesitation_count").default(0),
  hesitationMarkers: jsonb("hesitation_markers").$type<string[]>(), // "maybe", "idk", "i guess"
  
  // Model metadata
  modelVersion: varchar("model_version").default("distilbert-base-uncased-finetuned-sst-2-english"),
  
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
}, (table) => ({
  patientSourceIdx: index("sentiment_analysis_patient_source_idx").on(table.patientId, table.sourceType),
  polarityIdx: index("sentiment_analysis_polarity_idx").on(table.sentimentPolarity),
}));

export const insertSentimentAnalysisSchema = createInsertSchema(sentimentAnalysis).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export type InsertSentimentAnalysis = z.infer<typeof insertSentimentAnalysisSchema>;
export type SentimentAnalysis = typeof sentimentAnalysis.$inferSelect;

// Multi-modal risk scores (combined from all biomarker streams)
export const behaviorRiskScores = pgTable("behavior_risk_scores", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  patientId: varchar("patient_id").notNull().references(() => users.id),
  calculatedAt: timestamp("calculated_at").notNull(),
  
  // Component risk scores (0-100)
  behavioralRisk: decimal("behavioral_risk", { precision: 5, scale: 2 }).notNull(),
  digitalBiomarkerRisk: decimal("digital_biomarker_risk", { precision: 5, scale: 2 }).notNull(),
  cognitiveRisk: decimal("cognitive_risk", { precision: 5, scale: 2 }).notNull(),
  sentimentRisk: decimal("sentiment_risk", { precision: 5, scale: 2 }).notNull(),
  
  // Composite risk score (weighted combination)
  compositeRisk: decimal("composite_risk", { precision: 5, scale: 2 }).notNull(),
  riskLevel: varchar("risk_level").notNull(), // 'low', 'moderate', 'high', 'critical'
  
  // Model details
  modelType: varchar("model_type").default("transformer_xgboost_ensemble"),
  modelVersion: varchar("model_version"),
  featureContributions: jsonb("feature_contributions").$type<Record<string, number>>(),
  
  // Top risk factors
  topRiskFactors: jsonb("top_risk_factors").$type<Array<{factor: string; impact: number}>>(),
  
  // Confidence
  predictionConfidence: decimal("prediction_confidence", { precision: 5, scale: 3 }),
  
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
}, (table) => ({
  patientCalcIdx: index("behavior_risk_scores_patient_calc_idx").on(table.patientId, table.calculatedAt),
  riskLevelIdx: index("behavior_risk_scores_risk_level_idx").on(table.riskLevel),
}));

export const insertBehaviorRiskScoreSchema = createInsertSchema(behaviorRiskScores).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export type InsertBehaviorRiskScore = z.infer<typeof insertBehaviorRiskScoreSchema>;
export type BehaviorRiskScore = typeof behaviorRiskScores.$inferSelect;

// Deterioration trend detection (temporal patterns)
export const deteriorationTrends = pgTable("deterioration_trends", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  patientId: varchar("patient_id").notNull().references(() => users.id),
  detectedAt: timestamp("detected_at").notNull(),
  
  // Trend metadata
  trendType: varchar("trend_type").notNull(), // 'declining_engagement', 'mobility_drop', 'cognitive_decline', 'sentiment_deterioration'
  severity: varchar("severity").notNull(), // 'mild', 'moderate', 'severe'
  
  // Temporal analysis
  trendStartDate: timestamp("trend_start_date"),
  trendDurationDays: integer("trend_duration_days"),
  trendSlope: decimal("trend_slope", { precision: 10, scale: 5 }), // Rate of change
  
  // Statistical significance
  zScore: decimal("z_score", { precision: 6, scale: 3 }),
  pValue: decimal("p_value", { precision: 10, scale: 8 }),
  confidenceLevel: decimal("confidence_level", { precision: 5, scale: 3 }), // 0-1
  
  // Affected metrics
  affectedMetrics: jsonb("affected_metrics").$type<string[]>(),
  metricValues: jsonb("metric_values").$type<Record<string, number[]>>(),
  
  // Clinical interpretation
  clinicalSignificance: text("clinical_significance"),
  recommendedActions: jsonb("recommended_actions").$type<string[]>(),
  
  // Alert generated
  alertGenerated: boolean("alert_generated").default(false),
  alertId: varchar("alert_id"),
  
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
}, (table) => ({
  patientTypeIdx: index("deterioration_trends_patient_type_idx").on(table.patientId, table.trendType),
  severityIdx: index("deterioration_trends_severity_idx").on(table.severity),
  alertIdx: index("deterioration_trends_alert_idx").on(table.alertGenerated),
}));

export const insertDeteriorationTrendSchema = createInsertSchema(deteriorationTrends).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export type InsertDeteriorationTrend = z.infer<typeof insertDeteriorationTrendSchema>;
export type DeteriorationTrend = typeof deteriorationTrends.$inferSelect;

// Alerts for behavior AI system
export const behaviorAlerts = pgTable("behavior_alerts", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  patientId: varchar("patient_id").notNull().references(() => users.id),
  triggeredAt: timestamp("triggered_at").notNull(),
  
  // Alert metadata
  alertType: varchar("alert_type").notNull(), // 'high_risk', 'trend_detected', 'anomaly', 'critical_decline'
  severity: varchar("severity").notNull(), // 'low', 'medium', 'high', 'critical'
  priority: integer("priority").notNull(), // 1-5 (5=highest)
  
  // Alert content
  title: varchar("title").notNull(),
  message: text("message").notNull(),
  
  // Data sources
  sourceRiskScoreId: varchar("source_risk_score_id").references(() => behaviorRiskScores.id),
  sourceTrendId: varchar("source_trend_id").references(() => deteriorationTrends.id),
  
  // Delivery status
  emailSent: boolean("email_sent").default(false),
  emailSentAt: timestamp("email_sent_at"),
  smsSent: boolean("sms_sent").default(false),
  smsSentAt: timestamp("sms_sent_at"),
  dashboardNotified: boolean("dashboard_notified").default(true),
  
  // Resolution
  acknowledged: boolean("acknowledged").default(false),
  acknowledgedAt: timestamp("acknowledged_at"),
  acknowledgedBy: varchar("acknowledged_by").references(() => users.id),
  resolved: boolean("resolved").default(false),
  resolvedAt: timestamp("resolved_at"),
  resolutionNotes: text("resolution_notes"),
  
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
}, (table) => ({
  patientSeverityIdx: index("behavior_alerts_patient_severity_idx").on(table.patientId, table.severity),
  typeIdx: index("behavior_alerts_type_idx").on(table.alertType),
  acknowledgedIdx: index("behavior_alerts_acknowledged_idx").on(table.acknowledged),
}));

export const insertBehaviorAlertSchema = createInsertSchema(behaviorAlerts).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export type InsertBehaviorAlert = z.infer<typeof insertBehaviorAlertSchema>;
export type BehaviorAlert = typeof behaviorAlerts.$inferSelect;

// Mental Health Questionnaire Responses - Standardized instruments (PHQ-9, GAD-7, etc.)
export const mentalHealthResponses = pgTable("mental_health_responses", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  patientId: varchar("patient_id").notNull().references(() => users.id),
  
  // Questionnaire identification
  questionnaireType: varchar("questionnaire_type").notNull(), // 'PHQ9', 'GAD7', 'PCL5', 'PSS10'
  questionnaireVersion: varchar("questionnaire_version").default("1.0"),
  
  // Response data - array of question responses
  responses: jsonb("responses").$type<Array<{
    questionId: string;
    questionText: string;
    response: number | string; // Numeric scores or text responses
    responseText?: string; // Human-readable response
  }>>().notNull(),
  
  // Scoring results
  totalScore: integer("total_score"),
  maxScore: integer("max_score"),
  severityLevel: varchar("severity_level"), // 'minimal', 'mild', 'moderate', 'moderately_severe', 'severe'
  
  // Symptom cluster scores (domain-specific)
  clusterScores: jsonb("cluster_scores").$type<{
    [key: string]: {
      score: number;
      maxScore: number;
      label: string;
      items: string[];
    };
  }>(),
  
  // Crisis flags
  crisisDetected: boolean("crisis_detected").default(false),
  crisisQuestionIds: jsonb("crisis_question_ids").$type<string[]>(),
  crisisResponses: jsonb("crisis_responses").$type<Array<{
    questionId: string;
    questionText: string;
    response: number | string;
  }>>(),
  
  // Completion metadata
  completedAt: timestamp("completed_at").defaultNow(),
  durationSeconds: integer("duration_seconds"), // Time taken to complete
  
  // Privacy and consent
  allowStorage: boolean("allow_storage").default(true),
  allowClinicalSharing: boolean("allow_clinical_sharing").default(false),
  
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
}, (table) => ({
  patientTypeIdx: index("mh_responses_patient_type_idx").on(table.patientId, table.questionnaireType),
  severityIdx: index("mh_responses_severity_idx").on(table.severityLevel),
  crisisIdx: index("mh_responses_crisis_idx").on(table.crisisDetected),
  completedIdx: index("mh_responses_completed_idx").on(table.completedAt),
}));

export const insertMentalHealthResponseSchema = createInsertSchema(mentalHealthResponses).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export type InsertMentalHealthResponse = z.infer<typeof insertMentalHealthResponseSchema>;
export type MentalHealthResponse = typeof mentalHealthResponses.$inferSelect;

// AI-powered pattern analysis for mental health data
export const mentalHealthPatternAnalysis = pgTable("mental_health_pattern_analysis", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  patientId: varchar("patient_id").notNull().references(() => users.id),
  responseId: varchar("response_id").references(() => mentalHealthResponses.id),
  
  // Analysis type
  analysisType: varchar("analysis_type").notNull(), // 'symptom_clustering', 'temporal_trends', 'cross_questionnaire', 'llm_insights'
  
  // Pattern detection results
  patterns: jsonb("patterns").$type<Array<{
    patternType: string;
    patternName: string;
    description: string;
    severity: string;
    confidence: number;
    supportingData: any;
  }>>(),
  
  // Symptom clusters identified by LLM
  symptomClusters: jsonb("symptom_clusters").$type<{
    [key: string]: {
      clusterName: string;
      symptoms: string[];
      frequency: string;
      severity: string;
      neutralDescription: string;
    };
  }>(),
  
  // Temporal trends (changes over time)
  temporalTrends: jsonb("temporal_trends").$type<Array<{
    metric: string;
    direction: string; // 'improving', 'worsening', 'stable', 'fluctuating'
    magnitude: string;
    timeframe: string;
    dataPoints: Array<{ date: string; value: number }>;
  }>>(),
  
  // LLM-generated neutral summary (no diagnostic language)
  neutralSummary: text("neutral_summary"),
  
  // Key observations (non-diagnostic)
  keyObservations: jsonb("key_observations").$type<string[]>(),
  
  // Recommended actions (general wellness, not treatment)
  suggestedActions: jsonb("suggested_actions").$type<Array<{
    category: string;
    action: string;
    priority: string;
  }>>(),
  
  // LLM model information
  llmModel: varchar("llm_model").default("gpt-4o"),
  llmTokensUsed: integer("llm_tokens_used"),
  
  // Analysis metadata
  analysisVersion: varchar("analysis_version").default("1.0"),
  analysisCompletedAt: timestamp("analysis_completed_at").defaultNow(),
  
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
}, (table) => ({
  patientIdx: index("mh_analysis_patient_idx").on(table.patientId),
  responseIdx: index("mh_analysis_response_idx").on(table.responseId),
  analysisTypeIdx: index("mh_analysis_type_idx").on(table.analysisType),
}));

export const insertMentalHealthPatternAnalysisSchema = createInsertSchema(mentalHealthPatternAnalysis).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export type InsertMentalHealthPatternAnalysis = z.infer<typeof insertMentalHealthPatternAnalysisSchema>;
export type MentalHealthPatternAnalysis = typeof mentalHealthPatternAnalysis.$inferSelect;

// PainTrack - Chronic pain tracking platform
export const paintrackSessions = pgTable("paintrack_sessions", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  userId: varchar("user_id").notNull().references(() => users.id),
  
  // Module and tracking info
  module: varchar("module").notNull(), // 'ArthroTrack', 'MuscleTrack', 'PostOpTrack'
  joint: varchar("joint").notNull(), // 'knee', 'hip', 'shoulder', 'elbow', 'wrist', 'ankle'
  laterality: varchar("laterality"), // 'left', 'right', 'bilateral'
  
  // Dual-camera video URLs (S3)
  frontVideoUrl: varchar("front_video_url"), // Face camera
  jointVideoUrl: varchar("joint_video_url"), // Joint camera
  
  // Self-reported pain (VAS 0-10)
  patientVas: integer("patient_vas"), // Visual Analog Scale 0-10
  patientNotes: text("patient_notes"), // Optional patient notes
  medicationTaken: boolean("medication_taken").default(false),
  medicationDetails: text("medication_details"),
  
  // Session metadata
  recordingDuration: integer("recording_duration"), // seconds
  deviceType: varchar("device_type"), // 'iPhone 17 Pro', 'Android', etc.
  dualCameraSupported: boolean("dual_camera_supported").default(false),
  
  // Processing status
  status: varchar("status").default("pending"), // 'pending', 'processing', 'completed', 'failed'
  processingError: text("processing_error"),
  
  // Video quality metrics
  videoQuality: jsonb("video_quality").$type<{
    lighting: number; // mean luminance
    motionBlur: number;
    occlusion: number;
    frameRate: number;
    visibility: string; // 'good', 'fair', 'poor'
  }>(),
  
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
}, (table) => ({
  userIdx: index("paintrack_user_idx").on(table.userId),
  moduleIdx: index("paintrack_module_idx").on(table.module),
  jointIdx: index("paintrack_joint_idx").on(table.joint),
  statusIdx: index("paintrack_status_idx").on(table.status),
}));

export const insertPaintrackSessionSchema = createInsertSchema(paintrackSessions).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export type InsertPaintrackSession = z.infer<typeof insertPaintrackSessionSchema>;
export type PaintrackSession = typeof paintrackSessions.$inferSelect;

// Session metrics extracted from ML models
export const sessionMetrics = pgTable("session_metrics", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  sessionId: varchar("session_id").notNull().references(() => paintrackSessions.id),
  
  // Joint/body metrics (MediaPipe BlazePose)
  jointMetrics: jsonb("joint_metrics").$type<{
    maxFlexionAngle: number;
    maxExtensionAngle: number;
    rangeOfMotion: number;
    extensionSpeed: number;
    flexionSpeed: number;
    smoothness: number; // inversely related to jerk
    accelerationVariability: number;
    symmetryScore?: number; // for paired limbs
  }>(),
  
  // Facial metrics (MediaPipe FaceMesh)
  facialMetrics: jsonb("facial_metrics").$type<{
    eyebrowMovementAmplitude: number;
    blinkRate: number;
    lipCornerDisplacement: number;
    mouthCornerDisplacement: number;
    headOrientationStability: number;
  }>(),
  
  // Anomaly detection (IsolationForest, LSTM Autoencoder)
  anomalyScore: decimal("anomaly_score", { precision: 5, scale: 3 }), // Change from baseline
  baselineDeviation: decimal("baseline_deviation", { precision: 5, scale: 3 }),
  
  // Correlation analysis (XGBoost) - OBSERVATIONAL ONLY
  correlationScore: decimal("correlation_score", { precision: 5, scale: 3 }), // Correlation with VAS (not pain prediction)
  
  // Model versions
  modelVersions: jsonb("model_versions").$type<{
    blazePose: string;
    faceMesh: string;
    isolationForest: string;
    lstmAutoencoder?: string;
    xgboost?: string;
  }>(),
  
  processedAt: timestamp("processed_at").defaultNow(),
  createdAt: timestamp("created_at").defaultNow(),
}, (table) => ({
  sessionIdx: index("session_metrics_session_idx").on(table.sessionId),
}));

export const insertSessionMetricsSchema = createInsertSchema(sessionMetrics).omit({
  id: true,
  createdAt: true,
  processedAt: true,
});

export type InsertSessionMetrics = z.infer<typeof insertSessionMetricsSchema>;
export type SessionMetrics = typeof sessionMetrics.$inferSelect;

// Clinician notes on PainTrack sessions
export const clinicianNotes = pgTable("clinician_notes", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  sessionId: varchar("session_id").notNull().references(() => paintrackSessions.id),
  clinicianId: varchar("clinician_id").notNull().references(() => users.id),
  note: text("note").notNull(),
  
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
}, (table) => ({
  sessionIdx: index("clinician_notes_session_idx").on(table.sessionId),
  clinicianIdx: index("clinician_notes_clinician_idx").on(table.clinicianId),
}));

export const insertClinicianNoteSchema = createInsertSchema(clinicianNotes).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export type InsertClinicianNote = z.infer<typeof insertClinicianNoteSchema>;
export type ClinicianNote = typeof clinicianNotes.$inferSelect;

// ============================================================================
// DAILY FOLLOW-UP SYMPTOM TRACKING SYSTEM
// ============================================================================
// Comprehensive symptom tracking module for daily check-ins with conversational
// extraction, ML-based trend/anomaly detection, and clinician-ready summaries.
// All outputs are labeled "observational" and non-diagnostic per HIPAA compliance.
// ============================================================================

// Symptom check-ins - Structured daily check-in data
export const symptomCheckins = pgTable("symptom_checkins", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  userId: varchar("user_id").notNull().references(() => users.id),
  timestamp: timestamp("timestamp").notNull().defaultNow(),
  
  // Structured symptom metrics (0-10 scales for self-reported patient data)
  painLevel: integer("pain_level"), // 0-10, null if not reported
  fatigueLevel: integer("fatigue_level"), // 0-10, null if not reported
  breathlessnessLevel: integer("breathlessness_level"), // 0-10, null if not reported
  sleepQuality: integer("sleep_quality"), // 0-10, null if not reported
  mood: varchar("mood"), // 'great', 'good', 'okay', 'low', 'very_low'
  mobilityScore: integer("mobility_score"), // 0-10, null if not reported
  medicationsTaken: boolean("medications_taken"), // Did patient take meds today?
  
  // Free-form symptom data
  triggers: text("triggers").array(), // Environmental/activity triggers
  symptoms: text("symptoms").array(), // List of symptoms experienced
  note: text("note"), // Optional free-text patient note
  voiceNoteUrl: varchar("voice_note_url"), // Optional S3 URL for voice recording
  voiceNoteDuration: integer("voice_note_duration"), // seconds
  
  // Source tracking
  source: varchar("source").notNull().default("app"), // 'app', 'agent', 'voice'
  sessionId: varchar("session_id").references(() => chatSessions.id), // If from Agent Clona
  
  // Metadata
  deviceType: varchar("device_type"), // 'ios', 'android', 'web'
  createdAt: timestamp("created_at").defaultNow(),
});

export const insertSymptomCheckinSchema = createInsertSchema(symptomCheckins).omit({
  id: true,
  createdAt: true,
});

export type InsertSymptomCheckin = z.infer<typeof insertSymptomCheckinSchema>;
export type SymptomCheckin = typeof symptomCheckins.$inferSelect;

// Daily symptom check-in form validation schema
export const dailyCheckinFormSchema = z.object({
  painLevel: z.number().min(0).max(10), // Always required, never null (legal requirement)
  fatigueLevel: z.number().min(0).max(10),
  breathlessnessLevel: z.number().min(0).max(10),
  sleepQuality: z.number().min(0).max(10),
  mood: z.enum(["great", "good", "okay", "low", "very_low"]),
  mobilityScore: z.number().min(0).max(10),
  medicationsTaken: z.boolean(),
  symptoms: z.array(z.string()).default([]),
  triggers: z.array(z.string()).default([]),
  note: z.string().trim().optional(),
}).transform((data) => ({
  ...data,
  // Convert zero values to null for optional metrics (pain always sent)
  fatigueLevel: data.fatigueLevel > 0 ? data.fatigueLevel : null,
  breathlessnessLevel: data.breathlessnessLevel > 0 ? data.breathlessnessLevel : null,
  note: data.note && data.note.length > 0 ? data.note : null,
}));

export type DailyCheckinFormInput = z.infer<typeof dailyCheckinFormSchema>;

// Chat symptoms - Extracted symptoms from Agent Clona conversations
export const chatSymptoms = pgTable("chat_symptoms", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  userId: varchar("user_id").notNull().references(() => users.id),
  sessionId: varchar("session_id").notNull().references(() => chatSessions.id),
  messageId: varchar("message_id").references(() => chatMessages.id),
  
  // Raw conversation data
  rawText: text("raw_text").notNull(), // Original message text
  
  // AI-extracted structured data
  extractedJson: jsonb("extracted_json").$type<{
    locations?: string[]; // Body locations mentioned
    symptomTypes?: string[]; // Types of symptoms (headache, nausea, etc.)
    intensityMentions?: string[]; // Severity descriptors (mild, severe, etc.)
    temporalInfo?: string; // When symptoms started/duration
    aggravatingFactors?: string[];
    relievingFactors?: string[];
  }>().notNull(),
  
  // Extraction metadata
  confidence: decimal("confidence", { precision: 3, scale: 2 }), // 0.0-1.0 confidence score
  extractionModel: varchar("extraction_model").default("gpt-4o"), // AI model used
  
  // Link to structured check-in (if created)
  symptomCheckinId: varchar("symptom_checkin_id").references(() => symptomCheckins.id),
  
  createdAt: timestamp("created_at").defaultNow(),
});

export const insertChatSymptomSchema = createInsertSchema(chatSymptoms).omit({
  id: true,
  createdAt: true,
});

export type InsertChatSymptom = z.infer<typeof insertChatSymptomSchema>;
export type ChatSymptom = typeof chatSymptoms.$inferSelect;

// Mental Health Red Flags - AI-detected mental health concerns from Agent Clona conversations
export const mentalHealthRedFlags = pgTable("mental_health_red_flags", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  userId: varchar("user_id").notNull().references(() => users.id),
  sessionId: varchar("session_id").notNull().references(() => chatSessions.id),
  messageId: varchar("message_id").references(() => chatMessages.id),
  
  // Raw conversation data
  rawText: text("raw_text").notNull(), // Original message text that triggered detection
  
  // AI-extracted structured data
  extractedJson: jsonb("extracted_json").$type<{
    redFlagTypes?: string[]; // Types: 'suicidal_ideation', 'self_harm', 'severe_depression', 'severe_anxiety', 'crisis_language', 'substance_abuse', 'hopelessness'
    severityLevel?: 'low' | 'moderate' | 'high' | 'critical'; // Severity assessment
    specificConcerns?: string[]; // Specific phrases or concerns identified
    emotionalTone?: string; // Overall emotional tone (despair, panic, hopelessness, etc.)
    recommendedAction?: string; // Recommended clinical action
    crisisIndicators?: boolean; // Whether immediate crisis intervention may be needed
  }>().notNull(),
  
  // Extraction metadata
  confidence: decimal("confidence", { precision: 3, scale: 2 }), // 0.0-1.0 confidence score
  extractionModel: varchar("extraction_model").default("gpt-4o"), // AI model used
  
  // Clinical metadata
  severityScore: integer("severity_score"), // 0-100 numerical severity
  requiresImmediateAttention: boolean("requires_immediate_attention").default(false), // Crisis flag
  clinicianNotified: boolean("clinician_notified").default(false), // Whether doctor was alerted
  clinicianNotifiedAt: timestamp("clinician_notified_at"),
  
  // Audit trail
  reviewedBy: varchar("reviewed_by").references(() => users.id), // Clinician who reviewed
  reviewedAt: timestamp("reviewed_at"),
  reviewNotes: text("review_notes"),
  
  createdAt: timestamp("created_at").defaultNow(),
}, (table) => ({
  userIdIdx: index("mental_health_red_flags_user_id_idx").on(table.userId),
  sessionIdIdx: index("mental_health_red_flags_session_id_idx").on(table.sessionId),
  createdAtIdx: index("mental_health_red_flags_created_at_idx").on(table.createdAt),
}));

export const insertMentalHealthRedFlagSchema = createInsertSchema(mentalHealthRedFlags).omit({
  id: true,
  createdAt: true,
});

export type InsertMentalHealthRedFlag = z.infer<typeof insertMentalHealthRedFlagSchema>;
export type MentalHealthRedFlag = typeof mentalHealthRedFlags.$inferSelect;

// Passive metrics - Device-collected health data (wearables, phone sensors)
export const passiveMetrics = pgTable("passive_metrics", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  userId: varchar("user_id").notNull().references(() => users.id),
  date: timestamp("date").notNull(), // Date of data collection
  
  // Activity metrics
  steps: integer("steps"),
  activeMinutes: integer("active_minutes"),
  caloriesBurned: integer("calories_burned"),
  distanceMeters: integer("distance_meters"),
  
  // Heart metrics
  hrMean: integer("hr_mean"), // Average heart rate (bpm)
  hrMin: integer("hr_min"),
  hrMax: integer("hr_max"),
  hrv: integer("hrv"), // Heart rate variability (ms)
  restingHr: integer("resting_hr"),
  
  // Sleep metrics
  sleepMinutes: integer("sleep_minutes"), // Total sleep duration
  deepSleepMinutes: integer("deep_sleep_minutes"),
  remSleepMinutes: integer("rem_sleep_minutes"),
  lightSleepMinutes: integer("light_sleep_minutes"),
  awakeMinutes: integer("awake_minutes"),
  sleepScore: integer("sleep_score"), // 0-100 if device provides
  
  // Respiratory metrics
  respiratoryRate: integer("respiratory_rate"), // Breaths per minute
  spo2Mean: integer("spo2_mean"), // Average oxygen saturation %
  spo2Min: integer("spo2_min"),
  
  // Stress/recovery
  stressScore: integer("stress_score"), // 0-100 if device provides
  recoveryScore: integer("recovery_score"), // 0-100 if device provides
  
  // Device metadata
  deviceMeta: jsonb("device_meta").$type<{
    source: string; // 'fitbit', 'apple_watch', 'garmin', 'phone_sensor'
    model?: string;
    firmwareVersion?: string;
    batteryLevel?: number;
  }>(),
  
  // Sync metadata
  syncedAt: timestamp("synced_at").defaultNow(),
  dataQuality: varchar("data_quality"), // 'high', 'medium', 'low' based on completeness
  
  createdAt: timestamp("created_at").defaultNow(),
}, (table) => ({
  userDateIdx: index("passive_metrics_user_date_idx").on(table.userId, table.date),
}));

export const insertPassiveMetricSchema = createInsertSchema(passiveMetrics).omit({
  id: true,
  createdAt: true,
});

export type InsertPassiveMetric = z.infer<typeof insertPassiveMetricSchema>;
export type PassiveMetric = typeof passiveMetrics.$inferSelect;

// Trend reports - ML-generated trend analysis and anomaly detection
export const trendReports = pgTable("trend_reports", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  userId: varchar("user_id").notNull().references(() => users.id),
  periodStart: timestamp("period_start").notNull(),
  periodEnd: timestamp("period_end").notNull(),
  reportType: varchar("report_type").notNull(), // '3day', '7day', '15day', '30day'
  
  // Aggregated metrics (observational data only)
  aggregatedMetrics: jsonb("aggregated_metrics").$type<{
    // Average values across period
    avgPainLevel?: number;
    avgFatigueLevel?: number;
    avgBreathlessness?: number;
    avgSleepQuality?: number;
    avgMobilityScore?: number;
    
    // Device metrics averages
    avgSteps?: number;
    avgHrMean?: number;
    avgHrv?: number;
    avgSleepMinutes?: number;
    avgSpo2?: number;
    
    // Trends (increasing, stable, decreasing)
    painTrend?: string;
    fatigueTrend?: string;
    sleepTrend?: string;
    activityTrend?: string;
    
    // Most common symptoms
    topSymptoms?: Array<{ symptom: string; frequency: number }>;
    topTriggers?: Array<{ trigger: string; frequency: number }>;
  }>(),
  
  // ML-detected anomalies (observational, non-diagnostic)
  anomalies: jsonb("anomalies").$type<Array<{
    metricName: string; // e.g., "pain_level", "sleep_quality"
    date: string;
    value: number;
    expectedRange: { min: number; max: number };
    severity: "mild" | "moderate" | "significant"; // Observational severity
    description: string; // Patient-friendly description
  }>>(),
  
  // Correlational insights (observational patterns only)
  correlations: jsonb("correlations").$type<Array<{
    metric1: string;
    metric2: string;
    correlationType: "positive" | "negative" | "none";
    strength: number; // 0.0-1.0
    observationalNote: string; // e.g., "Lower sleep quality appears associated with higher pain levels"
  }>>(),
  
  // Clinician-ready summary (observational language)
  clinicianSummary: text("clinician_summary"), // Plain English summary for doctor review
  
  // Metadata
  generatedBy: varchar("generated_by").default("ml_trend_engine"), // Algorithm identifier
  dataPointsAnalyzed: integer("data_points_analyzed"), // Number of check-ins included
  confidenceScore: decimal("confidence_score", { precision: 3, scale: 2 }), // 0.0-1.0
  
  generatedAt: timestamp("generated_at").defaultNow(),
  reviewedByDoctor: boolean("reviewed_by_doctor").default(false),
  reviewedBy: varchar("reviewed_by").references(() => users.id),
  reviewedAt: timestamp("reviewed_at"),
  
  createdAt: timestamp("created_at").defaultNow(),
}, (table) => ({
  userPeriodIdx: index("trend_reports_user_period_idx").on(table.userId, table.periodStart, table.periodEnd),
}));

export const insertTrendReportSchema = createInsertSchema(trendReports).omit({
  id: true,
  createdAt: true,
  generatedAt: true,
});

export type InsertTrendReport = z.infer<typeof insertTrendReportSchema>;
export type TrendReport = typeof trendReports.$inferSelect;

// ============================================
// AI HEALTH ALERT ENGINE TABLES
// ============================================

// Trend Metrics - Z-scores, slopes, volatility for each tracked metric
export const aiTrendMetrics = pgTable("ai_trend_metrics", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  patientId: varchar("patient_id").notNull().references(() => users.id),
  
  // Metric identification
  metricName: varchar("metric_name").notNull(), // 'heart_rate', 'pain_level', 'fatigue', 'mobility', etc.
  metricCategory: varchar("metric_category").notNull(), // 'vital', 'symptom', 'activity', 'behavioral'
  
  // Raw and baseline values
  rawValue: decimal("raw_value", { precision: 10, scale: 4 }).notNull(),
  baseline14dMean: decimal("baseline_14d_mean", { precision: 10, scale: 4 }),
  baseline14dStd: decimal("baseline_14d_std", { precision: 10, scale: 4 }),
  
  // Z-score deviation (current - mean_14d) / std_14d
  zScore: decimal("z_score", { precision: 6, scale: 3 }),
  zScoreSeverity: varchar("z_score_severity"), // 'normal', 'elevated', 'high', 'critical'
  
  // Rolling slopes (linear regression on rolling windows)
  slope3d: decimal("slope_3d", { precision: 8, scale: 5 }), // 3-day slope
  slope7d: decimal("slope_7d", { precision: 8, scale: 5 }), // 7-day slope
  slope14d: decimal("slope_14d", { precision: 8, scale: 5 }), // 14-day slope
  slopeDirection: varchar("slope_direction"), // 'increasing', 'stable', 'decreasing'
  
  // Volatility Index = std(14-day values)
  volatilityIndex: decimal("volatility_index", { precision: 8, scale: 4 }),
  volatilityLevel: varchar("volatility_level"), // 'stable', 'moderate', 'high', 'extreme'
  
  // Composite Trend Risk Index (0-100)
  compositeTrendScore: decimal("composite_trend_score", { precision: 5, scale: 2 }),
  
  recordedAt: timestamp("recorded_at").notNull(),
  computedAt: timestamp("computed_at").defaultNow(),
}, (table) => ({
  patientMetricIdx: index("ai_trend_metrics_patient_metric_idx").on(table.patientId, table.metricName),
  recordedAtIdx: index("ai_trend_metrics_recorded_at_idx").on(table.recordedAt),
  zScoreIdx: index("ai_trend_metrics_zscore_idx").on(table.zScore),
}));

export const insertAiTrendMetricSchema = createInsertSchema(aiTrendMetrics).omit({
  id: true,
  computedAt: true,
});

export type InsertAiTrendMetric = z.infer<typeof insertAiTrendMetricSchema>;
export type AiTrendMetric = typeof aiTrendMetrics.$inferSelect;

// Engagement Metrics - Adherence, check-ins, engagement scores
export const aiEngagementMetrics = pgTable("ai_engagement_metrics", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  patientId: varchar("patient_id").notNull().references(() => users.id),
  
  // Date range for metrics
  periodStart: timestamp("period_start").notNull(),
  periodEnd: timestamp("period_end").notNull(),
  
  // Adherence Score = (completed_actions / expected_actions) * 100
  adherenceScore: decimal("adherence_score", { precision: 5, scale: 2 }),
  checkinsCompleted: integer("checkins_completed").default(0),
  checkinsExpected: integer("checkins_expected").default(0),
  capturesCompleted: integer("captures_completed").default(0), // Video/audio exams
  surveysCompleted: integer("surveys_completed").default(0),
  
  // Engagement Score (composite)
  engagementScore: decimal("engagement_score", { precision: 5, scale: 2 }),
  engagementTrend: varchar("engagement_trend"), // 'improving', 'stable', 'declining'
  engagementDrop14d: decimal("engagement_drop_14d", { precision: 5, scale: 2 }), // % drop vs 14-day baseline
  
  // Time-to-Alert metrics
  avgTimeToAlert: integer("avg_time_to_alert"), // seconds from anomaly to alert
  alertsGenerated: integer("alerts_generated").default(0),
  alertsAcknowledged: integer("alerts_acknowledged").default(0),
  alertsDismissed: integer("alerts_dismissed").default(0), // False positive proxy
  
  // Streak tracking
  currentStreak: integer("current_streak").default(0), // consecutive days of check-ins
  longestStreak: integer("longest_streak").default(0),
  
  computedAt: timestamp("computed_at").defaultNow(),
}, (table) => ({
  patientPeriodIdx: index("ai_engagement_metrics_patient_period_idx").on(table.patientId, table.periodStart),
}));

export const insertAiEngagementMetricSchema = createInsertSchema(aiEngagementMetrics).omit({
  id: true,
  computedAt: true,
});

export type InsertAiEngagementMetric = z.infer<typeof insertAiEngagementMetricSchema>;
export type AiEngagementMetric = typeof aiEngagementMetrics.$inferSelect;

// Quality of Life Metrics - Wellness index, functional status, self-care
export const aiQolMetrics = pgTable("ai_qol_metrics", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  patientId: varchar("patient_id").notNull().references(() => users.id),
  
  // Daily Wellness Index (0-100)
  wellnessIndex: decimal("wellness_index", { precision: 5, scale: 2 }),
  wellnessComponents: jsonb("wellness_components").$type<{
    moodScore: number;
    energyScore: number;
    mobilityTrend: number;
    adherenceContribution: number;
  }>(),
  wellnessTrend: varchar("wellness_trend"), // 'improving', 'stable', 'declining'
  
  // Functional Status Proxy (0-100)
  functionalStatus: decimal("functional_status", { precision: 5, scale: 2 }),
  functionalComponents: jsonb("functional_components").$type<{
    activityLevel: number;
    gaitSpeed: number | null;
    engagementFactor: number;
  }>(),
  
  // Self-care Consistency Score (0-100)
  selfcareScore: decimal("selfcare_score", { precision: 5, scale: 2 }),
  selfcareComponents: jsonb("selfcare_components").$type<{
    medicationAdherence: number;
    hydrationLogs: number;
    checkinStreak: number;
  }>(),
  
  // Daily Stability Score = 100 - volatility - negative_slopes - missed_checkins_penalty
  stabilityScore: decimal("stability_score", { precision: 5, scale: 2 }),
  
  // Organ-System Behavior Scores (statistical patterns, NOT medical)
  behaviorPatterns: jsonb("behavior_patterns").$type<{
    respiratoryLikePattern: number; // 0-100 based on activity, breath patterns
    fluidLikePattern: number; // 0-100 based on weight + activity trends
    moodNeuroPattern: number; // 0-100 based on mood + consistency
    behavioralStabilityPattern: number; // 0-100 based on interactions + check-ins
  }>(),
  
  recordedAt: timestamp("recorded_at").notNull(),
  computedAt: timestamp("computed_at").defaultNow(),
}, (table) => ({
  patientDateIdx: index("ai_qol_metrics_patient_date_idx").on(table.patientId, table.recordedAt),
  wellnessIdx: index("ai_qol_metrics_wellness_idx").on(table.wellnessIndex),
}));

export const insertAiQolMetricSchema = createInsertSchema(aiQolMetrics).omit({
  id: true,
  computedAt: true,
});

export type InsertAiQolMetric = z.infer<typeof insertAiQolMetricSchema>;
export type AiQolMetric = typeof aiQolMetrics.$inferSelect;

// AI Health Alerts - Comprehensive alert system with compliance guardrails
export const aiHealthAlerts = pgTable("ai_health_alerts", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  patientId: varchar("patient_id").notNull().references(() => users.id),
  
  // Alert classification
  alertType: varchar("alert_type").notNull(), // 'trend', 'engagement', 'qol'
  alertCategory: varchar("alert_category").notNull(), // 'zscore_deviation', 'slope_negative', 'volatility_high', 'missed_checkins', 'wellness_drop', etc.
  severity: varchar("severity").notNull(), // 'low', 'moderate', 'high', 'critical'
  priority: integer("priority").notNull(), // 1-10 (10 = most urgent)
  
  // Escalation Probability (ML ranking 0-1, NOT a diagnosis)
  escalationProbability: decimal("escalation_probability", { precision: 4, scale: 3 }),
  
  // Alert content
  title: varchar("title").notNull(),
  message: text("message").notNull(),
  
  // COMPLIANCE: Mandatory disclaimer
  disclaimer: text("disclaimer").notNull().default("This is an observational pattern alert. Not a diagnosis or medical opinion."),
  
  // Contributing metrics (JSON for flexibility)
  contributingMetrics: jsonb("contributing_metrics").$type<Array<{
    metricName: string;
    value: number;
    zScore?: number;
    slope?: number;
    threshold: number;
    contribution: string; // 'primary', 'secondary'
  }>>(),
  
  // Trigger details
  triggerRule: varchar("trigger_rule"), // Which rule triggered this alert
  triggerThreshold: decimal("trigger_threshold", { precision: 10, scale: 4 }),
  triggerValue: decimal("trigger_value", { precision: 10, scale: 4 }),
  
  // Status workflow
  status: varchar("status").notNull().default("new"), // 'new', 'acknowledged', 'dismissed', 'escalated', 'resolved'
  acknowledgedBy: varchar("acknowledged_by").references(() => users.id),
  acknowledgedAt: timestamp("acknowledged_at"),
  dismissedBy: varchar("dismissed_by").references(() => users.id),
  dismissedAt: timestamp("dismissed_at"),
  dismissReason: text("dismiss_reason"),
  
  // Notification tracking
  notifiedPatient: boolean("notified_patient").default(false),
  notifiedClinician: boolean("notified_clinician").default(false),
  smsAlertSent: boolean("sms_alert_sent").default(false),
  emailAlertSent: boolean("email_alert_sent").default(false),
  
  // Audit trail
  clinicianNotes: text("clinician_notes"),
  clinicianId: varchar("clinician_id").references(() => users.id),
  
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
}, (table) => ({
  patientStatusIdx: index("ai_health_alerts_patient_status_idx").on(table.patientId, table.status),
  severityIdx: index("ai_health_alerts_severity_idx").on(table.severity),
  typeIdx: index("ai_health_alerts_type_idx").on(table.alertType),
  createdAtIdx: index("ai_health_alerts_created_at_idx").on(table.createdAt),
}));

export const insertAiHealthAlertSchema = createInsertSchema(aiHealthAlerts).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export type InsertAiHealthAlert = z.infer<typeof insertAiHealthAlertSchema>;
export type AiHealthAlert = typeof aiHealthAlerts.$inferSelect;

// Alert Rules Configuration - Configurable thresholds
export const aiAlertRules = pgTable("ai_alert_rules", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  
  ruleName: varchar("rule_name").notNull().unique(),
  ruleCategory: varchar("rule_category").notNull(), // 'trend', 'engagement', 'qol'
  description: text("description"),
  
  // Thresholds
  thresholds: jsonb("thresholds").$type<{
    zScoreThreshold?: number; // Default 2.5
    slopeThreshold?: number;
    volatilityThreshold?: number;
    compositeScoreThreshold?: number;
    engagementDropThreshold?: number; // Default 30%
    wellnessDropThreshold?: number; // Default 20 points
    missedCheckinsThreshold?: number; // Default 3 in 48 hours
    adherenceMinimum?: number; // Default 60%
  }>(),
  
  severity: varchar("severity").notNull(), // Default severity for this rule
  enabled: boolean("enabled").default(true),
  
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
});

export const insertAiAlertRuleSchema = createInsertSchema(aiAlertRules).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export type InsertAiAlertRule = z.infer<typeof insertAiAlertRuleSchema>;
export type AiAlertRule = typeof aiAlertRules.$inferSelect;

// Clinician Workload Metrics - Track alert handling efficiency
export const clinicianWorkloadMetrics = pgTable("clinician_workload_metrics", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  clinicianId: varchar("clinician_id").notNull().references(() => users.id),
  
  periodStart: timestamp("period_start").notNull(),
  periodEnd: timestamp("period_end").notNull(),
  
  // Workload metrics
  alertsReceived: integer("alerts_received").default(0),
  alertsAcknowledged: integer("alerts_acknowledged").default(0),
  alertsDismissed: integer("alerts_dismissed").default(0),
  alertsEscalated: integer("alerts_escalated").default(0),
  avgResponseTimeSeconds: integer("avg_response_time_seconds"),
  
  // Workload reduction calculation
  manualChecksAvoided: integer("manual_checks_avoided").default(0),
  baselineManualChecks: integer("baseline_manual_checks").default(0),
  workloadReductionPercent: decimal("workload_reduction_percent", { precision: 5, scale: 2 }),
  
  computedAt: timestamp("computed_at").defaultNow(),
}, (table) => ({
  clinicianPeriodIdx: index("clinician_workload_clinician_period_idx").on(table.clinicianId, table.periodStart),
}));

export const insertClinicianWorkloadMetricSchema = createInsertSchema(clinicianWorkloadMetrics).omit({
  id: true,
  computedAt: true,
});

export type InsertClinicianWorkloadMetric = z.infer<typeof insertClinicianWorkloadMetricSchema>;
export type ClinicianWorkloadMetric = typeof clinicianWorkloadMetrics.$inferSelect;

// Doctor-Patient Assignments - Explicit authorization for HIPAA compliance
export const doctorPatientAssignments = pgTable("doctor_patient_assignments", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  
  // Core relationship
  doctorId: varchar("doctor_id").notNull().references(() => users.id),
  patientId: varchar("patient_id").notNull().references(() => users.id),
  
  // Assignment status
  status: varchar("status").notNull().default("active"), // 'active', 'inactive', 'pending', 'revoked'
  
  // Source of assignment (for audit trail)
  assignmentSource: varchar("assignment_source").notNull(), // 'appointment', 'prescription', 'referral', 'consultation', 'manual', 'intake'
  sourceReferenceId: varchar("source_reference_id"), // ID of the related record (appointment_id, prescription_id, etc.)
  
  // Consent tracking
  patientConsented: boolean("patient_consented").default(false),
  consentedAt: timestamp("consented_at"),
  consentMethod: varchar("consent_method"), // 'in_app', 'paper', 'verbal', 'implied'
  
  // Primary care relationship
  isPrimaryCareProvider: boolean("is_primary_care_provider").default(false),
  
  // Specialty access (for specialists with limited scope)
  accessScope: varchar("access_scope").default("full"), // 'full', 'limited', 'emergency_only'
  accessNotes: text("access_notes"),
  
  // Revocation tracking
  revokedAt: timestamp("revoked_at"),
  revokedBy: varchar("revoked_by").references(() => users.id),
  revocationReason: text("revocation_reason"),
  
  // Audit trail
  assignedBy: varchar("assigned_by").references(() => users.id), // Who created this assignment
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
}, (table) => ({
  doctorPatientIdx: index("doctor_patient_idx").on(table.doctorId, table.patientId),
  doctorActiveIdx: index("doctor_active_assignments_idx").on(table.doctorId, table.status),
  patientActiveIdx: index("patient_active_assignments_idx").on(table.patientId, table.status),
  uniqueActiveAssignment: index("unique_active_doctor_patient").on(table.doctorId, table.patientId, table.status),
}));

export const insertDoctorPatientAssignmentSchema = createInsertSchema(doctorPatientAssignments).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export type InsertDoctorPatientAssignment = z.infer<typeof insertDoctorPatientAssignmentSchema>;
export type DoctorPatientAssignment = typeof doctorPatientAssignments.$inferSelect;

// Patient Consent Requests - Doctor requests to add a patient
export const patientConsentRequests = pgTable("patient_consent_requests", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  
  // Request parties
  doctorId: varchar("doctor_id").notNull().references(() => users.id),
  patientId: varchar("patient_id").notNull().references(() => users.id),
  
  // Request status
  status: varchar("status").notNull().default("pending"), // 'pending', 'approved', 'rejected', 'expired'
  
  // Request details
  requestMessage: text("request_message"), // Optional message from doctor
  
  // Response details
  respondedAt: timestamp("responded_at"),
  responseMessage: text("response_message"), // Optional message from patient
  
  // Expiry (requests expire after 7 days)
  expiresAt: timestamp("expires_at"),
  
  // Audit trail
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
}, (table) => ({
  doctorPendingIdx: index("doctor_pending_requests_idx").on(table.doctorId, table.status),
  patientPendingIdx: index("patient_pending_requests_idx").on(table.patientId, table.status),
}));

export const insertPatientConsentRequestSchema = createInsertSchema(patientConsentRequests).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export type InsertPatientConsentRequest = z.infer<typeof insertPatientConsentRequestSchema>;
export type PatientConsentRequest = typeof patientConsentRequests.$inferSelect;

// =========================================================================
// DOCTOR-PATIENT CONSENT PERMISSIONS - Granular access control
// =========================================================================

export const doctorPatientConsentPermissions = pgTable("doctor_patient_consent_permissions", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  
  // Link to assignment
  assignmentId: varchar("assignment_id").notNull().references(() => doctorPatientAssignments.id),
  doctorId: varchar("doctor_id").notNull().references(() => users.id),
  patientId: varchar("patient_id").notNull().references(() => users.id),
  
  // ===== MANDATORY PERMISSIONS (Required for consent) =====
  // Health Data Sharing
  shareHealthData: boolean("share_health_data").notNull().default(false),
  
  // Doctor-Patient Confidentiality Agreement
  confidentialityAgreed: boolean("confidentiality_agreed").notNull().default(false),
  
  // ===== GRANULAR DATA ACCESS PERMISSIONS =====
  // Medical Files (lab results, imaging, reports)
  shareMedicalFiles: boolean("share_medical_files").notNull().default(false),
  
  // Medication Records
  shareMedications: boolean("share_medications").notNull().default(true),
  
  // AI Agent Messages (Clona/Lysa conversations)
  shareAIMessages: boolean("share_ai_messages").notNull().default(false),
  
  // Doctor-Patient Messages
  shareDoctorMessages: boolean("share_doctor_messages").notNull().default(true),
  
  // Daily Followup Insights
  shareDailyFollowups: boolean("share_daily_followups").notNull().default(true),
  
  // AI Health Alerts (disease predictions, deterioration)
  shareHealthAlerts: boolean("share_health_alerts").notNull().default(true),
  
  // Behavioral Insights (habits, mental health scores)
  shareBehavioralInsights: boolean("share_behavioral_insights").notNull().default(false),
  
  // Pain Tracking Data
  sharePainTracking: boolean("share_pain_tracking").notNull().default(true),
  
  // Vital Signs History
  shareVitalSigns: boolean("share_vital_signs").notNull().default(true),
  
  // ===== RESEARCH CONSENT =====
  // Consent to anonymized data for epidemiological research
  consentEpidemiologicalResearch: boolean("consent_epidemiological_research").notNull().default(false),
  
  // ===== TERMS AGREEMENT =====
  termsVersion: varchar("terms_version").notNull().default("1.0"),
  termsAgreedAt: timestamp("terms_agreed_at"),
  
  // Digital signature (for legal compliance)
  digitalSignature: text("digital_signature"), // Base64 encoded signature or typed name
  signatureMethod: varchar("signature_method"), // 'typed', 'drawn', 'biometric'
  
  // IP address and device info for audit
  consentIpAddress: varchar("consent_ip_address"),
  consentUserAgent: text("consent_user_agent"),
  
  // Audit trail
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
  modifiedBy: varchar("modified_by").references(() => users.id),
}, (table) => ({
  assignmentIdx: index("consent_permissions_assignment_idx").on(table.assignmentId),
  doctorPatientIdx: index("consent_permissions_doctor_patient_idx").on(table.doctorId, table.patientId),
}));

export const insertDoctorPatientConsentPermissionsSchema = createInsertSchema(doctorPatientConsentPermissions).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export type InsertDoctorPatientConsentPermissions = z.infer<typeof insertDoctorPatientConsentPermissionsSchema>;
export type DoctorPatientConsentPermissions = typeof doctorPatientConsentPermissions.$inferSelect;

// =========================================================================
// LYSA PATIENT MONITORING SYSTEM
// =========================================================================

// Lysa Monitoring Assignments - Active patient monitoring by doctors via Lysa AI
export const lysaMonitoringAssignments = pgTable("lysa_monitoring_assignments", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  
  // Relationship
  doctorId: varchar("doctor_id").notNull().references(() => users.id),
  patientId: varchar("patient_id").notNull().references(() => users.id),
  
  // Monitoring configuration
  monitoringLevel: varchar("monitoring_level").notNull().default("standard"), // 'minimal', 'standard', 'intensive', 'critical'
  isActive: boolean("is_active").notNull().default(true),
  
  // Monitoring parameters
  alertThresholds: jsonb("alert_thresholds").$type<{
    vitalSigns?: { heartRate?: { min: number; max: number }; bloodPressure?: { systolicMax: number; diastolicMax: number }; oxygenSaturation?: { min: number } };
    symptoms?: { painLevelMax: number; fatigueMax: number };
    medications?: { missedDoseAlert: boolean; interactionCheck: boolean };
    deterioration?: { riskThreshold: number };
  }>(),
  
  // Monitoring schedule
  checkFrequency: varchar("check_frequency").default("daily"), // 'hourly', 'every_4h', 'every_8h', 'daily', 'weekly'
  lastCheckAt: timestamp("last_check_at"),
  nextScheduledCheck: timestamp("next_scheduled_check"),
  
  // AI monitoring preferences
  autoGenerateSummaries: boolean("auto_generate_summaries").default(true),
  summaryFrequency: varchar("summary_frequency").default("daily"), // 'daily', 'weekly', 'on_change'
  lastSummaryAt: timestamp("last_summary_at"),
  
  // Alert preferences
  enableAlerts: boolean("enable_alerts").default(true),
  alertChannels: jsonb("alert_channels").$type<{
    inApp: boolean;
    email: boolean;
    sms: boolean;
  }>(),
  
  // Notes and context
  monitoringNotes: text("monitoring_notes"),
  focusAreas: jsonb("focus_areas").$type<string[]>(), // ['respiratory', 'cardiac', 'mental_health', 'medication_adherence']
  
  // Status tracking
  status: varchar("status").notNull().default("active"), // 'active', 'paused', 'completed', 'transferred'
  pausedAt: timestamp("paused_at"),
  pauseReason: text("pause_reason"),
  completedAt: timestamp("completed_at"),
  completionReason: text("completion_reason"),
  
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
}, (table) => ({
  doctorPatientIdx: index("lysa_monitoring_doctor_patient_idx").on(table.doctorId, table.patientId),
  doctorActiveIdx: index("lysa_monitoring_doctor_active_idx").on(table.doctorId, table.isActive),
  patientActiveIdx: index("lysa_monitoring_patient_active_idx").on(table.patientId, table.isActive),
  nextCheckIdx: index("lysa_monitoring_next_check_idx").on(table.nextScheduledCheck),
}));

export const insertLysaMonitoringAssignmentSchema = createInsertSchema(lysaMonitoringAssignments).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export type InsertLysaMonitoringAssignment = z.infer<typeof insertLysaMonitoringAssignmentSchema>;
export type LysaMonitoringAssignment = typeof lysaMonitoringAssignments.$inferSelect;

// Lysa Monitoring Events - All events and actions taken during monitoring
export const lysaMonitoringEvents = pgTable("lysa_monitoring_events", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  
  // Relationship to monitoring assignment
  monitoringAssignmentId: varchar("monitoring_assignment_id").notNull().references(() => lysaMonitoringAssignments.id),
  patientId: varchar("patient_id").notNull().references(() => users.id),
  doctorId: varchar("doctor_id").references(() => users.id),
  
  // Event classification
  eventType: varchar("event_type").notNull(), // 'check', 'alert', 'summary', 'intervention', 'communication', 'data_update', 'analysis'
  eventCategory: varchar("event_category").notNull(), // 'vital_signs', 'symptoms', 'medications', 'lab_results', 'imaging', 'behavior', 'ai_insight'
  severity: varchar("severity").default("info"), // 'info', 'low', 'moderate', 'high', 'critical'
  
  // Event details
  title: varchar("title").notNull(),
  description: text("description"),
  
  // AI-generated content
  aiAnalysis: text("ai_analysis"),
  aiRecommendations: jsonb("ai_recommendations").$type<Array<{
    recommendation: string;
    priority: 'immediate' | 'high' | 'medium' | 'low';
    category: string;
    reasoning: string;
  }>>(),
  aiConfidence: decimal("ai_confidence", { precision: 3, scale: 2 }), // 0.00 to 1.00
  
  // Data references
  sourceDataType: varchar("source_data_type"), // 'daily_followup', 'chat_message', 'lab_report', 'imaging', 'wearable'
  sourceDataId: varchar("source_data_id"),
  relatedMetrics: jsonb("related_metrics").$type<Array<{
    metricName: string;
    value: number;
    unit: string;
    baseline?: number;
    deviation?: number;
  }>>(),
  
  // Action taken
  actionRequired: boolean("action_required").default(false),
  actionTaken: varchar("action_taken"),
  actionTakenBy: varchar("action_taken_by").references(() => users.id),
  actionTakenAt: timestamp("action_taken_at"),
  
  // Communication flags
  notifiedDoctor: boolean("notified_doctor").default(false),
  notifiedPatient: boolean("notified_patient").default(false),
  notificationMethod: varchar("notification_method"), // 'in_app', 'email', 'sms', 'multiple'
  
  // Status
  status: varchar("status").default("new"), // 'new', 'viewed', 'acknowledged', 'resolved', 'escalated', 'dismissed'
  resolvedAt: timestamp("resolved_at"),
  resolvedBy: varchar("resolved_by").references(() => users.id),
  resolutionNotes: text("resolution_notes"),
  
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
}, (table) => ({
  monitoringAssignmentIdx: index("lysa_events_assignment_idx").on(table.monitoringAssignmentId),
  patientEventIdx: index("lysa_events_patient_idx").on(table.patientId, table.eventType),
  doctorEventIdx: index("lysa_events_doctor_idx").on(table.doctorId),
  severityIdx: index("lysa_events_severity_idx").on(table.severity),
  statusIdx: index("lysa_events_status_idx").on(table.status),
  createdAtIdx: index("lysa_events_created_at_idx").on(table.createdAt),
}));

export const insertLysaMonitoringEventSchema = createInsertSchema(lysaMonitoringEvents).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export type InsertLysaMonitoringEvent = z.infer<typeof insertLysaMonitoringEventSchema>;
export type LysaMonitoringEvent = typeof lysaMonitoringEvents.$inferSelect;

// Lysa Monitoring Artifacts - Generated reports, summaries, and documents
export const lysaMonitoringArtifacts = pgTable("lysa_monitoring_artifacts", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  
  // Relationship
  monitoringAssignmentId: varchar("monitoring_assignment_id").references(() => lysaMonitoringAssignments.id),
  patientId: varchar("patient_id").notNull().references(() => users.id),
  doctorId: varchar("doctor_id").references(() => users.id),
  
  // Artifact classification
  artifactType: varchar("artifact_type").notNull(), // 'daily_summary', 'weekly_report', 'trend_analysis', 'risk_assessment', 'care_plan', 'handoff_note', 'consultation_prep'
  artifactFormat: varchar("artifact_format").notNull().default("markdown"), // 'markdown', 'pdf', 'json', 'html'
  
  // Artifact metadata
  title: varchar("title").notNull(),
  description: text("description"),
  
  // Content
  content: text("content"), // Main content (markdown/html)
  structuredData: jsonb("structured_data").$type<{
    summary?: string;
    keyFindings?: string[];
    riskFactors?: Array<{ factor: string; level: string; trend: string }>;
    recommendations?: Array<{ action: string; priority: string; reasoning: string }>;
    metrics?: Array<{ name: string; value: number; trend: string; baseline: number }>;
    medications?: Array<{ name: string; status: string; adherence: number }>;
    upcomingActions?: Array<{ action: string; dueDate: string; priority: string }>;
  }>(),
  
  // File storage (for PDFs etc.)
  fileUrl: varchar("file_url"),
  fileSize: integer("file_size"),
  fileMimeType: varchar("file_mime_type"),
  
  // Time range covered
  periodStart: timestamp("period_start"),
  periodEnd: timestamp("period_end"),
  
  // Generation metadata
  generatedBy: varchar("generated_by").default("lysa_ai"), // 'lysa_ai', 'doctor', 'system'
  aiModelVersion: varchar("ai_model_version"),
  generationPrompt: text("generation_prompt"),
  
  // Review status
  requiresReview: boolean("requires_review").default(true),
  reviewedBy: varchar("reviewed_by").references(() => users.id),
  reviewedAt: timestamp("reviewed_at"),
  reviewNotes: text("review_notes"),
  approvedForPatient: boolean("approved_for_patient").default(false),
  
  // Sharing
  sharedWithPatient: boolean("shared_with_patient").default(false),
  sharedAt: timestamp("shared_at"),
  
  // Versioning
  version: integer("version").default(1),
  previousVersionId: varchar("previous_version_id"),
  
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
}, (table) => ({
  monitoringAssignmentIdx: index("lysa_artifacts_assignment_idx").on(table.monitoringAssignmentId),
  patientArtifactIdx: index("lysa_artifacts_patient_idx").on(table.patientId, table.artifactType),
  doctorArtifactIdx: index("lysa_artifacts_doctor_idx").on(table.doctorId),
  createdAtIdx: index("lysa_artifacts_created_at_idx").on(table.createdAt),
  periodIdx: index("lysa_artifacts_period_idx").on(table.periodStart, table.periodEnd),
}));

export const insertLysaMonitoringArtifactSchema = createInsertSchema(lysaMonitoringArtifacts).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export type InsertLysaMonitoringArtifact = z.infer<typeof insertLysaMonitoringArtifactSchema>;
export type LysaMonitoringArtifact = typeof lysaMonitoringArtifacts.$inferSelect;

// Lysa Clinical Insights - AI-generated clinical observations for doctor review
export const lysaClinicalInsights = pgTable("lysa_clinical_insights", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  
  // Relationship
  patientId: varchar("patient_id").notNull().references(() => users.id),
  doctorId: varchar("doctor_id").references(() => users.id),
  monitoringAssignmentId: varchar("monitoring_assignment_id").references(() => lysaMonitoringAssignments.id),
  
  // Insight classification
  insightType: varchar("insight_type").notNull(), // 'pattern_detection', 'trend_alert', 'risk_indicator', 'medication_concern', 'behavioral_change', 'symptom_correlation'
  insightCategory: varchar("insight_category").notNull(), // 'clinical', 'behavioral', 'medication', 'preventive', 'urgent'
  severity: varchar("severity").notNull().default("info"), // 'info', 'advisory', 'warning', 'urgent', 'critical'
  
  // Insight content
  title: varchar("title").notNull(),
  summary: text("summary").notNull(),
  detailedAnalysis: text("detailed_analysis"),
  
  // Evidence and reasoning
  evidencePoints: jsonb("evidence_points").$type<Array<{
    dataPoint: string;
    value: string;
    source: string;
    timestamp: string;
    significance: string;
  }>>(),
  aiReasoning: text("ai_reasoning"),
  confidenceScore: decimal("confidence_score", { precision: 3, scale: 2 }), // 0.00 to 1.00
  
  // Recommendations
  suggestedActions: jsonb("suggested_actions").$type<Array<{
    action: string;
    priority: 'immediate' | 'high' | 'medium' | 'low';
    category: string;
    rationale: string;
  }>>(),
  
  // Clinical references
  relatedDiagnoses: jsonb("related_diagnoses").$type<string[]>(),
  relatedMedications: jsonb("related_medications").$type<string[]>(),
  clinicalGuidelines: jsonb("clinical_guidelines").$type<Array<{
    guideline: string;
    source: string;
    relevance: string;
  }>>(),
  
  // Doctor interaction
  status: varchar("status").default("new"), // 'new', 'viewed', 'acknowledged', 'acted_upon', 'dismissed', 'archived'
  viewedAt: timestamp("viewed_at"),
  acknowledgedAt: timestamp("acknowledged_at"),
  doctorNotes: text("doctor_notes"),
  actionTaken: text("action_taken"),
  
  // Validity
  validUntil: timestamp("valid_until"),
  supersededBy: varchar("superseded_by"),
  
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
}, (table) => ({
  patientInsightIdx: index("lysa_insights_patient_idx").on(table.patientId, table.insightType),
  doctorInsightIdx: index("lysa_insights_doctor_idx").on(table.doctorId),
  severityIdx: index("lysa_insights_severity_idx").on(table.severity),
  statusIdx: index("lysa_insights_status_idx").on(table.status),
  createdAtIdx: index("lysa_insights_created_at_idx").on(table.createdAt),
}));

export const insertLysaClinicalInsightSchema = createInsertSchema(lysaClinicalInsights).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export type InsertLysaClinicalInsight = z.infer<typeof insertLysaClinicalInsightSchema>;
export type LysaClinicalInsight = typeof lysaClinicalInsights.$inferSelect;

// Doctor Integrations - Per-doctor OAuth tokens and connection settings for personal accounts
export const doctorIntegrations = pgTable("doctor_integrations", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  doctorId: varchar("doctor_id").notNull().references(() => users.id),
  
  // Integration type
  integrationType: varchar("integration_type").notNull(), // 'gmail', 'outlook', 'twilio', 'whatsapp_business'
  
  // Connection status
  status: varchar("status").notNull().default("disconnected"), // 'disconnected', 'connecting', 'connected', 'error', 'expired'
  lastSyncAt: timestamp("last_sync_at"),
  lastErrorAt: timestamp("last_error_at"),
  lastErrorMessage: text("last_error_message"),
  
  // OAuth tokens (encrypted at rest)
  accessToken: text("access_token"),
  refreshToken: text("refresh_token"),
  tokenExpiresAt: timestamp("token_expires_at"),
  tokenScope: text("token_scope"),
  
  // Provider-specific identifiers
  providerAccountId: varchar("provider_account_id"), // Gmail email, Twilio account SID, etc.
  providerAccountEmail: varchar("provider_account_email"),
  
  // Twilio-specific fields
  twilioPhoneNumber: varchar("twilio_phone_number"),
  twilioAccountSid: varchar("twilio_account_sid"),
  twilioApiKey: varchar("twilio_api_key"),
  twilioApiSecret: text("twilio_api_secret"),
  
  // WhatsApp Business specific
  whatsappBusinessId: varchar("whatsapp_business_id"),
  whatsappPhoneNumberId: varchar("whatsapp_phone_number_id"),
  whatsappDisplayNumber: varchar("whatsapp_display_number"),
  
  // Sync settings
  syncEnabled: boolean("sync_enabled").default(true),
  syncFrequencyMinutes: integer("sync_frequency_minutes").default(5),
  autoReplyEnabled: boolean("auto_reply_enabled").default(false),
  
  // Email-specific settings
  emailLabelsToSync: jsonb("email_labels_to_sync").$type<string[]>(), // ['INBOX', 'SENT', etc.]
  emailAutoCategorizationEnabled: boolean("email_auto_categorization_enabled").default(true),
  
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
}, (table) => ({
  doctorIntegrationIdx: index("doctor_integrations_doctor_idx").on(table.doctorId),
  typeIdx: index("doctor_integrations_type_idx").on(table.integrationType),
  statusIdx: index("doctor_integrations_status_idx").on(table.status),
  uniqueDoctorType: index("doctor_integrations_unique_idx").on(table.doctorId, table.integrationType),
}));

export const insertDoctorIntegrationSchema = createInsertSchema(doctorIntegrations).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export type InsertDoctorIntegration = z.infer<typeof insertDoctorIntegrationSchema>;
export type DoctorIntegration = typeof doctorIntegrations.$inferSelect;

// Doctor Email Sync - Cached synced emails from doctor's personal account
export const doctorEmails = pgTable("doctor_emails", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  doctorId: varchar("doctor_id").notNull().references(() => users.id),
  integrationId: varchar("integration_id").notNull().references(() => doctorIntegrations.id),
  
  // Original email identifiers
  providerMessageId: varchar("provider_message_id").notNull(), // Gmail message ID
  threadId: varchar("thread_id"),
  
  // Email metadata
  subject: varchar("subject"),
  fromEmail: varchar("from_email").notNull(),
  fromName: varchar("from_name"),
  toEmails: jsonb("to_emails").$type<string[]>(),
  ccEmails: jsonb("cc_emails").$type<string[]>(),
  
  // Email content
  snippet: text("snippet"),
  bodyPlain: text("body_plain"),
  bodyHtml: text("body_html"),
  
  // Email state
  isRead: boolean("is_read").default(false),
  isStarred: boolean("is_starred").default(false),
  labels: jsonb("labels").$type<string[]>(),
  
  // Patient linking
  linkedPatientId: varchar("linked_patient_id").references(() => users.id),
  patientLinkConfidence: decimal("patient_link_confidence", { precision: 3, scale: 2 }),
  
  // AI analysis
  aiCategory: varchar("ai_category"), // 'appointment_request', 'medical_question', 'prescription_refill', 'test_results', 'general', 'spam'
  aiPriority: varchar("ai_priority"), // 'urgent', 'high', 'normal', 'low'
  aiSummary: text("ai_summary"),
  aiSuggestedReply: text("ai_suggested_reply"),
  aiExtractedInfo: jsonb("ai_extracted_info").$type<{
    appointmentRequest?: { date?: string; time?: string; reason?: string };
    symptoms?: string[];
    medications?: string[];
    urgencyIndicators?: string[];
  }>(),
  
  // Reply tracking
  hasBeenReplied: boolean("has_been_replied").default(false),
  repliedAt: timestamp("replied_at"),
  
  // Timestamps
  receivedAt: timestamp("received_at").notNull(),
  syncedAt: timestamp("synced_at").defaultNow(),
  createdAt: timestamp("created_at").defaultNow(),
}, (table) => ({
  doctorEmailIdx: index("doctor_emails_doctor_idx").on(table.doctorId),
  threadIdx: index("doctor_emails_thread_idx").on(table.threadId),
  categoryIdx: index("doctor_emails_category_idx").on(table.aiCategory),
  patientIdx: index("doctor_emails_patient_idx").on(table.linkedPatientId),
  receivedIdx: index("doctor_emails_received_idx").on(table.receivedAt),
}));

export const insertDoctorEmailSchema = createInsertSchema(doctorEmails).omit({
  id: true,
  createdAt: true,
  syncedAt: true,
});

export type InsertDoctorEmail = z.infer<typeof insertDoctorEmailSchema>;
export type DoctorEmail = typeof doctorEmails.$inferSelect;

// Doctor WhatsApp Messages - Synced messages from doctor's WhatsApp Business
export const doctorWhatsappMessages = pgTable("doctor_whatsapp_messages", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  doctorId: varchar("doctor_id").notNull().references(() => users.id),
  integrationId: varchar("integration_id").notNull().references(() => doctorIntegrations.id),
  
  // WhatsApp identifiers
  waMessageId: varchar("wa_message_id").notNull(),
  waConversationId: varchar("wa_conversation_id"),
  
  // Message details
  direction: varchar("direction").notNull(), // 'inbound', 'outbound'
  fromNumber: varchar("from_number").notNull(),
  toNumber: varchar("to_number").notNull(),
  contactName: varchar("contact_name"),
  
  // Content
  messageType: varchar("message_type").notNull(), // 'text', 'image', 'document', 'audio', 'video', 'location'
  textContent: text("text_content"),
  mediaUrl: varchar("media_url"),
  mediaMimeType: varchar("media_mime_type"),
  
  // Patient linking
  linkedPatientId: varchar("linked_patient_id").references(() => users.id),
  
  // AI analysis
  aiCategory: varchar("ai_category"), // 'appointment_request', 'question', 'confirmation', 'general'
  aiPriority: varchar("ai_priority"),
  aiSuggestedReply: text("ai_suggested_reply"),
  aiExtractedInfo: jsonb("ai_extracted_info").$type<{
    appointmentRequest?: { date?: string; time?: string };
    intent?: string;
    urgency?: string;
  }>(),
  
  // Status
  status: varchar("status").notNull().default("received"), // 'received', 'read', 'replied', 'archived'
  repliedAt: timestamp("replied_at"),
  
  // Timestamps
  receivedAt: timestamp("received_at").notNull(),
  syncedAt: timestamp("synced_at").defaultNow(),
  createdAt: timestamp("created_at").defaultNow(),
}, (table) => ({
  doctorWaIdx: index("doctor_wa_doctor_idx").on(table.doctorId),
  conversationIdx: index("doctor_wa_conversation_idx").on(table.waConversationId),
  patientIdx: index("doctor_wa_patient_idx").on(table.linkedPatientId),
  receivedIdx: index("doctor_wa_received_idx").on(table.receivedAt),
}));

export const insertDoctorWhatsappMessageSchema = createInsertSchema(doctorWhatsappMessages).omit({
  id: true,
  createdAt: true,
  syncedAt: true,
});

export type InsertDoctorWhatsappMessage = z.infer<typeof insertDoctorWhatsappMessageSchema>;
export type DoctorWhatsappMessage = typeof doctorWhatsappMessages.$inferSelect;

// ============================================================================
// MULTI-AGENT COMMUNICATION SYSTEM
// ============================================================================

// Agent definitions (Lysa for doctors, Clona for patients)
export const agents = pgTable("agents", {
  id: varchar("id").primaryKey(), // 'lysa', 'clona', or custom agent IDs
  name: varchar("name").notNull(),
  description: text("description"),
  agentType: varchar("agent_type").notNull(), // 'assistant' (Lysa), 'companion' (Clona), 'system'
  targetRole: varchar("target_role").notNull(), // 'doctor', 'patient', 'all'
  
  // Agent persona and behavior
  personaJson: jsonb("persona_json").$type<{
    systemPrompt: string;
    personality: string;
    tone: string;
    specializations: string[];
    constraints: string[];
  }>(),
  
  // Memory policy configuration
  memoryPolicyJson: jsonb("memory_policy_json").$type<{
    shortTermTtlHours: number;
    longTermEnabled: boolean;
    vectorizationEnabled: boolean;
    maxMemoryPerPatient: number;
    summarizationThreshold: number;
  }>(),
  
  // OpenAI Assistants API integration
  openaiAssistantId: varchar("openai_assistant_id"),
  openaiModel: varchar("openai_model").default("gpt-4o"),
  
  // Status and versioning
  isActive: boolean("is_active").default(true),
  version: integer("version").default(1),
  
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
});

export const insertAgentSchema = createInsertSchema(agents).omit({
  createdAt: true,
  updatedAt: true,
});

export type InsertAgent = z.infer<typeof insertAgentSchema>;
export type Agent = typeof agents.$inferSelect;

// Agent tools registry
export const agentTools = pgTable("agent_tools", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  name: varchar("name").notNull().unique(),
  displayName: varchar("display_name").notNull(),
  description: text("description"),
  
  // Tool configuration
  toolType: varchar("tool_type").notNull(), // 'calendar', 'messaging', 'ehr_fetch', 'prescription_draft', 'lab_fetch', 'imaging_linker'
  configJson: jsonb("config_json").$type<{
    endpoint?: string;
    method?: string;
    parametersSchema?: Record<string, unknown>;
    responseSchema?: Record<string, unknown>;
    timeout?: number;
  }>(),
  
  // Parameters schema (JSON Schema format for OpenAI function calling)
  parametersSchema: jsonb("parameters_schema").$type<Record<string, unknown>>(),
  
  // Permissions and access control
  requiredPermissions: jsonb("required_permissions").$type<string[]>(),
  allowedRoles: jsonb("allowed_roles").$type<string[]>(), // ['doctor', 'patient']
  requiresApproval: boolean("requires_approval").default(false), // Human-in-the-loop
  approvalRole: varchar("approval_role"), // Role that must approve (e.g., 'doctor' for prescriptions)
  
  // Status and versioning
  isEnabled: boolean("is_enabled").default(true),
  version: integer("version").default(1),
  
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
});

export const insertAgentToolSchema = createInsertSchema(agentTools).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export type InsertAgentTool = z.infer<typeof insertAgentToolSchema>;
export type AgentTool = typeof agentTools.$inferSelect;

// Agent-tool assignments (which agents can use which tools)
export const agentToolAssignments = pgTable("agent_tool_assignments", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  agentId: varchar("agent_id").notNull().references(() => agents.id),
  toolId: varchar("tool_id").notNull().references(() => agentTools.id),
  
  // Override settings for this agent-tool combination
  customConfig: jsonb("custom_config").$type<Record<string, unknown>>(),
  isEnabled: boolean("is_enabled").default(true),
  
  createdAt: timestamp("created_at").defaultNow(),
}, (table) => ({
  agentToolIdx: index("agent_tool_idx").on(table.agentId, table.toolId),
}));

export const insertAgentToolAssignmentSchema = createInsertSchema(agentToolAssignments).omit({
  id: true,
  createdAt: true,
});

export type InsertAgentToolAssignment = z.infer<typeof insertAgentToolAssignmentSchema>;
export type AgentToolAssignment = typeof agentToolAssignments.$inferSelect;

// Agent conversations (threads between users and agents, or agent-to-agent)
// CRITICAL: Patient↔Doctor direct messages share the SAME conversation thread
export const agentConversations = pgTable("agent_conversations", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  
  // Conversation type determines routing rules
  // 'patient_clona' - Patient talks to their Clona agent
  // 'doctor_lysa' - Doctor talks to their Lysa assistant
  // 'patient_doctor' - Direct messaging between connected patient and doctor (SAME THREAD)
  // 'patient_lysa' - Patient talks to connected doctor's Lysa
  // 'doctor_clona' - Doctor talks to connected patient's Clona
  // 'clona_lysa' - Inter-agent communication between Clona and Lysa
  conversationType: varchar("conversation_type").notNull().default("patient_clona"),
  
  // Participants (supports multi-party in same thread)
  participant1Type: varchar("participant1_type").notNull(), // 'user', 'agent'
  participant1Id: varchar("participant1_id").notNull(),
  participant2Type: varchar("participant2_type").notNull(), // 'user', 'agent'
  participant2Id: varchar("participant2_id").notNull(),
  
  // Additional participants for multi-party (e.g., patient, doctor, and agents in same thread)
  additionalParticipants: jsonb("additional_participants").$type<Array<{ type: string; id: string; role: string }>>(),
  
  // Context - REQUIRED for cross-party communication (consent verification)
  patientId: varchar("patient_id").references(() => users.id), // Patient in this conversation
  doctorId: varchar("doctor_id").references(() => users.id), // Doctor in this conversation
  
  // Doctor-Patient connection reference (for consent verification)
  assignmentId: varchar("assignment_id"), // Reference to doctor_patient_assignments.id
  
  // Conversation metadata
  title: varchar("title"),
  status: varchar("status").notNull().default("active"), // 'active', 'archived', 'closed'
  
  // Message counts per participant (using JSON for flexibility)
  messageCount: integer("message_count").default(0),
  unreadCounts: jsonb("unread_counts").$type<Record<string, number>>().default({}), // {participantId: unreadCount}
  
  // Last activity
  lastMessageAt: timestamp("last_message_at"),
  lastMessagePreview: text("last_message_preview"),
  lastMessageSenderId: varchar("last_message_sender_id"), // Who sent the last message
  lastMessageSenderRole: varchar("last_message_sender_role"), // 'patient', 'doctor', 'clona', 'lysa'
  
  // OpenAI thread (if using Assistants API)
  openaiThreadId: varchar("openai_thread_id"),
  
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
}, (table) => ({
  conversationTypeIdx: index("conv_type_idx").on(table.conversationType),
  participant1Idx: index("conv_participant1_idx").on(table.participant1Type, table.participant1Id),
  participant2Idx: index("conv_participant2_idx").on(table.participant2Type, table.participant2Id),
  patientIdx: index("conv_patient_idx").on(table.patientId),
  doctorIdx: index("conv_doctor_idx").on(table.doctorId),
  assignmentIdx: index("conv_assignment_idx").on(table.assignmentId),
  lastMessageIdx: index("conv_last_message_idx").on(table.lastMessageAt),
}));

export const insertAgentConversationSchema = createInsertSchema(agentConversations).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export type InsertAgentConversation = z.infer<typeof insertAgentConversationSchema>;
export type AgentConversation = typeof agentConversations.$inferSelect;

// Agent messages with full envelope protocol
// CRITICAL: Patient↔Doctor messages tagged with senderRole for proper display
export const agentMessages = pgTable("agent_messages", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  msgId: varchar("msg_id").notNull().unique(), // UUID for message tracking
  conversationId: varchar("conversation_id").notNull().references(() => agentConversations.id),
  
  // Message envelope - From
  fromType: varchar("from_type").notNull(), // 'agent', 'user', 'system'
  fromId: varchar("from_id").notNull(),
  
  // Sender role for proper tagging (CRITICAL for Patient↔Doctor conversations)
  // 'patient' - Message from patient user
  // 'doctor' - Message from doctor user
  // 'clona' - Message from Agent Clona (patient's AI)
  // 'lysa' - Message from Assistant Lysa (doctor's AI)
  // 'system' - System-generated message
  senderRole: varchar("sender_role").notNull().default("patient"),
  
  // Sender display info (cached for UI performance)
  senderName: varchar("sender_name"), // e.g., "Dr. Smith" or "John Doe"
  senderAvatar: varchar("sender_avatar"), // Profile image URL
  
  // Message envelope - To (JSON array for multi-recipient)
  toJson: jsonb("to_json").$type<Array<{ type: string; id: string }>>().notNull(),
  
  // Message type and content
  messageType: varchar("message_type").notNull(), // 'chat', 'command', 'event', 'tool_call', 'ack'
  content: text("content"),
  payloadJson: jsonb("payload_json").$type<Record<string, unknown>>(),
  
  // Tool call specific fields
  toolCallId: varchar("tool_call_id"),
  toolName: varchar("tool_name"),
  toolInput: jsonb("tool_input").$type<Record<string, unknown>>(),
  toolOutput: jsonb("tool_output").$type<Record<string, unknown>>(),
  toolStatus: varchar("tool_status"), // 'pending', 'running', 'completed', 'failed', 'pending_approval'
  
  // Human-in-the-loop approval
  requiresApproval: boolean("requires_approval").default(false),
  approvalStatus: varchar("approval_status"), // 'pending', 'approved', 'rejected'
  approvedBy: varchar("approved_by").references(() => users.id),
  approvedAt: timestamp("approved_at"),
  approvalNotes: text("approval_notes"),
  
  // Delivery tracking
  delivered: boolean("delivered").default(false),
  deliveredAt: timestamp("delivered_at"),
  readAt: timestamp("read_at"),
  
  // PHI and HIPAA tracking
  containsPhi: boolean("contains_phi").default(false),
  phiRedacted: boolean("phi_redacted").default(false),
  
  // Metadata
  metadata: jsonb("metadata").$type<{
    clientInfo?: { platform?: string; version?: string };
    processingTime?: number;
    modelUsed?: string;
    tokensUsed?: number;
  }>(),
  
  createdAt: timestamp("created_at").defaultNow(),
}, (table) => ({
  conversationIdx: index("msg_conversation_idx").on(table.conversationId),
  fromIdx: index("msg_from_idx").on(table.fromType, table.fromId),
  senderRoleIdx: index("msg_sender_role_idx").on(table.senderRole),
  msgIdIdx: index("msg_id_idx").on(table.msgId),
  createdIdx: index("msg_created_idx").on(table.createdAt),
  toolCallIdx: index("msg_tool_call_idx").on(table.toolCallId),
  approvalIdx: index("msg_approval_idx").on(table.requiresApproval, table.approvalStatus),
}));

export const insertAgentMessageSchema = createInsertSchema(agentMessages).omit({
  id: true,
  createdAt: true,
});

export type InsertAgentMessage = z.infer<typeof insertAgentMessageSchema>;
export type AgentMessage = typeof agentMessages.$inferSelect;

// Agent tasks (background jobs, scheduled workflows)
export const agentTasks = pgTable("agent_tasks", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  
  // Task ownership
  agentId: varchar("agent_id").notNull().references(() => agents.id),
  userId: varchar("user_id").references(() => users.id), // User context if applicable
  conversationId: varchar("conversation_id").references(() => agentConversations.id),
  messageId: varchar("message_id").references(() => agentMessages.id),
  
  // Task definition
  taskType: varchar("task_type").notNull(), // 'tool_execution', 'scheduled_checkin', 'medication_reminder', 'workflow', 'inference'
  taskName: varchar("task_name"),
  
  // Scheduling
  scheduledAt: timestamp("scheduled_at"),
  recurringPattern: varchar("recurring_pattern"), // Cron expression for recurring tasks
  timezone: varchar("timezone").default("UTC"),
  
  // Execution
  status: varchar("status").notNull().default("pending"), // 'pending', 'queued', 'running', 'completed', 'failed', 'cancelled'
  priority: integer("priority").default(5), // 1-10, higher is more important
  
  // Payload and results
  inputPayload: jsonb("input_payload").$type<Record<string, unknown>>(),
  outputResult: jsonb("output_result").$type<Record<string, unknown>>(),
  errorMessage: text("error_message"),
  
  // Retry logic
  attempts: integer("attempts").default(0),
  maxAttempts: integer("max_attempts").default(3),
  lastAttemptAt: timestamp("last_attempt_at"),
  nextRetryAt: timestamp("next_retry_at"),
  
  // Worker info
  workerId: varchar("worker_id"),
  startedAt: timestamp("started_at"),
  completedAt: timestamp("completed_at"),
  
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
}, (table) => ({
  agentIdx: index("task_agent_idx").on(table.agentId),
  userIdx: index("task_user_idx").on(table.userId),
  statusIdx: index("task_status_idx").on(table.status),
  scheduledIdx: index("task_scheduled_idx").on(table.scheduledAt),
  typeIdx: index("task_type_idx").on(table.taskType),
}));

export const insertAgentTaskSchema = createInsertSchema(agentTasks).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export type InsertAgentTask = z.infer<typeof insertAgentTaskSchema>;
export type AgentTask = typeof agentTasks.$inferSelect;

// Agent memory (short-term and long-term with vector embeddings)
export const agentMemory = pgTable("agent_memory", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  
  // Memory ownership
  agentId: varchar("agent_id").notNull().references(() => agents.id),
  patientId: varchar("patient_id").references(() => users.id),
  userId: varchar("user_id").references(() => users.id),
  conversationId: varchar("conversation_id").references(() => agentConversations.id),
  
  // Memory type and storage
  memoryType: varchar("memory_type").notNull(), // 'short_term', 'long_term', 'episodic', 'semantic'
  storageType: varchar("storage_type").notNull(), // 'redis', 'postgres', 'vector'
  
  // Content
  content: text("content").notNull(),
  summary: text("summary"), // LLM-generated summary for long-term storage
  
  // Vector embedding for semantic search (stored as JSON array, to be used with pgvector)
  embedding: jsonb("embedding").$type<number[]>(),
  embeddingModel: varchar("embedding_model"),
  
  // Source tracking
  sourceType: varchar("source_type"), // 'message', 'tool_result', 'observation', 'inference'
  sourceId: varchar("source_id"),
  
  // Salience and importance
  importance: decimal("importance", { precision: 3, scale: 2 }).default("0.5"), // 0-1 scale
  accessCount: integer("access_count").default(0),
  lastAccessedAt: timestamp("last_accessed_at"),
  
  // TTL for short-term memory
  expiresAt: timestamp("expires_at"),
  
  // Metadata
  metadata: jsonb("metadata").$type<{
    topics?: string[];
    entities?: string[];
    sentiment?: string;
    keywords?: string[];
  }>(),
  
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
}, (table) => ({
  agentIdx: index("memory_agent_idx").on(table.agentId),
  patientIdx: index("memory_patient_idx").on(table.patientId),
  typeIdx: index("memory_type_idx").on(table.memoryType),
  expiresIdx: index("memory_expires_idx").on(table.expiresAt),
  importanceIdx: index("memory_importance_idx").on(table.importance),
}));

export const insertAgentMemorySchema = createInsertSchema(agentMemory).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export type InsertAgentMemory = z.infer<typeof insertAgentMemorySchema>;
export type AgentMemory = typeof agentMemory.$inferSelect;

// Agent audit logs (HIPAA-compliant comprehensive logging)
export const agentAuditLogs = pgTable("agent_audit_logs", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  
  // Actor (who performed the action)
  actorType: varchar("actor_type").notNull(), // 'user', 'agent', 'system', 'worker'
  actorId: varchar("actor_id").notNull(),
  actorRole: varchar("actor_role"), // 'doctor', 'patient', 'admin'
  
  // Action details
  action: varchar("action").notNull(), // 'message_sent', 'message_read', 'tool_called', 'phi_accessed', 'approval_granted', etc.
  actionCategory: varchar("action_category").notNull(), // 'communication', 'data_access', 'clinical_action', 'system'
  
  // Object (what was acted upon)
  objectType: varchar("object_type").notNull(), // 'message', 'conversation', 'patient_record', 'prescription', 'tool'
  objectId: varchar("object_id").notNull(),
  
  // Context
  patientId: varchar("patient_id").references(() => users.id), // If action involves a patient
  conversationId: varchar("conversation_id").references(() => agentConversations.id),
  messageId: varchar("message_id").references(() => agentMessages.id),
  
  // PHI access tracking (HIPAA requirement)
  phiAccessed: boolean("phi_accessed").default(false),
  phiCategories: jsonb("phi_categories").$type<string[]>(), // Types of PHI accessed
  accessReason: varchar("access_reason"), // Reason for PHI access
  
  // Details
  details: jsonb("details").$type<Record<string, unknown>>(),
  previousState: jsonb("previous_state").$type<Record<string, unknown>>(),
  newState: jsonb("new_state").$type<Record<string, unknown>>(),
  
  // Request context
  ipAddress: varchar("ip_address"),
  userAgent: text("user_agent"),
  sessionId: varchar("session_id"),
  
  // Outcome
  success: boolean("success").default(true),
  errorCode: varchar("error_code"),
  errorMessage: text("error_message"),
  
  // Timestamps
  timestamp: timestamp("timestamp").notNull().defaultNow(),
  createdAt: timestamp("created_at").defaultNow(),
}, (table) => ({
  actorIdx: index("audit_actor_idx").on(table.actorType, table.actorId),
  actionIdx: index("audit_action_idx").on(table.action),
  objectIdx: index("audit_object_idx").on(table.objectType, table.objectId),
  patientIdx: index("audit_patient_idx").on(table.patientId),
  timestampIdx: index("audit_timestamp_idx").on(table.timestamp),
  phiIdx: index("audit_phi_idx").on(table.phiAccessed),
}));

export const insertAgentAuditLogSchema = createInsertSchema(agentAuditLogs).omit({
  id: true,
  createdAt: true,
});

export type InsertAgentAuditLog = z.infer<typeof insertAgentAuditLogSchema>;
export type AgentAuditLog = typeof agentAuditLogs.$inferSelect;

// User presence tracking for real-time status
export const userPresence = pgTable("user_presence", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  userId: varchar("user_id").notNull().references(() => users.id).unique(),
  
  // Connection status
  isOnline: boolean("is_online").default(false),
  lastSeenAt: timestamp("last_seen_at"),
  
  // Active connections
  activeConnections: integer("active_connections").default(0),
  connectionIds: jsonb("connection_ids").$type<string[]>(),
  
  // Activity
  currentActivity: varchar("current_activity"), // 'typing', 'viewing', 'idle'
  currentConversationId: varchar("current_conversation_id"),
  
  // Device info
  lastDeviceInfo: jsonb("last_device_info").$type<{
    platform?: string;
    browser?: string;
    isMobile?: boolean;
  }>(),
  
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
}, (table) => ({
  userIdx: index("presence_user_idx").on(table.userId),
  onlineIdx: index("presence_online_idx").on(table.isOnline),
}));

export const insertUserPresenceSchema = createInsertSchema(userPresence).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export type InsertUserPresence = z.infer<typeof insertUserPresenceSchema>;
export type UserPresence = typeof userPresence.$inferSelect;

// Tool execution tracking
export const toolExecutions = pgTable("tool_executions", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  
  // Execution context
  agentId: varchar("agent_id").notNull(),
  userId: varchar("user_id").notNull().references(() => users.id),
  conversationId: varchar("conversation_id").references(() => agentConversations.id),
  messageId: varchar("message_id").references(() => agentMessages.id),
  
  // Tool info
  toolName: varchar("tool_name").notNull(),
  toolVersion: integer("tool_version").default(1),
  
  // Input/Output
  inputParameters: jsonb("input_parameters").$type<Record<string, unknown>>().notNull(),
  outputResult: jsonb("output_result").$type<Record<string, unknown>>(),
  
  // Status tracking
  status: varchar("status").notNull().default("pending"), // 'pending', 'running', 'completed', 'failed', 'pending_approval', 'approved', 'rejected'
  errorMessage: text("error_message"),
  errorCode: varchar("error_code"),
  
  // Performance metrics
  executionTimeMs: integer("execution_time_ms"),
  startedAt: timestamp("started_at"),
  completedAt: timestamp("completed_at"),
  
  // Patient context (for consent verification)
  patientId: varchar("patient_id").references(() => users.id),
  doctorId: varchar("doctor_id").references(() => users.id),
  
  // PHI tracking
  phiAccessed: boolean("phi_accessed").default(false),
  phiCategories: jsonb("phi_categories").$type<string[]>(),
  
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
}, (table) => ({
  agentIdx: index("tool_exec_agent_idx").on(table.agentId),
  userIdx: index("tool_exec_user_idx").on(table.userId),
  toolIdx: index("tool_exec_tool_idx").on(table.toolName),
  statusIdx: index("tool_exec_status_idx").on(table.status),
  patientIdx: index("tool_exec_patient_idx").on(table.patientId),
  createdIdx: index("tool_exec_created_idx").on(table.createdAt),
}));

export const insertToolExecutionSchema = createInsertSchema(toolExecutions).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export type InsertToolExecution = z.infer<typeof insertToolExecutionSchema>;
export type ToolExecution = typeof toolExecutions.$inferSelect;

// Approval queue for human-in-the-loop workflows
export const approvalQueue = pgTable("approval_queue", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  
  // Request details
  requestType: varchar("request_type").notNull(), // 'prescription_draft', 'treatment_change', 'data_export', 'patient_discharge'
  requesterId: varchar("requester_id").notNull(), // User or agent who initiated
  requesterType: varchar("requester_type").notNull(), // 'agent', 'user', 'system'
  
  // Approval target
  approverId: varchar("approver_id").references(() => users.id), // Specific approver if assigned
  approverRole: varchar("approver_role").notNull().default("doctor"), // Role that can approve
  
  // Context
  patientId: varchar("patient_id").references(() => users.id),
  conversationId: varchar("conversation_id").references(() => agentConversations.id),
  messageId: varchar("message_id").references(() => agentMessages.id),
  toolExecutionId: varchar("tool_execution_id").references(() => toolExecutions.id),
  
  // Request payload
  toolName: varchar("tool_name"),
  requestPayload: jsonb("request_payload").$type<Record<string, unknown>>().notNull(),
  requestSummary: text("request_summary"), // Human-readable summary
  
  // Risk assessment
  urgency: varchar("urgency").notNull().default("normal"), // 'low', 'normal', 'high', 'urgent'
  riskLevel: varchar("risk_level"), // 'low', 'medium', 'high'
  riskFactors: jsonb("risk_factors").$type<string[]>(),
  
  // Status tracking
  status: varchar("status").notNull().default("pending"), // 'pending', 'approved', 'rejected', 'expired', 'cancelled'
  
  // Decision
  decision: varchar("decision"), // 'approved', 'rejected', 'modified'
  decisionBy: varchar("decision_by").references(() => users.id),
  decisionAt: timestamp("decision_at"),
  decisionNotes: text("decision_notes"),
  modifiedPayload: jsonb("modified_payload").$type<Record<string, unknown>>(), // If modified before approval
  
  // Timeout
  expiresAt: timestamp("expires_at"),
  reminderSentAt: timestamp("reminder_sent_at"),
  escalatedAt: timestamp("escalated_at"),
  escalatedTo: varchar("escalated_to").references(() => users.id),
  
  // Execution result
  executionResult: jsonb("execution_result").$type<Record<string, unknown>>(),
  executedAt: timestamp("executed_at"),
  
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
}, (table) => ({
  requesterIdx: index("approval_requester_idx").on(table.requesterId, table.requesterType),
  approverIdx: index("approval_approver_idx").on(table.approverId),
  roleIdx: index("approval_role_idx").on(table.approverRole),
  patientIdx: index("approval_patient_idx").on(table.patientId),
  statusIdx: index("approval_status_idx").on(table.status),
  urgencyIdx: index("approval_urgency_idx").on(table.urgency),
  expiresIdx: index("approval_expires_idx").on(table.expiresAt),
  createdIdx: index("approval_created_idx").on(table.createdAt),
}));

export const insertApprovalQueueSchema = createInsertSchema(approvalQueue).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export type InsertApprovalQueue = z.infer<typeof insertApprovalQueueSchema>;
export type ApprovalQueue = typeof approvalQueue.$inferSelect;

// ============================================================================
// ML Training System Tables
// ============================================================================

// ML Training Consent - Patient's consent for using their data in ML training
export const mlTrainingConsent = pgTable("ml_training_consent", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  patientId: varchar("patient_id").notNull().references(() => users.id),
  
  // Overall consent status
  consentEnabled: boolean("consent_enabled").notNull().default(false),
  
  // Granular data type permissions
  dataTypes: jsonb("data_types").$type<{
    vitals: boolean;
    symptoms: boolean;
    medications: boolean;
    mentalHealth: boolean;
    behavioralData: boolean;
    wearableData: boolean;
    labResults: boolean;
    imagingData: boolean;
  }>().default({
    vitals: true,
    symptoms: true,
    medications: false,
    mentalHealth: false,
    behavioralData: true,
    wearableData: true,
    labResults: false,
    imagingData: false,
  }),
  
  // Data anonymization preferences
  anonymizationLevel: varchar("anonymization_level").default("full"), // 'full', 'partial', 'minimal'
  
  // Consent tracking
  consentVersion: varchar("consent_version").default("1.0"),
  consentSignedAt: timestamp("consent_signed_at"),
  consentWithdrawnAt: timestamp("consent_withdrawn_at"),
  withdrawalReason: text("withdrawal_reason"),
  
  // Data deletion request
  requestedDataDeletion: boolean("requested_data_deletion").default(false),
  dataDeletionRequestedAt: timestamp("data_deletion_requested_at"),
  dataDeletionCompletedAt: timestamp("data_deletion_completed_at"),
  
  // Audit trail
  lastModifiedBy: varchar("last_modified_by"),
  ipAddress: varchar("ip_address"),
  userAgent: text("user_agent"),
  
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
}, (table) => ({
  patientIdx: index("ml_consent_patient_idx").on(table.patientId),
  enabledIdx: index("ml_consent_enabled_idx").on(table.consentEnabled),
}));

export const insertMlTrainingConsentSchema = createInsertSchema(mlTrainingConsent).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export type InsertMlTrainingConsent = z.infer<typeof insertMlTrainingConsentSchema>;
export type MlTrainingConsent = typeof mlTrainingConsent.$inferSelect;

// ML Models Registry - Track all trained models with versioning
export const mlModels = pgTable("ml_models", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  
  // Model identification
  modelName: varchar("model_name").notNull(), // 'vital_lstm', 'patient_segmentation', 'deterioration_ensemble', 'disease_risk'
  modelType: varchar("model_type").notNull(), // 'lstm', 'kmeans', 'random_forest', 'gradient_boosting', 'ensemble'
  version: varchar("version").notNull(), // Semantic versioning: '1.0.0'
  
  // Model status
  status: varchar("status").notNull().default("training"), // 'training', 'evaluating', 'active', 'deprecated', 'failed'
  isActive: boolean("is_active").default(false),
  
  // Training configuration
  trainingConfig: jsonb("training_config").$type<{
    epochs?: number;
    batchSize?: number;
    learningRate?: number;
    layers?: number[];
    dropout?: number;
    optimizer?: string;
    lossFunction?: string;
    earlyStopping?: boolean;
    nClusters?: number; // For K-Means
    maxIterations?: number;
    nEstimators?: number; // For ensemble
    maxDepth?: number;
  }>(),
  
  // Training data info
  trainingDataSources: jsonb("training_data_sources").$type<{
    publicDatasets: string[];
    patientDataCount: number;
    totalRecords: number;
    dateRange: { start: string; end: string };
  }>(),
  
  // Performance metrics
  metrics: jsonb("metrics").$type<{
    accuracy?: number;
    precision?: number;
    recall?: number;
    f1Score?: number;
    auc?: number;
    rmse?: number;
    mae?: number;
    silhouetteScore?: number;
    inertia?: number;
    r2Score?: number;
    calibrationError?: number;
  }>(),
  
  // Model artifacts
  modelPath: varchar("model_path"), // Path to saved model file
  modelFormat: varchar("model_format").default("onnx"), // 'onnx', 'pytorch', 'joblib', 'tensorflow'
  modelSizeBytes: integer("model_size_bytes"),
  
  // Feature information
  featureNames: jsonb("feature_names").$type<string[]>(),
  featureImportance: jsonb("feature_importance").$type<Record<string, number>>(),
  
  // Training metadata
  trainedBy: varchar("trained_by"), // 'system', 'admin', user_id
  trainingStartedAt: timestamp("training_started_at"),
  trainingCompletedAt: timestamp("training_completed_at"),
  trainingDurationSeconds: integer("training_duration_seconds"),
  
  // Deployment info
  deployedAt: timestamp("deployed_at"),
  deprecatedAt: timestamp("deprecated_at"),
  deprecationReason: text("deprecation_reason"),
  
  // Comparison with previous version
  previousVersionId: varchar("previous_version_id"),
  improvementOverPrevious: jsonb("improvement_over_previous").$type<Record<string, number>>(),
  
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
}, (table) => ({
  modelNameIdx: index("ml_models_name_idx").on(table.modelName),
  versionIdx: index("ml_models_version_idx").on(table.modelName, table.version),
  statusIdx: index("ml_models_status_idx").on(table.status),
  activeIdx: index("ml_models_active_idx").on(table.isActive),
}));

export const insertMlModelSchema = createInsertSchema(mlModels).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export type InsertMlModel = z.infer<typeof insertMlModelSchema>;
export type MlModel = typeof mlModels.$inferSelect;

// ML Training Jobs - Track training job execution
export const mlTrainingJobs = pgTable("ml_training_jobs", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  
  // Job identification
  jobName: varchar("job_name").notNull(),
  modelName: varchar("model_name").notNull(),
  targetVersion: varchar("target_version").notNull(),
  
  // Job status
  status: varchar("status").notNull().default("queued"), // 'queued', 'preparing', 'training', 'evaluating', 'completed', 'failed', 'cancelled'
  priority: integer("priority").default(5), // 1-10, higher = more priority
  
  // Data sources
  dataSources: jsonb("data_sources").$type<{
    publicDatasets: Array<{
      id: string;
      name: string;
      source: string;
      recordCount: number;
    }>;
    patientData: {
      enabled: boolean;
      patientCount: number;
      recordCount: number;
      dataTypes: string[];
    };
  }>(),
  
  // Training configuration
  trainingConfig: jsonb("training_config").$type<Record<string, unknown>>(),
  
  // Progress tracking
  currentPhase: varchar("current_phase"), // 'data_loading', 'preprocessing', 'feature_engineering', 'training', 'evaluation', 'saving'
  progressPercent: integer("progress_percent").default(0),
  progressMessage: text("progress_message"),
  
  // Resource usage
  resourceUsage: jsonb("resource_usage").$type<{
    cpuPercent?: number;
    memoryMb?: number;
    gpuPercent?: number;
    gpuMemoryMb?: number;
  }>(),
  
  // Timing
  queuedAt: timestamp("queued_at").defaultNow(),
  startedAt: timestamp("started_at"),
  completedAt: timestamp("completed_at"),
  estimatedCompletionAt: timestamp("estimated_completion_at"),
  
  // Results
  resultModelId: varchar("result_model_id").references(() => mlModels.id),
  errorMessage: text("error_message"),
  errorDetails: jsonb("error_details").$type<{
    phase: string;
    stackTrace?: string;
    recoverable: boolean;
  }>(),
  
  // Notifications
  notifyOnCompletion: boolean("notify_on_completion").default(true),
  notificationEmail: varchar("notification_email"),
  
  // Initiated by
  initiatedBy: varchar("initiated_by"), // 'system', 'scheduler', user_id
  
  // Logs path
  logsPath: varchar("logs_path"),
  
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
}, (table) => ({
  statusIdx: index("ml_jobs_status_idx").on(table.status),
  modelIdx: index("ml_jobs_model_idx").on(table.modelName),
  priorityIdx: index("ml_jobs_priority_idx").on(table.priority, table.queuedAt),
}));

export const insertMlTrainingJobSchema = createInsertSchema(mlTrainingJobs).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export type InsertMlTrainingJob = z.infer<typeof insertMlTrainingJobSchema>;
export type MlTrainingJob = typeof mlTrainingJobs.$inferSelect;

// ML Training Contributions - Track patient data contributions (anonymized)
export const mlTrainingContributions = pgTable("ml_training_contributions", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  
  // Patient reference (hashed for privacy)
  patientIdHash: varchar("patient_id_hash").notNull(), // SHA-256 hash of patient ID
  consentId: varchar("consent_id").notNull().references(() => mlTrainingConsent.id),
  
  // Contribution details
  trainingJobId: varchar("training_job_id").references(() => mlTrainingJobs.id),
  modelId: varchar("model_id").references(() => mlModels.id),
  
  // Data contributed (counts only, no actual data)
  dataTypesContributed: jsonb("data_types_contributed").$type<string[]>(),
  recordCount: integer("record_count").notNull(),
  dateRangeStart: timestamp("date_range_start"),
  dateRangeEnd: timestamp("date_range_end"),
  
  // Anonymization applied
  anonymizationLevel: varchar("anonymization_level").notNull(),
  
  // Contribution status
  status: varchar("status").default("included"), // 'included', 'excluded', 'withdrawn'
  
  // Timestamp
  contributedAt: timestamp("contributed_at").defaultNow(),
  withdrawnAt: timestamp("withdrawn_at"),
  
  createdAt: timestamp("created_at").defaultNow(),
}, (table) => ({
  patientHashIdx: index("ml_contrib_patient_idx").on(table.patientIdHash),
  jobIdx: index("ml_contrib_job_idx").on(table.trainingJobId),
  modelIdx: index("ml_contrib_model_idx").on(table.modelId),
}));

export const insertMlTrainingContributionSchema = createInsertSchema(mlTrainingContributions).omit({
  id: true,
  createdAt: true,
});

export type InsertMlTrainingContribution = z.infer<typeof insertMlTrainingContributionSchema>;
export type MlTrainingContribution = typeof mlTrainingContributions.$inferSelect;

// Public Dataset Registry - Enhanced tracking for MIMIC-III, PhysioNet, etc.
export const publicDatasetRegistry = pgTable("public_dataset_registry", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  
  // Dataset identification
  datasetName: varchar("dataset_name").notNull().unique(),
  source: varchar("source").notNull(), // 'mimic_iii', 'mimic_iv', 'physionet', 'eicu', 'kaggle'
  version: varchar("version"),
  
  // Access information
  requiresCredentials: boolean("requires_credentials").default(true),
  credentialsConfigured: boolean("credentials_configured").default(false),
  accessUrl: varchar("access_url"),
  documentationUrl: varchar("documentation_url"),
  
  // Dataset metadata
  description: text("description"),
  recordCount: integer("record_count"),
  patientCount: integer("patient_count"),
  dateRange: jsonb("date_range").$type<{ start: string; end: string }>(),
  
  // Data types available
  dataTypes: jsonb("data_types").$type<string[]>(), // ['vitals', 'lab_results', 'medications', 'diagnoses', 'procedures']
  
  // File information
  totalSizeGb: decimal("total_size_gb", { precision: 10, scale: 2 }),
  fileFormats: jsonb("file_formats").$type<string[]>(), // ['csv', 'parquet', 'json']
  
  // Download/processing status
  downloadStatus: varchar("download_status").default("not_started"), // 'not_started', 'downloading', 'downloaded', 'processing', 'ready', 'failed'
  downloadProgress: integer("download_progress").default(0),
  localPath: varchar("local_path"),
  
  // Processing metadata
  preprocessedAt: timestamp("preprocessed_at"),
  preprocessingConfig: jsonb("preprocessing_config").$type<Record<string, unknown>>(),
  
  // Usage tracking
  usedInModels: jsonb("used_in_models").$type<string[]>(),
  lastUsedAt: timestamp("last_used_at"),
  
  // License information
  license: varchar("license"),
  citationRequired: boolean("citation_required").default(true),
  citation: text("citation"),
  
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
}, (table) => ({
  sourceIdx: index("public_dataset_source_idx").on(table.source),
  statusIdx: index("public_dataset_status_idx").on(table.downloadStatus),
}));

export const insertPublicDatasetRegistrySchema = createInsertSchema(publicDatasetRegistry).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export type InsertPublicDatasetRegistry = z.infer<typeof insertPublicDatasetRegistrySchema>;
export type PublicDatasetRegistry = typeof publicDatasetRegistry.$inferSelect;

// ML Training Audit Log - HIPAA-compliant audit trail for all ML operations
export const mlTrainingAuditLog = pgTable("ml_training_audit_log", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  
  // Event identification
  eventType: varchar("event_type").notNull(), // 'consent_granted', 'consent_withdrawn', 'data_extracted', 'training_started', 'model_deployed', 'data_deleted'
  eventCategory: varchar("event_category").notNull(), // 'consent', 'training', 'model', 'data_access'
  
  // Actor
  actorId: varchar("actor_id"), // User ID or 'system'
  actorType: varchar("actor_type").notNull(), // 'patient', 'admin', 'system', 'scheduler'
  
  // Resource affected
  resourceType: varchar("resource_type").notNull(), // 'consent', 'training_job', 'model', 'patient_data'
  resourceId: varchar("resource_id"),
  
  // Patient data involvement (PHI tracking)
  patientIdHash: varchar("patient_id_hash"), // Hashed patient ID if patient data involved
  phiAccessed: boolean("phi_accessed").default(false),
  phiCategories: jsonb("phi_categories").$type<string[]>(),
  
  // Event details
  eventDetails: jsonb("event_details").$type<Record<string, unknown>>(),
  previousState: jsonb("previous_state").$type<Record<string, unknown>>(),
  newState: jsonb("new_state").$type<Record<string, unknown>>(),
  
  // Request metadata
  ipAddress: varchar("ip_address"),
  userAgent: text("user_agent"),
  sessionId: varchar("session_id"),
  
  // Status
  success: boolean("success").default(true),
  errorMessage: text("error_message"),
  
  createdAt: timestamp("created_at").defaultNow(),
}, (table) => ({
  eventTypeIdx: index("ml_audit_event_idx").on(table.eventType),
  actorIdx: index("ml_audit_actor_idx").on(table.actorId),
  resourceIdx: index("ml_audit_resource_idx").on(table.resourceType, table.resourceId),
  patientIdx: index("ml_audit_patient_idx").on(table.patientIdHash),
  createdIdx: index("ml_audit_created_idx").on(table.createdAt),
}));

export const insertMlTrainingAuditLogSchema = createInsertSchema(mlTrainingAuditLog).omit({
  id: true,
  createdAt: true,
});

export type InsertMlTrainingAuditLog = z.infer<typeof insertMlTrainingAuditLogSchema>;
export type MlTrainingAuditLog = typeof mlTrainingAuditLog.$inferSelect;

// ============================================
// UNIVERSAL DEVICE DATA COLLECTION SYSTEM
// Supports: BP Monitor, Glucose Meter, Smart Scale, Thermometer, Stethoscope, Smartwatch (Whoop/Garmin/Apple/Oura/Samsung/Google)
// ============================================

// Device Readings - Universal table for all medical device data
export const deviceReadings = pgTable("device_readings", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  patientId: varchar("patient_id").notNull().references(() => users.id),
  
  // Device identification
  deviceType: varchar("device_type").notNull(), // 'bp_monitor', 'glucose_meter', 'smart_scale', 'thermometer', 'stethoscope', 'smartwatch'
  deviceBrand: varchar("device_brand"), // 'whoop', 'garmin', 'apple_watch', 'oura', 'samsung', 'google', 'fitbit', 'omron', 'freestyle', 'withings', 'manual'
  deviceModel: varchar("device_model"), // Specific model name
  
  // Data source
  source: varchar("source").notNull().default("manual"), // 'manual', 'auto_sync', 'api_import'
  wearableIntegrationId: varchar("wearable_integration_id").references(() => wearableIntegrations.id),
  
  // Timestamp
  recordedAt: timestamp("recorded_at").notNull().defaultNow(),
  
  // Blood Pressure Monitor Metrics
  bpSystolic: integer("bp_systolic"), // mmHg
  bpDiastolic: integer("bp_diastolic"), // mmHg
  bpPulse: integer("bp_pulse"), // bpm
  bpIrregularHeartbeat: boolean("bp_irregular_heartbeat"),
  bpBodyPosition: varchar("bp_body_position"), // 'sitting', 'standing', 'lying'
  bpArmUsed: varchar("bp_arm_used"), // 'left', 'right'
  
  // Glucose Meter Metrics
  glucoseValue: decimal("glucose_value", { precision: 5, scale: 1 }), // mg/dL
  glucoseContext: varchar("glucose_context"), // 'fasting', 'before_meal', 'after_meal', 'bedtime', 'random'
  glucoseUnit: varchar("glucose_unit").default("mg/dL"), // 'mg/dL', 'mmol/L'
  
  // Smart Scale Metrics
  weight: decimal("weight", { precision: 5, scale: 2 }), // kg or lbs
  weightUnit: varchar("weight_unit").default("kg"), // 'kg', 'lbs'
  bmi: decimal("bmi", { precision: 4, scale: 1 }),
  bodyFatPercentage: decimal("body_fat_percentage", { precision: 4, scale: 1 }),
  muscleMass: decimal("muscle_mass", { precision: 5, scale: 2 }),
  boneMass: decimal("bone_mass", { precision: 4, scale: 2 }),
  waterPercentage: decimal("water_percentage", { precision: 4, scale: 1 }),
  visceralFat: integer("visceral_fat"), // 1-59 scale
  metabolicAge: integer("metabolic_age"),
  
  // Thermometer Metrics
  temperature: decimal("temperature", { precision: 4, scale: 1 }), // Celsius or Fahrenheit
  temperatureUnit: varchar("temperature_unit").default("F"), // 'C', 'F'
  temperatureLocation: varchar("temperature_location"), // 'oral', 'ear', 'forehead', 'armpit', 'rectal'
  
  // Stethoscope Metrics (Digital)
  stethoscopeAudioUrl: varchar("stethoscope_audio_url"), // S3 URL for audio recording
  stethoscopeLocation: varchar("stethoscope_location"), // 'heart', 'lungs_left', 'lungs_right', 'abdomen'
  heartSoundsAnalysis: jsonb("heart_sounds_analysis").$type<{
    murmurs?: boolean;
    abnormalRhythm?: boolean;
    s3Sound?: boolean;
    s4Sound?: boolean;
    notes?: string;
  }>(),
  lungSoundsAnalysis: jsonb("lung_sounds_analysis").$type<{
    wheezing?: boolean;
    crackles?: boolean;
    rhonchi?: boolean;
    diminished?: boolean;
    notes?: string;
  }>(),
  
  // Smartwatch - Heart/Cardiovascular Metrics
  heartRate: integer("heart_rate"), // bpm
  restingHeartRate: integer("resting_heart_rate"), // bpm
  hrv: integer("hrv"), // Heart Rate Variability (ms RMSSD)
  hrvSdnn: integer("hrv_sdnn"), // SDNN in ms
  ecgData: jsonb("ecg_data").$type<{
    classification?: string; // 'sinus_rhythm', 'afib', 'inconclusive'
    heartRateDuringEcg?: number;
    recordingUrl?: string;
  }>(),
  afibDetected: boolean("afib_detected"),
  irregularRhythmAlert: boolean("irregular_rhythm_alert"),
  heartRateZones: jsonb("heart_rate_zones").$type<{
    resting?: number;
    fatBurn?: number;
    cardio?: number;
    peak?: number;
  }>(),
  
  // Smartwatch - Blood Oxygen / Respiratory
  spo2: integer("spo2"), // Blood oxygen percentage
  spo2Min: integer("spo2_min"),
  respiratoryRate: integer("respiratory_rate"), // Breaths per minute
  
  // Smartwatch - Sleep Metrics
  sleepDuration: integer("sleep_duration"), // Total minutes
  sleepDeepMinutes: integer("sleep_deep_minutes"),
  sleepRemMinutes: integer("sleep_rem_minutes"),
  sleepLightMinutes: integer("sleep_light_minutes"),
  sleepAwakeMinutes: integer("sleep_awake_minutes"),
  sleepScore: integer("sleep_score"), // 0-100
  sleepEfficiency: decimal("sleep_efficiency", { precision: 4, scale: 1 }), // Percentage
  sleepConsistency: integer("sleep_consistency"), // 0-100
  sleepDebt: integer("sleep_debt"), // Minutes of sleep debt
  sleepNeed: integer("sleep_need"), // Predicted sleep need in minutes
  
  // Smartwatch - Recovery/Readiness Metrics (Whoop, Oura, Garmin)
  recoveryScore: integer("recovery_score"), // 0-100 (Whoop Recovery, Oura Readiness)
  readinessScore: integer("readiness_score"), // 0-100
  bodyBattery: integer("body_battery"), // 0-100 (Garmin)
  strainScore: decimal("strain_score", { precision: 3, scale: 1 }), // 0-21 (Whoop Strain)
  stressScore: integer("stress_score"), // 0-100
  
  // Smartwatch - Temperature
  skinTemperature: decimal("skin_temperature", { precision: 4, scale: 2 }), // Deviation from baseline
  skinTempUnit: varchar("skin_temp_unit").default("C"),
  
  // Smartwatch - Activity/Fitness Metrics
  steps: integer("steps"),
  activeMinutes: integer("active_minutes"),
  caloriesBurned: integer("calories_burned"),
  distanceMeters: integer("distance_meters"),
  floorsClimbed: integer("floors_climbed"),
  standingHours: integer("standing_hours"),
  vo2Max: decimal("vo2_max", { precision: 4, scale: 1 }), // mL/kg/min
  trainingLoad: integer("training_load"), // Garmin Training Load
  trainingStatus: varchar("training_status"), // 'productive', 'peaking', 'unproductive', 'recovery', 'detraining'
  trainingReadiness: integer("training_readiness"), // 0-100
  fitnessAge: integer("fitness_age"),
  
  // Smartwatch - Running/Sport Specific (Garmin)
  runningDynamics: jsonb("running_dynamics").$type<{
    cadence?: number;
    strideLength?: number;
    groundContactTime?: number;
    verticalOscillation?: number;
    groundContactBalance?: number;
  }>(),
  lactateThreshold: integer("lactate_threshold"), // bpm
  performanceCondition: integer("performance_condition"), // -20 to +20
  
  // Smartwatch - Women's Health
  cycleDay: integer("cycle_day"),
  cyclePhase: varchar("cycle_phase"), // 'menstrual', 'follicular', 'ovulation', 'luteal'
  predictedOvulation: timestamp("predicted_ovulation"),
  periodLogged: boolean("period_logged"),
  
  // Smartwatch - Safety/Emergency
  fallDetected: boolean("fall_detected"),
  fallTimestamp: timestamp("fall_timestamp"),
  emergencySOSTriggered: boolean("emergency_sos_triggered"),
  
  // Data routing flags (which health section this data should appear in)
  routeToHypertension: boolean("route_to_hypertension").default(false),
  routeToDiabetes: boolean("route_to_diabetes").default(false),
  routeToCardiovascular: boolean("route_to_cardiovascular").default(false),
  routeToRespiratory: boolean("route_to_respiratory").default(false),
  routeToSleep: boolean("route_to_sleep").default(false),
  routeToMentalHealth: boolean("route_to_mental_health").default(false),
  routeToFitness: boolean("route_to_fitness").default(false),
  
  // ML/Health Alert integration
  processedForAlerts: boolean("processed_for_alerts").default(false),
  alertsGenerated: jsonb("alerts_generated").$type<string[]>(), // Array of alert IDs
  contributedToMlTraining: boolean("contributed_to_ml_training").default(false),
  
  // Notes and metadata
  notes: text("notes"),
  metadata: jsonb("metadata").$type<{
    syncBatchId?: string;
    firmwareVersion?: string;
    batteryLevel?: number;
    signalQuality?: string;
    rawData?: Record<string, unknown>;
  }>(),
  
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
}, (table) => ({
  patientDateIdx: index("device_readings_patient_date_idx").on(table.patientId, table.recordedAt),
  deviceTypeIdx: index("device_readings_device_type_idx").on(table.deviceType),
  sourceIdx: index("device_readings_source_idx").on(table.source),
}));

export const insertDeviceReadingSchema = createInsertSchema(deviceReadings).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export type InsertDeviceReading = z.infer<typeof insertDeviceReadingSchema>;
export type DeviceReading = typeof deviceReadings.$inferSelect;

// Device reading validation schemas for manual entry forms
export const bpReadingSchema = z.object({
  bpSystolic: z.number().min(60).max(250),
  bpDiastolic: z.number().min(30).max(150),
  bpPulse: z.number().min(30).max(220).optional(),
  bpBodyPosition: z.enum(['sitting', 'standing', 'lying']).optional(),
  bpArmUsed: z.enum(['left', 'right']).optional(),
  notes: z.string().optional(),
});

export const glucoseReadingSchema = z.object({
  glucoseValue: z.number().min(20).max(600),
  glucoseContext: z.enum(['fasting', 'before_meal', 'after_meal', 'bedtime', 'random']),
  glucoseUnit: z.enum(['mg/dL', 'mmol/L']).default('mg/dL'),
  notes: z.string().optional(),
});

export const weightReadingSchema = z.object({
  weight: z.number().min(20).max(500),
  weightUnit: z.enum(['kg', 'lbs']).default('kg'),
  bodyFatPercentage: z.number().min(1).max(70).optional(),
  notes: z.string().optional(),
});

export const temperatureReadingSchema = z.object({
  temperature: z.number().min(90).max(110), // Fahrenheit range
  temperatureUnit: z.enum(['C', 'F']).default('F'),
  temperatureLocation: z.enum(['oral', 'ear', 'forehead', 'armpit', 'rectal']).optional(),
  notes: z.string().optional(),
});

export const stethoscopeReadingSchema = z.object({
  stethoscopeLocation: z.enum(['heart', 'lungs_left', 'lungs_right', 'abdomen']),
  stethoscopeAudioUrl: z.string().optional(),
  heartSoundsAnalysis: z.object({
    murmurs: z.boolean().optional(),
    abnormalRhythm: z.boolean().optional(),
    s3Sound: z.boolean().optional(),
    s4Sound: z.boolean().optional(),
    notes: z.string().optional(),
  }).optional(),
  lungSoundsAnalysis: z.object({
    wheezing: z.boolean().optional(),
    crackles: z.boolean().optional(),
    rhonchi: z.boolean().optional(),
    diminished: z.boolean().optional(),
    notes: z.string().optional(),
  }).optional(),
  notes: z.string().optional(),
});

export const smartwatchReadingSchema = z.object({
  heartRate: z.number().min(30).max(220).optional(),
  restingHeartRate: z.number().min(30).max(120).optional(),
  hrv: z.number().min(0).max(300).optional(),
  spo2: z.number().min(70).max(100).optional(),
  respiratoryRate: z.number().min(6).max(40).optional(),
  sleepDuration: z.number().min(0).max(1440).optional(),
  sleepScore: z.number().min(0).max(100).optional(),
  recoveryScore: z.number().min(0).max(100).optional(),
  stressScore: z.number().min(0).max(100).optional(),
  steps: z.number().min(0).max(100000).optional(),
  caloriesBurned: z.number().min(0).max(10000).optional(),
  notes: z.string().optional(),
});

// Message envelope types for TypeScript
export const messageEnvelopeSchema = z.object({
  msgId: z.string().uuid(),
  from: z.object({
    type: z.enum(["agent", "user", "system"]),
    id: z.string(),
  }),
  to: z.array(z.object({
    type: z.enum(["agent", "user"]),
    id: z.string(),
  })),
  type: z.enum(["chat", "command", "event", "tool_call", "ack"]),
  timestamp: z.string().datetime(),
  payload: z.record(z.unknown()).optional(),
});

export type MessageEnvelope = z.infer<typeof messageEnvelopeSchema>;

// ============================================
// DEVICE CONNECT SYSTEM - Production Grade
// Supports: Smart BP, Glucose, Scale, Thermometer, Stethoscope, Smartwatch, Pulse Oximeter, Activity Trackers
// Vendors: Fitbit, Withings, Oura, Google Fit, iHealth (public), Dexcom, Garmin, Whoop, Samsung, Eko, Abbott (private)
// ============================================

// Device Models Catalog - Registry of all supported device types and capabilities
export const deviceModels = pgTable("device_models", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  
  // Device identification
  vendorId: varchar("vendor_id").notNull(), // 'fitbit', 'withings', 'garmin', 'whoop', 'oura', 'dexcom', 'omron', 'ihealth', 'eko', 'abbott', 'apple', 'google', 'samsung', 'generic_ble'
  vendorName: varchar("vendor_name").notNull(), // Display name
  modelName: varchar("model_name").notNull(), // e.g., "Charge 5", "BPM Connect", "G7"
  modelNumber: varchar("model_number"), // Manufacturer model number
  
  // Device type and category
  deviceType: varchar("device_type").notNull(), // 'smartwatch', 'bp_monitor', 'glucose_meter', 'scale', 'thermometer', 'stethoscope', 'pulse_oximeter', 'activity_tracker', 'cgm'
  deviceCategory: varchar("device_category").notNull(), // 'wearable', 'medical_device', 'home_health'
  
  // Capabilities - what metrics this device can provide
  capabilities: jsonb("capabilities").$type<{
    metrics: string[]; // ['heart_rate', 'hrv', 'spo2', 'sleep', 'steps', 'bp', 'glucose', 'weight', 'temperature', 'ecg', 'afib', 'respiratory_rate', 'stress', 'recovery', 'skin_temp']
    features: string[]; // ['continuous_monitoring', 'on_demand', 'auto_sync', 'manual_entry', 'audio_recording', 'ecg_recording']
    syncMethods: string[]; // ['oauth', 'ble', 'healthkit', 'google_fit', 'manual', 'webhook']
  }>().notNull(),
  
  // Pairing methods supported
  pairingMethods: jsonb("pairing_methods").$type<{
    oauth?: { authUrl: string; scopes: string[]; };
    ble?: { serviceUuids: string[]; characteristicUuids: string[]; };
    healthkit?: { dataTypes: string[]; };
    googleFit?: { dataTypes: string[]; };
    webhook?: { endpoint: string; };
    manual?: boolean;
  }>().notNull(),
  
  // Vendor API configuration (PHI: contains API endpoints)
  apiConfig: jsonb("api_config").$type<{
    baseUrl?: string;
    authEndpoint?: string;
    tokenEndpoint?: string;
    dataEndpoints?: Record<string, string>;
    webhookSecret?: string;
    rateLimit?: { requests: number; period: string; };
  }>(),
  
  // BAA and compliance
  baaRequired: boolean("baa_required").default(false),
  baaStatus: varchar("baa_status"), // 'not_required', 'pending', 'active', 'expired'
  hipaaCompliant: boolean("hipaa_compliant").default(false),
  fdaCleared: boolean("fda_cleared").default(false),
  fdaClearanceNumber: varchar("fda_clearance_number"),
  
  // Documentation
  setupInstructions: text("setup_instructions"),
  troubleshootingGuide: text("troubleshooting_guide"),
  docsUrl: varchar("docs_url"),
  supportUrl: varchar("support_url"),
  
  // Status
  isActive: boolean("is_active").default(true),
  isPublicApi: boolean("is_public_api").default(false), // true for Fitbit, Withings, Oura, etc.
  requiresPartnership: boolean("requires_partnership").default(false), // true for Dexcom, Garmin, Whoop, etc.
  
  // Metadata
  imageUrl: varchar("image_url"),
  metadata: jsonb("metadata").$type<{
    releaseDate?: string;
    discontinuedDate?: string;
    batteryType?: string;
    waterResistance?: string;
    connectivity?: string[];
  }>(),
  
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
}, (table) => ({
  vendorIdx: index("device_models_vendor_idx").on(table.vendorId),
  typeIdx: index("device_models_type_idx").on(table.deviceType),
  activeIdx: index("device_models_active_idx").on(table.isActive),
}));

export const insertDeviceModelSchema = createInsertSchema(deviceModels).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export type InsertDeviceModel = z.infer<typeof insertDeviceModelSchema>;
export type DeviceModel = typeof deviceModels.$inferSelect;

// Vendor Accounts - OAuth tokens and credentials for vendor APIs (PHI: encrypted tokens)
export const vendorAccounts = pgTable("vendor_accounts", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  userId: varchar("user_id").notNull().references(() => users.id),
  
  // Vendor identification
  vendorId: varchar("vendor_id").notNull(), // 'fitbit', 'withings', 'garmin', etc.
  vendorName: varchar("vendor_name").notNull(),
  
  // OAuth tokens (PHI: must be encrypted at rest)
  accessToken: text("access_token"), // Encrypted
  refreshToken: text("refresh_token"), // Encrypted
  tokenType: varchar("token_type").default("Bearer"),
  tokenExpiresAt: timestamp("token_expires_at"),
  tokenScope: text("token_scope"), // Granted scopes
  
  // Vendor user ID
  vendorUserId: varchar("vendor_user_id"), // User ID in vendor's system
  vendorUsername: varchar("vendor_username"), // Display name from vendor
  
  // Connection status
  connectionStatus: varchar("connection_status").default("pending"), // 'pending', 'connected', 'disconnected', 'expired', 'revoked', 'error'
  lastAuthAt: timestamp("last_auth_at"),
  lastRefreshAt: timestamp("last_refresh_at"),
  lastRefreshError: text("last_refresh_error"),
  refreshAttempts: integer("refresh_attempts").default(0),
  
  // Sync settings
  autoSync: boolean("auto_sync").default(true),
  syncFrequency: varchar("sync_frequency").default("hourly"), // 'real_time', 'hourly', 'daily', 'manual'
  lastSyncAt: timestamp("last_sync_at"),
  lastSyncStatus: varchar("last_sync_status"), // 'success', 'partial', 'failed'
  lastSyncError: text("last_sync_error"),
  syncedDataTypes: jsonb("synced_data_types").$type<string[]>(), // What data types are being synced
  
  // Webhook configuration
  webhookId: varchar("webhook_id"), // Vendor's webhook subscription ID
  webhookSecret: text("webhook_secret"), // Encrypted
  webhookActive: boolean("webhook_active").default(false),
  
  // Consent tracking
  consentGrantedAt: timestamp("consent_granted_at"),
  consentVersion: varchar("consent_version"),
  consentDataTypes: jsonb("consent_data_types").$type<string[]>(), // What data user consented to share
  
  // Metadata
  metadata: jsonb("metadata").$type<{
    profileData?: Record<string, unknown>;
    subscriptionId?: string;
    apiVersion?: string;
  }>(),
  
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
}, (table) => ({
  userVendorIdx: index("vendor_accounts_user_vendor_idx").on(table.userId, table.vendorId),
  statusIdx: index("vendor_accounts_status_idx").on(table.connectionStatus),
  expiryIdx: index("vendor_accounts_expiry_idx").on(table.tokenExpiresAt),
}));

export const insertVendorAccountSchema = createInsertSchema(vendorAccounts).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export type InsertVendorAccount = z.infer<typeof insertVendorAccountSchema>;
export type VendorAccount = typeof vendorAccounts.$inferSelect;

// Device Health - Real-time device status and health metrics
export const deviceHealth = pgTable("device_health", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  deviceConnectionId: varchar("device_connection_id").notNull().references(() => wearableIntegrations.id),
  
  // Device status
  status: varchar("status").notNull().default("unknown"), // 'online', 'offline', 'pairing', 'syncing', 'error', 'low_battery', 'unknown'
  statusMessage: text("status_message"),
  lastSeenAt: timestamp("last_seen_at"),
  
  // Battery
  batteryLevel: integer("battery_level"), // 0-100%
  batteryStatus: varchar("battery_status"), // 'charging', 'discharging', 'full', 'low', 'critical'
  estimatedBatteryLife: varchar("estimated_battery_life"), // e.g., "2 days"
  
  // Connectivity
  signalStrength: integer("signal_strength"), // RSSI or signal quality 0-100
  connectionType: varchar("connection_type"), // 'ble', 'wifi', 'cellular', 'usb'
  lastConnectionAt: timestamp("last_connection_at"),
  connectionErrors: integer("connection_errors").default(0),
  
  // Firmware
  firmwareVersion: varchar("firmware_version"),
  firmwareUpdateAvailable: boolean("firmware_update_available").default(false),
  latestFirmwareVersion: varchar("latest_firmware_version"),
  
  // Sync health
  syncSuccessRate: decimal("sync_success_rate", { precision: 5, scale: 2 }), // Last 30 days
  avgSyncLatency: integer("avg_sync_latency"), // Milliseconds
  lastSuccessfulSync: timestamp("last_successful_sync"),
  consecutiveFailures: integer("consecutive_failures").default(0),
  
  // Data quality
  dataQualityScore: integer("data_quality_score"), // 0-100
  missingDataTypes: jsonb("missing_data_types").$type<string[]>(),
  dataGaps: jsonb("data_gaps").$type<{ start: string; end: string; reason?: string; }[]>(),
  
  // Alerts
  healthAlerts: jsonb("health_alerts").$type<{
    type: string; // 'low_battery', 'sync_failed', 'firmware_update', 'connection_lost', 'data_quality'
    severity: string; // 'info', 'warning', 'critical'
    message: string;
    timestamp: string;
    acknowledged: boolean;
  }[]>(),
  
  // Metadata
  metadata: jsonb("metadata").$type<{
    wearPosition?: string;
    calibrationDate?: string;
    sensorHealth?: Record<string, unknown>;
  }>(),
  
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
}, (table) => ({
  deviceIdx: index("device_health_device_idx").on(table.deviceConnectionId),
  statusIdx: index("device_health_status_idx").on(table.status),
  lastSeenIdx: index("device_health_last_seen_idx").on(table.lastSeenAt),
}));

export const insertDeviceHealthSchema = createInsertSchema(deviceHealth).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export type InsertDeviceHealth = z.infer<typeof insertDeviceHealthSchema>;
export type DeviceHealth = typeof deviceHealth.$inferSelect;

// Device Pairing Sessions - Track pairing flows with audit trail
export const devicePairingSessions = pgTable("device_pairing_sessions", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  userId: varchar("user_id").notNull().references(() => users.id),
  
  // Device being paired
  deviceModelId: varchar("device_model_id").references(() => deviceModels.id),
  deviceType: varchar("device_type").notNull(),
  vendorId: varchar("vendor_id").notNull(),
  
  // Pairing method
  pairingMethod: varchar("pairing_method").notNull(), // 'oauth', 'ble', 'healthkit', 'google_fit', 'manual', 'qr_code'
  
  // Session state
  sessionStatus: varchar("session_status").notNull().default("initiated"), // 'initiated', 'pending_auth', 'pending_ble', 'pending_consent', 'completed', 'failed', 'expired', 'cancelled'
  
  // OAuth flow data
  oauthState: varchar("oauth_state"), // CSRF protection
  oauthCodeVerifier: text("oauth_code_verifier"), // PKCE
  oauthRedirectUri: varchar("oauth_redirect_uri"),
  
  // BLE pairing data
  bleDeviceId: varchar("ble_device_id"),
  bleServiceUuid: varchar("ble_service_uuid"),
  blePairingCode: varchar("ble_pairing_code"),
  
  // QR code pairing
  qrCodeToken: varchar("qr_code_token"),
  qrCodeExpiresAt: timestamp("qr_code_expires_at"),
  
  // Result
  resultDeviceConnectionId: varchar("result_device_connection_id").references(() => wearableIntegrations.id),
  resultVendorAccountId: varchar("result_vendor_account_id").references(() => vendorAccounts.id),
  
  // Error tracking
  errorCode: varchar("error_code"),
  errorMessage: text("error_message"),
  errorDetails: jsonb("error_details"),
  
  // Consent
  consentCaptured: boolean("consent_captured").default(false),
  consentTimestamp: timestamp("consent_timestamp"),
  consentVersion: varchar("consent_version"),
  consentedDataTypes: jsonb("consented_data_types").$type<string[]>(),
  
  // Session timing
  startedAt: timestamp("started_at").defaultNow(),
  expiresAt: timestamp("expires_at"),
  completedAt: timestamp("completed_at"),
  
  // Audit trail
  ipAddress: varchar("ip_address"),
  userAgent: text("user_agent"),
  
  // Metadata
  metadata: jsonb("metadata").$type<{
    stepHistory?: { step: string; timestamp: string; success: boolean; }[];
    deviceInfo?: Record<string, unknown>;
  }>(),
  
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
}, (table) => ({
  userIdx: index("device_pairing_user_idx").on(table.userId),
  statusIdx: index("device_pairing_status_idx").on(table.sessionStatus),
  oauthStateIdx: index("device_pairing_oauth_state_idx").on(table.oauthState),
  expiryIdx: index("device_pairing_expiry_idx").on(table.expiresAt),
}));

export const insertDevicePairingSessionSchema = createInsertSchema(devicePairingSessions).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export type InsertDevicePairingSession = z.infer<typeof insertDevicePairingSessionSchema>;
export type DevicePairingSession = typeof devicePairingSessions.$inferSelect;

// Device Sync Jobs - Track background sync operations
export const deviceSyncJobs = pgTable("device_sync_jobs", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  
  // Job identification
  jobType: varchar("job_type").notNull(), // 'scheduled_sync', 'on_demand_sync', 'webhook_process', 'token_refresh', 'health_check'
  
  // Target
  vendorAccountId: varchar("vendor_account_id").references(() => vendorAccounts.id),
  deviceConnectionId: varchar("device_connection_id").references(() => wearableIntegrations.id),
  userId: varchar("user_id").references(() => users.id),
  
  // Job state
  status: varchar("status").notNull().default("pending"), // 'pending', 'running', 'completed', 'failed', 'retrying', 'cancelled'
  priority: integer("priority").default(5), // 1-10, higher = more urgent
  
  // Execution
  scheduledFor: timestamp("scheduled_for"),
  startedAt: timestamp("started_at"),
  completedAt: timestamp("completed_at"),
  
  // Retry logic
  attempts: integer("attempts").default(0),
  maxAttempts: integer("max_attempts").default(3),
  nextRetryAt: timestamp("next_retry_at"),
  
  // Results
  recordsProcessed: integer("records_processed").default(0),
  recordsFailed: integer("records_failed").default(0),
  dataTypes: jsonb("data_types").$type<string[]>(), // What data types were synced
  dateRange: jsonb("date_range").$type<{ start: string; end: string; }>(),
  
  // Error tracking
  errorCode: varchar("error_code"),
  errorMessage: text("error_message"),
  errorStack: text("error_stack"),
  
  // Worker info
  workerId: varchar("worker_id"),
  
  // Metadata
  metadata: jsonb("metadata").$type<{
    triggerSource?: string; // 'schedule', 'user', 'webhook', 'system'
    webhookPayload?: Record<string, unknown>;
  }>(),
  
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
}, (table) => ({
  statusIdx: index("device_sync_jobs_status_idx").on(table.status),
  scheduledIdx: index("device_sync_jobs_scheduled_idx").on(table.scheduledFor),
  vendorIdx: index("device_sync_jobs_vendor_idx").on(table.vendorAccountId),
}));

export const insertDeviceSyncJobSchema = createInsertSchema(deviceSyncJobs).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export type InsertDeviceSyncJob = z.infer<typeof insertDeviceSyncJobSchema>;
export type DeviceSyncJob = typeof deviceSyncJobs.$inferSelect;

// Device Data Audit Log - HIPAA-compliant audit trail for all device data access
export const deviceDataAuditLog = pgTable("device_data_audit_log", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  
  // Actor
  actorId: varchar("actor_id").notNull(), // User or system ID
  actorType: varchar("actor_type").notNull(), // 'patient', 'clinician', 'system', 'vendor_webhook'
  actorRole: varchar("actor_role"), // 'patient', 'doctor', 'nurse', 'admin'
  
  // Action
  action: varchar("action").notNull(), // 'device_paired', 'device_unpaired', 'data_synced', 'data_viewed', 'data_exported', 'consent_granted', 'consent_revoked', 'token_refreshed'
  actionCategory: varchar("action_category").notNull(), // 'device_management', 'data_access', 'consent', 'sync'
  
  // Resource
  resourceType: varchar("resource_type").notNull(), // 'device_connection', 'vendor_account', 'device_reading', 'device_health'
  resourceId: varchar("resource_id"),
  patientId: varchar("patient_id"), // PHI: patient whose data was accessed
  
  // Details
  eventDetails: jsonb("event_details").$type<{
    deviceType?: string;
    vendorId?: string;
    dataTypes?: string[];
    recordCount?: number;
    dateRange?: { start: string; end: string; };
    previousValue?: Record<string, unknown>;
    newValue?: Record<string, unknown>;
  }>(),
  
  // Request context
  ipAddress: varchar("ip_address"),
  userAgent: text("user_agent"),
  sessionId: varchar("session_id"),
  requestId: varchar("request_id"),
  
  // Result
  success: boolean("success").default(true),
  errorMessage: text("error_message"),
  
  // PHI flag
  phiAccessed: boolean("phi_accessed").default(false),
  
  createdAt: timestamp("created_at").defaultNow(),
}, (table) => ({
  actorIdx: index("device_audit_actor_idx").on(table.actorId),
  actionIdx: index("device_audit_action_idx").on(table.action),
  resourceIdx: index("device_audit_resource_idx").on(table.resourceType, table.resourceId),
  patientIdx: index("device_audit_patient_idx").on(table.patientId),
  createdIdx: index("device_audit_created_idx").on(table.createdAt),
}));

export const insertDeviceDataAuditLogSchema = createInsertSchema(deviceDataAuditLog).omit({
  id: true,
  createdAt: true,
});

export type InsertDeviceDataAuditLog = z.infer<typeof insertDeviceDataAuditLogSchema>;
export type DeviceDataAuditLog = typeof deviceDataAuditLog.$inferSelect;

// Health Section Analytics - Per-section health metrics derived from device data
export const healthSectionAnalytics = pgTable("health_section_analytics", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  patientId: varchar("patient_id").notNull().references(() => users.id),
  
  // Section identification
  healthSection: varchar("health_section").notNull(), // 'hypertension', 'diabetes', 'cardiovascular', 'respiratory', 'sleep', 'mental_health', 'fitness'
  
  // Time window
  analysisDate: timestamp("analysis_date").notNull(),
  windowStart: timestamp("window_start").notNull(),
  windowEnd: timestamp("window_end").notNull(),
  windowDays: integer("window_days").default(7),
  
  // Data sources used
  deviceTypes: jsonb("device_types").$type<string[]>(), // Which device types contributed
  readingCount: integer("reading_count").default(0),
  
  // ML Predictions
  mlPrediction: jsonb("ml_prediction").$type<{
    predictedOutcome: string; // 'stable', 'improving', 'declining', 'at_risk', 'critical'
    confidence: number; // 0-1
    modelVersion: string;
    features: Record<string, number>;
  }>(),
  
  // Deterioration Index (0-100, higher = worse)
  deteriorationIndex: decimal("deterioration_index", { precision: 5, scale: 2 }),
  deteriorationTrend: varchar("deterioration_trend"), // 'improving', 'stable', 'worsening'
  deteriorationChangeRate: decimal("deterioration_change_rate", { precision: 5, scale: 2 }), // % change from previous
  
  // Risk Score (0-100)
  riskScore: decimal("risk_score", { precision: 5, scale: 2 }),
  riskLevel: varchar("risk_level"), // 'low', 'moderate', 'high', 'critical'
  riskFactors: jsonb("risk_factors").$type<{ factor: string; contribution: number; value: number; }[]>(),
  
  // Trend Analysis
  trendDirection: varchar("trend_direction"), // 'improving', 'stable', 'declining'
  trendStrength: decimal("trend_strength", { precision: 5, scale: 2 }), // 0-1
  trendMetrics: jsonb("trend_metrics").$type<{
    metric: string;
    values: number[];
    slope: number;
    r_squared: number;
    forecast: number[];
  }[]>(),
  
  // Stability Score (0-100, higher = more stable)
  stabilityScore: decimal("stability_score", { precision: 5, scale: 2 }),
  variabilityIndex: decimal("variability_index", { precision: 5, scale: 2 }),
  stabilityMetrics: jsonb("stability_metrics").$type<{
    metric: string;
    mean: number;
    std: number;
    cv: number; // Coefficient of variation
    range: { min: number; max: number; };
  }[]>(),
  
  // Alerts generated
  alertsGenerated: jsonb("alerts_generated").$type<{
    alertId: string;
    alertType: string;
    severity: string;
    message: string;
    triggeredAt: string;
    acknowledged: boolean;
  }[]>(),
  activeAlertCount: integer("active_alert_count").default(0),
  
  // Section-specific metrics (JSON for flexibility)
  sectionMetrics: jsonb("section_metrics").$type<Record<string, unknown>>(),
  
  // Baseline comparison
  baselineDeviation: decimal("baseline_deviation", { precision: 5, scale: 2 }), // % deviation from personal baseline
  baselineStatus: varchar("baseline_status"), // 'within_normal', 'elevated', 'concerning', 'critical'
  
  // Metadata
  metadata: jsonb("metadata").$type<{
    processingTime?: number;
    dataQuality?: number;
    missingDataTypes?: string[];
  }>(),
  
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
}, (table) => ({
  patientSectionIdx: index("health_analytics_patient_section_idx").on(table.patientId, table.healthSection),
  dateIdx: index("health_analytics_date_idx").on(table.analysisDate),
  riskIdx: index("health_analytics_risk_idx").on(table.riskLevel),
}));

export const insertHealthSectionAnalyticsSchema = createInsertSchema(healthSectionAnalytics).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export type InsertHealthSectionAnalytics = z.infer<typeof insertHealthSectionAnalyticsSchema>;
export type HealthSectionAnalytics = typeof healthSectionAnalytics.$inferSelect;

// ============================================================================
// ENVIRONMENTAL RISK MAP SYSTEM - Advanced Environmental Health Intelligence
// ============================================================================

// Patient Environment Profile - Links patients to their location and condition triggers
export const patientEnvironmentProfiles = pgTable("patient_environment_profiles", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  patientId: varchar("patient_id").notNull().references(() => users.id),
  
  // Location (HIPAA compliant - ZIP only, no GPS)
  zipCode: varchar("zip_code", { length: 10 }).notNull(),
  city: varchar("city"),
  state: varchar("state", { length: 2 }),
  timezone: varchar("timezone").default("America/New_York"),
  
  // Patient conditions for personalized triggers
  chronicConditions: jsonb("chronic_conditions").$type<string[]>(), // ['asthma', 'copd', 'heart_failure', 'arthritis', 'migraines', 'eczema']
  allergies: jsonb("allergies").$type<string[]>(), // ['pollen', 'mold', 'dust', 'pet_dander']
  
  // Alert preferences
  alertsEnabled: boolean("alerts_enabled").default(true),
  alertThresholds: jsonb("alert_thresholds").$type<{
    riskScore: number; // Notify when composite risk exceeds this (0-100)
    aqiThreshold: number; // Notify when AQI exceeds this
    pollenThreshold: number; // Notify when pollen exceeds this (0-12)
    temperatureMin: number; // Notify when temp drops below (Celsius)
    temperatureMax: number; // Notify when temp exceeds (Celsius)
    humidityMin: number; // Notify when humidity drops below (%)
    humidityMax: number; // Notify when humidity exceeds (%)
  }>(),
  
  // Notification channels
  pushNotifications: boolean("push_notifications").default(true),
  smsNotifications: boolean("sms_notifications").default(false),
  emailDigest: boolean("email_digest").default(true),
  digestFrequency: varchar("digest_frequency").default("daily"), // 'daily', 'weekly', 'never'
  
  // Consent for ML correlation
  correlationConsentGiven: boolean("correlation_consent_given").default(false),
  correlationConsentAt: timestamp("correlation_consent_at"),
  
  isActive: boolean("is_active").default(true),
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
}, (table) => ({
  patientIdx: index("env_profile_patient_idx").on(table.patientId),
  zipIdx: index("env_profile_zip_idx").on(table.zipCode),
}));

export const insertPatientEnvironmentProfileSchema = createInsertSchema(patientEnvironmentProfiles).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export type InsertPatientEnvironmentProfile = z.infer<typeof insertPatientEnvironmentProfileSchema>;
export type PatientEnvironmentProfile = typeof patientEnvironmentProfiles.$inferSelect;

// Condition Trigger Mapping - Maps conditions to environmental factors with weights
export const conditionTriggerMappings = pgTable("condition_trigger_mappings", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  
  // Condition this mapping applies to
  conditionCode: varchar("condition_code").notNull(), // 'asthma', 'copd', 'heart_failure', 'arthritis', 'migraines', 'eczema'
  conditionName: varchar("condition_name").notNull(),
  
  // Environmental factor this condition is sensitive to
  factorType: varchar("factor_type").notNull(), // 'pm25', 'pm10', 'ozone', 'humidity', 'temperature', 'pressure', 'pollen', 'mold', 'uv'
  factorName: varchar("factor_name").notNull(),
  
  // Trigger thresholds
  triggerThreshold: decimal("trigger_threshold", { precision: 10, scale: 4 }), // Value at which factor becomes concerning
  criticalThreshold: decimal("critical_threshold", { precision: 10, scale: 4 }), // Value at which factor becomes critical
  
  // Weight in risk calculation (0.0 to 1.0)
  baseWeight: decimal("base_weight", { precision: 5, scale: 4 }).default("0.5"),
  
  // Direction of impact
  impactDirection: varchar("impact_direction").notNull(), // 'higher_is_worse', 'lower_is_worse', 'both_extremes'
  
  // Evidence and recommendations
  clinicalEvidence: text("clinical_evidence"),
  recommendations: jsonb("recommendations").$type<string[]>(),
  
  isActive: boolean("is_active").default(true),
  createdAt: timestamp("created_at").defaultNow(),
}, (table) => ({
  conditionIdx: index("trigger_condition_idx").on(table.conditionCode),
  factorIdx: index("trigger_factor_idx").on(table.factorType),
}));

export const insertConditionTriggerMappingSchema = createInsertSchema(conditionTriggerMappings).omit({
  id: true,
  createdAt: true,
});

export type InsertConditionTriggerMapping = z.infer<typeof insertConditionTriggerMappingSchema>;
export type ConditionTriggerMapping = typeof conditionTriggerMappings.$inferSelect;

// Patient Trigger Weights - Personalized weights learned from patient-specific correlations
export const patientTriggerWeights = pgTable("patient_trigger_weights", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  patientId: varchar("patient_id").notNull().references(() => users.id),
  
  // Environmental factor
  factorType: varchar("factor_type").notNull(),
  
  // Personalized weight (learned from correlation analysis)
  personalizedWeight: decimal("personalized_weight", { precision: 5, scale: 4 }).notNull(),
  
  // Confidence in this weight (based on data volume)
  confidenceScore: decimal("confidence_score", { precision: 5, scale: 4 }),
  
  // How the weight was determined
  source: varchar("source").notNull(), // 'default', 'correlation', 'manual', 'ml_predicted'
  
  // Correlation statistics
  correlationCoefficient: decimal("correlation_coefficient", { precision: 5, scale: 4 }),
  pValue: decimal("p_value", { precision: 8, scale: 6 }),
  sampleSize: integer("sample_size"),
  
  lastUpdatedAt: timestamp("last_updated_at").defaultNow(),
  createdAt: timestamp("created_at").defaultNow(),
}, (table) => ({
  patientFactorIdx: index("patient_trigger_idx").on(table.patientId, table.factorType),
}));

export const insertPatientTriggerWeightSchema = createInsertSchema(patientTriggerWeights).omit({
  id: true,
  createdAt: true,
});

export type InsertPatientTriggerWeight = z.infer<typeof insertPatientTriggerWeightSchema>;
export type PatientTriggerWeight = typeof patientTriggerWeights.$inferSelect;

// Environmental Data Snapshots - Time-series environmental readings by ZIP code
export const environmentalDataSnapshots = pgTable("environmental_data_snapshots", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  zipCode: varchar("zip_code", { length: 10 }).notNull(),
  
  // Timestamp of measurement
  measuredAt: timestamp("measured_at").notNull(),
  
  // Weather data
  temperature: decimal("temperature", { precision: 6, scale: 2 }), // Celsius
  feelsLike: decimal("feels_like", { precision: 6, scale: 2 }),
  humidity: decimal("humidity", { precision: 5, scale: 2 }), // Percentage
  pressure: decimal("pressure", { precision: 7, scale: 2 }), // hPa
  windSpeed: decimal("wind_speed", { precision: 6, scale: 2 }), // m/s
  windDirection: integer("wind_direction"), // Degrees
  precipitation: decimal("precipitation", { precision: 6, scale: 2 }), // mm
  uvIndex: decimal("uv_index", { precision: 4, scale: 2 }),
  cloudCover: integer("cloud_cover"), // Percentage
  visibility: decimal("visibility", { precision: 8, scale: 2 }), // meters
  
  // Air Quality
  aqi: integer("aqi"), // 0-500 EPA scale
  aqiCategory: varchar("aqi_category"), // 'good', 'moderate', 'unhealthy_sensitive', 'unhealthy', 'very_unhealthy', 'hazardous'
  pm25: decimal("pm25", { precision: 7, scale: 3 }), // µg/m³
  pm10: decimal("pm10", { precision: 7, scale: 3 }),
  ozone: decimal("ozone", { precision: 7, scale: 3 }), // ppb
  no2: decimal("no2", { precision: 7, scale: 3 }), // ppb
  so2: decimal("so2", { precision: 7, scale: 3 }), // ppb
  co: decimal("co", { precision: 8, scale: 3 }), // ppm
  
  // Allergens
  pollenTreeCount: integer("pollen_tree_count"), // grains/m³
  pollenGrassCount: integer("pollen_grass_count"),
  pollenWeedCount: integer("pollen_weed_count"),
  pollenOverall: integer("pollen_overall"), // 0-12 scale
  pollenCategory: varchar("pollen_category"), // 'low', 'moderate', 'high', 'very_high'
  moldSporeCount: integer("mold_spore_count"),
  moldCategory: varchar("mold_category"),
  
  // Environmental Hazards
  activeHazards: jsonb("active_hazards").$type<{
    type: string; // 'fire', 'smoke', 'flood', 'heat_wave', 'air_quality_alert', 'disease_outbreak'
    severity: 'low' | 'moderate' | 'high' | 'extreme';
    title: string;
    description: string;
    source: string;
    startTime?: string;
    endTime?: string;
  }[]>(),
  
  // Data sources
  weatherSource: varchar("weather_source"), // 'openweathermap', 'weatherapi', 'noaa'
  aqiSource: varchar("aqi_source"), // 'airnow', 'iqair', 'breezometer'
  pollenSource: varchar("pollen_source"), // 'pollen.com', 'breezometer', 'ambee'
  hazardSource: varchar("hazard_source"), // 'epa', 'nasa_firms', 'nws'
  
  // Data quality
  dataQualityScore: decimal("data_quality_score", { precision: 4, scale: 2 }), // 0-100
  missingFields: jsonb("missing_fields").$type<string[]>(),
  
  createdAt: timestamp("created_at").defaultNow(),
}, (table) => ({
  zipTimeIdx: index("env_snapshot_zip_time_idx").on(table.zipCode, table.measuredAt),
  measuredAtIdx: index("env_snapshot_measured_idx").on(table.measuredAt),
}));

export const insertEnvironmentalDataSnapshotSchema = createInsertSchema(environmentalDataSnapshots).omit({
  id: true,
  createdAt: true,
});

export type InsertEnvironmentalDataSnapshot = z.infer<typeof insertEnvironmentalDataSnapshotSchema>;
export type EnvironmentalDataSnapshot = typeof environmentalDataSnapshots.$inferSelect;

// Patient Risk Scores - Computed personalized risk scores
export const patientEnvironmentRiskScores = pgTable("patient_environment_risk_scores", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  patientId: varchar("patient_id").notNull().references(() => users.id),
  snapshotId: varchar("snapshot_id").references(() => environmentalDataSnapshots.id),
  
  // Timestamp
  computedAt: timestamp("computed_at").notNull(),
  
  // Composite Risk Score (0-100)
  compositeRiskScore: decimal("composite_risk_score", { precision: 5, scale: 2 }).notNull(),
  riskLevel: varchar("risk_level").notNull(), // 'low', 'moderate', 'high', 'critical'
  
  // Component scores (each 0-100)
  weatherRiskScore: decimal("weather_risk_score", { precision: 5, scale: 2 }),
  airQualityRiskScore: decimal("air_quality_risk_score", { precision: 5, scale: 2 }),
  allergenRiskScore: decimal("allergen_risk_score", { precision: 5, scale: 2 }),
  hazardRiskScore: decimal("hazard_risk_score", { precision: 5, scale: 2 }),
  
  // Trend scores (normalized slopes)
  trend24hr: decimal("trend_24hr", { precision: 6, scale: 3 }), // -1 to +1 (negative = improving)
  trend48hr: decimal("trend_48hr", { precision: 6, scale: 3 }),
  trend72hr: decimal("trend_72hr", { precision: 6, scale: 3 }),
  
  // Volatility score (7-day standard deviation normalized)
  volatilityScore: decimal("volatility_score", { precision: 5, scale: 2 }),
  
  // Individual factor contributions
  factorContributions: jsonb("factor_contributions").$type<{
    factor: string;
    rawValue: number;
    normalizedValue: number;
    weight: number;
    contribution: number; // weight * normalizedValue
  }[]>(),
  
  // Top risk factors
  topRiskFactors: jsonb("top_risk_factors").$type<{
    factor: string;
    severity: 'low' | 'moderate' | 'high' | 'critical';
    recommendation: string;
  }[]>(),
  
  // Scoring metadata
  scoringVersion: varchar("scoring_version").default("1.0"),
  
  createdAt: timestamp("created_at").defaultNow(),
}, (table) => ({
  patientTimeIdx: index("risk_score_patient_time_idx").on(table.patientId, table.computedAt),
  riskLevelIdx: index("risk_score_level_idx").on(table.riskLevel),
}));

export const insertPatientEnvironmentRiskScoreSchema = createInsertSchema(patientEnvironmentRiskScores).omit({
  id: true,
  createdAt: true,
});

export type InsertPatientEnvironmentRiskScore = z.infer<typeof insertPatientEnvironmentRiskScoreSchema>;
export type PatientEnvironmentRiskScore = typeof patientEnvironmentRiskScores.$inferSelect;

// Environmental Forecasts - ML-predicted future risk
export const environmentalForecasts = pgTable("environmental_forecasts", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  patientId: varchar("patient_id").notNull().references(() => users.id),
  
  // When forecast was generated
  generatedAt: timestamp("generated_at").notNull(),
  
  // Forecast horizon
  forecastHorizon: varchar("forecast_horizon").notNull(), // '12hr', '24hr', '48hr', '72hr'
  forecastTargetTime: timestamp("forecast_target_time").notNull(),
  
  // Predicted risk scores
  predictedRiskScore: decimal("predicted_risk_score", { precision: 5, scale: 2 }).notNull(),
  predictedRiskLevel: varchar("predicted_risk_level").notNull(),
  confidenceInterval: jsonb("confidence_interval").$type<{
    lower: number;
    upper: number;
    confidence: number; // 0.95 for 95% CI
  }>(),
  
  // Component predictions
  predictedWeatherRisk: decimal("predicted_weather_risk", { precision: 5, scale: 2 }),
  predictedAirQualityRisk: decimal("predicted_air_quality_risk", { precision: 5, scale: 2 }),
  predictedAllergenRisk: decimal("predicted_allergen_risk", { precision: 5, scale: 2 }),
  
  // Key predicted values
  predictedValues: jsonb("predicted_values").$type<{
    factor: string;
    currentValue: number;
    predictedValue: number;
    changeDirection: 'increase' | 'decrease' | 'stable';
    changePercent: number;
  }[]>(),
  
  // Model metadata
  modelName: varchar("model_name").notNull(), // 'lightgbm_risk_v1', 'lstm_risk_v2'
  modelVersion: varchar("model_version").notNull(),
  featureImportance: jsonb("feature_importance").$type<Record<string, number>>(),
  
  // Actual values (filled in after forecast period passes)
  actualRiskScore: decimal("actual_risk_score", { precision: 5, scale: 2 }),
  forecastError: decimal("forecast_error", { precision: 5, scale: 2 }), // Absolute error
  
  createdAt: timestamp("created_at").defaultNow(),
}, (table) => ({
  patientTimeIdx: index("forecast_patient_time_idx").on(table.patientId, table.generatedAt),
  horizonIdx: index("forecast_horizon_idx").on(table.forecastHorizon),
}));

export const insertEnvironmentalForecastSchema = createInsertSchema(environmentalForecasts).omit({
  id: true,
  createdAt: true,
});

export type InsertEnvironmentalForecast = z.infer<typeof insertEnvironmentalForecastSchema>;
export type EnvironmentalForecast = typeof environmentalForecasts.$inferSelect;

// Symptom-Environment Correlations - Learned correlations between symptoms and environment
export const symptomEnvironmentCorrelations = pgTable("symptom_environment_correlations", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  patientId: varchar("patient_id").notNull().references(() => users.id),
  
  // Symptom being correlated
  symptomType: varchar("symptom_type").notNull(), // 'pain', 'fatigue', 'breathing_difficulty', 'headache', 'joint_stiffness'
  symptomSeverityMetric: varchar("symptom_severity_metric").notNull(), // 'vas_score', 'frequency', 'duration'
  
  // Environmental factor
  environmentalFactor: varchar("environmental_factor").notNull(), // 'pm25', 'humidity', 'pressure', 'pollen', etc.
  
  // Correlation statistics
  correlationType: varchar("correlation_type").notNull(), // 'spearman', 'pearson', 'cross_correlation'
  correlationCoefficient: decimal("correlation_coefficient", { precision: 6, scale: 4 }).notNull(),
  pValue: decimal("p_value", { precision: 10, scale: 8 }),
  isStatisticallySignificant: boolean("is_statistically_significant").default(false),
  
  // Cross-correlation lag analysis
  optimalLag: integer("optimal_lag"), // Hours of delay between env change and symptom change
  lagCorrelation: decimal("lag_correlation", { precision: 6, scale: 4 }),
  
  // Sample data
  sampleSize: integer("sample_size").notNull(),
  dataWindowDays: integer("data_window_days"), // How many days of data used
  
  // Interpretation
  relationshipStrength: varchar("relationship_strength"), // 'weak', 'moderate', 'strong', 'very_strong'
  relationshipDirection: varchar("relationship_direction"), // 'positive', 'negative', 'non_linear'
  interpretation: text("interpretation"), // AI-generated explanation
  
  // Confidence and quality
  confidenceScore: decimal("confidence_score", { precision: 5, scale: 4 }),
  dataQualityScore: decimal("data_quality_score", { precision: 5, scale: 4 }),
  
  lastAnalyzedAt: timestamp("last_analyzed_at").notNull(),
  createdAt: timestamp("created_at").defaultNow(),
}, (table) => ({
  patientSymptomIdx: index("correlation_patient_symptom_idx").on(table.patientId, table.symptomType),
  significantIdx: index("correlation_significant_idx").on(table.isStatisticallySignificant),
}));

export const insertSymptomEnvironmentCorrelationSchema = createInsertSchema(symptomEnvironmentCorrelations).omit({
  id: true,
  createdAt: true,
});

export type InsertSymptomEnvironmentCorrelation = z.infer<typeof insertSymptomEnvironmentCorrelationSchema>;
export type SymptomEnvironmentCorrelation = typeof symptomEnvironmentCorrelations.$inferSelect;

// Environmental Alerts - Alert history and active alerts
export const environmentalAlerts = pgTable("environmental_alerts", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  patientId: varchar("patient_id").notNull().references(() => users.id),
  
  // Alert type and trigger
  alertType: varchar("alert_type").notNull(), // 'acute_spike', 'forecast_deterioration', 'threshold_exceeded', 'correlation_trigger', 'hazard_warning'
  triggeredBy: varchar("triggered_by").notNull(), // The factor that triggered the alert (e.g., 'pm25', 'composite_risk')
  
  // Severity
  severity: varchar("severity").notNull(), // 'info', 'warning', 'critical', 'emergency'
  priority: integer("priority").notNull(), // 1-10 (10 = most urgent)
  
  // Alert content
  title: varchar("title").notNull(),
  message: text("message").notNull(),
  recommendations: jsonb("recommendations").$type<{
    action: string;
    urgency: 'immediate' | 'today' | 'this_week';
    category: 'medical' | 'lifestyle' | 'environmental' | 'medication';
  }[]>(),
  
  // Trigger values
  triggerValue: decimal("trigger_value", { precision: 10, scale: 4 }),
  thresholdValue: decimal("threshold_value", { precision: 10, scale: 4 }),
  percentOverThreshold: decimal("percent_over_threshold", { precision: 6, scale: 2 }),
  
  // Related data
  riskScoreId: varchar("risk_score_id").references(() => patientEnvironmentRiskScores.id),
  forecastId: varchar("forecast_id").references(() => environmentalForecasts.id),
  snapshotId: varchar("snapshot_id").references(() => environmentalDataSnapshots.id),
  
  // Status
  status: varchar("status").default("active"), // 'active', 'acknowledged', 'resolved', 'dismissed', 'expired'
  acknowledgedAt: timestamp("acknowledged_at"),
  resolvedAt: timestamp("resolved_at"),
  expiresAt: timestamp("expires_at"),
  
  // Notification tracking
  pushNotificationSent: boolean("push_notification_sent").default(false),
  smsNotificationSent: boolean("sms_notification_sent").default(false),
  emailNotificationSent: boolean("email_notification_sent").default(false),
  notificationSentAt: timestamp("notification_sent_at"),
  
  // User feedback
  wasHelpful: boolean("was_helpful"),
  userFeedback: text("user_feedback"),
  
  createdAt: timestamp("created_at").defaultNow(),
}, (table) => ({
  patientStatusIdx: index("env_alert_patient_status_idx").on(table.patientId, table.status),
  severityIdx: index("env_alert_severity_idx").on(table.severity),
  createdAtIdx: index("env_alert_created_idx").on(table.createdAt),
}));

export const insertEnvironmentalAlertSchema = createInsertSchema(environmentalAlerts).omit({
  id: true,
  createdAt: true,
});

export type InsertEnvironmentalAlert = z.infer<typeof insertEnvironmentalAlertSchema>;
export type EnvironmentalAlert = typeof environmentalAlerts.$inferSelect;

// Environmental Pipeline Jobs - Track background job execution
export const environmentalPipelineJobs = pgTable("environmental_pipeline_jobs", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  
  // Job type
  jobType: varchar("job_type").notNull(), // 'data_ingestion', 'risk_scoring', 'correlation_analysis', 'forecasting', 'alert_check', 'daily_summary'
  
  // Scope
  targetZipCodes: jsonb("target_zip_codes").$type<string[]>(), // Which ZIP codes were processed
  targetPatientIds: jsonb("target_patient_ids").$type<string[]>(), // Which patients were processed
  
  // Status
  status: varchar("status").notNull(), // 'pending', 'running', 'completed', 'failed', 'cancelled'
  startedAt: timestamp("started_at"),
  completedAt: timestamp("completed_at"),
  
  // Results
  recordsProcessed: integer("records_processed").default(0),
  recordsCreated: integer("records_created").default(0),
  recordsUpdated: integer("records_updated").default(0),
  alertsGenerated: integer("alerts_generated").default(0),
  
  // Errors
  errorMessage: text("error_message"),
  errorStack: text("error_stack"),
  retryCount: integer("retry_count").default(0),
  maxRetries: integer("max_retries").default(3),
  
  // Performance
  executionTimeMs: integer("execution_time_ms"),
  
  // Metadata
  triggerSource: varchar("trigger_source"), // 'scheduled', 'manual', 'webhook', 'on_demand'
  metadata: jsonb("metadata").$type<Record<string, unknown>>(),
  
  createdAt: timestamp("created_at").defaultNow(),
}, (table) => ({
  jobTypeStatusIdx: index("pipeline_job_type_status_idx").on(table.jobType, table.status),
  createdAtIdx: index("pipeline_job_created_idx").on(table.createdAt),
}));

export const insertEnvironmentalPipelineJobSchema = createInsertSchema(environmentalPipelineJobs).omit({
  id: true,
  createdAt: true,
});

export type InsertEnvironmentalPipelineJob = z.infer<typeof insertEnvironmentalPipelineJobSchema>;
export type EnvironmentalPipelineJob = typeof environmentalPipelineJobs.$inferSelect;

// =========================================================================
// ENHANCED RESEARCH CENTER SCHEMA
// =========================================================================

// Extended Research Consent - Granular data type permissions for epidemiological research
export const researchDataConsent = pgTable("research_data_consent", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  patientId: varchar("patient_id").notNull().references(() => users.id),
  
  // Overall consent status
  consentEnabled: boolean("consent_enabled").notNull().default(false),
  
  // Granular data type permissions
  dataTypePermissions: jsonb("data_type_permissions").$type<{
    dailyFollowups: boolean;
    healthAlerts: boolean;
    deteriorationIndex: boolean;
    mlPredictions: boolean;
    environmentalRisk: boolean;
    medications: boolean;
    vitals: boolean;
    immuneMarkers: boolean;
    behavioralData: boolean;
    mentalHealth: boolean;
    wearableData: boolean;
    labResults: boolean;
    conditions: boolean;
    demographics: boolean;
    painTracking: boolean;
    symptomJournal: boolean;
  }>().default({
    dailyFollowups: false,
    healthAlerts: false,
    deteriorationIndex: false,
    mlPredictions: false,
    environmentalRisk: false,
    medications: false,
    vitals: false,
    immuneMarkers: false,
    behavioralData: false,
    mentalHealth: false,
    wearableData: false,
    labResults: false,
    conditions: false,
    demographics: false,
    painTracking: false,
    symptomJournal: false,
  }),
  
  // Anonymization preferences
  anonymizationLevel: varchar("anonymization_level").default("full"), // 'full', 'partial', 'minimal'
  allowReidentification: boolean("allow_reidentification").default(false),
  
  // Data retention preferences
  dataRetentionYears: integer("data_retention_years").default(10),
  allowDataExport: boolean("allow_data_export").default(false),
  
  // Study participation preferences
  allowStudyEnrollment: boolean("allow_study_enrollment").default(true),
  allowContactForStudies: boolean("allow_contact_for_studies").default(false),
  preferredContactMethod: varchar("preferred_contact_method").default("email"), // 'email', 'phone', 'app'
  
  // Consent tracking
  consentVersion: varchar("consent_version").default("1.0"),
  consentSignedAt: timestamp("consent_signed_at"),
  consentUpdatedAt: timestamp("consent_updated_at"),
  consentWithdrawnAt: timestamp("consent_withdrawn_at"),
  withdrawalReason: text("withdrawal_reason"),
  
  // Legal
  legalBasis: varchar("legal_basis").default("consent"), // 'consent', 'public_interest', 'research_exemption'
  ethicsApprovalRef: varchar("ethics_approval_ref"),
  
  // Audit trail
  lastModifiedBy: varchar("last_modified_by"),
  ipAddress: varchar("ip_address"),
  userAgent: text("user_agent"),
  
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
}, (table) => ({
  patientIdx: index("research_consent_patient_idx").on(table.patientId),
  enabledIdx: index("research_consent_enabled_idx").on(table.consentEnabled),
}));

export const insertResearchDataConsentSchema = createInsertSchema(researchDataConsent).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export type InsertResearchDataConsent = z.infer<typeof insertResearchDataConsentSchema>;
export type ResearchDataConsent = typeof researchDataConsent.$inferSelect;

// Research Projects - Personal research mode for doctors
export const researchProjects = pgTable("research_projects", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  ownerId: varchar("owner_id").notNull().references(() => users.id),
  
  name: varchar("name").notNull(),
  description: text("description"),
  
  // Project type
  projectType: varchar("project_type").default("personal"), // 'personal', 'institutional', 'collaborative'
  
  // Status
  status: varchar("status").default("active"), // 'draft', 'active', 'paused', 'completed', 'archived'
  
  // Collaborators
  collaboratorIds: jsonb("collaborator_ids").$type<string[]>().default([]),
  
  // Settings
  isPublic: boolean("is_public").default(false),
  allowDataSharing: boolean("allow_data_sharing").default(false),
  
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
}, (table) => ({
  ownerIdx: index("research_project_owner_idx").on(table.ownerId),
  statusIdx: index("research_project_status_idx").on(table.status),
}));

export const insertResearchProjectSchema = createInsertSchema(researchProjects).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export type InsertResearchProject = z.infer<typeof insertResearchProjectSchema>;
export type ResearchProject = typeof researchProjects.$inferSelect;

// Research Studies - Full study lifecycle management
export const researchStudies = pgTable("research_studies", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  projectId: varchar("project_id").references(() => researchProjects.id),
  
  title: varchar("title").notNull(),
  description: text("description"),
  
  // Study status
  status: varchar("status").default("planning"), // 'planning', 'enrolling', 'follow_up', 'analysis', 'completed', 'terminated'
  
  // Timeline
  startDate: timestamp("start_date"),
  endDate: timestamp("end_date"),
  plannedDuration: integer("planned_duration"), // in days
  
  // Sample size
  targetSampleSize: integer("target_sample_size"),
  currentEnrollment: integer("current_enrollment").default(0),
  
  // Eligibility criteria
  inclusionCriteria: text("inclusion_criteria"),
  exclusionCriteria: text("exclusion_criteria"),
  inclusionCriteriaJson: jsonb("inclusion_criteria_json").$type<{
    minAge?: number;
    maxAge?: number;
    sex?: string[];
    conditions?: string[];
    medications?: string[];
    riskScoreMin?: number;
    riskScoreMax?: number;
    environmentalExposure?: Record<string, { min?: number; max?: number }>;
    immuneMarkers?: Record<string, { min?: number; max?: number }>;
  }>(),
  exclusionCriteriaJson: jsonb("exclusion_criteria_json").$type<{
    conditions?: string[];
    medications?: string[];
    excludeIfPregnant?: boolean;
    excludeIfImmunosuppressed?: boolean;
  }>(),
  
  // Study arms
  armsJson: jsonb("arms_json").$type<{
    armId: string;
    name: string;
    description: string;
    targetSize: number;
  }[]>().default([]),
  
  // Visit schedule
  visitScheduleJson: jsonb("visit_schedule_json").$type<{
    visitId: string;
    visitType: string;
    dayOffset: number;
    windowDays: number;
    required: boolean;
    assessments: string[];
  }[]>().default([]),
  
  // Auto-reanalysis configuration
  autoReanalysis: boolean("auto_reanalysis").default(false),
  reanalysisFrequency: varchar("reanalysis_frequency").default("weekly"), // 'daily', 'weekly', 'monthly'
  lastReanalysisAt: timestamp("last_reanalysis_at"),
  
  // Analysis configuration
  analysisSpecJson: jsonb("analysis_spec_json").$type<{
    primaryOutcome: string;
    secondaryOutcomes: string[];
    analysisType: string;
    modelType: string;
    covariates: string[];
    followupWindow: number;
  }>(),
  
  // Principal investigator
  ownerUserId: varchar("owner_user_id").notNull().references(() => users.id),
  
  // Linked cohort
  cohortId: varchar("cohort_id"),
  
  // Ethics and compliance
  ethicsApprovalNumber: varchar("ethics_approval_number"),
  ethicsApprovalDate: timestamp("ethics_approval_date"),
  dataProtectionAssessment: boolean("data_protection_assessment").default(false),
  
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
}, (table) => ({
  projectIdx: index("research_study_project_idx").on(table.projectId),
  statusIdx: index("research_study_status_idx").on(table.status),
  ownerIdx: index("research_study_owner_idx").on(table.ownerUserId),
}));

export const insertResearchStudySchema = createInsertSchema(researchStudies).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export type InsertResearchStudy = z.infer<typeof insertResearchStudySchema>;
export type ResearchStudy = typeof researchStudies.$inferSelect;

// Research Cohorts - Saved patient cohort definitions
export const researchCohorts = pgTable("research_cohorts", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  projectId: varchar("project_id").references(() => researchProjects.id),
  
  name: varchar("name").notNull(),
  description: text("description"),
  
  // Cohort definition (filter conditions)
  definitionJson: jsonb("definition_json").$type<{
    demographics: {
      minAge?: number;
      maxAge?: number;
      sex?: string[];
      birthYearMin?: number;
      birthYearMax?: number;
    };
    conditions: {
      include?: string[];
      exclude?: string[];
      categories?: string[];
    };
    medications: {
      include?: string[];
      exclude?: string[];
      classes?: string[];
    };
    riskScores: {
      deteriorationMin?: number;
      deteriorationMax?: number;
      compositeMin?: number;
      compositeMax?: number;
    };
    environmental: {
      aqiMin?: number;
      aqiMax?: number;
      pm25Min?: number;
      pm25Max?: number;
      pollenLevel?: string[];
    };
    immuneMarkers: {
      markerName?: string;
      minValue?: number;
      maxValue?: number;
    }[];
    followupPatterns: {
      minResponseRate?: number;
      avgSymptomScoreMin?: number;
      avgSymptomScoreMax?: number;
    };
    studyParticipation: string[];
    dateRanges: {
      enrollmentAfter?: string;
      enrollmentBefore?: string;
      lastFollowupAfter?: string;
    };
  }>().notNull(),
  
  // Computed stats (cached)
  cachedPatientCount: integer("cached_patient_count").default(0),
  cachedStats: jsonb("cached_stats").$type<{
    meanAge: number;
    genderDistribution: Record<string, number>;
    conditionDistribution: Record<string, number>;
    lastUpdated: string;
  }>(),
  
  // Status
  status: varchar("status").default("active"), // 'draft', 'active', 'archived'
  
  createdBy: varchar("created_by").notNull().references(() => users.id),
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
}, (table) => ({
  projectIdx: index("research_cohort_project_idx").on(table.projectId),
  createdByIdx: index("research_cohort_created_by_idx").on(table.createdBy),
}));

export const insertResearchCohortSchema = createInsertSchema(researchCohorts).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export type InsertResearchCohort = z.infer<typeof insertResearchCohortSchema>;
export type ResearchCohort = typeof researchCohorts.$inferSelect;

// Study Enrollments - Patient enrollment in studies
export const studyEnrollments = pgTable("study_enrollments", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  studyId: varchar("study_id").notNull().references(() => researchStudies.id),
  patientId: varchar("patient_id").notNull().references(() => users.id),
  
  // Study arm assignment
  arm: varchar("arm"),
  
  // Consent status
  consentStatus: varchar("consent_status").default("pending"), // 'pending', 'consented', 'withdrawn', 'declined'
  consentDate: timestamp("consent_date"),
  withdrawalDate: timestamp("withdrawal_date"),
  withdrawalReason: text("withdrawal_reason"),
  
  // Enrollment
  enrollmentDate: timestamp("enrollment_date").defaultNow(),
  screeningDate: timestamp("screening_date"),
  
  // Status
  status: varchar("status").default("active"), // 'screening', 'active', 'completed', 'withdrawn', 'lost_to_followup'
  
  // Notes
  notes: text("notes"),
  
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
}, (table) => ({
  studyIdx: index("study_enrollment_study_idx").on(table.studyId),
  patientIdx: index("study_enrollment_patient_idx").on(table.patientId),
  statusIdx: index("study_enrollment_status_idx").on(table.status),
  uniqueEnrollment: index("study_enrollment_unique_idx").on(table.studyId, table.patientId),
}));

export const insertStudyEnrollmentSchema = createInsertSchema(studyEnrollments).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export type InsertStudyEnrollment = z.infer<typeof insertStudyEnrollmentSchema>;
export type StudyEnrollment = typeof studyEnrollments.$inferSelect;

// Research Visits - Scheduled study visits
export const researchVisits = pgTable("research_visits", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  enrollmentId: varchar("enrollment_id").notNull().references(() => studyEnrollments.id),
  studyId: varchar("study_id").notNull().references(() => researchStudies.id),
  patientId: varchar("patient_id").notNull().references(() => users.id),
  
  // Visit type
  visitType: varchar("visit_type").notNull(), // 'screening', 'baseline', 'month_1', 'month_3', 'month_6', 'month_12', 'unscheduled'
  visitNumber: integer("visit_number"),
  
  // Scheduling
  scheduledDate: timestamp("scheduled_date"),
  windowStartDate: timestamp("window_start_date"),
  windowEndDate: timestamp("window_end_date"),
  actualDate: timestamp("actual_date"),
  
  // Status
  visitStatus: varchar("visit_status").default("scheduled"), // 'scheduled', 'completed', 'missed', 'cancelled', 'rescheduled'
  
  // Assessments completed
  assessmentsCompleted: jsonb("assessments_completed").$type<string[]>().default([]),
  
  // Notes
  notes: text("notes"),
  
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
}, (table) => ({
  enrollmentIdx: index("research_visit_enrollment_idx").on(table.enrollmentId),
  studyIdx: index("research_visit_study_idx").on(table.studyId),
  patientIdx: index("research_visit_patient_idx").on(table.patientId),
  scheduledIdx: index("research_visit_scheduled_idx").on(table.scheduledDate),
}));

export const insertResearchVisitSchema = createInsertSchema(researchVisits).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export type InsertResearchVisit = z.infer<typeof insertResearchVisitSchema>;
export type ResearchVisit = typeof researchVisits.$inferSelect;

// Research Measurements - Study-specific measurements
export const researchMeasurements = pgTable("research_measurements", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  visitId: varchar("visit_id").references(() => researchVisits.id),
  patientId: varchar("patient_id").notNull().references(() => users.id),
  studyId: varchar("study_id").references(() => researchStudies.id),
  
  // Measurement details
  name: varchar("name").notNull(), // e.g., 'systolic_bp', 'creatinine', 'phq9_score'
  value: varchar("value").notNull(), // stored as string, can be numeric or categorical
  valueNumeric: decimal("value_numeric", { precision: 15, scale: 5 }), // parsed numeric value
  units: varchar("units"),
  
  // Classification
  category: varchar("category"), // 'vital', 'lab', 'questionnaire', 'imaging', 'physical_exam'
  
  // Recording details
  recordedAt: timestamp("recorded_at").notNull().defaultNow(),
  recordedBy: varchar("recorded_by"),
  source: varchar("source").default("manual"), // 'manual', 'device', 'ehr_import', 'lab_import'
  
  // Quality
  isValid: boolean("is_valid").default(true),
  validationNotes: text("validation_notes"),
  
  createdAt: timestamp("created_at").defaultNow(),
}, (table) => ({
  visitIdx: index("research_measurement_visit_idx").on(table.visitId),
  patientIdx: index("research_measurement_patient_idx").on(table.patientId),
  nameIdx: index("research_measurement_name_idx").on(table.name),
  recordedAtIdx: index("research_measurement_recorded_idx").on(table.recordedAt),
}));

export const insertResearchMeasurementSchema = createInsertSchema(researchMeasurements).omit({
  id: true,
  createdAt: true,
});

export type InsertResearchMeasurement = z.infer<typeof insertResearchMeasurementSchema>;
export type ResearchMeasurement = typeof researchMeasurements.$inferSelect;

// Immune Markers - Specialized immune monitoring for research
export const researchImmuneMarkers = pgTable("research_immune_markers", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  patientId: varchar("patient_id").notNull().references(() => users.id),
  studyId: varchar("study_id").references(() => researchStudies.id),
  visitId: varchar("visit_id").references(() => researchVisits.id),
  
  // Marker details
  markerName: varchar("marker_name").notNull(), // e.g., 'CD4', 'CD8', 'IL6', 'TNF_alpha', 'CRP'
  value: decimal("value", { precision: 15, scale: 5 }).notNull(),
  units: varchar("units").notNull(),
  
  // Reference ranges
  referenceRangeLow: decimal("reference_range_low", { precision: 15, scale: 5 }),
  referenceRangeHigh: decimal("reference_range_high", { precision: 15, scale: 5 }),
  
  // Interpretation
  interpretation: varchar("interpretation"), // 'normal', 'low', 'high', 'critical_low', 'critical_high'
  
  // Source
  source: varchar("source").default("lab"), // 'lab', 'flow_cytometry', 'elisa', 'pcr'
  labName: varchar("lab_name"),
  specimenType: varchar("specimen_type"), // 'blood', 'serum', 'plasma', 'csf'
  
  // Timing
  collectionTime: timestamp("collection_time").notNull(),
  resultTime: timestamp("result_time"),
  
  createdAt: timestamp("created_at").defaultNow(),
}, (table) => ({
  patientIdx: index("immune_marker_patient_idx").on(table.patientId),
  markerIdx: index("immune_marker_name_idx").on(table.markerName),
  collectionIdx: index("immune_marker_collection_idx").on(table.collectionTime),
}));

export const insertResearchImmuneMarkerSchema = createInsertSchema(researchImmuneMarkers).omit({
  id: true,
  createdAt: true,
});

export type InsertResearchImmuneMarker = z.infer<typeof insertResearchImmuneMarkerSchema>;
export type ResearchImmuneMarker = typeof researchImmuneMarkers.$inferSelect;

// Research Locations - Geographic locations for environmental exposure tracking
export const researchLocations = pgTable("research_locations", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  
  name: varchar("name").notNull(),
  region: varchar("region"),
  city: varchar("city"),
  state: varchar("state"),
  country: varchar("country").default("USA"),
  zipCode: varchar("zip_code"),
  
  // Coordinates
  latitude: decimal("latitude", { precision: 10, scale: 7 }),
  longitude: decimal("longitude", { precision: 10, scale: 7 }),
  
  // Population data
  population: integer("population"),
  urbanRuralClassification: varchar("urban_rural_classification"), // 'urban', 'suburban', 'rural'
  
  createdAt: timestamp("created_at").defaultNow(),
}, (table) => ({
  zipCodeIdx: index("research_location_zipcode_idx").on(table.zipCode),
  coordinatesIdx: index("research_location_coords_idx").on(table.latitude, table.longitude),
}));

export const insertResearchLocationSchema = createInsertSchema(researchLocations).omit({
  id: true,
  createdAt: true,
});

export type InsertResearchLocation = z.infer<typeof insertResearchLocationSchema>;
export type ResearchLocation = typeof researchLocations.$inferSelect;

// Patient Locations - Track patient locations over time
export const researchPatientLocations = pgTable("research_patient_locations", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  patientId: varchar("patient_id").notNull().references(() => users.id),
  locationId: varchar("location_id").notNull().references(() => researchLocations.id),
  
  // Date range
  startDate: timestamp("start_date").notNull(),
  endDate: timestamp("end_date"),
  
  // Location type
  locationType: varchar("location_type").default("residence"), // 'residence', 'work', 'temporary'
  isPrimary: boolean("is_primary").default(true),
  
  createdAt: timestamp("created_at").defaultNow(),
}, (table) => ({
  patientIdx: index("patient_location_patient_idx").on(table.patientId),
  locationIdx: index("patient_location_location_idx").on(table.locationId),
  dateIdx: index("patient_location_date_idx").on(table.startDate, table.endDate),
}));

export const insertResearchPatientLocationSchema = createInsertSchema(researchPatientLocations).omit({
  id: true,
  createdAt: true,
});

export type InsertResearchPatientLocation = z.infer<typeof insertResearchPatientLocationSchema>;
export type ResearchPatientLocation = typeof researchPatientLocations.$inferSelect;

// Environmental Exposures - Daily environmental data per location
export const researchEnvironmentalExposures = pgTable("research_environmental_exposures", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  locationId: varchar("location_id").notNull().references(() => researchLocations.id),
  
  // Date
  date: timestamp("date").notNull(),
  
  // Air quality
  pm25: decimal("pm25", { precision: 8, scale: 3 }),
  pm10: decimal("pm10", { precision: 8, scale: 3 }),
  ozone: decimal("ozone", { precision: 8, scale: 3 }),
  airQualityIndex: integer("air_quality_index"),
  airQualityCategory: varchar("air_quality_category"), // 'good', 'moderate', 'unhealthy_sensitive', 'unhealthy', 'very_unhealthy', 'hazardous'
  
  // Weather
  temperatureHigh: decimal("temperature_high", { precision: 5, scale: 2 }),
  temperatureLow: decimal("temperature_low", { precision: 5, scale: 2 }),
  temperatureAvg: decimal("temperature_avg", { precision: 5, scale: 2 }),
  humidity: integer("humidity"),
  barometricPressure: decimal("barometric_pressure", { precision: 7, scale: 2 }),
  precipitation: decimal("precipitation", { precision: 6, scale: 2 }),
  uvIndex: integer("uv_index"),
  
  // Allergens
  pollenIndex: integer("pollen_index"),
  pollenTree: integer("pollen_tree"),
  pollenGrass: integer("pollen_grass"),
  pollenRagweed: integer("pollen_ragweed"),
  moldIndex: integer("mold_index"),
  
  // Other environmental factors
  wildfiresNearby: boolean("wildfires_nearby").default(false),
  dustStorm: boolean("dust_storm").default(false),
  
  // Data source
  dataSource: varchar("data_source"), // 'api', 'manual', 'imported'
  
  createdAt: timestamp("created_at").defaultNow(),
}, (table) => ({
  locationDateIdx: index("env_exposure_location_date_idx").on(table.locationId, table.date),
  dateIdx: index("env_exposure_date_idx").on(table.date),
}));

export const insertResearchEnvironmentalExposureSchema = createInsertSchema(researchEnvironmentalExposures).omit({
  id: true,
  createdAt: true,
});

export type InsertResearchEnvironmentalExposure = z.infer<typeof insertResearchEnvironmentalExposureSchema>;
export type ResearchEnvironmentalExposure = typeof researchEnvironmentalExposures.$inferSelect;

// Data Snapshots - Version control for analyses
export const researchDataSnapshots = pgTable("research_data_snapshots", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  
  description: varchar("description"),
  
  // Scope
  studyId: varchar("study_id").references(() => researchStudies.id),
  cohortId: varchar("cohort_id").references(() => researchCohorts.id),
  
  // Counts at snapshot time
  patientCount: integer("patient_count"),
  measurementCount: integer("measurement_count"),
  followupCount: integer("followup_count"),
  
  // Hash for data integrity
  dataHash: varchar("data_hash"),
  
  createdBy: varchar("created_by").references(() => users.id),
  createdAt: timestamp("created_at").defaultNow(),
}, (table) => ({
  studyIdx: index("data_snapshot_study_idx").on(table.studyId),
  createdAtIdx: index("data_snapshot_created_idx").on(table.createdAt),
}));

export const insertResearchDataSnapshotSchema = createInsertSchema(researchDataSnapshots).omit({
  id: true,
  createdAt: true,
});

export type InsertResearchDataSnapshot = z.infer<typeof insertResearchDataSnapshotSchema>;
export type ResearchDataSnapshot = typeof researchDataSnapshots.$inferSelect;

// Enhanced Research Analysis Reports - Detailed ML/statistical analysis reports
export const researchAnalysisReports = pgTable("research_analysis_reports", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  
  title: varchar("title").notNull(),
  
  // Links
  studyId: varchar("study_id").references(() => researchStudies.id),
  cohortId: varchar("cohort_id").references(() => researchCohorts.id),
  snapshotId: varchar("snapshot_id").references(() => researchDataSnapshots.id),
  
  // Analysis configuration
  analysisType: varchar("analysis_type").notNull(), // 'descriptive', 'risk_prediction', 'survival', 'causal', 'correlation'
  analysisSpecJson: jsonb("analysis_spec_json").$type<{
    outcome: string;
    outcomeType: 'binary' | 'continuous' | 'time_to_event';
    exposure?: string;
    covariates: string[];
    modelType: string;
    followupWindow: number;
    cvFolds?: number;
    regularization?: string;
  }>(),
  
  // Results
  resultsJson: jsonb("results_json").$type<{
    metrics: Record<string, number>;
    coefficients?: Record<string, { estimate: number; ci_low: number; ci_high: number; pvalue: number }>;
    featureImportance?: Record<string, number>;
    baselineCharacteristics?: Record<string, { mean?: number; sd?: number; median?: number; count?: number; percent?: number }>;
    survivalData?: { time: number; survival: number; ci_low: number; ci_high: number }[];
    rocData?: { fpr: number; tpr: number }[];
    calibrationData?: { predicted: number; observed: number }[];
  }>(),
  
  // AI-generated narrative
  generatedText: text("generated_text"),
  
  // Report sections
  abstract: text("abstract"),
  methods: text("methods"),
  results: text("results"),
  discussion: text("discussion"),
  limitations: text("limitations"),
  
  // Status
  status: varchar("status").default("draft"), // 'draft', 'generating', 'review', 'finalized', 'published'
  
  // Publish settings
  publishReady: boolean("publish_ready").default(false),
  publishedAt: timestamp("published_at"),
  
  createdBy: varchar("created_by").notNull().references(() => users.id),
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
}, (table) => ({
  studyIdx: index("analysis_report_study_idx").on(table.studyId),
  cohortIdx: index("analysis_report_cohort_idx").on(table.cohortId),
  typeIdx: index("analysis_report_type_idx").on(table.analysisType),
  statusIdx: index("analysis_report_status_idx").on(table.status),
}));

export const insertResearchAnalysisReportSchema = createInsertSchema(researchAnalysisReports).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export type InsertResearchAnalysisReport = z.infer<typeof insertResearchAnalysisReportSchema>;
export type ResearchAnalysisReport = typeof researchAnalysisReports.$inferSelect;

// Research Alerts - Health and research alerts for studies
export const researchAlerts = pgTable("research_alerts", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  patientId: varchar("patient_id").references(() => users.id),
  studyId: varchar("study_id").references(() => researchStudies.id),
  
  // Alert type
  alertType: varchar("alert_type").notNull(), // 'deterioration_risk', 'immune_drop', 'environmental_risk', 'adherence_issue', 'data_quality', 'protocol_deviation', 'threshold_breach'
  
  // Severity
  severity: varchar("severity").notNull(), // 'low', 'medium', 'high', 'critical'
  riskScore: decimal("risk_score", { precision: 5, scale: 2 }),
  
  // Content
  title: varchar("title").notNull(),
  message: text("message"),
  
  // Details (SHAP values, top features, thresholds)
  detailsJson: jsonb("details_json").$type<{
    topFeatures: { feature: string; value: number; contribution: number }[];
    threshold: number;
    actualValue: number;
    percentOverThreshold: number;
    recommendations: string[];
    trendData?: { date: string; value: number }[];
  }>(),
  
  // Status
  status: varchar("status").default("new"), // 'new', 'acknowledged', 'investigating', 'resolved', 'dismissed'
  acknowledgedAt: timestamp("acknowledged_at"),
  acknowledgedBy: varchar("acknowledged_by").references(() => users.id),
  resolvedAt: timestamp("resolved_at"),
  resolvedBy: varchar("resolved_by").references(() => users.id),
  resolution: text("resolution"),
  
  createdAt: timestamp("created_at").defaultNow(),
}, (table) => ({
  patientIdx: index("research_alert_patient_idx").on(table.patientId),
  studyIdx: index("research_alert_study_idx").on(table.studyId),
  typeIdx: index("research_alert_type_idx").on(table.alertType),
  statusIdx: index("research_alert_status_idx").on(table.status),
  severityIdx: index("research_alert_severity_idx").on(table.severity),
}));

export const insertResearchAlertSchema = createInsertSchema(researchAlerts).omit({
  id: true,
  createdAt: true,
});

export type InsertResearchAlert = z.infer<typeof insertResearchAlertSchema>;
export type ResearchAlert = typeof researchAlerts.$inferSelect;

// Daily Followup Templates - Reusable questionnaire templates
export const dailyFollowupTemplates = pgTable("daily_followup_templates", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  
  name: varchar("name").notNull(),
  description: text("description"),
  
  // Question definitions
  questionsJson: jsonb("questions_json").$type<{
    questionId: string;
    questionText: string;
    questionType: 'text' | 'numeric' | 'scale' | 'yes_no' | 'multiple_choice' | 'likert';
    required: boolean;
    options?: string[];
    validation?: {
      min?: number;
      max?: number;
      pattern?: string;
    };
    conditionalOn?: {
      questionId: string;
      value: string | number | boolean;
    };
  }[]>().notNull(),
  
  // Template settings
  estimatedDuration: integer("estimated_duration"), // in minutes
  category: varchar("category"), // 'symptoms', 'mood', 'medication', 'activity', 'general'
  
  // Status
  isActive: boolean("is_active").default(true),
  version: integer("version").default(1),
  
  createdBy: varchar("created_by").notNull().references(() => users.id),
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
}, (table) => ({
  categoryIdx: index("followup_template_category_idx").on(table.category),
  activeIdx: index("followup_template_active_idx").on(table.isActive),
}));

export const insertDailyFollowupTemplateSchema = createInsertSchema(dailyFollowupTemplates).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export type InsertDailyFollowupTemplate = z.infer<typeof insertDailyFollowupTemplateSchema>;
export type DailyFollowupTemplate = typeof dailyFollowupTemplates.$inferSelect;

// Daily Followup Assignments - Assign templates to patients
export const dailyFollowupAssignments = pgTable("daily_followup_assignments", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  templateId: varchar("template_id").notNull().references(() => dailyFollowupTemplates.id),
  patientId: varchar("patient_id").notNull().references(() => users.id),
  studyId: varchar("study_id").references(() => researchStudies.id),
  
  // Schedule
  frequency: varchar("frequency").default("daily"), // 'daily', 'weekly', 'biweekly', 'monthly'
  startDate: timestamp("start_date").notNull(),
  endDate: timestamp("end_date"),
  
  // Status
  isActive: boolean("is_active").default(true),
  
  // Notification settings
  notifyAt: varchar("notify_at"), // time of day, e.g., "09:00"
  reminderEnabled: boolean("reminder_enabled").default(true),
  
  createdBy: varchar("created_by").notNull().references(() => users.id),
  createdAt: timestamp("created_at").defaultNow(),
}, (table) => ({
  templateIdx: index("followup_assignment_template_idx").on(table.templateId),
  patientIdx: index("followup_assignment_patient_idx").on(table.patientId),
  studyIdx: index("followup_assignment_study_idx").on(table.studyId),
  activeIdx: index("followup_assignment_active_idx").on(table.isActive),
}));

export const insertDailyFollowupAssignmentSchema = createInsertSchema(dailyFollowupAssignments).omit({
  id: true,
  createdAt: true,
});

export type InsertDailyFollowupAssignment = z.infer<typeof insertDailyFollowupAssignmentSchema>;
export type DailyFollowupAssignment = typeof dailyFollowupAssignments.$inferSelect;

// Daily Followup Responses - Patient responses to templates
export const dailyFollowupResponses = pgTable("daily_followup_responses", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  assignmentId: varchar("assignment_id").notNull().references(() => dailyFollowupAssignments.id),
  patientId: varchar("patient_id").notNull().references(() => users.id),
  
  // Response date
  responseDate: timestamp("response_date").notNull(),
  
  // Answers
  responsesJson: jsonb("responses_json").$type<{
    questionId: string;
    answer: string | number | boolean | string[];
  }[]>().notNull(),
  
  // Completion
  completedAt: timestamp("completed_at"),
  isComplete: boolean("is_complete").default(false),
  
  // Computed scores (for quick aggregation)
  overallScore: decimal("overall_score", { precision: 5, scale: 2 }),
  symptomScore: decimal("symptom_score", { precision: 5, scale: 2 }),
  moodScore: decimal("mood_score", { precision: 5, scale: 2 }),
  
  createdAt: timestamp("created_at").defaultNow(),
}, (table) => ({
  assignmentIdx: index("followup_response_assignment_idx").on(table.assignmentId),
  patientIdx: index("followup_response_patient_idx").on(table.patientId),
  dateIdx: index("followup_response_date_idx").on(table.responseDate),
}));

export const insertDailyFollowupResponseSchema = createInsertSchema(dailyFollowupResponses).omit({
  id: true,
  createdAt: true,
});

export type InsertDailyFollowupResponse = z.infer<typeof insertDailyFollowupResponseSchema>;
export type DailyFollowupResponse = typeof dailyFollowupResponses.$inferSelect;

// Research Audit Logs - HIPAA-compliant audit trail
export const researchAuditLogs = pgTable("research_audit_logs", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  userId: varchar("user_id").notNull().references(() => users.id),
  
  // Action
  actionType: varchar("action_type").notNull(), // 'VIEW_DATA', 'EXPORT_CSV', 'RUN_ANALYSIS', 'CREATE_REPORT', 'MODIFY_STUDY', 'ENROLL_PATIENT', 'ACCESS_PHI'
  
  // Object
  objectType: varchar("object_type").notNull(), // 'study', 'cohort', 'report', 'patient', 'enrollment', 'measurement'
  objectId: varchar("object_id").notNull(),
  
  // Details
  metadataJson: jsonb("metadata_json").$type<{
    description?: string;
    patientIds?: string[];
    dataTypes?: string[];
    recordCount?: number;
    exportFormat?: string;
    ipAddress?: string;
    userAgent?: string;
  }>(),
  
  // Outcome
  success: boolean("success").default(true),
  errorMessage: text("error_message"),
  
  createdAt: timestamp("created_at").defaultNow(),
}, (table) => ({
  userIdx: index("research_audit_user_idx").on(table.userId),
  actionIdx: index("research_audit_action_idx").on(table.actionType),
  objectIdx: index("research_audit_object_idx").on(table.objectType, table.objectId),
  createdAtIdx: index("research_audit_created_idx").on(table.createdAt),
}));

export const insertResearchAuditLogSchema = createInsertSchema(researchAuditLogs).omit({
  id: true,
  createdAt: true,
});

export type InsertResearchAuditLog = z.infer<typeof insertResearchAuditLogSchema>;
export type ResearchAuditLog = typeof researchAuditLogs.$inferSelect;

// Analysis Jobs - Track background analysis executions
export const analysisJobs = pgTable("analysis_jobs", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  
  // Job type
  jobType: varchar("job_type").notNull(), // 'descriptive', 'risk_prediction', 'survival', 'causal', 'alert_check', 'reanalysis'
  
  // Links
  studyId: varchar("study_id").references(() => researchStudies.id),
  cohortId: varchar("cohort_id").references(() => researchCohorts.id),
  reportId: varchar("report_id").references(() => aiResearchReports.id),
  
  // Configuration
  configJson: jsonb("config_json").$type<Record<string, unknown>>(),
  
  // Status
  status: varchar("status").notNull().default("pending"), // 'pending', 'running', 'completed', 'failed', 'cancelled'
  progress: integer("progress").default(0), // 0-100
  currentPhase: varchar("current_phase"),
  
  // Timing
  startedAt: timestamp("started_at"),
  completedAt: timestamp("completed_at"),
  executionTimeMs: integer("execution_time_ms"),
  
  // Results
  resultSummary: jsonb("result_summary").$type<{
    recordsProcessed: number;
    metricsComputed: string[];
    alertsGenerated: number;
  }>(),
  
  // Errors
  errorMessage: text("error_message"),
  errorStack: text("error_stack"),
  retryCount: integer("retry_count").default(0),
  
  // Trigger
  triggerSource: varchar("trigger_source").default("manual"), // 'manual', 'scheduled', 'auto_reanalysis'
  
  createdBy: varchar("created_by").references(() => users.id),
  createdAt: timestamp("created_at").defaultNow(),
}, (table) => ({
  statusIdx: index("analysis_job_status_idx").on(table.status),
  studyIdx: index("analysis_job_study_idx").on(table.studyId),
  typeIdx: index("analysis_job_type_idx").on(table.jobType),
}));

export const insertAnalysisJobSchema = createInsertSchema(analysisJobs).omit({
  id: true,
  createdAt: true,
});

export type InsertAnalysisJob = z.infer<typeof insertAnalysisJobSchema>;
export type AnalysisJob = typeof analysisJobs.$inferSelect;

// ============================================
// RISK & EXPOSURES TABLES
// ============================================

// Healthcare Locations - For tracking where events occurred
export const healthcareLocations = pgTable("healthcare_locations", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  name: varchar("name").notNull(),
  locationType: varchar("location_type").notNull(), // 'hospital', 'clinic', 'pharmacy', 'lab', 'home', 'other'
  address: text("address"),
  city: varchar("city"),
  state: varchar("state"),
  zipCode: varchar("zip_code"),
  country: varchar("country").default("USA"),
  latitude: decimal("latitude"),
  longitude: decimal("longitude"),
  phoneNumber: varchar("phone_number"),
  isActive: boolean("is_active").default(true),
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
});

export const insertHealthcareLocationSchema = createInsertSchema(healthcareLocations).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export type InsertHealthcareLocation = z.infer<typeof insertHealthcareLocationSchema>;
export type HealthcareLocation = typeof healthcareLocations.$inferSelect;

// Patient Conditions - Diagnosis codes and conditions
export const patientConditions = pgTable("patient_conditions", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  patientId: varchar("patient_id").notNull().references(() => users.id),
  
  // Diagnosis information
  conditionCode: varchar("condition_code").notNull(), // ICD-10 code
  conditionName: varchar("condition_name").notNull(),
  conditionCategory: varchar("condition_category"), // 'infectious', 'chronic', 'acute', 'genetic', 'other'
  
  // Status
  status: varchar("status").default("active"), // 'active', 'resolved', 'chronic', 'in_remission'
  severity: varchar("severity"), // 'mild', 'moderate', 'severe'
  
  // Dates
  onsetDate: timestamp("onset_date"),
  diagnosisDate: timestamp("diagnosis_date"),
  resolutionDate: timestamp("resolution_date"),
  
  // Source
  diagnosedBy: varchar("diagnosed_by").references(() => users.id),
  sourceType: varchar("source_type").default("ehr"), // 'ehr', 'manual', 'import', 'self_reported'
  sourceRecordId: varchar("source_record_id"),
  
  // Notes
  notes: text("notes"),
  
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
}, (table) => ({
  patientIdx: index("patient_condition_patient_idx").on(table.patientId),
  codeIdx: index("patient_condition_code_idx").on(table.conditionCode),
  statusIdx: index("patient_condition_status_idx").on(table.status),
  categoryIdx: index("patient_condition_category_idx").on(table.conditionCategory),
}));

export const insertPatientConditionSchema = createInsertSchema(patientConditions).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export type InsertPatientCondition = z.infer<typeof insertPatientConditionSchema>;
export type PatientCondition = typeof patientConditions.$inferSelect;

// Patient Visits - Hospitalizations and clinic visits
export const patientVisits = pgTable("patient_visits", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  patientId: varchar("patient_id").notNull().references(() => users.id),
  
  // Visit type
  visitType: varchar("visit_type").notNull(), // 'outpatient', 'inpatient', 'emergency', 'telehealth', 'home_visit'
  
  // Location
  locationId: varchar("location_id").references(() => healthcareLocations.id),
  facilityName: varchar("facility_name"),
  
  // Timing
  admissionDate: timestamp("admission_date").notNull(),
  dischargeDate: timestamp("discharge_date"),
  lengthOfStay: integer("length_of_stay"), // in days
  
  // Severity indicators
  isHospitalization: boolean("is_hospitalization").default(false),
  icuAdmission: boolean("icu_admission").default(false),
  icuDays: integer("icu_days"),
  ventilatorRequired: boolean("ventilator_required").default(false),
  
  // Reason
  chiefComplaint: text("chief_complaint"),
  primaryDiagnosisCode: varchar("primary_diagnosis_code"),
  secondaryDiagnosesCodes: jsonb("secondary_diagnoses_codes").$type<string[]>(),
  
  // Outcome
  dischargeDisposition: varchar("discharge_disposition"), // 'home', 'skilled_nursing', 'rehab', 'deceased', 'other'
  
  // Source
  sourceType: varchar("source_type").default("ehr"), // 'ehr', 'manual', 'import'
  sourceRecordId: varchar("source_record_id"),
  
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
}, (table) => ({
  patientIdx: index("patient_visit_patient_idx").on(table.patientId),
  dateIdx: index("patient_visit_date_idx").on(table.admissionDate),
  typeIdx: index("patient_visit_type_idx").on(table.visitType),
}));

export const insertPatientVisitSchema = createInsertSchema(patientVisits).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export type InsertPatientVisit = z.infer<typeof insertPatientVisitSchema>;
export type PatientVisit = typeof patientVisits.$inferSelect;

// Infectious Events - Auto-derived from conditions + visits
export const infectiousEvents = pgTable("infectious_events", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  patientId: varchar("patient_id").notNull().references(() => users.id),
  
  // Infection details
  infectionType: varchar("infection_type").notNull(), // 'pneumonia', 'UTI', 'COVID-19', 'influenza', 'sepsis', etc.
  pathogen: varchar("pathogen"), // 'Streptococcus pneumoniae', 'E. coli', 'SARS-CoV-2', etc.
  pathogenCategory: varchar("pathogen_category"), // 'bacterial', 'viral', 'fungal', 'parasitic', 'unknown'
  
  // Severity
  severity: varchar("severity").notNull().default("moderate"), // 'mild', 'moderate', 'severe', 'critical'
  
  // Timeline
  onsetDate: timestamp("onset_date"),
  resolutionDate: timestamp("resolution_date"),
  durationDays: integer("duration_days"),
  
  // Hospitalization
  hospitalization: boolean("hospitalization").default(false),
  icuAdmission: boolean("icu_admission").default(false),
  ventilatorRequired: boolean("ventilator_required").default(false),
  
  // Source tracking (for ETL)
  relatedConditionId: varchar("related_condition_id").references(() => patientConditions.id),
  relatedConditionCode: varchar("related_condition_code"),
  relatedVisitId: varchar("related_visit_id").references(() => patientVisits.id),
  locationId: varchar("location_id").references(() => healthcareLocations.id),
  
  // Override tracking
  autoGenerated: boolean("auto_generated").default(true),
  manualOverride: boolean("manual_override").default(false),
  overriddenBy: varchar("overridden_by").references(() => users.id),
  overriddenAt: timestamp("overridden_at"),
  doctorNotes: text("doctor_notes"),
  
  // Processing metadata
  lastEtlProcessedAt: timestamp("last_etl_processed_at"),
  
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
}, (table) => ({
  patientIdx: index("infectious_event_patient_idx").on(table.patientId),
  typeIdx: index("infectious_event_type_idx").on(table.infectionType),
  onsetIdx: index("infectious_event_onset_idx").on(table.onsetDate),
  severityIdx: index("infectious_event_severity_idx").on(table.severity),
}));

export const insertInfectiousEventSchema = createInsertSchema(infectiousEvents).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export type InsertInfectiousEvent = z.infer<typeof insertInfectiousEventSchema>;
export type InfectiousEvent = typeof infectiousEvents.$inferSelect;

// Immunizations - Vaccine records
export const patientImmunizations = pgTable("patient_immunizations", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  patientId: varchar("patient_id").notNull().references(() => users.id),
  
  // Vaccine information
  vaccineCode: varchar("vaccine_code"), // CVX code
  vaccineName: varchar("vaccine_name").notNull(),
  vaccineManufacturer: varchar("vaccine_manufacturer"),
  lotNumber: varchar("lot_number"),
  
  // Dosing
  doseNumber: integer("dose_number"),
  seriesName: varchar("series_name"), // 'primary', 'booster', 'annual'
  seriesComplete: boolean("series_complete").default(false),
  
  // Administration
  administrationDate: timestamp("administration_date").notNull(),
  administrationRoute: varchar("administration_route"), // 'intramuscular', 'subcutaneous', 'oral', 'intranasal'
  administrationSite: varchar("administration_site"), // 'left_arm', 'right_arm', 'thigh', etc.
  
  // Location
  locationId: varchar("location_id").references(() => healthcareLocations.id),
  administeredBy: varchar("administered_by").references(() => users.id),
  
  // Status
  status: varchar("status").default("completed"), // 'completed', 'not_done', 'entered_in_error'
  
  // Reactions
  adverseReaction: boolean("adverse_reaction").default(false),
  reactionDetails: text("reaction_details"),
  
  // Source tracking
  sourceType: varchar("source_type").default("ehr"), // 'ehr', 'import', 'manual', 'prescription'
  sourceRecordId: varchar("source_record_id"),
  
  // Override tracking
  autoGenerated: boolean("auto_generated").default(false),
  manualOverride: boolean("manual_override").default(false),
  overriddenBy: varchar("overridden_by").references(() => users.id),
  overriddenAt: timestamp("overridden_at"),
  
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
}, (table) => ({
  patientIdx: index("immunization_patient_idx").on(table.patientId),
  vaccineIdx: index("immunization_vaccine_idx").on(table.vaccineCode),
  dateIdx: index("immunization_date_idx").on(table.administrationDate),
}));

export const insertPatientImmunizationSchema = createInsertSchema(patientImmunizations).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export type InsertPatientImmunization = z.infer<typeof insertPatientImmunizationSchema>;
export type PatientImmunization = typeof patientImmunizations.$inferSelect;

// Occupations - Patient job history
export const patientOccupations = pgTable("patient_occupations", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  patientId: varchar("patient_id").notNull().references(() => users.id),
  
  // Job details
  jobTitle: varchar("job_title").notNull(),
  industry: varchar("industry"), // 'healthcare', 'construction', 'manufacturing', 'office', etc.
  employer: varchar("employer"),
  
  // Work characteristics
  physicalDemandLevel: varchar("physical_demand_level"), // 'sedentary', 'light', 'moderate', 'heavy', 'very_heavy'
  shiftWork: boolean("shift_work").default(false),
  nightShift: boolean("night_shift").default(false),
  hoursPerWeek: integer("hours_per_week"),
  
  // Duration
  startDate: timestamp("start_date"),
  endDate: timestamp("end_date"),
  isCurrent: boolean("is_current").default(true),
  
  // Status
  status: varchar("status").default("active"), // 'active', 'inactive', 'retired'
  
  // Auto-enrichment tracking
  autoEnriched: boolean("auto_enriched").default(false),
  enrichedAt: timestamp("enriched_at"),
  
  // Doctor modification tracking
  createdBy: varchar("created_by").references(() => users.id),
  modifiedBy: varchar("modified_by").references(() => users.id),
  modifiedAt: timestamp("modified_at"),
  doctorNotes: text("doctor_notes"),
  
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
}, (table) => ({
  patientIdx: index("occupation_patient_idx").on(table.patientId),
  industryIdx: index("occupation_industry_idx").on(table.industry),
  currentIdx: index("occupation_current_idx").on(table.isCurrent),
}));

export const insertPatientOccupationSchema = createInsertSchema(patientOccupations).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export type InsertPatientOccupation = z.infer<typeof insertPatientOccupationSchema>;
export type PatientOccupation = typeof patientOccupations.$inferSelect;

// Occupational Exposures - Hazards associated with jobs
export const occupationalExposures = pgTable("occupational_exposures", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  occupationId: varchar("occupation_id").notNull().references(() => patientOccupations.id, { onDelete: "cascade" }),
  patientId: varchar("patient_id").notNull().references(() => users.id),
  
  // Exposure details
  exposureType: varchar("exposure_type").notNull(), // 'dust', 'chemicals', 'noise', 'heat', 'radiation', 'biological', 'ergonomic'
  exposureAgent: varchar("exposure_agent"), // Specific agent: 'silica', 'asbestos', 'lead', 'benzene', etc.
  exposureLevel: varchar("exposure_level").notNull().default("medium"), // 'low', 'medium', 'high'
  
  // Duration and frequency
  exposureFrequency: varchar("exposure_frequency"), // 'daily', 'weekly', 'occasional', 'rare'
  exposureDurationYears: decimal("exposure_duration_years"),
  
  // Protection
  protectiveEquipmentUsed: boolean("protective_equipment_used").default(false),
  protectiveEquipmentDetails: text("protective_equipment_details"),
  
  // Health impact
  healthImpactNotes: text("health_impact_notes"),
  
  // Auto-generation tracking
  autoGenerated: boolean("auto_generated").default(true),
  manualOverride: boolean("manual_override").default(false),
  modifiedBy: varchar("modified_by").references(() => users.id),
  modifiedAt: timestamp("modified_at"),
  doctorNotes: text("doctor_notes"),
  
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
}, (table) => ({
  occupationIdx: index("exposure_occupation_idx").on(table.occupationId),
  patientIdx: index("exposure_patient_idx").on(table.patientId),
  typeIdx: index("exposure_type_idx").on(table.exposureType),
}));

export const insertOccupationalExposureSchema = createInsertSchema(occupationalExposures).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export type InsertOccupationalExposure = z.infer<typeof insertOccupationalExposureSchema>;
export type OccupationalExposure = typeof occupationalExposures.$inferSelect;

// Genetic Risk Flags - Auto-derived from lab/genetic results
export const geneticRiskFlags = pgTable("genetic_risk_flags", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  patientId: varchar("patient_id").notNull().references(() => users.id),
  
  // Flag details
  flagName: varchar("flag_name").notNull(), // 'HLA-B*57:01', 'Factor V Leiden', 'BRCA1', 'MTHFR C677T'
  flagType: varchar("flag_type").notNull(), // 'monogenic', 'pharmacogenomic', 'polygenic_score', 'family_history'
  
  // Value and interpretation
  value: varchar("value").notNull(), // 'present', 'absent', 'high', 'intermediate', 'low', 'carrier'
  riskLevel: varchar("risk_level"), // 'low', 'moderate', 'high', 'critical'
  
  // Clinical significance
  clinicalImplications: text("clinical_implications"),
  affectedMedications: jsonb("affected_medications").$type<string[]>(),
  affectedConditions: jsonb("affected_conditions").$type<string[]>(),
  
  // Source
  source: varchar("source").notNull(), // 'genetic_panel', 'lab_result', 'external_report', 'self_reported'
  sourceRecordId: varchar("source_record_id"),
  testingProvider: varchar("testing_provider"),
  recordedDate: timestamp("recorded_date"),
  
  // Override tracking
  autoGenerated: boolean("auto_generated").default(true),
  manualOverride: boolean("manual_override").default(false),
  overriddenBy: varchar("overridden_by").references(() => users.id),
  overriddenAt: timestamp("overridden_at"),
  doctorNotes: text("doctor_notes"),
  
  createdAt: timestamp("created_at").defaultNow(),
  updatedAt: timestamp("updated_at").defaultNow(),
}, (table) => ({
  patientIdx: index("genetic_flag_patient_idx").on(table.patientId),
  flagIdx: index("genetic_flag_name_idx").on(table.flagName),
  typeIdx: index("genetic_flag_type_idx").on(table.flagType),
}));

export const insertGeneticRiskFlagSchema = createInsertSchema(geneticRiskFlags).omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

export type InsertGeneticRiskFlag = z.infer<typeof insertGeneticRiskFlagSchema>;
export type GeneticRiskFlag = typeof geneticRiskFlags.$inferSelect;

// Risk Exposures ETL Job Tracking - For background job management
export const riskExposuresEtlJobs = pgTable("risk_exposures_etl_jobs", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  
  // Job type
  jobType: varchar("job_type").notNull(), // 'infectious_events', 'immunizations', 'occupational_exposures', 'genetic_flags'
  
  // Status
  status: varchar("status").notNull().default("pending"), // 'pending', 'running', 'completed', 'failed'
  
  // Progress
  recordsProcessed: integer("records_processed").default(0),
  recordsCreated: integer("records_created").default(0),
  recordsUpdated: integer("records_updated").default(0),
  recordsSkipped: integer("records_skipped").default(0),
  
  // Timing
  startedAt: timestamp("started_at"),
  completedAt: timestamp("completed_at"),
  executionTimeMs: integer("execution_time_ms"),
  
  // Error handling
  errorMessage: text("error_message"),
  errorStack: text("error_stack"),
  
  // Last processed marker (for incremental processing)
  lastProcessedId: varchar("last_processed_id"),
  lastProcessedAt: timestamp("last_processed_at"),
  
  createdAt: timestamp("created_at").defaultNow(),
}, (table) => ({
  jobTypeIdx: index("etl_job_type_idx").on(table.jobType),
  statusIdx: index("etl_job_status_idx").on(table.status),
  createdIdx: index("etl_job_created_idx").on(table.createdAt),
}));

export const insertRiskExposuresEtlJobSchema = createInsertSchema(riskExposuresEtlJobs).omit({
  id: true,
  createdAt: true,
});

export type InsertRiskExposuresEtlJob = z.infer<typeof insertRiskExposuresEtlJobSchema>;
export type RiskExposuresEtlJob = typeof riskExposuresEtlJobs.$inferSelect;
