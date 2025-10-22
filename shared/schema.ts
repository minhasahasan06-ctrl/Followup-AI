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

// Session storage table (required for Replit Auth)
export const sessions = pgTable(
  "sessions",
  {
    sid: varchar("sid").primaryKey(),
    sess: jsonb("sess").notNull(),
    expire: timestamp("expire").notNull(),
  },
  (table) => [index("IDX_session_expire").on(table.expire)],
);

// Users table with role-based access
export const users = pgTable("users", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  email: varchar("email").unique().notNull(),
  passwordHash: varchar("password_hash").notNull(),
  firstName: varchar("first_name").notNull(),
  lastName: varchar("last_name").notNull(),
  profileImageUrl: varchar("profile_image_url"),
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
  
  // Email verification
  emailVerified: boolean("email_verified").default(false),
  verificationToken: varchar("verification_token"),
  verificationTokenExpires: timestamp("verification_token_expires"),
  
  // Password reset
  resetToken: varchar("reset_token"),
  resetTokenExpires: timestamp("reset_token_expires"),
  
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

// Chat sessions - groups related messages together as medical history
export const chatSessions = pgTable("chat_sessions", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  patientId: varchar("patient_id").notNull().references(() => users.id),
  agentType: varchar("agent_type", { length: 20 }).notNull(), // 'clona' or 'lysa'
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

// Medications
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
  createdAt: timestamp("created_at").defaultNow(),
});

export const insertMedicationSchema = createInsertSchema(medications).omit({
  id: true,
  createdAt: true,
});

export type InsertMedication = z.infer<typeof insertMedicationSchema>;
export type Medication = typeof medications.$inferSelect;

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
