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
  email: varchar("email").unique(),
  firstName: varchar("first_name"),
  lastName: varchar("last_name"),
  profileImageUrl: varchar("profile_image_url"),
  role: varchar("role", { length: 10 }).notNull().default("patient"), // 'patient' or 'doctor'
  medicalLicenseNumber: varchar("medical_license_number"),
  licenseVerified: boolean("license_verified").default(false),
  termsAccepted: boolean("terms_accepted").default(false),
  termsAcceptedAt: timestamp("terms_accepted_at"),
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

// Chat messages
export const chatMessages = pgTable("chat_messages", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
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
