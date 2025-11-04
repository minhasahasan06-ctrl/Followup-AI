import {
  users,
  patientProfiles,
  doctorProfiles,
  dailyFollowups,
  chatSessions,
  chatMessages,
  medications,
  drugs,
  drugInteractions,
  interactionAlerts,
  pharmacogenomicProfiles,
  drugGeneInteractions,
  dynamicTasks,
  autoJournals,
  calmActivities,
  behavioralInsights,
  researchConsents,
  aiResearchReports,
  educationalProgress,
  counselingSessions,
  trainingDatasets,
  healthInsightConsents,
  ehrConnections,
  wearableIntegrations,
  immuneBiomarkers,
  immuneDigitalTwins,
  environmentalRiskData,
  riskAlerts,
  referrals,
  creditTransactions,
  twoFactorAuth,
  medicalDocuments,
  medicalHistories,
  differentialDiagnoses,
  userSettings,
  type User,
  type UpsertUser,
  type PatientProfile,
  type InsertPatientProfile,
  type DoctorProfile,
  type InsertDoctorProfile,
  type DailyFollowup,
  type InsertDailyFollowup,
  type ChatSession,
  type InsertChatSession,
  type ChatMessage,
  type InsertChatMessage,
  type Medication,
  type InsertMedication,
  type Drug,
  type InsertDrug,
  type DrugInteraction,
  type InsertDrugInteraction,
  type InteractionAlert,
  type InsertInteractionAlert,
  type PharmacogenomicProfile,
  type InsertPharmacogenomicProfile,
  type DrugGeneInteraction,
  type InsertDrugGeneInteraction,
  type DynamicTask,
  type InsertDynamicTask,
  type AutoJournal,
  type InsertAutoJournal,
  type CalmActivity,
  type InsertCalmActivity,
  type BehavioralInsight,
  type InsertBehavioralInsight,
  type ResearchConsent,
  type InsertResearchConsent,
  type AIResearchReport,
  type InsertAIResearchReport,
  type EducationalProgress,
  type InsertEducationalProgress,
  type CounselingSession,
  type InsertCounselingSession,
  type TrainingDataset,
  type InsertTrainingDataset,
  type HealthInsightConsent,
  type InsertHealthInsightConsent,
  type EhrConnection,
  type InsertEhrConnection,
  type WearableIntegration,
  type InsertWearableIntegration,
  type ImmuneBiomarker,
  type InsertImmuneBiomarker,
  type ImmuneDigitalTwin,
  type InsertImmuneDigitalTwin,
  type EnvironmentalRiskData,
  type InsertEnvironmentalRiskData,
  type RiskAlert,
  type InsertRiskAlert,
  type Referral,
  type InsertReferral,
  type CreditTransaction,
  type InsertCreditTransaction,
  type TwoFactorAuth,
  type InsertTwoFactorAuth,
  type MedicalDocument,
  type InsertMedicalDocument,
  type MedicalHistory,
  type InsertMedicalHistory,
  type DifferentialDiagnosis,
  type InsertDifferentialDiagnosis,
  type UserSettings,
  type InsertUserSettings,
} from "@shared/schema";
import { db } from "./db";
import { eq, and, desc, sql } from "drizzle-orm";

export interface IStorage {
  // User operations
  getUser(id: string): Promise<User | undefined>;
  getUserByEmail(email: string): Promise<User | undefined>;
  createUser(user: Partial<User>): Promise<User>;
  upsertUser(user: UpsertUser): Promise<User>;
  updateUserRole(id: string, role: string, medicalLicenseNumber?: string, termsAccepted?: boolean): Promise<User | undefined>;
  updateVerificationToken(userId: string, token: string, expires: Date): Promise<User | undefined>;
  verifyEmail(token: string): Promise<User | undefined>;
  updateResetToken(userId: string, token: string, expires: Date): Promise<User | undefined>;
  resetPassword(token: string, newPasswordHash: string): Promise<User | undefined>;
  updatePhoneVerificationCode(userId: string, phoneNumber: string, code: string, expires: Date): Promise<User | undefined>;
  verifyPhoneNumber(userId: string): Promise<User | undefined>;
  updateSmsPreferences(userId: string, preferences: {
    smsNotificationsEnabled?: boolean;
    smsMedicationReminders?: boolean;
    smsAppointmentReminders?: boolean;
    smsDailyFollowups?: boolean;
    smsHealthAlerts?: boolean;
  }): Promise<User | undefined>;
  
  // Patient profile operations
  getPatientProfile(userId: string): Promise<PatientProfile | undefined>;
  upsertPatientProfile(profile: InsertPatientProfile): Promise<PatientProfile>;
  
  // Doctor profile operations
  getDoctorProfile(userId: string): Promise<DoctorProfile | undefined>;
  upsertDoctorProfile(profile: InsertDoctorProfile): Promise<DoctorProfile>;
  getAllPatients(): Promise<Array<User & { profile?: PatientProfile }>>;
  
  // Daily followup operations
  getDailyFollowup(patientId: string, date: Date): Promise<DailyFollowup | undefined>;
  getRecentFollowups(patientId: string, limit?: number): Promise<DailyFollowup[]>;
  createDailyFollowup(followup: InsertDailyFollowup): Promise<DailyFollowup>;
  updateDailyFollowup(id: string, data: Partial<DailyFollowup>): Promise<DailyFollowup | undefined>;
  
  // Chat session operations
  getActiveSession(patientId: string, agentType: string): Promise<ChatSession | undefined>;
  getPatientSessions(patientId: string, agentType?: string, limit?: number): Promise<ChatSession[]>;
  getSessionsInDateRange(patientId: string, startDate: Date, endDate: Date, agentType?: string): Promise<ChatSession[]>;
  createSession(session: InsertChatSession): Promise<ChatSession>;
  endSession(sessionId: string, aiSummary?: string, healthInsights?: any): Promise<ChatSession | undefined>;
  updateSessionMetadata(sessionId: string, data: Partial<ChatSession>): Promise<ChatSession | undefined>;
  getSessionMessages(sessionId: string): Promise<ChatMessage[]>;
  
  // Chat operations
  getChatMessages(userId: string, agentType: string, limit?: number): Promise<ChatMessage[]>;
  createChatMessage(message: InsertChatMessage): Promise<ChatMessage>;
  
  // Medication operations
  getActiveMedications(patientId: string): Promise<Medication[]>;
  createMedication(medication: InsertMedication): Promise<Medication>;
  updateMedication(id: string, data: Partial<Medication>): Promise<Medication | undefined>;
  
  // Dynamic task operations
  getActiveTasks(patientId: string): Promise<DynamicTask[]>;
  createDynamicTask(task: InsertDynamicTask): Promise<DynamicTask>;
  completeTask(id: string): Promise<DynamicTask | undefined>;
  
  // Auto journal operations
  getRecentJournals(patientId: string, limit?: number): Promise<AutoJournal[]>;
  createAutoJournal(journal: InsertAutoJournal): Promise<AutoJournal>;
  
  // Calm activity operations
  getCalmActivities(patientId: string): Promise<CalmActivity[]>;
  createCalmActivity(activity: InsertCalmActivity): Promise<CalmActivity>;
  updateCalmActivityEffectiveness(id: string, effectiveness: number): Promise<CalmActivity | undefined>;
  
  // Behavioral insight operations
  getRecentInsights(patientId: string, limit?: number): Promise<BehavioralInsight[]>;
  createBehavioralInsight(insight: InsertBehavioralInsight): Promise<BehavioralInsight>;
  
  // Research consent operations
  getResearchConsent(patientId: string): Promise<ResearchConsent | undefined>;
  getPendingConsents(): Promise<ResearchConsent[]>;
  createResearchConsent(consent: InsertResearchConsent): Promise<ResearchConsent>;
  updateConsentStatus(id: string, status: string, verifiedBy: string): Promise<ResearchConsent | undefined>;
  
  // AI research report operations
  getResearchReports(doctorId: string): Promise<AIResearchReport[]>;
  createResearchReport(report: InsertAIResearchReport): Promise<AIResearchReport>;
  
  // Educational progress operations
  getEducationalProgress(patientId: string): Promise<EducationalProgress[]>;
  upsertEducationalProgress(progress: InsertEducationalProgress): Promise<EducationalProgress>;
  
  // Counseling session operations
  getCounselingSessions(userId: string): Promise<CounselingSession[]>;
  createCounselingSession(session: InsertCounselingSession): Promise<CounselingSession>;
  updateCounselingSession(id: string, data: Partial<CounselingSession>): Promise<CounselingSession | undefined>;
  deleteCounselingSession(id: string): Promise<boolean>;
  
  // Training dataset operations
  getTrainingDatasets(userId: string): Promise<TrainingDataset[]>;
  createTrainingDataset(dataset: InsertTrainingDataset): Promise<TrainingDataset>;
  updateTrainingDataset(id: string, data: Partial<TrainingDataset>): Promise<TrainingDataset | undefined>;
  deleteTrainingDataset(id: string): Promise<boolean>;
  
  // Health insight consent operations
  getHealthInsightConsents(userId: string): Promise<HealthInsightConsent[]>;
  getActiveConsents(userId: string): Promise<HealthInsightConsent[]>;
  createHealthInsightConsent(consent: InsertHealthInsightConsent): Promise<HealthInsightConsent>;
  updateHealthInsightConsent(id: string, data: Partial<HealthInsightConsent>): Promise<HealthInsightConsent | undefined>;
  revokeConsent(id: string, reason?: string): Promise<HealthInsightConsent | undefined>;
  deleteHealthInsightConsent(id: string): Promise<boolean>;
  
  // EHR connection operations
  getEhrConnections(userId: string): Promise<EhrConnection[]>;
  createEhrConnection(connection: InsertEhrConnection): Promise<EhrConnection>;
  updateEhrConnection(id: string, data: Partial<EhrConnection>): Promise<EhrConnection | undefined>;
  deleteEhrConnection(id: string): Promise<boolean>;
  
  // Wearable integration operations
  getWearableIntegrations(userId: string): Promise<WearableIntegration[]>;
  createWearableIntegration(integration: InsertWearableIntegration): Promise<WearableIntegration>;
  updateWearableIntegration(id: string, data: Partial<WearableIntegration>): Promise<WearableIntegration | undefined>;
  deleteWearableIntegration(id: string): Promise<boolean>;
  
  // Referral operations
  getReferralByReferrerId(referrerId: string): Promise<Referral | undefined>;
  getReferralsByReferrerId(referrerId: string): Promise<Referral[]>;
  getReferralByCode(referralCode: string): Promise<Referral | undefined>;
  createReferral(referral: InsertReferral): Promise<Referral>;
  updateReferral(id: string, data: Partial<Referral>): Promise<Referral | undefined>;
  
  // Credit/wallet operations
  getUserById(userId: string): Promise<User | undefined>;
  updateUser(userId: string, data: Partial<User>): Promise<User | undefined>;
  getCreditTransactions(userId: string): Promise<CreditTransaction[]>;
  createCreditTransaction(transaction: InsertCreditTransaction): Promise<CreditTransaction>;
  
  // Admin verification operations
  getPendingDoctorVerifications(): Promise<Array<User & { doctorProfile?: DoctorProfile }>>;
  verifyDoctorLicense(userId: string, verified: boolean, notes: string, verifiedBy: string): Promise<DoctorProfile | undefined>;
  
  // Two-Factor Authentication operations
  get2FASettings(userId: string): Promise<TwoFactorAuth | undefined>;
  create2FASettings(settings: InsertTwoFactorAuth): Promise<TwoFactorAuth>;
  update2FASettings(userId: string, data: Partial<TwoFactorAuth>): Promise<TwoFactorAuth | undefined>;
  delete2FASettings(userId: string): Promise<boolean>;
  
  // Medical document operations
  getMedicalDocuments(userId: string): Promise<MedicalDocument[]>;
  getMedicalDocument(id: string): Promise<MedicalDocument | undefined>;
  createMedicalDocument(document: InsertMedicalDocument): Promise<MedicalDocument>;
  updateMedicalDocument(id: string, data: Partial<MedicalDocument>): Promise<MedicalDocument | undefined>;
  deleteMedicalDocument(id: string): Promise<boolean>;
  
  // Medical history operations
  getMedicalHistories(userId: string): Promise<MedicalHistory[]>;
  getMedicalHistoryBySession(sessionId: string): Promise<MedicalHistory | undefined>;
  createMedicalHistory(history: InsertMedicalHistory): Promise<MedicalHistory>;
  updateMedicalHistory(id: string, data: Partial<MedicalHistory>): Promise<MedicalHistory | undefined>;
  
  // Differential diagnosis operations
  getDifferentialDiagnoses(userId: string): Promise<DifferentialDiagnosis[]>;
  getDifferentialDiagnosisByHistory(medicalHistoryId: string): Promise<DifferentialDiagnosis | undefined>;
  createDifferentialDiagnosis(diagnosis: InsertDifferentialDiagnosis): Promise<DifferentialDiagnosis>;
  
  // User settings operations
  getUserSettings(userId: string): Promise<UserSettings | undefined>;
  upsertUserSettings(settings: InsertUserSettings): Promise<UserSettings>;
  
  // Immune biomarker operations
  getImmuneBiomarkers(userId: string, limit?: number): Promise<ImmuneBiomarker[]>;
  getImmuneBiomarkersInDateRange(userId: string, startDate: Date, endDate: Date): Promise<ImmuneBiomarker[]>;
  createImmuneBiomarker(biomarker: InsertImmuneBiomarker): Promise<ImmuneBiomarker>;
  
  // Immune digital twin operations
  getLatestImmuneDigitalTwin(userId: string): Promise<ImmuneDigitalTwin | undefined>;
  getImmuneDigitalTwins(userId: string, limit?: number): Promise<ImmuneDigitalTwin[]>;
  createImmuneDigitalTwin(digitalTwin: InsertImmuneDigitalTwin): Promise<ImmuneDigitalTwin>;
  
  // Environmental risk operations
  getLatestEnvironmentalRiskData(latitude: number, longitude: number): Promise<EnvironmentalRiskData | undefined>;
  getEnvironmentalRiskDataByLocation(zipCode: string, limit?: number): Promise<EnvironmentalRiskData[]>;
  createEnvironmentalRiskData(riskData: InsertEnvironmentalRiskData): Promise<EnvironmentalRiskData>;
  
  // Risk alert operations
  getActiveRiskAlerts(userId: string): Promise<RiskAlert[]>;
  getAllRiskAlerts(userId: string, limit?: number): Promise<RiskAlert[]>;
  createRiskAlert(alert: InsertRiskAlert): Promise<RiskAlert>;
  acknowledgeRiskAlert(id: string): Promise<RiskAlert | undefined>;
  resolveRiskAlert(id: string): Promise<RiskAlert | undefined>;
  dismissRiskAlert(id: string): Promise<RiskAlert | undefined>;
}

export class DatabaseStorage implements IStorage {
  // User operations
  async getUser(id: string): Promise<User | undefined> {
    const [user] = await db.select().from(users).where(eq(users.id, id));
    return user;
  }

  async upsertUser(userData: UpsertUser): Promise<User> {
    const [user] = await db
      .insert(users)
      .values(userData)
      .onConflictDoUpdate({
        target: users.id,
        set: {
          ...userData,
          updatedAt: new Date(),
        },
      })
      .returning();
    return user;
  }

  async getUserByEmail(email: string): Promise<User | undefined> {
    const [user] = await db.select().from(users).where(eq(users.email, email));
    return user;
  }

  async createUser(userData: Partial<User>): Promise<User> {
    const [user] = await db.insert(users).values(userData as any).returning();
    return user;
  }

  async updateUserRole(id: string, role: string, medicalLicenseNumber?: string, termsAccepted?: boolean): Promise<User | undefined> {
    const updateData: any = { 
      role, 
      medicalLicenseNumber, 
      updatedAt: new Date() 
    };
    
    if (termsAccepted) {
      updateData.termsAccepted = true;
      updateData.termsAcceptedAt = new Date();
    }
    
    const [user] = await db
      .update(users)
      .set(updateData)
      .where(eq(users.id, id))
      .returning();
    return user;
  }

  async updateVerificationToken(userId: string, token: string, expires: Date): Promise<User | undefined> {
    const [user] = await db
      .update(users)
      .set({
        verificationToken: token,
        verificationTokenExpires: expires,
        updatedAt: new Date(),
      })
      .where(eq(users.id, userId))
      .returning();
    return user;
  }

  async verifyEmail(token: string): Promise<User | undefined> {
    const [user] = await db
      .select()
      .from(users)
      .where(eq(users.verificationToken, token));

    if (!user || !user.verificationTokenExpires || user.verificationTokenExpires < new Date()) {
      return undefined;
    }

    const [verifiedUser] = await db
      .update(users)
      .set({
        emailVerified: true,
        verificationToken: null,
        verificationTokenExpires: null,
        updatedAt: new Date(),
      })
      .where(eq(users.id, user.id))
      .returning();
    return verifiedUser;
  }

  async updateResetToken(userId: string, token: string, expires: Date): Promise<User | undefined> {
    const [user] = await db
      .update(users)
      .set({
        resetToken: token,
        resetTokenExpires: expires,
        updatedAt: new Date(),
      })
      .where(eq(users.id, userId))
      .returning();
    return user;
  }

  async resetPassword(token: string, newPasswordHash: string): Promise<User | undefined> {
    const [user] = await db
      .select()
      .from(users)
      .where(eq(users.resetToken, token));

    if (!user || !user.resetTokenExpires || user.resetTokenExpires < new Date()) {
      return undefined;
    }

    const [updatedUser] = await db
      .update(users)
      .set({
        passwordHash: newPasswordHash,
        resetToken: null,
        resetTokenExpires: null,
        updatedAt: new Date(),
      })
      .where(eq(users.id, user.id))
      .returning();
    return updatedUser;
  }

  async updatePhoneVerificationCode(userId: string, phoneNumber: string, code: string, expires: Date): Promise<User | undefined> {
    const [user] = await db
      .update(users)
      .set({
        phoneNumber,
        phoneVerificationCode: code,
        phoneVerificationExpires: expires,
        phoneVerified: false,
        updatedAt: new Date(),
      })
      .where(eq(users.id, userId))
      .returning();
    return user;
  }

  async verifyPhoneNumber(userId: string): Promise<User | undefined> {
    const [user] = await db
      .update(users)
      .set({
        phoneVerified: true,
        phoneVerificationCode: null,
        phoneVerificationExpires: null,
        updatedAt: new Date(),
      })
      .where(eq(users.id, userId))
      .returning();
    return user;
  }

  async updateSmsPreferences(userId: string, preferences: {
    smsNotificationsEnabled?: boolean;
    smsMedicationReminders?: boolean;
    smsAppointmentReminders?: boolean;
    smsDailyFollowups?: boolean;
    smsHealthAlerts?: boolean;
  }): Promise<User | undefined> {
    const [user] = await db
      .update(users)
      .set({
        ...preferences,
        updatedAt: new Date(),
      })
      .where(eq(users.id, userId))
      .returning();
    return user;
  }

  // Patient profile operations
  async getPatientProfile(userId: string): Promise<PatientProfile | undefined> {
    const [profile] = await db
      .select()
      .from(patientProfiles)
      .where(eq(patientProfiles.userId, userId));
    return profile;
  }

  async upsertPatientProfile(profileData: InsertPatientProfile): Promise<PatientProfile> {
    const [profile] = await db
      .insert(patientProfiles)
      .values(profileData)
      .onConflictDoUpdate({
        target: patientProfiles.userId,
        set: {
          ...profileData,
          updatedAt: new Date(),
        },
      })
      .returning();
    return profile;
  }

  // Doctor profile operations
  async getDoctorProfile(userId: string): Promise<DoctorProfile | undefined> {
    const [profile] = await db
      .select()
      .from(doctorProfiles)
      .where(eq(doctorProfiles.userId, userId));
    return profile;
  }

  async upsertDoctorProfile(profileData: InsertDoctorProfile): Promise<DoctorProfile> {
    const [profile] = await db
      .insert(doctorProfiles)
      .values(profileData)
      .onConflictDoUpdate({
        target: doctorProfiles.userId,
        set: {
          ...profileData,
          updatedAt: new Date(),
        },
      })
      .returning();
    return profile;
  }

  async getAllPatients(): Promise<Array<User & { profile?: PatientProfile }>> {
    const patientsData = await db
      .select()
      .from(users)
      .leftJoin(patientProfiles, eq(users.id, patientProfiles.userId))
      .where(eq(users.role, 'patient'));
    
    return patientsData.map(row => ({
      ...row.users,
      profile: row.patient_profiles || undefined,
    }));
  }

  // Daily followup operations
  async getDailyFollowup(patientId: string, date: Date): Promise<DailyFollowup | undefined> {
    const startOfDay = new Date(date);
    startOfDay.setHours(0, 0, 0, 0);
    const endOfDay = new Date(date);
    endOfDay.setHours(23, 59, 59, 999);

    const [followup] = await db
      .select()
      .from(dailyFollowups)
      .where(
        and(
          eq(dailyFollowups.patientId, patientId),
          sql`${dailyFollowups.date} >= ${startOfDay}`,
          sql`${dailyFollowups.date} <= ${endOfDay}`
        )
      );
    return followup;
  }

  async getRecentFollowups(patientId: string, limit: number = 30): Promise<DailyFollowup[]> {
    const followups = await db
      .select()
      .from(dailyFollowups)
      .where(eq(dailyFollowups.patientId, patientId))
      .orderBy(desc(dailyFollowups.date))
      .limit(limit);
    return followups;
  }

  async createDailyFollowup(followupData: InsertDailyFollowup): Promise<DailyFollowup> {
    const [followup] = await db
      .insert(dailyFollowups)
      .values(followupData)
      .returning();
    return followup;
  }

  async updateDailyFollowup(id: string, data: Partial<DailyFollowup>): Promise<DailyFollowup | undefined> {
    const [followup] = await db
      .update(dailyFollowups)
      .set(data)
      .where(eq(dailyFollowups.id, id))
      .returning();
    return followup;
  }

  // Chat session operations
  async getActiveSession(patientId: string, agentType: string): Promise<ChatSession | undefined> {
    const [session] = await db
      .select()
      .from(chatSessions)
      .where(
        and(
          eq(chatSessions.patientId, patientId),
          eq(chatSessions.agentType, agentType),
          sql`${chatSessions.endedAt} IS NULL`
        )
      )
      .orderBy(desc(chatSessions.startedAt))
      .limit(1);
    return session;
  }

  async getPatientSessions(patientId: string, agentType?: string, limit: number = 50): Promise<ChatSession[]> {
    const conditions = [eq(chatSessions.patientId, patientId)];
    if (agentType) {
      conditions.push(eq(chatSessions.agentType, agentType));
    }
    
    const sessions = await db
      .select()
      .from(chatSessions)
      .where(and(...conditions))
      .orderBy(desc(chatSessions.startedAt))
      .limit(limit);
    return sessions;
  }

  async getSessionsInDateRange(
    patientId: string,
    startDate: Date,
    endDate: Date,
    agentType?: string
  ): Promise<ChatSession[]> {
    const conditions = [
      eq(chatSessions.patientId, patientId),
      sql`${chatSessions.startedAt} >= ${startDate}`,
      sql`${chatSessions.startedAt} <= ${endDate}`
    ];
    
    if (agentType) {
      conditions.push(eq(chatSessions.agentType, agentType));
    }
    
    const sessions = await db
      .select()
      .from(chatSessions)
      .where(and(...conditions))
      .orderBy(desc(chatSessions.startedAt));
    return sessions;
  }

  async createSession(sessionData: InsertChatSession): Promise<ChatSession> {
    const [session] = await db
      .insert(chatSessions)
      .values(sessionData)
      .returning();
    return session;
  }

  async endSession(sessionId: string, aiSummary?: string, healthInsights?: any): Promise<ChatSession | undefined> {
    const updateData: any = { 
      endedAt: new Date(),
      updatedAt: new Date()
    };
    
    if (aiSummary) {
      updateData.aiSummary = aiSummary;
    }
    if (healthInsights) {
      updateData.healthInsights = healthInsights;
    }
    
    const [session] = await db
      .update(chatSessions)
      .set(updateData)
      .where(eq(chatSessions.id, sessionId))
      .returning();
    return session;
  }

  async updateSessionMetadata(sessionId: string, data: Partial<ChatSession>): Promise<ChatSession | undefined> {
    const [session] = await db
      .update(chatSessions)
      .set({ ...data, updatedAt: new Date() })
      .where(eq(chatSessions.id, sessionId))
      .returning();
    return session;
  }

  async getSessionMessages(sessionId: string): Promise<ChatMessage[]> {
    const messages = await db
      .select()
      .from(chatMessages)
      .where(eq(chatMessages.sessionId, sessionId))
      .orderBy(chatMessages.createdAt);
    return messages;
  }

  // Chat operations
  async getChatMessages(userId: string, agentType: string, limit: number = 50): Promise<ChatMessage[]> {
    const messages = await db
      .select()
      .from(chatMessages)
      .where(and(eq(chatMessages.userId, userId), eq(chatMessages.agentType, agentType)))
      .orderBy(desc(chatMessages.createdAt))
      .limit(limit);
    return messages.reverse();
  }

  async createChatMessage(messageData: InsertChatMessage): Promise<ChatMessage> {
    const [message] = await db
      .insert(chatMessages)
      .values(messageData)
      .returning();
    return message;
  }

  // Medication operations
  async getActiveMedications(patientId: string): Promise<Medication[]> {
    const meds = await db
      .select()
      .from(medications)
      .where(and(eq(medications.patientId, patientId), eq(medications.active, true)))
      .orderBy(medications.name);
    return meds;
  }

  async createMedication(medicationData: InsertMedication): Promise<Medication> {
    const [medication] = await db
      .insert(medications)
      .values(medicationData)
      .returning();
    return medication;
  }

  async getMedicationByName(patientId: string, name: string): Promise<Medication | undefined> {
    const [medication] = await db
      .select()
      .from(medications)
      .where(sql`${medications.patientId} = ${patientId} AND LOWER(${medications.name}) = ${name.toLowerCase()} AND ${medications.active} = true`);
    return medication;
  }

  async updateMedication(id: string, data: Partial<Medication>): Promise<Medication | undefined> {
    const [medication] = await db
      .update(medications)
      .set(data)
      .where(eq(medications.id, id))
      .returning();
    return medication;
  }

  // Dynamic task operations
  async getActiveTasks(patientId: string): Promise<DynamicTask[]> {
    const tasks = await db
      .select()
      .from(dynamicTasks)
      .where(and(eq(dynamicTasks.patientId, patientId), eq(dynamicTasks.completed, false)))
      .orderBy(dynamicTasks.dueDate);
    return tasks;
  }

  async createDynamicTask(taskData: InsertDynamicTask): Promise<DynamicTask> {
    const [task] = await db
      .insert(dynamicTasks)
      .values(taskData)
      .returning();
    return task;
  }

  async completeTask(id: string): Promise<DynamicTask | undefined> {
    const [task] = await db
      .update(dynamicTasks)
      .set({ completed: true, completedAt: new Date() })
      .where(eq(dynamicTasks.id, id))
      .returning();
    return task;
  }

  // Auto journal operations
  async getRecentJournals(patientId: string, limit: number = 7): Promise<AutoJournal[]> {
    const journals = await db
      .select()
      .from(autoJournals)
      .where(eq(autoJournals.patientId, patientId))
      .orderBy(desc(autoJournals.date))
      .limit(limit);
    return journals;
  }

  async createAutoJournal(journalData: InsertAutoJournal): Promise<AutoJournal> {
    const [journal] = await db
      .insert(autoJournals)
      .values(journalData)
      .returning();
    return journal;
  }

  // Calm activity operations
  async getCalmActivities(patientId: string): Promise<CalmActivity[]> {
    const activities = await db
      .select()
      .from(calmActivities)
      .where(eq(calmActivities.patientId, patientId))
      .orderBy(desc(calmActivities.effectiveness));
    return activities;
  }

  async createCalmActivity(activityData: InsertCalmActivity): Promise<CalmActivity> {
    const [activity] = await db
      .insert(calmActivities)
      .values(activityData)
      .returning();
    return activity;
  }

  async updateCalmActivityEffectiveness(id: string, effectiveness: number): Promise<CalmActivity | undefined> {
    const [activity] = await db
      .update(calmActivities)
      .set({ 
        effectiveness, 
        timesUsed: sql`${calmActivities.timesUsed} + 1`,
        lastUsed: new Date() 
      })
      .where(eq(calmActivities.id, id))
      .returning();
    return activity;
  }

  // Behavioral insight operations
  async getRecentInsights(patientId: string, limit: number = 30): Promise<BehavioralInsight[]> {
    const insights = await db
      .select()
      .from(behavioralInsights)
      .where(eq(behavioralInsights.patientId, patientId))
      .orderBy(desc(behavioralInsights.date))
      .limit(limit);
    return insights;
  }

  async createBehavioralInsight(insightData: InsertBehavioralInsight): Promise<BehavioralInsight> {
    const [insight] = await db
      .insert(behavioralInsights)
      .values(insightData)
      .returning();
    return insight;
  }

  // Research consent operations
  async getResearchConsent(patientId: string): Promise<ResearchConsent | undefined> {
    const [consent] = await db
      .select()
      .from(researchConsents)
      .where(eq(researchConsents.patientId, patientId))
      .orderBy(desc(researchConsents.createdAt))
      .limit(1);
    return consent;
  }

  async getPendingConsents(): Promise<ResearchConsent[]> {
    const consents = await db
      .select()
      .from(researchConsents)
      .where(eq(researchConsents.status, 'pending'))
      .orderBy(desc(researchConsents.createdAt));
    return consents;
  }

  async createResearchConsent(consentData: InsertResearchConsent): Promise<ResearchConsent> {
    const [consent] = await db
      .insert(researchConsents)
      .values(consentData)
      .returning();
    return consent;
  }

  async updateConsentStatus(id: string, status: string, verifiedBy: string): Promise<ResearchConsent | undefined> {
    const [consent] = await db
      .update(researchConsents)
      .set({ 
        status, 
        verifiedBy, 
        verificationDate: new Date(),
        dataAnonymized: status === 'approved' 
      })
      .where(eq(researchConsents.id, id))
      .returning();
    return consent;
  }

  // AI research report operations
  async getResearchReports(doctorId: string): Promise<AIResearchReport[]> {
    const reports = await db
      .select()
      .from(aiResearchReports)
      .where(eq(aiResearchReports.createdBy, doctorId))
      .orderBy(desc(aiResearchReports.createdAt));
    return reports;
  }

  async createResearchReport(reportData: InsertAIResearchReport): Promise<AIResearchReport> {
    const [report] = await db
      .insert(aiResearchReports)
      .values(reportData)
      .returning();
    return report;
  }

  // Educational progress operations
  async getEducationalProgress(patientId: string): Promise<EducationalProgress[]> {
    const progress = await db
      .select()
      .from(educationalProgress)
      .where(eq(educationalProgress.patientId, patientId))
      .orderBy(desc(educationalProgress.updatedAt));
    return progress;
  }

  async upsertEducationalProgress(progressData: InsertEducationalProgress): Promise<EducationalProgress> {
    const [progress] = await db
      .insert(educationalProgress)
      .values(progressData)
      .onConflictDoUpdate({
        target: [educationalProgress.patientId, educationalProgress.moduleId],
        set: {
          ...progressData,
          updatedAt: new Date(),
        },
      })
      .returning();
    return progress;
  }

  // Counseling session operations
  async getCounselingSessions(userId: string): Promise<CounselingSession[]> {
    const sessions = await db
      .select()
      .from(counselingSessions)
      .where(eq(counselingSessions.userId, userId))
      .orderBy(desc(counselingSessions.sessionDate));
    return sessions;
  }

  async createCounselingSession(sessionData: InsertCounselingSession): Promise<CounselingSession> {
    const [session] = await db
      .insert(counselingSessions)
      .values(sessionData)
      .returning();
    return session;
  }

  async updateCounselingSession(id: string, data: Partial<CounselingSession>): Promise<CounselingSession | undefined> {
    const [session] = await db
      .update(counselingSessions)
      .set({ ...data, updatedAt: new Date() })
      .where(eq(counselingSessions.id, id))
      .returning();
    return session;
  }

  async deleteCounselingSession(id: string): Promise<boolean> {
    const result = await db
      .delete(counselingSessions)
      .where(eq(counselingSessions.id, id));
    return result.rowCount !== null && result.rowCount > 0;
  }

  // Training dataset operations
  async getTrainingDatasets(userId: string): Promise<TrainingDataset[]> {
    const datasets = await db
      .select()
      .from(trainingDatasets)
      .where(eq(trainingDatasets.uploadedBy, userId))
      .orderBy(desc(trainingDatasets.createdAt));
    return datasets;
  }

  async createTrainingDataset(datasetData: InsertTrainingDataset): Promise<TrainingDataset> {
    const [dataset] = await db
      .insert(trainingDatasets)
      .values(datasetData)
      .returning();
    return dataset;
  }

  async updateTrainingDataset(id: string, data: Partial<TrainingDataset>): Promise<TrainingDataset | undefined> {
    const [dataset] = await db
      .update(trainingDatasets)
      .set(data)
      .where(eq(trainingDatasets.id, id))
      .returning();
    return dataset;
  }

  async deleteTrainingDataset(id: string): Promise<boolean> {
    const result = await db
      .delete(trainingDatasets)
      .where(eq(trainingDatasets.id, id));
    return result.rowCount !== null && result.rowCount > 0;
  }

  // Health insight consent operations
  async getHealthInsightConsents(userId: string): Promise<HealthInsightConsent[]> {
    const consents = await db
      .select()
      .from(healthInsightConsents)
      .where(eq(healthInsightConsents.userId, userId))
      .orderBy(desc(healthInsightConsents.createdAt));
    return consents;
  }

  async getActiveConsents(userId: string): Promise<HealthInsightConsent[]> {
    const consents = await db
      .select()
      .from(healthInsightConsents)
      .where(
        and(
          eq(healthInsightConsents.userId, userId),
          eq(healthInsightConsents.consentGranted, true),
          eq(healthInsightConsents.syncStatus, "active")
        )
      )
      .orderBy(desc(healthInsightConsents.createdAt));
    return consents;
  }

  async createHealthInsightConsent(consentData: InsertHealthInsightConsent): Promise<HealthInsightConsent> {
    const [consent] = await db
      .insert(healthInsightConsents)
      .values(consentData)
      .returning();
    return consent;
  }

  async updateHealthInsightConsent(id: string, data: Partial<HealthInsightConsent>): Promise<HealthInsightConsent | undefined> {
    const [consent] = await db
      .update(healthInsightConsents)
      .set({ ...data, updatedAt: new Date() })
      .where(eq(healthInsightConsents.id, id))
      .returning();
    return consent;
  }

  async revokeConsent(id: string, reason?: string): Promise<HealthInsightConsent | undefined> {
    const [consent] = await db
      .update(healthInsightConsents)
      .set({
        consentGranted: false,
        syncStatus: "revoked",
        revokedAt: new Date(),
        revokedReason: reason,
        updatedAt: new Date(),
      })
      .where(eq(healthInsightConsents.id, id))
      .returning();
    return consent;
  }

  async deleteHealthInsightConsent(id: string): Promise<boolean> {
    const result = await db
      .delete(healthInsightConsents)
      .where(eq(healthInsightConsents.id, id));
    return result.rowCount !== null && result.rowCount > 0;
  }

  // EHR connection operations
  async getEhrConnections(userId: string): Promise<EhrConnection[]> {
    const connections = await db
      .select()
      .from(ehrConnections)
      .where(eq(ehrConnections.userId, userId))
      .orderBy(desc(ehrConnections.createdAt));
    return connections;
  }

  async createEhrConnection(connectionData: InsertEhrConnection): Promise<EhrConnection> {
    const [connection] = await db
      .insert(ehrConnections)
      .values(connectionData)
      .returning();
    return connection;
  }

  async updateEhrConnection(id: string, data: Partial<EhrConnection>): Promise<EhrConnection | undefined> {
    const [connection] = await db
      .update(ehrConnections)
      .set({ ...data, updatedAt: new Date() })
      .where(eq(ehrConnections.id, id))
      .returning();
    return connection;
  }

  async deleteEhrConnection(id: string): Promise<boolean> {
    const result = await db
      .delete(ehrConnections)
      .where(eq(ehrConnections.id, id));
    return result.rowCount !== null && result.rowCount > 0;
  }

  // Wearable integration operations
  async getWearableIntegrations(userId: string): Promise<WearableIntegration[]> {
    const integrations = await db
      .select()
      .from(wearableIntegrations)
      .where(eq(wearableIntegrations.userId, userId))
      .orderBy(desc(wearableIntegrations.createdAt));
    return integrations;
  }

  async createWearableIntegration(integrationData: InsertWearableIntegration): Promise<WearableIntegration> {
    const [integration] = await db
      .insert(wearableIntegrations)
      .values(integrationData)
      .returning();
    return integration;
  }

  async updateWearableIntegration(id: string, data: Partial<WearableIntegration>): Promise<WearableIntegration | undefined> {
    const [integration] = await db
      .update(wearableIntegrations)
      .set({ ...data, updatedAt: new Date() })
      .where(eq(wearableIntegrations.id, id))
      .returning();
    return integration;
  }

  async deleteWearableIntegration(id: string): Promise<boolean> {
    const result = await db
      .delete(wearableIntegrations)
      .where(eq(wearableIntegrations.id, id));
    return result.rowCount !== null && result.rowCount > 0;
  }

  // Referral operations
  async getReferralByReferrerId(referrerId: string): Promise<Referral | undefined> {
    const [referral] = await db
      .select()
      .from(referrals)
      .where(eq(referrals.referrerId, referrerId))
      .orderBy(desc(referrals.createdAt))
      .limit(1);
    return referral;
  }

  async getReferralsByReferrerId(referrerId: string): Promise<Referral[]> {
    const referralList = await db
      .select()
      .from(referrals)
      .where(eq(referrals.referrerId, referrerId))
      .orderBy(desc(referrals.createdAt));
    return referralList;
  }

  async getReferralByCode(referralCode: string): Promise<Referral | undefined> {
    const [referral] = await db
      .select()
      .from(referrals)
      .where(eq(referrals.referralCode, referralCode));
    return referral;
  }

  async createReferral(referralData: InsertReferral): Promise<Referral> {
    const [referral] = await db
      .insert(referrals)
      .values(referralData)
      .returning();
    return referral;
  }

  async updateReferral(id: string, data: Partial<Referral>): Promise<Referral | undefined> {
    const [referral] = await db
      .update(referrals)
      .set({ ...data, updatedAt: new Date() })
      .where(eq(referrals.id, id))
      .returning();
    return referral;
  }

  // Credit/wallet operations
  async getUserById(userId: string): Promise<User | undefined> {
    const [user] = await db
      .select()
      .from(users)
      .where(eq(users.id, userId));
    return user;
  }

  async updateUser(userId: string, data: Partial<User>): Promise<User | undefined> {
    const [user] = await db
      .update(users)
      .set({ ...data, updatedAt: new Date() })
      .where(eq(users.id, userId))
      .returning();
    return user;
  }

  async getCreditTransactions(userId: string): Promise<CreditTransaction[]> {
    const transactions = await db
      .select()
      .from(creditTransactions)
      .where(eq(creditTransactions.userId, userId))
      .orderBy(desc(creditTransactions.createdAt));
    return transactions;
  }

  async createCreditTransaction(transactionData: InsertCreditTransaction): Promise<CreditTransaction> {
    const [transaction] = await db
      .insert(creditTransactions)
      .values(transactionData)
      .returning();
    return transaction;
  }

  // Admin verification operations
  async getPendingDoctorVerifications(): Promise<Array<User & { doctorProfile?: DoctorProfile }>> {
    const doctors = await db
      .select()
      .from(users)
      .leftJoin(doctorProfiles, eq(users.id, doctorProfiles.userId))
      .where(
        and(
          eq(users.role, "doctor"),
          eq(doctorProfiles.licenseVerified, false)
        )
      );

    return doctors.map((row) => ({
      ...row.users,
      doctorProfile: row.doctor_profiles || undefined,
    }));
  }

  async verifyDoctorLicense(userId: string, verified: boolean, notes: string, verifiedBy: string): Promise<DoctorProfile | undefined> {
    const [updated] = await db
      .update(doctorProfiles)
      .set({
        licenseVerified: verified,
        verificationNotes: notes,
        verifiedBy,
        verifiedAt: new Date(),
      })
      .where(eq(doctorProfiles.userId, userId))
      .returning();

    return updated;
  }

  // Two-Factor Authentication operations
  async get2FASettings(userId: string): Promise<TwoFactorAuth | undefined> {
    const [settings] = await db
      .select()
      .from(twoFactorAuth)
      .where(eq(twoFactorAuth.userId, userId));
    return settings;
  }

  async create2FASettings(settingsData: InsertTwoFactorAuth): Promise<TwoFactorAuth> {
    const [settings] = await db
      .insert(twoFactorAuth)
      .values(settingsData)
      .returning();
    return settings;
  }

  async update2FASettings(userId: string, data: Partial<TwoFactorAuth>): Promise<TwoFactorAuth | undefined> {
    const [settings] = await db
      .update(twoFactorAuth)
      .set({ ...data, updatedAt: new Date() })
      .where(eq(twoFactorAuth.userId, userId))
      .returning();
    return settings;
  }

  async delete2FASettings(userId: string): Promise<boolean> {
    const result = await db
      .delete(twoFactorAuth)
      .where(eq(twoFactorAuth.userId, userId));
    return !!result;
  }

  // Medical document operations
  async getMedicalDocuments(userId: string): Promise<MedicalDocument[]> {
    const documents = await db
      .select()
      .from(medicalDocuments)
      .where(eq(medicalDocuments.userId, userId))
      .orderBy(desc(medicalDocuments.createdAt));
    return documents;
  }

  async getMedicalDocument(id: string): Promise<MedicalDocument | undefined> {
    const [document] = await db
      .select()
      .from(medicalDocuments)
      .where(eq(medicalDocuments.id, id));
    return document;
  }

  async createMedicalDocument(documentData: InsertMedicalDocument): Promise<MedicalDocument> {
    const [document] = await db
      .insert(medicalDocuments)
      .values(documentData)
      .returning();
    return document;
  }

  async updateMedicalDocument(id: string, data: Partial<MedicalDocument>): Promise<MedicalDocument | undefined> {
    const [document] = await db
      .update(medicalDocuments)
      .set({ ...data, updatedAt: new Date() })
      .where(eq(medicalDocuments.id, id))
      .returning();
    return document;
  }

  async deleteMedicalDocument(id: string): Promise<boolean> {
    const result = await db
      .delete(medicalDocuments)
      .where(eq(medicalDocuments.id, id));
    return !!result;
  }

  // Medical history operations
  async getMedicalHistories(userId: string): Promise<MedicalHistory[]> {
    const histories = await db
      .select()
      .from(medicalHistories)
      .where(eq(medicalHistories.userId, userId))
      .orderBy(desc(medicalHistories.createdAt));
    return histories;
  }

  async getMedicalHistoryBySession(sessionId: string): Promise<MedicalHistory | undefined> {
    const [history] = await db
      .select()
      .from(medicalHistories)
      .where(eq(medicalHistories.sessionId, sessionId));
    return history;
  }

  async createMedicalHistory(historyData: InsertMedicalHistory): Promise<MedicalHistory> {
    const [history] = await db
      .insert(medicalHistories)
      .values(historyData)
      .returning();
    return history;
  }

  async updateMedicalHistory(id: string, data: Partial<MedicalHistory>): Promise<MedicalHistory | undefined> {
    const [history] = await db
      .update(medicalHistories)
      .set({ ...data, updatedAt: new Date() })
      .where(eq(medicalHistories.id, id))
      .returning();
    return history;
  }

  // Differential diagnosis operations
  async getDifferentialDiagnoses(userId: string): Promise<DifferentialDiagnosis[]> {
    const diagnoses = await db
      .select()
      .from(differentialDiagnoses)
      .where(eq(differentialDiagnoses.userId, userId))
      .orderBy(desc(differentialDiagnoses.createdAt));
    return diagnoses;
  }

  async getDifferentialDiagnosisByHistory(medicalHistoryId: string): Promise<DifferentialDiagnosis | undefined> {
    const [diagnosis] = await db
      .select()
      .from(differentialDiagnoses)
      .where(eq(differentialDiagnoses.medicalHistoryId, medicalHistoryId));
    return diagnosis;
  }

  async createDifferentialDiagnosis(diagnosisData: InsertDifferentialDiagnosis): Promise<DifferentialDiagnosis> {
    const [diagnosis] = await db
      .insert(differentialDiagnoses)
      .values(diagnosisData)
      .returning();
    return diagnosis;
  }

  // User settings operations
  async getUserSettings(userId: string): Promise<UserSettings | undefined> {
    const [settings] = await db
      .select()
      .from(userSettings)
      .where(eq(userSettings.userId, userId));
    return settings;
  }

  async upsertUserSettings(settingsData: InsertUserSettings): Promise<UserSettings> {
    const [settings] = await db
      .insert(userSettings)
      .values(settingsData)
      .onConflictDoUpdate({
        target: userSettings.userId,
        set: {
          ...settingsData,
          updatedAt: new Date(),
        },
      })
      .returning();
    return settings;
  }

  // Drug interaction operations
  async createDrug(drugData: InsertDrug): Promise<Drug> {
    const [drug] = await db
      .insert(drugs)
      .values(drugData)
      .returning();
    return drug;
  }

  async getDrug(id: string): Promise<Drug | undefined> {
    const [drug] = await db
      .select()
      .from(drugs)
      .where(eq(drugs.id, id));
    return drug;
  }

  async getDrugByName(name: string): Promise<Drug | undefined> {
    const [drug] = await db
      .select()
      .from(drugs)
      .where(eq(drugs.name, name));
    return drug;
  }

  async searchDrugs(query: string): Promise<Drug[]> {
    const results = await db
      .select()
      .from(drugs)
      .where(sql`LOWER(${drugs.name}) LIKE ${`%${query.toLowerCase()}%`} OR LOWER(${drugs.genericName}) LIKE ${`%${query.toLowerCase()}%`}`)
      .limit(20);
    return results;
  }

  async createDrugInteraction(interactionData: InsertDrugInteraction): Promise<DrugInteraction> {
    const [interaction] = await db
      .insert(drugInteractions)
      .values(interactionData)
      .returning();
    return interaction;
  }

  async getDrugInteraction(drug1Id: string, drug2Id: string): Promise<DrugInteraction | undefined> {
    const [interaction] = await db
      .select()
      .from(drugInteractions)
      .where(
        sql`(${drugInteractions.drug1Id} = ${drug1Id} AND ${drugInteractions.drug2Id} = ${drug2Id}) OR (${drugInteractions.drug1Id} = ${drug2Id} AND ${drugInteractions.drug2Id} = ${drug1Id})`
      );
    return interaction;
  }

  async createInteractionAlert(alertData: InsertInteractionAlert): Promise<InteractionAlert> {
    const [alert] = await db
      .insert(interactionAlerts)
      .values(alertData)
      .returning();
    return alert;
  }

  async getActiveInteractionAlerts(patientId: string): Promise<InteractionAlert[]> {
    const alerts = await db
      .select()
      .from(interactionAlerts)
      .where(
        sql`${interactionAlerts.patientId} = ${patientId} AND ${interactionAlerts.alertStatus} = 'active'`
      )
      .orderBy(desc(interactionAlerts.criticalityScore), desc(interactionAlerts.createdAt));
    return alerts;
  }

  async getAllInteractionAlerts(patientId: string): Promise<InteractionAlert[]> {
    const alerts = await db
      .select()
      .from(interactionAlerts)
      .where(eq(interactionAlerts.patientId, patientId))
      .orderBy(desc(interactionAlerts.createdAt));
    return alerts;
  }

  async acknowledgeInteractionAlert(alertId: string, userId: string): Promise<InteractionAlert | undefined> {
    const [alert] = await db
      .update(interactionAlerts)
      .set({
        alertStatus: 'acknowledged',
        acknowledgedBy: userId,
        acknowledgedAt: new Date(),
        updatedAt: new Date(),
      })
      .where(eq(interactionAlerts.id, alertId))
      .returning();
    return alert;
  }

  async overrideInteractionAlert(
    alertId: string,
    doctorId: string,
    reason: string
  ): Promise<InteractionAlert | undefined> {
    const [alert] = await db
      .update(interactionAlerts)
      .set({
        alertStatus: 'overridden',
        overrideBy: doctorId,
        overrideReason: reason,
        overrideAt: new Date(),
        updatedAt: new Date(),
      })
      .where(eq(interactionAlerts.id, alertId))
      .returning();
    return alert;
  }

  async createPharmacogenomicProfile(profileData: InsertPharmacogenomicProfile): Promise<PharmacogenomicProfile> {
    const [profile] = await db
      .insert(pharmacogenomicProfiles)
      .values(profileData)
      .returning();
    return profile;
  }

  async getPharmacogenomicProfile(patientId: string): Promise<PharmacogenomicProfile | undefined> {
    const [profile] = await db
      .select()
      .from(pharmacogenomicProfiles)
      .where(eq(pharmacogenomicProfiles.patientId, patientId));
    return profile;
  }

  async updatePharmacogenomicProfile(
    patientId: string,
    updates: Partial<PharmacogenomicProfile>
  ): Promise<PharmacogenomicProfile | undefined> {
    const [profile] = await db
      .update(pharmacogenomicProfiles)
      .set({ ...updates, lastUpdated: new Date() })
      .where(eq(pharmacogenomicProfiles.patientId, patientId))
      .returning();
    return profile;
  }

  async createDrugGeneInteraction(interactionData: InsertDrugGeneInteraction): Promise<DrugGeneInteraction> {
    const [interaction] = await db
      .insert(drugGeneInteractions)
      .values(interactionData)
      .returning();
    return interaction;
  }

  async getDrugGeneInteractions(drugId: string): Promise<DrugGeneInteraction[]> {
    const interactions = await db
      .select()
      .from(drugGeneInteractions)
      .where(eq(drugGeneInteractions.drugId, drugId));
    return interactions;
  }

  // Immune biomarker operations
  async getImmuneBiomarkers(userId: string, limit: number = 30): Promise<ImmuneBiomarker[]> {
    const biomarkers = await db
      .select()
      .from(immuneBiomarkers)
      .where(eq(immuneBiomarkers.userId, userId))
      .orderBy(desc(immuneBiomarkers.measuredAt))
      .limit(limit);
    return biomarkers;
  }

  async getImmuneBiomarkersInDateRange(userId: string, startDate: Date, endDate: Date): Promise<ImmuneBiomarker[]> {
    const biomarkers = await db
      .select()
      .from(immuneBiomarkers)
      .where(
        and(
          eq(immuneBiomarkers.userId, userId),
          sql`${immuneBiomarkers.measuredAt} >= ${startDate}`,
          sql`${immuneBiomarkers.measuredAt} <= ${endDate}`
        )
      )
      .orderBy(desc(immuneBiomarkers.measuredAt));
    return biomarkers;
  }

  async createImmuneBiomarker(biomarkerData: InsertImmuneBiomarker): Promise<ImmuneBiomarker> {
    const [biomarker] = await db
      .insert(immuneBiomarkers)
      .values(biomarkerData)
      .returning();
    return biomarker;
  }

  // Immune digital twin operations
  async getLatestImmuneDigitalTwin(userId: string): Promise<ImmuneDigitalTwin | undefined> {
    const [digitalTwin] = await db
      .select()
      .from(immuneDigitalTwins)
      .where(eq(immuneDigitalTwins.userId, userId))
      .orderBy(desc(immuneDigitalTwins.predictedAt))
      .limit(1);
    return digitalTwin;
  }

  async getImmuneDigitalTwins(userId: string, limit: number = 30): Promise<ImmuneDigitalTwin[]> {
    const digitalTwins = await db
      .select()
      .from(immuneDigitalTwins)
      .where(eq(immuneDigitalTwins.userId, userId))
      .orderBy(desc(immuneDigitalTwins.predictedAt))
      .limit(limit);
    return digitalTwins;
  }

  async createImmuneDigitalTwin(digitalTwinData: InsertImmuneDigitalTwin): Promise<ImmuneDigitalTwin> {
    const [digitalTwin] = await db
      .insert(immuneDigitalTwins)
      .values(digitalTwinData)
      .returning();
    return digitalTwin;
  }

  // Environmental risk operations
  async getLatestEnvironmentalRiskData(latitude: number, longitude: number): Promise<EnvironmentalRiskData | undefined> {
    const [riskData] = await db
      .select()
      .from(environmentalRiskData)
      .where(
        and(
          sql`${environmentalRiskData.latitude} = ${latitude.toString()}`,
          sql`${environmentalRiskData.longitude} = ${longitude.toString()}`
        )
      )
      .orderBy(desc(environmentalRiskData.measuredAt))
      .limit(1);
    return riskData;
  }

  async getEnvironmentalRiskDataByLocation(zipCode: string, limit: number = 30): Promise<EnvironmentalRiskData[]> {
    const riskData = await db
      .select()
      .from(environmentalRiskData)
      .where(eq(environmentalRiskData.zipCode, zipCode))
      .orderBy(desc(environmentalRiskData.measuredAt))
      .limit(limit);
    return riskData;
  }

  async createEnvironmentalRiskData(riskData: InsertEnvironmentalRiskData): Promise<EnvironmentalRiskData> {
    const [data] = await db
      .insert(environmentalRiskData)
      .values(riskData)
      .returning();
    return data;
  }

  // Risk alert operations
  async getActiveRiskAlerts(userId: string): Promise<RiskAlert[]> {
    const alerts = await db
      .select()
      .from(riskAlerts)
      .where(
        and(
          eq(riskAlerts.userId, userId),
          eq(riskAlerts.status, 'active')
        )
      )
      .orderBy(desc(riskAlerts.priority), desc(riskAlerts.createdAt));
    return alerts;
  }

  async getAllRiskAlerts(userId: string, limit: number = 50): Promise<RiskAlert[]> {
    const alerts = await db
      .select()
      .from(riskAlerts)
      .where(eq(riskAlerts.userId, userId))
      .orderBy(desc(riskAlerts.createdAt))
      .limit(limit);
    return alerts;
  }

  async createRiskAlert(alertData: InsertRiskAlert): Promise<RiskAlert> {
    const [alert] = await db
      .insert(riskAlerts)
      .values(alertData)
      .returning();
    return alert;
  }

  async acknowledgeRiskAlert(id: string): Promise<RiskAlert | undefined> {
    const [alert] = await db
      .update(riskAlerts)
      .set({
        status: 'acknowledged',
        acknowledgedAt: new Date(),
        updatedAt: new Date(),
      })
      .where(eq(riskAlerts.id, id))
      .returning();
    return alert;
  }

  async resolveRiskAlert(id: string): Promise<RiskAlert | undefined> {
    const [alert] = await db
      .update(riskAlerts)
      .set({
        status: 'resolved',
        resolvedAt: new Date(),
        updatedAt: new Date(),
      })
      .where(eq(riskAlerts.id, id))
      .returning();
    return alert;
  }

  async dismissRiskAlert(id: string): Promise<RiskAlert | undefined> {
    const [alert] = await db
      .update(riskAlerts)
      .set({
        status: 'dismissed',
        updatedAt: new Date(),
      })
      .where(eq(riskAlerts.id, id))
      .returning();
    return alert;
  }
}

export const storage = new DatabaseStorage();
