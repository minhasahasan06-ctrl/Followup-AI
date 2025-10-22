import {
  users,
  patientProfiles,
  doctorProfiles,
  dailyFollowups,
  chatSessions,
  chatMessages,
  medications,
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
  referrals,
  creditTransactions,
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
  type Referral,
  type InsertReferral,
  type CreditTransaction,
  type InsertCreditTransaction,
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
}

export const storage = new DatabaseStorage();
