import {
  users,
  patientProfiles,
  doctorProfiles,
  dailyFollowups,
  voiceFollowups,
  chatSessions,
  chatMessages,
  medications,
  prescriptions,
  medicationChangeLog,
  dosageChangeRequests,
  drugs,
  drugInteractions,
  interactionAlerts,
  pharmacogenomicProfiles,
  drugGeneInteractions,
  medicationSchedules,
  medicationAdherence,
  dietaryPreferences,
  mealPlans,
  meals,
  nutritionEntries,
  companionCheckIns,
  companionEngagement,
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
  correlationPatterns,
  geneticVariants,
  pharmacogenomicReports,
  clinicalTrials,
  trialMatchScores,
  deteriorationPredictions,
  userLearningProfiles,
  habits,
  habitCompletions,
  streaks,
  milestones,
  mlRecommendations,
  rlRewards,
  dailyEngagement,
  doctorWellness,
  appointments,
  consultations,
  doctorPatientAssignments,
  doctorPatientConsentPermissions,
  doctorAvailability,
  emailThreads,
  emailMessages,
  callLogs,
  appointmentReminders,
  googleCalendarSync,
  googleCalendarSyncLogs,
  gmailSync,
  gmailSyncLogs,
  doctorIntegrations,
  doctorEmails,
  doctorWhatsappMessages,
  type User,
  type UpsertUser,
  type PatientProfile,
  type InsertPatientProfile,
  type DoctorProfile,
  type InsertDoctorProfile,
  type DailyFollowup,
  type InsertDailyFollowup,
  type VoiceFollowup,
  type InsertVoiceFollowup,
  type ChatSession,
  type InsertChatSession,
  type ChatMessage,
  type InsertChatMessage,
  type Medication,
  type InsertMedication,
  type Prescription,
  type InsertPrescription,
  type MedicationChangeLog,
  type InsertMedicationChangeLog,
  type DosageChangeRequest,
  type InsertDosageChangeRequest,
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
  type MedicationSchedule,
  type InsertMedicationSchedule,
  type MedicationAdherence,
  type InsertMedicationAdherence,
  type DietaryPreference,
  type InsertDietaryPreference,
  type MealPlan,
  type InsertMealPlan,
  type Meal,
  type InsertMeal,
  type NutritionEntry,
  type InsertNutritionEntry,
  type CompanionCheckIn,
  type InsertCompanionCheckIn,
  type CompanionEngagement,
  type InsertCompanionEngagement,
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
  type CorrelationPattern,
  type InsertCorrelationPattern,
  type GeneticVariant,
  type InsertGeneticVariant,
  type PharmacogenomicReport,
  type InsertPharmacogenomicReport,
  type ClinicalTrial,
  type InsertClinicalTrial,
  type TrialMatchScore,
  type InsertTrialMatchScore,
  type DeteriorationPrediction,
  type InsertDeteriorationPrediction,
  type UserLearningProfile,
  type InsertUserLearningProfile,
  type Habit,
  type InsertHabit,
  type HabitCompletion,
  type InsertHabitCompletion,
  type Streak,
  type InsertStreak,
  type Milestone,
  type InsertMilestone,
  type MLRecommendation,
  type InsertMLRecommendation,
  type RLReward,
  type InsertRLReward,
  type DailyEngagement,
  type InsertDailyEngagement,
  type DoctorWellness,
  type InsertDoctorWellness,
  type Appointment,
  type InsertAppointment,
  type DoctorPatientAssignment,
  type InsertDoctorPatientAssignment,
  patientConsentRequests,
  type PatientConsentRequest,
  type InsertPatientConsentRequest,
  type DoctorAvailability,
  type InsertDoctorAvailability,
  type EmailThread,
  type InsertEmailThread,
  type EmailMessage,
  type InsertEmailMessage,
  type CallLog,
  type InsertCallLog,
  type DoctorIntegration,
  type InsertDoctorIntegration,
  type DoctorEmail,
  type InsertDoctorEmail,
  type DoctorWhatsappMessage,
  type InsertDoctorWhatsappMessage,
  type AppointmentReminder,
  type InsertAppointmentReminder,
  type GoogleCalendarSync,
  type InsertGoogleCalendarSync,
  type GoogleCalendarSyncLog,
  type InsertGoogleCalendarSyncLog,
  type GmailSync,
  type InsertGmailSync,
  type GmailSyncLog,
  type InsertGmailSyncLog,
  paintrackSessions,
  type PaintrackSession,
  type InsertPaintrackSession,
  deviceReadings,
  type DeviceReading,
  type InsertDeviceReading,
} from "@shared/schema";
import { db } from "./db";
import { eq, and, or, desc, sql, gte, lte, gt, like, ilike, inArray, between } from "drizzle-orm";

export interface IStorage {
  // User operations
  getUser(id: string): Promise<User | undefined>;
  getUserByEmail(email: string): Promise<User | undefined>;
  getUserByPhoneNumber(phoneNumber: string): Promise<User | undefined>;
  getAllDoctors(): Promise<User[]>;
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
  getDoctorPatients(doctorId: string): Promise<Array<User & { profile?: PatientProfile }>>;
  
  // Doctor-Patient Assignment operations (HIPAA compliance)
  getDoctorPatientAssignment(doctorId: string, patientId: string): Promise<DoctorPatientAssignment | undefined>;
  getActiveDoctorAssignments(doctorId: string): Promise<DoctorPatientAssignment[]>;
  getPatientDoctorAssignments(patientId: string): Promise<DoctorPatientAssignment[]>;
  createDoctorPatientAssignment(assignment: InsertDoctorPatientAssignment): Promise<DoctorPatientAssignment>;
  updateDoctorPatientAssignment(id: string, data: Partial<DoctorPatientAssignment>): Promise<DoctorPatientAssignment | undefined>;
  revokeDoctorPatientAssignment(id: string, revokedBy: string, reason: string): Promise<DoctorPatientAssignment | undefined>;
  doctorHasPatientAccess(doctorId: string, patientId: string): Promise<boolean>;
  
  // Patient Consent Request operations
  searchPatientsByIdentifier(query: string): Promise<Array<{ user: User; profile: PatientProfile | null }>>;
  createPatientConsentRequest(request: InsertPatientConsentRequest): Promise<PatientConsentRequest>;
  getPendingConsentRequestsForPatient(patientId: string): Promise<PatientConsentRequest[]>;
  getPendingConsentRequestsForDoctor(doctorId: string): Promise<PatientConsentRequest[]>;
  getConsentRequest(id: string): Promise<PatientConsentRequest | undefined>;
  respondToConsentRequest(id: string, approved: boolean, responseMessage?: string): Promise<PatientConsentRequest | undefined>;
  generateFollowupPatientId(): Promise<string>;
  getPatientByFollowupId(followupPatientId: string): Promise<User | undefined>;
  
  // Consent Permissions operations (granular HIPAA permissions)
  createConsentPermissions(permissions: any): Promise<any>;
  getConsentPermissions(assignmentId: string): Promise<any>;
  getConsentPermissionsByDoctorPatient(doctorId: string, patientId: string): Promise<any>;
  getPatientConsentPermissions(patientId: string): Promise<any[]>;
  updateConsentPermissions(assignmentId: string, permissions: any): Promise<any>;
  
  // Daily followup operations
  getDailyFollowup(patientId: string, date: Date): Promise<DailyFollowup | undefined>;
  getRecentFollowups(patientId: string, limit?: number): Promise<DailyFollowup[]>;
  createDailyFollowup(followup: InsertDailyFollowup): Promise<DailyFollowup>;
  updateDailyFollowup(id: string, data: Partial<DailyFollowup>): Promise<DailyFollowup | undefined>;
  
  // Voice followup operations
  createVoiceFollowup(followup: InsertVoiceFollowup): Promise<VoiceFollowup>;
  getVoiceFollowup(id: string): Promise<VoiceFollowup | undefined>;
  getRecentVoiceFollowups(patientId: string, limit?: number): Promise<VoiceFollowup[]>;
  getAllVoiceFollowups(patientId: string): Promise<VoiceFollowup[]>;
  
  // Chat session operations
  getActiveSession(patientId: string, agentType: string, contextPatientId?: string): Promise<ChatSession | undefined>;
  getPatientSessions(patientId: string, agentType?: string, limit?: number): Promise<ChatSession[]>;
  getSessionsInDateRange(patientId: string, startDate: Date, endDate: Date, agentType?: string): Promise<ChatSession[]>;
  createSession(session: InsertChatSession & { contextPatientId?: string }): Promise<ChatSession>;
  endSession(sessionId: string, aiSummary?: string, healthInsights?: any): Promise<ChatSession | undefined>;
  updateSessionMetadata(sessionId: string, data: Partial<ChatSession>): Promise<ChatSession | undefined>;
  getSessionMessages(sessionId: string): Promise<ChatMessage[]>;
  
  // Chat operations
  getChatMessages(userId: string, agentType: string, contextPatientId?: string, limit?: number): Promise<ChatMessage[]>;
  createChatMessage(message: InsertChatMessage): Promise<ChatMessage>;
  
  // Medication operations
  getActiveMedications(patientId: string): Promise<Medication[]>;
  getAllMedications(patientId: string): Promise<Medication[]>;
  getPendingConfirmationMedications(patientId: string): Promise<Medication[]>;
  getInactiveMedications(patientId: string): Promise<Medication[]>;
  createMedication(medication: InsertMedication): Promise<Medication>;
  updateMedication(id: string, data: Partial<Medication>): Promise<Medication | undefined>;
  confirmMedication(id: string, confirmedBy: string): Promise<Medication | undefined>;
  discontinueMedication(id: string, discontinuedBy: string, reason: string, replacementId?: string): Promise<Medication | undefined>;
  reactivateMedication(id: string, reactivatedBy: string): Promise<Medication | undefined>;
  
  // Prescription operations
  getPrescriptions(patientId: string): Promise<Prescription[]>;
  getPrescriptionsByDoctor(doctorId: string): Promise<Prescription[]>;
  getPrescription(id: string): Promise<Prescription | undefined>;
  createPrescription(prescription: InsertPrescription): Promise<Prescription>;
  updatePrescription(id: string, data: Partial<Prescription>): Promise<Prescription | undefined>;
  acknowledgePrescription(id: string, acknowledgedBy: string): Promise<Prescription | undefined>;
  
  // Chronic care medication lifecycle operations
  getMedicationsBySpecialty(patientId: string, specialty: string): Promise<Medication[]>;
  getMedicationsByDrugClass(patientId: string, drugClassPattern: string): Promise<Medication[]>;
  supersedeMedication(oldMedicationId: string, newMedicationId: string, reason: string): Promise<Medication | undefined>;
  
  // Cross-specialty conflict operations
  getMedicationConflicts(patientId: string): Promise<any[]>;
  getPendingConflicts(doctorId: string): Promise<any[]>;
  createMedicationConflict(conflict: any): Promise<any>;
  getMedicationConflict(id: string): Promise<any | undefined>;
  updateMedicationConflict(id: string, data: any): Promise<any | undefined>;
  resolveMedicationConflict(id: string, resolution: any): Promise<any | undefined>;
  
  // Medication change log operations
  getMedicationChangelog(medicationId: string): Promise<MedicationChangeLog[]>;
  getPatientMedicationChangelog(patientId: string, limit?: number): Promise<MedicationChangeLog[]>;
  createMedicationChangeLog(log: InsertMedicationChangeLog): Promise<MedicationChangeLog>;
  
  // Dosage change request operations
  getDosageChangeRequests(patientId: string): Promise<DosageChangeRequest[]>;
  getPendingDosageChangeRequests(doctorId: string): Promise<DosageChangeRequest[]>;
  getDosageChangeRequest(id: string): Promise<DosageChangeRequest | undefined>;
  createDosageChangeRequest(request: InsertDosageChangeRequest): Promise<DosageChangeRequest>;
  approveDosageChangeRequest(id: string, doctorId: string, notes?: string): Promise<DosageChangeRequest | undefined>;
  rejectDosageChangeRequest(id: string, doctorId: string, notes: string): Promise<DosageChangeRequest | undefined>;
  
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
  verifyDoctorApplication(userId: string, verified: boolean, notes: string, verifiedBy: string): Promise<{ user: User | undefined; doctorProfile: DoctorProfile | undefined }>;
  
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
  getEnvironmentalRiskDataByUser(userId: string, limit?: number): Promise<EnvironmentalRiskData[]>;
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
  
  // Correlation pattern operations
  getCorrelationPatterns(userId: string, limit?: number): Promise<CorrelationPattern[]>;
  getCorrelationPatternsByType(userId: string, patternType: string): Promise<CorrelationPattern[]>;
  getHighSeverityPatterns(userId: string): Promise<CorrelationPattern[]>;
  createCorrelationPattern(pattern: InsertCorrelationPattern): Promise<CorrelationPattern>;
  
  // Genetic variant operations
  getGeneticVariants(userId: string): Promise<GeneticVariant[]>;
  getHighRiskVariants(userId: string): Promise<GeneticVariant[]>;
  createGeneticVariant(variant: InsertGeneticVariant): Promise<GeneticVariant>;
  
  // Pharmacogenomic report operations
  getPharmacogenomicReports(userId: string): Promise<PharmacogenomicReport[]>;
  getPharmacogenomicReport(id: string): Promise<PharmacogenomicReport | undefined>;
  createPharmacogenomicReport(report: InsertPharmacogenomicReport): Promise<PharmacogenomicReport>;
  updatePharmacogenomicReportStatus(id: string, status: string, error?: string): Promise<PharmacogenomicReport | undefined>;
  
  // Clinical trial operations
  getClinicalTrials(conditions?: string[], status?: string, limit?: number): Promise<ClinicalTrial[]>;
  getClinicalTrial(id: string): Promise<ClinicalTrial | undefined>;
  createClinicalTrial(trial: InsertClinicalTrial): Promise<ClinicalTrial>;
  updateClinicalTrial(id: string, data: Partial<ClinicalTrial>): Promise<ClinicalTrial | undefined>;
  
  // Trial match score operations
  getTrialMatchScores(userId: string, limit?: number): Promise<TrialMatchScore[]>;
  getHighMatchTrials(userId: string, minScore?: number): Promise<TrialMatchScore[]>;
  createTrialMatchScore(matchScore: InsertTrialMatchScore): Promise<TrialMatchScore>;
  updateTrialMatchStatus(id: string, status: string, notes?: string): Promise<TrialMatchScore | undefined>;
  
  // Deterioration prediction operations
  getDeteriorationPredictions(userId: string, limit?: number): Promise<DeteriorationPrediction[]>;
  getLatestPrediction(userId: string): Promise<DeteriorationPrediction | undefined>;
  getHighRiskPredictions(userId: string): Promise<DeteriorationPrediction[]>;
  createDeteriorationPrediction(prediction: InsertDeteriorationPrediction): Promise<DeteriorationPrediction>;
  updatePredictionOutcome(id: string, outcome: string, notes?: string): Promise<DeteriorationPrediction | undefined>;
  
  // Medication schedule operations
  getMedicationSchedules(medicationId: string): Promise<MedicationSchedule[]>;
  getPatientMedicationSchedules(patientId: string, activeOnly?: boolean): Promise<MedicationSchedule[]>;
  createMedicationSchedule(schedule: InsertMedicationSchedule): Promise<MedicationSchedule>;
  updateMedicationSchedule(id: string, data: Partial<MedicationSchedule>): Promise<MedicationSchedule | undefined>;
  
  // Medication adherence operations
  getMedicationAdherence(medicationId: string, limit?: number): Promise<MedicationAdherence[]>;
  getPatientAdherence(patientId: string, limit?: number): Promise<MedicationAdherence[]>;
  getPendingMedications(patientId: string): Promise<MedicationAdherence[]>;
  createMedicationAdherence(adherence: InsertMedicationAdherence): Promise<MedicationAdherence>;
  updateMedicationAdherence(id: string, data: Partial<MedicationAdherence>): Promise<MedicationAdherence | undefined>;
  
  // Dietary preference operations
  getDietaryPreferences(patientId: string): Promise<DietaryPreference | undefined>;
  upsertDietaryPreferences(preferences: InsertDietaryPreference): Promise<DietaryPreference>;
  
  // Meal plan operations
  getMealPlans(patientId: string, activeOnly?: boolean): Promise<MealPlan[]>;
  getActiveMealPlan(patientId: string): Promise<MealPlan | undefined>;
  createMealPlan(mealPlan: InsertMealPlan): Promise<MealPlan>;
  updateMealPlan(id: string, data: Partial<MealPlan>): Promise<MealPlan | undefined>;
  
  // Meal operations
  getMeals(patientId: string, mealPlanId?: string, limit?: number): Promise<Meal[]>;
  getMealsByDateRange(patientId: string, startDate: Date, endDate: Date): Promise<Meal[]>;
  getTodaysMeals(patientId: string): Promise<Meal[]>;
  createMeal(meal: InsertMeal): Promise<Meal>;
  updateMeal(id: string, data: Partial<Meal>): Promise<Meal | undefined>;
  
  // Nutrition entry operations
  getNutritionEntries(mealId: string): Promise<NutritionEntry[]>;
  getPatientNutritionEntries(patientId: string, limit?: number): Promise<NutritionEntry[]>;
  createNutritionEntry(entry: InsertNutritionEntry): Promise<NutritionEntry>;
  
  // Companion check-in operations
  getCompanionCheckIns(patientId: string, limit?: number): Promise<CompanionCheckIn[]>;
  getCheckInsByType(patientId: string, checkInType: string, limit?: number): Promise<CompanionCheckIn[]>;
  getRecentCheckIns(patientId: string, days: number): Promise<CompanionCheckIn[]>;
  createCompanionCheckIn(checkIn: InsertCompanionCheckIn): Promise<CompanionCheckIn>;
  
  // Companion engagement operations
  getCompanionEngagement(patientId: string): Promise<CompanionEngagement | undefined>;
  upsertCompanionEngagement(engagement: InsertCompanionEngagement): Promise<CompanionEngagement>;
  updateCompanionEngagement(patientId: string, data: Partial<CompanionEngagement>): Promise<CompanionEngagement | undefined>;
  
  // ML/RL System operations
  getUserLearningProfile(userId: string, agentType: string): Promise<UserLearningProfile | undefined>;
  upsertUserLearningProfile(profile: InsertUserLearningProfile): Promise<UserLearningProfile>;
  getHabits(userId: string): Promise<Habit[]>;
  createHabit(habit: InsertHabit): Promise<Habit>;
  updateHabit(id: string, data: Partial<Habit>): Promise<Habit | undefined>;
  getRecentHabitCompletions(userId: string, days: number): Promise<HabitCompletion[]>;
  createHabitCompletion(completion: InsertHabitCompletion): Promise<HabitCompletion>;
  getUserRecommendations(userId: string): Promise<MLRecommendation[]>;
  createMLRecommendation(recommendation: InsertMLRecommendation): Promise<MLRecommendation>;
  updateMLRecommendation(id: string, data: Partial<MLRecommendation>): Promise<MLRecommendation | undefined>;
  createRLReward(reward: InsertRLReward): Promise<RLReward>;
  getDailyEngagement(userId: string, date: Date): Promise<DailyEngagement | undefined>;
  upsertDailyEngagement(engagement: InsertDailyEngagement): Promise<DailyEngagement>;
  getDoctorWellnessHistory(userId: string, days: number): Promise<DoctorWellness[]>;
  createDoctorWellness(wellness: InsertDoctorWellness): Promise<DoctorWellness>;
  getMilestones(userId: string): Promise<Milestone[]>;
  createMilestone(milestone: InsertMilestone): Promise<Milestone>;
  updateMilestone(id: string, data: Partial<Milestone>): Promise<Milestone | undefined>;
  
  // ============================================================================
  // RECEPTIONIST & ASSISTANT LYSA FEATURES
  // ============================================================================
  
  // Appointment operations
  createAppointment(appointment: InsertAppointment): Promise<Appointment>;
  getAppointment(id: string): Promise<Appointment | undefined>;
  getAppointmentByExternalId(googleCalendarEventId: string): Promise<Appointment | undefined>;
  listAppointments(filters: {
    doctorId?: string;
    patientId?: string;
    startDate?: Date;
    endDate?: Date;
    status?: string;
    limit?: number;
    offset?: number;
  }): Promise<Appointment[]>;
  listUpcomingAppointments(userId: string, role: 'doctor' | 'patient', daysAhead?: number, limit?: number): Promise<Appointment[]>;
  updateAppointment(id: string, data: Partial<Appointment>): Promise<Appointment | undefined>;
  confirmAppointment(id: string, confirmedAt: Date): Promise<Appointment | undefined>;
  cancelAppointment(id: string, cancelledBy: string, reason: string): Promise<Appointment | undefined>;
  findAppointmentConflicts(doctorId: string, startTime: Date, endTime: Date, excludeId?: string): Promise<Appointment[]>;
  
  // Doctor availability operations
  setDoctorAvailability(availability: InsertDoctorAvailability): Promise<DoctorAvailability>;
  getDoctorAvailability(doctorId: string, dateRange?: { start: Date; end: Date }): Promise<DoctorAvailability[]>;
  removeDoctorAvailability(id: string): Promise<void>;
  
  // Email thread operations
  createEmailThread(thread: InsertEmailThread): Promise<EmailThread>;
  getEmailThread(id: string): Promise<EmailThread | undefined>;
  listEmailThreads(doctorId: string, filters?: { 
    status?: string; 
    category?: string; 
    isRead?: boolean; 
    patientId?: string;
    limit?: number;
    offset?: number;
  }): Promise<EmailThread[]>;
  updateEmailThread(id: string, data: Partial<EmailThread>): Promise<EmailThread | undefined>;
  markThreadRead(threadId: string): Promise<EmailThread | undefined>;
  
  // Email message operations
  createEmailMessage(message: InsertEmailMessage): Promise<EmailMessage>;
  getThreadMessages(threadId: string): Promise<EmailMessage[]>;
  markEmailSent(id: string, sentAt: Date): Promise<EmailMessage | undefined>;
  
  // Call log operations
  createCallLog(log: InsertCallLog): Promise<CallLog>;
  getCallLog(id: string): Promise<CallLog | undefined>;
  listCallLogs(doctorId: string, filters?: { 
    status?: string; 
    requiresFollowup?: boolean; 
    patientId?: string;
    limit?: number;
    offset?: number;
  }): Promise<CallLog[]>;
  listCallLogsByPatient(patientId: string, limit?: number): Promise<CallLog[]>;
  updateCallLog(id: string, data: Partial<CallLog>): Promise<CallLog | undefined>;
  
  // Appointment reminder operations
  createAppointmentReminder(reminder: InsertAppointmentReminder): Promise<AppointmentReminder>;
  getAppointmentReminders(appointmentId: string): Promise<AppointmentReminder[]>;
  listDueReminders(beforeTime: Date, limit?: number): Promise<AppointmentReminder[]>;
  markReminderSent(id: string, sentAt: Date, twilioSid?: string, sesSid?: string): Promise<AppointmentReminder | undefined>;
  markReminderFailed(id: string, error: string): Promise<AppointmentReminder | undefined>;
  confirmReminder(id: string): Promise<AppointmentReminder | undefined>;
  
  // Appointment triage operations
  updateAppointmentTriageResult(params: {
    appointmentId?: string;
    patientId: string;
    symptoms: string;
    urgencyLevel: 'emergency' | 'urgent' | 'routine' | 'non-urgent';
    triageAssessment: TriageAssessment;
    redFlags: string[];
    recommendations: string[];
    patientSelfAssessment?: string;
    durationMs: number;
  }): Promise<{ appointment?: Appointment; log: AppointmentTriageLog }>;
  createAppointmentTriageLog(logData: InsertAppointmentTriageLog): Promise<AppointmentTriageLog>;
  getAppointmentTriageLogs(patientId: string, limit?: number): Promise<AppointmentTriageLog[]>;
  
  // Google Calendar sync operations
  createGoogleCalendarSync(sync: InsertGoogleCalendarSync): Promise<GoogleCalendarSync>;
  getGoogleCalendarSync(doctorId: string): Promise<GoogleCalendarSync | undefined>;
  updateGoogleCalendarSync(doctorId: string, data: Partial<GoogleCalendarSync>): Promise<GoogleCalendarSync | undefined>;
  createGoogleCalendarSyncLog(log: InsertGoogleCalendarSyncLog): Promise<GoogleCalendarSyncLog>;
  getGoogleCalendarSyncLogs(doctorId: string, limit?: number): Promise<GoogleCalendarSyncLog[]>;
  getAppointmentByGoogleEventId(eventId: string): Promise<Appointment | undefined>;
  
  // Gmail sync operations
  createGmailSync(sync: InsertGmailSync): Promise<GmailSync>;
  getGmailSync(doctorId: string): Promise<GmailSync | undefined>;
  updateGmailSync(doctorId: string, data: Partial<GmailSync>): Promise<GmailSync | undefined>;
  deleteGmailSync(doctorId: string): Promise<boolean>;
  createGmailSyncLog(log: InsertGmailSyncLog): Promise<GmailSyncLog>;
  getGmailSyncLogs(doctorId: string, limit?: number): Promise<GmailSyncLog[]>;
  getEmailThreadByExternalId(externalThreadId: string): Promise<EmailThread | undefined>;
  
  // PainTrack operations
  createPaintrackSession(session: InsertPaintrackSession): Promise<PaintrackSession>;
  getPaintrackSessions(userId: string, limit?: number): Promise<PaintrackSession[]>;
  getPaintrackSession(id: string, userId: string): Promise<PaintrackSession | undefined>;

  // Device Readings operations (BP, glucose, scale, thermometer, stethoscope, smartwatch)
  createDeviceReading(reading: InsertDeviceReading): Promise<DeviceReading>;
  getDeviceReadings(patientId: string, options?: { 
    deviceType?: string; 
    limit?: number; 
    startDate?: Date; 
    endDate?: Date;
  }): Promise<DeviceReading[]>;
  getDeviceReading(id: string): Promise<DeviceReading | undefined>;
  getLatestDeviceReading(patientId: string, deviceType: string): Promise<DeviceReading | undefined>;
  getDeviceReadingsByType(patientId: string, deviceType: string, limit?: number): Promise<DeviceReading[]>;
  updateDeviceReading(id: string, data: Partial<DeviceReading>): Promise<DeviceReading | undefined>;
  deleteDeviceReading(id: string): Promise<boolean>;
  getDeviceReadingsForHealthAlerts(patientId: string, hours?: number): Promise<DeviceReading[]>;
  markDeviceReadingProcessedForAlerts(id: string, alertIds: string[]): Promise<DeviceReading | undefined>;

  // Doctor Integrations operations (per-doctor OAuth/API connections)
  getDoctorIntegrations(doctorId: string): Promise<DoctorIntegration[]>;
  getDoctorIntegrationByType(doctorId: string, integrationType: string): Promise<DoctorIntegration | undefined>;
  createDoctorIntegration(integration: InsertDoctorIntegration): Promise<DoctorIntegration>;
  updateDoctorIntegration(id: string, data: Partial<DoctorIntegration>): Promise<DoctorIntegration | undefined>;
  deleteDoctorIntegration(id: string): Promise<boolean>;
  
  // Doctor Emails operations
  getDoctorEmails(doctorId: string, options?: { category?: string; isRead?: boolean; limit?: number; offset?: number }): Promise<DoctorEmail[]>;
  getDoctorEmailByProviderId(doctorId: string, providerMessageId: string): Promise<DoctorEmail | undefined>;
  createDoctorEmail(email: InsertDoctorEmail): Promise<DoctorEmail>;
  updateDoctorEmail(id: string, data: Partial<DoctorEmail>): Promise<DoctorEmail | undefined>;
  
  // Doctor WhatsApp operations
  getDoctorWhatsappMessages(doctorId: string, options?: { status?: string; limit?: number }): Promise<DoctorWhatsappMessage[]>;
  createDoctorWhatsappMessage(message: InsertDoctorWhatsappMessage): Promise<DoctorWhatsappMessage>;
  updateDoctorWhatsappMessage(id: string, data: Partial<DoctorWhatsappMessage>): Promise<DoctorWhatsappMessage | undefined>;
  
  // Call log operations (enhanced)
  getCallLogs(doctorId: string, options?: { status?: string; limit?: number }): Promise<CallLog[]>;
  getCallLogByTwilioSid(twilioCallSid: string): Promise<CallLog | undefined>;
  createCallLog(log: InsertCallLog): Promise<CallLog>;
  updateCallLog(id: string, data: Partial<CallLog>): Promise<CallLog | undefined>;
  
  // Utility
  getUserByPhone(phone: string): Promise<User | undefined>;
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

  async getUserByPhoneNumber(phoneNumber: string): Promise<User | undefined> {
    const [user] = await db.select().from(users).where(eq(users.phoneNumber, phoneNumber));
    return user;
  }

  async getAllDoctors(): Promise<User[]> {
    return await db.select().from(users).where(eq(users.role, 'doctor'));
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

  async getDoctorPatients(doctorId: string): Promise<Array<User & { profile?: PatientProfile }>> {
    // Get unique patient IDs from appointments, prescriptions, and consultations
    const appointmentPatients = await db
      .selectDistinct({ patientId: appointments.patientId })
      .from(appointments)
      .where(eq(appointments.doctorId, doctorId));

    const prescriptionPatients = await db
      .selectDistinct({ patientId: prescriptions.patientId })
      .from(prescriptions)
      .where(eq(prescriptions.doctorId, doctorId));

    const consultationPatients = await db
      .selectDistinct({ patientId: consultations.patientId })
      .from(consultations)
      .where(eq(consultations.doctorId, doctorId));

    // Combine all patient IDs into a unique set
    const allPatientIds = new Set<string>();
    for (const p of appointmentPatients) {
      if (p.patientId) allPatientIds.add(p.patientId);
    }
    for (const p of prescriptionPatients) {
      if (p.patientId) allPatientIds.add(p.patientId);
    }
    for (const p of consultationPatients) {
      if (p.patientId) allPatientIds.add(p.patientId);
    }

    if (allPatientIds.size === 0) {
      return [];
    }

    // Fetch patient data with profiles
    const patientsData = await db
      .select()
      .from(users)
      .leftJoin(patientProfiles, eq(users.id, patientProfiles.userId))
      .where(
        and(
          eq(users.role, 'patient'),
          inArray(users.id, Array.from(allPatientIds))
        )
      );

    return patientsData.map(row => ({
      ...row.users,
      profile: row.patient_profiles || undefined,
    }));
  }

  // Doctor-Patient Assignment operations (HIPAA compliance)
  async getDoctorPatientAssignment(doctorId: string, patientId: string): Promise<DoctorPatientAssignment | undefined> {
    const [assignment] = await db
      .select()
      .from(doctorPatientAssignments)
      .where(
        and(
          eq(doctorPatientAssignments.doctorId, doctorId),
          eq(doctorPatientAssignments.patientId, patientId),
          eq(doctorPatientAssignments.status, 'active')
        )
      );
    return assignment;
  }

  async getActiveDoctorAssignments(doctorId: string): Promise<DoctorPatientAssignment[]> {
    return db
      .select()
      .from(doctorPatientAssignments)
      .where(
        and(
          eq(doctorPatientAssignments.doctorId, doctorId),
          eq(doctorPatientAssignments.status, 'active')
        )
      )
      .orderBy(desc(doctorPatientAssignments.createdAt));
  }

  async getPatientDoctorAssignments(patientId: string): Promise<DoctorPatientAssignment[]> {
    return db
      .select()
      .from(doctorPatientAssignments)
      .where(
        and(
          eq(doctorPatientAssignments.patientId, patientId),
          eq(doctorPatientAssignments.status, 'active')
        )
      )
      .orderBy(desc(doctorPatientAssignments.createdAt));
  }

  async createDoctorPatientAssignment(assignmentData: InsertDoctorPatientAssignment): Promise<DoctorPatientAssignment> {
    // Check if active assignment already exists
    const existing = await this.getDoctorPatientAssignment(assignmentData.doctorId, assignmentData.patientId);
    if (existing) {
      return existing;
    }
    
    const [assignment] = await db
      .insert(doctorPatientAssignments)
      .values(assignmentData)
      .returning();
    return assignment;
  }

  async updateDoctorPatientAssignment(id: string, data: Partial<DoctorPatientAssignment>): Promise<DoctorPatientAssignment | undefined> {
    const [assignment] = await db
      .update(doctorPatientAssignments)
      .set({ ...data, updatedAt: new Date() })
      .where(eq(doctorPatientAssignments.id, id))
      .returning();
    return assignment;
  }

  async revokeDoctorPatientAssignment(id: string, revokedBy: string, reason: string): Promise<DoctorPatientAssignment | undefined> {
    const [assignment] = await db
      .update(doctorPatientAssignments)
      .set({
        status: 'revoked',
        revokedAt: new Date(),
        revokedBy,
        revocationReason: reason,
        updatedAt: new Date(),
      })
      .where(eq(doctorPatientAssignments.id, id))
      .returning();
    return assignment;
  }

  async doctorHasPatientAccess(doctorId: string, patientId: string): Promise<boolean> {
    const assignment = await this.getDoctorPatientAssignment(doctorId, patientId);
    return !!assignment;
  }

  // Patient Consent Request operations
  async searchPatientsByIdentifier(query: string): Promise<Array<{ user: User; profile: PatientProfile | null }>> {
    const normalizedQuery = query.trim().toLowerCase();
    
    // Search by email, phone, or followup patient ID
    const matchingUsers = await db
      .select()
      .from(users)
      .where(
        and(
          eq(users.role, 'patient'),
          or(
            ilike(users.email, `%${normalizedQuery}%`),
            ilike(users.phoneNumber, `%${normalizedQuery}%`)
          )
        )
      )
      .limit(10);
    
    // Also search by followup patient ID in profiles
    const matchingProfiles = await db
      .select()
      .from(patientProfiles)
      .where(ilike(patientProfiles.followupPatientId, `%${normalizedQuery}%`))
      .limit(10);
    
    // Combine results
    const userIds = new Set<string>();
    const results: Array<{ user: User; profile: PatientProfile | null }> = [];
    
    for (const user of matchingUsers) {
      userIds.add(user.id);
      const [profile] = await db
        .select()
        .from(patientProfiles)
        .where(eq(patientProfiles.userId, user.id));
      results.push({ user, profile: profile || null });
    }
    
    for (const profile of matchingProfiles) {
      if (!userIds.has(profile.userId)) {
        const [user] = await db
          .select()
          .from(users)
          .where(eq(users.id, profile.userId));
        if (user) {
          results.push({ user, profile });
        }
      }
    }
    
    return results;
  }

  async createPatientConsentRequest(request: InsertPatientConsentRequest): Promise<PatientConsentRequest> {
    // Check if there's already a pending request
    const existing = await db
      .select()
      .from(patientConsentRequests)
      .where(
        and(
          eq(patientConsentRequests.doctorId, request.doctorId),
          eq(patientConsentRequests.patientId, request.patientId),
          eq(patientConsentRequests.status, 'pending')
        )
      );
    
    if (existing.length > 0) {
      return existing[0];
    }
    
    // Set expiry to 7 days from now
    const expiresAt = new Date();
    expiresAt.setDate(expiresAt.getDate() + 7);
    
    const [consentRequest] = await db
      .insert(patientConsentRequests)
      .values({
        ...request,
        expiresAt,
      })
      .returning();
    return consentRequest;
  }

  async getPendingConsentRequestsForPatient(patientId: string): Promise<PatientConsentRequest[]> {
    return await db
      .select()
      .from(patientConsentRequests)
      .where(
        and(
          eq(patientConsentRequests.patientId, patientId),
          eq(patientConsentRequests.status, 'pending'),
          gt(patientConsentRequests.expiresAt, new Date())
        )
      )
      .orderBy(desc(patientConsentRequests.createdAt));
  }

  async getPendingConsentRequestsForDoctor(doctorId: string): Promise<PatientConsentRequest[]> {
    return await db
      .select()
      .from(patientConsentRequests)
      .where(
        and(
          eq(patientConsentRequests.doctorId, doctorId),
          eq(patientConsentRequests.status, 'pending')
        )
      )
      .orderBy(desc(patientConsentRequests.createdAt));
  }

  async getConsentRequest(id: string): Promise<PatientConsentRequest | undefined> {
    const [request] = await db
      .select()
      .from(patientConsentRequests)
      .where(eq(patientConsentRequests.id, id));
    return request;
  }

  async respondToConsentRequest(id: string, approved: boolean, responseMessage?: string): Promise<PatientConsentRequest | undefined> {
    const [request] = await db
      .update(patientConsentRequests)
      .set({
        status: approved ? 'approved' : 'rejected',
        respondedAt: new Date(),
        responseMessage,
        updatedAt: new Date(),
      })
      .where(eq(patientConsentRequests.id, id))
      .returning();
    return request;
  }

  async generateFollowupPatientId(): Promise<string> {
    // Generate a unique FAI-XXXXXX format ID
    const chars = 'ABCDEFGHJKLMNPQRSTUVWXYZ23456789';
    let patientId: string;
    let exists = true;
    
    while (exists) {
      let randomPart = '';
      for (let i = 0; i < 6; i++) {
        randomPart += chars.charAt(Math.floor(Math.random() * chars.length));
      }
      patientId = `FAI-${randomPart}`;
      
      // Check if this ID already exists
      const [existing] = await db
        .select()
        .from(patientProfiles)
        .where(eq(patientProfiles.followupPatientId, patientId));
      exists = !!existing;
    }
    
    return patientId!;
  }

  async getPatientByFollowupId(followupPatientId: string): Promise<User | undefined> {
    const [profile] = await db
      .select()
      .from(patientProfiles)
      .where(eq(patientProfiles.followupPatientId, followupPatientId.toUpperCase()));
    
    if (!profile) return undefined;
    
    const [user] = await db
      .select()
      .from(users)
      .where(eq(users.id, profile.userId));
    
    return user;
  }

  // Consent Permissions operations (granular HIPAA permissions)
  async createConsentPermissions(permissions: any): Promise<any> {
    try {
      const [result] = await db
        .insert(doctorPatientConsentPermissions)
        .values(permissions)
        .returning();
      return result;
    } catch (error) {
      console.error("[Storage] Error creating consent permissions:", error);
      throw error;
    }
  }

  async getConsentPermissions(assignmentId: string): Promise<any> {
    const [permissions] = await db
      .select()
      .from(doctorPatientConsentPermissions)
      .where(eq(doctorPatientConsentPermissions.assignmentId, assignmentId));
    return permissions;
  }

  async getConsentPermissionsByDoctorPatient(doctorId: string, patientId: string): Promise<any> {
    const [permissions] = await db
      .select()
      .from(doctorPatientConsentPermissions)
      .where(
        and(
          eq(doctorPatientConsentPermissions.doctorId, doctorId),
          eq(doctorPatientConsentPermissions.patientId, patientId)
        )
      );
    return permissions;
  }

  async getPatientConsentPermissions(patientId: string): Promise<any[]> {
    const permissions = await db
      .select()
      .from(doctorPatientConsentPermissions)
      .where(eq(doctorPatientConsentPermissions.patientId, patientId));
    return permissions;
  }

  async updateConsentPermissions(assignmentId: string, permissionUpdates: any): Promise<any> {
    const [result] = await db
      .update(doctorPatientConsentPermissions)
      .set({
        ...permissionUpdates,
        updatedAt: new Date(),
      })
      .where(eq(doctorPatientConsentPermissions.assignmentId, assignmentId))
      .returning();
    return result;
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
          gte(dailyFollowups.date, startOfDay),
          lte(dailyFollowups.date, endOfDay)
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

  // Voice followup operations
  async createVoiceFollowup(followupData: InsertVoiceFollowup): Promise<VoiceFollowup> {
    const [followup] = await db
      .insert(voiceFollowups)
      .values(followupData)
      .returning();
    return followup;
  }

  async getVoiceFollowup(id: string): Promise<VoiceFollowup | undefined> {
    const [followup] = await db
      .select()
      .from(voiceFollowups)
      .where(eq(voiceFollowups.id, id))
      .limit(1);
    return followup;
  }

  async getRecentVoiceFollowups(patientId: string, limit: number = 10): Promise<VoiceFollowup[]> {
    const followups = await db
      .select()
      .from(voiceFollowups)
      .where(eq(voiceFollowups.patientId, patientId))
      .orderBy(desc(voiceFollowups.createdAt))
      .limit(limit);
    return followups;
  }

  async getAllVoiceFollowups(patientId: string): Promise<VoiceFollowup[]> {
    const followups = await db
      .select()
      .from(voiceFollowups)
      .where(eq(voiceFollowups.patientId, patientId))
      .orderBy(desc(voiceFollowups.createdAt));
    return followups;
  }

  // Chat session operations
  async getActiveSession(patientId: string, agentType: string, contextPatientId?: string): Promise<ChatSession | undefined> {
    const conditions = [
      eq(chatSessions.patientId, patientId),
      eq(chatSessions.agentType, agentType),
      sql`${chatSessions.endedAt} IS NULL`
    ];
    
    if (contextPatientId) {
      conditions.push(eq(chatSessions.contextPatientId, contextPatientId));
    } else {
      conditions.push(sql`${chatSessions.contextPatientId} IS NULL`);
    }
    
    const [session] = await db
      .select()
      .from(chatSessions)
      .where(and(...conditions))
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
      gte(chatSessions.startedAt, startDate),
      lte(chatSessions.startedAt, endDate)
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
  async getChatMessages(userId: string, agentType: string, contextPatientId?: string, limit: number = 50): Promise<ChatMessage[]> {
    const conditions = [
      eq(chatMessages.userId, userId),
      eq(chatMessages.agentType, agentType)
    ];
    
    if (contextPatientId) {
      conditions.push(eq(chatMessages.patientContextId, contextPatientId));
    } else {
      conditions.push(sql`${chatMessages.patientContextId} IS NULL`);
    }
    
    const messages = await db
      .select()
      .from(chatMessages)
      .where(and(...conditions))
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
    const normalizedName = name.toLowerCase();
    const [medication] = await db
      .select()
      .from(medications)
      .where(
        and(
          eq(medications.patientId, patientId),
          sql`LOWER(${medications.name}) = ${normalizedName}`,
          eq(medications.active, true)
        )
      );
    return medication;
  }

  async updateMedication(id: string, data: Partial<Medication>): Promise<Medication | undefined> {
    const [medication] = await db
      .update(medications)
      .set({ ...data, updatedAt: new Date() })
      .where(eq(medications.id, id))
      .returning();
    return medication;
  }

  async getAllMedications(patientId: string): Promise<Medication[]> {
    const meds = await db
      .select()
      .from(medications)
      .where(eq(medications.patientId, patientId))
      .orderBy(medications.name);
    return meds;
  }

  async getPendingConfirmationMedications(patientId: string): Promise<Medication[]> {
    const meds = await db
      .select()
      .from(medications)
      .where(and(
        eq(medications.patientId, patientId),
        eq(medications.status, "pending_confirmation")
      ))
      .orderBy(medications.createdAt);
    return meds;
  }

  async getInactiveMedications(patientId: string): Promise<Medication[]> {
    const meds = await db
      .select()
      .from(medications)
      .where(and(
        eq(medications.patientId, patientId),
        eq(medications.status, "inactive")
      ))
      .orderBy(desc(medications.discontinuedAt));
    return meds;
  }

  async confirmMedication(id: string, confirmedBy: string): Promise<Medication | undefined> {
    const [medication] = await db
      .update(medications)
      .set({
        status: "active",
        confirmedAt: new Date(),
        confirmedBy,
        updatedAt: new Date(),
      })
      .where(eq(medications.id, id))
      .returning();
    return medication;
  }

  async discontinueMedication(
    id: string,
    discontinuedBy: string,
    reason: string,
    replacementId?: string
  ): Promise<Medication | undefined> {
    const [medication] = await db
      .update(medications)
      .set({
        status: "inactive",
        active: false,
        discontinuedAt: new Date(),
        discontinuedBy,
        discontinuationReason: reason,
        replacementMedicationId: replacementId || null,
        updatedAt: new Date(),
      })
      .where(eq(medications.id, id))
      .returning();
    return medication;
  }

  async reactivateMedication(id: string, reactivatedBy: string): Promise<Medication | undefined> {
    const [medication] = await db
      .update(medications)
      .set({
        status: "active",
        active: true,
        discontinuedAt: null,
        discontinuedBy: null,
        discontinuationReason: null,
        replacementMedicationId: null,
        updatedAt: new Date(),
      })
      .where(eq(medications.id, id))
      .returning();
    return medication;
  }

  // Prescription operations
  async getPrescriptions(patientId: string): Promise<Prescription[]> {
    const rxs = await db
      .select()
      .from(prescriptions)
      .where(eq(prescriptions.patientId, patientId))
      .orderBy(desc(prescriptions.createdAt));
    return rxs;
  }

  async getPrescriptionsByDoctor(doctorId: string): Promise<Prescription[]> {
    const rxs = await db
      .select()
      .from(prescriptions)
      .where(eq(prescriptions.doctorId, doctorId))
      .orderBy(desc(prescriptions.createdAt));
    return rxs;
  }

  async getPrescription(id: string): Promise<Prescription | undefined> {
    const [rx] = await db
      .select()
      .from(prescriptions)
      .where(eq(prescriptions.id, id));
    return rx;
  }

  async createPrescription(prescriptionData: InsertPrescription): Promise<Prescription> {
    const [rx] = await db
      .insert(prescriptions)
      .values(prescriptionData)
      .returning();
    return rx;
  }

  async updatePrescription(id: string, data: Partial<Prescription>): Promise<Prescription | undefined> {
    const [rx] = await db
      .update(prescriptions)
      .set({ ...data, updatedAt: new Date() })
      .where(eq(prescriptions.id, id))
      .returning();
    return rx;
  }

  async acknowledgePrescription(id: string, acknowledgedBy: string): Promise<Prescription | undefined> {
    const [rx] = await db
      .update(prescriptions)
      .set({
        status: "acknowledged",
        acknowledgedAt: new Date(),
        acknowledgedBy,
        updatedAt: new Date(),
      })
      .where(eq(prescriptions.id, id))
      .returning();
    return rx;
  }

  // Chronic care medication lifecycle operations
  async getMedicationsBySpecialty(patientId: string, specialty: string): Promise<Medication[]> {
    const meds = await db
      .select()
      .from(medications)
      .where(and(
        eq(medications.patientId, patientId),
        eq(medications.specialty, specialty),
        eq(medications.active, true)
      ))
      .orderBy(medications.name);
    return meds;
  }

  async getMedicationsByDrugClass(patientId: string, drugClassPattern: string): Promise<Medication[]> {
    const meds = await db
      .select()
      .from(medications)
      .where(and(
        eq(medications.patientId, patientId),
        eq(medications.active, true)
      ))
      .orderBy(medications.name);
    
    // Filter by drug class using joined drug table
    const medsWithDrugs = await Promise.all(meds.map(async (med) => {
      if (med.drugId) {
        const [drug] = await db.select().from(drugs).where(eq(drugs.id, med.drugId));
        if (drug?.drugClass && drug.drugClass.toLowerCase().includes(drugClassPattern.toLowerCase())) {
          return med;
        }
      }
      return null;
    }));
    
    return medsWithDrugs.filter((m): m is Medication => m !== null);
  }

  async supersedeMedication(oldMedicationId: string, newMedicationId: string, reason: string): Promise<Medication | undefined> {
    const now = new Date();
    
    // Mark the old medication as superseded
    const [oldMed] = await db
      .update(medications)
      .set({
        status: "superseded",
        active: false,
        supersededBy: newMedicationId,
        supersededAt: now,
        supersessionReason: reason,
        updatedAt: now,
      })
      .where(eq(medications.id, oldMedicationId))
      .returning();
    
    return oldMed;
  }

  // Cross-specialty conflict operations
  async getMedicationConflicts(patientId: string): Promise<any[]> {
    const conflicts = await db.execute(sql`
      SELECT * FROM medication_conflicts 
      WHERE patient_id = ${patientId} 
      ORDER BY created_at DESC
    `);
    return conflicts.rows as any[];
  }

  async getPendingConflicts(doctorId: string): Promise<any[]> {
    const conflicts = await db.execute(sql`
      SELECT * FROM medication_conflicts 
      WHERE (doctor1_id = ${doctorId} OR doctor2_id = ${doctorId})
        AND status = 'pending'
      ORDER BY created_at DESC
    `);
    return conflicts.rows as any[];
  }

  async createMedicationConflict(conflict: any): Promise<any> {
    const result = await db.execute(sql`
      INSERT INTO medication_conflicts (
        patient_id, conflict_group_id, medication1_id, medication2_id,
        prescription1_id, prescription2_id, doctor1_id, doctor2_id,
        specialty1, specialty2, conflict_type, severity,
        description, detected_reason, status
      ) VALUES (
        ${conflict.patientId}, ${conflict.conflictGroupId}, ${conflict.medication1Id}, ${conflict.medication2Id},
        ${conflict.prescription1Id || null}, ${conflict.prescription2Id || null}, ${conflict.doctor1Id}, ${conflict.doctor2Id},
        ${conflict.specialty1}, ${conflict.specialty2}, ${conflict.conflictType}, ${conflict.severity},
        ${conflict.description}, ${conflict.detectedReason || null}, 'pending'
      ) RETURNING *
    `);
    return result.rows[0];
  }

  async getMedicationConflict(id: string): Promise<any | undefined> {
    const result = await db.execute(sql`
      SELECT * FROM medication_conflicts WHERE id = ${id}
    `);
    return result.rows[0];
  }

  async updateMedicationConflict(id: string, data: any): Promise<any | undefined> {
    const setClauses: string[] = [];
    const values: any[] = [];
    
    Object.entries(data).forEach(([key, value]) => {
      const snakeKey = key.replace(/([A-Z])/g, '_$1').toLowerCase();
      setClauses.push(`${snakeKey} = $${values.length + 1}`);
      values.push(value);
    });
    
    if (setClauses.length === 0) return undefined;
    
    const result = await db.execute(sql`
      UPDATE medication_conflicts 
      SET ${sql.raw(setClauses.join(', '))}, updated_at = NOW()
      WHERE id = ${id}
      RETURNING *
    `);
    return result.rows[0];
  }

  async resolveMedicationConflict(id: string, resolution: any): Promise<any | undefined> {
    const result = await db.execute(sql`
      UPDATE medication_conflicts SET
        status = 'resolved',
        resolution = ${resolution.resolution},
        resolution_details = ${resolution.resolutionDetails || null},
        resolved_by = ${resolution.resolvedBy},
        resolved_at = NOW(),
        updated_at = NOW()
      WHERE id = ${id}
      RETURNING *
    `);
    return result.rows[0];
  }

  // Medication change log operations
  async getMedicationChangelog(medicationId: string): Promise<MedicationChangeLog[]> {
    const logs = await db
      .select()
      .from(medicationChangeLog)
      .where(eq(medicationChangeLog.medicationId, medicationId))
      .orderBy(desc(medicationChangeLog.createdAt));
    return logs;
  }

  async getPatientMedicationChangelog(patientId: string, limit: number = 50): Promise<MedicationChangeLog[]> {
    const logs = await db
      .select()
      .from(medicationChangeLog)
      .where(eq(medicationChangeLog.patientId, patientId))
      .orderBy(desc(medicationChangeLog.createdAt))
      .limit(limit);
    return logs;
  }

  async createMedicationChangeLog(logData: InsertMedicationChangeLog): Promise<MedicationChangeLog> {
    const [log] = await db
      .insert(medicationChangeLog)
      .values(logData)
      .returning();
    return log;
  }

  // Dosage change request operations
  async getDosageChangeRequests(patientId: string): Promise<DosageChangeRequest[]> {
    const requests = await db
      .select()
      .from(dosageChangeRequests)
      .where(eq(dosageChangeRequests.patientId, patientId))
      .orderBy(desc(dosageChangeRequests.createdAt));
    return requests;
  }

  async getPendingDosageChangeRequests(doctorId: string): Promise<DosageChangeRequest[]> {
    // Get all pending requests for patients under this doctor
    const requests = await db
      .select()
      .from(dosageChangeRequests)
      .where(eq(dosageChangeRequests.status, "pending"))
      .orderBy(dosageChangeRequests.createdAt);
    return requests;
  }

  async getDosageChangeRequest(id: string): Promise<DosageChangeRequest | undefined> {
    const [request] = await db
      .select()
      .from(dosageChangeRequests)
      .where(eq(dosageChangeRequests.id, id));
    return request;
  }

  async createDosageChangeRequest(requestData: InsertDosageChangeRequest): Promise<DosageChangeRequest> {
    const [request] = await db
      .insert(dosageChangeRequests)
      .values(requestData)
      .returning();
    return request;
  }

  async approveDosageChangeRequest(id: string, doctorId: string, notes?: string): Promise<DosageChangeRequest | undefined> {
    const [request] = await db
      .update(dosageChangeRequests)
      .set({
        status: "approved",
        reviewedByDoctorId: doctorId,
        reviewedAt: new Date(),
        doctorNotes: notes || null,
        updatedAt: new Date(),
      })
      .where(eq(dosageChangeRequests.id, id))
      .returning();
    return request;
  }

  async rejectDosageChangeRequest(id: string, doctorId: string, notes: string): Promise<DosageChangeRequest | undefined> {
    const [request] = await db
      .update(dosageChangeRequests)
      .set({
        status: "rejected",
        reviewedByDoctorId: doctorId,
        reviewedAt: new Date(),
        doctorNotes: notes,
        updatedAt: new Date(),
      })
      .where(eq(dosageChangeRequests.id, id))
      .returning();
    return request;
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

  // Unified verification method that updates both users and doctorProfiles
  async verifyDoctorApplication(
    userId: string, 
    verified: boolean, 
    notes: string, 
    verifiedBy: string
  ): Promise<{ user: User | undefined; doctorProfile: DoctorProfile | undefined }> {
    // Start a transaction to update both tables atomically
    const now = new Date();
    
    // Update users table (authorization control)
    const [updatedUser] = await db
      .update(users)
      .set({
        adminVerified: verified,
        adminVerifiedAt: now,
        adminVerifiedBy: verifiedBy,
      })
      .where(eq(users.id, userId))
      .returning();

    // Update doctorProfiles table (compliance evidencing)
    const [updatedProfile] = await db
      .update(doctorProfiles)
      .set({
        licenseVerified: verified,
        verificationNotes: notes,
        verifiedBy,
        verifiedAt: now,
      })
      .where(eq(doctorProfiles.userId, userId))
      .returning();

    return { user: updatedUser, doctorProfile: updatedProfile };
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
    const searchPattern = `%${query.toLowerCase()}%`;
    const results = await db
      .select()
      .from(drugs)
      .where(
        or(
          sql`LOWER(${drugs.name}) LIKE ${searchPattern}`,
          sql`LOWER(${drugs.genericName}) LIKE ${searchPattern}`
        )
      )
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
        or(
          and(
            eq(drugInteractions.drug1Id, drug1Id),
            eq(drugInteractions.drug2Id, drug2Id)
          ),
          and(
            eq(drugInteractions.drug1Id, drug2Id),
            eq(drugInteractions.drug2Id, drug1Id)
          )
        )
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
        and(
          eq(interactionAlerts.patientId, patientId),
          eq(interactionAlerts.alertStatus, 'active')
        )
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
          gte(immuneBiomarkers.measuredAt, startDate),
          lte(immuneBiomarkers.measuredAt, endDate)
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

  async getEnvironmentalRiskDataByUser(userId: string, limit: number = 30): Promise<EnvironmentalRiskData[]> {
    const riskData = await db
      .select()
      .from(environmentalRiskData)
      .where(eq(environmentalRiskData.userId, userId))
      .orderBy(desc(environmentalRiskData.measuredAt))
      .limit(limit);
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

  // Correlation pattern operations
  async getCorrelationPatterns(userId: string, limit: number = 50): Promise<CorrelationPattern[]> {
    const patterns = await db
      .select()
      .from(correlationPatterns)
      .where(eq(correlationPatterns.userId, userId))
      .orderBy(desc(correlationPatterns.createdAt))
      .limit(limit);
    return patterns;
  }

  async getCorrelationPatternsByType(userId: string, patternType: string): Promise<CorrelationPattern[]> {
    const patterns = await db
      .select()
      .from(correlationPatterns)
      .where(
        and(
          eq(correlationPatterns.userId, userId),
          eq(correlationPatterns.patternType, patternType)
        )
      )
      .orderBy(desc(correlationPatterns.correlationStrength));
    return patterns;
  }

  async getHighSeverityPatterns(userId: string): Promise<CorrelationPattern[]> {
    const patterns = await db
      .select()
      .from(correlationPatterns)
      .where(
        and(
          eq(correlationPatterns.userId, userId),
          inArray(correlationPatterns.severity, ['high', 'critical'])
        )
      )
      .orderBy(desc(correlationPatterns.createdAt));
    return patterns;
  }

  async createCorrelationPattern(pattern: InsertCorrelationPattern): Promise<CorrelationPattern> {
    const [newPattern] = await db
      .insert(correlationPatterns)
      .values(pattern)
      .returning();
    return newPattern;
  }

  // Genetic variant operations
  async getGeneticVariants(userId: string): Promise<GeneticVariant[]> {
    const variants = await db
      .select()
      .from(geneticVariants)
      .where(eq(geneticVariants.userId, userId))
      .orderBy(desc(geneticVariants.createdAt));
    return variants;
  }

  async getHighRiskVariants(userId: string): Promise<GeneticVariant[]> {
    const variants = await db
      .select()
      .from(geneticVariants)
      .where(
        and(
          eq(geneticVariants.userId, userId),
          sql`${geneticVariants.riskLevel} IN ('high', 'critical')`
        )
      )
      .orderBy(desc(geneticVariants.createdAt));
    return variants;
  }

  async createGeneticVariant(variant: InsertGeneticVariant): Promise<GeneticVariant> {
    const [newVariant] = await db
      .insert(geneticVariants)
      .values(variant)
      .returning();
    return newVariant;
  }

  // Pharmacogenomic report operations
  async getPharmacogenomicReports(userId: string): Promise<PharmacogenomicReport[]> {
    const reports = await db
      .select()
      .from(pharmacogenomicReports)
      .where(eq(pharmacogenomicReports.userId, userId))
      .orderBy(desc(pharmacogenomicReports.testDate));
    return reports;
  }

  async getPharmacogenomicReport(id: string): Promise<PharmacogenomicReport | undefined> {
    const [report] = await db
      .select()
      .from(pharmacogenomicReports)
      .where(eq(pharmacogenomicReports.id, id))
      .limit(1);
    return report;
  }

  async createPharmacogenomicReport(report: InsertPharmacogenomicReport): Promise<PharmacogenomicReport> {
    const [newReport] = await db
      .insert(pharmacogenomicReports)
      .values(report)
      .returning();
    return newReport;
  }

  async updatePharmacogenomicReportStatus(id: string, status: string, error?: string): Promise<PharmacogenomicReport | undefined> {
    const [report] = await db
      .update(pharmacogenomicReports)
      .set({
        processingStatus: status,
        processingError: error,
        updatedAt: new Date(),
      })
      .where(eq(pharmacogenomicReports.id, id))
      .returning();
    return report;
  }

  // Clinical trial operations
  async getClinicalTrials(conditions?: string[], status?: string, limit: number = 50): Promise<ClinicalTrial[]> {
    let query = db.select().from(clinicalTrials);
    
    if (status) {
      query = query.where(eq(clinicalTrials.status, status));
    }
    
    const trials = await query
      .orderBy(desc(clinicalTrials.lastUpdated))
      .limit(limit);
    
    return trials;
  }

  async getClinicalTrial(id: string): Promise<ClinicalTrial | undefined> {
    const [trial] = await db
      .select()
      .from(clinicalTrials)
      .where(eq(clinicalTrials.id, id))
      .limit(1);
    return trial;
  }

  async createClinicalTrial(trial: InsertClinicalTrial): Promise<ClinicalTrial> {
    const [newTrial] = await db
      .insert(clinicalTrials)
      .values(trial)
      .returning();
    return newTrial;
  }

  async updateClinicalTrial(id: string, data: Partial<ClinicalTrial>): Promise<ClinicalTrial | undefined> {
    const [trial] = await db
      .update(clinicalTrials)
      .set({
        ...data,
        lastUpdated: new Date(),
      })
      .where(eq(clinicalTrials.id, id))
      .returning();
    return trial;
  }

  // Trial match score operations
  async getTrialMatchScores(userId: string, limit: number = 50): Promise<TrialMatchScore[]> {
    const matchScores = await db
      .select()
      .from(trialMatchScores)
      .where(eq(trialMatchScores.userId, userId))
      .orderBy(desc(trialMatchScores.overallScore))
      .limit(limit);
    return matchScores;
  }

  async getHighMatchTrials(userId: string, minScore: number = 0.7): Promise<TrialMatchScore[]> {
    const matchScores = await db
      .select()
      .from(trialMatchScores)
      .where(
        and(
          eq(trialMatchScores.userId, userId),
          gte(trialMatchScores.overallScore, minScore)
        )
      )
      .orderBy(desc(trialMatchScores.overallScore));
    return matchScores;
  }

  async createTrialMatchScore(matchScore: InsertTrialMatchScore): Promise<TrialMatchScore> {
    const [newMatchScore] = await db
      .insert(trialMatchScores)
      .values(matchScore)
      .returning();
    return newMatchScore;
  }

  async updateTrialMatchStatus(id: string, status: string, notes?: string): Promise<TrialMatchScore | undefined> {
    const [matchScore] = await db
      .update(trialMatchScores)
      .set({
        status,
        notes,
        updatedAt: new Date(),
      })
      .where(eq(trialMatchScores.id, id))
      .returning();
    return matchScore;
  }

  // Deterioration prediction operations
  async getDeteriorationPredictions(userId: string, limit: number = 30): Promise<DeteriorationPrediction[]> {
    const predictions = await db
      .select()
      .from(deteriorationPredictions)
      .where(eq(deteriorationPredictions.userId, userId))
      .orderBy(desc(deteriorationPredictions.predictionDate))
      .limit(limit);
    return predictions;
  }

  async getLatestPrediction(userId: string): Promise<DeteriorationPrediction | undefined> {
    const [prediction] = await db
      .select()
      .from(deteriorationPredictions)
      .where(eq(deteriorationPredictions.userId, userId))
      .orderBy(desc(deteriorationPredictions.predictionDate))
      .limit(1);
    return prediction;
  }

  async getHighRiskPredictions(userId: string): Promise<DeteriorationPrediction[]> {
    const predictions = await db
      .select()
      .from(deteriorationPredictions)
      .where(
        and(
          eq(deteriorationPredictions.userId, userId),
          inArray(deteriorationPredictions.riskLevel, ['high', 'critical'])
        )
      )
      .orderBy(desc(deteriorationPredictions.predictionDate));
    return predictions;
  }

  async createDeteriorationPrediction(prediction: InsertDeteriorationPrediction): Promise<DeteriorationPrediction> {
    const [newPrediction] = await db
      .insert(deteriorationPredictions)
      .values(prediction)
      .returning();
    return newPrediction;
  }

  async updatePredictionOutcome(id: string, outcome: string, notes?: string): Promise<DeteriorationPrediction | undefined> {
    const [prediction] = await db
      .update(deteriorationPredictions)
      .set({
        actualOutcome: outcome,
        outcomeNotes: notes,
        outcomeDate: new Date(),
        updatedAt: new Date(),
      })
      .where(eq(deteriorationPredictions.id, id))
      .returning();
    return prediction;
  }

  // Medication schedule operations
  async getMedicationSchedules(medicationId: string): Promise<MedicationSchedule[]> {
    const schedules = await db
      .select()
      .from(medicationSchedules)
      .where(eq(medicationSchedules.medicationId, medicationId));
    return schedules;
  }

  async getPatientMedicationSchedules(patientId: string, activeOnly: boolean = true): Promise<MedicationSchedule[]> {
    if (activeOnly) {
      const schedules = await db
        .select()
        .from(medicationSchedules)
        .where(
          and(
            eq(medicationSchedules.patientId, patientId),
            eq(medicationSchedules.active, true)
          )
        );
      return schedules;
    } else {
      const schedules = await db
        .select()
        .from(medicationSchedules)
        .where(eq(medicationSchedules.patientId, patientId));
      return schedules;
    }
  }

  async createMedicationSchedule(schedule: InsertMedicationSchedule): Promise<MedicationSchedule> {
    const [newSchedule] = await db
      .insert(medicationSchedules)
      .values(schedule)
      .returning();
    return newSchedule;
  }

  async updateMedicationSchedule(id: string, data: Partial<MedicationSchedule>): Promise<MedicationSchedule | undefined> {
    const [schedule] = await db
      .update(medicationSchedules)
      .set({ ...data, updatedAt: new Date() })
      .where(eq(medicationSchedules.id, id))
      .returning();
    return schedule;
  }

  // Medication adherence operations
  async getMedicationAdherence(medicationId: string, limit: number = 100): Promise<MedicationAdherence[]> {
    const adherence = await db
      .select()
      .from(medicationAdherence)
      .where(eq(medicationAdherence.medicationId, medicationId))
      .orderBy(desc(medicationAdherence.scheduledTime))
      .limit(limit);
    return adherence;
  }

  async getPatientAdherence(patientId: string, limit: number = 100): Promise<MedicationAdherence[]> {
    const adherence = await db
      .select()
      .from(medicationAdherence)
      .where(eq(medicationAdherence.patientId, patientId))
      .orderBy(desc(medicationAdherence.scheduledTime))
      .limit(limit);
    return adherence;
  }

  async getPendingMedications(patientId: string): Promise<MedicationAdherence[]> {
    const pending = await db
      .select()
      .from(medicationAdherence)
      .where(
        and(
          eq(medicationAdherence.patientId, patientId),
          eq(medicationAdherence.status, "pending")
        )
      )
      .orderBy(medicationAdherence.scheduledTime);
    return pending;
  }

  async createMedicationAdherence(adherence: InsertMedicationAdherence): Promise<MedicationAdherence> {
    const [newAdherence] = await db
      .insert(medicationAdherence)
      .values(adherence)
      .returning();
    return newAdherence;
  }

  async updateMedicationAdherence(id: string, data: Partial<MedicationAdherence>): Promise<MedicationAdherence | undefined> {
    const [adherence] = await db
      .update(medicationAdherence)
      .set(data)
      .where(eq(medicationAdherence.id, id))
      .returning();
    return adherence;
  }

  // Dietary preference operations
  async getDietaryPreferences(patientId: string): Promise<DietaryPreference | undefined> {
    const [preferences] = await db
      .select()
      .from(dietaryPreferences)
      .where(eq(dietaryPreferences.patientId, patientId));
    return preferences;
  }

  async upsertDietaryPreferences(preferences: InsertDietaryPreference): Promise<DietaryPreference> {
    const [upserted] = await db
      .insert(dietaryPreferences)
      .values(preferences)
      .onConflictDoUpdate({
        target: dietaryPreferences.patientId,
        set: { ...preferences, updatedAt: new Date() },
      })
      .returning();
    return upserted;
  }

  // Meal plan operations
  async getMealPlans(patientId: string, activeOnly: boolean = false): Promise<MealPlan[]> {
    if (activeOnly) {
      const plans = await db
        .select()
        .from(mealPlans)
        .where(
          and(
            eq(mealPlans.patientId, patientId),
            eq(mealPlans.active, true)
          )
        )
        .orderBy(desc(mealPlans.weekStartDate));
      return plans;
    } else {
      const plans = await db
        .select()
        .from(mealPlans)
        .where(eq(mealPlans.patientId, patientId))
        .orderBy(desc(mealPlans.weekStartDate));
      return plans;
    }
  }

  async getActiveMealPlan(patientId: string): Promise<MealPlan | undefined> {
    const [plan] = await db
      .select()
      .from(mealPlans)
      .where(
        and(
          eq(mealPlans.patientId, patientId),
          eq(mealPlans.active, true)
        )
      )
      .orderBy(desc(mealPlans.weekStartDate))
      .limit(1);
    return plan;
  }

  async createMealPlan(mealPlan: InsertMealPlan): Promise<MealPlan> {
    const [newPlan] = await db
      .insert(mealPlans)
      .values(mealPlan)
      .returning();
    return newPlan;
  }

  async updateMealPlan(id: string, data: Partial<MealPlan>): Promise<MealPlan | undefined> {
    const [plan] = await db
      .update(mealPlans)
      .set(data)
      .where(eq(mealPlans.id, id))
      .returning();
    return plan;
  }

  // Meal operations
  async getMeals(patientId: string, mealPlanId?: string, limit: number = 100): Promise<Meal[]> {
    if (mealPlanId) {
      const mealsData = await db
        .select()
        .from(meals)
        .where(
          and(
            eq(meals.patientId, patientId),
            eq(meals.mealPlanId, mealPlanId)
          )
        )
        .orderBy(desc(meals.scheduledTime))
        .limit(limit);
      return mealsData;
    } else {
      const mealsData = await db
        .select()
        .from(meals)
        .where(eq(meals.patientId, patientId))
        .orderBy(desc(meals.scheduledTime))
        .limit(limit);
      return mealsData;
    }
  }

  async getMealsByDateRange(patientId: string, startDate: Date, endDate: Date): Promise<Meal[]> {
    const mealsData = await db
      .select()
      .from(meals)
      .where(
        and(
          eq(meals.patientId, patientId),
          between(meals.scheduledTime, startDate, endDate)
        )
      )
      .orderBy(meals.scheduledTime);
    return mealsData;
  }

  async getTodaysMeals(patientId: string): Promise<Meal[]> {
    const today = new Date();
    today.setHours(0, 0, 0, 0);
    const tomorrow = new Date(today);
    tomorrow.setDate(tomorrow.getDate() + 1);
    return this.getMealsByDateRange(patientId, today, tomorrow);
  }

  async createMeal(meal: InsertMeal): Promise<Meal> {
    const [newMeal] = await db
      .insert(meals)
      .values(meal)
      .returning();
    return newMeal;
  }

  async updateMeal(id: string, data: Partial<Meal>): Promise<Meal | undefined> {
    const [meal] = await db
      .update(meals)
      .set(data)
      .where(eq(meals.id, id))
      .returning();
    return meal;
  }

  // Nutrition entry operations
  async getNutritionEntries(mealId: string): Promise<NutritionEntry[]> {
    const entries = await db
      .select()
      .from(nutritionEntries)
      .where(eq(nutritionEntries.mealId, mealId));
    return entries;
  }

  async getPatientNutritionEntries(patientId: string, limit: number = 100): Promise<NutritionEntry[]> {
    const entries = await db
      .select()
      .from(nutritionEntries)
      .where(eq(nutritionEntries.patientId, patientId))
      .orderBy(desc(nutritionEntries.createdAt))
      .limit(limit);
    return entries;
  }

  async createNutritionEntry(entry: InsertNutritionEntry): Promise<NutritionEntry> {
    const [newEntry] = await db
      .insert(nutritionEntries)
      .values(entry)
      .returning();
    return newEntry;
  }

  // Companion check-in operations
  async getCompanionCheckIns(patientId: string, limit: number = 100): Promise<CompanionCheckIn[]> {
    const checkIns = await db
      .select()
      .from(companionCheckIns)
      .where(eq(companionCheckIns.patientId, patientId))
      .orderBy(desc(companionCheckIns.checkedInAt))
      .limit(limit);
    return checkIns;
  }

  async getCheckInsByType(patientId: string, checkInType: string, limit: number = 50): Promise<CompanionCheckIn[]> {
    const checkIns = await db
      .select()
      .from(companionCheckIns)
      .where(
        and(
          eq(companionCheckIns.patientId, patientId),
          eq(companionCheckIns.checkInType, checkInType)
        )
      )
      .orderBy(desc(companionCheckIns.checkedInAt))
      .limit(limit);
    return checkIns;
  }

  async getRecentCheckIns(patientId: string, days: number): Promise<CompanionCheckIn[]> {
    const since = new Date();
    since.setDate(since.getDate() - days);
    const checkIns = await db
      .select()
      .from(companionCheckIns)
      .where(
        and(
          eq(companionCheckIns.patientId, patientId),
          gte(companionCheckIns.checkedInAt, since)
        )
      )
      .orderBy(desc(companionCheckIns.checkedInAt));
    return checkIns;
  }

  async createCompanionCheckIn(checkIn: InsertCompanionCheckIn): Promise<CompanionCheckIn> {
    const [newCheckIn] = await db
      .insert(companionCheckIns)
      .values(checkIn)
      .returning();
    return newCheckIn;
  }

  // Companion engagement operations
  async getCompanionEngagement(patientId: string): Promise<CompanionEngagement | undefined> {
    const [engagement] = await db
      .select()
      .from(companionEngagement)
      .where(eq(companionEngagement.patientId, patientId));
    return engagement;
  }

  async upsertCompanionEngagement(engagement: InsertCompanionEngagement): Promise<CompanionEngagement> {
    const [upserted] = await db
      .insert(companionEngagement)
      .values(engagement)
      .onConflictDoUpdate({
        target: companionEngagement.patientId,
        set: { ...engagement, updatedAt: new Date() },
      })
      .returning();
    return upserted;
  }

  async updateCompanionEngagement(patientId: string, data: Partial<CompanionEngagement>): Promise<CompanionEngagement | undefined> {
    const [engagement] = await db
      .update(companionEngagement)
      .set({ ...data, updatedAt: new Date() })
      .where(eq(companionEngagement.patientId, patientId))
      .returning();
    return engagement;
  }

  // ML/RL System operations
  async getUserLearningProfile(userId: string, agentType: string): Promise<UserLearningProfile | undefined> {
    const [profile] = await db
      .select()
      .from(userLearningProfiles)
      .where(and(
        eq(userLearningProfiles.userId, userId),
        eq(userLearningProfiles.agentType, agentType)
      ));
    return profile;
  }

  async upsertUserLearningProfile(profileData: InsertUserLearningProfile): Promise<UserLearningProfile> {
    const [profile] = await db
      .insert(userLearningProfiles)
      .values(profileData)
      .onConflictDoUpdate({
        target: [userLearningProfiles.userId, userLearningProfiles.agentType],
        set: { ...profileData, updatedAt: new Date() },
      })
      .returning();
    return profile;
  }

  async getHabits(userId: string): Promise<Habit[]> {
    const habitsList = await db
      .select()
      .from(habits)
      .where(eq(habits.userId, userId))
      .orderBy(desc(habits.createdAt));
    return habitsList;
  }

  async createHabit(habitData: InsertHabit): Promise<Habit> {
    const [habit] = await db
      .insert(habits)
      .values(habitData)
      .returning();
    return habit;
  }

  async updateHabit(id: string, data: Partial<Habit>): Promise<Habit | undefined> {
    const [habit] = await db
      .update(habits)
      .set({ ...data, updatedAt: new Date() })
      .where(eq(habits.id, id))
      .returning();
    return habit;
  }

  async getRecentHabitCompletions(userId: string, days: number): Promise<HabitCompletion[]> {
    const since = new Date();
    since.setDate(since.getDate() - days);
    
    const completions = await db
      .select()
      .from(habitCompletions)
      .innerJoin(habits, eq(habitCompletions.habitId, habits.id))
      .where(and(
        eq(habits.userId, userId),
        gte(habitCompletions.completionDate, since)
      ))
      .orderBy(desc(habitCompletions.completionDate));
    return completions.map(c => c.habit_completions);
  }

  async createHabitCompletion(completionData: InsertHabitCompletion): Promise<HabitCompletion> {
    const [completion] = await db
      .insert(habitCompletions)
      .values(completionData)
      .returning();
    return completion;
  }

  async getUserRecommendations(userId: string): Promise<MLRecommendation[]> {
    const recommendations = await db
      .select()
      .from(mlRecommendations)
      .where(eq(mlRecommendations.userId, userId))
      .orderBy(desc(mlRecommendations.createdAt));
    return recommendations;
  }

  async createMLRecommendation(recommendationData: InsertMLRecommendation): Promise<MLRecommendation> {
    const [recommendation] = await db
      .insert(mlRecommendations)
      .values(recommendationData)
      .returning();
    return recommendation;
  }

  async updateMLRecommendation(id: string, data: Partial<MLRecommendation>): Promise<MLRecommendation | undefined> {
    const [recommendation] = await db
      .update(mlRecommendations)
      .set({ ...data, updatedAt: new Date() })
      .where(eq(mlRecommendations.id, id))
      .returning();
    return recommendation;
  }

  async createRLReward(rewardData: InsertRLReward): Promise<RLReward> {
    const [reward] = await db
      .insert(rlRewards)
      .values(rewardData)
      .returning();
    return reward;
  }

  async getDailyEngagement(userId: string, date: Date): Promise<DailyEngagement | undefined> {
    const [engagement] = await db
      .select()
      .from(dailyEngagement)
      .where(and(
        eq(dailyEngagement.userId, userId),
        eq(dailyEngagement.date, date)
      ));
    return engagement;
  }

  async upsertDailyEngagement(engagementData: InsertDailyEngagement): Promise<DailyEngagement> {
    const [engagement] = await db
      .insert(dailyEngagement)
      .values(engagementData)
      .onConflictDoUpdate({
        target: [dailyEngagement.userId, dailyEngagement.date],
        set: { ...engagementData, updatedAt: new Date() },
      })
      .returning();
    return engagement;
  }

  async getDoctorWellnessHistory(userId: string, days: number): Promise<DoctorWellness[]> {
    const since = new Date();
    since.setDate(since.getDate() - days);
    
    const wellnessHistory = await db
      .select()
      .from(doctorWellness)
      .where(and(
        eq(doctorWellness.doctorId, userId),
        gte(doctorWellness.date, since)
      ))
      .orderBy(desc(doctorWellness.date));
    return wellnessHistory;
  }

  async createDoctorWellness(wellnessData: InsertDoctorWellness): Promise<DoctorWellness> {
    const [wellness] = await db
      .insert(doctorWellness)
      .values(wellnessData)
      .returning();
    return wellness;
  }

  async getMilestones(userId: string): Promise<Milestone[]> {
    const milestonesList = await db
      .select()
      .from(milestones)
      .where(eq(milestones.userId, userId))
      .orderBy(desc(milestones.achievedAt));
    return milestonesList;
  }

  async createMilestone(milestoneData: InsertMilestone): Promise<Milestone> {
    const [milestone] = await db
      .insert(milestones)
      .values(milestoneData)
      .returning();
    return milestone;
  }

  async updateMilestone(id: string, data: Partial<Milestone>): Promise<Milestone | undefined> {
    const [milestone] = await db
      .update(milestones)
      .set({ ...data, updatedAt: new Date() })
      .where(eq(milestones.id, id))
      .returning();
    return milestone;
  }

  // ============================================================================
  // RECEPTIONIST & ASSISTANT LYSA FEATURES - APPOINTMENTS
  // ============================================================================

  async createAppointment(appointmentData: InsertAppointment): Promise<Appointment> {
    const [appointment] = await db
      .insert(appointments)
      .values(appointmentData)
      .returning();
    return appointment;
  }

  async getAppointment(id: string): Promise<Appointment | undefined> {
    const [appointment] = await db
      .select()
      .from(appointments)
      .where(eq(appointments.id, id));
    return appointment;
  }

  async getAppointmentByExternalId(googleCalendarEventId: string): Promise<Appointment | undefined> {
    const [appointment] = await db
      .select()
      .from(appointments)
      .where(eq(appointments.googleCalendarEventId, googleCalendarEventId));
    return appointment;
  }

  async listAppointments(filters: {
    doctorId?: string;
    patientId?: string;
    startDate?: Date;
    endDate?: Date;
    status?: string;
    limit?: number;
    offset?: number;
  }): Promise<Appointment[]> {
    const conditions = [];
    
    if (filters.doctorId) {
      conditions.push(eq(appointments.doctorId, filters.doctorId));
    }
    if (filters.patientId) {
      conditions.push(eq(appointments.patientId, filters.patientId));
    }
    if (filters.startDate) {
      conditions.push(gte(appointments.startTime, filters.startDate));
    }
    if (filters.endDate) {
      conditions.push(lte(appointments.startTime, filters.endDate));
    }
    if (filters.status) {
      conditions.push(eq(appointments.status, filters.status));
    }

    let query = db
      .select()
      .from(appointments)
      .where(conditions.length > 0 ? and(...conditions) : undefined)
      .orderBy(desc(appointments.startTime));

    if (filters.limit) {
      query = query.limit(filters.limit);
    }
    if (filters.offset) {
      query = query.offset(filters.offset);
    }

    return await query;
  }

  async listUpcomingAppointments(
    userId: string, 
    role: 'doctor' | 'patient', 
    daysAhead: number = 30, 
    limit: number = 50
  ): Promise<Appointment[]> {
    const now = new Date();
    const futureDate = new Date();
    futureDate.setDate(futureDate.getDate() + daysAhead);

    const condition = role === 'doctor' 
      ? eq(appointments.doctorId, userId)
      : eq(appointments.patientId, userId);

    return await db
      .select()
      .from(appointments)
      .where(
        and(
          condition,
          gte(appointments.startTime, now),
          lte(appointments.startTime, futureDate)
        )
      )
      .orderBy(appointments.startTime)
      .limit(limit);
  }

  async updateAppointment(id: string, data: Partial<Appointment>): Promise<Appointment | undefined> {
    const [appointment] = await db
      .update(appointments)
      .set({ ...data, updatedAt: new Date() })
      .where(eq(appointments.id, id))
      .returning();
    return appointment;
  }

  async confirmAppointment(id: string, confirmedAt: Date): Promise<Appointment | undefined> {
    const [appointment] = await db
      .update(appointments)
      .set({
        confirmationStatus: 'confirmed',
        confirmedAt,
        updatedAt: new Date(),
      })
      .where(eq(appointments.id, id))
      .returning();
    return appointment;
  }

  async cancelAppointment(
    id: string, 
    cancelledBy: string, 
    reason: string
  ): Promise<Appointment | undefined> {
    const [appointment] = await db
      .update(appointments)
      .set({
        status: 'cancelled',
        cancelledBy,
        cancellationReason: reason,
        cancelledAt: new Date(),
        updatedAt: new Date(),
      })
      .where(eq(appointments.id, id))
      .returning();
    return appointment;
  }

  async findAppointmentConflicts(
    doctorId: string,
    startTime: Date,
    endTime: Date,
    excludeId?: string
  ): Promise<Appointment[]> {
    const conditions = [
      eq(appointments.doctorId, doctorId),
      or(
        // New appointment starts during existing
        and(
          gte(sql`${appointments.startTime}`, startTime),
          lte(sql`${appointments.startTime}`, endTime)
        ),
        // New appointment ends during existing
        and(
          gte(sql`${appointments.endTime}`, startTime),
          lte(sql`${appointments.endTime}`, endTime)
        ),
        // New appointment completely contains existing
        and(
          lte(sql`${appointments.startTime}`, startTime),
          gte(sql`${appointments.endTime}`, endTime)
        )
      ),
      // Only check non-cancelled appointments
      sql`${appointments.status} != 'cancelled'`,
    ];

    if (excludeId) {
      conditions.push(sql`${appointments.id} != ${excludeId}`);
    }

    return await db
      .select()
      .from(appointments)
      .where(and(...conditions));
  }

  // ============================================================================
  // RECEPTIONIST & ASSISTANT LYSA FEATURES - AVAILABILITY
  // ============================================================================

  async setDoctorAvailability(availabilityData: InsertDoctorAvailability): Promise<DoctorAvailability> {
    const [availability] = await db
      .insert(doctorAvailability)
      .values(availabilityData)
      .returning();
    return availability;
  }

  async getDoctorAvailability(
    doctorId: string,
    dateRange?: { start: Date; end: Date }
  ): Promise<DoctorAvailability[]> {
    const conditions = [eq(doctorAvailability.doctorId, doctorId)];

    if (dateRange) {
      conditions.push(
        or(
          // Recurring availability (no specific date)
          and(
            eq(doctorAvailability.isRecurring, true),
            sql`${doctorAvailability.specificDate} IS NULL`,
            lte(doctorAvailability.validFrom, dateRange.end),
            or(
              sql`${doctorAvailability.validUntil} IS NULL`,
              gte(doctorAvailability.validUntil, dateRange.start)
            )
          ),
          // Specific date overrides
          and(
            sql`${doctorAvailability.specificDate} IS NOT NULL`,
            gte(doctorAvailability.specificDate, dateRange.start),
            lte(doctorAvailability.specificDate, dateRange.end)
          )
        )
      );
    }

    return await db
      .select()
      .from(doctorAvailability)
      .where(and(...conditions))
      .orderBy(doctorAvailability.dayOfWeek, doctorAvailability.startTime);
  }

  async removeDoctorAvailability(id: string): Promise<void> {
    await db
      .delete(doctorAvailability)
      .where(eq(doctorAvailability.id, id));
  }

  // ============================================================================
  // RECEPTIONIST & ASSISTANT LYSA FEATURES - EMAIL
  // ============================================================================

  async createEmailThread(threadData: InsertEmailThread): Promise<EmailThread> {
    const [thread] = await db
      .insert(emailThreads)
      .values(threadData)
      .returning();
    return thread;
  }

  async getEmailThread(id: string): Promise<EmailThread | undefined> {
    const [thread] = await db
      .select()
      .from(emailThreads)
      .where(eq(emailThreads.id, id));
    return thread;
  }

  async listEmailThreads(
    doctorId: string,
    filters?: {
      status?: string;
      category?: string;
      isRead?: boolean;
      patientId?: string;
      limit?: number;
      offset?: number;
    }
  ): Promise<EmailThread[]> {
    const conditions = [eq(emailThreads.doctorId, doctorId)];

    if (filters?.status) {
      conditions.push(eq(emailThreads.status, filters.status));
    }
    if (filters?.category) {
      conditions.push(eq(emailThreads.category, filters.category));
    }
    if (filters?.isRead !== undefined) {
      conditions.push(eq(emailThreads.isRead, filters.isRead));
    }
    if (filters?.patientId) {
      conditions.push(eq(emailThreads.patientId, filters.patientId));
    }

    let query = db
      .select()
      .from(emailThreads)
      .where(and(...conditions))
      .orderBy(desc(emailThreads.lastMessageAt));

    if (filters?.limit) {
      query = query.limit(filters.limit);
    }
    if (filters?.offset) {
      query = query.offset(filters.offset);
    }

    return await query;
  }

  async updateEmailThread(id: string, data: Partial<EmailThread>): Promise<EmailThread | undefined> {
    const [thread] = await db
      .update(emailThreads)
      .set({ ...data, updatedAt: new Date() })
      .where(eq(emailThreads.id, id))
      .returning();
    return thread;
  }

  async markThreadRead(threadId: string): Promise<EmailThread | undefined> {
    const [thread] = await db
      .update(emailThreads)
      .set({ isRead: true, updatedAt: new Date() })
      .where(eq(emailThreads.id, threadId))
      .returning();
    return thread;
  }

  async createEmailMessage(messageData: InsertEmailMessage): Promise<EmailMessage> {
    const [message] = await db
      .insert(emailMessages)
      .values(messageData)
      .returning();

    // Update thread message count and last message time
    await db
      .update(emailThreads)
      .set({
        messageCount: sql`${emailThreads.messageCount} + 1`,
        lastMessageAt: new Date(),
        updatedAt: new Date(),
      })
      .where(eq(emailThreads.id, messageData.threadId));

    return message;
  }

  async getThreadMessages(threadId: string): Promise<EmailMessage[]> {
    return await db
      .select()
      .from(emailMessages)
      .where(eq(emailMessages.threadId, threadId))
      .orderBy(emailMessages.createdAt);
  }

  async markEmailSent(id: string, sentAt: Date): Promise<EmailMessage | undefined> {
    const [message] = await db
      .update(emailMessages)
      .set({ isSent: true, sentAt })
      .where(eq(emailMessages.id, id))
      .returning();
    return message;
  }

  // ============================================================================
  // RECEPTIONIST & ASSISTANT LYSA FEATURES - CALL LOGS
  // ============================================================================

  async createCallLog(logData: InsertCallLog): Promise<CallLog> {
    const [log] = await db
      .insert(callLogs)
      .values(logData)
      .returning();
    return log;
  }

  async getCallLog(id: string): Promise<CallLog | undefined> {
    const [log] = await db
      .select()
      .from(callLogs)
      .where(eq(callLogs.id, id));
    return log;
  }

  async listCallLogs(
    doctorId: string,
    filters?: {
      status?: string;
      requiresFollowup?: boolean;
      patientId?: string;
      limit?: number;
      offset?: number;
    }
  ): Promise<CallLog[]> {
    const conditions = [eq(callLogs.doctorId, doctorId)];

    if (filters?.status) {
      conditions.push(eq(callLogs.status, filters.status));
    }
    if (filters?.requiresFollowup !== undefined) {
      conditions.push(eq(callLogs.requiresFollowup, filters.requiresFollowup));
    }
    if (filters?.patientId) {
      conditions.push(eq(callLogs.patientId, filters.patientId));
    }

    let query = db
      .select()
      .from(callLogs)
      .where(and(...conditions))
      .orderBy(desc(callLogs.startTime));

    if (filters?.limit) {
      query = query.limit(filters.limit);
    }
    if (filters?.offset) {
      query = query.offset(filters.offset);
    }

    return await query;
  }

  async listCallLogsByPatient(patientId: string, limit: number = 50): Promise<CallLog[]> {
    return await db
      .select()
      .from(callLogs)
      .where(eq(callLogs.patientId, patientId))
      .orderBy(desc(callLogs.startTime))
      .limit(limit);
  }

  async updateCallLog(id: string, data: Partial<CallLog>): Promise<CallLog | undefined> {
    const [log] = await db
      .update(callLogs)
      .set({ ...data, updatedAt: new Date() })
      .where(eq(callLogs.id, id))
      .returning();
    return log;
  }

  // ============================================================================
  // RECEPTIONIST & ASSISTANT LYSA FEATURES - REMINDERS
  // ============================================================================

  async createAppointmentReminder(reminderData: InsertAppointmentReminder): Promise<AppointmentReminder> {
    const [reminder] = await db
      .insert(appointmentReminders)
      .values(reminderData)
      .returning();
    return reminder;
  }

  async getAppointmentReminders(appointmentId: string): Promise<AppointmentReminder[]> {
    return await db
      .select()
      .from(appointmentReminders)
      .where(eq(appointmentReminders.appointmentId, appointmentId))
      .orderBy(appointmentReminders.scheduledFor);
  }

  async listDueReminders(beforeTime: Date, limit: number = 100): Promise<AppointmentReminder[]> {
    return await db
      .select()
      .from(appointmentReminders)
      .where(
        and(
          eq(appointmentReminders.status, 'pending'),
          lte(appointmentReminders.scheduledFor, beforeTime)
        )
      )
      .orderBy(appointmentReminders.scheduledFor)
      .limit(limit);
  }

  async markReminderSent(
    id: string,
    sentAt: Date,
    twilioSid?: string,
    sesSid?: string
  ): Promise<AppointmentReminder | undefined> {
    const [reminder] = await db
      .update(appointmentReminders)
      .set({
        status: 'sent',
        sentAt,
        twilioMessageSid: twilioSid || sql`${appointmentReminders.twilioMessageSid}`,
        sesMessageId: sesSid || sql`${appointmentReminders.sesMessageId}`,
      })
      .where(eq(appointmentReminders.id, id))
      .returning();
    return reminder;
  }

  async markReminderFailed(id: string, error: string): Promise<AppointmentReminder | undefined> {
    const [reminder] = await db
      .update(appointmentReminders)
      .set({
        status: 'failed',
        error,
        retryCount: sql`${appointmentReminders.retryCount} + 1`,
      })
      .where(eq(appointmentReminders.id, id))
      .returning();
    return reminder;
  }

  async confirmReminder(id: string): Promise<AppointmentReminder | undefined> {
    const [reminder] = await db
      .update(appointmentReminders)
      .set({
        confirmed: true,
        confirmedAt: new Date(),
      })
      .where(eq(appointmentReminders.id, id))
      .returning();
    return reminder;
  }

  // ============================================================================
  // RECEPTIONIST & ASSISTANT LYSA FEATURES - SYMPTOM TRIAGE
  // ============================================================================

  async updateAppointmentTriageResult(params: {
    appointmentId?: string;
    patientId: string;
    symptoms: string;
    urgencyLevel: 'emergency' | 'urgent' | 'routine' | 'non-urgent';
    triageAssessment: TriageAssessment;
    redFlags: string[];
    recommendations: string[];
    patientSelfAssessment?: string;
    durationMs: number;
  }): Promise<{ appointment?: Appointment; log: AppointmentTriageLog }> {
    return await db.transaction(async (trx) => {
      let appointment: Appointment | undefined;

      // If appointmentId provided, update the appointment
      if (params.appointmentId) {
        // Fetch appointment with row-level lock to prevent concurrent modifications
        const [existingAppt] = await trx
          .select()
          .from(appointments)
          .where(eq(appointments.id, params.appointmentId))
          .for('update');

        if (!existingAppt) {
          throw new Error(`Appointment not found: ${params.appointmentId}`);
        }

        // Verify patient ownership
        if (existingAppt.patientId !== params.patientId) {
          throw new Error(`Patient mismatch: appointment belongs to different patient`);
        }

        // Prevent modification of cancelled appointments
        if (existingAppt.status === 'cancelled') {
          throw new Error(`Cannot triage cancelled appointment`);
        }

        // Check for concurrent triage (if triageAssessedAt is newer than expected)
        if (existingAppt.triageAssessedAt && existingAppt.triageAssessedAt > new Date(Date.now() - params.durationMs)) {
          throw new Error(`Concurrent triage detected: appointment was recently triaged`);
        }

        // Update appointment with triage results
        const [updated] = await trx
          .update(appointments)
          .set({
            symptoms: params.symptoms,
            urgencyLevel: params.urgencyLevel,
            triageAssessment: params.triageAssessment,
            triageAssessedAt: new Date(),
            updatedAt: new Date(),
          })
          .where(eq(appointments.id, params.appointmentId))
          .returning();

        appointment = updated;
      }

      // Always create audit log
      const [log] = await trx
        .insert(appointmentTriageLogs)
        .values({
          appointmentId: params.appointmentId || null,
          patientId: params.patientId,
          symptoms: params.symptoms,
          patientSelfAssessment: params.patientSelfAssessment || null,
          urgencyLevel: params.urgencyLevel,
          triageAssessment: params.triageAssessment,
          redFlags: params.redFlags,
          recommendations: params.recommendations,
          assessmentMethod: params.triageAssessment.assessedBy,
          durationMs: params.durationMs,
          createdAt: new Date(),
        })
        .returning();

      return { appointment, log };
    });
  }

  async createAppointmentTriageLog(logData: InsertAppointmentTriageLog): Promise<AppointmentTriageLog> {
    const [log] = await db
      .insert(appointmentTriageLogs)
      .values(logData)
      .returning();
    return log;
  }

  async getAppointmentTriageLogs(
    patientId: string,
    limit: number = 10
  ): Promise<AppointmentTriageLog[]> {
    return await db
      .select()
      .from(appointmentTriageLogs)
      .where(eq(appointmentTriageLogs.patientId, patientId))
      .orderBy(desc(appointmentTriageLogs.createdAt))
      .limit(limit);
  }

  async updateTriageLogReview(
    id: string,
    reviewData: {
      clinicianAgreed: boolean;
      clinicianOverrideLevel?: string;
      clinicianOverrideReason?: string;
      reviewedBy: string;
    }
  ): Promise<AppointmentTriageLog | undefined> {
    const [log] = await db
      .update(appointmentTriageLogs)
      .set({
        clinicianReviewed: true,
        clinicianAgreed: reviewData.clinicianAgreed,
        clinicianOverrideLevel: reviewData.clinicianOverrideLevel,
        clinicianOverrideReason: reviewData.clinicianOverrideReason,
        reviewedBy: reviewData.reviewedBy,
        reviewedAt: new Date(),
      })
      .where(eq(appointmentTriageLogs.id, id))
      .returning();
    return log;
  }

  // Google Calendar sync operations
  async createGoogleCalendarSync(sync: InsertGoogleCalendarSync): Promise<GoogleCalendarSync> {
    const [created] = await db
      .insert(googleCalendarSync)
      .values(sync)
      .returning();
    return created;
  }

  async getGoogleCalendarSync(doctorId: string): Promise<GoogleCalendarSync | undefined> {
    const [sync] = await db
      .select()
      .from(googleCalendarSync)
      .where(eq(googleCalendarSync.doctorId, doctorId));
    return sync;
  }

  async updateGoogleCalendarSync(
    doctorId: string,
    data: Partial<GoogleCalendarSync>
  ): Promise<GoogleCalendarSync | undefined> {
    const [updated] = await db
      .update(googleCalendarSync)
      .set({ ...data, updatedAt: new Date() })
      .where(eq(googleCalendarSync.doctorId, doctorId))
      .returning();
    return updated;
  }

  async createGoogleCalendarSyncLog(log: InsertGoogleCalendarSyncLog): Promise<GoogleCalendarSyncLog> {
    const [created] = await db
      .insert(googleCalendarSyncLogs)
      .values(log)
      .returning();
    return created;
  }

  async getGoogleCalendarSyncLogs(
    doctorId: string,
    limit: number = 50
  ): Promise<GoogleCalendarSyncLog[]> {
    return await db
      .select()
      .from(googleCalendarSyncLogs)
      .where(eq(googleCalendarSyncLogs.doctorId, doctorId))
      .orderBy(desc(googleCalendarSyncLogs.createdAt))
      .limit(limit);
  }

  async getAppointmentByGoogleEventId(eventId: string): Promise<Appointment | undefined> {
    const [appointment] = await db
      .select()
      .from(appointments)
      .where(eq(appointments.googleCalendarEventId, eventId));
    return appointment;
  }

  // Gmail sync operations
  async createGmailSync(sync: InsertGmailSync): Promise<GmailSync> {
    const [created] = await db
      .insert(gmailSync)
      .values(sync)
      .returning();
    return created;
  }

  async getGmailSync(doctorId: string): Promise<GmailSync | undefined> {
    const [sync] = await db
      .select()
      .from(gmailSync)
      .where(eq(gmailSync.doctorId, doctorId));
    return sync;
  }

  async updateGmailSync(
    doctorId: string,
    data: Partial<GmailSync>
  ): Promise<GmailSync | undefined> {
    const [updated] = await db
      .update(gmailSync)
      .set({ ...data, updatedAt: new Date() })
      .where(eq(gmailSync.doctorId, doctorId))
      .returning();
    return updated;
  }

  async deleteGmailSync(doctorId: string): Promise<boolean> {
    const result = await db
      .delete(gmailSync)
      .where(eq(gmailSync.doctorId, doctorId));
    return true;
  }

  async createGmailSyncLog(log: InsertGmailSyncLog): Promise<GmailSyncLog> {
    const [created] = await db
      .insert(gmailSyncLogs)
      .values(log)
      .returning();
    return created;
  }

  async getGmailSyncLogs(
    doctorId: string,
    limit: number = 50
  ): Promise<GmailSyncLog[]> {
    return await db
      .select()
      .from(gmailSyncLogs)
      .where(eq(gmailSyncLogs.doctorId, doctorId))
      .orderBy(desc(gmailSyncLogs.createdAt))
      .limit(limit);
  }

  async getEmailThreadByExternalId(externalThreadId: string): Promise<EmailThread | undefined> {
    const [thread] = await db
      .select()
      .from(emailThreads)
      .where(eq(emailThreads.externalThreadId, externalThreadId));
    return thread;
  }

  // PainTrack operations
  async createPaintrackSession(session: InsertPaintrackSession): Promise<PaintrackSession> {
    const [created] = await db.insert(paintrackSessions).values(session).returning();
    return created;
  }

  async getPaintrackSessions(userId: string, limit: number = 30): Promise<PaintrackSession[]> {
    return await db
      .select()
      .from(paintrackSessions)
      .where(eq(paintrackSessions.userId, userId))
      .orderBy(desc(paintrackSessions.createdAt))
      .limit(limit);
  }

  async getPaintrackSession(id: string, userId: string): Promise<PaintrackSession | undefined> {
    const [session] = await db
      .select()
      .from(paintrackSessions)
      .where(and(eq(paintrackSessions.id, id), eq(paintrackSessions.userId, userId)));
    return session;
  }

  // Device Readings operations (BP, glucose, scale, thermometer, stethoscope, smartwatch)
  async createDeviceReading(reading: InsertDeviceReading): Promise<DeviceReading> {
    const [created] = await db.insert(deviceReadings).values(reading).returning();
    return created;
  }

  async getDeviceReadings(patientId: string, options?: { 
    deviceType?: string; 
    limit?: number; 
    startDate?: Date; 
    endDate?: Date;
  }): Promise<DeviceReading[]> {
    const conditions = [eq(deviceReadings.patientId, patientId)];
    
    if (options?.deviceType) {
      conditions.push(eq(deviceReadings.deviceType, options.deviceType));
    }
    if (options?.startDate) {
      conditions.push(gte(deviceReadings.recordedAt, options.startDate));
    }
    if (options?.endDate) {
      conditions.push(lte(deviceReadings.recordedAt, options.endDate));
    }

    return await db
      .select()
      .from(deviceReadings)
      .where(and(...conditions))
      .orderBy(desc(deviceReadings.recordedAt))
      .limit(options?.limit || 100);
  }

  async getDeviceReading(id: string): Promise<DeviceReading | undefined> {
    const [reading] = await db
      .select()
      .from(deviceReadings)
      .where(eq(deviceReadings.id, id));
    return reading;
  }

  async getLatestDeviceReading(patientId: string, deviceType: string): Promise<DeviceReading | undefined> {
    const [reading] = await db
      .select()
      .from(deviceReadings)
      .where(and(
        eq(deviceReadings.patientId, patientId),
        eq(deviceReadings.deviceType, deviceType)
      ))
      .orderBy(desc(deviceReadings.recordedAt))
      .limit(1);
    return reading;
  }

  async getDeviceReadingsByType(patientId: string, deviceType: string, limit: number = 50): Promise<DeviceReading[]> {
    return await db
      .select()
      .from(deviceReadings)
      .where(and(
        eq(deviceReadings.patientId, patientId),
        eq(deviceReadings.deviceType, deviceType)
      ))
      .orderBy(desc(deviceReadings.recordedAt))
      .limit(limit);
  }

  async updateDeviceReading(id: string, data: Partial<DeviceReading>): Promise<DeviceReading | undefined> {
    const [updated] = await db
      .update(deviceReadings)
      .set({ ...data, updatedAt: new Date() })
      .where(eq(deviceReadings.id, id))
      .returning();
    return updated;
  }

  async deleteDeviceReading(id: string): Promise<boolean> {
    await db.delete(deviceReadings).where(eq(deviceReadings.id, id));
    return true;
  }

  async getDeviceReadingsForHealthAlerts(patientId: string, hours: number = 24): Promise<DeviceReading[]> {
    const since = new Date(Date.now() - hours * 60 * 60 * 1000);
    return await db
      .select()
      .from(deviceReadings)
      .where(and(
        eq(deviceReadings.patientId, patientId),
        eq(deviceReadings.processedForAlerts, false),
        gte(deviceReadings.recordedAt, since)
      ))
      .orderBy(desc(deviceReadings.recordedAt));
  }

  async markDeviceReadingProcessedForAlerts(id: string, alertIds: string[]): Promise<DeviceReading | undefined> {
    const [updated] = await db
      .update(deviceReadings)
      .set({ 
        processedForAlerts: true, 
        alertsGenerated: alertIds,
        updatedAt: new Date() 
      })
      .where(eq(deviceReadings.id, id))
      .returning();
    return updated;
  }

  // Doctor Integrations operations
  async getDoctorIntegrations(doctorId: string): Promise<DoctorIntegration[]> {
    return await db
      .select()
      .from(doctorIntegrations)
      .where(eq(doctorIntegrations.doctorId, doctorId));
  }

  async getDoctorIntegrationByType(doctorId: string, integrationType: string): Promise<DoctorIntegration | undefined> {
    const [integration] = await db
      .select()
      .from(doctorIntegrations)
      .where(and(
        eq(doctorIntegrations.doctorId, doctorId),
        eq(doctorIntegrations.integrationType, integrationType)
      ));
    return integration;
  }

  async createDoctorIntegration(integration: InsertDoctorIntegration): Promise<DoctorIntegration> {
    const [created] = await db
      .insert(doctorIntegrations)
      .values(integration)
      .returning();
    return created;
  }

  async updateDoctorIntegration(id: string, data: Partial<DoctorIntegration>): Promise<DoctorIntegration | undefined> {
    const [updated] = await db
      .update(doctorIntegrations)
      .set({ ...data, updatedAt: new Date() })
      .where(eq(doctorIntegrations.id, id))
      .returning();
    return updated;
  }

  async deleteDoctorIntegration(id: string): Promise<boolean> {
    await db.delete(doctorIntegrations).where(eq(doctorIntegrations.id, id));
    return true;
  }

  // Doctor Emails operations
  async getDoctorEmails(doctorId: string, options?: { category?: string; isRead?: boolean; limit?: number; offset?: number }): Promise<DoctorEmail[]> {
    const conditions = [eq(doctorEmails.doctorId, doctorId)];
    
    if (options?.category) {
      conditions.push(eq(doctorEmails.aiCategory, options.category));
    }
    if (options?.isRead !== undefined) {
      conditions.push(eq(doctorEmails.isRead, options.isRead));
    }

    return await db
      .select()
      .from(doctorEmails)
      .where(and(...conditions))
      .orderBy(desc(doctorEmails.receivedAt))
      .limit(options?.limit || 50)
      .offset(options?.offset || 0);
  }

  async getDoctorEmailByProviderId(doctorId: string, providerMessageId: string): Promise<DoctorEmail | undefined> {
    const [email] = await db
      .select()
      .from(doctorEmails)
      .where(and(
        eq(doctorEmails.doctorId, doctorId),
        eq(doctorEmails.providerMessageId, providerMessageId)
      ));
    return email;
  }

  async createDoctorEmail(email: InsertDoctorEmail): Promise<DoctorEmail> {
    const [created] = await db
      .insert(doctorEmails)
      .values(email)
      .returning();
    return created;
  }

  async updateDoctorEmail(id: string, data: Partial<DoctorEmail>): Promise<DoctorEmail | undefined> {
    const [updated] = await db
      .update(doctorEmails)
      .set(data)
      .where(eq(doctorEmails.id, id))
      .returning();
    return updated;
  }

  // Doctor WhatsApp operations
  async getDoctorWhatsappMessages(doctorId: string, options?: { status?: string; limit?: number }): Promise<DoctorWhatsappMessage[]> {
    const conditions = [eq(doctorWhatsappMessages.doctorId, doctorId)];
    
    if (options?.status) {
      conditions.push(eq(doctorWhatsappMessages.status, options.status));
    }

    return await db
      .select()
      .from(doctorWhatsappMessages)
      .where(and(...conditions))
      .orderBy(desc(doctorWhatsappMessages.receivedAt))
      .limit(options?.limit || 50);
  }

  async createDoctorWhatsappMessage(message: InsertDoctorWhatsappMessage): Promise<DoctorWhatsappMessage> {
    const [created] = await db
      .insert(doctorWhatsappMessages)
      .values(message)
      .returning();
    return created;
  }

  async updateDoctorWhatsappMessage(id: string, data: Partial<DoctorWhatsappMessage>): Promise<DoctorWhatsappMessage | undefined> {
    const [updated] = await db
      .update(doctorWhatsappMessages)
      .set(data)
      .where(eq(doctorWhatsappMessages.id, id))
      .returning();
    return updated;
  }

  // Call log operations (enhanced)
  async getCallLogs(doctorId: string, options?: { status?: string; limit?: number }): Promise<CallLog[]> {
    const conditions = [eq(callLogs.doctorId, doctorId)];
    
    if (options?.status) {
      conditions.push(eq(callLogs.status, options.status));
    }

    return await db
      .select()
      .from(callLogs)
      .where(and(...conditions))
      .orderBy(desc(callLogs.startTime))
      .limit(options?.limit || 50);
  }

  async getCallLogByTwilioSid(twilioCallSid: string): Promise<CallLog | undefined> {
    const [log] = await db
      .select()
      .from(callLogs)
      .where(eq(callLogs.twilioCallSid, twilioCallSid));
    return log;
  }

  async createCallLog(log: InsertCallLog): Promise<CallLog> {
    const [created] = await db
      .insert(callLogs)
      .values(log)
      .returning();
    return created;
  }

  async updateCallLog(id: string, data: Partial<CallLog>): Promise<CallLog | undefined> {
    const [updated] = await db
      .update(callLogs)
      .set({ ...data, updatedAt: new Date() })
      .where(eq(callLogs.id, id))
      .returning();
    return updated;
  }

  // Utility
  async getUserByPhone(phone: string): Promise<User | undefined> {
    const normalizedPhone = phone.replace(/\D/g, '');
    const [user] = await db
      .select()
      .from(users)
      .where(sql`REPLACE(${users.phoneNumber}, '+', '') LIKE '%' || ${normalizedPhone} || '%'`);
    return user;
  }
}

export const storage = new DatabaseStorage();
