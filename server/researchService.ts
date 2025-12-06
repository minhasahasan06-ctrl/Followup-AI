import { db } from "./db";
import { eq, and, desc, sql, gte, lte, inArray, or } from "drizzle-orm";
import {
  researchDataConsent,
  researchProjects,
  researchStudies,
  researchCohorts,
  studyEnrollments,
  researchVisits,
  researchMeasurements,
  researchImmuneMarkers,
  researchLocations,
  researchPatientLocations,
  researchEnvironmentalExposures,
  researchDataSnapshots,
  researchAnalysisReports,
  researchAlerts,
  dailyFollowupTemplates,
  dailyFollowupAssignments,
  dailyFollowupResponses,
  researchAuditLogs,
  analysisJobs,
  users,
  patientProfiles,
  dailyFollowups,
  medications,
  symptomCheckins,
  paintrackSessions,
  sessionMetrics,
  immuneBiomarkers,
  digitalBiomarkers,
  deteriorationPredictions,
  interactionAlerts,
  medicalDocuments,
  autoJournals,
  type ResearchDataConsent,
  type InsertResearchDataConsent,
  type ResearchProject,
  type InsertResearchProject,
  type ResearchStudy,
  type InsertResearchStudy,
  type ResearchCohort,
  type InsertResearchCohort,
  type StudyEnrollment,
  type InsertStudyEnrollment,
  type ResearchVisit,
  type InsertResearchVisit,
  type ResearchMeasurement,
  type InsertResearchMeasurement,
  type ResearchImmuneMarker,
  type InsertResearchImmuneMarker,
  type ResearchLocation,
  type InsertResearchLocation,
  type ResearchPatientLocation,
  type InsertResearchPatientLocation,
  type ResearchEnvironmentalExposure,
  type InsertResearchEnvironmentalExposure,
  type ResearchDataSnapshot,
  type InsertResearchDataSnapshot,
  type ResearchAnalysisReport,
  type InsertResearchAnalysisReport,
  type ResearchAlert,
  type InsertResearchAlert,
  type DailyFollowupTemplate,
  type InsertDailyFollowupTemplate,
  type DailyFollowupAssignment,
  type InsertDailyFollowupAssignment,
  type DailyFollowupResponse,
  type InsertDailyFollowupResponse,
  type ResearchAuditLog,
  type InsertResearchAuditLog,
  type AnalysisJob,
  type InsertAnalysisJob,
} from "@shared/schema";
import type { Storage } from './storage';

export interface ResearchMetrics {
  totalPatients: number;
  consentedPatients: number;
  activeStudies: number;
  activeCohorts: number;
  reportsGenerated: number;
  dataCompleteness: number;
}

interface AggregatedData {
  total: number;
  byCategory: Record<string, number>;
  trends: any[];
}

// Exported type for request-scoped audit context (thread-safe)
export interface AuditContext {
  userId: string;
  ipAddress?: string;
  userAgent?: string;
}

class ResearchService {
  private storage?: Storage;

  constructor(storage?: Storage) {
    this.storage = storage;
  }

  setStorage(storage: Storage) {
    this.storage = storage;
  }

  // HIPAA Audit logging helper - logs all research data access and mutations
  // Context is passed per-request to ensure thread-safety with concurrent requests
  private async logAudit(
    context: AuditContext | undefined,
    actionType: string,
    objectType: string,
    objectId: string,
    details?: Record<string, any>
  ): Promise<void> {
    try {
      const auditUserId = context?.userId || 'system';
      await db.insert(researchAuditLogs).values({
        userId: auditUserId,
        actionType,
        objectType,
        objectId,
        details: details || {},
        ipAddress: context?.ipAddress,
        userAgent: context?.userAgent,
      });
      console.log(`[HIPAA-AUDIT] Research: ${actionType} on ${objectType}:${objectId} by ${auditUserId}`);
    } catch (error) {
      console.error('[HIPAA-AUDIT] Failed to log research audit:', error);
    }
  }

  // Consent enforcement helper - returns only consented patient IDs
  private async getConsentedPatientIds(dataType?: string): Promise<string[]> {
    const consents = await db
      .select()
      .from(researchDataConsent)
      .where(eq(researchDataConsent.consentEnabled, true));
    
    if (!dataType) {
      return consents.map(c => c.patientId);
    }
    
    return consents
      .filter(c => {
        const permissions = c.dataTypePermissions as Record<string, boolean> | null;
        return permissions?.[dataType] === true;
      })
      .map(c => c.patientId);
  }

  // Verify patient has active research consent
  private async hasConsent(patientId: string, dataType?: string): Promise<boolean> {
    const [consent] = await db
      .select()
      .from(researchDataConsent)
      .where(
        and(
          eq(researchDataConsent.patientId, patientId),
          eq(researchDataConsent.consentEnabled, true)
        )
      );
    
    if (!consent) return false;
    if (!dataType) return true;
    
    const permissions = consent.dataTypePermissions as Record<string, boolean> | null;
    return permissions?.[dataType] === true;
  }

  // =========================================================================
  // LEGACY METHODS (from original implementation)
  // =========================================================================

  async queryFHIRData(query: { resourceType: string; parameters: Record<string, string>; limit?: number }): Promise<any> {
    console.warn('AWS HealthLake integration requires proper AWS SDK setup. Returning mock data.');
    return {
      resourceType: 'Bundle',
      type: 'searchset',
      total: 0,
      entry: [],
      message: 'AWS HealthLake integration available - requires AWS_HEALTHLAKE_DATASTORE_ID configuration',
    };
  }

  async getEpidemiologicalData(condition: string, dateRange?: { start: Date; end: Date }): Promise<AggregatedData> {
    const appointments = await (this.storage as any)?.getAppointmentsByCondition?.(condition) || [];
    
    const aggregated: AggregatedData = {
      total: appointments.length,
      byCategory: {},
      trends: [],
    };

    const byMonth: Record<string, number> = {};

    for (const appointment of appointments) {
      const month = new Date(appointment.startTime).toISOString().slice(0, 7);
      byMonth[month] = (byMonth[month] || 0) + 1;

      const category = appointment.appointmentType || 'unknown';
      aggregated.byCategory[category] = (aggregated.byCategory[category] || 0) + 1;
    }

    aggregated.trends = Object.entries(byMonth)
      .map(([month, count]) => ({ month, count }))
      .sort((a, b) => a.month.localeCompare(b.month));

    return aggregated;
  }

  async getPopulationHealthMetrics(doctorId?: string): Promise<any> {
    const patients = doctorId
      ? await (this.storage as any)?.getPatientsByDoctor?.(doctorId) || []
      : await this.storage?.getAllPatients() || [];

    const metrics = {
      totalPatients: patients.length,
      demographics: {
        ageGroups: {} as Record<string, number>,
        genderDistribution: {} as Record<string, number>,
      },
      healthConditions: {} as Record<string, number>,
      medicationUsage: {} as Record<string, number>,
      appointmentTrends: [] as any[],
    };

    for (const patient of patients) {
      if (patient.profile?.dateOfBirth) {
        const age = this.calculateAge(new Date(patient.profile.dateOfBirth));
        const ageGroup = this.getAgeGroup(age);
        metrics.demographics.ageGroups[ageGroup] = (metrics.demographics.ageGroups[ageGroup] || 0) + 1;
      }

      if (patient.profile?.gender) {
        const gender = patient.profile.gender;
        metrics.demographics.genderDistribution[gender] = (metrics.demographics.genderDistribution[gender] || 0) + 1;
      }
    }

    return metrics;
  }

  async generateResearchReport(studyType: string, parameters: Record<string, any>): Promise<any> {
    const report = {
      studyType,
      generatedAt: new Date(),
      parameters,
      findings: {},
      visualizations: [],
      recommendations: [] as string[],
    };

    switch (studyType) {
      case 'disease_prevalence':
        report.findings = await this.getEpidemiologicalData(parameters.condition);
        report.recommendations.push(
          'Increase screening programs for at-risk populations',
          'Implement preventive care initiatives',
          'Enhance patient education on early symptoms'
        );
        break;

      case 'medication_efficacy':
        report.findings = await this.analyzeMedicationEffectiveness(parameters.medicationName);
        break;

      case 'population_health':
        report.findings = await this.getPopulationHealthMetrics(parameters.doctorId);
        break;

      default:
        throw new Error(`Unsupported study type: ${studyType}`);
    }

    return report;
  }

  private async analyzeMedicationEffectiveness(medicationName: string): Promise<any> {
    return {
      medication: medicationName,
      patientsOnMedication: 0,
      averageAdherenceRate: 0,
      reportedSideEffects: [],
      efficacyScore: 0,
    };
  }

  private calculateAge(dateOfBirth: Date): number {
    const today = new Date();
    let age = today.getFullYear() - dateOfBirth.getFullYear();
    const monthDiff = today.getMonth() - dateOfBirth.getMonth();

    if (monthDiff < 0 || (monthDiff === 0 && today.getDate() < dateOfBirth.getDate())) {
      age--;
    }

    return age;
  }

  private getAgeGroup(age: number): string {
    if (age < 18) return '0-17';
    if (age < 35) return '18-34';
    if (age < 50) return '35-49';
    if (age < 65) return '50-64';
    return '65+';
  }

  // =========================================================================
  // RESEARCH DATA CONSENT OPERATIONS (with HIPAA audit logging)
  // =========================================================================
  
  async getResearchDataConsent(patientId: string): Promise<ResearchDataConsent | undefined> {
    const [consent] = await db
      .select()
      .from(researchDataConsent)
      .where(eq(researchDataConsent.patientId, patientId));
    return consent;
  }

  async createResearchDataConsent(consent: InsertResearchDataConsent, auditContext?: AuditContext): Promise<ResearchDataConsent> {
    const [created] = await db.insert(researchDataConsent).values(consent).returning();
    await this.logAudit(auditContext, 'CREATE', 'ResearchDataConsent', created.id, {
      patientId: consent.patientId,
      consentEnabled: consent.consentEnabled,
    });
    return created;
  }

  async updateResearchDataConsent(
    patientId: string, 
    data: Partial<InsertResearchDataConsent>,
    auditContext?: AuditContext
  ): Promise<ResearchDataConsent | undefined> {
    const [updated] = await db
      .update(researchDataConsent)
      .set({ ...data, updatedAt: new Date(), consentUpdatedAt: new Date() })
      .where(eq(researchDataConsent.patientId, patientId))
      .returning();
    if (updated) {
      await this.logAudit(auditContext, 'UPDATE', 'ResearchDataConsent', updated.id, {
        patientId,
        consentEnabled: data.consentEnabled,
        dataTypePermissions: data.dataTypePermissions,
      });
    }
    return updated;
  }

  async upsertResearchDataConsent(consent: InsertResearchDataConsent, auditContext?: AuditContext): Promise<ResearchDataConsent> {
    const existing = await this.getResearchDataConsent(consent.patientId);
    if (existing) {
      const updated = await this.updateResearchDataConsent(consent.patientId, consent, auditContext);
      return updated!;
    }
    return this.createResearchDataConsent(consent, auditContext);
  }

  async getConsentedPatients(): Promise<ResearchDataConsent[]> {
    return db
      .select()
      .from(researchDataConsent)
      .where(eq(researchDataConsent.consentEnabled, true));
  }

  async getConsentedPatientsForDataType(dataType: string): Promise<string[]> {
    const consents = await db
      .select()
      .from(researchDataConsent)
      .where(eq(researchDataConsent.consentEnabled, true));
    
    return consents
      .filter(c => {
        const permissions = c.dataTypePermissions as Record<string, boolean> | null;
        return permissions?.[dataType] === true;
      })
      .map(c => c.patientId);
  }

  // =========================================================================
  // RESEARCH PROJECTS OPERATIONS (Personal Research Mode)
  // =========================================================================

  async getResearchProjects(ownerId: string): Promise<ResearchProject[]> {
    return db
      .select()
      .from(researchProjects)
      .where(eq(researchProjects.ownerId, ownerId))
      .orderBy(desc(researchProjects.createdAt));
  }

  async getResearchProject(id: string): Promise<ResearchProject | undefined> {
    const [project] = await db
      .select()
      .from(researchProjects)
      .where(eq(researchProjects.id, id));
    return project;
  }

  async createResearchProject(project: InsertResearchProject, auditContext?: AuditContext): Promise<ResearchProject> {
    const [created] = await db.insert(researchProjects).values(project).returning();
    await this.logAudit(auditContext, 'CREATE', 'ResearchProject', created.id, {
      name: project.name,
      ownerId: project.ownerId,
    });
    return created;
  }

  async updateResearchProject(id: string, data: Partial<InsertResearchProject>, auditContext?: AuditContext): Promise<ResearchProject | undefined> {
    const [updated] = await db
      .update(researchProjects)
      .set({ ...data, updatedAt: new Date() })
      .where(eq(researchProjects.id, id))
      .returning();
    if (updated) {
      await this.logAudit(auditContext, 'UPDATE', 'ResearchProject', id, { changedFields: Object.keys(data) });
    }
    return updated;
  }

  async deleteResearchProject(id: string, auditContext?: AuditContext): Promise<boolean> {
    await this.logAudit(auditContext, 'DELETE', 'ResearchProject', id, {});
    await db.delete(researchProjects).where(eq(researchProjects.id, id));
    return true;
  }

  // =========================================================================
  // RESEARCH STUDIES OPERATIONS
  // =========================================================================

  async getResearchStudies(filters?: {
    projectId?: string;
    ownerId?: string;
    status?: string;
    limit?: number;
  }): Promise<ResearchStudy[]> {
    let query = db.select().from(researchStudies).$dynamic();
    
    const conditions = [];
    if (filters?.projectId) {
      conditions.push(eq(researchStudies.projectId, filters.projectId));
    }
    if (filters?.ownerId) {
      conditions.push(eq(researchStudies.ownerUserId, filters.ownerId));
    }
    if (filters?.status) {
      conditions.push(eq(researchStudies.status, filters.status));
    }
    
    if (conditions.length > 0) {
      query = query.where(and(...conditions));
    }
    
    query = query.orderBy(desc(researchStudies.createdAt));
    
    if (filters?.limit) {
      query = query.limit(filters.limit);
    }
    
    return query;
  }

  async getResearchStudy(id: string): Promise<ResearchStudy | undefined> {
    const [study] = await db
      .select()
      .from(researchStudies)
      .where(eq(researchStudies.id, id));
    return study;
  }

  async createResearchStudy(study: InsertResearchStudy, auditContext?: AuditContext): Promise<ResearchStudy> {
    const [created] = await db.insert(researchStudies).values(study).returning();
    await this.logAudit(auditContext, 'CREATE', 'ResearchStudy', created.id, {
      title: study.title,
      ownerId: study.ownerUserId,
      projectId: study.projectId,
    });
    return created;
  }

  async updateResearchStudy(id: string, data: Partial<InsertResearchStudy>, auditContext?: AuditContext): Promise<ResearchStudy | undefined> {
    const [updated] = await db
      .update(researchStudies)
      .set({ ...data, updatedAt: new Date() })
      .where(eq(researchStudies.id, id))
      .returning();
    if (updated) {
      await this.logAudit(auditContext, 'UPDATE', 'ResearchStudy', id, { changedFields: Object.keys(data), status: data.status });
    }
    return updated;
  }

  async getActiveStudies(): Promise<ResearchStudy[]> {
    return db
      .select()
      .from(researchStudies)
      .where(
        or(
          eq(researchStudies.status, "enrolling"),
          eq(researchStudies.status, "follow_up"),
          eq(researchStudies.status, "analysis")
        )
      )
      .orderBy(desc(researchStudies.createdAt));
  }

  // =========================================================================
  // RESEARCH COHORTS OPERATIONS
  // =========================================================================

  async getResearchCohorts(filters?: {
    projectId?: string;
    createdBy?: string;
    status?: string;
  }): Promise<ResearchCohort[]> {
    let query = db.select().from(researchCohorts).$dynamic();
    
    const conditions = [];
    if (filters?.projectId) {
      conditions.push(eq(researchCohorts.projectId, filters.projectId));
    }
    if (filters?.createdBy) {
      conditions.push(eq(researchCohorts.createdBy, filters.createdBy));
    }
    if (filters?.status) {
      conditions.push(eq(researchCohorts.status, filters.status));
    }
    
    if (conditions.length > 0) {
      query = query.where(and(...conditions));
    }
    
    return query.orderBy(desc(researchCohorts.createdAt));
  }

  async getResearchCohort(id: string): Promise<ResearchCohort | undefined> {
    const [cohort] = await db
      .select()
      .from(researchCohorts)
      .where(eq(researchCohorts.id, id));
    return cohort;
  }

  async createResearchCohort(cohort: InsertResearchCohort, auditContext?: AuditContext): Promise<ResearchCohort> {
    const [created] = await db.insert(researchCohorts).values(cohort).returning();
    await this.logAudit(auditContext, 'CREATE', 'ResearchCohort', created.id, {
      name: cohort.name,
      createdBy: cohort.createdBy,
      projectId: cohort.projectId,
    });
    return created;
  }

  async updateResearchCohort(id: string, data: Partial<InsertResearchCohort>, auditContext?: AuditContext): Promise<ResearchCohort | undefined> {
    const [updated] = await db
      .update(researchCohorts)
      .set({ ...data, updatedAt: new Date() })
      .where(eq(researchCohorts.id, id))
      .returning();
    if (updated) {
      await this.logAudit(auditContext, 'UPDATE', 'ResearchCohort', id, { changedFields: Object.keys(data), status: data.status });
    }
    return updated;
  }

  async previewCohort(definition: any): Promise<{
    patientCount: number;
    stats: {
      meanAge: number;
      genderDistribution: Record<string, number>;
      conditionDistribution: Record<string, number>;
    };
  }> {
    const consentedPatients = await this.getConsentedPatients();
    const consentedIds = consentedPatients.map(c => c.patientId);
    
    if (consentedIds.length === 0) {
      return {
        patientCount: 0,
        stats: { meanAge: 0, genderDistribution: {}, conditionDistribution: {} }
      };
    }
    
    const profiles = await db
      .select()
      .from(patientProfiles)
      .where(inArray(patientProfiles.userId, consentedIds));
    
    let filteredProfiles = profiles;
    
    if (definition?.demographics?.minAge || definition?.demographics?.maxAge) {
      const now = new Date();
      filteredProfiles = filteredProfiles.filter(p => {
        if (!p.dateOfBirth) return true;
        const age = Math.floor((now.getTime() - new Date(p.dateOfBirth).getTime()) / (365.25 * 24 * 60 * 60 * 1000));
        if (definition.demographics?.minAge && age < definition.demographics.minAge) return false;
        if (definition.demographics?.maxAge && age > definition.demographics.maxAge) return false;
        return true;
      });
    }
    
    if (definition?.conditions?.include && definition.conditions.include.length > 0) {
      filteredProfiles = filteredProfiles.filter(p => {
        const condition = p.immunocompromisedCondition?.toLowerCase() || '';
        return definition.conditions!.include!.some((c: string) => condition.includes(c.toLowerCase()));
      });
    }
    
    const now = new Date();
    const ages = filteredProfiles
      .filter(p => p.dateOfBirth)
      .map(p => Math.floor((now.getTime() - new Date(p.dateOfBirth!).getTime()) / (365.25 * 24 * 60 * 60 * 1000)));
    
    const meanAge = ages.length > 0 ? ages.reduce((a, b) => a + b, 0) / ages.length : 0;
    
    const conditionDistribution: Record<string, number> = {};
    filteredProfiles.forEach(p => {
      const condition = p.immunocompromisedCondition || 'Unknown';
      conditionDistribution[condition] = (conditionDistribution[condition] || 0) + 1;
    });
    
    return {
      patientCount: filteredProfiles.length,
      stats: {
        meanAge: Math.round(meanAge * 10) / 10,
        genderDistribution: { 'Unknown': filteredProfiles.length },
        conditionDistribution,
      }
    };
  }

  // =========================================================================
  // STUDY ENROLLMENTS OPERATIONS
  // =========================================================================

  async getStudyEnrollments(studyId: string): Promise<StudyEnrollment[]> {
    return db
      .select()
      .from(studyEnrollments)
      .where(eq(studyEnrollments.studyId, studyId))
      .orderBy(desc(studyEnrollments.enrollmentDate));
  }

  async getPatientEnrollments(patientId: string): Promise<StudyEnrollment[]> {
    return db
      .select()
      .from(studyEnrollments)
      .where(eq(studyEnrollments.patientId, patientId))
      .orderBy(desc(studyEnrollments.enrollmentDate));
  }

  async createStudyEnrollment(enrollment: InsertStudyEnrollment, auditContext?: AuditContext): Promise<StudyEnrollment> {
    const [created] = await db.insert(studyEnrollments).values(enrollment).returning();
    
    await db
      .update(researchStudies)
      .set({ 
        currentEnrollment: sql`${researchStudies.currentEnrollment} + 1`,
        updatedAt: new Date()
      })
      .where(eq(researchStudies.id, enrollment.studyId));
    
    await this.logAudit(auditContext, 'CREATE', 'StudyEnrollment', created.id, {
      studyId: enrollment.studyId,
      patientId: enrollment.patientId,
    });
    return created;
  }

  async updateStudyEnrollment(id: string, data: Partial<InsertStudyEnrollment>, auditContext?: AuditContext): Promise<StudyEnrollment | undefined> {
    const [updated] = await db
      .update(studyEnrollments)
      .set({ ...data, updatedAt: new Date() })
      .where(eq(studyEnrollments.id, id))
      .returning();
    if (updated) {
      await this.logAudit(auditContext, 'UPDATE', 'StudyEnrollment', id, { changedFields: Object.keys(data), status: data.status });
    }
    return updated;
  }

  async withdrawFromStudy(id: string, reason: string, auditContext?: AuditContext): Promise<StudyEnrollment | undefined> {
    const [enrollment] = await db
      .select()
      .from(studyEnrollments)
      .where(eq(studyEnrollments.id, id));
    
    if (!enrollment) return undefined;
    
    const [updated] = await db
      .update(studyEnrollments)
      .set({
        status: "withdrawn",
        consentStatus: "withdrawn",
        withdrawalDate: new Date(),
        withdrawalReason: reason,
        updatedAt: new Date()
      })
      .where(eq(studyEnrollments.id, id))
      .returning();
    
    await db
      .update(researchStudies)
      .set({ 
        currentEnrollment: sql`GREATEST(0, ${researchStudies.currentEnrollment} - 1)`,
        updatedAt: new Date()
      })
      .where(eq(researchStudies.id, enrollment.studyId));
    
    await this.logAudit(auditContext, 'WITHDRAW', 'StudyEnrollment', id, { 
      patientId: enrollment.patientId, 
      studyId: enrollment.studyId, 
      reason 
    });
    return updated;
  }

  // =========================================================================
  // RESEARCH VISITS OPERATIONS
  // =========================================================================

  async getResearchVisits(filters: {
    studyId?: string;
    patientId?: string;
    enrollmentId?: string;
    status?: string;
  }): Promise<ResearchVisit[]> {
    let query = db.select().from(researchVisits).$dynamic();
    
    const conditions = [];
    if (filters.studyId) conditions.push(eq(researchVisits.studyId, filters.studyId));
    if (filters.patientId) conditions.push(eq(researchVisits.patientId, filters.patientId));
    if (filters.enrollmentId) conditions.push(eq(researchVisits.enrollmentId, filters.enrollmentId));
    if (filters.status) conditions.push(eq(researchVisits.visitStatus, filters.status));
    
    if (conditions.length > 0) query = query.where(and(...conditions));
    
    return query.orderBy(researchVisits.scheduledDate);
  }

  async createResearchVisit(visit: InsertResearchVisit, auditContext?: AuditContext): Promise<ResearchVisit> {
    const [created] = await db.insert(researchVisits).values(visit).returning();
    await this.logAudit(auditContext, 'CREATE', 'ResearchVisit', created.id, {
      patientId: visit.patientId,
      studyId: visit.studyId,
      visitType: visit.visitType,
    });
    return created;
  }

  async updateResearchVisit(id: string, data: Partial<InsertResearchVisit>, auditContext?: AuditContext): Promise<ResearchVisit | undefined> {
    const [updated] = await db
      .update(researchVisits)
      .set({ ...data, updatedAt: new Date() })
      .where(eq(researchVisits.id, id))
      .returning();
    if (updated) {
      await this.logAudit(auditContext, 'UPDATE', 'ResearchVisit', id, { changedFields: Object.keys(data), status: data.visitStatus });
    }
    return updated;
  }

  // =========================================================================
  // RESEARCH MEASUREMENTS OPERATIONS
  // =========================================================================

  async getResearchMeasurements(filters: {
    patientId?: string;
    studyId?: string;
    visitId?: string;
    name?: string;
    category?: string;
    limit?: number;
  }): Promise<ResearchMeasurement[]> {
    let query = db.select().from(researchMeasurements).$dynamic();
    
    const conditions = [];
    if (filters.patientId) conditions.push(eq(researchMeasurements.patientId, filters.patientId));
    if (filters.studyId) conditions.push(eq(researchMeasurements.studyId, filters.studyId));
    if (filters.visitId) conditions.push(eq(researchMeasurements.visitId, filters.visitId));
    if (filters.name) conditions.push(eq(researchMeasurements.name, filters.name));
    if (filters.category) conditions.push(eq(researchMeasurements.category, filters.category));
    
    if (conditions.length > 0) query = query.where(and(...conditions));
    
    query = query.orderBy(desc(researchMeasurements.recordedAt));
    
    if (filters.limit) query = query.limit(filters.limit);
    
    return query;
  }

  async createResearchMeasurement(measurement: InsertResearchMeasurement, auditContext?: AuditContext): Promise<ResearchMeasurement> {
    const [created] = await db.insert(researchMeasurements).values(measurement).returning();
    await this.logAudit(auditContext, 'CREATE', 'ResearchMeasurement', created.id, {
      patientId: measurement.patientId,
      name: measurement.name,
      category: measurement.category,
    });
    return created;
  }

  async createResearchMeasurements(measurements: InsertResearchMeasurement[], auditContext?: AuditContext): Promise<ResearchMeasurement[]> {
    if (measurements.length === 0) return [];
    const created = await db.insert(researchMeasurements).values(measurements).returning();
    await this.logAudit(auditContext, 'BULK_CREATE', 'ResearchMeasurement', 'bulk', {
      count: created.length,
      patientIds: [...new Set(measurements.map(m => m.patientId))],
    });
    return created;
  }

  // =========================================================================
  // IMMUNE MARKERS OPERATIONS
  // =========================================================================

  async getResearchImmuneMarkers(filters: {
    patientId?: string;
    studyId?: string;
    markerName?: string;
    startDate?: Date;
    endDate?: Date;
    limit?: number;
  }): Promise<ResearchImmuneMarker[]> {
    let query = db.select().from(researchImmuneMarkers).$dynamic();
    
    const conditions = [];
    if (filters.patientId) conditions.push(eq(researchImmuneMarkers.patientId, filters.patientId));
    if (filters.studyId) conditions.push(eq(researchImmuneMarkers.studyId, filters.studyId));
    if (filters.markerName) conditions.push(eq(researchImmuneMarkers.markerName, filters.markerName));
    if (filters.startDate) conditions.push(gte(researchImmuneMarkers.collectionTime, filters.startDate));
    if (filters.endDate) conditions.push(lte(researchImmuneMarkers.collectionTime, filters.endDate));
    
    if (conditions.length > 0) query = query.where(and(...conditions));
    
    query = query.orderBy(desc(researchImmuneMarkers.collectionTime));
    
    if (filters.limit) query = query.limit(filters.limit);
    
    return query;
  }

  async createResearchImmuneMarker(marker: InsertResearchImmuneMarker, auditContext?: AuditContext): Promise<ResearchImmuneMarker> {
    const [created] = await db.insert(researchImmuneMarkers).values(marker).returning();
    await this.logAudit(auditContext, 'CREATE', 'ResearchImmuneMarker', created.id, {
      patientId: marker.patientId,
      markerName: marker.markerName,
    });
    return created;
  }

  // =========================================================================
  // ENVIRONMENTAL EXPOSURES OPERATIONS
  // =========================================================================

  async getResearchEnvironmentalExposures(filters: {
    locationId?: string;
    startDate?: Date;
    endDate?: Date;
    limit?: number;
  }): Promise<ResearchEnvironmentalExposure[]> {
    let query = db.select().from(researchEnvironmentalExposures).$dynamic();
    
    const conditions = [];
    if (filters.locationId) conditions.push(eq(researchEnvironmentalExposures.locationId, filters.locationId));
    if (filters.startDate) conditions.push(gte(researchEnvironmentalExposures.date, filters.startDate));
    if (filters.endDate) conditions.push(lte(researchEnvironmentalExposures.date, filters.endDate));
    
    if (conditions.length > 0) query = query.where(and(...conditions));
    
    query = query.orderBy(desc(researchEnvironmentalExposures.date));
    
    if (filters.limit) query = query.limit(filters.limit);
    
    return query;
  }

  async createResearchEnvironmentalExposure(exposure: InsertResearchEnvironmentalExposure): Promise<ResearchEnvironmentalExposure> {
    const [created] = await db.insert(researchEnvironmentalExposures).values(exposure).returning();
    return created;
  }

  // =========================================================================
  // RESEARCH LOCATIONS OPERATIONS
  // =========================================================================

  async getResearchLocations(): Promise<ResearchLocation[]> {
    return db.select().from(researchLocations);
  }

  async getResearchLocationByZip(zipCode: string): Promise<ResearchLocation | undefined> {
    const [location] = await db
      .select()
      .from(researchLocations)
      .where(eq(researchLocations.zipCode, zipCode));
    return location;
  }

  async createResearchLocation(location: InsertResearchLocation): Promise<ResearchLocation> {
    const [created] = await db.insert(researchLocations).values(location).returning();
    return created;
  }

  // =========================================================================
  // DATA SNAPSHOTS OPERATIONS
  // =========================================================================

  async createDataSnapshot(snapshot: InsertResearchDataSnapshot): Promise<ResearchDataSnapshot> {
    const [created] = await db.insert(researchDataSnapshots).values(snapshot).returning();
    return created;
  }

  async getDataSnapshots(studyId?: string): Promise<ResearchDataSnapshot[]> {
    if (studyId) {
      return db
        .select()
        .from(researchDataSnapshots)
        .where(eq(researchDataSnapshots.studyId, studyId))
        .orderBy(desc(researchDataSnapshots.createdAt));
    }
    return db.select().from(researchDataSnapshots).orderBy(desc(researchDataSnapshots.createdAt));
  }

  // =========================================================================
  // RESEARCH ANALYSIS REPORTS OPERATIONS
  // =========================================================================

  async getResearchAnalysisReports(filters?: {
    studyId?: string;
    cohortId?: string;
    createdBy?: string;
    analysisType?: string;
    status?: string;
    limit?: number;
  }): Promise<ResearchAnalysisReport[]> {
    let query = db.select().from(researchAnalysisReports).$dynamic();
    
    const conditions = [];
    if (filters?.studyId) conditions.push(eq(researchAnalysisReports.studyId, filters.studyId));
    if (filters?.cohortId) conditions.push(eq(researchAnalysisReports.cohortId, filters.cohortId));
    if (filters?.createdBy) conditions.push(eq(researchAnalysisReports.createdBy, filters.createdBy));
    if (filters?.analysisType) conditions.push(eq(researchAnalysisReports.analysisType, filters.analysisType));
    if (filters?.status) conditions.push(eq(researchAnalysisReports.status, filters.status));
    
    if (conditions.length > 0) query = query.where(and(...conditions));
    
    query = query.orderBy(desc(researchAnalysisReports.createdAt));
    
    if (filters?.limit) query = query.limit(filters.limit);
    
    return query;
  }

  async getResearchAnalysisReport(id: string): Promise<ResearchAnalysisReport | undefined> {
    const [report] = await db
      .select()
      .from(researchAnalysisReports)
      .where(eq(researchAnalysisReports.id, id));
    return report;
  }

  async createResearchAnalysisReport(report: InsertResearchAnalysisReport, auditContext?: AuditContext): Promise<ResearchAnalysisReport> {
    const [created] = await db.insert(researchAnalysisReports).values(report).returning();
    await this.logAudit(auditContext, 'CREATE', 'ResearchAnalysisReport', created.id, {
      title: report.title,
      analysisType: report.analysisType,
      studyId: report.studyId,
    });
    return created;
  }

  async updateResearchAnalysisReport(id: string, data: Partial<InsertResearchAnalysisReport>, auditContext?: AuditContext): Promise<ResearchAnalysisReport | undefined> {
    const [updated] = await db
      .update(researchAnalysisReports)
      .set({ ...data, updatedAt: new Date() })
      .where(eq(researchAnalysisReports.id, id))
      .returning();
    if (updated) {
      await this.logAudit(auditContext, 'UPDATE', 'ResearchAnalysisReport', id, { changedFields: Object.keys(data), status: data.status });
    }
    return updated;
  }

  // =========================================================================
  // RESEARCH ALERTS OPERATIONS
  // =========================================================================

  async getResearchAlerts(filters?: {
    patientId?: string;
    studyId?: string;
    alertType?: string;
    severity?: string;
    status?: string;
    limit?: number;
  }): Promise<ResearchAlert[]> {
    let query = db.select().from(researchAlerts).$dynamic();
    
    const conditions = [];
    if (filters?.patientId) conditions.push(eq(researchAlerts.patientId, filters.patientId));
    if (filters?.studyId) conditions.push(eq(researchAlerts.studyId, filters.studyId));
    if (filters?.alertType) conditions.push(eq(researchAlerts.alertType, filters.alertType));
    if (filters?.severity) conditions.push(eq(researchAlerts.severity, filters.severity));
    if (filters?.status) conditions.push(eq(researchAlerts.status, filters.status));
    
    if (conditions.length > 0) query = query.where(and(...conditions));
    
    query = query.orderBy(desc(researchAlerts.createdAt));
    
    if (filters?.limit) query = query.limit(filters.limit);
    
    return query;
  }

  async createResearchAlert(alert: InsertResearchAlert, auditContext?: AuditContext): Promise<ResearchAlert> {
    const [created] = await db.insert(researchAlerts).values(alert).returning();
    await this.logAudit(auditContext, 'CREATE', 'ResearchAlert', created.id, {
      patientId: alert.patientId,
      alertType: alert.alertType,
      severity: alert.severity,
    });
    return created;
  }

  async acknowledgeResearchAlert(id: string, userId: string, auditContext?: AuditContext): Promise<ResearchAlert | undefined> {
    const [updated] = await db
      .update(researchAlerts)
      .set({ status: "acknowledged", acknowledgedAt: new Date(), acknowledgedBy: userId })
      .where(eq(researchAlerts.id, id))
      .returning();
    if (updated) {
      await this.logAudit(auditContext, 'ACKNOWLEDGE', 'ResearchAlert', id, { acknowledgedBy: userId });
    }
    return updated;
  }

  async resolveResearchAlert(id: string, userId: string, resolution: string, auditContext?: AuditContext): Promise<ResearchAlert | undefined> {
    const [updated] = await db
      .update(researchAlerts)
      .set({ status: "resolved", resolvedAt: new Date(), resolvedBy: userId, resolution })
      .where(eq(researchAlerts.id, id))
      .returning();
    if (updated) {
      await this.logAudit(auditContext, 'RESOLVE', 'ResearchAlert', id, { resolvedBy: userId, resolution });
    }
    return updated;
  }

  // =========================================================================
  // DAILY FOLLOWUP TEMPLATES OPERATIONS
  // =========================================================================

  async getDailyFollowupTemplates(createdBy?: string): Promise<DailyFollowupTemplate[]> {
    if (createdBy) {
      return db
        .select()
        .from(dailyFollowupTemplates)
        .where(eq(dailyFollowupTemplates.createdBy, createdBy))
        .orderBy(desc(dailyFollowupTemplates.createdAt));
    }
    return db
      .select()
      .from(dailyFollowupTemplates)
      .where(eq(dailyFollowupTemplates.isActive, true))
      .orderBy(desc(dailyFollowupTemplates.createdAt));
  }

  async getDailyFollowupTemplate(id: string): Promise<DailyFollowupTemplate | undefined> {
    const [template] = await db
      .select()
      .from(dailyFollowupTemplates)
      .where(eq(dailyFollowupTemplates.id, id));
    return template;
  }

  async createDailyFollowupTemplate(template: InsertDailyFollowupTemplate, auditContext?: AuditContext): Promise<DailyFollowupTemplate> {
    const [created] = await db.insert(dailyFollowupTemplates).values(template).returning();
    await this.logAudit(auditContext, 'CREATE', 'DailyFollowupTemplate', created.id, {
      name: template.name,
      createdBy: template.createdBy,
    });
    return created;
  }

  async updateDailyFollowupTemplate(id: string, data: Partial<InsertDailyFollowupTemplate>, auditContext?: AuditContext): Promise<DailyFollowupTemplate | undefined> {
    const [updated] = await db
      .update(dailyFollowupTemplates)
      .set({ ...data, updatedAt: new Date() })
      .where(eq(dailyFollowupTemplates.id, id))
      .returning();
    if (updated) {
      await this.logAudit(auditContext, 'UPDATE', 'DailyFollowupTemplate', id, { changedFields: Object.keys(data) });
    }
    return updated;
  }

  // =========================================================================
  // DAILY FOLLOWUP ASSIGNMENTS OPERATIONS
  // =========================================================================

  async getDailyFollowupAssignments(filters: {
    patientId?: string;
    studyId?: string;
    templateId?: string;
    isActive?: boolean;
  }): Promise<DailyFollowupAssignment[]> {
    let query = db.select().from(dailyFollowupAssignments).$dynamic();
    
    const conditions = [];
    if (filters.patientId) conditions.push(eq(dailyFollowupAssignments.patientId, filters.patientId));
    if (filters.studyId) conditions.push(eq(dailyFollowupAssignments.studyId, filters.studyId));
    if (filters.templateId) conditions.push(eq(dailyFollowupAssignments.templateId, filters.templateId));
    if (filters.isActive !== undefined) conditions.push(eq(dailyFollowupAssignments.isActive, filters.isActive));
    
    if (conditions.length > 0) query = query.where(and(...conditions));
    
    return query.orderBy(desc(dailyFollowupAssignments.createdAt));
  }

  async createDailyFollowupAssignment(assignment: InsertDailyFollowupAssignment, auditContext?: AuditContext): Promise<DailyFollowupAssignment> {
    const [created] = await db.insert(dailyFollowupAssignments).values(assignment).returning();
    await this.logAudit(auditContext, 'CREATE', 'DailyFollowupAssignment', created.id, {
      patientId: assignment.patientId,
      templateId: assignment.templateId,
      studyId: assignment.studyId,
    });
    return created;
  }

  async updateDailyFollowupAssignment(id: string, data: Partial<InsertDailyFollowupAssignment>, auditContext?: AuditContext): Promise<DailyFollowupAssignment | undefined> {
    const [updated] = await db
      .update(dailyFollowupAssignments)
      .set(data)
      .where(eq(dailyFollowupAssignments.id, id))
      .returning();
    if (updated) {
      await this.logAudit(auditContext, 'UPDATE', 'DailyFollowupAssignment', id, { changedFields: Object.keys(data), isActive: data.isActive });
    }
    return updated;
  }

  // =========================================================================
  // DAILY FOLLOWUP RESPONSES OPERATIONS
  // =========================================================================

  async getDailyFollowupResponses(filters: {
    patientId?: string;
    assignmentId?: string;
    startDate?: Date;
    endDate?: Date;
    limit?: number;
  }): Promise<DailyFollowupResponse[]> {
    let query = db.select().from(dailyFollowupResponses).$dynamic();
    
    const conditions = [];
    if (filters.patientId) conditions.push(eq(dailyFollowupResponses.patientId, filters.patientId));
    if (filters.assignmentId) conditions.push(eq(dailyFollowupResponses.assignmentId, filters.assignmentId));
    if (filters.startDate) conditions.push(gte(dailyFollowupResponses.responseDate, filters.startDate));
    if (filters.endDate) conditions.push(lte(dailyFollowupResponses.responseDate, filters.endDate));
    
    if (conditions.length > 0) query = query.where(and(...conditions));
    
    query = query.orderBy(desc(dailyFollowupResponses.responseDate));
    
    if (filters.limit) query = query.limit(filters.limit);
    
    return query;
  }

  async createDailyFollowupResponse(response: InsertDailyFollowupResponse, auditContext?: AuditContext): Promise<DailyFollowupResponse> {
    const [created] = await db.insert(dailyFollowupResponses).values(response).returning();
    await this.logAudit(auditContext, 'CREATE', 'DailyFollowupResponse', created.id, {
      patientId: response.patientId,
      assignmentId: response.assignmentId,
    });
    return created;
  }

  // =========================================================================
  // RESEARCH AUDIT LOGS OPERATIONS
  // =========================================================================

  async createResearchAuditLog(log: InsertResearchAuditLog): Promise<ResearchAuditLog> {
    const [created] = await db.insert(researchAuditLogs).values(log).returning();
    return created;
  }

  async getResearchAuditLogs(filters?: {
    userId?: string;
    actionType?: string;
    objectType?: string;
    objectId?: string;
    startDate?: Date;
    endDate?: Date;
    limit?: number;
  }): Promise<ResearchAuditLog[]> {
    let query = db.select().from(researchAuditLogs).$dynamic();
    
    const conditions = [];
    if (filters?.userId) conditions.push(eq(researchAuditLogs.userId, filters.userId));
    if (filters?.actionType) conditions.push(eq(researchAuditLogs.actionType, filters.actionType));
    if (filters?.objectType) conditions.push(eq(researchAuditLogs.objectType, filters.objectType));
    if (filters?.objectId) conditions.push(eq(researchAuditLogs.objectId, filters.objectId));
    if (filters?.startDate) conditions.push(gte(researchAuditLogs.createdAt, filters.startDate));
    if (filters?.endDate) conditions.push(lte(researchAuditLogs.createdAt, filters.endDate));
    
    if (conditions.length > 0) query = query.where(and(...conditions));
    
    query = query.orderBy(desc(researchAuditLogs.createdAt));
    
    if (filters?.limit) query = query.limit(filters.limit);
    
    return query;
  }

  // =========================================================================
  // ANALYSIS JOBS OPERATIONS
  // =========================================================================

  async createAnalysisJob(job: InsertAnalysisJob, auditContext?: AuditContext): Promise<AnalysisJob> {
    const [created] = await db.insert(analysisJobs).values(job).returning();
    await this.logAudit(auditContext, 'CREATE', 'AnalysisJob', created.id, {
      analysisType: job.analysisType,
      studyId: job.studyId,
      cohortId: job.cohortId,
    });
    return created;
  }

  async getAnalysisJob(id: string): Promise<AnalysisJob | undefined> {
    const [job] = await db
      .select()
      .from(analysisJobs)
      .where(eq(analysisJobs.id, id));
    return job;
  }

  async updateAnalysisJob(id: string, data: Partial<InsertAnalysisJob>, auditContext?: AuditContext): Promise<AnalysisJob | undefined> {
    const [updated] = await db
      .update(analysisJobs)
      .set(data)
      .where(eq(analysisJobs.id, id))
      .returning();
    if (updated) {
      await this.logAudit(auditContext, 'UPDATE', 'AnalysisJob', id, { status: data.status, changedFields: Object.keys(data) });
    }
    return updated;
  }

  async getPendingAnalysisJobs(): Promise<AnalysisJob[]> {
    return db
      .select()
      .from(analysisJobs)
      .where(eq(analysisJobs.status, "pending"))
      .orderBy(analysisJobs.createdAt);
  }

  // =========================================================================
  // METRICS OPERATIONS
  // =========================================================================

  async getResearchMetrics(): Promise<ResearchMetrics> {
    const [totalResult] = await db
      .select({ count: sql<number>`count(*)::int` })
      .from(users)
      .where(eq(users.role, "patient"));
    
    const [consentedResult] = await db
      .select({ count: sql<number>`count(*)::int` })
      .from(researchDataConsent)
      .where(eq(researchDataConsent.consentEnabled, true));
    
    const [activeStudiesResult] = await db
      .select({ count: sql<number>`count(*)::int` })
      .from(researchStudies)
      .where(
        or(
          eq(researchStudies.status, "enrolling"),
          eq(researchStudies.status, "follow_up"),
          eq(researchStudies.status, "analysis")
        )
      );
    
    const [cohortsResult] = await db
      .select({ count: sql<number>`count(*)::int` })
      .from(researchCohorts)
      .where(eq(researchCohorts.status, "active"));
    
    const [reportsResult] = await db
      .select({ count: sql<number>`count(*)::int` })
      .from(researchAnalysisReports);
    
    const thirtyDaysAgo = new Date();
    thirtyDaysAgo.setDate(thirtyDaysAgo.getDate() - 30);
    
    const [followupsResult] = await db
      .select({ count: sql<number>`count(*)::int` })
      .from(dailyFollowups)
      .where(gte(dailyFollowups.createdAt, thirtyDaysAgo));
    
    const expectedFollowups = (consentedResult?.count || 0) * 30;
    const dataCompleteness = expectedFollowups > 0 
      ? Math.min(100, Math.round(((followupsResult?.count || 0) / expectedFollowups) * 100))
      : 0;
    
    return {
      totalPatients: totalResult?.count || 0,
      consentedPatients: consentedResult?.count || 0,
      activeStudies: activeStudiesResult?.count || 0,
      activeCohorts: cohortsResult?.count || 0,
      reportsGenerated: reportsResult?.count || 0,
      dataCompleteness,
    };
  }

  async getGenderDistribution(): Promise<Record<string, number>> {
    const consents = await this.getConsentedPatients();
    return { 'Unknown': consents.length };
  }

  async getConditionDistribution(): Promise<Record<string, number>> {
    const consents = await this.getConsentedPatients();
    const patientIds = consents.map(c => c.patientId);
    
    if (patientIds.length === 0) return {};
    
    const profiles = await db
      .select()
      .from(patientProfiles)
      .where(inArray(patientProfiles.userId, patientIds));
    
    const distribution: Record<string, number> = {};
    profiles.forEach(p => {
      const condition = p.immunocompromisedCondition || 'Unknown';
      distribution[condition] = (distribution[condition] || 0) + 1;
    });
    
    return distribution;
  }

  async getEnrollmentTrend(days: number = 30): Promise<{ date: string; count: number }[]> {
    const startDate = new Date();
    startDate.setDate(startDate.getDate() - days);
    
    const enrollments = await db
      .select()
      .from(studyEnrollments)
      .where(gte(studyEnrollments.enrollmentDate, startDate))
      .orderBy(studyEnrollments.enrollmentDate);
    
    const trendMap: Record<string, number> = {};
    enrollments.forEach(e => {
      if (e.enrollmentDate) {
        const dateStr = e.enrollmentDate.toISOString().split('T')[0];
        trendMap[dateStr] = (trendMap[dateStr] || 0) + 1;
      }
    });
    
    return Object.entries(trendMap)
      .map(([date, count]) => ({ date, count }))
      .sort((a, b) => a.date.localeCompare(b.date));
  }

  async getAgeDistribution(): Promise<{ ageGroup: string; count: number }[]> {
    const consents = await this.getConsentedPatients();
    const patientIds = consents.map(c => c.patientId);
    
    if (patientIds.length === 0) return [];
    
    const profiles = await db
      .select()
      .from(patientProfiles)
      .where(inArray(patientProfiles.userId, patientIds));
    
    const distribution: Record<string, number> = {};
    const now = new Date();
    
    profiles.forEach(p => {
      if (p.dateOfBirth) {
        const age = Math.floor((now.getTime() - new Date(p.dateOfBirth).getTime()) / (365.25 * 24 * 60 * 60 * 1000));
        const ageGroup = this.getAgeGroup(age);
        distribution[ageGroup] = (distribution[ageGroup] || 0) + 1;
      } else {
        distribution['Unknown'] = (distribution['Unknown'] || 0) + 1;
      }
    });
    
    return Object.entries(distribution)
      .map(([ageGroup, count]) => ({ ageGroup, count }));
  }

  // ============== CONSENT-AWARE DATA AGGREGATION LAYER ==============
  // All methods respect patient consent grants for specific data types

  private readonly dataTypeMap: Record<string, string> = {
    dailyFollowups: 'dailyFollowups',
    healthAlerts: 'healthAlerts',
    deteriorationIndex: 'deteriorationIndex',
    mlPredictions: 'mlPredictions',
    environmentalRisk: 'environmentalRisk',
    medications: 'medications',
    vitals: 'vitals',
    immuneMarkers: 'immuneMarkers',
    behavioralData: 'behavioralData',
    mentalHealth: 'mentalHealth',
    wearableData: 'wearableData',
    labResults: 'labResults',
    conditions: 'conditions',
    demographics: 'demographics',
    painTracking: 'painTracking',
    symptomJournal: 'symptomJournal',
  };

  async getPatientsConsentedForDataType(
    dataType: keyof typeof this.dataTypeMap,
    context?: AuditContext
  ): Promise<string[]> {
    const consents = await db
      .select()
      .from(researchDataConsent)
      .where(eq(researchDataConsent.consentEnabled, true));
    
    const consentedPatients = consents.filter(c => {
      const permissions = c.dataTypePermissions as Record<string, boolean> | null;
      return permissions && permissions[dataType] === true;
    });
    
    const patientIds = consentedPatients.map(c => c.patientId);
    
    await this.logAudit(context, 'query_consented_patients', 'consent', dataType, {
      dataType,
      count: patientIds.length,
    });
    
    return patientIds;
  }

  async getConsentedDailyFollowups(
    context?: AuditContext,
    dateRange?: { start: Date; end: Date },
    patientIds?: string[]
  ): Promise<any[]> {
    const consentedPatients = await this.getPatientsConsentedForDataType('dailyFollowups', context);
    if (consentedPatients.length === 0) return [];
    
    const targetPatients = patientIds 
      ? consentedPatients.filter(p => patientIds.includes(p))
      : consentedPatients;
    
    if (targetPatients.length === 0) return [];
    
    let query = db
      .select()
      .from(dailyFollowups)
      .where(inArray(dailyFollowups.patientId, targetPatients));
    
    if (dateRange) {
      query = db
        .select()
        .from(dailyFollowups)
        .where(and(
          inArray(dailyFollowups.patientId, targetPatients),
          gte(dailyFollowups.date, dateRange.start),
          lte(dailyFollowups.date, dateRange.end)
        ));
    }
    
    const results = await query;
    
    await this.logAudit(context, 'query_research_data', 'dailyFollowups', 'aggregation', {
      patientCount: targetPatients.length,
      recordCount: results.length,
      dateRange,
    });
    
    return results;
  }

  async getConsentedHealthAlerts(
    context?: AuditContext,
    dateRange?: { start: Date; end: Date },
    patientIds?: string[]
  ): Promise<any[]> {
    const consentedPatients = await this.getPatientsConsentedForDataType('healthAlerts', context);
    if (consentedPatients.length === 0) return [];
    
    const targetPatients = patientIds 
      ? consentedPatients.filter(p => patientIds.includes(p))
      : consentedPatients;
    
    if (targetPatients.length === 0) return [];
    
    const results = await db
      .select()
      .from(interactionAlerts)
      .where(inArray(interactionAlerts.patientId, targetPatients));
    
    await this.logAudit(context, 'query_research_data', 'healthAlerts', 'aggregation', {
      patientCount: targetPatients.length,
      recordCount: results.length,
    });
    
    return results;
  }

  async getConsentedDeteriorationScores(
    context?: AuditContext,
    dateRange?: { start: Date; end: Date },
    patientIds?: string[]
  ): Promise<any[]> {
    const consentedPatients = await this.getPatientsConsentedForDataType('deteriorationIndex', context);
    if (consentedPatients.length === 0) return [];
    
    const targetPatients = patientIds 
      ? consentedPatients.filter(p => patientIds.includes(p))
      : consentedPatients;
    
    if (targetPatients.length === 0) return [];
    
    let query = db
      .select()
      .from(deteriorationPredictions)
      .where(inArray(deteriorationPredictions.userId, targetPatients));
    
    if (dateRange) {
      query = db
        .select()
        .from(deteriorationPredictions)
        .where(and(
          inArray(deteriorationPredictions.userId, targetPatients),
          gte(deteriorationPredictions.predictionDate, dateRange.start),
          lte(deteriorationPredictions.predictionDate, dateRange.end)
        ));
    }
    
    const results = await query;
    
    await this.logAudit(context, 'query_research_data', 'deteriorationPredictions', 'aggregation', {
      patientCount: targetPatients.length,
      recordCount: results.length,
      dateRange,
    });
    
    return results;
  }

  async getConsentedMedications(
    context?: AuditContext,
    patientIds?: string[]
  ): Promise<any[]> {
    const consentedPatients = await this.getPatientsConsentedForDataType('medications', context);
    if (consentedPatients.length === 0) return [];
    
    const targetPatients = patientIds 
      ? consentedPatients.filter(p => patientIds.includes(p))
      : consentedPatients;
    
    if (targetPatients.length === 0) return [];
    
    const results = await db
      .select()
      .from(medications)
      .where(inArray(medications.patientId, targetPatients));
    
    await this.logAudit(context, 'query_research_data', 'medications', 'aggregation', {
      patientCount: targetPatients.length,
      recordCount: results.length,
    });
    
    return results;
  }

  async getConsentedImmuneBiomarkers(
    context?: AuditContext,
    dateRange?: { start: Date; end: Date },
    patientIds?: string[]
  ): Promise<any[]> {
    const consentedPatients = await this.getPatientsConsentedForDataType('immuneMarkers', context);
    if (consentedPatients.length === 0) return [];
    
    const targetPatients = patientIds 
      ? consentedPatients.filter(p => patientIds.includes(p))
      : consentedPatients;
    
    if (targetPatients.length === 0) return [];
    
    let query = db
      .select()
      .from(immuneBiomarkers)
      .where(inArray(immuneBiomarkers.userId, targetPatients));
    
    if (dateRange) {
      query = db
        .select()
        .from(immuneBiomarkers)
        .where(and(
          inArray(immuneBiomarkers.userId, targetPatients),
          gte(immuneBiomarkers.measuredAt, dateRange.start),
          lte(immuneBiomarkers.measuredAt, dateRange.end)
        ));
    }
    
    const results = await query;
    
    await this.logAudit(context, 'query_research_data', 'immuneBiomarkers', 'aggregation', {
      patientCount: targetPatients.length,
      recordCount: results.length,
      dateRange,
    });
    
    return results;
  }

  async getConsentedWearableData(
    context?: AuditContext,
    dateRange?: { start: Date; end: Date },
    patientIds?: string[]
  ): Promise<any[]> {
    const consentedPatients = await this.getPatientsConsentedForDataType('wearableData', context);
    if (consentedPatients.length === 0) return [];
    
    const targetPatients = patientIds 
      ? consentedPatients.filter(p => patientIds.includes(p))
      : consentedPatients;
    
    if (targetPatients.length === 0) return [];
    
    let query = db
      .select()
      .from(digitalBiomarkers)
      .where(inArray(digitalBiomarkers.patientId, targetPatients));
    
    if (dateRange) {
      query = db
        .select()
        .from(digitalBiomarkers)
        .where(and(
          inArray(digitalBiomarkers.patientId, targetPatients),
          gte(digitalBiomarkers.date, dateRange.start),
          lte(digitalBiomarkers.date, dateRange.end)
        ));
    }
    
    const results = await query;
    
    await this.logAudit(context, 'query_research_data', 'wearableData', 'aggregation', {
      patientCount: targetPatients.length,
      recordCount: results.length,
      dateRange,
    });
    
    return results;
  }

  async getConsentedSymptomJournal(
    context?: AuditContext,
    dateRange?: { start: Date; end: Date },
    patientIds?: string[]
  ): Promise<any[]> {
    const consentedPatients = await this.getPatientsConsentedForDataType('symptomJournal', context);
    if (consentedPatients.length === 0) return [];
    
    const targetPatients = patientIds 
      ? consentedPatients.filter(p => patientIds.includes(p))
      : consentedPatients;
    
    if (targetPatients.length === 0) return [];
    
    let query = db
      .select()
      .from(symptomCheckins)
      .where(inArray(symptomCheckins.userId, targetPatients));
    
    if (dateRange) {
      query = db
        .select()
        .from(symptomCheckins)
        .where(and(
          inArray(symptomCheckins.userId, targetPatients),
          gte(symptomCheckins.timestamp, dateRange.start),
          lte(symptomCheckins.timestamp, dateRange.end)
        ));
    }
    
    const results = await query;
    
    await this.logAudit(context, 'query_research_data', 'symptomCheckins', 'aggregation', {
      patientCount: targetPatients.length,
      recordCount: results.length,
      dateRange,
    });
    
    return results;
  }

  async getConsentedPainTracking(
    context?: AuditContext,
    dateRange?: { start: Date; end: Date },
    patientIds?: string[]
  ): Promise<any[]> {
    const consentedPatients = await this.getPatientsConsentedForDataType('painTracking', context);
    if (consentedPatients.length === 0) return [];
    
    const targetPatients = patientIds 
      ? consentedPatients.filter(p => patientIds.includes(p))
      : consentedPatients;
    
    if (targetPatients.length === 0) return [];
    
    let query = db
      .select()
      .from(paintrackSessions)
      .where(inArray(paintrackSessions.userId, targetPatients));
    
    if (dateRange) {
      query = db
        .select()
        .from(paintrackSessions)
        .where(and(
          inArray(paintrackSessions.userId, targetPatients),
          gte(paintrackSessions.createdAt, dateRange.start),
          lte(paintrackSessions.createdAt, dateRange.end)
        ));
    }
    
    const results = await query;
    
    await this.logAudit(context, 'query_research_data', 'painTracking', 'aggregation', {
      patientCount: targetPatients.length,
      recordCount: results.length,
      dateRange,
    });
    
    return results;
  }

  async getConsentedLabResults(
    context?: AuditContext,
    patientIds?: string[]
  ): Promise<any[]> {
    const consentedPatients = await this.getPatientsConsentedForDataType('labResults', context);
    if (consentedPatients.length === 0) return [];
    
    const targetPatients = patientIds 
      ? consentedPatients.filter(p => patientIds.includes(p))
      : consentedPatients;
    
    if (targetPatients.length === 0) return [];
    
    const results = await db
      .select()
      .from(medicalDocuments)
      .where(and(
        inArray(medicalDocuments.userId, targetPatients),
        eq(medicalDocuments.documentType, 'lab_report')
      ));
    
    await this.logAudit(context, 'query_research_data', 'labResults', 'aggregation', {
      patientCount: targetPatients.length,
      recordCount: results.length,
    });
    
    return results;
  }

  async getConsentedConditions(
    context?: AuditContext,
    patientIds?: string[]
  ): Promise<any[]> {
    const consentedPatients = await this.getPatientsConsentedForDataType('conditions', context);
    if (consentedPatients.length === 0) return [];
    
    const targetPatients = patientIds 
      ? consentedPatients.filter(p => patientIds.includes(p))
      : consentedPatients;
    
    if (targetPatients.length === 0) return [];
    
    const results = await db
      .select({
        userId: patientProfiles.userId,
        immunocompromisedCondition: patientProfiles.immunocompromisedCondition,
        diagnosisDate: patientProfiles.diagnosisDate,
        currentTreatmentPhase: patientProfiles.currentTreatmentPhase,
        allergies: patientProfiles.allergies,
        secondaryConditions: patientProfiles.secondaryConditions,
      })
      .from(patientProfiles)
      .where(inArray(patientProfiles.userId, targetPatients));
    
    await this.logAudit(context, 'query_research_data', 'conditions', 'aggregation', {
      patientCount: targetPatients.length,
      recordCount: results.length,
    });
    
    return results;
  }

  async getConsentedDemographics(
    context?: AuditContext,
    patientIds?: string[]
  ): Promise<any[]> {
    const consentedPatients = await this.getPatientsConsentedForDataType('demographics', context);
    if (consentedPatients.length === 0) return [];
    
    const targetPatients = patientIds 
      ? consentedPatients.filter(p => patientIds.includes(p))
      : consentedPatients;
    
    if (targetPatients.length === 0) return [];
    
    const profiles = await db
      .select({
        userId: patientProfiles.userId,
        dateOfBirth: patientProfiles.dateOfBirth,
        gender: patientProfiles.gender,
        ethnicity: patientProfiles.ethnicity,
        immunocompromisedCondition: patientProfiles.immunocompromisedCondition,
        diagnosisDate: patientProfiles.diagnosisDate,
        currentTreatmentPhase: patientProfiles.currentTreatmentPhase,
      })
      .from(patientProfiles)
      .where(inArray(patientProfiles.userId, targetPatients));
    
    await this.logAudit(context, 'query_research_data', 'demographics', 'aggregation', {
      patientCount: targetPatients.length,
      recordCount: profiles.length,
    });
    
    return profiles;
  }

  async getComprehensivePatientData(
    patientId: string,
    context?: AuditContext
  ): Promise<Record<string, any>> {
    const consent = await db
      .select()
      .from(researchDataConsent)
      .where(eq(researchDataConsent.patientId, patientId))
      .limit(1);
    
    if (consent.length === 0 || !consent[0].consentEnabled) {
      await this.logAudit(context, 'access_denied', 'patient_data', patientId, {
        reason: 'no_consent',
      });
      return { error: 'Patient has not consented to research participation' };
    }
    
    const permissions = consent[0].dataTypePermissions as Record<string, boolean> | null;
    const result: Record<string, any> = {
      patientId,
      consentedAt: consent[0].createdAt,
      dataTypes: {},
    };
    
    if (permissions?.dailyFollowups) {
      result.dataTypes.dailyFollowups = await db
        .select()
        .from(dailyFollowups)
        .where(eq(dailyFollowups.patientId, patientId))
        .orderBy(desc(dailyFollowups.date))
        .limit(100);
    }
    
    if (permissions?.medications) {
      result.dataTypes.medications = await db
        .select()
        .from(medications)
        .where(eq(medications.patientId, patientId));
    }
    
    if (permissions?.immuneMarkers) {
      result.dataTypes.immuneMarkers = await db
        .select()
        .from(immuneBiomarkers)
        .where(eq(immuneBiomarkers.userId, patientId))
        .orderBy(desc(immuneBiomarkers.measuredAt))
        .limit(100);
    }
    
    if (permissions?.wearableData) {
      result.dataTypes.wearableData = await db
        .select()
        .from(digitalBiomarkers)
        .where(eq(digitalBiomarkers.patientId, patientId))
        .orderBy(desc(digitalBiomarkers.date))
        .limit(100);
    }
    
    if (permissions?.symptomJournal) {
      result.dataTypes.symptomJournal = await db
        .select()
        .from(symptomCheckins)
        .where(eq(symptomCheckins.userId, patientId))
        .orderBy(desc(symptomCheckins.timestamp))
        .limit(100);
    }
    
    if (permissions?.painTracking) {
      result.dataTypes.painTracking = await db
        .select()
        .from(paintrackSessions)
        .where(eq(paintrackSessions.userId, patientId))
        .orderBy(desc(paintrackSessions.createdAt))
        .limit(50);
    }
    
    if (permissions?.deteriorationIndex) {
      result.dataTypes.deteriorationScores = await db
        .select()
        .from(deteriorationPredictions)
        .where(eq(deteriorationPredictions.userId, patientId))
        .orderBy(desc(deteriorationPredictions.predictionDate))
        .limit(50);
    }
    
    if (permissions?.demographics) {
      const profile = await db
        .select({
          dateOfBirth: patientProfiles.dateOfBirth,
          gender: patientProfiles.gender,
          ethnicity: patientProfiles.ethnicity,
          immunocompromisedCondition: patientProfiles.immunocompromisedCondition,
          diagnosisDate: patientProfiles.diagnosisDate,
          currentTreatmentPhase: patientProfiles.currentTreatmentPhase,
        })
        .from(patientProfiles)
        .where(eq(patientProfiles.userId, patientId))
        .limit(1);
      result.dataTypes.demographics = profile[0] || null;
    }
    
    if (permissions?.conditions) {
      const conditionsData = await db
        .select({
          immunocompromisedCondition: patientProfiles.immunocompromisedCondition,
          diagnosisDate: patientProfiles.diagnosisDate,
          currentTreatmentPhase: patientProfiles.currentTreatmentPhase,
          allergies: patientProfiles.allergies,
          secondaryConditions: patientProfiles.secondaryConditions,
        })
        .from(patientProfiles)
        .where(eq(patientProfiles.userId, patientId))
        .limit(1);
      result.dataTypes.conditions = conditionsData[0] || null;
    }
    
    await this.logAudit(context, 'query_comprehensive_data', 'patient_data', patientId, {
      dataTypesIncluded: Object.keys(result.dataTypes),
    });
    
    return result;
  }

  async getCohortAggregatedData(
    cohortId: string,
    dataTypes: string[],
    context?: AuditContext,
    dateRange?: { start: Date; end: Date }
  ): Promise<Record<string, any>> {
    const cohort = await db
      .select()
      .from(researchCohorts)
      .where(eq(researchCohorts.id, cohortId))
      .limit(1);
    
    if (cohort.length === 0) {
      return { error: 'Cohort not found' };
    }
    
    const patientIds = (cohort[0].patientIds as string[]) || [];
    if (patientIds.length === 0) {
      return { error: 'Cohort has no patients', cohortId };
    }
    
    const result: Record<string, any> = {
      cohortId,
      cohortName: cohort[0].name,
      totalPatients: patientIds.length,
      dataByType: {},
    };
    
    for (const dataType of dataTypes) {
      switch (dataType) {
        case 'dailyFollowups':
          result.dataByType.dailyFollowups = await this.getConsentedDailyFollowups(context, dateRange, patientIds);
          break;
        case 'healthAlerts':
          result.dataByType.healthAlerts = await this.getConsentedHealthAlerts(context, dateRange, patientIds);
          break;
        case 'deteriorationIndex':
          result.dataByType.deteriorationScores = await this.getConsentedDeteriorationScores(context, dateRange, patientIds);
          break;
        case 'medications':
          result.dataByType.medications = await this.getConsentedMedications(context, patientIds);
          break;
        case 'immuneMarkers':
          result.dataByType.immuneMarkers = await this.getConsentedImmuneBiomarkers(context, dateRange, patientIds);
          break;
        case 'wearableData':
          result.dataByType.wearableData = await this.getConsentedWearableData(context, dateRange, patientIds);
          break;
        case 'symptomJournal':
          result.dataByType.symptomJournal = await this.getConsentedSymptomJournal(context, dateRange, patientIds);
          break;
        case 'painTracking':
          result.dataByType.painTracking = await this.getConsentedPainTracking(context, dateRange, patientIds);
          break;
        case 'labResults':
          result.dataByType.labResults = await this.getConsentedLabResults(context, patientIds);
          break;
        case 'conditions':
          result.dataByType.conditions = await this.getConsentedConditions(context, patientIds);
          break;
        case 'demographics':
          result.dataByType.demographics = await this.getConsentedDemographics(context, patientIds);
          break;
      }
    }
    
    await this.logAudit(context, 'query_cohort_data', 'cohort', cohortId, {
      dataTypes,
      patientCount: patientIds.length,
      dateRange,
    });
    
    return result;
  }

  async getDataTypeStatistics(
    context?: AuditContext
  ): Promise<Record<string, { consentedPatients: number; totalRecords: number }>> {
    const stats: Record<string, { consentedPatients: number; totalRecords: number }> = {};
    
    for (const dataType of Object.keys(this.dataTypeMap)) {
      const patientIds = await this.getPatientsConsentedForDataType(
        dataType as keyof typeof this.dataTypeMap,
        context
      );
      
      let recordCount = 0;
      
      switch (dataType) {
        case 'dailyFollowups':
          if (patientIds.length > 0) {
            const result = await db
              .select({ count: sql<number>`count(*)` })
              .from(dailyFollowups)
              .where(inArray(dailyFollowups.patientId, patientIds));
            recordCount = Number(result[0]?.count || 0);
          }
          break;
        case 'medications':
          if (patientIds.length > 0) {
            const result = await db
              .select({ count: sql<number>`count(*)` })
              .from(medications)
              .where(inArray(medications.patientId, patientIds));
            recordCount = Number(result[0]?.count || 0);
          }
          break;
        case 'immuneMarkers':
          if (patientIds.length > 0) {
            const result = await db
              .select({ count: sql<number>`count(*)` })
              .from(immuneBiomarkers)
              .where(inArray(immuneBiomarkers.userId, patientIds));
            recordCount = Number(result[0]?.count || 0);
          }
          break;
        case 'wearableData':
          if (patientIds.length > 0) {
            const result = await db
              .select({ count: sql<number>`count(*)` })
              .from(digitalBiomarkers)
              .where(inArray(digitalBiomarkers.patientId, patientIds));
            recordCount = Number(result[0]?.count || 0);
          }
          break;
        case 'symptomJournal':
          if (patientIds.length > 0) {
            const result = await db
              .select({ count: sql<number>`count(*)` })
              .from(symptomCheckins)
              .where(inArray(symptomCheckins.userId, patientIds));
            recordCount = Number(result[0]?.count || 0);
          }
          break;
        case 'painTracking':
          if (patientIds.length > 0) {
            const result = await db
              .select({ count: sql<number>`count(*)` })
              .from(paintrackSessions)
              .where(inArray(paintrackSessions.userId, patientIds));
            recordCount = Number(result[0]?.count || 0);
          }
          break;
        case 'deteriorationIndex':
          if (patientIds.length > 0) {
            const result = await db
              .select({ count: sql<number>`count(*)` })
              .from(deteriorationPredictions)
              .where(inArray(deteriorationPredictions.userId, patientIds));
            recordCount = Number(result[0]?.count || 0);
          }
          break;
        case 'conditions':
          if (patientIds.length > 0) {
            const result = await db
              .select({ count: sql<number>`count(*)` })
              .from(patientProfiles)
              .where(inArray(patientProfiles.userId, patientIds));
            recordCount = Number(result[0]?.count || 0);
          }
          break;
        case 'demographics':
          if (patientIds.length > 0) {
            const result = await db
              .select({ count: sql<number>`count(*)` })
              .from(patientProfiles)
              .where(inArray(patientProfiles.userId, patientIds));
            recordCount = Number(result[0]?.count || 0);
          }
          break;
      }
      
      stats[dataType] = {
        consentedPatients: patientIds.length,
        totalRecords: recordCount,
      };
    }
    
    await this.logAudit(context, 'query_data_statistics', 'research_data', 'all', {
      dataTypes: Object.keys(stats),
    });
    
    return stats;
  }

  // =============================================================================
  // CSV IMPORT SYSTEM
  // =============================================================================

  private readonly importableDataTypes = [
    'patients',
    'conditions',
    'study_enrollments',
    'visits',
    'measurements',
    'environmental_exposures',
    'immune_markers',
  ] as const;

  async parseCSVPreview(
    csvData: string,
    context?: AuditContext
  ): Promise<{
    headers: string[];
    sampleRows: string[][];
    totalRows: number;
  }> {
    const lines = csvData.split('\n').filter(line => line.trim());
    if (lines.length === 0) {
      return { headers: [], sampleRows: [], totalRows: 0 };
    }

    const headers = lines[0].split(',').map(h => h.trim().replace(/^"|"$/g, ''));
    const sampleRows: string[][] = [];
    
    for (let i = 1; i < Math.min(lines.length, 6); i++) {
      const values = this.parseCSVLine(lines[i]);
      sampleRows.push(values);
    }

    await this.logAudit(context, 'csv_preview', 'import', 'preview', {
      headerCount: headers.length,
      totalRows: lines.length - 1,
    });

    return {
      headers,
      sampleRows,
      totalRows: lines.length - 1,
    };
  }

  private parseCSVLine(line: string): string[] {
    const result: string[] = [];
    let current = '';
    let inQuotes = false;
    
    for (let i = 0; i < line.length; i++) {
      const char = line[i];
      if (char === '"') {
        if (inQuotes && line[i + 1] === '"') {
          current += '"';
          i++;
        } else {
          inQuotes = !inQuotes;
        }
      } else if (char === ',' && !inQuotes) {
        result.push(current.trim());
        current = '';
      } else {
        current += char;
      }
    }
    result.push(current.trim());
    return result;
  }

  async getImportableDataTypes(): Promise<{
    type: string;
    label: string;
    requiredFields: string[];
    optionalFields: string[];
  }[]> {
    return [
      {
        type: 'patients',
        label: 'Patients',
        requiredFields: ['patient_id', 'date_of_birth', 'sex'],
        optionalFields: ['email', 'phone', 'zip_code', 'conditions', 'immunocompromised_status'],
      },
      {
        type: 'conditions',
        label: 'Patient Conditions',
        requiredFields: ['patient_id', 'condition_name', 'condition_code'],
        optionalFields: ['onset_date', 'status', 'severity', 'icd10_code'],
      },
      {
        type: 'study_enrollments',
        label: 'Study Enrollments',
        requiredFields: ['patient_id', 'study_id', 'enrollment_date'],
        optionalFields: ['status', 'arm', 'consent_date', 'withdrawn_date', 'withdrawal_reason'],
      },
      {
        type: 'visits',
        label: 'Study Visits',
        requiredFields: ['patient_id', 'study_id', 'visit_type', 'scheduled_date'],
        optionalFields: ['completed_date', 'status', 'notes', 'protocol_deviation'],
      },
      {
        type: 'measurements',
        label: 'Clinical Measurements',
        requiredFields: ['patient_id', 'measurement_type', 'value', 'date'],
        optionalFields: ['unit', 'method', 'range_low', 'range_high', 'is_abnormal'],
      },
      {
        type: 'environmental_exposures',
        label: 'Environmental Exposures',
        requiredFields: ['patient_id', 'location', 'date'],
        optionalFields: ['aqi', 'temperature', 'humidity', 'pm25', 'pollen_index', 'exposure_type'],
      },
      {
        type: 'immune_markers',
        label: 'Immune Markers',
        requiredFields: ['patient_id', 'marker_type', 'value', 'date'],
        optionalFields: ['unit', 'reference_range', 'is_abnormal', 'lab_name'],
      },
    ];
  }

  async validateColumnMapping(
    dataType: string,
    mapping: Record<string, string>,
    headers: string[],
    context?: AuditContext
  ): Promise<{
    valid: boolean;
    errors: string[];
    warnings: string[];
  }> {
    const dataTypes = await this.getImportableDataTypes();
    const typeConfig = dataTypes.find(t => t.type === dataType);
    
    if (!typeConfig) {
      return {
        valid: false,
        errors: [`Unknown data type: ${dataType}`],
        warnings: [],
      };
    }

    const errors: string[] = [];
    const warnings: string[] = [];

    for (const requiredField of typeConfig.requiredFields) {
      if (!mapping[requiredField] || !headers.includes(mapping[requiredField])) {
        errors.push(`Required field '${requiredField}' is not mapped to a column`);
      }
    }

    for (const csvHeader of headers) {
      const isMapped = Object.values(mapping).includes(csvHeader);
      if (!isMapped) {
        warnings.push(`Column '${csvHeader}' is not mapped and will be ignored`);
      }
    }

    await this.logAudit(context, 'validate_mapping', 'import', dataType, {
      mapping,
      valid: errors.length === 0,
      errorCount: errors.length,
      warningCount: warnings.length,
    });

    return {
      valid: errors.length === 0,
      errors,
      warnings,
    };
  }

  async importCSVData(
    dataType: string,
    csvData: string,
    mapping: Record<string, string>,
    studyId: string | null,
    context?: AuditContext
  ): Promise<{
    success: boolean;
    imported: number;
    skipped: number;
    errors: { row: number; message: string }[];
  }> {
    const lines = csvData.split('\n').filter(line => line.trim());
    if (lines.length < 2) {
      return { success: false, imported: 0, skipped: 0, errors: [{ row: 0, message: 'No data rows found' }] };
    }

    const headers = lines[0].split(',').map(h => h.trim().replace(/^"|"$/g, ''));
    const reverseMapping: Record<string, string> = {};
    for (const [field, csvCol] of Object.entries(mapping)) {
      reverseMapping[csvCol] = field;
    }

    let imported = 0;
    let skipped = 0;
    const errors: { row: number; message: string }[] = [];

    for (let i = 1; i < lines.length; i++) {
      try {
        const values = this.parseCSVLine(lines[i]);
        const rowData: Record<string, string> = {};
        
        for (let j = 0; j < headers.length && j < values.length; j++) {
          const field = reverseMapping[headers[j]];
          if (field) {
            rowData[field] = values[j];
          }
        }

        const importResult = await this.importSingleRow(dataType, rowData, studyId, context);
        if (importResult.success) {
          imported++;
        } else {
          skipped++;
          if (importResult.error) {
            errors.push({ row: i + 1, message: importResult.error });
          }
        }
      } catch (error) {
        skipped++;
        errors.push({ row: i + 1, message: error instanceof Error ? error.message : 'Unknown error' });
      }
    }

    await this.logAudit(context, 'import_csv_complete', 'import', dataType, {
      studyId,
      totalRows: lines.length - 1,
      imported,
      skipped,
      errorCount: errors.length,
    });

    return {
      success: imported > 0,
      imported,
      skipped,
      errors: errors.slice(0, 100),
    };
  }

  private async importSingleRow(
    dataType: string,
    rowData: Record<string, string>,
    studyId: string | null,
    context?: AuditContext
  ): Promise<{ success: boolean; error?: string }> {
    try {
      switch (dataType) {
        case 'study_enrollments':
          if (!studyId && !rowData.study_id) {
            return { success: false, error: 'study_id is required' };
          }
          await db.insert(studyEnrollments).values({
            id: crypto.randomUUID(),
            studyId: studyId || rowData.study_id,
            patientId: rowData.patient_id,
            enrollmentDate: new Date(rowData.enrollment_date),
            status: (rowData.status as any) || 'enrolled',
            arm: rowData.arm || null,
            consentDate: rowData.consent_date ? new Date(rowData.consent_date) : null,
            withdrawnDate: rowData.withdrawn_date ? new Date(rowData.withdrawn_date) : null,
            withdrawalReason: rowData.withdrawal_reason || null,
          });
          break;

        case 'visits':
          if (!studyId && !rowData.study_id) {
            return { success: false, error: 'study_id is required' };
          }
          await db.insert(visits).values({
            id: crypto.randomUUID(),
            patientId: rowData.patient_id,
            studyId: studyId || rowData.study_id,
            visitType: rowData.visit_type,
            scheduledDate: new Date(rowData.scheduled_date),
            completedDate: rowData.completed_date ? new Date(rowData.completed_date) : null,
            status: (rowData.status as any) || 'scheduled',
            notes: rowData.notes || null,
            protocolDeviation: rowData.protocol_deviation === 'true' || rowData.protocol_deviation === '1',
          });
          break;

        case 'immune_markers':
          await db.insert(immuneMarkers).values({
            id: crypto.randomUUID(),
            patientId: rowData.patient_id,
            markerType: rowData.marker_type,
            value: parseFloat(rowData.value),
            unit: rowData.unit || null,
            date: new Date(rowData.date),
            referenceRange: rowData.reference_range || null,
            isAbnormal: rowData.is_abnormal === 'true' || rowData.is_abnormal === '1',
            labName: rowData.lab_name || null,
          });
          break;

        case 'measurements':
          await db.insert(dailyFollowups).values({
            id: crypto.randomUUID(),
            patientId: rowData.patient_id,
            date: new Date(rowData.date),
            energyLevel: rowData.measurement_type === 'energy' ? parseInt(rowData.value) : null,
            painLevel: rowData.measurement_type === 'pain' ? parseInt(rowData.value) : null,
            sleepQuality: rowData.measurement_type === 'sleep' ? parseInt(rowData.value) : null,
            moodScore: rowData.measurement_type === 'mood' ? parseInt(rowData.value) : null,
            notes: `${rowData.measurement_type}: ${rowData.value} ${rowData.unit || ''}`,
          });
          break;

        default:
          return { success: false, error: `Data type '${dataType}' import not yet implemented` };
      }

      return { success: true };
    } catch (error) {
      return { success: false, error: error instanceof Error ? error.message : 'Database error' };
    }
  }

  async getImportHistory(
    studyId?: string,
    context?: AuditContext
  ): Promise<any[]> {
    await this.logAudit(context, 'view_import_history', 'import', 'history', { studyId });
    return [];
  }

  // =============================================================================
  // DATA QUALITY SERVICE
  // =============================================================================

  async getStudyDataQuality(
    studyId: string,
    context?: AuditContext
  ): Promise<{
    studyId: string;
    overallScore: number;
    metrics: {
      completeness: number;
      consistency: number;
      validity: number;
      timeliness: number;
    };
    fieldAnalysis: {
      field: string;
      missingPercent: number;
      outlierPercent: number;
      validPercent: number;
    }[];
    issues: {
      severity: 'critical' | 'warning' | 'info';
      field: string;
      message: string;
      affectedRecords: number;
    }[];
  }> {
    const study = await db.select().from(studies).where(eq(studies.id, studyId)).limit(1);
    if (study.length === 0) {
      throw new Error('Study not found');
    }

    const enrollments = await db.select().from(studyEnrollments).where(eq(studyEnrollments.studyId, studyId));
    const patientIds = enrollments.map(e => e.patientId);

    const fieldAnalysis: { field: string; missingPercent: number; outlierPercent: number; validPercent: number }[] = [];
    const issues: { severity: 'critical' | 'warning' | 'info'; field: string; message: string; affectedRecords: number }[] = [];

    if (patientIds.length > 0) {
      const followups = await db.select().from(dailyFollowups).where(inArray(dailyFollowups.patientId, patientIds));
      
      const energyMissing = followups.filter(f => f.energyLevel === null).length;
      const painMissing = followups.filter(f => f.painLevel === null).length;
      const sleepMissing = followups.filter(f => f.sleepQuality === null).length;
      const moodMissing = followups.filter(f => f.moodScore === null).length;

      const total = followups.length || 1;

      fieldAnalysis.push(
        { field: 'energyLevel', missingPercent: (energyMissing / total) * 100, outlierPercent: 0, validPercent: ((total - energyMissing) / total) * 100 },
        { field: 'painLevel', missingPercent: (painMissing / total) * 100, outlierPercent: 0, validPercent: ((total - painMissing) / total) * 100 },
        { field: 'sleepQuality', missingPercent: (sleepMissing / total) * 100, outlierPercent: 0, validPercent: ((total - sleepMissing) / total) * 100 },
        { field: 'moodScore', missingPercent: (moodMissing / total) * 100, outlierPercent: 0, validPercent: ((total - moodMissing) / total) * 100 }
      );

      if (energyMissing > total * 0.5) {
        issues.push({ severity: 'critical', field: 'energyLevel', message: 'More than 50% of energy level data is missing', affectedRecords: energyMissing });
      } else if (energyMissing > total * 0.2) {
        issues.push({ severity: 'warning', field: 'energyLevel', message: 'More than 20% of energy level data is missing', affectedRecords: energyMissing });
      }

      const outOfRangeEnergy = followups.filter(f => f.energyLevel !== null && (f.energyLevel < 1 || f.energyLevel > 10)).length;
      if (outOfRangeEnergy > 0) {
        issues.push({ severity: 'warning', field: 'energyLevel', message: 'Values outside expected range (1-10)', affectedRecords: outOfRangeEnergy });
        const idx = fieldAnalysis.findIndex(f => f.field === 'energyLevel');
        if (idx >= 0) fieldAnalysis[idx].outlierPercent = (outOfRangeEnergy / total) * 100;
      }
    }

    const completeness = fieldAnalysis.length > 0 
      ? fieldAnalysis.reduce((sum, f) => sum + f.validPercent, 0) / fieldAnalysis.length 
      : 100;
    
    const criticalCount = issues.filter(i => i.severity === 'critical').length;
    const warningCount = issues.filter(i => i.severity === 'warning').length;
    
    const overallScore = Math.max(0, Math.min(100, 
      completeness * 0.4 + 
      (100 - criticalCount * 20 - warningCount * 5) * 0.3 + 
      100 * 0.3
    ));

    await this.logAudit(context, 'data_quality_check', 'study', studyId, {
      overallScore,
      enrollmentCount: enrollments.length,
      issueCount: issues.length,
    });

    return {
      studyId,
      overallScore,
      metrics: {
        completeness,
        consistency: 100 - criticalCount * 10,
        validity: 100 - warningCount * 5,
        timeliness: 100,
      },
      fieldAnalysis,
      issues,
    };
  }

  async getCohortDataQuality(
    cohortId: string,
    context?: AuditContext
  ): Promise<{
    cohortId: string;
    patientCount: number;
    dataTypeCompletion: Record<string, { available: number; total: number; percent: number }>;
    overallCompleteness: number;
  }> {
    const cohort = await db.select().from(cohorts).where(eq(cohorts.id, cohortId)).limit(1);
    if (cohort.length === 0) {
      throw new Error('Cohort not found');
    }

    const patientIds = (cohort[0].patientIds as string[]) || [];
    const dataTypeCompletion: Record<string, { available: number; total: number; percent: number }> = {};
    
    const total = patientIds.length || 1;

    if (patientIds.length > 0) {
      const followupPatients = await db
        .selectDistinct({ patientId: dailyFollowups.patientId })
        .from(dailyFollowups)
        .where(inArray(dailyFollowups.patientId, patientIds));
      dataTypeCompletion.dailyFollowups = {
        available: followupPatients.length,
        total,
        percent: (followupPatients.length / total) * 100,
      };

      const alertPatients = await db
        .selectDistinct({ patientId: interactionAlerts.patientId })
        .from(interactionAlerts)
        .where(inArray(interactionAlerts.patientId, patientIds));
      dataTypeCompletion.healthAlerts = {
        available: alertPatients.length,
        total,
        percent: (alertPatients.length / total) * 100,
      };

      const immunePatients = await db
        .selectDistinct({ patientId: immuneMarkers.patientId })
        .from(immuneMarkers)
        .where(inArray(immuneMarkers.patientId, patientIds));
      dataTypeCompletion.immuneMarkers = {
        available: immunePatients.length,
        total,
        percent: (immunePatients.length / total) * 100,
      };
    }

    const completionValues = Object.values(dataTypeCompletion);
    const overallCompleteness = completionValues.length > 0
      ? completionValues.reduce((sum, c) => sum + c.percent, 0) / completionValues.length
      : 0;

    await this.logAudit(context, 'cohort_quality_check', 'cohort', cohortId, {
      patientCount: patientIds.length,
      overallCompleteness,
    });

    return {
      cohortId,
      patientCount: patientIds.length,
      dataTypeCompletion,
      overallCompleteness,
    };
  }

  async detectOutliers(
    dataType: string,
    field: string,
    patientIds?: string[],
    context?: AuditContext
  ): Promise<{
    field: string;
    statistics: {
      mean: number;
      stdDev: number;
      min: number;
      max: number;
      q1: number;
      median: number;
      q3: number;
    };
    outliers: { patientId: string; value: number; zScore: number }[];
    iqrOutliers: { patientId: string; value: number }[];
  }> {
    let values: { patientId: string; value: number }[] = [];

    if (dataType === 'dailyFollowups') {
      let query = db.select({ patientId: dailyFollowups.patientId, value: dailyFollowups.energyLevel }).from(dailyFollowups);
      if (patientIds && patientIds.length > 0) {
        query = db.select({ patientId: dailyFollowups.patientId, value: dailyFollowups.energyLevel }).from(dailyFollowups).where(inArray(dailyFollowups.patientId, patientIds));
      }
      const rows = await query;
      values = rows.filter(r => r.value !== null).map(r => ({ patientId: r.patientId, value: r.value as number }));
    } else if (dataType === 'immuneMarkers') {
      let query = db.select({ patientId: immuneMarkers.patientId, value: immuneMarkers.value }).from(immuneMarkers);
      if (patientIds && patientIds.length > 0) {
        query = db.select({ patientId: immuneMarkers.patientId, value: immuneMarkers.value }).from(immuneMarkers).where(inArray(immuneMarkers.patientId, patientIds));
      }
      const rows = await query;
      values = rows.map(r => ({ patientId: r.patientId, value: r.value }));
    }

    if (values.length === 0) {
      return {
        field,
        statistics: { mean: 0, stdDev: 0, min: 0, max: 0, q1: 0, median: 0, q3: 0 },
        outliers: [],
        iqrOutliers: [],
      };
    }

    const numericValues = values.map(v => v.value).sort((a, b) => a - b);
    const n = numericValues.length;
    const mean = numericValues.reduce((a, b) => a + b, 0) / n;
    const variance = numericValues.reduce((sum, v) => sum + Math.pow(v - mean, 2), 0) / n;
    const stdDev = Math.sqrt(variance);
    const min = numericValues[0];
    const max = numericValues[n - 1];
    const q1 = numericValues[Math.floor(n * 0.25)];
    const median = numericValues[Math.floor(n * 0.5)];
    const q3 = numericValues[Math.floor(n * 0.75)];
    const iqr = q3 - q1;

    const outliers = values
      .map(v => ({ ...v, zScore: stdDev > 0 ? (v.value - mean) / stdDev : 0 }))
      .filter(v => Math.abs(v.zScore) > 3);

    const iqrOutliers = values.filter(v => v.value < q1 - 1.5 * iqr || v.value > q3 + 1.5 * iqr);

    await this.logAudit(context, 'outlier_detection', dataType, field, {
      valueCount: values.length,
      outlierCount: outliers.length,
      iqrOutlierCount: iqrOutliers.length,
    });

    return {
      field,
      statistics: { mean, stdDev, min, max, q1, median, q3 },
      outliers,
      iqrOutliers,
    };
  }

  async getDateConsistencyReport(
    studyId: string,
    context?: AuditContext
  ): Promise<{
    studyId: string;
    issues: {
      type: 'future_date' | 'pre_enrollment' | 'post_withdrawal' | 'invalid_sequence';
      description: string;
      affectedRecords: number;
      examples: { patientId: string; date: string }[];
    }[];
    totalIssues: number;
  }> {
    const enrollments = await db.select().from(studyEnrollments).where(eq(studyEnrollments.studyId, studyId));
    const issues: {
      type: 'future_date' | 'pre_enrollment' | 'post_withdrawal' | 'invalid_sequence';
      description: string;
      affectedRecords: number;
      examples: { patientId: string; date: string }[];
    }[] = [];

    const now = new Date();
    let totalIssues = 0;

    for (const enrollment of enrollments) {
      const followups = await db.select().from(dailyFollowups).where(eq(dailyFollowups.patientId, enrollment.patientId));
      
      const futureDates = followups.filter(f => f.date > now);
      if (futureDates.length > 0) {
        issues.push({
          type: 'future_date',
          description: 'Records with dates in the future',
          affectedRecords: futureDates.length,
          examples: futureDates.slice(0, 3).map(f => ({ patientId: enrollment.patientId, date: f.date.toISOString() })),
        });
        totalIssues += futureDates.length;
      }

      const preEnrollment = followups.filter(f => f.date < enrollment.enrollmentDate);
      if (preEnrollment.length > 0) {
        issues.push({
          type: 'pre_enrollment',
          description: 'Records dated before enrollment',
          affectedRecords: preEnrollment.length,
          examples: preEnrollment.slice(0, 3).map(f => ({ patientId: enrollment.patientId, date: f.date.toISOString() })),
        });
        totalIssues += preEnrollment.length;
      }

      if (enrollment.withdrawnDate) {
        const postWithdrawal = followups.filter(f => f.date > enrollment.withdrawnDate!);
        if (postWithdrawal.length > 0) {
          issues.push({
            type: 'post_withdrawal',
            description: 'Records dated after withdrawal',
            affectedRecords: postWithdrawal.length,
            examples: postWithdrawal.slice(0, 3).map(f => ({ patientId: enrollment.patientId, date: f.date.toISOString() })),
          });
          totalIssues += postWithdrawal.length;
        }
      }
    }

    await this.logAudit(context, 'date_consistency_check', 'study', studyId, {
      enrollmentCount: enrollments.length,
      totalIssues,
    });

    return {
      studyId,
      issues,
      totalIssues,
    };
  }

  async getDataQualityHeatmap(
    studyId?: string,
    context?: AuditContext
  ): Promise<{
    dataTypes: string[];
    patients: string[];
    heatmap: Record<string, Record<string, number>>;
  }> {
    const dataTypes = ['dailyFollowups', 'healthAlerts', 'immuneMarkers', 'painTracking', 'symptomJournal'];
    
    let patientIds: string[] = [];
    if (studyId) {
      const enrollments = await db.select().from(studyEnrollments).where(eq(studyEnrollments.studyId, studyId));
      patientIds = enrollments.map(e => e.patientId);
    } else {
      const profiles = await db.select({ userId: patientProfiles.userId }).from(patientProfiles).limit(50);
      patientIds = profiles.map(p => p.userId);
    }

    const heatmap: Record<string, Record<string, number>> = {};

    for (const patientId of patientIds.slice(0, 20)) {
      heatmap[patientId] = {};
      
      const followupCount = await db.select({ count: sql<number>`count(*)` }).from(dailyFollowups).where(eq(dailyFollowups.patientId, patientId));
      heatmap[patientId].dailyFollowups = Math.min(100, Number(followupCount[0]?.count || 0) * 10);

      const alertCount = await db.select({ count: sql<number>`count(*)` }).from(interactionAlerts).where(eq(interactionAlerts.patientId, patientId));
      heatmap[patientId].healthAlerts = Math.min(100, Number(alertCount[0]?.count || 0) * 20);

      const immuneCount = await db.select({ count: sql<number>`count(*)` }).from(immuneMarkers).where(eq(immuneMarkers.patientId, patientId));
      heatmap[patientId].immuneMarkers = Math.min(100, Number(immuneCount[0]?.count || 0) * 15);

      const painCount = await db.select({ count: sql<number>`count(*)` }).from(paintrackSessions).where(eq(paintrackSessions.userId, patientId));
      heatmap[patientId].painTracking = Math.min(100, Number(painCount[0]?.count || 0) * 15);

      const symptomCount = await db.select({ count: sql<number>`count(*)` }).from(symptomCheckins).where(eq(symptomCheckins.userId, patientId));
      heatmap[patientId].symptomJournal = Math.min(100, Number(symptomCount[0]?.count || 0) * 10);
    }

    await this.logAudit(context, 'data_quality_heatmap', 'research', studyId || 'all', {
      patientCount: Object.keys(heatmap).length,
      dataTypeCount: dataTypes.length,
    });

    return {
      dataTypes,
      patients: Object.keys(heatmap),
      heatmap,
    };
  }
}

export let researchService: ResearchService = new ResearchService();

export function initResearchService(storage: Storage) {
  researchService.setStorage(storage);
}
