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

interface AuditContext {
  userId: string;
  ipAddress?: string;
  userAgent?: string;
}

class ResearchService {
  private storage?: Storage;
  private auditContext?: AuditContext;

  constructor(storage?: Storage) {
    this.storage = storage;
  }

  setStorage(storage: Storage) {
    this.storage = storage;
  }

  setAuditContext(context: AuditContext) {
    this.auditContext = context;
  }

  // HIPAA Audit logging helper - logs all research data access and mutations
  private async logAudit(
    actionType: string,
    objectType: string,
    objectId: string,
    details?: Record<string, any>,
    userId?: string
  ): Promise<void> {
    try {
      const auditUserId = userId || this.auditContext?.userId || 'system';
      await db.insert(researchAuditLogs).values({
        userId: auditUserId,
        actionType,
        objectType,
        objectId,
        details: details || {},
        ipAddress: this.auditContext?.ipAddress,
        userAgent: this.auditContext?.userAgent,
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

  async createResearchDataConsent(consent: InsertResearchDataConsent): Promise<ResearchDataConsent> {
    const [created] = await db.insert(researchDataConsent).values(consent).returning();
    await this.logAudit('CREATE', 'ResearchDataConsent', created.id, {
      patientId: consent.patientId,
      consentEnabled: consent.consentEnabled,
    }, consent.patientId);
    return created;
  }

  async updateResearchDataConsent(
    patientId: string, 
    data: Partial<InsertResearchDataConsent>
  ): Promise<ResearchDataConsent | undefined> {
    const [updated] = await db
      .update(researchDataConsent)
      .set({ ...data, updatedAt: new Date(), consentUpdatedAt: new Date() })
      .where(eq(researchDataConsent.patientId, patientId))
      .returning();
    if (updated) {
      await this.logAudit('UPDATE', 'ResearchDataConsent', updated.id, {
        patientId,
        consentEnabled: data.consentEnabled,
        dataTypePermissions: data.dataTypePermissions,
      }, patientId);
    }
    return updated;
  }

  async upsertResearchDataConsent(consent: InsertResearchDataConsent): Promise<ResearchDataConsent> {
    const existing = await this.getResearchDataConsent(consent.patientId);
    if (existing) {
      const updated = await this.updateResearchDataConsent(consent.patientId, consent);
      return updated!;
    }
    return this.createResearchDataConsent(consent);
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

  async createResearchProject(project: InsertResearchProject): Promise<ResearchProject> {
    const [created] = await db.insert(researchProjects).values(project).returning();
    await this.logAudit('CREATE', 'ResearchProject', created.id, {
      name: project.name,
      ownerId: project.ownerId,
    });
    return created;
  }

  async updateResearchProject(id: string, data: Partial<InsertResearchProject>): Promise<ResearchProject | undefined> {
    const [updated] = await db
      .update(researchProjects)
      .set({ ...data, updatedAt: new Date() })
      .where(eq(researchProjects.id, id))
      .returning();
    if (updated) {
      await this.logAudit('UPDATE', 'ResearchProject', id, { changedFields: Object.keys(data) });
    }
    return updated;
  }

  async deleteResearchProject(id: string): Promise<boolean> {
    await this.logAudit('DELETE', 'ResearchProject', id, {});
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

  async createResearchStudy(study: InsertResearchStudy): Promise<ResearchStudy> {
    const [created] = await db.insert(researchStudies).values(study).returning();
    await this.logAudit('CREATE', 'ResearchStudy', created.id, {
      title: study.title,
      ownerId: study.ownerUserId,
      projectId: study.projectId,
    });
    return created;
  }

  async updateResearchStudy(id: string, data: Partial<InsertResearchStudy>): Promise<ResearchStudy | undefined> {
    const [updated] = await db
      .update(researchStudies)
      .set({ ...data, updatedAt: new Date() })
      .where(eq(researchStudies.id, id))
      .returning();
    if (updated) {
      await this.logAudit('UPDATE', 'ResearchStudy', id, { changedFields: Object.keys(data), status: data.status });
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

  async createResearchCohort(cohort: InsertResearchCohort): Promise<ResearchCohort> {
    const [created] = await db.insert(researchCohorts).values(cohort).returning();
    await this.logAudit('CREATE', 'ResearchCohort', created.id, {
      name: cohort.name,
      createdBy: cohort.createdBy,
      projectId: cohort.projectId,
    });
    return created;
  }

  async updateResearchCohort(id: string, data: Partial<InsertResearchCohort>): Promise<ResearchCohort | undefined> {
    const [updated] = await db
      .update(researchCohorts)
      .set({ ...data, updatedAt: new Date() })
      .where(eq(researchCohorts.id, id))
      .returning();
    if (updated) {
      await this.logAudit('UPDATE', 'ResearchCohort', id, { changedFields: Object.keys(data), status: data.status });
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

  async createStudyEnrollment(enrollment: InsertStudyEnrollment): Promise<StudyEnrollment> {
    const [created] = await db.insert(studyEnrollments).values(enrollment).returning();
    
    await db
      .update(researchStudies)
      .set({ 
        currentEnrollment: sql`${researchStudies.currentEnrollment} + 1`,
        updatedAt: new Date()
      })
      .where(eq(researchStudies.id, enrollment.studyId));
    
    await this.logAudit('CREATE', 'StudyEnrollment', created.id, {
      studyId: enrollment.studyId,
      patientId: enrollment.patientId,
    });
    return created;
  }

  async updateStudyEnrollment(id: string, data: Partial<InsertStudyEnrollment>): Promise<StudyEnrollment | undefined> {
    const [updated] = await db
      .update(studyEnrollments)
      .set({ ...data, updatedAt: new Date() })
      .where(eq(studyEnrollments.id, id))
      .returning();
    if (updated) {
      await this.logAudit('UPDATE', 'StudyEnrollment', id, { changedFields: Object.keys(data), status: data.status });
    }
    return updated;
  }

  async withdrawFromStudy(id: string, reason: string): Promise<StudyEnrollment | undefined> {
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
    
    await this.logAudit('WITHDRAW', 'StudyEnrollment', id, { 
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

  async createResearchVisit(visit: InsertResearchVisit): Promise<ResearchVisit> {
    const [created] = await db.insert(researchVisits).values(visit).returning();
    await this.logAudit('CREATE', 'ResearchVisit', created.id, {
      patientId: visit.patientId,
      studyId: visit.studyId,
      visitType: visit.visitType,
    });
    return created;
  }

  async updateResearchVisit(id: string, data: Partial<InsertResearchVisit>): Promise<ResearchVisit | undefined> {
    const [updated] = await db
      .update(researchVisits)
      .set({ ...data, updatedAt: new Date() })
      .where(eq(researchVisits.id, id))
      .returning();
    if (updated) {
      await this.logAudit('UPDATE', 'ResearchVisit', id, { changedFields: Object.keys(data), status: data.visitStatus });
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

  async createResearchMeasurement(measurement: InsertResearchMeasurement): Promise<ResearchMeasurement> {
    const [created] = await db.insert(researchMeasurements).values(measurement).returning();
    await this.logAudit('CREATE', 'ResearchMeasurement', created.id, {
      patientId: measurement.patientId,
      name: measurement.name,
      category: measurement.category,
    });
    return created;
  }

  async createResearchMeasurements(measurements: InsertResearchMeasurement[]): Promise<ResearchMeasurement[]> {
    if (measurements.length === 0) return [];
    const created = await db.insert(researchMeasurements).values(measurements).returning();
    await this.logAudit('BULK_CREATE', 'ResearchMeasurement', 'bulk', {
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

  async createResearchImmuneMarker(marker: InsertResearchImmuneMarker): Promise<ResearchImmuneMarker> {
    const [created] = await db.insert(researchImmuneMarkers).values(marker).returning();
    await this.logAudit('CREATE', 'ResearchImmuneMarker', created.id, {
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

  async createResearchAnalysisReport(report: InsertResearchAnalysisReport): Promise<ResearchAnalysisReport> {
    const [created] = await db.insert(researchAnalysisReports).values(report).returning();
    await this.logAudit('CREATE', 'ResearchAnalysisReport', created.id, {
      title: report.title,
      analysisType: report.analysisType,
      studyId: report.studyId,
    });
    return created;
  }

  async updateResearchAnalysisReport(id: string, data: Partial<InsertResearchAnalysisReport>): Promise<ResearchAnalysisReport | undefined> {
    const [updated] = await db
      .update(researchAnalysisReports)
      .set({ ...data, updatedAt: new Date() })
      .where(eq(researchAnalysisReports.id, id))
      .returning();
    if (updated) {
      await this.logAudit('UPDATE', 'ResearchAnalysisReport', id, { changedFields: Object.keys(data), status: data.status });
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

  async createResearchAlert(alert: InsertResearchAlert): Promise<ResearchAlert> {
    const [created] = await db.insert(researchAlerts).values(alert).returning();
    await this.logAudit('CREATE', 'ResearchAlert', created.id, {
      patientId: alert.patientId,
      alertType: alert.alertType,
      severity: alert.severity,
    });
    return created;
  }

  async acknowledgeResearchAlert(id: string, userId: string): Promise<ResearchAlert | undefined> {
    const [updated] = await db
      .update(researchAlerts)
      .set({ status: "acknowledged", acknowledgedAt: new Date(), acknowledgedBy: userId })
      .where(eq(researchAlerts.id, id))
      .returning();
    if (updated) {
      await this.logAudit('ACKNOWLEDGE', 'ResearchAlert', id, { acknowledgedBy: userId });
    }
    return updated;
  }

  async resolveResearchAlert(id: string, userId: string, resolution: string): Promise<ResearchAlert | undefined> {
    const [updated] = await db
      .update(researchAlerts)
      .set({ status: "resolved", resolvedAt: new Date(), resolvedBy: userId, resolution })
      .where(eq(researchAlerts.id, id))
      .returning();
    if (updated) {
      await this.logAudit('RESOLVE', 'ResearchAlert', id, { resolvedBy: userId, resolution });
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

  async createDailyFollowupTemplate(template: InsertDailyFollowupTemplate): Promise<DailyFollowupTemplate> {
    const [created] = await db.insert(dailyFollowupTemplates).values(template).returning();
    await this.logAudit('CREATE', 'DailyFollowupTemplate', created.id, {
      name: template.name,
      createdBy: template.createdBy,
    });
    return created;
  }

  async updateDailyFollowupTemplate(id: string, data: Partial<InsertDailyFollowupTemplate>): Promise<DailyFollowupTemplate | undefined> {
    const [updated] = await db
      .update(dailyFollowupTemplates)
      .set({ ...data, updatedAt: new Date() })
      .where(eq(dailyFollowupTemplates.id, id))
      .returning();
    if (updated) {
      await this.logAudit('UPDATE', 'DailyFollowupTemplate', id, { changedFields: Object.keys(data) });
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

  async createDailyFollowupAssignment(assignment: InsertDailyFollowupAssignment): Promise<DailyFollowupAssignment> {
    const [created] = await db.insert(dailyFollowupAssignments).values(assignment).returning();
    await this.logAudit('CREATE', 'DailyFollowupAssignment', created.id, {
      patientId: assignment.patientId,
      templateId: assignment.templateId,
      studyId: assignment.studyId,
    });
    return created;
  }

  async updateDailyFollowupAssignment(id: string, data: Partial<InsertDailyFollowupAssignment>): Promise<DailyFollowupAssignment | undefined> {
    const [updated] = await db
      .update(dailyFollowupAssignments)
      .set(data)
      .where(eq(dailyFollowupAssignments.id, id))
      .returning();
    if (updated) {
      await this.logAudit('UPDATE', 'DailyFollowupAssignment', id, { changedFields: Object.keys(data), isActive: data.isActive });
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

  async createDailyFollowupResponse(response: InsertDailyFollowupResponse): Promise<DailyFollowupResponse> {
    const [created] = await db.insert(dailyFollowupResponses).values(response).returning();
    await this.logAudit('CREATE', 'DailyFollowupResponse', created.id, {
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

  async createAnalysisJob(job: InsertAnalysisJob): Promise<AnalysisJob> {
    const [created] = await db.insert(analysisJobs).values(job).returning();
    await this.logAudit('CREATE', 'AnalysisJob', created.id, {
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

  async updateAnalysisJob(id: string, data: Partial<InsertAnalysisJob>): Promise<AnalysisJob | undefined> {
    const [updated] = await db
      .update(analysisJobs)
      .set(data)
      .where(eq(analysisJobs.id, id))
      .returning();
    if (updated) {
      await this.logAudit('UPDATE', 'AnalysisJob', id, { status: data.status, changedFields: Object.keys(data) });
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
}

export let researchService: ResearchService = new ResearchService();

export function initResearchService(storage: Storage) {
  researchService.setStorage(storage);
}
