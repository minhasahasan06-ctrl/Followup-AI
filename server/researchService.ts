import type { Storage } from './storage';

interface ResearchQuery {
  resourceType: string;
  parameters: Record<string, string>;
  limit?: number;
}

interface AggregatedData {
  total: number;
  byCategory: Record<string, number>;
  trends: any[];
}

class ResearchService {
  private storage: Storage;

  constructor(storage: Storage) {
    this.storage = storage;
  }

  async queryFHIRData(query: ResearchQuery): Promise<any> {
    console.warn('AWS HealthLake integration requires proper AWS SDK setup. Returning mock data.');
    
    return {
      resourceType: 'Bundle',
      type: 'searchset',
      total: 0,
      entry: [],
      message: 'AWS HealthLake integration available - requires AWS_HEALTHLAKE_DATASTORE_ID configuration',
    };
  }

  async getEpidemiologicalData(
    condition: string,
    dateRange?: { start: Date; end: Date }
  ): Promise<AggregatedData> {
    const appointments = await this.storage.getAppointmentsByCondition?.(condition) || [];
    
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
      aggregated.byCategory[category] =
        (aggregated.byCategory[category] || 0) + 1;
    }

    aggregated.trends = Object.entries(byMonth)
      .map(([month, count]) => ({ month, count }))
      .sort((a, b) => a.month.localeCompare(b.month));

    return aggregated;
  }

  async getPopulationHealthMetrics(doctorId?: string): Promise<any> {
    const patients = doctorId
      ? await this.storage.getPatientsByDoctor?.(doctorId) || []
      : await this.storage.getAllPatients();

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
        metrics.demographics.ageGroups[ageGroup] =
          (metrics.demographics.ageGroups[ageGroup] || 0) + 1;
      }

      if (patient.profile?.gender) {
        const gender = patient.profile.gender;
        metrics.demographics.genderDistribution[gender] =
          (metrics.demographics.genderDistribution[gender] || 0) + 1;
      }
    }

    return metrics;
  }

  async generateResearchReport(
    studyType: string,
    parameters: Record<string, any>
  ): Promise<any> {
    const report = {
      studyType,
      generatedAt: new Date(),
      parameters,
      findings: {},
      visualizations: [],
      recommendations: [],
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
        report.findings = await this.analyzeMedicationEffectiveness(
          parameters.medicationName
        );
        break;

      case 'population_health':
        report.findings = await this.getPopulationHealthMetrics(parameters.doctorId);
        break;

      default:
        throw new Error(`Unsupported study type: ${studyType}`);
    }

    return report;
  }

  private async analyzeMedicationEffectiveness(
    medicationName: string
  ): Promise<any> {
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

    if (
      monthDiff < 0 ||
      (monthDiff === 0 && today.getDate() < dateOfBirth.getDate())
    ) {
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
}

export let researchService: ResearchService;

export function initResearchService(storage: Storage) {
  researchService = new ResearchService(storage);
}
