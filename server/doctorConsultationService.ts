import type { Storage } from './storage';
import { randomBytes } from 'crypto';

interface ConsultationRequest {
  requestingDoctorId: string;
  consultingDoctorId: string;
  patientId: string;
  reason: string;
  urgency: 'routine' | 'urgent' | 'emergency';
  shareRecordTypes: string[];
  expiresAt?: Date;
}

interface ConsultationAccess {
  id: string;
  requestingDoctorId: string;
  consultingDoctorId: string;
  patientId: string;
  reason: string;
  status: 'pending' | 'approved' | 'denied' | 'expired';
  shareRecordTypes: string[];
  accessToken: string;
  requestedAt: Date;
  respondedAt?: Date;
  expiresAt: Date;
}

class DoctorConsultationService {
  private storage: Storage;
  private consultations: Map<string, ConsultationAccess> = new Map();

  constructor(storage: Storage) {
    this.storage = storage;
  }

  async requestConsultation(request: ConsultationRequest): Promise<ConsultationAccess> {
    const requestingDoctor = await this.storage.getUser(request.requestingDoctorId);
    const consultingDoctor = await this.storage.getUser(request.consultingDoctorId);
    const patient = await this.storage.getUser(request.patientId);

    if (!requestingDoctor || !consultingDoctor || !patient) {
      throw new Error('Invalid doctor or patient ID');
    }

    if (requestingDoctor.role !== 'doctor' || consultingDoctor.role !== 'doctor') {
      throw new Error('Both users must be doctors');
    }

    const accessToken = randomBytes(32).toString('hex');
    const expiresAt = request.expiresAt || new Date(Date.now() + 7 * 24 * 60 * 60 * 1000);

    const consultation: ConsultationAccess = {
      id: randomBytes(16).toString('hex'),
      requestingDoctorId: request.requestingDoctorId,
      consultingDoctorId: request.consultingDoctorId,
      patientId: request.patientId,
      reason: request.reason,
      status: 'pending',
      shareRecordTypes: request.shareRecordTypes,
      accessToken,
      requestedAt: new Date(),
      expiresAt,
    };

    this.consultations.set(consultation.id, consultation);

    await this.notifyConsultingDoctor(consultation, consultingDoctor);

    return consultation;
  }

  async approveConsultation(
    consultationId: string,
    consultingDoctorId: string
  ): Promise<ConsultationAccess> {
    const consultation = this.consultations.get(consultationId);

    if (!consultation) {
      throw new Error('Consultation request not found');
    }

    if (consultation.consultingDoctorId !== consultingDoctorId) {
      throw new Error('Unauthorized to approve this consultation');
    }

    consultation.status = 'approved';
    consultation.respondedAt = new Date();
    this.consultations.set(consultationId, consultation);

    return consultation;
  }

  async denyConsultation(
    consultationId: string,
    consultingDoctorId: string,
    reason?: string
  ): Promise<ConsultationAccess> {
    const consultation = this.consultations.get(consultationId);

    if (!consultation) {
      throw new Error('Consultation request not found');
    }

    if (consultation.consultingDoctorId !== consultingDoctorId) {
      throw new Error('Unauthorized to deny this consultation');
    }

    consultation.status = 'denied';
    consultation.respondedAt = new Date();
    this.consultations.set(consultationId, consultation);

    return consultation;
  }

  async getPatientRecords(
    accessToken: string,
    recordTypes: string[]
  ): Promise<any> {
    const consultation = Array.from(this.consultations.values()).find(
      (c) => c.accessToken === accessToken && c.status === 'approved'
    );

    if (!consultation) {
      throw new Error('Invalid or unauthorized access token');
    }

    if (new Date() > consultation.expiresAt) {
      consultation.status = 'expired';
      throw new Error('Consultation access has expired');
    }

    const patient = await this.storage.getUser(consultation.patientId);
    if (!patient) {
      throw new Error('Patient not found');
    }

    const allowedRecords = consultation.shareRecordTypes;
    const requestedRecords = recordTypes.filter((type) =>
      allowedRecords.includes(type)
    );

    const records: any = {
      patient: {
        id: patient.id,
        firstName: patient.firstName,
        lastName: patient.lastName,
        dateOfBirth: patient.dateOfBirth,
        email: patient.email,
        phoneNumber: patient.phoneNumber,
      },
      requestedBy: consultation.requestingDoctorId,
      expiresAt: consultation.expiresAt,
      records: {},
    };

    for (const recordType of requestedRecords) {
      switch (recordType) {
        case 'medications':
          records.records.medications = await this.storage.getActiveMedicationsByPatient(consultation.patientId);
          break;
        case 'allergies':
          records.records.allergies = await this.storage.getAllergies(consultation.patientId);
          break;
        case 'appointments':
          records.records.appointments = await this.storage.getAppointmentsByPatient(consultation.patientId);
          break;
        case 'followups':
          records.records.followups = await this.storage.getFollowupsByPatient(consultation.patientId);
          break;
        case 'labs':
          records.records.labs = await this.getLabResults(consultation.patientId);
          break;
      }
    }

    return records;
  }

  async getConsultationsByDoctor(doctorId: string): Promise<ConsultationAccess[]> {
    return Array.from(this.consultations.values()).filter(
      (c) =>
        c.requestingDoctorId === doctorId || c.consultingDoctorId === doctorId
    );
  }

  async revokeConsultation(consultationId: string, doctorId: string): Promise<void> {
    const consultation = this.consultations.get(consultationId);

    if (!consultation) {
      throw new Error('Consultation not found');
    }

    if (
      consultation.requestingDoctorId !== doctorId &&
      consultation.consultingDoctorId !== doctorId
    ) {
      throw new Error('Unauthorized to revoke this consultation');
    }

    consultation.status = 'expired';
    consultation.expiresAt = new Date();
  }

  private async notifyConsultingDoctor(
    consultation: ConsultationAccess,
    consultingDoctor: any
  ): Promise<void> {
    console.log(
      `[NOTIFICATION] Dr. ${consultingDoctor.firstName} ${consultingDoctor.lastName} has a new consultation request for patient ${consultation.patientId}`
    );
  }

  private async getLabResults(patientId: string): Promise<any[]> {
    return [];
  }
}

export let doctorConsultationService: DoctorConsultationService;

export function initDoctorConsultationService(storage: Storage) {
  doctorConsultationService = new DoctorConsultationService(storage);
}
