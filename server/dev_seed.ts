/**
 * HIPAA Development Seed Script
 * 
 * This script populates the development database with SYNTHETIC,
 * de-identified data ONLY. It must NEVER be used with production
 * data, patient records, or any real PHI.
 * 
 * COMPLIANCE: All data generated here is completely fictional and
 * does not represent any real individuals or medical records.
 */

import { runConfigGuard } from './config_guard';
import { safeLogger } from './safe_logger';
import { db } from './db';
import { users, patientProfiles, doctorProfiles, dailyFollowups, medications } from '@shared/schema';
import { eq } from 'drizzle-orm';

// Synthetic data generators - all data is completely fictional
const SYNTHETIC_FIRST_NAMES = [
  'Alex', 'Jordan', 'Taylor', 'Morgan', 'Casey',
  'Riley', 'Quinn', 'Avery', 'Blake', 'Cameron',
  'Drew', 'Finley', 'Harper', 'Jamie', 'Kendall'
];

const SYNTHETIC_LAST_NAMES = [
  'Smith', 'Johnson', 'Williams', 'Brown', 'Jones',
  'Garcia', 'Miller', 'Davis', 'Rodriguez', 'Martinez',
  'Anderson', 'Taylor', 'Thomas', 'Moore', 'Jackson'
];

const SYNTHETIC_CITIES = [
  'Springfield', 'Riverside', 'Fairview', 'Georgetown', 'Madison',
  'Clinton', 'Franklin', 'Greenville', 'Bristol', 'Salem'
];

const SYNTHETIC_STATES = ['CA', 'TX', 'NY', 'FL', 'IL', 'PA', 'OH', 'GA', 'NC', 'MI'];

const SYNTHETIC_CONDITIONS = [
  'Seasonal allergies',
  'Mild hypertension',
  'Type 2 diabetes (well-controlled)',
  'Asthma',
  'Migraine headaches'
];

const SYNTHETIC_MEDICATIONS = [
  { name: 'Lisinopril', dosage: '10mg', frequency: 'Once daily' },
  { name: 'Metformin', dosage: '500mg', frequency: 'Twice daily' },
  { name: 'Omeprazole', dosage: '20mg', frequency: 'Once daily' },
  { name: 'Atorvastatin', dosage: '20mg', frequency: 'Once daily at bedtime' },
  { name: 'Amlodipine', dosage: '5mg', frequency: 'Once daily' }
];

const SYNTHETIC_SPECIALTIES = [
  'Internal Medicine',
  'Family Medicine',
  'Cardiology',
  'Endocrinology',
  'Pulmonology'
];

function generateSyntheticId(): string {
  return `dev-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
}

function randomElement<T>(arr: T[]): T {
  return arr[Math.floor(Math.random() * arr.length)];
}

function randomInt(min: number, max: number): number {
  return Math.floor(Math.random() * (max - min + 1)) + min;
}

function generateSyntheticEmail(firstName: string, lastName: string): string {
  const domain = 'dev-test.example.com';
  const randomNum = randomInt(100, 999);
  return `${firstName.toLowerCase()}.${lastName.toLowerCase()}${randomNum}@${domain}`;
}

function generateSyntheticPhone(): string {
  return `555-${randomInt(100, 999)}-${randomInt(1000, 9999)}`;
}

function generateSyntheticDOB(): Date {
  const year = randomInt(1950, 2000);
  const month = randomInt(0, 11);
  const day = randomInt(1, 28);
  return new Date(year, month, day);
}

function generateSyntheticAddress(): string {
  const num = randomInt(100, 9999);
  const streets = ['Main St', 'Oak Ave', 'Maple Dr', 'Cedar Ln', 'Park Rd'];
  return `${num} ${randomElement(streets)}`;
}

function generateSyntheticZip(): string {
  return `${randomInt(10000, 99999)}`;
}

interface SyntheticUser {
  id: string;
  email: string;
  firstName: string;
  lastName: string;
  role: 'patient' | 'doctor' | 'admin';
  phoneNumber: string;
  emailVerified: boolean;
}

function generateSyntheticUser(role: 'patient' | 'doctor' | 'admin'): SyntheticUser {
  const firstName = randomElement(SYNTHETIC_FIRST_NAMES);
  const lastName = randomElement(SYNTHETIC_LAST_NAMES);
  
  return {
    id: generateSyntheticId(),
    email: generateSyntheticEmail(firstName, lastName),
    firstName,
    lastName,
    role,
    phoneNumber: generateSyntheticPhone(),
    emailVerified: true
  };
}

export async function seedSyntheticData(options: { 
  patients?: number; 
  doctors?: number;
  clean?: boolean;
} = {}): Promise<void> {
  // CRITICAL: Run config guard before any seeding
  const guardResult = runConfigGuard(false);
  if (!guardResult.passed) {
    safeLogger.error('Config guard failed - cannot seed in production environment');
    throw new Error('Seeding blocked: production environment detected');
  }

  const patientCount = options.patients || 5;
  const doctorCount = options.doctors || 2;

  safeLogger.info('Starting synthetic data seed', { patientCount, doctorCount });

  try {
    // Generate and insert synthetic patients
    for (let i = 0; i < patientCount; i++) {
      const userData = generateSyntheticUser('patient');
      
      // Check if user with this email exists
      const existing = await db.select().from(users).where(eq(users.email, userData.email)).limit(1);
      if (existing.length > 0) {
        safeLogger.info('Skipping existing user');
        continue;
      }

      await db.insert(users).values({
        id: userData.id,
        email: userData.email,
        firstName: userData.firstName,
        lastName: userData.lastName,
        role: userData.role,
        phoneNumber: userData.phoneNumber,
        emailVerified: userData.emailVerified
      });

      // Create patient profile
      await db.insert(patientProfiles).values({
        id: generateSyntheticId(),
        userId: userData.id,
        followupPatientId: `FAI-DEV${randomInt(1000, 9999)}`,
        dateOfBirth: generateSyntheticDOB(),
        address: generateSyntheticAddress(),
        city: randomElement(SYNTHETIC_CITIES),
        state: randomElement(SYNTHETIC_STATES),
        zipCode: generateSyntheticZip(),
        country: 'USA',
        comorbidities: [randomElement(SYNTHETIC_CONDITIONS)],
        allergies: ['Penicillin'], // Common synthetic allergy
        currentMedications: [randomElement(SYNTHETIC_MEDICATIONS).name]
      });

      // Add some daily followups
      for (let j = 0; j < 3; j++) {
        const followupDate = new Date();
        followupDate.setDate(followupDate.getDate() - j);
        
        await db.insert(dailyFollowups).values({
          id: generateSyntheticId(),
          patientId: userData.id,
          date: followupDate,
          heartRate: randomInt(60, 100),
          bloodPressureSystolic: randomInt(110, 140),
          bloodPressureDiastolic: randomInt(70, 90),
          oxygenSaturation: randomInt(95, 100),
          sleepHours: String(randomInt(5, 9)),
          stepsCount: randomInt(2000, 10000),
          completed: true
        });
      }

      // Add medications
      const med = randomElement(SYNTHETIC_MEDICATIONS);
      await db.insert(medications).values({
        id: generateSyntheticId(),
        patientId: userData.id,
        name: med.name,
        dosage: med.dosage,
        frequency: med.frequency,
        isOTC: false,
        active: true,
        source: 'manual',
        status: 'active',
        addedBy: 'patient'
      });

      safeLogger.info(`Created synthetic patient ${i + 1}/${patientCount}`);
    }

    // Generate and insert synthetic doctors
    for (let i = 0; i < doctorCount; i++) {
      const userData = generateSyntheticUser('doctor');
      
      const existing = await db.select().from(users).where(eq(users.email, userData.email)).limit(1);
      if (existing.length > 0) {
        safeLogger.info('Skipping existing doctor');
        continue;
      }

      await db.insert(users).values({
        id: userData.id,
        email: userData.email,
        firstName: userData.firstName,
        lastName: userData.lastName,
        role: userData.role,
        phoneNumber: userData.phoneNumber,
        emailVerified: userData.emailVerified,
        organization: `${randomElement(SYNTHETIC_CITIES)} Medical Center`,
        medicalLicenseNumber: `DEV${randomInt(100000, 999999)}`,
        licenseCountry: 'USA',
        licenseVerified: true,
        adminVerified: true
      });

      // Create doctor profile
      await db.insert(doctorProfiles).values({
        id: generateSyntheticId(),
        userId: userData.id,
        specialties: [randomElement(SYNTHETIC_SPECIALTIES)],
        bio: 'This is a synthetic test doctor profile for development purposes only.',
        certifications: ['Board Certified'],
        education: [{ institution: 'Test Medical School', degree: 'MD', year: 2010 }]
      });

      safeLogger.info(`Created synthetic doctor ${i + 1}/${doctorCount}`);
    }

    safeLogger.info('Synthetic data seed completed successfully');

  } catch (error) {
    safeLogger.error('Synthetic data seed failed', { 
      error: error instanceof Error ? error.message : 'Unknown error' 
    });
    throw error;
  }
}

// CLI entry point
if (process.argv[1]?.includes('dev_seed')) {
  const args = process.argv.slice(2);
  const options: { patients?: number; doctors?: number } = {};
  
  for (let i = 0; i < args.length; i += 2) {
    if (args[i] === '--patients') options.patients = parseInt(args[i + 1]);
    if (args[i] === '--doctors') options.doctors = parseInt(args[i + 1]);
  }

  seedSyntheticData(options)
    .then(() => process.exit(0))
    .catch(() => process.exit(1));
}

export default seedSyntheticData;
