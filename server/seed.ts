/**
 * Database Seed Script - Creates sample data for Lysa AI features
 * This ensures all features work out-of-the-box with realistic test data
 */

import { db } from "./db";
import * as schema from "@shared/schema";
import { eq, and } from "drizzle-orm";
import crypto from "crypto";
import bcrypt from "bcryptjs";

const generateId = () => crypto.randomUUID();

// Sample patient data for realistic testing
const samplePatients = [
  {
    firstName: "Sarah",
    lastName: "Johnson",
    email: "sarah.johnson@email.com",
    phoneNumber: "+1-555-0101",
    dateOfBirth: "1985-03-15",
    allergies: ["Penicillin", "Sulfa drugs"],
    comorbidities: ["Type 2 Diabetes", "Hypertension"],
    immunocompromisedCondition: "Post-transplant immunosuppression"
  },
  {
    firstName: "Robert",
    lastName: "Martinez",
    email: "robert.martinez@email.com",
    phoneNumber: "+1-555-0103",
    dateOfBirth: "1978-07-22",
    allergies: ["Aspirin"],
    comorbidities: ["Chronic Kidney Disease Stage 3", "Anemia"],
    immunocompromisedCondition: "Dialysis patient"
  },
  {
    firstName: "Emily",
    lastName: "Chen",
    email: "emily.chen@email.com",
    phoneNumber: "+1-555-0105",
    dateOfBirth: "1992-11-08",
    allergies: ["Latex", "Iodine contrast"],
    comorbidities: ["Lupus", "Rheumatoid Arthritis"],
    immunocompromisedCondition: "Autoimmune disease on immunosuppressants"
  },
  {
    firstName: "James",
    lastName: "Williams",
    email: "james.williams@email.com",
    phoneNumber: "+1-555-0107",
    dateOfBirth: "1965-01-30",
    allergies: [],
    comorbidities: ["COPD", "Heart Failure NYHA Class II"],
    immunocompromisedCondition: "Cancer survivor"
  },
  {
    firstName: "Maria",
    lastName: "Garcia",
    email: "maria.garcia@email.com",
    phoneNumber: "+1-555-0109",
    dateOfBirth: "1988-09-12",
    allergies: ["Shellfish", "Tree nuts"],
    comorbidities: ["Multiple Sclerosis"],
    immunocompromisedCondition: "MS on disease-modifying therapy"
  }
];

// Sample doctor data
const sampleDoctors = [
  {
    firstName: "Dr. Amanda",
    lastName: "Thompson",
    email: "dr.thompson@clinic.com",
    phoneNumber: "+1-555-0201",
    specialties: ["Internal Medicine", "Immunology"],
    bio: "Board-certified internist specializing in immunocompromised patient care"
  },
  {
    firstName: "Dr. Michael",
    lastName: "Lee",
    email: "dr.lee@clinic.com",
    phoneNumber: "+1-555-0202",
    specialties: ["Oncology", "Hematology"],
    bio: "Oncologist with 15 years experience in cancer care and immunotherapy"
  }
];

export async function seedDatabase() {
  console.log("🌱 Starting database seed...");

  try {
    // Check if seed data already exists
    const existingPatients = await db
      .select()
      .from(schema.users)
      .where(eq(schema.users.email, "sarah.johnson@email.com"))
      .limit(1);

    if (existingPatients.length > 0) {
      console.log("✅ Seed data already exists, skipping...");
      return { success: true, message: "Seed data already exists" };
    }

    const hashedPassword = await bcrypt.hash("TestPassword123!", 10);
    const patientIds: string[] = [];
    const doctorIds: string[] = [];

    // Create patients
    console.log("👤 Creating sample patients...");
    for (const patient of samplePatients) {
      const userId = generateId();
      patientIds.push(userId);

      await db.insert(schema.users).values({
        id: userId,
        email: patient.email,
        firstName: patient.firstName,
        lastName: patient.lastName,
        phoneNumber: patient.phoneNumber,
        role: "patient",
        emailVerified: true,
        adminVerified: true
      });

      await db.insert(schema.patientProfiles).values({
        id: generateId(),
        userId: userId,
        followupPatientId: `FP-${Date.now().toString(36).toUpperCase()}-${Math.random().toString(36).substring(2, 6).toUpperCase()}`,
        dateOfBirth: new Date(patient.dateOfBirth),
        allergies: patient.allergies,
        comorbidities: patient.comorbidities,
        immunocompromisedCondition: patient.immunocompromisedCondition
      });

      console.log(`  ✓ Created patient: ${patient.firstName} ${patient.lastName}`);
    }

    // Create doctors
    console.log("👨‍⚕️ Creating sample doctors...");
    for (const doctor of sampleDoctors) {
      const userId = generateId();
      doctorIds.push(userId);

      await db.insert(schema.users).values({
        id: userId,
        email: doctor.email,
        firstName: doctor.firstName,
        lastName: doctor.lastName,
        phoneNumber: doctor.phoneNumber,
        role: "doctor",
        emailVerified: true,
        licenseVerified: true,
        adminVerified: true
      });

      await db.insert(schema.doctorProfiles).values({
        id: generateId(),
        userId: userId,
        specialties: doctor.specialties,
        bio: doctor.bio
      });

      console.log(`  ✓ Created doctor: ${doctor.firstName} ${doctor.lastName}`);
    }

    // Create doctor-patient assignments (HIPAA compliant relationships)
    console.log("🔗 Creating doctor-patient assignments...");
    for (const doctorId of doctorIds) {
      for (const patientId of patientIds) {
        await db.insert(schema.doctorPatientAssignments).values({
          id: generateId(),
          doctorId: doctorId,
          patientId: patientId,
          assignedBy: doctorId,
          status: "active",
          assignmentSource: "manual",
          accessScope: "full",
          patientConsented: true,
          consentedAt: new Date(),
          consentMethod: "in_app"
        });
      }
    }
    console.log(`  ✓ Created ${doctorIds.length * patientIds.length} assignments`);

    // Create sample appointments
    console.log("📅 Creating sample appointments...");
    const appointmentTypes = ["consultation", "followup", "virtual", "in-person"];
    const appointmentTitles = [
      "Quarterly diabetes check-up",
      "Blood pressure monitoring",
      "Medication review",
      "Symptom assessment",
      "Lab results discussion"
    ];

    for (let i = 0; i < patientIds.length; i++) {
      const patientId = patientIds[i];
      const doctorId = doctorIds[i % doctorIds.length];

      // Past appointment
      const pastStart = new Date(Date.now() - 7 * 24 * 60 * 60 * 1000);
      pastStart.setHours(9, 0, 0, 0);
      const pastEnd = new Date(pastStart);
      pastEnd.setMinutes(pastEnd.getMinutes() + 30);
      
      await db.insert(schema.appointments).values({
        id: generateId(),
        patientId: patientId,
        doctorId: doctorId,
        title: appointmentTitles[i % appointmentTitles.length],
        description: "Routine check-up appointment",
        appointmentType: appointmentTypes[i % appointmentTypes.length],
        startTime: pastStart,
        endTime: pastEnd,
        duration: 30,
        status: "completed"
      });

      // Upcoming appointment
      const futureStart = new Date(Date.now() + 7 * 24 * 60 * 60 * 1000);
      futureStart.setHours(10, 0, 0, 0);
      const futureEnd = new Date(futureStart);
      futureEnd.setMinutes(futureEnd.getMinutes() + 30);
      
      await db.insert(schema.appointments).values({
        id: generateId(),
        patientId: patientId,
        doctorId: doctorId,
        title: "Follow-up visit",
        description: "Follow-up appointment to review progress",
        appointmentType: "followup",
        startTime: futureStart,
        endTime: futureEnd,
        duration: 30,
        status: "confirmed"
      });
    }
    console.log(`  ✓ Created ${patientIds.length * 2} appointments`);

    // Create sample prescriptions
    console.log("💊 Creating sample prescriptions...");
    const medicationsList = [
      { name: "Metformin", dosage: "500mg", frequency: "twice daily", instructions: "Take with meals" },
      { name: "Lisinopril", dosage: "10mg", frequency: "once daily", instructions: "Take in the morning" },
      { name: "Atorvastatin", dosage: "20mg", frequency: "once daily", instructions: "Take at bedtime" },
      { name: "Prednisone", dosage: "10mg", frequency: "once daily", instructions: "Take with food" },
      { name: "Omeprazole", dosage: "20mg", frequency: "once daily", instructions: "Take before breakfast" }
    ];

    for (let i = 0; i < patientIds.length; i++) {
      const patientId = patientIds[i];
      const doctorId = doctorIds[i % doctorIds.length];
      const med = medicationsList[i % medicationsList.length];

      await db.insert(schema.prescriptions).values({
        id: generateId(),
        patientId: patientId,
        doctorId: doctorId,
        medicationName: med.name,
        dosage: med.dosage,
        frequency: med.frequency,
        dosageInstructions: med.instructions,
        quantity: 30,
        refills: 2,
        startDate: new Date(),
        status: "sent",
        notes: `Standard ${med.name} prescription for patient care`
      });
    }
    console.log(`  ✓ Created ${patientIds.length} prescriptions`);

    console.log("🎉 Database seed completed successfully!");
    return { 
      success: true, 
      message: "Seed completed",
      stats: {
        patients: patientIds.length,
        doctors: doctorIds.length,
        assignments: doctorIds.length * patientIds.length,
        appointments: patientIds.length * 2,
        prescriptions: patientIds.length
      }
    };

  } catch (error) {
    console.error("❌ Seed failed:", error);
    throw error;
  }
}

// Auto-run on import
if (process.env.AUTO_SEED === "true") {
  seedDatabase().catch(console.error);
}
