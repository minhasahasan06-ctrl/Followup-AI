#!/usr/bin/env python3
"""
Create database tables for the new doctor search and consultation features.
This script creates tables without dropping existing ones.
"""

from app.database import Base, engine
from app.models import (
    User,
    Hospital,
    Specialty,
    DoctorSpecialty,
    PatientDoctorConnection,
    PatientConsultation,
    AISymptomSession,
)

print("Creating new database tables...")
print("Models to create:")
print("- hospitals")
print("- specialties")
print("- doctor_specialties")
print("- patient_doctor_connections")
print("- patient_consultations")
print("- ai_symptom_sessions")
print()

try:
    Base.metadata.create_all(bind=engine)
    print("✅ Database tables created successfully!")
    print()
    print("The following tables are now available:")
    for table_name in Base.metadata.tables.keys():
        print(f"  - {table_name}")
except Exception as e:
    print(f"❌ Error creating tables: {e}")
    raise
