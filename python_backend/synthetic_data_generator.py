"""
Synthetic Data Generator for Enhanced Research Center
=======================================================
** DEMO/TESTING ENVIRONMENT ONLY **

This script generates SYNTHETIC test data for demonstrations and testing.
All generated data is clearly marked as synthetic and uses fake email domains.

SAFETY GUARDRAILS:
1. Requires ALLOW_SYNTHETIC_DATA=true environment variable
2. Uses @synthetic.example.com email domain (not real)
3. All names are clearly fake test names
4. Adds [SYNTHETIC] prefix to descriptions
5. Should NEVER be run against production databases

Generates test data for:
- research_patients (immune markers, demographics)
- studies with cohorts and enrollments
- daily_followup_templates and responses
- research_alerts
- ml_analyses and reports
- environmental exposures
- research_consent settings
"""

import os
import random
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import psycopg2
from psycopg2.extras import execute_values
import json
import sys

DATABASE_URL = os.environ.get("DATABASE_URL")
ALLOW_SYNTHETIC_DATA = os.environ.get("ALLOW_SYNTHETIC_DATA", "").lower() == "true"

# Safety check: require explicit opt-in
if not ALLOW_SYNTHETIC_DATA:
    print("=" * 70)
    print("SAFETY BLOCK: Synthetic data generation is disabled by default.")
    print("To enable, set environment variable: ALLOW_SYNTHETIC_DATA=true")
    print("WARNING: Only run this on development/demo databases, never production!")
    print("=" * 70)
    sys.exit(1)

FIRST_NAMES = [
    "James", "Mary", "John", "Patricia", "Robert", "Jennifer", "Michael", "Linda",
    "William", "Barbara", "David", "Elizabeth", "Richard", "Susan", "Joseph", "Jessica",
    "Thomas", "Sarah", "Charles", "Karen", "Christopher", "Lisa", "Daniel", "Nancy",
    "Matthew", "Betty", "Anthony", "Margaret", "Mark", "Sandra"
]

LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
    "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson",
    "Thomas", "Taylor", "Moore", "Jackson", "Martin", "Lee", "Perez", "Thompson",
    "White", "Harris", "Sanchez", "Clark", "Ramirez", "Lewis", "Robinson"
]

CONDITIONS = [
    "Rheumatoid Arthritis", "Lupus", "Multiple Sclerosis", "Crohn's Disease",
    "Ulcerative Colitis", "Psoriatic Arthritis", "Sjogren's Syndrome",
    "Type 1 Diabetes", "Hashimoto's Thyroiditis", "Celiac Disease",
    "Myasthenia Gravis", "Systemic Sclerosis", "Vasculitis", "Dermatomyositis"
]

MEDICATIONS = [
    "Methotrexate", "Prednisone", "Humira", "Enbrel", "Remicade", "Rituximab",
    "Azathioprine", "Cyclophosphamide", "Mycophenolate", "Tacrolimus",
    "Hydroxychloroquine", "Sulfasalazine", "Leflunomide", "Cyclosporine"
]

SYMPTOMS = [
    "Fatigue", "Joint Pain", "Muscle Weakness", "Rash", "Fever", "Weight Loss",
    "Dry Eyes", "Dry Mouth", "Numbness", "Tingling", "Swelling", "Stiffness",
    "Brain Fog", "Insomnia", "Nausea", "Abdominal Pain", "Headache", "Dizziness"
]

LOCATIONS = [
    ("Boston", "MA", 42.3601, -71.0589, "urban", 7),
    ("New York", "NY", 40.7128, -74.0060, "urban", 8),
    ("Los Angeles", "CA", 34.0522, -118.2437, "urban", 6),
    ("Chicago", "IL", 41.8781, -87.6298, "urban", 7),
    ("Houston", "TX", 29.7604, -95.3698, "urban", 5),
    ("Phoenix", "AZ", 33.4484, -112.0740, "urban", 4),
    ("Denver", "CO", 39.7392, -104.9903, "suburban", 3),
    ("Seattle", "WA", 47.6062, -122.3321, "urban", 6),
    ("Portland", "OR", 45.5152, -122.6784, "suburban", 5),
    ("Rural Vermont", "VT", 44.5588, -72.5778, "rural", 2)
]


def get_connection():
    return psycopg2.connect(DATABASE_URL)


def generate_uuid():
    return str(uuid.uuid4())


def random_date(start_days_ago: int, end_days_ago: int = 0) -> datetime:
    days_ago = random.randint(end_days_ago, start_days_ago)
    return datetime.now() - timedelta(days=days_ago)


def generate_immune_markers() -> Dict[str, Any]:
    return {
        "wbc": round(random.uniform(3.5, 11.0), 1),
        "lymphocytes": round(random.uniform(1.0, 4.0), 1),
        "neutrophils": round(random.uniform(1.5, 7.0), 1),
        "cd4_count": random.randint(300, 1200),
        "cd8_count": random.randint(200, 800),
        "cd4_cd8_ratio": round(random.uniform(0.5, 3.0), 2),
        "crp": round(random.uniform(0.1, 15.0), 1),
        "esr": random.randint(1, 80),
        "ana_titer": random.choice(["negative", "1:40", "1:80", "1:160", "1:320"]),
        "rf_factor": round(random.uniform(0, 100), 1),
        "anti_ccp": round(random.uniform(0, 300), 1),
        "complement_c3": round(random.uniform(80, 180), 0),
        "complement_c4": round(random.uniform(15, 45), 0),
        "igG": round(random.uniform(700, 1600), 0),
        "igA": round(random.uniform(70, 400), 0),
        "igM": round(random.uniform(40, 230), 0)
    }


def generate_vitals() -> Dict[str, Any]:
    return {
        "blood_pressure_systolic": random.randint(100, 150),
        "blood_pressure_diastolic": random.randint(60, 95),
        "heart_rate": random.randint(60, 100),
        "temperature": round(random.uniform(97.0, 99.5), 1),
        "weight_lbs": random.randint(120, 220),
        "oxygen_saturation": random.randint(95, 100)
    }


def generate_research_consent(patient_id: str, conn) -> None:
    consent_id = generate_uuid()
    data_types = [
        "dailyFollowups", "healthAlerts", "deteriorationIndex", "mlPredictions",
        "environmentalRisk", "medications", "vitals", "immuneMarkers",
        "behavioralData", "mentalHealth", "wearableData", "labResults"
    ]
    permissions = {dt: random.choice([True, True, True, False]) for dt in data_types}
    
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO research_consent (id, patient_id, consent_enabled, permissions, granted_at)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (patient_id) DO UPDATE SET permissions = EXCLUDED.permissions
        """, (consent_id, patient_id, True, json.dumps(permissions), random_date(180)))
    conn.commit()


def seed_locations(conn) -> List[str]:
    location_ids = []
    with conn.cursor() as cur:
        for city, state, lat, lon, loc_type, aqi in LOCATIONS:
            loc_id = generate_uuid()
            location_ids.append(loc_id)
            cur.execute("""
                INSERT INTO locations (id, name, city, state, latitude, longitude, location_type, air_quality_index, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW())
                ON CONFLICT DO NOTHING
            """, (loc_id, f"{city} Area", city, state, lat, lon, loc_type, aqi))
    conn.commit()
    return location_ids


def seed_patients(conn, count: int = 50) -> List[str]:
    patient_ids = []
    with conn.cursor() as cur:
        for i in range(count):
            patient_id = generate_uuid()
            patient_ids.append(patient_id)
            # Use clearly synthetic names with "Test" prefix
            first = f"Test{random.choice(FIRST_NAMES)}"
            last = f"Synthetic{random.choice(LAST_NAMES)}"
            # Use synthetic.example.com domain - clearly not real
            email = f"synth.patient{i}@synthetic.example.com"
            dob = random_date(365 * 70, 365 * 20)
            
            cur.execute("""
                INSERT INTO users (id, email, password, first_name, last_name, role, date_of_birth)
                VALUES (%s, %s, %s, %s, %s, 'patient', %s)
                ON CONFLICT (email) DO NOTHING
            """, (patient_id, email, "hashed_password", first, last, dob.date()))
            
            generate_research_consent(patient_id, conn)
    conn.commit()
    return patient_ids


def seed_patient_profiles(conn, patient_ids: List[str]) -> None:
    with conn.cursor() as cur:
        for patient_id in patient_ids:
            condition = random.choice(CONDITIONS)
            medications = random.sample(MEDICATIONS, k=random.randint(1, 4))
            allergies = random.sample(["Penicillin", "Sulfa", "NSAIDs", "Latex", "None"], k=random.randint(0, 2))
            
            immune_markers = generate_immune_markers()
            
            cur.execute("""
                INSERT INTO patient_profiles (user_id, primary_condition, medications, allergies, immune_markers, disease_severity, risk_score)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (user_id) DO UPDATE SET 
                    primary_condition = EXCLUDED.primary_condition,
                    medications = EXCLUDED.medications,
                    immune_markers = EXCLUDED.immune_markers
            """, (
                patient_id, 
                condition, 
                json.dumps(medications), 
                json.dumps(allergies),
                json.dumps(immune_markers),
                random.choice(["mild", "moderate", "severe"]),
                round(random.uniform(0, 15), 1)
            ))
    conn.commit()


def seed_studies(conn, patient_ids: List[str], count: int = 5) -> List[str]:
    study_ids = []
    study_names = [
        "Longitudinal Outcomes in Autoimmune Disorders",
        "Environmental Factors in Disease Flares",
        "Medication Adherence and Quality of Life",
        "Digital Biomarkers for Early Detection",
        "Immune Response Patterns in Chronic Conditions",
        "Mental Health Comorbidities Study",
        "Wearable Data for Symptom Prediction",
        "Personalized Treatment Response Analysis"
    ]
    
    with conn.cursor() as cur:
        for i in range(min(count, len(study_names))):
            study_id = generate_uuid()
            study_ids.append(study_id)
            
            start_date = random_date(180, 30)
            end_date = start_date + timedelta(days=random.randint(90, 365))
            
            inclusion_criteria = [
                {"type": "condition", "operator": "in", "value": random.sample(CONDITIONS, k=3)},
                {"type": "age", "operator": "between", "value": [18, 75]}
            ]
            
            exclusion_criteria = [
                {"type": "condition", "operator": "equals", "value": "Active Cancer"},
                {"type": "medication", "operator": "contains", "value": "Experimental Drug"}
            ]
            
            cur.execute("""
                INSERT INTO studies (
                    id, title, description, status, start_date, end_date,
                    principal_investigator, study_type, target_enrollment,
                    inclusion_criteria, exclusion_criteria, auto_reanalysis, reanalysis_frequency
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT DO NOTHING
            """, (
                study_id,
                f"[SYNTHETIC] {study_names[i]}",
                f"[SYNTHETIC DATA] A demo study examining {study_names[i].lower()} in immunocompromised patients.",
                random.choice(["planning", "enrolling", "follow_up", "analysis", "completed"]),
                start_date.date(),
                end_date.date(),
                f"Dr. {random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}",
                random.choice(["observational", "interventional", "registry"]),
                random.randint(50, 500),
                json.dumps(inclusion_criteria),
                json.dumps(exclusion_criteria),
                random.choice([True, False]),
                random.choice(["daily", "weekly", "monthly"])
            ))
            
            enrolled_patients = random.sample(patient_ids, k=min(len(patient_ids), random.randint(10, 30)))
            for patient_id in enrolled_patients:
                cur.execute("""
                    INSERT INTO study_enrollments (id, study_id, patient_id, enrollment_date, status)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT DO NOTHING
                """, (generate_uuid(), study_id, patient_id, random_date(90).date(), "active"))
    
    conn.commit()
    return study_ids


def seed_cohorts(conn, patient_ids: List[str], count: int = 8) -> List[str]:
    cohort_ids = []
    cohort_names = [
        "High-Risk Flare Patients",
        "Newly Diagnosed Cohort",
        "Treatment-Resistant Cases",
        "Remission Candidates",
        "Environmental Sensitivity Group",
        "Biologic Therapy Responders",
        "Young Adult Autoimmune",
        "Multi-Condition Overlap"
    ]
    
    with conn.cursor() as cur:
        for i in range(min(count, len(cohort_names))):
            cohort_id = generate_uuid()
            cohort_ids.append(cohort_id)
            
            criteria = {
                "ageRange": [random.randint(18, 30), random.randint(50, 75)],
                "conditions": random.sample(CONDITIONS, k=random.randint(2, 5)),
                "minRiskScore": round(random.uniform(0, 5), 1),
                "maxRiskScore": round(random.uniform(8, 15), 1)
            }
            
            selected_patients = random.sample(patient_ids, k=min(len(patient_ids), random.randint(5, 20)))
            
            cur.execute("""
                INSERT INTO cohorts (id, name, description, criteria, patient_ids, patient_count, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT DO NOTHING
            """, (
                cohort_id,
                cohort_names[i],
                f"Cohort of patients meeting criteria for {cohort_names[i].lower()}",
                json.dumps(criteria),
                selected_patients,
                len(selected_patients),
                random_date(60)
            ))
    conn.commit()
    return cohort_ids


def seed_daily_followup_templates(conn, count: int = 5) -> List[str]:
    template_ids = []
    template_data = [
        {
            "name": "Daily Symptom Check",
            "questions": [
                {"id": "q1", "type": "scale", "text": "Rate your overall pain (0-10)", "min": 0, "max": 10},
                {"id": "q2", "type": "scale", "text": "Rate your fatigue level (0-10)", "min": 0, "max": 10},
                {"id": "q3", "type": "multi_select", "text": "Which symptoms are you experiencing?", "options": SYMPTOMS[:8]},
                {"id": "q4", "type": "boolean", "text": "Did you take all prescribed medications today?"},
                {"id": "q5", "type": "text", "text": "Any additional notes about how you're feeling?"}
            ]
        },
        {
            "name": "Weekly Wellness Assessment",
            "questions": [
                {"id": "q1", "type": "scale", "text": "Overall quality of life this week (0-10)", "min": 0, "max": 10},
                {"id": "q2", "type": "scale", "text": "Sleep quality (0-10)", "min": 0, "max": 10},
                {"id": "q3", "type": "scale", "text": "Physical activity level (0-10)", "min": 0, "max": 10},
                {"id": "q4", "type": "scale", "text": "Stress level (0-10)", "min": 0, "max": 10},
                {"id": "q5", "type": "multi_select", "text": "Activities you were able to do", "options": ["Work", "Exercise", "Social", "Hobbies", "Household"]}
            ]
        },
        {
            "name": "Medication Side Effects",
            "questions": [
                {"id": "q1", "type": "multi_select", "text": "Side effects experienced", "options": ["Nausea", "Headache", "Dizziness", "Fatigue", "Rash", "None"]},
                {"id": "q2", "type": "scale", "text": "Severity of side effects (0-10)", "min": 0, "max": 10},
                {"id": "q3", "type": "boolean", "text": "Did side effects affect your daily activities?"},
                {"id": "q4", "type": "text", "text": "Describe any concerning symptoms"}
            ]
        },
        {
            "name": "Mental Health Check-in",
            "questions": [
                {"id": "q1", "type": "scale", "text": "Mood rating (0-10)", "min": 0, "max": 10},
                {"id": "q2", "type": "scale", "text": "Anxiety level (0-10)", "min": 0, "max": 10},
                {"id": "q3", "type": "boolean", "text": "Have you felt hopeless or down?"},
                {"id": "q4", "type": "boolean", "text": "Have you been able to enjoy activities?"},
                {"id": "q5", "type": "text", "text": "What's been on your mind?"}
            ]
        },
        {
            "name": "Flare Assessment",
            "questions": [
                {"id": "q1", "type": "boolean", "text": "Are you currently experiencing a flare?"},
                {"id": "q2", "type": "scale", "text": "Flare severity (0-10)", "min": 0, "max": 10},
                {"id": "q3", "type": "multi_select", "text": "Affected areas", "options": ["Joints", "Skin", "GI", "Neurological", "Cardiovascular", "Other"]},
                {"id": "q4", "type": "text", "text": "Possible triggers you've identified"}
            ]
        }
    ]
    
    with conn.cursor() as cur:
        for i, template in enumerate(template_data[:count]):
            template_id = generate_uuid()
            template_ids.append(template_id)
            cur.execute("""
                INSERT INTO daily_followup_templates (id, name, description, questions, is_active, created_at)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT DO NOTHING
            """, (
                template_id,
                template["name"],
                f"Template for {template['name'].lower()}",
                json.dumps(template["questions"]),
                True,
                random_date(90)
            ))
    conn.commit()
    return template_ids


def seed_followup_responses(conn, patient_ids: List[str], template_ids: List[str], count_per_patient: int = 14) -> None:
    with conn.cursor() as cur:
        for patient_id in patient_ids[:30]:
            for day in range(count_per_patient):
                template_id = random.choice(template_ids)
                response_date = datetime.now() - timedelta(days=day)
                
                responses = {
                    "q1": random.randint(1, 8),
                    "q2": random.randint(2, 7),
                    "q3": random.sample(SYMPTOMS[:8], k=random.randint(0, 4)),
                    "q4": random.choice([True, False]),
                    "q5": random.choice(["Feeling okay", "Some discomfort", "Better than yesterday", ""])
                }
                
                cur.execute("""
                    INSERT INTO daily_followup_responses (id, patient_id, template_id, responses, submitted_at)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT DO NOTHING
                """, (generate_uuid(), patient_id, template_id, json.dumps(responses), response_date))
    conn.commit()


def seed_research_alerts(conn, patient_ids: List[str], study_ids: List[str], count: int = 30) -> None:
    alert_types = ["deterioration", "flare_risk", "medication_adherence", "significant_change", "threshold_breach"]
    severities = ["low", "medium", "high", "critical"]
    
    with conn.cursor() as cur:
        for i in range(count):
            patient_id = random.choice(patient_ids)
            study_id = random.choice(study_ids) if study_ids and random.random() > 0.3 else None
            
            alert_type = random.choice(alert_types)
            severity = random.choice(severities)
            
            details = {
                "trigger": f"Patient showed {random.choice(['elevated', 'decreased', 'abnormal'])} {random.choice(['CRP', 'ESR', 'pain score', 'fatigue'])}",
                "baseline_value": round(random.uniform(1, 5), 1),
                "current_value": round(random.uniform(5, 10), 1),
                "percent_change": round(random.uniform(30, 150), 1),
                "recommendations": [
                    "Review recent symptom reports",
                    "Consider scheduling follow-up",
                    "Check medication adherence"
                ]
            }
            
            if alert_type == "deterioration":
                details["shap_features"] = [
                    {"feature": "crp_level", "importance": round(random.uniform(0.2, 0.4), 2)},
                    {"feature": "pain_score", "importance": round(random.uniform(0.1, 0.3), 2)},
                    {"feature": "sleep_quality", "importance": round(random.uniform(0.05, 0.15), 2)}
                ]
            
            cur.execute("""
                INSERT INTO research_alerts (id, patient_id, study_id, alert_type, severity, message, details, status, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT DO NOTHING
            """, (
                generate_uuid(),
                patient_id,
                study_id,
                alert_type,
                severity,
                f"{severity.upper()} {alert_type.replace('_', ' ').title()} detected for patient",
                json.dumps(details),
                random.choice(["active", "acknowledged", "resolved"]),
                random_date(30)
            ))
    conn.commit()


def seed_ml_analyses(conn, cohort_ids: List[str], study_ids: List[str], count: int = 15) -> List[str]:
    analysis_ids = []
    analysis_types = ["descriptive", "risk_prediction", "survival", "causal"]
    
    with conn.cursor() as cur:
        for i in range(count):
            analysis_id = generate_uuid()
            analysis_ids.append(analysis_id)
            
            analysis_type = random.choice(analysis_types)
            cohort_id = random.choice(cohort_ids) if cohort_ids else None
            study_id = random.choice(study_ids) if study_ids and random.random() > 0.5 else None
            
            config = {
                "analysisType": analysis_type,
                "targetVariable": random.choice(["flare_event", "hospitalization", "remission", "deterioration"]),
                "covariates": random.sample(["age", "sex", "disease_duration", "crp", "medication_count", "comorbidities"], k=random.randint(3, 5)),
                "modelType": random.choice(["logistic", "xgboost", "cox", "iptw"]) if analysis_type != "descriptive" else None
            }
            
            if analysis_type == "descriptive":
                results = {
                    "sample_size": random.randint(50, 200),
                    "demographics": {
                        "mean_age": round(random.uniform(40, 60), 1),
                        "female_pct": round(random.uniform(55, 75), 1),
                        "mean_disease_duration_years": round(random.uniform(3, 12), 1)
                    },
                    "clinical": {
                        "mean_crp": round(random.uniform(1, 8), 1),
                        "mean_das28": round(random.uniform(2, 5), 1),
                        "biologic_use_pct": round(random.uniform(30, 60), 1)
                    }
                }
            elif analysis_type == "risk_prediction":
                results = {
                    "auroc": round(random.uniform(0.72, 0.88), 3),
                    "auprc": round(random.uniform(0.45, 0.75), 3),
                    "brier_score": round(random.uniform(0.08, 0.18), 3),
                    "feature_importance": [
                        {"feature": "crp_level", "importance": round(random.uniform(0.15, 0.25), 3)},
                        {"feature": "pain_score", "importance": round(random.uniform(0.10, 0.18), 3)},
                        {"feature": "medication_adherence", "importance": round(random.uniform(0.08, 0.15), 3)},
                        {"feature": "sleep_hours", "importance": round(random.uniform(0.05, 0.12), 3)}
                    ],
                    "calibration": {"slope": round(random.uniform(0.85, 1.15), 2), "intercept": round(random.uniform(-0.1, 0.1), 2)}
                }
            elif analysis_type == "survival":
                results = {
                    "median_survival_days": random.randint(180, 720),
                    "hazard_ratios": [
                        {"variable": "high_crp", "hr": round(random.uniform(1.5, 2.5), 2), "ci_lower": round(random.uniform(1.1, 1.4), 2), "ci_upper": round(random.uniform(2.6, 3.5), 2), "p_value": round(random.uniform(0.001, 0.05), 4)},
                        {"variable": "biologic_use", "hr": round(random.uniform(0.4, 0.7), 2), "ci_lower": round(random.uniform(0.2, 0.35), 2), "ci_upper": round(random.uniform(0.75, 0.95), 2), "p_value": round(random.uniform(0.001, 0.02), 4)}
                    ],
                    "log_rank_p": round(random.uniform(0.001, 0.04), 4)
                }
            else:
                results = {
                    "ate": round(random.uniform(-0.15, 0.25), 3),
                    "ate_ci_lower": round(random.uniform(-0.25, 0.05), 3),
                    "ate_ci_upper": round(random.uniform(0.15, 0.45), 3),
                    "p_value": round(random.uniform(0.01, 0.08), 4),
                    "covariate_balance": {
                        "age": {"smd_before": round(random.uniform(0.2, 0.5), 2), "smd_after": round(random.uniform(0.01, 0.08), 2)},
                        "sex": {"smd_before": round(random.uniform(0.1, 0.3), 2), "smd_after": round(random.uniform(0.01, 0.05), 2)}
                    }
                }
            
            cur.execute("""
                INSERT INTO ml_analyses (id, name, analysis_type, cohort_id, study_id, config, results, status, created_at, completed_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT DO NOTHING
            """, (
                analysis_id,
                f"{analysis_type.replace('_', ' ').title()} Analysis #{i+1}",
                analysis_type,
                cohort_id,
                study_id,
                json.dumps(config),
                json.dumps(results),
                random.choice(["completed", "completed", "completed", "running", "failed"]),
                random_date(60),
                random_date(30) if random.random() > 0.2 else None
            ))
    conn.commit()
    return analysis_ids


def seed_research_reports(conn, analysis_ids: List[str], study_ids: List[str], count: int = 10) -> None:
    with conn.cursor() as cur:
        for i in range(count):
            analysis_id = random.choice(analysis_ids) if analysis_ids else None
            study_id = random.choice(study_ids) if study_ids else None
            
            narrative = {
                "abstract": f"This study examined outcomes in {random.randint(50, 200)} immunocompromised patients over {random.randint(6, 24)} months. Key findings include significant associations between inflammatory markers and disease progression.",
                "methods": "We conducted a retrospective cohort analysis using electronic health records. Statistical analyses included multivariate regression and survival modeling with appropriate adjustments for confounders.",
                "results": f"The primary outcome occurred in {random.randint(15, 35)}% of patients. Elevated CRP (>5 mg/L) was associated with {round(random.uniform(1.5, 2.5), 1)}x higher risk of adverse events (p<0.05).",
                "discussion": "These findings support the use of inflammatory markers for risk stratification. Limitations include the observational design and potential for unmeasured confounding."
            }
            
            cur.execute("""
                INSERT INTO research_reports (id, title, analysis_id, study_id, content, summary, status, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT DO NOTHING
            """, (
                generate_uuid(),
                f"Research Report: {random.choice(['Outcomes', 'Risk Factors', 'Treatment Response', 'Predictive Modeling'])} in Autoimmune Disease #{i+1}",
                analysis_id,
                study_id,
                json.dumps(narrative),
                narrative["abstract"][:200],
                random.choice(["draft", "final", "published"]),
                random_date(45)
            ))
    conn.commit()


def seed_environmental_exposures(conn, patient_ids: List[str], location_ids: List[str]) -> None:
    with conn.cursor() as cur:
        for patient_id in patient_ids:
            if random.random() > 0.3:
                loc_id = random.choice(location_ids) if location_ids else None
                
                for day in range(random.randint(7, 30)):
                    exposure_date = datetime.now() - timedelta(days=day)
                    
                    cur.execute("""
                        INSERT INTO environmental_exposures (id, patient_id, location_id, exposure_date, air_quality_index, temperature, humidity, uv_index, pollen_count, pollution_level)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT DO NOTHING
                    """, (
                        generate_uuid(),
                        patient_id,
                        loc_id,
                        exposure_date.date(),
                        random.randint(20, 150),
                        round(random.uniform(30, 95), 1),
                        round(random.uniform(20, 80), 1),
                        round(random.uniform(1, 11), 1),
                        random.randint(0, 500),
                        random.choice(["low", "moderate", "high", "very_high"])
                    ))
    conn.commit()


def seed_data_quality_metrics(conn, study_ids: List[str]) -> None:
    with conn.cursor() as cur:
        for study_id in study_ids:
            quality_metrics = {
                "overall_score": round(random.uniform(70, 95), 1),
                "completeness": round(random.uniform(75, 98), 1),
                "consistency": round(random.uniform(80, 99), 1),
                "timeliness": round(random.uniform(60, 95), 1),
                "variable_metrics": {
                    "age": {"missing_pct": round(random.uniform(0, 5), 1), "outlier_pct": round(random.uniform(0, 2), 1)},
                    "crp": {"missing_pct": round(random.uniform(5, 20), 1), "outlier_pct": round(random.uniform(1, 5), 1)},
                    "pain_score": {"missing_pct": round(random.uniform(10, 30), 1), "outlier_pct": round(random.uniform(0, 3), 1)},
                    "medication_adherence": {"missing_pct": round(random.uniform(15, 35), 1), "outlier_pct": round(random.uniform(0, 1), 1)}
                }
            }
            
            cur.execute("""
                INSERT INTO data_quality_metrics (id, study_id, metrics, score, assessed_at)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT DO NOTHING
            """, (
                generate_uuid(),
                study_id,
                json.dumps(quality_metrics),
                quality_metrics["overall_score"],
                random_date(7)
            ))
    conn.commit()


def seed_deterioration_indices(conn, patient_ids: List[str]) -> None:
    with conn.cursor() as cur:
        for patient_id in patient_ids:
            for day in range(random.randint(7, 21)):
                calc_date = datetime.now() - timedelta(days=day)
                
                factors = {
                    "symptom_severity": round(random.uniform(0, 3), 2),
                    "vital_signs": round(random.uniform(0, 2), 2),
                    "lab_markers": round(random.uniform(0, 3), 2),
                    "medication_adherence": round(random.uniform(0, 2), 2),
                    "activity_level": round(random.uniform(0, 2), 2),
                    "sleep_quality": round(random.uniform(0, 1.5), 2),
                    "mental_health": round(random.uniform(0, 1.5), 2)
                }
                composite_score = sum(factors.values())
                
                cur.execute("""
                    INSERT INTO deterioration_indices (id, patient_id, composite_score, factor_scores, risk_level, calculated_at)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT DO NOTHING
                """, (
                    generate_uuid(),
                    patient_id,
                    round(composite_score, 2),
                    json.dumps(factors),
                    "high" if composite_score > 10 else "medium" if composite_score > 5 else "low",
                    calc_date
                ))
    conn.commit()


def run_synthetic_data_generation():
    print("Starting synthetic data generation for Enhanced Research Center...")
    print("=" * 60)
    
    if not DATABASE_URL:
        print("ERROR: DATABASE_URL environment variable not set")
        return
    
    try:
        conn = get_connection()
        print("Connected to database")
        
        print("\n1. Seeding locations...")
        location_ids = seed_locations(conn)
        print(f"   Created {len(location_ids)} locations")
        
        print("\n2. Seeding patients...")
        patient_ids = seed_patients(conn, count=50)
        print(f"   Created {len(patient_ids)} patients with consent")
        
        print("\n3. Seeding patient profiles...")
        seed_patient_profiles(conn, patient_ids)
        print(f"   Created profiles for {len(patient_ids)} patients")
        
        print("\n4. Seeding studies...")
        study_ids = seed_studies(conn, patient_ids, count=5)
        print(f"   Created {len(study_ids)} studies with enrollments")
        
        print("\n5. Seeding cohorts...")
        cohort_ids = seed_cohorts(conn, patient_ids, count=8)
        print(f"   Created {len(cohort_ids)} cohorts")
        
        print("\n6. Seeding daily followup templates...")
        template_ids = seed_daily_followup_templates(conn, count=5)
        print(f"   Created {len(template_ids)} templates")
        
        print("\n7. Seeding followup responses...")
        seed_followup_responses(conn, patient_ids, template_ids, count_per_patient=14)
        print("   Created followup responses")
        
        print("\n8. Seeding research alerts...")
        seed_research_alerts(conn, patient_ids, study_ids, count=30)
        print("   Created 30 research alerts")
        
        print("\n9. Seeding ML analyses...")
        analysis_ids = seed_ml_analyses(conn, cohort_ids, study_ids, count=15)
        print(f"   Created {len(analysis_ids)} ML analyses")
        
        print("\n10. Seeding research reports...")
        seed_research_reports(conn, analysis_ids, study_ids, count=10)
        print("   Created 10 research reports")
        
        print("\n11. Seeding environmental exposures...")
        seed_environmental_exposures(conn, patient_ids, location_ids)
        print("   Created environmental exposure data")
        
        print("\n12. Seeding data quality metrics...")
        seed_data_quality_metrics(conn, study_ids)
        print("   Created data quality metrics")
        
        print("\n13. Seeding deterioration indices...")
        seed_deterioration_indices(conn, patient_ids)
        print("   Created deterioration index data")
        
        conn.close()
        print("\n" + "=" * 60)
        print("Synthetic data generation completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"ERROR: {e}")
        raise


if __name__ == "__main__":
    run_synthetic_data_generation()
