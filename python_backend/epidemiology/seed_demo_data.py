"""
Epidemiology Demo Data Seeder
=============================
Creates synthetic data for testing the epidemiology research module.
All data is anonymized and aggregated - no real patient information.
"""

import os
import random
import uuid
from datetime import datetime, timedelta
from typing import List, Tuple
import psycopg2
import psycopg2.extras

DB_URL = os.environ.get("DATABASE_URL", "")

DRUGS = [
    ("ACETAMINOPHEN", "Acetaminophen"),
    ("IBUPROFEN", "Ibuprofen"),
    ("METFORMIN", "Metformin"),
    ("LISINOPRIL", "Lisinopril"),
    ("ATORVASTATIN", "Atorvastatin"),
    ("OMEPRAZOLE", "Omeprazole"),
    ("AMLODIPINE", "Amlodipine"),
    ("METOPROLOL", "Metoprolol"),
    ("LOSARTAN", "Losartan"),
    ("GABAPENTIN", "Gabapentin"),
    ("PREDNISONE", "Prednisone"),
    ("AZITHROMYCIN", "Azithromycin"),
]

OUTCOMES = [
    ("HEPATOTOXICITY", "Hepatotoxicity"),
    ("NEPHROTOXICITY", "Nephrotoxicity"),
    ("CARDIAC_EVENT", "Cardiac Adverse Event"),
    ("GI_BLEEDING", "Gastrointestinal Bleeding"),
    ("RASH", "Skin Rash"),
    ("NAUSEA", "Nausea/Vomiting"),
    ("HYPOGLYCEMIA", "Hypoglycemia"),
    ("ANGIOEDEMA", "Angioedema"),
    ("MYOPATHY", "Myopathy"),
    ("INFECTION", "Secondary Infection"),
]

PATHOGENS = [
    ("COVID-19", "SARS-CoV-2"),
    ("INFLUENZA-A", "Influenza A"),
    ("INFLUENZA-B", "Influenza B"),
    ("RSV", "Respiratory Syncytial Virus"),
    ("STREP-PNEUMO", "Streptococcus pneumoniae"),
    ("MYCOPLASMA", "Mycoplasma pneumoniae"),
]

VACCINES = [
    ("COVID-19-MRNA", "COVID-19 mRNA Vaccine"),
    ("COVID-19-VIRAL", "COVID-19 Viral Vector Vaccine"),
    ("INFLUENZA-2024", "Influenza 2024-2025"),
    ("RSV-ADULT", "RSV Adult Vaccine"),
    ("PNEUMO-23", "Pneumococcal 23-valent"),
    ("SHINGRIX", "Shingles Vaccine"),
]


def seed_locations(cur) -> List[str]:
    """Seed location data if not exists and return location IDs."""
    cur.execute("SELECT id FROM locations LIMIT 1")
    if cur.fetchone():
        cur.execute("SELECT id FROM locations LIMIT 10")
        return [row[0] for row in cur.fetchall()]
    
    locations = [
        ("loc_1", "Downtown Clinic", "New York", "NY", "10001", "US"),
        ("loc_2", "Westside Health Center", "Los Angeles", "CA", "90001", "US"),
        ("loc_3", "Midwest Regional Hospital", "Chicago", "IL", "60601", "US"),
        ("loc_4", "Southern Medical Center", "Houston", "TX", "77001", "US"),
        ("loc_5", "Pacific Health System", "Seattle", "WA", "98101", "US"),
        ("loc_6", "Mountain View Clinic", "Denver", "CO", "80201", "US"),
        ("loc_7", "Atlantic General", "Miami", "FL", "33101", "US"),
        ("loc_8", "Northeast Medical", "Boston", "MA", "02101", "US"),
    ]
    
    location_ids = []
    for loc in locations:
        cur.execute("""
            INSERT INTO locations (id, name, city, state, postal_code, country)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO NOTHING
            RETURNING id
        """, loc)
        result = cur.fetchone()
        location_ids.append(loc[0])
    
    return location_ids


def seed_drug_signals(cur, location_ids: List[str]):
    """Seed aggregated drug safety signals."""
    cur.execute("SELECT id FROM drug_outcome_signals LIMIT 1")
    if cur.fetchone():
        print("Drug signals already seeded")
        return
    
    print("Seeding drug safety signals...")
    
    for drug_code, drug_name in DRUGS:
        for outcome_code, outcome_name in random.sample(OUTCOMES, k=random.randint(2, 5)):
            for location_id in random.sample(location_ids, k=random.randint(1, 4)):
                n_patients = random.randint(50, 2000)
                n_events = random.randint(5, n_patients // 3)
                
                estimate = random.uniform(0.5, 3.5)
                ci_margin = random.uniform(0.1, 0.5)
                
                cur.execute("""
                    INSERT INTO drug_outcome_signals (
                        drug_code, drug_name, outcome_code, outcome_name,
                        patient_location_id, estimate, ci_lower, ci_upper,
                        p_value, signal_strength, n_patients, n_events,
                        flagged, calculated_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
                """, (
                    drug_code, drug_name, outcome_code, outcome_name,
                    location_id, estimate, estimate - ci_margin, estimate + ci_margin,
                    random.uniform(0.001, 0.2),
                    random.uniform(20, 95),
                    n_patients, n_events,
                    estimate > 2.0 and random.random() > 0.5
                ))
    
    print(f"Seeded drug signals")


def seed_infectious_events(cur, location_ids: List[str]):
    """Seed aggregated infectious disease data."""
    cur.execute("SELECT id FROM infectious_events_aggregated LIMIT 1")
    if cur.fetchone():
        print("Infectious events already seeded")
        return
    
    print("Seeding infectious disease events...")
    
    base_date = datetime.utcnow() - timedelta(days=90)
    batch = []
    
    for pathogen_code, pathogen_name in PATHOGENS[:3]:
        for location_id in location_ids[:4]:
            for day_offset in range(0, 90, 7):
                event_date = base_date + timedelta(days=day_offset)
                
                wave_factor = 1 + 0.5 * abs((day_offset % 45) - 22) / 22
                base_cases = random.randint(10, 100)
                cases = int(base_cases * wave_factor * random.uniform(0.5, 1.5))
                deaths = int(cases * random.uniform(0.005, 0.03))
                
                batch.append((
                    pathogen_code, pathogen_name, location_id,
                    event_date.date(), cases, deaths, int(cases * 0.1),
                    random.choice(['mild', 'moderate', 'severe'])
                ))
    
    psycopg2.extras.execute_batch(cur, """
        INSERT INTO infectious_events_aggregated (
            pathogen_code, pathogen_name, patient_location_id,
            event_date, case_count, death_count, hospitalized_count,
            severity, created_at
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW())
    """, batch)
    
    print(f"Seeded {len(batch)} infectious events")


def seed_reproduction_numbers(cur, location_ids: List[str]):
    """Seed R0/Rt estimates."""
    cur.execute("SELECT id FROM reproduction_numbers LIMIT 1")
    if cur.fetchone():
        print("R numbers already seeded")
        return
    
    print("Seeding reproduction numbers...")
    
    for pathogen_code, _ in PATHOGENS:
        for location_id in location_ids:
            r_value = random.uniform(0.6, 2.5)
            ci_margin = random.uniform(0.1, 0.4)
            
            cur.execute("""
                INSERT INTO reproduction_numbers (
                    pathogen_code, location_id, calculation_date,
                    r_value, r_lower, r_upper, generation_interval_days,
                    sample_size, method, created_at
                ) VALUES (%s, %s, NOW(), %s, %s, %s, %s, %s, %s, NOW())
            """, (
                pathogen_code, location_id,
                r_value, max(0, r_value - ci_margin), r_value + ci_margin,
                random.uniform(3, 7),
                random.randint(100, 5000),
                random.choice(['EpiEstim', 'Wallinga-Teunis', 'Cori'])
            ))
    
    print(f"Seeded reproduction numbers")


def seed_immunizations(cur, location_ids: List[str]):
    """Seed aggregated immunization data."""
    cur.execute("SELECT id FROM immunization_aggregates LIMIT 1")
    if cur.fetchone():
        print("Immunizations already seeded")
        return
    
    print("Seeding immunization aggregates...")
    
    for vaccine_code, vaccine_name in VACCINES:
        for location_id in location_ids:
            total_population = random.randint(50000, 500000)
            for dose_number in range(1, random.randint(2, 4)):
                coverage_rate = random.uniform(0.3, 0.85) ** dose_number
                vaccinated = int(total_population * coverage_rate)
                
                cur.execute("""
                    INSERT INTO immunization_aggregates (
                        vaccine_code, vaccine_name, location_id,
                        dose_number, vaccinated_count, total_population,
                        coverage_rate, aggregation_date, created_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, NOW(), NOW())
                """, (
                    vaccine_code, vaccine_name, location_id,
                    dose_number, vaccinated, total_population,
                    coverage_rate
                ))
    
    print(f"Seeded immunization aggregates")


def seed_vaccine_effectiveness(cur, location_ids: List[str]):
    """Seed vaccine effectiveness estimates."""
    cur.execute("SELECT id FROM vaccine_effectiveness_estimates LIMIT 1")
    if cur.fetchone():
        print("Vaccine effectiveness already seeded")
        return
    
    print("Seeding vaccine effectiveness estimates...")
    
    vaccine_outcomes = [
        ("COVID-19-MRNA", "COVID-19", 0.85, 0.95),
        ("COVID-19-MRNA", "HOSPITALIZATION", 0.90, 0.98),
        ("COVID-19-VIRAL", "COVID-19", 0.70, 0.85),
        ("INFLUENZA-2024", "INFLUENZA-A", 0.40, 0.65),
        ("INFLUENZA-2024", "HOSPITALIZATION", 0.55, 0.75),
        ("RSV-ADULT", "RSV", 0.75, 0.88),
        ("PNEUMO-23", "STREP-PNEUMO", 0.60, 0.80),
    ]
    
    for location_id in location_ids:
        for vaccine_code, outcome_code, ve_low, ve_high in vaccine_outcomes:
            ve = random.uniform(ve_low, ve_high)
            ci_margin = random.uniform(0.03, 0.1)
            
            cur.execute("""
                INSERT INTO vaccine_effectiveness_estimates (
                    vaccine_code, outcome_code, location_id,
                    effectiveness, ci_lower, ci_upper,
                    n_vaccinated, n_unvaccinated, n_events_vaccinated,
                    n_events_unvaccinated, study_design, calculation_date, created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW())
            """, (
                vaccine_code, outcome_code, location_id,
                ve, max(0, ve - ci_margin), min(1, ve + ci_margin),
                random.randint(10000, 100000),
                random.randint(5000, 50000),
                random.randint(50, 500),
                random.randint(200, 2000),
                random.choice(['TND', 'cohort', 'case-control'])
            ))
    
    print(f"Seeded vaccine effectiveness estimates")


def seed_outbreaks(cur, location_ids: List[str]):
    """Seed outbreak records."""
    cur.execute("SELECT id FROM outbreaks LIMIT 1")
    if cur.fetchone():
        print("Outbreaks already seeded")
        return
    
    print("Seeding outbreaks...")
    
    for pathogen_code, pathogen_name in random.sample(PATHOGENS, k=4):
        for location_id in random.sample(location_ids, k=random.randint(1, 3)):
            start_date = datetime.utcnow() - timedelta(days=random.randint(30, 150))
            duration = random.randint(14, 60)
            
            cur.execute("""
                INSERT INTO outbreaks (
                    pathogen_code, pathogen_name, location_id,
                    start_date, end_date, status, peak_cases,
                    total_cases, total_deaths, severity_index, created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
            """, (
                pathogen_code, pathogen_name, location_id,
                start_date.date(),
                (start_date + timedelta(days=duration)).date() if random.random() > 0.3 else None,
                random.choice(['active', 'contained', 'resolved']),
                random.randint(50, 500),
                random.randint(200, 2000),
                random.randint(2, 50),
                random.uniform(0.3, 0.9)
            ))
    
    print(f"Seeded outbreaks")


def seed_ml_features(cur, location_ids: List[str]):
    """Seed ML feature tables for training pipelines."""
    
    cur.execute("SELECT id FROM ml_drug_features LIMIT 1")
    if not cur.fetchone():
        print("Seeding ML drug features...")
        for drug_code, _ in DRUGS:
            for location_id in location_ids:
                cur.execute("""
                    INSERT INTO ml_drug_features (
                        drug_code, location_id, feature_date,
                        prescription_count, adverse_event_count,
                        chronic_rate, avg_duration_days, created_at
                    ) VALUES (%s, %s, NOW(), %s, %s, %s, %s, NOW())
                """, (
                    drug_code, location_id,
                    random.randint(100, 5000),
                    random.randint(5, 200),
                    random.uniform(0.1, 0.6),
                    random.uniform(7, 180)
                ))
    
    cur.execute("SELECT id FROM ml_outbreak_features LIMIT 1")
    if not cur.fetchone():
        print("Seeding ML outbreak features...")
        for pathogen_code, _ in PATHOGENS:
            for location_id in location_ids:
                cur.execute("""
                    INSERT INTO ml_outbreak_features (
                        pathogen_code, location_id, feature_date,
                        case_count, death_count, hospitalization_rate,
                        r_estimate, doubling_time_days, created_at
                    ) VALUES (%s, %s, NOW(), %s, %s, %s, %s, %s, NOW())
                """, (
                    pathogen_code, location_id,
                    random.randint(10, 500),
                    random.randint(0, 20),
                    random.uniform(0.02, 0.15),
                    random.uniform(0.6, 2.5),
                    random.uniform(3, 21) if random.random() > 0.3 else None
                ))
    
    cur.execute("SELECT id FROM ml_vaccine_features LIMIT 1")
    if not cur.fetchone():
        print("Seeding ML vaccine features...")
        for vaccine_code, _ in VACCINES:
            for location_id in location_ids:
                cur.execute("""
                    INSERT INTO ml_vaccine_features (
                        vaccine_code, location_id, feature_date,
                        coverage_rate, doses_administered,
                        adverse_event_rate, effectiveness_estimate, created_at
                    ) VALUES (%s, %s, NOW(), %s, %s, %s, %s, NOW())
                """, (
                    vaccine_code, location_id,
                    random.uniform(0.3, 0.85),
                    random.randint(1000, 50000),
                    random.uniform(0.001, 0.02),
                    random.uniform(0.5, 0.95)
                ))
    
    print("ML features seeded")


INDUSTRIES = [
    ("HEALTHCARE", "Healthcare"),
    ("MANUFACTURING", "Manufacturing"),
    ("CONSTRUCTION", "Construction"),
    ("MINING", "Mining"),
    ("AGRICULTURE", "Agriculture"),
    ("CHEMICAL", "Chemical Industry"),
    ("TRANSPORTATION", "Transportation"),
]

HAZARDS = [
    ("ASBESTOS", "Asbestos Exposure"),
    ("SILICA", "Silica Dust"),
    ("LEAD", "Lead Exposure"),
    ("RADIATION", "Ionizing Radiation"),
    ("NOISE", "Occupational Noise"),
    ("CHEMICAL_SOLVENTS", "Chemical Solvents"),
    ("BIOLOGICAL_AGENTS", "Biological Agents"),
    ("ERGONOMIC", "Ergonomic Hazards"),
]

OCCUPATIONAL_OUTCOMES = [
    ("MESOTHELIOMA", "Mesothelioma"),
    ("SILICOSIS", "Silicosis"),
    ("HEARING_LOSS", "Occupational Hearing Loss"),
    ("LEUKEMIA", "Leukemia"),
    ("DERMATITIS", "Occupational Dermatitis"),
    ("ASTHMA", "Occupational Asthma"),
    ("CARPAL_TUNNEL", "Carpal Tunnel Syndrome"),
]

GENES = [
    ("CYP2D6", "Cytochrome P450 2D6"),
    ("CYP2C19", "Cytochrome P450 2C19"),
    ("CYP3A4", "Cytochrome P450 3A4"),
    ("VKORC1", "Vitamin K epoxide reductase"),
    ("SLCO1B1", "Solute carrier organic anion transporter"),
    ("HLA-B", "Human Leukocyte Antigen B"),
    ("TPMT", "Thiopurine methyltransferase"),
    ("DPYD", "Dihydropyrimidine dehydrogenase"),
]

VARIANTS = [
    ("rs1065852", "CYP2D6", "*4 allele"),
    ("rs4244285", "CYP2C19", "*2 allele"),
    ("rs12248560", "CYP2C19", "*17 allele"),
    ("rs9923231", "VKORC1", "-1639G>A"),
    ("rs4149056", "SLCO1B1", "521T>C"),
    ("rs2395029", "HLA-B", "*5701"),
    ("rs1800460", "TPMT", "*3B"),
    ("rs1801265", "DPYD", "*2A"),
]

GENETIC_OUTCOMES = [
    ("DRUG_TOXICITY", "Drug Toxicity"),
    ("POOR_RESPONSE", "Poor Drug Response"),
    ("BLEEDING_RISK", "Increased Bleeding Risk"),
    ("MYOPATHY_RISK", "Statin-Induced Myopathy"),
    ("HYPERSENSITIVITY", "Drug Hypersensitivity"),
    ("METABOLISM_SLOW", "Slow Drug Metabolism"),
    ("METABOLISM_FAST", "Ultra-Rapid Metabolism"),
]


def seed_occupational_signals(cur, location_ids: List[str]):
    """Seed occupational risk signals."""
    cur.execute("SELECT id FROM occupational_risk_signals LIMIT 1")
    if cur.fetchone():
        print("Occupational signals already seeded")
        return
    
    print("Seeding occupational risk signals...")
    
    for industry_code, industry_name in INDUSTRIES:
        for hazard_code, hazard_name in random.sample(HAZARDS, k=random.randint(2, 4)):
            for outcome_code, outcome_name in random.sample(OCCUPATIONAL_OUTCOMES, k=random.randint(1, 3)):
                for location_id in random.sample(location_ids, k=random.randint(1, 3)):
                    n_workers = random.randint(50, 5000)
                    n_events = random.randint(5, n_workers // 10)
                    
                    estimate = random.uniform(1.2, 8.0)
                    ci_margin = random.uniform(0.2, 1.0)
                    
                    cur.execute("""
                        INSERT INTO occupational_risk_signals (
                            industry_code, industry_name, occupation_code, occupation_name,
                            hazard_code, hazard_name, outcome_code, outcome_name,
                            location_id, model_type, effect_measure, estimate, ci_lower, ci_upper,
                            p_value, signal_strength, n_workers, n_events, mean_exposure_years,
                            flagged, suppressed, calculated_at
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
                    """, (
                        industry_code, industry_name,
                        f"{industry_code}_WORKER", f"{industry_name} Worker",
                        hazard_code, hazard_name,
                        outcome_code, outcome_name,
                        location_id, 'cox', 'HR',
                        estimate, estimate - ci_margin, estimate + ci_margin,
                        random.uniform(0.001, 0.1),
                        random.uniform(30, 95),
                        n_workers, n_events,
                        random.uniform(1, 25),
                        estimate > 2 and random.random() > 0.3,
                        False
                    ))
    
    print("Occupational signals seeded successfully")


def seed_genetic_associations(cur, location_ids: List[str]):
    """Seed variant-outcome associations."""
    cur.execute("SELECT id FROM variant_outcome_associations LIMIT 1")
    if cur.fetchone():
        print("Genetic associations already seeded")
        return
    
    print("Seeding genetic variant associations...")
    
    for rsid, gene_symbol, variant_desc in VARIANTS:
        gene_name = next((g[1] for g in GENES if g[0] == gene_symbol), None)
        
        for outcome_code, outcome_name in random.sample(GENETIC_OUTCOMES, k=random.randint(1, 3)):
            for location_id in random.sample(location_ids, k=random.randint(1, 2)):
                n_carriers = random.randint(100, 2000)
                n_non_carriers = random.randint(500, 5000)
                events_carriers = random.randint(10, n_carriers // 5)
                events_non_carriers = random.randint(10, n_non_carriers // 10)
                
                estimate = random.uniform(1.5, 5.0)
                ci_margin = random.uniform(0.2, 0.8)
                
                cur.execute("""
                    INSERT INTO variant_outcome_associations (
                        rsid, gene_symbol, gene_name, outcome_code, outcome_name,
                        location_id, model_type, effect_measure, estimate, ci_lower, ci_upper,
                        p_value, signal_strength, n_carriers, n_non_carriers,
                        events_carriers, events_non_carriers, flagged, suppressed, calculated_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
                """, (
                    rsid, gene_symbol, gene_name,
                    outcome_code, outcome_name,
                    location_id, 'logistic', 'OR',
                    estimate, estimate - ci_margin, estimate + ci_margin,
                    random.uniform(0.0001, 0.05),
                    random.uniform(40, 90),
                    n_carriers, n_non_carriers,
                    events_carriers, events_non_carriers,
                    estimate > 2 and random.random() > 0.4,
                    False
                ))
    
    print("Genetic associations seeded successfully")


def seed_gwas_results(cur):
    """Seed GWAS summary statistics."""
    cur.execute("SELECT id FROM gwas_results LIMIT 1")
    if cur.fetchone():
        print("GWAS results already seeded")
        return
    
    print("Seeding GWAS results...")
    
    traits = [
        ("T2D", "Type 2 Diabetes"),
        ("CAD", "Coronary Artery Disease"),
        ("HYPERTENSION", "Hypertension"),
        ("OBESITY", "Obesity"),
        ("ASTHMA", "Asthma"),
        ("RA", "Rheumatoid Arthritis"),
    ]
    
    for trait_code, trait_name in traits:
        for rsid, gene_symbol, _ in random.sample(VARIANTS, k=random.randint(3, 6)):
            p_value = 10 ** (-random.uniform(3, 12))
            
            cur.execute("""
                INSERT INTO gwas_results (
                    study_id, study_name, trait_code, trait_name,
                    rsid, gene_symbol, chromosome, position,
                    effect_allele, other_allele, beta, se, p_value,
                    odds_ratio, ci_lower, ci_upper, sample_size, ancestry,
                    genome_wide_significant, calculated_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
            """, (
                f"GWAS_{trait_code}_{random.randint(2020, 2024)}",
                f"{trait_name} GWAS Study",
                trait_code, trait_name,
                rsid, gene_symbol,
                str(random.randint(1, 22)),
                random.randint(1000000, 250000000),
                random.choice(['A', 'G', 'C', 'T']),
                random.choice(['A', 'G', 'C', 'T']),
                random.uniform(-0.5, 0.5),
                random.uniform(0.01, 0.1),
                p_value,
                random.uniform(0.8, 1.5),
                random.uniform(0.7, 0.9),
                random.uniform(1.1, 1.6),
                random.randint(10000, 500000),
                random.choice(['European', 'East Asian', 'African', 'Mixed']),
                p_value < 5e-8
            ))
    
    print("GWAS results seeded successfully")


def seed_pharmacogenomics(cur):
    """Seed pharmacogenomic interactions."""
    cur.execute("SELECT id FROM pharmacogenomic_interactions LIMIT 1")
    if cur.fetchone():
        print("Pharmacogenomics already seeded")
        return
    
    print("Seeding pharmacogenomic interactions...")
    
    pgx_drugs = [
        ("WARFARIN", "Warfarin"),
        ("CLOPIDOGREL", "Clopidogrel"),
        ("SIMVASTATIN", "Simvastatin"),
        ("CODEINE", "Codeine"),
        ("TAMOXIFEN", "Tamoxifen"),
        ("ABACAVIR", "Abacavir"),
        ("AZATHIOPRINE", "Azathioprine"),
        ("FLUOROURACIL", "5-Fluorouracil"),
    ]
    
    for drug_code, drug_name in pgx_drugs:
        for rsid, gene_symbol, _ in random.sample(VARIANTS, k=random.randint(1, 3)):
            gene_name = next((g[1] for g in GENES if g[0] == gene_symbol), None)
            
            cur.execute("""
                INSERT INTO pharmacogenomic_interactions (
                    rsid, gene_symbol, gene_name, drug_code, drug_name,
                    interaction_type, phenotype, recommendation,
                    evidence_level, clinical_impact, n_patients, suppressed, created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
            """, (
                rsid, gene_symbol, gene_name,
                drug_code, drug_name,
                random.choice(['Metabolism', 'Sensitivity', 'Toxicity', 'Response']),
                random.choice(['Poor Metabolizer', 'Intermediate Metabolizer', 'Normal Metabolizer', 'Ultra-Rapid Metabolizer']),
                random.choice(['Consider dose reduction', 'Use alternative drug', 'Standard dosing', 'Avoid use', 'Increased monitoring']),
                random.choice(['1A', '1B', '2A', '2B', '3']),
                random.choice(['high', 'moderate', 'low']),
                random.randint(50, 2000),
                False
            ))
    
    print("Pharmacogenomics seeded successfully")


def run_seeder():
    """Run all seeders."""
    if not DB_URL:
        print("ERROR: DATABASE_URL not set")
        return
    
    print("Starting epidemiology data seeding...")
    
    try:
        conn = psycopg2.connect(DB_URL)
        cur = conn.cursor()
        
        location_ids = seed_locations(cur)
        conn.commit()
        
        seed_drug_signals(cur, location_ids)
        conn.commit()
        
        seed_infectious_events(cur, location_ids)
        conn.commit()
        
        seed_reproduction_numbers(cur, location_ids)
        conn.commit()
        
        seed_immunizations(cur, location_ids)
        conn.commit()
        
        seed_vaccine_effectiveness(cur, location_ids)
        conn.commit()
        
        seed_outbreaks(cur, location_ids)
        conn.commit()
        
        seed_ml_features(cur, location_ids)
        conn.commit()
        
        # New occupational and genetic epidemiology data
        seed_occupational_signals(cur, location_ids)
        conn.commit()
        
        seed_genetic_associations(cur, location_ids)
        conn.commit()
        
        seed_gwas_results(cur)
        conn.commit()
        
        seed_pharmacogenomics(cur)
        conn.commit()
        
        cur.close()
        conn.close()
        
        print("Epidemiology data seeding completed successfully!")
        
    except Exception as e:
        print(f"Error during seeding: {e}")
        raise


if __name__ == "__main__":
    run_seeder()
