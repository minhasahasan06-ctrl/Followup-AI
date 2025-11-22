"""
Drug Seeding Script - RxNorm Integration
Populates the drugs table with standardized drug data from RxNorm API

Usage:
    python -m app.scripts.seed_drugs_rxnorm --limit 100

This script seeds common medications into the drugs table using RxNorm API.
"""

import asyncio
import argparse
import logging
import requests
from typing import List, Dict, Optional
from sqlalchemy import text
from datetime import datetime
import time

from app.database import get_db
from app.services.drug_normalization_service import DrugNormalizationService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# RxNorm API Base URL
RXNORM_BASE_URL = "https://rxnav.nlm.nih.gov/REST"

# Common medications to seed (top 100 most prescribed drugs in US)
COMMON_MEDICATIONS = [
    "Lisinopril", "Levothyroxine", "Atorvastatin", "Metformin", "Simvastatin",
    "Omeprazole", "Amlodipine", "Metoprolol", "Losartan", "Albuterol",
    "Gabapentin", "Hydrochlorothiazide", "Sertraline", "Montelukast", "Furosemide",
    "Escitalopram", "Bupropion", "Fluoxetine", "Citalopram", "Pantoprazole",
    "Rosuvastatin", "Pravastatin", "Tramadol", "Trazodone", "Duloxetine",
    "Amoxicillin", "Azithromycin", "Cephalexin", "Ciprofloxacin", "Doxycycline",
    "Prednisone", "Methylprednisolone", "Warfarin", "Clopidogrel", "Aspirin",
    "Ibuprofen", "Naproxen", "Acetaminophen", "Cyclobenzaprine", "Meloxicam",
    "Insulin", "Glipizide", "Sitagliptin", "Empagliflozin", "Semaglutide",
    "Alprazolam", "Clonazepam", "Lorazepam", "Zolpidem", "Hydroxyzine",
    "Carvedilol", "Propranolol", "Valsartan", "Spironolactone", "Enalapril",
    "Tamsulosin", "Finasteride", "Sildenafil", "Tadalafil", "Oxycodone",
    "Hydrocodone", "Codeine", "Morphine", "Fentanyl", "Buprenorphine",
    "Clonidine", "Methylphenidate", "Amphetamine", "Atomoxetine", "Guanfacine",
    "Quetiapine", "Aripiprazole", "Risperidone", "Olanzapine", "Lamotrigine",
    "Levetiracetam", "Topiramate", "Valproic acid", "Carbamazepine", "Phenytoin",
    "Buspirone", "Mirtazapine", "Venlafaxine", "Paroxetine", "Amitriptyline",
    "Nortriptyline", "Desipramine", "Vortioxetine", "Vilazodone", "Brexpiprazole",
    "Latuda", "Rexulti", "Trintellix", "Belsomra", "Dayvigo",
    "Eliquis", "Xarelto", "Pradaxa", "Savaysa", "Coumadin"
]


class DrugSeeder:
    """Seeds drugs table from RxNorm API"""

    def __init__(self):
        self.db = next(get_db())
        self.service = DrugNormalizationService(self.db)
        self.rxnorm_version = self._get_rxnorm_version()

    def _get_rxnorm_version(self) -> str:
        """Get current RxNorm version from API"""
        try:
            url = f"{RXNORM_BASE_URL}/version.json"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            version = data.get('version', datetime.now().strftime('%Y%m%d'))
            logger.info(f"RxNorm version: {version}")
            return version
        except Exception as e:
            logger.warning(f"Could not get RxNorm version: {str(e)}")
            return datetime.now().strftime('%Y%m%d')

    def seed_drug(self, drug_name: str) -> bool:
        """Seed a single drug from RxNorm"""
        try:
            # Check if already exists
            check_query = text("""
                SELECT COUNT(*) FROM drugs 
                WHERE LOWER(name) = LOWER(:name) OR LOWER(generic_name) = LOWER(:name)
            """)
            result = self.db.execute(check_query, {"name": drug_name}).fetchone()
            if result and result[0] > 0:
                logger.info(f"Drug '{drug_name}' already exists, skipping")
                return True

            # Normalize medication to get RxCUI
            normalized = self.service.normalize_medication(drug_name)
            
            if not normalized.get('rxcui'):
                logger.warning(f"Could not find RxCUI for '{drug_name}'")
                return False

            # Create drug record from RxNorm data
            drug_id = self.service.get_or_create_drug_from_rxcui(normalized['rxcui'])
            
            if drug_id:
                logger.info(f"✓ Seeded drug: {drug_name} (RxCUI: {normalized['rxcui']}, ID: {drug_id})")
                return True
            else:
                logger.error(f"✗ Failed to seed drug: {drug_name}")
                return False

        except Exception as e:
            logger.error(f"Error seeding drug '{drug_name}': {str(e)}")
            return False

    def seed_all(self, drug_list: List[str], rate_limit_delay: float = 0.5):
        """Seed multiple drugs with rate limiting"""
        total = len(drug_list)
        success_count = 0
        fail_count = 0

        logger.info(f"Starting to seed {total} drugs from RxNorm...")

        for i, drug_name in enumerate(drug_list, 1):
            logger.info(f"[{i}/{total}] Processing: {drug_name}")
            
            if self.seed_drug(drug_name):
                success_count += 1
            else:
                fail_count += 1

            # Rate limiting to avoid overwhelming RxNorm API
            time.sleep(rate_limit_delay)

        logger.info(f"""
        ===== Drug Seeding Complete =====
        Total: {total}
        Success: {success_count}
        Failed: {fail_count}
        RxNorm Version: {self.rxnorm_version}
        =================================
        """)

        return success_count, fail_count


def main():
    parser = argparse.ArgumentParser(description='Seed drugs table from RxNorm API')
    parser.add_argument('--limit', type=int, default=100, help='Number of drugs to seed (default: 100)')
    parser.add_argument('--delay', type=float, default=0.5, help='Delay between API calls in seconds (default: 0.5)')
    args = parser.parse_args()

    seeder = DrugSeeder()
    drug_list = COMMON_MEDICATIONS[:args.limit]
    
    success, failed = seeder.seed_all(drug_list, rate_limit_delay=args.delay)
    
    if failed > 0:
        logger.warning(f"Seeding completed with {failed} failures")
        exit(1)
    else:
        logger.info("All drugs seeded successfully!")
        exit(0)


if __name__ == "__main__":
    main()
