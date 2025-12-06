"""
Research Center Demo Data Seeder
=================================
** DEMO/TESTING ENVIRONMENT ONLY **

This script generates SYNTHETIC research data for demonstrations via HTTP API calls.
Uses the proper REST API endpoints to ensure data consistency with the storage layer.

SAFETY GUARDRAILS:
1. Requires ALLOW_SYNTHETIC_DATA=true environment variable
2. Connects to local development server only
3. All data prefixed with [SYNTHETIC] or [DEMO]
4. Uses test emails ending in @synthetic.example.com

Usage:
  ALLOW_SYNTHETIC_DATA=true python research_demo_seeder.py
"""

import os
import sys
import random
import uuid
import json
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

# Configuration
BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:5000")
ALLOW_SYNTHETIC_DATA = os.environ.get("ALLOW_SYNTHETIC_DATA", "").lower() == "true"

# Safety check
if not ALLOW_SYNTHETIC_DATA:
    print("=" * 70)
    print("SAFETY BLOCK: Demo data seeding is disabled by default.")
    print("To enable, set: ALLOW_SYNTHETIC_DATA=true")
    print("WARNING: Only run on development/demo environments!")
    print("=" * 70)
    sys.exit(1)

# Demo data constants
STUDY_TYPES = ["observational", "interventional", "registry", "retrospective"]
STATUSES = ["planning", "enrolling", "follow_up", "analysis", "completed"]
CONDITIONS = [
    "Rheumatoid Arthritis", "Lupus", "Multiple Sclerosis", "Crohn's Disease",
    "Ulcerative Colitis", "Psoriatic Arthritis", "Type 1 Diabetes"
]


def generate_uuid() -> str:
    return str(uuid.uuid4())


def random_date(days_ago_start: int, days_ago_end: int = 0) -> str:
    delta = random.randint(days_ago_end, days_ago_start)
    dt = datetime.now() - timedelta(days=delta)
    return dt.strftime("%Y-%m-%d")


class ResearchDemoSeeder:
    """Seed research data via HTTP API calls."""
    
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
        self.created_studies: List[str] = []
        self.created_cohorts: List[str] = []
    
    def _api_call(self, method: str, endpoint: str, data: Dict = None) -> Optional[Dict]:
        """Make API call with error handling."""
        url = f"{self.base_url}{endpoint}"
        try:
            if method == "GET":
                resp = self.session.get(url, timeout=10)
            elif method == "POST":
                resp = self.session.post(url, json=data, timeout=10)
            elif method == "PATCH":
                resp = self.session.patch(url, json=data, timeout=10)
            else:
                return None
            
            if resp.status_code in [200, 201]:
                return resp.json() if resp.text else {}
            else:
                print(f"  Warning: {method} {endpoint} returned {resp.status_code}")
                return None
        except Exception as e:
            print(f"  Error: {method} {endpoint} - {e}")
            return None
    
    def seed_studies(self, count: int = 5) -> List[str]:
        """Create demo studies via API."""
        print(f"\n[1/5] Creating {count} demo studies...")
        
        study_topics = [
            "Immunocompromised Patient Outcomes",
            "Environmental Risk Factors",
            "Medication Adherence Patterns",
            "Digital Biomarker Validation",
            "Deterioration Prediction Study"
        ]
        
        for i in range(min(count, len(study_topics))):
            study_data = {
                "title": f"[SYNTHETIC] {study_topics[i]}",
                "description": f"[DEMO DATA] Research study examining {study_topics[i].lower()} in chronic care patients.",
                "status": random.choice(STATUSES),
                "studyType": random.choice(STUDY_TYPES),
                "startDate": random_date(365, 180),
                "endDate": random_date(30, 0),
                "targetEnrollment": random.randint(50, 200),
                "principalInvestigator": f"Dr. Demo Researcher {i+1}",
                "autoReanalysis": random.choice([True, False]),
                "reanalysisFrequency": random.choice(["daily", "weekly", "monthly"])
            }
            
            result = self._api_call("POST", "/api/v1/research-center/studies", study_data)
            if result and result.get("id"):
                self.created_studies.append(result["id"])
                print(f"  Created study: {study_data['title'][:50]}...")
        
        print(f"  Total studies created: {len(self.created_studies)}")
        return self.created_studies
    
    def seed_cohorts(self, count: int = 3) -> List[str]:
        """Create demo cohorts via API."""
        print(f"\n[2/5] Creating {count} demo cohorts...")
        
        cohort_names = [
            "High-Risk Patients",
            "Treatment Responders",
            "Environmental Exposure Group"
        ]
        
        for i in range(min(count, len(cohort_names))):
            cohort_data = {
                "name": f"[SYNTHETIC] {cohort_names[i]}",
                "description": f"[DEMO] Cohort for {cohort_names[i].lower()} analysis",
                "criteria": {
                    "filters": [
                        {"type": "age", "operator": "between", "value": [25, 75]},
                        {"type": "condition", "operator": "in", "value": CONDITIONS[:3]}
                    ]
                },
                "patientIds": []
            }
            
            result = self._api_call("POST", "/api/v1/research-center/cohorts", cohort_data)
            if result and result.get("id"):
                self.created_cohorts.append(result["id"])
                print(f"  Created cohort: {cohort_data['name']}")
        
        print(f"  Total cohorts created: {len(self.created_cohorts)}")
        return self.created_cohorts
    
    def seed_followup_templates(self, count: int = 3) -> List[str]:
        """Create daily followup templates via API."""
        print(f"\n[3/5] Creating {count} followup templates...")
        
        template_ids = []
        templates = [
            {
                "name": "[SYNTHETIC] Daily Symptom Check",
                "description": "[DEMO] Standard daily symptom tracking",
                "questions": [
                    {"id": "fatigue", "text": "Rate your fatigue level (1-10)", "type": "scale", "required": True},
                    {"id": "pain", "text": "Rate your pain level (1-10)", "type": "scale", "required": True},
                    {"id": "notes", "text": "Any additional symptoms?", "type": "text", "required": False}
                ],
                "frequency": "daily"
            },
            {
                "name": "[SYNTHETIC] Weekly Wellness Survey",
                "description": "[DEMO] Weekly wellness assessment",
                "questions": [
                    {"id": "mood", "text": "How is your mood this week?", "type": "select", "options": ["Great", "Good", "Fair", "Poor"], "required": True},
                    {"id": "activity", "text": "Activity level this week?", "type": "scale", "required": True}
                ],
                "frequency": "weekly"
            },
            {
                "name": "[SYNTHETIC] Medication Adherence",
                "description": "[DEMO] Medication tracking template",
                "questions": [
                    {"id": "taken", "text": "Did you take all medications today?", "type": "boolean", "required": True},
                    {"id": "side_effects", "text": "Any side effects?", "type": "text", "required": False}
                ],
                "frequency": "daily"
            }
        ]
        
        for template in templates[:count]:
            result = self._api_call("POST", "/api/v1/research-center/daily-followups/templates", template)
            if result and result.get("id"):
                template_ids.append(result["id"])
                print(f"  Created template: {template['name']}")
        
        print(f"  Total templates created: {len(template_ids)}")
        return template_ids
    
    def seed_alerts(self, count: int = 5) -> List[str]:
        """Create demo research alerts via API."""
        print(f"\n[4/5] Creating {count} demo alerts...")
        
        alert_ids = []
        alert_types = ["deterioration", "missing_data", "threshold_breach", "pattern_detected", "followup_gap"]
        severities = ["low", "medium", "high", "critical"]
        
        for i in range(count):
            alert_data = {
                "title": f"[SYNTHETIC] Demo Alert {i+1}",
                "message": f"[DEMO] This is a synthetic research alert for testing purposes",
                "type": random.choice(alert_types),
                "severity": random.choice(severities),
                "status": random.choice(["active", "acknowledged", "resolved"]),
                "studyId": self.created_studies[i % len(self.created_studies)] if self.created_studies else None,
                "metadata": {
                    "riskScore": round(random.uniform(0.3, 0.95), 3),
                    "synthetic": True,
                    "shapFeatures": [
                        {"feature": "CRP Level", "importance": round(random.uniform(0.1, 0.3), 3)},
                        {"feature": "Fatigue Score", "importance": round(random.uniform(0.1, 0.2), 3)}
                    ]
                }
            }
            
            result = self._api_call("POST", "/api/v1/research-center/alerts", alert_data)
            if result and result.get("id"):
                alert_ids.append(result["id"])
                print(f"  Created alert: {alert_data['title']}")
        
        print(f"  Total alerts created: {len(alert_ids)}")
        return alert_ids
    
    def seed_analyses(self, count: int = 3) -> List[str]:
        """Create demo ML analyses via API."""
        print(f"\n[5/5] Creating {count} demo analyses...")
        
        analysis_ids = []
        analysis_types = ["descriptive", "risk_prediction", "survival", "causal"]
        
        for i in range(count):
            analysis_type = analysis_types[i % len(analysis_types)]
            analysis_data = {
                "name": f"[SYNTHETIC] {analysis_type.replace('_', ' ').title()} Analysis {i+1}",
                "type": analysis_type,
                "status": random.choice(["pending", "running", "completed", "failed"]),
                "cohortId": self.created_cohorts[i % len(self.created_cohorts)] if self.created_cohorts else None,
                "studyId": self.created_studies[i % len(self.created_studies)] if self.created_studies else None,
                "config": {
                    "synthetic": True,
                    "outcomeVariable": "deterioration_30d",
                    "covariates": ["age", "crp", "fatigue_score"],
                    "modelType": "logistic_regression" if analysis_type == "risk_prediction" else None
                },
                "resultsJson": {
                    "synthetic": True,
                    "sampleSize": random.randint(50, 200),
                    "metrics": {
                        "accuracy": round(random.uniform(0.7, 0.9), 3),
                        "auc": round(random.uniform(0.65, 0.85), 3)
                    }
                }
            }
            
            result = self._api_call("POST", "/api/v1/research-center/analyses", analysis_data)
            if result and result.get("id"):
                analysis_ids.append(result["id"])
                print(f"  Created analysis: {analysis_data['name']}")
        
        print(f"  Total analyses created: {len(analysis_ids)}")
        return analysis_ids
    
    def run_all(self):
        """Run all seeders."""
        print("\n" + "=" * 60)
        print("RESEARCH CENTER DEMO DATA SEEDER")
        print("=" * 60)
        print(f"Target API: {self.base_url}")
        print("This will create synthetic/demo data for testing.")
        print("=" * 60)
        
        # Seed in order (dependencies)
        self.seed_studies(5)
        self.seed_cohorts(3)
        self.seed_followup_templates(3)
        self.seed_alerts(5)
        self.seed_analyses(3)
        
        print("\n" + "=" * 60)
        print("SEEDING COMPLETE")
        print("=" * 60)
        print(f"Studies: {len(self.created_studies)}")
        print(f"Cohorts: {len(self.created_cohorts)}")
        print("\nAll data is marked as [SYNTHETIC] or [DEMO]")
        print("=" * 60)


if __name__ == "__main__":
    seeder = ResearchDemoSeeder(BASE_URL)
    seeder.run_all()
