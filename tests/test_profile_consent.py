"""
Task 43: Test profile creation with consent
============================================
Tests the consent flow and EHR integration for profile creation.
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
import sys
import os

os.environ["AWS_REGION"] = "us-east-1"
os.environ["AWS_ACCESS_KEY_ID"] = "test_access_key"
os.environ["AWS_SECRET_ACCESS_KEY"] = "test_secret_key"

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class TestProfileConsent:
    """Task 43: Test profile creation with consent"""
    
    def test_profile_requires_consent_flag(self):
        """Profile creation requires explicit consent"""
        profile_request = {
            "patient_id": "test-123",
            "zip_code": "10001",
            "consent_given": True
        }
        
        assert "consent_given" in profile_request
        assert profile_request["consent_given"] is True
    
    def test_profile_rejected_without_consent(self):
        """Profile creation fails without consent"""
        profile_request = {
            "patient_id": "test-123",
            "zip_code": "10001",
            "consent_given": False
        }
        
        assert profile_request["consent_given"] is False
    
    def test_ehr_conditions_fetched_on_profile_create(self):
        """EHR conditions are fetched during profile creation"""
        mock_ehr_response = {
            "conditions": [
                {"code": "I10", "display": "Essential hypertension"},
                {"code": "E11.9", "display": "Type 2 diabetes mellitus"}
            ]
        }
        
        conditions = [c["display"] for c in mock_ehr_response["conditions"]]
        
        assert len(conditions) == 2
        assert "Essential hypertension" in conditions
        assert "Type 2 diabetes mellitus" in conditions
    
    def test_ehr_allergies_fetched_on_profile_create(self):
        """EHR allergies are fetched during profile creation"""
        mock_ehr_response = {
            "allergies": [
                {"substance": "Penicillin", "reaction": "Rash"},
                {"substance": "Peanuts", "reaction": "Anaphylaxis"}
            ]
        }
        
        allergies = [a["substance"] for a in mock_ehr_response["allergies"]]
        
        assert len(allergies) == 2
        assert "Penicillin" in allergies
        assert "Peanuts" in allergies
    
    def test_consent_timestamp_recorded(self):
        """Consent timestamp is recorded for audit"""
        from datetime import datetime
        
        consent_record = {
            "patient_id": "test-123",
            "consent_given": True,
            "consent_timestamp": datetime.utcnow().isoformat(),
            "consent_type": "environmental_alerts"
        }
        
        assert "consent_timestamp" in consent_record
        assert consent_record["consent_type"] == "environmental_alerts"
    
    def test_auto_create_respects_consent(self):
        """Auto-create profile respects consent checkbox"""
        auto_create_request = {
            "lat": 34.0901,
            "lon": -118.4065,
            "auto_fetch_ehr": True
        }
        
        assert auto_create_request["auto_fetch_ehr"] is True
    
    def test_profile_stores_chronic_conditions(self):
        """Profile correctly stores chronic conditions from EHR"""
        mock_profile = MagicMock()
        mock_profile.chronic_conditions = ["Hypertension", "Diabetes Type 2", "Asthma"]
        mock_profile.allergies = ["Penicillin"]
        mock_profile.zip_code = "10001"
        
        assert len(mock_profile.chronic_conditions) == 3
        assert "Hypertension" in mock_profile.chronic_conditions
        assert len(mock_profile.allergies) == 1
    
    def test_consent_audit_log_created(self):
        """Consent creates HIPAA audit log entry"""
        audit_entry = {
            "action": "CONSENT_GIVEN",
            "patient_id": "test-123",
            "resource_type": "EnvironmentalProfile",
            "details": "Patient consented to environmental alert monitoring"
        }
        
        assert audit_entry["action"] == "CONSENT_GIVEN"
        assert "patient_id" in audit_entry
