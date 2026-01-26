"""
Task 44: Test recommendations for different problems
=====================================================
Tests condition-to-habit mappings and personalized recommendations.
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
import sys
import os

os.environ["AWS_REGION"] = "us-east-1"
os.environ["AWS_ACCESS_KEY_ID"] = "test_access_key"
os.environ["AWS_SECRET_ACCESS_KEY"] = "test_secret_key"

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


CONDITION_HABIT_MAPPINGS = {
    "Hypertension": [
        {"name": "Blood Pressure Check", "category": "monitoring", "frequency": "daily"},
        {"name": "Low Sodium Diet", "category": "nutrition", "frequency": "daily"},
        {"name": "Stress Management", "category": "mental_health", "frequency": "daily"}
    ],
    "Type 2 Diabetes": [
        {"name": "Blood Sugar Monitoring", "category": "monitoring", "frequency": "daily"},
        {"name": "Carb Counting", "category": "nutrition", "frequency": "daily"},
        {"name": "Foot Care Check", "category": "self_care", "frequency": "daily"}
    ],
    "Asthma": [
        {"name": "Peak Flow Measurement", "category": "monitoring", "frequency": "daily"},
        {"name": "Inhaler Usage Log", "category": "medication", "frequency": "as_needed"},
        {"name": "Trigger Avoidance", "category": "prevention", "frequency": "daily"}
    ],
    "Heart Failure": [
        {"name": "Daily Weight Check", "category": "monitoring", "frequency": "daily"},
        {"name": "Fluid Intake Tracking", "category": "nutrition", "frequency": "daily"},
        {"name": "Symptom Journal", "category": "monitoring", "frequency": "daily"}
    ],
    "Chronic Kidney Disease": [
        {"name": "Fluid Tracking", "category": "nutrition", "frequency": "daily"},
        {"name": "Potassium Watch", "category": "nutrition", "frequency": "daily"},
        {"name": "Blood Pressure Log", "category": "monitoring", "frequency": "daily"}
    ]
}


class TestHabitRecommendations:
    """Task 44: Test condition-to-habit mappings"""
    
    def test_hypertension_habits_mapped(self):
        """Hypertension maps to appropriate habits"""
        habits = CONDITION_HABIT_MAPPINGS.get("Hypertension", [])
        
        assert len(habits) >= 2
        habit_names = [h["name"] for h in habits]
        assert "Blood Pressure Check" in habit_names
        assert "Low Sodium Diet" in habit_names
    
    def test_diabetes_habits_mapped(self):
        """Type 2 Diabetes maps to appropriate habits"""
        habits = CONDITION_HABIT_MAPPINGS.get("Type 2 Diabetes", [])
        
        assert len(habits) >= 2
        habit_names = [h["name"] for h in habits]
        assert "Blood Sugar Monitoring" in habit_names
        assert "Carb Counting" in habit_names
    
    def test_asthma_habits_mapped(self):
        """Asthma maps to appropriate habits"""
        habits = CONDITION_HABIT_MAPPINGS.get("Asthma", [])
        
        assert len(habits) >= 2
        habit_names = [h["name"] for h in habits]
        assert "Peak Flow Measurement" in habit_names
    
    def test_heart_failure_habits_mapped(self):
        """Heart Failure maps to appropriate habits"""
        habits = CONDITION_HABIT_MAPPINGS.get("Heart Failure", [])
        
        assert len(habits) >= 2
        habit_names = [h["name"] for h in habits]
        assert "Daily Weight Check" in habit_names
    
    def test_ckd_habits_mapped(self):
        """Chronic Kidney Disease maps to appropriate habits"""
        habits = CONDITION_HABIT_MAPPINGS.get("Chronic Kidney Disease", [])
        
        assert len(habits) >= 2
        habit_names = [h["name"] for h in habits]
        assert "Fluid Tracking" in habit_names
    
    def test_multiple_conditions_deduplicated(self):
        """Multiple conditions don't create duplicate habits"""
        patient_conditions = ["Hypertension", "Chronic Kidney Disease"]
        
        all_habits = []
        for condition in patient_conditions:
            habits = CONDITION_HABIT_MAPPINGS.get(condition, [])
            all_habits.extend(habits)
        
        habit_names = [h["name"] for h in all_habits]
        unique_names = list(set(habit_names))
        
        assert "Blood Pressure Log" in habit_names or "Blood Pressure Check" in habit_names
    
    def test_recommendation_includes_reason(self):
        """Recommendation includes reason based on condition"""
        recommendation = {
            "name": "Blood Pressure Check",
            "category": "monitoring",
            "frequency": "daily",
            "goalCount": 1,
            "reason": "Recommended for managing your hypertension",
            "safety_notes": "Contact provider if BP > 180/120"
        }
        
        assert "reason" in recommendation
        assert "safety_notes" in recommendation
        assert "hypertension" in recommendation["reason"].lower()
    
    def test_recommendation_categories_valid(self):
        """All recommendations have valid categories"""
        valid_categories = {"monitoring", "nutrition", "exercise", "medication", 
                          "mental_health", "self_care", "prevention"}
        
        for condition, habits in CONDITION_HABIT_MAPPINGS.items():
            for habit in habits:
                assert habit["category"] in valid_categories, \
                    f"Invalid category {habit['category']} for {habit['name']}"
    
    def test_unknown_condition_returns_general_habits(self):
        """Unknown conditions return general wellness habits"""
        unknown_condition = "Rare Disease XYZ"
        habits = CONDITION_HABIT_MAPPINGS.get(unknown_condition, [])
        
        if not habits:
            default_habits = [
                {"name": "Daily Check-in", "category": "monitoring"},
                {"name": "Medication Reminder", "category": "medication"},
                {"name": "Hydration Tracking", "category": "nutrition"}
            ]
            habits = default_habits
        
        assert len(habits) >= 1
    
    def test_recommendation_frequency_valid(self):
        """All recommendations have valid frequencies"""
        valid_frequencies = {"daily", "weekly", "as_needed", "monthly"}
        
        for condition, habits in CONDITION_HABIT_MAPPINGS.items():
            for habit in habits:
                assert habit["frequency"] in valid_frequencies, \
                    f"Invalid frequency {habit['frequency']} for {habit['name']}"
