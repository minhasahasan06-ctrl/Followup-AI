"""
E.4: Unit test - CohortDSL validation rejects unknown fields
============================================================
Tests that CohortDSL Pydantic model rejects unknown/invalid fields.
"""

import pytest
from pydantic import ValidationError
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator
from enum import Enum


class CohortOperator(str, Enum):
    """Allowed operators for cohort filters"""
    EQUALS = "equals"
    NOT_EQUALS = "not_equals"
    GREATER_THAN = "greater_than"
    LESS_THAN = "less_than"
    GREATER_EQUAL = "greater_equal"
    LESS_EQUAL = "less_equal"
    IN = "in"
    NOT_IN = "not_in"
    BETWEEN = "between"
    IS_NULL = "is_null"
    IS_NOT_NULL = "is_not_null"


class AllowedCohortField(str, Enum):
    """Allowed fields for cohort filtering - no PHI fields"""
    AGE_BUCKET = "age_bucket"
    CONDITION_CODES = "condition_codes"
    RISK_BUCKET = "risk_bucket"
    ENGAGEMENT_BUCKET = "engagement_bucket"
    ADHERENCE_BUCKET = "adherence_bucket"
    TREATMENT_CATEGORY = "treatment_category"
    OUTCOME_CATEGORY = "outcome_category"
    REGION_BUCKET = "region_bucket"
    ENROLLMENT_PERIOD = "enrollment_period"


class CohortFilter(BaseModel):
    """A single filter condition in the cohort DSL"""
    field: AllowedCohortField
    operator: CohortOperator
    value: Optional[str | int | float | List[str]] = None
    
    @field_validator('value')
    @classmethod
    def validate_value_for_operator(cls, v, info):
        operator = info.data.get('operator')
        if operator in [CohortOperator.IS_NULL, CohortOperator.IS_NOT_NULL]:
            if v is not None:
                raise ValueError(f"IS_NULL/IS_NOT_NULL operators don't accept values")
        elif operator == CohortOperator.BETWEEN:
            if not isinstance(v, list) or len(v) != 2:
                raise ValueError("BETWEEN operator requires a list of 2 values")
        elif operator in [CohortOperator.IN, CohortOperator.NOT_IN]:
            if not isinstance(v, list):
                raise ValueError("IN/NOT_IN operators require a list of values")
        return v


class CohortDSL(BaseModel):
    """
    Cohort Definition DSL for safe, validated cohort queries.
    
    SECURITY: Only allows whitelisted fields and operators.
    Rejects any unknown fields to prevent PHI leakage.
    """
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    filters: List[CohortFilter] = Field(default_factory=list)
    logic: str = Field(default="AND", pattern="^(AND|OR)$")
    
    class Config:
        extra = "forbid"
    
    @field_validator('name')
    @classmethod
    def name_no_phi(cls, v):
        phi_patterns = ['patient', 'name', 'email', 'mrn', 'ssn', 'phone']
        if any(pattern in v.lower() for pattern in phi_patterns):
            raise ValueError("Cohort name cannot contain PHI-like patterns")
        return v


class TestCohortDSLValidation:
    """E.4: Test CohortDSL rejects unknown fields"""
    
    def test_valid_cohort_dsl(self):
        """Valid CohortDSL is accepted"""
        dsl = CohortDSL(
            name="High Risk Diabetics",
            description="Patients with high risk and diabetes",
            filters=[
                CohortFilter(
                    field=AllowedCohortField.RISK_BUCKET,
                    operator=CohortOperator.EQUALS,
                    value="high"
                ),
                CohortFilter(
                    field=AllowedCohortField.CONDITION_CODES,
                    operator=CohortOperator.IN,
                    value=["E11", "E10"]
                )
            ],
            logic="AND"
        )
        assert dsl.name == "High Risk Diabetics"
        assert len(dsl.filters) == 2
    
    def test_rejects_extra_fields(self):
        """CohortDSL rejects unknown fields"""
        with pytest.raises(ValidationError) as exc_info:
            CohortDSL(
                name="Test Cohort",
                filters=[],
                unknown_field="should fail",
                another_unknown=123
            )
        assert "extra" in str(exc_info.value).lower()
    
    def test_rejects_phi_field_in_filter(self):
        """CohortFilter rejects PHI field names"""
        with pytest.raises(ValidationError):
            CohortFilter(
                field="patient_id",
                operator=CohortOperator.EQUALS,
                value="12345"
            )
    
    def test_rejects_name_field_in_filter(self):
        """CohortFilter rejects 'name' field"""
        with pytest.raises(ValidationError):
            CohortFilter(
                field="name",
                operator=CohortOperator.EQUALS,
                value="John"
            )
    
    def test_rejects_email_field_in_filter(self):
        """CohortFilter rejects 'email' field"""
        with pytest.raises(ValidationError):
            CohortFilter(
                field="email",
                operator=CohortOperator.EQUALS,
                value="test@example.com"
            )
    
    def test_rejects_dob_field_in_filter(self):
        """CohortFilter rejects 'dob' field"""
        with pytest.raises(ValidationError):
            CohortFilter(
                field="dob",
                operator=CohortOperator.EQUALS,
                value="1990-01-01"
            )
    
    def test_rejects_address_field_in_filter(self):
        """CohortFilter rejects 'address' field"""
        with pytest.raises(ValidationError):
            CohortFilter(
                field="address",
                operator=CohortOperator.EQUALS,
                value="123 Main St"
            )
    
    def test_rejects_invalid_operator(self):
        """CohortFilter rejects invalid operators"""
        with pytest.raises(ValidationError):
            CohortFilter(
                field=AllowedCohortField.RISK_BUCKET,
                operator="LIKE",
                value="%test%"
            )
    
    def test_rejects_invalid_logic(self):
        """CohortDSL rejects invalid logic operators"""
        with pytest.raises(ValidationError):
            CohortDSL(
                name="Test",
                filters=[],
                logic="XOR"
            )
    
    def test_between_requires_two_values(self):
        """BETWEEN operator requires exactly 2 values"""
        with pytest.raises(ValidationError) as exc_info:
            CohortFilter(
                field=AllowedCohortField.AGE_BUCKET,
                operator=CohortOperator.BETWEEN,
                value="30-40"
            )
        assert "2 values" in str(exc_info.value)
    
    def test_in_requires_list(self):
        """IN operator requires a list of values"""
        with pytest.raises(ValidationError) as exc_info:
            CohortFilter(
                field=AllowedCohortField.CONDITION_CODES,
                operator=CohortOperator.IN,
                value="E11"
            )
        assert "list" in str(exc_info.value).lower()
    
    def test_is_null_rejects_value(self):
        """IS_NULL operator should not have a value"""
        with pytest.raises(ValidationError):
            CohortFilter(
                field=AllowedCohortField.RISK_BUCKET,
                operator=CohortOperator.IS_NULL,
                value="something"
            )
    
    def test_rejects_phi_in_cohort_name(self):
        """CohortDSL rejects PHI-like patterns in name"""
        with pytest.raises(ValidationError) as exc_info:
            CohortDSL(
                name="Patient John's Cohort",
                filters=[]
            )
        assert "PHI" in str(exc_info.value)
    
    def test_name_length_validation(self):
        """CohortDSL validates name length"""
        with pytest.raises(ValidationError):
            CohortDSL(name="", filters=[])
        
        with pytest.raises(ValidationError):
            CohortDSL(name="x" * 101, filters=[])
    
    def test_description_length_validation(self):
        """CohortDSL validates description length"""
        with pytest.raises(ValidationError):
            CohortDSL(
                name="Valid Name",
                description="x" * 501,
                filters=[]
            )


class TestAllowedFields:
    """E.4: Test only allowed fields are accepted"""
    
    def test_all_allowed_fields_work(self):
        """All AllowedCohortField values can be used"""
        for field in AllowedCohortField:
            filter = CohortFilter(
                field=field,
                operator=CohortOperator.IS_NOT_NULL
            )
            assert filter.field == field
    
    def test_allowed_fields_are_non_phi(self):
        """All allowed fields are safe, non-PHI fields"""
        phi_indicators = ['name', 'email', 'phone', 'address', 'ssn', 'mrn', 'dob', 'patient_id']
        
        for field in AllowedCohortField:
            for phi in phi_indicators:
                assert phi not in field.value.lower(), f"{field.value} contains PHI indicator {phi}"
