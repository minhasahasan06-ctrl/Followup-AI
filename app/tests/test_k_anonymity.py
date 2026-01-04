"""
E.2: Unit test - cohort preview enforces k-anon
===============================================
Tests that cohort operations return "insufficient size" when count < 25.
"""

import pytest
from app.services.privacy_firewall import k_anon_check
from app.services.tinker_privacy_firewall import TinkerPrivacyFirewall, PrivacyConfig


class TestKAnonymityCheck:
    """E.2: Test k-anonymity enforcement returns 'insufficient size' when count < 25"""
    
    def test_k_anon_check_passes_at_25(self):
        """k_anon_check passes when count == 25"""
        k_anon_check(25, k=25)
    
    def test_k_anon_check_passes_above_25(self):
        """k_anon_check passes when count > 25"""
        k_anon_check(100, k=25)
        k_anon_check(1000, k=25)
    
    def test_k_anon_check_fails_below_25(self):
        """k_anon_check raises ValueError when count < 25"""
        with pytest.raises(ValueError, match="K-ANONYMITY VIOLATION"):
            k_anon_check(24, k=25)
    
    def test_k_anon_check_fails_at_zero(self):
        """k_anon_check raises ValueError when count = 0"""
        with pytest.raises(ValueError, match="K-ANONYMITY VIOLATION"):
            k_anon_check(0, k=25)
    
    def test_k_anon_check_fails_at_one(self):
        """k_anon_check raises ValueError when count = 1"""
        with pytest.raises(ValueError, match="K-ANONYMITY VIOLATION"):
            k_anon_check(1, k=25)
    
    def test_k_anon_violation_message_includes_counts(self):
        """Error message includes both count and k values"""
        try:
            k_anon_check(10, k=25)
            pytest.fail("Should have raised ValueError")
        except ValueError as e:
            assert "10" in str(e)
            assert "25" in str(e)


class TestPrivacyFirewallKAnon:
    """E.2: Test TinkerPrivacyFirewall k-anonymity methods"""
    
    @pytest.fixture
    def firewall(self):
        config = PrivacyConfig(k_anonymity_threshold=25)
        return TinkerPrivacyFirewall(config)
    
    def test_check_k_anonymity_returns_true_at_25(self, firewall):
        """check_k_anonymity returns True when count >= 25"""
        assert firewall.check_k_anonymity(25) is True
        assert firewall.check_k_anonymity(100) is True
    
    def test_check_k_anonymity_returns_false_below_25(self, firewall):
        """check_k_anonymity returns False when count < 25"""
        assert firewall.check_k_anonymity(24) is False
        assert firewall.check_k_anonymity(0) is False
    
    def test_require_k_anonymity_raises_on_failure(self, firewall):
        """require_k_anonymity raises ValueError on failure"""
        with pytest.raises(ValueError, match="K-anonymity violation"):
            firewall.require_k_anonymity(10, context="cohort_preview")
    
    def test_require_k_anonymity_includes_context(self, firewall):
        """Error message includes operation context"""
        try:
            firewall.require_k_anonymity(5, context="cohort_preview")
            pytest.fail("Should have raised ValueError")
        except ValueError as e:
            assert "cohort_preview" in str(e)
    
    def test_enforce_k_anonymity_suppresses_data(self, firewall):
        """enforce_k_anonymity suppresses data when count < k"""
        data = {"patient_count": 10, "average_age": 45}
        result, passed = firewall.enforce_k_anonymity(data)
        
        assert passed is False
        assert result.get("suppressed") is True
        assert result.get("reason") == "k_anonymity"
        assert result.get("threshold") == 25
    
    def test_enforce_k_anonymity_passes_data(self, firewall):
        """enforce_k_anonymity returns data when count >= k"""
        data = {"patient_count": 100, "average_age": 45}
        result, passed = firewall.enforce_k_anonymity(data)
        
        assert passed is True
        assert result == data
    
    def test_suppress_small_cells_replaces_low_counts(self, firewall):
        """suppress_small_cells replaces counts < k with marker"""
        aggregates = {
            "condition_a": 100,
            "condition_b": 10,
            "condition_c": 50,
            "condition_d": 3,
        }
        result = firewall.suppress_small_cells(aggregates)
        
        assert result["condition_a"] == 100
        assert result["condition_b"] == "<k"
        assert result["condition_c"] == 50
        assert result["condition_d"] == "<k"
    
    def test_compute_safe_aggregates_suppresses_small_n(self, firewall):
        """compute_safe_aggregates suppresses when n < k"""
        values = [1, 2, 3, 4, 5]
        result = firewall.compute_safe_aggregates(values)
        
        assert result.get("suppressed") is True
        assert result.get("reason") == "k_anonymity"
        assert result.get("count") == 5
    
    def test_compute_safe_aggregates_returns_stats(self, firewall):
        """compute_safe_aggregates returns statistics when n >= k"""
        values = list(range(30))
        result = firewall.compute_safe_aggregates(values)
        
        assert result.get("suppressed", False) is False
        assert result.get("count") == 30
        assert "mean" in result
        assert "std" in result
        assert "min" in result
        assert "max" in result


class TestCohortPreviewKAnon:
    """E.2: Test cohort preview specific k-anonymity behavior"""
    
    @pytest.fixture
    def firewall(self):
        config = PrivacyConfig(k_anonymity_threshold=25)
        return TinkerPrivacyFirewall(config)
    
    def test_transform_cohort_query_fails_on_small_cohort(self, firewall):
        """transform_cohort_query returns None for cohorts < k"""
        query = {"condition": "diabetes"}
        result, audit = firewall.transform_cohort_query(query, patient_count=10)
        
        assert result is None
        assert audit.k_anon_passed is False
    
    def test_transform_cohort_query_passes_large_cohort(self, firewall):
        """transform_cohort_query returns data for cohorts >= k"""
        query = {"condition": "diabetes"}
        result, audit = firewall.transform_cohort_query(query, patient_count=100)
        
        assert result is not None
        assert audit.k_anon_passed is True
    
    def test_compute_category_distribution_suppresses(self, firewall):
        """compute_category_distribution suppresses small n"""
        categories = ["a", "b", "c", "a", "b"]
        result = firewall.compute_category_distribution(categories)
        
        assert result.get("suppressed") is True
        assert result.get("total") == 5
    
    def test_compute_category_distribution_returns_data(self, firewall):
        """compute_category_distribution returns distribution for large n"""
        categories = ["a"] * 30 + ["b"] * 20
        result = firewall.compute_category_distribution(categories)
        
        assert result.get("suppressed") is False
        assert result.get("total") == 50
        assert "distribution" in result
