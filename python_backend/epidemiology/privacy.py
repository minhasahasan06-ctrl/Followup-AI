"""
Privacy Guards for Epidemiology Research
=========================================
Implements HIPAA-compliant privacy protections including:
- Minimum cell size suppression
- Differential privacy (optional)
- Role-based access control
- Data de-identification
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import math
import random
import logging

logger = logging.getLogger(__name__)

MIN_CELL_SIZE = 10
DIFFERENTIAL_PRIVACY_EPSILON = 1.0


@dataclass
class PrivacyConfig:
    """Privacy configuration for analysis outputs"""
    min_cell_size: int = MIN_CELL_SIZE
    enable_differential_privacy: bool = False
    epsilon: float = DIFFERENTIAL_PRIVACY_EPSILON
    suppress_small_cells: bool = True
    round_counts: bool = False
    rounding_base: int = 5


class PrivacyGuard:
    """Enforces privacy protections on aggregated outputs"""
    
    def __init__(self, config: Optional[PrivacyConfig] = None):
        self.config = config or PrivacyConfig()
    
    def check_cell_size(self, count: int) -> bool:
        """Check if count meets minimum cell size requirement"""
        return count >= self.config.min_cell_size
    
    def suppress_if_small(self, value: Any, count: int, replacement: str = "<suppressed>") -> Any:
        """Suppress value if count is below threshold"""
        if not self.check_cell_size(count):
            return replacement
        return value
    
    def apply_laplace_noise(self, value: float, sensitivity: float = 1.0) -> float:
        """Apply Laplace noise for differential privacy"""
        if not self.config.enable_differential_privacy:
            return value
        
        scale = sensitivity / self.config.epsilon
        noise = random.uniform(-0.5, 0.5)
        noise = -scale * math.copysign(1, noise) * math.log(1 - 2 * abs(noise))
        return value + noise
    
    def round_to_base(self, count: int) -> int:
        """Round count to nearest base for additional privacy"""
        if not self.config.round_counts:
            return count
        
        base = self.config.rounding_base
        return base * round(count / base)
    
    def sanitize_signal(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """Apply privacy protections to a drug outcome signal"""
        n_patients = signal.get('n_patients', 0)
        n_events = signal.get('n_events', 0)
        
        if not self.check_cell_size(n_patients) or not self.check_cell_size(n_events):
            return {
                'drug_code': signal.get('drug_code'),
                'drug_name': signal.get('drug_name'),
                'outcome_code': signal.get('outcome_code'),
                'outcome_name': signal.get('outcome_name'),
                'patient_location_id': signal.get('patient_location_id'),
                'suppressed': True,
                'reason': 'Insufficient sample size for privacy protection'
            }
        
        sanitized = signal.copy()
        
        if self.config.enable_differential_privacy:
            sanitized['estimate'] = round(self.apply_laplace_noise(signal.get('estimate', 0), 0.1), 4)
            sanitized['ci_lower'] = round(self.apply_laplace_noise(signal.get('ci_lower', 0), 0.1), 4)
            sanitized['ci_upper'] = round(self.apply_laplace_noise(signal.get('ci_upper', 0), 0.1), 4)
        
        if self.config.round_counts:
            sanitized['n_patients'] = self.round_to_base(n_patients)
            sanitized['n_events'] = self.round_to_base(n_events)
        
        sanitized['suppressed'] = False
        return sanitized
    
    def sanitize_summary(self, summary: Dict[str, Any]) -> Dict[str, Any]:
        """Apply privacy protections to a drug outcome summary"""
        n_patients = summary.get('n_patients', 0)
        
        if not self.check_cell_size(n_patients):
            return {
                'drug_code': summary.get('drug_code'),
                'drug_name': summary.get('drug_name'),
                'outcome_code': summary.get('outcome_code'),
                'outcome_name': summary.get('outcome_name'),
                'patient_location_id': summary.get('patient_location_id'),
                'suppressed': True,
                'reason': 'Insufficient sample size for privacy protection'
            }
        
        sanitized = summary.copy()
        
        count_fields = ['n_patients', 'n_events', 'n_exposed', 'n_exposed_events', 
                       'n_unexposed', 'n_unexposed_events']
        
        for field in count_fields:
            if field in sanitized:
                count = sanitized[field]
                if not self.check_cell_size(count):
                    sanitized[field] = '<10'
                elif self.config.round_counts:
                    sanitized[field] = self.round_to_base(count)
        
        if self.config.enable_differential_privacy:
            if 'incidence_exposed' in sanitized and sanitized['incidence_exposed']:
                sanitized['incidence_exposed'] = round(
                    self.apply_laplace_noise(sanitized['incidence_exposed'], 0.01), 4
                )
            if 'incidence_unexposed' in sanitized and sanitized['incidence_unexposed']:
                sanitized['incidence_unexposed'] = round(
                    self.apply_laplace_noise(sanitized['incidence_unexposed'], 0.01), 4
                )
        
        sanitized['suppressed'] = False
        return sanitized
    
    def filter_signals_by_privacy(self, signals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter and sanitize a list of signals"""
        result = []
        for signal in signals:
            sanitized = self.sanitize_signal(signal)
            if not self.config.suppress_small_cells or not sanitized.get('suppressed'):
                result.append(sanitized)
        return result
    
    def validate_for_llm(self, data: Dict[str, Any]) -> bool:
        """Validate data is safe to send to LLM (no patient identifiers)"""
        unsafe_keys = {'patient_id', 'patient_ids', 'name', 'email', 'phone', 
                       'address', 'ssn', 'date_of_birth', 'mrn'}
        
        def check_recursive(obj, path=""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if key.lower() in unsafe_keys:
                        logger.warning(f"Unsafe key found at {path}.{key}")
                        return False
                    if not check_recursive(value, f"{path}.{key}"):
                        return False
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    if not check_recursive(item, f"{path}[{i}]"):
                        return False
            return True
        
        return check_recursive(data)


class RoleBasedAccessControl:
    """RBAC for epidemiology data access"""
    
    ADMIN_ROLES = {'admin', 'superadmin'}
    RESEARCHER_ROLES = {'researcher', 'epidemiologist', 'data_scientist'}
    DOCTOR_ROLES = {'doctor', 'physician', 'clinician'}
    VIEWER_ROLES = {'viewer', 'observer'}
    
    @classmethod
    def can_access_all_locations(cls, user_role: str) -> bool:
        """Check if user can access data from all locations"""
        return user_role in cls.ADMIN_ROLES or user_role in cls.RESEARCHER_ROLES
    
    @classmethod
    def can_access_my_patients(cls, user_role: str) -> bool:
        """Check if user can access their own patients' data"""
        return user_role in cls.DOCTOR_ROLES or user_role in cls.ADMIN_ROLES
    
    @classmethod
    def can_export_data(cls, user_role: str) -> bool:
        """Check if user can export aggregated data"""
        return user_role in cls.ADMIN_ROLES or user_role in cls.RESEARCHER_ROLES
    
    @classmethod
    def can_run_ml_analysis(cls, user_role: str) -> bool:
        """Check if user can run ML analyses"""
        return user_role in cls.ADMIN_ROLES or user_role in cls.RESEARCHER_ROLES
    
    @classmethod
    def get_allowed_scopes(cls, user_role: str) -> List[str]:
        """Get allowed analysis scopes for user role"""
        if user_role in cls.ADMIN_ROLES or user_role in cls.RESEARCHER_ROLES:
            return ['all', 'my_patients', 'research_cohort']
        elif user_role in cls.DOCTOR_ROLES:
            return ['my_patients']
        return []
