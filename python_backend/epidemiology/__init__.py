"""
Epidemiology Research Platform
==============================
State-of-the-art AI-driven epidemiology research modules including:
- Pharmaco-epidemiology (drug safety signals)
- Infectious disease epidemiology (outbreaks, R0, contact tracing)
- Vaccine epidemiology (coverage, effectiveness, adverse events)
- Occupational epidemiology
- Genetic/molecular epidemiology hooks
"""

from .models import (
    DrugPrescription,
    DrugExposure,
    AdverseEvent,
    DrugOutcomeSummary,
    DrugOutcomeSignal,
    InfectiousEvent,
    ContactTrace,
    Outbreak,
    ReproductionNumber,
    SerologyResult,
    Immunization,
    VaccineAdverseEvent,
    BiobankSample,
    GeneticVariant,
    GWASResult
)

from .privacy import PrivacyGuard, MIN_CELL_SIZE
from .audit import EpidemiologyAuditLogger
from .auth import verify_epidemiology_auth, AuthenticatedUser, Role

__all__ = [
    'DrugPrescription', 'DrugExposure', 'AdverseEvent',
    'DrugOutcomeSummary', 'DrugOutcomeSignal',
    'InfectiousEvent', 'ContactTrace', 'Outbreak', 
    'ReproductionNumber', 'SerologyResult',
    'Immunization', 'VaccineAdverseEvent',
    'BiobankSample', 'GeneticVariant', 'GWASResult',
    'PrivacyGuard', 'MIN_CELL_SIZE', 'EpidemiologyAuditLogger',
    'verify_epidemiology_auth', 'AuthenticatedUser', 'Role'
]
