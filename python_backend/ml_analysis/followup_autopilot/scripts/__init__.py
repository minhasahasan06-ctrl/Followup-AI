"""
Followup Autopilot Training Scripts

Production-grade ML training and daily operations scripts with:
- HIPAA-compliant logging (no PHI in logs)
- Consent verification before data access
- Model versioning and integrity checks
- Secure model storage
- Comprehensive audit logging

Scripts:
- train_risk_model.py: PyTorch LSTM multi-task risk prediction
- train_adherence_model.py: XGBoost adherence forecasting
- train_engagement_model.py: XGBoost notification timing optimization
- run_daily_aggregation.py: Daily feature aggregation job
- run_daily_autopilot.py: Daily autopilot inference sweep
"""

__version__ = "1.0.0"
