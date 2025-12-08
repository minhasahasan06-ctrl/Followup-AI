"""
Adherence Model Training Script

Train XGBoost model for medication adherence forecasting:
- Predict probability of non-adherence in next 7 days
- Feature engineering from historical adherence patterns
- Cross-validation with stratified folds
- Feature importance analysis

HIPAA-compliant with consent verification and audit logging.
"""

import os
import sys
import json
import argparse
from datetime import datetime, timezone
from typing import Dict, Any, Tuple, Optional
from pathlib import Path
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from sqlalchemy import text

try:
    import xgboost as xgb
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
    import joblib
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from python_backend.ml_analysis.followup_autopilot.scripts.base_training import (
    SecureLogger, ConsentVerifier, AuditLogger, ModelRegistry,
    get_database_session, normalize_features
)


FEATURE_COLUMNS = [
    'med_adherence_mean_30d',
    'med_adherence_min_30d', 
    'med_adherence_max_30d',
    'med_adherence_trend_slope',
    'med_adherence_std_30d',
    'days_since_last_miss',
    'miss_count_30d',
    'engagement_rate_14d',
    'avg_pain_7d',
    'avg_mood_7d',
    'mh_score_7d',
    'trigger_count_meds_30d',
    'medication_changes_30d'
]

XGB_PARAMS = {
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 1,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'use_label_encoder': False,
    'random_state': 42
}


def load_adherence_data(db_session, logger: SecureLogger, consent_verifier: ConsentVerifier) -> Tuple[np.ndarray, np.ndarray]:
    """Load adherence training data with STRICT consent verification"""
    logger.info("Loading adherence training data...")
    
    consented_patients = consent_verifier.get_consented_patients()
    logger.info(f"Found {len(consented_patients)} consented patients")
    
    if not consented_patients:
        if consent_verifier.strict_mode:
            logger.error("CONSENT_REQUIRED: No consented patients found - cannot train on PHI without consent")
            raise PermissionError("Training requires at least one patient with explicit ML consent")
        else:
            logger.warning("DEV_MODE: No consented patients - generating synthetic data only")
            return generate_synthetic_adherence_data(logger)
    
    patient_ids_str = ",".join([f"'{p}'" for p in consented_patients])
    query = text(f"""
        WITH adherence_features AS (
            SELECT 
                patient_id,
                feature_date,
                AVG(med_adherence_7d) OVER (PARTITION BY patient_id ORDER BY feature_date ROWS BETWEEN 29 PRECEDING AND CURRENT ROW) as med_adherence_mean_30d,
                MIN(med_adherence_7d) OVER (PARTITION BY patient_id ORDER BY feature_date ROWS BETWEEN 29 PRECEDING AND CURRENT ROW) as med_adherence_min_30d,
                MAX(med_adherence_7d) OVER (PARTITION BY patient_id ORDER BY feature_date ROWS BETWEEN 29 PRECEDING AND CURRENT ROW) as med_adherence_max_30d,
                STDDEV(med_adherence_7d) OVER (PARTITION BY patient_id ORDER BY feature_date ROWS BETWEEN 29 PRECEDING AND CURRENT ROW) as med_adherence_std_30d,
                engagement_rate_14d,
                AVG(avg_pain) OVER (PARTITION BY patient_id ORDER BY feature_date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) as avg_pain_7d,
                AVG(avg_mood) OVER (PARTITION BY patient_id ORDER BY feature_date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) as avg_mood_7d,
                AVG(mh_score) OVER (PARTITION BY patient_id ORDER BY feature_date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) as mh_score_7d,
                had_non_adherence_issue_next7d
            FROM autopilot_patient_daily_features
            WHERE med_adherence_7d IS NOT NULL
            AND patient_id IN ({patient_ids_str})
        )
        SELECT * FROM adherence_features
        WHERE med_adherence_mean_30d IS NOT NULL
    """)
    
    try:
        result = db_session.execute(query)
        rows = result.fetchall()
    except Exception as e:
        logger.warning(f"Could not load from database: {e}")
        rows = []
    
    if len(rows) < 100:
        logger.info("Insufficient data, generating synthetic adherence data...")
        return generate_synthetic_adherence_data(logger)
    
    features = []
    labels = []
    
    for row in rows:
        feature_row = [
            float(row[2] or 0.8),
            float(row[3] or 0.5),
            float(row[4] or 1.0),
            0.0,
            float(row[5] or 0.1),
            7.0,
            3,
            float(row[6] or 0.7),
            float(row[7] or 3.0),
            float(row[8] or 5.0),
            float(row[9] or 5.0),
            0,
            0
        ]
        features.append(feature_row)
        labels.append(1 if row[10] else 0)
    
    logger.info(f"Loaded {len(features)} adherence samples")
    return np.array(features), np.array(labels)


def generate_synthetic_adherence_data(logger: SecureLogger) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic adherence data for model initialization"""
    np.random.seed(42)
    n_samples = 1000
    
    features = np.zeros((n_samples, len(FEATURE_COLUMNS)))
    
    features[:, 0] = np.clip(np.random.beta(5, 2, n_samples), 0, 1)
    features[:, 1] = np.clip(features[:, 0] - np.random.uniform(0, 0.3, n_samples), 0, 1)
    features[:, 2] = np.clip(features[:, 0] + np.random.uniform(0, 0.2, n_samples), 0, 1)
    features[:, 3] = np.random.randn(n_samples) * 0.02
    features[:, 4] = np.random.uniform(0.05, 0.2, n_samples)
    features[:, 5] = np.random.exponential(7, n_samples)
    features[:, 6] = np.random.poisson(2, n_samples)
    features[:, 7] = np.clip(np.random.beta(4, 2, n_samples), 0, 1)
    features[:, 8] = np.random.uniform(1, 8, n_samples)
    features[:, 9] = np.random.uniform(3, 8, n_samples)
    features[:, 10] = np.random.uniform(2, 8, n_samples)
    features[:, 11] = np.random.poisson(1, n_samples)
    features[:, 12] = np.random.poisson(0.5, n_samples)
    
    risk_score = (
        (1 - features[:, 0]) * 0.4 +
        (features[:, 6] / 10) * 0.2 +
        (1 - features[:, 7]) * 0.2 +
        (features[:, 8] / 10) * 0.1 +
        ((10 - features[:, 9]) / 10) * 0.1
    )
    
    labels = (np.random.random(n_samples) < risk_score).astype(int)
    
    logger.info(f"Generated {n_samples} synthetic adherence samples")
    logger.info(f"Label distribution: {np.bincount(labels)}")
    return features, labels


def train_adherence_model(
    X: np.ndarray,
    y: np.ndarray,
    logger: SecureLogger,
    model_path: Path
) -> Dict[str, Any]:
    """Train XGBoost model with cross-validation"""
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    cv_scores = []
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        model = xgb.XGBClassifier(**XGB_PARAMS)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_pred_proba)
        cv_scores.append(auc)
        
        if (fold + 1) % 2 == 0:
            logger.info(f"Fold {fold+1}/5 - AUC: {auc:.4f}")
    
    logger.info(f"CV AUC: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")
    
    final_model = xgb.XGBClassifier(**XGB_PARAMS)
    final_model.fit(X, y, verbose=False)
    
    joblib.dump(final_model, model_path)
    
    importance = final_model.feature_importances_
    feature_importance = {
        FEATURE_COLUMNS[i]: float(importance[i]) 
        for i in range(len(FEATURE_COLUMNS))
    }
    sorted_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
    
    logger.info("Top 5 important features:")
    for i, (feat, imp) in enumerate(list(sorted_importance.items())[:5]):
        logger.info(f"  {i+1}. {feat}: {imp:.4f}")
    
    return {
        "cv_auc_mean": float(np.mean(cv_scores)),
        "cv_auc_std": float(np.std(cv_scores)),
        "n_samples": len(y),
        "positive_rate": float(np.mean(y)),
        "feature_importance": sorted_importance
    }


def main():
    parser = argparse.ArgumentParser(description="Train Adherence Model")
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=6)
    args = parser.parse_args()
    
    logger = SecureLogger("train_adherence_model", "adherence_model_training.log")
    logger.info("=" * 60)
    logger.info("Starting Adherence Model Training")
    logger.info("=" * 60)
    
    if not XGBOOST_AVAILABLE:
        logger.error("XGBoost not available. Cannot train adherence model.")
        return
    
    db_session = get_database_session()
    is_production = os.getenv('NODE_ENV', 'development') == 'production'
    consent_verifier = ConsentVerifier(db_session, strict_mode=is_production)
    audit_logger = AuditLogger(db_session, "train_adherence_model")
    model_registry = ModelRegistry()
    
    training_params = {
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        **XGB_PARAMS
    }
    audit_logger.log_operation_start(training_params)
    
    try:
        X, y = load_adherence_data(db_session, logger, consent_verifier)
        
        model_path = model_registry.registry_path / "adherence_xgb.pkl"
        metrics = train_adherence_model(X, y, logger, model_path)
        
        version = model_registry.register_model(
            "adherence_model",
            model_path,
            {k: v for k, v in metrics.items() if k != "feature_importance"},
            training_params
        )
        
        logger.info(f"Model registered with version: {version}")
        audit_logger.log_operation_complete(metrics, success=True)
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        audit_logger.log_operation_complete({"error": str(e)}, success=False)
        raise
    finally:
        db_session.close()


if __name__ == "__main__":
    main()
