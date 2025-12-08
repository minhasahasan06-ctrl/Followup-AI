"""
Engagement Model Training Script

Train XGBoost model for optimal notification timing:
- Predict probability of task completion by time of day
- Optimize follow-up scheduling for patient engagement
- Feature engineering from historical completion patterns

HIPAA-compliant with consent verification and audit logging.
"""

import os
import sys
import json
import argparse
from datetime import datetime, timezone
from typing import Dict, Any, Tuple
from pathlib import Path
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from sqlalchemy import text

try:
    import xgboost as xgb
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score, accuracy_score
    import joblib
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from python_backend.ml_analysis.followup_autopilot.scripts.base_training import (
    SecureLogger, ConsentVerifier, AuditLogger, ModelRegistry,
    get_database_session
)


FEATURE_COLUMNS = [
    'hour_of_day',
    'day_of_week',
    'is_weekend',
    'engagement_rate_14d',
    'avg_pain_7d',
    'avg_mood_7d',
    'mh_score_7d',
    'risk_score',
    'pending_tasks_count',
    'hours_since_last_checkin',
    'avg_response_time_7d',
    'task_type_encoded'
]

XGB_PARAMS = {
    'n_estimators': 80,
    'max_depth': 5,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 2,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'use_label_encoder': False,
    'random_state': 42
}


def load_engagement_data(db_session, logger: SecureLogger, consent_verifier: ConsentVerifier) -> Tuple[np.ndarray, np.ndarray]:
    """Load engagement training data with STRICT consent verification"""
    logger.info("Loading engagement training data...")
    
    consented_patients = consent_verifier.get_consented_patients()
    logger.info(f"Found {len(consented_patients)} consented patients")
    
    if not consented_patients:
        if consent_verifier.strict_mode:
            logger.error("CONSENT_REQUIRED: No consented patients found - cannot train on PHI without consent")
            raise PermissionError("Training requires at least one patient with explicit ML consent")
        else:
            logger.warning("DEV_MODE: No consented patients - generating synthetic data only")
            return generate_synthetic_engagement_data(logger)
    
    patient_ids_str = ",".join([f"'{p}'" for p in consented_patients])
    query = text(f"""
        SELECT 
            t.patient_id,
            EXTRACT(HOUR FROM t.due_at) as hour_of_day,
            EXTRACT(DOW FROM t.due_at) as day_of_week,
            CASE WHEN EXTRACT(DOW FROM t.due_at) IN (0, 6) THEN 1 ELSE 0 END as is_weekend,
            COALESCE(f.engagement_rate_14d, 0.5) as engagement_rate_14d,
            COALESCE(f.avg_pain, 3) as avg_pain_7d,
            COALESCE(f.avg_mood, 5) as avg_mood_7d,
            COALESCE(f.mh_score, 5) as mh_score_7d,
            COALESCE(s.risk_score, 5) as risk_score,
            0 as pending_tasks_count,
            24 as hours_since_last_checkin,
            12 as avg_response_time_7d,
            CASE t.task_type
                WHEN 'symptom_check' THEN 0
                WHEN 'med_adherence_check' THEN 1
                WHEN 'mh_check' THEN 2
                WHEN 'video_exam' THEN 3
                WHEN 'audio_check' THEN 4
                ELSE 5
            END as task_type_encoded,
            CASE WHEN t.status = 'completed' AND t.completed_at IS NOT NULL 
                 AND t.completed_at <= t.due_at + INTERVAL '6 hours' 
                 THEN 1 ELSE 0 END as completed_within_6h
        FROM autopilot_followup_tasks t
        LEFT JOIN autopilot_patient_daily_features f 
            ON t.patient_id = f.patient_id AND DATE(t.due_at) = f.feature_date
        LEFT JOIN autopilot_patient_states s 
            ON t.patient_id = s.patient_id
        WHERE t.due_at IS NOT NULL
        AND t.patient_id IN ({patient_ids_str})
    """)
    
    try:
        result = db_session.execute(query)
        rows = result.fetchall()
    except Exception as e:
        logger.warning(f"Could not load from database: {e}")
        rows = []
    
    if len(rows) < 100:
        logger.info("Insufficient data, generating synthetic engagement data...")
        return generate_synthetic_engagement_data(logger)
    
    features = []
    labels = []
    
    for row in rows:
        feature_row = [float(x or 0) for x in row[1:13]]
        features.append(feature_row)
        labels.append(int(row[13] or 0))
    
    logger.info(f"Loaded {len(features)} engagement samples")
    return np.array(features), np.array(labels)


def generate_synthetic_engagement_data(logger: SecureLogger) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic engagement data for model initialization"""
    np.random.seed(42)
    n_samples = 2000
    
    features = np.zeros((n_samples, len(FEATURE_COLUMNS)))
    
    features[:, 0] = np.random.randint(0, 24, n_samples)
    features[:, 1] = np.random.randint(0, 7, n_samples)
    features[:, 2] = (features[:, 1] >= 5).astype(float)
    features[:, 3] = np.clip(np.random.beta(5, 2, n_samples), 0, 1)
    features[:, 4] = np.random.uniform(1, 8, n_samples)
    features[:, 5] = np.random.uniform(3, 8, n_samples)
    features[:, 6] = np.random.uniform(2, 8, n_samples)
    features[:, 7] = np.random.uniform(1, 15, n_samples)
    features[:, 8] = np.random.poisson(2, n_samples)
    features[:, 9] = np.random.exponential(12, n_samples)
    features[:, 10] = np.random.exponential(6, n_samples)
    features[:, 11] = np.random.randint(0, 6, n_samples)
    
    hour = features[:, 0]
    optimal_hours = ((hour >= 9) & (hour <= 11)) | ((hour >= 17) & (hour <= 19))
    engagement = features[:, 3]
    mood = features[:, 5] / 10
    
    base_prob = 0.3
    prob = base_prob + optimal_hours * 0.25 + engagement * 0.2 + mood * 0.15
    prob = np.clip(prob, 0.1, 0.9)
    
    labels = (np.random.random(n_samples) < prob).astype(int)
    
    logger.info(f"Generated {n_samples} synthetic engagement samples")
    logger.info(f"Label distribution: {np.bincount(labels)}")
    return features, labels


def train_engagement_model(
    X: np.ndarray,
    y: np.ndarray,
    logger: SecureLogger,
    model_path: Path
) -> Dict[str, Any]:
    """Train XGBoost engagement model with cross-validation"""
    
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
    
    logger.info("Top 5 important features for engagement:")
    for i, (feat, imp) in enumerate(list(sorted_importance.items())[:5]):
        logger.info(f"  {i+1}. {feat}: {imp:.4f}")
    
    hours = np.arange(24)
    test_features = np.zeros((24, len(FEATURE_COLUMNS)))
    test_features[:, 0] = hours
    test_features[:, 1] = 2
    test_features[:, 3] = 0.7
    test_features[:, 5] = 6.0
    
    hour_probs = final_model.predict_proba(test_features)[:, 1]
    best_hours = np.argsort(hour_probs)[-3:][::-1]
    logger.info(f"Best hours for engagement: {best_hours.tolist()}")
    
    return {
        "cv_auc_mean": float(np.mean(cv_scores)),
        "cv_auc_std": float(np.std(cv_scores)),
        "n_samples": len(y),
        "completion_rate": float(np.mean(y)),
        "best_hours": best_hours.tolist(),
        "feature_importance": sorted_importance
    }


def main():
    parser = argparse.ArgumentParser(description="Train Engagement Model")
    args = parser.parse_args()
    
    logger = SecureLogger("train_engagement_model", "engagement_model_training.log")
    logger.info("=" * 60)
    logger.info("Starting Engagement Model Training")
    logger.info("=" * 60)
    
    if not XGBOOST_AVAILABLE:
        logger.error("XGBoost not available. Cannot train engagement model.")
        return
    
    db_session = get_database_session()
    is_production = os.getenv('NODE_ENV', 'development') == 'production'
    consent_verifier = ConsentVerifier(db_session, strict_mode=is_production)
    audit_logger = AuditLogger(db_session, "train_engagement_model")
    model_registry = ModelRegistry()
    
    audit_logger.log_operation_start(XGB_PARAMS)
    
    try:
        X, y = load_engagement_data(db_session, logger, consent_verifier)
        
        model_path = model_registry.registry_path / "engagement_xgb.pkl"
        metrics = train_engagement_model(X, y, logger, model_path)
        
        version = model_registry.register_model(
            "engagement_model",
            model_path,
            {k: v for k, v in metrics.items() if k != "feature_importance"},
            XGB_PARAMS
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
