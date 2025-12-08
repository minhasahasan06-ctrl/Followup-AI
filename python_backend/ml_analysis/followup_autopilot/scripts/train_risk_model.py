"""
Risk Model Training Script

Train PyTorch LSTM multi-task model for 7-day risk prediction:
- Clinical deterioration
- Mental health crisis
- Medication non-adherence

Features:
- Early stopping with patience
- Gradient clipping for stability
- Model checkpointing
- HIPAA-compliant logging
- Consent verification
- Model registry integration
"""

import os
import sys
import json
import argparse
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from sqlalchemy import text

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from python_backend.ml_analysis.followup_autopilot.scripts.base_training import (
    SecureLogger, ConsentVerifier, AuditLogger, ModelRegistry,
    get_database_session, normalize_features, create_sequences
)


SEQUENCE_LENGTH = 30
HIDDEN_DIM = 64
NUM_LAYERS = 2
LEARNING_RATE = 0.001
BATCH_SIZE = 32
MAX_EPOCHS = 100
PATIENCE = 10
GRADIENT_CLIP = 1.0

FEATURE_COLUMNS = [
    'avg_pain', 'avg_fatigue', 'avg_mood', 'checkins_count',
    'steps', 'resting_hr', 'sleep_hours', 'weight',
    'env_risk_score', 'pollen_index', 'aqi', 'temp_c',
    'med_adherence_7d', 'mh_score', 'video_resp_risk',
    'audio_emotion_score', 'pain_severity_score', 'engagement_rate_14d'
]

LABEL_COLUMNS = [
    'had_worsening_event_next7d',
    'had_mh_crisis_next7d', 
    'had_non_adherence_issue_next7d'
]


class LSTMRiskModel(nn.Module):
    """Multi-task LSTM for risk prediction"""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, num_outputs: int = 3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=0.2, bidirectional=True
        )
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_outputs),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        return self.fc(context)


def load_training_data(db_session, logger: SecureLogger, consent_verifier: ConsentVerifier) -> Tuple[np.ndarray, np.ndarray]:
    """Load and prepare training data with STRICT consent verification"""
    logger.info("Loading training data from database...")
    
    consented_patients = consent_verifier.get_consented_patients()
    logger.info(f"Found {len(consented_patients)} consented patients")
    
    if not consented_patients:
        if consent_verifier.strict_mode:
            logger.error("CONSENT_REQUIRED: No consented patients found - cannot train on PHI without consent")
            logger.error("Training aborted to prevent HIPAA violation")
            raise PermissionError("Training requires at least one patient with explicit ML consent")
        else:
            logger.warning("DEV_MODE: No consented patients - generating synthetic data only")
            return generate_synthetic_data(logger)
    else:
        patient_ids_str = ",".join([f"'{p}'" for p in consented_patients])
        query = text(f"""
            SELECT patient_id, feature_date,
                   avg_pain, avg_fatigue, avg_mood, checkins_count,
                   steps, resting_hr, sleep_hours, weight,
                   env_risk_score, pollen_index, aqi, temp_c,
                   med_adherence_7d, mh_score, video_resp_risk,
                   audio_emotion_score, pain_severity_score, engagement_rate_14d,
                   had_worsening_event_next7d, had_mh_crisis_next7d, had_non_adherence_issue_next7d
            FROM autopilot_patient_daily_features
            WHERE patient_id IN ({patient_ids_str})
            ORDER BY patient_id, feature_date
        """)
    
    result = db_session.execute(query)
    rows = result.fetchall()
    
    if len(rows) < SEQUENCE_LENGTH + 10:
        logger.warning(f"Insufficient data: {len(rows)} rows, need at least {SEQUENCE_LENGTH + 10}")
        return generate_synthetic_data(logger)
    
    features = []
    labels = []
    
    for row in rows:
        feature_row = [
            float(row[2] or 0), float(row[3] or 0), float(row[4] or 0), float(row[5] or 0),
            float(row[6] or 0), float(row[7] or 0), float(row[8] or 0), float(row[9] or 0),
            float(row[10] or 0), float(row[11] or 0), float(row[12] or 0), float(row[13] or 0),
            float(row[14] or 0), float(row[15] or 0), float(row[16] or 0),
            float(row[17] or 0), float(row[18] or 0), float(row[19] or 0)
        ]
        label_row = [
            1 if row[20] else 0,
            1 if row[21] else 0,
            1 if row[22] else 0
        ]
        features.append(feature_row)
        labels.append(label_row)
    
    logger.info(f"Loaded {len(features)} feature rows")
    return np.array(features), np.array(labels)


def generate_synthetic_data(logger: SecureLogger) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic training data for model initialization"""
    logger.info("Generating synthetic training data for model initialization...")
    
    np.random.seed(42)
    n_samples = 500
    n_features = len(FEATURE_COLUMNS)
    n_labels = len(LABEL_COLUMNS)
    
    features = np.random.randn(n_samples, n_features)
    features[:, 0:4] = np.clip(features[:, 0:4] * 2 + 5, 0, 10)
    features[:, 4] = np.clip(features[:, 4] * 3000 + 5000, 0, 20000)
    features[:, 5] = np.clip(features[:, 5] * 10 + 70, 50, 120)
    features[:, 6] = np.clip(features[:, 6] * 2 + 7, 0, 14)
    features[:, 12] = np.clip(features[:, 12] * 0.2 + 0.8, 0, 1)
    
    labels = np.zeros((n_samples, n_labels))
    high_risk_mask = features[:, 0] > 7
    labels[high_risk_mask, 0] = np.random.binomial(1, 0.3, high_risk_mask.sum())
    low_mood_mask = features[:, 2] < 3
    labels[low_mood_mask, 1] = np.random.binomial(1, 0.25, low_mood_mask.sum())
    low_adherence_mask = features[:, 12] < 0.6
    labels[low_adherence_mask, 2] = np.random.binomial(1, 0.4, low_adherence_mask.sum())
    
    logger.info(f"Generated {n_samples} synthetic samples")
    return features, labels


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    logger: SecureLogger,
    model_path: Path
) -> Dict[str, Any]:
    """Train model with early stopping and checkpointing"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    for epoch in range(MAX_EPOCHS):
        model.train()
        epoch_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                val_loss += criterion(outputs, batch_y).item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        scheduler.step(avg_val_loss)
        
        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch+1}/{MAX_EPOCHS} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_path)
            logger.info(f"Model checkpoint saved at epoch {epoch+1}")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
    
    model.load_state_dict(torch.load(model_path))
    
    return {
        "final_train_loss": train_losses[-1],
        "final_val_loss": val_losses[-1],
        "best_val_loss": best_val_loss,
        "epochs_trained": len(train_losses),
        "early_stopped": patience_counter >= PATIENCE
    }


def main():
    parser = argparse.ArgumentParser(description="Train Risk Model")
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    args = parser.parse_args()
    
    logger = SecureLogger("train_risk_model", "risk_model_training.log")
    logger.info("=" * 60)
    logger.info("Starting Risk Model Training")
    logger.info("=" * 60)
    
    if not TORCH_AVAILABLE:
        logger.error("PyTorch not available. Cannot train risk model.")
        return
    
    db_session = get_database_session()
    is_production = os.getenv('NODE_ENV', 'development') == 'production'
    consent_verifier = ConsentVerifier(db_session, strict_mode=is_production)
    audit_logger = AuditLogger(db_session, "train_risk_model")
    model_registry = ModelRegistry()
    
    training_params = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "sequence_length": SEQUENCE_LENGTH,
        "hidden_dim": HIDDEN_DIM,
        "num_layers": NUM_LAYERS
    }
    audit_logger.log_operation_start(training_params)
    
    try:
        features, labels = load_training_data(db_session, logger, consent_verifier)
        
        features_normalized, mean, std = normalize_features(features)
        
        X_seq, y_seq = create_sequences(features_normalized, labels, SEQUENCE_LENGTH)
        logger.info(f"Created {len(X_seq)} sequences for training")
        
        split_idx = int(len(X_seq) * 0.8)
        X_train, X_val = X_seq[:split_idx], X_seq[split_idx:]
        y_train, y_val = y_seq[:split_idx], y_seq[split_idx:]
        
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val),
            torch.FloatTensor(y_val)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
        
        input_dim = len(FEATURE_COLUMNS)
        model = LSTMRiskModel(input_dim, HIDDEN_DIM, NUM_LAYERS, len(LABEL_COLUMNS))
        
        model_path = model_registry.registry_path / "risk_model.pt"
        metrics = train_model(model, train_loader, val_loader, logger, model_path)
        
        normalization_path = model_registry.registry_path / "risk_model_norm.json"
        with open(normalization_path, 'w') as f:
            json.dump({
                "mean": mean.tolist(),
                "std": std.tolist(),
                "feature_columns": FEATURE_COLUMNS
            }, f)
        
        version = model_registry.register_model(
            "risk_model",
            model_path,
            metrics,
            training_params
        )
        
        logger.info(f"Model registered with version: {version}")
        logger.info(f"Training metrics: {json.dumps(metrics, indent=2)}")
        
        audit_logger.log_operation_complete(metrics, success=True)
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        audit_logger.log_operation_complete({"error": str(e)}, success=False)
        raise
    finally:
        db_session.close()


if __name__ == "__main__":
    main()
