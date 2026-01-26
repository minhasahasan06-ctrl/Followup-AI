"""
LSTM Model Training for Deterioration Prediction
=================================================

Phase 13: Trains LSTM deterioration model and saves to PostgreSQL (Neon).

Features:
- Train PyTorch LSTM on patient time-series data
- Save trained weights to PostgreSQL using ModelArtifactService
- Apply probability calibration (Platt scaling)
- Store calibration parameters for inference
- HIPAA audit logging of training operations
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os
import sys
import io
import uuid
import hashlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    import joblib
    TORCH_AVAILABLE = True
except ImportError:
    print("PyTorch not installed. Install with: pip install torch scikit-learn joblib")
    TORCH_AVAILABLE = False


class DeteriorationLSTM(nn.Module):
    """
    LSTM model for predicting patient deterioration
    
    Inputs: Time-series of health metrics (pain, respiratory rate, symptoms)
    Output: Binary deterioration risk (0 = stable, 1 = deteriorating)
    """
    def __init__(self, input_size=4, hidden_size=64, num_layers=2, dropout=0.3):
        super(DeteriorationLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use last hidden state
        last_hidden = h_n[-1]  # Shape: (batch_size, hidden_size)
        
        # Fully connected layers
        out = self.fc1(last_hidden)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        
        return out


class DeteriorationDataset(Dataset):
    """PyTorch dataset for deterioration prediction"""
    def __init__(self, sequences, labels):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.FloatTensor(labels)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


def load_patient_data(database_url: str, lookback_days: int = 14):
    """
    Load patient health data from database
    
    Returns:
        DataFrame with patient health measurements
    """
    engine = create_engine(database_url)
    
    # Query patient measurements (pain, respiratory rate, symptoms)
    query = """
    SELECT 
        patient_id,
        measurement_date,
        pain_score_facial,
        pain_score_self_reported,
        respiratory_rate,
        symptom_severity_score
    FROM patient_measurements
    WHERE measurement_date >= CURRENT_DATE - INTERVAL '{} days'
    ORDER BY patient_id, measurement_date
    """.format(lookback_days)
    
    df = pd.read_sql(query, engine)
    engine.dispose()
    
    return df


def create_sequences(df, sequence_length=7, target_days_ahead=3):
    """
    Create time-series sequences for LSTM training
    
    Args:
        df: DataFrame with patient measurements
        sequence_length: Number of days to look back (default 7)
        target_days_ahead: How many days ahead to predict (default 3)
    
    Returns:
        sequences: Input sequences (shape: [n_samples, sequence_length, n_features])
        labels: Binary deterioration labels (0 = stable, 1 = deteriorating)
    """
    sequences = []
    labels = []
    
    # Group by patient
    for patient_id, group in df.groupby('patient_id'):
        group = group.sort_values('measurement_date').reset_index(drop=True)
        
        # Need at least sequence_length + target_days_ahead days
        if len(group) < sequence_length + target_days_ahead:
            continue
        
        # Create sequences
        for i in range(len(group) - sequence_length - target_days_ahead):
            # Input sequence
            seq = group.iloc[i:i+sequence_length][
                ['pain_score_facial', 'pain_score_self_reported', 
                 'respiratory_rate', 'symptom_severity_score']
            ].values
            
            # Label: Did patient deteriorate in next N days?
            current_avg = seq[-3:].mean(axis=0).mean()  # Average of last 3 days
            future_avg = group.iloc[
                i+sequence_length:i+sequence_length+target_days_ahead
            ][['pain_score_facial', 'pain_score_self_reported', 
               'respiratory_rate', 'symptom_severity_score']].values.mean()
            
            # Deterioration = significant increase in metrics
            deterioration = 1 if future_avg > current_avg * 1.2 else 0
            
            sequences.append(seq)
            labels.append(deterioration)
    
    return np.array(sequences), np.array(labels)


def train_model(
    sequences,
    labels,
    input_size=4,
    hidden_size=64,
    num_layers=2,
    num_epochs=100,
    batch_size=32,
    learning_rate=0.001,
    model_save_path="./ml_models/deterioration_lstm.pt"
):
    """
    Train LSTM deterioration prediction model
    
    Args:
        sequences: Input sequences
        labels: Binary labels
        Other args: Model hyperparameters
    
    Returns:
        Trained model, scaler, training history
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        sequences, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Normalize features
    scaler = StandardScaler()
    
    # Reshape for scaling
    n_samples, seq_len, n_features = X_train.shape
    X_train_reshaped = X_train.reshape(-1, n_features)
    X_train_scaled = scaler.fit_transform(X_train_reshaped).reshape(n_samples, seq_len, n_features)
    
    X_test_reshaped = X_test.reshape(-1, n_features)
    X_test_scaled = scaler.transform(X_test_reshaped).reshape(X_test.shape)
    
    # Create datasets
    train_dataset = DeteriorationDataset(X_train_scaled, y_train)
    test_dataset = DeteriorationDataset(X_test_scaled, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Initialize model
    model = DeteriorationLSTM(input_size, hidden_size, num_layers)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    history = {'train_loss': [], 'test_loss': [], 'test_accuracy': []}
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        for sequences_batch, labels_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(sequences_batch)
            loss = criterion(outputs.squeeze(), labels_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for sequences_batch, labels_batch in test_loader:
                outputs = model(sequences_batch)
                loss = criterion(outputs.squeeze(), labels_batch)
                test_loss += loss.item()
                
                # Calculate accuracy
                predictions = (outputs.squeeze() > 0.5).float()
                correct += (predictions == labels_batch).sum().item()
                total += labels_batch.size(0)
        
        avg_test_loss = test_loss / len(test_loader)
        test_accuracy = correct / total
        
        history['train_loss'].append(avg_train_loss)
        history['test_loss'].append(avg_test_loss)
        history['test_accuracy'].append(test_accuracy)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], "
                  f"Train Loss: {avg_train_loss:.4f}, "
                  f"Test Loss: {avg_test_loss:.4f}, "
                  f"Test Accuracy: {test_accuracy:.4f}")
    
    print(f"Final Test Accuracy: {test_accuracy:.4f}")
    
    return model, scaler, history, (X_test_scaled, y_test)


def save_model_to_postgres(
    model: nn.Module,
    scaler: StandardScaler,
    model_id: str,
    training_samples: int,
    training_duration: float,
    metrics: dict,
    database_url: str
):
    """
    Save trained model to PostgreSQL using ModelArtifactService.
    
    Phase 13: Replaces file-based storage with database storage.
    """
    from app.database import get_db_session
    from app.models.ml_models import MLModel, MLModelArtifact
    from app.services.model_artifact_service import ModelArtifactService
    from app.services.audit_logger import HIPAAAuditLogger
    
    engine = create_engine(database_url)
    Session = sessionmaker(bind=engine)
    db = Session()
    
    try:
        existing = db.query(MLModel).filter(MLModel.id == model_id).first()
        if not existing:
            ml_model = MLModel(
                id=model_id,
                name="deterioration_lstm",
                version="1.0.0",
                model_type="pytorch",
                task_type="classification",
                input_schema={"type": "time_series", "features": 4, "sequence_length": 7},
                output_schema={"type": "probability", "range": [0, 1]},
                metrics=metrics,
                is_active=True,
                is_deployed=False,
                created_by="ml_training_pipeline"
            )
            db.add(ml_model)
            db.commit()
            print(f"Created MLModel record: {model_id}")
        else:
            existing.metrics = metrics
            existing.updated_at = datetime.utcnow()
            db.commit()
            print(f"Updated existing MLModel record: {model_id}")
        
        artifact_service = ModelArtifactService(db)
        
        artifact_service.save_pytorch_model(
            model_id=model_id,
            model=model,
            artifact_type="weights",
            compress=True,
            training_samples=training_samples,
            training_duration=training_duration,
            user_id="ml_training_pipeline"
        )
        print("Saved PyTorch model weights to PostgreSQL")
        
        artifact_service.save_sklearn_model(
            model_id=model_id,
            model=scaler,
            artifact_type="scaler",
            compress=True,
            training_samples=training_samples,
            user_id="ml_training_pipeline"
        )
        print("Saved StandardScaler to PostgreSQL")
        
        HIPAAAuditLogger.log_phi_access(
            actor_id="ml_training_pipeline",
            actor_role="system",
            patient_id="N/A",
            resource_type="ml_model",
            action="train",
            access_reason="LSTM deterioration model training completed",
            additional_context={
                "model_id": model_id,
                "training_samples": training_samples,
                "metrics": metrics
            }
        )
        
    finally:
        db.close()
        engine.dispose()


def calibrate_and_save(
    model: nn.Module,
    test_data: tuple,
    model_id: str,
    database_url: str
):
    """
    Calibrate model probabilities and save to PostgreSQL.
    
    Phase 13: Applies Platt scaling for well-calibrated probabilities.
    """
    from app.database import get_db_session
    from app.services.deterioration_calibration_service import CalibrationService
    
    X_test, y_test = test_data
    
    model.eval()
    with torch.no_grad():
        test_tensor = torch.FloatTensor(X_test)
        raw_probs = model(test_tensor).squeeze().numpy()
    
    logits = np.log(np.clip(raw_probs, 1e-10, 1-1e-10) / 
                   np.clip(1-raw_probs, 1e-10, 1-1e-10))
    
    engine = create_engine(database_url)
    Session = sessionmaker(bind=engine)
    db = Session()
    
    try:
        calibration_service = CalibrationService(db)
        
        result = calibration_service.calibrate_and_save(
            model_id=model_id,
            y_true=y_test,
            logits_or_probs=logits,
            method="platt",
            is_logits=True,
            user_id="ml_training_pipeline"
        )
        
        print(f"Calibration complete:")
        print(f"  ECE before: {result['metrics']['ece_before']:.4f}")
        print(f"  ECE after:  {result['metrics']['ece_after']:.4f}")
        print(f"  Improvement: {result['metrics']['improvement_percent']:.1f}%")
        
        return result
        
    finally:
        db.close()
        engine.dispose()


def generate_synthetic_data(n_patients: int = 100, days_per_patient: int = 30):
    """
    Generate synthetic patient data for training when real data is unavailable.
    
    This ensures the model can be trained and tested even without patient data.
    """
    np.random.seed(42)
    
    data = []
    for patient_id in range(n_patients):
        base_pain = np.random.uniform(0.1, 0.4)
        base_resp = np.random.uniform(14, 18)
        base_symptom = np.random.uniform(0.1, 0.3)
        
        deteriorates = np.random.random() < 0.2
        deterioration_day = np.random.randint(20, days_per_patient) if deteriorates else None
        
        for day in range(days_per_patient):
            if deteriorates and day >= deterioration_day:
                progression = min(1.0, (day - deterioration_day + 1) * 0.15)
                pain_facial = min(1.0, base_pain + progression + np.random.normal(0, 0.05))
                pain_self = min(1.0, base_pain + progression * 0.8 + np.random.normal(0, 0.05))
                resp_rate = base_resp + progression * 8 + np.random.normal(0, 1)
                symptom_score = min(1.0, base_symptom + progression + np.random.normal(0, 0.05))
            else:
                pain_facial = max(0, base_pain + np.random.normal(0, 0.05))
                pain_self = max(0, base_pain * 0.9 + np.random.normal(0, 0.05))
                resp_rate = base_resp + np.random.normal(0, 1)
                symptom_score = max(0, base_symptom + np.random.normal(0, 0.03))
            
            data.append({
                'patient_id': f"patient_{patient_id}",
                'measurement_date': datetime.now() - timedelta(days=days_per_patient - day),
                'pain_score_facial': max(0, min(1, pain_facial)),
                'pain_score_self_reported': max(0, min(1, pain_self)),
                'respiratory_rate': max(8, min(30, resp_rate)),
                'symptom_severity_score': max(0, min(1, symptom_score))
            })
    
    return pd.DataFrame(data)


def main():
    """Main training pipeline with PostgreSQL storage"""
    print("Starting LSTM deterioration model training...")
    print("Phase 13: Saving to PostgreSQL\n")
    
    DATABASE_URL = os.getenv("DATABASE_URL")
    if not DATABASE_URL:
        print("DATABASE_URL not set in environment")
        return
    
    start_time = datetime.now()
    model_id = f"deterioration_lstm_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    print("Loading patient data...")
    try:
        df = load_patient_data(DATABASE_URL, lookback_days=90)
        print(f"Loaded {len(df)} measurements from {df['patient_id'].nunique()} patients")
    except Exception as e:
        print(f"Could not load real data: {e}")
        print("Generating synthetic data for training...")
        df = generate_synthetic_data(n_patients=100, days_per_patient=30)
        print(f"Generated {len(df)} synthetic measurements")
    
    print("\nCreating time-series sequences...")
    sequences, labels = create_sequences(df, sequence_length=7, target_days_ahead=3)
    print(f"Created {len(sequences)} sequences")
    print(f"Deterioration cases: {int(labels.sum())} ({labels.mean()*100:.1f}%)")
    
    if len(sequences) < 50:
        print("\nInsufficient data. Generating more synthetic data...")
        df = generate_synthetic_data(n_patients=200, days_per_patient=45)
        sequences, labels = create_sequences(df, sequence_length=7, target_days_ahead=3)
        print(f"Created {len(sequences)} sequences from synthetic data")
    
    print("\nTraining LSTM model...")
    model, scaler, history, test_data = train_model(
        sequences,
        labels,
        input_size=4,
        hidden_size=64,
        num_layers=2,
        num_epochs=50,
        batch_size=32
    )
    
    training_duration = (datetime.now() - start_time).total_seconds()
    
    metrics = {
        "final_train_loss": history['train_loss'][-1],
        "final_test_loss": history['test_loss'][-1],
        "final_test_accuracy": history['test_accuracy'][-1],
        "training_duration_seconds": training_duration,
        "training_samples": len(sequences),
        "trained_at": datetime.now().isoformat()
    }
    
    print(f"\nSaving model to PostgreSQL...")
    save_model_to_postgres(
        model=model,
        scaler=scaler,
        model_id=model_id,
        training_samples=len(sequences),
        training_duration=training_duration,
        metrics=metrics,
        database_url=DATABASE_URL
    )
    
    print(f"\nCalibrating model probabilities...")
    calibrate_and_save(
        model=model,
        test_data=test_data,
        model_id=model_id,
        database_url=DATABASE_URL
    )
    
    print(f"\nTraining complete!")
    print(f"Model ID: {model_id}")
    print(f"Final Accuracy: {metrics['final_test_accuracy']:.4f}")
    print(f"Training Duration: {training_duration:.1f} seconds")
    print("\nModel is now available for inference via the prediction API.")


if __name__ == "__main__":
    if not TORCH_AVAILABLE:
        print("PyTorch not available. Exiting.")
        sys.exit(1)
    main()
