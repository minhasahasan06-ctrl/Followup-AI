"""
LSTM Model Training for Deterioration Prediction
Trains a custom LSTM model using baseline and deviation data
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os

# Import ML libraries with error handling
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
    print("❌ PyTorch not installed. Install with: pip install torch scikit-learn joblib")
    TORCH_AVAILABLE = False
    exit(1)


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
    
    # Save model
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_size': input_size,
        'hidden_size': hidden_size,
        'num_layers': num_layers,
        'scaler': scaler
    }, model_save_path)
    
    print(f"\n✅ Model saved to {model_save_path}")
    print(f"Final Test Accuracy: {test_accuracy:.4f}")
    
    return model, scaler, history


def main():
    """Main training pipeline"""
    print("🚀 Starting LSTM deterioration model training...")
    
    # Configuration
    DATABASE_URL = os.getenv("DATABASE_URL")
    if not DATABASE_URL:
        print("❌ DATABASE_URL not set in environment")
        return
    
    # Load data
    print("\n📊 Loading patient data...")
    df = load_patient_data(DATABASE_URL, lookback_days=90)
    print(f"Loaded {len(df)} measurements from {df['patient_id'].nunique()} patients")
    
    # Create sequences
    print("\n🔄 Creating time-series sequences...")
    sequences, labels = create_sequences(df, sequence_length=7, target_days_ahead=3)
    print(f"Created {len(sequences)} sequences")
    print(f"Deterioration cases: {labels.sum()} ({labels.mean()*100:.1f}%)")
    
    if len(sequences) < 100:
        print("⚠️  Warning: Small dataset. Need more patient data for robust training.")
        print("   Consider collecting more measurements or adjusting sequence parameters.")
    
    # Train model
    print("\n🎓 Training LSTM model...")
    model, scaler, history = train_model(
        sequences,
        labels,
        input_size=4,
        hidden_size=64,
        num_layers=2,
        num_epochs=100,
        batch_size=32
    )
    
    print("\n✅ Training complete!")
    print("\nTo use this model in the ML inference service:")
    print("1. Copy ml_models/deterioration_lstm.pt to your deployment")
    print("2. Update app/services/ml_inference.py to load this model")
    print("3. Create API endpoint for deterioration predictions")


if __name__ == "__main__":
    if not TORCH_AVAILABLE:
        print("PyTorch not available. Exiting.")
        exit(1)
    main()
