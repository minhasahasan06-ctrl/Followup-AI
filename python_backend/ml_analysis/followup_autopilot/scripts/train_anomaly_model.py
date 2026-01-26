"""
Anomaly Detection Model Training Script

Trains an IsolationForest model on patient daily features to detect
unusual/anomalous days that may indicate health changes.

Features used:
- avg_pain, avg_fatigue, avg_mood
- steps, resting_hr, sleep_hours
- env_risk_score, mh_score
- video_resp_risk, audio_emotion_score, pain_severity_score

Usage:
    python train_anomaly_model.py [--contamination 0.1] [--n_estimators 100]

HIPAA Compliance:
- All training uses anonymized/aggregated data only
- No PHI in model artifacts
- Audit logged

Wellness Positioning:
- Model output is for wellness monitoring ONLY
- NOT for medical diagnosis
"""

import os
import sys
import logging
import argparse
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not available. Install with: pip install scikit-learn")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from .base_training import BaseTrainer, TrainingConfig

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

FEATURE_COLUMNS = [
    'avg_pain',
    'avg_fatigue',
    'avg_mood',
    'steps',
    'resting_hr',
    'sleep_hours',
    'env_risk_score',
    'mh_score',
    'video_resp_risk',
    'audio_emotion_score',
    'pain_severity_score',
]


class AnomalyModelTrainer(BaseTrainer):
    """Trainer for IsolationForest anomaly detection model"""
    
    def __init__(
        self, 
        contamination: float = 0.1,
        n_estimators: int = 100,
        random_state: int = 42,
        db_url: Optional[str] = None
    ):
        config = TrainingConfig(
            model_name="anomaly_iforest",
            model_version="1.0.0",
            min_samples=50,
            batch_size=1000,
        )
        super().__init__(config, db_url)
        
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.scaler = None
        self.model = None
    
    def fetch_training_data(self) -> List[Dict[str, Any]]:
        """Fetch daily features for anomaly training"""
        if not self.db_url:
            logger.warning("No database URL, using synthetic data")
            return self._generate_synthetic_data()
        
        try:
            import psycopg2
            import psycopg2.extras
            
            conn = psycopg2.connect(self.db_url)
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            cur.execute("""
                SELECT 
                    avg_pain, avg_fatigue, avg_mood,
                    steps, resting_hr, sleep_hours,
                    env_risk_score, mh_score,
                    video_resp_risk, audio_emotion_score, pain_severity_score
                FROM autopilot_daily_features
                WHERE date >= NOW() - INTERVAL '180 days'
                ORDER BY date DESC
            """)
            
            rows = cur.fetchall()
            cur.close()
            conn.close()
            
            logger.info(f"Fetched {len(rows)} daily feature records for anomaly training")
            return [dict(row) for row in rows]
            
        except Exception as e:
            logger.error(f"Database fetch failed: {e}")
            return self._generate_synthetic_data()
    
    def _generate_synthetic_data(self, n_samples: int = 500) -> List[Dict[str, Any]]:
        """Generate synthetic training data for development/testing"""
        logger.info(f"Generating {n_samples} synthetic samples for anomaly training")
        
        np.random.seed(self.random_state)
        
        data = []
        for _ in range(n_samples):
            is_anomaly = np.random.random() < 0.1
            
            if is_anomaly:
                sample = {
                    'avg_pain': np.random.uniform(7, 10),
                    'avg_fatigue': np.random.uniform(7, 10),
                    'avg_mood': np.random.uniform(1, 4),
                    'steps': np.random.uniform(0, 1000),
                    'resting_hr': np.random.uniform(90, 120),
                    'sleep_hours': np.random.uniform(2, 4),
                    'env_risk_score': np.random.uniform(60, 100),
                    'mh_score': np.random.uniform(0.6, 1.0),
                    'video_resp_risk': np.random.uniform(0.5, 1.0),
                    'audio_emotion_score': np.random.uniform(0.6, 1.0),
                    'pain_severity_score': np.random.uniform(0.6, 1.0),
                }
            else:
                sample = {
                    'avg_pain': np.random.uniform(0, 4),
                    'avg_fatigue': np.random.uniform(0, 4),
                    'avg_mood': np.random.uniform(6, 10),
                    'steps': np.random.uniform(3000, 10000),
                    'resting_hr': np.random.uniform(55, 80),
                    'sleep_hours': np.random.uniform(6, 9),
                    'env_risk_score': np.random.uniform(0, 40),
                    'mh_score': np.random.uniform(0, 0.3),
                    'video_resp_risk': np.random.uniform(0, 0.3),
                    'audio_emotion_score': np.random.uniform(0, 0.3),
                    'pain_severity_score': np.random.uniform(0, 0.3),
                }
            
            data.append(sample)
        
        return data
    
    def preprocess_data(self, raw_data: List[Dict[str, Any]]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Convert raw data to feature matrix"""
        if not raw_data:
            raise ValueError("No data to preprocess")
        
        X = []
        for row in raw_data:
            features = []
            for col in FEATURE_COLUMNS:
                val = row.get(col, 0)
                features.append(float(val) if val is not None else 0.0)
            X.append(features)
        
        X = np.array(X, dtype=np.float32)
        
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        logger.info(f"Preprocessed {len(X)} samples with {len(FEATURE_COLUMNS)} features")
        
        return X_scaled, None
    
    def train_model(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """Train IsolationForest model"""
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required for anomaly model training")
        
        logger.info(f"Training IsolationForest with contamination={self.contamination}, n_estimators={self.n_estimators}")
        
        self.model = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1,
            warm_start=False,
        )
        
        self.model.fit(X)
        
        predictions = self.model.predict(X)
        scores = self.model.decision_function(X)
        
        n_anomalies = (predictions == -1).sum()
        anomaly_rate = n_anomalies / len(predictions)
        
        logger.info(f"Training complete: {n_anomalies}/{len(predictions)} anomalies detected ({anomaly_rate:.2%})")
        logger.info(f"Score range: [{scores.min():.4f}, {scores.max():.4f}], mean={scores.mean():.4f}")
        
        return {
            'n_samples': len(X),
            'n_anomalies': int(n_anomalies),
            'anomaly_rate': float(anomaly_rate),
            'score_min': float(scores.min()),
            'score_max': float(scores.max()),
            'score_mean': float(scores.mean()),
        }
    
    def save_model(self, path: Optional[str] = None) -> str:
        """Save trained model and scaler to disk"""
        if self.model is None:
            raise ValueError("No model to save - train first")
        
        model_path = path or str(MODELS_DIR / "anomaly_iforest.pkl")
        
        artifact = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': FEATURE_COLUMNS,
            'contamination': self.contamination,
            'n_estimators': self.n_estimators,
            'trained_at': datetime.utcnow().isoformat(),
            'version': self.config.model_version,
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(artifact, f)
        
        logger.info(f"Model saved to {model_path}")
        
        self.log_training_run({
            'model_path': model_path,
            'feature_columns': FEATURE_COLUMNS,
            'contamination': self.contamination,
            'n_estimators': self.n_estimators,
        })
        
        return model_path
    
    def run(self) -> Dict[str, Any]:
        """Full training pipeline"""
        logger.info("Starting anomaly model training pipeline")
        
        raw_data = self.fetch_training_data()
        
        if len(raw_data) < self.config.min_samples:
            logger.warning(f"Insufficient data ({len(raw_data)} < {self.config.min_samples}), using synthetic data")
            raw_data = self._generate_synthetic_data()
        
        X, _ = self.preprocess_data(raw_data)
        
        metrics = self.train_model(X)
        
        model_path = self.save_model()
        
        return {
            'status': 'success',
            'model_path': model_path,
            'metrics': metrics,
        }


def load_anomaly_model(path: Optional[str] = None):
    """Load trained anomaly model from disk"""
    model_path = path or str(MODELS_DIR / "anomaly_iforest.pkl")
    
    if not os.path.exists(model_path):
        logger.warning(f"Model not found at {model_path}")
        return None
    
    with open(model_path, 'rb') as f:
        artifact = pickle.load(f)
    
    logger.info(f"Loaded anomaly model from {model_path} (trained: {artifact.get('trained_at')})")
    return artifact


def score_anomaly(features: Dict[str, float], model_artifact: Optional[Dict] = None) -> float:
    """
    Score a single day's features for anomaly.
    
    Returns:
        float: Anomaly score between 0 (normal) and 1 (anomalous)
    """
    if model_artifact is None:
        model_artifact = load_anomaly_model()
    
    if model_artifact is None:
        return _fallback_anomaly_score(features)
    
    model = model_artifact['model']
    scaler = model_artifact['scaler']
    feature_columns = model_artifact['feature_columns']
    
    X = np.array([[features.get(col, 0) for col in feature_columns]], dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0)
    X_scaled = scaler.transform(X)
    
    raw_score = model.decision_function(X_scaled)[0]
    
    normalized_score = 1 / (1 + np.exp(raw_score * 5))
    
    return float(np.clip(normalized_score, 0, 1))


def _fallback_anomaly_score(features: Dict[str, float]) -> float:
    """Rule-based fallback when no trained model available"""
    score = 0.0
    
    if features.get('avg_pain', 0) > 7:
        score += 0.2
    if features.get('avg_fatigue', 0) > 7:
        score += 0.15
    if features.get('avg_mood', 10) < 4:
        score += 0.15
    if features.get('steps', 5000) < 1000:
        score += 0.1
    if features.get('resting_hr', 70) > 100:
        score += 0.15
    if features.get('sleep_hours', 7) < 4:
        score += 0.1
    if features.get('mh_score', 0) > 0.6:
        score += 0.15
    
    return min(score, 1.0)


def main():
    parser = argparse.ArgumentParser(description='Train IsolationForest anomaly model')
    parser.add_argument('--contamination', type=float, default=0.1, 
                       help='Expected proportion of anomalies (0-0.5)')
    parser.add_argument('--n-estimators', type=int, default=100,
                       help='Number of trees in the forest')
    parser.add_argument('--output', type=str, default=None,
                       help='Output model path')
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    db_url = os.environ.get('DATABASE_URL')
    
    trainer = AnomalyModelTrainer(
        contamination=args.contamination,
        n_estimators=args.n_estimators,
        db_url=db_url
    )
    
    result = trainer.run()
    
    print("\n" + "="*50)
    print("ANOMALY MODEL TRAINING COMPLETE")
    print("="*50)
    print(f"Model saved to: {result['model_path']}")
    print(f"Training samples: {result['metrics']['n_samples']}")
    print(f"Anomalies detected: {result['metrics']['n_anomalies']} ({result['metrics']['anomaly_rate']:.1%})")
    print("="*50)


if __name__ == '__main__':
    main()
