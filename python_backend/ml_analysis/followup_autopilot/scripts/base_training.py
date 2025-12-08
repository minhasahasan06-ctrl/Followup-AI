"""
Base Training Utilities

Shared utilities for all training scripts with HIPAA compliance:
- Secure data access with consent verification
- PHI-safe logging (patient IDs hashed via HMAC)
- Model registry integration
- Audit logging for all operations
"""

import os
import sys
import json
import hashlib
import hmac
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker


HMAC_SECRET = os.getenv('PHI_HMAC_SECRET', 'followup-ai-training-secret-key')
MODEL_REGISTRY_PATH = Path(__file__).parent.parent / 'models'
LOG_DIR = Path('/tmp/ml_training_logs')
LOG_DIR.mkdir(exist_ok=True)


class SecureLogger:
    """HIPAA-compliant logger that hashes patient IDs"""
    
    def __init__(self, name: str, log_file: Optional[str] = None):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        if log_file:
            file_handler = logging.FileHandler(LOG_DIR / log_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def hash_patient_id(self, patient_id: str) -> str:
        """Hash patient ID for PHI-safe logging"""
        return hmac.new(
            HMAC_SECRET.encode(),
            patient_id.encode(),
            hashlib.sha256
        ).hexdigest()[:12]
    
    def info(self, msg: str, patient_id: Optional[str] = None):
        if patient_id:
            msg = f"[patient:{self.hash_patient_id(patient_id)}] {msg}"
        self.logger.info(msg)
    
    def warning(self, msg: str, patient_id: Optional[str] = None):
        if patient_id:
            msg = f"[patient:{self.hash_patient_id(patient_id)}] {msg}"
        self.logger.warning(msg)
    
    def error(self, msg: str, patient_id: Optional[str] = None):
        if patient_id:
            msg = f"[patient:{self.hash_patient_id(patient_id)}] {msg}"
        self.logger.error(msg)


class ConsentVerifier:
    """Verify patient consent before accessing training data - STRICT MODE"""
    
    def __init__(self, db_session, strict_mode: bool = True):
        self.db = db_session
        self.strict_mode = strict_mode
        self.logger = logging.getLogger("ConsentVerifier")
        self._validated_db = False
        self._validate_db_session()
    
    def _validate_db_session(self):
        """Validate database session is usable"""
        if self.db is None:
            if self.strict_mode:
                raise ConnectionError("STRICT_MODE: Database session is None - cannot verify consent")
            else:
                self.logger.warning("DEV_MODE: Database session is None - will use synthetic data only")
        self._validated_db = self.db is not None
    
    def has_ml_training_consent(self, patient_id: str) -> bool:
        """Check if patient has consented to ML training data use - STRICT"""
        if not self._validated_db:
            return False
        
        try:
            result = self.db.execute(text("""
                SELECT consent_given FROM ml_training_consent 
                WHERE patient_id = :patient_id 
                AND consent_type = 'autopilot_training'
                AND consent_given = true
                AND (expires_at IS NULL OR expires_at > NOW())
            """), {"patient_id": patient_id})
            row = result.fetchone()
            has_consent = row is not None
            if not has_consent:
                self.logger.warning(f"CONSENT_DENIED: Patient consent not found or expired")
            return has_consent
        except Exception as e:
            self.logger.error(f"CONSENT_CHECK_FAILED: Database error - {str(e)}")
            return False
    
    def get_consented_patients(self) -> List[str]:
        """Get list of patients who have consented to ML training - STRICT"""
        if not self._validated_db:
            self.logger.warning("No valid database session - returning empty consent list")
            return []
        
        try:
            result = self.db.execute(text("""
                SELECT table_name FROM information_schema.tables 
                WHERE table_name = 'ml_training_consent' AND table_schema = 'public'
            """))
            consent_table_exists = result.fetchone() is not None
            
            if consent_table_exists:
                result = self.db.execute(text("""
                    SELECT DISTINCT patient_id FROM ml_training_consent
                    WHERE consent_type = 'autopilot_training'
                    AND consent_given = true
                    AND (expires_at IS NULL OR expires_at > NOW())
                """))
                consented = [row[0] for row in result.fetchall()]
                self.logger.info(f"Found {len(consented)} patients with explicit consent")
                return consented
            else:
                if self.strict_mode:
                    self.logger.error("STRICT_MODE: No consent table found - aborting to prevent HIPAA violation")
                    return []
                else:
                    self.logger.warning("DEV_MODE: No consent table - training scripts will use synthetic data only")
                    return []
        except Exception as e:
            self.logger.error(f"CONSENT_QUERY_FAILED: {str(e)}")
            return []


class AuditLogger:
    """Log training operations for HIPAA compliance"""
    
    def __init__(self, db_session, operation_name: str):
        self.db = db_session
        self.operation_name = operation_name
        self.start_time = datetime.now(timezone.utc)
    
    def log_operation_start(self, details: Dict[str, Any]):
        """Log the start of a training operation"""
        try:
            self.db.execute(text("""
                INSERT INTO autopilot_audit_log 
                (action, entity_type, entity_id, details, performed_by, performed_at)
                VALUES (:action, :entity_type, :entity_id, :details, :performed_by, :performed_at)
            """), {
                "action": f"{self.operation_name}_started",
                "entity_type": "ml_training",
                "entity_id": self.operation_name,
                "details": json.dumps(details),
                "performed_by": "system",
                "performed_at": self.start_time
            })
            self.db.commit()
        except Exception:
            pass
    
    def log_operation_complete(self, metrics: Dict[str, Any], success: bool = True):
        """Log the completion of a training operation"""
        try:
            duration = (datetime.now(timezone.utc) - self.start_time).total_seconds()
            self.db.execute(text("""
                INSERT INTO autopilot_audit_log 
                (action, entity_type, entity_id, details, performed_by, performed_at)
                VALUES (:action, :entity_type, :entity_id, :details, :performed_by, :performed_at)
            """), {
                "action": f"{self.operation_name}_{'completed' if success else 'failed'}",
                "entity_type": "ml_training",
                "entity_id": self.operation_name,
                "details": json.dumps({**metrics, "duration_seconds": duration}),
                "performed_by": "system",
                "performed_at": datetime.now(timezone.utc)
            })
            self.db.commit()
        except Exception:
            pass


class ModelRegistry:
    """Manage model versions and checksums"""
    
    def __init__(self):
        self.registry_path = MODEL_REGISTRY_PATH
        self.registry_path.mkdir(exist_ok=True)
        self.manifest_path = self.registry_path / 'manifest.json'
        self.manifest = self._load_manifest()
    
    def _load_manifest(self) -> Dict[str, Any]:
        if self.manifest_path.exists():
            with open(self.manifest_path, 'r') as f:
                return json.load(f)
        return {"models": {}, "version": "1.0.0"}
    
    def _save_manifest(self):
        with open(self.manifest_path, 'w') as f:
            json.dump(self.manifest, f, indent=2, default=str)
    
    def compute_checksum(self, file_path: Path) -> str:
        """Compute SHA256 checksum of model file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def register_model(
        self, 
        model_name: str, 
        file_path: Path,
        metrics: Dict[str, float],
        training_params: Dict[str, Any]
    ) -> str:
        """Register a trained model with version and checksum"""
        version = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        checksum = self.compute_checksum(file_path)
        
        self.manifest["models"][model_name] = {
            "version": version,
            "file_path": str(file_path),
            "checksum": checksum,
            "metrics": metrics,
            "training_params": training_params,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "active"
        }
        self._save_manifest()
        return version
    
    def verify_model(self, model_name: str) -> bool:
        """Verify model integrity using checksum"""
        if model_name not in self.manifest["models"]:
            return False
        
        model_info = self.manifest["models"][model_name]
        file_path = Path(model_info["file_path"])
        
        if not file_path.exists():
            return False
        
        current_checksum = self.compute_checksum(file_path)
        return current_checksum == model_info["checksum"]
    
    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get model metadata"""
        return self.manifest["models"].get(model_name)


def get_database_session():
    """Create database session for training"""
    database_url = os.getenv('DATABASE_URL')
    if not database_url:
        raise ValueError("DATABASE_URL environment variable not set")
    
    engine = create_engine(database_url)
    Session = sessionmaker(bind=engine)
    return Session()


def normalize_features(features: np.ndarray, epsilon: float = 1e-8) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z-score normalize features, return normalized data and parameters"""
    mean = np.nanmean(features, axis=0)
    std = np.nanstd(features, axis=0)
    std = np.where(std < epsilon, 1.0, std)
    normalized = (features - mean) / std
    normalized = np.nan_to_num(normalized, nan=0.0)
    return normalized, mean, std


def create_sequences(
    features: np.ndarray, 
    labels: np.ndarray,
    sequence_length: int = 30
) -> Tuple[np.ndarray, np.ndarray]:
    """Create sequences for LSTM training"""
    X, y = [], []
    for i in range(sequence_length, len(features)):
        X.append(features[i-sequence_length:i])
        y.append(labels[i])
    return np.array(X), np.array(y)
