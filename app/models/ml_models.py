"""
ML Inference Database Models
Tracks ML model versions, predictions, and audit logs for HIPAA compliance

Phase 13: Added MLModelArtifact and MLCalibrationParams for storing trained
model weights and calibration parameters in PostgreSQL (Neon).
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, JSON, Text, ForeignKey, Index, LargeBinary
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.database import Base


class MLModel(Base):
    """Track ML model versions and metadata"""
    __tablename__ = "ml_models"

    id = Column(String, primary_key=True, index=True)
    name = Column(String, nullable=False, index=True)  # e.g., "pain_detector", "deterioration_lstm"
    version = Column(String, nullable=False)  # e.g., "1.0.0"
    model_type = Column(String, nullable=False)  # e.g., "pytorch", "onnx", "sklearn"
    task_type = Column(String, nullable=False)  # e.g., "classification", "regression", "time_series"
    
    # Model metadata
    file_path = Column(String)  # Path to model file
    input_schema = Column(JSON)  # Expected input format
    output_schema = Column(JSON)  # Expected output format
    metrics = Column(JSON)  # Accuracy, precision, recall, etc.
    
    # Status and deployment
    is_active = Column(Boolean, default=True)
    is_deployed = Column(Boolean, default=False)
    deployed_at = Column(DateTime(timezone=True))
    
    # Audit fields
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    created_by = Column(String)
    
    # Relationships
    predictions = relationship("MLPrediction", back_populates="model")
    performance_logs = relationship("MLPerformanceLog", back_populates="model")

    __table_args__ = (
        Index('idx_ml_model_name_version', 'name', 'version'),
    )


class MLPrediction(Base):
    """Store all ML predictions for audit logging (HIPAA compliance)"""
    __tablename__ = "ml_predictions"

    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(String, ForeignKey("ml_models.id"), nullable=False, index=True)
    patient_id = Column(String, nullable=False, index=True)  # AWS Cognito user ID
    
    # Prediction data
    prediction_type = Column(String, nullable=False)  # e.g., "pain_detection", "deterioration_risk"
    input_data = Column(JSON, nullable=False)  # Input features (anonymized if needed)
    prediction_result = Column(JSON, nullable=False)  # Model output
    confidence_score = Column(Float)  # Model confidence (0-1)
    
    # Performance tracking
    inference_time_ms = Column(Float)  # Time taken for prediction
    cache_hit = Column(Boolean, default=False)  # Was result from cache?
    
    # Audit and compliance
    predicted_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    ip_address = Column(String)  # For security audit
    user_agent = Column(String)  # Browser/client info
    
    # Relationships
    model = relationship("MLModel", back_populates="predictions")

    __table_args__ = (
        Index('idx_ml_prediction_patient_type', 'patient_id', 'prediction_type'),
        Index('idx_ml_prediction_created', 'predicted_at'),
    )


class MLPerformanceLog(Base):
    """Track ML model performance metrics over time"""
    __tablename__ = "ml_performance_logs"

    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(String, ForeignKey("ml_models.id"), nullable=False, index=True)
    
    # Performance metrics
    metric_name = Column(String, nullable=False)  # e.g., "accuracy", "latency", "throughput"
    metric_value = Column(Float, nullable=False)
    metric_unit = Column(String)  # e.g., "ms", "%", "req/s"
    
    # Context
    sample_size = Column(Integer)  # Number of predictions in this measurement
    time_window_minutes = Column(Integer)  # Measurement window
    
    # Aggregation type
    aggregation_type = Column(String)  # e.g., "mean", "median", "p95", "p99"
    
    # Timestamps
    measured_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    window_start = Column(DateTime(timezone=True))
    window_end = Column(DateTime(timezone=True))
    
    # Relationships
    model = relationship("MLModel", back_populates="performance_logs")

    __table_args__ = (
        Index('idx_ml_perf_model_metric', 'model_id', 'metric_name'),
        Index('idx_ml_perf_measured', 'measured_at'),
    )


class MLBatchJob(Base):
    """Track batch prediction jobs for multiple patients"""
    __tablename__ = "ml_batch_jobs"

    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(String, ForeignKey("ml_models.id"), nullable=False, index=True)
    
    # Job metadata
    job_name = Column(String, nullable=False)
    job_type = Column(String, nullable=False)  # e.g., "bulk_prediction", "model_evaluation"
    status = Column(String, nullable=False, default="pending")  # pending, running, completed, failed
    
    # Batch details
    total_items = Column(Integer)
    processed_items = Column(Integer, default=0)
    failed_items = Column(Integer, default=0)
    
    # Timing
    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))
    estimated_completion = Column(DateTime(timezone=True))
    
    # Results
    results_summary = Column(JSON)  # Summary statistics
    error_log = Column(Text)  # Error messages if failed
    
    # Audit
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    created_by = Column(String)
    
    __table_args__ = (
        Index('idx_ml_batch_status', 'status'),
        Index('idx_ml_batch_created', 'created_at'),
    )


class MLModelArtifact(Base):
    """
    Store trained model weights and artifacts in PostgreSQL.
    
    Phase 13: Enables storing ONNX/PyTorch/sklearn model binaries directly
    in Neon PostgreSQL instead of file system or S3.
    """
    __tablename__ = "ml_model_artifacts"

    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(String, ForeignKey("ml_models.id"), nullable=False, index=True)
    
    artifact_type = Column(String, nullable=False)
    artifact_format = Column(String, nullable=False)
    artifact_data = Column(LargeBinary, nullable=False)
    artifact_size_bytes = Column(Integer, nullable=False)
    checksum_sha256 = Column(String, nullable=False)
    
    compression = Column(String, default="none")
    is_primary = Column(Boolean, default=False)
    
    training_data_hash = Column(String)
    training_samples = Column(Integer)
    training_duration_seconds = Column(Float)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    created_by = Column(String)
    
    __table_args__ = (
        Index('idx_artifact_model_type', 'model_id', 'artifact_type'),
        Index('idx_artifact_primary', 'model_id', 'is_primary'),
    )


class MLCalibrationParams(Base):
    """
    Store probability calibration parameters for ML models.
    
    Phase 13: Supports Platt scaling, temperature scaling, and isotonic regression
    parameters for converting raw model outputs to calibrated probabilities.
    """
    __tablename__ = "ml_calibration_params"

    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(String, ForeignKey("ml_models.id"), nullable=False, index=True)
    
    calibration_method = Column(String, nullable=False)
    
    platt_a = Column(Float)
    platt_b = Column(Float)
    
    temperature = Column(Float)
    
    isotonic_x = Column(JSON)
    isotonic_y = Column(JSON)
    
    ece_before = Column(Float)
    ece_after = Column(Float)
    brier_before = Column(Float)
    brier_after = Column(Float)
    reliability_diagram = Column(JSON)
    
    validation_samples = Column(Integer)
    
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    created_by = Column(String)
    
    __table_args__ = (
        Index('idx_calibration_model', 'model_id'),
        Index('idx_calibration_active', 'model_id', 'is_active'),
    )
