"""
Tremor Analysis Service - Accelerometer-based tremor detection
Analyzes phone accelerometer data for micro-shake detection
Uses FFT for frequency analysis and tremor classification
"""

import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq
from typing import Dict, Any, List
from datetime import datetime
import logging

from sqlalchemy.orm import Session
from app.models.video_ai_models import AccelerometerTremorData

logger = logging.getLogger(__name__)


class TremorAnalysisService:
    """
    Production-grade tremor analysis from phone accelerometer data
    Detects tremor frequency, amplitude, and classifies tremor types
    """
    
    # Tremor frequency ranges (Hz)
    PARKINSONIAN_TREMOR = (4, 6)  # 4-6 Hz resting tremor
    ESSENTIAL_TREMOR = (6, 12)  # 6-12 Hz action/postural tremor
    PHYSIOLOGICAL_TREMOR = (8, 12)  # 8-12 Hz normal tremor
    
    # Thresholds
    TREMOR_DETECTION_THRESHOLD_MG = 50  # millig (50mg = 0.5 m/s²)
    TREMOR_INDEX_SCALE_FACTOR = 10  # Scale amplitude to 0-100
    
    def __init__(self, db: Session):
        self.db = db
    
    def analyze_tremor(
        self,
        patient_id: str,
        timestamps: List[float],
        accel_x: List[float],
        accel_y: List[float],
        accel_z: List[float],
        device_info: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Analyze accelerometer data for tremor detection
        
        Args:
            patient_id: Patient ID
            timestamps: Timestamps in milliseconds
            accel_x/y/z: Acceleration in m/s² for each axis
            device_info: Device metadata
            
        Returns:
            Dict with tremor metrics and analysis
        """
        logger.info(f"Starting tremor analysis for patient {patient_id}, {len(timestamps)} samples")
        
        # Convert to numpy arrays
        timestamps_arr = np.array(timestamps)
        x = np.array(accel_x)
        y = np.array(accel_y)
        z = np.array(accel_z)
        
        # Calculate sampling rate
        time_diffs = np.diff(timestamps_arr)
        avg_sample_interval_ms = np.mean(time_diffs)
        sampling_rate = 1000.0 / avg_sample_interval_ms  # Convert to Hz
        duration = (timestamps_arr[-1] - timestamps_arr[0]) / 1000.0  # seconds
        
        logger.info(f"Sampling rate: {sampling_rate:.1f} Hz, Duration: {duration:.1f}s")
        
        # Calculate magnitude (resultant acceleration)
        magnitude = np.sqrt(x**2 + y**2 + z**2)
        
        # Remove gravity (DC component) using high-pass filter
        # Tremor is AC component, gravity is DC
        sos = signal.butter(4, 0.5, 'hp', fs=sampling_rate, output='sos')
        magnitude_filtered = signal.sosfilt(sos, magnitude)
        
        # Perform FFT
        N = len(magnitude_filtered)
        yf = fft(magnitude_filtered)
        xf = fftfreq(N, 1 / sampling_rate)[:N//2]
        power = 2.0/N * np.abs(yf[0:N//2])
        
        # Find dominant frequency (peak in tremor range 3-15 Hz)
        tremor_range_mask = (xf >= 3) & (xf <= 15)
        tremor_range_power = power[tremor_range_mask]
        tremor_range_freqs = xf[tremor_range_mask]
        
        if len(tremor_range_power) == 0:
            logger.warning("No data in tremor frequency range (3-15 Hz)")
            dominant_frequency = 0
            peak_amplitude = 0
        else:
            peak_idx = np.argmax(tremor_range_power)
            dominant_frequency = tremor_range_freqs[peak_idx]
            peak_amplitude = tremor_range_power[peak_idx]
        
        # Convert amplitude to millig (1g = 9.8 m/s²)
        peak_amplitude_mg = (peak_amplitude / 9.8) * 1000
        
        # Calculate frequency band powers
        low_freq_mask = (xf >= 0) & (xf < 4)
        tremor_freq_mask = (xf >= 4) & (xf < 12)
        high_freq_mask = (xf >= 12) & (xf < sampling_rate / 2)
        
        low_freq_power = np.sum(power[low_freq_mask])
        tremor_freq_power = np.sum(power[tremor_freq_mask])
        high_freq_power = np.sum(power[high_freq_mask])
        
        # Detect tremor
        tremor_detected = peak_amplitude_mg > self.TREMOR_DETECTION_THRESHOLD_MG
        
        # Calculate tremor index (0-100 score)
        tremor_index = min(100, peak_amplitude_mg * self.TREMOR_INDEX_SCALE_FACTOR)
        
        # Classify tremor type
        parkinsonian_likelihood = 0.0
        essential_tremor_likelihood = 0.0
        physiological_tremor = False
        
        if tremor_detected:
            # Parkinsonian (4-6 Hz)
            if self.PARKINSONIAN_TREMOR[0] <= dominant_frequency <= self.PARKINSONIAN_TREMOR[1]:
                parkinsonian_likelihood = min(100, (peak_amplitude_mg / 200) * 100)
            
            # Essential tremor (6-12 Hz)
            if self.ESSENTIAL_TREMOR[0] <= dominant_frequency <= self.ESSENTIAL_TREMOR[1]:
                essential_tremor_likelihood = min(100, (peak_amplitude_mg / 150) * 100)
            
            # Physiological tremor (8-12 Hz, low amplitude)
            if self.PHYSIOLOGICAL_TREMOR[0] <= dominant_frequency <= self.PHYSIOLOGICAL_TREMOR[1]:
                if peak_amplitude_mg < 100:
                    physiological_tremor = True
        
        # Create database record
        tremor_data = AccelerometerTremorData(
            patient_id=patient_id,
            timestamps=timestamps,
            accel_x=accel_x,
            accel_y=accel_y,
            accel_z=accel_z,
            sampling_rate_hz=float(sampling_rate),
            duration_seconds=float(duration),
            sample_count=len(timestamps),
            tremor_index=float(tremor_index),
            dominant_frequency_hz=float(dominant_frequency),
            tremor_amplitude_mg=float(peak_amplitude_mg),
            tremor_detected=bool(tremor_detected),
            low_freq_power=float(low_freq_power),
            tremor_freq_power=float(tremor_freq_power),
            high_freq_power=float(high_freq_power),
            parkinsonian_tremor_likelihood=float(parkinsonian_likelihood),
            essential_tremor_likelihood=float(essential_tremor_likelihood),
            physiological_tremor=bool(physiological_tremor),
            device_type=device_info.get('type', 'unknown'),
            device_model=device_info.get('model', 'unknown'),
            browser_info=device_info.get('browser', 'unknown'),
            analyzed_at=datetime.utcnow(),
        )
        
        self.db.add(tremor_data)
        self.db.commit()
        self.db.refresh(tremor_data)
        
        logger.info(f"Tremor analysis complete. ID: {tremor_data.id}, Index: {tremor_index:.1f}, Tremor: {tremor_detected}")
        
        return {
            'tremor_data_id': tremor_data.id,
            'tremor_detected': tremor_detected,
            'tremor_index': tremor_index,
            'dominant_frequency_hz': dominant_frequency,
            'peak_amplitude_mg': peak_amplitude_mg,
            'parkinsonian_likelihood': parkinsonian_likelihood,
            'essential_tremor_likelihood': essential_tremor_likelihood,
            'physiological_tremor': physiological_tremor,
            'sampling_rate_hz': sampling_rate,
            'duration_seconds': duration,
        }
    
    def get_latest_tremor(self, patient_id: str) -> Dict[str, Any] | None:
        """Get patient's most recent tremor analysis"""
        tremor_data = self.db.query(AccelerometerTremorData).filter(
            AccelerometerTremorData.patient_id == patient_id
        ).order_by(AccelerometerTremorData.created_at.desc()).first()
        
        if not tremor_data:
            return None
        
        return {
            'tremor_index': tremor_data.tremor_index,
            'tremor_detected': tremor_data.tremor_detected,
            'dominant_frequency_hz': tremor_data.dominant_frequency_hz,
            'parkinsonian_likelihood': tremor_data.parkinsonian_tremor_likelihood,
            'essential_tremor_likelihood': tremor_data.essential_tremor_likelihood,
            'physiological_tremor': tremor_data.physiological_tremor,
            'created_at': tremor_data.created_at.isoformat() if tremor_data.created_at else None,
        }
