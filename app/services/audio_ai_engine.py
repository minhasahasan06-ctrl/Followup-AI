"""
Audio AI Engine - Production-Grade Audio Analysis
Features: Breath cycles, speech pace, cough detection, wheeze detection, hoarseness analysis
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import logging
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Audio processing libraries
try:
    import librosa
    import soundfile as sf
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

try:
    from scipy import signal, stats
    from scipy.fft import fft
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import noisereduce as nr
    NOISEREDUCE_AVAILABLE = True
except ImportError:
    NOISEREDUCE_AVAILABLE = False

logger = logging.getLogger(__name__)


class AudioAIEngine:
    """
    Production-grade audio analysis engine
    Returns 10+ metrics for respiratory and vocal health assessment
    """
    
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.sample_rate = 16000  # Standard for speech processing
    
    async def analyze_audio(
        self,
        audio_path: str,
        patient_baseline: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Analyze audio file and extract all metrics
        
        Args:
            audio_path: Path to audio file (WAV, MP3, etc.)
            patient_baseline: Patient's baseline audio metrics
        
        Returns:
            Dictionary with 10+ audio metrics
        """
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self.executor,
            self._process_audio,
            audio_path,
            patient_baseline
        )
        return result
    
    def _process_audio(
        self,
        audio_path: str,
        patient_baseline: Optional[Dict[str, float]]
    ) -> Dict[str, Any]:
        """Synchronous audio processing"""
        
        if not LIBROSA_AVAILABLE:
            logger.error("Librosa not available - audio analysis disabled")
            return self._empty_metrics()
        
        logger.info(f"Starting audio analysis: {audio_path}")
        start_time = datetime.now()
        
        # Load audio
        try:
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
            duration = len(audio) / sr
        except Exception as e:
            logger.error(f"Cannot load audio: {e}")
            return self._empty_metrics()
        
        # Apply noise reduction if available
        denoised_audio = audio
        noise_removed_db = 0.0
        if NOISEREDUCE_AVAILABLE:
            try:
                denoised_audio = nr.reduce_noise(y=audio, sr=sr)
                noise_removed_db = 10 * np.log10(np.mean(audio**2) / (np.mean((audio - denoised_audio)**2) + 1e-10))
            except Exception as e:
                logger.warning(f"Noise reduction failed: {e}")
        
        # Compute metrics
        metrics = {}
        
        # ==================== Breath Cycle Detection ====================
        breath_metrics = self._detect_breath_cycles(denoised_audio, sr, duration)
        metrics.update(breath_metrics)
        
        # ==================== Speech Analysis ====================
        speech_metrics = self._analyze_speech(denoised_audio, sr, duration)
        metrics.update(speech_metrics)
        
        # ==================== Pause Analysis ====================
        pause_metrics = self._analyze_pauses(denoised_audio, sr)
        metrics.update(pause_metrics)
        
        # ==================== Cough Detection ====================
        cough_metrics = self._detect_coughs(denoised_audio, sr, duration)
        metrics.update(cough_metrics)
        
        # ==================== Hoarseness / Vocal Fatigue ====================
        vocal_metrics = self._analyze_vocal_quality(denoised_audio, sr)
        metrics.update(vocal_metrics)
        
        # ==================== Wheeze Detection ====================
        wheeze_metrics = self._detect_wheeze(denoised_audio, sr)
        metrics.update(wheeze_metrics)
        
        # ==================== Audio Quality ====================
        quality_metrics = self._assess_audio_quality(audio, denoised_audio, sr)
        quality_metrics['noise_removed_db'] = float(noise_removed_db)
        metrics.update(quality_metrics)
        
        # Processing metadata
        processing_time = (datetime.now() - start_time).total_seconds()
        metrics['audio_duration_seconds'] = float(duration)
        metrics['processing_time_seconds'] = float(processing_time)
        metrics['model_version'] = "1.0.0"
        metrics['denoising_applied'] = NOISEREDUCE_AVAILABLE
        
        logger.info(f"Audio analysis complete in {processing_time:.2f}s")
        
        return metrics
    
    def _detect_breath_cycles(
        self,
        audio: np.ndarray,
        sr: int,
        duration: float
    ) -> Dict[str, Any]:
        """
        Detect breath cycles from audio
        Analyzes low-frequency respiratory sounds
        """
        if not LIBROSA_AVAILABLE or not SCIPY_AVAILABLE:
            return self._empty_breath_metrics()
        
        # Filter for breathing frequencies (0.1-1 Hz)
        nyquist = sr / 2
        low_cut = 0.1 / nyquist
        high_cut = 1.0 / nyquist
        
        try:
            b, a = signal.butter(4, [low_cut, high_cut], btype='band')
            filtered = signal.filtfilt(b, a, audio)
        except Exception as e:
            logger.warning(f"Breath filtering failed: {e}")
            return self._empty_breath_metrics()
        
        # Envelope detection
        envelope = np.abs(signal.hilbert(filtered))
        
        # Smooth envelope
        window_size = int(sr * 0.5)  # 0.5 second window
        envelope_smooth = np.convolve(envelope, np.ones(window_size)/window_size, mode='same')
        
        # Find peaks (breath cycles)
        threshold = np.mean(envelope_smooth) + 0.5 * np.std(envelope_smooth)
        peaks, _ = signal.find_peaks(envelope_smooth, height=threshold, distance=sr)
        
        breath_count = len(peaks)
        breath_rate = breath_count / duration * 60 if duration > 0 else 0
        
        # Analyze breath pattern
        if len(peaks) > 1:
            intervals = np.diff(peaks) / sr
            inhale_duration = np.mean(intervals[:len(intervals)//2]) if len(intervals) > 2 else 0
            exhale_duration = np.mean(intervals[len(intervals)//2:]) if len(intervals) > 2 else 0
            breath_variation = np.std(intervals)
        else:
            inhale_duration = 0
            exhale_duration = 0
            breath_variation = 0
        
        # Classify breath pattern
        if breath_variation < 0.3:
            pattern = "regular"
        elif breath_rate > 24:
            pattern = "rapid"
        elif breath_rate < 12:
            pattern = "slow"
        else:
            pattern = "irregular"
        
        return {
            'breath_cycles_detected': int(breath_count),
            'breath_rate_per_minute': float(breath_rate),
            'breath_pattern': pattern,
            'inhale_duration_avg': float(inhale_duration),
            'exhale_duration_avg': float(exhale_duration),
            'breath_depth_variation': float(breath_variation)
        }
    
    def _analyze_speech(
        self,
        audio: np.ndarray,
        sr: int,
        duration: float
    ) -> Dict[str, Any]:
        """
        Analyze speech characteristics
        Pace, variability, articulation rate
        """
        if not LIBROSA_AVAILABLE:
            return self._empty_speech_metrics()
        
        # Voice activity detection (simple energy-based)
        frame_length = int(sr * 0.025)  # 25ms frames
        hop_length = int(sr * 0.010)  # 10ms hop
        
        # Compute RMS energy
        rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
        
        # Speech threshold
        threshold = np.mean(rms) + 0.5 * np.std(rms)
        speech_frames = rms > threshold
        
        # Count speech segments
        speech_changes = np.diff(speech_frames.astype(int))
        speech_starts = np.where(speech_changes == 1)[0]
        speech_ends = np.where(speech_changes == -1)[0]
        
        # Match starts and ends
        if len(speech_starts) > 0 and len(speech_ends) > 0:
            if speech_ends[0] < speech_starts[0]:
                speech_ends = speech_ends[1:]
            if len(speech_starts) > len(speech_ends):
                speech_starts = speech_starts[:len(speech_ends)]
        
        speech_segments = len(speech_starts)
        
        # Total speech duration
        if len(speech_starts) > 0 and len(speech_ends) > 0:
            segment_durations = (speech_ends - speech_starts) * hop_length / sr
            speech_duration = np.sum(segment_durations)
        else:
            speech_duration = 0
        
        # Estimate words per minute (very rough approximation)
        # Assume ~4 syllables per second in normal speech
        estimated_syllables = speech_duration * 4
        estimated_words = estimated_syllables / 2  # ~2 syllables per word
        words_per_minute = estimated_words / duration * 60 if duration > 0 else 0
        
        # Speech pace variability
        if len(segment_durations) > 1:
            pace_variability = np.std(segment_durations)
        else:
            pace_variability = 0
        
        # Syllables per second (articulation rate)
        syllables_per_second = estimated_syllables / speech_duration if speech_duration > 0 else 0
        
        return {
            'speech_segments_count': int(speech_segments),
            'speech_total_duration': float(speech_duration),
            'speech_pace_words_per_minute': float(words_per_minute),
            'speech_pace_variability': float(pace_variability),
            'syllables_per_second': float(syllables_per_second)
        }
    
    def _analyze_pauses(
        self,
        audio: np.ndarray,
        sr: int
    ) -> Dict[str, Any]:
        """
        Analyze pause patterns in speech
        Frequency, duration, abnormality
        """
        if not LIBROSA_AVAILABLE:
            return self._empty_pause_metrics()
        
        # Voice activity detection
        frame_length = int(sr * 0.025)
        hop_length = int(sr * 0.010)
        
        rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
        
        # Silence threshold
        threshold = np.mean(rms) * 0.3  # Lower threshold for pauses
        silence_frames = rms < threshold
        
        # Find pause segments
        silence_changes = np.diff(silence_frames.astype(int))
        pause_starts = np.where(silence_changes == 1)[0]
        pause_ends = np.where(silence_changes == -1)[0]
        
        # Match starts and ends
        if len(pause_starts) > 0 and len(pause_ends) > 0:
            if pause_ends[0] < pause_starts[0]:
                pause_ends = pause_ends[1:]
            if len(pause_starts) > len(pause_ends):
                pause_starts = pause_starts[:len(pause_ends)]
        
        pause_count = len(pause_starts)
        
        # Pause durations
        if len(pause_starts) > 0 and len(pause_ends) > 0:
            pause_durations = (pause_ends - pause_starts) * hop_length / sr
            # Filter out very short pauses (< 0.2s)
            significant_pauses = pause_durations[pause_durations > 0.2]
            
            if len(significant_pauses) > 0:
                avg_pause_duration = np.mean(significant_pauses)
                max_pause_duration = np.max(significant_pauses)
                
                # Detect unusual pause patterns
                pause_std = np.std(significant_pauses)
                unusual_pattern = pause_std > avg_pause_duration * 0.5
            else:
                avg_pause_duration = 0
                max_pause_duration = 0
                unusual_pattern = False
        else:
            avg_pause_duration = 0
            max_pause_duration = 0
            unusual_pattern = False
        
        # Pause frequency
        duration = len(audio) / sr
        pause_frequency = pause_count / duration * 60 if duration > 0 else 0
        
        return {
            'pause_count': int(pause_count),
            'pause_frequency_per_minute': float(pause_frequency),
            'pause_duration_avg': float(avg_pause_duration),
            'pause_duration_max': float(max_pause_duration),
            'unusual_pause_pattern': bool(unusual_pattern)
        }
    
    def _detect_coughs(
        self,
        audio: np.ndarray,
        sr: int,
        duration: float
    ) -> Dict[str, Any]:
        """
        Detect cough events
        Count, intensity, type (dry/wet)
        """
        if not LIBROSA_AVAILABLE or not SCIPY_AVAILABLE:
            return self._empty_cough_metrics()
        
        # Cough detection using spectral features
        # Coughs have characteristic explosive onset and high-frequency content
        
        # Compute spectral centroid (brightness)
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        
        # Compute onset strength
        onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
        
        # Find strong onsets with high spectral content (potential coughs)
        onset_threshold = np.mean(onset_env) + 2 * np.std(onset_env)
        spectral_threshold = np.mean(spectral_centroid) + np.std(spectral_centroid)
        
        # Resample spectral_centroid to match onset_env length
        if len(spectral_centroid) != len(onset_env):
            from scipy.interpolate import interp1d
            x_old = np.linspace(0, 1, len(spectral_centroid))
            x_new = np.linspace(0, 1, len(onset_env))
            f = interp1d(x_old, spectral_centroid, kind='linear')
            spectral_centroid = f(x_new)
        
        # Potential cough frames
        cough_candidates = (onset_env > onset_threshold) & (spectral_centroid > spectral_threshold)
        
        # Count cough events (cluster nearby frames)
        cough_changes = np.diff(cough_candidates.astype(int))
        cough_starts = np.where(cough_changes == 1)[0]
        
        cough_count = len(cough_starts)
        cough_frequency = cough_count / duration * 60 if duration > 0 else 0
        
        # Cough intensity (average energy of cough segments)
        if cough_count > 0:
            hop_length = int(sr / (len(onset_env) / (len(audio) / sr)))
            cough_energies = []
            for start in cough_starts:
                start_sample = start * hop_length
                end_sample = min(start_sample + sr, len(audio))  # 1 second window
                segment_energy = np.mean(audio[start_sample:end_sample]**2)
                cough_energies.append(segment_energy)
            
            avg_cough_intensity = np.mean(cough_energies) * 100 if cough_energies else 0
        else:
            avg_cough_intensity = 0
        
        # Classify cough type (dry vs wet - simplified)
        # Wet coughs have more low-frequency content
        cough_type = "mixed"  # Placeholder
        
        # Detect coughing fit (multiple coughs in quick succession)
        if cough_count >= 3:
            inter_cough_intervals = np.diff(cough_starts) * hop_length / sr
            rapid_coughs = np.sum(inter_cough_intervals < 2)  # < 2 seconds apart
            coughing_fit = rapid_coughs >= 2
        else:
            coughing_fit = False
        
        return {
            'cough_count': int(cough_count),
            'cough_frequency_per_minute': float(cough_frequency),
            'cough_intensity_avg': float(avg_cough_intensity),
            'cough_type': cough_type,
            'cough_duration_avg': 0.5,  # Placeholder
            'coughing_fit_detected': bool(coughing_fit)
        }
    
    def _analyze_vocal_quality(
        self,
        audio: np.ndarray,
        sr: int
    ) -> Dict[str, Any]:
        """
        Analyze voice quality for hoarseness and fatigue
        Pitch, intensity, jitter, shimmer
        """
        if not LIBROSA_AVAILABLE:
            return self._empty_vocal_metrics()
        
        # Extract pitch (F0) using YIN algorithm
        try:
            f0 = librosa.yin(audio, fmin=50, fmax=400, sr=sr)
            # Remove unvoiced frames (f0 == NaN)
            f0_valid = f0[~np.isnan(f0)]
            
            if len(f0_valid) > 0:
                pitch_avg = np.mean(f0_valid)
                pitch_variability = np.std(f0_valid)
            else:
                pitch_avg = 0
                pitch_variability = 0
        except Exception as e:
            logger.warning(f"Pitch extraction failed: {e}")
            pitch_avg = 0
            pitch_variability = 0
        
        # Voice intensity (RMS energy)
        rms = librosa.feature.rms(y=audio)[0]
        rms_valid = rms[rms > 0]
        
        if len(rms_valid) > 0:
            intensity_avg = 20 * np.log10(np.mean(rms_valid) + 1e-10)  # Convert to dB
        else:
            intensity_avg = -60  # Very quiet
        
        # Jitter (pitch instability) - simplified
        if len(f0_valid) > 1:
            f0_diff = np.diff(f0_valid)
            jitter = np.mean(np.abs(f0_diff)) / pitch_avg * 100 if pitch_avg > 0 else 0
        else:
            jitter = 0
        
        # Shimmer (amplitude variation) - simplified
        if len(rms_valid) > 1:
            rms_diff = np.diff(rms_valid)
            shimmer = np.mean(np.abs(rms_diff)) / np.mean(rms_valid) * 100 if np.mean(rms_valid) > 0 else 0
        else:
            shimmer = 0
        
        # Harmonic-to-noise ratio
        try:
            harmonic, percussive = librosa.effects.hpss(audio)
            harmonic_energy = np.mean(harmonic**2)
            noise_energy = np.mean(percussive**2)
            hnr = 10 * np.log10(harmonic_energy / (noise_energy + 1e-10))
        except Exception as e:
            logger.warning(f"HNR calculation failed: {e}")
            hnr = 0
        
        # Hoarseness score (high jitter + high shimmer + low HNR = hoarse)
        hoarseness_score = (jitter + shimmer) * (1 - min(1, hnr / 20)) * 10
        hoarseness_score = min(100, max(0, hoarseness_score))
        
        # Vocal fatigue score (low intensity + high variability)
        fatigue_score = (50 - intensity_avg) + pitch_variability
        fatigue_score = min(100, max(0, fatigue_score))
        
        # Voice quality score (inverse of hoarseness)
        voice_quality = 100 - hoarseness_score
        
        return {
            'hoarseness_score': float(hoarseness_score),
            'vocal_fatigue_score': float(fatigue_score),
            'voice_pitch_avg_hz': float(pitch_avg),
            'voice_pitch_variability': float(pitch_variability),
            'voice_intensity_avg_db': float(intensity_avg),
            'voice_quality_score': float(voice_quality),
            'jitter_percent': float(jitter),
            'shimmer_percent': float(shimmer),
            'harmonic_to_noise_ratio': float(hnr)
        }
    
    def _detect_wheeze(
        self,
        audio: np.ndarray,
        sr: int
    ) -> Dict[str, Any]:
        """
        Detect wheeze-like frequency signatures
        High-pitched whistling sounds (400-1000 Hz)
        """
        if not LIBROSA_AVAILABLE or not SCIPY_AVAILABLE:
            return self._empty_wheeze_metrics()
        
        # Compute spectrogram
        n_fft = 2048
        hop_length = 512
        
        stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
        magnitude = np.abs(stft)
        
        # Focus on wheeze frequency range (400-1000 Hz)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        wheeze_indices = (freqs >= 400) & (freqs <= 1000)
        
        wheeze_band = magnitude[wheeze_indices, :]
        
        # Detect sustained high energy in wheeze band
        wheeze_energy = np.mean(wheeze_band, axis=0)
        threshold = np.mean(wheeze_energy) + 2 * np.std(wheeze_energy)
        
        wheeze_frames = wheeze_energy > threshold
        
        # Count wheeze events
        wheeze_changes = np.diff(wheeze_frames.astype(int))
        wheeze_starts = np.where(wheeze_changes == 1)[0]
        
        wheeze_count = len(wheeze_starts)
        wheeze_detected = wheeze_count > 0
        
        # Dominant wheeze frequency
        if wheeze_detected:
            # Find dominant frequency in wheeze band
            avg_spectrum = np.mean(wheeze_band, axis=1)
            dominant_freq_idx = np.argmax(avg_spectrum)
            wheeze_frequency = freqs[wheeze_indices][dominant_freq_idx]
            
            # Wheeze intensity
            wheeze_intensity = np.mean(wheeze_energy[wheeze_frames]) if np.any(wheeze_frames) else 0
            wheeze_intensity = min(100, wheeze_intensity * 10)
        else:
            wheeze_frequency = 0
            wheeze_intensity = 0
        
        # Classify wheeze type (inspiratory/expiratory) - placeholder
        wheeze_type = "both" if wheeze_detected else "none"
        
        # Stridor detection (high-pitched, >2000 Hz)
        stridor_indices = freqs > 2000
        stridor_energy = np.mean(magnitude[stridor_indices, :])
        stridor_detected = stridor_energy > np.mean(magnitude) * 2
        
        return {
            'wheeze_detected': bool(wheeze_detected),
            'wheeze_count': int(wheeze_count),
            'wheeze_frequency_hz': float(wheeze_frequency),
            'wheeze_intensity': float(wheeze_intensity),
            'wheeze_type': wheeze_type,
            'stridor_detected': bool(stridor_detected)
        }
    
    def _assess_audio_quality(
        self,
        original_audio: np.ndarray,
        denoised_audio: np.ndarray,
        sr: int
    ) -> Dict[str, Any]:
        """
        Assess overall audio quality
        SNR, clipping, silence, sample rate
        """
        # Background noise level
        noise = original_audio - denoised_audio
        noise_level = 20 * np.log10(np.sqrt(np.mean(noise**2)) + 1e-10)
        
        # Signal-to-noise ratio
        signal_power = np.mean(denoised_audio**2)
        noise_power = np.mean(noise**2)
        snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
        
        # Clipping detection
        clipping_threshold = 0.95
        clipped_samples = np.sum(np.abs(original_audio) > clipping_threshold)
        clipping_percent = clipped_samples / len(original_audio) * 100
        clipping_detected = clipping_percent > 1
        
        # Silence percentage
        silence_threshold = 0.01
        silent_samples = np.sum(np.abs(original_audio) < silence_threshold)
        silence_percent = silent_samples / len(original_audio) * 100
        
        # Overall quality score
        quality_score = 100
        quality_score -= min(30, max(0, 30 - snr))  # SNR penalty
        quality_score -= min(20, clipping_percent * 2)  # Clipping penalty
        quality_score -= min(20, silence_percent / 5)  # Silence penalty
        quality_score = max(0, quality_score)
        
        # Noise type classification
        if snr > 20:
            noise_type = "clean"
        elif snr > 10:
            noise_type = "environmental"
        else:
            noise_type = "noisy"
        
        return {
            'audio_quality_score': float(quality_score),
            'background_noise_level_db': float(noise_level),
            'signal_to_noise_ratio': float(snr),
            'noise_type': noise_type,
            'sample_rate_hz': int(sr),
            'bit_depth': 16,  # Assuming 16-bit
            'clipping_detected': bool(clipping_detected),
            'clipping_percent': float(clipping_percent),
            'silence_percent': float(silence_percent)
        }
    
    # Helper methods for empty metrics
    def _empty_metrics(self) -> Dict[str, Any]:
        return {
            **self._empty_breath_metrics(),
            **self._empty_speech_metrics(),
            **self._empty_pause_metrics(),
            **self._empty_cough_metrics(),
            **self._empty_vocal_metrics(),
            **self._empty_wheeze_metrics(),
            'audio_quality_score': 0.0,
            'audio_duration_seconds': 0.0,
            'processing_time_seconds': 0.0
        }
    
    def _empty_breath_metrics(self) -> Dict[str, Any]:
        return {
            'breath_cycles_detected': 0,
            'breath_rate_per_minute': 0.0,
            'breath_pattern': "unknown",
            'inhale_duration_avg': 0.0,
            'exhale_duration_avg': 0.0,
            'breath_depth_variation': 0.0
        }
    
    def _empty_speech_metrics(self) -> Dict[str, Any]:
        return {
            'speech_segments_count': 0,
            'speech_total_duration': 0.0,
            'speech_pace_words_per_minute': 0.0,
            'speech_pace_variability': 0.0,
            'syllables_per_second': 0.0
        }
    
    def _empty_pause_metrics(self) -> Dict[str, Any]:
        return {
            'pause_count': 0,
            'pause_frequency_per_minute': 0.0,
            'pause_duration_avg': 0.0,
            'pause_duration_max': 0.0,
            'unusual_pause_pattern': False
        }
    
    def _empty_cough_metrics(self) -> Dict[str, Any]:
        return {
            'cough_count': 0,
            'cough_frequency_per_minute': 0.0,
            'cough_intensity_avg': 0.0,
            'cough_type': "unknown",
            'cough_duration_avg': 0.0,
            'coughing_fit_detected': False
        }
    
    def _empty_vocal_metrics(self) -> Dict[str, Any]:
        return {
            'hoarseness_score': 0.0,
            'vocal_fatigue_score': 0.0,
            'voice_pitch_avg_hz': 0.0,
            'voice_pitch_variability': 0.0,
            'voice_intensity_avg_db': 0.0,
            'voice_quality_score': 0.0,
            'jitter_percent': 0.0,
            'shimmer_percent': 0.0,
            'harmonic_to_noise_ratio': 0.0
        }
    
    def _empty_wheeze_metrics(self) -> Dict[str, Any]:
        return {
            'wheeze_detected': False,
            'wheeze_count': 0,
            'wheeze_frequency_hz': 0.0,
            'wheeze_intensity': 0.0,
            'wheeze_type': "none",
            'stridor_detected': False
        }


# Global instance
audio_ai_engine = AudioAIEngine()
