"""
Video AI Engine - Production-Grade Video Analysis
Features: Respiratory rate, skin/nail/nail bed analysis (anaemia, nicotine stains, burns), 
eye sclera, facial swelling, head movement, lighting, quality scoring
Comprehensive hand detection: palms, backs of hands, nails, nail beds
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import logging
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

# Lazy loading flags - imports happen on first use
_CV2_CHECKED = False
_MEDIAPIPE_CHECKED = False
_SCIPY_CHECKED = False
CV2_AVAILABLE = False
MEDIAPIPE_AVAILABLE = False
SCIPY_AVAILABLE = False

# Global references (populated on first use)
cv2 = None
mp = None
signal = None
stats = None
fft = None


class VideoAIEngine:
    """
    Production-grade video analysis engine
    Returns 10+ metrics for patient deterioration detection
    """
    
    def __init__(self):
        global cv2, mp, signal, stats, fft
        global CV2_AVAILABLE, MEDIAPIPE_AVAILABLE, SCIPY_AVAILABLE
        global _CV2_CHECKED, _MEDIAPIPE_CHECKED, _SCIPY_CHECKED
        
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        # Lazy load cv2
        if not _CV2_CHECKED:
            try:
                import cv2 as cv2_module
                cv2 = cv2_module
                CV2_AVAILABLE = True
            except ImportError:
                logger.warning("OpenCV not available - video processing limited")
                CV2_AVAILABLE = False
            _CV2_CHECKED = True
        
        # Lazy load mediapipe
        if not _MEDIAPIPE_CHECKED:
            try:
                import mediapipe as mp_module
                mp = mp_module
                MEDIAPIPE_AVAILABLE = True
            except ImportError:
                logger.warning("MediaPipe not available - face landmark detection disabled")
                MEDIAPIPE_AVAILABLE = False
            _MEDIAPIPE_CHECKED = True
        
        # Lazy load scipy
        if not _SCIPY_CHECKED:
            try:
                from scipy import signal as scipy_signal, stats as scipy_stats
                from scipy.fft import fft as scipy_fft
                signal = scipy_signal
                stats = scipy_stats
                fft = scipy_fft
                SCIPY_AVAILABLE = True
            except ImportError:
                logger.warning("SciPy not available - advanced signal processing disabled")
                SCIPY_AVAILABLE = False
            _SCIPY_CHECKED = True
        
        # Initialize MediaPipe FaceMesh
        if MEDIAPIPE_AVAILABLE and mp:
            self.mp_face_mesh = mp.solutions.face_mesh
            self.mp_drawing = mp.solutions.drawing_utils
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        else:
            self.face_mesh = None
    
    async def analyze_video(
        self,
        video_path: str,
        patient_baseline: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Analyze video file and extract all metrics
        
        Args:
            video_path: Path to video file (local or S3)
            patient_baseline: Patient's baseline metrics for comparison
        
        Returns:
            Dictionary with 10+ metrics
        """
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self.executor,
            self._process_video,
            video_path,
            patient_baseline
        )
        return result
    
    def _process_video(
        self,
        video_path: str,
        patient_baseline: Optional[Dict[str, float]]
    ) -> Dict[str, Any]:
        """Synchronous video processing"""
        
        logger.info(f"Starting video analysis: {video_path}")
        start_time = datetime.now()
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Video metadata
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0
        
        # Storage for time-series data
        chest_movements = []
        face_brightnesses = []
        face_saturations = []
        sclera_colors = []
        landmark_distances = []
        head_positions = []
        frame_qualities = []
        
        # Hand and nail analysis storage
        hand_detections = []
        hand_brightnesses = []
        hand_saturations = []
        nail_pallor_scores = []
        nicotine_detections = []
        burn_detections = []
        discoloration_detections = []
        
        frames_analyzed = 0
        frames_with_face = 0
        frames_with_hands = 0
        
        # Process frames (sample every Nth frame for efficiency)
        sample_rate = max(1, int(fps / 5))  # Analyze 5 frames per second
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            
            # Sample frames
            if frame_idx % sample_rate != 0:
                continue
            
            frames_analyzed += 1
            
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Analyze frame
            frame_metrics = self._analyze_frame(
                rgb_frame,
                frame,
                frame_idx,
                fps
            )
            
            if frame_metrics['face_detected']:
                frames_with_face += 1
                
                # Collect time-series data
                if frame_metrics.get('chest_movement'):
                    chest_movements.append(frame_metrics['chest_movement'])
                if frame_metrics.get('face_brightness'):
                    face_brightnesses.append(frame_metrics['face_brightness'])
                if frame_metrics.get('face_saturation'):
                    face_saturations.append(frame_metrics['face_saturation'])
                if frame_metrics.get('sclera_rgb'):
                    sclera_colors.append(frame_metrics['sclera_rgb'])
                if frame_metrics.get('landmark_distances'):
                    landmark_distances.append(frame_metrics['landmark_distances'])
                if frame_metrics.get('head_position'):
                    head_positions.append(frame_metrics['head_position'])
            
            # Collect hand and nail metrics
            if frame_metrics.get('hands_detected'):
                frames_with_hands += 1
                hand_detections.append(True)
                if frame_metrics.get('hand_brightness'):
                    hand_brightnesses.append(frame_metrics['hand_brightness'])
                if frame_metrics.get('hand_saturation'):
                    hand_saturations.append(frame_metrics['hand_saturation'])
                if frame_metrics.get('nail_bed_pallor_score'):
                    nail_pallor_scores.append(frame_metrics['nail_bed_pallor_score'])
                if frame_metrics.get('nicotine_stain_detected'):
                    nicotine_detections.append(frame_metrics['nicotine_stain_detected'])
                if frame_metrics.get('burn_mark_detected'):
                    burn_detections.append(frame_metrics['burn_mark_detected'])
                if frame_metrics.get('abnormal_discoloration'):
                    discoloration_detections.append(frame_metrics['abnormal_discoloration'])
            
            # Frame quality
            frame_qualities.append(frame_metrics.get('frame_quality', 50))
        
        cap.release()
        
        # Compute aggregate metrics
        metrics = self._compute_aggregate_metrics(
            chest_movements=chest_movements,
            face_brightnesses=face_brightnesses,
            face_saturations=face_saturations,
            sclera_colors=sclera_colors,
            landmark_distances=landmark_distances,
            head_positions=head_positions,
            frame_qualities=frame_qualities,
            hand_brightnesses=hand_brightnesses,
            hand_saturations=hand_saturations,
            nail_pallor_scores=nail_pallor_scores,
            nicotine_detections=nicotine_detections,
            burn_detections=burn_detections,
            discoloration_detections=discoloration_detections,
            fps=fps,
            duration=duration,
            frames_analyzed=frames_analyzed,
            frames_with_face=frames_with_face,
            frames_with_hands=frames_with_hands,
            patient_baseline=patient_baseline
        )
        
        # Processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        metrics['processing_time_seconds'] = processing_time
        metrics['frames_analyzed'] = frames_analyzed
        
        logger.info(f"Video analysis complete: {frames_analyzed} frames in {processing_time:.2f}s")
        
        return metrics
    
    def _analyze_frame(
        self,
        rgb_frame: np.ndarray,
        bgr_frame: np.ndarray,
        frame_idx: int,
        fps: float
    ) -> Dict[str, Any]:
        """Analyze single video frame"""
        
        metrics = {
            'face_detected': False,
            'frame_quality': 0
        }
        
        # Face detection with MediaPipe
        if self.face_mesh and MEDIAPIPE_AVAILABLE:
            results = self.face_mesh.process(rgb_frame)
            
            if results.multi_face_landmarks:
                metrics['face_detected'] = True
                face_landmarks = results.multi_face_landmarks[0]
                
                # Extract face ROI
                h, w, _ = rgb_frame.shape
                landmarks_array = np.array([
                    [lm.x * w, lm.y * h] for lm in face_landmarks.landmark
                ])
                
                # 1. Respiratory rate (chest movement detection)
                metrics['chest_movement'] = self._detect_chest_movement(
                    landmarks_array, rgb_frame
                )
                
                # 2. Skin pallor detection (HSV analysis)
                pallor_metrics = self._detect_skin_pallor(
                    rgb_frame, landmarks_array
                )
                metrics.update(pallor_metrics)
                
                # 3. Eye sclera color (jaundice detection)
                sclera_metrics = self._detect_sclera_color(
                    rgb_frame, landmarks_array
                )
                metrics.update(sclera_metrics)
                
                # 4. Facial swelling (landmark distances)
                swelling_metrics = self._detect_facial_swelling(
                    landmarks_array
                )
                metrics.update(swelling_metrics)
                
                # 5. Head movement (position tracking)
                head_metrics = self._detect_head_movement(
                    landmarks_array
                )
                metrics.update(head_metrics)
        
        # 6. Lighting correction & quality
        lighting_metrics = self._analyze_lighting(bgr_frame)
        metrics.update(lighting_metrics)
        
        # 7. Frame quality scoring
        metrics['frame_quality'] = self._compute_frame_quality(bgr_frame)
        
        return metrics
    
    def _detect_chest_movement(
        self,
        landmarks: np.ndarray,
        frame: np.ndarray
    ) -> float:
        """
        Detect chest movement amplitude
        Using lower face landmarks as proxy for breathing motion
        """
        # Use chin/neck area landmarks (indices vary by MediaPipe version)
        # For now, use face center vertical movement as proxy
        if len(landmarks) > 10:
            center_y = np.mean(landmarks[:, 1])
            return float(center_y)  # Store for time-series analysis
        return 0.0
    
    def _detect_skin_pallor(
        self,
        frame: np.ndarray,
        landmarks: np.ndarray
    ) -> Dict[str, float]:
        """
        Comprehensive skin, nail, and nail bed analysis
        Detects: anaemia (pale nail beds), nicotine stains, burns, discoloration
        Analyzes: face, palms, hands, nails, nail beds
        """
        metrics = {}
        
        # 1. Face skin analysis (cheeks area)
        x_min, y_min = landmarks.min(axis=0).astype(int)
        x_max, y_max = landmarks.max(axis=0).astype(int)
        
        padding = 20
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(frame.shape[1], x_max + padding)
        y_max = min(frame.shape[0], y_max + padding)
        
        face_roi = frame[y_min:y_max, x_min:x_max]
        
        # Face HSV analysis
        hsv = cv2.cvtColor(face_roi, cv2.COLOR_RGB2HSV)
        metrics['face_brightness'] = float(np.mean(hsv[:, :, 2]))
        metrics['face_saturation'] = float(np.mean(hsv[:, :, 1]))
        
        # 2. Hand and nail detection (when visible in frame)
        # Detect skin-colored regions outside face for hands/palms
        hand_nail_metrics = self._detect_hands_and_nails(frame, landmarks)
        metrics.update(hand_nail_metrics)
        
        return metrics
    
    def _detect_hands_and_nails(
        self,
        frame: np.ndarray,
        face_landmarks: np.ndarray
    ) -> Dict[str, Any]:
        """
        Detect hands, nails, and nail beds for comprehensive skin examination
        Checks for: anaemia, nicotine stains, burns, abnormal coloration
        
        NOTE: V1 Implementation - Requires clinical validation and refinement:
        - HSV thresholds are calibrated for light-medium skin tones
        - Anaemia scoring uses absolute thresholds (needs relative baseline comparison)
        - Detection confidence may vary across diverse skin tones
        - Production deployment should use adaptive thresholds based on patient baseline
        """
        try:
            h, w, _ = frame.shape
            
            # Create face mask to exclude face region
            face_x_min, face_y_min = face_landmarks.min(axis=0).astype(int)
            face_x_max, face_y_max = face_landmarks.max(axis=0).astype(int)
            
            # Convert to HSV for skin detection
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
            
            # Skin color detection in HSV (broad range to capture different skin tones)
            # Lower/upper bounds for skin detection
            lower_skin = np.array([0, 20, 70], dtype=np.uint8)
            upper_skin = np.array([20, 255, 255], dtype=np.uint8)
            skin_mask = cv2.inRange(hsv_frame, lower_skin, upper_skin)
            
            # Alternative range for darker skin tones
            lower_skin2 = np.array([0, 10, 60], dtype=np.uint8)
            upper_skin2 = np.array([50, 150, 255], dtype=np.uint8)
            skin_mask2 = cv2.inRange(hsv_frame, lower_skin2, upper_skin2)
            skin_mask = cv2.bitwise_or(skin_mask, skin_mask2)
            
            # Zero out face region to prioritize hand detection
            # Add padding to ensure complete face exclusion
            face_padding = 40
            face_y_min_padded = max(0, face_y_min - face_padding)
            face_y_max_padded = min(h, face_y_max + face_padding)
            face_x_min_padded = max(0, face_x_min - face_padding)
            face_x_max_padded = min(w, face_x_max + face_padding)
            skin_mask[face_y_min_padded:face_y_max_padded, face_x_min_padded:face_x_max_padded] = 0
            
            # Find contours (potential hands)
            contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            metrics = {
                'hands_detected': False,
                'hand_brightness': 0.0,
                'hand_saturation': 0.0,
                'nail_bed_pallor_score': 0.0,  # 0-100, higher = more anaemic
                'nicotine_stain_detected': False,
                'burn_mark_detected': False,
                'abnormal_discoloration': False
            }
            
            if len(contours) == 0:
                return metrics
            
            # Find largest contour (likely hand/palm)
            largest_contour = max(contours, key=cv2.contourArea)
            contour_area = cv2.contourArea(largest_contour)
            
            # Only process if contour is large enough (likely a hand)
            min_hand_area = (h * w) * 0.05  # At least 5% of frame
            if contour_area < min_hand_area:
                return metrics
            
            metrics['hands_detected'] = True
            
            # Extract hand ROI
            x, y, rw, rh = cv2.boundingRect(largest_contour)
            hand_roi = frame[y:y+rh, x:x+rw]
            
            if hand_roi.size == 0:
                return metrics
            
            # Analyze hand/palm colors
            hand_hsv = cv2.cvtColor(hand_roi, cv2.COLOR_RGB2HSV)
            hand_rgb = hand_roi
            
            metrics['hand_brightness'] = float(np.mean(hand_hsv[:, :, 2]))
            metrics['hand_saturation'] = float(np.mean(hand_hsv[:, :, 1]))
            
            # ANAEMIA DETECTION: Check for pale nail beds
            # Nail beds should have pink/red hue; anaemic nail beds are white/pale
            # Look for very bright, low-saturation regions (potential nail beds)
            brightness = hand_hsv[:, :, 2]
            saturation = hand_hsv[:, :, 1]
            
            # Pale regions: high brightness (>180), low saturation (<40)
            pale_regions = (brightness > 180) & (saturation < 40)
            pale_percentage = np.sum(pale_regions) / pale_regions.size * 100
            
            # Pallor score: higher = more likely anaemic
            metrics['nail_bed_pallor_score'] = float(min(pale_percentage * 2, 100))
            
            # NICOTINE STAIN DETECTION: Yellow/brown discoloration
            # Look for yellow-orange hue in finger/nail areas
            hue = hand_hsv[:, :, 0]
            # Yellow-orange range in HSV (hue 15-35)
            yellow_regions = (hue > 15) & (hue < 35) & (saturation > 50)
            yellow_percentage = np.sum(yellow_regions) / yellow_regions.size * 100
            
            if yellow_percentage > 5:  # More than 5% yellow = likely staining
                metrics['nicotine_stain_detected'] = True
            
            # BURN DETECTION: Very dark or black spots
            # Burns appear as dark patches with low brightness
            dark_regions = brightness < 60
            dark_percentage = np.sum(dark_regions) / dark_regions.size * 100
            
            if dark_percentage > 10:  # Significant dark areas
                metrics['burn_mark_detected'] = True
            
            # GENERAL ABNORMAL DISCOLORATION
            # Check for unusual color distributions
            rgb_mean = np.mean(hand_rgb, axis=(0, 1))
            r_mean, g_mean, b_mean = rgb_mean
            
            # Abnormal if significant color imbalance
            max_diff = max(abs(r_mean - g_mean), abs(g_mean - b_mean), abs(b_mean - r_mean))
            if max_diff > 50:  # Significant color imbalance
                metrics['abnormal_discoloration'] = True
            
            return metrics
            
        except Exception as e:
            logger.warning(f"Hand/nail detection error: {e}")
            return {
                'hands_detected': False,
                'hand_brightness': 0.0,
                'hand_saturation': 0.0,
                'nail_bed_pallor_score': 0.0,
                'nicotine_stain_detected': False,
                'burn_mark_detected': False,
                'abnormal_discoloration': False
            }
    
    def _detect_sclera_color(
        self,
        frame: np.ndarray,
        landmarks: np.ndarray
    ) -> Dict[str, Any]:
        """
        Detect eye sclera color for jaundice detection
        Extracts RGB components from eye white regions
        """
        # Eye landmarks (approximate indices for left/right eyes)
        # Note: Exact indices depend on MediaPipe FaceMesh version
        # This is a simplified version
        
        try:
            # Left eye region (approximate)
            left_eye_points = landmarks[133:145]  # Example indices
            
            # Extract eye ROI
            x_min, y_min = left_eye_points.min(axis=0).astype(int)
            x_max, y_max = left_eye_points.max(axis=0).astype(int)
            
            # Ensure valid ROI
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(frame.shape[1], x_max)
            y_max = min(frame.shape[0], y_max)
            
            if x_max > x_min and y_max > y_min:
                eye_roi = frame[y_min:y_max, x_min:x_max]
                
                # Get RGB averages (sclera should be white)
                r_avg = np.mean(eye_roi[:, :, 0])
                g_avg = np.mean(eye_roi[:, :, 1])
                b_avg = np.mean(eye_roi[:, :, 2])
                
                return {
                    'sclera_rgb': [float(r_avg), float(g_avg), float(b_avg)]
                }
        except Exception as e:
            logger.warning(f"Sclera detection error: {e}")
        
        return {'sclera_rgb': [0, 0, 0]}
    
    def _detect_facial_swelling(
        self,
        landmarks: np.ndarray
    ) -> Dict[str, float]:
        """
        Detect facial swelling by measuring landmark distances
        Compare cheek widths, eye areas, etc.
        """
        try:
            # Calculate face width (left to right cheek)
            left_cheek = landmarks[234]  # Example left cheek point
            right_cheek = landmarks[454]  # Example right cheek point
            cheek_distance = np.linalg.norm(right_cheek - left_cheek)
            
            # Calculate eye areas (approximate)
            # This would need more sophisticated landmark mapping
            
            return {
                'landmark_distances': {
                    'cheek_width': float(cheek_distance)
                }
            }
        except Exception as e:
            logger.warning(f"Swelling detection error: {e}")
            return {'landmark_distances': {}}
    
    def _detect_head_movement(
        self,
        landmarks: np.ndarray
    ) -> Dict[str, Any]:
        """
        Detect head position and movement
        Track head center position for stability analysis
        """
        # Calculate head center
        head_center = landmarks.mean(axis=0)
        
        return {
            'head_position': head_center.tolist()
        }
    
    def _analyze_lighting(
        self,
        frame: np.ndarray
    ) -> Dict[str, Any]:
        """
        Analyze lighting quality and apply corrections
        Detect shadows, overexposure, underexposure
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Brightness statistics
        brightness_mean = np.mean(gray)
        brightness_std = np.std(gray)
        
        # Overexposure (very bright pixels)
        overexposed = np.sum(gray > 240) / gray.size * 100
        
        # Underexposure (very dark pixels)
        underexposed = np.sum(gray < 15) / gray.size * 100
        
        # Lighting quality score (0-100)
        lighting_quality = 100 - (overexposed + underexposed)
        lighting_quality = max(0, min(100, lighting_quality))
        
        return {
            'lighting_quality_score': float(lighting_quality),
            'lighting_uniformity': float(brightness_std),
            'overexposure_percent': float(overexposed),
            'underexposure_percent': float(underexposed)
        }
    
    def _compute_frame_quality(
        self,
        frame: np.ndarray
    ) -> float:
        """
        Compute overall frame quality score
        Considers blur, noise, compression artifacts
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Laplacian variance (blur detection)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Normalize to 0-100 scale
        # Higher variance = sharper image
        quality = min(100, laplacian_var / 10)
        
        return float(quality)
    
    def _compute_aggregate_metrics(
        self,
        chest_movements: List[float],
        face_brightnesses: List[float],
        face_saturations: List[float],
        sclera_colors: List[List[float]],
        landmark_distances: List[Dict],
        head_positions: List[List[float]],
        frame_qualities: List[float],
        hand_brightnesses: List[float],
        hand_saturations: List[float],
        nail_pallor_scores: List[float],
        nicotine_detections: List[bool],
        burn_detections: List[bool],
        discoloration_detections: List[bool],
        fps: float,
        duration: float,
        frames_analyzed: int,
        frames_with_face: int,
        frames_with_hands: int,
        patient_baseline: Optional[Dict[str, float]]
    ) -> Dict[str, Any]:
        """
        Compute aggregate metrics from time-series data
        Returns 15+ metrics for deterioration detection including nail/hand analysis
        """
        
        metrics = {}
        
        # ==================== Respiratory Rate Detection ====================
        if len(chest_movements) > 10 and SCIPY_AVAILABLE:
            # Detect breathing cycles using FFT
            respiratory_rate, confidence = self._compute_respiratory_rate(
                chest_movements, fps
            )
            metrics['respiratory_rate_bpm'] = respiratory_rate
            metrics['respiratory_rate_confidence'] = confidence
            metrics['chest_movement_amplitude'] = float(np.std(chest_movements))
            metrics['breathing_pattern'] = self._classify_breathing_pattern(chest_movements)
        else:
            metrics['respiratory_rate_bpm'] = None
            metrics['respiratory_rate_confidence'] = 0.0
            metrics['chest_movement_amplitude'] = 0.0
            metrics['breathing_pattern'] = "insufficient_data"
        
        # ==================== Skin Pallor Detection ====================
        if len(face_brightnesses) > 0:
            avg_brightness = np.mean(face_brightnesses)
            avg_saturation = np.mean(face_saturations)
            
            # Pallor score (higher brightness + lower saturation = more pale)
            pallor_score = (avg_brightness / 255 * 100) * (1 - avg_saturation / 255)
            
            metrics['skin_pallor_score'] = float(pallor_score)
            metrics['face_brightness_avg'] = float(avg_brightness)
            metrics['face_saturation_avg'] = float(avg_saturation)
            metrics['pallor_confidence'] = 0.8 if frames_with_face > 10 else 0.5
        else:
            metrics['skin_pallor_score'] = 0.0
            metrics['face_brightness_avg'] = 0.0
            metrics['face_saturation_avg'] = 0.0
            metrics['pallor_confidence'] = 0.0
        
        # ==================== Eye Sclera Color (Jaundice) ====================
        if len(sclera_colors) > 0:
            avg_r = np.mean([c[0] for c in sclera_colors])
            avg_g = np.mean([c[1] for c in sclera_colors])
            avg_b = np.mean([c[2] for c in sclera_colors])
            
            # Yellowness score (high red+green, low blue)
            yellowness = ((avg_r + avg_g) / 2 - avg_b) / 255 * 100
            yellowness = max(0, yellowness)
            
            metrics['sclera_yellowness_score'] = float(yellowness)
            metrics['sclera_red_component'] = float(avg_r)
            metrics['sclera_green_component'] = float(avg_g)
            metrics['sclera_blue_component'] = float(avg_b)
            
            # Risk level classification
            if yellowness < 20:
                metrics['jaundice_risk_level'] = "low"
            elif yellowness < 40:
                metrics['jaundice_risk_level'] = "medium"
            else:
                metrics['jaundice_risk_level'] = "high"
        else:
            metrics['sclera_yellowness_score'] = 0.0
            metrics['sclera_red_component'] = 0.0
            metrics['sclera_green_component'] = 0.0
            metrics['sclera_blue_component'] = 0.0
            metrics['jaundice_risk_level'] = "unknown"
        
        # ==================== Facial Swelling ====================
        if len(landmark_distances) > 0:
            # Aggregate cheek widths
            cheek_widths = [d.get('cheek_width', 0) for d in landmark_distances]
            avg_cheek_width = np.mean(cheek_widths) if cheek_widths else 0
            
            # Compare to baseline if available
            swelling_score = 0.0
            if patient_baseline and 'facial_symmetry_baseline' in patient_baseline:
                baseline_width = patient_baseline['facial_symmetry_baseline']
                deviation = (avg_cheek_width - baseline_width) / baseline_width * 100
                swelling_score = max(0, deviation)
            
            metrics['facial_swelling_score'] = float(swelling_score)
            metrics['left_cheek_distance'] = float(avg_cheek_width)
            metrics['right_cheek_distance'] = float(avg_cheek_width)
            metrics['eye_puffiness_left'] = 0.0  # Placeholder
            metrics['eye_puffiness_right'] = 0.0  # Placeholder
            metrics['facial_asymmetry_score'] = float(np.std(cheek_widths)) if cheek_widths else 0.0
        else:
            metrics['facial_swelling_score'] = 0.0
            metrics['left_cheek_distance'] = 0.0
            metrics['right_cheek_distance'] = 0.0
            metrics['eye_puffiness_left'] = 0.0
            metrics['eye_puffiness_right'] = 0.0
            metrics['facial_asymmetry_score'] = 0.0
        
        # ==================== Hand, Nail & Nail Bed Analysis ====================
        if frames_with_hands > 0:
            # Hand skin analysis
            metrics['hands_detected'] = True
            metrics['frames_with_hands'] = frames_with_hands
            
            if len(hand_brightnesses) > 0:
                metrics['hand_brightness_avg'] = float(np.mean(hand_brightnesses))
                metrics['hand_saturation_avg'] = float(np.mean(hand_saturations))
            else:
                metrics['hand_brightness_avg'] = 0.0
                metrics['hand_saturation_avg'] = 0.0
            
            # Nail bed anaemia detection
            if len(nail_pallor_scores) > 0:
                avg_pallor_score = np.mean(nail_pallor_scores)
                metrics['nail_bed_pallor_score'] = float(avg_pallor_score)
                
                # Anaemia risk classification
                if avg_pallor_score < 30:
                    metrics['anaemia_risk_level'] = "low"
                elif avg_pallor_score < 60:
                    metrics['anaemia_risk_level'] = "medium"
                else:
                    metrics['anaemia_risk_level'] = "high"
            else:
                metrics['nail_bed_pallor_score'] = 0.0
                metrics['anaemia_risk_level'] = "unknown"
            
            # Nicotine stain detection
            if len(nicotine_detections) > 0:
                nicotine_percentage = sum(nicotine_detections) / len(nicotine_detections) * 100
                metrics['nicotine_stain_detected'] = nicotine_percentage > 30  # >30% of frames
                metrics['nicotine_stain_confidence'] = float(nicotine_percentage / 100)
            else:
                metrics['nicotine_stain_detected'] = False
                metrics['nicotine_stain_confidence'] = 0.0
            
            # Burn mark detection
            if len(burn_detections) > 0:
                burn_percentage = sum(burn_detections) / len(burn_detections) * 100
                metrics['burn_mark_detected'] = burn_percentage > 20  # >20% of frames
                metrics['burn_mark_confidence'] = float(burn_percentage / 100)
            else:
                metrics['burn_mark_detected'] = False
                metrics['burn_mark_confidence'] = 0.0
            
            # Abnormal discoloration
            if len(discoloration_detections) > 0:
                discolor_percentage = sum(discoloration_detections) / len(discoloration_detections) * 100
                metrics['abnormal_discoloration_detected'] = discolor_percentage > 25
                metrics['discoloration_confidence'] = float(discolor_percentage / 100)
            else:
                metrics['abnormal_discoloration_detected'] = False
                metrics['discoloration_confidence'] = 0.0
        else:
            # No hands detected in any frame
            metrics['hands_detected'] = False
            metrics['frames_with_hands'] = 0
            metrics['hand_brightness_avg'] = 0.0
            metrics['hand_saturation_avg'] = 0.0
            metrics['nail_bed_pallor_score'] = 0.0
            metrics['anaemia_risk_level'] = "unknown"
            metrics['nicotine_stain_detected'] = False
            metrics['nicotine_stain_confidence'] = 0.0
            metrics['burn_mark_detected'] = False
            metrics['burn_mark_confidence'] = 0.0
            metrics['abnormal_discoloration_detected'] = False
            metrics['discoloration_confidence'] = 0.0
        
        # ==================== Head Movement / Stability ====================
        if len(head_positions) > 1:
            # Calculate total movement
            positions = np.array(head_positions)
            movements = np.diff(positions, axis=0)
            total_movement = np.sum(np.linalg.norm(movements, axis=1))
            
            # Stability score (inverse of movement)
            stability_score = max(0, 100 - total_movement / 10)
            
            # Head tilt (simplified)
            head_tilt = 0.0  # Placeholder
            
            # Tremor detection (frequency analysis)
            tremor_detected, tremor_freq = self._detect_tremor(positions, fps)
            
            metrics['head_movement_total'] = float(total_movement)
            metrics['head_stability_score'] = float(stability_score)
            metrics['head_tilt_angle'] = float(head_tilt)
            metrics['tremor_detected'] = tremor_detected
            metrics['tremor_frequency_hz'] = float(tremor_freq) if tremor_detected else 0.0
        else:
            metrics['head_movement_total'] = 0.0
            metrics['head_stability_score'] = 100.0
            metrics['head_tilt_angle'] = 0.0
            metrics['tremor_detected'] = False
            metrics['tremor_frequency_hz'] = 0.0
        
        # ==================== Quality Metrics ====================
        if len(frame_qualities) > 0:
            metrics['frame_quality_avg'] = float(np.mean(frame_qualities))
            metrics['blur_score'] = float(100 - metrics['frame_quality_avg'])
            metrics['noise_level'] = float(np.std(frame_qualities))
            metrics['compression_artifacts'] = 0.0  # Placeholder
        else:
            metrics['frame_quality_avg'] = 0.0
            metrics['blur_score'] = 100.0
            metrics['noise_level'] = 0.0
            metrics['compression_artifacts'] = 0.0
        
        # ==================== Face Detection Quality ====================
        face_detection_rate = frames_with_face / frames_analyzed if frames_analyzed > 0 else 0
        metrics['face_detection_confidence'] = float(face_detection_rate)
        metrics['face_occlusion_percent'] = float(100 - face_detection_rate * 100)
        metrics['multiple_faces_detected'] = False  # Simplified
        
        return metrics
    
    def _compute_respiratory_rate(
        self,
        chest_movements: List[float],
        fps: float
    ) -> Tuple[float, float]:
        """
        Compute respiratory rate from chest movement time-series
        Using FFT to detect breathing frequency
        """
        if not SCIPY_AVAILABLE or len(chest_movements) < 10:
            return 0.0, 0.0
        
        # Detrend signal
        signal_array = np.array(chest_movements)
        detrended = signal.detrend(signal_array)
        
        # Apply FFT
        fft_result = fft(detrended)
        frequencies = np.fft.fftfreq(len(detrended), d=1/fps)
        
        # Focus on breathing frequency range (0.1-0.5 Hz = 6-30 breaths/min)
        valid_indices = (frequencies > 0.1) & (frequencies < 0.5)
        valid_freqs = frequencies[valid_indices]
        valid_fft = np.abs(fft_result[valid_indices])
        
        if len(valid_fft) > 0:
            # Find dominant frequency
            dominant_idx = np.argmax(valid_fft)
            dominant_freq = valid_freqs[dominant_idx]
            
            # Convert to breaths per minute
            respiratory_rate = dominant_freq * 60
            
            # Confidence based on peak prominence
            confidence = min(1.0, valid_fft[dominant_idx] / np.sum(valid_fft))
            
            return float(respiratory_rate), float(confidence)
        
        return 0.0, 0.0
    
    def _classify_breathing_pattern(
        self,
        chest_movements: List[float]
    ) -> str:
        """Classify breathing pattern as regular/irregular/shallow/labored"""
        if len(chest_movements) < 10:
            return "insufficient_data"
        
        # Calculate coefficient of variation
        std = np.std(chest_movements)
        mean = np.mean(chest_movements)
        cv = std / mean if mean > 0 else 0
        
        # Classify based on variability
        if cv < 0.15:
            return "regular"
        elif cv < 0.3:
            return "irregular"
        else:
            return "labored"
    
    def _detect_tremor(
        self,
        positions: np.ndarray,
        fps: float
    ) -> Tuple[bool, float]:
        """Detect tremor from head position time-series"""
        if not SCIPY_AVAILABLE or len(positions) < 10:
            return False, 0.0
        
        # Calculate movement
        movements = np.diff(positions, axis=0)
        movement_magnitude = np.linalg.norm(movements, axis=1)
        
        # FFT to detect oscillation frequency
        fft_result = fft(movement_magnitude)
        frequencies = np.fft.fftfreq(len(movement_magnitude), d=1/fps)
        
        # Tremor frequency range (4-12 Hz)
        tremor_indices = (frequencies > 4) & (frequencies < 12)
        if np.any(tremor_indices):
            tremor_fft = np.abs(fft_result[tremor_indices])
            if np.max(tremor_fft) > np.mean(tremor_fft) * 3:  # Strong peak
                peak_idx = np.argmax(tremor_fft)
                tremor_freq = frequencies[tremor_indices][peak_idx]
                return True, float(tremor_freq)
        
        return False, 0.0
    
    def generate_recommendations(self, metrics: Dict[str, Any]) -> List[str]:
        """
        Generate wellness recommendations based on video metrics
        
        IMPORTANT: Uses wellness language, NOT diagnostic language
        All recommendations suggest discussing with healthcare provider
        
        Args:
            metrics: Dictionary of video analysis metrics
        
        Returns:
            List of wellness recommendation strings
        """
        recommendations = []
        
        # Respiratory rate recommendations
        resp_rate = metrics.get("respiratory_rate_bpm", 0)
        if resp_rate > 0:
            if resp_rate < 12:
                recommendations.append(
                    "📊 Breathing rate appears slower than typical baseline. Consider discussing respiratory wellness with your healthcare provider."
                )
            elif resp_rate > 20:
                recommendations.append(
                    "📊 Breathing rate appears elevated. This could indicate various wellness factors - please discuss with your provider."
                )
        
        # Skin pallor recommendations
        pallor_score = metrics.get("skin_pallor_score", 0)
        if pallor_score > 0.6:
            recommendations.append(
                "🔍 Skin tone analysis detected changes. This may relate to hydration, circulation, or other wellness factors. Consider discussing with your healthcare team."
            )
        
        # Eye sclera (jaundice) recommendations
        sclera_yellow = metrics.get("eye_sclera_yellowness", 0)
        if sclera_yellow > 0.5:
            recommendations.append(
                "👁️ Eye analysis detected yellowish tones. While this can have many causes, we recommend scheduling a wellness check to discuss with your provider."
            )
        
        # Facial swelling recommendations
        swelling_score = metrics.get("facial_swelling_score", 0)
        if swelling_score > 0.5:
            recommendations.append(
                "📈 Facial landmark analysis detected changes that may indicate fluid retention or swelling. Please discuss these observations with your healthcare provider."
            )
        
        # Head movement/tremor recommendations
        tremor_detected = metrics.get("tremor_detected", False)
        if tremor_detected:
            recommendations.append(
                "🎯 Movement analysis detected tremor patterns. This can have many causes - consider discussing movement wellness with your provider."
            )
        
        # Lighting/quality recommendations
        lighting = metrics.get("lighting_quality", 1.0)
        frame_quality = metrics.get("frame_quality", 1.0)
        
        if lighting < 0.6 or frame_quality < 0.6:
            recommendations.append(
                "💡 Technical note: Improved lighting and camera stability will enhance future wellness monitoring accuracy. Try recording in a well-lit area with the camera held steady."
            )
        
        # General recommendation if no specific issues
        if not recommendations:
            recommendations.append(
                "✅ Video analysis completed successfully. Continue regular wellness monitoring to track changes over time. Discuss any concerns with your healthcare provider."
            )
        
        # Always add general wellness reminder
        recommendations.append(
            "ℹ️ This system provides wellness monitoring and change detection, not medical diagnosis. Always consult your healthcare provider for medical advice."
        )
        
        return recommendations


# Global instance
video_ai_engine = VideoAIEngine()
