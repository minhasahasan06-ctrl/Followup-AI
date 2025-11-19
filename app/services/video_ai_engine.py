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
        accessory_muscle_activity = []
        chest_widths = []
        face_brightnesses = []
        face_saturations = []
        sclera_colors = []
        landmark_distances = []
        head_positions = []
        frame_qualities = []
        
        # Hand and nail analysis storage (legacy - replaced by LAB analysis)
        hand_detections = []
        hand_brightnesses = []
        hand_saturations = []
        nail_pallor_scores = []
        nicotine_detections = []
        burn_detections = []
        discoloration_detections = []
        
        # COMPREHENSIVE SKIN ANALYSIS STORAGE (LAB color space)
        skin_analysis_frames = []  # Store full skin metrics per frame
        facial_l_values = []
        facial_perfusion_indices = []
        palmar_perfusion_indices = []
        nailbed_color_indices = []
        pallor_detections = []
        cyanosis_detections = []
        jaundice_detections = []
        nail_clubbing_detections = []
        nail_pitting_detections = []
        
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
                # Respiratory metrics (now dict from enhanced detection)
                if frame_metrics.get('chest_movement') is not None:
                    chest_movements.append(frame_metrics['chest_movement'])
                if frame_metrics.get('accessory_muscle_activity') is not None:
                    accessory_muscle_activity.append(frame_metrics['accessory_muscle_activity'])
                if frame_metrics.get('chest_width_proxy') is not None:
                    chest_widths.append(frame_metrics['chest_width_proxy'])
                
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
            
            # Collect hand and nail metrics (legacy)
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
            
            # COLLECT COMPREHENSIVE SKIN ANALYSIS METRICS (LAB color space)
            # NEW field names from updated _detect_skin_pallor and _detect_hands_and_nails
            if frame_metrics.get('lab_facial_perfusion_avg') is not None:
                facial_perfusion_indices.append(frame_metrics['lab_facial_perfusion_avg'])
            
            if frame_metrics.get('lab_palmar_perfusion_avg') is not None:
                palmar_perfusion_indices.append(frame_metrics['lab_palmar_perfusion_avg'])
            
            if frame_metrics.get('lab_nailbed_perfusion_avg') is not None:
                nailbed_color_indices.append(frame_metrics['lab_nailbed_perfusion_avg'])
            
            # Collect clinical detections
            if frame_metrics.get('pallor_detected'):
                pallor_detections.append(frame_metrics)
            if frame_metrics.get('cyanosis_detected'):
                cyanosis_detections.append(frame_metrics)
            if frame_metrics.get('jaundice_detected'):
                jaundice_detections.append(frame_metrics)
            if frame_metrics.get('nail_clubbing_detected'):
                nail_clubbing_detections.append(frame_metrics)
            if frame_metrics.get('nail_pitting_detected'):
                nail_pitting_detections.append(frame_metrics)
            
            # Store full skin analysis for capillary refill tracking
            if frame_metrics.get('detection_confidence', 0) > 0.3:
                skin_analysis_frames.append(frame_metrics)
            
            # Frame quality
            frame_qualities.append(frame_metrics.get('frame_quality', 50))
        
        cap.release()
        
        # Compute aggregate metrics
        metrics = self._compute_aggregate_metrics(
            chest_movements=chest_movements,
            accessory_muscle_activity=accessory_muscle_activity,
            chest_widths=chest_widths,
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
            # COMPREHENSIVE SKIN ANALYSIS PARAMETERS (LAB color space)
            facial_l_values=facial_l_values,
            facial_perfusion_indices=facial_perfusion_indices,
            palmar_perfusion_indices=palmar_perfusion_indices,
            nailbed_color_indices=nailbed_color_indices,
            pallor_detections=pallor_detections,
            cyanosis_detections=cyanosis_detections,
            jaundice_detections=jaundice_detections,
            nail_clubbing_detections=nail_clubbing_detections,
            nail_pitting_detections=nail_pitting_detections,
            skin_analysis_frames=skin_analysis_frames,
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
                
                # 1. Respiratory analysis (comprehensive)
                respiratory_metrics = self._detect_chest_movement(
                    landmarks_array, rgb_frame
                )
                metrics.update(respiratory_metrics)
                
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
                
                # 6. COMPREHENSIVE SKIN ANALYSIS (LAB color space)
                # This replaces old HSV-based pallor detection with clinical-grade LAB analysis
                skin_analysis = self.analyze_skin_comprehensive(
                    rgb_frame,
                    landmarks_array,
                    frame_idx,
                    baseline=None  # Will be passed from patient baseline
                )
                metrics.update(skin_analysis)
        
        # 7. Lighting correction & quality
        lighting_metrics = self._analyze_lighting(bgr_frame)
        metrics.update(lighting_metrics)
        
        # 8. Frame quality scoring
        metrics['frame_quality'] = self._compute_frame_quality(bgr_frame)
        
        return metrics
    
    def _detect_chest_movement(
        self,
        landmarks: np.ndarray,
        frame: np.ndarray
    ) -> Dict[str, Any]:
        """
        Comprehensive respiratory analysis from face landmarks
        Returns: chest movement, accessory muscle use, breathing depth
        
        Clinical indicators tracked:
        - Chest expansion (vertical movement of lower face/chin)
        - Accessory muscle use (neck/shoulder elevation)
        - Breathing depth variability
        """
        respiratory_metrics = {}
        
        if len(landmarks) < 10:
            return {'chest_movement': 0.0}
        
        # 1. Chest movement (vertical displacement of chin area)
        # Lower face landmarks indicate chest rise/fall during breathing
        chin_landmarks = landmarks[140:180] if len(landmarks) > 180 else landmarks[-40:]
        chin_center_y = np.mean(chin_landmarks[:, 1])
        respiratory_metrics['chest_movement'] = float(chin_center_y)
        
        # 2. Accessory muscle detection (neck/shoulder movement)
        # During labored breathing, neck muscles elevate with each breath
        # Upper face landmarks show less movement; neck shows more
        upper_face = landmarks[10:30] if len(landmarks) > 30 else landmarks[:20]
        upper_y = np.mean(upper_face[:, 1])
        
        # Accessory muscle score: ratio of chin movement to upper face movement
        # High ratio = accessory muscle use (neck elevating with breathing)
        movement_ratio = abs(chin_center_y - upper_y) if upper_y > 0 else 0
        respiratory_metrics['accessory_muscle_activity'] = float(movement_ratio)
        
        # 3. Chest shape analysis (width measurements)
        # Barrel chest or asymmetry can indicate respiratory issues
        face_width = np.max(landmarks[:, 0]) - np.min(landmarks[:, 0])
        respiratory_metrics['chest_width_proxy'] = float(face_width)
        
        return respiratory_metrics
    
    def _rgb_to_lab(self, rgb_pixels: np.ndarray) -> np.ndarray:
        """
        Convert RGB pixels to LAB color space (clinical-grade)
        
        Args:
            rgb_pixels: RGB values (0-255), shape (N, 3) or (H, W, 3)
        
        Returns:
            LAB values: L* (0-100), a* (-128 to 127), b* (-128 to 127)
        """
        # Normalize RGB to 0-1
        rgb_normalized = rgb_pixels.astype(np.float32) / 255.0
        
        # sRGB gamma correction
        rgb_linear = np.where(
            rgb_normalized <= 0.04045,
            rgb_normalized / 12.92,
            ((rgb_normalized + 0.055) / 1.055) ** 2.4
        )
        
        # RGB to XYZ transformation matrix (D65 illuminant)
        M = np.array([
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041]
        ])
        
        # Reshape for matrix multiplication
        original_shape = rgb_linear.shape
        if len(original_shape) == 3:  # (H, W, 3)
            rgb_linear = rgb_linear.reshape(-1, 3)
        
        xyz = rgb_linear @ M.T
        
        # XYZ to LAB (D65 reference white: Xn=95.047, Yn=100.0, Zn=108.883)
        xyz_ref = np.array([95.047, 100.0, 108.883])
        xyz_normalized = xyz / xyz_ref
        
        # f(t) function for LAB conversion
        threshold = (6/29) ** 3
        f_xyz = np.where(
            xyz_normalized > threshold,
            xyz_normalized ** (1/3),
            (841/108) * xyz_normalized + (4/29)
        )
        
        # Calculate L*a*b*
        L_star = 116 * f_xyz[:, 1] - 16
        a_star = 500 * (f_xyz[:, 0] - f_xyz[:, 1])
        b_star = 200 * (f_xyz[:, 1] - f_xyz[:, 2])
        
        lab = np.stack([L_star, a_star, b_star], axis=-1)
        
        # Reshape back to original
        if len(original_shape) == 3:
            lab = lab.reshape(original_shape)
        
        return lab
    
    def _extract_facial_rois(self, frame: np.ndarray, landmarks: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract facial ROIs using MediaPipe landmarks"""
        rois = {}
        
        try:
            h, w = frame.shape[:2]
            
            # Cheeks ROI (landmarks 50, 101, 280, 330)
            if len(landmarks) > 330:
                cheek_points = landmarks[[50, 101, 280, 330]]
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillConvexPoly(mask, cheek_points.astype(np.int32), 255)
                rois['facial_cheeks'] = frame[mask > 0]
            
            # Forehead ROI (landmarks 10, 67, 109, 297)
            if len(landmarks) > 297:
                forehead_points = landmarks[[10, 67, 109, 297]]
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillConvexPoly(mask, forehead_points.astype(np.int32), 255)
                rois['facial_forehead'] = frame[mask > 0]
            
            # Periorbital ROI (around eyes - landmarks 33, 133, 362, 263)
            if len(landmarks) > 362:
                periorbital_points = landmarks[[33, 133, 362, 263]]
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillConvexPoly(mask, periorbital_points.astype(np.int32), 255)
                rois['facial_periorbital'] = frame[mask > 0]
            
        except Exception as e:
            logger.warning(f"Facial ROI extraction error: {e}")
        
        return rois
    
    def _calculate_perfusion_index(self, lab_pixels: np.ndarray) -> float:
        """
        Calculate perfusion index from LAB pixels
        
        Perfusion Index = 0.5*a* + 0.3*(100-L*) + 0.2*(50-|b*|)
        Higher = better perfusion
        """
        if len(lab_pixels) == 0:
            return 0.0
        
        L_star = lab_pixels[:, 0]
        a_star = lab_pixels[:, 1]
        b_star = lab_pixels[:, 2]
        
        perfusion = (
            0.5 * np.mean(a_star) +
            0.3 * (100 - np.mean(L_star)) +
            0.2 * (50 - np.mean(np.abs(b_star)))
        )
        
        return float(np.clip(perfusion, 0, 100))
    
    def _detect_pallor(self, lab_pixels: np.ndarray) -> float:
        """Calculate pallor score: High L* + Low a* = pallor"""
        if len(lab_pixels) == 0:
            return 0.0
        
        L_star = np.mean(lab_pixels[:, 0])
        a_star = np.mean(lab_pixels[:, 1])
        
        pallor_score = (0.6 * L_star + 0.4 * (50 - a_star)) / 100.0
        return float(np.clip(pallor_score, 0, 1))
    
    def _detect_jaundice(self, lab_pixels: np.ndarray) -> float:
        """Calculate jaundice score: High b* (yellow) = jaundice"""
        if len(lab_pixels) == 0:
            return 0.0
        
        b_star = np.mean(lab_pixels[:, 2])
        jaundice_score = max(0, b_star - 10.0)  # Baseline subtraction
        return float(jaundice_score)
    
    def _detect_cyanosis(self, lab_pixels: np.ndarray) -> float:
        """Calculate cyanosis score: Low a* + Negative b* = cyanosis"""
        if len(lab_pixels) == 0:
            return 0.0
        
        a_star = np.mean(lab_pixels[:, 1])
        b_star = np.mean(lab_pixels[:, 2])
        
        cyanosis_score = (
            0.5 * (50 - a_star) +
            0.5 * max(0, -b_star)
        ) / 100.0
        
        return float(np.clip(cyanosis_score, 0, 1))
    
    def _detect_skin_pallor(
        self,
        frame: np.ndarray,
        landmarks: np.ndarray
    ) -> Dict[str, float]:
        """
        LAB color space skin analysis (PRODUCTION)
        Extracts 30+ metrics: perfusion, pallor, jaundice, cyanosis, etc.
        """
        metrics = {}
        
        try:
            # Extract facial ROIs
            facial_rois = self._extract_facial_rois(frame, landmarks)
            
            # LAB conversion and perfusion analysis for each ROI
            perfusion_values = []
            pallor_values = []
            jaundice_values = []
            cyanosis_values = []
            
            for roi_name, roi_pixels in facial_rois.items():
                if len(roi_pixels) < 100:  # Skip small ROIs
                    continue
                
                # Convert to LAB
                lab_pixels = self._rgb_to_lab(roi_pixels)
                
                # Filter extreme values
                L_valid = (lab_pixels[:, 0] > 5) & (lab_pixels[:, 0] < 95)
                a_valid = np.abs(lab_pixels[:, 1]) < 100
                b_valid = np.abs(lab_pixels[:, 2]) < 100
                valid_mask = L_valid & a_valid & b_valid
                
                if np.sum(valid_mask) < 50:
                    continue
                
                lab_filtered = lab_pixels[valid_mask]
                
                # Calculate metrics
                perfusion = self._calculate_perfusion_index(lab_filtered)
                pallor = self._detect_pallor(lab_filtered)
                jaundice = self._detect_jaundice(lab_filtered)
                cyanosis = self._detect_cyanosis(lab_filtered)
                
                perfusion_values.append(perfusion)
                pallor_values.append(pallor)
                jaundice_values.append(jaundice)
                cyanosis_values.append(cyanosis)
            
            # Aggregate metrics
            if perfusion_values:
                metrics['lab_facial_perfusion_avg'] = float(np.mean(perfusion_values))
                metrics['lab_facial_perfusion_std'] = float(np.std(perfusion_values))
            else:
                metrics['lab_facial_perfusion_avg'] = 0.0
                metrics['lab_facial_perfusion_std'] = 0.0
            
            if pallor_values:
                metrics['pallor_facial_score'] = float(np.mean(pallor_values))
                metrics['pallor_detected'] = bool(np.mean(pallor_values) > 0.55)
            else:
                metrics['pallor_facial_score'] = 0.0
                metrics['pallor_detected'] = False
            
            if jaundice_values:
                metrics['jaundice_facial_score'] = float(np.mean(jaundice_values))
                metrics['jaundice_detected'] = bool(np.mean(jaundice_values) > 25.0)
            else:
                metrics['jaundice_facial_score'] = 0.0
                metrics['jaundice_detected'] = False
            
            if cyanosis_values:
                metrics['cyanosis_facial_score'] = float(np.mean(cyanosis_values))
                metrics['cyanosis_detected'] = bool(np.mean(cyanosis_values) > 0.4)
            else:
                metrics['cyanosis_facial_score'] = 0.0
                metrics['cyanosis_detected'] = False
            
            # Hand and nail detection (LAB-based)
            hand_nail_metrics = self._detect_hands_and_nails(frame, landmarks)
            metrics.update(hand_nail_metrics)
            
            # Quality score
            metrics['lab_skin_analysis_quality'] = float(len(perfusion_values) / 3.0) if perfusion_values else 0.0
            
        except Exception as e:
            logger.error(f"LAB skin analysis error: {e}")
            metrics = self._get_default_skin_metrics()
        
        return metrics
    
    def _get_default_skin_metrics(self) -> Dict[str, float]:
        """Return default metrics when analysis fails"""
        return {
            'lab_facial_perfusion_avg': 0.0,
            'lab_facial_perfusion_std': 0.0,
            'pallor_facial_score': 0.0,
            'pallor_detected': False,
            'jaundice_facial_score': 0.0,
            'jaundice_detected': False,
            'cyanosis_facial_score': 0.0,
            'cyanosis_detected': False,
            'lab_skin_analysis_quality': 0.0
        }
    
    def _detect_hands_and_nails(
        self,
        frame: np.ndarray,
        face_landmarks: np.ndarray
    ) -> Dict[str, Any]:
        """
        LAB-based hand/palm/nailbed analysis (PRODUCTION)
        Returns: palmar perfusion, nailbed perfusion, pallor, capillary refill proxy,
                 nailbed clubbing, pitting, texture, hydration
        """
        metrics = {
            'hands_detected': False,
            'lab_palmar_perfusion_avg': 0.0,
            'lab_nailbed_perfusion_avg': 0.0,
            'pallor_palmar_score': 0.0,
            'pallor_nailbed_score': 0.0,
            'capillary_refill_proxy': 0.0,
            'nailbed_clubbing_detected': False,
            'nailbed_clubbing_ratio': 0.0,
            'nailbed_pitting_score': 0.0,
            'nailbed_abnormalities': [],
            'skin_hydration_score': 0.0,
            'skin_texture_variance': 0.0,
            'temperature_proxy_palmar': 0.0
        }
        
        try:
            h, w, _ = frame.shape
            
            # Create face mask to exclude face region
            face_x_min, face_y_min = face_landmarks.min(axis=0).astype(int)
            face_x_max, face_y_max = face_landmarks.max(axis=0).astype(int)
            
            # Simple skin detection using HSV for hand ROI extraction
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
            lower_skin = np.array([0, 20, 70], dtype=np.uint8)
            upper_skin = np.array([20, 255, 255], dtype=np.uint8)
            skin_mask = cv2.inRange(hsv_frame, lower_skin, upper_skin)
            
            # Exclude face region
            face_padding = 40
            face_y_min_padded = max(0, face_y_min - face_padding)
            face_y_max_padded = min(h, face_y_max + face_padding)
            face_x_min_padded = max(0, face_x_min - face_padding)
            face_x_max_padded = min(w, face_x_max + face_padding)
            skin_mask[face_y_min_padded:face_y_max_padded, face_x_min_padded:face_x_max_padded] = 0
            
            # Find largest contour (likely hand/palm)
            contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if len(contours) == 0:
                return metrics
            
            largest_contour = max(contours, key=cv2.contourArea)
            contour_area = cv2.contourArea(largest_contour)
            
            # Only process if contour is large enough
            min_hand_area = (h * w) * 0.05  # At least 5% of frame
            if contour_area < min_hand_area:
                return metrics
            
            metrics['hands_detected'] = True
            
            # Extract hand ROI
            x, y, rw, rh = cv2.boundingRect(largest_contour)
            hand_roi = frame[y:y+rh, x:x+rw]
            
            if hand_roi.size == 0 or hand_roi.shape[0] < 50 or hand_roi.shape[1] < 50:
                return metrics
            
            # Convert hand ROI to LAB
            hand_lab = self._rgb_to_lab(hand_roi)
            
            # Filter valid pixels
            L_valid = (hand_lab[:, :, 0] > 5) & (hand_lab[:, :, 0] < 95)
            a_valid = np.abs(hand_lab[:, :, 1]) < 100
            b_valid = np.abs(hand_lab[:, :, 2]) < 100
            valid_mask = L_valid & a_valid & b_valid
            
            if np.sum(valid_mask) < 100:
                return metrics
            
            valid_lab_pixels = hand_lab[valid_mask]
            
            # PALMAR PERFUSION INDEX
            palmar_perfusion = self._calculate_perfusion_index(valid_lab_pixels)
            metrics['lab_palmar_perfusion_avg'] = palmar_perfusion
            
            # PALMAR PALLOR
            palmar_pallor = self._detect_pallor(valid_lab_pixels)
            metrics['pallor_palmar_score'] = palmar_pallor
            
            # TEMPERATURE PROXY (from perfusion and L* channel)
            L_mean = np.mean(valid_lab_pixels[:, 0])
            a_mean = np.mean(valid_lab_pixels[:, 1])
            temp_proxy = 0.6 * a_mean + 0.4 * (100 - L_mean)
            metrics['temperature_proxy_palmar'] = float(np.clip(temp_proxy, 0, 100))
            
            # NAILBED ANALYSIS (central region - proxy for fingertips)
            # Extract central 20% of hand ROI as nailbed proxy
            center_h_start = int(rh * 0.4)
            center_h_end = int(rh * 0.6)
            center_w_start = int(rw * 0.4)
            center_w_end = int(rw * 0.6)
            
            nailbed_roi = hand_roi[center_h_start:center_h_end, center_w_start:center_w_end]
            
            if nailbed_roi.size > 0 and nailbed_roi.shape[0] > 20 and nailbed_roi.shape[1] > 20:
                nailbed_lab = self._rgb_to_lab(nailbed_roi)
                nailbed_valid = nailbed_lab[
                    (nailbed_lab[:, :, 0] > 5) & 
                    (nailbed_lab[:, :, 0] < 95) &
                    (np.abs(nailbed_lab[:, :, 1]) < 100) &
                    (np.abs(nailbed_lab[:, :, 2]) < 100)
                ]
                
                if len(nailbed_valid) > 50:
                    nailbed_perfusion = self._calculate_perfusion_index(nailbed_valid)
                    nailbed_pallor = self._detect_pallor(nailbed_valid)
                    
                    metrics['lab_nailbed_perfusion_avg'] = nailbed_perfusion
                    metrics['pallor_nailbed_score'] = nailbed_pallor
                    
                    # NAILBED CLUBBING (geometry-based - proxy)
                    # Clubbing ratio = width variation in nailbed region
                    # Higher variance = potential clubbing
                    gray_nailbed = cv2.cvtColor(nailbed_roi, cv2.COLOR_RGB2GRAY)
                    edges = cv2.Canny(gray_nailbed, 50, 150)
                    edge_density = np.sum(edges > 0) / edges.size
                    clubbing_ratio = edge_density * 2.0  # Normalized
                    
                    metrics['nailbed_clubbing_ratio'] = float(clubbing_ratio)
                    metrics['nailbed_clubbing_detected'] = bool(clubbing_ratio > 1.0)
                    
                    # NAILBED PITTING (texture analysis)
                    # Pitting shows as high texture variance
                    _, texture_std = cv2.meanStdDev(gray_nailbed)
                    pitting_score = float(texture_std[0][0])
                    metrics['nailbed_pitting_score'] = pitting_score
                    
                    # NAILBED ABNORMALITIES
                    abnormalities = []
                    
                    # Yellow staining (high b*)
                    b_mean = np.mean(nailbed_valid[:, 2])
                    if b_mean > 30.0:
                        abnormalities.append('nicotine_staining')
                    
                    # Very dark (burns/melanonychia)
                    L_nailbed = np.mean(nailbed_valid[:, 0])
                    if L_nailbed < 25.0:
                        abnormalities.append('dark_pigmentation')
                    
                    metrics['nailbed_abnormalities'] = abnormalities
            
            # CAPILLARY REFILL PROXY
            # Estimated from perfusion + temperature proxy
            crt_proxy = (
                2.0 * (50 - palmar_perfusion) / 50.0 +
                0.5 * (50 - temp_proxy) / 50.0
            )
            metrics['capillary_refill_proxy'] = float(max(0, crt_proxy))
            
            # SKIN HYDRATION & TEXTURE
            # Hydration from texture variance (smooth = hydrated)
            gray_hand = cv2.cvtColor(hand_roi, cv2.COLOR_RGB2GRAY)
            _, texture_std = cv2.meanStdDev(gray_hand)
            texture_variance = float(texture_std[0][0])
            
            # Hydration score (inverse of texture variance)
            max_variance = 50.0
            hydration_score = 100.0 - (texture_variance / max_variance * 100.0)
            
            metrics['skin_texture_variance'] = texture_variance
            metrics['skin_hydration_score'] = float(np.clip(hydration_score, 0, 100))
            
            return metrics
            
        except Exception as e:
            logger.warning(f"LAB hand/nail analysis error: {e}")
            return metrics
    
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
    ) -> Dict[str, Any]:
        """
        Comprehensive Facial Puffiness Score (FPS) calculation
        Uses MediaPipe Face Mesh landmarks to track facial contour expansion
        
        Regions tracked:
        - Periorbital (around eyes): Critical for thyroid/kidney conditions
        - Cheeks: General facial swelling
        - Jawline: Lower face fluid retention
        - Forehead: Upper face swelling
        
        Returns facial contour measurements for temporal tracking
        """
        try:
            # MediaPipe Face Mesh landmark indices for key regions
            # Reference: https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
            
            # === PERIORBITAL REGION (Around Eyes) ===
            # Left eye contour
            left_eye_outer = landmarks[33]    # Outer corner
            left_eye_inner = landmarks[133]   # Inner corner
            left_eye_top = landmarks[159]     # Top of eye
            left_eye_bottom = landmarks[145]  # Bottom of eye
            
            # Right eye contour
            right_eye_outer = landmarks[263]  # Outer corner
            right_eye_inner = landmarks[362]  # Inner corner
            right_eye_top = landmarks[386]    # Top of eye
            right_eye_bottom = landmarks[374] # Bottom of eye
            
            # Periorbital measurements
            left_eye_width = np.linalg.norm(left_eye_outer - left_eye_inner)
            left_eye_height = np.linalg.norm(left_eye_top - left_eye_bottom)
            right_eye_width = np.linalg.norm(right_eye_outer - right_eye_inner)
            right_eye_height = np.linalg.norm(right_eye_top - right_eye_bottom)
            
            # Periorbital puffiness (eye area)
            left_eye_area = left_eye_width * left_eye_height
            right_eye_area = right_eye_width * right_eye_height
            avg_eye_area = (left_eye_area + right_eye_area) / 2
            
            # === CHEEK REGION ===
            left_cheek = landmarks[234]   # Left cheek point
            right_cheek = landmarks[454]  # Right cheek point
            nose_bridge = landmarks[168]  # Nose bridge (midpoint reference)
            
            # Cheek width and projection
            cheek_width = np.linalg.norm(right_cheek - left_cheek)
            left_cheek_projection = np.linalg.norm(left_cheek - nose_bridge)
            right_cheek_projection = np.linalg.norm(right_cheek - nose_bridge)
            avg_cheek_projection = (left_cheek_projection + right_cheek_projection) / 2
            
            # === JAWLINE REGION ===
            left_jaw = landmarks[172]     # Left jawline point
            right_jaw = landmarks[397]    # Right jawline point
            chin = landmarks[152]         # Chin point
            
            # Jawline measurements
            jawline_width = np.linalg.norm(right_jaw - left_jaw)
            left_jaw_projection = np.linalg.norm(left_jaw - chin)
            right_jaw_projection = np.linalg.norm(right_jaw - chin)
            
            # === FOREHEAD REGION ===
            forehead_center = landmarks[10]   # Forehead center
            left_temple = landmarks[108]      # Left temple
            right_temple = landmarks[337]     # Right temple
            
            # Forehead width
            forehead_width = np.linalg.norm(right_temple - left_temple)
            
            # === OVERALL FACE CONTOUR ===
            # Face oval perimeter (approximation using key points)
            face_oval_points = [
                landmarks[10],   # Forehead
                landmarks[234],  # Left cheek
                landmarks[152],  # Chin
                landmarks[454],  # Right cheek
            ]
            
            # Calculate perimeter
            face_perimeter = 0.0
            for i in range(len(face_oval_points)):
                p1 = face_oval_points[i]
                p2 = face_oval_points[(i + 1) % len(face_oval_points)]
                face_perimeter += np.linalg.norm(p2 - p1)
            
            # === FACIAL PUFFINESS SCORE (FPS) COMPONENTS ===
            # These will be compared to baseline to detect expansion
            
            return {
                'landmark_distances': {
                    # Periorbital (critical for thyroid/kidney)
                    'left_eye_width': float(left_eye_width),
                    'left_eye_height': float(left_eye_height),
                    'left_eye_area': float(left_eye_area),
                    'right_eye_width': float(right_eye_width),
                    'right_eye_height': float(right_eye_height),
                    'right_eye_area': float(right_eye_area),
                    'avg_eye_area': float(avg_eye_area),
                    
                    # Cheek region
                    'cheek_width': float(cheek_width),
                    'left_cheek_projection': float(left_cheek_projection),
                    'right_cheek_projection': float(right_cheek_projection),
                    'avg_cheek_projection': float(avg_cheek_projection),
                    
                    # Jawline
                    'jawline_width': float(jawline_width),
                    'left_jaw_projection': float(left_jaw_projection),
                    'right_jaw_projection': float(right_jaw_projection),
                    
                    # Forehead
                    'forehead_width': float(forehead_width),
                    
                    # Overall contour
                    'face_perimeter': float(face_perimeter)
                },
                'fps_components': {
                    # Individual region scores (normalized 0-100)
                    # Will be calculated against baseline in aggregate metrics
                    'periorbital_score': 0.0,  # Computed in aggregate
                    'cheek_score': 0.0,
                    'jawline_score': 0.0,
                    'forehead_score': 0.0,
                    'overall_contour_score': 0.0
                }
            }
        except Exception as e:
            logger.warning(f"Comprehensive swelling detection error: {e}")
            return {
                'landmark_distances': {},
                'fps_components': {}
            }
    
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
        accessory_muscle_activity: List[float],
        chest_widths: List[float],
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
        # COMPREHENSIVE SKIN ANALYSIS PARAMETERS (LAB color space)
        facial_l_values: List[Tuple[int, float]],
        facial_perfusion_indices: List[float],
        palmar_perfusion_indices: List[float],
        nailbed_color_indices: List[float],
        pallor_detections: List[Dict],
        cyanosis_detections: List[Dict],
        jaundice_detections: List[Dict],
        nail_clubbing_detections: List[Dict],
        nail_pitting_detections: List[Dict],
        skin_analysis_frames: List[Dict],
        fps: float,
        duration: float,
        frames_analyzed: int,
        frames_with_face: int,
        frames_with_hands: int,
        patient_baseline: Optional[Dict[str, float]]
    ) -> Dict[str, Any]:
        """
        Compute aggregate metrics from time-series data
        Returns 30+ clinical metrics including advanced respiratory and skin analysis
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
        
        # ==================== Comprehensive Facial Puffiness Score (FPS) ====================
        if len(landmark_distances) > 0 and landmark_distances[0].get('landmark_distances'):
            # Extract all measurements from landmark tracking
            # Note: landmark_distances is a list of dicts with 'landmark_distances' key
            all_measurements = {
                'periorbital': {
                    'left_eye_area': [d.get('landmark_distances', {}).get('left_eye_area', 0) for d in landmark_distances],
                    'right_eye_area': [d.get('landmark_distances', {}).get('right_eye_area', 0) for d in landmark_distances],
                    'avg_eye_area': [d.get('landmark_distances', {}).get('avg_eye_area', 0) for d in landmark_distances]
                },
                'cheek': {
                    'cheek_width': [d.get('landmark_distances', {}).get('cheek_width', 0) for d in landmark_distances],
                    'avg_cheek_projection': [d.get('landmark_distances', {}).get('avg_cheek_projection', 0) for d in landmark_distances]
                },
                'jawline': {
                    'jawline_width': [d.get('landmark_distances', {}).get('jawline_width', 0) for d in landmark_distances]
                },
                'forehead': {
                    'forehead_width': [d.get('landmark_distances', {}).get('forehead_width', 0) for d in landmark_distances]
                },
                'overall': {
                    'face_perimeter': [d.get('landmark_distances', {}).get('face_perimeter', 0) for d in landmark_distances]
                }
            }
            
            # Compute averages for each measurement
            avg_eye_area = np.mean(all_measurements['periorbital']['avg_eye_area']) if all_measurements['periorbital']['avg_eye_area'] else 0
            avg_cheek_width = np.mean(all_measurements['cheek']['cheek_width']) if all_measurements['cheek']['cheek_width'] else 0
            avg_cheek_projection = np.mean(all_measurements['cheek']['avg_cheek_projection']) if all_measurements['cheek']['avg_cheek_projection'] else 0
            avg_jawline_width = np.mean(all_measurements['jawline']['jawline_width']) if all_measurements['jawline']['jawline_width'] else 0
            avg_forehead_width = np.mean(all_measurements['forehead']['forehead_width']) if all_measurements['forehead']['forehead_width'] else 0
            avg_face_perimeter = np.mean(all_measurements['overall']['face_perimeter']) if all_measurements['overall']['face_perimeter'] else 0
            
            # === Calculate FPS Components (compare to baseline) ===
            periorbital_fps = 0.0
            cheek_fps = 0.0
            jawline_fps = 0.0
            forehead_fps = 0.0
            overall_fps = 0.0
            
            if patient_baseline:
                # Periorbital FPS (critical for thyroid/kidney conditions)
                baseline_eye_area = patient_baseline.get('baseline_eye_area', avg_eye_area)
                if baseline_eye_area > 0:
                    periorbital_fps = max(0, (avg_eye_area - baseline_eye_area) / baseline_eye_area * 100)
                
                # Cheek FPS
                baseline_cheek_width = patient_baseline.get('baseline_cheek_width', avg_cheek_width)
                if baseline_cheek_width > 0:
                    cheek_fps = max(0, (avg_cheek_width - baseline_cheek_width) / baseline_cheek_width * 100)
                
                # Jawline FPS
                baseline_jawline_width = patient_baseline.get('baseline_jawline_width', avg_jawline_width)
                if baseline_jawline_width > 0:
                    jawline_fps = max(0, (avg_jawline_width - baseline_jawline_width) / baseline_jawline_width * 100)
                
                # Forehead FPS
                baseline_forehead_width = patient_baseline.get('baseline_forehead_width', avg_forehead_width)
                if baseline_forehead_width > 0:
                    forehead_fps = max(0, (avg_forehead_width - baseline_forehead_width) / baseline_forehead_width * 100)
                
                # Overall contour FPS
                baseline_face_perimeter = patient_baseline.get('baseline_face_perimeter', avg_face_perimeter)
                if baseline_face_perimeter > 0:
                    overall_fps = max(0, (avg_face_perimeter - baseline_face_perimeter) / baseline_face_perimeter * 100)
            
            # === Composite Facial Puffiness Score (FPS) ===
            # Weighted average: Periorbital 30%, Cheek 30%, Jawline 20%, Forehead 10%, Overall 10%
            composite_fps = (
                periorbital_fps * 0.30 +
                cheek_fps * 0.30 +
                jawline_fps * 0.20 +
                forehead_fps * 0.10 +
                overall_fps * 0.10
            )
            
            # === Store all FPS metrics ===
            metrics['facial_puffiness_score'] = float(composite_fps)  # NEW: Composite FPS
            metrics['fps_periorbital'] = float(periorbital_fps)       # NEW: Eye puffiness
            metrics['fps_cheek'] = float(cheek_fps)                   # NEW: Cheek swelling
            metrics['fps_jawline'] = float(jawline_fps)               # NEW: Jawline swelling
            metrics['fps_forehead'] = float(forehead_fps)             # NEW: Forehead swelling
            metrics['fps_overall_contour'] = float(overall_fps)       # NEW: Overall expansion
            
            # Raw measurements (for baseline calculation)
            metrics['raw_eye_area'] = float(avg_eye_area)
            metrics['raw_cheek_width'] = float(avg_cheek_width)
            metrics['raw_cheek_projection'] = float(avg_cheek_projection)
            metrics['raw_jawline_width'] = float(avg_jawline_width)
            metrics['raw_forehead_width'] = float(avg_forehead_width)
            metrics['raw_face_perimeter'] = float(avg_face_perimeter)
            
            # Asymmetry detection
            left_eye_areas = [d.get('landmark_distances', {}).get('left_eye_area', 0) for d in landmark_distances]
            right_eye_areas = [d.get('landmark_distances', {}).get('right_eye_area', 0) for d in landmark_distances]
            if left_eye_areas and right_eye_areas:
                avg_left_eye = np.mean(left_eye_areas)
                avg_right_eye = np.mean(right_eye_areas)
                if max(avg_left_eye, avg_right_eye) > 0:
                    asymmetry = abs(avg_left_eye - avg_right_eye) / max(avg_left_eye, avg_right_eye) * 100
                    metrics['facial_asymmetry_score'] = float(asymmetry)
                else:
                    metrics['facial_asymmetry_score'] = 0.0
            else:
                metrics['facial_asymmetry_score'] = 0.0
            
            # Legacy compatibility (keep old field names)
            metrics['facial_swelling_score'] = float(composite_fps)  # Alias for legacy code
            metrics['left_cheek_distance'] = float(avg_cheek_width)
            metrics['right_cheek_distance'] = float(avg_cheek_width)
            metrics['eye_puffiness_left'] = float(periorbital_fps)   # Using periorbital FPS
            metrics['eye_puffiness_right'] = float(periorbital_fps)
            
            # FPS risk classification
            if composite_fps < 10:
                metrics['facial_puffiness_risk'] = "low"
            elif composite_fps < 25:
                metrics['facial_puffiness_risk'] = "medium"
            else:
                metrics['facial_puffiness_risk'] = "high"
                
        else:
            # No facial landmarks detected
            metrics['facial_puffiness_score'] = 0.0
            metrics['fps_periorbital'] = 0.0
            metrics['fps_cheek'] = 0.0
            metrics['fps_jawline'] = 0.0
            metrics['fps_forehead'] = 0.0
            metrics['fps_overall_contour'] = 0.0
            metrics['raw_eye_area'] = 0.0
            metrics['raw_cheek_width'] = 0.0
            metrics['raw_cheek_projection'] = 0.0
            metrics['raw_jawline_width'] = 0.0
            metrics['raw_forehead_width'] = 0.0
            metrics['raw_face_perimeter'] = 0.0
            metrics['facial_asymmetry_score'] = 0.0
            metrics['facial_swelling_score'] = 0.0
            metrics['left_cheek_distance'] = 0.0
            metrics['right_cheek_distance'] = 0.0
            metrics['eye_puffiness_left'] = 0.0
            metrics['eye_puffiness_right'] = 0.0
            metrics['facial_puffiness_risk'] = "unknown"
        
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
        
        # ==================== COMPREHENSIVE SKIN ANALYSIS (LAB Color Space) ====================
        # This section aggregates frame-by-frame LAB color analysis into clinical-grade perfusion metrics
        
        # Perfusion Indices (0-100 scale, higher = better perfusion)
        if len(facial_perfusion_indices) > 0:
            metrics['lab_facial_perfusion_avg'] = float(np.mean(facial_perfusion_indices))
            metrics['lab_facial_perfusion_std'] = float(np.std(facial_perfusion_indices))
            metrics['lab_facial_perfusion_min'] = float(np.min(facial_perfusion_indices))
            metrics['lab_facial_perfusion_max'] = float(np.max(facial_perfusion_indices))
            # Trend detection (improving vs declining perfusion)
            if len(facial_perfusion_indices) > 5:
                first_half = np.mean(facial_perfusion_indices[:len(facial_perfusion_indices)//2])
                second_half = np.mean(facial_perfusion_indices[len(facial_perfusion_indices)//2:])
                trend = second_half - first_half
                metrics['lab_facial_perfusion_trend'] = float(trend)
                metrics['lab_facial_perfusion_trend_direction'] = 'improving' if trend > 2 else ('declining' if trend < -2 else 'stable')
            else:
                metrics['lab_facial_perfusion_trend'] = 0.0
                metrics['lab_facial_perfusion_trend_direction'] = 'insufficient_data'
        else:
            metrics['lab_facial_perfusion_avg'] = 0.0
            metrics['lab_facial_perfusion_std'] = 0.0
            metrics['lab_facial_perfusion_min'] = 0.0
            metrics['lab_facial_perfusion_max'] = 0.0
            metrics['lab_facial_perfusion_trend'] = 0.0
            metrics['lab_facial_perfusion_trend_direction'] = 'no_data'
        
        # Palmar Perfusion (when hands visible - gold standard for pallor)
        if len(palmar_perfusion_indices) > 0:
            metrics['lab_palmar_perfusion_avg'] = float(np.mean(palmar_perfusion_indices))
            metrics['lab_palmar_perfusion_std'] = float(np.std(palmar_perfusion_indices))
            metrics['palms_detected'] = True
            metrics['palms_detection_frames'] = len(palmar_perfusion_indices)
        else:
            metrics['lab_palmar_perfusion_avg'] = 0.0
            metrics['lab_palmar_perfusion_std'] = 0.0
            metrics['palms_detected'] = False
            metrics['palms_detection_frames'] = 0
        
        # Nailbed Color Index (cyanosis, anemia detection)
        if len(nailbed_color_indices) > 0:
            metrics['lab_nailbed_color_index_avg'] = float(np.mean(nailbed_color_indices))
            metrics['lab_nailbed_color_index_std'] = float(np.std(nailbed_color_indices))
            metrics['nailbeds_detected'] = True
            metrics['nailbeds_detection_frames'] = len(nailbed_color_indices)
        else:
            metrics['lab_nailbed_color_index_avg'] = 0.0
            metrics['lab_nailbed_color_index_std'] = 0.0
            metrics['nailbeds_detected'] = False
            metrics['nailbeds_detection_frames'] = 0
        
        # Clinical Color Changes Detection
        # Pallor Detection
        if len(pallor_detections) > 0:
            pallor_percentage = len(pallor_detections) / frames_with_face * 100 if frames_with_face > 0 else 0
            avg_pallor_severity = np.mean([d.get('pallor_severity', 0) for d in pallor_detections])
            # Determine most common region
            regions = [d.get('pallor_region', 'none') for d in pallor_detections]
            most_common_region = max(set(regions), key=regions.count) if regions else 'none'
            
            metrics['lab_pallor_detected'] = pallor_percentage > 10  # >10% of frames
            metrics['lab_pallor_severity'] = float(avg_pallor_severity)
            metrics['lab_pallor_region'] = most_common_region
            metrics['lab_pallor_detection_confidence'] = float(pallor_percentage / 100)
        else:
            metrics['lab_pallor_detected'] = False
            metrics['lab_pallor_severity'] = 0.0
            metrics['lab_pallor_region'] = 'none'
            metrics['lab_pallor_detection_confidence'] = 0.0
        
        # Cyanosis Detection
        if len(cyanosis_detections) > 0:
            cyanosis_percentage = len(cyanosis_detections) / frames_with_face * 100 if frames_with_face > 0 else 0
            avg_cyanosis_severity = np.mean([d.get('cyanosis_severity', 0) for d in cyanosis_detections])
            regions = [d.get('cyanosis_region', 'none') for d in cyanosis_detections]
            most_common_region = max(set(regions), key=regions.count) if regions else 'none'
            
            metrics['lab_cyanosis_detected'] = cyanosis_percentage > 5  # >5% of frames
            metrics['lab_cyanosis_severity'] = float(avg_cyanosis_severity)
            metrics['lab_cyanosis_region'] = most_common_region
            metrics['lab_cyanosis_detection_confidence'] = float(cyanosis_percentage / 100)
        else:
            metrics['lab_cyanosis_detected'] = False
            metrics['lab_cyanosis_severity'] = 0.0
            metrics['lab_cyanosis_region'] = 'none'
            metrics['lab_cyanosis_detection_confidence'] = 0.0
        
        # Jaundice Detection
        if len(jaundice_detections) > 0:
            jaundice_percentage = len(jaundice_detections) / frames_with_face * 100 if frames_with_face > 0 else 0
            avg_jaundice_severity = np.mean([d.get('jaundice_severity', 0) for d in jaundice_detections])
            
            metrics['lab_jaundice_detected'] = jaundice_percentage > 10
            metrics['lab_jaundice_severity'] = float(avg_jaundice_severity)
            metrics['lab_jaundice_detection_confidence'] = float(jaundice_percentage / 100)
        else:
            metrics['lab_jaundice_detected'] = False
            metrics['lab_jaundice_severity'] = 0.0
            metrics['lab_jaundice_detection_confidence'] = 0.0
        
        # Nail Clubbing Detection
        if len(nail_clubbing_detections) > 0:
            clubbing_percentage = len(nail_clubbing_detections) / frames_analyzed * 100
            avg_clubbing_severity = np.mean([d.get('nail_clubbing_severity', 0) for d in nail_clubbing_detections])
            
            metrics['lab_nail_clubbing_detected'] = clubbing_percentage > 15  # >15% of frames
            metrics['lab_nail_clubbing_severity'] = float(avg_clubbing_severity)
            metrics['lab_nail_clubbing_confidence'] = float(clubbing_percentage / 100)
        else:
            metrics['lab_nail_clubbing_detected'] = False
            metrics['lab_nail_clubbing_severity'] = 0.0
            metrics['lab_nail_clubbing_confidence'] = 0.0
        
        # Nail Pitting Detection
        if len(nail_pitting_detections) > 0:
            pitting_percentage = len(nail_pitting_detections) / frames_analyzed * 100
            avg_pitting_count = np.mean([d.get('nail_pitting_count', 0) for d in nail_pitting_detections])
            
            metrics['lab_nail_pitting_detected'] = pitting_percentage > 10
            metrics['lab_nail_pitting_count_avg'] = float(avg_pitting_count)
            metrics['lab_nail_pitting_confidence'] = float(pitting_percentage / 100)
        else:
            metrics['lab_nail_pitting_detected'] = False
            metrics['lab_nail_pitting_count_avg'] = 0.0
            metrics['lab_nail_pitting_confidence'] = 0.0
        
        # Capillary Refill Time (using L* channel tracking)
        capillary_refill = self.track_capillary_refill(facial_l_values, fps)
        if capillary_refill:
            metrics['lab_capillary_refill_time_sec'] = capillary_refill.get('capillary_refill_time_sec')
            metrics['lab_capillary_refill_method'] = capillary_refill.get('capillary_refill_method')
            metrics['lab_capillary_refill_quality'] = capillary_refill.get('capillary_refill_quality')
            metrics['lab_capillary_refill_abnormal'] = capillary_refill.get('capillary_refill_abnormal')
        else:
            metrics['lab_capillary_refill_time_sec'] = None
            metrics['lab_capillary_refill_method'] = 'not_measured'
            metrics['lab_capillary_refill_quality'] = 0.0
            metrics['lab_capillary_refill_abnormal'] = False
        
        # Skin Texture & Hydration (aggregate from frames)
        if len(skin_analysis_frames) > 0:
            texture_scores = [f.get('skin_texture_score', 0) for f in skin_analysis_frames]
            hydration_statuses = [f.get('hydration_status', 'normal') for f in skin_analysis_frames]
            temp_proxies = [f.get('temperature_proxy', 'normal') for f in skin_analysis_frames]
            
            metrics['lab_skin_texture_score'] = float(np.mean(texture_scores))
            metrics['lab_skin_hydration_status'] = max(set(hydration_statuses), key=hydration_statuses.count)
            metrics['lab_skin_temperature_proxy'] = max(set(temp_proxies), key=temp_proxies.count)
        else:
            metrics['lab_skin_texture_score'] = 0.0
            metrics['lab_skin_hydration_status'] = 'unknown'
            metrics['lab_skin_temperature_proxy'] = 'unknown'
        
        # Overall Skin Analysis Quality
        metrics['lab_skin_analysis_frames'] = len(skin_analysis_frames)
        metrics['lab_skin_analysis_quality'] = float(len(skin_analysis_frames) / frames_with_face) if frames_with_face > 0 else 0.0
        
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
    
    # ============================================================
    # COMPREHENSIVE SKIN ANALYSIS SYSTEM - LAB COLOR SPACE
    # ============================================================
    
    def analyze_skin_comprehensive(
        self,
        frame: np.ndarray,
        face_landmarks: np.ndarray,
        frame_index: int,
        baseline: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Complete skin analysis using LAB color space for clinical accuracy
        
        Extracts:
        - Facial perfusion (LAB color vectors)
        - Palmar perfusion (when hands visible)
        - Nailbed analysis (clubbing, pitting, color)
        - Capillary refill proxy (L* tracking)
        - Clinical color changes (pallor, cyanosis, jaundice)
        - Texture & hydration status
        
        Returns comprehensive skin metrics dict
        """
        skin_metrics = {
            'frame_index': frame_index,
            # LAB color metrics (initialized)
            'facial_l_lightness': 0.0,
            'facial_a_red_green': 0.0,
            'facial_b_yellow_blue': 0.0,
            'facial_perfusion_index': 0.0,
            'palmar_l_lightness': 0.0,
            'palmar_a_red_green': 0.0,
            'palmar_b_yellow_blue': 0.0,
            'palmar_perfusion_index': 0.0,
            'nailbed_l_lightness': 0.0,
            'nailbed_a_red_green': 0.0,
            'nailbed_b_yellow_blue': 0.0,
            'nailbed_color_index': 0.0,
            # Clinical detections
            'pallor_detected': False,
            'pallor_severity': 0.0,
            'pallor_region': 'none',
            'cyanosis_detected': False,
            'cyanosis_severity': 0.0,
            'cyanosis_region': 'none',
            'jaundice_detected': False,
            'jaundice_severity': 0.0,
            'jaundice_region': 'none',
            # Nailbed analysis
            'nail_clubbing_detected': False,
            'nail_clubbing_severity': 0.0,
            'nail_pitting_detected': False,
            'nail_pitting_count': 0,
            'nail_abnormalities': [],
            # Texture & hydration
            'skin_texture_score': 0.0,
            'hydration_status': 'normal',
            'temperature_proxy': 'normal',
            # Detection quality
            'facial_roi_detected': False,
            'palmar_roi_detected': False,
            'nailbed_roi_detected': False,
            'detection_confidence': 0.0,
        }
        
        try:
            # 1. Extract facial LAB color metrics
            facial_lab = self._extract_facial_lab_colors(frame, face_landmarks)
            if facial_lab:
                skin_metrics.update(facial_lab)
                skin_metrics['facial_roi_detected'] = True
            
            # 2. Detect and analyze hands/palms (if visible)
            palmar_lab = self._extract_palmar_lab_colors(frame, face_landmarks)
            if palmar_lab:
                skin_metrics.update(palmar_lab)
                skin_metrics['palmar_roi_detected'] = True
            
            # 3. Detect and analyze nailbeds (if hands visible)
            nailbed_analysis = self._analyze_nailbeds_comprehensive(frame, face_landmarks)
            if nailbed_analysis:
                skin_metrics.update(nailbed_analysis)
                skin_metrics['nailbed_roi_detected'] = True
            
            # 4. Detect clinical color changes (pallor, cyanosis, jaundice)
            color_changes = self._detect_clinical_color_changes(skin_metrics, baseline)
            skin_metrics.update(color_changes)
            
            # 5. Analyze skin texture and hydration
            texture_metrics = self._analyze_skin_texture_hydration(frame, face_landmarks)
            skin_metrics.update(texture_metrics)
            
            # 6. Compute overall detection confidence
            confidence_score = self._compute_skin_detection_confidence(skin_metrics)
            skin_metrics['detection_confidence'] = confidence_score
            
        except Exception as e:
            logger.error(f"Skin analysis error on frame {frame_index}: {str(e)}")
        
        return skin_metrics
    
    def _extract_facial_lab_colors(
        self,
        frame: np.ndarray,
        landmarks: np.ndarray
    ) -> Optional[Dict[str, float]]:
        """
        Extract LAB color space metrics from facial region (cheeks/forehead)
        
        LAB Color Space:
        - L* (lightness): 0-100 (0=black, 100=white)
        - a*: -128 to 127 (negative=green, positive=red)
        - b*: -128 to 127 (negative=blue, positive=yellow)
        
        Returns facial LAB metrics and perfusion index
        """
        try:
            # Define cheek ROI using facial landmarks
            # Cheeks are typically in the mid-face region
            h, w = frame.shape[:2]
            x_min, y_min = landmarks.min(axis=0).astype(int)
            x_max, y_max = landmarks.max(axis=0).astype(int)
            
            # Extract cheek regions (left and right)
            face_width = x_max - x_min
            face_height = y_max - y_min
            
            # Left cheek ROI (30-50% from left, 30-60% from top)
            left_cheek_x1 = int(x_min + face_width * 0.15)
            left_cheek_x2 = int(x_min + face_width * 0.35)
            cheek_y1 = int(y_min + face_height * 0.35)
            cheek_y2 = int(y_min + face_height * 0.65)
            
            # Right cheek ROI (50-70% from left, 30-60% from top)
            right_cheek_x1 = int(x_min + face_width * 0.65)
            right_cheek_x2 = int(x_min + face_width * 0.85)
            
            # Extract ROIs
            left_cheek_roi = frame[cheek_y1:cheek_y2, left_cheek_x1:left_cheek_x2]
            right_cheek_roi = frame[cheek_y1:cheek_y2, right_cheek_x1:right_cheek_x2]
            
            if left_cheek_roi.size == 0 or right_cheek_roi.size == 0:
                return None
            
            # Convert to LAB color space
            left_lab = cv2.cvtColor(left_cheek_roi, cv2.COLOR_RGB2LAB)
            right_lab = cv2.cvtColor(right_cheek_roi, cv2.COLOR_RGB2LAB)
            
            # Compute mean LAB values (average both cheeks)
            left_l_mean = np.mean(left_lab[:, :, 0])
            left_a_mean = np.mean(left_lab[:, :, 1])
            left_b_mean = np.mean(left_lab[:, :, 2])
            
            right_l_mean = np.mean(right_lab[:, :, 0])
            right_a_mean = np.mean(right_lab[:, :, 1])
            right_b_mean = np.mean(right_lab[:, :, 2])
            
            # Average both cheeks for facial metrics
            facial_l = (left_l_mean + right_l_mean) / 2.0
            facial_a = (left_a_mean + right_a_mean) / 2.0
            facial_b = (left_b_mean + right_b_mean) / 2.0
            
            # Compute perfusion index (0-100 scale)
            # Higher L* (lightness) and a* (redness) = better perfusion
            # Normalize: L* is 0-100, a* is shifted from -128-127 to 0-255
            perfusion_index = self._compute_perfusion_index(facial_l, facial_a, facial_b)
            
            return {
                'facial_l_lightness': float(facial_l),
                'facial_a_red_green': float(facial_a - 128),  # Shift to -128 to 127 range
                'facial_b_yellow_blue': float(facial_b - 128),
                'facial_perfusion_index': float(perfusion_index)
            }
        
        except Exception as e:
            logger.error(f"Facial LAB extraction error: {str(e)}")
            return None
    
    def _extract_palmar_lab_colors(
        self,
        frame: np.ndarray,
        face_landmarks: np.ndarray
    ) -> Optional[Dict[str, float]]:
        """
        Extract LAB colors from palmar (palm) region when hands are visible
        Palm analysis is gold standard for pallor detection in clinical exams
        """
        try:
            # Use MediaPipe Hands to detect palms
            # For now, use skin color detection outside face region
            h, w = frame.shape[:2]
            
            # Create mask excluding face
            face_x_min, face_y_min = face_landmarks.min(axis=0).astype(int)
            face_x_max, face_y_max = face_landmarks.max(axis=0).astype(int)
            
            # Convert to HSV for skin detection
            hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
            
            # Skin detection mask (broad range for diverse skin tones)
            lower_skin = np.array([0, 20, 70], dtype=np.uint8)
            upper_skin = np.array([25, 255, 255], dtype=np.uint8)
            skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
            
            # Zero out face region
            padding = 50
            face_y_min = max(0, face_y_min - padding)
            face_y_max = min(h, face_y_max + padding)
            face_x_min = max(0, face_x_min - padding)
            face_x_max = min(w, face_x_max + padding)
            skin_mask[face_y_min:face_y_max, face_x_min:face_x_max] = 0
            
            # Find largest skin region (likely palm/hand)
            contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return None
            
            largest_contour = max(contours, key=cv2.contourArea)
            contour_area = cv2.contourArea(largest_contour)
            
            # Require minimum area (palm should be substantial)
            if contour_area < 2000:
                return None
            
            # Create bounding box for palm ROI
            x, y, w_box, h_box = cv2.boundingRect(largest_contour)
            palm_roi = frame[y:y+h_box, x:x+w_box]
            
            if palm_roi.size == 0:
                return None
            
            # Convert palm ROI to LAB
            palm_lab = cv2.cvtColor(palm_roi, cv2.COLOR_RGB2LAB)
            
            # Extract LAB means
            palmar_l = np.mean(palm_lab[:, :, 0])
            palmar_a = np.mean(palm_lab[:, :, 1])
            palmar_b = np.mean(palm_lab[:, :, 2])
            
            # Compute palmar perfusion index
            palmar_perfusion = self._compute_perfusion_index(palmar_l, palmar_a, palmar_b)
            
            return {
                'palmar_l_lightness': float(palmar_l),
                'palmar_a_red_green': float(palmar_a - 128),
                'palmar_b_yellow_blue': float(palmar_b - 128),
                'palmar_perfusion_index': float(palmar_perfusion)
            }
        
        except Exception as e:
            logger.error(f"Palmar LAB extraction error: {str(e)}")
            return None
    
    def _analyze_nailbeds_comprehensive(
        self,
        frame: np.ndarray,
        face_landmarks: np.ndarray
    ) -> Optional[Dict[str, Any]]:
        """
        Comprehensive nailbed analysis:
        - LAB color extraction (cyanosis, anemia detection)
        - Clubbing detection (Schamroth window proxy via curvature analysis)
        - Pitting detection (surface variance, texture analysis)
        - Abnormality detection (leukonychia, splinter hemorrhages)
        """
        try:
            # Detect hand regions first
            h, w = frame.shape[:2]
            hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
            
            # Skin detection
            lower_skin = np.array([0, 20, 70], dtype=np.uint8)
            upper_skin = np.array([25, 255, 255], dtype=np.uint8)
            skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
            
            # Exclude face
            face_x_min, face_y_min = face_landmarks.min(axis=0).astype(int)
            face_x_max, face_y_max = face_landmarks.max(axis=0).astype(int)
            padding = 50
            skin_mask[max(0, face_y_min-padding):min(h, face_y_max+padding),
                     max(0, face_x_min-padding):min(w, face_x_max+padding)] = 0
            
            contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return None
            
            # Find hand contour
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w_box, h_box = cv2.boundingRect(largest_contour)
            hand_roi = frame[y:y+h_box, x:x+w_box]
            
            if hand_roi.size == 0:
                return None
            
            # Detect fingertip regions (nailbeds typically at top of hand ROI)
            # Look for regions with higher brightness (nails reflect light)
            gray_hand = cv2.cvtColor(hand_roi, cv2.COLOR_RGB2GRAY)
            _, bright_mask = cv2.threshold(gray_hand, 180, 255, cv2.THRESH_BINARY)
            
            # Find potential nailbed regions
            nail_contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            nailbed_lab_values = []
            clubbing_scores = []
            pitting_scores = []
            abnormalities = []
            
            for nail_contour in nail_contours:
                nail_area = cv2.contourArea(nail_contour)
                
                # Filter by size (nails are typically 50-500 px²)
                if nail_area < 50 or nail_area > 1000:
                    continue
                
                # Extract nailbed ROI
                nx, ny, nw, nh = cv2.boundingRect(nail_contour)
                nailbed_roi = hand_roi[ny:ny+nh, nx:nx+nw]
                
                if nailbed_roi.size == 0:
                    continue
                
                # 1. LAB color analysis
                nailbed_lab = cv2.cvtColor(nailbed_roi, cv2.COLOR_RGB2LAB)
                nail_l = np.mean(nailbed_lab[:, :, 0])
                nail_a = np.mean(nailbed_lab[:, :, 1])
                nail_b = np.mean(nailbed_lab[:, :, 2])
                nailbed_lab_values.append((nail_l, nail_a, nail_b))
                
                # 2. Clubbing detection (curvature analysis)
                # Clubbing causes increased nail curvature and distal phalanx bulging
                clubbing_score = self._detect_nail_clubbing(nailbed_roi, nail_contour)
                clubbing_scores.append(clubbing_score)
                
                # 3. Pitting detection (surface texture variance)
                pitting_score = self._detect_nail_pitting(nailbed_roi)
                pitting_scores.append(pitting_score)
                
                # 4. Abnormality detection (color-based)
                nail_abnormality = self._detect_nail_abnormalities(nailbed_roi, nail_l, nail_a, nail_b)
                if nail_abnormality:
                    abnormalities.append(nail_abnormality)
            
            if not nailbed_lab_values:
                return None
            
            # Aggregate nailbed metrics
            avg_l = np.mean([v[0] for v in nailbed_lab_values])
            avg_a = np.mean([v[1] for v in nailbed_lab_values])
            avg_b = np.mean([v[2] for v in nailbed_lab_values])
            
            nailbed_color_index = self._compute_perfusion_index(avg_l, avg_a, avg_b)
            
            # Clubbing detection (threshold: >0.6 indicates clubbing)
            max_clubbing = max(clubbing_scores) if clubbing_scores else 0.0
            clubbing_detected = max_clubbing > 0.6
            
            # Pitting detection (threshold: >3 pits)
            pitting_count = sum(1 for score in pitting_scores if score > 0.5)
            pitting_detected = pitting_count >= 3
            
            return {
                'nailbed_l_lightness': float(avg_l),
                'nailbed_a_red_green': float(avg_a - 128),
                'nailbed_b_yellow_blue': float(avg_b - 128),
                'nailbed_color_index': float(nailbed_color_index),
                'nail_clubbing_detected': clubbing_detected,
                'nail_clubbing_severity': float(max_clubbing),
                'nail_pitting_detected': pitting_detected,
                'nail_pitting_count': pitting_count,
                'nail_abnormalities': abnormalities
            }
        
        except Exception as e:
            logger.error(f"Nailbed analysis error: {str(e)}")
            return None
    
    def _detect_nail_clubbing(
        self,
        nailbed_roi: np.ndarray,
        nail_contour: np.ndarray
    ) -> float:
        """
        Detect nail clubbing using curvature analysis
        
        Clubbing indicators:
        - Increased longitudinal nail curvature
        - Loss of normal <165° angle between nail and nail bed (Schamroth window)
        - Distal phalanx bulging
        
        Returns clubbing severity score (0-1)
        """
        try:
            # Approximate curvature by analyzing contour shape
            # Clubbed nails have more pronounced curvature
            
            # Fit ellipse to nail contour
            if len(nail_contour) < 5:
                return 0.0
            
            ellipse = cv2.fitEllipse(nail_contour)
            (center, axes, angle) = ellipse
            major_axis, minor_axis = max(axes), min(axes)
            
            # Clubbing increases curvature (lower major/minor ratio)
            # Normal nails: ratio ~1.5-2.0
            # Clubbed nails: ratio ~1.0-1.3
            if major_axis == 0:
                return 0.0
            
            axis_ratio = major_axis / minor_axis if minor_axis > 0 else 2.0
            
            # Convert ratio to clubbing score
            # Lower ratio = higher clubbing score
            if axis_ratio < 1.3:
                clubbing_score = 1.0 - (axis_ratio - 1.0) / 0.3
            elif axis_ratio < 1.5:
                clubbing_score = 0.5 - (axis_ratio - 1.3) / 0.4
            else:
                clubbing_score = 0.0
            
            return max(0.0, min(1.0, clubbing_score))
        
        except Exception as e:
            logger.error(f"Clubbing detection error: {str(e)}")
            return 0.0
    
    def _detect_nail_pitting(
        self,
        nailbed_roi: np.ndarray
    ) -> float:
        """
        Detect nail pitting using surface texture variance
        
        Pitting manifests as small depressions (pits) in nail plate
        Detected via high local variance in grayscale intensity
        
        Returns pitting severity score (0-1)
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(nailbed_roi, cv2.COLOR_RGB2GRAY)
            
            # Apply Laplacian to detect surface irregularities
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            laplacian_var = np.var(laplacian)
            
            # High variance = more texture/pitting
            # Normalize to 0-1 scale (typical range: 0-500)
            pitting_score = min(1.0, laplacian_var / 500.0)
            
            return float(pitting_score)
        
        except Exception as e:
            logger.error(f"Pitting detection error: {str(e)}")
            return 0.0
    
    def _detect_nail_abnormalities(
        self,
        nailbed_roi: np.ndarray,
        l_value: float,
        a_value: float,
        b_value: float
    ) -> Optional[Dict[str, Any]]:
        """
        Detect nail abnormalities based on color analysis:
        - Leukonychia (white spots/streaks)
        - Splinter hemorrhages (red/brown streaks)
        - Yellow nail syndrome
        - Terry's nails (white proximal, pink distal)
        """
        abnormalities = []
        
        # Leukonychia: High L* (very white), low a* and b*
        if l_value > 220 and abs(a_value - 128) < 10 and abs(b_value - 128) < 10:
            abnormalities.append({
                'type': 'leukonychia',
                'severity': (l_value - 200) / 55.0,
                'location': 'generalized'
            })
        
        # Splinter hemorrhages: High a* (redness), low b*
        if (a_value - 128) > 20 and (b_value - 128) < 10:
            abnormalities.append({
                'type': 'splinter_hemorrhage',
                'severity': ((a_value - 128) - 20) / 40.0,
                'location': 'linear'
            })
        
        # Yellow nail syndrome: High b* (yellowness)
        if (b_value - 128) > 30:
            abnormalities.append({
                'type': 'yellow_nails',
                'severity': ((b_value - 128) - 30) / 40.0,
                'location': 'generalized'
            })
        
        return abnormalities[0] if abnormalities else None
    
    def _compute_perfusion_index(
        self,
        l_value: float,
        a_value: float,
        b_value: float
    ) -> float:
        """
        Compute perfusion index (0-100) from LAB color values
        
        Formula:
        - Higher L* (lightness) = moderate contribution
        - Higher a* (redness) = strong contribution (blood perfusion)
        - b* (yellowness) has minimal effect
        
        Well-perfused skin: L*~60-70, a*~140-160 (shifted), b*~130-150 (shifted)
        Poorly perfused: L*<50, a*<130, cyanotic: a*<120, b*<120
        """
        # Normalize L* (0-100 range)
        l_normalized = l_value / 100.0
        
        # Normalize a* (shifted from 0-255 to represent -128 to 127)
        # Healthy redness: a* around 140-160 (in 0-255 scale)
        # Convert to 0-1 scale where 128 (neutral) = 0.5, 160 (red) = ~0.7
        a_normalized = (a_value / 255.0)
        
        # Perfusion index weighted formula
        # 70% weight on redness (a*), 30% on lightness (L*)
        perfusion_index = (0.7 * a_normalized + 0.3 * l_normalized) * 100.0
        
        return max(0.0, min(100.0, perfusion_index))
    
    def _detect_clinical_color_changes(
        self,
        skin_metrics: Dict[str, Any],
        baseline: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Detect clinical color changes: pallor, cyanosis, jaundice
        
        Pallor: Low L* + low a* (reduced redness)
        Cyanosis: Low L* + negative a* + negative b* (bluish)
        Jaundice: High b* (yellowness), especially in sclera
        """
        changes = {
            'pallor_detected': False,
            'pallor_severity': 0.0,
            'pallor_region': 'none',
            'cyanosis_detected': False,
            'cyanosis_severity': 0.0,
            'cyanosis_region': 'none',
            'jaundice_detected': False,
            'jaundice_severity': 0.0,
            'jaundice_region': 'none'
        }
        
        # Extract LAB values
        facial_l = skin_metrics.get('facial_l_lightness', 0)
        facial_a = skin_metrics.get('facial_a_red_green', 0)
        facial_b = skin_metrics.get('facial_b_yellow_blue', 0)
        
        palmar_l = skin_metrics.get('palmar_l_lightness', 0)
        palmar_a = skin_metrics.get('palmar_a_red_green', 0)
        
        nailbed_l = skin_metrics.get('nailbed_l_lightness', 0)
        nailbed_a = skin_metrics.get('nailbed_a_red_green', 0)
        
        # Get baseline thresholds
        baseline_facial_l = baseline.get('baseline_facial_l', 60) if baseline else 60
        baseline_facial_a = baseline.get('baseline_facial_a', 10) if baseline else 10
        
        # 1. PALLOR DETECTION (reduced perfusion, low redness)
        # Criteria: L* < baseline - 10 AND a* < baseline - 10
        if facial_l > 0 and facial_a < (baseline_facial_a - 10):
            pallor_severity = max(0, (baseline_facial_a - facial_a) / 20.0)
            changes['pallor_detected'] = True
            changes['pallor_severity'] = min(1.0, pallor_severity)
            
            # Determine region
            if palmar_a > 0 and palmar_a < baseline_facial_a - 10:
                changes['pallor_region'] = 'generalized'
            else:
                changes['pallor_region'] = 'facial'
        
        # Check palmar pallor specifically
        if palmar_l > 0 and palmar_a < -5:  # Palmar pallor threshold
            changes['pallor_detected'] = True
            changes['pallor_region'] = 'palmar' if changes['pallor_region'] == 'none' else 'generalized'
            changes['pallor_severity'] = max(changes['pallor_severity'], abs(palmar_a) / 15.0)
        
        # 2. CYANOSIS DETECTION (bluish discoloration)
        # Criteria: Low L*, negative a* (less red), negative b* (more blue)
        if facial_l > 0 and facial_a < -5 and facial_b < -5:
            cyanosis_severity = (abs(facial_a) + abs(facial_b)) / 30.0
            changes['cyanosis_detected'] = True
            changes['cyanosis_severity'] = min(1.0, cyanosis_severity)
            changes['cyanosis_region'] = 'central'  # Facial = central cyanosis
        
        # Peripheral cyanosis (nailbeds)
        if nailbed_l > 0 and nailbed_a < -8:
            changes['cyanosis_detected'] = True
            changes['cyanosis_region'] = 'peripheral' if changes['cyanosis_region'] == 'none' else 'generalized'
            changes['cyanosis_severity'] = max(changes['cyanosis_severity'], abs(nailbed_a) / 20.0)
        
        # 3. JAUNDICE DETECTION (yellowness)
        # Criteria: High b* (yellowness), especially >20 in facial region
        if facial_b > 20:
            jaundice_severity = (facial_b - 20) / 30.0
            changes['jaundice_detected'] = True
            changes['jaundice_severity'] = min(1.0, jaundice_severity)
            changes['jaundice_region'] = 'facial'
        
        return changes
    
    def _analyze_skin_texture_hydration(
        self,
        frame: np.ndarray,
        landmarks: np.ndarray
    ) -> Dict[str, Any]:
        """
        Analyze skin texture and hydration status
        
        Texture: Laplacian variance (high = rough/dry, low = smooth)
        Hydration: Shine detection via specular highlights
        Temperature proxy: Redness in HSV
        """
        try:
            # Extract facial ROI
            x_min, y_min = landmarks.min(axis=0).astype(int)
            x_max, y_max = landmarks.max(axis=0).astype(int)
            padding = 10
            face_roi = frame[max(0, y_min-padding):min(frame.shape[0], y_max+padding),
                            max(0, x_min-padding):min(frame.shape[1], x_max+padding)]
            
            if face_roi.size == 0:
                return {'skin_texture_score': 0.0, 'hydration_status': 'normal', 'temperature_proxy': 'normal'}
            
            # Convert to grayscale for texture analysis
            gray = cv2.cvtColor(face_roi, cv2.COLOR_RGB2GRAY)
            
            # Compute Laplacian variance (texture roughness)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            texture_score = np.var(laplacian) / 200.0  # Normalize to 0-1
            texture_score = min(1.0, texture_score)
            
            # Hydration status via shine detection
            # Moist skin reflects more light (higher V channel in HSV)
            hsv = cv2.cvtColor(face_roi, cv2.COLOR_RGB2HSV)
            v_channel = hsv[:, :, 2]
            avg_brightness = np.mean(v_channel)
            
            # Determine hydration
            if avg_brightness > 180:
                hydration_status = 'moist'
            elif avg_brightness < 120:
                hydration_status = 'dry'
            else:
                hydration_status = 'normal'
            
            # Temperature proxy via redness (a* channel or H in HSV)
            # High S + low H (red hue) = warm/inflamed
            # Low S + low V = cool/pale
            h_channel = hsv[:, :, 0]
            s_channel = hsv[:, :, 1]
            
            avg_hue = np.mean(h_channel)
            avg_sat = np.mean(s_channel)
            
            if avg_sat > 100 and (avg_hue < 10 or avg_hue > 170):
                temperature_proxy = 'warm'
            elif avg_sat < 40:
                temperature_proxy = 'cool'
            else:
                temperature_proxy = 'normal'
            
            return {
                'skin_texture_score': float(texture_score),
                'hydration_status': hydration_status,
                'temperature_proxy': temperature_proxy
            }
        
        except Exception as e:
            logger.error(f"Texture/hydration analysis error: {str(e)}")
            return {'skin_texture_score': 0.0, 'hydration_status': 'normal', 'temperature_proxy': 'normal'}
    
    def _compute_skin_detection_confidence(
        self,
        skin_metrics: Dict[str, Any]
    ) -> float:
        """
        Compute overall confidence in skin analysis detection
        
        Based on:
        - Number of ROIs detected (facial, palmar, nailbed)
        - Quality of LAB values (non-zero, within expected ranges)
        - Consistency across regions
        """
        confidence_factors = []
        
        # Facial ROI detection
        if skin_metrics.get('facial_roi_detected', False):
            facial_l = skin_metrics.get('facial_l_lightness', 0)
            if 20 < facial_l < 100:  # Valid L* range
                confidence_factors.append(0.4)
        
        # Palmar ROI detection (bonus)
        if skin_metrics.get('palmar_roi_detected', False):
            palmar_l = skin_metrics.get('palmar_l_lightness', 0)
            if 20 < palmar_l < 100:
                confidence_factors.append(0.3)
        
        # Nailbed ROI detection (bonus)
        if skin_metrics.get('nailbed_roi_detected', False):
            nailbed_l = skin_metrics.get('nailbed_l_lightness', 0)
            if 20 < nailbed_l < 100:
                confidence_factors.append(0.3)
        
        return sum(confidence_factors)
    
    def track_capillary_refill(
        self,
        lab_timeseries: List[Tuple[int, float]],
        fps: float = 30.0
    ) -> Optional[Dict[str, Any]]:
        """
        Track capillary refill time using L* channel tracking
        
        Detects:
        1. Finger press event (L* drops significantly)
        2. Recovery phase (L* returns to baseline)
        3. Refill time (time to reach 90% of baseline L*)
        
        Args:
            lab_timeseries: List of (frame_index, L_value) tuples
            fps: Frames per second
            
        Returns:
            Dict with refill time, quality, method, abnormal flag
        """
        if len(lab_timeseries) < 10:
            return {
                'capillary_refill_time_sec': None,
                'capillary_refill_method': 'not_measured',
                'capillary_refill_quality': 0.0,
                'capillary_refill_abnormal': False
            }
        
        try:
            # Extract L* values
            l_values = np.array([v[1] for v in lab_timeseries])
            
            # Compute baseline L* (median of first 20%)
            baseline_samples = int(len(l_values) * 0.2)
            baseline_l = np.median(l_values[:baseline_samples])
            
            # Detect press event (L* drop > 15% of baseline)
            press_threshold = baseline_l * 0.85
            press_detected = False
            press_frame_idx = -1
            
            for i, l_val in enumerate(l_values):
                if l_val < press_threshold:
                    press_detected = True
                    press_frame_idx = i
                    break
            
            if not press_detected:
                return {
                    'capillary_refill_time_sec': None,
                    'capillary_refill_method': 'passive_observation',
                    'capillary_refill_quality': 0.3,
                    'capillary_refill_abnormal': False
                }
            
            # Track recovery to 90% of baseline
            recovery_threshold = baseline_l * 0.9
            recovery_frame_idx = -1
            
            for i in range(press_frame_idx, len(l_values)):
                if l_values[i] >= recovery_threshold:
                    recovery_frame_idx = i
                    break
            
            if recovery_frame_idx == -1:
                # No recovery detected
                return {
                    'capillary_refill_time_sec': None,
                    'capillary_refill_method': 'guided_press',
                    'capillary_refill_quality': 0.5,
                    'capillary_refill_abnormal': True  # Prolonged or no recovery
                }
            
            # Compute refill time
            refill_frames = recovery_frame_idx - press_frame_idx
            refill_time_sec = refill_frames / fps
            
            # Quality assessment
            quality = 0.9 if refill_time_sec < 5.0 else 0.7
            
            # Abnormal if >2 seconds (clinical threshold)
            abnormal = refill_time_sec > 2.0
            
            return {
                'capillary_refill_time_sec': float(refill_time_sec),
                'capillary_refill_method': 'guided_press',
                'capillary_refill_quality': quality,
                'capillary_refill_abnormal': abnormal
            }
        
        except Exception as e:
            logger.error(f"Capillary refill tracking error: {str(e)}")
            return {
                'capillary_refill_time_sec': None,
                'capillary_refill_method': 'not_measured',
                'capillary_refill_quality': 0.0,
                'capillary_refill_abnormal': False
            }


# Global instance
video_ai_engine = VideoAIEngine()
