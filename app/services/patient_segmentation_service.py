"""
Production-Grade K-Means Patient Segmentation Service
=====================================================

Implements scikit-learn K-Means clustering for patient phenotyping and segmentation
with production-grade features:
- Dynamic clustering based on actual patient data
- Elbow method for optimal K selection
- Silhouette scoring for cluster quality validation
- Feature importance per cluster
- Per-cluster statistics and interpretable labels
- Graceful fallback to centroid-based assignment when insufficient data
"""

import numpy as np
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import json

try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score, silhouette_samples
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("scikit-learn not available, using fallback segmentation")

logger = logging.getLogger(__name__)


class PatientSegmentationService:
    """
    Production-grade K-Means patient segmentation with scikit-learn.
    
    Features:
    - Dynamic clustering based on patient cohort data
    - Automatic K selection via elbow method
    - Cluster quality validation with silhouette score
    - Interpretable cluster labels based on centroid analysis
    - Feature importance ranking per cluster
    - Fallback to predefined centroids when data is insufficient
    """
    
    FEATURE_NAMES = [
        "phq9_score",
        "gad7_score",
        "pss10_score",
        "daily_steps",
        "sleep_hours",
        "checkin_rate",
        "symptom_count",
        "pain_level",
        "medication_adherence",
        "vital_stability",
        "age_normalized",
        "comorbidity_count"
    ]
    
    FEATURE_WEIGHTS = {
        "phq9_score": 1.5,
        "gad7_score": 1.3,
        "pss10_score": 1.2,
        "daily_steps": 0.8,
        "sleep_hours": 1.0,
        "checkin_rate": 1.2,
        "symptom_count": 1.4,
        "pain_level": 1.3,
        "medication_adherence": 1.5,
        "vital_stability": 1.4,
        "age_normalized": 0.6,
        "comorbidity_count": 1.2
    }
    
    CLINICAL_LABELS = {
        0: {
            "name": "Wellness Engaged",
            "description": "Actively engaged in health management with stable outcomes",
            "color": "#22c55e",
            "care_level": "standard",
            "icon": "heart-pulse"
        },
        1: {
            "name": "Moderate Risk",
            "description": "Some health challenges requiring enhanced monitoring",
            "color": "#eab308",
            "care_level": "enhanced",
            "icon": "activity"
        },
        2: {
            "name": "High Complexity",
            "description": "Multiple health challenges needing intensive support",
            "color": "#f97316",
            "care_level": "intensive",
            "icon": "alert-triangle"
        },
        3: {
            "name": "Critical Needs",
            "description": "Significant health concerns requiring immediate attention",
            "color": "#ef4444",
            "care_level": "urgent",
            "icon": "alert-circle"
        }
    }
    
    FALLBACK_CENTROIDS = {
        0: {
            "phq9_score": 0.15, "gad7_score": 0.12, "pss10_score": 0.10,
            "daily_steps": 0.7, "sleep_hours": 0.65, "checkin_rate": 0.85,
            "symptom_count": 0.1, "pain_level": 0.15, "medication_adherence": 0.9,
            "vital_stability": 0.85, "age_normalized": 0.4, "comorbidity_count": 0.1
        },
        1: {
            "phq9_score": 0.35, "gad7_score": 0.30, "pss10_score": 0.28,
            "daily_steps": 0.45, "sleep_hours": 0.50, "checkin_rate": 0.60,
            "symptom_count": 0.35, "pain_level": 0.35, "medication_adherence": 0.7,
            "vital_stability": 0.65, "age_normalized": 0.5, "comorbidity_count": 0.25
        },
        2: {
            "phq9_score": 0.55, "gad7_score": 0.50, "pss10_score": 0.48,
            "daily_steps": 0.25, "sleep_hours": 0.40, "checkin_rate": 0.40,
            "symptom_count": 0.55, "pain_level": 0.55, "medication_adherence": 0.5,
            "vital_stability": 0.45, "age_normalized": 0.6, "comorbidity_count": 0.45
        },
        3: {
            "phq9_score": 0.75, "gad7_score": 0.70, "pss10_score": 0.68,
            "daily_steps": 0.15, "sleep_hours": 0.30, "checkin_rate": 0.25,
            "symptom_count": 0.75, "pain_level": 0.70, "medication_adherence": 0.3,
            "vital_stability": 0.25, "age_normalized": 0.65, "comorbidity_count": 0.65
        }
    }
    
    def __init__(self, n_clusters: int = 4, random_state: int = 42):
        """
        Initialize segmentation service.
        
        Args:
            n_clusters: Number of clusters (default 4 for clinical phenotypes)
            random_state: Random seed for reproducibility
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.model: Optional[KMeans] = None
        self.scaler: Optional[StandardScaler] = None
        self.cluster_centroids: Optional[np.ndarray] = None
        self.cluster_labels_map: Dict[int, Dict] = {}
        self.feature_importance: Dict[int, List] = {}
        self.silhouette_avg: float = 0.0
        self.is_trained: bool = False
        self.training_stats: Dict = {}
    
    def fit(
        self,
        patient_features: List[Dict[str, float]],
        auto_select_k: bool = True,
        min_k: int = 2,
        max_k: int = 6
    ) -> Dict[str, Any]:
        """
        Train K-Means clustering model on patient cohort data.
        
        Args:
            patient_features: List of patient feature dictionaries
            auto_select_k: Whether to automatically select optimal K
            min_k: Minimum K for auto selection
            max_k: Maximum K for auto selection
        
        Returns:
            Training statistics and cluster information
        """
        if not SKLEARN_AVAILABLE:
            logger.warning("sklearn not available, using fallback centroids")
            return self._use_fallback_centroids()
        
        if len(patient_features) < 10:
            logger.warning(f"Insufficient data ({len(patient_features)} patients), using fallback centroids")
            return self._use_fallback_centroids()
        
        try:
            X = self._prepare_feature_matrix(patient_features)
            
            if X.shape[0] < max_k * 3:
                max_k = max(min_k, X.shape[0] // 3)
                logger.info(f"Reduced max_k to {max_k} due to sample size")
            
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            if auto_select_k:
                optimal_k, elbow_results = self._find_optimal_k(X_scaled, min_k, max_k)
                self.n_clusters = optimal_k
                logger.info(f"Auto-selected K={optimal_k} clusters")
            else:
                elbow_results = {}
            
            self.model = KMeans(
                n_clusters=self.n_clusters,
                random_state=self.random_state,
                n_init=10,
                max_iter=300,
                algorithm='lloyd'
            )
            
            cluster_labels = self.model.fit_predict(X_scaled)
            
            self.cluster_centroids = self.scaler.inverse_transform(self.model.cluster_centers_)
            
            if len(np.unique(cluster_labels)) > 1:
                self.silhouette_avg = silhouette_score(X_scaled, cluster_labels)
            else:
                self.silhouette_avg = 0.0
            
            self._assign_clinical_labels()
            self._compute_feature_importance(X, cluster_labels)
            
            self.training_stats = {
                "n_patients": len(patient_features),
                "n_clusters": self.n_clusters,
                "silhouette_score": round(self.silhouette_avg, 4),
                "inertia": round(self.model.inertia_, 4),
                "cluster_sizes": {
                    str(i): int(np.sum(cluster_labels == i))
                    for i in range(self.n_clusters)
                },
                "elbow_results": elbow_results,
                "trained_at": datetime.utcnow().isoformat()
            }
            
            self.is_trained = True
            logger.info(f"Trained K-Means with {self.n_clusters} clusters, silhouette={self.silhouette_avg:.3f}")
            
            return self.training_stats
            
        except Exception as e:
            logger.error(f"Training failed: {e}, using fallback centroids")
            return self._use_fallback_centroids()
    
    def _prepare_feature_matrix(self, patient_features: List[Dict[str, float]]) -> np.ndarray:
        """Convert patient feature dictionaries to numpy matrix."""
        X = []
        for pf in patient_features:
            row = []
            for feature in self.FEATURE_NAMES:
                value = pf.get(feature, 0.5)
                weight = self.FEATURE_WEIGHTS.get(feature, 1.0)
                row.append(float(value) * weight)
            X.append(row)
        return np.array(X)
    
    def _find_optimal_k(
        self,
        X_scaled: np.ndarray,
        min_k: int,
        max_k: int
    ) -> Tuple[int, Dict]:
        """
        Find optimal K using elbow method with silhouette validation.
        
        Returns:
            Tuple of (optimal_k, elbow_results_dict)
        """
        inertias = []
        silhouettes = []
        
        for k in range(min_k, max_k + 1):
            kmeans = KMeans(
                n_clusters=k,
                random_state=self.random_state,
                n_init=10,
                max_iter=300
            )
            labels = kmeans.fit_predict(X_scaled)
            inertias.append(kmeans.inertia_)
            
            if len(np.unique(labels)) > 1:
                sil = silhouette_score(X_scaled, labels)
            else:
                sil = 0.0
            silhouettes.append(sil)
        
        elbow_results = {
            "k_values": list(range(min_k, max_k + 1)),
            "inertias": [round(x, 2) for x in inertias],
            "silhouettes": [round(x, 4) for x in silhouettes]
        }
        
        best_silhouette_idx = np.argmax(silhouettes)
        optimal_k = min_k + best_silhouette_idx
        
        if len(inertias) >= 3:
            second_derivatives = []
            for i in range(1, len(inertias) - 1):
                d2 = inertias[i-1] - 2*inertias[i] + inertias[i+1]
                second_derivatives.append(d2)
            
            if second_derivatives:
                elbow_idx = np.argmax(second_derivatives) + 1
                elbow_k = min_k + elbow_idx
                
                if silhouettes[optimal_k - min_k] < 0.1:
                    optimal_k = elbow_k
        
        return optimal_k, elbow_results
    
    def _assign_clinical_labels(self):
        """
        Assign interpretable clinical labels to clusters based on centroid analysis.
        
        Maps cluster indices to clinical phenotypes by analyzing centroid feature patterns.
        """
        if self.cluster_centroids is None:
            self.cluster_labels_map = self.CLINICAL_LABELS.copy()
            return
        
        cluster_scores = []
        for i in range(self.n_clusters):
            centroid = self.cluster_centroids[i]
            centroid_dict = dict(zip(self.FEATURE_NAMES, centroid))
            
            risk_score = (
                centroid_dict.get("phq9_score", 0.5) * 1.5 +
                centroid_dict.get("gad7_score", 0.5) * 1.3 +
                centroid_dict.get("symptom_count", 0.5) * 1.4 +
                centroid_dict.get("pain_level", 0.5) * 1.3 -
                centroid_dict.get("daily_steps", 0.5) * 0.8 -
                centroid_dict.get("medication_adherence", 0.5) * 1.5 -
                centroid_dict.get("vital_stability", 0.5) * 1.4 -
                centroid_dict.get("checkin_rate", 0.5) * 1.2
            )
            cluster_scores.append((i, risk_score))
        
        sorted_clusters = sorted(cluster_scores, key=lambda x: x[1])
        
        label_indices = list(range(min(self.n_clusters, 4)))
        
        self.cluster_labels_map = {}
        for rank, (cluster_idx, _) in enumerate(sorted_clusters):
            if rank < len(label_indices):
                label_idx = label_indices[rank]
            else:
                label_idx = 3
            
            self.cluster_labels_map[cluster_idx] = self.CLINICAL_LABELS.get(
                label_idx,
                self.CLINICAL_LABELS[3]
            )
    
    def _compute_feature_importance(
        self,
        X: np.ndarray,
        labels: np.ndarray
    ):
        """
        Compute feature importance for each cluster.
        
        Uses variance ratio to identify defining features per cluster.
        """
        self.feature_importance = {}
        
        global_mean = np.mean(X, axis=0)
        global_std = np.std(X, axis=0) + 1e-6
        
        for cluster_idx in range(self.n_clusters):
            cluster_mask = labels == cluster_idx
            if np.sum(cluster_mask) < 2:
                self.feature_importance[cluster_idx] = []
                continue
            
            cluster_data = X[cluster_mask]
            cluster_mean = np.mean(cluster_data, axis=0)
            
            z_scores = (cluster_mean - global_mean) / global_std
            
            importance = []
            for i, (feature_name, z) in enumerate(zip(self.FEATURE_NAMES, z_scores)):
                importance.append({
                    "feature": feature_name,
                    "z_score": round(float(z), 3),
                    "direction": "elevated" if z > 0 else "reduced",
                    "magnitude": abs(round(float(z), 3)),
                    "cluster_mean": round(float(cluster_mean[i]), 3),
                    "global_mean": round(float(global_mean[i]), 3)
                })
            
            importance.sort(key=lambda x: x["magnitude"], reverse=True)
            self.feature_importance[cluster_idx] = importance[:5]
    
    def _use_fallback_centroids(self) -> Dict[str, Any]:
        """Initialize with predefined fallback centroids."""
        self.cluster_centroids = np.array([
            [self.FALLBACK_CENTROIDS[i].get(f, 0.5) for f in self.FEATURE_NAMES]
            for i in range(4)
        ])
        self.n_clusters = 4
        self.cluster_labels_map = self.CLINICAL_LABELS.copy()
        self.is_trained = True
        
        for i in range(4):
            self.feature_importance[i] = [
                {
                    "feature": f,
                    "z_score": 0.0,
                    "direction": "typical",
                    "magnitude": 0.0,
                    "cluster_mean": self.FALLBACK_CENTROIDS[i].get(f, 0.5),
                    "global_mean": 0.5
                }
                for f in self.FEATURE_NAMES[:5]
            ]
        
        self.training_stats = {
            "n_patients": 0,
            "n_clusters": 4,
            "silhouette_score": 0.0,
            "inertia": 0.0,
            "cluster_sizes": {},
            "method": "fallback_centroids",
            "trained_at": datetime.utcnow().isoformat()
        }
        
        logger.info("Using fallback centroids for segmentation")
        return self.training_stats
    
    def predict(
        self,
        patient_features: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Assign a single patient to a cluster.
        
        Args:
            patient_features: Dictionary of patient features
        
        Returns:
            Cluster assignment with confidence and recommendations
        """
        if not self.is_trained:
            self._use_fallback_centroids()
        
        feature_vector = np.array([
            patient_features.get(f, 0.5) * self.FEATURE_WEIGHTS.get(f, 1.0)
            for f in self.FEATURE_NAMES
        ]).reshape(1, -1)
        
        if self.model is not None and self.scaler is not None:
            try:
                X_scaled = self.scaler.transform(feature_vector)
                cluster_idx = int(self.model.predict(X_scaled)[0])
                
                distances = self.model.transform(X_scaled)[0]
                min_dist = distances[cluster_idx]
                max_dist = np.max(distances)
                confidence = 1.0 - (min_dist / (max_dist + 1e-6))
                
                sample_silhouette = silhouette_samples(
                    np.vstack([self.scaler.transform(self.cluster_centroids), X_scaled]),
                    np.append(np.arange(self.n_clusters), cluster_idx)
                )[-1]
                
                return self._build_prediction_result(
                    cluster_idx=cluster_idx,
                    confidence=confidence,
                    patient_features=patient_features,
                    distances=distances,
                    silhouette=sample_silhouette
                )
            except Exception as e:
                logger.warning(f"Prediction with trained model failed: {e}, using distance-based")
        
        return self._predict_with_centroids(patient_features)
    
    def _predict_with_centroids(
        self,
        patient_features: Dict[str, float]
    ) -> Dict[str, Any]:
        """Fallback prediction using distance to centroids."""
        distances = []
        
        for i in range(self.n_clusters):
            centroid = self.cluster_centroids[i]
            distance = 0.0
            
            for j, feature in enumerate(self.FEATURE_NAMES):
                patient_value = patient_features.get(feature, 0.5)
                weight = self.FEATURE_WEIGHTS.get(feature, 1.0)
                distance += weight * (patient_value - centroid[j]) ** 2
            
            distances.append(np.sqrt(distance))
        
        cluster_idx = int(np.argmin(distances))
        min_dist = distances[cluster_idx]
        max_dist = max(distances)
        confidence = 1.0 - (min_dist / (max_dist + 1e-6))
        
        return self._build_prediction_result(
            cluster_idx=cluster_idx,
            confidence=confidence,
            patient_features=patient_features,
            distances=np.array(distances),
            silhouette=None
        )
    
    def _build_prediction_result(
        self,
        cluster_idx: int,
        confidence: float,
        patient_features: Dict[str, float],
        distances: np.ndarray,
        silhouette: Optional[float]
    ) -> Dict[str, Any]:
        """Build standardized prediction result."""
        cluster_info = self.cluster_labels_map.get(cluster_idx, self.CLINICAL_LABELS[3])
        
        centroid = self.cluster_centroids[cluster_idx]
        deviations = []
        
        for i, feature in enumerate(self.FEATURE_NAMES):
            patient_value = patient_features.get(feature, 0.5)
            centroid_value = centroid[i]
            deviation = patient_value - centroid_value
            
            if abs(deviation) > 0.15:
                deviations.append({
                    "feature": feature,
                    "patient_value": round(patient_value, 3),
                    "cluster_average": round(centroid_value, 3),
                    "segment_average": round(centroid_value, 3),
                    "deviation": round(deviation, 3),
                    "direction": "above" if deviation > 0 else "below"
                })
        
        deviations.sort(key=lambda x: abs(x["deviation"]), reverse=True)
        
        alternative_clusters = []
        sorted_distances = sorted(enumerate(distances), key=lambda x: x[1])
        
        for alt_idx, alt_dist in sorted_distances[1:4]:
            alt_info = self.cluster_labels_map.get(alt_idx, self.CLINICAL_LABELS.get(alt_idx, {}))
            alternative_clusters.append({
                "cluster_id": alt_idx,
                "cluster_name": alt_info.get("name", f"Cluster {alt_idx}"),
                "distance": round(float(alt_dist), 4),
                "relative_fit": round(1.0 - (alt_dist / (max(distances) + 1e-6)), 3)
            })
        
        recommendations = self._get_recommendations(cluster_idx, deviations)
        
        is_fallback = self.model is None
        is_low_quality = self.silhouette_avg < 0.1 if self.silhouette_avg else True
        
        quality_status = "high"
        if is_fallback:
            quality_status = "fallback"
        elif is_low_quality:
            quality_status = "low"
        
        adjusted_confidence = confidence
        if is_low_quality and not is_fallback:
            adjusted_confidence = min(confidence, 0.7)
        
        return {
            "segment": {
                "cluster_id": cluster_idx,
                "cluster_name": cluster_info.get("name", f"Cluster {cluster_idx}"),
                "cluster_description": cluster_info.get("description", ""),
                "confidence": round(float(adjusted_confidence), 3),
                "distance_to_centroid": round(float(distances[cluster_idx]), 4),
                "silhouette_score": round(float(silhouette), 4) if silhouette else None,
                "care_level": cluster_info.get("care_level", "standard"),
                "color": cluster_info.get("color", "#6b7280"),
                "icon": cluster_info.get("icon", "user"),
                "feature_importance": self.feature_importance.get(cluster_idx, []),
                "recommended_interventions": recommendations
            },
            "alternative_segments": alternative_clusters,
            "key_deviations": deviations[:5],
            "model_info": {
                "method": "sklearn_kmeans" if self.model else "centroid_distance",
                "n_clusters": self.n_clusters,
                "model_silhouette": round(self.silhouette_avg, 4) if self.silhouette_avg else None,
                "quality": quality_status
            },
            "segmented_at": datetime.utcnow().isoformat()
        }
    
    def _get_recommendations(
        self,
        cluster_idx: int,
        deviations: List[Dict]
    ) -> List[str]:
        """Generate personalized recommendations based on cluster and deviations."""
        recommendations = []
        cluster_info = self.cluster_labels_map.get(cluster_idx, {})
        care_level = cluster_info.get("care_level", "standard")
        
        if care_level == "standard":
            recommendations.append("Continue current health management routine")
            recommendations.append("Maintain regular check-ins and activity tracking")
        elif care_level == "enhanced":
            recommendations.append("Increase check-in frequency to twice daily")
            recommendations.append("Review and optimize care plan with provider")
            recommendations.append("Consider additional monitoring for concerning metrics")
        elif care_level == "intensive":
            recommendations.append("Schedule comprehensive care review")
            recommendations.append("Activate intensive monitoring protocol")
            recommendations.append("Consider care coordination services")
            recommendations.append("Evaluate need for additional support resources")
        elif care_level == "urgent":
            recommendations.append("Urgent care team evaluation recommended")
            recommendations.append("Activate 24/7 monitoring protocol")
            recommendations.append("Ensure immediate care support access")
            recommendations.append("Consider emergency intervention if deteriorating")
        
        for dev in deviations[:3]:
            feature = dev.get("feature", "")
            direction = dev.get("direction", "")
            
            if feature == "phq9_score" and direction == "above":
                recommendations.append("Mental health evaluation and support recommended")
            elif feature == "gad7_score" and direction == "above":
                recommendations.append("Anxiety management intervention may be beneficial")
            elif feature == "medication_adherence" and direction == "below":
                recommendations.append("Medication adherence support program recommended")
            elif feature == "sleep_hours" and direction == "below":
                recommendations.append("Sleep quality assessment and hygiene intervention")
            elif feature == "daily_steps" and direction == "below":
                recommendations.append("Activity promotion program recommended")
            elif feature == "checkin_rate" and direction == "below":
                recommendations.append("Engagement optimization with personalized reminders")
            elif feature == "vital_stability" and direction == "below":
                recommendations.append("Vital sign monitoring frequency increase")
        
        return recommendations[:6]
    
    def get_cluster_statistics(self) -> Dict[str, Any]:
        """Get detailed statistics about all clusters."""
        if not self.is_trained:
            return {"error": "Model not trained"}
        
        cluster_stats = {}
        
        for i in range(self.n_clusters):
            cluster_info = self.cluster_labels_map.get(i, {})
            centroid = self.cluster_centroids[i] if self.cluster_centroids is not None else None
            
            centroid_features = {}
            if centroid is not None:
                for j, feature in enumerate(self.FEATURE_NAMES):
                    centroid_features[feature] = round(float(centroid[j]), 3)
            
            cluster_stats[str(i)] = {
                "cluster_id": i,
                "name": cluster_info.get("name", f"Cluster {i}"),
                "description": cluster_info.get("description", ""),
                "care_level": cluster_info.get("care_level", "standard"),
                "color": cluster_info.get("color", "#6b7280"),
                "size": self.training_stats.get("cluster_sizes", {}).get(str(i), 0),
                "centroid": centroid_features,
                "top_features": self.feature_importance.get(i, [])
            }
        
        return {
            "clusters": cluster_stats,
            "model_quality": {
                "silhouette_score": self.silhouette_avg,
                "n_clusters": self.n_clusters,
                "total_patients": self.training_stats.get("n_patients", 0),
                "inertia": self.training_stats.get("inertia", 0)
            },
            "feature_names": self.FEATURE_NAMES,
            "generated_at": datetime.utcnow().isoformat()
        }


_segmentation_service: Optional[PatientSegmentationService] = None


def get_segmentation_service() -> PatientSegmentationService:
    """Get or create singleton segmentation service."""
    global _segmentation_service
    if _segmentation_service is None:
        _segmentation_service = PatientSegmentationService()
    return _segmentation_service


def train_segmentation_model(
    patient_features: List[Dict[str, float]],
    auto_k: bool = True
) -> Dict[str, Any]:
    """Train the global segmentation model."""
    service = get_segmentation_service()
    return service.fit(patient_features, auto_select_k=auto_k)


def segment_patient(
    patient_features: Dict[str, float]
) -> Dict[str, Any]:
    """Segment a single patient using the global model."""
    service = get_segmentation_service()
    return service.predict(patient_features)
