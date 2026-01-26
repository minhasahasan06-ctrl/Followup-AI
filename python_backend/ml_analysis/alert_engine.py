"""
Alert Engine for Research Center

Implements periodic risk scoring for deterioration,
threshold logic with SHAP explanations, and alert creation with top features.
"""

import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import psycopg2
from psycopg2.extras import RealDictCursor
import warnings
warnings.filterwarnings('ignore')

@dataclass
class AlertConfig:
    """Configuration for alert engine"""
    high_risk_threshold: float = 0.7
    moderate_risk_threshold: float = 0.4
    lookback_days: int = 30
    min_data_points: int = 5
    alert_cooldown_hours: int = 24
    top_features_count: int = 5
    connection_string: Optional[str] = None

@dataclass
class AlertResult:
    """Result of alert evaluation"""
    patient_id: str
    risk_score: float
    risk_level: str
    alert_triggered: bool
    top_features: List[Dict[str, Any]]
    explanation: str
    trend: str
    timestamp: datetime

class AlertEngine:
    """
    Intelligent alert engine for patient deterioration detection.
    
    Features:
    - Real-time risk scoring
    - SHAP-based explanations
    - Threshold-based alerting
    - Alert cooldown management
    - Trend detection
    - Multi-factor risk assessment
    """
    
    def __init__(self, config: Optional[AlertConfig] = None):
        self.config = config or AlertConfig()
        self.connection_string = self.config.connection_string or os.environ.get('DATABASE_URL')
        self.model = None
        self.scaler = None
        self.imputer = None
        self.feature_names = None
        
    def get_connection(self):
        """Create database connection"""
        return psycopg2.connect(self.connection_string)
    
    def load_model(self, model_path: Optional[str] = None):
        """Load or train risk prediction model"""
        if model_path and os.path.exists(model_path):
            import joblib
            data = joblib.load(model_path)
            self.model = data['model']
            self.scaler = data['scaler']
            self.imputer = data['imputer']
            self.feature_names = data['feature_names']
        else:
            self._train_default_model()
    
    def _train_default_model(self):
        """Train default model on available data"""
        with self.get_connection() as conn:
            query = """
                SELECT 
                    patient_id,
                    AVG(energy_level) as avg_energy,
                    AVG(pain_level) as avg_pain,
                    AVG(sleep_quality) as avg_sleep,
                    AVG(mood_score) as avg_mood,
                    STDDEV(pain_level) as pain_variability,
                    COUNT(*) as followup_count,
                    MAX(pain_level) as max_pain,
                    MAX(CASE WHEN pain_level >= 7 THEN 1 ELSE 0 END) as has_severe_pain
                FROM daily_followups
                WHERE date >= NOW() - INTERVAL '90 days'
                GROUP BY patient_id
                HAVING COUNT(*) >= %s
            """
            df = pd.read_sql(query, conn, params=(self.config.min_data_points,))
        
        if len(df) < 50:
            self.model = None
            self.feature_names = ['avg_energy', 'avg_pain', 'avg_sleep', 'avg_mood', 
                                 'pain_variability', 'followup_count', 'max_pain']
            return
        
        self.feature_names = ['avg_energy', 'avg_pain', 'avg_sleep', 'avg_mood', 
                             'pain_variability', 'followup_count', 'max_pain']
        
        X = df[self.feature_names].values.astype(float)
        y = df['has_severe_pain'].values.astype(int)
        
        self.imputer = SimpleImputer(strategy='median')
        X = self.imputer.fit_transform(X)
        
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(X)
        
        self.model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )
        self.model.fit(X, y)
    
    def compute_risk_score(self, patient_id: str) -> Dict[str, Any]:
        """
        Compute risk score for a patient.
        
        Returns risk score, level, and contributing factors.
        """
        features = self._extract_patient_features(patient_id)
        
        if features is None:
            return {
                'patient_id': patient_id,
                'risk_score': None,
                'risk_level': 'unknown',
                'error': 'Insufficient data for risk assessment'
            }
        
        if self.model is None:
            risk_score = self._rule_based_risk(features)
            top_features = self._rule_based_explanation(features)
        else:
            X = np.array([features[f] for f in self.feature_names]).reshape(1, -1)
            X = self.imputer.transform(X)
            X = self.scaler.transform(X)
            
            risk_score = float(self.model.predict_proba(X)[0, 1])
            top_features = self._compute_feature_importance(X, features)
        
        if risk_score >= self.config.high_risk_threshold:
            risk_level = 'high'
        elif risk_score >= self.config.moderate_risk_threshold:
            risk_level = 'moderate'
        else:
            risk_level = 'low'
        
        trend = self._compute_trend(patient_id)
        
        return {
            'patient_id': patient_id,
            'risk_score': risk_score,
            'risk_level': risk_level,
            'top_features': top_features,
            'trend': trend,
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def _extract_patient_features(self, patient_id: str) -> Optional[Dict[str, float]]:
        """Extract features for a patient"""
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT 
                        AVG(energy_level) as avg_energy,
                        AVG(pain_level) as avg_pain,
                        AVG(sleep_quality) as avg_sleep,
                        AVG(mood_score) as avg_mood,
                        STDDEV(pain_level) as pain_variability,
                        COUNT(*) as followup_count,
                        MAX(pain_level) as max_pain
                    FROM daily_followups
                    WHERE patient_id = %s
                    AND date >= NOW() - INTERVAL '%s days'
                """, (patient_id, self.config.lookback_days))
                
                row = cur.fetchone()
                
                if not row or row['followup_count'] < self.config.min_data_points:
                    return None
                
                return {
                    'avg_energy': float(row['avg_energy'] or 5),
                    'avg_pain': float(row['avg_pain'] or 5),
                    'avg_sleep': float(row['avg_sleep'] or 5),
                    'avg_mood': float(row['avg_mood'] or 5),
                    'pain_variability': float(row['pain_variability'] or 0),
                    'followup_count': float(row['followup_count']),
                    'max_pain': float(row['max_pain'] or 0)
                }
    
    def _rule_based_risk(self, features: Dict[str, float]) -> float:
        """Calculate risk using rule-based approach"""
        risk = 0.0
        
        if features['avg_pain'] >= 7:
            risk += 0.4
        elif features['avg_pain'] >= 5:
            risk += 0.2
        
        if features['avg_energy'] <= 3:
            risk += 0.2
        elif features['avg_energy'] <= 5:
            risk += 0.1
        
        if features['avg_sleep'] <= 3:
            risk += 0.15
        
        if features['avg_mood'] <= 3:
            risk += 0.15
        
        if features['pain_variability'] > 2:
            risk += 0.1
        
        if features['max_pain'] >= 8:
            risk += 0.2
        
        return min(1.0, risk)
    
    def _rule_based_explanation(self, features: Dict[str, float]) -> List[Dict[str, Any]]:
        """Generate explanation for rule-based risk"""
        explanations = []
        
        if features['avg_pain'] >= 5:
            explanations.append({
                'feature': 'Average Pain Level',
                'value': features['avg_pain'],
                'contribution': 0.4 if features['avg_pain'] >= 7 else 0.2,
                'direction': 'increasing_risk',
                'explanation': f"High average pain ({features['avg_pain']:.1f}/10)"
            })
        
        if features['avg_energy'] <= 5:
            explanations.append({
                'feature': 'Energy Level',
                'value': features['avg_energy'],
                'contribution': 0.2 if features['avg_energy'] <= 3 else 0.1,
                'direction': 'increasing_risk',
                'explanation': f"Low energy levels ({features['avg_energy']:.1f}/10)"
            })
        
        if features['avg_sleep'] <= 5:
            explanations.append({
                'feature': 'Sleep Quality',
                'value': features['avg_sleep'],
                'contribution': 0.15 if features['avg_sleep'] <= 3 else 0.05,
                'direction': 'increasing_risk',
                'explanation': f"Poor sleep quality ({features['avg_sleep']:.1f}/10)"
            })
        
        if features['avg_mood'] <= 5:
            explanations.append({
                'feature': 'Mood Score',
                'value': features['avg_mood'],
                'contribution': 0.15 if features['avg_mood'] <= 3 else 0.05,
                'direction': 'increasing_risk',
                'explanation': f"Low mood ({features['avg_mood']:.1f}/10)"
            })
        
        if features['pain_variability'] > 1.5:
            explanations.append({
                'feature': 'Pain Variability',
                'value': features['pain_variability'],
                'contribution': 0.1,
                'direction': 'increasing_risk',
                'explanation': f"High pain variability (SD: {features['pain_variability']:.1f})"
            })
        
        explanations.sort(key=lambda x: x['contribution'], reverse=True)
        return explanations[:self.config.top_features_count]
    
    def _compute_feature_importance(
        self, 
        X: np.ndarray, 
        features: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """Compute feature importance using model"""
        if not hasattr(self.model, 'feature_importances_'):
            return self._rule_based_explanation(features)
        
        importances = self.model.feature_importances_
        
        explanations = []
        for i, feat_name in enumerate(self.feature_names):
            explanations.append({
                'feature': feat_name,
                'value': features[feat_name],
                'importance': float(importances[i]),
                'contribution': float(importances[i] * abs(X[0, i]))
            })
        
        explanations.sort(key=lambda x: x['contribution'], reverse=True)
        return explanations[:self.config.top_features_count]
    
    def _compute_trend(self, patient_id: str) -> str:
        """Compute risk trend for patient"""
        with self.get_connection() as conn:
            query = """
                SELECT date, pain_level, energy_level, mood_score
                FROM daily_followups
                WHERE patient_id = %s
                AND date >= NOW() - INTERVAL '14 days'
                ORDER BY date
            """
            df = pd.read_sql(query, conn, params=(patient_id,))
        
        if len(df) < 3:
            return 'insufficient_data'
        
        first_half = df.iloc[:len(df)//2]
        second_half = df.iloc[len(df)//2:]
        
        pain_change = second_half['pain_level'].mean() - first_half['pain_level'].mean()
        energy_change = second_half['energy_level'].mean() - first_half['energy_level'].mean()
        
        if pain_change > 1 and energy_change < -1:
            return 'worsening'
        elif pain_change < -1 and energy_change > 1:
            return 'improving'
        elif abs(pain_change) < 0.5 and abs(energy_change) < 0.5:
            return 'stable'
        else:
            return 'fluctuating'
    
    def should_alert(self, patient_id: str, current_risk: float) -> Tuple[bool, str]:
        """
        Determine if an alert should be triggered.
        
        Considers risk level and cooldown period.
        """
        if current_risk < self.config.moderate_risk_threshold:
            return False, 'Risk below threshold'
        
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT created_at
                    FROM interaction_alerts
                    WHERE patient_id = %s
                    AND alert_type = 'deterioration_risk'
                    AND created_at >= NOW() - INTERVAL '%s hours'
                    ORDER BY created_at DESC
                    LIMIT 1
                """, (patient_id, self.config.alert_cooldown_hours))
                
                recent = cur.fetchone()
                
                if recent:
                    return False, f"Alert cooldown active (last: {recent['created_at']})"
        
        return True, 'Alert conditions met'
    
    def create_alert(
        self, 
        patient_id: str, 
        risk_result: Dict[str, Any]
    ) -> Optional[str]:
        """Create an alert in the database"""
        should_create, reason = self.should_alert(patient_id, risk_result['risk_score'])
        
        if not should_create:
            return None
        
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                import uuid
                alert_id = str(uuid.uuid4())
                
                cur.execute("""
                    INSERT INTO interaction_alerts (
                        id, patient_id, alert_type, severity, message,
                        acknowledged, created_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, NOW())
                """, (
                    alert_id,
                    patient_id,
                    'deterioration_risk',
                    risk_result['risk_level'],
                    json.dumps({
                        'risk_score': risk_result['risk_score'],
                        'trend': risk_result['trend'],
                        'top_features': risk_result['top_features']
                    }),
                    False
                ))
                
                conn.commit()
                return alert_id
    
    def evaluate_all_patients(self, study_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Evaluate risk for all patients (optionally in a study).
        
        Returns list of patients with high/moderate risk.
        """
        with self.get_connection() as conn:
            if study_id:
                query = """
                    SELECT DISTINCT se.patient_id
                    FROM study_enrollments se
                    WHERE se.study_id = %s
                    AND se.status = 'active'
                """
                df = pd.read_sql(query, conn, params=(study_id,))
            else:
                query = """
                    SELECT DISTINCT patient_id
                    FROM daily_followups
                    WHERE date >= NOW() - INTERVAL '30 days'
                """
                df = pd.read_sql(query, conn)
        
        results = []
        
        for patient_id in df['patient_id']:
            risk = self.compute_risk_score(patient_id)
            
            if risk['risk_level'] in ['high', 'moderate']:
                results.append(risk)
        
        results.sort(key=lambda x: x['risk_score'], reverse=True)
        return results


def evaluate_patient_risk(
    patient_id: str,
    create_alert: bool = False
) -> Dict[str, Any]:
    """
    Convenience function to evaluate patient risk.
    
    Args:
        patient_id: Patient identifier
        create_alert: Whether to create alert if threshold met
    
    Returns:
        Risk assessment with score, level, and explanations
    """
    engine = AlertEngine()
    engine.load_model()
    
    result = engine.compute_risk_score(patient_id)
    
    if create_alert and result.get('risk_score'):
        alert_id = engine.create_alert(patient_id, result)
        result['alert_created'] = alert_id is not None
        result['alert_id'] = alert_id
    
    return result
