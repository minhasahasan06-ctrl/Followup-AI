"""
Risk Prediction Analysis Module for Research Center

Implements logistic regression, XGBoost/LightGBM models with 
AUROC/AUPRC, calibration metrics, and feature importance.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve,
    precision_recall_curve, brier_score_loss, log_loss,
    confusion_matrix, classification_report, calibration_curve
)
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

@dataclass
class RiskPredictionConfig:
    """Configuration for risk prediction analysis"""
    model_type: str = 'logistic_regression'  # 'logistic_regression', 'random_forest', 'gradient_boosting'
    outcome_variable: str = 'outcome'
    feature_variables: Optional[List[str]] = None
    n_folds: int = 5
    random_state: int = 42
    include_calibration: bool = True
    include_feature_importance: bool = True

class RiskPredictionAnalysis:
    """
    Risk prediction modeling for binary outcomes.
    
    Features:
    - Multiple model types (logistic, random forest, gradient boosting)
    - Cross-validation with stratification
    - AUROC and AUPRC metrics
    - Calibration curves and Brier score
    - Feature importance and coefficients
    - Predicted probability distributions
    """
    
    def __init__(self, df: pd.DataFrame, config: Optional[RiskPredictionConfig] = None):
        self.df = df
        self.config = config or RiskPredictionConfig()
        self.model = None
        self.scaler = None
        self.imputer = None
        self.feature_names = None
        
    def prepare_data(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare features and outcome for modeling"""
        if self.config.feature_variables:
            features = [f for f in self.config.feature_variables if f in self.df.columns]
        else:
            exclude = ['patient_id', self.config.outcome_variable, 'time', 'event']
            features = [c for c in self.df.columns 
                       if c not in exclude and pd.api.types.is_numeric_dtype(self.df[c])]
        
        self.feature_names = features
        
        X = self.df[features].values.astype(float)
        y = self.df[self.config.outcome_variable].values.astype(int)
        
        valid_mask = ~np.isnan(y)
        X = X[valid_mask]
        y = y[valid_mask]
        
        self.imputer = SimpleImputer(strategy='median')
        X = self.imputer.fit_transform(X)
        
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(X)
        
        return X, y, features
    
    def get_model(self):
        """Get model instance based on configuration"""
        if self.config.model_type == 'logistic_regression':
            return LogisticRegression(
                max_iter=1000,
                random_state=self.config.random_state,
                penalty='l2',
                C=1.0
            )
        elif self.config.model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=self.config.random_state,
                n_jobs=-1
            )
        elif self.config.model_type == 'gradient_boosting':
            return GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=self.config.random_state
            )
        else:
            raise ValueError(f"Unknown model type: {self.config.model_type}")
    
    def run_analysis(self) -> Dict[str, Any]:
        """Run complete risk prediction analysis"""
        X, y, features = self.prepare_data()
        
        if len(X) < 50:
            return {'error': 'Insufficient sample size (need >= 50)'}
        
        if y.sum() < 10 or (len(y) - y.sum()) < 10:
            return {'error': 'Insufficient events or non-events (need >= 10 each)'}
        
        cv = StratifiedKFold(n_splits=self.config.n_folds, shuffle=True, 
                            random_state=self.config.random_state)
        
        model = self.get_model()
        y_pred_proba = cross_val_predict(model, X, y, cv=cv, method='predict_proba')[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        model.fit(X, y)
        self.model = model
        
        results = {
            'model_type': self.config.model_type,
            'n_samples': len(y),
            'n_events': int(y.sum()),
            'event_rate': float(y.mean()),
            'n_features': len(features),
            'features': features,
            'n_folds': self.config.n_folds
        }
        
        results['metrics'] = self._compute_metrics(y, y_pred, y_pred_proba)
        
        results['roc_curve'] = self._compute_roc_curve(y, y_pred_proba)
        results['pr_curve'] = self._compute_pr_curve(y, y_pred_proba)
        
        if self.config.include_calibration:
            results['calibration'] = self._compute_calibration(y, y_pred_proba)
        
        if self.config.include_feature_importance:
            results['feature_importance'] = self._compute_feature_importance(features)
        
        results['confusion_matrix'] = self._compute_confusion_matrix(y, y_pred)
        
        results['risk_distribution'] = self._compute_risk_distribution(y_pred_proba)
        
        return results
    
    def _compute_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        y_pred_proba: np.ndarray
    ) -> Dict[str, float]:
        """Compute discrimination and calibration metrics"""
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        return {
            'auroc': float(roc_auc_score(y_true, y_pred_proba)),
            'auprc': float(average_precision_score(y_true, y_pred_proba)),
            'brier_score': float(brier_score_loss(y_true, y_pred_proba)),
            'log_loss': float(log_loss(y_true, y_pred_proba)),
            'sensitivity': float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
            'specificity': float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
            'ppv': float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0,
            'npv': float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0,
            'accuracy': float((tp + tn) / len(y_true))
        }
    
    def _compute_roc_curve(
        self, 
        y_true: np.ndarray, 
        y_pred_proba: np.ndarray
    ) -> Dict[str, Any]:
        """Compute ROC curve points"""
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        
        idx = np.argmax(tpr - fpr)
        optimal_threshold = float(thresholds[idx])
        
        return {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'thresholds': thresholds.tolist(),
            'optimal_threshold': optimal_threshold,
            'optimal_sensitivity': float(tpr[idx]),
            'optimal_specificity': float(1 - fpr[idx])
        }
    
    def _compute_pr_curve(
        self, 
        y_true: np.ndarray, 
        y_pred_proba: np.ndarray
    ) -> Dict[str, Any]:
        """Compute Precision-Recall curve"""
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        
        return {
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'thresholds': thresholds.tolist() if len(thresholds) > 0 else []
        }
    
    def _compute_calibration(
        self, 
        y_true: np.ndarray, 
        y_pred_proba: np.ndarray
    ) -> Dict[str, Any]:
        """Compute calibration curve and metrics"""
        prob_true, prob_pred = calibration_curve(y_true, y_pred_proba, n_bins=10)
        
        mean_predicted = y_pred_proba.mean()
        mean_observed = y_true.mean()
        calibration_ratio = mean_predicted / mean_observed if mean_observed > 0 else np.nan
        
        return {
            'prob_true': prob_true.tolist(),
            'prob_pred': prob_pred.tolist(),
            'calibration_slope': self._compute_calibration_slope(y_true, y_pred_proba),
            'calibration_intercept': self._compute_calibration_intercept(y_true, y_pred_proba),
            'calibration_ratio': float(calibration_ratio) if not np.isnan(calibration_ratio) else None
        }
    
    def _compute_calibration_slope(
        self, 
        y_true: np.ndarray, 
        y_pred_proba: np.ndarray
    ) -> float:
        """Compute calibration slope using logistic regression"""
        try:
            from sklearn.linear_model import LogisticRegression
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                log_odds = np.log(y_pred_proba / (1 - y_pred_proba + 1e-10))
                lr = LogisticRegression(penalty=None, max_iter=1000)
                lr.fit(log_odds.reshape(-1, 1), y_true)
                return float(lr.coef_[0][0])
        except:
            return np.nan
    
    def _compute_calibration_intercept(
        self, 
        y_true: np.ndarray, 
        y_pred_proba: np.ndarray
    ) -> float:
        """Compute calibration intercept"""
        try:
            mean_pred = np.log(y_pred_proba.mean() / (1 - y_pred_proba.mean() + 1e-10))
            mean_obs = np.log(y_true.mean() / (1 - y_true.mean() + 1e-10))
            return float(mean_obs - mean_pred)
        except:
            return np.nan
    
    def _compute_feature_importance(self, features: List[str]) -> List[Dict[str, Any]]:
        """Compute feature importance from fitted model"""
        importance = []
        
        if hasattr(self.model, 'coef_'):
            coefs = self.model.coef_[0]
            for i, feat in enumerate(features):
                importance.append({
                    'feature': feat,
                    'coefficient': float(coefs[i]),
                    'abs_importance': float(abs(coefs[i])),
                    'odds_ratio': float(np.exp(coefs[i]))
                })
        elif hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            for i, feat in enumerate(features):
                importance.append({
                    'feature': feat,
                    'importance': float(importances[i]),
                    'abs_importance': float(importances[i])
                })
        
        importance.sort(key=lambda x: x['abs_importance'], reverse=True)
        
        return importance
    
    def _compute_confusion_matrix(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray
    ) -> Dict[str, int]:
        """Compute confusion matrix elements"""
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return {
            'true_positive': int(tp),
            'true_negative': int(tn),
            'false_positive': int(fp),
            'false_negative': int(fn)
        }
    
    def _compute_risk_distribution(
        self, 
        y_pred_proba: np.ndarray
    ) -> Dict[str, Any]:
        """Compute risk score distribution"""
        percentiles = [10, 25, 50, 75, 90]
        
        return {
            'mean': float(y_pred_proba.mean()),
            'std': float(y_pred_proba.std()),
            'min': float(y_pred_proba.min()),
            'max': float(y_pred_proba.max()),
            'percentiles': {
                str(p): float(np.percentile(y_pred_proba, p)) for p in percentiles
            },
            'histogram': self._compute_histogram(y_pred_proba)
        }
    
    def _compute_histogram(
        self, 
        y_pred_proba: np.ndarray, 
        n_bins: int = 20
    ) -> Dict[str, List]:
        """Compute histogram of predicted probabilities"""
        hist, bin_edges = np.histogram(y_pred_proba, bins=n_bins, range=(0, 1))
        return {
            'counts': hist.tolist(),
            'bin_edges': bin_edges.tolist()
        }
    
    def predict(self, X_new: pd.DataFrame) -> np.ndarray:
        """Predict risk for new data"""
        if self.model is None:
            raise ValueError("Model not fitted. Call run_analysis() first.")
        
        X = X_new[self.feature_names].values.astype(float)
        X = self.imputer.transform(X)
        X = self.scaler.transform(X)
        
        return self.model.predict_proba(X)[:, 1]


def run_risk_prediction(
    df: pd.DataFrame,
    outcome_variable: str,
    feature_variables: Optional[List[str]] = None,
    model_type: str = 'logistic_regression',
    n_folds: int = 5
) -> Dict[str, Any]:
    """
    Convenience function to run risk prediction analysis.
    
    Args:
        df: DataFrame with features and outcome
        outcome_variable: Name of binary outcome column
        feature_variables: List of feature columns (None = auto-detect)
        model_type: 'logistic_regression', 'random_forest', 'gradient_boosting'
        n_folds: Number of CV folds
    
    Returns:
        Dictionary with model results, metrics, and diagnostics
    """
    config = RiskPredictionConfig(
        outcome_variable=outcome_variable,
        feature_variables=feature_variables,
        model_type=model_type,
        n_folds=n_folds
    )
    
    analysis = RiskPredictionAnalysis(df, config)
    return analysis.run_analysis()
