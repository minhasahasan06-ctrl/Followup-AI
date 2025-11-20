"""
Behavior AI ML Models Infrastructure
====================================

Multi-model ensemble for deterioration detection:
1. Transformer Encoder (PyTorch) - Sequence modeling for temporal patterns
2. XGBoost - Feature-based risk prediction
3. DistilBERT - Sentiment analysis and language biomarkers
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class BehaviorMLModels:
    """
    Unified ML models manager for Behavior AI Analysis
    
    Components:
    - Transformer Encoder: Sequence modeling for time-series biomarkers
    - XGBoost: Feature-based deterioration prediction  
    - DistilBERT: Sentiment analysis for text inputs
    """
    
    def __init__(self):
        self.transformer_model = None
        self.xgboost_model = None
        self.distilbert_model = None
        self.distilbert_tokenizer = None
        
        self.models_loaded = False
        
        logger.info("🤖 Behavior ML Models initialized (lazy loading)")
    
    def load_models(self):
        """Load all ML models into memory"""
        if self.models_loaded:
            logger.info("Models already loaded")
            return
        
        logger.info("="*70)
        logger.info("LOADING BEHAVIOR AI ML MODELS")
        logger.info("="*70)
        
        try:
            # 1. Load Transformer Encoder for sequences
            self._load_transformer_encoder()
            
            # 2. Load XGBoost for feature prediction
            self._load_xgboost_model()
            
            # 3. Load DistilBERT for sentiment
            self._load_distilbert_model()
            
            self.models_loaded = True
            logger.info("✅ All Behavior AI models loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load Behavior AI models: {e}")
            raise
    
    def _load_transformer_encoder(self):
        """
        Load PyTorch Transformer Encoder for sequence modeling
        
        Architecture:
        - Input: Time-series of biomarker vectors (14-day windows)
        - Encoder: 4-layer transformer with multi-head attention
        - Output: Risk score + hidden states for explainability
        """
        logger.info("Loading Transformer Encoder (PyTorch)...")
        
        try:
            import torch
            import torch.nn as nn
            
            # Check if pre-trained model exists
            import os
            model_path = "./models/behavior_transformer.pth"
            
            if os.path.exists(model_path):
                logger.info(f"Loading pre-trained model from {model_path}")
                self.transformer_model = torch.load(model_path)
                self.transformer_model.eval()
                logger.info("✅ Transformer model loaded from checkpoint")
            else:
                logger.warning("⚠️  No pre-trained transformer found - using initialized model")
                # Create model architecture (would need training)
                self.transformer_model = self._build_transformer_architecture()
                logger.info("✅ Transformer architecture initialized (needs training)")
        
        except ImportError:
            logger.warning("⚠️  PyTorch not available - transformer disabled")
            self.transformer_model = None
    
    def _build_transformer_architecture(self):
        """Build transformer encoder architecture"""
        try:
            import torch
            import torch.nn as nn
            
            class BehaviorTransformer(nn.Module):
                def __init__(
                    self,
                    input_dim=50,  # Number of features per timestep
                    d_model=128,    # Model dimensionality
                    nhead=8,        # Number of attention heads
                    num_layers=4,   # Transformer layers
                    dim_feedforward=512,
                    dropout=0.1
                ):
                    super().__init__()
                    
                    # Input embedding
                    self.input_embedding = nn.Linear(input_dim, d_model)
                    
                    # Positional encoding for time-series
                    self.pos_encoder = nn.Parameter(torch.zeros(1, 14, d_model))  # 14-day window
                    
                    # Transformer encoder
                    encoder_layer = nn.TransformerEncoderLayer(
                        d_model=d_model,
                        nhead=nhead,
                        dim_feedforward=dim_feedforward,
                        dropout=dropout,
                        batch_first=True
                    )
                    self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
                    
                    # Output heads
                    self.risk_head = nn.Sequential(
                        nn.Linear(d_model, 64),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.Linear(64, 1),
                        nn.Sigmoid()  # Risk score 0-1
                    )
                    
                    self.d_model = d_model
                
                def forward(self, x):
                    """
                    Args:
                        x: (batch, seq_len, input_dim) - 14 days of biomarker vectors
                    
                    Returns:
                        risk_score: (batch, 1) - deterioration risk 0-1
                        hidden_states: (batch, seq_len, d_model) - for explainability
                    """
                    # Embed input
                    x = self.input_embedding(x)  # (batch, seq_len, d_model)
                    
                    # Add positional encoding
                    x = x + self.pos_encoder[:, :x.size(1), :]
                    
                    # Transformer encoding
                    hidden_states = self.transformer(x)  # (batch, seq_len, d_model)
                    
                    # Global pooling (use last timestep)
                    pooled = hidden_states[:, -1, :]  # (batch, d_model)
                    
                    # Risk prediction
                    risk_score = self.risk_head(pooled)  # (batch, 1)
                    
                    return risk_score, hidden_states
            
            model = BehaviorTransformer()
            return model
        
        except ImportError:
            logger.error("PyTorch not available - cannot build transformer")
            return None
    
    def _load_xgboost_model(self):
        """
        Load XGBoost model for feature-based risk prediction
        
        Features (50 total):
        - Behavioral metrics: 15 features
        - Digital biomarkers: 20 features
        - Cognitive scores: 10 features
        - Sentiment metrics: 5 features
        """
        logger.info("Loading XGBoost feature model...")
        
        try:
            import xgboost as xgb
            import os
            
            model_path = "./models/behavior_xgboost.json"
            
            if os.path.exists(model_path):
                logger.info(f"Loading pre-trained XGBoost from {model_path}")
                self.xgboost_model = xgb.Booster()
                self.xgboost_model.load_model(model_path)
                logger.info("✅ XGBoost model loaded from checkpoint")
            else:
                logger.warning("⚠️  No pre-trained XGBoost found - using default params")
                # Create model with default parameters (would need training)
                params = {
                    'objective': 'binary:logistic',
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'n_estimators': 100,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'eval_metric': 'logloss'
                }
                self.xgboost_model = xgb.Booster(params=params)
                logger.info("✅ XGBoost initialized (needs training)")
        
        except ImportError:
            logger.warning("⚠️  XGBoost not available - feature model disabled")
            self.xgboost_model = None
    
    def _load_distilbert_model(self):
        """
        Load DistilBERT for sentiment analysis
        
        Model: distilbert-base-uncased-finetuned-sst-2-english
        Outputs: Sentiment polarity (-1 to +1) and confidence
        """
        logger.info("Loading DistilBERT for sentiment analysis...")
        
        try:
            from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
            import torch
            
            model_name = "distilbert-base-uncased-finetuned-sst-2-english"
            
            logger.info(f"Loading {model_name} from HuggingFace...")
            self.distilbert_tokenizer = DistilBertTokenizer.from_pretrained(model_name)
            self.distilbert_model = DistilBertForSequenceClassification.from_pretrained(model_name)
            self.distilbert_model.eval()
            
            logger.info("✅ DistilBERT sentiment model loaded")
        
        except ImportError:
            logger.warning("⚠️  Transformers library not available - sentiment disabled")
            self.distilbert_model = None
            self.distilbert_tokenizer = None
    
    def predict_sequence_risk(
        self,
        sequence_data: np.ndarray
    ) -> Dict[str, Any]:
        """
        Predict deterioration risk from time-series sequence
        
        Args:
            sequence_data: (seq_len, feature_dim) array of biomarkers over time
        
        Returns:
            {
                'risk_score': float (0-1),
                'attention_weights': array - for explainability,
                'model_type': 'transformer'
            }
        """
        if self.transformer_model is None:
            logger.warning("Transformer model not loaded - returning baseline risk")
            return {
                'risk_score': 0.5,
                'attention_weights': None,
                'model_type': 'baseline'
            }
        
        try:
            import torch
            
            # Convert to tensor and add batch dimension
            x = torch.FloatTensor(sequence_data).unsqueeze(0)  # (1, seq_len, feature_dim)
            
            with torch.no_grad():
                risk_score, hidden_states = self.transformer_model(x)
            
            risk = float(risk_score.item())
            
            return {
                'risk_score': risk,
                'attention_weights': hidden_states.numpy()[0],  # Remove batch dim
                'model_type': 'transformer'
            }
        
        except Exception as e:
            logger.error(f"Transformer prediction failed: {e}")
            return {
                'risk_score': 0.5,
                'attention_weights': None,
                'model_type': 'error'
            }
    
    def predict_feature_risk(
        self,
        features: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Predict risk from feature vector using XGBoost
        
        Args:
            features: Dictionary of feature name -> value
        
        Returns:
            {
                'risk_score': float (0-1),
                'feature_importance': dict - top contributing features,
                'model_type': 'xgboost'
            }
        """
        if self.xgboost_model is None:
            logger.warning("XGBoost model not loaded - using rule-based risk")
            return self._rule_based_risk_estimation(features)
        
        try:
            import xgboost as xgb
            
            # Extract feature values in correct order
            feature_names = sorted(features.keys())
            feature_vector = np.array([features[name] for name in feature_names])
            
            # Create DMatrix
            dmatrix = xgb.DMatrix(feature_vector.reshape(1, -1), feature_names=feature_names)
            
            # Predict
            risk_score = float(self.xgboost_model.predict(dmatrix)[0])
            
            # Get feature importance (SHAP values would be better)
            importance = self.xgboost_model.get_score(importance_type='weight')
            
            return {
                'risk_score': risk_score,
                'feature_importance': importance,
                'model_type': 'xgboost'
            }
        
        except Exception as e:
            logger.error(f"XGBoost prediction failed: {e}")
            return self._rule_based_risk_estimation(features)
    
    def analyze_sentiment(
        self,
        text: str
    ) -> Dict[str, Any]:
        """
        Analyze sentiment and extract language biomarkers
        
        Args:
            text: Input text (symptom description, check-in message, etc.)
        
        Returns:
            {
                'polarity': float (-1 to +1),
                'label': str ('positive', 'neutral', 'negative'),
                'confidence': float (0-1),
                'message_length': int,
                'word_count': int,
                'lexical_complexity': float,
                'negativity_ratio': float,
                'stress_keywords': list,
                'help_seeking_phrases': list,
                'hesitation_markers': list
            }
        """
        if self.distilbert_model is None or self.distilbert_tokenizer is None:
            logger.warning("DistilBERT not loaded - using rule-based sentiment")
            return self._rule_based_sentiment_analysis(text)
        
        try:
            import torch
            
            # Tokenize
            inputs = self.distilbert_tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )
            
            # Predict
            with torch.no_grad():
                outputs = self.distilbert_model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1)
            
            # DistilBERT SST-2: 0=negative, 1=positive
            neg_prob = float(probabilities[0][0].item())
            pos_prob = float(probabilities[0][1].item())
            
            # Convert to -1 to +1 polarity
            polarity = pos_prob - neg_prob
            
            # Determine label
            if polarity > 0.3:
                label = 'positive'
            elif polarity < -0.3:
                label = 'negative'
            else:
                label = 'neutral'
            
            confidence = max(pos_prob, neg_prob)
            
            # Extract language biomarkers
            biomarkers = self._extract_language_biomarkers(text)
            
            return {
                'polarity': polarity,
                'label': label,
                'confidence': confidence,
                **biomarkers
            }
        
        except Exception as e:
            logger.error(f"DistilBERT sentiment analysis failed: {e}")
            return self._rule_based_sentiment_analysis(text)
    
    def _extract_language_biomarkers(self, text: str) -> Dict[str, Any]:
        """Extract linguistic features from text"""
        
        words = text.lower().split()
        word_count = len(words)
        message_length = len(text)
        
        # Stress keywords
        stress_keywords = [
            'pain', 'hurt', 'tired', 'exhausted', 'weak', 'dizzy',
            'nauseous', 'worse', 'bad', 'terrible', 'awful', 'can\'t'
        ]
        found_stress = [kw for kw in stress_keywords if kw in text.lower()]
        
        # Help-seeking language
        help_phrases = [
            'help', 'need', 'doctor', 'emergency', 'urgent', 'worried',
            'concerned', 'scared', 'afraid'
        ]
        found_help = [phrase for phrase in help_phrases if phrase in text.lower()]
        
        # Hesitation markers
        hesitation_markers = [
            'maybe', 'idk', 'i don\'t know', 'i guess', 'i think',
            'probably', 'possibly', 'unsure', 'not sure'
        ]
        found_hesitation = [marker for marker in hesitation_markers if marker in text.lower()]
        
        # Negative words for negativity ratio
        negative_words = [
            'no', 'not', 'never', 'nothing', 'none', 'nobody', 'nowhere',
            'pain', 'hurt', 'bad', 'worse', 'worst', 'terrible', 'awful'
        ]
        neg_count = sum(1 for word in words if word in negative_words)
        negativity_ratio = neg_count / max(word_count, 1)
        
        # Lexical complexity (unique words / total words)
        unique_words = len(set(words))
        lexical_complexity = unique_words / max(word_count, 1)
        
        return {
            'message_length': message_length,
            'word_count': word_count,
            'lexical_complexity': lexical_complexity,
            'negativity_ratio': negativity_ratio,
            'stress_keyword_count': len(found_stress),
            'stress_keywords': found_stress,
            'help_seeking_detected': len(found_help) > 0,
            'help_seeking_phrases': found_help,
            'hesitation_count': len(found_hesitation),
            'hesitation_markers': found_hesitation
        }
    
    def _rule_based_risk_estimation(self, features: Dict[str, float]) -> Dict[str, Any]:
        """Fallback rule-based risk scoring when ML models unavailable"""
        
        risk_score = 0.0
        contributions = {}
        
        # Check-in consistency (weight: 0.2)
        if 'checkin_completion_rate' in features:
            completion = features['checkin_completion_rate']
            if completion < 0.5:
                risk_score += 0.2
                contributions['low_checkin_completion'] = 0.2
            elif completion < 0.7:
                risk_score += 0.1
                contributions['moderate_checkin_completion'] = 0.1
        
        # Medication adherence (weight: 0.25)
        if 'medication_adherence_rate' in features:
            adherence = features['medication_adherence_rate']
            if adherence < 0.6:
                risk_score += 0.25
                contributions['low_medication_adherence'] = 0.25
            elif adherence < 0.8:
                risk_score += 0.15
                contributions['moderate_medication_adherence'] = 0.15
        
        # Mobility drop (weight: 0.3)
        if features.get('mobility_drop_detected', False):
            risk_score += 0.3
            contributions['mobility_drop'] = 0.3
        
        # Sentiment decline (weight: 0.15)
        if 'avg_sentiment_polarity' in features:
            sentiment = features['avg_sentiment_polarity']
            if sentiment < -0.5:
                risk_score += 0.15
                contributions['negative_sentiment'] = 0.15
            elif sentiment < -0.2:
                risk_score += 0.08
                contributions['declining_sentiment'] = 0.08
        
        # Cognitive anomalies (weight: 0.1)
        if features.get('cognitive_anomaly_detected', False):
            risk_score += 0.1
            contributions['cognitive_anomaly'] = 0.1
        
        # Cap at 1.0
        risk_score = min(risk_score, 1.0)
        
        return {
            'risk_score': risk_score,
            'feature_importance': contributions,
            'model_type': 'rule_based'
        }
    
    def _rule_based_sentiment_analysis(self, text: str) -> Dict[str, Any]:
        """Fallback rule-based sentiment when DistilBERT unavailable"""
        
        biomarkers = self._extract_language_biomarkers(text)
        
        # Simple rule-based polarity
        polarity = 0.0
        
        # Negative impact
        polarity -= biomarkers['negativity_ratio'] * 2
        polarity -= biomarkers['stress_keyword_count'] * 0.1
        
        # Positive indicators (lack of negativity)
        if biomarkers['negativity_ratio'] < 0.1:
            polarity += 0.3
        
        # Clamp to -1 to +1
        polarity = max(-1.0, min(1.0, polarity))
        
        if polarity > 0.2:
            label = 'positive'
        elif polarity < -0.2:
            label = 'negative'
        else:
            label = 'neutral'
        
        return {
            'polarity': polarity,
            'label': label,
            'confidence': 0.6,  # Lower confidence for rule-based
            **biomarkers
        }


# Singleton instance
_behavior_ml_models_instance = None

def get_behavior_ml_models() -> BehaviorMLModels:
    """Get singleton instance of Behavior ML Models"""
    global _behavior_ml_models_instance
    
    if _behavior_ml_models_instance is None:
        _behavior_ml_models_instance = BehaviorMLModels()
    
    return _behavior_ml_models_instance
