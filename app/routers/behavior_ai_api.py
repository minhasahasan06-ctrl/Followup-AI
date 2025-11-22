"""
Behavior AI Analysis API
========================

FastAPI endpoints for Behavior AI system:
- Check-in submission
- Metrics calculation
- Risk scoring
- Trend detection
- Cognitive testing
- Sentiment analysis
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from sqlalchemy import and_

from app.database import get_db
from app.models.behavior_models import (
    BehaviorCheckin, BehaviorMetric, DigitalBiomarker,
    CognitiveTest, SentimentAnalysis, BehaviorRiskScore,
    DeteriorationTrend, BehaviorAlert
)
from app.services.behavioral_metrics_service import BehavioralMetricsService
from app.services.digital_biomarkers_service import DigitalBiomarkersService
from app.services.cognitive_test_service import CognitiveTestService
from app.services.sentiment_analysis_service import SentimentAnalysisService
from app.services.risk_scoring_engine import RiskScoringEngine
from app.services.deterioration_trend_engine import DeteriorationTrendEngine
from app.services.medication_adherence_service import MedicationAdherenceService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/behavior-ai", tags=["behavior-ai"])


# ===========================================================================================
# REQUEST/RESPONSE MODELS
# ===========================================================================================

class CheckinSubmission(BaseModel):
    """Daily check-in submission"""
    patient_id: str
    scheduled_time: datetime
    symptom_severity: Optional[int] = Field(None, ge=1, le=10)
    symptom_description: Optional[str] = None
    pain_level: Optional[int] = Field(None, ge=1, le=10)
    medication_taken: bool = False
    medication_skipped_reason: Optional[str] = None
    session_duration_seconds: Optional[int] = None


class DigitalBiomarkerSubmission(BaseModel):
    """Daily digital biomarker data"""
    patient_id: str
    date: datetime
    raw_data: Dict[str, Any] = Field(
        ...,
        description="Raw sensor data: step_count, hourly_steps, hourly_activity, accelerometer, screen_on_events, etc."
    )


class CognitiveTestSubmission(BaseModel):
    """Cognitive test results"""
    patient_id: str
    test_type: str = Field(..., description="reaction_time, tapping, memory, pattern_recall, instruction_follow")
    started_at: datetime
    completed_at: Optional[datetime] = None
    raw_results: Dict[str, Any]


class SentimentAnalysisRequest(BaseModel):
    """Request for sentiment analysis"""
    patient_id: str
    text: str
    source_type: str = Field(..., description="checkin, symptom_journal, chat, audio_transcript")
    source_id: Optional[str] = None


# ===========================================================================================
# ENDPOINTS
# ===========================================================================================

@router.post("/checkins")
async def submit_checkin(
    checkin: CheckinSubmission,
    db: Session = Depends(get_db)
):
    """
    Submit daily check-in
    
    Performs:
    1. Stores check-in data
    2. Detects avoidance language
    3. Analyzes sentiment (if text provided)
    4. Calculates latency
    """
    
    logger.info(f"Received check-in for patient {checkin.patient_id}")
    
    # Calculate response latency
    completed_at = datetime.utcnow()
    latency_minutes = int((completed_at - checkin.scheduled_time).total_seconds() / 60)
    
    # Detect avoidance language
    behavioral_service = BehavioralMetricsService(db)
    skip_reason = checkin.medication_skipped_reason or ""
    symptom_desc = checkin.symptom_description or ""
    combined_text = f"{skip_reason} {symptom_desc}"
    
    avoidance_result = behavioral_service.detect_avoidance_patterns(combined_text)
    
    # Analyze sentiment
    sentiment_polarity = None
    if combined_text.strip():
        sentiment_service = SentimentAnalysisService(db)
        sentiment_result = sentiment_service.analyze_text(
            patient_id=checkin.patient_id,
            text=combined_text,
            source_type='checkin'
        )
        sentiment_polarity = sentiment_result.get('polarity')
        
        # Store sentiment analysis
        sentiment_record = SentimentAnalysis(
            patient_id=checkin.patient_id,
            source_type='checkin',
            text_content=combined_text,
            analyzed_at=datetime.utcnow(),
            sentiment_polarity=sentiment_polarity,
            sentiment_label=sentiment_result.get('label'),
            sentiment_confidence=sentiment_result.get('confidence'),
            message_length_chars=sentiment_result.get('message_length'),
            word_count=sentiment_result.get('word_count'),
            lexical_complexity=sentiment_result.get('lexical_complexity'),
            negativity_ratio=sentiment_result.get('negativity_ratio'),
            stress_keyword_count=sentiment_result.get('stress_keyword_count'),
            stress_keywords=sentiment_result.get('stress_keywords'),
            help_seeking_detected=sentiment_result.get('help_seeking_detected'),
            help_seeking_phrases=sentiment_result.get('help_seeking_phrases'),
            hesitation_count=sentiment_result.get('hesitation_count'),
            hesitation_markers=sentiment_result.get('hesitation_markers')
        )
        db.add(sentiment_record)
    
    # Create check-in record
    checkin_record = BehaviorCheckin(
        patient_id=checkin.patient_id,
        scheduled_time=checkin.scheduled_time,
        completed_at=completed_at,
        response_latency_minutes=latency_minutes,
        skipped=False,
        symptom_severity=checkin.symptom_severity,
        symptom_description=checkin.symptom_description,
        pain_level=checkin.pain_level,
        medication_taken=checkin.medication_taken,
        medication_skipped_reason=checkin.medication_skipped_reason,
        session_duration_seconds=checkin.session_duration_seconds,
        avoidance_language_detected=avoidance_result['detected'],
        avoidance_phrases=avoidance_result['phrases'],
        sentiment_polarity=sentiment_polarity
    )
    
    db.add(checkin_record)
    db.commit()
    db.refresh(checkin_record)
    
    logger.info(f"✅ Check-in stored for patient {checkin.patient_id}")
    
    return {
        "status": "success",
        "checkin_id": checkin_record.id,
        "avoidance_detected": avoidance_result['detected'],
        "sentiment_polarity": sentiment_polarity
    }


@router.post("/digital-biomarkers")
async def submit_digital_biomarkers(
    data: DigitalBiomarkerSubmission,
    db: Session = Depends(get_db)
):
    """
    Submit daily digital biomarker data
    
    Calculates:
    - Activity metrics (steps, bursts, sedentary time)
    - Circadian rhythm stability
    - Phone usage patterns
    - Mobility changes
    """
    
    logger.info(f"Processing digital biomarkers for patient {data.patient_id}")
    
    biomarker_service = DigitalBiomarkersService(db)
    
    # Calculate biomarkers
    biomarkers = biomarker_service.calculate_daily_biomarkers(
        patient_id=data.patient_id,
        date=data.date,
        raw_data=data.raw_data
    )
    
    # Store biomarker record
    biomarker_record = DigitalBiomarker(
        patient_id=data.patient_id,
        date=data.date,
        **biomarkers
    )
    
    db.add(biomarker_record)
    db.commit()
    db.refresh(biomarker_record)
    
    logger.info(f"✅ Digital biomarkers stored for patient {data.patient_id}")
    
    return {
        "status": "success",
        "biomarker_id": biomarker_record.id,
        "mobility_drop_detected": biomarkers.get('mobility_drop_detected', False),
        "circadian_stability": biomarkers.get('circadian_rhythm_stability')
    }


@router.post("/cognitive-tests")
async def submit_cognitive_test(
    test: CognitiveTestSubmission,
    db: Session = Depends(get_db)
):
    """
    Submit cognitive test results
    
    Scores test and detects anomalies vs patient baseline
    """
    
    logger.info(f"Processing cognitive test for patient {test.patient_id}")
    
    cognitive_service = CognitiveTestService(db)
    
    # Score test
    scored_results = cognitive_service.score_test_results(
        patient_id=test.patient_id,
        test_type=test.test_type,
        raw_results=test.raw_results
    )
    
    # Duration
    duration_seconds = None
    if test.completed_at:
        duration_seconds = int((test.completed_at - test.started_at).total_seconds())
    
    # Store test record
    test_record = CognitiveTest(
        patient_id=test.patient_id,
        test_type=test.test_type,
        started_at=test.started_at,
        completed_at=test.completed_at,
        duration_seconds=duration_seconds,
        reaction_time_ms=scored_results.get('reaction_time_ms'),
        tapping_speed=scored_results.get('tapping_speed'),
        error_rate=scored_results.get('error_rate'),
        memory_score=scored_results.get('memory_score'),
        pattern_recall_accuracy=scored_results.get('pattern_recall_accuracy'),
        instruction_accuracy=scored_results.get('instruction_accuracy'),
        raw_results=scored_results.get('raw_results'),
        baseline_deviation=scored_results.get('baseline_deviation'),
        anomaly_detected=scored_results.get('anomaly_detected')
    )
    
    db.add(test_record)
    db.commit()
    db.refresh(test_record)
    
    logger.info(f"✅ Cognitive test scored for patient {test.patient_id}")
    
    return {
        "status": "success",
        "test_id": test_record.id,
        "anomaly_detected": test_record.anomaly_detected,
        "baseline_deviation": float(test_record.baseline_deviation) if test_record.baseline_deviation is not None else None
    }


@router.post("/sentiment-analysis")
async def analyze_sentiment(
    request: SentimentAnalysisRequest,
    db: Session = Depends(get_db)
):
    """
    Analyze text for sentiment and language biomarkers
    """
    
    logger.info(f"Analyzing sentiment for patient {request.patient_id}")
    
    sentiment_service = SentimentAnalysisService(db)
    
    # Analyze
    analysis_result = sentiment_service.analyze_text(
        patient_id=request.patient_id,
        text=request.text,
        source_type=request.source_type,
        source_id=request.source_id
    )
    
    # Store analysis
    analysis_record = SentimentAnalysis(
        patient_id=request.patient_id,
        source_type=request.source_type,
        source_id=request.source_id,
        text_content=request.text,
        analyzed_at=datetime.utcnow(),
        sentiment_polarity=analysis_result.get('polarity'),
        sentiment_label=analysis_result.get('label'),
        sentiment_confidence=analysis_result.get('confidence'),
        message_length_chars=analysis_result.get('message_length'),
        word_count=analysis_result.get('word_count'),
        lexical_complexity=analysis_result.get('lexical_complexity'),
        negativity_ratio=analysis_result.get('negativity_ratio'),
        stress_keyword_count=analysis_result.get('stress_keyword_count'),
        stress_keywords=analysis_result.get('stress_keywords'),
        help_seeking_detected=analysis_result.get('help_seeking_detected'),
        help_seeking_phrases=analysis_result.get('help_seeking_phrases'),
        hesitation_count=analysis_result.get('hesitation_count'),
        hesitation_markers=analysis_result.get('hesitation_markers')
    )
    
    db.add(analysis_record)
    db.commit()
    db.refresh(analysis_record)
    
    logger.info(f"✅ Sentiment analyzed for patient {request.patient_id}")
    
    return {
        "status": "success",
        "analysis_id": analysis_record.id,
        "polarity": float(analysis_result['polarity']),
        "label": analysis_result['label'],
        "help_seeking_detected": analysis_result.get('help_seeking_detected', False)
    }


@router.get("/risk-score/{patient_id}")
async def get_risk_score(
    patient_id: str,
    lookback_days: int = 7,
    db: Session = Depends(get_db)
):
    """
    Calculate comprehensive risk score for patient
    
    Combines:
    - Behavioral metrics
    - Digital biomarkers
    - Cognitive tests
    - Sentiment analysis
    """
    
    logger.info(f"Calculating risk score for patient {patient_id}")
    
    risk_engine = RiskScoringEngine(db)
    
    # Calculate risk
    risk_assessment = risk_engine.calculate_risk_score(
        patient_id=patient_id,
        lookback_days=lookback_days
    )
    
    # Store risk score
    risk_record = BehaviorRiskScore(
        patient_id=patient_id,
        calculated_at=datetime.utcnow(),
        behavioral_risk=risk_assessment['behavioral_risk'],
        digital_biomarker_risk=risk_assessment['digital_biomarker_risk'],
        cognitive_risk=risk_assessment['cognitive_risk'],
        sentiment_risk=risk_assessment['sentiment_risk'],
        composite_risk=risk_assessment['composite_risk'],
        risk_level=risk_assessment['risk_level'],
        model_type=risk_assessment.get('model_type'),
        top_risk_factors=risk_assessment.get('top_risk_factors'),
        prediction_confidence=risk_assessment.get('prediction_confidence')
    )
    
    db.add(risk_record)
    db.commit()
    db.refresh(risk_record)
    
    logger.info(f"✅ Risk score calculated: {risk_assessment['risk_level']}")
    
    return {
        "status": "success",
        "risk_score_id": risk_record.id,
        **risk_assessment
    }


@router.get("/trends/{patient_id}")
async def detect_trends(
    patient_id: str,
    lookback_days: int = 14,
    db: Session = Depends(get_db)
):
    """
    Detect deterioration trends for patient
    
    Identifies:
    - Declining engagement
    - Mobility drops
    - Cognitive decline
    - Sentiment deterioration
    """
    
    logger.info(f"Detecting trends for patient {patient_id}")
    
    trend_engine = DeteriorationTrendEngine(db)
    
    # Detect trends
    trends = trend_engine.detect_all_trends(
        patient_id=patient_id,
        lookback_days=lookback_days
    )
    
    # Store trends
    trend_ids = []
    for trend_data in trends:
        trend_record = DeteriorationTrend(
            patient_id=patient_id,
            detected_at=datetime.utcnow(),
            trend_type=trend_data['trend_type'],
            severity=trend_data['severity'],
            trend_start_date=trend_data.get('trend_start_date'),
            trend_duration_days=trend_data.get('trend_duration_days'),
            trend_slope=trend_data.get('trend_slope'),
            z_score=trend_data.get('z_score'),
            p_value=trend_data.get('p_value'),
            confidence_level=trend_data.get('confidence_level'),
            affected_metrics=trend_data.get('affected_metrics'),
            metric_values=trend_data.get('metric_values'),
            clinical_significance=trend_data.get('clinical_significance'),
            recommended_actions=trend_data.get('recommended_actions'),
            alert_generated=False
        )
        
        db.add(trend_record)
        db.commit()
        db.refresh(trend_record)
        
        trend_ids.append(trend_record.id)
    
    logger.info(f"✅ Detected {len(trends)} trends for patient {patient_id}")
    
    return {
        "status": "success",
        "trends_detected": len(trends),
        "trend_ids": trend_ids,
        "trends": trends
    }


@router.get("/dashboard/{patient_id}")
async def get_dashboard_data(
    patient_id: str,
    days: int = 7,
    db: Session = Depends(get_db)
):
    """
    Get complete dashboard data for patient
    
    Returns:
    - Latest risk score
    - Recent trends
    - Key metrics
    - Alerts
    """
    
    # Get latest risk score
    latest_risk = db.query(BehaviorRiskScore).filter(
        BehaviorRiskScore.patient_id == patient_id
    ).order_by(BehaviorRiskScore.calculated_at.desc()).first()
    
    # Get recent trends
    cutoff_date = datetime.utcnow() - timedelta(days=days)
    recent_trends = db.query(DeteriorationTrend).filter(
        and_(
            DeteriorationTrend.patient_id == patient_id,
            DeteriorationTrend.detected_at >= cutoff_date
        )
    ).order_by(DeteriorationTrend.detected_at.desc()).all()
    
    # Get unresolved alerts
    unresolved_alerts = db.query(BehaviorAlert).filter(
        and_(
            BehaviorAlert.patient_id == patient_id,
            BehaviorAlert.resolved == False
        )
    ).order_by(BehaviorAlert.triggered_at.desc()).all()
    
    return {
        "status": "success",
        "risk_score": {
            "composite_risk": float(latest_risk.composite_risk) if latest_risk and latest_risk.composite_risk is not None else None,
            "risk_level": latest_risk.risk_level if latest_risk else None,
            "calculated_at": latest_risk.calculated_at.isoformat() if latest_risk else None
        } if latest_risk else None,
        "recent_trends": [
            {
                "id": t.id,
                "type": t.trend_type,
                "severity": t.severity,
                "detected_at": t.detected_at.isoformat(),
                "clinical_significance": t.clinical_significance
            }
            for t in recent_trends
        ],
        "unresolved_alerts": [
            {
                "id": a.id,
                "type": a.alert_type,
                "severity": a.severity,
                "title": a.title,
                "message": a.message,
                "triggered_at": a.triggered_at.isoformat()
            }
            for a in unresolved_alerts
        ]
    }


# ===========================================================================================
# MEDICATION ADHERENCE ENDPOINTS
# ===========================================================================================

class AdherenceTrendPoint(BaseModel):
    """Single point in adherence trend"""
    date: str
    adherenceRate: float


class RegimenRisk(BaseModel):
    """Regimen risk assessment"""
    level: str = Field(..., description="Risk level: low, moderate, high, unknown")
    rationale: str = Field(..., description="Explanation of risk level")


class MissedDoseEscalation(BaseModel):
    """Missed dose escalation data"""
    count: int = Field(..., description="Total missed doses")
    severity: str = Field(..., description="Severity: none, warning, critical")


class MedicationAdherenceResponse(BaseModel):
    """Comprehensive medication adherence analytics"""
    currentAdherenceRate: Optional[float] = Field(None, description="Current adherence rate (0.0 - 1.0)")
    sevenDayTrend: List[AdherenceTrendPoint] = Field(default_factory=list, description="7-day adherence trend")
    regimenRisk: RegimenRisk
    missedDoseEscalation: MissedDoseEscalation


@router.get("/medication-adherence/{patient_id}", response_model=MedicationAdherenceResponse)
async def get_medication_adherence(
    patient_id: str,
    db: Session = Depends(get_db)
):
    """
    Get comprehensive medication adherence analytics
    
    Returns:
    - Current adherence rate
    - 7-day adherence trend for sparkline visualization
    - Regimen risk analysis (low/moderate/high)
    - Missed dose escalation data
    """
    logger.info(f"[ADHERENCE] Fetching medication adherence for patient {patient_id}")
    
    try:
        service = MedicationAdherenceService(db)
        analytics = service.get_adherence_analytics(patient_id)
        
        return MedicationAdherenceResponse(**analytics)
    
    except Exception as e:
        logger.error(f"[ADHERENCE] Error fetching adherence: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching medication adherence: {str(e)}")
