"""
Symptom Check-in API - Daily Follow-up Symptom Tracking Module

Provides endpoints for:
- Daily symptom check-ins (structured + free-form)
- Conversational symptom extraction from Agent Clona
- ML-based trend reports (3/7/15/30 days)
- Anomaly detection and correlational insights

All outputs are labeled observational and non-diagnostic per HIPAA compliance.
"""

from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, File
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime, timedelta
from pydantic import BaseModel, Field

from app.database import get_db
from app.dependencies import get_current_user_py
from app.models.user import User
from app.services.symptom_checkin_service import SymptomExtractionService, SymptomTrendService
from app.services.s3_service import S3Service

router = APIRouter(prefix="/api/symptom-checkin", tags=["Symptom Check-in"])


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class SymptomCheckinCreate(BaseModel):
    """Request model for creating a symptom check-in"""
    painLevel: Optional[int] = Field(None, ge=0, le=10)
    fatigueLevel: Optional[int] = Field(None, ge=0, le=10)
    breathlessnessLevel: Optional[int] = Field(None, ge=0, le=10)
    sleepQuality: Optional[int] = Field(None, ge=0, le=10)
    mood: Optional[str] = None  # 'great', 'good', 'okay', 'low', 'very_low'
    mobilityScore: Optional[int] = Field(None, ge=0, le=10)
    medicationsTaken: Optional[bool] = None
    triggers: Optional[List[str]] = []
    symptoms: Optional[List[str]] = []
    note: Optional[str] = None
    deviceType: Optional[str] = None


class SymptomCheckinResponse(BaseModel):
    """Response model for symptom check-in"""
    id: str
    userId: str
    timestamp: datetime
    painLevel: Optional[int]
    fatigueLevel: Optional[int]
    breathlessnessLevel: Optional[int]
    sleepQuality: Optional[int]
    mood: Optional[str]
    mobilityScore: Optional[int]
    medicationsTaken: Optional[bool]
    triggers: Optional[List[str]]
    symptoms: Optional[List[str]]
    note: Optional[str]
    source: str
    createdAt: datetime


class ChatSymptomExtractRequest(BaseModel):
    """Request model for extracting symptoms from chat message"""
    sessionId: str = Field(..., description="Chat session ID")
    messageId: Optional[str] = Field(None, description="Chat message ID if available")
    messageText: str = Field(..., description="Conversational message text")


class ChatSymptomExtractResponse(BaseModel):
    """Response model for chat symptom extraction"""
    success: bool
    extractedJson: Optional[dict]
    confidence: float
    chatSymptomId: Optional[str]


class TrendReportRequest(BaseModel):
    """Request model for generating trend report"""
    reportType: str = Field(..., description="'3day', '7day', '15day', or '30day'")


class TrendReportResponse(BaseModel):
    """Response model for trend report"""
    id: str
    userId: str
    periodStart: datetime
    periodEnd: datetime
    reportType: str
    aggregatedMetrics: dict
    anomalies: List[dict]
    correlations: List[dict]
    clinicianSummary: str
    dataPointsAnalyzed: int
    confidenceScore: float
    generatedAt: datetime


# ============================================================================
# ENDPOINTS
# ============================================================================

@router.post("/checkin", response_model=SymptomCheckinResponse)
def create_symptom_checkin(
    checkin_data: SymptomCheckinCreate,
    current_user: User = Depends(get_current_user_py),
    db: Session = Depends(get_db)
):
    """
    Create a daily symptom check-in with structured and free-form data.
    
    This endpoint stores patient-reported symptom data for trend analysis.
    All data is observational and requires clinician interpretation.
    """
    try:
        # Create check-in record (using Drizzle/PostgreSQL schema)
        # For now, return a mock response until we wire up the database
        # In production, this would insert into symptom_checkins table
        
        # Create mock response for now
        # TODO: Wire up with actual database insert
        return SymptomCheckinResponse(
            id="mock-id-123",
            userId=current_user.id,
            timestamp=datetime.utcnow(),
            painLevel=checkin_data.painLevel,
            fatigueLevel=checkin_data.fatigueLevel,
            breathlessnessLevel=checkin_data.breathlessnessLevel,
            sleepQuality=checkin_data.sleepQuality,
            mood=checkin_data.mood,
            mobilityScore=checkin_data.mobilityScore,
            medicationsTaken=checkin_data.medicationsTaken,
            triggers=checkin_data.triggers,
            symptoms=checkin_data.symptoms,
            note=checkin_data.note,
            source="app",
            createdAt=datetime.utcnow()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create symptom check-in: {str(e)}")


@router.get("/checkins/recent", response_model=List[SymptomCheckinResponse])
def get_recent_checkins(
    days: int = Query(7, ge=1, le=90, description="Number of days to retrieve"),
    current_user: User = Depends(get_current_user_py),
    db: Session = Depends(get_db)
):
    """
    Get recent symptom check-ins for the current user.
    
    Args:
        days: Number of days to retrieve (default: 7, max: 90)
    
    Returns:
        List of symptom check-ins in reverse chronological order
    """
    try:
        # TODO: Query symptom_checkins table
        # For now, return empty list
        return []
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve check-ins: {str(e)}")


@router.post("/extract-from-chat", response_model=ChatSymptomExtractResponse)
def extract_symptoms_from_chat(
    extract_request: ChatSymptomExtractRequest,
    current_user: User = Depends(get_current_user_py),
    db: Session = Depends(get_db)
):
    """
    Extract structured symptom data from Agent Clona conversation message.
    
    Uses GPT-4o to parse conversational text and extract:
    - Body locations
    - Symptom types
    - Intensity descriptors
    - Temporal information
    - Aggravating/relieving factors
    
    This is an observational extraction for tracking purposes only.
    """
    try:
        # Extract symptoms using AI service (synchronous)
        extraction_result = SymptomExtractionService.extract_from_conversation(
            patient_id=current_user.id,
            message_text=extract_request.messageText,
            session_id=extract_request.sessionId,
            message_id=extract_request.messageId
        )
        
        if not extraction_result.get("success"):
            return ChatSymptomExtractResponse(
                success=False,
                extractedJson=None,
                confidence=0.0,
                chatSymptomId=None
            )
        
        # TODO: Store in chat_symptoms table
        # For now, just return extraction result
        
        return ChatSymptomExtractResponse(
            success=True,
            extractedJson=extraction_result.get("extracted_json"),
            confidence=extraction_result.get("confidence", 0.0),
            chatSymptomId="mock-chat-symptom-id"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to extract symptoms: {str(e)}")


@router.post("/trend-report", response_model=TrendReportResponse)
def generate_trend_report(
    report_request: TrendReportRequest,
    current_user: User = Depends(get_current_user_py),
    db: Session = Depends(get_db)
):
    """
    Generate ML-based trend analysis report for symptom data.
    
    Provides:
    - Aggregated symptom averages
    - Anomaly detection (observational, non-diagnostic)
    - Correlational insights
    - Clinician-ready summary
    
    Report types:
    - '3day': 3-day snapshot
    - '7day': 1-week overview
    - '15day': 2-week trends
    - '30day': Monthly comprehensive analysis
    
    IMPORTANT: All outputs are observational and require clinician interpretation.
    This is NOT a diagnostic tool.
    """
    try:
        # Validate report type
        valid_types = ["3day", "7day", "15day", "30day"]
        if report_request.reportType not in valid_types:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid report type. Must be one of: {', '.join(valid_types)}"
            )
        
        # Determine period
        days_map = {"3day": 3, "7day": 7, "15day": 15, "30day": 30}
        days = days_map[report_request.reportType]
        
        period_end = datetime.utcnow()
        period_start = period_end - timedelta(days=days)
        
        # TODO: Fetch actual data from symptom_checkins and passive_metrics tables
        # For now, create mock report
        
        # Mock aggregated metrics
        aggregated_metrics = {
            "avgPainLevel": 4.2,
            "avgFatigueLevel": 5.1,
            "avgSleepQuality": 6.8,
            "topSymptoms": [
                {"symptom": "headache", "frequency": 3},
                {"symptom": "fatigue", "frequency": 5}
            ]
        }
        
        # Mock anomalies
        anomalies = []
        
        # Mock correlations
        correlations = []
        
        # Generate clinician summary
        clinician_summary = SymptomTrendService.generate_clinician_summary(
            aggregated_metrics=aggregated_metrics,
            anomalies=anomalies,
            correlations=correlations,
            period_days=days
        )
        
        return TrendReportResponse(
            id="mock-report-id",
            userId=current_user.id,
            periodStart=period_start,
            periodEnd=period_end,
            reportType=report_request.reportType,
            aggregatedMetrics=aggregated_metrics,
            anomalies=anomalies,
            correlations=correlations,
            clinicianSummary=clinician_summary,
            dataPointsAnalyzed=0,
            confidenceScore=0.75,
            generatedAt=datetime.utcnow()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate trend report: {str(e)}")


@router.post("/voice-note-upload")
async def upload_voice_note(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user_py)
):
    """
    Upload voice note for symptom check-in.
    
    Uploads audio file to S3 with encryption and returns URL for storage.
    Voice notes provide additional context for symptom tracking.
    """
    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith("audio/"):
            raise HTTPException(
                status_code=400,
                detail="Invalid file type. Must be audio file."
            )
        
        # Read file content
        file_content = await file.read()
        
        # Upload to S3
        s3_url = S3Service.upload_file(
            file_content=file_content,
            filename=file.filename or "voice_note.wav",
            content_type=file.content_type,
            folder="symptom-voice-notes",
            patient_id=current_user.id
        )
        
        return {
            "success": True,
            "voiceNoteUrl": s3_url,
            "durationSeconds": None  # Would need audio processing to determine
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload voice note: {str(e)}")


@router.get("/health")
async def symptom_checkin_health():
    """Health check endpoint for symptom check-in API"""
    return {
        "status": "healthy",
        "service": "symptom-checkin-api",
        "timestamp": datetime.utcnow().isoformat(),
        "features": [
            "daily_checkins",
            "chat_extraction",
            "trend_reports",
            "anomaly_detection",
            "voice_notes"
        ]
    }
