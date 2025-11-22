"""
Symptom Check-in API - Daily Follow-up Symptom Tracking Module

Provides endpoints for:
- Daily symptom check-ins (structured + free-form)
- Conversational symptom extraction from Agent Clona
- ML-based trend reports (3/7/15/30 days)
- Anomaly detection and correlational insights

All outputs are labeled observational and non-diagnostic per HIPAA compliance.
"""

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Request
from sqlalchemy.orm import Session
from sqlalchemy import desc, and_
from typing import List, Optional
from datetime import datetime, timedelta
from pydantic import BaseModel, Field

from app.database import get_db
from app.dependencies import get_current_user, require_role
from app.models.user import User
from app.models.symptom_checkin_models import SymptomCheckin, ChatSymptom, PassiveMetric, TrendReport
from app.services.symptom_checkin_service import SymptomExtractionService, SymptomTrendService
from app.services.s3_service import S3Service
from app.services.audit_logger import AuditLogger

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
# HELPER FUNCTIONS
# ============================================================================

def checkin_to_response(checkin: SymptomCheckin) -> SymptomCheckinResponse:
    """Convert SQLAlchemy SymptomCheckin to Pydantic response model"""
    return SymptomCheckinResponse(
        id=str(checkin.id),
        userId=str(checkin.user_id),
        timestamp=checkin.timestamp,
        painLevel=checkin.pain_level,
        fatigueLevel=checkin.fatigue_level,
        breathlessnessLevel=checkin.breathlessness_level,
        sleepQuality=checkin.sleep_quality,
        mood=checkin.mood,
        mobilityScore=checkin.mobility_score,
        medicationsTaken=checkin.medications_taken,
        triggers=checkin.triggers or [],
        symptoms=checkin.symptoms or [],
        note=checkin.note,
        source=checkin.source,
        createdAt=checkin.created_at
    )


def report_to_response(report: TrendReport) -> TrendReportResponse:
    """Convert SQLAlchemy TrendReport to Pydantic response model"""
    return TrendReportResponse(
        id=str(report.id),
        userId=str(report.user_id),
        periodStart=report.period_start,
        periodEnd=report.period_end,
        reportType=report.report_type,
        aggregatedMetrics=report.aggregated_metrics,
        anomalies=report.anomalies or [],
        correlations=report.correlations or [],
        clinicianSummary=report.clinician_summary,
        dataPointsAnalyzed=report.data_points_analyzed,
        confidenceScore=report.confidence_score or 0.0,
        generatedAt=report.generated_at
    )


# ============================================================================
# ENDPOINTS
# ============================================================================

@router.post("/checkin", response_model=SymptomCheckinResponse)
def create_symptom_checkin(
    checkin_data: SymptomCheckinCreate,
    request: Request,
    current_user: User = Depends(require_role("patient")),
    db: Session = Depends(get_db)
):
    """
    Create a daily symptom check-in with structured and free-form data.
    
    This endpoint stores patient-reported symptom data for trend analysis.
    All data is observational and requires clinician interpretation.
    """
    try:
        # Create check-in record
        checkin = SymptomCheckin(
            user_id=current_user.id,
            pain_level=checkin_data.painLevel,
            fatigue_level=checkin_data.fatigueLevel,
            breathlessness_level=checkin_data.breathlessnessLevel,
            sleep_quality=checkin_data.sleepQuality,
            mood=checkin_data.mood,
            mobility_score=checkin_data.mobilityScore,
            medications_taken=checkin_data.medicationsTaken,
            triggers=checkin_data.triggers or [],
            symptoms=checkin_data.symptoms or [],
            note=checkin_data.note,
            source="app",
            device_type=checkin_data.deviceType
        )
        
        db.add(checkin)
        db.commit()
        db.refresh(checkin)
        
        # HIPAA Audit Log (AFTER successful operation)
        AuditLogger.log_event(
            event_type="symptom_checkin_created",
            user_id=current_user.id,
            resource_type="symptom_checkin",
            resource_id=checkin.id,
            action="create",
            status="success",
            ip_address=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent")
        )
        
        # Convert to response model
        return checkin_to_response(checkin)
        
    except Exception as e:
        db.rollback()
        # HIPAA Audit Log (failure)
        AuditLogger.log_event(
            event_type="symptom_checkin_created",
            user_id=current_user.id,
            resource_type="symptom_checkin",
            resource_id=None,
            action="create",
            status="failure",
            metadata={"error": str(e)},
            ip_address=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent")
        )
        raise HTTPException(status_code=500, detail=f"Failed to create check-in: {str(e)}")


@router.get("/history", response_model=List[SymptomCheckinResponse])
def get_symptom_history(
    request: Request,
    days: int = 30,
    current_user: User = Depends(require_role("patient")),
    db: Session = Depends(get_db)
):
    """
    Get symptom check-in history for the authenticated patient.
    Returns check-ins from the last N days.
    """
    try:
        # Get check-ins from the last N days
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        checkins = db.query(SymptomCheckin).filter(
            and_(
                SymptomCheckin.user_id == current_user.id,
                SymptomCheckin.timestamp >= cutoff_date
            )
        ).order_by(desc(SymptomCheckin.timestamp)).all()
        
        # HIPAA Audit Log (AFTER successful operation)
        AuditLogger.log_event(
            event_type="symptom_checkin_viewed",
            user_id=current_user.id,
            resource_type="symptom_checkin",
            resource_id=None,
            action="view",
            status="success",
            metadata={"count": len(checkins), "days": days},
            ip_address=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent")
        )
        
        # Convert to response models
        return [checkin_to_response(c) for c in checkins]
        
    except Exception as e:
        # HIPAA Audit Log (failure)
        AuditLogger.log_event(
            event_type="symptom_checkin_viewed",
            user_id=current_user.id,
            resource_type="symptom_checkin",
            resource_id=None,
            action="view",
            status="failure",
            metadata={"error": str(e)},
            ip_address=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent")
        )
        raise HTTPException(status_code=500, detail=f"Failed to fetch history: {str(e)}")


@router.post("/extract-from-chat", response_model=ChatSymptomExtractResponse)
def extract_symptoms_from_chat(
    extract_request: ChatSymptomExtractRequest,
    request: Request,
    current_user: User = Depends(require_role("patient")),
    db: Session = Depends(get_db)
):
    """
    Extract structured symptom data from Agent Clona conversation using GPT-4o.
    
    Returns extracted symptom information with confidence score.
    This is observational data requiring clinician interpretation.
    """
    try:
        # Initialize extraction service
        extraction_service = SymptomExtractionService()
        
        # Extract symptoms using GPT-4o
        extraction_result = extraction_service.extract_from_conversation(
            message_text=extract_request.messageText
        )
        
        # Store extraction in database
        chat_symptom = ChatSymptom(
            user_id=current_user.id,
            session_id=extract_request.sessionId,
            message_id=extract_request.messageId,
            extracted_json=extraction_result.get("extracted_data", {}),
            confidence=extraction_result.get("confidence", 0.0),
            locations=extraction_result.get("locations", []),
            symptom_types=extraction_result.get("symptom_types", []),
            intensity_mentions=extraction_result.get("intensity_mentions", []),
            temporal_info=extraction_result.get("temporal_info"),
            aggravating_factors=extraction_result.get("aggravating_factors", []),
            relieving_factors=extraction_result.get("relieving_factors", []),
            extraction_model="gpt-4o"
        )
        
        db.add(chat_symptom)
        db.commit()
        db.refresh(chat_symptom)
        
        # HIPAA Audit Log (AFTER successful operation)
        AuditLogger.log_event(
            event_type="chat_symptom_extracted",
            user_id=current_user.id,
            resource_type="chat_symptom",
            resource_id=chat_symptom.id,
            action="extract",
            status="success",
            metadata={"session_id": extract_request.sessionId, "confidence": chat_symptom.confidence},
            ip_address=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent")
        )
        
        return ChatSymptomExtractResponse(
            success=True,
            extractedJson=chat_symptom.extracted_json or {},
            confidence=float(chat_symptom.confidence) if chat_symptom.confidence else 0.0,
            chatSymptomId=str(chat_symptom.id)
        )
        
    except Exception as e:
        db.rollback()
        # HIPAA Audit Log (failure)
        AuditLogger.log_event(
            event_type="chat_symptom_extracted",
            user_id=current_user.id,
            resource_type="chat_symptom",
            resource_id=extract_request.sessionId,
            action="extract",
            status="failure",
            metadata={"error": str(e)},
            ip_address=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent")
        )
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")


@router.post("/trend-report", response_model=TrendReportResponse)
def generate_trend_report(
    report_request: TrendReportRequest,
    request: Request,
    current_user: User = Depends(require_role("patient")),
    db: Session = Depends(get_db)
):
    """
    Generate ML-based trend report for symptom data.
    
    Available report types: 3day, 7day, 15day, 30day
    Returns aggregated metrics, anomalies, correlations, and clinician summary.
    All outputs are observational and non-diagnostic.
    """
    try:
        # Map report type to days
        report_type_map = {
            "3day": 3,
            "7day": 7,
            "15day": 15,
            "30day": 30
        }
        
        days = report_type_map.get(report_request.reportType)
        if not days:
            raise HTTPException(status_code=400, detail="Invalid report type. Must be 3day, 7day, 15day, or 30day")
        
        # Calculate period
        period_end = datetime.utcnow()
        period_start = period_end - timedelta(days=days)
        
        # Get check-ins for the period
        checkins = db.query(SymptomCheckin).filter(
            and_(
                SymptomCheckin.user_id == current_user.id,
                SymptomCheckin.timestamp >= period_start,
                SymptomCheckin.timestamp <= period_end
            )
        ).order_by(SymptomCheckin.timestamp).all()
        
        if not checkins:
            raise HTTPException(status_code=404, detail="No symptom data found for this period")
        
        # Initialize trend service
        trend_service = SymptomTrendService()
        
        # Generate trend analysis
        trend_analysis = trend_service.generate_trend_report(
            user_id=current_user.id,
            checkins=checkins,
            period_start=period_start,
            period_end=period_end
        )
        
        # Store report in database
        report = TrendReport(
            user_id=current_user.id,
            report_type=report_request.reportType,
            period_start=period_start,
            period_end=period_end,
            aggregated_metrics=trend_analysis["aggregated_metrics"],
            anomalies=trend_analysis["anomalies"],
            correlations=trend_analysis["correlations"],
            clinician_summary=trend_analysis["clinician_summary"],
            data_points_analyzed=len(checkins),
            confidence_score=trend_analysis["confidence_score"]
        )
        
        db.add(report)
        db.commit()
        db.refresh(report)
        
        # HIPAA Audit Log (AFTER successful operation)
        AuditLogger.log_event(
            event_type="trend_report_generated",
            user_id=current_user.id,
            resource_type="trend_report",
            resource_id=report.id,
            action="generate",
            status="success",
            metadata={"report_type": report_request.reportType, "data_points": len(checkins)},
            ip_address=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent")
        )
        
        return report_to_response(report)
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        # HIPAA Audit Log (failure)
        AuditLogger.log_event(
            event_type="trend_report_generated",
            user_id=current_user.id,
            resource_type="trend_report",
            resource_id=report_request.reportType,
            action="generate",
            status="failure",
            metadata={"error": str(e)},
            ip_address=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent")
        )
        raise HTTPException(status_code=500, detail=f"Failed to generate report: {str(e)}")


@router.get("/trend-reports", response_model=List[TrendReportResponse])
def get_trend_reports(
    request: Request,
    limit: int = 10,
    current_user: User = Depends(require_role("patient")),
    db: Session = Depends(get_db)
):
    """
    Get previously generated trend reports for the authenticated patient.
    """
    try:
        reports = db.query(TrendReport).filter(
            TrendReport.user_id == current_user.id
        ).order_by(desc(TrendReport.generated_at)).limit(limit).all()
        
        # HIPAA Audit Log (AFTER successful operation)
        AuditLogger.log_event(
            event_type="trend_reports_viewed",
            user_id=current_user.id,
            resource_type="trend_report",
            resource_id=None,
            action="view",
            status="success",
            metadata={"count": len(reports), "limit": limit},
            ip_address=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent")
        )
        
        return [report_to_response(r) for r in reports]
        
    except Exception as e:
        # HIPAA Audit Log (failure)
        AuditLogger.log_event(
            event_type="trend_reports_viewed",
            user_id=current_user.id,
            resource_type="trend_report",
            resource_id=None,
            action="view",
            status="failure",
            metadata={"error": str(e)},
            ip_address=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent")
        )
        raise HTTPException(status_code=500, detail=f"Failed to fetch reports: {str(e)}")


@router.post("/upload-voice-note")
async def upload_voice_note(
    file: UploadFile = File(...),
    checkin_id: Optional[str] = None,
    request: Request = None,
    current_user: User = Depends(require_role("patient")),
    db: Session = Depends(get_db)
):
    """
    Upload a voice note attachment for a symptom check-in.
    Stores in S3 with SSE-KMS encryption and links to check-in record.
    """
    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith("audio/"):
            raise HTTPException(status_code=400, detail="File must be an audio file")
        
        # Upload to S3
        s3_service = S3Service()
        file_content = await file.read()  # Async file read
        
        s3_key = f"voice-notes/{current_user.id}/{datetime.utcnow().strftime('%Y%m%d')}/{file.filename}"
        s3_url = await s3_service.upload_file(
            file_data=file_content,
            s3_key=s3_key,
            content_type=file.content_type
        )
        
        # Update check-in if checkin_id provided
        if checkin_id:
            checkin = db.query(SymptomCheckin).filter(
                and_(
                    SymptomCheckin.id == checkin_id,
                    SymptomCheckin.user_id == current_user.id
                )
            ).first()
            
            if checkin:
                checkin.voice_note_url = s3_url
                # Note: source remains as originally set
                db.commit()
        
        # HIPAA Audit Log (AFTER successful operation)
        AuditLogger.log_event(
            event_type="voice_note_uploaded",
            user_id=current_user.id,
            resource_type="voice_note",
            resource_id=checkin_id or "new",
            action="upload",
            status="success",
            ip_address=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent")
        )
        
        return {
            "success": True,
            "voiceNoteUrl": s3_url,
            "message": "Voice note uploaded successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        # HIPAA Audit Log (failure)
        AuditLogger.log_event(
            event_type="voice_note_uploaded",
            user_id=current_user.id,
            resource_type="voice_note",
            resource_id=checkin_id or "new",
            action="upload",
            status="failure",
            metadata={"error": str(e)},
            ip_address=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent")
        )
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.delete("/checkin/{checkin_id}")
def delete_symptom_checkin(
    checkin_id: str,
    request: Request,
    current_user: User = Depends(require_role("patient")),
    db: Session = Depends(get_db)
):
    """
    Delete a symptom check-in (patient can only delete their own).
    """
    try:
        checkin = db.query(SymptomCheckin).filter(
            and_(
                SymptomCheckin.id == checkin_id,
                SymptomCheckin.user_id == current_user.id
            )
        ).first()
        
        if not checkin:
            raise HTTPException(status_code=404, detail="Check-in not found")
        
        db.delete(checkin)
        db.commit()
        
        # HIPAA Audit Log (AFTER successful operation)
        AuditLogger.log_event(
            event_type="symptom_checkin_deleted",
            user_id=current_user.id,
            resource_type="symptom_checkin",
            resource_id=checkin_id,
            action="delete",
            status="success",
            ip_address=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent")
        )
        
        return {"success": True, "message": "Check-in deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        # HIPAA Audit Log (failure)
        AuditLogger.log_event(
            event_type="symptom_checkin_deleted",
            user_id=current_user.id,
            resource_type="symptom_checkin",
            resource_id=checkin_id,
            action="delete",
            status="failure",
            metadata={"error": str(e)},
            ip_address=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent")
        )
        raise HTTPException(status_code=500, detail=f"Delete failed: {str(e)}")
