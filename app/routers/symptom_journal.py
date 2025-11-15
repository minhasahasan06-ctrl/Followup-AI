"""
Symptom Journal Router - HIPAA-compliant visual symptom tracking
Monitors changes in appearance and breathing patterns WITHOUT making diagnoses
"""

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session
from sqlalchemy import desc, and_, func
from typing import List, Optional
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
import base64
import io
from PIL import Image
import json

from app.database import get_db
from app.models.symptom_journal import (
    SymptomImage, SymptomMeasurement, SymptomAlert, WeeklySummary,
    BodyArea, ChangeType, AlertSeverity
)
from app.models.user import User
from app.models.patient_doctor_connection import PatientDoctorConnection
from app.dependencies import require_role
from app.services.openai_service import analyze_symptom_image, generate_weekly_summary
from app.services.s3_service import upload_symptom_image, generate_presigned_url


router = APIRouter(prefix="/api/v1/symptom-journal", tags=["Symptom Journal"])


# Pydantic models
class ImageUploadResponse(BaseModel):
    image_id: int
    measurement_id: int
    body_area: str
    ai_observations: Optional[str]
    detected_changes: List[dict]
    alerts: List[dict]


class MeasurementResponse(BaseModel):
    id: int
    body_area: str
    created_at: datetime
    color_change_percent: Optional[float]
    area_change_percent: Optional[float]
    respiratory_rate_bpm: Optional[float]
    ai_observations: Optional[str]
    image_url: Optional[str]
    alerts: List[dict]


class AlertResponse(BaseModel):
    id: int
    severity: str
    change_type: str
    body_area: str
    title: str
    message: str
    change_percent: Optional[float]
    created_at: datetime
    acknowledged: bool


def calculate_color_metrics(image_data: bytes) -> dict:
    """
    Calculate basic color metrics from image
    NOT for medical diagnosis - only for change detection
    
    OPTIMIZATION: Resizes large images to prevent memory exhaustion
    """
    try:
        img = Image.open(io.BytesIO(image_data))
        img = img.convert('RGB')
        
        # Resize large images to prevent memory/CPU exhaustion
        # Max dimension of 800px is sufficient for color analysis
        max_dimension = 800
        if img.width > max_dimension or img.height > max_dimension:
            # Calculate new dimensions while maintaining aspect ratio
            if img.width > img.height:
                new_width = max_dimension
                new_height = int((max_dimension / img.width) * img.height)
            else:
                new_height = max_dimension
                new_width = int((max_dimension / img.height) * img.width)
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Calculate average RGB values
        pixels = list(img.getdata())
        total_pixels = len(pixels)
        
        avg_r = sum(p[0] for p in pixels) / total_pixels
        avg_g = sum(p[1] for p in pixels) / total_pixels
        avg_b = sum(p[2] for p in pixels) / total_pixels
        
        # Calculate brightness (perceived luminance)
        brightness = (avg_r * 0.299 + avg_g * 0.587 + avg_b * 0.114)
        
        # Calculate contrast (standard deviation of brightness)
        brightness_values = [(p[0] * 0.299 + p[1] * 0.587 + p[2] * 0.114) for p in pixels]
        mean_brightness = sum(brightness_values) / len(brightness_values)
        variance = sum((b - mean_brightness) ** 2 for b in brightness_values) / len(brightness_values)
        contrast = variance ** 0.5
        
        return {
            "avg_red": round(avg_r, 2),
            "avg_green": round(avg_g, 2),
            "avg_blue": round(avg_b, 2),
            "brightness": round(brightness, 2),
            "contrast": round(contrast, 2)
        }
    except Exception as e:
        print(f"Error calculating color metrics: {e}")
        return {}


def calculate_area_proxy(image_data: bytes, body_area: BodyArea) -> Optional[int]:
    """
    Calculate area proxy for swelling detection
    Counts non-background pixels in ROI
    NOT for medical grading - only for trend detection
    
    OPTIMIZATION: Resizes large images to prevent memory exhaustion
    """
    try:
        img = Image.open(io.BytesIO(image_data))
        img = img.convert('RGB')
        
        # Resize large images to prevent memory/CPU exhaustion
        max_dimension = 800
        if img.width > max_dimension or img.height > max_dimension:
            if img.width > img.height:
                new_width = max_dimension
                new_height = int((max_dimension / img.width) * img.height)
            else:
                new_height = max_dimension
                new_width = int((max_dimension / img.height) * img.width)
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        pixels = list(img.getdata())
        
        # Simple background detection (light colored pixels)
        # In production, this would use more sophisticated ROI selection
        non_white_pixels = sum(1 for p in pixels if (p[0] + p[1] + p[2]) < 700)
        
        return non_white_pixels
    except Exception as e:
        print(f"Error calculating area proxy: {e}")
        return None


def detect_significant_changes(
    current: SymptomMeasurement,
    previous: Optional[SymptomMeasurement],
    db: Session
) -> List[dict]:
    """
    Detect significant changes and create alerts if needed
    Uses non-diagnostic language
    """
    changes = []
    
    if not previous:
        return changes
    
    # Color change detection
    if current.color_change_percent and abs(current.color_change_percent) > 15:
        severity = AlertSeverity.ATTENTION if abs(current.color_change_percent) > 25 else AlertSeverity.CAUTION
        
        alert = SymptomAlert(
            patient_id=current.patient_id,
            measurement_id=current.id,
            severity=severity,
            change_type=ChangeType.COLOR,
            body_area=current.body_area,
            title="Color Pattern Change Detected",
            message=f"Your {current.body_area.value} color pattern has changed by {abs(current.color_change_percent):.1f}% compared to your previous check-in. Please consider checking in with your clinician.",
            change_percent=current.color_change_percent
        )
        db.add(alert)
        db.commit()
        
        changes.append({
            "type": "color",
            "severity": severity.value,
            "message": alert.message
        })
    
    # Swelling change detection
    if current.area_change_percent and abs(current.area_change_percent) > 10:
        severity = AlertSeverity.ATTENTION if abs(current.area_change_percent) > 20 else AlertSeverity.CAUTION
        
        direction = "increased" if current.area_change_percent > 0 else "decreased"
        alert = SymptomAlert(
            patient_id=current.patient_id,
            measurement_id=current.id,
            severity=severity,
            change_type=ChangeType.SWELLING,
            body_area=current.body_area,
            title="Size Change Detected",
            message=f"The measured area of your {current.body_area.value} has {direction} by {abs(current.area_change_percent):.1f}% compared to last week. Please discuss this change with your healthcare provider.",
            change_percent=current.area_change_percent
        )
        db.add(alert)
        db.commit()
        
        changes.append({
            "type": "swelling",
            "severity": severity.value,
            "message": alert.message
        })
    
    # Respiratory rate change detection
    if current.respiratory_rate_bpm and previous.respiratory_rate_bpm:
        rr_change_percent = ((current.respiratory_rate_bpm - previous.respiratory_rate_bpm) / previous.respiratory_rate_bpm) * 100
        
        if abs(rr_change_percent) > 20:
            severity = AlertSeverity.ATTENTION if abs(rr_change_percent) > 30 else AlertSeverity.CAUTION
            
            alert = SymptomAlert(
                patient_id=current.patient_id,
                measurement_id=current.id,
                severity=severity,
                change_type=ChangeType.RESPIRATORY_RATE,
                body_area=current.body_area,
                title="Breathing Pattern Change",
                message=f"Your estimated breathing rate has changed by {abs(rr_change_percent):.1f}% from your recent baseline. This data may be useful for your upcoming telehealth visit.",
                change_percent=rr_change_percent
            )
            db.add(alert)
            db.commit()
            
            changes.append({
                "type": "respiratory_rate",
                "severity": severity.value,
                "message": alert.message
            })
    
    return changes


@router.post("/upload", response_model=ImageUploadResponse)
async def upload_symptom_image_endpoint(
    file: UploadFile = File(...),
    body_area: str = Form(...),
    patient_notes: Optional[str] = Form(None),
    symptoms_reported: Optional[str] = Form(None),  # JSON string
    current_user: User = Depends(require_role("patient")),
    db: Session = Depends(get_db)
):
    """
    Upload a symptom image/video and analyze for changes
    Supports: legs, face, eyes, chest (for breathing)
    
    Returns:
    - Non-diagnostic observations
    - Change percentages
    - Early warning alerts if patterns change
    """
    # Validate body area
    try:
        body_area_enum = BodyArea(body_area)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid body area. Must be one of: {', '.join([a.value for a in BodyArea])}")
    
    # Read file data
    file_data = await file.read()
    
    # Upload to S3
    s3_result = await upload_symptom_image(
        file_data=file_data,
        patient_id=current_user.id,
        body_area=body_area_enum,
        content_type=file.content_type
    )
    
    # Create image record
    symptom_image = SymptomImage(
        patient_id=current_user.id,
        body_area=body_area_enum,
        s3_bucket=s3_result["bucket"],
        s3_key=s3_result["key"],
        s3_url=s3_result.get("url"),
        file_size=len(file_data),
        mime_type=file.content_type,
        capture_type="video" if "video" in file.content_type else "photo"
    )
    db.add(symptom_image)
    db.commit()
    db.refresh(symptom_image)
    
    # Calculate color metrics
    color_metrics = calculate_color_metrics(file_data) if "image" in file.content_type else {}
    
    # Calculate area proxy for swelling detection (legs, face)
    area_proxy = None
    if body_area_enum in [BodyArea.LEGS, BodyArea.FACE]:
        area_proxy = calculate_area_proxy(file_data, body_area_enum)
    
    # Get previous measurement for comparison
    previous_measurement = db.query(SymptomMeasurement).filter(
        SymptomMeasurement.patient_id == current_user.id,
        SymptomMeasurement.body_area == body_area_enum
    ).order_by(desc(SymptomMeasurement.created_at)).first()
    
    # Calculate changes
    color_change_percent = None
    area_change_percent = None
    
    if previous_measurement:
        if color_metrics.get("brightness") and previous_measurement.brightness:
            color_change_percent = ((color_metrics["brightness"] - previous_measurement.brightness) / previous_measurement.brightness) * 100
        
        if area_proxy and previous_measurement.roi_area_pixels:
            area_change_percent = ((area_proxy - previous_measurement.roi_area_pixels) / previous_measurement.roi_area_pixels) * 100
    
    # Get AI observations using OpenAI Vision API
    ai_observations = ""
    detected_changes_list = []
    
    if "image" in file.content_type:
        try:
            ai_result = await analyze_symptom_image(
                image_data=file_data,
                body_area=body_area_enum,
                previous_observations=previous_measurement.ai_observations if previous_measurement else None
            )
            ai_observations = ai_result.get("observations", "")
            detected_changes_list = ai_result.get("detected_changes", [])
        except Exception as e:
            print(f"Error getting AI observations: {e}")
    
    # Parse symptoms reported
    symptoms_dict = None
    if symptoms_reported:
        try:
            symptoms_dict = json.loads(symptoms_reported)
        except:
            pass
    
    # Create measurement record
    measurement = SymptomMeasurement(
        patient_id=current_user.id,
        body_area=body_area_enum,
        image_id=symptom_image.id,
        avg_red=color_metrics.get("avg_red"),
        avg_green=color_metrics.get("avg_green"),
        avg_blue=color_metrics.get("avg_blue"),
        brightness=color_metrics.get("brightness"),
        contrast=color_metrics.get("contrast"),
        color_change_percent=color_change_percent,
        roi_area_pixels=area_proxy,
        area_change_percent=area_change_percent,
        ai_observations=ai_observations,
        detected_changes=detected_changes_list,
        compared_to_measurement_id=previous_measurement.id if previous_measurement else None,
        patient_notes=patient_notes,
        symptoms_reported=symptoms_dict
    )
    db.add(measurement)
    db.commit()
    db.refresh(measurement)
    
    # Detect significant changes and create alerts
    alerts = detect_significant_changes(measurement, previous_measurement, db)
    
    return {
        "image_id": symptom_image.id,
        "measurement_id": measurement.id,
        "body_area": body_area_enum.value,
        "ai_observations": ai_observations,
        "detected_changes": detected_changes_list,
        "alerts": alerts
    }


@router.get("/measurements/recent")
async def get_recent_measurements(
    body_area: Optional[str] = None,
    days: int = 30,
    current_user: User = Depends(require_role("patient")),
    db: Session = Depends(get_db)
):
    """
    Get recent symptom measurements for the authenticated patient
    SECURITY: Uses current_user.id from authentication, not client-provided ID
    Optionally filter by body area
    """
    # Verify patient role (already enforced by require_role, but being explicit)
    if current_user.role != "patient":
        raise HTTPException(status_code=403, detail="Only patients can access their own symptom data")
    
    cutoff_date = datetime.utcnow() - timedelta(days=days)
    
    # Query only this patient's data (uses authenticated user ID)
    query = db.query(SymptomMeasurement).filter(
        SymptomMeasurement.patient_id == current_user.id,  # Authenticated patient ID only
        SymptomMeasurement.created_at >= cutoff_date
    )
    
    if body_area:
        try:
            body_area_enum = BodyArea(body_area)
            query = query.filter(SymptomMeasurement.body_area == body_area_enum)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid body area")
    
    measurements = query.order_by(desc(SymptomMeasurement.created_at)).all()
    
    # Get associated images and alerts
    result = []
    for m in measurements:
        image = db.query(SymptomImage).filter(SymptomImage.id == m.image_id).first()
        image_url = None
        if image and image.s3_key:
            image_url = generate_presigned_url(image.s3_bucket, image.s3_key)
        
        alerts = db.query(SymptomAlert).filter(SymptomAlert.measurement_id == m.id).all()
        
        result.append({
            "id": m.id,
            "body_area": m.body_area.value,
            "created_at": m.created_at,
            "color_change_percent": m.color_change_percent,
            "area_change_percent": m.area_change_percent,
            "respiratory_rate_bpm": m.respiratory_rate_bpm,
            "ai_observations": m.ai_observations,
            "patient_notes": m.patient_notes,
            "image_url": image_url,
            "alerts": [{
                "severity": a.severity.value,
                "title": a.title,
                "message": a.message
            } for a in alerts]
        })
    
    return {"measurements": result}


@router.get("/alerts")
async def get_alerts(
    acknowledged: Optional[bool] = None,
    current_user: User = Depends(require_role("patient")),
    db: Session = Depends(get_db)
):
    """
    Get symptom alerts for the authenticated patient
    SECURITY: Uses current_user.id from authentication, not client-provided ID
    Filter by acknowledged status
    """
    # Verify patient role (already enforced by require_role, but being explicit)
    if current_user.role != "patient":
        raise HTTPException(status_code=403, detail="Only patients can access their own alerts")
    
    # Query only this patient's alerts (uses authenticated user ID)
    query = db.query(SymptomAlert).filter(
        SymptomAlert.patient_id == current_user.id  # Authenticated patient ID only
    )
    
    if acknowledged is not None:
        query = query.filter(SymptomAlert.acknowledged == acknowledged)
    
    alerts = query.order_by(desc(SymptomAlert.created_at)).limit(50).all()
    
    return {
        "alerts": [{
            "id": a.id,
            "severity": a.severity.value,
            "change_type": a.change_type.value,
            "body_area": a.body_area.value,
            "title": a.title,
            "message": a.message,
            "change_percent": a.change_percent,
            "created_at": a.created_at,
            "acknowledged": a.acknowledged
        } for a in alerts]
    }


@router.post("/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(
    alert_id: int,
    current_user: User = Depends(require_role("patient")),
    db: Session = Depends(get_db)
):
    """
    Mark an alert as acknowledged for the authenticated patient
    SECURITY: Verifies alert belongs to authenticated patient
    """
    # Query alert with patient ownership verification
    alert = db.query(SymptomAlert).filter(
        SymptomAlert.id == alert_id,
        SymptomAlert.patient_id == current_user.id  # Verify ownership
    ).first()
    
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    
    alert.acknowledged = True
    alert.acknowledged_at = datetime.utcnow()
    db.commit()
    
    return {"message": "Alert acknowledged"}


@router.get("/trends")
async def get_symptom_trends(
    body_area: str,
    days: int = 30,
    current_user: User = Depends(require_role("patient")),
    db: Session = Depends(get_db)
):
    """
    Get trend data for a specific body area for the authenticated patient
    SECURITY: Uses current_user.id from authentication, not client-provided ID
    Shows change over time (NOT medical diagnosis)
    """
    # Verify patient role (already enforced by require_role, but being explicit)
    if current_user.role != "patient":
        raise HTTPException(status_code=403, detail="Only patients can access their own trend data")
    
    try:
        body_area_enum = BodyArea(body_area)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid body area")
    
    cutoff_date = datetime.utcnow() - timedelta(days=days)
    
    # Query only this patient's measurements (uses authenticated user ID)
    measurements = db.query(SymptomMeasurement).filter(
        SymptomMeasurement.patient_id == current_user.id,  # Authenticated patient ID only
        SymptomMeasurement.body_area == body_area_enum,
        SymptomMeasurement.created_at >= cutoff_date
    ).order_by(SymptomMeasurement.created_at).all()
    
    trend_data = []
    for m in measurements:
        trend_data.append({
            "date": m.created_at.isoformat(),
            "color_change": m.color_change_percent,
            "area_change": m.area_change_percent,
            "respiratory_rate": m.respiratory_rate_bpm,
            "brightness": m.brightness
        })
    
    return {
        "body_area": body_area_enum.value,
        "trend_data": trend_data,
        "measurement_count": len(measurements)
    }


# Doctor-facing endpoints
@router.get("/doctor/patient/{patient_id}/measurements")
async def get_patient_measurements_for_doctor(
    patient_id: str,
    body_area: Optional[str] = None,
    days: int = 30,
    current_user: User = Depends(require_role("doctor")),
    db: Session = Depends(get_db)
):
    """
    Doctor view: Get symptom measurements for a connected patient
    SECURITY: Verifies active patient-doctor connection AND that measurements belong to that patient
    """
    # Verify doctor role explicitly
    if current_user.role != "doctor":
        raise HTTPException(status_code=403, detail="Only doctors can access patient symptom data")
    
    # Verify active doctor-patient connection exists
    connection = db.query(PatientDoctorConnection).filter(
        and_(
            PatientDoctorConnection.patient_id == patient_id,
            PatientDoctorConnection.doctor_id == current_user.id,
            PatientDoctorConnection.status == "connected"
        )
    ).first()
    
    if not connection:
        raise HTTPException(
            status_code=403,
            detail="You do not have permission to view this patient's symptom data. Patient must be connected to you."
        )
    
    cutoff_date = datetime.utcnow() - timedelta(days=days)
    
    # Query measurements with strict patient_id verification
    # Double-check that measurements actually belong to the connected patient
    query = db.query(SymptomMeasurement).filter(
        and_(
            SymptomMeasurement.patient_id == patient_id,  # Must match path parameter
            SymptomMeasurement.patient_id == connection.patient_id,  # Must match connection
            SymptomMeasurement.created_at >= cutoff_date
        )
    )
    
    if body_area:
        try:
            body_area_enum = BodyArea(body_area)
            query = query.filter(SymptomMeasurement.body_area == body_area_enum)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid body area")
    
    measurements = query.order_by(desc(SymptomMeasurement.created_at)).all()
    
    result = []
    for m in measurements:
        image = db.query(SymptomImage).filter(SymptomImage.id == m.image_id).first()
        image_url = None
        if image and image.s3_key:
            image_url = generate_presigned_url(image.s3_bucket, image.s3_key)
        
        alerts = db.query(SymptomAlert).filter(SymptomAlert.measurement_id == m.id).all()
        
        result.append({
            "id": m.id,
            "body_area": m.body_area.value,
            "created_at": m.created_at,
            "color_change_percent": m.color_change_percent,
            "area_change_percent": m.area_change_percent,
            "respiratory_rate_bpm": m.respiratory_rate_bpm,
            "ai_observations": m.ai_observations,
            "patient_notes": m.patient_notes,
            "symptoms_reported": m.symptoms_reported,
            "image_url": image_url,
            "alerts": [{
                "severity": a.severity.value,
                "title": a.title,
                "message": a.message,
                "change_percent": a.change_percent
            } for a in alerts]
        })
    
    return {
        "patient_id": patient_id,
        "measurements": result
    }


@router.get("/doctor/patient/{patient_id}/summary")
async def get_patient_weekly_summary(
    patient_id: str,
    weeks: int = 4,
    current_user: User = Depends(require_role("doctor")),
    db: Session = Depends(get_db)
):
    """
    Doctor view: Get weekly summaries for a connected patient
    SECURITY: Verifies active patient-doctor connection AND that summaries belong to that patient
    """
    # Verify doctor role explicitly
    if current_user.role != "doctor":
        raise HTTPException(status_code=403, detail="Only doctors can access patient symptom data")
    
    # Verify active doctor-patient connection exists
    connection = db.query(PatientDoctorConnection).filter(
        and_(
            PatientDoctorConnection.patient_id == patient_id,
            PatientDoctorConnection.doctor_id == current_user.id,
            PatientDoctorConnection.status == "connected"
        )
    ).first()
    
    if not connection:
        raise HTTPException(
            status_code=403,
            detail="You do not have permission to view this patient's symptom data."
        )
    
    # Query summaries with strict patient_id verification
    # Double-check that summaries actually belong to the connected patient
    summaries = db.query(WeeklySummary).filter(
        and_(
            WeeklySummary.patient_id == patient_id,  # Must match path parameter
            WeeklySummary.patient_id == connection.patient_id  # Must match connection
        )
    ).order_by(desc(WeeklySummary.week_start)).limit(weeks).all()
    
    return {
        "patient_id": patient_id,
        "summaries": [{
            "id": s.id,
            "week_start": s.week_start,
            "week_end": s.week_end,
            "measurements_count": s.measurements_count,
            "significant_changes": s.significant_changes,
            "alert_count": s.alert_count,
            "legs_trend": s.legs_trend,
            "face_trend": s.face_trend,
            "eyes_trend": s.eyes_trend,
            "respiratory_trend": s.respiratory_trend,
            "clinician_summary": s.clinician_summary,
            "pdf_s3_key": s.pdf_s3_key
        } for s in summaries]
    }


@router.get("/compare", dependencies=[Depends(require_role("patient"))])
async def compare_measurements(
    body_area: BodyArea,
    measurement_id_1: int,
    measurement_id_2: int,
    current_user: User = Depends(require_role("patient")),
    db: Session = Depends(get_db)
):
    """
    Compare two symptom measurements side-by-side
    Shows change percentages and visual differences
    Patient access only - own data
    SECURITY: All measurements and images must belong to authenticated user
    """
    # SECURITY: Fetch both measurements with patient_id verification
    m1 = db.query(SymptomMeasurement).filter(
        and_(
            SymptomMeasurement.id == measurement_id_1,
            SymptomMeasurement.patient_id == current_user.id,  # CRITICAL: Must match authenticated user
            SymptomMeasurement.body_area == body_area
        )
    ).first()
    
    m2 = db.query(SymptomMeasurement).filter(
        and_(
            SymptomMeasurement.id == measurement_id_2,
            SymptomMeasurement.patient_id == current_user.id,  # CRITICAL: Must match authenticated user
            SymptomMeasurement.body_area == body_area
        )
    ).first()
    
    if not m1 or not m2:
        raise HTTPException(status_code=404, detail="One or both measurements not found or access denied")
    
    # SECURITY: Verify both measurements belong to same patient (defense in depth)
    if m1.patient_id != m2.patient_id or m1.patient_id != current_user.id:
        raise HTTPException(status_code=403, detail="Cannot compare measurements from different patients")
    
    # SECURITY: Get images with patient_id verification (defense in depth)
    img1 = db.query(SymptomImage).filter(
        and_(
            SymptomImage.id == m1.image_id,
            SymptomImage.patient_id == current_user.id  # CRITICAL: Verify image ownership
        )
    ).first() if m1.image_id else None
    
    img2 = db.query(SymptomImage).filter(
        and_(
            SymptomImage.id == m2.image_id,
            SymptomImage.patient_id == current_user.id  # CRITICAL: Verify image ownership
        )
    ).first() if m2.image_id else None
    
    # Generate presigned URLs
    img1_url = generate_presigned_url(img1.s3_bucket, img1.s3_key) if img1 else None
    img2_url = generate_presigned_url(img2.s3_bucket, img2.s3_key) if img2 else None
    
    # Calculate changes
    color_change = None
    brightness_change = None
    area_change = None
    
    if m1.avg_red and m2.avg_red:
        color_change = {
            "red_change": round(m2.avg_red - m1.avg_red, 2),
            "green_change": round(m2.avg_green - m1.avg_green, 2),
            "blue_change": round(m2.avg_blue - m1.avg_blue, 2),
            "overall_change_percent": round(
                abs((m2.brightness - m1.brightness) / m1.brightness * 100) if m1.brightness > 0 else 0,
                1
            )
        }
    
    if m1.roi_area_pixels and m2.roi_area_pixels:
        area_change = {
            "area_change_pixels": m2.roi_area_pixels - m1.roi_area_pixels,
            "area_change_percent": round(
                ((m2.roi_area_pixels - m1.roi_area_pixels) / m1.roi_area_pixels * 100) if m1.roi_area_pixels > 0 else 0,
                1
            )
        }
    
    # Time difference
    time_diff = (m2.created_at - m1.created_at).total_seconds() / 86400  # Days
    
    return {
        "body_area": body_area.value,
        "comparison": {
            "measurement_1": {
                "id": m1.id,
                "date": m1.created_at,
                "image_url": img1_url,
                "color_metrics": {
                    "red": m1.avg_red,
                    "green": m1.avg_green,
                    "blue": m1.avg_blue,
                    "brightness": m1.brightness
                },
                "area_pixels": m1.roi_area_pixels,
                "respiratory_rate": m1.respiratory_rate_bpm,
                "ai_observations": m1.ai_observations
            },
            "measurement_2": {
                "id": m2.id,
                "date": m2.created_at,
                "image_url": img2_url,
                "color_metrics": {
                    "red": m2.avg_red,
                    "green": m2.avg_green,
                    "blue": m2.avg_blue,
                    "brightness": m2.brightness
                },
                "area_pixels": m2.roi_area_pixels,
                "respiratory_rate": m2.respiratory_rate_bpm,
                "ai_observations": m2.ai_observations
            },
            "changes": {
                "days_between": round(time_diff, 1),
                "color_change": color_change,
                "area_change": area_change,
                "respiratory_rate_change": round(m2.respiratory_rate_bpm - m1.respiratory_rate_bpm, 1) if m1.respiratory_rate_bpm and m2.respiratory_rate_bpm else None
            }
        }
    }


@router.post("/generate-weekly-pdf/{patient_id}", dependencies=[Depends(require_role("doctor"))])
async def generate_weekly_pdf_report(
    patient_id: str,
    week_start: datetime,
    week_end: datetime,
    current_user: User = Depends(require_role("doctor")),
    db: Session = Depends(get_db)
):
    """
    Generate a weekly PDF report for doctor review
    Doctor access only - requires active patient connection
    """
    # Verify active doctor-patient connection
    connection = db.query(PatientDoctorConnection).filter(
        and_(
            PatientDoctorConnection.patient_id == patient_id,
            PatientDoctorConnection.doctor_id == current_user.id,
            PatientDoctorConnection.status == "connected"
        )
    ).first()
    
    if not connection:
        raise HTTPException(
            status_code=403,
            detail="You do not have permission to generate reports for this patient"
        )
    
    # Get or create weekly summary
    summary = db.query(WeeklySummary).filter(
        and_(
            WeeklySummary.patient_id == patient_id,
            WeeklySummary.week_start == week_start,
            WeeklySummary.week_end == week_end
        )
    ).first()
    
    if not summary:
        # Create new summary
        summary = WeeklySummary(
            patient_id=patient_id,
            week_start=week_start,
            week_end=week_end,
            measurements_count={},
            significant_changes=[],
            alert_count=0,
            pdf_generated=False
        )
        db.add(summary)
        db.commit()
        db.refresh(summary)
    
    # Generate PDF using service (implementation in next update)
    from app.services.pdf_service import generate_symptom_journal_pdf
    
    try:
        pdf_url = await generate_symptom_journal_pdf(db, patient_id, week_start, week_end)
        
        # Update summary with PDF info
        summary.pdf_generated = True
        summary.pdf_s3_key = pdf_url
        db.commit()
        
        return {
            "summary_id": summary.id,
            "pdf_url": pdf_url,
            "week_start": week_start,
            "week_end": week_end
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate PDF: {str(e)}")
