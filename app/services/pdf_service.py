"""
PDF Generation Service for Symptom Journal Reports
Generates structured weekly reports for doctor review
"""

from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import and_, desc
from typing import Optional
import io

from app.models.symptom_journal import SymptomMeasurement, SymptomImage, SymptomAlert, BodyArea
from app.services.s3_service import upload_file_to_s3


async def generate_symptom_journal_pdf(
    db: Session,
    patient_id: str,
    week_start: datetime,
    week_end: datetime
) -> str:
    """
    Generate a weekly PDF report for symptom journal
    
    Args:
        db: Database session
        patient_id: Patient ID
        week_start: Start of week
        week_end: End of week
        
    Returns:
        S3 URL of generated PDF
    """
    # Fetch measurements for the week
    measurements = db.query(SymptomMeasurement).filter(
        and_(
            SymptomMeasurement.patient_id == patient_id,
            SymptomMeasurement.created_at >= week_start,
            SymptomMeasurement.created_at <= week_end
        )
    ).order_by(SymptomMeasurement.created_at).all()
    
    # Fetch alerts for the week
    alerts = db.query(SymptomAlert).filter(
        and_(
            SymptomAlert.patient_id == patient_id,
            SymptomAlert.created_at >= week_start,
            SymptomAlert.created_at <= week_end
        )
    ).order_by(desc(SymptomAlert.severity)).all()
    
    # Generate PDF content (simplified HTML-to-PDF approach)
    pdf_content = generate_pdf_content(patient_id, week_start, week_end, measurements, alerts)
    
    # Convert to bytes (in production, use proper PDF library like pdfkit)
    pdf_bytes = pdf_content.encode('utf-8')
    
    # Upload to S3
    filename = f"symptom_journal_{patient_id}_{week_start.strftime('%Y%m%d')}.pdf"
    s3_key = await upload_file_to_s3(
        file_data=pdf_bytes,
        filename=filename,
        content_type="application/pdf"
    )
    
    return s3_key


def generate_pdf_content(
    patient_id: str,
    week_start: datetime,
    week_end: datetime,
    measurements: list,
    alerts: list
) -> str:
    """
    Generate PDF content as HTML (placeholder for actual PDF generation)
    In production, this would use a proper PDF library
    """
    # Group measurements by body area
    by_area = {}
    for m in measurements:
        area = m.body_area.value
        if area not in by_area:
            by_area[area] = []
        by_area[area].append(m)
    
    # Create summary report
    report = f"""
    SYMPTOM JOURNAL WEEKLY REPORT
    
    Patient ID: {patient_id}
    Period: {week_start.strftime('%Y-%m-%d')} to {week_end.strftime('%Y-%m-%d')}
    Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
    
    === SUMMARY ===
    Total Measurements: {len(measurements)}
    Alerts Generated: {len(alerts)}
    
    === BY BODY AREA ===
    """
    
    for area, area_measurements in by_area.items():
        report += f"\n{area.upper()}: {len(area_measurements)} measurements\n"
        
        # Add trend info
        if len(area_measurements) >= 2:
            first = area_measurements[0]
            last = area_measurements[-1]
            
            if first.brightness and last.brightness:
                change = ((last.brightness - first.brightness) / first.brightness * 100)
                report += f"  Brightness change: {change:.1f}%\n"
            
            if first.respiratory_rate_bpm and last.respiratory_rate_bpm:
                rr_change = last.respiratory_rate_bpm - first.respiratory_rate_bpm
                report += f"  Respiratory rate change: {rr_change:+.1f} BPM\n"
    
    # Add alerts
    if alerts:
        report += "\n=== ALERTS ===\n"
        for alert in alerts:
            report += f"\n{alert.severity.value.upper()}: {alert.title}\n"
            report += f"  {alert.message}\n"
            report += f"  Date: {alert.created_at.strftime('%Y-%m-%d %H:%M')}\n"
    
    report += "\n=== END OF REPORT ===\n"
    
    return report
