"""
Medication Adherence API - Patient-facing adherence tracking
HIPAA-compliant with Autopilot signal integration
"""

import logging
import uuid
import json
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import text
from pydantic import BaseModel, Field

from app.database import get_db
from app.auth import get_current_user
from app.models.user import User

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/medication-adherence", tags=["medication-adherence"])


class AdherenceLogCreate(BaseModel):
    medication_id: str
    scheduled_time: datetime
    taken_at: Optional[datetime] = None
    status: str = Field(..., pattern="^(taken|missed|skipped|late)$")
    notes: Optional[str] = Field(None, max_length=500)
    side_effects: Optional[List[str]] = None
    effectiveness_rating: Optional[int] = Field(None, ge=1, le=5)


class AdherenceLogBatch(BaseModel):
    logs: List[AdherenceLogCreate]


class AdherenceStats(BaseModel):
    total_scheduled: int
    taken_on_time: int
    taken_late: int
    missed: int
    skipped: int
    adherence_rate: float
    streak_current: int
    streak_best: int


def audit_log(db: Session, user_id: str, action: str, resource_type: str, 
              resource_id: str, details: Optional[Dict[str, Any]] = None):
    """HIPAA-compliant audit logging"""
    try:
        db.execute(
            text("""
                INSERT INTO hipaa_audit_logs (id, user_id, action, resource_type, resource_id, details, created_at)
                VALUES (:id, :user_id, :action, :resource_type, :resource_id, :details, NOW())
            """),
            {
                "id": str(uuid.uuid4()),
                "user_id": user_id,
                "action": action,
                "resource_type": resource_type,
                "resource_id": resource_id,
                "details": json.dumps(details if details else {})
            }
        )
        db.commit()
    except Exception as e:
        logger.warning(f"Audit log failed: {e}")


@router.post("/log")
def log_adherence(
    log: AdherenceLogCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Log medication adherence event"""
    if current_user.role != "patient":
        raise HTTPException(status_code=403, detail="Only patients can log adherence")
    
    result = db.execute(
        text("SELECT patient_id, name FROM medications WHERE id = :id"),
        {"id": log.medication_id}
    )
    row = result.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Medication not found")
    
    if row[0] != current_user.id:
        raise HTTPException(status_code=403, detail="This medication does not belong to you")
    
    medication_name = row[1]
    log_id = str(uuid.uuid4())
    
    try:
        db.execute(
            text("""
                INSERT INTO medication_adherence (
                    id, medication_id, patient_id, scheduled_time, taken_at,
                    status, notes, side_effects, effectiveness_rating,
                    logged_by, created_at
                ) VALUES (
                    :id, :medication_id, :patient_id, :scheduled_time, :taken_at,
                    :status, :notes, :side_effects, :effectiveness_rating,
                    'patient', NOW()
                )
            """),
            {
                "id": log_id,
                "medication_id": log.medication_id,
                "patient_id": current_user.id,
                "scheduled_time": log.scheduled_time,
                "taken_at": log.taken_at,
                "status": log.status,
                "notes": log.notes,
                "side_effects": json.dumps(log.side_effects) if log.side_effects else None,
                "effectiveness_rating": log.effectiveness_rating,
            }
        )
        db.commit()
        
        try:
            db.execute(
                text("""
                    INSERT INTO autopilot_signals (
                        id, patient_id, category, source, raw_payload, ml_score, created_at
                    ) VALUES (
                        :id, :patient_id, 'meds', 'medication_adherence', :payload, :score, NOW()
                    )
                """),
                {
                    "id": str(uuid.uuid4()),
                    "patient_id": current_user.id,
                    "payload": json.dumps({
                        "medication_id": log.medication_id,
                        "medication_name": medication_name,
                        "action": log.status,
                        "scheduled_time": log.scheduled_time.isoformat(),
                        "taken_at": log.taken_at.isoformat() if log.taken_at else None,
                    }),
                    "score": 1.0 if log.status == "taken" else (0.7 if log.status == "late" else 0.0)
                }
            )
            db.commit()
        except Exception as e:
            logger.warning(f"Failed to send Autopilot signal: {e}")
        
        audit_log(db, current_user.id, "log_adherence", "medication_adherence", log_id,
                  {"medication_id": log.medication_id, "status": log.status})
        
        return {
            "id": log_id,
            "status": log.status,
            "message": "Adherence logged successfully"
        }
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to log adherence: {e}")
        raise HTTPException(status_code=500, detail="Failed to log adherence")


@router.post("/log/batch")
def log_adherence_batch(
    batch: AdherenceLogBatch,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Log multiple adherence events at once"""
    if current_user.role != "patient":
        raise HTTPException(status_code=403, detail="Only patients can log adherence")
    
    results = []
    for log in batch.logs:
        try:
            result = db.execute(
                text("SELECT patient_id FROM medications WHERE id = :id"),
                {"id": log.medication_id}
            )
            row = result.fetchone()
            if row and row[0] == current_user.id:
                log_id = str(uuid.uuid4())
                db.execute(
                    text("""
                        INSERT INTO medication_adherence (
                            id, medication_id, patient_id, scheduled_time, taken_at,
                            status, notes, logged_by, created_at
                        ) VALUES (
                            :id, :medication_id, :patient_id, :scheduled_time, :taken_at,
                            :status, :notes, 'patient', NOW()
                        )
                    """),
                    {
                        "id": log_id,
                        "medication_id": log.medication_id,
                        "patient_id": current_user.id,
                        "scheduled_time": log.scheduled_time,
                        "taken_at": log.taken_at,
                        "status": log.status,
                        "notes": log.notes,
                    }
                )
                results.append({"id": log_id, "medication_id": log.medication_id, "success": True})
        except Exception as e:
            results.append({"medication_id": log.medication_id, "success": False, "error": str(e)})
    
    db.commit()
    
    success_count = sum(1 for r in results if r.get("success"))
    audit_log(db, current_user.id, "log_adherence_batch", "medication_adherence", "batch",
              {"total": len(batch.logs), "success": success_count})
    
    return {"results": results, "success_count": success_count, "total": len(batch.logs)}


@router.get("/stats")
def get_adherence_stats(
    days: int = Query(30, ge=1, le=365),
    medication_id: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get adherence statistics for the patient"""
    if current_user.role != "patient":
        raise HTTPException(status_code=403, detail="Only patients can view their adherence stats")
    
    start_date = datetime.utcnow() - timedelta(days=days)
    
    params = {"patient_id": current_user.id, "start_date": start_date}
    medication_filter = ""
    if medication_id:
        medication_filter = "AND medication_id = :medication_id"
        params["medication_id"] = medication_id
    
    result = db.execute(
        text(f"""
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN status = 'taken' THEN 1 ELSE 0 END) as taken_on_time,
                SUM(CASE WHEN status = 'late' THEN 1 ELSE 0 END) as taken_late,
                SUM(CASE WHEN status = 'missed' THEN 1 ELSE 0 END) as missed,
                SUM(CASE WHEN status = 'skipped' THEN 1 ELSE 0 END) as skipped
            FROM medication_adherence
            WHERE patient_id = :patient_id
            AND scheduled_time >= :start_date
            {medication_filter}
        """),
        params
    )
    
    row = result.fetchone()
    total = row[0] or 0
    taken_on_time = row[1] or 0
    taken_late = row[2] or 0
    missed = row[3] or 0
    skipped = row[4] or 0
    
    adherence_rate = ((taken_on_time + taken_late) / total * 100) if total > 0 else 0
    
    streak_result = db.execute(
        text(f"""
            SELECT status, DATE(scheduled_time) as log_date
            FROM medication_adherence
            WHERE patient_id = :patient_id
            {medication_filter}
            ORDER BY scheduled_time DESC
        """),
        params
    )
    
    current_streak = 0
    best_streak = 0
    temp_streak = 0
    
    for row in streak_result.fetchall():
        if row[0] in ('taken', 'late'):
            temp_streak += 1
            best_streak = max(best_streak, temp_streak)
            if current_streak == temp_streak - 1:
                current_streak = temp_streak
        else:
            temp_streak = 0
    
    return {
        "period_days": days,
        "total_scheduled": total,
        "taken_on_time": taken_on_time,
        "taken_late": taken_late,
        "missed": missed,
        "skipped": skipped,
        "adherence_rate": round(adherence_rate, 1),
        "streak_current": current_streak,
        "streak_best": best_streak,
        "wellness_note": "This is wellness monitoring data - not medical advice."
    }


@router.get("/history")
def get_adherence_history(
    days: int = Query(7, ge=1, le=90),
    medication_id: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get adherence history for the patient"""
    if current_user.role != "patient":
        raise HTTPException(status_code=403, detail="Only patients can view their adherence history")
    
    start_date = datetime.utcnow() - timedelta(days=days)
    
    params = {"patient_id": current_user.id, "start_date": start_date}
    medication_filter = ""
    if medication_id:
        medication_filter = "AND ma.medication_id = :medication_id"
        params["medication_id"] = medication_id
    
    result = db.execute(
        text(f"""
            SELECT 
                ma.id, ma.medication_id, m.name as medication_name,
                ma.scheduled_time, ma.taken_at, ma.status, ma.notes,
                ma.side_effects, ma.effectiveness_rating, ma.created_at
            FROM medication_adherence ma
            JOIN medications m ON ma.medication_id = m.id
            WHERE ma.patient_id = :patient_id
            AND ma.scheduled_time >= :start_date
            {medication_filter}
            ORDER BY ma.scheduled_time DESC
        """),
        params
    )
    
    history = []
    for row in result.fetchall():
        history.append({
            "id": row[0],
            "medication_id": row[1],
            "medication_name": row[2],
            "scheduled_time": row[3].isoformat() if row[3] else None,
            "taken_at": row[4].isoformat() if row[4] else None,
            "status": row[5],
            "notes": row[6],
            "side_effects": json.loads(row[7]) if row[7] else None,
            "effectiveness_rating": row[8],
            "created_at": row[9].isoformat() if row[9] else None,
        })
    
    return {"history": history, "count": len(history), "period_days": days}


@router.get("/today")
def get_today_schedule(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get today's medication schedule with adherence status"""
    if current_user.role != "patient":
        raise HTTPException(status_code=403, detail="Only patients can view their schedule")
    
    today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    today_end = today_start + timedelta(days=1)
    
    result = db.execute(
        text("""
            SELECT 
                m.id, m.name, m.dosage, m.frequency,
                ms.time_of_day, ms.with_food, ms.special_instructions,
                ma.id as log_id, ma.status, ma.taken_at
            FROM medications m
            LEFT JOIN medication_schedules ms ON m.id = ms.medication_id AND ms.active = true
            LEFT JOIN medication_adherence ma ON m.id = ma.medication_id 
                AND ma.scheduled_time >= :today_start AND ma.scheduled_time < :today_end
            WHERE m.patient_id = :patient_id AND m.active = true
            ORDER BY ms.time_of_day, m.name
        """),
        {"patient_id": current_user.id, "today_start": today_start, "today_end": today_end}
    )
    
    schedule = []
    for row in result.fetchall():
        schedule.append({
            "medication_id": row[0],
            "name": row[1],
            "dosage": row[2],
            "frequency": row[3],
            "scheduled_time": row[4],
            "with_food": row[5],
            "instructions": row[6],
            "log_id": row[7],
            "status": row[8] or "pending",
            "taken_at": row[9].isoformat() if row[9] else None,
        })
    
    taken_count = sum(1 for s in schedule if s["status"] in ("taken", "late"))
    total_count = len(schedule)
    
    return {
        "schedule": schedule,
        "summary": {
            "total": total_count,
            "completed": taken_count,
            "remaining": total_count - taken_count,
            "completion_rate": round(taken_count / total_count * 100, 1) if total_count > 0 else 0
        }
    }


@router.get("/reminders")
def get_medication_reminders(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get pending medication reminders"""
    if current_user.role != "patient":
        raise HTTPException(status_code=403, detail="Only patients can view their reminders")
    
    now = datetime.utcnow()
    window_start = now - timedelta(hours=1)
    window_end = now + timedelta(hours=2)
    
    result = db.execute(
        text("""
            SELECT 
                m.id, m.name, m.dosage, m.frequency,
                ms.time_of_day, ms.with_food, ms.special_instructions,
                ms.reminder_enabled
            FROM medications m
            JOIN medication_schedules ms ON m.id = ms.medication_id
            WHERE m.patient_id = :patient_id 
            AND m.active = true
            AND ms.active = true
            AND ms.reminder_enabled = true
            ORDER BY ms.time_of_day
        """),
        {"patient_id": current_user.id}
    )
    
    reminders = []
    for row in result.fetchall():
        scheduled_time = row[4]
        if scheduled_time:
            reminders.append({
                "medication_id": row[0],
                "name": row[1],
                "dosage": row[2],
                "frequency": row[3],
                "scheduled_time": scheduled_time,
                "with_food": row[5],
                "instructions": row[6],
            })
    
    return {"reminders": reminders, "count": len(reminders)}
