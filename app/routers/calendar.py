from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.database import get_db
from app.dependencies import get_current_doctor
from app.models.user import User
from app.services.google_calendar_service import GoogleCalendarService

router = APIRouter(prefix="/api/v1/calendar", tags=["calendar"])


@router.post("/sync")
async def sync_calendar(
    current_user: User = Depends(get_current_doctor),
    db: Session = Depends(get_db)
):
    service = GoogleCalendarService(db)
    result = service.sync_appointments_to_google(current_user.id)
    return result


@router.post("/sync-from-google")
async def sync_from_google(
    current_user: User = Depends(get_current_doctor),
    db: Session = Depends(get_db)
):
    service = GoogleCalendarService(db)
    result = service.sync_from_google_calendar(current_user.id)
    return result
