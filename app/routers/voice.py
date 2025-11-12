from fastapi import APIRouter, Depends, Response
from app.dependencies import get_current_user
from app.models.user import User
from app.services.voice_interface_service import VoiceInterfaceService
from pydantic import BaseModel

router = APIRouter(prefix="/api/v1/voice", tags=["voice"])


class TranscribeRequest(BaseModel):
    audio_file_path: str


class SpeechRequest(BaseModel):
    text: str
    voice: str = "nova"


class VoiceFollowupRequest(BaseModel):
    patient_id: str | None = None
    audio_file_path: str


@router.post("/transcribe")
async def transcribe_audio(
    request: TranscribeRequest,
    current_user: User = Depends(get_current_user)
):
    service = VoiceInterfaceService()
    result = await service.transcribe_audio(request.audio_file_path)
    return result


@router.post("/speech")
async def generate_speech(
    request: SpeechRequest,
    current_user: User = Depends(get_current_user)
):
    service = VoiceInterfaceService()
    audio_bytes = await service.generate_speech(request.text, request.voice)
    
    return Response(
        content=audio_bytes,
        media_type="audio/mpeg",
        headers={"Content-Disposition": "attachment; filename=speech.mp3"}
    )


@router.post("/followup")
async def voice_followup(
    request: VoiceFollowupRequest,
    current_user: User = Depends(get_current_user)
):
    service = VoiceInterfaceService()
    
    patient_id = request.patient_id if current_user.role == "doctor" else current_user.id
    
    result = await service.process_voice_followup(patient_id, request.audio_file_path)
    return result
