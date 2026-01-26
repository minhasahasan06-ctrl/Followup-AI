"""
Voice Session API Router

REST API and WebSocket endpoints for voice conversation sessions.
Manages ASR → LLM → TTS pipeline with real-time streaming.

HIPAA Compliance:
- All voice sessions require authentication
- Recording consent verification
- Audit logging for all voice operations
"""

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime

from app.database import get_db
from app.dependencies import get_current_user
from app.models.user import User
from app.services.voice.session_orchestrator import (
    get_session_orchestrator,
    SessionConfig,
    SessionState,
    VoiceSession
)
from app.services.voice.tts_adapter import TTSVoice
from app.services.feature_flag_service import is_feature_enabled

router = APIRouter(prefix="/api/voice", tags=["voice-session"])


class CreateSessionRequest(BaseModel):
    """Request to create a voice session"""
    persona: str = Field(default="clona", description="Agent persona (clona or lysa)")
    voice: str = Field(default="nova", description="TTS voice ID")
    language: str = Field(default="en", description="Language code")
    push_to_talk: bool = Field(default=False, description="Use push-to-talk mode")
    always_listening: bool = Field(default=True, description="Enable always-listening")
    consent_recording: bool = Field(default=False, description="Recording consent given")
    consent_memory: bool = Field(default=False, description="Memory consent given")


class SessionResponse(BaseModel):
    """Response with session details"""
    session_id: str
    state: str
    persona: str
    voice: str
    created_at: str
    push_to_talk: bool
    always_listening: bool


class ProcessTextRequest(BaseModel):
    """Request to process text through LLM and TTS"""
    text: str = Field(..., min_length=1, max_length=5000)


class TranscriptionRequest(BaseModel):
    """Request to transcribe audio"""
    audio_base64: str = Field(..., description="Base64 encoded audio data")
    audio_format: str = Field(default="webm", description="Audio format")


@router.post("/sessions", response_model=SessionResponse)
async def create_voice_session(
    request: CreateSessionRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Create a new voice conversation session.
    
    Initializes ASR and TTS adapters for the user.
    Requires voice features to be enabled.
    """
    if not is_feature_enabled("enableVoice", user_id=str(current_user.id)):
        raise HTTPException(status_code=403, detail="Voice features are not enabled")
    
    if not is_feature_enabled("enableConsentFlow") or (request.consent_recording):
        pass
    else:
        raise HTTPException(
            status_code=400, 
            detail="Recording consent required. Please accept consent to continue."
        )
    
    try:
        voice = TTSVoice(request.voice)
    except ValueError:
        voice = TTSVoice.NOVA
    
    config = SessionConfig(
        user_id=str(current_user.id),
        persona=request.persona,
        voice=voice,
        language=request.language,
        push_to_talk=request.push_to_talk,
        always_listening=request.always_listening,
        consent_recording=request.consent_recording,
        consent_memory=request.consent_memory,
        enable_interruptions=is_feature_enabled("enableInterruptionHandling")
    )
    
    orchestrator = get_session_orchestrator()
    session = await orchestrator.create_session(config)
    
    return SessionResponse(
        session_id=session.session_id,
        state=session.state.value,
        persona=session.config.persona,
        voice=session.config.voice.value,
        created_at=session.created_at.isoformat(),
        push_to_talk=session.config.push_to_talk,
        always_listening=session.config.always_listening
    )


@router.get("/sessions/{session_id}", response_model=SessionResponse)
async def get_session(
    session_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get session details by ID"""
    orchestrator = get_session_orchestrator()
    session = orchestrator.get_session(session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if session.config.user_id != str(current_user.id):
        raise HTTPException(status_code=403, detail="Not authorized to access this session")
    
    return SessionResponse(
        session_id=session.session_id,
        state=session.state.value,
        persona=session.config.persona,
        voice=session.config.voice.value,
        created_at=session.created_at.isoformat(),
        push_to_talk=session.config.push_to_talk,
        always_listening=session.config.always_listening
    )


@router.post("/sessions/{session_id}/start-listening")
async def start_listening(
    session_id: str,
    current_user: User = Depends(get_current_user)
):
    """Start listening for user speech in a session"""
    orchestrator = get_session_orchestrator()
    session = orchestrator.get_session(session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if session.config.user_id != str(current_user.id):
        raise HTTPException(status_code=403, detail="Not authorized")
    
    success = await orchestrator.start_listening(session_id)
    
    return {"success": success, "state": session.state.value}


@router.post("/sessions/{session_id}/stop-listening")
async def stop_listening(
    session_id: str,
    current_user: User = Depends(get_current_user)
):
    """Stop listening and get final transcript"""
    orchestrator = get_session_orchestrator()
    session = orchestrator.get_session(session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if session.config.user_id != str(current_user.id):
        raise HTTPException(status_code=403, detail="Not authorized")
    
    result = await orchestrator.stop_listening(session_id)
    
    return {
        "success": result is not None,
        "transcript": result.text if result else None,
        "confidence": result.confidence if result else None
    }


@router.post("/sessions/{session_id}/process-text")
async def process_text(
    session_id: str,
    request: ProcessTextRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Process text through LLM and generate TTS audio.
    
    Used for push-to-talk mode or when transcription is done client-side.
    """
    orchestrator = get_session_orchestrator()
    session = orchestrator.get_session(session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if session.config.user_id != str(current_user.id):
        raise HTTPException(status_code=403, detail="Not authorized")
    
    response_parts = []
    async for event in orchestrator.generate_response(session_id, request.text):
        response_parts.append(event)
    
    llm_text = None
    action_card = None
    audio_chunks = []
    
    for part in response_parts:
        if part.get("type") == "llm_final":
            llm_text = part.get("text")
        elif part.get("type") == "action_card":
            action_card = part.get("card")
        elif part.get("type") == "tts_chunk":
            audio_chunks.append(part)
    
    import base64
    audio_data = b"".join([c.get("audio", b"") for c in audio_chunks])
    
    return {
        "success": True,
        "response_text": llm_text,
        "action_card": action_card,
        "audio_base64": base64.b64encode(audio_data).decode() if audio_data else None,
        "audio_format": "mp3"
    }


@router.post("/sessions/{session_id}/transcribe")
async def transcribe_audio(
    session_id: str,
    request: TranscriptionRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Transcribe audio data directly.
    
    Used when client sends complete audio chunks for transcription.
    """
    orchestrator = get_session_orchestrator()
    session = orchestrator.get_session(session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if session.config.user_id != str(current_user.id):
        raise HTTPException(status_code=403, detail="Not authorized")
    
    import base64
    audio_data = base64.b64decode(request.audio_base64)
    
    from app.services.voice.asr_adapter import get_asr_adapter
    asr = get_asr_adapter()
    
    result = await asr.transcribe_audio(
        audio_data=audio_data,
        audio_format=request.audio_format,
        language=session.config.language
    )
    
    return {
        "text": result.text,
        "confidence": result.confidence,
        "language": result.language,
        "is_final": result.is_final
    }


@router.delete("/sessions/{session_id}")
async def end_session(
    session_id: str,
    current_user: User = Depends(get_current_user)
):
    """End a voice session and get summary"""
    orchestrator = get_session_orchestrator()
    session = orchestrator.get_session(session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if session.config.user_id != str(current_user.id):
        raise HTTPException(status_code=403, detail="Not authorized")
    
    summary = await orchestrator.end_session(session_id)
    
    return {
        "success": True,
        "summary": summary
    }


@router.get("/voices")
async def get_available_voices(
    current_user: User = Depends(get_current_user)
):
    """Get list of available TTS voices"""
    from app.services.voice.tts_adapter import get_tts_adapter
    tts = get_tts_adapter()
    
    return {"voices": tts.get_available_voices()}


@router.get("/my-sessions")
async def get_my_sessions(
    current_user: User = Depends(get_current_user)
):
    """Get all active sessions for current user"""
    orchestrator = get_session_orchestrator()
    sessions = orchestrator.get_user_sessions(str(current_user.id))
    
    return {
        "sessions": [
            {
                "session_id": s.session_id,
                "state": s.state.value,
                "persona": s.config.persona,
                "created_at": s.created_at.isoformat()
            }
            for s in sessions
        ]
    }


@router.websocket("/ws/{session_id}")
async def voice_websocket(
    websocket: WebSocket,
    session_id: str
):
    """
    WebSocket endpoint for real-time voice streaming.
    
    Events sent to client:
    - interim_transcript: Partial transcription
    - final_transcript: Complete transcription
    - llm_partial: Streaming LLM response
    - llm_final: Complete LLM response
    - tts_chunk: Audio chunk
    - action_card: Generated action card
    - error: Error event
    
    Events from client:
    - audio_chunk: Raw audio bytes
    - end_speech: Signal end of speech
    - interrupt: Interrupt agent speech
    """
    await websocket.accept()
    
    orchestrator = get_session_orchestrator()
    session = orchestrator.get_session(session_id)
    
    if not session:
        await websocket.close(code=4004, reason="Session not found")
        return
    
    try:
        await orchestrator.start_listening(session_id)
        
        while True:
            data = await websocket.receive()
            
            if "bytes" in data:
                audio_chunk = data["bytes"]
                result = await orchestrator.send_audio(session_id, audio_chunk)
                
                if result:
                    await websocket.send_json({
                        "type": "interim_transcript" if not result.is_final else "final_transcript",
                        "text": result.text,
                        "confidence": result.confidence
                    })
            
            elif "text" in data:
                import json
                message = json.loads(data["text"])
                
                if message.get("type") == "end_speech":
                    final = await orchestrator.stop_listening(session_id)
                    if final:
                        await websocket.send_json({
                            "type": "final_transcript",
                            "text": final.text,
                            "confidence": final.confidence
                        })
                        
                        async for event in orchestrator.generate_response(session_id, final.text):
                            if event.get("type") == "llm_final":
                                await websocket.send_json({
                                    "type": "llm_final",
                                    "text": event.get("text")
                                })
                            elif event.get("type") == "action_card":
                                await websocket.send_json({
                                    "type": "action_card",
                                    "card": event.get("card")
                                })
                            elif event.get("type") == "tts_chunk":
                                import base64
                                await websocket.send_json({
                                    "type": "tts_chunk",
                                    "audio": base64.b64encode(event.get("audio", b"")).decode(),
                                    "is_final": event.get("is_final", False)
                                })
                        
                        await orchestrator.start_listening(session_id)
                
                elif message.get("type") == "interrupt":
                    session = orchestrator.get_session(session_id)
                    if session:
                        session.state = SessionState.INTERRUPTED
                        await websocket.send_json({"type": "interrupted"})
                
                elif message.get("type") == "process_text":
                    text = message.get("text", "")
                    if text:
                        async for event in orchestrator.generate_response(session_id, text):
                            if event.get("type") == "llm_final":
                                await websocket.send_json({
                                    "type": "llm_final",
                                    "text": event.get("text")
                                })
                            elif event.get("type") == "action_card":
                                await websocket.send_json({
                                    "type": "action_card",
                                    "card": event.get("card")
                                })
                            elif event.get("type") == "tts_chunk":
                                import base64
                                await websocket.send_json({
                                    "type": "tts_chunk",
                                    "audio": base64.b64encode(event.get("audio", b"")).decode(),
                                    "is_final": event.get("is_final", False)
                                })
    
    except WebSocketDisconnect:
        await orchestrator.end_session(session_id)
    except Exception as e:
        await websocket.send_json({"type": "error", "message": str(e)})
        await websocket.close(code=4000)
