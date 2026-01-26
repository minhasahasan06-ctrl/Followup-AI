"""
Voice Session Orchestrator

Manages the complete voice conversation lifecycle.
Coordinates ASR → LLM → TTS pipeline with real-time streaming.

Features:
- Session lifecycle management
- Turn-taking with interruption handling
- Streaming audio processing
- Action card generation
- Memory integration
"""

import logging
import asyncio
import uuid
from typing import Optional, Dict, Any, List, Callable, AsyncGenerator
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timezone

from app.services.voice.asr_adapter import get_asr_adapter, ASRAdapter, TranscriptResult
from app.services.voice.tts_adapter import get_tts_adapter, TTSAdapter, TTSVoice, TTSAudioChunk
from app.services.feature_flag_service import is_feature_enabled

logger = logging.getLogger(__name__)


class SessionState(str, Enum):
    """Voice session states"""
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    SPEAKING = "speaking"
    INTERRUPTED = "interrupted"
    ENDED = "ended"
    ERROR = "error"


class ConsentType(str, Enum):
    """Types of consent for voice sessions"""
    RECORDING = "recording"
    MEMORY = "memory"
    TRANSCRIPTION = "transcription"


@dataclass
class SessionConfig:
    """Configuration for a voice session"""
    user_id: str
    persona: str = "clona"
    voice: TTSVoice = TTSVoice.NOVA
    language: str = "en"
    push_to_talk: bool = False
    always_listening: bool = True
    consent_recording: bool = False
    consent_memory: bool = False
    enable_interruptions: bool = True


@dataclass
class VoiceSession:
    """Active voice session data"""
    session_id: str
    config: SessionConfig
    state: SessionState = SessionState.IDLE
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_activity: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    transcript_history: List[Dict[str, Any]] = field(default_factory=list)
    action_cards: List[Dict[str, Any]] = field(default_factory=list)
    asr_session_id: Optional[str] = None
    current_tts_text: Optional[str] = None
    is_user_speaking: bool = False
    is_agent_speaking: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ActionCard:
    """Action card generated from voice conversation"""
    card_id: str
    title: str
    action_type: str
    body: str
    buttons: List[Dict[str, str]]
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class SessionOrchestrator:
    """
    Orchestrates voice conversation sessions.
    
    Manages:
    - Session lifecycle (create, update, end)
    - Audio streaming pipeline (ASR → LLM → TTS)
    - Turn-taking and interruptions
    - Action card generation
    - Memory integration
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._sessions: Dict[str, VoiceSession] = {}
        self._asr_adapter: Optional[ASRAdapter] = None
        self._tts_adapter: Optional[TTSAdapter] = None
        self._event_handlers: Dict[str, List[Callable]] = {}
    
    def _get_asr(self) -> ASRAdapter:
        """Lazy initialization of ASR adapter"""
        if self._asr_adapter is None:
            use_mock = not is_feature_enabled("enableVoice")
            self._asr_adapter = get_asr_adapter(use_mock=use_mock)
        return self._asr_adapter
    
    def _get_tts(self) -> TTSAdapter:
        """Lazy initialization of TTS adapter"""
        if self._tts_adapter is None:
            use_mock = not is_feature_enabled("enableVoice")
            self._tts_adapter = get_tts_adapter(use_mock=use_mock)
        return self._tts_adapter
    
    async def create_session(self, config: SessionConfig) -> VoiceSession:
        """
        Create a new voice session.
        
        Args:
            config: Session configuration
            
        Returns:
            Created VoiceSession
        """
        session_id = str(uuid.uuid4())
        
        session = VoiceSession(
            session_id=session_id,
            config=config,
            state=SessionState.IDLE
        )
        
        self._sessions[session_id] = session
        
        self.logger.info(
            f"Created voice session {session_id} for user {config.user_id} "
            f"with persona {config.persona}"
        )
        
        await self._emit_event("session_created", session)
        
        return session
    
    async def start_listening(self, session_id: str) -> bool:
        """
        Start listening for user speech.
        
        Args:
            session_id: Session to start listening
            
        Returns:
            True if successfully started
        """
        session = self._sessions.get(session_id)
        if not session:
            return False
        
        if session.state == SessionState.LISTENING:
            return True
        
        try:
            asr = self._get_asr()
            asr_session_id = await asr.start_streaming(
                language=session.config.language,
                on_interim=lambda r: asyncio.create_task(
                    self._handle_interim_transcript(session_id, r)
                ),
                on_final=lambda r: asyncio.create_task(
                    self._handle_final_transcript(session_id, r)
                )
            )
            
            session.asr_session_id = asr_session_id
            session.state = SessionState.LISTENING
            session.is_user_speaking = False
            session.last_activity = datetime.now(timezone.utc)
            
            await self._emit_event("listening_started", session)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start listening: {e}")
            session.state = SessionState.ERROR
            return False
    
    async def send_audio(
        self, 
        session_id: str, 
        audio_chunk: bytes
    ) -> Optional[TranscriptResult]:
        """
        Send audio chunk to session for processing.
        
        Args:
            session_id: Target session
            audio_chunk: Raw audio bytes
            
        Returns:
            TranscriptResult if available
        """
        session = self._sessions.get(session_id)
        if not session or not session.asr_session_id:
            return None
        
        if session.state == SessionState.SPEAKING and session.config.enable_interruptions:
            await self._handle_interruption(session)
        
        session.is_user_speaking = True
        session.last_activity = datetime.now(timezone.utc)
        
        asr = self._get_asr()
        result = await asr.send_audio_chunk(session.asr_session_id, audio_chunk)
        
        return result
    
    async def stop_listening(self, session_id: str) -> Optional[TranscriptResult]:
        """
        Stop listening and get final transcript.
        
        Args:
            session_id: Session to stop
            
        Returns:
            Final TranscriptResult
        """
        session = self._sessions.get(session_id)
        if not session or not session.asr_session_id:
            return None
        
        try:
            asr = self._get_asr()
            result = await asr.stop_streaming(session.asr_session_id)
            
            session.asr_session_id = None
            session.state = SessionState.PROCESSING
            session.is_user_speaking = False
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to stop listening: {e}")
            return None
    
    async def generate_response(
        self,
        session_id: str,
        user_text: str,
        stream: bool = True
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Generate agent response with streaming LLM and TTS.
        
        Args:
            session_id: Target session
            user_text: User's transcribed speech
            stream: Whether to stream the response
            
        Yields:
            Response events (llm_partial, tts_chunk, action_card, etc.)
        """
        session = self._sessions.get(session_id)
        if not session:
            return
        
        session.state = SessionState.PROCESSING
        session.transcript_history.append({
            "role": "user",
            "content": user_text,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        try:
            llm_response = await self._generate_llm_response(session, user_text)
            
            session.transcript_history.append({
                "role": "assistant",
                "content": llm_response["text"],
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            yield {
                "type": "llm_final",
                "text": llm_response["text"],
                "metadata": llm_response.get("metadata", {})
            }
            
            if llm_response.get("action_card"):
                yield {
                    "type": "action_card",
                    "card": llm_response["action_card"]
                }
                session.action_cards.append(llm_response["action_card"])
            
            session.state = SessionState.SPEAKING
            session.is_agent_speaking = True
            session.current_tts_text = llm_response["text"]
            
            tts = self._get_tts()
            voice = session.config.voice
            
            async for chunk in tts.stream_synthesize(
                llm_response["text"],
                voice=voice
            ):
                if session.state == SessionState.INTERRUPTED:
                    break
                    
                yield {
                    "type": "tts_chunk",
                    "audio": chunk.audio_data,
                    "chunk_index": chunk.chunk_index,
                    "is_final": chunk.is_final,
                    "format": chunk.format
                }
            
            session.state = SessionState.IDLE
            session.is_agent_speaking = False
            session.current_tts_text = None
            
            yield {"type": "response_complete"}
            
        except Exception as e:
            self.logger.error(f"Response generation failed: {e}")
            session.state = SessionState.ERROR
            yield {"type": "error", "message": str(e)}
    
    async def _generate_llm_response(
        self,
        session: VoiceSession,
        user_text: str
    ) -> Dict[str, Any]:
        """Generate LLM response using agent engine"""
        try:
            from app.config import get_openai_client, check_openai_baa_compliance
            check_openai_baa_compliance()
            client = get_openai_client()
            
            system_prompt = self._get_persona_prompt(session.config.persona)
            
            messages = [{"role": "system", "content": system_prompt}]
            
            for msg in session.transcript_history[-10:]:
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
            
            messages.append({"role": "user", "content": user_text})
            
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.7,
                max_tokens=500
            )
            
            response_text = response.choices[0].message.content
            
            action_card = self._detect_action_card(response_text)
            
            return {
                "text": response_text,
                "action_card": action_card,
                "metadata": {
                    "model": "gpt-4o",
                    "persona": session.config.persona
                }
            }
            
        except Exception as e:
            self.logger.error(f"LLM generation failed: {e}")
            return {
                "text": "I apologize, but I'm having trouble responding right now. Please try again.",
                "action_card": None
            }
    
    def _get_persona_prompt(self, persona: str) -> str:
        """Get system prompt for persona"""
        personas = {
            "clona": """You are Agent Clona, a caring and empathetic health companion. 
You speak warmly and supportively, helping patients with their health concerns.
You're designed for voice conversation, so keep responses concise and natural.
Always prioritize patient safety and recommend professional care when appropriate.

If you detect any concerning symptoms, clearly express concern and recommend appropriate action.
Keep responses under 3-4 sentences for natural conversation flow.""",

            "lysa": """You are Assistant Lysa, a professional medical assistant helping doctors.
You speak clearly and efficiently, providing clinical support.
You help with patient summaries, scheduling, and clinical workflows.
Keep responses professional but friendly.

For voice conversation, keep responses concise and actionable.
If you're generating an action (like drafting an email), mention it briefly."""
        }
        
        return personas.get(persona.lower(), personas["clona"])
    
    def _detect_action_card(self, text: str) -> Optional[Dict[str, Any]]:
        """Detect if response contains an actionable item"""
        action_triggers = {
            "draft email": "email",
            "schedule": "calendar",
            "set reminder": "reminder",
            "send message": "message",
            "book appointment": "appointment"
        }
        
        text_lower = text.lower()
        
        for trigger, action_type in action_triggers.items():
            if trigger in text_lower:
                return {
                    "card_id": str(uuid.uuid4()),
                    "title": f"{action_type.title()} Action",
                    "action_type": action_type,
                    "body": f"Would you like me to help with this {action_type}?",
                    "buttons": [
                        {"label": "Proceed", "action": "confirm"},
                        {"label": "Edit", "action": "edit"},
                        {"label": "Cancel", "action": "cancel"}
                    ],
                    "source_text": text[:200]
                }
        
        return None
    
    async def _handle_interruption(self, session: VoiceSession) -> None:
        """Handle user interruption during agent speech"""
        if not session.config.enable_interruptions:
            return
        
        self.logger.info(f"Handling interruption in session {session.session_id}")
        
        session.state = SessionState.INTERRUPTED
        session.is_agent_speaking = False
        
        await self._emit_event("interrupted", session)
    
    async def _handle_interim_transcript(
        self,
        session_id: str,
        result: TranscriptResult
    ) -> None:
        """Handle interim transcript from ASR"""
        session = self._sessions.get(session_id)
        if not session:
            return
        
        await self._emit_event("interim_transcript", {
            "session_id": session_id,
            "text": result.text,
            "confidence": result.confidence
        })
    
    async def _handle_final_transcript(
        self,
        session_id: str,
        result: TranscriptResult
    ) -> None:
        """Handle final transcript from ASR"""
        session = self._sessions.get(session_id)
        if not session:
            return
        
        await self._emit_event("final_transcript", {
            "session_id": session_id,
            "text": result.text,
            "confidence": result.confidence
        })
    
    async def end_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        End a voice session and return summary.
        
        Args:
            session_id: Session to end
            
        Returns:
            Session summary with transcripts and action cards
        """
        session = self._sessions.get(session_id)
        if not session:
            return None
        
        if session.asr_session_id:
            try:
                asr = self._get_asr()
                await asr.stop_streaming(session.asr_session_id)
            except Exception:
                pass
        
        session.state = SessionState.ENDED
        
        summary = {
            "session_id": session_id,
            "user_id": session.config.user_id,
            "persona": session.config.persona,
            "duration_seconds": (datetime.now(timezone.utc) - session.created_at).total_seconds(),
            "transcript_count": len(session.transcript_history),
            "action_cards": session.action_cards,
            "transcripts": session.transcript_history,
            "ended_at": datetime.now(timezone.utc).isoformat()
        }
        
        await self._emit_event("session_ended", summary)
        
        del self._sessions[session_id]
        
        self.logger.info(f"Ended voice session {session_id}")
        
        return summary
    
    def get_session(self, session_id: str) -> Optional[VoiceSession]:
        """Get session by ID"""
        return self._sessions.get(session_id)
    
    def get_user_sessions(self, user_id: str) -> List[VoiceSession]:
        """Get all sessions for a user"""
        return [s for s in self._sessions.values() if s.config.user_id == user_id]
    
    def on(self, event: str, handler: Callable) -> None:
        """Register event handler"""
        if event not in self._event_handlers:
            self._event_handlers[event] = []
        self._event_handlers[event].append(handler)
    
    async def _emit_event(self, event: str, data: Any) -> None:
        """Emit event to registered handlers"""
        handlers = self._event_handlers.get(event, [])
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(data)
                else:
                    handler(data)
            except Exception as e:
                self.logger.error(f"Event handler error for {event}: {e}")


_session_orchestrator: Optional[SessionOrchestrator] = None


def get_session_orchestrator() -> SessionOrchestrator:
    """Get singleton session orchestrator"""
    global _session_orchestrator
    if _session_orchestrator is None:
        _session_orchestrator = SessionOrchestrator()
    return _session_orchestrator
