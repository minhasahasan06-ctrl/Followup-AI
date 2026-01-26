"""
ASR (Automatic Speech Recognition) Adapter

Streaming speech-to-text with OpenAI Whisper API.
Provides mockable interface for testing and swappable backends.

Features:
- Streaming transcription with interim results
- Multi-language support
- Confidence scoring
- Mockable for testing
"""

import logging
import asyncio
import base64
from typing import Optional, AsyncGenerator, Dict, Any, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timezone
import io

logger = logging.getLogger(__name__)


class TranscriptType(str, Enum):
    """Type of transcript result"""
    INTERIM = "interim"
    FINAL = "final"


@dataclass
class TranscriptResult:
    """Result from speech recognition"""
    text: str
    is_final: bool
    confidence: float
    language: Optional[str] = None
    timestamp: datetime = None
    segment_id: Optional[str] = None
    duration_ms: Optional[int] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)


class ASRAdapter(ABC):
    """Abstract base class for ASR adapters"""
    
    @abstractmethod
    async def transcribe_audio(
        self,
        audio_data: bytes,
        audio_format: str = "webm",
        language: Optional[str] = None
    ) -> TranscriptResult:
        """Transcribe audio data to text"""
        pass
    
    @abstractmethod
    async def start_streaming(
        self,
        language: Optional[str] = None,
        on_interim: Optional[Callable[[TranscriptResult], None]] = None,
        on_final: Optional[Callable[[TranscriptResult], None]] = None
    ) -> str:
        """Start a streaming transcription session"""
        pass
    
    @abstractmethod
    async def send_audio_chunk(
        self,
        session_id: str,
        audio_chunk: bytes
    ) -> Optional[TranscriptResult]:
        """Send an audio chunk to streaming session"""
        pass
    
    @abstractmethod
    async def stop_streaming(self, session_id: str) -> TranscriptResult:
        """Stop streaming and get final transcript"""
        pass


class OpenAIWhisperAdapter(ASRAdapter):
    """
    OpenAI Whisper API adapter for speech recognition.
    
    Uses Whisper API for high-quality transcription.
    Simulates streaming by processing audio in chunks.
    """
    
    SUPPORTED_FORMATS = ["webm", "mp3", "wav", "m4a", "ogg", "flac"]
    MAX_FILE_SIZE = 25 * 1024 * 1024
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._sessions: Dict[str, Dict[str, Any]] = {}
        self._client = None
    
    def _get_client(self):
        """Lazy initialization of OpenAI client"""
        if self._client is None:
            from app.config import get_openai_client, check_openai_baa_compliance
            check_openai_baa_compliance()
            self._client = get_openai_client()
        return self._client
    
    async def transcribe_audio(
        self,
        audio_data: bytes,
        audio_format: str = "webm",
        language: Optional[str] = None
    ) -> TranscriptResult:
        """
        Transcribe audio data using Whisper API.
        
        Args:
            audio_data: Raw audio bytes
            audio_format: Audio format (webm, mp3, wav, etc.)
            language: Optional language code (e.g., 'en', 'es')
            
        Returns:
            TranscriptResult with transcribed text
        """
        if len(audio_data) > self.MAX_FILE_SIZE:
            raise ValueError(f"Audio file too large: {len(audio_data)} bytes (max {self.MAX_FILE_SIZE})")
        
        if audio_format not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format: {audio_format}")
        
        try:
            client = self._get_client()
            
            audio_file = io.BytesIO(audio_data)
            audio_file.name = f"audio.{audio_format}"
            
            params = {
                "model": "whisper-1",
                "file": audio_file,
                "response_format": "verbose_json"
            }
            
            if language:
                params["language"] = language
            
            response = client.audio.transcriptions.create(**params)
            
            confidence = 0.95
            if hasattr(response, 'segments') and response.segments:
                confidences = [s.get('confidence', 0.95) for s in response.segments if 'confidence' in s]
                if confidences:
                    confidence = sum(confidences) / len(confidences)
            
            return TranscriptResult(
                text=response.text,
                is_final=True,
                confidence=confidence,
                language=getattr(response, 'language', language),
                duration_ms=int(getattr(response, 'duration', 0) * 1000) if hasattr(response, 'duration') else None
            )
            
        except Exception as e:
            self.logger.error(f"Whisper transcription failed: {e}")
            raise
    
    async def start_streaming(
        self,
        language: Optional[str] = None,
        on_interim: Optional[Callable[[TranscriptResult], None]] = None,
        on_final: Optional[Callable[[TranscriptResult], None]] = None
    ) -> str:
        """
        Start a streaming transcription session.
        
        Note: Whisper doesn't support true streaming, so we buffer audio
        and process in chunks to simulate streaming behavior.
        """
        import uuid
        session_id = str(uuid.uuid4())
        
        self._sessions[session_id] = {
            "language": language,
            "on_interim": on_interim,
            "on_final": on_final,
            "audio_buffer": bytearray(),
            "transcripts": [],
            "started_at": datetime.now(timezone.utc),
            "chunk_count": 0,
            "last_process_time": None
        }
        
        self.logger.info(f"Started ASR session: {session_id}")
        return session_id
    
    async def send_audio_chunk(
        self,
        session_id: str,
        audio_chunk: bytes
    ) -> Optional[TranscriptResult]:
        """
        Send audio chunk to session buffer.
        
        Processes accumulated audio when buffer reaches threshold.
        """
        session = self._sessions.get(session_id)
        if not session:
            raise ValueError(f"Unknown session: {session_id}")
        
        session["audio_buffer"].extend(audio_chunk)
        session["chunk_count"] += 1
        
        buffer_size = len(session["audio_buffer"])
        if buffer_size >= 32000:
            try:
                result = await self.transcribe_audio(
                    bytes(session["audio_buffer"]),
                    audio_format="webm",
                    language=session["language"]
                )
                
                interim_result = TranscriptResult(
                    text=result.text,
                    is_final=False,
                    confidence=result.confidence * 0.9,
                    language=result.language,
                    segment_id=f"{session_id}-{session['chunk_count']}"
                )
                
                session["transcripts"].append(interim_result)
                
                if session["on_interim"]:
                    session["on_interim"](interim_result)
                
                session["audio_buffer"] = bytearray()
                session["last_process_time"] = datetime.now(timezone.utc)
                
                return interim_result
                
            except Exception as e:
                self.logger.warning(f"Chunk processing failed: {e}")
        
        return None
    
    async def stop_streaming(self, session_id: str) -> TranscriptResult:
        """
        Stop streaming session and get final transcript.
        """
        session = self._sessions.get(session_id)
        if not session:
            raise ValueError(f"Unknown session: {session_id}")
        
        try:
            final_result = None
            
            if len(session["audio_buffer"]) > 0:
                final_result = await self.transcribe_audio(
                    bytes(session["audio_buffer"]),
                    audio_format="webm",
                    language=session["language"]
                )
            
            all_texts = [t.text for t in session["transcripts"]]
            if final_result:
                all_texts.append(final_result.text)
            
            combined_text = " ".join(all_texts)
            
            result = TranscriptResult(
                text=combined_text,
                is_final=True,
                confidence=0.95,
                language=session["language"],
                segment_id=f"{session_id}-final"
            )
            
            if session["on_final"]:
                session["on_final"](result)
            
            return result
            
        finally:
            del self._sessions[session_id]
            self.logger.info(f"Stopped ASR session: {session_id}")


class MockASRAdapter(ASRAdapter):
    """
    Mock ASR adapter for testing.
    
    Simulates transcription with configurable responses.
    """
    
    def __init__(
        self,
        mock_responses: Optional[list[str]] = None,
        latency_ms: int = 100
    ):
        self.mock_responses = mock_responses or [
            "Hello, how can I help you today?",
            "I understand you're not feeling well.",
            "Let me connect you with your doctor."
        ]
        self.latency_ms = latency_ms
        self._response_index = 0
        self._sessions: Dict[str, Dict[str, Any]] = {}
    
    async def transcribe_audio(
        self,
        audio_data: bytes,
        audio_format: str = "webm",
        language: Optional[str] = None
    ) -> TranscriptResult:
        await asyncio.sleep(self.latency_ms / 1000)
        
        text = self.mock_responses[self._response_index % len(self.mock_responses)]
        self._response_index += 1
        
        return TranscriptResult(
            text=text,
            is_final=True,
            confidence=0.95,
            language=language or "en"
        )
    
    async def start_streaming(
        self,
        language: Optional[str] = None,
        on_interim: Optional[Callable[[TranscriptResult], None]] = None,
        on_final: Optional[Callable[[TranscriptResult], None]] = None
    ) -> str:
        import uuid
        session_id = str(uuid.uuid4())
        
        self._sessions[session_id] = {
            "language": language,
            "on_interim": on_interim,
            "on_final": on_final,
            "text_buffer": ""
        }
        
        return session_id
    
    async def send_audio_chunk(
        self,
        session_id: str,
        audio_chunk: bytes
    ) -> Optional[TranscriptResult]:
        session = self._sessions.get(session_id)
        if not session:
            return None
        
        await asyncio.sleep(self.latency_ms / 1000)
        
        text = self.mock_responses[self._response_index % len(self.mock_responses)]
        words = text.split()
        
        word_index = len(session["text_buffer"].split()) if session["text_buffer"] else 0
        if word_index < len(words):
            new_words = words[word_index:word_index + 2]
            session["text_buffer"] += " " + " ".join(new_words)
            
            result = TranscriptResult(
                text=session["text_buffer"].strip(),
                is_final=False,
                confidence=0.85,
                language=session["language"]
            )
            
            if session["on_interim"]:
                session["on_interim"](result)
            
            return result
        
        return None
    
    async def stop_streaming(self, session_id: str) -> TranscriptResult:
        session = self._sessions.get(session_id)
        if not session:
            return TranscriptResult(text="", is_final=True, confidence=0)
        
        text = self.mock_responses[self._response_index % len(self.mock_responses)]
        self._response_index += 1
        
        result = TranscriptResult(
            text=text,
            is_final=True,
            confidence=0.95,
            language=session["language"]
        )
        
        if session["on_final"]:
            session["on_final"](result)
        
        del self._sessions[session_id]
        return result


def get_asr_adapter(use_mock: bool = False) -> ASRAdapter:
    """Factory function to get ASR adapter"""
    if use_mock:
        return MockASRAdapter()
    return OpenAIWhisperAdapter()
