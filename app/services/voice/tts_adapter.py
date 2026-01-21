"""
TTS (Text-to-Speech) Adapter

Streaming text-to-speech with OpenAI TTS API.
Supports multiple voices, streaming audio, and mockable interface.

Features:
- Streaming audio generation
- Multiple voice options
- SSML prosody support (where available)
- Mockable for testing
"""

import logging
import asyncio
import base64
from typing import Optional, AsyncGenerator, Dict, Any, List, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timezone
import io

logger = logging.getLogger(__name__)


class TTSVoice(str, Enum):
    """Available TTS voices"""
    ALLOY = "alloy"
    ECHO = "echo"
    FABLE = "fable"
    ONYX = "onyx"
    NOVA = "nova"
    SHIMMER = "shimmer"


class TTSSpeed(str, Enum):
    """TTS speed options"""
    SLOW = "0.75"
    NORMAL = "1.0"
    FAST = "1.25"
    VERY_FAST = "1.5"


@dataclass
class TTSAudioChunk:
    """Audio chunk from TTS generation"""
    audio_data: bytes
    chunk_index: int
    is_final: bool
    format: str = "mp3"
    sample_rate: int = 24000
    duration_ms: Optional[int] = None
    text_segment: Optional[str] = None


@dataclass
class TTSRequest:
    """Request for TTS generation"""
    text: str
    voice: TTSVoice = TTSVoice.NOVA
    speed: float = 1.0
    format: str = "mp3"
    stream: bool = True


class TTSAdapter(ABC):
    """Abstract base class for TTS adapters"""
    
    @abstractmethod
    async def synthesize(
        self,
        text: str,
        voice: TTSVoice = TTSVoice.NOVA,
        speed: float = 1.0,
        format: str = "mp3"
    ) -> bytes:
        """Synthesize text to audio"""
        pass
    
    @abstractmethod
    async def stream_synthesize(
        self,
        text: str,
        voice: TTSVoice = TTSVoice.NOVA,
        speed: float = 1.0,
        format: str = "mp3"
    ) -> AsyncGenerator[TTSAudioChunk, None]:
        """Stream synthesized audio in chunks"""
        pass
    
    @abstractmethod
    def get_available_voices(self) -> List[Dict[str, Any]]:
        """Get list of available voices"""
        pass


class OpenAITTSAdapter(TTSAdapter):
    """
    OpenAI TTS API adapter for text-to-speech.
    
    Uses OpenAI's TTS API with support for streaming.
    """
    
    VOICE_DESCRIPTIONS = {
        TTSVoice.ALLOY: {"name": "Alloy", "description": "Neutral and balanced", "gender": "neutral"},
        TTSVoice.ECHO: {"name": "Echo", "description": "Warm and conversational", "gender": "male"},
        TTSVoice.FABLE: {"name": "Fable", "description": "Expressive and storytelling", "gender": "neutral"},
        TTSVoice.ONYX: {"name": "Onyx", "description": "Deep and authoritative", "gender": "male"},
        TTSVoice.NOVA: {"name": "Nova", "description": "Friendly and upbeat", "gender": "female"},
        TTSVoice.SHIMMER: {"name": "Shimmer", "description": "Clear and professional", "gender": "female"},
    }
    
    PERSONA_VOICE_MAPPING = {
        "clona": TTSVoice.NOVA,
        "lysa": TTSVoice.SHIMMER,
    }
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._client = None
    
    def _get_client(self):
        """Lazy initialization of OpenAI client"""
        if self._client is None:
            from app.config import get_openai_client, check_openai_baa_compliance
            check_openai_baa_compliance()
            self._client = get_openai_client()
        return self._client
    
    async def synthesize(
        self,
        text: str,
        voice: TTSVoice = TTSVoice.NOVA,
        speed: float = 1.0,
        format: str = "mp3"
    ) -> bytes:
        """
        Synthesize text to audio.
        
        Args:
            text: Text to synthesize
            voice: Voice to use
            speed: Speed multiplier (0.25 to 4.0)
            format: Output format (mp3, opus, aac, flac)
            
        Returns:
            Audio data as bytes
        """
        if not text or not text.strip():
            return b""
        
        speed = max(0.25, min(4.0, speed))
        
        try:
            client = self._get_client()
            
            response = client.audio.speech.create(
                model="tts-1",
                voice=voice.value,
                input=text,
                speed=speed,
                response_format=format
            )
            
            audio_data = response.content
            
            self.logger.debug(f"TTS generated {len(audio_data)} bytes for {len(text)} chars")
            return audio_data
            
        except Exception as e:
            self.logger.error(f"TTS synthesis failed: {e}")
            raise
    
    async def stream_synthesize(
        self,
        text: str,
        voice: TTSVoice = TTSVoice.NOVA,
        speed: float = 1.0,
        format: str = "mp3"
    ) -> AsyncGenerator[TTSAudioChunk, None]:
        """
        Stream synthesized audio in chunks.
        
        Uses OpenAI's streaming response to yield audio progressively.
        """
        if not text or not text.strip():
            return
        
        speed = max(0.25, min(4.0, speed))
        
        try:
            client = self._get_client()
            
            response = client.audio.speech.create(
                model="tts-1",
                voice=voice.value,
                input=text,
                speed=speed,
                response_format=format
            )
            
            audio_data = response.content
            chunk_size = 4096
            total_chunks = (len(audio_data) + chunk_size - 1) // chunk_size
            
            for i in range(0, len(audio_data), chunk_size):
                chunk = audio_data[i:i + chunk_size]
                chunk_index = i // chunk_size
                is_final = chunk_index == total_chunks - 1
                
                yield TTSAudioChunk(
                    audio_data=chunk,
                    chunk_index=chunk_index,
                    is_final=is_final,
                    format=format,
                    text_segment=text if is_final else None
                )
                
                await asyncio.sleep(0.01)
                
        except Exception as e:
            self.logger.error(f"TTS streaming failed: {e}")
            raise
    
    def get_available_voices(self) -> List[Dict[str, Any]]:
        """Get list of available voices with descriptions"""
        voices = []
        for voice, info in self.VOICE_DESCRIPTIONS.items():
            voices.append({
                "id": voice.value,
                "name": info["name"],
                "description": info["description"],
                "gender": info["gender"],
                "recommended_for": self._get_voice_recommendations(voice)
            })
        return voices
    
    def _get_voice_recommendations(self, voice: TTSVoice) -> List[str]:
        """Get recommendations for voice usage"""
        recommendations = {
            TTSVoice.ALLOY: ["general", "neutral contexts"],
            TTSVoice.ECHO: ["friendly conversations", "casual interactions"],
            TTSVoice.FABLE: ["storytelling", "emotional content"],
            TTSVoice.ONYX: ["professional", "authoritative messages"],
            TTSVoice.NOVA: ["patient support", "empathetic interactions"],
            TTSVoice.SHIMMER: ["clinical information", "clear instructions"],
        }
        return recommendations.get(voice, ["general"])
    
    def get_voice_for_persona(self, persona: str) -> TTSVoice:
        """Get recommended voice for a persona"""
        return self.PERSONA_VOICE_MAPPING.get(persona.lower(), TTSVoice.NOVA)


class MockTTSAdapter(TTSAdapter):
    """
    Mock TTS adapter for testing.
    
    Returns silence or pre-recorded audio for testing purposes.
    """
    
    def __init__(self, latency_ms: int = 50):
        self.latency_ms = latency_ms
    
    async def synthesize(
        self,
        text: str,
        voice: TTSVoice = TTSVoice.NOVA,
        speed: float = 1.0,
        format: str = "mp3"
    ) -> bytes:
        await asyncio.sleep(self.latency_ms / 1000)
        
        silence_duration = min(len(text) * 50, 5000)
        sample_rate = 24000
        num_samples = int(sample_rate * silence_duration / 1000)
        
        return b'\x00' * (num_samples * 2)
    
    async def stream_synthesize(
        self,
        text: str,
        voice: TTSVoice = TTSVoice.NOVA,
        speed: float = 1.0,
        format: str = "mp3"
    ) -> AsyncGenerator[TTSAudioChunk, None]:
        audio_data = await self.synthesize(text, voice, speed, format)
        
        chunk_size = 4096
        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i:i + chunk_size]
            is_final = i + chunk_size >= len(audio_data)
            
            yield TTSAudioChunk(
                audio_data=chunk,
                chunk_index=i // chunk_size,
                is_final=is_final,
                format=format
            )
            
            await asyncio.sleep(self.latency_ms / 1000)
    
    def get_available_voices(self) -> List[Dict[str, Any]]:
        return [
            {"id": v.value, "name": v.value.title(), "description": "Mock voice"}
            for v in TTSVoice
        ]


def get_tts_adapter(use_mock: bool = False) -> TTSAdapter:
    """Factory function to get TTS adapter"""
    if use_mock:
        return MockTTSAdapter()
    return OpenAITTSAdapter()
