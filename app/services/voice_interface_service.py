from typing import Dict
import openai
from app.config import settings, check_openai_baa_compliance


class VoiceInterfaceService:
    def __init__(self):
        self.openai_enabled = check_openai_baa_compliance()
        if self.openai_enabled and settings.OPENAI_API_KEY:
            openai.api_key = settings.OPENAI_API_KEY
    
    async def transcribe_audio(self, audio_file_path: str) -> Dict:
        if not self.openai_enabled:
            raise Exception("OpenAI BAA not signed. Voice transcription unavailable.")
        
        try:
            with open(audio_file_path, "rb") as audio_file:
                transcription = await openai.Audio.atranscribe(
                    model="whisper-1",
                    file=audio_file,
                    language="en"
                )
            
            return {
                "text": transcription.text,
                "duration": 0,
                "language": "en"
            }
        except Exception as e:
            print(f"Error transcribing audio: {e}")
            raise
    
    async def generate_speech(self, text: str, voice: str = "nova") -> bytes:
        if not self.openai_enabled:
            raise Exception("OpenAI BAA not signed. Speech synthesis unavailable.")
        
        try:
            response = await openai.Audio.speech.acreate(
                model="tts-1",
                voice=voice,
                input=text
            )
            
            return response.content
        except Exception as e:
            print(f"Error generating speech: {e}")
            raise
    
    async def process_voice_followup(self, patient_id: str, audio_file_path: str) -> Dict:
        transcription = await self.transcribe_audio(audio_file_path)
        
        return {
            "patient_id": patient_id,
            "transcription": transcription["text"],
            "analysis": "Voice followup processed successfully",
            "success": True
        }
