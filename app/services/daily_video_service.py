"""
Daily.co HIPAA-Compliant Video Service for Live Physical Examination Monitoring.
Integrates with Daily.co API for secure telehealth video sessions.
"""

from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import requests
import os


class DailyVideoService:
    """
    Service for managing HIPAA-compliant video sessions using Daily.co.
    
    HIPAA Features:
    - BAA signed with Daily.co (required - apply via Daily.co portal)
    - End-to-end encryption (default)
    - No PHI in URLs (randomized room names)
    - No cookies or local storage in HIPAA mode
    - Recording disabled by default (custom storage available)
    - Automatic data scrubbing
    """
    
    BASE_URL = "https://api.daily.co/v1"
    
    def __init__(self):
        """Initialize Daily.co service with API key from environment"""
        self.api_key = os.getenv("DAILY_API_KEY")
        if not self.api_key:
            raise ValueError("DAILY_API_KEY environment variable not set")
        
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def create_consultation_room(
        self,
        patient_id: str,
        doctor_id: str,
        duration_minutes: int = 60,
        enable_chat: bool = True,
        enable_recording: bool = False
    ) -> Dict[str, Any]:
        """
        Create a HIPAA-compliant video room for doctor-patient consultation.
        
        Args:
            patient_id: Patient's unique identifier (NOT included in room name)
            doctor_id: Doctor's unique identifier (NOT included in room name)
            duration_minutes: Session duration (default 60 minutes)
            enable_chat: Enable HIPAA-compliant text chat
            enable_recording: Enable recording (requires custom HIPAA-compliant storage)
            
        Returns:
            Dictionary with room details including URL and access tokens
        """
        exp_time = datetime.utcnow() + timedelta(minutes=duration_minutes + 30)
        
        payload = {
            "privacy": "private",
            "properties": {
                "enable_chat": enable_chat,
                "enable_recording": enable_recording,
                "enable_screenshare": True,
                "enable_knocking": True,
                "start_video_off": False,
                "start_audio_off": False,
                "enable_prejoin_ui": True,
                "exp": int(exp_time.timestamp()),
                "eject_at_room_exp": True,
                "max_participants": 2,
                "autojoin": False,
                "enable_network_ui": True,
                "enable_noise_cancellation_ui": True,
            }
        }
        
        response = requests.post(
            f"{self.BASE_URL}/rooms",
            headers=self.headers,
            json=payload
        )
        
        if response.status_code != 200:
            raise Exception(f"Failed to create room: {response.text}")
        
        room_data = response.json()
        
        patient_token = self._create_meeting_token(
            room_name=room_data["name"],
            user_name="Patient",
            is_owner=False
        )
        
        doctor_token = self._create_meeting_token(
            room_name=room_data["name"],
            user_name="Doctor",
            is_owner=True
        )
        
        return {
            "room_name": room_data["name"],
            "room_url": room_data["url"],
            "patient_token": patient_token,
            "doctor_token": doctor_token,
            "expires_at": exp_time.isoformat(),
            "config": {
                "chat_enabled": enable_chat,
                "recording_enabled": enable_recording,
                "max_duration_minutes": duration_minutes
            }
        }
    
    def _create_meeting_token(
        self,
        room_name: str,
        user_name: str,
        is_owner: bool = False
    ) -> str:
        """
        Create a meeting token for a specific participant.
        
        Args:
            room_name: Daily.co room name
            user_name: Display name (NO PHI - use generic names like "Patient" or "Doctor")
            is_owner: Whether user has owner privileges
            
        Returns:
            Meeting token string
        """
        payload = {
            "properties": {
                "room_name": room_name,
                "user_name": user_name,
                "is_owner": is_owner,
                "enable_screenshare": is_owner,
                "enable_recording": is_owner,
            }
        }
        
        response = requests.post(
            f"{self.BASE_URL}/meeting-tokens",
            headers=self.headers,
            json=payload
        )
        
        if response.status_code != 200:
            raise Exception(f"Failed to create meeting token: {response.text}")
        
        return response.json()["token"]
    
    def get_room_info(self, room_name: str) -> Dict[str, Any]:
        """Get information about an existing room"""
        response = requests.get(
            f"{self.BASE_URL}/rooms/{room_name}",
            headers=self.headers
        )
        
        if response.status_code != 200:
            raise Exception(f"Failed to get room info: {response.text}")
        
        return response.json()
    
    def delete_room(self, room_name: str) -> bool:
        """
        Delete a room after consultation is complete.
        Recommended for HIPAA compliance to minimize data retention.
        """
        response = requests.delete(
            f"{self.BASE_URL}/rooms/{room_name}",
            headers=self.headers
        )
        
        return response.status_code == 200
    
    def get_session_participants(self, room_name: str) -> Dict[str, Any]:
        """
        Get current participants in a room.
        Useful for monitoring active consultations.
        """
        response = requests.get(
            f"{self.BASE_URL}/rooms/{room_name}/participants",
            headers=self.headers
        )
        
        if response.status_code != 200:
            raise Exception(f"Failed to get participants: {response.text}")
        
        return response.json()
