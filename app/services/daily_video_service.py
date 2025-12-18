"""
Daily.co HIPAA-Compliant Video Service - Phase 12 Enhanced
============================================================

Production-grade video consultation service with:
- Deterministic room naming (appt-{uuid})
- Role-based token generation
- External provider support (Zoom/Meet)
- HIPAA-compliant session management
- Webhook signature validation
"""

from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from decimal import Decimal
import requests
import hmac
import hashlib
import logging
import re

from app.config import settings
from app.services.access_control import HIPAAAuditLogger, PHICategory

logger = logging.getLogger(__name__)


class DailyVideoService:
    """
    Production-grade Daily.co video service for HIPAA-compliant consultations.
    
    Features:
    - Deterministic room naming: appt-{appointment_id}
    - Role-based tokens (doctor = owner, patient = participant)
    - Webhook signature validation
    - External provider fallback (Zoom/Meet)
    - Automatic room cleanup
    """
    
    BASE_URL = "https://api.daily.co/v1"
    
    def __init__(self):
        """Initialize Daily.co service with API key from environment"""
        self.api_key = settings.DAILY_API_KEY
        self.domain = settings.DAILY_DOMAIN
        self.webhook_secret = settings.DAILY_WEBHOOK_SECRET
        
        if not self.api_key:
            logger.warning("DAILY_API_KEY not set - video features will be limited")
        
        self.headers = {
            "Authorization": f"Bearer {self.api_key}" if self.api_key else "",
            "Content-Type": "application/json"
        }
    
    @staticmethod
    def generate_room_name(appointment_id: str) -> str:
        """Generate deterministic room name from appointment ID"""
        clean_id = re.sub(r'[^a-zA-Z0-9-]', '', str(appointment_id))
        return f"appt-{clean_id}"
    
    @staticmethod
    def extract_appointment_id(room_name: str) -> Optional[str]:
        """Extract appointment ID from room name"""
        if room_name and room_name.startswith("appt-"):
            return room_name[5:]
        return None
    
    def validate_webhook_signature(self, payload: bytes, signature: str) -> bool:
        """Validate Daily.co webhook signature for security"""
        if not self.webhook_secret:
            logger.warning("DAILY_WEBHOOK_SECRET not set - webhook validation skipped")
            return True
        
        expected = hmac.new(
            self.webhook_secret.encode(),
            payload,
            hashlib.sha256
        ).hexdigest()
        
        return hmac.compare_digest(expected, signature)
    
    def create_room(
        self,
        appointment_id: str,
        duration_minutes: int = 60,
        enable_chat: bool = True,
        enable_recording: bool = False,
        max_participants: int = 2
    ) -> Dict[str, Any]:
        """
        Create a HIPAA-compliant video room for an appointment.
        
        Uses deterministic naming: appt-{appointment_id}
        
        Returns:
            Dictionary with room_name, room_url, expires_at
        """
        if not self.api_key:
            raise ValueError("DAILY_API_KEY not configured")
        
        room_name = self.generate_room_name(appointment_id)
        exp_time = datetime.utcnow() + timedelta(minutes=duration_minutes + 30)
        
        payload = {
            "name": room_name,
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
                "max_participants": max_participants,
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
        
        if response.status_code == 400 and "already exists" in response.text.lower():
            room_info = self.get_room_info(room_name)
            return {
                "room_name": room_info["name"],
                "room_url": room_info["url"],
                "expires_at": exp_time.isoformat(),
                "created": False
            }
        
        if response.status_code != 200:
            logger.error(f"Failed to create room: {response.text}")
            raise Exception(f"Failed to create room: {response.text}")
        
        room_data = response.json()
        
        logger.info(f"Created Daily room: {room_name}")
        
        return {
            "room_name": room_data["name"],
            "room_url": room_data["url"],
            "expires_at": exp_time.isoformat(),
            "created": True
        }
    
    def create_meeting_token(
        self,
        room_name: str,
        user_id: str,
        user_name: str,
        is_owner: bool = False,
        exp_seconds: int = 7200
    ) -> str:
        """
        Create a meeting token for a specific participant.
        
        Args:
            room_name: Daily.co room name
            user_id: Internal user ID (for tracking, not displayed)
            user_name: Display name (NO PHI - use "Doctor" or "Patient")
            is_owner: Whether user has owner privileges (doctors)
            exp_seconds: Token expiry in seconds (default 2 hours)
            
        Returns:
            Meeting token string
        """
        if not self.api_key:
            raise ValueError("DAILY_API_KEY not configured")
        
        exp_time = datetime.utcnow() + timedelta(seconds=exp_seconds)
        
        payload = {
            "properties": {
                "room_name": room_name,
                "user_name": user_name,
                "user_id": user_id,
                "is_owner": is_owner,
                "enable_screenshare": is_owner,
                "enable_recording": is_owner,
                "exp": int(exp_time.timestamp()),
            }
        }
        
        response = requests.post(
            f"{self.BASE_URL}/meeting-tokens",
            headers=self.headers,
            json=payload
        )
        
        if response.status_code != 200:
            logger.error(f"Failed to create meeting token: {response.text}")
            raise Exception(f"Failed to create meeting token: {response.text}")
        
        return response.json()["token"]
    
    def get_room_info(self, room_name: str) -> Dict[str, Any]:
        """Get information about an existing room"""
        response = requests.get(
            f"{self.BASE_URL}/rooms/{room_name}",
            headers=self.headers
        )
        
        if response.status_code != 200:
            raise Exception(f"Room not found: {room_name}")
        
        return response.json()
    
    def room_exists(self, room_name: str) -> bool:
        """Check if a room exists"""
        try:
            self.get_room_info(room_name)
            return True
        except Exception:
            return False
    
    def delete_room(self, room_name: str) -> bool:
        """
        Delete a room after consultation is complete.
        HIPAA requirement: minimize data retention.
        """
        response = requests.delete(
            f"{self.BASE_URL}/rooms/{room_name}",
            headers=self.headers
        )
        
        if response.status_code == 200:
            logger.info(f"Deleted Daily room: {room_name}")
            return True
        
        logger.warning(f"Failed to delete room {room_name}: {response.text}")
        return False
    
    def get_session_participants(self, room_name: str) -> Dict[str, Any]:
        """Get current participants in a room"""
        response = requests.get(
            f"{self.BASE_URL}/rooms/{room_name}/participants",
            headers=self.headers
        )
        
        if response.status_code != 200:
            raise Exception(f"Failed to get participants: {response.text}")
        
        return response.json()


class ExternalVideoProvider:
    """
    Handle external video providers (Zoom/Meet) for doctor-demand scenarios.
    
    These are only available when:
    1. Doctor has allow_external_video = true
    2. Doctor has configured the respective URLs
    3. Doctor explicitly selects external for an appointment
    """
    
    @staticmethod
    def validate_zoom_url(url: str) -> bool:
        """Validate Zoom join URL format"""
        if not url:
            return False
        return url.startswith("https://") and ("zoom.us" in url or "zoomgov.com" in url)
    
    @staticmethod
    def validate_meet_url(url: str) -> bool:
        """Validate Google Meet URL format"""
        if not url:
            return False
        return url.startswith("https://") and "meet.google.com" in url
    
    @staticmethod
    def validate_external_url(url: str, provider: str) -> bool:
        """Validate external provider URL"""
        if provider == "zoom":
            return ExternalVideoProvider.validate_zoom_url(url)
        elif provider == "meet":
            return ExternalVideoProvider.validate_meet_url(url)
        return False


class VideoUsageCalculator:
    """
    Calculate participant-minutes for billing.
    
    Formula: participant_minutes = num_participants × minutes_in_call
    Example: 1 doctor + 1 patient for 15 min = 2 × 15 = 30 participant-minutes
    """
    
    @staticmethod
    def calculate_session_minutes(duration_seconds: int) -> int:
        """Calculate billed minutes from duration (ceiling)"""
        import math
        return math.ceil(duration_seconds / 60)
    
    @staticmethod
    def calculate_cost(participant_minutes: int, rate_usd: Decimal = None) -> Decimal:
        """Calculate platform cost for participant minutes"""
        if rate_usd is None:
            rate_usd = Decimal(settings.DAILY_RATE_USD)
        return Decimal(participant_minutes) * rate_usd
    
    @staticmethod
    def calculate_overage(
        total_minutes: int,
        included_minutes: int,
        overage_rate: Decimal = None
    ) -> tuple[int, Decimal]:
        """
        Calculate overage minutes and cost.
        
        Returns:
            (overage_minutes, overage_cost_usd)
        """
        if overage_rate is None:
            overage_rate = Decimal(settings.OVERAGE_RATE_USD)
        
        overage_minutes = max(0, total_minutes - included_minutes)
        overage_cost = Decimal(overage_minutes) * overage_rate
        
        return overage_minutes, overage_cost
    
    @staticmethod
    def get_billing_month(dt: datetime = None) -> str:
        """Get billing month string (YYYY-MM)"""
        if dt is None:
            dt = datetime.utcnow()
        return dt.strftime("%Y-%m")
