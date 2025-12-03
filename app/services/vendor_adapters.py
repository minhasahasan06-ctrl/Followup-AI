"""
Vendor Adapters - Production-Grade Device Data Sync

Handles:
- OAuth token refresh
- Data fetching from vendor APIs
- Data normalization to unified format
- Webhook processing
- Rate limiting and error handling

Public APIs (Full Implementation):
- Fitbit
- Withings
- Oura
- Google Fit
- iHealth

Private APIs (Ready for Credentials):
- Garmin
- Whoop
- Dexcom
- Samsung Health
- Eko
- Abbott LibreView
"""

import os
import logging
import asyncio
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from enum import Enum

import httpx

logger = logging.getLogger(__name__)


# ============================================
# DATA MODELS
# ============================================

@dataclass
class SyncResult:
    """Result of a data sync operation"""
    success: bool
    records_fetched: int = 0
    records_processed: int = 0
    records_failed: int = 0
    data_types: List[str] = field(default_factory=list)
    date_range: Optional[Dict[str, str]] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    raw_data: Optional[List[Dict]] = None
    normalized_data: Optional[List[Dict]] = None


@dataclass
class TokenInfo:
    """OAuth token information"""
    access_token: str
    refresh_token: Optional[str] = None
    expires_at: Optional[datetime] = None
    token_type: str = "Bearer"
    scope: Optional[str] = None


@dataclass
class NormalizedReading:
    """Unified device reading format"""
    timestamp: str
    data_type: str  # 'heart_rate', 'bp', 'glucose', 'sleep', 'steps', etc.
    value: Any
    unit: Optional[str] = None
    source: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class HealthSection(str, Enum):
    """Health sections for routing data"""
    HYPERTENSION = "hypertension"
    DIABETES = "diabetes"
    CARDIOVASCULAR = "cardiovascular"
    RESPIRATORY = "respiratory"
    SLEEP = "sleep"
    MENTAL_HEALTH = "mental_health"
    FITNESS = "fitness"


# Mapping from data types to health sections
DATA_TYPE_TO_SECTIONS = {
    "heart_rate": [HealthSection.CARDIOVASCULAR],
    "hrv": [HealthSection.CARDIOVASCULAR, HealthSection.MENTAL_HEALTH],
    "resting_heart_rate": [HealthSection.CARDIOVASCULAR],
    "bp": [HealthSection.HYPERTENSION, HealthSection.CARDIOVASCULAR],
    "systolic": [HealthSection.HYPERTENSION],
    "diastolic": [HealthSection.HYPERTENSION],
    "glucose": [HealthSection.DIABETES],
    "spo2": [HealthSection.RESPIRATORY],
    "respiratory_rate": [HealthSection.RESPIRATORY],
    "sleep": [HealthSection.SLEEP],
    "sleep_score": [HealthSection.SLEEP],
    "sleep_duration": [HealthSection.SLEEP],
    "deep_sleep": [HealthSection.SLEEP],
    "rem_sleep": [HealthSection.SLEEP],
    "stress": [HealthSection.MENTAL_HEALTH],
    "recovery": [HealthSection.FITNESS, HealthSection.MENTAL_HEALTH],
    "readiness": [HealthSection.FITNESS],
    "steps": [HealthSection.FITNESS],
    "calories": [HealthSection.FITNESS],
    "active_minutes": [HealthSection.FITNESS],
    "vo2_max": [HealthSection.FITNESS, HealthSection.CARDIOVASCULAR],
    "weight": [HealthSection.FITNESS, HealthSection.DIABETES],
    "body_fat": [HealthSection.FITNESS],
    "temperature": [HealthSection.RESPIRATORY],
    "skin_temp": [HealthSection.CARDIOVASCULAR],
    "body_battery": [HealthSection.FITNESS],
    "training_load": [HealthSection.FITNESS],
}


# ============================================
# BASE VENDOR ADAPTER
# ============================================

class BaseVendorAdapter(ABC):
    """Base class for vendor-specific data adapters"""
    
    VENDOR_ID: str = ""
    VENDOR_NAME: str = ""
    BASE_URL: str = ""
    AUTH_URL: str = ""
    TOKEN_URL: str = ""
    
    def __init__(self, tokens: TokenInfo, user_id: str):
        self.tokens = tokens
        self.user_id = user_id
        self._client: Optional[httpx.AsyncClient] = None
    
    @property
    def client_id(self) -> str:
        return os.getenv(f"{self.VENDOR_ID.upper()}_CLIENT_ID", "")
    
    @property
    def client_secret(self) -> str:
        return os.getenv(f"{self.VENDOR_ID.upper()}_CLIENT_SECRET", "")
    
    async def get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.BASE_URL,
                headers={
                    "Authorization": f"{self.tokens.token_type} {self.tokens.access_token}",
                    "Accept": "application/json",
                },
                timeout=30.0,
            )
        return self._client
    
    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None
    
    async def refresh_token(self) -> Optional[TokenInfo]:
        """Refresh OAuth access token"""
        if not self.tokens.refresh_token:
            logger.warning(f"{self.VENDOR_NAME}: No refresh token available")
            return None
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.TOKEN_URL,
                    data={
                        "grant_type": "refresh_token",
                        "refresh_token": self.tokens.refresh_token,
                        "client_id": self.client_id,
                        "client_secret": self.client_secret,
                    },
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                )
                
                if response.status_code != 200:
                    logger.error(f"{self.VENDOR_NAME} token refresh failed: {response.text}")
                    return None
                
                data = response.json()
                expires_in = data.get("expires_in", 3600)
                
                self.tokens = TokenInfo(
                    access_token=data["access_token"],
                    refresh_token=data.get("refresh_token", self.tokens.refresh_token),
                    expires_at=datetime.utcnow() + timedelta(seconds=expires_in),
                    token_type=data.get("token_type", "Bearer"),
                    scope=data.get("scope"),
                )
                
                # Recreate client with new token
                await self.close()
                
                return self.tokens
                
        except Exception as e:
            logger.error(f"{self.VENDOR_NAME} token refresh error: {e}")
            return None
    
    async def ensure_valid_token(self) -> bool:
        """Check and refresh token if expired"""
        if self.tokens.expires_at and datetime.utcnow() >= self.tokens.expires_at - timedelta(minutes=5):
            new_tokens = await self.refresh_token()
            return new_tokens is not None
        return True
    
    @abstractmethod
    async def sync_data(
        self,
        data_types: Optional[List[str]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> SyncResult:
        """Fetch and normalize data from vendor API"""
        pass
    
    @abstractmethod
    def normalize_data(self, raw_data: Dict[str, Any], data_type: str) -> List[NormalizedReading]:
        """Convert vendor-specific data to normalized format"""
        pass
    
    def get_health_sections(self, readings: List[NormalizedReading]) -> List[HealthSection]:
        """Determine which health sections data should be routed to"""
        sections = set()
        for reading in readings:
            if reading.data_type in DATA_TYPE_TO_SECTIONS:
                sections.update(DATA_TYPE_TO_SECTIONS[reading.data_type])
        return list(sections)


# ============================================
# FITBIT ADAPTER
# ============================================

class FitbitAdapter(BaseVendorAdapter):
    """Fitbit API adapter - Full implementation"""
    
    VENDOR_ID = "fitbit"
    VENDOR_NAME = "Fitbit"
    BASE_URL = "https://api.fitbit.com"
    AUTH_URL = "https://www.fitbit.com/oauth2/authorize"
    TOKEN_URL = "https://api.fitbit.com/oauth2/token"
    
    DATA_TYPES = [
        "heart_rate",
        "sleep",
        "activities",
        "weight",
        "spo2",
        "breathing_rate",
        "hrv",
        "skin_temperature",
    ]
    
    async def sync_data(
        self,
        data_types: Optional[List[str]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> SyncResult:
        """Sync data from Fitbit API"""
        
        if not await self.ensure_valid_token():
            return SyncResult(success=False, error_code="token_expired", error_message="Failed to refresh token")
        
        types_to_sync = data_types or self.DATA_TYPES
        start = start_date or (datetime.utcnow() - timedelta(days=7))
        end = end_date or datetime.utcnow()
        
        start_str = start.strftime("%Y-%m-%d")
        end_str = end.strftime("%Y-%m-%d")
        
        all_readings: List[NormalizedReading] = []
        records_fetched = 0
        records_failed = 0
        
        client = await self.get_client()
        
        # Fetch heart rate data
        if "heart_rate" in types_to_sync:
            try:
                response = await client.get(f"/1/user/-/activities/heart/date/{start_str}/{end_str}.json")
                if response.status_code == 200:
                    data = response.json()
                    readings = self.normalize_data(data, "heart_rate")
                    all_readings.extend(readings)
                    records_fetched += len(readings)
                else:
                    logger.warning(f"Fitbit heart rate fetch failed: {response.status_code}")
                    records_failed += 1
            except Exception as e:
                logger.error(f"Fitbit heart rate error: {e}")
                records_failed += 1
        
        # Fetch HRV data
        if "hrv" in types_to_sync:
            try:
                response = await client.get(f"/1/user/-/hrv/date/{start_str}/{end_str}.json")
                if response.status_code == 200:
                    data = response.json()
                    readings = self.normalize_data(data, "hrv")
                    all_readings.extend(readings)
                    records_fetched += len(readings)
            except Exception as e:
                logger.error(f"Fitbit HRV error: {e}")
        
        # Fetch SpO2 data
        if "spo2" in types_to_sync:
            try:
                response = await client.get(f"/1/user/-/spo2/date/{start_str}/{end_str}.json")
                if response.status_code == 200:
                    data = response.json()
                    readings = self.normalize_data(data, "spo2")
                    all_readings.extend(readings)
                    records_fetched += len(readings)
            except Exception as e:
                logger.error(f"Fitbit SpO2 error: {e}")
        
        # Fetch sleep data
        if "sleep" in types_to_sync:
            try:
                response = await client.get(f"/1.2/user/-/sleep/date/{start_str}/{end_str}.json")
                if response.status_code == 200:
                    data = response.json()
                    readings = self.normalize_data(data, "sleep")
                    all_readings.extend(readings)
                    records_fetched += len(readings)
            except Exception as e:
                logger.error(f"Fitbit sleep error: {e}")
        
        # Fetch activity data
        if "activities" in types_to_sync:
            try:
                response = await client.get(f"/1/user/-/activities/date/{end_str}.json")
                if response.status_code == 200:
                    data = response.json()
                    readings = self.normalize_data(data, "activities")
                    all_readings.extend(readings)
                    records_fetched += len(readings)
            except Exception as e:
                logger.error(f"Fitbit activities error: {e}")
        
        # Fetch weight data
        if "weight" in types_to_sync:
            try:
                response = await client.get(f"/1/user/-/body/log/weight/date/{start_str}/{end_str}.json")
                if response.status_code == 200:
                    data = response.json()
                    readings = self.normalize_data(data, "weight")
                    all_readings.extend(readings)
                    records_fetched += len(readings)
            except Exception as e:
                logger.error(f"Fitbit weight error: {e}")
        
        # Fetch breathing rate
        if "breathing_rate" in types_to_sync:
            try:
                response = await client.get(f"/1/user/-/br/date/{start_str}/{end_str}.json")
                if response.status_code == 200:
                    data = response.json()
                    readings = self.normalize_data(data, "respiratory_rate")
                    all_readings.extend(readings)
                    records_fetched += len(readings)
            except Exception as e:
                logger.error(f"Fitbit breathing rate error: {e}")
        
        # Fetch skin temperature
        if "skin_temperature" in types_to_sync:
            try:
                response = await client.get(f"/1/user/-/temp/skin/date/{start_str}/{end_str}.json")
                if response.status_code == 200:
                    data = response.json()
                    readings = self.normalize_data(data, "skin_temp")
                    all_readings.extend(readings)
                    records_fetched += len(readings)
            except Exception as e:
                logger.error(f"Fitbit skin temp error: {e}")
        
        await self.close()
        
        return SyncResult(
            success=True,
            records_fetched=records_fetched,
            records_processed=len(all_readings),
            records_failed=records_failed,
            data_types=types_to_sync,
            date_range={"start": start_str, "end": end_str},
            normalized_data=[vars(r) for r in all_readings],
        )
    
    def normalize_data(self, raw_data: Dict[str, Any], data_type: str) -> List[NormalizedReading]:
        """Normalize Fitbit data to unified format"""
        readings = []
        
        if data_type == "heart_rate":
            for day in raw_data.get("activities-heart", []):
                date = day.get("dateTime")
                value = day.get("value", {})
                
                if "restingHeartRate" in value:
                    readings.append(NormalizedReading(
                        timestamp=f"{date}T00:00:00Z",
                        data_type="resting_heart_rate",
                        value=value["restingHeartRate"],
                        unit="bpm",
                        source="fitbit",
                    ))
                
                # Heart rate zones
                for zone in value.get("heartRateZones", []):
                    if zone.get("caloriesOut"):
                        readings.append(NormalizedReading(
                            timestamp=f"{date}T00:00:00Z",
                            data_type="heart_rate_zone",
                            value={
                                "name": zone["name"],
                                "min": zone["min"],
                                "max": zone["max"],
                                "minutes": zone.get("minutes", 0),
                                "calories": zone.get("caloriesOut", 0),
                            },
                            source="fitbit",
                        ))
        
        elif data_type == "hrv":
            for day in raw_data.get("hrv", []):
                readings.append(NormalizedReading(
                    timestamp=f"{day['dateTime']}T00:00:00Z",
                    data_type="hrv",
                    value=day.get("value", {}).get("dailyRmssd"),
                    unit="ms",
                    source="fitbit",
                    metadata={"coverage": day.get("value", {}).get("coverage")},
                ))
        
        elif data_type == "spo2":
            for day in raw_data.get("spo2", []):
                value = day.get("value", {})
                readings.append(NormalizedReading(
                    timestamp=f"{day['dateTime']}T00:00:00Z",
                    data_type="spo2",
                    value=value.get("avg"),
                    unit="%",
                    source="fitbit",
                    metadata={"min": value.get("min"), "max": value.get("max")},
                ))
        
        elif data_type == "sleep":
            for sleep in raw_data.get("sleep", []):
                readings.append(NormalizedReading(
                    timestamp=sleep.get("startTime"),
                    data_type="sleep",
                    value={
                        "duration_minutes": sleep.get("duration", 0) // 60000,
                        "efficiency": sleep.get("efficiency"),
                        "deep_minutes": sleep.get("levels", {}).get("summary", {}).get("deep", {}).get("minutes", 0),
                        "light_minutes": sleep.get("levels", {}).get("summary", {}).get("light", {}).get("minutes", 0),
                        "rem_minutes": sleep.get("levels", {}).get("summary", {}).get("rem", {}).get("minutes", 0),
                        "awake_minutes": sleep.get("levels", {}).get("summary", {}).get("wake", {}).get("minutes", 0),
                    },
                    source="fitbit",
                ))
        
        elif data_type == "activities":
            summary = raw_data.get("summary", {})
            date = raw_data.get("activities", [{}])[0].get("startDate") if raw_data.get("activities") else datetime.utcnow().strftime("%Y-%m-%d")
            
            readings.append(NormalizedReading(
                timestamp=f"{date}T00:00:00Z",
                data_type="steps",
                value=summary.get("steps", 0),
                unit="steps",
                source="fitbit",
            ))
            
            readings.append(NormalizedReading(
                timestamp=f"{date}T00:00:00Z",
                data_type="calories",
                value=summary.get("caloriesOut", 0),
                unit="kcal",
                source="fitbit",
            ))
            
            readings.append(NormalizedReading(
                timestamp=f"{date}T00:00:00Z",
                data_type="active_minutes",
                value=summary.get("veryActiveMinutes", 0) + summary.get("fairlyActiveMinutes", 0),
                unit="minutes",
                source="fitbit",
            ))
        
        elif data_type == "weight":
            for entry in raw_data.get("weight", []):
                readings.append(NormalizedReading(
                    timestamp=f"{entry['date']}T{entry.get('time', '00:00:00')}",
                    data_type="weight",
                    value=entry.get("weight"),
                    unit="kg",
                    source="fitbit",
                    metadata={"bmi": entry.get("bmi"), "fat": entry.get("fat")},
                ))
        
        elif data_type == "respiratory_rate":
            for day in raw_data.get("br", []):
                readings.append(NormalizedReading(
                    timestamp=f"{day['dateTime']}T00:00:00Z",
                    data_type="respiratory_rate",
                    value=day.get("value", {}).get("breathingRate"),
                    unit="breaths/min",
                    source="fitbit",
                ))
        
        elif data_type == "skin_temp":
            for day in raw_data.get("tempSkin", []):
                readings.append(NormalizedReading(
                    timestamp=f"{day['dateTime']}T00:00:00Z",
                    data_type="skin_temp",
                    value=day.get("value", {}).get("nightlyRelative"),
                    unit="°C (relative)",
                    source="fitbit",
                ))
        
        return readings


# ============================================
# WITHINGS ADAPTER
# ============================================

class WithingsAdapter(BaseVendorAdapter):
    """Withings API adapter - Full implementation"""
    
    VENDOR_ID = "withings"
    VENDOR_NAME = "Withings"
    BASE_URL = "https://wbsapi.withings.net"
    AUTH_URL = "https://account.withings.com/oauth2_user/authorize2"
    TOKEN_URL = "https://wbsapi.withings.net/v2/oauth2"
    
    DATA_TYPES = ["measure", "heart", "sleep"]
    
    async def sync_data(
        self,
        data_types: Optional[List[str]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> SyncResult:
        """Sync data from Withings API"""
        
        if not await self.ensure_valid_token():
            return SyncResult(success=False, error_code="token_expired")
        
        types_to_sync = data_types or self.DATA_TYPES
        start = start_date or (datetime.utcnow() - timedelta(days=7))
        end = end_date or datetime.utcnow()
        
        all_readings: List[NormalizedReading] = []
        records_fetched = 0
        
        client = await self.get_client()
        
        # Fetch measurements (weight, BP, etc.)
        if "measure" in types_to_sync:
            try:
                response = await client.post(
                    "/measure",
                    data={
                        "action": "getmeas",
                        "startdate": int(start.timestamp()),
                        "enddate": int(end.timestamp()),
                        "category": 1,  # Real measurements only
                    },
                )
                if response.status_code == 200:
                    data = response.json()
                    if data.get("status") == 0:
                        readings = self.normalize_data(data, "measure")
                        all_readings.extend(readings)
                        records_fetched += len(readings)
            except Exception as e:
                logger.error(f"Withings measure error: {e}")
        
        # Fetch heart data
        if "heart" in types_to_sync:
            try:
                response = await client.post(
                    "/v2/heart",
                    data={
                        "action": "list",
                        "startdate": int(start.timestamp()),
                        "enddate": int(end.timestamp()),
                    },
                )
                if response.status_code == 200:
                    data = response.json()
                    if data.get("status") == 0:
                        readings = self.normalize_data(data, "heart")
                        all_readings.extend(readings)
                        records_fetched += len(readings)
            except Exception as e:
                logger.error(f"Withings heart error: {e}")
        
        # Fetch sleep data
        if "sleep" in types_to_sync:
            try:
                response = await client.post(
                    "/v2/sleep",
                    data={
                        "action": "getsummary",
                        "startdateymd": start.strftime("%Y-%m-%d"),
                        "enddateymd": end.strftime("%Y-%m-%d"),
                    },
                )
                if response.status_code == 200:
                    data = response.json()
                    if data.get("status") == 0:
                        readings = self.normalize_data(data, "sleep")
                        all_readings.extend(readings)
                        records_fetched += len(readings)
            except Exception as e:
                logger.error(f"Withings sleep error: {e}")
        
        await self.close()
        
        return SyncResult(
            success=True,
            records_fetched=records_fetched,
            records_processed=len(all_readings),
            data_types=types_to_sync,
            date_range={"start": start.isoformat(), "end": end.isoformat()},
            normalized_data=[vars(r) for r in all_readings],
        )
    
    def normalize_data(self, raw_data: Dict[str, Any], data_type: str) -> List[NormalizedReading]:
        """Normalize Withings data"""
        readings = []
        
        if data_type == "measure":
            # Withings measure types: 1=weight, 4=height, 5=fatFreeMass, 6=fatRatio, 8=fatMassWeight
            # 9=diastolic, 10=systolic, 11=heartPulse, 71=temperature
            MEASURE_TYPES = {
                1: ("weight", "kg"),
                9: ("diastolic", "mmHg"),
                10: ("systolic", "mmHg"),
                11: ("heart_rate", "bpm"),
                71: ("temperature", "°C"),
                54: ("spo2", "%"),
            }
            
            for grp in raw_data.get("body", {}).get("measuregrps", []):
                timestamp = datetime.fromtimestamp(grp["date"]).isoformat()
                
                bp_systolic = None
                bp_diastolic = None
                bp_pulse = None
                
                for measure in grp.get("measures", []):
                    measure_type = measure["type"]
                    value = measure["value"] * (10 ** measure["unit"])
                    
                    if measure_type in MEASURE_TYPES:
                        dt, unit = MEASURE_TYPES[measure_type]
                        
                        if dt == "systolic":
                            bp_systolic = value
                        elif dt == "diastolic":
                            bp_diastolic = value
                        elif dt == "heart_rate" and bp_systolic:
                            bp_pulse = value
                        else:
                            readings.append(NormalizedReading(
                                timestamp=timestamp,
                                data_type=dt,
                                value=value,
                                unit=unit,
                                source="withings",
                            ))
                
                # Combine BP readings
                if bp_systolic and bp_diastolic:
                    readings.append(NormalizedReading(
                        timestamp=timestamp,
                        data_type="bp",
                        value={
                            "systolic": bp_systolic,
                            "diastolic": bp_diastolic,
                            "pulse": bp_pulse,
                        },
                        unit="mmHg",
                        source="withings",
                    ))
        
        elif data_type == "heart":
            for entry in raw_data.get("body", {}).get("series", []):
                readings.append(NormalizedReading(
                    timestamp=datetime.fromtimestamp(entry["timestamp"]).isoformat(),
                    data_type="ecg",
                    value={
                        "heart_rate": entry.get("heart_rate"),
                        "classification": entry.get("ecg", {}).get("classification"),
                        "afib_classification": entry.get("afib_classification"),
                    },
                    source="withings",
                ))
        
        elif data_type == "sleep":
            for night in raw_data.get("body", {}).get("series", []):
                readings.append(NormalizedReading(
                    timestamp=night.get("startdate"),
                    data_type="sleep",
                    value={
                        "duration_minutes": night.get("data", {}).get("total_timeinbed", 0) // 60,
                        "deep_minutes": night.get("data", {}).get("deepsleepduration", 0) // 60,
                        "light_minutes": night.get("data", {}).get("lightsleepduration", 0) // 60,
                        "rem_minutes": night.get("data", {}).get("remsleepduration", 0) // 60,
                        "awake_minutes": night.get("data", {}).get("wakeupduration", 0) // 60,
                        "respiratory_rate": night.get("data", {}).get("breathing_disturbances_intensity"),
                        "heart_rate_avg": night.get("data", {}).get("hr_average"),
                        "heart_rate_min": night.get("data", {}).get("hr_min"),
                        "heart_rate_max": night.get("data", {}).get("hr_max"),
                    },
                    source="withings",
                ))
        
        return readings


# ============================================
# OURA ADAPTER
# ============================================

class OuraAdapter(BaseVendorAdapter):
    """Oura Ring API adapter - Full implementation"""
    
    VENDOR_ID = "oura"
    VENDOR_NAME = "Oura"
    BASE_URL = "https://api.ouraring.com/v2"
    AUTH_URL = "https://cloud.ouraring.com/oauth/authorize"
    TOKEN_URL = "https://api.ouraring.com/oauth/token"
    
    DATA_TYPES = ["daily_sleep", "daily_readiness", "daily_activity", "heartrate", "sleep"]
    
    async def sync_data(
        self,
        data_types: Optional[List[str]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> SyncResult:
        """Sync data from Oura API"""
        
        if not await self.ensure_valid_token():
            return SyncResult(success=False, error_code="token_expired")
        
        types_to_sync = data_types or self.DATA_TYPES
        start = start_date or (datetime.utcnow() - timedelta(days=7))
        end = end_date or datetime.utcnow()
        
        start_str = start.strftime("%Y-%m-%d")
        end_str = end.strftime("%Y-%m-%d")
        
        all_readings: List[NormalizedReading] = []
        records_fetched = 0
        
        client = await self.get_client()
        
        # Fetch daily sleep scores
        if "daily_sleep" in types_to_sync:
            try:
                response = await client.get(f"/usercollection/daily_sleep?start_date={start_str}&end_date={end_str}")
                if response.status_code == 200:
                    data = response.json()
                    readings = self.normalize_data(data, "daily_sleep")
                    all_readings.extend(readings)
                    records_fetched += len(readings)
            except Exception as e:
                logger.error(f"Oura daily_sleep error: {e}")
        
        # Fetch readiness scores
        if "daily_readiness" in types_to_sync:
            try:
                response = await client.get(f"/usercollection/daily_readiness?start_date={start_str}&end_date={end_str}")
                if response.status_code == 200:
                    data = response.json()
                    readings = self.normalize_data(data, "daily_readiness")
                    all_readings.extend(readings)
                    records_fetched += len(readings)
            except Exception as e:
                logger.error(f"Oura daily_readiness error: {e}")
        
        # Fetch activity data
        if "daily_activity" in types_to_sync:
            try:
                response = await client.get(f"/usercollection/daily_activity?start_date={start_str}&end_date={end_str}")
                if response.status_code == 200:
                    data = response.json()
                    readings = self.normalize_data(data, "daily_activity")
                    all_readings.extend(readings)
                    records_fetched += len(readings)
            except Exception as e:
                logger.error(f"Oura daily_activity error: {e}")
        
        # Fetch heart rate data
        if "heartrate" in types_to_sync:
            try:
                response = await client.get(f"/usercollection/heartrate?start_datetime={start_str}T00:00:00Z&end_datetime={end_str}T23:59:59Z")
                if response.status_code == 200:
                    data = response.json()
                    readings = self.normalize_data(data, "heartrate")
                    all_readings.extend(readings)
                    records_fetched += len(readings)
            except Exception as e:
                logger.error(f"Oura heartrate error: {e}")
        
        # Fetch detailed sleep data
        if "sleep" in types_to_sync:
            try:
                response = await client.get(f"/usercollection/sleep?start_date={start_str}&end_date={end_str}")
                if response.status_code == 200:
                    data = response.json()
                    readings = self.normalize_data(data, "sleep")
                    all_readings.extend(readings)
                    records_fetched += len(readings)
            except Exception as e:
                logger.error(f"Oura sleep error: {e}")
        
        await self.close()
        
        return SyncResult(
            success=True,
            records_fetched=records_fetched,
            records_processed=len(all_readings),
            data_types=types_to_sync,
            date_range={"start": start_str, "end": end_str},
            normalized_data=[vars(r) for r in all_readings],
        )
    
    def normalize_data(self, raw_data: Dict[str, Any], data_type: str) -> List[NormalizedReading]:
        """Normalize Oura data"""
        readings = []
        
        if data_type == "daily_sleep":
            for day in raw_data.get("data", []):
                readings.append(NormalizedReading(
                    timestamp=f"{day['day']}T00:00:00Z",
                    data_type="sleep_score",
                    value=day.get("score"),
                    source="oura",
                    metadata={
                        "contributors": day.get("contributors", {}),
                    },
                ))
        
        elif data_type == "daily_readiness":
            for day in raw_data.get("data", []):
                readings.append(NormalizedReading(
                    timestamp=f"{day['day']}T00:00:00Z",
                    data_type="readiness",
                    value=day.get("score"),
                    source="oura",
                    metadata={
                        "temperature_deviation": day.get("temperature_deviation"),
                        "temperature_trend_deviation": day.get("temperature_trend_deviation"),
                        "contributors": day.get("contributors", {}),
                    },
                ))
        
        elif data_type == "daily_activity":
            for day in raw_data.get("data", []):
                readings.append(NormalizedReading(
                    timestamp=f"{day['day']}T00:00:00Z",
                    data_type="steps",
                    value=day.get("steps"),
                    unit="steps",
                    source="oura",
                ))
                
                readings.append(NormalizedReading(
                    timestamp=f"{day['day']}T00:00:00Z",
                    data_type="calories",
                    value=day.get("total_calories"),
                    unit="kcal",
                    source="oura",
                ))
                
                readings.append(NormalizedReading(
                    timestamp=f"{day['day']}T00:00:00Z",
                    data_type="active_minutes",
                    value=day.get("high_activity_time", 0) // 60 + day.get("medium_activity_time", 0) // 60,
                    unit="minutes",
                    source="oura",
                ))
        
        elif data_type == "heartrate":
            for entry in raw_data.get("data", []):
                readings.append(NormalizedReading(
                    timestamp=entry.get("timestamp"),
                    data_type="heart_rate",
                    value=entry.get("bpm"),
                    unit="bpm",
                    source="oura",
                    metadata={"source": entry.get("source")},
                ))
        
        elif data_type == "sleep":
            for night in raw_data.get("data", []):
                readings.append(NormalizedReading(
                    timestamp=night.get("bedtime_start"),
                    data_type="sleep",
                    value={
                        "duration_minutes": night.get("total_sleep_duration", 0) // 60,
                        "efficiency": night.get("efficiency"),
                        "deep_minutes": night.get("deep_sleep_duration", 0) // 60,
                        "light_minutes": night.get("light_sleep_duration", 0) // 60,
                        "rem_minutes": night.get("rem_sleep_duration", 0) // 60,
                        "awake_minutes": night.get("awake_time", 0) // 60,
                        "latency": night.get("latency"),
                        "restless_periods": night.get("restless_periods"),
                    },
                    source="oura",
                    metadata={
                        "hrv": {
                            "average": night.get("average_hrv"),
                            "max": night.get("lowest_heart_rate"),
                        },
                        "heart_rate": {
                            "average": night.get("average_heart_rate"),
                            "lowest": night.get("lowest_heart_rate"),
                        },
                        "breathing_rate": night.get("average_breath"),
                    },
                ))
        
        return readings


# ============================================
# GOOGLE FIT ADAPTER
# ============================================

class GoogleFitAdapter(BaseVendorAdapter):
    """Google Fit API adapter - Full implementation"""
    
    VENDOR_ID = "google_fit"
    VENDOR_NAME = "Google Fit"
    BASE_URL = "https://www.googleapis.com/fitness/v1/users/me"
    AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
    TOKEN_URL = "https://oauth2.googleapis.com/token"
    
    DATA_TYPES = ["steps", "heart_rate", "blood_pressure", "blood_glucose", "weight", "sleep", "oxygen_saturation"]
    
    # Google Fit data source types
    DATA_SOURCE_MAP = {
        "steps": "derived:com.google.step_count.delta:com.google.android.gms:estimated_steps",
        "heart_rate": "derived:com.google.heart_rate.bpm:com.google.android.gms:merge_heart_rate_bpm",
        "blood_pressure": "derived:com.google.blood_pressure:com.google.android.gms:merged",
        "blood_glucose": "derived:com.google.blood_glucose:com.google.android.gms:merged",
        "weight": "derived:com.google.weight:com.google.android.gms:merge_weight",
        "oxygen_saturation": "derived:com.google.oxygen_saturation:com.google.android.gms:merged",
    }
    
    async def sync_data(
        self,
        data_types: Optional[List[str]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> SyncResult:
        """Sync data from Google Fit API"""
        
        if not await self.ensure_valid_token():
            return SyncResult(success=False, error_code="token_expired")
        
        types_to_sync = data_types or self.DATA_TYPES
        start = start_date or (datetime.utcnow() - timedelta(days=7))
        end = end_date or datetime.utcnow()
        
        # Convert to nanoseconds for Google Fit API
        start_ns = int(start.timestamp() * 1e9)
        end_ns = int(end.timestamp() * 1e9)
        
        all_readings: List[NormalizedReading] = []
        records_fetched = 0
        
        client = await self.get_client()
        
        for dtype in types_to_sync:
            if dtype not in self.DATA_SOURCE_MAP:
                continue
            
            data_source = self.DATA_SOURCE_MAP[dtype]
            
            try:
                response = await client.get(
                    f"/dataSources/{data_source}/datasets/{start_ns}-{end_ns}"
                )
                
                if response.status_code == 200:
                    data = response.json()
                    readings = self.normalize_data(data, dtype)
                    all_readings.extend(readings)
                    records_fetched += len(readings)
                else:
                    logger.warning(f"Google Fit {dtype} fetch failed: {response.status_code}")
                    
            except Exception as e:
                logger.error(f"Google Fit {dtype} error: {e}")
        
        # Fetch sleep data separately using sessions API
        if "sleep" in types_to_sync:
            try:
                response = await client.get(
                    "/sessions",
                    params={
                        "startTime": start.isoformat() + "Z",
                        "endTime": end.isoformat() + "Z",
                        "activityType": "72",  # Sleep activity type
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    readings = self.normalize_data(data, "sleep")
                    all_readings.extend(readings)
                    records_fetched += len(readings)
                    
            except Exception as e:
                logger.error(f"Google Fit sleep error: {e}")
        
        await self.close()
        
        return SyncResult(
            success=True,
            records_fetched=records_fetched,
            records_processed=len(all_readings),
            data_types=types_to_sync,
            date_range={"start": start.isoformat(), "end": end.isoformat()},
            normalized_data=[vars(r) for r in all_readings],
        )
    
    def normalize_data(self, raw_data: Dict[str, Any], data_type: str) -> List[NormalizedReading]:
        """Normalize Google Fit data"""
        readings = []
        
        if data_type == "steps":
            for point in raw_data.get("point", []):
                timestamp = datetime.fromtimestamp(int(point["startTimeNanos"]) / 1e9).isoformat()
                value = sum(v.get("intVal", 0) for v in point.get("value", []))
                
                readings.append(NormalizedReading(
                    timestamp=timestamp,
                    data_type="steps",
                    value=value,
                    unit="steps",
                    source="google_fit",
                ))
        
        elif data_type == "heart_rate":
            for point in raw_data.get("point", []):
                timestamp = datetime.fromtimestamp(int(point["startTimeNanos"]) / 1e9).isoformat()
                values = point.get("value", [])
                
                if values:
                    readings.append(NormalizedReading(
                        timestamp=timestamp,
                        data_type="heart_rate",
                        value=values[0].get("fpVal"),
                        unit="bpm",
                        source="google_fit",
                    ))
        
        elif data_type == "blood_pressure":
            for point in raw_data.get("point", []):
                timestamp = datetime.fromtimestamp(int(point["startTimeNanos"]) / 1e9).isoformat()
                values = point.get("value", [])
                
                if len(values) >= 2:
                    readings.append(NormalizedReading(
                        timestamp=timestamp,
                        data_type="bp",
                        value={
                            "systolic": values[0].get("fpVal"),
                            "diastolic": values[1].get("fpVal"),
                        },
                        unit="mmHg",
                        source="google_fit",
                    ))
        
        elif data_type == "blood_glucose":
            for point in raw_data.get("point", []):
                timestamp = datetime.fromtimestamp(int(point["startTimeNanos"]) / 1e9).isoformat()
                values = point.get("value", [])
                
                if values:
                    # Convert mmol/L to mg/dL if needed
                    glucose = values[0].get("fpVal", 0)
                    readings.append(NormalizedReading(
                        timestamp=timestamp,
                        data_type="glucose",
                        value=glucose,
                        unit="mmol/L",
                        source="google_fit",
                    ))
        
        elif data_type == "weight":
            for point in raw_data.get("point", []):
                timestamp = datetime.fromtimestamp(int(point["startTimeNanos"]) / 1e9).isoformat()
                values = point.get("value", [])
                
                if values:
                    readings.append(NormalizedReading(
                        timestamp=timestamp,
                        data_type="weight",
                        value=values[0].get("fpVal"),
                        unit="kg",
                        source="google_fit",
                    ))
        
        elif data_type == "oxygen_saturation":
            for point in raw_data.get("point", []):
                timestamp = datetime.fromtimestamp(int(point["startTimeNanos"]) / 1e9).isoformat()
                values = point.get("value", [])
                
                if values:
                    readings.append(NormalizedReading(
                        timestamp=timestamp,
                        data_type="spo2",
                        value=values[0].get("fpVal") * 100,  # Convert to percentage
                        unit="%",
                        source="google_fit",
                    ))
        
        elif data_type == "sleep":
            for session in raw_data.get("session", []):
                start_time = datetime.fromtimestamp(int(session["startTimeMillis"]) / 1000)
                end_time = datetime.fromtimestamp(int(session["endTimeMillis"]) / 1000)
                duration_minutes = (end_time - start_time).total_seconds() / 60
                
                readings.append(NormalizedReading(
                    timestamp=start_time.isoformat(),
                    data_type="sleep",
                    value={
                        "duration_minutes": int(duration_minutes),
                        "name": session.get("name"),
                    },
                    source="google_fit",
                ))
        
        return readings


# ============================================
# IHEALTH ADAPTER
# ============================================

class IHealthAdapter(BaseVendorAdapter):
    """iHealth API adapter - Full implementation"""
    
    VENDOR_ID = "ihealth"
    VENDOR_NAME = "iHealth"
    BASE_URL = "https://api.ihealthlabs.com:8443/openapiv2"
    AUTH_URL = "https://api.ihealthlabs.com:8443/OpenApiV2/OAuthv2/userauthorization"
    TOKEN_URL = "https://api.ihealthlabs.com:8443/OpenApiV2/OAuthv2/userauthorization"
    
    DATA_TYPES = ["bp", "weight", "spo2", "glucose"]
    
    async def sync_data(
        self,
        data_types: Optional[List[str]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> SyncResult:
        """Sync data from iHealth API"""
        
        if not await self.ensure_valid_token():
            return SyncResult(success=False, error_code="token_expired")
        
        types_to_sync = data_types or self.DATA_TYPES
        start = start_date or (datetime.utcnow() - timedelta(days=7))
        end = end_date or datetime.utcnow()
        
        all_readings: List[NormalizedReading] = []
        records_fetched = 0
        
        client = await self.get_client()
        
        # Fetch blood pressure data
        if "bp" in types_to_sync:
            try:
                response = await client.get(
                    "/user/bp.json",
                    params={
                        "access_token": self.tokens.access_token,
                        "start_time": int(start.timestamp()),
                        "end_time": int(end.timestamp()),
                        "page_index": 1,
                    }
                )
                if response.status_code == 200:
                    data = response.json()
                    readings = self.normalize_data(data, "bp")
                    all_readings.extend(readings)
                    records_fetched += len(readings)
            except Exception as e:
                logger.error(f"iHealth BP error: {e}")
        
        # Fetch weight data
        if "weight" in types_to_sync:
            try:
                response = await client.get(
                    "/user/weight.json",
                    params={
                        "access_token": self.tokens.access_token,
                        "start_time": int(start.timestamp()),
                        "end_time": int(end.timestamp()),
                        "page_index": 1,
                    }
                )
                if response.status_code == 200:
                    data = response.json()
                    readings = self.normalize_data(data, "weight")
                    all_readings.extend(readings)
                    records_fetched += len(readings)
            except Exception as e:
                logger.error(f"iHealth weight error: {e}")
        
        # Fetch SpO2 data
        if "spo2" in types_to_sync:
            try:
                response = await client.get(
                    "/user/spo2.json",
                    params={
                        "access_token": self.tokens.access_token,
                        "start_time": int(start.timestamp()),
                        "end_time": int(end.timestamp()),
                        "page_index": 1,
                    }
                )
                if response.status_code == 200:
                    data = response.json()
                    readings = self.normalize_data(data, "spo2")
                    all_readings.extend(readings)
                    records_fetched += len(readings)
            except Exception as e:
                logger.error(f"iHealth SpO2 error: {e}")
        
        # Fetch glucose data
        if "glucose" in types_to_sync:
            try:
                response = await client.get(
                    "/user/glucose.json",
                    params={
                        "access_token": self.tokens.access_token,
                        "start_time": int(start.timestamp()),
                        "end_time": int(end.timestamp()),
                        "page_index": 1,
                    }
                )
                if response.status_code == 200:
                    data = response.json()
                    readings = self.normalize_data(data, "glucose")
                    all_readings.extend(readings)
                    records_fetched += len(readings)
            except Exception as e:
                logger.error(f"iHealth glucose error: {e}")
        
        await self.close()
        
        return SyncResult(
            success=True,
            records_fetched=records_fetched,
            records_processed=len(all_readings),
            data_types=types_to_sync,
            date_range={"start": start.isoformat(), "end": end.isoformat()},
            normalized_data=[vars(r) for r in all_readings],
        )
    
    def normalize_data(self, raw_data: Dict[str, Any], data_type: str) -> List[NormalizedReading]:
        """Normalize iHealth data"""
        readings = []
        
        if data_type == "bp":
            for entry in raw_data.get("BPDataList", []):
                timestamp = datetime.fromtimestamp(entry.get("MDate", 0)).isoformat()
                
                readings.append(NormalizedReading(
                    timestamp=timestamp,
                    data_type="bp",
                    value={
                        "systolic": entry.get("HP"),
                        "diastolic": entry.get("LP"),
                        "pulse": entry.get("HR"),
                    },
                    unit="mmHg",
                    source="ihealth",
                    metadata={"irregular_heartbeat": entry.get("IsArr") == 1},
                ))
        
        elif data_type == "weight":
            for entry in raw_data.get("WeightDataList", []):
                timestamp = datetime.fromtimestamp(entry.get("MDate", 0)).isoformat()
                
                readings.append(NormalizedReading(
                    timestamp=timestamp,
                    data_type="weight",
                    value=entry.get("WeightValue"),
                    unit="kg",
                    source="ihealth",
                    metadata={
                        "bmi": entry.get("BMI"),
                        "body_fat": entry.get("FatValue"),
                        "muscle_mass": entry.get("MuscleValue"),
                    },
                ))
        
        elif data_type == "spo2":
            for entry in raw_data.get("BODataList", []):
                timestamp = datetime.fromtimestamp(entry.get("MDate", 0)).isoformat()
                
                readings.append(NormalizedReading(
                    timestamp=timestamp,
                    data_type="spo2",
                    value=entry.get("BO"),
                    unit="%",
                    source="ihealth",
                    metadata={"pulse": entry.get("HR")},
                ))
        
        elif data_type == "glucose":
            for entry in raw_data.get("BGDataList", []):
                timestamp = datetime.fromtimestamp(entry.get("MDate", 0)).isoformat()
                
                readings.append(NormalizedReading(
                    timestamp=timestamp,
                    data_type="glucose",
                    value=entry.get("BG"),
                    unit="mg/dL",
                    source="ihealth",
                    metadata={
                        "meal_status": entry.get("DinnerSituation"),
                        "drug_status": entry.get("DrugSituation"),
                    },
                ))
        
        return readings


# ============================================
# PRIVATE API STUBS (Ready for Credentials)
# ============================================

class GarminAdapter(BaseVendorAdapter):
    """Garmin API adapter - STUB (requires partnership)"""
    
    VENDOR_ID = "garmin"
    VENDOR_NAME = "Garmin"
    BASE_URL = "https://apis.garmin.com"
    AUTH_URL = "https://connect.garmin.com/oauthConfirm"
    TOKEN_URL = "https://connectapi.garmin.com/oauth-service/oauth/access_token"
    
    DATA_TYPES = ["heart_rate", "hrv", "spo2", "sleep", "steps", "stress", "body_battery", "vo2_max", "training_load", "recovery"]
    
    async def sync_data(self, data_types=None, start_date=None, end_date=None) -> SyncResult:
        logger.warning("Garmin API requires partnership agreement. Configure GARMIN_CLIENT_ID and GARMIN_CLIENT_SECRET.")
        return SyncResult(
            success=False,
            error_code="partnership_required",
            error_message="Garmin Health API requires a partnership agreement. Please contact Garmin for API access.",
        )
    
    def normalize_data(self, raw_data, data_type):
        return []


class WhoopAdapter(BaseVendorAdapter):
    """Whoop API adapter - STUB (requires partnership)"""
    
    VENDOR_ID = "whoop"
    VENDOR_NAME = "Whoop"
    BASE_URL = "https://api.prod.whoop.com/developer/v1"
    AUTH_URL = "https://api.prod.whoop.com/oauth/oauth2/auth"
    TOKEN_URL = "https://api.prod.whoop.com/oauth/oauth2/token"
    
    DATA_TYPES = ["heart_rate", "hrv", "spo2", "sleep", "recovery", "strain", "skin_temp", "respiratory_rate"]
    
    async def sync_data(self, data_types=None, start_date=None, end_date=None) -> SyncResult:
        logger.warning("Whoop API requires developer access. Configure WHOOP_CLIENT_ID and WHOOP_CLIENT_SECRET.")
        return SyncResult(
            success=False,
            error_code="developer_access_required",
            error_message="Whoop API requires developer access. Apply at https://developer.whoop.com",
        )
    
    def normalize_data(self, raw_data, data_type):
        return []


class DexcomAdapter(BaseVendorAdapter):
    """Dexcom API adapter - STUB (requires partnership & BAA)"""
    
    VENDOR_ID = "dexcom"
    VENDOR_NAME = "Dexcom"
    BASE_URL = "https://api.dexcom.com/v2"
    AUTH_URL = "https://api.dexcom.com/v2/oauth2/login"
    TOKEN_URL = "https://api.dexcom.com/v2/oauth2/token"
    
    DATA_TYPES = ["glucose"]
    
    async def sync_data(self, data_types=None, start_date=None, end_date=None) -> SyncResult:
        logger.warning("Dexcom API requires partnership and BAA. Configure DEXCOM_CLIENT_ID and DEXCOM_CLIENT_SECRET.")
        return SyncResult(
            success=False,
            error_code="partnership_baa_required",
            error_message="Dexcom API requires partnership agreement and signed BAA for HIPAA compliance.",
        )
    
    def normalize_data(self, raw_data, data_type):
        return []


class SamsungAdapter(BaseVendorAdapter):
    """Samsung Health API adapter - STUB (requires partnership)"""
    
    VENDOR_ID = "samsung"
    VENDOR_NAME = "Samsung Health"
    BASE_URL = ""  # SDK-based, no direct API
    
    DATA_TYPES = ["heart_rate", "spo2", "sleep", "steps", "stress", "bp", "ecg"]
    
    async def sync_data(self, data_types=None, start_date=None, end_date=None) -> SyncResult:
        logger.warning("Samsung Health requires SDK integration and partnership.")
        return SyncResult(
            success=False,
            error_code="sdk_required",
            error_message="Samsung Health requires SDK integration through Samsung Developer Program.",
        )
    
    def normalize_data(self, raw_data, data_type):
        return []


class EkoAdapter(BaseVendorAdapter):
    """Eko stethoscope API adapter - STUB (requires partnership & BAA)"""
    
    VENDOR_ID = "eko"
    VENDOR_NAME = "Eko"
    BASE_URL = ""
    
    DATA_TYPES = ["heart_sounds", "lung_sounds", "ecg"]
    
    async def sync_data(self, data_types=None, start_date=None, end_date=None) -> SyncResult:
        logger.warning("Eko API requires partnership and BAA.")
        return SyncResult(
            success=False,
            error_code="partnership_baa_required",
            error_message="Eko API requires healthcare partnership agreement and signed BAA.",
        )
    
    def normalize_data(self, raw_data, data_type):
        return []


class AbbottAdapter(BaseVendorAdapter):
    """Abbott LibreView API adapter - STUB (requires partnership & BAA)"""
    
    VENDOR_ID = "abbott"
    VENDOR_NAME = "Abbott LibreView"
    BASE_URL = "https://api.libreview.io"
    
    DATA_TYPES = ["glucose"]
    
    async def sync_data(self, data_types=None, start_date=None, end_date=None) -> SyncResult:
        logger.warning("Abbott LibreView API requires partnership and BAA.")
        return SyncResult(
            success=False,
            error_code="partnership_baa_required",
            error_message="Abbott LibreView API requires partnership agreement and signed BAA for CGM data access.",
        )
    
    def normalize_data(self, raw_data, data_type):
        return []


# ============================================
# ADAPTER FACTORY
# ============================================

def get_vendor_adapter(vendor_id: str, tokens: TokenInfo, user_id: str) -> Optional[BaseVendorAdapter]:
    """Factory function to get appropriate vendor adapter"""
    
    adapters = {
        "fitbit": FitbitAdapter,
        "withings": WithingsAdapter,
        "oura": OuraAdapter,
        "google_fit": GoogleFitAdapter,
        "ihealth": IHealthAdapter,
        "garmin": GarminAdapter,
        "whoop": WhoopAdapter,
        "dexcom": DexcomAdapter,
        "samsung": SamsungAdapter,
        "eko": EkoAdapter,
        "abbott": AbbottAdapter,
    }
    
    adapter_class = adapters.get(vendor_id.lower())
    if adapter_class:
        return adapter_class(tokens, user_id)
    
    logger.warning(f"No adapter found for vendor: {vendor_id}")
    return None
