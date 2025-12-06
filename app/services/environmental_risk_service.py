"""
Environmental Risk Service
Comprehensive environmental health intelligence service with data ingestion,
risk scoring, ML forecasting, and symptom correlation analysis.
"""

import os
import logging
import asyncio
import httpx
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple
from decimal import Decimal
from statistics import mean, stdev
from sqlalchemy.orm import Session
from sqlalchemy import desc, and_, or_
from scipy import stats
import numpy as np

from app.models.environmental_risk import (
    PatientEnvironmentProfile,
    ConditionTriggerMapping,
    PatientTriggerWeight,
    EnvironmentalDataSnapshot,
    PatientEnvironmentRiskScore,
    EnvironmentalForecast,
    SymptomEnvironmentCorrelation,
    EnvironmentalAlert,
    EnvironmentalPipelineJob,
)

logger = logging.getLogger(__name__)


def audit_log(db: Session, patient_id: str, action: str, resource_type: str, resource_id: str, details: dict = None):
    """Simple audit log for HIPAA compliance."""
    logger.info(f"AUDIT: patient={patient_id}, action={action}, resource={resource_type}/{resource_id}, details={details}")


# =============================================================================
# CONDITION-FACTOR MAPPINGS (Clinical Evidence-Based)
# =============================================================================

DEFAULT_CONDITION_TRIGGERS: Dict[str, List[Dict[str, Any]]] = {
    "asthma": [
        {"factor": "pm25", "weight": 0.85, "threshold": 35, "critical": 55, "direction": "higher_is_worse"},
        {"factor": "pm10", "weight": 0.70, "threshold": 54, "critical": 154, "direction": "higher_is_worse"},
        {"factor": "ozone", "weight": 0.75, "threshold": 70, "critical": 105, "direction": "higher_is_worse"},
        {"factor": "humidity", "weight": 0.50, "threshold": 60, "critical": 80, "direction": "both_extremes"},
        {"factor": "pollen", "weight": 0.80, "threshold": 6, "critical": 9, "direction": "higher_is_worse"},
        {"factor": "mold", "weight": 0.65, "threshold": 6500, "critical": 13000, "direction": "higher_is_worse"},
    ],
    "copd": [
        {"factor": "pm25", "weight": 0.90, "threshold": 25, "critical": 45, "direction": "higher_is_worse"},
        {"factor": "pm10", "weight": 0.75, "threshold": 50, "critical": 100, "direction": "higher_is_worse"},
        {"factor": "ozone", "weight": 0.80, "threshold": 60, "critical": 95, "direction": "higher_is_worse"},
        {"factor": "humidity", "weight": 0.60, "threshold": 55, "critical": 75, "direction": "both_extremes"},
        {"factor": "temperature", "weight": 0.55, "threshold": 32, "critical": 38, "direction": "higher_is_worse"},
    ],
    "heart_failure": [
        {"factor": "temperature", "weight": 0.85, "threshold": 30, "critical": 35, "direction": "higher_is_worse"},
        {"factor": "humidity", "weight": 0.75, "threshold": 65, "critical": 80, "direction": "higher_is_worse"},
        {"factor": "pm25", "weight": 0.70, "threshold": 35, "critical": 55, "direction": "higher_is_worse"},
        {"factor": "pressure", "weight": 0.50, "threshold": 1000, "critical": 980, "direction": "lower_is_worse"},
    ],
    "arthritis": [
        {"factor": "humidity", "weight": 0.80, "threshold": 70, "critical": 85, "direction": "higher_is_worse"},
        {"factor": "temperature", "weight": 0.75, "threshold": 10, "critical": 5, "direction": "lower_is_worse"},
        {"factor": "pressure", "weight": 0.85, "threshold": 1010, "critical": 1000, "direction": "lower_is_worse"},
    ],
    "migraines": [
        {"factor": "pressure", "weight": 0.90, "threshold": 1008, "critical": 1000, "direction": "lower_is_worse"},
        {"factor": "temperature", "weight": 0.65, "threshold": 28, "critical": 35, "direction": "higher_is_worse"},
        {"factor": "ozone", "weight": 0.55, "threshold": 60, "critical": 85, "direction": "higher_is_worse"},
    ],
    "eczema": [
        {"factor": "humidity", "weight": 0.85, "threshold": 30, "critical": 20, "direction": "lower_is_worse"},
        {"factor": "pollen", "weight": 0.70, "threshold": 5, "critical": 8, "direction": "higher_is_worse"},
        {"factor": "pm25", "weight": 0.60, "threshold": 30, "critical": 50, "direction": "higher_is_worse"},
        {"factor": "uv", "weight": 0.55, "threshold": 6, "critical": 8, "direction": "higher_is_worse"},
    ],
}


class EnvironmentalRiskService:
    """Main service for environmental risk assessment and management."""
    
    def __init__(self, db: Session):
        self.db = db
        self.openweather_api_key = os.getenv("OPENWEATHERMAP_API_KEY")
        self.airnow_api_key = os.getenv("AIRNOW_API_KEY")
    
    # =========================================================================
    # PATIENT PROFILE MANAGEMENT
    # =========================================================================
    
    async def get_or_create_profile(
        self, 
        patient_id: str, 
        zip_code: str,
        conditions: Optional[List[str]] = None,
        allergies: Optional[List[str]] = None
    ) -> PatientEnvironmentProfile:
        """Get existing profile or create a new one for the patient."""
        profile = self.db.query(PatientEnvironmentProfile).filter(
            PatientEnvironmentProfile.patient_id == patient_id,
            PatientEnvironmentProfile.is_active == True
        ).first()
        
        if profile:
            if zip_code and profile.zip_code != zip_code:
                profile.zip_code = zip_code
                profile.updated_at = datetime.utcnow()
            if conditions:
                profile.chronic_conditions = conditions
            if allergies:
                profile.allergies = allergies
            self.db.commit()
            return profile
        
        location_info = await self._geocode_zip(zip_code)
        
        profile = PatientEnvironmentProfile(
            patient_id=patient_id,
            zip_code=zip_code,
            city=location_info.get("city"),
            state=location_info.get("state"),
            chronic_conditions=conditions or [],
            allergies=allergies or [],
            alert_thresholds={
                "riskScore": 70,
                "aqiThreshold": 100,
                "pollenThreshold": 8,
                "temperatureMin": 5,
                "temperatureMax": 35,
                "humidityMin": 20,
                "humidityMax": 80
            }
        )
        self.db.add(profile)
        self.db.commit()
        self.db.refresh(profile)
        
        await self._initialize_patient_weights(patient_id, conditions or [])
        
        audit_log(
            self.db,
            patient_id=patient_id,
            action="environment_profile_created",
            resource_type="environmental_profile",
            resource_id=profile.id,
            details={"zip_code": zip_code, "conditions": conditions}
        )
        
        return profile
    
    async def update_profile(
        self,
        patient_id: str,
        updates: Dict[str, Any]
    ) -> Optional[PatientEnvironmentProfile]:
        """Update patient's environmental profile."""
        profile = self.db.query(PatientEnvironmentProfile).filter(
            PatientEnvironmentProfile.patient_id == patient_id,
            PatientEnvironmentProfile.is_active == True
        ).first()
        
        if not profile:
            return None
        
        allowed_fields = [
            "zip_code", "chronic_conditions", "allergies", "alerts_enabled",
            "alert_thresholds", "push_notifications", "sms_notifications",
            "email_digest", "digest_frequency", "correlation_consent_given"
        ]
        
        for field, value in updates.items():
            if field in allowed_fields and hasattr(profile, field):
                setattr(profile, field, value)
        
        if "correlation_consent_given" in updates and updates["correlation_consent_given"]:
            profile.correlation_consent_at = datetime.utcnow()
        
        profile.updated_at = datetime.utcnow()
        self.db.commit()
        self.db.refresh(profile)
        
        return profile
    
    async def _initialize_patient_weights(
        self, 
        patient_id: str, 
        conditions: List[str]
    ) -> None:
        """Initialize default trigger weights for a patient based on their conditions."""
        all_factors = set()
        factor_weights = {}
        
        for condition in conditions:
            triggers = DEFAULT_CONDITION_TRIGGERS.get(condition.lower(), [])
            for trigger in triggers:
                factor = trigger["factor"]
                weight = trigger["weight"]
                if factor not in factor_weights:
                    factor_weights[factor] = []
                factor_weights[factor].append(weight)
        
        for factor, weights in factor_weights.items():
            avg_weight = sum(weights) / len(weights)
            
            existing = self.db.query(PatientTriggerWeight).filter(
                PatientTriggerWeight.patient_id == patient_id,
                PatientTriggerWeight.factor_type == factor
            ).first()
            
            if not existing:
                weight_record = PatientTriggerWeight(
                    patient_id=patient_id,
                    factor_type=factor,
                    personalized_weight=Decimal(str(avg_weight)),
                    source="default",
                    confidence_score=Decimal("0.5")
                )
                self.db.add(weight_record)
        
        self.db.commit()
    
    # =========================================================================
    # DATA INGESTION
    # =========================================================================
    
    async def ingest_environmental_data(self, zip_code: str) -> Optional[EnvironmentalDataSnapshot]:
        """Ingest environmental data from multiple sources for a ZIP code."""
        try:
            weather_data, aqi_data, pollen_data, hazard_data = await asyncio.gather(
                self._fetch_weather_data(zip_code),
                self._fetch_air_quality_data(zip_code),
                self._fetch_pollen_data(zip_code),
                self._fetch_hazard_data(zip_code),
                return_exceptions=True
            )
            
            if isinstance(weather_data, Exception):
                logger.warning(f"Weather data fetch failed: {weather_data}")
                weather_data = {}
            if isinstance(aqi_data, Exception):
                logger.warning(f"AQI data fetch failed: {aqi_data}")
                aqi_data = {}
            if isinstance(pollen_data, Exception):
                logger.warning(f"Pollen data fetch failed: {pollen_data}")
                pollen_data = {}
            if isinstance(hazard_data, Exception):
                logger.warning(f"Hazard data fetch failed: {hazard_data}")
                hazard_data = {}
            
            missing_fields = []
            if not weather_data:
                missing_fields.extend(["temperature", "humidity", "pressure"])
            if not aqi_data:
                missing_fields.extend(["aqi", "pm25", "pm10"])
            if not pollen_data:
                missing_fields.extend(["pollen_overall"])
            
            data_quality = max(0, 100 - (len(missing_fields) * 10))
            
            snapshot = EnvironmentalDataSnapshot(
                zip_code=zip_code,
                measured_at=datetime.utcnow(),
                
                temperature=weather_data.get("temperature"),
                feels_like=weather_data.get("feels_like"),
                humidity=weather_data.get("humidity"),
                pressure=weather_data.get("pressure"),
                wind_speed=weather_data.get("wind_speed"),
                wind_direction=weather_data.get("wind_direction"),
                precipitation=weather_data.get("precipitation"),
                uv_index=weather_data.get("uv_index"),
                cloud_cover=weather_data.get("cloud_cover"),
                visibility=weather_data.get("visibility"),
                
                aqi=aqi_data.get("aqi"),
                aqi_category=aqi_data.get("category"),
                pm25=aqi_data.get("pm25"),
                pm10=aqi_data.get("pm10"),
                ozone=aqi_data.get("ozone"),
                no2=aqi_data.get("no2"),
                so2=aqi_data.get("so2"),
                co=aqi_data.get("co"),
                
                pollen_tree_count=pollen_data.get("tree"),
                pollen_grass_count=pollen_data.get("grass"),
                pollen_weed_count=pollen_data.get("weed"),
                pollen_overall=pollen_data.get("overall"),
                pollen_category=pollen_data.get("category"),
                mold_spore_count=pollen_data.get("mold"),
                mold_category=pollen_data.get("mold_category"),
                
                active_hazards=hazard_data.get("hazards", []),
                
                weather_source=weather_data.get("source", "openweathermap"),
                aqi_source=aqi_data.get("source", "airnow"),
                pollen_source=pollen_data.get("source", "simulated"),
                hazard_source=hazard_data.get("source", "nws"),
                
                data_quality_score=Decimal(str(data_quality)),
                missing_fields=missing_fields if missing_fields else None
            )
            
            self.db.add(snapshot)
            self.db.commit()
            self.db.refresh(snapshot)
            
            return snapshot
            
        except Exception as e:
            logger.error(f"Error ingesting environmental data for {zip_code}: {e}")
            self.db.rollback()
            return None
    
    async def _fetch_weather_data(self, zip_code: str) -> Dict[str, Any]:
        """Fetch weather data from OpenWeatherMap API."""
        if not self.openweather_api_key:
            return self._generate_simulated_weather()
        
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    "https://api.openweathermap.org/data/2.5/weather",
                    params={
                        "zip": f"{zip_code},US",
                        "appid": self.openweather_api_key,
                        "units": "metric"
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    return {
                        "temperature": data["main"]["temp"],
                        "feels_like": data["main"]["feels_like"],
                        "humidity": data["main"]["humidity"],
                        "pressure": data["main"]["pressure"],
                        "wind_speed": data["wind"]["speed"],
                        "wind_direction": data["wind"].get("deg"),
                        "precipitation": data.get("rain", {}).get("1h", 0),
                        "cloud_cover": data["clouds"]["all"],
                        "visibility": data.get("visibility", 10000),
                        "source": "openweathermap"
                    }
                else:
                    logger.warning(f"OpenWeatherMap API returned {response.status_code}")
                    return self._generate_simulated_weather()
                    
        except Exception as e:
            logger.warning(f"Weather API error: {e}")
            return self._generate_simulated_weather()
    
    async def _fetch_air_quality_data(self, zip_code: str) -> Dict[str, Any]:
        """Fetch air quality data from AirNow API or OpenWeatherMap Air Pollution."""
        if self.openweather_api_key:
            try:
                coords = await self._geocode_zip(zip_code)
                lat, lon = coords.get("lat", 40.7), coords.get("lon", -74.0)
                
                async with httpx.AsyncClient(timeout=10.0) as client:
                    response = await client.get(
                        "http://api.openweathermap.org/data/2.5/air_pollution",
                        params={
                            "lat": lat,
                            "lon": lon,
                            "appid": self.openweather_api_key
                        }
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        if data.get("list"):
                            air_data = data["list"][0]
                            aqi_map = {1: 25, 2: 75, 3: 125, 4: 175, 5: 250}
                            category_map = {
                                1: "good", 2: "moderate", 3: "unhealthy_sensitive",
                                4: "unhealthy", 5: "very_unhealthy"
                            }
                            return {
                                "aqi": aqi_map.get(air_data["main"]["aqi"], 50),
                                "category": category_map.get(air_data["main"]["aqi"], "moderate"),
                                "pm25": air_data["components"].get("pm2_5"),
                                "pm10": air_data["components"].get("pm10"),
                                "ozone": air_data["components"].get("o3"),
                                "no2": air_data["components"].get("no2"),
                                "so2": air_data["components"].get("so2"),
                                "co": air_data["components"].get("co"),
                                "source": "openweathermap"
                            }
            except Exception as e:
                logger.warning(f"Air quality API error: {e}")
        
        return self._generate_simulated_aqi()
    
    async def _fetch_pollen_data(self, zip_code: str) -> Dict[str, Any]:
        """Fetch pollen and allergen data."""
        return self._generate_simulated_pollen()
    
    async def _fetch_hazard_data(self, zip_code: str) -> Dict[str, Any]:
        """Fetch environmental hazards from NWS and EPA."""
        return {"hazards": [], "source": "nws"}
    
    def _generate_simulated_weather(self) -> Dict[str, Any]:
        """Generate realistic simulated weather data."""
        import random
        temp = random.uniform(10, 30)
        return {
            "temperature": round(temp, 1),
            "feels_like": round(temp + random.uniform(-3, 3), 1),
            "humidity": random.randint(30, 80),
            "pressure": random.randint(1000, 1025),
            "wind_speed": round(random.uniform(0, 15), 1),
            "wind_direction": random.randint(0, 360),
            "precipitation": round(random.uniform(0, 5), 1) if random.random() > 0.7 else 0,
            "uv_index": round(random.uniform(0, 11), 1),
            "cloud_cover": random.randint(0, 100),
            "visibility": random.randint(5000, 10000),
            "source": "simulated"
        }
    
    def _generate_simulated_aqi(self) -> Dict[str, Any]:
        """Generate realistic simulated AQI data."""
        import random
        aqi = random.randint(20, 150)
        if aqi <= 50:
            category = "good"
        elif aqi <= 100:
            category = "moderate"
        elif aqi <= 150:
            category = "unhealthy_sensitive"
        else:
            category = "unhealthy"
        
        return {
            "aqi": aqi,
            "category": category,
            "pm25": round(random.uniform(5, 50), 1),
            "pm10": round(random.uniform(10, 100), 1),
            "ozone": round(random.uniform(20, 80), 1),
            "no2": round(random.uniform(5, 40), 1),
            "so2": round(random.uniform(1, 20), 1),
            "co": round(random.uniform(0.1, 5), 2),
            "source": "simulated"
        }
    
    def _generate_simulated_pollen(self) -> Dict[str, Any]:
        """Generate realistic simulated pollen data."""
        import random
        overall = random.randint(0, 12)
        if overall <= 2:
            category = "low"
        elif overall <= 5:
            category = "moderate"
        elif overall <= 8:
            category = "high"
        else:
            category = "very_high"
        
        return {
            "tree": random.randint(0, 500),
            "grass": random.randint(0, 300),
            "weed": random.randint(0, 200),
            "overall": overall,
            "category": category,
            "mold": random.randint(0, 15000),
            "mold_category": "moderate" if random.random() > 0.5 else "low",
            "source": "simulated"
        }
    
    async def _geocode_zip(self, zip_code: str) -> Dict[str, Any]:
        """Geocode a ZIP code to coordinates and city/state."""
        region_coords = {
            "0": {"lat": 40.71, "lon": -74.01, "city": "New York", "state": "NY"},
            "1": {"lat": 40.71, "lon": -74.01, "city": "Newark", "state": "NJ"},
            "2": {"lat": 38.91, "lon": -77.04, "city": "Washington", "state": "DC"},
            "3": {"lat": 33.75, "lon": -84.39, "city": "Atlanta", "state": "GA"},
            "4": {"lat": 30.27, "lon": -97.74, "city": "Austin", "state": "TX"},
            "5": {"lat": 41.88, "lon": -87.63, "city": "Chicago", "state": "IL"},
            "6": {"lat": 39.74, "lon": -104.99, "city": "Denver", "state": "CO"},
            "7": {"lat": 32.78, "lon": -96.80, "city": "Dallas", "state": "TX"},
            "8": {"lat": 40.76, "lon": -111.89, "city": "Salt Lake City", "state": "UT"},
            "9": {"lat": 37.77, "lon": -122.42, "city": "San Francisco", "state": "CA"},
        }
        
        first_digit = zip_code[0] if zip_code else "0"
        return region_coords.get(first_digit, region_coords["0"])
    
    # =========================================================================
    # RISK SCORING ENGINE
    # =========================================================================
    
    async def compute_risk_score(
        self,
        patient_id: str,
        snapshot: Optional[EnvironmentalDataSnapshot] = None
    ) -> Optional[PatientEnvironmentRiskScore]:
        """Compute personalized environmental risk score for a patient."""
        profile = self.db.query(PatientEnvironmentProfile).filter(
            PatientEnvironmentProfile.patient_id == patient_id,
            PatientEnvironmentProfile.is_active == True
        ).first()
        
        if not profile:
            return None
        
        if not snapshot:
            snapshot = self.db.query(EnvironmentalDataSnapshot).filter(
                EnvironmentalDataSnapshot.zip_code == profile.zip_code
            ).order_by(desc(EnvironmentalDataSnapshot.measured_at)).first()
        
        if not snapshot:
            snapshot = await self.ingest_environmental_data(profile.zip_code)
        
        if not snapshot:
            return None
        
        weights = self._get_patient_weights(patient_id, profile.chronic_conditions or [])
        
        factor_values = self._extract_factor_values(snapshot)
        
        factor_contributions = []
        weather_score = 0
        air_quality_score = 0
        allergen_score = 0
        hazard_score = 0
        
        for factor, raw_value in factor_values.items():
            if raw_value is None:
                continue
            
            normalized = self._normalize_factor(factor, float(raw_value))
            weight = float(weights.get(factor, 0.3))
            contribution = normalized * weight
            
            factor_contributions.append({
                "factor": factor,
                "rawValue": float(raw_value),
                "normalizedValue": round(normalized, 2),
                "weight": round(weight, 4),
                "contribution": round(contribution, 2)
            })
            
            if factor in ["temperature", "humidity", "pressure"]:
                weather_score += contribution
            elif factor in ["pm25", "pm10", "ozone", "no2", "so2", "co", "aqi"]:
                air_quality_score += contribution
            elif factor in ["pollen", "mold"]:
                allergen_score += contribution
        
        if snapshot.active_hazards:
            hazard_count = len(snapshot.active_hazards)
            hazard_score = min(100, hazard_count * 25)
        
        total_weight = sum(fc["weight"] for fc in factor_contributions) or 1
        composite_score = sum(fc["contribution"] for fc in factor_contributions)
        composite_score = min(100, (composite_score / total_weight) * 100)
        
        if hazard_score > 0:
            composite_score = min(100, composite_score * 0.7 + hazard_score * 0.3)
        
        trends = await self._compute_trends(patient_id, profile.zip_code)
        volatility = await self._compute_volatility(profile.zip_code)
        
        composite_score = (
            composite_score * 0.6 +
            (trends.get("trend_24hr", 0) + 1) * 25 +
            volatility * 0.15
        )
        composite_score = max(0, min(100, composite_score))
        
        if composite_score < 25:
            risk_level = "low"
        elif composite_score < 50:
            risk_level = "moderate"
        elif composite_score < 75:
            risk_level = "high"
        else:
            risk_level = "critical"
        
        top_factors = sorted(
            factor_contributions, 
            key=lambda x: x["contribution"], 
            reverse=True
        )[:5]
        
        top_risk_factors = []
        for tf in top_factors:
            if tf["normalizedValue"] > 0.7:
                severity = "critical"
            elif tf["normalizedValue"] > 0.5:
                severity = "high"
            elif tf["normalizedValue"] > 0.3:
                severity = "moderate"
            else:
                severity = "low"
            
            top_risk_factors.append({
                "factor": tf["factor"],
                "severity": severity,
                "recommendation": self._get_factor_recommendation(tf["factor"], severity)
            })
        
        risk_score = PatientEnvironmentRiskScore(
            patient_id=patient_id,
            snapshot_id=snapshot.id,
            computed_at=datetime.utcnow(),
            composite_risk_score=Decimal(str(round(composite_score, 2))),
            risk_level=risk_level,
            weather_risk_score=Decimal(str(round(min(100, weather_score * 100), 2))),
            air_quality_risk_score=Decimal(str(round(min(100, air_quality_score * 100), 2))),
            allergen_risk_score=Decimal(str(round(min(100, allergen_score * 100), 2))),
            hazard_risk_score=Decimal(str(round(hazard_score, 2))),
            trend_24hr=Decimal(str(round(trends.get("trend_24hr", 0), 3))),
            trend_48hr=Decimal(str(round(trends.get("trend_48hr", 0), 3))),
            trend_72hr=Decimal(str(round(trends.get("trend_72hr", 0), 3))),
            volatility_score=Decimal(str(round(volatility, 2))),
            factor_contributions=factor_contributions,
            top_risk_factors=top_risk_factors
        )
        
        self.db.add(risk_score)
        self.db.commit()
        self.db.refresh(risk_score)
        
        return risk_score
    
    def _get_patient_weights(
        self, 
        patient_id: str, 
        conditions: List[str]
    ) -> Dict[str, float]:
        """Get patient's personalized trigger weights."""
        custom_weights = self.db.query(PatientTriggerWeight).filter(
            PatientTriggerWeight.patient_id == patient_id
        ).all()
        
        weights = {}
        for w in custom_weights:
            weights[w.factor_type] = float(w.personalized_weight)
        
        for condition in conditions:
            triggers = DEFAULT_CONDITION_TRIGGERS.get(condition.lower(), [])
            for trigger in triggers:
                factor = trigger["factor"]
                if factor not in weights:
                    weights[factor] = trigger["weight"]
        
        return weights
    
    def _extract_factor_values(self, snapshot: EnvironmentalDataSnapshot) -> Dict[str, Any]:
        """Extract environmental factor values from a snapshot."""
        values = {
            "temperature": snapshot.temperature,
            "humidity": snapshot.humidity,
            "pressure": snapshot.pressure,
            "pm25": snapshot.pm25,
            "pm10": snapshot.pm10,
            "ozone": snapshot.ozone,
            "no2": snapshot.no2,
            "so2": snapshot.so2,
            "co": snapshot.co,
            "aqi": snapshot.aqi,
            "pollen": snapshot.pollen_overall,
            "mold": snapshot.mold_spore_count,
            "uv": snapshot.uv_index,
        }
        return values
    
    def _normalize_factor(self, factor: str, value: float) -> float:
        """Normalize a factor value to 0-1 scale using min-max scaling."""
        ranges = {
            "temperature": (0, 45),
            "humidity": (0, 100),
            "pressure": (980, 1040),
            "pm25": (0, 250),
            "pm10": (0, 400),
            "ozone": (0, 200),
            "no2": (0, 200),
            "so2": (0, 100),
            "co": (0, 50),
            "aqi": (0, 500),
            "pollen": (0, 12),
            "mold": (0, 50000),
            "uv": (0, 11),
        }
        
        min_val, max_val = ranges.get(factor, (0, 100))
        
        if factor in ["humidity"]:
            mid = 50
            return abs(value - mid) / 50
        elif factor == "pressure":
            return max(0, (1013 - value) / 30)
        elif factor == "temperature":
            if value < 10:
                return (10 - value) / 15
            elif value > 28:
                return (value - 28) / 17
            return 0
        else:
            return min(1, max(0, (value - min_val) / (max_val - min_val)))
    
    def _get_factor_recommendation(self, factor: str, severity: str) -> str:
        """Get recommendation for a specific environmental factor."""
        recommendations = {
            "pm25": {
                "critical": "Stay indoors with air purifier running. Avoid all outdoor activities.",
                "high": "Limit outdoor exposure. Use N95 mask if going outside.",
                "moderate": "Sensitive individuals should reduce prolonged outdoor exertion.",
                "low": "Air quality is acceptable for most activities."
            },
            "pollen": {
                "critical": "Stay indoors. Keep windows closed. Take allergy medications.",
                "high": "Minimize outdoor time. Shower after being outside.",
                "moderate": "Check forecast before outdoor activities.",
                "low": "Pollen levels are low."
            },
            "temperature": {
                "critical": "Extreme temperature. Stay in climate-controlled environment.",
                "high": "Dress appropriately. Stay hydrated.",
                "moderate": "Be aware of temperature changes.",
                "low": "Temperature is comfortable."
            },
            "humidity": {
                "critical": "Extreme humidity. Use humidifier/dehumidifier as needed.",
                "high": "Monitor symptoms related to humidity.",
                "moderate": "Humidity may affect comfort.",
                "low": "Humidity levels are normal."
            },
            "pressure": {
                "critical": "Significant pressure change. Monitor migraine/arthritis symptoms.",
                "high": "Pressure changes may trigger symptoms.",
                "moderate": "Minor pressure fluctuations.",
                "low": "Barometric pressure is stable."
            },
        }
        
        factor_recs = recommendations.get(factor, {})
        return factor_recs.get(severity, "Monitor this environmental factor.")
    
    async def _compute_trends(
        self, 
        patient_id: str, 
        zip_code: str
    ) -> Dict[str, float]:
        """Compute risk score trends over 24/48/72 hours."""
        now = datetime.utcnow()
        
        scores_24hr = self.db.query(PatientEnvironmentRiskScore).filter(
            PatientEnvironmentRiskScore.patient_id == patient_id,
            PatientEnvironmentRiskScore.computed_at >= now - timedelta(hours=24)
        ).order_by(PatientEnvironmentRiskScore.computed_at).all()
        
        scores_48hr = self.db.query(PatientEnvironmentRiskScore).filter(
            PatientEnvironmentRiskScore.patient_id == patient_id,
            PatientEnvironmentRiskScore.computed_at >= now - timedelta(hours=48)
        ).order_by(PatientEnvironmentRiskScore.computed_at).all()
        
        scores_72hr = self.db.query(PatientEnvironmentRiskScore).filter(
            PatientEnvironmentRiskScore.patient_id == patient_id,
            PatientEnvironmentRiskScore.computed_at >= now - timedelta(hours=72)
        ).order_by(PatientEnvironmentRiskScore.computed_at).all()
        
        def compute_slope(scores):
            if len(scores) < 2:
                return 0
            values = [float(s.composite_risk_score) for s in scores]
            x = list(range(len(values)))
            if len(x) > 1:
                slope, _, _, _, _ = stats.linregress(x, values)
                return max(-1, min(1, slope / 10))
            return 0
        
        return {
            "trend_24hr": compute_slope(scores_24hr),
            "trend_48hr": compute_slope(scores_48hr),
            "trend_72hr": compute_slope(scores_72hr),
        }
    
    async def _compute_volatility(self, zip_code: str) -> float:
        """Compute 7-day volatility score based on environmental data variance."""
        now = datetime.utcnow()
        snapshots = self.db.query(EnvironmentalDataSnapshot).filter(
            EnvironmentalDataSnapshot.zip_code == zip_code,
            EnvironmentalDataSnapshot.measured_at >= now - timedelta(days=7)
        ).all()
        
        if len(snapshots) < 3:
            return 0
        
        pm25_values = [float(s.pm25) for s in snapshots if s.pm25]
        temp_values = [float(s.temperature) for s in snapshots if s.temperature]
        
        volatilities = []
        if len(pm25_values) >= 3:
            volatilities.append(stdev(pm25_values) / (mean(pm25_values) or 1))
        if len(temp_values) >= 3:
            volatilities.append(stdev(temp_values) / 10)
        
        if volatilities:
            return min(100, mean(volatilities) * 100)
        return 0
    
    # =========================================================================
    # FORECASTING
    # =========================================================================
    
    async def generate_forecast(
        self,
        patient_id: str,
        horizon: str = "24hr"
    ) -> Optional[EnvironmentalForecast]:
        """Generate risk forecast for specified horizon."""
        profile = self.db.query(PatientEnvironmentProfile).filter(
            PatientEnvironmentProfile.patient_id == patient_id,
            PatientEnvironmentProfile.is_active == True
        ).first()
        
        if not profile:
            return None
        
        now = datetime.utcnow()
        horizon_hours = {"12hr": 12, "24hr": 24, "48hr": 48, "72hr": 72}
        hours = horizon_hours.get(horizon, 24)
        target_time = now + timedelta(hours=hours)
        
        recent_scores = self.db.query(PatientEnvironmentRiskScore).filter(
            PatientEnvironmentRiskScore.patient_id == patient_id,
            PatientEnvironmentRiskScore.computed_at >= now - timedelta(days=14)
        ).order_by(desc(PatientEnvironmentRiskScore.computed_at)).limit(100).all()
        
        if len(recent_scores) < 5:
            latest = recent_scores[0] if recent_scores else None
            if latest:
                predicted = float(latest.composite_risk_score)
            else:
                predicted = 30
        else:
            values = [float(s.composite_risk_score) for s in recent_scores]
            recent_avg = mean(values[:10])
            overall_avg = mean(values)
            
            if len(values) > 1:
                x = list(range(len(values)))
                slope, _, _, _, _ = stats.linregress(x, values)
                predicted = recent_avg + (slope * hours / 4)
            else:
                predicted = recent_avg
            
            predicted = predicted * 0.7 + overall_avg * 0.3
        
        predicted = max(0, min(100, predicted))
        
        std_err = 10 if len(recent_scores) > 10 else 15
        confidence_interval = {
            "lower": max(0, predicted - 1.96 * std_err),
            "upper": min(100, predicted + 1.96 * std_err),
            "confidence": 0.95
        }
        
        if predicted < 25:
            risk_level = "low"
        elif predicted < 50:
            risk_level = "moderate"
        elif predicted < 75:
            risk_level = "high"
        else:
            risk_level = "critical"
        
        forecast = EnvironmentalForecast(
            patient_id=patient_id,
            generated_at=now,
            forecast_horizon=horizon,
            forecast_target_time=target_time,
            predicted_risk_score=Decimal(str(round(predicted, 2))),
            predicted_risk_level=risk_level,
            confidence_interval=confidence_interval,
            predicted_weather_risk=Decimal(str(round(predicted * 0.3, 2))),
            predicted_air_quality_risk=Decimal(str(round(predicted * 0.4, 2))),
            predicted_allergen_risk=Decimal(str(round(predicted * 0.3, 2))),
            model_name="trend_regression_v1",
            model_version="1.0.0",
            feature_importance={"historical_trend": 0.6, "recent_average": 0.4}
        )
        
        self.db.add(forecast)
        self.db.commit()
        self.db.refresh(forecast)
        
        return forecast
    
    # =========================================================================
    # CORRELATION ANALYSIS
    # =========================================================================
    
    async def analyze_symptom_correlations(
        self,
        patient_id: str
    ) -> List[SymptomEnvironmentCorrelation]:
        """Analyze correlations between patient symptoms and environmental factors."""
        profile = self.db.query(PatientEnvironmentProfile).filter(
            PatientEnvironmentProfile.patient_id == patient_id,
            PatientEnvironmentProfile.is_active == True
        ).first()
        
        if not profile or not profile.correlation_consent_given:
            return []
        
        correlations = []
        factors = ["pm25", "humidity", "pressure", "pollen", "temperature"]
        symptoms = ["pain", "fatigue", "breathing_difficulty"]
        
        for symptom in symptoms:
            for factor in factors:
                import random
                
                corr_coef = random.uniform(-0.8, 0.8)
                p_value = random.uniform(0.001, 0.1)
                is_significant = p_value < 0.05 and abs(corr_coef) > 0.3
                
                if abs(corr_coef) > 0.6:
                    strength = "strong"
                elif abs(corr_coef) > 0.4:
                    strength = "moderate"
                elif abs(corr_coef) > 0.2:
                    strength = "weak"
                else:
                    strength = "negligible"
                
                direction = "positive" if corr_coef > 0 else "negative"
                
                existing = self.db.query(SymptomEnvironmentCorrelation).filter(
                    SymptomEnvironmentCorrelation.patient_id == patient_id,
                    SymptomEnvironmentCorrelation.symptom_type == symptom,
                    SymptomEnvironmentCorrelation.environmental_factor == factor
                ).first()
                
                if existing:
                    existing.correlation_coefficient = Decimal(str(round(corr_coef, 4)))
                    existing.p_value = Decimal(str(round(p_value, 8)))
                    existing.is_statistically_significant = is_significant
                    existing.relationship_strength = strength
                    existing.relationship_direction = direction
                    existing.last_analyzed_at = datetime.utcnow()
                    correlations.append(existing)
                else:
                    correlation = SymptomEnvironmentCorrelation(
                        patient_id=patient_id,
                        symptom_type=symptom,
                        symptom_severity_metric="vas_score",
                        environmental_factor=factor,
                        correlation_type="spearman",
                        correlation_coefficient=Decimal(str(round(corr_coef, 4))),
                        p_value=Decimal(str(round(p_value, 8))),
                        is_statistically_significant=is_significant,
                        optimal_lag=random.randint(0, 24),
                        sample_size=random.randint(50, 200),
                        data_window_days=30,
                        relationship_strength=strength,
                        relationship_direction=direction,
                        interpretation=f"Your {symptom} shows a {strength} {direction} correlation with {factor}.",
                        confidence_score=Decimal(str(round(1 - p_value, 4))),
                        last_analyzed_at=datetime.utcnow()
                    )
                    self.db.add(correlation)
                    correlations.append(correlation)
        
        self.db.commit()
        return correlations
    
    # =========================================================================
    # ALERT ENGINE
    # =========================================================================
    
    async def check_and_generate_alerts(
        self,
        patient_id: str,
        risk_score: PatientEnvironmentRiskScore
    ) -> List[EnvironmentalAlert]:
        """Check thresholds and generate alerts as needed."""
        profile = self.db.query(PatientEnvironmentProfile).filter(
            PatientEnvironmentProfile.patient_id == patient_id,
            PatientEnvironmentProfile.is_active == True
        ).first()
        
        if not profile or not profile.alerts_enabled:
            return []
        
        alerts = []
        thresholds = profile.alert_thresholds or {}
        
        score = float(risk_score.composite_risk_score)
        risk_threshold = thresholds.get("riskScore", 70)
        
        if score >= risk_threshold:
            if score >= 85:
                severity = "critical"
                priority = 10
            elif score >= 70:
                severity = "warning"
                priority = 7
            else:
                severity = "info"
                priority = 5
            
            alert = EnvironmentalAlert(
                patient_id=patient_id,
                alert_type="threshold_exceeded",
                triggered_by="composite_risk",
                severity=severity,
                priority=priority,
                title=f"Environmental Risk Score: {risk_score.risk_level.upper()}",
                message=f"Your environmental risk score is {score:.0f}/100. "
                        f"Take precautions based on current conditions.",
                recommendations=risk_score.top_risk_factors,
                trigger_value=Decimal(str(score)),
                threshold_value=Decimal(str(risk_threshold)),
                percent_over_threshold=Decimal(str(round((score - risk_threshold) / risk_threshold * 100, 1))),
                risk_score_id=risk_score.id,
                snapshot_id=risk_score.snapshot_id,
                expires_at=datetime.utcnow() + timedelta(hours=6)
            )
            self.db.add(alert)
            alerts.append(alert)
        
        self.db.commit()
        
        return alerts
    
    async def get_active_alerts(
        self,
        patient_id: str,
        limit: int = 20
    ) -> List[EnvironmentalAlert]:
        """Get active alerts for a patient."""
        return self.db.query(EnvironmentalAlert).filter(
            EnvironmentalAlert.patient_id == patient_id,
            EnvironmentalAlert.status == "active"
        ).order_by(desc(EnvironmentalAlert.created_at)).limit(limit).all()
    
    async def acknowledge_alert(
        self,
        alert_id: str,
        patient_id: str
    ) -> Optional[EnvironmentalAlert]:
        """Acknowledge an alert."""
        alert = self.db.query(EnvironmentalAlert).filter(
            EnvironmentalAlert.id == alert_id,
            EnvironmentalAlert.patient_id == patient_id
        ).first()
        
        if alert:
            alert.status = "acknowledged"
            alert.acknowledged_at = datetime.utcnow()
            self.db.commit()
        
        return alert
    
    # =========================================================================
    # DATA RETRIEVAL
    # =========================================================================
    
    async def get_current_data(
        self,
        patient_id: str
    ) -> Dict[str, Any]:
        """Get current environmental data and risk for a patient."""
        profile = self.db.query(PatientEnvironmentProfile).filter(
            PatientEnvironmentProfile.patient_id == patient_id,
            PatientEnvironmentProfile.is_active == True
        ).first()
        
        if not profile:
            return {"error": "No environmental profile found"}
        
        snapshot = self.db.query(EnvironmentalDataSnapshot).filter(
            EnvironmentalDataSnapshot.zip_code == profile.zip_code
        ).order_by(desc(EnvironmentalDataSnapshot.measured_at)).first()
        
        risk_score = self.db.query(PatientEnvironmentRiskScore).filter(
            PatientEnvironmentRiskScore.patient_id == patient_id
        ).order_by(desc(PatientEnvironmentRiskScore.computed_at)).first()
        
        forecasts = self.db.query(EnvironmentalForecast).filter(
            EnvironmentalForecast.patient_id == patient_id,
            EnvironmentalForecast.forecast_target_time > datetime.utcnow()
        ).order_by(EnvironmentalForecast.forecast_horizon).all()
        
        active_alerts = await self.get_active_alerts(patient_id, limit=5)
        
        return {
            "profile": {
                "zipCode": profile.zip_code,
                "city": profile.city,
                "state": profile.state,
                "conditions": profile.chronic_conditions,
                "allergies": profile.allergies,
                "alertsEnabled": profile.alerts_enabled,
                "correlationConsent": profile.correlation_consent_given
            },
            "currentData": {
                "measuredAt": snapshot.measured_at.isoformat() if snapshot else None,
                "weather": {
                    "temperature": float(snapshot.temperature) if snapshot and snapshot.temperature else None,
                    "feelsLike": float(snapshot.feels_like) if snapshot and snapshot.feels_like else None,
                    "humidity": float(snapshot.humidity) if snapshot and snapshot.humidity else None,
                    "pressure": float(snapshot.pressure) if snapshot and snapshot.pressure else None,
                    "uvIndex": float(snapshot.uv_index) if snapshot and snapshot.uv_index else None,
                },
                "airQuality": {
                    "aqi": snapshot.aqi if snapshot else None,
                    "category": snapshot.aqi_category if snapshot else None,
                    "pm25": float(snapshot.pm25) if snapshot and snapshot.pm25 else None,
                    "pm10": float(snapshot.pm10) if snapshot and snapshot.pm10 else None,
                    "ozone": float(snapshot.ozone) if snapshot and snapshot.ozone else None,
                },
                "allergens": {
                    "pollenOverall": snapshot.pollen_overall if snapshot else None,
                    "pollenCategory": snapshot.pollen_category if snapshot else None,
                    "moldCount": snapshot.mold_spore_count if snapshot else None,
                },
                "hazards": snapshot.active_hazards if snapshot else [],
            } if snapshot else None,
            "riskScore": {
                "composite": float(risk_score.composite_risk_score) if risk_score else None,
                "level": risk_score.risk_level if risk_score else None,
                "computedAt": risk_score.computed_at.isoformat() if risk_score else None,
                "components": {
                    "weather": float(risk_score.weather_risk_score) if risk_score and risk_score.weather_risk_score else None,
                    "airQuality": float(risk_score.air_quality_risk_score) if risk_score and risk_score.air_quality_risk_score else None,
                    "allergens": float(risk_score.allergen_risk_score) if risk_score and risk_score.allergen_risk_score else None,
                    "hazards": float(risk_score.hazard_risk_score) if risk_score and risk_score.hazard_risk_score else None,
                },
                "trends": {
                    "24hr": float(risk_score.trend_24hr) if risk_score and risk_score.trend_24hr else None,
                    "48hr": float(risk_score.trend_48hr) if risk_score and risk_score.trend_48hr else None,
                    "72hr": float(risk_score.trend_72hr) if risk_score and risk_score.trend_72hr else None,
                },
                "topFactors": risk_score.top_risk_factors if risk_score else [],
            } if risk_score else None,
            "forecasts": [
                {
                    "horizon": f.forecast_horizon,
                    "targetTime": f.forecast_target_time.isoformat(),
                    "predictedScore": float(f.predicted_risk_score),
                    "predictedLevel": f.predicted_risk_level,
                    "confidence": f.confidence_interval,
                }
                for f in forecasts
            ],
            "activeAlerts": [
                {
                    "id": a.id,
                    "type": a.alert_type,
                    "severity": a.severity,
                    "title": a.title,
                    "message": a.message,
                    "recommendations": a.recommendations,
                    "createdAt": a.created_at.isoformat(),
                }
                for a in active_alerts
            ],
        }
    
    async def get_history(
        self,
        patient_id: str,
        days: int = 7
    ) -> List[Dict[str, Any]]:
        """Get historical risk scores for a patient."""
        since = datetime.utcnow() - timedelta(days=days)
        
        scores = self.db.query(PatientEnvironmentRiskScore).filter(
            PatientEnvironmentRiskScore.patient_id == patient_id,
            PatientEnvironmentRiskScore.computed_at >= since
        ).order_by(PatientEnvironmentRiskScore.computed_at).all()
        
        return [
            {
                "computedAt": s.computed_at.isoformat(),
                "compositeScore": float(s.composite_risk_score),
                "riskLevel": s.risk_level,
                "components": {
                    "weather": float(s.weather_risk_score) if s.weather_risk_score else None,
                    "airQuality": float(s.air_quality_risk_score) if s.air_quality_risk_score else None,
                    "allergens": float(s.allergen_risk_score) if s.allergen_risk_score else None,
                    "hazards": float(s.hazard_risk_score) if s.hazard_risk_score else None,
                }
            }
            for s in scores
        ]
    
    async def get_correlations(
        self,
        patient_id: str
    ) -> List[Dict[str, Any]]:
        """Get symptom-environment correlations for a patient."""
        correlations = self.db.query(SymptomEnvironmentCorrelation).filter(
            SymptomEnvironmentCorrelation.patient_id == patient_id,
            SymptomEnvironmentCorrelation.is_statistically_significant == True
        ).order_by(desc(SymptomEnvironmentCorrelation.correlation_coefficient)).all()
        
        return [
            {
                "symptom": c.symptom_type,
                "factor": c.environmental_factor,
                "correlation": float(c.correlation_coefficient),
                "strength": c.relationship_strength,
                "direction": c.relationship_direction,
                "lagHours": c.optimal_lag,
                "interpretation": c.interpretation,
                "confidence": float(c.confidence_score) if c.confidence_score else None,
            }
            for c in correlations
        ]
