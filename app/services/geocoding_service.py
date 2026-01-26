"""
Geocoding Service
Reverse geocode latitude/longitude to ZIP code using Google or OpenCage API.
Stores ZIP only by default for privacy - never raw lat/lon without explicit consent.
"""

import os
import logging
import httpx
from typing import Optional, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

GOOGLE_GEOCODING_API_KEY = os.getenv("GOOGLE_GEOCODING_API_KEY")
OPENCAGE_API_KEY = os.getenv("OPENCAGE_API_KEY")


@dataclass
class GeocodingResult:
    """Result of a reverse geocoding operation."""
    zip_code: str
    city: Optional[str] = None
    state: Optional[str] = None
    country: Optional[str] = None
    formatted_address: Optional[str] = None
    source: str = "unknown"


class GeocodingService:
    """
    Geocoding service with privacy-first design.
    - Reverse geocodes lat/lon to ZIP code
    - Uses Google Geocoding API or OpenCage as fallback
    - Never stores raw lat/lon without explicit consent
    """
    
    def __init__(self):
        self.google_api_key = GOOGLE_GEOCODING_API_KEY
        self.opencage_api_key = OPENCAGE_API_KEY
        self.timeout = 10.0
    
    async def reverse_geocode(
        self,
        lat: float,
        lon: float
    ) -> Optional[GeocodingResult]:
        """
        Reverse geocode coordinates to location info.
        Returns ZIP code, city, state - never stores raw coordinates.
        
        Args:
            lat: Latitude
            lon: Longitude
            
        Returns:
            GeocodingResult with ZIP and location details, or None on failure
        """
        if self.google_api_key:
            result = await self._reverse_geocode_google(lat, lon)
            if result:
                return result
        
        if self.opencage_api_key:
            result = await self._reverse_geocode_opencage(lat, lon)
            if result:
                return result
        
        logger.warning("No geocoding API keys configured")
        return None
    
    async def _reverse_geocode_google(
        self,
        lat: float,
        lon: float
    ) -> Optional[GeocodingResult]:
        """Reverse geocode using Google Geocoding API."""
        try:
            url = "https://maps.googleapis.com/maps/api/geocode/json"
            params = {
                "latlng": f"{lat},{lon}",
                "key": self.google_api_key,
                "result_type": "postal_code"
            }
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(url, params=params)
                response.raise_for_status()
                data = response.json()
            
            if data.get("status") != "OK" or not data.get("results"):
                params["result_type"] = "locality"
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.get(url, params=params)
                    data = response.json()
            
            if data.get("status") != "OK" or not data.get("results"):
                logger.warning(f"Google geocoding failed: {data.get('status')}")
                return None
            
            result = data["results"][0]
            address_components = result.get("address_components", [])
            
            zip_code = None
            city = None
            state = None
            country = None
            
            for component in address_components:
                types = component.get("types", [])
                if "postal_code" in types:
                    zip_code = component.get("long_name")
                elif "locality" in types:
                    city = component.get("long_name")
                elif "administrative_area_level_1" in types:
                    state = component.get("short_name")
                elif "country" in types:
                    country = component.get("short_name")
            
            if not zip_code:
                logger.warning("No postal code found in Google response")
                return None
            
            return GeocodingResult(
                zip_code=zip_code,
                city=city,
                state=state,
                country=country,
                formatted_address=result.get("formatted_address"),
                source="google"
            )
            
        except Exception as e:
            logger.error(f"Google geocoding error: {e}")
            return None
    
    async def _reverse_geocode_opencage(
        self,
        lat: float,
        lon: float
    ) -> Optional[GeocodingResult]:
        """Reverse geocode using OpenCage API."""
        try:
            url = "https://api.opencagedata.com/geocode/v1/json"
            params = {
                "q": f"{lat},{lon}",
                "key": self.opencage_api_key,
                "no_annotations": 1,
                "limit": 1
            }
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(url, params=params)
                response.raise_for_status()
                data = response.json()
            
            if not data.get("results"):
                logger.warning("OpenCage returned no results")
                return None
            
            result = data["results"][0]
            components = result.get("components", {})
            
            zip_code = components.get("postcode")
            if not zip_code:
                logger.warning("No postal code in OpenCage response")
                return None
            
            return GeocodingResult(
                zip_code=zip_code,
                city=components.get("city") or components.get("town") or components.get("village"),
                state=components.get("state_code") or components.get("state"),
                country=components.get("country_code"),
                formatted_address=result.get("formatted"),
                source="opencage"
            )
            
        except Exception as e:
            logger.error(f"OpenCage geocoding error: {e}")
            return None
    
    async def geocode_address(
        self,
        address: str
    ) -> Optional[GeocodingResult]:
        """
        Forward geocode an address to get ZIP code.
        Useful for validating user-entered addresses.
        """
        if self.google_api_key:
            try:
                url = "https://maps.googleapis.com/maps/api/geocode/json"
                params = {
                    "address": address,
                    "key": self.google_api_key
                }
                
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.get(url, params=params)
                    response.raise_for_status()
                    data = response.json()
                
                if data.get("status") == "OK" and data.get("results"):
                    result = data["results"][0]
                    location = result.get("geometry", {}).get("location", {})
                    if location:
                        return await self.reverse_geocode(
                            location.get("lat"),
                            location.get("lng")
                        )
            except Exception as e:
                logger.error(f"Forward geocoding error: {e}")
        
        return None


geocoding_service = GeocodingService()


async def reverse_geocode(lat: float, lon: float) -> Optional[Dict[str, Any]]:
    """
    Convenience function for reverse geocoding.
    Returns dict with zip, city, state or None on failure.
    """
    result = await geocoding_service.reverse_geocode(lat, lon)
    if result:
        return {
            "zip": result.zip_code,
            "city": result.city,
            "state": result.state,
            "country": result.country
        }
    return None
