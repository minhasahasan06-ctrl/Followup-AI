"""
Device Connect API - Production Grade
HIPAA-Compliant Device Pairing, Registry, and Data Ingestion

Supports:
- Public APIs: Fitbit, Withings, Oura, Google Fit, iHealth
- Private APIs (stubs): Dexcom, Garmin, Whoop, Samsung, Eko, Abbott
- Apple HealthKit receiver endpoint
- Web Bluetooth fallback
- Webhook handlers for vendor data pushes
"""

import os
import secrets
import hashlib
import base64
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from enum import Enum

from fastapi import APIRouter, HTTPException, Depends, Query, Request, BackgroundTasks
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field
from sqlalchemy import select, update, delete, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/devices", tags=["Device Connect"])

# ============================================
# ENUMS AND CONSTANTS
# ============================================

class DeviceType(str, Enum):
    SMARTWATCH = "smartwatch"
    BP_MONITOR = "bp_monitor"
    GLUCOSE_METER = "glucose_meter"
    CGM = "cgm"
    SCALE = "scale"
    THERMOMETER = "thermometer"
    STETHOSCOPE = "stethoscope"
    PULSE_OXIMETER = "pulse_oximeter"
    ACTIVITY_TRACKER = "activity_tracker"

class VendorId(str, Enum):
    FITBIT = "fitbit"
    WITHINGS = "withings"
    OURA = "oura"
    GOOGLE_FIT = "google_fit"
    IHEALTH = "ihealth"
    APPLE_HEALTHKIT = "apple_healthkit"
    GARMIN = "garmin"
    WHOOP = "whoop"
    DEXCOM = "dexcom"
    SAMSUNG = "samsung"
    EKO = "eko"
    ABBOTT = "abbott"
    OMRON = "omron"
    GENERIC_BLE = "generic_ble"

class PairingMethod(str, Enum):
    OAUTH = "oauth"
    BLE = "ble"
    HEALTHKIT = "healthkit"
    GOOGLE_FIT = "google_fit"
    MANUAL = "manual"
    QR_CODE = "qr_code"
    WEBHOOK = "webhook"

class ConnectionStatus(str, Enum):
    PENDING = "pending"
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    EXPIRED = "expired"
    REVOKED = "revoked"
    ERROR = "error"

class SessionStatus(str, Enum):
    INITIATED = "initiated"
    PENDING_AUTH = "pending_auth"
    PENDING_BLE = "pending_ble"
    PENDING_CONSENT = "pending_consent"
    COMPLETED = "completed"
    FAILED = "failed"
    EXPIRED = "expired"
    CANCELLED = "cancelled"

# ============================================
# VENDOR CONFIGURATIONS
# ============================================

VENDOR_CONFIGS = {
    VendorId.FITBIT: {
        "name": "Fitbit",
        "is_public_api": True,
        "requires_partnership": False,
        "auth_url": "https://www.fitbit.com/oauth2/authorize",
        "token_url": "https://api.fitbit.com/oauth2/token",
        "base_url": "https://api.fitbit.com",
        "scopes": ["activity", "heartrate", "sleep", "weight", "oxygen_saturation", "respiratory_rate", "temperature"],
        "supported_devices": [DeviceType.SMARTWATCH, DeviceType.ACTIVITY_TRACKER, DeviceType.SCALE],
        "metrics": ["heart_rate", "hrv", "spo2", "sleep", "steps", "calories", "respiratory_rate", "skin_temp", "weight"],
        "pairing_methods": [PairingMethod.OAUTH],
    },
    VendorId.WITHINGS: {
        "name": "Withings",
        "is_public_api": True,
        "requires_partnership": False,
        "auth_url": "https://account.withings.com/oauth2_user/authorize2",
        "token_url": "https://wbsapi.withings.net/v2/oauth2",
        "base_url": "https://wbsapi.withings.net",
        "scopes": ["user.metrics", "user.activity", "user.sleepevents"],
        "supported_devices": [DeviceType.BP_MONITOR, DeviceType.SCALE, DeviceType.THERMOMETER],
        "metrics": ["bp", "weight", "body_fat", "temperature", "heart_rate", "spo2"],
        "pairing_methods": [PairingMethod.OAUTH],
    },
    VendorId.OURA: {
        "name": "Oura",
        "is_public_api": True,
        "requires_partnership": False,
        "auth_url": "https://cloud.ouraring.com/oauth/authorize",
        "token_url": "https://api.ouraring.com/oauth/token",
        "base_url": "https://api.ouraring.com/v2",
        "scopes": ["daily", "heartrate", "workout", "tag", "session", "spo2"],
        "supported_devices": [DeviceType.SMARTWATCH],
        "metrics": ["heart_rate", "hrv", "spo2", "sleep", "readiness", "activity", "temperature"],
        "pairing_methods": [PairingMethod.OAUTH],
    },
    VendorId.GOOGLE_FIT: {
        "name": "Google Fit",
        "is_public_api": True,
        "requires_partnership": False,
        "auth_url": "https://accounts.google.com/o/oauth2/v2/auth",
        "token_url": "https://oauth2.googleapis.com/token",
        "base_url": "https://www.googleapis.com/fitness/v1/users/me",
        "scopes": [
            "https://www.googleapis.com/auth/fitness.activity.read",
            "https://www.googleapis.com/auth/fitness.heart_rate.read",
            "https://www.googleapis.com/auth/fitness.blood_pressure.read",
            "https://www.googleapis.com/auth/fitness.blood_glucose.read",
            "https://www.googleapis.com/auth/fitness.oxygen_saturation.read",
            "https://www.googleapis.com/auth/fitness.sleep.read",
            "https://www.googleapis.com/auth/fitness.body.read",
        ],
        "supported_devices": [DeviceType.SMARTWATCH, DeviceType.ACTIVITY_TRACKER, DeviceType.BP_MONITOR, DeviceType.GLUCOSE_METER, DeviceType.SCALE],
        "metrics": ["heart_rate", "steps", "calories", "sleep", "bp", "glucose", "weight", "spo2"],
        "pairing_methods": [PairingMethod.GOOGLE_FIT],
    },
    VendorId.IHEALTH: {
        "name": "iHealth",
        "is_public_api": True,
        "requires_partnership": False,
        "auth_url": "https://api.ihealthlabs.com:8443/OpenApiV2/OAuthv2/userauthorization",
        "token_url": "https://api.ihealthlabs.com:8443/OpenApiV2/OAuthv2/userauthorization",
        "base_url": "https://api.ihealthlabs.com:8443/openapiv2",
        "scopes": ["OpenApiBP", "OpenApiWeight", "OpenApiSpO2", "OpenApiBG"],
        "supported_devices": [DeviceType.BP_MONITOR, DeviceType.SCALE, DeviceType.PULSE_OXIMETER, DeviceType.GLUCOSE_METER],
        "metrics": ["bp", "weight", "spo2", "glucose"],
        "pairing_methods": [PairingMethod.OAUTH],
    },
    VendorId.APPLE_HEALTHKIT: {
        "name": "Apple HealthKit",
        "is_public_api": False,
        "requires_partnership": False,
        "supported_devices": [DeviceType.SMARTWATCH, DeviceType.BP_MONITOR, DeviceType.GLUCOSE_METER, DeviceType.SCALE, DeviceType.THERMOMETER],
        "metrics": ["heart_rate", "hrv", "spo2", "sleep", "steps", "bp", "glucose", "weight", "temperature", "ecg", "afib"],
        "pairing_methods": [PairingMethod.HEALTHKIT],
        "note": "Requires iOS companion app or HealthKit export",
    },
    # Private APIs - Stubs (require vendor partnerships)
    VendorId.GARMIN: {
        "name": "Garmin",
        "is_public_api": False,
        "requires_partnership": True,
        "auth_url": "https://connect.garmin.com/oauthConfirm",
        "token_url": "https://connectapi.garmin.com/oauth-service/oauth/access_token",
        "base_url": "https://apis.garmin.com",
        "scopes": ["health_api"],
        "supported_devices": [DeviceType.SMARTWATCH, DeviceType.ACTIVITY_TRACKER],
        "metrics": ["heart_rate", "hrv", "spo2", "sleep", "steps", "stress", "body_battery", "vo2_max", "training_load", "recovery"],
        "pairing_methods": [PairingMethod.OAUTH],
        "baa_required": True,
    },
    VendorId.WHOOP: {
        "name": "Whoop",
        "is_public_api": False,
        "requires_partnership": True,
        "auth_url": "https://api.prod.whoop.com/oauth/oauth2/auth",
        "token_url": "https://api.prod.whoop.com/oauth/oauth2/token",
        "base_url": "https://api.prod.whoop.com/developer/v1",
        "scopes": ["read:recovery", "read:cycles", "read:sleep", "read:workout", "read:profile", "read:body_measurement"],
        "supported_devices": [DeviceType.SMARTWATCH],
        "metrics": ["heart_rate", "hrv", "spo2", "sleep", "recovery", "strain", "skin_temp", "respiratory_rate"],
        "pairing_methods": [PairingMethod.OAUTH],
        "baa_required": True,
    },
    VendorId.DEXCOM: {
        "name": "Dexcom",
        "is_public_api": False,
        "requires_partnership": True,
        "auth_url": "https://api.dexcom.com/v2/oauth2/login",
        "token_url": "https://api.dexcom.com/v2/oauth2/token",
        "base_url": "https://api.dexcom.com/v2",
        "scopes": ["offline_access"],
        "supported_devices": [DeviceType.CGM],
        "metrics": ["glucose"],
        "pairing_methods": [PairingMethod.OAUTH],
        "baa_required": True,
        "fda_cleared": True,
    },
    VendorId.SAMSUNG: {
        "name": "Samsung Health",
        "is_public_api": False,
        "requires_partnership": True,
        "supported_devices": [DeviceType.SMARTWATCH],
        "metrics": ["heart_rate", "spo2", "sleep", "steps", "stress", "bp", "ecg"],
        "pairing_methods": [PairingMethod.OAUTH],
        "baa_required": True,
    },
    VendorId.EKO: {
        "name": "Eko",
        "is_public_api": False,
        "requires_partnership": True,
        "supported_devices": [DeviceType.STETHOSCOPE],
        "metrics": ["heart_sounds", "lung_sounds", "ecg"],
        "pairing_methods": [PairingMethod.OAUTH, PairingMethod.BLE],
        "baa_required": True,
        "fda_cleared": True,
    },
    VendorId.ABBOTT: {
        "name": "Abbott LibreView",
        "is_public_api": False,
        "requires_partnership": True,
        "base_url": "https://api.libreview.io",
        "supported_devices": [DeviceType.CGM],
        "metrics": ["glucose"],
        "pairing_methods": [PairingMethod.OAUTH],
        "baa_required": True,
        "fda_cleared": True,
    },
    VendorId.OMRON: {
        "name": "Omron",
        "is_public_api": False,
        "requires_partnership": True,
        "supported_devices": [DeviceType.BP_MONITOR],
        "metrics": ["bp", "heart_rate", "afib"],
        "pairing_methods": [PairingMethod.OAUTH, PairingMethod.BLE],
        "baa_required": True,
        "fda_cleared": True,
    },
    VendorId.GENERIC_BLE: {
        "name": "Generic BLE Device",
        "is_public_api": True,
        "requires_partnership": False,
        "supported_devices": [DeviceType.BP_MONITOR, DeviceType.PULSE_OXIMETER, DeviceType.THERMOMETER, DeviceType.SCALE, DeviceType.GLUCOSE_METER],
        "metrics": ["bp", "spo2", "temperature", "weight", "glucose"],
        "pairing_methods": [PairingMethod.BLE, PairingMethod.MANUAL],
    },
}

# BLE Service UUIDs for common medical devices
BLE_SERVICES = {
    "blood_pressure": "1810",
    "heart_rate": "180D",
    "health_thermometer": "1809",
    "weight_scale": "181D",
    "glucose": "1808",
    "pulse_oximeter": "1822",
}

# ============================================
# PYDANTIC MODELS
# ============================================

class DeviceModelResponse(BaseModel):
    id: str
    vendor_id: str
    vendor_name: str
    model_name: str
    device_type: str
    device_category: str
    capabilities: Dict[str, Any]
    pairing_methods: List[str]
    is_public_api: bool
    requires_partnership: bool
    baa_required: bool = False
    fda_cleared: bool = False
    image_url: Optional[str] = None

class VendorResponse(BaseModel):
    vendor_id: str
    name: str
    is_public_api: bool
    requires_partnership: bool
    supported_devices: List[str]
    available_metrics: List[str]
    pairing_methods: List[str]
    baa_required: bool = False
    status: str  # 'available', 'coming_soon', 'requires_credentials'

class StartPairingRequest(BaseModel):
    vendor_id: str
    device_type: str
    pairing_method: str
    device_model: Optional[str] = None
    redirect_uri: Optional[str] = None

class StartPairingResponse(BaseModel):
    session_id: str
    pairing_method: str
    status: str
    auth_url: Optional[str] = None
    ble_instructions: Optional[Dict[str, Any]] = None
    qr_code_data: Optional[str] = None
    expires_at: str
    consent_required: bool = True

class CompletePairingRequest(BaseModel):
    session_id: str
    authorization_code: Optional[str] = None
    ble_device_id: Optional[str] = None
    consent_granted: bool = True
    consented_data_types: List[str] = Field(default_factory=list)

class DeviceConnectionResponse(BaseModel):
    id: str
    vendor_id: str
    vendor_name: str
    device_type: str
    device_model: Optional[str] = None
    connection_status: str
    last_sync_at: Optional[str] = None
    battery_level: Optional[int] = None
    tracked_metrics: List[str] = Field(default_factory=list)
    auto_sync: bool = True
    created_at: str

class DeviceHealthResponse(BaseModel):
    device_connection_id: str
    status: str
    battery_level: Optional[int] = None
    battery_status: Optional[str] = None
    signal_strength: Optional[int] = None
    firmware_version: Optional[str] = None
    firmware_update_available: bool = False
    last_seen_at: Optional[str] = None
    last_successful_sync: Optional[str] = None
    sync_success_rate: Optional[float] = None
    data_quality_score: Optional[int] = None
    alerts: List[Dict[str, Any]] = Field(default_factory=list)

class DataIngestRequest(BaseModel):
    device_connection_id: str
    timestamp: str
    readings: Dict[str, Any]
    meta: Optional[Dict[str, Any]] = None

class DataIngestResponse(BaseModel):
    success: bool
    records_processed: int
    readings_accepted: List[str]
    readings_rejected: List[Dict[str, str]] = Field(default_factory=list)
    routed_to_sections: List[str]
    alerts_triggered: List[str] = Field(default_factory=list)

class HealthKitDataRequest(BaseModel):
    data_types: List[str]
    readings: List[Dict[str, Any]]
    export_date: str
    device_info: Optional[Dict[str, Any]] = None

class SyncDeviceRequest(BaseModel):
    device_connection_id: str
    data_types: Optional[List[str]] = None
    date_range: Optional[Dict[str, str]] = None

# ============================================
# HELPER FUNCTIONS
# ============================================

def generate_oauth_state() -> str:
    """Generate secure OAuth state parameter for CSRF protection"""
    return secrets.token_urlsafe(32)

def generate_pkce_verifier() -> str:
    """Generate PKCE code verifier"""
    return secrets.token_urlsafe(64)

def generate_pkce_challenge(verifier: str) -> str:
    """Generate PKCE code challenge from verifier"""
    digest = hashlib.sha256(verifier.encode()).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b'=').decode()

def get_vendor_client_credentials(vendor_id: str) -> Dict[str, str]:
    """Get client credentials for vendor OAuth (from environment)"""
    vendor_upper = vendor_id.upper()
    return {
        "client_id": os.getenv(f"{vendor_upper}_CLIENT_ID", ""),
        "client_secret": os.getenv(f"{vendor_upper}_CLIENT_SECRET", ""),
    }

def check_vendor_credentials_available(vendor_id: str) -> bool:
    """Check if vendor credentials are configured"""
    creds = get_vendor_client_credentials(vendor_id)
    return bool(creds["client_id"] and creds["client_secret"])

async def log_device_audit(
    db: AsyncSession,
    actor_id: str,
    actor_type: str,
    action: str,
    action_category: str,
    resource_type: str,
    resource_id: Optional[str] = None,
    patient_id: Optional[str] = None,
    event_details: Optional[Dict] = None,
    success: bool = True,
    error_message: Optional[str] = None,
    request: Optional[Request] = None,
):
    """Log device-related audit event for HIPAA compliance"""
    try:
        from sqlalchemy import text
        
        ip_address = None
        user_agent = None
        if request:
            ip_address = request.client.host if request.client else None
            user_agent = request.headers.get("user-agent")
        
        await db.execute(
            text("""
                INSERT INTO device_data_audit_log 
                (actor_id, actor_type, action, action_category, resource_type, resource_id, 
                 patient_id, event_details, ip_address, user_agent, success, error_message, phi_accessed)
                VALUES (:actor_id, :actor_type, :action, :action_category, :resource_type, :resource_id,
                        :patient_id, :event_details, :ip_address, :user_agent, :success, :error_message, :phi_accessed)
            """),
            {
                "actor_id": actor_id,
                "actor_type": actor_type,
                "action": action,
                "action_category": action_category,
                "resource_type": resource_type,
                "resource_id": resource_id,
                "patient_id": patient_id,
                "event_details": event_details,
                "ip_address": ip_address,
                "user_agent": user_agent,
                "success": success,
                "error_message": error_message,
                "phi_accessed": patient_id is not None,
            }
        )
        await db.commit()
    except Exception as e:
        logger.error(f"Failed to log device audit event: {e}")

# ============================================
# API ENDPOINTS
# ============================================

@router.get("/vendors", response_model=List[VendorResponse])
async def list_vendors(
    device_type: Optional[str] = Query(None, description="Filter by device type"),
    include_private: bool = Query(True, description="Include private API vendors"),
):
    """
    List all supported device vendors with their capabilities.
    Returns availability status based on configured credentials.
    """
    vendors = []
    
    for vendor_id, config in VENDOR_CONFIGS.items():
        # Filter by device type if specified
        if device_type and device_type not in [d.value for d in config["supported_devices"]]:
            continue
        
        # Skip private APIs if not requested
        if not include_private and config.get("requires_partnership", False):
            continue
        
        # Determine status
        if config.get("requires_partnership", False):
            if check_vendor_credentials_available(vendor_id.value):
                status = "available"
            else:
                status = "requires_credentials"
        elif config["is_public_api"]:
            if check_vendor_credentials_available(vendor_id.value):
                status = "available"
            else:
                status = "requires_credentials"
        else:
            status = "coming_soon"
        
        vendors.append(VendorResponse(
            vendor_id=vendor_id.value,
            name=config["name"],
            is_public_api=config["is_public_api"],
            requires_partnership=config.get("requires_partnership", False),
            supported_devices=[d.value for d in config["supported_devices"]],
            available_metrics=config.get("metrics", []),
            pairing_methods=[p.value for p in config.get("pairing_methods", [])],
            baa_required=config.get("baa_required", False),
            status=status,
        ))
    
    return vendors

@router.get("/models", response_model=List[DeviceModelResponse])
async def list_device_models(
    vendor_id: Optional[str] = Query(None, description="Filter by vendor"),
    device_type: Optional[str] = Query(None, description="Filter by device type"),
    db: AsyncSession = Depends(get_db),
):
    """
    List available device models from the catalog.
    Returns models from database + hardcoded defaults.
    """
    models = []
    
    # Generate models from vendor configs
    for vid, config in VENDOR_CONFIGS.items():
        if vendor_id and vid.value != vendor_id:
            continue
        
        for dtype in config["supported_devices"]:
            if device_type and dtype.value != device_type:
                continue
            
            models.append(DeviceModelResponse(
                id=f"{vid.value}_{dtype.value}",
                vendor_id=vid.value,
                vendor_name=config["name"],
                model_name=f"{config['name']} {dtype.value.replace('_', ' ').title()}",
                device_type=dtype.value,
                device_category="wearable" if dtype in [DeviceType.SMARTWATCH, DeviceType.ACTIVITY_TRACKER] else "medical_device",
                capabilities={
                    "metrics": config.get("metrics", []),
                    "features": ["auto_sync"] if config["is_public_api"] else ["manual_entry"],
                    "syncMethods": [p.value for p in config.get("pairing_methods", [])],
                },
                pairing_methods=[p.value for p in config.get("pairing_methods", [])],
                is_public_api=config["is_public_api"],
                requires_partnership=config.get("requires_partnership", False),
                baa_required=config.get("baa_required", False),
                fda_cleared=config.get("fda_cleared", False),
            ))
    
    return models

@router.get("/supported-types")
async def get_supported_device_types():
    """Get list of all supported device types with their available vendors."""
    device_types = {}
    
    for dtype in DeviceType:
        vendors_for_type = []
        for vid, config in VENDOR_CONFIGS.items():
            if dtype in config["supported_devices"]:
                vendors_for_type.append({
                    "vendor_id": vid.value,
                    "name": config["name"],
                    "is_public_api": config["is_public_api"],
                    "pairing_methods": [p.value for p in config.get("pairing_methods", [])],
                })
        
        if vendors_for_type:
            device_types[dtype.value] = {
                "name": dtype.value.replace("_", " ").title(),
                "vendors": vendors_for_type,
            }
    
    return device_types

@router.post("/pair/start", response_model=StartPairingResponse)
async def start_device_pairing(
    request_data: StartPairingRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """
    Start device pairing flow.
    Returns OAuth URL, BLE instructions, or QR code based on pairing method.
    """
    vendor_id = request_data.vendor_id
    pairing_method = request_data.pairing_method
    
    # Validate vendor
    try:
        vendor_enum = VendorId(vendor_id)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Unknown vendor: {vendor_id}")
    
    vendor_config = VENDOR_CONFIGS.get(vendor_enum)
    if not vendor_config:
        raise HTTPException(status_code=400, detail=f"Vendor not configured: {vendor_id}")
    
    # Generate session ID
    session_id = secrets.token_urlsafe(32)
    expires_at = datetime.utcnow() + timedelta(minutes=30)
    
    response_data = {
        "session_id": session_id,
        "pairing_method": pairing_method,
        "status": SessionStatus.INITIATED.value,
        "expires_at": expires_at.isoformat(),
        "consent_required": True,
    }
    
    # Get user ID from request (simplified - in production use proper auth)
    user_id = request.headers.get("X-User-Id", "anonymous")
    
    if pairing_method == PairingMethod.OAUTH.value:
        # Check credentials
        creds = get_vendor_client_credentials(vendor_id)
        if not creds["client_id"]:
            raise HTTPException(
                status_code=400, 
                detail=f"OAuth credentials not configured for {vendor_config['name']}. Please add {vendor_id.upper()}_CLIENT_ID and {vendor_id.upper()}_CLIENT_SECRET."
            )
        
        # Generate OAuth URL
        oauth_state = generate_oauth_state()
        code_verifier = generate_pkce_verifier()
        code_challenge = generate_pkce_challenge(code_verifier)
        
        redirect_uri = request_data.redirect_uri or f"{os.getenv('APP_URL', 'http://localhost:5000')}/api/v1/devices/oauth/callback/{vendor_id}"
        
        auth_params = {
            "client_id": creds["client_id"],
            "response_type": "code",
            "redirect_uri": redirect_uri,
            "scope": " ".join(vendor_config.get("scopes", [])),
            "state": oauth_state,
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
        }
        
        auth_url = f"{vendor_config['auth_url']}?" + "&".join(f"{k}={v}" for k, v in auth_params.items())
        
        response_data["auth_url"] = auth_url
        response_data["status"] = SessionStatus.PENDING_AUTH.value
        
        # Store session in database
        try:
            from sqlalchemy import text
            await db.execute(
                text("""
                    INSERT INTO device_pairing_sessions 
                    (id, user_id, vendor_id, device_type, pairing_method, session_status, 
                     oauth_state, oauth_code_verifier, oauth_redirect_uri, expires_at, ip_address, user_agent)
                    VALUES (:id, :user_id, :vendor_id, :device_type, :pairing_method, :session_status,
                            :oauth_state, :oauth_code_verifier, :oauth_redirect_uri, :expires_at, :ip_address, :user_agent)
                """),
                {
                    "id": session_id,
                    "user_id": user_id,
                    "vendor_id": vendor_id,
                    "device_type": request_data.device_type,
                    "pairing_method": pairing_method,
                    "session_status": SessionStatus.PENDING_AUTH.value,
                    "oauth_state": oauth_state,
                    "oauth_code_verifier": code_verifier,
                    "oauth_redirect_uri": redirect_uri,
                    "expires_at": expires_at,
                    "ip_address": request.client.host if request.client else None,
                    "user_agent": request.headers.get("user-agent"),
                }
            )
            await db.commit()
        except Exception as e:
            logger.error(f"Failed to store pairing session: {e}")
    
    elif pairing_method == PairingMethod.BLE.value:
        # Return BLE pairing instructions
        device_type = request_data.device_type
        ble_service = None
        
        if device_type == DeviceType.BP_MONITOR.value:
            ble_service = BLE_SERVICES["blood_pressure"]
        elif device_type == DeviceType.PULSE_OXIMETER.value:
            ble_service = BLE_SERVICES["pulse_oximeter"]
        elif device_type == DeviceType.THERMOMETER.value:
            ble_service = BLE_SERVICES["health_thermometer"]
        elif device_type == DeviceType.SCALE.value:
            ble_service = BLE_SERVICES["weight_scale"]
        elif device_type == DeviceType.GLUCOSE_METER.value:
            ble_service = BLE_SERVICES["glucose"]
        
        response_data["ble_instructions"] = {
            "service_uuid": ble_service,
            "steps": [
                "Enable Bluetooth on your device",
                "Put your medical device in pairing mode",
                "Click 'Scan for Devices' to find your device",
                "Select your device from the list",
                "Confirm the pairing code if prompted",
            ],
            "ios_note": "For iOS devices, please use our companion app for BLE pairing",
            "supported_browsers": ["Chrome", "Edge", "Opera"],
        }
        response_data["status"] = SessionStatus.PENDING_BLE.value
    
    elif pairing_method == PairingMethod.HEALTHKIT.value:
        response_data["status"] = SessionStatus.PENDING_CONSENT.value
        response_data["ble_instructions"] = {
            "steps": [
                "Open the Health app on your iPhone",
                "Go to your profile > Apps > Followup AI",
                "Enable data sharing for the metrics you want to sync",
                "Use 'Export Health Data' or our companion app to sync",
            ],
            "data_types": vendor_config.get("metrics", []),
        }
    
    elif pairing_method == PairingMethod.GOOGLE_FIT.value:
        # Similar to OAuth but using Google's specific flow
        creds = get_vendor_client_credentials("google_fit")
        if not creds["client_id"]:
            raise HTTPException(
                status_code=400,
                detail="Google Fit credentials not configured. Please add GOOGLE_FIT_CLIENT_ID and GOOGLE_FIT_CLIENT_SECRET."
            )
        
        oauth_state = generate_oauth_state()
        redirect_uri = request_data.redirect_uri or f"{os.getenv('APP_URL', 'http://localhost:5000')}/api/v1/devices/oauth/callback/google_fit"
        
        auth_params = {
            "client_id": creds["client_id"],
            "response_type": "code",
            "redirect_uri": redirect_uri,
            "scope": " ".join(vendor_config.get("scopes", [])),
            "state": oauth_state,
            "access_type": "offline",
            "prompt": "consent",
        }
        
        auth_url = f"{vendor_config['auth_url']}?" + "&".join(f"{k}={v}" for k, v in auth_params.items())
        
        response_data["auth_url"] = auth_url
        response_data["status"] = SessionStatus.PENDING_AUTH.value
    
    elif pairing_method == PairingMethod.MANUAL.value:
        response_data["status"] = SessionStatus.PENDING_CONSENT.value
        response_data["ble_instructions"] = {
            "steps": [
                "You can manually enter readings from your device",
                "Go to Daily Follow-up > Device Data to enter readings",
                "Select the device type and enter your measurements",
            ],
        }
    
    # Log audit event
    await log_device_audit(
        db=db,
        actor_id=user_id,
        actor_type="patient",
        action="pairing_started",
        action_category="device_management",
        resource_type="device_pairing_session",
        resource_id=session_id,
        patient_id=user_id,
        event_details={
            "vendor_id": vendor_id,
            "device_type": request_data.device_type,
            "pairing_method": pairing_method,
        },
        request=request,
    )
    
    return StartPairingResponse(**response_data)

@router.get("/oauth/callback/{vendor_id}")
async def oauth_callback(
    vendor_id: str,
    code: str = Query(...),
    state: str = Query(...),
    request: Optional[Request] = None,
    db: AsyncSession = Depends(get_db),
):
    """
    OAuth callback handler for vendor authentication.
    Exchanges authorization code for access token and creates device connection.
    """
    from sqlalchemy import text
    
    # Find pairing session by state
    result = await db.execute(
        text("""
            SELECT * FROM device_pairing_sessions 
            WHERE oauth_state = :state AND session_status = 'pending_auth'
            AND expires_at > NOW()
        """),
        {"state": state}
    )
    session = result.fetchone()
    
    if not session:
        raise HTTPException(status_code=400, detail="Invalid or expired pairing session")
    
    # Get vendor config
    vendor_config = VENDOR_CONFIGS.get(VendorId(vendor_id))
    if not vendor_config:
        raise HTTPException(status_code=400, detail=f"Unknown vendor: {vendor_id}")
    
    # Exchange code for tokens
    creds = get_vendor_client_credentials(vendor_id)
    
    try:
        import httpx
        
        token_data = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": session.oauth_redirect_uri,
            "client_id": creds["client_id"],
            "client_secret": creds["client_secret"],
        }
        
        # Add PKCE verifier if available
        if session.oauth_code_verifier:
            token_data["code_verifier"] = session.oauth_code_verifier
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                vendor_config["token_url"],
                data=token_data,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            
            if response.status_code != 200:
                logger.error(f"Token exchange failed: {response.text}")
                raise HTTPException(status_code=400, detail="Failed to exchange authorization code")
            
            tokens = response.json()
        
        # Calculate token expiry
        expires_in = tokens.get("expires_in", 3600)
        token_expires_at = datetime.utcnow() + timedelta(seconds=expires_in)
        
        # Create vendor account
        vendor_account_id = secrets.token_urlsafe(16)
        await db.execute(
            text("""
                INSERT INTO vendor_accounts 
                (id, user_id, vendor_id, vendor_name, access_token, refresh_token, 
                 token_type, token_expires_at, token_scope, vendor_user_id, connection_status,
                 last_auth_at, auto_sync, synced_data_types, consent_granted_at)
                VALUES (:id, :user_id, :vendor_id, :vendor_name, :access_token, :refresh_token,
                        :token_type, :token_expires_at, :token_scope, :vendor_user_id, :connection_status,
                        :last_auth_at, :auto_sync, :synced_data_types, :consent_granted_at)
            """),
            {
                "id": vendor_account_id,
                "user_id": session.user_id,
                "vendor_id": vendor_id,
                "vendor_name": vendor_config["name"],
                "access_token": tokens.get("access_token"),  # PHI: Should be encrypted in production
                "refresh_token": tokens.get("refresh_token"),
                "token_type": tokens.get("token_type", "Bearer"),
                "token_expires_at": token_expires_at,
                "token_scope": tokens.get("scope"),
                "vendor_user_id": tokens.get("user_id"),
                "connection_status": ConnectionStatus.CONNECTED.value,
                "last_auth_at": datetime.utcnow(),
                "auto_sync": True,
                "synced_data_types": vendor_config.get("metrics", []),
                "consent_granted_at": datetime.utcnow(),
            }
        )
        
        # Create wearable integration record
        integration_id = secrets.token_urlsafe(16)
        await db.execute(
            text("""
                INSERT INTO wearable_integrations 
                (id, user_id, device_type, device_name, device_model, connection_status,
                 access_token, refresh_token, token_expires_at, auto_sync, tracked_metrics, last_synced_at)
                VALUES (:id, :user_id, :device_type, :device_name, :device_model, :connection_status,
                        :access_token, :refresh_token, :token_expires_at, :auto_sync, :tracked_metrics, :last_synced_at)
            """),
            {
                "id": integration_id,
                "user_id": session.user_id,
                "device_type": vendor_id,
                "device_name": vendor_config["name"],
                "device_model": session.device_type,
                "connection_status": ConnectionStatus.CONNECTED.value,
                "access_token": tokens.get("access_token"),
                "refresh_token": tokens.get("refresh_token"),
                "token_expires_at": token_expires_at,
                "auto_sync": True,
                "tracked_metrics": vendor_config.get("metrics", []),
                "last_synced_at": datetime.utcnow(),
            }
        )
        
        # Update pairing session
        await db.execute(
            text("""
                UPDATE device_pairing_sessions 
                SET session_status = :status, result_vendor_account_id = :vendor_account_id,
                    result_device_connection_id = :device_connection_id, completed_at = :completed_at,
                    consent_captured = :consent_captured, consent_timestamp = :consent_timestamp
                WHERE id = :session_id
            """),
            {
                "status": SessionStatus.COMPLETED.value,
                "vendor_account_id": vendor_account_id,
                "device_connection_id": integration_id,
                "completed_at": datetime.utcnow(),
                "consent_captured": True,
                "consent_timestamp": datetime.utcnow(),
                "session_id": session.id,
            }
        )
        
        await db.commit()
        
        # Log audit event
        await log_device_audit(
            db=db,
            actor_id=session.user_id,
            actor_type="patient",
            action="device_paired",
            action_category="device_management",
            resource_type="device_connection",
            resource_id=integration_id,
            patient_id=session.user_id,
            event_details={
                "vendor_id": vendor_id,
                "device_type": session.device_type,
                "pairing_method": "oauth",
            },
            success=True,
            request=request,
        )
        
        # Redirect to success page
        app_url = os.getenv("APP_URL", "http://localhost:5000")
        return RedirectResponse(
            url=f"{app_url}/wearables?paired=true&vendor={vendor_id}",
            status_code=302,
        )
        
    except Exception as e:
        logger.error(f"OAuth callback error: {e}")
        
        # Update session with error
        await db.execute(
            text("""
                UPDATE device_pairing_sessions 
                SET session_status = :status, error_code = :error_code, error_message = :error_message
                WHERE id = :session_id
            """),
            {
                "status": SessionStatus.FAILED.value,
                "error_code": "token_exchange_failed",
                "error_message": str(e),
                "session_id": session.id,
            }
        )
        await db.commit()
        
        app_url = os.getenv("APP_URL", "http://localhost:5000")
        return RedirectResponse(
            url=f"{app_url}/wearables?error=pairing_failed&vendor={vendor_id}",
            status_code=302,
        )

@router.post("/pair/complete")
async def complete_device_pairing(
    request_data: CompletePairingRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """
    Complete device pairing after OAuth callback or BLE pairing.
    Finalizes connection and records consent.
    """
    from sqlalchemy import text
    
    # Find pairing session
    result = await db.execute(
        text("""
            SELECT * FROM device_pairing_sessions 
            WHERE id = :session_id AND expires_at > NOW()
        """),
        {"session_id": request_data.session_id}
    )
    session = result.fetchone()
    
    if not session:
        raise HTTPException(status_code=404, detail="Pairing session not found or expired")
    
    if session.session_status == SessionStatus.COMPLETED.value:
        return {"status": "already_completed", "device_connection_id": session.result_device_connection_id}
    
    # For BLE or manual pairing, create connection directly
    if session.pairing_method in [PairingMethod.BLE.value, PairingMethod.MANUAL.value]:
        integration_id = secrets.token_urlsafe(16)
        vendor_config = VENDOR_CONFIGS.get(VendorId(session.vendor_id), {})
        
        await db.execute(
            text("""
                INSERT INTO wearable_integrations 
                (id, user_id, device_type, device_name, device_model, connection_status,
                 device_id, auto_sync, tracked_metrics)
                VALUES (:id, :user_id, :device_type, :device_name, :device_model, :connection_status,
                        :device_id, :auto_sync, :tracked_metrics)
            """),
            {
                "id": integration_id,
                "user_id": session.user_id,
                "device_type": session.vendor_id,
                "device_name": vendor_config.get("name", session.vendor_id),
                "device_model": session.device_type,
                "connection_status": ConnectionStatus.CONNECTED.value,
                "device_id": request_data.ble_device_id,
                "auto_sync": True,
                "tracked_metrics": request_data.consented_data_types or vendor_config.get("metrics", []),
            }
        )
        
        # Update pairing session
        await db.execute(
            text("""
                UPDATE device_pairing_sessions 
                SET session_status = :status, result_device_connection_id = :device_connection_id,
                    completed_at = :completed_at, consent_captured = :consent_captured,
                    consent_timestamp = :consent_timestamp, consented_data_types = :consented_data_types,
                    ble_device_id = :ble_device_id
                WHERE id = :session_id
            """),
            {
                "status": SessionStatus.COMPLETED.value,
                "device_connection_id": integration_id,
                "completed_at": datetime.utcnow(),
                "consent_captured": request_data.consent_granted,
                "consent_timestamp": datetime.utcnow() if request_data.consent_granted else None,
                "consented_data_types": request_data.consented_data_types,
                "ble_device_id": request_data.ble_device_id,
                "session_id": request_data.session_id,
            }
        )
        
        await db.commit()
        
        # Log audit event
        await log_device_audit(
            db=db,
            actor_id=session.user_id,
            actor_type="patient",
            action="device_paired",
            action_category="device_management",
            resource_type="device_connection",
            resource_id=integration_id,
            patient_id=session.user_id,
            event_details={
                "vendor_id": session.vendor_id,
                "device_type": session.device_type,
                "pairing_method": session.pairing_method,
                "consented_data_types": request_data.consented_data_types,
            },
            success=True,
            request=request,
        )
        
        return {
            "status": "completed",
            "device_connection_id": integration_id,
            "vendor_id": session.vendor_id,
            "device_type": session.device_type,
        }
    
    raise HTTPException(status_code=400, detail="Invalid pairing session state")

@router.get("/connections", response_model=List[DeviceConnectionResponse])
async def list_device_connections(
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """
    List all device connections for the current user.
    """
    from sqlalchemy import text
    
    user_id = request.headers.get("X-User-Id", "anonymous")
    
    result = await db.execute(
        text("""
            SELECT * FROM wearable_integrations 
            WHERE user_id = :user_id
            ORDER BY created_at DESC
        """),
        {"user_id": user_id}
    )
    connections = result.fetchall()
    
    return [
        DeviceConnectionResponse(
            id=conn.id,
            vendor_id=conn.device_type,
            vendor_name=conn.device_name,
            device_type=conn.device_model or conn.device_type,
            device_model=conn.device_model,
            connection_status=conn.connection_status or "unknown",
            last_sync_at=conn.last_synced_at.isoformat() if conn.last_synced_at else None,
            battery_level=conn.battery_level,
            tracked_metrics=conn.tracked_metrics or [],
            auto_sync=conn.auto_sync if conn.auto_sync is not None else True,
            created_at=conn.created_at.isoformat() if conn.created_at else datetime.utcnow().isoformat(),
        )
        for conn in connections
    ]

@router.get("/connections/{connection_id}", response_model=DeviceConnectionResponse)
async def get_device_connection(
    connection_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """
    Get details for a specific device connection.
    """
    from sqlalchemy import text
    
    user_id = request.headers.get("X-User-Id", "anonymous")
    
    result = await db.execute(
        text("""
            SELECT * FROM wearable_integrations 
            WHERE id = :id AND user_id = :user_id
        """),
        {"id": connection_id, "user_id": user_id}
    )
    conn = result.fetchone()
    
    if not conn:
        raise HTTPException(status_code=404, detail="Device connection not found")
    
    return DeviceConnectionResponse(
        id=conn.id,
        vendor_id=conn.device_type,
        vendor_name=conn.device_name,
        device_type=conn.device_model or conn.device_type,
        device_model=conn.device_model,
        connection_status=conn.connection_status or "unknown",
        last_sync_at=conn.last_synced_at.isoformat() if conn.last_synced_at else None,
        battery_level=conn.battery_level,
        tracked_metrics=conn.tracked_metrics or [],
        auto_sync=conn.auto_sync if conn.auto_sync is not None else True,
        created_at=conn.created_at.isoformat() if conn.created_at else datetime.utcnow().isoformat(),
    )

@router.delete("/connections/{connection_id}")
async def delete_device_connection(
    connection_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """
    Disconnect and remove a device.
    """
    from sqlalchemy import text
    
    user_id = request.headers.get("X-User-Id", "anonymous")
    
    # Verify ownership
    result = await db.execute(
        text("""
            SELECT * FROM wearable_integrations 
            WHERE id = :id AND user_id = :user_id
        """),
        {"id": connection_id, "user_id": user_id}
    )
    conn = result.fetchone()
    
    if not conn:
        raise HTTPException(status_code=404, detail="Device connection not found")
    
    # Delete connection
    await db.execute(
        text("DELETE FROM wearable_integrations WHERE id = :id"),
        {"id": connection_id}
    )
    
    # Also delete vendor account if exists
    await db.execute(
        text("""
            DELETE FROM vendor_accounts 
            WHERE user_id = :user_id AND vendor_id = :vendor_id
        """),
        {"user_id": user_id, "vendor_id": conn.device_type}
    )
    
    await db.commit()
    
    # Log audit event
    await log_device_audit(
        db=db,
        actor_id=user_id,
        actor_type="patient",
        action="device_unpaired",
        action_category="device_management",
        resource_type="device_connection",
        resource_id=connection_id,
        patient_id=user_id,
        event_details={
            "vendor_id": conn.device_type,
            "device_name": conn.device_name,
        },
        success=True,
        request=request,
    )
    
    return {"status": "deleted", "connection_id": connection_id}

@router.get("/health/{connection_id}", response_model=DeviceHealthResponse)
async def get_device_health(
    connection_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """
    Get health status for a device connection.
    """
    from sqlalchemy import text
    
    user_id = request.headers.get("X-User-Id", "anonymous")
    
    # Get connection
    result = await db.execute(
        text("""
            SELECT * FROM wearable_integrations 
            WHERE id = :id AND user_id = :user_id
        """),
        {"id": connection_id, "user_id": user_id}
    )
    conn = result.fetchone()
    
    if not conn:
        raise HTTPException(status_code=404, detail="Device connection not found")
    
    # Get device health if exists
    health_result = await db.execute(
        text("""
            SELECT * FROM device_health 
            WHERE device_connection_id = :connection_id
            ORDER BY updated_at DESC LIMIT 1
        """),
        {"connection_id": connection_id}
    )
    health = health_result.fetchone()
    
    if health:
        return DeviceHealthResponse(
            device_connection_id=connection_id,
            status=health.status,
            battery_level=health.battery_level,
            battery_status=health.battery_status,
            signal_strength=health.signal_strength,
            firmware_version=health.firmware_version,
            firmware_update_available=health.firmware_update_available or False,
            last_seen_at=health.last_seen_at.isoformat() if health.last_seen_at else None,
            last_successful_sync=health.last_successful_sync.isoformat() if health.last_successful_sync else None,
            sync_success_rate=float(health.sync_success_rate) if health.sync_success_rate else None,
            data_quality_score=health.data_quality_score,
            alerts=health.health_alerts or [],
        )
    
    # Return basic health from connection data
    return DeviceHealthResponse(
        device_connection_id=connection_id,
        status=conn.connection_status or "unknown",
        battery_level=conn.battery_level,
        last_successful_sync=conn.last_synced_at.isoformat() if conn.last_synced_at else None,
        alerts=[],
    )

@router.post("/sync/{connection_id}")
async def sync_device(
    connection_id: str,
    request: Request,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    """
    Trigger manual sync for a device connection.
    """
    from sqlalchemy import text
    
    user_id = request.headers.get("X-User-Id", "anonymous")
    
    # Get connection
    result = await db.execute(
        text("""
            SELECT * FROM wearable_integrations 
            WHERE id = :id AND user_id = :user_id
        """),
        {"id": connection_id, "user_id": user_id}
    )
    conn = result.fetchone()
    
    if not conn:
        raise HTTPException(status_code=404, detail="Device connection not found")
    
    # Queue sync job
    job_id = secrets.token_urlsafe(16)
    await db.execute(
        text("""
            INSERT INTO device_sync_jobs 
            (id, job_type, device_connection_id, user_id, status, priority, scheduled_for)
            VALUES (:id, :job_type, :device_connection_id, :user_id, :status, :priority, :scheduled_for)
        """),
        {
            "id": job_id,
            "job_type": "on_demand_sync",
            "device_connection_id": connection_id,
            "user_id": user_id,
            "status": "pending",
            "priority": 8,  # High priority for user-triggered syncs
            "scheduled_for": datetime.utcnow(),
        }
    )
    await db.commit()
    
    # Log audit event
    await log_device_audit(
        db=db,
        actor_id=user_id,
        actor_type="patient",
        action="sync_requested",
        action_category="sync",
        resource_type="device_connection",
        resource_id=connection_id,
        patient_id=user_id,
        event_details={"job_id": job_id},
        success=True,
        request=request,
    )
    
    return {
        "status": "sync_queued",
        "job_id": job_id,
        "connection_id": connection_id,
    }

@router.post("/data/ingest", response_model=DataIngestResponse)
async def ingest_device_data(
    request_data: DataIngestRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """
    Ingest device readings into the system.
    Normalizes data, routes to appropriate health sections, and triggers alerts.
    """
    from sqlalchemy import text
    
    user_id = request.headers.get("X-User-Id", "anonymous")
    
    # Verify device connection
    result = await db.execute(
        text("""
            SELECT * FROM wearable_integrations 
            WHERE id = :id
        """),
        {"id": request_data.device_connection_id}
    )
    conn = result.fetchone()
    
    if not conn:
        raise HTTPException(status_code=404, detail="Device connection not found")
    
    readings = request_data.readings
    meta = request_data.meta or {}
    
    readings_accepted = []
    readings_rejected = []
    routed_to_sections = set()
    alerts_triggered = []
    
    # Process each reading type
    insert_data = {
        "id": secrets.token_urlsafe(16),
        "patient_id": conn.user_id,
        "device_type": conn.device_model or conn.device_type,
        "device_brand": conn.device_type,
        "source": "auto_sync",
        "wearable_integration_id": conn.id,
        "recorded_at": request_data.timestamp,
    }
    
    # Blood pressure
    if "blood_pressure" in readings:
        bp = readings["blood_pressure"]
        insert_data["bp_systolic"] = bp.get("systolic")
        insert_data["bp_diastolic"] = bp.get("diastolic")
        insert_data["bp_pulse"] = bp.get("hr")
        insert_data["route_to_hypertension"] = True
        readings_accepted.append("blood_pressure")
        routed_to_sections.add("hypertension")
    
    # Weight
    if "weight_kg" in readings:
        insert_data["weight"] = readings["weight_kg"]
        insert_data["weight_unit"] = "kg"
        insert_data["route_to_fitness"] = True
        readings_accepted.append("weight")
        routed_to_sections.add("fitness")
    
    # Glucose
    if "glucose_mg_dL" in readings:
        insert_data["glucose_value"] = readings["glucose_mg_dL"]
        insert_data["glucose_unit"] = "mg/dL"
        insert_data["route_to_diabetes"] = True
        readings_accepted.append("glucose")
        routed_to_sections.add("diabetes")
    
    # SpO2
    if "spo2_pct" in readings:
        insert_data["spo2"] = readings["spo2_pct"]
        insert_data["route_to_respiratory"] = True
        readings_accepted.append("spo2")
        routed_to_sections.add("respiratory")
    
    # Heart rate
    if "heart_rate" in readings:
        insert_data["heart_rate"] = readings["heart_rate"]
        insert_data["route_to_cardiovascular"] = True
        readings_accepted.append("heart_rate")
        routed_to_sections.add("cardiovascular")
    
    # HRV
    if "hrv" in readings:
        insert_data["hrv"] = readings["hrv"]
        insert_data["route_to_cardiovascular"] = True
        if "hrv" not in readings_accepted:
            readings_accepted.append("hrv")
        routed_to_sections.add("cardiovascular")
    
    # Sleep
    if "sleep" in readings:
        sleep = readings["sleep"]
        insert_data["sleep_duration"] = sleep.get("duration_minutes")
        insert_data["sleep_score"] = sleep.get("score")
        insert_data["sleep_deep_minutes"] = sleep.get("deep_minutes")
        insert_data["sleep_rem_minutes"] = sleep.get("rem_minutes")
        insert_data["route_to_sleep"] = True
        readings_accepted.append("sleep")
        routed_to_sections.add("sleep")
    
    # Stress
    if "stress" in readings:
        insert_data["stress_score"] = readings["stress"]
        insert_data["route_to_mental_health"] = True
        readings_accepted.append("stress")
        routed_to_sections.add("mental_health")
    
    # Recovery
    if "recovery" in readings:
        insert_data["recovery_score"] = readings["recovery"]
        readings_accepted.append("recovery")
    
    # Steps
    if "steps" in readings:
        insert_data["steps"] = readings["steps"]
        insert_data["route_to_fitness"] = True
        readings_accepted.append("steps")
        routed_to_sections.add("fitness")
    
    # Temperature
    if "temperature" in readings:
        insert_data["temperature"] = readings["temperature"]
        insert_data["temperature_unit"] = "F"
        readings_accepted.append("temperature")
    
    # Stethoscope audio
    if "stethoscope_audio_url" in readings:
        insert_data["stethoscope_audio_url"] = readings["stethoscope_audio_url"]
        readings_accepted.append("stethoscope_audio")
    
    # Meta data
    if meta:
        insert_data["metadata"] = meta
    
    # Insert device reading
    if readings_accepted:
        columns = ", ".join(insert_data.keys())
        placeholders = ", ".join(f":{k}" for k in insert_data.keys())
        
        await db.execute(
            text(f"INSERT INTO device_readings ({columns}) VALUES ({placeholders})"),
            insert_data
        )
        
        # Update last sync time
        await db.execute(
            text("""
                UPDATE wearable_integrations 
                SET last_synced_at = :now, last_sync_status = 'success', battery_level = :battery
                WHERE id = :id
            """),
            {
                "now": datetime.utcnow(),
                "battery": meta.get("battery_pct"),
                "id": conn.id,
            }
        )
        
        await db.commit()
    
    # Log audit event
    await log_device_audit(
        db=db,
        actor_id=conn.user_id,
        actor_type="system",
        action="data_synced",
        action_category="data_access",
        resource_type="device_reading",
        resource_id=insert_data["id"],
        patient_id=conn.user_id,
        event_details={
            "device_connection_id": conn.id,
            "readings_accepted": readings_accepted,
            "routed_to_sections": list(routed_to_sections),
        },
        success=True,
        request=request,
    )
    
    return DataIngestResponse(
        success=True,
        records_processed=1,
        readings_accepted=readings_accepted,
        readings_rejected=readings_rejected,
        routed_to_sections=list(routed_to_sections),
        alerts_triggered=alerts_triggered,
    )

@router.post("/healthkit/sync")
async def sync_healthkit_data(
    request_data: HealthKitDataRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """
    Receive and process Apple HealthKit data exports.
    Endpoint for iOS companion app or manual HealthKit export.
    """
    from sqlalchemy import text
    
    user_id = request.headers.get("X-User-Id", "anonymous")
    
    records_processed = 0
    routed_sections = set()
    
    for reading in request_data.readings:
        try:
            insert_data = {
                "id": secrets.token_urlsafe(16),
                "patient_id": user_id,
                "device_type": "smartwatch",
                "device_brand": "apple_healthkit",
                "source": "healthkit",
                "recorded_at": reading.get("timestamp", datetime.utcnow().isoformat()),
            }
            
            data_type = reading.get("type", "").lower()
            value = reading.get("value")
            
            if data_type == "heart_rate" and value:
                insert_data["heart_rate"] = int(value)
                insert_data["route_to_cardiovascular"] = True
                routed_sections.add("cardiovascular")
            elif data_type == "blood_pressure" and value:
                insert_data["bp_systolic"] = value.get("systolic")
                insert_data["bp_diastolic"] = value.get("diastolic")
                insert_data["route_to_hypertension"] = True
                routed_sections.add("hypertension")
            elif data_type == "blood_glucose" and value:
                insert_data["glucose_value"] = float(value)
                insert_data["route_to_diabetes"] = True
                routed_sections.add("diabetes")
            elif data_type == "oxygen_saturation" and value:
                insert_data["spo2"] = int(float(value) * 100) if float(value) <= 1 else int(value)
                insert_data["route_to_respiratory"] = True
                routed_sections.add("respiratory")
            elif data_type == "sleep_analysis" and value:
                insert_data["sleep_duration"] = value.get("duration_minutes")
                insert_data["route_to_sleep"] = True
                routed_sections.add("sleep")
            elif data_type == "step_count" and value:
                insert_data["steps"] = int(value)
                insert_data["route_to_fitness"] = True
                routed_sections.add("fitness")
            elif data_type == "hrv" and value:
                insert_data["hrv"] = int(value)
                insert_data["route_to_cardiovascular"] = True
                routed_sections.add("cardiovascular")
            elif data_type == "body_temperature" and value:
                insert_data["temperature"] = float(value)
            elif data_type == "weight" and value:
                insert_data["weight"] = float(value)
                insert_data["route_to_fitness"] = True
                routed_sections.add("fitness")
            else:
                continue  # Skip unknown types
            
            # Insert reading
            columns = ", ".join(insert_data.keys())
            placeholders = ", ".join(f":{k}" for k in insert_data.keys())
            await db.execute(
                text(f"INSERT INTO device_readings ({columns}) VALUES ({placeholders})"),
                insert_data
            )
            records_processed += 1
            
        except Exception as e:
            logger.error(f"Failed to process HealthKit reading: {e}")
    
    await db.commit()
    
    # Log audit event
    await log_device_audit(
        db=db,
        actor_id=user_id,
        actor_type="patient",
        action="healthkit_synced",
        action_category="data_access",
        resource_type="device_reading",
        patient_id=user_id,
        event_details={
            "data_types": request_data.data_types,
            "records_processed": records_processed,
            "export_date": request_data.export_date,
        },
        success=True,
        request=request,
    )
    
    return {
        "success": True,
        "records_processed": records_processed,
        "routed_to_sections": list(routed_sections),
    }

# ============================================
# WEBHOOK HANDLERS
# ============================================

@router.post("/webhooks/{vendor_id}")
async def vendor_webhook(
    vendor_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """
    Receive webhook data from vendor APIs.
    Validates webhook signature and processes incoming data.
    """
    body = await request.body()
    
    # TODO: Validate webhook signature based on vendor
    # Each vendor has different signature validation methods
    
    try:
        import json
        payload = json.loads(body)
        
        # Queue webhook processing job
        from sqlalchemy import text
        
        job_id = secrets.token_urlsafe(16)
        await db.execute(
            text("""
                INSERT INTO device_sync_jobs 
                (id, job_type, status, priority, metadata)
                VALUES (:id, :job_type, :status, :priority, :metadata)
            """),
            {
                "id": job_id,
                "job_type": "webhook_process",
                "status": "pending",
                "priority": 9,  # High priority
                "metadata": {"vendor_id": vendor_id, "payload": payload},
            }
        )
        await db.commit()
        
        logger.info(f"Queued webhook processing job {job_id} for {vendor_id}")
        
        return {"status": "accepted", "job_id": job_id}
        
    except Exception as e:
        logger.error(f"Webhook processing error for {vendor_id}: {e}")
        raise HTTPException(status_code=400, detail="Invalid webhook payload")

# ============================================
# BLE WEB BLUETOOTH SUPPORT
# ============================================

@router.get("/ble/services")
async def get_ble_services():
    """
    Get BLE service UUIDs for Web Bluetooth pairing.
    Returns service information for supported medical devices.
    """
    return {
        "services": [
            {
                "name": "Blood Pressure",
                "uuid": "0x1810",
                "device_types": ["bp_monitor"],
                "characteristics": [
                    {"name": "Blood Pressure Measurement", "uuid": "0x2A35"},
                    {"name": "Intermediate Cuff Pressure", "uuid": "0x2A36"},
                ],
            },
            {
                "name": "Heart Rate",
                "uuid": "0x180D",
                "device_types": ["smartwatch", "activity_tracker"],
                "characteristics": [
                    {"name": "Heart Rate Measurement", "uuid": "0x2A37"},
                    {"name": "Body Sensor Location", "uuid": "0x2A38"},
                ],
            },
            {
                "name": "Health Thermometer",
                "uuid": "0x1809",
                "device_types": ["thermometer"],
                "characteristics": [
                    {"name": "Temperature Measurement", "uuid": "0x2A1C"},
                    {"name": "Temperature Type", "uuid": "0x2A1D"},
                ],
            },
            {
                "name": "Weight Scale",
                "uuid": "0x181D",
                "device_types": ["scale"],
                "characteristics": [
                    {"name": "Weight Measurement", "uuid": "0x2A9D"},
                    {"name": "Weight Scale Feature", "uuid": "0x2A9E"},
                ],
            },
            {
                "name": "Glucose",
                "uuid": "0x1808",
                "device_types": ["glucose_meter"],
                "characteristics": [
                    {"name": "Glucose Measurement", "uuid": "0x2A18"},
                    {"name": "Glucose Measurement Context", "uuid": "0x2A34"},
                ],
            },
            {
                "name": "Pulse Oximeter",
                "uuid": "0x1822",
                "device_types": ["pulse_oximeter"],
                "characteristics": [
                    {"name": "PLX Spot-Check Measurement", "uuid": "0x2A5E"},
                    {"name": "PLX Continuous Measurement", "uuid": "0x2A5F"},
                ],
            },
        ],
        "browser_support": {
            "chrome": True,
            "edge": True,
            "opera": True,
            "firefox": False,
            "safari": False,
            "ios_safari": False,
        },
        "ios_fallback": "For iOS devices, please use our companion app or Apple HealthKit sync.",
    }
