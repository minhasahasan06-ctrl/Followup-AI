"""
BLE Characteristic Parsers for Medical Devices
HIPAA-Compliant Parsing of Standard Bluetooth SIG Health Profiles

Supports parsing of:
- Blood Pressure Measurement (0x2A35)
- Heart Rate Measurement (0x2A37)
- Glucose Measurement (0x2A18)
- Temperature Measurement (0x2A1C)
- Weight Measurement (0x2A9D)
- Pulse Oximeter Spot-check (0x2A5E)
- Pulse Oximeter Continuous (0x2A5F)

Based on Bluetooth SIG GATT Specifications:
https://www.bluetooth.com/specifications/specs/gatt-specification-supplement/
"""

import logging
import struct
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import IntFlag, IntEnum

logger = logging.getLogger(__name__)


# ============================================
# DATA MODELS
# ============================================

@dataclass
class BLEReading:
    """Parsed BLE device reading"""
    data_type: str
    timestamp: datetime
    values: Dict[str, Any]
    unit: str
    flags: Dict[str, Optional[bool]]
    raw_data: bytes


# ============================================
# BLOOD PRESSURE MEASUREMENT (0x2A35)
# ============================================

class BPFlags(IntFlag):
    """Blood Pressure Measurement Flags byte"""
    UNIT_KPA = 0x01  # 0 = mmHg, 1 = kPa
    TIMESTAMP_PRESENT = 0x02
    PULSE_RATE_PRESENT = 0x04
    USER_ID_PRESENT = 0x08
    MEASUREMENT_STATUS_PRESENT = 0x10


def parse_blood_pressure(data: bytes) -> Optional[BLEReading]:
    """
    Parse Blood Pressure Measurement characteristic (UUID 0x2A35)
    
    Data format:
    - Flags (1 byte)
    - Systolic (SFLOAT - 2 bytes)
    - Diastolic (SFLOAT - 2 bytes)
    - MAP (SFLOAT - 2 bytes)
    - [Optional] Timestamp (7 bytes)
    - [Optional] Pulse Rate (SFLOAT - 2 bytes)
    - [Optional] User ID (1 byte)
    - [Optional] Measurement Status (2 bytes)
    """
    if len(data) < 7:
        logger.warning(f"Blood pressure data too short: {len(data)} bytes")
        return None
    
    try:
        flags = BPFlags(data[0])
        
        # Parse blood pressure values (SFLOAT format)
        systolic = parse_sfloat(data[1:3])
        diastolic = parse_sfloat(data[3:5])
        map_value = parse_sfloat(data[5:7])
        
        unit = "kPa" if flags & BPFlags.UNIT_KPA else "mmHg"
        
        # Convert kPa to mmHg if needed (1 kPa = 7.5 mmHg)
        if flags & BPFlags.UNIT_KPA:
            systolic *= 7.5
            diastolic *= 7.5
            map_value *= 7.5
            unit = "mmHg (converted)"
        
        values = {
            "systolic": round(systolic, 1),
            "diastolic": round(diastolic, 1),
            "map": round(map_value, 1),
        }
        
        offset = 7
        timestamp = datetime.utcnow()
        
        # Parse optional timestamp
        if flags & BPFlags.TIMESTAMP_PRESENT and len(data) >= offset + 7:
            timestamp = parse_datetime(data[offset:offset + 7])
            offset += 7
        
        # Parse optional pulse rate
        if flags & BPFlags.PULSE_RATE_PRESENT and len(data) >= offset + 2:
            values["pulse_rate"] = round(parse_sfloat(data[offset:offset + 2]), 1)
            offset += 2
        
        # Parse optional user ID
        if flags & BPFlags.USER_ID_PRESENT and len(data) >= offset + 1:
            values["user_id"] = data[offset]
            offset += 1
        
        return BLEReading(
            data_type="blood_pressure",
            timestamp=timestamp,
            values=values,
            unit=unit,
            flags={
                "unit_kpa": bool(flags & BPFlags.UNIT_KPA),
                "has_timestamp": bool(flags & BPFlags.TIMESTAMP_PRESENT),
                "has_pulse_rate": bool(flags & BPFlags.PULSE_RATE_PRESENT),
                "has_user_id": bool(flags & BPFlags.USER_ID_PRESENT),
            },
            raw_data=data,
        )
        
    except Exception as e:
        logger.error(f"Failed to parse blood pressure: {e}")
        return None


# ============================================
# HEART RATE MEASUREMENT (0x2A37)
# ============================================

class HRFlags(IntFlag):
    """Heart Rate Measurement Flags byte"""
    VALUE_FORMAT_UINT16 = 0x01  # 0 = UINT8, 1 = UINT16
    SENSOR_CONTACT_STATUS = 0x02
    SENSOR_CONTACT_SUPPORTED = 0x04
    ENERGY_EXPENDED_PRESENT = 0x08
    RR_INTERVAL_PRESENT = 0x10


def parse_heart_rate(data: bytes) -> Optional[BLEReading]:
    """
    Parse Heart Rate Measurement characteristic (UUID 0x2A37)
    
    Data format:
    - Flags (1 byte)
    - Heart Rate Value (1 or 2 bytes based on flags)
    - [Optional] Energy Expended (2 bytes)
    - [Optional] RR-Intervals (2 bytes each, can be multiple)
    """
    if len(data) < 2:
        logger.warning(f"Heart rate data too short: {len(data)} bytes")
        return None
    
    try:
        flags = HRFlags(data[0])
        offset = 1
        
        # Parse heart rate value
        if flags & HRFlags.VALUE_FORMAT_UINT16:
            hr_value = struct.unpack("<H", data[offset:offset + 2])[0]
            offset += 2
        else:
            hr_value = data[offset]
            offset += 1
        
        values = {"bpm": hr_value}
        
        # Check sensor contact
        sensor_contact = None
        if flags & HRFlags.SENSOR_CONTACT_SUPPORTED:
            sensor_contact = bool(flags & HRFlags.SENSOR_CONTACT_STATUS)
            values["sensor_contact"] = sensor_contact
        
        # Parse optional energy expended
        if flags & HRFlags.ENERGY_EXPENDED_PRESENT and len(data) >= offset + 2:
            values["energy_expended_kj"] = struct.unpack("<H", data[offset:offset + 2])[0]
            offset += 2
        
        # Parse optional RR intervals
        if flags & HRFlags.RR_INTERVAL_PRESENT:
            rr_intervals = []
            while offset + 2 <= len(data):
                rr_raw = struct.unpack("<H", data[offset:offset + 2])[0]
                # RR interval in 1/1024 seconds, convert to ms
                rr_ms = round(rr_raw * 1000 / 1024)
                rr_intervals.append(rr_ms)
                offset += 2
            
            if rr_intervals:
                values["rr_intervals_ms"] = rr_intervals
                # Calculate HRV (RMSSD) if we have multiple intervals
                if len(rr_intervals) >= 2:
                    diffs_squared = [(rr_intervals[i+1] - rr_intervals[i])**2 
                                     for i in range(len(rr_intervals) - 1)]
                    values["hrv_rmssd"] = round((sum(diffs_squared) / len(diffs_squared)) ** 0.5, 1)
        
        return BLEReading(
            data_type="heart_rate",
            timestamp=datetime.utcnow(),
            values=values,
            unit="bpm",
            flags={
                "format_uint16": bool(flags & HRFlags.VALUE_FORMAT_UINT16),
                "sensor_contact": sensor_contact,
                "has_energy": bool(flags & HRFlags.ENERGY_EXPENDED_PRESENT),
                "has_rr_intervals": bool(flags & HRFlags.RR_INTERVAL_PRESENT),
            },
            raw_data=data,
        )
        
    except Exception as e:
        logger.error(f"Failed to parse heart rate: {e}")
        return None


# ============================================
# GLUCOSE MEASUREMENT (0x2A18)
# ============================================

class GlucoseFlags(IntFlag):
    """Glucose Measurement Flags byte"""
    TIME_OFFSET_PRESENT = 0x01
    CONCENTRATION_TYPE_SAMPLE_LOCATION = 0x02
    CONCENTRATION_UNIT_MOL_L = 0x04  # 0 = kg/L, 1 = mol/L
    STATUS_ANNUNCIATION_PRESENT = 0x08
    CONTEXT_INFO_FOLLOWS = 0x10


class GlucoseSampleType(IntEnum):
    """Glucose sample type values"""
    RESERVED = 0
    CAPILLARY_WHOLE_BLOOD = 1
    CAPILLARY_PLASMA = 2
    VENOUS_WHOLE_BLOOD = 3
    VENOUS_PLASMA = 4
    ARTERIAL_WHOLE_BLOOD = 5
    ARTERIAL_PLASMA = 6
    UNDETERMINED_WHOLE_BLOOD = 7
    UNDETERMINED_PLASMA = 8
    INTERSTITIAL_FLUID = 9
    CONTROL_SOLUTION = 10


def parse_glucose(data: bytes) -> Optional[BLEReading]:
    """
    Parse Glucose Measurement characteristic (UUID 0x2A18)
    
    Data format:
    - Flags (1 byte)
    - Sequence Number (2 bytes)
    - Base Time (7 bytes)
    - [Optional] Time Offset (2 bytes signed)
    - [Optional] Glucose Concentration (SFLOAT - 2 bytes)
    - [Optional] Type-Sample Location (1 byte nibbles)
    - [Optional] Sensor Status Annunciation (2 bytes)
    """
    if len(data) < 10:
        logger.warning(f"Glucose data too short: {len(data)} bytes")
        return None
    
    try:
        flags = GlucoseFlags(data[0])
        sequence_number = struct.unpack("<H", data[1:3])[0]
        timestamp = parse_datetime(data[3:10])
        
        offset = 10
        values = {"sequence_number": sequence_number}
        
        # Parse optional time offset
        if flags & GlucoseFlags.TIME_OFFSET_PRESENT and len(data) >= offset + 2:
            time_offset_minutes = struct.unpack("<h", data[offset:offset + 2])[0]
            values["time_offset_minutes"] = time_offset_minutes
            offset += 2
        
        # Parse glucose concentration
        unit = "mg/dL"
        if len(data) >= offset + 2:
            glucose_raw = parse_sfloat(data[offset:offset + 2])
            
            if flags & GlucoseFlags.CONCENTRATION_UNIT_MOL_L:
                # Convert mol/L to mg/dL (1 mmol/L = 18 mg/dL)
                glucose_value = round(glucose_raw * 1000 * 18, 1)
                unit = "mg/dL (converted from mol/L)"
            else:
                # kg/L = g/dL * 1000 = mg/dL
                glucose_value = round(glucose_raw * 100000, 1)
            
            values["glucose"] = glucose_value
            offset += 2
        
        # Parse type and sample location
        if flags & GlucoseFlags.CONCENTRATION_TYPE_SAMPLE_LOCATION and len(data) >= offset + 1:
            type_location = data[offset]
            sample_type = type_location & 0x0F
            sample_location = (type_location >> 4) & 0x0F
            
            try:
                values["sample_type"] = GlucoseSampleType(sample_type).name.lower()
            except ValueError:
                values["sample_type"] = f"unknown_{sample_type}"
            
            values["sample_location"] = sample_location
            offset += 1
        
        return BLEReading(
            data_type="glucose",
            timestamp=timestamp,
            values=values,
            unit=unit,
            flags={
                "has_time_offset": bool(flags & GlucoseFlags.TIME_OFFSET_PRESENT),
                "has_type_location": bool(flags & GlucoseFlags.CONCENTRATION_TYPE_SAMPLE_LOCATION),
                "unit_mol_l": bool(flags & GlucoseFlags.CONCENTRATION_UNIT_MOL_L),
                "has_context": bool(flags & GlucoseFlags.CONTEXT_INFO_FOLLOWS),
            },
            raw_data=data,
        )
        
    except Exception as e:
        logger.error(f"Failed to parse glucose: {e}")
        return None


# ============================================
# TEMPERATURE MEASUREMENT (0x2A1C)
# ============================================

class TempFlags(IntFlag):
    """Temperature Measurement Flags byte"""
    UNIT_FAHRENHEIT = 0x01  # 0 = Celsius, 1 = Fahrenheit
    TIMESTAMP_PRESENT = 0x02
    TEMPERATURE_TYPE_PRESENT = 0x04


class TemperatureType(IntEnum):
    """Temperature type values"""
    RESERVED = 0
    ARMPIT = 1
    BODY = 2
    EAR = 3
    FINGER = 4
    GASTROINTESTINAL = 5
    MOUTH = 6
    RECTUM = 7
    TOE = 8
    TYMPANUM = 9


def parse_temperature(data: bytes) -> Optional[BLEReading]:
    """
    Parse Temperature Measurement characteristic (UUID 0x2A1C)
    
    Data format:
    - Flags (1 byte)
    - Temperature Value (FLOAT - 4 bytes IEEE-11073)
    - [Optional] Timestamp (7 bytes)
    - [Optional] Temperature Type (1 byte)
    """
    if len(data) < 5:
        logger.warning(f"Temperature data too short: {len(data)} bytes")
        return None
    
    try:
        flags = TempFlags(data[0])
        
        # Parse temperature (IEEE-11073 FLOAT)
        temperature = parse_float32(data[1:5])
        
        unit = "F" if flags & TempFlags.UNIT_FAHRENHEIT else "C"
        
        values = {"temperature": round(temperature, 2)}
        
        offset = 5
        timestamp = datetime.utcnow()
        
        # Parse optional timestamp
        if flags & TempFlags.TIMESTAMP_PRESENT and len(data) >= offset + 7:
            timestamp = parse_datetime(data[offset:offset + 7])
            offset += 7
        
        # Parse optional temperature type
        if flags & TempFlags.TEMPERATURE_TYPE_PRESENT and len(data) >= offset + 1:
            temp_type = data[offset]
            try:
                values["measurement_type"] = TemperatureType(temp_type).name.lower()
            except ValueError:
                values["measurement_type"] = f"unknown_{temp_type}"
        
        return BLEReading(
            data_type="temperature",
            timestamp=timestamp,
            values=values,
            unit=unit,
            flags={
                "unit_fahrenheit": bool(flags & TempFlags.UNIT_FAHRENHEIT),
                "has_timestamp": bool(flags & TempFlags.TIMESTAMP_PRESENT),
                "has_type": bool(flags & TempFlags.TEMPERATURE_TYPE_PRESENT),
            },
            raw_data=data,
        )
        
    except Exception as e:
        logger.error(f"Failed to parse temperature: {e}")
        return None


# ============================================
# WEIGHT MEASUREMENT (0x2A9D)
# ============================================

class WeightFlags(IntFlag):
    """Weight Measurement Flags byte"""
    UNIT_IMPERIAL = 0x01  # 0 = SI (kg), 1 = Imperial (lb)
    TIMESTAMP_PRESENT = 0x02
    USER_ID_PRESENT = 0x04
    BMI_HEIGHT_PRESENT = 0x08


def parse_weight(data: bytes) -> Optional[BLEReading]:
    """
    Parse Weight Measurement characteristic (UUID 0x2A9D)
    
    Data format:
    - Flags (1 byte)
    - Weight (2 bytes)
    - [Optional] Timestamp (7 bytes)
    - [Optional] User ID (1 byte)
    - [Optional] BMI (2 bytes)
    - [Optional] Height (2 bytes)
    """
    if len(data) < 3:
        logger.warning(f"Weight data too short: {len(data)} bytes")
        return None
    
    try:
        flags = WeightFlags(data[0])
        
        # Parse weight value
        weight_raw = struct.unpack("<H", data[1:3])[0]
        
        if flags & WeightFlags.UNIT_IMPERIAL:
            # Imperial: resolution of 0.01 lb
            weight = weight_raw * 0.01
            unit = "lb"
        else:
            # SI: resolution of 0.005 kg
            weight = weight_raw * 0.005
            unit = "kg"
        
        values = {"weight": round(weight, 2)}
        
        offset = 3
        timestamp = datetime.utcnow()
        
        # Parse optional timestamp
        if flags & WeightFlags.TIMESTAMP_PRESENT and len(data) >= offset + 7:
            timestamp = parse_datetime(data[offset:offset + 7])
            offset += 7
        
        # Parse optional user ID
        if flags & WeightFlags.USER_ID_PRESENT and len(data) >= offset + 1:
            values["user_id"] = data[offset]
            offset += 1
        
        # Parse optional BMI and height
        if flags & WeightFlags.BMI_HEIGHT_PRESENT and len(data) >= offset + 4:
            bmi_raw = struct.unpack("<H", data[offset:offset + 2])[0]
            values["bmi"] = round(bmi_raw * 0.1, 1)
            offset += 2
            
            height_raw = struct.unpack("<H", data[offset:offset + 2])[0]
            if flags & WeightFlags.UNIT_IMPERIAL:
                # Imperial: resolution of 0.1 inch
                values["height_in"] = round(height_raw * 0.1, 1)
            else:
                # SI: resolution of 0.001 m
                values["height_m"] = round(height_raw * 0.001, 3)
        
        return BLEReading(
            data_type="weight",
            timestamp=timestamp,
            values=values,
            unit=unit,
            flags={
                "unit_imperial": bool(flags & WeightFlags.UNIT_IMPERIAL),
                "has_timestamp": bool(flags & WeightFlags.TIMESTAMP_PRESENT),
                "has_user_id": bool(flags & WeightFlags.USER_ID_PRESENT),
                "has_bmi_height": bool(flags & WeightFlags.BMI_HEIGHT_PRESENT),
            },
            raw_data=data,
        )
        
    except Exception as e:
        logger.error(f"Failed to parse weight: {e}")
        return None


# ============================================
# PULSE OXIMETER SPOT-CHECK (0x2A5E)
# ============================================

class SpO2Flags(IntFlag):
    """PLX Spot-check Measurement Flags byte"""
    TIMESTAMP_PRESENT = 0x01
    MEASUREMENT_STATUS_PRESENT = 0x02
    DEVICE_SENSOR_STATUS_PRESENT = 0x04
    PULSE_AMPLITUDE_INDEX_PRESENT = 0x08


def parse_spo2(data: bytes) -> Optional[BLEReading]:
    """
    Parse PLX Spot-check Measurement characteristic (UUID 0x2A5E)
    
    Data format:
    - Flags (1 byte)
    - SpO2 (SFLOAT - 2 bytes)
    - PR (SFLOAT - 2 bytes)
    - [Optional] Timestamp (7 bytes)
    - [Optional] Measurement Status (2 bytes)
    - [Optional] Device/Sensor Status (3 bytes)
    - [Optional] Pulse Amplitude Index (SFLOAT - 2 bytes)
    """
    if len(data) < 5:
        logger.warning(f"SpO2 data too short: {len(data)} bytes")
        return None
    
    try:
        flags = SpO2Flags(data[0])
        
        spo2 = parse_sfloat(data[1:3])
        pulse_rate = parse_sfloat(data[3:5])
        
        values = {
            "spo2": round(spo2, 1),
            "pulse_rate": round(pulse_rate, 1),
        }
        
        offset = 5
        timestamp = datetime.utcnow()
        
        # Parse optional timestamp
        if flags & SpO2Flags.TIMESTAMP_PRESENT and len(data) >= offset + 7:
            timestamp = parse_datetime(data[offset:offset + 7])
            offset += 7
        
        # Parse optional pulse amplitude index
        if flags & SpO2Flags.PULSE_AMPLITUDE_INDEX_PRESENT:
            # Skip measurement and device status if present
            if flags & SpO2Flags.MEASUREMENT_STATUS_PRESENT:
                offset += 2
            if flags & SpO2Flags.DEVICE_SENSOR_STATUS_PRESENT:
                offset += 3
            
            if len(data) >= offset + 2:
                values["pulse_amplitude_index"] = round(parse_sfloat(data[offset:offset + 2]), 2)
        
        return BLEReading(
            data_type="spo2",
            timestamp=timestamp,
            values=values,
            unit="%",
            flags={
                "has_timestamp": bool(flags & SpO2Flags.TIMESTAMP_PRESENT),
                "has_status": bool(flags & SpO2Flags.MEASUREMENT_STATUS_PRESENT),
                "has_pai": bool(flags & SpO2Flags.PULSE_AMPLITUDE_INDEX_PRESENT),
            },
            raw_data=data,
        )
        
    except Exception as e:
        logger.error(f"Failed to parse SpO2: {e}")
        return None


# ============================================
# UTILITY FUNCTIONS
# ============================================

def parse_sfloat(data: bytes) -> float:
    """
    Parse SFLOAT (Short Float) - IEEE-11073 16-bit float
    
    Format: 4-bit exponent (signed) + 12-bit mantissa (signed)
    """
    if len(data) < 2:
        return 0.0
    
    raw = struct.unpack("<H", data)[0]
    
    # Special values
    if raw == 0x07FF:  # NaN
        return float('nan')
    if raw == 0x0800:  # NRes (not at this resolution)
        return float('nan')
    if raw == 0x07FE:  # +INFINITY
        return float('inf')
    if raw == 0x0802:  # -INFINITY
        return float('-inf')
    if raw == 0x0801:  # Reserved
        return float('nan')
    
    # Extract exponent and mantissa
    exponent = (raw >> 12) & 0x0F
    mantissa = raw & 0x0FFF
    
    # Handle signed values
    if exponent >= 8:
        exponent = exponent - 16
    if mantissa >= 2048:
        mantissa = mantissa - 4096
    
    return mantissa * (10 ** exponent)


def parse_float32(data: bytes) -> float:
    """
    Parse FLOAT (32-bit) - IEEE-11073 32-bit float
    
    Format: 8-bit exponent (signed) + 24-bit mantissa (signed)
    """
    if len(data) < 4:
        return 0.0
    
    raw = struct.unpack("<I", data)[0]
    
    # Extract exponent and mantissa
    exponent = (raw >> 24) & 0xFF
    mantissa = raw & 0x00FFFFFF
    
    # Handle signed values
    if exponent >= 128:
        exponent = exponent - 256
    if mantissa >= 0x800000:
        mantissa = mantissa - 0x1000000
    
    return mantissa * (10 ** exponent)


def parse_datetime(data: bytes) -> datetime:
    """
    Parse BLE DateTime (7 bytes)
    
    Format:
    - Year (2 bytes)
    - Month (1 byte)
    - Day (1 byte)
    - Hours (1 byte)
    - Minutes (1 byte)
    - Seconds (1 byte)
    """
    if len(data) < 7:
        return datetime.utcnow()
    
    try:
        year = struct.unpack("<H", data[0:2])[0]
        month = data[2]
        day = data[3]
        hour = data[4]
        minute = data[5]
        second = data[6]
        
        # Handle invalid values
        if year == 0:
            year = datetime.utcnow().year
        if month == 0 or month > 12:
            month = 1
        if day == 0 or day > 31:
            day = 1
        
        return datetime(year, month, day, hour, minute, second)
    except ValueError:
        return datetime.utcnow()


# ============================================
# PARSER FACTORY
# ============================================

CHARACTERISTIC_PARSERS = {
    "0x2A35": parse_blood_pressure,  # Blood Pressure Measurement
    "0x2A36": parse_blood_pressure,  # Intermediate Cuff Pressure
    "0x2A37": parse_heart_rate,       # Heart Rate Measurement
    "0x2A18": parse_glucose,          # Glucose Measurement
    "0x2A1C": parse_temperature,      # Temperature Measurement
    "0x2A9D": parse_weight,           # Weight Measurement
    "0x2A5E": parse_spo2,             # PLX Spot-check Measurement
    "0x2A5F": parse_spo2,             # PLX Continuous Measurement
}


def parse_ble_characteristic(uuid: str, data: bytes) -> Optional[BLEReading]:
    """
    Parse a BLE characteristic based on its UUID.
    
    Args:
        uuid: Characteristic UUID (e.g., "0x2A35" or "00002a35-0000-1000-8000-00805f9b34fb")
        data: Raw characteristic data bytes
        
    Returns:
        Parsed BLEReading or None if parsing failed
    """
    # Normalize UUID to short form
    normalized_uuid = uuid.upper()
    if len(normalized_uuid) > 6:
        # Extract 16-bit UUID from 128-bit format
        if "-" in normalized_uuid:
            short = normalized_uuid.split("-")[0][-4:]
            normalized_uuid = f"0x{short}"
    
    parser = CHARACTERISTIC_PARSERS.get(normalized_uuid)
    if parser:
        return parser(data)
    
    logger.warning(f"No parser available for characteristic: {uuid}")
    return None


def get_supported_characteristics() -> Dict[str, str]:
    """Get mapping of supported characteristic UUIDs to data types"""
    return {
        "0x2A35": "blood_pressure",
        "0x2A36": "blood_pressure",
        "0x2A37": "heart_rate",
        "0x2A18": "glucose",
        "0x2A1C": "temperature",
        "0x2A9D": "weight",
        "0x2A5E": "spo2",
        "0x2A5F": "spo2",
    }
