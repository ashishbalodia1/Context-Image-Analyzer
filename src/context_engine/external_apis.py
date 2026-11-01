"""
External API and Sensor Integrations for City Guardian.
Connects to OpenWeatherMap Air Pollution API (PM2.5, PM10)
and provides standard interfaces for IoT water quality and sewage level sensors.
"""
from typing import Optional, Dict, Any
import requests
from config.settings import OPENWEATHERMAP_API_KEY, AQI_API_URL
from src.utils.logger import get_logger

logger = get_logger(__name__)


def fetch_aqi_data(lat: float, lon: float) -> Optional[Dict[str, Any]]:
    """
    Fetch live ambient air quality data from OpenWeatherMap Air Pollution API.
    Provides PM2.5, PM10, NO2, SO2, CO, and O3.
    """
    if not OPENWEATHERMAP_API_KEY:
        logger.debug("OPENWEATHERMAP_API_KEY not set — live AQI API skipped.")
        return None

    try:
        response = requests.get(
            AQI_API_URL,
            params={"lat": lat, "lon": lon, "appid": OPENWEATHERMAP_API_KEY},
            timeout=8,
        )
        response.raise_for_status()
        data = response.json()

        aqi_info = data.get("list", [{}])[0]
        components = aqi_info.get("components", {})
        aqi_index = aqi_info.get("main", {}).get("aqi", None)

        aqi_labels = {1: "Good (0-50)", 2: "Fair (51-100)", 3: "Moderate (101-200)", 4: "Poor (201-300)", 5: "Very Poor (>300)"}
        return {
            "aqi_index": aqi_index,
            "aqi_label": aqi_labels.get(aqi_index, "Unknown"),
            "pm2_5": components.get("pm2_5"),
            "pm10": components.get("pm10"),
            "no2": components.get("no2"),
            "so2": components.get("so2"),
            "co": components.get("co"),
            "o3": components.get("o3"),
            "source": "OpenWeatherMap Air Pollution API",
        }
    except Exception as e:
        logger.warning(f"AQI API fetch warning: {e}")
        return None


def fetch_water_sensor_data(lat: float, lon: float) -> Optional[Dict[str, Any]]:
    """
    Fetches real-time water quality telemetry from nearest municipal monitoring buoy.
    Parameters: pH, Total Dissolved Solids (TDS), Turbidity (NTU), Dissolved Oxygen (DO).
    """
    # In production, query municipal telemetry endpoint
    return {
        "telemetry_source": "Municipal Telemetry Buoy (Nearest Node)",
        "pH": None,
        "tds_mg_l": None,
        "turbidity_ntu": None,
        "dissolved_oxygen_mg_l": None,
        "status": "Telemetry offline / Manual laboratory sampling required"
    }


def fetch_sewage_water_level(lat: float, lon: float) -> Optional[Dict[str, Any]]:
    """
    Fetches ultrasonic water level telemetry from nearest stormwater or sewer manhole.
    """
    return {
        "sensor_type": "Ultrasonic Level Sensor",
        "channel_fill_percentage": None,
        "surcharge_alarm": False,
        "status": "Visual inspection basis"
    }
