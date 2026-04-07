# tools/weather.py
import requests
from datetime import datetime, timedelta

STATION_LAT = 38.05   # your coordinates
STATION_LON = -84.50

def get_weather_for_visit(timestamp: datetime) -> dict:
    """Fetch hourly weather for a visit timestamp, using the archive API for past dates."""
    date_str = timestamp.strftime("%Y-%m-%d")
    hour = timestamp.hour

    # Use archive API for dates older than 2 days, forecast for recent/today
    if datetime.now() - timestamp > timedelta(days=2):
        url = "https://archive-api.open-meteo.com/v1/archive"
    else:
        url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": STATION_LAT,
        "longitude": STATION_LON,
        "hourly": "temperature_2m,precipitation,windspeed_10m,weathercode",
        "temperature_unit": "fahrenheit",
        "start_date": date_str,
        "end_date": date_str,
        "timezone": "America/New_York",
    }
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    return {
        "temperature_f": data["hourly"]["temperature_2m"][hour],
        "precipitation_mm": data["hourly"]["precipitation"][hour],
        "windspeed_mph": data["hourly"]["windspeed_10m"][hour],
        "weather_code": data["hourly"]["weathercode"][hour],
    }

if __name__ == "__main__":
    from datetime import datetime
    result = get_weather_for_visit(datetime.now())
    print(result)