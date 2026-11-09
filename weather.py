import json
import math
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4
from zoneinfo import ZoneInfo
from urllib.parse import urlencode
from urllib.request import urlopen

from backend.prepare_data import ROOT


def fetch_json(url):
    with urlopen(url, timeout=3) as response:
        return json.load(response)


def valid_measurement(temperature, rainfall):
    return all(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) for value in [temperature, rainfall]) and -80 <= temperature <= 60 and 0 <= rainfall <= 2000


def forecast_days(payload):
    units = payload["daily_units"]
    if units.get("temperature_2m_mean") not in ["°C", "C"] or units.get("precipitation_sum") != "mm":
        raise ValueError("Weather units must be Celsius and millimetres.")
    daily = payload["daily"]
    dates, temperatures, rainfall = daily["time"], daily["temperature_2m_mean"], daily["precipitation_sum"]
    if not len(dates) == len(temperatures) == len(rainfall) or len(set(dates)) != len(dates):
        raise ValueError("Weather dates and measurements must align.")
    result = {}
    for value, temperature, precipitation in zip(dates, temperatures, rainfall):
        parsed = date.fromisoformat(value)
        if parsed.isoformat() != value:
            raise ValueError("Weather dates must use ISO format.")
        if valid_measurement(temperature, precipitation):
            result[value] = {"date": value, "temperature_c": temperature, "rainfall_mm": precipitation}
    return result


class WeatherProvider:
    def __init__(self, baselines, cache_directory=None, fetcher=fetch_json, clock=None):
        self.baselines = baselines
        self.cache_directory = Path(cache_directory or ROOT / "artifacts/weather")
        self.fetcher = fetcher
        self.clock = clock or (lambda: datetime.now(timezone.utc))

    def days(self, location, start, count, refresh=True):
        now = self.clock()
        local_today = now.astimezone(ZoneInfo("Africa/Lusaka")).date()
        first = date.fromisoformat(start)
        key = f"{location['latitude']:.5f}_{location['longitude']:.5f}"
        path = self.cache_directory / f"history_{key}.json"
        cache = None
        try:
            candidate = json.loads(path.read_text())
            age = (now - datetime.fromisoformat(candidate["retrieved_at"])).total_seconds()
            if 0 <= age <= 48 * 3600:
                forecast_days(candidate["payload"])
                cache = candidate
        except (OSError, ValueError, KeyError, TypeError):
            pass
        recent = cache is not None and (now - datetime.fromisoformat(cache["retrieved_at"])).total_seconds() < 3 * 3600
        provider_failed = not refresh and not recent
        if refresh and not recent and first <= local_today + timedelta(days=15):
            query = urlencode({"latitude": location["latitude"], "longitude": location["longitude"], "daily": "temperature_2m_mean,precipitation_sum", "past_days": 20, "forecast_days": 16, "timezone": "Africa/Lusaka"})
            url = f"https://api.open-meteo.com/v1/forecast?{query}"
            try:
                payload = self.fetcher(url)
                parsed = forecast_days(payload)
                if local_today.isoformat() not in parsed:
                    raise ValueError("The provider did not return current weather.")
                candidate = {"retrieved_at": now.isoformat(), "source_url": url, "payload": payload}
                serialized = json.dumps(candidate, allow_nan=False)
                cache = candidate
                try:
                    self.cache_directory.mkdir(parents=True, exist_ok=True)
                    temporary = path.with_suffix(f".{uuid4().hex}.tmp")
                    temporary.write_text(serialized)
                    temporary.replace(path)
                except OSError:
                    pass
            except (OSError, ValueError, KeyError, TypeError):
                provider_failed = True
        available = forecast_days(cache["payload"]) if cache else {}
        baseline = self.baselines[location["baseline_key"]]
        result = []
        for offset in range(count):
            current = first + timedelta(days=offset)
            value = current.isoformat()
            if value in available:
                result.append({**available[value], "basis": "cached_weather" if provider_failed else "recent_weather" if current < local_today else "forecast", "source": "Open-Meteo", "source_url": cache["source_url"], "retrieved_at": cache["retrieved_at"]})
                continue
            typical = baseline["months"][str(current.month)]
            if not valid_measurement(typical.get("temperature_c"), typical.get("rainfall_mm")):
                raise ValueError("The seasonal baseline contains an invalid measurement.")
            result.append({"date": value, **typical, "basis": "seasonal_baseline", "source": "NASA POWER", "source_url": baseline["source_url"], "retrieved_at": baseline["retrieved_at"], "reference_period": baseline["period"], "reference_name": baseline["name"]})
        return result
