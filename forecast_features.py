import math
from datetime import date, timedelta

FEATURES = ["rainfall_lag_week", "rainfall_earlier_week", "temperature_lag_week", "temperature_earlier_week", "month_sin", "month_cos", "latitude", "longitude", "water_access", "sanitation_access", "previous_annual_occurrence"]
TARGET = "At least one reported suspected cholera case in the area during the following seven days. This predicts reported cholera activity, not an official outbreak declaration or an exact onset day."


def forecast_features(location, issue_date, weather):
    issued = date.fromisoformat(issue_date)
    recent = [weather.get((issued - timedelta(days=offset)).isoformat()) for offset in range(6, 13)]
    earlier = [weather.get((issued - timedelta(days=offset)).isoformat()) for offset in range(13, 20)]
    if any(day is None for day in recent + earlier):
        return None
    background = location.get("background") or {}
    context = (background.get("context") or {}) if issued.year >= 2010 else {}
    history = [record for record in background.get("annual_history", []) if record["year"] < issued.year and record["cases"] is not None]
    angle = 2 * math.pi * (issued.timetuple().tm_yday - 1) / 365.25
    return {"rainfall_lag_week": sum(day["rainfall_mm"] for day in recent), "rainfall_earlier_week": sum(day["rainfall_mm"] for day in earlier), "temperature_lag_week": sum(day["temperature_c"] for day in recent) / 7, "temperature_earlier_week": sum(day["temperature_c"] for day in earlier) / 7, "month_sin": math.sin(angle), "month_cos": math.cos(angle), "latitude": location["latitude"], "longitude": location["longitude"], "water_access": context.get("water_percentage"), "sanitation_access": context.get("sanitation_percentage"), "previous_annual_occurrence": sum(record["cases"] > 0 for record in history) / len(history) if history else None}


def case_windows(dataset):
    by_area = {}
    for record in dataset["daily_observations"]:
        by_area.setdefault(record["area_id"], {})[record["start"]] = record["cases"]
    examples = []
    for identifier, records in by_area.items():
        for start in sorted(records):
            issued = date.fromisoformat(start)
            counts = [records.get((issued + timedelta(days=offset)).isoformat()) for offset in range(7)]
            if any(value is None for value in counts):
                continue
            examples.append({"area_id": identifier, "issue_date": start, "window_end": (issued + timedelta(days=6)).isoformat(), "label": int(sum(counts) > 0), "case_count": sum(counts), "frequency": "daily_aggregation"})
    for record in dataset["model_observations"]:
        if record["duration_days"] != 7 or record["cases"] is None:
            continue
        if any(record["start"] <= value <= record["end"] for value in by_area.get(record["area_id"], {})):
            continue
        examples.append({"area_id": record["area_id"], "issue_date": record["start"], "window_end": record["end"], "label": int(record["cases"] > 0), "case_count": record["cases"], "frequency": "reported_week"})
    return sorted(examples, key=lambda example: (example["issue_date"], example["area_id"]))
