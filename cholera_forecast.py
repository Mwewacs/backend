import hashlib
import json
import math
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import pandas as pd

from backend.forecast_features import FEATURES, forecast_features
from backend.explanations import forecast_reasons, forest_effects
from backend.prepare_data import ROOT
from backend.references import forecast_references
from backend.weather import WeatherProvider


class CholeraOutlook:
    def __init__(self, reference, report, model, provider=None, clock=None):
        self.reference = reference
        self.report = report
        self.model = model
        self.provider = provider or WeatherProvider(reference["baselines"])
        self.clock = clock or (lambda: datetime.now(timezone.utc))

    @classmethod
    def from_files(cls, directory=None):
        output = Path(directory or ROOT / "artifacts")
        reference = json.loads((ROOT / "data/environment-reference.json").read_text())
        report = json.loads((output / "forecast-report.json").read_text())
        return cls(reference, report, joblib.load(output / "forecast-model.joblib"))

    def outlook(self, area, start, count=7, refresh_weather=True):
        location = self.reference["locations"].get(area["id"])
        if location is None:
            raise ValueError("Choose a district or local area with a weather reference.")
        issued = date.fromisoformat(start)
        weather = self.provider.days(location, (issued - timedelta(days=19)).isoformat(), count + 13, refresh=refresh_weather)
        by_date = {day["date"]: day for day in weather}
        dates = [(issued + timedelta(days=offset)).isoformat() for offset in range(count)]
        features = [forecast_features(location, value, by_date) for value in dates]
        if any(value is None for value in features):
            raise ValueError("Weather inputs could not be prepared for the cholera model.")
        probabilities = self.model.predict_proba(pd.DataFrame(features, columns=FEATURES).astype(float))[:, 1]
        effects = forest_effects(self.model, features)
        now = self.clock()
        today = now.astimezone(ZoneInfo("Africa/Lusaka")).date()
        days = []
        for index, (value, inputs, probability) in enumerate(zip(dates, features, probabilities)):
            score = float(probability)
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("The cholera model did not return a valid prediction.")
            first = date.fromisoformat(value)
            lagged = [by_date[(first - timedelta(days=offset)).isoformat()] for offset in range(6, 20)]
            bases = {day["basis"] for day in lagged}
            basis = next(iter(bases)) if len(bases) == 1 else "mixed"
            category = "high" if score >= self.report["risk_bands"]["high"] else "medium" if score >= self.report["risk_bands"]["medium"] else "low"
            days.append({"date": value, "window_start": value, "window_end": (first + timedelta(days=6)).isoformat(), "risk_score": score, "risk_category": category, "temperature_c": round((inputs["temperature_lag_week"] + inputs["temperature_earlier_week"]) / 2, 1), "rainfall_mm": round(inputs["rainfall_lag_week"] + inputs["rainfall_earlier_week"], 1), "input_window_start": (first - timedelta(days=19)).isoformat(), "input_window_end": (first - timedelta(days=6)).isoformat(), "basis": basis, "uses_seasonal_inputs": any(day["basis"] == "seasonal_baseline" for day in lagged), "source": "Open-Meteo / NASA POWER" if basis == "mixed" else lagged[0]["source"], "source_url": lagged[0]["source_url"], "retrieved_at": max(day["retrieved_at"] for day in lagged), "weather_inputs": lagged, "input_snapshot": inputs})
            contributions = effects["contributions"][index] if effects else None
            seasonal = all(day["basis"] == "seasonal_baseline" for day in lagged)
            days[-1]["reasons"] = forecast_reasons(inputs, value, seasonal, contributions)
            days[-1]["explanation"] = {"method": effects["method"], "baseline_score": effects["baseline_score"], "contributions": contributions} if effects else {"method": "input_summary"}
            days[-1]["references"] = forecast_references(lagged, location, inputs, self.report)
        first = days[0]
        signature = json.dumps({"area": area["id"], "days": days, "model": self.report["version"], "location": location}, sort_keys=True)
        identifier = hashlib.sha256(signature.encode()).hexdigest()[:24]
        prediction_identifier = hashlib.sha256(json.dumps({"area": area["id"], "day": first, "model": self.report["version"]}, sort_keys=True).encode()).hexdigest()[:24]
        prediction = {"id": prediction_identifier, "area_id": area["id"], "area_name": area["name"], "issued_at": now.isoformat(), "window_start": first["window_start"], "window_end": first["window_end"], "data_cutoff": min(date.fromisoformat(first["input_window_end"]), today - timedelta(days=1)).isoformat(), "risk_score": first["risk_score"], "risk_category": first["risk_category"], "model_version": self.report["version"], "model": self.report["selected_model"], "input_snapshot": first["input_snapshot"], "mode": "historical_replay" if issued < today else "forecast", "target_definition": self.report["target_definition"], "validated": self.report["validated"], "calibrated_probability": self.report["calibrated_probability"]}
        prediction["reasons"] = first["reasons"]
        prediction["explanation"] = first["explanation"]
        prediction["references"] = first["references"]
        return {"id": identifier, "area": {name: area[name] for name in ["id", "name", "region", "level"]}, "kind": "cholera_forecast", "experimental": not self.report["validated"], "calibrated_probability": self.report["calibrated_probability"], "confidence": "very limited" if first["uses_seasonal_inputs"] or area["id"] not in self.report.get("training_area_ids", []) else "limited", "generated_at": now.isoformat(), "from": days[0]["date"], "to": days[-1]["date"], "window_start": first["window_start"], "window_end": first["window_end"], "risk_score": first["risk_score"], "risk_category": first["risk_category"], "basis": first["basis"], "temperature_c": first["temperature_c"], "rainfall_mm": first["rainfall_mm"], "input_window_start": first["input_window_start"], "input_window_end": first["input_window_end"], "days": days, "location": {name: value for name, value in location.items() if name != "background"}, "background": location.get("background"), "model": self.report, "prediction": prediction}
