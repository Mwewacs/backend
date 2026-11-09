import json
import math
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4
from zoneinfo import ZoneInfo

import joblib
import pandas as pd

from backend.database import Repository
from backend.prepare_data import FEATURE_NAMES, ROOT, build_features, prepare_data


class AreaNotFound(Exception):
    pass


class ForecastUnavailable(Exception):
    pass


class ForecastService:
    def __init__(self, repository, dataset, model_report, model=None, outlook_engine=None):
        self.repository = repository
        self.dataset = dataset
        self.report = model_report
        self.model = model
        self.outlook_engine = outlook_engine
        self.overview_cache = None

    @classmethod
    def from_files(cls, directory=None):
        output = Path(directory or ROOT / "artifacts")
        dataset_path = output / "prepared-data.json"
        if not dataset_path.exists():
            prepare_data(output_directory=output)
        dataset = json.loads(dataset_path.read_text())
        repository = Repository(output / "cholera.sqlite3")
        repository.import_dataset(dataset)
        report_path = output / "model-report.json"
        report = json.loads(report_path.read_text()) if report_path.exists() else {"ready": False, "status": "not_trained", "reasons": ["The timing model has not completed training and evaluation."], "candidates": [], "data": {}}
        model_path = output / "active-model.joblib"
        model = joblib.load(model_path) if report.get("ready") and model_path.exists() else None
        from backend.cholera_forecast import CholeraOutlook
        engine = CholeraOutlook.from_files(output) if (output / "forecast-model.joblib").exists() else None
        return cls(repository, dataset, report, model, engine)

    def areas(self):
        return self.repository.areas()

    def area(self, identifier):
        area = self.repository.area(identifier)
        if area is None:
            raise AreaNotFound("The selected area was not found.")
        return area

    def forecast_status(self):
        ready = self.outlook_engine is not None or self.report.get("ready", False) and self.model is not None
        return {"status": "available" if ready else "unavailable", "reason": None if ready else "A timing forecast is not available until the model has passed independent evaluation with dated source records."}

    def calendar(self, identifier, start, end):
        area = self.area(identifier)
        records = self.repository.observations(identifier, start, end, daily=True)
        by_date = {record["start"]: record for record in records}
        days = []
        current, last = date.fromisoformat(start), date.fromisoformat(end)
        while current <= last:
            observed_date = current.isoformat()
            record = by_date.get(observed_date)
            conflicted = any(conflict["area_id"] == identifier and conflict["date"] == observed_date for conflict in self.dataset.get("conflicts", []))
            days.append({"date": observed_date, "cases": record["cases"] if record else None, "status": "reported" if record else "conflict" if conflicted else "not_reported", "source_url": record["source_url"] if record else None})
            current += timedelta(days=1)
        intervals = self.repository.observations(identifier, start, end)
        intervals = [record for record in intervals if record["duration_days"] > 1 and record["cases"] is not None and not record["age_restricted"]]
        return {"area": area, "from": start, "to": end, "days": days, "intervals": intervals, "forecast": self.forecast_status(), "predictions": [prediction for prediction in self.repository.predictions(identifier) if prediction["window_start"] <= end and prediction["window_end"] >= start]}

    def predict(self, identifier, issue_date):
        area = self.area(identifier)
        if self.outlook_engine is not None:
            if issue_date <= self.outlook_engine.report["training_cutoff"]:
                raise ForecastUnavailable("Select a forecast date after the model training cutoff.")
            return self.outlook(identifier, issue_date, 1)["prediction"]
        status = self.forecast_status()
        if status["status"] != "available":
            raise ForecastUnavailable(status["reason"])
        if issue_date <= self.report["training_cutoff"]:
            raise ForecastUnavailable("Select a forecast date after the model training cutoff.")
        features = build_features(identifier, issue_date, self.dataset["model_observations"])
        if features is None:
            raise ForecastUnavailable("Complete case-free history is not available for this area and forecast date.")
        frame = pd.DataFrame([features], columns=FEATURE_NAMES).astype(float)
        score = float(self.model.predict_proba(frame)[0][1])
        if not math.isfinite(score) or score < 0 or score > 1:
            raise ForecastUnavailable("The model did not return a valid risk score.")
        high = self.report["threshold"]
        medium = self.report["medium_threshold"]
        category = "high" if score >= high else "medium" if score >= medium else "low"
        prediction = {"id": uuid4().hex, "area_id": identifier, "area_name": area["name"], "issued_at": datetime.now(timezone.utc).isoformat(), "window_start": issue_date, "window_end": (date.fromisoformat(issue_date) + timedelta(days=self.report["horizon_days"] - 1)).isoformat(), "data_cutoff": (date.fromisoformat(issue_date) - timedelta(days=1)).isoformat(), "risk_score": score, "risk_category": category, "model_version": self.report["version"], "model": self.report["selected_model"], "input_snapshot": features, "mode": "historical_replay" if issue_date < date.today().isoformat() else "forecast", "target_definition": self.report["target_definition"]}
        self.repository.save_prediction(prediction)
        return prediction

    def predictions(self, identifier=None):
        if identifier:
            self.area(identifier)
        return self.repository.predictions(identifier)

    def alerts(self):
        status = self.forecast_status()
        if status["status"] != "available":
            return {"forecast": status, "alerts": []}
        if self.outlook_engine is not None:
            today = self.outlook_engine.clock().astimezone(ZoneInfo("Africa/Lusaka")).date().isoformat()
            overview = self.overview(today)
            return {"forecast": status, "alerts": [prediction for prediction in overview["areas"] if prediction["risk_category"] == "high"], "checked_areas": len(overview["areas"]), "window_start": overview["window_start"], "window_end": overview["window_end"], "model_version": overview["model_version"]}
        latest = {}
        for prediction in self.repository.predictions():
            key = (prediction["area_id"], prediction["window_start"], prediction["model_version"])
            if key not in latest:
                latest[key] = prediction
        return {"forecast": status, "alerts": sorted([prediction for prediction in latest.values() if prediction["risk_category"] == "high" and prediction["model_version"] == self.model_info().get("version") and prediction["mode"] != "historical_replay" and prediction["window_end"] >= date.today().isoformat()], key=lambda prediction: (-prediction["risk_score"], prediction["area_name"]))}

    def model_info(self):
        return self.outlook_engine.report if self.outlook_engine is not None else self.report

    def outlook(self, identifier, start, count=7):
        area = self.area(identifier)
        if self.outlook_engine is None:
            from backend.cholera_forecast import CholeraOutlook
            self.outlook_engine = CholeraOutlook.from_files()
        result = self.outlook_engine.outlook(area, start, count)
        self.repository.save_outlook(result)
        self.repository.save_prediction(result["prediction"])
        self.overview_cache = None
        return result

    def overview(self, start):
        if self.outlook_engine is None:
            raise ForecastUnavailable("The area forecast model is not available.")
        now = self.outlook_engine.clock()
        cached = self.overview_cache
        if cached and cached["window_start"] == start and cached["model_version"] == self.outlook_engine.report["version"] and 0 <= (now - datetime.fromisoformat(cached["generated_at"])).total_seconds() < 3 * 3600:
            return cached
        predictions = []
        for area in self.areas():
            location = self.outlook_engine.reference["locations"].get(area["id"])
            if location is None:
                continue
            result = self.outlook_engine.outlook(area, start, 1, refresh_weather=False)
            self.repository.save_prediction(result["prediction"])
            predictions.append({**result["prediction"], "region": area["region"], "level": area["level"], "latitude": location["latitude"], "longitude": location["longitude"], "basis": result["basis"], "uses_seasonal_inputs": result["days"][0]["uses_seasonal_inputs"]})
        self.overview_cache = {"generated_at": now.isoformat(), "window_start": start, "window_end": (date.fromisoformat(start) + timedelta(days=6)).isoformat(), "model_version": self.outlook_engine.report["version"], "areas": sorted(predictions, key=lambda prediction: (-prediction["risk_score"], prediction["area_name"]))}
        return self.overview_cache

    def record_request(self, record):
        self.repository.save_prediction_request(record)

    def prediction_requests(self, identifier=None):
        if identifier:
            self.area(identifier)
        return self.repository.prediction_requests(identifier)

    def health(self):
        return {"status": "ok", "database": "available" if self.repository.health() else "unavailable", "model_ready": self.forecast_status()["status"] == "available", "model_validated": self.model_info().get("validated", self.report.get("ready", False))}
