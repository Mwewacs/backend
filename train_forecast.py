import hashlib
import json
import platform
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import joblib
import numpy as np
import pandas as pd
import sklearn

from backend.forecast_features import FEATURES, TARGET, case_windows, forecast_features
from backend.prepare_data import ROOT
from backend.train_models import candidate_models, evaluate
from backend.weather import forecast_days


def historical_weather(location, start, end, directory=None):
    output = Path(directory or ROOT / "data/raw/training-weather")
    output.mkdir(parents=True, exist_ok=True)
    query = urlencode({"latitude": location["latitude"], "longitude": location["longitude"], "start_date": start, "end_date": end, "daily": "temperature_2m_mean,precipitation_sum", "models": "era5", "timezone": "Africa/Lusaka"})
    url = f"https://archive-api.open-meteo.com/v1/archive?{query}"
    key = hashlib.sha256(url.encode()).hexdigest()[:16]
    path = output / f"{key}.json"
    if path.exists():
        source = json.loads(path.read_text())
    else:
        with urlopen(url, timeout=30) as response:
            payload = json.load(response)
        forecast_days(payload)
        source = {"source_url": url, "retrieved_at": datetime.now(timezone.utc).isoformat(), "payload": payload}
        path.write_text(json.dumps(source, indent=2, allow_nan=False) + "\n")
    return forecast_days(source["payload"]), {"file": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path), "source_url": url, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def prepare_forecast_examples(dataset, reference, weather_loader=historical_weather):
    grouped = defaultdict(list)
    for example in case_windows(dataset):
        if example["area_id"] in reference["locations"]:
            grouped[(example["area_id"], example["issue_date"][:4])].append(example)
    prepared, rejected, sources = [], [], []
    for (identifier, year), examples in sorted(grouped.items()):
        location = reference["locations"][identifier]
        start = (date.fromisoformat(min(example["issue_date"] for example in examples)) - timedelta(days=19)).isoformat()
        end = (date.fromisoformat(max(example["issue_date"] for example in examples)) - timedelta(days=6)).isoformat()
        weather, source = weather_loader(location, start, end)
        sources.append({"area_id": identifier, "year": year, **source})
        for example in examples:
            features = forecast_features(location, example["issue_date"], weather)
            if features is None:
                rejected.append({"area_id": identifier, "date": example["issue_date"], "reason": "Incomplete lagged weather."})
                continue
            prepared.append({**example, "features": features})
    return sorted(prepared, key=lambda example: (example["issue_date"], example["area_id"])), sources, rejected


def chronological_validation(examples):
    dates = sorted({example["issue_date"] for example in examples})
    for cutoff in reversed(dates[:max(1, int(len(dates) * 0.8))]):
        training = [example for example in examples if example["window_end"] < cutoff]
        validation = [example for example in examples if example["issue_date"] >= cutoff]
        if len(training) < 30 or len(validation) < max(20, len(examples) * 0.2):
            continue
        if {example["label"] for example in training} == {0, 1} and {example["label"] for example in validation} == {0, 1}:
            return training, validation, cutoff
    return None


def frame(examples):
    return pd.DataFrame([example["features"] for example in examples], columns=FEATURES).astype(float)


def models(labels):
    candidates = candidate_models(labels)
    candidates["xgboost_constrained"].steps[-1][1].set_params(monotone_constraints=tuple(-1 if name in ["water_access", "sanitation_access"] else 1 if name == "previous_annual_occurrence" else 0 for name in FEATURES))
    return candidates


def fit_forecast(examples, sources, rejected=None, output_directory=None):
    output = Path(output_directory or ROOT / "artifacts")
    output.mkdir(parents=True, exist_ok=True)
    labels = np.array([example["label"] for example in examples])
    if set(labels) != {0, 1}:
        raise ValueError("Cholera training needs both reported-case and case-free windows.")
    split = chronological_validation(examples)
    candidates = []
    if split:
        training, validation, cutoff = split
        for name, pipeline in models([example["label"] for example in training]).items():
            pipeline.fit(frame(training), [example["label"] for example in training])
            probabilities = pipeline.predict_proba(frame(validation))[:, 1]
            candidates.append({"name": name, "status": "retrospective_validation", "training_rows": len(training), "validation": evaluate([example["label"] for example in validation], probabilities, 0.5), "test": None})
        selected = max(candidates, key=lambda candidate: (candidate["validation"]["f1"], candidate["validation"]["sensitivity"], -candidate["validation"]["brier_score"]))["name"]
    else:
        cutoff = None
        selected = "logistic_regression"
        candidates = [{"name": name, "status": "trained_exploratory", "training_rows": len(examples), "validation": None, "test": None} for name in models(labels)]
    signature = {"examples": examples, "sources": sources, "features": FEATURES, "seed": 42, "code_sha256": hashlib.sha256(Path(__file__).read_bytes() + (ROOT / "backend/forecast_features.py").read_bytes()).hexdigest()}
    version = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:12]
    directory = output / "forecast-models" / version
    directory.mkdir(parents=True, exist_ok=True)
    fitted = {}
    for name, pipeline in models(labels).items():
        pipeline.fit(frame(examples), labels)
        joblib.dump(pipeline, directory / f"{name}.joblib")
        fitted[name] = pipeline
    report = {"ready": True, "validated": False, "status": "research_forecast_model", "version": version, "created_at": datetime.now(timezone.utc).isoformat(), "selected_model": selected, "target_definition": TARGET, "horizon_days": 7, "features": FEATURES, "weather_lag_days": [6, 19], "calibrated_probability": False, "risk_bands": {"medium": 0.3, "high": 0.7}, "data": {"examples": len(examples), "positive_examples": int(labels.sum()), "negative_examples": int(len(labels) - labels.sum()), "areas": len({example["area_id"] for example in examples}), "years": sorted({example["issue_date"][:4] for example in examples}), "weather_sources": len(sources), "rejected_examples": len(rejected or [])}, "candidates": candidates, "evaluation": {"type": "retrospective_chronological_validation", "cutoff": cutoff, "independent_test": None, "selection": "Validation F1, then sensitivity, then Brier score. The selected model is refitted on all labelled data.", "window_purge": "Training outcomes end before validation issue dates."} if split else None, "reasons": ["Independent outbreak-level evaluation and probability calibration remain pending."], "limitations": ["Most negative and positive examples come from the same 2017 Lusaka outbreak and adjacent areas.", "Retrospective validation is not an independent outbreak test and may overstate generalisation.", "Source publication dates and retrospective covariate availability are not fully verified.", "Historical reanalysis and live modelled weather can have different error distributions.", "Predictions outside observed places, seasons and weather ranges are extrapolations.", "The target is reported suspected cholera activity, not laboratory-confirmed cholera or an official outbreak declaration.", "Risk bands are research display thresholds, not clinical action thresholds."], "weather_sources": sources, "rejected": rejected or [], "artifact": str(directory.relative_to(output) / f"{selected}.joblib"), "random_seed": 42}
    report["training_cutoff"] = max(example["window_end"] for example in examples)
    report["versions"] = {"python": platform.python_version(), "scikit_learn": sklearn.__version__, "numpy": np.__version__}
    report["full_training_rows"] = len(examples)
    report["training_area_ids"] = sorted({example["area_id"] for example in examples})
    report["sources"] = [{"name": "Dated cholera surveillance", "url": "https://cholera-taxonomy.middle-distance.com/locations/538/country_report"}, {"name": "Historical weather training inputs", "url": "https://open-meteo.com/en/docs/historical-weather-api"}, {"name": "Recent and forecast weather", "url": "https://open-meteo.com/en/docs"}, {"name": "Published district covariates", "url": "https://journals.plos.org/plosntds/article?id=10.1371/journal.pntd.0008227"}, {"name": "NASA POWER seasonal input fallback", "url": "https://power.larc.nasa.gov/docs/services/api/temporal/climatology/"}, {"name": "GeoNames area references", "url": "https://www.geonames.org/"}]
    classifier = fitted[selected].steps[-1][1]
    importance = getattr(classifier, "feature_importances_", None)
    report["feature_importance"] = dict(zip(FEATURES, map(float, importance))) if importance is not None else None
    joblib.dump(fitted[selected], output / "forecast-model.joblib")
    (output / "forecast-report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (directory / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


if __name__ == "__main__":
    dataset = json.loads((ROOT / "artifacts/prepared-data.json").read_text())
    reference = json.loads((ROOT / "data/environment-reference.json").read_text())
    examples, sources, rejected = prepare_forecast_examples(dataset, reference)
    report = fit_forecast(examples, sources, rejected)
    print(json.dumps({name: report[name] for name in ["version", "selected_model", "data", "evaluation", "candidates"]}, indent=2))
