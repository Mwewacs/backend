from datetime import date, timedelta

import numpy as np

from backend.prepare_data import FEATURE_NAMES, parse_observation
from backend.service import AreaNotFound, ForecastService


def given_records(counts=None, start="2020-01-01", location="AFR::ZMB::Lusaka::Lusaka::Kanyama", period_days=1):
    counts = counts if counts is not None else [0] * 14 + [0, 1, 0, 0, 0, 0, 0]
    first = date.fromisoformat(start)
    return [parse_observation({"Location": location, "TL": (first + timedelta(days=index * period_days)).isoformat(), "TR": (first + timedelta(days=index * period_days + period_days - 1)).isoformat(), "sCh": str(count)}, "example") for index, count in enumerate(counts)]


def given_examples(years=(2017, 2020, 2023)):
    return [{"area_id": "area", "issue_date": f"{year}-{month:02d}-15", "window_end": f"{year}-{month:02d}-21", "label": label, "onset_date": f"{year}-{month:02d}-17" if label else None, "outbreak_episode_id": f"outbreak-{year}" if label else None, "episode_group": str(year), "features": dict(zip(FEATURE_NAMES, [label, 0.0, 1.0, label * 10, 1.0]))} for year in years for month, label in [(1, 0), (2, 1)]]


class GivenRepository:
    def __init__(self, records=None):
        self.records = records if records is not None else given_records()
        self.saved = []
        self.saved_outlooks = []
        self.saved_requests = []
        self.selected_area = {"id": self.records[0]["area_id"], "name": "Kanyama", "region": "Lusaka", "level": "local_area", "first_date": "2020-01-01", "last_date": "2020-01-21", "daily_records": len(self.records)}
        self.listed_areas = [self.selected_area]

    def area(self, identifier):
        return next((area for area in self.listed_areas if area["id"] == identifier), None)

    def areas(self):
        return self.listed_areas

    def observations(self, identifier, start, end, daily=False):
        if not daily:
            return []
        return [record for record in self.records if record["area_id"] == identifier and start <= record["start"] <= end]

    def save_prediction(self, prediction):
        self.saved.append(prediction)

    def predictions(self, identifier=None):
        return [prediction for prediction in self.saved if identifier is None or prediction["area_id"] == identifier]

    def health(self):
        return True

    def save_outlook(self, outlook):
        self.saved_outlooks.append(outlook)

    def save_prediction_request(self, record):
        self.saved_requests.append(record)

    def prediction_requests(self, identifier=None):
        return [record for record in self.saved_requests if identifier is None or record["area_id"] == identifier]


class GivenModel:
    def __init__(self, score=0.8):
        self.score = score

    def predict_proba(self, frame):
        return np.array([[1 - self.score, self.score]])


def given_service(ready=True, score=0.8, records=None):
    repository = GivenRepository(records)
    report = {"ready": ready, "version": "model-1", "selected_model": "example", "threshold": 0.7, "medium_threshold": 0.3, "horizon_days": 7, "training_cutoff": "2019-12-31", "target_definition": "A candidate onset in the next seven days.", "candidates": [], "data": {}}
    dataset = {"model_observations": repository.records, "conflicts": []}
    return ForecastService(repository, dataset, report, GivenModel(score) if ready else None)


class GivenApiService:
    def __init__(self):
        self.calls = []
        self.failure = None
        self.audit_failure = None
        self.requests = []
        self.outlook_response = {"result": "outlook"}

    def result(self, operation, *arguments):
        if self.failure:
            raise self.failure
        if arguments and arguments[0] == "unknown":
            raise AreaNotFound("The selected area was not found.")
        self.calls.append((operation, *arguments))
        return {"result": operation}

    def health(self):
        return self.result("health")

    def areas(self):
        if self.failure:
            raise self.failure
        return [{"id": "area", "name": "Kanyama"}]

    def calendar(self, identifier, start, end):
        return self.result("calendar", identifier, start, end)

    def predict(self, identifier, issued):
        return self.result("predict", identifier, issued)

    def predictions(self, identifier=None):
        if self.failure:
            raise self.failure
        if identifier == "unknown":
            raise AreaNotFound("The selected area was not found.")
        self.calls.append(("predictions", identifier))
        return []

    def alerts(self):
        return self.result("alerts")

    def model_info(self):
        return self.result("model_info")

    def outlook(self, identifier, start, count):
        self.result("outlook", identifier, start, count)
        return self.outlook_response

    def overview(self, start):
        return self.result("overview", start)

    def record_request(self, record):
        if self.audit_failure:
            raise self.audit_failure
        self.requests.append(record)

    def prediction_requests(self, identifier=None):
        self.result("prediction_requests", identifier)
        return self.requests
