from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from backend.cholera_forecast import CholeraOutlook
from backend.forecast_features import TARGET
from backend.service import ForecastUnavailable
from backend.tests.given import given_service
from backend.tests.test_forecast import GivenWeatherProvider, given_location


class GivenAreaModel:
    def __init__(self, scores):
        self.scores = scores

    def predict_proba(self, frame):
        return np.array([[1 - self.scores[value], self.scores[value]] for value in frame["latitude"]])


def given_overview_service():
    service = given_service(ready=False)
    original = service.repository.selected_area
    service.repository.listed_areas = [{**original, "id": name.lower(), "name": name} for name in ["First", "Second", "Third", "Unmapped"]]
    locations = {name.lower(): {**given_location(), "latitude": latitude} for name, latitude in [("First", -15), ("Second", -14), ("Third", -13)]}
    report = {"ready": True, "validated": False, "calibrated_probability": False, "version": "model-1", "selected_model": "random_forest", "target_definition": TARGET, "risk_bands": {"medium": 0.3, "high": 0.7}}
    service.outlook_engine = CholeraOutlook({"locations": locations}, report, GivenAreaModel({-15: 0.7, -14: 0.9, -13: 0.2}), GivenWeatherProvider(), lambda: datetime(2020, 1, 20, 9, tzinfo=timezone.utc))
    return service


def test_forecasts_all_mapped_areas_and_sorts_highest_risk_first():
    service = given_overview_service()
    result = service.overview("2020-01-20")
    assert [(area["area_name"], area["risk_score"], area["risk_category"]) for area in result["areas"]] == [("Second", 0.9, "high"), ("First", 0.7, "high"), ("Third", 0.2, "low")]
    assert (result["window_start"], result["window_end"], result["model_version"]) == ("2020-01-20", "2020-01-26", "model-1")
    assert len(service.repository.saved) == 3


def test_does_not_assign_low_risk_to_an_unmapped_area():
    service = given_overview_service()
    result = service.overview("2020-01-20")
    assert [area["area_id"] for area in result["areas"]] == ["second", "first", "third"]


def test_alerts_cover_every_mapped_area_and_are_ranked():
    response = given_overview_service().alerts()
    assert response["checked_areas"] == 3
    assert [(area["area_id"], area["risk_score"]) for area in response["alerts"]] == [("second", 0.9), ("first", 0.7)]
    assert (response["window_start"], response["window_end"]) == ("2020-01-20", "2020-01-26")


def test_reuses_a_recent_overview_without_duplicate_inference():
    service = given_overview_service()
    first = service.overview("2020-01-20")
    assert service.overview("2020-01-20") is first
    assert len(service.repository.saved) == 3


def test_does_not_reuse_an_expired_overview():
    service = given_overview_service()
    first = service.overview("2020-01-20")
    service.outlook_engine.clock = lambda: datetime(2020, 1, 20, 12, tzinfo=timezone.utc)
    assert service.overview("2020-01-20") is not first
    assert len(service.repository.saved) == 6


def test_does_not_reuse_another_models_overview():
    service = given_overview_service()
    service.overview("2020-01-20")
    service.outlook_engine.report["version"] = "model-2"
    assert service.overview("2020-01-20")["model_version"] == "model-2"
    assert len(service.repository.saved) == 6


def test_unavailable_overview_is_not_an_all_clear():
    with pytest.raises(ForecastUnavailable) as failure:
        given_service(ready=False).overview("2020-01-20")
    assert str(failure.value) == "The area forecast model is not available."
