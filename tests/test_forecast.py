import json
from datetime import date, datetime, timedelta, timezone

import joblib
import numpy as np
import pytest

from backend.cholera_forecast import CholeraOutlook
from backend.forecast_features import FEATURES, TARGET, case_windows, forecast_features
from backend.tests.given import GivenModel, given_records, given_service
from backend.train_forecast import chronological_validation, fit_forecast, frame
from backend.weather import WeatherProvider, forecast_days


def given_weather(start="2020-01-01", count=30, rainfall=1):
    first = date.fromisoformat(start)
    return {(first + timedelta(days=index)).isoformat(): {"date": (first + timedelta(days=index)).isoformat(), "rainfall_mm": rainfall, "temperature_c": 25, "basis": "recent_weather", "source": "Example weather", "source_url": "https://example.org/weather", "retrieved_at": "2020-01-20T09:00:00+00:00"} for index in range(count)}


def given_location():
    return {"latitude": -15.43, "longitude": 28.24, "baseline_key": "09", "name": "Kanyama", "scope": "named_place", "background": None}


def given_forecast_examples(count=90):
    first = date(2020, 1, 1)
    return [{"area_id": "area", "issue_date": (first + timedelta(days=index)).isoformat(), "window_end": (first + timedelta(days=index + 6)).isoformat(), "label": index % 2, "features": {**dict.fromkeys(FEATURES, 0.0), "rainfall_lag_week": (index % 2) * 10}} for index in range(count)]


class GivenWeatherProvider:
    def days(self, location, start, count, refresh=True):
        return list(given_weather(start, count).values())


class GivenForecastModel(GivenModel):
    def predict_proba(self, frame):
        return np.tile([[1 - self.score, self.score]], (len(frame), 1))


def given_engine(identifier, score=0.8):
    report = {"ready": True, "validated": False, "calibrated_probability": False, "version": "case-model", "selected_model": "random_forest", "target_definition": TARGET, "risk_bands": {"medium": 0.3, "high": 0.7}, "training_cutoff": "2019-12-31", "training_area_ids": [identifier]}
    return CholeraOutlook({"locations": {identifier: given_location()}}, report, GivenForecastModel(score), GivenWeatherProvider(), lambda: datetime(2020, 1, 20, 9, tzinfo=timezone.utc))


@pytest.mark.parametrize("counts,label", [([0] * 7, 0), ([0, 0, 1, 0, 0, 0, 0], 1)])
def test_labels_observed_cholera_windows(counts, label):
    records = given_records(counts)
    assert case_windows({"daily_observations": records, "model_observations": records}) == [{"area_id": records[0]["area_id"], "issue_date": "2020-01-01", "window_end": "2020-01-07", "label": label, "case_count": sum(counts), "frequency": "daily_aggregation"}]


def test_does_not_label_incomplete_case_windows_as_negative():
    records = given_records([0] * 6)
    assert case_windows({"daily_observations": records, "model_observations": records}) == []


def test_uses_reported_weekly_counts_without_inventing_daily_cases():
    records = given_records([3], period_days=7)
    assert case_windows({"daily_observations": [], "model_observations": records}) == [{"area_id": records[0]["area_id"], "issue_date": "2020-01-01", "window_end": "2020-01-07", "label": 1, "case_count": 3, "frequency": "reported_week"}]


def test_does_not_duplicate_daily_cases_as_weekly_labels():
    records = given_records([0] * 7)
    assert len(case_windows({"daily_observations": records, "model_observations": given_records([0], period_days=7)})) == 1


def test_features_use_only_the_specified_lagged_weather():
    weather = given_weather()
    weather["2020-01-20"]["rainfall_mm"] = 1000
    features = forecast_features(given_location(), "2020-01-20", weather)
    assert {name: features[name] for name in ["rainfall_lag_week", "rainfall_earlier_week", "temperature_lag_week", "temperature_earlier_week"]} == {"rainfall_lag_week": 7, "rainfall_earlier_week": 7, "temperature_lag_week": 25, "temperature_earlier_week": 25}


def test_refuses_incomplete_weather_features():
    assert forecast_features(given_location(), "2020-01-20", given_weather(start="2020-01-02")) is None


def test_background_cases_exclude_the_target_year():
    location = given_location()
    location["background"] = {"annual_history": [{"year": 2018, "cases": 0}, {"year": 2019, "cases": 1}, {"year": 2020, "cases": 100}]}
    assert forecast_features(location, "2020-01-20", given_weather())["previous_annual_occurrence"] == 0.5


def test_unknown_background_stays_missing():
    features = forecast_features(given_location(), "2020-01-20", given_weather())
    assert [features[name] for name in ["water_access", "sanitation_access", "previous_annual_occurrence"]] == [None, None, None]


def test_validation_outcomes_do_not_overlap_training_outcomes():
    training, validation, cutoff = chronological_validation(given_forecast_examples())
    assert max(example["window_end"] for example in training) < cutoff
    assert min(example["issue_date"] for example in validation) == cutoff
    assert {example["label"] for example in training} == {0, 1}
    assert {example["label"] for example in validation} == {0, 1}


def test_does_not_claim_validation_from_too_few_examples():
    assert chronological_validation(given_forecast_examples(10)) is None


def test_trains_models_on_case_labels_and_saves_a_research_forecaster(tmp_path):
    examples = given_forecast_examples()
    report = fit_forecast(examples, [], output_directory=tmp_path)
    model = joblib.load(tmp_path / "forecast-model.joblib")
    probabilities = model.predict_proba(frame(examples[:2]))[:, 1]
    assert probabilities[1] > probabilities[0]
    assert report["ready"] is True
    assert report["validated"] is False
    assert report["calibrated_probability"] is False
    assert report["data"]["examples"] == 90
    assert len(report["candidates"]) == 4
    assert report["evaluation"]["independent_test"] is None
    assert (tmp_path / report["artifact"]).is_file()


def test_does_not_train_when_only_one_cholera_outcome_exists(tmp_path):
    with pytest.raises(ValueError) as failure:
        fit_forecast(given_forecast_examples()[::2], [], output_directory=tmp_path)
    assert str(failure.value) == "Cholera training needs both reported-case and case-free windows."
    assert not (tmp_path / "forecast-model.joblib").exists()


@pytest.mark.parametrize("score,category", [(0.0, "low"), (0.299, "low"), (0.3, "medium"), (0.699, "medium"), (0.7, "high"), (1.0, "high")])
def test_outlook_is_a_cholera_model_prediction(score, category):
    service = given_service(ready=False)
    identifier = service.repository.selected_area["id"]
    service.outlook_engine = given_engine(identifier, score)
    result = service.outlook(identifier, "2020-01-20", 7)
    assert {name: result[name] for name in ["kind", "risk_score", "risk_category", "window_start", "window_end", "experimental", "calibrated_probability"]} == {"kind": "cholera_forecast", "risk_score": score, "risk_category": category, "window_start": "2020-01-20", "window_end": "2020-01-26", "experimental": True, "calibrated_probability": False}
    assert len(result["days"]) == 7
    assert result["days"][-1]["window_end"] == "2020-02-01"
    assert service.repository.saved == [result["prediction"]]
    assert service.repository.saved_outlooks == [result]


@pytest.mark.parametrize("score", [-0.1, 1.1, float("nan")])
def test_does_not_publish_invalid_cholera_model_scores(score):
    with pytest.raises(ValueError) as failure:
        given_engine("area", score).outlook({"id": "area", "name": "Kanyama", "region": "Lusaka", "level": "local_area"}, "2020-01-20")
    assert str(failure.value) == "The cholera model did not return a valid prediction."


def given_baselines():
    return {"09": {"name": "Lusaka", "source_url": "https://example.org/climate", "retrieved_at": "2026-01-01T00:00:00+00:00", "period": "Example climatology", "months": {str(month): {"temperature_c": 24, "rainfall_mm": 3} for month in range(1, 13)}}}


def given_weather_payload():
    dates = [(date(2026, 11, 6) + timedelta(days=index)).isoformat() for index in range(7)]
    return {"daily_units": {"temperature_2m_mean": "°C", "precipitation_sum": "mm"}, "daily": {"time": dates, "temperature_2m_mean": [25] * 7, "precipitation_sum": [0] * 7}}


class GivenFetcher:
    def __init__(self, payload=None, failure=None):
        self.payload = payload
        self.failure = failure
        self.calls = []

    def __call__(self, url):
        self.calls.append(url)
        if self.failure:
            raise self.failure
        return self.payload


def test_live_weather_zero_is_preserved(tmp_path):
    provider = WeatherProvider(given_baselines(), tmp_path, GivenFetcher(given_weather_payload()), lambda: datetime(2026, 11, 6, 9, tzinfo=timezone.utc))
    days = provider.days(given_location(), "2026-11-06", 7)
    assert [(day["rainfall_mm"], day["basis"]) for day in days] == [(0, "forecast")] * 7


def test_missing_weather_is_replaced_with_identified_seasonal_inputs(tmp_path):
    payload = given_weather_payload()
    payload["daily"]["precipitation_sum"][1] = None
    provider = WeatherProvider(given_baselines(), tmp_path, GivenFetcher(payload), lambda: datetime(2026, 11, 6, 9, tzinfo=timezone.utc))
    days = provider.days(given_location(), "2026-11-06", 7)
    assert [days[1]["rainfall_mm"], days[1]["basis"], days[1]["source"]] == [3, "seasonal_baseline", "NASA POWER"]
    assert days[0]["rainfall_mm"] == 0


def test_offline_weather_still_supplies_real_seasonal_inputs(tmp_path):
    provider = WeatherProvider(given_baselines(), tmp_path, GivenFetcher(failure=OSError("offline")), lambda: datetime(2026, 11, 6, 9, tzinfo=timezone.utc))
    assert [(day["temperature_c"], day["rainfall_mm"], day["basis"]) for day in provider.days(given_location(), "2026-11-06", 7)] == [(24, 3, "seasonal_baseline")] * 7


def test_weather_with_incompatible_units_is_not_used():
    payload = given_weather_payload()
    payload["daily_units"]["temperature_2m_mean"] = "K"
    with pytest.raises(ValueError) as failure:
        forecast_days(payload)
    assert str(failure.value) == "Weather units must be Celsius and millimetres."


def test_misaligned_weather_is_not_used():
    payload = given_weather_payload()
    payload["daily"]["precipitation_sum"].pop()
    with pytest.raises(ValueError) as failure:
        forecast_days(payload)
    assert str(failure.value) == "Weather dates and measurements must align."


def test_area_overviews_can_use_seasonal_inputs_without_network_requests(tmp_path):
    fetcher = GivenFetcher(failure=OSError("offline"))
    provider = WeatherProvider(given_baselines(), tmp_path, fetcher, lambda: datetime(2026, 11, 6, 9, tzinfo=timezone.utc))
    days = provider.days(given_location(), "2026-11-06", 7, refresh=False)
    assert [(day["temperature_c"], day["rainfall_mm"], day["basis"]) for day in days] == [(24, 3, "seasonal_baseline")] * 7
    assert fetcher.calls == []


def test_area_overviews_prefer_recent_saved_weather(tmp_path):
    fetcher = GivenFetcher(given_weather_payload())
    provider = WeatherProvider(given_baselines(), tmp_path, fetcher, lambda: datetime(2026, 11, 6, 9, tzinfo=timezone.utc))
    provider.days(given_location(), "2026-11-06", 7)
    provider.clock = lambda: datetime(2026, 11, 6, 13, tzinfo=timezone.utc)
    days = provider.days(given_location(), "2026-11-06", 7, refresh=False)
    assert [(day["temperature_c"], day["rainfall_mm"], day["basis"]) for day in days] == [(25, 0, "cached_weather")] * 7
    assert len(fetcher.calls) == 1


def test_area_overviews_do_not_use_expired_weather(tmp_path):
    fetcher = GivenFetcher(given_weather_payload())
    provider = WeatherProvider(given_baselines(), tmp_path, fetcher, lambda: datetime(2026, 11, 6, 9, tzinfo=timezone.utc))
    provider.days(given_location(), "2026-11-06", 7)
    provider.clock = lambda: datetime(2026, 11, 8, 10, tzinfo=timezone.utc)
    days = provider.days(given_location(), "2026-11-06", 7, refresh=False)
    assert [(day["temperature_c"], day["rainfall_mm"], day["basis"]) for day in days] == [(24, 3, "seasonal_baseline")] * 7
    assert len(fetcher.calls) == 1
