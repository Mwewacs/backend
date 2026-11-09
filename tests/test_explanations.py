import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from backend.explanations import forecast_reasons, forest_effects
from backend.forecast_features import FEATURES
from backend.tests.given import GivenModel, given_service
from backend.tests.test_forecast import given_engine


def given_inputs(rainfall=0):
    return dict(zip(FEATURES, [rainfall, 0, 25, 25, 0, 1, -15, 28, 80, 60, 0.5]))


def given_explanation_model():
    model = Pipeline([("imputer", SimpleImputer()), ("classifier", RandomForestClassifier(n_estimators=3, max_depth=1, max_features=None, bootstrap=False, random_state=42))])
    return model.fit(pd.DataFrame([given_inputs(), given_inputs(10)], columns=FEATURES), [0, 1])


@pytest.mark.parametrize("rainfall,direction,effect", [(0, "lowers", -0.5), (10, "raises", 0.5), (5.000000000001, "lowers", -0.5)])
def test_reasons_follow_actual_tree_decisions_and_reconstruct_the_score(rainfall, direction, effect):
    model = given_explanation_model()
    inputs = given_inputs(rainfall)
    result = forest_effects(model, [inputs])
    contributions = {**dict.fromkeys(FEATURES, 0.0), "rainfall_lag_week": effect}
    score = model.predict_proba(pd.DataFrame([inputs], columns=FEATURES))[0][1]
    assert result == {"method": "forest_path", "baseline_score": 0.5, "contributions": [contributions]}
    assert result["baseline_score"] + sum(contributions.values()) == score
    assert forecast_reasons(inputs, "2026-11-06", contributions=contributions) == [{"key": "rainfall", "title": "Rainfall", "detail": f"{rainfall:.1f} mm across the 14 weather-input days.", "direction": direction, "contribution": effect}]


def test_does_not_invent_directional_reasons_for_an_unsupported_model():
    assert forest_effects(GivenModel(), [given_inputs()]) is None
    reasons = forecast_reasons(given_inputs(), "2026-11-06")
    assert [(reason["key"], reason["direction"], reason["contribution"]) for reason in reasons] == [("rainfall", "input", None), ("temperature", "input", None), ("season", "input", None)]


def test_ranks_the_three_strongest_local_input_groups():
    contributions = {**dict.fromkeys(FEATURES, 0.0), "latitude": 0.4, "rainfall_lag_week": 0.3, "temperature_lag_week": -0.2, "month_sin": 0.1}
    reasons = forecast_reasons(given_inputs(), "2026-11-06", contributions=contributions)
    assert [(reason["key"], reason["direction"], reason["contribution"]) for reason in reasons] == [("location", "raises", 0.4), ("rainfall", "raises", 0.3), ("temperature", "lowers", -0.2)]


def test_labels_seasonal_weather_without_claiming_recent_measurements():
    reasons = forecast_reasons(given_inputs(10), "2027-02-01", seasonal=True)
    assert reasons == [{"key": "rainfall", "title": "Seasonal rainfall", "detail": "10.0 mm across the 14 weather-input days.", "direction": "input", "contribution": None}, {"key": "temperature", "title": "Seasonal temperature", "detail": "25.0°C average across the 14 weather-input days.", "direction": "input", "contribution": None}, {"key": "season", "title": "Seasonal pattern", "detail": "The February pattern learned from past reported cases.", "direction": "input", "contribution": None}]


def test_preserves_reported_zero_water_access_in_the_reason():
    inputs = {**given_inputs(), "water_access": 0}
    contributions = {**dict.fromkeys(FEATURES, 0.0), "water_access": 0.2}
    assert forecast_reasons(inputs, "2026-11-06", contributions=contributions) == [{"key": "water", "title": "Historical water access", "detail": "0.0% improved water access in the historical source.", "direction": "raises", "contribution": 0.2}]


def test_missing_water_access_is_not_described_as_observed_zero_access():
    inputs = {**given_inputs(), "water_access": None}
    contributions = {**dict.fromkeys(FEATURES, 0.0), "water_access": 0.2}
    assert forecast_reasons(inputs, "2026-11-06", contributions=contributions) == [{"key": "water", "title": "Historical water access", "detail": "No dated water-access record; the model uses its training median.", "direction": "input", "contribution": 0.2}]


@pytest.mark.parametrize("effect", [0.0, 0.0001, float("nan")])
def test_does_not_claim_an_effect_from_zero_tiny_or_invalid_contributions(effect):
    contributions = {**dict.fromkeys(FEATURES, 0.0), "rainfall_lag_week": effect}
    assert forecast_reasons(given_inputs(), "2026-11-06", contributions=contributions) == forecast_reasons(given_inputs(), "2026-11-06")


def test_saves_each_forecast_windows_explanation_with_its_prediction():
    service = given_service(ready=False)
    identifier = service.repository.selected_area["id"]
    service.outlook_engine = given_engine(identifier)
    service.outlook_engine.model = given_explanation_model()
    result = service.outlook(identifier, "2020-01-20", 2)
    assert [day["explanation"]["method"] for day in result["days"]] == ["forest_path", "forest_path"]
    assert [day["reasons"] for day in result["days"]] == [[{"key": "rainfall", "title": "Rainfall", "detail": "14.0 mm across the 14 weather-input days.", "direction": "raises", "contribution": 0.5}]] * 2
    assert result["prediction"]["reasons"] == result["days"][0]["reasons"]
    assert service.repository.saved[0]["explanation"] == result["days"][0]["explanation"]
