import pytest

from backend.service import AreaNotFound, ForecastUnavailable
from backend.tests.given import given_records, given_service


def test_calendar_distinguishes_reported_zero_from_missing_data():
    service = given_service(records=given_records([0, 2]))
    response = service.calendar(service.repository.selected_area["id"], "2020-01-01", "2020-01-03")
    assert [(day["date"], day["cases"], day["status"]) for day in response["days"]] == [("2020-01-01", 0, "reported"), ("2020-01-02", 2, "reported"), ("2020-01-03", None, "not_reported")]


def test_calendar_exposes_conflicted_dates_without_a_case_count():
    service = given_service(records=given_records([0]))
    identifier = service.repository.selected_area["id"]
    service.dataset["conflicts"] = [{"area_id": identifier, "date": "2020-01-02"}]
    response = service.calendar(identifier, "2020-01-02", "2020-01-02")
    assert response["days"] == [{"date": "2020-01-02", "cases": None, "status": "conflict", "source_url": None}]


def test_rejects_unknown_areas():
    with pytest.raises(AreaNotFound) as failure:
        given_service().calendar("unknown", "2020-01-01", "2020-01-02")
    assert str(failure.value) == "The selected area was not found."


@pytest.mark.parametrize("score,category", [(0.0, "low"), (0.299, "low"), (0.3, "medium"), (0.699, "medium"), (0.7, "high"), (1.0, "high")])
def test_predicts_and_logs_exact_risk_boundaries(score, category):
    service = given_service(score=score)
    prediction = service.predict(service.repository.selected_area["id"], "2020-01-15")
    assert {name: prediction[name] for name in ["risk_score", "risk_category", "window_start", "window_end", "data_cutoff", "model_version", "mode"]} == {"risk_score": score, "risk_category": category, "window_start": "2020-01-15", "window_end": "2020-01-21", "data_cutoff": "2020-01-14", "model_version": "model-1", "mode": "historical_replay"}
    assert service.repository.saved == [prediction]


@pytest.mark.parametrize("score", [-0.1, 1.1, float("nan"), float("inf")])
def test_rejects_invalid_scores_without_logging(score):
    service = given_service(score=score)
    with pytest.raises(ForecastUnavailable) as failure:
        service.predict(service.repository.selected_area["id"], "2020-01-15")
    assert str(failure.value) == "The model did not return a valid risk score."
    assert service.repository.saved == []


def test_refuses_unvalidated_models():
    service = given_service(ready=False)
    with pytest.raises(ForecastUnavailable) as failure:
        service.predict(service.repository.selected_area["id"], "2020-01-15")
    assert str(failure.value) == "A timing forecast is not available until the model has passed independent evaluation with dated source records."
    assert service.repository.saved == []


def test_refuses_in_sample_forecast_dates():
    service = given_service()
    with pytest.raises(ForecastUnavailable) as failure:
        service.predict(service.repository.selected_area["id"], "2019-12-31")
    assert str(failure.value) == "Select a forecast date after the model training cutoff."


def test_refuses_missing_case_free_history():
    service = given_service(records=given_records()[1:])
    with pytest.raises(ForecastUnavailable) as failure:
        service.predict(service.repository.selected_area["id"], "2020-01-15")
    assert str(failure.value) == "Complete case-free history is not available for this area and forecast date."


def test_historical_replays_do_not_become_live_alerts():
    service = given_service()
    service.predict(service.repository.selected_area["id"], "2020-01-15")
    assert service.alerts() == {"forecast": {"status": "available", "reason": None}, "alerts": []}


def test_unavailable_alerts_are_not_an_all_clear():
    response = given_service(ready=False).alerts()
    assert response == {"forecast": {"status": "unavailable", "reason": "A timing forecast is not available until the model has passed independent evaluation with dated source records."}, "alerts": []}
