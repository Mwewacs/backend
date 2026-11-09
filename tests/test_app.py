import sqlite3
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from backend.app import create_app
from backend.service import ForecastUnavailable
from backend.tests.given import GivenApiService


@pytest.fixture
def service():
    return GivenApiService()


@pytest.fixture
def client(service):
    return create_app(service).test_client()


@pytest.mark.parametrize("path,operation", [("/health", "health"), ("/alerts", "alerts"), ("/model/info", "model_info")])
def test_returns_service_results(client, path, operation):
    response = client.get(path)
    assert response.status_code == 200
    assert response.json == {"result": operation}


def test_lists_areas(client):
    response = client.get("/areas")
    assert response.status_code == 200
    assert response.json == {"areas": [{"id": "area", "name": "Kanyama"}]}


def test_returns_current_cholera_outlooks(client, service):
    today = datetime.now(ZoneInfo("Africa/Lusaka")).date().isoformat()
    response = client.get("/outlook?area_id=area")
    assert response.status_code == 200
    assert response.json == {"result": "outlook"}
    assert service.calls == [("outlook", "area", today, 7)]


@pytest.mark.parametrize("count", ["0", "32", "-1", "7.0", "invalid"])
def test_rejects_invalid_outlook_lengths(client, service, count):
    response = client.get("/outlook", query_string={"area_id": "area", "days": count})
    assert response.status_code == 400
    assert response.json == {"error": {"code": "invalid_input", "message": "Select an outlook length of 1 to 31 days."}}
    assert service.calls == []


def test_does_not_present_historical_dates_as_current_outlooks(client, service):
    yesterday = (datetime.now(ZoneInfo("Africa/Lusaka")).date() - timedelta(days=1)).isoformat()
    response = client.get("/outlook", query_string={"area_id": "area", "from": yesterday})
    assert response.status_code == 400
    assert response.json == {"error": {"code": "invalid_input", "message": "Select outlook dates from today through the next 366 days."}}
    assert service.calls == []


def test_rejects_unknown_outlook_areas(client):
    response = client.get("/outlook?area_id=unknown")
    assert response.status_code == 404
    assert response.json == {"error": {"code": "area_not_found", "message": "The selected area was not found."}}


def test_returns_an_empty_area_list(client, service):
    service.areas = lambda: []
    response = client.get("/areas")
    assert response.status_code == 200
    assert response.json == {"areas": []}


@pytest.mark.parametrize("path", ["/health", "/areas", "/alerts", "/model/info", "/predictions", "/calendar?area_id=area&from=2020-01-01&to=2020-01-31"])
def test_returns_a_database_failure(client, service, path):
    service.failure = sqlite3.OperationalError("private connection details")
    response = client.get(path)
    assert response.status_code == 503
    assert response.json == {"error": {"code": "service_unavailable", "message": "The data service is temporarily unavailable."}}


def test_returns_a_valid_calendar_request(client, service):
    response = client.get("/calendar?area_id=area&from=2020-01-01&to=2020-01-31")
    assert response.status_code == 200
    assert response.json == {"result": "calendar"}
    assert service.calls == [("calendar", "area", "2020-01-01", "2020-01-31")]


@pytest.mark.parametrize("query,message", [("from=2020-01-01&to=2020-01-31", "An area identifier is required."), ("area_id=area&from=20200101&to=2020-01-31", "Use dates in YYYY-MM-DD format."), ("area_id=area&from=2023-02-29&to=2023-03-01", "Use valid calendar dates."), ("area_id=area&from=2020-02-01&to=2020-01-01", "Select an ordered date range of at most 366 days."), ("area_id=area&from=2020-01-01&to=2022-01-01", "Select an ordered date range of at most 366 days.")])
def test_rejects_invalid_calendar_requests(client, service, query, message):
    response = client.get(f"/calendar?{query}")
    assert response.status_code == 400
    assert response.json == {"error": {"code": "invalid_input", "message": message}}
    assert service.calls == []


def test_accepts_a_valid_leap_day(client):
    response = client.get("/calendar?area_id=area&from=2024-02-29&to=2024-02-29")
    assert response.status_code == 200
    assert response.json == {"result": "calendar"}


def test_returns_unknown_area_errors(client):
    response = client.get("/calendar?area_id=unknown&from=2020-01-01&to=2020-01-31")
    assert response.status_code == 404
    assert response.json == {"error": {"code": "area_not_found", "message": "The selected area was not found."}}


def test_returns_and_logs_a_prediction_request(client, service):
    response = client.post("/predict", json={"area_id": "area", "issue_date": "2020-01-15"})
    assert response.status_code == 201
    assert response.json == {"result": "predict"}
    assert service.calls == [("predict", "area", "2020-01-15")]


@pytest.mark.parametrize("payload,message", [(None, "A JSON object is required."), ([], "A JSON object is required."), ({}, "An area identifier is required."), ({"area_id": "area"}, "Use dates in YYYY-MM-DD format."), ({"area_id": "area", "issue_date": "2020-01-15", "horizon_days": 14}, "Only a 7-day forecast horizon is supported.")])
def test_rejects_invalid_prediction_requests(client, service, payload, message):
    response = client.post("/predict", json=payload)
    assert response.status_code == 400
    assert response.json == {"error": {"code": "invalid_input", "message": message}}
    assert service.calls == []


def test_explains_unavailable_forecasts(client, service):
    service.failure = ForecastUnavailable("Independent evaluation is required.")
    response = client.post("/predict", json={"area_id": "area", "issue_date": "2020-01-15"})
    assert response.status_code == 422
    assert response.json == {"error": {"code": "forecast_unavailable", "message": "Independent evaluation is required."}}


def test_rejects_malformed_json(client):
    response = client.post("/predict", data="{", content_type="application/json")
    assert response.status_code == 400
    assert response.json == {"error": {"code": "invalid_input", "message": "A JSON object is required."}}


def test_handles_a_prediction_logging_failure(client, service):
    service.failure = sqlite3.OperationalError("private path")
    response = client.post("/predict", json={"area_id": "area", "issue_date": "2020-01-15"})
    assert response.status_code == 503
    assert response.json == {"error": {"code": "service_unavailable", "message": "The data service is temporarily unavailable."}}


def test_returns_unknown_areas_for_predictions(client):
    response = client.post("/predict", json={"area_id": "unknown", "issue_date": "2020-01-15"})
    assert response.status_code == 404
    assert response.json == {"error": {"code": "area_not_found", "message": "The selected area was not found."}}


def test_lists_prediction_history(client, service):
    response = client.get("/predictions?area_id=area")
    assert response.status_code == 200
    assert response.json == {"predictions": []}
    assert service.calls == [("predictions", "area")]


def test_rejects_unknown_prediction_history_areas(client):
    response = client.get("/predictions?area_id=unknown")
    assert response.status_code == 404
    assert response.json == {"error": {"code": "area_not_found", "message": "The selected area was not found."}}


def test_rejects_blank_prediction_history_areas(client):
    response = client.get("/predictions?area_id=")
    assert response.status_code == 400
    assert response.json == {"error": {"code": "invalid_input", "message": "An area identifier is required."}}


def test_returns_json_for_unknown_routes(client):
    response = client.get("/unknown")
    assert response.status_code == 404
    assert response.json == {"error": {"code": "not_found", "message": "The requested resource was not found."}}


def test_returns_json_for_unsupported_methods(client):
    response = client.post("/areas")
    assert response.status_code == 405
    assert response.json == {"error": {"code": "method_not_allowed", "message": "The requested method is not supported."}}
