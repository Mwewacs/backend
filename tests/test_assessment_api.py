import sqlite3
from datetime import datetime, timedelta
from uuid import UUID
from zoneinfo import ZoneInfo

import pytest

from backend.app import create_app
from backend.tests.given import GivenApiService


def given_client():
    service = GivenApiService()
    return create_app(service).test_client(), service


def test_returns_the_current_area_overview():
    client, service = given_client()
    today = datetime.now(ZoneInfo("Africa/Lusaka")).date().isoformat()
    response = client.get("/overview")
    assert response.status_code == 200
    assert response.json == {"result": "overview"}
    assert service.calls == [("overview", today)]
    assert service.requests[0]["status_code"] == 200


@pytest.mark.parametrize("offset", [-1, 367])
def test_rejects_overview_dates_outside_the_forecast_range(offset):
    client, service = given_client()
    date = (datetime.now(ZoneInfo("Africa/Lusaka")).date() + timedelta(days=offset)).isoformat()
    response = client.get("/overview", query_string={"from": date})
    assert response.status_code == 400
    assert response.json == {"error": {"code": "invalid_input", "message": "Select overview dates from today through the next 366 days."}}
    assert service.calls == []
    assert service.requests[0]["status_code"] == 400


def test_records_the_complete_prediction_request_metadata():
    client, service = given_client()
    service.outlook_response = {"id": "outlook-1", "prediction": {"id": "prediction-1", "model_version": "model-1", "window_start": "2026-11-06", "window_end": "2026-11-12", "risk_score": 0.8, "risk_category": "high"}}
    response = client.get("/outlook?area_id=area")
    record = service.requests[0]
    assert response.status_code == 200
    assert UUID(record["id"]).hex == record["id"]
    assert datetime.fromisoformat(record["requested_at"]).utcoffset() == timedelta(0)
    assert record == {"id": record["id"], "requested_at": record["requested_at"], "method": "GET", "path": "/outlook", "area_id": "area", "issue_date": "", "days": "7", "status_code": 200, "prediction_id": "prediction-1", "outlook_id": "outlook-1", "model_version": "model-1", "window_start": "2026-11-06", "window_end": "2026-11-12", "risk_score": 0.8, "risk_category": "high", "error_code": None, "error_message": None}


def test_repeated_forecast_requests_have_distinct_audit_identifiers():
    client, service = given_client()
    client.get("/outlook?area_id=area")
    client.get("/outlook?area_id=area")
    assert len(service.requests) == 2
    assert service.requests[0]["id"] != service.requests[1]["id"]


def test_invalid_prediction_inputs_are_recorded_without_inference():
    client, service = given_client()
    response = client.post("/predict", json={})
    assert response.status_code == 400
    assert service.calls == []
    record = service.requests[0]
    assert (record["prediction_id"], record["risk_score"], record["error_code"], record["error_message"], record["status_code"]) == (None, None, "invalid_input", "An area identifier is required.", 400)


def test_preflight_requests_are_not_prediction_requests():
    client, service = given_client()
    response = client.options("/outlook")
    assert response.status_code == 200
    assert service.requests == []


def test_reading_the_log_does_not_create_another_prediction_request():
    client, service = given_client()
    response = client.get("/prediction-requests?area_id=area")
    assert response.status_code == 200
    assert response.json == {"requests": []}
    assert service.calls == [("prediction_requests", "area")]
    assert service.requests == []


@pytest.mark.parametrize("identifier,status,message", [("", 400, "An area identifier is required."), ("unknown", 404, "The selected area was not found.")])
def test_rejects_invalid_log_area_filters(identifier, status, message):
    client, service = given_client()
    response = client.get("/prediction-requests", query_string={"area_id": identifier})
    assert response.status_code == status
    assert response.json == {"error": {"code": "invalid_input" if status == 400 else "area_not_found", "message": message}}
    assert service.requests == []


def test_does_not_report_success_when_the_audit_cannot_be_saved():
    client, service = given_client()
    service.audit_failure = sqlite3.OperationalError("private database path")
    response = client.get("/outlook?area_id=area")
    assert response.status_code == 503
    assert response.json == {"error": {"code": "service_unavailable", "message": "The prediction request could not be recorded. Please try again."}}
    assert service.requests == []


@pytest.mark.parametrize("path", ["/overview", "/prediction-requests"])
def test_returns_safe_database_errors_for_assessment_views(path):
    client, service = given_client()
    service.failure = sqlite3.OperationalError("private database path")
    response = client.get(path)
    assert response.status_code == 503
    assert response.json == {"error": {"code": "service_unavailable", "message": "The data service is temporarily unavailable."}}
