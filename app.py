import re
import sqlite3
from datetime import date, datetime, timedelta, timezone
from uuid import uuid4
from zoneinfo import ZoneInfo

from flask import Flask, jsonify, request
from flask_cors import CORS

from backend.service import AreaNotFound, ForecastService, ForecastUnavailable


def valid_date(value):
    if not isinstance(value, str) or re.fullmatch(r"\d{4}-\d{2}-\d{2}", value) is None:
        raise ValueError("Use dates in YYYY-MM-DD format.")
    try:
        return date.fromisoformat(value)
    except ValueError:
        raise ValueError("Use valid calendar dates.") from None


def require_area(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("An area identifier is required.")
    return value.strip()


def create_app(service=None):
    app = Flask(__name__)
    CORS(app)
    forecast_service = service if service is not None else ForecastService.from_files()

    @app.after_request
    def record_prediction_request(response):
        if request.method == "OPTIONS" or request.path not in ["/predict", "/outlook", "/overview", "/alerts"]:
            return response
        payload = request.get_json(silent=True) if request.method == "POST" else request.args.to_dict()
        payload = payload if isinstance(payload, dict) else {}
        result = response.get_json(silent=True) or {}
        prediction = result.get("prediction", result)
        error = result.get("error", {})
        model = result.get("model")
        model_version = prediction.get("model_version") or (model.get("version") if isinstance(model, dict) else None)
        record = {"id": uuid4().hex, "requested_at": datetime.now(timezone.utc).isoformat(), "method": request.method, "path": request.path, "area_id": str(payload["area_id"])[:256] if "area_id" in payload else None, "issue_date": str(payload.get("issue_date", payload.get("from", result.get("window_start", ""))))[:32], "days": str(payload.get("days", "7"))[:32], "status_code": response.status_code, "prediction_id": prediction.get("id"), "outlook_id": result.get("id") if "prediction" in result else None, "model_version": model_version, "window_start": prediction.get("window_start"), "window_end": prediction.get("window_end"), "risk_score": prediction.get("risk_score"), "risk_category": prediction.get("risk_category"), "error_code": error.get("code"), "error_message": error.get("message")}
        try:
            forecast_service.record_request(record)
        except sqlite3.Error:
            failure = jsonify({"error": {"code": "service_unavailable", "message": "The prediction request could not be recorded. Please try again."}})
            failure.status_code = 503
            return failure
        return response

    @app.errorhandler(ValueError)
    def invalid_input(error):
        return jsonify({"error": {"code": "invalid_input", "message": str(error)}}), 400

    @app.errorhandler(AreaNotFound)
    def unknown_area(error):
        return jsonify({"error": {"code": "area_not_found", "message": str(error)}}), 404

    @app.errorhandler(ForecastUnavailable)
    def unavailable_forecast(error):
        return jsonify({"error": {"code": "forecast_unavailable", "message": str(error)}}), 422

    @app.errorhandler(sqlite3.Error)
    def unavailable_database(error):
        return jsonify({"error": {"code": "service_unavailable", "message": "The data service is temporarily unavailable."}}), 503

    @app.errorhandler(404)
    def unknown_route(error):
        return jsonify({"error": {"code": "not_found", "message": "The requested resource was not found."}}), 404

    @app.errorhandler(405)
    def unsupported_method(error):
        return jsonify({"error": {"code": "method_not_allowed", "message": "The requested method is not supported."}}), 405

    @app.get("/health")
    def health():
        return jsonify(forecast_service.health())

    @app.get("/areas")
    def areas():
        return jsonify({"areas": forecast_service.areas()})

    @app.get("/calendar")
    def calendar():
        identifier = require_area(request.args.get("area_id"))
        start = valid_date(request.args.get("from"))
        end = valid_date(request.args.get("to"))
        if end < start or (end - start).days > 365:
            raise ValueError("Select an ordered date range of at most 366 days.")
        return jsonify(forecast_service.calendar(identifier, start.isoformat(), end.isoformat()))

    @app.post("/predict")
    def predict():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            raise ValueError("A JSON object is required.")
        identifier = require_area(payload.get("area_id"))
        issue_date = valid_date(payload.get("issue_date"))
        if payload.get("horizon_days", 7) != 7:
            raise ValueError("Only a 7-day forecast horizon is supported.")
        return jsonify(forecast_service.predict(identifier, issue_date.isoformat())), 201

    @app.get("/predictions")
    def predictions():
        identifier = request.args.get("area_id")
        if identifier is not None:
            identifier = require_area(identifier)
        return jsonify({"predictions": forecast_service.predictions(identifier)})

    @app.get("/alerts")
    def alerts():
        return jsonify(forecast_service.alerts())

    @app.get("/model/info")
    def model_info():
        return jsonify(forecast_service.model_info())

    @app.get("/prediction-requests")
    def prediction_requests():
        identifier = request.args.get("area_id")
        if identifier is not None:
            identifier = require_area(identifier)
        return jsonify({"requests": forecast_service.prediction_requests(identifier)})

    @app.get("/overview")
    def overview():
        current_day = datetime.now(ZoneInfo("Africa/Lusaka")).date()
        start = valid_date(request.args.get("from", current_day.isoformat()))
        if start < current_day or start > current_day + timedelta(days=366):
            raise ValueError("Select overview dates from today through the next 366 days.")
        return jsonify(forecast_service.overview(start.isoformat()))

    @app.get("/outlook")
    def outlook():
        identifier = require_area(request.args.get("area_id"))
        current_day = datetime.now(ZoneInfo("Africa/Lusaka")).date()
        start = valid_date(request.args.get("from", current_day.isoformat()))
        count = request.args.get("days", "7")
        if not count.isascii() or not count.isdecimal() or not 1 <= int(count) <= 31:
            raise ValueError("Select an outlook length of 1 to 31 days.")
        if start < current_day or start + timedelta(days=int(count) - 1) > current_day + timedelta(days=366):
            raise ValueError("Select outlook dates from today through the next 366 days.")
        return jsonify(forecast_service.outlook(identifier, start.isoformat(), int(count)))

    return app
