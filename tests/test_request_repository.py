from backend.database import Repository


def given_request(identifier="request-1", area="area-1", requested="2026-11-06T09:00:00+00:00"):
    return {"id": identifier, "area_id": area, "requested_at": requested, "status_code": 200, "prediction_id": "prediction-1"}


def test_records_repeated_requests_without_deduplicating_them(tmp_path):
    repository = Repository(tmp_path / "audit.sqlite3")
    first = given_request()
    second = given_request("request-2", requested="2026-11-06T09:00:01+00:00")
    repository.save_prediction_request(first)
    repository.save_prediction_request(second)
    assert repository.prediction_requests() == [second, first]


def test_filters_requests_by_the_exact_area_identifier(tmp_path):
    repository = Repository(tmp_path / "audit.sqlite3")
    selected = given_request()
    repository.save_prediction_request(selected)
    repository.save_prediction_request(given_request("request-2", area="area-2"))
    assert repository.prediction_requests("area-1") == [selected]
    assert repository.prediction_requests("missing") == []


def test_unknown_area_inputs_can_be_audited_without_a_foreign_key(tmp_path):
    repository = Repository(tmp_path / "audit.sqlite3")
    record = {**given_request(area="unrecognised"), "status_code": 404, "prediction_id": None}
    repository.save_prediction_request(record)
    assert repository.prediction_requests("unrecognised") == [record]


def test_limits_the_log_to_the_latest_hundred_requests(tmp_path):
    repository = Repository(tmp_path / "audit.sqlite3")
    for index in range(101):
        repository.save_prediction_request(given_request(f"request-{index:03d}"))
    records = repository.prediction_requests()
    assert len(records) == 100
    assert records[0] == given_request("request-100")
    assert records[-1] == given_request("request-001")


def test_empty_request_logs_are_empty_lists(tmp_path):
    repository = Repository(tmp_path / "audit.sqlite3")
    assert repository.prediction_requests() == []
