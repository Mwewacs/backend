from backend.audit_monthly import monthly_readiness
from backend.tests.given import given_records


def given_months(count=60, start="2020-01-01", cases=0):
    records = given_records([cases] * count, start=start)
    return {"areas": [{"id": records[0]["area_id"], "name": "Kanyama", "level": "local_area"}], "daily_observations": records}


def test_counts_complete_months_and_previous_month_pairs():
    report = monthly_readiness(given_months())
    assert [(record["month"], record["cases"], record["area_level"]) for record in report["complete_months"]] == [("2020-01", 0, "local_area"), ("2020-02", 0, "local_area")]
    assert [record["month"] for record in report["consecutive_month_targets"]] == ["2020-02"]
    assert report["monthly_model_trained"] is False
    assert report["monthly_outbreak_labels_verified"] is False
    assert report["active_horizon_days"] == 7


def test_incomplete_months_do_not_become_zero_case_months():
    report = monthly_readiness(given_months(28, start="2020-02-01"))
    assert report["complete_months"] == []
    assert report["consecutive_month_targets"] == []


def test_preserves_reported_positive_months():
    report = monthly_readiness(given_months(29, start="2020-02-01", cases=1))
    assert report["complete_months"][0]["cases"] == 29
    assert report["consecutive_month_targets"] == []


def test_missing_case_counts_do_not_complete_a_month():
    dataset = given_months(29, start="2020-02-01")
    dataset["daily_observations"][0]["cases"] = None
    assert monthly_readiness(dataset)["complete_months"] == []


def test_month_pairs_cross_the_year_boundary():
    report = monthly_readiness(given_months(62, start="2019-12-01"))
    assert [record["month"] for record in report["consecutive_month_targets"]] == ["2020-01"]
