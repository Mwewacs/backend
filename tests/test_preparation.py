import pytest

from backend.prepare_data import build_examples, build_features, canonical_location, parse_observation, parse_surveillance, read_count, select_observations
from backend.tests.given import given_records


@pytest.mark.parametrize("value,expected", [("0", 0), ("3", 3), (None, None), ("", None)])
def test_reads_case_counts_without_turning_missing_into_zero(value, expected):
    assert read_count(value) == expected


@pytest.mark.parametrize("value", ["-1", "1.5", "invalid", "nan", "inf", True])
def test_rejects_invalid_case_counts(value):
    with pytest.raises(ValueError) as failure:
        read_count(value)
    assert str(failure.value) == "Case counts must be non-negative whole numbers."


def test_reconciles_province_codes():
    assert canonical_location("AFR::ZMB::ZM-09::Lusaka") == "AFR::ZMB::Lusaka::Lusaka"


@pytest.mark.parametrize("location", ["AFR::MWI::Lusaka", "AFR::ZMBevil", "", None])
def test_rejects_non_zambian_locations(location):
    with pytest.raises(ValueError) as failure:
        canonical_location(location)
    assert str(failure.value) == "A Zambian area identifier is required."


def test_deduplicates_agreeing_daily_reports():
    record = given_records([2])[0]
    selected, conflicts = select_observations([record, {**record, "collection_id": "other"}])
    assert len(selected) == 1
    assert selected[0]["collections"] == ["example", "other"]
    assert conflicts == []


def test_excludes_conflicting_daily_reports():
    record = given_records([2])[0]
    selected, conflicts = select_observations([record, {**record, "cases": 3}])
    assert selected == []
    assert conflicts == [{"area_id": record["area_id"], "date": "2020-01-01", "counts": [2, 3], "collections": ["example"]}]


def test_selects_weekly_records_separately():
    records = given_records([0, 0, 3], period_days=7)
    assert len(select_observations(records, 7)[0]) == 3
    assert select_observations(records)[0] == []


@pytest.mark.parametrize("field", ["age_restricted", "grouped"])
def test_excludes_incompatible_population_records(field):
    record = {**given_records([2])[0], field: True}
    assert select_observations([record]) == ([], [])


def test_builds_features_from_past_observations():
    records = given_records()
    assert build_features(records[0]["area_id"], "2020-01-15", records) == {"prior_cases": None, "month_sin": 0.0, "month_cos": 1.0, "peer_cases_last_7_days": 0, "peer_reporting_fraction": 0}


def test_ignores_future_peer_observations():
    records = given_records()
    future = given_records([999], start="2020-01-16", location="AFR::ZMB::Lusaka::Lusaka::Other")
    assert build_features(records[0]["area_id"], "2020-01-15", records + future) == build_features(records[0]["area_id"], "2020-01-15", records)


def test_excludes_records_published_after_the_issue_time():
    records = given_records()
    records[0]["available_at"] = "2020-01-16T00:00:00Z"
    assert build_features(records[0]["area_id"], "2020-01-15", records) is None


def test_does_not_fill_missing_history_with_zero():
    records = given_records()
    assert build_features(records[0]["area_id"], "2020-01-15", records[1:]) is None


def test_does_not_label_ongoing_transmission_as_a_new_onset():
    records = given_records()
    records[0]["cases"] = 2
    assert build_features(records[0]["area_id"], "2020-01-15", records) is None


def test_builds_a_future_onset_label():
    records = given_records()
    examples = build_examples(records)
    assert len(examples) == 1
    assert {name: examples[0][name] for name in ["issue_date", "window_end", "label", "onset_date"]} == {"issue_date": "2020-01-15", "window_end": "2020-01-21", "label": 1, "onset_date": "2020-01-16"}


def test_builds_an_observed_non_onset_label():
    examples = build_examples(given_records([0] * 21))
    assert [(example["label"], example["onset_date"]) for example in examples] == [(0, None)]


def test_requires_complete_future_outcomes_for_training():
    assert build_examples(given_records()[:-1]) == []


def test_builds_weekly_onset_examples():
    examples = build_examples(given_records([0, 0, 3], period_days=7))
    assert [(example["issue_date"], example["label"], example["onset_date"]) for example in examples] == [("2020-01-15", 1, "2020-01-15")]


def test_rejects_reversed_observation_periods():
    with pytest.raises(ValueError) as failure:
        parse_observation({"Location": "AFR::ZMB", "TL": "2020-01-02", "TR": "2020-01-01"}, "example")
    assert str(failure.value) == "The observation period ends before it starts."


def given_surveillance(**changes):
    return {"area_identifier": "area", "area_name": "Kanyama", "area_level": "subdistrict", "period_start": "2020-01-01", "period_end": "2020-01-07", "suspected_cases": "0", "reporting_status": "reported", "date_basis": "onset", "available_at": "2020-01-08T00:00:00Z", "source_reference": "https://example.org/report", **changes}


def test_imports_curated_weekly_records():
    record = parse_surveillance(given_surveillance())
    assert {name: record[name] for name in ["area_id", "cases", "duration_days", "date_basis", "source_url"]} == {"area_id": "area", "cases": 0, "duration_days": 7, "date_basis": "onset", "source_url": "https://example.org/report"}


def test_preserves_an_explicit_missing_report():
    assert parse_surveillance(given_surveillance(reporting_status="not_reported", suspected_cases=""))["cases"] is None


@pytest.mark.parametrize("changes,message", [({"area_identifier": ""}, "A required surveillance field is missing."), ({"reporting_status": "invalid"}, "The reporting status must be reported or not_reported."), ({"date_basis": "unknown"}, "The date basis must be onset, presentation or reporting."), ({"area_level": "country"}, "Use a district, subdistrict, local_area or facility area level."), ({"reporting_status": "not_reported"}, "Reported records need case counts; missing reports need blank counts.")])
def test_rejects_invalid_curated_records(changes, message):
    with pytest.raises(ValueError) as failure:
        parse_surveillance(given_surveillance(**changes))
    assert str(failure.value) == message
