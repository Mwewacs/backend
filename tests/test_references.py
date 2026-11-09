import pytest

from backend.references import forecast_references, source_reference
from backend.tests.given import given_service
from backend.tests.test_explanations import given_inputs
from backend.tests.test_forecast import given_engine, given_location, given_weather


def given_references(weather=None, location=None, inputs=None, report=None):
    return forecast_references(list(given_weather(count=2).values()) if weather is None else weather, given_location() if location is None else location, given_inputs() if inputs is None else inputs, {} if report is None else report)


def given_background():
    return {"name": "Lusaka", "scope": "parent_district", "source_url": "https://journals.plos.org/plosntds/article?id=10.1371/journal.pntd.0008227"}


def test_builds_a_linkable_reference_with_its_original_metadata():
    reference = source_reference("Source", "https://example.org/data", "forecast", "Weather inputs.", retrieved_at="2026-11-06T09:00:00Z")
    assert len(reference["id"]) == 12
    assert int(reference["id"], 16) >= 0
    assert reference == {"id": reference["id"], "name": "Source", "url": "https://example.org/data", "role": "forecast", "description": "Weather inputs.", "retrieved_at": "2026-11-06T09:00:00Z"}


@pytest.mark.parametrize("url", [None, "", "javascript:alert(1)", "file:///data", "https://", "https://[", "https://user:password@example.org/data"])
def test_does_not_publish_unlinkable_or_unsafe_reference_urls(url):
    assert source_reference("Source", url, "forecast", "Weather inputs.") is None


def test_groups_only_the_weather_dataset_actually_used_for_the_window():
    references = given_references()
    assert len(references) == 1
    reference = references[0]
    assert reference == {"id": reference["id"], "name": "Example weather weather inputs", "url": "https://example.org/weather", "role": "forecast", "description": "Rainfall and temperature used for this forecast.", "period_start": "2020-01-01", "period_end": "2020-01-02", "dates": ["2020-01-01", "2020-01-02"], "retrieved_at": "2020-01-20T09:00:00+00:00", "reference_period": None, "reference_location": None, "basis": "recent_weather"}


def test_keeps_actual_weather_and_seasonal_baseline_references_distinct():
    weather = list(given_weather(count=2).values())
    weather[1] = {**weather[1], "source": "NASA POWER", "source_url": "https://example.org/climate", "basis": "seasonal_baseline", "reference_period": "2001–2025", "reference_name": "Lusaka"}
    references = given_references(weather=weather)
    assert [(reference["name"], reference["url"], reference["dates"], reference["basis"]) for reference in references] == [("Example weather weather inputs", "https://example.org/weather", ["2020-01-01"], "recent_weather"), ("NASA POWER seasonal baseline", "https://example.org/climate", ["2020-01-02"], "seasonal_baseline")]
    assert references[1]["reference_period"] == "2001–2025"
    assert references[1]["reference_location"] == "Lusaka"


def test_dataset_versions_at_the_same_url_have_distinct_reference_identifiers():
    weather = list(given_weather(count=2).values())
    weather[1]["retrieved_at"] = "2020-01-21T09:00:00+00:00"
    references = given_references(weather=weather)
    assert len(references) == 2
    assert references[0]["id"] != references[1]["id"]


def test_cites_historical_context_when_reported_zero_is_used():
    location = {**given_location(), "background": given_background()}
    inputs = {**given_inputs(), "water_access": 0, "sanitation_access": None, "previous_annual_occurrence": None}
    reference = given_references(location=location, inputs=inputs)[1]
    assert (reference["url"], reference["role"], reference["reference_location"], reference["scope"]) == (given_background()["source_url"], "forecast", "Lusaka", "parent_district")
    assert reference["description"] == "Historical water access for Lusaka. WASH indicators in the study are derived from the 2014 survey. Parent district context is used for this local area."


def test_does_not_cite_missing_historical_indicators_as_observed_inputs():
    location = {**given_location(), "background": given_background()}
    inputs = {**given_inputs(), "water_access": None, "sanitation_access": None, "previous_annual_occurrence": None}
    assert [reference["url"] for reference in given_references(location=location, inputs=inputs)] == ["https://example.org/weather"]


def test_annual_history_does_not_claim_water_or_sanitation_inputs():
    location = {**given_location(), "background": given_background()}
    inputs = {**given_inputs(), "water_access": None, "sanitation_access": None, "previous_annual_occurrence": 0}
    assert given_references(location=location, inputs=inputs)[1]["description"] == "Historical annual cholera history for Lusaka. Parent district context is used for this local area."


def test_identifies_a_regional_weather_location_proxy():
    location = {**given_location(), "name": "Lusaka", "scope": "province_reference", "source_url": "https://www.geonames.org/909137"}
    reference = given_references(location=location)[1]
    assert reference["url"] == location["source_url"]
    assert reference["description"] == "Weather reference: Lusaka. A regional location is used as a proxy."


def test_training_sources_do_not_claim_unused_forecast_providers():
    sources = [{"name": "Dated cholera surveillance", "url": "https://example.org/cases"}, {"name": "Historical weather training inputs", "url": "https://example.org/archive"}, {"name": "NASA POWER seasonal input fallback", "url": "https://example.org/climate"}]
    references = given_references(report={"sources": sources})
    assert [(reference["role"], reference["url"]) for reference in references] == [("forecast", "https://example.org/weather"), ("training", "https://example.org/cases"), ("training", "https://example.org/archive")]


def test_includes_the_method_used_by_the_selected_random_forest():
    references = given_references(report={"selected_model": "random_forest"})
    assert [(reference["role"], reference["url"]) for reference in references if reference["role"] == "method"] == [("method", "https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf"), ("method", "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html")]


def test_does_not_cite_random_forest_as_the_method_for_another_model():
    assert [reference["role"] for reference in given_references(report={"selected_model": "logistic_regression"})] == ["forecast"]


def test_preserves_selected_window_references_with_the_saved_prediction():
    service = given_service(ready=False)
    identifier = service.repository.selected_area["id"]
    service.outlook_engine = given_engine(identifier)
    result = service.outlook(identifier, "2020-01-20", 2)
    assert result["days"][0]["references"][0]["dates"] == [f"2020-01-{day:02d}" for day in range(1, 15)]
    assert result["days"][1]["references"][0]["dates"] == [f"2020-01-{day:02d}" for day in range(2, 16)]
    assert service.repository.saved[0]["references"] == result["days"][0]["references"]
