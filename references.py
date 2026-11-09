import hashlib
import json
from urllib.parse import urlsplit


def source_reference(name, url, role, description, **details):
    if not isinstance(url, str):
        return None
    try:
        parsed = urlsplit(url)
        if parsed.scheme not in ["https", "http"] or not parsed.hostname or parsed.username or parsed.password:
            return None
    except ValueError:
        return None
    identifier = hashlib.sha256(json.dumps({"role": role, "url": url, **details}, sort_keys=True).encode()).hexdigest()[:12]
    return {"id": identifier, "name": name, "url": url, "role": role, "description": description, **details}


def forecast_references(weather, location, inputs, report):
    references = []
    grouped = {}
    for day in weather:
        key = (day["source"], day.get("source_url"), day.get("retrieved_at"), day.get("reference_period"), day.get("reference_name"))
        grouped.setdefault(key, []).append(day)
    for (name, url, retrieved, period, place), days in grouped.items():
        dates = sorted({day["date"] for day in days})
        seasonal = all(day["basis"] == "seasonal_baseline" for day in days)
        references.append(source_reference(f"{name} {'seasonal baseline' if seasonal else 'weather inputs'}", url, "forecast", "Seasonal rainfall and temperature used for this forecast." if seasonal else "Rainfall and temperature used for this forecast.", period_start=dates[0], period_end=dates[-1], dates=dates, retrieved_at=retrieved, reference_period=period, reference_location=place, basis="seasonal_baseline" if seasonal else "mixed" if len({day["basis"] for day in days}) > 1 else days[0]["basis"]))
    scope = location.get("scope")
    geographic_description = f"Weather reference: {location['name']}."
    if scope in ["parent_reference", "province_reference"]:
        geographic_description += " A regional location is used as a proxy."
    references.append(source_reference("GeoNames location reference", location.get("source_url"), "forecast", geographic_description, reference_location=location["name"], scope=scope))
    background = location.get("background")
    if background and any(inputs.get(name) is not None for name in ["water_access", "sanitation_access", "previous_annual_occurrence"]):
        fields = [label for name, label in [("water_access", "water access"), ("sanitation_access", "sanitation access"), ("previous_annual_occurrence", "annual cholera history")] if inputs.get(name) is not None]
        description = f"Historical {', '.join(fields)} for {background['name']}."
        if any(inputs.get(name) is not None for name in ["water_access", "sanitation_access"]):
            description += " WASH indicators in the study are derived from the 2014 survey."
        if background.get("scope") == "parent_district":
            description += " Parent district context is used for this local area."
        references.append(source_reference("Mwaba et al. (2020): Zambia district data", background.get("source_url"), "forecast", description, citation="Mwaba J, Debes AK, Shea P, et al. (2020). Identification of cholera hotspots in Zambia: A spatiotemporal analysis of cholera data from 2008 to 2017. PLoS Neglected Tropical Diseases 14(4): e0008227. doi:10.1371/journal.pntd.0008227.", reference_location=background["name"], scope=background.get("scope")))
    training_descriptions = {
        "Dated cholera surveillance": ("CholeraTaxonomy: Zambia case records", "Dated reported cholera cases used to create the model's training outcomes."),
        "Historical weather training inputs": ("Open-Meteo ERA5 training weather", "Historical rainfall and temperature matched to the dated cholera training records."),
        "Published district covariates": ("Mwaba et al. (2020): historical district context", "Annual cholera history and historical district water and sanitation indicators used in training.")
    }
    for source in report.get("sources", []):
        details = training_descriptions.get(source["name"])
        if details is None:
            continue
        name, description = details
        references.append(source_reference(name, source["url"], "training", description))
    if report.get("selected_model") == "random_forest":
        references.append(source_reference("Random Forest method: Breiman (2001)", "https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf", "method", "Original reference for the Random Forest algorithm.", citation="Breiman, L. (2001). Random Forests. Machine Learning, 45, 5–32. doi:10.1023/A:1010933404324."))
        references.append(source_reference("scikit-learn RandomForestClassifier", "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html", "method", "Documentation for the Random Forest implementation used by this app."))
    return [reference for reference in references if reference is not None]
