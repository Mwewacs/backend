import argparse
import hashlib
import json
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

from backend.prepare_data import ROOT


def normalized_name(value):
    value = re.sub(r"\b(district|province|urban|ctc)\b", "", value.lower())
    value = re.sub(r"\b(health cent(?:re|er)|basic school).*$", "", value)
    return re.sub(r"[^a-z0-9]", "", value)


def choose_place(records, name, province=None, district=False):
    matches = [record for record in records if normalized_name(name) in record["names"] and (not province or record["province"] == province)]
    priorities = ["ADM2", "PPLA", "PPLA2", "PPLC", "PPL", "PPLX", "HSP", "SCH"] if district else ["PPLC", "PPLA", "PPLA2", "PPLX", "PPL", "HSP", "SCH", "ADM2"]
    for feature in priorities:
        candidates = [record for record in matches if record["feature"] == feature]
        if len(candidates) == 1:
            return candidates[0]
    return None


def resolve_location(area, records, provinces):
    province = next((key for key, value in provinces.items() if value["name"] == area["region"]), None)
    place = choose_place(records, area["name"], province, area["level"] != "local_area")
    scope = "named_place" if area["level"] == "local_area" else "district_reference"
    if not place and area["level"] != "local_area":
        place = choose_place(records, area["name"], district=True)
    if not place:
        scope = "parent_reference"
        for parent in reversed(area.get("location", "").split("::")[3:-1]):
            place = choose_place(records, parent, province, district=True)
            if place:
                break
    if not place and province:
        scope = "province_reference"
        place = provinces[province]["place"]
    if not place:
        raise ValueError(f"A weather reference could not be resolved for {area['name']}.")
    if place["feature"] == "ADM2" and scope == "named_place":
        scope = "district_reference"
    return {"latitude": place["latitude"], "longitude": place["longitude"], "name": place["name"], "geonames_id": place["id"], "source_url": f"https://www.geonames.org/{place['id']}", "province": provinces[place["province"]]["name"], "baseline_key": place["province"], "scope": scope}


def prepare_environment(refresh=False):
    raw = ROOT / "data/raw/geography"
    raw.mkdir(parents=True, exist_ok=True)
    archive_path = raw / "ZM.zip"
    if refresh or not archive_path.exists():
        archive_path.write_bytes(urlopen("https://download.geonames.org/export/dump/ZM.zip", timeout=30).read())
    records = []
    with zipfile.ZipFile(archive_path) as archive:
        for line in archive.read("ZM.txt").decode().splitlines():
            values = line.split("\t")
            records.append({"id": values[0], "name": values[1], "names": {normalized_name(name) for name in [values[1], values[2], *values[3].split(",")] if name}, "latitude": float(values[4]), "longitude": float(values[5]), "feature": values[7], "province": values[10], "district": values[11]})
    provinces = {record["province"]: {"name": record["name"].replace(" Province", ""), "place": record} for record in records if record["feature"] == "ADM1"}
    output_path = ROOT / "data/environment-reference.json"
    previous = json.loads(output_path.read_text()) if output_path.exists() else {}
    baselines = previous.get("baselines", {})
    for code, province in provinces.items():
        if code in baselines and not refresh:
            continue
        place = province["place"]
        query = urlencode({"parameters": "T2M,PRECTOTCORR", "community": "AG", "latitude": place["latitude"], "longitude": place["longitude"], "format": "JSON"})
        url = f"https://power.larc.nasa.gov/api/temporal/climatology/point?{query}"
        payload = json.load(urlopen(url, timeout=30))
        (raw / f"climate-{code}.json").write_text(json.dumps(payload, indent=2) + "\n")
        parameters = payload["properties"]["parameter"]
        months = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
        baselines[code] = {"name": province["name"], "latitude": place["latitude"], "longitude": place["longitude"], "source": "NASA POWER", "source_url": url, "period": payload["header"]["range"], "retrieved_at": datetime.now(timezone.utc).isoformat(), "scope": "province_reference", "months": {str(index + 1): {"temperature_c": parameters["T2M"][month], "rainfall_mm": parameters["PRECTOTCORR"][month]} for index, month in enumerate(months)}}
        output_path.write_text(json.dumps({"baselines": baselines}, indent=2) + "\n")
    dataset = json.loads((ROOT / "artifacts/prepared-data.json").read_text())
    areas = [area for area in dataset["areas"] if area["level"] not in ["country", "province"]]
    locations = {}
    for area in areas:
        location = resolve_location(area, records, provinces)
        context = area if area.get("context") or area.get("annual_history") else None
        parent_names = list(reversed(area.get("location", "").split("::")[3:-1]))
        if location["scope"] != "province_reference":
            parent_names.append(location["name"])
        if context is None:
            for name in parent_names:
                context = next((candidate for candidate in areas if normalized_name(candidate["name"]) == normalized_name(name) and (candidate.get("context") or candidate.get("annual_history"))), None)
                if context:
                    break
        location["background"] = {"area_id": context["id"], "name": context["name"], "scope": "own_area" if context["id"] == area["id"] else "parent_district", "context": context.get("context"), "annual_history": [record for record in context.get("annual_history", []) if 2008 <= record["year"] <= 2017], "source_url": "https://journals.plos.org/plosntds/article?id=10.1371/journal.pntd.0008227"} if context else None
        locations[area["id"]] = location
    reference = {"prepared_at": datetime.now(timezone.utc).isoformat(), "geography_source": "https://download.geonames.org/export/dump/ZM.zip", "geography_sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(), "geography_license": "GeoNames CC BY", "geography_attribution": "https://www.geonames.org/", "locations": locations, "baselines": baselines}
    output_path.write_text(json.dumps(reference, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"areas": len(locations), "baselines": len(baselines), "proxies": [{"area": area["name"], "reference": locations[area["id"]]["name"], "scope": locations[area["id"]]["scope"]} for area in areas if locations[area["id"]]["scope"] in ["parent_reference", "province_reference"]]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true")
    prepare_environment(parser.parse_args().refresh)
