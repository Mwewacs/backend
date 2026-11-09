import csv
import argparse
import hashlib
import json
import math
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROVINCES = {
    "ZM-01": "Western", "ZM-02": "Central", "ZM-03": "Eastern",
    "ZM-04": "Luapula", "ZM-05": "Northern", "ZM-06": "North-Western",
    "ZM-07": "Southern", "ZM-08": "Copperbelt", "ZM-09": "Lusaka",
    "ZM-10": "Muchinga",
}
FEATURE_NAMES = ["prior_cases", "month_sin", "month_cos", "peer_cases_last_7_days", "peer_reporting_fraction"]


def read_count(value):
    if isinstance(value, bool):
        raise ValueError("Case counts must be non-negative whole numbers.")
    if value is None or str(value).strip() == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("Case counts must be non-negative whole numbers.") from None
    if not math.isfinite(number) or number < 0 or not number.is_integer():
        raise ValueError("Case counts must be non-negative whole numbers.")
    return int(number)


def canonical_location(location):
    if not isinstance(location, str) or not location:
        raise ValueError("A Zambian area identifier is required.")
    parts = location.replace("ZMB.Zambia", "ZMB").split("::")
    if len(parts) < 2 or parts[:2] != ["AFR", "ZMB"]:
        raise ValueError("A Zambian area identifier is required.")
    if len(parts) > 2:
        parts[2] = PROVINCES.get(parts[2], parts[2])
    return "::".join(parts)


def area_identifier(location):
    return hashlib.sha256(location.encode()).hexdigest()[:16]


def location_details(location):
    parts = location.split("::")
    level = "country" if len(parts) == 2 else "province" if len(parts) == 3 else "reported_area" if len(parts) == 4 else "local_area"
    return {"id": area_identifier(location), "name": parts[-1] if len(parts) > 2 else "Zambia", "location": location, "level": level, "region": parts[2] if len(parts) > 2 else "Zambia"}


def parse_observation(record, collection_identifier):
    location = canonical_location(record.get("Location", ""))
    start = date.fromisoformat(record["TL"])
    end = date.fromisoformat(record["TR"])
    if end < start:
        raise ValueError("The observation period ends before it starts.")
    count = read_count(record.get("sCh"))
    if record.get("available_at"):
        datetime.fromisoformat(record["available_at"].replace("Z", "+00:00"))
    age_restricted = any(str(record.get(name, "")).strip() for name in ["age_L", "age_U"])
    return {
        "area_id": area_identifier(location), "location": location,
        "start": start.isoformat(), "end": end.isoformat(), "cases": count,
        "duration_days": (end - start).days + 1,
        "collection_id": collection_identifier, "age_restricted": age_restricted,
        "grouped": any(character in location for character in "()|"),
        "available_at": record.get("available_at") or None,
        "date_basis": record.get("date_basis") or "not_verified",
        "outbreak_episode_id": record.get("outbreak_episode_id") or None,
        "source_url": f"https://cholera-taxonomy.middle-distance.com/observation_collections/{collection_identifier}",
    }


def parse_surveillance(record):
    required = ["area_identifier", "area_name", "area_level", "period_start", "period_end", "reporting_status", "date_basis", "available_at", "source_reference"]
    if any(not record.get(name, "").strip() for name in required):
        raise ValueError("A required surveillance field is missing.")
    if record["reporting_status"] not in ["reported", "not_reported"]:
        raise ValueError("The reporting status must be reported or not_reported.")
    if record["date_basis"] not in ["onset", "presentation", "reporting"]:
        raise ValueError("The date basis must be onset, presentation or reporting.")
    if record["area_level"] not in ["district", "subdistrict", "local_area", "facility"]:
        raise ValueError("Use a district, subdistrict, local_area or facility area level.")
    datetime.fromisoformat(record["available_at"].replace("Z", "+00:00"))
    identifier = record["area_identifier"]
    region = record.get("region", "").strip()
    normalized = parse_observation({"Location": f"AFR::ZMB::{region or identifier}::{identifier}", "TL": record["period_start"], "TR": record["period_end"], "sCh": record.get("suspected_cases"), "available_at": record["available_at"], "date_basis": record["date_basis"]}, "curated")
    if (record["reporting_status"] == "reported") != (normalized["cases"] is not None):
        raise ValueError("Reported records need case counts; missing reports need blank counts.")
    normalized.update({"area_id": identifier, "area_name": record["area_name"], "area_level": record["area_level"], "region": region or "Zambia", "source_url": record["source_reference"], "outbreak_episode_id": record.get("outbreak_episode_id") or None})
    return normalized


def select_observations(observations, duration_days=1):
    grouped = defaultdict(list)
    for observation in observations:
        if observation["duration_days"] != duration_days or observation["cases"] is None or observation["age_restricted"] or observation["grouped"]:
            continue
        if len(observation["location"].split("::")) < 4:
            continue
        grouped[(observation["area_id"], observation["start"])].append(observation)
    selected, conflicts = [], []
    for (identifier, observed_date), records in sorted(grouped.items()):
        counts = {record["cases"] for record in records}
        if len(counts) != 1:
            conflicts.append({"area_id": identifier, "date": observed_date, "counts": sorted(counts), "collections": sorted({record["collection_id"] for record in records})})
            continue
        observation = dict(records[0])
        observation["collections"] = sorted({record["collection_id"] for record in records})
        selected.append(observation)
    return selected, conflicts


def eligible_before(record, issue_date):
    if record["end"] >= issue_date:
        return False
    if not record.get("available_at"):
        return True
    available = datetime.fromisoformat(record["available_at"].replace("Z", "+00:00"))
    available = available.replace(tzinfo=timezone.utc) if available.tzinfo is None else available
    return available <= datetime.fromisoformat(issue_date).replace(tzinfo=timezone.utc)


def build_features(area_id, issue_date, observations, quiet_days=14):
    issued = date.fromisoformat(issue_date)
    eligible = [record for record in observations if eligible_before(record, issue_date)]
    own_records = {record["start"]: record for record in eligible if record["area_id"] == area_id}
    if not own_records:
        return None
    period_days = next(iter(own_records.values()))["duration_days"]
    history = [own_records.get((issued - timedelta(days=offset)).isoformat()) for offset in range(quiet_days, 0, -period_days)]
    if any(record is None for record in history):
        return None
    if any(record["cases"] != 0 for record in history):
        return None
    parent = history[0]["location"].rsplit("::", 1)[0]
    peers = [record for record in eligible if record["area_id"] != area_id and record["location"].rsplit("::", 1)[0] == parent and record["duration_days"] == period_days]
    peer_records = [record for record in peers if issued - timedelta(days=7) <= date.fromisoformat(record["start"]) < issued]
    peer_ids = {record["area_id"] for record in peers}
    peer_count = sum(record["cases"] for record in peer_records)
    fraction = len(peer_records) / ((7 // period_days) * len(peer_ids)) if peer_ids else 0
    earlier = [own_records.get((issued - timedelta(days=offset)).isoformat()) for offset in range(28, quiet_days, -period_days)]
    prior_cases = sum(record["cases"] for record in earlier) if all(record is not None for record in earlier) else None
    angle = 2 * math.pi * (issued.month - 1) / 12
    return {"prior_cases": prior_cases, "month_sin": math.sin(angle), "month_cos": math.cos(angle), "peer_cases_last_7_days": peer_count, "peer_reporting_fraction": fraction}


def build_examples(observations, quiet_days=14, horizon_days=7):
    by_area = defaultdict(dict)
    for observation in observations:
        by_area[observation["area_id"]][observation["start"]] = observation
    examples = []
    for identifier, records in sorted(by_area.items()):
        period_days = next(iter(records.values()))["duration_days"]
        for issue_date in sorted(records):
            features = build_features(identifier, issue_date, observations, quiet_days)
            if features is None:
                continue
            issued = date.fromisoformat(issue_date)
            future = [records.get((issued + timedelta(days=offset)).isoformat()) for offset in range(0, horizon_days, period_days)]
            if any(record is None for record in future):
                continue
            onset = next((record["start"] for record in future if record["cases"] > 0), None)
            outcome_available = max(record["available_at"] for record in future) if all(record["available_at"] for record in future) else None
            examples.append({"area_id": identifier, "issue_date": issue_date, "window_end": (issued + timedelta(days=horizon_days - 1)).isoformat(), "label": int(onset is not None), "onset_date": onset, "outcome_available_at": outcome_available, "outbreak_episode_id": records[onset].get("outbreak_episode_id") if onset else None, "episode_group": issue_date[:4], "features": features})
    return examples


def prepare_data(raw_directory=None, output_directory=None, curated_only=False):
    directory = Path(raw_directory or ROOT / "data/raw/cholera-taxonomy")
    output = Path(output_directory or ROOT / "artifacts")
    output.mkdir(parents=True, exist_ok=True)
    observations, rejected, sources = [], [], []
    for path in [] if curated_only else sorted(directory.glob("observations-*.csv")):
        identifier = path.stem.removeprefix("observations-")
        sources.append({"collection_id": identifier, "file": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        with path.open(newline="", encoding="utf-8-sig") as source:
            for row_number, record in enumerate(csv.DictReader(source), 2):
                try:
                    observations.append(parse_observation(record, identifier))
                except (ValueError, KeyError) as error:
                    rejected.append({"collection_id": identifier, "row": row_number, "reason": str(error)})
    curated_path = directory.parent / "surveillance.csv"
    if curated_only and not curated_path.exists():
        raise ValueError("data/raw/surveillance.csv is required for curated-only preparation.")
    if curated_path.exists():
        sources.append({"collection_id": "curated", "file": str(curated_path), "sha256": hashlib.sha256(curated_path.read_bytes()).hexdigest()})
        with curated_path.open(newline="", encoding="utf-8-sig") as source:
            for row_number, record in enumerate(csv.DictReader(source), 2):
                try:
                    observations.append(parse_surveillance(record))
                except (ValueError, KeyError) as error:
                    rejected.append({"collection_id": "curated", "row": row_number, "reason": str(error)})
    daily, conflicts = select_observations(observations)
    weekly, weekly_conflicts = select_observations(observations, 7)
    daily_area_ids = {record["area_id"] for record in daily}
    model_observations = daily + [record for record in weekly if record["area_id"] not in daily_area_ids]
    conflicts.extend(weekly_conflicts)
    areas = {}
    for observation in observations:
        if observation["grouped"]:
            continue
        identifier = observation["area_id"]
        if identifier not in areas:
            areas[identifier] = location_details(observation["location"])
            areas[identifier]["id"] = identifier
            for source_name, area_name in [("area_name", "name"), ("area_level", "level"), ("region", "region")]:
                if source_name in observation:
                    areas[identifier][area_name] = observation[source_name]
            areas[identifier].update({"first_date": observation["start"], "last_date": observation["end"], "daily_records": 0})
        areas[identifier]["first_date"] = min(areas[identifier]["first_date"], observation["start"])
        areas[identifier]["last_date"] = max(areas[identifier]["last_date"], observation["end"])
    for observation in daily:
        area = areas[observation["area_id"]]
        area["daily_records"] += 1
        area["first_daily_date"] = min(area.get("first_daily_date", observation["start"]), observation["start"])
        area["last_daily_date"] = max(area.get("last_daily_date", observation["end"]), observation["end"])
    annual_path = ROOT / "data/raw/cholera-district-annual-2008-2018.csv"
    if annual_path.exists():
        with annual_path.open(newline="", encoding="utf-8-sig") as source:
            for record in csv.DictReader(source):
                matching = next((area for area in areas.values() if area["name"] == record["District"] and area["level"] == "reported_area"), None)
                if matching is None:
                    location = f"annual::ZMB::{record['District']}"
                    matching = {"id": area_identifier(location), "name": record["District"], "location": location, "level": "district", "region": "Zambia", "first_date": "2008-01-01", "last_date": "2018-12-31", "daily_records": 0}
                    areas[matching["id"]] = matching
                matching["annual_history"] = [{"year": int(name[1:]), "cases": read_count(value)} for name, value in record.items() if name.startswith("y") and name[1:].isdigit()]
                matching["context"] = {"population_2010": read_count(record["pop2010"]), "water_percentage": float(record["water(%)"]), "sanitation_percentage": float(record["sanitation(%)"]), "poverty_percentage": float(record["poverty(%)"])}
    examples = build_examples(model_observations)
    dataset = {"areas": sorted(areas.values(), key=lambda area: (-area["daily_records"], area["name"])), "observations": observations, "daily_observations": daily, "model_observations": model_observations, "examples": examples, "conflicts": conflicts}
    summary = {
        "source_files": len(sources), "raw_records": len(observations), "daily_records": len(daily),
        "areas": len(areas), "weekly_records": len(weekly), "examples": len(examples), "positive_examples": sum(example["label"] for example in examples),
        "unique_onsets": len({(example["area_id"], example["onset_date"]) for example in examples if example["onset_date"]}),
        "years_with_onsets": sorted({example["episode_group"] for example in examples if example["label"]}),
        "rejected_records": rejected, "conflicts": conflicts, "sources": sources,
        "target_definition": "First reported suspected case after 14 consecutive reported case-free days, within the next 7 days. This is a research proxy for episode onset, not an official outbreak declaration.",
        "publication_dates_verified": bool(model_observations) and all(record["available_at"] for record in model_observations),
        "episode_ids_verified": any(example["label"] for example in examples) and all(example["outbreak_episode_id"] for example in examples if example["label"]),
        "limitations": ["Missing dates are not filled with zero.", "Conflicting daily counts are excluded.", "Clinical presentation dates do not necessarily equal symptom-onset dates.", "Regional year groups are used conservatively to keep the same year's observations together.", "The supplied annual covariates are context data, not weekly onset labels."],
    }
    (output / "prepared-data.json").write_text(json.dumps(dataset, indent=2) + "\n")
    (output / "preparation-report.json").write_text(json.dumps(summary, indent=2) + "\n")
    return dataset, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--curated-only", action="store_true")
    arguments = parser.parse_args()
    _, report = prepare_data(curated_only=arguments.curated_only)
    print(json.dumps({key: value for key, value in report.items() if key not in ["sources", "conflicts", "rejected_records"]}, indent=2))
