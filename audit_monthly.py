import calendar
import json
from collections import defaultdict

from backend.prepare_data import ROOT


def monthly_readiness(dataset):
    areas = {area["id"]: area for area in dataset["areas"]}
    months = defaultdict(dict)
    for record in dataset["daily_observations"]:
        months[(record["area_id"], record["start"][:7])][record["start"]] = record["cases"]
    complete = []
    for (identifier, month), records in sorted(months.items()):
        year, number = map(int, month.split("-"))
        if len(records) != calendar.monthrange(year, number)[1] or any(value is None for value in records.values()):
            continue
        complete.append({"area_id": identifier, "area_name": areas[identifier]["name"], "area_level": areas[identifier]["level"], "month": month, "cases": sum(records.values())})
    complete_keys = {(record["area_id"], record["month"]) for record in complete}
    paired = []
    for record in complete:
        year, number = map(int, record["month"].split("-"))
        previous = f"{year - 1}-12" if number == 1 else f"{year}-{number - 1:02d}"
        if (record["area_id"], previous) in complete_keys:
            paired.append(record)
    return {"scope": "Complete calendar months from cleaned daily reports only; longer source intervals are not divided into invented daily counts.", "daily_area_months": len(months), "complete_months": complete, "consecutive_month_targets": paired, "monthly_model_trained": False, "monthly_outbreak_labels_verified": False, "active_horizon_days": 7, "next_requirements": ["Consistent district identifiers and monthly case reports, including observed zero-case months.", "An explicit outbreak definition and verified reporting availability dates.", "Enough dated monthly records to reserve separate training, validation and final test periods."]}


def main():
    dataset = json.loads((ROOT / "artifacts/prepared-data.json").read_text())
    report = monthly_readiness(dataset)
    (ROOT / "artifacts/monthly-readiness.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"complete_months": len(report["complete_months"]), "consecutive_month_targets": len(report["consecutive_month_targets"]), "monthly_model_trained": False}))


if __name__ == "__main__":
    main()
