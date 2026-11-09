import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]


def fetch_sources(limit=20, delay=6):
    audit_path = ROOT / "data/source-audit.json"
    audit = json.loads(audit_path.read_text())
    directory = ROOT / "data/raw/cholera-taxonomy"
    directory.mkdir(parents=True, exist_ok=True)
    candidates = [
        collection
        for collection in audit["catalogEntries"]
        if collection["frequency"] != "Annual"
        and not (directory / f"observations-{collection['identifier']}.csv").exists()
    ]
    candidates.sort(key=lambda collection: collection["observationCount"], reverse=True)
    results = []
    for index, collection in enumerate(candidates[:limit]):
        if index:
            time.sleep(delay)
        identifier = collection["identifier"]
        url = f"https://cholera-taxonomy.middle-distance.com/observation_collections/{identifier}/download_observations"
        try:
            request = Request(url, headers={"User-Agent": "CholeraAcademicPrototype/0.1"})
            with urlopen(request, timeout=20) as response:
                content = response.read()
            if not content.lstrip().startswith(b"Location,"):
                results.append({"identifier": identifier, "status": "unexpected_format"})
                continue
            (directory / f"observations-{identifier}.csv").write_bytes(content)
            results.append({"identifier": identifier, "status": "downloaded"})
        except HTTPError as error:
            result = {"identifier": identifier, "status": error.code}
            if error.code == 429:
                result["retry_after_seconds"] = error.headers.get("Retry-After")
            results.append(result)
            if error.code == 429:
                break
        except (URLError, TimeoutError) as error:
            results.append({"identifier": identifier, "status": "failed", "message": str(error)})
    report = {"retrieved_at": datetime.now(timezone.utc).isoformat(), "results": results}
    (ROOT / "data/download-report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--delay", type=float, default=6)
    arguments = parser.parse_args()
    if arguments.limit < 1 or arguments.limit > 20 or arguments.delay < 6:
        parser.error("Use 1 to 20 downloads and a delay of at least 6 seconds.")
    print(json.dumps(fetch_sources(arguments.limit, arguments.delay), indent=2))
