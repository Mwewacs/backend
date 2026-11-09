import hashlib
import json
import math
from urllib.request import urlopen

from backend.prepare_data import ROOT


def simplify(points):
    if len(points) < 3:
        return points
    first, last = points[0], points[-1]
    horizontal, vertical = last[0] - first[0], last[1] - first[1]
    length = math.hypot(horizontal, vertical)
    distances = [abs(vertical * (point[0] - first[0]) - horizontal * (point[1] - first[1])) / length if length else math.dist(first, point) for point in points[1:-1]]
    maximum = max(distances, default=0)
    if maximum <= 0.5:
        return [first, last]
    index = distances.index(maximum) + 1
    return simplify(points[:index + 1])[:-1] + simplify(points[index:])


def save_interactive_geometry(geometry, projection, source):
    features = []
    for feature in geometry["features"]:
        polygons = feature["geometry"]["coordinates"]
        if feature["geometry"]["type"] == "Polygon":
            polygons = [polygons]
        reduced_polygons = []
        for polygon in polygons:
            rings = []
            for ring in polygon:
                projected = [[point[0] * projection["longitude_scale"], point[1] * projection["latitude_scale"]] for point in ring]
                reduced = simplify(projected)
                if len(reduced) < 4:
                    reduced = projected
                rings.append([[round(point[0] / projection["longitude_scale"], 5), round(point[1] / projection["latitude_scale"], 5)] for point in reduced])
            reduced_polygons.append(rings)
        outer = max((polygon[0] for polygon in reduced_polygons), key=lambda ring: abs(sum(first[0] * second[1] - second[0] * first[1] for first, second in zip(ring, ring[1:]))))
        cross_products = [first[0] * second[1] - second[0] * first[1] for first, second in zip(outer, outer[1:])]
        area = sum(cross_products)
        center = [round(sum((first[axis] + second[axis]) * cross for first, second, cross in zip(outer, outer[1:], cross_products)) / (3 * area), 5) for axis in [0, 1]] if area else outer[0]
        features.append({"type": "Feature", "properties": {"name": feature["properties"]["shapeName"], "center": center}, "geometry": {"type": "MultiPolygon", "coordinates": reduced_polygons}})
    output = ROOT / "backend/static/forecast-map"
    output.mkdir(parents=True, exist_ok=True)
    (output / "districts.geojson").write_text(json.dumps({"type": "FeatureCollection", "source": source, "features": features}, separators=(",", ":")))


def prepare_map():
    source_url = "https://www.geoboundaries.org/api/current/gbOpen/ZMB/ADM2/"
    with urlopen(source_url, timeout=30) as response:
        source = json.load(response)
    with urlopen(source["simplifiedGeometryGeoJSON"], timeout=30) as response:
        raw = response.read()
    geometry = json.loads(raw)
    directory = ROOT / "data/raw/geography"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "zambia-districts.geojson").write_bytes(raw)
    coordinates = []
    districts = []
    for feature in geometry["features"]:
        polygons = feature["geometry"]["coordinates"]
        if feature["geometry"]["type"] == "Polygon":
            polygons = [polygons]
        rings = [ring for polygon in polygons for ring in polygon]
        coordinates.extend(point for ring in rings for point in ring)
        districts.append({"name": feature["properties"]["shapeName"], "rings": rings})
    minimum_longitude = min(point[0] for point in coordinates)
    maximum_longitude = max(point[0] for point in coordinates)
    minimum_latitude = min(point[1] for point in coordinates)
    maximum_latitude = max(point[1] for point in coordinates)
    longitude_scale = 328 / (maximum_longitude - minimum_longitude)
    latitude_scale = 300 / (maximum_latitude - minimum_latitude)
    for district in districts:
        rings = []
        for ring in district.pop("rings"):
            projected = [[16 + (point[0] - minimum_longitude) * longitude_scale, 16 + (maximum_latitude - point[1]) * latitude_scale] for point in ring]
            reduced = simplify(projected)
            rings.append(reduced if len(reduced) >= 4 else projected)
        district["path"] = " ".join(" ".join(f"{'M' if index == 0 else 'L'} {point[0]:.2f} {point[1]:.2f}" for index, point in enumerate(ring)) + " Z" for ring in rings)
    result = {"view_box": "0 0 360 332", "districts": districts, "source": {"name": "GRID3 / Office of the Surveyor General via geoBoundaries", "url": source_url, "license": source["boundaryLicense"], "boundary_year": source["boundaryYearRepresented"], "boundary_id": source["boundaryID"], "sha256": hashlib.sha256(raw).hexdigest()}, "projection": {"minimum_longitude": minimum_longitude, "maximum_latitude": maximum_latitude, "longitude_scale": longitude_scale, "latitude_scale": latitude_scale, "padding": 16}}
    output = ROOT / "mobile/src/data"
    output.mkdir(parents=True, exist_ok=True)
    (output / "district-map.json").write_text(json.dumps(result, separators=(",", ":")))
    save_interactive_geometry(geometry, result["projection"], result["source"])
    print(json.dumps({"districts": len(districts), "boundary_year": source["boundaryYearRepresented"], "source": source_url}))


if __name__ == "__main__":
    prepare_map()
