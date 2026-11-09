import json

from backend.prepare_map import save_interactive_geometry


def given_geometry(tmp_path, monkeypatch, coordinates=None):
    monkeypatch.setattr("backend.prepare_map.ROOT", tmp_path)
    coordinates = coordinates if coordinates is not None else [[[[28, -15], [30, -15], [30, -13], [28, -13], [28, -15]]]]
    geometry = {"type": "FeatureCollection", "features": [{"properties": {"shapeName": "District"}, "geometry": {"type": "MultiPolygon", "coordinates": coordinates}}]}
    save_interactive_geometry(geometry, {"longitude_scale": 30, "latitude_scale": 30}, {"license": "CC BY 4.0"})
    return json.loads((tmp_path / "backend/static/forecast-map/districts.geojson").read_text())


def test_preserves_geographic_coordinates_and_district_centroid(tmp_path, monkeypatch):
    result = given_geometry(tmp_path, monkeypatch)
    assert result == {"type": "FeatureCollection", "source": {"license": "CC BY 4.0"}, "features": [{"type": "Feature", "properties": {"name": "District", "center": [29, -14]}, "geometry": {"type": "MultiPolygon", "coordinates": [[[[28, -15], [30, -15], [30, -13], [28, -13], [28, -15]]]]}}]}


def test_does_not_drop_islands_or_holes(tmp_path, monkeypatch):
    outer = [[28, -15], [30, -15], [30, -13], [28, -13], [28, -15]]
    hole = [[28.5, -14.5], [28.5, -14], [29, -14], [29, -14.5], [28.5, -14.5]]
    island = [[32, -12], [32.2, -12], [32.2, -11.8], [32, -11.8], [32, -12]]
    result = given_geometry(tmp_path, monkeypatch, [[outer, hole], [island]])
    assert result["features"][0]["geometry"]["coordinates"] == [[outer, hole], [island]]
    assert result["features"][0]["properties"]["center"] == [29, -14]


def test_uses_a_coordinate_when_a_ring_has_no_area(tmp_path, monkeypatch):
    ring = [[28, -15], [28, -15], [28, -15], [28, -15]]
    result = given_geometry(tmp_path, monkeypatch, [[ring]])
    assert result["features"][0]["properties"]["center"] == [28, -15]
    assert result["features"][0]["geometry"]["coordinates"] == [[ring]]
