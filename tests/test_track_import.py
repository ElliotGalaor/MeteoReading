"""Track import on structures met in real open data (Geotrek, data.gouv.fr, OpenStreetMap): meteo/gps.py."""
from __future__ import annotations

import json

import numpy as np
import pytest

from meteo.gps import chain_parts, parse_track, read_tracks


def steps(track) -> float:
    return float(np.diff(track.points["dist_km"].to_numpy()).max())


def line(lon0: float, lat0: float, n: int = 20, dlon: float = 0.002) -> list[list[float]]:
    return [[round(lon0 + i * dlon, 6), lat0] for i in range(n)]


# --- KML: points of interest are not part of the route (Geotrek KML) ----------------------------------------------
def test_kml_points_of_interest_are_left_out():
    coords = " ".join(f"{lon},{lat},1000" for lon, lat in line(6.5, 45.3))
    pois = "".join(f"<Placemark><name>POI {i}</name><Point><coordinates>{6.5 + i * 0.3},{45.0 + i * 0.2}</coordinates>"
                   f"</Point></Placemark>" for i in range(5))
    kml = (f'<kml xmlns="http://www.opengis.net/kml/2.2"><Document><name>Tour</name>{pois}'
           f"<Placemark><name>Tour des glaciers</name><LineString><coordinates>{coords}</coordinates></LineString>"
           f"</Placemark></Document></kml>").encode()
    t = parse_track(kml, "tour.kml")
    assert len(t.points) == 20 and steps(t) < 0.2 and t.name == "Tour des glaciers"


def test_kml_with_only_points_reads_them_in_order():
    pois = "".join(f"<Placemark><Point><coordinates>{6.5 + i * 0.01},45.0</coordinates></Point></Placemark>" for i in range(3))
    t = parse_track(f"<kml><Document>{pois}</Document></kml>".encode(), "places.kml")
    assert len(t.points) == 3


# --- GPX: the name of the track, not of its first waypoint (Geotrek GPX) ------------------------------------------
GEOTREK_GPX = """<?xml version="1.0" encoding="UTF-8"?>
<gpx xmlns="http://www.topografix.com/GPX/1/1" version="1.1">
  <wpt lat="45.07" lon="6.26"><name>Faune: Petit Apollon</name></wpt>
  <wpt lat="45.08" lon="6.27"><name>Refuge: Refuge du Pic</name></wpt>
  <trk><name>[Itinéraire] 10 - Vallon de la Buffe</name><trkseg>{pts}</trkseg></trk>
</gpx>"""


def gpx_points(lon0: float, lat0: float, n: int = 10, t0: int | None = None) -> str:
    out = []
    for i, (lon, lat) in enumerate(line(lon0, lat0, n)):
        time = f"<time>2025-07-15T{t0 + i // 60:02d}:{i % 60:02d}:00Z</time>" if t0 is not None else ""
        out.append(f'<trkpt lat="{lat}" lon="{lon}"><ele>1500</ele>{time}</trkpt>')
    return "".join(out)


def test_gpx_named_after_its_track_not_its_waypoints():
    t = parse_track(GEOTREK_GPX.format(pts=gpx_points(6.26, 45.07)).encode(), "trek.gpx")
    assert t.name == "10 - Vallon de la Buffe"          # without the "[Itinéraire]" tag
    assert len(t.points) == 10                           # the waypoints are not in the route


# --- Several tracks: kept apart, unless they follow each other ----------------------------------------------------
def gpx_tracks(*tracks: tuple[str, str]) -> bytes:
    body = "".join(f"<trk><name>{n}</name><trkseg>{p}</trkseg></trk>" for n, p in tracks)
    return f'<gpx xmlns="http://www.topografix.com/GPX/1/1">{body}</gpx>'.encode()


def test_separate_tracks_are_kept_apart():
    data = gpx_tracks(("Morning", gpx_points(5.70, 45.17, 10, t0=8)), ("Someone else", gpx_points(2.30, 48.80, 30, t0=9)))
    tracks = read_tracks(data, "two.gpx")
    assert [t.name for t in tracks] == ["Morning", "Someone else"]
    assert max(steps(t) for t in tracks) < 0.2
    assert parse_track(data, "two.gpx").name == "Someone else"          # the longest


def test_consecutive_days_are_joined():
    day1 = gpx_points(6.00, 45.00, 10, t0=8)
    end = line(6.00, 45.00, 10)[-1]
    day2 = gpx_points(end[0] + 0.001, end[1], 10, t0=10)                 # starts where day 1 ended, later
    tracks = read_tracks(gpx_tracks(("Day 1", day1), ("Day 2", day2)), "trip.gpx")
    assert len(tracks) == 1 and len(tracks[0].points) == 20 and tracks[0].has_time


def test_shuffled_points_are_not_the_default():
    rng = np.random.default_rng(1)
    shuffled = "".join(f'<trkpt lat="{45 + rng.uniform(0, 0.02)}" lon="{5.7 + rng.uniform(0, 0.02)}"/>' for _ in range(600))
    data = gpx_tracks(("Anonymous", shuffled), ("Real ride", gpx_points(5.70, 45.17, 40)))
    tracks = read_tracks(data, "trackpoints.gpx")
    assert [t.looks_scrambled for t in tracks] == [True, False]
    assert parse_track(data, "trackpoints.gpx").name == "Real ride"


def test_a_network_of_hundreds_of_lines_is_refused():
    features = [{"type": "Feature", "properties": {}, "geometry": {"type": "LineString", "coordinates": line(i * 0.1, 45.0, 3)}}
                for i in range(150)]
    data = json.dumps({"type": "FeatureCollection", "name": "Liaisons_PDIPR", "features": features}).encode()
    with pytest.raises(ValueError, match="150 separate lines"):
        read_tracks(data, "liaisons.geojson")


# --- MultiLineString from an OSM relation: parts chained by their nearest ends --------------------------------
def test_multilinestring_parts_are_chained():
    a, b, c = line(9.000, 41.9, 10), line(9.018, 41.9, 10), line(9.036, 41.9, 10)
    parts = [c, b[::-1], a]                                              # out of order, one reversed
    geo = {"type": "Feature", "properties": {"name": "GR 20 Variante"},
           "geometry": {"type": "MultiLineString", "coordinates": parts}}
    t = parse_track(json.dumps(geo).encode(), "relation.geojson")
    assert steps(t) < 0.2 and t.distance_km == pytest.approx(4.47, abs=0.05)   # 0.054° of longitude at 41.9° N


def test_timed_parts_go_in_time_order():
    late = [(45.0, 6.1, None, "2025-07-15T10:00:00Z"), (45.0, 6.11, None, "2025-07-15T10:10:00Z")]
    early = [(45.0, 6.0, None, "2025-07-15T09:00:00Z"), (45.0, 6.01, None, "2025-07-15T09:10:00Z")]
    assert [r[3][11:13] for r in chain_parts([late, early])] == ["09", "09", "10", "10"]


# --- JSON that is not GeoJSON, or not UTF-8 (data.gouv.fr exports) ------------------------------------------
def test_latin1_json_that_is_not_geojson():
    data = json.dumps({"Randonnee": [{"pratique": "Pédestre", "geometry": "01020000206A08"}]}, ensure_ascii=False)
    with pytest.raises(ValueError, match="not GeoJSON"):
        read_tracks(data.encode("latin-1"), "cd58-randonnee.json")


def test_latin1_geojson_is_read():
    geo = {"type": "Feature", "properties": {"name": "Boucle de l'Étang"},
           "geometry": {"type": "LineString", "coordinates": line(3.5, 47.0, 5)}}
    t = parse_track(json.dumps(geo, ensure_ascii=False).encode("latin-1"), "boucle.geojson")
    assert t.name == "Boucle de l'Étang"


# --- Activity guessed from French names ----------------------------------------------------------------------
@pytest.mark.parametrize(("name", "activity"), [
    ("GR® 3 - La Loire Sauvage à Pied", "Walking"), ("GR 20 Variante via Monte Incudine", "Walking"),
    ("Sentier des Pitchouns", "Walking"), ("Itinéraire pédestre", "Walking"),
    ("Enduro VTT de la Grave", "Cycling"), ("Tour à vélo", "Cycling"), ("Lyon to Marseille by car", "Driving"),
])
def test_activity_from_french_names(monkeypatch, name, activity):
    monkeypatch.syspath_prepend("dashboard")
    from route_page import _guess_activity

    assert _guess_activity(name) == activity
