"""Tests of the track built from points clicked on a map (no network)."""
from __future__ import annotations

import numpy as np
import pytest

from meteo import i18n
from meteo.gps import cumulative_distance, resample_path
from meteo.routing import drawn_name, parse_osrm, track_from_waypoints, waypoints_from_drawings

WAYPOINTS = [(45.0, 5.0), (45.05, 5.0), (45.05, 5.05)]


def test_waypoints_from_the_last_drawn_line():
    drawings = [
        {"type": "Feature", "geometry": {"type": "LineString", "coordinates": [[1, 1], [2, 2]]}},
        {"type": "Feature", "geometry": {"type": "Point", "coordinates": [3, 3]}},
        {"type": "Feature", "geometry": {"type": "LineString", "coordinates": [[5.0, 45.0], [5.1, 45.2]]}},
    ]
    assert waypoints_from_drawings(drawings) == [(45.0, 5.0), (45.2, 5.1)]   # (lat, lon)
    assert waypoints_from_drawings(None) == []


def test_parse_osrm():
    payload = {"code": "Ok", "routes": [{"distance": 12345.0, "duration": 3600.0,
                                         "geometry": {"coordinates": [[5.0, 45.0], [5.0, 45.1]]}}]}
    path, km, hours = parse_osrm(payload)
    assert path == [(45.0, 5.0), (45.1, 5.0)] and km == pytest.approx(12.345) and hours == 1.0
    with pytest.raises(RuntimeError):
        parse_osrm({"code": "NoRoute", "message": "Impossible route"})


def test_resample_path_even_spacing_and_cap():
    lat, lon = resample_path([45.0, 45.1], [5.0, 5.0], step_km=1.0)
    steps = np.diff(cumulative_distance(lat, lon))
    assert len(lat) == 13                                   # 11.1 km -> every km + the end point
    assert np.allclose(steps[:-1], 1.0, atol=1e-6)
    lat, _ = resample_path([45.0, 46.0], [5.0, 5.0], step_km=0.01, max_points=100)
    assert len(lat) <= 101


class FakeResponse:
    def __init__(self, payload, status=200):
        self.payload, self.status_code = payload, status

    def json(self):
        return self.payload


class FakeSession:
    """Routing answers with a detour through an extra point; elevation answers 100 m everywhere."""

    def __init__(self, routing_ok=True):
        self.routing_ok = routing_ok
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append(url)
        if "routed-" in url:
            if not self.routing_ok:
                return FakeResponse({"code": "Error"}, 503)
            coords = [[5.0, 45.0], [5.02, 45.03], [5.0, 45.05], [5.05, 45.05]]
            return FakeResponse({"code": "Ok", "routes": [{"distance": 1, "duration": 1, "geometry": {"coordinates": coords}}]})
        n = len(params["latitude"].split(","))
        return FakeResponse({"elevation": [100.0] * n})


def test_track_follows_the_route_and_gets_elevation():
    session = FakeSession()
    track, routed = track_from_waypoints(WAYPOINTS, "Cycling", True, session=session)
    straight, _ = track_from_waypoints(WAYPOINTS, "Cycling", False, session=FakeSession())
    assert routed and track.has_elevation
    assert track.distance_km > straight.distance_km        # the routed path makes a detour
    assert "cycling" in track.name
    assert any("routed-bike" in u for u in session.calls)


def test_falls_back_to_straight_lines_when_routing_fails():
    track, routed = track_from_waypoints(WAYPOINTS, "Walking", True, session=FakeSession(routing_ok=False))
    assert not routed
    assert track.distance_km == pytest.approx(5.56 + 3.93, abs=0.1)   # the two straight legs


def test_a_drawn_route_is_named_in_the_page_language():
    assert drawn_name("Cycling", 7.71) == "Drawn route (cycling, 7.7 km)"
    previous = i18n._provider
    i18n.set_provider(lambda: "fr")
    try:
        assert drawn_name("Cycling", 7.71) == "Itinéraire tracé (vélo, 7,7 km)"
        assert drawn_name("Walking", 12.0).startswith("Itinéraire tracé (marche, 12")
    finally:
        i18n.set_provider(previous)


def test_needs_two_points():
    with pytest.raises(ValueError):
        track_from_waypoints([(45.0, 5.0)], session=FakeSession())
