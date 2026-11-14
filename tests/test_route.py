"""Tests of the GPS track parsing and of the weather reading along a route (no network)."""
from __future__ import annotations

import json
from datetime import date

import numpy as np
import pandas as pd
import pytest

from meteo.gps import (bearing_deg, build_track, haversine_km, parse_track, passage_times, sample_points,
                       summary, to_gpx, with_times)
from meteo.route_weather import (choose_source, interpolate_at, read_route, summarize, wind_components,
                                 build_reading_table)

GPX = b"""<?xml version="1.0" encoding="UTF-8"?>
<gpx version="1.1" creator="test" xmlns="http://www.topografix.com/GPX/1/1">
  <metadata><name>Morning ride</name></metadata>
  <trk><trkseg>
    <trkpt lat="45.0000" lon="5.0000"><ele>200</ele><time>2025-07-15T06:00:00Z</time></trkpt>
    <trkpt lat="45.0000" lon="5.0000"><ele>200</ele><time>2025-07-15T06:00:05Z</time></trkpt>
    <trkpt lat="45.0450" lon="5.0000"><ele>300</ele><time>2025-07-15T06:15:00Z</time></trkpt>
    <trkpt lat="45.0900" lon="5.0000"><ele>450</ele><time>2025-07-15T06:30:00Z</time></trkpt>
  </trkseg></trk>
</gpx>"""


# --- Parsing ---------------------------------------------------------------------------
def test_gpx_track_with_times_and_elevation():
    t = parse_track(GPX, "ride.gpx")
    assert t.name == "Morning ride"
    assert len(t.points) == 3                      # the repeated point is dropped
    assert t.has_time and t.has_elevation
    assert t.distance_km == pytest.approx(10.0, abs=0.1)   # 0.09° of latitude ≈ 10 km


def test_gpx_route_without_times():
    gpx = b"""<gpx xmlns="http://www.topografix.com/GPX/1/0"><rte>
      <rtept lat="48.85" lon="2.35"/><rtept lat="48.80" lon="2.12"/></rte></gpx>"""
    t = parse_track(gpx, "plan.gpx")
    assert not t.has_time and not t.has_elevation
    assert t.name == "plan"


def test_kml_linestring_and_gx_track():
    line = b"""<kml xmlns="http://www.opengis.net/kml/2.2"><Placemark><name>Hike</name><LineString>
      <coordinates>5.0,45.0,200 5.01,45.01,250 5.02,45.02,300</coordinates></LineString></Placemark></kml>"""
    t = parse_track(line, "hike.kml")
    assert t.name == "Hike" and len(t.points) == 3 and t.points["lat"].iloc[0] == 45.0
    gx = b"""<kml xmlns="http://www.opengis.net/kml/2.2" xmlns:gx="http://www.google.com/kml/ext/2.2">
      <Placemark><gx:Track><when>2025-01-01T10:00:00Z</when><when>2025-01-01T10:10:00Z</when>
      <gx:coord>5.0 45.0 200</gx:coord><gx:coord>5.0 45.02 210</gx:coord></gx:Track></Placemark></kml>"""
    assert parse_track(gx, "rec.kml").has_time


def test_geojson_with_coord_times():
    obj = {"type": "Feature", "properties": {"name": "Run", "coordTimes": ["2025-01-01T10:00:00Z", "2025-01-01T10:05:00Z"]},
           "geometry": {"type": "LineString", "coordinates": [[2.35, 48.85, 35], [2.36, 48.86, 40]]}}
    t = parse_track(json.dumps(obj).encode(), "run.geojson")
    assert t.name == "Run" and t.has_time and len(t.points) == 2


def test_rejects_bad_files():
    with pytest.raises(ValueError):
        parse_track(b"lat,lon\n1,2", "track.csv")
    with pytest.raises(ValueError):
        parse_track(b"<gpx></gpx>", "empty.gpx")
    bomb = b"""<?xml version="1.0"?><!DOCTYPE lolz [<!ENTITY lol "lol"><!ENTITY lol2 "&lol;&lol;&lol;">]>
      <gpx><wpt lat="1" lon="1"><name>&lol2;</name></wpt></gpx>"""
    with pytest.raises(Exception):          # defusedxml refuses entity declarations
        parse_track(bomb, "bomb.gpx")


def test_non_monotonic_times_are_ignored():
    rows = [(45.0, 5.0, None, "2025-01-01T10:00:00Z"), (45.01, 5.0, None, "2025-01-01T09:00:00Z"),
            (45.02, 5.0, None, "2025-01-01T11:00:00Z")]
    assert not build_track(rows).has_time


def test_gpx_round_trip():
    t = parse_track(GPX, "ride.gpx")
    back = parse_track(to_gpx(t).encode(), "copy.gpx")
    assert len(back.points) == len(t.points) and back.has_time


# --- Geometry and timing -----------------------------------------------------------------
def test_haversine_and_bearing():
    assert float(haversine_km(48.8566, 2.3522, 45.7640, 4.8357)) == pytest.approx(392, abs=3)  # Paris-Lyon
    assert float(bearing_deg(45, 5, 46, 5)) == pytest.approx(0, abs=0.1)
    assert float(bearing_deg(45, 5, 45, 6)) == pytest.approx(90, abs=1)


def test_sampling_keeps_ends_and_respects_limits():
    rows = [(45 + i * 0.001, 5.0, None, None) for i in range(2000)]   # ~222 km
    t = build_track(rows)
    s = sample_points(t, max_points=40, min_step_km=1.0)
    assert len(s) <= 41
    assert s["dist_km"].iloc[0] == 0 and s["dist_km"].iloc[-1] == pytest.approx(t.distance_km)
    assert (s["bearing"].round() == 0).all()   # heading due north


def test_passage_times_with_climb_penalty():
    t = parse_track(GPX, "ride.gpx")
    start = pd.Timestamp("2025-07-15 08:00", tz="Europe/Paris")
    flat = passage_times(t, start, speed_kmh=10)
    assert (flat.iloc[-1] - flat.iloc[0]).total_seconds() / 3600 == pytest.approx(1.0, abs=0.02)
    climb = passage_times(t, start, speed_kmh=10, climb_min_per_100m=10)
    assert climb.iloc[-1] > flat.iloc[-1]
    assert summary(with_times(t, flat))["duration_h"] == pytest.approx(1.0, abs=0.02)


# --- Weather ---------------------------------------------------------------------------------
def test_source_selection():
    today = date(2026, 11, 14)
    assert choose_source(date(2026, 11, 15), date(2026, 11, 15), today) == "forecast"
    assert choose_source(date(2026, 10, 1), date(2026, 10, 1), today) == "forecast"
    assert choose_source(date(2025, 7, 15), date(2025, 7, 15), today) == "archive"
    with pytest.raises(ValueError):
        choose_source(date(2026, 12, 25), date(2026, 12, 25), today)


def _hourly(start="2025-07-15 00:00", hours=48, **cols):
    times = pd.date_range(start, periods=hours, freq="h", tz="UTC")
    base = {"temperature_2m": np.linspace(10, 10 + hours - 1, hours), "apparent_temperature": np.full(hours, 9.0),
            "precipitation": np.zeros(hours), "precipitation_probability": np.full(hours, 10.0),
            "weather_code": np.zeros(hours, dtype=int), "wind_speed_10m": np.full(hours, 20.0),
            "wind_direction_10m": np.zeros(hours), "wind_gusts_10m": np.full(hours, 30.0), "cloud_cover": np.zeros(hours)}
    base.update(cols)
    return pd.DataFrame({"time": times, **base})


def test_interpolation_is_linear_in_time_and_vector_for_wind():
    wd = np.zeros(48)
    wd[6], wd[7] = 350, 10
    h = _hourly(wind_direction_10m=wd)
    out = interpolate_at(h, pd.Timestamp("2025-07-15 06:30", tz="UTC"))
    assert out["temperature_2m"] == pytest.approx(16.5)
    assert min(out["wind_direction_10m"], 360 - out["wind_direction_10m"]) == pytest.approx(0, abs=0.5)


def test_wind_components():
    head, cross = wind_components([20, 20, 20], [0, 0, 0], [0, 180, 90])   # wind from the north
    assert head.tolist() == pytest.approx([20, -20, 0], abs=1e-9)
    assert cross.tolist() == pytest.approx([0, 0, 20], abs=1e-9)


def test_summary_finds_rain_and_gusts():
    t = parse_track(GPX, "ride.gpx")
    samples = sample_points(t)
    rain = np.zeros(48)
    rain[6] = 3.0
    hourly = [_hourly(precipitation=rain, wind_gusts_10m=np.full(48, 70.0)) for _ in range(len(samples))]
    df = build_reading_table(samples, hourly, [200] * len(samples))
    stats, insights = summarize(df, "Europe/Paris")
    assert stats["rain_mm"] > 0
    assert any(i.startswith("Rain likely") for i in insights)
    assert any("gusts" in i for i in insights)


class FakeSession:
    """Stands in for requests.Session: returns one location per requested coordinate."""

    def get(self, url, params, timeout):
        n = len(params["latitude"].split(","))
        times = pd.date_range(params["start_date"], periods=24 * 4, freq="h", tz="UTC")
        loc = {"elevation": 250.0, "timezone": "Europe/Paris",
               "hourly": {"time": (times.astype("int64") // 10**9).tolist(),
                          **{k: [12.0] * len(times) for k in ("temperature_2m", "apparent_temperature",
                                                              "wind_speed_10m", "wind_gusts_10m", "cloud_cover")},
                          "precipitation": [0.0] * len(times), "precipitation_probability": [5] * len(times),
                          "weather_code": [1] * len(times), "wind_direction_10m": [180.0] * len(times)}}

        class Resp:
            status_code = 200

            @staticmethod
            def json():
                return [loc] * n if n > 1 else loc
        return Resp()


def test_read_route_end_to_end_without_network():
    t = parse_track(GPX, "ride.gpx")
    r = read_route(t, session=FakeSession(), today=date(2026, 11, 14))
    assert r.source == "archive" and r.timezone == "Europe/Paris"
    assert len(r.points) >= 2
    assert (r.points["headwind"] < 0).all()   # wind from the south, riding north: tailwind
    assert r.insights[0].startswith("Dry all the way")
