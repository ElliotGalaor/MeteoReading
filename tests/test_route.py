"""Tests of the GPS track parsing and of the weather reading along a route (no network)."""
from __future__ import annotations

import json
from datetime import date

import numpy as np
import pandas as pd
import pytest

from meteo.gps import (
    bearing_deg,
    build_track,
    haversine_km,
    parse_track,
    passage_times,
    sample_points,
    summary,
    to_gpx,
    with_times,
)
from meteo.route_weather import (
    build_reading_table,
    choose_source,
    epoch_seconds,
    interpolate_at,
    read_route,
    summarize,
    wind_components,
)

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
    with pytest.raises(ValueError, match="entities"):  # defusedxml refuses entity declarations
        parse_track(bomb, "bomb.gpx")


@pytest.mark.parametrize(("data", "filename", "message"), [
    (b"", "empty.gpx", "empty"),
    (b"   \n", "empty.kml", "empty"),
    (GPX[: len(GPX) // 2], "cut.gpx", "not a valid GPX file"),
    (bytes(range(256)) * 4, "binary.gpx", "not a valid GPX file"),
    (b"<kml><Document>", "cut.kml", "not a valid KML file"),
    (b"", "empty.geojson", "empty"),
    (b'{"type": "LineString", "coordinates": [[2.35, 48.85]', "cut.geojson", "not a valid GeoJSON file"),
    (bytes(range(128, 256)), "binary.geojson", "not a valid GeoJSON file"),
    (b"[[2.35, 48.85], [2.13, 48.8]]", "array.geojson", "no Feature"),
    (b'{"type": "LineString", "coordinates": [[2.35], [2.13]]}', "short.geojson", "No line or points"),
])
def test_unreadable_files_give_a_plain_message(data, filename, message):
    # Only ValueError, which the page shows as a message (anything else would be a traceback)
    with pytest.raises(ValueError, match=message):
        parse_track(data, filename)


def test_short_coordinates_are_skipped():
    geo = {"type": "LineString", "coordinates": [[2.35], [2.35, 48.85], "x", [2.13, 48.80, 130]]}
    t = parse_track(json.dumps(geo).encode(), "line.geojson")
    assert len(t.points) == 2 and t.points["ele"].iloc[-1] == 130
    kml = b"""<kml xmlns:gx="http://www.google.com/kml/ext/2.2"><Placemark><gx:Track>
      <when>2025-07-15T06:00:00Z</when><gx:coord>5.0</gx:coord>
      <when>2025-07-15T06:10:00Z</when><gx:coord>5.0 45.0 200</gx:coord>
      <when>2025-07-15T06:20:00Z</when><gx:coord>5.0 45.05 210</gx:coord></gx:Track></Placemark></kml>"""
    assert len(parse_track(kml, "track.kml").points) == 2


def test_geojson_points_are_read_like_gpx_waypoints():
    geo = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "geometry": {"type": "Point", "coordinates": [2.35, 48.85]}, "properties": {}},
        {"type": "Feature", "geometry": {"type": "Point", "coordinates": [2.13, 48.80]}, "properties": {}}]}
    t = parse_track(json.dumps(geo).encode(), "points.geojson")
    assert len(t.points) == 2 and t.distance_km > 10


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
    assert choose_source(date(2026, 9, 1), date(2026, 9, 1), today) == "archive"   # the forecast API has no values then
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


@pytest.mark.parametrize(("rain", "prob", "start"), [
    (0.0, 5.0, "Dry all the way"),
    (0.1, 92.0, "Light rain or snow possible"),     # under the rain threshold, but likely: not "dry" (#78)
    (0.0, 60.0, "Light rain or snow possible"),
])
def test_dry_only_when_the_amount_and_the_probability_are_low(rain, prob, start):
    t = parse_track(GPX, "ride.gpx")
    samples = sample_points(t)
    hourly = [_hourly(precipitation=np.full(48, rain), precipitation_probability=np.full(48, prob))
              for _ in range(len(samples))]
    _, insights = summarize(build_reading_table(samples, hourly, [200] * len(samples)), "Europe/Paris")
    assert insights[0].startswith(start)


class FakeSession:
    """Stands in for requests.Session: returns one location per requested coordinate.

    `empty` lists the APIs ("forecast", "archive") that answer with nulls only, as the forecast API does for dates
    it accepts but no longer has."""

    def __init__(self, empty=()):
        self.empty, self.calls = empty, []

    def get(self, url, params, timeout):
        api = "archive" if "archive" in url else "forecast"
        self.calls.append(api)
        n = len(params["latitude"].split(","))
        times = pd.date_range(params["start_date"], periods=24 * 4, freq="h", tz="UTC")

        def values(v):
            return [None if api in self.empty else v] * len(times)
        loc = {"elevation": 250.0, "timezone": "Europe/Paris",
               "hourly": {"time": epoch_seconds(times).astype(int).tolist(),
                          **{k: values(12.0) for k in ("temperature_2m", "apparent_temperature",
                                                       "wind_speed_10m", "wind_gusts_10m", "cloud_cover")},
                          "precipitation": values(0.0), "precipitation_probability": values(5),
                          "weather_code": values(1), "wind_direction_10m": values(180.0)}}

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


def _ride_on(day: str):
    t = parse_track(GPX, "ride.gpx")
    return with_times(t, passage_times(t, pd.Timestamp(f"{day} 08:00", tz="Europe/Paris"), speed_kmh=15))


def test_empty_forecast_falls_back_to_the_archive():
    session = FakeSession(empty=("forecast",))
    r = read_route(_ride_on("2026-10-15"), session=session, today=date(2026, 11, 14))
    assert session.calls == ["forecast", "archive"]
    assert r.source == "archive" and r.points["temperature_2m"].notna().all()
    assert r.points["condition"].ne("Unknown").all()


def test_no_weather_data_gives_a_clear_message():
    # Two days ago: too recent for the archive, and the forecast API answers with nulls
    with pytest.raises(ValueError, match="no data for the times of this trip"):
        read_route(_ride_on("2026-11-12"), session=FakeSession(empty=("forecast", "archive")), today=date(2026, 11, 14))
    with pytest.raises(ValueError, match="no data"):
        read_route(_ride_on("2025-07-15"), session=FakeSession(empty=("archive",)), today=date(2026, 11, 14))


def test_no_value_is_repeated_beyond_the_forecast_horizon():
    # The rain probability stops after 10 hours while the other variables go on (as near Open-Meteo's horizon)
    prob = np.full(48, np.nan)
    prob[:10] = np.linspace(10, 55, 10)
    h = _hourly(precipitation_probability=prob)
    t0 = pd.Timestamp("2025-07-15 00:00", tz="UTC")
    assert interpolate_at(h, t0 + pd.Timedelta(hours=4, minutes=30))["precipitation_probability"] == pytest.approx(32.5)
    assert interpolate_at(h, t0 + pd.Timedelta(hours=9, minutes=30))["precipitation_probability"] == pytest.approx(55)
    assert np.isnan(interpolate_at(h, t0 + pd.Timedelta(hours=20))["precipitation_probability"])
    assert interpolate_at(h, t0 + pd.Timedelta(hours=20))["temperature_2m"] == pytest.approx(30)


def test_short_gaps_are_bridged_long_gaps_are_not():
    from meteo.route_weather import interp_valid

    x = np.arange(0, 10) * 3600.0
    y = np.arange(0, 10, dtype=float)
    y[3:5] = np.nan                                              # 3 hours between the valid values at 2 h and 5 h
    assert interp_valid(3.5 * 3600, x, y) == pytest.approx(3.5)
    y[3:7] = np.nan                                              # 5 hours between 2 h and 7 h: too long
    assert np.isnan(interp_valid(4 * 3600, x, y))
    assert interp_valid(-1800, x, y) == 0 and np.isnan(interp_valid(-7200, x, y))


def test_times_show_the_zone_when_the_clocks_change():
    from meteo.route_weather import clock_format

    night = pd.Series(pd.date_range("2026-10-25 00:30", periods=4, freq="30min", tz="UTC")).dt.tz_convert("Europe/Paris")
    assert night.dt.strftime(clock_format(night)).tolist() == ["02:30 CEST", "02:00 CET", "02:30 CET", "03:00 CET"]
    day = pd.Series(pd.date_range("2026-07-14 08:00", periods=3, freq="h", tz="Europe/Paris"))
    assert clock_format(day) == "%H:%M"
    two_days = pd.Series(pd.date_range("2026-07-14 20:00", periods=8, freq="h", tz="Europe/Paris"))
    assert clock_format(two_days) == "%a %H:%M"


def test_track_timezone_from_the_weather_service():
    from meteo.route_weather import track_timezone

    class Tz:
        def __init__(self, answer):
            self.answer = answer

        def get(self, url, params, timeout):
            if isinstance(self.answer, Exception):
                raise self.answer

            class Resp:
                status_code = 200
                json = staticmethod(lambda: {"timezone": self.answer})
            return Resp()

    assert track_timezone(35.68, 139.69, session=Tz("Asia/Tokyo")) == "Asia/Tokyo"
    assert track_timezone(35.68, 139.69, session=Tz(ConnectionError("offline"))) is None


def test_altitude_noise_is_not_counted_as_climbing():
    from meteo.gps import elevation_stats

    rng = np.random.default_rng(0)
    flat = pd.Series(100 + rng.uniform(-10, 10, 200))                 # ±10 m of noise on a flat road
    climb = pd.Series(np.linspace(100, 400, 200) + rng.uniform(-10, 10, 200))
    assert elevation_stats(flat)["ascent_m"] < 40                       # summing every difference gives ~1,300 m
    assert elevation_stats(climb)["ascent_m"] == pytest.approx(300, abs=40)


def test_unusable_timestamps_say_why():
    rows = [(45.0, 5.0 + i / 100, 200, f"2025-07-15T06:{i:02d}:00Z") for i in range(10)]
    rows[4], rows[5] = rows[5], rows[4]
    t = build_track(rows)
    assert not t.has_time and t.time_issue == "unordered"
    assert build_track([(45.0, 5.0 + i / 100, 200, None) for i in range(10)]).time_issue is None


def test_activity_is_guessed_from_whole_words(monkeypatch):
    monkeypatch.syspath_prepend("dashboard")
    from route_page import _guess_activity

    assert _guess_activity("Carnac coastal ride") == "Cycling"
    assert _guess_activity("Brunoy loop") == "Cycling"
    assert _guess_activity("Morning run") == "Walking"
    assert _guess_activity("Road trip to Nice") == "Driving"
    assert _guess_activity("Lyon to Marseille by car") == "Driving"
    assert _guess_activity("Two cars, one van") == "Driving"                                       # plurals too
    assert _guess_activity("Rouge A/R /les-sentiers-balises/trace-rouge-gpx-et-carte/") == "Walking"  # a link (#88)
