"""Route planning: travel time from slope and wind (#65), best departure (#54), ensemble spread (#61), official
warnings (#62), air quality (#63), rain in 15-minute steps (#64), and the 2030-2100 projection (#56)."""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from meteo.climate_ahead import LEVEL_NOW, nearest_region, patterns_from_daily, project_city, shifted
from meteo.departure import best, read_plan, sweep, wind_aware
from meteo.ensemble import spread
from meteo.gps import build_track, sample_points, with_times
from meteo.route_scores import OZONE_INFO, assess
from meteo.route_timing import bike_kmh, moving_hours, tobler_kmh
from meteo.route_weather import add_minutely, epoch_seconds, read_route, reread
from meteo.vigilance import at_points, parse

TH = {"hot_day": 30, "very_hot_day": 35, "tropical_night": 20, "frost_day": 0}


# --- Travel time --------------------------------------------------------------------------------------
def hill_track(climb_m=600.0, km=20.0):
    """North along a meridian: up `climb_m` over the first half, down again over the second."""
    n = 401
    lat = 45.0 + np.linspace(0, km / 111.2, n)
    ele = 200 + climb_m * (1 - np.abs(np.linspace(-1, 1, n)))
    return build_track(list(zip(lat, np.full(n, 6.0), ele, [None] * n, strict=True)), "Hill")


def test_descents_are_faster_than_the_flat_and_climbs_slower():
    for v in (bike_kmh, tobler_kmh):
        flat, down, up = (float(np.asarray(v(s, 20.0)).ravel()[0]) for s in (0.0, -0.06, 0.06))
        assert down > flat > up
    assert float(tobler_kmh(-0.05, 5.0)) == pytest.approx(5.0 * np.exp(0.175))   # Tobler's fastest slope
    assert float(bike_kmh(-0.3, 20.0)) == 55.0 and float(bike_kmh(0.3, 20.0)) == 4.0   # capped both ways
    t = hill_track()
    hours = moving_hours(t, "Cycling", 20.0)
    half = len(hours) // 2
    assert hours[-1] - hours[half] < hours[half]   # the way down takes less time than the way up


def test_a_headwind_lengthens_a_bike_ride_and_a_tailwind_shortens_it():
    t = hill_track(climb_m=0)
    still = moving_hours(t, "Cycling", 20.0)[-1]
    against = moving_hours(t, "Cycling", 20.0, headwind_kmh=np.full(len(t.points), 25.0))[-1]
    behind = moving_hours(t, "Cycling", 20.0, headwind_kmh=np.full(len(t.points), -25.0))[-1]
    assert still == pytest.approx(1.0, rel=0.01)   # 20 km at 20 km/h on the flat
    assert against > 1.4 * still and behind < 0.8 * still
    assert moving_hours(t, "Driving", 80.0)[-1] == pytest.approx(0.25, rel=0.01)


# --- A fake Open-Meteo: weather that changes over the day ----------------------------------------------
class Weather:
    """Forecast, archive and air quality answers. Mornings are windy from the north (a headwind riding north),
    afternoons calm; one afternoon hour has an ozone peak."""

    def __init__(self):
        self.calls = []

    def get(self, url, params, timeout=60, headers=None):
        self.calls.append(url)
        n = len(params["latitude"].split(","))
        start = pd.Timestamp(params.get("start_date", "2026-11-14"), tz="UTC")
        times = pd.date_range(start, periods=24 * 6, freq="h")
        hour = times.hour.to_numpy()
        if "air-quality" in url:
            block = {"time": epoch_seconds(times).astype(int).tolist(),
                     "european_aqi": np.where(hour == 15, 95, 25).tolist(),
                     "ozone": np.where(hour == 15, 200, 60).tolist(), "pm10": [12] * len(times),
                     "pm2_5": [6] * len(times), "nitrogen_dioxide": [10] * len(times)}
            loc = {"hourly": block}
        else:
            wind = np.where(hour < 12, 30.0, 3.0)
            block = {"time": epoch_seconds(times).astype(int).tolist(),
                     "temperature_2m": (8 + 6 * np.sin((hour - 9) / 24 * 2 * np.pi)).tolist(),
                     "apparent_temperature": (6 + 6 * np.sin((hour - 9) / 24 * 2 * np.pi)).tolist(),
                     "precipitation": [0.0] * len(times), "precipitation_probability": [5] * len(times),
                     "weather_code": [1] * len(times), "wind_speed_10m": wind.tolist(),
                     "wind_direction_10m": [0.0] * len(times), "wind_gusts_10m": (wind * 1.6).tolist(),
                     "cloud_cover": [20] * len(times), "is_day": ((hour >= 7) & (hour < 17)).astype(int).tolist()}
            loc = {"elevation": 300.0, "timezone": "Europe/Paris", "hourly": block}

        class Resp:
            status_code = 200

            @staticmethod
            def json():
                return [loc] * n if n > 1 else loc
        return Resp()


def planned(track, when="2026-11-15 08:00"):
    from meteo.departure import plan_track
    return plan_track(track, pd.Timestamp(when, tz="Europe/Paris"), "Cycling", 20.0)


def test_a_reading_keeps_its_forecast_and_can_be_read_again_for_free():
    t = hill_track(climb_m=0)
    w = Weather()
    r = read_route(planned(t), session=w, today=date(2026, 11, 14))
    assert any("air-quality" in u for u in w.calls)
    n_calls = len(w.calls)
    same = reread(r, planned(t))
    pd.testing.assert_frame_equal(same.points, r.points)
    later = reread(r, planned(t, "2026-11-15 14:00"))
    assert later.points["wind_speed_10m"].max() < r.points["wind_speed_10m"].max()   # calm afternoon
    assert len(w.calls) == n_calls                                                   # no new request
    assert reread(r, planned(t, "2026-11-25 08:00")) is None                         # beyond what was fetched


def test_the_wind_aware_pace_reads_the_weather_at_the_new_times():
    t = hill_track(climb_m=0)
    start = pd.Timestamp("2026-11-15 08:00", tz="Europe/Paris")
    r = read_route(planned(t), session=Weather(), today=date(2026, 11, 14))
    r2, t2 = wind_aware(r, planned(t), start, "Cycling", 20.0)
    took = (t2.points["time"].iloc[-1] - t2.points["time"].iloc[0]).total_seconds() / 3600
    assert took > 1.2                     # 30 km/h of headwind: well over the hour of still air
    assert r2.points["time"].max() == pd.Timestamp(t2.points["time"].iloc[-1])


def test_the_best_departure_is_the_reading_shown_once_chosen():
    t = hill_track(climb_m=0)
    w = Weather()
    fetched = read_route(planned(t), session=w, today=date(2026, 11, 14))
    sw = sweep(fetched, t, [date(2026, 11, 15)], "Europe/Paris", "Cycling", 20.0)
    assert len(sw) > 20 and sw["comfort"].nunique() > 1
    pick = best(sw)
    assert pick["departure"].hour >= 12            # the calm afternoon beats the windy morning
    # The page reads a chosen departure the same way: new times, a new request, then the wind-aware pace
    start = pick["departure"]
    new = read_route(planned(t, start.tz_convert("Europe/Paris").strftime("%Y-%m-%d %H:%M")), session=Weather(),
                     today=date(2026, 11, 14))
    shown = wind_aware(new, planned(t, start.tz_convert("Europe/Paris").strftime("%Y-%m-%d %H:%M")), start,
                       "Cycling", 20.0)[0]
    a = assess(shown.points, "Cycling", "Europe/Paris")
    assert (a.danger.value, a.comfort.value) == (pick["danger"], pick["comfort"])
    assert read_plan(fetched, t, pd.Timestamp("2026-11-30 08:00", tz="Europe/Paris"), "Cycling", 20.0) is None


# --- Air quality, official warnings, 15-minute rain ------------------------------------------------------
def test_an_ozone_peak_lowers_comfort_and_is_advised_on():
    t = hill_track(climb_m=0)
    r = read_route(planned(t, "2026-11-15 15:40"), session=Weather(), today=date(2026, 11, 14))  # 14:40 UTC
    assert r.points["ozone"].max() >= OZONE_INFO
    with_ozone = assess(r.points, "Cycling", "Europe/Paris")
    clean = assess(r.points.assign(ozone=60.0, european_aqi=25.0), "Cycling", "Europe/Paris")
    assert with_ozone.comfort.value < clean.comfort.value
    assert any("ozone" in a.title.lower() for a in with_ozone.advice)
    assert not any("ozone" in a.title.lower() for a in clean.advice)


VIGILANCE = {"product": {"periods": [
    {"echeance": "J", "begin_validity_time": "2026-11-15T00:00:00Z", "end_validity_time": "2026-11-16T00:00:00Z",
     "timelaps": {"domain_ids": [
         {"domain_id": "38", "max_color_id": 3, "phenomenon_items": [
             {"phenomenon_id": "3", "phenomenon_max_color_id": 3}, {"phenomenon_id": "1", "phenomenon_max_color_id": 2}]},
         {"domain_id": "7301", "max_color_id": 2, "phenomenon_items": [{"phenomenon_id": "5", "phenomenon_max_color_id": 2}]},
         {"domain_id": "26", "max_color_id": 1, "phenomenon_items": []}]}}]}}


def test_an_orange_warning_on_the_route_raises_the_danger_and_the_advice():
    w = parse(VIGILANCE)
    assert set(w["domain"]) == {"38", "7301", "26"}
    pts = pd.DataFrame({"dist_km": [0.0, 10.0, 20.0],
                        "time": pd.to_datetime(["2026-11-15 08:00", "2026-11-15 09:00", "2026-11-16 09:00"], utc=True)})
    deps = [{"dist_km": 0.0, "code": "26", "name": "Drôme"}, {"dist_km": 10.0, "code": "38", "name": "Isère"},
            {"dist_km": 20.0, "code": "73", "name": "Savoie"}]
    v = at_points(pts, deps, w)
    assert v["vigilance_color"].tolist() == [1, 3, 1]    # green; orange; Savoie's sector, but the day after
    assert "thunderstorms" in v["vigilance_text"].iloc[1] and "Isère (38)" in v["vigilance_text"].iloc[1]
    v2 = at_points(pts.assign(time=pd.to_datetime(["2026-11-15 08:00"] * 3, utc=True)), deps, w)
    assert v2["vigilance_color"].tolist() == [1, 3, 2]   # a sector of Savoie counts for the whole département
    r = read_route(planned(hill_track(climb_m=0)), session=Weather(), today=date(2026, 11, 14))
    base = assess(r.points, "Cycling", "Europe/Paris")
    warned = assess(r.points.assign(vigilance_color=3, vigilance_text="Orange vigilance for thunderstorms"),
                    "Cycling", "Europe/Paris")
    assert warned.danger.value > base.danger.value + 25
    assert warned.advice[0].title.startswith("Official orange warning")
    with pytest.raises(RuntimeError):
        parse({"unexpected": True})


def test_rain_in_15_minute_steps_replaces_the_hourly_rain():
    times = pd.to_datetime(["2026-11-15 08:00", "2026-11-15 08:20", "2026-11-15 12:00"], utc=True)
    df = pd.DataFrame({"time": times, "precipitation": [0.0, 0.0, 0.5]})
    q = pd.date_range("2026-11-15 07:00", periods=12, freq="15min", tz="UTC")
    rain = np.zeros(12)
    rain[5:7] = 1.0                       # 1 mm in each 15 minutes from 08:15 to 08:30
    frames = [pd.DataFrame({"time": q, "precipitation": rain})] * 3
    out = add_minutely(df.copy(), frames)
    assert out["precipitation"].iloc[1] == pytest.approx(4.0)   # 1 mm per 15 min = 4 mm/h
    assert out["precip_15min"].tolist() == [True, True, False]  # 12:00 is past the 15-minute values
    assert out["precipitation"].iloc[2] == 0.5


# --- Ensemble ---------------------------------------------------------------------------------------------
def ensemble_frame(spread_c, rainy_share, times):
    rng = np.random.default_rng(3)
    cols = {"time": times}
    for m in range(51):
        name = "" if m == 0 else f"_member{m:02d}"
        cols[f"temperature_2m{name}"] = 10 + rng.normal(0, spread_c / 2.56, len(times))
        cols[f"precipitation{name}"] = np.full(len(times), 1.0 if m < rainy_share * 51 else 0.0)
        cols[f"wind_speed_10m{name}"] = np.full(len(times), 10.0)
        cols[f"wind_gusts_10m{name}"] = np.full(len(times), 50.0 if m < 10 else 20.0)
    return pd.DataFrame(cols)


def test_a_wider_spread_of_runs_gives_wider_bands_and_less_confidence():
    times = pd.date_range("2026-11-15", periods=48, freq="h", tz="UTC")
    reading = pd.DataFrame({"dist_km": np.linspace(0, 20, 5), "time": times[8:13]})
    idx = np.arange(5)
    tomorrow = spread([ensemble_frame(1.0, 0.0, times)] * 5, idx, reading, gust_danger=40)
    in_six_days = spread([ensemble_frame(6.0, 0.5, times)] * 5, idx, reading, gust_danger=40)
    assert in_six_days.temp_spread > 3 * tomorrow.temp_spread
    assert (tomorrow.confidence, in_six_days.confidence) == ("High", "Low")
    assert in_six_days.rain_anywhere == pytest.approx(26 / 51)
    assert tomorrow.hazard_shares["gust"] == pytest.approx(10 / 51)


# --- 2030-2100 ----------------------------------------------------------------------------------------------
def test_model_patterns_and_the_projection_of_a_city():
    dates = pd.date_range("1995-01-01", "1996-12-31", freq="D")
    summer = dates.month.isin([6, 7, 8])
    past = pd.DataFrame({"date": dates, "window": 0, "tx": 20.0, "tn": 10.0, "rr": 2.0})
    future = past.assign(window=1, tx=np.where(summer, 23.0, 21.0), tn=11.0, rr=np.where(summer, 1.6, 2.0))
    p = patterns_from_daily(pd.concat([past, future])).set_index("season")
    warming = p["warming"].iloc[0]
    assert p.loc["JJA", "d_tx"] > p.loc["DJF", "d_tx"]            # summer days warm faster
    assert p.loc["JJA", "r_rr"] < 0                              # and get drier
    assert p.loc["DJF", "d_tx"] == pytest.approx(1 / warming)

    days = pd.date_range("1961-01-01", "2025-12-31", freq="D")
    city = pd.DataFrame({"date": days, "tx": np.where(days.month.isin([7]), 29.5, 15.0), "tn": 5.0, "rr": 1.0})
    models = pd.concat([p.reset_index().assign(model="A"),
                        p.reset_index().assign(model="B", d_tx=p["d_tx"].to_numpy() * 2)])
    out = project_city(city, list(range(1961, 2026)), models, TH, (1961, 1990), (2011, 2025))
    hot = out[out["indicator"] == "hot_days"].set_index("period")
    assert hot.loc["2011-2025", "median"] == 0                   # 29.5 °C: just below 30
    assert hot.loc["2100", "low"] <= hot.loc["2100", "median"] <= hot.loc["2100", "high"]
    assert hot.loc["2100", "high"] == pytest.approx(31)   # every July day crosses 30 °C
    assert shifted(city.head(3), p.reset_index(), 0.0)["tx"].tolist() == [15.0] * 3   # no warming to come
    assert shifted(city.head(3), p.reset_index(), 4.0 - LEVEL_NOW)["tx"].iloc[0] > 15
    assert nearest_region(43.3, 5.4) == "South" and nearest_region(48.4, -4.5) == "West"


def test_sample_points_remember_their_track_rows():
    t = hill_track()
    s = sample_points(with_times(t, pd.Series(pd.date_range("2026-11-15", periods=len(t.points), freq="min",
                                                            tz="UTC"))))
    assert (t.points["dist_km"].to_numpy()[s["track_idx"]] == s["dist_km"].to_numpy()).all()
