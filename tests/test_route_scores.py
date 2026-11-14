"""Danger and comfort scores along a route, and the advice (meteo/route_scores.py)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from meteo.route_scores import assess, combine, ramp


def reading(n: int = 10, **cols) -> pd.DataFrame:
    """A weather reading like route_weather builds it: mild, dry, calm, by day, unless told otherwise."""
    base = {
        "dist_km": np.linspace(0, 45, n),
        "time": pd.date_range("2026-06-20 08:00", periods=n, freq="15min", tz="UTC"),
        "hours_here": np.full(n, 0.25),
        "temperature_2m": np.full(n, 17.0), "apparent_temperature": np.full(n, 17.0),
        "precipitation": np.zeros(n), "weather_code": np.ones(n, dtype=int),
        "wind_speed_10m": np.full(n, 8.0), "wind_gusts_10m": np.full(n, 15.0),
        "headwind": np.zeros(n), "is_day": np.ones(n), "dew_point_2m": np.full(n, 9.0),
        "snowfall": np.zeros(n), "visibility": np.full(n, 20000.0), "uv_index": np.full(n, 3.0),
    }
    for name, value in cols.items():
        base[name] = np.asarray(value) if np.ndim(value) else np.full(n, value)
    return pd.DataFrame(base)


def at(n: int, i: int, value, default) -> np.ndarray:
    """A column equal to `default` everywhere except at point i."""
    out = np.full(n, default, dtype=float if isinstance(default, float) else object)
    out[i] = value
    return out


# --- Building blocks ------------------------------------------------------------------------
def test_ramp_goes_both_ways_and_ignores_missing_values():
    assert ramp([30, 40, 50, 60], 40, 60).tolist() == [0, 0, 50, 100]
    assert ramp([-5, -20, -40], -10, -30).tolist() == [0, 50, 100]   # cold: the lower, the worse
    assert ramp([np.nan], 0, 10).tolist() == [0]


def test_hazards_add_up_like_independent_risks():
    h = pd.DataFrame({"a": [50.0, 0.0], "b": [50.0, 30.0]})
    assert combine(h).tolist() == pytest.approx([75, 30])


# --- Danger ---------------------------------------------------------------------------------
def test_calm_mild_day_is_safe_and_pleasant():
    a = assess(reading(), "Cycling", "Europe/Paris")
    assert a.danger.value == 0 and a.danger.level == "Low"
    assert a.comfort.value == 100 and a.comfort.level == "Very pleasant"
    assert [x.title for x in a.advice] == ["Good conditions"]


def test_the_same_gusts_are_dangerous_on_a_bike_not_in_a_car():
    gusty = reading(wind_gusts_10m=60.0)
    bike, car = assess(gusty, "Cycling", "UTC"), assess(gusty, "Driving", "UTC")
    assert bike.danger.value == 50 and bike.danger.level == "High"
    assert car.danger.value == 0
    assert bike.danger.factors[0].name == "Gusts"


def test_the_worst_moment_drives_the_route_score():
    n = 10
    df = reading(n, weather_code=at(n, 6, 95, 1))   # a thunderstorm at one point only
    a = assess(df, "Walking", "Europe/Paris")
    # 70% of the peak (70) + 30% of the average over time (7)
    assert a.danger.value == round(0.7 * 70 + 0.3 * 7)
    storm = a.danger.factors[0]
    assert storm.name == "Thunderstorm" and storm.km == pytest.approx(30)
    assert a.advice[0].title.startswith("Thunderstorms around km 30 (11:30)")   # 09:30 UTC, summer time in Paris
    assert a.advice[-1].kind == "info" and "warnings" in a.advice[-1].title


def test_borderline_hazards_get_no_advice_and_warnings_start_at_moderate():
    light = assess(reading(wind_gusts_10m=43.0), "Cycling", "UTC")   # gust score 7.5, under the threshold of 10
    assert not light.danger.factors
    assert not any(x.title.startswith("Gusts") or "warnings" in x.title for x in light.advice)
    strong = assess(reading(wind_gusts_10m=52.0), "Cycling", "UTC")  # 30: moderate
    titles = [x.title for x in strong.advice]
    assert titles[0] == "Gusts up to 52 km/h" and "warnings" in titles[-1]


def test_ice_from_freezing_rain_or_rain_near_zero():
    freezing = assess(reading(weather_code=66), "Driving", "UTC")
    assert freezing.danger.factors[0].name == "Ice" and freezing.danger.factors[0].value == 90
    near_zero = assess(reading(temperature_2m=0.5, apparent_temperature=-3.0, precipitation=0.5, weather_code=61),
                       "Cycling", "UTC")
    assert near_zero.danger.factors[0].name == "Ice" and near_zero.danger.factors[0].value == 70
    assert any(x.title.startswith("Risk of ice") for x in near_zero.advice)


def test_archive_readings_without_visibility_or_uv_still_score():
    df = reading().drop(columns=["visibility", "uv_index", "is_day", "dew_point_2m", "snowfall"])
    a = assess(df, "Walking", "UTC")
    assert a.danger.value == 0 and a.comfort.value == 100


def test_unknown_activity():
    with pytest.raises(ValueError, match="Unknown activity"):
        assess(reading(), "Skiing", "UTC")


# --- Comfort --------------------------------------------------------------------------------
def test_rain_and_headwind_lower_the_comfort_of_a_bike_ride():
    wet = assess(reading(precipitation=1.0, weather_code=61), "Cycling", "UTC")
    assert wet.comfort.value == 100 - 25 and wet.comfort.factors[0].name == "Rain"
    windy = assess(reading(headwind=25.0), "Cycling", "UTC")
    assert windy.comfort.value == 100 - 30 and windy.comfort.factors[0].name == "Wind"
    assert assess(reading(headwind=25.0), "Driving", "UTC").comfort.value == 100
    assert any(x.title.startswith("Headwind on 100%") for x in windy.advice)


def test_comfort_is_averaged_over_time_not_over_points():
    n = 4
    hot = at(n, 0, 32.0, 17.0)   # 10 °C above the walking range at the first point...
    hours = np.array([3.0, 0.25, 0.25, 0.25])   # ...where the walker spends most of the time
    a = assess(reading(n, apparent_temperature=hot, hours_here=hours), "Walking", "UTC")
    assert a.comfort.value == round(100 - 30 * 3 / 3.75)
    assert any(x.title.startswith("Heat") for x in a.advice)


def test_night_advice_depends_on_the_activity():
    df = reading(is_day=at(10, 9, 0.0, 1.0))
    bike = assess(df, "Cycling", "UTC")
    lights = next(x for x in bike.advice if x.title.startswith("In the dark"))
    assert lights.kind == "danger" and "lights" in lights.text
    assert "Headlights" in next(x for x in assess(df, "Driving", "UTC").advice if x.title.startswith("In the dark")).text
    assert not any("warnings" in x.title for x in bike.advice)   # darkness alone: no weather warning to check


def test_points_table_for_the_charts():
    a = assess(reading(precipitation=1.0), "Walking", "UTC")
    assert {"dist_km", "time", "danger", "comfort", "h_Gusts", "p_Rain"} <= set(a.points.columns)
    assert len(a.points) == 10 and a.points["comfort"].between(0, 100).all()
