"""Planning a trip: passage times from the travel-time model, the wind-aware pace on a bike, and the best time to leave.

Every departure is read from the forecast already fetched for the route (`route_weather.reread`): sweeping the day
costs no API call. A plan is read the same way whatever the departure, so the reading of the best departure in the
sweep is the reading the page shows once that departure is chosen:
1. times from the slope (`route_timing.moving_hours`, still air);
2. the forecast read at those times;
3. on a bike, times again with the head or tail wind of that reading, and the forecast read at the new times.
"""
from __future__ import annotations

from collections.abc import Callable
from datetime import date, datetime, time

import numpy as np
import pandas as pd

from .gps import Track, times_from_hours, with_times
from .route_scores import DANGER_LEVELS, assess, level
from .route_timing import moving_hours
from .route_weather import RouteReading, reread

STEP_MIN = 30
FIRST_HOUR, LAST_HOUR = 5, 20     # departures swept from 05:00 to 20:00
SAFE_DANGER = 25                  # below this the danger is "Low": the best departure is the most comfortable of those
DARK_SLACK_H = 0.25               # up to 15 minutes in the dark count as daylight when departures are compared


def plan_track(track: Track, start: pd.Timestamp, activity: str, flat_kmh: float, hours_per_day: float | None = None,
               headwind_kmh=None) -> Track:
    """The track with the time at each point for a departure at `start`."""
    hours = moving_hours(track, activity, flat_kmh, headwind_kmh)
    return with_times(track, times_from_hours(track, start, hours, hours_per_day))


def headwind_along(reading: RouteReading, track: Track) -> np.ndarray:
    """Head (+) or tail (-) wind at each point of the track, from the reading's points (km/h)."""
    pts = reading.points
    return np.interp(track.points["dist_km"].to_numpy(dtype=float), pts["dist_km"].to_numpy(dtype=float),
                     np.nan_to_num(pts["headwind"].to_numpy(dtype=float)))


def wind_aware(reading: RouteReading, track: Track, start: pd.Timestamp, activity: str, flat_kmh: float,
               hours_per_day: float | None = None) -> tuple[RouteReading, Track]:
    """On a bike: the times redone with the forecast wind, and the forecast read again at those times. Unchanged on
    foot or by car, or when the forecast fetched does not cover the new times."""
    if activity != "Cycling":
        return reading, track
    timed = plan_track(track, start, activity, flat_kmh, hours_per_day, headwind_along(reading, track))
    again = reread(reading, timed)
    return (again, timed) if again is not None else (reading, track)


def read_plan(fetched: RouteReading, track: Track, start: pd.Timestamp, activity: str, flat_kmh: float,
              hours_per_day: float | None = None) -> RouteReading | None:
    """The reading of a departure at `start`, from a forecast already fetched for this track (None when it does not
    cover the trip)."""
    timed = plan_track(track, start, activity, flat_kmh, hours_per_day)
    first = reread(fetched, timed)
    if first is None:
        return None
    return wind_aware(first, timed, start, activity, flat_kmh, hours_per_day)[0]


def departures(day: date, tz: str, step_min: int = STEP_MIN, first: int = FIRST_HOUR,
               last: int = LAST_HOUR) -> list[pd.Timestamp]:
    out = []
    for minutes in range(first * 60, last * 60 + 1, step_min):
        naive = pd.Timestamp(datetime.combine(day, time(minutes // 60, minutes % 60)))
        out.append(naive.tz_localize(tz, nonexistent="shift_forward", ambiguous=True))
    return out


def sweep(fetched: RouteReading, track: Track, days: list[date], tz: str, activity: str, flat_kmh: float,
          hours_per_day: float | None = None, decorate: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
          now: pd.Timestamp | None = None) -> pd.DataFrame:
    """Danger, comfort, rain, headwind and darkness for every departure of `days`, from the fetched forecast.
    `decorate` adds columns to each reading before it is scored (the official warnings). Departures already past
    (`now`) or whose trip ends beyond the forecast are left out."""
    rows = []
    for day in days:
        for start in departures(day, tz):
            if now is not None and start < now:
                continue
            r = read_plan(fetched, track, start, activity, flat_kmh, hours_per_day)
            if r is None:
                continue
            pts = decorate(r.points) if decorate else r.points
            a = assess(pts, activity, tz)
            w = pts["hours_here"].to_numpy(dtype=float)
            dark = float(np.sum(w[pts["is_day"].fillna(1).to_numpy(dtype=float) < 0.5])) if "is_day" in pts else 0.0
            rows.append({"departure": start, "arrival": pts["time"].max().tz_convert(tz), "danger": a.danger.value,
                         "danger_level": a.danger.level, "comfort": a.comfort.value, "comfort_level": a.comfort.level,
                         "rain_mm": r.stats["rain_mm"], "headwind_share": r.stats["headwind_share"],
                         "dark_hours": dark})
    return pd.DataFrame(rows, columns=["departure", "arrival", "danger", "danger_level", "comfort", "comfort_level",
                                       "rain_mm", "headwind_share", "dark_hours"])


def best(sw: pd.DataFrame) -> pd.Series | None:
    """The best departure: of those with a low danger, the ones that keep the most of the trip in daylight, and of
    these the most comfortable (the earliest on a tie); without any, the least dangerous.
    Darkness only adds a few danger points on foot, so a mild evening could otherwise win over a cool morning and
    send a walker into the night (#77)."""
    if sw.empty:
        return None
    safe = sw[sw["danger"] < SAFE_DANGER]
    if not safe.empty:
        if "dark_hours" in safe:
            safe = safe[safe["dark_hours"] <= safe["dark_hours"].min() + DARK_SLACK_H]
        return safe.sort_values(["comfort", "departure"], ascending=[False, True]).iloc[0]
    return sw.sort_values(["danger", "comfort", "departure"], ascending=[True, False, True]).iloc[0]


def best_window(sw: pd.DataFrame, pick: pd.Series, margin: int = 5) -> tuple[pd.Timestamp, pd.Timestamp]:
    """The run of departures around the best one that stay as good (comfort within `margin` points, same danger
    level, no more darkness)."""
    sw = sw.sort_values("departure").reset_index(drop=True)
    i = int(sw.index[sw["departure"] == pick["departure"]][0])
    ok = (sw["comfort"] >= pick["comfort"] - margin) & (
        sw["danger"].map(lambda v: level(v, DANGER_LEVELS)) == level(pick["danger"], DANGER_LEVELS))
    if "dark_hours" in sw:
        ok &= sw["dark_hours"] <= pick["dark_hours"] + DARK_SLACK_H
    a = b = i
    while a > 0 and ok[a - 1] and (sw["departure"][a] - sw["departure"][a - 1]) <= pd.Timedelta(minutes=STEP_MIN):
        a -= 1
    while b < len(sw) - 1 and ok[b + 1] and (sw["departure"][b + 1] - sw["departure"][b]) <= pd.Timedelta(minutes=STEP_MIN):
        b += 1
    return sw["departure"][a], sw["departure"][b]
