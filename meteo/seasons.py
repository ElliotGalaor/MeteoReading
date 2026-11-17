"""Best weeks of the year for walking or cycling in a city, and the dates of its warm and cold seasons.

Comfortable day (daily values only: TX, TN, RR and the mean wind FFM)
    - daytime temperature within the activity's comfortable range, the range where the route comfort score loses no
      temperature points (`meteo/route_scores.py`, PROFILES[...].comfort_range). The daytime temperature is
      0.75 TX + 0.25 TN, about the mean of 10:00 to 18:00;
    - a dry day: rainfall under the rainy-day threshold (1 mm);
    - mean wind under WIND_COMFORT_KMH (20 km/h), where the walking comfort score starts to lose points for wind.
      A daily mean has no direction, so cycling uses the same limit instead of the headwind.
    Sunshine and humidity are not in every station's files: they are not used.

Seasons
    The warm season is the run of days whose normal maximum (smoothed over ±15 days) is at least SUMMER_TX, the cold
    season the run where it is at most WINTER_TX. Some cities have neither: Brest's normal maximum never reaches
    25 °C, Nice's never falls to 10 °C.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .climatology import day_of_year, doy_climatology
from .route_scores import PROFILES, WIND_COMFORT_KMH

SUMMER_TX = 25.0          # °C: warm season, normal maximum at or above
WINTER_TX = 10.0          # °C: cold season, normal maximum at or below
ACTIVITIES = ("Walking", "Cycling")
MIN_WIND_COVERAGE = 0.8   # below this share of days with a wind value, the wind is left out of "comfortable"


def daytime_temperature(tx, tn) -> np.ndarray:
    return 0.75 * np.asarray(tx, dtype=float) + 0.25 * np.asarray(tn, dtype=float)


def comfortable(daily: pd.DataFrame, activity: str, rain_day: float, use_wind: bool = True) -> pd.Series:
    """True for a comfortable day, False for another, NaN when a value it needs is missing."""
    low, high = PROFILES[activity].comfort_range
    t = daytime_temperature(daily["tx"], daily["tn"])
    rr = daily["rr"].to_numpy(dtype=float)
    ok = (t >= low) & (t <= high) & (rr < rain_day)
    known = ~np.isnan(t) & ~np.isnan(rr)
    if use_wind:
        wind = daily["ffm"].to_numpy(dtype=float) * 3.6   # m/s -> km/h
        ok &= wind < WIND_COMFORT_KMH
        known &= ~np.isnan(wind)
    return pd.Series(np.where(known, ok.astype(float), np.nan), index=daily.index)


def week_of_year(dates) -> pd.Series:
    """Week 1 to 52 of the year (7-day blocks from 1 January; the last day or two go in week 52)."""
    doy = day_of_year(dates)
    return ((doy - 1) // 7 + 1).clip(upper=52)


def week_label(week: int) -> str:
    """'12 May': the first day of the week, on a 365-day year."""
    return _day_label(7 * (int(week) - 1) + 1)


def _day_label(doy: int) -> str:
    """'12 May' for a day 1-365 of a year without 29 February."""
    d = pd.Timestamp("2001-01-01") + pd.Timedelta(days=int(doy) - 1)
    return f"{d.day} {d.strftime('%b')}"


def wind_coverage(daily: pd.DataFrame, period: tuple[int, int]) -> float:
    d = daily[pd.to_datetime(daily["date"]).dt.year.between(*period)]
    return float(d["ffm"].notna().mean()) if len(d) else 0.0


def best_weeks(daily: pd.DataFrame, activity: str, periods: dict[str, tuple[int, int]],
               rain_day: float) -> tuple[pd.DataFrame, bool]:
    """Share of comfortable days in each week of the year, for each period. Returns (week, period, share, n_days)
    and whether the wind was used (only if every period has wind values on most days)."""
    use_wind = all(wind_coverage(daily, p) >= MIN_WIND_COVERAGE for p in periods.values())
    ok = comfortable(daily, activity, rain_day, use_wind)
    d = pd.DataFrame({"date": pd.to_datetime(daily["date"]).to_numpy(), "ok": ok.to_numpy()})
    d["week"] = week_of_year(d["date"]).to_numpy()
    d["year"] = d["date"].dt.year
    rows = []
    for label, (y0, y1) in periods.items():
        g = d[d["year"].between(y0, y1)].dropna(subset=["ok", "week"]).groupby("week")["ok"].agg(["mean", "count"])
        rows += [{"week": int(w), "period": label, "share": float(r["mean"]), "n_days": int(r["count"])}
                 for w, r in g.iterrows()]
    return pd.DataFrame(rows, columns=["week", "period", "share", "n_days"]), use_wind


@dataclass
class Season:
    start: int      # first day (1-365)
    end: int        # last day (1-365; smaller than start when the season spans the new year)
    days: int

    @property
    def label(self) -> str:
        return f"{_day_label(self.start)} to {_day_label(self.end)}"


def _longest_run(mask: np.ndarray) -> tuple[int, int] | None:
    """(start, length) of the longest run of True in a circular array of 365 days."""
    if not mask.any():
        return None
    if mask.all():
        return 0, len(mask)
    shift = int(np.argmin(mask))                 # start the scan on a False day: no run is cut in two
    rolled = np.roll(mask, -shift)
    best, start = None, None
    for i, m in enumerate(np.append(rolled, False)):
        if m and start is None:
            start = i
        elif not m and start is not None:
            if best is None or i - start > best[1]:
                best = (start, i - start)
            start = None
    return (best[0] + shift) % len(mask), best[1]


def season(daily: pd.DataFrame, period: tuple[int, int], warm: bool = True) -> Season | None:
    """The warm (or cold) season of a period, from the normal maximum smoothed over ±15 days; None if the normal
    never crosses the threshold."""
    clim = doy_climatology(daily, "tx", period)["mean"].to_numpy()
    if np.isnan(clim).all():
        return None
    mask = clim >= SUMMER_TX if warm else clim <= WINTER_TX
    run = _longest_run(mask)
    if run is None:
        return None
    start, length = run
    if length == 365:
        return Season(1, 365, 365)
    return Season(start + 1, (start + length - 1) % 365 + 1, length)


def shift_days(then: Season | None, now: Season | None) -> tuple[int, int] | None:
    """(days earlier the season starts now, days later it ends) between two periods, None if one is missing."""
    if then is None or now is None or then.days == 365 or now.days == 365:
        return None
    def gap(a: int, b: int) -> int:   # a - b on the circle, in -182..182
        return int((a - b + 182) % 365 - 182)
    return gap(then.start, now.start), gap(now.end, then.end)
