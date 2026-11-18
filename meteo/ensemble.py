"""How sure the route forecast is: the 51 runs of the ECMWF ensemble along the route.

An ensemble forecast runs the same model many times from slightly different starts: where the runs agree the
forecast is reliable, where they spread it is not. The Open-Meteo Ensemble API gives the 51 runs of ECMWF IFS
(0.25°, up to 15 days). Each run is read at the time the traveller passes, at ten points of the route (the spread
varies slowly along it), then:
- temperature and wind: the 10th-90th percentile range of the runs (80% of them inside);
- rain: the share of runs with at least 0.2 mm/h, at each point and anywhere on the route;
- hazards: the share of runs with gusts or heavy rain above the activity's danger thresholds somewhere on the way;
- confidence: high when temperatures stay within 2 °C and the runs agree on rain (under 20% or over 80% of them),
  low when they spread over 5 °C or split on rain (35% to 65%), medium otherwise.
Ensemble requests weigh more on the free quota (each run counts as variables), so the page asks before fetching.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from .download import make_session
from .route_weather import RAIN_THRESHOLD, epoch_seconds, get_locations, subset

ENSEMBLE_API = "https://ensemble-api.open-meteo.com/v1/ensemble"
MODEL = "ecmwf_ifs025"
MEMBERS = 51
VARIABLES = ["temperature_2m", "precipitation", "wind_speed_10m", "wind_gusts_10m"]
POINTS = 10
HEAVY_RAIN = 4.0     # mm/h: where the heavy-rain hazard starts (route_scores)


def weight(n_points: int, n_days: int) -> float:
    """Open-Meteo calls counted: (variables x runs / 10) x (days / 14) per point."""
    return min(n_points, POINTS) * len(VARIABLES) * MEMBERS / 10 * max(n_days / 14, 1)


def fetch(points: pd.DataFrame, start: date, end: date, session=None) -> tuple[list[pd.DataFrame], np.ndarray]:
    """Hourly values of every run at ten of the points. Returns (frames, rows of the points)."""
    session = session or make_session()
    idx = subset(len(points), POINTS)
    chunk = points.iloc[idx]
    params = {"latitude": ",".join(f"{v:.5f}" for v in chunk["lat"]),
              "longitude": ",".join(f"{v:.5f}" for v in chunk["lon"]),
              "hourly": ",".join(VARIABLES), "models": MODEL, "start_date": start.isoformat(),
              "end_date": end.isoformat(), "timezone": "auto", "timeformat": "unixtime", "wind_speed_unit": "kmh"}
    frames = []
    for loc in get_locations(session, ENSEMBLE_API, params):
        h = loc["hourly"]
        df = pd.DataFrame({k: pd.to_numeric(pd.Series(v), errors="coerce") for k, v in h.items() if k != "time"})
        df.insert(0, "time", pd.to_datetime(h["time"], unit="s", utc=True))
        frames.append(df)
    return frames, idx


def members_at(frame: pd.DataFrame, variable: str, t: pd.Timestamp) -> np.ndarray:
    """Every run's value of `variable` at time `t` (linear between hours; NaN outside the hours given)."""
    cols = [c for c in frame.columns if c == variable or c.startswith(f"{variable}_member")]
    x = epoch_seconds(frame["time"])
    tx = float(epoch_seconds([t])[0])
    if not cols or tx < x[0] or tx > x[-1]:
        return np.full(max(len(cols), 1), np.nan)
    j = int(np.clip(np.searchsorted(x, tx) - 1, 0, len(x) - 2))
    a = (tx - x[j]) / (x[j + 1] - x[j]) if x[j + 1] > x[j] else 0.0
    v = frame[cols].to_numpy(dtype=float)
    return v[j] * (1 - a) + v[j + 1] * a


@dataclass
class Spread:
    points: pd.DataFrame       # per point of the reading: t_p10, t_p90, wind_p10, wind_p90, gust_p90, rain_chance
    rain_anywhere: float       # share of runs with rain somewhere on the route
    hazard_shares: dict        # "gust" / "heavy" -> share of runs where it happens somewhere on the route
    temp_spread: float         # mean width of the 10-90% temperature range along the route (°C)
    confidence: str            # "High", "Medium", "Low" or "Unknown" (a code: the page names it)
    n_members: int


def spread(frames: list[pd.DataFrame], idx: np.ndarray, reading: pd.DataFrame, gust_danger: float) -> Spread:
    """The runs at the passage times of the reading (its points: dist_km, time). `gust_danger`: gusts (km/h) where
    the activity's danger starts."""
    rows, hits, n = [], {}, 0
    for frame, i in zip(frames, idx, strict=True):
        t = reading["time"].iloc[i]
        temp, rain = members_at(frame, "temperature_2m", t), members_at(frame, "precipitation", t)
        wind, gust = members_at(frame, "wind_speed_10m", t), members_at(frame, "wind_gusts_10m", t)
        n = max(n, int(np.isfinite(temp).sum()))
        rows.append({"dist_km": reading["dist_km"].iloc[i],
                     "t_p10": np.nanpercentile(temp, 10) if np.isfinite(temp).any() else np.nan,
                     "t_p90": np.nanpercentile(temp, 90) if np.isfinite(temp).any() else np.nan,
                     "wind_p10": np.nanpercentile(wind, 10) if np.isfinite(wind).any() else np.nan,
                     "wind_p90": np.nanpercentile(wind, 90) if np.isfinite(wind).any() else np.nan,
                     "gust_p90": np.nanpercentile(gust, 90) if np.isfinite(gust).any() else np.nan,
                     "rain_chance": float(np.mean(rain >= RAIN_THRESHOLD)) if np.isfinite(rain).any() else np.nan})
        if np.isfinite(rain).any():   # which runs meet each condition somewhere on the route
            for k, v in (("rain", rain >= RAIN_THRESHOLD), ("gust", np.nan_to_num(gust) >= gust_danger),
                         ("heavy", rain >= HEAVY_RAIN)):
                hits[k] = v if k not in hits or len(v) != len(hits[k]) else hits[k] | v
    at = pd.DataFrame(rows)
    dist = reading["dist_km"].to_numpy(dtype=float)
    out = pd.DataFrame({"dist_km": dist})
    for c in ("t_p10", "t_p90", "wind_p10", "wind_p90", "gust_p90", "rain_chance"):
        ok = at[c].notna().to_numpy()
        out[c] = np.interp(dist, at["dist_km"][ok], at[c][ok]) if ok.sum() >= 2 else np.nan
    width = float(np.nanmean(at["t_p90"] - at["t_p10"])) if len(at) else np.nan
    anywhere = float(hits["rain"].mean()) if "rain" in hits else np.nan
    shares = {"gust": float(hits["gust"].mean()) if "gust" in hits else np.nan,      # gusts ≥ gust_danger
              "heavy": float(hits["heavy"].mean()) if "heavy" in hits else np.nan}    # rain ≥ HEAVY_RAIN mm/h
    if not np.isfinite(width):
        confidence = "Unknown"
    elif width >= 5 or 0.35 <= anywhere <= 0.65:
        confidence = "Low"
    elif width <= 2 and (anywhere <= 0.2 or anywhere >= 0.8):
        confidence = "High"
    else:
        confidence = "Medium"
    return Spread(out, anywhere, shares, width, confidence, n)
