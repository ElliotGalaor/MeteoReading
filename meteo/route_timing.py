"""Travel time along a track, stretch by stretch, from the slope (and, on a bike, the wind).

Walking: Tobler's hiking function, v = 6 exp(-3.5 |s + 0.05|) km/h for a slope s, scaled so that the flat speed is the
one chosen: a gentle descent (-5%) is the fastest, steep slopes slow down both up and down.

Cycling: a constant effort. The power that holds the chosen flat speed without wind is kept everywhere, and the speed
of each stretch is the one where that power balances gravity, rolling resistance and air drag, with the forecast
head or tail wind:
    P = (m g (sin a + Crr cos a) + 1/2 rho CdA (v + w)|v + w|) v
for a rider and bike of 85 kg, Crr = 0.005, CdA = 0.40 m², rho = 1.2 kg/m³ (a road bike, hands on the hoods).
Speed is capped at 55 km/h downhill (bends, braking) and kept at 4 km/h or more on steep climbs (pushing the bike).

Driving: the flat speed (roads, limits and traffic matter more than slopes).

The old estimate (flat speed plus minutes per 100 m of ascent, `gps.moving_hours`) stays available for comparison.
"""
from __future__ import annotations

import numpy as np

from .gps import Track, _smooth

MASS_KG = 85.0
G = 9.81
CRR = 0.005
CDA_M2 = 0.40
RHO = 1.2
MAX_DESCENT_KMH = 55.0
MIN_BIKE_KMH = 4.0
MIN_WALK_KMH = 0.5
SLOPE_HALF_WINDOW_KM = 0.1   # slope of each point over ±100 m: GPS and terrain altitudes are noisy over a few metres
MAX_SLOPE = 0.35


def slopes(track: Track) -> np.ndarray:
    """Slope (rise / run) at each point of the track, over ±100 m of a lightly smoothed profile; 0 without
    altitudes."""
    p = track.points
    d = p["dist_km"].to_numpy(dtype=float)
    if not track.has_elevation or d[-1] <= 0:
        return np.zeros(len(p))
    e = _smooth(p["ele"].interpolate(limit_direction="both")).to_numpy(dtype=float)
    up = np.interp(np.minimum(d + SLOPE_HALF_WINDOW_KM, d[-1]), d, e)
    down = np.interp(np.maximum(d - SLOPE_HALF_WINDOW_KM, 0), d, e)
    run = (np.minimum(d + SLOPE_HALF_WINDOW_KM, d[-1]) - np.maximum(d - SLOPE_HALF_WINDOW_KM, 0)) * 1000
    with np.errstate(invalid="ignore", divide="ignore"):
        s = np.where(run > 0, (up - down) / run, 0.0)
    return np.clip(np.nan_to_num(s), -MAX_SLOPE, MAX_SLOPE)


def tobler_kmh(slope, flat_kmh: float) -> np.ndarray:
    """Walking speed on a slope, Tobler's function scaled to `flat_kmh` on the flat."""
    s = np.asarray(slope, dtype=float)
    return np.maximum(flat_kmh * np.exp(-3.5 * (np.abs(s + 0.05) - 0.05)), MIN_WALK_KMH)


def flat_power(flat_kmh: float) -> float:
    """Watts that hold `flat_kmh` on the flat without wind."""
    v = flat_kmh / 3.6
    return (MASS_KG * G * CRR + 0.5 * RHO * CDA_M2 * v * v) * v


def bike_kmh(slope, flat_kmh: float, headwind_kmh=0.0) -> np.ndarray:
    """Cycling speed at constant power on each slope, against `headwind_kmh` (negative: tailwind)."""
    s = np.asarray(slope, dtype=float)
    w = np.broadcast_to(np.asarray(headwind_kmh, dtype=float) / 3.6, s.shape)
    w = np.nan_to_num(w)
    power = flat_power(flat_kmh)
    angle = np.arctan(s)
    a = MASS_KG * G * (np.sin(angle) + CRR * np.cos(angle))

    def excess(v):
        rel = v + w
        return (a + 0.5 * RHO * CDA_M2 * rel * np.abs(rel)) * v - power

    lo, hi = np.full(s.shape, 0.05), np.full(s.shape, 40.0)   # m/s; excess(lo) < 0 < excess(hi) on any road
    for _ in range(60):   # bisection: the power curve crosses zero once for v > 0
        mid = (lo + hi) / 2
        neg = excess(mid) < 0
        lo, hi = np.where(neg, mid, lo), np.where(neg, hi, mid)
    return np.clip((lo + hi) / 2 * 3.6, MIN_BIKE_KMH, MAX_DESCENT_KMH)


def speeds(track: Track, activity: str, flat_kmh: float, headwind_kmh=None) -> np.ndarray:
    """Speed (km/h) on each leg between consecutive points. `headwind_kmh`: wind against the rider at each point
    (cycling only)."""
    if flat_kmh <= 0:
        raise ValueError("Speed must be positive")
    s = slopes(track)
    leg = (s[:-1] + s[1:]) / 2
    if activity == "Walking":
        return tobler_kmh(leg, flat_kmh)
    if activity == "Cycling":
        head = np.zeros(len(s)) if headwind_kmh is None else np.nan_to_num(np.asarray(headwind_kmh, dtype=float))
        return bike_kmh(leg, flat_kmh, (head[:-1] + head[1:]) / 2)
    return np.full(len(leg), float(flat_kmh))


def moving_hours(track: Track, activity: str, flat_kmh: float, headwind_kmh=None) -> np.ndarray:
    """Hours on the move to reach each point."""
    legs_km = np.diff(track.points["dist_km"].to_numpy(dtype=float))
    v = speeds(track, activity, flat_kmh, headwind_kmh)
    return np.concatenate([[0.0], np.cumsum(legs_km / v)])
