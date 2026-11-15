"""Weather reading along a GPS track.

For each point sampled along the route, fetch the hourly weather from Open-Meteo and
interpolate it at the time the traveller is there:
- trip in the next 15 days or the last ~2 months -> forecast API (best models for the
  area, e.g. Météo-France AROME/ARPEGE over France);
- older trip -> ERA5 reanalysis archive (what the weather actually was).
If the forecast API returns empty values for the trip's hours, the archive is used instead when it covers them.

The wind is projected on the direction of travel to get head/tail/cross wind.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd

from .download import make_session
from .gps import Track, sample_points, segment_km

FORECAST_API = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_API = "https://archive-api.open-meteo.com/v1/archive"
# The forecast API accepts dates up to 92 days back, but only returns values for about the last 65 days
# (nulls before): stay well inside, and fall back to the archive if values are missing anyway.
FORECAST_PAST_DAYS = 60
FORECAST_FUTURE_DAYS = 15    # and forecasts 16 days ahead (today included)
ARCHIVE_DELAY_DAYS = 6       # ERA5 is published with a ~5-day delay
HOURLY = ["temperature_2m", "apparent_temperature", "precipitation", "precipitation_probability",
          "weather_code", "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m", "cloud_cover",
          # for the danger and comfort scores (route_scores.py); the archive has no probability, visibility or UV
          "is_day", "dew_point_2m", "snowfall", "visibility", "uv_index"]
RAIN_THRESHOLD = 0.2         # mm/h: below this, the hour counts as dry
ICE_RAIN_MM = 0.1            # mm/h of rain or drizzle at 1 °C or below: risk of ice (also used by route_scores)
MAX_GAP_S = 3 * 3600         # interpolate across at most 3 hours without a value
CORE = ["temperature_2m", "precipitation", "wind_speed_10m", "weather_code"]   # needed for every reading

# WMO weather codes -> (label, icon)
WEATHER_CODES = {
    0: ("Clear sky", "☀️"), 1: ("Mainly clear", "🌤️"), 2: ("Partly cloudy", "⛅"), 3: ("Overcast", "☁️"),
    45: ("Fog", "🌫️"), 48: ("Freezing fog", "🌫️"),
    51: ("Light drizzle", "🌦️"), 53: ("Drizzle", "🌦️"), 55: ("Heavy drizzle", "🌧️"),
    56: ("Freezing drizzle", "🌧️"), 57: ("Heavy freezing drizzle", "🌧️"),
    61: ("Light rain", "🌦️"), 63: ("Rain", "🌧️"), 65: ("Heavy rain", "🌧️"),
    66: ("Freezing rain", "🌧️"), 67: ("Heavy freezing rain", "🌧️"),
    71: ("Light snow", "🌨️"), 73: ("Snow", "❄️"), 75: ("Heavy snow", "❄️"), 77: ("Snow grains", "🌨️"),
    80: ("Light showers", "🌦️"), 81: ("Showers", "🌧️"), 82: ("Violent showers", "⛈️"),
    85: ("Snow showers", "🌨️"), 86: ("Heavy snow showers", "❄️"),
    95: ("Thunderstorm", "⛈️"), 96: ("Thunderstorm with hail", "⛈️"), 99: ("Severe thunderstorm with hail", "⛈️"),
}
COMPASS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]


@dataclass
class RouteReading:
    track: Track
    points: pd.DataFrame            # one row per sampled point, weather at passage time
    source: str                     # "forecast" or "archive"
    timezone: str
    insights: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)


# --- Source selection -----------------------------------------------------------------
def choose_source(start: date, end: date, today: date | None = None) -> str:
    today = today or date.today()
    if start >= today - timedelta(days=FORECAST_PAST_DAYS) and end <= today + timedelta(days=FORECAST_FUTURE_DAYS):
        return "forecast"
    if end <= today - timedelta(days=ARCHIVE_DELAY_DAYS):
        return "archive"
    if end > today + timedelta(days=FORECAST_FUTURE_DAYS):
        raise ValueError(f"Forecasts only cover the next {FORECAST_FUTURE_DAYS + 1} days: pick an earlier date.")
    raise ValueError("This trip straddles the forecast and archive periods: pick a single period.")


def api_weight(n_points: int, n_days: int) -> float:
    """Open-Meteo counts (variables / 10) x (days / 14) calls per location."""
    return n_points * max(len(HOURLY) / 10, 1) * max(n_days / 14, 1)


# --- Fetching ----------------------------------------------------------------------------
def fetch_hourly(points: pd.DataFrame, start: date, end: date, source: str, session=None,
                 batch: int = 50) -> tuple[list[pd.DataFrame], list[float], str]:
    """Hourly weather for each point. Returns (one DataFrame per point, model elevations, timezone)."""
    session = session or make_session()
    api = FORECAST_API if source == "forecast" else ARCHIVE_API
    frames, elevations, tz = [], [], "UTC"
    for i in range(0, len(points), batch):
        chunk = points.iloc[i:i + batch]
        params = {
            "latitude": ",".join(f"{v:.5f}" for v in chunk["lat"]),
            "longitude": ",".join(f"{v:.5f}" for v in chunk["lon"]),
            "hourly": ",".join(HOURLY),
            "start_date": start.isoformat(), "end_date": end.isoformat(),
            "timezone": "auto", "timeformat": "unixtime", "wind_speed_unit": "kmh",
        }
        try:
            resp = session.get(api, params=params, timeout=60)
        except OSError as exc:   # requests' errors (connection, timeout, retries exhausted) are OSErrors
            raise RuntimeError("The weather service is not answering: try again in a moment.") from exc
        if resp.status_code != 200:
            try:
                reason = resp.json().get("reason", resp.text[:200])
            except ValueError:
                reason = resp.text[:200]
            raise RuntimeError(f"Open-Meteo error ({resp.status_code}): {reason}")
        payload = resp.json()
        for loc in payload if isinstance(payload, list) else [payload]:
            h = loc["hourly"]
            df = pd.DataFrame({k: h.get(k) for k in HOURLY if k in h})
            df.insert(0, "time", pd.to_datetime(h["time"], unit="s", utc=True))
            frames.append(df)
            elevations.append(loc.get("elevation"))
            tz = loc.get("timezone", tz)
    return frames, elevations, tz


# --- Interpolation at passage time ---------------------------------------------------------
def epoch_seconds(times) -> np.ndarray:
    """Seconds since 1970 whatever the datetime resolution (pandas 3 may store s, ms, us or ns)."""
    t = pd.to_datetime(pd.Series(times), utc=True)
    return ((t - pd.Timestamp("1970-01-01", tz="UTC")) / pd.Timedelta(seconds=1)).to_numpy(dtype=float)


def interp_valid(tx: float, x: np.ndarray, y: np.ndarray, max_gap: float = MAX_GAP_S) -> float:
    """Linear interpolation at `tx` between the valid (non-NaN) hours around it.

    NaN when the valid hours around `tx` are more than `max_gap` seconds apart, or when `tx` is more than an hour
    outside them: near the forecast horizon, a variable stops before the others (e.g. the rain probability), and
    repeating its last value would show a forecast that does not exist."""
    ok = ~np.isnan(y)
    x, y = x[ok], y[ok]
    if not len(x):
        return np.nan
    i = int(np.searchsorted(x, tx))
    if i < len(x) and x[i] == tx:
        return float(y[i])
    if i == 0 or i == len(x):   # before the first or after the last valid hour
        j = 0 if i == 0 else len(x) - 1
        return float(y[j]) if abs(x[j] - tx) <= 3600 else np.nan
    if x[i] - x[i - 1] > max_gap:
        return np.nan
    return float(np.interp(tx, x[i - 1:i + 1], y[i - 1:i + 1]))


def interpolate_at(hourly: pd.DataFrame, t: pd.Timestamp) -> dict:
    """Weather at an exact time from hourly values (linear in time, wind as a vector)."""
    x = epoch_seconds(hourly["time"])
    tx = float(epoch_seconds([t])[0])

    def values(col: str) -> np.ndarray:
        return pd.to_numeric(hourly[col], errors="coerce").to_numpy(dtype=float)

    out = {}
    for col in ("temperature_2m", "apparent_temperature", "precipitation", "precipitation_probability",
                "wind_speed_10m", "wind_gusts_10m", "cloud_cover", "is_day", "dew_point_2m", "snowfall",
                "visibility", "uv_index"):
        out[col] = interp_valid(tx, x, values(col)) if col in hourly else np.nan
    if "wind_direction_10m" in hourly and "wind_speed_10m" in hourly:
        ws, wd = values("wind_speed_10m"), np.radians(values("wind_direction_10m"))
        u, v = interp_valid(tx, x, ws * np.sin(wd)), interp_valid(tx, x, ws * np.cos(wd))
        out["wind_direction_10m"] = float((np.degrees(np.arctan2(u, v)) + 360) % 360) if pd.notna(u + v) else np.nan
    out["weather_code"] = None
    if "weather_code" in hourly:
        codes = values("weather_code")
        ok = np.flatnonzero(~np.isnan(codes))
        if len(ok):
            nearest = ok[np.argmin(np.abs(x[ok] - tx))]
            if abs(x[nearest] - tx) <= 3600:
                out["weather_code"] = int(codes[nearest])
    return out


def missing_hours(samples: pd.DataFrame, hourly: list[pd.DataFrame]) -> int:
    """Number of points whose passage time is not framed by complete hourly values (the API returned nulls)."""
    missing = 0
    for h, t in zip(hourly, samples["time"], strict=True):
        dt = np.abs(epoch_seconds(h["time"]) - float(epoch_seconds([t])[0]))
        near = h.loc[dt <= 3600, [c for c in CORE if c in h]]
        if near.empty or len(near.columns) < len(CORE) or near.isna().any().any():
            missing += 1
    return missing


def wind_components(speed, wind_from_deg, bearing_deg):
    """Headwind (> 0) / tailwind (< 0) and crosswind, for a traveller heading `bearing_deg`."""
    angle = np.radians(np.asarray(wind_from_deg) - np.asarray(bearing_deg))
    return np.asarray(speed) * np.cos(angle), np.abs(np.asarray(speed) * np.sin(angle))


def compass(deg: float) -> str:
    return COMPASS[int(((deg % 360) + 22.5) // 45) % 8]


def build_reading_table(samples: pd.DataFrame, hourly: list[pd.DataFrame], elevations: list) -> pd.DataFrame:
    rows = [interpolate_at(h, t) for h, t in zip(hourly, samples["time"], strict=True)]
    df = pd.concat([samples.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    df["model_ele"] = elevations
    df["ele_profile"] = df["ele"].where(df["ele"].notna(), df["model_ele"])
    df["headwind"], df["crosswind"] = wind_components(df["wind_speed_10m"], df["wind_direction_10m"], df["bearing"])
    df["condition"] = [WEATHER_CODES.get(c, ("Unknown", "❔"))[0] if c is not None else "Unknown" for c in df["weather_code"]]
    df["icon"] = [WEATHER_CODES.get(c, ("Unknown", "❔"))[1] if c is not None else "❔" for c in df["weather_code"]]
    df["wind_from"] = [compass(d) if pd.notna(d) else "" for d in df["wind_direction_10m"]]
    # Time spent around each point (half of each neighbouring leg), to turn mm/h into mm encountered
    t = epoch_seconds(df["time"]) / 3600
    edges = np.concatenate([[t[0]], (t[:-1] + t[1:]) / 2, [t[-1]]])
    df["hours_here"] = np.diff(edges)
    return df


# --- Summary and insights ------------------------------------------------------------------
def clock_format(local: pd.Series) -> str:
    """strftime format for the times of a trip: the day on multi-day trips, the zone when the clocks change
    during the trip (otherwise 02:30 CEST would be followed by 02:00 CET)."""
    fmt = "%a %H:%M" if local.dt.date.nunique() > 1 else "%H:%M"
    return fmt + " %Z" if len({t.utcoffset() for t in local}) > 1 else fmt


def whole(x) -> str:
    """A whole number for display, without the "-0" that rounding -0.4 gives."""
    return f"{int(round(float(x)))}"


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Index ranges [start, end] of consecutive True values."""
    runs, start = [], None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        if not m and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask) - 1))
    return runs


def summarize(df: pd.DataFrame, tz: str) -> tuple[dict, list[str]]:
    local = df["time"].dt.tz_convert(tz)
    clock = clock_format(local)
    rain = df["precipitation"].fillna(0)
    rainy = (rain >= RAIN_THRESHOLD).to_numpy()
    seg_km = segment_km(df["dist_km"])
    total_km = float(df["dist_km"].iloc[-1]) or 1.0
    head = df["headwind"].to_numpy()
    stats = {
        "temp_min": float(df["temperature_2m"].min()), "temp_max": float(df["temperature_2m"].max()),
        "feels_min": float(df["apparent_temperature"].min()),
        "rain_mm": float((rain * df["hours_here"]).sum()),
        "rain_share": float(seg_km[rainy].sum() / total_km),
        "max_rain_prob": float(df["precipitation_probability"].max()) if df["precipitation_probability"].notna().any() else None,
        "gust_max": float(df["wind_gusts_10m"].max()),
        "headwind_share": float(seg_km[head > 5].sum() / total_km),
        "tailwind_share": float(seg_km[head < -5].sum() / total_km),
        "mean_headwind": float(np.average(head, weights=seg_km)) if seg_km.sum() else 0.0,
    }

    insights = []
    runs = _runs(rainy)
    if not runs:
        prob = stats["max_rain_prob"]
        insights.append("Dry all the way" + (f" (rain probability at most {prob:.0f}%)." if prob is not None else "."))
    else:
        for a, b in runs[:3]:
            km_a, km_b = df["dist_km"].iloc[a], df["dist_km"].iloc[b]
            span = f"km {km_a:.0f}" if a == b else f"km {km_a:.0f} to {km_b:.0f}"
            insights.append(f"Rain likely from {span}, around {local.iloc[a].strftime(clock)} "
                            f"(up to {rain.iloc[a:b + 1].max():.1f} mm/h).")
        amount = "only a trace of rain (under 0.1 mm)" if stats["rain_mm"] < 0.1 else f"about {stats['rain_mm']:.1f} mm of rain"
        insights.append(f"In total, {amount} on {stats['rain_share']:.0%} of the route.")

    i_min = int(df["temperature_2m"].idxmin())
    t_line = f"Temperature from {whole(stats['temp_min'])} °C to {whole(stats['temp_max'])} °C"
    if stats["temp_max"] - stats["temp_min"] >= 3:
        t_line += f" (coldest at km {df['dist_km'].iloc[i_min]:.0f}"
        ele = df["ele_profile"].iloc[i_min]
        t_line += f", {ele:.0f} m)." if pd.notna(ele) else ")."
    else:
        t_line += "."
    if stats["feels_min"] < stats["temp_min"] - 2:
        t_line += f" Feels like {whole(stats['feels_min'])} °C with the wind."
    insights.append(t_line)

    if stats["headwind_share"] >= 0.25 or stats["tailwind_share"] >= 0.25:
        parts = []
        if stats["headwind_share"] >= 0.1:
            parts.append(f"headwind on {stats['headwind_share']:.0%} of the route")
        if stats["tailwind_share"] >= 0.1:
            parts.append(f"tailwind on {stats['tailwind_share']:.0%}")
        insights.append("Wind: " + ", ".join(parts) + f" (mean along the route: {stats['mean_headwind']:+.0f} km/h, "
                        "positive = headwind).")
    else:
        insights.append("Little wind along the direction of travel.")
    if stats["gust_max"] >= 50:
        i_g = int(df["wind_gusts_10m"].idxmax())
        insights.append(f"Strong gusts up to {stats['gust_max']:.0f} km/h near km {df['dist_km'].iloc[i_g]:.0f}.")
    codes = set(df["weather_code"].dropna().astype(int))
    if codes & {95, 96, 99}:
        insights.append("⚠️ Thunderstorms possible on the route.")
    if codes & {56, 57, 66, 67, 71, 73, 75, 77, 85, 86} or ((df["temperature_2m"] <= 1) & (rain >= ICE_RAIN_MM)).any():
        insights.append("⚠️ Risk of snow or ice.")
    return stats, insights


def track_timezone(lat: float, lon: float, session=None) -> str | None:
    """IANA time zone at a point (Open-Meteo resolves it from the coordinates), or None if it cannot be had."""
    session = session or make_session()
    try:
        resp = session.get(FORECAST_API, timeout=15, params={
            "latitude": f"{lat:.3f}", "longitude": f"{lon:.3f}", "timezone": "auto", "daily": "sunrise",
            "forecast_days": 1})
        return resp.json().get("timezone") if resp.status_code == 200 else None
    except (OSError, ValueError):
        return None


# --- Entry point ----------------------------------------------------------------------------------
def read_route(track: Track, max_points: int = 40, session=None, today: date | None = None) -> RouteReading:
    """Weather reading along a track whose points all have a time (real or estimated)."""
    if not track.has_time:
        raise ValueError("The track has no times: provide a start time and a speed first")
    samples = sample_points(track, max_points=max_points)
    start, end = samples["time"].min().date(), samples["time"].max().date()
    # one day of margin on each side for time zones and interpolation at day boundaries
    today = today or date.today()
    source = choose_source(start, end, today)
    q_start, q_end = start - timedelta(days=1), end + timedelta(days=1)
    if source == "forecast":
        f_start = max(q_start, today - timedelta(days=FORECAST_PAST_DAYS))
        f_end = min(q_end, today + timedelta(days=FORECAST_FUTURE_DAYS))
        hourly, elevations, tz = fetch_hourly(samples, f_start, f_end, source, session=session)
        if missing_hours(samples, hourly) and end <= today - timedelta(days=ARCHIVE_DELAY_DAYS):
            source = "archive"   # the forecast API has no values for these days, the archive does
    if source == "archive":
        hourly, elevations, tz = fetch_hourly(samples, q_start, q_end, source, session=session)
    if missing_hours(samples, hourly):
        raise ValueError("The weather service has no data for the times of this trip. Weather is available for the "
                         f"next {FORECAST_FUTURE_DAYS + 1} days and for the past; the most recent days may not be "
                         "complete yet.")
    df = build_reading_table(samples, hourly, elevations)
    stats, insights = summarize(df, tz)
    return RouteReading(track=track, points=df, source=source, timezone=tz, insights=insights, stats=stats)


def nearby_station_obs(con, points: pd.DataFrame, tz: str, radius_km: float = 5.0) -> pd.DataFrame:
    """Daily observations of the Météo-France stations close to the route, on the day of passage.

    `con` is a connection to the project's DuckDB warehouse (only the downloaded départements are covered).
    """
    p = points[["dist_km", "lat", "lon", "time"]].copy()
    p["day"] = p["time"].dt.tz_convert(tz).dt.date
    con.register("route_points", p[["dist_km", "lat", "lon", "day"]])
    try:
        return con.sql(f"""
            WITH near AS (
                SELECT s.station_id, s.station_name, s.altitude,
                       arg_min(p.dist_km, haversine_km(p.lat, p.lon, s.lat, s.lon)) AS route_km,
                       arg_min(p.day, haversine_km(p.lat, p.lon, s.lat, s.lon))     AS day,
                       min(haversine_km(p.lat, p.lon, s.lat, s.lon))                AS distance_km
                FROM dim_station s, route_points p
                WHERE haversine_km(p.lat, p.lon, s.lat, s.lon) <= {radius_km}
                GROUP BY ALL
            )
            SELECT n.route_km, n.station_name, n.altitude, round(n.distance_km, 1) AS distance_km,
                   n.day, o.tn, o.tx, o.rr, o.fxi
            FROM near n
            JOIN obs_daily o ON o.station_id = n.station_id AND o.date = n.day
            WHERE o.tn IS NOT NULL OR o.tx IS NOT NULL OR o.rr IS NOT NULL
            ORDER BY n.route_km
        """).df()
    finally:
        con.unregister("route_points")
