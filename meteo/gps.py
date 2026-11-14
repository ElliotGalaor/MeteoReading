"""GPS tracks: parsing (GPX, KML, GeoJSON), distances, sampling and passage times.

Pure functions with no network access, covered by unit tests. Uploaded XML files are
parsed with defusedxml, which rejects malicious documents (entity expansion attacks).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd
from defusedxml import ElementTree as ET

MAX_POINTS = 200_000
EARTH_RADIUS_KM = 6371.0

# Rough travel-time model: flat speed (km/h) + extra minutes per 100 m of ascent
# (Naismith's rule for walking: 1 hour per 600 m of ascent = 10 min per 100 m).
ACTIVITIES = {
    "Walking": {"speed": 4.5, "climb_min_per_100m": 10.0},
    "Cycling": {"speed": 20.0, "climb_min_per_100m": 6.0},
    "Driving": {"speed": 80.0, "climb_min_per_100m": 0.0},
}


@dataclass
class Track:
    name: str
    points: pd.DataFrame  # lat, lon, ele (NaN if unknown), time (UTC, NaT if unknown), dist_km

    @property
    def has_time(self) -> bool:
        return bool(self.points["time"].notna().all())

    @property
    def has_elevation(self) -> bool:
        return bool(self.points["ele"].notna().mean() > 0.9)

    @property
    def distance_km(self) -> float:
        return float(self.points["dist_km"].iloc[-1])


# --- Parsing ------------------------------------------------------------------------
def parse_track(data: bytes, filename: str) -> Track:
    """Read a GPX, KML or GeoJSON file into a Track."""
    ext = Path(filename).suffix.lower()
    name = Path(filename).stem
    if ext == ".gpx":
        rows, found_name = _parse_gpx(data)
    elif ext == ".kml":
        rows, found_name = _parse_kml(data)
    elif ext in (".geojson", ".json"):
        rows, found_name = _parse_geojson(data)
    else:
        raise ValueError(f"Unsupported file type '{ext}': use .gpx, .kml or .geojson")
    return build_track(rows, found_name or name)


def build_track(rows: list[tuple], name: str = "Track") -> Track:
    """(lat, lon, ele, time) rows -> clean Track with cumulative distance."""
    if len(rows) > MAX_POINTS:
        raise ValueError(f"Track too large ({len(rows):,} points, maximum {MAX_POINTS:,})")
    df = pd.DataFrame(rows, columns=["lat", "lon", "ele", "time"])
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df["ele"] = pd.to_numeric(df["ele"], errors="coerce")
    df = df[df["lat"].between(-90, 90) & df["lon"].between(-180, 180)]
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce", format="ISO8601")
    # Drop repeated points (GPS devices log many while standing still)
    same = (df["lat"].diff() == 0) & (df["lon"].diff() == 0)
    df = df[~same].reset_index(drop=True)
    if len(df) < 2:
        raise ValueError("The track needs at least 2 valid points")
    # Times are only used if (almost) every point has one and they move forward
    t = df["time"]
    if t.notna().mean() < 0.9 or not t.dropna().is_monotonic_increasing:
        df["time"] = pd.NaT
    elif t.isna().any():
        df["time"] = t.interpolate(limit_direction="both")
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df["dist_km"] = cumulative_distance(df["lat"].to_numpy(), df["lon"].to_numpy())
    return Track(name=name, points=df)


def _local(tag: str) -> str:
    """'{namespace}trkpt' -> 'trkpt' (works with any GPX/KML version)."""
    return tag.rsplit("}", 1)[-1]


def _parse_gpx(data: bytes) -> tuple[list[tuple], str | None]:
    root = ET.fromstring(data)
    found: dict[str, list[tuple]] = {"trkpt": [], "rtept": [], "wpt": []}
    name = None
    for el in root.iter():
        tag = _local(el.tag)
        if tag in found:
            ele = time = None
            for child in el:
                ctag = _local(child.tag)
                if ctag == "ele":
                    ele = child.text
                elif ctag == "time":
                    time = child.text
            found[tag].append((el.get("lat"), el.get("lon"), ele, time))
        elif tag == "name" and name is None and el.text and el.text.strip():
            name = el.text.strip()
    rows = found["trkpt"] or found["rtept"] or found["wpt"]
    if not rows:
        raise ValueError("No track, route or waypoints found in the GPX file")
    return rows, name


def _parse_kml(data: bytes) -> tuple[list[tuple], str | None]:
    root = ET.fromstring(data)
    name = next((el.text.strip() for el in root.iter() if _local(el.tag) == "name" and el.text), None)
    # gx:Track (with timestamps) takes priority over plain LineString coordinates
    whens = [el.text for el in root.iter() if _local(el.tag) == "when"]
    coords = [el.text for el in root.iter() if _local(el.tag) == "coord"]
    if coords and len(whens) == len(coords):
        rows = []
        for when, c in zip(whens, coords):
            parts = c.split()
            rows.append((parts[1], parts[0], parts[2] if len(parts) > 2 else None, when))
        return rows, name
    rows = []
    for el in root.iter():
        if _local(el.tag) == "coordinates" and el.text:
            for chunk in el.text.split():
                parts = chunk.split(",")
                if len(parts) >= 2:
                    rows.append((parts[1], parts[0], parts[2] if len(parts) > 2 else None, None))
    if not rows:
        raise ValueError("No coordinates found in the KML file")
    return rows, name


def _parse_geojson(data: bytes) -> tuple[list[tuple], str | None]:
    obj = json.loads(data)
    features = obj.get("features") or ([obj] if obj.get("type") == "Feature" else [{"geometry": obj}])
    rows, name = [], None
    for feat in features:
        geom = feat.get("geometry") or {}
        props = feat.get("properties") or {}
        name = name or props.get("name")
        times = props.get("coordTimes") or props.get("times")
        lines = {"LineString": [geom.get("coordinates")],
                 "MultiLineString": geom.get("coordinates")}.get(geom.get("type"), [])
        for li, line in enumerate(lines or []):
            line_times = times[li] if times and isinstance(times[0], list) else times
            for pi, c in enumerate(line or []):
                t = line_times[pi] if line_times and pi < len(line_times) else None
                rows.append((c[1], c[0], c[2] if len(c) > 2 else None, t))
    if not rows:
        raise ValueError("No LineString found in the GeoJSON file")
    return rows, name


# --- Geometry -----------------------------------------------------------------------
def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def cumulative_distance(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    seg = haversine_km(lat[:-1], lon[:-1], lat[1:], lon[1:])
    return np.concatenate([[0.0], np.cumsum(seg)])


def bearing_deg(lat1, lon1, lat2, lon2):
    """Initial compass bearing from point 1 to point 2 (0 = north, 90 = east)."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(np.asarray(lon2) - np.asarray(lon1))
    x = np.sin(dl) * np.cos(p2)
    y = np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl)
    return (np.degrees(np.arctan2(x, y)) + 360) % 360


def _smooth(e: pd.Series) -> pd.Series:
    """Light rolling median against GPS altitude noise (skipped on short tracks, where it would erase climbs)."""
    return e.rolling(5, center=True, min_periods=1).median() if len(e) >= 15 else e


def elevation_stats(ele: pd.Series) -> dict:
    """Ascent and descent on a lightly smoothed profile (raw GPS altitude is noisy)."""
    e = ele.dropna()
    if len(e) < 2:
        return {"ascent_m": None, "descent_m": None, "min_ele": None, "max_ele": None}
    diff = _smooth(e).diff().dropna()
    return {"ascent_m": float(diff[diff > 0].sum()), "descent_m": float(-diff[diff < 0].sum()),
            "min_ele": float(e.min()), "max_ele": float(e.max())}


def cumulative_ascent(ele: pd.Series) -> np.ndarray:
    e = _smooth(ele.interpolate(limit_direction="both").fillna(0))
    return np.concatenate([[0.0], np.cumsum(np.clip(np.diff(e.to_numpy()), 0, None))])


# --- Sampling and timing ------------------------------------------------------------
def sample_points(track: Track, max_points: int = 40, min_step_km: float = 1.0) -> pd.DataFrame:
    """Points spread evenly along the track where the weather is read.

    The step is at least `min_step_km` (finer than the weather models' grid is pointless)
    and the number of points is capped to keep API usage low. Start and end are always kept.
    """
    df = track.points
    total = track.distance_km
    step = max(total / max(max_points - 1, 1), min_step_km)
    targets = np.append(np.arange(0, total, step), total)
    idx = np.unique(np.clip(np.searchsorted(df["dist_km"].to_numpy(), targets), 0, len(df) - 1))
    s = df.iloc[idx].reset_index(drop=True).copy()
    # Direction of travel at each sample: from the previous to the next sample (more stable than GPS jitter)
    prev = s.shift(1).fillna(s.iloc[0])
    nxt = s.shift(-1).fillna(s.iloc[-1])
    s["bearing"] = bearing_deg(prev["lat"], prev["lon"], nxt["lat"], nxt["lon"])
    return s


def passage_times(track: Track, start: pd.Timestamp, speed_kmh: float,
                  climb_min_per_100m: float = 0.0) -> pd.Series:
    """Estimated time at each point of a planned trip."""
    if speed_kmh <= 0:
        raise ValueError("Speed must be positive")
    df = track.points
    hours = df["dist_km"].to_numpy() / speed_kmh
    if climb_min_per_100m and track.has_elevation:
        hours = hours + cumulative_ascent(df["ele"]) / 100 * climb_min_per_100m / 60
    start = pd.Timestamp(start)
    start = start.tz_convert("UTC") if start.tzinfo else start.tz_localize("UTC")
    return pd.Series(start + pd.to_timedelta(hours, unit="h"), index=df.index)


def with_times(track: Track, times: pd.Series) -> Track:
    pts = track.points.copy()
    pts["time"] = times.to_numpy()
    pts["time"] = pd.to_datetime(pts["time"], utc=True)
    return Track(name=track.name, points=pts)


def summary(track: Track) -> dict:
    df = track.points
    out = {"name": track.name, "points": len(df), "distance_km": track.distance_km, **elevation_stats(df["ele"])}
    if track.has_time:
        out["start"] = df["time"].iloc[0]
        out["end"] = df["time"].iloc[-1]
        out["duration_h"] = (out["end"] - out["start"]).total_seconds() / 3600
    return out


def to_gpx(track: Track) -> str:
    """Write a Track back to GPX 1.1 (used to build the sample tracks)."""
    name = escape(track.name)
    lines = ['<?xml version="1.0" encoding="UTF-8"?>',
             '<gpx version="1.1" creator="Météo Analytics" xmlns="http://www.topografix.com/GPX/1/1">',
             f"  <metadata><name>{name}</name></metadata>",
             f"  <trk><name>{name}</name><trkseg>"]
    for p in track.points.itertuples():
        ele = f"<ele>{p.ele:.0f}</ele>" if pd.notna(p.ele) else ""
        time = f"<time>{p.time.strftime('%Y-%m-%dT%H:%M:%SZ')}</time>" if pd.notna(p.time) else ""
        lines.append(f'    <trkpt lat="{p.lat:.6f}" lon="{p.lon:.6f}">{ele}{time}</trkpt>')
    lines += ["  </trkseg></trk>", "</gpx>"]
    return "\n".join(lines) + "\n"
