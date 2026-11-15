"""GPS tracks: parsing (GPX, KML, GeoJSON), distances, sampling and passage times.

Pure functions with no network access, covered by unit tests. Uploaded XML files are
parsed with defusedxml, which rejects malicious documents (entity expansion attacks).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from xml.etree.ElementTree import ParseError
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd
from defusedxml import DefusedXmlException
from defusedxml import ElementTree as ET

MAX_POINTS = 200_000
ELE_DEADBAND_M = 5.0      # altitude changes smaller than this are noise
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
    time_issue: str | None = None   # why the file's timestamps were not used, if it had some

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
    """Read a GPX, KML or GeoJSON file into a Track. Any unreadable file raises a ValueError with a plain message."""
    ext = Path(filename).suffix.lower()
    name = Path(filename).stem
    parsers = {".gpx": ("GPX", _parse_gpx), ".kml": ("KML", _parse_kml),
               ".geojson": ("GeoJSON", _parse_geojson), ".json": ("GeoJSON", _parse_geojson)}
    if ext not in parsers:
        raise ValueError(f"Unsupported file type '{ext}': use .gpx, .kml or .geojson")
    kind, parser = parsers[ext]
    if not data.strip():
        raise ValueError("the file is empty.")
    try:
        rows, found_name = parser(data)
    except DefusedXmlException:
        raise ValueError("the file declares XML entities or a DTD, which are refused for security reasons.") from None
    except ParseError as exc:
        line = exc.position[0] if getattr(exc, "position", None) else "?"
        raise ValueError(f"this is not a valid {kind} file: the XML is broken or cut off (line {line}).") from None
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise ValueError("this is not a valid GeoJSON file: the JSON is broken or cut off.") from None
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
    time_issue = None
    if t.notna().any() and t.notna().mean() < 0.9:
        time_issue = "too many points have no timestamp"
    elif t.notna().any() and not t.dropna().is_monotonic_increasing:
        time_issue = "the timestamps are not in order"
    if time_issue or t.isna().all():
        df["time"] = pd.NaT
    elif t.isna().any():
        df["time"] = t.interpolate(limit_direction="both")
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df["dist_km"] = cumulative_distance(df["lat"].to_numpy(), df["lon"].to_numpy())
    return Track(name=name, points=df, time_issue=time_issue)


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
        for when, c in zip(whens, coords, strict=True):
            parts = (c or "").split()
            if len(parts) >= 2:
                rows.append((parts[1], parts[0], parts[2] if len(parts) > 2 else None, when))
        if rows:
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


def _point(c) -> tuple | None:
    """GeoJSON position [lon, lat(, ele)] -> (lat, lon, ele), or None if it is not one."""
    if isinstance(c, (list, tuple)) and len(c) >= 2:
        return c[1], c[0], c[2] if len(c) > 2 else None
    return None


def _parse_geojson(data: bytes) -> tuple[list[tuple], str | None]:
    obj = json.loads(data)
    if not isinstance(obj, dict):
        raise ValueError("this GeoJSON file holds no Feature, FeatureCollection or geometry.")
    features = obj.get("features") or ([obj] if obj.get("type") == "Feature" else [{"geometry": obj}])
    rows, points, name = [], [], None
    for feat in features if isinstance(features, list) else []:
        if not isinstance(feat, dict):
            continue
        geom = feat.get("geometry") if isinstance(feat.get("geometry"), dict) else {}
        props = feat.get("properties") if isinstance(feat.get("properties"), dict) else {}
        name = name or props.get("name")
        times = props.get("coordTimes") or props.get("times")
        times = times if isinstance(times, list) else None
        kind, coords = geom.get("type"), geom.get("coordinates")
        if kind in ("Point", "MultiPoint"):
            # Points only count if there is no line, like the waypoints of a GPX file
            for c in [coords] if kind == "Point" else coords or []:
                if (p := _point(c)) is not None:
                    points.append((*p, None))
            continue
        lines = {"LineString": [coords], "MultiLineString": coords}.get(kind) or []
        for li, line in enumerate(lines if isinstance(lines, list) else []):
            line_times = times[li] if times and isinstance(times[0], list) and li < len(times) else times
            for pi, c in enumerate(line if isinstance(line, list) else []):
                if (p := _point(c)) is not None:
                    t = line_times[pi] if line_times and pi < len(line_times) else None
                    rows.append((*p, t))
    rows = rows or points
    if not rows:
        raise ValueError("No line or points found in the GeoJSON file")
    return rows, name


# --- Geometry -----------------------------------------------------------------------
def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def cumulative_distance(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    seg = haversine_km(lat[:-1], lon[:-1], lat[1:], lon[1:])
    return np.concatenate([[0.0], np.cumsum(seg)])


def segment_km(dist_km) -> np.ndarray:
    """Distance around each point of a polyline (half of each neighbouring leg): shares of a route are over km."""
    d = np.asarray(dist_km, dtype=float)
    if len(d) < 2:
        return np.ones(len(d))
    return np.diff(np.concatenate([[0.0], (d[:-1] + d[1:]) / 2, [d[-1]]]))


def bearing_deg(lat1, lon1, lat2, lon2):
    """Initial compass bearing from point 1 to point 2 (0 = north, 90 = east)."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(np.asarray(lon2) - np.asarray(lon1))
    x = np.sin(dl) * np.cos(p2)
    y = np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl)
    return (np.degrees(np.arctan2(x, y)) + 360) % 360


def _smooth(e: pd.Series) -> pd.Series:
    """Rolling median (removes spikes) then mean (removes jitter) against altitude noise; skipped on short tracks,
    where it would erase climbs."""
    if len(e) < 15:
        return e
    return e.rolling(5, center=True, min_periods=1).median().rolling(5, center=True, min_periods=1).mean()


def climbs(e: np.ndarray, band: float = ELE_DEADBAND_M) -> tuple[np.ndarray, np.ndarray]:
    """Metres gained and lost at each point, ignoring ups and downs smaller than `band` (a dead band, as GPS apps
    do): summing every small difference would count the ±10 m noise of GPS or DEM altitudes as climbing."""
    gain, loss = np.zeros(len(e)), np.zeros(len(e))
    if not len(e):
        return gain, loss
    ref = e[0]
    for i in range(1, len(e)):
        if e[i] - ref >= band:
            gain[i], ref = e[i] - ref, e[i]
        elif ref - e[i] >= band:
            loss[i], ref = ref - e[i], e[i]
    return gain, loss


def elevation_stats(ele: pd.Series) -> dict:
    """Ascent and descent on a lightly smoothed profile, with a dead band (raw altitude is noisy)."""
    e = ele.dropna()
    if len(e) < 2:
        return {"ascent_m": None, "descent_m": None, "min_ele": None, "max_ele": None}
    gain, loss = climbs(_smooth(e).to_numpy())
    return {"ascent_m": float(gain.sum()), "descent_m": float(loss.sum()),
            "min_ele": float(e.min()), "max_ele": float(e.max())}


def cumulative_ascent(ele: pd.Series) -> np.ndarray:
    e = _smooth(ele.interpolate(limit_direction="both").fillna(0))
    return np.cumsum(climbs(e.to_numpy())[0])


def resample_path(lat, lon, step_km: float = 0.2, max_points: int = 1000) -> tuple[np.ndarray, np.ndarray]:
    """Points evenly spaced every `step_km` along a polyline (the step grows if needed to stay under `max_points`)."""
    lat, lon = np.asarray(lat, dtype=float), np.asarray(lon, dtype=float)
    dist = cumulative_distance(lat, lon)
    total = dist[-1]
    if total == 0:
        return lat[:1], lon[:1]
    step = max(step_km, total / max(max_points - 1, 1))
    targets = np.append(np.arange(0, total, step), total)
    return np.interp(targets, dist, lat), np.interp(targets, dist, lon)


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
