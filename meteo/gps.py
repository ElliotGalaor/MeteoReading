"""GPS tracks: parsing (GPX, KML, GeoJSON), distances, sampling and passage times.

Pure functions with no network access, covered by unit tests. Uploaded XML files are
parsed with defusedxml, which rejects malicious documents (entity expansion attacks).
"""
from __future__ import annotations

import json
import re
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

    @property
    def looks_scrambled(self) -> bool:
        """Far longer than its own extent: points out of order (OpenStreetMap shuffles anonymous traces)."""
        p = self.points
        diag = float(haversine_km(p["lat"].min(), p["lon"].min(), p["lat"].max(), p["lon"].max()))
        return self.distance_km > 10 and self.distance_km > 60 * max(diag, 0.05)


# --- Parsing ------------------------------------------------------------------------
JOIN_KM = 0.5        # consecutive tracks whose ends are closer than this are one route (the days of a trip)
MAX_TRACKS = 100     # more separate lines than this: a network or a collection of routes, not one route
GEOJSON_TYPES = {"Feature", "FeatureCollection", "Point", "MultiPoint", "LineString", "MultiLineString",
                 "Polygon", "MultiPolygon", "GeometryCollection"}


@dataclass
class _Piece:
    """One route of a file (a GPX track, a KML placemark, a GeoJSON feature), as lines to chain into one."""
    name: str | None
    parts: list[list[tuple]]   # (lat, lon, ele, time) rows of each line


def read_tracks(data: bytes, filename: str) -> list[Track]:
    """The routes of a GPX, KML or GeoJSON file, in file order.

    Tracks, placemarks or features are kept apart, except consecutive ones whose ends touch (the days of one trip).
    Any unreadable file raises a ValueError with a plain message."""
    ext = Path(filename).suffix.lower()
    stem = Path(filename).stem
    parsers = {".gpx": ("GPX", _parse_gpx), ".kml": ("KML", _parse_kml),
               ".geojson": ("GeoJSON", _parse_geojson), ".json": ("GeoJSON", _parse_geojson)}
    if ext not in parsers:
        raise ValueError(f"Unsupported file type '{ext}': use .gpx, .kml or .geojson")
    kind, parser = parsers[ext]
    if not data.strip():
        raise ValueError("the file is empty.")
    try:
        pieces, file_name = parser(data)
    except DefusedXmlException:
        raise ValueError("the file declares XML entities or a DTD, which are refused for security reasons.") from None
    except ParseError as exc:
        line = exc.position[0] if getattr(exc, "position", None) else "?"
        raise ValueError(f"this is not a valid {kind} file: the XML is broken or cut off (line {line}).") from None
    except json.JSONDecodeError:
        raise ValueError("this is not a valid GeoJSON file: the JSON is broken or cut off.") from None
    return _assemble(pieces, file_name or stem)


def parse_track(data: bytes, filename: str) -> Track:
    """One track from a file: its only route, or the longest one when it holds several (see read_tracks)."""
    return main_track(read_tracks(data, filename))


def main_track(tracks: list[Track]) -> Track:
    """The track to use by default: the longest one whose points are in order."""
    sane = [t for t in tracks if not t.looks_scrambled] or tracks
    return max(sane, key=lambda t: t.distance_km)


def _clean_name(name: str | None) -> str | None:
    """'[Itinéraire] GR 3' -> 'GR 3' (a category tag some exports put before the name)."""
    if not name or not name.strip():
        return None
    return re.sub(r"^\[[^\]]{1,30}\]\s*", "", name.strip()) or name.strip()


def _assemble(pieces: list[_Piece], default_name: str) -> list[Track]:
    total = sum(len(rows) for p in pieces for rows in p.parts)
    if total > MAX_POINTS:
        raise ValueError(f"Track too large ({total:,} points, maximum {MAX_POINTS:,})")
    tracks = []
    for p in pieces:
        try:
            tracks.append(build_track(chain_parts(p.parts), _clean_name(p.name) or default_name))
        except ValueError:
            continue   # a line with fewer than 2 valid points
    if not tracks:
        raise ValueError("The track needs at least 2 valid points")
    tracks = _join_touching(tracks)
    if len(tracks) > MAX_TRACKS:
        raise ValueError(f"the file holds {len(tracks):,} separate lines: it looks like a network or a collection of "
                         "routes, not one route. Export a single route.")
    names = [t.name for t in tracks]
    for i, t in enumerate(tracks, 1):   # tell apart tracks with the same name
        if names.count(t.name) > 1:
            t.name = f"{t.name} ({i})"
    return tracks


def _rows(track: Track) -> list[tuple]:
    p = track.points
    times = [t.isoformat() if pd.notna(t) else None for t in p["time"]]
    return list(zip(p["lat"], p["lon"], p["ele"], times, strict=True))


def _join_touching(tracks: list[Track]) -> list[Track]:
    """Joins consecutive tracks when one ends where the next starts (and, with times, the next starts later)."""
    out = [tracks[0]]
    for nxt in tracks[1:]:
        cur = out[-1]
        a, b = cur.points.iloc[-1], nxt.points.iloc[0]
        gap = float(haversine_km(a["lat"], a["lon"], b["lat"], b["lon"]))
        if cur.has_time and nxt.has_time:
            in_order = nxt.points["time"].iloc[0] >= cur.points["time"].iloc[-1]
        else:
            in_order = not cur.has_time and not nxt.has_time
        if gap <= JOIN_KM and in_order:
            out[-1] = build_track(_rows(cur) + _rows(nxt), cur.name)
        else:
            out.append(nxt)
    return out


def chain_parts(parts: list[list[tuple]]) -> list[tuple]:
    """One line from the pieces of a MultiLineString (or a KML MultiGeometry): from the longest piece, the nearest
    piece is added to whichever end it is closest to, reversed if needed. Timed pieces go in time order instead."""
    parts = [p for p in parts if p]
    if len(parts) <= 1:
        return parts[0] if parts else []
    if all(p[0][3] and p[-1][3] for p in parts):
        try:
            return [r for p in sorted(parts, key=lambda p: pd.Timestamp(p[0][3])) for r in p]
        except (ValueError, TypeError):
            pass

    def ends(p):
        try:
            return float(p[0][0]), float(p[0][1]), float(p[-1][0]), float(p[-1][1])
        except (TypeError, ValueError):
            return np.nan, np.nan, np.nan, np.nan

    e = np.array([ends(p) for p in parts])
    order = np.argsort([-len(p) for p in parts])
    chain = list(parts[order[0]])
    left = [i for i in range(len(parts)) if i != order[0]]
    while left:
        s_lat, s_lon = float(chain[0][0]), float(chain[0][1])
        t_lat, t_lon = float(chain[-1][0]), float(chain[-1][1])
        sub = e[left]
        # distances: tail -> start, tail -> end (reversed), end -> head, start -> head (reversed)
        d = np.stack([haversine_km(t_lat, t_lon, sub[:, 0], sub[:, 1]), haversine_km(t_lat, t_lon, sub[:, 2], sub[:, 3]),
                      haversine_km(sub[:, 2], sub[:, 3], s_lat, s_lon), haversine_km(sub[:, 0], sub[:, 1], s_lat, s_lon)])
        d = np.where(np.isnan(d), np.inf, d)
        how, k = np.unravel_index(int(np.argmin(d)), d.shape)
        piece = parts[left.pop(int(k))]
        if how == 0:
            chain += piece
        elif how == 1:
            chain += piece[::-1]
        elif how == 2:
            chain = list(piece) + chain
        else:
            chain = list(piece[::-1]) + chain
    return chain


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


def _child_text(el, name: str) -> str | None:
    for child in el:
        if _local(child.tag) == name and child.text and child.text.strip():
            return child.text.strip()
    return None


def _gpx_point(el) -> tuple:
    return el.get("lat"), el.get("lon"), _child_text(el, "ele"), _child_text(el, "time")


def _parse_gpx(data: bytes) -> tuple[list[_Piece], str | None]:
    root = ET.fromstring(data)
    meta = next((el for el in root if _local(el.tag) == "metadata"), None)
    # GPX 1.1: metadata/name; GPX 1.0: a name right under <gpx> (never a waypoint's name)
    file_name = _child_text(meta, "name") if meta is not None else _child_text(root, "name")
    tracks, routes, waypoints = [], [], []
    for el in root.iter():
        tag = _local(el.tag)
        if tag == "trk":
            rows = [_gpx_point(p) for seg in el if _local(seg.tag) == "trkseg" for p in seg if _local(p.tag) == "trkpt"]
            tracks.append(_Piece(_child_text(el, "name") or file_name, [rows]))
        elif tag == "rte":
            rows = [_gpx_point(p) for p in el if _local(p.tag) == "rtept"]
            routes.append(_Piece(_child_text(el, "name") or file_name, [rows]))
        elif tag == "wpt":
            waypoints.append(_gpx_point(el))
    # Recorded tracks first, then planned routes; waypoints only when there is neither (a list of places to visit)
    pieces = [p for p in tracks if p.parts[0]] or [p for p in routes if p.parts[0]]
    if not pieces and waypoints:
        pieces = [_Piece(file_name, [waypoints])]
    if not pieces:
        raise ValueError("No track, route or waypoints found in the GPX file")
    return pieces, file_name


def _kml_coords(text: str | None) -> list[tuple]:
    rows = []
    for chunk in (text or "").split():
        parts = chunk.split(",")
        if len(parts) >= 2:
            rows.append((parts[1], parts[0], parts[2] if len(parts) > 2 else None, None))
    return rows


def _parse_kml(data: bytes) -> tuple[list[_Piece], str | None]:
    root = ET.fromstring(data)
    doc = next((el for el in root.iter() if _local(el.tag) == "Document"), root)
    file_name = _child_text(doc, "name")
    pieces, points = [], []
    for pm in (el for el in root.iter() if _local(el.tag) == "Placemark"):
        name = _child_text(pm, "name") or file_name
        # gx:Track (with timestamps) takes priority over plain coordinates
        tracks = [el for el in pm.iter() if _local(el.tag) == "Track"]
        if tracks:
            parts = []
            for tr in tracks:
                whens = [el.text for el in tr if _local(el.tag) == "when"]
                coords = [el.text for el in tr if _local(el.tag) == "coord"]
                rows = []
                for when, c in zip(whens, coords, strict=False):
                    v = (c or "").split()
                    if len(v) >= 2:
                        rows.append((v[1], v[0], v[2] if len(v) > 2 else None, when))
                parts.append(rows)
            if any(parts):
                pieces.append(_Piece(name, parts))
                continue
        # Lines only: the coordinates of Point placemarks are points of interest, not part of the route
        lines = [el for el in pm.iter() if _local(el.tag) in ("LineString", "LinearRing")]
        parts = [_kml_coords(_child_text(line, "coordinates")) for line in lines]
        if any(parts):
            pieces.append(_Piece(name, [p for p in parts if p]))
        elif not lines:
            for pt in (el for el in pm.iter() if _local(el.tag) == "Point"):
                points += _kml_coords(_child_text(pt, "coordinates"))
    if not pieces and len(points) >= 2:   # only places: read them in order, like GPX waypoints
        pieces = [_Piece(file_name, [points])]
    if not pieces:
        raise ValueError("No line found in the KML file")
    return pieces, file_name


def _point(c) -> tuple | None:
    """GeoJSON position [lon, lat(, ele)] -> (lat, lon, ele), or None if it is not one."""
    if isinstance(c, (list, tuple)) and len(c) >= 2:
        return c[1], c[0], c[2] if len(c) > 2 else None
    return None


def _load_json(data: bytes):
    try:
        return json.loads(data)
    except UnicodeDecodeError:   # not UTF-8: some open-data exports are in Latin-1
        return json.loads(data.decode("latin-1"))


def _lines(geom: dict) -> list:
    """The lines of a geometry (LineString, MultiLineString, GeometryCollection)."""
    kind, coords = geom.get("type"), geom.get("coordinates")
    if kind == "LineString":
        return [coords]
    if kind == "MultiLineString":
        return coords if isinstance(coords, list) else []
    if kind == "GeometryCollection":
        return [line for g in geom.get("geometries") or [] if isinstance(g, dict) for line in _lines(g)]
    return []


def _parse_geojson(data: bytes) -> tuple[list[_Piece], str | None]:
    obj = _load_json(data)
    if not isinstance(obj, dict) or not (obj.get("type") in GEOJSON_TYPES or "features" in obj):
        raise ValueError("this JSON file is not GeoJSON: it has no Feature, FeatureCollection or geometry.")
    file_name = obj.get("name") if isinstance(obj.get("name"), str) else None
    features = obj.get("features") or ([obj] if obj.get("type") == "Feature" else [{"geometry": obj}])
    pieces, points = [], []
    for feat in features if isinstance(features, list) else []:
        if not isinstance(feat, dict):
            continue
        geom = feat.get("geometry") if isinstance(feat.get("geometry"), dict) else {}
        props = feat.get("properties") if isinstance(feat.get("properties"), dict) else {}
        name = props.get("name") if isinstance(props.get("name"), str) else None
        times = props.get("coordTimes") or props.get("times")
        times = times if isinstance(times, list) else None
        kind, coords = geom.get("type"), geom.get("coordinates")
        if kind in ("Point", "MultiPoint"):
            # Points only count if there is no line, like the waypoints of a GPX file
            for c in [coords] if kind == "Point" else coords or []:
                if (p := _point(c)) is not None:
                    points.append((*p, None))
            continue
        parts = []
        for li, line in enumerate(_lines(geom)):
            line_times = times[li] if times and isinstance(times[0], list) and li < len(times) else times
            rows = []
            for pi, c in enumerate(line if isinstance(line, list) else []):
                if (p := _point(c)) is not None:
                    t = line_times[pi] if line_times and pi < len(line_times) else None
                    rows.append((*p, t))
            if rows:
                parts.append(rows)
        if parts:
            pieces.append(_Piece(name or file_name, parts))
    if not pieces and points:
        pieces = [_Piece(file_name, [points])]
    if not pieces:
        raise ValueError("No line or points found in the GeoJSON file")
    return pieces, file_name


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
    s["track_idx"] = idx   # row of the track: another plan of the same track gives its new times here
    # Direction of travel at each sample: from the previous to the next sample (more stable than GPS jitter)
    prev = s.shift(1).fillna(s.iloc[0])
    nxt = s.shift(-1).fillna(s.iloc[-1])
    s["bearing"] = bearing_deg(prev["lat"], prev["lon"], nxt["lat"], nxt["lon"])
    return s


def moving_hours(track: Track, speed_kmh: float, climb_min_per_100m: float = 0.0) -> np.ndarray:
    """Hours on the move to reach each point (flat speed, plus time for climbs)."""
    if speed_kmh <= 0:
        raise ValueError("Speed must be positive")
    df = track.points
    hours = df["dist_km"].to_numpy() / speed_kmh
    if climb_min_per_100m and track.has_elevation:
        hours = hours + cumulative_ascent(df["ele"]) / 100 * climb_min_per_100m / 60
    return hours


def passage_times(track: Track, start: pd.Timestamp, speed_kmh: float, climb_min_per_100m: float = 0.0,
                  hours_per_day: float | None = None) -> pd.Series:
    """Estimated time at each point of a planned trip.

    With hours_per_day, the trip is split into days: after that many hours on the move, it goes on the next day at
    the same departure time (the stages of a multi-day route)."""
    return times_from_hours(track, start, moving_hours(track, speed_kmh, climb_min_per_100m), hours_per_day)


def times_from_hours(track: Track, start: pd.Timestamp, hours: np.ndarray,
                     hours_per_day: float | None = None) -> pd.Series:
    """Time at each point from the hours on the move to reach it (see passage_times for `hours_per_day`)."""
    hours = np.asarray(hours, dtype=float)
    start = pd.Timestamp(start)
    start = start.tz_convert("UTC") if start.tzinfo else start.tz_localize("UTC")
    if hours_per_day and hours_per_day > 0:
        day = np.floor(hours / hours_per_day - 1e-9).clip(min=0)
        offset = pd.to_timedelta(day, unit="D") + pd.to_timedelta(hours - day * hours_per_day, unit="h")
        return pd.Series(start + offset, index=track.points.index)
    return pd.Series(start + pd.to_timedelta(hours, unit="h"), index=track.points.index)


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
