"""Build a track from points clicked on a map.

- optionally follow roads and paths between the waypoints (OSRM routing engine on
  OpenStreetMap data, foot / bike / car profiles, hosted by FOSSGIS for light use);
- resample the line every few hundred metres;
- add the altitude of each point (Open-Meteo elevation API, 90 m terrain model),
  so that ascent and travel times stay realistic.
"""
from __future__ import annotations

import numpy as np

from .download import make_session
from .fmt import num
from .gps import Track, build_track, resample_path
from .i18n import t

OSRM_SERVERS = {
    "Walking": "https://routing.openstreetmap.de/routed-foot",
    "Cycling": "https://routing.openstreetmap.de/routed-bike",
    "Driving": "https://routing.openstreetmap.de/routed-car",
}
ELEVATION_API = "https://api.open-meteo.com/v1/elevation"
GEOCODING_API = "https://geocoding-api.open-meteo.com/v1/search"
MAX_WAYPOINTS = 50
STEP_KM = {"Walking": 0.1, "Cycling": 0.2, "Driving": 0.5}


def waypoints_from_drawings(drawings: list[dict] | None) -> list[tuple[float, float]]:
    """(lat, lon) waypoints of the most recent line drawn on the map (GeoJSON features from Leaflet.draw)."""
    lines = [f["geometry"]["coordinates"] for f in drawings or []
             if (f.get("geometry") or {}).get("type") == "LineString"]
    if not lines:
        return []
    return [(float(c[1]), float(c[0])) for c in lines[-1]]


def parse_osrm(payload: dict) -> tuple[list[tuple[float, float]], float, float]:
    """OSRM /route response -> ((lat, lon) path, distance in km, duration in hours)."""
    if payload.get("code") != "Ok" or not payload.get("routes"):
        raise RuntimeError(t("err.no_route", reason=payload.get("message") or payload.get("code")))
    route = payload["routes"][0]
    path = [(lat, lon) for lon, lat in route["geometry"]["coordinates"]]
    return path, route["distance"] / 1000, route["duration"] / 3600


def route_path(waypoints: list[tuple[float, float]], activity: str, session=None) -> list[tuple[float, float]]:
    """Path following roads/paths through the waypoints, for a walking, cycling or driving profile."""
    if len(waypoints) > MAX_WAYPOINTS:
        raise ValueError(t("err.waypoints", n=len(waypoints), max=MAX_WAYPOINTS))
    session = session or make_session()
    coords = ";".join(f"{lon:.6f},{lat:.6f}" for lat, lon in waypoints)
    resp = session.get(f"{OSRM_SERVERS[activity]}/route/v1/driving/{coords}",
                       params={"overview": "full", "geometries": "geojson"}, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(t("err.routing", status=resp.status_code))
    path, _, _ = parse_osrm(resp.json())
    return path


def fetch_elevation(lat, lon, session=None, batch: int = 100) -> list[float]:
    """Terrain altitude of each point (Open-Meteo accepts 100 coordinates per request)."""
    session = session or make_session()
    out: list[float] = []
    for i in range(0, len(lat), batch):
        resp = session.get(ELEVATION_API, timeout=30, params={
            "latitude": ",".join(f"{v:.5f}" for v in lat[i:i + batch]),
            "longitude": ",".join(f"{v:.5f}" for v in lon[i:i + batch])})
        if resp.status_code != 200:
            raise RuntimeError(t("err.elevation", status=resp.status_code))
        out += resp.json()["elevation"]
    return out


def geocode(name: str, session=None, count: int = 5) -> list[dict]:
    """Places matching a name: [{'label', 'lat', 'lon'}]."""
    session = session or make_session()
    resp = session.get(GEOCODING_API, params={"name": name, "count": count, "language": "en", "format": "json"},
                       timeout=20)
    resp.raise_for_status()
    out = []
    for r in resp.json().get("results") or []:
        parts = [r.get("name"), r.get("admin1"), r.get("country")]
        out.append({"label": ", ".join(p for p in parts if p), "lat": r["latitude"], "lon": r["longitude"]})
    return out


def drawn_name(activity: str, km: float) -> str:
    """Name of a route drawn on the map, in the page's language: "Drawn route (cycling, 7.7 km)"."""
    return t("route.drawn_name", activity=t(f"activity.{activity}").lower(), km=num(km, 1))


def track_from_waypoints(waypoints: list[tuple[float, float]], activity: str = "Cycling", follow_paths: bool = True,
                         name: str | None = None, session=None) -> tuple[Track, bool]:
    """Track through clicked waypoints. Returns (track, whether roads/paths were followed).

    If the routing service fails, the waypoints are joined by straight lines instead.
    """
    if len(waypoints) < 2:
        raise ValueError(t("err.two_clicks"))
    session = session or make_session()
    routed = False
    path = waypoints
    if follow_paths:
        try:
            path = route_path(waypoints, activity, session)
            routed = True
        except (RuntimeError, ValueError, OSError):
            path = waypoints
    lat, lon = resample_path([p[0] for p in path], [p[1] for p in path], STEP_KM.get(activity, 0.2))
    try:
        ele = fetch_elevation(lat, lon, session)
    except (RuntimeError, OSError):
        ele = [np.nan] * len(lat)
    rows = [(la, lo, e, None) for la, lo, e in zip(lat, lon, ele, strict=True)]
    track = build_track(rows, "route")
    track.name = name or drawn_name(activity, track.distance_km)
    return track, routed
