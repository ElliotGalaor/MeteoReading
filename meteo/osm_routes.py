"""Waymarked routes near a track from OpenStreetMap, where no Geotrek site covers the area.

The list comes from Waymarked Trails (https://waymarkedtrails.org), which indexes the hiking, cycling and mountain-bike
route relations of OpenStreetMap: one request per kind of route, for the track's box. Only local and regional routes
are kept: a European or national path (E4, GR 4) runs for hundreds of kilometres and is not a route to read in one go.

A route's line is loaded only when asked, from the OpenStreetMap API (`/relation/<id>/full`), and its ways are chained
by their nearest ends. Data © OpenStreetMap contributors, ODbL.
"""
from __future__ import annotations

import logging
import math
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .download import quick_session
from .geotrek import Suggestion, search_box
from .gps import MAX_POINTS, Track, build_track, chain_parts

log = logging.getLogger(__name__)

SITES = {"hiking": "Hiking", "cycling": "Cycling", "mtb": "Mountain bike"}
KEEP_GROUPS = {"LOC", "REG"}          # local and regional routes (not INT / NAT long-distance paths)
MAX_BOX_KM = 10.0                     # the listing is by area only: keep the box tight around the track
MAX_ROUTE_KM = 300.0
OSM_API = "https://api.openstreetmap.org/api/0.6"


def mercator(lon: float, lat: float) -> tuple[float, float]:
    """Web Mercator metres (the coordinates Waymarked Trails' API expects)."""
    x = lon * 20037508.34 / 180
    y = math.log(math.tan((90 + lat) * math.pi / 360)) / (math.pi / 180) * 20037508.34 / 180
    return x, y


def nearby_relations(lat, lon, radius_km: float = 10.0, session=None) -> tuple[list[Suggestion], list[str]]:
    """Local and regional OSM routes in the track's box (widened by the radius, at most MAX_BOX_KM)."""
    lat, lon = np.asarray(lat, dtype=float), np.asarray(lon, dtype=float)
    margin = min(radius_km, MAX_BOX_KM)
    w, s, e, n = search_box(lat, lon, margin)
    x0, y0 = mercator(w, s)
    x1, y1 = mercator(e, n)
    session = session or quick_session()

    def one(site: str) -> list[Suggestion]:
        resp = session.get(f"https://{site}.waymarkedtrails.org/api/v1/list/by_area",
                           params={"bbox": f"{x0:.0f},{y0:.0f},{x1:.0f},{y1:.0f}", "limit": 100}, timeout=15)
        resp.raise_for_status()
        out = []
        for r in resp.json().get("results", []):
            if r.get("type") != "relation" or r.get("group") not in KEEP_GROUPS:
                continue
            name = r.get("name") or r.get("ref") or f"Route {r['id']}"
            if r.get("ref") and r.get("name") and r["ref"] not in r["name"]:
                name = f"{r['ref']} · {r['name']}"
            out.append(Suggestion(
                source="OpenStreetMap", trek_id=int(r["id"]), name=name, practice=SITES[site], length_km=None,
                ascent_m=None, duration_h=None, distance_km=None, gpx_url=None,
                page_url=f"https://www.openstreetmap.org/relation/{r['id']}", route_type="",
                passes_near=False, provider="OpenStreetMap"))
        return out

    found, failed = [], []
    with ThreadPoolExecutor(max_workers=len(SITES)) as pool:
        futures = {pool.submit(one, site): site for site in SITES}
        for fut, site in futures.items():
            try:
                found += fut.result()
            except (OSError, ValueError, KeyError, TypeError) as exc:   # requests' errors are OSErrors
                log.warning("Waymarked Trails %s: %s", site, exc)
                failed.append(f"OpenStreetMap ({SITES[site].lower()})")
    unique = {s.trek_id: s for s in found}       # a route can be both hiking and MTB: keep one
    return sorted(unique.values(), key=lambda s: s.name), failed


def relation_track(relation_id: int, name: str | None = None, session=None) -> Track:
    """The line of an OSM route relation, its ways chained by their nearest ends."""
    session = session or quick_session()
    try:
        resp = session.get(f"{OSM_API}/relation/{int(relation_id)}/full.json", timeout=30)
    except OSError as exc:
        raise RuntimeError("OpenStreetMap is not answering: try again in a moment.") from exc
    if resp.status_code != 200:
        raise RuntimeError(f"Could not load this OpenStreetMap route ({resp.status_code})")
    elements = resp.json().get("elements", [])
    nodes = {e["id"]: (e["lat"], e["lon"]) for e in elements if e.get("type") == "node"}
    ways = {e["id"]: e.get("nodes", []) for e in elements if e.get("type") == "way"}
    rel = next((e for e in elements if e.get("type") == "relation" and e.get("id") == int(relation_id)), None)
    if rel is None:
        raise ValueError("This OpenStreetMap relation is empty")
    parts = []
    for m in rel.get("members", []):
        if m.get("type") == "way" and m.get("ref") in ways and m.get("role", "") in ("", "forward", "backward", "main"):
            pts = [nodes[i] for i in ways[m["ref"]] if i in nodes]
            if len(pts) >= 2:
                parts.append([(la, lo, None, None) for la, lo in pts])
    if sum(len(p) for p in parts) > MAX_POINTS:
        raise ValueError("This route is too large to read in one go")
    track = build_track(chain_parts(parts), name or rel.get("tags", {}).get("name") or f"Relation {relation_id}")
    if track.distance_km > MAX_ROUTE_KM:
        raise ValueError(f"This route is {track.distance_km:,.0f} km long: too long to read in one go")
    return track
