"""Routes near a track, from the public Geotrek APIs of French parks and départements.

Geotrek (https://geotrek.fr) is the open-source software many parks, départements and tourist offices use to publish
their hiking, cycling and trail routes. Each of them runs its own Geotrek-admin, whose API v2 is public: there is
no national API (the aggregator built for the Cartes IGN app is reserved to that app). So the instances below are
queried directly, only those whose area is near the track, with a bounding box (`in_bbox`).

The routes remain the property of the structures that publish them: they are listed with their source and a link
to their page, and a route's GPX is downloaded only when the visitor asks to read the weather along it.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

from .download import make_session
from .gps import haversine_km

log = logging.getLogger(__name__)

MAX_GPX_BYTES = 5_000_000


@dataclass(frozen=True)
class Instance:
    name: str
    short: str                                 # for tables
    api: str                                   # Geotrek-admin API v2 root
    portal: str | None                         # public site: a route's page is {portal}/trek/{id}
    bbox: tuple[float, float, float, float]    # area of its routes' starts (lon_min, lat_min, lon_max, lat_max)


# Public Geotrek-admin instances whose API v2 answered (November 2026), with the extent of their routes' starts.
# More can be added here: the API root of a Geotrek-rando site is written in its home page.
INSTANCES = [
    Instance("Parc national des Écrins", "PN des Écrins", "https://geotrek-admin.ecrins-parcnational.fr/api/v2",
             "https://rando.ecrins-parcnational.fr", (5.86, 44.43, 6.82, 45.10)),
    Instance("Parc national de la Vanoise", "PN de la Vanoise", "https://adminrando.vanoise.com/api/v2",
             "https://rando.vanoise.com", (6.20, 45.09, 7.14, 45.70)),
    Instance("Parc naturel régional du Vercors", "PNR du Vercors", "https://geo.pnr-vercors.fr/api/v2",
             "https://rando.parc-du-vercors.fr", (4.77, 44.60, 5.77, 45.26)),
    Instance("Chemins des Parcs (parcs naturels régionaux de Provence-Alpes-Côte d'Azur)", "Chemins des Parcs",
             "https://admin.cheminsdesparcs.fr/api/v2", "https://www.cheminsdesparcs.fr", (4.43, 43.19, 7.16, 44.86)),
    Instance("Parc national des Cévennes", "PN des Cévennes", "https://geotrek-admin.cevennes-parcnational.net/api/v2",
             "https://destination.cevennes-parcnational.fr", (3.21, 43.95, 4.20, 44.73)),
    Instance("Parc naturel régional Livradois-Forez", "PNR Livradois-Forez", "https://geotrek-admin.parc-livradois-forez.org/api/v2",
             "https://rando.parc-livradois-forez.org", (3.23, 45.25, 3.95, 46.12)),
    Instance("Département de Loire-Atlantique", "Loire-Atlantique", "https://admin-rando.loire-atlantique.fr/api/v2",
             "https://rando.loire-atlantique.fr", (-2.55, 46.93, -0.93, 47.80)),
    Instance("Département du Doubs", "Doubs", "https://explore-admin.doubs.fr/api/v2",
             "https://explore.doubs.fr", (5.73, 46.59, 6.95, 47.58)),
    Instance("Jura (département)", "Jura", "https://geotrek-cd39-aggregator.makina-corpus.net/api/v2",
             None, (5.33, 46.26, 6.94, 47.51)),
    Instance("Parc naturel régional du Haut-Jura", "PNR du Haut-Jura", "https://randoadmin.parc-haut-jura.fr/api/v2",
             None, (5.45, 45.97, 6.40, 46.90)),
    Instance("Parc naturel régional des Vosges du Nord", "PNR des Vosges du Nord", "https://geotrek.parc-vosges-nord.fr/api/v2",
             "https://rando.parc-vosges-nord.fr", (7.03, 48.74, 7.95, 49.15)),
]

# Geotrek practices (names vary between instances) -> the activities of the route tool
CYCLING_WORDS = ("vtt", "vélo", "velo", "cyclo", "gravel", "bike")


@dataclass
class Suggestion:
    source: str
    trek_id: int
    name: str
    practice: str
    length_km: float | None
    ascent_m: float | None
    duration_h: float | None
    distance_km: float          # from the track to the route's start
    gpx_url: str
    page_url: str | None

    @property
    def activity(self) -> str:
        """The activity of the route tool that fits the practice (walking unless it is a bike practice)."""
        return "Cycling" if any(w in self.practice.lower() for w in CYCLING_WORDS) else "Walking"


def search_box(lat, lon, radius_km: float) -> tuple[float, float, float, float]:
    """Bounding box of the track, widened by radius_km on every side."""
    lat, lon = np.asarray(lat, dtype=float), np.asarray(lon, dtype=float)
    dlat = radius_km / 111.0
    dlon = radius_km / (111.0 * max(np.cos(np.radians(lat.mean())), 0.1))
    return float(lon.min() - dlon), float(lat.min() - dlat), float(lon.max() + dlon), float(lat.max() + dlat)


def overlaps(a: tuple, b: tuple) -> bool:
    return a[0] <= b[2] and b[0] <= a[2] and a[1] <= b[3] and b[1] <= a[3]


def _get(session, url: str, params: dict | None = None) -> dict:
    resp = session.get(url, params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()


def _practices(session, inst: Instance) -> dict[int, str]:
    data = _get(session, f"{inst.api}/trek_practice/", {"language": "fr", "fields": "id,name", "page_size": 100})
    return {p["id"]: p["name"] for p in data.get("results", [])}


def _treks(session, inst: Instance, box: tuple, track_lat: np.ndarray, track_lon: np.ndarray) -> list[Suggestion]:
    """Routes of one instance in the box, with the distance from the track to their start."""
    params = {"in_bbox": ",".join(f"{v:.4f}" for v in box), "language": "fr", "page_size": 500,
              "fields": "id,name,departure_geom,length_2d,ascent,duration,practice,gpx"}
    data = _get(session, f"{inst.api}/trek/", params)
    practices = _practices(session, inst)
    out = []
    for t in data.get("results", []):
        start, gpx = t.get("departure_geom"), t.get("gpx")
        if not start or not gpx or len(start) < 2:
            continue
        dist = float(np.min(haversine_km(track_lat, track_lon, start[1], start[0])))
        length = t.get("length_2d")
        out.append(Suggestion(
            source=inst.short, trek_id=int(t["id"]), name=(t.get("name") or "").strip() or f"Route {t['id']}",
            practice=practices.get(t.get("practice"), ""), length_km=length / 1000 if length else None,
            ascent_m=t.get("ascent"), duration_h=t.get("duration"), distance_km=dist, gpx_url=gpx,
            page_url=f"{inst.portal}/trek/{t['id']}" if inst.portal else None))
    return out


def nearby_treks(lat, lon, radius_km: float = 10.0, limit: int = 10, session=None,
                 instances: list[Instance] = INSTANCES) -> tuple[list[Suggestion], list[str]]:
    """Routes starting within radius_km of the track, nearest first.

    Returns (suggestions, sources that could not be reached). Only the instances whose area is near the track are
    queried, in parallel; one that does not answer is skipped."""
    lat, lon = np.asarray(lat, dtype=float), np.asarray(lon, dtype=float)
    if len(lat) > 300:   # the distance to the track does not need every point
        keep = np.linspace(0, len(lat) - 1, 300).astype(int)
        lat, lon = lat[keep], lon[keep]
    box = search_box(lat, lon, radius_km)
    near = [i for i in instances if overlaps(box, i.bbox)]
    if not near:
        return [], []
    session = session or make_session()
    found, failed = [], []
    with ThreadPoolExecutor(max_workers=min(len(near), 6)) as pool:
        futures = {pool.submit(_treks, session, inst, box, lat, lon): inst for inst in near}
        for fut, inst in futures.items():
            try:
                found += fut.result()
            except (OSError, ValueError, KeyError, TypeError) as exc:   # requests' errors are OSErrors
                log.warning("Geotrek %s: %s", inst.name, exc)
                failed.append(inst.name)
    found = [s for s in found if s.distance_km <= radius_km]
    found.sort(key=lambda s: (s.distance_km, s.name))
    unique, seen = [], set()
    for s in found:   # the same route published by two instances (a park and its département)
        key = (s.name.lower(), round(s.distance_km, 1))
        if key not in seen:
            seen.add(key)
            unique.append(s)
    return unique[:limit], failed


def fetch_gpx(url: str, session=None) -> bytes:
    """The GPX file of a route (only from the instances above)."""
    if not any(url.startswith(i.api.rsplit("/api/", 1)[0] + "/") for i in INSTANCES):
        raise ValueError("Not a Geotrek route of the known instances")
    session = session or make_session()
    try:
        resp = session.get(url, timeout=20)
    except OSError as exc:
        raise RuntimeError("The route's website is not answering: try again in a moment.") from exc
    if resp.status_code != 200:
        raise RuntimeError(f"Could not download the route ({resp.status_code})")
    if len(resp.content) > MAX_GPX_BYTES:
        raise ValueError("This route's GPX file is too large")
    return resp.content
