"""A track from a link: a Geotrek route page, an OpenStreetMap relation or trace, or a GPS file on an open-data site.

Only known sites are fetched (no request to an arbitrary address from the server), over HTTPS, with a size limit.
"""
from __future__ import annotations

import gzip
import re
from urllib.parse import urlparse

from .download import quick_session
from .geotrek import INSTANCES, fetch_route_file, practice_activity, practice_names
from .gps import Track, read_tracks
from .i18n import t as tr
from .osm_routes import relation_track

MAX_BYTES = 20 * 1024 * 1024
FILE_HOSTS = ("static.data.gouv.fr", "www.data.gouv.fr", "data.ecrins-parcnational.fr", "eurovelo.com")
EXTENSIONS = (".gpx", ".kml", ".geojson", ".json")


def _host(url: str) -> str:
    return (urlparse(url).hostname or "").lower()


def _geotrek_instance(host: str):
    for inst in INSTANCES:
        if host in (_host(inst.api), _host(inst.portal or "")):
            return inst
    return None


def _download(url: str, session) -> bytes:
    try:
        resp = session.get(url, timeout=30, stream=True)
    except OSError as exc:
        raise RuntimeError(tr("err.site_down")) from exc
    if resp.status_code != 200:
        raise RuntimeError(tr("err.site_status", status=resp.status_code))
    data = b""
    for chunk in resp.iter_content(1 << 16):
        data += chunk
        if len(data) > MAX_BYTES:
            raise ValueError(tr("err.file_20mb"))
    return gzip.decompress(data) if data[:2] == b"\x1f\x8b" else data


def tracks_from_link(url: str, session=None) -> list[Track]:
    """The tracks behind a link (see the module docstring for what is accepted)."""
    url = url.strip()
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.hostname:
        raise ValueError(tr("err.https"))
    host, path = _host(url), parsed.path
    session = session or quick_session()

    # Geotrek: a route page (/trek/<id>…), the API (/api/v2/trek/<id>/) or a GPX (/api/<lang>/treks/<id>/…)
    inst = _geotrek_instance(host)
    if inst is not None:
        m = re.search(r"/treks?/(\d+)", path)
        if not m:
            raise ValueError(tr("err.geotrek_link"))
        resp = session.get(f"{inst.api}/trek/{m.group(1)}/", params={"fields": "name,gpx,practice", "language": "fr"},
                           timeout=15)
        if resp.status_code != 200 or not resp.json().get("gpx"):
            raise RuntimeError(tr("err.geotrek_none", site=inst.name, id=m.group(1)))
        info = resp.json()
        tracks = read_tracks(*fetch_route_file(info["gpx"], session=session))
        # The practice published by the site ("À pied", "VTT"…), not a guess from the name (#75)
        practice = practice_names(session, inst).get(info.get("practice"), "") if info.get("practice") else ""
        for t in tracks:
            t.name = info.get("name") or t.name
            t.activity = practice_activity(practice)
        return tracks

    # OpenStreetMap: a route relation, or a public GPS trace
    if host in ("www.openstreetmap.org", "openstreetmap.org"):
        if m := re.match(r"/relation/(\d+)", path):
            return [relation_track(int(m.group(1)), session=session)]
        if m := re.search(r"/traces?/(\d+)", path):
            data = _download(f"https://www.openstreetmap.org/trace/{m.group(1)}/data", session)
            return read_tracks(data, f"trace-{m.group(1)}.gpx")
        raise ValueError(tr("err.osm_link"))

    # A GPS file on an open-data site
    if any(host == h or host.endswith("." + h) for h in FILE_HOSTS):
        name = path.rsplit("/", 1)[-1] or "track"
        if not name.lower().endswith(EXTENSIONS):
            raise ValueError(tr("err.link_ext"))
        return read_tracks(_download(url, session), name)

    known = ", ".join(sorted({_host(i.portal) for i in INSTANCES if i.portal}))
    raise ValueError(tr("err.link_site", geotrek=known, hosts=", ".join(FILE_HOSTS)))
