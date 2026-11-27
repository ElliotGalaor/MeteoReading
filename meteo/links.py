"""A track from a link: a Geotrek route page, an OpenStreetMap relation or trace, a GPS file on any site, or a web
page that offers one (#88).

Geotrek and OpenStreetMap links go to those sites' APIs. Any other HTTPS link goes through meteo.web_fetch, which
refuses private and local addresses (the server must not reach its own network) and limits the size.
"""
from __future__ import annotations

import gzip
import html
import re
from pathlib import PurePosixPath
from urllib.parse import urlencode, urljoin, urlparse

from .download import quick_session
from .geotrek import INSTANCES, fetch_route_file, practice_activity, practice_names
from .gps import Track, read_tracks
from .i18n import t as tr
from .osm_routes import relation_track
from .web_fetch import MAX_BYTES, Fetched, fetch

EXTENSIONS = (".gpx", ".kml", ".geojson", ".json")
TYPES = {"application/gpx+xml": ".gpx", "application/gpx": ".gpx", "application/vnd.google-earth.kml+xml": ".kml",
         "application/geo+json": ".geojson", "application/vnd.geo+json": ".geojson"}
HTML_TYPES = ("text/html", "application/xhtml+xml")
MAX_FILES = 5        # GPS files read from one page
# A GPS file named in a page: a link, an attribute or a script ("…/trace.gpx", data-gpx='…', src: "…kml?v=2")
_FILE_IN_PAGE = re.compile(r"""["']([^"'\s<>,*]+?\.(?:gpx|kml|geojson)(?:[?#][^"'\s<>]*)?)["']""", re.IGNORECASE)
_BLANK = b"\xef\xbb\xbf \t\r\n"   # a BOM and white space before a document
_TITLE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)


def _host(url: str) -> str:
    return (urlparse(url).hostname or "").lower()


def _geotrek_instance(host: str):
    for inst in INSTANCES:
        if host in (_host(inst.api), _host(inst.portal or "")):
            return inst
    return None


def _site(url: str) -> str:
    return _host(url).removeprefix("www.")


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


def tracks_from_link(url: str, session=None, fetcher=fetch) -> list[Track]:
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

    # Any other site: a GPS file, or a web page that offers one
    return _tracks_from_site(url, fetcher)


def is_html(f: Fetched) -> bool:
    head = f.data[:512].lstrip(_BLANK).lower()
    return f.content_type in HTML_TYPES or head.startswith((b"<!doctype html", b"<html"))


def track_file_name(f: Fetched) -> str | None:
    """A file name whose extension tells read_tracks how to read the answer, or None if it is not a GPS file.

    From the name the site gives (Content-Disposition), the address, the content type, or the first bytes: a
    download such as "admin-ajax.php?action=…" names its file only in its headers."""
    url_name = PurePosixPath(urlparse(f.url).path).name
    for name in (f.filename, url_name):
        if PurePosixPath(name).suffix.lower() in EXTENSIONS:
            return name
    stem = PurePosixPath(f.filename or url_name).stem or "track"
    if f.content_type in TYPES:
        return stem + TYPES[f.content_type]
    if is_html(f):
        return None
    head = f.data[:4096].lstrip(_BLANK).lower()
    if head.startswith(b"<") and b"<gpx" in head:
        return stem + ".gpx"
    if head.startswith(b"<") and b"<kml" in head:
        return stem + ".kml"
    if head.startswith(b"{") and b'"type"' in head:
        return stem + ".geojson"
    return None


def _wp_trip_summary(text: str) -> list[str]:
    """The track download of the WP Trip Summary plugin (WordPress): admin-ajax.php with the post's id and a nonce
    written in the page's script. The nonce expires within a day, hence a page read afresh each time."""
    def var(name: str, pattern: str = r"[^'\"]+") -> str | None:
        m = re.search(rf"\b{name}\s*=\s*['\"]?({pattern})", text)
        return m.group(1) if m else None

    if not re.search(r"\babp01_hasTrack\s*=\s*true", text):
        return []
    ajax, post, nonce = var("abp01_ajaxUrl"), var("abp01_postId", r"\d+"), var("abp01_nonceDownload", r"\w+")
    if not (ajax and post and nonce):
        return []
    action = var("abp01_downloadTrackAction", r"[\w-]+") or "abp01_download_track"
    return [f"{ajax}?{urlencode({'action': action, 'abp01_nonce_download': nonce, 'abp01_postId': post})}"]


def files_in_page(page: str, page_url: str) -> list[str]:
    """The GPS files a web page offers on its own site, in page order: links, attributes or scripts naming a .gpx,
    .kml or .geojson file, and the WP Trip Summary download."""
    text = page.replace("\\/", "/")                      # addresses inside JSON: https:\/\/…
    found = [html.unescape(m) for m in _FILE_IN_PAGE.findall(text)] + _wp_trip_summary(text)
    site, urls = _site(page_url), []
    for f in found:
        url = urljoin(page_url, f)
        if urlparse(url).scheme == "https" and _site(url) == site and url not in urls:
            urls.append(url)
    return urls


def _title(page: str) -> str:
    m = _TITLE.search(page[:200_000])
    return html.unescape(re.sub(r"\s+", " ", m.group(1))).strip()[:100] if m else ""


def _tracks_from_site(url: str, fetcher) -> list[Track]:
    first = fetcher(url)
    name = track_file_name(first)
    if name:
        return read_tracks(first.data, name)
    if not is_html(first):
        raise ValueError(tr("err.link_no_track"))
    page = first.data.decode("utf-8", errors="replace")
    title, budget, tracks = _title(page), MAX_BYTES - len(first.data), []
    for file_url in files_in_page(page, first.url)[:MAX_FILES]:
        if budget <= 0:
            break
        try:
            f = fetcher(file_url, max_bytes=budget)
            budget -= len(f.data)
            name = track_file_name(f)
            found = read_tracks(f.data, name) if name else []
        except (ValueError, RuntimeError):      # a dead link or a file that is not a track: try the next one
            continue
        for t in found:
            if title and t.name == PurePosixPath(name).stem:   # the file has no name of its own: the page's title
                t.name = title
        tracks += found
    if not tracks:
        raise ValueError(tr("err.link_no_track"))
    return tracks
