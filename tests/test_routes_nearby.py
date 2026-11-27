"""OpenStreetMap routes, links to routes and multi-day plans, without the network."""
from __future__ import annotations

import pandas as pd
import pytest

from meteo.gps import build_track, passage_times
from meteo.links import track_file_name, tracks_from_link
from meteo.osm_routes import mercator, nearby_relations, relation_track
from meteo.web_fetch import Fetched


class Resp:
    def __init__(self, data=None, status=200, content=b""):
        self.data, self.status_code, self.content = data, status, content

    def json(self):
        return self.data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise OSError(f"HTTP {self.status_code}")

    def iter_content(self, size):
        yield self.content


def relation_json(rid: int = 7):
    """A relation of two ways listed out of order, the second one drawn backwards."""
    nodes = [{"type": "node", "id": i, "lat": 45.0, "lon": 6.0 + i * 0.002} for i in range(1, 11)]
    ways = [{"type": "way", "id": 1, "nodes": [10, 9, 8, 7, 6]}, {"type": "way", "id": 2, "nodes": [1, 2, 3, 4, 5, 6]}]
    rel = {"type": "relation", "id": rid, "tags": {"name": "Boucle du lac"},
           "members": [{"type": "way", "ref": 1, "role": ""}, {"type": "way", "ref": 2, "role": ""}]}
    return {"elements": nodes + ways + [rel]}


class OSMSession:
    def __init__(self):
        self.urls = []

    def get(self, url, params=None, timeout=None, stream=False):
        self.urls.append(url)
        if "waymarkedtrails" in url:
            site = url.split("//")[1].split(".")[0]
            if site == "mtb":
                raise ConnectionError("down")
            return Resp({"results": [
                {"type": "relation", "id": 7, "ref": "12", "name": "Boucle du lac", "group": "LOC"},
                {"type": "relation", "id": 8, "ref": "GR 4", "name": "De la Méditerranée à l'Océan", "group": "NAT"},
            ]})
        if url.endswith("/relation/7/full.json"):
            return Resp(relation_json())
        raise AssertionError(url)


def test_mercator():
    x, y = mercator(5.15, 44.08)
    assert round(x) == 573295 and abs(y - 5477831) < 2


def test_osm_routes_keep_local_ones_and_skip_a_site_that_fails():
    found, failed = nearby_relations([45.0, 45.01], [6.0, 6.01], radius_km=30, session=OSMSession())
    assert [(s.trek_id, s.name, s.provider) for s in found] == [(7, "12 · Boucle du lac", "OpenStreetMap")]
    assert found[0].page_url == "https://www.openstreetmap.org/relation/7" and found[0].distance_km is None
    assert failed == ["OpenStreetMap (mountain bike)"]


def test_relation_ways_are_chained():
    t = relation_track(7, session=OSMSession())
    assert t.name == "Boucle du lac" and len(t.points) == 10
    assert t.points["lon"].is_monotonic_increasing or t.points["lon"].is_monotonic_decreasing


# --- Links ------------------------------------------------------------------------------------------------------
GPX = (b'<gpx xmlns="http://www.topografix.com/GPX/1/1"><trk><name>x</name><trkseg>'
       + b"".join(f'<trkpt lat="45.0" lon="{6 + i * 0.001}"/>'.encode() for i in range(5)) + b"</trkseg></trk></gpx>")


class LinkSession(OSMSession):
    def get(self, url, params=None, timeout=None, stream=False):
        api = "https://geotrek-admin.ecrins-parcnational.fr/api/v2"
        if url == f"{api}/trek/995106/":     # a mountain-bike route of the Écrins
            return Resp({"name": "10 - Vallon de la Buffe", "practice": 2,
                         "gpx": f"{api.removesuffix('/v2')}/fr/treks/995106/a.gpx"})
        if url == f"{api}/trek/996754/":     # a walk whose name says nothing of it
            return Resp({"name": "Le vallon de Martignare", "practice": 4,
                         "gpx": f"{api.removesuffix('/v2')}/fr/treks/996754/a.gpx"})
        if url == f"{api}/trek_practice/":
            return Resp({"results": [{"id": 2, "name": "VTT"}, {"id": 4, "name": "A pied"}]})
        if url.endswith("/a.kml"):
            return Resp(status=404)                     # Geotrek's KML is tried first; here only the GPX exists
        if url.endswith("/a.gpx") or url.endswith("boucle.gpx"):
            return Resp(content=GPX)
        return super().get(url, params, timeout, stream)


@pytest.mark.parametrize("url", ["https://rando.ecrins-parcnational.fr/trek/995106-10-Vallon-de-la-Buffe",
                                 "https://geotrek-admin.ecrins-parcnational.fr/api/v2/trek/995106/"])
def test_geotrek_links(url):
    tracks = tracks_from_link(url, session=LinkSession())
    assert [t.name for t in tracks] == ["10 - Vallon de la Buffe"]


def test_a_geotrek_link_gives_the_practice_the_site_publishes():
    """The activity comes from the route's practice, not from a guess on its name (#75)."""
    bike = tracks_from_link("https://rando.ecrins-parcnational.fr/trek/995106", session=LinkSession())[0]
    walk = tracks_from_link("https://rando.ecrins-parcnational.fr/trek/996754", session=LinkSession())[0]
    assert (bike.activity, walk.activity) == ("Cycling", "Walking")
    assert tracks_from_link("https://www.openstreetmap.org/relation/7", session=LinkSession())[0].activity is None


class Site:
    """Any other site, as meteo.web_fetch would give it: answers by address, records the addresses asked."""

    def __init__(self, answers: dict[str, Fetched]):
        self.answers, self.calls = answers, []

    def __call__(self, url, max_bytes=None):
        self.calls.append(url)
        if url not in self.answers:
            raise RuntimeError("The site answered with an error (404)")
        return self.answers[url]


def page(url: str, body: str) -> Fetched:
    return Fetched(url, body.encode(), "text/html")


CLUB = "https://club.example/sentiers/trace-rouge/"
AJAX = ("https://club.example/wp-admin/admin-ajax.php?action=abp01_download_track&abp01_nonce_download=37866299fd"
        "&abp01_postId=29630")
# The script WP Trip Summary writes in a WordPress page (amisdesaintevictoire.asso.fr, #88)
WP_TRIP_SUMMARY = """<html><head><title>Tracé rouge &#8211; GPX et Carte</title></head><body><script>
var abp01_imgBase = 'https://club.example/wp-content/plugins/wp-trip-summary/media/img';
var abp01_ajaxUrl = 'https://club.example/wp-admin/admin-ajax.php';
var abp01_ajaxGetTrackAction = 'abp01_get_track';
var abp01_downloadTrackAction = 'abp01_download_track';
var abp01_hasInfo = true;
var abp01_hasTrack = true;
var abp01_postId = '29630';
var abp01_nonceGet = 'a4eb0b6176';
var abp01_nonceDownload = '37866299fd';
</script></body></html>"""


def test_osm_and_open_data_links():
    assert tracks_from_link("https://www.openstreetmap.org/relation/7", session=LinkSession())[0].name == "Boucle du lac"
    url = "https://static.data.gouv.fr/resources/x/boucle.gpx"
    t = tracks_from_link(url, fetcher=Site({url: Fetched(url, GPX, "application/octet-stream")}))
    assert len(t[0].points) == 5


def test_a_wordpress_page_with_wp_trip_summary_gives_its_track():
    site = Site({CLUB: page(CLUB, WP_TRIP_SUMMARY),
                 AJAX: Fetched(AJAX, GPX.replace(b"<name>x</name>", b""), "application/gpx", "track-29630.gpx")})
    tracks = tracks_from_link(CLUB, fetcher=site)
    assert site.calls == [CLUB, AJAX]
    assert [t.name for t in tracks] == ["Tracé rouge – GPX et Carte"]   # the file has no name: the page's title
    assert len(tracks[0].points) == 5


def test_a_page_links_its_files_on_its_own_site_only():
    body = """<a href="/files/boucle.gpx">GPX</a> <a href="https://other.example/x.gpx">elsewhere</a>
              <script>var t = {"kml": "https:\\/\\/www.club.example\\/f\\/loop.kml?v=2"};</script>
              <a href="/files/dead.gpx">old</a> <input accept=".gpx,.kml"> <a href="/files/boucle.gpx">again</a>"""
    gpx = "https://club.example/files/boucle.gpx"
    site = Site({CLUB: page(CLUB, body), gpx: Fetched(gpx, GPX, "application/octet-stream")})
    tracks = tracks_from_link(CLUB, fetcher=site)
    assert site.calls == [CLUB, gpx, "https://www.club.example/f/loop.kml?v=2", "https://club.example/files/dead.gpx"]
    assert [t.name for t in tracks] == ["x"]          # the dead link and the missing KML are skipped


def test_the_file_type_comes_from_the_headers_or_the_first_bytes():
    def name(**kw):
        return track_file_name(Fetched(kw.pop("url", "https://s.example/download"), kw.pop("data", b"x"), **kw))

    assert name(content_type="", filename="Boucle.GPX") == "Boucle.GPX"
    assert name(url="https://s.example/a/route.kml", content_type="text/plain") == "route.kml"
    assert name(content_type="application/vnd.google-earth.kml+xml") == "download.kml"
    assert name(content_type="application/octet-stream", data=b"\xef\xbb\xbf <?xml version='1.0'?><gpx>") == "download.gpx"
    assert name(content_type="", data=b'{"type": "FeatureCollection", "features": []}') == "download.geojson"
    assert name(content_type="text/html", data=b"<gpx>") is None                        # a page, not a file
    assert name(content_type="text/csv", data=b"lat,lon") is None


@pytest.mark.parametrize(("url", "answers", "message"), [
    ("http://rando.ecrins-parcnational.fr/trek/1", {}, "https://"),
    ("https://rando.ecrins-parcnational.fr/fr/accueil", {}, "does not point to a route"),
    ("https://www.openstreetmap.org/node/1", {}, "relation"),
    (CLUB, {CLUB: page(CLUB, "<html><p>No map here</p></html>")}, "No GPS track found"),
    ("https://s.example/data.csv", {"https://s.example/data.csv": Fetched("https://s.example/data.csv", b"a,b",
                                                                          "text/csv")}, "No GPS track found"),
])
def test_links_refused(url, answers, message):
    with pytest.raises(ValueError, match=message):
        tracks_from_link(url, session=LinkSession(), fetcher=Site(answers))


# --- Multi-day plans ---------------------------------------------------------------------------------------------
def test_multi_day_plan_goes_on_the_next_day_at_the_same_time():
    rows = [(45.0, 6.0 + i * 0.01, None, None) for i in range(200)]       # about 157 km, flat
    track = build_track(rows)
    start = pd.Timestamp("2026-07-01 08:00", tz="Europe/Paris")
    one_go = passage_times(track, start, speed_kmh=20)
    staged = passage_times(track, start, speed_kmh=20, hours_per_day=4)
    assert one_go.iloc[-1] - one_go.iloc[0] == pd.Timedelta(hours=track.distance_km / 20)
    local = staged.dt.tz_convert("Europe/Paris")
    assert local.dt.date.nunique() == 2                                  # 7.9 h on the move: 2 days of 4 h
    day2 = local[local.dt.date == local.dt.date.iloc[-1]]
    assert day2.iloc[0].strftime("%H:%M") == "08:00" or day2.iloc[0].hour == 8
    assert (local.dt.hour.between(8, 12)).all()                          # never on the move after noon
