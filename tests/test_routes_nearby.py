"""OpenStreetMap routes, links to routes and multi-day plans, without the network."""
from __future__ import annotations

import pandas as pd
import pytest

from meteo.gps import build_track, passage_times
from meteo.links import tracks_from_link
from meteo.osm_routes import mercator, nearby_relations, relation_track


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
        if url == "https://geotrek-admin.ecrins-parcnational.fr/api/v2/trek/995106/":
            return Resp({"name": "10 - Vallon de la Buffe",
                         "gpx": "https://geotrek-admin.ecrins-parcnational.fr/api/fr/treks/995106/a.gpx"})
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


def test_osm_and_open_data_links():
    assert tracks_from_link("https://www.openstreetmap.org/relation/7", session=LinkSession())[0].name == "Boucle du lac"
    t = tracks_from_link("https://static.data.gouv.fr/resources/x/boucle.gpx", session=LinkSession())
    assert len(t[0].points) == 5


@pytest.mark.parametrize(("url", "message"), [
    ("http://rando.ecrins-parcnational.fr/trek/1", "https://"),
    ("https://example.com/route.gpx", "not accepted"),
    ("https://static.data.gouv.fr/resources/x/data.csv", ".gpx, .kml or .geojson"),
    ("https://rando.ecrins-parcnational.fr/fr/accueil", "does not point to a route"),
    ("https://www.openstreetmap.org/node/1", "relation"),
])
def test_links_refused(url, message):
    with pytest.raises(ValueError, match=message):
        tracks_from_link(url, session=LinkSession())


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
