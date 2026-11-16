"""Routes near a track from the Geotrek APIs (meteo/geotrek.py), without the network."""
from __future__ import annotations

import pytest

from meteo.geotrek import Instance, fetch_gpx, nearby_treks, overlaps, search_box

ALPS = Instance("Parc des Alpes", "Alpes", "https://alps.example/api/v2", "https://rando.alps.example", (6.0, 44.5, 7.0, 45.5))
SEA = Instance("Parc de la mer", "Mer", "https://sea.example/api/v2", None, (-3.0, 47.0, -2.0, 48.0))
DOWN = Instance("Parc injoignable", "Injoignable", "https://down.example/api/v2", None, (6.0, 44.5, 7.0, 45.5))


class Resp:
    def __init__(self, data, status=200, content=b""):
        self.data, self.status_code, self.content = data, status, content

    def json(self):
        return self.data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise OSError(f"HTTP {self.status_code}")


class FakeSession:
    """Two treks near (6.5, 45.0), one far away, one published twice; the "down" instance does not answer."""

    def __init__(self):
        self.urls = []

    def get(self, url, params=None, timeout=None):
        self.urls.append(url)
        if url.startswith("https://down.example"):
            raise ConnectionError("no answer")
        if url.endswith("/trek_practice/"):
            return Resp({"results": [{"id": 1, "name": "À pied"}, {"id": 2, "name": "VTT"}]})
        if url.endswith("/trek/"):
            assert params["in_bbox"].count(",") == 3
            return Resp({"results": [
                {"id": 10, "name": "Lac d'en haut", "departure_geom": [6.51, 45.0], "length_2d": 8200, "ascent": 450,
                 "duration": 3.0, "practice": 1, "gpx": "https://alps.example/api/fr/treks/10/lac.gpx"},
                {"id": 11, "name": "Col en VTT", "departure_geom": [6.55, 45.02], "length_2d": 25000, "ascent": 900,
                 "duration": 2.5, "practice": 2, "gpx": "https://alps.example/api/fr/treks/11/col.gpx"},
                {"id": 12, "name": "Loin d'ici", "departure_geom": [6.95, 45.45], "length_2d": 5000, "ascent": 100,
                 "duration": 1.0, "practice": 1, "gpx": "https://alps.example/api/fr/treks/12/loin.gpx"},
                {"id": 13, "name": "Sans GPX", "departure_geom": [6.5, 45.0], "practice": 1, "gpx": None},
            ]})
        raise AssertionError(url)


def test_search_box_and_overlap():
    box = search_box([45.0, 45.1], [6.5, 6.6], radius_km=11.1)
    assert box[1] == pytest.approx(44.9, abs=1e-3) and box[3] == pytest.approx(45.2, abs=1e-3)
    assert overlaps(box, ALPS.bbox) and not overlaps(box, SEA.bbox)


def test_nearby_treks_nearest_first_within_the_radius():
    session = FakeSession()
    found, failed = nearby_treks([45.0, 45.0], [6.50, 6.52], radius_km=10, session=session,
                                 instances=[ALPS, SEA, DOWN])
    assert [s.trek_id for s in found] == [10, 11]                  # 12 starts too far, 13 has no GPX
    assert found[0].distance_km < found[1].distance_km < 10
    assert found[0].practice == "À pied" and found[0].activity == "Walking"
    assert found[1].practice == "VTT" and found[1].activity == "Cycling"
    assert found[0].page_url == "https://rando.alps.example/trek/10" and found[0].length_km == pytest.approx(8.2)
    assert failed == ["Parc injoignable"]                          # skipped, not an error
    assert not any(u.startswith("https://sea.example") for u in session.urls)   # far instance not queried


def test_a_route_published_twice_is_listed_once():
    twin = Instance("Département des Alpes", "Dép. Alpes", "https://alps.example/api/v2", None, ALPS.bbox)
    found, _ = nearby_treks([45.0], [6.50], radius_km=10, session=FakeSession(), instances=[ALPS, twin])
    assert [s.trek_id for s in found] == [10, 11]


def test_no_instance_near_the_track():
    session = FakeSession()
    assert nearby_treks([48.85], [2.35], session=session, instances=[ALPS, SEA]) == ([], [])
    assert session.urls == []


def test_gpx_only_from_known_instances():
    with pytest.raises(ValueError, match="Not a Geotrek route"):
        fetch_gpx("https://evil.example/route.gpx")

    class Big:
        def get(self, url, timeout=None):
            return Resp(None, content=b"x" * 6_000_000)
    with pytest.raises(ValueError, match="too large"):
        fetch_gpx("https://geotrek-admin.ecrins-parcnational.fr/api/fr/treks/1/a.gpx", session=Big())
