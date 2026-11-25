"""A page or a file from any HTTPS site: the address guard, redirects, limits (meteo/web_fetch.py), no network."""
from __future__ import annotations

import gzip
import socket

import pytest
from urllib3 import HTTPHeaderDict

from meteo.web_fetch import BlockedAddress, fetch, public_address

PUBLIC = "93.184.215.14"


def resolver(table: dict[str, list[str]]):
    def resolve(host, port, type=None):
        if host not in table:
            raise socket.gaierror("unknown host")
        return [(socket.AF_INET6 if ":" in ip else socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, port))
                for ip in table[host]]
    return resolve


class Resp:
    def __init__(self, status=200, body=b"", headers=None):
        self.status, self.body, self.headers = status, body, HTTPHeaderDict(headers or {})
        self.released = False

    def stream(self, size, decode_content=True):
        for i in range(0, len(self.body), size):
            yield self.body[i:i + size]

    def release_conn(self):
        self.released = True


class Opener:
    """Answers by host and path, and records where each request was sent (the address checked, the host)."""

    def __init__(self, answers: dict[tuple[str, str], Resp]):
        self.answers, self.calls = answers, []

    def __call__(self, ip, host, target):
        self.calls.append((ip, host, target))
        return self.answers.get((host, target.split("?")[0]), Resp(404))


@pytest.mark.parametrize("ip", ["127.0.0.1", "10.0.0.1", "172.16.5.4", "192.168.1.1", "169.254.169.254",
                                "100.64.0.1", "0.0.0.0", "224.0.0.1", "::1", "fd00::1", "fe80::1%eth0",
                                "::ffff:127.0.0.1"])
def test_private_and_local_addresses_are_refused(ip):
    with pytest.raises(BlockedAddress):
        public_address("site.example", resolver({"site.example": [ip]}))


def test_one_private_address_among_public_ones_is_enough_to_refuse():
    with pytest.raises(BlockedAddress):
        public_address("site.example", resolver({"site.example": [PUBLIC, "10.0.0.1"]}))
    assert public_address("site.example", resolver({"site.example": [PUBLIC, "2606:2800:21f:cb07::1"]})) == PUBLIC


@pytest.mark.parametrize("host", ["localhost", "metadata.google.internal", "printer.local", "nas.home.arpa"])
def test_local_names_are_refused_without_asking_the_dns(host):
    with pytest.raises(BlockedAddress):
        public_address(host, resolver({}))


def test_an_unknown_host_is_a_site_down():
    with pytest.raises(RuntimeError, match="not answering"):
        public_address("nowhere.example", resolver({}))


def test_the_request_goes_to_the_address_checked():
    opener = Opener({("site.example", "/a.gpx"): Resp(body=b"<gpx/>", headers={"Content-Type": "Application/GPX+XML; charset=utf-8"})})
    f = fetch("https://site.example/a.gpx?v=2", resolve=resolver({"site.example": [PUBLIC]}), opener=opener)
    assert opener.calls == [(PUBLIC, "site.example", "/a.gpx?v=2")]
    assert (f.data, f.content_type, f.url) == (b"<gpx/>", "application/gpx+xml", "https://site.example/a.gpx?v=2")


@pytest.mark.parametrize(("url", "error", "match"), [
    ("http://site.example/a.gpx", ValueError, "https://"),
    ("https://site.example:8443/a.gpx", BlockedAddress, "private or local"),
    ("https://10.0.0.1/a.gpx", BlockedAddress, "private or local"),
])
def test_links_refused_before_any_request(url, error, match):
    opener = Opener({})
    with pytest.raises(error, match=match):
        fetch(url, resolve=resolver({"site.example": [PUBLIC], "10.0.0.1": ["10.0.0.1"]}), opener=opener)
    assert opener.calls == []


def test_a_redirect_is_checked_again():
    opener = Opener({("site.example", "/go"): Resp(302, headers={"Location": "https://intranet.example/secret"})})
    resolve = resolver({"site.example": [PUBLIC], "intranet.example": ["10.1.2.3"]})
    with pytest.raises(BlockedAddress):
        fetch("https://site.example/go", resolve=resolve, opener=opener)
    assert [c[1] for c in opener.calls] == ["site.example"]          # the intranet was never asked


def test_redirects_are_followed_then_limited():
    opener = Opener({("site.example", "/old"): Resp(301, headers={"Location": "/new.gpx"}),
                     ("site.example", "/new.gpx"): Resp(body=b"<gpx/>"),
                     ("site.example", "/loop"): Resp(302, headers={"Location": "/loop"})})
    resolve = resolver({"site.example": [PUBLIC]})
    assert fetch("https://site.example/old", resolve=resolve, opener=opener).url == "https://site.example/new.gpx"
    with pytest.raises(RuntimeError, match="redirects too many times"):
        fetch("https://site.example/loop", resolve=resolve, opener=opener)


def test_size_limit_status_and_file_name():
    resolve = resolver({"site.example": [PUBLIC]})
    big = Opener({("site.example", "/big.gpx"): Resp(body=b"x" * 5000)})
    with pytest.raises(ValueError, match="20 MB"):
        fetch("https://site.example/big.gpx", max_bytes=1000, resolve=resolve, opener=big)
    with pytest.raises(RuntimeError, match="404"):
        fetch("https://site.example/missing", resolve=resolve, opener=Opener({}))
    named = Opener({("site.example", "/wp-admin/admin-ajax.php"): Resp(
        body=gzip.compress(b"<gpx/>"), headers={"Content-Disposition": 'attachment; filename="../track-29630.gpx"'})})
    f = fetch("https://site.example/wp-admin/admin-ajax.php?action=x", resolve=resolve, opener=named)
    assert (f.filename, f.data) == ("track-29630.gpx", b"<gpx/>")    # no path kept; a gzipped body unpacked


def test_a_gzip_bomb_stops_at_the_limit():
    bomb = Opener({("site.example", "/b.gpx.gz"): Resp(body=gzip.compress(b"\0" * 2_000_000))})
    with pytest.raises(ValueError, match="20 MB"):
        fetch("https://site.example/b.gpx.gz", max_bytes=100_000, resolve=resolver({"site.example": [PUBLIC]}),
              opener=bomb)
