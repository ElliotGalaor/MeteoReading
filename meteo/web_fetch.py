"""A page or a file from any HTTPS site, for a link a visitor pastes (#88).

The server must not become a way into its own network (server-side request forgery):
- every address of the host must be public, so not 127.0.0.1, 10.x, the Cloud Run metadata server
  (169.254.169.254, metadata.google.internal) or an IPv6 local address;
- the connection goes to the address that was checked, so a second DNS answer cannot swap it (DNS rebinding),
  with the certificate still checked against the host's name;
- HTTPS on port 443 only, redirects followed one by one and checked again, a size limit and timeouts.
"""
from __future__ import annotations

import ipaddress
import socket
import zlib
from collections.abc import Callable
from dataclasses import dataclass
from email.message import Message
from pathlib import PurePosixPath
from urllib.parse import urljoin, urlparse

import certifi
import urllib3

from .download import USER_AGENT
from .i18n import t as tr

MAX_BYTES = 20 * 1024 * 1024
MAX_REDIRECTS = 5
TIMEOUT = urllib3.Timeout(connect=10, read=30)
REDIRECTS = (301, 302, 303, 307, 308)
LOCAL_NAMES = ("localhost", ".localhost", ".local", ".internal", ".lan", ".home.arpa")

Resolver = Callable[..., list]
Opener = Callable[[str, str, str], "urllib3.BaseHTTPResponse"]


class BlockedAddress(ValueError):
    """The link points to a private or local address."""

    def __init__(self) -> None:
        super().__init__(tr("err.link_blocked"))


@dataclass
class Fetched:
    url: str            # after redirects
    data: bytes
    content_type: str   # lower case, without its parameters
    filename: str = ""  # from Content-Disposition, if the site gives one


def public_address(host: str, resolve: Resolver = socket.getaddrinfo) -> str:
    """The address to connect to, if every address of the host is public; else BlockedAddress."""
    host = host.lower().rstrip(".")
    if host == LOCAL_NAMES[0] or host.endswith(LOCAL_NAMES[1:]):
        raise BlockedAddress
    try:
        infos = resolve(host, 443, type=socket.SOCK_STREAM)
    except (socket.gaierror, UnicodeError) as exc:
        raise RuntimeError(tr("err.site_down")) from exc
    addresses = []
    for info in infos:
        ip = ipaddress.ip_address(info[4][0].split("%")[0])
        if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
            ip = ip.ipv4_mapped
        addresses.append(ip)
    if not addresses or any(not ip.is_global or ip.is_multicast for ip in addresses):
        raise BlockedAddress
    return str(addresses[0])


def _open(ip: str, host: str, target: str):
    """GET target from host, connected to ip; the certificate is checked against host."""
    pool = urllib3.HTTPSConnectionPool(ip, 443, server_hostname=host, assert_hostname=host, cert_reqs="CERT_REQUIRED",
                                       ca_certs=certifi.where(), timeout=TIMEOUT, retries=False, maxsize=1)
    headers = {"Host": host, "User-Agent": USER_AGENT, "Accept-Encoding": "gzip, deflate",
               "Accept": "text/html,application/gpx+xml,application/vnd.google-earth.kml+xml,application/geo+json,"
                         "application/json;q=0.9,*/*;q=0.8"}
    try:
        return pool.urlopen("GET", target, headers=headers, redirect=False, retries=False, preload_content=False)
    except urllib3.exceptions.HTTPError as exc:
        raise RuntimeError(tr("err.site_down")) from exc


def _read(resp, limit: int) -> bytes:
    data = bytearray()
    try:
        for chunk in resp.stream(1 << 16, decode_content=True):
            data += chunk
            if len(data) > limit:
                raise ValueError(tr("err.file_20mb"))
    except urllib3.exceptions.HTTPError as exc:
        raise RuntimeError(tr("err.site_down")) from exc
    finally:
        resp.release_conn()
    if data[:2] == b"\x1f\x8b":                     # a gzipped file (.gpx.gz), unpacked within the same limit
        unpack = zlib.decompressobj(16 + zlib.MAX_WBITS)
        try:
            data = unpack.decompress(bytes(data), limit + 1)
        except zlib.error as exc:
            raise ValueError(tr("err.file_type", ext=".gz")) from exc
        if len(data) > limit:
            raise ValueError(tr("err.file_20mb"))
    return bytes(data)


def _filename(disposition: str | None) -> str:
    if not disposition:
        return ""
    msg = Message()
    msg["content-disposition"] = disposition
    return PurePosixPath((msg.get_filename() or "").replace("\\", "/")).name


def fetch(url: str, max_bytes: int = MAX_BYTES, resolve: Resolver = socket.getaddrinfo,
          opener: Opener = _open) -> Fetched:
    """The page or file at url (see the module docstring for the checks)."""
    for _ in range(MAX_REDIRECTS + 1):
        parsed = urlparse(url)
        if parsed.scheme != "https" or not parsed.hostname:
            raise ValueError(tr("err.https"))
        try:
            port = parsed.port
            host = parsed.hostname.encode("idna").decode("ascii")
        except (ValueError, UnicodeError) as exc:
            raise ValueError(tr("err.https")) from exc
        if port not in (None, 443):
            raise BlockedAddress
        ip = public_address(host, resolve)
        target = (parsed.path or "/") + (f"?{parsed.query}" if parsed.query else "")
        resp = opener(ip, host, target)
        if resp.status in REDIRECTS and resp.headers.get("Location"):
            resp.release_conn()
            url = urljoin(url, resp.headers["Location"])
            continue
        if resp.status != 200:
            resp.release_conn()
            raise RuntimeError(tr("err.site_status", status=resp.status))
        kind = (resp.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        name = _filename(resp.headers.get("Content-Disposition"))
        return Fetched(url, _read(resp, max_bytes), kind, name)
    raise RuntimeError(tr("err.link_redirects"))
