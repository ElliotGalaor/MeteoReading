"""Official Météo-France weather warnings (vigilance) on the route.

- The départements crossed come from the French government's geographic API (geo.api.gouv.fr, no key), at about
  fifteen points of the route.
- The current vigilance comes from Météo-France's public API (portail-api.meteofrance.fr, product "DPVigilance").
  It needs a free application key, created by the owner of the site and given to the app in the
  METEOFRANCE_API_KEY environment variable (a secret on Cloud Run, never in the code). Without it, the page only
  links to vigilance.meteofrance.fr.
- Vigilance covers today and tomorrow ("J" and "J1"). Since 1 September 2026 it can be issued for parts of a
  département: a warning whose zone code starts with the département's code (e.g. "2901", or "2910" for the coast)
  applies to the route as soon as the route crosses that département, the cautious reading.

The format parsed is the one documented for DPVigilance v1 (product > periods > timelaps > domain_ids); anything
else is reported as "could not be read" rather than guessed.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from .download import make_session
from .i18n import fmt_date
from .i18n import t as tr
from .route_weather import subset

VIGILANCE_API = "https://public-api.meteofrance.fr/public/DPVigilance/v1/cartevigilance/encours"
GEO_API = "https://geo.api.gouv.fr/communes"
OFFICIAL_PAGE = "https://vigilance.meteofrance.fr"
KEY_ENV = "METEOFRANCE_API_KEY"
POINTS = 15
COLOURS = {1: "green", 2: "yellow", 3: "orange", 4: "red"}   # named tr("vigilance.colour_…")
PHENOMENA = {1: "wind", 2: "rain and flooding", 3: "thunderstorms", 4: "floods", 5: "snow and ice", 6: "heatwave",
             7: "extreme cold", 8: "avalanches", 9: "waves and coastal flooding"}   # English; tr("vigilance.ph_1")…


def api_key() -> str | None:
    return os.environ.get(KEY_ENV) or None


def departments(points: pd.DataFrame, session=None) -> list[dict]:
    """The département at about fifteen points of the route: [{dist_km, code, name}] (points outside France left
    out)."""
    session = session or make_session()
    out = []
    for i in subset(len(points), POINTS):
        p = points.iloc[i]
        try:
            resp = session.get(GEO_API, timeout=10, params={"lat": f"{p['lat']:.5f}", "lon": f"{p['lon']:.5f}",
                                                           "fields": "nom,codeDepartement,departement", "format": "json"})
            hits = resp.json() if resp.status_code == 200 else []
        except (OSError, ValueError):
            hits = []
        if hits:
            dep = hits[0].get("departement") or {}
            out.append({"dist_km": float(p["dist_km"]), "code": hits[0].get("codeDepartement") or dep.get("code"),
                        "name": dep.get("nom") or hits[0].get("codeDepartement")})
    return out


def fetch(key: str, session=None) -> dict:
    session = session or make_session()
    try:
        resp = session.get(VIGILANCE_API, timeout=15, headers={"apikey": key, "accept": "application/json"})
    except OSError as exc:
        raise RuntimeError(tr("vigilance.err_down")) from exc
    if resp.status_code != 200:
        raise RuntimeError(tr("vigilance.err_status", status=resp.status_code))
    try:
        return resp.json()
    except ValueError as exc:
        raise RuntimeError(tr("vigilance.err_read")) from exc


def parse(payload: dict) -> pd.DataFrame:
    """One row per zone and period: domain, start, end (UTC), colour (1-4), phenomena [(id, colour)]."""
    try:
        periods = payload["product"]["periods"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError(tr("vigilance.err_format")) from exc
    rows = []
    for period in periods:
        start = pd.Timestamp(period.get("begin_validity_time"))
        end = pd.Timestamp(period.get("end_validity_time"))
        start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
        end = end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")
        domains = (period.get("timelaps") or {}).get("domain_ids") or []
        for d in domains:
            items = [(int(it.get("phenomenon_id", 0)), int(it.get("phenomenon_max_color_id", 1)))
                     for it in d.get("phenomenon_items") or []]
            rows.append({"domain": str(d.get("domain_id")), "start": start, "end": end,
                         "colour": int(d.get("max_color_id", 1)), "phenomena": items})
    return pd.DataFrame(rows, columns=["domain", "start", "end", "colour", "phenomena"])


def phenomenon(pid: int) -> str:
    """Name of a Météo-France phenomenon in the current language."""
    return tr(f"vigilance.ph_{pid}") if pid in PHENOMENA else tr("vigilance.ph_other")


def _applies(domain: str, code: str | None) -> bool:
    return bool(code) and (domain == code or (domain.startswith(code) and len(domain) > len(code)))


def at_points(points: pd.DataFrame, deps: list[dict], warnings: pd.DataFrame) -> pd.DataFrame:
    """Vigilance colour and wording at each point of the reading (dist_km, time), from the département of the
    nearest point looked up. Adds vigilance_color (1-4, 1 where there is none or none applies) and vigilance_text."""
    out = points.copy()
    out["vigilance_color"] = 1
    out["vigilance_text"] = ""
    if not deps or warnings.empty:
        return out
    dep_km = np.array([d["dist_km"] for d in deps])
    for i, (km, t) in enumerate(zip(out["dist_km"], pd.to_datetime(out["time"], utc=True), strict=True)):
        dep = deps[int(np.argmin(np.abs(dep_km - km)))]
        hits = warnings[warnings["domain"].map(lambda d, c=dep["code"]: _applies(d, c))
                        & (warnings["start"] <= t) & (warnings["end"] > t)]
        if hits.empty:
            continue
        top = hits.loc[hits["colour"].idxmax()]
        phen = sorted({phenomenon(p) for _, r in hits.iterrows() for p, c in r["phenomena"]
                       if c >= max(top["colour"], 2)})
        out.loc[i, "vigilance_color"] = int(top["colour"])
        if top["colour"] >= 2:
            end = pd.Timestamp(top["end"])
            out.loc[i, "vigilance_text"] = tr("vigilance.text", colour=tr(f"vigilance.colour_{int(top['colour'])}"),
                                             phenomena=", ".join(phen) or tr("vigilance.weather"), dep=dep["name"],
                                             code=dep["code"], until=f"{fmt_date(end, 'weekday')} {end:%H:%M}")
    return out


def covered_until(warnings: pd.DataFrame) -> pd.Timestamp | None:
    return None if warnings.empty else warnings["end"].max()
