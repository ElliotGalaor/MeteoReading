"""Extraction : découverte et téléchargement massif des fichiers Météo-France.

Les URLs ne sont pas codées en dur : elles sont découvertes via l'API data.gouv.fr,
ce qui rend le pipeline robuste aux mises à jour du jeu de données.

Points clés :
- téléchargements parallèles (ThreadPoolExecutor) ;
- reprise sur erreur réseau (retry exponentiel) ;
- idempotence : un fichier déjà présent avec la bonne taille n'est pas re-téléchargé,
  un fichier mis à jour côté serveur (taille différente) l'est ;
- traçabilité : un manifeste JSON garde l'origine, la taille et l'empreinte de chaque fichier.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

log = logging.getLogger(__name__)

# Ex. : .../BASE/QUOT/Q_13_previous-1950-2024_RR-T-Vent.csv.gz
RESOURCE_PATTERN = re.compile(
    r"/Q_(?P<dep>[0-9]{2,3}|2A|2B)_(?P<period>previous|latest|avant)-[0-9-]*_RR-T-Vent\.csv\.gz$"
)


@dataclass
class Resource:
    dep: str
    period: str
    url: str
    filesize: int | None

    @property
    def filename(self) -> str:
        return self.url.rsplit("/", 1)[-1]


def make_session() -> requests.Session:
    """Session HTTP avec retry exponentiel sur les erreurs transitoires."""
    retry = Retry(
        total=5,
        backoff_factor=1.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET", "HEAD"),
    )
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry, pool_maxsize=16))
    session.headers["User-Agent"] = "meteo-analytics/1.0 (projet portfolio data analyst)"
    return session


def discover_resources(
    payload: dict, departments: list[str] | str, periods: list[str]
) -> list[Resource]:
    """Filtre la liste des ressources data.gouv.fr (réponse JSON de l'API)."""
    found: dict[tuple[str, str], Resource] = {}
    for res in payload.get("resources", []):
        m = RESOURCE_PATTERN.search(res.get("url", ""))
        if not m:
            continue
        dep, period = m["dep"], m["period"]
        if departments != "all" and dep not in departments:
            continue
        if period not in periods:
            continue
        found[(dep, period)] = Resource(dep, period, res["url"], res.get("filesize"))
    return sorted(found.values(), key=lambda r: (r.dep, r.period))


def fetch_catalog(session: requests.Session, api_url: str) -> dict:
    log.info("Interrogation de l'API data.gouv.fr…")
    resp = session.get(api_url, timeout=60)
    resp.raise_for_status()
    return resp.json()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_one(session: requests.Session, res: Resource, dest_dir: Path) -> dict:
    """Télécharge un fichier (écriture atomique via un .part) et renvoie son entrée de manifeste."""
    dest = dest_dir / res.filename
    status = "cached"
    if not (dest.exists() and res.filesize and dest.stat().st_size == res.filesize):
        status = "downloaded"
        tmp = dest.with_suffix(dest.suffix + ".part")
        with session.get(res.url, stream=True, timeout=120) as resp:
            resp.raise_for_status()
            with open(tmp, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
        tmp.replace(dest)
    return {
        **asdict(res),
        "file": dest.name,
        "bytes": dest.stat().st_size,
        "sha256": _sha256(dest),
        "status": status,
        "retrieved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def download_all(cfg) -> list[dict]:
    """Point d'entrée de l'étape d'extraction."""
    dest_dir = cfg.raw_dir / "meteo_france"
    dest_dir.mkdir(parents=True, exist_ok=True)
    session = make_session()

    resources = discover_resources(fetch_catalog(session, cfg.dataset_api), cfg.departments, cfg.periods)
    if not resources:
        raise RuntimeError("Aucun fichier trouvé : vérifier les départements et périodes de config.yaml")
    total_mb = sum(r.filesize or 0 for r in resources) / 1e6
    log.info("%d fichiers à synchroniser (%.0f Mo)", len(resources), total_mb)

    manifest, errors = [], []
    with ThreadPoolExecutor(max_workers=cfg.workers) as pool:
        futures = {pool.submit(download_one, session, r, dest_dir): r for r in resources}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Téléchargement", unit="fichier"):
            res = futures[fut]
            try:
                manifest.append(fut.result())
            except Exception as exc:  # on continue : un fichier en échec ne bloque pas les autres
                log.error("Échec %s : %s", res.filename, exc)
                errors.append(res.filename)

    manifest.sort(key=lambda m: m["file"])
    with open(cfg.raw_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    n_new = sum(m["status"] == "downloaded" for m in manifest)
    log.info("%d téléchargés, %d déjà à jour, %d en échec", n_new, len(manifest) - n_new, len(errors))
    if errors:
        raise RuntimeError(f"{len(errors)} fichier(s) en échec : relancer la commande pour reprendre")
    return manifest
