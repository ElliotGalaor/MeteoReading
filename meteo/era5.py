"""Validation croisée : observations Météo-France vs réanalyse ERA5 (API Open-Meteo).

Pourquoi : vérifier la cohérence de nos séries avec une source indépendante, et mesurer
les limites d'un produit maillé (~25 km) souvent utilisé faute de stations.

L'API gratuite est limitée (600 appels/min, 5 000/h, 10 000/jour) et compte les appels
au prorata du volume demandé : (nb variables / 10) x (nb jours / 14). Le client
calcule ce poids, s'auto-limite par minute et met les réponses en cache.
"""
from __future__ import annotations

import json
import logging
import time
from collections import deque
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config
from .download import make_session
from .warehouse import connect

log = logging.getLogger(__name__)
VARIABLES = ["temperature_2m_max", "temperature_2m_min"]


class RateLimiter:
    """Fenêtre glissante d'une minute sur le poids des appels."""

    def __init__(self, max_per_minute: float):
        self.max = max_per_minute
        self.calls: deque[tuple[float, float]] = deque()

    def wait(self, weight: float) -> None:
        while True:
            now = time.monotonic()
            while self.calls and now - self.calls[0][0] > 60:
                self.calls.popleft()
            used = sum(w for _, w in self.calls)
            if used + weight <= self.max or not self.calls:
                self.calls.append((now, weight))
                return
            pause = 60 - (now - self.calls[0][0]) + 0.5
            log.info("  quota/minute atteint (%.0f appels) : pause de %.0f s", used, pause)
            time.sleep(pause)


def call_weight(n_variables: int, start: str, end: str) -> float:
    n_days = (date.fromisoformat(end) - date.fromisoformat(start)).days + 1
    return max(n_variables / 10, 0.1) * max(n_days / 14, 1)


def fetch_city(session, cfg: Config, city: str, lat: float, lon: float, limiter: RateLimiter, cache: Path) -> pd.DataFrame:
    e = cfg.era5
    path = cache / f"{city}_{e['start']}_{e['end']}.json".replace(" ", "_")
    if not path.exists():
        params = {"latitude": lat, "longitude": lon, "start_date": e["start"], "end_date": e["end"],
                  "daily": ",".join(VARIABLES), "timezone": "Europe/Paris", "models": "era5"}
        limiter.wait(call_weight(len(VARIABLES), e["start"], e["end"]))
        for attempt in range(4):
            resp = session.get(e["api"], params=params, timeout=120)
            if resp.status_code == 429:
                reason = resp.json().get("reason", "")
                if "Minutely" not in reason:
                    raise RuntimeError(f"Quota Open-Meteo dépassé : {reason}. Relancer plus tard (le cache est conservé).")
                log.warning("  429 (quota minute), nouvel essai dans 60 s")
                time.sleep(60)
                continue
            resp.raise_for_status()
            path.write_text(resp.text, encoding="utf-8")
            break
        else:
            raise RuntimeError(f"Échec Open-Meteo pour {city}")
    daily = json.loads(path.read_text(encoding="utf-8"))["daily"]
    return pd.DataFrame({"date": pd.to_datetime(daily["time"]),
                         "tx_era5": daily["temperature_2m_max"], "tn_era5": daily["temperature_2m_min"]})


def compare(obs: pd.DataFrame, era: pd.DataFrame, hot_threshold: float) -> dict:
    d = obs.merge(era, on="date").dropna()
    err_tx, err_tn = d["tx_era5"] - d["tx"], d["tn_era5"] - d["tn"]
    hot = d["tx"] >= hot_threshold
    return {
        "n_days": len(d),
        "bias_tx": err_tx.mean(), "bias_tn": err_tn.mean(),
        "rmse_tx": float(np.sqrt((err_tx**2).mean())), "rmse_tn": float(np.sqrt((err_tn**2).mean())),
        "corr_tx": d["tx"].corr(d["tx_era5"]), "corr_tn": d["tn"].corr(d["tn_era5"]),
        "bias_tx_hot_days": err_tx[hot].mean() if hot.any() else np.nan,
        "hot_days_obs": int(hot.sum()), "hot_days_era5": int((d["tx_era5"] >= hot_threshold).sum()),
    }


def run_era5_validation(cfg: Config) -> pd.DataFrame:
    e = cfg.era5
    cache = cfg.raw_dir / "era5"
    cache.mkdir(parents=True, exist_ok=True)
    with connect(cfg) as con:
        stations = con.sql("SELECT city, station_name, lat, lon FROM ref_station WHERE priority = 1 ORDER BY city").df()
        obs = con.sql(f"""SELECT city, date, tx, tn FROM city_daily
                          WHERE date BETWEEN DATE '{e['start']}' AND DATE '{e['end']}'""").df()
    obs["date"] = pd.to_datetime(obs["date"])

    session, limiter = make_session(), RateLimiter(e.get("max_weighted_calls_per_minute", 400))
    total = call_weight(len(VARIABLES), e["start"], e["end"]) * len(stations)
    log.info("%d villes, poids total ≈ %.0f appels API (quota gratuit : 10 000/jour)", len(stations), total)
    rows = []
    for s in stations.itertuples():
        era = fetch_city(session, cfg, s.city, s.lat, s.lon, limiter, cache)
        rows.append({"city": s.city, "station": s.station_name,
                     **compare(obs[obs["city"] == s.city], era, cfg.thresholds["hot_day"])})
        log.info("  %-17s biais TX %+.2f °C, TN %+.2f °C", s.city, rows[-1]["bias_tx"], rows[-1]["bias_tn"])

    res = pd.DataFrame(rows).round(3)
    out = cfg.reports_dir / "tables"
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "era5_validation.csv", index=False)
    return res
