"""Each city in a warmer France: the official French warming levels (TRACC) applied to its observed days.

Method (a "delta" method with model patterns), computed when the data is built:
1. Warming levels. The TRACC (trajectoire de réchauffement de référence pour l'adaptation au changement climatique)
   plans for +2 °C in 2030, +2.7 °C in 2050 and +4 °C in 2100 in mainland France, compared with the pre-industrial
   climate. France had warmed by about +1.7 °C by 2020 (Météo-France): the levels are +0.3, +1.0 and +2.3 °C beyond
   the recent period.
2. How that warming splits between seasons, days and nights, and how rainfall changes, comes from seven
   high-resolution CMIP6 models (HighResMIP, through the Open-Meteo Climate API) at four points of France. In each
   model, the change of each season's mean maximum and minimum between 1995-2006 and 2039-2050 is divided by the
   change of the annual mean temperature (°C per °C of warming), and so is the change of each season's rainfall
   (fraction per °C). `python -m meteo climate` downloads them once, into reports/tables/climate_patterns.csv
   (versioned, like the ERA5 comparison).
3. The days each city measured in 2011-2025 are shifted by these changes times the warming still to come, model by
   model, and the indicators are counted again. The range across the seven models is kept, never a single number.

Limits: the day-to-day variability stays the observed one (heatwaves get hotter, not longer); the patterns come from
four points rather than from each city; the TRACC is a planning trajectory, not a forecast.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config
from .download import make_session
from .era5 import RateLimiter, call_weight
from .gps import haversine_km

log = logging.getLogger(__name__)

CLIMATE_API = "https://climate-api.open-meteo.com/v1/climate"
MODELS = ["CMCC_CM2_VHR4", "FGOALS_f3_H", "HiRAM_SIT_HR", "MRI_AGCM3_2_S", "EC_Earth3P_HR", "MPI_ESM1_2_XR",
          "NICAM16_8S"]
VARIABLES = ["temperature_2m_max", "temperature_2m_min", "precipitation_sum"]
REGIONS = {"North": (49.5, 3.0), "West": (47.5, -1.5), "East": (46.5, 5.5), "South": (43.5, 4.5)}
WINDOWS = (("1995-01-01", "2006-12-31"), ("2039-01-01", "2050-12-31"))
LEVEL_NOW = 1.7                 # °C above pre-industrial in mainland France around 2020 (Météo-France)
LEVELS = {"2030": 2.0, "2050": 2.7, "2100": 4.0}   # TRACC: °C above pre-industrial reached in that year
SEASON_OF_MONTH = {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM", 6: "JJA", 7: "JJA", 8: "JJA",
                   9: "SON", 10: "SON", 11: "SON"}
HOURLY_BUDGET = 4500          # weighted calls per hour, under the free 5,000
PATTERNS_FILE = "climate_patterns.csv"
RESULT_FILE = "climate_ahead.csv"


# --- Model patterns (one-off download) ----------------------------------------------------------------
def patterns_from_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Per model and season: °C of change of the mean maximum (d_tx) and minimum (d_tn) per °C of annual warming,
    and the fractional change of rainfall per °C (r_rr), between the two windows. `df` holds one model's daily
    series: date, tx, tn, rr, window (0 = past, 1 = future)."""
    df = df.assign(season=pd.to_datetime(df["date"]).dt.month.map(SEASON_OF_MONTH).to_numpy())
    annual = df.assign(t=(df["tx"] + df["tn"]) / 2).groupby("window")["t"].mean()
    warming = float(annual.loc[1] - annual.loc[0])
    s = df.groupby(["window", "season"])[["tx", "tn", "rr"]].mean()
    rows = []
    for season in ("DJF", "MAM", "JJA", "SON"):
        past, future = s.loc[(0, season)], s.loc[(1, season)]
        rows.append({"season": season, "warming": warming,
                     "d_tx": (future["tx"] - past["tx"]) / warming, "d_tn": (future["tn"] - past["tn"]) / warming,
                     "r_rr": (future["rr"] / past["rr"] - 1) / warming if past["rr"] > 0 else 0.0})
    return pd.DataFrame(rows)


def fetch_patterns(cfg: Config, session=None) -> pd.DataFrame:
    """Downloads the models' daily series at the four points and returns their patterns
    (region, model, season, warming, d_tx, d_tn, r_rr). About 5,300 weighted calls in all: the client spreads them
    out to stay within the free per-minute and per-hour quotas."""
    session = session or make_session()
    limiter = RateLimiter(cfg.era5.get("max_weighted_calls_per_minute", 400))
    sent: list[tuple[float, float]] = []   # calls of the last hour: the free quota is 5,000 an hour
    rows = []
    for region, (lat, lon) in REGIONS.items():
        frames = {}
        for w, (start, end) in enumerate(WINDOWS):
            weight = call_weight(len(VARIABLES) * len(MODELS), start, end)
            limiter.wait(weight)
            _wait_hour(sent, weight, HOURLY_BUDGET)
            log.info("  %s %s-%s (%.0f weighted calls)", region, start[:4], end[:4], weight)
            resp = session.get(CLIMATE_API, timeout=300, params={
                "latitude": lat, "longitude": lon, "start_date": start, "end_date": end,
                "models": ",".join(MODELS), "daily": ",".join(VARIABLES)})
            resp.raise_for_status()
            frames[w] = resp.json()["daily"]
        for model in MODELS:
            parts = []
            for w, daily in frames.items():
                parts.append(pd.DataFrame({
                    "date": daily["time"], "window": w,
                    "tx": pd.to_numeric(pd.Series(daily[f"temperature_2m_max_{model}"]), errors="coerce"),
                    "tn": pd.to_numeric(pd.Series(daily[f"temperature_2m_min_{model}"]), errors="coerce"),
                    "rr": pd.to_numeric(pd.Series(daily[f"precipitation_sum_{model}"]), errors="coerce")}))
            p = patterns_from_daily(pd.concat(parts, ignore_index=True).dropna())
            p.insert(0, "model", model)
            p.insert(0, "region", region)
            rows.append(p)
    return pd.concat(rows, ignore_index=True)


def _wait_hour(sent: list[tuple[float, float]], weight: float, budget: float, window: float = 3600) -> None:
    """Waits until `weight` more calls fit in the hourly budget."""
    while True:
        now = time.monotonic()
        sent[:] = [(t, w) for t, w in sent if now - t < window]
        if sum(w for _, w in sent) + weight <= budget or not sent:
            sent.append((now, weight))
            return
        pause = window - (now - sent[0][0]) + 1
        log.info("  hourly quota: pausing %.0f min", pause / 60)
        time.sleep(pause)


def run_patterns(cfg: Config) -> Path:
    path = cfg.reports_dir / "tables" / PATTERNS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    fetch_patterns(cfg).round(4).to_csv(path, index=False)
    log.info("Model patterns: %s", path)
    return path


# --- Projection of each city --------------------------------------------------------------------------
def nearest_region(lat: float, lon: float) -> str:
    return min(REGIONS, key=lambda r: float(haversine_km(lat, lon, *REGIONS[r])))


def indicators(daily: pd.DataFrame, th: dict) -> dict[str, float]:
    """Yearly means of the indicators over the days given (complete years only are passed in), by code (the
    dashboard names them in each language: t("ind.hot_days"), …)."""
    years = pd.to_datetime(daily["date"]).dt.year
    n_years = years.nunique()
    tx, tn, rr = daily["tx"], daily["tn"], daily["rr"]
    return {
        "tmean": float(((tx + tn) / 2).mean()),
        "hot_days": float((tx >= th["hot_day"]).sum() / n_years),
        "very_hot_days": float((tx >= th["very_hot_day"]).sum() / n_years),
        "tropical_nights": float((tn >= th["tropical_night"]).sum() / n_years),
        "frost_days": float((tn < th["frost_day"]).sum() / n_years),
        "rr": float(rr.sum() / n_years),
    }


UNITS = {"tmean": "°C", "rr": "mm"}   # the others are days per year


def shifted(daily: pd.DataFrame, patterns: pd.DataFrame, delta: float) -> pd.DataFrame:
    """The days with the warming of one model added: `delta` °C of annual warming, split by season."""
    p = patterns.set_index("season")
    season = pd.to_datetime(daily["date"]).dt.month.map(SEASON_OF_MONTH)
    out = daily.copy()
    out["tx"] = daily["tx"] + season.map(p["d_tx"]).to_numpy() * delta
    out["tn"] = daily["tn"] + season.map(p["d_tn"]).to_numpy() * delta
    out["rr"] = daily["rr"] * np.maximum(1 + season.map(p["r_rr"]).to_numpy() * delta, 0)
    return out


def project_city(daily: pd.DataFrame, valid_years: list[int], patterns: pd.DataFrame, th: dict,
                 old: tuple[int, int], recent: tuple[int, int]) -> pd.DataFrame:
    """Indicators of one city: observed in `old` and `recent`, then at each TRACC level (min, median and max of the
    models). Columns: indicator (code), period ("1961-1990", "2011-2025", then the TRACC year), low, median, high."""
    years = pd.to_datetime(daily["date"]).dt.year
    base = daily[years.isin(valid_years).to_numpy()]
    b_years = pd.to_datetime(base["date"]).dt.year
    rows = []
    for label, (y0, y1) in ((f"{old[0]}-{old[1]}", old), (f"{recent[0]}-{recent[1]}", recent)):
        for name, v in indicators(base[b_years.between(y0, y1).to_numpy()], th).items():
            rows.append({"indicator": name, "period": label, "low": v, "median": v, "high": v})
    now = base[b_years.between(*recent).to_numpy()]
    for label, level in LEVELS.items():
        per_model = [indicators(shifted(now, patterns[patterns["model"] == m], level - LEVEL_NOW), th)
                     for m in patterns["model"].unique()]
        for name in per_model[0]:
            vals = np.array([pm[name] for pm in per_model])
            rows.append({"indicator": name, "period": label, "low": float(vals.min()),
                         "median": float(np.median(vals)), "high": float(vals.max())})
    return pd.DataFrame(rows)


def climate_ahead(t: dict, cfg: Config) -> pd.DataFrame | None:
    """Every city's indicators at the TRACC levels (None without the model patterns file)."""
    path = cfg.reports_dir / "tables" / PATTERNS_FILE
    if not path.exists():
        log.info("No %s: run `python -m meteo climate` once to add the 2030-2100 view", PATTERNS_FILE)
        return None
    patterns = pd.read_csv(path)
    a = cfg.analysis
    daily, yearly = t["city_daily"], t["city_yearly"]
    out = []
    for c in cfg.cities:
        y = yearly[(yearly["city"] == c.name) & yearly["valid_temp"] & yearly["valid_rain"]]["year"].tolist()
        region = nearest_region(c.lat, c.lon)
        r = project_city(daily[daily["city"] == c.name], y, patterns[patterns["region"] == region], cfg.thresholds,
                         tuple(a["old_period"]), tuple(a["recent_period"]))
        r.insert(0, "region", region)
        r.insert(0, "city", c.name)
        out.append(r)
    return pd.concat(out, ignore_index=True)
