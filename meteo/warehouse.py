"""Transform & load: build the DuckDB analytical warehouse.

Layered ("medallion") architecture:
    bronze : raw CSV files + manifest                 (data/raw, immutable)
    silver : stg_daily -> obs_flags -> obs_daily      (typing, quality control)
    gold   : dim_station, ref_station, city_*         (analysis-ready tables)

The business logic is written in SQL (sql/ folder) and run in order.
Python only orchestrates and injects the parameters from config.yaml.
"""
from __future__ import annotations

import json
import logging
import time
from datetime import date
from pathlib import Path
from string import Template

import duckdb

from .config import PROJECT_ROOT, Config

log = logging.getLogger(__name__)

SQL_DIR = PROJECT_ROOT / "sql"
SQL_STEPS = ["01_staging.sql", "02_quality.sql", "03_stations.sql", "04_gold.sql"]
GOLD_TABLES = ["dim_station", "ref_station", "station_offsets", "city_daily", "city_monthly", "city_yearly",
               "city_heatwaves", "city_dry_spells", "dq_summary", "dq_rejected", "network_yearly"]


def render_sql(name: str, params: dict) -> str:
    """Load a SQL file and substitute its ${...} parameters."""
    return Template((SQL_DIR / name).read_text(encoding="utf-8")).substitute(params)


def raw_files(cfg: Config) -> list[Path]:
    files = sorted((cfg.raw_dir / "meteo_france").glob("Q_*_RR-T-Vent.csv.gz"))
    if not files:
        raise FileNotFoundError("No raw file: run `python -m meteo download` first")
    return files


def sql_list(paths: list[Path]) -> str:
    """Python list -> SQL list literal (forward-slash paths for DuckDB)."""
    return "[" + ", ".join("'" + p.as_posix().replace("'", "''") + "'" for p in paths) + "]"


def load_cities(con: duckdb.DuckDBPyConnection, cfg: Config) -> None:
    con.execute("""
        CREATE OR REPLACE TABLE dim_city (
            city VARCHAR PRIMARY KEY, dep VARCHAR, lat DOUBLE, lon DOUBLE, alt DOUBLE,
            station_ids VARCHAR[]
        )""")
    con.executemany(
        "INSERT INTO dim_city VALUES (?, ?, ?, ?, ?, ?)",
        [(c.name, c.dep, c.lat, c.lon, c.alt, list(c.stations)) for c in cfg.cities],
    )


def load_rejected(con: duckdb.DuckDBPyConnection, cfg: Config) -> None:
    """The values checked by hand and found wrong (config.yaml, rejected_values), read by 02_quality.sql."""
    con.execute("""
        CREATE OR REPLACE TABLE rejected_values (
            station_id VARCHAR, date DATE, variable VARCHAR, value DOUBLE,
            reason_en VARCHAR, reason_fr VARCHAR, source VARCHAR
        )""")
    if cfg.rejected:
        con.executemany(
            "INSERT INTO rejected_values VALUES (?, ?, ?, ?, ?, ?, ?)",
            [(r.station, r.date, r.variable, r.value, r.reason.get("en", ""), r.reason.get("fr", ""), r.source)
             for r in cfg.rejected],
        )


def build_params(cfg: Config, files: list[Path], last_date: date | None = None) -> dict:
    a, t = cfg.analysis, cfg.thresholds
    period_days = (date(a["end_year"] + 1, 1, 1) - date(a["start_year"], 1, 1)).days
    return {
        "files": sql_list(files),
        "start_year": a["start_year"],
        "end_year": a["end_year"],
        "period_days": period_days,
        "min_completeness": a["station_min_completeness"],
        "max_distance_km": a["station_max_distance_km"],
        "max_altitude_diff_m": a.get("station_max_altitude_diff_m", 10_000),
        "month_min_completeness": a["month_min_completeness"],
        "year_min_completeness": a["year_min_completeness"],
        "last_date": (last_date or date.today()).isoformat(),
        **{k: v for k, v in t.items()},
    }


def build_warehouse(cfg: Config, files: list[Path] | None = None) -> dict:
    files = files or raw_files(cfg)
    cfg.warehouse.parent.mkdir(parents=True, exist_ok=True)
    stats: dict = {"n_files": len(files), "raw_bytes": sum(f.stat().st_size for f in files)}

    with duckdb.connect(str(cfg.warehouse)) as con:
        load_cities(con, cfg)
        load_rejected(con, cfg)
        params = build_params(cfg, files)
        for name in SQL_STEPS:
            t0 = time.perf_counter()
            if name == "04_gold.sql":
                # the calendar runs up to the last date actually observed
                params["last_date"] = con.sql("SELECT max(date) FROM stg_daily").fetchall()[0][0].isoformat()
            con.execute(render_sql(name, params))
            stats[f"duration_{name}"] = round(time.perf_counter() - t0, 2)
            log.info("  %-16s %.1f s", name, stats[f"duration_{name}"])

        stats.update(collect_stats(con))
        export_parquet(con, cfg)

    with open(cfg.warehouse.parent / "pipeline_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False, default=str)
    log.info("%s raw rows, %s stations, %d cities", f"{stats['n_rows']:,}", stats["n_stations"], stats["n_cities"])
    return stats


def collect_stats(con: duckdb.DuckDBPyConnection) -> dict:
    n_rows, n_stations, d_min, d_max = con.sql(
        "SELECT count(*), count(DISTINCT station_id), min(date), max(date) FROM stg_daily"
    ).fetchall()[0]
    n_valid, n_values = con.sql(
        "SELECT sum(n) FILTER (WHERE status = 'valid'), sum(n) FILTER (WHERE status <> 'missing') FROM dq_summary"
    ).fetchall()[0]
    return {
        "n_rows": n_rows,
        "n_stations": n_stations,
        "first_date": d_min,
        "last_date": d_max,
        "n_measures": int(n_values),
        "n_valid_measures": int(n_valid),
        "n_duplicates": con.sql("SELECT coalesce(sum(n - 1), 0) FROM dq_duplicates").fetchall()[0][0],
        "n_cities": con.sql("SELECT count(DISTINCT city) FROM ref_station").fetchall()[0][0],
    }


def export_parquet(con: duckdb.DuckDBPyConnection, cfg: Config) -> None:
    """Export the gold tables to Parquet (readable by Power BI, Tableau, Spark, pandas…)."""
    out = cfg.warehouse.parent / "gold"
    out.mkdir(parents=True, exist_ok=True)
    for table in GOLD_TABLES:
        con.execute(f"COPY {table} TO '{(out / table).as_posix()}.parquet' (FORMAT parquet, COMPRESSION zstd)")


def connect(cfg: Config, read_only: bool = True) -> duckdb.DuckDBPyConnection:
    if not Path(cfg.warehouse).exists():
        raise FileNotFoundError("Warehouse missing: run `python -m meteo build` first")
    return duckdb.connect(str(cfg.warehouse), read_only=read_only)
