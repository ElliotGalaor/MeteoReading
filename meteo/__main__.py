"""Command-line interface.

    python -m meteo download     # 1. bulk download of the Météo-France files
    python -m meteo build        # 2. DuckDB warehouse: cleaning, quality checks, aggregates
    python -m meteo analyze      # 3. statistics + figures + results.json
    python -m meteo era5         # 4. (optional) cross-validation against ERA5 / Open-Meteo
    python -m meteo report       # 5. self-contained HTML report
    python -m meteo all          # run everything
    python -m meteo query "SELECT ..."   # query the warehouse (or run a .sql file)
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from contextlib import contextmanager
from pathlib import Path

from .config import load_config

log = logging.getLogger("meteo")


@contextmanager
def step(name: str):
    log.info("━━ %s", name)
    t0 = time.perf_counter()
    yield
    log.info("✓ %s done in %.1f s", name, time.perf_counter() - t0)


def cmd_download(cfg, args):
    from .download import download_all

    with step("Data collection"):
        download_all(cfg)


def cmd_build(cfg, args):
    from .warehouse import build_warehouse

    with step("Warehouse build"):
        build_warehouse(cfg)


def cmd_analyze(cfg, args):
    from .analysis import run_analysis

    with step("Statistical analysis"):
        run_analysis(cfg)


def cmd_era5(cfg, args):
    from .era5 import run_era5_validation

    with step("ERA5 cross-validation"):
        run_era5_validation(cfg)


def cmd_report(cfg, args):
    from .report import build_report

    with step("HTML report"):
        path = build_report(cfg)
    log.info("Report: %s", path)


def cmd_all(cfg, args):
    cmd_download(cfg, args)
    cmd_build(cfg, args)
    if not args.no_era5:
        cmd_era5(cfg, args)
    cmd_analyze(cfg, args)
    cmd_report(cfg, args)


def cmd_query(cfg, args):
    import duckdb
    import pandas as pd

    sql = args.sql
    if sql.endswith(".sql") and Path(sql).exists():
        sql = Path(sql).read_text(encoding="utf-8")
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    with duckdb.connect(str(cfg.warehouse), read_only=True) as con:
        # A file can hold several queries separated by ";"
        for statement in [s.strip() for s in sql.split(";") if s.strip()]:
            title = next((l[2:].strip() for l in statement.splitlines() if l.startswith("--")), None)
            if title:
                print(f"\n=== {title}")
            print(con.sql(statement).df().to_string(index=False, max_rows=args.max_rows))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="meteo", description="Climate analysis pipeline on Météo-France open data")
    parser.add_argument("--config", help="path to config.yaml", default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, fn in [("download", cmd_download), ("build", cmd_build), ("analyze", cmd_analyze),
                     ("era5", cmd_era5), ("report", cmd_report)]:
        sub.add_parser(name).set_defaults(func=fn)
    p_all = sub.add_parser("all")
    p_all.add_argument("--no-era5", action="store_true", help="skip the Open-Meteo API calls")
    p_all.set_defaults(func=cmd_all)
    p_q = sub.add_parser("query")
    p_q.add_argument("sql", help="SQL query or path to a .sql file")
    p_q.add_argument("--max-rows", type=int, default=30)
    p_q.set_defaults(func=cmd_query)

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    cfg = load_config(args.config)
    args.func(cfg, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
