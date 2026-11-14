"""Command-line interface.

    python -m meteo download     # 1. bulk download of the Météo-France files
    python -m meteo build        # 2. DuckDB warehouse: cleaning, quality checks, aggregates
    python -m meteo analyze      # 3. statistics + figures + results.json
    python -m meteo era5         # 4. (optional) cross-validation against ERA5 / Open-Meteo
    python -m meteo report       # 5. self-contained HTML report
    python -m meteo all          # run everything
    python -m meteo query "SELECT ..."   # query the warehouse (or run a .sql file)
    python -m meteo route track.gpx --start "2026-11-20 09:00" --activity cycling   # weather along a GPS track
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
            title = next((line[2:].strip() for line in statement.splitlines() if line.startswith("--")), None)
            if title:
                print(f"\n=== {title}")
            print(con.sql(statement).df().to_string(index=False, max_rows=args.max_rows))


def cmd_route(cfg, args):
    import pandas as pd

    from .gps import ACTIVITIES, parse_track, passage_times, summary, with_times
    from .route_weather import read_route

    path = Path(args.track)
    track = parse_track(path.read_bytes(), path.name)
    if args.start or not track.has_time:
        if not args.start:
            raise SystemExit("This track has no timestamps: add --start 'YYYY-MM-DD HH:MM'")
        preset = ACTIVITIES[args.activity.capitalize()]
        start = pd.Timestamp(args.start, tz=args.tz)
        track = with_times(track, passage_times(track, start, args.speed or preset["speed"],
                                                preset["climb_min_per_100m"]))
    s = summary(track)
    reading = read_route(track)
    pts = reading.points
    local = pts["time"].dt.tz_convert(reading.timezone)
    print(f"\n{track.name}: {s['distance_km']:.1f} km, {local.iloc[0]:%a %d %b %Y %H:%M} -> {local.iloc[-1]:%H:%M} "
          f"({reading.timezone}), source: {reading.source}\n")
    for line in reading.insights:
        print(f"  - {line}")
    table = pd.DataFrame({
        "km": pts["dist_km"].round(1), "time": local.dt.strftime("%H:%M"), "ele_m": pts["ele_profile"].round(0),
        "conditions": pts["condition"], "temp_c": pts["temperature_2m"].round(1),
        "rain_mm_h": pts["precipitation"].round(1), "wind": pts["wind_from"] + " " + pts["wind_speed_10m"].round(0).astype(int).astype(str),
        "headwind": pts["headwind"].round(0), "gusts": pts["wind_gusts_10m"].round(0),
    })
    print("\n" + table.to_string(index=False))
    if args.out:
        table.to_csv(args.out, index=False)
        print(f"\nSaved to {args.out}")
    if args.png:
        from .viz import route_profile

        sub = (f"{s['distance_km']:.1f} km · {local.iloc[0]:%d %b %Y %H:%M} to {local.iloc[-1]:%H:%M} "
               f"({reading.timezone}) · {'ERA5 archive' if reading.source == 'archive' else 'forecast'}")
        route_profile(pts, track.points, Path(args.png), track.name, sub)
        print(f"Chart saved to {args.png}")


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
    p_r = sub.add_parser("route", help="weather reading along a GPS track (GPX, KML, GeoJSON)")
    p_r.add_argument("track", help="path to the track file")
    p_r.add_argument("--start", help="departure time 'YYYY-MM-DD HH:MM' (planned trip); default: the track's own times")
    p_r.add_argument("--activity", choices=["walking", "cycling", "driving"], default="cycling")
    p_r.add_argument("--speed", type=float, help="average speed in km/h (default: the activity's)")
    p_r.add_argument("--tz", default="Europe/Paris", help="time zone of --start")
    p_r.add_argument("--out", help="save the point-by-point reading to this CSV file")
    p_r.add_argument("--png", help="save a profile chart (elevation, temperature, rain, wind) to this PNG file")
    p_r.set_defaults(func=cmd_route)

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
