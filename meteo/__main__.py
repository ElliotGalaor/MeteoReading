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


def cmd_climate(cfg, args):
    from .climate_ahead import run_patterns

    with step("Climate model patterns (Open-Meteo Climate API, once)"):
        run_patterns(cfg)


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


def split_sql(text: str) -> list[str]:
    """Statements of a SQL script: split on the semicolons that are not in a string, a quoted name or a comment.
    Each statement keeps the comments above it (the first one is its title)."""
    parts, buf, code, quote, i = [], [], False, None, 0
    while i < len(text):
        c = text[i]
        if quote:                                    # inside '...' or "..." ('' and "" close and reopen)
            quote = None if c == quote else quote
        elif c in "'\"":
            quote = c
        elif text.startswith("--", i) or text.startswith("/*", i):
            end = text.find("\n", i) if c == "-" else text.find("*/", i + 2)
            end = len(text) if end < 0 else end + (0 if c == "-" else 2)
            buf.append(text[i:end])
            i = end
            continue
        elif c == ";":
            parts.append(("".join(buf), code))
            buf, code = [], False
            i += 1
            continue
        buf.append(c)
        code = code or not c.isspace()
        i += 1
    parts.append(("".join(buf), code))
    return [p.strip() for p, has_code in parts if has_code]   # a part with only comments is not a statement


def cmd_query(cfg, args):
    import duckdb
    import pandas as pd

    sql = args.sql
    if sql.strip().lower().endswith(".sql"):
        if not Path(sql).exists():
            raise SystemExit(f"error: file not found: {sql}")
        sql = Path(sql).read_text(encoding="utf-8")
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    with duckdb.connect(str(cfg.warehouse), read_only=True) as con:
        for n, statement in enumerate(split_sql(sql), 1):
            title = next((line.strip()[2:].strip() for line in statement.splitlines() if line.strip().startswith("--")), None)
            if title:
                print(f"\n=== {title}")
            try:
                rel = con.sql(statement)
                # Only the rows shown are fetched: SELECT * on 25 million rows must not load them all first
                df = rel.limit(args.max_rows + 1).df() if rel is not None else None
            except duckdb.Error as exc:
                raise SystemExit(f"error in query {n}: {str(exc).strip()}") from None
            if df is None:
                print("OK")
                continue
            print(df.head(args.max_rows).to_string(index=False))
            if len(df) > args.max_rows:
                print(f"... more rows: only the first {args.max_rows} are shown (--max-rows)")


def cmd_route(cfg, args):
    import pandas as pd

    from .gps import ACTIVITIES, main_track, passage_times, read_tracks, summary, with_times
    from .route_scores import assess
    from .route_weather import clock_format, read_route, track_timezone

    path = Path(args.track)
    if not path.is_file():
        raise SystemExit(f"error: file not found: {path}")
    try:
        tracks = read_tracks(path.read_bytes(), path.name)
        track = main_track(tracks)
    except ValueError as exc:
        raise SystemExit(f"error: could not read {path.name}: {exc}") from None
    if len(tracks) > 1:
        print(f"note: {path.name} holds {len(tracks)} separate tracks; reading the longest one in order, "
              f"{track.name} ({track.distance_km:.1f} km)")
    if args.start or not track.has_time:
        if not args.start:
            raise SystemExit("error: this track has no timestamps: add --start 'YYYY-MM-DD HH:MM'")
        try:
            naive = pd.Timestamp(args.start)
        except ValueError:
            raise SystemExit(f"error: --start {args.start!r} is not a date and time like '2026-11-20 09:00'") from None
        preset = ACTIVITIES[args.activity.capitalize()]
        tz = args.tz
        if tz is None:   # the time zone where the track starts
            first = track.points.iloc[0]
            tz = track_timezone(float(first["lat"]), float(first["lon"])) or "Europe/Paris"
        start = naive.tz_localize(tz, nonexistent="shift_forward", ambiguous=True)
        speed = args.speed if args.speed is not None else preset["speed"]
        track = with_times(track, passage_times(track, start, speed, preset["climb_min_per_100m"]))
    s = summary(track)
    try:
        reading = read_route(track)
    except (ValueError, RuntimeError) as exc:
        raise SystemExit(f"error: {exc}") from None
    pts = reading.points
    local = pts["time"].dt.tz_convert(reading.timezone)
    end = f"{local.iloc[-1]:%H:%M}" if local.iloc[-1].date() == local.iloc[0].date() else f"{local.iloc[-1]:%a %d %b %H:%M}"
    print(f"\n{track.name}: {s['distance_km']:.1f} km, {local.iloc[0]:%a %d %b %Y %H:%M} -> {end} "
          f"({reading.timezone}), source: {reading.source}\n")
    for line in reading.insights:
        print(f"  - {line}")
    scores = assess(pts, args.activity.capitalize(), reading.timezone)
    worst = ", ".join(f"{f.name.lower()} {f.value:.0f} at km {f.km:.0f}" for f in scores.danger.factors[:3])
    lost = ", ".join(f"{f.name.lower()} -{f.value:.0f}" for f in scores.comfort.factors[:3])
    print(f"\n  Danger  {scores.danger.value:3d}/100 ({scores.danger.level})" + (f": {worst}" if worst else ""))
    print(f"  Comfort {scores.comfort.value:3d}/100 ({scores.comfort.level})" + (f": {lost}" if lost else ""))
    print(f"\n  What to plan for ({scores.activity.lower()}):")
    for a in scores.advice:
        print(f"  - {a.title}: {a.text}")
    table = pd.DataFrame({
        "km": pts["dist_km"].round(1), "time": local.dt.strftime(clock_format(local)), "ele_m": pts["ele_profile"].round(0),
        "conditions": pts["condition"], "temp_c": pts["temperature_2m"].round(1),
        "rain_mm_h": pts["precipitation"].round(1), "wind": pts["wind_from"] + " " + pts["wind_speed_10m"].round(0).astype(int).astype(str),
        "headwind": pts["headwind"].round(0) + 0.0, "gusts": pts["wind_gusts_10m"].round(0),   # + 0.0: no "-0.0"
    })
    print("\n" + table.to_string(index=False))
    if args.out:
        table.to_csv(args.out, index=False)
        print(f"\nSaved to {args.out}")
    if args.png:
        from .viz import route_profile

        sub = (f"{s['distance_km']:.1f} km · {local.iloc[0]:%d %b %Y %H:%M} to {end} "
               f"({reading.timezone}) · {'ERA5 archive' if reading.source == 'archive' else 'forecast'}")
        route_profile(pts, track.points, Path(args.png), track.name, sub)
        print(f"Chart saved to {args.png}")


def _positive(kind):
    def check(text: str):
        try:
            value = kind(text)
        except ValueError:
            raise argparse.ArgumentTypeError(f"{text!r} is not a number") from None
        if value <= 0:
            raise argparse.ArgumentTypeError(f"must be greater than 0, not {text}")
        return value
    return check


def main(argv: list[str] | None = None) -> int:
    # Before argparse prints anything (--help, errors): "Météo-France" must not come out garbled in pipes
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(prog="meteo", description="Climate analysis pipeline on Météo-France open data")
    parser.add_argument("--config", help="path to config.yaml", default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, fn, text in [
        ("download", cmd_download, "1. bulk download of the Météo-France files"),
        ("build", cmd_build, "2. DuckDB warehouse: cleaning, quality checks, aggregates"),
        ("analyze", cmd_analyze, "3. statistics, figures and results.json"),
        ("era5", cmd_era5, "4. (optional) cross-validation against ERA5 / Open-Meteo"),
        ("climate", cmd_climate, "(once) climate model patterns for the 2030-2100 view (Open-Meteo Climate API)"),
        ("report", cmd_report, "5. self-contained HTML report"),
    ]:
        sub.add_parser(name, help=text, description=text).set_defaults(func=fn)
    p_all = sub.add_parser("all", help="run every step", description="Run every step, from download to report.")
    p_all.add_argument("--no-era5", action="store_true", help="skip the Open-Meteo API calls")
    p_all.set_defaults(func=cmd_all)
    p_q = sub.add_parser("query", help="query the warehouse (read-only)",
                         description="Run a SQL query, or every query of a .sql file, on the warehouse (read-only).")
    p_q.add_argument("sql", help="SQL query or path to a .sql file")
    p_q.add_argument("--max-rows", type=_positive(int), default=30, help="rows shown per query (default: 30)")
    p_q.set_defaults(func=cmd_query)
    p_r = sub.add_parser("route", help="weather reading along a GPS track (GPX, KML, GeoJSON)")
    p_r.add_argument("track", help="path to the track file")
    p_r.add_argument("--start", help="departure time 'YYYY-MM-DD HH:MM' (planned trip); default: the track's own times")
    p_r.add_argument("--activity", choices=["walking", "cycling", "driving"], default="cycling",
                     help="sets the speed of a planned trip, and the danger and comfort scores")
    p_r.add_argument("--speed", type=_positive(float), help="average speed in km/h (default: the activity's)")
    p_r.add_argument("--tz", help="time zone of --start (default: the one where the track starts)")
    p_r.add_argument("--out", help="save the point-by-point reading to this CSV file")
    p_r.add_argument("--png", help="save a profile chart (elevation, temperature, rain, wind) to this PNG file")
    p_r.set_defaults(func=cmd_route)

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    cfg = load_config(args.config)
    args.func(cfg, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
