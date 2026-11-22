"""End-to-end test of the warehouse on a small dataset in the Météo-France format.

The dataset deliberately contains the pitfalls found in the real data:
suspect quality code, impossible value, TN > TX, duplicate, ID with a leading zero,
distant station, incomplete station, a station that moves and a gust rejected by hand.
"""
from __future__ import annotations

import gzip
import math
from datetime import date, timedelta

import duckdb
import pytest
import yaml

from meteo.config import _rejected, load_config
from meteo.warehouse import build_warehouse

HEADER = "NUM_POSTE;NOM_USUEL;LAT;LON;ALTI;AAAAMMJJ;RR;QRR;TN;QTN;TX;QTX;TM;QTM;FFM;QFFM;FXI;QFXI"
START, END = date(2000, 1, 1), date(2003, 12, 31)
DAY_SUSPECT, DAY_OUT, DAY_INCONSISTENT, DAY_DUP = date(2001, 3, 1), date(2001, 3, 2), date(2001, 3, 3), date(2001, 3, 4)
DAY_GUST, DAY_FIXED = date(2001, 3, 5), date(2001, 3, 6)   # a wrong gust, and a rejection the raw file no longer matches


def base_tx(d: date) -> float:
    return round(20 + 8 * math.sin(2 * math.pi * d.timetuple().tm_yday / 365), 1)


def row(sid, name, lat, lon, alt, d, tx, tn, rr=0.0, qtx=1, fxi=10.0):
    return f"{sid};{name};{lat};{lon};{alt};{d:%Y%m%d};{rr};1;{tn};1;{tx};{qtx};;;3.0;1;{fxi};1"


def days(a: date, b: date):
    d = a
    while d <= b:
        yield d
        d += timedelta(days=1)


@pytest.fixture(scope="module")
def warehouse(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("wh")
    raw = tmp / "raw" / "meteo_france"
    raw.mkdir(parents=True)

    # Département 06: the right station, a distant station, an incomplete station
    lines = [HEADER]
    for d in days(START, END):
        tx, tn = base_tx(d), round(base_tx(d) - 10, 1)
        rr = 5.0 if d.toordinal() % 3 == 0 else 0.0
        if d == DAY_SUSPECT:
            lines.append(row("06000001", "GOOD", 43.70, 7.26, 5, d, tx, tn, rr, qtx=2))
        elif d == DAY_OUT:
            lines.append(row("06000001", "GOOD", 43.70, 7.26, 5, d, 99.0, tn, rr))
        elif d == DAY_INCONSISTENT:
            lines.append(row("06000001", "GOOD", 43.70, 7.26, 5, d, 5.0, 12.0, rr))
        elif d == DAY_GUST:
            lines.append(row("06000001", "GOOD", 43.70, 7.26, 5, d, tx, tn, rr, fxi=50.0))
        else:
            lines.append(row("06000001", "GOOD", 43.70, 7.26, 5, d, tx, tn, rr))
        if d == DAY_DUP:
            lines.append(lines[-1])
        lines.append(row("06000002", "FAR", 44.70, 7.26, 5, d, tx, tn, rr))           # ~110 km
        if d.year < 2002:
            lines.append(row("06000003", "SHORT", 43.71, 7.27, 5, d, tx, tn, rr))     # half of the time

    # Département 13: one station closes end of 2001, the new one opens early 2001 (1 °C warmer)
    lines13 = [HEADER]
    for d in days(START, END):
        tx, tn = base_tx(d), round(base_tx(d) - 10, 1)
        if d.year <= 2001:
            lines13.append(row("13000001", "OLD", 43.30, 5.37, 10, d, tx, tn))
        if d.year >= 2001:
            lines13.append(row("13000002", "NEW", 43.31, 5.38, 10, d, round(tx + 1, 1), round(tn + 1, 1)))

    for dep, content in (("06", lines), ("13", lines13)):
        with gzip.open(raw / f"Q_{dep}_previous-1950-2024_RR-T-Vent.csv.gz", "wt", encoding="utf-8") as f:
            f.write("\n".join(content) + "\n")

    cfg_path = tmp / "config.yaml"
    cfg_path.write_text(yaml.safe_dump({
        "source": {"dataset_api": "http://example.invalid", "periods": ["previous"], "workers": 1},
        "paths": {"raw": "raw", "warehouse": "wh.duckdb", "reports": "reports"},
        "analysis": {"start_year": 2000, "end_year": 2003, "reference_period": [2000, 2003],
                     "old_period": [2000, 2001], "recent_period": [2002, 2003],
                     "station_min_completeness": 0.85, "station_max_distance_km": 40,
                     "station_max_altitude_diff_m": 200, "month_min_completeness": 0.9,
                     "year_min_completeness": 0.95},
        "thresholds": {"hot_day": 25, "very_hot_day": 27, "tropical_night": 20, "frost_day": 0,
                       "rain_day": 1, "heavy_rain": 20, "heatwave_min_days": 3},
        "cities": [
            {"name": "Testville", "dep": "06", "lat": 43.70, "lon": 7.26, "alt": 10},
            {"name": "Chaintown", "dep": "13", "lat": 43.30, "lon": 5.37, "alt": 10,
             "stations": ["13000002", "13000001"]},
        ],
        "rejected_values": [
            {"station": "06000001", "date": DAY_GUST, "variable": "fxi", "value": 50.0,
             "reason": {"en": "Too strong", "fr": "Trop forte"}, "source": "https://example.invalid"},
            {"station": "06000001", "date": DAY_FIXED, "variable": "FXI", "value": 45.0, "reason": {"en": "Fixed since"}},
        ],
    }), encoding="utf-8")
    cfg = load_config(cfg_path)
    stats = build_warehouse(cfg)
    con = duckdb.connect(str(cfg.warehouse), read_only=True)
    yield con, stats
    con.close()


def one(con, sql):
    return con.sql(sql).fetchall()[0][0]


def test_station_id_keeps_leading_zero(warehouse):
    con, _ = warehouse
    assert one(con, "SELECT count(*) FROM dim_station WHERE station_id = '06000001'") == 1


def test_duplicate_detected(warehouse):
    con, stats = warehouse
    assert stats["n_duplicates"] == 1


def test_rejected_values(warehouse):
    con, _ = warehouse
    tx = dict(con.sql("SELECT status, n FROM dq_summary WHERE variable = 'TX'").fetchall())
    assert tx["suspect"] == 1 and tx["out_of_range"] == 1 and tx["inconsistent"] == 1
    nulls = one(con, f"""SELECT count(*) FROM obs_daily WHERE station_id = '06000001' AND tx IS NULL
                         AND date IN ('{DAY_SUSPECT}', '{DAY_OUT}', '{DAY_INCONSISTENT}')""")
    assert nulls == 3


def test_a_value_rejected_by_hand_drops_out(warehouse):
    con, _ = warehouse
    fxi = dict(con.sql("SELECT status, n FROM dq_summary WHERE variable = 'FXI'").fetchall())
    assert fxi["rejected"] == 1
    assert one(con, f"SELECT fxi FROM obs_daily WHERE station_id = '06000001' AND date = '{DAY_GUST}'") is None
    assert one(con, "SELECT max(fxi) FROM city_daily WHERE city = 'Testville'") == 10.0


def test_a_rejection_lapses_when_the_raw_file_no_longer_holds_the_value(warehouse):
    con, _ = warehouse
    assert dict(con.sql("SELECT date, applied FROM dq_rejected").fetchall()) == {DAY_GUST: True, DAY_FIXED: False}
    assert one(con, f"SELECT fxi FROM obs_daily WHERE station_id = '06000001' AND date = '{DAY_FIXED}'") == 10.0


def test_a_rejected_value_needs_a_known_variable():
    with pytest.raises(ValueError, match="unknown variable"):
        _rejected([{"station": "06000001", "date": "2001-03-05", "variable": "gust", "value": 50.0}])


def test_station_choice(warehouse):
    con, _ = warehouse
    assert one(con, "SELECT station_id FROM ref_station WHERE city = 'Testville'") == "06000001"


def test_splicing_corrects_the_bias(warehouse):
    con, _ = warehouse
    offsets = [r[0] for r in con.sql("SELECT offset_tx FROM station_offsets WHERE city = 'Chaintown'").fetchall()]
    assert len(offsets) == 12
    assert all(abs(o - 1.0) < 0.01 for o in offsets)
    # 2000 is only covered by the old station: brought to the level of the new one (+1 °C)
    t2000 = one(con, "SELECT avg(tx) FROM city_daily WHERE city = 'Chaintown' AND year(date) = 2000")
    expected = sum(base_tx(d) + 1 for d in days(date(2000, 1, 1), date(2000, 12, 31))) / 366
    assert t2000 == pytest.approx(expected, abs=0.02)


def test_full_calendar_and_valid_years(warehouse):
    con, _ = warehouse
    n_days = (END - START).days + 1
    assert one(con, "SELECT count(*) FROM city_daily") == 2 * n_days
    assert one(con, "SELECT count(*) FROM city_yearly WHERE valid_temp") == 8


def test_dry_spells(warehouse):
    con, _ = warehouse
    # rain one day in three -> dry spells of 2 days at most
    assert one(con, "SELECT max(n_days) FROM city_dry_spells WHERE city = 'Testville'") == 2
