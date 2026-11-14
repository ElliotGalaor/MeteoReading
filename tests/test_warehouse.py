"""Test de bout en bout de l'entrepôt sur un mini-jeu de données au format Météo-France.

Le jeu contient volontairement les pièges rencontrés dans les vraies données :
code qualité douteux, valeur impossible, TN > TX, doublon, identifiant à zéro initial,
station éloignée, station incomplète et station qui déménage.
"""
from __future__ import annotations

import gzip
import math
from datetime import date, timedelta

import duckdb
import pytest
import yaml

from meteo.config import load_config
from meteo.warehouse import build_warehouse

HEADER = "NUM_POSTE;NOM_USUEL;LAT;LON;ALTI;AAAAMMJJ;RR;QRR;TN;QTN;TX;QTX;TM;QTM;FFM;QFFM;FXI;QFXI"
START, END = date(2000, 1, 1), date(2003, 12, 31)
DAY_SUSPECT, DAY_OUT, DAY_INCOHERENT, DAY_DUP = date(2001, 3, 1), date(2001, 3, 2), date(2001, 3, 3), date(2001, 3, 4)


def base_tx(d: date) -> float:
    return round(20 + 8 * math.sin(2 * math.pi * d.timetuple().tm_yday / 365), 1)


def row(sid, name, lat, lon, alt, d, tx, tn, rr=0.0, qtx=1):
    return f"{sid};{name};{lat};{lon};{alt};{d:%Y%m%d};{rr};1;{tn};1;{tx};{qtx};;;3.0;1;10.0;1"


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

    # Département 06 : la bonne station, une station loin, une station incomplète
    lines = [HEADER]
    for d in days(START, END):
        tx, tn = base_tx(d), round(base_tx(d) - 10, 1)
        rr = 5.0 if d.toordinal() % 3 == 0 else 0.0
        if d == DAY_SUSPECT:
            lines.append(row("06000001", "BONNE", 43.70, 7.26, 5, d, tx, tn, rr, qtx=2))
        elif d == DAY_OUT:
            lines.append(row("06000001", "BONNE", 43.70, 7.26, 5, d, 99.0, tn, rr))
        elif d == DAY_INCOHERENT:
            lines.append(row("06000001", "BONNE", 43.70, 7.26, 5, d, 5.0, 12.0, rr))
        else:
            lines.append(row("06000001", "BONNE", 43.70, 7.26, 5, d, tx, tn, rr))
        if d == DAY_DUP:
            lines.append(lines[-1])
        lines.append(row("06000002", "LOIN", 44.70, 7.26, 5, d, tx, tn, rr))           # ~110 km
        if d.year < 2002:
            lines.append(row("06000003", "COURTE", 43.71, 7.27, 5, d, tx, tn, rr))     # 50 % du temps

    # Département 13 : une station ferme fin 2001, la nouvelle ouvre début 2001 (1 °C plus chaude)
    lines13 = [HEADER]
    for d in days(START, END):
        tx, tn = base_tx(d), round(base_tx(d) - 10, 1)
        if d.year <= 2001:
            lines13.append(row("13000001", "ANCIENNE", 43.30, 5.37, 10, d, tx, tn))
        if d.year >= 2001:
            lines13.append(row("13000002", "NOUVELLE", 43.31, 5.38, 10, d, round(tx + 1, 1), round(tn + 1, 1)))

    for dep, content in (("06", lines), ("13", lines13)):
        with gzip.open(raw / f"Q_{dep}_previous-1950-2024_RR-T-Vent.csv.gz", "wt", encoding="utf-8") as f:
            f.write("\n".join(content) + "\n")

    cfg_path = tmp / "config.yaml"
    cfg_path.write_text(yaml.safe_dump({
        "source": {"dataset_api": "http://exemple.invalid", "periods": ["previous"], "workers": 1},
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
    }), encoding="utf-8")
    cfg = load_config(cfg_path)
    stats = build_warehouse(cfg)
    con = duckdb.connect(str(cfg.warehouse), read_only=True)
    yield con, stats
    con.close()


def one(con, sql):
    return con.sql(sql).fetchall()[0][0]


def test_identifiant_garde_le_zero_initial(warehouse):
    con, _ = warehouse
    assert one(con, "SELECT count(*) FROM dim_station WHERE station_id = '06000001'") == 1


def test_doublon_detecte(warehouse):
    con, stats = warehouse
    assert stats["n_duplicates"] == 1


def test_valeurs_rejetees(warehouse):
    con, _ = warehouse
    tx = dict(con.sql("SELECT status, n FROM dq_summary WHERE variable = 'TX'").fetchall())
    assert tx["douteuse"] == 1 and tx["hors_bornes"] == 1 and tx["incoherente"] == 1
    nulls = one(con, f"""SELECT count(*) FROM obs_daily WHERE station_id = '06000001' AND tx IS NULL
                         AND date IN ('{DAY_SUSPECT}', '{DAY_OUT}', '{DAY_INCOHERENT}')""")
    assert nulls == 3


def test_choix_de_la_station(warehouse):
    con, _ = warehouse
    assert one(con, "SELECT station_id FROM ref_station WHERE city = 'Testville'") == "06000001"


def test_raccordement_corrige_le_biais(warehouse):
    con, _ = warehouse
    offsets = [r[0] for r in con.sql("SELECT offset_tx FROM station_offsets WHERE city = 'Chaintown'").fetchall()]
    assert len(offsets) == 12
    assert all(abs(o - 1.0) < 0.01 for o in offsets)
    # 2000 n'est couverte que par l'ancienne station : ramenée au niveau de la nouvelle (+1 °C)
    t2000 = one(con, "SELECT avg(tx) FROM city_daily WHERE city = 'Chaintown' AND year(date) = 2000")
    expected = sum(base_tx(d) + 1 for d in days(date(2000, 1, 1), date(2000, 12, 31))) / 366
    assert t2000 == pytest.approx(expected, abs=0.02)


def test_calendrier_complet_et_annees_valides(warehouse):
    con, _ = warehouse
    n_days = (END - START).days + 1
    assert one(con, "SELECT count(*) FROM city_daily") == 2 * n_days
    assert one(con, "SELECT count(*) FROM city_yearly WHERE valid_temp") == 8


def test_sequences_seches(warehouse):
    con, _ = warehouse
    # pluie un jour sur trois -> séquences sèches de 2 jours au plus
    assert one(con, "SELECT max(n_days) FROM city_dry_spells WHERE city = 'Testville'") == 2
