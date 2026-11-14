"""Tests des fonctions statistiques et de mise en forme."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from meteo.download import discover_resources
from meteo.fmt import fr, sfr, verdict
from meteo.stats import anomalies, composite, daily_outliers, record_counts, seasonal, trend


def test_tendance_retrouve_la_pente():
    rng = np.random.default_rng(0)
    years = np.arange(1951, 2026)
    values = 0.03 * years + rng.normal(0, 0.2, len(years))   # 0,3 °C par décennie
    t = trend(years, values)
    assert t.slope == pytest.approx(0.3, abs=0.05)
    assert t.ci_low < 0.3 < t.ci_high
    assert t.significant


def test_tendance_absente_sur_du_bruit():
    rng = np.random.default_rng(1)
    t = trend(np.arange(1951, 2026), rng.normal(0, 1, 75))
    assert not t.significant


def test_tendance_refuse_une_serie_trop_courte():
    with pytest.raises(ValueError):
        trend([2020, 2021, 2022], [1, 2, 3])


def test_anomalies_nulles_en_moyenne_sur_la_reference():
    df = pd.DataFrame({"city": ["A"] * 4 + ["B"] * 4, "year": [2000, 2001, 2002, 2003] * 2,
                       "t": [10, 12, 14, 16, 20, 22, 24, 26]})
    out = anomalies(df, ["t"], (2000, 2001))
    assert out.loc[out["year"].between(2000, 2001), "t_anom"].abs().sum() == pytest.approx(1 + 1 + 1 + 1)
    assert out.loc[(out["city"] == "B") & (out["year"] == 2003), "t_anom"].item() == pytest.approx(5)


def test_anomalies_relatives_et_plusieurs_cles():
    df = pd.DataFrame({"city": ["A", "A", "A", "A"], "season": ["Été", "Été", "Hiver", "Hiver"],
                       "year": [2000, 2001, 2000, 2001], "rr": [100, 150, 200, 200]})
    out = anomalies(df, ["rr"], (2000, 2001), by=["city", "season"], relative=True)
    assert out["rr_anom"].tolist() == pytest.approx([-20, 20, 0, 0])


def test_composite_ignore_les_annees_trop_incompletes():
    df = pd.DataFrame({"city": list("ABCDE") + list("AB"), "year": [2000] * 5 + [2001] * 2,
                       "x_anom": [1, 2, 3, 4, 5, 10, 10]})
    res = composite(df, "x_anom")
    assert list(res.index) == [2000]
    assert res.loc[2000] == 3


def test_hiver_regroupe_decembre_et_l_annee_suivante():
    rows = [{"city": "A", "year": y, "month": m, "tmean": float(m), "valid_temp": True}
            for y in (2000, 2001) for m in range(1, 13)]
    s = seasonal(pd.DataFrame(rows))
    hiver_2001 = s[(s["season"] == "Hiver") & (s["year"] == 2001)]["tmean"].item()
    assert hiver_2001 == pytest.approx((12 + 1 + 2) / 3)        # déc. 2000 + jan. et fév. 2001
    assert s[(s["season"] == "Hiver") & (s["year"] == 2000)].empty  # déc. 1999 absent


def test_records_dans_un_climat_qui_se_rechauffe():
    dates = pd.date_range("1951-01-01", "2010-12-31", freq="D")
    trend_c = (dates.year - 1951) * 0.05
    daily = pd.DataFrame({"city": "A", "date": dates, "tx": 20 + trend_c, "tn": 10 + trend_c})
    rc = record_counts(daily, 1951)
    assert rc["hot"].sum() > 0 and rc["cold"].sum() == 0


def test_journee_atypique_detectee():
    dates = pd.date_range("1991-01-01", "2020-12-31", freq="D")
    rng = np.random.default_rng(2)
    t = 12 + 8 * np.sin(2 * np.pi * dates.day_of_year / 365) + rng.normal(0, 2, len(dates))
    daily = pd.DataFrame({"city": "A", "date": dates, "tmean": t})
    daily.loc[daily["date"] == "2010-01-15", "tmean"] += 25
    hot, _, _ = daily_outliers(daily, (1991, 2020), top=1)
    assert hot["date"].iloc[0] == pd.Timestamp("2010-01-15")


def test_decouverte_des_fichiers():
    payload = {"resources": [
        {"url": "https://x/Q_06_previous-1950-2024_RR-T-Vent.csv.gz", "filesize": 10},
        {"url": "https://x/Q_06_latest-2025-2026_RR-T-Vent.csv.gz", "filesize": 5},
        {"url": "https://x/Q_06_previous-1950-2024_autres-parametres.csv.gz", "filesize": 7},
        {"url": "https://x/Q_13_previous-1950-2024_RR-T-Vent.csv.gz", "filesize": 9},
        {"url": "https://x/Q_2A_avant-1949_RR-T-Vent.csv.gz", "filesize": 3},
    ]}
    res = discover_resources(payload, ["06"], ["previous", "latest"])
    assert [(r.dep, r.period) for r in res] == [("06", "latest"), ("06", "previous")]
    assert len(discover_resources(payload, "all", ["previous", "latest", "avant"])) == 4


def test_nombres_a_la_francaise():
    assert fr(0.3315, 2) == "0,33"
    assert fr(25026059, 0) == "25 026 059"
    assert sfr(-1.24, 1) == "−1,2"
    assert sfr(0.5, 1) == "+0,5"
    assert verdict({"significant": False, "mk_p_value": 0.07}) == "à la limite de la significativité"
