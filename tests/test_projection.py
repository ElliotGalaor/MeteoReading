"""Projection of the coming days (meteo/projection.py), on synthetic series whose behaviour is known."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from meteo.projection import design, project, skill

TH = {"frost_day": 0, "hot_day": 30, "rain_day": 1}
PHI = 0.7          # day-to-day persistence of the temperature anomalies
WET_WET, DRY_WET = 0.7, 0.2   # chance of rain after a rainy day, after a dry day


def city(last="2026-11-11", years=33, seed=1, offset=0.0) -> pd.DataFrame:
    """Daily TX, TN and rainfall: seasonal cycle + slight warming + AR(1) anomalies; rain as a Markov chain."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(pd.Timestamp(last) - pd.DateOffset(years=years), last, freq="D")
    n = len(dates)
    season = -np.cos(2 * np.pi * (dates.day_of_year.to_numpy() - 15) / 365.25)
    trend = 0.3 * (dates.year.to_numpy() - 2000) / 10
    anom = np.zeros(n)
    for i in range(1, n):
        anom[i] = PHI * anom[i - 1] + rng.normal(0, 2.0)
    wet = np.zeros(n, dtype=bool)
    for i in range(1, n):
        wet[i] = rng.random() < (WET_WET if wet[i - 1] else DRY_WET)
    tx = 17 + 9 * season + trend + anom + offset
    return pd.DataFrame({"date": dates, "tx": tx, "tn": tx - 8, "rr": np.where(wet, rng.gamma(2, 3, n) + 1, 0.0)})


def with_last_day(d: pd.DataFrame, gap: float, rainy: bool) -> pd.DataFrame:
    """The same series, with the last day `gap` degrees above the normal of the series, rainy or dry."""
    d = d.copy()
    last = d.index[-1]
    expected = 17 - 9 * np.cos(2 * np.pi * (d.loc[last, "date"].day_of_year - 15) / 365.25) + 0.3 * 2.69
    d.loc[last, "tx"] = expected + gap
    d.loc[last, "tn"] = d.loc[last, "tx"] - 8
    d.loc[last, "rr"] = 8.0 if rainy else 0.0
    return d


@pytest.fixture(scope="module")
def daily():
    return city()


def test_days_run_from_the_day_after_the_last_measurement(daily):
    d = daily.copy()
    d.loc[d.index[-3:], "rr"] = np.nan          # rainfall of the last two days not published yet
    p = project(d, pd.Timestamp("2026-11-20"), TH)
    assert p.anchor == pd.Timestamp("2026-11-08")   # the last day with TX, TN and rainfall
    assert p.days["date"].iloc[0] == pd.Timestamp("2026-11-09") and p.days["date"].iloc[-1] == pd.Timestamp("2026-11-20")
    assert list(p.days["lead"]) == list(range(1, 13))
    assert p.recent["date"].iloc[-1] == p.anchor and len(p.recent) == 21


def test_until_before_the_last_measurement_still_gives_one_day(daily):
    p = project(daily, pd.Timestamp("2020-01-01"), TH)
    assert len(p.days) == 1 and p.days["lead"].iloc[0] == 1


def test_a_warm_gap_fades_at_the_pace_of_the_past(daily):
    p = project(with_last_day(daily, gap=6, rainy=False), pd.Timestamp("2026-12-11"), TH)
    gap = p.days["tx"] - p.days["tx_normal"]
    # AR(1) with phi = 0.7: 4.2 degrees left the next day, 2.9 the day after, nothing after a month
    assert gap.iloc[0] == pytest.approx(6 * PHI, abs=0.5)
    assert gap.iloc[1] == pytest.approx(6 * PHI**2, abs=0.5)
    assert abs(gap.iloc[-1]) < 0.5
    assert p.days["tx_kept"].iloc[0] == pytest.approx(PHI, abs=0.05)
    # The range widens with the lead time, up to the spread of the anomalies (sd 2 / sqrt(1 - phi^2) = 2.8)
    width = p.days["tx_high"] - p.days["tx_low"]
    assert width.iloc[0] < width.iloc[-1]
    assert width.iloc[-1] == pytest.approx(2 * 1.2816 * 2 / np.sqrt(1 - PHI**2), rel=0.15)
    assert (p.days["tx_low"] < p.days["tx"]).all() and (p.days["tx"] < p.days["tx_high"]).all()


def test_the_normal_follows_the_season_and_the_warming(daily):
    p = project(daily, pd.Timestamp("2026-11-30"), TH)
    expected = 17 - 9 * np.cos(2 * np.pi * (p.days["date"].dt.day_of_year - 15) / 365.25) + 0.3 * 2.69
    assert np.allclose(p.days["tx_normal"], expected, atol=0.4)
    assert (np.diff(p.days["tx_normal"]) < 0).all()        # late November: cooling every day


def test_rain_after_a_rainy_or_a_dry_day(daily):
    wet = project(with_last_day(daily, 0, rainy=True), pd.Timestamp("2026-12-11"), TH).days
    dry = project(with_last_day(daily, 0, rainy=False), pd.Timestamp("2026-12-11"), TH).days
    assert wet["p_rain"].iloc[0] == pytest.approx(WET_WET, abs=0.05)
    assert dry["p_rain"].iloc[0] == pytest.approx(DRY_WET, abs=0.05)
    # Weeks later, both are back to the usual share of rainy days: 0.2 / (1 - 0.7 + 0.2) = 0.4
    for days in (wet, dry):
        assert days["p_rain"].iloc[-1] == pytest.approx(0.4, abs=0.06)
        assert days["p_rain_normal"].iloc[-1] == pytest.approx(0.4, abs=0.06)
    assert (wet["rain_mm"] > 1).all()


def test_frost_and_heat_chances():
    cold = project(city(offset=-12), pd.Timestamp("2026-11-18"), TH).days
    warm = project(city(last="2026-07-20", offset=8), pd.Timestamp("2026-07-27"), TH).days
    assert cold["p_frost"].iloc[-1] > 0.5 and (cold["p_hot"] == 0).all()
    assert warm["p_hot"].iloc[-1] > 0.5 and (warm["p_frost"] == 0).all()
    assert cold["p_frost"].between(0, 1).all() and warm["p_hot"].between(0, 1).all()


def test_skill_beats_the_normal_first_then_joins_it(daily):
    s = skill(daily, "tx", max_lead=10, test_days=730)
    assert list(s["lead"]) == list(range(1, 11)) and (s["n"] > 600).all()
    assert s["projection"].iloc[0] < 0.8 * s["normal"].iloc[0]
    assert s["projection"].iloc[-1] == pytest.approx(s["normal"].iloc[-1], rel=0.05)
    assert s["projection"].iloc[:4].is_monotonic_increasing


def test_design_has_trend_and_harmonics():
    x = design(pd.DatetimeIndex(["2000-01-01", "2010-01-01"]))
    assert x.shape == (2, 8) and x[0, 1] == 0 and x[1, 1] == pytest.approx(1, abs=0.01)
