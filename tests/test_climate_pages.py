"""Climate pages: records at the time, the last weeks, seasons and best weeks, episodes, storms, units (#55-#60)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from meteo.climatology import (
    day_of_year,
    latest,
    mean_series,
    month_so_far,
    ordinal,
    record_envelope,
    records_between,
    records_set,
)
from meteo.episodes import day_in_context, detect, dry_spells, rank_text
from meteo.pivot import datasets
from meteo.route_scores import PROFILES, WIND_COMFORT_KMH
from meteo.seasons import best_weeks, comfortable, season, shift_days
from meteo.wind import gust_records, storm_trends, storm_years

TH = {"hot_day": 30, "very_hot_day": 35, "tropical_night": 20, "frost_day": 0, "rain_day": 1, "heavy_rain": 20,
      "heatwave_min_days": 3}


def daily(years=range(1951, 2001), city="X", amplitude=8.0, mean=13.0, trend=0.0, seed=0):
    """A city with a seasonal cycle (warmest mid-July), a little noise and an optional trend per year."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(f"{min(years)}-01-01", f"{max(years)}-12-31", freq="D")
    doy = dates.day_of_year.to_numpy()
    t = mean - amplitude * np.cos(2 * np.pi * (doy - 15) / 365.25) + trend * (dates.year - min(years))
    t = t + rng.normal(0, 1.0, len(dates))
    return pd.DataFrame({"city": city, "date": dates, "tmean": t, "tx": t + 5, "tn": t - 5,
                         "rr": np.where(rng.random(len(dates)) < 0.3, 4.0, 0.0), "ffm": 3.0, "fxi": 10.0})


# --- Records and the last weeks ------------------------------------------------------------------
def test_day_of_year_leaves_out_29_february():
    d = pd.Series(pd.to_datetime(["2023-03-01", "2024-03-01", "2024-02-29", "2024-12-31"]))
    assert day_of_year(d).tolist()[:2] == [60, 60]
    assert np.isnan(day_of_year(d).iloc[2]) and day_of_year(d).iloc[3] == 365


def test_records_count_only_the_years_before():
    d = daily(range(1951, 1971))
    d.loc[d["date"] == "1965-07-14", "tx"] = 60.0          # an all-time record...
    env = record_envelope(d, "tx", 1965)                   # ...not yet known before 1965
    assert env.set_index(["month", "day"]).loc[(7, 14), "high"] < 60
    r = records_set(d, "tx", 1965)
    hit = r[r["date"] == pd.Timestamp("1965-07-14")].iloc[0]
    assert hit["value"] == 60 and hit["previous"] < 60 and 1951 <= hit["previous_year"] < 1965
    assert records_set(d, "tx", 1966).query("value >= 60").empty   # the next year has to beat 60


def test_records_between_finds_hot_and_cold_records_in_a_window():
    d = daily(range(1951, 1981))
    d.loc[d["date"] == "1980-06-10", "tx"] = 99.0
    d.loc[d["date"] == "1980-06-12", "tn"] = -99.0
    r = records_between(d, "1980-06-01", "1980-06-30", 1951)
    assert {"hot", "cold"} <= set(r["kind"])
    assert pd.Timestamp("1980-06-10") in set(r.loc[r["kind"] == "hot", "date"])
    assert records_between(d, "1955-06-01", "1955-06-30", 1951).empty   # inside the first 10 years: warm-up


def test_mean_series_corrects_for_cities_not_measured_yet():
    a, b = daily(city="A", mean=10, seed=1), daily(city="B", mean=20, seed=2)
    b.loc[b["date"] == b["date"].max(), "tmean"] = np.nan     # B has not published its last day
    s = mean_series(pd.concat([a, b]), "tmean", (1961, 1990), min_share=0.5)
    last = s.iloc[-1]
    assert last["n_cities"] == 1
    # B's normal fills in for it: about 5 °C above A's value (B is 10 °C warmer), not A's value alone
    assert last["value"] - a["tmean"].iloc[-1] == pytest.approx(5, abs=1.5)


def test_month_so_far_ranks_the_same_days_of_every_year():
    d = daily(range(1951, 2001), trend=0.05)      # warming: recent years warmer
    d = d[d["date"] <= "2000-07-10"]               # the last month is measured up to the 10th
    s = d[["date", "tmean"]].rename(columns={"tmean": "value"})
    m = month_so_far(s, (1961, 1990), 1951)
    assert (m["month"], m["last_day"], m["n_years"]) == (7, 10, 50)
    assert m["anomaly"] > 0 and m["rank_warm"] <= 5
    clim = pd.DataFrame({"doy": range(1, 366), "mean": 0.0, "p10": -1.0, "p90": 1.0})
    assert len(latest(s, clim, 30)) == 30


def test_ordinal():
    assert [ordinal(n) for n in (1, 2, 3, 4, 11, 12, 13, 21, 22, 101, 111)] == \
        ["1st", "2nd", "3rd", "4th", "11th", "12th", "13th", "21st", "22nd", "101st", "111th"]


# --- Seasons and best weeks --------------------------------------------------------------------------
def test_comfortable_days_use_the_route_comfort_thresholds():
    low, high = PROFILES["Walking"].comfort_range
    day = pd.DataFrame({"tx": [high, high + 4, high], "tn": [high, high + 4, high], "rr": [0.0, 0.0, 0.0],
                        "ffm": [1.0, 1.0, (WIND_COMFORT_KMH + 5) / 3.6]})
    ok = comfortable(day, "Walking", rain_day=1)
    assert ok.tolist() == [1.0, 0.0, 0.0]          # too hot; too windy
    assert comfortable(day.assign(rr=2.0), "Walking", rain_day=1).sum() == 0   # a rainy day is never comfortable
    assert comfortable(day.assign(ffm=np.nan), "Walking", rain_day=1, use_wind=False).tolist() == [1.0, 0.0, 1.0]


def test_best_weeks_leave_the_wind_out_when_it_was_not_measured():
    d = daily(range(1951, 2001))
    d.loc[d["date"].dt.year < 1980, "ffm"] = np.nan
    w, use_wind = best_weeks(d, "Walking", {"old": (1961, 1990), "new": (1991, 2000)}, 1)
    assert not use_wind and set(w["period"]) == {"old", "new"} and w["week"].between(1, 52).all()


def test_seasons_start_earlier_when_the_climate_warms():
    old = daily(range(1951, 1981), amplitude=9, mean=15)     # normal TX peaks around 29 °C
    new = daily(range(1991, 2021), amplitude=9, mean=17)
    d = pd.concat([old, new])
    then, now = season(d, (1951, 1980)), season(d, (1991, 2020))
    assert then and now and now.days > then.days
    earlier, later = shift_days(then, now)
    assert earlier > 0 and later > 0
    cold = season(daily(amplitude=9, mean=11), (1961, 1990), warm=False)   # winter spans the new year
    assert cold.start > cold.end and 0 < cold.days < 365
    assert season(daily(amplitude=2, mean=10), (1961, 1990)) is None   # never reaches 25 °C


# --- Episodes and one day in context ------------------------------------------------------------------
def test_episodes_need_a_peak_and_go_on_through_short_breaks():
    dates = pd.date_range("2003-08-01", periods=12, freq="D")
    v = np.array([20, 26, 27, 29, 25, 24.5, 27, 26, 20, 26, 26, 20], dtype=float)
    e = detect(dates, v, spic=28, sdeb=25.5, sint=24, heat=True)
    assert len(e) == 1                     # days 10-11 never reach the peak threshold
    row = e.iloc[0]
    assert (row["start"], row["end"]) == (dates[1], dates[7])   # the 2-day dip stays above Sint
    assert row["peak"] == 29 and row["intensity"] > 0
    cold = detect(dates, -v, spic=-28, sdeb=-25.5, sint=-24, heat=False)
    assert len(cold) == 1 and cold.iloc[0]["peak"] == -29


def test_dry_spells_are_ranked_against_the_city_usual():
    spells = pd.DataFrame({"city": ["South", "North"], "start_date": pd.to_datetime(["2003-06-01", "1976-05-01"]),
                           "end_date": pd.to_datetime(["2003-08-30", "1976-06-10"]), "n_days": [90, 40]})
    yearly = pd.DataFrame({"city": ["South"] * 30 + ["North"] * 30, "year": list(range(1991, 2021)) * 2,
                           "longest_dry_spell": [80] * 30 + [15] * 30})
    r = dry_spells(spells, yearly, (1991, 2020))
    assert r["city"].tolist() == ["North", "South"]   # 40 days in the north is rarer than 90 in the south


def test_one_day_in_context():
    d = daily(range(1951, 2001))
    d.loc[d["date"] == "2000-08-15", "tx"] = 50.0
    ctx = day_in_context(d, "2000-08-15")
    assert ctx["tx"]["rank"] == 1 and ctx["tx"]["word"] == "hottest" and ctx["tx"]["record_year"] == 2000
    assert rank_text(ctx["tx"], "2000-08-15") == "1st hottest 15 August of 50 years"
    assert ctx["ffm"]["value"] == pytest.approx(3.0 * 3.6)   # wind in km/h


# --- Storms, gusts and units ------------------------------------------------------------------------
def test_storm_days_count_valid_years_only():
    d = daily(range(1951, 2001))
    d["fxi"] = 10.0
    d.loc[d["date"].dt.year >= 1990, "fxi"] = np.where(d.loc[d["date"].dt.year >= 1990, "date"].dt.day == 1,
                                                       30.0, 10.0)   # 108 km/h on the 1st of each month
    d.loc[d["date"].dt.year == 1960, "fxi"] = np.nan                 # a year without an anemometer
    y = storm_years(d, 2000)
    assert not y.set_index("year").loc[1960, "valid"]
    assert y.set_index("year").loc[1995, "storm_days"] == 12
    t = storm_trends(y).iloc[0]
    assert t["slope"] > 0 and t["n_years"] == 49
    rec = gust_records(d)
    assert rec.iloc[0]["gust_kmh"] == pytest.approx(108)


def test_wind_is_in_kmh_in_the_pivot_table():
    for ds in datasets(TH).values():
        for key, m in ds.measures.items():
            if key in ("ffm", "fxi"):
                assert m.unit == "km/h" and m.sql.startswith("3.6 *")
