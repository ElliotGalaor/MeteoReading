"""Storms and gusts: days with gusts of 100 km/h or more, their trend, and the strongest gusts recorded.

The daily strongest gust (FXI) is stored in m/s by Météo-France and shown in km/h. Gust series are more often broken
than temperature series, and anemometers and their sites changed over the decades: a year only counts if gusts were
measured on at least MIN_COVERAGE of its days, and a trend needs at least MIN_YEARS such years.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .stats import trend

MS_TO_KMH = 3.6
STORM_KMH = 100.0      # a storm day: strongest gust at or above this
MIN_COVERAGE = 0.95    # share of the days of a year with a gust value
MIN_YEARS = 20


def storm_years(daily: pd.DataFrame, last_year: int, threshold_kmh: float = STORM_KMH) -> pd.DataFrame:
    """Storm days per city and year, with the share of days that have a gust value and whether the year counts.
    Columns: city, year, storm_days, coverage, valid."""
    d = daily[pd.to_datetime(daily["date"]).dt.year <= last_year]
    d = d.assign(year=pd.to_datetime(d["date"]).dt.year.to_numpy(),
                 storm=(d["fxi"] * MS_TO_KMH >= threshold_kmh).to_numpy(), has=d["fxi"].notna().to_numpy())
    g = d.groupby(["city", "year"]).agg(storm_days=("storm", "sum"), coverage=("has", "mean")).reset_index()
    g["valid"] = g["coverage"] >= MIN_COVERAGE
    return g


def storm_trends(years: pd.DataFrame) -> pd.DataFrame:
    """Trend of storm days per city (days per year, per decade), on the valid years only.
    Columns: city, first_year, n_years, mean_days, slope, p_value, mk_p_value, significant."""
    rows = []
    for city, g in years[years["valid"]].groupby("city"):
        row = {"city": city, "first_year": int(g["year"].min()), "n_years": len(g),
               "mean_days": float(g["storm_days"].mean())}
        if len(g) >= MIN_YEARS and g["storm_days"].std() > 0:
            t = trend(g["year"], g["storm_days"])
            row.update(slope=t.slope, p_value=t.p_value, mk_p_value=t.mk_p_value, significant=t.significant)
        else:
            row.update(slope=np.nan, p_value=np.nan, mk_p_value=np.nan, significant=False)
        rows.append(row)
    return pd.DataFrame(rows, columns=["city", "first_year", "n_years", "mean_days", "slope", "p_value", "mk_p_value",
                                       "significant"])


def gust_records(daily: pd.DataFrame, per_city: int = 1) -> pd.DataFrame:
    """The strongest gusts of each city (km/h), strongest first. Columns: city, date, gust_kmh."""
    d = daily.dropna(subset=["fxi"])
    d = d.assign(gust_kmh=d["fxi"] * MS_TO_KMH).sort_values("gust_kmh", ascending=False)
    top = d.groupby("city").head(per_city)
    return top[["city", "date", "gust_kmh"]].reset_index(drop=True)
