"""Daily climatology: normals by calendar day, records "at the time", the last weeks against the normal.

Pure functions on the daily table (`city_daily`: city, date, tn, tx, tmean, rr, ffm, fxi), covered by unit tests.
Calendar days are numbered on a 365-day year: 29 February is left out of the normals, and in a leap year the days
from 1 March keep the number they have in other years.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .i18n import month_name, ordinal  # noqa: F401  (ordinal: "1st" / "1er", for the pages)

MONTH_NAMES = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
               "November", "December"]


def day_of_year(dates: pd.Series) -> pd.Series:
    """Day 1 to 365 of each date, 29 February excluded (NaN): 1 March is day 60 every year."""
    d = pd.to_datetime(pd.Series(dates))
    doy = d.dt.day_of_year - (d.dt.is_leap_year & (d.dt.month > 2)).astype(int)
    return doy.where(~((d.dt.month == 2) & (d.dt.day == 29)))


def doy_climatology(daily: pd.DataFrame, col: str, ref: tuple[int, int], half_window: int = 15) -> pd.DataFrame:
    """Smoothed daily normal (mean, 10th and 90th percentiles) over a ±`half_window`-day window, from the years
    of `ref`. One row per calendar day: doy, mean, p10, p90."""
    d = daily[pd.to_datetime(daily["date"]).dt.year.between(*ref)][["date", col]].dropna()
    doy = day_of_year(d["date"])
    keep = doy.notna().to_numpy()
    days = doy.to_numpy()[keep].astype(int) - 1
    vals = d[col].to_numpy(dtype=float)[keep]
    years = pd.to_datetime(d["date"]).dt.year.to_numpy()[keep]
    grid = np.full((ref[1] - ref[0] + 1, 365), np.nan)
    grid[years - ref[0], days] = vals
    rows = []
    for i in range(365):
        cols = [(i + k) % 365 for k in range(-half_window, half_window + 1)]
        w = grid[:, cols].ravel()
        w = w[~np.isnan(w)]
        rows.append((i + 1, w.mean(), np.percentile(w, 10), np.percentile(w, 90)) if len(w) else
                    (i + 1, np.nan, np.nan, np.nan))
    return pd.DataFrame(rows, columns=["doy", "mean", "p10", "p90"])


# --- Records "at the time" --------------------------------------------------------------------
def record_envelope(daily: pd.DataFrame, col: str, before_year: int) -> pd.DataFrame:
    """Highest and lowest value of `col` measured on each calendar day (month, day) in the years before
    `before_year`, with the year of each (the first year that reached it). Columns: month, day, high, high_year,
    low, low_year."""
    d = daily[pd.to_datetime(daily["date"]).dt.year < before_year][["date", col]].dropna()
    if d.empty:
        return pd.DataFrame(columns=["month", "day", "high", "high_year", "low", "low_year"])
    dates = pd.to_datetime(d["date"])
    d = d.assign(month=dates.dt.month.to_numpy(), day=dates.dt.day.to_numpy(), year=dates.dt.year.to_numpy())
    d = d.sort_values("date")
    g = d.groupby(["month", "day"])[col]
    hi, lo = d.loc[g.idxmax()], d.loc[g.idxmin()]
    out = pd.DataFrame({"month": hi["month"].to_numpy(), "day": hi["day"].to_numpy(), "high": hi[col].to_numpy(),
                        "high_year": hi["year"].to_numpy()})
    return out.merge(pd.DataFrame({"month": lo["month"].to_numpy(), "day": lo["day"].to_numpy(),
                                   "low": lo[col].to_numpy(), "low_year": lo["year"].to_numpy()}), on=["month", "day"])


def records_set(daily: pd.DataFrame, col: str, year: int, high: bool = True) -> pd.DataFrame:
    """Days of `year` whose value equalled or beat the record of their calendar day in the previous years.
    Columns: date, value, previous (record before), previous_year."""
    y = daily[pd.to_datetime(daily["date"]).dt.year == year][["date", col]].dropna()
    if y.empty:
        return pd.DataFrame(columns=["date", "value", "previous", "previous_year"])
    env = record_envelope(daily, col, year)
    dates = pd.to_datetime(y["date"])
    y = y.assign(month=dates.dt.month.to_numpy(), day=dates.dt.day.to_numpy()).merge(env, on=["month", "day"])
    key, year_key = ("high", "high_year") if high else ("low", "low_year")
    hit = y[y[col] >= y[key]] if high else y[y[col] <= y[key]]
    return pd.DataFrame({"date": pd.to_datetime(hit["date"]).to_numpy(), "value": hit[col].to_numpy(),
                         "previous": hit[key].to_numpy(), "previous_year": hit[year_key].to_numpy()})


def records_between(daily: pd.DataFrame, start, end, start_year: int, warmup_years: int = 10) -> pd.DataFrame:
    """Daily records set between `start` and `end` in every city: highest TX or lowest TN of their calendar day
    since `start_year`. The first `warmup_years` only set the records, as on the extremes page (Meehl, 2009).
    Columns: city, date, kind ("hot" or "cold"), value, previous, previous_year."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    d = daily[pd.to_datetime(daily["date"]).dt.year >= start_year]
    rows = []
    for year in range(start.year, end.year + 1):
        if year < start_year + warmup_years:
            continue
        for city, c in d.groupby("city"):
            for col, high, kind in (("tx", True, "hot"), ("tn", False, "cold")):
                r = records_set(c, col, year, high)
                r = r[(r["date"] >= start) & (r["date"] <= end)]
                rows += [{"city": city, "kind": kind, **row} for row in r.to_dict("records")]
    out = pd.DataFrame(rows, columns=["city", "kind", "date", "value", "previous", "previous_year"])
    return out.sort_values(["date", "city"]).reset_index(drop=True)


# --- The last weeks against the normal -----------------------------------------------------------
def anomalies(daily: pd.DataFrame, col: str, ref: tuple[int, int]) -> pd.DataFrame:
    """`col` minus the smoothed normal of its city and calendar day. Columns: city, date, value, normal, anomaly."""
    out = []
    for city, c in daily.groupby("city"):
        clim = doy_climatology(c, col, ref)
        doy = day_of_year(c["date"])
        normal = doy.map(clim.set_index("doy")["mean"])
        out.append(pd.DataFrame({"city": city, "date": pd.to_datetime(c["date"]).to_numpy(),
                                 "value": c[col].to_numpy(dtype=float), "normal": normal.to_numpy(dtype=float)}))
    df = pd.concat(out, ignore_index=True)
    df["anomaly"] = df["value"] - df["normal"]
    return df


def mean_series(daily: pd.DataFrame, col: str, ref: tuple[int, int], min_share: float = 0.75) -> pd.DataFrame:
    """Mean of the cities, day by day. A day where some cities are missing is still comparable: its value is the
    mean normal of all the cities plus the mean anomaly of those measured (at least `min_share` of them).
    Columns: date, value, n_cities."""
    a = anomalies(daily, col, ref)
    n_all = a["city"].nunique()
    normal = a.groupby("date")["normal"].mean()
    g = a.dropna(subset=["anomaly"]).groupby("date")["anomaly"].agg(["mean", "count"])
    g = g[g["count"] >= min_share * n_all]
    return pd.DataFrame({"date": g.index, "value": (normal.reindex(g.index) + g["mean"]).to_numpy(),
                         "n_cities": g["count"].to_numpy()}).reset_index(drop=True)


def latest(series: pd.DataFrame, clim: pd.DataFrame, days: int = 60) -> pd.DataFrame:
    """The last `days` days of a daily series (date, value) with the normal of each day (mean, p10, p90)."""
    s = series.dropna(subset=["value"]).sort_values("date")
    if s.empty:
        return pd.DataFrame(columns=["date", "value", "mean", "p10", "p90", "anomaly"])
    last = pd.to_datetime(s["date"]).max()
    s = s[pd.to_datetime(s["date"]) > last - pd.Timedelta(days=days)].copy()
    s["doy"] = day_of_year(s["date"]).to_numpy()
    s = s.merge(clim, on="doy", how="left")
    s["anomaly"] = s["value"] - s["mean"]
    return s.drop(columns="doy")


def month_so_far(series: pd.DataFrame, ref: tuple[int, int], start_year: int, min_share: float = 0.9) -> dict | None:
    """The month of the last measured day, from its 1st to that day: its mean, its gap to the normal (the same days
    of the years of `ref`), and its rank among the same days of every year since `start_year` (1 = warmest).
    A year counts if at least `min_share` of those days were measured."""
    s = series.dropna(subset=["value"])
    if s.empty:
        return None
    dates = pd.to_datetime(s["date"])
    last = dates.max()
    same_days = (dates.dt.month == last.month) & (dates.dt.day <= last.day) & (dates.dt.year >= start_year)
    m = s[same_days.to_numpy()].assign(year=dates[same_days].dt.year.to_numpy())
    g = m.groupby("year")["value"].agg(["mean", "count"])
    g = g[g["count"] >= min_share * last.day]
    if last.year not in g.index:
        return None
    means = g["mean"]
    normal = means[(means.index >= ref[0]) & (means.index <= ref[1])].mean()
    value = float(means[last.year])
    return {"year": int(last.year), "month": int(last.month), "month_name": month_name(last.month),
            "last_day": int(last.day), "last_date": last, "value": value, "normal": float(normal),
            "anomaly": value - float(normal), "rank_warm": int((means > value).sum() + 1),
            "rank_cold": int((means < value).sum() + 1), "n_years": int(len(means)),
            "first_year": int(means.index.min())}


