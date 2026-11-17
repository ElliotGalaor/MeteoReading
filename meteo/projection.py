"""Projection of the coming days from past measurements only: a statistical outlook, NOT a weather forecast.

It knows nothing of the weather on its way (fronts, storms, heat waves announced by the forecasters). For each
city it combines what the past says about the season with how long a warm or cold spell usually lasts:

- normal: the seasonal cycle (3 harmonics) plus a linear trend, fitted on the last 30 years of daily values and
  extrapolated to the projected day, so that it includes the warming of recent decades;
- anomaly: the gap between the last measured day and its normal fades over the days ahead at the pace measured
  in the past. For each lead time h, a regression of the anomaly h days later on the anomaly of the day, over the
  days of the same season (±45 days) of those 30 years;
- range: the 10th and 90th percentiles of that regression's errors, i.e. 80% of past cases;
- rain: the share of rainy days h days after a rainy (or a dry) day, in the same season of those 30 years, and
  the median amount of the rainy days at that time of year.

Lead times are counted from the last measured day: the data is published with a delay of a few days, so the
first projected days may already be past.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

YEARS = 30            # length of the training period
SEASON_DAYS = 45      # half-width of the seasonal window of the anomaly regressions
NORMAL_DAYS = 15      # half-width of the window of the rain normals (share of rainy days, typical amount)
HARMONICS = 3
LOW, HIGH = 0.10, 0.90
ORIGIN = pd.Timestamp("2000-01-01")


def design(dates: pd.DatetimeIndex) -> np.ndarray:
    """Columns of the normal: constant, trend (per decade) and the harmonics of the year."""
    t = np.asarray((dates - ORIGIN).days, dtype=float) / 3652.5
    w = 2 * np.pi * np.asarray(dates.day_of_year, dtype=float) / 365.25
    cols = [np.ones_like(t), t]
    for k in range(1, HARMONICS + 1):
        cols += [np.cos(k * w), np.sin(k * w)]
    return np.column_stack(cols)


def fit_normal(s: pd.Series) -> np.ndarray:
    s = s.dropna()
    coef, *_ = np.linalg.lstsq(design(s.index), s.to_numpy(dtype=float), rcond=None)
    return coef


def normal(dates: pd.DatetimeIndex, coef: np.ndarray) -> np.ndarray:
    return design(dates) @ coef


def season_gap(dates: pd.DatetimeIndex, ref: pd.Timestamp) -> np.ndarray:
    """Distance in days between each date and `ref` in the calendar, whatever the year (0 to 182)."""
    return np.abs((np.asarray(dates.day_of_year) - ref.day_of_year + 182) % 365 - 182)


def daily_frame(daily: pd.DataFrame) -> pd.DataFrame:
    """One row per day (date index, missing days as NaN), columns tx, tn, rr."""
    d = daily[["date", "tx", "tn", "rr"]].copy()
    d["date"] = pd.to_datetime(d["date"])
    return d.set_index("date").sort_index().asfreq("D")


def _lagged_pairs(values: np.ndarray, base: np.ndarray, h: int) -> tuple[np.ndarray, np.ndarray]:
    """Indices i in `base` with a value at i and at i + h."""
    i = base[base + h < len(values)]
    i = i[~np.isnan(values[i]) & ~np.isnan(values[i + h])]
    return i, i + h


@dataclass(frozen=True)
class TempModel:
    """Normal and anomaly regressions of one variable, trained up to `anchor` (the last measured day)."""
    coef: np.ndarray
    anchor: pd.Timestamp
    anomaly: float                                  # anomaly of the anchor day
    fits: dict[int, tuple[float, float, np.ndarray]]  # lead -> (intercept, slope, errors of the regression)


def temp_model(s: pd.Series, anchor: pd.Timestamp, leads: range) -> TempModel:
    train = s[(s.index > anchor - pd.DateOffset(years=YEARS)) & (s.index <= anchor)]
    coef = fit_normal(train)
    a = train.to_numpy(dtype=float) - normal(train.index, coef)
    base = np.where(season_gap(train.index, anchor) <= SEASON_DAYS)[0]
    fits = {}
    for h in leads:
        i, j = _lagged_pairs(a, base, h)
        b, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(i)), a[i]]), a[j], rcond=None)
        fits[h] = (float(b[0]), float(b[1]), a[j] - (b[0] + b[1] * a[i]))
    return TempModel(coef, anchor, float(a[-1]), fits)


def project_temp(model: TempModel, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Central value, 80% range and normal of each date; `samples` = the value plus each past error."""
    rows = []
    for date, norm in zip(dates, normal(dates, model.coef), strict=True):
        b0, b1, err = model.fits[(date - model.anchor).days]
        value = norm + b0 + b1 * model.anomaly
        rows.append((date, value, value + np.quantile(err, LOW), value + np.quantile(err, HIGH), norm, value + err))
    return pd.DataFrame(rows, columns=["date", "value", "low", "high", "normal", "samples"])


def project_rain(rr: pd.Series, anchor: pd.Timestamp, dates: pd.DatetimeIndex, rain_day: float) -> pd.DataFrame:
    """Chance of a rainy day after a day like the anchor day, the usual chance and the typical amount."""
    train = rr[(rr.index > anchor - pd.DateOffset(years=YEARS)) & (rr.index <= anchor)]
    v = train.to_numpy(dtype=float)
    wet = np.where(np.isnan(v), np.nan, (v >= rain_day).astype(float))
    state = wet[-1]
    base = np.where((season_gap(train.index, anchor) <= SEASON_DAYS) & (wet == state))[0]
    rows = []
    for date in dates:
        i, j = _lagged_pairs(wet, base, (date - anchor).days)
        near = season_gap(train.index, date) <= NORMAL_DAYS
        usual = np.nanmean(wet[near])
        amounts = v[near & (wet == 1)]
        rows.append((date, wet[j].mean(), usual, float(np.median(amounts)) if len(amounts) else 0.0))
    return pd.DataFrame(rows, columns=["date", "p_rain", "p_rain_normal", "rain_mm"])


@dataclass(frozen=True)
class Projection:
    anchor: pd.Timestamp       # last day with TX, TN and rainfall
    days: pd.DataFrame         # one row per projected day, from the day after the anchor
    recent: pd.DataFrame       # the last measured days, with their normals


def project(daily: pd.DataFrame, until: pd.Timestamp, thresholds: dict, recent_days: int = 21) -> Projection:
    """Projection of every day from the last measured day to `until`, for one city."""
    d = daily_frame(daily)
    anchor = d.dropna(subset=["tx", "tn", "rr"]).index.max()
    until = max(pd.Timestamp(until), anchor + pd.Timedelta(days=1))
    dates = pd.date_range(anchor + pd.Timedelta(days=1), until, freq="D")
    leads = range(1, len(dates) + 1)

    days = pd.DataFrame({"date": dates, "lead": list(leads)})
    recent = d.loc[anchor - pd.Timedelta(days=recent_days - 1):anchor].reset_index()
    for var in ("tx", "tn"):
        model = temp_model(d[var], anchor, leads)
        p = project_temp(model, dates)
        days[[var, f"{var}_low", f"{var}_high", f"{var}_normal"]] = p[["value", "low", "high", "normal"]].to_numpy()
        days[f"_{var}_samples"] = p["samples"]
        days[f"{var}_kept"] = [model.fits[h][1] for h in leads]   # share of the anchor's gap still there
        recent[f"{var}_normal"] = normal(pd.DatetimeIndex(recent["date"]), model.coef)
    # Chances of frost and of a hot day: the share of past errors that would have crossed the threshold
    days["p_frost"] = [float((s < thresholds["frost_day"]).mean()) for s in days.pop("_tn_samples")]
    days["p_hot"] = [float((s >= thresholds["hot_day"]).mean()) for s in days.pop("_tx_samples")]
    rain = project_rain(d["rr"], anchor, dates, thresholds["rain_day"])
    days[["p_rain", "p_rain_normal", "rain_mm"]] = rain[["p_rain", "p_rain_normal", "rain_mm"]].to_numpy()
    return Projection(anchor, days, recent)


def skill(daily: pd.DataFrame, var: str = "tx", max_lead: int = 10, test_days: int = 730) -> pd.DataFrame:
    """Out-of-sample check: mean absolute error of the projection and of the normal alone, by lead time.

    Trained on the 30 years before the test period (the last `test_days` days), then run from every day of the
    test period; the regressions are fitted for each month of the year, as the projection fits them for its
    season."""
    s = daily_frame(daily)[var]
    end = s.last_valid_index()
    split = end - pd.Timedelta(days=test_days)
    train = s[(s.index > split - pd.DateOffset(years=YEARS)) & (s.index <= split)]
    coef = fit_normal(train)
    a = s.to_numpy(dtype=float) - normal(s.index, coef)
    in_train = (s.index > train.index[0]) & (s.index <= split - pd.Timedelta(days=max_lead))
    in_test = s.index > split
    errors = {h: ([], []) for h in range(1, max_lead + 1)}
    for month in range(1, 13):
        mid = pd.Timestamp(2001, month, 15)
        base = np.where(in_train & (season_gap(s.index, mid) <= SEASON_DAYS))[0]
        test = np.where(in_test & (s.index.month == month))[0]
        for h in errors:
            i, j = _lagged_pairs(a, base, h)
            b, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(i)), a[i]]), a[j], rcond=None)
            ti, tj = _lagged_pairs(a, test, h)
            errors[h][0].extend(np.abs(b[0] + b[1] * a[ti] - a[tj]))
            errors[h][1].extend(np.abs(a[tj]))
    return pd.DataFrame([(h, np.mean(p), np.mean(n), len(p)) for h, (p, n) in errors.items()],
                        columns=["lead", "projection", "normal", "n"])
