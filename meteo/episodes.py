"""Notable episodes (heatwaves, cold waves, dry spells) and the weather of one day in context.

Heatwaves and cold waves follow the method of Météo-France's heatwave and cold-wave indicators (Ouzeau et al., 2016,
"Heat waves analysis over France in present and future climate"), applied to each city's daily mean temperature
instead of the national indicator:
- three thresholds, percentiles of the daily mean over the reference period: Spic, Sdeb and Sint, the 99.5th, 97.5th
  and 95th for heat, the 0.5th, 2.5th and 5th for cold;
- an episode is a run of days at or beyond Sdeb that reaches Spic at least once. It carries on through up to 3 days
  that fall short of Sdeb but stay beyond Sint (a slightly cooler day in the middle of a heatwave);
- its intensity is the sum, over its days, of the distance between the daily mean and Sdeb (degree-days).
Dry spells are runs of days with less than the rainy-day threshold (1 mm); their intensity is their length.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .climatology import MONTH_NAMES, day_of_year, doy_climatology, ordinal

HEAT_PERCENTILES = (99.5, 97.5, 95.0)   # Spic, Sdeb, Sint
COLD_PERCENTILES = (0.5, 2.5, 5.0)
MAX_BREAK_DAYS = 3
MIN_DRY_DAYS = 20                       # dry spells listed: at least this long


def thresholds(daily: pd.DataFrame, ref: tuple[int, int], heat: bool = True) -> tuple[float, float, float]:
    """(Spic, Sdeb, Sint) from the daily mean temperatures of the reference period."""
    t = daily[pd.to_datetime(daily["date"]).dt.year.between(*ref)]["tmean"].dropna().to_numpy()
    return tuple(float(np.percentile(t, q)) for q in (HEAT_PERCENTILES if heat else COLD_PERCENTILES))


def detect(dates, values, spic: float, sdeb: float, sint: float, heat: bool = True,
           max_break: int = MAX_BREAK_DAYS) -> pd.DataFrame:
    """Episodes in a daily series (one value per consecutive day, NaN when missing).
    Columns: start, end, n_days, peak, peak_date, intensity."""
    dates = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
    sign = 1.0 if heat else -1.0
    v = sign * np.asarray(values, dtype=float)
    spic, sdeb, sint = sign * spic, sign * sdeb, sign * sint
    deb = np.nan_to_num(v, nan=-np.inf) >= sdeb
    intm = np.nan_to_num(v, nan=-np.inf) >= sint
    runs, start = [], None
    for i, m in enumerate(np.append(deb, False)):
        if m and start is None:
            start = i
        elif not m and start is not None:
            runs.append([start, i - 1])
            start = None
    merged = []
    for r in runs:   # join runs separated by a short break that stays beyond Sint
        if merged and r[0] - merged[-1][1] - 1 <= max_break and intm[merged[-1][1] + 1:r[0]].all():
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    rows = []
    for a, b in merged:
        seg = v[a:b + 1]
        if np.nanmax(seg) < spic:
            continue
        i_peak = a + int(np.nanargmax(seg))
        rows.append({"start": dates[a], "end": dates[b], "n_days": b - a + 1, "peak": sign * v[i_peak],
                     "peak_date": dates[i_peak], "intensity": float(np.nansum(seg - sdeb))})
    return pd.DataFrame(rows, columns=["start", "end", "n_days", "peak", "peak_date", "intensity"])


def episodes(daily: pd.DataFrame, ref: tuple[int, int], heat: bool = True) -> pd.DataFrame:
    """Heatwaves (or cold waves) of every city, most intense first. Adds city, the thresholds and the kind."""
    out = []
    for city, c in daily.groupby("city"):
        c = c.sort_values("date")
        spic, sdeb, sint = thresholds(c, ref, heat)
        e = detect(c["date"], c["tmean"], spic, sdeb, sint, heat)
        e.insert(0, "city", city)
        e["threshold"] = sdeb
        out.append(e)
    df = pd.concat(out, ignore_index=True) if out else pd.DataFrame()
    df["kind"] = "heatwave" if heat else "cold wave"
    return df.sort_values("intensity", ascending=False).reset_index(drop=True)


def dry_spells(spells: pd.DataFrame, yearly: pd.DataFrame, ref: tuple[int, int],
               min_days: int = MIN_DRY_DAYS) -> pd.DataFrame:
    """The long runs without rain (table city_dry_spells: city, start_date, end_date, n_days), most unusual first.

    Summers are always dry around the Mediterranean, so a spell is ranked by its length divided by the city's usual
    longest spell of a year (the median over the reference period, from city_yearly): `intensity` 2 = twice as long
    as the longest spell of a normal year."""
    usual = yearly[yearly["year"].between(*ref)].groupby("city")["longest_dry_spell"].median()
    s = spells[spells["n_days"] >= min_days].copy()
    s["usual"] = s["city"].map(usual).to_numpy(dtype=float)
    s["intensity"] = s["n_days"] / s["usual"]
    s["peak_date"] = pd.to_datetime(s["end_date"])
    return s.sort_values(["intensity", "start_date"], ascending=[False, True]).reset_index(drop=True)


# --- One day in context ---------------------------------------------------------------------------------
VARIABLES = {   # column -> (label, unit, scale to the unit shown, word for a high rank, word for a low rank)
    "tx": ("Maximum temperature", "°C", 1.0, "hottest", "coolest"),
    "tn": ("Minimum temperature", "°C", 1.0, "warmest", "coldest"),
    "rr": ("Rainfall", "mm", 1.0, "wettest", "driest"),
    "ffm": ("Mean wind", "km/h", 3.6, "windiest", "calmest"),
    "fxi": ("Strongest gust", "km/h", 3.6, "windiest", "calmest"),
}


def same_day(daily: pd.DataFrame, day) -> pd.DataFrame:
    """The rows of the same calendar day (month and day) in every year."""
    day = pd.Timestamp(day)
    dates = pd.to_datetime(daily["date"])
    return daily[((dates.dt.month == day.month) & (dates.dt.day == day.day)).to_numpy()]


def day_in_context(daily: pd.DataFrame, day) -> dict:
    """Each value of one day in one city (km/h for wind), its rank among the same calendar day of every year, and
    the records of that calendar day. {col: {value, rank, n, word, record, record_year}}."""
    day = pd.Timestamp(day)
    same = same_day(daily, day)
    dates = pd.to_datetime(same["date"])
    row = same[(dates == day).to_numpy()]
    out = {}
    for col, (label, unit, scale, high, low) in VARIABLES.items():
        values = same[col].astype(float) * scale
        years = dates.dt.year
        ok = values.notna().to_numpy()
        v = float(row[col].iloc[0]) * scale if len(row) and pd.notna(row[col].iloc[0]) else None
        info = {"label": label, "unit": unit, "value": v, "n": int(ok.sum())}
        if ok.any():
            i_max = int(np.argmax(np.where(ok, values, -np.inf)))
            info.update(record=float(values.iloc[i_max]), record_year=int(years.iloc[i_max]))
            if col == "tn":
                i_min = int(np.argmin(np.where(ok, values, np.inf)))
                info.update(record=float(values.iloc[i_min]), record_year=int(years.iloc[i_min]))
        if v is not None:
            higher, lower = int((values > v).sum()), int((values < v).sum())
            # Ranked from the end it is closest to: "3rd hottest" or "2nd coolest"
            if higher <= lower:
                info.update(rank=higher + 1, word=high)
            else:
                info.update(rank=lower + 1, word=low)
            if col == "rr" and v < 1:
                info.update(rank=None, word="dry")
        out[col] = info
    return out


def rank_text(info: dict, day) -> str:
    """'3rd hottest 15 August of 75 years'."""
    day = pd.Timestamp(day)
    if info.get("value") is None:
        return "not measured"
    if info.get("word") == "dry":
        return "a dry day"
    return f"{ordinal(info['rank'])} {info['word']} {day.day} {MONTH_NAMES[day.month - 1]} of {info['n']} years"


def days_around(daily: pd.DataFrame, day, ref: tuple[int, int], half: int = 10) -> pd.DataFrame:
    """The days around `day` with the normals of TX and TN (smoothed over ±15 days)."""
    day = pd.Timestamp(day)
    dates = pd.to_datetime(daily["date"])
    win = daily[((dates >= day - pd.Timedelta(days=half)) & (dates <= day + pd.Timedelta(days=half))).to_numpy()].copy()
    win["doy"] = day_of_year(win["date"]).to_numpy()
    for col in ("tx", "tn"):
        clim = doy_climatology(daily, col, ref).set_index("doy")["mean"]
        win[f"{col}_normal"] = win["doy"].map(clim).to_numpy()
    return win.drop(columns="doy")
