"""Pure statistical functions (no I/O), covered by unit tests."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy import stats

SEASONS = {12: "Winter", 1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring", 5: "Spring",
           6: "Summer", 7: "Summer", 8: "Summer", 9: "Autumn", 10: "Autumn", 11: "Autumn"}
SEASON_ORDER = ["Winter", "Spring", "Summer", "Autumn"]


@dataclass
class Trend:
    """Linear trend expressed per decade, with two complementary tests."""
    slope: float          # OLS slope (unit per decade)
    ci_low: float         # 95% confidence interval of the OLS slope
    ci_high: float
    p_value: float        # Student t-test on the OLS slope
    r2: float
    sen_slope: float      # Theil-Sen slope (robust to outliers)
    mk_tau: float         # Kendall's tau (Mann-Kendall test, non-parametric)
    mk_p_value: float
    n: int

    @property
    def significant(self) -> bool:
        return self.mk_p_value < 0.05 and self.p_value < 0.05

    def to_dict(self) -> dict:
        d = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in asdict(self).items()}
        d["significant"] = self.significant
        return d


def trend(x, y, per: float = 10.0) -> Trend:
    """Trend of y against x (years), expressed per `per` units of x."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    if len(x) < 10:
        raise ValueError(f"Series too short for a trend ({len(x)} points)")
    ols = stats.linregress(x, y)
    t_crit = stats.t.ppf(0.975, len(x) - 2)
    sen = stats.theilslopes(y, x)
    tau, mk_p = stats.kendalltau(x, y)
    return Trend(
        slope=ols.slope * per,
        ci_low=(ols.slope - t_crit * ols.stderr) * per,
        ci_high=(ols.slope + t_crit * ols.stderr) * per,
        p_value=float(ols.pvalue),
        r2=float(ols.rvalue**2),
        sen_slope=float(sen.slope) * per,
        mk_tau=float(tau),
        mk_p_value=float(mk_p),
        n=int(len(x)),
    )


def anomalies(df: pd.DataFrame, cols: list[str], ref: tuple[int, int],
              by: str | list[str] = "city", time: str = "year", relative: bool = False) -> pd.DataFrame:
    """Difference between each value and the mean of its group over the reference period.

    relative=True returns a difference in % (useful for rainfall, which varies a lot between cities).
    """
    in_ref = df[time].between(*ref)
    normals = df[in_ref].groupby(by)[cols].mean().add_suffix("__ref")
    out = df.join(normals, on=by)
    for c in cols:
        base = out[f"{c}__ref"]
        out[f"{c}_anom"] = (out[c] / base - 1) * 100 if relative else out[c] - base
    return out.drop(columns=normals.columns)


def composite(df: pd.DataFrame, col: str, time: str = "year", min_share: float = 0.8) -> pd.Series:
    """Mean of an anomaly across cities, for each time step.

    Averaging anomalies (rather than raw values) keeps the indicator stable when a city
    is missing a year. A time step is dropped when fewer than `min_share` of the cities
    have data.
    """
    n_total = df["city"].nunique()
    g = df.groupby(time)[col]
    res = g.mean()
    return res[g.count() >= min_share * n_total]


def seasonal(monthly: pd.DataFrame, col: str = "tmean") -> pd.DataFrame:
    """Seasonal means. Winter N covers December N-1, January and February N."""
    m = monthly[monthly["valid_temp"]].copy()
    m["season"] = m["month"].map(SEASONS)
    m["season_year"] = m["year"] + (m["month"] == 12).astype(int)
    g = m.groupby(["city", "season_year", "season"])[col].agg(["mean", "count"]).reset_index()
    g = g[g["count"] == 3].drop(columns="count").rename(columns={"mean": col, "season_year": "year"})
    return g


def period_mean(df: pd.DataFrame, col: str, period: tuple[int, int], by: str = "city") -> pd.Series:
    return df[df["year"].between(*period)].groupby(by)[col].mean()


def record_counts(daily: pd.DataFrame, start_year: int, warmup_years: int = 10) -> pd.DataFrame:
    """Count the daily records broken each year (method of Meehl et al., 2009).

    For each city and calendar day, a hot record is a TX higher than every previous TX on
    that date; a cold record is a TN lower than every previous TN. In a stable climate there
    would be as many hot records as cold ones. The first `warmup_years` only initialise
    the records.
    """
    d = daily[["city", "date", "tn", "tx"]].copy()
    d = d[~((d["date"].dt.month == 2) & (d["date"].dt.day == 29))]
    d["md"] = d["date"].dt.strftime("%m-%d")
    d["year"] = d["date"].dt.year
    d = d.sort_values(["city", "md", "year"])
    keys = ["city", "md"]
    for col, fn, flag in (("tx", "cummax", "hot"), ("tn", "cummin", "cold")):
        running = getattr(d.groupby(keys)[col], fn)()
        running = running.groupby([d["city"], d["md"]]).ffill()
        previous = running.groupby([d["city"], d["md"]]).shift()
        d[flag] = (d[col] > previous) if flag == "hot" else (d[col] < previous)
    d = d[d["year"] >= start_year + warmup_years]
    return d.groupby("year")[["hot", "cold"]].sum().astype(int)


def daily_outliers(daily: pd.DataFrame, ref: tuple[int, int], col: str = "tmean", top: int = 10):
    """Most unusual days compared with the normal for that day (z-score).

    The normal and standard deviation of each day of the year are smoothed over ±15 days.
    """
    d = daily[["city", "date", col]].dropna().copy()
    d["doy"] = d["date"].dt.day_of_year.clip(upper=365)
    ref_mask = d["date"].dt.year.between(*ref)
    clim = d[ref_mask].groupby(["city", "doy"])[col].agg(["mean", "std"]).reset_index()

    def smooth(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("doy")
        for c in ("mean", "std"):
            ext = pd.concat([g[c].iloc[-15:], g[c], g[c].iloc[:15]])  # wrap around the year
            g[c] = ext.rolling(31, center=True).mean().iloc[15:-15].to_numpy()
        return g

    clim = pd.concat([smooth(g) for _, g in clim.groupby("city")])
    d = d.merge(clim, on=["city", "doy"])
    d["anomaly"] = d[col] - d["mean"]
    d["z"] = d["anomaly"] / d["std"]
    cols = ["city", "date", col, "mean", "anomaly", "z"]
    hot = d.nlargest(top, "z")[cols]
    cold = d.nsmallest(top, "z")[cols]
    return hot, cold, d
