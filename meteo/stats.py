"""Fonctions statistiques pures (sans entrée/sortie), testées unitairement."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy import stats

SEASONS = {12: "Hiver", 1: "Hiver", 2: "Hiver", 3: "Printemps", 4: "Printemps", 5: "Printemps",
           6: "Été", 7: "Été", 8: "Été", 9: "Automne", 10: "Automne", 11: "Automne"}
SEASON_ORDER = ["Hiver", "Printemps", "Été", "Automne"]


@dataclass
class Trend:
    """Tendance linéaire exprimée par décennie, avec deux tests complémentaires."""
    slope: float          # pente OLS (unité / décennie)
    ci_low: float         # intervalle de confiance à 95 % de la pente OLS
    ci_high: float
    p_value: float        # test de Student sur la pente OLS
    r2: float
    sen_slope: float      # pente de Theil-Sen (robuste aux valeurs extrêmes)
    mk_tau: float         # tau de Kendall (test de Mann-Kendall, non paramétrique)
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
    """Tendance de y en fonction de x (années), exprimée pour `per` unités de x."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    if len(x) < 10:
        raise ValueError(f"Série trop courte pour une tendance ({len(x)} points)")
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
    """Écart de chaque valeur à la moyenne de son groupe sur la période de référence.

    relative=True renvoie un écart en % (utile pour la pluie, très différente d'une ville à l'autre).
    """
    in_ref = df[time].between(*ref)
    normals = df[in_ref].groupby(by)[cols].mean().add_suffix("__ref")
    out = df.join(normals, on=by)
    for c in cols:
        base = out[f"{c}__ref"]
        out[f"{c}_anom"] = (out[c] / base - 1) * 100 if relative else out[c] - base
    return out.drop(columns=normals.columns)


def composite(df: pd.DataFrame, col: str, time: str = "year", min_share: float = 0.8) -> pd.Series:
    """Moyenne entre villes d'une anomalie, pour chaque pas de temps.

    On moyenne des anomalies (et non des valeurs brutes) : l'indicateur ne bouge pas
    si une ville manque une année. Un pas de temps est ignoré si moins de
    `min_share` des villes sont disponibles.
    """
    n_total = df["city"].nunique()
    g = df.groupby(time)[col]
    res = g.mean()
    return res[g.count() >= min_share * n_total]


def seasonal(monthly: pd.DataFrame, col: str = "tmean") -> pd.DataFrame:
    """Moyennes saisonnières. L'hiver N regroupe décembre N-1, janvier et février N."""
    m = monthly[monthly["valid_temp"]].copy()
    m["season"] = m["month"].map(SEASONS)
    m["season_year"] = m["year"] + (m["month"] == 12).astype(int)
    g = m.groupby(["city", "season_year", "season"])[col].agg(["mean", "count"]).reset_index()
    g = g[g["count"] == 3].drop(columns="count").rename(columns={"mean": col, "season_year": "year"})
    return g


def period_mean(df: pd.DataFrame, col: str, period: tuple[int, int], by: str = "city") -> pd.Series:
    return df[df["year"].between(*period)].groupby(by)[col].mean()


def record_counts(daily: pd.DataFrame, start_year: int, warmup_years: int = 10) -> pd.DataFrame:
    """Compte les records quotidiens battus chaque année (méthode de Meehl et al., 2009).

    Pour chaque ville et chaque jour calendaire, un record chaud est un TX supérieur à tous
    les TX des années précédentes à la même date ; un record froid, un TN inférieur à tous
    les TN précédents. Dans un climat stable, il y aurait autant de records chauds que froids.
    Les `warmup_years` premières années servent seulement à initialiser les records.
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
    """Journées les plus anormales par rapport à la normale du jour (score z).

    La normale et l'écart-type de chaque jour de l'année sont lissés sur ±15 jours.
    """
    d = daily[["city", "date", col]].dropna().copy()
    d["doy"] = d["date"].dt.day_of_year.clip(upper=365)
    ref_mask = d["date"].dt.year.between(*ref)
    clim = d[ref_mask].groupby(["city", "doy"])[col].agg(["mean", "std"]).reset_index()

    def smooth(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("doy")
        for c in ("mean", "std"):
            ext = pd.concat([g[c].iloc[-15:], g[c], g[c].iloc[:15]])  # année circulaire
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
