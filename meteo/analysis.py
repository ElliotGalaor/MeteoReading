"""Analyses : chaque fonction répond à une question métier et renvoie des chiffres clés.

Q1. De combien la France s'est-elle réchauffée depuis 1951, et est-ce significatif ?
Q2. Le réchauffement est-il homogène entre villes, saisons, minimales et maximales ?
Q3. Les extrêmes (chaleur, nuits tropicales, gel) évoluent-ils ?
Q4. Les records de chaleur sont-ils plus fréquents que les records de froid ?
Q5. Les précipitations ont-elles changé (cumul, fortes pluies, sécheresses) ?
Q6. Quels profils climatiques distinguent les villes (ACP + clustering) ?
Q7. Quelles journées ont été les plus anormales ?
+  Qualité des données et représentativité du réseau.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from . import viz
from .config import Config
from .stats import SEASON_ORDER, anomalies, composite, daily_outliers, period_mean, record_counts, seasonal, trend
from .warehouse import connect

log = logging.getLogger(__name__)


def load(cfg: Config) -> dict[str, pd.DataFrame]:
    with connect(cfg) as con:
        tables = {t: con.sql(f"SELECT * FROM {t}").df() for t in (
            "city_daily", "city_monthly", "city_yearly", "ref_station", "station_offsets",
            "dq_summary", "network_yearly", "city_heatwaves")}
    # Analyse limitée aux années complètes (l'année en cours reste dans l'entrepôt et le dashboard)
    a = cfg.analysis
    d = tables["city_daily"]
    d["date"] = pd.to_datetime(d["date"])
    tables["city_daily"] = d[d["date"].dt.year.between(a["start_year"], a["end_year"])].reset_index(drop=True)
    for name in ("city_monthly", "city_yearly"):
        t = tables[name]
        tables[name] = t[t["year"].between(a["start_year"], a["end_year"])].reset_index(drop=True)
    hw = tables["city_heatwaves"]
    tables["city_heatwaves"] = hw[pd.to_datetime(hw["start_date"]).dt.year <= a["end_year"]]
    return tables


def _r(x, n=2):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), n)


# --- Q1 ----------------------------------------------------------------------
def warming(t, cfg, figs) -> dict:
    a = cfg.analysis
    ref, old, recent = tuple(a["reference_period"]), tuple(a["old_period"]), tuple(a["recent_period"])
    y = t["city_yearly"][t["city_yearly"]["valid_temp"]]
    y = anomalies(y, ["tmean", "tn", "tx"], ref)
    fr = composite(y, "tmean_anom")
    tr = trend(fr.index, fr.values)
    tr_recent = trend(fr.loc[1981:].index, fr.loc[1981:].values)

    viz.warming_stripes(fr, figs / "01_warming_stripes.png", ref)
    viz.anomaly_bars(fr, figs / "02_anomalie_france.png", tr, ref)
    viz.anomaly_small_multiples(y, figs / "03_anomalies_par_ville.png", ref)

    old_mean, recent_mean = fr.loc[old[0]:old[1]].mean(), fr.loc[recent[0]:recent[1]].mean()
    top = fr.sort_values(ascending=False)
    return {
        "france_trend_1951": tr.to_dict(),
        "france_trend_1981": tr_recent.to_dict(),
        "delta_recent_vs_old": _r(recent_mean - old_mean),
        "warmest_years": [{"year": int(k), "anomaly": _r(v)} for k, v in top.head(5).items()],
        "n_top10_since_2011": int((top.head(10).index >= 2011).sum()),
        "coldest_year": {"year": int(top.index[-1]), "anomaly": _r(top.iloc[-1])},
        "tmax_trend": trend(*_xy(composite(y, "tx_anom"))).to_dict(),
        "tmin_trend": trend(*_xy(composite(y, "tn_anom"))).to_dict(),
    }, y, fr


def _decade_label(d: int, start: int, end: int) -> str:
    """1950 -> '1951-59' si l'analyse démarre en 1951 ; 2020 -> '2020-25'."""
    return f"{max(d, start)}-{str(min(d + 9, end))[2:]}"


def _xy(s: pd.Series):
    return s.index.values, s.values


# --- Q2 ----------------------------------------------------------------------
def heterogeneity(t, cfg, figs, y) -> tuple[dict, pd.DataFrame]:
    a = cfg.analysis
    old, recent = tuple(a["old_period"]), tuple(a["recent_period"])
    rows = []
    for city, d in y.groupby("city"):
        tr = trend(d["year"], d["tmean"])
        rows.append({"city": city, **{k: tr.to_dict()[k] for k in ("slope", "ci_low", "ci_high", "mk_p_value", "significant")},
                     "tmean_old": period_mean(d, "tmean", old).iloc[0],
                     "tmean_recent": period_mean(d, "tmean", recent).iloc[0]})
    cities = pd.DataFrame(rows)
    cities["delta"] = cities["tmean_recent"] - cities["tmean_old"]
    viz.city_trends(cities, figs / "04_tendance_par_ville.png")

    s = seasonal(t["city_monthly"])
    s = s[s["year"].between(a["start_year"], a["end_year"])]
    season_tr = (s.groupby(["city", "season"])
                 .apply(lambda g: trend(g["year"], g["tmean"]).slope, include_groups=False)
                 .unstack()[SEASON_ORDER])
    s_anom = anomalies(s, ["tmean"], tuple(a["reference_period"]), by=["city", "season"])
    fr_season = {season: trend(*_xy(composite(g, "tmean_anom"))).to_dict()
                 for season, g in s_anom.groupby("season")}
    fastest = max(fr_season, key=lambda k: fr_season[k]["slope"])
    titles = {"Été": "L'été se réchauffe le plus vite", "Printemps": "Le printemps se réchauffe le plus vite",
              "Hiver": "L'hiver se réchauffe le plus vite", "Automne": "L'automne se réchauffe le plus vite"}
    viz.season_heatmap(season_tr.sort_values("Été", ascending=False), figs / "05_tendance_saisons.png", titles[fastest])

    return {
        "cities": cities.round(3).to_dict(orient="records"),
        "fastest_city": cities.loc[cities["slope"].idxmax(), "city"],
        "slowest_city": cities.loc[cities["slope"].idxmin(), "city"],
        "n_significant": int(cities["significant"].sum()),
        "season_trends": {k: fr_season[k] for k in SEASON_ORDER},
        "fastest_season": fastest,
        "season_by_city": season_tr.round(3).reset_index().to_dict(orient="records"),
    }, cities


# --- Q3 ----------------------------------------------------------------------
def extremes(t, cfg, figs) -> dict:
    a, th = cfg.analysis, cfg.thresholds
    old, recent = tuple(a["old_period"]), tuple(a["recent_period"])
    y = t["city_yearly"][t["city_yearly"]["valid_temp"]].copy()
    y["decade"] = y["year"] // 10 * 10
    labels = {
        "hot_days": f"Journées chaudes (TX ≥ {th['hot_day']} °C)",
        "very_hot_days": f"Journées très chaudes (TX ≥ {th['very_hot_day']} °C)",
        "tropical_nights": f"Nuits tropicales (TN ≥ {th['tropical_night']} °C)",
        "frost_days": f"Jours de gel (TN < {th['frost_day']} °C)",
    }
    by_decade = y.groupby("decade")[list(labels) + ["heatwave_days"]].mean()
    by_decade.index = [_decade_label(d, a["start_year"], a["end_year"]) for d in by_decade.index]
    hot_up = by_decade["hot_days"].iloc[-1] > by_decade["hot_days"].iloc[0]
    frost_down = by_decade["frost_days"].iloc[-1] < by_decade["frost_days"].iloc[0]
    title = ("Plus de chaleur extrême, beaucoup moins de gel" if hot_up and frost_down
             else "Évolution des jours extrêmes")
    viz.extremes_by_decade(by_decade, figs / "06_extremes_decennies.png", labels, title)

    per_city = pd.DataFrame({
        "city": sorted(y["city"].unique()),
    }).set_index("city")
    for col in ("hot_days", "tropical_nights", "frost_days", "heatwave_days"):
        per_city[f"{col}_old"] = period_mean(y, col, old)
        per_city[f"{col}_recent"] = period_mean(y, col, recent)
    per_city = per_city.reset_index()
    viz.dumbbell(per_city, figs / "07_jours_chauds_par_ville.png", "hot_days_old", "hot_days_recent",
                 (f"{old[0]}-{old[1]}", f"{recent[0]}-{recent[1]}"),
                 "Les journées à 30 °C se multiplient partout",
                 f"Nombre moyen de journées avec TX ≥ {th['hot_day']} °C par an",
                 "Jours par an")

    fr = {col: (period_mean(y, col, old).mean(), period_mean(y, col, recent).mean()) for col in list(labels) + ["heatwave_days"]}
    hw = t["city_heatwaves"]
    longest = hw.sort_values(["n_days", "tx_peak"], ascending=False).head(5)
    return {
        "france_old_vs_recent": {k: {"old": _r(o, 1), "recent": _r(r, 1), "ratio": _r(r / o if o else None)} for k, (o, r) in fr.items()},
        "by_decade": by_decade.round(2).reset_index(names="decade").to_dict(orient="records"),
        "per_city": per_city.round(1).to_dict(orient="records"),
        "hottest_day": _hottest_day(t["city_daily"]),
        "longest_heatwaves": [{"city": r.city, "start": str(r.start_date)[:10], "days": int(r.n_days),
                               "tx_peak": _r(r.tx_peak, 1)} for r in longest.itertuples()],
        "hot_days_trend": trend(*_xy(composite(y, "hot_days"))).to_dict(),
        "frost_days_trend": trend(*_xy(composite(y, "frost_days"))).to_dict(),
    }


def _hottest_day(daily: pd.DataFrame) -> dict:
    r = daily.loc[daily["tx"].idxmax()]
    return {"city": r["city"], "date": r["date"].strftime("%Y-%m-%d"), "tx": _r(r["tx"], 1)}


# --- Q4 ----------------------------------------------------------------------
def records(t, cfg, figs) -> dict:
    a = cfg.analysis
    rc = record_counts(t["city_daily"][t["city_daily"]["date"].dt.year <= a["end_year"]], a["start_year"])
    rc["decade"] = (rc.index // 10 * 10)
    by_dec = rc.groupby("decade")[["hot", "cold"]].sum()
    by_dec.index = [_decade_label(d, a["start_year"] + 10, a["end_year"]) for d in by_dec.index]
    by_dec["ratio"] = by_dec["hot"] / by_dec["cold"]
    viz.records_ratio(by_dec, figs / "08_records.png")
    return {"by_decade": by_dec.round(2).reset_index(names="decade").to_dict(orient="records"),
            "last_ratio": _r(by_dec["ratio"].iloc[-1], 1)}


# --- Q5 ----------------------------------------------------------------------
def precipitation(t, cfg, figs) -> dict:
    a = cfg.analysis
    ref = tuple(a["reference_period"])
    y = t["city_yearly"][t["city_yearly"]["valid_rain"]]
    y = anomalies(y, ["rr"], ref, relative=True)
    fr = composite(y, "rr_anom")
    tr = trend(fr.index, fr.values)
    viz.precip_anomaly(fr, figs / "09_pluie_france.png", tr, ref)

    s = seasonal(t["city_monthly"].assign(valid_temp=t["city_monthly"]["valid_rain"]), "rr")
    s = s[s["year"].between(a["start_year"], a["end_year"])]
    s["rr"] = s["rr"] * 3  # moyenne mensuelle -> cumul saisonnier
    s = anomalies(s, ["rr"], ref, by=["city", "season"], relative=True)
    seasons = {k: trend(*_xy(composite(g, "rr_anom"))).to_dict() for k, g in s.groupby("season")}
    return {
        "annual_trend_pct": tr.to_dict(),
        "season_trends_pct": {k: seasons[k] for k in SEASON_ORDER},
        "heavy_rain_trend": trend(*_xy(composite(y, "heavy_rain_days"))).to_dict(),
        "dry_spell_trend": trend(*_xy(composite(y, "longest_dry_spell"))).to_dict(),
        "wettest_year": int(fr.idxmax()), "driest_year": int(fr.idxmin()),
        "driest_year_anomaly_pct": _r(fr.min(), 1),
    }


# --- Q6 ----------------------------------------------------------------------
def climate_profiles(t, cfg, figs) -> dict:
    ref = tuple(cfg.analysis["reference_period"])
    m = t["city_monthly"]
    m = m[m["year"].between(*ref)]
    normals = m.groupby(["city", "month"])[["tmean", "rr"]].mean().unstack("month")
    normals.columns = [f"{v}_{mo:02d}" for v, mo in normals.columns]
    X = StandardScaler().fit_transform(normals)

    sil = {k: float(silhouette_score(X, KMeans(k, n_init=20, random_state=42).fit_predict(X))) for k in range(2, 7)}
    k = 3  # <= 3 groupes : nombre de couleurs distinguables par tous (y compris daltoniens)
    km = KMeans(k, n_init=20, random_state=42).fit(X)
    pca = PCA(2).fit(X)
    pcs = pca.transform(X)
    if np.corrcoef(pcs[:, 0], normals.filter(like="tmean").mean(axis=1))[0, 1] < 0:
        pcs[:, 0] *= -1  # oriente l'axe 1 vers le chaud, plus lisible
    coords = pd.DataFrame({"city": normals.index, "pc1": pcs[:, 0], "pc2": pcs[:, 1], "cluster": km.labels_})

    # Nommer les groupes à partir de leurs caractéristiques moyennes
    tm = normals.filter(like="tmean_")
    rr = normals.filter(like="rr_")
    feat = pd.DataFrame({
        "t_annual": tm.mean(axis=1),
        "amplitude": tm["tmean_07"] - tm["tmean_01"],
        "summer_rain": rr[["rr_06", "rr_07", "rr_08"]].sum(axis=1),
        "annual_rain": rr.sum(axis=1),
        "cluster": km.labels_,
    })
    prof = feat.groupby("cluster").mean()
    # (règles simples et explicables ; les noms décrivent le centre de chaque groupe)
    names, remaining = {}, list(prof.index)
    med = prof.loc[remaining, "summer_rain"].idxmin()
    names[med] = "Sud : chaud, été sec"
    remaining.remove(med)
    oce = prof.loc[remaining, "amplitude"].idxmin()
    names[oce] = "Façade atlantique : très arrosée, faible amplitude"
    remaining.remove(oce)
    for c in remaining:
        names[c] = "Nord et Centre : plus frais, amplitude marquée"
    # ordre d'affichage fixe : couleur 1 = océanique, 2 = méditerranéen, 3 = semi-continental
    order = {oce: 0, med: 1, **{c: 2 for c in remaining}}
    coords["cluster"] = coords["cluster"].map(order)
    names = {order[c]: n for c, n in names.items()}
    viz.clusters_scatter(coords, figs / "10_profils_climatiques.png", names, pca.explained_variance_ratio_,
                         "Trois grandes familles de climat ressortent des données")

    clim = m.groupby(["city", "month"])["tmean"].mean().unstack()
    clim = clim.loc[clim.mean(axis=1).sort_values(ascending=False).index]
    viz.climatology_heatmap(clim, figs / "11_climatologie.png", ref)

    prof.index = [names[order[c]] for c in prof.index]
    return {
        "silhouette": {str(k): round(v, 3) for k, v in sil.items()},
        "k": k,
        "explained_variance": [round(float(v), 3) for v in pca.explained_variance_ratio_],
        "groups": {names[g]: sorted(coords.loc[coords["cluster"] == g, "city"].tolist()) for g in sorted(names)},
        "profiles": prof.round(1).reset_index(names="group").to_dict(orient="records"),
    }


# --- Q7 ----------------------------------------------------------------------
def outliers(t, cfg) -> dict:
    hot, cold, d = daily_outliers(t["city_daily"], tuple(cfg.analysis["reference_period"]))
    fmt = lambda df: [{"city": r.city, "date": r.date.strftime("%Y-%m-%d"), "tmean": _r(r.tmean, 1),
                       "normal": _r(r.mean, 1), "anomaly": _r(r.anomaly, 1), "z": _r(r.z, 1)}
                      for r in df.itertuples()]
    return {"hottest_vs_normal": fmt(hot), "coldest_vs_normal": fmt(cold),
            "n_days_z_above_3": int((d["z"].abs() > 3).sum()), "n_days_total": int(len(d))}


# --- Qualité -----------------------------------------------------------------
def data_quality(t, cfg, figs) -> dict:
    dq = t["dq_summary"].pivot_table(index="variable", columns="status", values="n", aggfunc="sum").fillna(0)
    dq["total"] = dq.sum(axis=1)
    dq["pct_absentes"] = dq["absente"] / dq["total"] * 100
    present = dq["total"] - dq["absente"]
    dq["pct_rejetees"] = (present - dq["valide"]) / present * 100
    viz.network(t["network_yearly"][t["network_yearly"]["year"] <= cfg.analysis["end_year"]],
                figs / "12_reseau_stations.png")
    off = (t["station_offsets"].groupby(["city", "station_id"])
           .agg(n_overlap_days=("n_overlap_days", "sum"), offset_tn=("offset_tn", "mean"),
                offset_tx=("offset_tx", "mean"), corr_tx=("corr_tx", "mean"))
           .round(2).reset_index())
    return {
        "by_variable": dq.astype(float).round(3).reset_index().to_dict(orient="records"),
        "ref_stations": t["ref_station"].astype({"first_date": str, "last_date": str}).round(4).to_dict(orient="records"),
        "offsets": off.to_dict(orient="records"),
    }


# -----------------------------------------------------------------------------
def run_analysis(cfg: Config) -> dict:
    figs = cfg.figures_dir
    figs.mkdir(parents=True, exist_ok=True)
    tables_dir = cfg.reports_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    t = load(cfg)
    log.info("Données chargées : %s jours-villes", f"{len(t['city_daily']):,}".replace(",", " "))

    res: dict = {}
    res["warming"], y_anom, fr = warming(t, cfg, figs)
    res["heterogeneity"], cities = heterogeneity(t, cfg, figs, y_anom)
    res["extremes"] = extremes(t, cfg, figs)
    res["records"] = records(t, cfg, figs)
    res["precipitation"] = precipitation(t, cfg, figs)
    res["profiles"] = climate_profiles(t, cfg, figs)
    res["outliers"] = outliers(t, cfg)
    res["quality"] = data_quality(t, cfg, figs)

    era5_path = tables_dir / "era5_validation.csv"
    if era5_path.exists():
        era5 = pd.read_csv(era5_path)
        bias_tx, corr_tx = era5["bias_tx"].mean(), era5["corr_tx"].mean()
        spread = max(era5["bias_tx"].std(), era5["bias_tn"].std())
        if corr_tx > 0.95 and spread > 0.8:
            title = "ERA5 suit bien la météo du jour, mais lisse les effets locaux"
        elif abs(bias_tx) > 0.3:
            title = f"La réanalyse ERA5 {'sous' if bias_tx < 0 else 'sur'}estime les maximales"
        else:
            title = "La réanalyse ERA5 colle aux stations"
        viz.era5_bias(era5, figs / "13_validation_era5.png", title, f"{cfg.era5['start'][:4]}-{cfg.era5['end'][:4]}")
        worst = era5.loc[era5["bias_tn"].abs().idxmax()]
        res["era5"] = {
            "title": title,
            "mean_bias_tx": _r(bias_tx), "mean_bias_tn": _r(era5["bias_tn"].mean()),
            "mean_corr_tx": _r(corr_tx, 3), "mean_corr_tn": _r(era5["corr_tn"].mean(), 3),
            "mean_rmse_tx": _r(era5["rmse_tx"].mean()),
            "hot_days_obs": int(era5["hot_days_obs"].sum()), "hot_days_era5": int(era5["hot_days_era5"].sum()),
            "worst_city": worst["city"], "worst_bias_tn": _r(worst["bias_tn"]),
            "cities": era5.round(3).to_dict(orient="records"),
        }

    stats_path = cfg.warehouse.parent / "pipeline_stats.json"
    if stats_path.exists():
        res["pipeline"] = json.loads(stats_path.read_text(encoding="utf-8"))
    res["config"] = {k: cfg.analysis[k] for k in ("start_year", "end_year", "reference_period", "old_period", "recent_period",
                                                  "month_min_completeness", "year_min_completeness")}
    res["config"]["thresholds"] = cfg.thresholds

    # Tables de résultats (CSV) : réutilisables dans Excel / Power BI
    fr.rename("anomalie_c").to_csv(tables_dir / "anomalie_france.csv", index_label="annee")
    cities.round(3).to_csv(tables_dir / "tendances_villes.csv", index=False)
    pd.DataFrame(res["extremes"]["per_city"]).to_csv(tables_dir / "extremes_villes.csv", index=False)

    with open(cfg.reports_dir / "resultats.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False, default=_json_default)
    log.info("%d figures, résultats dans %s", len(list(figs.glob("*.png"))), cfg.reports_dir / "resultats.json")
    return res


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else round(float(o), 4)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (pd.Timestamp,)):
        return o.strftime("%Y-%m-%d")
    return str(o)
