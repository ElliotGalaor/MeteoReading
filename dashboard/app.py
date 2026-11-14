"""Dashboard interactif Météo Analytics.

Lancement (depuis la racine du projet) :
    streamlit run dashboard/app.py
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ui import (
    CFG, COOL, DIVERGING, END, GRID, HEAT, INK, INK2, MUTED, OLD, RAIN, RECENT, REF, SERIES, START,
    SURFACE, TH, WARM, chart, city_colors, doy_climatology, era5, example_queries, fr, page_header,
    pvalue, results, rolling, run_user_sql, sfr, signed_color, source_note, table, warehouse_ready,
)
from meteo.stats import SEASONS, trend

st.set_page_config(page_title="Météo Analytics", page_icon="🌡️", layout="wide")

CITIES = [c.name for c in CFG.cities]
INDICATORS = {  # libellé -> (colonne, unité, disponible par saison)
    "Température moyenne": ("tmean", "°C", True),
    "Température maximale": ("tx", "°C", True),
    "Température minimale": ("tn", "°C", True),
    "Cumul de pluie": ("rr", "mm", True),
    f"Journées chaudes (TX ≥ {TH['hot_day']} °C)": ("hot_days", "jours", False),
    f"Journées très chaudes (TX ≥ {TH['very_hot_day']} °C)": ("very_hot_days", "jours", False),
    f"Nuits tropicales (TN ≥ {TH['tropical_night']} °C)": ("tropical_nights", "jours", False),
    f"Jours de gel (TN < {TH['frost_day']} °C)": ("frost_days", "jours", False),
    f"Jours en épisode de chaleur (≥ {TH['heatwave_min_days']} j à {TH['hot_day']} °C)": ("heatwave_days", "jours", False),
    f"Jours de forte pluie (≥ {TH['heavy_rain']} mm)": ("heavy_rain_days", "jours", False),
    "Plus longue période sèche": ("longest_dry_spell", "jours", False),
    "Température maximale de l'année": ("tx_max", "°C", False),
    "Température minimale de l'année": ("tn_min", "°C", False),
}
COUNT_INDICATORS = {k: v for k, v in INDICATORS.items() if v[1] == "jours"}
SEASON_LABELS = ["Année entière", "Hiver", "Printemps", "Été", "Automne"]
THE_SEASON = {"Hiver": "L'hiver", "Printemps": "Le printemps", "Été": "L'été", "Automne": "L'automne"}
MONTHS = ["Jan", "Fév", "Mar", "Avr", "Mai", "Juin", "Juil", "Août", "Sep", "Oct", "Nov", "Déc"]


# =============================================================================
def page_overview():
    r = results()
    w, ex, pl = r["warming"], r["extremes"], r.get("pipeline", {})
    tr = w["france_trend_1951"]
    page_header(
        f"Le climat des villes françaises, {START}-{END}",
        f"{fr(pl.get('n_rows', 0), 0)} relevés quotidiens Météo-France, {fr(pl.get('n_stations', 0), 0)} stations, "
        f"20 grandes villes. Toutes les valeurs sont des écarts à la normale {REF[0]}-{REF[1]} sauf mention contraire.",
    )

    hot = ex["france_old_vs_recent"]["hot_days"]
    k = st.columns(4)
    k[0].metric("Réchauffement par décennie", f"{sfr(tr['slope'], 2)} °C", border=True,
                help=f"Pente de la régression linéaire {START}-{END} sur la moyenne des 20 villes. "
                     f"Intervalle de confiance à 95 % : {fr(tr['ci_low'], 2)} à {fr(tr['ci_high'], 2)} °C. "
                     f"Test de Mann-Kendall : {pvalue(tr['mk_p_value'])}.")
    k[1].metric(f"{RECENT[0]}-{str(RECENT[1])[2:]} vs {OLD[0]}-{str(OLD[1])[2:]}", f"{sfr(w['delta_recent_vs_old'], 2)} °C", border=True,
                help="Différence de température moyenne entre les deux périodes (moyenne des 20 villes).")
    k[2].metric(f"Journées ≥ {TH['hot_day']} °C", f"× {fr(hot['ratio'], 1)}", border=True,
                help=f"{fr(hot['old'], 1)} jours par an et par ville en {OLD[0]}-{OLD[1]}, "
                     f"{fr(hot['recent'], 1)} en {RECENT[0]}-{RECENT[1]}.")
    k[3].metric("Records chauds / froids", f"{fr(r['records']['last_ratio'], 0)} : 1",
                border=True, help="Records quotidiens battus depuis 2020 : TX la plus haute contre TN la plus basse "
                                  "jamais mesurée à cette date. Dans un climat stable, le rapport serait proche de 1.")

    fr_anom = pd.read_csv(CFG.reports_dir / "tables" / "anomalie_france.csv")
    years, anom = fr_anom["annee"], fr_anom["anomalie_c"]
    fig = go.Figure(go.Bar(
        x=years, y=[1] * len(years), marker=dict(color=anom, colorscale=DIVERGING, cmid=0, line_width=0),
        customdata=anom, hovertemplate="<b>%{x}</b><br>%{customdata:+.2f} °C par rapport à la normale<extra></extra>"))
    fig.update_layout(bargap=0, yaxis=dict(visible=False), xaxis=dict(showline=False, showgrid=False, dtick=10))
    chart(fig, 170, "Chaque bande est une année", f"Écart de température annuelle à la normale {REF[0]}-{REF[1]}, "
          "moyenne des 20 villes (survolez une bande)", margin=dict(l=0, r=0, b=28))

    left, right = st.columns([3, 2], gap="large")
    with left:
        fig = go.Figure()
        fig.add_bar(x=years, y=anom, marker_color=signed_color(anom), name="Écart annuel", showlegend=False,
                    hovertemplate="%{y:+.2f} °C<extra>%{x}</extra>")
        fig.add_scatter(x=years, y=rolling(anom), mode="lines", line=dict(color=INK, width=2.5),
                        name="Moyenne glissante 10 ans", hovertemplate="%{y:+.2f} °C<extra>moy. 10 ans</extra>")
        fig.update_layout(yaxis_title="°C", hovermode="x unified")
        chart(fig, 420, f"{sfr(tr['slope'], 2)} °C par décennie depuis {START}",
              "Rouge : année plus chaude que la normale, bleu : plus froide · "
              + ("les 10 années les plus chaudes sont toutes postérieures à 2010" if w["n_top10_since_2011"] == 10
                 else f"{w['n_top10_since_2011']} des 10 années les plus chaudes sont postérieures à 2010"))
    with right:
        cities = pd.DataFrame(r["heterogeneity"]["cities"])
        ref = table("ref_station").query("priority == 1")[["city", "lat", "lon", "station_name"]]
        m = cities.merge(ref, on="city")
        fig = go.Figure(go.Scattermap(
            lat=m["lat"], lon=m["lon"], mode="markers",
            marker=dict(size=17, color=m["slope"], colorscale=HEAT, cmin=0, opacity=0.95,
                        colorbar=dict(title=dict(text="°C/déc.", side="top"), thickness=12, len=0.6)),
            customdata=np.stack([m["city"], m["slope"], m["delta"], m["station_name"]], axis=1),
            hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]:+.2f} °C par décennie"
                          "<br>%{customdata[2]:+.2f} °C entre 1961-1990 et 2011-2025"
                          "<br><span style='color:#52514e'>station : %{customdata[3]}</span><extra></extra>"))
        fig.update_layout(map=dict(style="carto-positron", center=dict(lat=46.1, lon=2.9), zoom=4.0))
        chart(fig, 420, "Vitesse de réchauffement par ville", "Survolez un point pour le détail",
              margin=dict(l=0, r=0, b=0))

    st.subheader("Ce qu'il faut retenir")
    pr, h = r["precipitation"], r["heterogeneity"]
    findings = [
        f"<b>Réchauffement net et significatif</b> : {sfr(tr['slope'], 2)} °C par décennie depuis {START}, "
        f"et {sfr(w['france_trend_1981']['slope'], 2)} °C depuis 1981 : il accélère.",
        f"<b>{THE_SEASON[h['fastest_season']]}</b> est la saison qui se réchauffe le plus vite "
        f"({sfr(h['season_trends'][h['fastest_season']]['slope'], 2)} °C par décennie). "
        f"{h['fastest_city']} est la ville la plus touchée, {h['slowest_city']} la moins.",
        f"<b>Les extrêmes chauds explosent</b> : journées ≥ {TH['very_hot_day']} °C × "
        f"{fr(ex['france_old_vs_recent']['very_hot_days']['ratio'], 1)}, nuits tropicales × "
        f"{fr(ex['france_old_vs_recent']['tropical_nights']['ratio'], 1)}, jours de gel "
        f"{sfr((ex['france_old_vs_recent']['frost_days']['ratio'] - 1) * 100, 0)} %.",
        f"<b>Pluie : pas de tendance significative</b> sur le cumul annuel "
        f"({sfr(pr['annual_trend_pct']['slope'], 1)} % par décennie, Mann-Kendall {pvalue(pr['annual_trend_pct']['mk_p_value'])}).",
        f"<b>Record absolu</b> : {fr(ex['hottest_day']['tx'], 1)} °C à {ex['hottest_day']['city']} le "
        f"{pd.Timestamp(ex['hottest_day']['date']).strftime('%d/%m/%Y')}.",
    ]
    for f_ in findings:
        st.markdown(f"<div class='finding'>{f_}</div>", unsafe_allow_html=True)
    source_note()


# =============================================================================
def _city_series(col: str, season: str) -> pd.DataFrame:
    """Série annuelle (ou saisonnière) par ville pour un indicateur."""
    valid = "valid_rain" if col == "rr" else "valid_temp"
    if season == "Année entière":
        y = table("city_yearly")
        return y[y[valid] & y["year"].between(START, END)][["city", "year", col]]
    m = table("city_monthly")
    m = m[m[valid]].copy()
    m["season"] = m["month"].map(SEASONS)
    m["year"] = m["year"] + (m["month"] == 12).astype(int)
    g = (m[m["season"] == season].groupby(["city", "year"])[col]
         .agg(["sum" if col == "rr" else "mean", "count"]).reset_index())
    g = g[(g["count"] == 3) & g["year"].between(START, END)]
    return g.rename(columns={g.columns[2]: col})[["city", "year", col]]


def page_city():
    page_header("Explorer une ville", "Comparez jusqu'à trois villes sur l'indicateur de votre choix.")
    f = st.columns([2.2, 2, 1.3, 1.6])
    sel = f[0].multiselect("Villes", CITIES, default=["Paris", "Marseille"], max_selections=3,
                           help="Trois villes au maximum, pour que chaque couleur reste bien distincte.")
    ind = f[1].selectbox("Indicateur", list(INDICATORS))
    col, unit, seasonal_ok = INDICATORS[ind]
    season = f[2].selectbox("Saison", SEASON_LABELS if seasonal_ok else ["Année entière"], disabled=not seasonal_ok,
                            help="Hiver = décembre de l'année précédente, janvier et février.")
    mode = f[3].segmented_control("Affichage", ["Valeurs", "Écart à la normale"], default="Valeurs")
    period = st.slider("Période", START, END, (START, END))
    if not sel:
        st.info("Choisissez au moins une ville.")
        return

    colors = city_colors(sel, "city")
    df = _city_series(col, season)
    df = df[df["city"].isin(sel)].copy()
    relative = col == "rr"
    anomaly = mode == "Écart à la normale"
    if anomaly:
        normals = df[df["year"].between(*REF)].groupby("city")[col].mean()
        base = df["city"].map(normals)
        df[col] = (df[col] / base - 1) * 100 if relative else df[col] - base
    df = df[df["year"].between(*period)]
    y_unit = "%" if (anomaly and relative) else unit

    fig = go.Figure()
    for city in sel:
        d = df[df["city"] == city].sort_values("year")
        c = colors[city]
        fig.add_scatter(x=d["year"], y=d[col], mode="lines", line=dict(color=c, width=1.2), opacity=0.45,
                        legendgroup=city, showlegend=False, name=city,
                        hovertemplate=f"{city} : %{{y:.1f}} {y_unit}<extra></extra>")
        fig.add_scatter(x=d["year"], y=rolling(d[col].reset_index(drop=True)).to_numpy(), mode="lines",
                        line=dict(color=c, width=2.8), legendgroup=city, name=city,
                        hovertemplate=f"{city} (moy. 10 ans) : %{{y:.1f}} {y_unit}<extra></extra>")
    if anomaly:
        fig.add_hline(y=0, line=dict(color=MUTED, width=1))
    fig.update_layout(hovermode="x unified", yaxis_title=y_unit)
    label = ind if season == "Année entière" else f"{ind} · {season.lower()}"
    chart(fig, 460, label + (" : écart à la normale" if anomaly else ""),
          f"Trait fin : valeur annuelle · trait épais : moyenne glissante sur 10 ans · {period[0]}-{period[1]}")

    cols = st.columns(len(sel))
    for c_, city in zip(cols, sel):
        d = df[df["city"] == city]
        try:
            t = trend(d["year"], d[col])
            verdict = "significative" if t.significant else "non significative"
            c_.metric(f"Tendance à {city}", f"{sfr(t.slope, 2)} {y_unit} / décennie", border=True,
                      help=f"Régression linéaire, IC 95 % [{fr(t.ci_low, 2)} ; {fr(t.ci_high, 2)}]. "
                           f"Pente de Theil-Sen : {sfr(t.sen_slope, 2)}.")
            c_.markdown(f"<div class='kpi-note'>Mann-Kendall {pvalue(t.mk_p_value)} · tendance {verdict}</div>",
                        unsafe_allow_html=True)
        except ValueError:
            c_.metric(f"Tendance à {city}", "période trop courte", border=True)

    st.subheader("Saisonnalité : avant et maintenant")
    mcol = col if col in ("tmean", "tx", "tn", "rr") else "tmean"
    m = table("city_monthly")
    m = m[m["city"].isin(sel) & m["valid_rain" if mcol == "rr" else "valid_temp"]]
    fig = go.Figure()
    for city in sel:
        for per, width, opacity in ((OLD, 1.6, 0.5), (REF, 2.8, 1.0)):
            d = m[(m["city"] == city) & m["year"].between(*per)].groupby("month")[mcol].mean()
            fig.add_scatter(x=MONTHS, y=d.values, mode="lines+markers", name=f"{city} {per[0]}-{per[1]}",
                            line=dict(color=colors[city], width=width), opacity=opacity, marker=dict(size=8),
                            hovertemplate=f"{city} {per[0]}-{per[1]} : %{{y:.1f}} {'mm' if mcol == 'rr' else '°C'}<extra></extra>")
    fig.update_layout(hovermode="x unified", yaxis_title="mm" if mcol == "rr" else "°C")
    name = {"tmean": "Température moyenne", "tx": "Température maximale", "tn": "Température minimale", "rr": "Pluie"}[mcol]
    chart(fig, 400, f"{name} mois par mois", f"Normale {OLD[0]}-{OLD[1]} (trait clair) et {REF[0]}-{REF[1]} (trait foncé)")

    with st.expander("Stations de mesure utilisées"):
        rs = table("ref_station")
        rs = rs[rs["city"].isin(sel)].copy()
        rs["rôle"] = np.where(rs["priority"] == 1, "principale", "complément (série raccordée)")
        st.dataframe(rs[["city", "rôle", "station_id", "station_name", "altitude", "distance_km",
                         "first_date", "last_date", "completeness_temp"]],
                     hide_index=True, column_config={
                         "city": "Ville", "station_id": "N° Météo-France", "station_name": "Station",
                         "altitude": st.column_config.NumberColumn("Altitude", format="%d m"),
                         "distance_km": st.column_config.NumberColumn("Distance", format="%.1f km"),
                         "first_date": st.column_config.DateColumn("Début", format="DD/MM/YYYY"),
                         "last_date": st.column_config.DateColumn("Fin", format="DD/MM/YYYY"),
                         "completeness_temp": st.column_config.ProgressColumn(
                             f"Complétude {START}-{END}", min_value=0, max_value=1, format="percent")})
    source_note()


# =============================================================================
def page_year():
    page_header("Une année en détail", "Chaque jour de l'année comparé à la normale de la ville.")
    d_all = table("city_daily")
    last_year = int(d_all.dropna(subset=["tx"])["date"].dt.year.max())
    f = st.columns([2, 1.4, 2.4])
    city = f[0].selectbox("Ville", CITIES, index=CITIES.index("Paris"))
    years = list(range(last_year, START - 1, -1))
    year = f[1].selectbox("Année", years, index=years.index(2022) if 2022 in years else 0,
                          format_func=lambda y: f"{y} (en cours)" if y > END else str(y))
    var_label = f[2].segmented_control("Variable", ["Maximale", "Minimale", "Moyenne"], default="Maximale")
    col = {"Maximale": "tx", "Minimale": "tn", "Moyenne": "tmean"}[var_label or "Maximale"]

    d = d_all[d_all["city"] == city]
    clim = doy_clim(city, col)
    yr = d[d["date"].dt.year == year][["date", col]].dropna().copy()
    yr = yr[~((yr["date"].dt.month == 2) & (yr["date"].dt.day == 29))]
    yr["doy"] = yr["date"].dt.day_of_year - ((yr["date"].dt.is_leap_year) & (yr["date"].dt.month > 2)).astype(int)
    yr = yr.merge(clim, on="doy")
    yr["ecart"] = yr[col] - yr["mean"]
    ref_dates = pd.Timestamp(year=year, month=1, day=1) + pd.to_timedelta(clim["doy"] - 1, unit="D")

    k = st.columns(4)
    k[0].metric("Écart à la normale", f"{sfr(yr['ecart'].mean(), 1)} °C", border=True,
                help="Écart moyen sur l'année entre la valeur du jour et la normale du jour.")
    k[1].metric("Jours > normale", f"{(yr['ecart'] > 0).sum()} j", border=True,
                help=f"Sur {len(yr)} jours mesurés.")
    above, below = (yr[col] > yr["p90"]).sum(), (yr[col] < yr["p10"]).sum()
    k[2].metric("Jours > 90e centile", f"{above}", border=True,
                help="Au-dessus du 90e centile de la normale : environ 10 % des jours (≈ 36 par an) dans un climat stable.")
    i_max = yr[col].idxmax()
    k[3].metric("Maximum", f"{fr(yr.loc[i_max, col], 1)} °C", border=True,
                help=f"Le {yr.loc[i_max, 'date'].strftime('%d/%m/%Y')}")

    fig = go.Figure()
    fig.add_scatter(x=ref_dates, y=clim["p90"], mode="lines", line=dict(width=0), hoverinfo="skip", showlegend=False)
    fig.add_scatter(x=ref_dates, y=clim["p10"], mode="lines", line=dict(width=0), fill="tonexty",
                    fillcolor="rgba(195,194,183,0.35)", name=f"80 % des jours en {REF[0]}-{REF[1]}", hoverinfo="skip")
    fig.add_scatter(x=ref_dates, y=clim["mean"], mode="lines", line=dict(color=MUTED, width=1.6),
                    name=f"Normale {REF[0]}-{REF[1]}", hoverinfo="skip")
    fig.add_scatter(x=yr["date"], y=yr[col], mode="lines", line=dict(color=INK, width=1.6), name=str(year),
                    customdata=np.stack([yr["mean"], yr["ecart"]], axis=1),
                    hovertemplate="<b>%{x|%d/%m}</b> : %{y:.1f} °C<br>normale %{customdata[0]:.1f} °C · "
                                  "écart %{customdata[1]:+.1f} °C<extra></extra>")
    hot, cold = yr[yr[col] > yr["p90"]], yr[yr[col] < yr["p10"]]
    fig.add_scatter(x=hot["date"], y=hot[col], mode="markers", name="Au-dessus du 90e centile",
                    marker=dict(color=WARM, size=8, line=dict(color=SURFACE, width=1.5)), hoverinfo="skip")
    fig.add_scatter(x=cold["date"], y=cold[col], mode="markers", name="Sous le 10e centile",
                    marker=dict(color=COOL, size=8, line=dict(color=SURFACE, width=1.5)), hoverinfo="skip")
    month_starts = [pd.Timestamp(year=year, month=m, day=1) for m in range(1, 13)]
    fig.update_layout(yaxis_title="°C", xaxis=dict(tickvals=month_starts, ticktext=MONTHS), hovermode="closest")
    chart(fig, 480, f"Température {var_label.lower()} à {city} en {year}",
          f"Comparée à la normale {REF[0]}-{REF[1]} lissée sur ±15 jours")

    yr["month"] = yr["date"].dt.month
    mon = yr.groupby("month")["ecart"].mean()
    fig = go.Figure(go.Bar(x=[MONTHS[i - 1] for i in mon.index], y=mon.values, marker_color=signed_color(mon.values),
                           hovertemplate="%{x} : %{y:+.1f} °C<extra></extra>"))
    fig.add_hline(y=0, line=dict(color=MUTED, width=1))
    fig.update_layout(yaxis_title="°C", showlegend=False)
    chart(fig, 320, "Écart à la normale, mois par mois")
    source_note()


@st.cache_data(show_spinner=False)
def doy_clim(city: str, col: str) -> pd.DataFrame:
    d = table("city_daily")
    return doy_climatology(d[d["city"] == city], col, REF)


# =============================================================================
def page_extremes():
    page_header("Extrêmes et records", "Comment la fréquence des journées extrêmes a évolué, ville par ville.")
    r = results()
    ind = st.selectbox("Indicateur", list(COUNT_INDICATORS))
    col = COUNT_INDICATORS[ind][0]
    scale = RAIN if col in ("frost_days", "heavy_rain_days") else HEAT

    y = table("city_yearly")
    y = y[y["valid_temp"] & y["year"].between(START, END)].copy()
    y["decade"] = y["year"] // 10 * 10
    pv = y.pivot_table(index="city", columns="decade", values=col, aggfunc="mean")
    pv.columns = [f"{max(c, START)}-{str(min(c + 9, END))[2:]}" for c in pv.columns]
    pv = pv.loc[pv.iloc[:, -1].sort_values().index]
    fig = go.Figure(go.Heatmap(
        z=pv.values, x=pv.columns, y=pv.index, colorscale=scale, xgap=2, ygap=2,
        text=np.vectorize(lambda v: "" if np.isnan(v) else fr(v, 0))(pv.values), texttemplate="%{text}", textfont=dict(size=11),
        colorbar=dict(title=dict(text="jours/an", side="top"), thickness=12),
        hovertemplate="<b>%{y}</b>, %{x} : %{z:.1f} jours par an<extra></extra>"))
    fig.update_layout(xaxis=dict(side="top", showline=False, type="category", showgrid=False),
                      yaxis=dict(showline=False, type="category", showgrid=False))
    chart(fig, 640, ind, margin=dict(l=120, t=40), subtitle="Nombre moyen de jours par an, par décennie (villes triées selon la dernière décennie)")

    old = y[y["year"].between(*OLD)].groupby("city")[col].mean()
    new = y[y["year"].between(*RECENT)].groupby("city")[col].mean()
    dd = pd.DataFrame({"old": old, "new": new}).sort_values("new")
    fig = go.Figure()
    xs, ys = [], []
    for c, row in dd.iterrows():
        xs += [row["old"], row["new"], None]
        ys += [c, c, None]
    fig.add_scatter(x=xs, y=ys, mode="lines", line=dict(color=GRID, width=3), hoverinfo="skip", showlegend=False)
    fig.add_scatter(x=dd["old"], y=dd.index, mode="markers", name=f"{OLD[0]}-{OLD[1]}",
                    marker=dict(color=SERIES[0], size=11, line=dict(color=SURFACE, width=2)),
                    hovertemplate="%{y} : %{x:.1f} jours<extra>" + f"{OLD[0]}-{OLD[1]}</extra>")
    fig.add_scatter(x=dd["new"], y=dd.index, mode="markers", name=f"{RECENT[0]}-{RECENT[1]}",
                    marker=dict(color=SERIES[1], size=11, line=dict(color=SURFACE, width=2)),
                    hovertemplate="%{y} : %{x:.1f} jours<extra>" + f"{RECENT[0]}-{RECENT[1]}</extra>")
    fig.update_layout(xaxis_title="jours par an", yaxis=dict(showgrid=False))
    chart(fig, 600, "Avant / maintenant", f"Moyenne {OLD[0]}-{OLD[1]} contre {RECENT[0]}-{RECENT[1]}")

    left, right = st.columns([1, 1], gap="large")
    with left:
        rec = pd.DataFrame(r["records"]["by_decade"])
        fig = go.Figure(go.Bar(
            x=rec["decade"].astype(str), y=rec["ratio"], marker_color=[WARM if v >= 1 else COOL for v in rec["ratio"]],
            customdata=np.stack([rec["hot"], rec["cold"]], axis=1),
            text=[f"{fr(v, 1)} : 1" for v in rec["ratio"]], textposition="outside",
            hovertemplate="<b>%{x}</b><br>%{customdata[0]} records chauds<br>%{customdata[1]} records froids<extra></extra>"))
        fig.add_hline(y=1, line=dict(color=MUTED, width=1))
        top = max(32, rec["ratio"].max() * 2)
        ticks = [v for v in (0.5, 1, 2, 4, 8, 16, 32, 64) if v <= top]
        fig.update_layout(yaxis=dict(type="log", title="records chauds / froids", range=[np.log10(0.7), np.log10(top)],
                                     tickvals=ticks, ticktext=[f"{fr(v, 0) if v >= 1 else '1 : 2'}{' : 1' if v >= 1 else ''}" for v in ticks]),
                          xaxis=dict(type="category"), showlegend=False)
        chart(fig, 420, "Records de chaleur contre records de froid", "Records quotidiens battus, 20 villes (échelle logarithmique)")
    with right:
        st.markdown(f"**Épisodes de chaleur** (au moins {TH['heatwave_min_days']} jours consécutifs à {TH['hot_day']} °C ou plus)")
        hw = table("city_heatwaves")
        hw = hw[hw["start_date"].dt.year <= END]
        c1, c2 = st.columns([2, 1])
        cities = c1.multiselect("Villes", CITIES, default=[], placeholder="Toutes les villes", key="hw_cities")
        min_days = c2.number_input("Durée minimale", 3, 60, 7)
        hw = hw[hw["n_days"] >= min_days]
        if cities:
            hw = hw[hw["city"].isin(cities)]
        st.dataframe(hw.sort_values(["n_days", "tx_peak"], ascending=False).head(200), hide_index=True, height=330,
                     column_order=["city", "start_date", "end_date", "n_days", "tx_peak", "tx_mean"],
                     column_config={
                         "city": "Ville",
                         "start_date": st.column_config.DateColumn("Début", format="DD/MM/YYYY"),
                         "end_date": st.column_config.DateColumn("Fin", format="DD/MM/YYYY"),
                         "n_days": st.column_config.NumberColumn("Durée", format="%d j"),
                         "tx_peak": st.column_config.NumberColumn("Pic", format="%.1f °C"),
                         "tx_mean": st.column_config.NumberColumn("TX moyenne", format="%.1f °C")})
        st.caption(f"{len(hw)} épisodes correspondent aux filtres.")
    source_note()


# =============================================================================
def page_quality():
    page_header("Données, méthode et qualité",
                "D'où viennent les chiffres, comment ils sont nettoyés, et jusqu'où on peut leur faire confiance.")
    r = results()
    pl = r.get("pipeline", {})
    build_s = sum(v for k, v in pl.items() if k.startswith("duration_"))
    st.graphviz_chart(f"""
    digraph {{
      rankdir=LR; bgcolor="transparent"; nodesep=0.35; ranksep=0.45;
      node [shape=box style="rounded,filled" fillcolor="#ffffff" color="{GRID}" fontname="Segoe UI" fontsize=11 fontcolor="{INK}" margin="0.18,0.1"];
      edge [color="{MUTED}" arrowsize=0.7];
      api  [label="API data.gouv.fr\\ndécouverte des fichiers"];
      raw  [label="{pl.get('n_files', 0)} fichiers CSV.gz\\n{fr(pl.get('raw_bytes', 0) / 1e6, 0)} Mo · bronze"];
      stg  [label="Staging DuckDB\\n{fr(pl.get('n_rows', 0) / 1e6, 1)} M lignes typées"];
      dq   [label="Contrôle qualité\\ncodes, bornes, cohérence"];
      st   [label="Stations de référence\\n+ raccordement des séries"];
      gold [label="Tables gold\\njour · mois · année"];
      out  [label="Analyses · rapport\\ndashboard · SQL" fillcolor="{SERIES[0]}" fontcolor="#ffffff" color="{SERIES[0]}"];
      api -> raw -> stg -> dq -> st -> gold -> out;
    }}""", width="stretch")
    st.caption(f"Pipeline complet : téléchargement parallèle (~30 s), construction de l'entrepôt en {fr(build_s, 1)} s, "
               f"analyses en ~6 s. Commande unique : `python -m meteo all`.")

    k = st.columns(4)
    k[0].metric("Lignes brutes", fr(pl.get("n_rows", 0), 0), border=True)
    k[1].metric("Stations", fr(pl.get("n_stations", 0), 0), border=True)
    k[2].metric("Valeurs rejetées", fr(pl.get("n_measures", 0) - pl.get("n_valid_measures", 0), 0), border=True,
                help=f"Sur {fr(pl.get('n_measures', 0) / 1e6, 1)} millions de mesures présentes : "
                     "codes « douteux », valeurs impossibles ou TN > TX.")
    k[3].metric("Doublons station-jour", fr(pl.get("n_duplicates", 0), 0), border=True)

    st.subheader("Bilan qualité par variable")
    dq = pd.DataFrame(r["quality"]["by_variable"])
    labels = {"TN": "Température min.", "TX": "Température max.", "TM": "Température moy. horaire",
              "RR": "Pluie", "FFM": "Vent moyen", "FXI": "Rafale max."}
    dq["variable"] = dq["variable"].map(labels)
    for c in ("douteuse", "hors_bornes", "incoherente"):
        if c not in dq:
            dq[c] = 0
    st.dataframe(dq[["variable", "valide", "absente", "douteuse", "hors_bornes", "incoherente", "pct_absentes"]],
                 hide_index=True, column_config={
                     "variable": "Variable",
                     "valide": st.column_config.NumberColumn("Valides", format="localized"),
                     "absente": st.column_config.NumberColumn("Absentes", format="localized"),
                     "douteuse": st.column_config.NumberColumn("Douteuses (code 2)", format="localized"),
                     "hors_bornes": st.column_config.NumberColumn("Hors bornes", format="localized"),
                     "incoherente": st.column_config.NumberColumn("TN > TX", format="localized"),
                     "pct_absentes": st.column_config.ProgressColumn("% absentes", min_value=0, max_value=100, format="%.0f %%")})
    st.markdown(
        "Météo-France valide déjà ses données : très peu de valeurs sont rejetées. **Le vrai enjeu est la complétude** : "
        "la majorité des postes ne mesurent que la pluie, et le réseau évolue (ouvertures, fermetures, déménagements). "
        "D'où la sélection de stations longues et complètes, et le raccordement de séries quand une station a déménagé.")

    nw = table("network_yearly")
    nw = nw[nw["year"] <= END]
    fig = go.Figure()
    fig.add_scatter(x=nw["year"], y=nw["n_stations_rain"], name="avec mesure de pluie", line=dict(color=SERIES[0], width=2.2),
                    hovertemplate="%{y} stations<extra>pluie</extra>")
    fig.add_scatter(x=nw["year"], y=nw["n_stations_temp"], name="avec mesure de température", line=dict(color=SERIES[1], width=2.2),
                    hovertemplate="%{y} stations<extra>température</extra>")
    fig.update_layout(hovermode="x unified", yaxis=dict(rangemode="tozero", title="stations actives"))
    chart(fig, 380, "Le réseau de mesure change au fil du temps", "Stations actives par an dans les 20 départements téléchargés")

    st.subheader("Stations de référence et raccordements")
    off = pd.DataFrame(r["quality"]["offsets"])
    rs = table("ref_station")
    rs = rs.merge(off[["station_id", "offset_tn", "offset_tx", "n_overlap_days"]], on="station_id", how="left")
    for c in ("offset_tn", "offset_tx"):  # texte : une case vide plutôt que « None »
        rs[c] = [f"{sfr(v, 2)} °C" if pd.notna(v) else "" for v in rs[c]]
    rs["n_overlap_days"] = [fr(v, 0) if pd.notna(v) else "" for v in rs["n_overlap_days"]]
    st.dataframe(rs[["city", "priority", "station_name", "altitude", "distance_km", "first_date", "last_date",
                     "completeness_temp", "offset_tn", "offset_tx", "n_overlap_days"]],
                 hide_index=True, height=420, column_config={
                     "city": "Ville", "priority": st.column_config.NumberColumn("Rang", help="1 = station principale"),
                     "station_name": "Station",
                     "altitude": st.column_config.NumberColumn("Alt.", format="%d m"),
                     "distance_km": st.column_config.NumberColumn("Distance", format="%.1f km"),
                     "first_date": st.column_config.DateColumn("Début", format="YYYY"),
                     "last_date": st.column_config.DateColumn("Fin", format="YYYY"),
                     "completeness_temp": st.column_config.ProgressColumn(f"Complétude {START}-{END}", min_value=0, max_value=1, format="percent"),
                     "offset_tn": st.column_config.TextColumn("Correction TN", help="Biais moyen corrigé, calculé mois par mois"),
                     "offset_tx": st.column_config.TextColumn("Correction TX"),
                     "n_overlap_days": st.column_config.TextColumn("Jours communs", help="Jours mesurés par les deux stations")})

    e = era5()
    if e is not None:
        st.subheader("Validation croisée avec la réanalyse ERA5")
        e = e.sort_values("bias_tn")
        fig = go.Figure()
        fig.add_bar(y=e["city"], x=e["bias_tx"], orientation="h", name="T° maximale", marker_color=SERIES[1],
                    hovertemplate="%{y} : %{x:+.2f} °C<extra>TX</extra>")
        fig.add_bar(y=e["city"], x=e["bias_tn"], orientation="h", name="T° minimale", marker_color=SERIES[0],
                    hovertemplate="%{y} : %{x:+.2f} °C<extra>TN</extra>")
        fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.08, xaxis_title="écart moyen ERA5 − station (°C)",
                          yaxis=dict(showgrid=False))
        chart(fig, 620, r.get("era5", {}).get("title", "ERA5 contre stations"),
              f"{CFG.era5['start'][:4]}-{CFG.era5['end'][:4]} · ERA5 (grille ~25 km) via l'API Open-Meteo")
        x = r.get("era5", {})
        e["hot_ratio"] = e["hot_days_era5"] / e["hot_days_obs"].clip(lower=1)
        miss = e[e["hot_days_obs"] >= 50].sort_values("hot_ratio").iloc[0]
        st.markdown(
            f"Corrélation moyenne des TX quotidiennes : **{fr(x.get('mean_corr_tx', 0), 3)}** : la réanalyse suit très bien "
            f"la météo du jour. Mais une maille de 25 km mélange mer, ville et campagne : **{x.get('worst_city')}** a un biais "
            f"de {sfr(x.get('worst_bias_tn', 0), 1)} °C sur les minimales, et à **{miss['city']}** ERA5 ne compte que "
            f"{fr(int(miss['hot_days_era5']), 0)} journées ≥ {TH['hot_day']} °C contre {fr(int(miss['hot_days_obs']), 0)} observées. "
            "Pour étudier les extrêmes locaux, les stations restent indispensables.")

    with st.expander("Limites de l'analyse"):
        st.markdown(f"""
- **Séries non homogénéisées** : changements d'instruments, urbanisation autour des stations ou déménagements
  peuvent créer des ruptures. Météo-France publie des séries homogénéisées pour la recherche ; ici, seuls les
  déménagements identifiés sont corrigés (raccordement par biais mensuel).
- **Une station par ville** : elle représente la ville, pas tout son territoire (îlot de chaleur, relief).
- **Tendance linéaire** : utile pour résumer, mais le réchauffement accélère depuis les années 1980.
- **Autocorrélation** : les tests supposent des années indépendantes ; la p-value est donc un peu optimiste.
- **Année en cours** : {END + 1} est incomplète et exclue des statistiques annuelles.
""")
    source_note(" ; ERA5 : Copernicus Climate Change Service, via Open-Meteo")


# =============================================================================
TABLE_DOCS = {
    "city_daily": "1 ligne par ville et par jour (TN, TX, Tmoy, pluie, vent)",
    "city_monthly": "agrégats mensuels par ville + validité",
    "city_yearly": "agrégats annuels + indicateurs d'extrêmes",
    "city_heatwaves": "épisodes de chaleur (jours consécutifs ≥ 30 °C)",
    "city_dry_spells": "séquences de jours secs",
    "ref_station": "stations de référence de chaque ville",
    "dim_station": "toutes les stations : métadonnées et complétude",
    "obs_daily": f"vue : les {fr(results().get('pipeline', {}).get('n_rows', 0), 0)} relevés validés, toutes stations",
    "dq_summary": "bilan qualité par variable et statut",
}


def _load_example():
    st.session_state["sql_text"] = example_queries()[st.session_state["sql_example"]]


def page_sql():
    page_header("Requêtes SQL", "Interrogez directement l'entrepôt DuckDB (lecture seule).")
    ex = example_queries()
    if "sql_text" not in st.session_state:
        st.session_state["sql_text"] = next(iter(ex.values()))
    left, right = st.columns([3, 1.3], gap="large")
    with left:
        st.selectbox("Exemple", list(ex), key="sql_example", on_change=_load_example)
        sql = st.text_area("Requête", key="sql_text", height=260)
        run = st.button("Exécuter", type="primary", icon=":material/play_arrow:")
    with right:
        st.markdown("**Tables disponibles**")
        for name, doc in TABLE_DOCS.items():
            st.markdown(f"<div style='margin-bottom:6px'><code>{name}</code><br>"
                        f"<span style='color:{INK2};font-size:0.85rem'>{doc}</span></div>", unsafe_allow_html=True)
    if run or "sql_result" not in st.session_state:
        t0 = time.perf_counter()
        try:
            st.session_state["sql_result"] = (run_user_sql(sql), time.perf_counter() - t0, None)
        except Exception as exc:  # erreur SQL : on l'affiche telle quelle
            st.session_state["sql_result"] = (None, 0, str(exc))
    df, elapsed, err = st.session_state["sql_result"]
    if err:
        st.error(err)
        return
    st.caption(f"{fr(len(df), 0)} ligne(s) en {fr(elapsed * 1000, 0)} ms (affichage limité à 10 000 lignes)")
    st.dataframe(df, hide_index=True)
    st.download_button("Télécharger en CSV", df.to_csv(index=False).encode("utf-8"), "resultat.csv", "text/csv",
                       icon=":material/download:")


# =============================================================================
if not warehouse_ready():
    st.error("Les données ne sont pas encore prêtes. Depuis la racine du projet, lancez :")
    st.code("python -m meteo all", language="bash")
    st.stop()

nav = st.navigation([
    st.Page(page_overview, title="Vue d'ensemble", icon=":material/thermostat:", default=True),
    st.Page(page_city, title="Explorer une ville", icon=":material/location_city:", url_path="ville"),
    st.Page(page_year, title="Une année en détail", icon=":material/calendar_month:", url_path="annee"),
    st.Page(page_extremes, title="Extrêmes et records", icon=":material/local_fire_department:", url_path="extremes"),
    st.Page(page_quality, title="Données et qualité", icon=":material/fact_check:", url_path="qualite"),
    st.Page(page_sql, title="Requêtes SQL", icon=":material/terminal:", url_path="sql"),
])
with st.sidebar:
    st.markdown("### Météo Analytics")
    st.caption(f"Climat de 20 villes françaises, {START}-{END}, à partir des données ouvertes de Météo-France.")
nav.run()
